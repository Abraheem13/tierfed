"""Federated simulation engine.

Self-contained (no Flower dependency) so that tier-wise communication,
compression, DP noise, attack surface and per-round diagnostics are measured
exactly. Differential privacy is applied HERE, to every strategy, with
per-client clipping (v2 applied it inside two strategies only and clipped the
aggregate).
"""
from __future__ import annotations

import time

from typing import Callable, Dict, List

import numpy as np
import torch

from .client import Client
from .metrics import TrajectoryStats, classification_metrics, weighted_average
from .privacy.dp import DPConfig, calibrate_sigma, private_aggregate
from .privacy.inversion import attack_update
from .privacy.membership import membership_inference
from .theory.divergence import DivergenceTracker
from .utils import clone_state, get_logger

log = get_logger()


def _platt(p_val, y_val, p_test):
    """Platt scaling fitted on a site's validation split, applied to its test
    predictions; zero communication."""
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    lv = np.log(np.clip(p_val, eps, 1 - eps) / np.clip(1 - p_val, eps, 1))
    lt = np.log(np.clip(p_test, eps, 1 - eps) / np.clip(1 - p_test, eps, 1))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lv.reshape(-1, 1), y_val)
    return lr.predict_proba(lt.reshape(-1, 1))[:, 1]


def _cal_slope(y, p):
    """Calibration slope: coefficient of logit(p) in a logistic fit of y (1 = ideal)."""
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1)).reshape(-1, 1)
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(z, y).coef_[0, 0])


class FederatedServer:
    def __init__(self, dataset, model_fn: Callable, strategy, device, cfg: Dict,
                 dp: DPConfig | None = None):
        self.dataset, self.model_fn, self.strategy = dataset, model_fn, strategy
        self.device, self.cfg = device, cfg
        self.dp = dp or DPConfig()
        self.global_model = model_fn().to(device)
        # the global state lives on CPU: client slices arrive on CPU, DP noise and the
        # schedulers work on CPU, and clients move the broadcast to their own device
        self.global_state = {k: v.detach().clone().cpu() for k, v in self.global_model.state_dict().items()}
        self.rounds = cfg.get("rounds", 60)
        self.local_epochs = cfg.get("local_epochs", 1)
        self.participation = cfg.get("participation", 1.0)
        self.seed = cfg.get("seed", 42)
        # Personalised evaluation needs per-client held-out data (patient-grouped).
        # With val_split=True the held-out patients are divided into validation
        # (round selection, calibration) and test (reporting).
        self.local_va = None
        if cfg.get("val_split", False):
            self.local_tr, self.local_va, self.local_te = dataset.client_val_test_split(
                frac=cfg.get("client_test_frac", 0.2), val_share=cfg.get("val_share", 0.5), seed=self.seed)
        else:
            self.local_tr, self.local_te = dataset.client_test_split(
                frac=cfg.get("client_test_frac", 0.2), seed=self.seed)
        self.clients = [Client(k, dataset, model_fn, device, cfg, train_idx=self.local_tr[k])
                        for k in range(dataset.n_clients)]
        if hasattr(strategy, "set_client_sizes"):
            strategy.set_client_sizes([c.n for c in self.clients])
        if cfg.get("common_init", False):
            # round-0 broadcast of the (data-independent) initial shared model, download
            # only: needed when the first synchronisation comes late (one-shot
            # federation), since averaging models from different inits is meaningless.
            # Off by default, so earlier logs reproduce; private tensors stay per site.
            priv = set(strategy.personal_keys())
            shared = [k for k in self.global_state if k not in priv]
            for c in self.clients:
                c.set_parameters(self.global_state, shared)
        self.history: List[Dict[str, float]] = []
        self.round_bits: List[Dict[str, float]] = []      # for the network model
        self.upload_bits = 0.0
        self.download_bits = 0.0
        self.dp_rounds = 0
        self.divergence = DivergenceTracker()
        self.attack_rows: List[Dict[str, float]] = []
        self.noise_gen = torch.Generator(device="cpu").manual_seed(self.seed + 7919)
        # Poisson client sampling (each site independently with prob. q) is used
        # whenever DP subsampling amplification is claimed, or when asked for.
        self.poisson = (self.dp.enabled and self.dp.sample_rate < 1.0) or cfg.get("participation_mode") == "poisson"
        self.q = self.dp.sample_rate if self.dp.enabled and self.dp.sample_rate < 1.0 else float(self.participation)
        self.planned_releases = None
        self.dp_safe = False
        if self.dp.enabled:
            # Under client-level DP the transmission schedule is observable. A
            # schedule computed from client updates is a data-dependent release the
            # accountant does not cover, so only data-independent schedules are run
            # (allow_unsafe_dp reproduces the v3 equal-sigma table, disclosed as flawed).
            if not getattr(strategy, "dp_safe", False) and not cfg.get("allow_unsafe_dp", False):
                raise ValueError(f"strategy '{strategy.name}' chooses its schedule from client "
                                 "updates and is not covered by the DP accountant")
            self.dp_safe = bool(getattr(strategy, "dp_safe", False))
            if self.dp_safe:
                self.planned_releases = strategy.planned_release_rounds(self.rounds)
            if self.dp.target_epsilon:
                if self.planned_releases is None:
                    raise ValueError("target_epsilon requires a data-independent schedule")
                self.dp.sigma = calibrate_sigma(float(self.dp.target_epsilon), len(self.planned_releases),
                                                self.dp.sample_rate, self.dp.delta)

    # ------------------------------------------------------------------ #
    def _sample(self, r: int) -> List[int]:
        n = len(self.clients)
        if self.poisson:
            rng = np.random.default_rng(self.seed * 1000 + r)
            return [k for k in range(n) if rng.random() < self.q]
        m = max(1, int(round(self.participation * n)))
        if m >= n:
            return list(range(n))
        rng = np.random.default_rng(self.seed * 1000 + r)
        return sorted(rng.choice(n, size=m, replace=False).tolist())

    def run(self) -> Dict:
        priv = self.strategy.personal_keys()
        attack_every = int(self.cfg.get("attack_every", 0))
        for r in range(1, self.rounds + 1):
            if hasattr(self.strategy, "begin_round"):
                self.strategy.begin_round(r)
            keys = self.strategy.send_keys(r)
            sel = self._sample(r)
            payload = {k: self.global_state[k] for k in keys}
            dl_bits = len(sel) * sum(v.numel() * 32 for v in payload.values())
            self.download_bits += dl_bits

            states, weights, stats, senders = [], [], [], []
            per_client_bits, compute_s = [], []
            for cid in sel:
                c = self.clients[cid]
                c.set_parameters(self.global_state, keys)
                t0 = time.perf_counter()
                s = c.train(self.strategy, payload if keys else None, self.local_epochs, self.seed)
                compute_s.append(time.perf_counter() - t0)
                c.stash_private(priv)
                raw = c.get_slice(keys)
                self.strategy.after_local(cid, self.global_state, raw, keys, s["steps"], s["lr"])
                stats.append(s)
                if not keys or not self.strategy.client_sends(cid, self.global_state, raw, keys, r):
                    per_client_bits.append(0.0)
                    continue
                cs, bits = self.strategy.compress_client_state(raw, keys, reference=self.global_state, cid=cid)
                per_client_bits.append(bits)
                states.append(cs); weights.append(c.n); senders.append(cid)
                if attack_every and r > 1 and cid == sel[0]:
                    # attack the k-th transmitting round of THIS client, k % attack_every == 0,
                    # so strategies that transmit rarely are attacked on the rounds they do transmit
                    self._tx_count = getattr(self, "_tx_count", 0) + 1
                    if self._tx_count % attack_every == 0:
                        self.attack_rows.append(self._attack(r, cs, keys, c))

            ul_bits = float(sum(per_client_bits))
            self.upload_bits += ul_bits
            self.round_bits.append({"round": r, "upload_bits": ul_bits, "download_bits": float(dl_bits),
                                    "max_client_bits": float(max(per_client_bits) if per_client_bits else 0.0),
                                    "n_senders": float(len(senders)), "n_selected": float(len(sel)),
                                    "compute_s_max": float(max(compute_s) if compute_s else 0.0),
                                    "client_bits": [float(b) for b in per_client_bits]})
            if hasattr(self.strategy, "observe_loss") and stats:
                self.strategy.observe_loss(r, float(np.mean([s["loss"] for s in stats])))

            dp_info = {}
            if self.dp.enabled and keys and (states or self.dp_safe):
                # A planned release happens even if Poisson sampling drew nobody
                # (noise-only update), so the accountant's release count is exact.
                denom = self.q * len(self.clients) if self.poisson else None
                self.global_state, dp_info = private_aggregate(
                    self.global_state, states, keys, self.dp.max_norm, self.dp.sigma, self.noise_gen,
                    denominator=denom)
                self.dp_rounds += 1
                dp_info["dp_cohort"] = float(len(states))
                if self.dp_safe:
                    # DP-safe strategies never see raw client slices
                    self.strategy.observe(self.global_state, [], keys, r)
                else:
                    self.strategy.observe(self.global_state, states, keys, r)
            elif self.dp.enabled:
                pass   # nothing transmitted: no release, no budget
            else:
                self.global_state = self.strategy.aggregate(self.global_state, states, weights, r)
                self.strategy.observe(self.global_state, states, keys, r)
            self.divergence.update(r, self.global_state, states, keys)

            if r % self.cfg.get("eval_every", 1) == 0 or r == self.rounds:
                row = self._evaluate(r, stats, keys)
                row.update(dp_info)
                self.history.append(row)
        return self.summarise()

    # ------------------------------------------------------------------ #
    def _attack(self, r, cs, keys, client) -> Dict[str, float]:
        """Gradient-inversion attack on the ACTUAL transmitted slice of this round."""
        model = self.model_fn().to(self.device)
        model.load_state_dict({**{k: v.to(self.device) for k, v in client.model.state_dict().items()},
                               **{k: self.global_state[k].to(self.device) for k in keys}})
        delta = {k: (cs[k].float() - self.global_state[k].float()) for k in keys}
        idx = self.local_tr[client.cid][: self.cfg.get("attack_batch", 8)]
        x_true = torch.as_tensor(self.dataset.x_train[idx], dtype=torch.float32)
        res = attack_update(model, delta, x_true, self.device, steps=self.cfg.get("attack_steps", 300),
                            seed=self.seed + r)
        res.update({"round": float(r), "n_params_exposed": float(sum(v.numel() for v in delta.values())),
                    "n_tensors_exposed": float(len(delta))})
        return res

    def _evaluate(self, r: int, stats, keys) -> Dict[str, float]:
        """Federated evaluation: each client evaluates on its OWN patient-grouped
        held-out split using its own personalised parameters. Clients whose
        held-out split has a single class cannot yield AUROC and are excluded
        identically for every method; their count is logged. A pooled AUROC over
        all held-out predictions is also reported."""
        per_client, weights = [], []
        y_all, p_all, n_single = [], [], 0
        val_metrics, val_w, cal_metrics, cal_w, site_auroc = [], [], [], [], []
        final = (r == self.rounds)
        ft = self.cfg.get("eval_finetune_epochs", 0) if final else 0
        ft_list = sorted({int(j) for j in (ft if isinstance(ft, (list, tuple)) else [ft]) if int(j) > 0})
        ft_metrics, ft_w = {j: [] for j in ft_list}, {j: [] for j in ft_list}
        for cid, c in enumerate(self.clients):
            idx = self.local_te[cid]
            if len(idx) == 0:
                continue
            c.set_parameters(self.global_state, keys)
            x = torch.as_tensor(self.dataset.x_train[idx], dtype=torch.float32)
            y = self.dataset.y_train[idx]
            p = c.predict(x)
            y_all.append(y); p_all.append(p)
            pv = yv = None
            if self.local_va is not None and len(self.local_va[cid]):
                vidx = self.local_va[cid]
                xv = torch.as_tensor(self.dataset.x_train[vidx], dtype=torch.float32)
                yv = self.dataset.y_train[vidx]; pv = c.predict(xv)
                if len(np.unique(yv)) == 2:
                    val_metrics.append(classification_metrics(yv, pv)); val_w.append(len(vidx))
            if ft_list and len(np.unique(y)) == 2:
                for j, mj in self._finetuned_metrics(c, x, y, ft_list).items():
                    ft_metrics[j].append(mj); ft_w[j].append(len(idx))
            if len(np.unique(y)) < 2:
                n_single += 1
                continue
            m = classification_metrics(y, p)
            per_client.append(m); weights.append(len(idx)); site_auroc.append(m["auroc"])
            if pv is not None and yv is not None and len(np.unique(yv)) == 2:
                cm = classification_metrics(y, _platt(pv, yv, p))
                cm["slope"] = _cal_slope(y, p)
                cal_metrics.append(cm); cal_w.append(len(idx))
        fed = weighted_average(per_client, weights)
        pooled = classification_metrics(np.concatenate(y_all), np.concatenate(p_all)) if y_all else {}

        self.global_model.load_state_dict(self.global_state)
        self.global_model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(self.dataset.x_test, dtype=torch.float32)
            pg = np.concatenate([torch.softmax(self.global_model(xt[i:i + 4096].to(self.device)), 1)[:, 1].cpu().numpy()
                                 for i in range(0, len(xt), 4096)])
        glob = classification_metrics(self.dataset.y_test, pg)

        row = {"round": float(r),
               "train_loss": float(np.mean([s["loss"] for s in stats])) if stats else np.nan,
               "grad_sq": float(np.mean([s["grad_sq"] for s in stats])) if stats else np.nan,
               "upload_mb": self.upload_bits / 8e6, "download_mb": self.download_bits / 8e6,
               "payload_fraction": float(sum(self.strategy.sizes[k] for k in keys) /
                                         max(sum(self.strategy.sizes.values()), 1)),
               "n_clients_single_class": float(n_single),
               "n_clients_evaluated": float(len(per_client))}
        row.update({f"fed_{k}": v for k, v in fed.items()})
        if val_metrics:
            row["fed_val_auroc"] = weighted_average(val_metrics, val_w).get("auroc", np.nan)
        if cal_metrics:
            cal = weighted_average(cal_metrics, cal_w)
            row["fed_cal_ece"], row["fed_cal_brier"] = cal.get("ece", np.nan), cal.get("brier", np.nan)
            row["fed_cal_slope"] = cal.get("slope", np.nan)   # slope of the UNcalibrated scores
        if site_auroc:
            sa = np.asarray(site_auroc, float); sa = sa[np.isfinite(sa)]
            if sa.size:
                row["fed_auroc_site_min"] = float(sa.min()); row["fed_auroc_site_p10"] = float(np.percentile(sa, 10))
        for j in ft_list:
            if ft_metrics[j]:
                row[f"fed_ft{j}_auroc"] = weighted_average(ft_metrics[j], ft_w[j]).get("auroc", np.nan)
        row.update({f"pooled_{k}": v for k, v in pooled.items()})
        row.update({f"global_{k}": v for k, v in glob.items()})
        # the tracker's own 'round' is the last synchronisation round, not this round
        row.update({k: v for k, v in self.divergence.latest().items() if k != "round"})
        if hasattr(self.strategy, "K"):
            row.update({f"K_{g}": float(v) for g, v in self.strategy.K.items()})
        return row

    def _finetuned_metrics(self, c, x, y, epochs_list) -> Dict[int, Dict[str, float]]:
        """E-3 control: fine-tune the site's final model for j local epochs on its
        own training split (all parameters, plain local loss), evaluate after each
        j in `epochs_list`, then restore the model and optimiser state. Training is
        over when this runs, so the reported trajectory is untouched; the result
        shows how much accuracy a site obtains by local adaptation alone."""
        import copy
        saved = copy.deepcopy(c.model.state_dict()); saved_opt = copy.deepcopy(c.opt_state)

        class _Plain:
            def local_objective_extra(self, model, gp, loss, cid=None):
                return loss
        out, done = {}, 0
        for j in epochs_list:
            c.train(_Plain(), None, j - done, self.seed + 104729)
            done = j
            out[j] = classification_metrics(y, c.predict(x))
        c.model.load_state_dict(saved); c.opt_state = saved_opt
        return out

    # ------------------------------------------------------------------ #
    def _membership(self) -> Dict[str, float]:
        """Loss-threshold membership inference (Yeom et al. 2018) against the
        personalised models, averaged over clients."""
        aucs = []
        for cid, c in enumerate(self.clients):
            tr, te = self.local_tr[cid], self.local_te[cid]
            if len(te) < 5 or len(tr) < 5:
                continue
            m = min(len(tr), len(te), 2000)
            xi = torch.as_tensor(self.dataset.x_train[tr[:m]], dtype=torch.float32)
            yi = torch.as_tensor(self.dataset.y_train[tr[:m]], dtype=torch.long)
            xo = torch.as_tensor(self.dataset.x_train[te[:m]], dtype=torch.float32)
            yo = torch.as_tensor(self.dataset.y_train[te[:m]], dtype=torch.long)
            c.set_parameters(self.global_state, self.strategy.send_keys(self.rounds))
            aucs.append(membership_inference(c.per_sample_loss(xi, yi), c.per_sample_loss(xo, yo)))
        return {"mia_auc_mean": float(np.mean(aucs)) if aucs else np.nan,
                "mia_auc_max": float(np.max(aucs)) if aucs else np.nan}

    def summarise(self) -> Dict:
        out: Dict[str, float] = {}
        for m in ("fed_auroc", "fed_auprc", "fed_f1", "fed_f1_best", "fed_ece", "fed_brier",
                  "pooled_auroc", "pooled_auprc", "global_auroc", "global_auprc", "global_f1"):
            vals = [h[m] for h in self.history if m in h and not np.isnan(h[m])]
            if vals:
                out.update(TrajectoryStats(m, vals).summary(
                    patience=self.cfg.get("early_stop_patience", 10)))
        # E-1: report test AUROC at the round chosen on VALIDATION, and the bytes spent
        # until then; E-2: mean over post-synchronisation rounds in the last 20.
        if any("fed_val_auroc" in h for h in self.history):
            vals = [(h.get("fed_val_auroc", np.nan), -i) for i, h in enumerate(self.history)]
            vals = [v for v in vals if np.isfinite(v[0])]
            i_best = -max(vals)[1] if vals else len(self.history) - 1   # earliest round on ties
            hb = self.history[i_best]
            out["fed_val_auroc_valsel"] = float(hb.get("fed_val_auroc", np.nan))
            out["fed_auroc_valsel"] = float(hb.get("fed_auroc", np.nan))
            out["fed_auprc_valsel"] = float(hb.get("fed_auprc", np.nan))
            out["valsel_round"] = float(i_best + 1)
            out["upload_mb_valsel"] = float(hb.get("upload_mb", np.nan))
            for k in ("fed_cal_ece", "fed_cal_brier", "fed_cal_slope", "fed_ece", "fed_brier",
                      "fed_auroc_site_min", "fed_auroc_site_p10", "n_clients_single_class"):
                if k in hb:
                    out[f"{k}_valsel"] = float(hb[k])
        ups = [h["upload_mb"] for h in self.history]; au = [h.get("fed_auroc", np.nan) for h in self.history]
        post = [au[i] for i in range(1, len(ups)) if ups[i] > ups[i - 1] + 1e-12 and i >= len(ups) - 20]
        # a method that never synchronises has no phase: use the plain last-20 mean
        out["fed_auroc_phase"] = float(np.nanmean(post)) if post else float(np.nanmean(au[-20:]))
        out["fed_auroc_last20_mean"] = float(np.nanmean(au[-20:]))
        last = self.history[-1]
        for k in [k for k in last if k.startswith("fed_ft")] + [
                "fed_auroc_site_min", "fed_auroc_site_p10", "fed_cal_ece", "fed_cal_brier", "fed_cal_slope"]:
            if k in last:
                out[f"{k}_final"] = float(last[k])
        out["total_upload_mb"] = self.upload_bits / 8e6
        out["total_download_mb"] = self.download_bits / 8e6
        out["rounds"] = float(self.rounds)
        out["rounds_transmitting"] = float(sum(1 for b in self.round_bits if b["upload_bits"] > 0))
        out["realised_payload_fraction"] = float(np.mean([h["payload_fraction"] for h in self.history]))
        out["n_clients_single_class_final"] = float(self.history[-1].get("n_clients_single_class", 0))
        out.update(self.divergence.summary())
        if self.cfg.get("membership_inference", False):
            out.update(self._membership())
        if self.attack_rows:
            for k in ("inversion_nmse", "inversion_feature_corr", "n_params_exposed"):
                out[f"{k}_mean"] = float(np.nanmean([a[k] for a in self.attack_rows]))
        return {"summary": out, "history": self.history, "round_bits": self.round_bits,
                "attack": self.attack_rows, "strategy": self.strategy.describe(),
                "dp": self.dp.summary(self.dp_rounds)}
