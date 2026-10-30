"""Federated simulation engine.

Self-contained (no Flower dependency) so that tier-wise communication,
compression, DP noise, attack surface and per-round diagnostics are measured
exactly. Differential privacy is applied HERE, to every strategy, with
per-client clipping (v2 applied it inside two strategies only and clipped the
aggregate).
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch

from .client import Client
from .metrics import TrajectoryStats, classification_metrics, weighted_average
from .privacy.dp import DPConfig, private_aggregate
from .privacy.inversion import attack_update
from .privacy.membership import membership_inference
from .theory.divergence import DivergenceTracker
from .utils import clone_state, get_logger

log = get_logger()


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
        self.local_tr, self.local_te = dataset.client_test_split(
            frac=cfg.get("client_test_frac", 0.2), seed=self.seed)
        self.clients = [Client(k, dataset, model_fn, device, cfg, train_idx=self.local_tr[k])
                        for k in range(dataset.n_clients)]
        self.history: List[Dict[str, float]] = []
        self.round_bits: List[Dict[str, float]] = []      # for the network model
        self.upload_bits = 0.0
        self.download_bits = 0.0
        self.dp_rounds = 0
        self.divergence = DivergenceTracker()
        self.attack_rows: List[Dict[str, float]] = []
        self.noise_gen = torch.Generator(device="cpu").manual_seed(self.seed + 7919)

    # ------------------------------------------------------------------ #
    def _sample(self, r: int) -> List[int]:
        n = len(self.clients)
        m = max(1, int(round(self.participation * n)))
        if m >= n:
            return list(range(n))
        rng = np.random.default_rng(self.seed * 1000 + r)
        return sorted(rng.choice(n, size=m, replace=False).tolist())

    def run(self) -> Dict:
        priv = self.strategy.personal_keys()
        attack_every = int(self.cfg.get("attack_every", 0))
        for r in range(1, self.rounds + 1):
            keys = self.strategy.send_keys(r)
            sel = self._sample(r)
            payload = {k: self.global_state[k] for k in keys}
            dl_bits = len(sel) * sum(v.numel() * 32 for v in payload.values())
            self.download_bits += dl_bits

            states, weights, stats, senders = [], [], [], []
            per_client_bits = []
            for cid in sel:
                c = self.clients[cid]
                c.set_parameters(self.global_state, keys)
                s = c.train(self.strategy, payload if keys else None, self.local_epochs, self.seed)
                c.stash_private(priv)
                raw = c.get_slice(keys)
                self.strategy.after_local(cid, self.global_state, raw, keys, s["steps"], s["lr"])
                stats.append(s)
                if not keys or not self.strategy.client_sends(cid, self.global_state, raw, keys, r):
                    per_client_bits.append(0.0)
                    continue
                cs = self.strategy.compress_client_state(raw, keys)
                bits = self.strategy.measure_upload([cs], keys)
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
                                    "n_senders": float(len(senders))})
            if hasattr(self.strategy, "observe_loss"):
                self.strategy.observe_loss(r, float(np.mean([s["loss"] for s in stats])))

            dp_info = {}
            if keys and states and self.dp.enabled:
                self.global_state, dp_info = private_aggregate(
                    self.global_state, states, keys, self.dp.max_norm, self.dp.sigma, self.noise_gen)
                self.dp_rounds += 1
                # schedulers still see the raw (pre-noise) slices for their diagnostics
                self.strategy.observe(self.global_state, states, keys, r)
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
        for cid, c in enumerate(self.clients):
            idx = self.local_te[cid]
            if len(idx) == 0:
                continue
            c.set_parameters(self.global_state, keys)
            x = torch.as_tensor(self.dataset.x_train[idx], dtype=torch.float32)
            y = self.dataset.y_train[idx]
            p = c.predict(x)
            y_all.append(y); p_all.append(p)
            if len(np.unique(y)) < 2:
                n_single += 1
                continue
            per_client.append(classification_metrics(y, p)); weights.append(len(idx))
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
        row.update({f"pooled_{k}": v for k, v in pooled.items()})
        row.update({f"global_{k}": v for k, v in glob.items()})
        row.update(self.divergence.latest())
        if hasattr(self.strategy, "K"):
            row.update({f"K_{g}": float(v) for g, v in self.strategy.K.items()})
        return row

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
