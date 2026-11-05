"""Reference bounds on the SAME per-site train/validation/test splits as the
audit runs (E-4): a centralised model trained on the pooled training splits of
all sites, and a gradient-boosted tree ensemble as a model-quality reference.

Both are scored exactly like the federated runs: each site's test split,
sample-weighted, sites with a single-class split excluded. The epoch (MLP) or
number of boosting iterations (trees) is chosen on the pooled validation
splits, never on test. The centralised MLP gets the same sample passes as a
60-round federated run with one local epoch (60 epochs over the pooled data).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..data import load_dataset
from ..losses import build_loss
from ..metrics import classification_metrics, weighted_average
from ..models import build_model
from ..utils import get_logger, save_json, set_seed
from .runner import ARCH

log = get_logger()


def _site_auroc(ds, parts, prob_of):
    """Sample-weighted per-site AUROC; prob_of(idx) returns P(y = 1)."""
    ms, ws, n_single = [], [], 0
    for idx in parts:
        y = ds.y_train[idx]
        if len(idx) == 0:
            continue
        if len(np.unique(y)) < 2:
            n_single += 1
            continue
        ms.append(classification_metrics(y, prob_of(idx))); ws.append(len(idx))
    out = weighted_average(ms, ws)
    out["n_single"] = n_single
    return out


def central_mlp(ds, tr, va, te, seed, epochs, device="cpu"):
    set_seed(seed)
    pooled = np.concatenate(tr)
    model = build_model("mlp", ds.input_shape, 2, **ARCH["mlp"]).to(device)
    y = ds.y_train[pooled]
    loss_fn = build_loss("weighted_ce", [int((y == 0).sum()), int((y == 1).sum())], device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = ds.loader_for(pooled, batch_size=64, seed=seed)
    x_all = torch.as_tensor(ds.x_train, dtype=torch.float32)

    def prob_of(idx):
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(x_all[idx].to(device)), 1)[:, 1].cpu().numpy()

    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward(); opt.step()
        v, t = _site_auroc(ds, va, prob_of), _site_auroc(ds, te, prob_of)
        hist.append({"epoch": ep, "val_auroc": v.get("auroc", np.nan), "test_auroc": t.get("auroc", np.nan),
                     "test_auprc": t.get("auprc", np.nan), "test_ece": t.get("ece", np.nan),
                     "n_single": t["n_single"]})
    return hist


def boosted(ds, tr, va, te, seed, max_iter=600, step=10):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pooled = np.concatenate(tr)
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.05, early_stopping=False,
                                         random_state=seed)
    clf.fit(ds.x_train[pooled].reshape(len(pooled), -1), ds.y_train[pooled])
    va_all, te_all = np.concatenate(va), np.concatenate(te)
    xv = ds.x_train[va_all].reshape(len(va_all), -1); xt = ds.x_train[te_all].reshape(len(te_all), -1)
    pos_v = {int(i): k for k, i in enumerate(va_all)}; pos_t = {int(i): k for k, i in enumerate(te_all)}
    hist = []
    for it, (pv, pt) in enumerate(zip(clf.staged_predict_proba(xv), clf.staged_predict_proba(xt)), start=1):
        if it % step:
            continue
        v = _site_auroc(ds, va, lambda idx: pv[[pos_v[int(i)] for i in idx], 1])
        t = _site_auroc(ds, te, lambda idx: pt[[pos_t[int(i)] for i in idx], 1])
        hist.append({"epoch": it, "val_auroc": v.get("auroc", np.nan), "test_auroc": t.get("auroc", np.nan),
                     "test_auprc": t.get("auprc", np.nan), "test_ece": t.get("ece", np.nan),
                     "n_single": t["n_single"]})
    return hist


def summarise(hist):
    v = np.array([h["val_auroc"] for h in hist], float)
    t = np.array([h["test_auroc"] for h in hist], float)
    i = int(np.nanargmax(v))                       # first maximum = earliest on ties
    return {"fed_auroc_valsel": float(t[i]), "fed_val_auroc_valsel": float(v[i]),
            "valsel_round": float(hist[i]["epoch"]), "fed_auprc_valsel": float(hist[i]["test_auprc"]),
            "fed_ece_valsel": float(hist[i]["test_ece"]), "fed_auroc_final": float(t[-1]),
            "fed_auroc_last10_mean": float(np.nanmean(t[-10:])), "total_upload_mb": float("nan"),
            "n_clients_single_class_final": float(hist[-1]["n_single"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Centralised and gradient-boosted references (E-4)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--alpha", type=float, default=None, help="Dirichlet alpha (hetero sweep); default specialty sites")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    dkw = {"partition": "dirichlet", "alpha": a.alpha} if a.alpha is not None else {}
    if a.smoke:
        a.dataset, a.clients, a.epochs = "synthetic", 4, 4
        dkw = {**dkw, "n_per_client": 120, "n_features": 20}
    logs = Path(a.out) / "logs"; logs.mkdir(parents=True, exist_ok=True)
    het = f"_het_a{a.alpha:g}" if a.alpha is not None else ""
    for seed in a.seeds:
        ds = load_dataset(a.dataset, n_clients=a.clients, seed=seed, **dkw)
        # the same split the federated runs use (FederatedServer with val_split=True)
        tr, va, te = ds.client_val_test_split(frac=0.3, val_share=1 / 3, seed=seed)
        for model, fn in (("central", lambda: central_mlp(ds, tr, va, te, seed, a.epochs)),
                          ("hgb", lambda: boosted(ds, tr, va, te, seed))):
            tag = f"{a.dataset}_mlp_{model}_R{a.epochs}_C{ds.n_clients}_s{seed}_au{het}_bound"
            f = logs / f"{tag}_summary.json"
            if f.exists():
                continue
            hist = fn()
            s = summarise(hist)
            save_json(f, {"summary": s, "history": hist, "config": {"strategy": model, "seed": seed,
                          "dataset": a.dataset, "dataset_kwargs": dkw, "tag": f"_au{het}_bound",
                          "epochs": a.epochs, "split": "70/10/20 patient-grouped per site"}})
            log.info(f"{model:>8s} | seed {seed}{het} | test AUROC at val-selected "
                     f"{'epoch' if model == 'central' else 'iteration'} {s['fed_auroc_valsel']:.4f}")


if __name__ == "__main__":
    main()
