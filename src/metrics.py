"""Evaluation metrics.

Adds AUPRC (and prevalence-normalised AUPRC lift), calibration, and a set of
*standard* trajectory statistics reported alongside the custom post-peak
degradation (PPD) statistic, so that no conclusion rests on PPD alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, roc_auc_score)


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def best_f1_threshold(y_true: np.ndarray, p: np.ndarray):
    """Threshold-optimised F1 (companion to F1 at the fixed 0.5 operating point)."""
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    k = int(np.nanargmax(f1))
    t = float(thr[min(k, len(thr) - 1)]) if len(thr) else 0.5
    return float(f1[k]), t


def classification_metrics(y_true, p, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    keys = ("auroc", "auprc", "auprc_lift", "f1", "f1_best", "f1_best_threshold",
            "ece", "brier", "prevalence")
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in keys}
    prevalence = float(y_true.mean())
    out: Dict[str, float] = {}
    out["auroc"] = float(roc_auc_score(y_true, p))
    out["auprc"] = float(average_precision_score(y_true, p))
    out["auprc_lift"] = float(out["auprc"] / prevalence) if prevalence > 0 else float("nan")
    out["f1"] = float(f1_score(y_true, (p >= threshold).astype(int), zero_division=0))
    fb, tb = best_f1_threshold(y_true, p)
    out["f1_best"], out["f1_best_threshold"] = fb, tb
    out["ece"] = expected_calibration_error(y_true, p)
    out["brier"] = float(brier_score_loss(y_true, p))
    out["prevalence"] = prevalence
    return out


def weighted_average(metrics: List[Dict[str, float]], weights: Sequence[float]) -> Dict[str, float]:
    if not metrics:
        return {}
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    out = {}
    for k in set().union(*[set(m) for m in metrics]):
        vals = np.array([m.get(k, np.nan) for m in metrics], dtype=float)
        mask = ~np.isnan(vals)
        out[k] = float(np.sum(vals[mask] * w[mask]) / w[mask].sum()) if mask.any() else float("nan")
    return out


@dataclass
class TrajectoryStats:
    """Round-indexed summary of a single run for one metric."""
    metric: str
    values: List[float] = field(default_factory=list)

    @property
    def final(self) -> float: return float(self.values[-1])
    @property
    def best(self) -> float: return float(np.nanmax(self.values))
    @property
    def best_round(self) -> int: return int(np.nanargmax(self.values)) + 1
    @property
    def ppd(self) -> float:
        """Peak minus final: accuracy lost by training past the peak."""
        return float(self.best - self.final)
    @property
    def auc_of_curve(self) -> float:
        """Mean over rounds: rewards being good throughout training."""
        return float(np.nanmean(self.values))

    def last_k_mean(self, k: int = 10) -> float:
        return float(np.nanmean(self.values[-k:]))

    def early_stopped_value(self, patience: int = 10) -> Dict[str, float]:
        """What patience-based early stopping would actually deploy.

        Reported so PPD cannot be dismissed as penalising overfitting that
        standard early stopping would have handled.
        """
        best, best_r, wait, r = -np.inf, 0, 0, 0
        for r, v in enumerate(self.values):
            if v > best:
                best, best_r, wait = v, r, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        return {"value": float(best), "round": float(best_r + 1), "stopped_round": float(r + 1)}

    def summary(self, patience: int = 10) -> Dict[str, float]:
        es = self.early_stopped_value(patience)
        return {
            f"{self.metric}_final": self.final,
            f"{self.metric}_best": self.best,
            f"{self.metric}_best_round": float(self.best_round),
            f"{self.metric}_ppd": self.ppd,
            f"{self.metric}_auc_curve": self.auc_of_curve,
            f"{self.metric}_last10_mean": self.last_k_mean(10),
            f"{self.metric}_earlystop_value": es["value"],
            f"{self.metric}_earlystop_round": es["round"],
        }
