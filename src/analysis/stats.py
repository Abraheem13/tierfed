"""Statistical analysis with the rigour the reviews demanded.

Improvements over the original three-seed paired t-test:
  * >= 10 seeds by default, so the paired test has real power;
  * BCa bootstrap confidence intervals that do not assume normality;
  * Holm-Bonferroni correction across the full family of comparisons;
  * Wilcoxon signed-rank reported only when n is large enough for it to be able
    to reach significance (n >= 6), with an explicit note otherwise;
  * Cohen's d with the Hedges' g small-sample correction and a bootstrap CI;
  * a non-inferiority test for claims of the form "no worse than".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Paired standardised effect size with small-sample correction."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan")
    g = d.mean() / d.std(ddof=1)
    J = 1 - 3 / (4 * (n - 1) - 1) if n > 1 else 1.0
    return float(g * J)


def bootstrap_ci(values: Sequence[float], stat=np.mean, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """BCa bootstrap interval."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return {"lo": float("nan"), "hi": float("nan"), "point": float(v[0]) if v.size else float("nan")}
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(v, v.size, replace=True)) for _ in range(n_boot)])
    point = float(stat(v))
    z0 = sps.norm.ppf(np.clip((boots < point).mean(), 1e-6, 1 - 1e-6))
    jack = np.array([stat(np.delete(v, i)) for i in range(v.size)])
    jbar = jack.mean()
    denom = 6.0 * ((((jbar - jack) ** 2).sum()) ** 1.5)
    a = (((jbar - jack) ** 3).sum() / denom) if denom > 0 else 0.0
    out = {}
    for name, q in (("lo", alpha / 2), ("hi", 1 - alpha / 2)):
        z = sps.norm.ppf(q)
        adj = sps.norm.cdf(z0 + (z0 + z) / max(1 - a * (z0 + z), 1e-9))
        out[name] = float(np.quantile(boots, np.clip(adj, 0, 1)))
    out["point"] = point
    return out


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05):
    """Step-down Holm correction; returns adjusted p-values and reject flags."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj, adj < alpha


def paired_comparison(a: Sequence[float], b: Sequence[float], label: str = "",
                      seed: int = 0) -> Dict[str, float]:
    """Full paired report for one contrast (a = method, b = baseline)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = len(a)
    out: Dict[str, float] = {"comparison": label, "n_seeds": n,
                             "mean_method": float(a.mean()) if n else np.nan,
                             "mean_baseline": float(b.mean()) if n else np.nan,
                             "mean_diff": float((a - b).mean()) if n else np.nan}
    if n >= 2:
        t, p = sps.ttest_rel(a, b)
        out["t_stat"], out["t_p"] = float(t), float(p)
        out["hedges_g"] = hedges_g(a, b)
        ci = bootstrap_ci(a - b, seed=seed)
        out["diff_ci_lo"], out["diff_ci_hi"] = ci["lo"], ci["hi"]
    # Wilcoxon can never reach alpha=0.05 below n=6; say so instead of printing 0.25.
    if n >= 6:
        try:
            w, wp = sps.wilcoxon(a, b)
            out["wilcoxon_stat"], out["wilcoxon_p"] = float(w), float(wp)
        except ValueError:
            out["wilcoxon_stat"], out["wilcoxon_p"] = np.nan, np.nan
    else:
        out["wilcoxon_stat"], out["wilcoxon_p"] = np.nan, np.nan
        out["wilcoxon_note"] = "n<6: signed-rank cannot reach p<0.05"
    return out


def non_inferiority(a: Sequence[float], b: Sequence[float], margin: float,
                    alpha: float = 0.05) -> Dict[str, float]:
    """One-sided test that `a` is not worse than `b` by more than `margin`.

    Used for the peak-AUROC claim, which is an equivalence claim and must not
    be argued from a non-significant difference test.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "p": float("nan"), "non_inferior": False}
    t = (d.mean() + margin) / (d.std(ddof=1) / np.sqrt(n))
    p = 1 - sps.t.cdf(t, df=n - 1)
    return {"n": n, "margin": margin, "mean_diff": float(d.mean()),
            "t_stat": float(t), "p": float(p), "non_inferior": bool(p < alpha)}


def summarise_runs(df: pd.DataFrame, group_cols: Sequence[str],
                   metrics: Sequence[str], seed: int = 0) -> pd.DataFrame:
    """Mean, SD, and bootstrap CI per configuration."""
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = len(g)
        for m in metrics:
            if m not in g:
                continue
            v = g[m].astype(float).values
            row[f"{m}_mean"] = float(np.nanmean(v))
            row[f"{m}_std"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
            ci = bootstrap_ci(v, seed=seed)
            row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = ci["lo"], ci["hi"]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_all(df: pd.DataFrame, method_col: str, baseline: str, metrics: Sequence[str],
                seed_col: str = "seed", alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """All methods vs one baseline, across metrics, with Holm correction."""
    rows: List[Dict[str, float]] = []
    base = df[df[method_col] == baseline].set_index(seed_col)
    for method in sorted(df[method_col].unique()):
        if method == baseline:
            continue
        cur = df[df[method_col] == method].set_index(seed_col)
        common = sorted(set(cur.index) & set(base.index))
        for m in metrics:
            if m not in df:
                continue
            r = paired_comparison(cur.loc[common, m].values, base.loc[common, m].values,
                                  f"{method}_vs_{baseline}", seed=seed)
            r["metric"] = m; r["method"] = method; r["baseline"] = baseline
            rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty and "t_p" in out:
        adj, rej = holm_bonferroni(out["t_p"].fillna(1.0).values, alpha)
        out["p_holm"] = adj
        out["significant_holm"] = rej
    return out
