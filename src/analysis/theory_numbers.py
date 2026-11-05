"""Numbers for the theory paragraphs, computed from the logs and from the
accountants (nothing typed by hand).

  differentiation potential  Psi of Proposition 2 from the dispersions measured at
                             the end of warm-up (every tier synchronised every round,
                             so the interval bias of the proxy is common and cancels)
  accountant gap at q = 1    noise multiplier of the exact Gaussian-DP accountant
                             relative to the RDP one; it does not depend on the
                             release count, because both depend on R_tx / z^2 only
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..privacy.dp import calibrate_sigma
from ..privacy.gdp import eps_gdp, sigma_gdp
from ..theory.differentiation import closed_form

GROUPS = {  # macro prefix -> filename pattern of TierFed runs (rho = 0.25, T_w = 5)
    "Mlp": "diabetes_mlp_tierfed_R60_C20_s*_summary.json",
    "Resnet": "diabetes_resnet_tierfed_R60_C20_s*_architecture_resnet_tierfed_summary.json",
    "Transformer": "diabetes_transformer_tierfed_R60_C20_s*_architecture_transformer_tierfed_summary.json",
    "Wide": "diabetes_mlp_wide_tierfed_R60_C20_s*_wide_summary.json",
    "Heart": "heart_mlp_tierfed_R60_C4_s*_real_summary.json",
    "Cnn": "medmnist_cnn_tierfed_R60_C*_s*_summary.json",
    "AuResnet": "diabetes_resnet_tierfed_R60_C20_s*_au_bb_tierfed_summary.json",
    "AuTransformer": "diabetes_transformer_tierfed_R60_C20_s*_au_bb_tierfed_summary.json",
}


def warmup_psi(summary_file: str):
    d = json.loads(Path(summary_file).read_text())
    sch = d.get("schedule") or {}
    w = {k[2:]: v for k, v in sch.items() if k.startswith("w_")}
    tw = int((d.get("config", {}).get("strategy_kwargs") or {}).get("warmup", 5))
    h = Path(summary_file.replace("_summary.json", "_history.csv"))
    if len(w) != 2 or tw < 1 or not h.exists():
        return None
    hist = pd.read_csv(h)
    if len(hist) < tw or int(d.get("config", {}).get("eval_every", 1)) != 1:
        return None            # needs one history row per round (read by position)
    row = hist.iloc[tw - 1]
    g = {t: float(row.get(f"divergence_{t}", np.nan)) for t in w}
    if any(not np.isfinite(v) or v <= 0 for v in g.values()):
        return None
    cf = closed_form(w, g, 0.25)
    K = {t: sch.get(f"K_{t}") for t in w}
    return {"psi": cf["psi"], "ratio": cf["K"]["medium"] / cf["K"]["slow"],
            "K_slow": K["slow"], "K_medium": K["medium"]}


def write(logs: Path, W) -> None:
    for name, pat in GROUPS.items():
        files = [f for f in glob.glob(str(logs / pat)) if "_au_" not in f or name.startswith("Au")]
        if name in ("Mlp", "Cnn"):   # only the plain main-benchmark runs, no tagged variants
            files = [f for f in files if re.search(r"_s\d+_summary\.json$", f)]
        rows = [r for r in (warmup_psi(f) for f in files) if r]
        if not rows:
            continue
        p = np.array([r["psi"] for r in rows]); q = np.array([r["ratio"] for r in rows])
        W.num(f"psi{name}Pct", f"{100 * p.mean():.1f}" if p.mean() >= 0.01 else f"{100 * p.mean():.2f}")
        W.num(f"psi{name}SdPct", f"{100 * p.std(ddof=1):.1f}" if len(p) > 1 else "--")
        W.num(f"ratio{name}", f"{q.mean():.2f}")
        W.num(f"psi{name}Seeds", str(len(rows)))
        ks = pd.Series([f"({int(r['K_slow'])},{int(r['K_medium'])})" for r in rows if r["K_slow"] is not None])
        if len(ks):
            W.num(f"modalK{name}", ks.mode().iloc[0])
    # exact Gaussian accounting at q = 1 (independent of the release count)
    for eps, lab in ((1.0, "One"), (2.0, "Two"), (4.0, "Four"), (8.0, "Eight")):
        zr, zg = calibrate_sigma(eps, 60, 1.0, 1e-5), sigma_gdp(eps, 60, 1e-5)
        W.num(f"gdpSigmaCut{lab}Pct", f"{100 * (1 - zg / zr):.0f}")
        W.num(f"gdpEpsAtRdp{lab}", f"{eps_gdp(60, zr, 1e-5):.2f}")
