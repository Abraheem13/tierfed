"""Wide model + socket link emulation (v4).

1. Train the ~1.8M-parameter `mlp_wide` on Diabetes for the main strategies.
2. Replay every run's logged per-client upload bytes through real loopback TCP
   connections paced at each profile's uplink rate (src/netemu.py), and combine
   with the logged per-round compute time into wall-clock time-to-target-AUROC.
Results: <out>/emulation/<tag>__<profile>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..network import PROFILES, cost_run
from ..netemu import emulate_run, time_to_target
from .runner import run_single

TF = {"warmup": 5, "rho": 0.25}
STRATS = [("fedavg", {}), ("fedper", {}), ("fixed", dict(k_slow=4, k_med=4)), ("tierfed", TF),
          ("adacomm", {})]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--profiles", nargs="+", default=["hospital_wan", "nr_typical", "nr_cell_edge"])
    ap.add_argument("--slow-profiles", nargs="+", default=["nr_cell_edge"],
                    help="profiles emulated for the first seed only (long wall-clock)")
    ap.add_argument("--targets", nargs="+", type=float, default=[0.60, 0.62])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    emu_dir = Path(a.out) / "emulation"; emu_dir.mkdir(parents=True, exist_ok=True)
    for strat, skw in STRATS:
        for seed in a.seeds:
            cfg = dict(dataset=a.dataset, model="mlp_wide", strategy=strat, strategy_kwargs=skw,
                       rounds=a.rounds, n_clients=20, seed=seed, device=a.device, out_dir=a.out, tag="_wide")
            run_single(cfg, skip_existing=True)
            stem = f"{a.dataset}_mlp_wide_{strat}_R{a.rounds}_C20_s{seed}_wide"
            sj = Path(a.out) / "logs" / f"{stem}_summary.json"
            hc = Path(a.out) / "logs" / f"{stem}_history.csv"
            if not sj.exists():
                continue
            summ = json.loads(sj.read_text())
            hist = pd.read_csv(hc).to_dict("records")
            for prof in a.profiles:
                if prof in a.slow_profiles and seed != a.seeds[0]:
                    continue
                f = emu_dir / f"{stem}__{prof}.json"
                if f.exists():
                    continue
                p = PROFILES[prof]
                up = emulate_run(summ["round_bits"], p.uplink_mbps, p.rtt_ms)
                row = {"strategy": strat, "seed": seed, "profile": prof,
                       "emu_upload_s": float(sum(up)),
                       "analytic_upload_s": cost_run(summ["round_bits"], p)[f"net_{prof}_comm_time_s"],
                       "compute_s": float(sum(rb.get("compute_s_max", 0.0) for rb in summ["round_bits"])),
                       "fed_auroc_final": summ["summary"].get("fed_auroc_final"),
                       "upload_mb": summ["summary"].get("total_upload_mb"),
                       "n_params": summ.get("tier_report", {}).get("total", {}).get("params")}
                for t in a.targets:
                    tt = time_to_target(hist, summ["round_bits"], up, t)
                    row[f"tt_{t:g}_s"] = tt["time_to_target_s"]
                    row[f"rt_{t:g}"] = tt["round_to_target"]
                f.write_text(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
