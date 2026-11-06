"""Second-machine replication (S-9). Every run in results/crossmachine/<machine>/logs
is compared with the run of the same name (same configuration and seed) in
results/logs, and the heterogeneity gate and the timing test are recomputed from
the second machine's runs alone, on the same settings, methods and seeds.

  python -m src.analysis.crossmachine --results results

Per pair: the configuration (every key but the device), the bits of every round
and site, the periods the planner chose, the validation-selected round, and the
test AUROC at that round and at the final round. Writes
results/tables/crossmachine.json; <machine>/machine.json describes the machine.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .audit_report import PRACTICAL, gate_table, load, timing_table

SKIP_CFG = {"device"}
BIT_KEYS = ("upload_bits", "download_bits", "client_bits", "max_client_bits", "n_senders", "n_selected")
DATA_DEPENDENT = {"lag", "tierfed", "adacomm", "fedlama"}   # adaptive: their decisions read client data


def _f(x):
    return float(x) if x is not None and np.isfinite(float(x)) else float("nan")


def compare_pair(a: dict, b: dict) -> dict:
    ca = {k: v for k, v in a["config"].items() if k not in SKIP_CFG}
    cb = {k: v for k, v in b["config"].items() if k not in SKIP_CFG}
    ra, rb = a.get("round_bits") or [], b.get("round_bits") or []
    bits_same = len(ra) == len(rb) and all(all(x.get(k) == y.get(k) for k in BIT_KEYS) for x, y in zip(ra, rb))
    sa_, sb_ = a.get("schedule") or {}, b.get("schedule") or {}
    sa, sb = a["summary"], b["summary"]
    mb_a, mb_b = _f(sa.get("total_upload_mb")), _f(sb.get("total_upload_mb"))
    return {"strategy": a["config"].get("strategy"),
            "config_diff": sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)),
            "bits_same": bool(bits_same),
            "K_same": all(sa_.get(k) == sb_.get(k) for k in ("K_slow", "K_medium")),
            "sel_same": sa.get("valsel_round") == sb.get("valsel_round"),
            "d_auroc_valsel": abs(_f(sa.get("fed_auroc_valsel")) - _f(sb.get("fed_auroc_valsel"))),
            "d_auroc_final": abs(_f(sa.get("fed_auroc_final")) - _f(sb.get("fed_auroc_final"))),
            "mb_rel_diff": abs(mb_a - mb_b) / mb_b if mb_b > 0 else 0.0}


def _stats(v):
    v = [x for x in v if np.isfinite(x)]
    return {"median": statistics.median(v), "max": max(v)} if v else {"median": None, "max": None}


def _won(g):
    if g is None or g.empty:
        return []
    return sorted(g[(g.gain_lo > 0) & (g.gain_vs_uniform >= PRACTICAL)].setting.tolist())


def _verdicts(t):
    """Leak verdicts as the paper states them: an observer AUC of at least 0.75."""
    out = {}
    for _, r in (t.iterrows() if t is not None and not t.empty else []):
        aucs = [r.auc_tx_rounds, r.auc_target_tx_rounds, r.auc_target_mb, r.auc_total_mb]
        out[r["name"]] = {"auc_max": float(np.nanmax(aucs)), "leaks": bool(np.nanmax(aucs) >= 0.75)}
    return out


def machine_report(results: Path, mdir: Path, dataset: str) -> dict:
    logs = results / "logs"
    pairs, unmatched = {}, 0
    for f in sorted((mdir / "logs").glob("*_summary.json")):
        ref = logs / f.name
        if not ref.exists():
            unmatched += 1
            continue
        pairs[f.name] = compare_pair(json.loads(f.read_text()), json.loads(ref.read_text()))
    rows = list(pairs.values())
    out = {"machine": json.loads((mdir / "machine.json").read_text()) if (mdir / "machine.json").exists() else {},
           "pairs": len(rows), "unmatched": unmatched,
           "config_differs": sum(bool(r["config_diff"]) for r in rows),
           "bits_same": sum(r["bits_same"] for r in rows),
           "bits_same_data_independent": sum(r["bits_same"] for r in rows if r["strategy"] not in DATA_DEPENDENT),
           "pairs_data_independent": sum(r["strategy"] not in DATA_DEPENDENT for r in rows),
           "pairs_adaptive": sum(r["strategy"] in DATA_DEPENDENT for r in rows),
           "bits_same_adaptive": sum(r["bits_same"] for r in rows if r["strategy"] in DATA_DEPENDENT),
           "adaptive_mb_rel_diff_max": max([r["mb_rel_diff"] for r in rows if r["strategy"] in DATA_DEPENDENT],
                                           default=0.0),
           "K_same": sum(r["K_same"] for r in rows),
           "sel_same": sum(r["sel_same"] for r in rows),
           "d_auroc_valsel": _stats([r["d_auroc_valsel"] for r in rows]),
           "d_auroc_final": _stats([r["d_auroc_final"] for r in rows]),
           "max_gap_run": max(pairs, key=lambda k: pairs[k]["d_auroc_valsel"]) if pairs else None,
           "failures": {k: r for k, r in pairs.items() if r["config_diff"]
                        or (not r["bits_same"] and r["strategy"] not in DATA_DEPENDENT)}}
    # the gate and the timing test on each machine's copy of the same runs
    if rows:
        df2 = load(str(mdir), dataset)
        df1 = load(str(results), dataset)
        if not df2.empty and not df1.empty:
            keep = set(pairs)
            df1 = df1[df1.file.map(lambda p: Path(p).name in keep)]
            df2 = df2[df2.file.map(lambda p: Path(p).name in keep)]
            g1, g2 = gate_table(df1), gate_table(df2)
            if not g1.empty and not g2.empty:
                m = g1.set_index("setting")[["gain_vs_uniform", "gain_lo", "gain_hi"]].join(
                    g2.set_index("setting")[["gain_vs_uniform", "gain_lo", "gain_hi"]], rsuffix="_2", how="inner")
                out["gate"] = {"settings": m.index.tolist(), "won_ref": _won(g1), "won_second": _won(g2),
                               "same": _won(g1) == _won(g2),
                               "gain_ref": m.gain_vs_uniform.round(4).tolist(),
                               "gain_second": m.gain_vs_uniform_2.round(4).tolist(),
                               "gain_absdiff_max": float((m.gain_vs_uniform - m.gain_vs_uniform_2).abs().max())}
            t1, t2 = _verdicts(timing_table(df1)), _verdicts(timing_table(df2))
            common = sorted(set(t1) & set(t2))
            if common:
                out["timing"] = {"methods": common, "ref": {n: t1[n] for n in common},
                                 "second": {n: t2[n] for n in common},
                                 "same": all(t1[n]["leaks"] == t2[n]["leaks"] for n in common)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", default="diabetes")
    a = ap.parse_args(argv)
    root = Path(a.results) / "crossmachine"
    rep = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        rep[mdir.name] = r = machine_report(Path(a.results), mdir, a.dataset)
        print(f"{mdir.name}: {r['pairs']} pairs ({r['unmatched']} without a counterpart) | config differs "
              f"{r['config_differs']} | bits identical {r['bits_same_data_independent']}/{r['pairs_data_independent']} "
              f"(data-independent), {r['bits_same']}/{r['pairs']} (all) | periods identical {r['K_same']} | same "
              f"selected round {r['sel_same']} | |dAUROC| at selection {r['d_auroc_valsel']}, final {r['d_auroc_final']}")
        if "gate" in r:
            print(f"  gate: settings won {r['gate']['won_ref']} (reference) vs {r['gate']['won_second']} (second); "
                  f"max |d gain| {r['gate']['gain_absdiff_max']:.4f}")
        if "timing" in r:
            print(f"  timing verdicts identical: {r['timing']['same']}")
    Path(a.results, "tables").mkdir(parents=True, exist_ok=True)
    Path(a.results, "tables", "crossmachine.json").write_text(json.dumps(rep, indent=1))
    for k, r in rep.items():
        for f, why in list(r["failures"].items())[:10]:
            print(f"  WARNING {k}: {f} -> {why}")


if __name__ == "__main__":
    main()
