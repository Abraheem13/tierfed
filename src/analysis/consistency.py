"""Consistency audit (S-8): recompute every transmitting-round count, every
upload volume and every DP release count from the schedule definitions, and
compare them with the logs.

  python -m src.analysis.consistency --results results

Checks per run (where they apply):
  internal   rounds_transmitting and total_upload_mb equal the per-round log
  schedule   the transmitting rounds equal the rounds the stated rule gives
             (fixed periods: the legacy rule or the shared due rule; the DP
             planner: the shared due rule with its K; full rate: every round;
             local-only: none)
  bits       each transmitting round sends n_senders x 32 x (parameters of the
             due tiers) bits, for uncompressed full-participation runs
  accountant the DP accountant charged exactly the planned releases
Exits with status 1 if any check fails.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from ..strategies.schedule_rules import due_tiers, release_rounds, stagger_phases

FULL_RATE = {"fedavg", "fedper", "fedprox", "scaffold"}


def _fixed_due(kw, R):
    ks, km = int(kw.get("k_slow", 5)), int(kw.get("k_med", 2))
    warm = int(kw.get("warmup", 0))
    if not kw.get("schedule", True):
        return {r: ["medium", "slow"] for r in range(1, R + 1)}
    if kw.get("aligned") or kw.get("stagger"):
        K = {"slow": ks, "medium": km}
        ph = stagger_phases(K) if kw.get("stagger") else None
        return {r: due_tiers(r, K, warm, ph) for r in range(1, R + 1)}
    out = {}
    for r in range(1, R + 1):                      # the legacy rule of the v3-v12 runs
        if r <= warm or r == 1:
            out[r] = ["medium", "slow"]
        else:
            out[r] = sorted(g for g, k in (("slow", ks), ("medium", km)) if r % k == 0)
    return out


def expected_due(d):
    """Due shared tiers per round for data-independent schedules, else None."""
    c = d.get("config", {}); kw = c.get("strategy_kwargs") or {}
    R, s = int(c.get("rounds", 60)), c.get("strategy")
    if s == "local":
        return {r: [] for r in range(1, R + 1)}
    if s in FULL_RATE:
        return {r: ["medium", "slow"] for r in range(1, R + 1)}
    if s == "fixed" and kw.get("private_head", True):
        return _fixed_due(kw, R)
    if s == "tierfed_dp":
        sch = d.get("schedule") or {}
        K = {g: int(sch[f"K_{g}"]) for g in ("slow", "medium") if f"K_{g}" in sch}
        if len(K) == 2:
            return {r: due_tiers(r, K, int(kw.get("warmup", 5))) for r in range(1, R + 1)}
    return None


def check(d):
    fails, done = [], []
    s, rb, c = d["summary"], d.get("round_bits") or [], d.get("config", {})
    if not rb:
        return fails, done
    tx = sum(1 for b in rb if b["upload_bits"] > 0)
    mb = sum(b["upload_bits"] for b in rb) / 8e6
    done.append("internal")
    if int(s.get("rounds_transmitting", -1)) != tx:
        fails.append(f"rounds_transmitting {s.get('rounds_transmitting')} != {tx} in the round log")
    if abs(float(s.get("total_upload_mb", -1)) - mb) > 1e-6 * max(1.0, mb):
        fails.append(f"total_upload_mb {s.get('total_upload_mb')} != {mb} in the round log")

    dpc = c.get("dp") or {}
    poisson = bool(dpc.get("enabled")) and float(dpc.get("sample_rate", 1.0)) < 1.0
    full_part = float(c.get("participation", 1.0)) >= 1.0 and not poisson
    due = expected_due(d)
    if due is not None and full_part and c.get("strategy") != "lag":
        done.append("schedule")
        planned = [r for r in sorted(due) if due[r]]
        logged = [int(b["round"]) for b in rb if b["upload_bits"] > 0]
        if planned != logged:
            fails.append(f"transmitting rounds {logged[:8]}... ({len(logged)}) != rule {planned[:8]}... ({len(planned)})")
        tr = d.get("tier_report") or {}
        if not c.get("compressor") and tr and not (c.get("strategy_kwargs") or {}).get("warmup_compressor"):
            done.append("bits")
            per = {g: tr[g]["params"] * 32.0 for g in ("slow", "medium", "fast")}
            mult = 2.0 if c.get("strategy") == "scaffold" else 1.0
            head = per["fast"] if c.get("strategy") in ("fedavg", "fedprox", "scaffold") else 0.0
            for b in rb:
                r = int(b["round"]); g = due.get(r, [])
                exp = (sum(per[t] for t in g) + (head if g else 0.0)) * b["n_senders"] * mult
                if abs(b["upload_bits"] - exp) > 1e-6 * max(1.0, exp):
                    fails.append(f"round {r}: {b['upload_bits']:.0f} bits != {exp:.0f} from the due tiers")
                    break
        if dpc.get("enabled"):
            done.append("accountant")
            charged = int((d.get("dp") or {}).get("dp_rounds_charged", -1))
            if charged != len(planned):
                fails.append(f"accountant charged {charged} releases, the rule gives {len(planned)}")
    return fails, done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--pattern", default="*_summary.json")
    a = ap.parse_args(argv)
    counts, failures = {}, {}
    files = sorted(glob.glob(str(Path(a.results) / "logs" / a.pattern)))
    for f in files:
        d = json.loads(Path(f).read_text())
        if "summary" not in d or "config" not in d:
            continue
        fails, done = check(d)
        for k in done:
            counts[k] = counts.get(k, 0) + 1
        if fails:
            failures[Path(f).name] = fails
    out = {"files": len(files), "checks_run": counts, "n_failed": len(failures), "failures": failures}
    Path(a.results, "tables").mkdir(parents=True, exist_ok=True)
    Path(a.results, "tables", "consistency.json").write_text(json.dumps(out, indent=1))
    print(f"{len(files)} logs | checks run: {counts} | runs with a failure: {len(failures)}")
    for k, v in list(failures.items())[:10]:
        print(" ", k, "->", v[0])
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
