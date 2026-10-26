#!/usr/bin/env bash
# Runs the same configuration twice and asserts bit-identical summaries.
# v2 runs with identical configs differed by up to 0.006 AUROC; the paper's
# claims are only meaningful if this check passes on the machine that produced
# the logs.
set -euo pipefail
export PYTHONHASHSEED=0
cd "$(dirname "$0")/.."
DS="${1:-diabetes}"; DEV="${2:-cpu}"
OUT=results/determinism
rm -rf "$OUT"; mkdir -p "$OUT"
for rep in a b; do
  python -m src.experiments.run_one --dataset "$DS" --strategy tierfed --seed 42 --rounds 8 --clients 20 \
      --device "$DEV" --out "$OUT/$rep" --strategy-kwargs '{"warmup":2,"rho":0.25}' >/dev/null
done
python - <<'EOF'
import json, glob, sys
a = json.load(open(glob.glob("results/determinism/a/logs/*_summary.json")[0]))["summary"]
b = json.load(open(glob.glob("results/determinism/b/logs/*_summary.json")[0]))["summary"]
keys = ["fed_auroc_final", "fed_auprc_final", "total_upload_mb", "divergence_total_mean"]
diff = {k: abs(a[k] - b[k]) for k in keys}
print("determinism check:", diff)
if max(diff.values()) > 0:
    print("FAIL: repeated runs differ; fix the environment before producing logs"); sys.exit(1)
print("PASS: repeated runs are bit-identical")
EOF
