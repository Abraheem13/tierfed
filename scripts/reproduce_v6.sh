#!/usr/bin/env bash
# One-click reproduction of every v6 run and number used in the paper (Linux/macOS).
#   bash scripts/reproduce_v6.sh [core|gate|resnet|analysis|all] [workers]
# Finished runs are skipped, so the script can be stopped and restarted at any time.
# Every part except analysis first checks that repeated runs are bit-identical on this machine.
# core   : main table, reference bounds, frontier, DP at q = 1, peak         (235 runs)
# gate   : heterogeneity sweep and timing side channel                       (140 runs)
# resnet : ResNet backbone, the regime where Proposition 2 predicts a gain    (30 runs)
set -u
PART=${1:-all}; J=${2:-4}; PY=${PY:-python3}
export PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH=0   # figure PDFs without a creation date: byte-identical on every rebuild
cd "$(dirname "$0")/.."
mkdir -p results/v6_logs
S10="42 43 44 45 46 47 48 49 50 51"; S6="42 43 44 45 46 47"; S5="42 43 44 45 46"
MAIN="local fedavg fedprox fedper fedlama adacomm8 lag_uniform lag fixed44 fixed44_st fixed88 tierfed tierfed_st"
A="-m src.experiments.run_audit --device cpu --out results"

core() {
  for s in $S10; do echo "$A --stage main --only $MAIN --seeds $s"; done
  for s in $S6; do echo "$A --stage dp --eps 4 --q 1 --only fedper fixed44 fixed88 oneshot tierfed_dp --seeds $s"; done
  for s in $S10; do echo "-m src.experiments.run_bounds --seeds $s --out results"; done
  for s in $S5; do echo "$A --stage frontier --only fixed22 fixed66 adacomm4 adacomm16 tierfed_rho0.15 tierfed_rho0.4 --seeds $s"; done
  for s in $S5; do echo "$A --stage frontier2 --only lag_xi5000 lag_xi10000 --seeds $s"; done
  for s in $S5; do echo "$A --stage peak --only tierfed_st_w8 fixed44_st_w8 tierfed_all --seeds $s"; done
}
gate() {
  for a in 0.1 0.3 1 10; do for s in $S5; do
    echo "$A --stage hetero --alpha $a --only fedper local fixed44 fixed88 tierfed --seeds $s"; done; done
  for s in $S5; do echo "$A --stage timing --model mlp --only tierfed lag_xi2000 lag_xi5000 fixed44 --seeds $s"; done
}
resnet() {
  for s in $S6; do echo "$A --stage backbone --model resnet --only fedper local fixed44 fixed15 tierfed --seeds $s"; done
}
run() {   # one python process per line, J at a time; each writes its own log
  nl -nln | while read -r i cmd; do echo "$i $cmd"; done |
    xargs -P "$J" -L 1 sh -c 'i=$1; shift; '"$PY"' "$@" > results/v6_logs/job_$i.log 2>&1' _
}
analysis() {
  $PY -m src.analysis.consistency --results results
  $PY -m src.analysis.crossmachine --results results
  $PY -m src.analysis.audit_report --results results --dataset diabetes > results/tables/audit_report_stdout.txt
  $PY -m src.analysis.audit_tex --results results --paper paper/generated --dataset diabetes
  $PY -m src.analysis.audit_figures --results results --out paper/figures --dataset diabetes
  $PY -m src.analysis.results_figures --results results --out paper/figures
  $PY -m src.analysis.collect --results results --dataset diabetes --model mlp
  $PY -m src.analysis.make_figures --results results --dataset diabetes --model mlp   # supplement figure
  if command -v pdflatex >/dev/null; then
    (cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null; bibtex main >/dev/null;
     pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null
     pdflatex -interaction=nonstopmode supp.tex >/dev/null; bibtex supp >/dev/null
     pdflatex -interaction=nonstopmode supp.tex >/dev/null; pdflatex -interaction=nonstopmode supp.tex >/dev/null)
  fi
}
$PY -c "from src.data import load_dataset; load_dataset('diabetes', n_clients=20, seed=42)" >/dev/null   # cache the data once
if [ "$PART" != "analysis" ]; then   # repeated runs must be bit-identical before any log is produced
  PY="$PY" bash scripts/check_determinism.sh diabetes cpu || { echo "determinism check failed: fix the environment first"; exit 1; }
fi
case "$PART" in
  core) core | run ;;
  gate) gate | run ;;
  resnet) resnet | run ;;
  analysis) analysis ;;
  all) { core; gate; resnet; } | run; analysis ;;
esac
echo "V6 $PART DONE"
