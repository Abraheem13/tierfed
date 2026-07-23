#!/usr/bin/env bash
# One-shot environment setup. Usage:  bash scripts/setup_env.sh [conda|venv]
set -euo pipefail
MODE="${1:-venv}"
PYVER=3.11

if [ "$MODE" = "conda" ]; then
  conda create -y -n nfl python=$PYVER
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate nfl
else
  python$PYVER -m venv .venv || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade pip wheel

# CUDA 12.1 build; change cu121 -> cu118 for older drivers, or drop the
# index-url entirely for a CPU-only install.
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          f"| {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
PY

echo
echo "Environment ready. Verify with:  python -m pytest tests/ -q"
