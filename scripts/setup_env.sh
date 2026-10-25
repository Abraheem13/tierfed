#!/usr/bin/env bash
# One-shot environment setup.  bash scripts/setup_env.sh [venv|conda]
set -euo pipefail
MODE="${1:-venv}"
if [ "$MODE" = "conda" ]; then
  conda create -y -n tierfed python=3.11
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate tierfed
else
  python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python -m pip install --upgrade pip wheel
# RTX 5080 (Blackwell, sm_120) needs a CUDA 12.8+ build of torch.
pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
echo "Now: python -m pytest tests -q"
