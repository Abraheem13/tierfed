# Windows environment setup (no WSL). Run from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# Creates .venv with a CUDA 12.8 build of torch (RTX 5080 / Blackwell needs cu128).
$ErrorActionPreference = "Stop"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Write-Error "python not found on PATH; install Python 3.11 or 3.12 from python.org first" }
python -c "import sys; assert sys.version_info >= (3,10), 'need Python >= 3.10'"
if (-not (Test-Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
& .\.venv\Scripts\python.exe -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE')"
Write-Host "Environment ready. Activate with:  .\.venv\Scripts\Activate.ps1"
