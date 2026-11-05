# CPU-only environment for a laptop without an NVIDIA GPU. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_cpu.ps1
# Uses Python 3.12 or 3.11 when the py launcher has them (the pinned versions of
# requirements.txt); with a newer Python it installs current releases instead.
$ErrorActionPreference = "Stop"
$ver = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
  foreach ($v in @("3.12", "3.11")) { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $ver = $v; break } }
}
if (-not (Test-Path .venv)) {
  if ($ver) { & py "-$ver" -m venv .venv }
  elseif (Get-Command python -ErrorAction SilentlyContinue) { python -m venv .venv }
  else { Write-Error "no Python found; install Python 3.12 from python.org (tick 'Add to PATH')" }
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip wheel
$minor = [int](& $py -c "import sys; print(sys.version_info[1])")
if ($minor -le 12) {
  & $py -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
  & $py -m pip install -r requirements.txt
} else {
  & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  & $py -m pip install numpy pandas scikit-learn scipy matplotlib PyYAML pytest ucimlrepo medmnist
}
& $py -c "import torch, sklearn, scipy, numpy, pandas; print('torch', torch.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__)"
& $py -m pytest -q tests\test_audit.py
Write-Host "CPU environment ready"
