# Dimensional Light installer (Windows).
# Creates a private Python environment in .venv, installs the dependencies and
# downloads the depth AI models.  Safe to run again: finished steps are skipped.
#
#   Install.bat                         (double-click, recommended)
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-SkipUltra] [-Cpu]
#
#   -SkipUltra  don't download the Ultra model (Marigold, ~2.5 GB); Fast and Pro still work
#   -Cpu        force the CPU build of PyTorch (no NVIDIA GPU; much slower)
param([switch]$SkipUltra, [switch]$Cpu)

# Native tools (pip, python) print warnings on stderr; Windows PowerShell 5.1 can turn
# those into terminating errors under "Stop".  Every step checks its exit code instead.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) {
    Write-Host "`nERROR: $msg" -ForegroundColor Red
    Write-Host "See docs\INSTALL.md (Troubleshooting) for help." -ForegroundColor Yellow
    exit 1
}
function Check($what) { if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit code $LASTEXITCODE)." } }

Write-Host "Dimensional Light installer" -ForegroundColor White
Write-Host "Folder: $PSScriptRoot"

# 1. Python 3.10 - 3.13 --------------------------------------------------------
Step "Looking for Python 3.10 - 3.13"
$python = $null
foreach ($cand in @("py -3.12", "py -3.11", "py -3.13", "py -3.10", "python")) {
    $out = cmd /c "$cand --version 2>&1"
    if ($LASTEXITCODE -eq 0 -and "$out" -match "Python 3\.(1[0-3])\.") { $python = $cand; break }
}
if (-not $python) {
    Fail ("Python 3.10-3.13 was not found. Install it from https://www.python.org/downloads/ " +
          "(tick 'Add python.exe to PATH'), then run this installer again.")
}
Write-Host "Using: $python ($out)"

# 2. Private environment --------------------------------------------------------
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Step "Creating the Python environment (.venv)"
    cmd /c "$python -m venv `"$PSScriptRoot\.venv`""
    Check "Creating the environment"
}
& $py -m pip install --upgrade pip --quiet
Check "Updating pip"

# 3. PyTorch: CUDA build for NVIDIA GPUs, CPU build otherwise ------------------------
$hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($Cpu -or -not $hasNvidia) {
    Step "Installing PyTorch (CPU build; no NVIDIA GPU detected, so this will be slow)"
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    $gpu = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    Step "Installing PyTorch with CUDA 12.8 for: $gpu  (~2.5 GB download)"
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
}
Check "Installing PyTorch"

# 4. Other packages ---------------------------------------------------------------
Step "Installing the other packages"
& $py -m pip install numpy opencv-python pillow tifffile transformers diffusers accelerate scipy
Check "Installing packages"

# 5. Depth AI models (downloaded once into your Hugging Face cache) -------------------
Step "Downloading Depth Anything V2 Large (Fast mode, ~1.3 GB)"
& $py -c "from transformers import AutoModelForDepthEstimation as M; M.from_pretrained('depth-anything/Depth-Anything-V2-Large-hf'); print('ok')"
Check "Downloading Depth Anything V2"

Step "Downloading Apple Depth Pro (Pro mode, default, ~1.9 GB)"
& $py -c "from transformers import AutoImageProcessor as P, AutoModelForDepthEstimation as M; P.from_pretrained('apple/DepthPro-hf'); M.from_pretrained('apple/DepthPro-hf'); print('ok')"
Check "Downloading Depth Pro"

if ($SkipUltra) {
    Write-Host "`nSkipping the Ultra model (-SkipUltra). It downloads automatically the first time you pick Ultra."
} else {
    Step "Downloading Marigold (Ultra mode, ~2.5 GB)"
    & $py -c "import torch; from diffusers import MarigoldDepthPipeline as D; D.from_pretrained('prs-eth/marigold-depth-v1-1', variant='fp16', torch_dtype=torch.float16); print('ok')"
    Check "Downloading Marigold"
}

# 6. Quick self-check ---------------------------------------------------------------
Step "Checking the installation"
& $py -c "import torch, cv2, transformers; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')"
Check "Self-check"

$plugin = Join-Path $PSScriptRoot "DimensionalLight.lrdevplugin"
Write-Host "`nInstalled successfully." -ForegroundColor Green
Write-Host "Last step, in Lightroom Classic:"
Write-Host "  File > Plug-in Manager > Add > select this folder:"
Write-Host "  $plugin" -ForegroundColor White
Write-Host "Then select a photo and use File > Plug-in Extras > Dimensional Light..."
Write-Host "`nTry it without Lightroom:  .venv\Scripts\python.exe app\app.py --demo"
