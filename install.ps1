# Dimensional Light installer (Windows).  Creates .venv and installs dependencies.
# NVIDIA RTX 50-series (Blackwell) needs the CUDA 12.8 PyTorch build, used below.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install numpy opencv-python pillow tifffile transformers diffusers accelerate scipy

Write-Host "`nPre-downloading the Depth Anything V2 Large model (~1.3 GB)..."
& $py -c "from transformers import AutoModelForDepthEstimation as M; M.from_pretrained('depth-anything/Depth-Anything-V2-Large-hf'); print('model ready')"
Write-Host "`nPre-downloading Apple Depth Pro (~1.9 GB) and Marigold for Ultra (~2.5 GB)..."
& $py -c "from transformers import AutoImageProcessor as P, AutoModelForDepthEstimation as M; P.from_pretrained('apple/DepthPro-hf'); M.from_pretrained('apple/DepthPro-hf'); print('depth pro ready')"
& $py -c "import torch; from diffusers import MarigoldDepthPipeline as D; D.from_pretrained('prs-eth/marigold-depth-v1-1', variant='fp16', torch_dtype=torch.float16); print('marigold ready')"

Write-Host "`nDone. In Lightroom Classic: File > Plug-in Manager > Add > select DimensionalLight.lrdevplugin"
Write-Host "Test the app without Lightroom:  .\.venv\Scripts\python.exe app\app.py --demo"
