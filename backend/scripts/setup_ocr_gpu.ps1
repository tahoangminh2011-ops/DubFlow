#requires -Version 5.1
$ErrorActionPreference = "Stop"

Write-Host "=== VideoOCR V5 Turbo - NVIDIA OCR setup ===" -ForegroundColor Cyan

$Python = (Get-Command python -ErrorAction Stop).Source
Write-Host "Python: $Python"

Write-Host "`n[1/4] Removing conflicting ONNX Runtime CPU package..." -ForegroundColor Yellow
& $Python -m pip uninstall -y onnxruntime onnxruntime-gpu

Write-Host "`n[2/4] Installing ONNX Runtime GPU + CUDA/cuDNN runtime wheels..." -ForegroundColor Yellow
& $Python -m pip install --upgrade `
  "onnxruntime-gpu==1.20.1" `
  "nvidia-cuda-runtime-cu12" `
  "nvidia-cublas-cu12" `
  "nvidia-cudnn-cu12"

Write-Host "`n[3/4] Verifying NVIDIA driver..." -ForegroundColor Yellow
$nvs = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvs) {
    throw "nvidia-smi không tồn tại. Kiểm tra driver NVIDIA trước khi chạy OCR."
}
& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

Write-Host "`n[4/4] Running GPU doctor..." -ForegroundColor Yellow
& $Python scripts\ocr_gpu_doctor.py

Write-Host "`nDONE. Hãy đóng mọi Python/Uvicorn đang chạy và khởi động lại backend." -ForegroundColor Green
