$ErrorActionPreference = "Stop"
Write-Host "=== VideoOCR V3 GPU setup ==="
python -c "import sys; print('Python', sys.version)"
try { nvidia-smi } catch { Write-Warning "nvidia-smi not found; CUDA GPU may be unavailable." }
python -m pip uninstall -y onnxruntime 2>$null
python -m pip install --upgrade pip
python -m pip install -r requirements-ocr-gpu.txt
python -c "import onnxruntime as ort; print('ORT', ort.__version__); print('Providers:', ort.get_available_providers()); assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDAExecutionProvider unavailable'"
Write-Host "GPU OCR runtime is ready. Restart the backend before starting a new OCR job."
