from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Allow direct execution: `python scripts\ocr_gpu_doctor.py` from the project root
# or from any working directory. Python otherwise puts `scripts/` on sys.path
# and cannot resolve the sibling top-level `pipeline/` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.video_ocr.cuda_bootstrap import prepare_cuda_dlls, try_load_core


def modver(name: str):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "installed")
    except Exception as exc:
        return f"NOT_AVAILABLE ({exc.__class__.__name__}: {exc})"


def _nvidia_smi():
    nvs = shutil.which("nvidia-smi")
    if not nvs:
        return None
    try:
        out = subprocess.check_output(
            [nvs, "--query-gpu=name,driver_version,memory.total,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        )
        return out.strip()
    except Exception as exc:
        return f"detected but query failed: {exc}"


def main():
    print("=== VideoOCR V5 Turbo GPU Doctor ===")
    print(f"Python: {sys.version.split()[0]}")
    print("NVIDIA-SMI:", _nvidia_smi() or "NOT FOUND")

    print("\n[1] Preparing CUDA DLL search path...")
    dirs = prepare_cuda_dlls(verbose=True)
    print("DLL directories discovered:", len(dirs))

    core = try_load_core()
    print("Core CUDA DLLs loaded:", len(core.get("loaded", [])))
    if core.get("failed"):
        for item in core["failed"][:10]:
            print("  FAILED:", item[0], "=>", item[1])

    try:
        ort = importlib.import_module("onnxruntime")
        providers = list(ort.get_available_providers())
        print("\n[2] ONNX Runtime:", getattr(ort, "__version__", "unknown"))
        print("Execution providers:", ", ".join(providers) or "none")
        print("CUDAExecutionProvider (available):", "YES" if "CUDAExecutionProvider" in providers else "NO")

        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                preload(directory="")
                print("ORT preload_dlls: OK")
            except Exception as exc:
                print("ORT preload_dlls: FAILED:", exc)
    except Exception as exc:
        print("\n[2] ONNX Runtime: NOT_AVAILABLE", repr(exc))

    for name in ("rapidocr", "paddle", "paddleocr"):
        print(f"{name}: {modver(name)}")

    print("\n[3] Actual ORT CUDA session probe")
    actual_ok = False
    try:
        from pipeline.video_ocr.ocr_provider import RapidPPV5Provider
        probe = RapidPPV5Provider(lang="auto", backend="rapidocr", use_gpu="auto")
        result = probe._probe_actual_cuda_runtime()
        print("Probe model:", result.get("model"))
        print("Session providers:", ", ".join(result.get("providers") or []) or "none")
        print("Actual CUDA session:", "YES" if result.get("ok") else "NO")
        if result.get("error"): print("Probe error:", result.get("error"))
        actual_ok = bool(result.get("ok"))
    except Exception as exc:
        print("Probe failed:", repr(exc))

    print("\n[4] OCR environment")
    print("OCR_USE_GPU:", os.getenv("OCR_USE_GPU", "auto"))
    print("OCR_BACKEND:", os.getenv("OCR_BACKEND", "auto"))
    print("OCR_REQUIRE_GPU_FOR_TARGET:", os.getenv("OCR_REQUIRE_GPU_FOR_TARGET", "1"))

    print("\n[5] Recommendation")
    print("1. NVIDIA GPU + actual ORT session selecting CUDAExecutionProvider => GPU OCR path is verified.")
    print("2. NVIDIA present but CUDA EP cannot load => run scripts\\setup_ocr_gpu.ps1 and restart PowerShell/backend.")
    print("3. Do not keep CPU `onnxruntime` and GPU `onnxruntime-gpu` together.")

if __name__ == "__main__":
    main()
