from __future__ import annotations

"""Windows NVIDIA CUDA DLL bootstrap for ONNX Runtime / RapidOCR.

This module does not install drivers or modify the system globally. It only discovers
CUDA/cuDNN DLLs already installed in the current Python environment (including NVIDIA
pip wheels) and makes them visible to the current process before ONNX Runtime creates
an execution provider.
"""

import ctypes
import importlib.util
import os
import sys
from pathlib import Path
from typing import List


_DLL_DIR_HANDLES: List[object] = []


def _site_roots() -> List[Path]:
    roots = [Path(sys.prefix)]
    try:
        import site
        roots.extend(Path(x) for x in site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    out = []
    seen = set()
    for root in roots:
        try:
            r = root.resolve()
        except Exception:
            r = root
        if str(r).lower() not in seen:
            seen.add(str(r).lower())
            out.append(r)
    return out


def _candidate_dirs() -> List[Path]:
    dirs: List[Path] = []
    # NVIDIA pip wheels.
    package_names = [
        ("nvidia", "cublas", "bin"),
        ("nvidia", "cudnn", "bin"),
        ("nvidia", "cuda_runtime", "bin"),
        ("nvidia", "cuda_nvrtc", "bin"),
        ("nvidia", "cufft", "bin"),
        ("nvidia", "curand", "bin"),
        ("nvidia", "cusolver", "bin"),
        ("nvidia", "cusparse", "bin"),
    ]
    for root in _site_roots():
        for parts in package_names:
            p = root.joinpath(*parts)
            if p.is_dir():
                dirs.append(p)
        # Some environments install `site-packages\nvidia\...\bin`.
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            for p in nvidia.glob("*/bin"):
                if p.is_dir():
                    dirs.append(p)

    # System CUDA/cuDNN installations.
    for env in ("CUDA_PATH", "CUDA_HOME", "CUDNN_HOME"):
        value = os.getenv(env)
        if value:
            base = Path(value)
            for p in (base / "bin", base / "lib", base):
                if p.is_dir():
                    dirs.append(p)

    # De-duplicate.
    out = []
    seen = set()
    for p in dirs:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def prepare_cuda_dlls(verbose: bool = True) -> List[str]:
    """Expose discovered CUDA/cuDNN DLL directories to the current process."""
    if os.name != "nt":
        return []

    loaded_dirs: List[str] = []
    for directory in _candidate_dirs():
        s = str(directory)
        if s not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
        try:
            handle = os.add_dll_directory(s)
            _DLL_DIR_HANDLES.append(handle)
            loaded_dirs.append(s)
        except Exception:
            # add_dll_directory can fail on older Windows/Python; PATH still helps.
            pass

    # ORT >= 1.21 has an official DLL preloader. Use it when present.
    try:
        import onnxruntime as ort
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                preload(directory="")
                if verbose:
                    print("[OCR GPU] ONNX Runtime preload_dlls(directory='') applied")
            except Exception as exc:
                if verbose:
                    print(f"[OCR GPU] ORT preload_dlls failed: {exc}")
    except Exception:
        pass

    if verbose and loaded_dirs:
        print("[OCR GPU] CUDA DLL directories:")
        for d in loaded_dirs:
            print(f"  - {d}")
    return loaded_dirs


def nvidia_present() -> bool:
    return bool(importlib.util.find_spec("pynvml")) or bool(
        any(os.path.exists(p) for p in (
            os.environ.get("CUDA_PATH", ""),
            os.environ.get("CUDA_HOME", ""),
        ))
    )


def try_load_core() -> dict:
    """Load the core CUDA libraries when available and report what succeeded.

    Loading cublasLt/cudnn before ORT's CUDA provider is useful on Windows because
    the provider DLL otherwise may be unable to resolve these transitive dependencies.
    """
    prepare_cuda_dlls(verbose=False)
    loaded = []
    failed = []
    if os.name != "nt":
        return {"loaded": loaded, "failed": failed}

    names = []
    for pat in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "cudnn64_*.dll"):
        for d in _candidate_dirs():
            names.extend(str(x) for x in d.glob(pat))
    # Prefer unique filenames, load each once.
    seen = set()
    for path in names:
        key = os.path.basename(path).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            ctypes.WinDLL(path)
            loaded.append(path)
        except Exception as exc:
            failed.append((path, str(exc)))
    return {"loaded": loaded, "failed": failed}
