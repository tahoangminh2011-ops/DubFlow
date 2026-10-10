from __future__ import annotations
import os, time, subprocess
from dataclasses import dataclass
from typing import Any, Dict
try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

@dataclass
class BenchmarkSnapshot:
    video_duration: float
    total_seconds: float
    scan_seconds: float
    ocr_seconds: float
    timeline_seconds: float
    decode_fps: float
    ocr_fps: float
    cpu_percent: float
    ram_mb: float
    vram_mb: float | None
    gpu_util_percent: float | None
    estimated_60m_seconds: float

class BenchmarkManager:
    def __init__(self):
        self.started = time.monotonic()
        self._cpu: list[float] = []
        self._ram: list[float] = []
        self._gpu_util: list[float] = []
        self._vram: list[float] = []

    def cpu(self):
        v = psutil.cpu_percent(interval=None) if psutil else 0.0
        self._cpu.append(float(v))
        if psutil:
            try: self._ram.append(psutil.Process(os.getpid()).memory_info().rss / 1048576)
            except Exception: pass
        self._sample_nvidia()
        return v

    def _sample_nvidia(self):
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=2, check=True)
            first = (p.stdout.splitlines() or [""])[0]
            util, mem = [x.strip() for x in first.split(",")[:2]]
            self._gpu_util.append(float(util)); self._vram.append(float(mem))
        except Exception: pass

    @staticmethod
    def duration_matrix(observed: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
        targets = (10, 30, 60, 120)
        out = {}
        for minutes in targets:
            item = observed.get(minutes) or observed.get(float(minutes))
            out[str(minutes)] = item if item is not None else {"status": "not_run"}
        return out

    def report(self, video_duration, scan_seconds, ocr_seconds, timeline_seconds, frames_scanned, ocr_frames) -> Dict[str, Any]:
        total = max(0.001, time.monotonic() - self.started)
        current_ram = (psutil.Process(os.getpid()).memory_info().rss / 1048576) if psutil else 0
        ram_peak = max(self._ram, default=current_ram)
        gpu_util = sum(self._gpu_util) / len(self._gpu_util) if self._gpu_util else None
        vram = max(self._vram) if self._vram else None
        est60 = total * (3600 / max(.001, float(video_duration)))
        return {
            "video_duration_seconds": float(video_duration),
            "scan_seconds": round(scan_seconds, 3),
            "ocr_seconds": round(ocr_seconds, 3),
            "timeline_seconds": round(timeline_seconds, 3),
            "total_seconds": round(total, 3),
            "decode_fps": round(frames_scanned / max(scan_seconds, .001), 2),
            "ocr_fps": round(ocr_frames / max(ocr_seconds, .001), 2),
            "cpu_percent_avg": round(sum(self._cpu) / max(1, len(self._cpu)), 1),
            "ram_mb_current": round(current_ram, 1),
            "ram_mb_peak": round(ram_peak, 1),
            "gpu_util_percent_avg": round(gpu_util, 1) if gpu_util is not None else None,
            "vram_mb_peak": round(vram, 1) if vram is not None else None,
            "estimated_60min_seconds": round(est60, 1),
            "estimated_60min_minutes": round(est60 / 60, 2),
            "duration_estimates": {str(m): round(total * ((m * 60) / max(.001, float(video_duration))), 1) for m in (10, 30, 60, 120)},
            "benchmark_targets_minutes": [10, 30, 60, 120],
        }
