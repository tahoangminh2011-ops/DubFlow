from __future__ import annotations

import math, os, shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .detector import TextActivityDetector
from .fast_change import FastChangeDetector
from .models import CandidateWindow, VideoMetadata


@dataclass
class Sample:
    timestamp: float
    frame_index: int
    image: np.ndarray
    preferred_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class ScanTracePoint:
    timestamp: float
    frame_index: int
    roi_presence: bool
    global_presence: bool
    roi_textlike: float
    global_textlike: float
    roi_changed: bool
    global_changed: bool
    roi_bbox: Optional[Tuple[int, int, int, int]]


class AdaptiveFrameSampler:
    """Ultra-light high-recall temporal first pass.

    The scan is intentionally independent from OCR. It samples at 30 FPS by default so a
    ~0.1 s subtitle has multiple chances to be observed, while each frame is reduced to a
    tiny grayscale image. ROI analysis is performed every frame. Global fallback uses a
    cheaper whole-frame change gate and invokes the richer global text gate only on suspicious
    frames. The trace is retained for frame-local boundary refinement without another full
    video decode pass.
    """

    def __init__(self, ffmpeg: str = "ffmpeg", scan_fps: float = 20.0, width: int = 224,
                 workers: int = 1, diff_threshold: float = 0.018, global_textlike_threshold: float = 0.0075,
                 global_line_density_threshold: float = 0.05, hwaccel: str = "auto"):
        self.ffmpeg = ffmpeg
        self.scan_fps = max(24.0, float(scan_fps))
        self.width = max(160, int(width))
        self.workers = max(1, int(workers))
        self.detector = TextActivityDetector(threshold=diff_threshold)
        self.global_textlike_threshold = max(0.004, float(global_textlike_threshold))
        self.global_line_density_threshold = max(0.035, float(global_line_density_threshold))
        self.fast_detector = FastChangeDetector(threshold=diff_threshold, down_w=self.width)
        self.hwaccel = str(hwaccel or "auto").lower()
        self._cuda_filter_checked = False
        self._cuda_scale_available = False
        self.last_trace: List[ScanTracePoint] = []

    def _ensure_cuda_scale_capability(self) -> bool:
        if self._cuda_filter_checked:
            return self._cuda_scale_available
        self._cuda_filter_checked = True
        if self.hwaccel in {"none", "off", "0"}:
            return False
        try:
            import shutil as _shutil
            if not _shutil.which("nvidia-smi"):
                return False
            out = subprocess.run(
                [self.ffmpeg, "-hide_banner", "-filters"],
                capture_output=True, text=True, timeout=5
            ).stdout.lower()
            self._cuda_scale_available = ("scale_cuda" in out) or ("scale_npp" in out)
        except Exception:
            self._cuda_scale_available = False
        return self._cuda_scale_available

    def _run_chunk(self, path: Path, start: float, end: float, source_width: int, source_height: int) -> Iterator[Tuple[float, np.ndarray]]:
        dur = max(0.01, end - start)
        sh = max(2, int(round(float(source_height) / max(1, source_width) * self.width)))
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", os.getenv("OCR_FFMPEG_THREADS", "0")]
        hw = self.hwaccel
        if hw == "auto":
            # Prefer NVIDIA decode when the local FFmpeg build exposes it; otherwise stay on the
            # stable software path. Importantly, AUTO now means an actual detection attempt rather
            # than silently behaving like NONE.
            try:
                help_out = subprocess.run([self.ffmpeg, "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=3).stdout.lower()
                hw = "cuda" if "cuda" in help_out and shutil.which("nvidia-smi") else "none"
            except Exception:
                hw = "none"
        cuda_scale = bool(hw == "cuda" and self._ensure_cuda_scale_capability())
        if cuda_scale:
            # NVDEC + GPU scaling + tiny host transfer. The final fps filter runs after the
            # downscale, so only ~224px-wide frames cross the PCIe boundary.
            vf = f"scale_cuda={self.width}:{sh}:interp_algo=bilinear,hwdownload,format=gray,fps={self.scan_fps:g}"
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        else:
            vf = f"fps={self.scan_fps:g},scale={self.width}:{sh}:flags=fast_bilinear,format=gray"
            if hw not in {"", "none", "off", "0"}:
                cmd += ["-hwaccel", hw]
        cmd += [
            "-ss", f"{start:.4f}", "-i", str(path), "-t", f"{dur:.4f}",
            "-vf", vf, "-an", "-sn", "-dn", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1024 * 1024)
        frame_bytes = self.width * sh
        idx = 0
        try:
            while True:
                raw = p.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                yield start + idx / self.scan_fps, np.frombuffer(raw, dtype=np.uint8).reshape((sh, self.width))
                idx += 1
        finally:
            try:
                p.stdout.close()
            except Exception:
                pass
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()

    @staticmethod
    def _roi_px(region: Dict[str, float], w: int, h: int):
        x = float(region.get("x", 0)); y = float(region.get("y", 0))
        rw = float(region.get("w", 100)); rh = float(region.get("h", 100))
        x0 = max(0, min(max(0, w - 1), int(round(x * w / 100))))
        y0 = max(0, min(max(0, h - 1), int(round(y * h / 100))))
        x1 = max(x0 + 1, min(w, int(round((x + rw) * w / 100))))
        y1 = max(y0 + 1, min(h, int(round((y + rh) * h / 100))))
        return x0, y0, x1, y1

    @staticmethod
    def _source_bbox_from_small(b: Optional[Tuple[int, int, int, int]], sw: int, sh: int, meta: VideoMetadata):
        if not b:
            return None
        sx = meta.width / max(1, sw); sy = meta.height / max(1, sh)
        x0, y0, x1, y1 = b
        return int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))

    @staticmethod
    def _cheap_global(previous: Optional[np.ndarray], current: np.ndarray):
        """Whole-frame fallback with no text morphology/connected-components."""
        if previous is None:
            return 1.0, 1.0, 1.0
        a = cv2.resize(previous, (96, 54), interpolation=cv2.INTER_AREA)
        b = cv2.resize(current, (96, 54), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(a, b)
        change = diff >= 14
        diff_ratio = float(change.mean())
        rows = change.mean(axis=1)
        mean_rows = float(rows.mean())
        if mean_rows <= 1e-6:
            return diff_ratio, 0.0, 0.0
        bh = max(3, int(round(b.shape[0] * 0.11)))
        sm = np.convolve(rows, np.ones(bh, np.float32), mode="valid") / float(bh)
        concentration = float(sm.max()) / max(mean_rows, 1e-6)
        idx = int(np.argmax(rows)); y0 = max(0, min(b.shape[0] - bh, idx - bh // 2))
        coverage = float(np.mean(np.any(change[y0:y0 + bh] > 0, axis=0)))
        return diff_ratio, concentration, coverage

    @staticmethod
    def _ultra_signal(previous: Optional[np.ndarray], current: np.ndarray):
        """O(1)ish temporal gate per tiny frame; no morphology/gradient/CC.

        The gate runs on every high-cadence sample. Expensive FastChangeDetector analysis is
        only invoked when this signal says the frame contains a concentrated horizontal change.
        This preserves 30 FPS temporal coverage for very short subtitles without paying the full
        detector cost on every stable frame.
        """
        cur = current if current.ndim == 2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        if previous is None:
            h, w = cur.shape[:2]
            return {"diff_ratio": 1.0, "band_concentration": 1.0, "band_coverage": 1.0,
                    "event": True, "scene_cut": False}
        prev = previous if previous.ndim == 2 else cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        if prev.shape != cur.shape:
            prev = cv2.resize(prev, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_AREA)
        d = cv2.absdiff(prev, cur)
        change = (d >= 12)
        diff_ratio = float(change.mean())
        h, w = change.shape[:2]
        if h < 8 or w < 16:
            return {"diff_ratio": diff_ratio, "band_concentration": 1.0,
                    "band_coverage": float(change.mean()), "event": diff_ratio >= 0.002,
                    "scene_cut": diff_ratio >= 0.45}
        rows = change.mean(axis=1)
        best = 0.0
        best_y0 = 0
        best_bh = max(2, int(round(h * 0.09)))
        for frac in (0.05, 0.08, 0.11, 0.16):
            bh = max(2, min(h, int(round(h * frac))))
            if bh >= h:
                score = float(rows.mean()); y0 = 0
            else:
                sm = np.convolve(rows, np.ones(bh, np.float32), mode="valid") / float(bh)
                y0 = int(np.argmax(sm)); score = float(sm[y0])
            if score > best:
                best = score; best_y0 = y0; best_bh = bh
        mean_row = float(rows.mean())
        concentration = best / max(mean_row, 1e-6)
        band = change[best_y0:min(h, best_y0 + best_bh)]
        coverage = float(np.mean(np.any(band, axis=0)))
        # Full-frame scene cuts tend to have high diff but weak horizontal concentration.
        scene_cut = diff_ratio >= 0.35 and concentration < 1.35
        event = bool(
            (diff_ratio >= 0.0018 and concentration >= 1.45 and coverage >= 0.055)
            or (diff_ratio >= 0.007 and concentration >= 1.30 and coverage >= 0.035)
            or (diff_ratio >= 0.020 and coverage >= 0.045)
        ) and not scene_cut
        # Only this stricter signal may force the more expensive ROI/global text detector to
        # run between checkpoints. The broader `event` flag remains diagnostic; using it directly
        # caused the heavy detector to run on almost every frame in fast-moving gameplay footage.
        text_event = bool(
            diff_ratio >= 0.0012 and concentration >= 3.00 and coverage >= 0.120
            and not scene_cut
        )
        return {"diff_ratio": diff_ratio, "band_concentration": concentration,
                "band_coverage": coverage, "event": event, "text_event": text_event,
                "scene_cut": scene_cut}

    def scan(self, path: Path, meta: VideoMetadata, region: Dict[str, float],
             event_cb: Optional[Callable[[Dict], None]] = None,
             progress_cb: Optional[Callable[[int, int, Dict], None]] = None) -> List[CandidateWindow]:
        roi_full = self._roi_px(region, meta.width, meta.height)
        roi_is_full = roi_full == (0, 0, meta.width, meta.height)
        windows: List[CandidateWindow] = []
        self.last_trace = []
        active_start: Optional[float] = None
        last_strong_event: Optional[float] = None
        last_emit: Optional[float] = None
        preferred_bbox: Optional[Tuple[int, int, int, int]] = None
        active_region_id = "roi"
        prev_small: Optional[np.ndarray] = None
        prev_fr = None
        prev_fg = None
        prev_roi_presence = False
        prev_global_presence = False
        roi_present_streak = roi_absent_streak = 0
        global_present_streak = global_absent_streak = 0
        last_replacement_event = -1e9
        def nearest_source_frame(ts: float) -> int:
            pts = meta.pts
            if not pts:
                return max(0, int(round(float(ts) * max(meta.fps, 1e-9))))
            lo, hi = 0, len(pts) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if pts[mid] < ts:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > 0 and abs(pts[lo-1] - ts) <= abs(pts[lo] - ts):
                return lo - 1
            return lo
        def source_time(ts: float) -> float:
            if meta.frame_count and meta.pts:
                return float(meta.time_for_frame(nearest_source_frame(float(ts))))
            return float(ts)

        total = max(1, int(math.ceil(meta.duration * self.scan_fps)))
        done = 0
        chunk = max(120.0, meta.duration / max(1, self.workers))
        starts = [i * chunk for i in range(max(1, int(math.ceil(meta.duration / chunk))))]
        iter_chunks = [(0.0, meta.duration)] if self.workers <= 1 else [
            (st, min(meta.duration, st + chunk)) for st in starts
        ]

        for st, en in iter_chunks:
            for ts, small in self._run_chunk(path, st, en, meta.width, meta.height):
                done += 1
                sh, sw = small.shape[:2]
                rx0, ry0, rx1, ry1 = self._roi_px(region, sw, sh)
                roi_img = small[ry0:ry1, rx0:rx1]

                signal = self._ultra_signal(prev_small, small)
                # IMPORTANT: compare ROI pixels with the previous ROI, never with the previous
                # full frame. The old code resized a full-frame image to the ROI's shape; this
                # made diff/text-presence metrics geometrically invalid whenever the ROI was not full.
                prev_roi_img = None
                if prev_small is not None:
                    prev_roi_img = prev_small[ry0:ry1, rx0:rx1]
                    if prev_roi_img.size == 0:
                        prev_roi_img = None
                roi_due = (done == 1) or signal.get("text_event", False) or (done % max(1, int(round(self.scan_fps * 0.20))) == 0)
                # Monitor full-frame text independently even when ROI looks quiet. Previously global
                # detection only ran on temporal events; a misplaced ROI could therefore suppress all
                # candidates for an entire video. At 320px this full-frame gate is still lightweight.
                global_interval = max(1, int(round(self.scan_fps * 0.35)))
                global_due = roi_is_full or (done == 1) or signal.get("text_event", False) or (done % global_interval == 0)
                if roi_due:
                    fr = self.fast_detector.analyze(prev_roi_img, roi_img)
                    prev_fr = fr
                else:
                    fr = prev_fr
                if roi_is_full:
                    fg = fr
                    prev_fg = fg
                elif global_due:
                    fg = self.fast_detector.analyze(prev_small, small)
                    prev_fg = fg
                else:
                    fg = prev_fg
                if fr is None or fg is None:
                    from .fast_change import FastActivity
                    blank = FastActivity(False, 0.0, float(signal["diff_ratio"]), 0.0, 0.0, 0.0,
                                         None, 0.0, 0.0, False, False, False, False, 1.0, 0.0)
                    fr = fr or blank
                    fg = fg or blank

                raw_roi_presence = bool(fr.presence)
                raw_global_presence = bool(fg.presence)
                roi_present_streak = roi_present_streak + 1 if raw_roi_presence else 0
                roi_absent_streak = roi_absent_streak + 1 if not raw_roi_presence else 0
                global_present_streak = global_present_streak + 1 if raw_global_presence else 0
                global_absent_streak = global_absent_streak + 1 if not raw_global_presence else 0

                roi_prev_state = prev_roi_presence
                global_prev_state = prev_global_presence
                strong_roi_appearance = (not roi_prev_state and raw_roi_presence and fr.textlike >= 0.006 and fr.textlike_delta >= 0.008 and fr.diff_ratio >= 0.007) or (signal.get("text_event", False) and not roi_prev_state and raw_roi_presence and roi_present_streak >= 1)
                strong_global_appearance = (not global_prev_state and raw_global_presence and fg.textlike >= 0.006 and fg.textlike_delta >= 0.008 and fg.diff_ratio >= 0.007) or (signal.get("text_event", False) and not global_prev_state and raw_global_presence and (global_present_streak >= 1 or global_due))
                strong_roi_disappearance = (roi_prev_state and not raw_roi_presence and fr.textlike <= 0.0028 and fr.textlike_delta >= 0.008 and fr.diff_ratio >= 0.007)
                strong_global_disappearance = (global_prev_state and not raw_global_presence and fg.textlike <= 0.0028 and fg.textlike_delta >= 0.008 and fg.diff_ratio >= 0.007)

                appearance = bool(
                    ((not roi_prev_state) and roi_present_streak >= 2) or
                    ((not global_prev_state) and global_present_streak >= 2) or
                    strong_roi_appearance or strong_global_appearance
                )

                raw_replacement = bool(
                    roi_prev_state and raw_roi_presence and fr.replacement
                    and fr.diff_ratio >= max(0.018, self.fast_detector.threshold * 1.00)
                    and fr.textlike_delta >= max(0.0045, self.fast_detector.threshold * 0.25)
                    and fr.line_density >= 0.055
                ) or bool(
                    global_prev_state and raw_global_presence and fg.replacement
                    and fg.diff_ratio >= max(0.025, self.fast_detector.threshold * 1.30)
                    and fg.textlike_delta >= 0.006
                    and fg.line_density >= max(0.055, self.global_line_density_threshold)
                )
                replacement = bool(raw_replacement and (ts - last_replacement_event >= 0.65))
                if replacement:
                    last_replacement_event = ts
                    # IMPORTANT: a replacement is an event *inside the currently active subtitle run*.
                    # Do not turn every visual replacement into a separate OCR window: that can create
                    # hundreds/thousands of tiny windows on a long video and is the main source of the
                    # previous 7+ minute verification phase. Interior recognition samples the active
                    # run and can promote a sustained lexical mismatch to one full detector pass.

                roi_presence = roi_prev_state
                global_presence = global_prev_state
                if not roi_prev_state and roi_present_streak >= 2:
                    roi_presence = True
                if roi_prev_state and roi_absent_streak >= 3:
                    roi_presence = False
                if not global_prev_state and global_present_streak >= 2:
                    global_presence = True
                if global_prev_state and global_absent_streak >= 3:
                    global_presence = False

                any_prev_presence = bool(roi_prev_state or global_prev_state)
                all_regions_inactive = bool(not roi_presence and not global_presence)
                disappearance = bool(
                    active_start is not None and any_prev_presence and all_regions_inactive and
                    (roi_absent_streak >= 2 or global_absent_streak >= 2 or
                     strong_roi_disappearance or strong_global_disappearance)
                )

                if appearance:
                    if active_start is None:
                        # Store the observed transition time here. The 90 ms scan margin is
                        # applied exactly once when CandidateWindow is emitted; applying it here
                        # too shifted cue starts early by 180 ms in the previous implementation.
                        active_start = max(0.0, source_time(ts))
                        # Select provenance from the raw detector observations that triggered this
                        # appearance. Smoothed state deliberately lags by two observations, so using
                        # roi_presence/global_presence here could mislabel a global-only subtitle as ROI.
                        active_region_id = "global" if (
                            raw_global_presence and (
                                not raw_roi_presence
                                or float(fg.textlike) >= float(fr.textlike) * 1.20
                                or float(fg.line_density) >= float(fr.line_density) * 1.20
                            )
                        ) else "roi"
                    last_strong_event = source_time(ts)
                    heavy_roi = self.detector.analyze(None, roi_img) if roi_presence else None
                    heavy_global = self.detector.analyze(None, small) if (
                        not roi_is_full and global_presence
                    ) else None
                    rb = self._source_bbox_from_small(
                        (heavy_roi.bbox[0] + rx0, heavy_roi.bbox[1] + ry0,
                         heavy_roi.bbox[2] + rx0, heavy_roi.bbox[3] + ry0), sw, sh, meta
                    ) if heavy_roi is not None and heavy_roi.bbox else None
                    gb = self._source_bbox_from_small(heavy_global.bbox, sw, sh, meta) if heavy_global and heavy_global.bbox else None
                    if active_region_id == "global":
                        preferred_bbox = gb
                    else:
                        preferred_bbox = rb or (gb if roi_is_full else None) or preferred_bbox
                elif replacement:
                    last_strong_event = source_time(ts)
                    if preferred_bbox is None and active_region_id == "roi" and fr.bbox:
                        preferred_bbox = self._source_bbox_from_small(
                            (fr.bbox[0] + rx0, fr.bbox[1] + ry0, fr.bbox[2] + rx0, fr.bbox[3] + ry0), sw, sh, meta
                        )
                elif disappearance and active_start is not None:
                    et = source_time(ts)
                    windows.append(CandidateWindow(
                        max(0.0, active_start - 0.09), min(meta.duration, et + 0.09),
                        "text_disappearance", preferred_bbox, 4, active_region_id,
                    ))
                    active_start = None; last_strong_event = None; preferred_bbox = None; active_region_id = "roi"

                if active_start is not None and not roi_presence and not global_presence and last_strong_event is not None and source_time(ts) - last_strong_event >= 0.18:
                    et = source_time(ts)
                    windows.append(CandidateWindow(
                        max(0.0, active_start - 0.09), min(meta.duration, et + 0.07),
                        "text_clear", preferred_bbox, 2, active_region_id,
                    ))
                    active_start = None; last_strong_event = None; preferred_bbox = None; active_region_id = "roi"

                # Keep one compact trace point per fast frame. This is used later for frame-local
                # boundary bracketing so the engine does not decode the whole video again.
                # Never map VFR frames through timestamp * FPS.  Keep the actual source
                # frame index selected from the video PTS table.
                scan_frame = nearest_source_frame(float(ts))
                self.last_trace.append(ScanTracePoint(
                    timestamp=float(source_time(ts)), frame_index=int(scan_frame),
                    roi_presence=bool(roi_presence), global_presence=bool(global_presence),
                    roi_textlike=float(fr.textlike), global_textlike=float(fg.textlike),
                    roi_changed=bool(fr.changed or appearance or replacement),
                    global_changed=bool(fg.changed), roi_bbox=fr.bbox,
                ))

                event = bool(appearance or replacement or disappearance)
                event_ts = source_time(ts)
                if event_cb and event and (last_emit is None or event_ts - last_emit >= 0.35):
                    event_cb({
                        "phase": "scan", "timestamp": round(event_ts, 4), "samples_scanned": done,
                        "roi_change": round(fr.score, 4), "global_change": round(fg.score, 4),
                        "roi_textlike": round(fr.textlike, 4), "global_textlike": round(fg.textlike, 4),
                        "textlike_delta": round(max(fr.textlike_delta, fg.textlike_delta), 5),
                        "roi_line_density": round(fr.line_density, 4),
                        "global_line_density": round(fg.line_density, 4),
                        "bbox": preferred_bbox, "fallback": bool(global_presence and not roi_presence),
                    })
                    last_emit = event_ts

                prev_small = small
                prev_roi_presence = roi_presence
                prev_global_presence = global_presence
                if progress_cb and (done == total or done % max(1, int(self.scan_fps * 2)) == 0):
                    progress_cb(min(done, total), total, {"duration_scanned": round(min(meta.duration, done / self.scan_fps), 2), "samples_scanned": done})

        if active_start is not None:
            windows.append(CandidateWindow(max(0.0, active_start - 0.09), meta.duration, "text_transition", preferred_bbox, 3, active_region_id))

        windows = self._merge(windows)
        if event_cb:
            event_cb({"phase": "scan_done", "windows": len(windows), "scan_fps": self.scan_fps})
        return windows

    @staticmethod
    def _merge(windows: List[CandidateWindow]) -> List[CandidateWindow]:
        if not windows:
            return []
        out: List[CandidateWindow] = []
        # High-priority replacement events stay independent from broad active windows. Only two
        # replacement events that overlap are merged, preventing false segmentation while retaining
        # every genuine lexical transition for full OCR.
        replacements: List[CandidateWindow] = []
        broad: List[CandidateWindow] = []
        for w in windows:
            (replacements if w.reason == "text_replacement" else broad).append(w)

        def merge_group(items: List[CandidateWindow], gap: float):
            ans: List[CandidateWindow] = []
            for w in sorted(items, key=lambda x: (x.start_time, x.end_time)):
                if ans and w.region_id == ans[-1].region_id and w.start_time <= ans[-1].end_time + gap:
                    p = ans[-1]
                    p.end_time = max(p.end_time, w.end_time)
                    p.start_time = min(p.start_time, w.start_time)
                    p.priority = max(p.priority, w.priority)
                    if p.preferred_bbox is None:
                        p.preferred_bbox = w.preferred_bbox
                else:
                    ans.append(CandidateWindow(
                        w.start_time, w.end_time, w.reason, w.preferred_bbox, w.priority, w.region_id
                    ))
            return ans

        broad = merge_group(broad, gap=0.02)
        replacements = merge_group(replacements, gap=0.05)
        all_windows = broad + replacements
        all_windows.sort(key=lambda x: (x.start_time, -x.priority, x.end_time))
        # Never merge a high-priority replacement into a broad window. This is the key that keeps
        # each lexical transition independently recognizable by the temporal OCR stage.
        return all_windows


class AdaptiveSampler:
    """Compatibility planner retained only for legacy callers/tests.

    The V4 engine does not execute this planner. It uses TemporalOCRProcessor instead. Keeping this
    tiny facade prevents third-party imports from breaking while guaranteeing no dense full-video
    OCR grid can be generated accidentally.
    """
    def __init__(self, recovery_seconds: float = 1.5, burst_fps: float = 30.0,
                 boundary_seconds: float = 0.15, interior_fps: float = 0.5):
        self.recovery_seconds = max(0.5, float(recovery_seconds))
        self.burst_fps = max(1.0, float(burst_fps))
        self.boundary_seconds = max(0.03, float(boundary_seconds))
        self.interior_fps = max(0.0, float(interior_fps))

    def plan(self, windows, meta):
        out=[]
        for w in windows or []:
            s=float(w.start_time); e=min(float(meta.duration), float(w.end_time))
            if e <= s: continue
            pts=[s + min(self.boundary_seconds, max(0.01,(e-s)*0.25)),
                 max(s+0.01, e-min(self.boundary_seconds, max(0.01,(e-s)*0.25)))]
            if w.reason == "text_replacement":
                pts=[(s+e)*0.5]
            elif e-s >= 2.0:
                pts.append((s+e)*0.5)
            for t in sorted(set(round(max(s,min(e,x)),5) for x in pts)):
                out.append((t, "burst" if w.priority>=4 else "recovery"))
        return out
