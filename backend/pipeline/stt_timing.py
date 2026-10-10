"""Visual subtitle-boundary timing for STT projects.

STT remains the only source of subtitle *text*.  This module only detects when the
burned-in subtitle region turns on/off and uses those visual intervals to replace
Whisper/Groq segment boundaries when a reliable match is found.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _region_pixels(cap, region: Optional[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    width = int(cap.get(3) or 0)
    height = int(cap.get(4) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Không đọc được kích thước video để căn timing STT")
    r = dict(region or {})
    try:
        x = float(r.get("x", 0)); y = float(r.get("y", 72))
        w = float(r.get("w", r.get("width", 100)))
        h = float(r.get("h", r.get("height", 25)))
    except (TypeError, ValueError):
        x, y, w, h = 0.0, 72.0, 100.0, 25.0
    x = _clamp(x, 0, 100); y = _clamp(y, 0, 100)
    w = _clamp(w, 1, 100 - x); h = _clamp(h, 1, 100 - y)
    x1 = int(round(width * x / 100)); y1 = int(round(height * y / 100))
    x2 = int(round(width * (x + w) / 100)); y2 = int(round(height * (y + h) / 100))
    return x1, y1, min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))


def _frame_metric(frame, bounds) -> Tuple[float, float, float]:
    """Return (bright_text_ratio, edge_ratio, subtitle_activity)."""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = bounds
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, 0.0, 0.0

    h, w = crop.shape[:2]
    scale = min(1.0, 640.0 / max(w, 1))
    if scale < 0.98:
        crop = cv2.resize(crop, (max(96, int(w * scale)), max(40, int(h * scale))), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]

    # White / near-white subtitle strokes with dark outlines.
    bright = ((gray >= 180) & (sat <= 170)) | (gray >= 215)
    bright = bright.astype(np.uint8) * 255
    # Remove isolated background speckles; keep text strokes.
    kernel = np.ones((2, 2), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    bright_ratio = float((bright > 0).mean())

    edges = cv2.Canny(gray, 60, 170)
    edge_ratio = float((edges > 0).mean())

    # Subtitle activity is dominated by sparse bright/outlined strokes.  The
    # edge term is only a backup for colored/outlined subtitles.
    activity = min(1.0, bright_ratio * 5.5 + edge_ratio * 0.22)
    return bright_ratio, edge_ratio, activity


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(__import__("numpy").percentile(values, q))


def _smooth(values: List[float], radius: int = 2) -> List[float]:
    if radius <= 0 or len(values) <= 2:
        return list(values)
    out: List[float] = []
    n = len(values)
    for i in range(n):
        a = max(0, i - radius); b = min(n, i + radius + 1)
        out.append(sum(values[a:b]) / max(1, b - a))
    return out


def _detect_visual_intervals(samples: List[Tuple[float, float, float, float]], fps: float) -> List[Dict[str, float]]:
    """Convert visual activity samples into non-overlapping subtitle ON/OFF intervals."""
    if len(samples) < 5:
        return []
    signals = _smooth([s[3] for s in samples], max(1, round(fps * 0.08)))
    baseline = _percentile(signals, 20)
    p60 = _percentile(signals, 60)
    p85 = _percentile(signals, 85)
    spread = max(0.001, p85 - baseline)
    min_activity = float(getattr(config, "STT_VISUAL_MIN_ACTIVITY", 0.020))
    on_thr = max(min_activity, baseline + max(0.008, spread * 0.32))
    off_thr = max(min_activity * 0.48, baseline + spread * 0.12)

    on_needed = max(2, int(round(fps * float(getattr(config, "STT_VISUAL_MIN_STATE_SECONDS", 0.12)))))
    off_needed = max(2, int(round(fps * float(getattr(config, "STT_VISUAL_MIN_STATE_SECONDS", 0.12)))))

    def transition_time(i0: int, i1: int, threshold: float) -> float:
        a_t, b_t = samples[i0][0], samples[i1][0]
        a_v, b_v = signals[i0], signals[i1]
        if abs(b_v - a_v) < 1e-8:
            return b_t
        ratio = _clamp((threshold - a_v) / (b_v - a_v), 0.0, 1.0)
        return a_t + (b_t - a_t) * ratio

    active = False
    on_count = 0
    off_count = 0
    start_idx = None
    intervals: List[Dict[str, float]] = []

    for i, value in enumerate(signals):
        if not active:
            if value >= on_thr:
                on_count += 1
            else:
                on_count = 0
            if on_count >= on_needed:
                j = max(0, i - on_needed + 1)
                prev = max(0, j - 1)
                start_time = transition_time(prev, j, on_thr)
                active = True
                start_idx = j
                off_count = 0
                on_count = 0
        else:
            if value <= off_thr:
                off_count += 1
            else:
                off_count = 0
            if off_count >= off_needed:
                j = max(0, i - off_needed + 1)
                prev = max(0, j - 1)
                end_time = transition_time(prev, j, off_thr)
                if start_idx is not None and end_time > start_time + 0.12:
                    intervals.append({"start": float(start_time), "end": float(end_time)})
                active = False
                start_idx = None
                off_count = 0
                on_count = 0

    if active and start_idx is not None:
        end_time = float(samples[-1][0])
        if end_time > start_time + 0.12:
            intervals.append({"start": float(start_time), "end": end_time})

    # Small gaps inside the same subtitle animation are merged.
    merge_gap = float(getattr(config, "STT_VISUAL_EVENT_MERGE_GAP", 0.12))
    merged: List[Dict[str, float]] = []
    for iv in intervals:
        if merged and iv["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], iv["end"])
        else:
            merged.append(dict(iv))
    return merged


def _match_visual_interval(seg: Dict[str, Any], intervals: List[Dict[str, float]], used: set, window: float) -> Optional[int]:
    s = float(seg.get("start", 0) or 0)
    e = float(seg.get("end", s + 0.05) or s + 0.05)
    if e <= s:
        return None
    sc = (s + e) / 2
    sd = e - s
    best_idx = None
    best_score = 0.55
    for i, iv in enumerate(intervals):
        if i in used:
            continue
        a, b = iv["start"], iv["end"]
        c = (a + b) / 2
        vd = max(0.05, b - a)
        center_delta = abs(c - sc)
        if center_delta > window:
            continue
        overlap = max(0.0, min(e, b) - max(s, a))
        overlap_ratio = overlap / max(0.05, min(sd, vd))
        start_delta = abs(a - s)
        end_delta = abs(b - e)
        if overlap_ratio < 0.12 and start_delta > 0.55 and center_delta > 0.80:
            continue
        score = (
            overlap_ratio * 6.2
            + max(0.0, 1.0 - start_delta / window) * 2.4
            + max(0.0, 1.0 - end_delta / window) * 2.0
            - center_delta * 0.70
            - abs(vd - sd) * 0.05
        )
        if center_delta <= 0.40:
            score += 0.9
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _scan_video(video_path: Path, region: Optional[Dict[str, Any]]) -> Tuple[List[Tuple[float, float, float, float]], float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video để căn timing STT: {video_path}")
    try:
        fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps_video <= 0:
            fps_video = 25.0
        scan_fps = max(4.0, min(float(getattr(config, "STT_VISUAL_SCAN_FPS", 12.0)), fps_video))
        frame_step = max(1, int(round(fps_video / scan_fps)))
        effective_fps = fps_video / frame_step
        bounds = _region_pixels(cap, region or {"x": 0, "y": 72, "w": 100, "h": 25})
        samples: List[Tuple[float, float, float, float]] = []
        frame_no = 0
        next_frame = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no >= next_frame:
                tm = frame_no / fps_video
                w, ed, act = _frame_metric(frame, bounds)
                samples.append((tm, w, ed, act))
                next_frame += frame_step
            frame_no += 1
        return samples, effective_fps
    finally:
        cap.release()


def _join_cue_text(parts: List[str]) -> str:
    out: List[str] = []
    for raw in parts:
        text = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not text:
            continue
        if out and (text == out[-1] or text in out[-1] or out[-1] in text):
            if len(text) > len(out[-1]):
                out[-1] = text
            continue
        out.append(text)
    return " ".join(out)


def align_stt_segments_to_video(video_path: str | Path, segments: List[Dict[str, Any]], region: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Use burned-in subtitle ON/OFF as the master timeline; keep STT/OCR text."""
    if not segments or not bool(getattr(config, "STT_VISUAL_TIMING_ALIGN", True)):
        return list(segments or [])
    path = Path(video_path)
    if not path.exists():
        logger.warning("[STT TIMING] Video không tồn tại: %s", path)
        return list(segments)
    try:
        samples, fps = _scan_video(path, region)
        intervals = _detect_visual_intervals(samples, fps)
        if not intervals:
            logger.warning("[STT TIMING] Không phát hiện được khoảng subtitle trên vùng quét; giữ timestamp nguồn")
            return [dict(s) for s in segments]

        window = max(0.60, float(getattr(config, "STT_VISUAL_TIMING_WINDOW", 2.50)))
        source = sorted((dict(s) for s in segments), key=lambda x: float(x.get("start", 0) or 0))
        buckets: List[List[Dict[str, Any]]] = [[] for _ in intervals]
        leftover: List[Dict[str, Any]] = []
        for original in source:
            match_idx = _match_visual_interval(original, intervals, set(), window)
            if match_idx is None:
                leftover.append(original)
            else:
                buckets[match_idx].append(original)

        for original in leftover:
            match_idx = _match_visual_interval(original, intervals, set(), min(window, 1.15))
            if match_idx is None:
                continue
            buckets[match_idx].append(original)

        matched = sum(len(b) for b in buckets)
        if matched < max(2, int(round(len(source) * 0.42))):
            logger.warning("[STT TIMING] Visual match quá thưa (%d/%d); giữ timestamp nguồn", matched, len(source))
            return [dict(s) for s in segments]

        out: List[Dict[str, Any]] = []
        adjusted = 0
        for i, iv in enumerate(intervals):
            members = buckets[i]
            if not members:
                continue
            members.sort(key=lambda x: float(x.get("start", 0) or 0))
            base = dict(members[0])
            stt_start = float(members[0].get("start", 0) or 0)
            stt_end = float(members[-1].get("end", stt_start) or stt_start)
            a, b = float(iv["start"]), float(iv["end"])
            if out and a < float(out[-1]["end"]) - 1e-3:
                a = float(out[-1]["end"])
            if b <= a + 0.08:
                continue
            texts = [str(m.get("text", m.get("source_text", "")) or "") for m in members]
            base["start"] = round(a, 3)
            base["end"] = round(b, 3)
            base["text"] = _join_cue_text(texts)
            base["stt_start"] = round(stt_start, 3)
            base["stt_end"] = round(stt_end, 3)
            base["source_start"] = round(a, 3)
            base["source_end"] = round(b, 3)
            base["timing_source"] = "stt+visual-boundary"
            base["visual_timing_adjusted"] = abs(a - stt_start) >= 0.04 or abs(b - stt_end) >= 0.04 or len(members) > 1
            base.pop("timeline_overlap", None)
            if base["visual_timing_adjusted"]:
                adjusted += 1
            out.append(base)

        unmatched = []
        for original in leftover:
            if any(original in b for b in buckets):
                continue
            unmatched.append(original)
        for original in unmatched:
            a = float(original.get("start", 0) or 0)
            b = float(original.get("end", a + 0.05) or a + 0.05)
            if b <= a:
                continue
            if any(min(b, float(s["end"])) - max(a, float(s["start"])) > 0.12 for s in out):
                continue
            item = dict(original)
            item.setdefault("timing_source", "stt")
            item["stt_start"] = round(a, 3)
            item["stt_end"] = round(b, 3)
            item["source_start"] = a
            item["source_end"] = b
            item["visual_timing_adjusted"] = False
            out.append(item)
        out.sort(key=lambda x: float(x.get("start", 0) or 0))
        for i in range(1, len(out)):
            prev, cur = out[i - 1], out[i]
            if float(cur["start"]) < float(prev["end"]):
                cur["start"] = float(prev["end"])
                if float(cur["end"]) <= float(cur["start"]) + 0.05:
                    cur["end"] = float(cur["start"]) + 0.08
                cur["source_start"], cur["source_end"] = cur["start"], cur["end"]
        for i, seg in enumerate(out):
            seg["segment_id"] = i
        logger.info("[STT TIMING] Visual intervals=%d | cues=%d | merged_from=%d | adjusted=%d | scan_fps=%.1f",
                    len(intervals), len(out), len(source), adjusted, fps)
        return out
    except Exception as exc:
        logger.exception("[STT TIMING] Alignment lỗi; giữ nguyên timestamp nguồn: %s", exc)
        return [dict(s) for s in segments]


__all__ = ["align_stt_segments_to_video"]
