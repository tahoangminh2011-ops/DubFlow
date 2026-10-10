"""pipeline/subtitle.py - Subtitle utilities.
1:1 timing lock: phụ đề dịch xuất hiện/kết thúc cùng lúc lời nói gốc.
"""
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import config

logger = logging.getLogger(__name__)

MAX_SUBTITLE_CHARS = int(getattr(config, "MAX_SUBTITLE_CHARS", 42))
MIN_SEGMENT_SECONDS = 0.001
GAP_SECONDS = float(getattr(config, "SEGMENT_GAP_SECONDS", 0.02))


def _f(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_srt_time(t) -> str:
    t = max(0.0, _f(t, 0.0))
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def normalize_segments(segments) -> List[Dict[str, Any]]:
    """Chuẩn hóa SRT mà không thay đổi timing nguồn; chỉ đánh dấu overlap nếu có."""
    # Preserve MASTER TIMELINE order exactly. Never sort here because
    # translated_texts are positional (segment i <-> translation i).
    raw = [dict(seg) for seg in (segments or [])]
    out: List[Dict[str, Any]] = []
    for seg in raw:
        s = dict(seg)
        start = max(0.0, _f(s.get("start"), 0.0))
        end = _f(s.get("end", s.get("stop")), 0.0)
        if end <= start:
            end = start + MIN_SEGMENT_SECONDS
        # Preserve source cue timing exactly; do not trim previous cue to fix
        # overlaps here. Chunk de-duplication is handled in STT before this layer.
        s["start"] = round(start, 3)
        s["end"] = round(max(start + MIN_SEGMENT_SECONDS, end), 3)
        if out and s["start"] < out[-1]["end"]:
            s["timeline_overlap"] = True
            out[-1]["timeline_overlap"] = True
        s["text"] = str(s.get("text", s.get("source_text", "")) or "").strip()
        out.append(s)
    return out

def lock_source_timing(segments) -> List[Dict[str, Any]]:
    """Khóa timing đúng bằng timestamp nguồn (1:1)."""
    return normalize_segments(segments)


def split_long_text(text: str, max_chars: int = MAX_SUBTITLE_CHARS) -> List[str]:
    t = str(text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    words = re.split(r"\s+", t)
    lines: List[str] = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip() if cur else w
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            while len(w) > max_chars:
                lines.append(w[:max_chars])
                w = w[max_chars:]
            cur = w
    if cur:
        lines.append(cur)
    return lines


def align_split_segments(segments, translated_texts) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Canh chỉnh 1:1 với segment nguồn. KHÔNG tách segment."""
    segments = normalize_segments(segments)
    translated = [str(t or "").strip() for t in (translated_texts or [])]
    while len(translated) < len(segments):
        translated.append("")
    if len(translated) > len(segments):
        translated = translated[:len(segments)]
    logger.info(f"Timing correction: {len(segments)} segments (1:1)")
    return [dict(s) for s in segments], translated


def write_srt(segments, texts, out_path, **kwargs):
    locked = lock_source_timing(segments)
    if not locked:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        return out
    texts = [str(t or "").strip() for t in (texts or [])]
    while len(texts) < len(locked):
        texts.append("")
    lines: List[str] = []
    idx = 0
    for seg, tr in zip(locked, texts):
        idx += 1
        lines.append(str(idx))
        lines.append(f"{fmt_srt_time(seg.get('start', 0))} --> {fmt_srt_time(seg.get('end', 0))}")
        lines.append("\n".join(split_long_text(tr)) if tr else "")
        lines.append("")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Đã ghi SRT: {out} ({idx} dòng phụ đề)")
    return out


def write_bilingual_srt(segments, texts, out_path, **kwargs):
    locked = lock_source_timing(segments)
    texts = [str(t or "").strip() for t in (texts or [])]
    while len(texts) < len(locked):
        texts.append("")
    lines: List[str] = []
    idx = 0
    for seg, tr in zip(locked, texts):
        original = str(seg.get("text", "") or "").strip()
        body_lines = ([original] if original else []) + (split_long_text(tr) if tr else [])
        idx += 1
        lines.append(str(idx))
        lines.append(f"{fmt_srt_time(seg.get('start', 0))} --> {fmt_srt_time(seg.get('end', 0))}")
        lines.append("\n".join(body_lines))
        lines.append("")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def build_subtitle_lines(segments, texts=None, translated_texts=None, **kwargs):
    texts = texts or translated_texts or []
    locked = lock_source_timing(segments)
    texts = [str(t or "").strip() for t in texts]
    while len(texts) < len(locked):
        texts.append("")
    return [(seg["start"], seg["end"], tr) for seg, tr in zip(locked, texts)]


def parse_srt(path) -> List[Dict[str, Any]]:
    content = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", content.strip())
    out: List[Dict[str, Any]] = []

    def _ts(s: str) -> float:
        s = s.strip().replace(",", ":")
        parts = s.split(":")
        try:
            if len(parts) == 4:
                h, m, sec, ms = parts
            elif len(parts) == 3:
                h, m, sec = parts
                ms = "0"
            else:
                return 0.0
            return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0
        except ValueError:
            return 0.0

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        timing_idx = 0
        if "-->" not in lines[0] and "-->" in lines[1]:
            timing_idx = 1
        m = re.match(r"(.+?)\s*-->\s*(.+)", lines[timing_idx])
        if not m:
            continue
        text = " ".join(lines[timing_idx + 1:])
        out.append({"start": _ts(m.group(1)), "end": _ts(m.group(2)), "text": text})
    return out


__all__ = [
    "normalize_segments", "lock_source_timing", "align_split_segments",
    "split_long_text", "fmt_srt_time", "write_srt", "write_bilingual_srt",
    "build_subtitle_lines", "parse_srt",
]