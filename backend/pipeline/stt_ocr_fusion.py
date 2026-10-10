"""STT + OCR master-timeline fusion.

STT remains the authoritative source for spoken content. OCR is used as a visual
subtitle clock when an OCR cue can be matched to an STT cue in time/text space.
The result is one immutable master timeline consumed by translation/TTS/render.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


def _text(v: Any) -> str:
    return str(v or "").strip()


def _norm(v: Any) -> str:
    s = unicodedata.normalize("NFKC", _text(v)).casefold()
    s = re.sub(r"[^\w\u00C0-\u024F\u1E00-\u1EFF]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def _similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Token overlap is cheap and works well for OCR accents/punctuation variants.
    ta, tb = set(a.split()), set(b.split())
    if ta and tb:
        j = len(ta & tb) / max(1, len(ta | tb))
    else:
        j = 0.0
    # Character containment catches OCR strings with merged/split tokens.
    shorter, longer = sorted((a, b), key=len)
    contain = len(shorter) / max(1, len(longer)) if shorter in longer else 0.0
    return max(j, contain)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(0.001, max(a1, b1) - min(a0, b0))
    return inter / union


def _score(stt: Dict[str, Any], ocr: Dict[str, Any], window: float) -> float:
    s0, s1 = float(stt.get("start", 0) or 0), float(stt.get("end", 0) or 0)
    o0, o1 = float(ocr.get("start", 0) or 0), float(ocr.get("end", 0) or 0)
    ov = _overlap(s0, s1, o0, o1)
    sc = _similarity(stt.get("text", ""), ocr.get("text", ""))
    sc += max(0.0, float(ocr.get("confidence", 0) or 0)) * 0.08
    center = abs(((s0 + s1) / 2.0) - ((o0 + o1) / 2.0))
    temporal = max(0.0, 1.0 - center / max(0.1, window))
    return ov * 0.58 + sc * 0.30 + temporal * 0.12


def fuse_stt_ocr(stt_segments: List[Dict[str, Any]],
                 ocr_segments: List[Dict[str, Any]],
                 window: float = 2.0,
                 match_threshold: float = 0.42,
                 text_similarity_min: float = 0.10) -> List[Dict[str, Any]]:
    """Create the single master timeline.

    - STT text is preserved.
    - OCR text is metadata only.
    - A reliable OCR match locks start/end to OCR's visual boundary.
    - Unmatched STT keeps its own timing.
    - No segment overlaps the preceding master segment.
    """
    stt = sorted([dict(x) for x in (stt_segments or [])], key=lambda x: float(x.get("start", 0) or 0))
    ocr = sorted([dict(x) for x in (ocr_segments or []) if _text(x.get("text"))],
                 key=lambda x: float(x.get("start", 0) or 0))
    used = set()
    out: List[Dict[str, Any]] = []
    j0 = 0

    for idx, s in enumerate(stt):
        s0 = float(s.get("start", 0) or 0)
        s1 = max(s0 + 0.05, float(s.get("end", s0 + 0.05) or s0 + 0.05))
        while j0 < len(ocr) and float(ocr[j0].get("end", 0) or 0) < s0 - window:
            j0 += 1
        best_i, best_score = None, -1.0
        k = j0
        while k < len(ocr):
            o = ocr[k]
            o0 = float(o.get("start", 0) or 0)
            if o0 > s1 + window:
                break
            if k in used:
                k += 1
                continue
            sim = _similarity(s.get("text", ""), o.get("text", ""))
            score = _score({**s, "end": s1}, o, window)
            # Require either meaningful text agreement or real temporal overlap.
            ov = _overlap(s0, s1, o0, float(o.get("end", o0) or o0))
            if score >= match_threshold and (sim >= text_similarity_min or ov >= 0.20):
                if score > best_score:
                    best_i, best_score = k, score
            k += 1

        item = dict(s)
        item["start"] = s0
        item["end"] = s1
        item["master_timeline"] = True
        item["timing_locked"] = False
        item["timing_source"] = "stt"
        item["source_text"] = _text(s.get("text", ""))
        item["fusion_index"] = idx
        if best_i is not None:
            o = ocr[best_i]
            o0 = float(o.get("start", 0) or 0)
            o1 = max(o0 + 0.05, float(o.get("end", o0 + 0.05) or o0 + 0.05))
            # OCR is the visual clock; STT remains the content clock.
            item["start"] = o0
            item["end"] = o1
            item["timing_locked"] = True
            item["timing_source"] = "stt+ocr"
            item["ocr_text"] = _text(o.get("text", ""))
            item["ocr_confidence"] = float(o.get("confidence", 0) or 0)
            item["fusion_score"] = round(best_score, 4)
            used.add(best_i)
        else:
            item["ocr_text"] = ""
            item["fusion_score"] = 0.0
        out.append(item)

    # Master timeline is strictly ordered and non-overlapping.
    out.sort(key=lambda x: (float(x.get("start", 0) or 0), float(x.get("end", 0) or 0)))
    fixed: List[Dict[str, Any]] = []
    for item in out:
        a = float(item.get("start", 0) or 0)
        b = float(item.get("end", a + 0.05) or a + 0.05)
        if fixed and a < float(fixed[-1].get("end", 0) or 0):
            a = float(fixed[-1].get("end", 0) or 0)
        if b <= a + 0.05:
            continue
        item["start"], item["end"] = round(a, 4), round(b, 4)
        fixed.append(item)
    return fixed
