from __future__ import annotations
import cv2, numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Activity:
    changed: bool
    score: float
    textlike: float
    boxes: List[Tuple[int, int, int, int]]
    bbox: Optional[Tuple[int, int, int, int]] = None
    textlike_delta: float = 0.0

class TextActivityDetector:
    """Cheap, high-recall temporal activity detector.

    It is deliberately NOT an OCR confidence filter. Its job is to create candidate
    windows, so uncertain activity is retained for the heavy OCR stage.
    """
    def __init__(self, threshold: float = 0.018, min_component: int = 8):
        self.threshold = float(threshold)
        self.min_component = int(min_component)

    def _text_mask(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return np.zeros((1, 1), np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # Adaptive threshold alone turns textured video backgrounds into nearly-full masks.
        # Use morphological top-hat/black-hat contrast to isolate glyph-sized bright/dark strokes,
        # then keep only plausible text-line connected components.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 11))
        whitehat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, wh = cv2.threshold(whitehat, 18, 255, cv2.THRESH_BINARY)
        _, bh = cv2.threshold(blackhat, 18, 255, cv2.THRESH_BINARY)
        raw = cv2.bitwise_or(wh, bh)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)), iterations=1)
        clean = np.zeros_like(raw)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        h, w = raw.shape[:2]
        for i in range(1, n):
            x, y, ww, hh, area = [int(v) for v in stats[i]]
            if area < self.min_component or ww < 2 or hh < 2:
                continue
            if ww * hh > 0.18 * w * h:
                continue
            ar = ww / max(1, hh)
            if ar < 0.25 or ar > 40.0:
                continue
            if hh > 0.30 * h:
                continue
            cv2.drawContours(clean, [np.array([[x,y],[x+ww,y],[x+ww,y+hh],[x,y+hh]], dtype=np.int32)], -1, 255, -1)
        return clean

    def _boxes(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        boxes: List[Tuple[int, int, int, int]] = []
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        h, w = mask.shape[:2]
        for i in range(1, n):
            x, y, ww, hh, area = [int(v) for v in stats[i]]
            if area < self.min_component or ww < 3 or hh < 2:
                continue
            if ww * hh > 0.35 * w * h:
                continue
            ar = ww / max(1, hh)
            if ar < 0.3 and hh > 0.3 * h:
                continue
            boxes.append((x, y, x + ww, y + hh))
        return boxes

    @staticmethod
    def _union(boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        if not boxes:
            return None
        return (
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        )

    @staticmethod
    def _resize_gray(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        pg = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if prev.ndim == 3 else prev
        cg = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY) if cur.ndim == 3 else cur
        return cv2.resize(pg, (cg.shape[1], cg.shape[0]), interpolation=cv2.INTER_AREA)

    def analyze(self, prev: Optional[np.ndarray], cur: np.ndarray) -> Activity:
        gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY) if cur.ndim == 3 else cur
        mask = self._text_mask(cur)
        textlike = float(np.count_nonzero(mask)) / max(1, mask.size)
        boxes = self._boxes(mask)
        bbox = self._union(boxes)

        if prev is None:
            return Activity(
                changed=True, score=1.0, textlike=textlike,
                boxes=boxes, bbox=bbox, textlike_delta=textlike,
            )

        prev_gray = self._resize_gray(prev, cur)
        delta = cv2.absdiff(prev_gray, gray)
        diff = float(np.mean(delta > 12))

        prev_mask = self._text_mask(prev)
        prev_like = float(np.count_nonzero(prev_mask)) / max(1, prev_mask.size)
        textlike_delta = abs(textlike - prev_like)
        resized_prev_mask = cv2.resize(prev_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        inter = float(np.count_nonzero(cv2.bitwise_and(mask, resized_prev_mask)))
        union = float(np.count_nonzero(cv2.bitwise_or(mask, resized_prev_mask)))
        mask_iou = inter / max(1.0, union)

        # Raw frame motion is deliberately NOT enough to open an OCR window. Movies
        # commonly move in the subtitle band every frame. We react to text-density
        # changes or text-mask structural changes while retaining the raw diff as a
        # supporting signal. This prevents stable subtitles from creating a 60-minute
        # burst window, but still keeps ambiguous transitions for verification.
        threshold = float(self.threshold)
        text_transition = textlike_delta >= threshold * 0.55
        appearance = prev_like <= 0.010 and textlike >= 0.012
        disappearance = prev_like >= 0.012 and textlike <= 0.010
        structural_text_change = mask_iou < 0.42 and max(prev_like, textlike) >= 0.012 and diff >= threshold * 0.35
        motion_supported_text_change = diff >= threshold * 2.5 and textlike_delta >= threshold * 0.35
        changed = appearance or disappearance or text_transition or structural_text_change or motion_supported_text_change
        score = max(textlike_delta * 2.5, (1.0 - mask_iou) * 0.5 if max(prev_like, textlike) >= 0.012 else 0.0, min(1.0, diff * 0.25))
        return Activity(
            changed=bool(changed), score=float(score), textlike=textlike,
            boxes=boxes, bbox=bbox, textlike_delta=float(textlike_delta),
        )
