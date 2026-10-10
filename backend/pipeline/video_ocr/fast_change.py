from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FastActivity:
    changed: bool
    score: float
    diff_ratio: float
    textlike: float
    textlike_delta: float
    line_score: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    line_density: float = 0.0
    line_coverage: float = 0.0
    presence: bool = False
    appearance: bool = False
    disappearance: bool = False
    replacement: bool = False
    mask_iou: float = 1.0
    structure_delta: float = 0.0


class FastChangeDetector:
    """Very cheap temporal text-line gate.

    The hot path must stay CPU-light because it runs for every 15 FPS sample.  It therefore
    avoids morphology, connected components, threshold pyramids and OCR.  Instead it combines:
    - local contrast against a tiny Gaussian blur;
    - NumPy gradient density;
    - horizontal concentration (subtitle-like line structure);
    - temporal mask similarity.

    This detector is a *candidate generator*, never an OCR confidence filter.  Ambiguous
    activity is retained for the heavy OCR stage.
    """

    def __init__(self, threshold: float = 0.018, down_w: int = 192):
        self.threshold = float(threshold)
        self.down_w = max(96, int(down_w))

    def _gray_small(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((1, self.down_w), np.uint8)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        if w != self.down_w:
            nh = max(12, int(round(h * self.down_w / max(1, w))))
            image = cv2.resize(image, (self.down_w, nh), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def _features(gray: np.ndarray):
        """Return a binary stroke-like mask and text-line statistics."""
        if gray is None or gray.size == 0:
            return np.zeros((1, 1), np.uint8), 0.0, 0.0, 0.0, 0.0, None

        # Cheap local-contrast estimate.  It works for bright text, dark outlined text and
        # subtitles over moving backgrounds without the expensive top-hat/black-hat passes.
        blur = cv2.blur(gray, (3, 3))
        local = cv2.absdiff(gray, blur)
        gx = np.abs(gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16))
        gy = np.abs(gray[1:, :].astype(np.int16) - gray[:-1, :].astype(np.int16))
        edges = np.zeros_like(gray, dtype=bool)
        edges[:, 1:] |= gx >= 8
        edges[1:, :] |= gy >= 8
        mask = (edges & (local >= 8)).astype(np.uint8)

        h, w = mask.shape
        row_density = mask.mean(axis=1).astype(np.float32)
        total = float(row_density.sum())
        if total <= 0.0:
            return mask, 0.0, 0.0, 0.0, 0.0, None

        # Subtitle lines are spatially concentrated in a relatively short horizontal band.
        best = None
        for frac in (0.05, 0.08, 0.12, 0.16):
            bh = max(3, min(h, int(round(h * frac))))
            if bh >= h:
                i = 0
                score = float(row_density.sum())
            else:
                smooth = np.convolve(row_density, np.ones(bh, np.float32), mode="valid")
                i = int(np.argmax(smooth))
                score = float(smooth[i])
            concentration = score / max(total, 1e-9)
            candidate = (concentration, score, i, bh)
            if best is None or candidate[0] > best[0]:
                best = candidate

        concentration, score, y0, bh = best
        y1 = min(h, y0 + bh)
        band = mask[y0:y1]
        band_density = float(band.mean())
        col_spread = float(np.mean(np.any(band > 0, axis=0)))

        # Tiny subtitle-like line mask.  Do not morphology-clean it here; temporal similarity
        # works better when we preserve the raw stroke layout.
        bbox = None
        if band.size:
            ys, xs = np.where(band > 0)
            if len(xs) >= 4:
                bbox = (int(xs.min()), int(y0 + ys.min()), int(xs.max() + 1), int(y0 + ys.max() + 1))

        # textlike is intentionally conservative; line_score carries most of the subtitle signal.
        textlike = float(mask.mean())
        line_score = float(concentration)
        return mask, textlike, line_score, band_density, col_spread, bbox

    def _text_presence(self, gray: np.ndarray):
        mask, textlike, line_score, _band, _cov, _bbox = self._features(self._gray_small(gray))
        return mask, textlike, line_score

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        aa = a > 0
        bb = b > 0
        inter = float(np.count_nonzero(aa & bb))
        union = float(np.count_nonzero(aa | bb))
        return inter / max(1.0, union)

    @staticmethod
    def _mask_corr(a: np.ndarray, b: np.ndarray) -> float:
        x = a.astype(np.float32).ravel()
        y = b.astype(np.float32).ravel()
        x -= x.mean(); y -= y.mean()
        den = float(np.linalg.norm(x) * np.linalg.norm(y))
        return 0.0 if den <= 1e-8 else float(np.dot(x, y) / den)

    @staticmethod
    def _change_band_stats(prev: np.ndarray, cur: np.ndarray):
        """Cheap temporal-change concentration used to catch subtitle transitions even when
        the subtitle detector cannot form a confident text mask at tiny scan resolution.

        Subtitle appearance/disappearance usually creates a compact horizontal band of pixel
        changes. We use that as a candidate generator only; OCR remains the authority.
        """
        if prev is None or cur is None or prev.size == 0 or cur.size == 0:
            return 0.0, 0.0, 0.0
        d = cv2.absdiff(prev, cur)
        change = (d >= 12).astype(np.uint8)
        h, w = change.shape[:2]
        if h < 4 or w < 8:
            return float(change.mean()), 0.0, 0.0
        rows = change.mean(axis=1)
        # Best compact horizontal band covering roughly a subtitle line or two.
        best = 0.0
        for frac in (0.04, 0.06, 0.09, 0.12, 0.18):
            bh = max(2, min(h, int(round(h * frac))))
            if bh >= h:
                score = float(rows.mean())
            else:
                sm = np.convolve(rows, np.ones(bh, np.float32), mode="valid") / float(bh)
                score = float(sm.max())
            best = max(best, score)
        total = float(rows.mean())
        concentration = best / max(total, 1e-6)
        # Coverage says whether the changed band spans enough of the screen to resemble a line.
        idx = int(np.argmax(rows))
        band_h = max(2, int(round(h * 0.09)))
        y0 = max(0, min(h - band_h, idx - band_h // 2))
        band = change[y0:y0+band_h]
        coverage = float(np.mean(np.any(band > 0, axis=0)))
        return total, concentration, coverage

    def analyze(self, previous: Optional[np.ndarray], current: np.ndarray) -> FastActivity:
        cur = self._gray_small(current)
        cur_mask, cur_textlike, cur_line, cur_band, cur_cov, cur_bbox = self._features(cur)

        if previous is None:
            initial_presence = bool(
                (cur_line >= 0.22 and cur_band >= 0.10 and cur_cov >= 0.34 and cur_textlike >= 0.010)
                or (cur_line >= 0.48 and cur_band >= 0.070 and cur_cov >= 0.24 and cur_textlike >= 0.008)
                # Low-resolution CJK/outlined subtitles can have thin strokes after downscale.
                # This permissive branch only opens a candidate; full OCR decides whether it is text.
                or (cur_line >= 0.34 and cur_band >= 0.022 and cur_cov >= 0.10 and cur_textlike >= 0.0024)
                or (cur_line >= 0.56 and cur_band >= 0.014 and cur_cov >= 0.075 and cur_textlike >= 0.0016)
            )
            return FastActivity(changed=True, score=1.0, diff_ratio=1.0, textlike=cur_textlike,
                                textlike_delta=cur_textlike, line_score=cur_line, bbox=cur_bbox,
                                line_density=cur_band, line_coverage=cur_cov, presence=initial_presence,
                                appearance=initial_presence, disappearance=False, replacement=False,
                                mask_iou=1.0, structure_delta=1.0)

        prev = self._gray_small(previous)
        if prev.shape != cur.shape:
            prev = cv2.resize(prev, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_AREA)
        prev_mask, prev_textlike, prev_line, prev_band, prev_cov, prev_bbox = self._features(prev)

        diff = cv2.absdiff(prev, cur)
        diff_ratio = float(np.mean(diff >= 14))
        text_delta = abs(cur_textlike - prev_textlike)
        band_delta = abs(cur_band - prev_band)
        mask_iou = self._mask_iou(prev_mask, cur_mask)
        corr = self._mask_corr(prev_mask, cur_mask)
        change_ratio, change_conc, change_cov = self._change_band_stats(prev, cur)

        # Strong text presence remains the primary state, but temporal change concentration is
        # an intentionally permissive safety net. At tiny scan resolution a subtitle can be too
        # small for the text mask while still producing a clear horizontal transition.
        # The old gate required an unusually strong horizontal concentration on the *whole frame*.
        # That caused real subtitles (especially large outlined CJK text) to report presence=False
        # even when the line was visibly obvious.  Keep the detector permissive because it only
        # opens candidate windows; OCR/verification remains the authority.
        def _presence(line, band, cov, density):
            return bool(
                (line >= 0.22 and band >= 0.10 and cov >= 0.34 and density >= 0.010)
                or (line >= 0.48 and band >= 0.070 and cov >= 0.24 and density >= 0.008)
                # High-recall candidate-only branch for small/faded subtitles. False positives are
                # retained for OCR verification; they are not emitted directly as subtitle text.
                or (line >= 0.34 and band >= 0.022 and cov >= 0.10 and density >= 0.0024)
                or (line >= 0.56 and band >= 0.014 and cov >= 0.075 and density >= 0.0016)
            )
        prev_present = _presence(prev_line, prev_band, prev_cov, prev_textlike)
        cur_present = _presence(cur_line, cur_band, cur_cov, cur_textlike)

        transition_signal = bool(
            diff_ratio >= 0.0024
            and change_conc >= 1.65
            and change_cov >= 0.08
        )
        appearance = (
            ((not prev_present) and cur_present)
            or ((not prev_present) and transition_signal and (cur_line >= 0.40 or cur_band >= 0.015 or cur_textlike >= 0.003))
        )
        disappearance = (
            (prev_present and (not cur_present))
            or (prev_present and transition_signal and cur_band < max(0.014, prev_band * 0.62))
        )

        # A subtitle replacement must be a structural *step*, not merely a different edge mask.
        # Moving scenes can make mask IoU almost zero on every frame while the subtitle remains
        # unchanged. The reliable low-cost cues are a jump in line density/coverage or a clear
        # change in the detected text-band width, supported by concentrated temporal change.
        def width_ratio(a, b):
            if not a or not b:
                return 0.0
            aw=max(1, int(a[2]-a[0])); bw=max(1, int(b[2]-b[0]))
            return abs(aw-bw)/max(1, max(aw,bw))
        bbox_width_delta = width_ratio(prev_bbox, cur_bbox)
        band_step = abs(cur_band - prev_band)
        coverage_step = abs(cur_cov - prev_cov)
        structure_delta = max(
            min(1.0, band_step / 0.045),
            min(1.0, coverage_step / 0.10),
            min(1.0, bbox_width_delta / 0.35),
            min(1.0, text_delta / 0.018),
        )
        # IMPORTANT: stable subtitles frequently produce different tiny masks after resize,
        # compression and anti-aliasing. Treating bbox/text-mask drift as a lexical replacement
        # creates a candidate almost every sample. Lexical changes are recovered later by the
        # recognition-only interior sampler, so the fast gate only emits a replacement when the
        # *visual transition itself* is strong and broad enough.
        structure_changed = bool(
            (band_step >= 0.026 and coverage_step >= 0.032)
            or (band_step >= 0.038)
            or (coverage_step >= 0.070)
        )
        replacement = bool(
            prev_present and cur_present
            and diff_ratio >= max(0.020, self.threshold * 1.05)
            and change_conc >= 1.45
            and change_cov >= 0.10
            and structure_changed
        )

        # Conservative density-change trigger for animation/fade-in/fade-out.  This is deliberately
        # weaker than the state transitions, but still produces a verification window.
        density_change = (
            (prev_present or cur_present)
            and max(prev_line, cur_line) >= 0.48
            and max(prev_band, cur_band) >= 0.025
            and band_delta >= 0.012
            and diff_ratio >= 0.0025
            and (change_conc >= 1.45 or change_cov >= 0.20)
        )

        changed = bool(appearance or disappearance or replacement or density_change)
        score = max(
            1.0 if appearance or disappearance else 0.0,
            min(1.0, abs(cur_line - prev_line) * 2.0),
            min(1.0, (1.0 - max(0.0, mask_iou)) * 0.7) if prev_present and cur_present else 0.0,
            min(1.0, diff_ratio * 8.0) if replacement else 0.0,
            min(1.0, max(0.0, change_conc - 1.0) / 2.5) if transition_signal else 0.0,
        )
        return FastActivity(
            changed=changed,
            score=float(score),
            diff_ratio=diff_ratio,
            textlike=cur_textlike,
            textlike_delta=text_delta,
            line_score=cur_line,
            bbox=cur_bbox,
            line_density=cur_band,
            line_coverage=cur_cov,
            presence=cur_present,
            appearance=bool(appearance),
            disappearance=bool(disappearance),
            replacement=bool(replacement),
            mask_iou=float(mask_iou),
            structure_delta=float(structure_delta),
        )

    def analyze_pair(self, previous: Optional[np.ndarray], current: np.ndarray,
                     roi: Optional[Tuple[int, int, int, int]] = None):
        """Analyze full-frame and a ROI from the same already-resized grayscale frame."""
        full = self.analyze(previous, current)
        if roi is None:
            return full, None
        x0, y0, x1, y1 = roi
        cur_crop = current[max(0,y0):max(y0+1,y1), max(0,x0):max(x0+1,x1)]
        prev_crop = None
        if previous is not None:
            prev_crop = previous[max(0,y0):max(y0+1,y1), max(0,x0):max(x0+1,x1)]
        return full, self.analyze(prev_crop, cur_crop)
