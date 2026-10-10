from __future__ import annotations
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _norm(s: str) -> str:
    return " ".join(str(s or "").casefold().split())


def _lev(a: Sequence[Any], b: Sequence[Any]) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def text_cer(pred: str, truth: str) -> float:
    t = _norm(truth).replace(" ", "")
    p = _norm(pred).replace(" ", "")
    return _lev(list(p), list(t)) / max(1, len(t))


def text_wer(pred: str, truth: str) -> float:
    t = _norm(truth).split()
    p = _norm(pred).split()
    return _lev(p, t) / max(1, len(t))


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / max(union, 1e-9)


def _best_match(pred: Dict[str, Any], truths: List[Dict[str, Any]], used: set[int], threshold: float) -> int | None:
    best_i, best_score = None, 0.0
    preg = str(pred.get("region_id", "roi"))
    for i, gt in enumerate(truths):
        if i in used or str(gt.get("region_id", "roi")) != preg:
            continue
        score = _overlap(float(pred.get("start_time", 0.0)), float(pred.get("end_time", 0.0)),
                         float(gt.get("start_time", 0.0)), float(gt.get("end_time", 0.0)))
        # Text agreement is a secondary signal; temporal overlap stays primary.
        sim = 1.0 - text_cer(str(pred.get("text", "")), str(gt.get("text", "")))
        score = 0.80 * score + 0.20 * max(0.0, sim)
        if score >= threshold and score > best_score:
            best_i, best_score = i, score
    return best_i


class GroundTruthEvaluator:
    """Ground-truth evaluator for subtitle recall and frame-boundary accuracy."""

    def evaluate(self, predicted: Iterable[Dict[str, Any]], ground_truth: Iterable[Dict[str, Any]],
                 match_threshold: float = 0.45) -> Dict[str, Any]:
        preds = [dict(x) for x in predicted]
        truths = [dict(x) for x in ground_truth]
        used: set[int] = set()
        matches: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        start_errors: List[float] = []
        end_errors: List[float] = []
        cer: List[float] = []
        wer: List[float] = []
        for pred in sorted(preds, key=lambda x: float(x.get("start_time", 0.0))):
            i = _best_match(pred, truths, used, match_threshold)
            if i is None:
                continue
            used.add(i)
            gt = truths[i]
            matches.append((pred, gt))
            ps, pe = int(pred.get("frame_start", -1)), int(pred.get("frame_end", -1))
            gs, ge = int(gt.get("frame_start", ps)), int(gt.get("frame_end", pe))
            start_errors.append(abs(ps - gs)); end_errors.append(abs(pe - ge))
            cer.append(text_cer(str(pred.get("text", "")), str(gt.get("text", ""))))
            wer.append(text_wer(str(pred.get("text", "")), str(gt.get("text", ""))))
        matched = len(matches)
        return {
            "ground_truth_segments": len(truths),
            "predicted_segments": len(preds),
            "matched_segments": matched,
            "subtitle_recall": matched / max(1, len(truths)),
            "subtitle_precision": matched / max(1, len(preds)),
            "character_error_rate": sum(cer) / max(1, len(cer)),
            "word_error_rate": sum(wer) / max(1, len(wer)),
            "start_boundary_error_frames_mean": sum(start_errors) / max(1, len(start_errors)),
            "end_boundary_error_frames_mean": sum(end_errors) / max(1, len(end_errors)),
            "start_boundary_error_frames_max": max(start_errors, default=0),
            "end_boundary_error_frames_max": max(end_errors, default=0),
            "unmatched_ground_truth": [i for i in range(len(truths)) if i not in used],
        }
