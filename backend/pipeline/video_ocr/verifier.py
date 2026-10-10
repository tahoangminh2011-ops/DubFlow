from __future__ import annotations
from typing import Any, Dict
from .normalizer import similarity, weighted_consensus
from .tracker import TextTrack

class OCRVerifier:
    """Cross-frame consensus + explicit verification queue.

    Low confidence is never a reason to discard a track. It only increases verification.
    """
    def __init__(self, high_recall: bool = True):
        self.high_recall = bool(high_recall)

    def verify(self, track: TextTrack) -> Dict[str, Any]:
        if not track.observations:
            return {"verified": False, "needs_review": True, "text": "", "verification_queue": True}
        texts = [(o.text, max(0.01, o.confidence)) for o in track.observations]
        canonical = weighted_consensus(texts)
        agreeing = [o for o in track.observations if similarity(canonical, o.text) >= 0.72]
        avg = sum(o.confidence for o in agreeing) / max(1, len(agreeing))
        needs = (
            any(o.needs_review for o in track.observations)
            or len(agreeing) < 2
            or avg < 0.55
        )
        return {
            "verified": len(agreeing) >= 2 and avg >= 0.45,
            "needs_review": bool(needs),
            "text": canonical,
            "support_frames": len(agreeing),
            "observations": len(track.observations),
            "mean_confidence": round(avg, 4),
            "verification_queue": bool(needs),
        }
