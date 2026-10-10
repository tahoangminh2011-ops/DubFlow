from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

@dataclass(frozen=True)
class VideoMetadata:
    duration: float
    width: int
    height: int
    timebase: str
    fps: float
    vfr: bool
    frame_count: int
    pts: Tuple[float, ...] = ()

    def time_for_frame(self, frame_index: int) -> float:
        if self.pts and 0 <= frame_index < len(self.pts):
            return float(self.pts[frame_index])
        return float(frame_index) / max(self.fps, 1e-9)

@dataclass
class TextDetection:
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    frame_index: int
    timestamp: float
    region_id: str = "roi"
    line_index: int = 0
    source: str = "local"
    needs_review: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OCRObservation:
    frame_index: int
    timestamp: float
    detections: List[TextDetection]
    sample_kind: str = "adaptive"

@dataclass
class CandidateWindow:
    start_time: float
    end_time: float
    reason: str
    preferred_bbox: Optional[Tuple[float, float, float, float]] = None
    priority: int = 1
    # The candidate generator must preserve the region that triggered this window.
    # A full-frame/global cue must never fall back to a user ROI that may be misplaced.
    region_id: str = "roi"

@dataclass
class TrackSegment:
    tracking_id: int
    frame_start: int
    frame_end: int
    start_time: float
    end_time: float
    text: str
    confidence: float
    bbox: List[float]
    line_count: int
    classification: str = "unknown"
    needs_review: bool = False
    region_id: str = "roi"
    source: str = "local"
    verification: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OCRRunResult:
    segments: List[TrackSegment]
    audit: Dict[str, Any]
    metadata: VideoMetadata
