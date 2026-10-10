from pydantic import BaseModel
from typing import List, Optional, Tuple, Any

class VideoMetadata(BaseModel):
    duration: float
    width: int
    height: int
    fps: float
    timebase: str
    vfr: bool
    total_frames: int

class SubtitleSegment(BaseModel):
    id: int
    tracking_id: int
    frame_start: int
    frame_end: int
    start_time: float
    end_time: float
    duration: float
    text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    line_count: int
    needs_verification: bool = False

class JobOutput(BaseModel):
    video: VideoMetadata
    segments: List[SubtitleSegment]
