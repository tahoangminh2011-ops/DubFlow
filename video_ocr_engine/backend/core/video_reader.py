# File: /workspace/video_ocr_engine/backend/core/video_reader.py
import cv2
import logging
from models.schemas import VideoMetadata

class VideoReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
    def get_metadata(self) -> VideoMetadata:
        return VideoMetadata(
            duration=self.duration,
            width=self.width,
            height=self.height,
            fps=self.fps,
            timebase=f"1/{int(self.fps)}" if self.fps else "UNKNOWN",
            vfr=False, # Simplified for mock, real VFR needs ffprobe
            total_frames=self.total_frames
        )
        
    def get_frame(self, frame_idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return ret, frame, timestamp
        
    def release(self):
        self.cap.release()
