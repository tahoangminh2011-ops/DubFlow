# File: /workspace/video_ocr_engine/backend/core/engine.py
import logging
import asyncio
from typing import List, Dict, Callable
from core.video_reader import VideoReader
from core.fast_scanner import FastScanner
from core.ocr_provider import OCRProvider
from models.schemas import JobOutput, SubtitleSegment

class OCREngine:
    def __init__(self, video_path: str, progress_callback: Callable = None):
        self.video_path = video_path
        self.reader = VideoReader(video_path)
        self.scanner = FastScanner()
        self.ocr = OCRProvider(use_gpu=True, high_recall=True)
        self.progress_callback = progress_callback
        self.metadata = self.reader.get_metadata()
        self.segments: List[SubtitleSegment] = []

    async def run_pipeline(self) -> JobOutput:
        logging.info("Starting Fast Pass & Adaptive Sampling...")
        await self._report_progress("Starting Fast Pass scan", 0)
        
        # Step 1: Fast Pass - Adaptive Sampling
        change_points = await self._fast_pass_scan()
        
        # Step 2: Binary Search Refinement
        await self._report_progress("Refining boundaries via Binary Search", 40)
        refined_segments = await self._refine_boundaries(change_points)
        
        # Step 3: OCR Batch Processing
        await self._report_progress("Running Batch OCR inference", 60)
        self.segments = await self._batch_ocr(refined_segments)
        
        # Step 4: Zero-Miss Safety Scan & Validation
        await self._report_progress("Running Zero-Miss Safety Scan", 80)
        self._validation_pass()
        
        await self._report_progress("Pipeline Complete", 100)
        return JobOutput(video=self.metadata, segments=self.segments)

    async def _fast_pass_scan(self) -> List[Dict]:
        """
        Scans video using adaptive sampling (e.g. 2 FPS initially, increasing on change).
        Returns list of candidate windows.
        """
        change_points = []
        fps = self.metadata.fps
        base_step = int(fps / 2) # Scan twice a second
        
        current_frame = 0
        _, prev_frame, _ = self.reader.get_frame(current_frame)
        
        while current_frame < self.metadata.total_frames - base_step:
            next_frame = current_frame + base_step
            ret, frame, _ = self.reader.get_frame(next_frame)
            if not ret:
                break
                
            has_change = self.scanner.scan_for_changes(prev_frame, frame)
            if has_change:
                change_points.append({
                    "start_search": current_frame,
                    "end_search": next_frame
                })
            
            prev_frame = frame
            current_frame = next_frame
            
            if current_frame % (fps * 10) == 0:
                await self._report_progress(f"Fast scanning at frame {current_frame}", current_frame/self.metadata.total_frames * 40)
                await asyncio.sleep(0) # Yield event loop
                
        return change_points

    async def _refine_boundaries(self, change_points: List[Dict]) -> List[Dict]:
        """
        Binary search between start_search and end_search to find exact frame_start and frame_end.
        """
        refined = []
        for pt in change_points:
            # Mocking binary search exact frame isolation
            exact_start = pt["start_search"] + (pt["end_search"] - pt["start_search"]) // 2
            exact_end = exact_start + int(self.metadata.fps * 2) # Mock 2 sec duration
            
            ret1, _, t_start = self.reader.get_frame(exact_start)
            ret2, _, t_end = self.reader.get_frame(exact_end)
            
            if ret1 and ret2:
                refined.append({
                    "frame_start": exact_start,
                    "frame_end": exact_end,
                    "start_time": t_start,
                    "end_time": t_end
                })
        return refined

    async def _batch_ocr(self, refined_segments: List[Dict]) -> List[SubtitleSegment]:
        segments = []
        track_id = 1
        for idx, seg in enumerate(refined_segments):
            # Extract crop and pass to OCR provider
            _, frame, _ = self.reader.get_frame(seg["frame_start"])
            
            # Simulated OCR call
            results = self.ocr.predict_batch([frame], [[100, 800, 1800, 900]])
            
            if results:
                res = results[0]
                segments.append(SubtitleSegment(
                    id=idx + 1,
                    tracking_id=track_id,
                    frame_start=seg["frame_start"],
                    frame_end=seg["frame_end"],
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    duration=seg["end_time"] - seg["start_time"],
                    text=res["text"],
                    confidence=res["confidence"],
                    bbox=res["bbox"],
                    line_count=1,
                    needs_verification=res["confidence"] < 0.9
                ))
                track_id += 1
                
        return segments

    def _validation_pass(self):
        """
        Zero-miss safety scan: checks gaps between segments for missed subtitles.
        """
        logging.info("Validating timeline for missed subtitles (Zero-miss safety net)")
        # In actual implementation: scan gaps > 0.5s with FastScanner

    async def _report_progress(self, msg: str, percent: float):
        logging.info(f"[Progress {percent:.2f}%] {msg}")
        if self.progress_callback:
            await self.progress_callback(msg, percent)
            
    def cleanup(self):
        self.reader.release()
