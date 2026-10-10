from __future__ import annotations
from typing import Callable, Optional, Tuple
from .models import VideoMetadata, TrackSegment
from .reader import VideoReader


class BoundaryRefiner:
    """Frame-accurate refinement with logarithmic search + local OCR validation.

    The logarithmic search uses a cheap visual predicate. Only the final few frames around the
    predicted boundary are checked by an optional OCR verifier. This keeps frame accuracy without
    running OCR at every binary-search probe.
    """
    def __init__(self, reader: VideoReader, meta: VideoMetadata, predicate: Callable,
                 max_search_seconds: float = 2.0, verify_fn: Optional[Callable[[int], Optional[bool]]] = None,
                 verify_radius_frames: int = 3, start_predicate: Optional[Callable] = None,
                 end_predicate: Optional[Callable] = None):
        self.reader = reader
        self.meta = meta
        self.predicate = predicate
        self.start_predicate = start_predicate or predicate
        self.end_predicate = end_predicate or predicate
        self.max_search_frames = max(6, int(round(max_search_seconds * max(meta.fps, 1.0))))
        self.verify_fn = verify_fn
        self.verify_radius_frames = max(1, int(verify_radius_frames))

    def _bin_first(self, lo: int, hi: int, fn: Callable[[int], bool]) -> int:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fn(mid): hi = mid
            else: lo = mid
        return hi

    def _bin_last(self, lo: int, hi: int, fn: Callable[[int], bool]) -> int:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fn(mid): lo = mid
            else: hi = mid
        return lo

    def _bracket_last_true(self, start_true: int, limit: int, fn: Callable[[int], bool]):
        lo = int(start_true); step = 1
        while lo < limit:
            probe = min(limit, lo + step)
            if fn(probe):
                lo = probe
                if lo >= limit: return None
                step *= 2
            else:
                return lo, probe
        return None

    def _bracket_first_true(self, start_false: int, limit: int, fn: Callable[[int], bool]):
        hi = int(start_false); step = 1
        while hi > limit:
            probe = max(limit, hi - step)
            if fn(probe):
                return probe, hi
            hi = probe
            if hi <= limit: break
            step *= 2
        return None

    def _local_verify_start(self, center: int, lower: int, upper: int) -> Optional[int]:
        if not self.verify_fn:
            return None
        vals=[]
        for i in range(max(lower, center-self.verify_radius_frames), min(upper, center+self.verify_radius_frames)+1):
            try:
                v=self.verify_fn(int(i))
            except Exception:
                v=None
            if v is not None:
                vals.append((i,bool(v)))
        positives=[i for i,v in vals if v]
        return min(positives) if positives else None

    def _local_verify_end(self, center: int, lower: int, upper: int) -> Optional[int]:
        if not self.verify_fn:
            return None
        vals=[]
        for i in range(max(lower, center-self.verify_radius_frames), min(upper, center+self.verify_radius_frames)+1):
            try:
                v=self.verify_fn(int(i))
            except Exception:
                v=None
            if v is not None:
                vals.append((i,bool(v)))
        positives=[i for i,v in vals if v]
        return max(positives) if positives else None

    def refine(self, seg: TrackSegment, before_frame: Optional[int], after_frame: Optional[int]) -> TrackSegment:
        fs=max(0,int(seg.frame_start)); fe=max(fs,int(seg.frame_end))
        start_active=lambda i: bool(self.start_predicate(int(i), seg.text, tuple(seg.bbox)))
        end_active=lambda i: bool(self.end_predicate(int(i), seg.text, tuple(seg.bbox)))

        if start_active(fs):
            lo_false=max(0,int(before_frame)) if before_frame is not None and before_frame < fs else max(0,fs-self.max_search_frames)
            if not start_active(lo_false):
                fs=self._bin_first(lo_false,fs,start_active)
            else:
                bracket=self._bracket_first_true(fs,max(0,fs-self.max_search_frames),start_active)
                if bracket: fs=self._bin_first(bracket[0],bracket[1],start_active)
        if self.verify_fn:
            corrected=self._local_verify_start(fs,max(0,fs-self.verify_radius_frames-1),min(fe,fs+self.verify_radius_frames+1))
            if corrected is not None:
                fs=corrected

        if end_active(fe):
            limit=min(self.meta.frame_count-1 if self.meta.frame_count else fe+self.max_search_frames,
                      int(after_frame) if after_frame is not None and after_frame>fe else fe+self.max_search_frames)
            bracket=self._bracket_last_true(fe,limit,end_active)
            if bracket:
                fe=self._bin_last(bracket[0],bracket[1],end_active)
            elif after_frame is not None and int(after_frame)>fe and end_active(int(after_frame)):
                fe=int(after_frame)
        if self.verify_fn:
            corrected=self._local_verify_end(fe,max(fs,fe-self.verify_radius_frames-1),min(self.meta.frame_count-1 if self.meta.frame_count else fe+self.verify_radius_frames+1,fe+self.verify_radius_frames+1))
            if corrected is not None:
                fe=max(fs,corrected)

        st=self.meta.time_for_frame(fs); en=self.meta.time_for_frame(fe)
        return TrackSegment(seg.tracking_id,fs,fe,st,en,seg.text,seg.confidence,seg.bbox,seg.line_count,
                            seg.classification,seg.needs_review,seg.region_id,seg.source,seg.verification)
