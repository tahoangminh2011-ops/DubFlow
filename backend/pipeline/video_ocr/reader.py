from __future__ import annotations
import json, math, os, subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
from .models import VideoMetadata

class VideoReader:
    def __init__(self, path: Path, ffprobe: str = "ffprobe"):
        self.path = Path(path)
        self.ffprobe = ffprobe
        self.metadata: Optional[VideoMetadata] = None
        self._cap = None
        self._last_index = -1

    @staticmethod
    def _fps(value: str) -> float:
        try: return float(Fraction(value))
        except Exception:
            try: return float(value)
            except Exception: return 30.0

    def probe(self, with_pts: bool = True) -> VideoMetadata:
        cmd = [self.ffprobe, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,duration,time_base,avg_frame_rate,r_frame_rate,nb_frames,start_time",
               "-of", "json", str(self.path)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        st = json.loads(p.stdout).get("streams", [{}])[0]
        width = int(st.get("width") or 0); height = int(st.get("height") or 0)
        duration = float(st.get("duration") or 0.0)
        avg_rate = str(st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1")
        raw_rate = str(st.get("r_frame_rate") or avg_rate)
        fps = self._fps(avg_rate)
        nb = int(st.get("nb_frames") or 0)
        tb = str(st.get("time_base") or "1/1000")
        start_time = float(st.get("start_time") or 0.0) if str(st.get("start_time") or "").strip() not in {"", "N/A"} else 0.0
        pts: List[float] = []
        # Many MP4/MKV containers omit nb_frames even for ordinary CFR recordings. Treating
        # nb_frames=0 as proof of VFR caused ffprobe to enumerate every frame before OCR started
        # (minutes of startup latency on long videos). Compare numeric rates and inspect short PTS
        # windows first; only enumerate all PTS if the stream looks variable or the user requests it.
        force_pts = os.getenv("OCR_FORCE_PTS", "0").strip().lower() in {"1","true","yes"}
        avg_fps = self._fps(avg_rate); nominal_fps = self._fps(raw_rate)
        rate_mismatch = abs(avg_fps - nominal_fps) > max(0.02, 0.001 * max(avg_fps, nominal_fps, 1.0))
        sampled_vfr = False
        if with_pts and not force_pts and not rate_mismatch and duration > 0:
            sampled_vfr = self._sampled_pts_look_vfr(duration)
        likely_vfr = rate_mismatch or sampled_vfr
        if with_pts and (force_pts or likely_vfr):
            pts = self._read_pts()
        if pts:
            duration = max(duration, pts[-1])
            if not nb: nb = len(pts)
        deltas = [pts[i]-pts[i-1] for i in range(1, len(pts)) if pts[i] > pts[i-1]]
        vfr = bool(deltas and (max(deltas)-min(deltas) > max(0.001, sum(deltas)/len(deltas)*0.01))) if pts else bool(sampled_vfr)
        if not pts and start_time:
            # Keep the stream start offset visible in metadata; individual frame timestamps continue
            # to use the zero-based media timebase unless PTS is explicitly loaded.
            duration = max(0.0, duration)
        self.metadata = VideoMetadata(duration=duration, width=width, height=height,
                                       timebase=tb, fps=fps, vfr=vfr, frame_count=max(nb, len(pts)),
                                       pts=tuple(pts))
        return self.metadata

    def _sampled_pts_look_vfr(self, duration: float) -> bool:
        """Inspect brief start/middle/end PTS windows without decoding the full stream.

        If the sampled frame intervals vary materially, the caller falls back to a complete PTS
        table so boundary calculations remain frame-accurate for VFR media.
        """
        starts = sorted(set([0.0, max(0.0, float(duration) * 0.50), max(0.0, float(duration) - 2.0)]))
        deltas: List[float] = []
        for start in starts:
            try:
                cmd = [self.ffprobe, "-v", "error", "-read_intervals", f"{start:.3f}%+1.25",
                       "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time,pkt_pts_time",
                       "-of", "csv=p=0", str(self.path)]
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=True)
            except Exception:
                continue
            values: List[float] = []
            for line in p.stdout.splitlines():
                vals = [x.strip() for x in line.split(",") if x.strip() and x.strip() != "N/A"]
                if not vals:
                    continue
                try:
                    value = float(vals[0])
                except Exception:
                    continue
                if not values or value > values[-1]:
                    values.append(value)
            for a, b in zip(values, values[1:]):
                delta = b - a
                # Ignore seek gaps and timestamp discontinuities between interval boundaries.
                if 0.0001 < delta < 0.25:
                    deltas.append(delta)
        if len(deltas) < 12:
            return False
        mean = sum(deltas) / len(deltas)
        return (max(deltas) - min(deltas)) > max(0.001, mean * 0.01)

    def _read_pts(self) -> List[float]:
        cmd = [self.ffprobe, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "frame=best_effort_timestamp_time,pkt_pts_time",
               "-of", "csv=p=0", str(self.path)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
        except Exception:
            return []
        out: List[float] = []
        for line in p.stdout.splitlines():
            parts = [x.strip() for x in line.split(",") if x.strip() and x.strip() != "N/A"]
            if not parts: continue
            try: out.append(float(parts[0]))
            except Exception: continue
        return out

    def nearest_frame(self, timestamp: float) -> int:
        if self.metadata is None: self.probe(with_pts=True)
        pts = self.metadata.pts
        if not pts: return max(0, int(round(timestamp*self.metadata.fps)))
        lo, hi = 0, len(pts)-1
        while lo < hi:
            mid = (lo+hi)//2
            if pts[mid] < timestamp: lo = mid+1
            else: hi = mid
        if lo > 0 and abs(pts[lo-1]-timestamp) <= abs(pts[lo]-timestamp): return lo-1
        return lo

    def frame_time(self, index: int) -> float:
        if self.metadata is None: self.probe(with_pts=True)
        return self.metadata.time_for_frame(index)

    def get_frame(self, index: int):
        index = max(0, int(index))
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.path))
            self._last_index = -1
        if not self._cap.isOpened(): return None

        # Boundary refinement often probes nearby frames. Read forward sequentially when possible
        # instead of forcing an expensive decoder seek for every binary-search probe.
        if self._last_index >= 0 and index >= self._last_index and index - self._last_index <= 6:
            frame = None
            while self._last_index < index:
                ok, frame = self._cap.read()
                if not ok:
                    self._last_index = -1
                    return None
                self._last_index += 1
            return frame

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if ok:
            self._last_index = index
            return frame
        self._last_index = -1
        return None

    def iter_selected_frames(self, indices):
        """Yield selected frame indices/images in forward order using one decoder pass.

        Only selected frames are yielded/copied; intermediate frames are decoded transiently.
        This is substantially cheaper than a random seek for every OCR sample.
        """
        targets=iter(sorted(set(int(i) for i in indices if int(i) >= 0)))
        try:
            target=next(targets)
        except StopIteration:
            return
        if self._cap is not None:
            try:self._cap.release()
            except Exception:pass
            self._cap=None
            self._last_index=-1
        cap=cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            return
        current=0
        try:
            while target is not None:
                ok,frame=cap.read()
                if not ok:
                    break
                if current==target:
                    yield current,frame.copy()
                    try: target=next(targets)
                    except StopIteration: target=None
                current+=1
        finally:
            try:cap.release()
            except Exception:pass

    def close(self):
        if self._cap is not None:
            try: self._cap.release()
            except Exception: pass
            self._cap = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
