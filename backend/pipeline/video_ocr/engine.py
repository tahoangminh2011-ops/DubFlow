from __future__ import annotations
import math, os, time, logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2, numpy as np
from .benchmark import BenchmarkManager
from .boundary import BoundaryRefiner
from .classifier import SubtitleClassifier
from .deduplicator import Deduplicator
from .exporter import Exporter
from .models import OCRObservation, OCRRunResult, TextDetection, TrackSegment, VideoMetadata
from .color import ColorProfile
from .timeline import TimelineBuilder
from .normalizer import normalize_text, similarity
from .ocr_provider import create_ocr_provider
from .reader import VideoReader
from .sampler import AdaptiveFrameSampler
from .tracker import TextTracker, TextTrack, TrackObservation
from .validator import TimelineValidator
from .verifier import OCRVerifier
from .temporal_recognizer import TemporalOCRProcessor

log = logging.getLogger(__name__)

class VideoOCREngine:
    BUILD_ID = "VideoOCR-V6.4-HOTFIX-20261010"

    """High-recall video OCR + temporal subtitle tracking engine.

    Scope is intentionally limited to OCR, subtitle detection/tracking, timeline,
    validation and JSON/SRT export. Uncertain detections remain in the verification queue.
    """
    def __init__(self, ffmpeg="ffmpeg", ffprobe="ffprobe", config: Optional[Dict[str, Any]] = None):
        self.cfg = config or {}
        self.ffmpeg = ffmpeg; self.ffprobe = ffprobe
        self.scan_fps = float(self.cfg.get("scan_fps", 24.0))
        self.scan_width = int(self.cfg.get("scan_width", 320))
        self.scan_workers = int(self.cfg.get("scan_workers", 1))
        self.recovery = float(self.cfg.get("recovery_seconds", 1.5))
        self.burst_fps = float(self.cfg.get("burst_fps", 30.0))
        self.batch_size = int(self.cfg.get("batch_size", 16))
        self.mode = str(self.cfg.get("mode", "HIGH_RECALL")).upper()

    @staticmethod
    def _roi(region, w, h):
        x = float(region.get("x", 0)); y = float(region.get("y", 65))
        rw = float(region.get("w", 100)); rh = float(region.get("h", 35))
        x0 = max(0, min(w - 1, int(round(x * w / 100))))
        y0 = max(0, min(h - 1, int(round(y * h / 100))))
        x1 = max(x0 + 1, min(w, int(round((x + rw) * w / 100))))
        y1 = max(y0 + 1, min(h, int(round((y + rh) * h / 100))))
        return x0, y0, x1, y1

    def _emit(self, cb, payload):
        if cb:
            try:
                cb(dict(payload))
            except Exception:
                log.exception("OCR progress callback failed")

    @staticmethod
    def _iou(a, b):
        if not a or not b: return 0.0
        ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
        ix = max(0, min(ax1, bx1) - max(ax0, bx0)); iy = max(0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        aa = max(1, (ax1 - ax0) * (ay1 - ay0)); ab = max(1, (bx1 - bx0) * (by1 - by0))
        return inter / max(1, aa + ab - inter)

    @staticmethod
    def _fast_subtitle_band(crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Find a conservative text-band proposal for sparse recovery frames.

        This is a crop accelerator only. If the horizontal gradient concentration is weak,
        callers must keep the full ROI so recall is never traded for speed.
        """
        if crop is None or crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        h, w = gray.shape[:2]
        if h < 24 or w < 80:
            return None
        gx = np.abs(gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16)).mean(axis=1)
        gy = np.abs(gray[1:, :].astype(np.int16) - gray[:-1, :].astype(np.int16)).mean(axis=1)
        row = gx * 0.72 + np.pad(gy, (1, 0), mode="edge") * 0.28
        win = max(8, min(int(h * 0.16), int(h * 0.34)))
        sm = np.convolve(row, np.ones(win, np.float32) / win, mode="same")
        peak = float(sm.max()); mean = float(sm.mean())
        if peak <= mean * 1.35:
            return None
        cy = int(np.argmax(sm))
        pad_y = max(10, int(h * 0.11))
        y0 = max(0, cy - win // 2 - pad_y); y1 = min(h, cy + win // 2 + pad_y)
        if (y1 - y0) < 0.16 * h or (y1 - y0) > 0.62 * h:
            return None
        # Keep the entire width: words can start/end far from the projection peak.
        return (0, y0, w, y1)

    @staticmethod
    def _glyph_signature(crop: np.ndarray) -> np.ndarray:
        """Cheap target-glyph signature used only during boundary probes."""
        if crop is None or crop.size == 0:
            return np.zeros((32, 96), np.uint8)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = cv2.resize(gray, (96, 32), interpolation=cv2.INTER_AREA)
        blur = cv2.blur(gray, (3, 3))
        local = cv2.absdiff(gray, blur)
        gx = np.abs(gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16))
        gy = np.abs(gray[1:, :].astype(np.int16) - gray[:-1, :].astype(np.int16))
        edge = np.zeros_like(gray, dtype=bool)
        edge[:,1:] |= gx >= 8
        edge[1:,:] |= gy >= 8
        return (edge & (local >= 7)).astype(np.uint8) * 255

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        aa = a > 80; bb = b > 80
        inter = float(np.count_nonzero(aa & bb)); union = float(np.count_nonzero(aa | bb))
        return inter / max(1.0, union)

    @staticmethod
    def _mask_corr(a: np.ndarray, b: np.ndarray) -> float:
        x = a.astype(np.float32).ravel(); y = b.astype(np.float32).ravel()
        x -= x.mean(); y -= y.mean()
        den = float(np.linalg.norm(x) * np.linalg.norm(y))
        return 0.0 if den <= 1e-6 else float(np.dot(x, y) / den)

    def _boundary_active(self, reader: VideoReader, seg: TrackSegment, frame_index: int,
                         color_profile: ColorProfile, anchor_signature: Optional[np.ndarray] = None,
                         negative_signature: Optional[np.ndarray] = None) -> bool:
        frame = reader.get_frame(frame_index)
        if frame is None: return False
        bx0, by0, bx1, by1 = map(float, seg.bbox)
        x0 = max(0, int(math.floor(min(bx0, bx1))) - 3)
        y0 = max(0, int(math.floor(min(by0, by1))) - 3)
        x1 = min(frame.shape[1], int(math.ceil(max(bx0, bx1))) + 3)
        y1 = min(frame.shape[0], int(math.ceil(max(by0, by1))) + 3)
        if x1 <= x0 or y1 <= y0: return False
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 160)
        edge_ratio = float(np.mean(edges > 0))
        sig = self._glyph_signature(crop)
        text_ratio = float(np.mean(sig > 80))
        active = edge_ratio >= float(self.cfg.get("boundary_edge_threshold", 0.020))
        active = active or (0.003 <= text_ratio <= 0.70 and edge_ratio >= 0.008)
        if anchor_signature is not None:
            iou = self._mask_iou(anchor_signature, sig)
            corr = self._mask_corr(anchor_signature, sig)
            positive = max(iou, corr)
            target_match = positive >= float(self.cfg.get("boundary_signature_min", 0.30))
            if negative_signature is not None:
                niou = self._mask_iou(negative_signature, sig)
                ncorr = self._mask_corr(negative_signature, sig)
                negative = max(niou, ncorr)
                # At a same-position subtitle change, classify the frame by which visual glyph
                # signature it matches. This prevents the next subtitle from reactivating the end.
                target_match = target_match and positive >= (negative + float(self.cfg.get("boundary_signature_margin", 0.08)))
            active = active and target_match
        if color_profile.enabled:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
            th, ts, tv = [float(x) for x in color_profile.hsv]
            hd = np.minimum(np.abs(hsv[:, :, 0] - th), 180.0 - np.abs(hsv[:, :, 0] - th))
            dist = np.sqrt((hd * 1.1) ** 2 + (hsv[:, :, 1] - ts) ** 2 + (hsv[:, :, 2] - tv) ** 2)
            color_ratio = float(np.mean(dist <= float(color_profile.tolerance)))
            color_ok = color_ratio >= float(self.cfg.get("boundary_color_ratio", 0.015))
            if anchor_signature is not None:
                # Color is supporting evidence only; a slightly anti-aliased/blurred frame must
                # not invalidate an otherwise strong glyph match.
                active = active and (color_ok or text_ratio >= 0.004 or edge_ratio >= 0.012)
            else:
                active = active or color_ok
        return bool(active)

    def _verification_requests(self, tracks: List[TextTrack], verification_cfg: Dict[str, Any]):
        max_per_track = int(verification_cfg.get("max_samples_per_track", 4))
        max_total = max(1, int(verification_cfg.get("max_total_samples", 64)))
        out = []
        # Prioritize uncertain/singleton tracks but cap synchronous work. Any excess track remains
        # in the timeline/review queue; it is not deleted merely because verification was deferred.
        ordered_tracks = sorted(tracks, key=lambda tr: (
            0 if any(getattr(o, "needs_review", False) for o in tr.observations) else 1,
            len(tr.observations),
            min((float(o.confidence) for o in tr.observations), default=1.0),
            tr.tracking_id,
        ))
        for tr in ordered_tracks:
            if len(out) >= max_total:
                break
            if len(tr.observations) == 0: continue
            # HIGH_RECALL always verifies weak/singleton tracks; balanced modes only verify flagged tracks.
            flagged = len(tr.observations) < 2 or any(o.needs_review for o in tr.observations)
            if not flagged: continue
            first, last = tr.first, tr.last
            times = [first.timestamp, last.timestamp]
            if last.timestamp - first.timestamp > 0.08:
                times.append((first.timestamp + last.timestamp) / 2.0)
            if max_per_track >= 4 and last.timestamp - first.timestamp > 0.30:
                q = first.timestamp + (last.timestamp - first.timestamp) * 0.25
                times.append(q)
            seen = set()
            for ts in times:
                key = round(float(ts), 4)
                if key in seen: continue
                seen.add(key)
                out.append((tr, float(ts)))
                if len([x for x in out if x[0] is tr]) >= max_per_track: break
        return out

    def _targeted_verify(self, path: Path, meta: VideoMetadata, tracks: List[TextTrack],
                         provider_impl, region, color_profile, on_event):
        """Cheap verification using recognition-only crops.

        Verification must not rerun full detector OCR over an entire ROI. A weak track is verified
        directly on its latest localized bbox; this keeps HIGH_RECALL while avoiding another heavy
        detector pass for every singleton track.
        """
        requests = self._verification_requests(tracks, self.cfg.get("verification", {}))
        if not requests or not hasattr(provider_impl, "recognize_text_only_batch"):
            return 0
        images = []; metas = []; owners = []
        with VideoReader(path, self.ffprobe) as br:
            for tr, ts in requests:
                fi = br.nearest_frame(ts)
                frame = br.get_frame(fi)
                if frame is None or not tr.observations:
                    continue
                # Use the most recent stable bbox, padded only slightly. This is deliberately not
                # the whole configured ROI; background leakage is a major source of OCR errors.
                bx = tr.last.bbox
                x0, y0, x1, y1 = map(float, bx)
                bw = max(2.0, x1 - x0); bh = max(2.0, y1 - y0)
                pad_x = max(8.0, bw * 0.08); pad_y = max(5.0, bh * 0.38)
                ax0 = max(0, int(math.floor(x0 - pad_x))); ay0 = max(0, int(math.floor(y0 - pad_y)))
                ax1 = min(frame.shape[1], int(math.ceil(x1 + pad_x))); ay1 = min(frame.shape[0], int(math.ceil(y1 + pad_y)))
                crop = frame[ay0:ay1, ax0:ax1]
                if crop.size:
                    images.append(crop); metas.append((fi, br.frame_time(fi))); owners.append(tr)
        if not images:
            return 0

        pairs = provider_impl.recognize_text_only_batch(
            images,
            [10_000_000_000 + i for i in range(len(images))],
            [x[1] for x in metas],
            batch_size=self.batch_size,
        )
        added = 0
        for i, (tr, pair) in enumerate(zip(owners, pairs)):
            _sid, ts, text, score = pair
            text = normalize_text(text)
            if not text:
                continue
            sim = max((similarity(text, o.text) for o in tr.observations), default=0.0)
            # Keep low-score text if it is textually plausible: HIGH_RECALL never deletes it.
            if sim >= 0.48 or float(score) >= 0.55:
                tr.add(TrackObservation(
                    int(metas[i][0]), float(ts), text, float(score), tuple(tr.last.bbox),
                    float(score) < 0.55, "verification", tr.region_id,
                ))
                added += 1
        self._emit(on_event, {
            "phase": "verification", "phase_done": added, "phase_total": len(requests),
            "verification_samples": len(requests), "verified_observations": added, "overall_percent": 72,
        })
        return added

    @staticmethod
    def _trace_bracket(trace, target_time: float, fps: float, forward: bool, max_seconds: float = 1.5):
        """Find the nearest false/true transition bracket from the cached fast-scan trace."""
        if not trace:
            return None
        t = float(target_time)
        max_t = t + max_seconds if forward else t - max_seconds
        idx = min(range(len(trace)), key=lambda i: abs(float(trace[i].timestamp) - t))
        if forward:
            lo = idx
            hi = min(len(trace) - 1, idx + max(2, int(round(max_seconds * max(fps, 1.0)))))
            return trace[lo].frame_index, trace[hi].frame_index
        hi = idx
        lo = max(0, idx - max(2, int(round(max_seconds * max(fps, 1.0)))))
        return trace[lo].frame_index, trace[hi].frame_index

    def _boundary_probe(self, reader: VideoReader, seg: TrackSegment, frame_index: int,
                        provider_impl, color_profile) -> Optional[bool]:
        """Target-text check used only around a refined boundary (small local crop)."""
        frame = reader.get_frame(int(frame_index))
        if frame is None or not hasattr(provider_impl, "recognize_text_only_batch"):
            return None
        x0, y0, x1, y1 = map(float, seg.bbox)
        bw = max(2.0, x1 - x0); bh = max(2.0, y1 - y0)
        px = max(8.0, bw * 0.10); py = max(5.0, bh * 0.45)
        ax0 = max(0, int(math.floor(x0-px))); ay0 = max(0, int(math.floor(y0-py)))
        ax1 = min(frame.shape[1], int(math.ceil(x1+px))); ay1 = min(frame.shape[0], int(math.ceil(y1+py)))
        crop = frame[ay0:ay1, ax0:ax1]
        if crop.size == 0:
            return None
        try:
            pair = provider_impl.recognize_text_only_batch(
                [crop], [10_000_000_001], [reader.frame_time(int(frame_index))], batch_size=1,
            )
            if not pair:
                return None
            text = normalize_text(pair[0][2]); score = float(pair[0][3])
            target = normalize_text(seg.text)
            if not text or not target:
                return False
            sim = similarity(text, target)
            # Boundary verification must be target-aware: a different text with high OCR
            # confidence is not evidence that the target subtitle is still visible.
            return bool(
                sim >= 0.82
                or (sim >= float(self.cfg.get("boundary_text_similarity", 0.66)) and score >= 0.55)
            )
        except Exception:
            return None

    def run(self, video_path: Path, region: Dict[str, float], model: str = "", provider: str = "local",
            color_profile: Optional[Dict[str, Any]] = None,
            on_event: Optional[Callable[[Dict], None]] = None,
            on_segment: Optional[Callable[[TrackSegment, int, int], None]] = None,
            cancel_cb: Optional[Callable[[], bool]] = None) -> OCRRunResult:
        started = time.monotonic(); bench = BenchmarkManager(); path = Path(video_path)
        log.info("[VIDEO OCR BUILD] %s", self.BUILD_ID)
        job_dir = Path(self.cfg.get("job_dir", path.parent / ".ocr_job")); job_dir.mkdir(parents=True, exist_ok=True)
        reader = VideoReader(path, self.ffprobe); meta = reader.probe(with_pts=True)
        self._emit(on_event,{"type":"metadata","phase":"metadata","phase_done":1,"phase_total":1,"overall_percent":2,"video":meta.__dict__})
        if meta.width <= 0 or meta.height <= 0: raise RuntimeError("Không đọc được metadata video")
        try:
            # Initialize OCR before the fast scan so the job reports the actual backend/GPU state
            # immediately. When GPU is explicitly requested, fail fast instead of spending minutes
            # scanning and only discovering later that OCR is CPU-bound.
            color = ColorProfile.from_dict(color_profile)
            provider_impl = create_ocr_provider(
                model, provider, color_profile=color, lang=self.cfg.get("lang","vi"),
                backend=self.cfg.get("backend","auto"), use_gpu=self.cfg.get("use_gpu","auto"),
                gpu_id=int(self.cfg.get("gpu_id",0)), model_dir=self.cfg.get("model_dir","")
            )
            # Provider construction is lazy. Initialize the actual sessions before reporting GPU state.
            if hasattr(provider_impl, "_make_engine"):
                provider_impl._make_engine(provider_impl)
            latin_provider = None
            if str(self.cfg.get("lang","auto")).lower() == "auto":
                try:
                    latin_provider = create_ocr_provider(
                        model, provider, color_profile=color, lang="latin",
                        backend=self.cfg.get("backend","auto"), use_gpu=self.cfg.get("use_gpu","auto"),
                        gpu_id=int(self.cfg.get("gpu_id",0)), model_dir=self.cfg.get("model_dir","")
                    )
                    latin_provider._make_engine(latin_provider)
                except Exception as exc:
                    log.info("[OCR V5] latin secondary unavailable: %s", exc)
                    latin_provider = None
            gpu_active = bool(getattr(provider_impl, "gpu_active", False))
            gpu_requested = str(self.cfg.get("use_gpu","auto")).lower()
            self._emit(on_event, {"phase":"ocr_backend", "overall_percent":2.5,
                                 "ocr_backend":provider_impl.engine_name,
                                 "ocr_backend_kind":getattr(provider_impl, "_engine_kind", None),
                                 "ocr_gpu_requested":gpu_requested,
                                 "ocr_gpu_active":gpu_active,
                                 "runtime_providers":list(getattr(provider_impl, "runtime_providers", []) or []),
                                 "lang":self.cfg.get("lang","vi")})
            log.info("[OCR V2] backend=%s kind=%s lang_requested=%s lang_resolved=%s gpu_active=%s providers=%s",
                     provider_impl.engine_name, getattr(provider_impl, "_engine_kind", None),
                     getattr(provider_impl, "requested_lang", self.cfg.get("lang", "auto")),
                     getattr(provider_impl, "lang", self.cfg.get("lang", "auto")),
                     gpu_active, getattr(provider_impl, "runtime_providers", []))
            if gpu_requested in {"1","true","yes","cuda","gpu"} and not gpu_active:
                raise RuntimeError("OCR_USE_GPU yêu cầu GPU nhưng OCR backend không kích hoạt GPU. Xem phase=ocr_backend/runtime_providers để kiểm tra ONNX Runtime/Paddle.")

            scan_start = time.monotonic()
            sampler = AdaptiveFrameSampler(self.ffmpeg, self.scan_fps, self.scan_width, self.scan_workers,
                                           float(self.cfg.get("scan_diff_threshold", 0.018)),
                                           float(self.cfg.get("global_textlike_threshold", 0.0075)),
                                           float(self.cfg.get("global_line_density_threshold", 0.05)),
                                           str(self.cfg.get("hwaccel", "auto")))
            scan_progress = {"done": 0, "total": 0, "last_bench": 0.0}
            def on_scan_progress(done, total, stats):
                scan_progress["done"] = int(done); scan_progress["total"] = int(total)
                now = time.monotonic()
                if now - float(scan_progress.get("last_bench", 0.0)) >= 2.0:
                    bench.cpu(); scan_progress["last_bench"] = now
                payload = {"phase":"scan","phase_done":done,"phase_total":total,
                           "overall_percent":3+20*done/max(1,total),"stats":stats,
                           "candidates_so_far":stats.get("candidate_windows", 0),
                           "ocr_backend":provider_impl.engine_name,
                           "ocr_gpu_active":bool(getattr(provider_impl,"gpu_active",False)),
                           "runtime_providers":list(getattr(provider_impl,"runtime_providers",[]) or [])}
                self._emit(on_event,payload)
            windows = sampler.scan(path, meta, region, on_event, on_scan_progress)
            # If the primary ROI/global gate found nothing, immediately run one permissive fallback
            # scan. This is rare but critical for high-recall mode: no OCR should be skipped solely
            # because the first tiny-gate configuration was too conservative. The fallback is still
            # lightweight (10 FPS grayscale) and never becomes a full-resolution OCR pass.
            fallback_used = False
            if not windows:
                fallback_used = True
                self._emit(on_event, {"phase":"scan_fallback", "overall_percent":23,
                                      "message":"Primary fast scan found no candidate; running permissive global fallback"})
                fallback = AdaptiveFrameSampler(
                    self.ffmpeg,
                    5.0,
                    max(224, min(288, self.scan_width)),
                    1,
                    max(0.008, float(self.cfg.get("scan_diff_threshold",0.018)) * 0.60),
                    max(0.0025, float(self.cfg.get("global_textlike_threshold",0.005)) * 0.55),
                    max(0.025, float(self.cfg.get("global_line_density_threshold",0.05)) * 0.70),
                    str(self.cfg.get("hwaccel","auto")),
                )
                windows = fallback.scan(path, meta, {"x":0,"y":0,"w":100,"h":100}, on_event, None)
            if not windows:
                self._emit(on_event, {"phase":"scan_rescue", "overall_percent":23.5,
                                      "message":"ROI/global temporal gates found no candidates; running a permissive high-resolution global gate"})
                rescue = AdaptiveFrameSampler(
                    self.ffmpeg, max(24.0, self.scan_fps), max(320, self.scan_width), 1,
                    max(0.006, float(self.cfg.get("scan_diff_threshold",0.018)) * 0.45),
                    max(0.0018, float(self.cfg.get("global_textlike_threshold",0.005)) * 0.45),
                    max(0.025, float(self.cfg.get("global_line_density_threshold",0.04)) * 0.65),
                    str(self.cfg.get("hwaccel","auto")),
                )
                windows = rescue.scan(path, meta, {"x":0,"y":0,"w":100,"h":100}, on_event, None)
                for w in windows:
                    w.region_id = "global"
            scan_elapsed = time.monotonic() - scan_start
            self._emit(on_event,{"phase":"scan_done","phase_done":1,"phase_total":1,"overall_percent":24,
                                 "candidates":len(windows),"scan_seconds":round(scan_elapsed,2)})
            if cancel_cb and cancel_cb(): raise RuntimeError("OCR job cancelled")

            # V4: event-driven OCR. One detector pass establishes geometry, recognition-only
            # tracking handles interior frames, and full detection is repeated only for suspected
            # lexical changes. This removes the old dense full-OCR sampling bottleneck.
            temporal = TemporalOCRProcessor(
                batch_size=self.batch_size,
                sample_step=float(self.cfg.get("recognition_sample_step", 0.50)),
                max_dense_seconds=float(self.cfg.get("recognition_dense_seconds", 4.0)),
            )
            temporal_start = time.monotonic()
            if not windows:
                raise RuntimeError(
                    "OCR không tìm được candidate sau 3 lượt scan. Job dừng có chẩn đoán thay vì xuất timeline rỗng. "
                    "Kiểm tra video/contrast; xem các phase scan_rescue và debug frame trong log."
                )
            temporal.last_trace = getattr(sampler, "last_trace", [])
            def _live_segment(seg):
                preview = Exporter._seg(0, seg)
                # Browser/editor consumers historically read start/start_sec; Exporter uses the
                # canonical start_time/end_time names. Emit both so a provisional segment never
                # appears as 00:00.0 while OCR is still running.
                preview.update({
                    "start": preview["start_time"], "end": preview["end_time"],
                    "start_sec": preview["start_time"], "end_sec": preview["end_time"],
                    "source_start": preview["start_time"], "source_end": preview["end_time"],
                    "source_text": preview["text"],
                    "timing_source": "streaming_preview",
                    "ocr_locked": False, "timing_locked": False,
                })
                self._emit(on_event, {
                    "type": "segment_preview",
                    "segment": preview,
                    "overall_percent": max(24.0, min(81.0, 24.0 + 44.0 * max(0.0, min(1.0, seg.end_time / max(meta.duration, 1e-9))))),
                })

            observations, temporal_stats = temporal.run(
                path, meta, windows, region, provider_impl,
                ffprobe=self.ffprobe, on_event=on_event, cancel_cb=cancel_cb,
                secondary_provider=latin_provider, on_live_segment=_live_segment,
            )
            ocr_elapsed = time.monotonic() - temporal_start
            total = int(temporal_stats.full_detection_frames + temporal_stats.full_change_detection_frames
                        + temporal_stats.recognition_only_frames)
            self._emit(on_event, {
                "phase":"recognition_done", "phase_done":1, "phase_total":1,
                "overall_percent":68,
                "full_detection_frames":temporal_stats.full_detection_frames,
                "full_change_detection_frames":temporal_stats.full_change_detection_frames,
                "recognition_only_frames":temporal_stats.recognition_only_frames,
                "recognition_only_line_crops":temporal_stats.recognition_only_line_crops,
                "ocr_seconds":round(ocr_elapsed,3),
            })
            if cancel_cb and cancel_cb():
                raise RuntimeError("OCR job cancelled")

            tracker=TextTracker(max_gap=float(self.cfg.get("track_max_gap",1.75)),sim_threshold=float(self.cfg.get("track_similarity",0.50)))
            tracks=tracker.ingest(observations)
            verifier=OCRVerifier(high_recall=self.mode=="HIGH_RECALL")
            classifier=SubtitleClassifier(self.mode)
            initial_verifs={tr.tracking_id:verifier.verify(tr) for tr in tracks}

            # Explicit targeted verification. Its output is merged into the existing tracks.
            self._targeted_verify(path,meta,tracks,provider_impl,region,color,on_event)
            verifs={tr.tracking_id:verifier.verify(tr) for tr in tracks}
            classes={tr.tracking_id:classifier.classify(tr) for tr in tracks}

            builder=TimelineBuilder(meta, max_observation_gap=min(0.90, float(self.cfg.get("track_max_gap", 1.10))))
            segments=builder.coarse(tracks,tracker.consensus,classes)
            if not segments:
                raise RuntimeError(
                    f"OCR có {len(windows)} candidate windows nhưng không tạo segment "
                    f"(observations={len(observations)}, tracks={len(tracks)}). Không đánh dấu hoàn tất 100%."
                )
            boundary_start=time.monotonic()
            with VideoReader(path,self.ffprobe) as br:
                coarse_order=sorted(range(len(segments)), key=lambda j:(segments[j].start_time,segments[j].region_id,segments[j].tracking_id))
                for i,s in enumerate(segments):
                    tr=next((x for x in tracks if x.tracking_id==s.tracking_id),None)
                    if not tr: continue
                    before=max((o.frame_index for o in tr.observations if o.frame_index<s.frame_start),default=None)
                    after=min((o.frame_index for o in tr.observations if o.frame_index>s.frame_end),default=None)
                    start_negative_seg=None; end_negative_seg=None
                    pos=coarse_order.index(i)
                    for k in range(pos-1,-1,-1):
                        cand=segments[coarse_order[k]]
                        if cand.region_id==s.region_id:
                            start_negative_seg=cand; break
                    for k in range(pos+1,len(coarse_order)):
                        cand=segments[coarse_order[k]]
                        if cand.region_id==s.region_id:
                            end_negative_seg=cand; break
                    # Adjacent same-region subtitles are stronger transition brackets than a
                    # disappearing-global-text heuristic. This is what lets A->B at the same
                    # coordinates get exact frame boundaries even while text remains on screen.
                    if start_negative_seg is not None and start_negative_seg.end_time <= s.start_time + 0.25:
                        before=int(start_negative_seg.frame_end)
                    if end_negative_seg is not None and end_negative_seg.start_time >= s.end_time - 0.25:
                        after=int(end_negative_seg.frame_start)
                    if before is None and s.frame_start>0:
                        before=max(0,s.frame_start-int(max(2,self.scan_fps*0.35)))
                    if after is None and s.frame_end<meta.frame_count-1:
                        after=min(meta.frame_count-1,s.frame_end+int(max(2,round(self.scan_fps*0.35))))
                    anchor_frame=max(0,int(s.frame_start)); anchor=br.get_frame(anchor_frame); anchor_sig=None; start_negative_sig=None; end_negative_sig=None
                    if anchor is not None:
                        bx0,by0,bx1,by1=map(float,s.bbox)
                        ax0=max(0,int(math.floor(min(bx0,bx1)))-3); ay0=max(0,int(math.floor(min(by0,by1)))-3)
                        ax1=min(anchor.shape[1],int(math.ceil(max(bx0,bx1)))+3); ay1=min(anchor.shape[0],int(math.ceil(max(by0,by1)))+3)
                        if ax1>ax0 and ay1>ay0: anchor_sig=self._glyph_signature(anchor[ay0:ay1,ax0:ax1])
                    def signature_for_segment(seg_ref, frame_index):
                        if seg_ref is None or frame_index is None: return None
                        nf=br.get_frame(int(frame_index))
                        if nf is None: return None
                        bx0,by0,bx1,by1=map(float,seg_ref.bbox)
                        nx0=max(0,int(math.floor(min(bx0,bx1)))-3); ny0=max(0,int(math.floor(min(by0,by1)))-3)
                        nx1=min(nf.shape[1],int(math.ceil(max(bx0,bx1)))+3); ny1=min(nf.shape[0],int(math.ceil(max(by0,by1)))+3)
                        if nx1<=nx0 or ny1<=ny0: return None
                        return self._glyph_signature(nf[ny0:ny1,nx0:nx1])

                    start_negative_sig=signature_for_segment(start_negative_seg, before)
                    end_negative_sig=signature_for_segment(end_negative_seg, after)
                    # Fallback to an observation prototype when no neighboring segment is available.
                    if start_negative_sig is None and before is not None:
                        prev_obs=next((o for o in tr.observations if o.frame_index==before),None)
                        if prev_obs is not None and similarity(prev_obs.text,s.text) < 0.90:
                            start_negative_sig=signature_for_segment(
                                type("_S",(),{"bbox":prev_obs.bbox})(), before)
                    if end_negative_sig is None and after is not None:
                        nxt_obs=next((o for o in tr.observations if o.frame_index==after),None)
                        if nxt_obs is not None and similarity(nxt_obs.text,s.text) < 0.90:
                            end_negative_sig=signature_for_segment(
                                type("_S",(),{"bbox":nxt_obs.bbox})(), after)
                    trace = getattr(locals().get("temporal", None), "last_trace", None) or getattr(locals().get("sampler", None), "last_trace", None) or []
                    # Use cached 30 FPS fast-scan transitions as a tight coarse bracket. For CFR/VFR
                    # the trace frame index has already been mapped back to the actual source PTS.
                    trace_before = None; trace_after = None
                    if trace:
                        target = float(s.start_time)
                        idx = min(range(len(trace)), key=lambda k: abs(float(trace[k].timestamp) - target))
                        for k in range(idx, -1, -1):
                            if not (trace[k].roi_presence or trace[k].global_presence):
                                trace_before = trace[k].frame_index; break
                        target_end = float(s.end_time)
                        idxe = min(range(len(trace)), key=lambda k: abs(float(trace[k].timestamp) - target_end))
                        for k in range(idxe, len(trace)):
                            if not (trace[k].roi_presence or trace[k].global_presence):
                                trace_after = trace[k].frame_index; break
                    before = trace_before if trace_before is not None else before
                    after = trace_after if trace_after is not None else after
                    ref=BoundaryRefiner(
                        br, meta,
                        lambda fn,text,bbox:self._boundary_active(br,s,fn,color,anchor_sig,end_negative_sig),
                        max_search_seconds=float(self.cfg.get("boundary_search_seconds",0.60)),
                        verify_fn=lambda fn:self._boundary_probe(br,s,fn,provider_impl,color),
                        verify_radius_frames=int(self.cfg.get("boundary_verify_radius_frames",1)),
                        start_predicate=lambda fn,text,bbox:self._boundary_active(br,s,fn,color,anchor_sig,start_negative_sig),
                        end_predicate=lambda fn,text,bbox:self._boundary_active(br,s,fn,color,anchor_sig,end_negative_sig),
                    )
                    segments[i]=ref.refine(s,before,after)
                    segments[i].verification=dict(verifs.get(s.tracking_id,{}))
                    segments[i].verification.update(classes.get(s.tracking_id,{}))
                    segments[i].verification["boundary_refined"] = True

                # Zero-miss safety scan: never assume a temporal gap is empty. Check multiple points
                # against the cheap detector and mark suspicious gaps for targeted OCR verification.
                safety_gaps=[]; ordered=sorted(segments,key=lambda x:x.start_time)
                gap_detector=__import__("pipeline.video_ocr.detector",fromlist=["TextActivityDetector"]).TextActivityDetector(
                    threshold=float(self.cfg.get("scan_diff_threshold",0.018)))
                for a,b in zip(ordered,ordered[1:]):
                    gap=b.start_time-a.end_time
                    if gap<=0.12: continue
                    sample_times=[a.end_time+gap*0.25,a.end_time+gap*0.50,a.end_time+gap*0.75]
                    prev_frame=None; prev_global=None; suspicious=False
                    for ts in sample_times:
                        fi=br.nearest_frame(ts); f=br.get_frame(fi)
                        if f is None: continue
                        x0,y0,x1,y1=self._roi(region,f.shape[1],f.shape[0]); crop=f[y0:y1,x0:x1]
                        act=gap_detector.analyze(prev_frame,crop)
                        gw=min(640,f.shape[1]); gh=max(1,round(f.shape[0]*gw/f.shape[1]))
                        global_small=cv2.resize(f,(gw,gh), interpolation=cv2.INTER_AREA)
                        gact=gap_detector.analyze(prev_global,global_small)
                        suspicious = suspicious or (prev_frame is not None and act.changed) or act.textlike>=0.012 or (prev_global is not None and gact.changed) or gact.textlike>=0.012
                        prev_frame=crop; prev_global=global_small
                    if suspicious:
                        safety_gaps.append({"from_tracking_id":a.tracking_id,"to_tracking_id":b.tracking_id,
                                            "start":a.end_time,"end":b.start_time,"action":"verification_queue"})
                safety={"gaps_checked":max(0,len(ordered)-1),
                        "suspicious_gaps":safety_gaps,"mode":self.mode}
            timeline_elapsed=time.monotonic()-boundary_start

            segments=Deduplicator().run(segments)
            validation=TimelineValidator().validate(segments,meta)
            # Validation issues are explicitly recorded as a verification queue; nothing is silently deleted.
            verification_queue=[]
            for issue in validation.get("issues",[]): verification_queue.append(issue)
            for s in segments:
                if s.needs_review or s.verification.get("needs_review"): verification_queue.append({"type":"segment_needs_review","tracking_id":s.tracking_id})
            audit={
                "build_id": self.BUILD_ID,
                "engine":"VideoOCREngine-v6.4-hotfix","mode":self.mode,"candidate_windows":len(windows),
                "scan_fallback_used": bool(locals().get("fallback_used", False)),
                "planned_ocr_samples": total,"observations":len(observations),"tracks":len(tracks),
                "segments":len(segments),"needs_review":sum(1 for s in segments if s.needs_review),
                "watermark_candidates":sum(1 for s in segments if s.classification=="watermark"),
                "verification_queue":verification_queue,"validation":validation,"safety_scan":safety,
                "ocr_backend": provider_impl.engine_name,
                "ocr_backend_kind": getattr(provider_impl, "_engine_kind", None),
                "ocr_language": self.cfg.get("lang", "auto"),
                "ocr_gpu_active": bool(getattr(provider_impl, "gpu_active", False)),
                "runtime_providers": list(getattr(provider_impl, "runtime_providers", []) or []),
                "planned_ocr_frames": total, "ocr_input_frames": total,
                "full_detection_frames": int(getattr(temporal_stats, "full_detection_frames", 0)),
                "full_change_detection_frames": int(getattr(temporal_stats, "full_change_detection_frames", 0)),
                "recognition_only_frames": int(getattr(temporal_stats, "recognition_only_frames", 0)),
                "recognition_only_line_crops": int(getattr(temporal_stats, "recognition_only_line_crops", 0)),
                "benchmark":bench.report(meta.duration,scan_elapsed,ocr_elapsed,timeline_elapsed,scan_progress["done"],total),
                "selection_policy":"MAXIMUM_RECALL; uncertain candidates retained until verification",
            }
            for idx,seg in enumerate(segments,1):
                if on_segment:
                    on_segment(seg,idx,len(segments))
                self._emit(on_event,{"type":"segment","segment_index":idx,"segment_total":len(segments),"segment":Exporter._seg(idx,seg),
                                     "overall_percent":round(82+15*idx/max(1,len(segments)),2)})
            self._emit(on_event,{"phase":"audit","phase_done":1,"phase_total":1,"overall_percent":100,
                                 "segments":len(segments),"needs_review":audit["needs_review"],"benchmark":audit["benchmark"],
                                 "validation":validation,"safety_scan":safety,"verification_queue":len(verification_queue)})
            return OCRRunResult(segments,audit,meta)
        finally:
            reader.close()
            try:
                if job_dir.exists():
                    import shutil; shutil.rmtree(job_dir,ignore_errors=True)
            except Exception: pass


def run_video_ocr(*args,**kwargs):
    return VideoOCREngine(kwargs.pop("ffmpeg","ffmpeg"),kwargs.pop("ffprobe","ffprobe"),kwargs.pop("config",None)).run(*args,**kwargs)
