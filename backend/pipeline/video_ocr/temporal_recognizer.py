from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

from .models import OCRObservation, TextDetection, VideoMetadata, CandidateWindow
from .reader import VideoReader
from .tracker import TextTracker
from .normalizer import normalize_text, similarity, weighted_consensus


@dataclass
class TemporalOCRStats:
    full_detection_frames: int = 0
    recognition_only_frames: int = 0
    recognition_only_line_crops: int = 0
    full_change_detection_frames: int = 0
    empty_anchor_windows: int = 0


class TemporalOCRProcessor:
    """Event-driven OCR with one decoder pass for all OCR-selected frames.

    Full detector/recognizer is used only for candidate anchors and lexical replacement events.
    Interior samples use recognition-only on the previously localized text-line crop. All requested
    frame indices are decoded together in one forward pass to avoid re-decoding a 60-minute video
    three times.
    """

    def __init__(self, batch_size: int = 16, sample_step: float = 0.75, max_dense_seconds: float = 4.0):
        self.batch_size = max(1, min(64, int(batch_size)))
        self.sample_step = max(0.40, float(sample_step))
        self.max_dense_seconds = max(2.0, float(max_dense_seconds))
        self.last_trace = []

    @staticmethod
    def _roi(region: Dict[str, float], w: int, h: int):
        x = float(region.get("x", 0)); y = float(region.get("y", 65))
        rw = float(region.get("w", 100)); rh = float(region.get("h", 35))
        x0 = max(0, min(w - 1, int(round(x * w / 100))))
        y0 = max(0, min(h - 1, int(round(y * h / 100))))
        x1 = max(x0 + 1, min(w, int(round((x + rw) * w / 100))))
        y1 = max(y0 + 1, min(h, int(round((y + rh) * h / 100))))
        return x0, y0, x1, y1

    @staticmethod
    def _pad_box(box, frame_shape, px=0.10, py=0.42):
        # Tight crop is critical for recognition-only inference: excessive vertical padding lets
        # background faces/logos leak into the recognizer and is a major source of hallucinated text.
        h, w = frame_shape[:2]
        x0, y0, x1, y1 = map(float, box)
        bw = max(2.0, x1 - x0); bh = max(2.0, y1 - y0)
        dx = max(8.0, bw * px); dy = max(5.0, bh * py)
        return (
            max(0, int(math.floor(x0 - dx))), max(0, int(math.floor(y0 - dy))),
            min(w, int(math.ceil(x1 + dx))), min(h, int(math.ceil(y1 + dy))),
        )

    @staticmethod
    def _line_groups(obs: OCRObservation):
        groups = []
        by_region: Dict[str, List[TextDetection]] = {}
        for d in obs.detections:
            if normalize_text(d.text) and d.bbox:
                by_region.setdefault(d.region_id or "roi", []).append(d)
        for rid, ds in by_region.items():
            lines = TextTracker._line_groups(ds)
            for line in lines:
                if not line:
                    continue
                box = (
                    min(d.bbox[0] for d in line), min(d.bbox[1] for d in line),
                    max(d.bbox[2] for d in line), max(d.bbox[3] for d in line),
                )
                text = TextTracker._join_items(line)
                conf = sum(float(d.confidence) for d in line) / len(line)
                groups.append({"region_id": rid, "bbox": box, "text": text, "confidence": conf})
        return groups

    @staticmethod
    def _nearest_frame(meta: VideoMetadata, timestamp: float) -> int:
        pts = meta.pts
        if not pts:
            return max(0, int(round(float(timestamp) * max(meta.fps, 1e-9))))
        lo, hi = 0, len(pts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if pts[mid] < timestamp:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0 and abs(pts[lo - 1] - timestamp) <= abs(pts[lo] - timestamp):
            return lo - 1
        return lo

    def _sample_times(self, w: CandidateWindow) -> List[float]:
        s, e = float(w.start_time), float(w.end_time)
        span = max(0.0, e - s)
        if span <= 0.24:
            return [round((s + e) / 2.0, 4)]
        if span <= 1.0:
            return [round(s + span * 0.33, 4), round(s + span * 0.67, 4)]
        # Adaptive consensus: sample densely during the first ~1.2 s after appearance, then
        # relax cadence while the subtitle remains stable. This catches rapid replacements without
        # OCR-ing every frame of a long stable caption.
        points=[]
        dense_end=min(e-0.08, s+1.2)
        t=s+0.24
        while t < dense_end:
            points.append(round(t,4)); t += 0.24
        t=max(dense_end, s+0.72)
        while t < e-0.08:
            points.append(round(t,4)); t += self.sample_step
        points.append(round(max(s+0.06, e-0.08),4))
        return sorted(set(x for x in points if s <= x <= e))

    @staticmethod
    def _read_all_requested(path: Path, frame_indices: Sequence[int], ffprobe: str):
        targets = sorted(set(int(x) for x in frame_indices if int(x) >= 0))
        out = {}
        if not targets:
            return out
        with VideoReader(path, ffprobe) as reader:
            for fi, frame in reader.iter_selected_frames(targets):
                out[int(fi)] = frame
        return out

    @staticmethod
    def _crop_for_window(frame, region, window):
        h, w = frame.shape[:2]
        source_region = str(getattr(window, "region_id", "roi") or "roi").lower()
        if source_region == "global" or str(window.reason).startswith("global"):
            # Fast-gate bboxes are not text-detector boxes and may cover only a few strokes.
            # A global fallback must not be cropped back into a potentially misplaced user ROI.
            return frame, (0, 0), "global"
        # Use the complete configured ROI as the OCR anchor crop. A bbox from the 320px gate is
        # only a locator, not a detector box; using it here clipped leading/trailing CJK characters.
        x0, y0, x1, y1 = TemporalOCRProcessor._roi(region, w, h)
        return frame[y0:y1, x0:x1], (x0, y0), "roi"

    @staticmethod
    def _run_full(provider, images, metas):
        """Run full OCR while preserving one-to-one ownership even for duplicate source frames."""
        if not images:
            return []
        # Provider output exposes frame_index but multiple input crops can refer to the same source
        # frame. Give each input a synthetic unique id for inference and map it back afterwards.
        base_ids = [10_000_000_000 + i for i in range(len(metas))]
        dets = provider.recognize_batch(
            images,
            base_ids,
            [x[1] for x in metas],
            [x[2] for x in metas],
            region_id="roi",
            region_ids=[x[3] for x in metas],
            batch_size=max(1, min(64, len(images))),
        )
        by_owner: Dict[int, List[TextDetection]] = {}
        for d in dets:
            idx = int(d.frame_index) - 10_000_000_000
            if 0 <= idx < len(metas):
                by_owner.setdefault(idx, []).append(
                    TextDetection(d.text, d.confidence, d.bbox, metas[idx][0], metas[idx][1], d.region_id,
                                  d.line_index, d.source, d.needs_review, dict(d.metadata))
                )
        out = []
        for i, meta in enumerate(metas):
            ds = by_owner.get(i, [])
            if ds:
                out.append({"owner": i, "wi": meta[4], "obs": OCRObservation(int(meta[0]), float(meta[1]), ds, meta[5])})
        return out

    def run(self, path: Path, meta: VideoMetadata, windows: List[CandidateWindow], region: Dict[str, float],
            provider, ffprobe: str = "ffprobe", on_event=None, cancel_cb=None, secondary_provider=None, on_live_segment=None):
        stats = TemporalOCRStats()
        observations: List[OCRObservation] = []
        if not windows:
            if on_event:
                on_event({"phase": "recognition", "full_detection_frames": 0, "recognition_only_frames": 0,
                          "recognition_only_line_crops": 0, "full_change_detection_frames": 0, "planned_ocr_frames": 0})
            return observations, stats

        # -------- Stage A: separate cheap replacement verification from expensive full OCR. --------
        # Broad windows establish stable text/bboxes. Replacement windows are only promoted to full
        # detector OCR when a recognition-only probe says the text actually changed. This removes the
        # main V3 bottleneck: dozens of false replacement events caused by moving backgrounds.
        broad_windows=[(wi,w) for wi,w in enumerate(windows) if w.reason != "text_replacement"]
        replacement_windows=[(wi,w) for wi,w in enumerate(windows) if w.reason == "text_replacement"]
        full_requests=[]; replacement_probe_requests=[]; rec_requests=[]
        for wi,w in broad_windows:
            s0=max(0.0,float(w.start_time)); e0=min(meta.duration,float(w.end_time)); span=max(0.0,e0-s0)
            times=[s0+min(0.28,max(0.06,span*0.30))]
            if span>8.0: times.append(s0+span*0.68)
            for ts in times:
                full_requests.append((self._nearest_frame(meta,ts),float(ts),wi,w))
            for ts in self._sample_times(w):
                rec_requests.append((self._nearest_frame(meta,ts),float(ts),wi))
        for wi,w in replacement_windows:
            s0=max(0.0,float(w.start_time)); e0=min(meta.duration,float(w.end_time)); span=max(0.0,e0-s0)
            replacement_probe_requests.append((self._nearest_frame(meta,s0+span*0.5),float(s0+span*0.5),wi,w))

        # Decode every requested source frame only once. Replacement probes share this frame map.
        all_indices=[x[0] for x in full_requests]+[x[0] for x in rec_requests]+[x[0] for x in replacement_probe_requests]
        frame_map=self._read_all_requested(path,all_indices,ffprobe)

        # Deduplicate broad full requests only by frame/crop identity.
        unique_full=[]; seen=set()
        for req in full_requests:
            fi,ts,wi,w=req
            key=(int(fi),wi,tuple(w.preferred_bbox) if w.preferred_bbox else None)
            if key in seen: continue
            seen.add(key); unique_full.append(req)

        state_by_window: Dict[int, Dict[str, Any]]={}
        full_obs_by_window: Dict[int,List[OCRObservation]]={}
        live_tracker=TextTracker(max_gap=1.0,sim_threshold=0.50)
        live_emitted=set()

        def emit_live(obs: OCRObservation):
            if not on_live_segment: return
            for tr in live_tracker.update([obs]):
                text=weighted_consensus((x.text,x.confidence) for x in tr.observations)
                if not text: continue
                first,last=tr.first,tr.last
                bbox=[min(x.bbox[0] for x in tr.observations),min(x.bbox[1] for x in tr.observations),
                      max(x.bbox[2] for x in tr.observations),max(x.bbox[3] for x in tr.observations)]
                from .models import TrackSegment
                preview=TrackSegment(tr.tracking_id,first.frame_index,last.frame_index,first.timestamp,last.timestamp,text,
                                     max(x.confidence for x in tr.observations),bbox,
                                     max(1,text.count("\n")+1),"subtitle_candidate",True,tr.region_id,tr.source,
                                     {"provisional":True,"timing_locked":False,"ocr_locked":False})
                # Emit on every meaningful update, not only once, so the editor can watch a sentence stabilize.
                on_live_segment(preview)
                live_emitted.add(tr.tracking_id)

        # -------- Stage B: full detector/recognizer only for broad anchors. --------
        def run_full_requests(requests, kind="burst"):
            if not requests: return 0
            images=[]; metas=[]; count=0
            for fi,ts,wi,w in requests:
                frame=frame_map.get(int(fi))
                if frame is None: continue
                crop,off,rid=self._crop_for_window(frame,region,w)
                if crop is None or crop.size==0: continue
                images.append(crop); metas.append((int(fi),float(ts),off,rid,wi,kind))
                if len(images)>=self.batch_size:
                    parsed=self._run_full(provider,images,metas)
                    for ob in parsed:
                        observations.append(ob["obs"]); full_obs_by_window.setdefault(int(ob["wi"]),[]).append(ob["obs"]); emit_live(ob["obs"])
                    count+=len(parsed); images=[]; metas=[]
            if images:
                parsed=self._run_full(provider,images,metas)
                for ob in parsed:
                    observations.append(ob["obs"]); full_obs_by_window.setdefault(int(ob["wi"]),[]).append(ob["obs"]); emit_live(ob["obs"])
                count+=len(parsed)
            return count

        stats.full_detection_frames=run_full_requests(unique_full,"burst")

        # Select the strongest anchor for each broad window using exact window ownership.
        for wi,w in broad_windows:
            vals=full_obs_by_window.get(wi,[])
            if not vals:
                stats.empty_anchor_windows+=1; continue
            anchor=max(vals,key=lambda o:max((float(d.confidence) for d in o.detections),default=0.0))
            groups=self._line_groups(anchor)
            if groups:
                text="\n".join(g["text"] for g in groups if g["text"])
                import unicodedata
                latin_chars=sum(1 for ch in text if ch.isalpha() and "LATIN" in unicodedata.name(ch,""))
                letters=sum(1 for ch in text if ch.isalpha())
                state_by_window[wi]={"groups":groups,"last_text":text,"language":"latin" if letters and latin_chars/max(1,letters)>=0.55 else "primary"}

        # -------- Stage C: targeted verification of replacement events. --------
        promoted=[]
        if replacement_probe_requests and hasattr(provider,"recognize_text_only_batch"):
            probe_images=[]; probe_meta=[]
            for fi,ts,wi,w in replacement_probe_requests:
                frame=frame_map.get(int(fi))
                if frame is None: continue
                # Find nearest earlier broad state so we compare against the actual lexical anchor.
                candidate=None; best_dist=1e9
                for bwi,bw in broad_windows:
                    st=state_by_window.get(bwi)
                    if not st: continue
                    dist=abs(float(bw.end_time)-float(ts))
                    if float(bw.end_time)<=float(ts)+0.14 and dist<best_dist:
                        candidate=st; best_dist=dist
                if candidate and candidate.get("groups"):
                    boxes=[g["bbox"] for g in candidate["groups"]]
                    x0=min(b[0] for b in boxes); y0=min(b[1] for b in boxes); x1=max(b[2] for b in boxes); y1=max(b[3] for b in boxes)
                    # Generous horizontal padding prevents a longer replacement from being clipped.
                    box=self._pad_box((x0,y0,x1,y1),frame.shape[:2],0.42,0.48)
                    crop=frame[box[1]:box[3],box[0]:box[2]]
                elif w.preferred_bbox:
                    box=self._pad_box(w.preferred_bbox,frame.shape[:2],0.42,0.50); crop=frame[box[1]:box[3],box[0]:box[2]]
                else:
                    crop,_off,_rid=self._crop_for_window(frame,region,w)
                if crop is not None and crop.size:
                    probe_images.append(crop); probe_meta.append((fi,ts,wi,w,candidate))
            if probe_images:
                # Select language per candidate window. Empty/weak probes stay high-recall and are promoted.
                prim_i=[]; lat_i=[]
                for i,m in enumerate(probe_meta):
                    cand=m[4]
                    (lat_i if secondary_provider is not None and cand and cand.get("language")=="latin" else prim_i).append(i)
                probe_pairs={}
                if prim_i:
                    vals=provider.recognize_text_only_batch([probe_images[i] for i in prim_i],[30_000_000_000+i for i in prim_i],[probe_meta[i][1] for i in prim_i],batch_size=self.batch_size)
                    for i,pair in zip(prim_i,vals): probe_pairs[i]=(pair[2],pair[3])
                if lat_i and secondary_provider is not None:
                    vals=secondary_provider.recognize_text_only_batch([probe_images[i] for i in lat_i],[40_000_000_000+i for i in lat_i],[probe_meta[i][1] for i in lat_i],batch_size=self.batch_size)
                    for i,pair in zip(lat_i,vals): probe_pairs[i]=(pair[2],pair[3])
                for i,m in enumerate(probe_meta):
                    text=normalize_text(probe_pairs.get(i,("",0.0))[0]); score=float(probe_pairs.get(i,("",0.0))[1])
                    candidate=m[4]; base=normalize_text(candidate.get("last_text","") if candidate else "")
                    sim=similarity(text,base) if text and base else 0.0
                    # High-recall rule: empty/weak probe or no reference => promote. Strong same-text probe
                    # rejects the false replacement; a lexical difference promotes to full OCR.
                    same=(text and base and sim>=0.88 and score>=0.55)
                    if not same:
                        promoted.append(m[:4])
                    else:
                        stats.replacement_probe_rejected = int(getattr(stats,"replacement_probe_rejected",0))+1
                stats.replacement_probe_frames=len(probe_meta)
        else:
            promoted=[x[:4] for x in replacement_probe_requests]

        # -------- Stage D: promoted replacement events get one full detector/recognizer frame. --------
        stats.full_change_detection_frames=run_full_requests(promoted,"change")

        # -------- Stage E: recognition-only interior consensus on broad windows. --------
        rec_images=[]; rec_meta=[]
        for fi,ts,wi in rec_requests:
            frame=frame_map.get(int(fi)); state=state_by_window.get(wi)
            if frame is None or not state: continue
            for gi,g in enumerate(state["groups"]):
                bx=self._pad_box(g["bbox"],frame.shape[:2],0.18,0.40)
                crop=frame[bx[1]:bx[3],bx[0]:bx[2]]
                if crop.size: rec_images.append(crop); rec_meta.append((int(fi),float(ts),wi,gi,g,bx))

        rec_out=[]
        if rec_images and hasattr(provider,"recognize_text_only_batch"):
            primary_idx=[]; latin_idx=[]
            for i,m in enumerate(rec_meta):
                state=state_by_window.get(m[2],{})
                (latin_idx if secondary_provider is not None and state.get("language")=="latin" else primary_idx).append(i)
            if primary_idx:
                pairs=provider.recognize_text_only_batch([rec_images[i] for i in primary_idx],[10_000_000_000+i for i in primary_idx],[rec_meta[i][1] for i in primary_idx],batch_size=self.batch_size)
                for i,pair in zip(primary_idx,pairs): rec_out.append((rec_meta[i],pair[2],pair[3]))
            if latin_idx and secondary_provider is not None:
                pairs=secondary_provider.recognize_text_only_batch([rec_images[i] for i in latin_idx],[20_000_000_000+i for i in latin_idx],[rec_meta[i][1] for i in latin_idx],batch_size=self.batch_size)
                for i,pair in zip(latin_idx,pairs): rec_out.append((rec_meta[i],pair[2],pair[3]))
        stats.recognition_only_frames=len(set((m[0],m[1],m[2]) for m,_,_ in rec_out))
        stats.recognition_only_line_crops=len(rec_out)

        # Recognition disagreement is already useful evidence: the localized recognition
        # observation is kept so the tracker can split a lexical replacement. A full detector
        # is only needed once per *sustained mismatch run* (for a new bbox/layout), not once per
        # 0.5s sample. This is a major throughput win on long subtitle regions.
        change_requests=[]; by_sample={}
        for m,text,score in rec_out:
            fi,ts,wi,gi,g,bx=m
            by_sample.setdefault((wi,fi,round(ts,5)),[]).append((g,text,score))

        mismatch_points: Dict[int,List[Tuple[float,int]]] = {}
        for (wi,fi,ts),rows in sorted(by_sample.items(),key=lambda x:(x[0][0],x[0][2])):
            state=state_by_window.get(wi)
            if not state: continue
            composed="\n".join(normalize_text(r[1]) for r in sorted(rows,key=lambda r:(r[0]["bbox"][1],r[0]["bbox"][0])) if normalize_text(r[1]))
            base=normalize_text(state.get("last_text",""))
            ds=[]
            for g,text,score in rows:
                text=normalize_text(text)
                if not text: continue
                ds.append(TextDetection(text=text,confidence=float(score),bbox=tuple(g["bbox"]),frame_index=int(fi),timestamp=float(ts),
                                       region_id=str(g["region_id"]),source="PP-OCRv5:recognition-only",needs_review=float(score)<0.55))
            if ds: observations.append(OCRObservation(int(fi),float(ts),ds,"recovery"))
            if composed and base and similarity(composed,base)<0.72:
                mismatch_points.setdefault(wi,[]).append((float(ts),int(fi)))

        # Cluster disagreement samples into temporal runs. A genuine subtitle replacement can
        # last for seconds, but a single full-detector anchor is enough to localize its new box.
        for wi, points in mismatch_points.items():
            points=sorted(points)
            clusters=[]; cur=[points[0]]
            for item in points[1:]:
                if item[0]-cur[-1][0] <= max(0.80, self.sample_step*1.75):
                    cur.append(item)
                else:
                    clusters.append(cur); cur=[item]
            clusters.append(cur)
            for cluster in clusters:
                ts,fi=min(cluster, key=lambda x:x[0])
                change_requests.append((self._nearest_frame(meta,float(ts)), float(ts), wi, windows[wi]))
        if change_requests:
            stats.full_change_detection_frames += run_full_requests(change_requests,"change")

        if cancel_cb and cancel_cb(): raise RuntimeError("OCR job cancelled")
        if on_event:
            on_event({"phase":"recognition","full_detection_frames":stats.full_detection_frames,
                      "recognition_only_frames":stats.recognition_only_frames,
                      "recognition_only_line_crops":stats.recognition_only_line_crops,
                      "full_change_detection_frames":stats.full_change_detection_frames,
                      "empty_anchor_windows":stats.empty_anchor_windows,
                      "replacement_probe_frames":int(getattr(stats,"replacement_probe_frames",0)),
                      "replacement_probe_rejected":int(getattr(stats,"replacement_probe_rejected",0)),
                      "planned_ocr_frames":stats.full_detection_frames+stats.full_change_detection_frames+stats.recognition_only_frames})
        return observations,stats

