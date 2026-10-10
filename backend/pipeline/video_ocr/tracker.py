from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .models import OCRObservation, TextDetection
from .normalizer import compact_text, normalize_text, similarity, weighted_consensus


def _iou(a, b):
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area = max(1, (ax1 - ax0) * (ay1 - ay0)) + max(1, (bx1 - bx0) * (by1 - by0)) - inter
    return inter / max(1, area)


def _union(ds: List[TextDetection]):
    if not ds:
        return None
    return (
        min(d.bbox[0] for d in ds), min(d.bbox[1] for d in ds),
        max(d.bbox[2] for d in ds), max(d.bbox[3] for d in ds),
    )


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

@dataclass
class TrackObservation:
    frame_index: int
    timestamp: float
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    needs_review: bool = False
    sample_kind: str = "adaptive"
    region_id: str = "roi"

@dataclass
class TextTrack:
    tracking_id: int
    region_id: str
    observations: List[TrackObservation] = field(default_factory=list)
    source: str = "local"
    def add(self, o: TrackObservation):
        self.observations.append(o)
    @property
    def first(self): return self.observations[0]
    @property
    def last(self): return self.observations[-1]

class TextTracker:
    def __init__(self, max_gap: float = 2.2, sim_threshold: float = 0.52):
        self.max_gap = max(0.5, float(max_gap))
        self.sim_threshold = float(sim_threshold)
        self.tracks: List[TextTrack] = []
        self._next = 1

    @staticmethod
    def _join_items(items: List[TextDetection]) -> str:
        import unicodedata
        def has_cjk(text: str) -> bool:
            return any("CJK UNIFIED IDEOGRAPH" in unicodedata.name(ch, "") for ch in str(text or ""))
        out = ""
        for d in sorted(items, key=lambda x: (x.bbox[0], x.bbox[1])):
            t = normalize_text(d.text)
            if not t:
                continue
            if not out:
                out = t
            elif has_cjk(out[-1:]) or has_cjk(t[:1]):
                out += t
            else:
                out += " " + t
        return normalize_text(out)

    @staticmethod
    def _line_groups(part: List[TextDetection]) -> List[List[TextDetection]]:
        # OCR engines can return one box per word or one box per full line. Build true text-line
        # groups from vertical overlap/baseline proximity before building the subtitle observation.
        groups: List[Dict] = []
        for d in sorted(part, key=lambda x: ((x.bbox[1] + x.bbox[3]) * 0.5, x.bbox[0])):
            y0, y1 = float(d.bbox[1]), float(d.bbox[3])
            cy = (y0 + y1) * 0.5
            h = max(1.0, y1 - y0)
            best = None; best_score = -1.0
            for g in groups:
                gy = float(g["cy"]); gh = max(1.0, float(g["h"]))
                overlap = max(0.0, min(y1, g["y1"]) - max(y0, g["y0"])) / max(1.0, min(y1-y0, g["y1"]-g["y0"]))
                center_score = max(0.0, 1.0 - abs(cy - gy) / max(1.0, 0.75 * max(h, gh)))
                score = max(overlap, center_score)
                if score >= 0.35 and score > best_score:
                    best_score = score; best = g
            if best is None:
                groups.append({"items":[d], "cy":cy, "h":h, "y0":y0, "y1":y1})
            else:
                best["items"].append(d)
                best["cy"] = sum((x.bbox[1] + x.bbox[3]) * 0.5 for x in best["items"]) / len(best["items"])
                best["h"] = max(float(best["h"]), h)
                best["y0"] = min(float(best["y0"]), y0)
                best["y1"] = max(float(best["y1"]), y1)
        groups.sort(key=lambda g: g["y0"])
        return [sorted(g["items"], key=lambda x: x.bbox[0]) for g in groups]

    @staticmethod
    def _combine_lines(lines: List[List[TextDetection]]) -> List[Tuple[str, float, Tuple[float,float,float,float], bool, str]]:
        line_records=[]
        for items in lines:
            if not items: continue
            text=TextTracker._join_items(items)
            if not text: continue
            box=_union(items)
            conf=sum(float(d.confidence) for d in items)/len(items)
            review=any(d.needs_review for d in items)
            source=items[0].source
            line_records.append({"text":text,"conf":conf,"box":box,"review":review,"source":source})
        if not line_records:
            return []

        # Adjacent lines with close baselines and horizontal overlap form one multi-line subtitle.
        merged=[]
        cur=line_records[0]
        for nxt in line_records[1:]:
            a=cur["box"]; b=nxt["box"]
            ah=max(1.0,a[3]-a[1]); bh=max(1.0,b[3]-b[1])
            gap=max(0.0,b[1]-cur["box"][3])
            xover=max(0.0,min(a[2],b[2])-max(a[0],b[0]))
            xunion=max(1.0,max(a[2],b[2])-min(a[0],b[0]))
            xratio=xover/xunion
            close=(gap <= 1.35*max(ah,bh)) and xratio >= 0.05
            if close:
                cur={
                    "text": cur["text"]+"\n"+nxt["text"],
                    "conf": (cur["conf"]+nxt["conf"])*0.5,
                    "box": _union([type("D", (), {"bbox":cur["box"]})(), type("D", (), {"bbox":nxt["box"]})()]),
                    "review": cur["review"] or nxt["review"],
                    "source": cur["source"],
                }
            else:
                merged.append(cur); cur=nxt
        merged.append(cur)
        return [(x["text"],x["conf"],x["box"],x["review"],x["source"]) for x in merged]

    @staticmethod
    def _frame_groups(obs: OCRObservation) -> List[Tuple[str, float, Tuple[float, float, float, float], bool, str, str]]:
        ds = [d for d in obs.detections if normalize_text(d.text) and d.bbox]
        partitions: Dict[str, List[TextDetection]] = {}
        for d in ds:
            partitions.setdefault(d.region_id or "roi", []).append(d)
        out=[]
        for region_id, part in partitions.items():
            for text, conf, box, review, source in TextTracker._combine_lines(TextTracker._line_groups(part)):
                out.append((text,conf,box,review,source,region_id))
        return out

    def ingest(self, observations: List[OCRObservation]) -> List[TextTrack]:
        self.tracks = []
        self._next = 1
        return self.update(observations)

    def update(self, observations: List[OCRObservation]) -> List[TextTrack]:
        for obs in sorted(observations, key=lambda x: (x.frame_index, x.timestamp)):
            for text, conf, box, review, source, obs_region in self._frame_groups(obs):
                if not box:
                    continue
                best = None; best_score = 0.0
                for track in self.tracks:
                    if not track.observations:
                        continue
                    last = track.last
                    gap = float(obs.timestamp - last.timestamp)
                    if gap < -0.001 or gap > self.max_gap:
                        continue
                    sim = similarity(text, last.text)
                    iou = _iou(box, last.bbox)
                    c1, c2 = _center(box), _center(last.bbox)
                    height = max(1.0, last.bbox[3] - last.bbox[1])
                    cy_gap = abs(c1[1] - c2[1]) / height

                    # Never merge a clear lexical change solely because the subtitle occupies the
                    # same pixels.  "Hello" -> "Hello John" is a new subtitle; OCR noise such as
                    # "Hel1o" -> "Hello" remains one track because similarity stays high.
                    a = compact_text(text); b = compact_text(last.text)
                    contained_change = bool(a and b and (a in b or b in a) and
                                            abs(len(a) - len(b)) / max(1, max(len(a), len(b))) >= 0.35)
                    strong_text_match = sim >= max(self.sim_threshold, 0.68) and not contained_change
                    cross_region = track.region_id != obs_region
                    region_compatible = (not cross_region) or (strong_text_match and iou >= 0.28)
                    score = (sim * 1.35) + (iou * 0.55) + (0.18 if cy_gap <= 1.0 else 0.0) - min(0.2, gap * 0.03)
                    # Geometry can rescue tiny one/two-character OCR strings only when the text is
                    # still substantially similar; it cannot override an obvious text replacement.
                    if region_compatible and strong_text_match and cy_gap <= 2.0 and score > best_score:
                        best_score, best = score, track
                if best is None:
                    best = TextTrack(self._next, obs_region, source=source)
                    self._next += 1
                    self.tracks.append(best)
                best.add(TrackObservation(
                    obs.frame_index, obs.timestamp, text, conf, box,
                    review, obs.sample_kind, obs_region,
                ))
        return [t for t in self.tracks if t.observations]

    def consensus(self, track: TextTrack) -> str:
        return weighted_consensus(
            (o.text, o.confidence * (1.15 if o.sample_kind == "burst" else 1.0))
            for o in track.observations
        )
