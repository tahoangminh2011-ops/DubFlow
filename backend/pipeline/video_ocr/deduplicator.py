from __future__ import annotations
from typing import List
from .models import TrackSegment
from .normalizer import similarity


def _iou(a, b):
    if not a or not b: return 0.0
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix=max(0.0,min(ax1,bx1)-max(ax0,bx0)); iy=max(0.0,min(ay1,by1)-max(ay0,by0))
    inter=ix*iy
    aa=max(1.0,(ax1-ax0)*(ay1-ay0)); ab=max(1.0,(bx1-bx0)*(by1-by0))
    return inter/max(1.0,aa+ab-inter)


def _merge(a: TrackSegment,b: TrackSegment)->TrackSegment:
    winner=a if a.confidence>=b.confidence else b
    verification=dict(a.verification or {}); verification.update(dict(b.verification or {}))
    return TrackSegment(
        tracking_id=winner.tracking_id,
        frame_start=min(a.frame_start,b.frame_start), frame_end=max(a.frame_end,b.frame_end),
        start_time=min(a.start_time,b.start_time), end_time=max(a.end_time,b.end_time),
        text=winner.text, confidence=max(a.confidence,b.confidence),
        bbox=[min(a.bbox[0],b.bbox[0]),min(a.bbox[1],b.bbox[1]),max(a.bbox[2],b.bbox[2]),max(a.bbox[3],b.bbox[3])],
        line_count=max(a.line_count,b.line_count), classification=winner.classification,
        needs_review=bool(a.needs_review or b.needs_review), region_id=winner.region_id,
        source=winner.source, verification=verification,
    )


class Deduplicator:
    """Only merge genuine duplicate/cross-source observations.

    Identical subtitles separated by a real gap remain distinct cues. A 0.35s temporal tolerance
    is intentionally not used because it destroys legitimate short repeated subtitles.
    """
    def run(self,segments:List[TrackSegment])->List[TrackSegment]:
        out=[]
        for s in sorted(segments,key=lambda x:(x.start_time,x.end_time,x.region_id,x.tracking_id)):
            merged=False
            for j in range(len(out)-1,-1,-1):
                p=out[j]
                gap=s.start_time-p.end_time
                if gap>0.08: break
                if p.region_id!=s.region_id and {p.region_id,s.region_id}!={"roi","global"}: continue
                if similarity(p.text,s.text)<0.97: continue
                overlap=max(0.0,min(p.end_time,s.end_time)-max(p.start_time,s.start_time))
                geom=_iou(p.bbox,s.bbox)>=0.45
                if overlap>0.0 or (gap<=0.04 and geom):
                    out[j]=_merge(p,s); merged=True; break
            if not merged: out.append(s)
        for i,s in enumerate(out,1):
            s.verification=dict(s.verification or {}); s.verification["segment_id"]=i
        return sorted(out,key=lambda x:(x.start_time,x.region_id,x.tracking_id))
