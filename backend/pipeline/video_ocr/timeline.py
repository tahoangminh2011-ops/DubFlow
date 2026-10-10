from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from .models import TrackSegment, VideoMetadata
from .normalizer import similarity
from .tracker import TextTrack

class TimelineBuilder:
    def __init__(self, meta:VideoMetadata, max_observation_gap: float = 1.05):
        self.meta=meta
        self.max_observation_gap=max(0.25,float(max_observation_gap))
    def _runs(self,track:TextTrack):
        runs=[]; cur=[]
        for o in track.observations:
            if not cur: cur=[o]; continue
            prev=cur[-1]; sim=similarity(prev.text,o.text); gap=o.timestamp-prev.timestamp
            gap_break = gap > self.max_observation_gap
            change=(gap_break or (sim<0.76 and gap<=1.0) or (sim<0.84 and gap<=0.25 and (prev.sample_kind=="burst" or o.sample_kind=="burst")))
            if change:
                runs.append(cur); cur=[o]
            else: cur.append(o)
        if cur:runs.append(cur)
        return runs

    def coarse(self,tracks:List[TextTrack],consensus_fn:Callable[[TextTrack],str],classification:Dict[int,Dict[str,Any]])->List[TrackSegment]:
        out=[]
        for t in tracks:
            for run in self._runs(t):
                txt=consensus_fn(t)
                # Prefer a consensus limited to the run, not to the entire track.
                from .normalizer import weighted_consensus
                txt=weighted_consensus((o.text,o.confidence) for o in run) or txt
                b=[min(o.bbox[0] for o in run),min(o.bbox[1] for o in run),max(o.bbox[2] for o in run),max(o.bbox[3] for o in run)]
                conf=sum(o.confidence for o in run)/len(run)
                c=classification.get(t.tracking_id,{"label":"unknown","confidence":conf,"exclude":False})
                out.append(TrackSegment(t.tracking_id,run[0].frame_index,run[-1].frame_index,run[0].timestamp,run[-1].timestamp,txt,conf,b,len(run[0].text.split("\n")),c["label"],any(o.needs_review for o in run),t.region_id,t.source,{}))
        out.sort(key=lambda s:(s.start_time,s.region_id,s.tracking_id)); return out
