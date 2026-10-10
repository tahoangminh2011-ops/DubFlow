from __future__ import annotations
from typing import Any, Dict
from .tracker import TextTrack
from .normalizer import similarity

class SubtitleClassifier:
    """Conservative labeler. Classification never removes text in HIGH_RECALL mode."""
    def __init__(self, mode="HIGH_RECALL"): self.mode=str(mode or "HIGH_RECALL").upper()
    def classify(self, track:TextTrack)->Dict[str,Any]:
        obs=track.observations
        if not obs:return {"label":"unknown","confidence":0.0,"exclude":False}
        span=max(0.0,obs[-1].timestamp-obs[0].timestamp)
        texts=[o.text for o in obs]
        conf=sum(o.confidence for o in obs)/len(obs)
        same=all(similarity(texts[0],t)>=0.94 for t in texts[1:]) if texts else True
        distinct=[]
        for t in texts:
            if not any(similarity(t,d)>=0.90 for d in distinct): distinct.append(t)
        changes=len(distinct)
        static_long=span>=25.0 and same and changes<=1
        if static_long and conf>=0.80:
            return {"label":"watermark","confidence":round(min(1.0,conf+0.10),3),"exclude":False,
                    "reason":"long_persistent_same_text"}
        if len(obs)>=2 or span>=0.08:
            return {"label":"subtitle_candidate","confidence":round(conf,3),"exclude":False,
                    "reason":"temporal_text_evidence"}
        return {"label":"unknown","confidence":round(conf,3),"exclude":False,
                "reason":"singleton_retained_for_verification"}
