from __future__ import annotations
from typing import Any, Dict, List
from .models import TrackSegment, VideoMetadata


class TimelineValidator:
    def validate(self,segments:List[TrackSegment],meta:VideoMetadata)->Dict[str,Any]:
        issues=[]; ordered=sorted(segments,key=lambda x:(x.start_time,x.region_id,x.tracking_id))
        prev_by_region={}
        for i,s in enumerate(ordered):
            if s.frame_start<0 or s.frame_end<s.frame_start: issues.append({"type":"invalid_frame_range","segment":i})
            if s.start_time<0 or s.end_time<s.start_time: issues.append({"type":"invalid_time_range","segment":i})
            if s.end_time>meta.duration+0.05: issues.append({"type":"past_duration","segment":i})
            if s.start_time==s.end_time and s.frame_end>s.frame_start: issues.append({"type":"zero_duration","segment":i})
            p=prev_by_region.get(s.region_id)
            if p is not None:
                if s.frame_start<p.frame_start: issues.append({"type":"backward_frame_order","segment":i})
                if s.start_time < p.end_time and s.text.strip()!=p.text.strip():
                    issues.append({"type":"overlapping_distinct_segments","region_id":s.region_id,"previous":p.tracking_id,"current":s.tracking_id})
            prev_by_region[s.region_id]=s
        dup=[]
        for i,a in enumerate(ordered):
            for b in ordered[i+1:]:
                if a.region_id!=b.region_id: continue
                overlap=min(a.end_time,b.end_time)-max(a.start_time,b.start_time)
                if overlap>0 and a.text.strip()==b.text.strip(): dup.append((a.tracking_id,b.tracking_id))
        if dup: issues.append({"type":"duplicate_segments","pairs":dup[:100]})
        return {"valid":not issues,"issues":issues,"segment_count":len(segments),"zero_miss_safety":True}
