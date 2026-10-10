from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from .models import OCRRunResult, TrackSegment

def _srt_time(sec:float)->str:
    ms=max(0,int(round(float(sec)*1000))); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); s,ms=divmod(ms,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

class Exporter:
    def json_dict(self,result:OCRRunResult)->Dict[str,Any]:
        m=result.metadata
        return {"video":{"duration":m.duration,"width":m.width,"height":m.height,"timebase":m.timebase,"vfr":m.vfr,"fps":m.fps,"frame_count":m.frame_count},
                "segments":[self._seg(i+1,s) for i,s in enumerate(result.segments)],"audit":result.audit}
    @staticmethod
    def _seg(i,s:TrackSegment)->Dict[str,Any]:
        return {"id":i,"tracking_id":s.tracking_id,"frame_start":s.frame_start,"frame_end":s.frame_end,"start_time":round(s.start_time,6),"end_time":round(s.end_time,6),"duration":round(max(0,s.end_time-s.start_time),6),"text":s.text,"confidence":round(float(s.confidence),4),"bbox":[round(float(v),2) for v in s.bbox],"line_count":s.line_count,"classification":s.classification,"needs_review":bool(s.needs_review),"region_id":s.region_id,"ocr_source":s.source,"verification":s.verification or {}}
    def write_json(self,result,path:Path)->Path:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(self.json_dict(result),ensure_ascii=False,indent=2),encoding="utf-8"); return path
    def write_srt(self,result,path:Path)->Path:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        lines=[]
        for i,s in enumerate(result.segments,1):
            lines += [str(i),f"{_srt_time(s.start_time)} --> {_srt_time(s.end_time)}",s.text.strip(),""]
        path.write_text("\n".join(lines),encoding="utf-8"); return path
