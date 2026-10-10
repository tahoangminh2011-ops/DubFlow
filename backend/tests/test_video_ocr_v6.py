from pathlib import Path
import numpy as np
import cv2

from pipeline.video_ocr.fast_change import FastChangeDetector
from pipeline.video_ocr.temporal_recognizer import TemporalOCRProcessor
from pipeline.video_ocr.models import CandidateWindow


def test_full_frame_outlined_cjk_subtitle_is_present():
    # Reproduces the geometry visible in the supplied editor screenshot: a large outlined subtitle
    # sits inside the full video but may be just outside the user's manually drawn ROI.
    img=np.full((492,881),70,np.uint8)
    cv2.putText(img,'我们得赶快去收集物资了',(305,445),cv2.FONT_HERSHEY_SIMPLEX,0.95,245,3,cv2.LINE_AA)
    d=FastChangeDetector(.018,224)
    a=d.analyze(None,img)
    assert a.presence is True
    assert a.bbox is not None


def test_preferred_bbox_is_used_for_anchor_crop():
    frame=np.zeros((720,1280,3),np.uint8)
    frame[600:660,200:1000]=255
    w=CandidateWindow(1.0,2.0,'text_disappearance',(180,595,1010,665),4)
    crop,off,rid=TemporalOCRProcessor._crop_for_window(frame,{'x':0,'y':86,'w':100,'h':9},w)
    assert off[0] < 200 and off[1] < 600
    assert crop.shape[1] > 800 and crop.shape[0] > 70
    assert rid=='roi'


def test_exporter_preview_contains_legacy_timing_aliases():
    from pipeline.video_ocr.engine import VideoOCREngine
    from pipeline.video_ocr.models import VideoMetadata
    from pipeline.video_ocr.models import TrackSegment
    seg=TrackSegment(1,30,60,1.0,2.0,'Hello',0.9,[100,500,400,540],1)
    holder={}
    e=VideoOCREngine(config={})
    e._emit(lambda p: holder.update(p), {'type':'noop'})
    preview={
      'id':0,'start_time':seg.start_time,'end_time':seg.end_time,
      'text':seg.text
    }
    preview.update({'start':preview['start_time'],'end':preview['end_time'],'start_sec':preview['start_time'],'end_sec':preview['end_time']})
    assert preview['start']==1.0 and preview['end']==2.0
