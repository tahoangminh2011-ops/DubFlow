from __future__ import annotations
from pathlib import Path
import numpy as np

from pipeline.video_ocr.detector import TextActivityDetector
from pipeline.video_ocr.evaluation import GroundTruthEvaluator, text_cer, text_wer
from pipeline.video_ocr.models import OCRObservation, TextDetection, TrackSegment, VideoMetadata
from pipeline.video_ocr.normalizer import weighted_consensus
from pipeline.video_ocr.timeline import TimelineBuilder
from pipeline.video_ocr.tracker import TextTracker
from pipeline.video_ocr.exporter import Exporter


def det(text, frame, ts, bbox=(100, 100, 300, 140), conf=.95, region="roi"):
    return TextDetection(text, conf, bbox, frame, ts, region_id=region, needs_review=conf < .55)


def test_detector_does_not_burst_on_stable_motion():
    d = TextActivityDetector(threshold=.018)
    base = np.zeros((120, 300, 3), np.uint8)
    base[:, :, 0] = np.arange(300, dtype=np.uint8)[None, :]
    first = d.analyze(None, base)
    assert first.changed
    prev = base.copy()
    changes = 0
    for i in range(1, 8):
        cur = np.roll(base, i * 3, axis=1)
        a = d.analyze(prev, cur)
        changes += int(a.changed)
        prev = cur
    # Stable moving background must not turn the scan into a continuous burst.
    assert changes <= 2


def test_cross_frame_consensus_keeps_clean_text():
    value = weighted_consensus([("Hello", .95), ("Hel1o", .45), ("Hello", .90), ("Hello", .88)])
    assert value == "Hello"


def test_multiline_and_multiple_regions_are_kept_separate():
    obs = OCRObservation(10, 1.0, [
        det("Hello everyone", 10, 1.0, (100, 700, 500, 740), region="roi"),
        det("How are you?", 10, 1.0, (105, 746, 450, 786), region="roi"),
        det("你好", 10, 1.0, (1400, 50, 1500, 80), region="global"),
    ])
    tracker = TextTracker()
    tracks = tracker.ingest([obs])
    assert len(tracks) == 2
    roi = next(t for t in tracks if t.region_id == "roi")
    assert roi.observations[0].text == "Hello everyone\nHow are you?"


def test_timeline_uses_pts_for_vfr():
    meta = VideoMetadata(10.0, 1920, 1080, "1/1000", 30.0, True, 4, pts=(0.0, .031, .083, .120))
    obs = [OCRObservation(1, .031, [det("Hi", 1, .031)]), OCRObservation(2, .083, [det("Hi", 2, .083)])]
    tracks = TextTracker(max_gap=1.0).ingest(obs)
    segs = TimelineBuilder(meta).coarse(tracks, TextTracker().consensus, {1:{"label":"unknown","confidence":.9,"exclude":False}})
    # The segment timestamps must follow the actual PTS values, not frame/fps.
    assert segs[0].start_time == .031
    assert segs[0].end_time == .083


def test_evaluation_reports_recall_and_frame_errors():
    truth = [{"frame_start":100,"frame_end":200,"start_time":3.33,"end_time":6.66,"text":"Hello world","region_id":"roi"}]
    pred = [{"frame_start":101,"frame_end":200,"start_time":3.36,"end_time":6.66,"text":"Hello world","region_id":"roi"}]
    result = GroundTruthEvaluator().evaluate(pred, truth)
    assert result["subtitle_recall"] == 1.0
    assert result["subtitle_precision"] == 1.0
    assert result["start_boundary_error_frames_mean"] == 1.0
    assert result["end_boundary_error_frames_mean"] == 0.0
    assert text_cer("Hello", "Hello") == 0.0
    assert text_wer("Hello world", "Hello world") == 0.0


def test_json_schema_contains_master_frame_timeline(tmp_path: Path):
    meta = VideoMetadata(60.0, 1920, 1080, "1/1000", 30.0, False, 1800, ())
    seg = TrackSegment(42, 375, 474, 12.5, 15.8, "Hello everyone", .99, [200,850,1700,940], 1)
    from pipeline.video_ocr.models import OCRRunResult
    result = OCRRunResult([seg], {"selection_policy":"MAXIMUM_RECALL"}, meta)
    out = tmp_path / "ocr.json"
    Exporter().write_json(result, out)
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    s = data["segments"][0]
    for key in ("frame_start","frame_end","start_time","end_time","text","confidence","bbox","tracking_id"):
        assert key in s


def test_deduplicator_merges_roi_global_without_losing_timeline():
    from pipeline.video_ocr.deduplicator import Deduplicator
    a = TrackSegment(1, 10, 20, 1.0, 2.0, "Hello", .90, [100, 700, 400, 740], 1, region_id="roi")
    b = TrackSegment(2, 12, 25, 1.4, 2.5, "Hello", .96, [100, 700, 400, 740], 1, region_id="global")
    out = Deduplicator().run([a, b])
    assert len(out) == 1
    assert out[0].frame_start == 10 and out[0].frame_end == 25


def test_boundary_refinement_finds_local_last_frame_before_next_subtitle():
    from pipeline.video_ocr.boundary import BoundaryRefiner
    meta = VideoMetadata(3.0, 640, 360, "1/1000", 30.0, False, 90, tuple(i/30 for i in range(90)))
    seg = TrackSegment(1, 10, 20, meta.time_for_frame(10), meta.time_for_frame(20), "Hello", .95, [0, 0, 100, 40], 1)
    active_frames = set(range(10, 25))
    refined = BoundaryRefiner(None, meta, lambda i, t, b: i in active_frames, max_search_seconds=2).refine(seg, 5, 30)
    assert refined.frame_end == 24


def test_fast_change_detector_has_textline_signal():
    from pipeline.video_ocr.fast_change import FastChangeDetector
    import cv2, numpy as np
    d=FastChangeDetector(.018,down_w=128)
    base=np.full((180,320),50,np.uint8)
    cue=base.copy()
    cv2.putText(cue,'Hello everyone',(50,150),cv2.FONT_HERSHEY_SIMPLEX,.8,240,2,cv2.LINE_AA)
    _,_,line=d._text_presence(d._gray_small(cue))
    assert line >= 0.55


def test_fast_change_detector_stays_cheap_on_stable_frames():
    from pipeline.video_ocr.fast_change import FastChangeDetector
    d = FastChangeDetector(threshold=.018)
    base = np.zeros((180, 320), np.uint8)
    base[:, :] = 80
    assert d.analyze(None, base).changed
    changes = 0
    prev = base.copy()
    for i in range(1, 12):
        cur = np.roll(base, i * 2, axis=1)
        changes += int(d.analyze(prev, cur).changed)
        prev = cur
    assert changes <= 3


def test_fast_change_detector_catches_short_brightness_transition():
    from pipeline.video_ocr.fast_change import FastChangeDetector
    d = FastChangeDetector(threshold=.018)
    base = np.full((180, 320), 40, np.uint8)
    prev = base.copy()
    cue = base.copy()
    cv2 = __import__('cv2')
    cv2.putText(cue, 'Hello', (90, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 240, 2, cv2.LINE_AA)
    assert d.analyze(prev, cue).changed


def test_sampler_global_fallback_can_find_center_text():
    # This is a unit-level check of the global gate: a subtitle moved away from the bottom ROI
    # must remain eligible for the fallback path.
    from pipeline.video_ocr.fast_change import FastChangeDetector
    import cv2, numpy as np
    d=FastChangeDetector(.018,down_w=128)
    base=np.full((180,320),35,np.uint8)
    cue=base.copy()
    cv2.putText(cue,'CENTER SUB',(92,92),cv2.FONT_HERSHEY_SIMPLEX,.7,245,2,cv2.LINE_AA)
    a=d.analyze(base,cue)
    assert a.changed and a.line_score >= .65 and a.diff_ratio >= .006


def test_benchmark_report_returns_runtime_metrics():
    from pipeline.video_ocr.benchmark import BenchmarkManager
    b = BenchmarkManager()
    report = b.report(60.0, 1.0, 2.0, 0.5, 60, 30)
    assert report["decode_fps"] == 60.0
    assert report["ocr_fps"] == 15.0
    assert report["benchmark_targets_minutes"] == [10, 30, 60, 120]


def test_tracker_merges_word_boxes_into_one_line():
    obs = OCRObservation(20, 2.0, [
        det("Hello", 20, 2.0, (100, 700, 220, 740)),
        det("everyone", 20, 2.0, (230, 701, 390, 740)),
    ])
    tracks = TextTracker().ingest([obs])
    assert len(tracks) == 1
    assert tracks[0].observations[0].text == "Hello everyone"


def test_tracker_splits_clear_text_replacement_same_bbox():
    obs = [
        OCRObservation(30, 3.0, [det("Hello", 30, 3.0)]),
        OCRObservation(31, 3.033, [det("Hello John", 31, 3.033)]),
    ]
    tracks = TextTracker().ingest(obs)
    assert len(tracks) == 2


def test_ocr_language_mapping_keeps_auto_and_latin_separate():
    from pipeline.video_ocr.ocr_provider import RapidPPV5Provider
    auto = RapidPPV5Provider(lang="auto", backend="rapidocr", use_gpu=False)
    vi = RapidPPV5Provider(lang="vi", backend="rapidocr", use_gpu=False)
    zh = RapidPPV5Provider(lang="zh", backend="rapidocr", use_gpu=False)
    assert auto.requested_lang == "auto" and auto.lang == "ch"
    assert vi.lang == "latin"
    assert zh.lang == "ch"
    assert vi._cache_key() != zh._cache_key()


def test_sampler_does_not_close_roi_cue_when_global_fallback_disappears():
    # Regression guard for the V3 bug: global fallback can disappear while the main ROI still
    # contains the subtitle. The cue must remain active until both regions are inactive.
    from pipeline.video_ocr.sampler import AdaptiveFrameSampler
    from pipeline.video_ocr.models import VideoMetadata
    import cv2
    sampler = AdaptiveFrameSampler('ffmpeg', 15, 256, 1, .018, .005, .05, 'none')
    frames=[]; base=np.full((144,256),30,np.uint8)
    for i in range(8):
        fr=base.copy()
        if 1<=i<=5:
            cv2.putText(fr,'ROI TEXT',(30,132),cv2.FONT_HERSHEY_SIMPLEX,.55,240,2,cv2.LINE_AA)
        frames.append(fr)
    sampler._run_chunk=lambda *args,**kwargs: ((i/15.0,fr) for i,fr in enumerate(frames))
    meta=VideoMetadata(.55,256,144,'1/30',15.0,False,len(frames),tuple(i/15 for i in range(len(frames))))
    windows=sampler.scan(Path('synthetic.mp4'),meta,{'x':0,'y':65,'w':100,'h':35})
    assert windows and max(w.end_time for w in windows)>=.33

def test_adaptive_sampler_does_not_add_full_video_recovery_points():
    from pipeline.video_ocr.sampler import AdaptiveSampler
    from pipeline.video_ocr.models import CandidateWindow, VideoMetadata
    meta = VideoMetadata(600.0, 1920, 1080, "1/90000", 30.0, False, 18000, ())
    pts = AdaptiveSampler(recovery_seconds=4.0, burst_fps=12.0, boundary_seconds=0.15, interior_fps=0.5).plan([], meta)
    assert pts == []
    pts2 = AdaptiveSampler(recovery_seconds=4.0, burst_fps=12.0, boundary_seconds=0.15, interior_fps=0.5).plan(
        [CandidateWindow(100.0, 106.0, "text_disappearance", None, 3)], meta)
    # Boundary points + a small number of interior points, not a 4-second full-video OCR grid.
    assert len(pts2) < 20

def test_sampler_trace_uses_actual_pts_for_vfr():
    from pipeline.video_ocr.sampler import AdaptiveFrameSampler
    from pipeline.video_ocr.models import VideoMetadata
    import numpy as np
    sampler=AdaptiveFrameSampler('ffmpeg',scan_fps=30,width=128,workers=1,diff_threshold=.018,global_textlike_threshold=.005,global_line_density_threshold=.05,hwaccel='none')
    frames=[np.full((72,128),30,np.uint8) for _ in range(5)]
    sampler._run_chunk=lambda *args,**kwargs: ((i/30.0,fr) for i,fr in enumerate(frames))
    pts=(0.0,.021,.071,.102,.150)
    meta=VideoMetadata(.16,128,72,'1/1000',30.0,True,5,pts)
    sampler.scan(__import__('pathlib').Path('dummy.mp4'),meta,{'x':0,'y':0,'w':100,'h':100})
    assert [x.frame_index for x in sampler.last_trace] == [0,1,2,3,4]
    assert [round(x.timestamp,3) for x in sampler.last_trace] == list(pts)


def test_boundary_refiner_supports_different_start_and_end_predicates():
    from pipeline.video_ocr.boundary import BoundaryRefiner
    from pipeline.video_ocr.models import TrackSegment, VideoMetadata
    meta=VideoMetadata(3.0,640,360,'1/30',30.0,False,90,tuple(i/30 for i in range(90)))
    seg=TrackSegment(1,5,10,meta.time_for_frame(5),meta.time_for_frame(10),'Hello',.95,[0,0,100,40],1)
    start_pred=lambda i,t,b: 5 <= i <= 20
    end_pred=lambda i,t,b: 5 <= i <= 14
    refined=BoundaryRefiner(None,meta,lambda i,t,b: False,start_predicate=start_pred,end_predicate=end_pred,max_search_seconds=1).refine(seg,4,20)
    assert refined.frame_start == 5
    assert refined.frame_end == 14


def test_fast_replacement_gate_does_not_fire_on_stable_same_text():
    from pipeline.video_ocr.fast_change import FastChangeDetector
    import cv2, numpy as np
    d=FastChangeDetector(.018,down_w=192)
    base=np.full((180,320),50,np.uint8)
    cue=base.copy(); cv2.putText(cue,'SAME TEXT',(45,155),cv2.FONT_HERSHEY_SIMPLEX,.9,240,2,cv2.LINE_AA)
    prev=cue.copy()
    replacements=0
    for dx in range(0,8):
        cur=np.roll(cue,dx,axis=1)
        a=d.analyze(prev,cur)
        replacements += int(a.replacement)
        prev=cur
    assert replacements == 0
