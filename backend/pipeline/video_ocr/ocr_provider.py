from __future__ import annotations
import logging
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .cuda_bootstrap import prepare_cuda_dlls, try_load_core
from .color import ColorAwarePreprocessor, ColorProfile
from .models import TextDetection
from .normalizer import normalize_text, similarity

log = logging.getLogger(__name__)


class OCRProvider(ABC):
    name = "abstract"
    supports_batch = False

    def __init__(self, color_profile: Optional[ColorProfile] = None):
        self.color_profile = color_profile or ColorProfile()
        self.pre = ColorAwarePreprocessor()

    @abstractmethod
    def recognize_batch(
        self,
        images: Sequence[np.ndarray],
        frame_indices: Sequence[int],
        timestamps: Sequence[float],
        offsets: Sequence[Tuple[int, int]],
        region_id: str = "roi",
        region_ids: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> List[TextDetection]:
        raise NotImplementedError

    def recognize_one(self, image, frame_index, timestamp, offset=(0, 0), region_id="roi", **kwargs):
        return self.recognize_batch([image], [frame_index], [timestamp], [offset], region_id=region_id, **kwargs)


class RapidPPV5Provider(OCRProvider):
    """PP-OCRv5 Mobile local provider.

    PaddleOCR 3.x is used with native list/batch inference when available. RapidOCR is
    the lightweight fallback and remains bounded by the engine batch size. Both paths
    are local and image-only.
    """

    name = "PP-OCRv5-mobile"
    supports_batch = True
    _lock = threading.Lock()
    _engine_cache = {}
    _engine_kind = None
    _engine_name = None
    _engine = None

    def __init__(self, color_profile=None, lang="auto", backend="auto", model_dir="", use_gpu=None, gpu_id=0):
        super().__init__(color_profile)
        requested = str(lang or "auto").lower().strip()
        self.requested_lang = requested
        self.lang = {"zh":"ch", "ja":"japan", "ko":"korean", "vi":"latin"}.get(requested, requested)
        # PP-OCRv5's base mobile recognizer is the safe automatic fallback (Chinese/English/Japanese).
        # Explicit source-language selection is preferred for multilingual/Latin subtitles.
        if self.lang == "auto": self.lang = "ch"
        self.backend = str(backend or "auto").lower()
        self.model_dir = str(model_dir or "")
        self.use_gpu = os.getenv("OCR_USE_GPU", "auto") if use_gpu is None else str(use_gpu)
        self.gpu_id = int(gpu_id or 0)
        self.gpu_active = False
        self.runtime_providers = []

    @staticmethod
    def _probe_actual_cuda_runtime() -> Dict[str, Any]:
        """Create a tiny ONNX Runtime session to verify CUDA is actually usable.

        `onnxruntime.get_available_providers()` can report CUDA even when the provider DLL cannot
        be loaded at session creation time. This probe prevents the engine from reporting a false
        GPU-active state and silently running the real OCR on CPU.
        """
        info: Dict[str, Any] = {"ok": False, "providers": [], "model": None, "error": None}
        try:
            import onnxruntime as ort
            info["available"] = list(ort.get_available_providers())
            from rapidocr import RapidOCR  # noqa: F401
            import rapidocr
            models = Path(rapidocr.__file__).resolve().parent / "models"
            det_candidates = sorted(models.glob("*PP-OCRv5*det*mobile*.onnx"))
            rec_candidates = sorted(models.glob("*PP-OCRv5*rec*mobile*.onnx"))
            if not det_candidates and not rec_candidates:
                candidates = sorted(models.glob("*.onnx"))
                det_candidates = candidates[:1]
            if not det_candidates:
                info["error"] = "Không tìm thấy ONNX detection model để probe CUDA"
                return info
            chosen = {"det": det_candidates[0], "rec": rec_candidates[0] if rec_candidates else None}
            target_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in info["available"] else ["CPUExecutionProvider"]
            sessions = {}
            for role, model in chosen.items():
                if model is None: continue
                try:
                    sess = ort.InferenceSession(str(model), providers=target_providers)
                    sessions[role] = list(sess.get_providers())
                except Exception as exc:
                    sessions[role] = [f"ERROR: {exc}"]
            det_ok = bool(sessions.get("det") and sessions["det"][0] == "CUDAExecutionProvider")
            rec_ok = bool(not chosen.get("rec") or (sessions.get("rec") and sessions["rec"][0] == "CUDAExecutionProvider"))
            info["providers"] = sessions.get("det", [])
            info["model"] = str(chosen.get("det"))
            info["models"] = {k: str(v) if v else None for k,v in chosen.items()}
            info["session_providers"] = sessions
            info["ok"] = bool(det_ok and rec_ok)
            info["det_cuda"] = det_ok
            info["rec_cuda"] = rec_ok
            if not info["ok"]:
                info["error"] = "Detection/recognition ORT session did not select CUDAExecutionProvider"
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    @property
    def engine_name(self) -> str:
        return str(self._engine_name or self.name)

    def _cache_key(self):
        return (self.backend, self.lang, self.model_dir, str(self.use_gpu).lower(), self.gpu_id)

    def _make_engine(self, provider: "RapidPPV5Provider"):
        cls = type(self)
        # Windows/NVIDIA: make CUDA/cuDNN DLLs from NVIDIA pip wheels or CUDA_PATH
        # visible BEFORE importing/creating RapidOCR/ONNX Runtime sessions.
        prepare_cuda_dlls(verbose=False)
        cuda_core = try_load_core()
        if cuda_core.get("loaded"):
            log.info("[OCR V2] preloaded CUDA DLLs=%d", len(cuda_core["loaded"]))
        key = self._cache_key()
        with cls._lock:
            cached = cls._engine_cache.get(key)
            if cached is not None:
                self._engine, self._engine_kind, self._engine_name, self.gpu_active, self.runtime_providers = cached
                return self._engine
            errors: List[str] = []

            def _cuda_ort_available():
                try:
                    import onnxruntime as ort
                    return "CUDAExecutionProvider" in list(ort.get_available_providers())
                except Exception:
                    return False

            # Auto mode chooses native PaddleOCR when it is installed because its pipeline exposes
            # list/batch inference. Otherwise RapidOCR remains the lightweight ONNX fallback.
            # Explicit backend selection always wins.
            if provider.backend in {"paddle", "paddleocr", "paddle3"}:
                prefer_paddle = True
            elif provider.backend in {"rapidocr", "rapid-ppocrv5"}:
                prefer_paddle = False
            else:
                try:
                    import importlib.util
                    have_paddle = bool(importlib.util.find_spec("paddle") and importlib.util.find_spec("paddleocr"))
                    paddle_cuda = False
                    if have_paddle:
                        import paddle
                        paddle_cuda = bool(getattr(paddle.device, "is_compiled_with_cuda", lambda: False)())
                    # Prefer native Paddle only when it can actually exploit CUDA. On CPU-only
                    # machines RapidOCR/ONNX is lighter and avoids an unnecessary heavyweight stack.
                    prefer_paddle = bool(have_paddle and paddle_cuda)
                except Exception:
                    prefer_paddle = False
            if prefer_paddle:
                try:
                    import paddle
                    from paddleocr import PaddleOCR
                    requested = provider.use_gpu.lower()
                    if requested == "auto":
                        use_gpu = bool(getattr(paddle.device, "is_compiled_with_cuda", lambda: False)())
                    else:
                        use_gpu = requested in {"1", "true", "yes", "cuda", "gpu"}
                    rec_model = "latin_PP-OCRv5_mobile_rec" if provider.lang.lower() in {"vi", "latin"} else "PP-OCRv5_mobile_rec"
                    # PaddleOCR's pipeline language selector is not the model-family identifier.
                    # For the Latin PP-OCRv5 recognizer use a supported Latin language selector
                    # while pinning the actual recognition model explicitly.
                    paddle_lang = "en" if provider.lang.lower() in {"vi", "latin"} else provider.lang
                    kw = dict(
                        ocr_version="PP-OCRv5",
                        text_detection_model_name="PP-OCRv5_mobile_det",
                        text_recognition_model_name=rec_model,
                        lang=paddle_lang,
                        device=f"gpu:{provider.gpu_id}" if use_gpu else "cpu",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        text_recognition_batch_size=16,
                        text_det_limit_side_len=1280,
                    )
                    if provider.model_dir:
                        root = Path(provider.model_dir)
                        kw["text_detection_model_dir"] = str(root / "det")
                        kw["text_recognition_model_dir"] = str(root / "rec")
                    engine = PaddleOCR(**kw)
                    try:
                        import onnxruntime as ort
                        providers = list(ort.get_available_providers())
                    except Exception:
                        providers = []
                    self.gpu_active = bool(use_gpu)
                    self.runtime_providers = providers
                    cls._engine_cache[key] = (engine, "paddle3", "PP-OCRv5-mobile-Paddle", self.gpu_active, self.runtime_providers)
                    self._engine, self._engine_kind, self._engine_name = engine, "paddle3", "PP-OCRv5-mobile-Paddle"
                    log.info("[OCR V2] backend=PaddleOCR3 native-batch device=%s", kw["device"])
                    return self._engine
                except Exception as exc:
                    errors.append(f"PaddleOCR3: {exc}")
                    if provider.backend in {"paddle", "paddleocr", "paddle3"}:
                        raise RuntimeError("Không khởi tạo được PaddleOCR 3.x: " + str(exc))

            try:
                prepare_cuda_dlls(verbose=False)
                from rapidocr import RapidOCR, EngineType, LangDet, LangRec, ModelType, OCRVersion
                use_cuda = provider.use_gpu.lower() in {"1", "true", "yes", "cuda", "gpu", "auto"}
                providers = []
                try:
                    import onnxruntime as ort
                    providers = list(ort.get_available_providers())
                except Exception:
                    pass
                if "CUDAExecutionProvider" not in providers:
                    use_cuda = False
                lang_key = provider.lang.lower().strip()
                if lang_key in {"vi", "latin", "en", "fr", "de", "es", "pt", "id", "it", "nl", "pl", "tr", "ms", "da", "fi", "sv", "no"}:
                    rec_lang = LangRec.LATIN
                else:
                    try:
                        rec_lang = getattr(LangRec, lang_key.upper())
                    except Exception:
                        rec_lang = LangRec.CH
                params = {
                    "Global.text_score": 0.08,
                    "Global.use_cls": False,
                    "Det.engine_type": EngineType.ONNXRUNTIME,
                    "Det.lang_type": LangDet.CH,
                    "Det.model_type": ModelType.MOBILE,
                    "Det.ocr_version": OCRVersion.PPOCRV5,
                    "Det.limit_side_len": 1280,
                    "Det.limit_type": "max",
                    "Det.box_thresh": 0.18,
                    "Det.max_candidates": 400,
                    "Rec.engine_type": EngineType.ONNXRUNTIME,
                    "Rec.lang_type": rec_lang,
                    "Rec.model_type": ModelType.MOBILE,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.rec_batch_num": int(os.getenv("OCR_REC_BATCH_NUM", "8")),
                    "EngineConfig.onnxruntime.use_cuda": use_cuda,
                    "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": provider.gpu_id,
                }
                engine = RapidOCR(params=params)
                if provider.use_gpu.lower() in {"1", "true", "yes", "cuda", "gpu"} and not use_cuda:
                    raise RuntimeError("OCR_USE_GPU yêu cầu CUDAExecutionProvider nhưng ONNX Runtime GPU/CUDA chưa khả dụng. Cài onnxruntime-gpu và CUDA/cuDNN tương thích.")
                actual = self._probe_actual_cuda_runtime() if use_cuda else {"ok": False, "providers": ["CPUExecutionProvider"], "error": None}
                if use_cuda and not actual.get("ok"):
                    raise RuntimeError(
                        "CUDAExecutionProvider có trong available providers nhưng không chạy được ở ORT session: "
                        + str(actual.get("error") or actual.get("providers"))
                    )
                self.gpu_active = bool(use_cuda and actual.get("ok"))
                self.runtime_providers = list(actual.get("providers") or providers)
                if use_cuda and "CUDAExecutionProvider" not in providers:
                    self.gpu_active = False
                cls._engine_cache[key] = (engine, "rapidocr", "RapidOCR-PP-OCRv5-mobile", self.gpu_active, self.runtime_providers)
                self._engine, self._engine_kind, self._engine_name = engine, "rapidocr", "RapidOCR-PP-OCRv5-mobile"
                # If an NVIDIA GPU is present and the target profile requires GPU, do not silently
                # spend 20+ minutes on CPU after a CUDA DLL failure.
                require_gpu = os.getenv("OCR_REQUIRE_GPU_FOR_TARGET", "1").strip().lower() in {"1","true","yes","on"}
                nvidia_detected = False
                try:
                    import shutil as _shutil
                    nvidia_detected = bool(_shutil.which("nvidia-smi"))
                except Exception:
                    pass
                if require_gpu and nvidia_detected and not self.gpu_active:
                    raise RuntimeError(
                        "NVIDIA được phát hiện nhưng RapidOCR chưa chạy CUDA. "
                        "Hãy chạy scripts\\setup_ocr_gpu.ps1 rồi khởi động lại backend."
                    )
                log.info("[OCR V2] backend=RapidOCR cuda_requested=%s gpu_active=%s providers=%s",
                         use_cuda, self.gpu_active, providers)
                return cls._engine
            except Exception as exc:
                errors.append(f"RapidOCR: {exc}")

            # Auto mode gets a final Paddle fallback if RapidOCR cannot initialize (e.g. a
            # missing ORT CUDA runtime). This keeps deployment resilient without changing an
            # explicitly requested backend.
            if provider.backend == "auto" and not prefer_paddle:
                try:
                    fallback = RapidPPV5Provider(color_profile=provider.color_profile, lang=provider.requested_lang,
                                                 backend="paddle", model_dir=provider.model_dir,
                                                 use_gpu=provider.use_gpu, gpu_id=provider.gpu_id)
                    engine = fallback._make_engine(fallback)
                    self._engine, self._engine_kind, self._engine_name = fallback._engine, fallback._engine_kind, fallback._engine_name
                    self.gpu_active = bool(fallback.gpu_active)
                    self.runtime_providers = list(fallback.runtime_providers or [])
                    cls._engine_cache[key] = (self._engine, self._engine_kind, self._engine_name, self.gpu_active, self.runtime_providers)
                    return engine
                except Exception as exc:
                    errors.append(f"PaddleOCR fallback: {exc}")

            raise RuntimeError("Không khởi tạo được OCRProvider PP-OCRv5 Mobile: " + " | ".join(errors))

    @staticmethod
    def _parse(result: Any) -> List[Tuple[Any, str, float]]:
        found: List[Tuple[Any, str, float]] = []

        def add(box, text, score):
            t = normalize_text(text)
            if not t:
                return
            try:
                s = float(score or 0.0)
            except Exception:
                s = 0.0
            found.append((box, t, s))

        def walk(obj):
            if obj is None:
                return

            # PaddleOCR 3.x Result exposes .json, which is intentionally parsed first.
            try:
                raw_json = getattr(obj, "json", None)
                if raw_json:
                    data = raw_json() if callable(raw_json) else raw_json
                    if isinstance(data, str):
                        import json
                        data = json.loads(data)
                    if data is not obj:
                        walk(data)
                        if found:
                            return
            except Exception:
                pass

            boxes = getattr(obj, "boxes", None)
            txts = getattr(obj, "txts", None)
            scores = getattr(obj, "scores", None)
            # Recognition-only RapidOCR output has txts/scores but no boxes. Preserve it for
            # the temporal recognizer path instead of dropping it as an unparseable result.
            if boxes is None and txts is not None:
                tt = list(txts) if isinstance(txts, (list, tuple)) else [txts]
                ss = list(scores) if isinstance(scores, (list, tuple)) else []
                for i, t in enumerate(tt):
                    add(None, t, ss[i] if i < len(ss) else 0.0)
                return
            if boxes is not None and txts is not None:
                bb = list(boxes)
                tt = list(txts)
                ss = list(scores) if scores is not None else []
                for i, t in enumerate(tt):
                    add(bb[i] if i < len(bb) else None, t, ss[i] if i < len(ss) else 0.0)
                return

            if isinstance(obj, dict):
                texts = obj.get("rec_texts")
                scores = obj.get("rec_scores")
                boxes = obj.get("rec_boxes") or obj.get("dt_polys") or obj.get("rec_polys") or obj.get("dt_boxes")
                if isinstance(texts, (list, tuple)):
                    for i, t in enumerate(texts):
                        box = boxes[i] if isinstance(boxes, (list, tuple)) and i < len(boxes) else None
                        score = scores[i] if isinstance(scores, (list, tuple)) and i < len(scores) else 0.0
                        add(box, t, score)
                    return
                for v in obj.values():
                    walk(v)
                return

            if isinstance(obj, (list, tuple)):
                for x in obj:
                    if isinstance(x, (list, tuple)) and len(x) >= 3 and isinstance(x[1], (str, bytes)):
                        add(x[0], x[1], x[2])
                    elif isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[-1], (list, tuple)) and len(x[-1]) >= 2:
                        add(x[0], x[-1][0], x[-1][1])
                    else:
                        walk(x)

        walk(result)
        return found

    @staticmethod
    def _xyxy(box, ox, oy):
        if box is None:
            return None
        try:
            a = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if a.shape[0] < 2:
            return None
        x0 = float(a[:, 0].min() + ox)
        y0 = float(a[:, 1].min() + oy)
        x1 = float(a[:, 0].max() + ox)
        y1 = float(a[:, 1].max() + oy)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    @staticmethod
    def _best(detections: List[Tuple[Any, str, float]]) -> List[Tuple[Any, str, float]]:
        # Preserve all text boxes for high recall, only dedupe exact boxes within one inference.
        return detections

    def _paddle_batch(self, images: Sequence[np.ndarray], batch_size: int) -> List[List[Tuple[Any, str, float]]]:
        results = list(self._engine.predict(list(images), batch_size=int(batch_size)))
        # Paddle returns one result object per input image. Pad safely on unexpected providers.
        parsed = [self._parse(r) for r in results]
        while len(parsed) < len(images):
            parsed.append([])
        return parsed[:len(images)]


    @staticmethod
    def _latin_score(text: str) -> float:
        s = normalize_text(text)
        if not s:
            return 0.0
        import unicodedata
        letters = 0
        latin = 0
        for ch in s:
            if not ch.isalpha():
                continue
            letters += 1
            name = unicodedata.name(ch, "")
            if "LATIN" in name:
                latin += 1
        return latin / max(1, letters)

    @classmethod
    def _choose_auto_results(cls, primary: List[Tuple[Any, str, float]],
                             secondary: List[Tuple[Any, str, float]]) -> List[Tuple[Any, str, float]]:
        if not primary:
            return secondary
        if not secondary:
            return primary
        p_conf = max((float(x[2]) for x in primary), default=0.0)
        s_conf = max((float(x[2]) for x in secondary), default=0.0)
        p_text = " ".join(x[1] for x in primary)
        s_text = " ".join(x[1] for x in secondary)
        ps = cls._latin_score(p_text)
        ss = cls._latin_score(s_text)
        # For Latin subtitles, prefer the Latin recognizer when it is not materially weaker.
        if ss >= 0.65 and (ps < 0.35 or s_conf >= p_conf - 0.08):
            return secondary
        return primary if p_conf >= s_conf - 0.05 else secondary

    def _rapid_one(self, image: np.ndarray) -> List[Tuple[Any, str, float]]:
        try:
            return self._best(self._parse(self._engine(image)))
        except Exception as exc:
            log.debug("RapidOCR inference failed: %s", exc)
            return []

    def recognize_batch(
        self,
        images,
        frame_indices,
        timestamps,
        offsets,
        region_id="roi",
        region_ids=None,
        batch_size=16,
        **kwargs,
    ) -> List[TextDetection]:
        if not images:
            return []
        self._make_engine(self)
        bs = max(1, min(64, int(batch_size or 16)))

        # First pass uses original crops. Color guidance is a targeted retry only for weak/empty
        # OCR, not a second inference for every frame.
        parsed_by_owner: Dict[int, List[Tuple[Any, str, float]]] = {i: [] for i in range(len(images))}

        def infer_native(imgs: Sequence[np.ndarray], owners: Sequence[int]):
            if not imgs:
                return
            if self._engine_kind == "paddle3":
                for start in range(0, len(imgs), bs):
                    batch = list(imgs[start:start + bs])
                    try:
                        batch_results = self._paddle_batch(batch, min(bs, len(batch)))
                    except Exception as exc:
                        log.warning("PaddleOCR native batch failed; per-image fallback for this batch: %s", exc)
                        batch_results = [self._parse(self._engine(img)) for img in batch]
                    for local_i, dets in enumerate(batch_results):
                        owner = owners[start + local_i]
                        parsed_by_owner[owner].extend(self._best(dets))
            else:
                # RapidOCR's public API is image-at-a-time, so exploit bounded concurrency instead
                # of pretending batch_size is supported. ONNX Runtime inference is thread-safe;
                # cap workers to keep CPU/VRAM bounded. GPU defaults to 2 concurrent calls, CPU to 2.
                from concurrent.futures import ThreadPoolExecutor
                workers = max(1, min(int(os.getenv("OCR_RAPID_WORKERS", "2")), len(imgs), 4))
                if workers == 1:
                    for owner, img in zip(owners, imgs):
                        parsed_by_owner[owner].extend(self._rapid_one(img))
                else:
                    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr-rapid") as pool:
                        futures = {owner: pool.submit(self._rapid_one, img) for owner, img in zip(owners, imgs)}
                        for owner in owners:
                            try:
                                parsed_by_owner[owner].extend(futures[owner].result())
                            except Exception as exc:
                                log.debug("RapidOCR worker failed for frame %s: %s", owner, exc)

        base_owners = list(range(len(images)))
        infer_native(list(images), base_owners)
        if self.requested_lang == "auto":
            # AUTO must resolve script reliably. A CJK recognizer can return high-confidence
            # garbage for Latin/Vietnamese subtitles, so only a confidence threshold is not enough.
            # Run the Latin companion on anchor/full-OCR frames whose primary text is not clearly
            # Latin, then select the script-consistent result. The full OCR frame count is small;
            # interior frames are handled by a language-specific recognition-only provider later.
            primary = {i: list(v) for i, v in parsed_by_owner.items()}
            candidates = []
            for i, dets in primary.items():
                text = " ".join(x[1] for x in dets)
                latin = self._latin_score(text)
                best_conf = max((float(x[2]) for x in dets), default=0.0)
                if (not dets) or latin < 0.55 or best_conf < 0.72:
                    candidates.append(i)
            if candidates:
                try:
                    secondary_provider = RapidPPV5Provider(
                        color_profile=self.color_profile, lang="latin", backend=self.backend,
                        model_dir=self.model_dir, use_gpu=self.use_gpu, gpu_id=self.gpu_id,
                    )
                    secondary_provider._make_engine(secondary_provider)
                    secondary = {i: [] for i in candidates}
                    if secondary_provider._engine_kind == "paddle3":
                        for start in range(0, len(candidates), bs):
                            owners = candidates[start:start+bs]
                            parsed = secondary_provider._paddle_batch([images[i] for i in owners], len(owners))
                            for j, owner in enumerate(owners):
                                secondary[owner] = parsed[j]
                    else:
                        from concurrent.futures import ThreadPoolExecutor
                        workers = max(1, min(int(os.getenv("OCR_RAPID_WORKERS", "2")), len(candidates), 4))
                        if workers == 1:
                            for owner in candidates:
                                secondary[owner] = self._best(secondary_provider._rapid_one(images[owner]))
                        else:
                            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr-latin") as pool:
                                futures = {owner: pool.submit(secondary_provider._rapid_one, images[owner]) for owner in candidates}
                                for owner in candidates:
                                    try:
                                        secondary[owner] = self._best(futures[owner].result())
                                    except Exception:
                                        secondary[owner] = []
                    for i in candidates:
                        parsed_by_owner[i] = self._choose_auto_results(primary[i], secondary.get(i, []))
                except Exception as exc:
                    log.debug("Auto Latin companion skipped: %s", exc)

        if self.color_profile.enabled:
            weak = []
            for i, dets in parsed_by_owner.items():
                best_conf = max((float(x[2]) for x in dets), default=0.0)
                if not dets or best_conf < 0.72:
                    weak.append(i)
            if weak:
                retry_store={i:[] for i in weak}
                # Targeted second inference only for weak frames; merge competing results instead
                # of appending duplicates to the observation stream.
                original_store=parsed_by_owner
                parsed_by_owner=retry_store
                infer_native([self.pre.apply(images[i], self.color_profile) for i in weak], weak)
                retry_store=parsed_by_owner
                parsed_by_owner=original_store
                for i in weak:
                    parsed_by_owner[i]=self._merge_variant_results(parsed_by_owner.get(i,[]),retry_store.get(i,[]))

        out: List[TextDetection] = []
        for i, (fi, ts, (ox, oy)) in enumerate(zip(frame_indices, timestamps, offsets)):
            rid = region_ids[i] if region_ids is not None and i < len(region_ids) else region_id
            for box, text, conf in self._best(parsed_by_owner.get(i, [])):
                xy = self._xyxy(box, ox, oy)
                if xy is None:
                    continue
                c = max(0.0, min(1.0, float(conf)))
                out.append(TextDetection(
                    text=text,
                    confidence=c,
                    bbox=xy,
                    frame_index=int(fi),
                    timestamp=float(ts),
                    region_id=str(rid or "roi"),
                    source=self.engine_name,
                    needs_review=c < 0.55,
                ))
        return out

    def _rapid_rec_once(self, image: np.ndarray) -> Tuple[str, float]:
        if image is None or image.size == 0:
            return "", 0.0
        try:
            result = self._engine(image, use_det=False, use_cls=False, use_rec=True)
            texts = getattr(result, "txts", None)
            scores = getattr(result, "scores", None)
            if texts:
                texts=[normalize_text(x) for x in list(texts) if normalize_text(x)]
                scores=list(scores) if scores is not None else []
                if not texts:
                    return "",0.0
                # Keep the whole recognition result. A localized crop can contain two subtitle
                # lines; choosing one highest-confidence item silently deletes the other line.
                joined=" ".join(texts)
                usable=[float(scores[i]) for i in range(min(len(scores),len(texts)))]
                score=sum(usable)/len(usable) if usable else 0.0
                return normalize_text(joined), score
            parsed=self._parse(result)
            if parsed:
                _box,text,score=max(parsed,key=lambda x:float(x[2]))
                return normalize_text(text),float(score)
        except Exception as exc:
            log.debug("RapidOCR recognition-only failed: %s", exc)
        return "",0.0

    @staticmethod
    def _enhance_rec_crop(image: np.ndarray) -> np.ndarray:
        """Conservative enhancement for weak OCR only; avoids aggressive binarization."""
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        gray=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        clahe=cv2.createCLAHE(clipLimit=1.6,tileGridSize=(8,8))
        enhanced=clahe.apply(gray)
        blur=cv2.GaussianBlur(enhanced,(0,0),0.8)
        sharp=cv2.addWeighted(enhanced,1.35,blur,-0.35,0)
        return cv2.cvtColor(sharp,cv2.COLOR_GRAY2BGR)

    def _rapid_rec_one(self, image: np.ndarray) -> Tuple[str, float]:
        """Recognition-only inference with a single weak-result retry on a conservative variant."""
        if image is None or image.size == 0:
            return "",0.0
        h,w=image.shape[:2]
        crop=image
        if h < 28 or w < 160:
            scale=max(1.0,min(2.0,48.0/max(1.0,h)))
            crop=cv2.resize(image,(max(32,int(round(w*scale))),max(24,int(round(h*scale)))),interpolation=cv2.INTER_CUBIC)
        text,score=self._rapid_rec_once(crop)
        weak=(not text) or score < 0.78 or len(text) <= 2
        if not weak:
            return text,score
        enhanced=self._enhance_rec_crop(crop)
        text2,score2=self._rapid_rec_once(enhanced)
        if text2 and (score2 > score + 0.03 or not text):
            return text2,score2
        return text,score

    def recognize_text_only_batch(
        self, images: Sequence[np.ndarray], frame_indices: Sequence[int], timestamps: Sequence[float],
        offsets: Optional[Sequence[Tuple[int,int]]] = None, batch_size: int = 16, **kwargs
    ) -> List[Tuple[int, float, str, float]]:
        """Recognize already-localized line crops without a detector when the backend supports it.

        RapidOCR exposes a true recognition-only path. PaddleOCR 3.x is kept on its native batch
        pipeline for these tiny line crops; the detector sees only the already-localized crop, so this
        remains much cheaper than full-frame OCR and, crucially, does not silently return empty text.
        """
        if not images:
            return []
        self._make_engine(self)
        if self._engine_kind == "paddle3":
            parsed_batches = []
            for start in range(0, len(images), max(1, min(64, int(batch_size or 16)))):
                batch = list(images[start:start + max(1, min(64, int(batch_size or 16)))])
                try:
                    parsed_batches.extend(self._paddle_batch(batch, min(max(1, int(batch_size or 16)), len(batch))))
                except Exception:
                    parsed_batches.extend([[] for _ in batch])
            vals=[]
            for dets in parsed_batches:
                if dets:
                    best=max(dets, key=lambda x: float(x[2]))
                    vals.append((normalize_text(best[1]), float(best[2])))
                else:
                    vals.append(("",0.0))
        else:
            default_workers = "2"
            workers = max(1, min(int(os.getenv("OCR_RAPID_REC_WORKERS", os.getenv("OCR_RAPID_WORKERS", default_workers))), len(images), 4))
            if workers == 1:
                vals=[self._rapid_rec_one(img) for img in images]
            else:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr-rec") as pool:
                    vals=list(pool.map(self._rapid_rec_one, images))
        return [(int(frame_indices[i]), float(timestamps[i]), text, float(score)) for i,(text,score) in enumerate(vals)]

    def runtime_info(self) -> Dict[str, Any]:
        return {
            "engine": self.engine_name,
            "kind": self._engine_kind,
            "gpu_active": bool(self.gpu_active),
            "runtime_providers": list(self.runtime_providers or []),
            "language": self.lang,
            "recognition_only": self._engine_kind == "rapidocr",
            "cuda_dlls_ready": bool(try_load_core().get("loaded")),
        }


def create_ocr_provider(model: str = "", provider: str = "local", color_profile: Optional[ColorProfile] = None, **kwargs) -> OCRProvider:
    p = str(provider or "local").lower().strip()
    if p not in {"local", "rapidocr", "pp-ocr", "ppocr", "rapid-ppocrv5"}:
        raise ValueError(f"OCR provider không hợp lệ trong V2: {provider}. Chỉ dùng local PP-OCRv5 Mobile.")
    return RapidPPV5Provider(color_profile=color_profile, **kwargs)
