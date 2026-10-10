"""main.py - FlowAI Backend (BẢN SẠCH).
CHỈ HỖ TRỢ 4 PROVIDER: Google, xKiro, Groq, Gemini.
Đã loại: OpenAI, DeepL, DeepSeek, ElevenLabs, Edge TTS.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import inspect
import asyncio
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import logging
import re
import zipfile
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from storage import storage
from pipeline import audio as audio_mod
from pipeline import stt as stt_mod
from pipeline import translate as translate_mod
from pipeline import render as render_mod
from pipeline import subtitle as subtitle_mod
from pipeline import tts as tts_mod
from pipeline.video_ocr import VideoOCREngine, Exporter, GroundTruthEvaluator

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, "INFO"),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
try:
    logger.info("[CONFIG] STT model=%s | quality=%s | storage=%s", getattr(config, "GROQ_STT_MODEL", "whisper-large-v3-turbo"), getattr(config, "GROQ_STT_QUALITY_MODEL", "whisper-large-v3"), config.DATA_ROOT)
except Exception:
    pass

# ============================================================
# STORAGE INIT
# ============================================================
if not hasattr(subtitle_mod, "lock_source_timing"):
    def _lock_source_timing(segments):
        return segments
    subtitle_mod.lock_source_timing = _lock_source_timing

try:
    _test_id = "test_storage"
    storage.upsert_project({"id": _test_id, "name": "test"})
    storage.delete_project(_test_id)
    logger.info("✅ Storage hoạt động tốt")
except Exception as e:
    logger.error(f"❌ Storage KHÔNG HOẠT ĐỘNG: {e}")

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(title="FlowAI Backend - Xkira AI Enhanced", version="2.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

processing_locks = {}
cancel_flags = {}
# Explicit pipeline phase prevents /tts-all from racing an in-flight translation/STT job.
processing_phases = {}
phase_lock = threading.RLock()
lock_lock = threading.Lock()

# OCR realtime event bus. Events are in-memory per running server/job and are never
# written to persistent storage; final OCR JSON/SRT remain the durable artifacts.
from collections import defaultdict, deque
from queue import Queue, Empty
_ocr_event_lock = threading.RLock()
_ocr_event_history = defaultdict(lambda: deque(maxlen=300))
_ocr_event_subscribers = defaultdict(set)
_ocr_event_seq = defaultdict(int)

def _publish_ocr_event(pid: str, event: dict):
    payload = dict(event or {})
    with _ocr_event_lock:
        _ocr_event_seq[pid] += 1
        payload["seq"] = _ocr_event_seq[pid]
        payload.setdefault("ts", time.time())
        _ocr_event_history[pid].append(payload)
        for q in list(_ocr_event_subscribers.get(pid, ())):
            try: q.put_nowait(payload)
            except Exception: pass
    return payload

def _clear_ocr_event_bus(pid: str):
    with _ocr_event_lock:
        _ocr_event_history.pop(pid, None)
        _ocr_event_seq.pop(pid, None)
        _ocr_event_subscribers.pop(pid, None)

STT_TIMEOUT_SECONDS = 3600
TRANSLATION_TIMEOUT_SECONDS = 3600
PROGRESS_HEARTBEAT_SECONDS = 1.0

# ============================================================
# LOCK HELPERS
# ============================================================
def acquire_processing_lock(project_id: str) -> bool:
    with lock_lock:
        if processing_locks.get(project_id, False):
            return False
        processing_locks[project_id] = True
        return True

def release_processing_lock(project_id: str):
    with lock_lock:
        processing_locks[project_id] = False

def set_processing_phase(project_id: str, phase: str):
    with phase_lock:
        if phase:
            processing_phases[project_id] = str(phase)
        else:
            processing_phases.pop(project_id, None)

def get_processing_phase(project_id: str) -> str:
    with phase_lock:
        return str(processing_phases.get(project_id, "idle"))

def is_cancelled(project_id: str) -> bool:
    return cancel_flags.get(project_id, False)

# ============================================================
# LANGUAGE & PROVIDER MAPS (CHỈ 4 PROVIDER)
# ============================================================
LANG_NAME_TO_CODE = {
    "Tự động nhận diện": "auto",
    "Tiếng Anh": "en",
    "Tiếng Trung": "zh",
    "Tiếng Hàn": "ko",
    "Tiếng Nhật": "ja",
    "Tiếng Việt": "vi",
}
LANG_TARGET_NAME = {
    "Tiếng Việt": "Tiếng Việt (Vietnamese)",
    "English": "English",
    "中文": "Chinese",
    "日本語": "Japanese",
    "한국어": "Korean",
}
TRANSLATE_MAP = {
    "Xkira AI": "xkira",
    "Google Translate": "google",
    "Groq LLM (LLaMA)": "groq",
    "Groq LLM": "groq",
    "Gemini LLM": "gemini",
}
STT_MAP = {
    "Groq Whisper": "groq",
    "Gemini STT": "gemini",
    "Google STT": "google",
}
TTS_MAP = {
    "xkira_tts": "xkira_tts",
    "google_tts": "google_tts",
    "Xkira TTS": "xkira_tts",
    "Google TTS": "google_tts",
    "Google Cloud TTS": "google_tts",
    "edge_tts": "edge_tts",
    "Microsoft Edge TTS": "edge_tts",
    "Edge TTS": "edge_tts",
}

_ALLOWED_STT = {"groq", "gemini", "google"}
_ALLOWED_TRANSLATE = {"groq", "xkira", "gemini", "google"}
_ALLOWED_TTS = {"xkira_tts", "edge_tts"}

PROVIDER_OPTIONS = {
    "stt": [o for o in config.PROVIDER_OPTIONS.get("stt", []) if o.get("id") in _ALLOWED_STT],
    "translate": [o for o in config.PROVIDER_OPTIONS.get("translate", []) if o.get("id") in _ALLOWED_TRANSLATE],
    "tts": [o for o in config.PROVIDER_OPTIONS.get("tts", []) if o.get("id") in _ALLOWED_TTS],
    "ocr": [o for o in config.PROVIDER_OPTIONS.get("ocr", []) if o.get("id") == "local"],
}

# ============================================================
# VOICE ID MAP
# ============================================================
VOICE_ID_MAP = {
    # Human-readable aliases only. Actual xKiro IDs are resolved from the live catalog.
    "female_sweet": "Sweet Female (Vietnamese)",
    "male_confident": "Confident Male (Vietnamese)",
    "female_gentle": "Gentle Female (Vietnamese)",
    "female_standard": "Standard Female (Vietnamese)",
    "Sweet Female": "Sweet Female (Vietnamese)",
    "Confident Male": "Confident Male (Vietnamese)",
    "Gentle Female": "Gentle Female (Vietnamese)",
    "Standard Female": "Standard Female (Vietnamese)",
    "Nữ — Ngọt ngào (Xkira)": "Sweet Female (Vietnamese)",
    "Nam — Tự tin (Xkira)": "Confident Male (Vietnamese)",
    "Nữ — Nhẹ nhàng (Xkira)": "Gentle Female (Vietnamese)",
    "Nữ — Chuẩn mực (Xkira)": "Standard Female (Vietnamese)",
}

def _resolve_voice_id(voice: str) -> str:
    v = str(voice or "").strip()
    v = VOICE_ID_MAP.get(v, v)
    if not v:
        v = "Sweet Female (Vietnamese)"
    # tts.py performs exact live-catalog resolution and deliberately raises on mismatch.
    return tts_mod._resolve_voice_id(v, "xkira_tts")


def _resolve_google_voice(voice: str) -> str:
    v = str(voice or "").strip()
    google_map = dict(getattr(config, "GOOGLE_TTS_VOICE_MAP", {}) or {})
    if v in google_map:
        return google_map[v]
    if re.match(r"^[a-z]{2}-[A-Z]{2}-", v):
        return v
    return "vi-VN-Standard-A"

def _resolve_voice_for_provider(voice: str, tts_id: str) -> str:
    if tts_id == "xkira_tts":
        return _resolve_voice_id(voice)
    if tts_id == "google_tts":
        return _resolve_google_voice(voice)
    if tts_id == "edge_tts":
        edge_map = dict(getattr(config, "EDGE_TTS_VOICE_MAP", {}) or {})
        v = str(voice or "").strip()
        return edge_map.get(v, v or getattr(config, "DEFAULT_EDGE_VOICE", "vi-VN-NamMinhNeural"))
    return str(voice or "").strip()

# ============================================================
# RESOLVE HELPERS
# ============================================================
def _resolve_translate_id(name: str) -> str:
    return TRANSLATE_MAP.get(name, name)

def _resolve_stt_id(name: str) -> str:
    resolved = STT_MAP.get(name, name)
    # xKiro currently does not expose speech-to-text in its public API.
    # Never call the old /v1/stt route; translate legacy saved selections
    # to Groq so existing projects keep working.
    if resolved == "xkira":
        return "groq"
    return resolved


def _select_stt_provider(requested: str, keys: dict) -> str:
    """Return an actually available STT provider. xKiro is translation/TTS only."""
    resolved = STT_MAP.get(requested, requested)
    if resolved == "xkira":
        if keys.get("groq") or keys.get("groq_keys"):
            logger.warning("STT xKiro không khả dụng; tự động chuyển sang Groq Whisper.")
            return "groq"
        if keys.get("gemini"):
            logger.warning("STT xKiro không khả dụng; tự động chuyển sang Gemini STT.")
            return "gemini"
        raise RuntimeError("xKiro hiện không hỗ trợ STT. Hãy thêm GROQ_API_KEY hoặc GEMINI_API_KEY để nhận diện giọng nói.")
    return resolved

def _resolve_tts_settings(project: dict, keys: dict):
    tts_provider = project.get("tts_provider") or config.DEFAULT_PROVIDERS.get("tts_provider", "xkira_tts")
    tts_id = TTS_MAP.get(tts_provider, tts_provider)
    voice_raw = project.get("voice", "female_sweet")
    voice = _resolve_voice_for_provider(voice_raw, tts_id)
    if tts_id == "xkira_tts":
        if not config.get_xkira_api_keys():
            raise RuntimeError("Chưa cấu hình XKIRA_API_KEY_1...XKIRA_API_KEY_8")
        # Do not pin rendering to one account. tts.py scheduler selects a healthy account per request.
        return tts_id, voice, None
    if tts_id == "google_tts":
        api_key = keys.get("google_tts") or keys.get("google")
        if not api_key:
            raise RuntimeError("Chưa cấu hình Google Cloud credentials.")
        return tts_id, voice, api_key
    if tts_id == "edge_tts":
        return tts_id, voice, None
    raise RuntimeError(f"TTS provider '{tts_id}' không được hỗ trợ (chỉ xKiro/Edge).")


def _project_or_404(pid: str) -> dict:
    p = storage.get_project(pid)
    if not p:
        raise HTTPException(404, "Không tìm thấy dự án")
    return p

def _set_error(pid: str, message: str):
    storage.update_project(pid, status="Lỗi", error=message or "Lỗi không xác định")

def _update_progress(pid, step, total, name, details=None, percentage=None):
    if percentage is None:
        percentage = int((step / total) * 100) if total > 0 else 0
    storage.update_project(pid, progress={
        "step": step,
        "total_steps": total,
        "step_name": name,
        "percentage": max(0, min(100, int(percentage))),
        "details": details or {},
    })

def _run_with_timeout(func, timeout_seconds: float, *, label: str, **kwargs):
    result, error = {}, {}
    def runner():
        try:
            result["value"] = func(**kwargs)
        except BaseException as exc:
            error["value"] = exc
    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise TimeoutError(f"{label} timeout sau {int(timeout_seconds)}s")
    if "value" in error:
        raise error["value"]
    return result.get("value")

def _parse_rate(v, default=1.0) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v or "").strip()
    if not s:
        return default
    if s.endswith("%"):
        try:
            return round(1.0 + float(s[:-1]) / 100.0, 4)
        except Exception:
            return default
    try:
        return float(s)
    except Exception:
        return default

def _parse_pitch(v, default=0.0) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v or "").strip().replace("Hz", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return default

# ============================================================
# PYDANTIC MODELS
# ============================================================
class KeysIn(BaseModel):
    gemini: Optional[str] = None
    groq: Optional[str] = None
    xkira: Optional[str] = None
    xkira_tts: Optional[str] = None
    google_tts: Optional[str] = None

class TranscriptUpdateIn(BaseModel):
    texts: List[str]

class RenderIn(BaseModel):
    keep_original_audio: bool = False
    rate: Any = 1.0
    pitch: Any = 0.0
    include_subtitle: bool = True
    include_dubbing: bool = True
    quality: str = "1080p"
    orig_volume: float = 0.2
    dub_volume: float = 1.0
    reduce_original_voice: bool = False
    original_voice_reduction: float = 0.0

class OcrRegionIn(BaseModel):
    x: float = Field(20, ge=0, le=100)
    y: float = Field(74, ge=0, le=100)
    w: float = Field(60, ge=0, le=100)
    h: float = Field(16, ge=0, le=100)

class OcrIn(BaseModel):
    region: OcrRegionIn = OcrRegionIn()
    retranslate: bool = False

class TtsPreviewIn(BaseModel):
    text: str
    voice: str
    language: str = "vi"
    provider: str = "xkira_tts"
    rate: Any = 1.0
    pitch: Any = 0.0
    segment_index: Optional[int] = None

class TtsSegmentIn(BaseModel):
    segment_index: int
    text: Optional[str] = None
    voice: str = "female_sweet"
    provider: str = "xkira_tts"
    rate: Any = 1.0
    pitch: Any = 0.0
    language: str = "vi"
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    sync_to_original_timing: bool = True
    read_original_audio: bool = False
    use_translated_text_only: bool = True

class TtsAllIn(BaseModel):
    voice: Optional[str] = None
    provider: Optional[str] = None
    rate: Any = 1.0
    pitch: Any = 0.0

# ============================================================
# FRONTEND PATH
# ============================================================
_APP_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd()
SEARCH_PATHS = [
    # Prefer the packaged, versioned frontend beside this backend so an older
    # frontend under DATA_ROOT cannot silently override the separated UI.
    _APP_DIR / "frontend",
    config.DATA_ROOT / "frontend",
    _CWD / "frontend",
    _APP_DIR,
    _CWD,
]
FRONTEND_DIR = None
FRONTEND_FILE = None
for p in SEARCH_PATHS:
    if p.is_dir():
        html_files = list(p.glob("*.html"))
        if html_files:
            FRONTEND_DIR = p
            preferred = [p / "Lingvo_xKiro_Qwen_fixed.html", p / "Lingvo.html", p / "lingvo.html", p / "test.html"]
            FRONTEND_FILE = next((x for x in preferred if x.is_file()), html_files[0])
            break
if FRONTEND_FILE is None:
    # Single-file Lingvo frontend chạy cùng backend.
    candidates = [
        _APP_DIR / "Lingvo.html",
        _APP_DIR / "lingvo.html",
        _APP_DIR / "Lingvo_Backend_Fixed.html",
        _APP_DIR / "test.html",
        _APP_DIR / "Lingvo_Single_File(1).html",
    ]
    FRONTEND_FILE = next((p for p in candidates if p.is_file()), candidates[0])
logger.info(f"🌐 Frontend path: {FRONTEND_FILE} | exists={FRONTEND_FILE.is_file()}")

# ============================================================
# ROUTES: Health & Frontend
# ============================================================
@app.get("/", include_in_schema=False)
def root():
    if FRONTEND_FILE.is_file():
        return FileResponse(str(FRONTEND_FILE), media_type="text/html")
    return {"message": "FlowAI Video Translator API", "version": "2.3.0",
            "docs": "/docs", "health": "/health"}

@app.get("/test.html", include_in_schema=False)
def frontend_page():
    if not FRONTEND_FILE.is_file():
        raise HTTPException(404, "Không tìm thấy test.html")
    return FileResponse(str(FRONTEND_FILE), media_type="text/html")

if FRONTEND_DIR is not None and FRONTEND_DIR.is_dir() and FRONTEND_DIR.name == "frontend":
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=False), name="frontend")

@app.get("/health")
def health_check():
    return {"status": "healthy", "storage": True, "database": True, "version": "2.3.0"}

@app.get("/api/health")
def api_health_check():
    return health_check()

# ============================================================
# ROUTES: Settings
# ============================================================
@app.get("/api/settings/keys")
def get_keys():
    data = config.get_api_keys()
    out = {k: config.masked(v) for k, v in data.items() if k != "xkira_keys"}
    # One generic xKiro key pool serves translation/TTS. OCR is local-only in V2.
    if config.get_xkira_api_keys():
        out["xkira_key_count"] = min(8, len(config.get_xkira_api_keys()))
    return out

@app.post("/api/settings/keys")
def set_keys(body: KeysIn):
    saved = config.save_api_keys(body.dict(exclude_none=True))
    return {k: config.masked(v) for k, v in saved.items()}

# Compatibility API used by Lingvo.html. Settings are persisted separately from API keys.
_PROVIDER_SETTINGS_DEFAULTS = {
    "stt_provider": "groq",
    "ocr_provider": "local",
    "translate_provider": "groq",
    "tts_provider": "xkira_tts",
    "translate_refine": False,
    "edge_voice": getattr(config, "DEFAULT_EDGE_VOICE", "vi-VN-HoaiMyNeural"),
    "xkira_voice": "",
    "tts_speed": 1.0,
    "provider_config": {},
    "stt_model": getattr(config, "DEFAULT_MODELS", {}).get("stt", ""),
    "ocr_model": getattr(config, "DEFAULT_MODELS", {}).get("ocr", ""),
    "translate_model": getattr(config, "DEFAULT_MODELS", {}).get("translate", ""),
    "tts_model": getattr(config, "DEFAULT_MODELS", {}).get("tts", "xkiro-voice"),
    "full_model": getattr(config, "DEFAULT_MODELS", {}).get("full", ""),
    "ocr_mode": getattr(config, "OCR_MODE", "HIGH_RECALL"),
}

def _provider_settings_path() -> Path:
    root = Path(getattr(config, "DATA_ROOT", _APP_DIR))
    root.mkdir(parents=True, exist_ok=True)
    return root / "provider_settings.json"

def _read_provider_settings() -> dict:
    path = _provider_settings_path()
    data = dict(_PROVIDER_SETTINGS_DEFAULTS)
    try:
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                data.update(saved)
    except Exception as exc:
        logger.warning("Không đọc được provider_settings.json: %s", exc)
    return data

@app.get("/api/settings/providers")
def get_provider_settings():
    return _read_provider_settings()

@app.post("/api/settings/providers")
def save_provider_settings(body: dict):
    current = _read_provider_settings()
    allowed = set(_PROVIDER_SETTINGS_DEFAULTS)
    for key, value in (body or {}).items():
        if key in allowed:
            current[key] = value
    current["stt_provider"] = current.get("stt_provider") if current.get("stt_provider") in {"groq", "gemini", "google"} else "groq"
    current["ocr_provider"] = "local"
    current["translate_provider"] = current.get("translate_provider") if current.get("translate_provider") in {"groq", "xkira", "gemini", "google"} else "groq"
    current["tts_provider"] = current.get("tts_provider") if current.get("tts_provider") in {"xkira_tts", "edge_tts"} else "xkira_tts"
    _provider_settings_path().write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current

@app.get("/api/settings/{provider}/models")
def get_provider_models(provider: str):
    provider = str(provider or "").strip().lower()
    if provider == "local":
        return ["PP-OCRv5-mobile"]
    if provider == "xkira":
        try:
            models = translate_mod.get_xkira_qwen_models(force=False, include_paid=getattr(config, "XKIRA_USE_PAID_QWEN", True))
            return [str(m.get("id")) for m in models if m.get("id")]
        except Exception as exc:
            logger.warning("Không lấy được model xKiro: %s", exc)
            return []
    return []

@app.get("/api/tts/voices")
def get_xkira_voices(language: str = "vi", limit: int = 4, refresh: bool = False):
    limit=max(1,min(int(limit or 4),20))
    try:
        voices=tts_mod._get_live_xkira_voices(force=bool(refresh))
        return voices[:4] if language.lower().startswith('vi') else voices[:limit]
    except Exception as exc:
        logger.warning("Không lấy được catalog xKiro: %s",exc)
        return []

@app.get("/api/models")
def get_model_catalog(modality: str = "all", refresh: bool = False):
    """Live xKiro model catalog used by the per-stage model picker.

    The catalog is public and changes independently of this application, so the
    UI never hardcodes a stale list. ``modality=all`` exposes the full gateway
    catalog; task-specific values filter it for the corresponding picker.
    """
    allowed = {"chat", "image", "tts", "stt", "embedding", "ocr", "moderation", "video", "music", "all"}
    mod = str(modality or "all").strip().lower()
    if mod not in allowed:
        raise HTTPException(400, f"modality không hợp lệ: {mod}")
    import requests as _requests
    try:
        resp = _requests.get(
            "https://api.xkiro.com/v1/models",
            params={"modality": mod},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"object": data.get("object", "list"), "data": data.get("data", []),
                "source": "xkiro_live", "modality": mod}
    except Exception as exc:
        logger.warning("Không lấy được live xKiro model catalog (%s): %s", mod, exc)
        return {"object": "list", "data": [], "source": "unavailable", "modality": mod,
                "error": str(exc)}


@app.get("/api/quota")
def get_quota():
    projects = storage.list_projects()
    done = sum(1 for p in projects if str(p.get("status") or "") in ("Hoàn tất", "Sẵn sàng chỉnh sửa", "Đã dịch", "Đã lồng tiếng"))
    return {"used": 0, "total": 30, "projects_processed": done, "projects": len(projects)}

@app.get("/api/providers")
def get_providers():
    return PROVIDER_OPTIONS

@app.get("/api/voices")
def get_voices(provider: Optional[str] = None):
    try:
        all_voices = tts_mod.get_available_voices()
        if provider:
            all_voices = [v for v in all_voices if v.get("provider") == provider]
        return all_voices
    except Exception as exc:
        logger.warning("Lỗi lấy voices: %s", exc)
        return []

# ============================================================
# ROUTES: TTS Preview
# ============================================================
@app.post("/api/tts/preview")
def tts_preview(body: TtsPreviewIn):
    if not body.text or not body.text.strip():
        raise HTTPException(400, "Thiếu nội dung văn bản")
    tts_id = TTS_MAP.get(body.provider, body.provider)
    keys = config.get_api_keys()
    if tts_id == "xkira_tts":
        if not config.get_xkira_api_keys():
            raise HTTPException(400, "Chưa cấu hình XKIRA_API_KEY_1...XKIRA_API_KEY_8.")
    elif tts_id == "google_tts":
        if not (keys.get("google_tts") or keys.get("google")):
            raise HTTPException(400, "Chưa cấu hình Google Cloud credentials.")
    elif tts_id != "edge_tts":
        raise HTTPException(400, f"Provider '{tts_id}' không được hỗ trợ (chỉ xKiro/Edge).")
    config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PREVIEW_DIR / f"preview_{uuid.uuid4().hex[:8]}.mp3"
    try:
        voice = _resolve_voice_for_provider(body.voice, tts_id)
        if tts_id == "xkira_tts":
            result = tts_mod.generate_tts_for_segments(
                segments=[{"start": 0.0, "end": 30.0}], translated_texts=[body.text],
                voice=voice, provider="xkira_tts", api_key=None, output_dir=config.PREVIEW_DIR,
                max_workers=1, rate=_parse_rate(body.rate), pitch=_parse_pitch(body.pitch))
            item = result[0] if result else {}
            if not item.get("audio"):
                raise RuntimeError(item.get("error") or "xKiro không tạo được audio thử")
            audio_path = Path(str(item["audio"]))
            if audio_path != out_path:
                # Keep preview cache outside project output; serve the generated file directly.
                pass
        else:
            api_key = keys.get("google_tts") or keys.get("google") if tts_id == "google_tts" else None
            audio_path, _ = tts_mod._generate_tts_with_provider(
                text=body.text, voice=voice, provider=tts_id, api_key=api_key,
                output_path=out_path, rate=_parse_rate(body.rate), pitch=_parse_pitch(body.pitch))
        if not audio_path.exists() or audio_path.stat().st_size < 500:
            raise RuntimeError("TTS tạo audio rỗng")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("TTS preview lỗi: %s", exc)
        # Provider-specific retry/cooldown stays internal to tts.py; expose one stable 400/500 message only.
        raise HTTPException(500, "Không thể tạo audio thử. Kiểm tra voice, API key và trạng thái provider.")
    return FileResponse(str(audio_path), media_type="audio/mpeg", filename=audio_path.name)

# ============================================================
# ROUTES: TTS Segment
# ============================================================
@app.post("/api/projects/{pid}/tts")
def tts_generate_segment(pid: str, body: TtsSegmentIn):
    try:
        p = _project_or_404(pid)
        segments = [dict(x) for x in (p.get("segments") or [])]
        translated_texts = p.get("translated_texts") or []
        idx = int(body.segment_index)
        if idx < 0 or idx >= len(segments):
            raise HTTPException(400, f"Segment {idx} không tồn tại")
        text = (body.text or "").strip() or (str(translated_texts[idx]).strip() if idx < len(translated_texts) else "")
        if not text:
            raise HTTPException(400, "Không có bản dịch để tạo giọng đọc")
        tts_id = TTS_MAP.get(body.provider or p.get("tts_provider") or "xkira_tts", body.provider or p.get("tts_provider") or "xkira_tts")
        voice = _resolve_voice_for_provider(body.voice or p.get("voice") or "", tts_id)
        keys=config.get_api_keys()
        tts_key=None if tts_id=="xkira_tts" else None
        if tts_id=="xkira_tts" and not config.get_xkira_api_keys(): raise HTTPException(400,"Chưa cấu hình XKIRA_API_KEY_1...XKIRA_API_KEY_8.")
        if tts_id=="google_tts" and not (keys.get("google_tts") or keys.get("google")): raise HTTPException(400,"Chưa cấu hình Google Cloud credentials.")
        out_dir=config.OUTPUT_DIR / pid / "tts"; out_dir.mkdir(parents=True,exist_ok=True)
        result=tts_mod.generate_tts_for_segments(
            segments=[segments[idx]], translated_texts=[text], voice=voice, provider=tts_id, api_key=tts_key,
            output_dir=out_dir, max_workers=1, rate=_parse_rate(body.rate), pitch=_parse_pitch(body.pitch))
        r=result[0] if result else {}
        if not r.get("audio"):
            raise HTTPException(500, f"Không tạo được audio: {r.get('error','Không rõ lỗi')}")
        audio_path=Path(str(r["audio"]))
        actual_voice=r.get("voice",voice); actual_provider=r.get("provider",tts_id)
        duration=float(r.get("duration",tts_mod.get_audio_duration(str(audio_path))) or 0)
        tts_map=dict(p.get("tts_segments") or {})
        dub_start = float(r.get("dub_start_sec", segments[idx].get("start", 0)) or 0)
        tts_map[str(idx)]={"text":text,"voice":actual_voice,"provider":actual_provider,"audio_path":str(audio_path),"audio":str(audio_path),"duration":duration,"start_sec":dub_start,"end_sec":dub_start + max(0.0, duration),"created_at":time.strftime("%Y-%m-%d %H:%M:%S")}
        segments[idx]["has_dubbing"]=True; segments[idx]["dub_status"]="done"; segments[idx]["dub_error"]=None; segments[idx]["dub_start_sec"]=dub_start; segments[idx]["dub_duration"]=duration; segments[idx]["dub_end_sec"]=dub_start + max(0.0, duration)
        storage.update_project(pid,tts_segments=tts_map,segments=segments,status="Sẵn sàng chỉnh sửa",error=None)
        return {"ok":True,"segment_index":idx,"audio_path":str(audio_path),"duration":duration,"provider":actual_provider,"voice":actual_voice,"dub_status":"done",
                "dub_start_sec":dub_start,"dub_duration":duration,"dub_end_sec":dub_start + max(0.0, duration),
                "start_sec":dub_start,"end_sec":dub_start + max(0.0, duration),
                "message":f"Đã tạo giọng đọc cho đoạn {idx+1}"}
    except HTTPException: raise
    except Exception as e:
        logger.error("❌ TTS segment lỗi: %s",traceback.format_exc())
        raise HTTPException(500,f"Lỗi tạo giọng đọc: {str(e)}")

# ============================================================
# TRANSLATION QUALITY / AUTO-REPAIR HELPERS
# ============================================================
def _source_text(seg: dict) -> str:
    seg = seg or {}
    for key in ("source_text", "text", "source", "original", "clean_text", "original_text", "transcript"):
        v = seg.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""

def _time_pair(seg: dict):
    seg = seg or {}
    try: st = float(seg.get("start", seg.get("start_sec", 0)) or 0)
    except Exception: st = 0.0
    try: en = float(seg.get("end", seg.get("end_sec", st)) or st)
    except Exception: en = st
    return st, en

def _apply_nonlinguistic_passthrough(segments, translated, target_lang):
    """Defensive fill for numeric-only/short acknowledgement subtitles."""
    tx=list(translated or [])
    while len(tx) < len(segments or []):
        tx.append('')
    for i, seg in enumerate(segments or []):
        if str(tx[i] or '').strip():
            continue
        src=_source_text(seg)
        if not src:
            continue
        compact=re.sub(r"\s+", "", src)
        if compact and re.fullmatch(r"[\d.,:+\-/%×xX()]+", compact):
            tx[i]=src
        elif 'vi' in str(target_lang or '').lower() and re.fullmatch(r"(?i)ok(?:ay)?", src.strip()):
            tx[i]='Được'
        elif 'vi' in str(target_lang or '').lower() and re.fullmatch(r"(?i)yes", src.strip()):
            tx[i]='Vâng'
        elif 'vi' in str(target_lang or '').lower() and re.fullmatch(r"(?i)no", src.strip()):
            tx[i]='Không'
    return tx


def _missing_infos(segments, translated):
    out=[]
    for i, seg in enumerate(segments or []):
        tx = str(translated[i] if i < len(translated) else "" or "").strip()
        if tx: continue
        st,en=_time_pair(seg)
        out.append({"index":i,"segment_number":i+1,"start_sec":st,"end_sec":en,"source_text":_source_text(seg)})
    return out

def _suspicious_indices(segments, translated, target_lang, max_items=120):
    cjk=re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\u3131-\ud7a3]")
    out=[]
    for i,seg in enumerate(segments or []):
        src=_source_text(seg); tx=str(translated[i] if i<len(translated) else "" or "").strip()
        if not src: continue
        suspicious=(not tx)
        suspicious = suspicious or (tx==src and (cjk.search(src) or len(src)>=8))
        suspicious = suspicious or (target_lang and "vi" in target_lang.lower() and cjk.search(tx))
        suspicious = suspicious or bool(re.match(r"^\s*\d+[.:)\-]", tx))
        if len(src)>=16:
            ratio=len(tx)/max(1,len(src)) if tx else 0
            suspicious = suspicious or ratio<0.20 or ratio>4.50
        if suspicious:
            out.append(i)
            if len(out)>=max_items: break
    return out

def _translation_provider(p: dict):
    keys=config.get_api_keys()
    provider=_resolve_translate_id(p.get("translate_provider") or config.DEFAULT_PROVIDERS.get("translate_provider","groq"))
    if provider=='groq': key=keys.get('groq')
    elif provider=='xkira': key=keys.get('xkira_keys') or keys.get('xkira')
    elif provider=='gemini': key=keys.get('gemini')
    elif provider=='google': key=None
    else: raise RuntimeError(f"Translation provider '{provider}' không được hỗ trợ.")
    if provider!='google' and not key: raise RuntimeError(f"Thiếu API key cho Translation provider '{provider}'")
    return provider,key

def _repair_translations(pid: str, indices, reason=""):
    p=_project_or_404(pid)
    segments=[dict(x) for x in (p.get('segments') or [])]
    tx=list(p.get('translated_texts') or [])
    target=LANG_TARGET_NAME.get(p.get('dst_lang','Tiếng Việt'),p.get('dst_lang','Tiếng Việt'))
    tx=_apply_nonlinguistic_passthrough(segments,tx,target)
    indices=sorted({int(i) for i in (indices or []) if 0<=int(i)<len(segments)})
    if not indices:
        return {"ok":not _missing_infos(segments,tx),"repaired":0,"missing":_missing_infos(segments,tx),"reviewed":0}
    provider,key=_translation_provider(p)
    subset=[segments[i] for i in indices]
    extra=(p.get('additional_prompt') or '').strip()
    prompt=("ĐÂY LÀ VÒNG SOÁT/SỬA PHỤ ĐỀ. Dịch lại chính xác từng câu sang "+target+
            ". Ưu tiên đúng nghĩa, đúng ngữ cảnh hội thoại/game; giữ nguyên tên game, nhân vật, vật phẩm, kỹ năng và thương hiệu; không thêm ý, không bịa; mỗi câu 1:1. "
            "Chỉ trả về đúng số câu tương ứng, đánh số từ 1.")
    if extra: prompt += " | " + extra
    corrected=translate_mod.translate_segments_fast(
        segments=subset,target_lang=target,api_key=key,provider=provider,
        max_workers=config.MAX_TRANSLATE_WORKERS,force=True,auto_retry=max(0,int(getattr(config,'TRANSLATION_REPAIR_RETRY',1))),
        context_window=2,glossary=p.get('glossary') or [],additional_prompt=prompt,
        model=(p.get('provider_config') or {}).get(provider,{}).get('model') or None)
    repaired=0
    for li,gi in enumerate(indices):
        val=str(corrected[li] if li<len(corrected) else '').strip()
        if val:
            if tx[gi]!=val: repaired+=1
            tx[gi]=val
    missing=_missing_infos(segments,tx)
    pairs=[dict(seg,translation=str(tx[i] or '').strip(),translation_id=i+1) for i,seg in enumerate(segments)]
    storage.update_project(pid,translated_texts=tx,translation_pairs=pairs,translation_complete=not missing,translation_missing_segments=missing)
    if missing:
        first=missing[0]
        logger.warning("⚠️ Soát dịch còn thiếu %d | segment #%d idx=%d | %s",len(missing),first['segment_number'],first['index'],first['source_text'][:160])
    else:
        logger.info("✅ Soát dịch hoàn tất | reviewed=%d repaired=%d",len(indices),repaired)
    return {"ok":not missing,"repaired":repaired,"missing":missing,"reviewed":len(indices)}

def _prepare_translations_before_tts(pid: str):
    """Prepare translations for TTS without making completeness a hard blocker.

    Some segments may legitimately remain untranslated (provider error, quota,
    manual deletion, etc.).  TTS should still run for every segment that has
    usable translated text; untranslated segments are skipped by the TTS worker
    and can be fixed later without blocking the rest of the project.
    """
    p=_project_or_404(pid)
    segments=[dict(x) for x in (p.get('segments') or [])]
    tx=list(p.get('translated_texts') or [])
    while len(tx)<len(segments): tx.append('')
    target=LANG_TARGET_NAME.get(p.get('dst_lang','Tiếng Việt'),p.get('dst_lang','Tiếng Việt'))
    tx=_apply_nonlinguistic_passthrough(segments,tx,target)
    missing=[i for i in range(len(segments)) if not str(tx[i] or '').strip()]
    if missing and getattr(config,'TRANSLATION_AUTO_REPAIR_MISSING',True):
        logger.warning("⚠️ /tts-all phát hiện %d đoạn thiếu → thử dịch bổ sung trước TTS",len(missing))
        try:
            storage.update_project(pid,status='Đang dịch bổ sung',error=None)
            _update_progress(pid,3,6,'Dịch bổ sung trước TTS',{'missing':missing[:50],'total_missing':len(missing)},55)
            res=_repair_translations(pid,missing,'before_tts_missing')
            p=_project_or_404(pid); tx=list(p.get('translated_texts') or [])
            while len(tx)<len(p.get('segments') or []): tx.append('')
            if not res.get('ok'):
                logger.warning("⚠️ Dịch bổ sung chưa đủ; tiếp tục TTS với %d đoạn đã có bản dịch",
                               sum(1 for x in tx if str(x or '').strip()))
        except Exception as exc:
            # Translation repair is an optional best-effort step. Never let it
            # prevent already-translated segments from being dubbed.
            logger.warning("⚠️ Dịch bổ sung trước TTS thất bại, tiếp tục phần đã dịch: %s", exc)
            p=_project_or_404(pid); tx=list(p.get('translated_texts') or [])
            while len(tx)<len(p.get('segments') or []): tx.append('')
    if getattr(config,'TRANSLATION_AUTO_REVIEW',True):
        target=LANG_TARGET_NAME.get(p.get('dst_lang','Tiếng Việt'),p.get('dst_lang','Tiếng Việt'))
        suspicious=_suspicious_indices(p.get('segments') or [],tx,target,int(getattr(config,'TRANSLATION_REVIEW_MAX_ITEMS',120)))
        if suspicious:
            logger.info("🔍 Pre-TTS review phát hiện %d đoạn nghi ngờ",len(suspicious))
            try:
                _repair_translations(pid,suspicious,'before_tts_quality')
                p=_project_or_404(pid); tx=list(p.get('translated_texts') or [])
                while len(tx)<len(p.get('segments') or []): tx.append('')
            except Exception as exc:
                logger.warning("⚠️ Pre-TTS review thất bại, giữ bản dịch hiện có: %s", exc)
    target=LANG_TARGET_NAME.get(p.get('dst_lang','Tiếng Việt'),p.get('dst_lang','Tiếng Việt'))
    tx=_apply_nonlinguistic_passthrough(p.get('segments') or [],tx,target)
    missing_infos=_missing_infos(p.get('segments') or [],tx)
    if tx != list(p.get('translated_texts') or []):
        pairs=[dict(seg,translation=str(tx[i] or '').strip(),translation_id=i+1) for i,seg in enumerate(p.get('segments') or [])]
        storage.update_project(pid,translated_texts=tx,translation_pairs=pairs,translation_complete=not missing_infos,translation_missing_segments=missing_infos)
        p=_project_or_404(pid)
    # `ok` now means "safe to proceed with partial TTS", not "100% translated".
    return p,{"ok":True,"partial":bool(missing_infos),"missing":missing_infos,"repaired":0,"reviewed":0}

# ============================================================
# ROUTES: TTS All
# ============================================================
@app.post("/api/projects/{pid}/tts-all")
def tts_generate_all(pid: str, body: TtsAllIn = TtsAllIn()):
    p = _project_or_404(pid)
    segments = p.get("segments", [])
    if not segments:
        raise HTTPException(400, "Dự án chưa có phân đoạn để tạo TTS.")
    # Missing translations are repaired automatically inside the worker instead of
    # rejecting /tts-all with HTTP 400.
    keys = config.get_api_keys()
    provider = body.provider or p.get("tts_provider") or config.DEFAULT_PROVIDERS.get("tts_provider", "xkira_tts")
    tts_id = TTS_MAP.get(provider, provider)
    voice_raw = body.voice or p.get("voice") or "female_sweet"
    voice = _resolve_voice_for_provider(voice_raw, tts_id)
    tts_key = None
    if tts_id == "xkira_tts":
        tts_key = None
        if not config.get_xkira_api_keys():
            raise HTTPException(400, "Chưa cấu hình XKIRA_API_KEY_1...XKIRA_API_KEY_8.")
    elif tts_id == "google_tts":
        tts_key = keys.get("google_tts") or keys.get("google")
        if not tts_key:
            raise HTTPException(400, "Chưa cấu hình Google Cloud credentials.")
    elif tts_id == "edge_tts":
        tts_key = None
    else:
        raise HTTPException(400, f"Provider '{tts_id}' không được hỗ trợ (xKiro/Edge/Google).")
    rate = _parse_rate(body.rate)
    pitch = _parse_pitch(body.pitch)
    total = len(segments)

    def run():
        phase_lock_acquired = False
        try:
            # HARD PHASE GATE: TTS waits for any active process/translation/review phase.
            # This fixes the race where /tts-all won the generic lock before a translation
            # worker had a chance to acquire it.
            wait_started = time.time()
            while True:
                phase = get_processing_phase(pid)
                if phase == "idle" and acquire_processing_lock(pid):
                    phase_lock_acquired = True
                    set_processing_phase(pid, "tts")
                    break
                if is_cancelled(pid):
                    logger.info("[PHASE-GATE] TTS %s hủy trong lúc chờ pipeline hiện tại", pid)
                    return
                if time.time() - wait_started > 1800:
                    raise TimeoutError("TTS chờ pipeline trước đó quá 30 phút")
                time.sleep(0.20)
            time.sleep(1.0)

            prepared, check = _prepare_translations_before_tts(pid)
            missing_before_tts = check.get("missing") or []
            if missing_before_tts:
                logger.warning("⚠️ /tts-all chạy partial: %d đoạn chưa có bản dịch sẽ được bỏ qua TTS", len(missing_before_tts))
                storage.update_project(pid, translation_missing_segments=missing_before_tts, translation_complete=False)
            prepared = _project_or_404(pid)
            segments = prepared.get("segments", [])
            translated = list(prepared.get("translated_texts", []))
            storage.update_project(pid, status="Đang tạo giọng đọc", error=None)
            _update_progress(pid, 4, 6, f"Tạo giọng đọc (0/{total})", {"done": 0, "total": total, "translation_check": "ok"}, 55)
            out_dir = config.OUTPUT_DIR / pid / "tts"
            out_dir.mkdir(parents=True, exist_ok=True)
            results = tts_mod.generate_tts_for_segments(
                segments=segments, translated_texts=translated, voice=voice,
                provider=tts_id, api_key=tts_key, output_dir=out_dir,
                max_workers=config.MAX_TTS_WORKERS, rate=rate, pitch=pitch,
            )
            tts_map, failed, skipped = {}, [], []
            updated_segments = [dict(x) for x in segments]
            for r in results:
                idx = int(r.get("index", -1))
                if r.get("audio"):
                    tts_map[str(idx)] = {
                        "text": r["text"], "audio": r["audio"], "audio_path": r["audio"],
                        "duration": r["duration"], "slot_duration": float(segments[idx].get("end",0) or 0) - float(segments[idx].get("start",0) or 0) if 0 <= idx < total else 0.0,
                        "provider": r.get("provider", tts_id), "voice": r.get("voice", voice),
                        "start_sec": float(r.get("dub_start_sec", segments[idx].get("start", 0)) or 0) if 0 <= idx < total else 0.0,
                        "end_sec": float(r.get("dub_start_sec", segments[idx].get("start", 0)) or 0) + float(r.get("duration", 0) or 0) if 0 <= idx < total else 0.0,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    if 0 <= idx < len(updated_segments):
                        dub_start = float(r.get("dub_start_sec", updated_segments[idx].get("start", 0)) or 0)
                        dub_duration = float(r.get("duration", 0) or 0)
                        updated_segments[idx]["has_dubbing"] = True
                        updated_segments[idx]["dub_status"] = "done"
                        updated_segments[idx]["dub_error"] = None
                        updated_segments[idx]["dub_start_sec"] = dub_start
                        updated_segments[idx]["dub_duration"] = dub_duration
                        updated_segments[idx]["dub_end_sec"] = dub_start + max(0.0, dub_duration)
                else:
                    if r.get("skipped") or r.get("skip_reason") == "missing_translation":
                        skipped.append(idx)
                        status = "skipped"
                    else:
                        failed.append(idx)
                        status = "failed"
                    if 0 <= idx < len(updated_segments):
                        updated_segments[idx]["has_dubbing"] = False
                        updated_segments[idx]["dub_status"] = status
                        updated_segments[idx]["dub_error"] = r.get("error")
            storage.update_project(pid, tts_segments=tts_map, segments=updated_segments,
                                   tts_missing_segments=sorted(set(failed + skipped)),
                                   tts_complete=(len(failed) == 0 and len(skipped) == 0))
            if failed or skipped:
                # Partial completion is a normal, exportable state. Missing/failed
                # segments are recorded for later retry but never block the batch.
                storage.update_project(pid, status="Sẵn sàng chỉnh sửa", error=None)
                _update_progress(pid, 4, 6, "Tạo giọng đọc hoàn tất một phần",
                                 {"done": len(tts_map), "total": total,
                                  "failed": failed[:20], "skipped": skipped[:20]}, 90)
                for r in results:
                    if not r.get("audio") and not r.get("skipped"):
                        logger.error("❌ tts-all %s segment=%s error=%s", pid, r.get("index"), r.get("error", "Không rõ lỗi"))
                logger.warning("⚠️ tts-all %s: done=%d/%d | failed=%d | skipped=%d | export vẫn được phép",
                               pid, len(tts_map), total, len(failed), len(skipped))
            else:
                storage.update_project(pid, status="Sẵn sàng chỉnh sửa", error=None)
                _update_progress(pid, 4, 6, "Tạo giọng đọc hoàn tất",
                                 {"done": len(tts_map), "total": total, "failed": [], "skipped": []}, 90)
                fallback_count = sum(1 for r in results if r.get("fallback"))
                logger.info("✅ tts-all %s: thành công %d/%d đoạn | fallback=%d", pid, len(tts_map), total, fallback_count)
        except Exception:
            logger.error(traceback.format_exc())
            _set_error(pid, "Lỗi tạo giọng đọc hàng loạt")
        finally:
            if phase_lock_acquired:
                set_processing_phase(pid, "")
                release_processing_lock(pid)

    waiting_for_translation = get_processing_phase(pid) != "idle"
    with lock_lock:
        waiting_for_translation = waiting_for_translation or bool(processing_locks.get(pid, False))
    threading.Thread(target=run, daemon=True).start()
    return {
        "ok": True,
        "status": "Đang chờ dịch hoàn tất" if waiting_for_translation else "Đang tạo giọng đọc",
        "total": total,
    }

# ============================================================
# ROUTES: Projects CRUD
# ============================================================
def _api_segment(seg: dict, translated=None, index=None) -> dict:
    """Normalize stored/STT segment fields for the Lingvo editor."""
    raw = dict(seg or {})
    if index is None: index = raw.get("index")
    source = raw.get("source_text")
    if source is None: source = raw.get("text")
    if source is None: source = raw.get("source")
    if source is None: source = raw.get("original")
    if source is None: source = raw.get("clean_text")
    if source is None: source = raw.get("original_text")
    if source is None: source = raw.get("transcript")
    tx = raw.get("translated_text")
    if tx is None: tx = raw.get("translation")
    if tx is None: tx = translated
    try: idx = int(index if index is not None else 0)
    except (TypeError, ValueError): idx = 0
    try: start = float(raw.get("start_sec", raw.get("start", 0)) or 0)
    except (TypeError, ValueError): start = 0.0
    try: end = float(raw.get("end_sec", raw.get("end", start)) or start)
    except (TypeError, ValueError): end = start
    out = dict(raw)
    timing_source = str(raw.get("timing_source") or "")
    locked = bool(raw.get("ocr_locked") or raw.get("timing_locked") or timing_source.startswith("ocr_"))
    if locked:
        # OCR MASTER is the immutable source timeline. Expose exactly the same values
        # through every API path so the editor cannot accidentally fall back to STT timing.
        source_start = raw.get("source_start", start)
        source_end = raw.get("source_end", end)
        try:
            start = float(source_start)
            end = float(source_end)
        except (TypeError, ValueError):
            pass
        out["source_start"] = start
        out["source_end"] = end
        out["ocr_locked"] = True
        out["timing_locked"] = True
    out.update({"index": idx, "start_sec": start, "end_sec": end,
                "source_text": str(source or ""), "translated_text": str(tx or "")})
    out["has_dubbing"] = bool(raw.get("has_dubbing") or raw.get("dub_status") == "done" or raw.get("dubbed"))
    return out

def _project_derived_ready(project: dict) -> bool:
    segs = project.get("segments") or []
    if not segs:
        return False
    tx = project.get("translated_texts") or []
    complete = bool(project.get("translation_complete")) and len(tx) >= len(segs) and all(str(tx[i] or "").strip() for i in range(len(segs)))
    # Even when translation is not complete, STT has already produced an editable
    # transcript. Do not treat a stale queued/processing label as a reason to rerun STT.
    return bool(complete or len(segs) > 0)

def _normalize_project_state(project: dict) -> dict:
    p = dict(project or {})
    busy_statuses = {"Đang chờ xử lý", "Đang xử lý", "Đang tách âm thanh", "Đang nhận diện giọng nói", "Đang dịch", "Đang soát bản dịch"}
    status = str(p.get("status") or "")
    pid = str(p.get("id") or "")
    actively_running = bool(pid and processing_locks.get(pid, False))
    if status in busy_statuses and not actively_running and _project_derived_ready(p):
        p["status"] = "Sẵn sàng chỉnh sửa"
        p["stage"] = None
        if p.get("translation_complete"):
            prog = dict(p.get("progress") or {})
            prog.update({"step": 3, "total_steps": 6, "step_name": "Dịch thuật", "percentage": 100, "details": {"segments": len(p.get("segments") or [])}})
            p["progress"] = prog
        storage.update_project(pid, status="Sẵn sàng chỉnh sửa", stage=None)
    return p

@app.get("/api/projects")
def list_projects():
    return [_normalize_project_state(p) for p in storage.list_projects()]

@app.get("/api/projects/{pid}")
def get_project(pid: str):
    p = _normalize_project_state(_project_or_404(pid))
    raw_segments = p.get("segments", []) or []
    translated_texts = p.get("translated_texts", []) or []
    segments = [_api_segment(seg, translated_texts[i] if i < len(translated_texts) else None, i)
                for i, seg in enumerate(raw_segments)]
    pairs = p.get("translation_pairs", []) or []
    if not pairs and segments:
        pairs = [dict(seg, translation=seg.get("translated_text", ""), translation_id=i + 1)
                 for i, seg in enumerate(segments)]
    return {**p, "segments": segments, "translation_pairs": pairs,
            "translation_missing_segments": p.get("translation_missing_segments", []) or []}

@app.patch("/api/projects/{pid}")
def patch_project(pid: str, body: Dict[str, Any]):
    _project_or_404(pid)
    allowed = {"name", "src_lang", "dst_lang", "voice", "want_subtitle",
               "keep_original_audio", "stt_provider", "translate_provider", "tts_provider",
               "additional_prompt", "glossary", "subtitle_style", "render", "filter_transcript_noise",
               "edge_voice", "xkira_voice", "tts_speed", "translate_refine", "ocr_region", "subtitle_source", "ocr_color_profile", "ocr_mode", "ocr_provider", "ocr_model", "ocr_lang"}
    updates = {k: v for k, v in (body or {}).items() if k in allowed}
    if updates:
        storage.update_project(pid, **updates)
    return {"ok": True, "updated": list(updates.keys())}

@app.get("/api/projects/{pid}/progress")
def get_progress(pid: str):
    p = _project_or_404(pid)
    progress = p.get("progress", {})
    missing = p.get("translation_missing_segments", []) or []
    return {"status": p.get("status", "Unknown"), "step": progress.get("step", 0),
            "total_steps": progress.get("total_steps", 6),
            "step_name": progress.get("step_name", ""),
            "percentage": progress.get("percentage", 0),
            "details": progress.get("details", {}), "error": p.get("error"),
            "cancelled": cancel_flags.get(pid, False),
            "segments_count": len(p.get("segments", [])),
            "translated_count": sum(1 for x in (p.get("translated_texts", []) or []) if str(x or '').strip()),
            "missing_translations": missing}

@app.get("/api/projects/{pid}/status")
def get_status(pid: str):
    return get_progress(pid)

@app.get("/api/projects/{pid}/transcript")
def get_transcript(pid: str):
    p = _project_or_404(pid)
    raw_segments = p.get("segments", []) or []
    translated = p.get("translated_texts", []) or []
    segments = [_api_segment(seg, translated[i] if i < len(translated) else None, i)
                for i, seg in enumerate(raw_segments)]
    pairs = p.get("translation_pairs") or []
    if not pairs and segments:
        pairs = [dict(seg, translation=seg.get("translated_text", ""), translation_id=i + 1)
                 for i, seg in enumerate(segments)]
    return {"segments": segments, "translated_texts": translated, "items": pairs,
            "translation_complete": bool(p.get("translation_complete")),
            "translation_missing_segments": p.get("translation_missing_segments", []) or [],
            "ocr_texts": p.get("ocr_texts", [])}

@app.patch("/api/projects/{pid}/segments/{segment_index}")
def patch_segment(pid: str, segment_index: int, body: dict):
    p = _project_or_404(pid)
    segments = [dict(x) for x in (p.get("segments") or [])]
    if segment_index < 0 or segment_index >= len(segments):
        raise HTTPException(404, "Segment không tồn tại")
    seg = segments[segment_index]
    requested_timing = any(k in (body or {}) for k in ("start_sec", "end_sec", "start", "end", "source_start", "source_end"))
    if requested_timing and str(p.get("subtitle_source") or "stt").lower() == "ocr":
        raise HTTPException(409, "OCR MASTER TIMELINE đã khóa theo frame video; không được sửa start/end.")
    timing_changed = False
    for key in ("start_sec", "end_sec", "start", "end", "speed", "dub_start_sec", "has_dubbing"):
        if key not in (body or {}):
            continue
        val = body[key]
        if key in {"start_sec", "start", "end_sec", "end", "speed", "dub_start_sec"}:
            try:
                val = float(val)
            except Exception:
                raise HTTPException(400, f"{key} phải là số")
        if key == "start_sec":
            seg["start"] = val
            seg["source_start"] = val
            timing_changed = True
        elif key == "end_sec":
            seg["end"] = val
            seg["source_end"] = val
            timing_changed = True
        elif key == "start":
            seg["start"] = val
            seg["source_start"] = val
            timing_changed = True
        elif key == "end":
            seg["end"] = val
            seg["source_end"] = val
            timing_changed = True
        else:
            seg[key] = val
    if seg.get("end", 0) <= seg.get("start", 0):
        raise HTTPException(400, "Thời gian kết thúc phải lớn hơn bắt đầu")
    storage.update_project(pid, segments=segments)
    return {"ok": True, "segment_index": segment_index,
            "segment": _api_segment(seg, (p.get("translated_texts") or [""] * len(segments))[segment_index], segment_index)}

@app.put("/api/projects/{pid}/transcript")
def update_transcript(pid: str, body: TranscriptUpdateIn):
    p = _project_or_404(pid)
    segments = p.get("segments", [])
    if not segments:
        raise HTTPException(400, "Dự án chưa có segments")
    if len(body.texts) != len(segments):
        raise HTTPException(400, f"Số câu không khớp: gửi {len(body.texts)}, cần {len(segments)}")
    translated_texts = [str(x).strip() for x in body.texts]
    old_texts = list(p.get("translated_texts") or [])
    # Editing translation invalidates only the dubbing for changed segments.
    tts_segments = dict(p.get("tts_segments") or {})
    changed = []
    for i, new_text in enumerate(translated_texts):
        old_text = str(old_texts[i]).strip() if i < len(old_texts) else ""
        if new_text != old_text:
            changed.append(i)
            tts_segments.pop(str(i), None)
            segments[i]["has_dubbing"] = False
            segments[i]["dub_status"] = "none"
            tts_file = config.OUTPUT_DIR / pid / "tts" / f"seg{i:04d}.mp3"
            try:
                if tts_file.exists(): tts_file.unlink()
            except Exception:
                pass
    pairs = [dict(seg, translation=translated_texts[i], translation_id=i + 1)
             for i, seg in enumerate(segments)]
    storage.update_project(pid, translated_texts=translated_texts,
                           translation_pairs=pairs, segments=segments,
                           tts_segments=tts_segments,
                           translation_complete=all(t.strip() for t in translated_texts))
    return {"ok": True, "changed_indices": changed}

@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    cancel_flags.pop(pid, None)
    release_processing_lock(pid)
    storage.delete_project(pid)
    return {"ok": True}

@app.get("/api/projects/{pid}/logs")
def get_project_logs(pid: str, lines: int = 50):
    try:
        log_file = config.LOGS_DIR / "app.log"
        if not log_file.exists():
            return {"logs": [], "message": "Log file chưa tồn tại"}
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        keywords = [pid, pid[:8], "pipeline", "ERROR", "WARNING", "STT", "translate", "TTS", "render"]
        filtered = [l.strip() for l in all_lines[-2000:]
                    if any(k.lower() in l.lower() for k in keywords if k)]
        if not filtered:
            filtered = [l.strip() for l in all_lines[-lines:]]
        result = filtered[-lines:] if len(filtered) > lines else filtered
        return {"logs": result, "total": len(result), "file": str(log_file)}
    except Exception as e:
        return {"logs": [], "error": str(e)}

@app.delete("/api/projects/{pid}/tts/{segment_index}")
def delete_tts_segment(pid: str, segment_index: int):
    p = _project_or_404(pid)
    tts_file = config.OUTPUT_DIR / pid / "tts" / f"seg{segment_index:04d}.mp3"
    if tts_file.exists():
        try:
            tts_file.unlink()
        except Exception:
            pass
    if str(segment_index) in p.get("tts_segments", {}):
        del p["tts_segments"][str(segment_index)]
    segments=[dict(x) for x in (p.get("segments") or [])]
    if 0 <= segment_index < len(segments):
        segments[segment_index]["has_dubbing"]=False
        segments[segment_index]["dub_status"]="none"
        segments[segment_index].pop("dub_error",None)
    storage.update_project(pid, tts_segments=p.get("tts_segments", {}), segments=segments)
    return {"ok": True, "segment_index": segment_index}

@app.post("/api/projects/{pid}/segments")
def add_segment(pid: str, body: dict):
    p=_project_or_404(pid)
    if str(p.get("subtitle_source") or "stt").lower() == "ocr":
        raise HTTPException(409, "OCR MASTER TIMELINE là 1:1 với video; không thể thêm segment thủ công.")
    segments=[dict(x) for x in (p.get('segments') or [])]
    start=float(body.get('start_sec',body.get('start',0)) or 0); end=float(body.get('end_sec',body.get('end',start+1)) or start+1)
    if end<=start: raise HTTPException(400,'end phải lớn hơn start')
    seg={'start':start,'end':end,'text':str(body.get('source_text') or body.get('text') or ''),'segment_id':len(segments),'timing_source':'manual'}
    if body.get('parent_index') is not None: seg['parent_index']=int(body['parent_index'])
    segments.append(seg); segments.sort(key=lambda x:float(x.get('start',0) or 0))
    for i,x in enumerate(segments): x['segment_id']=i
    tx=list(p.get('translated_texts') or []); tx=[*tx]+['']*max(0,len(segments)-len(tx))
    storage.update_project(pid,segments=segments,translated_texts=tx,translation_pairs=[dict(x,translation=tx[i],translation_id=i+1) for i,x in enumerate(segments)],translation_complete=all(str(x).strip() for x in tx))
    return _api_segment(segments[-1],tx[-1],len(segments)-1)

@app.delete("/api/projects/{pid}/segments/{segment_index}")
def delete_segment(pid: str, segment_index: int):
    p=_project_or_404(pid)
    if str(p.get("subtitle_source") or "stt").lower() == "ocr":
        raise HTTPException(409, "Không thể xóa cue OCR gốc; hãy chỉnh/xóa bản dịch hoặc chạy OCR lại.")
    segments=[dict(x) for x in (p.get('segments') or [])]
    if segment_index<0 or segment_index>=len(segments): raise HTTPException(404,'Segment không tồn tại')
    segments.pop(segment_index); tx=list(p.get('translated_texts') or []);
    if segment_index<len(tx): tx.pop(segment_index)
    while len(tx)<len(segments): tx.append('')
    storage.update_project(pid,segments=segments,translated_texts=tx,translation_pairs=[dict(x,translation=tx[i],translation_id=i+1) for i,x in enumerate(segments)],translation_complete=bool(segments) and all(str(x).strip() for x in tx))
    return {'ok':True}

@app.delete("/api/projects/{pid}/translation/{segment_index}")
def delete_translation_segment(pid: str, segment_index: int):
    p=_project_or_404(pid); tx=list(p.get('translated_texts') or [])
    segments=[dict(x) for x in (p.get('segments') or [])]
    if segment_index<0 or segment_index>=len(segments): raise HTTPException(404,'Translation segment không tồn tại')
    while len(tx)<len(segments): tx.append('')
    tx[segment_index]=''
    tts_map=dict(p.get('tts_segments') or {})
    tts_map.pop(str(segment_index),None)
    tts_file=config.OUTPUT_DIR / pid / 'tts' / f'seg{segment_index:04d}.mp3'
    try: tts_file.unlink(missing_ok=True)
    except Exception: pass
    segments[segment_index]['has_dubbing']=False
    segments[segment_index]['dub_status']='none'
    segments[segment_index].pop('dub_error',None)
    storage.update_project(pid,segments=segments,translated_texts=tx,
                           translation_pairs=[dict(x,translation=tx[i],translation_id=i+1) for i,x in enumerate(segments)],
                           translation_complete=False,tts_segments=tts_map)
    return {'ok':True,'segment_index':segment_index}

@app.delete("/api/projects/{pid}/translations")
def clear_all_translations(pid: str):
    p = _project_or_404(pid)
    segments = [dict(x) for x in (p.get("segments") or [])]
    for seg in segments:
        seg["has_dubbing"] = False
        seg["dub_status"] = "none"
        seg.pop("dub_error", None)
    tts_dir=config.OUTPUT_DIR / pid / "tts"
    if tts_dir.exists():
        try: shutil.rmtree(tts_dir)
        except Exception: pass
    storage.update_project(
        pid, segments=segments, translated_texts=[""] * len(segments),
        translation_pairs=[dict(seg, translation="", translation_id=i + 1) for i, seg in enumerate(segments)],
        translation_complete=False, tts_segments={}, output_audio=None)
    return {"ok": True, "message": f"Đã xóa {len(segments)} bản dịch"}

@app.delete("/api/projects/{pid}/dubbing")
def clear_all_dubbing(pid: str):
    p = _project_or_404(pid)
    tts_dir = config.OUTPUT_DIR / pid / "tts"
    if tts_dir.exists():
        try:
            shutil.rmtree(tts_dir)
        except Exception:
            pass
    output_dir = config.OUTPUT_DIR / pid
    if output_dir.exists():
        try:
            for f in output_dir.glob("dubbed*.mp4"):
                f.unlink()
            for f in output_dir.glob("*.mp4"):
                if f.name != "source.mp4":
                    f.unlink()
        except Exception:
            pass
    segments=[dict(x) for x in (p.get("segments") or [])]
    for seg in segments:
        seg["has_dubbing"]=False; seg["dub_status"]="none"; seg.pop("dub_error",None)
    storage.update_project(pid, segments=segments, output_video=None, output_audio=None, output_srt=p.get("output_srt"), tts_segments={})
    return {"ok": True, "message": "Đã xóa tất cả lồng tiếng và audio TTS"}

# ============================================================
# ROUTES: Translation Review / Repair
# ============================================================
class TranslationReviewIn(BaseModel):
    full: bool = False

@app.post("/api/projects/{pid}/review-translations")
def review_translations(pid: str, body: TranslationReviewIn = TranslationReviewIn()):
    p=_project_or_404(pid)
    if not p.get('segments'): raise HTTPException(400,'Dự án chưa có phân đoạn')
    if not acquire_processing_lock(pid): raise HTTPException(409,'Dự án đang được xử lý')
    set_processing_phase(pid, 'translation_review')
    storage.update_project(pid,status='Đang soát bản dịch',error=None)
    _update_progress(pid,3,6,'Soát bản dịch',{'full':bool(body.full)},5)
    def run():
        try:
            segs=p.get('segments') or []
            tx=list(p.get('translated_texts') or [])
            while len(tx)<len(segs): tx.append('')
            target=LANG_TARGET_NAME.get(p.get('dst_lang','Tiếng Việt'),p.get('dst_lang','Tiếng Việt'))
            idx=list(range(len(segs))) if body.full else _suspicious_indices(segs,tx,target,int(getattr(config,'TRANSLATION_REVIEW_MAX_ITEMS',120)))
            res=_repair_translations(pid,idx,'manual_full' if body.full else 'manual_suspicious')
            if res.get('ok'):
                storage.update_project(pid,status='Sẵn sàng chỉnh sửa',error=None)
                _update_progress(pid,3,6,'Soát bản dịch hoàn tất',{'reviewed':res.get('reviewed',0),'repaired':res.get('repaired',0),'missing':0},100)
            else:
                missing=res.get('missing') or []
                first=missing[0] if missing else {}
                msg=f"Còn {len(missing)} đoạn chưa dịch được sau khi soát. Đầu tiên: segment #{first.get('segment_number',0)} | {first.get('source_text','')[:140]}"
                storage.update_project(pid,status='Sẵn sàng chỉnh sửa',error=msg,translation_missing_segments=missing,translation_complete=False)
                _update_progress(pid,3,6,'Soát bản dịch chưa đủ',{'reviewed':res.get('reviewed',0),'repaired':res.get('repaired',0),'missing':missing[:50]},85)
        except Exception:
            logger.error('❌ Review translation lỗi: %s',traceback.format_exc())
            _set_error(pid,'Lỗi soát bản dịch')
        finally:
            set_processing_phase(pid, "")
            release_processing_lock(pid)
    threading.Thread(target=run,daemon=True).start()
    return {'ok':True,'status':'Đang soát bản dịch','full':bool(body.full),'total':len(p.get('segments') or [])}

# ============================================================
# ROUTES: Translate Segment
# ============================================================

@app.post("/api/projects/{pid}/translate")
def translate_project(pid: str, body: dict = None):
    p = _project_or_404(pid)
    segments = p.get("segments") or []
    if not segments:
        raise HTTPException(400, "Dự án chưa có phân đoạn")
    if not acquire_processing_lock(pid):
        raise HTTPException(409, "Dự án đang được xử lý")
    set_processing_phase(pid, "translation")
    storage.update_project(pid, status="Đang dịch", error=None)
    def run():
        try:
            keys = config.get_api_keys()
            target_lang = LANG_TARGET_NAME.get(p.get("dst_lang", "Tiếng Việt"), p.get("dst_lang", "Tiếng Việt"))
            trans_id = _resolve_translate_id(p.get("translate_provider") or config.DEFAULT_PROVIDERS.get("translate_provider", "groq"))
            trans_key = keys.get(trans_id) if trans_id in {"groq", "gemini", "xkira"} else None
            if trans_id != "google" and not trans_key:
                raise RuntimeError(f"Thiếu API key cho Translation provider '{trans_id}'")
            translated = translate_mod.translate_segments_fast(
                segments=segments, target_lang=target_lang, api_key=trans_key,
                provider=trans_id, max_workers=config.MAX_TRANSLATE_WORKERS, force=True,
                glossary=p.get("glossary") or [], additional_prompt=p.get("additional_prompt") or "")
            if len(translated) != len(segments):
                raise RuntimeError(f"Dịch không hoàn tất: {len(translated)}/{len(segments)}")
            if bool(p.get("translate_refine")) and all(str(x).strip() for x in translated):
                refine_prompt=(p.get("additional_prompt") or "") + (" | " if (p.get("additional_prompt") or "") else "") + "Soát lại lần hai: giữ nguyên tên game, nhân vật, thương hiệu, vật phẩm và thuật ngữ; không bịa; mỗi câu ngắn gọn nhưng đủ nghĩa để đọc thoại; giữ 1:1 số câu."
                translated=translate_mod.translate_segments_fast(
                    segments=segments, target_lang=target_lang, api_key=trans_key, provider=trans_id,
                    max_workers=config.MAX_TRANSLATE_WORKERS, force=True,
                    glossary=p.get("glossary") or [], additional_prompt=refine_prompt)
                if len(translated)!=len(segments) or not all(str(x).strip() for x in translated):
                    raise RuntimeError("Vòng soát lại bản dịch không hoàn tất")
            pairs = [dict(seg, translation=str(translated[i] or "").strip(), translation_id=i+1)
                     for i, seg in enumerate(segments)]
            updated = [dict(x) for x in segments]
            for x in updated:
                x["has_dubbing"] = False
                x["dub_status"] = "none"
                x.pop("dub_error", None)
            storage.update_project(
                pid, segments=updated, translated_texts=translated, translation_pairs=pairs,
                translation_complete=all(str(t).strip() for t in translated),
                tts_segments={}, status="Sẵn sàng chỉnh sửa", error=None)
            _update_progress(pid, 3, 6, "Dịch thuật", {"segments": len(segments)}, 100)
        except Exception as e:
            logger.error("Translate job lỗi: %s", traceback.format_exc())
            _set_error(pid, str(e) or repr(e))
        finally:
            set_processing_phase(pid, "")
            release_processing_lock(pid)
    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "status": "Đang dịch", "total": len(segments)}

@app.post("/api/projects/{pid}/translate-segment")
def translate_segment(pid: str, body: dict):
    p = _project_or_404(pid)
    segments = p.get("segments", [])
    translated_texts = list(p.get("translated_texts", []))
    segment_idx = body.get("segment_index")
    if segment_idx is None:
        segment_idx = body.get("index")
    if segment_idx is None:
        raise HTTPException(400, "Thiếu segment_index")
    try:
        segment_idx = int(segment_idx)
    except Exception:
        raise HTTPException(400, "segment_index phải là số nguyên")
    if segment_idx < 0 or segment_idx >= len(segments):
        raise HTTPException(400, f"Segment {segment_idx} không tồn tại")
    text = body.get("text", "") or segments[segment_idx].get("text", "")
    if not text or not str(text).strip():
        raise HTTPException(400, "Không có nội dung để dịch")
    while len(translated_texts) < len(segments):
        translated_texts.append("")
    target_lang = LANG_TARGET_NAME.get(body.get("target_lang", "Tiếng Việt"), "Tiếng Việt")
    keys = config.get_api_keys()
    translate_provider = p.get("translate_provider") or config.DEFAULT_PROVIDERS.get("translate_provider", "groq")
    trans_id = _resolve_translate_id(translate_provider)
    trans_key = None
    if trans_id == "groq":
        trans_key = keys.get("groq")
    elif trans_id == "xkira":
        trans_key = keys.get("xkira_keys") or keys.get("xkira")
    elif trans_id == "gemini":
        trans_key = keys.get("gemini")
    elif trans_id == "google":
        trans_key = None
    else:
        raise HTTPException(400, f"Provider '{trans_id}' không được hỗ trợ (chỉ Groq/xKiro/Gemini/Google).")
    if trans_id != "google" and not trans_key:
        raise HTTPException(400, f"Thiếu API key cho {translate_provider}")
    try:
        left=max(0,segment_idx-2); right=min(len(segments),segment_idx+3)
        local_segments=segments[left:right]
        translated_all=translate_mod.translate_segments_fast(
            local_segments,target_lang=target_lang,api_key=trans_key,provider=trans_id,force=True,context_window=0,
            glossary=p.get("glossary") or [],additional_prompt=(p.get("additional_prompt") or "")+" | Chỉ trả đúng từng câu tương ứng; giữ nguyên tên riêng và thuật ngữ.",
            model=(p.get("provider_config") or {}).get(trans_id, {}).get("model") or None)
        translation=(translated_all[segment_idx-left] if segment_idx-left < len(translated_all) else "").strip()
        if not translation or not translation.strip():
            raise HTTPException(500, "Dịch thất bại: không nhận được kết quả")
        translated_texts[segment_idx] = translation
        pairs = [dict(seg, translation=translated_texts[i], translation_id=i + 1)
                 for i, seg in enumerate(segments)]
        storage.update_project(pid, translated_texts=translated_texts, translation_pairs=pairs)
        seg=segments[segment_idx]
        return {"ok": True, "translation": translation, "translated_text": translation,
                "segment_index": segment_idx, "index": segment_idx,
                "original": str(seg.get("text",seg.get("source_text",""))),
                "start": float(seg.get("start",0) or 0), "end": float(seg.get("end",0) or 0)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translate segment lỗi: {e}")
        raise HTTPException(500, f"Dịch thất bại: {str(e)}")

# ============================================================
# ROUTES: URL import / preview
# ============================================================
_URL_IMPORTS = {}
_URL_IMPORT_LOCK = threading.Lock()

@app.post("/api/videos/import-url")
def import_video_url(body: dict):
    url=str((body or {}).get('url') or '').strip()
    if not url: raise HTTPException(400,'Thiếu URL video')
    try:
        import yt_dlp
        opts={'quiet':True,'no_warnings':True,'noplaylist':True,'skip_download':True,'socket_timeout':30}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info=ydl.extract_info(url,download=False)
        if not info: raise RuntimeError('Không lấy được thông tin video')
        entry=info['entries'][0] if info.get('entries') else info
        iid=uuid.uuid4().hex[:12]
        data={'id':iid,'url':url,'title':str(entry.get('title') or 'Video'),
              'duration_minutes':round(float(entry.get('duration') or 0)/60,2),
              'thumbnail':str(entry.get('thumbnail') or ''),'uploader':str(entry.get('uploader') or ''),
              'platform':str(entry.get('extractor_key') or entry.get('extractor') or ''),
              'webpage_url':str(entry.get('webpage_url') or url),
              'preview_url':str(entry.get('url') or ''), 'direct_url':str(entry.get('url') or '')}
        with _URL_IMPORT_LOCK: _URL_IMPORTS[iid]=data
        return data
    except Exception as e:
        raise HTTPException(400,f'Không thể phân tích URL: {e}')

@app.get("/api/videos/import-url/{import_id}")
def get_imported_url(import_id: str):
    with _URL_IMPORT_LOCK: data=_URL_IMPORTS.get(import_id)
    if not data: raise HTTPException(404,'Import URL không tồn tại')
    return data

# ============================================================
# ROUTES: Create Project
# ============================================================
@app.post("/api/projects")
def create_project(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    src_lang: str = Form("Tự động nhận diện"),
    dst_lang: str = Form("Tiếng Việt"),
    voice: str = Form("female_sweet"),
    want_subtitle: bool = Form(True),
    keep_original_audio: bool = Form(False),
    stt_provider: str = Form(None),
    translate_provider: str = Form(None),
    tts_provider: str = Form(None),
    stt_model: str = Form(None),
    ocr_model: str = Form(None),
    translate_model: str = Form(None),
    tts_model: str = Form(None),
    full_model: str = Form(None),
    ocr_provider: str = Form(None),
    subtitle_source: str = Form("stt"),
    filter_transcript_noise: bool = Form(False),
    ocr_region: str = Form("{}"),
    translate_refine: bool = Form(False),
    url: Optional[str] = Form(None),
    import_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if not file and not url:
        raise HTTPException(400, "Cần tải file video hoặc nhập URL")
    pid = uuid.uuid4().hex[:12]
    proj_dir = config.UPLOAD_DIR / pid
    proj_dir.mkdir(parents=True, exist_ok=True)
    video_path = proj_dir / "source.mp4"
    if file:
        try:
            with open(video_path, "wb") as out:
                shutil.copyfileobj(file.file, out)
        except Exception as e:
            shutil.rmtree(proj_dir, ignore_errors=True)
            raise HTTPException(500, f"Không thể lưu file: {e}")
    else:
        try:
            reused=False
            if import_id:
                with _URL_IMPORT_LOCK: meta=_URL_IMPORTS.get(import_id)
                if meta and meta.get('url')==url:
                    # Reuse metadata-only import; downloader still runs once here.
                    pass
            audio_mod.download_from_url(url, video_path)
        except Exception as e:
            shutil.rmtree(proj_dir, ignore_errors=True)
            raise HTTPException(400, f"Không tải được video từ URL: {e}")
    project = {
        "id": pid, "name": name, "src_lang": src_lang, "dst_lang": dst_lang,
        "voice": voice, "want_subtitle": want_subtitle,
        "want_dubbing": True, "keep_original_audio": keep_original_audio,
        "subtitle_source": "ocr" if str(subtitle_source).lower()=="ocr" else "stt",
        "filter_transcript_noise": bool(filter_transcript_noise),
        "ocr_region": (lambda r: {"x":float(r.get("x",0)),"y":float(r.get("y",72)),"w":float(r.get("w",r.get("width",100))),"h":float(r.get("h",r.get("height",25)))})
                       (json.loads(ocr_region) if str(ocr_region).strip().startswith("{") else {"x":0,"y":72,"w":100,"h":25}),
        "translate_refine": bool(translate_refine),
        "additional_prompt": "", "glossary": [],
        "render": {"include_subtitle": bool(want_subtitle), "include_dubbing": True,
                   "quality": "1080p", "orig_volume": 0.2, "dub_volume": 1.0,
                   "reduce_original_voice": False, "original_voice_reduction": 0.0},
        "stt_provider": stt_provider or config.DEFAULT_PROVIDERS.get("stt_provider", "groq"),
        "translate_provider": translate_provider or config.DEFAULT_PROVIDERS.get("translate_provider", "groq"),
        "tts_provider": tts_provider or config.DEFAULT_PROVIDERS.get("tts_provider", "xkira_tts"),
        "stt_model": stt_model or getattr(config, "DEFAULT_MODELS", {}).get("stt", ""),
        "ocr_model": ocr_model or getattr(config, "DEFAULT_MODELS", {}).get("ocr", ""),
        "ocr_lang": "auto",
        "translate_model": translate_model or getattr(config, "DEFAULT_MODELS", {}).get("translate", ""),
        "tts_model": tts_model or getattr(config, "DEFAULT_MODELS", {}).get("tts", "xkiro-voice"),
        "full_model": full_model or getattr(config, "DEFAULT_MODELS", {}).get("full", ""),
        "ocr_provider": (ocr_provider or "local") if str(subtitle_source or "stt").lower()=="ocr" else (ocr_provider or "local"),
        "ocr_mode": getattr(config, "OCR_MODE", "HIGH_RECALL"),
        "ocr_color_profile": {"enabled": False, "rgb": [255,255,255], "hsv": [0,0,255], "radius": 10, "tolerance": 42, "x": 0.5, "y": 0.9},
        "edge_voice": getattr(config,"DEFAULT_EDGE_VOICE","vi-VN-NamMinhNeural"),
        "xkira_voice": voice if (tts_provider or "xkira_tts")=="xkira_tts" else "",
        "status": "Đang chờ xử lý", "created": time.strftime("%d/%m/%Y %H:%M:%S"),
        "created_ts": time.time(), "video_path": str(video_path),
        "segments": [], "translated_texts": [], "translation_pairs": [],
        "translation_complete": False, "error": None,
        "progress": {"step": 0, "total_steps": 6, "step_name": "Chờ xử lý",
                     "percentage": 0, "details": {}},
    }
    storage.upsert_project(project)
    # Mark the project as active synchronously before scheduling the background worker,
    # so an immediate /tts-all request cannot win the race during process startup.
    set_processing_phase(pid, "process")
    background_tasks.add_task(_process_job_background, pid)
    return {"id": pid, "project_id": pid, "status": "Đang chờ xử lý"}

def _process_job_background(pid: str):
    if not acquire_processing_lock(pid):
        logger.warning(f"⚠️ Project {pid} đang được xử lý, bỏ qua")
        return
    set_processing_phase(pid, "process")
    try:
        _process_job(pid)
    except Exception as e:
        logger.error(f"❌ Process job {pid} lỗi: {e}")
        _set_error(pid, str(e))
    finally:
        set_processing_phase(pid, "")
        release_processing_lock(pid)

def _reset_project_for_reprocess(pid: str):
    """Clear derived STT/translation/TTS/render data while keeping the source video/config."""
    p = _project_or_404(pid)
    storage.update_project(
        pid, segments=[], translated_texts=[], translation_pairs=[],
        translation_complete=False, translation_missing_segments=[], tts_segments={},
        transcript_srt=None, output_srt=None, output_audio=None, output_video=None,
        latest_output_id=None, error=None, status="Đang xử lý",
    )
    # Remove only generated files. The source video lives outside OUTPUT_DIR.
    out_dir = config.OUTPUT_DIR / pid
    if out_dir.exists():
        try:
            shutil.rmtree(out_dir)
        except Exception as exc:
            logger.warning("Không xóa được output cũ %s: %s", out_dir, exc)
    return True

@app.post("/api/projects/{pid}/process")
def process_project(pid: str, force: bool = Query(False)):
    project = _project_or_404(pid)

    # IMPORTANT: /process is idempotent by default. Opening an already processed
    # project must NEVER start STT/translation again. A real re-run is explicit via
    # /reprocess-stt (or /process?force=true for backward compatibility).
    if not force and project.get("segments"):
        normalized = _normalize_project_state(project)
        return {
            "ok": True,
            "status": normalized.get("status", "Sẵn sàng chỉnh sửa"),
            "project_id": pid,
            "reused": True,
            "segments": len(project.get("segments") or []),
            "translated": sum(1 for x in (project.get("translated_texts") or []) if str(x or "").strip()),
        }

    if not acquire_processing_lock(pid):
        return {"ok": True, "status": "Đang xử lý", "already_running": True}
    set_processing_phase(pid, "process")
    cancel_flags.pop(pid, None)
    if force:
        _reset_project_for_reprocess(pid)
    storage.update_project(pid, status="Đang xử lý", stage=None, error=None)

    def run_process():
        try:
            _process_job(pid)
        except Exception as e:
            logger.error(f"Process job lỗi: {e}")
            _set_error(pid, str(e))
        finally:
            set_processing_phase(pid, "")
            release_processing_lock(pid)

    thread = threading.Thread(target=run_process, daemon=True)
    thread.start()
    return {"ok": True, "status": "Đang xử lý", "project_id": pid, "reused": False}

def _normalize_ocr_region_value(value):
    r = dict(value or {}) if isinstance(value, dict) else {}
    try:
        x = float(r.get("x", 0)); y = float(r.get("y", 72))
        w = float(r.get("w", r.get("width", 100))); h = float(r.get("h", r.get("height", 25)))
    except (TypeError, ValueError):
        x, y, w, h = 0.0, 72.0, 100.0, 25.0
    x = max(0.0, min(100.0, x)); y = max(0.0, min(100.0, y))
    w = max(1.0, min(100.0 - x, w)); h = max(1.0, min(100.0 - y, h))
    return {"x": round(x, 3), "y": round(y, 3), "w": round(w, 3), "h": round(h, 3)}


def _start_reprocess(pid: str, source: str = "stt", region: Optional[Dict[str, Any]] = None):
    _project_or_404(pid)
    normalized_source = "ocr" if str(source or "stt").lower() == "ocr" else "stt"
    normalized_region = _normalize_ocr_region_value(region)
    storage.update_project(pid, subtitle_source=normalized_source, ocr_region=normalized_region)
    if not acquire_processing_lock(pid):
        return {"ok": True, "status": "Đang xử lý", "already_running": True, "project_id": pid}
    set_processing_phase(pid, "process")
    cancel_flags.pop(pid, None)
    _reset_project_for_reprocess(pid)
    storage.update_project(pid, subtitle_source=normalized_source, ocr_region=normalized_region, status="Đang xử lý", stage="QUEUING", error=None)

    def run_reprocess():
        try:
            _process_job(pid)
        except Exception as e:
            logger.error("Reprocess %s lỗi: %s", normalized_source.upper(), e, exc_info=True)
            _set_error(pid, str(e))
        finally:
            set_processing_phase(pid, "")
            release_processing_lock(pid)

    threading.Thread(target=run_reprocess, daemon=True).start()
    return {"ok": True, "status": "Đang xử lý", "project_id": pid, "reprocessed": True, "subtitle_source": normalized_source}


@app.post("/api/projects/{pid}/reprocess-analysis")
def reprocess_analysis(pid: str, body: Optional[Dict[str, Any]] = None):
    """Explicit re-run for either STT or OCR. Only this endpoint clears derived data."""
    body = body or {}
    return _start_reprocess(pid, body.get("subtitle_source", "stt"), body.get("ocr_region"))


@app.post("/api/projects/{pid}/reprocess-stt")
def reprocess_stt(pid: str, body: Optional[Dict[str, Any]] = None):
    """Backward-compatible endpoint; respects subtitle_source when supplied so OCR can run too."""
    body = body or {}
    return _start_reprocess(pid, body.get("subtitle_source", "stt"), body.get("ocr_region"))


def _serialize_ocr_segment(seg, segment_id: int):
    out = Exporter._seg(int(segment_id), seg)
    out.update({
        "index": int(segment_id) - 1,
        "segment_id": int(segment_id) - 1,
        "start": out["start_time"],
        "end": out["end_time"],
        "start_sec": out["start_time"],
        "end_sec": out["end_time"],
        "source_start": out["start_time"],
        "source_end": out["end_time"],
        "source_text": out["text"],
        "timing_source": "ocr_master_frame",
        "ocr_locked": True,
        "timing_locked": True,
    })
    return out


def _resolve_ocr_lang(project: dict) -> str:
    """Resolve OCR language independently from translation/STT language.

    The editor may be configured e.g. EN -> VI while the burned subtitle in the source video
    is Chinese/Japanese/Korean. OCR must not inherit the translation source language. Unless the
    user explicitly selects an OCR language, keep OCR in AUTO mode.
    """
    explicit = str(project.get("ocr_lang") or "").strip().lower()
    mapping = {"tự động nhận diện":"auto", "auto":"auto", "vi":"latin", "tiếng việt":"latin",
               "en":"en", "tiếng anh":"en", "zh":"ch", "tiếng trung":"ch", "ja":"japan",
               "tiếng nhật":"japan", "ko":"korean", "tiếng hàn":"korean"}
    return mapping.get(explicit, "auto")


def _run_ocr_v2(pid: str, project: dict, video_path: Path, region: dict):
    """Run only the new VideoOCREngine and publish realtime OCR events.

    This function deliberately has no STT/audio/translation/render dependency.
    It persists partial source segments so the editor can render them immediately.
    """
    _clear_ocr_event_bus(pid)
    state = {
        "segments": [],
        "last_progress": 0.0,
        "last_event_emit_ts": 0.0,
        "last_event_phase": "",
        "last_persist_ts": 0.0,
        "last_persist_percent": -1,
    }
    exporter = Exporter()
    provider = str(project.get("ocr_provider") or "local").lower()
    if provider not in {"local", "rapidocr", "pp-ocr", "ppocr", "rapid-ppocrv5"}:
        raise RuntimeError("OCR V2 hiện dùng PP-OCRv5 Mobile local; provider OCR cũ/remote không được tự động dùng lại.")
    model = str(project.get("ocr_model") or "").strip()
    color_profile = project.get("ocr_color_profile") or {}
    mode = str(project.get("ocr_mode") or "HIGH_RECALL").upper()
    engine_cfg = {
        "scan_fps": float(getattr(config, "OCR_SCAN_FPS", os.getenv("OCR_V2_SCAN_FPS", "20"))),
        "scan_width": int(getattr(config, "OCR_SCAN_WIDTH", os.getenv("OCR_V2_SCAN_WIDTH", "224"))),
        "scan_workers": max(1, min(4, int(getattr(config, "OCR_SCAN_WORKERS", os.getenv("OCR_V2_SCAN_WORKERS", "1"))))),
        "hwaccel": getattr(config, "OCR_HWACCEL", os.getenv("OCR_V2_HWACCEL", "auto")),
        "recovery_seconds": max(1.0, float(getattr(config, "OCR_RECOVERY_SECONDS", os.getenv("OCR_V2_RECOVERY_SECONDS", "4.0")))),
        "burst_fps": max(15.0, float(getattr(config, "OCR_BURST_FPS", os.getenv("OCR_V2_BURST_FPS", "30")))),
        "batch_size": max(1, min(64, int(getattr(config, "OCR_BATCH_SIZE", os.getenv("OCR_V2_BATCH_SIZE", "16"))))),
        "boundary_probe_seconds": float(getattr(config, "OCR_BOUNDARY_PROBE_SECONDS", os.getenv("OCR_V2_BOUNDARY_PROBE_SECONDS", "0.15"))),
        "interior_ocr_fps": float(getattr(config, "OCR_INTERIOR_FPS", os.getenv("OCR_V2_INTERIOR_FPS", "0.5"))),
        "recognition_sample_step": float(getattr(config, "OCR_RECOGNITION_SAMPLE_STEP", os.getenv("OCR_RECOGNITION_SAMPLE_STEP", "0.45"))),
        "track_max_gap": max(0.75, float(getattr(config, "OCR_MAX_TRACK_GAP", os.getenv("OCR_V2_TRACK_MAX_GAP", "1.75")))),
        "track_similarity": float(getattr(config, "OCR_TRACK_SIMILARITY", os.getenv("OCR_TRACK_SIMILARITY", "0.50"))),
        "scan_diff_threshold": float(getattr(config, "OCR_DIFF_THRESHOLD", os.getenv("OCR_DIFF_THRESHOLD", "0.018"))),
        "global_textlike_threshold": float(getattr(config, "OCR_GLOBAL_TEXTLIKE_THRESHOLD", os.getenv("OCR_V2_GLOBAL_TEXTLIKE_THRESHOLD", "0.005"))),
        "global_line_density_threshold": float(getattr(config, "OCR_GLOBAL_LINE_DENSITY_THRESHOLD", os.getenv("OCR_V2_GLOBAL_LINE_DENSITY_THRESHOLD", "0.04"))),
        "boundary_edge_threshold": float(getattr(config, "OCR_BOUNDARY_EDGE_THRESHOLD", os.getenv("OCR_BOUNDARY_EDGE_THRESHOLD", "0.020"))),
        "verification": {"max_samples_per_track": 4},
        "boundary_verify_radius_frames": max(1, min(2, int(getattr(config, "OCR_BOUNDARY_VERIFY_RADIUS_FRAMES", "1")))),
        "boundary_search_seconds": 0.60,
        "mode": mode,
        "lang": _resolve_ocr_lang(project),
        "backend": getattr(config, "OCR_BACKEND", os.getenv("OCR_BACKEND", "auto")),
        "use_gpu": getattr(config, "OCR_USE_GPU", os.getenv("OCR_USE_GPU", "auto")),
        "require_gpu_for_target": bool(getattr(config, "OCR_REQUIRE_GPU_FOR_TARGET", True)),
        "gpu_id": int(getattr(config, "OCR_GPU_ID", os.getenv("OCR_GPU_ID", "0"))),
        "model_dir": getattr(config, "PP_OCR_MODEL_DIR", os.getenv("PP_OCR_MODEL_DIR", "")),
        "job_dir": config.CACHE_DIR / "ocr_v2" / pid,
    }

    def emit(details):
        d = dict(details or {})
        d["ocr_engine"] = "VideoOCREngine-v5-turbo"
        d["ocr_mode"] = mode
        phase = str(d.get("phase") or "processing")
        if "overall_percent" in d:
            state["last_progress"] = float(d.get("overall_percent") or 0)
        pct = int(max(0, min(100, round(state["last_progress"]))))
        now = time.monotonic()

        # Fast-scan callbacks can be much more frequent than durable project updates.
        # Keep SSE responsive while throttling storage writes/log volume to avoid making
        # SQLite/file I/O the bottleneck of the OCR engine.
        terminal_or_phase_change = phase != state["last_event_phase"]
        should_emit = terminal_or_phase_change or (now - state["last_event_emit_ts"] >= 0.25)
        if should_emit:
            _publish_ocr_event(pid, {"type": "progress", **d})
            state["last_event_emit_ts"] = now
            state["last_event_phase"] = phase

        should_persist = (
            terminal_or_phase_change
            or (now - state["last_persist_ts"] >= 0.75 and pct != state["last_persist_percent"])
        )
        if should_persist:
            _update_progress(pid, 2, 6, f"OCR · {phase}", d, pct)
            state["last_persist_ts"] = now
            state["last_persist_percent"] = pct

        if should_emit:
            sample_count = d.get("ocr_frames", d.get("samples_scanned", d.get("phase_done", "-")))
            logger.info(
                "[OCR V2] %s | %.1f%% | samples=%s | candidates=%s | review=%s",
                phase, float(d.get("overall_percent", state["last_progress"])),
                sample_count, d.get("candidates", d.get("candidates_so_far", 0)), d.get("needs_review", 0),
            )

    def on_segment(seg, index, total):
        sd = _serialize_ocr_segment(seg, index)
        # Do not let a late callback rewrite an earlier segment with a new index.
        state["segments"] = [x for x in state["segments"] if int(x.get("id", 0)) != int(index)]
        state["segments"].append(sd)
        state["segments"].sort(key=lambda x: (float(x.get("start_time", 0)), int(x.get("id", 0))))
        texts = [str(x.get("text", "") or "") for x in state["segments"]]
        storage.update_project(
            pid, segments=state["segments"], ocr_texts=texts,
            translated_texts=["" for _ in state["segments"]],
            translation_pairs=[dict(x, translation="", translation_id=i + 1) for i, x in enumerate(state["segments"])],
            ocr_engine="VideoOCREngine-v5-turbo", ocr_mode=mode, status="Đang OCR phụ đề", stage="OCR", error=None,
        )
        _publish_ocr_event(pid, {
            "type": "segment", "segment": sd, "segment_index": index, "segments_count": len(state["segments"]),
            "phase": "recognition", "message": f"Đã khóa câu {index}/{total}",
        })
        _update_progress(
            pid, 2, 6, f"OCR · đã khóa {len(state['segments'])} câu",
            {"phase": "recognition", "segments": len(state["segments"]), "last_segment": sd, "streaming": True},
            min(98, max(1, int(state["last_progress"]))),
        )

    cancel_cb = lambda: is_cancelled(pid)
    result = VideoOCREngine(config.FFMPEG_PATH, config.FFPROBE_PATH, engine_cfg).run(
        video_path, region, model=model, provider=provider, color_profile=color_profile,
        on_event=emit, on_segment=on_segment, cancel_cb=cancel_cb,
    )
    json_path = config.OUTPUT_DIR / pid / "ocr_timeline.json"
    srt_path = config.OUTPUT_DIR / pid / "ocr_timeline.srt"
    exporter.write_json(result, json_path); exporter.write_srt(result, srt_path)
    final_segments = [_serialize_ocr_segment(s, i + 1) for i, s in enumerate(result.segments)]
    storage.update_project(
        pid, segments=final_segments, ocr_texts=[x["text"] for x in final_segments],
        translated_texts=["" for _ in final_segments],
        translation_pairs=[dict(x, translation="", translation_id=i + 1) for i, x in enumerate(final_segments)],
        translation_complete=False, translation_missing_segments=[], tts_segments={},
        ocr_audit=result.audit, ocr_engine="VideoOCREngine-v5-turbo", ocr_mode=mode,
        ocr_json=str(json_path), ocr_srt=str(srt_path), transcript_srt=str(srt_path),
        status="Sẵn sàng chỉnh sửa", stage=None, error=None,
    )
    _update_progress(pid, 6, 6, "OCR hoàn tất — timeline frame-accurate",
                     {"segments": len(final_segments), "ocr_audit": result.audit, "streaming": True,
                      "json": str(json_path), "srt": str(srt_path)}, 100)
    _publish_ocr_event(pid, {"type": "complete", "phase": "complete", "overall_percent": 100,
                             "segments": final_segments, "audit": result.audit,
                             "json_path": str(json_path), "srt_path": str(srt_path), "message": "OCR hoàn tất"})
    return final_segments, result.audit


def _process_job(pid: str):
    project = storage.get_project(pid)
    if not project:
        return
    if is_cancelled(pid):
        storage.update_project(pid, status="Đã hủy")
        return
    try:
        keys = config.get_api_keys()
        video_path = Path(project["video_path"])
        if not video_path.exists():
            raise RuntimeError(f"Video không tồn tại: {video_path}")
        audio_wav = video_path.parent / "audio.wav"
        subtitle_source = str(project.get("subtitle_source") or "stt").lower()
        # OCR MODE IS IMAGE-ONLY. STT/audio must never participate in OCR text or timing.
        if subtitle_source == "ocr":
            region = project.get("ocr_region") or {"x": 0, "y": 65, "w": 100, "h": 35}
            _update_progress(pid, 1, 6, "OCR V2 · khởi tạo engine mới", {"phase": "metadata"}, 1)
            storage.update_project(pid, status="Đang OCR phụ đề", stage="OCR", error=None, ocr_engine="VideoOCREngine-v5-turbo", ocr_mode=str(project.get("ocr_mode") or "HIGH_RECALL"))
            try:
                _run_ocr_v2(pid, project, video_path, region)
            except RuntimeError as exc:
                if str(exc).lower().endswith("cancelled") or is_cancelled(pid):
                    storage.update_project(pid, status="Đã hủy", stage=None, error=None)
                    _publish_ocr_event(pid, {"type": "cancelled", "message": "OCR đã hủy"})
                    return
                raise
            return
        else:
            _update_progress(pid, 1, 6, "Tách âm thanh")
            storage.update_project(pid, status="Đang tách âm thanh")
            audio_mod.extract_audio(video_path, audio_wav)
            if is_cancelled(pid):
                storage.update_project(pid, status="Đã hủy"); return
            _update_progress(pid, 2, 6, "Nhận diện giọng nói")
            storage.update_project(pid, status="Đang nhận diện giọng nói")
            stt_provider=project.get("stt_provider") or config.DEFAULT_PROVIDERS.get("stt_provider","groq")
            stt_id=_select_stt_provider(stt_provider, keys)
            stt_key=keys.get("groq") if stt_id=="groq" else keys.get(stt_id) if stt_id not in {"google"} else ""
            if stt_id not in {"google"} and not stt_key: raise RuntimeError(f"Thiếu API key cho STT provider '{stt_provider}'")
            lang_code=LANG_NAME_TO_CODE.get(project.get("src_lang"),"auto")
            glossary_hint="; ".join(str(x.get("source")) for x in project.get("glossary",[]) if isinstance(x,dict) and x.get("source"))
            prompt_hint=(project.get("additional_prompt") or "")
            if glossary_hint: prompt_hint=(prompt_hint+" | " if prompt_hint else "")+"Giữ đúng tên riêng/thuật ngữ: "+glossary_hint
            # ========================================================
            # STT IS STANDALONE. OCR is never imported/called in STT mode.
            # No OCR timing, no visual fusion, no audio/environment text fusion.
            # ========================================================
            segments = _run_with_timeout(
                stt_mod.transcribe_fast, STT_TIMEOUT_SECONDS, label=f"STT/{stt_id}",
                audio_path=str(audio_wav), api_key=stt_key or "", provider=stt_id,
                language=None if lang_code == "auto" else lang_code,
                initial_prompt=prompt_hint,
                model=(project.get("stt_model") or None),
            )
            if not segments:
                raise RuntimeError("STT không nhận diện được lời thoại nào trong video.")
            if hasattr(stt_mod, "normalize_segments"):
                segments = stt_mod.normalize_segments(segments)
            if hasattr(stt_mod, "merge_micro_segments"):
                segments = stt_mod.merge_micro_segments(segments)
            logger.info("[STT ONLY] hoàn tất %d đoạn | OCR=disabled | model=%s",
                        len(segments), project.get("stt_model") or "provider-default")
            _update_progress(pid, 2, 6, "Nhận diện giọng nói",
                             {"segments": len(segments), "ocr_used": False,
                              "stt_model": project.get("stt_model") or "provider-default"}, 33)
            # V21.1: tạo SRT tiếng Trung ngay sau khi STT xong, không cần chờ dịch/render.
            # Đây là transcript gốc 1:1 theo Master Timeline; render cuối sẽ ghi đè cùng file.
            stt_out_dir = config.OUTPUT_DIR / pid
            stt_out_dir.mkdir(parents=True, exist_ok=True)
            transcript_srt = stt_out_dir / "transcript.srt"
            source_texts = [str(seg.get("text", seg.get("source_text", "")) or "") for seg in segments]
            subtitle_mod.write_srt(subtitle_mod.lock_source_timing(segments), source_texts, transcript_srt)
            storage.update_project(pid, segments=segments, transcript_srt=str(transcript_srt))
            logger.info(f"[STT] Đã tạo transcript.srt: {transcript_srt} ({len(source_texts)} đoạn)")
            _update_progress(pid,2,6,"Nhận diện giọng nói",{"segments":len(segments),"transcript_srt":str(transcript_srt)},33)
        if is_cancelled(pid):
            storage.update_project(pid, status="Đã hủy")
            return
        _update_progress(pid, 3, 6, "Dịch thuật")
        storage.update_project(pid, status="Đang dịch")
        target_lang = LANG_TARGET_NAME.get(project["dst_lang"], project["dst_lang"])
        translate_provider = project.get("translate_provider") or config.DEFAULT_PROVIDERS.get("translate_provider", "groq")
        trans_id = _resolve_translate_id(translate_provider)
        trans_key = None
        if trans_id == "groq":
            trans_key = keys.get("groq")
        elif trans_id == "xkira":
            trans_key = keys.get("xkira_keys") or keys.get("xkira")
        elif trans_id == "gemini":
            trans_key = keys.get("gemini")
        elif trans_id == "google":
            trans_key = None
        else:
            raise RuntimeError(f"Translation provider '{trans_id}' không được hỗ trợ (chỉ Groq/xKiro/Gemini/Google).")
        if trans_id != "google" and not trans_key:
            raise RuntimeError(f"Thiếu API key cho Translation provider '{translate_provider}'")
        extra_prompt = project.get("additional_prompt") or ""
        translated = translate_mod.translate_segments_fast(
            segments=segments, target_lang=target_lang, api_key=trans_key,
            provider=trans_id, max_workers=config.MAX_TRANSLATE_WORKERS,
            glossary=project.get("glossary") or [], additional_prompt=extra_prompt,
            model=project.get("translate_model") or (project.get("provider_config") or {}).get(trans_id, {}).get("model") or None)
        if bool(project.get("translate_refine")) and all(str(x).strip() for x in translated):
            refine_prompt=(extra_prompt+" | " if extra_prompt else "") + "Đây là vòng soát lại bản dịch lần hai. Giữ nguyên thuật ngữ và tên riêng; ưu tiên câu tiếng Việt tự nhiên, ngắn gọn để đọc thoại; không đổi số lượng câu."
            translated=translate_mod.translate_segments_fast(segments=segments,target_lang=target_lang,api_key=trans_key,provider=trans_id,max_workers=config.MAX_TRANSLATE_WORKERS,force=True,glossary=project.get("glossary") or [],additional_prompt=refine_prompt, model=project.get("translate_model") or (project.get("provider_config") or {}).get(trans_id, {}).get("model") or None)
        if len(translated) != len(segments):
            raise RuntimeError(f"Dịch không hoàn tất: {len(translated)}/{len(segments)}")
        # FIX: chỉ nhận kết quả align nếu nó GIỮ NGUYÊN số đoạn (đảm bảo 1:1)
        if hasattr(subtitle_mod, "align_split_segments"):
            try:
                aligned_segments, aligned_translated = subtitle_mod.align_split_segments(segments, translated)
                if len(aligned_segments) == len(translated) and len(aligned_translated) == len(translated):
                    segments, translated = aligned_segments, aligned_translated
                else:
                    logger.warning(
                        "⚠️ align_split_segments đổi số đoạn (%s -> %s); giữ nguyên 1:1",
                        len(segments), len(aligned_segments))
            except Exception as e:
                logger.warning(f"⚠️ align_split_segments lỗi, giữ nguyên segments: {e}")
        storage.update_project(pid, segments=segments)
        pairs = []
        for i, (src, dst) in enumerate(zip(segments, translated), 1):
            item = dict(src)
            item["translation"] = str(dst or "").strip()
            item["translation_id"] = i
            pairs.append(item)
        # Final truth check: a segment is translated only when its text is non-empty.
        # Keep explicit 0-based index + 1-based editor number + timing + source text
        # so a missing segment is immediately findable in the UI/log.
        missing_infos = []
        for i, txt in enumerate(translated):
            if not str(txt or '').strip():
                seg = segments[i] if i < len(segments) else {}
                try: start_sec = float(seg.get('start', seg.get('start_sec', 0)) or 0)
                except Exception: start_sec = 0.0
                try: end_sec = float(seg.get('end', seg.get('end_sec', start_sec)) or start_sec)
                except Exception: end_sec = start_sec
                source = str(seg.get('text', seg.get('source_text', '')) or '').strip()
                item = {
                    "index": i,
                    "segment_number": i + 1,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "source_text": source,
                }
                missing_infos.append(item)
                logger.warning("⚠️ Missing translation: segment #%s (idx=%s) %0.2f-%0.2fs | %s",
                               i + 1, i, start_sec, end_sec, source[:160].replace('\n', ' '))
        all_translated = not missing_infos and len(translated) == len(segments)
        if missing_infos:
            missing_count = len(missing_infos)
            first = missing_infos[0]
            error_msg = (f"Còn {missing_count} đoạn chưa dịch được. "
                         f"Xem segment #{first['segment_number']} | "
                         f"{first['start_sec']:.2f}s–{first['end_sec']:.2f}s | "
                         f"{first['source_text'][:120]}")
            storage.update_project(pid, status="Sẵn sàng chỉnh sửa", error=error_msg,
                                   translation_missing_segments=missing_infos,
                                   translated_texts=translated, translation_pairs=pairs, translation_complete=False)
        else:
            storage.update_project(pid, status="Sẵn sàng chỉnh sửa", error=None,
                                   translation_missing_segments=[], translated_texts=translated,
                                   translation_pairs=pairs, translation_complete=True)
        _update_progress(pid, 3, 6, "Dịch thuật", {"segments": len(pairs)}, 100 if all_translated else 80)
        cancel_flags.pop(pid, None)
    except Exception as e:
        logger.error(f"❌ Process job lỗi: {traceback.format_exc()}")
        _set_error(pid, str(e) or repr(e))

# ============================================================
# ROUTES: Voice / visual analysis
# ============================================================
def _analysis_store(p,kind): return dict(p.get('voice_analysis' if kind=='voice' else 'visual_analysis') or {'status':'NONE'})

@app.get("/api/projects/{pid}/voice-analysis")
def get_voice_analysis(pid:str):
    p=_project_or_404(pid); return _analysis_store(p,'voice')

@app.post("/api/projects/{pid}/voice-analysis")
def start_voice_analysis(pid:str, body:dict=None):
    p=_project_or_404(pid); segments=[dict(x) for x in (p.get('segments') or [])]
    speakers=[{'id':'speaker_1','label':'Người nói 1','gender':'unknown','segments':len(segments),'voice':''}]
    for x in segments: x['speaker_label']='Người nói 1'; x['speaker_id']='speaker_1'
    data={'status':'DONE','speakers':speakers,'updated_at':time.time()}
    storage.update_project(pid,voice_analysis=data,segments=segments)
    return data

@app.get("/api/projects/{pid}/visual-analysis")
def get_visual_analysis(pid:str):
    p=_project_or_404(pid); return _analysis_store(p,'visual')

@app.post("/api/projects/{pid}/visual-analysis")
def start_visual_analysis(pid:str, body:dict=None):
    p=_project_or_404(pid); video=Path(p.get('video_path',''))
    duration=0.0
    try: duration=audio_mod.get_video_info(video).get('duration',0.0)
    except Exception: pass
    scenes=[]
    if duration>0:
        for i in range(min(8,max(3,int(duration//60)+1))):
            tm=duration*(i+0.5)/min(8,max(3,int(duration//60)+1))
            scenes.append({'start':round(tm,2),'end':round(min(duration,tm+5),2),'description':f'Khung hình kiểm tra tại {tm:.1f}s'})
    data={'status':'DONE','summary':f'Đã phân tích {len(scenes)} mốc hình ảnh để hỗ trợ ngữ cảnh.','scenes':scenes,'updated_at':time.time()}
    storage.update_project(pid,visual_analysis=data)
    return data

@app.patch("/api/projects/{pid}/speakers/{speaker_id}")
def save_speaker(pid:str,speaker_id:str,body:dict):
    p=_project_or_404(pid); a=_analysis_store(p,'voice'); speakers=list(a.get('speakers') or [])
    for sp in speakers:
        if str(sp.get('id'))==speaker_id: sp['voice']=str((body or {}).get('voice') or '')
    a['speakers']=speakers; storage.update_project(pid,voice_analysis=a); return {'ok':True}

# ============================================================
# ROUTES: Render
# ============================================================
@app.post("/api/projects/{pid}/render")
def render_project(pid: str, body: RenderIn = RenderIn()):
    p = _project_or_404(pid)
    segments = p.get("segments") or []
    translated = list(p.get("translated_texts") or [])
    while len(translated) < len(segments):
        translated.append("")
    if len(translated) > len(segments):
        translated = translated[:len(segments)]
    if body.include_dubbing:
        tts_map = p.get("tts_segments") or {}
        missing = [i for i in range(len(segments))
                   if not isinstance(tts_map.get(str(i)), dict)
                   or not (tts_map.get(str(i) or {}).get("audio") or tts_map.get(str(i) or {}).get("audio_path"))]
        if missing:
            logger.warning("⚠️ Render partial: bỏ qua %d segment chưa có TTS: %s", len(missing), missing[:20])
    render_cfg = dict(p.get("render") or {})
    render_cfg.update({
        "include_subtitle": bool(body.include_subtitle),
        "include_dubbing": bool(body.include_dubbing),
        "quality": body.quality,
        "orig_volume": float(body.orig_volume),
        "dub_volume": float(body.dub_volume),
        "reduce_original_voice": bool(body.reduce_original_voice),
        "original_voice_reduction": float(body.original_voice_reduction),
    })
    storage.update_project(pid, status="Đang render", error=None, render=render_cfg)

    def run_render():
        _render_job(pid, body.keep_original_audio, _parse_rate(body.rate), _parse_pitch(body.pitch),
                    include_subtitle=body.include_subtitle, include_dubbing=body.include_dubbing,
                    orig_volume=body.orig_volume, dub_volume=body.dub_volume,
                    reduce_original_voice=body.reduce_original_voice, original_voice_reduction=body.original_voice_reduction)

    thread = threading.Thread(target=run_render, daemon=True)
    thread.start()
    return {"ok": True, "status": "Đang render", "include_subtitle": body.include_subtitle,
            "include_dubbing": body.include_dubbing}

def _render_job(pid: str, keep_original_audio: bool, rate: float, pitch: float,
                include_subtitle: bool = True, include_dubbing: bool = True,
                orig_volume: float = 0.2, dub_volume: float = 1.0,
                reduce_original_voice: bool = False, original_voice_reduction: float = 0.0):
    project = storage.get_project(pid)
    if not project:
        return
    try:
        translated = list(project.get("translated_texts") or [])
        segments = project.get("segments") or []
        while len(translated) < len(segments):
            translated.append("")
        if len(translated) > len(segments):
            translated = translated[:len(segments)]
        missing_translation = [i for i, txt in enumerate(translated) if not str(txt or '').strip()]
        if missing_translation:
            # Self-heal only the missing rows; keep 1:1 indexes and never touch source timing/STT.
            try:
                keys_retry = config.get_api_keys()
                provider_retry = _resolve_translate_id(project.get("translate_provider") or config.DEFAULT_PROVIDERS.get("translate_provider", "groq"))
                target_retry = LANG_TARGET_NAME.get(project.get("dst_lang", "Tiếng Việt"), project.get("dst_lang", "Tiếng Việt"))
                key_retry = keys_retry.get(provider_retry) or (keys_retry.get("groq") if provider_retry == "groq" else None)
                if provider_retry in {"groq", "xkira"} and key_retry:
                    retry_segments = []
                    for idx in missing_translation:
                        src = dict(segments[idx])
                        src["text"] = str(src.get("text", src.get("source_text", "")) or "").strip()
                        retry_segments.append(src)
                    retry_texts = translate_mod.translate_segments_fast(
                        segments=retry_segments, target_lang=target_retry, api_key=key_retry,
                        provider=provider_retry, max_workers=max(1, min(4, config.MAX_TRANSLATE_WORKERS)),
                        force=True, auto_retry=1, glossary=project.get("glossary") or [],
                        additional_prompt="Chỉ dịch các câu còn thiếu. Trả về bản dịch tự nhiên, không thêm giải thích.",
                        model=(project.get("provider_config") or {}).get(provider_retry, {}).get("model") or None)
                    for idx, value in zip(missing_translation, retry_texts):
                        if str(value or "").strip():
                            translated[idx] = str(value).strip()
                    missing_translation = [i for i, txt in enumerate(translated) if not str(txt or '').strip()]
                    logger.info("[TRANSLATE-RETRY] recovered missing translations; still_missing=%d", len(missing_translation))
                    storage.update_project(pid, translated_texts=translated,
                        translation_pairs=[dict(seg, translation=str(translated[i] or ""), translation_id=i+1) for i,seg in enumerate(segments)],
                        translation_complete=not missing_translation)
            except Exception as retry_exc:
                logger.warning("[TRANSLATE-RETRY] không thể tự dịch lại đoạn thiếu: %s", retry_exc)
            if missing_translation:
                logger.warning("⚠️ Render partial: %d segment vẫn thiếu bản dịch; các đoạn còn lại vẫn tiếp tục render", len(missing_translation))
        _update_progress(pid, 4, 6, "Chuẩn bị render")
        storage.update_project(pid, status="Đang render video")
        keys = config.get_api_keys()
        if include_dubbing:
            tts_id, voice, tts_key = _resolve_tts_settings(project, keys)
        else:
            tts_id, voice, tts_key = "edge_tts", str(project.get("voice") or getattr(config, "DEFAULT_EDGE_VOICE", "vi-VN-NamMinhNeural")), None
        video_path = Path(project["video_path"])
        out_dir = config.OUTPUT_DIR / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / "dubbed.mp4"
        existing_tts = project.get("tts_segments") if include_dubbing else None
        _update_progress(pid, 5, 6, "Kết xuất video")
        reduction=float(original_voice_reduction or 0.0) if reduce_original_voice else 0.0
        effective_orig_volume=max(0.0,float(orig_volume))*(1.0-max(0.0,min(1.0,reduction)))
        render_kwargs = dict(
            video_path=video_path, segments=segments, translated_texts=translated,
            tts_provider=tts_id, voice=voice, tts_api_key=tts_key,
            output_path=out_video, keep_original_audio=keep_original_audio,
            max_workers=config.MAX_TTS_WORKERS, rate=rate, pitch=pitch,
            existing_tts=existing_tts, include_dubbing=include_dubbing,
            include_subtitle=include_subtitle, burn_subtitle=include_subtitle,
            orig_volume=effective_orig_volume, dub_volume=dub_volume)
        render_mod.render_final_video_fast(**render_kwargs)

        out_srt = out_dir / "subtitle.srt"
        # subtitle.srt is written by render.py from the final dubbing schedule so each
        # translated cue stays visible until its TTS sentence finishes. Do not overwrite it.
        # SRT nguồn được giữ đúng timing đã sửa để người dùng có thể kiểm tra.
        src_srt = out_dir / "transcript.srt"
        source_texts = [str(seg.get("text", seg.get("source_text", "")) or "") for seg in segments]
        subtitle_mod.write_srt(subtitle_mod.lock_source_timing(segments), source_texts, src_srt)

        output_audio = out_dir / "final_audio.m4a"
        latest_project = storage.get_project(pid) or project
        tts_map_final = latest_project.get("tts_segments") or {}
        missing_tts = [i for i in range(len(segments))
                       if not isinstance(tts_map_final.get(str(i)), dict)
                       or not ((tts_map_final.get(str(i)) or {}).get("audio")
                               or (tts_map_final.get(str(i)) or {}).get("audio_path"))]
        render_warnings = []
        if missing_translation:
            render_warnings.append(f"Bỏ qua {len(missing_translation)} segment chưa có bản dịch")
        if include_dubbing and missing_tts:
            render_warnings.append(f"Bỏ qua {len(missing_tts)} segment chưa có TTS")
        storage.update_project(
            pid, status="Hoàn tất", output_video=str(out_video),
            output_srt=str(out_srt), transcript_srt=str(src_srt),
            output_audio=str(output_audio) if output_audio.exists() else None,
            render_warnings=render_warnings, latest_output_id=uuid.uuid4().hex[:12], error=None)
        _update_progress(pid, 6, 6, "Hoàn tất", {"video": str(out_video)}, 100)
    except Exception as e:
        logger.error(f"Render job lỗi: {traceback.format_exc()}")
        _set_error(pid, str(e) or repr(e))

# ============================================================
# ROUTES: Cancellation
# ============================================================
@app.post("/api/projects/{pid}/cancel")
def cancel_project(pid: str):
    _project_or_404(pid)
    cancel_flags[pid] = True
    storage.update_project(pid, status="Đang hủy", error=None)
    return {"ok": True, "status": "Đang hủy"}

# ============================================================
# ROUTES: Stream
# ============================================================
@app.get("/api/projects/{pid}/stream")
def stream_project(pid: str, request: Request):
    p = _project_or_404(pid)
    output_video = p.get("output_video")
    if output_video and Path(output_video).exists():
        logger.info(f"📹 Stream dubbed video cho project {pid}")
        return _serve_file_with_range(request, Path(output_video), "video/mp4", inline=True)
    video_path = Path(p.get("video_path", ""))
    if not video_path.exists():
        raise HTTPException(404, "Video không tồn tại")
    codec = "unknown"
    try:
        result = subprocess.run(
            [config.FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10)
        codec = result.stdout.strip().lower()
    except Exception as e:
        logger.warning(f"Không thể detect codec: {e}")
    supported_codecs = {"h264", "avc", "vp8", "vp9", "av1", "mpeg4", "mp4v"}
    if codec in supported_codecs:
        logger.info(f"📹 Stream trực tiếp source (codec={codec})")
        return _serve_file_with_range(request, video_path, "video/mp4", inline=True)
    logger.info(f"🔄 Transcode {codec} → H.264 cho project {pid}...")
    cache_dir = config.CACHE_DIR / "stream"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{pid}_h264.mp4"
    if not cached.exists():
        cmd = [config.FFMPEG_PATH, "-y", "-i", str(video_path),
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
               "-c:a", "copy", "-movflags", "+faststart", str(cached)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                logger.error(f"FFmpeg transcode lỗi: {result.stderr[:500]}")
                raise HTTPException(500, f"Transcode thất bại: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            raise HTTPException(500, "Transcode timeout (video quá dài)")
    if cached.exists():
        return _serve_file_with_range(request, cached, "video/mp4", inline=True)
    raise HTTPException(500, "Không thể tạo file stream")

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_STREAM_CHUNK_SIZE = 1024 * 1024

def _iter_file_range(path: Path, start: int, end: int, chunk_size: int = _STREAM_CHUNK_SIZE):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk

def _serve_file_with_range(request: Request, path: Path, media_type: str, inline: bool = True):
    file_size = path.stat().st_size
    disposition = "inline" if inline else "attachment"
    headers_base = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'{disposition}; filename="{path.name}"',
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Range, Content-Type",
    }
    range_header = request.headers.get("range")
    if range_header:
        match = _RANGE_RE.match(range_header)
        if match:
            start_s, end_s = match.groups()
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                return Response(status_code=416,
                                headers={"Content-Range": f"bytes */{file_size}", **headers_base})
            headers = {**headers_base,
                       "Content-Range": f"bytes {start}-{end}/{file_size}",
                       "Content-Length": str(end - start + 1)}
            return StreamingResponse(_iter_file_range(path, start, end),
                                     status_code=206, media_type=media_type, headers=headers)
    headers = {**headers_base, "Content-Length": str(file_size)}
    return StreamingResponse(_iter_file_range(path, 0, file_size - 1),
                             status_code=200, media_type=media_type, headers=headers)

# ============================================================
# ROUTES: Download
# ============================================================
@app.get("/api/projects/{pid}/links")
def project_links(pid: str):
    p = _project_or_404(pid)
    out_dir = config.OUTPUT_DIR / pid
    links = {}
    def add(key, path, media="application/octet-stream"):
        if path and Path(path).exists():
            links[key] = f"/api/projects/{pid}/download?type={key}"
    add("video_final", p.get("output_video"), "video/mp4")
    add("translated_srt", p.get("output_srt"), "text/plain")
    add("transcript_srt", p.get("transcript_srt"), "text/plain")
    add("audio_dubbing", p.get("output_audio"), "audio/mp4")
    src_audio = Path(p.get("video_path", "")).parent / "audio.wav"
    add("audio_original", src_audio, "audio/wav")
    add("video_original", p.get("video_path"), "video/mp4")
    if out_dir.exists():
        cfg = out_dir / "project.json"
        if not cfg.exists():
            cfg.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
        add("project_config", cfg, "application/json")
    return links

@app.get("/api/projects/{pid}/download")
def download_project(pid: str, request: Request, type: str = "video", segment: Optional[int] = None, download: int = 0):
    p = _project_or_404(pid)
    out_dir = config.OUTPUT_DIR / pid
    if type in {"audio", "audio_dubbing"}:
        if segment is not None:
            f = out_dir / "tts" / f"seg{int(segment):04d}.mp3"
            if not f.exists():
                raise HTTPException(404, "Audio segment chưa sẵn sàng")
            return _serve_file_with_range(request, f, "audio/mpeg", inline=not bool(download))
        if type == "audio_dubbing":
            path = p.get("output_audio")
            if path and Path(path).exists():
                return FileResponse(str(path), media_type="audio/mp4", filename=Path(path).name)
        tts_dir = out_dir / "tts"
        files = sorted(tts_dir.glob("seg*.mp3")) if tts_dir.exists() else []
        if not files:
            raise HTTPException(404, "Chưa có file giọng đọc")
        zip_path = config.TEMP_DIR / f"{pid}_tts.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, f.name)
        return FileResponse(str(zip_path), media_type="application/zip", filename=f"{pid}_tts.zip")
    if type == "zip":
        zip_path = config.TEMP_DIR / f"{pid}_project.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        candidates = []
        for key in ("output_video","output_srt","transcript_srt","output_audio","video_path"):
            if p.get(key) and Path(p[key]).exists():
                candidates.append((Path(p[key]), Path(p[key]).name))
        tts_dir = out_dir / "tts"
        for f in sorted(tts_dir.glob("seg*.mp3")) if tts_dir.exists() else []:
            candidates.append((f, f"tts/{f.name}"))
        cfg = out_dir / "project.json"
        cfg.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
        candidates.append((cfg, "project.json"))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for path, arc in candidates:
                z.write(path, arc)
        return FileResponse(str(zip_path), media_type="application/zip", filename=zip_path.name)
    mapping = {
        "video": (p.get("output_video"), "video/mp4", True),
        "srt": (p.get("output_srt"), "text/plain", False),
        "translated_srt": (p.get("output_srt"), "text/plain", False),
        "transcript_srt": (p.get("transcript_srt"), "text/plain", False),
        "source": (p.get("video_path"), "video/mp4", True),
        "video_original": (p.get("video_path"), "video/mp4", True),
        "video_final": (p.get("output_video"), "video/mp4", True),
        "audio_original": (str(Path(p.get("video_path", "")).parent / "audio.wav"), "audio/wav", False),
        "audio_dubbing": (p.get("output_audio"), "audio/mp4", False),
        "project_config": (str(out_dir / "project.json"), "application/json", False),
    }
    if type not in mapping:
        raise HTTPException(400, "type không hợp lệ")
    path, media, inline = mapping[type]
    if type == "project_config" and not Path(path).exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
    if not path or not Path(path).exists():
        raise HTTPException(404, "File chưa sẵn sàng")
    return _serve_file_with_range(request, Path(path), media, inline=(inline and not bool(download)))

@app.get("/api/projects/{pid}/ocr/events")
def stream_ocr_events(pid: str, request: Request):
    """Server-Sent Events stream for live OCR logs, progress and partial/final segments."""
    _project_or_404(pid)
    q = Queue(maxsize=200)
    with _ocr_event_lock:
        history = list(_ocr_event_history.get(pid, ()))
        _ocr_event_subscribers[pid].add(q)

    terminal_types = {"complete", "error", "cancelled"}

    def encode(ev):
        return "id: %s\ndata: %s\n\n" % (
            ev.get("seq", ""),
            json.dumps(ev, ensure_ascii=False, default=str, separators=(",", ":")),
        )

    async def generate():
        try:
            # Replay buffered events first. If the terminal event is already in history,
            # close the stream immediately instead of replaying forever with heartbeats.
            for ev in history:
                yield encode(ev)
                if ev.get("type") in terminal_types:
                    return

            while True:
                # Starlette/FastAPI exposes request.is_disconnected() as an async API.
                # The previous stream called a missing disconnect helper, which crashed
                # the SSE response while the OCR worker continued in the background.
                if await request.is_disconnected():
                    return
                try:
                    # q.get() is blocking, so move it off the event loop. A short timeout
                    # lets us re-check client disconnects regularly without busy-spinning.
                    ev = await asyncio.to_thread(q.get, True, 2.0)
                except Empty:
                    yield ": heartbeat\n\n"
                    continue
                yield encode(ev)
                if ev.get("type") in terminal_types:
                    return
        finally:
            with _ocr_event_lock:
                _ocr_event_subscribers[pid].discard(q)

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


# ============================================================
# ROUTES: OCR
# ============================================================
@app.post("/api/projects/{pid}/ocr/evaluate")
def evaluate_ocr(pid: str, body: dict):
    """Compare the persisted OCR master timeline with supplied ground truth."""
    project = _project_or_404(pid)
    predicted = list(project.get("segments") or [])
    ground_truth = body.get("segments") if isinstance(body, dict) else None
    if not isinstance(ground_truth, list):
        raise HTTPException(400, "body.segments phải là một mảng ground-truth OCR")
    return {"ok": True, "metrics": GroundTruthEvaluator().evaluate(predicted, ground_truth)}


@app.post("/api/projects/{pid}/ocr")
def ocr_project(pid: str, body: OcrIn):
    """Run OCR on the selected subtitle box; it may create the whole OCR timeline."""
    p = _project_or_404(pid)
    video_path = Path(p.get("video_path", ""))
    if not video_path.exists():
        raise HTTPException(404, "Không tìm thấy file video gốc")
    region = _normalize_ocr_region_value(body.region.model_dump() if hasattr(body.region, "model_dump") else body.region.dict())
    if not acquire_processing_lock(pid):
        return {"ok": True, "status": "Đang xử lý", "already_running": True}
    cancel_flags.pop(pid, None)
    storage.update_project(pid, subtitle_source="ocr", ocr_region=region, status="Đang đọc OCR", stage="OCR", error=None)

    def run_ocr():
        try:
            _ocr_job(pid, region, body.retranslate)
        except Exception as exc:
            logger.error("OCR job lỗi: %s", exc, exc_info=True)
            _set_error(pid, str(exc))
        finally:
            release_processing_lock(pid)

    threading.Thread(target=run_ocr, daemon=True).start()
    return {"ok": True, "status": "Đang đọc OCR", "subtitle_source": "ocr", "ocr_region": region}


def _ocr_job(pid: str, region: Dict[str, float], retranslate: bool):
    """Compatibility entry point for /ocr; OCR/timeline always use VideoOCREngine-v2."""
    project = storage.get_project(pid)
    if not project:
        return
    try:
        video_path = Path(project.get("video_path", ""))
        if not video_path.exists():
            raise RuntimeError("Video không tồn tại")
        _run_ocr_v2(pid, project, video_path, region)
        if retranslate:
            # Existing translation feature is kept outside the OCR engine.
            fresh = storage.get_project(pid) or project
            ocr_segments = list(fresh.get("segments") or [])
            keys = config.get_api_keys()
            target_lang = LANG_TARGET_NAME.get(fresh.get("dst_lang"), fresh.get("dst_lang"))
            translate_provider = fresh.get("translate_provider") or config.DEFAULT_PROVIDERS.get("translate_provider", "groq")
            trans_id = _resolve_translate_id(translate_provider)
            trans_key = keys.get("groq") if trans_id=="groq" else (keys.get("xkira_keys") or keys.get("xkira")) if trans_id=="xkira" else keys.get("gemini") if trans_id=="gemini" else None
            if trans_id != "google" and not trans_key:
                raise RuntimeError(f"Thiếu API key cho Translation provider '{translate_provider}'")
            translated = translate_mod.translate_segments_fast(
                ocr_segments, target_lang, trans_key, provider=trans_id, force=True,
                glossary=fresh.get("glossary") or [], additional_prompt=fresh.get("additional_prompt") or "",
                model=(fresh.get("provider_config") or {}).get(trans_id, {}).get("model") or None,
            )
            final_translated=[str(translated[i] if i < len(translated) else "").strip() for i in range(len(ocr_segments))]
            pairs=[dict(seg, translation=final_translated[i], translation_id=i+1) for i,seg in enumerate(ocr_segments)]
            storage.update_project(pid, translated_texts=final_translated, translation_pairs=pairs,
                translation_complete=all(final_translated),
                translation_missing_segments=[] if all(final_translated) else [i for i,t in enumerate(final_translated) if not t])
            storage.update_project(pid, status="Đã dịch", stage=None, error=None)
    except RuntimeError as exc:
        if str(exc).lower().endswith("cancelled") or is_cancelled(pid):
            storage.update_project(pid, status="Đã hủy", stage=None, error=None)
            _publish_ocr_event(pid, {"type":"cancelled", "message":"OCR đã hủy"})
            return
        logger.error("OCR job lỗi: %s", exc, exc_info=True)
        _set_error(pid, str(exc))
    except Exception as exc:
        logger.error("OCR job lỗi: %s", traceback.format_exc())
        _set_error(pid, str(exc) or repr(exc))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting FlowAI Backend — OCR Master Frame-Locked")
    logger.info(f"📂 Data root: {config.DATA_ROOT}")
    logger.info("🔗 API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False,
                timeout_keep_alive=300, log_level=config.LOG_LEVEL.lower())