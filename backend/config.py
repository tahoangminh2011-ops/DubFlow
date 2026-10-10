"""config.py - FlowAI Ultra Fast"""
BACKEND_CONFIG_VERSION = "V15_VIDEO_OCR_V2_HIGH_RECALL"
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Load the backend-local .env deterministically, regardless of the process working
# directory. Existing process environment variables still take precedence.
_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)

# ============================================================
# PATHS - Mặc định lưu toàn bộ dữ liệu runtime trên ổ D:
# Có thể đổi bằng VIDEOAI_STORAGE_ROOT. DATA_ROOT cũ vẫn được hỗ trợ
# nhưng chỉ dùng khi VIDEOAI_STORAGE_ROOT chưa được đặt.
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DNV = os.getenv("VIDEOAI_STORAGE_ROOT", "").strip()
DATA_ENV = os.getenv("DATA_ROOT", "").strip()
DATA_ROOT = Path(STORAGE_DNV or DATA_ENV or r"D:\VideoTranslatorData").expanduser()
UPLOAD_DIR = DATA_ROOT / "uploads"
OUTPUT_DIR = DATA_ROOT / "outputs"
CACHE_DIR = DATA_ROOT / "cache"
LOGS_DIR = DATA_ROOT / "logs"
PREVIEW_DIR = DATA_ROOT / "preview"
TEMP_DIR = DATA_ROOT / "temp"
PROJECTS_DIR = DATA_ROOT / "projects"
BACKEND_DIR = DATA_ROOT / "backend"
DB_PATH = BACKEND_DIR / "projects.json"
STT_CACHE_DIR = CACHE_DIR / "stt"
TRANSLATE_CACHE_DIR = CACHE_DIR / "translate"
TTS_CACHE_DIR = CACHE_DIR / "tts"
OCR_CACHE_DIR = CACHE_DIR / "ocr"
RENDER_CACHE_DIR = CACHE_DIR / "render"
AUDIO_CACHE_DIR = CACHE_DIR / "audio"
VIDEO_CACHE_DIR = CACHE_DIR / "video"

# ============================================================
# MODEL / EXTERNAL CACHE ROOTS - đặt trước khi import các thư viện model
# ============================================================
MODEL_DIR = DATA_ROOT / "models"
HUGGINGFACE_HOME = MODEL_DIR / "huggingface"
HF_HUB_CACHE_DIR = HUGGINGFACE_HOME / "hub"
HF_XET_CACHE_DIR = HUGGINGFACE_HOME / "xet"
HF_ASSETS_CACHE_DIR = HUGGINGFACE_HOME / "assets"
TORCH_HOME_DIR = MODEL_DIR / "torch"
FASTER_WHISPER_MODEL_DIR = MODEL_DIR / "faster-whisper"
for _model_d in [HUGGINGFACE_HOME, HF_HUB_CACHE_DIR, HF_XET_CACHE_DIR, HF_ASSETS_CACHE_DIR,
                 TORCH_HOME_DIR, FASTER_WHISPER_MODEL_DIR]:
    _model_d.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HUGGINGFACE_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
os.environ["HF_XET_CACHE"] = str(HF_XET_CACHE_DIR)
os.environ["HF_ASSETS_CACHE"] = str(HF_ASSETS_CACHE_DIR)
os.environ["TORCH_HOME"] = str(TORCH_HOME_DIR)
# Chỉ chuyển TEMP/TMP của process backend sang E; không thay đổi TEMP toàn Windows.
TMP_PROCESS_DIR = DATA_ROOT / "temp"
TMP_PROCESS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TMP_PROCESS_DIR)
os.environ["TMP"] = str(TMP_PROCESS_DIR)
try:
    import tempfile as _tempfile
    _tempfile.tempdir = str(TMP_PROCESS_DIR)
except Exception:
    pass

for d in [UPLOAD_DIR, OUTPUT_DIR, CACHE_DIR, LOGS_DIR, PREVIEW_DIR, TEMP_DIR,
          PROJECTS_DIR, STT_CACHE_DIR, TRANSLATE_CACHE_DIR, TTS_CACHE_DIR,
          OCR_CACHE_DIR, RENDER_CACHE_DIR, AUDIO_CACHE_DIR, VIDEO_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
BACKEND_DIR.mkdir(parents=True, exist_ok=True)
if not DB_PATH.exists():
    DB_PATH.write_text("{}", encoding="utf-8")

# ============================================================
# FFMPEG & LOGGING
# ============================================================
_ffmpeg_env = os.getenv("FFMPEG_PATH", "").strip()
_ffprobe_env = os.getenv("FFPROBE_PATH", "").strip()
_ffmpeg_candidates = [_ffmpeg_env, shutil.which("ffmpeg"), r"C:\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"]
_ffprobe_candidates = [_ffprobe_env, shutil.which("ffprobe"), r"C:\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe"]
FFMPEG_PATH = next((x for x in _ffmpeg_candidates if x and Path(x).exists()), next((x for x in _ffmpeg_candidates if x), "ffmpeg"))
FFPROBE_PATH = next((x for x in _ffprobe_candidates if x and Path(x).exists()), next((x for x in _ffprobe_candidates if x), "ffprobe"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "app.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================
# PROVIDER OPTIONS — đúng với giao diện Lingvo
# STT: Groq + Gemini | Translation: Groq + Gemini | TTS: xKiro + Edge TTS
# ============================================================
PROVIDER_OPTIONS = {
    "stt": [
        {"id": "groq", "name": "Groq Whisper", "type": "free", "description": "Nhanh, chính xác"},
        {"id": "gemini", "name": "Gemini STT", "type": "free", "description": "Google Gemini"},
    ],
    "translate": [
        {"id": "groq", "name": "Groq LLM", "type": "free", "description": "Nhanh, đúng ngữ cảnh"},
        {"id": "xkira", "name": "xKiro Qwen", "type": "free", "description": "Qwen models qua xKiro"},
        {"id": "gemini", "name": "Gemini LLM", "type": "free", "description": "Google Gemini"},
    ],
    "ocr": [
        {"id": "local", "name": "PP-OCRv5 Mobile · High-Recall V2", "type": "free", "description": "Local OCR · adaptive sampling · frame-accurate timeline"},
    ],
    "tts": [
        {"id": "xkira_tts", "name": "xKiro TTS", "type": "free", "description": "xKiro AI voices"},
        {"id": "edge_tts", "name": "Microsoft Edge TTS", "type": "free", "description": "Edge neural voices"},
    ],
}

# ============================================================
# GROQ MODELS
# FIX: llama-3.1-8b-instant đã bị Groq gỡ (404) -> dùng model còn sống
# ============================================================
GROQ_TRANSLATE_MODEL = os.getenv("GROQ_TRANSLATE_MODEL", "openai/gpt-oss-120b")
MAX_TRANSLATE_WORKERS = int(os.getenv("MAX_TRANSLATE_WORKERS", "10"))
# Render/TTS worker setting consumed by main.py and pipeline.tts.
MAX_TTS_WORKERS = max(1, min(16, int(os.getenv("MAX_TTS_WORKERS", "8"))))

GROQ_TRANSLATE_FALLBACK_MODELS = [
    x.strip() for x in os.getenv(
        "GROQ_TRANSLATE_FALLBACK_MODELS",
        "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b"
    ).split(",") if x.strip()
]
DEFAULT_PROVIDERS = {
    "stt_provider": "groq",
    "translate_provider": "groq",
    "tts_provider": "xkira_tts",
}

# Per-stage model selection. Empty means provider default / live catalog default.
DEFAULT_MODELS = {
    "stt": os.getenv("STT_MODEL", ""),
    "ocr": os.getenv("OCR_MODEL", "PP-OCRv5-mobile"),
    "translate": os.getenv("TRANSLATE_MODEL", ""),
    "tts": os.getenv("TTS_MODEL", "xkiro-voice"),
    "full": os.getenv("FULL_MODEL", ""),
}

# ============================================================
# XKIRO TTS
# FIX: 8 tài khoản x 3 concurrent/account = 24 concurrent tổng
# ============================================================
XKIRA_API_BASE = os.getenv("XKIRA_API_BASE", "https://api.xkiro.com/v1").rstrip("/")
XKIRA_TTS_MODEL = os.getenv("XKIRA_TTS_MODEL", "xkiro-voice")
XKIRA_CONCURRENT_PER_ACCOUNT = max(1, min(2, int(os.getenv("XKIRA_CONCURRENT_PER_ACCOUNT", "1"))))
XKIRA_MAX_CONCURRENT = int(os.getenv("XKIRA_MAX_CONCURRENT", "10"))
XKIRA_USE_PAID_QWEN = os.getenv("XKIRA_USE_PAID_QWEN", "true").strip().lower() == "true"
XKIRA_MODEL_CACHE_SECONDS = int(os.getenv("XKIRA_MODEL_CACHE_SECONDS", "300"))

# ============================================================
# VOICE MAPS - ID THẬT của xKiro
# ============================================================
XKIRA_VOICE_MAP = {}
DEFAULT_XKIRA_VOICE = ""

# Google TTS Voice Map (main.py cần để resolve voice Google)
GOOGLE_TTS_VOICE_MAP = {
    "vi-VN-Standard-A": "vi-VN-Standard-A",
    "vi-VN-Standard-B": "vi-VN-Standard-B",
    "vi-VN-Standard-C": "vi-VN-Standard-C",
    "vi-VN-Standard-D": "vi-VN-Standard-D",
    "Google Tiếng Việt (Standard A)": "vi-VN-Standard-A",
    "Google Tiếng Việt (Standard B)": "vi-VN-Standard-B",
}
DEFAULT_GOOGLE_TTS_VOICE = "vi-VN-Standard-A"

EDGE_TTS_VOICE_MAP = {
    "vi-VN-HoaiMyNeural": "vi-VN-HoaiMyNeural",
    "vi-VN-NamMinhNeural": "vi-VN-NamMinhNeural",
    "Hoài My - Tự nhiên": "vi-VN-HoaiMyNeural",
    "Nam Minh - Tự nhiên": "vi-VN-NamMinhNeural",
    "Default": "vi-VN-NamMinhNeural",
}
DEFAULT_EDGE_VOICE = "vi-VN-NamMinhNeural"

# URL import / OCR / analysis
URL_PREVIEW_TIMEOUT = int(os.getenv("URL_PREVIEW_TIMEOUT", "45"))
URL_DOWNLOAD_TIMEOUT = int(os.getenv("URL_DOWNLOAD_TIMEOUT", "3600"))
# ============================================================
# ============================================================
# VIDEO OCR V2 — HIGH RECALL / FRAME ACCURATE
# OCR đọc pixel video; không dùng STT/audio/translation để tạo text hoặc timeline.
# Chính sách mặc định: MAXIMUM RECALL. Confidence thấp -> verification queue, không drop.
# ============================================================
OCR_ENGINE = "video_ocr_v2"
OCR_MODE = os.getenv("OCR_V2_MODE", "HIGH_RECALL").strip().upper() or "HIGH_RECALL"
OCR_SCAN_FPS = max(15.0, min(24.0, float(os.getenv("OCR_V2_SCAN_FPS", "20"))))
OCR_SCAN_WIDTH = max(224, int(os.getenv("OCR_V2_SCAN_WIDTH", "224")))
OCR_SCAN_WORKERS = max(1, min(4, int(os.getenv("OCR_V2_SCAN_WORKERS", "1"))))
OCR_RECOVERY_SECONDS = max(2.0, float(os.getenv("OCR_V2_RECOVERY_SECONDS", "8.0")))
OCR_BURST_FPS = max(8.0, min(30.0, float(os.getenv("OCR_V2_BURST_FPS", "12"))))
OCR_BATCH_SIZE = max(1, min(64, int(os.getenv("OCR_V2_BATCH_SIZE", "16"))))
OCR_BOUNDARY_PROBE_SECONDS = max(0.10, min(0.50, float(os.getenv("OCR_V2_BOUNDARY_PROBE_SECONDS", "0.15"))))
OCR_INTERIOR_FPS = max(0.5, min(4.0, float(os.getenv("OCR_V2_INTERIOR_FPS", "1"))))
OCR_GLOBAL_TEXTLIKE_THRESHOLD = max(0.002, float(os.getenv("OCR_V2_GLOBAL_TEXTLIKE_THRESHOLD", "0.005")))
OCR_GLOBAL_LINE_DENSITY_THRESHOLD = max(0.02, float(os.getenv("OCR_V2_GLOBAL_LINE_DENSITY_THRESHOLD", "0.05")))
OCR_MAX_TRACK_GAP = max(0.6, float(os.getenv("OCR_V2_MAX_TRACK_GAP", "1.10")))
OCR_TRACK_SIMILARITY = float(os.getenv("OCR_V2_TRACK_SIMILARITY", "0.50"))
OCR_DIFF_THRESHOLD = float(os.getenv("OCR_V2_DIFF_THRESHOLD", "0.018"))
OCR_TEXTLIKE_THRESHOLD = float(os.getenv("OCR_V2_TEXTLIKE_THRESHOLD", "0.012"))
OCR_BOUNDARY_MAX_PROBES = max(6, min(40, int(os.getenv("OCR_V2_BOUNDARY_MAX_PROBES", "18"))))
OCR_BOUNDARY_PIXEL_THRESHOLD = float(os.getenv("OCR_V2_BOUNDARY_PIXEL_THRESHOLD", "0.08"))
OCR_BOUNDARY_COLOR_TOLERANCE = float(os.getenv("OCR_V2_BOUNDARY_COLOR_TOLERANCE", "42"))
OCR_REQUIRE_GPU_FOR_TARGET = os.getenv("OCR_REQUIRE_GPU_FOR_TARGET", "1").strip().lower() not in {"0","false","no","off"}
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "auto").strip().lower()
OCR_BACKEND = os.getenv("OCR_BACKEND", "auto").strip().lower() or "auto"
OCR_GPU_ID = max(0, int(os.getenv("OCR_GPU_ID", "0")))
OCR_HWACCEL = os.getenv("OCR_V2_HWACCEL", "auto").strip().lower() or "auto"
OCR_CPU_THREADS = max(1, min(16, int(os.getenv("OCR_CPU_THREADS", "4"))))
OCR_RAPID_WORKERS = max(1, min(4, int(os.getenv("OCR_RAPID_WORKERS", "3"))))
PP_OCR_LANG = os.getenv("PP_OCR_LANG", "auto").strip().lower() or "auto"
PP_OCR_MODEL_DIR = os.getenv("PP_OCR_MODEL_DIR", "").strip()
OCR_CPU_TARGET = float(os.getenv("OCR_CPU_TARGET", "75"))
OCR_PROGRESS_LOG_INTERVAL = max(0.25, float(os.getenv("OCR_PROGRESS_LOG_INTERVAL", "1.0")))
OCR_JOB_CACHE = os.getenv("OCR_V2_JOB_CACHE", "1").strip().lower() not in {"0", "false", "no"}
# API KEYS
# ============================================================
def get_xkira_api_keys() -> List[str]:
    """Return up to 10 xKiro keys, including the legacy UI alias XKIRA.

    The Settings UI historically saved provider ``xkira`` as environment variable
    ``XKIRA``. OCR expects the numbered pool, so accept that legacy alias and
    treat it as key #1 for backward compatibility.
    """
    keys = []
    seen = set()
    for i in range(1, 9):
        value = os.getenv(f"XKIRA_API_KEY_{i}")
        value = value.strip() if value else ""
        if value and value not in seen:
            keys.append(value); seen.add(value)
    for env_name in ("XKIRA_API_KEY", "XKIRA"):
        value = os.getenv(env_name)
        value = value.strip() if value else ""
        if value and value not in seen:
            keys.append(value); seen.add(value)
    return keys


def get_groq_api_keys() -> List[str]:
    keys = []
    for i in range(1, 17):
        value = os.getenv(f"GROQ_API_KEY_{i}")
        if value and value.strip():
            keys.append(value.strip())
    if not keys:
        value = os.getenv("GROQ_API_KEY")
        if value and value.strip():
            keys.append(value.strip())
    return keys


def get_api_keys() -> Dict[str, Optional[str]]:
    keys = {
        "google_tts": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "gemini": os.getenv("GEMINI_API_KEY"),
    }
    xkira_keys = get_xkira_api_keys()
    if xkira_keys:
        keys["xkira"] = xkira_keys[0]
        keys["xkira_tts"] = xkira_keys[0]
        keys["xkira_keys"] = xkira_keys
    groq_keys = get_groq_api_keys()
    if groq_keys:
        keys["groq"] = groq_keys[0]
        keys["groq_keys"] = groq_keys
    return {k: v for k, v in keys.items() if v}


def save_api_keys(keys: Dict[str, str]) -> Dict[str, str]:
    saved = {}
    for key, value in keys.items():
        if not value:
            continue
        raw_key = str(key or "").strip().lower()
        env_key = str(key or "").upper()
        # Generic xKiro key pool is used by translation/TTS; OCR V2 is local-only.
        if raw_key == "xkira":
            env_key = "XKIRA_API_KEY_1"
        elif raw_key.startswith("xkira_") and raw_key.rsplit("_", 1)[-1].isdigit():
            suffix = raw_key.rsplit("_", 1)[-1]
            if suffix.isdigit() and 1 <= int(suffix) <= 8:
                env_key = f"XKIRA_API_KEY_{int(suffix)}"
        os.environ[env_key] = str(value).strip()
        saved[key] = str(value).strip()
    return saved


def masked(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(masked(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {masked(v)}" for k, v in value.items())
    value = str(value).strip()
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]
# V19: TTS must fully fit the immutable source subtitle slot.
TTS_HARD_FIT = os.getenv("TTS_HARD_FIT", "1") == "1"
# If xKiro TTS fails for a segment, retry that segment through Edge TTS.
TTS_FALLBACK_PROVIDER = os.getenv("TTS_FALLBACK_PROVIDER", "edge_tts").strip().lower()
TTS_MAX_SPEEDUP_HARD = float(os.getenv("TTS_MAX_SPEEDUP_HARD", "4.0"))

# Groq Whisper proactive pacing: 18.75 requests/minute, below 20 RPM base.
STT_GLOBAL_INTERVAL = float(os.getenv("STT_GLOBAL_INTERVAL", "3.05"))

# FIX TIMELINE #11: gộp cue trùng KHÔNG được dịch mốc thời gian.
# Chỉ xóa cue khi văn bản gần như trùng khít (>= ngưỡng) VÀ chồng nhau nhiều.
STT_SAFE_DEDUPE = os.getenv("STT_SAFE_DEDUPE", "true").lower() == "true"
STT_DEDUPE_SIMILARITY = float(os.getenv("STT_DEDUPE_SIMILARITY", "0.95"))
STT_DEDUPE_MIN_OVERLAP = float(os.getenv("STT_DEDUPE_MIN_OVERLAP", "0.60"))
# V18: strict word-based cue rebuild.
STT_FINAL_CUE_MAX_DURATION = float(os.getenv("STT_FINAL_CUE_MAX_DURATION", "4.2"))
STT_FINAL_CUE_MAX_CHARS = int(os.getenv("STT_FINAL_CUE_MAX_CHARS", "42"))
STT_FINAL_CUE_PAUSE_CUT = float(os.getenv("STT_FINAL_CUE_PAUSE_CUT", "0.22"))
STT_FINAL_CUE_MIN_CHARS_FOR_PAUSE = int(os.getenv("STT_FINAL_CUE_MIN_CHARS_FOR_PAUSE", "4"))
# V19: không cho cue cực ngắn trở thành slot TTS bất khả thi; chỉ merge khi rất gần nhau.
STT_MICRO_SEGMENT_MAX_DURATION = float(os.getenv("STT_MICRO_SEGMENT_MAX_DURATION", "0.30"))
STT_MICRO_SEGMENT_MAX_GAP = float(os.getenv("STT_MICRO_SEGMENT_MAX_GAP", "0.15"))
STT_MICRO_SEGMENT_MAX_COMBINED = float(os.getenv("STT_MICRO_SEGMENT_MAX_COMBINED", "4.20"))
# V18: strict named-entity preservation.
TRANSLATION_STRICT_NAME_PRESERVE = os.getenv("TRANSLATION_STRICT_NAME_PRESERVE", "true").strip().lower() == "true"
TRANSLATION_PROTECT_LATIN_TERMS = os.getenv("TRANSLATION_PROTECT_LATIN_TERMS", "true").strip().lower() == "true"
TRANSLATION_AUTO_ENTITY_AUDIT = os.getenv("TRANSLATION_AUTO_ENTITY_AUDIT", "true").strip().lower() == "true"
TRANSLATION_ENTITY_AUDIT_BATCH = int(os.getenv("TRANSLATION_ENTITY_AUDIT_BATCH", "60"))
TRANSLATION_ENTITY_AUDIT_MAX_CJK_LEN = int(os.getenv("TRANSLATION_ENTITY_AUDIT_MAX_CJK_LEN", "8"))
TRANSLATION_TERM_REPAIR_SINGLE = os.getenv("TRANSLATION_TERM_REPAIR_SINGLE", "true").strip().lower() == "true"

# VideoOCR V4: recognition-only cadence inside active subtitle windows.
OCR_RECOGNITION_SAMPLE_STEP = max(0.35, float(os.getenv("OCR_RECOGNITION_SAMPLE_STEP", "0.50")))
