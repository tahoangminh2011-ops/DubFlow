"""pipeline/tts.py - Xkira TTS robust, no-provider-fallback.
Duration fitting NEVER slows audio down. Speed-up capped at TTS_MAX_SPEEDUP (provider supports up to 4x).
Missing time padded with silence. Fixes slow/muffled/overlapping voice.
"""
import asyncio
import base64
import html
import logging
import random
import re
import subprocess
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import requests
import config

logger = logging.getLogger(__name__)

MAX_WORKERS = max(1, int(getattr(config, "MAX_TTS_WORKERS", 8)))
REQUEST_TIMEOUT = max(10, int(getattr(config, "TTS_TIMEOUT", 60)))
RETRY_COUNT = max(1, int(getattr(config, "TTS_RETRY_COUNT", 5)))
BACKOFF_BASE = max(0.25, float(getattr(config, "TTS_BACKOFF_BASE", 1)))
XKIRA_API_BASE = str(getattr(config, "XKIRA_API_BASE", "https://api.xkiro.com/v1")).rstrip("/")
XKIRA_TTS_URL = XKIRA_API_BASE + "/audio/speech"
XKIRA_TTS_MODEL = str(getattr(config, "XKIRA_TTS_MODEL", "xkiro-voice"))
DURATION_FIT_ENABLED = bool(getattr(config, "TTS_FIT_DURATION", True))
DURATION_FIT_TOLERANCE = float(getattr(config, "TTS_DURATION_TOLERANCE", 0.15))
TTS_MAX_SPEEDUP = min(1.60, max(1.05, float(getattr(config, "TTS_MAX_SPEEDUP", 1.60))))
TTS_HARD_FIT = bool(getattr(config, "TTS_HARD_FIT", True))
TTS_MAX_SPEEDUP_HARD = min(1.60, max(TTS_MAX_SPEEDUP, float(getattr(config, "TTS_MAX_SPEEDUP_HARD", TTS_MAX_SPEEDUP))))
TTS_MIN_SPEEDUP = 1.0
SLOT_TOLERANCE_MAX = float(getattr(config, "TTS_SLOT_TOLERANCE_MAX", 0.12))
SLOT_TOLERANCE_MIN = float(getattr(config, "TTS_SLOT_TOLERANCE_MIN", 0.03))
CACHE_POLICY = "V33_TTS_BATCH50_CJK_SAFE"
TTS_BATCH_SIZE = max(1, min(50, int(getattr(config, "TTS_BATCH_SIZE", 50))))


def slot_tolerance(slot: float) -> float:
    """FIX TIMELINE #6: mức cho phép tràn slot, tính theo GIÂY chứ không theo %.

    Công thức cũ `max(0.05, 0.15 * slot)` cho slot 6s được phép chứa 6.9s tiếng
    nói, tràn 0.9s đè sang câu kế tiếp; nhiều câu liên tiếp như vậy khiến tiếng
    lồng nghe như trễ dần so với hình. Nay trần tuyệt đối là SLOT_TOLERANCE_MAX,
    nên slot càng dài cũng không được tràn thêm.
    """
    try:
        slot = float(slot or 0.0)
    except (TypeError, ValueError):
        slot = 0.0
    if slot <= 0:
        return SLOT_TOLERANCE_MIN
    return max(SLOT_TOLERANCE_MIN, min(SLOT_TOLERANCE_MAX, DURATION_FIT_TOLERANCE * slot))
XKIRA_CONCURRENT_PER_ACCOUNT = max(1, int(getattr(config, "XKIRA_CONCURRENT_PER_ACCOUNT", 1)))

XKIRA_VOICE_MAP = {}

_xkira_key_index = 0
_xkira_key_lock = threading.Lock()
_xkira_key_fail_count: Dict[str, int] = {}
_account_pool_lock = threading.RLock()
_account_state: Dict[str, Dict[str, Any]] = {}
_xkira_voice_cache: List[Dict[str, Any]] = []
_xkira_voice_cache_at = 0.0
_xkira_voice_cache_lock = threading.Lock()
PREFERRED_VI_VOICE_NAMES = [
    "Sweet Female (Vietnamese)",
    "Confident Male (Vietnamese)",
    "Gentle Female (Vietnamese)",
    "Standard Female (Vietnamese)",
]

# xKiro account scheduler state is deliberately kept in-process.
# Each account has its own concurrency counter and cooldown. Batch TTS never statically
# maps segment N -> key N; it selects whichever account is currently available.


def _get_live_xkira_voices(force: bool = False) -> List[Dict[str, Any]]:
    """Return ONLY exact named Vietnamese xKiro voices from the live public catalog."""
    global _xkira_voice_cache, _xkira_voice_cache_at
    now = time.time()
    with _xkira_voice_cache_lock:
        if not force and _xkira_voice_cache and now - _xkira_voice_cache_at < 300:
            return list(_xkira_voice_cache)
    found: List[Dict[str, Any]] = []
    for wanted in PREFERRED_VI_VOICE_NAMES:
        candidates: List[Dict[str, Any]] = []
        queries = [
            {"q": wanted, "limit": 20},
            {"q": wanted.replace(" (Vietnamese)", ""), "limit": 20},
            {"languageKey": "vi", "limit": 100},
        ]
        for params in queries:
            try:
                resp = requests.get(XKIRA_API_BASE + "/audio/voices", params=params, timeout=12)
                if resp.status_code != 200:
                    continue
                payload = resp.json() if resp.content else {}
                arr = payload.get("voices", []) if isinstance(payload, dict) else []
                if isinstance(arr, list):
                    candidates.extend(v for v in arr if isinstance(v, dict))
                if candidates:
                    # exact match can be decided below; no need to hammer the public catalog.
                    exact = next((v for v in candidates if str(v.get("name") or "").strip().casefold() == wanted.casefold() and
                                  (str(v.get("languageKey") or "").strip().casefold() == "vi" or
                                   str(v.get("locale") or "").strip().casefold().startswith("vi") or
                                   "vietnamese" in str(v.get("name") or "").casefold())), None)
                    if exact and exact.get("id"):
                        found.append(exact)
                        break
            except Exception as exc:
                logger.debug("xKiro voice catalog query %s lỗi: %s", wanted, exc)
    out: List[Dict[str, Any]] = []
    seen = set()
    for v in found:
        vid = str(v.get("id") or "").strip()
        if not vid or vid.casefold() in seen:
            continue
        seen.add(vid.casefold())
        out.append({
            "provider": "xkira_tts", "voice_id": vid, "id": vid,
            "name": str(v.get("name") or vid),
            "language": str(v.get("languageKey") or "vi"),
            "locale": str(v.get("locale") or ""),
            "gender": str(v.get("gender") or "").lower(),
            "type": "vip" if v.get("isVip") else "free",
            "isVip": bool(v.get("isVip")),
        })
    with _xkira_voice_cache_lock:
        _xkira_voice_cache = out
        _xkira_voice_cache_at = now
    if len(out) < 4:
        logger.warning("xKiro chỉ tìm thấy %d/4 giọng Việt yêu cầu; không tự thay thế voice.", len(out))
    else:
        logger.info("xKiro catalog: đã xác nhận đúng 4 giọng Việt yêu cầu")
    return list(out)


def _get_xkira_account_state(key: str) -> Dict[str, Any]:
    with _account_pool_lock:
        state = _account_state.get(key)
        if state is None:
            state = {"active": 0, "cooldown_until": 0.0, "disabled_until": 0.0, "fail_count": 0}
            _account_state[key] = state
        return state


def _acquire_xkira_account(preferred_key: Optional[str] = None, timeout: float = 120.0) -> str:
    keys = _get_xkira_keys()
    if not keys:
        raise RuntimeError("Xkira TTS: thiếu API key")
    deadline = time.time() + max(1.0, float(timeout))
    while time.time() < deadline:
        now = time.time()
        with _account_pool_lock:
            ordered = list(keys)
            if preferred_key in ordered:
                ordered.remove(preferred_key)
                ordered.insert(0, preferred_key)
            best = None
            best_score = None
            future_wait = None
            for key in ordered:
                st = _get_xkira_account_state(key)
                if st["disabled_until"] > now:
                    future_wait = min(future_wait, st["disabled_until"] - now) if future_wait is not None else st["disabled_until"] - now
                    continue
                cd = max(0.0, float(st["cooldown_until"]) - now)
                if cd > 0:
                    future_wait = min(future_wait, cd) if future_wait is not None else cd
                    continue
                if int(st["active"]) >= XKIRA_CONCURRENT_PER_ACCOUNT:
                    continue
                score = (int(st["active"]), float(st["cooldown_until"]))
                if best is None or score < best_score:
                    best, best_score = key, score
            if best is not None:
                st = _get_xkira_account_state(best)
                st["active"] += 1
                return best
        sleep_for = min(0.20, max(0.02, future_wait or 0.05))
        time.sleep(sleep_for)
    raise TimeoutError("Xkiro scheduler: không có account TTS rảnh trong thời gian chờ")


def _release_xkira_account(key: str, cooldown: float = 0.0, disabled: float = 0.0, failed: bool = False) -> None:
    if not key:
        return
    with _account_pool_lock:
        st = _get_xkira_account_state(key)
        st["active"] = max(0, int(st["active"]) - 1)
        if cooldown > 0:
            st["cooldown_until"] = max(float(st["cooldown_until"]), time.time() + float(cooldown))
        if disabled > 0:
            st["disabled_until"] = max(float(st["disabled_until"]), time.time() + float(disabled))
        if failed:
            st["fail_count"] = int(st["fail_count"]) + 1
        else:
            st["fail_count"] = 0


def _mark_xkira_key_success(key: str) -> None:
    with _account_pool_lock:
        st = _get_xkira_account_state(key)
        st["fail_count"] = 0
        st["cooldown_until"] = 0.0


def _mark_xkira_key_fail(key: str, cooldown: float = 0.0) -> None:
    with _account_pool_lock:
        st = _get_xkira_account_state(key)
        st["fail_count"] = int(st["fail_count"]) + 1
        if cooldown > 0:
            st["cooldown_until"] = max(float(st["cooldown_until"]), time.time() + cooldown)


def _get_xkira_keys() -> List[str]:
    keys: List[str] = []
    try:
        getter = getattr(config, "get_xkira_api_keys", None)
        if getter:
            keys = list(getter() or [])
    except Exception:
        keys = []
    if not keys:
        try:
            api_keys = config.get_api_keys()
            keys = api_keys.get("xkira_keys") or []
            if not keys and api_keys.get("xkira"):
                keys = [api_keys["xkira"]]
            if not keys and api_keys.get("xkira_tts"):
                keys = [api_keys["xkira_tts"]]
        except Exception:
            keys = []
    return [str(k).strip() for k in keys if str(k).strip()]


def _get_next_xkira_key() -> Optional[str]:
    global _xkira_key_index
    keys = _get_xkira_keys()
    if not keys:
        return None
    with _xkira_key_lock:
        key = keys[_xkira_key_index % len(keys)]
        _xkira_key_index += 1
    return key


def _mark_xkira_key_success(key: str) -> None:
    with _xkira_key_lock:
        _xkira_key_fail_count[key] = 0


def _mark_xkira_key_fail(key: str) -> None:
    with _xkira_key_lock:
        _xkira_key_fail_count[key] = _xkira_key_fail_count.get(key, 0) + 1


def _retry_after_seconds(resp, default: float) -> float:
    try:
        value = float(resp.headers.get("Retry-After") or "")
        if value > 0:
            return min(300.0, value) + random.uniform(0.05, 0.25)
    except (TypeError, ValueError):
        pass
    return min(30.0, max(0.5, float(default))) + random.uniform(0.05, 0.25)


def _sanitize_for_tts(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"<[^>]*>", " ", value)
    value = re.sub(r"`.*?`", " ", value, flags=re.S)
    value = re.sub(r"[*_~`]+", " ", value)
    value = "".join(" " if unicodedata.category(ch) in {"Cc", "Cf"} and ch not in "\n\t" else ch for ch in value)
    allowed_punct = set('.,!?;:\'"()-/')
    out = []
    for ch in value:
        cat = unicodedata.category(ch)
        if ch.isalnum() or ch.isspace() or ch in allowed_punct:
            out.append(ch)
        elif cat.startswith("P") and ch in allowed_punct:
            out.append(ch)
        else:
            out.append(" ")
    value = re.sub(r"\s+", " ", "".join(out)).strip()
    return value


def _sanitize_for_tts_ultra(text: str) -> str:
    value = _sanitize_for_tts(text)
    value = re.sub(r"[;:/()\"'-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _sanitize_for_tts_plain(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"<[^>]*>", " ", value)
    value = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in value)
    return re.sub(r"\s+", " ", value).strip()


def _split_text(text: str, max_chars: int = 1800) -> List[str]:
    text = _sanitize_for_tts_ultra(text)
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    cur = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not cur else f"{cur} {sentence}"
        if len(candidate) <= max_chars:
            cur = candidate
            continue
        if cur:
            chunks.append(cur)
            cur = ""
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        cur = sentence
    if cur:
        chunks.append(cur)
    return chunks


def _valid_audio(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 500:
            return False
        return get_audio_duration(str(path)) > 0.05
    except OSError:
        return False


def get_audio_duration(audio_path: str) -> float:
    try:
        result = subprocess.run(
            [config.FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return max(0.0, float(result.stdout.strip()))
    except Exception:
        pass
    return 0.0


EDGE_VOICE_MAP = dict(getattr(config, "EDGE_TTS_VOICE_MAP", {}) or {
    "vi-VN-HoaiMyNeural": "vi-VN-HoaiMyNeural",
    "vi-VN-NamMinhNeural": "vi-VN-NamMinhNeural",
    "Hoài My - Tự nhiên": "vi-VN-HoaiMyNeural",
    "Nam Minh - Tự nhiên": "vi-VN-NamMinhNeural",
})

def _resolve_voice_id(voice: str, provider: str) -> str:
    voice_str = str(voice or "").strip()
    if provider == "xkira_tts":
        live = _get_live_xkira_voices()
        ids = {str(v.get("voice_id")) for v in live}
        if voice_str in ids:
            return voice_str
        for v in live:
            if str(v.get("name") or "").strip().casefold() == voice_str.casefold():
                return str(v["voice_id"])
        aliases = {
            "female_sweet": "Sweet Female (Vietnamese)",
            "sweet-female-vietnamese": "Sweet Female (Vietnamese)",
            "female_gentle": "Gentle Female (Vietnamese)",
            "gentle-female-vietnamese": "Gentle Female (Vietnamese)",
            "female_standard": "Standard Female (Vietnamese)",
            "standard-female-vietnamese": "Standard Female (Vietnamese)",
            "male_confident": "Confident Male (Vietnamese)",
            "confident-male-vietnamese": "Confident Male (Vietnamese)",
        }
        wanted = aliases.get(voice_str.casefold())
        if wanted:
            for v in live:
                if str(v.get("name") or "").strip().casefold() == wanted.casefold():
                    return str(v["voice_id"])
        raise RuntimeError(f"xKiro voice không hợp lệ hoặc chưa có trong catalog Việt: {voice_str or '(trống)'}")
    if provider == "edge_tts":
        return EDGE_VOICE_MAP.get(voice_str, voice_str or getattr(config, "DEFAULT_EDGE_VOICE", "vi-VN-NamMinhNeural"))
    return voice_str


def _atempo_filter_chain(factor: float) -> str:
    if factor <= 0:
        return "atempo=1.0"
    stages = []
    f = factor
    while f > 2.0:
        stages.append(2.0)
        f /= 2.0
    while f < 0.5:
        stages.append(0.5)
        f /= 0.5
    stages.append(f)
    return ",".join(f"atempo={s:.4f}" for s in stages)


def _fit_audio_to_duration(audio_path: Path, target_duration: float) -> float:
    """Hard-fit TTS into the immutable source slot without deleting words.

    If speech is longer than the slot, time-compress the whole audio (atempo),
    never trim the tail. This prevents the next segment from starting while the
    previous one is still speaking and prevents render from cutting final words.
    """
    current = get_audio_duration(str(audio_path))
    if current <= 0 or target_duration <= 0:
        return current
    tolerance = slot_tolerance(target_duration)
    if current <= target_duration + tolerance:
        # Keep a small amount of trailing silence so playback has a deterministic end.
        if current < target_duration - 0.05:
            pad = target_duration - current
            tmp = audio_path.with_suffix('.pad.mp3')
            cmd=[config.FFMPEG_PATH,'-y','-i',str(audio_path),'-af',f'apad=pad_dur={pad:.3f}',
                 '-vn','-ac','1','-ar','44100','-c:a','libmp3lame','-b:a','160k',str(tmp)]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
            if r.returncode==0 and _valid_audio(tmp):
                tmp.replace(audio_path); return get_audio_duration(str(audio_path))
            tmp.unlink(missing_ok=True)
        return current

    allowed = max(1.05, float(TTS_MAX_SPEEDUP_HARD if TTS_HARD_FIT else TTS_MAX_SPEEDUP))
    factor = min(current / target_duration, allowed)
    if factor <= 1.001:
        return current
    tmp = audio_path.with_suffix('.fit.mp3')
    try:
        cmd=[config.FFMPEG_PATH,'-y','-i',str(audio_path),'-filter:a',_atempo_filter_chain(factor),
             '-vn','-ac','1','-ar','44100','-c:a','libmp3lame','-b:a','160k',str(tmp)]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        if r.returncode==0 and _valid_audio(tmp):
            tmp.replace(audio_path)
            fitted=get_audio_duration(str(audio_path))
            logger.info('[TTS-FIT] %s: %.2fs -> %.2fs | slot=%.2fs | speed=%.2fx',audio_path.name,current,fitted,target_duration,factor)
            return fitted
    finally:
        tmp.unlink(missing_ok=True)
    return current


def _concat_generated_parts(parts: List[Path], output_path: Path) -> Tuple[Path, float]:
    if not parts:
        raise RuntimeError("Xkira không tạo được audio chunk nào")
    if len(parts) == 1:
        return parts[0], get_audio_duration(str(parts[0]))
    list_file = output_path.with_suffix(".concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '" + str(p).replace("'", "'\\''") + "'\n")
    tmp = output_path.with_suffix(".joined.mp3")
    try:
        r = subprocess.run([config.FFMPEG_PATH, "-y", "-f", "concat", "-safe", "0",
                            "-i", str(list_file), "-c:a", "libmp3lame", "-b:a", "128k", str(tmp)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not _valid_audio(tmp):
            raise RuntimeError(f"FFmpeg concat TTS lỗi: {(r.stderr or '')[-500:]}")
        tmp.replace(output_path)
        return output_path, get_audio_duration(str(output_path))
    finally:
        list_file.unlink(missing_ok=True)
        for p in parts:
            if p != output_path:
                p.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)


def _cache_meta_path(audio_path: Path) -> Path:
    return audio_path.with_suffix('.meta.json')

def _cache_matches(audio_path: Path, text: str, voice: str, provider: str, rate: float, pitch: float) -> bool:
    try:
        m = _cache_meta_path(audio_path)
        if not audio_path.exists() or not m.exists():
            return False
        data = __import__('json').loads(m.read_text(encoding='utf-8'))
        return (
            data.get('policy') == CACHE_POLICY
            and data.get('text') == text
            and data.get('voice') == voice
            and data.get('provider') == provider
            and abs(float(data.get('rate', 1)) - float(rate)) < 1e-6
            and abs(float(data.get('pitch', 0)) - float(pitch)) < 1e-6
        )
    except Exception:
        return False

def _write_cache_meta(audio_path: Path, text: str, voice: str, provider: str, rate: float, pitch: float):
    try:
        data = {'policy': CACHE_POLICY, 'text': text, 'voice': voice, 'provider': provider,
                'rate': float(rate), 'pitch': float(pitch)}
        _cache_meta_path(audio_path).write_text(__import__('json').dumps(data, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass

class XKiroRateLimited(RuntimeError):
    def __init__(self, retry_after: float, message: str = "xKiro rate limited"):
        super().__init__(message)
        self.retry_after = max(0.1, float(retry_after))

class XKiroTransient(RuntimeError):
    pass

class XKiroAuthError(RuntimeError):
    pass

def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", str(text or "")))

def _contains_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff]", str(text or "")))

def _contains_korean(text: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7af]", str(text or "")))

def _edge_voice_for_text(text: str) -> str:
    # xKiro Vietnamese voices reject CJK input. Pick a matching Edge neural voice
    # for the deterministic fallback instead of retrying the same HTTP 400.
    if _contains_japanese(text):
        return "ja-JP-NanamiNeural"
    if _contains_korean(text):
        return "ko-KR-SunHiNeural"
    if _contains_cjk(text):
        return "zh-CN-XiaoxiaoNeural"
    if re.search(r"[\u0e00-\u0e7f]", str(text or "")):
        return "th-TH-PremwadeeNeural"
    if re.search(r"[\u0400-\u04ff]", str(text or "")):
        return "ru-RU-SvetlanaNeural"
    return getattr(config, "DEFAULT_EDGE_VOICE", "vi-VN-NamMinhNeural")

def generate_tts_for_segments(segments: List[Dict[str, Any]], translated_texts: List[str],
                              voice: str, provider: str = "xkira_tts",
                              api_key: Optional[str] = None, output_dir: Optional[Path] = None,
                              max_workers: int = MAX_WORKERS, rate: float = 1.0, pitch: float = 0.0,
                              progress_cb: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
    if len(segments) != len(translated_texts):
        raise ValueError(f"Số segments ({len(segments)}) và bản dịch ({len(translated_texts)}) không khớp")
    if not segments:
        return []
    if provider == "xkira_tts" and not _get_xkira_keys() and not api_key:
        raise RuntimeError("Xkira TTS: chưa cấu hình XKIRA_API_KEY_1...XKIRA_API_KEY_8")
    output_dir = Path(output_dir or (config.TTS_CACHE_DIR / "segments"))
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(max_workers or MAX_WORKERS), MAX_WORKERS))
    keys = _get_xkira_keys() if provider == "xkira_tts" else []
    if provider == "xkira_tts" and keys:
        workers = min(workers, len(keys) * XKIRA_CONCURRENT_PER_ACCOUNT)
    logger.info("TTS %d đoạn | provider=%s | workers=%d | accounts=%d | per_account=%d",
                len(segments), provider, workers, len(keys), XKIRA_CONCURRENT_PER_ACCOUNT)
    started = time.time()

    def generate_one(idx: int) -> Dict[str, Any]:
        raw = str(translated_texts[idx] or "").strip()
        if not raw:
            return {"index": idx, "text": raw, "audio": None, "duration": 0.0, "error": "Text rỗng"}
        output = output_dir / f"seg{idx:04d}.mp3"
        target = max(0.0, float(segments[idx].get("end", 0) or 0) - float(segments[idx].get("start", 0) or 0))
        # A Vietnamese xKiro voice cannot synthesize CJK. Do not send a known
        # invalid request to every account; use the matching Edge voice directly.
        if provider == "xkira_tts" and (_contains_cjk(raw) or _contains_japanese(raw) or _contains_korean(raw)):
            fb_voice = _edge_voice_for_text(raw)
            try:
                fb_audio, fb_duration = _generate_tts_with_provider(
                    text=raw, voice=fb_voice, provider="edge_tts", api_key=None,
                    output_path=output, rate=rate, pitch=pitch)
                _write_cache_meta(Path(fb_audio), raw, fb_voice, "edge_tts", rate, pitch)
                logger.warning("Segment %d: xKiro voice không hỗ trợ CJK -> Edge %s", idx + 1, fb_voice)
                return {"index": idx, "text": raw, "audio": str(fb_audio), "duration": fb_duration,
                        "slot_duration": target, "cached": False, "provider": "edge_tts",
                        "voice": fb_voice, "fallback": True, "error": "xKiro voice không hỗ trợ ký tự CJK"}
            except Exception as exc:
                logger.warning("Segment %d: Edge CJK fallback lỗi: %s", idx + 1, exc)
        if _valid_audio(output) and _cache_matches(output, raw, voice, provider, rate, pitch):
            dur = get_audio_duration(str(output))
            return {"index": idx, "text": raw, "audio": str(output), "duration": dur,
                    "slot_duration": target, "cached": True, "provider": provider,
                    "voice": voice, "fit_verified": False, "timing_policy": CACHE_POLICY}
        last_error = None
        # For xKiro, each attempt gets a freshly scheduled account. A 429 never gets retried
        # against the same account while it is cooling down.
        attempts = max(2, min(6, int(getattr(config, "TTS_RETRY_COUNT", 4)))) if provider == "xkira_tts" else 1
        for attempt in range(attempts):
            key = api_key if provider == "xkira_tts" and api_key else None
            acquired = None
            try:
                if provider == "xkira_tts":
                    acquired = _acquire_xkira_account(preferred_key=key, timeout=120)
                    key = acquired
                audio_path, duration = _generate_tts_with_provider(
                    text=raw, voice=voice, provider=provider, api_key=key,
                    output_path=output, rate=rate, pitch=pitch)
                # V32: TTS generation is content-only. Send the translated text once at the
                # requested/natural rate and keep the returned audio duration exactly.
                # Source STT timing is NOT a reason to request a faster second TTS reading.
                _write_cache_meta(Path(audio_path), raw, voice, provider, rate, pitch)
                return {"index": idx, "text": raw, "audio": str(audio_path), "duration": duration,
                        "slot_duration": target, "cached": False, "provider": provider, "voice": voice,
                        "fit_verified": False, "timing_policy": CACHE_POLICY}
            except XKiroRateLimited as exc:
                last_error = exc
                if acquired:
                    _release_xkira_account(acquired, cooldown=exc.retry_after, failed=True)
                    acquired = None
                if attempt + 1 < attempts:
                    continue
            except XKiroAuthError as exc:
                last_error = exc
                if acquired:
                    _release_xkira_account(acquired, disabled=60.0, failed=True)
                    acquired = None
                if attempt + 1 < attempts:
                    continue
            except XKiroTransient as exc:
                last_error = exc
                if acquired:
                    _release_xkira_account(acquired, cooldown=min(5.0, 0.5 * (2 ** attempt)), failed=True)
                    acquired = None
                if attempt + 1 < attempts:
                    time.sleep(min(5.0, 0.4 * (2 ** attempt)) + random.uniform(0.0, 0.2))
                    continue
            except Exception as exc:
                last_error = exc
                if acquired:
                    _release_xkira_account(acquired, failed=True)
                    acquired = None
                # 400/404/voice validation errors are deterministic and should not be hammered.
                break
            finally:
                if acquired:
                    _release_xkira_account(acquired)
        fallback = str(getattr(config, "TTS_FALLBACK_PROVIDER", "") or "").strip().lower()
        if provider == "xkira_tts" and fallback == "edge_tts" and last_error:
            try:
                fb_voice = _edge_voice_for_text(raw)
                fb_audio, fb_duration = _generate_tts_with_provider(
                    text=raw, voice=fb_voice, provider="edge_tts", api_key=None,
                    output_path=output.with_name(output.stem + ".fallback.mp3"), rate=rate, pitch=pitch)
                # Keep fallback speech natural; renderer controls placement.
                Path(fb_audio).replace(output)
                _write_cache_meta(output, raw, fb_voice, "edge_tts", rate, pitch)
                return {"index": idx, "text": raw, "audio": str(output), "duration": fb_duration,
                        "cached": False, "provider": "edge_tts", "voice": fb_voice, "fallback": True,
                        "error": str(last_error)}
            except Exception as fb_exc:
                last_error = RuntimeError(f"xKiro: {last_error}; Edge fallback: {fb_exc}")
        logger.error("Segment %d TTS lỗi: %s", idx, last_error or "không rõ")
        return {"index": idx, "text": raw, "audio": None, "duration": 0.0,
                "cached": False, "error": str(last_error or "TTS thất bại")}

    results: List[Optional[Dict[str, Any]]] = [None] * len(segments)
    completed = 0
    # TTS batch=50: keep at most 50 segment jobs in-flight per wave. This avoids
    # creating 1,448 futures at once while still using the full xKiro key pool.
    batch_ranges = [(start, min(len(segments), start + TTS_BATCH_SIZE)) for start in range(0, len(segments), TTS_BATCH_SIZE)]
    logger.info("TTS batch=%d | tổng batch=%d", TTS_BATCH_SIZE, len(batch_ranges))
    for batch_no, (b0, b1) in enumerate(batch_ranges, 1):
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"tts-b{batch_no}") as pool:
            future_map = {pool.submit(generate_one, i): i for i in range(b0, b1)}
            for fut in as_completed(future_map):
                idx = future_map[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    results[idx] = {"index": idx, "text": str(translated_texts[idx] or ""),
                                    "audio": None, "duration": 0.0, "error": str(exc)}
                completed += 1
                if progress_cb and (completed % 5 == 0 or completed == len(segments)):
                    try:
                        progress_cb(completed, len(segments))
                    except Exception:
                        pass
        logger.info("TTS batch %d/%d hoàn tất | %d-%d | tổng=%d/%d", batch_no, len(batch_ranges), b0 + 1, b1, completed, len(segments))
    final = [r or {"index": i, "text": str(translated_texts[i] or ""), "audio": None,
                    "duration": 0.0, "error": "Unknown"} for i, r in enumerate(results)]
    valid = sum(1 for r in final if r.get("audio") and _valid_audio(Path(str(r["audio"]))))
    failed = [r["index"] for r in final if not r.get("audio")]
    logger.info("TTS hoàn tất %d/%d trong %.2fs | dynamic-account scheduler",
                valid, len(segments), time.time() - started)
    if failed:
        logger.warning("TTS còn lỗi %d đoạn: %s", len(failed), failed[:20])
    return final


def _generate_tts_with_provider(text: str, voice: str, provider: str, api_key: Optional[str],
                                output_path: Path, rate: float = 1.0, pitch: float = 0.0) -> Tuple[Path, float]:
    funcs = {"xkira_tts": _generate_xkira_tts, "edge_tts": _generate_edge_tts}
    func = funcs.get(provider)
    if not func:
        raise ValueError(f"Provider {provider} không hỗ trợ")
    return func(text, voice, api_key, output_path, rate, pitch)


def _xkira_request(text: str, voice_id: str, api_key: str, output_path: Path, rate: float) -> Tuple[Path, float]:
    if not api_key:
        raise RuntimeError("xKiro TTS: thiếu account key")
    payload = {
        "model": XKIRA_TTS_MODEL, "input": text, "voice": voice_id,
        "response_format": "mp3", "speed": max(0.25, min(4.0, float(rate or 1.0))),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(XKIRA_TTS_URL, json=payload, headers=headers, timeout=(10, REQUEST_TIMEOUT))
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise XKiroTransient(str(exc)) from exc
    if resp.status_code == 200:
        if not resp.content:
            raise XKiroTransient("xKiro trả về audio rỗng")
        output_path.write_bytes(resp.content)
        if not _valid_audio(output_path):
            raise XKiroTransient("xKiro trả về file audio không hợp lệ")
        return output_path, get_audio_duration(str(output_path))
    if resp.status_code == 429:
        wait = _retry_after_seconds(resp, default=2.0)
        raise XKiroRateLimited(wait, f"HTTP 429; cooldown {wait:.1f}s")
    if resp.status_code in (401, 402, 403):
        raise XKiroAuthError(f"xKiro TTS HTTP {resp.status_code}: {resp.text[:500]}")
    if resp.status_code in (500, 502, 503, 504, 408):
        raise XKiroTransient(f"xKiro TTS HTTP {resp.status_code}: {resp.text[:300]}")
    if resp.status_code in (400, 404):
        # Deterministic request/voice/model problem; do not rotate through accounts and hammer the API.
        raise RuntimeError(f"xKiro TTS HTTP {resp.status_code}: {resp.text[:500]}")
    raise XKiroTransient(f"xKiro TTS HTTP {resp.status_code}: {resp.text[:300]}")


def _generate_xkira_tts(text, voice, api_key=None, output_path=None, rate=1.0, pitch=0.0):
    voice_id = _resolve_voice_id(voice, "xkira_tts")
    clean = _sanitize_for_tts(text)
    if not clean:
        raise RuntimeError("Text sau khi làm sạch rỗng")
    output_path = Path(output_path)
    attempts = [clean]
    ultra = _sanitize_for_tts_ultra(clean)
    if ultra and ultra not in attempts:
        attempts.append(ultra)
    plain = _sanitize_for_tts_plain(clean)
    if plain and plain not in attempts:
        attempts.append(plain)
    last_error = None
    for candidate in attempts:
        try:
            return _xkira_request(candidate, voice_id, api_key or _get_xkira_keys()[0], output_path, rate)
        except RuntimeError as exc:
            last_error = exc
            # Only try progressively safer text forms after HTTP 400.
            if not str(exc).startswith("xKiro TTS HTTP 400"):
                raise
    chunks = _split_text(clean, 1800)
    if len(chunks) > 1:
        parts: List[Path] = []
        try:
            for n, chunk in enumerate(chunks):
                part = output_path.with_name(f"{output_path.stem}.part{n:02d}.mp3")
                part_path, _ = _xkira_request(chunk, voice_id, api_key or _get_xkira_keys()[0], part, rate)
                parts.append(part_path)
            return _concat_generated_parts(parts, output_path)
        except Exception as exc:
            for p in parts:
                p.unlink(missing_ok=True)
            last_error = exc
    raise RuntimeError(str(last_error or "xKiro TTS thất bại"))


async def _edge_save_async(text, voice_id, rate_str, pitch_str, output_path):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice_id, rate=rate_str, pitch=pitch_str)
    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                f.write(chunk.get("data", b""))

def _generate_edge_tts(text, voice, api_key=None, output_path=None, rate=1.0, pitch=0.0):
    import asyncio
    voice_id = _resolve_voice_id(voice, "edge_tts")
    rate_str = f"{int(round((float(rate)-1)*100)):+d}%"
    pitch_str = f"{int(round(float(pitch))):+d}Hz"
    asyncio.run(_edge_save_async(text, voice_id, rate_str, pitch_str, output_path))
    if not _valid_audio(Path(output_path)):
        raise RuntimeError("Edge TTS tạo audio rỗng")
    return Path(output_path), get_audio_duration(str(output_path))

def get_available_voices(provider: Optional[str] = None) -> List[Dict[str, Any]]:
    voices: List[Dict[str, Any]] = []
    if not provider or provider == "xkira_tts":
        voices.extend(_get_live_xkira_voices())
    if not provider or provider == "edge_tts":
        voices.extend([
            {"provider": "edge_tts", "voice_id": "vi-VN-HoaiMyNeural", "name": "Hoài My — Tự nhiên (Edge TTS)", "language": "vi-VN", "gender": "female", "type": "free"},
            {"provider": "edge_tts", "voice_id": "vi-VN-NamMinhNeural", "name": "Nam Minh — Tự nhiên (Edge TTS)", "language": "vi-VN", "gender": "male", "type": "free"},
        ])
    return voices

def get_voice_by_id(voice_id: str, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return next((v for v in get_available_voices(provider) if v.get("voice_id") == voice_id), None)


def resolve_voice_name(voice_id: str, provider: str) -> str:
    v = get_voice_by_id(voice_id, provider)
    return v.get("name", voice_id) if v else voice_id


__all__ = [
    "generate_tts_for_segments", "get_available_voices", "get_voice_by_id",
    "resolve_voice_name", "_generate_tts_with_provider", "_generate_xkira_tts",
    "_resolve_voice_id", "get_audio_duration", "_generate_edge_tts",
]