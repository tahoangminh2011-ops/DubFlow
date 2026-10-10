"""pipeline/stt.py - STT chunk song song, fail-fast >20% lỗi."""
import os
import time
import subprocess
import socket
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import httpx
import config
from pipeline.stt_timing import align_stt_segments_to_video

logger = logging.getLogger(__name__)

MAX_WORKERS = int(getattr(config, 'MAX_STT_WORKERS', 15))
CHUNK_DURATION = int(getattr(config, 'STT_CHUNK_DURATION', 60))
REQUEST_TIMEOUT = int(getattr(config, 'STT_TIMEOUT', 120))
RETRY_COUNT = int(getattr(config, 'STT_RETRY_COUNT', 4))
RATE_LIMIT_WAIT = float(getattr(config, 'STT_BACKOFF_BASE', 5.0))
RATE_LIMIT_RETRY_COUNT = int(getattr(config, 'STT_429_RETRY_COUNT', 8))
NETWORK_RETRY_COUNT = max(2, int(getattr(config, 'STT_NETWORK_RETRY_COUNT', 3)))
NETWORK_BACKOFF_MAX = float(getattr(config, 'STT_NETWORK_BACKOFF_MAX', 8.0))
CONNECT_TIMEOUT = float(getattr(config, 'STT_CONNECT_TIMEOUT', 20.0))
GROQ_HOST = "api.groq.com"
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

_scheduler_next_request = 0.0
_scheduler_lock = threading.RLock()
_scheduler_block_until = 0.0

def _enforce_rate_limit(api_key: str):
    """Pace Groq STT requests with a global safety interval.

    Workers still prepare chunks/results concurrently. The request lane is paced
    conservatively so a burst of 2 workers does not immediately trip RPM limits.
    The interval is configurable via STT_GLOBAL_INTERVAL.
    """
    interval = float(getattr(config, "STT_GLOBAL_INTERVAL", 4.00))
    global _scheduler_next_request, _scheduler_block_until
    with _scheduler_lock:
        now = time.time()
        target = max(now, _scheduler_next_request, _scheduler_block_until)
        _scheduler_next_request = target + interval
    wait_needed = target - now
    if wait_needed > 0:
        time.sleep(wait_needed)

def _block_scheduler(seconds: float):
    global _scheduler_block_until
    with _scheduler_lock:
        _scheduler_block_until = max(_scheduler_block_until, time.time() + max(0.0, seconds))


def _get_stt_keys(api_key: Optional[str] = None) -> List[str]:
    keys: List[str] = []
    getter = getattr(config, "get_groq_api_keys", None)
    if getter:
        try:
            keys = list(getter() or [])
        except Exception:
            keys = []
    if not keys and api_key and str(api_key).strip():
        keys = [str(api_key).strip()]
    if not keys:
        single = getattr(config, "GROQ_API_KEY", "")
        if single and str(single).strip():
            keys = [str(single).strip()]
    keys = [k for k in keys if k and str(k).strip()]
    if keys:
        logger.info(f"STT: Đã tìm thấy {len(keys)} Groq API keys")
    return keys


def _check_groq_connectivity(log_only: bool = False) -> bool:
    try:
        socket.getaddrinfo(GROQ_HOST, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        if log_only:
            logger.warning(f"Groq DNS lỗi (bỏ qua): {e}")
            return False
        raise RuntimeError(f"Groq DNS lỗi: {e}")
    try:
        with socket.create_connection((GROQ_HOST, 443), timeout=CONNECT_TIMEOUT):
            pass
    except OSError as e:
        if log_only:
            logger.warning(f"Groq TCP/443 không kết nối được (bỏ qua): {e}")
            return False
        raise RuntimeError(f"Groq TCP/443 không kết nối được: {e}")
    return True



def _is_suspicious_segment(seg: Dict[str, Any]) -> bool:
    """Conservative quality gate: only flag clearly weak Whisper segments."""
    text = str(seg.get("text", "") or "").strip()
    if not text:
        return True
    if len(text) <= 1:
        return True
    logprob = seg.get("avg_logprob")
    no_speech = seg.get("no_speech_prob")
    if logprob is not None:
        try:
            if float(logprob) < float(getattr(config, "STT_SUSPICIOUS_MIN_LOGPROB", -1.0)):
                return True
        except Exception:
            pass
    if no_speech is not None:
        try:
            if float(no_speech) > float(getattr(config, "STT_SUSPICIOUS_MAX_NOSPEECH", 0.70)) and len(text) < 8:
                return True
        except Exception:
            pass
    # Obvious repetition is often a bad decode rather than real dialogue.
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 8:
        half = len(compact)//2
        if compact[:half] == compact[half:]:
            return True
    # For Chinese speech, isolated OCR-like symbols are strong hallucination signals.
    if getattr(config, "STT_LANGUAGE_DEFAULT", "zh") == "zh":
        bad_symbols = sum(1 for ch in compact if ch in "@{}[]<>^~\\|")
        if bad_symbols >= 1 and bad_symbols / max(1, len(compact)) >= 0.10:
            return True
    return False

def transcribe_fast(audio_path: str, api_key: str, provider: str = "groq",
                    language: Optional[str] = None, max_workers: int = MAX_WORKERS,
                    chunk_seconds: int = CHUNK_DURATION, initial_prompt: str = "",
                    model: Optional[str] = None) -> List[Dict[str, Any]]:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {audio_path}")
    max_workers = max(1, min(int(max_workers or MAX_WORKERS), MAX_WORKERS))
    logger.info(
        f"STT với {provider} | model={model or getattr(config, 'GROQ_STT_MODEL', 'whisper-large-v3-turbo')} "
        f"| quality_model={getattr(config, 'GROQ_STT_QUALITY_MODEL', 'whisper-large-v3')} "
        f"| chunk={chunk_seconds}s | overlap={float(getattr(config, 'STT_CHUNK_OVERLAP', 2.0)):.2f}s "
        f"| workers={max_workers} | word_ts={bool(getattr(config, 'STT_REQUEST_WORD_TIMESTAMPS', True))} "
        f"| visual_align={bool(getattr(config, 'STT_VISUAL_TIMING_ALIGN', False))}"
    )
    start_time = time.time()
    duration = _get_audio_duration(audio_path)
    if duration <= 0:
        raise ValueError(f"Audio không hợp lệ: {duration}")
    logger.info(f"Thời lượng: {duration:.2f}s")
    stt_keys = _get_stt_keys(api_key)
    if not stt_keys:
        raise RuntimeError(f"{provider}: thiếu API key")
    if provider == "groq":
        _check_groq_connectivity()
    chunks = _split_audio_into_chunks(audio_path, chunk_seconds)
    logger.info(f"{len(chunks)} chunks x {chunk_seconds}s "
                f"(overlap={float(getattr(config, 'STT_CHUNK_OVERLAP', 0.0)):.2f}s, "
                f"clamp={bool(getattr(config, 'STT_CLAMP_TO_CHUNK', True))}), "
                f"xử lý song song {max_workers} luồng...")
    all_segments: List[Dict[str, Any]] = []
    errors = []
    key_index = 0
    key_lock = threading.Lock()
    key_fail_count = {k: 0 for k in stt_keys}
    MAX_KEY_FAIL = max(3, len(stt_keys))

    def get_next_key(exclude=None):
        nonlocal key_index
        exclude = set(exclude or [])
        with key_lock:
            available_keys = [k for k in stt_keys
                              if k not in exclude and key_fail_count.get(k, 0) < MAX_KEY_FAIL]
            if not available_keys:
                for k in stt_keys:
                    key_fail_count[k] = 0
                available_keys = [k for k in stt_keys if k not in exclude] or stt_keys
            key = available_keys[key_index % len(available_keys)]
            key_index += 1
            return key

    def is_network_error(exc):
        s = str(exc).lower()
        return (isinstance(exc, (httpx.ConnectError, httpx.NetworkError, httpx.ConnectTimeout,
                                 httpx.ReadError, httpx.RemoteProtocolError,
                                 socket.gaierror, ConnectionError, OSError))
                or "getaddrinfo" in s or "timeout" in s or "connection" in s)

    def process_chunk(item: Tuple[str, float, float, int]) -> List[Dict[str, Any]]:
        chunk_path, chunk_start, chunk_end, idx = item
        tried_keys = set()
        last_error = None
        attempts = max(3, int(getattr(config, "STT_KEY_RETRY_ATTEMPTS", 3)))
        try:
            for key_try in range(attempts):
                current_key = get_next_key(exclude=tried_keys)
                tried_keys.add(current_key)
                try:
                    return _transcribe_chunk_with_retry(
                        chunk_path, current_key, provider, language, idx, chunk_start,
                        initial_prompt, chunk_end=chunk_end, model_override=model)
                except Exception as e:
                    last_error = e
                    logger.error(f"Chunk {idx+1}, key {key_try+1}/{attempts} lỗi: {e}")
                    if is_network_error(e):
                        with key_lock:
                            key_fail_count[current_key] = min(key_fail_count.get(current_key, 0) + 1, MAX_KEY_FAIL)
                        # Retry/backoff đã được xử lý ở tầng request; tầng này chỉ đổi key.
                        if key_try < attempts - 1:
                            continue
                    elif "429" in str(e) or "rate limit" in str(e).lower():
                        # 429 is organization-level. Do not rotate keys — that
                        # stamps the same org RPM limit and makes recognition slower
                        # and more error-prone. Inner retry already waited.
                        raise RuntimeError(f"Chunk {idx+1}: {e}")
                    else:
                        raise
            raise RuntimeError(f"Chunk {idx+1}: tất cả key đều lỗi. Lỗi cuối: {last_error}")
        finally:
            try:
                if str(chunk_path) != str(audio_path):
                    os.unlink(chunk_path)
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, item): item[3] for item in chunks}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                segments = future.result()
                if segments:
                    all_segments.extend(segments)
                    logger.info(f"Chunk {idx+1}: {len(segments)} đoạn")
                else:
                    logger.warning(f"Chunk {idx+1}: không nhận diện được lời thoại")
            except Exception as e:
                logger.error(f"Chunk {idx+1} thất bại: {e}")
                errors.append((idx, str(e)))

    if all_segments:
        all_segments = dedupe_chunk_overlap_segments(all_segments)

    # FIX TIMELINE #4: một chunk lỗi = mất trọn cửa sổ thời gian đó khỏi timeline.
    # Trước đây tỉ lệ <=20% vẫn đi tiếp và chỉ ghi warning, nên người dùng thấy
    # "phụ đề không khớp lời gốc" dù mốc còn lại vẫn đúng. Nay báo lỗi kèm
    # đúng khoảng thời gian bị mất để chạy lại đúng chỗ đó.
    failure_ratio = len(errors) / max(1, len(chunks))
    max_failure_ratio = float(getattr(config, "STT_MAX_CHUNK_FAILURE_RATIO", 0.0))
    if errors and failure_ratio > max_failure_ratio:
        window_by_idx = {item[3]: (item[1], item[2]) for item in chunks}
        lost = []
        for idx, msg in sorted(errors):
            a, b = window_by_idx.get(idx, (0.0, 0.0))
            lost.append(f"chunk {idx+1} [{_fmt_clock(a)}–{_fmt_clock(b)}]: {msg}")
        lost_seconds = sum(max(0.0, window_by_idx.get(i, (0.0, 0.0))[1] - window_by_idx.get(i, (0.0, 0.0))[0])
                           for i, _ in errors)
        raise RuntimeError(
            f"STT mất {len(errors)}/{len(chunks)} chunk (~{lost_seconds:.0f}s lời thoại) nên timeline sẽ "
            f"thiếu đoạn. Không xuất bản kết quả thiếu. Chi tiết: " + " | ".join(lost[:10]))
    if not all_segments:
        raise RuntimeError(f"Không nhận diện được segment nào. Lỗi: {errors}")
    all_segments.sort(key=lambda x: x.get("start", 0))
    # V18: word-level rebuild before quality review. Long Groq cues are split from actual words,
    # so quality review evaluates the same final cue units the editor will receive.
    all_segments = promote_word_timestamps_to_cue_timing(all_segments)
    all_segments = rebuild_segments_from_words(all_segments)
    all_segments = safe_dedupe_segments(all_segments)
    all_segments = enforce_word_based_nonoverlap(all_segments)
    all_segments = normalize_segments(all_segments)

    # Optional second pass is deliberately conservative: it retries only weak cues,
    # never the whole video. This keeps quota/time predictable while improving bad decodes.
    if provider == "groq" and getattr(config, "STT_SECOND_PASS", True):
        # Quality-pass phải thật nhỏ, không được biến một lần STT bình thường
        # thành lần nhận diện lại hàng chục cue.
        candidates = [s for s in all_segments
                      if not s.get("quality_checked") and _is_suspicious_segment(s)]
        def _quality_score(seg):
            text = str(seg.get("text", "") or "").strip()
            score = 0.0
            if not text: score += 10.0
            if len(text) <= 1: score += 6.0
            try: score += max(0.0, -float(seg.get("avg_logprob", 0.0)) - 1.0) * 3.0
            except Exception: pass
            try: score += max(0.0, float(seg.get("no_speech_prob", 0.0)) - 0.65) * 8.0
            except Exception: pass
            return score
        candidates.sort(key=_quality_score, reverse=True)
        max_items = max(0, int(getattr(config, "STT_SUSPICIOUS_MAX", 3)))
        suspicious = candidates[:max_items]
        if suspicious:
            logger.info(f"STT quality-pass: CHỈ kiểm tra lại {len(suspicious)} đoạn nghi ngờ (không chạy lại toàn bộ video)")
            # Re-run only the original chunk around each weak cue. The overlap/dedupe
            # logic below keeps one canonical cue stream.
            repaired = []
            for n, seg in enumerate(suspicious, 1):
                try:
                    a = max(0.0, float(seg.get("start", 0)) - 1.5)
                    b = float(seg.get("end", a + 0.5)) + 1.5
                    if b <= a: continue
                    # Generate a tiny temporary WAV around the suspicious cue.
                    tmp = Path(audio_path).with_name(f"{Path(audio_path).stem}_repair_{n:03d}.wav")
                    cmd = [config.FFMPEG_PATH, "-y", "-ss", f"{a:.3f}", "-t", f"{max(0.25,b-a):.3f}",
                           "-i", str(audio_path), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(tmp)]
                    subprocess.run(cmd, capture_output=True, check=True, timeout=60)
                    if not tmp.exists(): continue
                    key = stt_keys[n % len(stt_keys)]
                    rr = _transcribe_chunk_with_retry(
                        str(tmp), key, provider, language, 0, a,
                        initial_prompt, chunk_end=b,
                        model_override=getattr(config, "GROQ_STT_QUALITY_MODEL", None)
                    )
                    for x in rr:
                        x["start"] = float(x.get("start",0))
                        x["end"] = float(x.get("end",0))
                    # Pick the candidate with more text; for Chinese this avoids replacing a
                    # complete cue with a short hallucinated fragment.
                    if rr:
                        candidate = max(rr, key=lambda x: len(str(x.get("text", ""))))
                        # Keep the MASTER TIMELINE timestamps from the original segment.
                        # A quality retry may replace text only; it may never move a cue.
                        candidate["quality_repair"] = True
                        repaired.append((seg, candidate))
                    try: tmp.unlink()
                    except Exception: pass
                except Exception as e:
                    logger.warning(f"STT quality-pass câu {n} bỏ qua: {e}")
            for old_seg, candidate in repaired:
                old_len=len(str(old_seg.get("text", "")))
                new_len=len(str(candidate.get("text", "")))
                if new_len >= old_len and new_len > 1:
                    old_seg.update({"text":candidate.get("text"), "quality_repair":True})
                # Quality-pass chỉ được thay VĂN BẢN. Mốc start/end của STT giữ nguyên
                # để nó không bao giờ dịch được master timeline.
                old_seg["quality_checked"] = True
            for seg in suspicious:
                seg["quality_checked"] = True
    # Final pass: only cautious sentence grouping after quality repair.
    all_segments = regroup_stt_sentences(all_segments)
    all_segments = safe_dedupe_segments(all_segments)
    all_segments = enforce_word_based_nonoverlap(all_segments)
    all_segments = normalize_segments(all_segments)
    all_segments = merge_micro_segments(all_segments)
    if errors:
        logger.warning(f"STT hoàn tất một phần: {len(all_segments)} segments, còn {len(errors)} chunk lỗi")
    else:
        word_timed_count = sum(1 for s in all_segments if s.get("word_timed") or s.get("timing_source") == "groq_word")
        logger.info(f"STT xong: {len(all_segments)} đoạn trong {time.time()-start_time:.2f}s | word_timed={word_timed_count}/{len(all_segments)} | model={getattr(config, 'GROQ_STT_MODEL', 'whisper-large-v3-turbo')}")
    return all_segments


def clamp_segments_to_chunk(segments: List[Dict[str, Any]], chunk_start: float,
                            chunk_end: Optional[float] = None,
                            tolerance: Optional[float] = None) -> List[Dict[str, Any]]:
    """FIX TIMELINE #3: kẹp mốc cue vào đúng cửa sổ audio thực của chunk.

    Whisper pad audio lên 30s trước khi giải mã, nên chunk cuối (thường ngắn hơn
    chunk_duration) hay trả về mốc vượt ra ngoài audio thật, kèm câu ảo sinh ra
    trong phần im lặng đã pad. Đây là lý do các phân đoạn CUỐI lệch tệ nhất.

    Quy tắc:
      - cue bắt đầu ngoài biên chunk (quá tolerance) → bỏ hẳn (đuôi pad ảo)
      - cue tràn đuôi → cắt end về đúng chunk_end
      - cue sớm hơn chunk_start → đẩy về chunk_start
    Trợ giúp giữ nguyên thứ tự và không tạo cue mới.
    """
    if not segments:
        return []
    if not bool(getattr(config, "STT_CLAMP_TO_CHUNK", True)) or chunk_end is None:
        return list(segments)
    tol = float(getattr(config, "STT_CLAMP_TOLERANCE", 0.25) if tolerance is None else tolerance)
    lo = float(chunk_start)
    hi = float(chunk_end)
    if hi <= lo:
        return list(segments)
    kept: List[Dict[str, Any]] = []
    for seg in segments:
        a = float(seg.get("start", 0) or 0)
        b = float(seg.get("end", 0) or 0)
        # Câu ảo trong phần pad: bắt đầu sau khi audio thật đã hết.
        if a >= hi - 0.05 + tol:
            seg["dropped_padding_hallucination"] = True
            logger.info("[CLAMP] bỏ cue ảo ngoài biên chunk: %.3f-%.3f (biên %.3f-%.3f) | %s",
                        a, b, lo, hi, str(seg.get("text", ""))[:40])
            continue
        new_a = min(max(a, lo), hi - 0.05)
        new_b = min(max(b, new_a + 0.05), hi)
        if abs(new_a - a) > 0.001 or abs(new_b - b) > 0.001:
            seg["clamped_to_chunk"] = True
            seg["clamp_shift"] = round(max(abs(new_a - a), abs(new_b - b)), 3)
        seg["start"], seg["end"] = new_a, new_b
        seg["source_start"], seg["source_end"] = new_a, new_b
        seg["chunk_window"] = [lo, hi]
        kept.append(seg)
    return kept


def _transcribe_chunk_with_retry(chunk_path, api_key, provider, language, idx, chunk_start,
                                 initial_prompt="", chunk_end=None, model_override=None):

    last_error = None
    total_attempts = max(RETRY_COUNT, NETWORK_RETRY_COUNT, RATE_LIMIT_RETRY_COUNT)
    for attempt in range(total_attempts):
        try:
            if provider == "groq":
                segments = _transcribe_groq_chunk(chunk_path, api_key, language, initial_prompt, model=model_override)
            elif provider == "openai":
                segments = _transcribe_openai_chunk(chunk_path, api_key, language, model_override=model_override)
            elif provider == "xkira":
                raise RuntimeError("xKiro hiện không hỗ trợ STT qua API công khai; hãy dùng Groq hoặc Gemini.")
            elif provider == "google":
                segments = _transcribe_google_chunk(chunk_path, language)
            else:
                raise ValueError(f"Provider {provider} không hỗ trợ")
            offset = float(chunk_start or 0.0)
            for seg in segments:
                seg["start"] = float(seg.get("start", 0) or 0) + offset
                seg["end"] = float(seg.get("end", 0) or 0) + offset
                # Groq returns word timestamps relative to the uploaded chunk.
                # Promote them to absolute video/audio time before any merge/dedupe.
                if isinstance(seg.get("words"), list):
                    abs_words = []
                    for w in seg.get("words") or []:
                        try:
                            ws = float(w.get("start", 0) or 0) + offset
                            we = float(w.get("end", 0) or 0) + offset
                        except (TypeError, ValueError, AttributeError):
                            continue
                        wt = str(w.get("word", "") or "").strip()
                        if wt and we > ws:
                            abs_words.append({"word": wt, "start": ws, "end": we})
                    if abs_words:
                        seg["words"] = abs_words
                seg["chunk_index"] = idx
                seg["chunk_start"] = offset
            # FIX TIMELINE #3: không nhận mốc vượt ra ngoài audio thật của chunk.
            segments = clamp_segments_to_chunk(segments, offset, chunk_end)
            return segments
        except (httpx.ConnectError, httpx.NetworkError, httpx.ConnectTimeout,
                httpx.ReadError, httpx.RemoteProtocolError,
                socket.gaierror, ConnectionError, OSError) as e:
            last_error = e
            if attempt < total_attempts - 1:
                wait = min(1.5 * (2 ** attempt), NETWORK_BACKOFF_MAX) + (0.15 * attempt)
                logger.warning(f"Chunk {idx+1}: lỗi mạng (lần {attempt+1}/{total_attempts}) → thử lại sau {wait:.1f}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Chunk {idx+1}: lỗi mạng sau {total_attempts} lần: {e}")
        except httpx.TimeoutException as e:
            last_error = e
            if attempt < total_attempts - 1:
                wait = min(1.5 * (2 ** attempt), NETWORK_BACKOFF_MAX) + (0.15 * attempt)
                logger.warning(f"Chunk {idx+1}: timeout (lần {attempt+1}/{total_attempts}) → thử lại sau {wait:.1f}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Chunk {idx+1}: timeout sau {total_attempts} lần: {e}")
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                # Prefer server Retry-After when it is available; otherwise use a
                # conservative backoff. Do not spin workers against the same org.
                msg = str(e)
                m = re.search(r"retry[-_ ]after\s*[:=]\s*([0-9.]+)", msg, re.I)
                server_wait = float(m.group(1)) if m else 0.0
                wait = max(server_wait, float(getattr(config, "STT_429_MIN_WAIT", 8.0)) * (1.35 ** min(attempt, 6)))
                wait = min(wait, float(getattr(config, "STT_429_MAX_WAIT", 60.0)))
                _block_scheduler(wait)
                logger.warning(f"Chunk {idx+1}: rate limit 429, scheduler khóa {wait:.1f}s rồi retry ({attempt+1}/{total_attempts})")
                last_error = e
                if attempt < total_attempts - 1:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Chunk {idx+1}: Groq 429 sau {total_attempts} lần. Giảm số luồng STT hoặc đợi rồi chạy lại.")
            raise RuntimeError(f"Chunk {idx+1}: {e}")
    raise RuntimeError(f"Chunk {idx+1} thất bại sau {total_attempts} lần. Lỗi cuối: {last_error}")


def _transcribe_groq_chunk(audio_path: str, api_key: str, language: Optional[str], initial_prompt: str = "", model: Optional[str] = None):

    if not api_key:
        raise RuntimeError("Groq: thiếu API key")
    _enforce_rate_limit(api_key)
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    files = [
        ("file", (Path(audio_path).name, audio_data, "audio/wav")),
        ("model", (None, str(model or getattr(config, "GROQ_STT_MODEL", "whisper-large-v3-turbo")))),
        ("response_format", (None, "verbose_json")),
        ("timestamp_granularities[]", (None, "segment")),
        ("temperature", (None, "0")),
    ]
    # FIX TIMELINE #5: word-level timestamp kích hoạt bước tinh chỉnh mốc có lỗi lặp
    # văn bản/mốc đã biết của Whisper, trong khi code không dùng tới `words`.
    if bool(getattr(config, "STT_REQUEST_WORD_TIMESTAMPS", False)):
        files.append(("timestamp_granularities[]", (None, "word")))
    if language and language != "auto":
        files.append(("language", (None, language)))
    prompt_text = (initial_prompt or "").strip()
    if language == "zh":
        base = "请准确识别视频中的中文普通话对白。只输出实际听到的中文语音，不要翻译、不要补写、不要改写、不要猜测听不清的词；保留人名、地名、数字、游戏名、角色名、物品名、技能名、品牌和其他专有名词的原始写法；不要凭空添加 @ { } [ ] < > / 等符号。"
        prompt_text = base + (" " + prompt_text if prompt_text else "")
    if prompt_text:
        files.append(("prompt", (None, prompt_text[:1800])))
    with httpx.Client(verify=False,
                      timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=min(REQUEST_TIMEOUT, CONNECT_TIMEOUT)),
                      trust_env=False) as client:
        response = client.post(GROQ_TRANSCRIPTION_URL, headers=headers, files=files)
        if response.status_code != 200:
            error_msg = response.text[:200]
            if response.status_code == 413:
                raise RuntimeError("Groq: file quá lớn (>25MB) – giảm chunk_duration xuống 15s")
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after", "")
                suffix = f" retry-after={retry_after}" if retry_after else ""
                raise RuntimeError(f"Groq API lỗi 429{suffix}: {error_msg}")
            raise RuntimeError(f"Groq API lỗi {response.status_code}: {error_msg}")
        result = response.json()
        segments = []
        raw_segments = result.get("segments") or []
        words = result.get("words") or []
        if raw_segments:
            for seg in raw_segments:
                text = str(seg.get("text", "") or "").strip()
                rs = float(seg.get("start", 0) or 0)
                re_ = float(seg.get("end", 0) or 0)
                if not text or re_ <= rs:
                    continue
                # IMPORTANT: Whisper's segment timestamps are the canonical source timeline.
                # Do NOT replace them with the min/max of word timestamps: word-level spans can
                # cross cue boundaries and make the editor subtitle blocks overlap/drift.
                item = {"start": rs, "end": re_, "text": text, "timing_source": "segment"}
                # Keep word timestamps for optional downstream timing refinement.
                if words:
                    local_words = []
                    for w in words:
                        try:
                            ws = float(w.get("start", 0) or 0); we = float(w.get("end", 0) or 0)
                        except (TypeError, ValueError, AttributeError):
                            continue
                        wt = str(w.get("word", "") or "").strip()
                        if wt and we > ws and ws >= rs - 0.15 and we <= re_ + 0.15:
                            local_words.append({"word": wt, "start": ws, "end": we})
                    if local_words:
                        item["words"] = local_words
                for k in ("avg_logprob", "no_speech_prob", "compression_ratio"):
                    if k in seg:
                        item[k] = seg.get(k)
                segments.append(item)
        elif "text" in result and result["text"].strip():
            segments.append({"start": 0.0, "end": _get_audio_duration(Path(audio_path)),
                             "text": result["text"].strip()})
        return segments



_SENTENCE_FINAL_RE = re.compile(r"[。！？!?；;]$")

def _cjk_text_join(parts: List[str]) -> str:
    vals = [str(x or "").strip() for x in parts if str(x or "").strip()]
    if not vals:
        return ""
    joined = ""
    for v in vals:
        if not joined:
            joined = v
            continue
        # Chinese subtitle text generally does not need a space at joins.
        if re.search(r"[\u4e00-\u9fff]$", joined) and re.match(r"[\u4e00-\u9fff]", v):
            joined += v
        elif re.search(r"[\u4e00-\u9fff]$", joined) and v[:1] in "，。！？；：、）》」』】":
            joined += v
        else:
            joined = (joined + " " + v).strip()
    return joined.strip()


def _join_word_tokens(tokens: List[str]) -> str:
    """Join Groq word tokens without inventing spaces inside Chinese text."""
    vals = [str(t or "").strip() for t in tokens if str(t or "").strip()]
    if not vals:
        return ""
    out = ""
    no_space_before = "，。！？；：、）》」』】）)]}>…,.!?;:"
    no_space_after = "《「『【（([{<"
    for tok in vals:
        if not out:
            out = tok
            continue
        prev = out[-1:] or ""
        first = tok[:1] or ""
        if first in no_space_before or prev in no_space_after:
            out += tok
        elif re.search(r"[\u4e00-\u9fff]$", out) and re.match(r"[\u4e00-\u9fff]", tok):
            out += tok
        else:
            out += " " + tok
    return out.strip()


def rebuild_segments_from_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rebuild long ASR cues from word timestamps instead of leaving one large block.

    Groq/Whisper can return a very long segment even when several natural subtitle
    boundaries exist inside it. This pass creates cues from the actual word timing,
    using punctuation/pauses first and hard duration/character caps second.
    It never invents a time: every cue starts at its first word and ends at its last word.
    """
    max_dur = float(getattr(config, "STT_FINAL_CUE_MAX_DURATION", 4.2))
    max_chars = int(getattr(config, "STT_FINAL_CUE_MAX_CHARS", 42))
    pause_cut = float(getattr(config, "STT_FINAL_CUE_PAUSE_CUT", 0.22))
    min_chars_for_pause = int(getattr(config, "STT_FINAL_CUE_MIN_CHARS_FOR_PAUSE", 4))
    final_punct = re.compile(r"[。！？!?；;]$")
    rebuilt: List[Dict[str, Any]] = []

    for raw in sorted((dict(x) for x in (segments or [])), key=lambda x: float(x.get("start", 0) or 0)):
        raw_words=[]
        for w in (raw.get("words") or []):
            try:
                ws=float(w.get("start", 0) or 0); we=float(w.get("end", 0) or 0)
            except Exception:
                continue
            wt=str(w.get("word", "") or "").strip()
            if wt and we>ws:
                raw_words.append({"word":wt,"start":ws,"end":we})
        raw_words.sort(key=lambda x:(x["start"],x["end"]))
        if len(raw_words) < 2:
            rebuilt.append(raw)
            continue

        group=[]
        base_meta={k:v for k,v in raw.items() if k not in {"start","end","text","words","segment_id"}}
        prev_word=None
        def flush(reason=None):
            nonlocal group
            if not group:
                return
            text=_join_word_tokens([x["word"] for x in group])
            if not text:
                group=[]
                return
            item=dict(base_meta)
            item.update({
                "start":float(group[0]["start"]),
                "end":float(group[-1]["end"]),
                "text":text,
                "words":list(group),
                "word_timed":True,
                "timing_source":"groq_word",
                "word_rebuilt":True,
            })
            if reason:
                item["word_split_reason"]=reason
            rebuilt.append(item)
            group=[]

        for w in raw_words:
            if prev_word and group:
                gap=max(0.0,float(w["start"])-float(prev_word["end"]))
                current_text=_join_word_tokens([x["word"] for x in group])
                current_dur=float(prev_word["end"])-float(group[0]["start"])
                current_chars=len(re.sub(r"\s+", "", current_text))
                # Natural pause boundary: only cut when there is already enough text.
                if gap >= pause_cut and current_chars >= min_chars_for_pause:
                    flush("pause")
                elif current_dur >= max_dur or current_chars >= max_chars:
                    # Prefer the last natural word boundary. If no punctuation exists,
                    # the current word itself is the boundary and no time is fabricated.
                    flush("duration_or_chars")
            group.append(w)
            prev_word=w
            if final_punct.search(_join_word_tokens([x["word"] for x in group])):
                flush("punctuation")
        flush("eof")

    # Any original cue that had no usable word timings remains untouched.
    if not rebuilt:
        return list(segments or [])
    rebuilt.sort(key=lambda x: (float(x.get("start",0) or 0), float(x.get("end",0) or 0)))
    for i,seg in enumerate(rebuilt):
        seg["segment_id"]=i
    return rebuilt

def promote_word_timestamps_to_cue_timing(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use Groq word timestamps as the cue timing when available.

    Segment text remains the recognition source. Only start/end are tightened to
    the first/last aligned word belonging to that segment. If a segment has no
    usable word timestamps, its original timestamps are retained.
    """
    out = []
    for raw in segments or []:
        seg = dict(raw)
        words = []
        for w in (seg.get("words") or []):
            try:
                ws = float(w.get("start", 0) or 0)
                we = float(w.get("end", 0) or 0)
            except (TypeError, ValueError, AttributeError):
                continue
            wt = str(w.get("word", "") or "").strip()
            if wt and we > ws:
                words.append({"word": wt, "start": ws, "end": we})
        if len(words) >= int(getattr(config, "STT_WORD_TIMING_MIN_WORDS", 1)):
            words.sort(key=lambda x: (x["start"], x["end"]))
            a = words[0]["start"]
            b = words[-1]["end"]
            old_a = float(seg.get("start", a) or a)
            old_b = float(seg.get("end", b) or b)
            # Never let word timing escape the original segment by a large amount.
            # A small inward trim removes leading/trailing silence while preserving
            # the ASR cue's intended region.
            if a >= old_a - 0.35 and b <= old_b + 0.35 and b > a:
                seg["start"], seg["end"] = a, b
                seg["timing_source"] = "groq_word"
                seg["word_timed"] = True
        out.append(seg)
    return out

def regroup_stt_sentences(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Conservatively merge adjacent ASR cues into speech sentences.

    Merge only when the gap is short, the previous cue is not sentence-final,
    and the merged cue stays within duration/character limits. Word timestamps
    are retained and become the merged cue's timing source.
    """
    ordered = sorted((dict(x) for x in (segments or [])),
                     key=lambda x: float(x.get("start", 0) or 0))
    if not ordered:
        return []
    max_gap = float(getattr(config, "STT_SENTENCE_MERGE_MAX_GAP", 0.18))
    max_dur = float(getattr(config, "STT_SENTENCE_MAX_DURATION", 4.2))
    max_chars = int(getattr(config, "STT_SENTENCE_MAX_CHARS", 42))
    out = [ordered[0]]
    for cur in ordered[1:]:
        prev = out[-1]
        pa, pb = float(prev.get("start", 0) or 0), float(prev.get("end", 0) or 0)
        ca, cb = float(cur.get("start", 0) or 0), float(cur.get("end", 0) or 0)
        gap = max(0.0, ca - pb)
        ptxt = str(prev.get("text", "") or "").strip()
        ctxt = str(cur.get("text", "") or "").strip()
        merged_text = _cjk_text_join([ptxt, ctxt])
        should_merge = (
            not prev.get("word_rebuilt") or not prev.get("word_split_reason")
        ) and (
            gap <= max_gap
            and not _SENTENCE_FINAL_RE.search(ptxt)
            and cb > pa
            and (cb - pa) <= max_dur
            and len(re.sub(r"\s+", "", merged_text)) <= max_chars
        )
        if should_merge:
            prev["text"] = merged_text
            prev["end"] = cb
            prev["timing_source"] = "groq_word" if (prev.get("word_timed") or cur.get("word_timed")) else "segment"
            prev["sentence_grouped"] = True
            merged_words = list(prev.get("words") or []) + list(cur.get("words") or [])
            if merged_words:
                merged_words.sort(key=lambda x: float(x.get("start", 0) or 0))
                prev["words"] = merged_words
                prev["start"] = float(merged_words[0].get("start", prev["start"]) or prev["start"])
                prev["end"] = float(merged_words[-1].get("end", prev["end"]) or prev["end"])
            continue
        out.append(cur)
    for i, seg in enumerate(out):
        seg["segment_id"] = i
    return out

def enforce_word_based_nonoverlap(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Final guard: remove cue overlap without midpoint shifts.

    When two word-timed cues overlap because of ASR boundary noise, preserve the
    later cue's start and trim only the previous cue's end to that start. If that
    would invalidate the previous cue, preserve its end and move the later start
    only to the previous end as a last resort.
    """
    out = sorted((dict(x) for x in (segments or [])),
                 key=lambda x: (float(x.get("start", 0) or 0), float(x.get("end", 0) or 0)))
    cleaned = []
    for seg in out:
        a = max(0.0, float(seg.get("start", 0) or 0))
        b = max(a + 0.05, float(seg.get("end", 0) or 0))
        seg["start"], seg["end"] = a, b
        if cleaned:
            prev = cleaned[-1]
            pa, pb = float(prev["start"]), float(prev["end"])
            if a < pb:
                if a > pa + 0.05:
                    prev["end"] = a
                else:
                    a = pb
                    if a >= b - 0.05:
                        # Duplicate/degenerate overlap; keep the longer cue.
                        if (b - a) > (pb - pa):
                            cleaned[-1] = seg
                        continue
                    seg["start"] = a
                seg["timeline_repaired"] = True
                prev["timeline_repaired"] = True
        cleaned.append(seg)
    for i, seg in enumerate(cleaned):
        seg["segment_id"] = i
        seg["source_start"] = float(seg.get("start", 0) or 0)
        seg["source_end"] = float(seg.get("end", 0) or 0)
    return cleaned

def _transcribe_openai_chunk(audio_path: str, api_key: str, language: Optional[str]):
    if not api_key:
        raise RuntimeError("OpenAI: thiếu API key")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT)
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    response = client.audio.transcriptions.create(
        model=str(model_override or "whisper-1"), file=(Path(audio_path).name, audio_data),
        response_format="verbose_json",
        language=language if language and language != "auto" else None)
    segments = []
    if hasattr(response, 'segments') and response.segments:
        for seg in response.segments:
            text = getattr(seg, "text", "").strip()
            if text:
                segments.append({"start": float(getattr(seg, "start", 0)),
                                 "end": float(getattr(seg, "end", 0)), "text": text})
    elif hasattr(response, 'text') and response.text:
        segments.append({"start": 0.0, "end": _get_audio_duration(Path(audio_path)),
                         "text": response.text.strip()})
    return segments


def _transcribe_xkira_chunk(audio_path: str, api_key: str, language: Optional[str]):
    raise RuntimeError(
        "xKiro hiện không cung cấp Speech-to-Text (STT) qua API công khai. "
        "Hãy dùng Groq Whisper hoặc Gemini STT."
    )

def _transcribe_google_chunk(audio_path: str, language: Optional[str]):
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=language or "vi-VN")
        if text:
            return [{"start": 0.0, "end": _get_audio_duration(Path(audio_path)), "text": text}]
        return []
    except ImportError:
        raise RuntimeError("SpeechRecognition chưa cài. Cài: pip install SpeechRecognition")
    except Exception as e:
        raise RuntimeError(f"Google STT lỗi: {e}")


def _split_audio_into_chunks(audio_path: Path, chunk_duration: int) -> List[Tuple[str, float, float, int]]:
    """Split audio with a small overlap and return the real chunk start time.

    The old implementation used idx * chunk_duration for offsets and ignored
    STT_CHUNK_OVERLAP, which could clip words at boundaries and misplace them.
    """
    duration = _get_audio_duration(audio_path)
    if duration <= chunk_duration:
        return [(str(audio_path), 0.0, duration, 0)]
    chunks: List[Tuple[str, float, float, int]] = []
    base_name, base_dir = audio_path.stem, audio_path.parent
    overlap = max(0.0, min(float(getattr(config, "STT_CHUNK_OVERLAP", 0.0)), float(chunk_duration) * 0.25))
    step = max(1.0, float(chunk_duration) - overlap)
    # FIX TIMELINE #7: bẻ chunk đúng giữa mốc cứng 30s sẽ cắt ngang câu đang nói.
    # Câu bị cắt sinh 2 cue rọi ở hai chunk liền nhau, cue ở chunk sau bị dính vào
    # đúng mốc biên (30.0 / 60.0 / ...) nên lệch rất nặng. Kéo mốc cắt về khoảng
    # lặng gần nhất để không chunk nào cắt ngang lời thoại.
    snap = bool(getattr(config, "STT_SNAP_TO_SILENCE", True))
    snap_window = float(getattr(config, "STT_SNAP_WINDOW", 5.0))
    silence_points = _detect_silence_midpoints(audio_path) if snap else []
    if snap:
        logger.info(f"Bẻ chunk theo khoảng lặng: tìm thấy {len(silence_points)} điểm lặng")
    idx = 0
    start = 0.0
    while start < duration - 1e-6:
        end = min(duration, start + float(chunk_duration))
        if snap and end < duration - 1e-6:
            # Trần = end (không được vượt cửa sổ 30s của Whisper), sàn = start + 5s.
            snapped = _snap_to_silence(end, silence_points, snap_window, start, upper_bound=end)
            if start + 5.0 <= snapped <= end + 1e-6:
                if abs(snapped - end) > 0.01:
                    logger.info(f"Chunk {idx}: kéo mốc cắt {end:.2f}s → {snapped:.2f}s (khoảng lặng)")
                end = snapped
        chunk_path = base_dir / f"{base_name}_chunk_{idx:03d}.wav"
        cmd = [config.FFMPEG_PATH, "-y", "-ss", f"{start:.3f}", "-t", f"{max(0.05, end-start):.3f}",
               "-i", str(audio_path), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(chunk_path)]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                chunks.append((str(chunk_path), start, end, idx))
            else:
                raise RuntimeError(f"Chunk {idx} rỗng")
        except Exception as e:
            raise RuntimeError(f"Tách chunk {idx}: {e}")
        if end >= duration - 1e-6:
            break
        # Tiếp tục từ đúng mốc đã cắt để không hở và không chồng timeline.
        start = min(duration, end - overlap) if overlap > 0 else end
        idx += 1
    if not chunks:
        raise RuntimeError("Không tách được chunk nào")
    return chunks


def _fmt_clock(seconds: float) -> str:
    """Giây -> mm:ss để log/thông báo lỗi chỉ đúng vị trí trong video."""
    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        total = 0
    return f"{total // 60:02d}:{total % 60:02d}"


def _detect_silence_midpoints(audio_path: Path) -> List[float]:
    """Danh sách điểm giữa các khoảng lặng, dùng làm chỗ bẻ chunk an toàn."""
    noise_db = float(getattr(config, "STT_SILENCE_NOISE_DB", -35.0))
    min_len = float(getattr(config, "STT_SILENCE_MIN_LEN", 0.25))
    try:
        r = subprocess.run(
            [config.FFMPEG_PATH, "-hide_banner", "-i", str(audio_path),
             "-af", f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"],
            capture_output=True, text=True, timeout=600)
    except Exception as e:
        logger.warning(f"Không dò được khoảng lặng ({e}); bẻ chunk theo mốc cứng")
        return []
    points: List[float] = []
    start = None
    for m in re.finditer(r"silence_(start|end):\s*(-?[0-9.]+)", r.stderr or ""):
        kind, val = m.group(1), float(m.group(2))
        if kind == "start":
            start = val
        elif start is not None:
            points.append(round((start + val) / 2.0, 3))
            start = None
    return sorted(p for p in points if p > 0)


def _snap_to_silence(target: float, points: List[float], window: float,
                     lower_bound: float, upper_bound: Optional[float] = None) -> float:
    """Kéo mốc cắt về khoảng lặng gần nhất NẰM TRONG phạm vi cho phép.

    Chỉ xét các điểm thỏa lower_bound < p <= upper_bound. Nếu chọn điểm gần nhất
    mà không xét trần thì điểm tốt nhất hay bị điểm ngoài trần "giành chỗ" rồi
    bị loại, khiến mốc cắt rơi về mốc cứng 30s — đúng chỗ đang nói dở.
    """
    if not points:
        return target
    hi = target if upper_bound is None else float(upper_bound)
    best, best_d = None, window + 1.0
    for p in points:
        if p <= lower_bound + 0.5 or p > hi + 1e-6:
            continue
        d = abs(p - target)
        if d <= window and d < best_d:
            best, best_d = p, d
    return best if best is not None else target


def _text_norm(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def dedupe_chunk_overlap_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate cues produced by overlapped STT chunks without changing cue timing."""
    ordered = sorted((dict(x) for x in segments), key=lambda x: float(x.get("start", 0) or 0))
    out: List[Dict[str, Any]] = []
    for seg in ordered:
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue
        a = float(seg.get("start", 0) or 0); b = float(seg.get("end", 0) or 0)
        duplicate_index = None
        best_score = 0.0
        for j in range(len(out) - 1, max(-1, len(out) - 8), -1):
            prev = out[j]
            pa = float(prev.get("start", 0) or 0); pb = float(prev.get("end", 0) or 0)
            overlap = min(b, pb) - max(a, pa)
            if overlap <= 0.05:
                if a - pb > 2.5:
                    break
                continue
            sim = SequenceMatcher(None, _text_norm(prev.get("text", "")), _text_norm(text)).ratio()
            if sim >= 0.82:
                score = sim + 0.01 * min(len(text), 200)
                if score > best_score:
                    duplicate_index = j; best_score = score
        if duplicate_index is None:
            seg["source_start"] = a; seg["source_end"] = b
            out.append(seg)
        else:
            # Keep the clearer/longer transcription and its own timestamps.
            prev = out[duplicate_index]
            if len(text) > len(str(prev.get("text", "") or "")):
                chosen = seg
            else:
                chosen = prev
            chosen["deduped_overlap"] = True
            chosen["source_start"] = float(chosen.get("start", 0) or 0)
            chosen["source_end"] = float(chosen.get("end", 0) or 0)
            out[duplicate_index] = chosen
    return sorted(out, key=lambda x: float(x.get("start", 0) or 0))

def stabilize_stt_timeline(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove boundary duplicates and prevent overlapping STT cues.

    Whisper segment timestamps stay canonical. This pass only repairs duplicate/overlapping
    recognition artifacts so the editor gets one monotonic cue stream instead of stacked blocks.
    """
    ordered = sorted((dict(x) for x in (segments or [])), key=lambda x: float(x.get("start", 0) or 0))
    out: List[Dict[str, Any]] = []
    for seg in ordered:
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue
        a = max(0.0, float(seg.get("start", 0) or 0))
        b = max(a + 0.05, float(seg.get("end", 0) or 0))
        seg["start"], seg["end"] = a, b
        # Remove duplicates, including chunk-boundary duplicates with imperfect text similarity.
        dup = None
        for j in range(len(out) - 1, max(-1, len(out) - 12), -1):
            prev = out[j]
            pa = float(prev.get("start", 0) or 0); pb = float(prev.get("end", 0) or 0)
            ov = min(b, pb) - max(a, pa)
            if ov <= 0.0:
                if a - pb > 2.5:
                    break
                continue
            sim = SequenceMatcher(None, _text_norm(prev.get("text", "")), _text_norm(text)).ratio()
            shorter = max(0.05, min(pb-pa, b-a))
            overlap_ratio = ov / shorter
            if (sim >= 0.72 and overlap_ratio >= 0.35) or (sim >= 0.88 and ov >= 0.15):
                dup = j
                break
        if dup is not None:
            prev = out[dup]
            if len(text) > len(str(prev.get("text", "") or "")):
                chosen = seg
            else:
                chosen = prev
            chosen["deduped_overlap"] = True
            out[dup] = chosen
            continue
        out.append(seg)

    # Enforce monotonic cue boundaries. Normally this does nothing; it is a final guard against
    # Whisper/overlap artifacts producing stacked timeline cards. For substantial overlap, split at
    # the midpoint instead of letting one cue sit on top of another.
    for i in range(1, len(out)):
        prev, cur = out[i-1], out[i]
        pa, pb = float(prev["start"]), float(prev["end"])
        ca, cb = float(cur["start"]), float(cur["end"])
        if ca < pb:
            boundary = ca if (pb-ca) <= 0.20 else round((pa + cb) / 2.0, 3)
            boundary = max(pa + 0.05, min(cb - 0.05, boundary))
            prev["end"] = boundary
            cur["start"] = boundary
            prev["timeline_repaired"] = True
            cur["timeline_repaired"] = True
    for i, seg in enumerate(out):
        seg["segment_id"] = i
        seg.setdefault("source_start", float(seg.get("start", 0) or 0))
        seg.setdefault("source_end", float(seg.get("end", 0) or 0))
    return out


def _get_audio_duration(audio_path: Path) -> float:
    try:
        cmd = [config.FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        return float(result.stdout.strip())
    except Exception as e1:
        try:
            from pydub import AudioSegment
            return len(AudioSegment.from_file(audio_path)) / 1000.0
        except Exception as e2:
            raise RuntimeError(f"Không thể đọc duration {audio_path}: {e1}, {e2}")


def normalize_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize numeric fields without altering Whisper source timing.

    Timing is source-of-truth. Overlap correction here used to trim cue ends and
    made the editor timeline differ from the timestamps returned by STT/OCR.
    """
    if not segments:
        return []
    ordered = sorted((dict(x) for x in segments), key=lambda x: float(x.get("start", 0) or 0))
    normalized: List[Dict[str, Any]] = []
    for seg in ordered:
        start = max(0.0, float(seg.get("start", 0) or 0))
        end = float(seg.get("end", 0) or 0)
        if end <= start:
            end = start + 0.05
        seg["start"], seg["end"] = start, end
        seg.setdefault("source_start", start); seg.setdefault("source_end", end)
        if normalized and start < float(normalized[-1].get("end", 0) or 0):
            seg["timeline_overlap"] = True
            normalized[-1]["timeline_overlap"] = True
        normalized.append(seg)
    for i, seg in enumerate(normalized):
        seg["segment_id"] = i
    return normalized

def merge_micro_segments(segments: List[Dict[str, Any]], min_duration: float = None, max_duration: float = None) -> List[Dict[str, Any]]:
    """V19: merge only obviously artificial micro-cues.

    A cue shorter than ~0.30s is normally impossible to voice in Vietnamese without
    extreme compression. We only merge it when the next/previous cue is immediately
    adjacent (small gap), neither boundary is a strong sentence boundary, and the
    combined cue remains within the configured max duration. This avoids creating
    0.08–0.24s TTS slots while preserving normal short utterances.
    """
    raw_list=[]
    micro_thr=float(min_duration if min_duration is not None else getattr(config,"STT_MICRO_SEGMENT_MAX_DURATION",0.30))
    max_comb=float(max_duration if max_duration is not None else getattr(config,"STT_MICRO_SEGMENT_MAX_COMBINED",4.20))
    max_gap=float(getattr(config,"STT_MICRO_SEGMENT_MAX_GAP",0.15))
    for raw0 in sorted((dict(x) for x in (segments or [])), key=lambda x: float(x.get("start",0) or 0)):
        txt=str(raw0.get("text", "") or "").strip()
        if not txt:
            continue
        try:
            a=max(0.0,float(raw0.get("start",0) or 0)); b=float(raw0.get("end",0) or 0)
        except Exception:
            continue
        if b<=a: b=a+0.05
        raw0["start"],raw0["end"]=a,b
        raw0.setdefault("source_start",a); raw0.setdefault("source_end",b)
        raw_list.append(raw0)

    out=[]
    for seg in raw_list:
        dur=float(seg["end"])-float(seg["start"])
        if not out:
            out.append(seg); continue
        prev=out[-1]
        gap=max(0.0,float(seg["start"])-float(prev["end"]))
        prev_txt=str(prev.get("text","") or "").strip()
        seg_txt=str(seg.get("text","") or "").strip()
        prev_strong=bool(re.search(r"[。！？!?]$",prev_txt))
        # Merge only if current cue is micro. We also allow merging a micro previous
        # cue into the current one through this same pass by checking prev duration.
        merge_prev = (float(prev["end"])-float(prev["start"]) < micro_thr)
        should_merge = (dur < micro_thr or merge_prev) and gap <= max_gap and not prev_strong
        combined=float(seg["end"])-float(prev["start"])
        if should_merge and combined <= max_comb:
            # Do not invent a midpoint; use the union of the two observed speech slots.
            # Text is concatenated only once, respecting the original order.
            combined_text = prev_txt + seg_txt
            cjk_count = sum(1 for ch in combined_text if "\u3400" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff")
            joiner = "" if prev_txt and seg_txt and cjk_count >= max(2, len(combined_text) * 0.30) else " "
            prev["text"]=(prev_txt+joiner+seg_txt).strip()
            if isinstance(prev.get("words"),list) or isinstance(seg.get("words"),list):
                prev_words=list(prev.get("words") or []) + list(seg.get("words") or [])
                prev["words"]=sorted(prev_words,key=lambda w: float(w.get("start",0) or 0))
                prev["word_timed"]=True
            prev["end"]=max(float(prev["end"]),float(seg["end"]))
            prev["source_end"]=prev["end"]
            prev["micro_merged"]=True
            continue
        out.append(seg)

    for i,seg in enumerate(out):
        seg["segment_id"]=i
        if max_duration and (float(seg["end"])-float(seg["start"]))>float(max_duration):
            seg["long_cue"]=True
    if len(out) < len(raw_list):
        logger.info("STT V19: gộp %d micro-cue để tránh slot TTS quá ngắn", len(raw_list)-len(out))
    return out

def safe_dedupe_segments(segments):
    """Xóa cue TRÙNG KHÍT do chồng lấn chunk, KHÔNG BAO GIỜ dịch mốc thời gian.

    Thay thế cho `stabilize_stt_timeline` của các bản trước. Hàm cũ có hai hành vi
    gây lệch timeline:
      1. Khi hai cue chồng nhau > 0.20s, nó đặt lại biên = trung điểm (start+end),
         kéo cue sau hiện MUỘN hơn (đo được tới +670 ms) hoặc SỚM hơn (-525 ms)
         so với lời thoại thật, và kéo dài/cắt ngắn cue trước (tới -905 ms).
      2. Ngưỡng gộp quá lỏng (giống 72%) nên xóa cả câu KHÁC NGHĨA — ví dụ
         "Tôi sẽ không bỏ cậu đâu!" bị xóa vì giống "Tôi sẽ không bỏ cuộc đâu!".

    Hàm này chỉ xóa khi văn bản gần như trùng khít (mặc định >= 0.95) VÀ chồng nhau
    nhiều (>= 60% cue ngắn hơn) — tức đúng trường hợp một câu bị nhận diện hai lần ở
    hai chunk liền nhau. Mốc start/end của mọi cue sống sót được giữ NGUYÊN VĂN.
    """
    if not getattr(config, "STT_SAFE_DEDUPE", True):
        return list(segments or [])
    sim_min = float(getattr(config, "STT_DEDUPE_SIMILARITY", 0.95))
    ov_min = float(getattr(config, "STT_DEDUPE_MIN_OVERLAP", 0.60))
    ordered = sorted((dict(x) for x in (segments or [])),
                     key=lambda x: (float(x.get("start", 0) or 0), float(x.get("end", 0) or 0)))
    out = []
    for seg in ordered:
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue
        a = float(seg.get("start", 0) or 0); b = float(seg.get("end", 0) or 0)
        is_dup = False
        for prev in reversed(out[-6:]):
            pa = float(prev.get("start", 0) or 0); pb = float(prev.get("end", 0) or 0)
            ov = min(b, pb) - max(a, pa)
            if ov <= 0:
                continue
            shorter = max(0.05, min(pb - pa, b - a))
            if ov / shorter < ov_min:
                continue
            if SequenceMatcher(None, _text_norm(prev.get("text", "")), _text_norm(text)).ratio() >= sim_min:
                # giữ bản có văn bản dài hơn, nhưng KHÔNG đổi mốc của bản đang giữ
                if len(text) > len(str(prev.get("text", "") or "")):
                    prev["text"] = text
                prev["deduped_duplicate"] = True
                is_dup = True
                break
        if not is_dup:
            out.append(seg)
    removed = len(ordered) - len(out)
    if removed:
        logger.info("STT gộp %d cue trùng khít (mốc thời gian giữ nguyên)", removed)
    return out


__all__ = ["transcribe_fast", "normalize_segments", "merge_micro_segments", "safe_dedupe_segments", "align_stt_segments_to_video"]
