"""pipeline/render.py - Render video + TTS audio + subtitle.
burn_subtitle=False (default): stream copy video + audio mới, ghi SRT riêng.
burn_subtitle=True: re-encode libx264 + subtitles filter.
Audio mix theo chunk khi số segment lớn. Silent-bed toàn thời lượng.
Partial TTS mode: missing/failed TTS clips are skipped so one bad segment
never blocks the final video export. V26 also prevents adjacent TTS clips from masking
trailing words: full clips are preserved, with only small content-preserving speed fit
when required, and chunk boundaries are chosen so no clip is cut at a chunk edge.
"""
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import config
from pipeline import subtitle as subtitle_mod
from pipeline import tts as tts_mod

logger = logging.getLogger(__name__)

MAX_TTS_WORKERS = int(getattr(config, "MAX_TTS_WORKERS", 24))
FFMPEG_TIMEOUT = int(getattr(config, "RENDER_TIMEOUT", 3600))
SEGMENT_SILENCE_GAP = 0.05
# TTS is generated at natural rate. During render, only a small, content-preserving
# time compression is allowed when the natural clip would collide with the next line.
DUB_MAX_ATEMPO = float(getattr(config, "DUB_MAX_ATEMPO", 1.60))
DUB_EARLY_HEADROOM = float(getattr(config, "DUB_EARLY_HEADROOM", 0.20))
BURN_SUBTITLE = bool(getattr(config, "RENDER_BURN_SUBTITLE", False))
AUDIO_CHUNK_SEGMENTS = max(50, int(getattr(config, "RENDER_AUDIO_CHUNK", 400)))
_COPY_VIDEO_CODECS = {"h264", "hevc", "hev1", "hvc1", "mpeg4", "vp9", "av1"}


def render_final_video_fast(video_path: Path, segments: List[Dict[str, Any]],
                            translated_texts: List[str], tts_provider: str, voice: str,
                            tts_api_key: Optional[str], output_path: Path,
                            keep_original_audio: bool = False, max_workers: int = MAX_TTS_WORKERS,
                            rate: float = 1.0, pitch: float = 0.0,
                            existing_tts_results: Optional[List[Dict[str, Any]]] = None,
                            existing_tts: Optional[Dict[str, Any]] = None,
                            burn_subtitle: Optional[bool] = None,
                            include_dubbing: bool = True,
                            include_subtitle: bool = False,
                            orig_volume: float = 0.35,
                            dub_volume: float = 1.0) -> Path:
    start_time = time.time()
    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video gốc không tồn tại: {video_path}")
    if len(segments) != len(translated_texts):
        raise ValueError("Số segments và bản dịch không khớp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Bắt đầu render cho {video_path}")

    if include_dubbing:
        if existing_tts is not None:
            tts_results = _normalize_existing_tts(existing_tts, segments, output_path.parent / "tts")
        elif existing_tts_results is not None:
            tts_results = list(existing_tts_results)
        else:
            tts_results = tts_mod.generate_tts_for_segments(
                segments=segments, translated_texts=translated_texts, voice=voice,
                provider=tts_provider, api_key=tts_api_key,
                output_dir=output_path.parent / "tts",
                max_workers=max_workers, rate=rate, pitch=pitch)
        tts_results = (list(tts_results) + [None] * len(segments))[:len(segments)]
        missing = _find_missing_tts(tts_results)
        logger.info(f"TTS hợp lệ: {len(segments) - len(missing)}/{len(segments)}")
        if missing:
            logger.warning("⚠️ Render partial: bỏ qua TTS segment thiếu/lỗi: %s", [i + 1 for i in missing[:50]])
        # Never reject the whole export because one or two clips are missing.
        # Natural-rate audio is kept as-is; the mixer simply skips missing entries.
        for i, result in enumerate(tts_results):
            if not result or not result.get("audio"):
                continue
            try:
                result["duration"] = tts_mod.get_audio_duration(str(result["audio"]))
            except Exception:
                result["duration"] = float(result.get("duration") or 0.0)
        # IMPORTANT: do not let adjacent natural-rate clips overlap. When a clip is
        # longer than the gap to the next source segment, we use only a small,
        # content-preserving atempo adjustment (max ~1.60x). We never trim the tail
        # and never use the old 2x-4x hard fit. This prevents the last 1-3 words from
        # being masked by the next segment while keeping subtitles/STT timing intact.
        tts_results = _prepare_dub_schedule(tts_results, segments, _get_video_duration(video_path))
    else:
        tts_results = []
        missing = []
    logger.info("Render: dubbing=%s, subtitles=%s", include_dubbing, include_subtitle)
    srt_path = output_path.parent / "subtitle.srt"
    _generate_srt(segments, translated_texts, srt_path, tts_results if include_dubbing else None)
    audio_final = output_path.parent / "final_audio.m4a"
    if include_dubbing:
        _compose_audio(video_path, segments, tts_results, audio_final, keep_original_audio,
                       orig_volume=orig_volume, dub_volume=dub_volume)
        merge_audio = audio_final
    else:
        merge_audio = video_path
    burn = bool(include_subtitle) if burn_subtitle is None else bool(burn_subtitle)
    _merge_audio_video_subtitle(video_path, merge_audio, srt_path, output_path, burn_subtitle=burn)
    _validate_rendered_video(output_path, expected_duration=_get_video_duration(video_path))
    logger.info(f"Render hoàn tất trong {time.time() - start_time:.2f}s: {output_path}")
    return output_path


def _audio_valid(value: Any) -> bool:
    try:
        p = Path(str(value))
        return p.exists() and p.stat().st_size > 500 and tts_mod.get_audio_duration(str(p)) > 0.05
    except Exception:
        return False


def _find_missing_tts(results: List[Optional[Dict[str, Any]]]) -> List[int]:
    return [i for i, r in enumerate(results) if not r or not r.get("audio") or not _audio_valid(r.get("audio"))]


def _normalize_existing_tts(existing_tts: Any, segments: List[Dict[str, Any]],
                            tts_dir: Path) -> List[Optional[Dict[str, Any]]]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(segments)
    if not existing_tts:
        return results
    items = existing_tts.items() if isinstance(existing_tts, dict) else enumerate(existing_tts)
    for key, value in items:
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(segments) or not isinstance(value, dict):
            continue
        raw_path = value.get("audio") or value.get("audio_path") or value.get("path")
        candidates = []
        if raw_path:
            candidates.append(Path(str(raw_path)))
        candidates.append(tts_dir / f"seg{idx:04d}.mp3")
        audio_path = next((p for p in candidates if _audio_valid(p)), None)
        if not audio_path:
            continue
        duration = float(value.get("duration") or tts_mod.get_audio_duration(str(audio_path)))
        slot = max(0.0, float(segments[idx].get("end", 0) or 0) - float(segments[idx].get("start", 0) or 0))
        # V22: never refit cached/generated speech to the source subtitle duration.
        # The source STT timeline remains unchanged; audio keeps the natural TTS rate.
        results[idx] = {"index": idx, "text": value.get("text", ""), "audio": str(audio_path),
                        "duration": float(duration or 0), "slot_duration": slot, "provider": value.get("provider"),
                        "voice": value.get("voice"), "cached": True,
                        "fit_verified": True, "timing_policy": "V22_NATURAL_RATE_ONLY"}
    return results


def _generate_srt(segments: List[Dict[str, Any]], texts: List[str], srt_path: Path,
                  tts_results: Optional[List[Optional[Dict[str, Any]]]] = None) -> None:
    """Write translated captions against the actual dub schedule when available.

    Source transcript timing remains immutable in transcript.srt. The translated
    caption track follows the scheduled TTS start/end so short OCR/STT cues do not
    make the translated sentence disappear while it is still being spoken.
    """
    texts = [str(t or "").strip() for t in texts]
    while len(texts) < len(segments):
        texts.append("")
    rows = []
    for i, source in enumerate(segments):
        seg = dict(source or {})
        result = (tts_results[i] if tts_results and i < len(tts_results) else None)
        if result and result.get("audio"):
            try:
                st = float(result.get("dub_start_sec", seg.get("start", 0)) or 0)
                en = float(result.get("dub_end_sec", st + float(result.get("duration") or 0)) or 0)
                if en > st:
                    seg["start"], seg["end"] = st, en
                    seg["source_start"], seg["source_end"] = st, en
                    seg["timing_locked"] = False
                    seg["ocr_locked"] = False
            except (TypeError, ValueError):
                pass
        rows.append((seg, texts[i]))
    lines = []
    for idx, (seg, text) in enumerate(rows, 1):
        try:
            st = max(0.0, float(seg.get("start", 0) or 0))
            en = max(st + 0.05, float(seg.get("end", st + 0.05) or st + 0.05))
        except (TypeError, ValueError):
            st, en = 0.0, 0.05
        lines.extend([str(idx), f"{subtitle_mod.fmt_srt_time(st)} --> {subtitle_mod.fmt_srt_time(en)}",
                      "\n".join(subtitle_mod.split_long_text(text)) if text else "", ""])
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Đã ghi SRT theo lịch lồng tiếng: %s (%d dòng)", srt_path, len(rows))


def _get_video_duration(video_path: Path) -> float:
    r = subprocess.run([config.FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                       capture_output=True, text=True, timeout=20)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"Không đọc được thời lượng video: {(r.stderr or '')[-500:]}")
    try:
        duration = float(r.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("Thời lượng video không hợp lệ") from exc
    if duration <= 0:
        raise RuntimeError("Thời lượng video <= 0")
    return duration


def _has_audio_stream(video_path: Path) -> bool:
    r = subprocess.run([config.FFPROBE_PATH, "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
                       capture_output=True, text=True, timeout=15)
    return r.returncode == 0 and bool(r.stdout.strip())


def _get_video_codec(video_path: Path) -> str:
    r = subprocess.run([config.FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                       capture_output=True, text=True, timeout=10)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip().lower()
    return "unknown"


def _atempo_chain(factor: float) -> str:
    """Build a pitch-preserving atempo chain for factor > 1."""
    factor = max(0.01, float(factor))
    stages = []
    while factor > 2.0:
        stages.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        stages.append(0.5)
        factor /= 0.5
    stages.append(factor)
    return ",".join(f"atempo={x:.6f}" for x in stages)


def _speed_audio_copy(audio: Path, factor: float, work_dir: Path, stem: str) -> Tuple[Path, float]:
    """Create a content-complete speed-adjusted copy; never trim."""
    factor = max(1.0, float(factor))
    if factor <= 1.0001:
        return audio, tts_mod.get_audio_duration(str(audio))
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{stem}.fit{factor:.3f}.mp3"
    if out.exists() and out.stat().st_size > 500:
        dur = tts_mod.get_audio_duration(str(out))
        if dur > 0.05:
            return out, dur
    tmp = out.with_suffix('.tmp.mp3')
    cmd = [config.FFMPEG_PATH, '-y', '-i', str(audio), '-filter:a', _atempo_chain(factor),
           '-c:a', 'libmp3lame', '-b:a', '128k', str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Không thể speed-fit TTS {audio.name}: {(r.stderr or '')[-800:]}")
    tmp.replace(out)
    dur = tts_mod.get_audio_duration(str(out))
    if dur <= 0.05:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"TTS fit tạo audio rỗng: {audio.name}")
    return out, dur


def _speech_anchor_start(seg: Dict[str, Any]) -> float:
    """Use the first word timestamp as the actual speech onset when available."""
    words = seg.get("words") or []
    starts = []
    if isinstance(words, list):
        for w in words:
            try:
                v = float(w.get("start", 0) or 0)
                if v >= 0:
                    starts.append(v)
            except (TypeError, ValueError, AttributeError):
                continue
    return min(starts) if starts else max(0.0, float(seg.get("start", 0) or 0))


def _speech_anchor_end(seg: Dict[str, Any]) -> float:
    """Use the last word timestamp as the spoken end when available."""
    words = seg.get("words") or []
    ends = []
    if isinstance(words, list):
        for w in words:
            try:
                v = float(w.get("end", 0) or 0)
                if v >= 0:
                    ends.append(v)
            except (TypeError, ValueError, AttributeError):
                continue
    return max(ends) if ends else max(_speech_anchor_start(seg), float(seg.get("end", 0) or 0))


def _prepare_dub_schedule(tts_results: List[Optional[Dict[str, Any]]],
                          segments: List[Dict[str, Any]], total_duration: float) -> List[Optional[Dict[str, Any]]]:
    """Speech-anchored dubbing schedule; never truncate generated speech.

    TTS is generated independently at the requested/natural rate. Here we only decide
    where the already-generated clip should start. Word-level STT timestamps are used
    as speech anchors when present. If a clip barely overruns the conversational window,
    a local pitch-preserving speed-up up to DUB_MAX_ATEMPO is allowed. Larger mismatches
    preserve the full audio and shift the following dubbing rather than cutting words.
    """
    scheduled = list(tts_results)
    valid_indices = [i for i, r in enumerate(scheduled)
                     if r and r.get('audio') and _audio_valid(r.get('audio'))]
    if not valid_indices:
        return scheduled

    work_dir = Path(config.OUTPUT_DIR) / '_render_tts_fit'
    prev_end = 0.0
    for pos, i in enumerate(valid_indices):
        seg = segments[i]
        desired_start = min(total_duration, _speech_anchor_start(seg))
        source_end = min(total_duration, max(desired_start, _speech_anchor_end(seg)))
        next_source_start = total_duration
        if pos + 1 < len(valid_indices):
            next_i = valid_indices[pos + 1]
            next_source_start = min(total_duration,
                                    max(desired_start + 0.001, _speech_anchor_start(segments[next_i])))

        raw_audio = Path(str(scheduled[i]['audio']))
        natural_dur = float(scheduled[i].get('duration') or tts_mod.get_audio_duration(str(raw_audio)) or 0)
        if natural_dur <= 0.05:
            continue

        # Use real pre-speech silence, when present, to start a little earlier.
        gap_before = max(0.0, desired_start - prev_end)
        early = min(DUB_EARLY_HEADROOM, max(0.0, gap_before - SEGMENT_SILENCE_GAP))
        actual_start = (max(0.0, desired_start - early)
                        if prev_end <= desired_start + 1e-6
                        else prev_end + SEGMENT_SILENCE_GAP)

        # The useful window extends to the next spoken onset, not to the current subtitle end.
        available = max(0.0, next_source_start - actual_start - SEGMENT_SILENCE_GAP)
        factor = 1.0
        audio_path = raw_audio
        final_dur = natural_dur
        if available > 0.05 and natural_dur > available:
            needed = natural_dur / available
            if needed <= DUB_MAX_ATEMPO:
                factor = needed
                audio_path, final_dur = _speed_audio_copy(raw_audio, factor, work_dir, f"seg{i:04d}")
                logger.info('[DUB-SOFT-FIT] seg%04d: %.2fs -> %.2fs | window=%.2fs | speed=%.2fx | full content',
                            i, natural_dur, final_dur, available, factor)
            else:
                logger.info('[DUB-NATURAL] seg%04d: %.2fs > window %.2fs; preserve full audio and shift following dub',
                            i, natural_dur, available)

        actual_end = min(total_duration, actual_start + final_dur)
        scheduled[i] = dict(scheduled[i], audio=str(audio_path), duration=final_dur,
                            natural_duration=natural_dur, source_speech_start=desired_start,
                            source_speech_end=source_end, dub_start_sec=actual_start,
                            dub_end_sec=actual_end, dub_speed=factor,
                            timing_policy='V32_SPEECH_ANCHORED_NATURAL')
        prev_end = actual_end
    return scheduled

def _chunk_cuts(events: List[Tuple[float, float, Path]], total_duration: float) -> List[Tuple[float, float]]:
    """Split filter graphs only at safe boundaries so no TTS clip crosses a chunk cut."""
    if not events:
        return [(0.0, total_duration)]
    cuts: List[Tuple[float, float]] = []
    n = max(1, AUDIO_CHUNK_SEGMENTS)
    i = 0
    a = 0.0
    while i < len(events):
        j = min(i + n, len(events))
        b = events[j - 1][0] + events[j - 1][1]
        # Include any following events whose starts fall before the candidate boundary.
        k = j
        while k < len(events) and events[k][0] < b - 1e-6:
            b = max(b, events[k][0] + events[k][1])
            k += 1
        b = min(total_duration, max(a + 0.05, b))
        cuts.append((a, b))
        a = b
        i = k
    if cuts[-1][1] < total_duration - 1e-6:
        cuts.append((cuts[-1][1], total_duration))
    return cuts

def _mix_range(video_path: Path, events: List[Tuple[float, float, Path]],
               range_start: float, range_end: float, keep_original: bool,
               output_path: Path, orig_volume: float = 0.35, dub_volume: float = 1.0) -> Path:
    span = max(0.05, range_end - range_start)
    inputs: List[str] = []
    filters: List[str] = []
    labels: List[str] = []
    filters.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{span:.6f},asetpts=PTS-STARTPTS[bed]")
    labels.append("[bed]")
    input_index = 0
    if keep_original:
        if not _has_audio_stream(video_path):
            raise RuntimeError("keep_original_audio=true nhưng video không có audio gốc")
        inputs += ["-ss", f"{range_start:.6f}", "-t", f"{span:.6f}", "-i", str(video_path)]
        filters.append(f"[{input_index}:a]aresample=44100,volume={max(0.0,float(orig_volume)):.4f}[orig]")
        labels.append("[orig]")
        input_index += 1
    for (start, seg_dur, audio) in events:
        delay_ms = int(round((start - range_start) * 1000))
        label = f"tts{input_index}"
        inputs += ["-i", str(audio)]
        # Never hard-trim the beginning/end of generated speech here. TTS fitting is done before render.
        # Keeping the full clip prevents the final words from being cut off.
        filters.append(f"[{input_index}:a]aresample=44100,volume={max(0.0,float(dub_volume)):.4f},asetpts=PTS-STARTPTS,"
                       f"adelay={delay_ms}|{delay_ms},asetpts=PTS-STARTPTS[{label}]")
        labels.append(f"[{label}]")
        input_index += 1
    joined = "".join(labels)
    filters.append(f"{joined}amix=inputs={len(labels)}:duration=first:dropout_transition=0:normalize=0,"
                   f"aformat=sample_rates=44100:channel_layouts=stereo,"
                   f"atrim=0:{span:.6f},asetpts=PTS-STARTPTS[out]")
    tmp = output_path.with_suffix(".tmp.m4a")
    # FIX WinError 206: ghi filter graph ra file thay vì command line
    filter_script = output_path.with_suffix(".filter.txt")
    filter_script.write_text(";".join(filters), encoding="utf-8")
    cmd = [config.FFMPEG_PATH, "-y", *inputs,
           "-filter_complex_script", str(filter_script),
           "-map", "[out]", "-vn", "-c:a", "aac", "-b:a", "160k",
           "-t", f"{span:.6f}", "-movflags", "+faststart", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg tạo audio tổng lỗi: {(r.stderr or '')[-1800:]}")
        tmp.replace(output_path)
    finally:
        filter_script.unlink(missing_ok=True)
    return output_path


def _concat_parts(parts: List[Path], output_path: Path) -> Path:
    list_file = output_path.with_suffix(".concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '" + str(p).replace("'", "'\\''") + "'\n")
    tmp = output_path.with_suffix(".tmp.m4a")
    try:
        r = subprocess.run([config.FFMPEG_PATH, "-y", "-f", "concat", "-safe", "0",
                            "-i", str(list_file), "-c", "copy",
                            "-movflags", "+faststart", str(tmp)],
                           capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
        if r.returncode != 0 or not tmp.exists():
            raise RuntimeError(f"FFmpeg concat audio lỗi: {(r.stderr or '')[-1800:]}")
        tmp.replace(output_path)
    finally:
        list_file.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return output_path


def _compose_audio(video_path: Path, segments: List[Dict[str, Any]],
                   tts_results: List[Dict[str, Any]], output_path: Path,
                   keep_original: bool, orig_volume: float = 0.35, dub_volume: float = 1.0) -> Path:
    total_duration = _get_video_duration(video_path)
    events: List[Tuple[float, float, Path]] = []
    for i, (seg, result) in enumerate(zip(segments, tts_results)):
        if not result or not result.get("audio"):
            logger.info("[RENDER-SKIP-TTS] segment %d: không có audio, giữ khoảng im lặng", i + 1)
            continue
        audio = Path(str(result["audio"]))
        if not _audio_valid(audio):
            logger.warning("[RENDER-SKIP-TTS] segment %d: audio không hợp lệ, bỏ qua", i + 1)
            continue
        start = max(0.0, float(result.get("dub_start_sec", seg.get("start", 0)) or 0))
        dur = float(result.get("duration") or tts_mod.get_audio_duration(str(audio)) or 0.0)
        if dur <= 0.001 or start >= total_duration:
            logger.warning(f"Bỏ qua TTS segment {i + 1} vì audio/timeline rỗng hoặc ngoài video")
            continue
        dur = min(dur, max(0.0, total_duration - start))
        events.append((start, dur, audio))
    if not events:
        logger.warning("⚠️ Không có TTS hợp lệ; vẫn xuất video với audio nền/im lặng theo cấu hình")
    events.sort(key=lambda e: e[0])
    if len(events) <= AUDIO_CHUNK_SEGMENTS:
        _mix_range(video_path, events, 0.0, total_duration, keep_original, output_path, orig_volume=orig_volume, dub_volume=dub_volume)
    else:
        chunk_paths: List[Path] = []
        for idx, (a, b) in enumerate(_chunk_cuts(events, total_duration)):
            part_events = [e for e in events if a - 1e-6 <= e[0] < b - 1e-6]
            part = output_path.with_suffix(f".part{idx:03d}.m4a")
            _mix_range(video_path, part_events, a, b, keep_original, part, orig_volume=orig_volume, dub_volume=dub_volume)
            chunk_paths.append(part)
        _concat_parts(chunk_paths, output_path)
        for p in chunk_paths:
            p.unlink(missing_ok=True)
    final_duration = tts_mod.get_audio_duration(str(output_path))
    if final_duration < max(0.1, total_duration - 0.10):
        raise RuntimeError(f"Audio tổng không đủ timeline: {final_duration:.2f}s / video {total_duration:.2f}s")
    logger.info(f"Final audio {final_duration:.2f}s (video {total_duration:.2f}s)")
    return output_path


def _merge_audio_video_subtitle(video_path: Path, audio_path: Path, srt_path: Path,
                                output_path: Path, burn_subtitle: bool = False) -> Path:
    duration = _get_video_duration(video_path)
    video_str = str(video_path).replace("\\", "/")
    audio_str = str(audio_path).replace("\\", "/")
    tmp = output_path.with_suffix(".render.tmp.mp4")
    if burn_subtitle:
        srt_str = str(srt_path).replace("\\", "/")
        srt_escaped = srt_str.replace(":", "\\:").replace("'", "\\'")
        vf = (f"subtitles='{srt_escaped}':"
              f"force_style='FontSize=24,PrimaryColour=&H00FFFFFF,"
              f"OutlineColour=&H00000000,Outline=2'")
        cmd = [config.FFMPEG_PATH, "-y", "-i", video_str, "-i", audio_str,
               "-map", "0:v:0", "-map", "1:a:0", "-vf", vf,
               "-t", f"{duration:.3f}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
               "-movflags", "+faststart", str(tmp)]
    else:
        codec = _get_video_codec(video_path)
        if codec in _COPY_VIDEO_CODECS:
            cmd = [config.FFMPEG_PATH, "-y", "-i", video_str, "-i", audio_str,
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "copy", "-c:a", "copy",
                   "-t", f"{duration:.3f}", "-movflags", "+faststart", str(tmp)]
        else:
            logger.warning(f"Codec video {codec} không hỗ trợ copy vào mp4, re-encode")
            cmd = [config.FFMPEG_PATH, "-y", "-i", video_str, "-i", audio_str,
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-t", f"{duration:.3f}",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                   "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg merge lỗi: {(r.stderr or '')[-1800:]}")
    if not tmp.exists() or tmp.stat().st_size < 1024:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg không tạo được video đầu ra")
    tmp.replace(output_path)
    mode = "burn subtitle" if burn_subtitle else "phụ đề rời (copy video)"
    logger.info(f"FFmpeg merge thành công ({mode})")
    return output_path


def _validate_rendered_video(output_path: Path, expected_duration: Optional[float] = None) -> None:
    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise RuntimeError("Video đầu ra không tồn tại hoặc quá nhỏ")
    r = subprocess.run([config.FFPROBE_PATH, "-v", "error",
                        "-show_entries", "stream=codec_type,codec_name",
                        "-of", "csv=p=0", str(output_path)],
                       capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        raise RuntimeError(f"Không kiểm tra được video đầu ra: {(r.stderr or '')[-500:]}")
    streams = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    tokens = set()
    for line in streams:
        for part in line.split(","):
            part = part.strip()
            if part:
                tokens.add(part)
    has_video = "video" in tokens
    has_audio = "audio" in tokens
    if not has_video or not has_audio:
        raise RuntimeError(f"Video đầu ra thiếu stream (video={has_video}, audio={has_audio}): {streams}")
    if expected_duration:
        d = tts_mod.get_audio_duration(str(output_path))
        if d <= 0 or abs(d - expected_duration) > max(1.0, expected_duration * 0.02):
            raise RuntimeError(f"Video đầu ra sai thời lượng: {d:.2f}s (kỳ vọng {expected_duration:.2f}s)")


__all__ = ["render_final_video_fast"]