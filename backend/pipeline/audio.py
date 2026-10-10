"""pipeline/audio.py - Extract audio, get video info, download from URL."""
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import config

logger = logging.getLogger(__name__)

EXTRACT_TIMEOUT = getattr(config, 'AUDIO_EXTRACT_TIMEOUT', 600)
PROBE_TIMEOUT = getattr(config, 'AUDIO_PROBE_TIMEOUT', 30)


def extract_audio(video_path: Path, output_path: Path,
                  sample_rate: int = 16000, channels: int = 1,
                  format: str = "wav") -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video không tồn tại: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Tách audio từ: {video_path} -> {output_path}")
    try:
        subprocess.run([config.FFMPEG_PATH, "-version"], capture_output=True, check=True)
    except Exception:
        raise RuntimeError("FFmpeg không tìm thấy. Vui lòng cài FFmpeg và thêm vào PATH.")
    # Báo rõ khi video không có track âm thanh, thay vì để ffmpeg báo lỗi khó hiểu
    # vì `-map 0:a:0` không tìm thấy stream.
    if not has_audio_stream(video_path):
        raise RuntimeError(
            "Video này không có track âm thanh nên không thể nhận diện lời thoại. "
            "Nếu phim có phụ đề cháy sẵn trên hình, hãy chuyển sang chế độ Phụ đề cứng (OCR).")
    try:
        # FIX TIMELINE #1: giây 0 của WAV phải là giây 0 của VIDEO.
        # Nhiều container (MKV/TS/MP4 có edit list, AAC priming delay) có
        # audio stream start_time != 0. Nếu không xử lý, FFmpeg bỏ phần lệch đầu
        # và toàn bộ mốc STT bị sớm đi đúng bằng lượng lệch đó.
        #   -map 0:a:0                          : chỉ lấy audio track đầu, không đoán
        #   aresample=async=1:first_pts=0        : chèn im lặng cho phần lệch đầu,
        #                                          và bù các khoảng trống packet giữa file
        cmd = [
            config.FFMPEG_PATH, "-i", str(video_path), "-vn",
            "-map", "0:a:0",
            "-af", "aresample=async=1:first_pts=0",
            "-acodec", "pcm_s16le" if format == "wav" else "libmp3lame",
            "-ar", str(sample_rate), "-ac", str(channels),
            "-threads", "2", "-y", str(output_path),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=EXTRACT_TIMEOUT)
        if not output_path.exists():
            raise RuntimeError("FFmpeg không tạo file output")
        file_size = output_path.stat().st_size
        logger.info(f"Tách audio thành công: {file_size} bytes")
        verify_audio_timeline(video_path, output_path)
        return output_path
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Tách audio timeout sau {EXTRACT_TIMEOUT}s")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg lỗi: {e.stderr}")
        raise RuntimeError(f"Tách audio thất bại: {e.stderr[:200]}")
    except Exception as e:
        logger.error(f"Lỗi tách audio: {e}")
        raise


def has_audio_stream(video_path: Path) -> bool:
    """Video có track âm thanh hay không."""
    try:
        r = subprocess.run([config.FFPROBE_PATH, "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
                           capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        return r.returncode == 0 and bool((r.stdout or "").strip())
    except Exception:
        return True  # không chặn pipeline nếu ffprobe lỗi


def probe_duration(media_path: Path) -> float:
    """Đọc thời lượng bằng ffprobe; trả 0.0 nếu không đọc được."""
    try:
        r = subprocess.run([config.FFPROBE_PATH, "-v", "error", "-show_entries",
                            "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", str(media_path)],
                           capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        return float((r.stdout or "0").strip())
    except Exception:
        return 0.0


def audio_stream_start_time(video_path: Path) -> float:
    """start_time của audio track đầu tiên (lượng lệch mốc trong container)."""
    try:
        r = subprocess.run([config.FFPROBE_PATH, "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=start_time", "-of",
                            "default=noprint_wrappers=1:nokey=1", str(video_path)],
                           capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        return float((r.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


def verify_audio_timeline(video_path: Path, audio_path: Path,
                          max_drift: float = 0.30) -> Dict[str, float]:
    """Kiểm tra WAV tách ra có cùng trục thời gian với video hay không.

    Không raise: chỉ ghi log cảnh báo kèm số đo, để pipeline không chết vì file
    có mốc hơi lệch, nhưng lệch lớn thì không còn im lặng như trước.
    """
    video_duration = probe_duration(video_path)
    audio_duration = probe_duration(audio_path)
    offset = audio_stream_start_time(video_path)
    drift = abs(video_duration - audio_duration) if (video_duration and audio_duration) else 0.0
    report = {"video_duration": video_duration, "audio_duration": audio_duration,
              "audio_start_time": offset, "drift": drift}
    if offset > 0.05:
        logger.info(f"Audio track lệch mốc container {offset:.3f}s → đã bù im lặng để giữ trục video")
    if drift > max_drift:
        logger.warning(
            f"CẢNH BÁO TIMELIND: audio {audio_duration:.3f}s vs video {video_duration:.3f}s "
            f"(lệch {drift:.3f}s). Mốc phụ đề có thể không khớp hình.")
    else:
        logger.info(f"Timeline audio khớp video (lệch {drift:.3f}s)")
    return report


def get_video_info(video_path: Path) -> Dict[str, Any]:
    if not video_path.exists():
        return {"exists": False}
    try:
        import json
        cmd = [config.FFPROBE_PATH, "-v", "quiet", "-print_format", "json",
               "-show_format", "-show_streams", str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        if result.returncode != 0:
            return {"exists": True, "error": "ffprobe failed"}
        data = json.loads(result.stdout)
        format_info = data.get("format", {})
        streams = data.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        info = {
            "exists": True,
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
            "size_mb": int(format_info.get("size", 0)) / (1024 * 1024),
            "format": format_info.get("format_name", "unknown"),
            "bitrate": int(format_info.get("bit_rate", 0)),
        }
        if video_stream:
            fps_str = video_stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else None
            except (ValueError, ZeroDivisionError):
                fps = None
            info["video"] = {
                "codec": video_stream.get("codec_name", "unknown"),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": fps,
            }
        if audio_stream:
            info["audio"] = {
                "codec": audio_stream.get("codec_name", "unknown"),
                "channels": int(audio_stream.get("channels", 0)),
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
            }
        logger.info(f"Video info: duration={info['duration']:.2f}s, size={info['size_mb']:.1f}MB")
        return info
    except Exception as e:
        logger.error(f"Lỗi lấy video info: {e}")
        return {"exists": True, "error": str(e)}


def download_from_url(url: str, output_path: Path, progress_cb=None) -> Path:
    import yt_dlp
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download từ URL: {url}")

    def progress_hook(d):
        if d["status"] == "downloading":
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes", 0)
            if total > 0 and progress_cb:
                progress_cb((downloaded / total) * 100)
        elif d["status"] == "finished":
            logger.info(f"Download hoàn tất: {d.get('filename')}")

    ydl_opts = {
        "outtmpl": str(output_path.with_suffix("")) + ".%(ext)s",
        "format": "bv*[ext=mp4]+ba*[ext=m4a]/b[ext=mp4]/b/worst",
        "progress_hooks": [progress_hook] if progress_cb else [],
        "quiet": True, "no_warnings": True, "extract_flat": False,
        "merge_output_format": "mp4", "socket_timeout": 60, "retries": 10,
        "fragment_retries": 10, "file_access_retries": 5,
        "noplaylist": True, "windowsfilenames": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        possible_files = [p for p in output_path.parent.glob(f"{output_path.stem}.*")
                          if p.is_file() and p.name != output_path.name and p.suffix.lower() in {".mp4",".mkv",".webm",".mov",".m4v"}]
        if possible_files:
            downloaded_file = max(possible_files, key=lambda p: p.stat().st_size)
            if downloaded_file != output_path:
                if output_path.exists():
                    output_path.unlink()
                downloaded_file.rename(output_path)
        if not output_path.exists():
            raise RuntimeError("Không tìm thấy file đã download")
        logger.info(f"Download thành công: {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Không thể download video: {e}")