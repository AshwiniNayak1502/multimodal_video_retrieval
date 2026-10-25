"""Playback proxy generator ensuring browser compatibility (MP4/H.264/AAC faststart)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..settings import settings

logger = logging.getLogger("app.playback")


def get_video_stream_info(video_path: Path | str, ffprobe_path: str = "ffprobe") -> dict:
    """Introspect video and audio streams using ffprobe."""
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=15,
        )
        if proc.returncode == 0 and proc.stdout:
            data = json.loads(proc.stdout)
            streams = data.get("streams", [])
            fmt = data.get("format", {})
            v_codec = next((s.get("codec_name") for s in streams if s.get("codec_type") == "video"), None)
            a_codec = next((s.get("codec_name") for s in streams if s.get("codec_type") == "audio"), None)
            duration = float(fmt.get("duration", 0.0) or 0.0)
            return {
                "video_codec": v_codec,
                "audio_codec": a_codec,
                "duration": duration,
            }
    except Exception as exc:
        logger.warning("ffprobe stream introspection failed for %s: %s", video_path, exc)

    return {"video_codec": None, "audio_codec": None, "duration": 0.0}


def ensure_playback_proxy(
    source_video_path: Path | str,
    video_id: str,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
) -> Path:
    """Ensure a fast-start, web-compatible MP4 file is available for browser playback.

    If the source file is already an MP4 with H.264 video and AAC/no audio, it can be
    served directly or remuxed with faststart. For MKV or non-web codecs, transcode
    to a cached MP4 under data/playback/<video_id>.mp4.
    """
    src = Path(source_video_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source video file not found: {src}")

    settings.ensure_directories()
    dest = (settings.playback_dir / f"{video_id}.mp4").resolve()

    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    info = get_video_stream_info(src, ffprobe_path=ffprobe_path)
    v_codec = info.get("video_codec")
    a_codec = info.get("audio_codec")

    # If already mp4 with h264 and aac, we can stream copy to faststart mp4 quickly
    can_stream_copy = (
        src.suffix.lower() == ".mp4"
        and v_codec in ("h264", "avc1")
        and (a_codec in ("aac", None))
    )

    cmd = [ffmpeg_path, "-y", "-i", str(src)]
    if can_stream_copy:
        cmd.extend(["-c", "copy", "-movflags", "+faststart", str(dest)])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(dest),
        ])

    logger.info("Generating playback proxy for %s (%s) -> %s...", src.name, video_id, dest.name)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=120,
        )
        if proc.returncode != 0:
            err = proc.stderr[-500:] if proc.stderr else "Unknown ffmpeg error"
            logger.error("FFmpeg playback proxy generation failed: %s", err)
            # If transcoding failed but original exists, return original as fallback
            return src
    except Exception as exc:
        logger.error("FFmpeg execution error: %s", exc)
        return src

    # Verify duration matches within 0.1s
    if dest.is_file() and dest.stat().st_size > 0:
        dest_info = get_video_stream_info(dest, ffprobe_path=ffprobe_path)
        orig_dur = info.get("duration", 0.0)
        dest_dur = dest_info.get("duration", 0.0)
        if orig_dur > 0 and dest_dur > 0 and abs(orig_dur - dest_dur) > 0.5:
            logger.warning(
                "Playback proxy duration discrepancy for %s: orig=%.2fs, proxy=%.2fs",
                video_id,
                orig_dur,
                dest_dur,
            )
        return dest

    return src
