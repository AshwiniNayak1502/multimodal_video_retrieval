"""Metadata extraction module for video files."""

from __future__ import annotations

import datetime
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2

logger = logging.getLogger("module1.metadata")


@dataclass
class VideoMetadata:
    """Standardized metadata representation for an ingested video."""

    video_id: str
    file_name: str
    file_path: str
    file_size_bytes: int
    file_extension: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int
    video_codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_channels: Optional[int] = None
    audio_sample_rate: Optional[int] = None
    modified_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to JSON-serializable dictionary."""
        return asdict(self)


def _probe_file_streams(
    video_path: Path, ffprobe_path: str = "ffprobe"
) -> dict[str, Any]:
    """Run ffprobe to gather full stream and format metadata."""
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
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
            return json.loads(proc.stdout)
    except Exception as exc:
        logger.debug("ffprobe failed on %s: %s", video_path.name, exc)
    return {}


def extract_video_metadata(
    video_path: Union[str, Path],
    video_id: str,
    ffprobe_path: str = "ffprobe",
) -> VideoMetadata:
    """Extract complete technical metadata from a video file.

    Combines ffprobe stream introspection with OpenCV container metrics.

    Args:
        video_path: Target video file path.
        video_id: Deterministic video ID.
        ffprobe_path: Executable path for ffprobe.

    Returns:
        Structured VideoMetadata instance.
    """
    path = Path(video_path).resolve()
    stat_info = path.stat()
    file_size = stat_info.st_size
    modified_iso = datetime.datetime.fromtimestamp(
        stat_info.st_mtime, tz=datetime.timezone.utc
    ).isoformat()

    # Query ffprobe
    probe_data = _probe_file_streams(path, ffprobe_path=ffprobe_path)
    streams = probe_data.get("streams", [])
    fmt = probe_data.get("format", {})

    video_stream: dict[str, Any] = {}
    audio_stream: dict[str, Any] = {}

    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type == "video" and not video_stream:
            video_stream = s
        elif codec_type == "audio" and not audio_stream:
            audio_stream = s

    # Base OpenCV extraction
    width = 0
    height = 0
    fps = 0.0
    frame_count = 0
    duration_cv2 = 0.0

    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps > 0 and frame_count > 0:
            duration_cv2 = frame_count / fps
        cap.release()

    # Prefer ffprobe values when available
    if video_stream:
        width = int(video_stream.get("width", width))
        height = int(video_stream.get("height", height))
        video_codec = video_stream.get("codec_name", "unknown")
    else:
        video_codec = "unknown"

    duration = 0.0
    if "duration" in fmt:
        try:
            duration = float(fmt["duration"])
        except (ValueError, TypeError):
            duration = duration_cv2
    elif video_stream and "duration" in video_stream:
        try:
            duration = float(video_stream["duration"])
        except (ValueError, TypeError):
            duration = duration_cv2
    else:
        duration = duration_cv2

    has_audio = bool(audio_stream)
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    audio_channels = (
        int(audio_stream["channels"])
        if audio_stream and "channels" in audio_stream
        else None
    )
    audio_sample_rate = (
        int(audio_stream["sample_rate"])
        if audio_stream and "sample_rate" in audio_stream
        else None
    )

    metadata = VideoMetadata(
        video_id=video_id,
        file_name=path.name,
        file_path=str(path),
        file_size_bytes=file_size,
        file_extension=path.suffix.lower(),
        duration_seconds=round(duration, 3),
        width=width,
        height=height,
        fps=round(fps, 3),
        frame_count=frame_count,
        video_codec=video_codec,
        has_audio=has_audio,
        audio_codec=audio_codec,
        audio_channels=audio_channels,
        audio_sample_rate=audio_sample_rate,
        modified_at=modified_iso,
    )

    logger.info(
        "Extracted metadata for %s: duration=%.2fs, resolution=%dx%d, fps=%.2f, audio=%s (%s)",
        video_id,
        metadata.duration_seconds,
        metadata.width,
        metadata.height,
        metadata.fps,
        metadata.has_audio,
        metadata.audio_codec or "none",
    )

    return metadata
