"""Audio track extraction using FFmpeg (extracted once per video)."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .config import (
    DEFAULT_AUDIO_CHANNELS,
    DEFAULT_AUDIO_CODEC,
    DEFAULT_AUDIO_SAMPLE_RATE,
)
from .video_validation import check_audio_stream_exists

logger = logging.getLogger("module1.audio")


@dataclass
class AudioExtractionResult:
    """Outcome of extracting audio from a video."""

    has_audio: bool
    audio_path: Optional[str] = None
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    channels: int = DEFAULT_AUDIO_CHANNELS
    file_size_bytes: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return asdict(self)


def extract_audio(
    video_path: Union[str, Path],
    video_id: str,
    output_audio_dir: Union[str, Path],
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    channels: int = DEFAULT_AUDIO_CHANNELS,
    codec: str = DEFAULT_AUDIO_CODEC,
    known_has_audio: Optional[bool] = None,
) -> AudioExtractionResult:
    """Extract audio track from video file exactly once using FFmpeg.

    Converts the primary audio track into standard 16kHz mono WAV suitable
    for WhisperX transcription. If the video does not contain an audio track,
    gracefully skips extraction without raising an error.

    Args:
        video_path: Path to source video file.
        video_id: Deterministic video identifier.
        output_audio_dir: Target directory for WAV files (e.g. output/audio).
        ffmpeg_path: Path or command name for FFmpeg binary.
        ffprobe_path: Path or command name for FFprobe binary.
        sample_rate: Audio sampling rate in Hz (default: 16000).
        channels: Number of channels (default: 1 for mono).
        codec: Audio codec (default: pcm_s16le).
        known_has_audio: Optional pre-validated boolean indicating audio presence.

    Returns:
        AudioExtractionResult detailing extracted audio path or absence of audio.
    """
    v_path = Path(video_path).resolve()
    target_dir = Path(output_audio_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_wav = target_dir / f"{video_id}.wav"

    # Check for presence of audio track
    has_audio = (
        known_has_audio
        if known_has_audio is not None
        else check_audio_stream_exists(v_path, ffprobe_path=ffprobe_path)
    )

    if not has_audio:
        logger.info(
            "Video %s (%s) has no audio track. Gracefully skipping audio extraction.",
            v_path.name,
            video_id,
        )
        return AudioExtractionResult(
            has_audio=False,
            audio_path=None,
            sample_rate=sample_rate,
            channels=channels,
            file_size_bytes=0,
        )

    logger.info(
        "Extracting audio track for %s (id=%s) -> %s",
        v_path.name,
        video_id,
        target_wav.name,
    )

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(v_path),
        "-vn",
        "-acodec",
        codec,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(target_wav),
    ]

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
            err = (
                f"FFmpeg audio extraction failed for {v_path.name}: "
                f"{proc.stderr.strip() or 'Unknown error'}"
            )
            logger.error(err)
            return AudioExtractionResult(
                has_audio=True, audio_path=None, error_message=err
            )

        if not target_wav.exists() or target_wav.stat().st_size == 0:
            err = f"Extracted audio file missing or 0 bytes: {target_wav}"
            logger.error(err)
            return AudioExtractionResult(
                has_audio=True, audio_path=None, error_message=err
            )

        file_size = target_wav.stat().st_size
        logger.info(
            "Audio extracted successfully: %s (size=%d bytes, rate=%dHz, channels=%d)",
            target_wav.name,
            file_size,
            sample_rate,
            channels,
        )

        return AudioExtractionResult(
            has_audio=True,
            audio_path=str(target_wav.resolve()),
            sample_rate=sample_rate,
            channels=channels,
            file_size_bytes=file_size,
        )

    except Exception as exc:
        err = f"Exception running FFmpeg on {v_path.name}: {exc}"
        logger.error(err, exc_info=True)
        return AudioExtractionResult(
            has_audio=True, audio_path=None, error_message=err
        )
