"""Video file integrity and stream validation module."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Union

import cv2

from .config import SUPPORTED_EXTENSIONS

logger = logging.getLogger("module1.validation")


@dataclass
class ValidationResult:
    """Detailed outcome of validating a video file."""

    is_valid: bool
    error_message: Optional[str] = None
    has_video_stream: bool = False
    has_audio_stream: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_seconds: float = 0.0


def check_audio_stream_exists(
    video_path: Path, ffprobe_path: str = "ffprobe"
) -> bool:
    """Use ffprobe to check if the video file contains at least one audio stream.

    Args:
        video_path: Path to the video file.
        ffprobe_path: Executable name or full path to ffprobe.

    Returns:
        True if an audio stream is detected, False otherwise.
    """
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index,codec_name",
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
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout:
            data = json.loads(proc.stdout)
            streams = data.get("streams", [])
            return len(streams) > 0
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
        logger.debug(
            "ffprobe audio check failed for %s: %s (defaulting to False)",
            video_path.name,
            e,
        )
    return False


def validate_video(
    video_path: Union[str, Path],
    ffprobe_path: str = "ffprobe",
    supported_extensions: Optional[Set[str]] = None,
) -> ValidationResult:
    """Perform comprehensive validation on a video file.

    Verifies existence, non-empty size, supported file extension, and readable
    video stream via OpenCV. Detects presence of an audio stream via ffprobe.

    Args:
        video_path: Target video file path.
        ffprobe_path: Path to ffprobe binary.
        supported_extensions: Set of supported extensions (defaults to {".mp4", ".mkv"}).

    Returns:
        ValidationResult with validation status and stream metrics.
    """
    path = Path(video_path).resolve()
    valid_exts = {
        ext.lower() for ext in (supported_extensions or SUPPORTED_EXTENSIONS)
    }

    if not path.exists():
        err = f"File does not exist: {path}"
        logger.error(err)
        return ValidationResult(is_valid=False, error_message=err)

    if not path.is_file():
        err = f"Path is not a regular file: {path}"
        logger.error(err)
        return ValidationResult(is_valid=False, error_message=err)

    if path.stat().st_size == 0:
        err = f"File is empty (0 bytes): {path}"
        logger.error(err)
        return ValidationResult(is_valid=False, error_message=err)

    if path.suffix.lower() not in valid_exts:
        err = f"Unsupported extension '{path.suffix}'. Supported extensions: {valid_exts}"
        logger.error(err)
        return ValidationResult(is_valid=False, error_message=err)

    # OpenCV VideoCapture validation
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            err = f"OpenCV failed to open video container: {path.name}"
            logger.error(err)
            return ValidationResult(is_valid=False, error_message=err)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            err = f"Invalid video resolution ({width}x{height}) in {path.name}"
            logger.error(err)
            return ValidationResult(is_valid=False, error_message=err)

        # Attempt to decode the very first frame to confirm stream integrity
        ret, frame = cap.read()
        if not ret or frame is None:
            err = f"Failed to decode initial frame; file may be corrupt: {path.name}"
            logger.error(err)
            return ValidationResult(is_valid=False, error_message=err)

        duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0

        # Check for audio stream
        has_audio = check_audio_stream_exists(path, ffprobe_path=ffprobe_path)

        logger.info(
            "Video validation successful: %s (%dx%d, %.2f fps, %d frames, %.2fs, audio=%s)",
            path.name,
            width,
            height,
            fps,
            frame_count,
            duration,
            has_audio,
        )

        return ValidationResult(
            is_valid=True,
            has_video_stream=True,
            has_audio_stream=has_audio,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )

    except Exception as exc:
        err = f"Unexpected error while validating {path.name}: {exc}"
        logger.error(err, exc_info=True)
        return ValidationResult(is_valid=False, error_message=err)

    finally:
        if cap is not None:
            cap.release()
