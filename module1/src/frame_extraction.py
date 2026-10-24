"""Representative frame extraction using OpenCV via time-based sampling."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Union

import cv2

from .config import DEFAULT_FRAME_INTERVAL_SECONDS

logger = logging.getLogger("module1.frames")


@dataclass
class FrameRecord:
    """Metadata for an extracted representative video frame."""

    video_id: str
    frame_index: int
    source_frame_number: int
    timestamp_seconds: float
    file_path: str
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return asdict(self)


def extract_frames(
    video_path: Union[str, Path],
    video_id: str,
    output_dir: Union[str, Path],
    interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS,
) -> List[FrameRecord]:
    """Extract representative frames from a video file at a configurable time interval.

    Performs time-based sampling using OpenCV (e.g. ~0s, 2s, 4s, 6s...), writes
    frames to disk under `output_dir/<video_id>/`, and records exact timestamps.

    Args:
        video_path: Path to the source video.
        video_id: Deterministic video ID for directory naming and records.
        output_dir: Parent directory for frames (e.g. output/frames).
        interval_seconds: Sampling step in seconds. Must be > 0.

    Returns:
        List of FrameRecord objects containing exact timestamps and file paths.

    Raises:
        ValueError: If interval_seconds <= 0 or video cannot be opened.
    """
    if interval_seconds <= 0:
        raise ValueError(
            f"interval_seconds must be positive, got {interval_seconds}"
        )

    path = Path(video_path).resolve()
    target_frames_dir = Path(output_dir) / video_id
    target_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file with OpenCV: {path}")

    records: List[FrameRecord] = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            fps = 25.0
            logger.warning(
                "FPS reported as 0 or negative for %s; defaulting to 25.0",
                path.name,
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (
            (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0
        )

        logger.info(
            "Extracting frames for %s (id=%s) at interval=%.2fs (duration=%.2fs, fps=%.2f)",
            path.name,
            video_id,
            interval_seconds,
            duration,
            fps,
        )

        # Build list of target sample times in seconds: 0.0, interval, 2*interval...
        target_timestamps: list[float] = [0.0]
        if duration > 0:
            current_t = interval_seconds
            while current_t < duration:
                target_timestamps.append(current_t)
                current_t += interval_seconds

        sample_idx = 0
        for target_sec in target_timestamps:
            target_frame_num = int(round(target_sec * fps))
            if total_frames > 0 and target_frame_num >= total_frames:
                target_frame_num = max(0, total_frames - 1)

            # Seek by frame number
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
            ret, frame = cap.read()

            if not ret or frame is None:
                # Fallback to msec seek
                cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
                ret, frame = cap.read()

            if not ret or frame is None:
                logger.debug(
                    "Could not decode frame at target time %.2fs (frame %d)",
                    target_sec,
                    target_frame_num,
                )
                continue

            # Determine true timestamp from decoder
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            read_frame_num = max(0, current_frame_pos - 1)

            if pos_msec > 0:
                true_timestamp = pos_msec / 1000.0
            else:
                true_timestamp = (
                    (read_frame_num / fps) if fps > 0 else target_sec
                )

            true_timestamp = round(true_timestamp, 3)

            # Format filename with sample index and exact timestamp
            frame_filename = (
                f"frame_{sample_idx:04d}_{true_timestamp:.3f}s.jpg"
            )
            frame_path = target_frames_dir / frame_filename

            # Save JPEG frame
            success = cv2.imwrite(
                str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if not success:
                logger.error("Failed to write frame image to %s", frame_path)
                continue

            h, w = frame.shape[:2]
            record = FrameRecord(
                video_id=video_id,
                frame_index=sample_idx,
                source_frame_number=read_frame_num,
                timestamp_seconds=true_timestamp,
                file_path=str(frame_path.resolve()),
                width=w,
                height=h,
            )
            records.append(record)
            sample_idx += 1

        logger.info(
            "Extracted %d frames for video %s to %s",
            len(records),
            video_id,
            target_frames_dir,
        )
        return records

    finally:
        cap.release()
