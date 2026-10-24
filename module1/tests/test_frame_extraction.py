"""Unit tests for time-based representative frame extraction."""

from pathlib import Path

import cv2
import pytest

from src.frame_extraction import extract_frames
from src.id_generation import generate_video_id


def test_frame_sampling_interval_and_timestamps(
    sample_video_with_audio: Path, tmp_path: Path
):
    """Verifies that frames are sampled at expected time intervals (~0s, 2s) with exact timestamps."""
    video_id = generate_video_id(sample_video_with_audio)
    frames_dir = tmp_path / "frames"
    interval = 2.0

    records = extract_frames(
        video_path=sample_video_with_audio,
        video_id=video_id,
        output_dir=frames_dir,
        interval_seconds=interval,
    )

    # For a 4.0-second video at 2.0s interval, we expect at least 2 frames (t=0.0s, t=2.0s)
    assert len(records) >= 2

    # Check first frame is close to 0.0s
    assert abs(records[0].timestamp_seconds - 0.0) < 0.1
    # Check second frame is close to 2.0s
    assert abs(records[1].timestamp_seconds - 2.0) < 0.25

    # Check file paths exist on disk and are valid JPEG images
    for rec in records:
        f_path = Path(rec.file_path)
        assert f_path.exists()
        assert f_path.stat().st_size > 0
        img = cv2.imread(str(f_path))
        assert img is not None
        assert img.shape[0] == rec.height
        assert img.shape[1] == rec.width
        assert rec.width > 0
        assert rec.height > 0


def test_frame_extraction_invalid_interval(
    sample_video_with_audio: Path, tmp_path: Path
):
    """Non-positive interval must raise ValueError."""
    video_id = generate_video_id(sample_video_with_audio)
    with pytest.raises(ValueError):
        extract_frames(
            video_path=sample_video_with_audio,
            video_id=video_id,
            output_dir=tmp_path / "frames",
            interval_seconds=0.0,
        )
