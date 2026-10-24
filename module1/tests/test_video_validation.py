"""Unit tests for video validation and integrity verification."""

from pathlib import Path

from src.video_validation import validate_video


def test_validate_valid_video_with_audio(sample_video_with_audio: Path):
    """A valid MP4 video with audio passes validation completely."""
    result = validate_video(sample_video_with_audio)
    assert result.is_valid is True
    assert result.has_video_stream is True
    assert result.has_audio_stream is True
    assert result.width > 0
    assert result.height > 0
    assert result.fps == 25.0
    assert result.duration_seconds > 0.0
    assert result.error_message is None


def test_validate_video_without_audio(sample_video_no_audio: Path):
    """A valid MKV video without audio passes validation and flags no audio."""
    result = validate_video(sample_video_no_audio)
    assert result.is_valid is True
    assert result.has_video_stream is True
    assert result.has_audio_stream is False
    assert result.width == 320
    assert result.height == 240
    assert result.error_message is None


def test_validate_corrupt_video_fails_gracefully(sample_corrupt_video: Path):
    """A corrupt video file fails validation with a clear error without crashing."""
    result = validate_video(sample_corrupt_video)
    assert result.is_valid is False
    assert result.error_message is not None
    assert "corrupt" in result.error_message.lower() or "failed" in result.error_message.lower()


def test_validate_zero_byte_file(tmp_path: Path):
    """A 0-byte file fails validation immediately with an empty file error."""
    empty = tmp_path / "zero.mp4"
    empty.touch()
    result = validate_video(empty)
    assert result.is_valid is False
    assert result.error_message is not None
    assert "empty" in result.error_message.lower()
