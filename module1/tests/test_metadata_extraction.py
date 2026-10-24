"""Unit tests for metadata extraction."""

from pathlib import Path

from src.id_generation import generate_video_id
from src.metadata_extraction import extract_video_metadata


def test_metadata_extraction_with_audio(sample_video_with_audio: Path):
    """Extracts complete metadata for a video with audio."""
    video_id = generate_video_id(sample_video_with_audio)
    meta = extract_video_metadata(sample_video_with_audio, video_id=video_id)

    assert meta.video_id == video_id
    assert meta.file_name == sample_video_with_audio.name
    assert meta.file_size_bytes > 0
    assert meta.width > 0
    assert meta.height > 0
    assert meta.fps == 25.0
    assert meta.duration_seconds > 0.0
    assert meta.has_audio is True
    assert meta.audio_codec in ("aac", "pcm_s16le")
    assert meta.video_codec in ("h264", "unknown")


def test_metadata_extraction_no_audio(sample_video_no_audio: Path):
    """Extracts metadata for video without audio and accurately reflects has_audio=False."""
    video_id = generate_video_id(sample_video_no_audio)
    meta = extract_video_metadata(sample_video_no_audio, video_id=video_id)

    assert meta.video_id == video_id
    assert meta.width == 320
    assert meta.height == 240
    assert meta.has_audio is False
    assert meta.audio_codec is None
