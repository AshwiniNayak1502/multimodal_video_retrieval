"""Unit tests for deterministic video ID generation."""

import re
from pathlib import Path

import pytest

from src.id_generation import generate_video_id


def test_deterministic_id_consistency(sample_video_with_audio: Path):
    """Running ID generation twice on the same file produces identical IDs."""
    id1 = generate_video_id(sample_video_with_audio)
    id2 = generate_video_id(sample_video_with_audio)
    assert id1 == id2
    assert re.match(r"^vid_[0-9a-f]{16}$", id1)


def test_different_files_produce_different_ids(
    sample_video_with_audio: Path, sample_video_no_audio: Path
):
    """Different video files must have distinct IDs."""
    id1 = generate_video_id(sample_video_with_audio)
    id2 = generate_video_id(sample_video_no_audio)
    assert id1 != id2


def test_nonexistent_file_raises_error(tmp_path: Path):
    """Passing a non-existent file path must raise FileNotFoundError."""
    missing_path = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        generate_video_id(missing_path)


def test_empty_file_raises_error(tmp_path: Path):
    """Passing a 0-byte file must raise ValueError."""
    empty_file = tmp_path / "empty.mp4"
    empty_file.touch()
    with pytest.raises(ValueError, match="empty"):
        generate_video_id(empty_file)
