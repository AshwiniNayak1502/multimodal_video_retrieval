"""Unit tests for video discovery module."""

from pathlib import Path

from src.video_discovery import discover_videos


def test_discover_supported_extensions(tmp_path: Path):
    """Verifies that only .mp4 and .mkv files are discovered."""
    (tmp_path / "clip1.mp4").touch()
    (tmp_path / "clip2.mkv").touch()
    (tmp_path / "clip3.MP4").touch()
    (tmp_path / "ignore.avi").touch()
    (tmp_path / "ignore.txt").touch()
    (tmp_path / "ignore.jpg").touch()

    videos = discover_videos(tmp_path)
    discovered_names = [v.name.lower() for v in videos]

    assert len(videos) == 3
    assert "clip1.mp4" in discovered_names
    assert "clip2.mkv" in discovered_names
    assert "clip3.mp4" in discovered_names
    assert "ignore.avi" not in discovered_names


def test_discover_recursive_subdirectories(tmp_path: Path):
    """Verifies discovery works across nested subdirectories."""
    sub = tmp_path / "season_1" / "episodes"
    sub.mkdir(parents=True)
    (sub / "ep1.mp4").touch()
    (tmp_path / "root.mkv").touch()

    videos = discover_videos(tmp_path, recursive=True)
    assert len(videos) == 2


def test_discover_nonexistent_directory(tmp_path: Path):
    """Nonexistent input directory returns empty list safely."""
    missing = tmp_path / "nonexistent_dir"
    videos = discover_videos(missing)
    assert videos == []


def test_discover_empty_directory(tmp_path: Path):
    """Empty directory returns empty list without error."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    videos = discover_videos(empty_dir)
    assert videos == []
