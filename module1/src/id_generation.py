"""Deterministic video ID generation based on file metadata and content."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def generate_video_id(video_path: Union[str, Path]) -> str:
    """Generate a stable, deterministic video_id for a given video file.

    The ID is constructed using a cryptographic SHA-256 hash incorporating:
      - The file name (normalized)
      - The exact file size in bytes
      - Content byte samples (full file if <= 16MB, otherwise header, middle,
        and footer chunks)

    This ensures re-running the pipeline on the same video produces the exact
    same ID, preventing collisions and random UUID drift across runs.

    Args:
        video_path: Path to the video file.

    Returns:
        Deterministic alphanumeric identifier prefixed with 'vid_' (e.g. 'vid_a1b2c3d4e5f60718').

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the file is not a regular file or is empty.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Video file is empty (0 bytes): {path}")

    hasher = hashlib.sha256()

    # Incorporate normalized name and size
    hasher.update(path.name.lower().encode("utf-8"))
    hasher.update(str(file_size).encode("utf-8"))

    chunk_size = 64 * 1024  # 64 KB
    max_full_read_bytes = 16 * 1024 * 1024  # 16 MB

    with open(path, "rb") as f:
        if file_size <= max_full_read_bytes:
            # Full content hash for files up to 16MB
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        else:
            # For large video files: sample head (8MB), middle (4MB), tail (4MB)
            head_bytes = 8 * 1024 * 1024
            mid_bytes = 4 * 1024 * 1024
            tail_bytes = 4 * 1024 * 1024

            # Read head
            bytes_read = 0
            while bytes_read < head_bytes:
                to_read = min(chunk_size, head_bytes - bytes_read)
                chunk = f.read(to_read)
                if not chunk:
                    break
                hasher.update(chunk)
                bytes_read += len(chunk)

            # Read middle
            mid_start = (file_size - mid_bytes) // 2
            f.seek(mid_start)
            bytes_read = 0
            while bytes_read < mid_bytes:
                to_read = min(chunk_size, mid_bytes - bytes_read)
                chunk = f.read(to_read)
                if not chunk:
                    break
                hasher.update(chunk)
                bytes_read += len(chunk)

            # Read tail
            tail_start = max(0, file_size - tail_bytes)
            f.seek(tail_start)
            bytes_read = 0
            while bytes_read < tail_bytes:
                to_read = min(chunk_size, tail_bytes - bytes_read)
                chunk = f.read(to_read)
                if not chunk:
                    break
                hasher.update(chunk)
                bytes_read += len(chunk)

    hex_digest = hasher.hexdigest()[:16]
    return f"vid_{hex_digest}"
