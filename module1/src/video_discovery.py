"""Video file discovery module for Module 1."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set, Union

from .config import SUPPORTED_EXTENSIONS

logger = logging.getLogger("module1.discovery")


def discover_videos(
    input_dir: Union[str, Path],
    recursive: bool = True,
    supported_extensions: Optional[Set[str]] = None,
) -> List[Path]:
    """Scan a directory for video files matching supported extensions (.mp4, .mkv).

    Args:
        input_dir: Path to directory to scan.
        recursive: Whether to search subdirectories recursively.
        supported_extensions: Set of extensions to match (defaults to {".mp4", ".mkv"}).

    Returns:
        Sorted list of resolved Path objects for all discovered valid video files.
    """
    directory = Path(input_dir).resolve()
    valid_exts = {
        ext.lower() for ext in (supported_extensions or SUPPORTED_EXTENSIONS)
    }

    if not directory.exists():
        logger.warning("Input directory does not exist: %s", directory)
        return []

    if not directory.is_dir():
        logger.warning("Input path is not a directory: %s", directory)
        return []

    pattern = "**/*" if recursive else "*"
    found_videos: List[Path] = []

    for item in directory.glob(pattern):
        # Skip directories
        if not item.is_file():
            continue

        # Skip hidden files or OS artifacts (e.g. .DS_Store, ._*)
        if item.name.startswith(".") or item.name.startswith("~$"):
            continue

        if item.suffix.lower() in valid_exts:
            found_videos.append(item.resolve())

    found_videos.sort()
    logger.info(
        "Discovered %d video(s) in %s with extensions %s",
        len(found_videos),
        directory,
        valid_exts,
    )
    return found_videos
