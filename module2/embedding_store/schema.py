"""Typed schemas and data transfer records for Module 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class VideoRecord:
    """Consolidated video entry ingested from Module 1."""

    video_id: str
    video_filename: str
    video_path: str
    video_relpath: Optional[str]
    duration_seconds: float
    fps: float
    width: int
    height: int
    frame_interval_seconds: float
    has_audio: bool
    language: Optional[str]
    transcript_path: Optional[str]
    m1_metadata_path: str
    m1_processed_at: str
    indexed_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class VisualItem:
    """Individual representative video frame prepared for embedding and storage."""

    frame_id: str  # f"{video_id}_frame_{frame_index:05d}"
    video_id: str
    frame_index: int
    source_frame_number: Optional[int]
    timestamp_seconds: float
    timestamp_formatted: str  # "HH:MM:SS"
    frame_path: str
    frame_relpath: Optional[str]
    width: int
    height: int
    content_hash: str  # SHA-256 of image bytes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TextItem:
    """Individual transcript speech segment prepared for embedding and storage."""

    segment_uid: str  # f"{video_id}_seg_{segment_id:05d}"
    video_id: str
    segment_id: int
    transcript_text: str  # normalized / stripped
    start_timestamp: float
    end_timestamp: float
    start_formatted: str  # "HH:MM:SS"
    end_formatted: str  # "HH:MM:SS"
    language: Optional[str]
    content_hash: str  # SHA-256 of (text + "|" + start + "|" + end)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class IndexFailure:
    """Record of a frame or segment that failed during extraction or embedding."""

    video_id: str
    modality: str  # 'visual' or 'text'
    item_key: str
    reason: str
    failed_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
