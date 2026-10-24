"""Centralized configuration module for Module 1.

All paths, sampling intervals, audio settings, and WhisperX model
parameters are defined here in one place.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Set

import torch

# Default Constants
SUPPORTED_EXTENSIONS: Set[str] = {".mp4", ".mkv"}
DEFAULT_FRAME_INTERVAL_SECONDS: float = 2.0
DEFAULT_WHISPER_MODEL: str = "tiny"
DEFAULT_VAD_METHOD: str = "silero"
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_AUDIO_SAMPLE_RATE: int = 16000
DEFAULT_AUDIO_CHANNELS: int = 1
DEFAULT_AUDIO_CODEC: str = "pcm_s16le"


def get_default_device() -> str:
    """Detect available hardware accelerator, falling back to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_default_compute_type(device: Optional[str] = None) -> str:
    """Determine default compute type based on target device.

    CTranslate2 supports float16 on CUDA, and int8 or float32 on CPU.
    """
    dev = device or get_default_device()
    return "float16" if dev.startswith("cuda") else "int8"


@dataclass
class Module1Config:
    """Configuration container for Module 1 pipeline operations."""

    # Base directories
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    input_videos_dir: Path = field(default=Path())
    output_dir: Path = field(default=Path())
    frames_dir: Path = field(default=Path())
    audio_dir: Path = field(default=Path())
    transcripts_dir: Path = field(default=Path())
    metadata_dir: Path = field(default=Path())

    # Supported video extensions
    supported_extensions: Set[str] = field(
        default_factory=lambda: set(SUPPORTED_EXTENSIONS)
    )

    # Sampling parameters
    frame_interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS

    # Audio extraction settings
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    audio_channels: int = DEFAULT_AUDIO_CHANNELS
    audio_codec: str = DEFAULT_AUDIO_CODEC

    # WhisperX transcription settings
    whisper_model: str = DEFAULT_WHISPER_MODEL
    vad_method: str = DEFAULT_VAD_METHOD
    device: str = field(default_factory=get_default_device)
    compute_type: str = field(default_factory=get_default_compute_type)
    batch_size: int = DEFAULT_BATCH_SIZE

    # Binary tools paths
    ffmpeg_path: str = field(
        default_factory=lambda: shutil.which("ffmpeg") or "ffmpeg"
    )
    ffprobe_path: str = field(
        default_factory=lambda: shutil.which("ffprobe") or "ffprobe"
    )

    def __post_init__(self) -> None:
        """Resolve paths relative to base_dir if default paths are unset."""
        if not self.input_videos_dir or self.input_videos_dir == Path():
            self.input_videos_dir = self.base_dir / "input" / "videos"
        else:
            self.input_videos_dir = Path(self.input_videos_dir).resolve()

        if not self.output_dir or self.output_dir == Path():
            self.output_dir = self.base_dir / "output"
        else:
            self.output_dir = Path(self.output_dir).resolve()

        if not self.frames_dir or self.frames_dir == Path():
            self.frames_dir = self.output_dir / "frames"
        else:
            self.frames_dir = Path(self.frames_dir).resolve()

        if not self.audio_dir or self.audio_dir == Path():
            self.audio_dir = self.output_dir / "audio"
        else:
            self.audio_dir = Path(self.audio_dir).resolve()

        if not self.transcripts_dir or self.transcripts_dir == Path():
            self.transcripts_dir = self.output_dir / "transcripts"
        else:
            self.transcripts_dir = Path(self.transcripts_dir).resolve()

        if not self.metadata_dir or self.metadata_dir == Path():
            self.metadata_dir = self.output_dir / "metadata"
        else:
            self.metadata_dir = Path(self.metadata_dir).resolve()

        # Re-evaluate compute_type if device was customized
        if self.device == "cpu" and self.compute_type == "float16":
            self.compute_type = "int8"

    def ensure_directories(self) -> None:
        """Ensure all required input and output directories exist on disk."""
        self.input_videos_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
            elif isinstance(value, set):
                d[key] = sorted(list(value))
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Module1Config:
        """Create config instance from dictionary."""
        copied = dict(data)
        path_keys = [
            "base_dir",
            "input_videos_dir",
            "output_dir",
            "frames_dir",
            "audio_dir",
            "transcripts_dir",
            "metadata_dir",
        ]
        for pk in path_keys:
            if pk in copied and copied[pk]:
                copied[pk] = Path(copied[pk])
        if "supported_extensions" in copied and isinstance(
            copied["supported_extensions"], (list, tuple)
        ):
            copied["supported_extensions"] = set(copied["supported_extensions"])
        return cls(**copied)

    @classmethod
    def from_json(cls, json_path: Path | str) -> Module1Config:
        """Load configuration from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
