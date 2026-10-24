"""Centralized configuration container for Module 2.

Defines all directories, model names, batch sizes, hardware execution targets,
and runtime execution modes (online cache vs. offline operation).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

# Default constants
DEFAULT_VISUAL_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_VISUAL_FALLBACK_MODEL = "google/siglip-base-patch16-224"
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIGLIP_DIM = 768
DEFAULT_MINILM_DIM = 384


def get_default_device() -> str:
    """Detect available hardware accelerator, falling back to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Module2Config:
    """Configuration container for Module 2 embedding and vector store operations."""

    # Base directories
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    module1_root: Path = field(default=Path())
    module1_output_dir: Path = field(default=Path())
    module1_input_dir: Path = field(default=Path())

    output_dir: Path = field(default=Path())
    vector_db_dir: Path = field(default=Path())
    log_file: Optional[Path] = None

    # Model specifications
    visual_model_name: str = DEFAULT_VISUAL_MODEL
    text_model_name: str = DEFAULT_TEXT_MODEL

    # Hardware & batching
    device: str = field(default_factory=get_default_device)
    visual_batch_size: Optional[int] = None
    text_batch_size: int = 64

    # Caching and offline behavior
    models_dir: Optional[Path] = None
    offline_mode: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Resolve directory paths and set dynamic batch sizes."""
        if not self.base_dir or self.base_dir == Path():
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(self.base_dir).resolve()

        if not self.module1_root or self.module1_root == Path():
            self.module1_root = (self.base_dir.parent / "module1").resolve()
        else:
            self.module1_root = Path(self.module1_root).resolve()

        if not self.module1_output_dir or self.module1_output_dir == Path():
            self.module1_output_dir = (self.module1_root / "output").resolve()
        else:
            self.module1_output_dir = Path(self.module1_output_dir).resolve()

        if not self.module1_input_dir or self.module1_input_dir == Path():
            self.module1_input_dir = (self.module1_root / "input").resolve()
        else:
            self.module1_input_dir = Path(self.module1_input_dir).resolve()

        if not self.output_dir or self.output_dir == Path():
            self.output_dir = (self.base_dir / "output").resolve()
        else:
            self.output_dir = Path(self.output_dir).resolve()

        if not self.vector_db_dir or self.vector_db_dir == Path():
            self.vector_db_dir = (self.output_dir / "vector_db").resolve()
        else:
            self.vector_db_dir = Path(self.vector_db_dir).resolve()

        if self.log_file is None:
            self.log_file = (self.output_dir / "module2.log").resolve()
        else:
            self.log_file = Path(self.log_file).resolve()

        if self.models_dir is not None:
            self.models_dir = Path(self.models_dir).resolve()

        if self.visual_batch_size is None:
            self.visual_batch_size = 32 if self.device.startswith("cuda") else 16

    def ensure_directories(self) -> None:
        """Ensure all required output and vector db directories exist on disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file and self.log_file.parent:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.models_dir:
            self.models_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Module2Config:
        """Create config instance from dictionary."""
        copied = dict(data)
        path_keys = [
            "base_dir",
            "module1_root",
            "module1_output_dir",
            "module1_input_dir",
            "output_dir",
            "vector_db_dir",
            "log_file",
            "models_dir",
        ]
        for pk in path_keys:
            if pk in copied and copied[pk] is not None:
                copied[pk] = Path(copied[pk])
        return cls(**copied)

    @classmethod
    def from_json(cls, json_path: Path | str) -> Module2Config:
        """Load configuration from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
