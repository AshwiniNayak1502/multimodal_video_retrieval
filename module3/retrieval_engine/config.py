"""Configuration container for Module 3 Multimodal Query Processing and Retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import ValidationError


@dataclass
class Module3Config:
    """Central configuration for query processing, calibration, fusion, and segment retrieval."""

    # Base directories
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    module2_dir: Path = field(default=Path())
    module1_dir: Path = field(default=Path())
    vector_db_dir: Path = field(default=Path())
    output_dir: Path = field(default=Path())
    log_file: Optional[Path] = None

    # Query constraints
    min_top_k: int = 3
    max_top_k: int = 5
    default_top_k: int = 5
    max_query_chars: int = 500

    # Candidate retrieval pool bounds
    visual_pool: int = 50
    text_pool: int = 50

    # Calibration & relevance gating thresholds (empirically calibrated via tools/tune_thresholds.py)
    visual_min_prob: float = 0.15
    visual_fallback_scale: float = 12.0
    visual_fallback_bias: float = -2.5
    text_min_cosine: float = 0.22
    text_floor: float = 0.20
    text_ceiling: float = 0.80
    final_min_score: float = 0.0  # Single gating per modality; 0.0 eliminates double gating

    # Fusion weights & options
    fusion_mode: str = "weighted"  # "weighted" or "noisy_or"
    visual_weight: float = 0.6
    text_weight: float = 0.4
    agreement_bonus: float = 0.05
    use_rrf_tiebreaker: bool = True
    rrf_k: int = 60

    # Segment linking & deduplication
    link_tolerance_s: Optional[float] = None
    seek_offset_s: float = 1.0

    # Query vector LRU cache size
    query_cache_size: int = 128
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Resolve paths dynamically relative to workspace hierarchy."""
        if not self.base_dir or self.base_dir == Path():
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(self.base_dir).resolve()

        if not self.module2_dir or self.module2_dir == Path():
            self.module2_dir = (self.base_dir.parent / "module2").resolve()
        else:
            self.module2_dir = Path(self.module2_dir).resolve()

        if not self.module1_dir or self.module1_dir == Path():
            self.module1_dir = (self.base_dir.parent / "module1").resolve()
        else:
            self.module1_dir = Path(self.module1_dir).resolve()

        if not self.output_dir or self.output_dir == Path():
            self.output_dir = (self.base_dir / "output").resolve()
        else:
            self.output_dir = Path(self.output_dir).resolve()

        if not self.vector_db_dir or self.vector_db_dir == Path():
            self.vector_db_dir = (self.module2_dir / "output" / "vector_db").resolve()
        else:
            self.vector_db_dir = Path(self.vector_db_dir).resolve()

        if self.log_file is None:
            self.log_file = (self.output_dir / "module3.log").resolve()
        else:
            self.log_file = Path(self.log_file).resolve()

    def ensure_directories(self) -> None:
        """Ensure all required output directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file and self.log_file.parent:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Module3Config:
        """Construct configuration instance from dictionary with unknown key rejection."""
        copied = dict(data)
        # Safely ignore deprecated dedup_similarity_threshold for backward compatibility
        copied.pop("dedup_similarity_threshold", None)

        known_fields = {f.name for f in fields(cls)}
        unknown_keys = set(copied.keys()) - known_fields
        if unknown_keys:
            sorted_unknown = sorted(unknown_keys)
            raise ValidationError(
                f"Unknown configuration key(s): {sorted_unknown}",
                user_friendly_message=f"Configuration error: unrecognized setting(s): {', '.join(sorted_unknown)}",
            )

        path_keys = [
            "base_dir",
            "module2_dir",
            "module1_dir",
            "vector_db_dir",
            "output_dir",
            "log_file",
        ]
        for pk in path_keys:
            if pk in copied and copied[pk] is not None:
                copied[pk] = Path(copied[pk])
        return cls(**copied)

    @classmethod
    def from_json(cls, json_path: Path | str) -> Module3Config:
        """Load configuration from a JSON file with validation."""
        path = Path(json_path)
        if not path.is_file():
            raise ValidationError(
                f"Configuration file not found: {path}",
                user_friendly_message=f"Configuration file '{path}' does not exist.",
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"Malformed JSON in configuration file '{path}': {exc}",
                user_friendly_message=f"Configuration file '{path}' contains invalid JSON: {exc}",
            )
        except Exception as exc:
            raise ValidationError(
                f"Error reading configuration file '{path}': {exc}",
                user_friendly_message=f"Could not read configuration file '{path}': {exc}",
            )

        if not isinstance(data, dict):
            raise ValidationError(
                f"Configuration file '{path}' must contain a JSON object, got {type(data).__name__}",
                user_friendly_message=f"Configuration file '{path}' must contain a JSON object.",
            )

        return cls.from_dict(data)
