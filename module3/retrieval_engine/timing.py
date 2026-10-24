"""Telemetry and per-stage latency recording for retrieval pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict


class Timer:
    """Context manager to measure elapsed execution time in milliseconds."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self._elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        if self._elapsed_ms > 0.0:
            return self._elapsed_ms
        if self.start_time > 0.0:
            return (time.perf_counter() - self.start_time) * 1000.0
        return 0.0

    @elapsed_ms.setter
    def elapsed_ms(self, val: float) -> None:
        self._elapsed_ms = val


@dataclass
class SearchTimings:
    """Latency metrics recorded per stage during a query search."""

    query_encode_visual_ms: float = 0.0
    query_encode_text_ms: float = 0.0
    search_visual_ms: float = 0.0
    search_text_ms: float = 0.0
    calibrate_fuse_ms: float = 0.0
    segment_build_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Return rounded milliseconds dictionary."""
        return {
            k: round(v, 2)
            for k, v in asdict(self).items()
        }
