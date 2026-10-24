"""Module 3: Multimodal Semantic Query Processing & Timestamped Segment Retrieval."""

from .calibration import (
    calibrate_text_score,
    calibrate_visual_score,
    extract_siglip_calibration_parameters,
)
from .config import Module3Config
from .errors import (
    CorruptIndexError,
    IndexNotReadyError,
    ModelMismatchError,
    ModalityUnavailableError,
    RetrievalEngineError,
    ValidationError,
)
from .fusion import compute_fused_score, compute_rrf_score
from .query_service import QueryService, SearchResponse
from .segments import Result, format_timestamp

__all__ = [
    "Module3Config",
    "QueryService",
    "SearchResponse",
    "Result",
    "format_timestamp",
    "compute_fused_score",
    "compute_rrf_score",
    "calibrate_visual_score",
    "calibrate_text_score",
    "extract_siglip_calibration_parameters",
    "RetrievalEngineError",
    "ValidationError",
    "IndexNotReadyError",
    "ModelMismatchError",
    "CorruptIndexError",
    "ModalityUnavailableError",
]
