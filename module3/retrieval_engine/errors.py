"""Exception hierarchy for Module 3 Multimodal Retrieval Engine."""

from __future__ import annotations


class RetrievalEngineError(Exception):
    """Base exception for all retrieval engine errors."""

    def __init__(self, message: str, user_friendly_message: str | None = None) -> None:
        super().__init__(message)
        self.user_friendly_message = user_friendly_message or message


class ValidationError(RetrievalEngineError):
    """Raised when query inputs (text, top_k, video_ids) fail validation."""
    pass


class IndexNotReadyError(RetrievalEngineError):
    """Raised when the vector index or metadata database is empty or uninitialized."""
    pass


class ModelMismatchError(RetrievalEngineError):
    """Raised when manifest models or dimensions mismatch loaded embedders."""
    pass


class CorruptIndexError(RetrievalEngineError):
    """Raised when vector indices or metadata storage fail validation or are corrupted."""
    pass


class ModalityUnavailableError(RetrievalEngineError):
    """Raised or noted when an embedding modality cannot be utilized."""
    pass
