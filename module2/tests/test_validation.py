"""Unit tests for the validation integrity checker."""

from pathlib import Path

import pytest

from embedding_store.config import Module2Config
from embedding_store.embedders import FakeTextEmbedder, FakeVisualEmbedder
from embedding_store.indexer import VideoIndexer
from embedding_store.validation import validate_integrity


def test_validation_passes_on_valid_index(temp_config: Module2Config):
    """Verify validation passes on a properly indexed store."""
    indexer = VideoIndexer(
        config=temp_config,
        visual_embedder=FakeVisualEmbedder(dim=768),
        text_embedder=FakeTextEmbedder(dim=384),
    )
    indexer.run_indexing()

    assert validate_integrity(temp_config) is True


def test_validation_fails_on_missing_index_file(temp_config: Module2Config):
    """Verify validation reports failure if an index file is deleted."""
    indexer = VideoIndexer(
        config=temp_config,
        visual_embedder=FakeVisualEmbedder(dim=768),
        text_embedder=FakeTextEmbedder(dim=384),
    )
    indexer.run_indexing()

    vis_file = temp_config.vector_db_dir / "visual.index"
    vis_file.unlink()

    assert validate_integrity(temp_config) is False
