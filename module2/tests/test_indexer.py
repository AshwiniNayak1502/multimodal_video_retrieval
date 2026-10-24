"""Unit tests for incremental indexer, duplicate safety, and error tolerance."""

import json
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import pytest
from PIL import Image

from embedding_store.config import Module2Config
from embedding_store.embedders import FakeTextEmbedder, FakeVisualEmbedder
from embedding_store.indexer import VideoIndexer


class CountingFakeVisualEmbedder(FakeVisualEmbedder):
    """Spy wrapper counting embed_images invocations."""

    def __init__(self, dim: int = 768):
        super().__init__(dim=dim)
        self.call_count: int = 0
        self.embedded_items_count: int = 0

    def embed_images(self, images: Sequence[Union[Image.Image, Path, str]]) -> np.ndarray:
        self.call_count += 1
        self.embedded_items_count += len(images)
        return super().embed_images(images)


class CountingFakeTextEmbedder(FakeTextEmbedder):
    """Spy wrapper counting embed_texts invocations."""

    def __init__(self, dim: int = 384):
        super().__init__(dim=dim)
        self.call_count: int = 0
        self.embedded_items_count: int = 0

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        self.call_count += 1
        self.embedded_items_count += len(texts)
        return super().embed_texts(texts)


def test_incremental_indexing_and_idempotence(temp_config: Module2Config):
    """Verify first run indexes items, and second run adds 0 vectors with 0 model calls."""
    vis_spy = CountingFakeVisualEmbedder()
    txt_spy = CountingFakeTextEmbedder()

    indexer = VideoIndexer(
        config=temp_config,
        visual_embedder=vis_spy,
        text_embedder=txt_spy,
    )

    # First run
    s1 = indexer.run_indexing()
    # vid_test001: 4 frames, 2 segments; vid_silent002: 3 frames, 0 segments
    assert s1.visual_embedded == 7
    assert s1.text_embedded == 2
    assert indexer.vis_mgr is not None and indexer.txt_mgr is not None
    assert indexer.vis_mgr.ntotal == 7
    assert indexer.txt_mgr.ntotal == 2

    # Verify spy was called
    first_vis_calls = vis_spy.call_count
    first_txt_calls = txt_spy.call_count
    assert first_vis_calls > 0
    assert first_txt_calls > 0

    # Reset counters
    vis_spy.call_count = 0
    vis_spy.embedded_items_count = 0
    txt_spy.call_count = 0
    txt_spy.embedded_items_count = 0

    # Second run (idempotent)
    s2 = indexer.run_indexing()
    assert s2.visual_embedded == 0
    assert s2.text_embedded == 0
    assert s2.visual_skipped == 7
    assert s2.text_skipped == 2
    assert vis_spy.call_count == 0
    assert txt_spy.call_count == 0
    assert indexer.vis_mgr.ntotal == 7
    assert indexer.txt_mgr.ntotal == 2


def test_modify_single_frame_reembeds_only_that_frame(temp_config: Module2Config):
    """Verify modifying a frame file re-embeds only that single frame."""
    vis_spy = CountingFakeVisualEmbedder()
    txt_spy = CountingFakeTextEmbedder()

    indexer = VideoIndexer(
        config=temp_config,
        visual_embedder=vis_spy,
        text_embedder=txt_spy,
    )
    indexer.run_indexing()

    # Modify frame 1 of vid_test001
    frame_1_path = temp_config.module1_output_dir / "frames" / "vid_test001" / "frame_0001_2.000s.jpg"
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    img.save(frame_1_path, format="JPEG")

    vis_spy.embedded_items_count = 0
    s = indexer.run_indexing()
    assert s.visual_embedded == 1
    assert s.visual_skipped == 6
    assert vis_spy.embedded_items_count == 1
    assert indexer.vis_mgr is not None
    assert indexer.vis_mgr.ntotal == 7


def test_corrupt_frame_skipped_while_others_succeed(temp_config: Module2Config):
    """Verify a corrupted frame image is logged and skipped, while other frames index successfully."""
    # Corrupt frame 2
    frame_2_path = temp_config.module1_output_dir / "frames" / "vid_test001" / "frame_0002_4.000s.jpg"
    frame_2_path.write_bytes(b"NOT_A_VALID_JPEG_HEADER")

    indexer = VideoIndexer(config=temp_config, visual_embedder=FakeVisualEmbedder(), text_embedder=FakeTextEmbedder())
    s = indexer.run_indexing()
    # 7 total frames: 1 corrupted, 6 valid
    assert s.visual_failed == 1
    assert s.visual_embedded == 6
    assert indexer.vis_mgr is not None
    assert indexer.vis_mgr.ntotal == 6


def test_prune_missing_videos(temp_config: Module2Config):
    """Verify videos whose Module 1 metadata JSON is removed are pruned."""
    indexer = VideoIndexer(config=temp_config, visual_embedder=FakeVisualEmbedder(), text_embedder=FakeTextEmbedder())
    indexer.run_indexing()
    assert len(indexer.store.list_videos()) == 2

    # Remove metadata JSON for vid_silent002
    meta_silent = temp_config.module1_output_dir / "metadata" / "vid_silent002.json"
    meta_silent.unlink()

    s = indexer.run_indexing(prune_missing=True)
    videos = indexer.store.list_videos()
    assert len(videos) == 1
    assert videos[0]["video_id"] == "vid_test001"
    # Visual count went from 7 to 4
    assert indexer.vis_mgr is not None
    assert indexer.vis_mgr.ntotal == 4


def test_self_heal_readds_missing_vector(temp_config: Module2Config):
    """Verify verify_and_heal_integrity re-adds a deleted vector with correct ID and dimension."""
    indexer = VideoIndexer(config=temp_config, visual_embedder=FakeVisualEmbedder(), text_embedder=FakeTextEmbedder())
    indexer.run_indexing()
    assert indexer.vis_mgr is not None
    assert indexer.vis_mgr.ntotal == 7

    # Find one visual vector from vid_test001
    cursor = indexer.store.connection.cursor()
    cursor.execute("SELECT vector_id, frame_id FROM visual_vectors WHERE video_id = 'vid_test001' LIMIT 1")
    row = cursor.fetchone()
    target_id = row["vector_id"]
    target_frame_id = row["frame_id"]

    # Delete from SQLite and remove from FAISS
    with indexer.store.transaction():
        indexer.store.connection.execute("DELETE FROM visual_vectors WHERE vector_id = ?", (target_id,))
    indexer.vis_mgr.remove([target_id])
    assert indexer.vis_mgr.ntotal == 6

    # Verify target_id is gone
    assert target_id not in indexer.vis_mgr.get_id_set()

    # Run indexing / integrity healing
    healed_summary = indexer.run_indexing()
    assert indexer.vis_mgr.ntotal == 7

    # Check that vector is re-added in SQLite and FAISS
    cursor.execute("SELECT vector_id, embedding FROM visual_vectors WHERE frame_id = ?", (target_frame_id,))
    new_row = cursor.fetchone()
    assert new_row is not None
    readded_id = new_row["vector_id"]
    assert readded_id in indexer.vis_mgr.get_id_set()

    # Check dimension
    emb = np.frombuffer(new_row["embedding"], dtype=np.float32)
    assert emb.shape[0] == indexer.vis_mgr.dim
    assert indexer.vis_mgr.dim == 768

