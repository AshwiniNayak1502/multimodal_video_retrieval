"""Unit tests for VideoIndex retrieval API, cross-modal linking, and hit formats."""

from pathlib import Path

import numpy as np
import pytest

from embedding_store.config import Module2Config
from embedding_store.embedders import FakeTextEmbedder, FakeVisualEmbedder
from embedding_store.indexer import VideoIndexer
from embedding_store.retrieval import VideoIndex


@pytest.fixture
def populated_index(temp_config: Module2Config) -> Module2Config:
    """Run indexer on synthetic Module 1 tree to populate database and FAISS."""
    indexer = VideoIndexer(
        config=temp_config,
        visual_embedder=FakeVisualEmbedder(dim=768),
        text_embedder=FakeTextEmbedder(dim=384),
    )
    indexer.run_indexing()
    return temp_config


def test_retrieval_hit_schemas(populated_index: Module2Config):
    """Verify search_visual and search_text hit dictionaries have all required fields."""
    fake_vis = FakeVisualEmbedder(dim=768)
    fake_txt = FakeTextEmbedder(dim=384)

    with VideoIndex.open(
        populated_index, visual_embedder=fake_vis, text_embedder=fake_txt
    ) as v_index:
        # 1. Visual Search
        vis_hits = v_index.search_visual("presentation slide", top_k=2)
        assert len(vis_hits) == 2
        h_vis = vis_hits[0]

        expected_vis_keys = {
            "modality",
            "vector_id",
            "score",
            "video_id",
            "video_filename",
            "video_path",
            "frame_id",
            "frame_index",
            "frame_path",
            "timestamp_seconds",
            "timestamp_formatted",
            "frame_interval_seconds",
            "embedding_model",
        }
        assert expected_vis_keys.issubset(h_vis.keys())
        assert h_vis["modality"] == "visual"
        assert isinstance(h_vis["score"], float)
        assert isinstance(h_vis["timestamp_seconds"], float)
        assert Path(h_vis["frame_path"]).is_file()

        # 2. Text Search
        txt_hits = v_index.search_text("indexing visual frames", top_k=2)
        assert len(txt_hits) == 2
        h_txt = txt_hits[0]

        expected_txt_keys = {
            "modality",
            "vector_id",
            "score",
            "video_id",
            "video_filename",
            "video_path",
            "segment_id",
            "transcript_text",
            "start_timestamp",
            "end_timestamp",
            "start_formatted",
            "end_formatted",
            "language",
            "embedding_model",
        }
        assert expected_txt_keys.issubset(h_txt.keys())
        assert h_txt["modality"] == "text"
        assert h_txt["start_timestamp"] <= h_txt["end_timestamp"]


def test_video_ids_candidate_filter(populated_index: Module2Config):
    """Verify over-fetch candidate filtering by video_ids."""
    with VideoIndex.open(
        populated_index,
        visual_embedder=FakeVisualEmbedder(dim=768),
        text_embedder=FakeTextEmbedder(dim=384),
    ) as v_index:
        # Search restricted to vid_silent002
        hits = v_index.search_visual(
            query="test",
            top_k=5,
            video_ids=["vid_silent002"],
        )
        assert len(hits) == 3
        assert all(h["video_id"] == "vid_silent002" for h in hits)


def test_cross_modal_linking_helpers(populated_index: Module2Config):
    """Verify get_frames_in_window and get_segments_in_window linking."""
    with VideoIndex.open(
        populated_index,
        visual_embedder=FakeVisualEmbedder(dim=768),
        text_embedder=FakeTextEmbedder(dim=384),
    ) as v_index:
        # vid_test001 frames: 0.0, 2.0, 4.0, 6.0
        frames = v_index.get_frames_in_window("vid_test001", 1.0, 5.0)
        assert len(frames) == 2  # 2.0 and 4.0
        assert frames[0]["timestamp_seconds"] == 2.0
        assert frames[1]["timestamp_seconds"] == 4.0

        # vid_test001 segments: [0.158 -> 3.402], [4.383 -> 6.966]
        segs = v_index.get_segments_in_window("vid_test001", 0.0, 2.0)
        assert len(segs) == 1
        assert segs[0]["segment_id"] == 0
