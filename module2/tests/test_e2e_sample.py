"""End-to-End Validation covering all 15 Core System Requirements and Real Model tests."""

from pathlib import Path

import numpy as np
import pytest

from embedding_store.config import Module2Config
from embedding_store.embedders import (
    FakeTextEmbedder,
    FakeVisualEmbedder,
    MiniLMTextEmbedder,
    SigLIPVisualEmbedder,
)
from embedding_store.indexer import VideoIndexer
from embedding_store.retrieval import VideoIndex


class UncalledEmbedder:
    """Sentinel embedder verifying no calls are made during pure vector search."""

    def __init__(self, dim: int = 768, model_name: str = "mock"):
        self._dim = dim
        self._model_name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_images(self, *args, **kwargs):
        raise AssertionError("Visual embedder was invoked unexpectedly!")

    def embed_texts(self, *args, **kwargs):
        raise AssertionError("Text embedder was invoked unexpectedly!")

    def embed_query_text(self, *args, **kwargs):
        raise AssertionError("Query embedder was invoked unexpectedly!")


def test_e2e_15_spec_requirements(temp_config: Module2Config):
    """End-to-end synthetic verification of all 15 core architectural facts and invariants."""
    vis_embedder = FakeVisualEmbedder(dim=768)
    txt_embedder = FakeTextEmbedder(dim=384)

    indexer = VideoIndexer(
        config=temp_config,
        visual_embedder=vis_embedder,
        text_embedder=txt_embedder,
    )

    # 1. Frames and segments discovered, 2. visual embeddings generated, 3. text embeddings generated
    summary = indexer.run_indexing()

    # 4. Dimensions correct (768 for visual, 384 for text)
    assert indexer.visual_embedder.dim == 768
    assert indexer.text_embedder.dim == 384
    assert indexer.vis_mgr is not None and indexer.txt_mgr is not None
    assert indexer.vis_mgr.index.d == 768
    assert indexer.txt_mgr.index.d == 384

    # 5. Vector counts == expected input counts
    # vid_test001: 4 frames, 2 segments; vid_silent002: 3 frames, 0 segments
    expected_vis_count = 7
    expected_txt_count = 2
    assert indexer.vis_mgr.ntotal == expected_vis_count
    assert indexer.txt_mgr.ntotal == expected_txt_count

    # 6. Metadata row count == vector count
    cur = indexer.store.connection.cursor()
    cur.execute("SELECT COUNT(*) FROM visual_vectors")
    assert cur.fetchone()[0] == expected_vis_count
    cur.execute("SELECT COUNT(*) FROM text_vectors")
    assert cur.fetchone()[0] == expected_txt_count

    # 7. Every vector_id resolves to a metadata row (both directions)
    vis_faiss_ids = indexer.vis_mgr.get_id_set()
    cur.execute("SELECT vector_id FROM visual_vectors")
    vis_sql_ids = set(int(r[0]) for r in cur.fetchall())
    assert vis_faiss_ids == vis_sql_ids

    txt_faiss_ids = indexer.txt_mgr.get_id_set()
    cur.execute("SELECT vector_id FROM text_vectors")
    txt_sql_ids = set(int(r[0]) for r in cur.fetchall())
    assert txt_faiss_ids == txt_sql_ids

    # 8. Every visual vector has timestamp_seconds equal to Module 1 frame timestamp
    cur.execute("SELECT timestamp_seconds FROM visual_vectors WHERE video_id = 'vid_test001' ORDER BY frame_index ASC")
    vis_ts = [float(r[0]) for r in cur.fetchall()]
    assert vis_ts == [0.0, 2.0, 4.0, 6.0]

    # 9. Every text vector has start/end equal to Module 1 segment times
    cur.execute("SELECT start_timestamp, end_timestamp FROM text_vectors WHERE video_id = 'vid_test001' ORDER BY segment_id ASC")
    txt_ts = [(float(r[0]), float(r[1])) for r in cur.fetchall()]
    assert txt_ts == [(0.158, 3.402), (4.383, 6.966)]

    # 10. Indexes save
    indexer.save_indices()
    assert (temp_config.vector_db_dir / "visual.index").is_file()
    assert (temp_config.vector_db_dir / "text.index").is_file()
    assert (temp_config.vector_db_dir / "index_manifest.json").is_file()

    # 11. Indexes reload in a fresh VideoIndex without calling any embedder
    uncalled_vis = UncalledEmbedder(dim=768, model_name="mock/siglip-fake")
    uncalled_txt = UncalledEmbedder(dim=384, model_name="mock/minilm-fake")
    with VideoIndex.open(temp_config, visual_embedder=uncalled_vis, text_embedder=uncalled_txt) as reloaded_index:
        assert reloaded_index.vis_mgr.ntotal == expected_vis_count
        assert reloaded_index.txt_mgr.ntotal == expected_txt_count

        # Query with precomputed vector (embedder must NOT be invoked)
        v_dummy = np.zeros(768, dtype=np.float32)
        v_dummy[0] = 1.0
        hits = reloaded_index.search_visual(query=v_dummy, top_k=2)

        # 12. Similarity search returns metadata
        # 13. Hits contain video_filename and video_path
        assert len(hits) == 2
        assert "video_filename" in hits[0]
        assert "video_path" in hits[0]
        assert hits[0]["video_filename"] in ("lecture_sample.mp4", "silent_clip.mkv")

    # 14. Hit timestamps equal original Module 1 values
    with VideoIndex.open(temp_config, visual_embedder=vis_embedder, text_embedder=txt_embedder) as search_index:
        # Retrieve vector for frame 1 (timestamp 2.0s)
        cur.execute("SELECT embedding FROM visual_vectors WHERE video_id = 'vid_test001' AND frame_index = 1")
        f1_blob = cur.fetchone()[0]
        f1_vec = np.frombuffer(f1_blob, dtype=np.float32)

        exact_vis_hits = search_index.search_visual(f1_vec, top_k=1)
        assert len(exact_vis_hits) == 1
        assert exact_vis_hits[0]["video_id"] == "vid_test001"
        assert exact_vis_hits[0]["timestamp_seconds"] == 2.0
        assert exact_vis_hits[0]["timestamp_formatted"] == "00:00:02"

        # Search with segment 0 text
        txt_hits = search_index.search_text("Welcome to multimodal semantic indexing.", top_k=1)
        assert len(txt_hits) == 1
        assert txt_hits[0]["video_id"] == "vid_test001"
        assert txt_hits[0]["segment_id"] == 0
        assert txt_hits[0]["start_timestamp"] == 0.158
        assert txt_hits[0]["end_timestamp"] == 3.402

    # 15. Video with has_audio=False yields visual vectors and 0 text vectors
    cur.execute("SELECT COUNT(*) FROM visual_vectors WHERE video_id = 'vid_silent002'")
    assert cur.fetchone()[0] == 3
    cur.execute("SELECT COUNT(*) FROM text_vectors WHERE video_id = 'vid_silent002'")
    assert cur.fetchone()[0] == 0


@pytest.mark.real_models
def test_real_models_sample_indexing():
    """Run real SigLIP2 and MiniLM on sample output if cached and sample exists."""
    real_sample_id = "vid_b7c97c1b56e0feed"
    project_root = Path(__file__).resolve().parent.parent.parent
    sample_json = project_root / "module1" / "output" / "metadata" / f"{real_sample_id}.json"

    if not sample_json.is_file():
        pytest.skip(f"Real model test skipped: sample output {sample_json} is absent.")

    # Check if models can be loaded locally
    try:
        vis_emb = SigLIPVisualEmbedder(offline_mode=True)
        txt_emb = MiniLMTextEmbedder(offline_mode=True)
    except Exception as exc:
        pytest.skip(f"Real model test skipped: local models not available in cache: {exc}")

    test_cfg = Module2Config(
        base_dir=project_root / "module2",
        module1_root=project_root / "module1",
        offline_mode=True,
    )

    indexer = VideoIndexer(config=test_cfg, visual_embedder=vis_emb, text_embedder=txt_emb)
    indexer.run_indexing(video_id=real_sample_id, force_reindex=True)

    with VideoIndex.open(test_cfg, visual_embedder=vis_emb, text_embedder=txt_emb) as v_index:
        # Search text
        txt_hits = v_index.search_text("Welcome to multimodal semantic indexing", top_k=1)
        assert len(txt_hits) == 1
        assert txt_hits[0]["segment_id"] == 0
        assert np.isclose(txt_hits[0]["start_timestamp"], 0.158, atol=1e-3)
        assert np.isclose(txt_hits[0]["end_timestamp"], 3.402, atol=1e-3)

        # Search visual
        vis_hits = v_index.search_visual("lecture presentation slide", top_k=1)
        assert len(vis_hits) == 1
        assert vis_hits[0]["video_filename"] == "lecture_sample.mp4"
        assert vis_hits[0]["timestamp_seconds"] in [0.0, 2.0, 4.0, 6.0]
