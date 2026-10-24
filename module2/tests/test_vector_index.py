"""Unit tests for FAISS VectorIndexManager and auto-rebuild mechanics."""

from pathlib import Path

import numpy as np
import pytest

from embedding_store.metadata_store import MetadataStore
from embedding_store.schema import VideoRecord, VisualItem
from embedding_store.vector_index import VectorIndexManager, load_or_rebuild_index


def test_vector_index_add_remove_search():
    """Verify FAISS IndexIDMap2 add, remove, and cosine search operations."""
    dim = 4
    mgr = VectorIndexManager(dim=dim, modality="visual")
    assert mgr.ntotal == 0

    # 3 vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    vecs = np.vstack([v1, v2, v3])
    ids = np.array([101, 102, 103], dtype=np.int64)

    mgr.add(vecs, ids)
    assert mgr.ntotal == 3
    assert mgr.get_id_set() == {101, 102, 103}

    # Search for v1
    scores, res_ids = mgr.search(v1, top_k=2)
    assert len(scores) == 2
    assert res_ids[0] == 101
    assert np.isclose(scores[0], 1.0, atol=1e-5)

    # Remove id 102
    removed = mgr.remove([102])
    assert removed == 1
    assert mgr.ntotal == 2
    assert mgr.get_id_set() == {101, 103}


def test_atomic_save_and_load(tmp_path: Path):
    """Verify saving to disk and re-loading from file."""
    dim = 8
    mgr = VectorIndexManager(dim=dim, modality="text")
    vecs = np.random.randn(5, dim).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    mgr.add(vecs, ids)
    save_file = tmp_path / "test_text.index"
    mgr.save_index(save_file)
    assert save_file.is_file()

    loaded_mgr = VectorIndexManager.load_index(save_file, dim=dim, modality="text")
    assert loaded_mgr.ntotal == 5
    assert loaded_mgr.get_id_set() == {10, 20, 30, 40, 50}

    # Verify search yields identical top score
    orig_scores, orig_ids = mgr.search(vecs[0], 1)
    load_scores, load_ids = loaded_mgr.search(vecs[0], 1)
    assert orig_ids[0] == load_ids[0]
    assert np.isclose(orig_scores[0], load_scores[0])


def test_corrupt_index_auto_rebuild_from_sqlite(tmp_path: Path):
    """Verify corrupt or missing .index file triggers automatic rebuild from SQLite BLOBs."""
    db_file = tmp_path / "meta.db"
    store = MetadataStore(db_file)
    store.upsert_video(
        VideoRecord(
            video_id="vid_recon",
            video_filename="v.mp4",
            video_path="/p/v.mp4",
            video_relpath=None,
            duration_seconds=5.0,
            fps=25.0,
            width=100,
            height=100,
            frame_interval_seconds=2.0,
            has_audio=False,
            language=None,
            transcript_path=None,
            m1_metadata_path="/m.json",
            m1_processed_at="now",
            indexed_at="now",
        )
    )

    # Insert 3 vectors into SQLite
    dim = 4
    for i in range(3):
        v = np.zeros(dim, dtype=np.float32)
        v[i] = 1.0
        item = VisualItem(
            frame_id=f"vid_recon_frame_{i:05d}",
            video_id="vid_recon",
            frame_index=i,
            source_frame_number=i * 50,
            timestamp_seconds=float(i * 2),
            timestamp_formatted=f"00:00:0{i*2}",
            frame_path=f"/f/{i}.jpg",
            frame_relpath=None,
            width=100,
            height=100,
            content_hash=f"h_{i}",
        )
        store.insert_visual_vector(item, v, "model_test")

    index_file = tmp_path / "visual.index"
    manifest_file = tmp_path / "manifest.json"

    # Case 1: Index file does not exist -> auto rebuilds
    mgr1 = load_or_rebuild_index("visual", index_file, manifest_file, store, dim, "model_test")
    assert mgr1.ntotal == 3
    assert index_file.is_file()

    # Case 2: Corrupt index file (overwrite with garbage bytes) -> auto rebuilds
    index_file.write_bytes(b"CORRUPTED_GARBAGE_BYTES_123456789")
    mgr2 = load_or_rebuild_index("visual", index_file, manifest_file, store, dim, "model_test")
    assert mgr2.ntotal == 3

    # Case 3: Search on rebuilt index produces correct top hit
    q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    scores, ids = mgr2.search(q, 1)
    assert scores[0] == 1.0
    # Vector 1 was inserted second -> vector_id is 2
    assert ids[0] == 2
