"""Tests for Module 2 video vector purge helper and integrity persistence."""

from pathlib import Path
import faiss
import pytest

from embedding_store.config import Module2Config
from embedding_store.embedders import FakeTextEmbedder, FakeVisualEmbedder
from embedding_store.indexer import VideoIndexer, purge_video_from_store
from embedding_store.metadata_store import MetadataStore
from embedding_store.validation import validate_integrity
from embedding_store.vector_index import load_manifest, load_or_rebuild_index


def test_purge_video_from_store_removes_sqlite_faiss_manifest(temp_config: Module2Config):
    """Verify purge_video_from_store removes video rows, vectors, and manifest counts without ML models."""
    # 1. Index 2 candidate videos from synthetic fixture using fake embedders
    indexer = VideoIndexer(
        config=temp_config,
        visual_embedder=FakeVisualEmbedder(dim=768),
        text_embedder=FakeTextEmbedder(dim=384),
    )
    summary = indexer.run_indexing()
    assert len(indexer.store.list_videos()) == 2
    assert indexer.vis_mgr is not None
    assert indexer.txt_mgr is not None

    total_vis_before = indexer.vis_mgr.ntotal
    total_txt_before = indexer.txt_mgr.ntotal
    assert total_vis_before > 0
    assert total_txt_before > 0

    # Ensure validation passes initially
    assert validate_integrity(temp_config) is True

    # 2. Purge vid_silent002 without using VideoIndexer or ML models
    vis_removed, txt_removed = purge_video_from_store("vid_silent002", config=temp_config)
    assert vis_removed > 0  # silent video has visual frames

    # 3. Verify SQLite state
    store = MetadataStore(temp_config.vector_db_dir / "metadata.db")
    videos = store.list_videos()
    assert len(videos) == 1
    assert videos[0]["video_id"] == "vid_test001"

    cur = store.connection.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM visual_vectors WHERE video_id = 'vid_silent002'")
    assert cur.fetchone()["c"] == 0
    cur.execute("SELECT COUNT(*) AS c FROM text_vectors WHERE video_id = 'vid_silent002'")
    assert cur.fetchone()["c"] == 0

    # Remaining SQLite counts
    cur.execute("SELECT vector_id FROM visual_vectors")
    sql_vis_ids = set(int(r["vector_id"]) for r in cur.fetchall())
    cur.execute("SELECT vector_id FROM text_vectors")
    sql_txt_ids = set(int(r["vector_id"]) for r in cur.fetchall())
    store.close()

    # 4. Verify FAISS indices on disk
    vis_index = faiss.read_index(str(temp_config.vector_db_dir / "visual.index"))
    txt_index = faiss.read_index(str(temp_config.vector_db_dir / "text.index"))
    assert vis_index.ntotal == total_vis_before - vis_removed
    assert vis_index.ntotal == len(sql_vis_ids)
    assert txt_index.ntotal == total_txt_before - txt_removed
    assert txt_index.ntotal == len(sql_txt_ids)

    # 5. Verify manifest counts match
    manifest = load_manifest(temp_config.vector_db_dir / "index_manifest.json")
    assert manifest["counts"]["visual"] == vis_index.ntotal
    assert manifest["counts"]["text"] == txt_index.ntotal

    # 6. Verify full validation suite passes (100% PASS on all 9 checks)
    assert validate_integrity(temp_config) is True


def test_purge_restart_simulation_no_resurrection(temp_config: Module2Config):
    """Simulate app restart after purge: load_or_rebuild_index does NOT resurrect purged vectors."""
    indexer = VideoIndexer(
        config=temp_config,
        visual_embedder=FakeVisualEmbedder(dim=768),
        text_embedder=FakeTextEmbedder(dim=384),
    )
    indexer.run_indexing()

    # Purge vid_test001
    vis_rem, txt_rem = purge_video_from_store("vid_test001", config=temp_config)
    assert vis_rem > 0
    assert txt_rem > 0

    # Restart simulation: open brand new MetadataStore and invoke load_or_rebuild_index
    store = MetadataStore(temp_config.vector_db_dir / "metadata.db")
    vis_mgr = load_or_rebuild_index(
        modality="visual",
        index_file=temp_config.vector_db_dir / "visual.index",
        manifest_file=temp_config.vector_db_dir / "index_manifest.json",
        store=store,
        expected_dim=768,
        expected_model="mock/siglip-fake",
    )
    txt_mgr = load_or_rebuild_index(
        modality="text",
        index_file=temp_config.vector_db_dir / "text.index",
        manifest_file=temp_config.vector_db_dir / "index_manifest.json",
        store=store,
        expected_dim=384,
        expected_model="mock/minilm-fake",
    )

    # Make sure vid_test001 is nowhere in SQLite or FAISS
    cur = store.connection.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM videos WHERE video_id = 'vid_test001'")
    assert cur.fetchone()["c"] == 0
    cur.execute("SELECT COUNT(*) AS c FROM visual_vectors WHERE video_id = 'vid_test001'")
    assert cur.fetchone()["c"] == 0
    cur.execute("SELECT COUNT(*) AS c FROM text_vectors WHERE video_id = 'vid_test001'")
    assert cur.fetchone()["c"] == 0

    # Vid silent002 vectors should remain intact
    cur.execute("SELECT COUNT(*) AS c FROM visual_vectors WHERE video_id = 'vid_silent002'")
    silent_vis_count = cur.fetchone()["c"]
    assert silent_vis_count > 0
    assert vis_mgr.ntotal == silent_vis_count
    store.close()

