"""Unit and integration tests for video deletion feature."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import faiss
import pytest
import streamlit as st

from app.db.app_db import AppDatabase
from app.services.auth_service import User
from app.services.video_service import (
    delete_user_video,
    list_user_videos,
    reconcile_interrupted_deletions,
    save_uploaded_stream,
)
from app.settings import settings
from app.ui.library_view import render_library_view, show_delete_confirmation

# Import Module 2 test fixtures and classes
import sys
_M2_DIR = settings.project_root / "module2"
if str(_M2_DIR) not in sys.path:
    sys.path.insert(0, str(_M2_DIR))

from embedding_store.config import Module2Config
from embedding_store.embedders import FakeTextEmbedder, FakeVisualEmbedder
from embedding_store.indexer import VideoIndexer
from embedding_store.metadata_store import MetadataStore
from embedding_store.validation import validate_integrity
from embedding_store.vector_index import load_manifest, load_or_rebuild_index


@pytest.fixture
def test_env(tmp_path: Path, monkeypatch):
    """Hermetic isolated environment for app DB, storage, and Module 2 vector DB."""
    app_db_path = tmp_path / "app.db"
    db = AppDatabase(db_path=app_db_path)

    # Monkeypatch settings to point to tmp_path
    storage_root = tmp_path / "module1" / "input" / "videos"
    storage_root.mkdir(parents=True, exist_ok=True)
    m1_out = tmp_path / "module1" / "output"
    (m1_out / "frames").mkdir(parents=True, exist_ok=True)
    (m1_out / "audio").mkdir(parents=True, exist_ok=True)
    (m1_out / "transcripts").mkdir(parents=True, exist_ok=True)
    (m1_out / "metadata").mkdir(parents=True, exist_ok=True)

    playback_dir = tmp_path / "data" / "playback"
    playback_dir.mkdir(parents=True, exist_ok=True)

    vdb_dir = tmp_path / "module2" / "output" / "vector_db"
    vdb_dir.mkdir(parents=True, exist_ok=True)

    orig_root = settings.project_root
    orig_playback = settings.playback_dir
    object.__setattr__(settings, "project_root", tmp_path)
    object.__setattr__(settings, "playback_dir", playback_dir)
    monkeypatch.setattr("app.services.video_service.get_storage_root", lambda: storage_root)

    # Create two test users
    u1_id = db.execute_write(
        "INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
        ("alice@example.com", "Alice", "hashA", "2026-01-01T00:00:00Z"),
    )
    u2_id = db.execute_write(
        "INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
        ("bob@example.com", "Bob", "hashB", "2026-01-01T00:00:00Z"),
    )

    m2_cfg = Module2Config(
        base_dir=tmp_path / "module2",
        output_dir=tmp_path / "module2" / "output",
        vector_db_dir=vdb_dir,
        visual_model_name="mock/siglip-fake",
        text_model_name="mock/minilm-fake",
        offline_mode=True,
    )
    m2_cfg.ensure_directories()

    try:
        yield {
            "db": db,
            "u1": u1_id,
            "u2": u2_id,
            "tmp": tmp_path,
            "storage_root": storage_root,
            "m1_out": m1_out,
            "playback_dir": playback_dir,
            "m2_cfg": m2_cfg,
        }
    finally:
        object.__setattr__(settings, "project_root", orig_root)
        object.__setattr__(settings, "playback_dir", orig_playback)


def _seed_m2_vectors(m2_cfg: Module2Config, video_id: str, num_vis: int = 3, num_txt: int = 2):
    """Seed Module 2 metadata.db and FAISS indexes with dummy vectors using fake embedders."""
    from PIL import Image
    import numpy as np

    store = MetadataStore(m2_cfg.vector_db_dir / "metadata.db")
    store.set_meta("visual_dim", "768")
    store.set_meta("text_dim", "384")
    store.set_meta("visual_model", "mock/siglip-fake")
    store.set_meta("text_model", "mock/minilm-fake")

    from embedding_store.schema import VideoRecord, VisualItem, TextItem
    v_rec = VideoRecord(
        video_id=video_id,
        video_filename=f"{video_id}.mp4",
        video_path=f"/fake/{video_id}.mp4",
        video_relpath=None,
        duration_seconds=10.0,
        fps=1.0,
        width=64,
        height=64,
        frame_interval_seconds=2.0,
        has_audio=1,
        language="en",
        transcript_path=None,
        m1_metadata_path=f"/fake/{video_id}.json",
        m1_processed_at="2026-01-01T00:00:00Z",
        indexed_at="2026-01-01T00:00:00Z",
    )
    store.upsert_video(v_rec)

    vis_mgr = load_or_rebuild_index("visual", m2_cfg.vector_db_dir / "visual.index", m2_cfg.vector_db_dir / "index_manifest.json", store, 768, "mock/siglip-fake")
    txt_mgr = load_or_rebuild_index("text", m2_cfg.vector_db_dir / "text.index", m2_cfg.vector_db_dir / "index_manifest.json", store, 384, "mock/minilm-fake")

    vis_ids = []
    vis_vecs = []
    for i in range(num_vis):
        item = VisualItem(
            frame_id=f"{video_id}_f_{i}",
            video_id=video_id,
            frame_index=i,
            source_frame_number=i,
            timestamp_seconds=float(i * 2),
            timestamp_formatted=f"00:0{i * 2}",
            frame_path=f"/fake/{video_id}/f_{i}.jpg",
            frame_relpath=None,
            width=64,
            height=64,
            content_hash=f"hash_{i}",
        )
        vec = np.ones(768, dtype=np.float32) / np.sqrt(768)
        vid = store.insert_visual_vector(item, vec, "mock/siglip-fake")
        vis_ids.append(vid)
        vis_vecs.append(vec)

    txt_ids = []
    txt_vecs = []
    for j in range(num_txt):
        t_item = TextItem(
            segment_uid=f"{video_id}_s_{j}",
            video_id=video_id,
            segment_id=j,
            transcript_text=f"Test transcript speech {j}",
            start_timestamp=float(j * 4),
            end_timestamp=float((j + 1) * 4),
            start_formatted=f"00:0{j * 4}",
            end_formatted=f"00:0{(j + 1) * 4}",
            language="en",
            content_hash=f"thash_{j}",
        )
        tvec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        tid = store.insert_text_vector(t_item, tvec, "mock/minilm-fake")
        txt_ids.append(tid)
        txt_vecs.append(tvec)

    vis_mgr.add(np.vstack(vis_vecs), vis_ids)
    txt_mgr.add(np.vstack(txt_vecs), txt_ids)
    vis_mgr.save_index(m2_cfg.vector_db_dir / "visual.index")
    txt_mgr.save_index(m2_cfg.vector_db_dir / "text.index")

    from embedding_store.vector_index import save_manifest
    save_manifest(m2_cfg.vector_db_dir / "index_manifest.json", {
        "schema_version": "2.0.0",
        "models": {"visual": "mock/siglip-fake", "text": "mock/minilm-fake"},
        "dims": {"visual": 768, "text": 384},
        "counts": {"visual": vis_mgr.ntotal, "text": txt_mgr.ntotal},
    })
    store.connection.commit()
    store.close()


# ---------------------------------------------------------------------------
# TEST 1: Regression for Argument-Order Bug in UI Callback
# ---------------------------------------------------------------------------
def test_ui_argument_order_regression(monkeypatch):
    """Verify library UI invokes delete_user_video with exact (user_id, video_db_id) keyword arguments."""
    called_kwargs = {}

    def mock_delete_user_video(user_id: int, video_db_id: int, db=None, on_index_changed=None):
        called_kwargs["user_id"] = user_id
        called_kwargs["video_db_id"] = video_db_id
        return True, "Video deleted."

    monkeypatch.setattr("app.ui.library_view.delete_user_video", mock_delete_user_video)

    test_user = User(id=42, email="user42@example.com", display_name="User 42", created_at="2026-01-01T00:00:00Z")
    test_video = {"id": 105, "original_filename": "sample.mp4", "video_id": "vid_abc123"}

    # Mock st.button to simulate clicking "Yes, Delete"
    rerun_called = []
    monkeypatch.setattr(st, "rerun", lambda: rerun_called.append(True))

    def fake_button(label, *args, **kwargs):
        if "Yes, Delete" in label:
            return True
        return False

    monkeypatch.setattr(st, "button", fake_button)

    # Set up stale session state
    st.session_state["search_response_cache"] = {"results": ["dummy"]}
    st.session_state["selected_result_idx"] = 1
    st.session_state["search_active_query"] = "test query"
    st.session_state["player_seek_vid_abc123"] = 15
    st.session_state["last_result_vid_abc123"] = "10_20"

    index_changed_called = []
    # Test underlying dialog function directly via __wrapped__
    fn = getattr(show_delete_confirmation, "__wrapped__", show_delete_confirmation)
    fn(
        video=test_video,
        user=test_user,
        db=None,
        on_index_changed=lambda: index_changed_called.append(True),
    )

    # Assert correct argument order was received!
    assert called_kwargs.get("user_id") == 42
    assert called_kwargs.get("video_db_id") == 105

    # Assert flash message was stored
    assert st.session_state.get("flash_success") == "Video deleted."

    # Assert stale session keys were purged!
    assert "search_response_cache" not in st.session_state
    assert "selected_result_idx" not in st.session_state
    assert "search_active_query" not in st.session_state
    assert "player_seek_vid_abc123" not in st.session_state
    assert "last_result_vid_abc123" not in st.session_state

    # Assert callback was called
    assert len(index_changed_called) == 1
    assert len(rerun_called) == 1


# ---------------------------------------------------------------------------
# TEST 2: Service - Full Deletion of a Single-Owner Video
# ---------------------------------------------------------------------------
def test_full_single_owner_deletion(test_env):
    """Full deletion removes app row, upload file/dir, frames, audio, transcript, metadata JSON, proxy, M2 rows, and FAISS vectors."""
    db = test_env["db"]
    u1 = test_env["u1"]
    m1_out = test_env["m1_out"]
    playback_dir = test_env["playback_dir"]
    m2_cfg = test_env["m2_cfg"]

    # 1. Upload video
    stream = io.BytesIO(b"single owner video data 123456789")
    ok, msg, saved_row = save_uploaded_stream(u1, "single.mp4", stream, db=db)
    assert ok is True
    row_id = saved_row["id"]
    vid_id = saved_row["video_id"]
    stored_path = Path(saved_row["stored_path"])
    assert stored_path.is_file()

    # Create dummy shared artifacts
    meta_json = m1_out / "metadata" / f"{vid_id}.json"
    meta_json.write_text("{}", encoding="utf-8")
    trans_json = m1_out / "transcripts" / f"{vid_id}.json"
    trans_json.write_text("{}", encoding="utf-8")
    audio_wav = m1_out / "audio" / f"{vid_id}.wav"
    audio_wav.write_bytes(b"RIFFdummywav")
    frames_dir = m1_out / "frames" / vid_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "frame_0000.jpg").write_bytes(b"dummyjpg")
    proxy_mp4 = playback_dir / f"{vid_id}.mp4"
    proxy_mp4.write_bytes(b"dummymp4")

    # Mark video as completed and register in M2
    db.execute_write(
        "UPDATE videos SET status = 'completed', playback_path = ? WHERE id = ?",
        (str(proxy_mp4.resolve()), row_id),
    )
    _seed_m2_vectors(m2_cfg, vid_id, num_vis=3, num_txt=2)

    # Verify Module 2 initial state
    m2_store = MetadataStore(m2_cfg.vector_db_dir / "metadata.db")
    assert len(m2_store.list_videos()) == 1
    m2_store.close()

    # 2. Delete video
    del_ok, del_msg = delete_user_video(user_id=u1, video_db_id=row_id, db=db)
    assert del_ok is True
    assert "removed from your library" in del_msg.lower()

    # 3. Assert App DB row removed
    assert db.execute_read_one("SELECT * FROM videos WHERE id = ?", (row_id,)) is None

    # 4. Assert user's own upload file and hex dir removed
    assert not stored_path.is_file()
    assert not stored_path.parent.exists()

    # 5. Assert shared artifacts removed
    assert not meta_json.is_file()
    assert not trans_json.is_file()
    assert not audio_wav.is_file()
    assert not frames_dir.exists()
    assert not proxy_mp4.is_file()

    # 6. Assert Module 2 rows and vectors purged
    m2_store = MetadataStore(m2_cfg.vector_db_dir / "metadata.db")
    assert len(m2_store.list_videos()) == 0
    cur = m2_store.connection.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM visual_vectors WHERE video_id = ?", (vid_id,))
    assert cur.fetchone()["c"] == 0
    cur.execute("SELECT COUNT(*) AS c FROM text_vectors WHERE video_id = ?", (vid_id,))
    assert cur.fetchone()["c"] == 0
    m2_store.close()

    # 7. Assert FAISS indices emptied and manifest counts updated
    vis_index = faiss.read_index(str(m2_cfg.vector_db_dir / "visual.index"))
    txt_index = faiss.read_index(str(m2_cfg.vector_db_dir / "text.index"))
    assert vis_index.ntotal == 0
    assert txt_index.ntotal == 0

    manifest = load_manifest(m2_cfg.vector_db_dir / "index_manifest.json")
    assert manifest["counts"]["visual"] == 0
    assert manifest["counts"]["text"] == 0


# ---------------------------------------------------------------------------
# TEST 3: Other Videos Unaffected
# ---------------------------------------------------------------------------
def test_other_videos_unaffected_when_deleting_one(test_env):
    """When video A and video B are indexed, deleting A leaves B completely intact."""
    db = test_env["db"]
    u1 = test_env["u1"]
    m2_cfg = test_env["m2_cfg"]
    m1_out = test_env["m1_out"]

    # Upload video A
    stream_a = io.BytesIO(b"video content AAAAAA 111")
    _, _, row_a = save_uploaded_stream(u1, "videoA.mp4", stream_a, db=db)
    # Upload video B
    stream_b = io.BytesIO(b"video content BBBBBB 222")
    _, _, row_b = save_uploaded_stream(u1, "videoB.mp4", stream_b, db=db)

    db.execute_write("UPDATE videos SET status = 'completed' WHERE id IN (?, ?)", (row_a["id"], row_b["id"]))

    vid_a = row_a["video_id"]
    vid_b = row_b["video_id"]

    # Seed M2 for both
    _seed_m2_vectors(m2_cfg, vid_a, num_vis=2, num_txt=1)
    _seed_m2_vectors(m2_cfg, vid_b, num_vis=4, num_txt=2)

    # Create dummy metadata JSON for both
    meta_a = m1_out / "metadata" / f"{vid_a}.json"
    meta_a.write_text("{}", encoding="utf-8")
    meta_b = m1_out / "metadata" / f"{vid_b}.json"
    meta_b.write_text("{}", encoding="utf-8")

    # Delete video A
    del_ok, _ = delete_user_video(user_id=u1, video_db_id=row_a["id"], db=db)
    assert del_ok is True

    # Video A is gone
    assert db.execute_read_one("SELECT * FROM videos WHERE id = ?", (row_a["id"],)) is None
    assert not meta_a.exists()

    # Video B remains completely intact
    b_in_db = db.execute_read_one("SELECT * FROM videos WHERE id = ?", (row_b["id"],))
    assert b_in_db is not None
    assert Path(b_in_db["stored_path"]).is_file()
    assert meta_b.is_file()

    # Module 2 has only video B
    m2_store = MetadataStore(m2_cfg.vector_db_dir / "metadata.db")
    vids = m2_store.list_videos()
    assert len(vids) == 1
    assert vids[0]["video_id"] == vid_b
    m2_store.close()

    vis_index = faiss.read_index(str(m2_cfg.vector_db_dir / "visual.index"))
    txt_index = faiss.read_index(str(m2_cfg.vector_db_dir / "text.index"))
    assert vis_index.ntotal == 4
    assert txt_index.ntotal == 2


# ---------------------------------------------------------------------------
# TEST 4: Shared Content Multi-User Refcount Lifecycle
# ---------------------------------------------------------------------------
def test_shared_content_multi_user_lifecycle(test_env):
    """Two users share video_id. User 1 deletes: own file removed; shared artifacts & vectors intact for User 2. User 2 deletes: full purge."""
    db = test_env["db"]
    u1 = test_env["u1"]
    u2 = test_env["u2"]
    m1_out = test_env["m1_out"]
    m2_cfg = test_env["m2_cfg"]

    shared_content = b"IDENTICAL VIDEO STREAM SHARED ACROSS USERS 9999"

    # User 1 uploads
    s1 = io.BytesIO(shared_content)
    _, _, row1 = save_uploaded_stream(u1, "shared_clip.mp4", s1, db=db)
    vid_id = row1["video_id"]
    db.execute_write("UPDATE videos SET status = 'completed' WHERE id = ?", (row1["id"],))

    # User 2 uploads identical content -> reused index
    s2 = io.BytesIO(shared_content)
    _, _, row2 = save_uploaded_stream(u2, "shared_clip.mp4", s2, db=db)
    assert row2["video_id"] == vid_id

    u1_file = Path(row1["stored_path"])
    u2_file = Path(row2["stored_path"])
    assert u1_file != u2_file
    assert u1_file.is_file()
    assert u2_file.is_file()

    # Create shared metadata file
    shared_meta = m1_out / "metadata" / f"{vid_id}.json"
    shared_meta.write_text("{}", encoding="utf-8")

    # Seed M2 vectors
    _seed_m2_vectors(m2_cfg, vid_id, num_vis=3, num_txt=2)

    # 1. User 1 deletes their video
    del_ok1, _ = delete_user_video(user_id=u1, video_db_id=row1["id"], db=db)
    assert del_ok1 is True

    # User 1's own upload file is removed
    assert not u1_file.is_file()
    assert not u1_file.parent.exists()

    # User 2's file and row are intact!
    assert u2_file.is_file()
    assert db.execute_read_one("SELECT * FROM videos WHERE id = ?", (row2["id"],)) is not None

    # Shared artifacts and Module 2 vectors remain untouched
    assert shared_meta.is_file()
    vis_index = faiss.read_index(str(m2_cfg.vector_db_dir / "visual.index"))
    assert vis_index.ntotal == 3

    # 2. User 2 deletes their video
    del_ok2, _ = delete_user_video(user_id=u2, video_db_id=row2["id"], db=db)
    assert del_ok2 is True

    # User 2's file removed
    assert not u2_file.is_file()
    assert not u2_file.parent.exists()

    # Now all shared artifacts and Module 2 vectors are purged!
    assert not shared_meta.is_file()
    vis_index = faiss.read_index(str(m2_cfg.vector_db_dir / "visual.index"))
    assert vis_index.ntotal == 0


# ---------------------------------------------------------------------------
# TEST 5: Authorization Security
# ---------------------------------------------------------------------------
def test_deletion_authorization_enforced(test_env):
    """User B cannot delete User A's video."""
    db = test_env["db"]
    u1 = test_env["u1"]
    u2 = test_env["u2"]

    s1 = io.BytesIO(b"alice private content")
    _, _, row1 = save_uploaded_stream(u1, "alice.mp4", s1, db=db)

    del_forbidden, err = delete_user_video(user_id=u2, video_db_id=row1["id"], db=db)
    assert del_forbidden is False
    assert "access denied" in err.lower() or "not found" in err.lower()

    # Alice's row still exists
    assert db.execute_read_one("SELECT * FROM videos WHERE id = ?", (row1["id"],)) is not None


# ---------------------------------------------------------------------------
# TEST 6: Status Gating
# ---------------------------------------------------------------------------
def test_status_gating_processing_and_pending(test_env):
    """Processing/indexing is refused; pending is deletable and marked 'deleting' immediately."""
    db = test_env["db"]
    u1 = test_env["u1"]

    s = io.BytesIO(b"gating content 1234")
    _, _, row = save_uploaded_stream(u1, "gating.mp4", s, db=db)
    row_id = row["id"]

    # 1. Test 'processing' status refusal
    db.execute_write("UPDATE videos SET status = 'processing' WHERE id = ?", (row_id,))
    ok_proc, msg_proc = delete_user_video(user_id=u1, video_db_id=row_id, db=db)
    assert ok_proc is False
    assert "currently processing" in msg_proc.lower()
    assert db.execute_read_one("SELECT status FROM videos WHERE id = ?", (row_id,))["status"] == "processing"

    # 2. Test 'indexing' status refusal
    db.execute_write("UPDATE videos SET status = 'indexing' WHERE id = ?", (row_id,))
    ok_idx, msg_idx = delete_user_video(user_id=u1, video_db_id=row_id, db=db)
    assert ok_idx is False
    assert "currently indexing" in msg_idx.lower()

    # 3. Test 'pending' status: deletable and deleted cleanly
    db.execute_write("UPDATE videos SET status = 'pending' WHERE id = ?", (row_id,))
    ok_pend, msg_pend = delete_user_video(user_id=u1, video_db_id=row_id, db=db)
    assert ok_pend is True
    assert db.execute_read_one("SELECT * FROM videos WHERE id = ?", (row_id,)) is None


# ---------------------------------------------------------------------------
# TEST 7: Failure Injection and Idempotency
# ---------------------------------------------------------------------------
def test_failure_injection_and_idempotent_retry(test_env, monkeypatch):
    """Forced file removal error leaves row in 'deleting' state; retry completes cleanly."""
    db = test_env["db"]
    u1 = test_env["u1"]

    s = io.BytesIO(b"failure injection content")
    _, _, row = save_uploaded_stream(u1, "fail_inject.mp4", s, db=db)
    row_id = row["id"]
    db.execute_write("UPDATE videos SET status = 'completed' WHERE id = ?", (row_id,))

    # Inject failure on safe_unlink_file
    def mock_fail_unlink(path, *args, **kwargs):
        raise PermissionError(f"Simulated Windows file lock on {path.name}")

    monkeypatch.setattr("app.services.video_service.safe_unlink_file", mock_fail_unlink)

    # Attempt deletion -> fails gracefully
    ok, err = delete_user_video(user_id=u1, video_db_id=row_id, db=db)
    assert ok is False
    assert "Simulated Windows file lock" in err

    # Verify row left in 'deleting' state with error message
    db_row = db.execute_read_one("SELECT * FROM videos WHERE id = ?", (row_id,))
    assert db_row is not None
    assert db_row["status"] == "deleting"
    assert "Simulated Windows file lock" in db_row["error_message"]

    # Now un-monkeypatch (unlock file) and retry
    monkeypatch.undo()
    monkeypatch.setattr("app.services.video_service.get_storage_root", lambda: test_env["storage_root"])

    ok_retry, msg_retry = delete_user_video(user_id=u1, video_db_id=row_id, db=db)
    assert ok_retry is True
    assert db.execute_read_one("SELECT * FROM videos WHERE id = ?", (row_id,)) is None


# ---------------------------------------------------------------------------
# TEST 8: Startup Reconciliation of 'deleting' Rows
# ---------------------------------------------------------------------------
def test_reconcile_interrupted_deletions_on_startup(test_env):
    """Rows left in 'deleting' from a crashed session are reconciled on startup."""
    db = test_env["db"]
    u1 = test_env["u1"]

    s = io.BytesIO(b"interrupted deletion content")
    _, _, row = save_uploaded_stream(u1, "interrupted.mp4", s, db=db)
    row_id = row["id"]

    # Put row into 'deleting'
    db.execute_write("UPDATE videos SET status = 'deleting' WHERE id = ?", (row_id,))

    # Simulate startup reconciliation
    reconciled_count = reconcile_interrupted_deletions(db=db)
    assert reconciled_count == 1
    assert db.execute_read_one("SELECT * FROM videos WHERE id = ?", (row_id,)) is None
