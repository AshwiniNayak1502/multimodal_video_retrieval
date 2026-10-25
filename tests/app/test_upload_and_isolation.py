"""Unit tests for upload validation, path traversal security, and user isolation."""

import io
import tempfile
from pathlib import Path
import pytest

from app.db.app_db import AppDatabase
from app.services.video_service import (
    delete_user_video,
    list_user_videos,
    sanitize_filename,
    save_uploaded_stream,
    validate_upload_extension,
)


@pytest.fixture
def isolated_env():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "isolation_test.db"
        db = AppDatabase(db_path=db_path)

        # Create two test users
        u1_id = db.execute_write(
            "INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            ("user_a@example.com", "User A", "hashA", "2026-01-01T00:00:00Z"),
        )
        u2_id = db.execute_write(
            "INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            ("user_b@example.com", "User B", "hashB", "2026-01-01T00:00:00Z"),
        )

        yield {"db": db, "u1": u1_id, "u2": u2_id, "tmp": Path(tmp_dir)}


def test_filename_sanitization():
    assert sanitize_filename("../../../etc/passwd.mp4") == "passwd.mp4"
    assert sanitize_filename("..\\..\\malicious.mp4") == "malicious.mp4"
    assert sanitize_filename("Normal Video (1).mp4") == "Normal Video _1_.mp4"
    assert sanitize_filename("clean_clip.mkv") == "clean_clip.mkv"


def test_upload_extension_validation():
    ok, _ = validate_upload_extension("video.mp4")
    assert ok is True
    ok, _ = validate_upload_extension("clip.MKV")
    assert ok is True
    ok, err = validate_upload_extension("exploit.exe")
    assert ok is False
    assert "unsupported" in err.lower()
    ok, err = validate_upload_extension("script.py")
    assert ok is False


def test_duplicate_upload_rejection_same_user(isolated_env):
    db = isolated_env["db"]
    u1 = isolated_env["u1"]

    dummy_content = b"fake video bytes for testing 1234567890"

    # First upload succeeds
    stream1 = io.BytesIO(dummy_content)
    ok1, msg1, row1 = save_uploaded_stream(u1, "sample.mp4", stream1, db=db)
    assert ok1 is True
    assert row1 is not None

    # Exact duplicate by same user is rejected
    stream2 = io.BytesIO(dummy_content)
    ok2, msg2, row2 = save_uploaded_stream(u1, "sample.mp4", stream2, db=db)
    assert ok2 is False
    assert "already uploaded" in msg2.lower()
    assert row2 is None


def test_cross_user_shared_content_and_refcounted_delete(isolated_env):
    db = isolated_env["db"]
    u1 = isolated_env["u1"]
    u2 = isolated_env["u2"]

    dummy_content = b"identical content uploaded by both users"

    # User A uploads
    s1 = io.BytesIO(dummy_content)
    ok1, _, r1 = save_uploaded_stream(u1, "lecture.mp4", s1, db=db)
    assert ok1 is True
    vid_id = r1["video_id"]
    r1_id = r1["id"]

    # Mark as completed in DB to simulate finished indexing
    db.execute_write(
        "UPDATE videos SET status = 'completed', duration_seconds = 10.0 WHERE id = ?",
        (r1_id,),
    )

    # User B uploads exact same video -> instant index reuse
    s2 = io.BytesIO(dummy_content)
    ok2, msg2, r2 = save_uploaded_stream(u2, "lecture.mp4", s2, db=db)
    assert ok2 is True
    assert "ready instantly" in msg2.lower()
    assert r2["status"] == "completed"
    assert r2["video_id"] == vid_id
    r2_id = r2["id"]

    # Check both users own their independent rows
    vids_a = list_user_videos(u1, db=db)
    vids_b = list_user_videos(u2, db=db)
    assert len(vids_a) == 1
    assert len(vids_b) == 1
    assert vids_a[0]["id"] == r1_id
    assert vids_b[0]["id"] == r2_id

    # User A deletes their video
    del_ok1, _ = delete_user_video(u1, r1_id, db=db)
    assert del_ok1 is True

    # User A now has 0 videos, but User B still owns theirs!
    assert len(list_user_videos(u1, db=db)) == 0
    assert len(list_user_videos(u2, db=db)) == 1

    # User B deletes their video
    del_ok2, _ = delete_user_video(u2, r2_id, db=db)
    assert del_ok2 is True
    assert len(list_user_videos(u2, db=db)) == 0


def test_user_isolation_strictly_enforced(isolated_env):
    db = isolated_env["db"]
    u1 = isolated_env["u1"]
    u2 = isolated_env["u2"]

    # User A uploads video A
    s1 = io.BytesIO(b"content for user A only")
    save_uploaded_stream(u1, "userA_clip.mp4", s1, db=db)

    # User B uploads video B
    s2 = io.BytesIO(b"content for user B only")
    save_uploaded_stream(u2, "userB_clip.mp4", s2, db=db)

    list_a = list_user_videos(u1, db=db)
    list_b = list_user_videos(u2, db=db)

    assert len(list_a) == 1
    assert len(list_b) == 1
    assert list_a[0]["original_filename"] == "userA_clip.mp4"
    assert list_b[0]["original_filename"] == "userB_clip.mp4"

    # User B cannot delete User A's video
    del_forbidden, err = delete_user_video(u2, list_a[0]["id"], db=db)
    assert del_forbidden is False
    assert "denied" in err.lower() or "not found" in err.lower()
