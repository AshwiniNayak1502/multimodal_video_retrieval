"""Unit tests for search service, user isolation, query rewrite, and Key Moments."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.db.app_db import AppDatabase
from app.services.search_service import (
    KeyMoment,
    SearchEngineWrapper,
    normalize_search_query,
)


def test_query_normalization():
    # Strip leading intent verbs
    q1, orig1 = normalize_search_query("show me the part where Java scoped values are explained")
    assert q1 == "Java scoped values are explained"
    assert orig1 == "show me the part where Java scoped values are explained"

    q2, orig2 = normalize_search_query("please can you find the moment when the chart appears")
    assert q2 == "the chart appears"

    q3, orig3 = normalize_search_query("find where the error occurs")
    assert q3 == "the error occurs"

    # Already direct query -> unchanged
    q4, orig4 = normalize_search_query("Java 25 scoped values")
    assert q4 == "Java 25 scoped values"
    assert orig4 is None

    # Off mode -> unchanged
    q5, orig5 = normalize_search_query("show me where it happened", mode="off")
    assert q5 == "show me where it happened"
    assert orig5 is None


def test_search_isolation_user_without_videos_returns_early():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = AppDatabase(db_path=Path(tmp_dir) / "search_test.db")

        # Mock wrapper to test user scoping logic
        wrapper = SearchEngineWrapper.__new__(SearchEngineWrapper)
        wrapper.service = MagicMock()
        wrapper._lock = MagicMock()

        # User 99 owns no videos
        resp = SearchEngineWrapper.search_user_videos(
            wrapper,
            user_id=99,
            query="test query",
            db=db,
        )
        assert resp.status == "ok"
        assert len(resp.results) == 0
        assert "upload a video first" in resp.message.lower()
        # Ensure underlying neural index was NEVER called
        wrapper.service.search.assert_not_called()


def test_search_user_isolation_scoping_to_owned_videos():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = AppDatabase(db_path=Path(tmp_dir) / "search_test.db")

        u1 = db.execute_write("INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)", ("a@x.com", "A", "h", "now"))
        u2 = db.execute_write("INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)", ("b@x.com", "B", "h", "now"))

        # User 1 has completed video vid_001
        db.execute_write(
            "INSERT INTO videos (user_id, video_id, original_filename, stored_path, size_bytes, content_sha256, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'completed', 'now')",
            (u1, "vid_001", "a.mp4", "/path/a.mp4", 100, "hash1"),
        )

        # User 2 has completed video vid_002
        db.execute_write(
            "INSERT INTO videos (user_id, video_id, original_filename, stored_path, size_bytes, content_sha256, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'completed', 'now')",
            (u2, "vid_002", "b.mp4", "/path/b.mp4", 100, "hash2"),
        )

        wrapper = SearchEngineWrapper.__new__(SearchEngineWrapper)
        mock_svc = MagicMock()
        mock_svc.search.return_value = MagicMock(debug={}, results=[])
        wrapper.service = mock_svc
        wrapper._lock = MagicMock()

        # Search as User 1 -> video_ids must strictly contain only ["vid_001"]
        SearchEngineWrapper.search_user_videos(wrapper, user_id=u1, query="test", db=db)
        call_kwargs = mock_svc.search.call_args[1]
        assert call_kwargs["video_ids"] == ["vid_001"]
        assert "vid_002" not in call_kwargs["video_ids"]

        # Search as User 2 -> video_ids must strictly contain only ["vid_002"]
        SearchEngineWrapper.search_user_videos(wrapper, user_id=u2, query="test", db=db)
        call_kwargs2 = mock_svc.search.call_args[1]
        assert call_kwargs2["video_ids"] == ["vid_002"]
        assert "vid_001" not in call_kwargs2["video_ids"]


def test_key_moments_assembly():
    wrapper = SearchEngineWrapper.__new__(SearchEngineWrapper)
    mock_svc = MagicMock()
    mock_index = MagicMock()

    # Mock surrounding segments in window
    mock_index.get_segments_in_window.return_value = [
        {"start_timestamp": 10.5, "start_formatted": "00:00:10", "transcript_text": "First concept explained"},
        {"start_timestamp": 25.0, "start_formatted": "00:00:25", "transcript_text": "Second concept detailed"},
    ]
    mock_svc.index = mock_index
    wrapper.service = mock_svc
    wrapper._lock = MagicMock()

    # Search hit at 15.0s
    mock_hit = MagicMock()
    mock_hit.video_id = "vid_123"
    mock_hit.seek_seconds = 15.0
    mock_hit.start_formatted = "00:00:15"
    mock_hit.transcript_text = "Direct match segment"
    mock_hit.frame_path = "/path/frame.jpg"

    moments = SearchEngineWrapper.get_key_moments(
        wrapper,
        video_id="vid_123",
        active_start_s=15.0,
        active_end_s=20.0,
        all_search_results=[mock_hit],
        window_s=30.0,
    )

    assert len(moments) == 3
    # Check chronological ordering
    ts = [m.timestamp_seconds for m in moments]
    assert ts == sorted(ts)
    assert ts == [10.5, 15.0, 25.0]
    assert moments[1].label == "Search Match"
    assert moments[0].label == "Transcript"
