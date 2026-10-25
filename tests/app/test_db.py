"""Unit tests for SQLite application database layer."""

import tempfile
from pathlib import Path
import pytest

from app.db.app_db import AppDatabase


def test_db_init_and_tables_creation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_app.db"
        db = AppDatabase(db_path=db_path)
        assert db_path.is_file()

        # Check tables existence
        tables = db.execute_read(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {t["name"] for t in tables}
        assert "users" in table_names
        assert "sessions" in table_names
        assert "videos" in table_names


def test_db_foreign_keys_and_cascades():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_app.db"
        db = AppDatabase(db_path=db_path)

        user_id = db.execute_write(
            "INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            ("user@example.com", "User", "hash123", "2026-01-01T00:00:00Z"),
        )
        assert user_id > 0

        # Insert session
        db.execute_write(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            ("tokenhash1", user_id, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        )

        # Check session exists
        sess = db.execute_read_one("SELECT * FROM sessions WHERE token_hash = ?", ("tokenhash1",))
        assert sess is not None
        assert sess["user_id"] == user_id

        # Delete user -> cascade deletes session
        db.execute_write("DELETE FROM users WHERE id = ?", (user_id,))
        sess_after = db.execute_read_one("SELECT * FROM sessions WHERE token_hash = ?", ("tokenhash1",))
        assert sess_after is None


def test_db_persistence_across_reopen():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_app.db"
        db1 = AppDatabase(db_path=db_path)
        uid = db1.execute_write(
            "INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            ("persist@example.com", "Persist", "hash1", "2026-01-01T00:00:00Z"),
        )
        assert uid > 0

        # Open new database instance pointing to same file
        db2 = AppDatabase(db_path=db_path)
        row = db2.execute_read_one("SELECT * FROM users WHERE id = ?", (uid,))
        assert row is not None
        assert row["email"] == "persist@example.com"
