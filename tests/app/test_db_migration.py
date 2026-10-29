"""Tests for SQLite database schema migrations, compatibility with legacy DBs, and idempotency."""

import sqlite3
from pathlib import Path
import pytest

from app.db.app_db import AppDatabase

OLD_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_login_at TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    video_id TEXT,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    playback_path TEXT,
    size_bytes INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'indexing', 'completed', 'failed', 'deleting')),
    stage TEXT,
    error_message TEXT,
    duration_seconds REAL,
    has_audio INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(user_id, content_sha256),
    UNIQUE(user_id, video_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_videos_user_id ON videos(user_id);
CREATE INDEX IF NOT EXISTS idx_videos_video_id ON videos(video_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
"""


def test_fresh_db_initialization(tmp_path: Path):
    """Requirement 8: Fresh database initializes with all required tables, columns, and indices."""
    db_file = tmp_path / "fresh_test.db"
    db = AppDatabase(db_path=db_file)

    # Check columns in users table
    cols = db.execute_read("PRAGMA table_info(users)")
    col_names = {c["name"] for c in cols}
    assert "email_verified" in col_names
    assert "email_verified_at" in col_names
    assert "google_sub" in col_names
    assert "has_usable_password" in col_names

    # Check supporting tables
    tables = db.execute_read("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = {t["name"] for t in tables}
    assert "pending_registrations" in table_names
    assert "email_send_log" in table_names

    # Check unique partial index on google_sub
    indexes = db.execute_read("SELECT name, sql FROM sqlite_master WHERE type='index' AND name='idx_users_google_sub'")
    assert len(indexes) == 1
    sql_upper = (indexes[0]["sql"] or "").upper()
    assert "WHERE GOOGLE_SUB IS NOT NULL" in sql_upper


def test_migration_from_old_schema_preserves_rows(tmp_path: Path):
    """Requirement 8: Existing database created with old schema is migrated and preserves rows."""
    db_file = tmp_path / "legacy_test.db"

    # Pre-populate DB with old schema and rows
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.executescript(OLD_SCHEMA_SQL)
    conn.execute(
        """
        INSERT INTO users (email, display_name, password_hash, created_at, failed_attempts)
        VALUES ('legacy@example.com', 'Legacy User', 'argon_hash_legacy', '2026-01-01T00:00:00Z', 0)
        """
    )
    conn.commit()
    conn.close()

    # Now open with AppDatabase to trigger migration
    db = AppDatabase(db_path=db_file)

    # Check migrated columns
    cols = db.execute_read("PRAGMA table_info(users)")
    col_names = {c["name"] for c in cols}
    assert "email_verified" in col_names
    assert "email_verified_at" in col_names
    assert "google_sub" in col_names
    assert "has_usable_password" in col_names

    # Verify existing data preserved
    row = db.execute_read_one("SELECT * FROM users WHERE email = 'legacy@example.com'")
    assert row is not None
    assert row["display_name"] == "Legacy User"
    assert row["password_hash"] == "argon_hash_legacy"
    assert row["email_verified"] == 0  # legacy default
    assert row["has_usable_password"] == 1  # legacy default


def test_migration_idempotency_multiple_runs(tmp_path: Path):
    """Requirement 8: Running database initialization repeatedly is a safe no-op."""
    db_file = tmp_path / "idempotent_test.db"
    db = AppDatabase(db_path=db_file)

    # Trigger initialization multiple times
    db._init_db()
    db._init_db()

    # Ensure tables still functional and healthy
    db.execute_write(
        "INSERT INTO users (email, display_name, password_hash, created_at) VALUES ('idem@example.com', 'Idem', 'hash', '2026-01-01')"
    )
    user = db.execute_read_one("SELECT * FROM users WHERE email = 'idem@example.com'")
    assert user is not None
