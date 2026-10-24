"""Persistent metadata and vector BLOB storage backed by SQLite.

Implements transactional ACID storage for video records, extracted frame
metadata, speech transcript segments, and indexing error logs.
Stores L2-normalized float32 vector BLOBs enabling 100% offline rebuild of
FAISS index files without re-running any machine learning models.
"""

from __future__ import annotations

import contextlib
import datetime
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .logging_config import get_logger
from .schema import IndexFailure, TextItem, VideoRecord, VisualItem

logger = get_logger("metadata_store")

SCHEMA_VERSION = "2.0.0"
INDEX_SCHEMA_VERSION = SCHEMA_VERSION


class MetadataStore:
    """Manages SQLite persistent storage for Module 2 metadata and embeddings."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self.init_schema()

    def _connect(self) -> None:
        """Establish connection with foreign keys and WAL journal mode enabled."""
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA journal_mode = WAL;")

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager providing an atomic database transaction."""
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def init_schema(self) -> None:
        """Initialize database tables, constraints, and indexes."""
        with self.transaction() as cur:
            # 1. Meta table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

            # 2. Videos table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    video_filename TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    video_relpath TEXT,
                    duration_seconds REAL NOT NULL,
                    fps REAL NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    frame_interval_seconds REAL NOT NULL,
                    has_audio INTEGER NOT NULL,
                    language TEXT,
                    transcript_path TEXT,
                    m1_metadata_path TEXT NOT NULL,
                    m1_processed_at TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                );
            """)

            # 3. Visual vectors table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS visual_vectors (
                    vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    modality TEXT NOT NULL DEFAULT 'visual' CHECK(modality='visual'),
                    frame_id TEXT NOT NULL UNIQUE,
                    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
                    frame_index INTEGER NOT NULL,
                    source_frame_number INTEGER,
                    timestamp_seconds REAL NOT NULL CHECK(timestamp_seconds>=0),
                    timestamp_formatted TEXT NOT NULL,
                    frame_path TEXT NOT NULL,
                    frame_relpath TEXT,
                    width INTEGER,
                    height INTEGER,
                    content_hash TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

            # 4. Text vectors table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS text_vectors (
                    vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    modality TEXT NOT NULL DEFAULT 'text' CHECK(modality='text'),
                    segment_uid TEXT NOT NULL UNIQUE,
                    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
                    segment_id INTEGER NOT NULL,
                    transcript_text TEXT NOT NULL,
                    start_timestamp REAL NOT NULL CHECK(start_timestamp>=0),
                    end_timestamp REAL NOT NULL,
                    start_formatted TEXT NOT NULL,
                    end_formatted TEXT NOT NULL,
                    language TEXT,
                    content_hash TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    CHECK(end_timestamp>=start_timestamp)
                );
            """)

            # 5. Index failures table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS index_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    item_key TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    failed_at TEXT NOT NULL
                );
            """)

            # Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_visual_video_id ON visual_vectors(video_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_visual_video_ts ON visual_vectors(video_id, timestamp_seconds);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_text_video_id ON text_vectors(video_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_text_video_ts ON text_vectors(video_id, start_timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_failures_video ON index_failures(video_id);")

            # Store schema version
            cur.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?);",
                (SCHEMA_VERSION,),
            )

    def get_meta(self, key: str) -> Optional[str]:
        """Fetch a configuration value from the meta table."""
        cur = self.connection.cursor()
        cur.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Set or update a configuration key in the meta table."""
        with self.transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, str(value)),
            )

    def check_or_set_models(
        self,
        visual_model: str,
        visual_dim: int,
        text_model: str,
        text_dim: int,
    ) -> None:
        """Ensure current configuration matches stored metadata models.

        Refuses to mix different model architectures or dimensions into the same database.
        """
        stored_vm = self.get_meta("visual_model")
        stored_vd = self.get_meta("visual_dim")
        stored_tm = self.get_meta("text_model")
        stored_td = self.get_meta("text_dim")

        if stored_vm is None:
            # First initialization
            self.set_meta("visual_model", visual_model)
            self.set_meta("visual_dim", str(visual_dim))
            self.set_meta("text_model", text_model)
            self.set_meta("text_dim", str(text_dim))
            return

        mismatches: List[str] = []
        if stored_vm != visual_model:
            mismatches.append(f"visual_model (stored='{stored_vm}', configured='{visual_model}')")
        if stored_vd is None or int(stored_vd) != visual_dim:
            mismatches.append(f"visual_dim (stored={stored_vd}, configured={visual_dim})")
        if stored_tm != text_model:
            mismatches.append(f"text_model (stored='{stored_tm}', configured='{text_model}')")
        if stored_td is None or int(stored_td) != text_dim:
            mismatches.append(f"text_dim (stored={stored_td}, configured={text_dim})")

        if mismatches:
            err = (
                f"Model configuration mismatch detected in {self.db_path.name}!\n"
                f"Conflicting settings: {', '.join(mismatches)}.\n"
                "Refusing to mix vectors from different embedding models.\n"
                "To wipe and re-embed all items with the new model, run:\n"
                "  python main.py index --rebuild"
            )
            logger.error(err)
            raise ValueError(err)

    def find_video_by_path(self, video_path: str) -> Optional[str]:
        """Find video_id of an existing entry matching the given video_path."""
        cur = self.connection.cursor()
        cur.execute("SELECT video_id FROM videos WHERE video_path = ?", (video_path,))
        row = cur.fetchone()
        return row["video_id"] if row else None

    def upsert_video(self, record: VideoRecord) -> None:
        """Insert or replace a video record."""
        cur = self.connection.cursor()
        cur.execute("""
            INSERT INTO videos (
                video_id, video_filename, video_path, video_relpath,
                duration_seconds, fps, width, height, frame_interval_seconds,
                has_audio, language, transcript_path, m1_metadata_path,
                m1_processed_at, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                video_filename = excluded.video_filename,
                video_path = excluded.video_path,
                video_relpath = excluded.video_relpath,
                duration_seconds = excluded.duration_seconds,
                fps = excluded.fps,
                width = excluded.width,
                height = excluded.height,
                frame_interval_seconds = excluded.frame_interval_seconds,
                has_audio = excluded.has_audio,
                language = excluded.language,
                transcript_path = excluded.transcript_path,
                m1_metadata_path = excluded.m1_metadata_path,
                m1_processed_at = excluded.m1_processed_at,
                indexed_at = excluded.indexed_at
        """, (
            record.video_id,
            record.video_filename,
            record.video_path,
            record.video_relpath,
            record.duration_seconds,
            record.fps,
            record.width,
            record.height,
            record.frame_interval_seconds,
            1 if record.has_audio else 0,
            record.language,
            record.transcript_path,
            record.m1_metadata_path,
            record.m1_processed_at,
            record.indexed_at,
        ))

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve video record by video_id."""
        cur = self.connection.cursor()
        cur.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_videos(self) -> List[Dict[str, Any]]:
        """List all indexed video records."""
        cur = self.connection.cursor()
        cur.execute("SELECT * FROM videos ORDER BY video_id ASC")
        return [dict(r) for r in cur.fetchall()]

    def delete_video(self, video_id: str) -> Tuple[List[int], List[int]]:
        """Delete a video and cascade delete its vectors.

        Returns:
            Tuple of (removed_visual_vector_ids, removed_text_vector_ids)
        """
        cur = self.connection.cursor()
        cur.execute("SELECT vector_id FROM visual_vectors WHERE video_id = ?", (video_id,))
        vis_ids = [int(r["vector_id"]) for r in cur.fetchall()]

        cur.execute("SELECT vector_id FROM text_vectors WHERE video_id = ?", (video_id,))
        txt_ids = [int(r["vector_id"]) for r in cur.fetchall()]

        cur.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        cur.execute("DELETE FROM index_failures WHERE video_id = ?", (video_id,))

        logger.info(
            "Purged video %s (removed %d visual vectors, %d text vectors).",
            video_id,
            len(vis_ids),
            len(txt_ids),
        )
        return vis_ids, txt_ids

    def get_existing_visual_items(self, video_id: str) -> Dict[str, Dict[str, Any]]:
        """Retrieve existing visual items for video_id mapped by frame_id."""
        cur = self.connection.cursor()
        cur.execute("""
            SELECT vector_id, frame_id, content_hash, timestamp_seconds, embedding_model
            FROM visual_vectors WHERE video_id = ?
        """, (video_id,))
        return {r["frame_id"]: dict(r) for r in cur.fetchall()}

    def get_existing_text_items(self, video_id: str) -> Dict[str, Dict[str, Any]]:
        """Retrieve existing text items for video_id mapped by segment_uid."""
        cur = self.connection.cursor()
        cur.execute("""
            SELECT vector_id, segment_uid, content_hash, start_timestamp, end_timestamp, embedding_model
            FROM text_vectors WHERE video_id = ?
        """, (video_id,))
        return {r["segment_uid"]: dict(r) for r in cur.fetchall()}

    def insert_visual_vector(
        self,
        item: VisualItem,
        embedding: np.ndarray,
        embedding_model: str,
    ) -> int:
        """Insert a visual frame vector and metadata.

        Returns:
            Newly assigned auto-incrementing vector_id.
        """
        cur = self.connection.cursor()
        blob = embedding.astype(np.float32).tobytes()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cur.execute("""
            INSERT INTO visual_vectors (
                modality, frame_id, video_id, frame_index, source_frame_number,
                timestamp_seconds, timestamp_formatted, frame_path, frame_relpath,
                width, height, content_hash, embedding_model, embedding_dim,
                embedding, created_at
            ) VALUES ('visual', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.frame_id,
            item.video_id,
            item.frame_index,
            item.source_frame_number,
            item.timestamp_seconds,
            item.timestamp_formatted,
            item.frame_path,
            item.frame_relpath,
            item.width,
            item.height,
            item.content_hash,
            embedding_model,
            len(embedding),
            blob,
            now,
        ))
        return int(cur.lastrowid or 0)

    def insert_text_vector(
        self,
        item: TextItem,
        embedding: np.ndarray,
        embedding_model: str,
    ) -> int:
        """Insert a speech transcript text vector and metadata.

        Returns:
            Newly assigned auto-incrementing vector_id.
        """
        cur = self.connection.cursor()
        blob = embedding.astype(np.float32).tobytes()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cur.execute("""
            INSERT INTO text_vectors (
                modality, segment_uid, video_id, segment_id, transcript_text,
                start_timestamp, end_timestamp, start_formatted, end_formatted,
                language, content_hash, embedding_model, embedding_dim,
                embedding, created_at
            ) VALUES ('text', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.segment_uid,
            item.video_id,
            item.segment_id,
            item.transcript_text,
            item.start_timestamp,
            item.end_timestamp,
            item.start_formatted,
            item.end_formatted,
            item.language,
            item.content_hash,
            embedding_model,
            len(embedding),
            blob,
            now,
        ))
        return int(cur.lastrowid or 0)

    def delete_visual_vector(self, vector_id: int) -> None:
        """Delete an individual visual vector by vector_id."""
        cur = self.connection.cursor()
        cur.execute("DELETE FROM visual_vectors WHERE vector_id = ?", (vector_id,))

    def delete_text_vector(self, vector_id: int) -> None:
        """Delete an individual text vector by vector_id."""
        cur = self.connection.cursor()
        cur.execute("DELETE FROM text_vectors WHERE vector_id = ?", (vector_id,))

    def record_failure(self, failure: IndexFailure) -> None:
        """Record an indexing failure."""
        cur = self.connection.cursor()
        cur.execute("""
            INSERT INTO index_failures (video_id, modality, item_key, reason, failed_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            failure.video_id,
            failure.modality,
            failure.item_key,
            failure.reason,
            failure.failed_at,
        ))

    def clear_failures(self, video_id: str, modality: Optional[str] = None, item_key: Optional[str] = None) -> None:
        """Clear resolved indexing failures upon successful indexing."""
        cur = self.connection.cursor()
        if item_key and modality:
            cur.execute(
                "DELETE FROM index_failures WHERE video_id = ? AND modality = ? AND item_key = ?",
                (video_id, modality, item_key),
            )
        elif modality:
            cur.execute(
                "DELETE FROM index_failures WHERE video_id = ? AND modality = ?",
                (video_id, modality),
            )
        else:
            cur.execute("DELETE FROM index_failures WHERE video_id = ?", (video_id,))

    def get_visual_metadata_by_ids(self, vector_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Retrieve visual metadata rows for given vector IDs joined with video attributes."""
        if not vector_ids:
            return {}
        cur = self.connection.cursor()
        placeholders = ",".join("?" for _ in vector_ids)
        query = f"""
            SELECT v.vector_id, v.modality, v.frame_id, v.video_id, v.frame_index,
                   v.source_frame_number, v.timestamp_seconds, v.timestamp_formatted,
                   v.frame_path, v.frame_relpath, v.width, v.height, v.content_hash,
                   v.embedding_model,
                   vid.video_filename, vid.video_path, vid.video_relpath,
                   vid.frame_interval_seconds
            FROM visual_vectors v
            JOIN videos vid ON v.video_id = vid.video_id
            WHERE v.vector_id IN ({placeholders})
        """
        cur.execute(query, vector_ids)
        return {int(r["vector_id"]): dict(r) for r in cur.fetchall()}

    def get_text_metadata_by_ids(self, vector_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Retrieve text metadata rows for given vector IDs joined with video attributes."""
        if not vector_ids:
            return {}
        cur = self.connection.cursor()
        placeholders = ",".join("?" for _ in vector_ids)
        query = f"""
            SELECT t.vector_id, t.modality, t.segment_uid, t.video_id, t.segment_id,
                   t.transcript_text, t.start_timestamp, t.end_timestamp,
                   t.start_formatted, t.end_formatted, t.language, t.content_hash,
                   t.embedding_model,
                   vid.video_filename, vid.video_path, vid.video_relpath
            FROM text_vectors t
            JOIN videos vid ON t.video_id = vid.video_id
            WHERE t.vector_id IN ({placeholders})
        """
        cur.execute(query, vector_ids)
        return {int(r["vector_id"]): dict(r) for r in cur.fetchall()}

    def get_frames_in_window(
        self, video_id: str, start: float, end: float
    ) -> List[Dict[str, Any]]:
        """Retrieve all visual frame entries within a timestamp window [start, end]."""
        cur = self.connection.cursor()
        cur.execute("""
            SELECT vector_id, frame_id, video_id, frame_index, timestamp_seconds,
                   timestamp_formatted, frame_path, frame_relpath, width, height,
                   embedding_model
            FROM visual_vectors
            WHERE video_id = ? AND timestamp_seconds >= ? AND timestamp_seconds <= ?
            ORDER BY timestamp_seconds ASC
        """, (video_id, start, end))
        return [dict(r) for r in cur.fetchall()]

    def get_segments_in_window(
        self, video_id: str, start: float, end: float
    ) -> List[Dict[str, Any]]:
        """Retrieve all transcript segments overlapping a timestamp window [start, end]."""
        cur = self.connection.cursor()
        cur.execute("""
            SELECT vector_id, segment_uid, video_id, segment_id, transcript_text,
                   start_timestamp, end_timestamp, start_formatted, end_formatted,
                   language, embedding_model
            FROM text_vectors
            WHERE video_id = ? AND NOT (end_timestamp < ? OR start_timestamp > ?)
            ORDER BY start_timestamp ASC
        """, (video_id, start, end))
        return [dict(r) for r in cur.fetchall()]

    def get_all_embeddings(
        self, modality: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch all vector IDs and unpacked float32 embeddings for FAISS index rebuild.

        Returns:
            Tuple of (vector_ids: int64 ndarray, embeddings: float32 2D ndarray)
        """
        table = "visual_vectors" if modality == "visual" else "text_vectors"
        cur = self.connection.cursor()
        cur.execute(f"SELECT vector_id, embedding_dim, embedding FROM {table} ORDER BY vector_id ASC")
        rows = cur.fetchall()

        if not rows:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)

        vector_ids: List[int] = []
        vectors: List[np.ndarray] = []

        dim = int(rows[0]["embedding_dim"])
        for r in rows:
            vector_ids.append(int(r["vector_id"]))
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            vectors.append(vec)

        id_arr = np.array(vector_ids, dtype=np.int64)
        mat_arr = np.vstack(vectors).astype(np.float32)
        return id_arr, mat_arr

    def stats(self) -> Dict[str, Any]:
        """Compute summary metrics of database contents."""
        cur = self.connection.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM videos")
        video_count = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM visual_vectors")
        visual_count = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM text_vectors")
        text_count = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM index_failures")
        failure_count = cur.fetchone()["c"]

        return {
            "videos": video_count,
            "visual_vectors": visual_count,
            "text_vectors": text_count,
            "failures": failure_count,
            "visual_model": self.get_meta("visual_model"),
            "visual_dim": self.get_meta("visual_dim"),
            "text_model": self.get_meta("text_model"),
            "text_dim": self.get_meta("text_dim"),
        }
