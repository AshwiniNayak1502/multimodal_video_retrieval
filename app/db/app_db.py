"""Application database access layer managing users, sessions, and video ownership."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..settings import settings

logger = logging.getLogger("app.db")


class AppDatabase:
    """Thread-safe SQLite manager for application state with ACID transactions."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path or settings.app_db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new connection configured with WAL and foreign key support."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        """Execute schema.sql to initialize tables and indices."""
        schema_path = Path(__file__).resolve().parent / "schema.sql"
        if not schema_path.is_file():
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.transaction() as conn:
            conn.executescript(schema_sql)
        logger.info("Initialized application database at %s", self.db_path)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Provide a transactional scope around a series of operations."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute_read(
        self, query: str, params: Tuple[Any, ...] = ()
    ) -> List[Dict[str, Any]]:
        """Execute a parameterized SELECT query and return rows as dictionaries."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def execute_read_one(
        self, query: str, params: Tuple[Any, ...] = ()
    ) -> Optional[Dict[str, Any]]:
        """Execute a parameterized query returning at most one row."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def execute_write(
        self, query: str, params: Tuple[Any, ...] = ()
    ) -> int:
        """Execute a single parameterized write query and return lastrowid."""
        with self.transaction() as conn:
            cursor = conn.execute(query, params)
            return cursor.lastrowid or cursor.rowcount

    def execute_many(
        self, query: str, params_list: List[Tuple[Any, ...]]
    ) -> int:
        """Execute parameterized write queries for multiple parameter tuples."""
        with self.transaction() as conn:
            cursor = conn.executemany(query, params_list)
            return cursor.rowcount


# Singleton instance
_db_instance: Optional[AppDatabase] = None


def get_db(db_path: Optional[Path | str] = None) -> AppDatabase:
    """Return a singleton AppDatabase instance or a custom path instance."""
    global _db_instance
    if db_path is not None:
        return AppDatabase(db_path=db_path)
    env_path = os.getenv("VRS_APP_DB_PATH")
    if env_path:
        return AppDatabase(db_path=env_path)
    if _db_instance is None:
        _db_instance = AppDatabase()
    return _db_instance
