"""Background single-worker ingestion queue orchestrating Module 1 and Module 2 subprocesses."""

from __future__ import annotations

import datetime
import json
import logging
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from ..db.app_db import AppDatabase, get_db
from ..settings import settings
from .playback_service import ensure_playback_proxy

logger = logging.getLogger("app.ingestion")


def recover_interrupted_jobs(db: Optional[AppDatabase] = None) -> int:
    """Mark any jobs left in 'processing' or 'indexing' state as failed with retry enabled."""
    database = db or get_db()
    rows = database.execute_write(
        """
        UPDATE videos
        SET status = 'failed',
            stage = 'interrupted',
            error_message = 'Processing was interrupted by an application restart or crash. Click Retry to re-process.'
        WHERE status IN ('processing', 'indexing')
        """
    )
    if rows > 0:
        logger.info("Recovered %d interrupted video ingestion jobs from prior session.", rows)
    return rows


class IngestionRunner:
    """Threaded single-worker pipeline runner processing one video at a time to bound RAM."""

    def __init__(
        self,
        db: Optional[AppDatabase] = None,
        on_index_updated: Optional[Callable[[], None]] = None,
    ) -> None:
        self.db = db or get_db()
        self.on_index_updated = on_index_updated
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background worker thread if not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="IngestionWorker",
                daemon=True,
            )
            self._thread.start()
            logger.info("Background video ingestion worker started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal worker to stop and wait for completion."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            logger.info("Background video ingestion worker stopped.")

    def active_job_count(self) -> int:
        """Count number of jobs currently pending, processing, or indexing."""
        row = self.db.execute_read_one(
            "SELECT COUNT(*) as cnt FROM videos WHERE status IN ('pending', 'processing', 'indexing')"
        )
        return int(row["cnt"]) if row and "cnt" in row else 0

    def _worker_loop(self) -> None:
        """Continuous polling loop picking up pending video jobs."""
        while not self._stop_event.is_set():
            try:
                job = self.db.execute_read_one(
                    """
                    SELECT * FROM videos
                    WHERE status = 'pending'
                    ORDER BY id ASC
                    LIMIT 1
                    """
                )
                if job:
                    self._process_job(job)
                else:
                    # Sleep briefly when no jobs are pending
                    time.sleep(1.0)
            except Exception as loop_exc:
                logger.error("Error in ingestion runner loop: %s", loop_exc, exc_info=True)
                time.sleep(2.0)

    def _process_job(self, job: dict) -> None:
        """Execute Stage 1 (Module 1 Ingestion) followed by Stage 2 (Module 2 Indexing)."""
        video_db_id = job["id"]
        user_id = job["user_id"]
        video_id = job.get("video_id")
        stored_path = job.get("stored_path")
        orig_filename = job.get("original_filename")

        logger.info(
            "Starting ingestion for video DB ID %d: %s (%s)...",
            video_db_id,
            orig_filename,
            video_id,
        )

        # Mark as processing
        self.db.execute_write(
            """
            UPDATE videos
            SET status = 'processing', stage = 'stage1_extracting_and_transcribing'
            WHERE id = ?
            """,
            (video_db_id,),
        )

        v_path = Path(stored_path).resolve()
        if not v_path.is_file():
            err = f"Source video file not found on disk: {stored_path}"
            logger.error(err)
            self.db.execute_write(
                """
                UPDATE videos
                SET status = 'failed', stage = 'validation_failed', error_message = ?
                WHERE id = ?
                """,
                (err, video_db_id),
            )
            return

        # -------------------------------------------------------------
        # STAGE 1: Module 1 subprocess (frames, audio, transcription)
        # -------------------------------------------------------------
        m1_dir = (settings.project_root / "module1").resolve()
        cmd_m1 = [sys.executable, "main.py", "--video", str(v_path)]
        if settings.whisper_model:
            cmd_m1.extend(["--whisper-model", settings.whisper_model])

        logger.info("Executing Module 1 subprocess: %s (cwd=%s)", " ".join(cmd_m1), m1_dir)
        try:
            p1 = subprocess.run(
                cmd_m1,
                cwd=str(m1_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=600,  # 10 minutes max for long video
            )
        except Exception as p1_exc:
            err = f"Module 1 subprocess failed to launch: {p1_exc}"
            logger.error(err)
            self.db.execute_write(
                "UPDATE videos SET status = 'failed', stage = 'module1_error', error_message = ? WHERE id = ?",
                (err, video_db_id),
            )
            return

        if p1.returncode != 0:
            err_tail = p1.stderr[-800:] if p1.stderr else p1.stdout[-800:] or "Unknown error"
            logger.error("Module 1 failed for %s (exit code %d): %s", orig_filename, p1.returncode, err_tail)
            self.db.execute_write(
                "UPDATE videos SET status = 'failed', stage = 'module1_failed', error_message = ? WHERE id = ?",
                (f"Frame extraction or transcription failed: {err_tail.strip()}", video_db_id),
            )
            return

        # Inspect Module 1 output metadata JSON to verify completion
        meta_json_path = settings.project_root / "module1" / "output" / "metadata" / f"{video_id}.json"
        if not meta_json_path.is_file():
            err = f"Module 1 completed but metadata JSON was not written to {meta_json_path}"
            logger.error(err)
            self.db.execute_write(
                "UPDATE videos SET status = 'failed', stage = 'module1_missing_output', error_message = ? WHERE id = ?",
                (err, video_db_id),
            )
            return

        try:
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
            m1_status = meta_data.get("processing_status")
            if m1_status not in ("completed", "completed_no_audio"):
                err = f"Module 1 reported non-completed status: {m1_status}"
                logger.error(err)
                self.db.execute_write(
                    "UPDATE videos SET status = 'failed', stage = 'module1_incomplete', error_message = ? WHERE id = ?",
                    (err, video_db_id),
                )
                return

            v_meta = meta_data.get("video_metadata", {})
            duration = float(v_meta.get("duration_seconds") or 0.0)
            has_audio = 1 if v_meta.get("has_audio") else 0

            # Update duration and audio metadata
            self.db.execute_write(
                "UPDATE videos SET duration_seconds = ?, has_audio = ? WHERE id = ?",
                (duration, has_audio, video_db_id),
            )
        except Exception as json_exc:
            logger.warning("Error reading Module 1 metadata JSON: %s", json_exc)
            has_audio = 0

        # Generate playback proxy in background
        playback_path_str = None
        try:
            proxy_path = ensure_playback_proxy(v_path, video_id)
            playback_path_str = str(proxy_path.resolve())
            self.db.execute_write(
                "UPDATE videos SET playback_path = ? WHERE id = ?",
                (playback_path_str, video_db_id),
            )
        except Exception as proxy_exc:
            logger.warning("Playback proxy creation warning for %s: %s", video_id, proxy_exc)

        # -------------------------------------------------------------
        # STAGE 2: Module 2 subprocess (embeddings & FAISS index)
        # -------------------------------------------------------------
        self.db.execute_write(
            """
            UPDATE videos
            SET status = 'indexing', stage = 'stage2_generating_embeddings_and_indexing'
            WHERE id = ?
            """,
            (video_db_id,),
        )

        m2_dir = (settings.project_root / "module2").resolve()
        cmd_m2 = [sys.executable, "main.py", "index", "--video-id", video_id]

        logger.info("Executing Module 2 subprocess: %s (cwd=%s)", " ".join(cmd_m2), m2_dir)
        try:
            p2 = subprocess.run(
                cmd_m2,
                cwd=str(m2_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=600,
            )
        except Exception as p2_exc:
            err = f"Module 2 subprocess failed to launch: {p2_exc}"
            logger.error(err)
            self.db.execute_write(
                "UPDATE videos SET status = 'failed', stage = 'module2_error', error_message = ? WHERE id = ?",
                (err, video_db_id),
            )
            return

        if p2.returncode != 0:
            err_tail = p2.stderr[-800:] if p2.stderr else p2.stdout[-800:] or "Unknown error"
            logger.error("Module 2 failed for %s (exit code %d): %s", video_id, p2.returncode, err_tail)
            self.db.execute_write(
                "UPDATE videos SET status = 'failed', stage = 'module2_failed', error_message = ? WHERE id = ?",
                (f"Embedding generation or vector indexing failed: {err_tail.strip()}", video_db_id),
            )
            return

        # Verification: Check that vectors actually exist in Module 2 SQLite metadata.db
        m2_db_path = settings.project_root / "module2" / "output" / "vector_db" / "metadata.db"
        if not m2_db_path.is_file():
            err = "Module 2 reported success but vector metadata.db was not found on disk."
            logger.error(err)
            self.db.execute_write(
                "UPDATE videos SET status = 'failed', stage = 'module2_missing_db', error_message = ? WHERE id = ?",
                (err, video_db_id),
            )
            return

        try:
            conn2 = sqlite3.connect(str(m2_db_path), timeout=10.0)
            cur2 = conn2.cursor()
            cur2.execute("SELECT COUNT(*) FROM visual_vectors WHERE video_id = ?", (video_id,))
            v_vec_count = cur2.fetchone()[0]
            cur2.execute("SELECT COUNT(*) FROM text_vectors WHERE video_id = ?", (video_id,))
            t_vec_count = cur2.fetchone()[0]
            conn2.close()

            if v_vec_count == 0 and t_vec_count == 0:
                err = f"Module 2 produced 0 visual and 0 text vectors for video_id {video_id}."
                logger.error(err)
                self.db.execute_write(
                    "UPDATE videos SET status = 'failed', stage = 'module2_zero_vectors', error_message = ? WHERE id = ?",
                    (err, video_db_id),
                )
                return
        except Exception as check_exc:
            logger.warning("Error verifying vector counts in metadata.db: %s", check_exc)

        # Mark completed!
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.db.execute_write(
            """
            UPDATE videos
            SET status = 'completed', stage = 'completed', error_message = NULL, completed_at = ?
            WHERE id = ?
            """,
            (now_iso, video_db_id),
        )
        logger.info(
            "Video %s (%s) fully completed and indexed successfully!",
            orig_filename,
            video_id,
        )

        # Trigger index reload callback so search service has fresh index
        if self.on_index_updated is not None:
            try:
                self.on_index_updated()
            except Exception as cb_exc:
                logger.warning("Error calling on_index_updated callback: %s", cb_exc)


# Singleton runner
_runner_instance: Optional[IngestionRunner] = None


def get_ingestion_runner(
    on_index_updated: Optional[Callable[[], None]] = None,
) -> IngestionRunner:
    """Return or initialize singleton IngestionRunner instance."""
    global _runner_instance
    if _runner_instance is None:
        recover_interrupted_jobs()
        _runner_instance = IngestionRunner(on_index_updated=on_index_updated)
        _runner_instance.start()
    elif on_index_updated is not None and _runner_instance.on_index_updated is None:
        _runner_instance.on_index_updated = on_index_updated
    return _runner_instance
