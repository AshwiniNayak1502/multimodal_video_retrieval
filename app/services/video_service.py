"""Video management service handling uploads, validation, storage isolation, and ownership."""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import logging
import os
import re
import secrets
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..db.app_db import AppDatabase, get_db
from ..settings import settings

logger = logging.getLogger("app.video")

_SAFE_FILENAME_REGEX = re.compile(r"[^a-zA-Z0-9._ -]")
_SAFE_VIDEO_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")


def validate_video_id(video_id: str) -> bool:
    """Validate video_id contains only alphanumeric characters, underscores, and hyphens."""
    if not video_id or not isinstance(video_id, str):
        return False
    return bool(_SAFE_VIDEO_ID_REGEX.match(video_id))


def is_path_safe(path: Path, allowed_roots: List[Path]) -> bool:
    """Ensure path is resolved and strictly resides inside one of allowed_roots."""
    try:
        resolved = path.resolve()
        for root in allowed_roots:
            resolved_root = root.resolve()
            if resolved == resolved_root or resolved.is_relative_to(resolved_root):
                return True
        return False
    except Exception:
        return False


def safe_unlink_file(file_path: Path, retries: int = 3, delay: float = 0.1) -> None:
    """Unlink file with retries for transient locks / Windows PermissionError."""
    if not file_path.is_file():
        return
    for attempt in range(retries):
        try:
            file_path.unlink(missing_ok=True)
            return
        except PermissionError as pe:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.error("PermissionError unlinking %s after %d attempts: %s", file_path, retries, pe)
                raise
        except Exception as exc:
            logger.error("Error unlinking %s: %s", file_path, exc)
            raise


def safe_remove_tree(dir_path: Path, retries: int = 3, delay: float = 0.1) -> None:
    """Remove directory tree with retries for transient locks / Windows PermissionError."""
    if not dir_path.is_dir():
        return
    for attempt in range(retries):
        try:
            shutil.rmtree(dir_path, ignore_errors=False)
            return
        except PermissionError as pe:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.error("PermissionError removing directory %s after %d attempts: %s", dir_path, retries, pe)
                raise
        except Exception as exc:
            logger.error("Error removing directory %s: %s", dir_path, exc)
            raise



def get_storage_root() -> Path:
    """Return the base storage root under module1/input/videos."""
    root = (settings.project_root / "module1" / "input" / "videos").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_filename(filename: str) -> str:
    """Sanitize original filename to prevent path traversal and shell issues."""
    name = Path(filename).name
    cleaned = _SAFE_FILENAME_REGEX.sub("_", name)
    cleaned = cleaned.strip(" ._")
    return cleaned or "video.mp4"


def compute_video_id(video_path: Path | str) -> str:
    """Compute deterministic video_id using pure hashlib logic from module1.

    Loads module1/src/id_generation.py dynamically without importing WhisperX.
    """
    path = Path(video_path).resolve()
    id_gen_path = settings.project_root / "module1" / "src" / "id_generation.py"
    if not id_gen_path.is_file():
        id_gen_path = Path(__file__).resolve().parents[2] / "module1" / "src" / "id_generation.py"
    if not id_gen_path.is_file():
        raise FileNotFoundError(f"id_generation.py not found at {id_gen_path}")

    spec = importlib.util.spec_from_file_location("module1_id_generation", str(id_gen_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {id_gen_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_video_id(path)


def validate_upload_extension(filename: str) -> Tuple[bool, Optional[str]]:
    """Validate that filename has a supported container extension (.mp4, .mkv)."""
    p = Path(filename)
    ext = p.suffix.lower()
    if ext not in settings.supported_extensions:
        return False, f"Unsupported file type '{ext}'. Allowed formats: {', '.join(settings.supported_extensions)}."
    return True, None


def save_uploaded_stream(
    user_id: int,
    filename: str,
    stream: Any,
    expected_size: Optional[int] = None,
    db: Optional[AppDatabase] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Stream uploaded file to disk with chunked SHA-256 hashing and user isolation.

    Storage directory: module1/input/videos/u<user_id>/<random-hex>/<sanitized_name>
    """
    database = db or get_db()
    clean_name = sanitize_filename(filename)

    valid_ext, ext_err = validate_upload_extension(clean_name)
    if not valid_ext:
        return False, ext_err or "Unsupported file format.", None

    storage_root = get_storage_root()
    random_hex = secrets.token_hex(8)
    dest_dir = storage_root / f"u{user_id}" / random_hex
    dest_path = dest_dir / clean_name

    # Path traversal validation: ensure resolved path is inside storage_root
    try:
        resolved = dest_path.resolve()
        if not resolved.is_relative_to(storage_root):
            return False, "Security violation: destination path traverses outside storage root.", None
    except Exception as exc:
        return False, f"Invalid destination path: {exc}", None

    dest_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    bytes_written = 0
    hasher = hashlib.sha256()

    try:
        with open(dest_path, "wb") as f:
            # Handle streamlit UploadedFile or standard file-like object
            if hasattr(stream, "read"):
                chunk_size = 64 * 1024  # 64 KB
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        dest_path.unlink(missing_ok=True)
                        dest_dir.rmdir()
                        return (
                            False,
                            f"File size exceeds maximum allowable limit of {settings.max_upload_mb} MB.",
                            None,
                        )
                    hasher.update(chunk)
                    f.write(chunk)
            elif isinstance(stream, bytes):
                bytes_written = len(stream)
                if bytes_written > max_bytes:
                    return False, f"File size exceeds {settings.max_upload_mb} MB.", None
                hasher.update(stream)
                f.write(stream)
            else:
                return False, "Invalid stream reader provided.", None

        if bytes_written == 0:
            dest_path.unlink(missing_ok=True)
            dest_dir.rmdir()
            return False, "Uploaded file is empty (0 bytes).", None

    except Exception as write_exc:
        dest_path.unlink(missing_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
        return False, f"Failed writing file to disk: {write_exc}", None

    content_hash = hasher.hexdigest()

    # Check for same user duplicate
    existing_user_vid = database.execute_read_one(
        "SELECT * FROM videos WHERE user_id = ? AND content_sha256 = ?",
        (user_id, content_hash),
    )
    if existing_user_vid:
        dest_path.unlink(missing_ok=True)
        shutil.rmtree(dest_dir, ignore_errors=True)
        return (
            False,
            f"You have already uploaded this exact video ('{existing_user_vid['original_filename']}').",
            None,
        )

    # Compute deterministic video ID
    try:
        vid_id = compute_video_id(dest_path)
    except Exception as id_exc:
        dest_path.unlink(missing_ok=True)
        shutil.rmtree(dest_dir, ignore_errors=True)
        return False, f"Failed generating deterministic video identifier: {id_exc}", None

    # Check if another user already ingested identical content (same video_id)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    already_indexed = database.execute_read_one(
        "SELECT * FROM videos WHERE video_id = ? AND status = 'completed' LIMIT 1",
        (vid_id,),
    )
    if already_indexed:
        logger.info(
            "Video content %s already indexed by another user. Reusing existing vector index for user %d.",
            vid_id,
            user_id,
        )
        row_id = database.execute_write(
            """
            INSERT INTO videos (
                user_id, video_id, original_filename, stored_path, playback_path,
                size_bytes, content_sha256, status, stage, duration_seconds,
                has_audio, created_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', 'completed', ?, ?, ?, ?)
            """,
            (
                user_id,
                vid_id,
                clean_name,
                str(dest_path.resolve()),
                already_indexed.get("playback_path"),
                bytes_written,
                content_hash,
                already_indexed.get("duration_seconds"),
                already_indexed.get("has_audio", 0),
                now_iso,
                now_iso,
            ),
        )
        saved_row = database.execute_read_one("SELECT * FROM videos WHERE id = ?", (row_id,))
        return True, "Video recognized from existing indexed library; ready instantly!", saved_row

    # New video to ingest
    row_id = database.execute_write(
        """
        INSERT INTO videos (
            user_id, video_id, original_filename, stored_path, size_bytes,
            content_sha256, status, stage, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 'pending', 'queued', ?)
        """,
        (
            user_id,
            vid_id,
            clean_name,
            str(dest_path.resolve()),
            bytes_written,
            content_hash,
            now_iso,
        ),
    )
    saved_row = database.execute_read_one("SELECT * FROM videos WHERE id = ?", (row_id,))
    return True, "Video uploaded successfully. Ingestion queued.", saved_row


def get_thumbnail_path(video_id: str) -> Optional[str]:
    """Find the first extracted frame JPEG from module1/output/frames/<video_id>."""
    if not video_id:
        return None
    frames_dir = settings.project_root / "module1" / "output" / "frames" / video_id
    if not frames_dir.is_dir():
        return None
    # Prefer frame_0000 or lowest timestamp
    frames = sorted(list(frames_dir.glob("frame_*.jpg")))
    if frames:
        return str(frames[0].resolve())
    return None


def list_user_videos(
    user_id: int,
    db: Optional[AppDatabase] = None,
) -> List[Dict[str, Any]]:
    """Return all videos owned by user_id with real thumbnails and vector stats."""
    database = db or get_db()
    rows = database.execute_read(
        "SELECT * FROM videos WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    )
    result: List[Dict[str, Any]] = []
    for r in rows:
        vid_id = r.get("video_id")
        thumb = get_thumbnail_path(vid_id) if vid_id else None
        item = dict(r)
        item["thumbnail_path"] = thumb
        item["exists_on_disk"] = Path(r["stored_path"]).is_file() if r.get("stored_path") else False
        result.append(item)
    return result


def delete_user_video(
    user_id: int,
    video_db_id: int,
    db: Optional[AppDatabase] = None,
    on_index_changed: Optional[Callable[[], None]] = None,
) -> Tuple[bool, str]:
    """Remove user ownership of video, purging files and vector embeddings safely and idempotently.

    Order of operations:
    1. Verify ownership (id AND user_id). Refuse if 'processing' or 'indexing'.
    2. Set status='deleting' so video stops appearing in search.
    3. Compute refcount of OTHER rows sharing the same video_id.
    4. ALWAYS delete this row's own stored upload file and now-empty parent directory.
    5. Only if refcount is 0, purge shared artifacts: Module 2 vectors FIRST, then Module 1
       frames, audio, transcripts, metadata JSON, and playback proxy.
    6. Delete app-DB row LAST. If any critical step fails, row remains 'deleting' for retry.
    """
    database = db or get_db()
    target = database.execute_read_one(
        "SELECT * FROM videos WHERE id = ? AND user_id = ?",
        (video_db_id, user_id),
    )
    if not target:
        return False, "Video record not found or access denied."

    status = target.get("status")
    if status in ("processing", "indexing"):
        return False, f"Video is currently {status} and cannot be deleted. Please wait until processing completes."

    vid_id = target.get("video_id")
    if vid_id and not validate_video_id(vid_id):
        err = f"Security violation: invalid video_id format '{vid_id}'."
        logger.error(err)
        return False, err

    # Step 2: Mark deleting immediately
    if status != "deleting":
        database.execute_write(
            "UPDATE videos SET status = 'deleting' WHERE id = ?",
            (video_db_id,),
        )

    try:
        # Step 3: Compute refcount of other owners
        other_refcount = 0
        if vid_id:
            other_owners = database.execute_read_one(
                "SELECT COUNT(*) AS count FROM videos WHERE video_id = ? AND id != ?",
                (vid_id, video_db_id),
            )
            other_refcount = int(other_owners["count"]) if other_owners else 0

        # Step 4: Always delete own upload file and its directory
        stored_path = target.get("stored_path")
        storage_root = get_storage_root()
        if stored_path:
            sp = Path(stored_path)
            if not is_path_safe(sp, [storage_root]):
                err = f"Security violation: stored_path '{stored_path}' traverses outside storage root."
                logger.error(err)
                return False, err

            safe_unlink_file(sp)
            parent_dir = sp.parent
            if is_path_safe(parent_dir, [storage_root]) and parent_dir.is_dir():
                if not any(parent_dir.iterdir()):
                    safe_remove_tree(parent_dir)
                    user_dir = parent_dir.parent
                    if is_path_safe(user_dir, [storage_root]) and user_dir.is_dir() and not any(user_dir.iterdir()):
                        safe_remove_tree(user_dir)

        # Step 5: If no other owners remain, purge shared artifacts
        if other_refcount == 0 and vid_id:
            logger.info("Refcount for video_id %s dropped to 0. Purging derived and vector files...", vid_id)

            # 5a. Module 2 data FIRST (holding shared indexing lock)
            from .ingestion_runner import get_indexing_lock
            indexing_lock = get_indexing_lock()
            lock_acquired = indexing_lock.acquire(timeout=0.5)
            if not lock_acquired:
                return False, "Indexing is currently in progress. Please try deleting again shortly."
            try:
                m2_dir = (settings.project_root / "module2").resolve()
                if str(m2_dir) not in sys.path:
                    sys.path.insert(0, str(m2_dir))
                from embedding_store import purge_video_from_store
                from embedding_store.config import Module2Config
                m2_cfg = Module2Config(base_dir=m2_dir)
                purge_video_from_store(vid_id, config=m2_cfg)
            finally:
                indexing_lock.release()

            # Refresh in-memory search index
            if on_index_changed is not None:
                try:
                    on_index_changed()
                except Exception as cb_exc:
                    logger.warning("on_index_changed callback error: %s", cb_exc)
            else:
                try:
                    from .search_service import _search_engine_instance
                    if _search_engine_instance is not None:
                        _search_engine_instance.reload()
                except Exception as rel_exc:
                    logger.warning("Search engine reload error: %s", rel_exc)

            # 5b. Shared Module 1 files
            m1_out = (settings.project_root / "module1" / "output").resolve()
            for subdir, ext in [("metadata", ".json"), ("transcripts", ".json"), ("audio", ".wav")]:
                f_path = m1_out / subdir / f"{vid_id}{ext}"
                if is_path_safe(f_path, [m1_out]):
                    safe_unlink_file(f_path)

            frames_dir = m1_out / "frames" / vid_id
            if is_path_safe(frames_dir, [m1_out]):
                safe_remove_tree(frames_dir)

            # 5c. Playback proxy
            playback_path = target.get("playback_path")
            pb_root = settings.playback_dir.resolve()
            if playback_path:
                pb_file = Path(playback_path)
                if is_path_safe(pb_file, [pb_root, (settings.project_root / "data" / "playback").resolve()]):
                    safe_unlink_file(pb_file)
            def_pb = pb_root / f"{vid_id}.mp4"
            if is_path_safe(def_pb, [pb_root]):
                safe_unlink_file(def_pb)
        elif vid_id:
            logger.info(
                "Refcount for video_id %s is %d. Preserving shared derived files and vectors for other owners.",
                vid_id,
                other_refcount,
            )

        # Step 6: Delete app DB row LAST
        database.execute_write("DELETE FROM videos WHERE id = ?", (video_db_id,))
        logger.info("Successfully deleted video DB ID %d (video_id=%s) for user %d.", video_db_id, vid_id, user_id)
        return True, "Video removed from your library."

    except Exception as exc:
        err_msg = f"Failed to delete video: {exc}"
        logger.error("Error deleting video DB ID %d for user %d: %s", video_db_id, user_id, exc, exc_info=True)
        database.execute_write(
            "UPDATE videos SET status = 'deleting', error_message = ? WHERE id = ?",
            (str(exc), video_db_id),
        )
        return False, f"Failed to delete video: {exc}. Please retry."


def reconcile_interrupted_deletions(db: Optional[AppDatabase] = None) -> int:
    """Attempt to complete deletion for any rows left in 'deleting' state from an interrupted session."""
    database = db or get_db()
    rows = database.execute_read("SELECT id, user_id FROM videos WHERE status = 'deleting'")
    cleaned = 0
    for r in rows:
        ok, msg = delete_user_video(user_id=r["user_id"], video_db_id=r["id"], db=database)
        if ok:
            cleaned += 1
        else:
            logger.warning("Could not reconcile deleting video id %d on startup: %s", r["id"], msg)
    if cleaned > 0:
        logger.info("Reconciled %d interrupted video deletion(s) from prior session.", cleaned)
    return cleaned



def retry_failed_video(
    user_id: int,
    video_db_id: int,
    db: Optional[AppDatabase] = None,
) -> Tuple[bool, str]:
    """Reset a failed or interrupted video back to pending state for re-ingestion."""
    database = db or get_db()
    target = database.execute_read_one(
        "SELECT * FROM videos WHERE id = ? AND user_id = ? AND status = 'failed'",
        (video_db_id, user_id),
    )
    if not target:
        return False, "Only failed videos can be retried."

    database.execute_write(
        """
        UPDATE videos
        SET status = 'pending', stage = 'queued', error_message = NULL
        WHERE id = ?
        """,
        (video_db_id,),
    )
    return True, "Video queued for retry."
