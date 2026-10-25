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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..db.app_db import AppDatabase, get_db
from ..settings import settings

logger = logging.getLogger("app.video")

_SAFE_FILENAME_REGEX = re.compile(r"[^a-zA-Z0-9._ -]")


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
) -> Tuple[bool, str]:
    """Remove user ownership of video. Purge physical files only when remaining refcount is 0."""
    database = db or get_db()
    target = database.execute_read_one(
        "SELECT * FROM videos WHERE id = ? AND user_id = ?",
        (video_db_id, user_id),
    )
    if not target:
        return False, "Video record not found or access denied."

    vid_id = target.get("video_id")
    stored_path = target.get("stored_path")
    playback_path = target.get("playback_path")

    # Remove ownership row
    database.execute_write("DELETE FROM videos WHERE id = ?", (video_db_id,))

    # Check remaining owners for this video_id
    if vid_id:
        remaining = database.execute_read_one(
            "SELECT COUNT(*) AS count FROM videos WHERE video_id = ?",
            (vid_id,),
        )
        refcount = int(remaining["count"]) if remaining else 0
        if refcount == 0:
            logger.info("Refcount for video_id %s dropped to 0. Purging derived and stored files...", vid_id)
            # Delete stored file and parent dir
            if stored_path:
                sp = Path(stored_path)
                sp.unlink(missing_ok=True)
                if sp.parent.is_dir() and not any(sp.parent.iterdir()):
                    shutil.rmtree(sp.parent, ignore_errors=True)

            # Delete playback proxy
            if playback_path:
                Path(playback_path).unlink(missing_ok=True)
            elif (settings.playback_dir / f"{vid_id}.mp4").is_file():
                (settings.playback_dir / f"{vid_id}.mp4").unlink(missing_ok=True)

            # Delete module1 outputs
            m1_out = settings.project_root / "module1" / "output"
            (m1_out / "metadata" / f"{vid_id}.json").unlink(missing_ok=True)
            (m1_out / "transcripts" / f"{vid_id}.json").unlink(missing_ok=True)
            (m1_out / "audio" / f"{vid_id}.wav").unlink(missing_ok=True)
            frames_dir = m1_out / "frames" / vid_id
            if frames_dir.is_dir():
                shutil.rmtree(frames_dir, ignore_errors=True)

    return True, "Video removed from your library."


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
