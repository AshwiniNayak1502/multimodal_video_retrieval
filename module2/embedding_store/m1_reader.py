"""Module 1 Output Reader and Path Re-anchoring Engine.

Parses Module 1 master metadata, frame manifests, and transcripts. Validates
timestamps, verifies frame images, computes content hashes, and dynamically
re-anchors absolute paths to the local filesystem without modifying Module 1 outputs.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import Module2Config
from .logging_config import get_logger
from .schema import IndexFailure, TextItem, VideoRecord, VisualItem

logger = get_logger("m1_reader")


def format_timestamp(seconds: float) -> str:
    """Format floating point seconds into 'HH:MM:SS' string using floor.

    Examples:
        0.0 -> "00:00:00"
        2.856 -> "00:00:02"
        123.4 -> "00:02:03"
    """
    if not math.isfinite(seconds) or seconds < 0:
        return "00:00:00"
    total_seconds = int(math.floor(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def compute_file_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file's raw content bytes."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_sha256(text: str, start: float, end: float) -> str:
    """Compute deterministic SHA-256 hash for a transcript segment."""
    normalized_text = text.strip()
    raw = f"{normalized_text}|{start:.3f}|{end:.3f}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def resolve_frame_path(
    record: Dict[str, Any],
    video_id: str,
    config: Module2Config,
) -> Tuple[str, Optional[str]]:
    """Resolve frame path by trying the stored absolute path then re-anchoring.

    Returns:
        Tuple of (resolved_absolute_path_str, relative_path_str_or_none)
    """
    raw_path_str = record.get("file_path", "")
    p = Path(raw_path_str) if raw_path_str else None

    resolved_path: Optional[Path] = None
    filename = p.name if p else f"frame_{record.get('frame_index', 0):04d}.jpg"

    # 1. Try recorded absolute path
    if p and p.is_file():
        resolved_path = p.resolve()
    else:
        # 2. Re-anchor to module1_output_dir/frames/<video_id>/<filename>
        reanchored = config.module1_output_dir / "frames" / video_id / filename
        if reanchored.is_file():
            resolved_path = reanchored.resolve()
        else:
            # Fallback to search inside frames/<video_id> directory
            frames_dir = config.module1_output_dir / "frames" / video_id
            if frames_dir.is_dir():
                matching = list(frames_dir.glob(f"*{record.get('frame_index', 0):04d}*.jpg"))
                if matching:
                    resolved_path = matching[0].resolve()

    if resolved_path is None:
        # Keep recorded path or predicted reanchored path
        final_abs = str(p.resolve()) if p else str(config.module1_output_dir / "frames" / video_id / filename)
    else:
        final_abs = str(resolved_path)

    # Compute relative path relative to module1_output_dir
    relpath: Optional[str] = None
    try:
        final_path_obj = Path(final_abs)
        relpath = final_path_obj.relative_to(config.module1_output_dir).as_posix()
    except Exception:
        relpath = f"frames/{video_id}/{Path(final_abs).name}"

    return final_abs, relpath


def resolve_transcript_path(
    stored_path: Optional[str],
    video_id: str,
    config: Module2Config,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve transcript JSON path by checking stored path then re-anchoring."""
    if stored_path:
        p = Path(stored_path)
        if p.is_file():
            resolved = p.resolve()
            try:
                rel = resolved.relative_to(config.module1_output_dir).as_posix()
            except Exception:
                rel = f"transcripts/{video_id}.json"
            return str(resolved), rel

    reanchored = config.module1_output_dir / "transcripts" / f"{video_id}.json"
    if reanchored.is_file():
        resolved = reanchored.resolve()
        try:
            rel = resolved.relative_to(config.module1_output_dir).as_posix()
        except Exception:
            rel = f"transcripts/{video_id}.json"
        return str(resolved), rel

    return None, None


def resolve_video_path(
    master: Dict[str, Any],
    config: Module2Config,
) -> Tuple[str, Optional[str]]:
    """Resolve the source video path by checking recorded path then searching input dir.

    Does NOT import module1; matches by file_name. If multiple matches occur,
    logs a warning and picks the first sorted.
    """
    v_meta = master.get("video_metadata") or master.get("video_properties") or {}
    source_file = master.get("source_file") or {}

    recorded_path_str = (
        v_meta.get("file_path")
        or master.get("source_video_path")
        or source_file.get("file_path")
        or ""
    )
    filename = (
        v_meta.get("file_name")
        or master.get("file_name")
        or source_file.get("file_name")
        or (Path(recorded_path_str).name if recorded_path_str else "")
    )

    p = Path(recorded_path_str) if recorded_path_str else None
    if p and p.is_file():
        resolved = p.resolve()
        relpath = None
        try:
            relpath = resolved.relative_to(config.module1_input_dir).as_posix()
        except Exception:
            pass
        return str(resolved), relpath

    # Search module1_input_dir recursively for filename
    if filename and config.module1_input_dir.is_dir():
        candidates = sorted(list(config.module1_input_dir.rglob(filename)))
        if candidates:
            if len(candidates) > 1:
                logger.warning(
                    "Multiple matching video files found for %s: %s. Using first: %s",
                    filename,
                    [str(c) for c in candidates],
                    candidates[0],
                )
            resolved = candidates[0].resolve()
            relpath = None
            try:
                relpath = resolved.relative_to(config.module1_input_dir).as_posix()
            except Exception:
                pass
            return str(resolved), relpath

    logger.warning(
        "Could not find video file '%s' on disk. Keeping originally recorded path: %s",
        filename,
        recorded_path_str,
    )
    return recorded_path_str, None


def read_master_metadata(
    json_path: Path,
    config: Module2Config,
) -> Optional[Tuple[VideoRecord, List[VisualItem], List[TextItem], List[IndexFailure]]]:
    """Parse and validate a single Module 1 master metadata JSON.

    Returns:
        Tuple of (VideoRecord, List[VisualItem], List[TextItem], List[IndexFailure])
        or None if the video is skipped / invalid.
    """
    logger.debug("Reading master metadata JSON: %s", json_path.name)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to parse master metadata JSON %s: %s", json_path, exc)
        return None

    video_id = data.get("video_id")
    if not video_id:
        logger.warning("Missing 'video_id' in master metadata: %s", json_path)
        return None

    # Status validation: "completed", "completed_no_audio", or dict {"status": "completed"}
    status_field = data.get("processing_status")
    if isinstance(status_field, dict):
        status = status_field.get("status")
    else:
        status = status_field

    if status not in ("completed", "completed_no_audio"):
        logger.info(
            "Skipping video %s: processing_status is '%s' (not completed).",
            video_id,
            status,
        )
        return None

    # Video properties
    v_meta = data.get("video_metadata") or data.get("video_properties") or {}
    source_file = data.get("source_file") or {}

    duration = float(v_meta.get("duration_seconds", 0.0) or data.get("pipeline_duration_seconds", 0.0))
    fps = float(v_meta.get("fps", 0.0))
    width = int(v_meta.get("width", 0))
    height = int(v_meta.get("height", 0))

    # Frame interval
    frame_extract = data.get("frame_extraction") or data.get("frames") or {}
    interval = float(frame_extract.get("frame_interval_seconds", 2.0))

    # Audio & Transcription info
    audio_info = data.get("audio_extraction") or data.get("audio") or {}
    trans_info = data.get("transcription") or data.get("transcript") or {}

    has_audio = bool(
        audio_info.get("has_audio")
        or v_meta.get("has_audio")
        or trans_info.get("has_audio")
    )
    language = trans_info.get("language")

    stored_transcript_path = (
        trans_info.get("transcript_path")
        or trans_info.get("transcript_file_path")
    )
    resolved_transcript_path, transcript_relpath = resolve_transcript_path(
        stored_transcript_path, video_id, config
    )

    resolved_video_path, video_relpath = resolve_video_path(data, config)
    filename = (
        v_meta.get("file_name")
        or data.get("file_name")
        or source_file.get("file_name")
        or Path(resolved_video_path).name
    )

    processed_at = data.get("processed_at") or datetime.datetime.now(datetime.timezone.utc).isoformat()

    failures: List[IndexFailure] = []

    # ------------------ Frames Extraction ------------------
    raw_frames: List[Dict[str, Any]] = (
        frame_extract.get("frames")
        or frame_extract.get("frame_records")
        or []
    )

    manifest_path = config.module1_output_dir / "frames" / video_id / "frames_manifest.json"
    manifest_frames: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)
                if isinstance(manifest_data, list):
                    manifest_frames = manifest_data
                elif isinstance(manifest_data, dict):
                    manifest_frames = manifest_data.get("frames") or manifest_data.get("frame_records") or []
        except Exception as mf_exc:
            logger.warning("Could not read frames_manifest.json for %s: %s", video_id, mf_exc)

    if raw_frames and manifest_frames and len(raw_frames) != len(manifest_frames):
        logger.warning(
            "Frame count mismatch for video %s: master metadata has %d, frames_manifest has %d",
            video_id,
            len(raw_frames),
            len(manifest_frames),
        )

    frames_to_use = raw_frames if raw_frames else manifest_frames
    if not frames_to_use:
        logger.warning("Video %s has no frame records; skipping.", video_id)
        return None

    visual_items: List[VisualItem] = []
    for f_rec in frames_to_use:
        f_idx = int(f_rec.get("frame_index", len(visual_items)))
        frame_id = f"{video_id}_frame_{f_idx:05d}"
        source_fnum = f_rec.get("source_frame_number")

        ts_sec = f_rec.get("timestamp_seconds")
        if ts_sec is None or not math.isfinite(ts_sec) or ts_sec < 0:
            msg = f"Invalid timestamp_seconds={ts_sec} on frame {f_idx}"
            logger.warning("Skipping frame %s: %s", frame_id, msg)
            failures.append(
                IndexFailure(
                    video_id=video_id,
                    modality="visual",
                    item_key=frame_id,
                    reason=msg,
                    failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
            )
            continue

        resolved_fpath, f_relpath = resolve_frame_path(f_rec, video_id, config)
        f_obj = Path(resolved_fpath)
        if not f_obj.is_file():
            msg = f"Frame image file does not exist on disk: {resolved_fpath}"
            logger.warning("Skipping frame %s: %s", frame_id, msg)
            failures.append(
                IndexFailure(
                    video_id=video_id,
                    modality="visual",
                    item_key=frame_id,
                    reason=msg,
                    failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
            )
            continue

        try:
            content_hash = compute_file_sha256(f_obj)
        except Exception as hash_exc:
            msg = f"Failed to compute frame sha256 for {resolved_fpath}: {hash_exc}"
            logger.warning("Skipping frame %s: %s", frame_id, msg)
            failures.append(
                IndexFailure(
                    video_id=video_id,
                    modality="visual",
                    item_key=frame_id,
                    reason=msg,
                    failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
            )
            continue

        f_w = int(f_rec.get("width") or width)
        f_h = int(f_rec.get("height") or height)
        ts_formatted = format_timestamp(float(ts_sec))

        visual_items.append(
            VisualItem(
                frame_id=frame_id,
                video_id=video_id,
                frame_index=f_idx,
                source_frame_number=source_fnum,
                timestamp_seconds=float(ts_sec),
                timestamp_formatted=ts_formatted,
                frame_path=resolved_fpath,
                frame_relpath=f_relpath,
                width=f_w,
                height=f_h,
                content_hash=content_hash,
            )
        )

    # ------------------ Transcript Extraction ------------------
    text_items: List[TextItem] = []
    if has_audio and resolved_transcript_path:
        t_path = Path(resolved_transcript_path)
        if t_path.is_file():
            try:
                with open(t_path, "r", encoding="utf-8") as tf:
                    t_data = json.load(tf)
                raw_segments = t_data.get("segments", [])
                for seg in raw_segments:
                    seg_id = int(seg.get("segment_id", len(text_items)))
                    seg_uid = f"{video_id}_seg_{seg_id:05d}"
                    raw_text = seg.get("text", "")
                    clean_text = raw_text.strip()
                    if not clean_text:
                        logger.debug("Skipping empty text segment %s", seg_uid)
                        continue

                    start_t = seg.get("start_time")
                    end_t = seg.get("end_time")

                    if (
                        start_t is None
                        or end_t is None
                        or not math.isfinite(start_t)
                        or not math.isfinite(end_t)
                        or start_t < 0
                        or end_t < start_t
                    ):
                        msg = f"Invalid segment timing: start={start_t}, end={end_t}"
                        logger.warning("Skipping text segment %s: %s", seg_uid, msg)
                        failures.append(
                            IndexFailure(
                                video_id=video_id,
                                modality="text",
                                item_key=seg_uid,
                                reason=msg,
                                failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            )
                        )
                        continue

                    start_f = float(start_t)
                    end_f = float(end_t)
                    seg_hash = compute_text_sha256(clean_text, start_f, end_f)

                    text_items.append(
                        TextItem(
                            segment_uid=seg_uid,
                            video_id=video_id,
                            segment_id=seg_id,
                            transcript_text=clean_text,
                            start_timestamp=start_f,
                            end_timestamp=end_f,
                            start_formatted=format_timestamp(start_f),
                            end_formatted=format_timestamp(end_f),
                            language=t_data.get("language") or language,
                            content_hash=seg_hash,
                        )
                    )
            except Exception as t_exc:
                logger.warning(
                    "Error reading transcript file %s for video %s: %s",
                    t_path,
                    video_id,
                    t_exc,
                )

    video_record = VideoRecord(
        video_id=video_id,
        video_filename=filename,
        video_path=resolved_video_path,
        video_relpath=video_relpath,
        duration_seconds=duration,
        fps=fps,
        width=width,
        height=height,
        frame_interval_seconds=interval,
        has_audio=has_audio,
        language=language,
        transcript_path=resolved_transcript_path,
        m1_metadata_path=str(json_path.resolve()),
        m1_processed_at=processed_at,
        indexed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    return video_record, visual_items, text_items, failures


def iter_videos(
    config: Module2Config,
) -> Iterator[Tuple[VideoRecord, List[VisualItem], List[TextItem], List[IndexFailure]]]:
    """Discover and yield parsed records for every completed video in Module 1 metadata dir."""
    metadata_dir = config.module1_output_dir / "metadata"
    if not metadata_dir.is_dir():
        logger.warning("Module 1 metadata directory not found: %s", metadata_dir)
        return

    json_files = sorted(list(metadata_dir.glob("*.json")))
    logger.info("Found %d master metadata JSON file(s) in %s", len(json_files), metadata_dir)

    for jf in json_files:
        parsed = read_master_metadata(jf, config)
        if parsed is not None:
            yield parsed
