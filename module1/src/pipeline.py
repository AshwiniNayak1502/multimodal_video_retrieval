"""End-to-end processing pipeline orchestrator for Module 1."""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .audio_extraction import extract_audio
from .config import Module1Config
from .frame_extraction import extract_frames
from .id_generation import generate_video_id
from .metadata_extraction import extract_video_metadata
from .transcription import LocalTranscriber
from .video_discovery import discover_videos
from .video_validation import validate_video

logger = logging.getLogger("module1.pipeline")


@dataclass
class PipelineResult:
    """Outcome of processing an individual video through the pipeline."""

    video_id: str
    video_path: str
    status: str  # "completed", "completed_no_audio", "skipped", "failed"
    error_message: Optional[str] = None
    metadata_path: Optional[str] = None
    frames_count: int = 0
    has_audio: bool = False
    segment_count: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to JSON dictionary."""
        return asdict(self)


@dataclass
class PipelineBatchResult:
    """Outcome of running the pipeline on a collection of videos."""

    total_videos: int
    processed_count: int
    skipped_count: int
    failed_count: int
    results: List[PipelineResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert batch result to JSON dictionary."""
        return {
            "total_videos": self.total_videos,
            "processed_count": self.processed_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "results": [r.to_dict() for r in self.results],
        }


def is_already_processed(video_id: str, config: Module1Config) -> bool:
    """Check if all expected outputs for video_id exist on disk and are valid."""
    meta_path = config.metadata_dir / f"{video_id}.json"
    if not meta_path.exists():
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        status = data.get("processing_status")
        if status not in ("completed", "completed_no_audio"):
            return False

        # Verify frames directory has files
        frames_dir = config.frames_dir / video_id
        if not frames_dir.exists() or not any(frames_dir.glob("*.jpg")):
            return False

        # If audio track was detected, verify audio and transcript files exist
        has_audio = data.get("video_metadata", {}).get("has_audio", False)
        if has_audio:
            audio_path = config.audio_dir / f"{video_id}.wav"
            transcript_path = config.transcripts_dir / f"{video_id}.json"
            if not audio_path.exists() or not transcript_path.exists():
                return False

        return True
    except Exception as exc:
        logger.debug(
            "Error inspecting prior processing metadata for %s: %s",
            video_id,
            exc,
        )
        return False


def process_video(
    video_path: Union[str, Path],
    config: Optional[Module1Config] = None,
    force: bool = False,
    transcriber: Optional[LocalTranscriber] = None,
) -> PipelineResult:
    """Execute complete Module 1 processing on a single video file.

    Stages:
      1. Validation (integrity, container check)
      2. Deterministic ID generation
      3. Skip check (detect prior full processing unless force=True)
      4. Video metadata extraction
      5. Time-based frame sampling (OpenCV)
      6. Single-pass audio track extraction (FFmpeg)
      7. Speech transcription & word alignment (WhisperX, local)
      8. Persist final consolidated metadata JSON

    Args:
        video_path: Path to target video file (.mp4, .mkv).
        config: Module configuration (defaults to standard Module1Config).
        force: If True, re-process even if outputs already exist.
        transcriber: Optional pre-warmed LocalTranscriber instance.

    Returns:
        PipelineResult detailing the execution outcome.
    """
    cfg = config or Module1Config()
    cfg.ensure_directories()
    start_time = time.time()
    v_path = Path(video_path).resolve()

    logger.info("==================================================")
    logger.info("Starting processing for video: %s", v_path.name)

    # Stage 1: Validation
    validation = validate_video(
        v_path,
        ffprobe_path=cfg.ffprobe_path,
        supported_extensions=cfg.supported_extensions,
    )
    if not validation.is_valid:
        err = validation.error_message or "Validation failed"
        logger.error("Validation failed for %s: %s", v_path.name, err)
        return PipelineResult(
            video_id="unknown",
            video_path=str(v_path),
            status="failed",
            error_message=err,
            duration_seconds=round(time.time() - start_time, 3),
        )

    # Stage 2: ID Generation
    try:
        video_id = generate_video_id(v_path)
        logger.info("Generated deterministic video_id: %s", video_id)
    except Exception as id_exc:
        err = f"Failed to generate video_id for {v_path.name}: {id_exc}"
        logger.error(err, exc_info=True)
        return PipelineResult(
            video_id="unknown",
            video_path=str(v_path),
            status="failed",
            error_message=err,
            duration_seconds=round(time.time() - start_time, 3),
        )

    # Stage 3: Skip check
    if not force and is_already_processed(video_id, cfg):
        logger.info(
            "Video %s (%s) is already fully processed. Skipping (use --force to reprocess).",
            v_path.name,
            video_id,
        )
        meta_path = cfg.metadata_dir / f"{video_id}.json"
        return PipelineResult(
            video_id=video_id,
            video_path=str(v_path),
            status="skipped",
            metadata_path=str(meta_path),
            duration_seconds=round(time.time() - start_time, 3),
        )

    # Stage 4: Metadata extraction
    logger.info(
        "[Stage 1/4] Extracting video metadata for %s...", v_path.name
    )
    try:
        metadata = extract_video_metadata(
            video_path=v_path,
            video_id=video_id,
            ffprobe_path=cfg.ffprobe_path,
        )
    except Exception as meta_exc:
        err = f"Failed to extract metadata for {v_path.name}: {meta_exc}"
        logger.error(err, exc_info=True)
        return PipelineResult(
            video_id=video_id,
            video_path=str(v_path),
            status="failed",
            error_message=err,
            duration_seconds=round(time.time() - start_time, 3),
        )

    # Stage 5: Frame extraction
    logger.info(
        "[Stage 2/4] Extracting representative frames (interval=%.2fs)...",
        cfg.frame_interval_seconds,
    )
    try:
        frame_records = extract_frames(
            video_path=v_path,
            video_id=video_id,
            output_dir=cfg.frames_dir,
            interval_seconds=cfg.frame_interval_seconds,
        )
    except Exception as frame_exc:
        err = f"Failed to extract frames for {v_path.name}: {frame_exc}"
        logger.error(err, exc_info=True)
        return PipelineResult(
            video_id=video_id,
            video_path=str(v_path),
            status="failed",
            error_message=err,
            duration_seconds=round(time.time() - start_time, 3),
        )

    # Stage 6: Audio extraction
    logger.info("[Stage 3/4] Extracting audio track once via FFmpeg...")
    try:
        audio_result = extract_audio(
            video_path=v_path,
            video_id=video_id,
            output_audio_dir=cfg.audio_dir,
            ffmpeg_path=cfg.ffmpeg_path,
            ffprobe_path=cfg.ffprobe_path,
            sample_rate=cfg.audio_sample_rate,
            channels=cfg.audio_channels,
            codec=cfg.audio_codec,
            known_has_audio=metadata.has_audio,
        )
    except Exception as audio_exc:
        err = f"Audio extraction encountered exception for {v_path.name}: {audio_exc}"
        logger.error(err, exc_info=True)
        return PipelineResult(
            video_id=video_id,
            video_path=str(v_path),
            status="failed",
            error_message=err,
            duration_seconds=round(time.time() - start_time, 3),
        )

    # Stage 7: Transcription
    transcript_info: Dict[str, Any] = {
        "has_audio": audio_result.has_audio,
        "transcript_path": None,
        "language": None,
        "segment_count": 0,
        "word_count": 0,
        "has_word_alignment": False,
    }

    if audio_result.has_audio and audio_result.audio_path:
        logger.info(
            "[Stage 4/4] Transcribing audio with local WhisperX (%s)...",
            cfg.whisper_model,
        )
        try:
            worker = transcriber or LocalTranscriber(
                model_name=cfg.whisper_model,
                device=cfg.device,
                compute_type=cfg.compute_type,
                vad_method=cfg.vad_method,
                batch_size=cfg.batch_size,
            )
            t_res = worker.transcribe_audio(
                audio_path=audio_result.audio_path,
                video_id=video_id,
                output_dir=cfg.transcripts_dir,
            )
            transcript_info = {
                "has_audio": True,
                "transcript_path": t_res.transcript_path,
                "language": t_res.language,
                "segment_count": t_res.segment_count,
                "word_count": t_res.word_count,
                "has_word_alignment": t_res.has_word_alignment,
            }
        except Exception as t_exc:
            logger.error(
                "Transcription failed for %s: %s", v_path.name, t_exc, exc_info=True
            )
            transcript_info["error"] = str(t_exc)
    else:
        logger.info(
            "[Stage 4/4] Skipping transcription (no audio track present)."
        )

    pipeline_duration = round(time.time() - start_time, 3)
    status = (
        "completed" if audio_result.has_audio else "completed_no_audio"
    )

    # Stage 8: Store consolidated metadata JSON
    final_metadata = {
        "video_id": video_id,
        "source_video_path": str(v_path),
        "file_name": v_path.name,
        "file_size_bytes": metadata.file_size_bytes,
        "processing_status": status,
        "error_message": None,
        "processed_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "pipeline_duration_seconds": pipeline_duration,
        "video_metadata": metadata.to_dict(),
        "frame_extraction": {
            "frame_interval_seconds": cfg.frame_interval_seconds,
            "extracted_frame_count": len(frame_records),
            "frames_dir": str(
                (cfg.frames_dir / video_id).resolve()
            ),
            "frames": [f.to_dict() for f in frame_records],
        },
        "audio_extraction": audio_result.to_dict(),
        "transcription": transcript_info,
    }

    meta_file = cfg.metadata_dir / f"{video_id}.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "Successfully finished processing %s (video_id=%s, status=%s, frames=%d, duration=%.2fs)",
        v_path.name,
        video_id,
        status,
        len(frame_records),
        pipeline_duration,
    )

    return PipelineResult(
        video_id=video_id,
        video_path=str(v_path),
        status=status,
        metadata_path=str(meta_file.resolve()),
        frames_count=len(frame_records),
        has_audio=audio_result.has_audio,
        segment_count=transcript_info["segment_count"],
        duration_seconds=pipeline_duration,
    )


def process_all_videos(
    config: Optional[Module1Config] = None,
    force: bool = False,
) -> PipelineBatchResult:
    """Discover all valid videos in input directory and process them sequentially.

    Args:
        config: Module configuration.
        force: If True, force reprocessing of already completed videos.

    Returns:
        PipelineBatchResult with cumulative summary metrics.
    """
    cfg = config or Module1Config()
    cfg.ensure_directories()

    discovered = discover_videos(
        input_dir=cfg.input_videos_dir,
        recursive=True,
        supported_extensions=cfg.supported_extensions,
    )

    if not discovered:
        logger.warning(
            "No videos found in input directory: %s", cfg.input_videos_dir
        )
        return PipelineBatchResult(
            total_videos=0,
            processed_count=0,
            skipped_count=0,
            failed_count=0,
            results=[],
        )

    logger.info(
        "Starting batch processing of %d discovered video(s)...", len(discovered)
    )

    # Initialize transcriber once to share model weights across batch
    transcriber = LocalTranscriber(
        model_name=cfg.whisper_model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        vad_method=cfg.vad_method,
        batch_size=cfg.batch_size,
    )

    results: List[PipelineResult] = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, video_path in enumerate(discovered, 1):
        logger.info(
            "--- Processing batch item %d of %d: %s ---",
            idx,
            len(discovered),
            video_path.name,
        )
        try:
            res = process_video(
                video_path=video_path,
                config=cfg,
                force=force,
                transcriber=transcriber,
            )
            results.append(res)
            if res.status in ("completed", "completed_no_audio"):
                processed_count += 1
            elif res.status == "skipped":
                skipped_count += 1
            else:
                failed_count += 1
        except Exception as batch_exc:
            logger.error(
                "Unhandled error during processing of %s: %s",
                video_path.name,
                batch_exc,
                exc_info=True,
            )
            failed_count += 1
            results.append(
                PipelineResult(
                    video_id="unknown",
                    video_path=str(video_path),
                    status="failed",
                    error_message=str(batch_exc),
                )
            )

    logger.info(
        "Batch run complete: Total=%d, Processed=%d, Skipped=%d, Failed=%d",
        len(discovered),
        processed_count,
        skipped_count,
        failed_count,
    )

    return PipelineBatchResult(
        total_videos=len(discovered),
        processed_count=processed_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        results=results,
    )
