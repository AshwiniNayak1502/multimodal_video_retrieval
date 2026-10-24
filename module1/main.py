"""CLI entrypoint for Module 1: Video Ingestion, Frames & Transcription."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import Module1Config
from src.logging_config import setup_logging
from src.pipeline import process_all_videos, process_video


def build_parser() -> argparse.ArgumentParser:
    """Create and configure command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Module 1: Video Ingestion, Representative Frame Extraction, and Audio Transcription.\n"
            "Processes local video files (.mp4/.mkv), extracts time-stamped frames, audio track,\n"
            "and generates word-aligned speech transcripts via local WhisperX."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Discover and process all supported videos (.mp4, .mkv) in the input directory.",
    )
    group.add_argument(
        "--video",
        type=str,
        help="Path to a single video file to process.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of video(s) even if outputs already exist.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Sampling interval in seconds for representative frames (default: 2.0).",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Whisper model name for transcription (default: 'tiny', e.g. 'base', 'small').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom JSON configuration file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to output log file.",
    )

    return parser


def main() -> int:
    """Main execution function for Module 1 CLI."""
    parser = build_parser()
    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file_path = Path(args.log_file) if args.log_file else None
    logger = setup_logging(level=log_level, log_file=log_file_path)

    logger.info("Initializing Module 1 pipeline...")

    # Load base configuration
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            logger.error("Specified config file not found: %s", cfg_path)
            return 1
        config = Module1Config.from_json(cfg_path)
        logger.info("Loaded configuration from %s", cfg_path)
    else:
        config = Module1Config()

    # Apply CLI overrides
    if args.interval is not None:
        if args.interval <= 0:
            logger.error("--interval must be greater than 0.")
            return 1
        config.frame_interval_seconds = args.interval
        logger.info(
            "CLI override: frame_interval_seconds = %.2fs", args.interval
        )

    if args.whisper_model:
        config.whisper_model = args.whisper_model
        logger.info("CLI override: whisper_model = '%s'", args.whisper_model)

    config.ensure_directories()

    # Execute single video mode
    if args.video:
        target_path = Path(args.video).resolve()
        if not target_path.exists():
            logger.error("Target video file not found: %s", target_path)
            return 1

        res = process_video(
            video_path=target_path,
            config=config,
            force=args.force,
        )

        logger.info("--------------------------------------------------")
        logger.info("Execution Summary:")
        logger.info("  Video:         %s", target_path.name)
        logger.info("  Video ID:      %s", res.video_id)
        logger.info("  Status:        %s", res.status)
        logger.info("  Frames:        %d", res.frames_count)
        logger.info("  Audio:         %s", res.has_audio)
        logger.info("  Segments:      %d", res.segment_count)
        logger.info("  Duration:      %.2fs", res.duration_seconds)
        if res.metadata_path:
            logger.info("  Metadata:      %s", res.metadata_path)
        if res.error_message:
            logger.error("  Error Details: %s", res.error_message)
        logger.info("--------------------------------------------------")

        return 0 if res.status in ("completed", "completed_no_audio", "skipped") else 1

    # Execute batch all mode
    if args.all:
        batch_res = process_all_videos(
            config=config,
            force=args.force,
        )

        logger.info("==================================================")
        logger.info("Batch Execution Summary:")
        logger.info("  Total Discovered: %d", batch_res.total_videos)
        logger.info("  Processed:        %d", batch_res.processed_count)
        logger.info("  Skipped:          %d", batch_res.skipped_count)
        logger.info("  Failed:           %d", batch_res.failed_count)
        logger.info("==================================================")

        return 0 if batch_res.failed_count == 0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
