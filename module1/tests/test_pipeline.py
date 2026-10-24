"""End-to-end integration tests for Module 1 pipeline."""

import json
from pathlib import Path

from src.config import Module1Config
from src.id_generation import generate_video_id
from src.pipeline import is_already_processed, process_all_videos, process_video


def test_pipeline_end_to_end_single_video(
    sample_video_with_audio: Path, temp_config: Module1Config
):
    """Executes full pipeline on a valid video and validates all outputs."""
    result = process_video(sample_video_with_audio, config=temp_config)

    assert result.status == "completed"
    assert result.video_id != "unknown"
    assert result.frames_count > 0
    assert result.has_audio is True

    # 1. Frames verification
    frames_dir = temp_config.frames_dir / result.video_id
    assert frames_dir.exists()
    frames = list(frames_dir.glob("*.jpg"))
    assert len(frames) == result.frames_count

    # 2. Audio verification
    audio_file = temp_config.audio_dir / f"{result.video_id}.wav"
    assert audio_file.exists()
    assert audio_file.stat().st_size > 0

    # 3. Transcript verification
    transcript_file = temp_config.transcripts_dir / f"{result.video_id}.json"
    assert transcript_file.exists()
    with open(transcript_file, "r", encoding="utf-8") as f:
        t_data = json.load(f)
    assert t_data["video_id"] == result.video_id
    assert "segments" in t_data

    # 4. Metadata verification
    meta_file = temp_config.metadata_dir / f"{result.video_id}.json"
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    assert m_data["video_id"] == result.video_id
    assert m_data["processing_status"] == "completed"
    assert m_data["frame_extraction"]["extracted_frame_count"] == result.frames_count
    assert len(m_data["frame_extraction"]["frames"]) == result.frames_count


def test_pipeline_skip_already_processed(
    sample_video_with_audio: Path, temp_config: Module1Config
):
    """Second run without force must be skipped."""
    res1 = process_video(sample_video_with_audio, config=temp_config)
    assert res1.status == "completed"
    assert is_already_processed(res1.video_id, temp_config) is True

    res2 = process_video(sample_video_with_audio, config=temp_config, force=False)
    assert res2.status == "skipped"

    # Re-running with force=True must re-process
    res3 = process_video(sample_video_with_audio, config=temp_config, force=True)
    assert res3.status == "completed"


def test_pipeline_no_audio_video(
    sample_video_no_audio: Path, temp_config: Module1Config
):
    """Processes video with no audio track gracefully; frames extracted, audio skipped."""
    result = process_video(sample_video_no_audio, config=temp_config)

    assert result.status == "completed_no_audio"
    assert result.frames_count > 0
    assert result.has_audio is False

    # Check frames exist
    frames_dir = temp_config.frames_dir / result.video_id
    assert frames_dir.exists()
    assert len(list(frames_dir.glob("*.jpg"))) == result.frames_count

    # Check audio and transcript files do NOT exist
    audio_file = temp_config.audio_dir / f"{result.video_id}.wav"
    assert not audio_file.exists()

    # Check metadata reflects completed_no_audio
    meta_file = temp_config.metadata_dir / f"{result.video_id}.json"
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    assert m_data["processing_status"] == "completed_no_audio"
    assert m_data["audio_extraction"]["has_audio"] is False


def test_pipeline_corrupt_video_fails_gracefully(
    sample_corrupt_video: Path, temp_config: Module1Config
):
    """Corrupt video fails gracefully without raising unhandled exception."""
    result = process_video(sample_corrupt_video, config=temp_config)
    assert result.status == "failed"
    assert result.error_message is not None


def test_pipeline_batch_execution(
    sample_video_with_audio: Path,
    sample_video_no_audio: Path,
    sample_corrupt_video: Path,
    temp_config: Module1Config,
):
    """Batch run handles a mix of valid, no-audio, and corrupt videos cleanly."""
    # Place all videos into temp_config input_videos_dir
    in_dir = temp_config.input_videos_dir
    import shutil
    shutil.copy(sample_video_with_audio, in_dir / "vid1.mp4")
    shutil.copy(sample_video_no_audio, in_dir / "vid2.mkv")
    shutil.copy(sample_corrupt_video, in_dir / "vid3.mp4")

    batch_result = process_all_videos(config=temp_config)
    assert batch_result.total_videos == 3
    assert batch_result.processed_count == 2  # 1 with audio + 1 no audio
    assert batch_result.failed_count == 1      # corrupt video
