"""Shared fixtures and synthetic data generators for Module 2 test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pytest
from PIL import Image

from embedding_store.config import Module2Config
from embedding_store.embedders import FakeTextEmbedder, FakeVisualEmbedder


@pytest.fixture
def synthetic_m1_tree(tmp_path: Path) -> Path:
    """Build a complete, synthetic Module 1 output filesystem in tmp_path.

    Creates:
    - input/videos/lecture_sample.mp4 & silent_clip.mkv
    - output/metadata/vid_test001.json (completed, audio, 4 frames, 2 transcript segments)
    - output/frames/vid_test001/frame_0000_0.000s.jpg, ..., frame_0003_6.000s.jpg & frames_manifest.json
    - output/transcripts/vid_test001.json
    - output/metadata/vid_silent002.json (completed_no_audio, 3 frames, no transcript)
    - output/frames/vid_silent002/frame_0000_0.000s.jpg, ..., frame_0002_4.000s.jpg
    - output/metadata/vid_failed003.json (processing_status = 'failed')
    """
    m1_root = tmp_path / "module1"
    input_videos = m1_root / "input" / "videos"
    output_meta = m1_root / "output" / "metadata"
    output_frames = m1_root / "output" / "frames"
    output_trans = m1_root / "output" / "transcripts"

    input_videos.mkdir(parents=True, exist_ok=True)
    output_meta.mkdir(parents=True, exist_ok=True)
    output_frames.mkdir(parents=True, exist_ok=True)
    output_trans.mkdir(parents=True, exist_ok=True)

    # 1. Dummy video files
    video_mp4 = input_videos / "lecture_sample.mp4"
    video_mp4.write_bytes(b"dummy_mp4_bytes")

    video_silent = input_videos / "silent_clip.mkv"
    video_silent.write_bytes(b"dummy_mkv_bytes")

    # 2. vid_test001: 4 frames, 2 transcript segments
    vid1 = "vid_test001"
    vid1_frames_dir = output_frames / vid1
    vid1_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records_1 = []
    ts_list_1 = [0.0, 2.0, 4.0, 6.0]
    for idx, ts in enumerate(ts_list_1):
        filename = f"frame_{idx:04d}_{ts:.3f}s.jpg"
        fpath = vid1_frames_dir / filename
        # Create a small distinct Pillow image
        color = ((idx * 50) % 255, (idx * 80) % 255, (idx * 110) % 255)
        img = Image.new("RGB", (64, 64), color=color)
        img.save(fpath, format="JPEG")

        frame_records_1.append({
            "video_id": vid1,
            "frame_index": idx,
            "source_frame_number": idx * 50,
            "timestamp_seconds": ts,
            "file_path": str(fpath.resolve()),
            "width": 64,
            "height": 64,
        })

    # Save frames_manifest.json
    with open(vid1_frames_dir / "frames_manifest.json", "w", encoding="utf-8") as f:
        json.dump(frame_records_1, f, indent=2)

    # Transcript for vid_test001
    vid1_transcript_file = output_trans / f"{vid1}.json"
    transcript_data_1 = {
        "video_id": vid1,
        "audio_path": str(m1_root / "output" / "audio" / f"{vid1}.wav"),
        "language": "en",
        "segment_count": 2,
        "word_count": 14,
        "segments": [
            {
                "segment_id": 0,
                "start_time": 0.158,
                "end_time": 3.402,
                "text": "Welcome to multimodal semantic indexing.",
                "words": [],
            },
            {
                "segment_id": 1,
                "start_time": 4.383,
                "end_time": 6.966,
                "text": "This module indexes visual frames and speech transcripts.",
                "words": [],
            },
        ],
    }
    with open(vid1_transcript_file, "w", encoding="utf-8") as f:
        json.dump(transcript_data_1, f, indent=2)

    # Master metadata for vid_test001
    master_data_1 = {
        "video_id": vid1,
        "source_video_path": str(video_mp4.resolve()),
        "file_name": "lecture_sample.mp4",
        "file_size_bytes": 10240,
        "processing_status": "completed",
        "processed_at": "2026-10-10T12:00:00.000000+00:00",
        "pipeline_duration_seconds": 8.0,
        "video_metadata": {
            "video_id": vid1,
            "file_name": "lecture_sample.mp4",
            "file_path": str(video_mp4.resolve()),
            "duration_seconds": 8.0,
            "width": 64,
            "height": 64,
            "fps": 25.0,
            "has_audio": True,
        },
        "frame_extraction": {
            "frame_interval_seconds": 2.0,
            "extracted_frame_count": 4,
            "frames_dir": str(vid1_frames_dir.resolve()),
            "frames": frame_records_1,
        },
        "audio_extraction": {
            "has_audio": True,
            "audio_path": str(m1_root / "output" / "audio" / f"{vid1}.wav"),
        },
        "transcription": {
            "has_audio": True,
            "transcript_path": str(vid1_transcript_file.resolve()),
            "language": "en",
            "segment_count": 2,
        },
    }
    with open(output_meta / f"{vid1}.json", "w", encoding="utf-8") as f:
        json.dump(master_data_1, f, indent=2)

    # 3. vid_silent002: no audio, 3 frames
    vid2 = "vid_silent002"
    vid2_frames_dir = output_frames / vid2
    vid2_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records_2 = []
    ts_list_2 = [0.0, 2.0, 4.0]
    for idx, ts in enumerate(ts_list_2):
        filename = f"frame_{idx:04d}_{ts:.3f}s.jpg"
        fpath = vid2_frames_dir / filename
        img = Image.new("RGB", (64, 64), color=(100, 100, idx * 50))
        img.save(fpath, format="JPEG")

        frame_records_2.append({
            "video_id": vid2,
            "frame_index": idx,
            "source_frame_number": idx * 50,
            "timestamp_seconds": ts,
            "file_path": str(fpath.resolve()),
            "width": 64,
            "height": 64,
        })

    master_data_2 = {
        "video_id": vid2,
        "source_video_path": str(video_silent.resolve()),
        "file_name": "silent_clip.mkv",
        "file_size_bytes": 5120,
        "processing_status": "completed_no_audio",
        "processed_at": "2026-10-10T12:05:00.000000+00:00",
        "pipeline_duration_seconds": 6.0,
        "video_metadata": {
            "video_id": vid2,
            "file_name": "silent_clip.mkv",
            "file_path": str(video_silent.resolve()),
            "duration_seconds": 6.0,
            "width": 64,
            "height": 64,
            "fps": 25.0,
            "has_audio": False,
        },
        "frame_extraction": {
            "frame_interval_seconds": 2.0,
            "extracted_frame_count": 3,
            "frames_dir": str(vid2_frames_dir.resolve()),
            "frames": frame_records_2,
        },
        "audio_extraction": {
            "has_audio": False,
            "audio_path": None,
        },
        "transcription": {
            "has_audio": False,
            "transcript_path": None,
            "language": None,
            "segment_count": 0,
        },
    }
    with open(output_meta / f"{vid2}.json", "w", encoding="utf-8") as f:
        json.dump(master_data_2, f, indent=2)

    # 4. vid_failed003: failed processing status
    vid3 = "vid_failed003"
    master_data_3 = {
        "video_id": vid3,
        "file_name": "bad_video.mp4",
        "processing_status": "failed",
        "error_message": "Container decode error",
    }
    with open(output_meta / f"{vid3}.json", "w", encoding="utf-8") as f:
        json.dump(master_data_3, f, indent=2)

    return m1_root


@pytest.fixture
def temp_config(tmp_path: Path, synthetic_m1_tree: Path) -> Module2Config:
    """Create a temporary Module2Config wired to synthetic Module 1 tree."""
    m2_root = tmp_path / "module2"
    cfg = Module2Config(
        base_dir=m2_root,
        module1_root=synthetic_m1_tree,
        module1_output_dir=synthetic_m1_tree / "output",
        module1_input_dir=synthetic_m1_tree / "input",
        output_dir=m2_root / "output",
        vector_db_dir=m2_root / "output" / "vector_db",
        log_file=m2_root / "output" / "module2.log",
        visual_model_name="mock/siglip-fake",
        text_model_name="mock/minilm-fake",
        device="cpu",
        offline_mode=True,
    )
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def fake_embedders() -> Tuple[FakeVisualEmbedder, FakeTextEmbedder]:
    """Provide deterministic mock embedders with default standard dims."""
    return FakeVisualEmbedder(dim=768), FakeTextEmbedder(dim=384)
