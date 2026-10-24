"""Shared fixtures and synthetic data generators for Module 3 test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

# Ensure module2 and module3 are accessible
_MODULE2_DIR = (Path(__file__).resolve().parent.parent.parent / "module2").resolve()
_MODULE3_DIR = (Path(__file__).resolve().parent.parent).resolve()

for p in (_MODULE2_DIR, _MODULE3_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from embedding_store.config import Module2Config
from embedding_store.embedders import FakeTextEmbedder, FakeVisualEmbedder
from embedding_store.indexer import VideoIndexer
from embedding_store.retrieval import VideoIndex

from retrieval_engine.config import Module3Config
from retrieval_engine.query_service import QueryService


@pytest.fixture
def synthetic_m1_tree(tmp_path: Path) -> Path:
    """Build a complete, synthetic Module 1 output tree in tmp_path."""
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
        color = ((idx * 60) % 255, (idx * 80) % 255, (idx * 100) % 255)
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

    with open(vid1_frames_dir / "frames_manifest.json", "w", encoding="utf-8") as f:
        json.dump(frame_records_1, f, indent=2)

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
                "start_time": 0.150,
                "end_time": 3.400,
                "text": "Welcome to multimodal semantic query processing.",
                "words": [],
            },
            {
                "segment_id": 1,
                "start_time": 4.200,
                "end_time": 6.800,
                "text": "This module retrieves timestamped video segments.",
                "words": [],
            },
        ],
    }
    with open(vid1_transcript_file, "w", encoding="utf-8") as f:
        json.dump(transcript_data_1, f, indent=2)

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
        color = ((idx * 40) % 255, (idx * 90) % 255, (idx * 120) % 255)
        img = Image.new("RGB", (64, 64), color=color)
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

    with open(vid2_frames_dir / "frames_manifest.json", "w", encoding="utf-8") as f:
        json.dump(frame_records_2, f, indent=2)

    master_data_2 = {
        "video_id": vid2,
        "source_video_path": str(video_silent.resolve()),
        "file_name": "silent_clip.mkv",
        "file_size_bytes": 8192,
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

    return m1_root


@pytest.fixture
def indexed_test_environment(
    tmp_path: Path, synthetic_m1_tree: Path
) -> Tuple[QueryService, Module3Config, FakeVisualEmbedder, FakeTextEmbedder]:
    """Index synthetic M1 data using mock embedders and return QueryService."""
    m2_root = tmp_path / "module2"
    m2_out = m2_root / "output"
    vdb_dir = m2_out / "vector_db"

    m2_cfg = Module2Config(
        base_dir=m2_root,
        module1_root=synthetic_m1_tree,
        module1_output_dir=synthetic_m1_tree / "output",
        module1_input_dir=synthetic_m1_tree / "input" / "videos",
        output_dir=m2_out,
        vector_db_dir=vdb_dir,
        visual_model_name="mock/siglip-fake",
        text_model_name="mock/minilm-fake",
        device="cpu",
        offline_mode=True,
    )
    m2_cfg.ensure_directories()

    vis_emb = FakeVisualEmbedder(dim=768, model_name="mock/siglip-fake")
    txt_emb = FakeTextEmbedder(dim=384, model_name="mock/minilm-fake")

    # Run Module 2 indexer with fakes
    indexer = VideoIndexer(config=m2_cfg, visual_embedder=vis_emb, text_embedder=txt_emb)
    summary = indexer.run_indexing()
    assert summary.videos_processed >= 2
    indexer.store.close()

    # Configure Module 3 pointing to this temporary environment
    m3_root = tmp_path / "module3"
    m3_out = m3_root / "output"
    m3_cfg = Module3Config(
        base_dir=m3_root,
        module2_dir=m2_root,
        module1_dir=synthetic_m1_tree,
        vector_db_dir=vdb_dir,
        output_dir=m3_out,
        visual_min_prob=0.10,
        text_min_cosine=0.10,
        final_min_score=0.15,
    )
    m3_cfg.ensure_directories()

    v_index = VideoIndex.open(
        config=m2_cfg,
        visual_embedder=vis_emb,
        text_embedder=txt_emb,
    )

    service = QueryService(config=m3_cfg, video_index=v_index, warmup=False)
    return service, m3_cfg, vis_emb, txt_emb


@pytest.fixture(autouse=True)
def redirect_test_logging(tmp_path: Path) -> None:
    """Redirect module3 logging away from production module3/output/module3.log (Item 11e)."""
    from retrieval_engine.logging_config import setup_logging
    test_log = tmp_path / "test_module3.log"
    setup_logging(log_file=test_log)


@pytest.fixture
def synthetic_cli_config_file(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any], tmp_path: Path
) -> Path:
    """Write temporary config.json pointing to synthetic index for CLI testing (Item 11d)."""
    _, m3_cfg, _, _ = indexed_test_environment
    cfg_file = tmp_path / "synthetic_cli_config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(m3_cfg.to_dict(), f, indent=2)
    return cfg_file

