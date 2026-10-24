"""Unit tests for Module 1 output reader and path re-anchoring engine."""

import json
from pathlib import Path

import pytest

from embedding_store.config import Module2Config
from embedding_store.m1_reader import (
    format_timestamp,
    iter_videos,
    read_master_metadata,
    resolve_frame_path,
    resolve_video_path,
)


def test_format_timestamp():
    """Verify exact floor-based HH:MM:SS string formatting."""
    assert format_timestamp(0.0) == "00:00:00"
    assert format_timestamp(2.0) == "00:00:02"
    assert format_timestamp(2.856) == "00:00:02"
    assert format_timestamp(59.9) == "00:00:59"
    assert format_timestamp(60.0) == "00:01:00"
    assert format_timestamp(123.4) == "00:02:03"
    assert format_timestamp(3665.1) == "01:01:05"
    assert format_timestamp(-1.0) == "00:00:00"


def test_iter_videos_discovery(temp_config: Module2Config):
    """Verify iter_videos discovers completed videos and skips failed ones."""
    videos = list(iter_videos(temp_config))
    # Synthetic tree has vid_test001 (completed), vid_silent002 (completed_no_audio), vid_failed003 (failed)
    vid_ids = [v[0].video_id for v in videos]
    assert "vid_test001" in vid_ids
    assert "vid_silent002" in vid_ids
    assert "vid_failed003" not in vid_ids


def test_read_master_metadata_completed_video(temp_config: Module2Config):
    """Verify master metadata, frames, and transcript segments parsing."""
    json_path = temp_config.module1_output_dir / "metadata" / "vid_test001.json"
    parsed = read_master_metadata(json_path, temp_config)
    assert parsed is not None

    v_rec, vis_items, txt_items, fails = parsed
    assert v_rec.video_id == "vid_test001"
    assert v_rec.video_filename == "lecture_sample.mp4"
    assert v_rec.has_audio is True

    # 4 frames extracted
    assert len(vis_items) == 4
    assert [item.timestamp_seconds for item in vis_items] == [0.0, 2.0, 4.0, 6.0]
    assert vis_items[0].frame_id == "vid_test001_frame_00000"
    assert vis_items[1].frame_id == "vid_test001_frame_00001"
    assert vis_items[0].timestamp_formatted == "00:00:00"
    assert vis_items[1].timestamp_formatted == "00:00:02"
    assert Path(vis_items[0].frame_path).is_file()

    # 2 transcript segments
    assert len(txt_items) == 2
    assert txt_items[0].segment_uid == "vid_test001_seg_00000"
    assert txt_items[0].start_timestamp == 0.158
    assert txt_items[0].end_timestamp == 3.402
    assert txt_items[0].transcript_text == "Welcome to multimodal semantic indexing."
    assert txt_items[0].start_formatted == "00:00:00"
    assert txt_items[0].end_formatted == "00:00:03"

    assert txt_items[1].segment_uid == "vid_test001_seg_00001"
    assert txt_items[1].start_timestamp == 4.383
    assert txt_items[1].end_timestamp == 6.966
    assert len(fails) == 0


def test_read_silent_video(temp_config: Module2Config):
    """Verify video with has_audio=False yields frames and zero text segments without error."""
    json_path = temp_config.module1_output_dir / "metadata" / "vid_silent002.json"
    parsed = read_master_metadata(json_path, temp_config)
    assert parsed is not None

    v_rec, vis_items, txt_items, fails = parsed
    assert v_rec.video_id == "vid_silent002"
    assert v_rec.has_audio is False
    assert len(vis_items) == 3
    assert len(txt_items) == 0
    assert len(fails) == 0


def test_reanchoring_when_stored_absolute_path_missing(temp_config: Module2Config, tmp_path: Path):
    """Verify path re-anchoring when stored paths point to another machine."""
    vid_id = "vid_reanchor_test"
    other_machine_path = r"D:\AnotherUser\Project\module1\output\frames\vid_reanchor_test\frame_0000_0.000s.jpg"

    # Create real frame on local disk
    local_frame_dir = temp_config.module1_output_dir / "frames" / vid_id
    local_frame_dir.mkdir(parents=True, exist_ok=True)
    local_frame_path = local_frame_dir / "frame_0000_0.000s.jpg"
    local_frame_path.write_bytes(b"image_content")

    f_record = {
        "frame_index": 0,
        "file_path": other_machine_path,
        "timestamp_seconds": 0.0,
    }

    resolved_abs, rel = resolve_frame_path(f_record, vid_id, temp_config)
    assert Path(resolved_abs).is_file()
    assert resolved_abs == str(local_frame_path.resolve())
    assert rel == f"frames/{vid_id}/frame_0000_0.000s.jpg"


def test_resolve_video_path_search(temp_config: Module2Config):
    """Verify video path resolution searches module1_input_dir recursively."""
    master = {
        "source_video_path": r"Z:\MissingMachine\lecture_sample.mp4",
        "file_name": "lecture_sample.mp4",
        "video_metadata": {
            "file_name": "lecture_sample.mp4",
            "file_path": r"Z:\MissingMachine\lecture_sample.mp4",
        },
    }
    resolved_abs, rel = resolve_video_path(master, temp_config)
    assert Path(resolved_abs).is_file()
    assert Path(resolved_abs).name == "lecture_sample.mp4"


def test_invalid_segment_skipped_and_logged(temp_config: Module2Config):
    """Verify transcript segments with end < start or invalid timestamps are skipped and recorded."""
    vid_id = "vid_bad_seg"
    meta_path = temp_config.module1_output_dir / "metadata" / f"{vid_id}.json"
    trans_path = temp_config.module1_output_dir / "transcripts" / f"{vid_id}.json"
    frames_dir = temp_config.module1_output_dir / "frames" / vid_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    img_file = frames_dir / "frame_0000_0.000s.jpg"
    img_file.write_bytes(b"dummy_frame")

    trans_data = {
        "video_id": vid_id,
        "segments": [
            {"segment_id": 0, "start_time": 5.0, "end_time": 2.0, "text": "Backwards segment"},
            {"segment_id": 1, "start_time": 1.0, "end_time": 3.0, "text": "Valid segment"},
            {"segment_id": 2, "start_time": -1.0, "end_time": 3.0, "text": "Negative start"},
        ],
    }
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(trans_data, f)

    meta_data = {
        "video_id": vid_id,
        "processing_status": "completed",
        "file_name": "test.mp4",
        "video_metadata": {"duration_seconds": 10.0, "has_audio": True},
        "frame_extraction": {
            "frames": [{"frame_index": 0, "timestamp_seconds": 0.0, "file_path": str(img_file)}],
        },
        "transcription": {"has_audio": True, "transcript_path": str(trans_path)},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)

    parsed = read_master_metadata(meta_path, temp_config)
    assert parsed is not None
    _, _, txt_items, fails = parsed
    # Only segment 1 is valid
    assert len(txt_items) == 1
    assert txt_items[0].segment_id == 1
    # 2 failures recorded
    assert len(fails) == 2
    assert all(f.modality == "text" for f in fails)
