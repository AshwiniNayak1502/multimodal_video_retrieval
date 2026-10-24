"""Unit tests for SQLite persistent metadata store and BLOB management."""

from pathlib import Path

import numpy as np
import pytest

from embedding_store.metadata_store import MetadataStore
from embedding_store.schema import IndexFailure, TextItem, VideoRecord, VisualItem


@pytest.fixture
def store(tmp_path: Path) -> MetadataStore:
    db_file = tmp_path / "test_metadata.db"
    return MetadataStore(db_file)


def _sample_video_record(vid_id: str = "vid_001") -> VideoRecord:
    return VideoRecord(
        video_id=vid_id,
        video_filename="test.mp4",
        video_path=f"/path/to/{vid_id}.mp4",
        video_relpath=f"videos/{vid_id}.mp4",
        duration_seconds=10.0,
        fps=25.0,
        width=1280,
        height=720,
        frame_interval_seconds=2.0,
        has_audio=True,
        language="en",
        transcript_path=f"/path/to/{vid_id}.json",
        m1_metadata_path=f"/path/to/meta/{vid_id}.json",
        m1_processed_at="2026-10-10T10:00:00",
        indexed_at="2026-10-10T10:05:00",
    )


def test_schema_init_and_foreign_keys(store: MetadataStore):
    """Verify tables and foreign key cascade delete enforcement."""
    cur = store.connection.cursor()
    cur.execute("PRAGMA foreign_keys")
    fk_status = cur.fetchone()[0]
    assert fk_status == 1

    # Insert video
    v_rec = _sample_video_record("vid_fk_test")
    store.upsert_video(v_rec)

    # Insert visual item referencing video
    vis_item = VisualItem(
        frame_id="vid_fk_test_frame_00000",
        video_id="vid_fk_test",
        frame_index=0,
        source_frame_number=0,
        timestamp_seconds=0.0,
        timestamp_formatted="00:00:00",
        frame_path="/tmp/f0.jpg",
        frame_relpath="frames/f0.jpg",
        width=1280,
        height=720,
        content_hash="hash0",
    )
    vec = np.ones(4, dtype=np.float32)
    vid_id_num = store.insert_visual_vector(vis_item, vec, "model_v")

    # Delete video -> verify cascade deleted visual vector
    store.delete_video("vid_fk_test")
    cur.execute("SELECT COUNT(*) FROM visual_vectors WHERE vector_id = ?", (vid_id_num,))
    assert cur.fetchone()[0] == 0


def test_embedding_blob_storage_and_unpacking(store: MetadataStore):
    """Verify float32 vectors are stored as BLOBs and accurately reconstructed."""
    v_rec = _sample_video_record("vid_blob_test")
    store.upsert_video(v_rec)

    vis_item = VisualItem(
        frame_id="vid_blob_test_frame_00001",
        video_id="vid_blob_test",
        frame_index=1,
        source_frame_number=50,
        timestamp_seconds=2.0,
        timestamp_formatted="00:00:02",
        frame_path="/tmp/f1.jpg",
        frame_relpath="frames/f1.jpg",
        width=640,
        height=360,
        content_hash="hash1",
    )
    original_vec = np.array([0.1, -0.5, 0.9, -0.3], dtype=np.float32)
    vec_id = store.insert_visual_vector(vis_item, original_vec, "test_model")
    assert vec_id > 0

    # Retrieve all embeddings
    ids, mat = store.get_all_embeddings("visual")
    assert len(ids) == 1
    assert ids[0] == vec_id
    assert np.allclose(mat[0], original_vec)


def test_model_mismatch_refusal(store: MetadataStore):
    """Verify store refuses to mix different model architectures or dimensions."""
    store.check_or_set_models("model_a", 768, "model_b", 384)

    # Calling with same parameters passes
    store.check_or_set_models("model_a", 768, "model_b", 384)

    # Calling with different model raises ValueError
    with pytest.raises(ValueError) as exc_info:
        store.check_or_set_models("model_c", 768, "model_b", 384)
    assert "--rebuild" in str(exc_info.value)


def test_window_queries(store: MetadataStore):
    """Verify get_frames_in_window and get_segments_in_window SQL helpers."""
    vid = "vid_win_test"
    store.upsert_video(_sample_video_record(vid))

    # Insert 3 frames at 0s, 2s, 4s
    for idx, ts in enumerate([0.0, 2.0, 4.0]):
        item = VisualItem(
            frame_id=f"{vid}_frame_{idx:05d}",
            video_id=vid,
            frame_index=idx,
            source_frame_number=idx * 50,
            timestamp_seconds=ts,
            timestamp_formatted=f"00:00:0{int(ts)}",
            frame_path=f"/path/{idx}.jpg",
            frame_relpath=None,
            width=100,
            height=100,
            content_hash=f"h{idx}",
        )
        store.insert_visual_vector(item, np.zeros(4, dtype=np.float32), "m")

    # Insert 2 segments: (0.0 -> 2.5), (3.0 -> 5.0)
    store.insert_text_vector(
        TextItem(
            segment_uid=f"{vid}_seg_00000",
            video_id=vid,
            segment_id=0,
            transcript_text="first speech",
            start_timestamp=0.0,
            end_timestamp=2.5,
            start_formatted="00:00:00",
            end_formatted="00:00:02",
            language="en",
            content_hash="th0",
        ),
        np.zeros(4, dtype=np.float32),
        "tm",
    )
    store.insert_text_vector(
        TextItem(
            segment_uid=f"{vid}_seg_00001",
            video_id=vid,
            segment_id=1,
            transcript_text="second speech",
            start_timestamp=3.0,
            end_timestamp=5.0,
            start_formatted="00:00:03",
            end_formatted="00:00:05",
            language="en",
            content_hash="th1",
        ),
        np.zeros(4, dtype=np.float32),
        "tm",
    )

    # Frame window [1.5, 3.5] should only capture frame at 2.0s
    frames = store.get_frames_in_window(vid, 1.5, 3.5)
    assert len(frames) == 1
    assert frames[0]["timestamp_seconds"] == 2.0

    # Segment window [2.0, 3.2] overlaps BOTH segments
    segs = store.get_segments_in_window(vid, 2.0, 3.2)
    assert len(segs) == 2
