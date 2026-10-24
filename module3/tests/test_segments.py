"""Unit tests for segment assembly, temporal cross-linking, and deduplication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval_engine.segments import (
    EvidenceHit,
    Result,
    assemble_and_deduplicate_segments,
    format_timestamp,
)


def test_deduplicate_consecutive_near_identical_frames() -> None:
    """Consecutive frames within tolerance must be merged, keeping the best score."""
    # Three consecutive frames at 0s, 2s, 4s of the same scene in vid1
    v_hits = [
        EvidenceHit(
            modality="visual",
            video_id="vid_1",
            video_filename="clip.mp4",
            video_path="/path/clip.mp4",
            score=0.70,
            raw_score=0.20,
            start_seconds=0.0,
            end_seconds=1.0,
            rank=3,
            matched_at_seconds=0.0,
            frame_id="f0",
        ),
        EvidenceHit(
            modality="visual",
            video_id="vid_1",
            video_filename="clip.mp4",
            video_path="/path/clip.mp4",
            score=0.88,  # Peak score
            raw_score=0.24,
            start_seconds=1.0,
            end_seconds=3.0,
            rank=1,
            matched_at_seconds=2.0,
            frame_id="f1",
        ),
        EvidenceHit(
            modality="visual",
            video_id="vid_1",
            video_filename="clip.mp4",
            video_path="/path/clip.mp4",
            score=0.72,
            raw_score=0.21,
            start_seconds=3.0,
            end_seconds=5.0,
            rank=2,
            matched_at_seconds=4.0,
            frame_id="f2",
        ),
    ]

    meta_map = {
        "vid_1": {
            "video_id": "vid_1",
            "file_name": "clip.mp4",
            "duration_seconds": 10.0,
            "frame_interval_seconds": 2.0,
        }
    }

    results = assemble_and_deduplicate_segments(
        visual_hits=v_hits,
        text_hits=[],
        video_meta_map=meta_map,
        final_min_score=0.20,
        link_tolerance_s=2.0,
    )

    # Must be deduplicated into 1 unified segment
    assert len(results) == 1
    res = results[0]
    # Retains the best score (0.88)
    assert res.final_score == 0.88
    # Spans from earliest (0.0) to latest (5.0)
    assert res.start_seconds == 0.0
    assert res.end_seconds == 5.0
    assert res.frame_id == "f1"


def test_retain_distinct_scenes_separated_in_time() -> None:
    """Distinct scenes from the same video separated in time must both be retained."""
    v_hits = [
        # Scene 1 at t=2.0s
        EvidenceHit(
            modality="visual",
            video_id="vid_1",
            video_filename="clip.mp4",
            video_path="/path/clip.mp4",
            score=0.85,
            raw_score=0.23,
            start_seconds=1.0,
            end_seconds=3.0,
            rank=1,
            matched_at_seconds=2.0,
            frame_id="f1",
        ),
        # Scene 2 at t=50.0s (well beyond 2s tolerance)
        EvidenceHit(
            modality="visual",
            video_id="vid_1",
            video_filename="clip.mp4",
            video_path="/path/clip.mp4",
            score=0.80,
            raw_score=0.22,
            start_seconds=49.0,
            end_seconds=51.0,
            rank=2,
            matched_at_seconds=50.0,
            frame_id="f2",
        ),
    ]

    meta_map = {
        "vid_1": {
            "video_id": "vid_1",
            "file_name": "clip.mp4",
            "duration_seconds": 60.0,
            "frame_interval_seconds": 2.0,
        }
    }

    results = assemble_and_deduplicate_segments(
        visual_hits=v_hits,
        text_hits=[],
        video_meta_map=meta_map,
        final_min_score=0.20,
        link_tolerance_s=2.0,
    )

    assert len(results) == 2
    assert results[0].start_seconds == 1.0
    assert results[1].start_seconds == 49.0


def test_never_join_across_different_video_ids() -> None:
    """Evidence from different videos must never be fused, even with identical timestamps."""
    v_hits = [
        EvidenceHit(
            modality="visual",
            video_id="vid_alpha",
            video_filename="a.mp4",
            video_path="/path/a.mp4",
            score=0.85,
            raw_score=0.23,
            start_seconds=2.0,
            end_seconds=4.0,
            rank=1,
            matched_at_seconds=3.0,
        )
    ]
    t_hits = [
        EvidenceHit(
            modality="text",
            video_id="vid_beta",  # Different video ID
            video_filename="b.mp4",
            video_path="/path/b.mp4",
            score=0.90,
            raw_score=0.60,
            start_seconds=2.0,
            end_seconds=4.0,
            rank=1,
            segment_id=0,
            transcript_text="Topic in beta",
        )
    ]

    meta_map = {
        "vid_alpha": {"video_id": "vid_alpha", "file_name": "a.mp4", "duration_seconds": 10.0},
        "vid_beta": {"video_id": "vid_beta", "file_name": "b.mp4", "duration_seconds": 10.0},
    }

    results = assemble_and_deduplicate_segments(
        visual_hits=v_hits,
        text_hits=t_hits,
        video_meta_map=meta_map,
        final_min_score=0.20,
    )

    assert len(results) == 2
    # Verify separate videos
    res_vids = {r.video_id for r in results}
    assert res_vids == {"vid_alpha", "vid_beta"}


def test_duplicate_filename_disambiguation() -> None:
    """When two videos share the same filename, a short ID suffix must be appended."""
    v_hits = [
        EvidenceHit(
            modality="visual",
            video_id="vid_111122223333",
            video_filename="same_name.mp4",
            video_path="/a/same_name.mp4",
            score=0.85,
            raw_score=0.23,
            start_seconds=1.0,
            end_seconds=3.0,
            rank=1,
        ),
        EvidenceHit(
            modality="visual",
            video_id="vid_999988887777",
            video_filename="same_name.mp4",
            video_path="/b/same_name.mp4",
            score=0.80,
            raw_score=0.22,
            start_seconds=1.0,
            end_seconds=3.0,
            rank=2,
        ),
    ]

    meta_map = {
        "vid_111122223333": {"video_id": "vid_111122223333", "file_name": "same_name.mp4"},
        "vid_999988887777": {"video_id": "vid_999988887777", "file_name": "same_name.mp4"},
    }

    results = assemble_and_deduplicate_segments(
        visual_hits=v_hits,
        text_hits=[],
        video_meta_map=meta_map,
        final_min_score=0.20,
    )

    assert len(results) == 2
    assert "vid_11112222" in results[0].video_filename
    assert "vid_99998888" in results[1].video_filename


def test_result_json_serializability() -> None:
    """Verify Result.to_dict() produces strictly valid JSON without numpy or Path objects."""
    res = Result(
        video_id="vid_test",
        video_filename="test.mp4",
        start_seconds=1.234,
        end_seconds=4.567,
        start_formatted="00:00:01",
        end_formatted="00:00:04",
        span_type="approx_from_frame",
        final_score=0.8765,
        visual_score=0.8765,
        text_score=None,
        modalities=["visual"],
        reason="Visual frame matches query",
        seek_seconds=0.234,
        video_path="C:/videos/test.mp4",
        matched_at_seconds=2.0,
        frame_id="f_001",
        frame_path="C:/frames/f_001.jpg",
        rrf_score=0.01234,
        video_file_exists=True,
    )

    d = res.to_dict()
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert deserialized["video_id"] == "vid_test"
    assert isinstance(deserialized["start_seconds"], float)
    assert isinstance(deserialized["video_file_exists"], bool)
    assert "resolution_seconds" in deserialized


def test_dedup_distant_frames_not_merged_and_inputs_not_mutated() -> None:
    """Regression test for R6: hits at 2/6/10/14s must NOT chain-merge into a 1-15s union."""
    v_hits = [
        EvidenceHit(
            modality="visual",
            video_id="vid_test",
            video_filename="test.mp4",
            video_path="/path/test.mp4",
            score=0.80,
            raw_score=0.22,
            start_seconds=1.0,
            end_seconds=3.0,
            rank=1,
            frame_id="f1",
            frame_index=1,
            matched_at_seconds=2.0,
        ),
        EvidenceHit(
            modality="visual",
            video_id="vid_test",
            video_filename="test.mp4",
            video_path="/path/test.mp4",
            score=0.75,
            raw_score=0.21,
            start_seconds=5.0,
            end_seconds=7.0,
            rank=2,
            frame_id="f3",
            frame_index=3,
            matched_at_seconds=6.0,
        ),
        EvidenceHit(
            modality="visual",
            video_id="vid_test",
            video_filename="test.mp4",
            video_path="/path/test.mp4",
            score=0.70,
            raw_score=0.20,
            start_seconds=9.0,
            end_seconds=11.0,
            rank=3,
            frame_id="f5",
            frame_index=5,
            matched_at_seconds=10.0,
        ),
        EvidenceHit(
            modality="visual",
            video_id="vid_test",
            video_filename="test.mp4",
            video_path="/path/test.mp4",
            score=0.65,
            raw_score=0.19,
            start_seconds=13.0,
            end_seconds=15.0,
            rank=4,
            frame_id="f7",
            frame_index=7,
            matched_at_seconds=14.0,
        ),
    ]

    meta_map = {
        "vid_test": {
            "video_id": "vid_test",
            "file_name": "test.mp4",
            "duration_seconds": 30.0,
            "frame_interval_seconds": 2.0,
        }
    }

    # Record initial spans to ensure immutability
    original_spans = [(h.start_seconds, h.end_seconds) for h in v_hits]

    results = assemble_and_deduplicate_segments(
        visual_hits=v_hits,
        text_hits=[],
        video_meta_map=meta_map,
        final_min_score=0.20,
    )

    # Must produce 4 distinct results, NOT 1 giant merged result
    assert len(results) == 4, f"Expected 4 distinct results, got {len(results)}"
    spans = [(r.start_seconds, r.end_seconds) for r in results]
    assert spans == [(1.0, 3.0), (5.0, 7.0), (9.0, 11.0), (13.0, 15.0)]

    # Assert input hits were not mutated
    current_spans = [(h.start_seconds, h.end_seconds) for h in v_hits]
    assert current_spans == original_spans, "Input EvidenceHit objects were mutated!"


def test_raw_cosine_tie_break_before_rrf() -> None:
    """When calibrated scores are equal, candidate with higher raw cosine must rank first (Item 4)."""
    # Candidate A: calibrated 0.95, raw cosine 0.22
    # Candidate B: calibrated 0.95, raw cosine 0.18
    hit_a = EvidenceHit(
        modality="visual",
        video_id="vid_1",
        video_filename="clip.mp4",
        video_path="/path/clip.mp4",
        score=0.95,
        raw_score=0.22,
        start_seconds=0.0,
        end_seconds=2.0,
        rank=2,  # RRF rank 2
        matched_at_seconds=1.0,
    )
    hit_b = EvidenceHit(
        modality="visual",
        video_id="vid_1",
        video_filename="clip.mp4",
        video_path="/path/clip.mp4",
        score=0.95,
        raw_score=0.18,
        start_seconds=10.0,
        end_seconds=12.0,
        rank=1,  # RRF rank 1
        matched_at_seconds=11.0,
    )
    meta = {"vid_1": {"video_id": "vid_1", "file_name": "clip.mp4", "duration_seconds": 60.0}}

    results = assemble_and_deduplicate_segments(
        visual_hits=[hit_a, hit_b],
        text_hits=[],
        video_meta_map=meta,
        use_rrf=True,
    )
    assert len(results) == 2
    # Candidate A has higher raw cosine (0.22 > 0.18), so it must break the tie and rank #1
    assert results[0].start_seconds == 0.0
    assert results[0].visual_raw_cosine == 0.22
    assert results[1].start_seconds == 10.0
    assert results[1].visual_raw_cosine == 0.18


def test_link_tolerance_s_behavior() -> None:
    """Verify link_tolerance_s connects visual and text hits when distance is within tolerance (Item 5)."""
    # Visual hit at [4.0, 6.0]
    v_hit = EvidenceHit(
        modality="visual",
        video_id="vid_1",
        video_filename="clip.mp4",
        video_path="/path/clip.mp4",
        score=0.85,
        raw_score=0.20,
        start_seconds=4.0,
        end_seconds=6.0,
        rank=1,
        matched_at_seconds=5.0,
    )
    # Text hit at [8.0, 10.0] -> distance is 2.0s from visual window end
    t_hit = EvidenceHit(
        modality="text",
        video_id="vid_1",
        video_filename="clip.mp4",
        video_path="/path/clip.mp4",
        score=0.75,
        raw_score=0.40,
        start_seconds=8.0,
        end_seconds=10.0,
        rank=1,
        transcript_text="Test transcript",
    )
    meta = {"vid_1": {"video_id": "vid_1", "file_name": "clip.mp4", "duration_seconds": 60.0, "frame_interval_seconds": 2.0}}

    # With link_tolerance_s = 1.0 (distance 2.0s > 1.0s) -> NOT linked, 2 separate results
    results_unlinked = assemble_and_deduplicate_segments(
        visual_hits=[v_hit],
        text_hits=[t_hit],
        video_meta_map=meta,
        link_tolerance_s=1.0,
    )
    assert len(results_unlinked) == 2

    # With link_tolerance_s = 3.0 (distance 2.0s <= 3.0s) -> LINKED into 1 combined result
    results_linked = assemble_and_deduplicate_segments(
        visual_hits=[v_hit],
        text_hits=[t_hit],
        video_meta_map=meta,
        link_tolerance_s=3.0,
    )
    assert len(results_linked) == 1
    assert results_linked[0].span_type == "combined"
    assert "visual" in results_linked[0].modalities
    assert "text" in results_linked[0].modalities


