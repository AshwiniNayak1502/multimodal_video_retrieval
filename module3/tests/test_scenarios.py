"""Comprehensive verification test suite implementing Scenarios A through K."""

from __future__ import annotations

import concurrent.futures
from typing import Any, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from retrieval_engine.config import Module3Config
from retrieval_engine.query_service import QueryService


def test_scenario_a_visual_only_silent_video(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario A: Visual-only query on a silent video retrieves the correct frame."""
    service, _, vis_emb, _ = indexed_test_environment

    # Retrieve frame 1 from silent video (vid_silent002 at t=2.0s)
    silent_frames = service.index.get_frames_in_window("vid_silent002", 1.0, 3.0)
    assert len(silent_frames) >= 1
    target_frame_path = silent_frames[0]["frame_path"]

    # In mocked environment, wire the mock embedder query vector to match the frame vector
    with Image.open(target_frame_path) as img:
        frame_vec = vis_emb.embed_images([img])[0]
    vis_emb.embed_query_text = lambda text: frame_vec

    # Search visual with query vector
    resp = service.search(query="silent clip visual query", top_k=5, video_ids=["vid_silent002"])
    assert resp.status == "ok"
    assert len(resp.results) > 0
    top = resp.results[0]
    assert top.video_id == "vid_silent002"
    assert "visual" in top.modalities
    assert "text" not in top.modalities
    assert top.span_type == "approx_from_frame"
    assert top.matched_at_seconds == pytest.approx(2.0, abs=0.1)


def test_scenario_b_transcript_paraphrase_query(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario B: Transcript paraphrase query retrieves the right segment with correct timestamps."""
    service, _, _, txt_emb = indexed_test_environment

    # vid_test001 segment 1 is "This module retrieves timestamped video segments." (start=4.2s, end=6.8s)
    seg_vec = txt_emb.embed_texts(["This module retrieves timestamped video segments."])[0]
    txt_emb.embed_query_text = lambda text: seg_vec

    resp = service.search(
        query="retrieves timestamped video segments",
        top_k=5,
        video_ids=["vid_test001"],
    )
    assert resp.status == "ok"
    assert len(resp.results) > 0
    matching_segs = [
        r for r in resp.results
        if r.video_id == "vid_test001" and (r.start_seconds <= 5.0 <= r.end_seconds)
    ]
    assert len(matching_segs) > 0
    seg = matching_segs[0]
    assert "text" in seg.modalities
    assert seg.start_seconds == pytest.approx(4.2, abs=0.1)
    assert seg.end_seconds == pytest.approx(6.8, abs=0.1)


def test_scenario_c_two_videos_one_relevant_one_not(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario C: Two videos, one relevant / one not -> relevant ranks first, unrelated not in results (Item 11a)."""
    service, cfg, vis_emb, txt_emb = indexed_test_environment

    # Wire txt_emb to match vid_test001 speech segment
    seg_text = "Welcome to multimodal semantic query processing."
    rel_vec = txt_emb.embed_texts([seg_text])[0]
    txt_emb.embed_query_text = lambda text: rel_vec

    # Set vis_emb to zero vector so silent clip visual frames do not falsely trigger
    vis_emb.embed_query_text = lambda text: np.zeros(768, dtype=np.float32)

    cfg.final_min_score = 0.20
    resp = service.search(query="multimodal semantic query processing", top_k=5)

    # Assert unconditionally
    assert resp.status == "ok"
    assert len(resp.results) > 0
    vids = [r.video_id for r in resp.results]
    assert vids[0] == "vid_test001"
    assert "vid_silent002" not in vids


def test_scenario_d_relevant_segments_in_multiple_videos(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario D: Relevant segments in multiple videos each returned with its own correct video_id/timestamps (Item 11a)."""
    service, cfg, vis_emb, txt_emb = indexed_test_environment

    # vid_test001 text segment 0
    seg_vec = txt_emb.embed_texts(["Welcome to multimodal semantic query processing."])[0]
    txt_emb.embed_query_text = lambda text: seg_vec

    # vid_silent002 visual frame 0
    silent_frames = service.index.get_frames_in_window("vid_silent002", 0.0, 1.0)
    with Image.open(silent_frames[0]["frame_path"]) as img:
        frame_vec = vis_emb.embed_images([img])[0]
    vis_emb.embed_query_text = lambda text: frame_vec

    cfg.final_min_score = 0.05
    resp = service.search(query="video test", top_k=5)

    # Assert unconditionally
    assert resp.status == "ok"
    assert len(resp.results) >= 2
    res_vids = {r.video_id for r in resp.results}
    assert "vid_test001" in res_vids
    assert "vid_silent002" in res_vids
    for r in resp.results:
        assert r.start_seconds <= r.end_seconds
        assert r.video_id.startswith("vid_")


def test_scenario_e_query_with_no_related_content(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario E: Query with no related content -> status='no_match' via per-modality gate rejection (Item 11b)."""
    service, cfg, vis_emb, txt_emb = indexed_test_environment

    # Query vectors that yield zero cosine similarities
    vis_emb.embed_query_text = lambda text: np.zeros(768, dtype=np.float32)
    txt_emb.embed_query_text = lambda text: np.zeros(384, dtype=np.float32)

    # Keep standard gating threshold (do NOT set artificial 0.999)
    cfg.final_min_score = 0.0
    resp = service.search(query="extraterrestrial alien spacecraft landing on mars volcano", top_k=5)

    assert resp.status == "no_match"
    assert resp.message == "No related video segments found for your query."
    assert len(resp.results) == 0
    assert len(resp.debug["best_rejected_candidates"]) > 0


def test_scenario_f_timestamp_correctness_known_event(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario F: Known event in video -> returned video_id and time within one frame interval."""
    service, _, vis_emb, _ = indexed_test_environment

    # Event in vid_test001 at t=2.0s
    frames = service.index.get_frames_in_window("vid_test001", 1.5, 2.5)
    assert len(frames) == 1
    event_frame = frames[0]
    expected_ts = event_frame["timestamp_seconds"]  # 2.0s
    video_meta = service.index.get_video("vid_test001") or {}
    interval = float(video_meta.get("frame_interval_seconds", 2.0))

    with Image.open(event_frame["frame_path"]) as img:
        event_vec = vis_emb.embed_images([img])[0]
    vis_emb.embed_query_text = lambda q: event_vec

    resp = service.search(query="frame event", top_k=5, video_ids=["vid_test001"])
    assert resp.status == "ok"
    matched = [r for r in resp.results if r.matched_at_seconds is not None]
    assert len(matched) > 0
    first_match = matched[0]
    assert abs(first_match.matched_at_seconds - expected_ts) <= interval


def test_scenario_g_consecutive_near_identical_frames_deduped(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario G: Many consecutive near-identical frames -> deduped; distinct scenes retained."""
    from retrieval_engine.segments import EvidenceHit, assemble_and_deduplicate_segments

    hits = [
        EvidenceHit("visual", "vid_1", "clip.mp4", "/path", 0.80, 0.2, 0.0, 1.0, 1, matched_at_seconds=0.0),
        EvidenceHit("visual", "vid_1", "clip.mp4", "/path", 0.85, 0.2, 1.0, 3.0, 2, matched_at_seconds=2.0),
        EvidenceHit("visual", "vid_1", "clip.mp4", "/path", 0.82, 0.2, 3.0, 5.0, 3, matched_at_seconds=4.0),
        # Scene B at 30s
        EvidenceHit("visual", "vid_1", "clip.mp4", "/path", 0.78, 0.2, 29.0, 31.0, 4, matched_at_seconds=30.0),
        EvidenceHit("visual", "vid_1", "clip.mp4", "/path", 0.79, 0.2, 31.0, 33.0, 5, matched_at_seconds=32.0),
    ]

    meta = {"vid_1": {"video_id": "vid_1", "file_name": "clip.mp4", "duration_seconds": 60.0}}

    deduped = assemble_and_deduplicate_segments(
        visual_hits=hits,
        text_hits=[],
        video_meta_map=meta,
        final_min_score=0.2,
        link_tolerance_s=2.0,
    )

    # Exactly 2 distinct scenes should remain
    assert len(deduped) == 2
    # Scene 1 span covers 0.0s to 5.0s
    assert deduped[0].start_seconds == 0.0
    assert deduped[0].end_seconds == 5.0
    # Scene 2 span covers 29.0s to 33.0s
    assert deduped[1].start_seconds == 29.0
    assert deduped[1].end_seconds == 33.0


def test_scenario_h_persistence_without_reindexing(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario H: Close and reopen VideoIndex/QueryService and query without re-indexing."""
    service, cfg, vis_emb, txt_emb = indexed_test_environment

    vis_emb.embed_images = MagicMock(side_effect=RuntimeError("embed_images must not be called at query time!"))
    txt_emb.embed_texts = MagicMock(side_effect=RuntimeError("embed_texts must not be called at query time!"))

    service.reload_index()
    resp = service.search(query="persistence validation query", top_k=5)
    assert resp.status in ("ok", "no_match")
    vis_emb.embed_images.assert_not_called()
    txt_emb.embed_texts.assert_not_called()


def test_scenario_i_validation_errors_no_crash(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario I: Empty query, bad K (2, 3, 5, 6, 50, bool, str), unknown video_id -> clear errors, no crash (Item 2)."""
    service, _, _, _ = indexed_test_environment

    # Empty
    r1 = service.search(query="")
    assert r1.status == "error"

    # Bad K boundaries: 2 (rejected), 3 (accepted), 5 (accepted), 6 (rejected), 50 (rejected)
    r_k2 = service.search(query="valid", top_k=2)
    assert r_k2.status == "error"
    assert "between 3 and 5" in r_k2.message

    r_k3 = service.search(query="valid", top_k=3)
    assert r_k3.status in ("ok", "no_match")

    r_k5 = service.search(query="valid", top_k=5)
    assert r_k5.status in ("ok", "no_match")

    r_k6 = service.search(query="valid", top_k=6)
    assert r_k6.status == "error"
    assert "between 3 and 5" in r_k6.message

    r_k50 = service.search(query="valid", top_k=50)
    assert r_k50.status == "error"
    assert "between 3 and 5" in r_k50.message

    # Boolean top_k
    r_kbool = service.search(query="valid", top_k=True)  # type: ignore
    assert r_kbool.status == "error"
    assert "between 3 and 5" in r_kbool.message

    # String top_k
    r_kstr = service.search(query="valid", top_k="invalid")  # type: ignore
    assert r_kstr.status == "error"
    assert "between 3 and 5" in r_kstr.message

    # Unknown video_id
    r5 = service.search(query="valid", top_k=5, video_ids=["unknown_fake_id"])
    assert r5.status == "error"
    assert "unknown_fake_id" in r5.message


def test_scenario_j_missing_modality_graceful_degradation(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario J: Missing text modality / empty text index -> visual results returned and notice reported (Item 11a, Item 8)."""
    service, _, vis_emb, _ = indexed_test_environment

    silent_frames = service.index.get_frames_in_window("vid_silent002", 0.0, 1.0)
    with Image.open(silent_frames[0]["frame_path"]) as img:
        frame_vec = vis_emb.embed_images([img])[0]
    vis_emb.embed_query_text = lambda text: frame_vec

    resp = service.search(query="silent clip query", top_k=5, video_ids=["vid_silent002"])

    # Assert unconditionally
    assert resp.status == "ok"
    assert len(resp.results) > 0
    for r in resp.results:
        assert r.video_id == "vid_silent002"
        assert "visual" in r.modalities
        assert "text" not in r.modalities
    assert any("Transcript search unavailable" in n for n in resp.notices)


def test_scenario_k_repeated_queries_models_and_index_reused(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Scenario K: Repeated queries -> models/indexes reused and per-stage timings recorded."""
    service, _, _, _ = indexed_test_environment

    # Run query 1
    r1 = service.search(query="repeated query alpha", top_k=5)
    assert "query_encode_visual_ms" in r1.timings_ms
    assert "query_encode_text_ms" in r1.timings_ms
    assert "search_visual_ms" in r1.timings_ms
    assert "search_text_ms" in r1.timings_ms
    assert "calibrate_fuse_ms" in r1.timings_ms
    assert "segment_build_ms" in r1.timings_ms
    assert "total_ms" in r1.timings_ms

    # Run query 2 (different query)
    r2 = service.search(query="repeated query beta", top_k=5)
    assert r2.debug["models_reused"] is True
    assert r2.debug["index_reused"] is True

    # Run query 1 again (cache hit)
    r3 = service.search(query="repeated query alpha", top_k=5)
    assert r3.debug["cache_hit_visual"] is True
    assert r3.debug["cache_hit_text"] is True


def test_health_and_list_videos_thread_deadlock_regression(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Item 1: Call health() and list_videos() in worker threads with 5-second timeout to verify no deadlock."""
    service, _, _, _ = indexed_test_environment

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_health = executor.submit(service.health)
        h = f_health.result(timeout=5.0)
        assert h["status"] in ("ready", "degraded")

        f_videos = executor.submit(service.list_videos)
        v = f_videos.result(timeout=5.0)
        assert len(v) >= 2


@pytest.mark.real_model
@pytest.mark.parametrize(
    "query,expected_vid,expected_start,expected_end,expected_status",
    [
        (
            "What are scoped values in Java and how are they better than ThreadLocal?",
            "vid_e9f58d28adfac07e",
            0.0,
            6.0,
            "ok",
        ),
        (
            "a purple title card that says Java 25 Feature Highlight",
            "vid_e9f58d28adfac07e",
            0.0,
            5.0,
            "ok",
        ),
        (
            "slide with the heading Code Example",
            "vid_e9f58d28adfac07e",
            14.0,
            26.0,
            "ok",
        ),
        (
            "Welcome to the Multimodal Video Retrieval Project",
            "vid_d3147dcb7284cc89",
            0.0,
            3.0,
            "ok",
        ),
        (
            "recipe for baking sourdough bread",
            None,
            None,
            None,
            "no_match",
        ),
        (
            "underwater scuba diving with dolphins",
            None,
            None,
            None,
            "no_match",
        ),
    ],
)
def test_real_model_indexed_database_search(
    query: str,
    expected_vid: Optional[str],
    expected_start: Optional[float],
    expected_end: Optional[float],
    expected_status: str,
) -> None:
    """Real-model retrieval test parametrised with positive and negative queries (Item 11c)."""
    cfg = Module3Config()
    db_file = cfg.vector_db_dir / "metadata.db"
    if not db_file.is_file():
        pytest.skip("Module 2 vector database does not exist on disk.")

    try:
        service = QueryService(config=cfg, warmup=True)
        health = service.health()
        if health["status"] not in ("ready", "degraded"):
            pytest.skip("Module 2 index is not ready with indexed vectors.")
    except Exception as exc:
        pytest.skip(f"Offline model cache or vector index unavailable: {exc}")

    resp = service.search(query=query, top_k=5)
    assert resp.status == expected_status

    if expected_status == "ok":
        assert len(resp.results) > 0
        top = resp.results[0]
        if expected_vid is not None:
            assert top.video_id == expected_vid
        if expected_start is not None and expected_end is not None:
            assert top.start_seconds <= expected_end + 3.0
            assert top.end_seconds >= max(0.0, expected_start - 3.0)
    else:
        assert len(resp.results) == 0
        assert resp.message == "No related video segments found for your query."
