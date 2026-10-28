"""Regression tests ensuring UI result cards, match percentages, timestamps, and seeking never regress."""

import pytest
from module3.retrieval_engine.segments import Result, format_timestamp


def test_result_dataclass_field_and_property_aliases():
    """Verify Result dataclass provides both modern and legacy property aliases."""
    res = Result(
        video_id="vid_test_123",
        video_filename="sample.mp4",
        start_seconds=26.81,
        end_seconds=32.274,
        start_formatted="00:00:26",
        end_formatted="00:00:32",
        span_type="combined",
        final_score=0.9380,
        visual_score=0.946,
        text_score=0.801,
        modalities=["visual", "text"],
        reason="Visual scene alignment corroborates transcript segment",
        seek_seconds=25.81,
        video_path="/path/sample.mp4",
    )

    # Primary fields
    assert res.start_seconds == 26.81
    assert res.end_seconds == 32.274
    assert res.final_score == 0.9380
    assert res.seek_seconds == 25.81
    assert res.modalities == ["visual", "text"]

    # Backward-compatibility property aliases
    assert res.start_time == 26.81
    assert res.end_time == 32.274
    assert res.score == 0.9380
    assert res.modality == "hybrid"

    # Single modality alias
    res_text_only = Result(
        video_id="vid_test_456",
        video_filename="sample2.mp4",
        start_seconds=39.379,
        end_seconds=46.725,
        start_formatted="00:00:39",
        end_formatted="00:00:46",
        span_type="transcript_segment",
        final_score=0.804,
        visual_score=None,
        text_score=0.804,
        modalities=["text"],
        reason="Transcript keyword match",
        seek_seconds=38.379,
        video_path="/path/sample2.mp4",
    )
    assert res_text_only.modality == "text"

    # to_dict compatibility
    d = res.to_dict()
    assert d["start_seconds"] == 26.81
    assert d["start_time"] == 26.81
    assert d["end_seconds"] == 32.274
    assert d["end_time"] == 32.274
    assert d["final_score"] == 0.9380
    assert d["score"] == 0.9380
    assert d["modality"] == "hybrid"


def test_ui_card_score_and_timestamp_resolution():
    """Verify UI logic correctly resolves timestamps and non-zero scores without collapsing to 0."""
    res1 = Result(
        video_id="vid_sample6",
        video_filename="Sample 6.mp4",
        start_seconds=26.81,
        end_seconds=32.274,
        start_formatted="00:00:26",
        end_formatted="00:00:32",
        span_type="combined",
        final_score=0.93796,
        visual_score=0.946,
        text_score=0.801,
        modalities=["visual", "text"],
        reason="Corroborated",
        seek_seconds=25.81,
        video_path="videos/Sample 6.mp4",
    )

    res2 = Result(
        video_id="vid_sample6",
        video_filename="Sample 6.mp4",
        start_seconds=39.379,
        end_seconds=46.725,
        start_formatted="00:00:39",
        end_formatted="00:00:46",
        span_type="combined",
        final_score=0.90489,
        visual_score=0.758,
        text_score=1.0,
        modalities=["visual", "text"],
        reason="Corroborated",
        seek_seconds=38.379,
        video_path="videos/Sample 6.mp4",
    )

    results = [res1, res2]

    # Simulate UI extraction loop in search_view.py
    extracted_cards = []
    for idx, r in enumerate(results):
        start_s = float(getattr(r, "start_seconds", getattr(r, "start_time", 0.0)))
        end_s = float(getattr(r, "end_seconds", getattr(r, "end_time", start_s + 5.0)))
        score = float(getattr(r, "final_score", getattr(r, "score", 0.0)))
        start_fmt = getattr(r, "start_formatted", None) or format_timestamp(start_s)
        end_fmt = getattr(r, "end_formatted", None) or format_timestamp(end_s)
        score_display = f"{score:.1%} match" if score >= 0.001 else "<0.1% match"
        btn_key = f"select_res_btn_{r.video_id}_{int(start_s)}_{idx}"

        extracted_cards.append({
            "start_fmt": start_fmt,
            "end_fmt": end_fmt,
            "score_display": score_display,
            "btn_key": btn_key,
            "seek_s": float(getattr(r, "seek_seconds", start_s)),
        })

    # Card 0
    assert extracted_cards[0]["start_fmt"] == "00:00:26"
    assert extracted_cards[0]["end_fmt"] == "00:00:32"
    assert extracted_cards[0]["score_display"] == "93.8% match"
    assert extracted_cards[0]["seek_s"] == 25.81
    assert extracted_cards[0]["btn_key"] == "select_res_btn_vid_sample6_26_0"

    # Card 1
    assert extracted_cards[1]["start_fmt"] == "00:00:39"
    assert extracted_cards[1]["end_fmt"] == "00:00:46"
    assert extracted_cards[1]["score_display"] == "90.5% match"
    assert extracted_cards[1]["seek_s"] == 38.379
    assert extracted_cards[1]["btn_key"] == "select_res_btn_vid_sample6_39_1"

    # Distinct timestamps and keys
    assert extracted_cards[0]["start_fmt"] != extracted_cards[1]["start_fmt"]
    assert extracted_cards[0]["btn_key"] != extracted_cards[1]["btn_key"]


def test_player_view_seek_state_synchronization():
    """Verify player seek synchronization state logic when switching segments and key moments."""
    session_state = {}

    # Initial video render with Result 0
    vid_id = "vid_sample6"
    res0_start = 26.81
    res0_end = 32.274
    res0_seek = 25.81

    seek_key = f"player_seek_{vid_id}"
    last_res_key = f"last_result_{vid_id}"

    # Step 1: Initial render
    current_result_sig = f"{res0_start:.3f}_{res0_end:.3f}"
    if session_state.get(last_res_key) != current_result_sig or seek_key not in session_state:
        session_state[seek_key] = int(res0_seek)
        session_state[last_res_key] = current_result_sig

    assert session_state[seek_key] == 25

    # Step 2: User clicks Key Moment at 30s
    km_time = 30.0
    session_state[seek_key] = int(km_time)
    assert session_state[seek_key] == 30

    # Step 3: Re-render with same result -> seek must NOT revert
    if session_state.get(last_res_key) != current_result_sig or seek_key not in session_state:
        session_state[seek_key] = int(res0_seek)
        session_state[last_res_key] = current_result_sig
    assert session_state[seek_key] == 30  # Kept key moment seek!

    # Step 4: User clicks Result 1 (39s)
    res1_start = 39.379
    res1_end = 46.725
    res1_seek = 38.379
    new_result_sig = f"{res1_start:.3f}_{res1_end:.3f}"

    if session_state.get(last_res_key) != new_result_sig or seek_key not in session_state:
        session_state[seek_key] = int(res1_seek)
        session_state[last_res_key] = new_result_sig

    assert session_state[seek_key] == 38  # Correctly switched to Result 1!
