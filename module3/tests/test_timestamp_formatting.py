"""Unit tests for timestamp formatting and approximate visual span construction."""

from __future__ import annotations

import pytest

from retrieval_engine.segments import (
    build_approx_visual_span,
    format_timestamp,
    intervals_overlap_or_adjacent,
)


def test_format_timestamp_short_durations() -> None:
    """Verify seconds format correctly for values under one hour."""
    assert format_timestamp(0.0) == "00:00:00"
    assert format_timestamp(0.999) == "00:00:00"
    assert format_timestamp(1.0) == "00:00:01"
    assert format_timestamp(59.0) == "00:00:59"
    # Crucial rule: no premature round-up that shifts the second forward
    assert format_timestamp(59.999) == "00:00:59"
    assert format_timestamp(60.0) == "00:01:00"
    assert format_timestamp(61.5) == "00:01:01"
    assert format_timestamp(3599.9) == "00:59:59"


def test_format_timestamp_long_durations_over_one_hour() -> None:
    """Verify hours formatting handles >= 1 hour without capping at 23."""
    assert format_timestamp(3600.0) == "01:00:00"
    assert format_timestamp(3661.0) == "01:01:01"
    assert format_timestamp(7322.0) == "02:02:02"
    # Multi-day video streams (>= 24 hours, hours not capped at 23)
    assert format_timestamp(86400.0) == "24:00:00"
    assert format_timestamp(100000.0) == "27:46:40"
    assert format_timestamp(360000.0) == "100:00:00"


def test_build_approx_visual_span_boundaries() -> None:
    """Verify visual frame approximate spans are clamped to start and duration."""
    # Matched at t=0.0 -> clamped at 0.0
    s, e = build_approx_visual_span(matched_at=0.0, interval_s=2.0, duration_s=10.0)
    assert s == 0.0
    assert e == 1.0

    # Matched in middle t=4.0 -> [3.0, 5.0]
    s, e = build_approx_visual_span(matched_at=4.0, interval_s=2.0, duration_s=10.0)
    assert s == 3.0
    assert e == 5.0

    # Matched at end near duration t=9.5 -> [8.5, 10.0] clamped at duration
    s, e = build_approx_visual_span(matched_at=9.5, interval_s=2.0, duration_s=10.0)
    assert s == 8.5
    assert e == 10.0

    # Without known duration (duration_s=0.0)
    s, e = build_approx_visual_span(matched_at=5.0, interval_s=2.0, duration_s=0.0)
    assert s == 4.0
    assert e == 6.0


def test_intervals_overlap_or_adjacent() -> None:
    """Verify temporal adjacency test respects tolerance."""
    # Clearly overlapping
    assert intervals_overlap_or_adjacent(1.0, 3.0, 2.0, 4.0, tolerance_s=0.0) is True

    # Touching boundaries
    assert intervals_overlap_or_adjacent(1.0, 3.0, 3.0, 5.0, tolerance_s=0.0) is True

    # Gap of 1.0 with tolerance 2.0 -> True
    assert intervals_overlap_or_adjacent(1.0, 3.0, 4.0, 6.0, tolerance_s=2.0) is True

    # Gap of 2.0 with tolerance 1.0 -> False
    assert intervals_overlap_or_adjacent(1.0, 3.0, 5.0, 7.0, tolerance_s=1.0) is False
