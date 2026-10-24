"""Unit tests for multimodal fusion and Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from retrieval_engine.fusion import compute_fused_score, compute_rrf_score


def test_missing_modality_is_not_treated_as_zero() -> None:
    """Critical requirement: missing modality is missing evidence, not score=0."""
    # Visual only
    score_v, reason_v = compute_fused_score(
        visual_score=0.75,
        text_score=None,
        visual_weight=0.6,
        text_weight=0.4,
    )
    assert score_v == 0.75
    assert "Visual match" in reason_v

    # Text only
    score_t, reason_t = compute_fused_score(
        visual_score=None,
        text_score=0.82,
        visual_weight=0.6,
        text_weight=0.4,
    )
    assert score_t == 0.82
    assert "Speech transcript match" in reason_t


def test_dual_modality_fusion_noisy_or() -> None:
    """Verify noisy-OR cross-modal fusion: fused = 1 - (1 - v) * (1 - t)."""
    # visual=0.60, text=0.80 -> 1 - (0.40 * 0.20) = 0.92
    fused, reason = compute_fused_score(
        visual_score=0.60,
        text_score=0.80,
        fusion_mode="noisy_or",
    )
    assert fused == pytest.approx(0.92, rel=1e-4)
    assert "Cross-modal match" in reason
    assert "noisy-OR fused" in reason


def test_dual_modality_fusion_weighted_mode() -> None:
    """Verify weighted cross-modal fusion with configurable weights and agreement bonus (Item 3)."""
    # (0.6 * 0.60 + 0.4 * 0.80) / 1.0 + 0.05 = 0.36 + 0.32 + 0.05 = 0.73
    fused, reason = compute_fused_score(
        visual_score=0.60,
        text_score=0.80,
        visual_weight=0.6,
        text_weight=0.4,
        agreement_bonus=0.05,
        fusion_mode="weighted",
    )
    assert fused == pytest.approx(0.73, rel=1e-4)
    assert "weighted fused: 0.73" in reason
    assert "bonus=0.05" in reason

    # Changing weights changes the score
    fused_vis_heavy, _ = compute_fused_score(
        visual_score=0.60,
        text_score=0.80,
        visual_weight=0.9,
        text_weight=0.1,
        agreement_bonus=0.05,
        fusion_mode="weighted",
    )
    # (0.9*0.6 + 0.1*0.8)/1.0 + 0.05 = 0.54 + 0.08 + 0.05 = 0.67
    assert fused_vis_heavy == pytest.approx(0.67, rel=1e-4)
    assert fused_vis_heavy != fused

    # Single modality keeps its own calibrated score (not diluted by missing modality)
    fused_single_v, _ = compute_fused_score(
        visual_score=0.75,
        text_score=None,
        visual_weight=0.9,
        text_weight=0.1,
        fusion_mode="weighted",
    )
    assert fused_single_v == 0.75

    fused_single_t, _ = compute_fused_score(
        visual_score=None,
        text_score=0.85,
        visual_weight=0.9,
        text_weight=0.1,
        fusion_mode="weighted",
    )
    assert fused_single_t == 0.85


def test_fused_score_capped_at_one() -> None:
    """Verify fused score never exceeds 1.0 in either mode."""
    fused_w, _ = compute_fused_score(
        visual_score=0.98,
        text_score=0.99,
        agreement_bonus=0.10,
        fusion_mode="weighted",
    )
    assert fused_w <= 1.0
    assert fused_w == 1.0

    fused_nor, _ = compute_fused_score(
        visual_score=0.98,
        text_score=0.99,
        fusion_mode="noisy_or",
    )
    assert fused_nor <= 1.0


def test_fusion_invalid_mode_raises() -> None:
    """Verify unknown fusion mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown fusion_mode"):
        compute_fused_score(0.5, 0.5, fusion_mode="invalid_mode")



def test_compute_rrf_score() -> None:
    """Verify RRF rank calculation rewards top-ranked items."""
    # Rank 1 in both modalities
    rrf_top = compute_rrf_score(visual_rank=1, text_rank=1, rrf_k=60, visual_weight=0.6, text_weight=0.4)
    expected_top = 0.6 * (1.0 / 61.0) + 0.4 * (1.0 / 61.0)
    assert rrf_top == pytest.approx(expected_top, rel=1e-4)

    # Rank 10 in both modalities
    rrf_lower = compute_rrf_score(visual_rank=10, text_rank=10, rrf_k=60, visual_weight=0.6, text_weight=0.4)
    expected_lower = 0.6 * (1.0 / 70.0) + 0.4 * (1.0 / 70.0)
    assert rrf_lower == pytest.approx(expected_lower, rel=1e-4)

    assert rrf_top > rrf_lower

    # Missing one modality in RRF
    rrf_vis_only = compute_rrf_score(visual_rank=1, text_rank=None, rrf_k=60, visual_weight=0.6, text_weight=0.4)
    assert rrf_vis_only == pytest.approx(0.6 * (1.0 / 61.0), rel=1e-4)
