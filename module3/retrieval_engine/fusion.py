"""Segment-level multimodal score fusion and Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def compute_fused_score(
    visual_score: Optional[float],
    text_score: Optional[float],
    visual_weight: float = 0.6,
    text_weight: float = 0.4,
    agreement_bonus: float = 0.05,
    fusion_mode: str = "weighted",
) -> Tuple[float, str]:
    """Compute combined calibrated score renormalized across available evidence.

    In 'weighted' mode (default):
        For cross-modal matches with both visual and text evidence:
            fused = (wv * v + wt * t) / (wv + wt) + agreement_bonus
        clipped to <= 1.0.
        Single-modality segments keep their own calibrated score unchanged
        (a missing modality is never treated as 0.0).

    In 'noisy_or' mode:
        fused = 1.0 - (1.0 - v) * (1.0 - t)
        Single-modality segments keep their own calibrated score unchanged.

    Args:
        visual_score: Calibrated visual probability in [0, 1] or None.
        text_score: Calibrated text score in [0, 1] or None.
        visual_weight: Weight assigned to visual modality (>= 0).
        text_weight: Weight assigned to text modality (>= 0).
        agreement_bonus: Additive agreement bonus for cross-modal matches.
        fusion_mode: "weighted" (default) or "noisy_or".

    Returns:
        (final_score, rationale_string)
    """
    if fusion_mode not in ("weighted", "noisy_or"):
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}. Expected 'weighted' or 'noisy_or'.")

    has_vis = visual_score is not None
    has_txt = text_score is not None

    if has_vis and has_txt:
        v = min(1.0, max(0.0, float(visual_score)))
        t = min(1.0, max(0.0, float(text_score)))

        if fusion_mode == "noisy_or":
            fused = 1.0 - (1.0 - v) * (1.0 - t)
            fused = min(1.0, max(0.0, float(fused)))
            reason = (
                f"Cross-modal match: visual ({visual_score:.2f}) and transcript ({text_score:.2f}) "
                f"corroborate this segment (noisy-OR fused: {fused:.2f})."
            )
            return float(fused), reason
        elif fusion_mode == "weighted":
            wv = float(visual_weight)
            wt = float(text_weight)
            total_w = wv + wt
            if total_w > 0:
                base = (wv * v + wt * t) / total_w
            else:
                base = (v + t) / 2.0
            bonus = max(0.0, float(agreement_bonus))
            fused = min(1.0, max(0.0, base + bonus))
            reason = (
                f"Cross-modal match: visual ({visual_score:.2f}) and transcript ({text_score:.2f}) "
                f"corroborate this segment (weighted fused: {fused:.2f}, bonus={bonus:.2f})."
            )
            return float(fused), reason

    elif has_vis and not has_txt:
        # Single visual modality: preserve calibrated score, missing text is not 0
        score = float(visual_score)
        reason = f"Visual match: frame features align with query ({score:.2f})."
        return score, reason

    elif has_txt and not has_vis:
        # Single text modality: preserve calibrated score, missing visual is not 0
        score = float(text_score)
        reason = f"Speech transcript match: dialogue semantics align with query ({score:.2f})."
        return score, reason

    return 0.0, "No evidence available."


def compute_rrf_score(
    visual_rank: Optional[int],
    text_rank: Optional[int],
    rrf_k: int = 60,
    visual_weight: float = 0.6,
    text_weight: float = 0.4,
) -> float:
    """Compute Reciprocal Rank Fusion score as a secondary ranking/tie-breaking signal.

    Args:
        visual_rank: 1-based rank in visual candidate pool, or None.
        text_rank: 1-based rank in text candidate pool, or None.
        rrf_k: Smoothing constant (default 60).
        visual_weight: Weight for visual modality rank.
        text_weight: Weight for text modality rank.

    Returns:
        RRF score float.
    """
    score = 0.0
    if visual_rank is not None and visual_rank > 0:
        score += visual_weight * (1.0 / (rrf_k + visual_rank))
    if text_rank is not None and text_rank > 0:
        score += text_weight * (1.0 / (rrf_k + text_rank))
    return float(score)
