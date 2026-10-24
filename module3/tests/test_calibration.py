"""Unit tests for SigLIP and MiniLM score calibration and gating."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from retrieval_engine.calibration import (
    calibrate_text_score,
    calibrate_visual_score,
    extract_siglip_calibration_parameters,
    is_text_relevant,
    is_visual_relevant,
    sigmoid,
)


def test_sigmoid_numerical_stability() -> None:
    """Verify sigmoid behaves properly across extreme values."""
    assert sigmoid(0.0) == pytest.approx(0.5, rel=1e-4)
    assert sigmoid(100.0) == pytest.approx(1.0, rel=1e-4)
    assert sigmoid(-100.0) == pytest.approx(0.0, rel=1e-4)


def test_calibrate_visual_score_siglip_formulation() -> None:
    """Verify SigLIP native sigmoid calibration computes exp(scale)*cos + bias."""
    logit_scale = 4.724  # exp(4.724) ~ 112.6
    logit_bias = -16.77

    # High match cosine ~ 0.22 -> logit = 112.6 * 0.22 - 16.77 = ~8.0 -> prob ~ 0.999
    p_high = calibrate_visual_score(0.22, logit_scale=logit_scale, logit_bias=logit_bias)
    assert 0.95 <= p_high <= 1.0

    # Intermediate cosine ~ 0.15 -> logit = 112.6 * 0.15 - 16.77 = ~0.12 -> prob ~ 0.53
    p_mid = calibrate_visual_score(0.15, logit_scale=logit_scale, logit_bias=logit_bias)
    assert 0.40 <= p_mid <= 0.65

    # Low match cosine ~ 0.08 -> logit = 112.6 * 0.08 - 16.77 = ~ -7.7 -> prob ~ 0.0004
    p_low = calibrate_visual_score(0.08, logit_scale=logit_scale, logit_bias=logit_bias)
    assert 0.0 <= p_low <= 0.01

    # Monotonicity check
    assert p_high > p_mid > p_low


def test_calibrate_visual_score_fallback() -> None:
    """Verify fallback logistic calibration applies when parameters are absent."""
    p_high = calibrate_visual_score(0.35, logit_scale=None, logit_bias=None, fallback_scale=10.0, fallback_bias=-2.0)
    p_mid = calibrate_visual_score(0.20, logit_scale=None, logit_bias=None, fallback_scale=10.0, fallback_bias=-2.0)
    p_low = calibrate_visual_score(0.05, logit_scale=None, logit_bias=None, fallback_scale=10.0, fallback_bias=-2.0)

    assert 0.0 <= p_low <= p_mid <= p_high <= 1.0
    assert p_high > p_mid > p_low


def test_calibrate_text_score_monotonic_linear() -> None:
    """Verify MiniLM text score calibration clamps below floor and above ceiling."""
    floor = 0.20
    ceiling = 0.80

    # Below floor -> 0.0
    assert calibrate_text_score(0.15, floor=floor, ceiling=ceiling) == 0.0
    assert calibrate_text_score(0.00, floor=floor, ceiling=ceiling) == 0.0

    # Exactly at floor -> 0.0
    assert calibrate_text_score(0.20, floor=floor, ceiling=ceiling) == 0.0

    # At midpoint (0.50) -> (0.50 - 0.20)/(0.80 - 0.20) = 0.30/0.60 = 0.50
    assert calibrate_text_score(0.50, floor=floor, ceiling=ceiling) == pytest.approx(0.50, rel=1e-4)

    # Exactly at ceiling -> 1.0
    assert calibrate_text_score(0.80, floor=floor, ceiling=ceiling) == 1.0

    # Above ceiling -> 1.0
    assert calibrate_text_score(0.95, floor=floor, ceiling=ceiling) == 1.0


def test_relevance_gating_functions() -> None:
    """Verify threshold evaluation functions."""
    assert is_visual_relevant(0.25, min_prob=0.20) is True
    assert is_visual_relevant(0.15, min_prob=0.20) is False

    assert is_text_relevant(0.35, min_cosine=0.25) is True
    assert is_text_relevant(0.20, min_cosine=0.25) is False


def test_extract_siglip_calibration_parameters_mock() -> None:
    """Verify parameter extractor handles mock and missing attributes gracefully."""
    class DummyModel:
        def __init__(self) -> None:
            self.logit_scale = 4.5
            self.logit_bias = -15.0

    class DummyEmbedder:
        def __init__(self) -> None:
            self._model = DummyModel()

    embedder = DummyEmbedder()
    res = extract_siglip_calibration_parameters(embedder)
    assert res is not None
    scale, bias = res
    assert scale == 4.5
    assert bias == -15.0

    # Empty embedder returns None
    assert extract_siglip_calibration_parameters(None) is None
