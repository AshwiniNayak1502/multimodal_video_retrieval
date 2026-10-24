"""Score calibration and relevance gating for visual and text modalities."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import numpy as np

from .logging_config import get_logger

logger = get_logger("calibration")


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute numerically stable sigmoid."""
    clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def extract_siglip_calibration_parameters(embedder: Any) -> Optional[Tuple[float, float]]:
    """Extract logit_scale and logit_bias scalars from loaded SigLIP model.

    Returns:
        (logit_scale, logit_bias) as floats if accessible, or None.
    """
    if embedder is None:
        return None

    # Check for direct attribute accessor or underlying _model
    model = getattr(embedder, "_model", None) or getattr(embedder, "model", None)
    if model is None:
        # Check if embedder itself exposes logit_scale/logit_bias
        model = embedder

    try:
        scale_param = getattr(model, "logit_scale", None)
        bias_param = getattr(model, "logit_bias", None)

        if scale_param is not None and bias_param is not None:
            # Handle PyTorch tensor / Parameter or float
            if hasattr(scale_param, "item"):
                scale_val = float(scale_param.detach().cpu().item())
            else:
                scale_val = float(scale_param)

            if hasattr(bias_param, "item"):
                bias_val = float(bias_param.detach().cpu().item())
            else:
                bias_val = float(bias_param)

            return scale_val, bias_val
    except Exception as exc:
        logger.warning("Could not extract SigLIP logit_scale / logit_bias: %s", exc)

    return None


def calibrate_visual_score(
    raw_cosine: float,
    logit_scale: Optional[float] = None,
    logit_bias: Optional[float] = None,
    fallback_scale: float = 12.0,
    fallback_bias: float = -2.5,
) -> float:
    """Calibrate visual raw cosine similarity into a principled probability in [0, 1].

    For SigLIP models, uses the model's native sigmoid loss calibration:
        p = sigmoid(exp(logit_scale) * cosine + logit_bias)
    If parameters are unavailable (e.g. mock/test embedders), uses a monotonic
    logistic mapping with fallback_scale and fallback_bias.
    """
    if logit_scale is not None and logit_bias is not None:
        # Mathematically grounded SigLIP formulation:
        # logits = exp(logit_scale) * cosine + logit_bias
        exp_scale = math.exp(logit_scale)
        logit = exp_scale * float(raw_cosine) + logit_bias
        prob = float(sigmoid(logit))
    else:
        # Fallback logistic transformation
        logit = fallback_scale * float(raw_cosine) + fallback_bias
        prob = float(sigmoid(logit))

    return float(np.clip(prob, 0.0, 1.0))


def calibrate_text_score(
    raw_cosine: float,
    floor: float = 0.20,
    ceiling: float = 0.80,
) -> float:
    """Calibrate MiniLM raw cosine similarity into [0, 1] using monotonic linear scaling.

    Values below floor map to 0.0, values above ceiling map to 1.0, and
    values between floor and ceiling scale linearly.
    """
    if ceiling <= floor:
        return 1.0 if raw_cosine >= floor else 0.0

    scaled = (float(raw_cosine) - floor) / (ceiling - floor)
    return float(np.clip(scaled, 0.0, 1.0))


def is_visual_relevant(calibrated_prob: float, min_prob: float) -> bool:
    """Evaluate whether a visual candidate satisfies the minimum relevance gate."""
    return calibrated_prob >= min_prob


def is_text_relevant(raw_cosine: float, min_cosine: float) -> bool:
    """Evaluate whether a text candidate satisfies the minimum relevance gate."""
    return raw_cosine >= min_cosine
