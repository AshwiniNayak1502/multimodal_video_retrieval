"""Offline threshold tuning utility to calibrate gating and rejection thresholds.

Evaluates a dataset of positive and negative (query, expected_video_id, expected_time_range)
examples on the real video library, computes per-modality raw-cosine and calibrated-score
distributions for true matches vs false alarms, and determines optimal gating thresholds
that keep all positives while rejecting negatives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure parent path and offline environment are configured
_MODULE3_DIR = Path(__file__).resolve().parent.parent
if str(_MODULE3_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE3_DIR))

import retrieval_engine.offline_env  # noqa: F401
from retrieval_engine.calibration import (
    calibrate_text_score,
    calibrate_visual_score,
    extract_siglip_calibration_parameters,
)
from retrieval_engine.config import Module3Config
from retrieval_engine.query_service import QueryService

REAL_BENCHMARK_EXAMPLES = [
    # Positive: Multimodal S1
    {
        "query": "Java 25 scoped values",
        "expected_video_id": "vid_e9f58d28adfac07e",
        "expected_time_range": [0.0, 5.5],
        "modality": "both",
        "is_positive": True,
        "description": "S1: Title slide & opening speech on scoped values in Java 25",
    },
    # Positive: Spoken S2
    {
        "query": "automatically managed within a defined scope",
        "expected_video_id": "vid_e9f58d28adfac07e",
        "expected_time_range": [8.0, 14.5],
        "modality": "text",
        "is_positive": True,
        "description": "S2: Spoken segment 2 on safer and cleaner scoped values",
    },
    # Positive: Visual S3
    {
        "query": "slide with the heading Code Example in yellow on a black background",
        "expected_video_id": "vid_e9f58d28adfac07e",
        "expected_time_range": [14.0, 18.0],
        "modality": "visual",
        "is_positive": True,
        "description": "S3: Code Example slide with ScopedValue.where Java code",
    },
    # Positive: Paraphrase S4
    {
        "query": "a safer alternative to ThreadLocal in modern Java",
        "expected_video_id": "vid_e9f58d28adfac07e",
        "expected_time_range": [0.0, 14.5],
        "modality": "text",
        "is_positive": True,
        "description": "S4: Paraphrase of thread local replacement in Java",
    },
    # Positive: Other video S5
    {
        "query": "Welcome to the Multimodal Video Retrieval Project",
        "expected_video_id": "vid_d3147dcb7284cc89",
        "expected_time_range": [0.0, 3.0],
        "modality": "text",
        "is_positive": True,
        "description": "S5: Opening greeting in sample presentation",
    },
    # Positive: Silent video visual S6
    {
        "query": "SMPTE color bars test pattern with vertical colored bars",
        "expected_video_id": "vid_8139c3d6507670de",
        "expected_time_range": [0.0, 6.0],
        "modality": "visual",
        "is_positive": True,
        "description": "S6: Silent clip SMPTE color bars visual pattern",
    },
    # Negatives: S7 and ungrounded queries
    {
        "query": "recipe for baking sourdough bread with flour and yeast",
        "expected_video_id": None,
        "expected_time_range": None,
        "modality": "none",
        "is_positive": False,
        "description": "S7 Negative: Cooking recipe with no evidence in videos",
    },
    {
        "query": "underwater scuba diving with dolphins and sharks in coral reef",
        "expected_video_id": None,
        "expected_time_range": None,
        "modality": "none",
        "is_positive": False,
        "description": "Negative: Marine biology unrelated query",
    },
    {
        "query": "space shuttle rocket launch into orbit around mars",
        "expected_video_id": None,
        "expected_time_range": None,
        "modality": "none",
        "is_positive": False,
        "description": "Negative: Space exploration unrelated query",
    },
    {
        "query": "stock market trading algorithms and cryptocurrency price charts",
        "expected_video_id": None,
        "expected_time_range": None,
        "modality": "none",
        "is_positive": False,
        "description": "Negative: Finance unrelated query",
    },
]


def evaluate_modality_distributions(
    service: QueryService,
    examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Measure raw cosine and calibrated score distributions per modality for positives vs negatives."""
    index = service.index
    logit_scale, logit_bias = None, None
    if hasattr(index, "visual_embedder"):
        params = extract_siglip_calibration_parameters(index.visual_embedder)
        if params is not None:
            logit_scale, logit_bias = params

    pos_v_raw: List[float] = []
    pos_v_cal: List[float] = []
    neg_v_raw: List[float] = []
    neg_v_cal: List[float] = []

    pos_t_raw: List[float] = []
    pos_t_cal: List[float] = []
    neg_t_raw: List[float] = []
    neg_t_cal: List[float] = []

    per_query_details: List[Dict[str, Any]] = []

    for ex in examples:
        query = ex["query"]
        is_pos = ex["is_positive"]
        exp_vid = ex.get("expected_video_id")
        exp_tr = ex.get("expected_time_range")
        mod = ex.get("modality", "both")

        # Encode query
        v_vec = index.visual_embedder.embed_query_text(query)
        t_vec = index.text_embedder.embed_query_text(query)

        v_hits = index.search_visual(v_vec, top_k=5)
        t_hits = index.search_text(t_vec, top_k=5)

        # Evaluate visual hits
        best_pos_v_raw = None
        best_pos_v_cal = None
        highest_v_raw = float(v_hits[0]["score"]) if v_hits else 0.0
        highest_v_cal = calibrate_visual_score(
            highest_v_raw, logit_scale, logit_bias,
            service.config.visual_fallback_scale, service.config.visual_fallback_bias
        )

        for h in v_hits:
            raw_c = float(h["score"])
            cal_p = calibrate_visual_score(
                raw_c, logit_scale, logit_bias,
                service.config.visual_fallback_scale, service.config.visual_fallback_bias
            )
            vid = h["video_id"]
            ts = float(h["timestamp_seconds"])
            if is_pos and mod in ("visual", "both") and vid == exp_vid:
                if exp_tr is None or (exp_tr[0] - 2.0 <= ts <= exp_tr[1] + 2.0):
                    if best_pos_v_raw is None or raw_c > best_pos_v_raw:
                        best_pos_v_raw = raw_c
                        best_pos_v_cal = cal_p

        # Evaluate text hits
        best_pos_t_raw = None
        best_pos_t_cal = None
        highest_t_raw = float(t_hits[0]["score"]) if t_hits else 0.0
        highest_t_cal = calibrate_text_score(
            highest_t_raw, service.config.text_floor, service.config.text_ceiling
        )

        for h in t_hits:
            raw_c = float(h["score"])
            cal_s = calibrate_text_score(
                raw_c, service.config.text_floor, service.config.text_ceiling
            )
            vid = h["video_id"]
            st = float(h["start_timestamp"])
            et = float(h["end_timestamp"])
            if is_pos and mod in ("text", "both") and vid == exp_vid:
                if exp_tr is None or max(0.0, min(et, exp_tr[1]) - max(st, exp_tr[0])) > 0:
                    if best_pos_t_raw is None or raw_c > best_pos_t_raw:
                        best_pos_t_raw = raw_c
                        best_pos_t_cal = cal_s

        if is_pos:
            if best_pos_v_raw is not None:
                pos_v_raw.append(best_pos_v_raw)
                pos_v_cal.append(best_pos_v_cal)
            if best_pos_t_raw is not None:
                pos_t_raw.append(best_pos_t_raw)
                pos_t_cal.append(best_pos_t_cal)
        else:
            neg_v_raw.append(highest_v_raw)
            neg_v_cal.append(highest_v_cal)
            neg_t_raw.append(highest_t_raw)
            neg_t_cal.append(highest_t_cal)

        per_query_details.append({
            "query": query,
            "is_positive": is_pos,
            "modality": mod,
            "best_pos_v": {"raw": best_pos_v_raw, "cal": best_pos_v_cal},
            "best_pos_t": {"raw": best_pos_t_raw, "cal": best_pos_t_cal},
            "highest_v": {"raw": highest_v_raw, "cal": highest_v_cal},
            "highest_t": {"raw": highest_t_raw, "cal": highest_t_cal},
        })

    def calc_stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
        a = np.array(arr)
        return {
            "min": round(float(np.min(a)), 4),
            "max": round(float(np.max(a)), 4),
            "mean": round(float(np.mean(a)), 4),
            "median": round(float(np.median(a)), 4),
        }

    return {
        "visual": {
            "pos_raw": calc_stats(pos_v_raw),
            "pos_cal": calc_stats(pos_v_cal),
            "neg_raw": calc_stats(neg_v_raw),
            "neg_cal": calc_stats(neg_v_cal),
            "pos_count": len(pos_v_raw),
            "neg_count": len(neg_v_raw),
        },
        "text": {
            "pos_raw": calc_stats(pos_t_raw),
            "pos_cal": calc_stats(pos_t_cal),
            "neg_raw": calc_stats(neg_t_raw),
            "neg_cal": calc_stats(neg_t_cal),
            "pos_count": len(pos_t_raw),
            "neg_count": len(neg_t_raw),
        },
        "details": per_query_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Module 3 relevance thresholds.")
    parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Path to optional JSON file with custom tuning examples.",
    )
    parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Path to save updated config JSON with recommended thresholds.",
    )
    args = parser.parse_args()

    cfg = Module3Config()
    service = QueryService(config=cfg, warmup=True)

    if args.examples:
        with open(args.examples, "r", encoding="utf-8") as f:
            examples = json.load(f)
        print(f"Loaded {len(examples)} custom examples from {args.examples}.")
    else:
        examples = REAL_BENCHMARK_EXAMPLES
        print(f"Using ground-truth benchmark suite ({len(examples)} queries).")

    print("\nRunning evaluation across modalities on indexed library...")
    analysis = evaluate_modality_distributions(service, examples)

    print("\n" + "=" * 70)
    print("EMPIRICAL SCORE DISTRIBUTIONS PER MODALITY")
    print("=" * 70)

    vis = analysis["visual"]
    print("VISUAL MODALITY (SigLIP2):")
    print(f"  Positives (N={vis['pos_count']}):")
    print(f"    Raw Cosine:       Min={vis['pos_raw']['min']:.4f}, Mean={vis['pos_raw']['mean']:.4f}, Max={vis['pos_raw']['max']:.4f}")
    print(f"    Calibrated Prob:  Min={vis['pos_cal']['min']:.4f}, Mean={vis['pos_cal']['mean']:.4f}, Max={vis['pos_cal']['max']:.4f}")
    print(f"  Negatives / False Alarms (N={vis['neg_count']}):")
    print(f"    Raw Cosine:       Min={vis['neg_raw']['min']:.4f}, Mean={vis['neg_raw']['mean']:.4f}, Max={vis['neg_raw']['max']:.4f}")
    print(f"    Calibrated Prob:  Min={vis['neg_cal']['min']:.4f}, Mean={vis['neg_cal']['mean']:.4f}, Max={vis['neg_cal']['max']:.4f}")

    txt = analysis["text"]
    print("\nTEXT MODALITY (MiniLM-L6-v2):")
    print(f"  Positives (N={txt['pos_count']}):")
    print(f"    Raw Cosine:       Min={txt['pos_raw']['min']:.4f}, Mean={txt['pos_raw']['mean']:.4f}, Max={txt['pos_raw']['max']:.4f}")
    print(f"    Calibrated Score: Min={txt['pos_cal']['min']:.4f}, Mean={txt['pos_cal']['mean']:.4f}, Max={txt['pos_cal']['max']:.4f}")
    print(f"  Negatives / False Alarms (N={txt['neg_count']}):")
    print(f"    Raw Cosine:       Min={txt['neg_raw']['min']:.4f}, Mean={txt['neg_raw']['mean']:.4f}, Max={txt['neg_raw']['max']:.4f}")
    print(f"    Calibrated Score: Min={txt['neg_cal']['min']:.4f}, Mean={txt['neg_cal']['mean']:.4f}, Max={txt['neg_cal']['max']:.4f}")

    # Compute optimal single-gate thresholds:
    # Visual gate on calibrated prob (or raw cosine):
    # We want threshold <= pos_min and > neg_max (if separable) or mid-point
    vis_neg_max_cal = vis["neg_cal"]["max"]
    vis_pos_min_cal = vis["pos_cal"]["min"]
    if vis_pos_min_cal > vis_neg_max_cal:
        rec_vis_prob = round((vis_pos_min_cal + vis_neg_max_cal) / 2.0, 3)
    else:
        # Fallback to keep positives
        rec_vis_prob = round(max(0.05, min(0.15, vis_pos_min_cal * 0.9)), 3)

    txt_neg_max_raw = txt["neg_raw"]["max"]
    txt_pos_min_raw = txt["pos_raw"]["min"]
    if txt_pos_min_raw > txt_neg_max_raw:
        rec_txt_cosine = round((txt_pos_min_raw + txt_neg_max_raw) / 2.0, 3)
    else:
        rec_txt_cosine = round(max(0.15, min(0.25, txt_pos_min_raw * 0.9)), 3)

    print("\n" + "-" * 70)
    print("RECOMMENDED SINGLE-GATE THRESHOLDS:")
    print(f"  visual_min_prob:  {rec_vis_prob:.3f} (Pos min={vis_pos_min_cal:.4f}, Neg max={vis_neg_max_cal:.4f})")
    print(f"  text_min_cosine:  {rec_txt_cosine:.3f} (Pos min={txt_pos_min_raw:.4f}, Neg max={txt_neg_max_raw:.4f})")
    print("=" * 70)

    for item in analysis["details"]:
        tag = "POS" if item["is_positive"] else "NEG"
        q = item["query"]
        v_cal = item["best_pos_v"]["cal"] if item["is_positive"] else item["highest_v"]["cal"]
        t_cal = item["best_pos_t"]["cal"] if item["is_positive"] else item["highest_t"]["cal"]
        v_str = f"{v_cal:.4f}" if v_cal is not None else "N/A"
        t_str = f"{t_cal:.4f}" if t_cal is not None else "N/A"
        print(f"  [{tag}] \"{q[:42]}\" -> Best Vis={v_str}, Best Txt={t_str}")

    if args.save_config:
        save_p = Path(args.save_config)
        cfg.visual_min_prob = rec_vis_prob
        cfg.text_min_cosine = rec_txt_cosine
        # Remove double gating by setting final_min_score to 0.0 or lowest gated threshold
        cfg.final_min_score = 0.0
        cfg.min_top_k = 3
        cfg.max_top_k = 5
        cfg.default_top_k = 5
        with open(save_p, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        print(f"\nSaved updated configuration to {save_p}")


if __name__ == "__main__":
    main()
