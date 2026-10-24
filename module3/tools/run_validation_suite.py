"""Run Item 9 and Item 3 validation suite against real offline models and index.

Saves --json --debug outputs to module3/output/validation_runs/ and reports
before/after fusion comparison (noisy_or vs weighted) and negative gate rejection evidence.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_MODULE3_DIR = Path(__file__).resolve().parent.parent
if str(_MODULE3_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE3_DIR))

import retrieval_engine.offline_env  # noqa: F401
from retrieval_engine.config import Module3Config
from retrieval_engine.query_service import QueryService

POSITIVE_QUERIES = [
    "What are scoped values in Java and how are they better than ThreadLocal?",
    "a purple title card that says Java 25 Feature Highlight",
    "slide with the heading Code Example",
    "Welcome to the Multimodal Video Retrieval Project",
]

NEGATIVE_QUERIES = [
    "recipe for baking sourdough bread",
    "underwater scuba diving with dolphins",
    "a person riding a bicycle on a road",
    "stock market trading algorithms",
]

SILENT_CHECK = {
    "query": "SMPTE color bars test pattern with vertical colored bars",
    "video_ids": ["vid_8139c3d6507670de"],
}

LOGGED_REAL_QUERIES_FOR_FUSION_COMPARISON = [
    "What are scoped values in Java and how are they better than ThreadLocal?",
    "a purple title card that says Java 25 Feature Highlight",
    "slide with the heading Code Example",
]


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", s)[:50]


def main() -> None:
    cfg = Module3Config()
    cfg.ensure_directories()
    val_dir = cfg.output_dir / "validation_runs"
    val_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RUNNING ITEM 9 REAL-MODEL VALIDATION RUNS")
    print("=" * 72)

    # Use default configured mode ("weighted")
    service = QueryService(config=cfg, warmup=True)

    # 1. Positives
    print("\n--- POSITIVE QUERIES ---")
    for q in POSITIVE_QUERIES:
        resp = service.search(query=q, top_k=5)
        d = resp.to_dict()
        slug = slugify(q)
        out_file = val_dir / f"pos_{slug}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        print(f"[{resp.status.upper()}] \"{q}\" -> {len(resp.results)} results saved to {out_file.name}")
        for r_idx, r in enumerate(resp.results, 1):
            raw_s = f"(vis_raw={r.visual_raw_cosine}, txt_raw={r.text_raw_cosine})"
            print(f"    #{r_idx}: {r.video_filename} [{r.start_formatted} - {r.end_formatted}] score={r.final_score:.4f} {raw_s} reason={r.reason}")

    # 2. Negatives
    print("\n--- NEGATIVE QUERIES (Must return status='no_match') ---")
    neg_passed = True
    for q in NEGATIVE_QUERIES:
        resp = service.search(query=q, top_k=5)
        d = resp.to_dict()
        slug = slugify(q)
        out_file = val_dir / f"neg_{slug}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        status_ok = resp.status == "no_match"
        if not status_ok:
            neg_passed = False
        mark = "PASS" if status_ok else "FAIL"
        print(f"[{mark}] [{resp.status.upper()}] \"{q}\" -> {len(resp.results)} results saved to {out_file.name}")
        if resp.debug.get("best_rejected_candidates"):
            for c in resp.debug["best_rejected_candidates"][:2]:
                print(f"    Gate rejected: {c.get('modality')} {c.get('rejection_reason')}")

    # 3. Silent video check
    print("\n--- SILENT VIDEO RESTRICTED CHECK ---")
    q = SILENT_CHECK["query"]
    vids = SILENT_CHECK["video_ids"]
    resp = service.search(query=q, top_k=5, video_ids=vids)
    d = resp.to_dict()
    out_file = val_dir / "silent_check_vid_8139c3d6507670de.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print(f"[{resp.status.upper()}] Restricted to {vids}: {len(resp.results)} result(s)")
    for r_idx, r in enumerate(resp.results, 1):
        print(f"    #{r_idx}: {r.video_filename} [{r.start_formatted} - {r.end_formatted}] score={r.final_score:.4f} modalities={r.modalities}")

    # 4. Item 3 Before vs After Comparison (noisy_or vs weighted)
    print("\n" + "=" * 72)
    print("ITEM 3: FUSION RANKING COMPARISON (noisy_or vs weighted)")
    print("=" * 72)

    cfg_noisy = Module3Config()
    cfg_noisy.fusion_mode = "noisy_or"
    service_noisy = QueryService(config=cfg_noisy, video_index=service.index, warmup=False)

    comparison_report_lines: List[str] = [
        "=" * 72,
        "FUSION COMPARISON REPORT: NOISY-OR (BEFORE) VS WEIGHTED (AFTER)",
        "=" * 72,
    ]

    for q in LOGGED_REAL_QUERIES_FOR_FUSION_COMPARISON:
        resp_before = service_noisy.search(query=q, top_k=5)
        resp_after = service.search(query=q, top_k=5)

        comparison_report_lines.append(f"\nQUERY: \"{q}\"")
        comparison_report_lines.append("-" * 72)
        comparison_report_lines.append(f"{'Rank':<5} | {'Noisy-OR (Before)':<30} | {'Score':<8} | {'Weighted (After)':<30} | {'Score':<8}")
        comparison_report_lines.append("-" * 72)

        max_len = max(len(resp_before.results), len(resp_after.results))
        for i in range(max_len):
            b_str = "None"
            b_sc = "N/A"
            if i < len(resp_before.results):
                br = resp_before.results[i]
                b_str = f"{br.video_filename} [{br.start_formatted}-{br.end_formatted}]"
                b_sc = f"{br.final_score:.4f}"

            a_str = "None"
            a_sc = "N/A"
            if i < len(resp_after.results):
                ar = resp_after.results[i]
                a_str = f"{ar.video_filename} [{ar.start_formatted}-{ar.end_formatted}]"
                a_sc = f"{ar.final_score:.4f}"

            comparison_report_lines.append(f"{i+1:<5} | {b_str:<30} | {b_sc:<8} | {a_str:<30} | {a_sc:<8}")

    comp_text = "\n".join(comparison_report_lines)
    print(comp_text)

    comp_file = cfg.output_dir / "fusion_comparison_report.txt"
    with open(comp_file, "w", encoding="utf-8") as f:
        f.write(comp_text + "\n")
    print(f"\nSaved comparison to {comp_file}")


if __name__ == "__main__":
    main()
