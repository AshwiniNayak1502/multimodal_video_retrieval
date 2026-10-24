"""Latency and Throughput Benchmark Utility for Module 3 Retrieval Engine.

Measures cold vs warm per-stage timings (query encoding, FAISS search, calibration/fusion,
segment construction, total latency) across varied queries, prints detailed statistics,
and records model load counts and cache hit counters to latency_report.txt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_MODULE3_DIR = Path(__file__).resolve().parent.parent
if str(_MODULE3_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE3_DIR))

import retrieval_engine.offline_env  # noqa: F401
from retrieval_engine.config import Module3Config
from retrieval_engine.query_service import QueryService

BENCHMARK_QUERIES = [
    "What are scoped values in Java and how are they better than ThreadLocal?",
    "a purple title card that says Java 25 Feature Highlight",
    "slide with the heading Code Example",
    "Welcome to the Multimodal Video Retrieval Project",
    "recipe for baking sourdough bread",
    "underwater scuba diving with dolphins",
]


def run_benchmark(n_iterations: int = 5, output_file: Optional[Path] = None) -> str:
    cfg = Module3Config()
    cfg.ensure_directories()
    out_path = output_file or (cfg.output_dir / "latency_report.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("MODULE 3 RETRIEVAL ENGINE LATENCY & TELEMETRY BENCHMARK REPORT")
    lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Iterations per query: {n_iterations}")
    lines.append(f"Test queries ({len(BENCHMARK_QUERIES)}):")
    for idx, q in enumerate(BENCHMARK_QUERIES, 1):
        lines.append(f"  [{idx}] \"{q}\"")
    lines.append("=" * 72)

    # Initialize service without warmup to capture true cold-start on query 1
    t0 = time.perf_counter()
    service = QueryService(config=cfg, warmup=False)
    init_ms = (time.perf_counter() - t0) * 1000.0
    lines.append(f"QueryService init (lazy models, warmup=False): {init_ms:.2f} ms")
    lines.append("-" * 72)

    # Run Cold queries (first time each query is seen by un-warmed service)
    cold_results: List[Dict[str, Any]] = []
    for idx, q in enumerate(BENCHMARK_QUERIES, 1):
        resp = service.search(query=q, top_k=5)
        cold_results.append({
            "query": q,
            "status": resp.status,
            "results_count": len(resp.results),
            "timings": resp.timings_ms,
            "debug": resp.debug,
        })

    # Run Warm queries (repeated executions with LRU cache & warmed embedders)
    warm_results: List[Dict[str, Any]] = []
    for iteration in range(n_iterations):
        for q in BENCHMARK_QUERIES:
            resp = service.search(query=q, top_k=5)
            warm_results.append({
                "iteration": iteration + 1,
                "query": q,
                "timings": resp.timings_ms,
                "debug": resp.debug,
            })

    # Stage breakdown helper
    stages = [
        ("query_encode_visual_ms", "Query Encode Visual (SigLIP2)"),
        ("query_encode_text_ms", "Query Encode Text (MiniLM)"),
        ("search_visual_ms", "FAISS Visual Search (IndexFlatIP)"),
        ("search_text_ms", "FAISS Text Search (IndexFlatIP)"),
        ("calibrate_fuse_ms", "Relevance Gating & Calibration"),
        ("segment_build_ms", "Segment Assembly & Ranking"),
        ("total_ms", "Total End-to-End Search"),
    ]

    lines.append("\n1. COLD EXECUTION LATENCY (Uncached, First Encounter)")
    lines.append("-" * 72)
    lines.append(f"{'Stage':<35} | {'Query 1 (Cold Models)':<20} | {'Cold Mean (Q1..QN)':<15}")
    lines.append("-" * 72)
    q1_timings = cold_results[0]["timings"]
    for key, label in stages:
        q1_val = q1_timings.get(key, 0.0)
        mean_cold = float(np.mean([r["timings"].get(key, 0.0) for r in cold_results]))
        lines.append(f"{label:<35} | {q1_val:>17.2f} ms | {mean_cold:>12.2f} ms")

    lines.append("\n2. WARM EXECUTION LATENCY (LRU Query Cache Active, N={})".format(len(warm_results)))
    lines.append("-" * 72)
    lines.append(f"{'Stage':<35} | {'Min (ms)':<9} | {'Mean (ms)':<9} | {'P95 (ms)':<9} | {'Max (ms)':<9}")
    lines.append("-" * 72)
    for key, label in stages:
        vals = [r["timings"].get(key, 0.0) for r in warm_results]
        min_v = float(np.min(vals))
        mean_v = float(np.mean(vals))
        p95_v = float(np.percentile(vals, 95))
        max_v = float(np.max(vals))
        lines.append(f"{label:<35} | {min_v:>9.2f} | {mean_v:>9.2f} | {p95_v:>9.2f} | {max_v:>9.2f}")

    # Telemetry and Load Counters
    last_debug = warm_results[-1]["debug"]
    lines.append("\n3. MODEL & INDEX REUSE TELEMETRY")
    lines.append("-" * 72)
    lines.append(f"Visual Model Load Count: {last_debug.get('visual_load_count')} (Must be <= 1 for single-process session)")
    lines.append(f"Text Model Load Count:   {last_debug.get('text_load_count')} (Must be <= 1 for single-process session)")
    lines.append(f"Models Reused Flag:      {last_debug.get('models_reused')}")
    lines.append(f"Index Reused Flag:       {last_debug.get('index_reused')}")
    lines.append(f"Query Cache Hits Visual: {last_debug.get('cache_hits_visual')} / {len(BENCHMARK_QUERIES) * (n_iterations + 1)}")
    lines.append(f"Query Cache Hits Text:   {last_debug.get('cache_hits_text')} / {len(BENCHMARK_QUERIES) * (n_iterations + 1)}")

    lines.append("\n4. QUERY-BY-QUERY SUMMARY (Cold vs Warm Average)")
    lines.append("-" * 72)
    lines.append(f"{'Query Snippet':<40} | {'Status':<8} | {'Cold Total':<10} | {'Warm Total':<10}")
    lines.append("-" * 72)
    for q_idx, q in enumerate(BENCHMARK_QUERIES):
        cold_tot = cold_results[q_idx]["timings"].get("total_ms", 0.0)
        warm_tots = [r["timings"].get("total_ms", 0.0) for r in warm_results if r["query"] == q]
        warm_avg = float(np.mean(warm_tots)) if warm_tots else 0.0
        status = cold_results[q_idx]["status"]
        lines.append(f"{q[:38]:<40} | {status:<8} | {cold_tot:>8.2f} ms | {warm_avg:>8.2f} ms")

    lines.append("=" * 72)
    report_text = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark QueryService latency and throughput.")
    parser.add_argument("-n", "--iterations", type=int, default=5, help="Number of warm iterations per query")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output latency report file path")
    args = parser.parse_args()

    out_file = Path(args.output) if args.output else None
    report = run_benchmark(n_iterations=args.iterations, output_file=out_file)
    print(report)


if __name__ == "__main__":
    main()
