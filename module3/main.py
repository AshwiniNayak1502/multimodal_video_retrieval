"""CLI entrypoint for Module 3 Multimodal Semantic Query Processing & Segment Retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to sys.path so retrieval_engine can be imported directly
_MODULE3_DIR = Path(__file__).resolve().parent
if str(_MODULE3_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE3_DIR))

from retrieval_engine import offline_env  # noqa: F401
from retrieval_engine.config import Module3Config
from retrieval_engine.errors import ValidationError
from retrieval_engine.logging_config import setup_logging
from retrieval_engine.query_service import QueryService


def format_search_text_output(response_dict: dict, debug: bool = False) -> str:
    """Format a SearchResponse dictionary into human-readable console text."""
    lines: List[str] = []
    status = response_dict.get("status", "")
    message = response_dict.get("message", "")
    results = response_dict.get("results", [])
    timings = response_dict.get("timings_ms", {})
    dbg = response_dict.get("debug", {})
    notices = response_dict.get("notices", [])

    lines.append("=" * 60)
    lines.append("MULTIMODAL VIDEO SEGMENT RETRIEVAL")
    lines.append("=" * 60)
    lines.append(f"Status:  {status.upper()}")
    lines.append(f"Message: {message}")

    for notice in notices:
        lines.append(f"Notice:  {notice}")
    if dbg.get("warning"):
        lines.append(f"Warning: {dbg['warning']}")
    if dbg.get("degraded_modalities"):
        lines.append(f"Notice:  Degraded search mode for: {', '.join(dbg['degraded_modalities'])}")

    lines.append("-" * 60)

    if status == "ok" and results:
        lines.append(f"Retrieved Top-{len(results)} Segment(s):")
        for idx, res in enumerate(results, start=1):
            lines.append(f"\n[{idx}] Video: {res.get('video_filename')} (ID: {res.get('video_id')})")
            lines.append(
                f"    Time Range:  {res.get('start_formatted')} -> {res.get('end_formatted')} "
                f"({res.get('start_seconds'):.2f}s - {res.get('end_seconds'):.2f}s)"
            )
            lines.append(f"    Seek Point:  {res.get('seek_seconds'):.2f}s")
            lines.append(f"    Span Type:   {res.get('span_type')} (res: {res.get('resolution_seconds', 0.0)}s)")
            lines.append(f"    Modalities:  {', '.join(res.get('modalities', []))}")
            lines.append(
                f"    Scores:      Final={res.get('final_score'):.4f} "
                f"(Visual={res.get('visual_score')}, Text={res.get('text_score')})"
            )
            raw_scores = []
            if res.get("visual_raw_cosine") is not None:
                raw_scores.append(f"vis_raw={res.get('visual_raw_cosine'):.4f}")
            if res.get("text_raw_cosine") is not None:
                raw_scores.append(f"txt_raw={res.get('text_raw_cosine'):.4f}")
            if raw_scores:
                lines.append(f"    Raw Cosine:  {', '.join(raw_scores)}")
            lines.append(f"    Rationale:   {res.get('reason')}")
            if res.get("transcript_text"):
                lines.append(f"    Transcript:  \"{res.get('transcript_text')}\"")
            if res.get("frame_id"):
                lines.append(f"    Keyframe ID: {res.get('frame_id')} (at {res.get('matched_at_seconds')}s)")
            if not res.get("video_file_exists"):
                lines.append("    Notice:      Source video file not found at local path.")

    if status == "no_match" and dbg.get("best_rejected_candidates"):
        lines.append("\nBest Candidates (Rejected by Relevance Gates):")
        for c in dbg["best_rejected_candidates"]:
            lines.append(
                f"  [{c.get('modality').upper()}] {c.get('video_filename')} "
                f"(raw={c.get('raw_cosine')}, cal={c.get('calibrated_score')}): {c.get('rejection_reason')}"
            )

    if debug or (status == "ok" and not debug):
        lines.append("\n" + "-" * 60)
        lines.append("Execution Latency (ms):")
        lines.append(
            f"  Encode Visual: {timings.get('query_encode_visual_ms', 0):.2f} ms | "
            f"Encode Text: {timings.get('query_encode_text_ms', 0):.2f} ms"
        )
        lines.append(
            f"  Search Visual: {timings.get('search_visual_ms', 0):.2f} ms | "
            f"Search Text: {timings.get('search_text_ms', 0):.2f} ms"
        )
        lines.append(
            f"  Calibration & Fuse: {timings.get('calibrate_fuse_ms', 0):.2f} ms | "
            f"Segment Build: {timings.get('segment_build_ms', 0):.2f} ms"
        )
        lines.append(f"  Total Latency:      {timings.get('total_ms', 0):.2f} ms")

    if debug and dbg:
        lines.append("\nDebug Diagnostics:")
        lines.append(
            f"  Cache Hits (this query): Visual={dbg.get('cache_hit_visual')}, Text={dbg.get('cache_hit_text')}"
        )
        lines.append(
            f"  Cumulative Cache Hits:   Visual={dbg.get('cache_hits_visual', 0)}, Text={dbg.get('cache_hits_text', 0)}"
        )
        lines.append(
            f"  Model Load Counts:       Visual={dbg.get('visual_load_count', 0)}, Text={dbg.get('text_load_count', 0)}"
        )
        lines.append(f"  Candidates Retrieved:    {dbg.get('candidates_retrieved')}")
        lines.append(f"  Candidates Passed Gate:  {dbg.get('candidates_passed_gating')}")
        lines.append(f"  Models Reused: {dbg.get('models_reused')}, Index Reused: {dbg.get('index_reused')}")

        trace = dbg.get("trace", {})
        if trace:
            lines.append("\nPer-Search Trace (R10):")
            lines.append(f"  Query: \"{trace.get('query_received')}\" (normalised: \"{trace.get('query_normalised')}\")")
            lines.append("  Modalities:")
            for m, mdata in trace.get("modalities", {}).items():
                lines.append(f"    - {m}: searched={mdata.get('searched')}, reason={mdata.get('reason')}, pool={mdata.get('candidate_count')}")

            lines.append("  Top Raw Hits:")
            for m, mhits in trace.get("top_raw_hits", {}).items():
                lines.append(f"    [{m.upper()}] {len(mhits)} hits:")
                for h in mhits[:3]:
                    if m == "visual":
                        lines.append(f"      • {h.get('video_filename')} @ {h.get('timestamp_seconds')}s (cos={h.get('raw_cosine')}, cal={h.get('calibrated_score')})")
                    else:
                        lines.append(f"      • {h.get('video_filename')} @ {h.get('start_timestamp')}-{h.get('end_timestamp')}s (cos={h.get('raw_cosine')}, cal={h.get('calibrated_score')})")

            lines.append("  Relevance Gating:")
            for gname, ginfo in trace.get("gate_decisions", {}).items():
                if isinstance(ginfo, dict) and "passed_count" in ginfo:
                    lines.append(f"    - {gname}: passed={ginfo.get('passed_count')}, rejected={ginfo.get('rejected_count')}")

            if "expect_evaluation" in trace:
                ee = trace["expect_evaluation"]
                lines.append("\nExpectation Evaluation (--expect):")
                lines.append(f"  Target: {ee.get('expected')}")
                lines.append(f"  Status: {ee.get('status')}")
                if ee.get("status") == "MATCHED":
                    lines.append(f"  Result: Found at Rank {ee.get('rank')}")
                else:
                    lines.append(f"  Reason: {ee.get('reason', ee.get('error'))}")

    lines.append("=" * 60)
    return "\n".join(lines)


def load_cli_config(config_arg: Optional[str] = None) -> Module3Config:
    """Load configuration from explicit path or default config.json if present."""
    cfg_path: Optional[Path] = None
    if config_arg:
        cfg_path = Path(config_arg)
        if not cfg_path.is_file():
            print(f"Error: Specified configuration file not found: {config_arg}", file=sys.stderr)
            sys.exit(1)
    else:
        default_path = _MODULE3_DIR / "config.json"
        if default_path.is_file():
            cfg_path = default_path

    if cfg_path is not None:
        try:
            return Module3Config.from_json(cfg_path)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            print(f"Error: Failed to load configuration from {cfg_path}: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"Error: Unexpected error reading {cfg_path}: {exc}", file=sys.stderr)
            sys.exit(1)

    return Module3Config()


def cmd_search(args: argparse.Namespace) -> int:
    """Execute search query command."""
    cfg = load_cli_config(args.config)
    setup_logging(log_file=cfg.log_file)
    service = QueryService(config=cfg, warmup=True)

    # CLI flags override configuration file values
    top_k = args.top_k if args.top_k is not None else cfg.default_top_k
    vis_w = args.visual_weight if args.visual_weight is not None else cfg.visual_weight
    txt_w = args.text_weight if args.text_weight is not None else cfg.text_weight

    mod_weights = None
    if args.visual_weight is not None or args.text_weight is not None:
        mod_weights = {
            "visual": vis_w,
            "text": txt_w,
        }

    response = service.search(
        query=args.query,
        top_k=top_k,
        video_ids=args.video_id,
        modality_weights=mod_weights,
        expect_segment=args.expect,
    )

    resp_dict = response.to_dict()

    if args.json:
        print(json.dumps(resp_dict, indent=2))
    else:
        print(format_search_text_output(resp_dict, debug=args.debug))

    return 0 if response.status in ("ok", "no_match") else 1


def cmd_interactive(args: argparse.Namespace) -> int:
    """Interactive REPL mode: loads models and index once, loops over queries (R9)."""
    cfg = load_cli_config(args.config)
    setup_logging(log_file=cfg.log_file)
    print("=" * 60)
    print("MULTIMODAL VIDEO RETRIEVAL - INTERACTIVE REPL")
    print("=" * 60)
    print("Initializing models and vector index (one-time load)...")
    service = QueryService(config=cfg, warmup=True)
    print("Ready! Enter your queries below.")
    print("Commands:")
    print("  :top_k <N>      - set top_k result limit (3 to 5)")
    print("  :debug on|off   - toggle verbose debug diagnostics")
    print("  :expect <spec>  - set expected segment (e.g. videoplayback.mp4:0.0-5.0)")
    print("  :expect clear   - clear expected segment target")
    print("  :quit | :exit   - exit REPL")
    print("-" * 60)

    curr_top_k = args.top_k if args.top_k is not None else cfg.default_top_k
    curr_debug = args.debug
    curr_expect: Optional[str] = args.expect

    while True:
        try:
            line = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive session.")
            break

        if not line:
            continue
        if line in (":quit", ":exit", "quit", "exit", ":q"):
            print("Exiting interactive session.")
            break
        if line.startswith(":top_k"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                val = int(parts[1])
                if 3 <= val <= 5:
                    curr_top_k = val
                    print(f"[REPL] top_k set to {curr_top_k}")
                else:
                    print("[REPL] Error: top_k must be between 3 and 5.")
            else:
                print("[REPL] Usage: :top_k <integer 3-5>")
            continue
        if line.startswith(":debug"):
            parts = line.split()
            if len(parts) >= 2:
                curr_debug = parts[1].lower() in ("on", "true", "1")
                print(f"[REPL] debug set to {curr_debug}")
            else:
                print("[REPL] Usage: :debug on|off")
            continue
        if line.startswith(":expect"):
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                val = parts[1].strip()
                curr_expect = None if val.lower() == "clear" else val
                print(f"[REPL] expect target set to: {curr_expect}")
            else:
                print("[REPL] Usage: :expect <filename>:<start>-<end> or :expect clear")
            continue

        response = service.search(
            query=line,
            top_k=curr_top_k,
            expect_segment=curr_expect,
        )
        resp_dict = response.to_dict()

        if args.json:
            print(json.dumps(resp_dict, indent=2))
        else:
            print(format_search_text_output(resp_dict, debug=curr_debug))

        dbg = resp_dict.get("debug", {})
        print(
            f"\n[Telemetry] Cache Hits: Vis={dbg.get('cache_hits_visual')}, Txt={dbg.get('cache_hits_text')} | "
            f"Model Loads: Vis={dbg.get('visual_load_count')}, Txt={dbg.get('text_load_count')}"
        )

    return 0


def cmd_videos(args: argparse.Namespace) -> int:
    """List indexed video records."""
    cfg = load_cli_config(args.config)
    service = QueryService(config=cfg, warmup=False)
    video_list = service.list_videos()

    if args.json:
        print(json.dumps(video_list, indent=2))
        return 0

    print("=" * 60)
    print("INDEXED VIDEOS IN RETRIEVAL STORE")
    print("=" * 60)
    if not video_list:
        print("No indexed videos found. Run 'python main.py index --all' in module2 first.")
    else:
        for v in video_list:
            print(f"Video ID:    {v['video_id']}")
            print(f"  Filename:  {v['filename']}")
            print(f"  Duration:  {v['duration_seconds']:.2f}s | Has Audio: {v['has_audio']}")
            print(f"  Vectors:   Visual={v['visual_vector_count']} | Text={v['text_vector_count']}")
            print(f"  Path:      {v['video_path']} (Exists: {v['exists_on_disk']})")
            if v.get("has_zero_vectors"):
                print("  STATUS:    WARNING - missing vectors!")
            if v.get("no_speech_detected"):
                print("  STATUS:    NOTE - no speech detected in audio track")
            print("-" * 60)
    print("=" * 60)
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Report retrieval index readiness and health."""
    cfg = load_cli_config(args.config)
    service = QueryService(config=cfg, warmup=False)
    h = service.health()

    if args.json:
        print(json.dumps(h, indent=2))
        return 0 if h["status"] in ("ready", "degraded") else 1

    print("=" * 60)
    print("MODULE 3 RETRIEVAL ENGINE HEALTH CHECK")
    print("=" * 60)
    print(f"Status:          {h['status'].upper()}")
    print(f"Message:         {h['message']}")
    print(f"Indexed Videos:  {h['indexed_videos']}")
    print(f"Visual Vectors:  {h['visual_vectors']}")
    print(f"Text Vectors:    {h['text_vectors']}")
    print(f"Visual Model:    {h['visual_model']}")
    print(f"Text Model:      {h['text_model']}")
    encoders = h.get("encoders", {})
    if encoders:
        print("Encoders Readiness:")
        for ename, edata in encoders.items():
            print(f"  {ename.capitalize()}: ready={edata.get('ready')}, loads={edata.get('load_count')}, error={edata.get('error')}")
    if h.get("videos_with_zero_vectors"):
        print(f"Missing Vectors: {h['videos_with_zero_vectors']}")
    if h.get("videos_without_speech"):
        print(f"Videos Without Speech: {h['videos_without_speech']}")
    print(f"Schema Version:  {h['schema_version']}")
    print("=" * 60)
    return 0 if h["status"] in ("ready", "degraded") else 1


def main() -> None:
    """CLI parser and router."""
    parser = argparse.ArgumentParser(
        description="Module 3: Multimodal Semantic Query Processing & Timestamped Segment Retrieval"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: search
    search_parser = subparsers.add_parser("search", help="Execute semantic query search")
    search_parser.add_argument("query", type=str, help="Natural language query string")
    search_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.json file",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top segments to return (3 to 5, default 5)",
    )
    search_parser.add_argument(
        "--video-id",
        action="append",
        default=None,
        help="Filter search to specific video ID (can be specified multiple times)",
    )
    search_parser.add_argument(
        "--visual-weight",
        type=float,
        default=None,
        help="Visual modality weight (default 0.6)",
    )
    search_parser.add_argument(
        "--text-weight",
        type=float,
        default=None,
        help="Text modality weight (default 0.4)",
    )
    search_parser.add_argument(
        "--expect",
        type=str,
        default=None,
        help="Expected ground-truth target span (e.g. videoplayback.mp4:0.0-5.0)",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON response",
    )
    search_parser.add_argument(
        "--debug",
        action="store_true",
        help="Output detailed per-stage timings, debug telemetry, and trace",
    )

    # Command: interactive (R9)
    inter_parser = subparsers.add_parser("interactive", help="Start interactive search REPL session")
    inter_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.json file",
    )
    inter_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Initial top_k result limit (3 to 5, default 5)",
    )
    inter_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode by default in REPL",
    )
    inter_parser.add_argument(
        "--expect",
        type=str,
        default=None,
        help="Default expected ground-truth target span",
    )
    inter_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format in REPL",
    )

    # Command: videos
    videos_parser = subparsers.add_parser("videos", help="List all indexed videos")
    videos_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.json file",
    )
    videos_parser.add_argument("--json", action="store_true", help="Output JSON array")

    # Command: health
    health_parser = subparsers.add_parser("health", help="Check vector index health and readiness")
    health_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.json file",
    )
    health_parser.add_argument("--json", action="store_true", help="Output JSON health status")

    args = parser.parse_args()

    if args.command == "search":
        sys.exit(cmd_search(args))
    elif args.command == "interactive":
        sys.exit(cmd_interactive(args))
    elif args.command == "videos":
        sys.exit(cmd_videos(args))
    elif args.command == "health":
        sys.exit(cmd_health(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
