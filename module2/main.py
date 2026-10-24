"""CLI entrypoint for Module 2: Visual + Text Embedding Generation & Persistent Vector Storage."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Insert Module 2 directory to sys.path to enable clean package execution
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure offline environment configuration is applied before heavy ML libraries
import embedding_store.offline_env  # noqa: F401

from embedding_store.config import Module2Config
from embedding_store.embedders import download_models
from embedding_store.indexer import VideoIndexer
from embedding_store.logging_config import setup_logging
from embedding_store.retrieval import VideoIndex
from embedding_store.validation import validate_integrity


def build_parser() -> argparse.ArgumentParser:
    """Create and configure command-line argument parser for Module 2."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Module 2: Visual + Text Embedding Generation & Persistent Vector Storage.\n"
            "Fully local, offline-at-runtime Multimodal Video Retrieval indexer and vector store."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 0. check-models
    check_parser = subparsers.add_parser(
        "check-models",
        help="Check offline availability, load encoders, embed a probe, and print versions.",
    )
    check_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom JSON configuration file.",
    )

    # 1. download-models
    subparsers.add_parser(
        "download-models",
        help="One-time command to download and cache HuggingFace models locally.",
    )

    # 2. index
    index_parser = subparsers.add_parser(
        "index",
        help="Incrementally index videos from Module 1 output into persistent vector store.",
    )
    group = index_parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--all",
        action="store_true",
        help="Index all completed videos discovered in Module 1 metadata directory.",
    )
    group.add_argument(
        "--video-id",
        type=str,
        help="Index only a specific video ID (e.g. vid_d3147dcb7284cc89).",
    )

    index_parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-embedding and replacement of items even if content hash and model match.",
    )
    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe SQLite database and FAISS index files and re-embed all items from scratch.",
    )
    index_parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Remove database videos whose Module 1 metadata JSON no longer exists.",
    )
    index_parser.add_argument(
        "--visual-only",
        action="store_true",
        help="Index only visual frames (skip speech transcripts).",
    )
    index_parser.add_argument(
        "--text-only",
        action="store_true",
        help="Index only speech transcripts (skip visual frames).",
    )
    index_parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Override execution device ('cpu' or 'cuda').",
    )
    index_parser.add_argument(
        "--save-each",
        action="store_true",
        help="Save FAISS indices to disk after each video instead of only at the end.",
    )
    index_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom JSON configuration file.",
    )
    index_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )
    index_parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to output log file.",
    )

    # 3. validate
    val_parser = subparsers.add_parser(
        "validate",
        help="Run database and vector index integrity checks.",
    )
    val_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom JSON configuration file.",
    )

    # 4. search
    search_parser = subparsers.add_parser(
        "search",
        help="Execute visual, text, or multimodal semantic search.",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query text string.",
    )
    search_parser.add_argument(
        "--modality",
        type=str,
        choices=["visual", "text", "both"],
        default="both",
        help="Target search modality (default: 'both').",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve (default: 5).",
    )
    search_parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Filter search results to a specific video ID.",
    )
    search_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom JSON configuration file.",
    )

    # 5. stats
    stats_parser = subparsers.add_parser(
        "stats",
        help="Display summary metrics of the vector store and indexed media.",
    )
    stats_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom JSON configuration file.",
    )

    return parser


def main() -> int:
    """Main CLI execution handler."""
    parser = build_parser()
    args = parser.parse_args()

    # Load configuration
    cfg_file = getattr(args, "config", None)
    if cfg_file:
        config_path = Path(cfg_file)
        if not config_path.is_file():
            print(f"Error: Specified configuration file not found: {config_path}", file=sys.stderr)
            return 1
        config = Module2Config.from_json(config_path)
    else:
        config = Module2Config()

    if getattr(args, "device", None):
        config.device = args.device

    # Setup logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    log_file_path = Path(args.log_file) if getattr(args, "log_file", None) else config.log_file
    logger = setup_logging(level=log_level, log_file=log_file_path)

    # ------------------ Command: check-models ------------------
    if args.command == "check-models":
        import numpy as np
        import torch
        import transformers
        import sentence_transformers
        import huggingface_hub
        from embedding_store.embedders import (
            SigLIPVisualEmbedder,
            MiniLMTextEmbedder,
            resolve_local_model_path,
        )

        print("==================================================")
        print("Checking Offline Multimodal Model Encoders")
        print("==================================================")
        print(f"  PyTorch Version:             {torch.__version__}")
        print(f"  Transformers Version:        {transformers.__version__}")
        print(f"  SentenceTransformers:        {sentence_transformers.__version__}")
        print(f"  HuggingFace Hub Version:     {huggingface_hub.__version__}")
        print("--------------------------------------------------")

        pass_visual = False
        pass_text = False

        # 1. Visual Model Check
        print(f"Loading Visual Encoder '{config.visual_model_name}' (offline)...")
        resolved_vis = resolve_local_model_path(
            config.visual_model_name,
            cache_dir=str(config.models_dir) if config.models_dir else None,
            offline=True,
        )
        print(f"  Resolved Local Snapshot:     {resolved_vis}")
        try:
            vis_embedder = SigLIPVisualEmbedder(
                model_name=config.visual_model_name,
                device=config.device,
                models_dir=config.models_dir,
                offline_mode=True,
            )
            vis_vec = vis_embedder.embed_query_text("probe")
            print(f"  Visual Dim:                  {vis_embedder.dim}")
            print(f"  Probe Embedding Shape:       {vis_vec.shape}")
            vis_norm = float(np.linalg.norm(vis_vec))
            print(f"  Probe Vector L2 Norm:        {vis_norm:.4f}")
            print("  Visual Encoder:              OK")
            pass_visual = True
        except Exception as exc:
            print(f"  Visual Encoder FAILED:       {exc}")

        print("--------------------------------------------------")

        # 2. Text Model Check
        print(f"Loading Text Encoder '{config.text_model_name}' (offline)...")
        resolved_txt = resolve_local_model_path(
            config.text_model_name,
            cache_dir=str(config.models_dir) if config.models_dir else None,
            offline=True,
        )
        print(f"  Resolved Local Snapshot:     {resolved_txt}")
        try:
            txt_embedder = MiniLMTextEmbedder(
                model_name=config.text_model_name,
                device=config.device,
                models_dir=config.models_dir,
                offline_mode=True,
            )
            txt_vec = txt_embedder.embed_query_text("probe")
            print(f"  Text Dim:                    {txt_embedder.dim}")
            print(f"  Probe Embedding Shape:       {txt_vec.shape}")
            txt_norm = float(np.linalg.norm(txt_vec))
            print(f"  Probe Vector L2 Norm:        {txt_norm:.4f}")
            print("  Text Encoder:                OK")
            pass_text = True
        except Exception as exc:
            print(f"  Text Encoder FAILED:         {exc}")

        print("==================================================")
        if pass_visual and pass_text:
            print("check-models: PASS")
            return 0
        else:
            print("check-models: FAIL")
            return 1

    # ------------------ Command: download-models ------------------
    if args.command == "download-models":
        download_models(config)
        return 0

    # ------------------ Command: validate ------------------
    if args.command == "validate":
        passed = validate_integrity(config)
        return 0 if passed else 1

    # ------------------ Command: stats ------------------
    if args.command == "stats":
        with VideoIndex.open(config) as v_index:
            s = v_index.stats()
            print("==================================================")
            print("Module 2 Vector Database Summary Statistics")
            print("==================================================")
            print(f"  Indexed Videos:        {s['videos']}")
            print(f"  Visual Vectors:        {s['visual_vectors']} (FAISS: {s['faiss_visual_count']})")
            print(f"  Text Vectors:          {s['text_vectors']} (FAISS: {s['faiss_text_count']})")
            print(f"  Recorded Failures:     {s['failures']}")
            print(f"  Visual Model:          {s['visual_model']} (dim={s['visual_dim']})")
            print(f"  Text Model:            {s['text_model']} (dim={s['text_dim']})")
            print("==================================================")
        return 0

    # ------------------ Command: index ------------------
    if args.command == "index":
        if not args.all and not args.video_id and not args.rebuild and not args.prune_missing:
            # Default to --all if no specific target is provided
            args.all = True

        indexer = VideoIndexer(config=config)
        summary = indexer.run_indexing(
            video_id=args.video_id,
            force_reindex=args.force_reindex,
            rebuild=args.rebuild,
            prune_missing=args.prune_missing,
            visual_only=args.visual_only,
            text_only=args.text_only,
            save_each=args.save_each,
        )
        return 0

    # ------------------ Command: search ------------------
    if args.command == "search":
        vids_filter = [args.video_id] if args.video_id else None
        with VideoIndex.open(config) as v_index:
            if args.modality in ("visual", "both"):
                vis_hits = v_index.search_visual(
                    query=args.query,
                    top_k=args.top_k,
                    video_ids=vids_filter,
                )
                print("\n==================== Visual Search Results ====================")
                if not vis_hits:
                    print("  No visual matches found.")
                for rank, h in enumerate(vis_hits, 1):
                    print(
                        f"  #{rank} [Score: {h['score']:.4f}] Video: {h['video_filename']} ({h['video_id']})\n"
                        f"     Timestamp: {h['timestamp_formatted']} ({h['timestamp_seconds']:.2f}s) | Frame: {h['frame_id']}\n"
                        f"     Path:      {h['frame_path']}"
                    )

            if args.modality in ("text", "both"):
                txt_hits = v_index.search_text(
                    query=args.query,
                    top_k=args.top_k,
                    video_ids=vids_filter,
                )
                print("\n==================== Text Search Results ====================")
                if not txt_hits:
                    print("  No text matches found.")
                for rank, h in enumerate(txt_hits, 1):
                    print(
                        f"  #{rank} [Score: {h['score']:.4f}] Video: {h['video_filename']} ({h['video_id']})\n"
                        f"     Timestamp: {h['start_formatted']} -> {h['end_formatted']} ({h['start_timestamp']:.2f}s - {h['end_timestamp']:.2f}s)\n"
                        f"     Text:      \"{h['transcript_text']}\""
                    )
            print("===============================================================\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
