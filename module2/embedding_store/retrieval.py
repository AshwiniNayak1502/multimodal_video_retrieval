"""Query and Retrieval API for Module 3 and Downstream User Interfaces.

Provides the primary interface (VideoIndex):
- Cross-modal visual search (queries in natural language or image vectors).
- Speech transcript text search (queries in natural language or text vectors).
- Over-fetching filter for specific video_id candidate sets.
- Query-time dynamic path re-anchoring.
- Temporal cross-linking (get_frames_in_window and get_segments_in_window).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .config import Module2Config
from .embedders import (
    MiniLMTextEmbedder,
    SigLIPVisualEmbedder,
    TextEmbedderProtocol,
    VisualEmbedderProtocol,
)
from .logging_config import get_logger
from .metadata_store import INDEX_SCHEMA_VERSION, MetadataStore
from .vector_index import VectorIndexManager, load_or_rebuild_index, save_manifest

logger = get_logger("retrieval")


class VideoIndex:
    """Consolidated retrieval interface querying visual and text vector stores."""

    def __init__(
        self,
        config: Module2Config,
        store: MetadataStore,
        vis_mgr: VectorIndexManager,
        txt_mgr: VectorIndexManager,
        visual_embedder: Optional[VisualEmbedderProtocol] = None,
        text_embedder: Optional[TextEmbedderProtocol] = None,
    ) -> None:
        self.config = config
        self.store = store
        self.vis_mgr = vis_mgr
        self.txt_mgr = txt_mgr
        self._visual_embedder = visual_embedder
        self._text_embedder = text_embedder

    @classmethod
    def open(
        cls,
        config: Optional[Module2Config] = None,
        visual_embedder: Optional[VisualEmbedderProtocol] = None,
        text_embedder: Optional[TextEmbedderProtocol] = None,
    ) -> VideoIndex:
        """Open persistent storage, load FAISS indices once, and prepare for queries."""
        cfg = config or Module2Config()
        cfg.ensure_directories()

        db_path = cfg.vector_db_dir / "metadata.db"
        if not db_path.is_file():
            logger.warning("Metadata database does not exist at %s. Creating new empty database.", db_path)

        store = MetadataStore(db_path)

        vis_dim = (
            visual_embedder.dim
            if visual_embedder and hasattr(visual_embedder, "dim")
            else int(store.get_meta("visual_dim") or 768)
        )
        txt_dim = (
            text_embedder.dim
            if text_embedder and hasattr(text_embedder, "dim")
            else int(store.get_meta("text_dim") or 384)
        )

        vis_model = (
            visual_embedder.model_name
            if visual_embedder and hasattr(visual_embedder, "model_name")
            else store.get_meta("visual_model") or cfg.visual_model_name
        )
        txt_model = (
            text_embedder.model_name
            if text_embedder and hasattr(text_embedder, "model_name")
            else store.get_meta("text_model") or cfg.text_model_name
        )

        vis_index_file = cfg.vector_db_dir / "visual.index"
        txt_index_file = cfg.vector_db_dir / "text.index"
        manifest_file = cfg.vector_db_dir / "index_manifest.json"

        vis_mgr = load_or_rebuild_index(
            modality="visual",
            index_file=vis_index_file,
            manifest_file=manifest_file,
            store=store,
            expected_dim=vis_dim,
            expected_model=vis_model,
        )

        txt_mgr = load_or_rebuild_index(
            modality="text",
            index_file=txt_index_file,
            manifest_file=manifest_file,
            store=store,
            expected_dim=txt_dim,
            expected_model=txt_model,
        )

        if not manifest_file.is_file():
            manifest_data = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "models": {
                    "visual": vis_model,
                    "text": txt_model,
                },
                "dims": {
                    "visual": vis_dim,
                    "text": txt_dim,
                },
                "counts": {
                    "visual": vis_mgr.ntotal,
                    "text": txt_mgr.ntotal,
                },
                "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            save_manifest(manifest_file, manifest_data)

        return cls(
            config=cfg,
            store=store,
            vis_mgr=vis_mgr,
            txt_mgr=txt_mgr,
            visual_embedder=visual_embedder,
            text_embedder=text_embedder,
        )

    @property
    def visual_embedder(self) -> VisualEmbedderProtocol:
        """Lazy load visual embedder on first string query."""
        if self._visual_embedder is None:
            self._visual_embedder = SigLIPVisualEmbedder(
                model_name=self.config.visual_model_name,
                device=self.config.device,
                batch_size=self.config.visual_batch_size or 16,
                models_dir=self.config.models_dir,
                offline_mode=self.config.offline_mode,
            )
        return self._visual_embedder

    @property
    def text_embedder(self) -> TextEmbedderProtocol:
        """Lazy load text embedder on first string query."""
        if self._text_embedder is None:
            self._text_embedder = MiniLMTextEmbedder(
                model_name=self.config.text_model_name,
                device=self.config.device,
                batch_size=self.config.text_batch_size,
                models_dir=self.config.models_dir,
                offline_mode=self.config.offline_mode,
            )
        return self._text_embedder

    def close(self) -> None:
        """Close database connection."""
        self.store.close()

    def __enter__(self) -> VideoIndex:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _resolve_result_path(
        self,
        stored_path: str,
        relpath: Optional[str],
        base_dir: Path,
    ) -> str:
        """Verify if stored absolute path exists; fallback to base_dir / relpath."""
        p = Path(stored_path)
        if p.exists():
            return str(p.resolve())
        if relpath:
            candidate = base_dir / relpath
            if candidate.exists():
                return str(candidate.resolve())
        return stored_path

    def search_visual(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 10,
        video_ids: Optional[Sequence[str]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve visually similar frames matching text or image vector query."""
        if top_k <= 0:
            return []

        if isinstance(query, str):
            query_vec = self.visual_embedder.embed_query_text(query)
        else:
            query_vec = np.ascontiguousarray(query, dtype=np.float32)
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm

        target_vids = set(video_ids) if video_ids is not None else None

        # Over-fetching loop when filtering by video_ids
        fetch_k = max(top_k * 5, 50) if target_vids is not None else top_k
        max_available = self.vis_mgr.ntotal

        matching_hits: List[Dict[str, Any]] = []

        while True:
            actual_k = min(fetch_k, max_available)
            if actual_k <= 0:
                break

            scores, ids = self.vis_mgr.search(query_vec, actual_k)
            if len(ids) == 0:
                break

            valid_ids = [int(i) for i in ids if int(i) != -1]
            meta_map = self.store.get_visual_metadata_by_ids(valid_ids)

            matching_hits.clear()
            for vec_id, score in zip(valid_ids, scores):
                if min_score is not None and score < min_score:
                    continue

                row = meta_map.get(vec_id)
                if not row:
                    logger.warning("Visual vector_id %d not found in SQLite metadata. Dropping hit.", vec_id)
                    continue

                if target_vids is not None and row["video_id"] not in target_vids:
                    continue

                resolved_frame = self._resolve_result_path(
                    row["frame_path"], row["frame_relpath"], self.config.module1_output_dir
                )
                resolved_video = self._resolve_result_path(
                    row["video_path"], row["video_relpath"], self.config.module1_input_dir
                )

                hit = {
                    "modality": "visual",
                    "vector_id": int(vec_id),
                    "score": round(float(score), 4),
                    "video_id": row["video_id"],
                    "video_filename": row["video_filename"],
                    "video_path": resolved_video,
                    "frame_id": row["frame_id"],
                    "frame_index": int(row["frame_index"]),
                    "frame_path": resolved_frame,
                    "timestamp_seconds": float(row["timestamp_seconds"]),
                    "timestamp_formatted": row["timestamp_formatted"],
                    "frame_interval_seconds": float(row["frame_interval_seconds"]),
                    "embedding_model": row["embedding_model"],
                }
                matching_hits.append(hit)

                if len(matching_hits) >= top_k:
                    break

            if len(matching_hits) >= top_k or actual_k >= max_available:
                break

            # Grow fetch_k if not enough hits were collected
            fetch_k = min(fetch_k * 2, max_available)

        return matching_hits[:top_k]

    def search_text(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 10,
        video_ids: Optional[Sequence[str]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve semantically similar speech segments matching text or query vector."""
        if top_k <= 0:
            return []

        if isinstance(query, str):
            query_vec = self.text_embedder.embed_query_text(query)
        else:
            query_vec = np.ascontiguousarray(query, dtype=np.float32)
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm

        target_vids = set(video_ids) if video_ids is not None else None

        fetch_k = max(top_k * 5, 50) if target_vids is not None else top_k
        max_available = self.txt_mgr.ntotal

        matching_hits: List[Dict[str, Any]] = []

        while True:
            actual_k = min(fetch_k, max_available)
            if actual_k <= 0:
                break

            scores, ids = self.txt_mgr.search(query_vec, actual_k)
            if len(ids) == 0:
                break

            valid_ids = [int(i) for i in ids if int(i) != -1]
            meta_map = self.store.get_text_metadata_by_ids(valid_ids)

            matching_hits.clear()
            for vec_id, score in zip(valid_ids, scores):
                if min_score is not None and score < min_score:
                    continue

                row = meta_map.get(vec_id)
                if not row:
                    logger.warning("Text vector_id %d not found in SQLite metadata. Dropping hit.", vec_id)
                    continue

                if target_vids is not None and row["video_id"] not in target_vids:
                    continue

                resolved_video = self._resolve_result_path(
                    row["video_path"], row["video_relpath"], self.config.module1_input_dir
                )

                hit = {
                    "modality": "text",
                    "vector_id": int(vec_id),
                    "score": round(float(score), 4),
                    "video_id": row["video_id"],
                    "video_filename": row["video_filename"],
                    "video_path": resolved_video,
                    "segment_id": int(row["segment_id"]),
                    "transcript_text": row["transcript_text"],
                    "start_timestamp": float(row["start_timestamp"]),
                    "end_timestamp": float(row["end_timestamp"]),
                    "start_formatted": row["start_formatted"],
                    "end_formatted": row["end_formatted"],
                    "language": row["language"],
                    "embedding_model": row["embedding_model"],
                }
                matching_hits.append(hit)

                if len(matching_hits) >= top_k:
                    break

            if len(matching_hits) >= top_k or actual_k >= max_available:
                break

            fetch_k = min(fetch_k * 2, max_available)

        return matching_hits[:top_k]

    def get_frames_in_window(
        self, video_id: str, start: float, end: float
    ) -> List[Dict[str, Any]]:
        """Retrieve frames in a time range for cross-modal verification."""
        raw_items = self.store.get_frames_in_window(video_id, start, end)
        for item in raw_items:
            item["frame_path"] = self._resolve_result_path(
                item["frame_path"], item.get("frame_relpath"), self.config.module1_output_dir
            )
        return raw_items

    def get_segments_in_window(
        self, video_id: str, start: float, end: float
    ) -> List[Dict[str, Any]]:
        """Retrieve speech transcript segments overlapping a time window."""
        return self.store.get_segments_in_window(video_id, start, end)

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve video details by video_id."""
        row = self.store.get_video(video_id)
        if row:
            row["video_path"] = self._resolve_result_path(
                row["video_path"], row.get("video_relpath"), self.config.module1_input_dir
            )
        return row

    def list_videos(self) -> List[Dict[str, Any]]:
        """List all indexed video records with resolved paths."""
        videos = self.store.list_videos()
        for v in videos:
            v["video_path"] = self._resolve_result_path(
                v["video_path"], v.get("video_relpath"), self.config.module1_input_dir
            )
        return videos

    def stats(self) -> Dict[str, Any]:
        """Return system statistics including FAISS vector counts."""
        db_stats = self.store.stats()
        db_stats["faiss_visual_count"] = self.vis_mgr.ntotal
        db_stats["faiss_text_count"] = self.txt_mgr.ntotal
        return db_stats
