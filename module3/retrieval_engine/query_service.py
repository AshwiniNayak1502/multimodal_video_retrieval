"""QueryService: Unified multi-modal semantic search, calibration, and timestamped retrieval."""

from __future__ import annotations

import collections
import json
import math
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure offline environment is configured first
from . import offline_env  # noqa: F401

# Add Module 2 to sys.path to enable direct imports
_MODULE2_DIR = (Path(__file__).resolve().parent.parent.parent / "module2").resolve()
if str(_MODULE2_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE2_DIR))

from embedding_store.config import Module2Config
from embedding_store.retrieval import VideoIndex

from .calibration import (
    calibrate_text_score,
    calibrate_visual_score,
    extract_siglip_calibration_parameters,
    is_text_relevant,
    is_visual_relevant,
)
from .config import Module3Config
from .errors import (
    IndexNotReadyError,
    ModelMismatchError,
    RetrievalEngineError,
    ValidationError,
)
from .logging_config import get_logger
from .segments import (
    EvidenceHit,
    Result,
    assemble_and_deduplicate_segments,
    build_approx_visual_span,
)
from .timing import SearchTimings, Timer

logger = get_logger("query_service")

SCHEMA_VERSION = "3.1.0"


@dataclass
class SearchResponse:
    """Standardized response contract for QueryService search requests."""

    schema_version: str
    status: str  # "ok" | "no_match" | "index_not_ready" | "error"
    message: str
    results: List[Result]
    timings_ms: Dict[str, float]
    debug: Dict[str, Any]
    notices: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResponse to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "message": self.message,
            "results": [r.to_dict() for r in self.results],
            "timings_ms": self.timings_ms,
            "debug": self.debug,
            "notices": list(self.notices),
        }


class QueryService:
    """Thread-safe, process-singleton query processor and segment retriever."""

    def __init__(
        self,
        config: Optional[Module3Config] = None,
        video_index: Optional[VideoIndex] = None,
        warmup: bool = True,
    ) -> None:
        self.config = config or Module3Config()
        self.config.ensure_directories()

        self._lock = threading.RLock()
        self._visual_query_cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()
        self._text_query_cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()

        self._visual_encoder_status: Dict[str, Any] = {"ready": True, "error": None}
        self._text_encoder_status: Dict[str, Any] = {"ready": True, "error": None}
        self._warmup_error: Optional[str] = None
        self._query_cache_hits_visual: int = 0
        self._query_cache_hits_text: int = 0

        self._models_reused = False
        self._index_reused = False

        if video_index is not None:
            self.index = video_index
        else:
            m2_cfg = Module2Config(
                base_dir=self.config.module2_dir,
                output_dir=self.config.module2_dir / "output",
                vector_db_dir=self.config.vector_db_dir,
                offline_mode=True,
            )
            logger.info("Initializing VideoIndex from %s...", self.config.vector_db_dir)
            self.index = VideoIndex.open(m2_cfg)

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Warm up query encoders once at startup to eliminate cold-start latency."""
        logger.info("Warming up visual and text query encoders...")
        probe = "warmup query"
        vis_err = None
        txt_err = None

        if hasattr(self.index, "vis_mgr") and self.index.vis_mgr.ntotal > 0:
            try:
                _ = self.index.visual_embedder.embed_query_text(probe)
                self._visual_encoder_status = {"ready": True, "error": None}
            except Exception as exc:
                vis_err = repr(exc)
                logger.error("Warmup failed for visual encoder: %s\n%s", vis_err, traceback.format_exc())
                self._visual_encoder_status = {"ready": False, "error": vis_err}

        if hasattr(self.index, "txt_mgr") and self.index.txt_mgr.ntotal > 0:
            try:
                _ = self.index.text_embedder.embed_query_text(probe)
                self._text_encoder_status = {"ready": True, "error": None}
            except Exception as exc:
                txt_err = repr(exc)
                logger.error("Warmup failed for text encoder: %s\n%s", txt_err, traceback.format_exc())
                self._text_encoder_status = {"ready": False, "error": txt_err}

        if vis_err or txt_err:
            self._warmup_error = f"Warmup errors: visual={vis_err}, text={txt_err}"
            logger.error("Query encoder warmup completed with failures: %s", self._warmup_error)
        else:
            logger.info("Query encoders warmed up successfully.")

    def reload_index(self) -> None:
        """Reload VideoIndex under the lock to reflect new indexing runs without restarting."""
        with self._lock:
            logger.info("Reloading VideoIndex from persistent storage...")
            curr_vis_emb = getattr(self.index, "_visual_embedder", None)
            curr_txt_emb = getattr(self.index, "_text_embedder", None)
            self.index.close()
            m2_cfg = Module2Config(
                base_dir=self.config.module2_dir,
                output_dir=self.config.module2_dir / "output",
                vector_db_dir=self.config.vector_db_dir,
                offline_mode=True,
            )
            self.index = VideoIndex.open(
                m2_cfg,
                visual_embedder=curr_vis_emb,
                text_embedder=curr_txt_emb,
            )
            self._visual_query_cache.clear()
            self._text_query_cache.clear()
            logger.info("VideoIndex reloaded successfully.")

    def close(self) -> None:
        """Close index connection and release resources."""
        with self._lock:
            self.index.close()

    def __enter__(self) -> QueryService:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _list_videos_unlocked(self) -> List[Dict[str, Any]]:
        """Internal helper to list videos and counts without acquiring self._lock."""
        videos = self.index.list_videos()
        result: List[Dict[str, Any]] = []
        for v in videos:
            vid = v.get("video_id", "")
            cur = self.index.store.connection.cursor()
            cur.execute("SELECT COUNT(*) AS c FROM visual_vectors WHERE video_id = ?", (vid,))
            v_count = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) AS c FROM text_vectors WHERE video_id = ?", (vid,))
            t_count = cur.fetchone()["c"]
            v_path = v.get("video_path", "")
            has_audio = bool(v.get("has_audio", False))
            no_speech_detected = bool(has_audio and t_count == 0)
            has_zero_vectors = bool(v_count == 0)
            result.append({
                "video_id": vid,
                "filename": v.get("video_filename", ""),
                "duration_seconds": float(v.get("duration_seconds", 0.0)),
                "has_audio": has_audio,
                "visual_vector_count": int(v_count),
                "text_vector_count": int(t_count),
                "video_path": v_path,
                "exists_on_disk": Path(v_path).is_file() if v_path else False,
                "has_zero_vectors": has_zero_vectors,
                "no_speech_detected": no_speech_detected,
            })
        return result

    def list_videos(self) -> List[Dict[str, Any]]:
        """Return all indexed videos with metadata and path resolution."""
        with self._lock:
            return self._list_videos_unlocked()

    def health(self) -> Dict[str, Any]:
        """Report index readiness, vector counts, model information, and per-encoder readiness."""
        with self._lock:
            stats = self.index.stats()
            vid_count = stats.get("indexed_videos") or stats.get("videos") or 0
            vis_count = stats.get("faiss_visual_count", 0) or stats.get("visual_vectors", 0)
            txt_count = stats.get("faiss_text_count", 0) or stats.get("text_vectors", 0)

            # Check for videos with zero vectors or no speech detected
            zero_vector_vids: List[str] = []
            no_speech_vids: List[str] = []
            for v in self._list_videos_unlocked():
                if v.get("has_zero_vectors"):
                    zero_vector_vids.append(v["video_id"])
                if v.get("no_speech_detected"):
                    no_speech_vids.append(v["video_id"])

            is_ready = vid_count > 0 and (vis_count > 0 or txt_count > 0)
            status_str = "ready" if is_ready else "not_ready"
            if is_ready and zero_vector_vids:
                status_str = "degraded"

            msg = (
                f"Index ready: {vid_count} video(s), {vis_count} visual vector(s), {txt_count} text vector(s)."
                if is_ready
                else "Index is empty or not initialized. Run 'python main.py index --all' in module2 first."
            )
            if zero_vector_vids:
                msg += f" WARNING: Videos with missing vectors detected: {zero_vector_vids}. Run 'python main.py index --all' to heal."
            if no_speech_vids:
                msg += f" (Note: No speech detected in {len(no_speech_vids)} audio track(s))."

            vis_emb = getattr(self.index, "visual_embedder", None)
            txt_emb = getattr(self.index, "text_embedder", None)

            return {
                "status": status_str,
                "message": msg,
                "indexed_videos": vid_count,
                "visual_vectors": vis_count,
                "text_vectors": txt_count,
                "visual_model": stats.get("visual_model", ""),
                "text_model": stats.get("text_model", ""),
                "encoders": {
                    "visual": {
                        "model": getattr(vis_emb, "model_name", ""),
                        "ready": self._visual_encoder_status.get("ready", True),
                        "error": self._visual_encoder_status.get("error"),
                        "load_count": getattr(vis_emb, "load_count", 0),
                    },
                    "text": {
                        "model": getattr(txt_emb, "model_name", ""),
                        "ready": self._text_encoder_status.get("ready", True),
                        "error": self._text_encoder_status.get("error"),
                        "load_count": getattr(txt_emb, "load_count", 0),
                    },
                },
                "videos_with_zero_vectors": zero_vector_vids,
                "videos_without_speech": no_speech_vids,
                "schema_version": SCHEMA_VERSION,
            }

    def get_frame_path(self, frame_id: str) -> Optional[str]:
        """Resolve and verify absolute local frame path by frame_id."""
        with self._lock:
            cur = self.index.store.connection.cursor()
            cur.execute(
                "SELECT frame_path, frame_relpath FROM visual_vectors WHERE frame_id = ? LIMIT 1",
                (frame_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self.index._resolve_result_path(
                row["frame_path"], row["frame_relpath"], self.index.config.module1_output_dir
            )

    def validate_inputs(
        self,
        query: Any,
        top_k: Any,
        video_ids: Optional[Sequence[str]],
        modality_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Validate search inputs according to strict system specification."""
        if not isinstance(query, str) or not query.strip():
            raise ValidationError(
                "Query string cannot be empty or whitespace only.",
                user_friendly_message="Please enter a valid search query.",
            )

        if len(query) > self.config.max_query_chars:
            raise ValidationError(
                f"Query exceeds maximum character limit of {self.config.max_query_chars}.",
                user_friendly_message=f"Query is too long (maximum {self.config.max_query_chars} characters).",
            )

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValidationError(
                f"Invalid top_k: {top_k}. Must be an integer between {self.config.min_top_k} and {self.config.max_top_k} inclusive.",
                user_friendly_message=f"top_k must be an integer between {self.config.min_top_k} and {self.config.max_top_k}.",
            )

        if top_k < self.config.min_top_k or top_k > self.config.max_top_k:
            raise ValidationError(
                f"top_k={top_k} is out of bounds. Must be between {self.config.min_top_k} and {self.config.max_top_k} inclusive.",
                user_friendly_message=f"top_k must be between {self.config.min_top_k} and {self.config.max_top_k}.",
            )

        if modality_weights is not None:
            if not isinstance(modality_weights, dict):
                raise ValidationError(
                    "modality_weights must be a dictionary with 'visual' and/or 'text' keys.",
                    user_friendly_message="Invalid modality weights format.",
                )
            vw = modality_weights.get("visual", self.config.visual_weight)
            tw = modality_weights.get("text", self.config.text_weight)
            for m_name, w in [("visual", vw), ("text", tw)]:
                if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w) or w < 0:
                    raise ValidationError(
                        f"Modality weight for '{m_name}' must be a finite, non-negative number. Got: {w}",
                        user_friendly_message=f"Modality weight for '{m_name}' must be a non-negative number.",
                    )
            if float(vw) == 0.0 and float(tw) == 0.0:
                raise ValidationError(
                    "At least one modality weight must be greater than 0.",
                    user_friendly_message="Visual and text weights cannot both be zero.",
                )

        if video_ids is not None:
            if not isinstance(video_ids, (list, tuple, set)):
                raise ValidationError("video_ids must be a list or sequence of string IDs.")

            # Verify that all requested video_ids exist in metadata database
            known_videos = {v["video_id"] for v in self.index.store.list_videos()}
            for vid in video_ids:
                if not isinstance(vid, str):
                    raise ValidationError(f"Invalid video_id type: {type(vid)}. Must be string.")
                if vid not in known_videos:
                    raise ValidationError(
                        f"Unknown video_id '{vid}'. Indexed videos: {sorted(known_videos)}.",
                        user_friendly_message=f"Video ID '{vid}' was not found in the indexed library.",
                    )

    def _preflight_check(self) -> None:
        """Check index readiness, vector population, and model consistency."""
        stats = self.index.stats()
        indexed_videos = stats.get("indexed_videos") or stats.get("videos") or 0
        vis_count = stats.get("faiss_visual_count", 0) or stats.get("visual_vectors", 0)
        txt_count = stats.get("faiss_text_count", 0) or stats.get("text_vectors", 0)

        if indexed_videos == 0 or (vis_count == 0 and txt_count == 0):
            raise IndexNotReadyError(
                "Index is empty or not initialized. Run 'python main.py index --all' in module2 first.",
                user_friendly_message="Index is empty or not initialized. Run 'python main.py index --all' in module2 first.",
            )

        # Check manifest models & dimensions against loaded embedders if manifest is present
        manifest_file = self.config.vector_db_dir / "index_manifest.json"
        if manifest_file.is_file():
            import json
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest_models = manifest.get("models", {})
                manifest_dims = manifest.get("dims", {})

                # Check visual model and dim
                stored_vm = manifest_models.get("visual")
                stored_vd = manifest_dims.get("visual")
                if hasattr(self.index, "visual_embedder") and self.index.visual_embedder is not None:
                    curr_vm = self.index.visual_embedder.model_name
                    curr_vd = self.index.visual_embedder.dim
                    if stored_vm and curr_vm and stored_vm != curr_vm:
                        raise ModelMismatchError(
                            f"Visual model mismatch: manifest expects '{stored_vm}', active embedder is '{curr_vm}'.",
                            user_friendly_message=f"Model mismatch detected: stored index uses '{stored_vm}' but runtime uses '{curr_vm}'.",
                        )
                    if stored_vd and curr_vd and int(stored_vd) != int(curr_vd):
                        raise ModelMismatchError(
                            f"Visual dimension mismatch: manifest expects {stored_vd}, active embedder has {curr_vd}.",
                            user_friendly_message=f"Embedding dimension mismatch: stored index has {stored_vd} but runtime has {curr_vd}.",
                        )

                # Check text model and dim
                stored_tm = manifest_models.get("text")
                stored_td = manifest_dims.get("text")
                if hasattr(self.index, "text_embedder") and self.index.text_embedder is not None:
                    curr_tm = self.index.text_embedder.model_name
                    curr_td = self.index.text_embedder.dim
                    if stored_tm and curr_tm and stored_tm != curr_tm:
                        raise ModelMismatchError(
                            f"Text model mismatch: manifest expects '{stored_tm}', active embedder is '{curr_tm}'.",
                            user_friendly_message=f"Model mismatch detected: stored index uses '{stored_tm}' but runtime uses '{curr_tm}'.",
                        )
                    if stored_td and curr_td and int(stored_td) != int(curr_td):
                        raise ModelMismatchError(
                            f"Text dimension mismatch: manifest expects {stored_td}, active embedder has {curr_td}.",
                            user_friendly_message=f"Embedding dimension mismatch: stored index has {stored_td} but runtime has {curr_td}.",
                        )
            except (ModelMismatchError, IndexNotReadyError):
                raise
            except Exception as exc:
                logger.warning("Manifest preflight check warning: %s", exc)

    def _get_query_embedding_visual(self, query: str) -> Tuple[np.ndarray, bool]:
        """Retrieve visual query vector with LRU caching."""
        normalized = query.strip().lower()
        if normalized in self._visual_query_cache:
            self._visual_query_cache.move_to_end(normalized)
            self._query_cache_hits_visual += 1
            return self._visual_query_cache[normalized], True

        try:
            vec = self.index.visual_embedder.embed_query_text(query)
            self._visual_encoder_status = {"ready": True, "error": None}
        except Exception as exc:
            self._visual_encoder_status = {"ready": False, "error": repr(exc)}
            raise

        self._visual_query_cache[normalized] = vec
        if len(self._visual_query_cache) > self.config.query_cache_size:
            self._visual_query_cache.popitem(last=False)
        return vec, False

    def _get_query_embedding_text(self, query: str) -> Tuple[np.ndarray, bool]:
        """Retrieve text query vector with LRU caching."""
        normalized = query.strip().lower()
        if normalized in self._text_query_cache:
            self._text_query_cache.move_to_end(normalized)
            self._query_cache_hits_text += 1
            return self._text_query_cache[normalized], True

        try:
            vec = self.index.text_embedder.embed_query_text(query)
            self._text_encoder_status = {"ready": True, "error": None}
        except Exception as exc:
            self._text_encoder_status = {"ready": False, "error": repr(exc)}
            raise

        self._text_query_cache[normalized] = vec
        if len(self._text_query_cache) > self.config.query_cache_size:
            self._text_query_cache.popitem(last=False)
        return vec, False

    def search(
        self,
        query: str,
        top_k: int = 5,
        video_ids: Optional[Sequence[str]] = None,
        modality_weights: Optional[Dict[str, float]] = None,
        expect_segment: Optional[str] = None,
    ) -> SearchResponse:
        """Execute multimodal semantic query search and timestamped segment retrieval."""
        timings = SearchTimings()
        trace: Dict[str, Any] = {
            "query_received": query,
            "query_normalised": query.strip().lower() if isinstance(query, str) else "",
            "modalities": {},
            "top_raw_hits": {},
            "gate_decisions": {},
            "linking_and_fusion": [],
            "final_ranking": [],
        }

        vis_loads = getattr(getattr(self.index, "visual_embedder", None), "load_count", 0)
        txt_loads = getattr(getattr(self.index, "text_embedder", None), "load_count", 0)

        notices: List[str] = []
        debug_info: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "video_ids": list(video_ids) if video_ids else None,
            "cache_hit_visual": False,
            "cache_hit_text": False,
            "cache_hits_visual": self._query_cache_hits_visual,
            "cache_hits_text": self._query_cache_hits_text,
            "visual_load_count": vis_loads,
            "text_load_count": txt_loads,
            "models_reused": self._models_reused,
            "index_reused": self._index_reused,
            "degraded_modalities": [],
            "best_rejected_candidates": [],
            "trace": trace,
        }

        with self._lock:
            # Step 1: Input Validation
            try:
                self.validate_inputs(query, top_k, video_ids, modality_weights)
            except ValidationError as val_err:
                logger.warning("Validation error on search request: %s", val_err)
                return SearchResponse(
                    schema_version=SCHEMA_VERSION,
                    status="error",
                    message=val_err.user_friendly_message,
                    results=[],
                    timings_ms=timings.to_dict(),
                    debug=debug_info,
                    notices=notices,
                )

            # Step 2: Preflight Verification
            try:
                self._preflight_check()
            except IndexNotReadyError as inr:
                logger.info("Search refused: index not ready (%s)", inr)
                return SearchResponse(
                    schema_version=SCHEMA_VERSION,
                    status="index_not_ready",
                    message=inr.user_friendly_message,
                    results=[],
                    timings_ms=timings.to_dict(),
                    debug=debug_info,
                    notices=notices,
                )
            except (ModelMismatchError, RetrievalEngineError) as err:
                logger.error("Preflight error on search request: %s", err)
                return SearchResponse(
                    schema_version=SCHEMA_VERSION,
                    status="error",
                    message=err.user_friendly_message,
                    results=[],
                    timings_ms=timings.to_dict(),
                    debug=debug_info,
                    notices=notices,
                )
            except Exception as exc:
                logger.exception("Unexpected error during search preflight: %s", exc)
                return SearchResponse(
                    schema_version=SCHEMA_VERSION,
                    status="error",
                    message="An unexpected system error occurred during retrieval preflight.",
                    results=[],
                    timings_ms=timings.to_dict(),
                    debug=debug_info,
                    notices=notices,
                )

            clean_query = query.strip()
            vis_weights = (
                modality_weights.get("visual", self.config.visual_weight)
                if modality_weights
                else self.config.visual_weight
            )
            txt_weights = (
                modality_weights.get("text", self.config.text_weight)
                if modality_weights
                else self.config.text_weight
            )

            with Timer() as total_timer:
                # Step 3: Query Encoding
                vis_cached = False
                txt_cached = False
                vis_vec = None
                txt_vec = None
                vis_err = None
                txt_err = None

                # Visual query encoding
                with Timer() as q_vis_timer:
                    try:
                        vis_vec, vis_cached = self._get_query_embedding_visual(clean_query)
                    except Exception as exc:
                        vis_name = getattr(getattr(self.index, "visual_embedder", None), "model_name", "google/siglip2-base-patch16-224")
                        vis_err = f"Failed to encode visual query with model '{vis_name}': {repr(exc)}"
                        logger.error("Visual query embedding failed: %s\n%s", repr(exc), traceback.format_exc())
                timings.query_encode_visual_ms = q_vis_timer.elapsed_ms

                # Text query encoding
                with Timer() as q_txt_timer:
                    try:
                        txt_vec, txt_cached = self._get_query_embedding_text(clean_query)
                    except Exception as exc:
                        txt_name = getattr(getattr(self.index, "text_embedder", None), "model_name", "sentence-transformers/all-MiniLM-L6-v2")
                        txt_err = f"Failed to encode text query with model '{txt_name}': {repr(exc)}"
                        logger.error("Text query embedding failed: %s\n%s", repr(exc), traceback.format_exc())
                timings.query_encode_text_ms = q_txt_timer.elapsed_ms

                debug_info["cache_hit_visual"] = vis_cached
                debug_info["cache_hit_text"] = txt_cached
                debug_info["cache_hits_visual"] = self._query_cache_hits_visual
                debug_info["cache_hits_text"] = self._query_cache_hits_text

                # Check encoder failure modes (R1)
                if vis_err is not None and txt_err is not None:
                    error_msg = "Both query encoders failed to process the request. Please check system logs."
                    logger.error("Both query encoders failed:\n- Visual: %s\n- Text: %s", vis_err, txt_err)
                    debug_info["degraded_modalities"] = ["visual", "text"]
                    debug_info["encoder_errors"] = {"visual": vis_err, "text": txt_err}
                    trace["modalities"] = {
                        "visual": {"searched": False, "reason": vis_err, "candidate_count": 0},
                        "text": {"searched": False, "reason": txt_err, "candidate_count": 0},
                    }
                    return SearchResponse(
                        schema_version=SCHEMA_VERSION,
                        status="error",
                        message=error_msg,
                        results=[],
                        timings_ms=timings.to_dict(),
                        debug=debug_info,
                        notices=notices,
                    )

                degraded = []
                if vis_err is not None:
                    degraded.append("visual")
                    debug_info["warning"] = "Visual search encoder degraded."
                    debug_info["visual_error"] = vis_err
                    notices.append("Visual search unavailable for this query; transcript results only.")
                    logger.warning("Degraded search (visual encoder): %s", vis_err)
                if txt_err is not None:
                    degraded.append("text")
                    debug_info["warning"] = "Text search encoder degraded."
                    debug_info["text_error"] = txt_err
                    notices.append("Transcript search unavailable for this query; visual results only.")
                    logger.warning("Degraded search (text encoder): %s", txt_err)
                debug_info["degraded_modalities"] = degraded

                # Step 4: Candidate Retrieval per Modality
                visual_available = self.index.vis_mgr.ntotal > 0 and vis_vec is not None
                text_available = self.index.txt_mgr.ntotal > 0 and txt_vec is not None

                if video_ids:
                    try:
                        cur = self.index.store.connection.cursor()
                        placeholders = ",".join("?" for _ in video_ids)
                        cur.execute(f"SELECT COUNT(*) FROM visual_vectors WHERE video_id IN ({placeholders})", tuple(video_ids))
                        vis_row = cur.fetchone()
                        if vis_row and vis_row[0] == 0:
                            visual_available = False
                        cur.execute(f"SELECT COUNT(*) FROM text_vectors WHERE video_id IN ({placeholders})", tuple(video_ids))
                        txt_row = cur.fetchone()
                        if txt_row and txt_row[0] == 0:
                            text_available = False
                    except Exception as e:
                        logger.debug("Failed checking video_ids modality availability: %s", e)

                if not visual_available and vis_err is None:
                    notices.append("Visual search unavailable for this query; transcript results only.")
                if not text_available and txt_err is None:
                    notices.append("Transcript search unavailable for this query; visual results only.")

                debug_info["modality_available"] = {
                    "visual": visual_available,
                    "text": text_available,
                }

                visual_raw_hits: List[Dict[str, Any]] = []
                text_raw_hits: List[Dict[str, Any]] = []

                with Timer() as s_vis_timer:
                    if visual_available and vis_vec is not None:
                        visual_raw_hits = self.index.search_visual(
                            query=vis_vec,
                            top_k=self.config.visual_pool,
                            video_ids=video_ids,
                        )
                timings.search_visual_ms = s_vis_timer.elapsed_ms

                with Timer() as s_txt_timer:
                    if text_available and txt_vec is not None:
                        text_raw_hits = self.index.search_text(
                            query=txt_vec,
                            top_k=self.config.text_pool,
                            video_ids=video_ids,
                        )
                timings.search_text_ms = s_txt_timer.elapsed_ms

                debug_info["candidates_retrieved"] = {
                    "visual_pool_count": len(visual_raw_hits),
                    "text_pool_count": len(text_raw_hits),
                }

                trace["modalities"] = {
                    "visual": {
                        "searched": visual_available,
                        "reason": "active" if visual_available else (vis_err or "no_vectors_in_index"),
                        "candidate_count": len(visual_raw_hits),
                    },
                    "text": {
                        "searched": text_available,
                        "reason": "active" if text_available else (txt_err or "no_vectors_in_index"),
                        "candidate_count": len(text_raw_hits),
                    },
                }

                # Step 5: Score Calibration & Relevance Gating (Single Gating)
                with Timer() as cal_timer:
                    # Video metadata cache to avoid up to 50 redundant SQLite queries (Item 10)
                    video_meta_cache: Dict[str, Dict[str, Any]] = {}

                    def get_cached_video(vid: str) -> Dict[str, Any]:
                        if vid not in video_meta_cache:
                            v = self.index.get_video(vid)
                            video_meta_cache[vid] = v if v else {}
                        return video_meta_cache[vid]

                    # Extract SigLIP model calibration parameters if available
                    logit_scale, logit_bias = None, None
                    if hasattr(self.index, "visual_embedder"):
                        params = extract_siglip_calibration_parameters(self.index.visual_embedder)
                        if params is not None:
                            logit_scale, logit_bias = params

                    calibrated_v_hits: List[EvidenceHit] = []
                    calibrated_t_hits: List[EvidenceHit] = []
                    rejected_candidates: List[Dict[str, Any]] = []

                    raw_v_trace: List[Dict[str, Any]] = []
                    for rank_idx, hit in enumerate(visual_raw_hits, start=1):
                        raw_c = float(hit["score"])
                        cal_prob = calibrate_visual_score(
                            raw_cosine=raw_c,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            fallback_scale=self.config.visual_fallback_scale,
                            fallback_bias=self.config.visual_fallback_bias,
                        )
                        t_sec = float(hit["timestamp_seconds"])
                        interval_s = float(hit.get("frame_interval_seconds", 2.0))
                        video_row = get_cached_video(hit["video_id"])
                        dur_s = float(video_row.get("duration_seconds", 0.0))
                        s_sec, e_sec = build_approx_visual_span(t_sec, interval_s, dur_s)

                        if rank_idx <= 5:
                            raw_v_trace.append({
                                "vector_id": hit.get("vector_id"),
                                "frame_id": hit.get("frame_id"),
                                "video_filename": hit.get("video_filename"),
                                "timestamp_seconds": t_sec,
                                "span": [s_sec, e_sec],
                                "raw_cosine": round(raw_c, 4),
                                "calibrated_score": round(cal_prob, 4),
                            })

                        if is_visual_relevant(cal_prob, self.config.visual_min_prob):
                            calibrated_v_hits.append(
                                EvidenceHit(
                                    modality="visual",
                                    video_id=hit["video_id"],
                                    video_filename=hit["video_filename"],
                                    video_path=hit["video_path"],
                                    score=cal_prob,
                                    raw_score=raw_c,
                                    start_seconds=s_sec,
                                    end_seconds=e_sec,
                                    rank=rank_idx,
                                    frame_id=hit["frame_id"],
                                    frame_index=hit["frame_index"],
                                    frame_path=hit["frame_path"],
                                    matched_at_seconds=t_sec,
                                    frame_interval_seconds=interval_s,
                                )
                            )
                        else:
                            rejected_candidates.append({
                                "modality": "visual",
                                "video_id": hit["video_id"],
                                "video_filename": hit["video_filename"],
                                "timestamp_seconds": t_sec,
                                "raw_cosine": round(raw_c, 4),
                                "calibrated_score": round(cal_prob, 4),
                                "gate": "visual_min_prob",
                                "threshold": self.config.visual_min_prob,
                                "rejection_reason": f"calibrated_prob ({cal_prob:.4f}) < visual_min_prob ({self.config.visual_min_prob:.4f})",
                            })

                    raw_t_trace: List[Dict[str, Any]] = []
                    for rank_idx, hit in enumerate(text_raw_hits, start=1):
                        raw_c = float(hit["score"])
                        cal_score = calibrate_text_score(
                            raw_cosine=raw_c,
                            floor=self.config.text_floor,
                            ceiling=self.config.text_ceiling,
                        )
                        st_s = float(hit["start_timestamp"])
                        et_s = float(hit["end_timestamp"])

                        if rank_idx <= 5:
                            raw_t_trace.append({
                                "vector_id": hit.get("vector_id"),
                                "segment_id": hit.get("segment_id"),
                                "video_filename": hit.get("video_filename"),
                                "start_timestamp": st_s,
                                "end_timestamp": et_s,
                                "raw_cosine": round(raw_c, 4),
                                "calibrated_score": round(cal_score, 4),
                                "transcript_text": hit.get("transcript_text", "")[:60],
                            })

                        if is_text_relevant(raw_c, self.config.text_min_cosine):
                            calibrated_t_hits.append(
                                EvidenceHit(
                                    modality="text",
                                    video_id=hit["video_id"],
                                    video_filename=hit["video_filename"],
                                    video_path=hit["video_path"],
                                    score=cal_score,
                                    raw_score=raw_c,
                                    start_seconds=st_s,
                                    end_seconds=et_s,
                                    rank=rank_idx,
                                    segment_id=hit["segment_id"],
                                    transcript_text=hit["transcript_text"],
                                )
                            )
                        else:
                            rejected_candidates.append({
                                "modality": "text",
                                "video_id": hit["video_id"],
                                "video_filename": hit["video_filename"],
                                "start_timestamp": st_s,
                                "end_timestamp": et_s,
                                "raw_cosine": round(raw_c, 4),
                                "calibrated_score": round(cal_score, 4),
                                "gate": "text_min_cosine",
                                "threshold": self.config.text_min_cosine,
                                "rejection_reason": f"raw_cosine ({raw_c:.4f}) < text_min_cosine ({self.config.text_min_cosine:.4f})",
                            })

                    trace["top_raw_hits"] = {
                        "visual": raw_v_trace,
                        "text": raw_t_trace,
                    }

                    debug_info["candidates_passed_gating"] = {
                        "visual_passed": len(calibrated_v_hits),
                        "text_passed": len(calibrated_t_hits),
                    }

                    trace["gate_decisions"] = {
                        "visual_gate": {
                            "metric": "calibrated_prob",
                            "threshold": self.config.visual_min_prob,
                            "passed_count": len(calibrated_v_hits),
                            "rejected_count": len(visual_raw_hits) - len(calibrated_v_hits),
                        },
                        "text_gate": {
                            "metric": "raw_cosine",
                            "threshold": self.config.text_min_cosine,
                            "passed_count": len(calibrated_t_hits),
                            "rejected_count": len(text_raw_hits) - len(calibrated_t_hits),
                        },
                        "rejected_candidates": rejected_candidates[:10],
                    }

                    # Keep best rejected candidates for explainability
                    debug_info["best_rejected_candidates"] = rejected_candidates[:5]
                timings.calibrate_fuse_ms = cal_timer.elapsed_ms

                # Step 6: Segment Linking, Fusion, Dedup, and Ranking
                with Timer() as seg_timer:
                    target_vids = {h.video_id for h in calibrated_v_hits} | {
                        h.video_id for h in calibrated_t_hits
                    }
                    meta_map: Dict[str, Dict[str, Any]] = {
                        vid: get_cached_video(vid) for vid in target_vids
                    }

                    all_results = assemble_and_deduplicate_segments(
                        visual_hits=calibrated_v_hits,
                        text_hits=calibrated_t_hits,
                        video_meta_map=meta_map,
                        visual_weight=vis_weights,
                        text_weight=txt_weights,
                        agreement_bonus=self.config.agreement_bonus,
                        final_min_score=self.config.final_min_score,
                        use_rrf=self.config.use_rrf_tiebreaker,
                        rrf_k=self.config.rrf_k,
                        link_tolerance_s=self.config.link_tolerance_s,
                        seek_offset_s=self.config.seek_offset_s,
                        fusion_mode=self.config.fusion_mode,
                    )

                    # Top-K slicing
                    final_results = all_results[:top_k]

                    trace["linking_and_fusion"] = [
                        {
                            "video_filename": r.video_filename,
                            "span": [round(r.start_seconds, 2), round(r.end_seconds, 2)],
                            "span_type": r.span_type,
                            "visual_score": round(r.visual_score, 4) if r.visual_score is not None else None,
                            "text_score": round(r.text_score, 4) if r.text_score is not None else None,
                            "final_fused_score": round(r.final_score, 4),
                            "rrf_score": round(r.rrf_score, 4) if r.rrf_score is not None else None,
                            "reason": r.reason,
                        }
                        for r in all_results
                    ]

                    trace["final_ranking"] = [
                        {
                            "rank": idx,
                            "video_filename": r.video_filename,
                            "start_seconds": round(r.start_seconds, 2),
                            "end_seconds": round(r.end_seconds, 2),
                            "final_score": round(r.final_score, 4),
                            "modalities": r.modalities,
                            "span_type": r.span_type,
                        }
                        for idx, r in enumerate(final_results, start=1)
                    ]

                timings.segment_build_ms = seg_timer.elapsed_ms

                timings.total_ms = total_timer.elapsed_ms

                # Step 7: Expectation Evaluation (R10)
                if expect_segment:
                    try:
                        exp_file, exp_range = expect_segment.split(":")
                        exp_s, exp_e = [float(x) for x in exp_range.split("-")]
                        matched_res = None
                        matched_rank = None
                        for r_idx, r in enumerate(final_results, start=1):
                            if (exp_file in r.video_filename or r.video_filename in exp_file):
                                overlap = max(0.0, min(r.end_seconds, exp_e) - max(r.start_seconds, exp_s))
                                if overlap > 0.0 or abs(r.start_seconds - exp_s) <= 2.0:
                                    matched_res = r
                                    matched_rank = r_idx
                                    break

                        if matched_res is not None:
                            trace["expect_evaluation"] = {
                                "expected": expect_segment,
                                "status": "MATCHED",
                                "rank": matched_rank,
                                "result": matched_res.to_dict(),
                            }
                        else:
                            # Diagnose exact failure point
                            diag = "never retrieved from index"
                            for h in visual_raw_hits:
                                if exp_file in h.get("video_filename", "") and abs(float(h.get("timestamp_seconds", 0.0)) - exp_s) <= 4.0:
                                    diag = f"retrieved by index (visual cos={h['score']:.4f}) but rejected by gate or outranked"
                                    break
                            for h in text_raw_hits:
                                if exp_file in h.get("video_filename", ""):
                                    diag = f"retrieved by index (text cos={h['score']:.4f}) but rejected by gate or outranked"
                                    break
                            for r_all_idx, r_all in enumerate(all_results, start=1):
                                if exp_file in r_all.video_filename:
                                    diag = f"passed gating and assembled at rank {r_all_idx}, but outranked by top_k={top_k}"
                                    break

                            trace["expect_evaluation"] = {
                                "expected": expect_segment,
                                "status": "MISSED",
                                "reason": diag,
                            }
                    except Exception as exc:
                        trace["expect_evaluation"] = {
                            "expected": expect_segment,
                            "status": "ERROR_PARSING",
                            "error": repr(exc),
                        }

                # Telemetry flags and counters update
                self._models_reused = True
                self._index_reused = True
                debug_info["models_reused"] = (
                    getattr(getattr(self.index, "visual_embedder", None), "load_count", 1) <= 1
                    and getattr(getattr(self.index, "text_embedder", None), "load_count", 1) <= 1
                )
                debug_info["visual_load_count"] = getattr(getattr(self.index, "visual_embedder", None), "load_count", 0)
                debug_info["text_load_count"] = getattr(getattr(self.index, "text_embedder", None), "load_count", 0)
                debug_info["trace"] = trace

        # Log trace at INFO level
        logger.info(
            "Search trace for query '%s' (status=%s, retrieved=%d):\n%s",
            clean_query,
            "ok" if final_results else "no_match",
            len(final_results),
            json.dumps(trace, indent=2, default=str),
        )

        if not final_results:
            return SearchResponse(
                schema_version=SCHEMA_VERSION,
                status="no_match",
                message="No related video segments found for your query.",
                results=[],
                timings_ms=timings.to_dict(),
                debug=debug_info,
                notices=notices,
            )

        return SearchResponse(
            schema_version=SCHEMA_VERSION,
            status="ok",
            message=f"Found {len(final_results)} relevant video segment(s).",
            results=final_results,
            timings_ms=timings.to_dict(),
            debug=debug_info,
            notices=notices,
        )
