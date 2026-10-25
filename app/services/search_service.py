"""Search service integrating Module 3 with user isolation, query normalization, and Key Moments."""

from __future__ import annotations

import logging
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..db.app_db import AppDatabase, get_db
from ..settings import settings

logger = logging.getLogger("app.search")

# Add module2 and module3 to sys.path once via singleton adapter
_ROOT = settings.project_root
_MODULE2_DIR = (_ROOT / "module2").resolve()
_MODULE3_DIR = (_ROOT / "module3").resolve()

for p in (_MODULE2_DIR, _MODULE3_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retrieval_engine import Module3Config, QueryService, format_timestamp
from retrieval_engine.query_service import SearchResponse

# Intent-stripping regex patterns for light query normalization
_LEADING_INTENT_PATTERNS = [
    re.compile(r"^(?:please\s+)?(?:can\s+you\s+)?show\s+me\s+(?:the\s+part\s+where|the\s+moment\s+when|where)\s+", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?(?:can\s+you\s+)?find\s+(?:the\s+part\s+where|the\s+moment\s+when|where)\s+", re.IGNORECASE),
    re.compile(r"^(?:tell\s+me\s+)?(?:the\s+part\s+where|the\s+moment\s+when)\s+", re.IGNORECASE),
    re.compile(r"^(?:show\s+me|find|locate)\s+", re.IGNORECASE),
    re.compile(r"^(?:where\s+is|where\s+does)\s+", re.IGNORECASE),
]


def normalize_search_query(query: str, mode: str = "light") -> Tuple[str, Optional[str]]:
    """Deterministically strip leading intent verbs if mode == 'light'.

    Returns (normalized_query, original_query_or_none_if_unchanged).
    """
    raw = (query or "").strip()
    if mode != "light" or not raw:
        return raw, None

    cleaned = raw
    for pat in _LEADING_INTENT_PATTERNS:
        m = pat.match(cleaned)
        if m:
            candidate = cleaned[m.end():].strip(" ,:?-")
            if candidate:  # Do not strip down to empty string
                cleaned = candidate
                break

    if cleaned != raw:
        return cleaned, raw
    return raw, None


@dataclass
class KeyMoment:
    """Timeline entry for Key Moments navigation."""

    timestamp_seconds: float
    timestamp_formatted: str
    label: str
    description: str
    source_type: str  # "search_hit" | "transcript" | "visual"
    frame_path: Optional[str] = None


class SearchEngineWrapper:
    """Process-wide lock-safe singleton wrapper around Module 3 QueryService."""

    def __init__(self, config_path: Optional[Path | str] = None) -> None:
        cfg_file = Path(config_path or (_MODULE3_DIR / "config.json")).resolve()
        if cfg_file.is_file():
            self.config = Module3Config.from_json(cfg_file)
        else:
            self.config = Module3Config()

        self._lock = threading.RLock()
        logger.info("Initializing process-wide QueryService (warmup=True)...")
        self.service = QueryService(config=self.config, warmup=True)

    def reload(self) -> None:
        """Reload index in a thread-safe manner after indexing runs."""
        with self._lock:
            self.service.reload_index()

    def search_user_videos(
        self,
        user_id: int,
        query: str,
        top_k: int = 5,
        modality_mode: str = "Combined",  # "Combined" | "Transcript only" | "Visual only"
        filter_video_id: Optional[str] = None,
        db: Optional[AppDatabase] = None,
    ) -> SearchResponse:
        """Execute user-isolated search strictly scoping candidate pool to user's completed videos."""
        database = db or get_db()

        # Step 1: Look up all completed video_ids owned by this user
        if filter_video_id:
            user_videos = database.execute_read(
                "SELECT video_id FROM videos WHERE user_id = ? AND video_id = ? AND status = 'completed'",
                (user_id, filter_video_id),
            )
        else:
            user_videos = database.execute_read(
                "SELECT video_id FROM videos WHERE user_id = ? AND status = 'completed'",
                (user_id,),
            )

        valid_vids = [v["video_id"] for v in user_videos if v.get("video_id")]

        # Enforce user isolation: if user owns no completed videos, return early without touching index!
        if not valid_vids:
            return SearchResponse(
                schema_version="3.1.0",
                status="ok",
                message="No indexed videos available in your library to search. Please upload a video first.",
                results=[],
                timings_ms={},
                debug={"user_id": user_id, "video_ids": []},
                notices=["Your video library has no completed videos yet."],
            )

        # Step 2: Map UI modality modes to QueryService modalities parameter
        if modality_mode == "Transcript only":
            selected_modalities = ("text",)
        elif modality_mode == "Visual only":
            selected_modalities = ("visual",)
        else:
            selected_modalities = ("visual", "text")

        # Step 3: Query normalization
        normalized_q, original_q = normalize_search_query(
            query, mode=settings.query_rewrite
        )

        with self._lock:
            resp = self.service.search(
                query=normalized_q,
                top_k=min(top_k, settings.max_top_k),
                video_ids=valid_vids,
                modalities=selected_modalities,
            )

            # Enrich debug trace with normalization details
            if original_q:
                resp.debug["original_query"] = original_q
                resp.debug["normalized_query"] = normalized_q
                if "trace" in resp.debug:
                    resp.debug["trace"]["query_original"] = original_q
                    resp.debug["trace"]["query_rewritten"] = normalized_q

            return resp

    def get_key_moments(
        self,
        video_id: str,
        active_start_s: float,
        active_end_s: float,
        all_search_results: Sequence[Any],
        window_s: float = 60.0,
    ) -> List[KeyMoment]:
        """Assemble chronological Key Moments for the active video.

        Sources:
        1. Other search hits in the same video.
        2. Surrounding transcript segments in [start - W, end + W].
        3. Nearest visual frames if audio has no speech.
        """
        moments: List[KeyMoment] = []
        seen_timestamps: set[float] = set()

        # 1. Other search hits in this same video
        for r in all_search_results:
            if getattr(r, "video_id", None) == video_id:
                seek = round(float(getattr(r, "seek_seconds", 0.0)), 1)
                if seek not in seen_timestamps:
                    seen_timestamps.add(seek)
                    excerpt = getattr(r, "transcript_text", "") or "Visual match"
                    moments.append(
                        KeyMoment(
                            timestamp_seconds=seek,
                            timestamp_formatted=getattr(r, "start_formatted", format_timestamp(seek)),
                            label="Search Match",
                            description=excerpt[:120] + ("..." if len(excerpt) > 120 else ""),
                            source_type="search_hit",
                            frame_path=getattr(r, "frame_path", None),
                        )
                    )

        # 2. Surrounding transcript segments
        with self._lock:
            w_start = max(0.0, active_start_s - window_s)
            w_end = active_end_s + window_s
            try:
                surrounding = self.service.index.get_segments_in_window(video_id, w_start, w_end)
                for seg in surrounding:
                    seg_start = round(float(seg.get("start_timestamp", 0.0)), 1)
                    if seg_start not in seen_timestamps:
                        seen_timestamps.add(seg_start)
                        text = (seg.get("transcript_text") or "").strip()
                        if text:
                            moments.append(
                                KeyMoment(
                                    timestamp_seconds=seg_start,
                                    timestamp_formatted=seg.get("start_formatted", format_timestamp(seg_start)),
                                    label="Transcript",
                                    description=text[:100] + ("..." if len(text) > 100 else ""),
                                    source_type="transcript",
                                    frame_path=None,
                                )
                            )
            except Exception as e:
                logger.debug("Could not fetch surrounding transcript segments: %s", e)

        # 3. If no speech detected, fetch nearby visual frames
        if len(moments) <= 1:
            with self._lock:
                try:
                    w_start = max(0.0, active_start_s - window_s)
                    w_end = active_end_s + window_s
                    frames = self.service.index.get_frames_in_window(video_id, w_start, w_end)
                    for f in frames[:6]:  # Limit to 6 key frames
                        ts = round(float(f.get("timestamp_seconds", 0.0)), 1)
                        if ts not in seen_timestamps:
                            seen_timestamps.add(ts)
                            moments.append(
                                KeyMoment(
                                    timestamp_seconds=ts,
                                    timestamp_formatted=f.get("timestamp_formatted", format_timestamp(ts)),
                                    label="Visual Frame",
                                    description="Visual frame sampled from video track",
                                    source_type="visual",
                                    frame_path=f.get("frame_path"),
                                )
                            )
                except Exception as e:
                    logger.debug("Could not fetch visual window frames: %s", e)

        # Sort all moments chronologically
        moments.sort(key=lambda m: m.timestamp_seconds)
        return moments


# Process-wide singleton instance
_search_engine_instance: Optional[SearchEngineWrapper] = None
_search_engine_lock = threading.Lock()


def get_search_engine(config_path: Optional[Path | str] = None) -> SearchEngineWrapper:
    """Return the process-wide QueryService search engine wrapper."""
    global _search_engine_instance
    with _search_engine_lock:
        if _search_engine_instance is None:
            _search_engine_instance = SearchEngineWrapper(config_path=config_path)
        return _search_engine_instance
