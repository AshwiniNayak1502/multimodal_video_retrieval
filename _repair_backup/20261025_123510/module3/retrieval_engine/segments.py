"""Timestamp formatting, segment construction, cross-modal linking, and deduplication."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .fusion import compute_fused_score, compute_rrf_score
from .logging_config import get_logger

logger = get_logger("segments")


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS string.

    Supports >= 1 hour, hours are never capped at 23, and uses integer truncation
    so rounding never shifts the second boundary.
    """
    total_sec = max(0, int(seconds))
    hrs = total_sec // 3600
    rem = total_sec % 3600
    mins = rem // 60
    secs = rem % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def build_approx_visual_span(
    matched_at: float,
    interval_s: float,
    duration_s: float,
) -> Tuple[float, float]:
    """Build an approximate temporal span around a matched frame.

    Span is [max(0.0, matched_at - interval_s / 2), min(duration_s, matched_at + interval_s / 2)].
    """
    half_interval = max(0.1, interval_s / 2.0)
    start = max(0.0, matched_at - half_interval)
    end = matched_at + half_interval
    if duration_s > 0.0:
        end = min(duration_s, end)
    if end < start:
        end = start
    return start, end


def intervals_overlap_or_adjacent(
    s1: float,
    e1: float,
    s2: float,
    e2: float,
    tolerance_s: float,
) -> bool:
    """Check if two time intervals overlap or lie within tolerance_s."""
    return max(s1, s2) <= (min(e1, e2) + tolerance_s)


@dataclass
class EvidenceHit:
    """Individual evidence hit from either visual or text modality."""

    modality: str  # "visual" or "text"
    video_id: str
    video_filename: str
    video_path: str
    score: float  # Calibrated score
    raw_score: float  # Raw cosine similarity
    start_seconds: float
    end_seconds: float
    rank: int  # 1-based rank within modality candidate pool
    # Visual specific
    frame_id: Optional[str] = None
    frame_index: Optional[int] = None
    frame_path: Optional[str] = None
    matched_at_seconds: Optional[float] = None
    frame_interval_seconds: float = 2.0
    # Text specific
    segment_id: Optional[int] = None
    transcript_text: Optional[str] = None


@dataclass
class Result:
    """Consolidated video segment retrieval result."""

    video_id: str
    video_filename: str
    start_seconds: float
    end_seconds: float
    start_formatted: str
    end_formatted: str
    span_type: str  # "approx_from_frame", "transcript_segment", "combined"
    final_score: float
    visual_score: Optional[float]
    text_score: Optional[float]
    modalities: List[str]
    reason: str
    seek_seconds: float
    video_path: str
    matched_at_seconds: Optional[float] = None
    transcript_text: Optional[str] = None
    frame_id: Optional[str] = None
    frame_path: Optional[str] = None
    rrf_score: float = 0.0
    video_file_exists: bool = True
    resolution_seconds: float = 2.0
    visual_raw_cosine: Optional[float] = None
    text_raw_cosine: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to fully JSON-serializable dictionary."""
        return {
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "start_seconds": round(float(self.start_seconds), 3),
            "end_seconds": round(float(self.end_seconds), 3),
            "start_formatted": self.start_formatted,
            "end_formatted": self.end_formatted,
            "span_type": self.span_type,
            "final_score": round(float(self.final_score), 4),
            "visual_score": round(float(self.visual_score), 4) if self.visual_score is not None else None,
            "text_score": round(float(self.text_score), 4) if self.text_score is not None else None,
            "visual_raw_cosine": round(float(self.visual_raw_cosine), 4) if self.visual_raw_cosine is not None else None,
            "text_raw_cosine": round(float(self.text_raw_cosine), 4) if self.text_raw_cosine is not None else None,
            "modalities": list(self.modalities),
            "reason": self.reason,
            "seek_seconds": round(float(self.seek_seconds), 3),
            "video_path": str(self.video_path),
            "matched_at_seconds": round(float(self.matched_at_seconds), 3) if self.matched_at_seconds is not None else None,
            "transcript_text": self.transcript_text,
            "frame_id": self.frame_id,
            "frame_path": str(self.frame_path) if self.frame_path else None,
            "rrf_score": round(float(self.rrf_score), 5),
            "video_file_exists": bool(self.video_file_exists),
            "resolution_seconds": round(float(self.resolution_seconds), 2),
        }


def assemble_and_deduplicate_segments(
    visual_hits: List[EvidenceHit],
    text_hits: List[EvidenceHit],
    video_meta_map: Dict[str, Dict[str, Any]],
    visual_weight: float = 0.6,
    text_weight: float = 0.4,
    agreement_bonus: float = 0.05,
    final_min_score: float = 0.25,
    use_rrf: bool = True,
    rrf_k: int = 60,
    link_tolerance_s: Optional[float] = None,
    seek_offset_s: float = 1.0,
    fusion_mode: str = "weighted",
) -> List[Result]:
    """Group, temporally link, fuse, deduplicate, and score candidate segments across videos.

    Guarantees:
    1. Never joins across different video_ids.
    2. Merges contiguous frames from the same scene (consecutive frame_index or 0-tolerance overlap).
    3. Retains distinct scenes from the same video separated in time without chain-merging.
    4. Input hits are never mutated (copies are made).
    5. Honest spans:
       - visual-only: [t - interval/2, t + interval/2] clamped, span_type="approx_from_frame"
       - text-linked: transcript segment span clamped, span_type="transcript_segment"
       - combined: transcript span when corroborated, span_type="combined"
    6. Filters out any candidate below final_min_score before ranking.
    """
    import dataclasses

    # Group hits by video_id
    all_video_ids = set()
    v_by_vid: Dict[str, List[EvidenceHit]] = {}
    t_by_vid: Dict[str, List[EvidenceHit]] = {}

    for h in visual_hits:
        all_video_ids.add(h.video_id)
        v_by_vid.setdefault(h.video_id, []).append(h)

    for h in text_hits:
        all_video_ids.add(h.video_id)
        t_by_vid.setdefault(h.video_id, []).append(h)

    # Detect duplicate filenames across distinct video_ids for disambiguation
    filename_to_vids: Dict[str, List[str]] = {}
    for vid, meta in video_meta_map.items():
        fname = meta.get("file_name") or meta.get("video_filename") or vid
        filename_to_vids.setdefault(fname, []).append(vid)

    assembled_results: List[Result] = []

    for vid in all_video_ids:
        meta = video_meta_map.get(vid, {})
        video_filename = meta.get("file_name") or meta.get("video_filename") or vid
        if len(filename_to_vids.get(video_filename, [])) > 1:
            short_id = vid[:12]
            display_filename = f"{video_filename} ({short_id})"
        else:
            display_filename = video_filename

        resolved_video_path = meta.get("video_path", "")
        video_file_exists = Path(resolved_video_path).is_file() if resolved_video_path else False
        duration_s = float(meta.get("duration_seconds", 0.0))
        frame_interval = float(meta.get("frame_interval_seconds", 2.0))
        eff_link_tol = link_tolerance_s if link_tolerance_s is not None else frame_interval

        vid_v_hits = v_by_vid.get(vid, [])
        vid_t_hits = t_by_vid.get(vid, [])

        # Sort visual hits chronologically
        vid_v_hits.sort(key=lambda x: x.start_seconds)

        # Deduplicate consecutive visual frames only (frame_index diff==1 or 0-tolerance overlap)
        deduped_v_hits: List[EvidenceHit] = []
        for vh in vid_v_hits:
            vh_item = dataclasses.replace(vh)  # Never mutate input hits
            if not deduped_v_hits:
                deduped_v_hits.append(vh_item)
            else:
                last_vh = deduped_v_hits[-1]
                # Check if consecutive:
                is_consecutive = False
                if last_vh.frame_index is not None and vh_item.frame_index is not None:
                    is_consecutive = (abs(vh_item.frame_index - last_vh.frame_index) == 1)
                else:
                    is_consecutive = intervals_overlap_or_adjacent(
                        last_vh.start_seconds, last_vh.end_seconds,
                        vh_item.start_seconds, vh_item.end_seconds,
                        tolerance_s=0.0,
                    )

                if is_consecutive:
                    # Consecutive frames of the same scene:
                    # Keep the higher scoring one as primary evidence, but extend time span
                    new_start = min(last_vh.start_seconds, vh_item.start_seconds)
                    new_end = max(last_vh.end_seconds, vh_item.end_seconds)
                    if vh_item.score > last_vh.score:
                        merged = dataclasses.replace(vh_item, start_seconds=new_start, end_seconds=new_end)
                    else:
                        merged = dataclasses.replace(last_vh, start_seconds=new_start, end_seconds=new_end)
                    deduped_v_hits[-1] = merged
                else:
                    # Distinct frame separated in time: retain without merging
                    deduped_v_hits.append(vh_item)

        # Build candidate segments by linking deduped visual hits with text hits
        used_t_indices = set()

        for vh in deduped_v_hits:
            linked_text: Optional[EvidenceHit] = None
            linked_t_idx: Optional[int] = None

            # Frame instant interval [t - interval/2, t + interval/2]
            vh_instant = vh.matched_at_seconds if vh.matched_at_seconds is not None else (vh.start_seconds + vh.end_seconds) / 2.0
            vh_inst_start = max(0.0, vh_instant - frame_interval / 2.0)
            vh_inst_end = vh_instant + frame_interval / 2.0

            for t_idx, th in enumerate(vid_t_hits):
                # Link visual hit to transcript hit when frame window [t-interval/2, t+interval/2] comes within tolerance of transcript span
                if intervals_overlap_or_adjacent(
                    vh_inst_start, vh_inst_end,
                    th.start_seconds, th.end_seconds,
                    tolerance_s=eff_link_tol,
                ):
                    if linked_text is None or th.score > linked_text.score:
                        linked_text = th
                        linked_t_idx = t_idx

            if linked_text is not None and linked_t_idx is not None:
                used_t_indices.add(linked_t_idx)
                fused_s, reason = compute_fused_score(
                    visual_score=vh.score,
                    text_score=linked_text.score,
                    visual_weight=visual_weight,
                    text_weight=text_weight,
                    agreement_bonus=agreement_bonus,
                    fusion_mode=fusion_mode,
                )
                if fused_s < final_min_score:
                    continue

                # Combined span is the corroborated transcript segment span
                start_sec = max(0.0, linked_text.start_seconds)
                end_sec = linked_text.end_seconds
                if duration_s > 0.0:
                    end_sec = min(duration_s, end_sec)

                rrf = compute_rrf_score(
                    visual_rank=vh.rank,
                    text_rank=linked_text.rank,
                    rrf_k=rrf_k,
                    visual_weight=visual_weight,
                    text_weight=text_weight,
                )

                seek_sec = max(0.0, start_sec - seek_offset_s)

                assembled_results.append(
                    Result(
                        video_id=vid,
                        video_filename=display_filename,
                        start_seconds=start_sec,
                        end_seconds=end_sec,
                        start_formatted=format_timestamp(start_sec),
                        end_formatted=format_timestamp(end_sec),
                        span_type="combined",
                        final_score=fused_s,
                        visual_score=vh.score,
                        text_score=linked_text.score,
                        visual_raw_cosine=vh.raw_score,
                        text_raw_cosine=linked_text.raw_score,
                        modalities=["visual", "text"],
                        reason=reason,
                        seek_seconds=seek_sec,
                        video_path=resolved_video_path or vh.video_path,
                        matched_at_seconds=vh.matched_at_seconds,
                        transcript_text=linked_text.transcript_text,
                        frame_id=vh.frame_id,
                        frame_path=vh.frame_path,
                        rrf_score=rrf,
                        video_file_exists=video_file_exists,
                        resolution_seconds=0.1,
                    )
                )
            else:
                # Visual only segment: honest span around frame
                fused_s, reason = compute_fused_score(
                    visual_score=vh.score,
                    text_score=None,
                    visual_weight=visual_weight,
                    text_weight=text_weight,
                    agreement_bonus=agreement_bonus,
                    fusion_mode=fusion_mode,
                )
                if fused_s < final_min_score:
                    continue

                start_sec = max(0.0, vh.start_seconds)
                end_sec = vh.end_seconds
                if duration_s > 0.0:
                    end_sec = min(duration_s, end_sec)

                rrf = compute_rrf_score(
                    visual_rank=vh.rank,
                    text_rank=None,
                    rrf_k=rrf_k,
                    visual_weight=visual_weight,
                    text_weight=text_weight,
                )

                seek_sec = max(0.0, start_sec - seek_offset_s)

                assembled_results.append(
                    Result(
                        video_id=vid,
                        video_filename=display_filename,
                        start_seconds=start_sec,
                        end_seconds=end_sec,
                        start_formatted=format_timestamp(start_sec),
                        end_formatted=format_timestamp(end_sec),
                        span_type="approx_from_frame",
                        final_score=fused_s,
                        visual_score=vh.score,
                        text_score=None,
                        visual_raw_cosine=vh.raw_score,
                        text_raw_cosine=None,
                        modalities=["visual"],
                        reason=reason,
                        seek_seconds=seek_sec,
                        video_path=resolved_video_path or vh.video_path,
                        matched_at_seconds=vh.matched_at_seconds,
                        transcript_text=None,
                        frame_id=vh.frame_id,
                        frame_path=vh.frame_path,
                        rrf_score=rrf,
                        video_file_exists=video_file_exists,
                        resolution_seconds=frame_interval,
                    )
                )

        # Remaining unlinked text hits
        for t_idx, th in enumerate(vid_t_hits):
            if t_idx in used_t_indices:
                continue

            fused_s, reason = compute_fused_score(
                visual_score=None,
                text_score=th.score,
                visual_weight=visual_weight,
                text_weight=text_weight,
                agreement_bonus=agreement_bonus,
                fusion_mode=fusion_mode,
            )
            if fused_s < final_min_score:
                continue

            rrf = compute_rrf_score(
                visual_rank=None,
                text_rank=th.rank,
                rrf_k=rrf_k,
                visual_weight=visual_weight,
                text_weight=text_weight,
            )

            start_sec = max(0.0, th.start_seconds)
            end_sec = th.end_seconds
            if duration_s > 0.0:
                end_sec = min(duration_s, end_sec)

            seek_sec = max(0.0, start_sec - seek_offset_s)

            assembled_results.append(
                Result(
                    video_id=vid,
                    video_filename=display_filename,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    start_formatted=format_timestamp(start_sec),
                    end_formatted=format_timestamp(end_sec),
                    span_type="transcript_segment",
                    final_score=fused_s,
                    visual_score=None,
                    text_score=th.score,
                    visual_raw_cosine=None,
                    text_raw_cosine=th.raw_score,
                    modalities=["text"],
                    reason=reason,
                    seek_seconds=seek_sec,
                    video_path=resolved_video_path or th.video_path,
                    matched_at_seconds=None,
                    transcript_text=th.transcript_text,
                    frame_id=None,
                    frame_path=None,
                    rrf_score=rrf,
                    video_file_exists=video_file_exists,
                    resolution_seconds=0.1,
                )
            )

    # Secondary deduplication: if two segments from the same video actually overlap, keep higher score without union-extending spans
    assembled_results.sort(key=lambda r: (r.video_id, r.start_seconds))
    final_deduped: List[Result] = []
    for r in assembled_results:
        if not final_deduped:
            final_deduped.append(r)
        else:
            prev = final_deduped[-1]
            if prev.video_id == r.video_id and intervals_overlap_or_adjacent(
                prev.start_seconds, prev.end_seconds, r.start_seconds, r.end_seconds, tolerance_s=0.0
            ):
                # Overlapping segments in same video: retain higher scoring result without union-extending spans
                if r.final_score > prev.final_score:
                    final_deduped[-1] = r
            else:
                final_deduped.append(r)

    def _effective_raw_cosine(r: Result) -> float:
        if r.visual_raw_cosine is not None and r.text_raw_cosine is not None:
            total_w = visual_weight + text_weight
            if total_w > 0:
                return (visual_weight * r.visual_raw_cosine + text_weight * r.text_raw_cosine) / total_w
            return (r.visual_raw_cosine + r.text_raw_cosine) / 2.0
        elif r.visual_raw_cosine is not None:
            return r.visual_raw_cosine
        elif r.text_raw_cosine is not None:
            return r.text_raw_cosine
        return -1.0

    # Rank results: primary by final_score descending, secondary by raw_cosine tiebreaker, tertiary by rrf_score
    if use_rrf:
        final_deduped.sort(
            key=lambda r: (
                round(r.final_score, 4),
                round(_effective_raw_cosine(r), 4),
                round(r.rrf_score, 5),
            ),
            reverse=True,
        )
    else:
        final_deduped.sort(
            key=lambda r: (
                round(r.final_score, 4),
                round(_effective_raw_cosine(r), 4),
            ),
            reverse=True,
        )

    return final_deduped
