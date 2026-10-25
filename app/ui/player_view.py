"""Player view displaying video player synced to seek timestamps and Key Moments timeline."""

from typing import Any, Dict, List, Optional
from pathlib import Path
import streamlit as st
from app.services.playback_service import ensure_playback_proxy
from app.services.search_service import SearchEngineWrapper


def format_timestamp(seconds: float) -> str:
    """Formats seconds into MM:SS or HH:MM:SS format."""
    total_sec = max(0, int(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def render_player_view(
    selected_result: Any,
    video_record: Dict[str, Any],
    search_wrapper: SearchEngineWrapper,
    all_search_hits: Optional[List[Any]] = None,
):
    """Renders the synchronized video player and Key Moments section."""
    start_time = float(getattr(selected_result, "start_time", 0.0))
    end_time = float(getattr(selected_result, "end_time", start_time + 5.0))
    score = float(getattr(selected_result, "score", 0.0))
    excerpt = getattr(selected_result, "transcript_text", "") or "Visual segment match"
    
    # Active seek time stored in session state
    seek_key = f"player_seek_{video_record.get('video_id')}"
    if seek_key not in st.session_state:
        st.session_state[seek_key] = int(start_time)

    current_seek = st.session_state[seek_key]

    # Resolve playable path (generates MP4 faststart proxy for MKV/non-web containers)
    stored_path = video_record.get("stored_path", "")
    video_id = video_record.get("video_id", "")
    try:
        playable_path = str(ensure_playback_proxy(stored_path, video_id))
    except Exception as e:
        playable_path = stored_path

    # Player Header
    st.markdown(
        f"""
        <div class="video-player-header">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-weight: 700; font-size: 1.15rem; color: #F8FAFC;">
                        {video_record.get('original_filename', 'Video Player')}
                    </span>
                    <span class="timestamp-pill" style="margin-left: 10px;">
                        Matched: {format_timestamp(start_time)} - {format_timestamp(end_time)}
                    </span>
                </div>
                <div>
                    <span class="score-badge">Match Score: {score:.1%}</span>
                </div>
            </div>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 6px;">
                <em>"{excerpt}"</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_player, col_moments = st.columns([3.2, 1.8])

    with col_player:
        if Path(playable_path).is_file():
            # Render st.video with start_time
            st.video(
                playable_path,
                format="video/mp4",
                start_time=current_seek,
                autoplay=True,
            )
        else:
            st.error(f"Playable video file not found on disk at: {playable_path}")

        # Controls under player
        st.caption(f"Currently cued at: **{format_timestamp(current_seek)}** (Segment: {format_timestamp(start_time)} - {format_timestamp(end_time)})")

    with col_moments:
        st.markdown("#### ⏱️ Key Moments")
        # Fetch real key moments
        key_moments = search_wrapper.get_key_moments(
            video_id=video_id,
            active_start_s=start_time,
            active_end_s=end_time,
            all_search_results=all_search_hits or [],
        )

        if not key_moments:
            st.caption("No additional key moments detected in this segment window.")
        else:
            for idx, km in enumerate(key_moments):
                km_time = km.timestamp_seconds
                km_label = km.label
                km_type = km.source_type
                km_desc = km.description

                type_icon = "🎯" if km_type == "search_hit" else "💬"
                is_current = (int(km_time) == current_seek)

                btn_text = f"{type_icon} {format_timestamp(km_time)} - {km_desc}"
                if len(btn_text) > 42:
                    btn_text = btn_text[:39] + "..."

                if st.button(
                    btn_text,
                    key=f"km_btn_{idx}_{int(km_time)}",
                    use_container_width=True,
                    help=f"Jump player to {format_timestamp(km_time)} ({km_label}: {km_desc})",
                ):
                    st.session_state[seek_key] = int(km_time)
                    st.rerun()
