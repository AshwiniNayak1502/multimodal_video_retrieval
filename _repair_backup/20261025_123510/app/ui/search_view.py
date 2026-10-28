"""Search View for VideoContextEngine with modal search, diagnostics, and result selection."""

from typing import Any, Dict, List, Optional
from pathlib import Path
import streamlit as st
from app.db.app_db import AppDatabase, get_db
from app.services.auth_service import User
from app.services.search_service import SearchEngineWrapper
from app.services.video_service import get_thumbnail_path, list_user_videos
from app.ui.player_view import format_timestamp, render_player_view


def render_search_view(
    search_wrapper: SearchEngineWrapper,
    user: User,
    db: Optional[AppDatabase] = None,
):
    """Renders the main semantic search screen."""
    database = db or get_db()

    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h2 style="font-weight: 700; color: #F8FAFC; margin-bottom: 0.3rem;">Search in Your Videos</h2>
            <div style="color: #94A3B8; font-size: 0.95rem;">
                Locate exact moments across spoken transcripts and visual frames in your indexed library.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Search Bar & Mode Controls
    col_input, col_mode, col_btn = st.columns([3.5, 1.5, 1.0])

    with col_input:
        query_text = st.text_input(
            "Query",
            placeholder="e.g. 'purple title card', 'deep learning architecture', 'data flow chart'",
            label_visibility="collapsed",
            key="search_query_input",
        )

    with col_mode:
        search_mode = st.selectbox(
            "Modality",
            options=["Combined", "Transcript only", "Visual only"],
            index=0,
            label_visibility="collapsed",
            key="search_modality_select",
        )

    with col_btn:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True, key="exec_search_btn")

    # Handle Search Submission
    if search_clicked:
        if not query_text.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching video corpus across multimodal indices..."):
                response = search_wrapper.search_user_videos(
                    user_id=user.id,
                    query=query_text.strip(),
                    top_k=10,
                    modality_mode=search_mode,
                    db=database,
                )
                st.session_state["search_active_query"] = query_text.strip()
                st.session_state["search_response_cache"] = response
                st.session_state["selected_result_idx"] = 0

    # Render cached search results
    cached_response = st.session_state.get("search_response_cache")
    if cached_response is None:
        # Initial empty state
        st.markdown(
            """
            <div class="vce-card" style="text-align: center; padding: 3rem 1rem; margin-top: 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🎯</div>
                <div style="font-weight: 600; font-size: 1.1rem; color: #E2E8F0;">Multimodal Semantic Retrieval</div>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 0.4rem;">
                    Type a concept, phrase, or visual description above to find precise timestamps in your videos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    status = getattr(cached_response, "status", "ok")
    results = getattr(cached_response, "results", [])
    notices = getattr(cached_response, "notices", [])
    debug_trace = getattr(cached_response, "debug", {})
    message = getattr(cached_response, "message", "")

    # Display system notices if present
    for notice in notices:
        st.info(f"ℹ️ {notice}")

    # Handle Status Outcomes
    if status == "index_not_ready":
        st.info(
            "⏳ Your videos are currently being ingested and indexed. "
            "Please check back in a moment or visit the Video Library."
        )
        return
    elif status == "error":
        st.error(f"⚠️ Search error: {message or 'An internal error occurred.'}")
        return
    elif status == "no_match" or not results:
        msg_text = message or "No related video found"
        st.markdown(
            f"""
            <div class="vce-card" style="text-align: center; padding: 2.5rem 1rem; margin-top: 1rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🔎</div>
                <div style="font-weight: 600; font-size: 1.05rem; color: #E2E8F0;">{msg_text}</div>
                <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 0.3rem;">
                    Try refining your query, switching search modalities, or uploading more videos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Results view
    st.markdown(f"#### Results ({len(results)} matches found)")

    # User's videos map for metadata lookup
    user_videos = {v["video_id"]: v for v in list_user_videos(user.id, db=database)}

    selected_idx = st.session_state.get("selected_result_idx", 0)
    if selected_idx >= len(results):
        selected_idx = 0

    col_list, col_player = st.columns([1.8, 2.2])

    with col_list:
        for idx, res in enumerate(results):
            vid_id = getattr(res, "video_id", "")
            vid_record = user_videos.get(vid_id, {})
            v_title = vid_record.get("original_filename", f"Video {vid_id[:8]}")
            
            modality = getattr(res, "modality", "hybrid")
            score = float(getattr(res, "score", 0.0))
            start_s = float(getattr(res, "start_time", 0.0))
            end_s = float(getattr(res, "end_time", start_s + 5.0))
            excerpt = getattr(res, "transcript_text", "") or "Visual scene alignment"
            if len(excerpt) > 90:
                excerpt = excerpt[:87] + "..."

            is_selected = (idx == selected_idx)
            card_class = "result-card-container vce-card-active" if is_selected else "result-card-container"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span class="modality-chip modality-{modality}">{modality}</span>
                        <span class="score-badge">{score:.1%} match</span>
                    </div>
                    <div style="font-weight: 600; font-size: 0.95rem; color: #F8FAFC; margin-bottom: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                        {v_title}
                    </div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px;">
                        <em>"{excerpt}"</em>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="timestamp-pill">⏱️ {format_timestamp(start_s)} - {format_timestamp(end_s)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Button to select card
            if st.button(
                f"▶️ Play Segment {idx + 1}" if not is_selected else f"✓ Selected ({format_timestamp(start_s)})",
                key=f"select_res_btn_{idx}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state["selected_result_idx"] = idx
                # Reset seek position to start of new segment
                st.session_state[f"player_seek_{vid_id}"] = int(start_s)
                st.rerun()

    with col_player:
        selected_res = results[selected_idx]
        selected_vid_id = getattr(selected_res, "video_id", "")
        selected_vid_record = user_videos.get(selected_vid_id, {
            "video_id": selected_vid_id,
            "original_filename": f"Video {selected_vid_id[:8]}",
            "stored_path": "",
            "playback_path": None,
        })

        render_player_view(
            selected_result=selected_res,
            video_record=selected_vid_record,
            search_wrapper=search_wrapper,
            all_search_hits=results,
        )

    # Search diagnostics expander
    with st.expander("🛠️ Search Diagnostics & Model Trace"):
        st.markdown("**Query Execution Trace:**")
        st.json(debug_trace)
