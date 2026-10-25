"""Video Library view showing user videos, metadata, thumbnails, retry, and delete."""

from typing import Any, Dict, Optional
from pathlib import Path
import streamlit as st
from app.db.app_db import AppDatabase, get_db
from app.services.auth_service import User
from app.services.video_service import delete_user_video, get_thumbnail_path, list_user_videos
from app.services.ingestion_runner import IngestionRunner


@st.dialog("Confirm Deletion")
def show_delete_confirmation(video: Dict[str, Any], user: User, db: Optional[AppDatabase] = None):
    """Confirmation modal before deleting a video."""
    st.warning(f"Are you sure you want to delete **{video.get('original_filename')}** from your library?")
    st.caption("If other users also have this video in their library, their copy remains intact.")
    
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("🗑️ Yes, Delete", type="primary", use_container_width=True, key=f"confirm_del_btn_{video['id']}"):
            delete_user_video(video["id"], user.id, db=db)
            st.success("Video deleted from your library.")
            st.rerun()
    with col_no:
        if st.button("Cancel", use_container_width=True, key=f"cancel_del_btn_{video['id']}"):
            st.rerun()


def render_library_view(
    ingestion_runner: IngestionRunner,
    user: User,
    db: Optional[AppDatabase] = None,
):
    """Renders the library of videos owned by the authenticated user."""
    database = db or get_db()

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-weight: 700; color: #F8FAFC; margin-bottom: 0.3rem;">Video Library</h2>
            <div style="color: #94A3B8; font-size: 0.95rem;">
                Browse, inspect extraction details, monitor pipeline progress, or manage your indexed videos.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    videos = list_user_videos(user.id, db=database)
    if not videos:
        st.markdown(
            """
            <div class="vce-card" style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🎬</div>
                <div style="font-weight: 600; font-size: 1.1rem; color: #E2E8F0;">No videos in your library yet</div>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 0.4rem; margin-bottom: 1.2rem;">
                    Upload an MP4 or MKV video to start semantic visual and audio search.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("➕ Go to Upload", type="primary"):
            st.session_state["active_nav"] = "📤 Upload Video"
            st.rerun()
        return

    for video in videos:
        status = video.get("status", "pending")
        stage = video.get("stage", "")
        duration = video.get("duration_seconds")
        dur_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "Calculating..."
        has_audio = video.get("has_audio")

        with st.container():
            st.markdown('<div class="vce-card">', unsafe_allow_html=True)
            col_thumb, col_info, col_actions = st.columns([1.5, 3.5, 1.2])

            with col_thumb:
                thumb_path = get_thumbnail_path(video.get("video_id", ""))
                if thumb_path and Path(thumb_path).is_file():
                    st.image(str(thumb_path), use_container_width=True)
                else:
                    st.markdown(
                        """
                        <div style="width: 100%; height: 110px; background: #1C1C38; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #64748B; font-size: 1.8rem;">
                            🎬
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_info:
                st.markdown(f"### {video.get('original_filename')}")
                
                # Status pill
                if status == "completed":
                    status_badge = '<span class="score-badge" style="background: rgba(34, 197, 94, 0.15); color: #4ADE80;">✅ Completed</span>'
                elif status in ("processing", "indexing"):
                    status_badge = f'<span class="score-badge" style="background: rgba(245, 158, 11, 0.15); color: #FBBF24;">⏳ {stage or status}...</span>'
                elif status == "pending":
                    status_badge = '<span class="score-badge" style="background: rgba(148, 163, 184, 0.15); color: #94A3B8;">🕒 Pending Queue</span>'
                else:
                    status_badge = f'<span class="score-badge" style="background: rgba(239, 68, 68, 0.15); color: #F87171;">❌ Failed: {video.get("error_message") or "Interrupted"}</span>'

                st.markdown(
                    f"""
                    <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 6px;">
                        {status_badge}
                        <span class="timestamp-pill">⏱️ {dur_str}</span>
                        <span class="timestamp-pill">ID: {video.get('video_id', '')[:12]}...</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if has_audio == 0 and status == "completed":
                    st.caption("🔇 *No speech detected in this video (visual-only index).*")

            with col_actions:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                if status == "failed":
                    if st.button("🔄 Retry", key=f"retry_btn_{video['id']}", use_container_width=True):
                        database.execute_write(
                            "UPDATE videos SET status = 'pending', stage = 'queued', error_message = NULL WHERE id = ?",
                            (video["id"],),
                        )
                        ingestion_runner.start()
                        st.info("Requeued for processing.")
                        st.rerun()

                if st.button("🗑️ Delete", key=f"del_btn_{video['id']}", use_container_width=True):
                    show_delete_confirmation(video, user, db=database)

            # Technical details expander
            with st.expander("🔍 Extraction & Indexing Details"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**Storage Path:** `{video.get('stored_path')}`")
                    st.markdown(f"**File Size:** `{video.get('size_bytes', 0) / (1024*1024):.2f} MB`")
                with c2:
                    st.markdown(f"**Created:** `{video.get('created_at')}`")
                    st.markdown(f"**Completed:** `{video.get('completed_at') or 'In progress'}`")
                with c3:
                    st.markdown(f"**Audio Track:** `{'Yes' if has_audio else ('No speech' if has_audio == 0 else 'Unknown')}`")
                    st.markdown(f"**SHA-256:** `{video.get('content_sha256', '')[:16]}...`")

            st.markdown('</div>', unsafe_allow_html=True)
