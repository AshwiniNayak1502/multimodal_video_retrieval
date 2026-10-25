"""Upload View for VideoContextEngine."""

from typing import Optional
import streamlit as st
from app.db.app_db import AppDatabase, get_db
from app.services.auth_service import User
from app.services.video_service import save_uploaded_stream
from app.services.ingestion_runner import IngestionRunner


def render_upload_view(
    ingestion_runner: IngestionRunner,
    user: User,
    db: Optional[AppDatabase] = None,
):
    """Renders the video upload screen."""
    database = db or get_db()

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-weight: 700; color: #F8FAFC; margin-bottom: 0.3rem;">Upload Video</h2>
            <div style="color: #94A3B8; font-size: 0.95rem;">
                Upload MP4 or MKV videos for automated frame sampling, transcription (WhisperX), and multimodal indexing (SigLIP2 & MiniLM).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a video file (.mp4 or .mkv)",
        type=["mp4", "mkv"],
        help="Maximum file size 2GB. Formats: MP4, MKV.",
        key="main_video_file_uploader",
    )

    if uploaded_file is not None:
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        file_size_mb = uploaded_file.size / (1024 * 1024)
        with col_meta1:
            st.metric("Filename", uploaded_file.name)
        with col_meta2:
            st.metric("Size", f"{file_size_mb:.2f} MB")
        with col_meta3:
            st.metric("Format", uploaded_file.name.split(".")[-1].upper())

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        if st.button("🚀 Start Upload & Ingestion", type="primary", use_container_width=True, key="submit_upload_btn"):
            progress_bar = st.progress(0, text="Uploading and computing SHA-256 hash...")
            try:
                success, msg, video_record = save_uploaded_stream(
                    user_id=user.id,
                    filename=uploaded_file.name,
                    stream=uploaded_file,
                    expected_size=uploaded_file.size,
                    db=database,
                )
                progress_bar.progress(70, text="File validated and registered.")

                if not success or video_record is None:
                    st.error(f"Upload failed: {msg}")
                else:
                    status = video_record.get("status")
                    if status == "pending":
                        # Ensure background worker is running
                        ingestion_runner.start()
                        progress_bar.progress(100, text="Done!")
                        st.success(
                            f"✅ '{uploaded_file.name}' successfully queued for processing! "
                            f"You can monitor its progress in the Video Library or sidebar."
                        )
                    elif status == "completed":
                        progress_bar.progress(100, text="Done!")
                        st.info(
                            f"⚡ Instant Index: This exact video content is already indexed in the system. "
                            f"It is immediately available in your library and search!"
                        )
            except Exception as e:
                st.error(f"Unexpected error during upload: {str(e)}")
