"""Application Shell layout with sidebar navigation, user account dialog, and video status list."""

from typing import Any, Dict, List, Optional
import streamlit as st
from app.db.app_db import AppDatabase, get_db
from app.services.auth_service import User, change_password, logout_session, set_password
from app.services.video_service import list_user_videos


@st.dialog("My Account")
def show_account_dialog(user: User, db: Optional[AppDatabase] = None):
    """Account settings dialog for password management and profile info."""
    st.markdown(f"**Name:** {user.display_name}")
    st.markdown(f"**Email:** {user.email}")
    st.markdown(f"**User ID:** `u{user.id}`")
    st.divider()

    if getattr(user, "has_usable_password", 1) == 0:
        st.info("You sign in with Google.")
        st.markdown("#### Set a Password")
        st.caption("You can optionally set a password to allow signing in directly with email and password.")
        new_pw = st.text_input("New Password (min 8 chars, letters & digits)", type="password", key="dialog_set_new_pw")
        confirm_pw = st.text_input("Confirm New Password", type="password", key="dialog_set_confirm_pw")

        if st.button("Set Password", type="primary", key="dialog_set_pw_btn"):
            if not new_pw:
                st.error("Please enter a new password.")
            elif new_pw != confirm_pw:
                st.error("New passwords do not match.")
            else:
                success, err = set_password(user.id, new_pw, confirm_pw, db=db)
                if success:
                    st.success("Password set successfully! You can now log in with email and password.")
                    user.has_usable_password = 1
                else:
                    st.error(err or "Failed to set password.")
    else:
        st.markdown("#### Change Password")
        old_pw = st.text_input("Current Password", type="password", key="dialog_old_pw")
        new_pw = st.text_input("New Password (min 8 chars, letters & digits)", type="password", key="dialog_new_pw")
        confirm_pw = st.text_input("Confirm New Password", type="password", key="dialog_confirm_pw")

        if st.button("Update Password", type="primary", key="dialog_update_pw_btn"):
            if not old_pw or not new_pw:
                st.error("Please fill in both current and new passwords.")
            elif new_pw != confirm_pw:
                st.error("New passwords do not match.")
            else:
                success, err = change_password(user.id, old_pw, new_pw, confirm_pw, db=db)
                if success:
                    st.success("Password updated successfully!")
                else:
                    st.error(err or "Failed to change password.")


def render_sidebar(user: User, db: Optional[AppDatabase] = None) -> str:
    """Renders the persistent left sidebar.
    
    Returns the selected active view: 'search', 'upload', or 'library'.
    """
    database = db or get_db()
    with st.sidebar:
        # App Branding Header
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem;">
                <div style="width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #7C5CFF, #A78BFA); display: flex; align-items: center; justify-content: center; font-weight: 800; color: white;">
                    VCE
                </div>
                <div>
                    <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC;">VideoContext</div>
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">Semantic Engine</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Main Navigation
        nav_options = ["🔍 Search", "📤 Upload Video", "🎬 Video Library"]
        nav_map = {
            "🔍 Search": "search",
            "📤 Upload Video": "upload",
            "🎬 Video Library": "library",
        }

        if "active_nav" not in st.session_state:
            st.session_state["active_nav"] = "🔍 Search"

        current_idx = nav_options.index(st.session_state["active_nav"]) if st.session_state["active_nav"] in nav_options else 0

        selected_label = st.radio(
            "Navigation",
            options=nav_options,
            index=current_idx,
            label_visibility="collapsed",
            key="sidebar_navigation_radio",
        )
        st.session_state["active_nav"] = selected_label
        selected_view = nav_map[selected_label]

        st.markdown("---")

        # Sidebar Quick Video List
        st.markdown(
            """
            <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; color: #94A3B8; margin-bottom: 0.5rem;">
                Your Videos
            </div>
            """,
            unsafe_allow_html=True,
        )

        videos = list_user_videos(user.id, db=database)
        if not videos:
            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #64748B; font-style: italic; padding: 0.5rem 0;">
                    No videos uploaded yet.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for v in videos[:6]:  # Show top 6 recent
                status = v.get("status", "pending")
                stage = v.get("stage", "")
                fname = v.get("original_filename", "Video")
                if len(fname) > 22:
                    fname = fname[:19] + "..."

                if status == "completed":
                    dot_class = "status-completed"
                    badge_text = "Ready"
                elif status in ("processing", "indexing"):
                    dot_class = "status-processing"
                    badge_text = stage or status
                else:
                    dot_class = "status-failed"
                    badge_text = "Failed"

                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 5px 0; font-size: 0.82rem; border-bottom: 1px solid #1C1C38;">
                        <span style="color: #CBD5E1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{v.get('original_filename')}">
                            <span class="status-dot {dot_class}"></span>{fname}
                        </span>
                        <span style="font-size: 0.72rem; color: #94A3B8; margin-left: 6px;">{badge_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if len(videos) > 6:
                st.caption(f"+ {len(videos) - 6} more in Library")

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("---")

        # User profile & logout section
        initials = (user.display_name or user.email or "U")[:2].upper()
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.75rem;">
                <div style="width: 32px; height: 32px; border-radius: 50%; background: #2A2A54; border: 1px solid #7C5CFF; display: flex; align-items: center; justify-content: center; font-weight: 700; color: #E2E8F0; font-size: 0.8rem;">
                    {initials}
                </div>
                <div style="overflow: hidden;">
                    <div style="font-weight: 600; font-size: 0.88rem; color: #F1F5F9; white-space: nowrap; text-overflow: ellipsis;">{user.display_name}</div>
                    <div style="font-size: 0.74rem; color: #64748B; white-space: nowrap; text-overflow: ellipsis;">{user.email}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_acc, col_logout = st.columns(2)
        with col_acc:
            if st.button("Account", key="sidebar_account_btn", use_container_width=True):
                show_account_dialog(user, db=database)
        with col_logout:
            if st.button("Log Out", key="sidebar_logout_btn", use_container_width=True):
                token = st.session_state.get("session_token")
                if token:
                    logout_session(token, db=database)
                st.session_state.clear()
                try:
                    st.logout()
                except Exception:
                    pass
                st.rerun()

    return selected_view
