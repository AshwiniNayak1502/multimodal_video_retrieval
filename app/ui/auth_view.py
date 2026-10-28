"""Authentication views for VideoContextEngine (Login & Sign Up)."""

from typing import Optional
import streamlit as st
from app.db.app_db import AppDatabase, get_db
from app.services.auth_service import (
    User,
    login_user,
    register_user,
)


def render_auth_view(db: Optional[AppDatabase] = None) -> Optional[User]:
    """Renders the authentication screen.
    
    Returns the authenticated User on successful login, or None.
    """
    database = db or get_db()

    st.markdown(
        """
        <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 2rem;">
            <div style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #A78BFA, #7C5CFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                VideoContextEngine
            </div>
            <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 0.4rem;">
                Local Multimodal Video Semantic Retrieval System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔑 Log In", "📝 Create Account"])

        with tab_login:
            st.markdown("#### Welcome Back")
            login_email = st.text_input("Email", key="login_email_input", placeholder="name@example.com")
            login_password = st.text_input("Password", type="password", key="login_password_input")

            # Disabled Google OAuth with tooltip
            st.markdown(
                """
                <div class="oauth-disabled-btn" title="Not configured (local mode) — all auth runs strictly offline with Argon2id.">
                    <span><svg style="vertical-align: middle; margin-right: 8px;" width="18" height="18" viewBox="0 0 24 24"><path fill="#64748B" d="M21.35 11.1h-9.17v2.98h5.27c-.23 1.22-.92 2.26-1.96 2.96v2.46h3.17c1.86-1.71 2.93-4.23 2.93-7.24 0-.47-.04-.94-.1-1.39l-.14.23z"/><path fill="#64748B" d="M12.18 21c2.65 0 4.88-.88 6.51-2.38l-3.17-2.46c-.88.59-2 .94-3.34.94-2.57 0-4.75-1.73-5.53-4.06H3.34v2.54C4.98 18.77 8.32 21 12.18 21z"/><path fill="#64748B" d="M6.65 13.04c-.2-.59-.31-1.22-.31-1.87s.11-1.28.31-1.87V6.76H3.34C2.7 8.03 2.33 9.47 2.33 11.17s.37 3.14 1.01 4.41l3.31-2.54z"/><path fill="#64748B" d="M12.18 5.08c1.44 0 2.74.5 3.76 1.47l2.82-2.82C17.05 2.1 14.82 1.17 12.18 1.17 8.32 1.17 4.98 3.4 3.34 6.76l3.31 2.54c.78-2.33 2.96-4.06 5.53-4.06z"/></svg>Continue with Google (Disabled in Local Mode)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Log In", type="primary", use_container_width=True, key="login_btn"):
                if not login_email or not login_password:
                    st.error("Please enter both email and password.")
                else:
                    res = login_user(login_email, login_password, db=database)
                    if res.success and res.user and res.token:
                        st.session_state["session_token"] = res.token
                        st.session_state["user"] = res.user
                        st.session_state.pop("workspace_init_error", None)
                        st.session_state["workspace_ready"] = False
                        st.rerun()
                    else:
                        st.error(res.error_message or "Invalid email or password.")

        with tab_register:
            st.markdown("#### Create Account")
            reg_name = st.text_input("Full Name or Nickname", key="reg_name_input", placeholder="Alex Doe")
            reg_email = st.text_input("Email", key="reg_email_input", placeholder="alex@example.com")
            reg_password = st.text_input("Password (min 8 chars, letters & numbers)", type="password", key="reg_password_input")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm_input")

            if st.button("Create Account", type="primary", use_container_width=True, key="reg_btn"):
                if not reg_name.strip():
                    st.error("Please provide a name.")
                elif not reg_email.strip():
                    st.error("Please provide an email address.")
                elif reg_password != reg_confirm:
                    st.error("Passwords do not match.")
                else:
                    res = register_user(
                        email=reg_email,
                        password=reg_password,
                        confirm_password=reg_confirm,
                        display_name=reg_name.strip(),
                        db=database,
                    )
                    if res.success and res.user and res.token:
                        st.session_state["session_token"] = res.token
                        st.session_state["user"] = res.user
                        st.session_state.pop("workspace_init_error", None)
                        st.session_state["workspace_ready"] = False
                        st.rerun()
                    else:
                        st.error(res.error_message or "Registration failed.")

    return None
