"""Verification view for entering email verification code with live resend countdown."""

from __future__ import annotations

import datetime
from typing import Optional
import streamlit as st

from app.db.app_db import AppDatabase, get_db
from app.services.auth_service import (
    cancel_registration,
    mask_email,
    resend_registration_code,
    verify_registration,
)
from app.settings import settings


@st.fragment(run_every=1)
def render_resend_button(pending_id: str, database: AppDatabase) -> None:
    """Live countdown fragment for the Resend Code button."""
    pending = database.execute_read_one(
        "SELECT last_sent_at FROM pending_registrations WHERE id = ?", (pending_id,)
    )
    remaining_seconds = 0
    if pending and pending.get("last_sent_at"):
        try:
            last_dt = datetime.datetime.fromisoformat(pending["last_sent_at"])
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            elapsed = (now_dt - last_dt).total_seconds()
            remaining_seconds = max(0, int(settings.resend_cooldown_seconds - elapsed) + 1)
        except Exception:
            remaining_seconds = 0

    if remaining_seconds > 0:
        st.button(
            f"Resend Code ({remaining_seconds}s)",
            disabled=True,
            use_container_width=True,
            key="resend_code_btn_disabled",
        )
    else:
        if st.button("Resend Code", use_container_width=True, key="resend_code_btn"):
            with st.spinner("Sending verification email…"):
                success, err_msg = resend_registration_code(pending_id, db=database)
            if success:
                st.success("A new verification code has been sent.")
                st.rerun()
            else:
                st.error(err_msg or "Failed to resend verification code.")


def render_verify_email_view(db: Optional[AppDatabase] = None) -> None:
    """Renders the standalone verification code input screen."""
    database = db or get_db()
    pending_id = st.session_state.get("pending_registration_id")
    email = st.session_state.get("pending_email") or ""

    st.markdown(
        """
        <style>
        [data-testid="stStatusWidget"] {
            display: none !important;
            visibility: hidden !important;
        }
        </style>
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
        st.markdown("### Verify Your Email")
        masked = mask_email(email)
        st.markdown(
            f"<p style='color: #CBD5E1; font-size: 0.95rem; margin-bottom: 1.5rem;'>"
            f"We've sent a {settings.verification_code_length}-digit verification code to "
            f"<strong style='color: #F8FAFC;'>{masked}</strong>. Please enter it below to verify your account."
            f"</p>",
            unsafe_allow_html=True,
        )

        code_val = st.text_input(
            "Verification Code",
            key="verify_code_input",
            max_chars=settings.verification_code_length,
            placeholder="123456",
        )

        if st.button("Verify Email", type="primary", use_container_width=True, key="verify_submit_btn"):
            cleaned_code = (code_val or "").replace(" ", "").strip()
            if not cleaned_code:
                st.error("Please enter the verification code.")
            else:
                with st.spinner("Verifying…"):
                    res = verify_registration(pending_id, cleaned_code, db=database)

                if res.success and res.user and res.token:
                    st.session_state["session_token"] = res.token
                    st.session_state["user"] = res.user
                    st.session_state.pop("pending_registration_id", None)
                    st.session_state.pop("pending_email", None)
                    st.session_state.pop("workspace_init_error", None)
                    st.session_state["workspace_ready"] = False
                    st.rerun()
                else:
                    st.error(res.error_message or "Invalid verification code. Please try again.")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        col_resend, col_change = st.columns(2)

        with col_resend:
            render_resend_button(pending_id, database)

        with col_change:
            if st.button("Change Email", use_container_width=True, key="change_email_btn"):
                disp_name, orig_email = cancel_registration(pending_id, db=database)
                st.session_state.pop("pending_registration_id", None)
                st.session_state.pop("pending_email", None)
                if disp_name:
                    st.session_state["prefill_name"] = disp_name
                if orig_email:
                    st.session_state["prefill_email"] = orig_email
                st.rerun()

