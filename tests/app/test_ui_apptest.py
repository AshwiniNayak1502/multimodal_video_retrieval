"""Streamlit AppTest integration tests for VideoContextEngine UI."""

import os
import pytest
from unittest.mock import MagicMock
import streamlit as st
from streamlit.testing.v1 import AppTest
from pathlib import Path


@pytest.fixture
def app_entrypoint():
    return str(Path(__file__).parents[2] / "app" / "streamlit_app.py")


def test_unauthenticated_landing_page(app_entrypoint):
    """Verifies that an unauthenticated user sees the login/register screen."""
    st.cache_resource.clear()
    at = AppTest.from_file(app_entrypoint, default_timeout=15)
    at.run()
    assert not at.exception
    # Should see the App title and login inputs
    assert any("VideoContextEngine" in str(m.value) for m in at.markdown)
    assert len(at.text_input) >= 2  # email and password fields


def test_registration_validation_error(app_entrypoint):
    """Verifies UI error feedback when registration has mismatched passwords."""
    st.cache_resource.clear()
    at = AppTest.from_file(app_entrypoint, default_timeout=15)
    at.run()
    assert not at.exception

    # Set inputs on register tab
    at.text_input(key="reg_name_input").input("Test User")
    at.text_input(key="reg_email_input").input("tester@example.com")
    at.text_input(key="reg_password_input").input("Password123")
    at.text_input(key="reg_confirm_input").input("DifferentPassword")
    at.button(key="reg_btn").click().run()

    assert not at.exception
    assert any("Passwords do not match" in str(err.value) for err in at.error)


def test_authenticated_shell_rendering(app_entrypoint, tmp_path, monkeypatch):
    """Verifies that a valid session token renders the main sidebar, navigation, and search view."""
    st.cache_resource.clear()

    # Mock QueryService to prevent expensive model loading during UI test
    mock_qs = MagicMock()
    mock_qs.search.return_value = MagicMock(
        status="ok",
        results=[],
        notices=[],
        debug={},
        message="No videos in library.",
    )
    monkeypatch.setattr("app.services.search_service.QueryService", lambda *a, **k: mock_qs)

    from app.db.app_db import AppDatabase
    from app.services.auth_service import login_user, register_user

    test_db_path = tmp_path / "app_ui_test.db"
    monkeypatch.setenv("VRS_APP_DB_PATH", str(test_db_path))

    db = AppDatabase(db_path=test_db_path)
    res_reg = register_user(
        email="alice@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Alice Tester",
        db=db,
    )
    assert res_reg.success is True

    res_login = login_user("alice@example.com", "Password123", db=db)
    assert res_login.success is True
    token = res_login.token

    at = AppTest.from_file(app_entrypoint, default_timeout=15)
    # Inject valid session token
    at.session_state["session_token"] = token
    at.run()

    assert not at.exception
    # Should see navigation radio
    assert len(at.sidebar.radio) >= 1
    # Main search input should be present
    assert len(at.text_input) >= 1
    # User display name should be in sidebar
    assert any("Alice Tester" in str(m.value) for m in at.sidebar.markdown)
