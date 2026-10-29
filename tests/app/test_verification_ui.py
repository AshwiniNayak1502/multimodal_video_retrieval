"""UI integration and static analysis tests for email verification and auth security guards."""

import ast
from pathlib import Path
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app_entrypoint():
    return str(Path(__file__).parents[2] / "app" / "streamlit_app.py")


def test_ui_ast_guard_register_user_never_imported_or_called_in_ui():
    """Requirement 10 & 3.4: AST guard ensuring app/ui/*.py never imports or invokes register_user."""
    ui_dir = Path(__file__).parents[2] / "app" / "ui"
    python_files = list(ui_dir.glob("*.py"))
    assert len(python_files) > 0

    for py_file in python_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # Check for: from ... import register_user
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name != "register_user", (
                        f"Violation: {py_file.name} imports register_user. "
                        f"UI must only use start_registration() / verify_registration()."
                    )
            # Check for direct calls to register_user(...)
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "register_user":
                    pytest.fail(f"Violation: {py_file.name} invokes register_user() directly.")
                elif isinstance(func, ast.Attribute) and func.attr == "register_user":
                    pytest.fail(f"Violation: {py_file.name} invokes .register_user() directly.")


def test_no_running_or_debug_strings_in_auth_ui():
    """Requirement 10 & 5.6: No 'Running...' or raw debug strings in auth UI code."""
    auth_view_path = Path(__file__).parents[2] / "app" / "ui" / "auth_view.py"
    verify_view_path = Path(__file__).parents[2] / "app" / "ui" / "verify_email_view.py"

    auth_code = auth_view_path.read_text(encoding="utf-8")
    verify_code = verify_view_path.read_text(encoding="utf-8")

    assert "Running..." not in auth_code
    assert "Running..." not in verify_code
    assert "print(" not in auth_code
    assert "print(" not in verify_code


def test_env_and_secrets_examples_contain_no_real_secrets():
    """Requirement 10 & Ground Rule 3: Example files contain only placeholders, no real secrets."""
    project_root = Path(__file__).parents[2]
    env_example = (project_root / ".env.example").read_text(encoding="utf-8")
    secrets_example = (project_root / ".streamlit" / "secrets.toml.example").read_text(encoding="utf-8")

    # Verify no accidental live Google client secret format
    assert "GOCSPX-" not in secrets_example
    assert "GOCSPX-" not in env_example

    # Verify standard placeholders exist
    assert "YOUR_GOOGLE_CLIENT_ID" in secrets_example
    assert "CHANGE_ME" in secrets_example
    assert "your_16_char_google_app_password" in env_example
    assert "generate_at_least_32_random_characters_secret_here" in env_example


def test_ui_verification_view_renders_via_apptest(app_entrypoint, tmp_path, monkeypatch):
    """Requirement 9: AppTest verifies email verification view renders masked address, code input, and buttons."""
    st.cache_resource.clear()
    test_db = tmp_path / "ui_verification_test.db"
    monkeypatch.setenv("VRS_APP_DB_PATH", str(test_db))

    from app.db.app_db import AppDatabase
    from app.services.auth_service import start_registration

    db = AppDatabase(db_path=test_db)
    res_start = start_registration(
        email="alexander@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Alexander",
        db=db,
    )
    assert res_start.success is True

    # Run Streamlit AppTest
    at = AppTest.from_file(app_entrypoint, default_timeout=15)
    # Seed session state for pending verification
    at.session_state["pending_registration_id"] = res_start.pending_id
    at.session_state["pending_email"] = "alexander@example.com"
    at.run()

    assert not at.exception
    # Verify heading and masked email
    assert any("Verify Your Email" in str(m.value) for m in at.markdown)
    assert any("a***r@example.com" in str(m.value) for m in at.markdown)

    # Verify code input widget
    assert len(at.text_input) >= 1
    assert any(inp.key == "verify_code_input" for inp in at.text_input)

    # Verify submit button
    assert any(btn.key == "verify_submit_btn" for btn in at.button)
    assert any(btn.key == "change_email_btn" for btn in at.button)

    # Enter wrong code and submit
    at.text_input(key="verify_code_input").input("999999")
    at.button(key="verify_submit_btn").click().run()

    assert not at.exception
    assert any("Invalid verification code" in str(err.value) for err in at.error)

    # Click Change Email
    at.button(key="change_email_btn").click().run()
    assert not at.exception
    # Should return to Create Account tab with email prefilled
    assert "pending_registration_id" not in at.session_state
    assert at.text_input(key="reg_email_input").value == "alexander@example.com"
    assert at.text_input(key="reg_name_input").value == "Alexander"

