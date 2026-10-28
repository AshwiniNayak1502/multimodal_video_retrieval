"""Regression tests for loading spinner removal, lazy auth initialization, and graceful error handling."""

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app_entrypoint():
    return str(Path(__file__).parents[2] / "app" / "streamlit_app.py")


def test_static_cache_resource_decorators_disable_default_spinner():
    """Requirement 1 & Step 3.1: Verify every @st.cache_resource explicitly passes show_spinner=False."""
    app_path = Path(__file__).parents[2] / "app" / "streamlit_app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    cached_funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                # Check for @st.cache_resource or @st.cache_resource(...)
                is_cache_res = False
                if isinstance(dec, ast.Attribute) and dec.attr == "cache_resource":
                    is_cache_res = True
                    pytest.fail(
                        f"Function '{node.name}' uses bare @st.cache_resource without show_spinner=False."
                    )
                elif isinstance(dec, ast.Call):
                    func_node = dec.func
                    if isinstance(func_node, ast.Attribute) and func_node.attr == "cache_resource":
                        is_cache_res = True
                        # Inspect keyword arguments for show_spinner
                        kw_dict = {kw.arg: kw.value for kw in dec.keywords}
                        assert "show_spinner" in kw_dict, (
                            f"Function '{node.name}' has @st.cache_resource missing explicit show_spinner argument."
                        )
                        val_node = kw_dict["show_spinner"]
                        if isinstance(val_node, ast.Constant):
                            # Must be False or a clean custom string
                            assert val_node.value is False or (
                                isinstance(val_node.value, str) and len(val_node.value.strip()) > 0
                            ), f"Function '{node.name}' show_spinner must be False or a custom string, got {val_node.value}"
                        else:
                            pytest.fail(f"Function '{node.name}' show_spinner argument must be a constant.")

                if is_cache_res:
                    cached_funcs[node.name] = dec

    assert "get_app_database" in cached_funcs
    assert "get_search_wrapper" in cached_funcs
    assert "get_ingestion_worker" in cached_funcs


def test_logged_out_run_lazy_initialization(app_entrypoint, monkeypatch):
    """Requirement 4 & Step 3.2: Logged out runs must not construct QueryService or call search wrappers."""
    st.cache_resource.clear()

    qs_mock = MagicMock()
    qs_construct_count = 0

    def mock_query_service_init(*args, **kwargs):
        nonlocal qs_construct_count
        qs_construct_count += 1
        return qs_mock

    monkeypatch.setattr("app.services.search_service.QueryService", mock_query_service_init)

    at = AppTest.from_file(app_entrypoint, default_timeout=15)
    at.run()
    assert not at.exception

    # QueryService and SearchEngineWrapper must NOT have been constructed
    assert qs_construct_count == 0

    # Login and registration inputs must render
    assert any("VideoContextEngine" in str(m.value) for m in at.markdown)
    assert at.text_input(key="login_email_input") is not None
    assert at.text_input(key="login_password_input") is not None
    assert at.text_input(key="reg_name_input") is not None

    # Simulate entering registration tab details (tab interaction)
    at.text_input(key="reg_name_input").input("Bob Visitor")
    at.run()
    assert not at.exception
    assert qs_construct_count == 0


def test_authenticated_run_initializes_once(app_entrypoint, tmp_path, monkeypatch):
    """Requirement 2, 5 & Step 3.3: Authenticated user initializes search once; reruns do not re-initialize."""
    st.cache_resource.clear()

    from app.db.app_db import AppDatabase
    from app.services.auth_service import login_user, register_user

    test_db_path = tmp_path / "app_ui_test_auth.db"
    monkeypatch.setenv("VRS_APP_DB_PATH", str(test_db_path))

    db = AppDatabase(db_path=test_db_path)
    reg_res = register_user("auth_user@example.com", "Password123", "Password123", "Auth User", db=db)
    assert reg_res.success is True
    login_res = login_user("auth_user@example.com", "Password123", db=db)
    assert login_res.success is True
    token = login_res.token

    qs_construct_count = 0
    mock_qs = MagicMock()
    mock_qs.search.return_value = MagicMock(status="ok", results=[], notices=[], debug={}, message="")

    def mock_qs_init(*args, **kwargs):
        nonlocal qs_construct_count
        qs_construct_count += 1
        return mock_qs

    monkeypatch.setattr("app.services.search_service.QueryService", mock_qs_init)

    at = AppTest.from_file(app_entrypoint, default_timeout=15)
    at.session_state["session_token"] = token
    at.run()

    assert not at.exception
    assert qs_construct_count == 1
    assert len(at.sidebar.radio) >= 1
    assert any("Auth User" in str(m.value) for m in at.sidebar.markdown)

    # Subsequent run (simulating user interaction or fragment rerun)
    at.run()
    assert not at.exception
    assert qs_construct_count == 1  # No repeated initialization!

    at.run()
    assert not at.exception
    assert qs_construct_count == 1


def test_auth_flow_invalid_creds_success_and_logout(app_entrypoint, tmp_path, monkeypatch):
    """Requirement 4 & Step 3.4: Invalid credentials show error; valid login works; logout returns to login."""
    st.cache_resource.clear()

    from app.db.app_db import AppDatabase
    from app.services.auth_service import register_user

    test_db_path = tmp_path / "app_ui_auth_flow.db"
    monkeypatch.setenv("VRS_APP_DB_PATH", str(test_db_path))

    db = AppDatabase(db_path=test_db_path)
    reg_res = register_user("clara@example.com", "Password123", "Password123", "Clara Oswald", db=db)
    assert reg_res.success is True

    # Mock QueryService
    mock_qs = MagicMock()
    monkeypatch.setattr("app.services.search_service.QueryService", lambda *a, **k: mock_qs)

    at = AppTest.from_file(app_entrypoint, default_timeout=15)
    at.run()
    assert not at.exception

    # 1. Invalid login attempt
    at.text_input(key="login_email_input").input("clara@example.com")
    at.text_input(key="login_password_input").input("WrongPassword999")
    at.button(key="login_btn").click().run()

    assert not at.exception
    assert any("Invalid email or password" in str(err.value) for err in at.error)

    # 2. Correct login attempt
    at.text_input(key="login_email_input").input("clara@example.com")
    at.text_input(key="login_password_input").input("Password123")
    at.button(key="login_btn").click().run()

    assert not at.exception
    assert len(at.sidebar.radio) >= 1
    assert any("Clara Oswald" in str(m.value) for m in at.sidebar.markdown)

    # 3. Log out
    at.sidebar.button(key="sidebar_logout_btn").click().run()
    assert not at.exception
    assert at.text_input(key="login_email_input") is not None
    assert len(at.sidebar.radio) == 0


def test_search_wrapper_init_failure_handled_safely(app_entrypoint, tmp_path, monkeypatch, caplog):
    """Requirement 6 & Step 3.5: Failed init logs error, shows user-friendly message, does not retry on every rerun."""
    st.cache_resource.clear()

    from app.db.app_db import AppDatabase
    from app.services.auth_service import login_user, register_user

    test_db_path = tmp_path / "app_ui_test_failure.db"
    monkeypatch.setenv("VRS_APP_DB_PATH", str(test_db_path))

    db = AppDatabase(db_path=test_db_path)
    reg_res = register_user("error_user@example.com", "Password123", "Password123", "Error User", db=db)
    assert reg_res.success is True
    login_res = login_user("error_user@example.com", "Password123", db=db)
    assert login_res.success is True
    token = login_res.token

    init_attempt_count = 0
    should_fail = True

    def failing_get_search_engine(*args, **kwargs):
        nonlocal init_attempt_count
        init_attempt_count += 1
        if should_fail:
            raise RuntimeError("Corrupted FAISS index vector file header.")
        mock_wrapper = MagicMock()
        return mock_wrapper

    monkeypatch.setattr("app.services.search_service.get_search_engine", failing_get_search_engine)

    with caplog.at_level(logging.ERROR):
        at = AppTest.from_file(app_entrypoint, default_timeout=15)
        at.session_state["session_token"] = token
        at.run()

    # Must NOT have uncaught app exception
    assert not at.exception

    # User-facing message must be friendly
    assert any("Video search could not be started" in str(err.value) for err in at.error)

    # Must NOT reveal raw stack trace, function name, or file paths in the error
    for err in at.error:
        err_str = str(err.value)
        assert "Traceback" not in err_str
        assert "failing_get_search_engine" not in err_str
        assert "Corrupted FAISS" not in err_str
        assert ".py" not in err_str

    # Must have logged full traceback to logger
    assert any("Failed to initialize search engine" in record.message for record in caplog.records)

    # Init must have been attempted once
    assert init_attempt_count == 1

    # Rerun without clicking retry: must NOT re-attempt heavy init!
    at.run()
    assert not at.exception
    assert init_attempt_count == 1  # Still 1!
    assert any("Video search could not be started" in str(err.value) for err in at.error)

    # Now simulate fixing the error and clicking Retry
    should_fail = False
    at.button(key="retry_init_btn").click().run()

    assert not at.exception
    assert init_attempt_count == 2
    # Workspace should now render successfully
    assert len(at.sidebar.radio) >= 1
