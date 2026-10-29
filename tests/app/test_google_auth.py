"""Tests for Google OpenID Connect integration, claim validation, account linking, and security rules."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import streamlit as st

from app.db.app_db import AppDatabase
from app.services.auth_service import (
    google_sign_in,
    hash_password,
    login_user,
    register_user,
    validate_session,
    verify_password,
)
from app.settings import settings


@pytest.fixture
def db(tmp_path: Path) -> AppDatabase:
    """Isolated test database instance."""
    db_file = tmp_path / "test_google_auth.db"
    return AppDatabase(db_path=db_file)


@pytest.fixture(autouse=True)
def enable_google_setting(monkeypatch):
    """Enable Google auth setting for all tests in this module."""
    monkeypatch.setenv("VRS_ENABLE_GOOGLE", "true")


def test_google_sign_in_new_account(db, monkeypatch):
    """Requirement 7: New Google user creates account with unusable password and issues session."""
    claims = {
        "iss": "https://accounts.google.com",
        "sub": "google-sub-1001",
        "email": "newbie@example.com",
        "name": "Newbie Google",
        "email_verified": True,
    }
    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", lambda *a, **k: claims)

    res = google_sign_in("dummy_token", db=db)
    assert res.success is True
    assert res.user is not None
    assert res.user.email == "newbie@example.com"
    assert res.user.display_name == "Newbie Google"
    assert res.user.has_usable_password == 0
    assert res.token is not None

    # App session is valid
    user = validate_session(res.token, db=db)
    assert user is not None
    assert user.id == res.user.id

    # Password hash stored in DB is an unusable Argon2id hash
    row = db.execute_read_one("SELECT * FROM users WHERE id = ?", (res.user.id,))
    assert row["google_sub"] == "google-sub-1001"
    assert row["has_usable_password"] == 0
    assert row["email_verified"] == 1


def test_google_sign_in_known_sub_match(db, monkeypatch):
    """Requirement 7: Matching by google_sub signs into the existing account."""
    claims = {
        "iss": "https://accounts.google.com",
        "sub": "google-sub-2002",
        "email": "primary@example.com",
        "name": "Known User",
        "email_verified": True,
    }
    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", lambda *a, **k: claims)

    # First sign-in creates account
    res1 = google_sign_in("token_1", db=db)
    assert res1.success is True
    first_id = res1.user.id

    # Second sign-in with same sub (even if email claim changed on Google)
    claims["email"] = "alias@example.com"
    res2 = google_sign_in("token_2", db=db)
    assert res2.success is True
    assert res2.user.id == first_id


def test_google_sign_in_link_verified_local_account(db, monkeypatch):
    """Requirement 7: Local account with email_verified=1 safely links google_sub."""
    # Pre-existing local verified account with known password
    res_reg = register_user(
        email="verified_local@example.com",
        password="LocalPassword123",
        confirm_password="LocalPassword123",
        display_name="Local Verified",
        db=db,
    )
    local_id = res_reg.user.id

    claims = {
        "iss": "https://accounts.google.com",
        "sub": "google-sub-3003",
        "email": "verified_local@example.com",
        "name": "Local Verified",
        "email_verified": True,
    }
    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", lambda *a, **k: claims)

    res_google = google_sign_in("token_link", db=db)
    assert res_google.success is True
    assert res_google.user.id == local_id

    # Verify google_sub linked, usable password preserved
    row = db.execute_read_one("SELECT * FROM users WHERE id = ?", (local_id,))
    assert row["google_sub"] == "google-sub-3003"
    assert row["has_usable_password"] == 1

    # Local password still works
    res_login = login_user("verified_local@example.com", "LocalPassword123", db=db)
    assert res_login.success is True


def test_google_sign_in_takeover_defense_unverified_account(db, monkeypatch):
    """Requirement 7: Pre-hijack defense - unverified local account password is wiped and sessions revoked."""
    pwd_hash = hash_password("AttackerPassword123")
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    future_iso = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=2)).isoformat()

    # Pre-existing unverified account (victim's email registered by third party)
    victim_id = db.execute_write(
        """
        INSERT INTO users (email, display_name, password_hash, created_at, failed_attempts, email_verified, has_usable_password)
        VALUES ('victim@example.com', 'Victim', ?, ?, 0, 0, 1)
        """,
        (pwd_hash, now_iso),
    )

    # Active session for this unverified account
    from app.services.auth_service import hash_token
    token_str = "attacker-active-session-token"
    db.execute_write(
        "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (hash_token(token_str), victim_id, now_iso, future_iso),
    )

    claims = {
        "iss": "https://accounts.google.com",
        "sub": "google-sub-victim",
        "email": "victim@example.com",
        "name": "Real Victim",
        "email_verified": True,
    }
    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", lambda *a, **k: claims)

    res = google_sign_in("token_takeover", db=db)
    assert res.success is True
    assert res.user.id == victim_id

    # Prior session is revoked
    old_session = db.execute_read_one("SELECT * FROM sessions WHERE token_hash = ?", (hash_token(token_str),))
    assert old_session["revoked_at"] is not None

    # Prior password stops working
    row = db.execute_read_one("SELECT * FROM users WHERE id = ?", (victim_id,))
    assert row["has_usable_password"] == 0
    assert not verify_password(row["password_hash"], "AttackerPassword123")


def test_google_sign_in_refuses_when_sub_linked_elsewhere(db, monkeypatch):
    """Requirement 7: Refuse linking if email is already linked to a different google_sub."""
    # Account linked to sub-A
    db.execute_write(
        """
        INSERT INTO users (email, display_name, password_hash, created_at, failed_attempts, email_verified, google_sub, has_usable_password)
        VALUES ('conflict@example.com', 'Conflict User', 'dummy_hash', '2026-01-01', 0, 1, 'sub-existing-A', 0)
        """
    )

    # Different sub claims this email
    claims = {
        "iss": "https://accounts.google.com",
        "sub": "sub-incoming-B",
        "email": "conflict@example.com",
        "name": "Conflict User",
        "email_verified": True,
    }
    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", lambda *a, **k: claims)

    res = google_sign_in("token_conflict", db=db)
    assert res.success is False
    assert "linked to a different Google account" in (res.error_message or "")


def test_google_sign_in_rejected_when_email_verified_false(db, monkeypatch):
    """Requirement 7: Reject when Google reports email_verified is False."""
    claims = {
        "iss": "https://accounts.google.com",
        "sub": "sub-unverified",
        "email": "not_verified@gmail.com",
        "name": "Unverified",
        "email_verified": False,
    }
    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", lambda *a, **k: claims)

    res = google_sign_in("token_unverified", db=db)
    assert res.success is False
    assert "not verified" in (res.error_message or "")


def test_google_sign_in_rejected_bad_issuer(db, monkeypatch):
    """Requirement 7: Reject when token issuer is not Google."""
    claims = {
        "iss": "https://rogue-auth.example.com",
        "sub": "sub-rogue",
        "email": "rogue@example.com",
        "name": "Rogue",
        "email_verified": True,
    }
    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", lambda *a, **k: claims)

    res = google_sign_in("token_rogue", db=db)
    assert res.success is False
    assert "couldn't verify your Google account" in (res.error_message or "")


def test_google_sign_in_rejected_when_verifier_raises(db, monkeypatch):
    """Requirement 7: Gracefully handle expired/invalid tokens from Google."""
    def _raise(*args, **kwargs):
        raise ValueError("Token expired")

    monkeypatch.setattr("google.oauth2.id_token.verify_oauth2_token", _raise)

    res = google_sign_in("token_expired", db=db)
    assert res.success is False
    assert "couldn't verify your Google account" in (res.error_message or "")


def test_google_sign_in_disabled(db, monkeypatch):
    """Requirement 7: Reject when Google auth is disabled."""
    monkeypatch.setenv("VRS_ENABLE_GOOGLE", "false")
    res = google_sign_in("token_any", db=db)
    assert res.success is False
    assert "not enabled" in (res.error_message or "")
