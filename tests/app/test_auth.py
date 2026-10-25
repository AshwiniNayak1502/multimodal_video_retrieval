"""Unit tests for authentication, password hashing, lockout, and sessions."""

import datetime
import tempfile
from pathlib import Path
import pytest

from app.db.app_db import AppDatabase
from app.services.auth_service import (
    change_password,
    hash_password,
    hash_token,
    login_user,
    logout_session,
    register_user,
    validate_password_strength,
    validate_session,
    verify_password,
)


@pytest.fixture
def test_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "auth_test.db"
        yield AppDatabase(db_path=db_path)


def test_password_hashing():
    pwd = "SecurePassword123"
    h = hash_password(pwd)
    assert h != pwd
    assert h.startswith("$argon2id$")
    assert verify_password(h, pwd) is True
    assert verify_password(h, "WrongPassword123") is False


def test_password_strength_validation():
    assert validate_password_strength("short1")[0] is False
    assert validate_password_strength("nonumbershere")[0] is False
    assert validate_password_strength("12345678")[0] is False
    assert validate_password_strength("ValidPass123")[0] is True


def test_registration_and_duplicate_handling(test_db):
    res1 = register_user(
        email="Alice@Example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Alice",
        db=test_db,
    )
    assert res1.success is True
    assert res1.user is not None
    assert res1.user.email == "alice@example.com"
    assert res1.token is not None

    # Duplicate registration (case-insensitive)
    res_dup = register_user(
        email="ALICE@example.com",
        password="Password123",
        confirm_password="Password123",
        db=test_db,
    )
    assert res_dup.success is False
    assert "already exists" in res_dup.error_message.lower()

    # Password mismatch
    res_mismatch = register_user(
        email="bob@example.com",
        password="Password123",
        confirm_password="Password456",
        db=test_db,
    )
    assert res_mismatch.success is False
    assert "match" in res_mismatch.error_message.lower()


def test_login_success_and_wrong_password(test_db):
    register_user(
        email="charlie@example.com",
        password="CorrectPassword123",
        confirm_password="CorrectPassword123",
        db=test_db,
    )

    # Wrong password
    login_fail = login_user("charlie@example.com", "WrongPass123", db=test_db)
    assert login_fail.success is False
    assert "invalid email or password" in login_fail.error_message.lower()

    # Wrong email
    login_unknown = login_user("nobody@example.com", "Password123", db=test_db)
    assert login_unknown.success is False
    assert "invalid email or password" in login_unknown.error_message.lower()

    # Correct login
    login_ok = login_user("Charlie@example.com", "CorrectPassword123", db=test_db)
    assert login_ok.success is True
    assert login_ok.user is not None
    assert login_ok.token is not None


def test_lockout_after_consecutive_failures(test_db):
    register_user(
        email="target@example.com",
        password="TargetPassword123",
        confirm_password="TargetPassword123",
        db=test_db,
    )

    # 4 failed attempts
    for _ in range(4):
        res = login_user("target@example.com", "BadPass123", db=test_db)
        assert res.success is False
        assert "locked" not in res.error_message.lower()

    # 5th failed attempt triggers lockout
    res5 = login_user("target@example.com", "BadPass123", db=test_db)
    assert res5.success is False

    # 6th attempt (even with right password) is locked out
    res_locked = login_user("target@example.com", "TargetPassword123", db=test_db)
    assert res_locked.success is False
    assert "locked" in res_locked.error_message.lower()


def test_session_validation_expiry_and_logout(test_db):
    reg = register_user(
        email="session@example.com",
        password="Password123",
        confirm_password="Password123",
        db=test_db,
    )
    token = reg.token
    assert token is not None

    # Valid session
    user = validate_session(token, db=test_db)
    assert user is not None
    assert user.email == "session@example.com"

    # Logout revokes session
    assert logout_session(token, db=test_db) is True
    user_after_logout = validate_session(token, db=test_db)
    assert user_after_logout is None

    # New login session
    login_res = login_user("session@example.com", "Password123", db=test_db)
    new_token = login_res.token
    assert validate_session(new_token, db=test_db) is not None

    # Simulate expired session
    expired_token = "expired_token_raw"
    token_h = hash_token(expired_token)
    past_iso = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat()
    test_db.execute_write(
        "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token_h, reg.user.id, past_iso, past_iso),
    )
    assert validate_session(expired_token, db=test_db) is None


def test_change_password(test_db):
    reg = register_user(
        email="change@example.com",
        password="OldPassword123",
        confirm_password="OldPassword123",
        db=test_db,
    )
    uid = reg.user.id

    # Wrong current password
    ok, err = change_password(uid, "WrongOld123", "NewPassword123", "NewPassword123", db=test_db)
    assert ok is False
    assert "current password is incorrect" in err.lower()

    # Successful change
    ok, err = change_password(uid, "OldPassword123", "NewPassword123", "NewPassword123", db=test_db)
    assert ok is True

    # Login with old password fails
    assert login_user("change@example.com", "OldPassword123", db=test_db).success is False

    # Login with new password succeeds
    assert login_user("change@example.com", "NewPassword123", db=test_db).success is True
