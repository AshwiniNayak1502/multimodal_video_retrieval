"""Tests for email verification, code generation, rate limiting, and legacy account flows."""

import datetime
from pathlib import Path
import pytest

from app.db.app_db import AppDatabase
from app.services.auth_service import (
    cancel_registration,
    hash_password,
    login_user,
    mask_email,
    register_user,
    resend_registration_code,
    start_registration,
    validate_session,
    verify_registration,
)
from app.services.email_service import EmailConfigurationError, SmtpMailer
from app.settings import settings


@pytest.fixture
def db(tmp_path: Path) -> AppDatabase:
    """Isolated test database instance."""
    db_file = tmp_path / "test_email_auth.db"
    return AppDatabase(db_path=db_file)


def test_mask_email():
    """Verify email masking for various formats."""
    assert mask_email("alex@example.com") == "a***x@example.com"
    assert mask_email("me@domain.com") == "m***@domain.com"
    assert mask_email("john.doe@gmail.com") == "j***e@gmail.com"
    assert mask_email("invalid-email") == "invalid-email"


def test_registration_success_and_verification_flow(db, fake_mailer):
    """Requirement 1: Start registration -> code sent -> verify code -> session and user."""
    res_start = start_registration(
        email="alice@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Alice Doe",
        db=db,
    )
    assert res_start.success is True
    assert res_start.requires_verification is True
    assert res_start.token is None
    assert res_start.pending_id is not None
    assert res_start.email == "alice@example.com"

    # Mailer sent verification code
    assert len(fake_mailer.sent_emails) == 1
    sent_msg = fake_mailer.sent_emails[0]
    assert sent_msg["to"] == "alice@example.com"
    assert sent_msg["kind"] == "verification"
    code = sent_msg["code"]
    assert len(code) == settings.verification_code_length

    # Verify with correct code
    res_verify = verify_registration(res_start.pending_id, code, db=db)
    assert res_verify.success is True
    assert res_verify.user is not None
    assert res_verify.user.email == "alice@example.com"
    assert res_verify.user.display_name == "Alice Doe"
    assert res_verify.token is not None

    # App session is valid and reflects email_verified
    user = validate_session(res_verify.token, db=db)
    assert user is not None
    assert user.id == res_verify.user.id

    # Pending registration must be cleaned up
    pending = db.execute_read_one("SELECT * FROM pending_registrations WHERE id = ?", (res_start.pending_id,))
    assert pending is None


def test_wrong_code_and_attempt_limit_lockout(db, fake_mailer):
    """Requirement 2: Wrong code increments attempts and locks out after max attempts."""
    res_start = start_registration(
        email="bob@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Bob",
        db=db,
    )
    pending_id = res_start.pending_id

    # First 4 wrong attempts
    for _ in range(4):
        res_fail = verify_registration(pending_id, "000000", db=db)
        assert res_fail.success is False
        assert res_fail.error_message == "Invalid verification code. Please try again."

    # 5th wrong attempt triggers invalidation
    res_max = verify_registration(pending_id, "000000", db=db)
    assert res_max.success is False
    assert res_max.error_message == "Verification code expired. Please request a new code."

    # Pending row must be deleted
    pending = db.execute_read_one("SELECT * FROM pending_registrations WHERE id = ?", (pending_id,))
    assert pending is None


def test_expired_code_rejection(db, fake_mailer):
    """Requirement 2: Expired verification codes are rejected."""
    res_start = start_registration(
        email="charlie@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Charlie",
        db=db,
    )
    pending_id = res_start.pending_id
    code = fake_mailer.sent_emails[0]["code"]

    # Manually expire the code in the DB
    past_iso = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=15)).isoformat()
    db.execute_write(
        "UPDATE pending_registrations SET code_expires_at = ? WHERE id = ?",
        (past_iso, pending_id),
    )

    res_verify = verify_registration(pending_id, code, db=db)
    assert res_verify.success is False
    assert res_verify.error_message == "Verification code expired. Please request a new code."


def test_code_reuse_rejected(db, fake_mailer):
    """Requirement 2: Codes cannot be reused once verified."""
    res_start = start_registration(
        email="dave@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Dave",
        db=db,
    )
    pending_id = res_start.pending_id
    code = fake_mailer.sent_emails[0]["code"]

    res_ok = verify_registration(pending_id, code, db=db)
    assert res_ok.success is True

    # Attempt to reuse code
    res_reuse = verify_registration(pending_id, code, db=db)
    assert res_reuse.success is False
    assert "Verification session expired" in (res_reuse.error_message or "")


def test_resend_cooldown_and_invalidation(db, fake_mailer):
    """Requirement 3: Cooldown enforced, old code invalidated, new code works."""
    res_start = start_registration(
        email="eve@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Eve",
        db=db,
    )
    pending_id = res_start.pending_id
    first_code = fake_mailer.sent_emails[0]["code"]

    # Immediate resend fails due to cooldown
    ok, err = resend_registration_code(pending_id, db=db)
    assert ok is False
    assert "second(s) before requesting a new code" in (err or "")

    # Advance DB timestamp to simulate cooldown expiration
    past_sent = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=65)).isoformat()
    db.execute_write(
        "UPDATE pending_registrations SET last_sent_at = ? WHERE id = ?",
        (past_sent, pending_id),
    )

    # Resend code succeeds
    ok_resend, err_resend = resend_registration_code(pending_id, db=db)
    assert ok_resend is True
    assert len(fake_mailer.sent_emails) == 2
    new_code = fake_mailer.sent_emails[1]["code"]

    # Old code is rejected
    res_old = verify_registration(pending_id, first_code, db=db)
    assert res_old.success is False

    # New code succeeds
    res_new = verify_registration(pending_id, new_code, db=db)
    assert res_new.success is True
    assert res_new.user is not None


def test_hourly_send_cap_enforced(db, fake_mailer):
    """Requirement 3: Per-email send cap is enforced."""
    res_start = start_registration(
        email="frank@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Frank",
        db=db,
    )
    pending_id = res_start.pending_id

    # Populate email_send_log to simulate 5 previous sends within the hour
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for _ in range(5):
        db.execute_write(
            "INSERT INTO email_send_log (email, sent_at, kind) VALUES (?, ?, ?)",
            ("frank@example.com", now_iso, "resend"),
        )

    # Cooldown passed
    past_sent = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120)).isoformat()
    db.execute_write("UPDATE pending_registrations SET last_sent_at = ? WHERE id = ?", (past_sent, pending_id))

    ok, err = resend_registration_code(pending_id, db=db)
    assert ok is False
    assert "Too many verification attempts" in (err or "")


def test_delivery_failure_leaves_no_traces(db, fake_mailer):
    """Requirement 4: Delivery failure leaves no user or pending row and shows exact message."""
    fake_mailer.should_fail = True

    res = start_registration(
        email="grace@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Grace",
        db=db,
    )
    assert res.success is False
    assert res.error_message == "We couldn't send the verification email. Check the email address or try again."

    # Assert no user and no pending row created
    user = db.execute_read_one("SELECT * FROM users WHERE email = 'grace@example.com'")
    assert user is None
    pending = db.execute_read_one("SELECT * FROM pending_registrations WHERE email = 'grace@example.com'")
    assert pending is None


def test_smtp_not_configured_message(db, monkeypatch):
    """Requirement 4: Missing SMTP settings returns clear configuration notice."""
    mailer = SmtpMailer()
    monkeypatch.setenv("VRS_SMTP_HOST", "")
    with pytest.raises(EmailConfigurationError) as exc_info:
        mailer.send_verification_email("test@example.com", "123456", 10)
    assert "Email delivery isn't configured on this server" in str(exc_info.value)


def test_duplicate_email_enumeration_safety(db, fake_mailer):
    """Requirement 5: Duplicate registration behaves identically outward and sends owner notice."""
    # Pre-register verified user
    register_user(
        email="helen@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Helen",
        db=db,
    )
    fake_mailer.clear()

    # Attempt to register with same email
    res_dup = start_registration(
        email="helen@example.com",
        password="AnotherPassword123",
        confirm_password="AnotherPassword123",
        display_name="Impostor",
        db=db,
    )

    # Identical outward appearance
    assert res_dup.success is True
    assert res_dup.requires_verification is True
    assert res_dup.pending_id is not None
    assert res_dup.token is None

    # Owner notice sent instead of verification code
    assert len(fake_mailer.sent_emails) == 1
    notice = fake_mailer.sent_emails[0]
    assert notice["to"] == "helen@example.com"
    assert notice["kind"] == "security_notice"

    # Dummy pending registration rejects any code
    res_code = verify_registration(res_dup.pending_id, "123456", db=db)
    assert res_code.success is False
    assert res_code.error_message == "Invalid verification code. Please try again."


def test_cancel_registration_and_prefill(db, fake_mailer):
    """Cancel registration deletes pending row and returns name/email for prefill."""
    res_start = start_registration(
        email="ian@example.com",
        password="Password123",
        confirm_password="Password123",
        display_name="Ian Malcolm",
        db=db,
    )
    pending_id = res_start.pending_id

    disp, email = cancel_registration(pending_id, db=db)
    assert disp == "Ian Malcolm"
    assert email == "ian@example.com"

    pending = db.execute_read_one("SELECT * FROM pending_registrations WHERE id = ?", (pending_id,))
    assert pending is None


def test_legacy_account_login_requires_verification(db, fake_mailer):
    """Requirement 6: Unverified legacy user receives code on login, no session until verified."""
    pwd_hash = hash_password("LegacyPass123")
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Create unverified legacy user directly in DB
    user_id = db.execute_write(
        """
        INSERT INTO users (email, display_name, password_hash, created_at, failed_attempts, email_verified, email_verified_at, has_usable_password)
        VALUES ('jack@example.com', 'Jack', ?, ?, 0, 0, NULL, 1)
        """,
        (pwd_hash, now_iso),
    )

    # Attempt login with valid password
    res_login = login_user("jack@example.com", "LegacyPass123", db=db)
    assert res_login.success is True
    assert res_login.requires_verification is True
    assert res_login.token is None
    assert res_login.pending_id is not None

    # Code was emailed
    assert len(fake_mailer.sent_emails) == 1
    code = fake_mailer.sent_emails[0]["code"]

    # Verify code
    res_verify = verify_registration(res_login.pending_id, code, db=db)
    assert res_verify.success is True
    assert res_verify.token is not None
    assert res_verify.user.id == user_id

    # User in DB is now verified
    user_row = db.execute_read_one("SELECT * FROM users WHERE id = ?", (user_id,))
    assert user_row["email_verified"] == 1


def test_validate_session_rejects_unverified_user(db):
    """Requirement 6: Defense in depth - validate_session rejects email_verified=0."""
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_dt.isoformat()
    expires_iso = (now_dt + datetime.timedelta(hours=1)).isoformat()

    # Create unverified user
    user_id = db.execute_write(
        """
        INSERT INTO users (email, display_name, password_hash, created_at, failed_attempts, email_verified, has_usable_password)
        VALUES ('unverified@example.com', 'Unverified', 'dummy_hash', ?, 0, 0, 1)
        """,
        (now_iso,),
    )

    # Insert valid session for unverified user
    from app.services.auth_service import hash_token
    token_str = "secret-session-token-for-unverified"
    token_h = hash_token(token_str)
    db.execute_write(
        "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token_h, user_id, now_iso, expires_iso),
    )

    # validate_session must return None
    user = validate_session(token_str, db=db)
    assert user is None

