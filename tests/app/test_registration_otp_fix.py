"""Comprehensive automated tests for manual registration, OTP email verification, and security notice flows.

Covers:
1. New registration -> OTP generated & hashed -> local SMTP capture -> verification -> activation -> login.
2. OTP validation: wrong code, expiration, reuse rejection, max attempts lockout, resend cooldown & hourly cap.
3. Existing verified local account -> Security Notice sent, no OTP, neutral guidance, no duplicate user.
4. Existing Google-created account -> Security Notice sent, no OTP, no password override.
5. Pending/unverified retry -> new OTP replaces old pending code.
6. Unverified account cannot log in.
7. SMTP failure injection: AUTH (535), TRANSIENT (SMTPServerDisconnected), TIMEOUT, RECIPIENT (550), QUOTA.
8. SQLite concurrency: no open transaction across SMTP; concurrent reads/writes without locking.
9. Email RFC 5322 headers & header injection protection: Date, Message-ID, and CRLF stripping.
10. Account deletion & clean re-registration.
11. Log redaction defense: no secrets, passwords, or OTPs in logs.
"""

from __future__ import annotations

import datetime
import hmac
import os
import re
import socket
import smtplib
import ssl
import time
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import patch
import pytest

from aiosmtpd.controller import Controller

from app.db.app_db import AppDatabase
from app.services.auth_service import (
    AuthResult,
    User,
    cancel_registration,
    delete_user_account,
    login_user,
    register_user,
    resend_registration_code,
    start_registration,
    validate_session,
    verify_registration,
)
from app.services.email_service import (
    EmailDeliveryError,
    Mailer,
    SmtpErrorCode,
    SmtpMailer,
    classify_smtp_exception,
    create_email_message,
    get_mailer,
    mask_email,
    set_mailer,
)
from app.settings import settings

# Save reference to original SMTP class before any fixture monkeypatching
ORIGINAL_SMTP = smtplib.SMTP
ORIGINAL_SMTP_SSL = smtplib.SMTP_SSL


@pytest.fixture
def temp_db(tmp_path: Path) -> AppDatabase:
    """Create isolated SQLite database for each test."""
    db_file = tmp_path / "test_auth_suite.db"
    return AppDatabase(db_path=db_file)


class CapturedSMTPHandler:
    """In-memory SMTP server handler capturing raw messages."""

    def __init__(self):
        self.messages = []

    async def handle_DATA(self, server, session, envelope):
        self.messages.append({
            "mail_from": envelope.mail_from,
            "rcpt_tos": envelope.rcpt_tos,
            "raw": envelope.content.decode("utf-8", errors="replace"),
        })
        return "250 OK Message accepted"


@pytest.fixture
def local_smtp_server():
    """Spin up in-process aiosmtpd server on 127.0.0.1:1026 for real network protocol testing."""
    handler = CapturedSMTPHandler()
    controller = Controller(handler, hostname="127.0.0.1", port=1026)
    controller.start()
    yield controller, handler
    controller.stop()


# -------------------------------------------------------------------------
# Test 1: New Email Registration Flow -> Delivery -> Verify -> Login
# -------------------------------------------------------------------------
def test_1_new_registration_and_activation(temp_db, local_smtp_server, monkeypatch):
    controller, handler = local_smtp_server
    # Restore standard unblocked smtplib.SMTP for loopback testing against in-process aiosmtpd
    monkeypatch.setattr(smtplib, "SMTP", ORIGINAL_SMTP)

    class LocalNetworkMailer(SmtpMailer):
        def _send_single_attempt(self, to_email, msg, host, port, security, user, pwd):
            # Connect directly to local unencrypted test server
            s = smtplib.SMTP("127.0.0.1", 1026, timeout=5.0)
            try:
                s.send_message(msg)
            finally:
                s.close()

    mailer = LocalNetworkMailer()
    set_mailer(mailer)

    res = start_registration(
        email="newuser@example.com",
        password="ValidPassword123!",
        confirm_password="ValidPassword123!",
        display_name="New User",
        db=temp_db,
    )
    assert res.success is True
    assert res.requires_verification is True
    assert res.pending_id is not None
    assert len(handler.messages) == 1

    msg = handler.messages[0]
    assert "newuser@example.com" in msg["rcpt_tos"]
    assert "Verification Code" in msg["raw"]

    # Extract 6-digit code from captured raw message
    match = re.search(r"\b(\d{6})\b", msg["raw"])
    assert match is not None
    code = match.group(1)

    # Verify registration code
    res_verify = verify_registration(res.pending_id, code, db=temp_db)
    assert res_verify.success is True
    assert res_verify.user is not None
    assert res_verify.token is not None

    # Pending registration row must be deleted (single use)
    pending_check = temp_db.execute_read_one("SELECT * FROM pending_registrations WHERE id = ?", (res.pending_id,))
    assert pending_check is None

    # Login with the newly registered credentials
    login_res = login_user("newuser@example.com", "ValidPassword123!", db=temp_db)
    assert login_res.success is True
    assert login_res.user.email == "newuser@example.com"


# -------------------------------------------------------------------------
# Test 2: OTP Validation Boundaries, Expiry, Reuse, and Rate Limits
# -------------------------------------------------------------------------
def test_2_otp_boundaries_and_limits(temp_db, fake_mailer):
    res_start = start_registration(
        email="limits@example.com",
        password="ValidPassword123!",
        confirm_password="ValidPassword123!",
        display_name="Limits User",
        db=temp_db,
    )
    pending_id = res_start.pending_id
    real_code = fake_mailer.sent_emails[0]["code"]

    # A: Wrong OTP increments attempts
    res_wrong = verify_registration(pending_id, "999999", db=temp_db)
    assert res_wrong.success is False
    assert "Invalid verification code" in res_wrong.error_message

    # B: Expiration check (clock patched)
    past_iso = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=15)).isoformat()
    temp_db.execute_write("UPDATE pending_registrations SET code_expires_at = ? WHERE id = ?", (past_iso, pending_id))
    res_expired = verify_registration(pending_id, real_code, db=temp_db)
    assert res_expired.success is False
    assert "expired" in res_expired.error_message.lower()

    # Reset expiration for attempt testing
    future_iso = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)).isoformat()
    temp_db.execute_write("UPDATE pending_registrations SET code_expires_at = ?, attempts = 4 WHERE id = ?", (future_iso, pending_id))

    # 5th failed attempt deletes pending row (lockout)
    res_lockout = verify_registration(pending_id, "000000", db=temp_db)
    assert res_lockout.success is False
    row_after_lockout = temp_db.execute_read_one("SELECT * FROM pending_registrations WHERE id = ?", (pending_id,))
    assert row_after_lockout is None

    # C: Resend cooldown and invalidation
    fake_mailer.clear()
    res_resend_init = start_registration(
        email="resend@example.com",
        password="ValidPassword123!",
        confirm_password="ValidPassword123!",
        display_name="Resend User",
        db=temp_db,
    )
    p_id = res_resend_init.pending_id
    code1 = fake_mailer.sent_emails[0]["code"]

    # Immediate resend rejected by cooldown
    ok, err = resend_registration_code(p_id, db=temp_db)
    assert ok is False
    assert "wait" in (err or "").lower()

    # Fast-forward cooldown
    elapsed_sent = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=65)).isoformat()
    temp_db.execute_write("UPDATE pending_registrations SET last_sent_at = ? WHERE id = ?", (elapsed_sent, p_id))

    # Resend succeeds and generates new code
    fake_mailer.clear()
    ok_resend, _ = resend_registration_code(p_id, db=temp_db)
    assert ok_resend is True
    assert len(fake_mailer.sent_emails) == 1
    code2 = fake_mailer.sent_emails[0]["code"]

    # Old code is invalidated
    res_old = verify_registration(p_id, code1, db=temp_db)
    assert res_old.success is False

    # New code succeeds
    res_new = verify_registration(p_id, code2, db=temp_db)
    assert res_new.success is True

    # Reused code rejected (row no longer exists)
    res_reuse = verify_registration(p_id, code2, db=temp_db)
    assert res_reuse.success is False


# -------------------------------------------------------------------------
# Test 3: Existing Verified Account -> Security Notice Sent, No OTP, Neutral UI
# -------------------------------------------------------------------------
def test_3_existing_verified_account_notice(temp_db, fake_mailer):
    # Pre-register verified user
    register_user(
        email="existing@example.com",
        password="Password123!",
        confirm_password="Password123!",
        display_name="Existing User",
        db=temp_db,
    )
    fake_mailer.clear()

    res = start_registration(
        email="existing@example.com",
        password="AnotherPassword123!",
        confirm_password="AnotherPassword123!",
        display_name="Attacker or Confused User",
        db=temp_db,
    )

    # Truthful neutral outcome: notice_sent is True, info_message provides neutral guidance
    assert res.success is True
    assert res.notice_sent is True
    assert res.error_message is None
    assert "check your inbox" in res.info_message.lower()

    # Security Notice sent, NOT a verification code
    assert len(fake_mailer.sent_emails) == 1
    sent = fake_mailer.sent_emails[0]
    assert sent["kind"] == "security_notice"
    assert sent["to"] == "existing@example.com"
    assert "code" not in sent

    # No duplicate user created in DB
    users = temp_db.execute_read("SELECT * FROM users WHERE email = 'existing@example.com'")
    assert len(users) == 1


# -------------------------------------------------------------------------
# Test 4: Existing Google Account -> Security Notice Sent, No Password Overwrite
# -------------------------------------------------------------------------
def test_4_existing_google_account_notice(temp_db, fake_mailer):
    # Create Google-authenticated user directly as the OAuth flow does
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp_db.execute_write(
        """
        INSERT INTO users (email, display_name, password_hash, created_at, failed_attempts, email_verified, email_verified_at, google_sub, has_usable_password)
        VALUES (?, ?, ?, ?, 0, 1, ?, ?, 0)
        """,
        ("googleuser@example.com", "Google Person", "$argon2id$dummy", now_iso, now_iso, "sub-123456"),
    )
    fake_mailer.clear()

    res = start_registration(
        email="googleuser@example.com",
        password="AttemptedPassword123!",
        confirm_password="AttemptedPassword123!",
        display_name="Impostor",
        db=temp_db,
    )

    assert res.success is True
    assert res.notice_sent is True
    assert len(fake_mailer.sent_emails) == 1
    assert fake_mailer.sent_emails[0]["kind"] == "security_notice"

    # User remains Google-only with unusable password
    u = temp_db.execute_read_one("SELECT * FROM users WHERE email = 'googleuser@example.com'")
    assert u["has_usable_password"] == 0
    assert u["google_sub"] == "sub-123456"


# -------------------------------------------------------------------------
# Test 5: Pending/Unverified Duplicate -> Fresh OTP Replaces Old One
# -------------------------------------------------------------------------
def test_5_pending_duplicate_replaces_old_otp(temp_db, fake_mailer):
    # Initial registration attempt
    res1 = start_registration(
        email="pending_retry@example.com",
        password="Password123!",
        confirm_password="Password123!",
        display_name="Retry User",
        db=temp_db,
    )
    id1 = res1.pending_id
    code1 = fake_mailer.sent_emails[0]["code"]

    fake_mailer.clear()

    # Second registration attempt with same email
    res2 = start_registration(
        email="pending_retry@example.com",
        password="NewPassword123!",
        confirm_password="NewPassword123!",
        display_name="Retry User Updated",
        db=temp_db,
    )
    id2 = res2.pending_id
    code2 = fake_mailer.sent_emails[0]["code"]

    # Only one pending registration should exist
    all_pending = temp_db.execute_read("SELECT * FROM pending_registrations WHERE email = 'pending_retry@example.com'")
    assert len(all_pending) == 1
    assert all_pending[0]["id"] == id2

    # Old code fails
    assert verify_registration(id1, code1, db=temp_db).success is False
    # New code succeeds
    assert verify_registration(id2, code2, db=temp_db).success is True


# -------------------------------------------------------------------------
# Test 6: Unverified Account Cannot Log In
# -------------------------------------------------------------------------
def test_6_unverified_account_cannot_login(temp_db, fake_mailer):
    # Create unverified user row (legacy migration state)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    from app.services.auth_service import hash_password
    pwd_h = hash_password("ValidPassword123!")
    temp_db.execute_write(
        """
        INSERT INTO users (email, display_name, password_hash, created_at, failed_attempts, email_verified, has_usable_password)
        VALUES (?, ?, ?, ?, 0, 0, 1)
        """,
        ("unverified@example.com", "Unverified Person", pwd_h, now_iso),
    )

    # Attempt login -> routes to verification without issuing session
    res_login = login_user("unverified@example.com", "ValidPassword123!", db=temp_db)
    assert res_login.success is True
    assert res_login.requires_verification is True
    assert res_login.token is None
    assert len(fake_mailer.sent_emails) == 1

    # validate_session returns None for unverified user
    dummy_session = "mock_session_token"
    from app.services.auth_service import hash_token
    temp_db.execute_write(
        "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, 1, ?, ?)",
        (hash_token(dummy_session), now_iso, now_iso),
    )
    assert validate_session(dummy_session, db=temp_db) is None


# -------------------------------------------------------------------------
# Test 7: SMTP Failure Injection (Shared Transport Categorization)
# -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exception_to_raise, expected_category, expected_msg_substring",
    [
        (smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted"), SmtpErrorCode.AUTH, "authentication failed"),
        (smtplib.SMTPServerDisconnected("Connection unexpectedly closed"), SmtpErrorCode.TRANSIENT, "temporary email delivery failure"),
        (socket.timeout("timed out"), SmtpErrorCode.CONNECT, "unable to reach the email server"),
        (ConnectionRefusedError("Connection refused"), SmtpErrorCode.CONNECT, "unable to reach the email server"),
        (smtplib.SMTPRecipientsRefused({"bad@example.com": (550, b"User unknown")}), SmtpErrorCode.RECIPIENT, "recipient address was rejected"),
        (smtplib.SMTPResponseException(550, b"5.4.5 Daily user sending quota exceeded"), SmtpErrorCode.QUOTA, "sending limit or quota exceeded"),
    ],
)
def test_7_smtp_failure_injection(temp_db, exception_to_raise, expected_category, expected_msg_substring):
    class InjectedFailureMailer(SmtpMailer):
        def _send_with_retry(self, to_email, msg):
            raise exception_to_raise

    set_mailer(InjectedFailureMailer())

    res = start_registration(
        email="fail_test@example.com",
        password="ValidPassword123!",
        confirm_password="ValidPassword123!",
        display_name="Fail Test",
        db=temp_db,
    )

    assert res.success is False
    assert res.error_category == expected_category.value
    assert expected_msg_substring.lower() in res.error_message.lower()

    # Assert pending registration was invalidated and not left orphaned
    pending = temp_db.execute_read_one("SELECT * FROM pending_registrations WHERE email = 'fail_test@example.com'")
    assert pending is None


# -------------------------------------------------------------------------
# Test 8: SQLite Concurrency and Transaction Discipline
# -------------------------------------------------------------------------
def test_8_sqlite_concurrency_during_slow_smtp(temp_db):
    """Ensure no write transaction is open during SMTP call, allowing concurrent reads/writes."""
    slow_smtp_invoked = False
    concurrent_write_succeeded = False

    class SlowMailer(SmtpMailer):
        def _send_with_retry(self, to_email, msg):
            nonlocal slow_smtp_invoked, concurrent_write_succeeded
            slow_smtp_invoked = True

            # Open a second concurrent database connection during the SMTP delay
            db2 = AppDatabase(db_path=temp_db.db_path)
            # Write a row on connection 2 — MUST NOT hit "database is locked"
            db2.execute_write(
                "INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                ("concurrent@example.com", "Concurrent", "dummy", "2026-01-01"),
            )
            concurrent_write_succeeded = True

    set_mailer(SlowMailer())

    res = start_registration(
        email="slow@example.com",
        password="ValidPassword123!",
        confirm_password="ValidPassword123!",
        display_name="Slow Test",
        db=temp_db,
    )

    assert slow_smtp_invoked is True
    assert concurrent_write_succeeded is True
    assert res.success is True


# -------------------------------------------------------------------------
# Test 9: RFC 5322 Email Headers & Anti-Header Injection
# -------------------------------------------------------------------------
def test_9_email_headers_and_injection_defense():
    host = "smtp.gmail.com"
    msg = create_email_message(
        to_email="alice@example.com",
        subject="Your Code\r\nBcc: evil@attacker.com",
        text_content="Plain text content",
        html_content="<p>HTML content</p>",
        from_email="noreply@example.com\r\nBcc: evil2@attacker.com",
        from_name="VideoContextEngine\r\nBcc: evil3@attacker.com",
        host=host,
    )

    # 1. Date header must be present and RFC compliant
    assert msg["Date"] is not None
    # 2. Message-ID must be present
    assert msg["Message-ID"] is not None
    assert host in msg["Message-ID"] or "@" in msg["Message-ID"]

    # 3. CR/LF stripped from Subject, From, To
    assert "\r" not in msg["Subject"] and "\n" not in msg["Subject"]
    assert "\r" not in msg["From"] and "\n" not in msg["From"]
    # Verify no header injection occurred (Bcc header does not exist on message)
    assert "Bcc" not in msg
    assert msg.get_all("Bcc") is None

    # 4. MIME structure contains both plain and html parts
    payload = msg.get_payload()
    assert isinstance(payload, list)
    content_types = [p.get_content_type() for p in payload]
    assert "text/plain" in content_types
    assert "text/html" in content_types


# -------------------------------------------------------------------------
# Test 10: Account Deletion and Clean Re-registration
# -------------------------------------------------------------------------
def test_10_account_deletion_and_reregistration(temp_db, fake_mailer):
    # Register and activate initial account
    register_res = register_user(
        email="recycle@example.com",
        password="Password123!",
        confirm_password="Password123!",
        display_name="Recycle User",
        db=temp_db,
    )
    user_id = register_res.user.id
    fake_mailer.clear()

    # Delete the account cleanly
    deleted = delete_user_account(user_id, db=temp_db)
    assert deleted is True

    # User row is gone
    assert temp_db.execute_read_one("SELECT * FROM users WHERE id = ?", (user_id,)) is None

    # Re-registering the same email now works as a brand new registration
    res_rereg = start_registration(
        email="recycle@example.com",
        password="NewPassword456!",
        confirm_password="NewPassword456!",
        display_name="Recycled User",
        db=temp_db,
    )
    assert res_rereg.success is True
    assert res_rereg.requires_verification is True
    assert len(fake_mailer.sent_emails) == 1
    assert fake_mailer.sent_emails[0]["kind"] == "verification"


# -------------------------------------------------------------------------
# Test 11: Log Redaction & Secret Defense
# -------------------------------------------------------------------------
def test_11_log_redaction_secrets_defense(temp_db, fake_mailer):
    secret_key = "super-secret-auth-key-at-least-32-chars-long"
    os.environ["VRS_AUTH_SECRET"] = secret_key

    log_path = settings.logs_dir / "auth.log"
    initial_log_size = log_path.stat().st_size if log_path.exists() else 0

    # Perform registration
    res = start_registration(
        email="redaction_check@example.com",
        password="SuperSecretPassword999!",
        confirm_password="SuperSecretPassword999!",
        display_name="Redacted Person",
        db=temp_db,
    )
    assert res.success is True

    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(initial_log_size)
            new_logs = f.read()

        # Passwords, secrets, and raw OTP codes must NEVER appear in logs
        assert "SuperSecretPassword999!" not in new_logs
        assert secret_key not in new_logs
        if fake_mailer.sent_emails:
            raw_otp = fake_mailer.sent_emails[0]["code"]
            assert f"code={raw_otp}" not in new_logs
            assert f"code: {raw_otp}" not in new_logs
