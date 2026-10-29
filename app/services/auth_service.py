"""Authentication and session management service backed by Argon2id, SQLite, and Google OpenID Connect."""

from __future__ import annotations

import datetime
import hashlib
import hmac
import logging
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from ..db.app_db import AppDatabase, get_db
from ..settings import settings
from .email_service import EmailConfigurationError, EmailDeliveryError, get_mailer

logger = logging.getLogger("app.auth")


def _get_auth_secret() -> str:
    """Retrieve active authentication secret from environment or settings."""
    return os.getenv("VRS_AUTH_SECRET") or settings.auth_secret or ""

# Initialize Argon2id password hasher with secure parameters
_ph = PasswordHasher(time_cost=2, memory_cost=65536, parallelism=1)

_EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class User:
    """Authenticated user profile."""

    id: int
    email: str
    display_name: str
    created_at: str
    has_usable_password: int = 1


@dataclass
class AuthResult:
    """Result container for authentication workflows."""

    success: bool
    error_message: Optional[str] = None
    user: Optional[User] = None
    token: Optional[str] = None
    pending_id: Optional[str] = None
    email: Optional[str] = None
    requires_verification: bool = False


def hash_password(password: str) -> str:
    """Generate secure Argon2id hash for plaintext password."""
    return _ph.hash(password)


def verify_password(hash_val: str, password: str) -> bool:
    """Verify password against stored Argon2id hash in constant time."""
    try:
        return _ph.verify(hash_val, password)
    except (VerificationError, VerifyMismatchError, InvalidHashError):
        return False


def hash_token(token: str) -> str:
    """Cryptographic SHA-256 digest of session token for safe persistence."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
    """Validate password length and character composition."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    has_letter = any(c.isalpha() for c in password)
    has_digit = any(c.isdigit() for c in password)
    if not (has_letter and has_digit):
        return False, "Password must contain at least one letter and at least one number."
    return True, None


def mask_email(email: str) -> str:
    """Mask email address for privacy display (e.g. alex@example.com -> a***x@example.com)."""
    if not email or "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "***"
    else:
        masked_local = local[0] + "***" + local[-1]
    return f"{masked_local}@{domain}"


def _issue_session(
    user_id: int,
    database: AppDatabase,
    now_dt: Optional[datetime.datetime] = None,
) -> str:
    """Create session row and return opaque token string."""
    now = now_dt or datetime.datetime.now(datetime.timezone.utc)
    now_iso = now.isoformat()
    session_token = secrets.token_urlsafe(32)
    token_h = hash_token(session_token)
    expires_at = (
        now + datetime.timedelta(minutes=settings.session_ttl_minutes)
    ).isoformat()

    database.execute_write(
        """
        INSERT INTO sessions (token_hash, user_id, created_at, expires_at)
        VALUES (?, ?, ?, ?)
        """,
        (token_h, user_id, now_iso, expires_at),
    )
    return session_token


def _purge_expired_pending(database: AppDatabase, now_iso: str) -> None:
    """Purge abandoned pending registration records."""
    try:
        database.execute_write(
            "DELETE FROM pending_registrations WHERE code_expires_at < ?",
            (now_iso,),
        )
    except Exception as exc:
        logger.debug("Failed opportunistic pending registrations purge: %s", exc)


def _hash_verification_code(pending_id: str, salt: str, code: str, secret: str) -> str:
    """Generate keyed HMAC-SHA256 digest of verification code."""
    payload = f"{pending_id}:{salt}:{code}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def _check_send_rate_limits(
    email: str, database: AppDatabase, now_dt: datetime.datetime
) -> Tuple[bool, Optional[str]]:
    """Enforce per-email and global hourly send limits."""
    one_hour_ago = (now_dt - datetime.timedelta(hours=1)).isoformat()

    # Check global send volume
    global_row = database.execute_read_one(
        "SELECT COUNT(*) as cnt FROM email_send_log WHERE sent_at > ?",
        (one_hour_ago,),
    )
    if global_row and int(global_row["cnt"]) >= settings.email_max_per_hour:
        return False, "Server email capacity reached. Please try again later."

    # Check per-email send volume
    email_row = database.execute_read_one(
        "SELECT COUNT(*) as cnt FROM email_send_log WHERE email = ? COLLATE NOCASE AND sent_at > ?",
        (email, one_hour_ago),
    )
    if email_row and int(email_row["cnt"]) >= settings.resend_max_per_hour:
        return False, "Too many verification attempts for this email address. Please try again in an hour."

    return True, None


def start_registration(
    email: str,
    password: str,
    confirm_password: str,
    display_name: Optional[str] = None,
    db: Optional[AppDatabase] = None,
) -> AuthResult:
    """Initiate registration workflow with password validation and email verification code."""
    database = db or get_db()
    email_clean = (email or "").strip().lower()
    disp = (display_name or "").strip()

    if not disp:
        return AuthResult(success=False, error_message="Please provide a name.")

    if not email_clean or not _EMAIL_REGEX.match(email_clean):
        return AuthResult(success=False, error_message="Please enter a valid email address.")

    if password != confirm_password:
        return AuthResult(success=False, error_message="Passwords do not match.")

    is_strong, strength_err = validate_password_strength(password)
    if not is_strong:
        return AuthResult(success=False, error_message=strength_err)

    auth_secret = _get_auth_secret()
    if len(auth_secret) < 32:
        return AuthResult(
            success=False,
            error_message="Server authentication secret (VRS_AUTH_SECRET) is missing or shorter than 32 characters.",
        )

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_dt.isoformat()
    _purge_expired_pending(database, now_iso)

    # Check if verification is bypassed via configuration
    if not settings.require_email_verification:
        return register_user(
            email=email_clean,
            password=password,
            confirm_password=confirm_password,
            display_name=disp,
            db=database,
        )

    # Check for existing account (enumeration-safe duplicate policy)
    existing_user = database.execute_read_one(
        "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email_clean,)
    )
    if existing_user and int(existing_user.get("email_verified") or 0) == 1:
        # Match computation time of standard registration
        _ = hash_password(password)

        # Send security notification notice to the account owner
        try:
            mailer = get_mailer()
            mailer.send_security_notice(email_clean)
            database.execute_write(
                "INSERT INTO email_send_log (email, sent_at, kind) VALUES (?, ?, ?)",
                (email_clean, now_iso, "security_notice"),
            )
        except EmailConfigurationError as exc:
            return AuthResult(success=False, error_message=str(exc))
        except Exception as exc:
            logger.error("Failed to send duplicate security notice: %s", type(exc).__name__)
            return AuthResult(
                success=False,
                error_message="We couldn't send the verification email. Check the email address or try again.",
            )

        # Return identical outward appearance with an inert pending token
        dummy_pending_id = secrets.token_hex(16)
        dummy_salt = secrets.token_hex(16)
        dummy_hash = _hash_verification_code(dummy_pending_id, dummy_salt, "DUMMY", auth_secret)
        code_expires_at = (now_dt + datetime.timedelta(minutes=settings.verification_ttl_minutes)).isoformat()

        database.execute_write(
            "DELETE FROM pending_registrations WHERE email = ? COLLATE NOCASE", (email_clean,)
        )
        database.execute_write(
            """
            INSERT INTO pending_registrations
            (id, email, display_name, password_hash, code_hash, code_salt, created_at, code_expires_at, last_sent_at, attempts, resend_count, resend_window_started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
            """,
            (dummy_pending_id, email_clean, disp, "DUMMY_INERT", dummy_hash, dummy_salt, now_iso, code_expires_at, now_iso, now_iso),
        )

        return AuthResult(
            success=True,
            pending_id=dummy_pending_id,
            email=email_clean,
            requires_verification=True,
        )

    # Check rate limits
    can_send, rate_err = _check_send_rate_limits(email_clean, database, now_dt)
    if not can_send:
        return AuthResult(success=False, error_message=rate_err)

    # Generate cryptographically secure verification code
    code = "".join(str(secrets.randbelow(10)) for _ in range(settings.verification_code_length))
    salt = secrets.token_hex(16)
    pending_id = secrets.token_hex(16)
    code_hash = _hash_verification_code(pending_id, salt, code, auth_secret)
    pwd_hash = hash_password(password)
    code_expires_at = (now_dt + datetime.timedelta(minutes=settings.verification_ttl_minutes)).isoformat()

    # Deliver verification email before committing pending state
    try:
        mailer = get_mailer()
        mailer.send_verification_email(email_clean, code, settings.verification_ttl_minutes)
    except EmailConfigurationError as exc:
        return AuthResult(success=False, error_message=str(exc))
    except Exception as exc:
        logger.error("Failed to send verification email: %s", type(exc).__name__)
        return AuthResult(
            success=False,
            error_message="We couldn't send the verification email. Check the email address or try again.",
        )

    # Record send log and write pending registration row (replacing any prior pending row)
    database.execute_write(
        "INSERT INTO email_send_log (email, sent_at, kind) VALUES (?, ?, ?)",
        (email_clean, now_iso, "verification"),
    )
    database.execute_write(
        "DELETE FROM pending_registrations WHERE email = ? COLLATE NOCASE", (email_clean,)
    )
    database.execute_write(
        """
        INSERT INTO pending_registrations
        (id, email, display_name, password_hash, code_hash, code_salt, created_at, code_expires_at, last_sent_at, attempts, resend_count, resend_window_started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
        """,
        (pending_id, email_clean, disp, pwd_hash, code_hash, salt, now_iso, code_expires_at, now_iso, now_iso),
    )

    return AuthResult(
        success=True,
        pending_id=pending_id,
        email=email_clean,
        requires_verification=True,
    )


def verify_registration(
    pending_id: str,
    code: str,
    db: Optional[AppDatabase] = None,
) -> AuthResult:
    """Verify submitted code, create verified user record, delete pending row, and issue session."""
    database = db or get_db()
    submitted_code = (code or "").replace(" ", "").strip()

    if not pending_id:
        return AuthResult(
            success=False,
            error_message="Verification session expired or invalid. Please start again.",
        )

    auth_secret = _get_auth_secret()
    if len(auth_secret) < 32:
        return AuthResult(
            success=False,
            error_message="Server authentication secret (VRS_AUTH_SECRET) is missing or shorter than 32 characters.",
        )

    pending = database.execute_read_one(
        "SELECT * FROM pending_registrations WHERE id = ?", (pending_id,)
    )
    if not pending:
        return AuthResult(
            success=False,
            error_message="Verification session expired or invalid. Please start again.",
        )

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_dt.isoformat()

    # Check code expiration
    try:
        expires_at = datetime.datetime.fromisoformat(pending["code_expires_at"])
        if now_dt > expires_at:
            return AuthResult(
                success=False,
                error_message="Verification code expired. Please request a new code.",
            )
    except Exception:
        return AuthResult(
            success=False,
            error_message="Verification code expired. Please request a new code.",
        )

    # Compute and verify code hash in constant time
    expected_hash = pending["code_hash"]
    computed_hash = _hash_verification_code(
        pending_id, pending["code_salt"], submitted_code, auth_secret
    )
    is_match = hmac.compare_digest(expected_hash, computed_hash)

    if not is_match or pending["password_hash"] == "DUMMY_INERT":
        attempts = int(pending.get("attempts") or 0) + 1
        if attempts >= settings.verification_max_attempts:
            database.execute_write("DELETE FROM pending_registrations WHERE id = ?", (pending_id,))
            return AuthResult(
                success=False,
                error_message="Verification code expired. Please request a new code.",
            )
        database.execute_write(
            "UPDATE pending_registrations SET attempts = ? WHERE id = ?",
            (attempts, pending_id),
        )
        return AuthResult(
            success=False,
            error_message="Invalid verification code. Please try again.",
        )

    # Verification successful: insert or update user in a single atomic transaction
    email = pending["email"]
    display_name = pending["display_name"]
    pwd_hash = pending["password_hash"]

    try:
        with database.transaction() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE email = ? COLLATE NOCASE", (email,)
            ).fetchone()

            if existing:
                user_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE users
                    SET email_verified = 1, email_verified_at = ?, password_hash = ?, has_usable_password = 1, last_login_at = ?
                    WHERE id = ?
                    """,
                    (now_iso, pwd_hash, now_iso, user_id),
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO users
                    (email, display_name, password_hash, created_at, failed_attempts, email_verified, email_verified_at, has_usable_password, last_login_at)
                    VALUES (?, ?, ?, ?, 0, 1, ?, 1, ?)
                    """,
                    (email, display_name, pwd_hash, now_iso, now_iso, now_iso),
                )
                user_id = cursor.lastrowid

            # Delete pending registration (single-use)
            conn.execute("DELETE FROM pending_registrations WHERE id = ?", (pending_id,))

            # Issue session token inside same transaction
            session_token = secrets.token_urlsafe(32)
            token_h = hash_token(session_token)
            session_expires = (
                now_dt + datetime.timedelta(minutes=settings.session_ttl_minutes)
            ).isoformat()
            conn.execute(
                """
                INSERT INTO sessions (token_hash, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token_h, user_id, now_iso, session_expires),
            )

    except sqlite3.IntegrityError:
        logger.error("Concurrent registration collision for email %s", email)
        return AuthResult(
            success=False,
            error_message="Registration failed due to a conflict. Please try again.",
        )
    except Exception as exc:
        logger.error("Failed to complete user registration: %s", exc)
        return AuthResult(
            success=False,
            error_message="Registration failed due to a database error. Please try again.",
        )

    user = User(
        id=user_id,
        email=email,
        display_name=display_name,
        created_at=now_iso,
        has_usable_password=1,
    )
    return AuthResult(success=True, user=user, token=session_token)


def resend_registration_code(
    pending_id: str, db: Optional[AppDatabase] = None
) -> Tuple[bool, Optional[str]]:
    """Generate and dispatch a new verification code, invalidating previous codes and resetting attempts."""
    database = db or get_db()
    if not pending_id:
        return False, "Verification session expired or not found."

    auth_secret = _get_auth_secret()
    if len(auth_secret) < 32:
        return False, "Server authentication secret (VRS_AUTH_SECRET) is missing or shorter than 32 characters."

    pending = database.execute_read_one(
        "SELECT * FROM pending_registrations WHERE id = ?", (pending_id,)
    )
    if not pending:
        return False, "Verification session expired or not found."

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_dt.isoformat()

    # Enforce resend cooldown
    try:
        last_sent_dt = datetime.datetime.fromisoformat(pending["last_sent_at"])
        elapsed = (now_dt - last_sent_dt).total_seconds()
        if elapsed < settings.resend_cooldown_seconds:
            remaining = int(settings.resend_cooldown_seconds - elapsed) + 1
            return False, f"Please wait {remaining} second(s) before requesting a new code."
    except Exception:
        pass

    # Check send rate limits
    email = pending["email"]
    can_send, rate_err = _check_send_rate_limits(email, database, now_dt)
    if not can_send:
        return False, rate_err

    # Inert pending registration (duplicate account notice)
    if pending["password_hash"] == "DUMMY_INERT":
        try:
            mailer = get_mailer()
            mailer.send_security_notice(email)
            database.execute_write(
                "INSERT INTO email_send_log (email, sent_at, kind) VALUES (?, ?, ?)",
                (email, now_iso, "security_notice"),
            )
            database.execute_write(
                "UPDATE pending_registrations SET last_sent_at = ? WHERE id = ?",
                (now_iso, pending_id),
            )
            return True, None
        except Exception:
            return False, "We couldn't send the verification email. Check the email address or try again."

    # Generate fresh verification code and extend expiration
    code = "".join(str(secrets.randbelow(10)) for _ in range(settings.verification_code_length))
    salt = secrets.token_hex(16)
    code_hash = _hash_verification_code(pending_id, salt, code, auth_secret)
    code_expires_at = (now_dt + datetime.timedelta(minutes=settings.verification_ttl_minutes)).isoformat()

    try:
        mailer = get_mailer()
        mailer.send_verification_email(email, code, settings.verification_ttl_minutes)
    except EmailConfigurationError as exc:
        return False, str(exc)
    except Exception as exc:
        logger.error("Failed to send verification code resend: %s", type(exc).__name__)
        return False, "We couldn't send the verification email. Check the email address or try again."

    database.execute_write(
        "INSERT INTO email_send_log (email, sent_at, kind) VALUES (?, ?, ?)",
        (email, now_iso, "resend"),
    )
    database.execute_write(
        """
        UPDATE pending_registrations
        SET code_hash = ?, code_salt = ?, code_expires_at = ?, last_sent_at = ?,
            attempts = 0, resend_count = resend_count + 1
        WHERE id = ?
        """,
        (code_hash, salt, code_expires_at, now_iso, pending_id),
    )
    return True, None


def cancel_registration(
    pending_id: str, db: Optional[AppDatabase] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Cancel pending registration row and return (display_name, email) for form prefill."""
    database = db or get_db()
    if not pending_id:
        return None, None

    pending = database.execute_read_one(
        "SELECT display_name, email FROM pending_registrations WHERE id = ?", (pending_id,)
    )
    database.execute_write("DELETE FROM pending_registrations WHERE id = ?", (pending_id,))
    if pending:
        return pending["display_name"], pending["email"]
    return None, None


def register_user(
    email: str,
    password: str,
    confirm_password: str,
    display_name: Optional[str] = None,
    db: Optional[AppDatabase] = None,
) -> AuthResult:
    """Trusted-caller and test helper to create an immediately verified user account.
    
    The user-facing UI must not call this directly; it calls start_registration().
    """
    database = db or get_db()
    email_clean = (email or "").strip().lower()

    if not email_clean or not _EMAIL_REGEX.match(email_clean):
        return AuthResult(success=False, error_message="Please enter a valid email address.")

    if password != confirm_password:
        return AuthResult(success=False, error_message="Passwords do not match.")

    is_strong, strength_err = validate_password_strength(password)
    if not is_strong:
        return AuthResult(success=False, error_message=strength_err)

    existing = database.execute_read_one(
        "SELECT id FROM users WHERE email = ? COLLATE NOCASE", (email_clean,)
    )
    if existing:
        return AuthResult(
            success=False,
            error_message="An account with this email address already exists.",
        )

    disp = (display_name or "").strip() or email_clean.split("@")[0].capitalize()
    pwd_hash = hash_password(password)
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_dt.isoformat()

    try:
        user_id = database.execute_write(
            """
            INSERT INTO users (email, display_name, password_hash, created_at, failed_attempts, email_verified, email_verified_at, has_usable_password)
            VALUES (?, ?, ?, ?, 0, 1, ?, 1)
            """,
            (email_clean, disp, pwd_hash, now_iso, now_iso),
        )
    except Exception:
        return AuthResult(
            success=False,
            error_message="Registration failed due to a database error. Please try again.",
        )

    session_token = _issue_session(user_id, database, now_dt)
    user = User(
        id=user_id,
        email=email_clean,
        display_name=disp,
        created_at=now_iso,
        has_usable_password=1,
    )
    return AuthResult(success=True, user=user, token=session_token)


def login_user(
    email: str,
    password: str,
    db: Optional[AppDatabase] = None,
) -> AuthResult:
    """Authenticate credentials with lockout protection, legacy account verification, and generic error."""
    database = db or get_db()
    email_clean = (email or "").strip().lower()

    if not email_clean or not password:
        return AuthResult(success=False, error_message="Invalid email or password.")

    user_row = database.execute_read_one(
        "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email_clean,)
    )
    if not user_row:
        return AuthResult(success=False, error_message="Invalid email or password.")

    user_id = int(user_row["id"])
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_dt.isoformat()

    # Check account lockout
    locked_until = user_row.get("locked_until")
    if locked_until:
        try:
            lock_dt = datetime.datetime.fromisoformat(locked_until)
            if lock_dt > now_dt:
                rem_seconds = int((lock_dt - now_dt).total_seconds())
                rem_minutes = max(1, (rem_seconds + 59) // 60)
                return AuthResult(
                    success=False,
                    error_message=f"Account is temporarily locked due to multiple failed login attempts. Please try again in {rem_minutes} minute(s).",
                )
        except Exception:
            pass

    # Verify password against hash
    is_valid = verify_password(user_row["password_hash"], password)
    if not is_valid:
        failed_attempts = int(user_row.get("failed_attempts") or 0) + 1
        new_locked_until = None
        if failed_attempts >= settings.lockout_attempts:
            new_locked_until = (
                now_dt + datetime.timedelta(minutes=settings.lockout_duration_minutes)
            ).isoformat()

        database.execute_write(
            """
            UPDATE users
            SET failed_attempts = ?, locked_until = ?
            WHERE id = ?
            """,
            (failed_attempts, new_locked_until, user_id),
        )
        return AuthResult(success=False, error_message="Invalid email or password.")

    # Check email verification status (legacy accounts created before migration have email_verified = 0)
    email_verified = int(user_row.get("email_verified") or 0)
    if settings.require_email_verification and email_verified == 0:
        # User has valid password but unverified email -> route to verification flow without issuing session
        auth_secret = _get_auth_secret()
        if len(auth_secret) < 32:
            return AuthResult(
                success=False,
                error_message="Server authentication secret (VRS_AUTH_SECRET) is missing or shorter than 32 characters.",
            )

        code = "".join(str(secrets.randbelow(10)) for _ in range(settings.verification_code_length))
        salt = secrets.token_hex(16)
        pending_id = secrets.token_hex(16)
        code_hash = _hash_verification_code(pending_id, salt, code, auth_secret)
        code_expires_at = (now_dt + datetime.timedelta(minutes=settings.verification_ttl_minutes)).isoformat()

        try:
            mailer = get_mailer()
            mailer.send_verification_email(email_clean, code, settings.verification_ttl_minutes)
        except EmailConfigurationError as exc:
            return AuthResult(success=False, error_message=str(exc))
        except Exception as exc:
            logger.error("Failed to send legacy account verification email: %s", type(exc).__name__)
            return AuthResult(
                success=False,
                error_message="We couldn't send the verification email. Check the email address or try again.",
            )

        database.execute_write(
            "DELETE FROM pending_registrations WHERE email = ? COLLATE NOCASE", (email_clean,)
        )
        database.execute_write(
            """
            INSERT INTO pending_registrations
            (id, email, display_name, password_hash, code_hash, code_salt, created_at, code_expires_at, last_sent_at, attempts, resend_count, resend_window_started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
            """,
            (pending_id, email_clean, user_row["display_name"], user_row["password_hash"], code_hash, salt, now_iso, code_expires_at, now_iso, now_iso),
        )
        database.execute_write(
            "INSERT INTO email_send_log (email, sent_at, kind) VALUES (?, ?, ?)",
            (email_clean, now_iso, "legacy_verification"),
        )

        return AuthResult(
            success=True,
            pending_id=pending_id,
            email=email_clean,
            requires_verification=True,
        )

    # Successful login: reset failed attempts, update last_login_at
    database.execute_write(
        """
        UPDATE users
        SET failed_attempts = 0, locked_until = NULL, last_login_at = ?
        WHERE id = ?
        """,
        (now_iso, user_id),
    )

    session_token = _issue_session(user_id, database, now_dt)
    user = User(
        id=user_id,
        email=user_row["email"],
        display_name=user_row["display_name"],
        created_at=user_row["created_at"],
        has_usable_password=int(user_row.get("has_usable_password", 1)),
    )
    return AuthResult(success=True, user=user, token=session_token)


def google_sign_in(
    id_token_str: Optional[str],
    db: Optional[AppDatabase] = None,
) -> AuthResult:
    """Validate Google ID token claims and complete account linking or creation."""
    is_enabled = (
        os.getenv("VRS_ENABLE_GOOGLE", "").lower() in ("true", "1", "yes")
        if "VRS_ENABLE_GOOGLE" in os.environ
        else settings.enable_google
    )
    if not is_enabled:
        return AuthResult(success=False, error_message="Google sign-in is not enabled.")

    if not id_token_str or not isinstance(id_token_str, str):
        return AuthResult(
            success=False,
            error_message="We couldn't verify your Google account. Please try again.",
        )

    # Determine client_id from Streamlit secrets
    client_id = None
    try:
        from streamlit.runtime.secrets import secrets_singleton
        auth_sec = secrets_singleton.get("auth", {})
        client_id = auth_sec.get("client_id")
    except Exception:
        pass

    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        claims = id_token.verify_oauth2_token(
            id_token_str,
            google_requests.Request(),
            audience=client_id,
        )
    except Exception as exc:
        logger.error("Google ID token verification failed: %s", exc)
        return AuthResult(
            success=False,
            error_message="We couldn't verify your Google account. Please try again.",
        )

    # Validate claims strictly
    issuer = claims.get("iss", "")
    if issuer not in ("accounts.google.com", "https://accounts.google.com"):
        logger.error("Invalid token issuer: %s", issuer)
        return AuthResult(
            success=False,
            error_message="We couldn't verify your Google account. Please try again.",
        )

    if claims.get("email_verified") is not True:
        logger.warning("Google account email is not verified by Google.")
        return AuthResult(
            success=False,
            error_message="Your Google account email is not verified.",
        )

    sub = str(claims.get("sub") or "").strip()
    email = str(claims.get("email") or "").strip().lower()
    name = (claims.get("name") or "").strip() or email.split("@")[0].capitalize()

    if not sub or not email:
        logger.error("Missing required claims in Google ID token: sub=%s, email=%s", bool(sub), bool(email))
        return AuthResult(
            success=False,
            error_message="We couldn't verify your Google account. Please try again.",
        )

    database = db or get_db()
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_dt.isoformat()

    # Rule 1: Match by google_sub first
    user_by_sub = database.execute_read_one(
        "SELECT * FROM users WHERE google_sub = ?", (sub,)
    )

    if user_by_sub:
        user_row = user_by_sub
    else:
        # Check if account exists with this email
        user_by_email = database.execute_read_one(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)
        )

        if user_by_email:
            # Rule 6: If google_sub is already linked to a different account than this email
            if user_by_email.get("google_sub") and user_by_email["google_sub"] != sub:
                logger.error("Sub conflict: sub %s attempted to link to email already linked to %s", sub, user_by_email["google_sub"])
                return AuthResult(
                    success=False,
                    error_message="This email is linked to a different Google account.",
                )

            # Rule 2: Local account has email_verified = 1 -> safe to link
            if int(user_by_email.get("email_verified") or 0) == 1:
                database.execute_write(
                    "UPDATE users SET google_sub = ? WHERE id = ?",
                    (sub, user_by_email["id"]),
                )
                user_row = user_by_email
            else:
                # Rule 3: Local account has email_verified = 0 (takeover defense)
                # Revoke all existing sessions and overwrite password with unusable random Argon2id hash
                database.execute_write(
                    "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                    (now_iso, user_by_email["id"]),
                )
                unusable_pwd = secrets.token_hex(32)
                unusable_hash = hash_password(unusable_pwd)
                database.execute_write(
                    """
                    UPDATE users
                    SET google_sub = ?, password_hash = ?, has_usable_password = 0,
                        email_verified = 1, email_verified_at = ?
                    WHERE id = ?
                    """,
                    (sub, unusable_hash, now_iso, user_by_email["id"]),
                )
                user_row = database.execute_read_one("SELECT * FROM users WHERE id = ?", (user_by_email["id"],))
        else:
            # Rule 4: Create new user account with unusable random password hash
            unusable_pwd = secrets.token_hex(32)
            unusable_hash = hash_password(unusable_pwd)
            try:
                user_id = database.execute_write(
                    """
                    INSERT INTO users
                    (email, display_name, password_hash, created_at, failed_attempts, email_verified, email_verified_at, google_sub, has_usable_password)
                    VALUES (?, ?, ?, ?, 0, 1, ?, ?, 0)
                    """,
                    (email, name, unusable_hash, now_iso, now_iso, sub),
                )
                user_row = {
                    "id": user_id,
                    "email": email,
                    "display_name": name,
                    "created_at": now_iso,
                    "has_usable_password": 0,
                }
            except sqlite3.IntegrityError:
                return AuthResult(
                    success=False,
                    error_message="Google sign-in could not be completed due to an account conflict.",
                )

    # Rule 5: Purge any pending registration for that email
    database.execute_write("DELETE FROM pending_registrations WHERE email = ? COLLATE NOCASE", (email,))

    # Rule 7: Update last_login_at and issue normal app session
    user_id = int(user_row["id"])
    database.execute_write(
        "UPDATE users SET last_login_at = ? WHERE id = ?",
        (now_iso, user_id),
    )
    session_token = _issue_session(user_id, database, now_dt)

    user = User(
        id=user_id,
        email=user_row["email"],
        display_name=user_row["display_name"],
        created_at=user_row["created_at"],
        has_usable_password=int(user_row.get("has_usable_password", 1)),
    )
    return AuthResult(success=True, user=user, token=session_token)


def validate_session(
    token: Optional[str],
    db: Optional[AppDatabase] = None,
) -> Optional[User]:
    """Validate opaque session token against SQLite sessions table and verify email status."""
    if not token or not isinstance(token, str):
        return None

    database = db or get_db()
    token_h = hash_token(token)
    row = database.execute_read_one(
        """
        SELECT s.token_hash, s.expires_at, s.revoked_at,
               u.id AS user_id, u.email, u.display_name, u.created_at,
               u.email_verified, u.has_usable_password
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.token_hash = ?
        """,
        (token_h,),
    )
    if not row:
        return None

    if row["revoked_at"] is not None:
        return None

    # Defense-in-depth: unverified users cannot reach protected application features
    if settings.require_email_verification and int(row.get("email_verified") or 0) != 1:
        return None

    # Check session expiration
    try:
        expires_at = datetime.datetime.fromisoformat(row["expires_at"])
        if datetime.datetime.now(datetime.timezone.utc) > expires_at:
            return None
    except Exception:
        return None

    return User(
        id=int(row["user_id"]),
        email=row["email"],
        display_name=row["display_name"],
        created_at=row["created_at"],
        has_usable_password=int(row.get("has_usable_password", 1)),
    )


def logout_session(
    token: Optional[str],
    db: Optional[AppDatabase] = None,
) -> bool:
    """Revoke session token in database."""
    if not token:
        return False

    database = db or get_db()
    token_h = hash_token(token)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows_affected = database.execute_write(
        "UPDATE sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
        (now_iso, token_h),
    )
    return rows_affected > 0


def change_password(
    user_id: int,
    current_password: str,
    new_password: str,
    confirm_password: str,
    db: Optional[AppDatabase] = None,
) -> Tuple[bool, Optional[str]]:
    """Update user password with current password verification."""
    database = db or get_db()
    user_row = database.execute_read_one(
        "SELECT password_hash, has_usable_password FROM users WHERE id = ?", (user_id,)
    )
    if not user_row:
        return False, "User not found."

    if not verify_password(user_row["password_hash"], current_password):
        return False, "Current password is incorrect."

    if new_password != confirm_password:
        return False, "New passwords do not match."

    is_strong, strength_err = validate_password_strength(new_password)
    if not is_strong:
        return False, strength_err

    new_hash = hash_password(new_password)
    database.execute_write(
        "UPDATE users SET password_hash = ?, has_usable_password = 1 WHERE id = ?",
        (new_hash, user_id),
    )
    return True, None


def set_password(
    user_id: int,
    new_password: str,
    confirm_password: str,
    db: Optional[AppDatabase] = None,
) -> Tuple[bool, Optional[str]]:
    """Set initial password for an account that currently signs in with Google only."""
    database = db or get_db()
    if new_password != confirm_password:
        return False, "New passwords do not match."

    is_strong, strength_err = validate_password_strength(new_password)
    if not is_strong:
        return False, strength_err

    new_hash = hash_password(new_password)
    database.execute_write(
        "UPDATE users SET password_hash = ?, has_usable_password = 1 WHERE id = ?",
        (new_hash, user_id),
    )
    return True, None
