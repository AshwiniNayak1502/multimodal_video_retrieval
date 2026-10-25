"""Authentication and session management service backed by Argon2id and SQLite."""

from __future__ import annotations

import datetime
import hashlib
import re
import secrets
from dataclasses import dataclass
from typing import Optional, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from ..db.app_db import AppDatabase, get_db
from ..settings import settings

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


@dataclass
class AuthResult:
    """Result container for authentication workflows."""

    success: bool
    error_message: Optional[str] = None
    user: Optional[User] = None
    token: Optional[str] = None


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


def register_user(
    email: str,
    password: str,
    confirm_password: str,
    display_name: Optional[str] = None,
    db: Optional[AppDatabase] = None,
) -> AuthResult:
    """Register a new user account with strict validation."""
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
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        user_id = database.execute_write(
            """
            INSERT INTO users (email, display_name, password_hash, created_at, failed_attempts)
            VALUES (?, ?, ?, ?, 0)
            """,
            (email_clean, disp, pwd_hash, now_iso),
        )
    except Exception as exc:
        return AuthResult(
            success=False,
            error_message="Registration failed due to a database error. Please try again.",
        )

    # Auto-login upon registration: issue session token
    session_token = secrets.token_urlsafe(32)
    token_h = hash_token(session_token)
    expires_at = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(minutes=settings.session_ttl_minutes)
    ).isoformat()

    database.execute_write(
        """
        INSERT INTO sessions (token_hash, user_id, created_at, expires_at)
        VALUES (?, ?, ?, ?)
        """,
        (token_h, user_id, now_iso, expires_at),
    )

    user = User(id=user_id, email=email_clean, display_name=disp, created_at=now_iso)
    return AuthResult(success=True, user=user, token=session_token)


def login_user(
    email: str,
    password: str,
    db: Optional[AppDatabase] = None,
) -> AuthResult:
    """Authenticate user credentials with lockout protection and generic failure message."""
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

    # Check lockout
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

    # Verify password
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

    # Successful login: reset failed attempts, update last_login_at
    database.execute_write(
        """
        UPDATE users
        SET failed_attempts = 0, locked_until = NULL, last_login_at = ?
        WHERE id = ?
        """,
        (now_iso, user_id),
    )

    # Generate session token
    session_token = secrets.token_urlsafe(32)
    token_h = hash_token(session_token)
    expires_at = (
        now_dt + datetime.timedelta(minutes=settings.session_ttl_minutes)
    ).isoformat()

    database.execute_write(
        """
        INSERT INTO sessions (token_hash, user_id, created_at, expires_at)
        VALUES (?, ?, ?, ?)
        """,
        (token_h, user_id, now_iso, expires_at),
    )

    user = User(
        id=user_id,
        email=user_row["email"],
        display_name=user_row["display_name"],
        created_at=user_row["created_at"],
    )
    return AuthResult(success=True, user=user, token=session_token)


def validate_session(
    token: Optional[str],
    db: Optional[AppDatabase] = None,
) -> Optional[User]:
    """Validate opaque session token against SQLite sessions table."""
    if not token or not isinstance(token, str):
        return None

    database = db or get_db()
    token_h = hash_token(token)
    row = database.execute_read_one(
        """
        SELECT s.token_hash, s.expires_at, s.revoked_at,
               u.id AS user_id, u.email, u.display_name, u.created_at
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

    # Check expiry
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
        "SELECT password_hash FROM users WHERE id = ?", (user_id,)
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
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (new_hash, user_id),
    )
    return True, None
