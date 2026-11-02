# VideoContextEngine Authentication & Registration Trace

This document provides a line-by-line inspection and trace of the authentication and registration flow in `VideoContextEngine`.

---

## 1. Create Account Click & Validation Flow

- **Trigger (`app/ui/auth_view.py:235-250`)**:
  - User submits the "Create Account" tab form via `st.button("Create Account", key="reg_btn")`.
  - Client-side validation:
    - Line 236: `reg_name.strip()` must not be empty (`"Please provide a name."`).
    - Line 238: `reg_email.strip()` must not be empty (`"Please provide an email address."`).
    - Line 240: `reg_password == reg_confirm` (`"Passwords do not match."`).
  - Line 243-250: If valid, enters spinner `"Sending verification email…"` and calls `start_registration(email=reg_email, password=reg_password, confirm_password=reg_confirm, display_name=reg_name.strip(), db=database)`.

- **Handler (`app/services/auth_service.py:187-231`)**:
  - Line 196: Normalizes email with `email_clean = (email or "").strip().lower()`.
  - Line 197: Normalizes display name with `disp = (display_name or "").strip()`.
  - Line 199-200: Validates display name presence.
  - Line 202-203: Validates email regex `^[^@\s]+@[^@\s]+\.[^@\s]+$`.
  - Line 205-206: Re-checks password confirmation equality.
  - Line 208-210: Calls `validate_password_strength(password)`:
    - Minimum length: 8 characters.
    - Must contain at least one letter and at least one digit.
  - Line 212-217: Retrieves `auth_secret` via `_get_auth_secret()`. Checks `len(auth_secret) >= 32`.
  - Line 219-221: Generates `now_dt = datetime.datetime.now(datetime.timezone.utc)`, calls `_purge_expired_pending(database, now_iso)`.
  - Line 224-231: If `not settings.require_email_verification`, immediately delegates to `register_user(...)`.

---

## 2. Existing-Account Lookup

- **Lookup Query (`app/services/auth_service.py:234-236`)**:
  ```sql
  SELECT * FROM users WHERE email = ? COLLATE NOCASE
  ```
  - Compared with `email_clean` (lowercased and stripped).
  - Note on Gmail plus-addressing (`user+tag@gmail.com`): It is treated as a distinct email string because no plus-address normalization is applied.
- **Google-created Users**:
  - Stored in the **same** `users` table (`app/db/schema.sql:5-18`).
  - Google accounts created via `google_sign_in` (`app/services/auth_service.py:901-916`) have:
    - `email_verified = 1`
    - `google_sub = <Google Subject ID>` (enforced by partial unique index `idx_users_google_sub`)
    - `has_usable_password = 0`
    - `password_hash = Argon2id hash of 64-char random hex token` (unusable).
- **Unverified vs Pending Users**:
  - In-flight manual registrations reside strictly in `pending_registrations` table, **not** in `users`.
  - Legacy users created prior to migration may exist in `users` with `email_verified = 0`.
  - In `start_registration` (line 237), the branch check is:
    ```python
    if existing_user and int(existing_user.get("email_verified") or 0) == 1:
    ```
    If an existing user in `users` has `email_verified == 0`, it does not enter Branch A; it falls through to Branch B (which would subsequently trigger a duplicate conflict on verification).

---

## 3. Branch A (Existing Account) vs Branch B (New Email)

### Branch A: Existing Account (`app/services/auth_service.py:237-281`)
1. **Timing equalization**: Computes `_ = hash_password(password)` (Argon2id) to match computation duration of new registrations.
2. **Security Notice dispatch**:
   ```python
   mailer = get_mailer()
   mailer.send_security_notice(email_clean)
   database.execute_write(
       "INSERT INTO email_send_log (email, sent_at, kind) VALUES (?, ?, ?)",
       (email_clean, now_iso, "security_notice"),
   )
   ```
3. **Exception collapse (Defect)**:
   ```python
   except Exception as exc:
       logger.error("Failed to send duplicate security notice: %s", type(exc).__name__)
       return AuthResult(
           success=False,
           error_message="We couldn't send the verification email. Check the email address or try again.",
       )
   ```
4. **Dummy Pending State**:
   - Generates random `dummy_pending_id` (32-hex) and `dummy_salt`.
   - Computes HMAC-SHA256 digest over `"DUMMY"`.
   - Replaces any pending row in `pending_registrations` with `password_hash = "DUMMY_INERT"`.
5. **UI Return**:
   - Returns `AuthResult(success=True, pending_id=dummy_pending_id, email=email_clean, requires_verification=True)`.
   - **Deceptive state**: The UI tells the user a verification code was sent, but no OTP was generated or sent (only a Security Notice).

### Branch B: New Email (`app/services/auth_service.py:283-331`)
1. **Rate Limiting (`lines 284-286`)**:
   - Calls `_check_send_rate_limits(email_clean, database, now_dt)`.
   - Checks global hourly cap (`settings.email_max_per_hour = 50`) and per-email hourly cap (`settings.resend_max_per_hour = 5`).
2. **OTP Generation (`lines 289-295`)**:
   - 6-digit numeric string using `secrets.randbelow(10)`.
   - 16-byte random salt (`secrets.token_hex(16)`).
   - Random `pending_id` (`secrets.token_hex(16)`).
   - HMAC-SHA256 hash using `VRS_AUTH_SECRET`: `hmac.new(secret, f"{pending_id}:{salt}:{code}", sha256)`.
   - Argon2id hash of password.
   - Expiration set to `now_dt + settings.verification_ttl_minutes` (default 10 min).
3. **SMTP Send (`lines 297-307`)**:
   - Calls `mailer.send_verification_email(email_clean, code, settings.verification_ttl_minutes)`.
   - If `Exception` raised, maps to `"We couldn't send the verification email. Check the email address or try again."`.
4. **Storage (`lines 309-324`)**:
   - Executed **only after** successful SMTP send.
   - Inserts row in `email_send_log` (`kind="verification"`).
   - Deletes prior pending registration for `email_clean`.
   - Inserts new row into `pending_registrations`.
5. **UI Return**:
   - Returns `AuthResult(success=True, pending_id=pending_id, email=email_clean, requires_verification=True)`.

---

## 4. Mailer Functions Comparison: Security Notice vs OTP

| Feature / Behavior | Security Notice (`send_security_notice`) | OTP Verification (`send_verification_email`) |
| :--- | :--- | :--- |
| **Declaration** | `app/services/email_service.py:328` | `app/services/email_service.py:301` |
| **Underlying Send Function** | `self._send(to_email, msg)` (`line 256`) | `self._send(to_email, msg)` (`line 256`) |
| **Config Object** | `settings.smtp_*`, `settings.email_from` | `settings.smtp_*`, `settings.email_from` |
| **From Header** | `f'"{settings.email_from_name}" <{settings.email_from or settings.smtp_username}>'` | Identical |
| **Subject** | `f"Security Notice: {settings.email_from_name} Account"` | `f"{settings.email_from_name} Verification Code"` |
| **Date Header** | **Missing** | **Missing** |
| **Message-ID Header** | **Missing** | **Missing** |
| **Display Name Encoding** | Unencoded string interpolation (no `formataddr`) | Unencoded string interpolation (no `formataddr`) |
| **Message Class** | `email.mime.multipart.MIMEMultipart("alternative")` | `email.mime.multipart.MIMEMultipart("alternative")` |
| **Connection Protocol** | STARTTLS (port 587) or SSL (port 465) | Identical |
| **Handshake Sequence** | `connect` → `ehlo()` → `starttls()` → `ehlo()` → `login()` | Identical |
| **Send Method** | `server.sendmail(sender, [to_email], msg.as_string())` | Identical |
| **Refused Recipient Inspection**| Ignored (return dict of `sendmail` is discarded) | Ignored (return dict of `sendmail` is discarded) |
| **Retry Logic** | None | None |
| **Exception Handling in Caller**| Collapsed to generic `"We couldn't send the verification email..."` | Collapsed to generic `"We couldn't send the verification email..."` |

### Why Did the Notice Deliver while the OTP Seemed to Fail?
Because when testing with an existing Google account, **the application only ever executed the Security Notice path**. The OTP generation and send function (`send_verification_email`) was **never invoked** for that submission. The appearance of "OTP failure" was created either by the user observing the Security Notice in Spam instead of the expected OTP, or by a subsequent attempt hitting an SMTP exception or rate limit that returned the generic error string.

---

## 5. UI Result Propagation

Exact conditions under which `"We couldn't send the verification email. Check the email address or try again."` is displayed:

1. **`app/services/auth_service.py:255`**: In `start_registration()`, Branch A (existing account), if `mailer.send_security_notice` or the immediate `INSERT INTO email_send_log` raises any unhandled `Exception` (e.g. `EmailDeliveryError`, `SMTPServerDisconnected`, SQLite busy).
2. **`app/services/auth_service.py:306`**: In `start_registration()`, Branch B (new email), if `mailer.send_verification_email` raises any `Exception` (e.g. `EmailDeliveryError` wrapping 535, timeout, connection drop, socket error).
3. **`app/services/auth_service.py:530`**: In `resend_registration_code()`, if resending a Security Notice (`DUMMY_INERT`) raises any `Exception`.
4. **`app/services/auth_service.py:545`**: In `resend_registration_code()`, if resending an OTP code raises any `Exception`.
5. **`app/services/auth_service.py:726`**: In `login_user()`, if sending a legacy verification email raises any `Exception`.

In all these cases, `auth_service.py` returns `AuthResult(success=False, error_message=...)` or `(False, ...)`, and `app/ui/auth_view.py:263` renders `st.error(res.error_message)`.

---

## 6. Verification Page & Lifecycle

- **Verification Entrypoint (`app/ui/verify_email_view.py:80-116`)**:
  - Displays masked email: `mask_email(email)`.
  - Prompts for `settings.verification_code_length` (6-digit) input.
  - Submits to `verify_registration(pending_id, cleaned_code, db=database)`.
- **Validation Steps (`app/services/auth_service.py:334-475`)**:
  - Loads pending registration by `pending_id`. If absent, returns `"Verification session expired or invalid."`.
  - Validates `code_expires_at` against `datetime.now(timezone.utc)`.
  - Verifies code hash in constant time:
    ```python
    computed_hash = _hash_verification_code(pending_id, pending["code_salt"], submitted_code, auth_secret)
    is_match = hmac.compare_digest(expected_hash, computed_hash)
    ```
  - **Inert dummy trap**: If `pending["password_hash"] == "DUMMY_INERT"` or hash mismatch:
    - Increments `attempts`.
    - If `attempts >= settings.verification_max_attempts` (5): deletes pending row, returns expired/lockout error.
    - Else: updates attempt count in DB, returns `"Invalid verification code. Please try again."`.
  - **Atomic User Activation**:
    - Uses `with database.transaction() as conn:`.
    - If existing user in `users`: updates `email_verified=1`, `password_hash`, `has_usable_password=1`.
    - If new user: inserts row into `users` with `email_verified=1`.
    - Deletes `pending_registrations` row (single-use guarantee).
    - Inserts new session in `sessions` table.
- **Resend Handler (`app/services/auth_service.py:478-561`)**:
  - Checks cooldown (`settings.resend_cooldown_seconds = 60s`).
  - Checks hourly limits via `_check_send_rate_limits`.
  - Generates new random code, new salt, and updates `pending_registrations` row in place. Invalidates previous code.

---

## 7. Login Behavior

- **Unverified Accounts**:
  - Cannot log in.
  - In `login_user()` (`auth_service.py:702-750`), if `email_verified == 0`:
    - Generates OTP code, sends verification email, creates pending registration, and returns `requires_verification=True` without issuing a session token.
  - In `validate_session()` (`auth_service.py:971-972`):
    ```python
    if settings.require_email_verification and int(row.get("email_verified") or 0) != 1:
        return None
    ```
- **Pending Registrations**:
  - Cannot log in because no record exists in `users`. Returns generic `"Invalid email or password."`.
- **Google-created Accounts without Password**:
  - Stored with `has_usable_password = 0` and random unusable Argon2id password hash.
  - Password login fails `verify_password()`. Consecutive attempts trigger standard lockout.
  - Can set password after logging in via Google, or sign in via Google.

---

## 8. Account Deletion

- **Foreign Key Cascades (`app/db/schema.sql`)**:
  - `sessions.user_id REFERENCES users(id) ON DELETE CASCADE`.
  - `videos.user_id REFERENCES users(id) ON DELETE CASCADE`.
  - `PRAGMA foreign_keys = ON;` is enforced on every connection in `AppDatabase._get_connection()`.
- **Orphan Status**:
  - Deleting a user row removes all sessions and video records for that user.
  - `pending_registrations` is not foreign-keyed to `users` (keyed by email). Re-registration deletes any existing pending row for that email (`DELETE FROM pending_registrations WHERE email = ? COLLATE NOCASE`).
  - `email_send_log` retains audit history by email.

---

## 9. SQLite Concurrency & Transactions

- **Connection Factory (`app/db/app_db.py:25-35`)**:
  - `timeout=30.0`
  - `check_same_thread=False`
  - `PRAGMA foreign_keys = ON;`
  - `PRAGMA journal_mode = WAL;`
- **Short-lived Connections**:
  - `execute_read()` and `execute_read_one()` acquire a connection, execute, fetch, and close immediately.
  - `execute_write()` uses `self.transaction()`, which commits on exit and closes in `finally`.
- **Transaction Discipline relative to SMTP**:
  - In `start_registration()`, SMTP send (`mailer.send_verification_email` / `mailer.send_security_notice`) is executed **outside** any database transaction.
  - No database connection or write lock is held open across SMTP network I/O.

---

## 10. Configuration & Environment Loading

- **Loading Order (`app/settings.py:8-25`)**:
  - `dotenv.load_dotenv(override=False)` is invoked at module level without an explicit path.
  - In dev mode, this resolves against `os.getcwd()`.
- **Key Parameters**:
  - `VRS_AUTH_SECRET`: Must be >= 32 characters. Automatically falls back to persistent `<CONFIG_DIR>/secret.key` if empty.
  - `VRS_SMTP_*`: `VRS_SMTP_HOST`, `VRS_SMTP_PORT` (587), `VRS_SMTP_SECURITY` (`starttls`), `VRS_SMTP_USERNAME`, `VRS_SMTP_PASSWORD`, `VRS_EMAIL_FROM`, `VRS_EMAIL_FROM_NAME`.
  - Rate limits: `VRS_RESEND_COOLDOWN_SECONDS` (60), `VRS_RESEND_MAX_PER_HOUR` (5), `VRS_EMAIL_MAX_PER_HOUR` (50).

---

## Branch State Conclusion (Pre-Run Analysis)

Based on static code analysis of `app/services/auth_service.py`, `app/services/email_service.py`, and the database state:
1. The developer attempted manual registration using an email that already existed in `users` with `email_verified = 1` (created via Google OAuth / Continue with Google).
2. The registration handler entered **Branch A** (`auth_service.py:237`).
3. The app invoked `send_security_notice(email_clean)`, which successfully connected to Gmail SMTP and delivered the Security Notice email (landing in Spam due to absence of SPF/DKIM/DMARC and missing standard headers `Date` and `Message-ID`).
4. No OTP was generated or sent.
5. If the Security Notice send encountered any transient SMTP connection hiccup (or if the subsequent retry/resend failed), the broad `except Exception` collapsed the error into `"We couldn't send the verification email. Check the email address or try again."`.
6. Furthermore, even when the Security Notice send succeeded, the UI routed to the verification code entry view claiming a 6-digit verification code had been sent, misleading the user.
