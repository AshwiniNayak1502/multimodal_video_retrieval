"""Standalone SMTP connectivity and authentication smoke test.

Performs staged verification: Connect -> EHLO -> STARTTLS/SSL -> EHLO -> Login -> (optional test send) -> Quit.
No secrets are logged or embedded.
"""

from __future__ import annotations

import argparse
import email.utils
import os
import socket
import ssl
import sys
import time
from email.message import EmailMessage
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import settings


def mask_email_display(email_addr: str) -> str:
    """Mask email for safe console printing (e.g. u***@example.com)."""
    if not email_addr or "@" not in email_addr:
        return "<redacted>"
    local, domain = email_addr.split("@", 1)
    masked_local = (local[0] + "***" + (local[-1] if len(local) > 1 else "")) if local else "***"
    return f"{masked_local}@{domain}"


def classify_error(exc: Exception) -> str:
    """Classify SMTP/network exceptions into standardized categories."""
    import smtplib
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return "AUTH"
    if isinstance(exc, (smtplib.SMTPConnectError, ConnectionRefusedError, ConnectionResetError)):
        return "CONNECT"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "TIMEOUT"
    if isinstance(exc, ssl.SSLError):
        return "TLS"
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return "RECIPIENT"
    if isinstance(exc, smtplib.SMTPServerDisconnected):
        return "TRANSIENT"
    if isinstance(exc, smtplib.SMTPResponseException):
        code = getattr(exc, "smtp_code", -1)
        if 400 <= code < 500:
            return "TRANSIENT"
        if code in (550, 551, 552, 553, 554):
            msg = str(exc)
            if "quota" in msg.lower() or "limit" in msg.lower():
                return "QUOTA"
            return "RECIPIENT"
    return "UNKNOWN"


def run_selfcheck(to_email: str | None = None, timeout: float = 15.0) -> int:
    import smtplib

    host = os.getenv("VRS_SMTP_HOST") or settings.smtp_host
    port = int(os.getenv("VRS_SMTP_PORT") or settings.smtp_port or 587)
    security = (os.getenv("VRS_SMTP_SECURITY") or settings.smtp_security or "starttls").lower()
    username = os.getenv("VRS_SMTP_USERNAME") or settings.smtp_username
    password = (os.getenv("VRS_SMTP_PASSWORD") or settings.smtp_password or "").strip()
    from_addr = os.getenv("VRS_EMAIL_FROM") or settings.email_from or username

    print("=" * 60)
    print(" VideoContextEngine SMTP Self-Check")
    print("=" * 60)
    print(f"Host:       {host}:{port}")
    print(f"Security:   {security}")
    print(f"Username:   {mask_email_display(username)}")
    print(f"Password:   {'set (' + str(len(password)) + ' chars, no whitespace)' if password and ' ' not in password else ('set with spaces' if ' ' in password else 'not set')}")
    print(f"From:       {mask_email_display(from_addr)}")
    if to_email:
        print(f"Test Recipient: {mask_email_display(to_email)}")
    print("-" * 60)

    if not host or not username or not password:
        print("STAGE [CONFIG]: FAILED - Missing required SMTP settings (host, username, or password)")
        return 1
    print("STAGE [CONFIG]: OK")

    server = None
    stage = "INITIALIZE"
    try:
        # Stage 1: TCP Connect
        stage = "CONNECT"
        t0 = time.perf_counter()
        context = ssl.create_default_context()
        if security == "ssl":
            server = smtplib.SMTP_SSL(host, port, context=context, timeout=timeout)
        else:
            server = smtplib.SMTP(host, port, timeout=timeout)
        elapsed_connect = (time.perf_counter() - t0) * 1000
        print(f"STAGE [CONNECT]: OK ({elapsed_connect:.1f}ms)")

        # Stage 2: Initial EHLO
        if security != "ssl":
            stage = "EHLO_1"
            t0 = time.perf_counter()
            code, msg = server.ehlo()
            elapsed_ehlo1 = (time.perf_counter() - t0) * 1000
            print(f"STAGE [EHLO_1]: OK (code {code}, {elapsed_ehlo1:.1f}ms)")

            # Stage 3: STARTTLS
            stage = "STARTTLS"
            t0 = time.perf_counter()
            code, msg = server.starttls(context=context)
            elapsed_tls = (time.perf_counter() - t0) * 1000
            print(f"STAGE [STARTTLS]: OK (code {code}, {elapsed_tls:.1f}ms)")

            # Stage 4: Post-STARTTLS EHLO
            stage = "EHLO_2"
            t0 = time.perf_counter()
            code, msg = server.ehlo()
            elapsed_ehlo2 = (time.perf_counter() - t0) * 1000
            print(f"STAGE [EHLO_2]: OK (code {code}, {elapsed_ehlo2:.1f}ms)")

        # Stage 5: Authentication
        stage = "LOGIN"
        t0 = time.perf_counter()
        code, msg = server.login(username, password)
        elapsed_login = (time.perf_counter() - t0) * 1000
        print(f"STAGE [LOGIN]: OK (code {code}, {elapsed_login:.1f}ms)")

        # Stage 6: Optional Send
        if to_email:
            stage = "SEND"
            t0 = time.perf_counter()
            em = EmailMessage()
            em["From"] = email.utils.formataddr((settings.email_from_name, from_addr))
            em["To"] = to_email
            em["Subject"] = f"[{settings.email_from_name}] SMTP Test Diagnostic"
            em["Date"] = email.utils.formatdate(localtime=False)
            em["Message-ID"] = email.utils.make_msgid(domain=host if "." in host else "localhost")
            em.set_content("This is an automated smoke test verification from VideoContextEngine.")
            
            refused = server.send_message(em)
            elapsed_send = (time.perf_counter() - t0) * 1000
            if refused:
                print(f"STAGE [SEND]: FAILED - Recipients refused: {list(refused.keys())} ({elapsed_send:.1f}ms)")
                return 1
            print(f"STAGE [SEND]: OK ({elapsed_send:.1f}ms)")

        # Stage 7: QUIT
        stage = "QUIT"
        try:
            server.quit()
        except Exception:
            pass
        print("STAGE [QUIT]: OK")

        print("=" * 60)
        print("RESULT: SUCCESS - All verified stages passed.")
        print("=" * 60)
        return 0

    except Exception as exc:
        category = classify_error(exc)
        smtp_code = getattr(exc, "smtp_code", None)
        print(f"STAGE [{stage}]: FAILED")
        print(f"Category:   {category}")
        print(f"Exception:  {type(exc).__name__}")
        if smtp_code is not None:
            print(f"SMTP Code:  {smtp_code}")
        # Redacted error message (strip passwords or credentials if echoed)
        safe_msg = str(exc)
        if password and password in safe_msg:
            safe_msg = safe_msg.replace(password, "<REDACTED>")
        print(f"Message:    {safe_msg[:200]}")
        print("=" * 60)
        return 1
    finally:
        if server:
            try:
                server.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoContextEngine SMTP Diagnostic Smoke Check")
    parser.add_argument("--to", help="Optional recipient address for test message dispatch", default=None)
    parser.add_argument("--timeout", type=float, default=15.0, help="Connection timeout in seconds")
    args = parser.parse_args()

    sys.exit(run_selfcheck(to_email=args.to, timeout=args.timeout))


if __name__ == "__main__":
    main()
