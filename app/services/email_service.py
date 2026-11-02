"""Email delivery service using standard library smtplib with SMTP/STARTTLS/SSL support and diagnostics."""

from __future__ import annotations

import email.utils
from enum import Enum
import logging
import os
import re
import socket
import smtplib
import ssl
import time
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Dict, List, Optional, Protocol, Tuple

from ..settings import settings

logger = logging.getLogger("app.email")


class SmtpErrorCode(str, Enum):
    """Standardized error categories for SMTP and email delivery failures."""
    CONFIG = "CONFIG"
    AUTH = "AUTH"
    CONNECT = "CONNECT"
    TLS = "TLS"
    RECIPIENT = "RECIPIENT"
    QUOTA = "QUOTA"
    TRANSIENT = "TRANSIENT"
    UNKNOWN = "UNKNOWN"


USER_FACING_SMTP_MESSAGES: Dict[SmtpErrorCode, str] = {
    SmtpErrorCode.CONFIG: "Email delivery isn't configured on this server. Please check SMTP settings.",
    SmtpErrorCode.AUTH: "Email service authentication failed. Please verify the sender credentials.",
    SmtpErrorCode.CONNECT: "Unable to reach the email server. Please check your network connection or try again later.",
    SmtpErrorCode.TLS: "Secure connection to email server failed. Please verify security settings.",
    SmtpErrorCode.RECIPIENT: "The recipient address was rejected by the mail server. Please check the address.",
    SmtpErrorCode.QUOTA: "Email sending limit or quota exceeded. Please try again later.",
    SmtpErrorCode.TRANSIENT: "Temporary email delivery failure. Please try again in a few moments.",
    SmtpErrorCode.UNKNOWN: "We couldn't send the verification email. Check the email address or try again.",
}


class EmailConfigurationError(RuntimeError):
    """Raised when SMTP settings are not configured."""
    pass


class EmailDeliveryError(RuntimeError):
    """Raised when email delivery fails technically with categorized diagnostic metadata."""

    def __init__(
        self,
        message: str,
        category: SmtpErrorCode = SmtpErrorCode.UNKNOWN,
        smtp_code: Optional[int] = None,
        refused_recipients: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.smtp_code = smtp_code
        self.refused_recipients = refused_recipients or []


@dataclass
class SmtpDiagnosticResult:
    """Structured result returned by SMTP connectivity diagnosis."""
    success: bool
    step_reached: str
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class EmailDeliveryResult:
    """Result returned by email send operations."""
    success: bool
    error_category: Optional[SmtpErrorCode] = None
    error_message: Optional[str] = None
    smtp_code: Optional[int] = None
    refused_recipients: Optional[List[str]] = None


def classify_smtp_exception(exc: Exception) -> Tuple[SmtpErrorCode, Optional[int], str]:
    """Classify SMTP/network exceptions into standardized category, code, and user-safe message."""
    if isinstance(exc, EmailConfigurationError):
        return SmtpErrorCode.CONFIG, None, str(exc)

    if isinstance(exc, EmailDeliveryError):
        cat = exc.category
        code = exc.smtp_code
        user_msg = USER_FACING_SMTP_MESSAGES.get(cat, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.UNKNOWN])
        return cat, code, user_msg

    if isinstance(exc, smtplib.SMTPAuthenticationError):
        code = getattr(exc, "smtp_code", 535)
        return SmtpErrorCode.AUTH, code, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.AUTH]

    if isinstance(exc, ssl.SSLError):
        return SmtpErrorCode.TLS, None, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.TLS]

    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return SmtpErrorCode.RECIPIENT, 550, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.RECIPIENT]

    if isinstance(exc, smtplib.SMTPServerDisconnected):
        return SmtpErrorCode.TRANSIENT, None, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.TRANSIENT]

    if isinstance(exc, smtplib.SMTPResponseException):
        code = getattr(exc, "smtp_code", -1)
        err_str = str(exc).lower()
        if code in (421, 450, 451, 452) or (400 <= code < 500):
            return SmtpErrorCode.TRANSIENT, code, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.TRANSIENT]
        if "quota" in err_str or "daily limit" in err_str or "5.4.5" in err_str:
            return SmtpErrorCode.QUOTA, code, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.QUOTA]
        if code in (550, 551, 552, 553, 554):
            return SmtpErrorCode.RECIPIENT, code, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.RECIPIENT]
        return SmtpErrorCode.UNKNOWN, code, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.UNKNOWN]

    if isinstance(exc, (socket.timeout, TimeoutError)):
        return SmtpErrorCode.CONNECT, None, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.CONNECT]

    if isinstance(exc, (ConnectionRefusedError, ConnectionResetError)):
        return SmtpErrorCode.CONNECT, None, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.CONNECT]

    # In Python 3.13, SMTPException subclasses OSError. Only treat non-SMTP OS/socket errors as CONNECT.
    if isinstance(exc, (socket.error, OSError)) and not isinstance(exc, smtplib.SMTPException):
        return SmtpErrorCode.CONNECT, None, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.CONNECT]

    return SmtpErrorCode.UNKNOWN, None, USER_FACING_SMTP_MESSAGES[SmtpErrorCode.UNKNOWN]


def mask_email(email_addr: str) -> str:
    """Mask email address for privacy-safe logs and display (e.g. alex@example.com -> a***x@example.com)."""
    if not email_addr or "@" not in email_addr:
        return email_addr or ""
    local, domain = email_addr.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "***"
    else:
        masked_local = local[0] + "***" + local[-1]
    return f"{masked_local}@{domain}"


def sanitize_header_value(val: str) -> str:
    """Strip CR and LF to prevent email header injection attacks."""
    if not val:
        return ""
    return re.sub(r"[\r\n]+", " ", str(val)).strip()


def create_email_message(
    to_email: str,
    subject: str,
    text_content: str,
    html_content: str,
    from_email: str,
    from_name: str,
    host: str,
) -> EmailMessage:
    """Build a standards-compliant (RFC 5322) multipart email with Date, Message-ID, and UTF-8 parts."""
    clean_to = sanitize_header_value(to_email)
    clean_subject = sanitize_header_value(subject)
    clean_from_name = sanitize_header_value(from_name)
    clean_from_email = sanitize_header_value(from_email)

    msg = EmailMessage()
    msg["Subject"] = clean_subject
    msg["From"] = email.utils.formataddr((clean_from_name, clean_from_email))
    msg["To"] = clean_to
    msg["Date"] = email.utils.formatdate(localtime=False)
    domain = host if "." in host else "localhost"
    msg["Message-ID"] = email.utils.make_msgid(domain=domain)

    # Set plain-text body and attach HTML alternative
    msg.set_content(text_content, charset="utf-8")
    msg.add_alternative(html_content, subtype="html", charset="utf-8")
    return msg


class Mailer(Protocol):
    """Protocol for mail delivery backends."""

    def send_verification_email(self, to_email: str, code: str, ttl_minutes: int) -> None:
        """Send verification code email with expiry."""
        ...

    def send_security_notice(self, to_email: str) -> None:
        """Send security notification for duplicate registration attempt."""
        ...


def diagnose_smtp_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    security: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    from_email: Optional[str] = None,
    to_email: Optional[str] = None,
    timeout: float = 15.0,
) -> SmtpDiagnosticResult:
    """Perform step-by-step diagnostic test on SMTP connection, TLS handshake, and credentials."""
    s_host = (host or os.getenv("VRS_SMTP_HOST") or settings.smtp_host or "").strip()
    s_port = int(port or os.getenv("VRS_SMTP_PORT") or settings.smtp_port or 587)
    s_sec = (security or os.getenv("VRS_SMTP_SECURITY") or settings.smtp_security or "starttls").lower().strip()
    s_user = (username or os.getenv("VRS_SMTP_USERNAME") or settings.smtp_username or "").strip()
    # Strip whitespace and surrounding quotes (Google App Passwords often pasted with spaces)
    raw_pwd = password or os.getenv("VRS_SMTP_PASSWORD") or settings.smtp_password or ""
    s_pwd = raw_pwd.strip().strip("'\"")
    s_from = (from_email or os.getenv("VRS_EMAIL_FROM") or settings.email_from or s_user).strip()

    # Step 1: Configuration check
    if not (s_host and s_user and s_pwd):
        missing = []
        if not s_host: missing.append("VRS_SMTP_HOST")
        if not s_user: missing.append("VRS_SMTP_USERNAME")
        if not s_pwd: missing.append("VRS_SMTP_PASSWORD")
        return SmtpDiagnosticResult(
            success=False,
            step_reached="Configuration Verification",
            error_category="CONFIG_MISSING",
            error_message=f"Missing required SMTP configuration: {', '.join(missing)}",
            recommendation="Configure your SMTP credentials in settings or environment variables.",
        )

    # Step 2: TCP Connection & DNS Resolution
    server = None
    context = ssl.create_default_context()
    try:
        if s_sec == "ssl":
            server = smtplib.SMTP_SSL(s_host, s_port, context=context, timeout=timeout)
        else:
            server = smtplib.SMTP(s_host, s_port, timeout=timeout)
            server.ehlo()
    except (socket.gaierror, socket.herror) as dns_err:
        return SmtpDiagnosticResult(
            success=False,
            step_reached="DNS Resolution",
            error_category="DNS_LOOKUP_FAILED",
            error_message=f"Cannot resolve host '{s_host}': {dns_err}",
            recommendation="Check the SMTP host spelling (e.g. smtp.gmail.com) and your internet connection.",
        )
    except (socket.timeout, TimeoutError) as t_err:
        return SmtpDiagnosticResult(
            success=False,
            step_reached="TCP Connection",
            error_category="CONNECTION_TIMEOUT",
            error_message=f"Connection timed out reaching {s_host}:{s_port} ({t_err})",
            recommendation=f"Port {s_port} may be blocked by your firewall, router, or ISP.",
        )
    except (ConnectionRefusedError, ConnectionResetError, OSError) as conn_err:
        return SmtpDiagnosticResult(
            success=False,
            step_reached="TCP Connection",
            error_category="CONNECTION_REFUSED",
            error_message=f"Connection refused at {s_host}:{s_port}: {conn_err}",
            recommendation=f"Ensure port {s_port} matches your provider's SMTP specification.",
        )

    # Step 3: TLS Handshake (for STARTTLS)
    if s_sec != "ssl":
        try:
            server.starttls(context=context)
            server.ehlo()
        except ssl.SSLError as ssl_err:
            if server:
                try: server.quit()
                except Exception: pass
            return SmtpDiagnosticResult(
                success=False,
                step_reached="TLS Handshake",
                error_category="TLS_HANDSHAKE_FAILED",
                error_message=f"STARTTLS negotiation failed: {ssl_err}",
                recommendation="Verify your SSL/TLS certificates and security settings (use STARTTLS for port 587).",
            )
        except Exception as starttls_err:
            if server:
                try: server.quit()
                except Exception: pass
            return SmtpDiagnosticResult(
                success=False,
                step_reached="TLS Handshake",
                error_category="STARTTLS_ERROR",
                error_message=f"STARTTLS command rejected: {starttls_err}",
                recommendation="Server may not support STARTTLS on this port. Try port 465 with SSL.",
            )

    # Step 4: Authentication
    try:
        server.login(s_user, s_pwd)
    except smtplib.SMTPAuthenticationError as auth_err:
        if server:
            try: server.quit()
            except Exception: pass
        code = getattr(auth_err, "smtp_code", 535)
        err_text = str(auth_err)
        recommendation = (
            "Authentication failed. Please verify username and password.\n"
            "If using Gmail:\n"
            "  1. Ensure 2-Step Verification is enabled on your Google Account.\n"
            "  2. Create a 16-character App Password at https://myaccount.google.com/apppasswords.\n"
            "  3. Use the App Password without whitespace."
        )
        return SmtpDiagnosticResult(
            success=False,
            step_reached="SMTP Authentication",
            error_category="AUTH_FAILED_535" if code == 535 or "5.7.8" in err_text else "AUTH_ERROR",
            error_message=f"SMTP authentication rejected (code {code}): {err_text}",
            recommendation=recommendation,
        )
    except Exception as auth_gen_err:
        if server:
            try: server.quit()
            except Exception: pass
        return SmtpDiagnosticResult(
            success=False,
            step_reached="SMTP Authentication",
            error_category="AUTH_EXCEPTION",
            error_message=f"Authentication exception: {auth_gen_err}",
            recommendation="Check provider authentication mechanisms.",
        )

    # Step 5: Optional Test Message
    if to_email:
        try:
            msg = create_email_message(
                to_email=to_email,
                subject=f"[{settings.email_from_name}] SMTP Test Diagnostic",
                text_content="This is an automated diagnostic test message from VideoContextEngine.",
                html_content="<p>This is an automated diagnostic test message from VideoContextEngine.</p>",
                from_email=s_from,
                from_name=settings.email_from_name,
                host=s_host,
            )
            refused = server.send_message(msg)
            if refused:
                raise smtplib.SMTPRecipientsRefused(refused)
        except smtplib.SMTPRecipientsRefused as rec_err:
            if server:
                try: server.quit()
                except Exception: pass
            return SmtpDiagnosticResult(
                success=False,
                step_reached="Message Delivery",
                error_category="RECIPIENT_REFUSED",
                error_message=f"Recipient rejected: {rec_err}",
                recommendation=f"The server rejected delivery to {to_email}.",
            )
        except Exception as send_err:
            if server:
                try: server.quit()
                except Exception: pass
            return SmtpDiagnosticResult(
                success=False,
                step_reached="Message Delivery",
                error_category="SEND_FAILED",
                error_message=f"Failed sending test email: {send_err}",
                recommendation="Check sender domain policy (SPF/DKIM) or relay permissions.",
            )

    try:
        server.quit()
    except Exception:
        pass

    return SmtpDiagnosticResult(
        success=True,
        step_reached="Complete",
        error_category=None,
        error_message=None,
        recommendation="SMTP connection, TLS handshake, and credentials verified successfully."
    )


class SmtpMailer:
    """Hardened, unified production SMTP mailer with TLS/SSL, timeout, retry, and RFC 5322 compliance."""

    def __init__(self, timeout: float = 25.0) -> None:
        self.timeout = timeout

    def _ensure_configured(self) -> Tuple[str, int, str, str, str, str]:
        host = (os.getenv("VRS_SMTP_HOST") if "VRS_SMTP_HOST" in os.environ else settings.smtp_host or "").strip()
        user = (os.getenv("VRS_SMTP_USERNAME") if "VRS_SMTP_USERNAME" in os.environ else settings.smtp_username or "").strip()
        raw_pwd = os.getenv("VRS_SMTP_PASSWORD") if "VRS_SMTP_PASSWORD" in os.environ else settings.smtp_password or ""
        pwd = raw_pwd.strip().strip("'\"")

        if not (host and user and pwd):
            raise EmailConfigurationError(
                "Email delivery isn't configured on this server. "
                "Please configure SMTP credentials in .env according to docs/AUTH_SETUP.md."
            )

        port = int(os.getenv("VRS_SMTP_PORT") if "VRS_SMTP_PORT" in os.environ else settings.smtp_port or 587)
        security = (os.getenv("VRS_SMTP_SECURITY") if "VRS_SMTP_SECURITY" in os.environ else settings.smtp_security or "starttls").lower().strip()
        from_email = (os.getenv("VRS_EMAIL_FROM") if "VRS_EMAIL_FROM" in os.environ else settings.email_from or user).strip()
        return host, port, security, user, pwd, from_email

    def _send_single_attempt(
        self,
        to_email: str,
        msg: EmailMessage,
        host: str,
        port: int,
        security: str,
        user: str,
        pwd: str,
    ) -> None:
        context = ssl.create_default_context()
        server = None
        try:
            t0 = time.perf_counter()
            if security == "ssl":
                server = smtplib.SMTP_SSL(host, port, context=context, timeout=self.timeout)
            else:
                server = smtplib.SMTP(host, port, timeout=self.timeout)
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()

            t_tls = (time.perf_counter() - t0) * 1000
            t_login_start = time.perf_counter()
            server.login(user, pwd)
            t_login = (time.perf_counter() - t_login_start) * 1000

            t_send_start = time.perf_counter()
            refused = server.send_message(msg)
            t_send = (time.perf_counter() - t_send_start) * 1000

            if refused:
                logger.warning("SMTP server refused recipients for message: %s", list(refused.keys()))
                raise smtplib.SMTPRecipientsRefused(refused)

            logger.info(
                "Successfully delivered message via SMTP to %s (tls_handshake=%.1fms, login=%.1fms, send=%.1fms)",
                mask_email(to_email),
                t_tls,
                t_login,
                t_send,
            )
        finally:
            if server:
                try:
                    server.quit()
                except Exception:
                    try:
                        server.close()
                    except Exception:
                        pass

    def _send_with_retry(self, to_email: str, msg: EmailMessage) -> None:
        host, port, security, user, pwd, _ = self._ensure_configured()

        for attempt in (1, 2):
            try:
                self._send_single_attempt(to_email, msg, host, port, security, user, pwd)
                return
            except EmailConfigurationError:
                raise
            except smtplib.SMTPAuthenticationError as auth_exc:
                code = getattr(auth_exc, "smtp_code", 535)
                logger.error("SMTP Authentication Error (%s): Bad credentials for user %s", code, mask_email(user))
                raise EmailDeliveryError(
                    "SMTP authentication failed (535): Bad credentials. If using Gmail, you must enable 2-Step Verification and use a 16-character Google App Password without spaces.",
                    category=SmtpErrorCode.AUTH,
                    smtp_code=code,
                ) from auth_exc
            except smtplib.SMTPRecipientsRefused as rec_exc:
                logger.error("SMTP Recipient Refused for %s: %s", mask_email(to_email), rec_exc)
                raise EmailDeliveryError(
                    f"Recipient was refused by the mail server: {mask_email(to_email)}",
                    category=SmtpErrorCode.RECIPIENT,
                    smtp_code=550,
                    refused_recipients=list(rec_exc.recipients.keys()) if hasattr(rec_exc, "recipients") else [to_email],
                ) from rec_exc
            except (smtplib.SMTPServerDisconnected, socket.timeout, TimeoutError) as transient_exc:
                if attempt == 1:
                    logger.warning("Transient SMTP error on attempt 1 (%s); retrying in 1.0s...", type(transient_exc).__name__)
                    time.sleep(1.0)
                    continue
                logger.error("SMTP transient delivery failed after retry: %s", type(transient_exc).__name__)
                cat = SmtpErrorCode.CONNECT if isinstance(transient_exc, (socket.timeout, TimeoutError)) else SmtpErrorCode.TRANSIENT
                raise EmailDeliveryError(
                    f"SMTP delivery failed: {type(transient_exc).__name__}",
                    category=cat,
                ) from transient_exc
            except smtplib.SMTPResponseException as resp_exc:
                code = getattr(resp_exc, "smtp_code", -1)
                if attempt == 1 and (code in (421, 450, 451, 452) or (400 <= code < 500)):
                    logger.warning("Transient SMTP response code %s on attempt 1; retrying in 1.0s...", code)
                    time.sleep(1.0)
                    continue
                cat, code_val, _ = classify_smtp_exception(resp_exc)
                logger.error("SMTP response error: code=%s, category=%s", code, cat)
                raise EmailDeliveryError(
                    f"SMTP server rejected message with code {code}",
                    category=cat,
                    smtp_code=code,
                ) from resp_exc
            except Exception as exc:
                cat, code_val, _ = classify_smtp_exception(exc)
                logger.error("Failed to deliver email: %s (category: %s)", type(exc).__name__, cat)
                raise EmailDeliveryError(
                    f"Email delivery error: {type(exc).__name__}",
                    category=cat,
                    smtp_code=code_val,
                ) from exc

    def send_verification_email(self, to_email: str, code: str, ttl_minutes: int) -> None:
        """Send verification code email with expiry and RFC 5322 compliance."""
        host, _, _, user, _, from_email = self._ensure_configured()
        from_name = settings.email_from_name or "VideoContextEngine"
        subject = f"{from_name} Verification Code"

        text_body = (
            f"Hello,\n\n"
            f"Your verification code for {from_name} is:\n\n"
            f"    {code}\n\n"
            f"This code will expire in {ttl_minutes} minutes.\n"
            f"If you did not request this verification, please ignore this email.\n"
        )
        html_body = f"""<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0A0A1F; color: #E2E8F0; padding: 24px;">
    <div style="max-width: 480px; margin: 0 auto; background: #13132B; border: 1px solid #23234A; border-radius: 12px; padding: 32px;">
        <h2 style="color: #F8FAFC; margin-top: 0;">Verify Your Email</h2>
        <p style="color: #94A3B8; font-size: 0.95rem;">Please use the verification code below to complete your registration:</p>
        <div style="text-align: center; margin: 28px 0;">
            <span style="display: inline-block; font-size: 2rem; font-weight: 700; letter-spacing: 6px; padding: 12px 24px; background: #1C1C3D; border: 1px solid #7C5CFF; border-radius: 8px; color: #A78BFA; font-family: monospace;">{code}</span>
        </div>
        <p style="color: #94A3B8; font-size: 0.85rem;">This code is valid for <strong>{ttl_minutes} minutes</strong>.</p>
        <hr style="border: none; border-top: 1px solid #23234A; margin: 24px 0;" />
        <p style="color: #64748B; font-size: 0.8rem; margin-bottom: 0;">If you didn't request this code, you can safely ignore this email.</p>
    </div>
</body>
</html>"""
        msg = create_email_message(
            to_email=to_email,
            subject=subject,
            text_content=text_body,
            html_content=html_body,
            from_email=from_email,
            from_name=from_name,
            host=host,
        )
        self._send_with_retry(to_email, msg)

    def send_security_notice(self, to_email: str) -> None:
        """Send security notification for duplicate registration attempt with RFC 5322 compliance."""
        host, _, _, user, _, from_email = self._ensure_configured()
        from_name = settings.email_from_name or "VideoContextEngine"
        subject = f"Security Notice: {from_name} Account"

        text_body = (
            f"Hello,\n\n"
            f"Someone recently attempted to create an account with this email address on {from_name}.\n\n"
            f"If this was you, please return to the application and log in with your existing password, or use Continue with Google.\n"
            f"If you did not make this request, you can safely ignore this email — your account remains secure.\n"
        )
        html_body = f"""<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0A0A1F; color: #E2E8F0; padding: 24px;">
    <div style="max-width: 480px; margin: 0 auto; background: #13132B; border: 1px solid #23234A; border-radius: 12px; padding: 32px;">
        <h2 style="color: #F8FAFC; margin-top: 0;">Security Notice</h2>
        <p style="color: #94A3B8; font-size: 0.95rem;">Someone recently attempted to register an account with your email address.</p>
        <p style="color: #CBD5E1; font-size: 0.9rem;">If this was you, please return to the application and <strong>Log In</strong> with your existing credentials, or sign in using <strong>Continue with Google</strong>.</p>
        <hr style="border: none; border-top: 1px solid #23234A; margin: 24px 0;" />
        <p style="color: #64748B; font-size: 0.8rem; margin-bottom: 0;">If you did not attempt this registration, no action is needed — your account is safe.</p>
    </div>
</body>
</html>"""
        msg = create_email_message(
            to_email=to_email,
            subject=subject,
            text_content=text_body,
            html_content=html_body,
            from_email=from_email,
            from_name=from_name,
            host=host,
        )
        self._send_with_retry(to_email, msg)


# Default mailer instance & test override hook
_mailer_instance: Optional[Mailer] = None


def get_mailer() -> Mailer:
    """Retrieve current mailer instance (SmtpMailer by default, or test override)."""
    global _mailer_instance
    if _mailer_instance is None:
        return SmtpMailer()
    return _mailer_instance


def set_mailer(mailer: Optional[Mailer]) -> None:
    """Set or reset the mailer implementation (used by test fixtures)."""
    global _mailer_instance
    _mailer_instance = mailer
