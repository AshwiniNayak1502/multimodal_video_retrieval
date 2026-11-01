"""Email delivery service using standard library smtplib with SMTP/STARTTLS/SSL support and diagnostics."""

from __future__ import annotations

import logging
import os
import socket
import smtplib
import ssl
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Protocol, Tuple

from ..settings import settings

logger = logging.getLogger("app.email")


class EmailConfigurationError(RuntimeError):
    """Raised when SMTP settings are not configured."""
    pass


class EmailDeliveryError(RuntimeError):
    """Raised when email delivery fails technically."""
    pass


@dataclass
class SmtpDiagnosticResult:
    """Structured result returned by SMTP connectivity diagnosis."""
    success: bool
    step_reached: str
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    recommendation: Optional[str] = None


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
    timeout: float = 10.0,
) -> SmtpDiagnosticResult:
    """Perform step-by-step diagnostic test on SMTP connection, TLS handshake, and credentials."""
    s_host = host or os.getenv("VRS_SMTP_HOST") or settings.smtp_host
    s_port = int(port or os.getenv("VRS_SMTP_PORT") or settings.smtp_port)
    s_sec = (security or os.getenv("VRS_SMTP_SECURITY") or settings.smtp_security or "starttls").lower()
    s_user = username or os.getenv("VRS_SMTP_USERNAME") or settings.smtp_username
    s_pwd = password or os.getenv("VRS_SMTP_PASSWORD") or settings.smtp_password
    s_from = from_email or os.getenv("VRS_EMAIL_FROM") or settings.email_from or s_user

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
            "  3. Use the App Password (not your standard Google account password)."
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
            msg = MIMEMultipart("alternative")
            msg["From"] = f'"{settings.email_from_name}" <{s_from}>'
            msg["To"] = to_email
            msg["Subject"] = f"[{settings.email_from_name}] SMTP Test Diagnostic"
            msg.attach(MIMEText("This is an automated diagnostic test message from VideoContextEngine.", "plain"))
            server.sendmail(s_from, [to_email], msg.as_string())
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
    """Production SMTP mailer with TLS/SSL, timeout, and diagnostic error reporting."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def _ensure_configured(self) -> None:
        host = os.getenv("VRS_SMTP_HOST") if "VRS_SMTP_HOST" in os.environ else settings.smtp_host
        user = os.getenv("VRS_SMTP_USERNAME") if "VRS_SMTP_USERNAME" in os.environ else settings.smtp_username
        pwd = os.getenv("VRS_SMTP_PASSWORD") if "VRS_SMTP_PASSWORD" in os.environ else settings.smtp_password
        if not (host and user and pwd):
            raise EmailConfigurationError(
                "Email delivery isn't configured on this server. "
                "Please configure SMTP credentials in .env according to docs/AUTH_SETUP.md."
            )

    def _create_message(self, to_email: str, subject: str, text_content: str, html_content: str) -> MIMEMultipart:
        msg = MIMEMultipart("alternative")
        from_header = f'"{settings.email_from_name}" <{settings.email_from or settings.smtp_username}>'
        msg["From"] = from_header
        msg["To"] = to_email
        msg["Subject"] = subject

        part_text = MIMEText(text_content, "plain", "utf-8")
        part_html = MIMEText(html_content, "html", "utf-8")
        msg.attach(part_text)
        msg.attach(part_html)
        return msg

    def _send(self, to_email: str, msg: MIMEMultipart) -> None:
        self._ensure_configured()
        context = ssl.create_default_context()
        security = (settings.smtp_security or "starttls").lower()

        server = None
        try:
            if security == "ssl":
                server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=context, timeout=self.timeout)
            else:
                server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=self.timeout)
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()

            server.login(settings.smtp_username, settings.smtp_password)
            sender = settings.email_from or settings.smtp_username
            server.sendmail(sender, [to_email], msg.as_string())
        except EmailConfigurationError:
            raise
        except smtplib.SMTPAuthenticationError as auth_exc:
            logger.error("SMTP Authentication Error (535): Invalid username or password: %s", auth_exc)
            raise EmailDeliveryError(
                "SMTP authentication failed (535): Bad credentials. If using Gmail, you must enable 2-Step Verification and use a 16-character Google App Password."
            ) from auth_exc
        except smtplib.SMTPResponseException as exc:
            code = getattr(exc, "smtp_code", "unknown")
            logger.error("SMTP response error: %s (code: %s)", type(exc).__name__, code)
            raise EmailDeliveryError(f"SMTP error code {code}") from exc
        except (socket.timeout, TimeoutError) as t_exc:
            logger.error("SMTP connection timeout: %s", t_exc)
            raise EmailDeliveryError("SMTP connection timed out. Check network connection and port settings.") from t_exc
        except (ConnectionRefusedError, ConnectionResetError, OSError) as conn_exc:
            logger.error("SMTP connection refused: %s", conn_exc)
            raise EmailDeliveryError("Could not connect to SMTP server. Verify server host and port.") from conn_exc
        except Exception as exc:
            logger.error("Failed to deliver email: %s", type(exc).__name__)
            raise EmailDeliveryError(f"Email delivery error: {type(exc).__name__}") from exc
        finally:
            if server:
                try:
                    server.quit()
                except Exception:
                    pass

    def send_verification_email(self, to_email: str, code: str, ttl_minutes: int) -> None:
        subject = f"{settings.email_from_name} Verification Code"
        text_body = (
            f"Hello,\n\n"
            f"Your verification code for {settings.email_from_name} is:\n\n"
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
        msg = self._create_message(to_email, subject, text_body, html_body)
        self._send(to_email, msg)

    def send_security_notice(self, to_email: str) -> None:
        subject = f"Security Notice: {settings.email_from_name} Account"
        text_body = (
            f"Hello,\n\n"
            f"Someone recently attempted to create an account with this email address on {settings.email_from_name}.\n\n"
            f"If this was you, please log in with your existing password or use Continue with Google.\n"
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
        msg = self._create_message(to_email, subject, text_body, html_body)
        self._send(to_email, msg)


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
