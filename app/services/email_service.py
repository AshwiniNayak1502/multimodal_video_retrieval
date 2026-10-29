"""Email delivery service using standard library smtplib with SMTP/STARTTLS/SSL support."""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Protocol

from ..settings import settings

logger = logging.getLogger("app.email")


class EmailConfigurationError(RuntimeError):
    """Raised when SMTP settings are not configured."""
    pass


class EmailDeliveryError(RuntimeError):
    """Raised when email delivery fails technically."""
    pass


class Mailer(Protocol):
    """Protocol for mail delivery backends."""

    def send_verification_email(self, to_email: str, code: str, ttl_minutes: int) -> None:
        """Send verification code email with expiry."""
        ...

    def send_security_notice(self, to_email: str) -> None:
        """Send security notification for duplicate registration attempt."""
        ...


class SmtpMailer:
    """Production SMTP mailer with TLS/SSL and timeout support."""

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
        except smtplib.SMTPResponseException as exc:
            # Technical cause logged server-side only: class and SMTP status code
            logger.error(
                "SMTP response error during send: %s (code: %s)",
                type(exc).__name__,
                getattr(exc, "smtp_code", "unknown"),
            )
            raise EmailDeliveryError(f"SMTP error code {getattr(exc, 'smtp_code', 'unknown')}") from exc
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
