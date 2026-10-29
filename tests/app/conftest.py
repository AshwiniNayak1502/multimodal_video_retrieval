"""Pytest fixtures for VideoContextEngine app test suite."""

from __future__ import annotations

import smtplib
from typing import Any, Dict, List, Optional
import pytest

from app.services.email_service import Mailer, set_mailer


class FakeMailer:
    """In-memory test mailer that records all outbound messages without networking."""

    def __init__(self) -> None:
        self.sent_emails: List[Dict[str, Any]] = []
        self.should_fail: bool = False
        self.failure_exception: Optional[Exception] = None

    def send_verification_email(self, to_email: str, code: str, ttl_minutes: int) -> None:
        if self.should_fail:
            raise self.failure_exception or smtplib.SMTPException("Simulated SMTP delivery failure")
        self.sent_emails.append({
            "to": to_email,
            "code": code,
            "ttl_minutes": ttl_minutes,
            "kind": "verification",
        })

    def send_security_notice(self, to_email: str) -> None:
        if self.should_fail:
            raise self.failure_exception or smtplib.SMTPException("Simulated SMTP delivery failure")
        self.sent_emails.append({
            "to": to_email,
            "kind": "security_notice",
        })

    def clear(self) -> None:
        self.sent_emails.clear()
        self.should_fail = False
        self.failure_exception = None


@pytest.fixture(autouse=True)
def prevent_real_email_and_socket(monkeypatch):
    """Hard rule: Disallow real SMTP sockets in tests and route through in-memory fake."""
    fake = FakeMailer()
    set_mailer(fake)

    # Monkeypatch smtplib as defence in depth against accidental socket opens
    def _blocked_smtp(*args, **kwargs):
        raise RuntimeError("Real SMTP connection attempted during test execution!")

    monkeypatch.setattr(smtplib, "SMTP", _blocked_smtp)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _blocked_smtp)

    # Provide a default valid VRS_AUTH_SECRET for tests if not set
    monkeypatch.setenv("VRS_AUTH_SECRET", "super-secret-key-that-is-at-least-32-chars-long-for-testing!!")

    yield fake

    set_mailer(None)


@pytest.fixture
def fake_mailer(prevent_real_email_and_socket) -> FakeMailer:
    """Convenience fixture to inspect and configure the fake mailer in individual tests."""
    return prevent_real_email_and_socket

