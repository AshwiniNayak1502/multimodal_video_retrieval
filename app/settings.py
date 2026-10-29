"""Centralized application settings and environment variable configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# D1 fix: load .env into environment before evaluating os.getenv defaults
load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    """Environment-driven application configuration."""

    # Project root
    project_root: Path = Path(__file__).resolve().parent.parent

    # Data directory
    data_dir: Path = Path(
        os.getenv("VRS_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
    ).resolve()

    # App SQLite database
    app_db_path: Path = Path(
        os.getenv("VRS_APP_DB_PATH", str(data_dir / "app.db"))
    ).resolve()

    # Playback proxy directory
    playback_dir: Path = Path(
        os.getenv("VRS_PLAYBACK_DIR", str(data_dir / "playback"))
    ).resolve()

    # Video upload and storage
    max_upload_mb: int = int(os.getenv("VRS_MAX_UPLOAD_MB", "2048"))
    supported_extensions: tuple[str, ...] = (".mp4", ".mkv")

    # Ingestion & Whisper
    whisper_model: str = os.getenv("VRS_WHISPER_MODEL", "tiny")

    # Authentication & Session Security
    session_ttl_minutes: int = int(os.getenv("VRS_SESSION_TTL_MINUTES", "720"))  # 12 hours
    lockout_attempts: int = int(os.getenv("VRS_LOCKOUT_ATTEMPTS", "5"))
    lockout_duration_minutes: int = int(os.getenv("VRS_LOCKOUT_DURATION_MINUTES", "5"))
    enable_google: bool = os.getenv("VRS_ENABLE_GOOGLE", "false").lower() in ("true", "1", "yes")

    # Email Verification & Rate Limiting
    auth_secret: str = os.getenv("VRS_AUTH_SECRET", "")
    require_email_verification: bool = os.getenv(
        "VRS_REQUIRE_EMAIL_VERIFICATION", "true"
    ).lower() in ("true", "1", "yes")
    verification_code_length: int = int(os.getenv("VRS_VERIFICATION_CODE_LENGTH", "6"))
    verification_ttl_minutes: int = int(os.getenv("VRS_VERIFICATION_TTL_MINUTES", "10"))
    verification_max_attempts: int = int(os.getenv("VRS_VERIFICATION_MAX_ATTEMPTS", "5"))
    resend_cooldown_seconds: int = int(os.getenv("VRS_RESEND_COOLDOWN_SECONDS", "60"))
    resend_max_per_hour: int = int(os.getenv("VRS_RESEND_MAX_PER_HOUR", "5"))
    email_max_per_hour: int = int(os.getenv("VRS_EMAIL_MAX_PER_HOUR", "50"))

    # SMTP Mailer Settings
    smtp_host: str = os.getenv("VRS_SMTP_HOST", "")
    smtp_port: int = int(os.getenv("VRS_SMTP_PORT", "587"))
    smtp_security: str = os.getenv("VRS_SMTP_SECURITY", "starttls").lower()
    smtp_username: str = os.getenv("VRS_SMTP_USERNAME", "")
    smtp_password: str = os.getenv("VRS_SMTP_PASSWORD", "")
    email_from: str = os.getenv("VRS_EMAIL_FROM", "")
    email_from_name: str = os.getenv("VRS_EMAIL_FROM_NAME", "VideoContextEngine")

    # Search & Retrieval
    query_rewrite: str = os.getenv("VRS_QUERY_REWRITE", "light")  # "off" or "light"
    default_top_k: int = int(os.getenv("VRS_DEFAULT_TOP_K", "5"))
    max_top_k: int = int(os.getenv("VRS_MAX_TOP_K", "10"))
    min_top_k: int = int(os.getenv("VRS_MIN_TOP_K", "3"))

    def ensure_directories(self) -> None:
        """Create required runtime storage directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.app_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.playback_dir.mkdir(parents=True, exist_ok=True)


# Default singleton instance
settings = Settings()
