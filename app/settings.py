"""Centralized application settings and environment variable configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
