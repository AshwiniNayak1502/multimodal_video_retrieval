"""Logging configuration for Module 2."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a standardized logger for Module 2.

    Args:
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional path to append log entries.
        logger_name: Optional specific logger name (default: "module2").

    Returns:
        Configured logger instance.
    """
    name = logger_name or "module2"
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging is invoked multiple times
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a child logger under the module2 namespace."""
    if not name.startswith("module2"):
        name = f"module2.{name}"
    return logging.getLogger(name)
