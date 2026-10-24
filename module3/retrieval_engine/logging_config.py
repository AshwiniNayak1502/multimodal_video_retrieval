"""Logging configuration for Module 3 retrieval engine."""

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
    """Configure and return a standardized logger for Module 3.

    Args:
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional path to append log entries.
        logger_name: Optional specific logger name (default: "module3").

    Returns:
        Configured logger instance.
    """
    name = logger_name or "module3"
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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
    """Retrieve a child logger under the module3 namespace."""
    if not name.startswith("module3"):
        name = f"module3.{name}"
    return logging.getLogger(name)
