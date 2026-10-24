"""Logging configuration for Module 1."""

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
    """Configure and return a standardized logger for Module 1.

    Args:
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional path to append log entries.
        logger_name: Optional specific logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name or "module1")
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called multiple times
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
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate log messages
    logger.propagate = False

    return logger
