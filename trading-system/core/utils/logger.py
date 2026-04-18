"""Centralised logger — rotating file handlers for system, trades, and errors logs."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ANSI colour codes for console output
_COLOURS = {
    logging.DEBUG:    "\033[36m",   # cyan
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Root log directory — resolved at import time; honours LOG_DIR env var
_LOG_DIR = Path(os.environ.get("LOG_DIR", str(Path(__file__).resolve().parents[3] / "logs")))

# Max size per log file (bytes) and number of backups
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5

# Track which loggers have already been configured to avoid duplicate handlers
_configured: set[str] = set()


class _ColouredFormatter(logging.Formatter):
    """Formatter that injects ANSI colour codes based on log level."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelno, "")
        message = super().format(record)
        return f"{colour}{message}{_RESET}"


def _build_rotating_handler(log_file: Path, level: int) -> RotatingFileHandler:
    """Create a rotating file handler for *log_file* at *level*."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(log_file),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a configured Logger for *name*.

    First call for a given name attaches three handlers:
    - Console (stderr) with colour coding, at DEBUG level.
    - ``logs/system.log``  — all levels >= INFO.
    - ``logs/errors.log``  — all levels >= ERROR.

    Loggers whose name starts with ``"trade"`` also get a
    ``logs/trades.log`` handler so that trade lifecycle events
    are captured in a separate file.

    Subsequent calls for the same name return the existing logger.

    Args:
        name:  Module name, typically ``__name__``.
        level: Optional override for the logger's effective level.
               Defaults to DEBUG so that handlers can filter independently.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if name in _configured:
        return logger

    _configured.add(name)
    logger.setLevel(level if level is not None else logging.DEBUG)
    logger.propagate = False  # avoid double-logging to root

    # ── Console handler ──────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG)
    console.setFormatter(
        _ColouredFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    )
    logger.addHandler(console)

    # ── system.log ───────────────────────────────────────────────────────────
    logger.addHandler(_build_rotating_handler(_LOG_DIR / "system.log", logging.INFO))

    # ── errors.log ───────────────────────────────────────────────────────────
    logger.addHandler(_build_rotating_handler(_LOG_DIR / "errors.log", logging.ERROR))

    # ── trades.log (trade-specific loggers) ──────────────────────────────────
    if name.startswith("trade") or "trade" in name.lower():
        logger.addHandler(
            _build_rotating_handler(_LOG_DIR / "trades.log", logging.INFO)
        )

    return logger
