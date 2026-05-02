"""Logging configuration.

Centralizes logging setup so every script logs the same way: timestamps,
module names, levels, and optional file output. Call setup_logging() once
at the start of each script.

Example:
    from probe_extraction.utils.logging import setup_logging
    setup_logging(level="INFO", log_dir="logs", log_name="01_extract")
    logger = logging.getLogger(__name__)
    logger.info("Starting extraction...")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


# ============================================================================
# Format strings
# ============================================================================

# Console: short timestamp, level, module, message
CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# File: full ISO timestamp, useful for grepping logs after long runs
FILE_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s:%(lineno)d %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# Public entry point
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    log_name: str | None = None,
    log_to_file: bool = True,
) -> Path | None:
    """Configure root logger with console + optional file output.

    Idempotent: calling twice does not duplicate handlers.

    Args:
        level: Minimum level to log. One of DEBUG, INFO, WARNING, ERROR.
        log_dir: Directory for log files. If None or log_to_file is False,
            no file logging.
        log_name: Stem for log filenames. Defaults to "run". Final filename
            is "{log_name}_{timestamp}.log" so reruns don't overwrite.
        log_to_file: Master switch for file logging.

    Returns:
        Path to the log file if file logging enabled, else None.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers (idempotency)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # ----- Console handler (always on) -----
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(console)

    # ----- File handler (optional) -----
    log_path: Path | None = None
    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = log_name or "run"
        log_path = log_dir / f"{stem}_{timestamp}.log"

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
        )
        root.addHandler(file_handler)

    # ----- Tame chatty third-party loggers -----
    # transformers logs at INFO during model loading: useful but verbose.
    # Bump it down a level so our own INFO logs aren't drowned.
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    return log_path