"""
Central logging configuration for the framework.

Library code (engines, data layer, strategies, collectors) uses ordinary
module loggers::

    import logging
    logger = logging.getLogger(__name__)

Entry points (GUIs, CLI tools) call :func:`setup_logging` once at startup to
attach a console handler and a per-run log file under ``logs/``, so every
warning the engines emit (FX conversions, data-load failures, leakage
heuristics) is captured somewhere a user can find after the run.

Calling :func:`setup_logging` more than once is safe — handlers are only
attached the first time.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(app_name: str = "framework",
                  log_dir: str = "logs",
                  level: int = logging.INFO,
                  console: bool = True) -> Optional[Path]:
    """
    Configure root logging with a console handler and a per-run log file.

    Args:
        app_name: Prefix for the log file name (e.g. "backtest", "optimizer")
        log_dir: Directory for log files (created if missing)
        level: Root log level
        console: Also log to stderr (default True)

    Returns:
        Path of the log file for this run, or None if only console logging
        could be configured (e.g. the log directory is not writable).
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return None
    _CONFIGURED = True

    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if console and not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)

    log_path: Optional[Path] = None
    try:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = directory / f"{app_name}_{stamp}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        logging.getLogger(__name__).info("Logging to %s", log_path)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not create log file in %r; console logging only.", log_dir)

    return log_path
