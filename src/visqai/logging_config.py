"""
logging_config.py
==================
Central loguru configuration for visqai: a colorized console sink plus a
rotating file sink under the repo's logs/ directory.

Usage
-----
CLI entrypoints call configure_logging() once, near the top of main():

    from visqai.logging_config import configure_logging
    configure_logging()

Everywhere else, just import loguru's logger directly and use it -- loguru's
logger is a global singleton, unlike stdlib logging's per-module
getLogger(__name__) pattern:

    from loguru import logger
    logger.info("...")

Existing call sites that still use Python's stdlib `logging.getLogger(...)`
(across visqai.eval/*, and third-party libraries like optuna/torch) do NOT
need to be rewritten: configure_logging() installs an InterceptHandler on
the stdlib root logger that forwards every stdlib log record into loguru's
sinks, so "all logging" ends up in the same place without touching every
call site.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from loguru import logger

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
)

_configured = False


def _default_log_dir() -> Path:
    """logs/ at the repo root (src/visqai/logging_config.py -> repo root is
    two parents up), overridable via the VISQAI_LOG_DIR env var."""
    if os.environ.get("VISQAI_LOG_DIR"):
        return Path(os.environ["VISQAI_LOG_DIR"])
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "logs"


class InterceptHandler(logging.Handler):
    """Redirects stdlib `logging` records into loguru. Installed on the
    stdlib root logger by configure_logging() so every existing
    logging.getLogger(...) call site (visqai.eval/*, optuna, torch, ...) is
    captured by loguru's sinks without modification."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(
    level: str = "INFO",
    file_level: str = "DEBUG",
    log_dir: Path | str | None = None,
    rotation: str = "10 MB",
    retention: str = "14 days",
    name: str = "visqai",
    force: bool = False,
):
    """Configure loguru's console + rotating-file sinks and route stdlib
    `logging` into them. Idempotent -- safe to call from every CLI entrypoint;
    only the first call takes effect unless force=True.

    Parameters
    ----------
    level      : minimum level for the console sink.
    file_level : minimum level for the file sink (more verbose by default).
    log_dir    : directory for log files. Defaults to <repo_root>/logs, or
                 $VISQAI_LOG_DIR if set.
    rotation   : loguru rotation policy (size or time based).
    retention  : how long to keep rotated log files.
    name       : log filename prefix (one file per day: "<name>_YYYY-MM-DD.log").
    force      : reconfigure even if configure_logging() was already called.

    Returns the configured loguru logger.
    """
    global _configured
    if _configured and not force:
        return logger

    target_dir = Path(log_dir) if log_dir is not None else _default_log_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=level, format=_CONSOLE_FORMAT, colorize=True)
    logger.add(
        target_dir / f"{name}_{{time:YYYY-MM-DD}}.log",
        level=file_level,
        format=_FILE_FORMAT,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,  # process/thread-safe -- safe under Optuna's parallel trials
        backtrace=True,
        diagnose=False,  # don't leak variable values into log files by default
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    _configured = True
    logger.debug(f"Logging configured: console>={level}, file>={file_level} at {target_dir}")
    return logger


def get_logger(name: str | None = None):
    """Convenience accessor for stdlib-getLogger-style call sites
    (`get_logger(__name__)`). Configures logging with defaults if it hasn't
    been configured yet, and returns loguru's logger -- loguru has no
    per-module logger instances; it auto-detects the calling module for the
    `{name}` format field on every call. `name` is bound as an extra
    `module` field (visible with a format string that references
    `{extra[module]}`; the default format already shows the real calling
    module via `{name}`, so this is mainly for custom filtering/formatting)."""
    if not _configured:
        configure_logging()
    return logger.bind(module=name) if name else logger
