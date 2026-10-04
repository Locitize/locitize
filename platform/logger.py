"""Rotating per-subsystem logging for the LOCITIZE platform.

This module is the single place that configures Python's logging for the whole
platform (Architecture section 9.1). Every other module obtains its logger with
get_logger(name) and never configures handlers itself, so there is no global
mutable logging state scattered across the codebase.

Design points:
- One rotating file handler per subsystem channel (launcher, assistant, speech,
  llm, tts, benchmark, errors), writing under logs/ inside the platform dir.
- A shared root handler routes WARNING+ from every logger into errors.log so
  failures are centralised.
- ASCII-only, UTC timestamps. Secrets are never logged (the caller is
  responsible for not passing secret values; config redaction lives in config.py).
- configure_logging() is idempotent: calling it twice does not double up
  handlers. It is only ever called from launcher.main(), never at import time,
  so importing this module has no side effects (supports AC2).
"""

from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Marker attribute we stamp on handlers we create, so a second
# configure_logging() call can detect and skip re-adding them.
_LOCITIZE_HANDLER_FLAG = "_locitize_handler"

# Environment override for the log directory (Architecture M5.12). This is the
# single seam that keeps test and benchmark log output OUT of the tracked
# production logs/ directory: the test suite's conftest sets LOCITIZE_LOG_DIR to a
# pytest tmp_path so a live user's real logs/ is never polluted by a test run
# (the carried log-pollution defect the M3+4 owner session found). It sits above
# settings in precedence (env beats file beats the code default), consistent with
# the rest of the LOCITIZE_* override chain (config.py section 7).
_LOG_DIR_ENV = "LOCITIZE_LOG_DIR"
# M15.7: the product-name spelling, checked FIRST wherever _LOG_DIR_ENV is
# read. The old name keeps working; see config._apply_env_overrides for the
# same policy and its reasoning.
_LOG_DIR_ENV_NEW = "LOCITIZE_LOG_DIR"


def resolve_log_dir(settings: Any) -> Path:
    """Return the directory every log file is written under.

    Precedence (highest first): the LOCITIZE_LOG_DIR environment variable, then
    <data root>/logs. Keeping this in ONE function means the launcher's service
    logs, the logging handlers, and the benchmark runner all agree on the same
    directory, so a test that sets LOCITIZE_LOG_DIR redirects every one of them
    in a single stroke (Architecture M5.12).

    The data root, never the install directory (DEC-M14-9, defect NEW-QA-M14-8).
    A log records what happened on THIS user's machine: it is the only evidence
    available when they ask for help about something that already happened, it
    must survive the reinstall that breakage usually prompts, and the install
    directory may not be writable at all. settings.data_dir is read with no
    base_dir fallback on purpose: Settings.__post_init__ already guarantees the
    field is set (it equals base_dir when no separate root was resolved), so a
    fallback here would only be a second, silent way for logs to land in the
    install tree.
    """
    override = os.environ.get(_LOG_DIR_ENV_NEW) or os.environ.get(_LOG_DIR_ENV)
    if override:
        return Path(override)
    return Path(settings.data_dir) / "logs"


class _UtcFormatter(logging.Formatter):
    """Formatter that renders timestamps in UTC for reproducible, sortable logs."""

    # Using a converter (not localtime) keeps log timestamps machine-independent.
    converter = time.gmtime


def _make_formatter() -> logging.Formatter:
    # Fixed ASCII format shared by every handler (Architecture section 9.1).
    return _UtcFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")


def configure_logging(settings: Any) -> None:
    """Configure per-subsystem rotating loggers from a Settings object.

    `settings` is duck-typed to avoid importing config here (config imports
    nothing from logger, so accepting the object keeps the dependency one-way).
    It must expose `settings.logging` (channels, levels, max_bytes,
    backup_count) and `settings.data_dir` (the user data root, DEC-M14-9).
    """
    log_cfg = settings.logging
    # Resolve via the shared helper so an LOCITIZE_LOG_DIR override (used by the
    # test suite and benchmark scripts) redirects the handlers away from the
    # tracked production logs/ directory (Architecture M5.12).
    logs_dir = resolve_log_dir(settings)
    # Create the log directory lazily here (inside main()'s bootstrap), never at
    # import time. exist_ok keeps re-runs safe.
    logs_dir.mkdir(parents=True, exist_ok=True)

    console_level = _level_value(log_cfg.level_console, logging.INFO)
    file_level = _level_value(log_cfg.level_file, logging.DEBUG)
    formatter = _make_formatter()

    # Per-channel file handlers. Each channel logger writes only to its own file
    # and does not propagate to the root (except via the shared errors handler
    # attached at root below), so channel logs stay separated.
    for channel in log_cfg.channels:
        logger = logging.getLogger(channel)
        logger.setLevel(logging.DEBUG)
        if not _has_locitize_handler(logger, kind="channel-file"):
            handler = RotatingFileHandler(
                logs_dir / f"{channel}.log",
                maxBytes=log_cfg.max_bytes,
                backupCount=log_cfg.backup_count,
                encoding="utf-8",
            )
            handler.setLevel(file_level)
            handler.setFormatter(formatter)
            _tag(handler, "channel-file")
            logger.addHandler(handler)
        # Keep propagation on so the root errors handler can capture WARNING+.
        logger.propagate = True

    # Shared errors handler at the root: captures WARNING+ from every logger so a
    # single errors.log gives a centralised failure view (Architecture 9.1).
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not _has_locitize_handler(root, kind="errors-file"):
        errors_handler = RotatingFileHandler(
            logs_dir / "errors.log",
            maxBytes=log_cfg.max_bytes,
            backupCount=log_cfg.backup_count,
            encoding="utf-8",
        )
        errors_handler.setLevel(logging.WARNING)
        errors_handler.setFormatter(formatter)
        _tag(errors_handler, "errors-file")
        root.addHandler(errors_handler)

    # Console handler for the launcher's own INFO output.
    if not _has_locitize_handler(root, kind="console"):
        console = logging.StreamHandler()
        console.setLevel(console_level)
        console.setFormatter(formatter)
        _tag(console, "console")
        # Only surface the launcher channel and warnings on the console to avoid
        # spamming stdout with DEBUG from every subsystem.
        console.addFilter(_ConsoleFilter())
        root.addHandler(console)


class _ConsoleFilter(logging.Filter):
    """Let through launcher-channel records and any WARNING+ from any channel.

    This keeps the console readable (launcher narration plus real problems)
    without dumping every subsystem's DEBUG line to the terminal.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "launcher" or record.levelno >= logging.WARNING


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a subsystem channel.

    If configure_logging() has not run yet, this still returns a valid logger;
    it simply has no LOCITIZE handlers until configuration happens. Callers must not
    rely on handler side effects at import time.
    """
    return logging.getLogger(name)


def _level_value(name: str, default: int) -> int:
    """Translate a level name (e.g. "INFO") to its numeric value, safely."""
    value = logging.getLevelName(str(name).upper())
    # getLevelName returns a string like "Level XYZ" for unknown names; guard it.
    return value if isinstance(value, int) else default


def _tag(handler: logging.Handler, kind: str) -> None:
    setattr(handler, _LOCITIZE_HANDLER_FLAG, kind)


def _has_locitize_handler(logger: logging.Logger, kind: str) -> bool:
    return any(getattr(h, _LOCITIZE_HANDLER_FLAG, None) == kind for h in logger.handlers)
