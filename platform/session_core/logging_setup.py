"""Session readers share LOCITIZE logging without import-time file writes."""
import logging


def get_logger(name):
    return logging.getLogger(f"locitize.sessions.{name}")
