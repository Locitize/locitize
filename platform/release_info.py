"""Product identity and deliberately minimal, user-reviewable diagnostics."""
from __future__ import annotations

import importlib.metadata
import sys

VERSION = "1.0.4"


def diagnostics():
    lines = [f"locitize {VERSION}", f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
             f"Platform: {sys.platform}", "Session discovery: local files only (7 providers)"]
    for name in ("PySide6", "PyYAML", "psutil", "Pillow"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        lines.append(f"{name}: {version}")
    lines += ["", "No usernames, paths, transcripts, credentials or environment values are included.",
              "Local inference does not disable third-party tools' telemetry or network access.",
              "Session readers adapted from Session Portal (MIT). See THIRD_PARTY.md."]
    return "\n".join(lines)
