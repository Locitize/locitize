"""Pytest bootstrap for the LOCITIZE platform test suite.

Milestone-1 modules are flat top-level modules (import launcher, import health,
...) that expect the platform directory on sys.path (Architecture section 1).

This conftest lives in tests/ (not the platform root) on purpose: the platform
directory is literally named "platform" and carries an __init__.py, so a conftest
placed there would be inferred by pytest as the module "platform.conftest" and
collide with Python's stdlib `platform`. Keeping the conftest under tests/ (which
has no __init__.py) sidesteps that package inference entirely; pytest imports the
tests as plain top-level modules and this file adds the platform dir to sys.path.
"""

import os
import sys
import tempfile
from pathlib import Path

# Parent of tests/ is the platform directory holding the flat modules.
_PLATFORM_DIR = str(Path(__file__).resolve().parent.parent)
if _PLATFORM_DIR not in sys.path:
    sys.path.insert(0, _PLATFORM_DIR)

# Log isolation (Architecture M5.12, carried defect AC11). Point every log writer
# -- the logging handlers, the launcher's service logs, and the benchmark runner --
# at a throwaway temp directory for the ENTIRE test session BEFORE any platform
# module is imported, so a test run can never write into the tracked production
# logs/ directory a live user's session uses. This is set at conftest import time
# (not in a fixture) precisely so import-time or module-level logging in any test
# still lands in the temp dir. logger.resolve_log_dir reads this variable first.
if not os.environ.get("LOCITIZE_LOG_DIR"):
    os.environ["LOCITIZE_LOG_DIR"] = tempfile.mkdtemp(prefix="locitize-test-logs-")
