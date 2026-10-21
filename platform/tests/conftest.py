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


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_recycle_bin(monkeypatch):
    """Keep every test out of the owner's real Recycle Bin.

    recycle.send_to_recycle_bin is swapped for a stand-in that applies the
    same refusals (files only, nothing half-done) and then removes the files,
    so a test can check what is gone. The calls are recorded on the fixture
    value. tests/test_recycle.py reaches the real function through
    `recycle._real_send_to_recycle_bin` where it needs to.
    """
    import recycle

    calls: list[list[str]] = []

    def fake(paths):
        paths = list(paths)
        if not paths:
            return
        for path in paths:
            if not os.path.isfile(path):
                raise OSError(f"{path} is not a file")
        calls.append(paths)
        for path in paths:
            os.remove(path)

    monkeypatch.setattr(recycle, "_real_send_to_recycle_bin", recycle.send_to_recycle_bin, raising=False)
    monkeypatch.setattr(recycle, "send_to_recycle_bin", fake)
    return calls
