"""AC8 harness: fail if any *.log file under logs/ is still tracked by git.

QA finding O-3 / DevOps recommendation: routine platform runs write to
Codebase/platform/logs/, and those files must not show up as tracked/modified in
git status. This script asks git which files under logs/ are tracked and exits
nonzero if any of them is a *.log file (a tracked .gitkeep placeholder is fine and
expected). Run from Codebase/platform: `python scripts/verify_logs_untracked.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Platform dir is the parent of scripts/. The logs pathspec is relative to it.
PLATFORM_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = PLATFORM_DIR / "logs"


def tracked_log_files() -> list[str]:
    """Return the *.log files under logs/ that git currently tracks (may be empty).

    Uses `git ls-files` scoped to the logs directory. git resolves the pathspec
    against the repository, so this works regardless of the current working
    directory as long as it is inside the repo.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", str(LOGS_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # git not available or not a repo: report loudly rather than passing blindly.
        raise RuntimeError(f"git ls-files failed: {result.stderr.strip()}")
    tracked = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [path for path in tracked if path.endswith(".log")]


def main() -> int:
    try:
        offenders = tracked_log_files()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1
    if offenders:
        print("FAIL: these *.log files are still tracked by git (should be ignored):")
        for path in offenders:
            print(f"  {path}")
        print("Remedy: git rm --cached <path> for each, and ensure logs/*.log is in .gitignore")
        return 1
    print("PASS: no *.log file under logs/ is tracked by git")
    return 0


if __name__ == "__main__":
    sys.exit(main())
