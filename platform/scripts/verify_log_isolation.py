"""AC11 harness: prove the test suite does not pollute the production logs/ dir.

The carried defect (found during the M3+4 owner session): running tests / QA
sessions wrote artifacts (qa_status.log, qa_switch.log, ...) into the REAL runtime
directory Codebase/platform/logs/ that a live user's session also writes to. The
M3 AC8 check only proved those files were not git-TRACKED; an untracked file still
clutters a real user's runtime log folder. This is the stronger check.

Mechanism (Architecture M5.12):
1. Snapshot the mtime + content hash of every file under logs/ (create the dir if
   absent).
2. Run the FULL pytest suite as a subprocess with NO LOCITIZE_LOG_DIR override in
   the child environment (it is explicitly removed), so the ONLY thing keeping test
   output out of logs/ is the suite's own conftest, which sets LOCITIZE_LOG_DIR to a
   throwaway temp dir. If that mechanism is broken, this check catches it.
3. Re-snapshot logs/ and exit nonzero if ANY file changed, appeared, or vanished.

Run from Codebase/platform: `python scripts/verify_log_isolation.py [--json]`.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Platform dir is the parent of scripts/.
PLATFORM_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = PLATFORM_DIR / "logs"


def snapshot(directory: Path) -> dict[str, tuple[int, str]]:
    """Map each file under `directory` to (mtime_ns, sha256) for change detection."""
    result: dict[str, tuple[int, str]] = {}
    if not directory.exists():
        return result
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            rel = str(path.relative_to(directory))
            result[rel] = (path.stat().st_mtime_ns, hashlib.sha256(data).hexdigest())
    return result


def diff(before: dict, after: dict) -> dict[str, list[str]]:
    """Return the changed/added/removed file sets between two snapshots."""
    before_keys, after_keys = set(before), set(after)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed = sorted(k for k in before_keys & after_keys if before[k] != after[k])
    return {"added": added, "removed": removed, "changed": changed}


def main(argv: list[str] | None = None) -> int:
    as_json = "--json" in (argv if argv is not None else sys.argv[1:])

    # Ensure the production dir exists so an absent dir is not mistaken for "clean".
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    before = snapshot(LOGS_DIR)

    # Run the suite with LOCITIZE_LOG_DIR explicitly removed from the child env, so
    # the only isolation in force is the suite's own conftest redirect.
    child_env = dict(os.environ)
    child_env.pop("LOCITIZE_LOG_DIR", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(PLATFORM_DIR),
        env=child_env,
        capture_output=True,
        text=True,
    )

    after = snapshot(LOGS_DIR)
    changes = diff(before, after)
    polluted = any(changes.values())
    ok = (proc.returncode == 0) and not polluted

    report = {
        "ok": ok,
        "pytest_returncode": proc.returncode,
        "logs_dir": str(LOGS_DIR),
        "polluted": polluted,
        "changes": changes,
    }
    if as_json:
        print(json.dumps(report))
    else:
        print(f"pytest exit: {proc.returncode}")
        if polluted:
            print("FAIL: the test run changed the production logs/ directory:")
            for kind, items in changes.items():
                for item in items:
                    print(f"  {kind}: {item}")
        elif proc.returncode != 0:
            print("FAIL: pytest did not pass; see its output below")
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
        else:
            print("PASS: full pytest run left logs/ byte-for-byte unchanged")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
