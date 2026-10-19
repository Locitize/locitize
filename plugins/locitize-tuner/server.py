"""Start the bundled standalone helper through its private dependency environment."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> int:
    """Reject an incomplete package; never substitute a checkout or external install."""
    package = Path(__file__).resolve().parent
    entry = package / "runtime" / "local_tuning_mcp.py"
    if not entry.is_file():
        print("Incomplete plugin package: bundled runtime is missing. Reinstall the complete package.",
              file=sys.stderr)
        return 2
    from bootstrap import data_dir, helper_python, migrate_legacy, reconfigure
    data = data_dir()
    if migrate_legacy(data) is not None:
        reconfigure(data, package)
    python = helper_python(data)
    if not python.is_file():
        print("Run bootstrap.py once to install the standalone helper dependencies.", file=sys.stderr)
        return 2
    os.environ["LOCITIZE_DATA_DIR"] = str(data)
    # Overlay preserves protocol stdio and avoids a second wrapper process owner.
    os.chdir(entry.parent)
    # Only the fixed --serve and --desktop-local modes are forwarded; MCP runs
    # take no arguments.
    extra = sys.argv[1:3] if sys.argv[1:2] in (["--serve"], ["--desktop-local"]) else []
    os.execv(str(python), [str(python), str(entry), *extra])


if __name__ == "__main__":
    raise SystemExit(main())
