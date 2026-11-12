"""Keep the desktop running: start it again when it crashes.

The desktop owns the model server, the router and the chat app. When its process
dies without being closed (a crash, a kill from outside), every app that talks to
the local models loses them until someone notices and opens locitize again - one
owner found it down in the morning after it died in the night (2026-11-11).

This runs the desktop as a child and starts it again when it exits with an error,
unless the user closed it (exit code 0) or it is failing as it starts (a broken
install that a restart would only repeat). At most MAX_RESTARTS restarts within
WINDOW_S, so a desktop that keeps crashing is left stopped. The new desktop clears
the model server the dead one left behind (gpu_ledger.orphaned_servers).

Standard library only, so it runs on the same interpreter as the desktop.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

MAX_RESTARTS = 3
WINDOW_S = 600.0
# A desktop that dies sooner than this after starting is failing, not crashing:
# a missing Qt, a broken settings file. Starting it again would fail the same way.
MIN_UPTIME_S = 30.0
PAUSE_S = 5.0

# Only the plain desktop is kept. The text menu and one-off commands (--health,
# --smoke-start ...) run once, as before.
KEPT_ARGS = {"--desktop", "--start-hidden"}


def keeps(args: list[str]) -> bool:
    """Is this a desktop start the keeper should look after?"""
    return "--desktop" in args and set(args) <= KEPT_ARGS


def keep(
    command: list[str],
    *,
    run: Callable[[list[str]], int] = subprocess.call,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda text: None,
) -> int:
    """Run `command` until it exits cleanly; start it again after a crash.

    Returns the last exit code: 0 when the user closed it, the crash code when
    it was given up on.
    """
    restarts: list[float] = []
    while True:
        started = clock()
        code = run(command)
        ended = clock()
        if code == 0:
            return 0
        if ended - started < MIN_UPTIME_S:
            log(f"the desktop exited with code {code} {ended - started:.0f} s after starting; "
                "not starting it again (it is failing to start, not crashing)")
            return code
        restarts = [t for t in restarts if ended - t < WINDOW_S]
        if len(restarts) >= MAX_RESTARTS:
            log(f"the desktop exited with code {code}, its {len(restarts) + 1}th crash in "
                f"{WINDOW_S / 60:.0f} minutes; leaving it stopped")
            return code
        restarts.append(ended)
        log(f"the desktop exited with code {code} after {ended - started:.0f} s; "
            f"starting it again in {PAUSE_S:.0f} s")
        sleep(PAUSE_S)


def _file_log(text: str) -> None:
    """Append to <data root>/logs/keeper.log. Never raises."""
    try:
        from config import resolve_data_dir  # noqa: PLC0415 - only when there is something to say

        path = Path(resolve_data_dir()) / "logs" / "keeper.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
    except Exception:  # noqa: BLE001 - a log line is never worth failing over
        pass


def main(args: list[str], interpreter_flags: list[str] | None = None) -> int:
    """Start launcher.py with `args`, kept running when it is the desktop."""
    here = Path(__file__).resolve().parent
    command = [sys.executable, *(interpreter_flags or []), str(here / "launcher.py"), *args]
    if not keeps(args):
        return subprocess.call(command)
    try:
        return keep(command, log=_file_log)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main(sys.argv[1:]))
