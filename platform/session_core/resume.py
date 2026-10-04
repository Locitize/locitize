"""Resume execution layer.

Providers build :class:`ResumeCommand` value objects; this module is the only
place that actually spawns a terminal. Kept separate so resume-command
construction stays unit-testable (tests assert the command string, no
subprocess) and so a future cross-platform backend can live here without
touching providers.
"""
from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

from .config import (
    AMP_DIR,
    CODEX_EXE_DIR,
    CODEX_PROGRAMS_EXE_DIR,
    CREATE_NO_WINDOW,
    GROK_EXE,
)
from .models import ResumeCommand

# Win32 SW_MAXIMIZE. Python exposes SW_HIDE but not this show-state constant.
SW_MAXIMIZE = 3


# -- Executable discovery (mirrors v1) ---------------------------------------
def find_codex_exe() -> str:
    candidates = [
        CODEX_PROGRAMS_EXE_DIR / "codex.exe",
        CODEX_EXE_DIR / "codex.exe",
    ]
    if CODEX_EXE_DIR.exists():
        for d in sorted(CODEX_EXE_DIR.iterdir(), reverse=True):
            candidates.append(d / "codex.exe")
    found = shutil.which("codex")
    if found:
        candidates.append(Path(found))
    for exe in candidates:
        if exe.is_file():
            return str(exe)
    return ""


def find_claude_exe() -> str:
    return shutil.which("claude") or "claude"


def find_grok_exe() -> str:
    found = shutil.which("grok")
    if found:
        return found
    if GROK_EXE.exists():
        return str(GROK_EXE)
    return "grok"


def find_amp_exe() -> str:
    found = shutil.which("amp")
    if found:
        return found
    local = AMP_DIR / "bin" / "amp.exe"
    if local.exists():
        return str(local)
    return "amp"


def has_windows_terminal() -> bool:
    return shutil.which("wt") is not None


# -- PowerShell single-quoting (mirrors v1) ----------------------------------
# PowerShell treats the typographic quotes U+2018..U+201B exactly like the
# ASCII apostrophe, so each must be doubled too, or a folder named with one
# (x + U+2019 + '; calc; ') would end the string early and run the rest.
_PS_SINGLE_QUOTES = ("'", "\u2018", "\u2019", "\u201a", "\u201b")


def ps_single_quote(value: str) -> str:
    for quote in _PS_SINGLE_QUOTES:
        value = value.replace(quote, quote * 2)
    return "'" + value + "'"


def ps_encoded(command: str) -> str:
    """The -EncodedCommand form of a PowerShell command (base64 UTF-16LE).

    Passing the command encoded means no other parser sees it: Windows
    Terminal splits its own command line on ';', which would otherwise turn
    part of a folder name into a separate wt command.
    """
    return base64.b64encode(command.encode("utf-16-le")).decode("ascii")


def _maximized_startupinfo():
    """Build Windows startup metadata that asks the new console to maximize."""
    startupinfo_type = getattr(subprocess, "STARTUPINFO", None)
    if startupinfo_type is None:
        return None
    startupinfo = startupinfo_type()
    startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    startupinfo.wShowWindow = SW_MAXIMIZE
    return startupinfo


def _start_powershell(cwd: str, command: str) -> subprocess.Popen:
    """Start a provider command in a visible interactive PowerShell terminal."""
    cwd = cwd or str(Path.home())
    if not Path(cwd).exists():
        raise FileNotFoundError(f"Session working directory does not exist: {cwd}")
    startup = f"Set-Location -LiteralPath {ps_single_quote(cwd)}; {command}"
    if has_windows_terminal():
        return subprocess.Popen(
            ["wt", "--maximized", "-d", cwd, "powershell", "-NoExit",
             "-EncodedCommand", ps_encoded(startup)],
            creationflags=CREATE_NO_WINDOW,
        )

    # Launch PowerShell itself as the process attached to a new console. Using
    # `cmd /c start` behind CREATE_NO_WINDOW lets the child inherit a hidden,
    # console-less boundary, which interactive CLIs correctly reject.
    return subprocess.Popen(
        ["powershell", "-NoExit", "-EncodedCommand", ps_encoded(startup)],
        cwd=cwd,
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        startupinfo=_maximized_startupinfo(),
    )


def launch(cmd: ResumeCommand) -> None:
    """Execute a :class:`ResumeCommand` in a maximized terminal."""
    _start_powershell(cmd.cwd, cmd.shell_command)
