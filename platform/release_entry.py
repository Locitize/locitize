"""Packaged desktop entry point using the bundled, ordinary Python interpreter."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def main():
    # Child service interpreters inherit the same user-site isolation. Keeping a
    # real interpreter is necessary for Whisper/TTS/setup subprocess contracts.
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from runtime_layout import bundled_python, environment_root

    bundled = bundled_python(here)
    if bundled:
        import subprocess
        target = environment_root(here) / ".venv"
        interpreter = target / "Scripts" / "python.exe"
        if not interpreter.is_file():
            subprocess.run([str(bundled), "-B", "-E", "-s", "-m", "venv", "--system-site-packages", str(target)],
                           check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if Path(sys.prefix).resolve() != target.resolve():
            executable = interpreter.with_name("pythonw.exe") if not sys.argv[1:] else interpreter
            child = subprocess.Popen([str(executable), "-B", "-E", "-s", str(here / "release_entry.py"), *sys.argv[1:]],
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if not sys.argv[1:] else 0)
            return child.wait() if sys.argv[1:] else 0
    if sys.argv[1:] == ["--setup"]:
        from setup_wizard import run
        return run()
    if not sys.argv[1:]:
        from config import resolve_data_dir
        if not (resolve_data_dir() / "settings.yaml").is_file():
            from setup_wizard import run
            return run()
    from launcher import main as launch

    return launch(sys.argv[1:] or ["--desktop"])


if __name__ == "__main__":
    raise SystemExit(main())
