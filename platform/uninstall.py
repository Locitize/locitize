"""Safe uninstall (M18.16): remove what setup created, keep what is yours.

Owner request: once installed, LOCITIZE must be as easy to remove as it was to
add - a real dialog, not a hunt through folders. Two rules shape this file:

1. STDLIB ONLY, like the setup wizard and for the same reason: this runs under
   the bootstrap `python` while the virtual environments it is deleting are the
   thing being deleted. It can never depend on them.
2. THE USER'S MODELS ARE SACRED. Downloaded GGUFs and fine-tuned runs exist
   nowhere else on the machine (imports are hardlinks, but downloads and
   training outputs are originals). "Keep my model files" is ON by default and
   preserves <data root>/models and <data root>/finetune - everything else in
   the data root (logs, config, binaries, chat state) is re-creatable by setup.

What an uninstall removes:
- the three virtual environments (.venv, .webui-venv, finetune-studio/.venv)
- the data root (minus the kept folders when the checkbox is on)
- the %LOCALAPPDATA%/LOCITIZE fallback root
- the Desktop shortcut

What it never touches: the cloned source tree itself (the dialog tells the user
to delete the folder for a complete removal), model files anywhere else on the
machine, and anything outside the paths above. LOCITIZE's own running services
are stopped first; nothing belonging to another program is ever terminated.

Entry points: `LOCITIZE.bat --uninstall`, or Settings > Uninstall in the app.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Data-root folders preserved by "Keep my model files".
KEPT_DIRS = ("models", "finetune", "sessions")


def resolve_data_root() -> Path:
    """The same resolution order config.py uses, in stdlib form."""
    explicit = (os.environ.get("LOCITIZE_DATA_DIR") or "").strip()
    if explicit:
        return Path(explicit)
    portable = BASE_DIR / "locitize-data"
    if portable.is_dir():
        return portable
    local = (os.environ.get("LOCALAPPDATA") or "").strip()
    if local:
        return Path(local) / "LOCITIZE"
    return Path.home() / ".locitize"


def desktop_shortcut() -> Path | None:
    """The Desktop LOCITIZE.lnk, honouring OneDrive redirection. Best-effort."""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "[Environment]::GetFolderPath('Desktop')"],
            capture_output=True, text=True, timeout=15, check=False,
            creationflags=_NO_WINDOW,
        )
        desktop = (proc.stdout or "").strip()
        if desktop:
            return Path(desktop) / "LOCITIZE.lnk"
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def uninstall_plan(
    repo_dir: Path | None = None,
    data_root: Path | None = None,
    keep_models: bool = True,
) -> tuple[list[Path], list[Path]]:
    """Pure planner: (paths to delete, paths deliberately kept).

    With keep_models, the data root is expanded into its children so the kept
    folders survive in place; otherwise the whole root is one delete target.
    Only paths that exist are returned - the plan is what will really happen.
    """
    repo = Path(repo_dir) if repo_dir is not None else REPO_DIR
    root = Path(data_root) if data_root is not None else resolve_data_root()

    delete: list[Path] = []
    keep: list[Path] = []

    for venv in (repo / ".venv", repo / ".webui-venv",
                 repo / "finetune-studio" / ".venv"):
        if venv.exists():
            delete.append(venv)

    if root.exists():
        if keep_models:
            for child in sorted(root.iterdir()):
                if child.name.lower() in KEPT_DIRS:
                    keep.append(child)
                else:
                    delete.append(child)
        else:
            delete.append(root)

    local = (os.environ.get("LOCALAPPDATA") or "").strip()
    if local:
        fallback = Path(local) / "LOCITIZE"
        if fallback.exists() and fallback.resolve() != root.resolve():
            if keep_models and any((fallback / name).exists() for name in KEPT_DIRS):
                for child in sorted(fallback.iterdir()):
                    (keep if child.name.lower() in KEPT_DIRS else delete).append(child)
            else:
                delete.append(fallback)

    shortcut = desktop_shortcut()
    if shortcut is not None and shortcut.exists():
        delete.append(shortcut)

    return delete, keep


def stop_locitize_processes(data_root: Path, repo_dir: Path) -> int:
    """Stop processes running FROM this install's own paths. Never another app's.

    Matched strictly by executable path prefix (the data root's bin and the
    install's venvs), the same ownership rule as the GPU ledger. Best-effort:
    a failure to stop something surfaces later as a locked-file message.
    """
    prefixes = [str(data_root), str(repo_dir / ".venv"),
                str(repo_dir / ".webui-venv"),
                str(repo_dir / "finetune-studio" / ".venv")]
    clauses = " -or ".join(
        f"$_.Path -like '{p}\\*'".replace("'", "''") if "'" in p
        else f"$_.Path -like '{p}\\*'"
        for p in prefixes
    )
    script = (
        "$procs = Get-Process | Where-Object { " + clauses + " }; "
        "$procs | Stop-Process -Force -ErrorAction SilentlyContinue; "
        "($procs | Measure-Object).Count"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30, check=False,
            creationflags=_NO_WINDOW,
        )
        return int((proc.stdout or "0").strip() or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def execute(delete: list[Path], say=print) -> list[str]:
    """Delete each planned path; return the failures (empty = clean)."""
    failures: list[str] = []
    for target in delete:
        try:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            say(f"  removed {target}")
        except OSError as exc:
            failures.append(f"{target}: {exc}")
            say(f"  FAILED  {target}: {exc}")
    return failures


def run_gui() -> int:
    """The themed confirm dialog, then the uninstall. Returns exit code."""
    import tkinter as tk
    from tkinter import ttk

    import setup_wizard  # stdlib-only, brings the shared dark theme

    root_win = tk.Tk()
    root_win.title("Uninstall LOCITIZE")
    setup_wizard._apply_theme(root_win)
    setup_wizard._brand_window(root_win)
    root_win.geometry("640x520")

    frame = ttk.Frame(root_win, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Uninstall LOCITIZE",
              font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(
        frame,
        text="This removes what setup installed on this machine. Your model "
             "files elsewhere on disk are never touched.",
        wraplength=600, justify="left", style="Muted.TLabel",
    ).pack(anchor="w", pady=(4, 10))

    keep_var = tk.BooleanVar(value=True)
    listing = tk.Text(frame, height=14, wrap="none", relief="flat")
    setup_wizard._style_text(listing)
    listing.pack(fill="both", expand=True, pady=(0, 10))

    def refresh_listing() -> None:
        delete, keep = uninstall_plan(keep_models=keep_var.get())
        listing.configure(state="normal")
        listing.delete("1.0", "end")
        listing.insert("end", "Will remove:\n")
        for p in delete:
            listing.insert("end", f"  - {p}\n")
        if keep:
            listing.insert("end", "\nWill KEEP (models, fine-tuned runs and session details):\n")
            for p in keep:
                listing.insert("end", f"  + {p}\n")
        if not delete:
            listing.insert("end", "\nNothing to remove - LOCITIZE is not installed.\n")
        listing.configure(state="disabled")

    ttk.Checkbutton(
        frame,
        text="Keep my models, fine-tuned runs and session details",
        variable=keep_var, command=refresh_listing,
    ).pack(anchor="w")
    status = ttk.Label(frame, text="", style="Muted.TLabel", wraplength=600)
    status.pack(anchor="w", pady=(6, 0))

    nav = ttk.Frame(frame)
    nav.pack(fill="x", pady=(10, 0))
    outcome = {"code": 1}

    def do_uninstall() -> None:
        delete, _keep = uninstall_plan(keep_models=keep_var.get())
        stopped = stop_locitize_processes(resolve_data_root(), REPO_DIR)
        if stopped:
            status.configure(text=f"stopped {stopped} running LOCITIZE process(es)...")
            root_win.update_idletasks()
        failures = execute(delete, say=lambda _t: None)
        if failures:
            status.configure(
                text="Some items could not be removed (close LOCITIZE and any "
                     "terminals in these folders, then run uninstall again): "
                     + failures[0]
            )
            refresh_listing()
            return
        outcome["code"] = 0
        status.configure(
            text="Done. To remove LOCITIZE completely, delete this folder: "
                 f"{REPO_DIR}"
        )
        refresh_listing()
        uninstall_btn.configure(state="disabled")
        close_btn.configure(text="Close")

    uninstall_btn = ttk.Button(nav, text="Uninstall", command=do_uninstall,
                               style="Accent.TButton")
    uninstall_btn.pack(side="right")
    close_btn = ttk.Button(nav, text="Cancel", command=root_win.destroy)
    close_btn.pack(side="right", padx=6)

    refresh_listing()
    root_win.mainloop()
    return outcome["code"]


def main() -> int:
    if "--headless" in sys.argv:
        keep = "--delete-models" not in sys.argv
        delete, kept = uninstall_plan(keep_models=keep)
        stop_locitize_processes(resolve_data_root(), REPO_DIR)
        failures = execute(delete)
        for p in kept:
            print(f"  kept    {p}")
        print("uninstall " + ("complete" if not failures else
                              f"finished with {len(failures)} failure(s)"))
        print(f"To remove LOCITIZE completely, delete: {REPO_DIR}")
        return 0 if not failures else 1
    try:
        return run_gui()
    except Exception as exc:  # noqa: BLE001 - no window? fall back to headless
        sys.stderr.write(f"could not open the uninstall window ({exc}); "
                         f"run with --headless instead\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
