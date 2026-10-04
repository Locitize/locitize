"""First-run setup wizard: the Tkinter bootstrap front end (M15).

The window a new user meets. It exists because of one circular problem: the real
LOCITIZE desktop is PySide6, and on a fresh machine PySide6 is not installed yet,
so the UI that installs the dependencies cannot itself be one of them. tkinter
ships with CPython on Windows, so this window always draws - on a bare `python`,
with no venv, no wheels, and no network having been touched.

Four screens, in the order a person actually decides things:

  1. Welcome   - what LOCITIZE is, and what this machine already has.
  2. Features  - tick what you want. Implied features tick themselves, visibly.
  3. Confirm   - the itemised list and the real download total, before consent.
  4. Install   - live log, then a Launch button that hands off to the Qt desktop.

The wizard holds no policy of its own. setup_plan decides what a selection costs
and setup_env knows how to do it; this file draws them and owns the thread
boundary, exactly the way desktop.py draws gui_controller. Install work runs on a
worker thread and reports back through a queue polled by the Tk event loop, so a
2.6 GB pip resolve never freezes the window.

Nothing is downloaded before the Install button on screen 3. ASCII only.
"""

from __future__ import annotations

import os
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

import setup_env
import setup_plan

WINDOW_TITLE = "LoCiTiZe setup"
PAD = 12

# M18.5 (wizard polish): the same dark palette as the Qt desktop, so the first
# window a user meets already looks like the product - not stock-gray tkinter.
BG = "#1e1f22"        # window ground
PANEL = "#2b2d30"     # text/log surfaces
EDGE = "#3a3e44"      # borders
INK = "#e8eaed"       # body text
MUTED = "#9aa0a6"     # secondary text
ACCENT = "#4f8cff"    # primary action / step eyebrow
ACCENT_HOVER = "#67a0ff"
STEPS_TOTAL = 4

# The logo wordmark, letter by letter: capitals carry the icon's blue-to-violet
# gradient, lowercase is the icon's near-white (sampled from
# assets/locitize_icon_1024.png).
WORDMARK: tuple[tuple[str, str], ...] = (
    ("L", "#5dacfe"), ("o", "#e9e9ea"), ("C", "#699ffd"), ("i", "#e9e9ea"),
    ("T", "#7196fe"), ("i", "#e9e9ea"), ("Z", "#8087fe"), ("e", "#e9e9ea"),
)


def _wordmark_title(parent: tk.Misc, lead: str) -> tk.Text:
    """A heading of `lead` followed by the LoCiTiZe wordmark in the logo's colors.

    A one-line read-only Text, because a Label has one color and the wordmark
    needs one per letter.
    """
    plain = ("Segoe UI", 16, "bold")
    brand = ("Segoe UI", 16, "bold italic")
    letters = "".join(ch for ch, _ in WORDMARK)
    text = tk.Text(
        parent, height=1, width=len(lead) + len(letters) + 1, wrap="none",
        background=BG, foreground=INK, font=plain, borderwidth=0,
        highlightthickness=0, relief="flat", cursor="arrow", takefocus=0,
    )
    text.insert("end", lead)
    for index, (ch, color) in enumerate(WORDMARK):
        tag = f"wordmark{index}"
        text.tag_configure(tag, foreground=color, font=brand)
        text.insert("end", ch, tag)
    text.configure(state="disabled")
    return text


def _fmt_duration(seconds: float) -> str:
    """'47s' or '6m 12s' - the shape a person reads at a glance."""
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m {total % 60:02d}s"



def _apply_theme(root: tk.Tk) -> None:
    """Dark-theme every ttk widget class the wizard uses (stdlib-only).

    Built on the 'clam' theme because it is the one stock theme whose element
    colors are fully styleable; the Windows-native themes ignore most of these
    settings. Wrapped in try/except as a whole: if any of it fails on an exotic
    Tk build, the wizard still runs - stock-gray beats not drawing at all.
    """
    try:
        style = ttk.Style(root)
        style.theme_use("clam")
        root.configure(background=BG)
        style.configure(".", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Step.TLabel", foreground=ACCENT, font=("Segoe UI", 9, "bold"))
        style.configure(
            "TButton", background=PANEL, foreground=INK, bordercolor=EDGE,
            focuscolor=EDGE, padding=(14, 6),
        )
        style.map(
            "TButton",
            background=[("disabled", BG), ("active", "#33373e")],
            foreground=[("disabled", MUTED)],
        )
        style.configure(
            "Accent.TButton", background=ACCENT, foreground="#ffffff",
            bordercolor=ACCENT,
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", PANEL), ("active", ACCENT_HOVER)],
            foreground=[("disabled", MUTED)],
        )
        style.configure(
            "TCheckbutton", background=BG, foreground=INK, focuscolor=BG,
            indicatorbackground=PANEL, indicatorforeground=ACCENT,
        )
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure(
            "Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL,
            bordercolor=EDGE, lightcolor=ACCENT, darkcolor=ACCENT,
        )
        style.configure(
            "Vertical.TScrollbar", background=PANEL, troughcolor=BG,
            bordercolor=BG, arrowcolor=MUTED,
        )
    except tk.TclError:
        pass


def _style_text(widget: tk.Text) -> None:
    """Dark-surface a tk.Text (the classic widgets take colors directly)."""
    widget.configure(
        background=PANEL, foreground=INK, insertbackground=INK,
        selectbackground=ACCENT, selectforeground="#ffffff",
        highlightthickness=0, borderwidth=0, padx=10, pady=8,
        font=("Segoe UI", 10),
    )


def _brand_window(root: tk.Tk) -> None:
    """Give the wizard LOCITIZE's icon and taskbar identity, and center it.

    Same AppUserModelID fix as the desktop (M18.3): without an explicit AUMID a
    python-hosted window groups under Python on the taskbar with python's icon.
    Everything here is cosmetic and best-effort - never block the bootstrap.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "Locitize.LOCITIZE.Setup"
            )
        except (OSError, AttributeError):
            pass
    ico = Path(__file__).resolve().parent / "assets" / "locitize_launcher.ico"
    if ico.is_file():
        try:
            root.iconbitmap(default=str(ico))
        except tk.TclError:
            pass
    try:
        root.update_idletasks()
        width, height = 760, 560
        x = (root.winfo_screenwidth() - width) // 2
        y = max(0, (root.winfo_screenheight() - height) // 2 - 30)
        root.geometry(f"{width}x{height}+{x}+{y}")
    except tk.TclError:
        pass


def _registry_id(filename: str) -> str:
    """modelhub.registry_id_for, imported lazily (modelhub is stdlib-only)."""
    import modelhub

    return modelhub.registry_id_for(filename)


def _venv_exe_in(venv_dir: Path) -> Path:
    """The interpreter inside an arbitrary venv directory."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


class SetupWizard:
    """The four-screen wizard. One instance per run; not reusable after close."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("760x560")
        self.root.minsize(680, 520)
        _apply_theme(self.root)
        _brand_window(self.root)

        self._events: queue.Queue[tuple[str, str]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._state: dict[str, bool] = {}
        self._machine = setup_env.Machine()
        self._vars: dict[str, tk.BooleanVar] = {}
        self._launch_ok = False
        # Owner decision 2026-08-29: LOCITIZE ships no model names, so there
        # is no catalog picker. The first_model step SCANS this machine for
        # GGUFs the user already has and imports them; a machine with none is
        # pointed at their own HuggingFace search after launch.
        self._found_models: list[dict] = []
        # Consent for the V-NONE rung. False until the user ticks it on the
        # confirm screen; download_verified refuses an unchecked binary without it.
        self._unverified_ok = False

        self._frame = ttk.Frame(self.root, padding=PAD)
        self._frame.pack(fill="both", expand=True)
        self._show_welcome()

    # -- screen 1 ---------------------------------------------------------

    def _clear(self) -> ttk.Frame:
        for child in self._frame.winfo_children():
            child.destroy()
        return self._frame

    def _heading(
        self,
        parent: tk.Misc,
        title: str,
        subtitle: str,
        step: int | None = None,
        wordmark: bool = False,
    ) -> None:
        if step is not None:
            ttk.Label(
                parent, text=f"STEP {step} OF {STEPS_TOTAL}", style="Step.TLabel"
            ).pack(anchor="w", pady=(0, 2))
        if wordmark:
            _wordmark_title(parent, title).pack(anchor="w")
        else:
            ttk.Label(parent, text=title, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            parent, text=subtitle, wraplength=700, justify="left",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, PAD))

    def _show_welcome(self) -> None:
        body = self._clear()
        self._heading(
            body,
            "Welcome to ",
            "Download it. Locitize it. Talk to it. Everything runs on this machine "
            "- no cloud call, no API key, nothing leaves the machine.",
            step=1,
            wordmark=True,
        )
        status = ttk.Label(body, text="Checking what this machine already has...")
        status.pack(anchor="w")
        detail = tk.Text(body, height=12, wrap="word", relief="flat")
        _style_text(detail)
        detail.pack(fill="both", expand=True, pady=PAD)
        detail.configure(state="disabled")

        nav = ttk.Frame(body)
        nav.pack(fill="x")
        next_btn = ttk.Button(nav, text="Next", state="disabled", style="Accent.TButton")
        next_btn.pack(side="right")
        ttk.Button(nav, text="Cancel", command=self.root.destroy).pack(side="right", padx=6)

        def scan() -> None:
            # No queue event here: _run_async sequences this against the Tk thread
            # by watching the worker, and a stray event would be drained later by
            # the install screen's pump.
            self._state, self._machine = setup_env.detect_state()

        def finish() -> None:
            lines = []
            if self._machine.gpu_name:
                lines.append(f"GPU          : {self._machine.gpu_name}")
            else:
                lines.append("GPU          : none detected (CPU build will be used)")
            found = setup_env.find_llama_server()
            lines.append(f"llama.cpp    : {found or 'not found - will be downloaded'}")
            venv = "present" if self._state.get("platform_venv") else "will be created"
            lines.append(f"Python venv  : {venv}")
            # Name what is already here rather than "3 of 19 components",
            # which read like a step counter and said nothing about which.
            present = [
                setup_plan.REQUIREMENTS[key].label
                for key, ok in self._state.items()
                if ok and key in setup_plan.REQUIREMENTS
            ]
            lines.append("")
            if present:
                lines.append("Already on this PC (will be skipped):")
                lines.extend(f"  - {label}" for label in present)
            else:
                lines.append("Nothing installed yet - this is a fresh setup.")
            lines.append("")
            lines.append("Next, choose which features to install.")
            for note in self._machine.notes:
                lines.append("")
                lines.append(note)
            detail.configure(state="normal")
            detail.insert("1.0", "\n".join(lines))
            detail.configure(state="disabled")
            status.configure(text="Ready.")
            next_btn.configure(state="normal", command=self._show_features)

        self._run_async(scan, finish)

    # -- screen 2 ---------------------------------------------------------

    def _show_features(self) -> None:
        body = self._clear()
        self._heading(
            body,
            "Choose what you need",
            "Tick the capabilities you want. You can run setup again later to add "
            "more - nothing already installed is downloaded twice.",
            step=2,
        )

        canvas = tk.Canvas(body, highlightthickness=0, background=BG, borderwidth=0)
        scroll = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="top", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        total = ttk.Label(inner, text="")
        for feature in setup_plan.FEATURES:
            var = tk.BooleanVar(value=feature.default_on)
            self._vars[feature.key] = var
            row = ttk.Frame(inner)
            row.pack(fill="x", pady=(6, 0), anchor="w")
            box = ttk.Checkbutton(
                row,
                text=feature.label,
                variable=var,
                command=lambda: self._refresh_total(total),
            )
            box.pack(anchor="w")
            if feature.core:
                var.set(True)
                box.configure(state="disabled")
            ttk.Label(
                row, text=feature.summary, wraplength=640, justify="left",
                style="Muted.TLabel",
            ).pack(anchor="w", padx=(24, 0))

        # First model (M15.11): no picker and no shipped names. When nothing
        # is registered yet, the machine is scanned for GGUFs the user already
        # owns, and the install step imports them - their models, their
        # hardware, their choice already made.
        if not self._state.get("first_model"):
            # Perf audit 2026-08-31: the disk walk used to run SYNCHRONOUSLY
            # on the Tk thread right here - on a OneDrive-backed Documents
            # tree that froze the wizard for minutes with zero feedback. The
            # scan now runs through _run_async like the welcome probe, behind
            # a visible "scanning..." line; a Back-and-Next keeps the first
            # result instead of walking the disk again.
            row = ttk.Frame(inner)
            row.pack(fill="x", pady=(PAD, 0), anchor="w")
            self._tune_var = tk.BooleanVar(value=False)

            def render_found(found: list[dict]) -> None:
                try:
                    if not row.winfo_exists():
                        return  # the user left this screen mid-scan
                    for child in row.winfo_children():
                        child.destroy()
                except tk.TclError:
                    return
                if found:
                    found_gb = sum(f["size"] for f in found) / 2**30
                    ttk.Label(
                        row,
                        text=(
                            f"Found {len(found)} model file(s) already "
                            f"on this machine ({found_gb:.1f} GB). Setup will import "
                            f"them - nothing is copied or moved."
                        ),
                        wraplength=640, justify="left",
                    ).pack(anchor="w")
                    # M18.17 (owner report: "setup takes forever"): the context
                    # measurement is the long tail of an install - real
                    # generation probes per model. Make its cost visible and
                    # optional, capped at ~15 minutes either way; whatever is
                    # not measured keeps a safe default and can be tuned any
                    # time from the Models page.
                    ttk.Checkbutton(
                        row,
                        text=(
                            "Also measure each model's best context now (up to "
                            "~15 minutes). Off keeps install fast; you can tune "
                            "any model later from the Models page."
                        ),
                        variable=self._tune_var,
                    ).pack(anchor="w", pady=(4, 0))
                else:
                    ttk.Label(
                        row,
                        text=(
                            "No local model files found. After setup, use the "
                            "Models page to search huggingface.co and download one "
                            "that fits your hardware - every download is "
                            "checksum-verified."
                        ),
                        wraplength=640, justify="left",
                    ).pack(anchor="w")

            if self._found_models:
                render_found(self._found_models)
            else:
                ttk.Label(
                    row,
                    text=(
                        "Scanning this machine for model files you already "
                        "have..."
                    ),
                    wraplength=640, justify="left", style="Muted.TLabel",
                ).pack(anchor="w")

                def scan() -> None:
                    self._found_models = setup_env.find_local_models()

                def finish() -> None:
                    render_found(self._found_models)

                self._run_async(scan, finish)

        total.pack(anchor="w", pady=(PAD, 0))
        self._refresh_total(total)

        nav = ttk.Frame(body)
        nav.pack(fill="x", pady=(PAD, 0))
        ttk.Button(
            nav, text="Next", command=self._show_confirm, style="Accent.TButton"
        ).pack(side="right")
        ttk.Button(nav, text="Back", command=self._show_welcome).pack(side="right", padx=6)

    def _selection(self) -> tuple[str, ...]:
        return tuple(key for key, var in self._vars.items() if var.get())

    def _refresh_total(self, label: ttk.Label) -> None:
        plan = setup_plan.plan_for(self._selection(), self._state)
        # Implied features tick themselves so the checklist never disagrees with
        # what is about to be downloaded.
        for key in plan.selected:
            var = self._vars.get(key)
            if var is not None and not var.get():
                var.set(True)
        label.configure(text=setup_plan.summarize(plan))

    # -- screen 3 ---------------------------------------------------------

    def _show_confirm(self) -> None:
        plan = setup_plan.plan_for(self._selection(), self._state)
        body = self._clear()
        self._heading(
            body,
            "Ready to install",
            setup_plan.summarize(plan),
            step=3,
        )

        listing = tk.Text(body, height=14, wrap="word", relief="flat")
        _style_text(listing)
        listing.pack(fill="both", expand=True)
        for step in plan.steps:
            mark = "have" if step.satisfied else "get "
            size = "" if not step.size_mb else f"  ({setup_plan.format_size(step.size_mb)})"
            listing.insert("end", f"[{mark}] {step.requirement.label}{size}\n")
            listing.insert("end", f"       {step.requirement.detail}\n\n")
        listing.configure(state="disabled")

        # The V-NONE rung, surfaced rather than buried. GitHub does not publish a
        # checksum for every release asset; when it does, this is ignored and the
        # publisher digest is enforced instead.
        needs_binary = any(
            s.requirement.kind == setup_plan.KIND_BINARY for s in plan.pending
        )
        if needs_binary:
            consent = tk.BooleanVar(value=False)

            def sync() -> None:
                self._unverified_ok = bool(consent.get())

            ttk.Checkbutton(
                body,
                text=(
                    "Allow a download to proceed when its publisher offers no "
                    "checksum (size is still enforced, host is still allowlisted)"
                ),
                variable=consent,
                command=sync,
            ).pack(anchor="w", pady=(PAD, 0))

        nav = ttk.Frame(body)
        nav.pack(fill="x", pady=(PAD, 0))
        start = ttk.Button(
            nav, text="Install", command=lambda: self._show_install(plan),
            style="Accent.TButton",
        )
        start.pack(side="right")
        ttk.Button(nav, text="Back", command=self._show_features).pack(side="right", padx=6)
        if plan.nothing_to_do:
            start.configure(text="Launch locitize", command=self._launch)

    # -- screen 4 ---------------------------------------------------------

    def _show_install(self, plan: setup_plan.InstallPlan) -> None:
        body = self._clear()
        self._heading(
            body, "Installing", "This can take a while. You can watch it work.",
            step=4,
        )

        bar = ttk.Progressbar(body, mode="determinate", maximum=max(1, len(plan.pending)))
        bar.pack(fill="x")
        log = tk.Text(body, height=16, wrap="word", relief="flat")
        _style_text(log)
        log.pack(fill="both", expand=True, pady=PAD)
        log.configure(state="disabled")

        nav = ttk.Frame(body)
        nav.pack(fill="x")
        launch = ttk.Button(
            nav, text="Launch locitize", state="disabled", command=self._launch,
            style="Accent.TButton",
        )
        launch.pack(side="right")
        close = ttk.Button(nav, text="Close", command=self.root.destroy)
        close.pack(side="right", padx=6)

        def append(text: str) -> None:
            log.configure(state="normal")
            log.insert("end", text + "\n")
            log.see("end")
            log.configure(state="disabled")

        def pump() -> None:
            drained = False
            while True:
                try:
                    kind, payload = self._events.get_nowait()
                except queue.Empty:
                    break
                drained = True
                if kind == "log":
                    append(payload)
                elif kind == "step":
                    bar.step(1)
                elif kind == "done":
                    append("")
                    append(payload)
                    if self._launch_ok:
                        launch.configure(state="normal")
                    return
            if drained or (self._worker and self._worker.is_alive()):
                self.root.after(120, pump)
            else:
                self.root.after(300, pump)

        self._worker = threading.Thread(
            target=self._install, args=(plan,), daemon=True
        )
        self._worker.start()
        self.root.after(120, pump)

    def _install(self, plan: setup_plan.InstallPlan) -> None:
        """Run the plan on a worker thread. Never raises into the Tk loop."""
        say: Callable[[str], None] = lambda text: self._events.put(("log", text))
        failures: list[str] = []
        essential_failed = False
        discovered: dict[str, str] = {}

        install_started = time.monotonic()
        try:
            venv_exe = setup_env.venv_python_path()
            lock = threading.Lock()
            state_flags = {"essential_failed": False}

            def run_one(step) -> None:
                req = step.requirement
                if self._state.get(step.key):
                    say(f"-- {req.label}: ok, already present")
                    self._events.put(("step", ""))
                    return
                say(f"-- {req.label} ...")
                local: dict[str, str] = {}
                step_started = time.monotonic()
                result = self._run_step(step, venv_exe, say, local)
                elapsed = _fmt_duration(time.monotonic() - step_started)
                with lock:
                    discovered.update(local)
                    if result.ok:
                        say(f"   ok ({req.label}, {elapsed}): {result.message}")
                    else:
                        say(f"   FAILED ({req.label}, {elapsed}): {result.message}")
                        failures.append(req.label)
                        if req.essential:
                            state_flags["essential_failed"] = True
                self._events.put(("step", ""))

            def run_chain(steps) -> None:
                for step in steps:
                    run_one(step)

            # M18.17 (owner report: install felt far slower than a normal
            # setup): the steps are almost all NETWORK-bound (a ~0.7 GB
            # llama.cpp, a ~1.7 GB pip resolve, voice weights), and running
            # them one after another made the wall time their SUM. Phased now:
            #   A. virtual environments, sequential (seconds; everything else
            #      needs them to exist);
            #   B. every download concurrently - pips chained per venv (two
            #      pip runs must not share one venv), winget installs chained
            #      (winget serializes itself anyway), binaries and weights as
            #      independent jobs;
            #   C. model import last (needs the platform venv's yaml).
            # Wall time becomes roughly the LONGEST download, not the total.
            pending = list(plan.pending)
            venv_steps = [s2 for s2 in pending if s2.requirement.kind == setup_plan.KIND_VENV]
            for step in venv_steps:
                run_one(step)

            # One re-detect after the venvs exist (M15.6): marks steps the
            # machine already has satisfied instead of re-downloading them.
            fresh, _machine = setup_env.detect_state()
            self._state.update({k: v for k, v in fresh.items() if v})

            remaining = [
                s2 for s2 in pending
                if s2.requirement.kind != setup_plan.KIND_VENV
                and s2.key != "first_model"
            ]
            by_key = {s2.key: s2 for s2 in remaining}
            jobs: list[list] = []
            # Every pip into the platform venv is ONE chain: the CUDA Kokoro
            # step replaces the torch the CPU step installed, and two pips
            # resolving into one site-packages at once corrupt each other.
            platform_pips = [
                by_key.pop(k)
                for k in ("base_pip", "gui_pip", "pillow_pip", "kokoro_pip", "kokoro_pip_cuda")
                if k in by_key
            ]
            if platform_pips:
                jobs.append(platform_pips)
            winget_chain = [
                by_key.pop(k) for k in ("node_runtime", "caddy", "harness_clis")
                if k in by_key
            ]
            if winget_chain:
                jobs.append(winget_chain)
            for step in remaining:
                if step.key in by_key:
                    jobs.append([by_key.pop(step.key)])
            if jobs:
                with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
                    list(pool.map(run_chain, jobs))

            for step in pending:
                if step.key == "first_model":
                    run_one(step)
            essential_failed = state_flags["essential_failed"]

            if discovered and venv_exe.is_file():
                say("-- Saving paths to settings.yaml")
                written = setup_env.write_settings_paths(
                    venv_exe, self._machine.data_root, discovered
                )
                say(f"   {'ok' if written.ok else 'FAILED'}: {written.message}")

            # M17.11: selecting Open WebUI installs its venv/package, but the chat
            # chooser only uses it when openwebui.enabled is true (ships false,
            # opt-in). Flip it on here so a chosen Open WebUI is actually reached
            # instead of chat silently falling back to the built-in llama.cpp UI.
            if self._wants("openwebui") and venv_exe.is_file():
                say("-- Enabling Open WebUI as a chat option")
                enabled = setup_env.enable_openwebui_via_venv(
                    venv_exe, self._machine.data_root
                )
                say(f"   {'ok' if enabled.ok else 'note'}: {enabled.message}")

            # Final step (M17.7): measure each model's real context ceiling on
            # THIS GPU and apply it, instead of leaving every model at the safe
            # 8192 default. Runs only once llama.cpp and the models are in place;
            # skips itself honestly when either is absent or there is no GPU. It
            # is the measured tuner, not a computed guess - and it is resumable,
            # so a user who closes the window has every already-measured model
            # kept and the rest done on a later run.
            tune_var = getattr(self, "_tune_var", None)
            wants_tune = bool(tune_var is not None and tune_var.get())
            if not wants_tune:
                say("-- Context measuring deferred - models keep a safe default;")
                say("   tune any model from the Models page whenever you like")
            elif venv_exe.is_file() and setup_env.find_llama_server():
                say(
                    "-- Measuring the best context for each model on your GPU"
                    " (capped at ~15 minutes; resumable via locitize.vbs --setup)"
                )
                # M18.17: a hard time budget keeps first-run setup bounded. The
                # measurement is resumable by design, so a big model library
                # finishes across later runs instead of holding install hostage.
                measured = setup_env.measure_contexts(venv_exe, say, budget_s=900)
                say(f"   {'ok' if measured.ok else 'note'}: {measured.message}")

            # M17.12: drop a Desktop shortcut to locitize.vbs, the no-console
            # launcher, so the everyday way in never flashes a black console
            # window. Best-effort: a failure here never fails setup.
            say("-- Creating a Desktop shortcut (no-console launcher)")
            shortcut = setup_env.create_desktop_shortcut()
            say(f"   {'ok' if shortcut.ok else 'note'}: {shortcut.message}")
            # M22: the shared model store - one folder every local-AI tool
            # can feed from; scanning and downloads both use it.
            store = setup_env.ensure_model_store(say)
            say(f"   {'ok' if store.ok else 'note'}: {store.message}")
            # M21: the claude-local shim - `claude-local` in any terminal
            # runs Claude Code against whatever model LOCITIZE is serving.
            # Best-effort and an honest skip when claude is not installed.
            shim = setup_env.install_claude_local(say)
            say(f"   {'ok' if shim.ok else 'note'}: {shim.message}")
        except Exception as exc:  # noqa: BLE001 - a wizard must not die silently
            failures.append(str(exc))
            essential_failed = True

        self._launch_ok = not essential_failed and setup_env.venv_python_path().is_file()
        total = _fmt_duration(time.monotonic() - install_started)
        if not failures:
            summary = f"Setup complete in {total}. locitize is ready."
        elif essential_failed:
            summary = (
                f"Setup could not finish after {total}: " + ", ".join(failures) + ". "
                "Nothing was left half-written; run setup again after fixing the above."
            )
        else:
            summary = (
                f"Setup finished in {total} with optional items skipped: " + ", ".join(failures)
                + ". locitize will start; those features stay off until installed."
            )
        self._events.put(("done", summary))

    def _run_step(
        self,
        step: setup_plan.Step,
        venv_exe: Path,
        say: Callable[[str], None],
        discovered: dict[str, str],
    ) -> setup_env.StepResult:
        """Dispatch one step by requirement key. Returns, never raises."""
        key = step.key
        kind = step.requirement.kind

        if key == "platform_venv":
            return setup_env.create_venv(setup_env.REPO_DIR / ".venv")
        if key == "webui_venv":
            return setup_env.create_venv(setup_env.REPO_DIR / ".webui-venv")
        if key == "finetune_venv":
            return setup_env.create_venv(
                setup_env.REPO_DIR / "finetune-studio" / ".venv"
            )

        if kind == setup_plan.KIND_PIP:
            gpu_voice = self._wants("voice_out_gpu") and not self._state.get("kokoro_pip_cuda")
            if key == "kokoro_pip" and gpu_voice and self._machine.has_nvidia:
                # The CUDA set carries kokoro itself; installing the CPU torch
                # first would be 250 MB thrown away minutes later.
                return setup_env.StepResult(
                    key, True, "covered by the GPU build that follows", skipped=True
                )
            if key == "kokoro_pip_cuda" and not self._machine.has_nvidia:
                return setup_env.StepResult(
                    key, False,
                    "no NVIDIA card detected (nvidia-smi); the CPU voice engine "
                    "is installed and works, this step needs a GPU",
                )
            packages = setup_env.PIP_SETS.get(key, ())
            target = venv_exe
            if key == "webui_pip":
                target = _venv_exe_in(setup_env.REPO_DIR / ".webui-venv")
            elif key == "finetune_pip":
                target = _venv_exe_in(setup_env.REPO_DIR / "finetune-studio" / ".venv")
            return setup_env.pip_install(target, packages, on_output=lambda t: say("   " + t))

        if key == "llama_cpp":
            # Detection runs again here, not just at scan time: the user may have
            # unpacked one themselves between screen 1 and pressing Install, and a
            # found binary turns a 360 MB download into a one-line config write.
            found = setup_env.find_llama_server()
            if found:
                discovered["llama_cpp"] = found
                return setup_env.StepResult(key, True, f"found existing {found}", skipped=True)
            result, server = setup_env.install_llama_cpp(
                self._machine.bin_dir,
                self._machine.has_nvidia,
                confirm_unverified=self._unverified_ok,
                say=say,
            )
            if server:
                discovered["llama_cpp"] = server
            return result

        if key == "node_runtime":
            return setup_env.winget_install("OpenJS.NodeJS.LTS")
        if key == "caddy":
            return setup_env.winget_install("CaddyServer.Caddy")
        if key == "harness_clis":
            return self._install_harnesses(say)

        if key == "whisper_bin":
            result, paths = setup_env.install_whisper(
                self._machine.bin_dir, say, confirm_unverified=self._unverified_ok
            )
            discovered.update(paths)
            return result

        if key == "whisper_model":
            result, paths = setup_env.install_whisper_weights(
                self._machine.data_root / "whisper", say
            )
            discovered.update(paths)
            return result

        if key == "kokoro_weights":
            result, paths = setup_env.install_kokoro_weights(
                self._machine.data_root / "kokoro", say
            )
            discovered.update(paths)
            return result

        if key == "vision_projector":
            # Which vision model to run is the user's choice, not shipped
            # taste (owner decision 2026-08-29). Their imported models may
            # already include one; otherwise Get models finds the pair.
            return setup_env.StepResult(
                key, True,
                "choose a vision model (with its mmproj) on the Models page",
                skipped=True,
            )

        if key == "first_model":
            return self._import_found_models(venv_exe, say)

        return setup_env.StepResult(
            key, True, "no automatic installer; configure on the Settings page", skipped=True
        )

    def _wants(self, feature_key: str) -> bool:
        var = self._vars.get(feature_key)
        return bool(var is not None and var.get())

    def _import_found_models(
        self, venv_exe: Path, say: Callable[[str], None]
    ) -> setup_env.StepResult:
        """Import the GGUFs the features screen found on this machine.

        Each file is given a name in LOCITIZE's models folder without copying
        (hardlink on the same volume, absolute path otherwise) and registered
        through config.append_model_entry in the venv interpreter. A name that
        is already registered is skipped, so re-running setup never duplicates
        a row. No model is ever downloaded here: if the scan found nothing,
        the honest answer is the Models page's search, in the user's own words.
        """
        key = "first_model"
        # Seed settings.yaml + models.yaml FIRST (M17.6): the launcher's
        # bootstrap normally does this, but the wizard runs before the launcher
        # ever starts, so without it every register/path-write hits an absent
        # registry. Unconditional - it must run even when no models are found,
        # so the later llama.cpp path write has a settings.yaml to edit.
        seeded = setup_env.seed_data_root(
            venv_exe, self._machine.data_root, setup_env.BASE_DIR
        )
        if not seeded.ok:
            return setup_env.StepResult(key, False, seeded.message)
        found = self._found_models or setup_env.find_local_models(say=say)
        if not found:
            return setup_env.StepResult(
                key, True,
                "no local models found; search huggingface.co from the Models "
                "page after launch (downloads are checksum-verified)",
                skipped=True,
            )
        models_dir = self._machine.data_root / "models"
        imported = skipped = failed = 0
        for entry in found:
            location = setup_env.place_into_models_dir(entry["path"], models_dir)
            # M18.18: carry the model's vision projector when one sits beside
            # it on disk, so a VL model imports SEEING instead of text-only.
            mmproj_src = setup_env.pair_mmproj(entry["path"])
            mmproj = (
                setup_env.place_into_models_dir(mmproj_src, models_dir)
                if mmproj_src else ""
            )
            result = setup_env.register_model_via_venv(
                venv_exe,
                {
                    "data_root": str(self._machine.data_root),
                    "model_id": _registry_id(entry["name"]),
                    "name": Path(entry["name"]).stem,
                    "location": location,
                    "mmproj": mmproj,
                    "description": "Imported by first-run setup from this machine.",
                    "notes": f"Found at {entry['path']} during first-run scan.",
                },
            )
            if result.ok:
                imported += 1
                say(f"   imported {entry['name']}")
            elif "already exists" in result.message:
                skipped += 1
            else:
                failed += 1
                say(f"   {entry['name']}: {result.message[:80]}")
        summary = f"imported {imported} model(s)"
        if skipped:
            summary += f", {skipped} already registered"
        if failed:
            summary += f", {failed} failed"
        return setup_env.StepResult(key, imported > 0 or skipped > 0, summary)


    # -- plumbing ---------------------------------------------------------

    def _run_async(self, work: Callable[[], None], done: Callable[[], None]) -> None:
        """Run `work` off-thread, then `done` back on the Tk thread."""

        def runner() -> None:
            try:
                work()
            except Exception:  # noqa: BLE001 - a probe failure is not a crash
                self._events.put(("scan-done", ""))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

        def poll() -> None:
            if thread.is_alive():
                self.root.after(120, poll)
                return
            done()

        self.root.after(120, poll)

    def _launch(self) -> None:
        """Hand off to the real desktop and close the wizard."""
        exe = setup_env.venv_python_path()
        launcher = setup_env.BASE_DIR / "launcher.py"
        if exe.is_file() and launcher.is_file():
            try:
                import subprocess  # noqa: PLC0415 - only needed on the way out

                subprocess.Popen(
                    [str(exe), str(launcher), "--desktop"],
                    cwd=str(setup_env.BASE_DIR),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError:
                pass
        self.root.destroy()


def needs_setup() -> bool:
    """True when a bare launch would fail, i.e. the venv or its GUI deps are absent.

    This is the locitize.bat gate. It stays cheap and stdlib-only: one file check
    plus one import probe, so a normal start pays almost nothing for it.
    """
    exe = setup_env.venv_python_path()
    if not exe.is_file():
        return True
    return not setup_env.modules_present(exe, ("PySide6",))


def run() -> int:
    """Show the wizard. Returns a process exit code."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        message = (
            f"locitize setup could not open a window ({exc}). "
            f"Run 'locitize.bat --terminal' to set up from the terminal."
        )
        sys.stderr.write(message + "\n")
        # The launcher hides the console, so stderr alone could go unseen.
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, WINDOW_TITLE, 0x10)
        return 2
    SetupWizard(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
