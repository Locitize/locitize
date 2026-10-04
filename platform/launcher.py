"""LOCITIZE platform launcher - the single entry point.

`python launcher.py` bootstraps config + logging, runs the health verify ladder,
renders the banner/status/models/applications, and drives the interactive menu
(Architecture section 3). It owns no domain logic; it delegates to config,
logger, health, models, services, documentation, and plugins.

Import must be side-effect free (AC2): all work happens inside main(), guarded by
the __main__ block. Non-interactive flags make the platform testable without a
TTY or a GPU:
  --health --json  run the ladder, print the HealthReport as JSON, exit 0/1.
  --health         same, human-readable table.
  --no-menu        bootstrap + ladder + render, then exit (render smoke test).
  --service-status [--json]   print the managed-service snapshot and exit 0
                   (honest idle state when nothing is running; no GPU needed).
  --smoke-start MODEL_ID [--ctx-size N] [--gpu-layers N] [--json]
                   start the real llama.cpp service for MODEL_ID, confirm
                   readiness via its health endpoint, then ALWAYS clean up, and
                   print a JSON outcome. Exits 0 only when the model became ready
                   and shutdown left no process behind.
  --transcribe AUDIO_FILE [--json]
                   transcribe AUDIO_FILE via whisper-server (starts it if it is
                   not already listening on the reserved port, then restores the
                   prior state), printing a JSON transcript record. (M3)
  --smoke-start-whisper [--json]
                   start the real whisper-server, confirm TCP readiness, then
                   ALWAYS clean up; JSON outcome, exit 0 only if ready+clean. (M3)
  --smoke-listen --duration N [--json]
                   start whisper-stream (mic capture) as a managed service, wait
                   N seconds, stop it cleanly; proves start/capture/exit lifecycle
                   only (no content claim). (M3)
  --listen [--duration N] [--json]
                   live mic transcription: start whisper-stream, echo the
                   transcript for N seconds (default 15), then stop cleanly. The
                   owner grades voice quality (manual AC9). (M3)
  --second-eye --goal "GOAL" [--interval N] [--diff-threshold N]
               [--min-judge-interval N] [--voice NAME] [--stop-file PATH]
                   continuous real-time watcher: screen-diff-triggered vision
                   judgment against GOAL, with an instant spoken correction via
                   Kokoro when something looks wrong. Both the vision model and
                   kokoro stay warm for the whole session. Runs until Ctrl+C, or
                   until --stop-file exists (the reliable stop path for a caller
                   in another process, e.g. an external screen-watcher integration).
When stdin is not a TTY and no flag is given, the launcher defaults to
--health --json rather than blocking on input().

Exit codes: 0 clean; 1 has-FAIL health (in --health modes) or unexpected error
handled at the boundary; 2 unrecoverable bootstrap (config) error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from logger import resolve_log_dir


def _frame_diff_fraction(a: Any, b: Any) -> float:
    """Fraction of pixels that changed between two same-size grayscale PIL images.

    Used by --second-eye to decide whether a captured frame is different enough
    from the last-judged one to be worth a vision-model call. Pixel deltas below
    16/255 are treated as noise (compression/dithering), not real change.
    """
    from PIL import ImageChops

    diff = ImageChops.difference(a, b)
    histogram = diff.histogram()
    changed = sum(histogram[16:])
    total = sum(histogram)
    return changed / total if total else 0.0



# Desktop must run under the project .venv (PySide6). System Python311 has no
# Qt bindings; starting GPU services then dying on import tore down llama/router
# (blank phone chats). Check + re-exec BEFORE any ServiceManager work.
_DESKTOP_REEXEC_ENV = "LOCITIZE_DESKTOP_VENV_REEXEC"


def _desktop_venv_python() -> Path | None:
    """Prefer repo-root .venv, then platform-local .venv. None if missing."""
    here = Path(__file__).resolve().parent
    names = (
        ("Scripts", "python.exe"),
        ("bin", "python"),
    )
    roots = (here.parent / ".venv", here / ".venv")
    for root in roots:
        for scripts, exe in names:
            candidate = root / scripts / exe
            if candidate.is_file():
                return candidate
    return None


def _interpreters_equivalent(a: Path, b: Path) -> bool:
    """True when a and b are the same env (python.exe vs pythonw.exe OK)."""
    try:
        ar, br = a.resolve(), b.resolve()
    except OSError:
        ar, br = a, b
    if ar == br:
        return True
    # Same Scripts/bin directory + python[w] sibling counts as the venv.
    if ar.parent == br.parent and ar.stem.rstrip("w") == br.stem.rstrip("w"):
        return True
    return False


def _current_has_pyside6() -> bool:
    import importlib.util

    return importlib.util.find_spec("PySide6") is not None


def _venv_has_pyside6(venv_python: Path) -> bool:
    """Ask the target interpreter (not us) whether PySide6 imports."""
    import subprocess

    if not venv_python.is_file():
        return False
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            [str(venv_python), "-c", "import PySide6"],
            capture_output=True,
            timeout=60,
            creationflags=creation,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _desktop_missing_remedy(current: str, venv: Path | None) -> str:
    venv_hint = str(venv) if venv is not None else r"<repo>\.venv\Scripts\python.exe"
    return (
        "[XX] locitize Desktop requires PySide6 in the project .venv.\n"
        f"  Current interpreter: {current}\n"
        f"  Expected venv python: {venv_hint}\n"
        "  Fix: run locitize.bat  (uses .venv automatically), or:\n"
        f"    {venv_hint} launcher.py --desktop\n"
        "  Or install:  .venv\\Scripts\\pip install PySide6\n"
        "  Terminal menu (no Qt):  locitize.bat --terminal"
    )


def ensure_desktop_interpreter(out_fn=print) -> int | None:
    """Make --desktop use .venv with PySide6, or fail loud before services start.

    Returns None when the current process is safe to continue. Returns a non-zero
    exit code when Desktop cannot start (caller must exit without starting
    llama/router). May os.execv into .venv and never return.
    """
    import os

    current = Path(sys.executable) if sys.executable else Path("python")
    venv_py = _desktop_venv_python()
    has_qt = _current_has_pyside6()

    if has_qt:
        # Prefer staying put when we already have Qt. If we are somehow outside
        # .venv but have PySide6, still proceed (dev / unusual installs).
        return None

    # No PySide6 in this interpreter. Try one re-exec into .venv when it has Qt.
    already = os.environ.get(_DESKTOP_REEXEC_ENV) == "1"
    if (
        venv_py is not None
        and not already
        and not _interpreters_equivalent(current, venv_py)
        and _venv_has_pyside6(venv_py)
    ):
        out_fn(
            f"[>>] Desktop: current interpreter lacks PySide6 "
            f"({current}); re-exec once into .venv ({venv_py})."
        )
        env = os.environ.copy()
        env[_DESKTOP_REEXEC_ENV] = "1"
        # Absolute script path so cwd does not matter after exec.
        script = str(Path(__file__).resolve())
        argv = [str(venv_py), script, *sys.argv[1:]]
        os.execve(str(venv_py), argv, env)  # noqa: S606 - intentional handoff
        return 2  # pragma: no cover - execve does not return

    out_fn(_desktop_missing_remedy(str(current), venv_py))
    return 2


# The platform banner. Fixed ASCII header (no emojis / smart typography).
BANNER = r"""
================================================================
   locitize - Download it. Locitize it. Talk to it.
   Local AI platform
================================================================
""".strip(
    "\n"
)


class Launcher:
    """Orchestrates bootstrap, the verify ladder, rendering, and the menu.

    Collaborators are injected so tests can drive main() with fakes (a stub
    HealthChecker, a fake input source) and assert output/exit code without real
    hardware. `_deps` overrides are used only by tests; production passes None and
    the real collaborators are built.
    """

    def __init__(self, base_dir: Path | None = None, deps: dict[str, Any] | None = None) -> None:
        self._base_dir = base_dir
        # Test injection hooks (all optional). Keys: config_load, providers,
        # health_checker, input_fn, output_fn.
        self._deps = deps or {}
        self._out = self._deps.get("output_fn", print)

    # ---- bootstrap -------------------------------------------------------- #

    def _bootstrap(self) -> tuple[Any, Any, Any, list[Any]]:
        """Load config + models and configure logging. Returns collaborators.

        Raises BootstrapError on an unrecoverable config problem so main() can map
        it to exit code 2 without leaking a traceback.
        """
        from config import Config
        from logger import configure_logging, get_logger

        config_load = self._deps.get("config_load", Config.load)
        settings, models, issues = config_load(self._base_dir)

        # A missing models.yaml is unrecoverable (nothing to launch/registry-fy).
        fatal = [i for i in issues if i.level == "ERROR" and i.source == "models.yaml"]
        if fatal:
            messages = "; ".join(f"{i.source}: {i.message}" for i in fatal)
            raise BootstrapError(f"configuration error - {messages}")

        # DEC-M14-9: before the first log line is written, move any pre-M14 user
        # data (logs, transcripts, the Open WebUI store, reports, the generated
        # Caddyfile) out of the install tree and into the data root. Copy-only
        # and idempotent, so this is a no-op on every run after the first. It
        # runs here, in the one bootstrap every surface goes through, rather
        # than inside Config.load, so that loading configuration stays a read.
        migration_outcome = self._migrate_user_data(settings)

        configure_logging(settings)
        # M17.1: point the privacy ledger at the data root, in the one bootstrap
        # every surface goes through, so egress recording is armed before any
        # request can be made.
        try:
            import egress_log

            egress_log.configure(settings.data_dir)
        except Exception:  # noqa: BLE001 - a witness, never a gate
            pass
        log = get_logger("launcher")
        self._log_migration(log, migration_outcome, settings.data_dir)
        for issue in issues:
            # Surface parse warnings/errors into the log for later inspection.
            log.warning("config %s: %s", issue.source, issue.message)
        return settings, models, log, issues

    def _migrate_user_data(self, settings: Any) -> Any:
        """Run the one-time install-tree -> data-root copy. Never fatal.

        A migration failure must not stop LOCITIZE from starting: the originals are
        untouched by construction, so the honest response is to keep running on
        the data root and report, which is what _log_migration then does.
        """
        try:
            import migration

            return migration.migrate_install_data(
                settings.base_dir,
                settings.data_dir,
                webui_port=settings.ports.openwebui,
            )
        except Exception as exc:  # noqa: BLE001 - startup must survive this
            self._out(f"data migration could not run: {exc}")
            return None

    def _log_migration(self, log: Any, outcome: Any, data_dir: str) -> None:
        """Record what the migration did, in the log and on the terminal.

        The in-app notice (desktop) is the user-facing half of DEC-M14-9 item 7;
        this is the other half, so a headless or terminal session still has a
        written record naming both real paths.

        `data_dir` is required rather than defaulted so that the marker - the
        only thing that knows whether the notice is still owed - can never be
        left unread by a call site that simply forgot the argument.
        """
        import migration

        if outcome is None or not getattr(outcome, "ran", False):
            return
        for item in outcome.items:
            # "nothing to copy" is the normal case for most items on most
            # machines; logging it would bury the lines that matter.
            if item.action == migration.ACTION_NO_SOURCE:
                continue
            log.info(
                "data migration: %s %s (%s -> %s)%s",
                item.label,
                item.action,
                item.source,
                item.destination,
                f": {item.error}" if item.error else "",
            )
        if outcome.complete and (
            outcome.copied or migration.pending_notice(data_dir)
        ):
            # The notice claims the user's data "was copied", so it is spoken
            # only when the migration is actually finished - never while an item
            # is deferred waiting for Open WebUI to stop (DEC-M14-9 rule 5).
            #
            # The second clause is round 7's HIGH-1r: the run that finally
            # clears a deferral often copies NOTHING itself (Open WebUI created
            # the destination, or the user removed the install-side original),
            # yet it is the run on which an earlier run's copy becomes final. On
            # the first clause alone this surface said nothing at all, so a
            # headless user whose transcripts really had moved was never told.
            # A machine with nothing to migrate has no marker and no owed
            # notice, so it stays silent rather than claiming a copy that never
            # happened. Since round 8 that second clause is evidence-gated too:
            # pending_notice requires the marker's audit trail to name a real
            # copy, so a marker written by a run that deferred its only item and
            # copied nothing can no longer put words in this surface's mouth.
            log.info("data migration: %s", outcome.notice)
            self._out(outcome.notice)
        if outcome.deferred:
            labels = ", ".join(item.label for item in outcome.deferred)
            message = (
                f"locitize could not copy {labels} yet because Open WebUI is "
                "running and its database must not be copied while it is open. "
                "Nothing was lost: close Open WebUI and start locitize again, and "
                "the copy will finish then."
            )
            log.info("data migration deferred: %s", message)
            self._out(message)
        if not outcome.ok:
            # Rule 5: a partial failure is announced with both real paths and a
            # next step, and is NOT marked as done.
            message = (
                "Some of your existing locitize data could not be copied into the "
                f"data folder: {'; '.join(outcome.errors)}. Nothing was deleted "
                "- the originals are still in the locitize folder. Close anything "
                "using those files and start locitize again to retry."
            )
            log.warning("data migration incomplete: %s", message)
            self._out(message)

    # ---- verify ladder ---------------------------------------------------- #

    def _run_ladder(self, settings: Any, models: Any) -> Any:
        """Run the full health probe roster and return a HealthReport."""
        from health import HealthChecker, HealthProviders

        # Tests may inject a ready-made checker or a providers bundle.
        checker = self._deps.get("health_checker")
        if checker is None:
            providers = self._deps.get("providers") or HealthProviders.defaults()
            checker = HealthChecker(settings, models.models, providers)
        return checker.run_all()

    # ---- rendering -------------------------------------------------------- #

    def _build_whisper_controller(self, settings: Any, manager: Any) -> Any:
        """Build a SingleServiceController for whisper-server on a shared manager.

        The spec is built lazily (only when start() is called), so an unset whisper
        path never breaks launcher startup -- it surfaces as an honest error only if
        the owner actually tries to start whisper. Tests may inject
        `process_factory`/`whisper_spec_builder` to drive this with a fake process.
        """
        from services import PortAllocator, SingleServiceController, make_process_factory
        from whisper import build_whisper_server_spec

        allocator = PortAllocator(
            settings.ports.range_start,
            settings.ports.range_end,
            settings.ports.allocation,
        )
        factory = self._deps.get("process_factory") or make_process_factory(allocator)
        spec_builder = self._deps.get("whisper_spec_builder") or (
            lambda: build_whisper_server_spec(
                settings, str(resolve_log_dir(settings) / "whisper_server.log")
            )
        )
        return SingleServiceController(manager, spec_builder, factory)

    def _run_gui(self, settings: Any, models: Any) -> int:
        """Launch the Tkinter command center (M4, Architecture G1-G3).

        Reuses _build_controller / _build_whisper_controller unchanged so the GUI
        drives the same ModelController / whisper controller / ServiceManager the
        terminal path uses - the existing atexit.register(stop_all) no-orphan
        guarantee therefore already covers this process, and gui_controller adds no
        new lifecycle code. The health mark shown in the footer is the startup
        ladder's overall result. Tk is imported inside gui.py; a machine with no
        display surfaces the ImportError/TclError here with a remedy rather than a
        traceback (the automated suite never touches this path).
        """
        _model_registry, manager, controller = self._build_controller(settings, models)
        whisper_controller = self._build_whisper_controller(settings, manager)
        self._service_manager = manager
        self._whisper_controller = whisper_controller
        # M10.1: the shared ModelController + registry are reused by the GUI assistant
        # session, vision-describe, and benchmark-one builders (all on the SHARED
        # manager), so they are stored for those injected callables to reach.
        self._gui_model_controller = controller
        self._gui_registry = _model_registry
        # The model router (owner request 2026-09-02) must switch on THIS
        # controller - the session-long one backed by the shared manager. Bound
        # here rather than inside _build_controller, which short-lived paths
        # (benchmark, vision, smoke-start) also call: binding there let the
        # router drift onto a throwaway controller that knows nothing about the
        # running server, and the next picker change would have started a
        # SECOND llama-server beside the first.
        self._session_controller = controller
        self._session_registry = _model_registry
        # Start it with the SESSION, not with Open WebUI. Tying it to
        # _start_openwebui alone meant nothing answered the router port until the
        # owner happened to click Chat - so a browser already open on Open WebUI,
        # or any other OpenAI-compatible client, found nothing listening.
        self._ensure_router(settings)

        import atexit

        atexit.register(manager.stop_all)

        report = self._run_ladder(settings, models)

        from gui_controller import GuiController
        from health import DefaultSystemInfoProvider, NvidiaSmiGpuInfoProvider

        gc = GuiController(
            settings,
            _model_registry,
            controller,
            whisper_controller,
            manager,
            listen_fn=lambda seconds, emit: self._gui_listen(settings, seconds, emit),
            speak_fn=lambda text, voice: self._gui_speak(settings, text, voice),
            # M9-lite: start Open WebUI on the SHARED manager (no orphan) when the
            # owner accepts the offer-start dialog; returns True when it went ready.
            openwebui_start_fn=lambda: self._start_openwebui(settings),
            # M10: the single-front-door builders. Each reuses an existing CLI path on
            # the GUI's SHARED, atexit-backstopped manager so the no-orphan guarantee
            # (M10.4) covers the assistant's model/mic/voice, the vision model, and the
            # benchmark model. gui.py never sees any of this wiring (presentation only).
            assistant_start_fn=lambda events, voice, speak: self._gui_assistant_start(
                settings, models, events, voice=voice, speaking=speak
            ),
            describe_fn=lambda path, prompt: self._gui_describe(
                settings, models, path, prompt
            ),
            memory_search_fn=lambda query: self._gui_memory_search(settings, query),
            benchmark_fn=lambda model_id: self._gui_benchmark_one(
                settings, models, model_id
            ),
            benchmark_history_fn=lambda rows: self._gui_benchmark_history(
                settings, rows
            ),
            # Owner request: host RAM/GPU specs + per-model VRAM/spillage. Same
            # providers the `health` command already uses (Architecture section 13).
            gpu_provider=NvidiaSmiGpuInfoProvider(),
            sys_provider=DefaultSystemInfoProvider(),
            apply_noise_suppression_fn=getattr(
                getattr(self, "_audio_suppressor", None), "set_mode", None
            ),
            second_eye_fn=lambda goal, interval, stop_file: self._second_eye(
                settings,
                models,
                goal,
                voice=None,
                interval_s=interval,
                diff_threshold=0.03,
                min_judge_interval_s=2.0,
                as_json=False,
                stop_file=stop_file,
            ),
        )
        try:
            import gui
        except Exception as exc:  # noqa: BLE001 - Tk/display unavailable is a clean stop
            manager.stop_all()
            self._out(
                f"[XX] could not start the GUI ({exc}); Tkinter may be unavailable. "
                f"Use locitize.bat for the terminal menu."
            )
            return 2
        try:
            return gui.run(gc, report.overall.value)
        finally:
            # Belt-and-braces: even if the window's own shutdown was bypassed, make
            # sure nothing is left running (shutdown() is idempotent via stop_all).
            manager.stop_all()

    def _run_desktop(self, settings: Any, models: Any) -> int:
        """Launch the PySide6 desktop over the existing GUI controller (M12).

        This intentionally mirrors `_run_gui`: both entry points construct the
        same controller with the same injected callables and the same shared,
        atexit-backed ServiceManager. Only the presentation module differs. Keeping
        that symmetry preserves the proven lifecycle and no-orphan behavior while
        Qt replaces Tk as the default view.

        Memory lock: Desktop runs via project .venv (PySide6) only. Probe and
        re-exec BEFORE building the ServiceManager / starting the router, so a
        missing Qt binding fails loud without starting then tearing down llama.
        """
        gate = ensure_desktop_interpreter(self._out)
        if gate is not None:
            return gate
        _model_registry, manager, controller = self._build_controller(settings, models)
        whisper_controller = self._build_whisper_controller(settings, manager)
        self._service_manager = manager
        self._whisper_controller = whisper_controller
        # The desktop reuses the same shared collaborators as the retired Tk view.
        # Assistant, vision, and benchmark work therefore remain off the GUI thread
        # and every child is still owned by the one shared manager.
        self._gui_model_controller = controller
        self._gui_registry = _model_registry
        # The model router (owner request 2026-09-02) must switch on THIS
        # controller - the session-long one backed by the shared manager. Bound
        # here rather than inside _build_controller, which short-lived paths
        # (benchmark, vision, smoke-start) also call: binding there let the
        # router drift onto a throwaway controller that knows nothing about the
        # running server, and the next picker change would have started a
        # SECOND llama-server beside the first.
        self._session_controller = controller
        self._session_registry = _model_registry
        # Start it with the SESSION, not with Open WebUI. Tying it to
        # _start_openwebui alone meant nothing answered the router port until the
        # owner happened to click Chat - so a browser already open on Open WebUI,
        # or any other OpenAI-compatible client, found nothing listening.
        self._ensure_router(settings)

        import atexit

        atexit.register(manager.stop_all)

        report = self._run_ladder(settings, models)

        from gui_controller import GuiController
        from health import DefaultSystemInfoProvider, NvidiaSmiGpuInfoProvider

        gc = GuiController(
            settings,
            _model_registry,
            controller,
            whisper_controller,
            manager,
            listen_fn=lambda seconds, emit: self._gui_listen(settings, seconds, emit),
            speak_fn=lambda text, voice: self._gui_speak(settings, text, voice),
            openwebui_start_fn=lambda: self._start_openwebui(settings),
            assistant_start_fn=lambda events, voice, speak: self._gui_assistant_start(
                settings, models, events, voice=voice, speaking=speak
            ),
            describe_fn=lambda path, prompt: self._gui_describe(
                settings, models, path, prompt
            ),
            memory_search_fn=lambda query: self._gui_memory_search(settings, query),
            benchmark_fn=lambda model_id: self._gui_benchmark_one(
                settings, models, model_id
            ),
            benchmark_history_fn=lambda rows: self._gui_benchmark_history(
                settings, rows
            ),
            # Owner request: host RAM/GPU specs + per-model VRAM/spillage. Same
            # providers the `health` command already uses (Architecture section 13).
            gpu_provider=NvidiaSmiGpuInfoProvider(),
            sys_provider=DefaultSystemInfoProvider(),
            apply_noise_suppression_fn=getattr(
                getattr(self, "_audio_suppressor", None), "set_mode", None
            ),
            second_eye_fn=lambda goal, interval, stop_file: self._second_eye(
                settings,
                models,
                goal,
                voice=None,
                interval_s=interval,
                diff_threshold=0.03,
                min_judge_interval_s=2.0,
                as_json=False,
                stop_file=stop_file,
            ),
        )
        try:
            import desktop
        except ModuleNotFoundError as exc:
            manager.stop_all()
            # Only a missing PySide6 module gets the install remedy. A missing
            # desktop helper is a code/install fault and must not be misdiagnosed.
            if exc.name == "PySide6" or (exc.name or "").startswith("PySide6."):
                self._out(
                    f"[XX] could not start locitize Desktop ({exc}); PySide6 is not "
                    f"installed. Run 'pip install -r requirements.txt', or run "
                    f"'locitize.bat --terminal' for the terminal menu."
                )
            else:
                self._out(
                    f"[XX] could not load locitize Desktop ({exc}); missing module "
                    f"'{exc.name or 'unknown'}'. Repair the locitize installation or "
                    f"run 'locitize.bat --terminal' for the terminal menu."
                )
            return 2
        except Exception as exc:  # noqa: BLE001 - clean up and report import fault
            manager.stop_all()
            self._out(
                f"[XX] could not load locitize Desktop "
                f"({type(exc).__name__}: {exc}). Repair the locitize installation or "
                f"run 'locitize.bat --terminal' for the terminal menu."
            )
            return 2
        try:
            return desktop.run(gc, report.overall.value)
        finally:
            # The close hook calls shutdown(); this backstop covers construction or
            # event-loop failures and is safe because stop_all() is idempotent.
            manager.stop_all()

    def _gui_listen(self, settings: Any, seconds: float, emit: Any) -> bool:
        """Run one whisper-stream capture window for the GUI, streaming segments.

        Builds the stream controller on the GUI's SHARED, atexit-backstopped
        ServiceManager (self._service_manager, set in _run_gui) rather than a fresh
        local one (H-2 fix). Registering whisper-stream on the shared manager means
        gui_controller.shutdown().stop_all() AND the single atexit.register(stop_all)
        backstop both tear it down, so a window close during a Listen can never
        outlive the process even if this method's own finally is skipped when its
        daemon ops thread is killed at interpreter finalize. Reuses _tail_listen so
        the GUI transcript is deduplicated and VAD-gated exactly like the terminal
        one. Returns True on a clean start + stop. Runs on gui_controller's ops
        worker, never the UI thread.
        """
        from services import ServiceStatus
        from whisper import build_whisper_stream_spec

        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "whisper_stream.log"

        # Reuse the shared manager so whisper-stream inherits the GUI's no-orphan
        # backstop (H-2). _service_manager is always set before the GUI wires
        # listen_fn; fall back to a fresh manager only if somehow unset.
        manager, controller = self._build_service_controller(
            settings,
            lambda: build_whisper_stream_spec(settings, str(log_path)),
            manager=getattr(self, "_service_manager", None),
        )
        started = False
        try:
            try:
                status = controller.start()
            except ValueError as exc:
                emit(f"cannot start whisper-stream: {exc}")
                return False
            if status is not ServiceStatus.RUNNING:
                emit(f"whisper-stream did not start ({status.value}); see logs")
                return False
            started = True
            self._tail_listen(log_path, seconds, emit=emit)
        finally:
            controller.stop()
        return started

    def _gui_speak(self, settings: Any, text: str, voice: str) -> tuple[bool, str]:
        """Synthesize+play one line for the GUI Voice panel. Returns (ok, detail).

        The Kokoro service is started ONCE on the GUI's shared, atexit-backstopped
        ServiceManager and reused for every Speak test / Audition clip, so a session
        of many clips leaves exactly zero orphan processes (stop_all tears it down on
        window close). Runs on gui_controller's ops worker, never the UI thread.
        """
        from services import ServiceStatus
        from tts import KokoroClient, build_kokoro_server_spec

        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "kokoro_server.log"

        controller = getattr(self, "_kokoro_controller", None)
        if controller is None or not controller.is_running():
            _manager, controller = self._build_service_controller(
                settings,
                lambda: build_kokoro_server_spec(settings, str(log_path)),
                manager=getattr(self, "_service_manager", None),
            )
            try:
                status = controller.start()
            except ValueError as exc:
                return False, f"cannot start Kokoro: {exc}"
            if status is not ServiceStatus.RUNNING:
                return False, "Kokoro service did not start; see logs/kokoro_server.log"
            self._kokoro_controller = controller

        port = controller.resolved_port or settings.ports.kokoro
        client = KokoroClient(port)
        try:
            # One-off menu speak: keep the throwaway wav in-tree and let speak()
            # delete it after playback (SEC-M6-1), never in the OS temp dir.
            client.speak(
                text,
                voice=voice,
                speed=settings.tts.speed,
                play=settings.tts.autoplay,
                temp_dir=str(log_dir),
            )
        except Exception as exc:  # noqa: BLE001 - report synth/play faults honestly
            return False, f"text-to-speech failed: {exc}"
        return True, "spoken"

    def _gui_assistant_start(
        self,
        settings: Any,
        models: Any,
        events: Any,
        *,
        voice: str | None = None,
        speaking: bool = True,
        model_id: str | None = None,
    ) -> Any:
        """Build the GUI assistant session on the SHARED manager; return a handle (M10.4).

        Mirrors _assistant's seam construction but wires the M7 loop to the GUI:
        (a) the mic + Kokoro are built on self._service_manager (the shared,
        atexit-backstopped manager) so window-close/End/crash all reap them (RM3);
        (b) capture is GuiSttSource over the UNCHANGED _MicSttSource +
        PushToTalkSttSource, driven by a talk gate instead of the console Enter;
        (c) KokoroTtsSink.on_speaking is wired to events.on_state so the Talk button
        shows speaking; (d) emit routes to events.on_reply and GuiSttSource fires
        events.on_user/on_state, so every turn marshals through result_q; (e)
        AssistantLoop.run() runs on a dedicated daemon SESSION thread (Thread D) so
        the UI thread never blocks. No new loop or lifecycle code is added -- it
        reuses the M7 loop, seams, HalfDuplexGate, and ModelController.

        Raises (caught by GuiController._do_start_assistant, turned into an honest
        assistant_error Result) on a model/voice/service start failure -- never a
        fabricated ready state.
        """
        import queue as _queue
        import threading
        import time
        from threading import Event

        from assistant import (
            VOICE_MODE_SYSTEM_PROMPT,
            AssistantLoop,
            ConversationState,
            HalfDuplexGate,
            KokoroTtsSink,
            PrintTtsSink,
            clean_for_speech,
        )
        from gui_controller import GuiSttSource
        from llm import LlamaCppClient
        from memory import ConversationMemory, resolve_memory_dir
        from services import ServiceStatus
        from tts import KokoroClient

        controller = self._gui_model_controller
        registry = self._gui_registry
        target_id = model_id or settings.launcher.default_model
        model = registry.get(target_id)
        if model is None or not model.location:
            raise RuntimeError(
                f"chat model '{target_id}' is not available (unknown id or empty "
                f"location); set launcher.default_model or the model's location"
            )
        # Ensure the chat model is RUNNING on the shared controller (start if none is
        # up, switch if a different one is). An honest failure raises with the reason.
        if controller.running_model_id != target_id:
            if controller.running_model_id is None:
                status = controller.start(target_id)
            else:
                status = controller.switch(target_id)
            if status is not ServiceStatus.RUNNING:
                raise RuntimeError(
                    f"chat model '{target_id}' did not start ({status.value}); see logs"
                )
        llm_port = controller.running_port or settings.ports.llama_cpp
        llm_client = LlamaCppClient(llm_port)

        # Voice OUT setup on the shared manager (unless the speak toggle is off). A
        # start failure degrades to text-only rather than aborting the whole session.
        voice_name = ""
        half_duplex_gate: Any = None
        tts_sink: Any = None
        if speaking:
            voice_name, warning = self._resolve_voice(settings, voice)
            if warning:
                self._out(f"  {warning}")
            from tts import build_kokoro_server_spec

            log_dir = resolve_log_dir(settings)
            log_dir.mkdir(parents=True, exist_ok=True)
            klog = log_dir / "kokoro_server.log"
            kcontroller = getattr(self, "_kokoro_controller", None)
            if kcontroller is None or not kcontroller.is_running():
                _m, kcontroller = self._build_service_controller(
                    settings,
                    lambda: build_kokoro_server_spec(settings, str(klog)),
                    manager=self._service_manager,
                )
                kstatus = kcontroller.start()
                if kstatus is not ServiceStatus.RUNNING:
                    raise RuntimeError(
                        "Kokoro voice service did not start; see logs/kokoro_server.log"
                    )
                self._kokoro_controller = kcontroller
            tts_port = kcontroller.resolved_port or settings.ports.kokoro
            half_duplex_gate = HalfDuplexGate(
                tail_s=settings.assistant.half_duplex_tail_s
            )
            tts_sink = KokoroTtsSink(
                KokoroClient(tts_port),
                speed=settings.tts.speed,
                speaking_gate=half_duplex_gate,
                lead_silence_ms=settings.tts.reply_lead_silence_ms,
                # M10.3: drive the Talk button's speaking state; on_speaking(False)
                # returns it to idle (GuiSttSource re-affirms idle next turn anyway).
                on_speaking=lambda active: events.on_state(
                    "speaking" if active else "idle"
                ),
            )
        if tts_sink is None:
            tts_sink = PrintTtsSink()

        interrupt = Event()
        talk_gate: "Any" = _queue.Queue()
        # The mic uses the SHARED manager (RM3): a bypassed window-close cannot orphan
        # whisper-stream because the shared manager's atexit backstop reaps it.
        mic_source = _MicSttSource(
            self,
            settings,
            interrupt,
            speaking_gate=half_duplex_gate,
            manager=self._service_manager,
        )
        stt = GuiSttSource(
            mic_source,
            talk_gate,
            on_state=events.on_state,
            on_user=events.on_user,
            max_capture_s=settings.assistant.max_capture_s,
            settle_s=settings.assistant.utterance_settle_s,
            emit=self._out,
        )

        session_id = time.strftime("%Y-%m-%dT%H%M%S")
        state = ConversationState(session_id=session_id)
        # A second leading system message breaks chat templates that only allow one
        # (e.g. Qwen3's: "System message must be at the beginning"), so the voice
        # conditioning is folded into the single system message instead of appended
        # as its own turn.
        system_prompt = settings.assistant.system_prompt
        if speaking:
            system_prompt = f"{system_prompt}\n\n{VOICE_MODE_SYSTEM_PROMPT}"
        state.append("system", system_prompt)
        memory = ConversationMemory(
            resolve_memory_dir(settings.data_dir, settings.memory.dir),
            enabled=settings.memory.enabled,
        )

        loop = AssistantLoop(
            stt,
            llm_client,
            tts_sink,
            state,
            voice=voice_name,
            speaking=speaking,
            speak_per_sentence=settings.assistant.speak_per_sentence,
            context_size=model.context_size,
            response_reserve_tokens=settings.assistant.response_reserve_tokens,
            max_history_turns=settings.assistant.max_history_turns,
            measure_stt=True,
            memory=memory,
            recall_limit=settings.memory.search_limit,
            interrupt=interrupt,
            emit=events.on_reply,
            text_filter=clean_for_speech,
            half_duplex_gate=half_duplex_gate,
        )

        session_thread = threading.Thread(
            target=loop.run, name="locitize-gui-assistant", daemon=True
        )
        session_thread.start()
        return _GuiAssistantHandle(
            stt=stt,
            loop=loop,
            tts_sink=tts_sink,
            mic_source=mic_source,
            thread=session_thread,
            port=llm_port,
        )

    def _gui_describe(
        self, settings: Any, models: Any, image_path: str, prompt: str | None
    ) -> tuple[bool, str]:
        """Describe an image via the existing --describe path on the SHARED controller.

        Reuses QwenVisionModel over the GUI's shared ModelController (M8.1): the
        controller switches to the VL model WITH --mmproj, the loopback VisionClient
        POSTs the base64 image + prompt, and the real answer is returned. The VL model
        stays on the shared manager (reaped at window close), so no orphan. Vision is
        exclusive with a Talk session (it switches the running model), enforced by
        gui_controller. Returns (ok, answer-or-remedy). Runs on the ops worker.
        """
        from vision import QwenVisionModel, VisionClient, VisionError

        if not Path(image_path).is_file():
            return False, f"image not found: {image_path}"
        controller = self._gui_model_controller
        client_factory = self._deps.get("vision_client_factory") or (
            lambda port: VisionClient(port)
        )
        # Same resolution the CLI --describe path uses: the hardcoded default id
        # matches no row in a scan-imported registry (defect 2026-09-02), so the
        # GUI Describe button was broken in exactly the same way.
        from models import ModelRegistry
        from vision import resolve_vision_model_id

        resolved, _alternatives = resolve_vision_model_id(
            ModelRegistry(models, settings), settings.vision.model
        )
        if not resolved:
            return False, (
                "no model in your list declares the vision capability - run "
                "Detect capabilities to pair projectors and tag the models "
                "that can see"
            )
        vision = QwenVisionModel(
            controller,
            default_prompt=settings.vision.prompt,
            model_id=resolved,
            client_factory=client_factory,
        )
        question = (prompt or settings.vision.prompt or "").strip()
        try:
            answer = vision.describe(image_path, question)
        except VisionError as exc:
            remedy = f" ({exc.remedy})" if exc.remedy else ""
            return False, f"{exc}{remedy}"
        return True, answer

    def _gui_memory_search(self, settings: Any, query: str) -> list[dict]:
        """Search past conversations read-only via the existing ConversationMemory (M8.3).

        A non-empty query runs the substring search; a blank query returns the most
        recent tail (recall). Read-only local JSONL scan -- no model, no network, no
        write -- so it is always available even during a Talk session. Returns a list
        of {role, text} dicts for the Memory panel. Runs on the ops worker.
        """
        from memory import ConversationMemory, resolve_memory_dir

        mem = ConversationMemory(
            resolve_memory_dir(settings.data_dir, settings.memory.dir),
            enabled=settings.memory.enabled,
        )
        q = (query or "").strip()
        limit = settings.memory.search_limit
        hits = mem.search(q, limit) if q else mem.recall(limit)
        return [{"role": h.role, "text": h.text} for h in hits]

    def _gui_benchmark_one(
        self, settings: Any, models: Any, model_id: str
    ) -> tuple[bool, str, float | None]:
        """Run the existing single-config benchmark for one model on the SHARED controller.

        Reuses the M5 BenchmarkRunner (one model, one run, no sweep) so the GUI adds
        no new benchmark code. The visible value comes directly from the successful
        result's measured predicted_per_second field; overall_score remains the
        separate deterministic quality percentage in the detailed report. Exclusive
        with a Talk session (enforced by gui_controller). Returns
        (ok, detail, generation_tok_s). Runs on the ops worker.
        """
        from benchmark import BenchmarkConflictError, BenchmarkRunner
        from health import DefaultSystemInfoProvider, NvidiaSmiGpuInfoProvider

        controller = self._gui_model_controller
        registry = self._gui_registry
        runner = BenchmarkRunner(
            controller,
            registry,
            settings,
            gpu_provider=NvidiaSmiGpuInfoProvider(),
            sys_provider=DefaultSystemInfoProvider(),
            out=self._out,
        )
        try:
            summary = runner.run([model_id], runs=1, sweep=False, resume=False)
        except BenchmarkConflictError as exc:
            return False, f"benchmark refused: {exc}", None
        ok = summary.get("failed", 1) == 0 and summary.get("ok", 0) > 0
        counts = (
            f"{summary.get('ok', 0)} ok, {summary.get('failed', 0)} failed "
            f"of {summary.get('scenarios', 0)} scenarios"
        )
        successful = [
            row
            for row in summary.get("results", [])
            if isinstance(row, dict) and row.get("ok") is True
        ]
        speed = successful[-1].get("predicted_per_second") if successful else None
        try:
            generation_tok_s = float(speed) if speed is not None else None
        except (TypeError, ValueError):
            generation_tok_s = None
        if ok and generation_tok_s is None:
            return False, f"benchmark produced no generation tok/s; {counts}", None

        if generation_tok_s is None:
            return ok, counts, None
        quality = successful[-1].get("overall_score")
        detail = f"{generation_tok_s:.1f} tok/s generation"
        if isinstance(quality, (int, float)) and not isinstance(quality, bool):
            detail += f"; quality {float(quality):.1f}%"
        return ok, f"{detail}; {counts}", generation_tok_s

    @staticmethod
    def _gui_benchmark_history(
        settings: Any, model_rows: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Load last measured generation throughput for the GUI model inventory."""
        from benchmark import latest_generation_speeds, resolve_results_dir

        path = resolve_results_dir(settings) / "benchmark_results.jsonl"
        return latest_generation_speeds(path, model_rows)

    def _render(
        self,
        settings: Any,
        models: Any,
        report: Any,
        registry: Any,
        controller: Any = None,
        whisper_controller: Any = None,
    ) -> None:
        """Print banner, system status, running services, models, applications."""
        from health import status_mark

        self._out(BANNER)
        self._out("")
        self._out("System Status")
        self._out("-------------")
        for result in report.results:
            mark = status_mark(result.status)
            line = f"  {mark} {result.name}: {result.detail}"
            self._out(line)
            if result.remedy and result.status.value != "PASS":
                self._out(f"       remedy: {result.remedy}")
        self._out(f"  Overall: {report.overall.value}")
        self._out("")

        # Honest running-service panel: shows the live model service (if any) with
        # its resolved port and PID, never a fabricated "running" line.
        if controller is not None:
            self._out("Services")
            self._out("--------")
            snap = controller.snapshot()
            if not snap["services"]:
                self._out("  (no model service started)")
            else:
                for svc in snap["services"]:
                    self._out(
                        f"  {svc['status']} {svc['model_id']} "
                        f"- port {svc['port']} pid {svc['pid']}"
                    )
            # Honest whisper (speech-to-text) status: shown only once the whisper
            # service has been started this session; otherwise reported STOPPED,
            # never a fabricated "available". Kokoro (voice OUT) is not shown here
            # because it is not installed -- the health panel reports it as a
            # WARNING and this milestone does not wire text-to-speech.
            if whisper_controller is not None:
                wsnap = whisper_controller.snapshot()
                if wsnap["services"]:
                    wsvc = wsnap["services"][0]
                    self._out(
                        f"  {wsvc['status']} whisper (speech-to-text) "
                        f"- port {wsvc['port']} pid {wsvc['pid']}"
                    )
                else:
                    self._out("  STOPPED whisper (speech-to-text) - not started")
            self._out("")

        self._out("Installed Models")
        self._out("----------------")
        installed = [m for m in models.models if m.status == "installed"]
        if not installed:
            self._out("  (none declared with status: installed)")
        for index, model in enumerate(installed, start=1):
            # Honest empty-state: an unset location is shown, not faked.
            loc = model.location if model.location else "(location not set)"
            self._out(f"  {index}. {model.name} [{model.id}] - {loc}")
        # Show declared-but-future models so nothing is hidden.
        future = [m for m in models.models if m.status == "future"]
        for model in future:
            self._out(f"  -  {model.name} [{model.id}] - future (not yet installed)")
        self._out("")

        self._out("Applications")
        self._out("------------")
        for app in registry.applications():
            state = "available" if app.available else "not available"
            self._out(f"  - {app.name} [{app.id}] - {state}")
            if not app.available and app.unavailable_reason:
                self._out(f"       {app.unavailable_reason}")
        self._out("")

    # ---- menu ------------------------------------------------------------- #

    def _menu_loop(
        self,
        settings: Any,
        models: Any,
        registry: Any,
        controller: Any,
        whisper_controller: Any,
        input_fn: Any,
    ) -> str | None:
        """Interactive menu. Reads choices until blank/q/quit.

        Model numbers start (or switch to) a real llama.cpp service via the model
        controller. 'whisper' starts/stops the whisper-server; 'listen' runs the
        live mic capture (owner voice-quality path). Application ids and the special
        actions (desktop, health, settings, docs, stop) are dispatched here.
        Returning "desktop" transfers control only after the caller's finally
        block has stopped this terminal session's shared manager.
        """
        from config import settings_to_dict

        installed = [m for m in models.models if m.status == "installed"]
        self._out("Enter a model number to start/switch to that model, an")
        self._out(
            "application id, 'desktop', 'whisper', 'listen', 'chat', 'health', "
            "'settings', 'docs', 'stop', or 'q'."
        )
        # Owner request 2026-09-02. A model number alone still starts the model
        # exactly as registered; the optional suffix picks a thinking level for
        # THIS launch only, without writing anything to models.yaml. Advertised
        # here because an unadvertised suffix is an undiscoverable feature.
        self._out(
            "Add a thinking level to start with it: '3 low', '3 off', "
            "'3 low/2048' (level/budget). Levels come from the model's own "
            "chat template."
        )
        while True:
            try:
                choice = input_fn("locitize> ").strip()
            except EOFError:
                break
            if choice in ("", "q", "quit"):
                break
            if choice in ("desktop", "gui"):
                # Return a handoff request instead of launching here. The run()
                # finally block stops the terminal manager before _run_desktop()
                # creates the Qt manager, so two owners never overlap.
                return "desktop"
            if choice == "chat":
                # M9-lite: the chat-UI chooser. Shares the one resolve_chat_choice
                # decision path with the GUI and the --chat-ui flag.
                self._dispatch_chat(settings, controller, input_fn)
                continue
            if choice == "health":
                report = self._run_ladder(settings, models)
                self._render(
                    settings, models, report, registry, controller, whisper_controller
                )
                continue
            if choice == "whisper":
                self._dispatch_whisper(whisper_controller)
                continue
            if choice == "listen":
                duration = 15.0
                self._run_listen(settings, duration, as_json=False, smoke=False)
                continue
            if choice == "settings":
                # Redacted dump (never leak env-sourced paths / future secrets).
                dump = settings_to_dict(settings, redact=True)
                for group, values in dump.items():
                    self._out(f"{group}: {values}")
                continue
            if choice == "docs":
                self._out(f"Documentation: {settings.base_dir / 'docs'}")
                continue
            if choice == "stop":
                self._dispatch_stop(controller)
                continue
            head, _, tail = choice.partition(" ")
            if head.isdigit():
                self._dispatch_model(
                    controller, installed, int(head), reasoning_text=tail
                )
                continue
            # Otherwise treat it as an application id.
            self._dispatch_app(registry, choice, settings, models)

    def _dispatch_model(
        self,
        controller: Any,
        installed: list,
        number: int,
        reasoning_text: str = "",
    ) -> None:
        """Start (or cleanly switch to) the chosen model as a real service.

        Selecting a model that is not currently running starts it; selecting a
        different model while one runs switches (stops the old, confirms it exited,
        starts the new). Failures show a remedy line, never a raw traceback.

        Owner request 2026-09-02: `reasoning_text` is the optional thinking level
        typed after the model number ("3 low", "3 off", "3 low/2048"). It is a
        ONE-LAUNCH override - models.yaml is never written here, so the next
        plain "3" starts the model at its registered setting again. An empty
        string means "use the row's own reasoning:", which is why the keyword is
        omitted entirely rather than passed as None: build_start_spec reads None
        as the deliberate "force thinking off for this scenario".
        """
        from config import parse_reasoning_choice
        from services import ServiceStatus

        if not (1 <= number <= len(installed)):
            self._out(f"  no model numbered {number}")
            return
        model = installed[number - 1]
        overrides: dict[str, Any] = {}
        if reasoning_text.strip():
            picked, problems = parse_reasoning_choice(reasoning_text, model.id)
            for problem in problems:
                self._out(f"  {problem}")
            if picked is None:
                self._out("  not starting; fix the thinking level and retry")
                return
            overrides["reasoning"] = picked
        running = controller.running_model_id
        if running and running != model.id:
            self._out(f"  switching from {running} to {model.name} ...")
        else:
            self._out(f"  starting {model.name} ...")
        if overrides:
            self._out(f"  thinking for this launch: {overrides['reasoning']}")
        try:
            status = controller.switch(model.id, **overrides)
        except ValueError as exc:
            # Honest guard: no llama.cpp path or model location set yet.
            self._out(f"  cannot start {model.name}: {exc}")
            return
        snap = controller.snapshot()
        svc = next(
            (s for s in snap["services"] if s["model_id"] == model.id), None
        )
        if status is ServiceStatus.RUNNING and svc is not None:
            self._out(
                f"  {model.name} is RUNNING on port {svc['port']} (pid {svc['pid']})"
            )
        else:
            detail = status.value if status is not None else "unknown"
            self._out(
                f"  {model.name} did not start ({detail}); "
                f"see logs/errors.log, check VRAM/port, then retry"
            )

    def _dispatch_stop(self, controller: Any) -> None:
        """Stop the currently running model service, if any."""
        from services import ServiceStatus

        running = controller.running_model_id
        if not running:
            self._out("  no model service is running")
            return
        status = controller.stop()
        if status is ServiceStatus.STOPPED:
            self._out(f"  stopped {running}")
        else:
            self._out(f"  stop of {running} reported {status.value}")

    def _active_model_port(self, controller: Any) -> int | None:
        """Return the port of the currently running llama.cpp model, or None.

        Reads the ModelController's own snapshot so the chat chooser always targets
        the real resolved port (which may have auto-incremented off 8080), never a
        guessed default. None means no model is running -> the chooser returns
        NO_MODEL and opens neither UI onto a dead backend.
        """
        running = getattr(controller, "running_model_id", None)
        if not running:
            return None
        try:
            snap = controller.snapshot()
        except Exception:  # noqa: BLE001 - honest degrade, never crash the menu
            return None
        for svc in snap.get("services", []):
            if svc.get("model_id") == running:
                return svc.get("port")
        return None

    def _dispatch_chat(self, settings: Any, controller: Any, input_fn: Any) -> None:
        """Resolve and act on the chat-UI choice for the terminal menu (M9.3).

        Runs the pure resolve_chat_choice, then renders the honest terminal
        presentation: a numbered prompt on ASK, a y/N start offer on
        OFFER_START_OPENWEBUI, and a one-line reason on every degrade. The actual
        browser open reuses _open_chat_browser. The self._chat_ui_override (from
        --chat-ui) beats the persisted preference for this one action.
        """
        from webui import (
            ChatDecision,
            resolve_chat_choice,
            webui_available,
        )

        model_port = self._active_model_port(controller)
        installed = webui_available(settings)
        ready = self._openwebui_ready(settings) if installed else False
        override = getattr(self, "_chat_ui_override", None)

        resolution = resolve_chat_choice(
            preferred=settings.chat.preferred_ui,
            cli_override=override,
            model_running=model_port is not None,
            webui_installed=installed,
            webui_ready=ready,
        )
        decision = resolution.decision

        if decision is ChatDecision.NO_MODEL:
            self._out(f"  {resolution.reason}")
            return
        if decision is ChatDecision.DEGRADE_TO_LLAMACPP:
            self._out(f"  {resolution.reason}")
            self._open_chat_browser(settings, self._llamacpp_chat_url(settings, model_port))
            return
        if decision is ChatDecision.OPEN_LLAMACPP:
            self._open_chat_browser(settings, self._llamacpp_chat_url(settings, model_port))
            return
        if decision is ChatDecision.OPEN_OPENWEBUI:
            self._open_chat_browser(settings, self._openwebui_chat_url(settings))
            return
        if decision is ChatDecision.OFFER_START_OPENWEBUI:
            # Open WebUI is the default chat: start it rather than ask, and on
            # failure say why instead of quietly opening llama.cpp's page.
            self._out(
                "  starting Open WebUI (first launch sets up its database "
                "and can take up to 5 minutes)..."
            )
            if self._start_openwebui(settings):
                self._open_chat_browser(settings, self._openwebui_chat_url(settings))
            else:
                self._out(
                    "  Open WebUI did not become ready. Check its log in the data "
                    "folder, or run with --chat-ui llamacpp for the built-in page."
                )
            return
        # ASK: present the numbered choice; the owner may remember it.
        self._prompt_chat_choice(settings, controller, model_port, input_fn)

    def _prompt_chat_choice(
        self, settings: Any, controller: Any, model_port: int | None, input_fn: Any
    ) -> None:
        """Numbered terminal prompt for the ASK case (M9.3), with 'r' to remember."""
        self._out("  Choose a chat UI:")
        self._out("    1) llama.cpp web UI (built-in, zero setup)")
        self._out("    2) Open WebUI (rich chat, history)")
        self._out("    [append r to remember, e.g. '2r']")
        raw = input_fn("  choice> ").strip().lower()
        remember = raw.endswith("r")
        pick = raw[:-1].strip() if remember else raw
        if pick == "1":
            choice = "llamacpp"
        elif pick == "2":
            choice = "openwebui"
        else:
            self._out("  no choice made")
            return
        if remember:
            self._remember_chat_ui(settings, choice)
        # Re-resolve now that a concrete UI was picked, reusing the same paths
        # (openwebui may still need starting).
        saved_override = getattr(self, "_chat_ui_override", None)
        self._chat_ui_override = choice
        try:
            self._dispatch_chat(settings, controller, input_fn)
        finally:
            self._chat_ui_override = saved_override

    def _remember_chat_ui(self, settings: Any, choice: str) -> None:
        """Persist the chat-UI choice via the targeted atomic write_chat_ui."""
        from config import write_chat_ui

        try:
            write_chat_ui(settings.data_dir, choice)
            settings.chat.preferred_ui = choice  # reflect in the live session
            self._out(f"  remembered: chat will open {choice} next time")
        except (ValueError, OSError) as exc:
            self._out(f"  could not save preference: {exc}")

    def _llamacpp_chat_url(self, settings: Any, model_port: int | None) -> str:
        """Loopback URL for the built-in llama.cpp web UI on the resolved model port."""
        port = model_port or settings.ports.llama_cpp
        return f"http://127.0.0.1:{port}/"

    def _openwebui_chat_url(self, settings: Any) -> str:
        """Loopback URL for the Open WebUI service on its reserved port."""
        return f"http://127.0.0.1:{settings.ports.openwebui}/"

    def _open_chat_browser(self, settings: Any, url: str) -> None:
        """Open `url` in the browser, or print it when chat_open_browser is false."""
        import webbrowser

        if settings.gui.chat_open_browser:
            webbrowser.open(url)
            self._out(f"  opened {url}")
        else:
            self._out(f"  open this in your browser: {url}")

    def _openwebui_ready(self, settings: Any) -> bool:
        """True if something is already answering on the Open WebUI loopback port."""
        return self._port_open("127.0.0.1", settings.ports.openwebui)

    @staticmethod
    def _describe_placement(controller: Any, model_id: str) -> str:
        """One line on the running model's GPU placement; honest when unmeasurable."""
        import gpu_ledger

        pid = None
        for svc in controller.snapshot().get("services", []):
            if svc.get("model_id") == model_id and svc.get("pid"):
                pid = svc["pid"]
        placement = gpu_ledger.placement_of(pid) if pid else None
        if placement is None:
            return f"{model_id}: GPU placement not measurable here"
        return placement.describe(model_id)

    def _ensure_router(self, settings: Any) -> bool:
        """Start the model router if enabled and not already up. True when ready.

        Owner request 2026-09-02: Open WebUI pointed straight at llama-server saw
        the ONE model it had loaded. Pointed at the router, its picker lists the
        whole registry and choosing an entry switches what LOCITIZE serves. This
        must be running BEFORE Open WebUI starts, because webui.backend_base_url
        bakes the router URL into the child process env.

        Honest degrade: any failure returns False and leaves the caller to start
        Open WebUI against whatever backend_base_url resolves to, rather than
        refusing to open a chat UI at all.
        """
        if not getattr(settings, "router", None) or not settings.router.enabled:
            return False
        if getattr(self, "_router", None) is not None:
            return True
        controller = getattr(self, "_session_controller", None)
        registry = getattr(self, "_session_registry", None)
        if controller is None or registry is None:
            self._out("  model router needs a model controller; not started")
            return False

        from logger import get_logger
        from router import ModelRouter
        from services import ServiceStatus

        def switch_fn(model_id: str):
            status = controller.switch(model_id)
            ok = status is ServiceStatus.RUNNING
            if ok:
                # Say where the new model's memory went. On Windows a model
                # that overflows the card still loads and answers /health,
                # then crawls; this line is the one place that says so
                # (owner report 2026-09-03, "when I switch models Open WebUI
                # does not chat"). To the log as well as the console: the
                # desktop runs under pythonw, where print goes nowhere.
                line = self._describe_placement(controller, model_id)
                self._out(f"  {line}")
                get_logger("launcher").info("router switch: %s", line)
            return ok, getattr(status, "value", str(status))

        # Owner request 2026-09-03: Open WebUI as the single interface, voice
        # included. whisper and Kokoro run BESIDE the model on their own ports,
        # so keeping them warm costs no VRAM the LLM wanted - the mic button
        # answers immediately instead of paying a service start per press.
        audio = self._start_speech_services(settings) if settings.router.audio else {}
        router = ModelRouter(
            port=settings.ports.router,
            registry_fn=registry.launchable,
            running_model_id_fn=lambda: controller.running_model_id,
            port_provider=lambda: controller.running_port,
            switch_fn=switch_fn,
            handoff_note=settings.router.handoff_note,
            audio=audio,
            voice_turns=settings.router.voice_turns,
        )
        try:
            router.start()
        except OSError as exc:
            self._out(f"  model router could not bind port "
                      f"{settings.ports.router}: {exc}")
            return False
        self._router = router
        self._out(f"  model router on {router.base_url()} "
                  f"({len(registry.launchable())} models listed)")
        return True

    def _start_speech_services(self, settings: Any) -> dict:
        """Start whisper + Kokoro on the shared manager; return the audio wiring.

        Both are OPTIONAL: a machine with no whisper model or no Kokoro
        checkpoint gets a router whose /v1/audio/* routes answer an honest 503,
        which is strictly better than the bare 404 Open WebUI would otherwise
        show. Neither failure blocks the router or the chat path.

        The ports are read back from the controllers per request rather than
        captured, because the PortAllocator may reassign 8091/8092 if something
        else holds them.
        """
        manager = getattr(self, "_service_manager", None)
        if manager is None:
            return {}
        from pathlib import Path as _Path

        wiring: dict = {}

        # Open WebUI sends browser recordings through the router. Keep the audio
        # processor as an injected capability so router.py owns no filesystem or
        # subprocess policy, and tests can replace it without launching FFmpeg.
        from audio_filter import AudioNoiseSuppressor

        suppressor = AudioNoiseSuppressor(
            settings.speech.noise_suppression,
            _Path(settings.data_dir) / "tmp",
        )
        self._audio_suppressor = suppressor
        wiring["filter_upload"] = suppressor.process
        wiring["noise_suppression_mode"] = lambda: suppressor.mode

        whisper_ctrl = getattr(self, "_whisper_controller", None)
        if whisper_ctrl is not None:
            try:
                whisper_ctrl.start()
                wiring["whisper_port"] = lambda: (
                    whisper_ctrl.resolved_port if whisper_ctrl.is_running() else None
                )
                self._out("  speech-to-text ready")
            except Exception as exc:  # noqa: BLE001 - honest degrade
                self._out(f"  speech-to-text not started: {exc}")

        try:
            from tts import build_kokoro_server_spec

            log_path = resolve_log_dir(settings) / "kokoro_server.log"
            _mgr, kokoro_ctrl = self._build_service_controller(
                settings,
                lambda: build_kokoro_server_spec(settings, str(log_path)),
                manager=manager,
            )
            kokoro_ctrl.start()
            wiring["kokoro_port"] = lambda: (
                kokoro_ctrl.resolved_port if kokoro_ctrl.is_running() else None
            )
            voices_dir = settings.paths.kokoro_voices
            wiring["voices"] = lambda: sorted(
                f.stem for f in _Path(voices_dir).glob("*.pt")
            ) if voices_dir else []
            wiring["default_voice"] = lambda: settings.tts.voice
            self._out("  text-to-speech ready")
        except Exception as exc:  # noqa: BLE001 - honest degrade
            self._out(f"  text-to-speech not started: {exc}")
        return wiring

    def _stop_router(self) -> None:
        """Tear the router down (idempotent), mirroring the manager stop_all."""
        router = getattr(self, "_router", None)
        if router is not None:
            router.stop()
            self._router = None

    def _start_openwebui(self, settings: Any) -> bool:
        """Start the Open WebUI service on the shared manager; True if it went ready.

        Reuses the shared ServiceManager so the atexit/finally stop_all covers Open
        WebUI too (no orphan on menu quit). If a service is already listening on the
        port, treats it as ready. Returns False (honest degrade) on any failure.
        """
        from services import ServiceStatus
        from webui import build_openwebui_spec

        # Before the child starts: its env bakes in backend_base_url, which
        # points at the router when one is enabled.
        self._ensure_router(settings)
        # And the env is only a SEED - on an already-initialized DATA_DIR the
        # row in webui.db wins (defect 2026-09-02: the picker said "No models
        # available" because the persisted row still named the llama-server
        # port). Reconciled here, while Open WebUI is stopped, because it
        # caches this config and writes it back on shutdown.
        from webui import (
            ensure_single_openwebui_processes,
            reconcile_call_audio_context,
            reconcile_call_audio_playback,
            reconcile_call_barge_in,
            reconcile_call_silence,
            reconcile_default_model,
            reconcile_frontend_version,
            reconcile_persisted_audio,
            reconcile_persisted_backend,
            reconcile_voice_interruption,
            wait_loopback_port_free,
            wait_openwebui_healthy,
            webui_venv_python,
        )

        # No automatic upgrade at start: Open WebUI is installed at the version
        # setup pins (setup_env.OPENWEBUI_VERSION); upgrading is an explicit
        # setup action, never a silent download on every launch.
        webui_venv = webui_venv_python(settings).parent.parent
        self._out(
            f"  {ensure_single_openwebui_processes(settings.ports.openwebui, venv_dir=webui_venv)}"
        )
        self._out(f"  {reconcile_persisted_backend(settings)}")
        # Same seed-vs-database trap for the mic and read-aloud buttons: the
        # audio ENGINE cannot be set by env at all in this build.
        self._out(f"  {reconcile_persisted_audio(settings)}")
        # Use reliable phone turn-taking and the end-of-speech wait the owner
        # chose rather than the bundle's fixed two seconds. Voice interruption
        # stays off because phone-speaker echo otherwise stops its own TTS and
        # submits the echo as a new prompt.
        self._out(f"  {reconcile_voice_interruption(settings)}")
        self._out(f"  {reconcile_call_silence(settings)}")
        self._out(f"  {reconcile_call_audio_context(settings)}")
        self._out(f"  {reconcile_call_barge_in(settings)}")
        self._out(f"  {reconcile_call_audio_playback(settings)}")
        # A mobile PWA can retain an edited immutable chunk when its hashed URL
        # and SvelteKit version stay unchanged. Publish the patched content's
        # cache identity only after both Call-mode rewrites have landed.
        self._out(f"  {reconcile_frontend_version(settings)}")
        # No global model default is persisted. The model picker carried by each
        # Open WebUI request is authoritative through the router.
        self._out(f"  {reconcile_default_model(settings)}")
        # Do NOT early-return on TCP ready: a dying post-kill listener races that check
        # and skips start, leaving :8096 dead (local refuse + Serve 502). Always
        # (re)start on the Serve port, then confirm /health.
        port = int(settings.ports.openwebui)
        if not wait_loopback_port_free(port, timeout_s=15.0):
            # Last-chance unlock so PortAllocator does not auto-reassign off :8096
            # (Serve still maps HTTPS -> 127.0.0.1:8096).
            self._out(
                f"  {ensure_single_openwebui_processes(port, force=True, venv_dir=webui_venv)}"
            )
            wait_loopback_port_free(port, timeout_s=10.0)
        manager = getattr(self, "_service_manager", None)
        if manager is None:
            self._out("  Open WebUI FAIL: no service manager; cannot start")
            return False
        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "openwebui.log"
        try:
            _mgr, controller = self._build_service_controller(
                settings,
                lambda: build_openwebui_spec(settings, str(log_path)),
                manager=manager,
            )
            status = controller.start()
        except Exception as exc:  # noqa: BLE001 - honest degrade, never crash
            self._out(f"  Open WebUI start failed: {exc}")
            return False
        # controller.start already awaited ready_timeout; confirm the Serve port
        # (not a reassigned one) answers /health. Clear FAIL if still down.
        confirm_s = (
            5.0
            if status is ServiceStatus.RUNNING
            else float(getattr(settings.openwebui, "ready_timeout_s", 60) or 60)
        )
        if wait_openwebui_healthy(port, timeout_s=min(confirm_s, 60.0)):
            return True
        # One fallback path some OWUI builds expose while /health is late.
        if wait_openwebui_healthy(
            port, timeout_s=5.0, path="/api/config"
        ):
            return True
        self._out(
            f"  Open WebUI FAIL: http://127.0.0.1:{port}/health still down "
            f"after upgrade/start (status={getattr(status, 'value', status)}); "
            f"see {log_path}"
        )
        return False

    def _dispatch_whisper(self, whisper_controller: Any) -> None:
        """Toggle the whisper-server service: start it if stopped, else stop it."""
        from services import ServiceStatus

        if whisper_controller.is_running():
            status = whisper_controller.stop()
            if status is ServiceStatus.STOPPED:
                self._out("  whisper-server stopped")
            else:
                self._out(f"  whisper-server stop reported {status.value}")
            return
        self._out("  starting whisper-server ...")
        try:
            status = whisper_controller.start()
        except ValueError as exc:
            # Honest guard: whisper binary or model path not configured yet.
            self._out(f"  cannot start whisper-server: {exc}")
            return
        if status is ServiceStatus.RUNNING:
            self._out(
                f"  whisper-server is RUNNING on port "
                f"{whisper_controller.resolved_port} (pid {whisper_controller.pid})"
            )
        else:
            self._out(
                f"  whisper-server did not start ({status.value}); "
                f"see logs/whisper_server.log"
            )

    def _dispatch_app(
        self, registry: Any, app_id: str, settings: Any = None, models: Any = None
    ) -> None:
        """Launch an application id, printing the milestone note if unavailable.

        The Benchmark Suite (M5.11) is special-cased: selecting it from the menu
        actually runs the benchmark suite over every launchable model (honest
        per-scenario progress output), rather than the no-op plugin hook. This is
        the interactive surface AC16 grades. The port/owner conflict policy applies:
        if a model is already being served, the run refuses with a remedy.
        """
        if app_id == "benchmark-suite" and settings is not None:
            plugin = registry.get(app_id)
            if plugin is None or not plugin.available:
                self._out(f"  {registry.unavailable_message(app_id)}")
                return
            import argparse as _argparse

            menu_args = _argparse.Namespace(
                model="all", runs=3, sweep=False, resume=False, json=False
            )
            self._out("  Running benchmark suite over all launchable models...")
            self._run_benchmark(settings, models, menu_args)
            return

        # Voice Assistant (M7.9): selecting it runs the live assistant loop -- voice
        # in (mic), voice out (Kokoro), interruptible -- its menu label finally true.
        # A `assistant_runner` dep lets tests substitute the session without a real
        # model start; the default runs the real mic+speech session.
        if app_id == "voice-assistant" and settings is not None:
            plugin = registry.get(app_id)
            if plugin is None or not plugin.available:
                self._out(f"  {registry.unavailable_message(app_id)}")
                return
            runner = self._deps.get("assistant_runner")
            if runner is not None:
                runner(settings, models)
            else:
                self._out("  Starting the locitize voice assistant...")
                self._assistant(
                    settings,
                    models,
                    text=False,
                    no_speak=False,
                    voice=None,
                    model_id=None,
                    timings=False,
                    as_json=False,
                )
            return

        # Vision (M8): selecting it from the menu asks for a local image path and
        # runs the single-image describe path (qwen2-5-vl + --mmproj). A
        # `vision_runner` dep lets tests substitute the session without a real model
        # start; the default prompts for a path and calls the real describe flow.
        if app_id == "vision" and settings is not None:
            plugin = registry.get(app_id)
            if plugin is None or not plugin.available:
                self._out(f"  {registry.unavailable_message(app_id)}")
                return
            runner = self._deps.get("vision_runner")
            if runner is not None:
                runner(settings, models)
                return
            try:
                path = input("  image path (blank to cancel)> ").strip()
            except EOFError:
                path = ""
            if not path:
                self._out("  vision canceled (no image path)")
                return
            self._describe(settings, models, path, None, as_json=False)
            return

        code = registry.launch(app_id)
        if code == 2:
            self._out(f"  unknown application '{app_id}'")
        elif code == 3:
            self._out(f"  {registry.unavailable_message(app_id)}")
        else:
            self._out(f"  {app_id} finished (code {code})")

    # ---- entry ------------------------------------------------------------ #

    def run(self, argv: list[str] | None = None) -> int:
        """Parse args and run the requested mode. Returns the process exit code."""
        args = _parse_args(argv)
        try:
            settings, models, log, issues = self._bootstrap()
        except BootstrapError as exc:
            self._out(f"[XX] {exc}")
            return 2
        self._current_settings = settings
        # M9-lite: --chat-ui overrides the persisted chat.preferred_ui for the menu
        # 'chat' action for this run (None = use the persisted preference).
        self._chat_ui_override = getattr(args, "chat_ui", None)

        # Production stack liveness (ports only; ignores RAM/VRAM ladder).
        if getattr(args, "stack_health", False):
            return self._stack_health(settings, args.json)

        # Non-interactive service inspection: honest snapshot, no GPU, no menu.
        if args.service_status:
            return self._service_status(settings, models, args.json)

        if args.egress:
            return self._egress_report(settings, args.json)
        if args.gpu:
            return self._gpu_status(settings, args.json)
        if args.gpu_free:
            return self._gpu_free(settings, args.json)
        if args.rtx_report:
            return self._rtx_report(settings)
        if args.detect_capabilities:
            return self._detect_capabilities(settings, models)
        if args.check_updates:
            return self._check_updates(settings)

        # Non-interactive real llama.cpp smoke start (starts, confirms, cleans up).
        if args.smoke_start:
            return self._smoke_start(
                settings, models, args.smoke_start, args.ctx_size, args.gpu_layers,
                args.reasoning
            )

        # Real transcription of an audio file via whisper-server (M3 centerpiece).
        if args.transcribe:
            return self._transcribe(settings, args.transcribe, args.json)

        # Real whisper-server smoke start (starts, confirms readiness, cleans up).
        if args.smoke_start_whisper:
            return self._smoke_start_whisper(settings)

        # Kokoro TTS lifecycle proof: start, ready, synthesize a wav, clean stop.
        if args.smoke_tts:
            return self._smoke_tts(settings)

        # Open WebUI lifecycle proof (M9-lite): start, /health ready, clean stop, no
        # orphan. No model download (embedding fetch disabled by the spec env).
        if args.smoke_openwebui:
            return self._smoke_openwebui(settings)

        # Speak one line via the Kokoro service (M6 centerpiece: real audio bytes).
        if args.speak is not None:
            return self._speak(settings, args.speak, args.voice, args.wav, args.json)

        # Speak the sample sentence in every on-disk voice so the owner can choose.
        if args.audition:
            return self._audition(settings, args.json)

        # whisper-stream lifecycle-only proof (start, wait, clean stop; no content).
        if args.smoke_listen:
            duration = args.duration if args.duration is not None else 3.0
            return self._run_listen(settings, duration, args.json, smoke=True)

        # Live mic transcription for the owner (AC9 manual voice-quality grade).
        if args.listen:
            duration = args.duration if args.duration is not None else 15.0
            return self._run_listen(settings, duration, args.json, smoke=False)

        # Built-in voice-assistant loop (M7). Ensures a chat model is running, then
        # runs turns (text from stdin, or mic via whisper-stream), speaking each
        # sentence through Kokoro unless --no-speak. Cleans up on exit (no orphan).
        if args.assistant:
            return self._assistant(
                settings,
                models,
                text=args.text,
                no_speak=args.no_speak,
                voice=args.voice,
                model_id=args.model,
                timings=args.timings,
                as_json=args.json,
                listen_continuous=getattr(args, "listen_continuous", False),
            )

        # Single-image vision Q&A (M8). Switches to qwen2-5-vl (+ --mmproj) via the
        # ModelController, answers a question about one image, then cleans up.
        if args.describe is not None:
            return self._describe(
                settings, models, args.describe, args.prompt, args.json,
                args.model or "",
            )

        # Continuous real-time watcher: diff-triggered vision judgment + instant
        # spoken correction, both services kept warm for the whole session.
        if args.second_eye:
            return self._second_eye(
                settings,
                models,
                goal=args.goal,
                voice=args.voice,
                interval_s=args.interval,
                diff_threshold=args.diff_threshold,
                min_judge_interval_s=args.min_judge_interval,
                as_json=args.json,
                stop_file=args.stop_file,
            )

        # M5 benchmark suite (non-interactive). Drives the real ModelController;
        # measures server /completion timings only; refuses if a model is already
        # served (port busy). Never runs a weight download.
        if args.benchmark:
            return self._run_benchmark(settings, models, args)

        # M12: Qt is the canonical desktop. The hidden Tk flag remains for exactly
        # one rollback milestone and still uses the proven legacy entry point.
        # M13 precedence rule: --terminal beats --desktop/--gui/--gui-tk, so
        # `locitize.bat --terminal` reaches the menu even though the shortcut itself
        # passes --desktop. Checked before the window branches, never after.
        if not args.terminal:
            if args.gui_tk:
                return self._run_gui(settings, models)
            if args.desktop or args.gui:
                return self._run_desktop(settings, models)

        report = self._run_ladder(settings, models)

        # --health --json: structured output for QA/CI. Never appends the journal.
        if args.health and args.json:
            self._out(report.to_json())
            return 0 if report.overall.value != "FAIL" else 1
        if args.health:
            self._render_health_table(report)
            return 0 if report.overall.value != "FAIL" else 1

        # Build the application registry and the model controller. The controller
        # and its ServiceManager are shared with the cleanup path below.
        registry = self._build_registry(settings, models)
        _model_registry, manager, controller = self._build_controller(settings, models)
        # The whisper-server controller shares the SAME ServiceManager, so one
        # stop_all() tears down both the model and the whisper service -- neither
        # can be orphaned on exit (L-1 guarantee extended to whisper).
        whisper_controller = self._build_whisper_controller(settings, manager)
        self._service_manager = manager
        self._whisper_controller = whisper_controller
        # L-1: guarantee no orphaned llama.cpp/whisper process survives launcher
        # exit or a crash mid-session. atexit covers an unexpected interpreter exit;
        # the try/finally below covers the normal menu-quit path.
        import atexit

        atexit.register(manager.stop_all)

        self._render(settings, models, report, registry, controller, whisper_controller)

        if args.no_menu:
            manager.stop_all()
            return 0

        # Decide interactive vs. non-interactive. No TTY and no flag -> JSON.
        input_fn = self._deps.get("input_fn")
        if input_fn is None:
            if not sys.stdin.isatty():
                manager.stop_all()
                self._out(report.to_json())
                return 0 if report.overall.value != "FAIL" else 1
            input_fn = input

        next_surface = None
        try:
            next_surface = self._menu_loop(
                settings, models, registry, controller, whisper_controller, input_fn
            )
        finally:
            # L-1: stop every started service on the way out (normal quit path).
            manager.stop_all()
            # Append a journal entry on graceful interactive exit only.
            if settings.launcher.auto_journal:
                self._append_journal(settings, report)
        if next_surface == "desktop":
            return self._run_desktop(settings, models)
        return 0

    def _stack_health(self, settings: Any, as_json: bool) -> int:
        """Probe managed listeners (8080/8093/8096/8091/8092) + portal :4200."""
        from health import check_stack_liveness, format_stack_health_table

        report = check_stack_liveness(settings)
        if as_json:
            self._out(report.to_json())
        else:
            self._out(format_stack_health_table(report))
        return 0 if report.overall.value != "FAIL" else 1

    def _render_health_table(self, report: Any) -> None:

        """Human-readable health table for `--health` (no JSON)."""
        from health import status_mark

        for result in report.results:
            self._out(f"{status_mark(result.status)} {result.name}: {result.detail}")
            if result.remedy and result.status.value != "PASS":
                self._out(f"    remedy: {result.remedy}")
        self._out(f"Overall: {report.overall.value}")

    def _build_registry(self, settings: Any, models: Any) -> Any:
        """Construct the plugin registry with an injected PluginContext."""
        from logger import get_logger
        from models import ModelRegistry
        from plugins import PluginContext, PluginRegistry
        from services import ServiceManager

        context = PluginContext(
            config=settings,
            logger_factory=get_logger,
            service_manager=ServiceManager(),
            model_registry=ModelRegistry(models, settings),
        )
        return PluginRegistry(context)

    def _build_controller(
        self, settings: Any, models: Any, log_path: Any = None
    ) -> tuple[Any, Any, Any]:
        """Build the (ModelRegistry, ServiceManager, ModelController) trio.

        The controller owns model start/switch/stop; the manager owns process
        supervision and stop_all cleanup. The PortAllocator reads the reserved
        loopback range and allocation policy from settings, so an occupied port
        (e.g. the owner's pre-existing 8080) is auto-reassigned within the range.
        When `log_path` is given (the --smoke-start path), each service's child
        output is routed to that file for post-mortem inspection; otherwise output
        goes to the null device.
        """
        from models import ModelRegistry
        from services import (
            ModelController,
            PortAllocator,
            ServiceManager,
            make_process_factory,
        )

        registry = ModelRegistry(models, settings)
        manager = ServiceManager()
        allocator = PortAllocator(
            settings.ports.range_start,
            settings.ports.range_end,
            settings.ports.allocation,
        )
        # Tests inject a fake process factory / spec builder so the whole
        # start/switch/stop/cleanup path is exercised without a real binary.
        factory = self._deps.get("process_factory") or make_process_factory(allocator)

        injected_builder = self._deps.get("spec_builder")
        if injected_builder is not None:
            spec_builder = injected_builder
        elif log_path is not None:
            def spec_builder(model_id, ctx=None, gpu=None, **spec_kwargs):
                # **spec_kwargs forwards the M5 benchmark sweep overrides
                # (server_args/draft_model/spec_config) when the benchmark runner
                # drives this controller; the smoke path passes none.
                spec = registry.build_start_spec(model_id, ctx, gpu, **spec_kwargs)
                spec.log_path = str(log_path)
                return spec
        else:
            spec_builder = registry.build_start_spec

        # Refuse to stack a model on top of a LOCITIZE server we do not own
        # (owner-observed 2026-09-03; see ModelController._stale_server_check).
        # gpu_ledger is the SAME detector the Offload GPU button uses, so the
        # two surfaces cannot disagree about which servers are strays.
        own_bin_dir = str(Path(settings.data_dir) / "bin")

        def stale_server_check(own_pids: set) -> list[str]:
            import gpu_ledger

            rows = gpu_ledger.mark_ours(
                gpu_ledger.parse_compute_apps(gpu_ledger.query_compute_apps()),
                bin_dir=own_bin_dir,
            )
            # is_locitize means "one of our binaries". Ours-and-tracked is
            # fine - a switch stops it first. Ours-and-UNTRACKED is the stray
            # from a crashed or force-killed session, the case this exists for.
            return [
                f"{Path(row.name).name} pid {row.pid}"
                for row in rows
                if row.is_locitize and row.pid not in own_pids
            ]

        controller = ModelController(
            manager, spec_builder, factory, stale_server_check
        )
        return registry, manager, controller

    def _egress_report(self, settings: Any, as_json: bool) -> int:
        """Show the privacy ledger (M17.1)."""
        import egress_log

        summary = egress_log.summarize(settings.data_dir)
        if as_json:
            import json

            self._out(json.dumps({
                "total": summary.total, "hosts": summary.hosts,
                "recent": summary.recent,
            }))
            return 0
        self._out(egress_log.render_line(summary))
        for r in summary.recent:
            self._out(f"  {r.get('ts', '')}  {r.get('host', '?'):24} {r.get('reason', '')}")
        return 0

    def _gpu_processes(self, settings: Any) -> list:
        """Classified GPU process list for THIS machine (may be empty)."""
        import gpu_ledger
        from pathlib import Path

        bin_dir = Path(settings.data_dir) / "bin"
        raw = gpu_ledger.parse_compute_apps(gpu_ledger.query_compute_apps())
        return gpu_ledger.mark_ours(raw, bin_dir=bin_dir)

    def _gpu_status(self, settings: Any, as_json: bool) -> int:
        """Show what is holding the GPU: VRAM figures and every process (M17.8)."""
        import gpu_ledger
        from health import NvidiaSmiGpuInfoProvider

        gpus = NvidiaSmiGpuInfoProvider().gpus()
        procs = self._gpu_processes(settings)
        if as_json:
            import json

            self._out(json.dumps({
                "vram": [
                    {"name": g.name, "total_mb": g.vram_total_mb,
                     "free_mb": g.vram_free_mb}
                    for g in (gpus or [])
                ],
                "processes": [
                    {"pid": p.pid, "name": p.name, "used_mb": p.used_mb,
                     "is_locitize": p.is_locitize}
                    for p in procs
                ],
            }))
            return 0
        if gpus:
            for g in gpus:
                used = g.vram_total_mb - g.vram_free_mb
                self._out(
                    f"{g.name}: {used:.0f} MB used of {g.vram_total_mb:.0f} MB "
                    f"({g.vram_free_mb:.0f} MB free)"
                )
        else:
            self._out("no NVIDIA GPU detected (or nvidia-smi unavailable).")
        self._out(gpu_ledger.summarize(procs))
        for p in procs:
            tag = "locitize" if p.is_locitize else "other"
            mem = f"{p.used_mb:.0f} MB" if p.used_mb is not None else "  (n/a)"
            self._out(f"  [{tag:8}] pid {p.pid:>7}  {mem:>9}  {p.short_name()}")
        if any(p.is_locitize for p in procs):
            self._out("run 'locitize.bat --gpu-free' to free locitize's own.")
        return 0

    def _gpu_free(self, settings: Any, as_json: bool) -> int:
        """Free GPU memory held by LOCITIZE's OWN servers only (M17.8).

        Terminates by pid every GPU process gpu_ledger classified as ours (a
        crashed session or a measurement probe that outlived its parent). A
        foreign app is never a candidate. Reports what it freed; freeing nothing
        because nothing of ours was running is a success, not an error.
        """
        import gpu_ledger

        procs = self._gpu_processes(settings)
        pids = gpu_ledger.freeable_pids(procs)
        results = gpu_ledger.terminate_pids(pids)
        freed = [pid for pid, ok in results.items() if ok]
        if as_json:
            import json

            self._out(json.dumps({"freed": freed, "results": results}))
            return 0
        if not pids:
            self._out("nothing to free: locitize has no servers holding the GPU.")
            return 0
        self._out(f"freed {len(freed)} of {len(pids)} locitize server(s): {freed}")
        stuck = [pid for pid, ok in results.items() if not ok]
        if stuck:
            self._out(f"could not stop (already gone, or needs elevation): {stuck}")
        return 0

    def _rtx_report(self, settings: Any) -> int:
        """Render the measured per-GPU compatibility matrix to a file (M18.2)."""
        import json
        import time
        from pathlib import Path

        import gguf_meta
        import rtx_report
        from config import Config
        from health import NvidiaSmiGpuInfoProvider

        _s, models, _issues = Config.load(settings.data_dir)
        state_path = Path(settings.data_dir) / "reports" / "ctx_ceilings.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}

        rows = []
        for model in models.models:
            try:
                weights_mb = Path(model.location).stat().st_size / rtx_report.MB
            except OSError:
                continue
            baseline = (state.get(model.id) or {}).get("baseline")
            quant = model.quantization
            if not quant:
                try:
                    header = gguf_meta.read_gguf_header(model.location)
                    quant = header.architecture
                except Exception:  # noqa: BLE001 - cosmetic only
                    quant = ""
            rows.append(
                rtx_report.ModelRow(
                    model_id=model.id,
                    name=model.name,
                    weights_mb=weights_mb,
                    context_size=int(model.context_size),
                    baseline_tok_s=float(baseline) if baseline else None,
                    quantization=quant,
                )
            )

        gpus = NvidiaSmiGpuInfoProvider().gpus()
        gpu_name, vram_total = "", None
        if gpus:
            primary = max(gpus, key=lambda g: g.vram_total_mb)
            gpu_name, vram_total = primary.name, primary.vram_total_mb

        text = rtx_report.render_markdown(
            gpu_name, vram_total, rows, time.strftime("%Y-%m-%d %H:%M")
        )
        target = Path(settings.data_dir) / "reports" / "rtx_report.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self._out(f"wrote {target} ({len(rows)} models measured on {gpu_name or 'no GPU'})")
        return 0

    def _check_updates(self, settings: Any) -> int:
        """Report whether a newer llama.cpp exists (M17.4). Never downloads."""
        import egress_log
        import setup_env

        server = (settings.paths.llama_cpp or "").strip() or setup_env.find_llama_server()
        has_nvidia, _ = setup_env.detect_nvidia()
        with egress_log.reason("update-check"):
            result = setup_env.check_llama_update(server, want_cuda=has_nvidia)
        self._out(f"llama.cpp: {result['detail']}")
        return 0

    def _detect_capabilities(self, settings: Any, models: Any) -> int:
        """Auto-fill each model's capabilities from its GGUF (M17.2)."""
        import gguf_meta
        from pathlib import Path

        from config import write_model_capabilities
        from models import ModelRegistry

        registry = ModelRegistry(models, settings)
        updated = 0
        for model in registry.launchable():
            loc = (model.location or "").strip()
            if not loc or not Path(loc).is_file():
                continue
            caps = gguf_meta.detect_capabilities(loc)
            # M18.18: before stripping "vision" for a missing projector, try to
            # PAIR one - people keep the mmproj beside the model file, and the
            # import may have predated pairing. The origin dir recorded at
            # import ("Found at ... during") is checked as well as the model's
            # current directory; a found projector is linked into the models
            # dir and written through the mmproj chokepoint.
            mmproj_now = (model.mmproj or "").strip()
            import re as _re

            import setup_env
            from config import write_model_mmproj

            _origin_match = _re.search(r"Found at (.+?) during", model.notes or "")
            _origin = _origin_match.group(1).strip() if _origin_match else ""
            if mmproj_now and _origin and Path(_origin).is_file():
                # Re-validate an existing pairing against the CURRENT rules.
                # An earlier looser stem rule once handed four text-only
                # Qwen3.8 models the Qwen3-VL projector; if today's pairing
                # would not produce this projector, unpair it (which also
                # withdraws the "vision" the projector was vouching for).
                expected = setup_env.pair_mmproj(_origin, allow_generic=True)
                if not expected or Path(expected).name != Path(mmproj_now).name:
                    try:
                        write_model_mmproj(settings.data_dir, model.id, "")
                        mmproj_now = ""
                        self._out(f"  {model.id}: unpaired projector (rules no longer match)")
                    except ValueError as exc:
                        self._out(f"  {model.id}: could not unpair - {exc}")
            if not mmproj_now:
                # Pair ONLY from the model's ORIGIN directory (recorded at
                # import) - never the shared models dir, whose single stray
                # projector once mis-paired a different family. A generically
                # named projector is trusted only when the arch already says
                # vision; a stem-MATCHED projector is proof by itself, so it
                # also adds "vision" for archs that hide it (Kimi-VL is
                # deepseek2 in its own header).
                paired = ""
                if _origin:
                    # allow_generic self-guards inside pair_mmproj (a generic
                    # projector is only accepted next to a vision-arch model),
                    # so it is always safe to permit here.
                    paired = setup_env.pair_mmproj(_origin, allow_generic=True)
                if paired:
                    placed = setup_env.place_into_models_dir(paired, Path(loc).parent)
                    try:
                        write_model_mmproj(settings.data_dir, model.id, placed)
                        mmproj_now = placed
                        if "vision" not in caps:
                            caps.append("vision")
                        self._out(f"  {model.id}: paired projector {Path(placed).name}")
                    except ValueError as exc:
                        self._out(f"  {model.id}: projector found but not written - {exc}")
            if mmproj_now and "vision" not in caps:
                # A projector on the row IS the proof of vision - stable across
                # re-runs even for archs whose header hides it (Kimi-VL says
                # deepseek2). Without this, a second run would downgrade a
                # paired model back to text-only.
                caps.append("vision")
            if "vision" in caps and not mmproj_now:
                caps = [c for c in caps if c != "vision"]
            if sorted(caps) != sorted(model.capabilities):
                try:
                    write_model_capabilities(settings.data_dir, model.id, caps)
                    updated += 1
                    self._out(f"  {model.id}: {', '.join(caps) or '(none)'}")
                except Exception as exc:  # noqa: BLE001
                    self._out(f"  {model.id}: could not write - {exc}")
        self._out(f"detected and updated {updated} model(s)")
        return 0

    def _service_status(self, settings: Any, models: Any, as_json: bool) -> int:
        """Print the managed-service snapshot and exit 0 (AC6).

        A freshly launched process manages no services, so this honestly reports
        an idle/empty state. It never spawns a process and needs no GPU.
        """
        import json

        _registry, _manager, controller = self._build_controller(settings, models)
        snapshot = controller.snapshot()
        if as_json:
            self._out(json.dumps(snapshot))
        else:
            running = snapshot["running_model"] or "(none)"
            self._out(f"Running model: {running}")
            if not snapshot["services"]:
                self._out("Services: (none started)")
            for svc in snapshot["services"]:
                self._out(
                    f"  {svc['status']} {svc['model_id']} "
                    f"- port {svc['port']} pid {svc['pid']}"
                )
        return 0

    def _smoke_start(
        self,
        settings: Any,
        models: Any,
        model_id: str,
        ctx_size: Any,
        gpu_layers: Any,
        reasoning_text: str = "",
    ) -> int:
        """Start MODEL_ID for real, confirm readiness, ALWAYS clean up (AC7).

        Emits a JSON outcome and returns exit 0 only when the model genuinely
        became ready AND shutdown left no process behind. Any early failure still
        runs the cleanup in the finally block; an orphaned process is always a
        defect, so the harness confirms the manager reports nothing running before
        claiming a clean exit.
        """
        import atexit
        import json
        import time

        from services import ServiceStatus

        result: dict[str, Any] = {
            "model_id": model_id,
            "outcome": "failed",
            "reason": "",
            "resolved_port": None,
            "pid": None,
            "elapsed_s": 0.0,
        }

        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"smoke_{_safe_filename(model_id)}.log"

        _registry, manager, controller = self._build_controller(
            settings, models, log_path=log_path
        )
        self._service_manager = manager
        # Safety net: if this harness is killed mid-load, still tear the child down.
        atexit.register(manager.stop_all)

        started = time.monotonic()
        ready = False
        clean = False
        # Set when the cleanup block below raises. Kept separate from
        # result["reason"] because on the ready-then-unclean path that field is
        # already occupied by the readiness message, and silently dropping the
        # cleanup exception is what made the 2026-08-22 pythonw/CTRL_BREAK
        # failure (see services.ManagedProcess.stop) so hard to diagnose from
        # the JSON verdict alone.
        cleanup_error = ""
        # Owner request 2026-09-02: --reasoning is a one-launch override, parsed
        # HERE rather than in the argparse type= so a bad level reports through
        # the same JSON outcome every other smoke-start failure uses, instead of
        # argparse exiting 2 with a bare usage line.
        overrides: dict[str, Any] = {}
        if (reasoning_text or "").strip():
            from config import parse_reasoning_choice

            picked, problems = parse_reasoning_choice(reasoning_text, model_id)
            if picked is None:
                result["reason"] = "; ".join(problems) or "unreadable --reasoning"
                result["elapsed_s"] = round(time.monotonic() - started, 2)
                print(json.dumps(result, indent=2))
                return 1
            overrides["reasoning"] = picked
            result["reasoning"] = picked
        try:
            try:
                status = controller.start(
                    model_id, ctx_size, gpu_layers, **overrides
                )
            except ValueError as exc:
                # Config/model guard (unknown id, missing path/location).
                result["reason"] = str(exc)
                status = None

            snapshot = controller.snapshot()
            svc = next(
                (s for s in snapshot["services"] if s["model_id"] == model_id), None
            )
            if svc is not None:
                result["resolved_port"] = svc["port"]
                result["pid"] = svc["pid"]

            if status is ServiceStatus.RUNNING:
                ready = True
                result["outcome"] = "ready"
                result["reason"] = "health endpoint confirmed ready"
                # M15.4: one real generation, so the caller learns what this
                # context actually PERFORMS like, not just that it loaded. A
                # spilled KV cache serves /health perfectly and generates 10-15x
                # slower; autotune's throughput floor reads this field. Failure
                # to measure is reported as null, never as a fabricated number.
                result["tokens_per_second"] = self._smoke_throughput(
                    result["resolved_port"]
                )
                # And where the memory actually went. A model that overflows
                # the card still loads and serves on Windows; the shared figure
                # is the measured overflow (gpu_ledger.GpuPlacement).
                result["gpu_placement"] = self._smoke_placement(result["pid"])
            elif status is not None:
                result["outcome"] = "failed"
                # D-M4-2: report the timeout that actually applied to THIS model
                # (its per-model ready_timeout_s if set, else the global default),
                # so a failure never misstates the window as 60s when a longer
                # per-model window was in force.
                model = _registry.get(model_id)
                effective_timeout = (
                    model.ready_timeout_s
                    if model is not None and model.ready_timeout_s is not None
                    else settings.services.ready_timeout_s
                )
                result["reason"] = self._smoke_failure_reason(
                    log_path, effective_timeout
                )
        finally:
            # ALWAYS clean up, whatever happened above.
            try:
                manager.stop_all()
                clean = self._confirm_clean(manager)
            except BaseException as exc:  # noqa: BLE001 - cleanup must report, not raise
                clean = False
                cleanup_error = f"{type(exc).__name__}: {exc}"
                if not result["reason"]:
                    result["reason"] = f"cleanup error: {cleanup_error}"

        # Bug fixed 2026-08-22 (found via a real autotune run: the printed verdict
        # said outcome=ready/reason="health endpoint confirmed ready" while the
        # exit code was 1, because `ready` was set the moment /health answered and
        # never revisited once `clean` came back False in the finally block above -
        # the JSON lied about which half of "ready and clean" actually failed. The
        # exit code already required both; the JSON must say so too, with the
        # honest reason, or a caller parsing this JSON (autotune.py's
        # run_smoke_trial, or anyone else) has no way to tell a shutdown failure
        # from a genuine ready success.
        if ready and not clean:
            result["outcome"] = "failed"
            result["reason"] = (
                "model became ready but did not shut down cleanly afterward "
                "(a process may still be running)"
            )
            if cleanup_error:
                # Name the exception that broke the shutdown. Without this the
                # verdict says only "did not shut down cleanly", which reads
                # like an OOM-flavoured VRAM problem and sent the 2026-08-22
                # investigation down entirely the wrong road.
                result["reason"] += f"; cleanup raised {cleanup_error}"

        result["elapsed_s"] = round(time.monotonic() - started, 2)
        self._out(json.dumps(result))
        # Exit 0 only when the model became ready and no process was left behind.
        return 0 if (ready and clean) else 1

    def _run_benchmark(self, settings: Any, models: Any, args: Any) -> int:
        """Run the M5 benchmark suite non-interactively (AC14 CLI path).

        Builds the real ModelController (with a benchmark child-log so a failed load
        is inspectable), constructs the BenchmarkRunner with the real GPU/RAM
        providers, and runs every requested scenario. All lifecycle goes through the
        controller (no orphan; the atexit backstop covers a crash). Refuses cleanly
        (exit 1, honest remedy) if a model is already served on the reserved port.
        Emits a JSON summary with --json, else a human summary.
        """
        import atexit
        import json

        from benchmark import BenchmarkConflictError, BenchmarkRunner
        from health import DefaultSystemInfoProvider, NvidiaSmiGpuInfoProvider

        log_path = resolve_log_dir(settings) / "benchmark_llama.log"
        registry, manager, controller = self._build_controller(
            settings, models, log_path=log_path
        )
        self._service_manager = manager
        atexit.register(manager.stop_all)

        # Which models: an explicit --model, or every launchable model.
        if args.model and args.model != "all":
            model_ids = [args.model]
        else:
            model_ids = [m.id for m in registry.launchable()]

        runner = BenchmarkRunner(
            controller,
            registry,
            settings,
            gpu_provider=NvidiaSmiGpuInfoProvider(),
            sys_provider=DefaultSystemInfoProvider(),
            out=self._out,
        )

        summary: dict[str, Any] = {}
        exit_code = 0
        try:
            summary = runner.run(
                model_ids,
                runs=args.runs,
                sweep=args.sweep,
                resume=args.resume,
            )
        except BenchmarkConflictError as exc:
            # Honest refusal with a remedy; nothing was started.
            self._out(f"[!!] benchmark refused: {exc}")
            summary = {"refused": str(exc)}
            exit_code = 1
        finally:
            manager.stop_all()
            if not self._confirm_clean(manager):
                self._out(
                    "[XX] warning: a benchmark process may still be running - run "
                    "'locitize.bat --gpu' to see it, or --gpu-free to clear it."
                )
                exit_code = 1

        if args.json:
            self._out(json.dumps(summary))
        elif "refused" not in summary:
            self._out(
                f"Benchmark session {summary.get('session_id')}: "
                f"{summary.get('ok', 0)} ok, {summary.get('failed', 0)} failed "
                f"of {summary.get('scenarios', 0)} scenarios. "
                f"Results appended to docs/benchmark_results.jsonl / .md."
            )
        return exit_code

    def _smoke_failure_reason(self, log_path: Any, timeout_s: float) -> str:
        """Build a specific failure reason from the child's log tail (not swallowed).

        Reads the last chunk of the service log so the JSON reason cites what
        llama.cpp actually reported (e.g. an out-of-memory or model-load error)
        rather than a generic message.
        """
        detail = ""
        try:
            data = Path(log_path).read_bytes()
            tail = data[-1500:].decode("utf-8", errors="replace").strip()
            # Keep the last non-empty line, collapsed to ASCII-safe single line.
            lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            if lines:
                detail = lines[-1][:300]
        except OSError:
            detail = ""
        base = f"did not become ready within {timeout_s:.0f}s"
        return f"{base}; last log: {detail}" if detail else base

    def _confirm_clean(self, manager: Any) -> bool:
        """True only if the manager reports no service still running/erroring.

        After stop_all, every service must be STOPPED. A STOPPED_ERROR means a
        process could not be confirmed gone -- that is NOT a clean exit.
        """
        from services import ServiceStatus

        statuses = manager.monitor().values()
        return all(s is ServiceStatus.STOPPED for s in statuses)

    # ---- whisper / voice pipeline (Milestone 3) --------------------------- #

    def _build_service_controller(
        self, settings: Any, spec_builder: Any, manager: Any = None
    ) -> tuple[Any, Any]:
        """Build a (ServiceManager, SingleServiceController) for one service.

        Used by the whisper-server / whisper-stream harness paths. The controller
        reuses the exact ManagedProcess start/stop/port machinery M2 proved for
        llama.cpp. Pass `manager` to reuse an existing (already atexit-backstopped)
        ServiceManager -- the GUI listen path does this so whisper-stream registers
        on the SHARED manager and is covered by shutdown().stop_all() and the one
        atexit backstop (H-2 fix). When `manager` is None a fresh manager is built;
        that manager is NOT atexit-registered here, so a caller relying on the
        backstop for a fresh manager must register it (the terminal --smoke-listen
        path does). Tests inject `process_factory` to run the whole lifecycle with a
        fake process; production builds a real PortAllocator-backed factory.
        """
        from services import (
            PortAllocator,
            ServiceManager,
            SingleServiceController,
            make_process_factory,
        )

        if manager is None:
            manager = ServiceManager()
        allocator = PortAllocator(
            settings.ports.range_start,
            settings.ports.range_end,
            settings.ports.allocation,
        )
        factory = self._deps.get("process_factory") or make_process_factory(allocator)
        controller = SingleServiceController(manager, spec_builder, factory)
        return manager, controller

    def _port_open(self, host: str, port: int) -> bool:
        """True if something is already listening on host:port (loopback TCP probe).

        Used by --transcribe to decide whether a whisper-server is already running
        (leave it alone) or must be started for this one request (stop it after).
        """
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                return False

    def _transcribe(self, settings: Any, audio_file: str, as_json: bool) -> int:
        """Transcribe an audio file via whisper-server; always restore prior state.

        If a whisper-server is already listening on the reserved port, it is used
        and left running. Otherwise this starts one, uses it, and stops it again on
        the way out -- either way no orphan process is left behind. Emits a JSON
        record and returns 0 only when a transcript came back.
        """
        import json
        import time

        from services import ServiceStatus
        from whisper import build_whisper_server_spec, transcribe_file

        host = "127.0.0.1"
        port = settings.ports.whisper
        result: dict[str, Any] = {
            "audio_file": audio_file,
            "transcript": "",
            "outcome": "failed",
            "reason": "",
            "elapsed_s": 0.0,
        }
        started_clock = time.monotonic()

        if not Path(audio_file).is_file():
            result["reason"] = f"audio file not found: {audio_file}"
            return self._emit_result(result, as_json)

        manager = None
        controller = None
        started_by_us = False
        resolved_port = port
        try:
            if self._port_open(host, port):
                # A whisper-server (or the owner's interactive one) is already up;
                # reuse it and leave it running.
                resolved_port = port
            else:
                log_path = resolve_log_dir(settings) / "whisper_server.log"
                (resolve_log_dir(settings)).mkdir(parents=True, exist_ok=True)
                manager, controller = self._build_service_controller(
                    settings,
                    lambda: build_whisper_server_spec(settings, str(log_path)),
                )
                import atexit

                atexit.register(manager.stop_all)
                try:
                    status = controller.start()
                except ValueError as exc:
                    result["reason"] = str(exc)
                    return self._emit_result(result, as_json)
                if status is not ServiceStatus.RUNNING:
                    result["reason"] = (
                        f"whisper-server did not start ({status.value}); "
                        f"see logs/whisper_server.log"
                    )
                    return self._emit_result(result, as_json)
                started_by_us = True
                resolved_port = controller.resolved_port or port

            try:
                text = transcribe_file(audio_file, resolved_port, host)
            except Exception as exc:  # noqa: BLE001 - report transport/parse errors
                result["reason"] = f"transcription request failed: {exc}"
                return self._emit_result(result, as_json, elapsed_from=started_clock)
            result["transcript"] = text
            result["outcome"] = "ok"
            result["reason"] = "transcribed by whisper-server"
        finally:
            # Only stop what THIS command started; an already-running server stays.
            if started_by_us and controller is not None:
                controller.stop()
        return self._emit_result(result, as_json, elapsed_from=started_clock)

    @staticmethod
    def _smoke_placement(pid: Any) -> dict[str, Any] | None:
        """{"dedicated_mb", "shared_mb", "spilled"} for the served pid, or None.

        None means "could not measure" (not Windows, no counters, pid gone),
        never a fabricated zero.
        """
        import gpu_ledger

        placement = gpu_ledger.placement_of(pid)
        if placement is None:
            return None
        return {
            "dedicated_mb": placement.dedicated_mb,
            "shared_mb": placement.shared_mb,
            "spilled": placement.spilled,
        }

    @staticmethod
    def _smoke_throughput(port: Any) -> float | None:
        """One warm-up plus one timed /completion; tok/s or an honest None.

        Deliberately tiny (96 tokens) and deterministic. On a healthy config
        this adds ~2s to a smoke start; on a spilled one ~15s - and that slow
        answer is exactly the information the caller is paying for.
        """
        import json as _json
        import urllib.request

        if not port:
            return None
        url = f"http://127.0.0.1:{int(port)}/completion"
        payload = _json.dumps(
            {
                "prompt": "List the planets of the solar system in order.",
                "n_predict": 96,
                "temperature": 0,
                "seed": 42,
            }
        ).encode("utf-8")

        def once() -> float | None:
            request = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                body = _json.loads(response.read().decode("utf-8", "replace"))
            timings = body.get("timings") or {}
            predicted_n = timings.get("predicted_n")
            value = timings.get("predicted_per_second")
            # Same one-token guard as the benchmark: an immediate EOS reports a
            # sentinel-like 1,000,000 tok/s that is not generation throughput.
            if (
                isinstance(predicted_n, int)
                and predicted_n > 1
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                return float(value)
            return None

        try:
            once()  # warm-up, discarded
            return once()
        except Exception:  # noqa: BLE001 - a probe failure is a null, not a crash
            return None

    def _smoke_start_whisper(self, settings: Any) -> int:
        """Start the real whisper-server, confirm readiness, ALWAYS clean up (AC4).

        Emits a JSON outcome and returns 0 only when the server became ready AND
        shutdown left no process behind. Mirrors the llama.cpp --smoke-start
        contract; readiness here is the TCP port-connect probe (whisper-server has
        no /health route).
        """
        import atexit
        import json
        import time

        from services import ServiceStatus
        from whisper import build_whisper_server_spec

        result: dict[str, Any] = {
            "service": "whisper_server",
            "outcome": "failed",
            "reason": "",
            "resolved_port": None,
            "pid": None,
            "elapsed_s": 0.0,
        }
        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "smoke_whisper_server.log"

        manager, controller = self._build_service_controller(
            settings, lambda: build_whisper_server_spec(settings, str(log_path))
        )
        atexit.register(manager.stop_all)

        started_clock = time.monotonic()
        ready = False
        clean = False
        try:
            try:
                status = controller.start()
            except ValueError as exc:
                result["reason"] = str(exc)
                status = None
            result["resolved_port"] = controller.resolved_port
            result["pid"] = controller.pid
            if status is ServiceStatus.RUNNING:
                ready = True
                result["outcome"] = "ready"
                result["reason"] = "port readiness confirmed"
            elif status is not None:
                result["reason"] = self._smoke_failure_reason(
                    log_path, settings.services.ready_timeout_s
                )
        finally:
            try:
                manager.stop_all()
                clean = self._confirm_clean(manager)
            except Exception as exc:  # noqa: BLE001 - cleanup must report, not raise
                clean = False
                if not result["reason"]:
                    result["reason"] = f"cleanup error: {exc}"

        # Same fix as _smoke_start (2026-08-22): the exit code already requires
        # both ready and clean; the printed JSON must agree, or a stale
        # outcome=ready/reason="port readiness confirmed" verdict survives a
        # shutdown that actually failed.
        if ready and not clean:
            result["outcome"] = "failed"
            result["reason"] = (
                "whisper-server became ready but did not shut down cleanly "
                "afterward (a process may still be running)"
            )

        result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
        self._out(json.dumps(result))
        return 0 if (ready and clean) else 1

    # ---- Kokoro text-to-speech / voice OUT (Milestone 6) ------------------ #

    def _resolve_voice(self, settings: Any, requested: str | None) -> tuple[str, str]:
        """Pick the voice to speak, degrading an unknown request to the default.

        Returns (voice, warning). An unknown requested voice is NOT an error: it
        degrades to settings.tts.voice with a warning line (Architecture M6.4), so a
        typo never crashes the speak path.
        """
        from tts import list_voices

        voices = list_voices(settings.paths.kokoro_voices)
        default = settings.tts.voice
        if requested and voices and requested not in voices:
            return default, f"unknown voice '{requested}'; using default '{default}'"
        return (requested or default), ""

    def _smoke_tts(self, settings: Any) -> int:
        """Start the real kokoro TTS service, synthesize a wav, ALWAYS clean up (AC12).

        Proves the full start -> /health readiness -> real synthesis -> clean stop
        lifecycle and emits a JSON outcome carrying the produced wav path and byte
        count, so QA has a non-interactive proof that real audio bytes were produced
        and no orphan survived. Returns 0 only when the service became ready, a
        non-empty wav was produced, AND shutdown left no process behind.
        """
        import atexit
        import json
        import time

        from services import ServiceStatus
        from tts import KokoroClient, build_kokoro_server_spec

        result: dict[str, Any] = {
            "service": "kokoro_server",
            "outcome": "failed",
            "reason": "",
            "resolved_port": None,
            "pid": None,
            "wav_path": None,
            "wav_bytes": 0,
            "voice": settings.tts.voice,
            "elapsed_s": 0.0,
        }
        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "smoke_kokoro_server.log"
        wav_path = log_dir / "smoke_tts.wav"

        try:
            manager, controller = self._build_service_controller(
                settings, lambda: build_kokoro_server_spec(settings, str(log_path))
            )
        except Exception as exc:  # noqa: BLE001 - config guard surfaces as JSON
            result["reason"] = str(exc)
            self._out(json.dumps(result))
            return 1
        atexit.register(manager.stop_all)

        started_clock = time.monotonic()
        ready = False
        synthesized = False
        clean = False
        try:
            try:
                status = controller.start()
            except ValueError as exc:
                result["reason"] = str(exc)
                status = None
            result["resolved_port"] = controller.resolved_port
            result["pid"] = controller.pid
            if status is ServiceStatus.RUNNING:
                ready = True
                port = controller.resolved_port or settings.ports.kokoro
                # Real synthesis of the sample sentence via the running service; no
                # playback (headless proof), just prove non-empty wav bytes.
                client = KokoroClient(port)
                try:
                    client.speak(
                        settings.tts.sample_sentence,
                        voice=settings.tts.voice,
                        speed=settings.tts.speed,
                        out_path=str(wav_path),
                        play=False,
                    )
                    size = wav_path.stat().st_size if wav_path.exists() else 0
                    result["wav_path"] = str(wav_path)
                    result["wav_bytes"] = size
                    synthesized = size > 0
                    if synthesized:
                        result["outcome"] = "ready"
                        result["reason"] = "synthesized sample sentence"
                    else:
                        result["reason"] = "service ready but produced an empty wav"
                except Exception as exc:  # noqa: BLE001 - report synth faults honestly
                    result["reason"] = f"synthesis failed: {exc}"
            elif status is not None:
                result["reason"] = self._smoke_failure_reason(
                    log_path, settings.services.ready_timeout_s
                )
        finally:
            try:
                manager.stop_all()
                clean = self._confirm_clean(manager)
            except Exception as exc:  # noqa: BLE001 - cleanup must report, not raise
                clean = False
                if not result["reason"]:
                    result["reason"] = f"cleanup error: {exc}"

        # Same fix as _smoke_start (2026-08-22): the exit code already requires
        # ready, synthesized, AND clean; the printed JSON must agree, or a stale
        # outcome=ready/reason="synthesized sample sentence" verdict survives a
        # shutdown that actually failed.
        if ready and synthesized and not clean:
            result["outcome"] = "failed"
            result["reason"] = (
                "kokoro synthesized the sample sentence but did not shut down "
                "cleanly afterward (a process may still be running)"
            )

        result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
        self._out(json.dumps(result))
        return 0 if (ready and synthesized and clean) else 1

    def _smoke_openwebui(self, settings: Any) -> int:
        """Start the real Open WebUI service, confirm /health, ALWAYS clean up (AC19).

        Proves the full start -> /health readiness (within openwebui.ready_timeout_s,
        which covers the slow first-run database migration) -> clean stop with NO
        orphan lifecycle for Open WebUI, reusing the exact SingleServiceController /
        ServiceManager machinery that runs llama.cpp/whisper/kokoro. Emits a JSON
        outcome and returns 0 only when the service became ready AND shutdown left no
        process behind (the caller cross-checks tasklist for open-webui/uvicorn).

        The first-run embedding-model network fetch stays DISABLED via the spec env
        (openwebui.disable_embedding_fetch, default true), so this proof performs no
        model download.
        """
        import atexit
        import json
        import time

        from services import ServiceStatus
        from webui import build_openwebui_spec

        result: dict[str, Any] = {
            "service": "openwebui",
            "outcome": "failed",
            "reason": "",
            "resolved_port": None,
            "pid": None,
            "ready_timeout_s": settings.openwebui.ready_timeout_s,
            "elapsed_s": 0.0,
        }
        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "smoke_openwebui.log"

        try:
            manager, controller = self._build_service_controller(
                settings, lambda: build_openwebui_spec(settings, str(log_path))
            )
        except Exception as exc:  # noqa: BLE001 - config/install guard as JSON
            result["reason"] = str(exc)
            self._out(json.dumps(result))
            return 1
        atexit.register(manager.stop_all)

        started_clock = time.monotonic()
        ready = False
        clean = False
        try:
            try:
                status = controller.start()
            except ValueError as exc:
                result["reason"] = str(exc)
                status = None
            result["resolved_port"] = controller.resolved_port
            result["pid"] = controller.pid
            if status is ServiceStatus.RUNNING:
                ready = True
                result["outcome"] = "ready"
                result["reason"] = "open-webui reported /health 200"
            elif status is not None:
                result["reason"] = self._smoke_failure_reason(
                    log_path, settings.openwebui.ready_timeout_s
                )
        finally:
            try:
                manager.stop_all()
                clean = self._confirm_clean(manager)
            except Exception as exc:  # noqa: BLE001 - cleanup must report, not raise
                clean = False
                if not result["reason"]:
                    result["reason"] = f"cleanup error: {exc}"

        # Same fix as _smoke_start (2026-08-22): the exit code already requires
        # both ready and clean; the printed JSON must agree, or a stale
        # outcome=ready/reason="open-webui reported /health 200" verdict
        # survives a shutdown that actually failed.
        if ready and not clean:
            result["outcome"] = "failed"
            result["reason"] = (
                "open-webui became ready but did not shut down cleanly afterward "
                "(a process may still be running)"
            )

        result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
        self._out(json.dumps(result))
        return 0 if (ready and clean) else 1

    def _tts_session(self, settings: Any) -> tuple[Any, Any, bool, int]:
        """Ensure a kokoro service is running; return (manager, controller, started, port).

        Reuses an already-listening server on the reserved port (leaving it up) or
        starts one for this command. `started` says whether THIS call started it, so
        the caller stops only what it started -- the same no-orphan discipline as
        --transcribe. Raises on a config/start failure so the caller reports it.
        """
        import atexit

        from services import ServiceStatus
        from tts import build_kokoro_server_spec

        host = "127.0.0.1"
        port = settings.ports.kokoro
        if self._port_open(host, port):
            return None, None, False, port

        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "kokoro_server.log"
        manager, controller = self._build_service_controller(
            settings, lambda: build_kokoro_server_spec(settings, str(log_path))
        )
        atexit.register(manager.stop_all)
        status = controller.start()
        if status is not ServiceStatus.RUNNING:
            controller.stop()
            reason = self._smoke_failure_reason(
                log_path, settings.services.ready_timeout_s
            )
            raise RuntimeError(f"kokoro service did not start ({status.value}); {reason}")
        return manager, controller, True, (controller.resolved_port or port)

    def _speak(
        self,
        settings: Any,
        text: str,
        voice: str | None,
        wav: str | None,
        as_json: bool,
    ) -> int:
        """Synthesize one line via the Kokoro service and play it (M6, AC13).

        Starts the service if needed, synthesizes to a wav (default logs/tts_speak.wav
        so the verify script can find it), plays it when tts.autoplay is set, then
        restores the prior state. Emits a JSON record with the produced wav path and
        byte count. Returns 0 only when a non-empty wav was produced.
        """
        import json
        import time

        from tts import KokoroClient, TtsUnavailableError

        voice_name, warning = self._resolve_voice(settings, voice)
        out_path = wav or str(resolve_log_dir(settings) / "tts_speak.wav")
        resolve_log_dir(settings).mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {
            "text": text,
            "voice": voice_name,
            "wav_path": out_path,
            "wav_bytes": 0,
            "outcome": "failed",
            "reason": warning,
            "elapsed_s": 0.0,
        }
        started_clock = time.monotonic()

        manager = None
        controller = None
        started_by_us = False
        try:
            try:
                manager, controller, started_by_us, port = self._tts_session(settings)
            except Exception as exc:  # noqa: BLE001 - start/config failure -> honest JSON
                result["reason"] = str(exc)
                result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
                self._emit_tts(result, as_json)
                return 1
            client = KokoroClient(port)
            try:
                client.speak(
                    text,
                    voice=voice_name,
                    speed=settings.tts.speed,
                    out_path=out_path,
                    play=settings.tts.autoplay,
                )
            except TtsUnavailableError as exc:
                result["reason"] = f"text-to-speech is unavailable: {exc.remedy or exc}"
                result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
                self._emit_tts(result, as_json)
                return 1
            size = Path(out_path).stat().st_size if Path(out_path).exists() else 0
            result["wav_bytes"] = size
            if size > 0:
                result["outcome"] = "ok"
                result["reason"] = (warning + " spoken" if warning else "spoken").strip()
        finally:
            if started_by_us and controller is not None:
                controller.stop()
        result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
        self._emit_tts(result, as_json)
        return 0 if result["outcome"] == "ok" else 1

    def _audition(self, settings: Any, as_json: bool) -> int:
        """Speak the sample sentence in every on-disk voice so the owner can choose.

        Starts the service if needed, walks the on-disk voice list (printing which
        voice is speaking), then restores prior state. Returns 0 when at least one
        voice was auditioned.
        """
        import json
        import time

        from tts import KokoroClient, audition, list_voices

        voices = list_voices(settings.paths.kokoro_voices)
        result: dict[str, Any] = {
            "voices": voices,
            "count": len(voices),
            "outcome": "failed",
            "reason": "",
            "elapsed_s": 0.0,
        }
        started_clock = time.monotonic()
        if not voices:
            result["reason"] = (
                "no voices found; set paths.kokoro_voices to the dir of voice .pt files"
            )
            result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
            self._emit_tts(result, as_json)
            return 1

        manager = None
        controller = None
        started_by_us = False
        try:
            try:
                manager, controller, started_by_us, port = self._tts_session(settings)
            except Exception as exc:  # noqa: BLE001 - honest JSON on start failure
                result["reason"] = str(exc)
                result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
                self._emit_tts(result, as_json)
                return 1
            # autoplay must be on for an audition to be useful; force play here so a
            # headless-default autoplay=false still lets the owner hear each voice.
            client = KokoroClient(port, player=None)
            audition(
                client,
                voices,
                sentence=settings.tts.sample_sentence,
                emit=self._out,
                # Keep each per-voice throwaway wav in-tree and cleaned up (SEC-M6-1).
                temp_dir=str(resolve_log_dir(settings)),
            )
            result["outcome"] = "ok"
            result["reason"] = f"auditioned {len(voices)} voices"
        finally:
            if started_by_us and controller is not None:
                controller.stop()
        result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
        self._emit_tts(result, as_json)
        return 0 if result["outcome"] == "ok" else 1

    def _emit_tts(self, result: dict, as_json: bool) -> None:
        """Print a TTS outcome record (JSON or a short human summary)."""
        import json

        if as_json:
            self._out(json.dumps(result))
            return
        self._out(f"outcome: {result['outcome']}")
        if result.get("wav_path"):
            self._out(f"wav: {result['wav_path']} ({result.get('wav_bytes', 0)} bytes)")
        if result.get("reason"):
            self._out(f"reason: {result['reason']}")

    def _describe(
        self,
        settings: Any,
        models: Any,
        image_path: str,
        prompt: str | None,
        as_json: bool,
        model_id: str = "",
    ) -> int:
        """Single-image vision Q&A against the real qwen2-5-vl + mmproj (M8.1, AC15).

        Builds the ModelController, constructs a QwenVisionModel over it, and calls
        describe(): the controller switches to qwen2-5-vl WITH --mmproj (its
        models.yaml row carries the projector path), the client POSTs the base64
        image + prompt to /v1/chat/completions, and the real answer is returned.
        On exit it stops whatever model it started so no llama-server is orphaned
        (the M8.2 one-model-at-a-time discipline). Returns 0 only on a real answer.
        """
        import atexit
        import json
        import time

        from vision import (
            QwenVisionModel,
            VisionClient,
            VisionError,
            resolve_vision_model_id,
        )

        # Defect fixed 2026-09-02: this used the hardcoded VISION_MODEL_ID as the
        # only candidate, so --describe could not find a vision model on a
        # scan-imported registry (whose rows are named after the file). The id is
        # now RESOLVED: --model, else settings.vision.model, else the rows that
        # declare the vision capability. Resolution happens before the image check
        # so an unresolvable id is reported even for a bad path.
        from models import ModelRegistry

        registry = ModelRegistry(models, settings)
        preferred = (model_id or settings.vision.model or "").strip()
        resolved, alternatives = resolve_vision_model_id(registry, preferred)

        question = (prompt or settings.vision.prompt or "").strip()
        result: dict[str, Any] = {
            "image": image_path,
            "prompt": question,
            "model": resolved,
            "answer": "",
            "outcome": "failed",
            "reason": "",
            "elapsed_s": 0.0,
        }
        started_clock = time.monotonic()

        if not resolved:
            result["reason"] = (
                f"no vision model named {preferred!r} in your model list"
                if preferred
                else "no model in your list declares the vision capability"
            )
            if alternatives:
                result["available"] = alternatives
                result["reason"] += f"; available: {', '.join(alternatives)}"
            else:
                result["reason"] += (
                    "; run --detect-capabilities to pair projectors and tag "
                    "the models that can see"
                )
            result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
            self._emit_vision(result, as_json)
            return 1
        if alternatives:
            result["available"] = alternatives

        # Fail fast on a missing image before spinning up a model (no needless VRAM).
        if not Path(image_path).is_file():
            result["reason"] = f"image not found: {image_path}"
            result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
            self._emit_vision(result, as_json)
            return 1

        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        vision_log = log_dir / "vision_llama.log"
        _registry, manager, controller = self._build_controller(
            settings, models, log_path=vision_log
        )
        atexit.register(manager.stop_all)
        # Inject the real loopback VisionClient factory (bound to the resolved port);
        # a test double can be injected via self._deps for headless coverage.
        client_factory = self._deps.get("vision_client_factory") or (
            lambda port: VisionClient(port)
        )
        vision = QwenVisionModel(
            controller,
            default_prompt=settings.vision.prompt,
            model_id=resolved,
            client_factory=client_factory,
        )
        try:
            answer = vision.describe(image_path, question)
            result["answer"] = answer
            result["prompt"] = question or settings.vision.prompt
            result["outcome"] = "ok"
            result["reason"] = "real model answer"
        except VisionError as exc:
            remedy = f" ({exc.remedy})" if exc.remedy else ""
            result["reason"] = f"{exc}{remedy}"
        finally:
            # Stop whatever the describe path started; no orphaned llama-server.
            manager.stop_all()
        result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
        self._emit_vision(result, as_json)
        return 0 if result["outcome"] == "ok" else 1

    def _emit_vision(self, result: dict, as_json: bool) -> None:
        """Print a vision-describe outcome record (JSON or a short human summary)."""
        import json

        if as_json:
            self._out(json.dumps(result))
            return
        self._out(f"outcome: {result['outcome']}")
        if result.get("answer"):
            self._out(f"answer: {result['answer']}")
        if result.get("reason") and result["outcome"] != "ok":
            self._out(f"reason: {result['reason']}")

    def _second_eye(
        self,
        settings: Any,
        models: Any,
        goal: str,
        voice: str | None,
        interval_s: float,
        diff_threshold: float,
        min_judge_interval_s: float,
        as_json: bool,
        stop_file: str | None = None,
    ) -> int:
        """Continuous screen-diff-triggered vision judgment with an instant spoken
        correction (the "second eye" mode for real-time screen watching).

        Unlike --describe (which starts/stops the vision model per call) and --speak
        (which starts/stops kokoro per call), this keeps BOTH services warm for the
        whole watch session so a judgment is only inference latency, not a cold model
        load -- the difference between "instant" and a 20-30s wait per correction.

        Loop: grab the desktop on `interval_s`, skip judging until the downscaled
        grayscale frame differs from the last-judged frame by at least
        `diff_threshold` (fraction of changed pixels) AND at least
        `min_judge_interval_s` has passed since the last judgment (a floor so a
        constantly-changing screen, e.g. video, cannot spam the model). On a
        triggered frame, ask the vision model whether anything looks wrong against
        the stated goal; a reply other than exactly "OK" is spoken immediately.

        Runs until Ctrl+C, or until `stop_file` (if given) appears on disk -- checked
        once per loop iteration. A stop file is the reliable way to stop this from
        another process (e.g. an external tool driving second-eye mode): Windows can
        only deliver CTRL_BREAK_EVENT to a child that owns a console, which a
        CREATE_NO_WINDOW-spawned background process does not have, so a signal-based
        stop silently never arrives and the caller ends up hard-killing (orphaning the
        vision model and Kokoro). Polling a file has no such requirement. Always stops
        both services on exit (or on a fatal start failure) so nothing is left
        orphaned, matching --describe/--speak.
        """
        import atexit
        import json
        import time

        from PIL import ImageGrab

        from tts import KokoroClient, TtsUnavailableError
        from vision import QwenVisionModel, VisionError

        goal = (goal or "").strip()
        if not goal:
            self._out(
                'error: --second-eye requires --goal "what you are doing" '
                "(e.g. --goal \"learning Python, building a CLI tool\")"
            )
            return 2

        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        voice_name, voice_warning = self._resolve_voice(settings, voice)
        if voice_warning:
            self._out(voice_warning)

        vision_log = log_dir / "vision_llama.log"
        _registry, vision_manager, vision_controller = self._build_controller(
            settings, models, log_path=vision_log
        )
        atexit.register(vision_manager.stop_all)
        vision = QwenVisionModel(vision_controller)

        tts_manager = None
        tts_controller = None
        tts_started_by_us = False
        kokoro_client: Any = None
        try:
            tts_manager, tts_controller, tts_started_by_us, tts_port = self._tts_session(
                settings
            )
            kokoro_client = KokoroClient(tts_port)
        except Exception as exc:  # noqa: BLE001 - honest degrade: watch without voice
            self._out(f"warning: text-to-speech unavailable ({exc}); watching silently")

        # M15.10: rewritten after a live test in which a screen showing a
        # terminal was judged "OK" against the goal "practicing piano sheet
        # music". The old wording anchored the model on errors ("clearly wrong
        # -- an error message, a mistake in code...") and mentioned goal
        # deviation last, so a small VL model took the easy OK. The deviation
        # check is now the FIRST question, asked as its own step, and the
        # owner's name no longer ships in the prompt.
        judge_prompt = (
            f"You are a second pair of eyes on the user's screen. Their stated "
            f"goal right now: {goal}. Look at this screenshot and answer two "
            "questions in order. FIRST: does the screen actually show work on "
            "that goal? If it clearly shows something unrelated, reply with ONE "
            "short spoken sentence noting the drift, e.g. 'This does not look "
            "like <goal> - you are in <what you see>.' SECOND: only if the "
            "screen matches the goal, check for visible problems - an error "
            "message, a mistake, a wrong value - and reply with ONE short "
            "spoken correction if you see one. If the screen matches the goal "
            "and nothing looks wrong, reply with exactly: OK"
        )

        stop_path = Path(stop_file) if stop_file else None
        self._out(
            f'[second-eye] watching. goal: "{goal}". '
            + (f"stop file: {stop_path}" if stop_path else "Ctrl+C to stop.")
        )
        last_judged_small: Any = None
        last_judge_at = 0.0
        judged_count = 0
        corrections_count = 0
        try:
            while True:
                if stop_path is not None and stop_path.exists():
                    break
                now = time.monotonic()
                frame = ImageGrab.grab()
                small = frame.convert("L").resize((320, 180))
                should_judge = last_judged_small is None or (
                    _frame_diff_fraction(small, last_judged_small) >= diff_threshold
                    and (now - last_judge_at) >= min_judge_interval_s
                )
                if not should_judge:
                    time.sleep(interval_s)
                    continue
                last_judged_small = small
                last_judge_at = now
                judged_count += 1

                shot_path = log_dir / "second_eye_frame.png"
                frame.save(shot_path)
                try:
                    answer = vision.describe(str(shot_path), judge_prompt)
                except VisionError as exc:
                    self._out(f"[second-eye] vision error: {exc} ({exc.remedy})")
                    time.sleep(interval_s)
                    continue

                verdict = answer.strip()
                if verdict.upper() != "OK":
                    corrections_count += 1
                    self._out(f"[second-eye] correction: {verdict}")
                    if kokoro_client is not None:
                        try:
                            kokoro_client.speak(
                                verdict,
                                voice=voice_name,
                                speed=settings.tts.speed,
                                play=settings.tts.autoplay,
                                temp_dir=str(log_dir),
                            )
                        except TtsUnavailableError as exc:
                            self._out(f"[second-eye] tts unavailable: {exc.remedy or exc}")
                elif as_json:
                    self._out(json.dumps({"judged": judged_count, "verdict": "OK"}))
                time.sleep(interval_s)
        except KeyboardInterrupt:
            pass
        finally:
            vision_manager.stop_all()
            if tts_started_by_us and tts_controller is not None:
                tts_controller.stop()
            # The judged frame is a full screenshot; it exists only to hand to
            # the vision model and must not outlive the session on disk.
            try:
                (log_dir / "second_eye_frame.png").unlink(missing_ok=True)
            except OSError:
                pass
        self._out(
            f"[second-eye] stopped. judged {judged_count} frame(s), "
            f"spoke {corrections_count} correction(s)."
        )
        return 0

    def _assistant(
        self,
        settings: Any,
        models: Any,
        *,
        text: bool,
        no_speak: bool,
        voice: str | None,
        model_id: str | None,
        timings: bool,
        as_json: bool,
        listen_continuous: bool = False,
    ) -> int:
        """Run the built-in assistant loop (M7). Ensure a model, converse, clean up.

        This is the CLI surface AC14 (text mode) and AC17 (mic) grade. It ensures a
        chat model is RUNNING via the existing ModelController (starting the default
        model if none is up -- an honest failure with a remedy if it cannot start),
        composes an LlmClient against that model's resolved loopback port, and runs
        AssistantLoop with the injected STT/TTS seams. On exit it stops whatever it
        started (model, Kokoro service, mic) so no child process is orphaned.

        A single Ctrl-C interrupts the current reply (generation + audio stop, the
        next turn proceeds); a double Ctrl-C, EOF, or 'quit' ends the session.
        """
        import atexit
        import json
        import signal
        import time
        from threading import Event

        from assistant import (
            VOICE_MODE_SYSTEM_PROMPT,
            AssistantLoop,
            ConversationState,
            HalfDuplexGate,
            KokoroTtsSink,
            PrintTtsSink,
            PushToTalkSttSource,
            TextSttSource,
            clean_for_speech,
        )
        from llm import LlamaCppClient
        from memory import ConversationMemory, resolve_memory_dir
        from services import ServiceStatus
        from tts import KokoroClient

        target_id = model_id or settings.launcher.default_model
        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        llama_log = log_dir / "assistant_llama.log"

        registry, manager, controller = self._build_controller(
            settings, models, log_path=llama_log
        )
        atexit.register(manager.stop_all)
        model = registry.get(target_id)
        if model is None or not model.location:
            reason = (
                f"chat model '{target_id}' is not available (unknown id or empty "
                f"location); set launcher.default_model or pass --model"
            )
            self._emit_assistant_error(reason, as_json)
            return 1

        # Ensure the model is running (start it if the controller has none up).
        started_model = False
        if controller.running_model_id != target_id:
            try:
                status = controller.start(target_id)
            except ValueError as exc:
                self._emit_assistant_error(str(exc), as_json)
                return 1
            if status is not ServiceStatus.RUNNING:
                reason = self._smoke_failure_reason(
                    llama_log, settings.services.ready_timeout_s
                )
                self._emit_assistant_error(
                    f"chat model '{target_id}' did not start ({status.value}); {reason}",
                    as_json,
                )
                manager.stop_all()
                return 1
            started_model = True
        llm_port = controller.running_port or settings.ports.llama_cpp
        llm_client = LlamaCppClient(llm_port)

        # Voice OUT setup (unless --no-speak). A TTS start failure degrades to
        # text-only rather than aborting the whole session (honest degradation).
        speaking = not no_speak
        voice_name = ""
        tts_sink: Any = None
        tts_manager = None
        tts_controller = None
        tts_started = False
        # D-M7-3: half-duplex gate shared by the TTS sink (sets it while speaking) and
        # the mic source (drops segments while it is set), so the assistant never
        # transcribes its own Kokoro output back as a user turn. Only meaningful when
        # actually speaking through a live mic; text mode leaves it None.
        half_duplex_gate = (
            HalfDuplexGate(tail_s=settings.assistant.half_duplex_tail_s)
            if (speaking and not text)
            else None
        )
        if speaking:
            voice_name, warning = self._resolve_voice(settings, voice)
            if warning:
                self._out(f"  {warning}")
            try:
                tts_manager, tts_controller, tts_started, tts_port = self._tts_session(
                    settings
                )
            except Exception as exc:  # noqa: BLE001 - degrade to text-only honestly
                self._out(f"  text-to-speech unavailable: {exc}; continuing text-only")
                speaking = False
            else:
                # wav_dir makes each spoken clip also land as a durable wav under
                # logs/, so the assistant's real audio output is inspectable (the
                # M7 audible-smoke evidence), not just played and gone.
                tts_sink = KokoroTtsSink(
                    KokoroClient(tts_port),
                    speed=settings.tts.speed,
                    wav_dir=log_dir,
                    speaking_gate=half_duplex_gate,
                    lead_silence_ms=settings.tts.reply_lead_silence_ms,
                )
        if tts_sink is None:
            # No-op sink so the loop's speak path has a target even when silent.
            tts_sink = PrintTtsSink()

        interrupt = Event()
        mic_source: Any = None
        if text:
            stt: Any = TextSttSource()
            measure_stt = False
        else:
            mic_source = _MicSttSource(
                self, settings, interrupt, speaking_gate=half_duplex_gate
            )
            # D-M7-6: push-to-talk is the DEFAULT voice capture. --listen-continuous
            # (or assistant.capture_mode: continuous) opts back into the legacy
            # always-listening mic, which is fragile in an open-mic room. In
            # push-to-talk the mic only feeds the assistant inside an Enter-opened
            # window; _MicSttSource is driven segment-by-segment via poll_segment.
            continuous = (
                listen_continuous
                or settings.assistant.capture_mode == "continuous"
            )
            if continuous:
                stt = mic_source
            else:
                stt = PushToTalkSttSource(
                    mic_source,
                    emit=self._out,
                    max_capture_s=settings.assistant.max_capture_s,
                    settle_s=settings.assistant.utterance_settle_s,
                )
            measure_stt = True

        session_id = time.strftime("%Y-%m-%dT%H%M%S")
        state = ConversationState(session_id=session_id)
        # D-M7-2(a): in voice mode, condition the model to answer briefly in plain
        # spoken text (no markdown), so replies are usable aloud. Only when actually
        # speaking -- text-only sessions keep the model's normal formatting. Folded
        # into the single system message (not appended as a second one) because a
        # second leading system message breaks chat templates that only allow one
        # (e.g. Qwen3's: "System message must be at the beginning").
        system_prompt = settings.assistant.system_prompt
        if speaking:
            system_prompt = f"{system_prompt}\n\n{VOICE_MODE_SYSTEM_PROMPT}"
        state.append("system", system_prompt)
        memory = ConversationMemory(
            resolve_memory_dir(settings.data_dir, settings.memory.dir),
            enabled=settings.memory.enabled,
        )

        def emit_timings(t: Any) -> None:
            if timings:
                self._out("TIMINGS " + json.dumps(t.as_dict()))

        loop = AssistantLoop(
            stt,
            llm_client,
            tts_sink,
            state,
            voice=voice_name,
            speaking=speaking,
            speak_per_sentence=settings.assistant.speak_per_sentence,
            context_size=model.context_size,
            response_reserve_tokens=settings.assistant.response_reserve_tokens,
            max_history_turns=settings.assistant.max_history_turns,
            measure_stt=measure_stt,
            memory=memory,
            recall_limit=settings.memory.search_limit,
            interrupt=interrupt,
            emit=self._out,
            on_timings=emit_timings,
            # D-M7-2(b)/D-M7-4: strip markdown AND emoji/non-speech glyphs before
            # speaking and before console display, so Kokoro never voices "asterisk
            # asterisk" or an emoji ("face with smiling eyes"). Applied in both mic
            # and text assistant modes (both are --assistant).
            text_filter=clean_for_speech,
            half_duplex_gate=half_duplex_gate,
        )

        # SIGINT: a single Ctrl-C sets the interrupt (stop the current reply); a
        # second within 1.5s raises KeyboardInterrupt to quit the session.
        last_sigint = [0.0]

        def _on_sigint(_signum: Any, _frame: Any) -> None:
            now = time.monotonic()
            interrupt.set()
            if mic_source is not None:
                mic_source.stop()
            # D-M7-6: also stop the push-to-talk wrapper (if that is the active source)
            # so a capture window in progress ends promptly on interrupt.
            stt_stop = getattr(stt, "stop", None)
            if callable(stt_stop):
                stt_stop()
            if now - last_sigint[0] < 1.5:
                raise KeyboardInterrupt
            last_sigint[0] = now

        previous_handler = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, _on_sigint)
        except (ValueError, OSError):
            # Not on the main thread (e.g. some test harnesses): skip the handler;
            # EOF/'quit' still ends the session cleanly.
            previous_handler = None

        if not as_json:
            mode = "text" if text else "voice"
            self._out(
                f"locitize assistant ready ({mode} mode, model {target_id}"
                f"{'' if speaking else ', text-only'}). "
                f"{'Type a prompt; quit/EOF to end.' if text else 'Speak; Ctrl-C to interrupt, twice to quit.'}"
            )
        code = 0
        try:
            code = loop.run()
        except KeyboardInterrupt:
            self._out("\n[assistant session ended]")
        finally:
            try:
                if hasattr(tts_sink, "close"):
                    tts_sink.close()
            except Exception:  # noqa: BLE001 - cleanup must not raise
                pass
            if mic_source is not None:
                mic_source.close()
            if tts_started and tts_controller is not None:
                tts_controller.stop()
            if started_model:
                controller.stop()
            manager.stop_all()
            if previous_handler is not None:
                try:
                    signal.signal(signal.SIGINT, previous_handler)
                except (ValueError, OSError):
                    pass

        clean = self._confirm_clean(manager)
        if as_json:
            self._out(
                json.dumps(
                    {
                        "outcome": "ok" if clean else "orphan",
                        "model": target_id,
                        "session_id": session_id,
                        "turns": max(0, len(state.messages) - 1) // 2,
                        "clean": clean,
                    }
                )
            )
        return 0 if clean else 1

    def _emit_assistant_error(self, reason: str, as_json: bool) -> None:
        """Print an honest assistant start failure (JSON or a short line)."""
        import json

        if as_json:
            self._out(json.dumps({"outcome": "failed", "reason": reason}))
        else:
            self._out(f"[XX] {reason}")

    def _run_listen(
        self, settings: Any, duration: float, as_json: bool, smoke: bool
    ) -> int:
        """Run whisper-stream for a fixed window, then stop it cleanly.

        smoke=True (AC6): assert only the lifecycle -- started, then terminated with
        no orphan -- making NO claim about captured content. smoke=False (--listen,
        AC9): additionally echo the live transcript so the owner can grade voice
        quality. Returns 0 only when the process started and exited cleanly.
        """
        import atexit
        import json
        import time

        from services import ServiceStatus
        from whisper import build_whisper_stream_spec

        result: dict[str, Any] = {
            "service": "whisper_stream",
            "outcome": "failed",
            "reason": "",
            "pid": None,
            "duration_s": duration,
            "elapsed_s": 0.0,
        }
        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "whisper_stream.log"

        manager, controller = self._build_service_controller(
            settings, lambda: build_whisper_stream_spec(settings, str(log_path))
        )
        atexit.register(manager.stop_all)

        started_clock = time.monotonic()
        started_ok = False
        clean = False
        try:
            try:
                status = controller.start()
            except ValueError as exc:
                result["reason"] = str(exc)
                status = None
            result["pid"] = controller.pid
            if status is ServiceStatus.RUNNING:
                started_ok = True
                if smoke:
                    # Lifecycle-only: hold the process for the fixed window.
                    time.sleep(duration)
                else:
                    self._out(
                        f"Listening for {duration:.0f}s - speak now; "
                        f"transcript appears below:"
                    )
                    self._tail_listen(log_path, duration)
            elif status is not None:
                result["reason"] = (
                    f"whisper-stream did not start ({status.value}); "
                    f"see logs/whisper_stream.log"
                )
        finally:
            controller.stop()
            clean = self._confirm_clean(manager)

        if started_ok and clean:
            result["outcome"] = "ok"
            result["reason"] = "whisper-stream started, captured, and exited cleanly"
        elif started_ok and not clean:
            result["reason"] = "whisper-stream did not terminate cleanly (possible orphan)"
        result["elapsed_s"] = round(time.monotonic() - started_clock, 2)
        if not smoke:
            # Surface the raw capture for the owner's manual AC9 read.
            result["capture_log"] = str(log_path)
        if as_json or smoke:
            self._out(json.dumps(result))
        else:
            self._out(f"  outcome: {result['outcome']} ({result['reason']})")
        return 0 if result["outcome"] == "ok" else 1

    def _tail_listen(self, log_path: Any, duration: float, emit: Any = None) -> None:
        """Echo new bytes appended to the capture log for `duration` seconds.

        whisper-stream writes its live transcription to the child-output log; this
        polls the file for growth and surfaces new content so the owner sees the
        transcript in real time (for the manual AC9 grade). Best-effort: a transient
        read error is ignored so a locked-file moment never aborts the session.

        `emit` is the sink for each deduplicated segment; it defaults to self._out
        (the terminal path) and the GUI passes a callback that marshals the line
        onto its result queue - so both surfaces share the identical M3
        deduplicated + VAD-gated pipeline with no duplicated capture code.

        The raw child log is left exactly as whisper-stream wrote it (debuggability);
        only LOCITIZE's echoed transcript is de-duplicated here, because whisper-stream
        transcribes overlapping windows and emits a boundary-spanning utterance in
        two windows (defect D-M3-1 / AC9). Incomplete trailing lines are buffered
        across polls so a segment split by a read boundary is not falsely counted.
        """
        import time

        from whisper import StartupNoiseGate, TranscriptDeduplicator

        sink = emit if emit is not None else self._out
        deadline = time.monotonic() + duration
        position = 0
        pending = ""  # partial line carried across reads until its newline arrives
        dedup = TranscriptDeduplicator()
        # Startup-noise gate (D-M7-1): ignore whisper-stream's engine/boot output
        # until the "[Start speaking]" banner, then filter engine diagnostics. One
        # gate per capture session so the banner-crossed state persists across polls.
        gate = StartupNoiseGate()
        while time.monotonic() < deadline:
            try:
                with open(log_path, "rb") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
                if chunk:
                    pending += chunk.decode("utf-8", errors="replace")
                    pending = self._emit_listen_lines(
                        pending, dedup, final=False, emit=sink, gate=gate
                    )
            except OSError:
                pass
            time.sleep(0.5)
        # Flush the last buffered line (whisper-stream may not end it with a newline).
        self._emit_listen_lines(pending, dedup, final=True, emit=sink, gate=gate)

    def _emit_listen_lines(
        self, buffer: str, dedup: Any, final: bool, emit: Any = None, gate: Any = None
    ) -> str:
        """Split `buffer` into transcript segments, emit the non-duplicates.

        whisper-stream separates results with newlines and, for in-place updates,
        carriage returns; both are treated as segment boundaries so each is checked
        against the deduplicator. Returns the leftover partial segment (empty string
        when final=True, since there is nothing more coming to complete it). `emit`
        defaults to self._out; the GUI supplies its own sink.

        `gate` (optional) is a whisper.StartupNoiseGate: when supplied, each raw line
        is first run through the startup-noise gate (defect D-M7-1) so whisper-stream
        boot diagnostics and "### Transcription" block markers are dropped and only a
        line's real spoken words reach the deduplicator. With gate=None the behavior
        is unchanged (the existing dedup-only listen tests exercise that path).
        """
        sink = emit if emit is not None else self._out
        # Normalize CR / CRLF to LF so every segment boundary splits uniformly.
        normalized = buffer.replace("\r\n", "\n").replace("\r", "\n")
        parts = normalized.split("\n")
        # The last part is a still-incomplete line unless we are flushing at the
        # end, in which case it is a complete final segment and stays in `parts`.
        tail = "" if final else parts.pop()
        for part in parts:
            # Startup-noise gate first (D-M7-1): drops engine/boot lines and the
            # pre-"[Start speaking]" preamble, and reduces a timestamped line to its
            # spoken words. None means "not speech" -> skip before dedup entirely.
            if gate is not None:
                segment = gate.surfaced_text(part)
                if segment is None:
                    continue
            else:
                segment = part
            if dedup.accept(segment):
                sink(segment)
        return tail

    def _emit_result(
        self, result: dict, as_json: bool, elapsed_from: float | None = None
    ) -> int:
        """Print a transcription result record and return its exit code.

        Centralizes the JSON-vs-human output and the exit-code mapping for the
        --transcribe path so every early-return uses the same honest shape.
        """
        import json
        import time

        if elapsed_from is not None:
            result["elapsed_s"] = round(time.monotonic() - elapsed_from, 2)
        if as_json:
            self._out(json.dumps(result))
        else:
            self._out(f"outcome: {result['outcome']}")
            if result.get("transcript"):
                self._out(f"transcript: {result['transcript']}")
            if result.get("reason"):
                self._out(f"reason: {result['reason']}")
        return 0 if result["outcome"] == "ok" else 1

    def _append_journal(self, settings: Any, report: Any) -> None:
        """Append a journal entry summarizing this interactive session."""
        from documentation import DevelopmentJournal, JournalEntry

        # <data root>/reports, never <install>/docs (DEC-M14-9): the journal is a
        # record of this user's own sessions, so it is user data, and the install
        # tree is read-only at runtime (invariant W1). Resolved through the
        # benchmark helper so the journal and the benchmark reports always land
        # in the same folder.
        from benchmark import resolve_results_dir

        reports = resolve_results_dir(settings)
        journal = DevelopmentJournal(reports / "development_journal.md")
        issues = ", ".join(
            f"{r.name}={r.status.value}" for r in report.results if r.status.value != "PASS"
        )
        entry = JournalEntry.now(
            completed=f"launcher session; health overall {report.overall.value}",
            issues=issues or "none",
            fixes="none",
            next_steps="none",
        )
        journal.append(entry)


class _MicSttSource:
    """Real microphone STT source for the assistant (M7.8), reusing the M3 pipeline.

    Wraps the existing whisper-stream capture path: it starts whisper-stream once
    (via the launcher's shared service-controller builder), then each
    next_utterance() waits for the next new, deduplicated transcript segment written
    to the child log -- exactly the VAD-gated, TranscriptDeduplicator-cleaned
    pipeline the --listen path uses (no new STT code, Architecture M7.8). A turn is
    the next detected utterance; the source returns None once stopped so the loop
    ends cleanly. Not unit-tested here (it drives a real binary); it is exercised by
    the owner's manual live session (AC17) and shares the M3-proven capture code.
    """

    def __init__(
        self,
        launcher: "Launcher",
        settings: Any,
        interrupt: Any,
        speaking_gate: Any = None,
        manager: Any = None,
    ) -> None:
        from whisper import (
            StartupNoiseGate,
            TranscriptDeduplicator,
            build_whisper_stream_spec,
        )

        self._launcher = launcher
        self._settings = settings
        self._interrupt = interrupt
        # D-M7-3/3b half-duplex gate: the TTS sink records the wall-clock intervals it
        # spoke; captured segments whose OWN capture window overlaps a spoken interval
        # are the assistant's own audio echoing back and are DROPPED rather than queued
        # as a turn -- deterministically, regardless of when the poll reads them.
        self._speaking_gate = speaking_gate
        # Guard margin for the capture-overlap test (D-M7-3b), absorbing anchor jitter.
        self._overlap_guard_s = settings.assistant.half_duplex_guard_s
        self._stopped = False
        self._dedup = TranscriptDeduplicator()
        # D-M7-1: the mic source must emit ONLY real transcribed speech. This gate
        # drops whisper-stream's boot diagnostics (ggml/CUDA init, SDL probe, model
        # load, "### Transcription" markers) and everything before the
        # "[Start speaking]" banner, so the engine's own output never becomes a
        # fabricated user turn to the assistant LLM.
        self._gate = StartupNoiseGate()
        self._position = 0
        self._pending = ""  # partial line carried across polls
        self._queue: list[str] = []

        log_dir = resolve_log_dir(settings)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / "assistant_whisper_stream.log"
        # RM3 (M10.4): the GUI assistant session passes the GUI's SHARED,
        # atexit-backstopped ServiceManager so whisper-stream is reaped by BOTH
        # gui_controller.shutdown().stop_all() AND the single atexit backstop on that
        # manager (window close, End assistant, or a bypassed close all reap it). The
        # CLI --assistant path passes no manager, so a FRESH one is built and gets its
        # own atexit backstop here (unchanged behavior). Whether the manager is shared
        # is recorded so we do not double-register an atexit for a manager that already
        # has one.
        self._owns_manager = manager is None
        self._manager, self._controller = launcher._build_service_controller(
            settings,
            lambda: build_whisper_stream_spec(settings, str(self._log_path)),
            manager=manager,
        )
        if self._owns_manager:
            # Fresh manager (CLI path): register the no-orphan backstop here so an
            # abnormal interpreter exit that skips _assistant's finally/close() still
            # reaps the whisper-stream mic child, matching the model and Kokoro
            # managers in the same method (Review M-1). stop_all is idempotent, so the
            # normal-path close() and this backstop can both run without harm. A
            # SHARED manager already has its atexit backstop (registered by _run_gui),
            # so we do NOT register a second one for it.
            import atexit

            atexit.register(self._manager.stop_all)
        from services import ServiceStatus

        status = self._controller.start()
        self._running = status is ServiceStatus.RUNNING
        if not self._running:
            launcher._out(
                "  microphone capture unavailable (whisper-stream did not start); "
                "see logs/assistant_whisper_stream.log"
            )

    def stop(self) -> None:
        """Signal the source to end (next_utterance returns None)."""
        self._stopped = True

    def next_utterance(self) -> str | None:
        """Block until the next transcript segment arrives; None when stopped.

        Polls the whisper-stream child log for newly appended, deduplicated
        segments. A double Ctrl-C raises KeyboardInterrupt through here to quit; a
        single Ctrl-C (interrupt set) during idle is treated as "end this session"
        so the owner is never trapped in the loop.
        """
        import time

        if not self._running or self._stopped:
            return None
        while not self._stopped:
            self._poll_once()
            if self._queue:
                return self._queue.pop(0)
            time.sleep(0.3)
        return None

    def _poll_once(self) -> None:
        """Read new bytes from the capture log and queue any complete segments."""
        try:
            with open(self._log_path, "rb") as handle:
                handle.seek(self._position)
                chunk = handle.read()
                self._position = handle.tell()
        except OSError:
            return
        if not chunk:
            return
        self._pending += chunk.decode("utf-8", errors="replace")
        # Reuse the launcher's dedup+split so the mic path and --listen path share
        # the identical M3 segmenting (D-M3-1 dedup, D-M3-2 VAD gating), now with the
        # D-M7-1 startup-noise gate so only real speech reaches the assistant loop.
        # The emit callback additionally applies the D-M7-3 half-duplex drop: any
        # segment captured while the assistant is speaking (its own TTS echoing back)
        # is discarded here instead of being queued as a fabricated user turn.
        self._pending = self._launcher._emit_listen_lines(
            self._pending,
            self._dedup,
            final=False,
            emit=self._queue_segment,
            gate=self._gate,
        )

    def flush(self) -> None:
        """Discard everything captured while a push-to-talk window was closed (D-M7-6).

        Reads any log the child appended since the last poll and clears the queue, so
        only speech captured AFTER this call -- inside the owner's just-opened window
        -- becomes the utterance. This is the structural fix for the always-listening
        failures: ambient hallucinations and the assistant's own TTS echo are captured
        while the window is closed and are dropped here, never reaching the LLM.
        """
        if not self._running:
            return
        self._poll_once()
        self._queue.clear()

    def poll_segment(self, timeout_s: float) -> str | None:
        """Return the next captured transcript segment, or None within timeout_s (D-M7-6).

        The push-to-talk source calls this repeatedly during an open window. It polls
        the whisper-stream log and returns the next queued (deduped, startup-gated,
        half-duplex-checked) segment; if none arrives before timeout_s elapses it
        returns None so the caller's settle/max-capture timing stays responsive and a
        silent room can never hang the turn.
        """
        import time

        if not self._running or self._stopped:
            return None
        deadline = time.monotonic() + max(0.0, timeout_s)
        while not self._stopped:
            self._poll_once()
            if self._queue:
                return self._queue.pop(0)
            if time.monotonic() >= deadline:
                return None
            # Short sleep so a double Ctrl-C (stop) and the deadline are honored
            # promptly; capped by the remaining time so we never overshoot timeout_s.
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return None

    def _queue_segment(self, segment: str) -> None:
        """Queue a transcript segment unless the half-duplex gate says drop it (D-M7-3b).

        Correctness is capture-time: the segment's absolute capture interval (anchored
        from whisper's per-segment timestamps by self._gate) is tested for overlap
        against the speaking gate's recorded intervals. A segment captured while the
        assistant was speaking -- its own Kokoro output re-heard through the mic -- is
        dropped even when whisper transcribes and this poll reads it LONG after
        playback ended (the exact race that reopened D-M7-3). A segment captured
        strictly after the assistant finished (genuine user speech) passes. When a
        line carries no timestamp, drop_captured_segment falls back to the poll-time
        flag. self._gate.last_capture_interval was just set by _emit_listen_lines for
        this very segment in the same single-threaded poll.
        """
        from assistant import drop_captured_segment

        interval = getattr(self._gate, "last_capture_interval", None)
        if drop_captured_segment(
            segment, self._speaking_gate, interval, self._overlap_guard_s
        ):
            return
        self._queue.append(segment)

    def close(self) -> None:
        """Stop whisper-stream and confirm no orphan (shared no-orphan discipline).

        When this source owns its manager (CLI path) it also stop_all()s it. On the
        GUI SHARED manager it stops ONLY its own whisper-stream controller and leaves
        the manager alone: the chat model and Kokoro on that shared manager are reaped
        by the AssistantHandle / gui_controller.shutdown().stop_all() / atexit backstop
        (M10.4), so End-assistant does not nuke a model the owner may still chat with.
        """
        self._stopped = True
        try:
            self._controller.stop()
            if self._owns_manager:
                self._manager.stop_all()
        except Exception:  # noqa: BLE001 - cleanup must report via caller, not raise
            pass


class _GuiAssistantHandle:
    """Handle to a live GUI assistant session (M10.4).

    Returned by _gui_assistant_start. The ops worker / shutdown path drives it; the
    UI thread only ever calls talk()/interrupt() (both non-blocking). It owns the
    dedicated session thread and the assistant-specific whisper/Kokoro teardown.
    """

    def __init__(
        self,
        stt: Any,
        loop: Any,
        tts_sink: Any,
        mic_source: Any,
        thread: Any,
        port: int | None,
    ) -> None:
        self._stt = stt
        self._loop = loop
        self._tts = tts_sink
        self._mic = mic_source
        self._thread = thread
        # The resolved chat-model loopback port, so the monitor (Thread C) animates
        # tokens/s + context fill during the spoken conversation (M10.2).
        self.port = port
        self._stopped = False

    def talk(self) -> None:
        """Open ONE capture window (a Talk click). Non-blocking queue put (Thread A)."""
        self._stt.talk()

    def interrupt(self) -> None:
        """Barge-in: stop generation + audio mid-reply, like a single CLI Ctrl-C (M7.5)."""
        self._loop.interrupt.set()
        drain = getattr(self._tts, "drain", None)
        if callable(drain):
            drain()

    def stop(self) -> None:
        """End the session and reap its whisper/Kokoro children (idempotent, M10.4).

        The STOP sentinel unblocks the session thread's talk-gate wait; interrupt +
        mic.stop() end any capture window in progress promptly; the thread is joined
        bounded (daemon, so the process exits regardless). It stops the sink's
        playback worker and the whisper-stream mic (assistant-specific). The chat
        model + Kokoro service stay on the shared manager and are reaped by
        gui_controller.shutdown().stop_all() / the atexit backstop, so End-assistant
        does not disturb a model the owner may still chat with or the Voice panel.
        """
        if self._stopped:
            return
        self._stopped = True
        self._loop.interrupt.set()
        self._stt.stop()
        self._mic.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            if hasattr(self._tts, "close"):
                self._tts.close()
        except Exception:  # noqa: BLE001 - teardown must never raise across the boundary
            pass
        try:
            self._mic.close()
        except Exception:  # noqa: BLE001 - teardown must never raise across the boundary
            pass


class BootstrapError(RuntimeError):
    """Raised for an unrecoverable configuration problem (mapped to exit code 2)."""


def _safe_filename(name: str) -> str:
    """Turn a model id into a filesystem-safe log filename fragment."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the non-interactive flags (Architecture section 3)."""
    parser = argparse.ArgumentParser(
        prog="launcher", description="locitize platform launcher"
    )
    parser.add_argument(
        "--health", action="store_true", help="run the health ladder and exit"
    )
    parser.add_argument(
        "--stack-health",
        dest="stack_health",
        action="store_true",
        help="production stack liveness: TCP/HTTP probe 8080/8093/8096/8091/8092/4200 "
        "(PASS/FAIL; ignores RAM/VRAM). Prefer this over --health for ops green/red",
    )

    parser.add_argument(
        "--json", action="store_true", help="with --health, emit JSON"
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="bootstrap, verify, render, then exit (no interactive menu)",
    )
    parser.add_argument(
        "--service-status",
        action="store_true",
        help="print the managed-service status snapshot and exit (no GPU needed)",
    )
    parser.add_argument(
        "--egress",
        action="store_true",
        help="print the privacy ledger: every outbound connection this machine "
        "has made and why, or that it has made none. Proves the local-only "
        "promise instead of asserting it",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="show what is holding the GPU right now: total/used/free VRAM and "
        "every process on the card, marking which are locitize's own",
    )
    parser.add_argument(
        "--rtx-report",
        action="store_true",
        help="write the measured per-GPU compatibility report (models, contexts, "
        "tok/s, what a bigger card would unlock) to reports/rtx_report.md. "
        "Local only; sharing the file is your call",
    )
    parser.add_argument(
        "--gpu-free",
        action="store_true",
        help="free the GPU memory held by locitize's OWN servers (a crashed "
        "session or a measurement probe left running). Never touches another "
        "app's processes",
    )
    parser.add_argument(
        "--detect-capabilities",
        action="store_true",
        help="scan every registered model's GGUF for what it can do (tools, "
        "vision, reasoning) and write the findings into models.yaml",
    )
    parser.add_argument(
        "--check-updates",
        action="store_true",
        help="check the installed llama.cpp against the latest release (one "
        "allowlisted request, recorded in the privacy ledger). Never downloads",
    )
    parser.add_argument(
        "--smoke-start",
        metavar="MODEL_ID",
        default=None,
        help="start the real llama.cpp service for MODEL_ID, confirm readiness, "
        "then clean up; prints a JSON outcome",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=None,
        help="override the model's context size (used with --smoke-start)",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help="override the model's GPU layer count (used with --smoke-start)",
    )
    parser.add_argument(
        "--reasoning",
        metavar="LEVEL",
        default=None,
        help="thinking level for this launch, e.g. 'low', 'off', 'low/2048' "
        "(level/budget); levels come from the model's own chat template. Used "
        "with --smoke-start; the interactive menu takes the same form as a "
        "suffix after the model number.",
    )
    parser.add_argument(
        "--transcribe",
        metavar="AUDIO_FILE",
        default=None,
        help="transcribe AUDIO_FILE via whisper-server (starts it if not already "
        "running, then restores the prior state); prints a JSON transcript",
    )
    parser.add_argument(
        "--smoke-start-whisper",
        action="store_true",
        help="start the real whisper-server, confirm readiness, then clean up; "
        "prints a JSON outcome",
    )
    parser.add_argument(
        "--smoke-listen",
        action="store_true",
        help="start whisper-stream (mic capture), wait --duration seconds, stop it "
        "cleanly; prints a JSON lifecycle outcome (no content asserted)",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="live mic transcription: start whisper-stream, echo the transcript for "
        "--duration seconds, then stop cleanly",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="capture window in seconds for --smoke-listen (default 3) / --listen "
        "(default 15)",
    )
    parser.add_argument(
        "--speak",
        metavar="TEXT",
        default=None,
        help="synthesize TEXT via the Kokoro TTS service (starting it if needed, "
        "then restoring prior state) and play it; prints a JSON outcome with the "
        "produced wav path",
    )
    parser.add_argument(
        "--voice",
        metavar="VOICE",
        default=None,
        help="with --speak: the voice name (an on-disk .pt); defaults to tts.voice",
    )
    parser.add_argument(
        "--wav",
        metavar="OUT_PATH",
        default=None,
        help="with --speak: write the wav to OUT_PATH instead of the default "
        "logs/tts_speak.wav",
    )
    parser.add_argument(
        "--audition",
        action="store_true",
        help="speak one fixed sample sentence in each on-disk voice in sequence so "
        "you can pick a preferred voice (starts/stops the TTS service)",
    )
    parser.add_argument(
        "--smoke-tts",
        action="store_true",
        help="start the real kokoro TTS service, confirm /health readiness, "
        "synthesize the sample sentence to a wav, then clean up; prints a JSON "
        "outcome (functional proof that real audio bytes were produced, no orphan)",
    )
    parser.add_argument(
        "--smoke-openwebui",
        action="store_true",
        help="start the real Open WebUI managed service, confirm /health "
        "readiness within openwebui.ready_timeout_s (covers the slow first-run DB "
        "migration), then clean-stop and confirm no orphan; prints a JSON outcome",
    )
    parser.add_argument(
        "--chat-ui",
        dest="chat_ui",
        choices=["ask", "llamacpp", "openwebui"],
        default=None,
        help="override the saved chat-UI preference for this run",
    )
    parser.add_argument(
        "--assistant",
        action="store_true",
        help="run the built-in voice assistant: you speak, the model answers, and "
        "Kokoro reads the reply aloud sentence by sentence. Ensures a chat model is "
        "running first, then cleans up on exit",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="with --assistant: read typed prompts from stdin instead of the mic "
        "(the scriptable/headless path); exits on EOF or 'quit'",
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="with --assistant: print replies only, do not synthesize/play audio",
    )
    parser.add_argument(
        "--listen-continuous",
        dest="listen_continuous",
        action="store_true",
        help="with --assistant (mic): use the legacy always-listening capture instead "
        "of the default push-to-talk. Fragile in an open-mic room (ambient "
        "hallucination and self-echo); push-to-talk is recommended",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="with --assistant: print real per-stage TurnTimings (stt/llm/tts ms) "
        "after each turn",
    )
    parser.add_argument(
        "--describe",
        metavar="IMAGE",
        default=None,
        help="single-image vision Q&A: switch to a vision model (+ --mmproj) and "
        "answer a question about IMAGE. Single image only, not video. Pair with "
        "--prompt/--ask to set the question; --model to pick which vision model "
        "(default: settings.vision.model, else the first row tagged 'vision'); "
        "--json for a structured result",
    )
    parser.add_argument(
        "--prompt",
        "--ask",
        dest="prompt",
        metavar="QUESTION",
        default=None,
        help="with --describe: the question to ask about the image "
        "(defaults to settings.vision.prompt)",
    )
    parser.add_argument(
        "--second-eye",
        dest="second_eye",
        action="store_true",
        help="continuous real-time watcher (the 'second eye'): grabs the "
        "screen, judges only frames that changed since the last judgment against "
        "--goal via the vision model, and speaks any correction immediately via "
        "Kokoro. Both services stay warm for the session. Runs until Ctrl+C",
    )
    parser.add_argument(
        "--goal",
        metavar="GOAL",
        default=None,
        help="with --second-eye: what you are doing right now, e.g. "
        '"learning Python, building a CLI tool"',
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="with --second-eye: seconds between screen polls (default 1.0)",
    )
    parser.add_argument(
        "--diff-threshold",
        dest="diff_threshold",
        type=float,
        default=0.03,
        help="with --second-eye: fraction of the downscaled frame that must change "
        "to trigger a judgment (default 0.03)",
    )
    parser.add_argument(
        "--min-judge-interval",
        dest="min_judge_interval",
        type=float,
        default=4.0,
        help="with --second-eye: minimum seconds between judgments even under "
        "constant change (default 4.0)",
    )
    parser.add_argument(
        "--stop-file",
        dest="stop_file",
        metavar="PATH",
        default=None,
        help="with --second-eye: exit cleanly (stopping the vision model and "
        "Kokoro) once PATH exists on disk, checked once per loop iteration. The "
        "reliable way to stop this from another process; a background locitize "
        "process has no console, so Ctrl+C cannot reach it",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="launch the PySide6 locitize Desktop command center instead of the "
        "terminal menu",
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="launch the native PySide6 locitize Desktop command center",
    )
    # M13: locitize.bat now carries an implicit --desktop, so the terminal menu needs
    # its own explicit flag to stay reachable from that one remaining shortcut.
    # Bare `python launcher.py` still opens the menu, unchanged.
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="force the interactive terminal menu (wins over --desktop/--gui)",
    )
    parser.add_argument(
        "--gui-tk",
        dest="gui_tk",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    # M5 benchmark suite (Architecture M5.11). --benchmark runs the non-interactive
    # runner; --model selects one model (or 'all'/omitted for every launchable one);
    # --runs sets the timed-run count; --sweep uses each model's declared
    # benchmark_sweep axes; --resume skips already-completed scenarios.
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="run the benchmark suite over installed models (server /completion "
        "timings only), append results, and write back scores",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        default=None,
        help="with --benchmark: the model id to benchmark ('all' or omitted = every "
        "launchable model)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="with --benchmark: number of timed /completion runs per scenario "
        "(a warm-up run is always discarded first); default 3",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="with --benchmark: benchmark each model across its declared "
        "benchmark_sweep axes instead of only its current config",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="with --benchmark: skip scenarios already recorded (ok) in "
        "docs/benchmark_results.jsonl and continue",
    )
    return parser.parse_args(argv)


def _force_utf8_output() -> None:
    """Make stdout/stderr encode arbitrary model text without crashing (DEF-QA-1).

    On Windows, piped/redirected/captured stdout defaults to the legacy ANSI code
    page (cp1252), which raises UnicodeEncodeError the moment a reply contains an
    emoji or any non-Latin-1 character -- the documented scriptable/headless --text
    path is exactly that captured context. Reconfigure both streams to UTF-8 with
    errors="backslashreplace" (mirroring the project CLI) so any
    character is either encoded or safely escaped, never fatal, and ASCII output is
    unchanged. Guarded because a stream may lack reconfigure (already-wrapped or a
    detached stream).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    """Process entry point. Catches unexpected errors at the boundary (exit 1)."""
    _force_utf8_output()
    launcher = Launcher()
    try:
        return launcher.run(argv)
    except Exception as exc:  # noqa: BLE001 - boundary guard, never leak a traceback
        # Log the traceback to errors.log if logging is up; always print a terse line.
        try:
            from logger import get_logger

            get_logger("errors").exception("unexpected error in launcher")
        except Exception:  # noqa: BLE001 - logging must never mask the original error
            pass
        print(f"[XX] Unexpected error: {exc} - see logs/errors.log")
        return 1


if __name__ == "__main__":
    sys.exit(main())
