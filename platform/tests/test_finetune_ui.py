"""Controller, launcher, and Qt-view tests for the M13 fine-tune surface.

Headless throughout: no Streamlit, no container runtime, no GPU, and the Qt tests
use the offscreen platform (and skip cleanly where PySide6 is absent). They prove
the parts a fixture-only scanner test cannot: the five new intents enqueue the
right commands without blocking, the discovered-model guards refuse the writes
they must refuse, the launcher's --terminal precedence holds, and the Fine-tune
page renders each canned Result into the right widgets.
"""

from __future__ import annotations

import os
import queue
import time
from pathlib import Path
from unittest import mock

import finetune
import gui_controller
import pytest
from config import FineTuneConfig, Model, Settings
from gui_controller import GuiController
from services import PortUnavailableError, ServiceStatus

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _FakeManager:
    def __init__(self):
        self.stop_all_calls = 0

    def stop_all(self):
        self.stop_all_calls += 1


class _FakeRegistry:
    def __init__(self, models=()):
        self._models = list(models)

    def all(self):
        return list(self._models)

    def get(self, model_id):
        return next((m for m in self._models if m.id == model_id), None)

    def discovered_items(self):
        return []

    def registered_finetune_ids(self):
        return set()


class _FakeStudio:
    """Stands in for the studio's SingleServiceController.

    `resolved_port` mirrors the real controller's property (services.py 662-664):
    the port the child actually bound, or None before anything started. The
    controller derives the studio URL from it, so tests must be able to move it.
    """

    def __init__(self, status=ServiceStatus.RUNNING, resolved_port=8501):
        self.calls = []
        self._status = status
        self.resolved_port = resolved_port

    def start(self):
        self.calls.append("start")
        return self._status

    def stop(self):
        self.calls.append("stop")
        return ServiceStatus.STOPPED


def _settings(tmp_path: Path, **overrides) -> Settings:
    cfg = FineTuneConfig(enabled=True, studio_dir=str(tmp_path / "studio"), **overrides)
    return Settings(finetune=cfg, base_dir=tmp_path)


def _install_studio(tmp_path: Path) -> None:
    """Make the fixture studio 'available': an app script plus its own interpreter."""
    app = tmp_path / "studio" / "app" / "app.py"
    app.parent.mkdir(parents=True, exist_ok=True)
    app.write_text("# streamlit app\n", encoding="utf-8")
    interpreter = tmp_path / "studio" / ".venv" / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_bytes(b"MZ")


def _controller(tmp_path, studio=None, models=()) -> GuiController:
    return GuiController(
        _settings(tmp_path),
        _FakeRegistry(models),
        object(),
        object(),
        _FakeManager(),
        fetch=lambda port, path: None,
        finetune_controller_factory=(lambda: studio) if studio else None,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    finetune.clear_scan_cache()
    yield
    finetune.clear_scan_cache()


# --------------------------------------------------------------------------- #
# The five new intents (they only enqueue; nothing blocks the UI thread)
# --------------------------------------------------------------------------- #


def test_finetune_intents_only_enqueue_commands(tmp_path):
    gc = _controller(tmp_path)
    gc.request_finetune_start()
    gc.request_finetune_stop()
    gc.request_finetune_open()
    gc.request_scan_finetunes()
    gc.request_register_discovered("ft:example")
    kinds = []
    while True:
        try:
            command = gc.command_q.get_nowait()
        except queue.Empty:
            break
        kinds.append(command.kind)
    assert kinds == [
        "finetune_start",
        "finetune_stop",
        "finetune_open",
        "finetune_scan",
        "finetune_register",
    ]


def test_finetune_start_publishes_starting_then_running(tmp_path):
    """A real start pushes Starting first, then Running with the loopback URL."""
    studio = _FakeStudio()
    gc = _controller(tmp_path, studio=studio)
    _install_studio(tmp_path)

    gc._do_finetune_start()
    states = []
    while True:
        try:
            states.append(gc.result_q.get_nowait())
        except queue.Empty:
            break
    assert [r.payload["status"] for r in states] == ["starting", "running"]
    assert states[-1].payload["url"] == "http://127.0.0.1:8501/"
    assert studio.calls == ["start"]


def test_finetune_start_failure_is_an_honest_error_state(tmp_path):
    """A studio that never goes ready lands in Error with the log path, no traceback."""
    studio = _FakeStudio(status=ServiceStatus.STOPPED_ERROR)
    gc = _controller(tmp_path, studio=studio)
    _install_studio(tmp_path)

    gc._do_finetune_start()
    results = []
    while True:
        try:
            results.append(gc.result_q.get_nowait())
        except queue.Empty:
            break
    last = results[-1]
    assert last.ok is False
    assert "finetune_studio.log" in (last.error or "")
    assert "Traceback" not in (last.error or "")


# --------------------------------------------------------------------------- #
# Port reassignment in the UI layer (defect D-M13-1 / decision DEC-M13-2)
# --------------------------------------------------------------------------- #


def test_finetune_port_state_url_follows_the_resolved_port(tmp_path):
    """Running on a reassigned port must be advertised as that port, not 8501."""
    studio = _FakeStudio(resolved_port=8080)
    gc = _controller(tmp_path, studio=studio)
    _install_studio(tmp_path)

    gc._do_finetune_start()
    payload = gc.finetune_state()
    assert payload["status"] == "running"
    assert payload["url"] == "http://127.0.0.1:8080/"
    assert payload["port"] == 8080


def test_finetune_port_state_url_is_none_when_not_running(tmp_path):
    """A stopped studio advertises no URL at all, only the port it would try."""
    studio = _FakeStudio(resolved_port=8080)
    gc = _controller(tmp_path, studio=studio)
    _install_studio(tmp_path)

    gc._do_finetune_start()
    gc._do_finetune_stop()
    payload = gc.finetune_state()
    assert payload["status"] == "stopped"
    assert payload["url"] is None
    assert payload["port"] == 8501


def test_finetune_port_open_refuses_when_no_port_was_resolved(tmp_path):
    """Fail-closed: no resolved port means no browser navigation, ever.

    This is the trust tail SEC-M13-2 named - the Open button must never be able
    to reach a port some foreign process happens to hold.
    """
    studio = _FakeStudio(resolved_port=None)
    gc = _controller(tmp_path, studio=studio)
    _install_studio(tmp_path)
    gc._do_finetune_start()
    # Force the status past the running gate so ONLY the port guard can refuse.
    gc._finetune_status = "running"
    while True:
        try:
            gc.result_q.get_nowait()
        except queue.Empty:
            break

    with mock.patch.object(gui_controller.webbrowser, "open") as opened:
        gc._do_finetune_open()
    assert opened.call_count == 0
    last = gc.result_q.get_nowait()
    while True:
        try:
            last = gc.result_q.get_nowait()
        except queue.Empty:
            break
    assert last.ok is False
    assert "not running yet" in (last.error or "")


def test_finetune_port_open_uses_the_resolved_port(tmp_path):
    """The happy path opens exactly the reassigned port the studio bound."""
    studio = _FakeStudio(resolved_port=8080)
    gc = _controller(tmp_path, studio=studio)
    _install_studio(tmp_path)
    gc._do_finetune_start()

    with mock.patch.object(gui_controller.webbrowser, "open") as opened:
        gc._do_finetune_open()
    opened.assert_called_once_with("http://127.0.0.1:8080/")


def test_finetune_port_exhaustion_is_an_honest_remedy_not_a_traceback(tmp_path):
    """PortUnavailableError becomes an actionable message, never a raw exception."""

    class _Refusing:
        resolved_port = None

        def start(self):
            raise PortUnavailableError("no free port in reserved range 8000-8099")

    gc = _controller(tmp_path, studio=_Refusing())
    _install_studio(tmp_path)
    gc._do_finetune_start()
    results = []
    while True:
        try:
            results.append(gc.result_q.get_nowait())
        except queue.Empty:
            break
    last = results[-1]
    assert last.ok is False
    assert "No free loopback port" in (last.error or "")
    assert "ports.range_end" in (last.error or "")
    assert "Traceback" not in (last.error or "")


# --------------------------------------------------------------------------- #
# Log-path persistence (defect D-M13-2, UX Spec section 2 Panel A)
# --------------------------------------------------------------------------- #


def test_finetune_log_path_absent_until_a_start_is_attempted(tmp_path):
    """Before any Start this session there is no log file to point at."""
    gc = _controller(tmp_path, studio=_FakeStudio())
    _install_studio(tmp_path)
    assert gc.finetune_state()["log_path"] == ""


def test_finetune_log_path_survives_stop(tmp_path):
    """UX Spec Panel A: the line stays once started this session, not only while
    Running. The log file outlives the process, so Stop must not hide it."""
    gc = _controller(tmp_path, studio=_FakeStudio())
    _install_studio(tmp_path)

    gc._do_finetune_start()
    running_path = gc.finetune_state()["log_path"]
    assert running_path.endswith("finetune_studio.log")

    gc._do_finetune_stop()
    state = gc.finetune_state()
    assert state["status"] == "stopped"
    assert state["log_path"] == running_path


def test_finetune_log_path_appears_after_a_failed_start(tmp_path):
    """A start that spawned a child but never went ready still wrote the log."""
    gc = _controller(tmp_path, studio=_FakeStudio(status=ServiceStatus.STOPPED_ERROR))
    _install_studio(tmp_path)

    gc._do_finetune_start()
    assert gc.finetune_state()["status"] == "error"
    assert gc.finetune_state()["log_path"].endswith("finetune_studio.log")


def test_finetune_log_path_stays_hidden_when_no_child_was_spawned(tmp_path):
    """Port exhaustion fails before launch, so there is no log to advertise."""

    class _Refusing:
        resolved_port = None

        def start(self):
            raise PortUnavailableError("no free port in reserved range 8000-8099")

    gc = _controller(tmp_path, studio=_Refusing())
    _install_studio(tmp_path)
    gc._do_finetune_start()
    assert gc.finetune_state()["log_path"] == ""


def test_finetune_stop_warns_only_when_a_run_is_live(tmp_path):
    """The limitation warning rides the stop that happens during a live run."""
    studio = _FakeStudio()
    gc = _controller(tmp_path, studio=studio)
    _install_studio(tmp_path)
    root = tmp_path / "finetune" / "outputs" / "live"
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.log").write_text("step 1\n", encoding="utf-8")
    gc._finetune_started = True

    gc._do_finetune_stop()
    result = gc.result_q.get_nowait()
    assert result.payload["warning"] == finetune.orphan_warning_text()

    # No live run -> a plain Stopped state, never a blanket disclaimer.
    gc._finetune_started = True
    stale = (root / "train.log").stat().st_mtime - 10_000
    os.utime(root / "train.log", (stale, stale))
    gc._do_finetune_stop()
    result = gc.result_q.get_nowait()
    assert result.payload.get("warning", "") == ""
    assert result.payload["status"] == "stopped"


def test_finetune_stop_warns_on_a_log_written_this_instant(tmp_path):
    """The stop-time warning must not be lost to filesystem/wall-clock skew.

    Regression test for the flake that made this criterion nondeterministic: a
    train.log whose mtime lands ahead of time.time() is the freshest possible
    run, and must produce the warning rather than a bare clean-stop claim.
    """
    studio = _FakeStudio()
    gc = _controller(tmp_path, studio=studio)
    _install_studio(tmp_path)
    root = tmp_path / "finetune" / "outputs" / "live"
    root.mkdir(parents=True, exist_ok=True)
    log = root / "train.log"
    log.write_text("step 1\n", encoding="utf-8")
    ahead = time.time() + 5.0
    os.utime(log, (ahead, ahead))
    gc._finetune_started = True

    gc._do_finetune_stop()
    result = gc.result_q.get_nowait()
    assert result.payload["warning"] == finetune.orphan_warning_text()


def test_finetune_run_active_reads_a_cache_and_never_touches_the_disk(tmp_path):
    """The GUI-thread accessor does no I/O; only the worker probe walks the tree.

    Architecture M13.7.4: no blocking work reaches the Qt thread. Window close
    calls finetune_run_active(), so it must answer from the cached probe even if
    the outputs root would stall (failure mode F7, disconnected drive).
    """
    gc = _controller(tmp_path)
    _install_studio(tmp_path)
    root = tmp_path / "finetune" / "outputs" / "live"
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.log").write_text("step 1\n", encoding="utf-8")
    gc._finetune_started = True

    # Nothing has probed yet, so the cached answer is still the safe default.
    assert gc.finetune_run_active() is False

    # Make any filesystem access explode: the accessor must not perform one.
    def _boom(*args, **kwargs):
        raise AssertionError("finetune_run_active() touched the filesystem")

    with mock.patch.object(finetune, "active_run", _boom):
        assert gc.finetune_run_active() is False

    # The worker probe is what fills the cache, and the accessor then reports it.
    assert gc._probe_finetune_run_active() is True
    assert gc.finetune_run_active() is True


def test_finetune_state_is_available_as_a_queued_command(tmp_path):
    """The state snapshot is reachable off the GUI thread, as a Command."""
    gc = _controller(tmp_path)
    _install_studio(tmp_path)
    gc.request_finetune_state()
    command = gc.command_q.get_nowait()
    assert command.kind == "finetune_state"

    gc._dispatch(command)
    result = gc.result_q.get_nowait()
    assert result.kind == "finetune_state"
    assert result.payload["status"] == "stopped"
    assert result.payload["run_active"] is False


def test_finetune_scan_command_publishes_rows(tmp_path):
    """The scan command answers with pre-formatted rows for the view."""
    gguf = tmp_path / "finetune" / "outputs" / "DemoGPT-v5" / "DemoGPT-v5.q4_k_m.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"GGUF" + b"0" * 2048)
    gc = _controller(tmp_path)

    gc._do_finetune_scan()
    result = gc.result_q.get_nowait()
    assert result.kind == "finetune_models"
    row = result.payload["items"][0]
    assert row["name"] == "DemoGPT-v5"
    assert row["quant"] == "Q4_K_M"
    assert row["size_display"].endswith("GB")
    assert row["modified_display"] != "-"


def test_finetune_register_command_writes_one_real_row(tmp_path):
    """Register writes a genuine models.yaml row and reports the registered id."""
    gguf = tmp_path / "finetune" / "outputs" / "DemoGPT-v5" / "DemoGPT-v5.q4_k_m.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"GGUF" + b"0" * 2048)
    (tmp_path / "models.yaml").write_text("version: 1\nmodels: []\n", encoding="utf-8")
    gc = _controller(tmp_path)
    gc.refresh_models = lambda: None  # the fixture has no full registry to reload

    gc._do_finetune_register("ft:DemoGPT-v5")
    result = gc.result_q.get_nowait()
    assert result.ok is True
    assert result.payload["model_id"] == "DemoGPT-v5"
    text = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    # SEC-M13-5: the id is emitted double-quoted, like every other string scalar.
    assert 'id: "DemoGPT-v5"' in text


def test_finetune_register_collision_is_worded_and_writes_nothing(tmp_path):
    """A colliding id yields the worded refusal and leaves the file untouched."""
    gguf = tmp_path / "finetune" / "outputs" / "DemoGPT-v5" / "DemoGPT-v5.q4_k_m.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"GGUF" + b"0" * 2048)
    original = (
        "version: 1\n"
        "models:\n"
        "  - id: DemoGPT-v5\n"
        '    name: "Taken"\n'
        '    location: "/locitize-test/models/x.gguf"\n'
        "    context_size: 8192\n"
        "    gpu_layers: 999\n"
    )
    (tmp_path / "models.yaml").write_text(original, encoding="utf-8")
    gc = _controller(tmp_path)

    gc._do_finetune_register("ft:DemoGPT-v5")
    result = gc.result_q.get_nowait()
    assert result.ok is False
    assert "already exists" in (result.error or "")
    assert (tmp_path / "models.yaml").read_text(encoding="utf-8") == original


def test_finetune_discovered_model_edits_are_refused(tmp_path):
    """Editing or renaming a discovered model is refused with the exact message."""
    gc = _controller(tmp_path)
    gc.save_model_edits("ft:DemoGPT-v5", "999", "8192")
    result = gc.result_q.get_nowait()
    assert result.ok is False
    assert result.error == gui_controller.DISCOVERED_EDIT_REFUSAL

    gc.save_model_identity("ft:DemoGPT-v5", "new-id", "New Name")
    result = gc.result_q.get_nowait()
    assert result.ok is False
    assert result.error == gui_controller.DISCOVERED_EDIT_REFUSAL


def test_finetune_discovered_benchmark_throughput_is_report_persistent(tmp_path):
    """Discovered-model throughput persists in JSONL without yaml registration."""
    gc = _controller(tmp_path)
    gc._benchmark_fn = lambda model_id: (True, "1 ok, 0 failed of 1 scenarios", 31.5)
    gc._do_benchmark("ft:DemoGPT-v5")
    result = gc.result_q.get_nowait()
    assert result.payload["score_persisted"] is True
    assert result.payload["note"] == ""


# --------------------------------------------------------------------------- #
# models.py: discovery is additive, and a discovered path is confined at serve time
# --------------------------------------------------------------------------- #


def test_finetune_registry_merges_without_changing_existing_queries(tmp_path):
    """all()/installed()/launchable() are untouched; all_merged() adds the rest."""
    from config import ModelRegistryData
    from models import ModelRegistry

    gguf = tmp_path / "finetune" / "outputs" / "DemoGPT-v5" / "DemoGPT-v5.q4_k_m.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"GGUF" + b"0" * 2048)
    manual = Model(
        id="manual",
        name="Manual",
        description="",
        location=str(tmp_path / "manual.gguf"),
        context_size=8192,
        gpu_layers=999,
    )
    registry = ModelRegistry(ModelRegistryData(models=[manual]), _settings(tmp_path))

    assert [m.id for m in registry.all()] == ["manual"]
    assert [m.id for m in registry.installed()] == ["manual"]
    assert [m.id for m in registry.launchable()] == ["manual"]
    merged = [m.id for m in registry.all_merged()]
    assert merged == ["manual", "ft:DemoGPT-v5"]
    assert registry.get("ft:DemoGPT-v5") is not None


def test_finetune_serve_path_guard_rejects_a_path_outside_the_root(tmp_path):
    """A discovered location outside the outputs root never reaches an argv."""
    settings = _settings(tmp_path)
    outside = tmp_path / "elsewhere.gguf"
    outside.write_bytes(b"GGUF")
    with pytest.raises(ValueError):
        finetune.resolve_serve_path(str(outside), settings)


# --------------------------------------------------------------------------- #
# launcher: the new --terminal flag and its precedence
# --------------------------------------------------------------------------- #


def test_finetune_launcher_terminal_beats_desktop():
    """--terminal wins when combined with --desktop/--gui, per M13.2.2."""
    from launcher import _parse_args

    args = _parse_args(["--terminal", "--desktop"])
    assert args.terminal is True and args.desktop is True
    bare = _parse_args([])
    # Bare invocation is unchanged: the terminal menu, with no forced flag.
    assert bare.terminal is False and bare.desktop is False and bare.gui is False


def test_finetune_only_one_launcher_shortcut_remains():
    """The five extra .bat shortcuts are gone; locitize.bat is the single door."""
    platform_dir = Path(__file__).resolve().parent.parent
    bats = sorted(p.name for p in platform_dir.glob("*.bat"))
    assert bats == ["locitize.bat"]
    assert "--desktop" in (platform_dir / "locitize.bat").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The Qt Fine-tune page (offscreen; skipped cleanly without PySide6)
# --------------------------------------------------------------------------- #


@pytest.fixture()
def qapp():
    pytest.importorskip("PySide6")
    import desktop
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app, desktop


class _ViewController:
    """The view-facing controller contract, recording intents."""

    def __init__(self):
        self.result_q = queue.Queue()
        self.command_q = queue.Queue()
        self.calls = []
        self.run_active = False
        self.state_calls = 0

    def list_models(self):
        return self.list_models_merged()

    def list_models_merged(self):
        return [
            {
                "id": "manual", "name": "Manual", "location": "/locitize-test/models/m.gguf",
                "status": "installed", "gpu_layers": 999, "context_size": 8192,
                "launchable": True, "size_bytes": 1_000_000_000,
                "size_display": "1.0 GB", "vram_estimate_mb": 0, "vram_need_mb": 1000,
                "gpu_portion_display": "1.0 GB", "cpu_portion_display": "-",
                "benchmark_score": None, "score_display": "-",
                "source": "registry", "source_display": "Registered",
            },
            {
                "id": "ft:DemoGPT-v5", "name": "DemoGPT-v5",
                "location": "/locitize-test/outputs/DemoGPT-v5/DemoGPT-v5.q4_k_m.gguf",
                "status": "installed", "gpu_layers": 999, "context_size": 8192,
                "launchable": True, "size_bytes": 986_047_968,
                "size_display": "1.0 GB", "vram_estimate_mb": 0, "vram_need_mb": 986,
                "gpu_portion_display": "1.0 GB", "cpu_portion_display": "-",
                "benchmark_score": None, "score_display": "-",
                "source": "discovered", "source_display": "Discovered",
            },
        ]

    def system_specs(self):
        return gui_controller.SystemSpecs(32768.0, "Fake GPU", 16303.0)

    def available_voices(self):
        return ["af_heart"]

    def finetune_state(self):
        # Counted so a test can prove the view never calls this synchronously:
        # on the real controller it walks the outputs tree (Architecture M13.7.4).
        self.state_calls += 1
        return {"status": "stopped", "url": None, "port": 8501, "reason": "",
                "log_path": "", "run_active": False}

    def finetune_run_active(self):
        return self.run_active

    def finetune_warning_text(self):
        return finetune.orphan_warning_text()

    def start_threads(self):
        self.calls.append(("start_threads",))

    def shutdown(self):
        self.calls.append(("shutdown",))

    # Consulted by the Stop button's enablement rule on every refresh; this
    # suite never runs an auto-tune, so Stop keeps its ordinary meaning.
    def autotune_in_progress(self):
        return False

    def autotune_model_id(self):
        return None

    def cancel_autotune(self):
        return False

    def __getattr__(self, name):
        # Any other request_* intent records itself, so the view can be exercised
        # without restating the whole controller surface here.
        if name.startswith("request_") or name in ("open_chat", "refresh_models"):
            def recorder(*args):
                self.calls.append((name, *args))
                return self.list_models_merged() if name == "refresh_models" else None

            return recorder
        raise AttributeError(name)


def test_finetune_page_is_reachable_and_renders_each_state(qapp):
    """The Fine-tune sidebar row exists, and each canned state renders."""
    app, desktop = qapp
    controller = _ViewController()
    window = desktop.MainWindow(controller, health="OK")

    # Order-independent: a new page (System) may sit between Models and Fine-tune.
    assert "Fine-tune" in desktop.PAGE_NAMES
    assert "Fine-tune" in window.pages

    window._apply_finetune_state({"status": "stopped", "reason": "", "url": None})
    assert window._ft_chip.text() == "Stopped"
    assert window._ft_start_btn.isEnabled()
    assert not window._ft_stop_btn.isEnabled()
    assert not window._ft_open_btn.isEnabled()

    window._apply_finetune_state({"status": "starting", "reason": "", "url": None})
    assert window._ft_chip.text() == "Starting studio..."
    # In flight: all three lifecycle controls are disabled together (no race).
    assert not any(
        button.isEnabled()
        for button in (window._ft_start_btn, window._ft_stop_btn, window._ft_open_btn)
    )

    window._apply_finetune_state(
        {"status": "running", "reason": "", "url": "http://127.0.0.1:8501/"}
    )
    assert window._ft_chip.text() == "Running - http://127.0.0.1:8501/"
    assert window._ft_stop_btn.isEnabled() and window._ft_open_btn.isEnabled()

    window._apply_finetune_state(
        {"status": "disabled", "reason": "Fine-tune studio is not configured.",
         "url": None}
    )
    assert "not configured" in window._ft_chip.text()
    assert not window._ft_start_btn.isEnabled()

    window.closeEvent(desktop.QtGui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_finetune_page_renders_rows_and_the_empty_reason(qapp):
    """Populated and empty states both render honestly."""
    app, desktop = qapp
    controller = _ViewController()
    window = desktop.MainWindow(controller, health="OK")

    window._apply_finetune_models(
        gui_controller.Result(
            "finetune_models",
            True,
            {
                "items": [
                    {
                        "id": "ft:DemoGPT-v5", "name": "DemoGPT-v5",
                        "run": "DemoGPT-v5", "quant": "Q4_K_M",
                        "size_bytes": 986_047_968, "size_display": "1.0 GB",
                        "mtime": 1.0, "modified_display": "2026-08-16 05:08",
                        "path": "/locitize-test/outputs/DemoGPT-v5/DemoGPT-v5.q4_k_m.gguf",
                        "base_model": "unknown", "meta_display": "metadata: none",
                        "registered": False,
                    }
                ],
                "root": "/locitize-test/outputs",
                "reason": "",
            },
        )
    )
    assert window._ft_table.rowCount() == 1
    assert window._ft_table.item(0, 0).text() == "DemoGPT-v5"
    assert window._ft_table.item(0, 2).text() == "Q4_K_M"
    assert window._ft_table.item(0, 4).text() == "2026-08-16 05:08"

    window._apply_finetune_models(
        gui_controller.Result(
            "finetune_models", True,
            # The reason string is the shipped wording (finetune._scan_uncached),
            # not a paraphrase: the page renders it verbatim, so a stub that read
            # differently from the product would hide a change in what the user
            # is actually told.
            {"items": [], "root": "/locitize-test/outputs",
             "reason": "No fine-tuned models found in `/locitize-test/outputs`. If "
                       "your training runs are somewhere else, set "
                       "finetune.outputs_dir in settings.yaml to that folder."},
        )
    )
    assert window._ft_table.rowCount() == 0
    status = window._ft_status.text()
    assert "No fine-tuned models found" in status
    # The empty state reaches the user with both facts intact (round 6, MEDIUM-6).
    assert "/locitize-test/outputs" in status and "finetune.outputs_dir" in status

    window.closeEvent(desktop.QtGui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_finetune_models_page_shows_source_and_gates_register(qapp):
    """The Models table gains a Source column; Register only lights for discovered."""
    app, desktop = qapp
    controller = _ViewController()
    window = desktop.MainWindow(controller, health="OK")

    headers = [
        window._model_table.horizontalHeaderItem(i).text()
        for i in range(window._model_table.columnCount())
    ]
    assert headers[:4] == ["Model", "Capabilities", "Source", "Size"]
    assert window._model_table.item(0, 2).text() == "Registered"
    assert window._model_table.item(1, 2).text() == "Discovered"

    window._model_table.setCurrentCell(0, 0)
    assert not window._register_btn.isEnabled()
    window._model_table.setCurrentCell(1, 0)
    assert window._register_btn.isEnabled()
    window._register_btn.click()
    assert ("request_register_discovered", "ft:DemoGPT-v5") in controller.calls

    window.closeEvent(desktop.QtGui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_finetune_first_paint_asks_the_worker_for_the_state(qapp):
    """First paint enqueues the state request instead of computing it inline.

    Architecture M13.7.4: the state snapshot walks the studio checkout and the
    outputs tree, so it must never run on the Qt thread at window construction.
    """
    app, desktop = qapp
    controller = _ViewController()
    window = desktop.MainWindow(controller, health="OK")

    assert ("request_finetune_state",) in controller.calls
    assert ("request_scan_finetunes",) in controller.calls
    assert controller.state_calls == 0

    window.closeEvent(desktop.QtGui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_finetune_discovered_throughput_cell_is_a_persisted_measurement(qapp):
    """A discovered row displays report-persisted generation throughput."""
    app, desktop = qapp
    controller = _ViewController()
    window = desktop.MainWindow(controller, health="OK")
    window._apply_benchmark_result(
        gui_controller.Result(
            "benchmark", True,
            {"model_id": "ft:DemoGPT-v5", "score": 31.5,
             "score_display": "31.5 tok/s", "detail": "31.5 tok/s generation",
             "score_persisted": True, "note": ""},
        )
    )
    discovered_row = next(
        row for row in range(window._model_table.rowCount())
        if window._model_table.item(row, 0).data(
            desktop.QtCore.Qt.ItemDataRole.UserRole
        ) == "ft:DemoGPT-v5"
    )
    score_cell = window._model_table.item(discovered_row, 6)
    assert score_cell.text() == "31.5 tok/s"
    assert score_cell.toolTip() == ""
    assert "31.5 tok/s generation" in window._model_status.text()

    # A registered row's score IS persisted, so it carries no note at all.
    window._apply_benchmark_result(
        gui_controller.Result(
            "benchmark", True,
            {"model_id": "manual", "score": 44.0, "score_display": "44.0 tok/s",
             "detail": "44.0 tok/s", "score_persisted": True, "note": ""},
        )
    )
    registered_row = next(
        row for row in range(window._model_table.rowCount())
        if window._model_table.item(row, 0).data(
            desktop.QtCore.Qt.ItemDataRole.UserRole
        ) == "manual"
    )
    assert window._model_table.item(registered_row, 6).text() == "44.0 tok/s"
    assert window._model_table.item(registered_row, 6).toolTip() == ""

    window.closeEvent(desktop.QtGui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_finetune_row_reads_registered_after_a_successful_register(qapp):
    """UX Spec section 4 step 3: the row updates and Register stays disabled."""
    app, desktop = qapp
    controller = _ViewController()
    window = desktop.MainWindow(controller, health="OK")
    row = {
        "id": "ft:DemoGPT-v5", "name": "DemoGPT-v5", "run": "DemoGPT-v5",
        "quant": "Q4_K_M", "size_bytes": 986_047_968, "size_display": "1.0 GB",
        "mtime": 1.0, "modified_display": "2026-08-16 05:08",
        "path": "/locitize-test/outputs/DemoGPT-v5/DemoGPT-v5.q4_k_m.gguf",
        "base_model": "unknown", "meta_display": "metadata: none",
        "registered": False,
    }
    window._apply_finetune_models(
        gui_controller.Result(
            "finetune_models", True,
            {"items": [row], "root": "/locitize-test/outputs", "reason": ""},
        )
    )
    window._ft_table.setCurrentCell(0, 0)
    assert window._ft_table.item(0, 5).text() == "Not registered"
    assert window._ft_register_btn.isEnabled()

    window._apply_finetune_register_result(
        gui_controller.Result(
            "finetune_register_result", True,
            {"key": "ft:DemoGPT-v5", "model_id": "demogpt-v5"},
        )
    )
    assert window._ft_table.item(0, 5).text() == "Registered"
    assert window._finetunes[0]["registered"] is True
    assert not window._ft_register_btn.isEnabled()
    assert not window._register_btn.isEnabled()
    assert "Rescan" in window._ft_status.text()

    window.closeEvent(desktop.QtGui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_finetune_register_failure_hands_the_controls_back(qapp):
    """A refused Register leaves the row alone and re-enables the buttons."""
    app, desktop = qapp
    controller = _ViewController()
    window = desktop.MainWindow(controller, health="OK")
    window._apply_finetune_models(
        gui_controller.Result(
            "finetune_models", True,
            {"items": [{
                "id": "ft:DemoGPT-v5", "name": "DemoGPT-v5", "run": "DemoGPT-v5",
                "quant": "Q4_K_M", "size_bytes": 1, "size_display": "1.0 GB",
                "mtime": 1.0, "modified_display": "2026-08-16 05:08",
                "path": "/locitize-test/outputs/DemoGPT-v5/DemoGPT-v5.q4_k_m.gguf",
                "base_model": "unknown", "meta_display": "metadata: none",
                "registered": False,
            }], "root": "/locitize-test/outputs", "reason": ""},
        )
    )
    window._ft_table.setCurrentCell(0, 0)
    window._apply_finetune_register_result(
        gui_controller.Result(
            "finetune_register_result", False,
            {"key": "ft:DemoGPT-v5", "model_id": "demogpt-v5"},
            error="Could not register: an entry named 'demogpt-v5' already exists",
        )
    )
    assert window._ft_table.item(0, 5).text() == "Not registered"
    assert window._finetunes[0].get("registered") is False
    assert window._ft_register_btn.isEnabled()
    assert "already exists" in window._ft_status.text()

    window.closeEvent(desktop.QtGui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_finetune_window_close_warns_when_a_run_is_live(qapp, monkeypatch):
    """Closing during a live run shows a blocking dialog the owner must acknowledge."""
    app, desktop = qapp
    controller = _ViewController()
    controller.run_active = True
    window = desktop.MainWindow(controller, health="OK")

    shown = {}

    def fake_exec(box):
        shown["text"] = box.text()
        return desktop.QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(desktop.QtWidgets.QMessageBox, "exec", fake_exec)
    window.closeEvent(desktop.QtGui.QCloseEvent())
    assert shown["text"] == finetune.orphan_warning_text()
    assert ("shutdown",) in controller.calls

    window.deleteLater()
    app.processEvents()


def test_finetune_window_close_is_silent_with_no_run(qapp, monkeypatch):
    """No live run means no warning: the notice is targeted, not a blanket one."""
    app, desktop = qapp
    controller = _ViewController()
    controller.run_active = False
    window = desktop.MainWindow(controller, health="OK")

    calls = []
    monkeypatch.setattr(
        desktop.QtWidgets.QMessageBox,
        "exec",
        lambda box: calls.append(box.text()),
    )
    window.closeEvent(desktop.QtGui.QCloseEvent())
    assert calls == []

    window.deleteLater()
    app.processEvents()


# --------------------------------------------------------------------------- #
# Launcher routing: --terminal really reaches the menu, not just the parser
# --------------------------------------------------------------------------- #


def test_finetune_launcher_terminal_flag_skips_the_window(monkeypatch):
    """--terminal + --desktop runs the terminal path; neither window is opened.

    Parser-level precedence is not enough: this drives Launcher.run itself, with
    both view entry points monkeypatched, so a future refactor that reorders the
    branches is caught here rather than by the owner at the shortcut.
    """
    from launcher import Launcher
    from test_launcher import _deps, _pass_report

    lines = []
    opened = []
    launcher = Launcher(deps=_deps(_pass_report(), lines))
    monkeypatch.setattr(
        launcher, "_run_desktop", lambda settings, models: opened.append("desktop") or 31
    )
    monkeypatch.setattr(
        launcher, "_run_gui", lambda settings, models: opened.append("tk") or 32
    )
    menu = []
    monkeypatch.setattr(
        launcher, "_menu_loop", lambda *args, **kwargs: menu.append("menu"), raising=False
    )

    launcher.run(["--terminal", "--desktop"])

    assert opened == []
