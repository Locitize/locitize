"""Launcher routing tests for the M12 Qt default and one-milestone Tk fallback."""

import atexit
import builtins
import os
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

import gui_controller
import pytest
from config import Settings
import launcher
from launcher import Launcher
from services import ServiceManager
from test_launcher import _deps, _models, _pass_report, _StubChecker


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("--desktop", "desktop"),
        ("--gui", "desktop"),
        ("--gui-tk", "tk"),
    ],
)
def test_m12_gui_flags_route_to_the_expected_view(monkeypatch, flag, expected):
    """Canonical and compatibility flags choose Qt; only the hidden hatch uses Tk."""
    lines = []
    calls = []
    launcher = Launcher(deps=_deps(_pass_report(), lines))
    monkeypatch.setattr(
        launcher,
        "_run_desktop",
        lambda settings, models: calls.append("desktop") or 31,
    )
    monkeypatch.setattr(
        launcher,
        "_run_gui",
        lambda settings, models: calls.append("tk") or 32,
    )

    code = launcher.run([flag])

    assert calls == [expected]
    assert code == (31 if expected == "desktop" else 32)


def test_gui_shortcut_prefers_repository_venv_and_routes_to_qt():
    """The single shortcut resolves the Qt venv before any fallback.

    M13 collapsed the five launcher shortcuts into LOCITIZE.bat, which now carries
    the implicit --desktop, so this M12 guarantee (venv resolution order, Qt not
    Tk) is asserted against that one remaining file.
    """
    shortcut = Path(__file__).parents[1] / "LOCITIZE.bat"
    text = shortcut.read_text(encoding="utf-8").lower()

    parent_venv = 'if exist "..\\.venv\\scripts\\python.exe"'
    local_venv = 'else if exist ".venv\\scripts\\python.exe"'
    assert text.index(parent_venv) < text.index(local_venv)
    assert 'set "py=..\\.venv\\scripts\\python.exe"' in text
    assert '"%py%" launcher.py --desktop' in text
    assert "tkinter" not in text


@pytest.mark.skipif(os.name != "nt", reason="LOCITIZE.bat is Windows-only")
def test_gui_shortcut_executes_parent_venv_python_with_gui_flag(tmp_path):
    """A hermetic batch launch records the chosen executable and never opens Qt."""
    source = Path(__file__).parents[1] / "LOCITIZE.bat"
    platform_dir = tmp_path / "platform"
    scripts_dir = tmp_path / ".venv" / "Scripts"
    platform_dir.mkdir()
    # Build a real no-pip venv so the batch file exercises Windows interpreter
    # resolution instead of relying on a static text assertion alone.
    venv.EnvBuilder(with_pip=False).create(tmp_path / ".venv")
    shortcut = platform_dir / source.name
    shortcut.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    probe = tmp_path / "launch-probe.txt"
    (platform_dir / "launcher.py").write_text(
        "# Temporary launcher probe used only by this test.\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['LOCITIZE_LAUNCH_PROBE']).write_text(\n"
        "    sys.executable + '\\n' + ' '.join(sys.argv[1:]), encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["LOCITIZE_LAUNCH_PROBE"] = str(probe)

    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(shortcut)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    selected_python, flag = probe.read_text(encoding="utf-8").splitlines()
    assert Path(selected_python).resolve() == (scripts_dir / "python.exe").resolve()
    assert flag == "--desktop"


@pytest.mark.parametrize("choice", ["gui", "desktop"])
def test_menu_handoff_stops_terminal_manager_before_desktop(monkeypatch, tmp_path, choice):
    """Both menu spellings transfer only after the terminal manager is stopped."""
    settings = Settings()
    settings.base_dir = tmp_path
    settings.launcher.auto_journal = False
    inputs = iter([choice])

    def config_load(_base=None, env=None):
        return settings, _models(), []

    deps = {
        "config_load": config_load,
        "health_checker": _StubChecker(_pass_report()),
        "output_fn": lambda _line: None,
        "input_fn": lambda _prompt: next(inputs),
    }
    stop_ids = []
    real_stop_all = ServiceManager.stop_all

    def recording_stop_all(manager):
        stop_ids.append(id(manager))
        return real_stop_all(manager)

    monkeypatch.setattr(ServiceManager, "stop_all", recording_stop_all)
    launcher = Launcher(deps=deps)

    def fake_desktop(_settings, _models_data):
        assert id(launcher._service_manager) in stop_ids
        return 44

    monkeypatch.setattr(launcher, "_run_desktop", fake_desktop)

    assert launcher.run([]) == 44
    assert stop_ids.count(id(launcher._service_manager)) == 1


class _RecordingManager:
    """Minimal shared manager recorder for desktop construction tests."""

    def __init__(self):
        self.stop_count = 0

    def stop_all(self):
        self.stop_count += 1


def _patch_desktop_builders(monkeypatch, launcher, manager):
    """Install deterministic collaborators around the desktop import boundary."""
    registry = object()
    controller = object()
    whisper = object()
    monkeypatch.setattr(
        launcher,
        "_build_controller",
        lambda _settings, _models_data: (registry, manager, controller),
    )

    def build_whisper(_settings, supplied_manager):
        assert supplied_manager is manager
        return whisper

    monkeypatch.setattr(launcher, "_build_whisper_controller", build_whisper)
    monkeypatch.setattr(launcher, "_run_ladder", lambda _settings, _models_data: _pass_report())
    return registry, controller, whisper


def test_run_desktop_wires_one_shared_manager_and_final_cleanup(monkeypatch):
    """The Qt view receives the full GuiController contract and one manager owner."""
    lines = []
    launcher = Launcher(deps=_deps(_pass_report(), lines))
    manager = _RecordingManager()
    registry, controller, whisper = _patch_desktop_builders(monkeypatch, launcher, manager)
    constructed = []
    registered = []

    class FakeGuiController:
        """Capture constructor collaborators without starting worker threads."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            constructed.append(self)

    def run_view(view_controller, health):
        assert view_controller is constructed[0]
        assert health == "PASS"
        assert manager.stop_count == 0
        return 73

    monkeypatch.setattr(gui_controller, "GuiController", FakeGuiController)
    monkeypatch.setattr(atexit, "register", lambda callback: registered.append(callback))
    monkeypatch.setitem(sys.modules, "desktop", SimpleNamespace(run=run_view))
    settings = object()
    models_data = object()

    code = launcher._run_desktop(settings, models_data)

    assert code == 73
    assert len(constructed) == 1
    view_controller = constructed[0]
    assert view_controller.args == (
        settings,
        registry,
        controller,
        whisper,
        manager,
    )
    assert set(view_controller.kwargs) == {
        "listen_fn",
        "speak_fn",
        "openwebui_start_fn",
        "assistant_start_fn",
        "describe_fn",
        "memory_search_fn",
        "benchmark_fn",
        "benchmark_history_fn",
        "gpu_provider",
        "sys_provider",
        "apply_noise_suppression_fn",
        "second_eye_fn",
    }
    assert getattr(registered[0], "__self__", None) is manager
    assert launcher._service_manager is manager
    assert manager.stop_count == 1


@pytest.mark.parametrize(
    ("import_error", "expected", "forbidden"),
    [
        (
            ModuleNotFoundError("No module named 'PySide6'", name="PySide6"),
            "PySide6 is not installed",
            "missing module 'PySide6'",
        ),
        (
            ValueError("broken desktop module"),
            "ValueError: broken desktop module",
            "PySide6 is not installed",
        ),
    ],
)
def test_run_desktop_import_failures_are_accurate_and_clean(
    monkeypatch, import_error, expected, forbidden
):
    """Import failures always clean the manager and only missing Qt gets its remedy."""
    lines = []
    launcher = Launcher(deps=_deps(_pass_report(), lines))
    manager = _RecordingManager()
    _patch_desktop_builders(monkeypatch, launcher, manager)
    monkeypatch.setattr(atexit, "register", lambda _callback: None)
    monkeypatch.delitem(sys.modules, "desktop", raising=False)
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "desktop":
            raise import_error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    assert launcher._run_desktop(object(), object()) == 2
    assert manager.stop_count == 1
    output = "\n".join(lines)
    assert expected in output
    assert forbidden not in output
    # D-M13-4: bare LOCITIZE.bat now implies --desktop, so the remedy must name the
    # flag that actually reaches the terminal menu, not the thing that just failed.
    assert "LOCITIZE.bat --terminal" in output
    assert "use LOCITIZE.bat for the terminal menu" not in output


def test_vbs_launcher_ships_and_delegates_to_the_bat():
    """M17.12: the official no-console launcher exists next to LOCITIZE.bat,
    delegates to it, and hides the console ONLY for a plain, already-set-up
    launch (args or a missing venv keep it visible for setup/terminal output)."""
    vbs = Path(__file__).parents[1] / "LOCITIZE.vbs"
    assert vbs.is_file()
    text = vbs.read_text(encoding="utf-8")
    assert "LOCITIZE.bat" in text          # reuses all batch launch logic
    assert "shell.Run" in text             # launches via WScript.Shell.Run
    # Hidden (style 0) only when there are no args AND a venv exists; else visible.
    assert "style = 0" in text
    assert "style = 1" in text
    assert "WScript.Arguments.Count = 0" in text


def test_create_desktop_shortcut_skips_when_vbs_absent(tmp_path, monkeypatch):
    """Best-effort: no LOCITIZE.vbs to link -> a skipped success, never a failure
    that would break setup (a missing shortcut costs nothing)."""
    import setup_env

    monkeypatch.setattr(setup_env, "BASE_DIR", tmp_path)  # no LOCITIZE.vbs here
    result = setup_env.create_desktop_shortcut()
    assert result.ok and result.skipped


# ---- Desktop .venv / PySide6 harden (fail loud, no silent Python311 die) ---- #


def test_desktop_venv_python_prefers_repo_root(tmp_path, monkeypatch):
    """Repo-root .venv wins over a platform-local .venv when both exist."""
    repo = tmp_path / "locitize"
    platform = repo / "platform"
    platform.mkdir(parents=True)
    root_scripts = repo / ".venv" / "Scripts"
    local_scripts = platform / ".venv" / "Scripts"
    root_scripts.mkdir(parents=True)
    local_scripts.mkdir(parents=True)
    root_py = root_scripts / "python.exe"
    local_py = local_scripts / "python.exe"
    root_py.write_text("", encoding="utf-8")
    local_py.write_text("", encoding="utf-8")
    monkeypatch.setattr(launcher, "__file__", str(platform / "launcher.py"))
    assert launcher._desktop_venv_python() == root_py


def test_ensure_desktop_interpreter_ok_when_current_has_pyside6(monkeypatch):
    """Already-on-Qt interpreters proceed without re-exec."""
    lines = []
    monkeypatch.setattr(launcher, "_current_has_pyside6", lambda: True)
    calls = []
    monkeypatch.setattr(os, "execve", lambda *a, **k: calls.append((a, k)))
    assert launcher.ensure_desktop_interpreter(lines.append) is None
    assert calls == []
    assert lines == []


def test_ensure_desktop_interpreter_reexecs_into_venv_once(monkeypatch, tmp_path):
    """System Python without Qt re-execs once into .venv that has PySide6."""
    venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("", encoding="utf-8")
    monkeypatch.setattr(launcher, "_desktop_venv_python", lambda: venv_py)
    monkeypatch.setattr(launcher, "_current_has_pyside6", lambda: False)
    monkeypatch.setattr(launcher, "_venv_has_pyside6", lambda _p: True)
    monkeypatch.setattr(
        launcher, "_interpreters_equivalent", lambda _a, _b: False
    )
    monkeypatch.delenv(launcher._DESKTOP_REEXEC_ENV, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Python311" / "python.exe"))
    monkeypatch.setattr(sys, "argv", ["launcher.py", "--desktop"])
    recorded = {}

    def fake_execve(path, argv, env):
        recorded["path"] = path
        recorded["argv"] = list(argv)
        recorded["env_flag"] = env.get(launcher._DESKTOP_REEXEC_ENV)
        raise SystemExit(97)

    monkeypatch.setattr(os, "execve", fake_execve)
    lines = []
    with pytest.raises(SystemExit) as exc:
        launcher.ensure_desktop_interpreter(lines.append)
    assert exc.value.code == 97
    assert recorded["path"] == str(venv_py)
    assert recorded["argv"][0] == str(venv_py)
    assert "--desktop" in recorded["argv"]
    assert recorded["env_flag"] == "1"
    assert any("re-exec once into .venv" in line for line in lines)


def test_ensure_desktop_interpreter_fails_loud_when_venv_missing(monkeypatch, tmp_path):
    """No .venv and no PySide6: loud remedy, exit 2, no execve."""
    monkeypatch.setattr(launcher, "_desktop_venv_python", lambda: None)
    monkeypatch.setattr(launcher, "_current_has_pyside6", lambda: False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Python311" / "python.exe"))
    calls = []
    monkeypatch.setattr(os, "execve", lambda *a, **k: calls.append(a))
    lines = []
    assert launcher.ensure_desktop_interpreter(lines.append) == 2
    assert calls == []
    joined = "\n".join(lines)
    assert "PySide6" in joined
    assert "LOCITIZE.bat --terminal" in joined
    assert "Current interpreter:" in joined


def test_ensure_desktop_interpreter_no_reexec_loop(monkeypatch, tmp_path):
    """After one re-exec, missing Qt still fails loud (no infinite execve)."""
    venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("", encoding="utf-8")
    monkeypatch.setattr(launcher, "_desktop_venv_python", lambda: venv_py)
    monkeypatch.setattr(launcher, "_current_has_pyside6", lambda: False)
    monkeypatch.setattr(launcher, "_venv_has_pyside6", lambda _p: False)
    monkeypatch.setenv(launcher._DESKTOP_REEXEC_ENV, "1")
    calls = []
    monkeypatch.setattr(os, "execve", lambda *a, **k: calls.append(a))
    lines = []
    assert launcher.ensure_desktop_interpreter(lines.append) == 2
    assert calls == []
    assert "PySide6" in "\n".join(lines)


def test_run_desktop_fails_before_services_when_gate_blocks(monkeypatch):
    """Missing PySide6 must not build the manager or call stop_all."""
    lines = []
    launcher_obj = Launcher(deps=_deps(_pass_report(), lines))
    built = []

    def boom(*_a, **_k):
        built.append("controller")
        raise AssertionError("must not build controller when desktop gate fails")

    monkeypatch.setattr(launcher_obj, "_build_controller", boom)
    monkeypatch.setattr(launcher, "ensure_desktop_interpreter", lambda _out: 2)
    code = launcher_obj._run_desktop(object(), object())
    assert code == 2
    assert built == []
