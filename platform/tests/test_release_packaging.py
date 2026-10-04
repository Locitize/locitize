"""Exercise install integrity and user-data isolation without touching the desktop."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from release_info import VERSION
from runtime_layout import environment_root


def test_packaged_environment_is_versioned_and_source_layout_preserved(tmp_path, monkeypatch):
    code = tmp_path / "installation" / "platform"
    code.mkdir(parents=True)
    data = tmp_path / "user data"
    monkeypatch.setenv("LOCITIZE_DATA_DIR", str(data))
    assert environment_root(code) == code.parent
    runtime = code.parent / "runtime"
    runtime.mkdir()
    (runtime / "python.exe").touch()
    assert environment_root(code) == data / "environments" / VERSION
    assert not data.exists()


def test_setup_shortcut_uses_packaged_launcher_without_bootstrap_python(tmp_path, monkeypatch):
    import setup_env
    code = tmp_path / "platform"
    code.mkdir()
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "python.exe").touch()
    (tmp_path / "locitize.exe").touch()
    monkeypatch.setattr(setup_env, "BASE_DIR", code)
    calls = []
    monkeypatch.setattr(setup_env, "_run", lambda argv, **kw: (calls.append(argv) or (0, "")))
    assert setup_env.create_desktop_shortcut().ok
    script = calls[0][-1]
    assert "locitize.exe" in script and "wscript.exe" not in script


def test_packaged_uninstall_does_not_run_legacy_data_removal(monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    import desktop
    import runtime_layout
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(runtime_layout, "bundled_python", lambda: Path("runtime/python.exe"))
    notices = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: notices.append(args))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("spawned legacy uninstaller"))
    desktop.MainWindow._on_uninstall(None)
    assert notices and "Keep the separate locitize data folder" in notices[0][2]
    app.processEvents()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows installer")
@pytest.mark.parametrize("mutation", ["valid", "corrupt", "traversal"])
def test_installer_verifies_before_publication_and_preserves_previous_version(tmp_path, mutation):
    package = tmp_path / "extracted package"
    package.mkdir()
    script = package / "Install locitize.ps1"
    shutil.copy2(Path(__file__).parents[1] / "scripts" / "install_release.ps1", script)
    content = b"fixture application"
    (package / "locitize.exe").write_bytes(content)
    files = {"locitize.exe": hashlib.sha256(content).hexdigest()}
    if mutation == "corrupt":
        (package / "locitize.exe").write_bytes(b"tampered")
    if mutation == "traversal":
        files["../outside"] = "0" * 64
    (package / "release-manifest.json").write_text(json.dumps({"version": VERSION, "files": files}))
    target = tmp_path / "installed versions"
    older = target / "0.1"
    older.mkdir(parents=True)
    (older / "keep.txt").write_text("previous")
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
               "-InstallRoot", str(target), "-NoShortcuts"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert (older / "keep.txt").read_text() == "previous"
    if mutation == "valid":
        assert result.returncode == 0, result.stderr
        assert (target / VERSION / "locitize.exe").read_bytes() == content
        repeat = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert repeat.returncode != 0
        assert (target / VERSION / "locitize.exe").read_bytes() == content
        assert not list(target.glob(".staging-*"))
    else:
        assert result.returncode != 0
        assert not (target / VERSION).exists()
