"""Focused proof for the consumer Models-folder and Open WebUI install flows."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import config
import egress_log
import setup_env
from config import Settings
from openwebui_install import CommandOutcome, OpenWebUIInstaller, _friendly_failure


def _settings(tmp_path: Path) -> Settings:
    code = tmp_path / "Codebase" / "platform"
    code.mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    (data / "settings.yaml").write_text(
        "version: 2\n"
        "paths:\n  model_store: \"\"\n"
        "openwebui:\n  enabled: false\n"
        "models_hub:\n  download_dir: \"\"\n",
        encoding="utf-8",
    )
    settings = Settings()
    object.__setattr__(settings, "base_dir", code)
    object.__setattr__(settings, "data_dir", data)
    return settings


def test_model_store_writer_is_targeted_and_clears_legacy(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    old_store = tmp_path / "old-models"
    original = (
        "# owner comment\nversion: 2\npaths:\n"
        "  llama_cpp: \"\"\n  model_store: \"\" # keep this comment\n"
        f"models_hub:\n  enabled: true\n  download_dir: {old_store}\n"
    )
    (data / "settings.yaml").write_text(original, encoding="utf-8")
    chosen = tmp_path / "Models folder"
    config.write_model_store(data, chosen)
    text = (data / "settings.yaml").read_text(encoding="utf-8")
    assert "# owner comment" in text and "# keep this comment" in text
    loaded = __import__("yaml").safe_load(text)
    assert loaded["paths"]["model_store"] == str(chosen)
    assert loaded["models_hub"]["download_dir"] == ""


def test_model_store_legacy_value_migrates_with_warning(tmp_path):
    legacy = tmp_path / "legacy-models"
    (tmp_path / "settings.yaml").write_text(
        "version: 2\npaths:\n  model_store: \"\"\n"
        f"models_hub:\n  download_dir: {legacy}\n",
        encoding="utf-8",
    )
    (tmp_path / "models.yaml").write_text("version: 1\nmodels: []\n", encoding="utf-8")
    settings, _models, issues = config.Config.load(tmp_path)
    assert settings.paths.model_store == str(legacy)
    assert config.resolve_model_store(settings) == legacy
    assert any("deprecated" in issue.message for issue in issues)


def test_model_store_validation_rejects_install_tree_and_probes_folder(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    refused = setup_env.validate_model_store(
        install / "models", install_dir=install
    )
    assert not refused.ok and "outside" in refused.message
    chosen = tmp_path / "owned" / "models"
    accepted = setup_env.validate_model_store(chosen, install_dir=install)
    assert accepted.ok and Path(accepted.message).is_dir()
    assert not list(chosen.glob(".locitize-write-test-*"))


def test_controller_model_store_change_invalidates_downloader_and_refuses_busy(tmp_path):
    from gui_controller import GuiController

    settings = _settings(tmp_path)
    registry = SimpleNamespace(all=lambda: [])
    manager = SimpleNamespace(stop_all=lambda **_kwargs: None)
    controller = GuiController(
        settings, registry, object(), object(), manager,
        fetch=lambda _port, _path: None,
    )
    controller._hub_downloader = object()
    chosen = tmp_path / "new-models"
    controller._do_model_store_change(str(chosen))
    result = controller.result_q.get_nowait()
    assert result.ok and result.payload["path"] == str(chosen.resolve())
    assert controller._hub_downloader is None
    assert settings.paths.model_store == str(chosen.resolve())

    controller._hub_thread = SimpleNamespace(is_alive=lambda: True)
    controller._do_model_store_change(str(tmp_path / "another"))
    refusal = controller.result_q.get_nowait()
    assert not refusal.ok and "current model download" in refusal.error


def test_installer_dry_run_orchestrates_enables_and_records_egress(tmp_path):
    settings = _settings(tmp_path)
    calls = []

    def runner(argv, cancel, on_line, set_process):
        calls.append(list(argv))
        if argv[1:3] == ["-m", "venv"]:
            python = Path(argv[-1]) / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
        on_line("fake pip progress")
        return CommandOutcome(0, ("fake pip progress",))

    usage = lambda _path: SimpleNamespace(free=20 * 1024**3)
    installer = OpenWebUIInstaller(
        settings, threading.Lock(), runner=runner,
        verifier=lambda _settings: (True, "verified"), disk_usage=usage,
        base_python=tmp_path / "python.exe",
    )
    egress_log.configure(settings.data_dir)
    completed = threading.Event()
    outcomes = []
    states = []
    assert installer.start(
        lambda payload: states.append(payload["phase"]),
        lambda result: (outcomes.append(result), completed.set()),
    )
    assert completed.wait(3)
    assert outcomes[0].ok and installer.state == "succeeded"
    assert any(call[1:3] == ["-m", "venv"] for call in calls)
    assert any("open-webui==0.11.4" in call for call in calls)
    assert "venv" in states and "install" in states and "complete" in states
    assert settings.openwebui.enabled is True
    assert "enabled: true" in (Path(settings.data_dir) / "settings.yaml").read_text()
    rows = [json.loads(line) for line in egress_log.egress_path(settings.data_dir).read_text().splitlines()]
    assert [row["outcome"] for row in rows] == ["started", "succeeded"]
    assert all(row["kind"] == "external-installer" for row in rows)


def test_installer_cancel_is_single_instance_and_reaches_cancelled(tmp_path):
    settings = _settings(tmp_path)
    python = tmp_path / "Codebase" / ".webui-venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    entered = threading.Event()

    def runner(argv, cancel, on_line, set_process):
        entered.set()
        assert cancel.wait(3)
        return CommandOutcome(1, ("cancelled",), cancelled=True)

    installer = OpenWebUIInstaller(
        settings, threading.Lock(), runner=runner,
        verifier=lambda _settings: (True, "verified"),
        disk_usage=lambda _path: SimpleNamespace(free=20 * 1024**3),
    )
    completed = threading.Event()
    outcomes = []
    assert installer.start(lambda _payload: None,
                           lambda result: (outcomes.append(result), completed.set()))
    assert entered.wait(3)
    assert not installer.start(lambda _payload: None, lambda _result: None)
    installer.cancel()
    assert completed.wait(3)
    assert outcomes[0].state == "cancelled"
    assert installer.state == "cancelled"


def test_installer_dry_run_failure_sets_failed_state_and_log_link(tmp_path):
    settings = _settings(tmp_path)
    python = tmp_path / "Codebase" / ".webui-venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    installer = OpenWebUIInstaller(
        settings, threading.Lock(),
        runner=lambda *_args: CommandOutcome(1, ("certificate verify failed",)),
        verifier=lambda _settings: (True, "verified"),
        disk_usage=lambda _path: SimpleNamespace(free=20 * 1024**3),
    )
    completed = threading.Event()
    outcomes = []
    assert installer.start(lambda _payload: None,
                           lambda result: (outcomes.append(result), completed.set()))
    assert completed.wait(3)
    assert installer.state == "failed"
    assert "TLS certificate" in outcomes[0].message
    assert Path(outcomes[0].log_path).is_file()


def test_installer_failure_messages_are_specific_only_for_agreed_cases(tmp_path):
    log = tmp_path / "openwebui-install.log"
    assert "offline or DNS" in _friendly_failure("getaddrinfo failed", log)
    assert "TLS certificate" in _friendly_failure("certificate verify failed", log)
    assert "disk is full" in _friendly_failure("WinError 112", log)
    generic = _friendly_failure("antivirus locked a file", log)
    assert str(log) in generic and "antivirus" not in generic
    assert OpenWebUIInstaller._safe_endpoint(
        "https://user:secret@example.test/simple"
    ) == "https://example.test"


def test_release_builder_refuses_unsupported_python():
    from scripts import build_release

    build_release.require_supported_python((3, 11, 9))
    build_release.require_supported_python((3, 12, 2))
    for version in ((3, 10, 9), (3, 13, 0)):
        try:
            build_release.require_supported_python(version)
        except RuntimeError as exc:
            assert "3.11 or 3.12" in str(exc)
        else:
            raise AssertionError(f"accepted unsupported Python {version}")
