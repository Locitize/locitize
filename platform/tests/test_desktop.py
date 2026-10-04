"""Offscreen smoke tests for the thin M12 PySide6 presentation shell."""

import os
import queue
from types import SimpleNamespace

import gui_controller
import pytest

# Qt must choose the headless platform before its binding is imported. Machines
# without the optional desktop dependency skip these view tests cleanly while the
# controller suite remains runnable.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class FakeGuiController:
    """Small recorder exposing only the view-facing GuiController contract."""

    def __init__(self):
        self.result_q = queue.Queue()
        self.command_q = queue.Queue()
        self.calls = []
        self.shutdown_count = 0
        # None when no auto-tune is running; the model id while one is.
        self.autotune_running_model = None

    def list_models(self):
        return [
            {
                "id": "local-chat",
                "name": "Local Chat",
                "location": "/locitize-test/models/local-chat.gguf",
                "status": "installed",
                "gpu_layers": 60,
                "context_size": 16384,
                "launchable": True,
                "size_bytes": 10_300_000_000,
                "size_display": "10.3 GB",
                "vram_estimate_mb": 10000,
                "vram_need_mb": 10300,
                "gpu_portion_display": "10.3 GB",
                "cpu_portion_display": "-",
                "benchmark_score": None,
                "score_display": "-",
            },
            {
                "id": "local-code",
                "name": "Local Code",
                "location": "/locitize-test/models/local-code.gguf",
                "status": "installed",
                "gpu_layers": 42,
                "context_size": 8192,
                "launchable": True,
                "size_bytes": 7_200_000_000,
                "size_display": "7.2 GB",
                "vram_estimate_mb": 7000,
                "vram_need_mb": 7200,
                "gpu_portion_display": "7.2 GB",
                "cpu_portion_display": "-",
                "benchmark_score": 31.5,
                "score_display": "31.5 tok/s",
            },
        ]

    def system_specs(self):
        return gui_controller.SystemSpecs(
            ram_total_mb=32768.0, gpu_name="Fake GPU", gpu_vram_total_mb=16303.0
        )

    def available_voices(self):
        return ["af_heart"]

    def phone_access(self):
        return getattr(self, "_phone_access", None)

    def start_threads(self):
        self.calls.append(("start_threads",))

    def shutdown(self):
        self.shutdown_count += 1

    def request_start(self, model_id, reasoning=None):
        # reasoning is the 2026-09-02 one-launch thinking override; recorded so
        # the tests below can assert Start forwards None by default.
        self.calls.append(("request_start", model_id, reasoning))

    def request_stop(self):
        self.calls.append(("request_stop",))

    def request_whisper_toggle(self):
        self.calls.append(("request_whisper_toggle",))

    def request_listen(self, seconds):
        self.calls.append(("request_listen", seconds))

    def request_speak(self, value, voice):
        self.calls.append(("request_speak", value, voice))

    def request_audition(self):
        self.calls.append(("request_audition",))

    def request_start_assistant(self, voice, speak):
        self.calls.append(("request_start_assistant", voice, speak))

    def request_talk(self):
        self.calls.append(("request_talk",))

    def request_interrupt(self):
        self.calls.append(("request_interrupt",))

    def request_end_assistant(self):
        self.calls.append(("request_end_assistant",))

    def request_describe(self, path, prompt):
        self.calls.append(("request_describe", path, prompt))

    def request_memory_search(self, query):
        self.calls.append(("request_memory_search", query))

    def request_benchmark(self, model_id):
        self.calls.append(("request_benchmark", model_id))

    def request_chat(self):
        self.calls.append(("request_chat",))

    def open_chat(self, choice, remember):
        self.calls.append(("open_chat", choice, remember))

    def start_openwebui_and_open(self, remember):
        self.calls.append(("start_openwebui_and_open", remember))

    def detect_harnesses(self):
        return {"claude": None, "codex": None, "opencode": None}

    def last_project_dir(self):
        return ""

    def request_launch_harness(self, choice, project_dir, remember):
        self.calls.append(("request_launch_harness", choice, project_dir, remember))

    def request_install_harness(self, harness):
        self.calls.append(("request_install_harness", harness))

    def save_model_edits(self, model_id, gpu_text, context_text):
        self.calls.append(("save_model_edits", model_id, gpu_text, context_text))

    def save_model_identity(self, model_id, id_text, name_text):
        self.calls.append(("save_model_identity", model_id, id_text, name_text))

    def request_autotune_context(self, model_id):
        self.calls.append(("request_autotune_context", model_id))
        # Mirrors the real controller closely enough for the Stop button's
        # auto-tune branch: an accepted request means a tune is now in flight.
        self.autotune_running_model = model_id

    def autotune_in_progress(self):
        return self.autotune_running_model is not None

    def autotune_model_id(self):
        return self.autotune_running_model

    def cancel_autotune(self):
        self.calls.append(("cancel_autotune",))
        if self.autotune_running_model is None:
            return False
        self.autotune_running_model = None
        return True


@pytest.fixture(scope="module")
def qapp():
    """Load Qt lazily so each selected desktop test can skip with exit code 0."""
    pytest.importorskip("PySide6")
    import desktop
    from PySide6 import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app, desktop, QtGui


def test_model_table_defaults_to_fastest_generation_throughput(qapp):
    """The initial inventory ranks measured models fastest-first, missing last."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    rows = fake.list_models()
    rows[0]["benchmark_tok_s"] = None
    rows[1]["benchmark_tok_s"] = 31.5
    fake.list_models = lambda: rows

    window = desktop.MainWindow(fake, health="OK")

    assert window._sort_col == "benchmark_tok_s"
    assert window._sort_desc is True
    assert window._model_table.item(0, 0).text() == "Local Code"
    assert window._model_table.item(1, 0).text() == "Local Chat"

    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_desktop_smoke_builds_all_pages_and_shutdowns_once(qapp):
    """The window has every required destination and one close-to-shutdown path."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")

    assert tuple(window.pages) == desktop.PAGE_NAMES
    assert window._stack.count() == len(desktop.PAGE_NAMES)
    # Models is the landing page: every other page depends on a running model,
    # so opening on Talk showed a disabled button with no explanation.
    assert window._sidebar.currentItem().text() == "Home"
    assert window.minimumWidth() >= 960
    assert window.minimumHeight() >= 680

    close_event = qt_gui.QCloseEvent()
    window.closeEvent(close_event)
    window.closeEvent(qt_gui.QCloseEvent())

    assert close_event.isAccepted()
    assert fake.shutdown_count == 1
    window.deleteLater()
    app.processEvents()


def test_desktop_drain_applies_assistant_state_and_metrics(qapp):
    """The QTimer slot drains the controller queue and updates target widgets."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    sample = gui_controller.MetricsSample(
        gen_tokens_s=42.5,
        prompt_tokens_s=18.0,
        prompt_tokens_total=120,
        gen_tokens_total=36,
        n_tokens_max=156,
        n_ctx=4096,
        requests_processing=1,
    )
    fake.result_q.put(
        gui_controller.Result(
            "assistant_started",
            True,
            {
                "port": 8080,
                "running_model_id": "local-chat",
                "running_port": 8080,
            },
        )
    )
    fake.result_q.put(gui_controller.Result("assistant_state", True, {"state": "listening"}))
    fake.result_q.put(gui_controller.Result("metrics", True, {"sample": sample}))

    window._drain()

    assert window._talk_btn.text() == "Listening... speak now"
    assert not window._talk_btn.isEnabled()
    assert window._chat_page_btn.isEnabled()
    assert "42.5 tok/s" in window._talk_monitor.text()
    assert window._models_monitor.text() == window._talk_monitor.text()

    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_disabled_primary_actions_render_with_muted_background(qapp):
    """Disabled primary controls render grey instead of retaining enabled blue."""
    app, desktop, qt_gui = qapp
    previous_style = app.styleSheet()
    app.setStyleSheet(desktop.APP_STYLE)
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    enabled = desktop.QtWidgets.QPushButton()
    disabled = desktop.QtWidgets.QPushButton()
    for button in (enabled, disabled):
        button.setObjectName("primaryButton")
        button.resize(100, 40)
        button.show()
    disabled.setEnabled(False)
    app.processEvents()

    try:
        # Save and Chat begin disabled and share the selector exercised by the
        # blank reference buttons, avoiding text pixels in the color sample.
        assert not window._save_btn.isEnabled()
        assert not window._chat_page_btn.isEnabled()
        assert window._save_btn.objectName() == "primaryButton"
        assert window._chat_page_btn.objectName() == "primaryButton"
        enabled_color = enabled.grab().toImage().pixelColor(10, 10).name()
        disabled_color = disabled.grab().toImage().pixelColor(10, 10).name()
        disabled_text = disabled.palette().color(
            qt_gui.QPalette.ColorGroup.Disabled,
            qt_gui.QPalette.ColorRole.ButtonText,
        ).name()
        assert enabled_color == "#4f8cff"
        assert disabled_color == "#292b2f"
        assert disabled_text == "#6f7379"
        assert disabled_color != enabled_color
    finally:
        window.closeEvent(qt_gui.QCloseEvent())
        window.deleteLater()
        enabled.deleteLater()
        disabled.deleteLater()
        app.setStyleSheet(previous_style)
        app.processEvents()


def test_model_lifecycle_clicks_apply_success_and_error_results(qapp):
    """Start, Switch, and Stop clicks delegate while Results own displayed state."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")

    window._start_btn.click()
    # reasoning is None: the Thinking box sits on its do-nothing default, so
    # Start behaves exactly as it did before the 2026-09-02 override existed.
    assert fake.calls[-1] == ("request_start", "local-chat", None)
    window._apply(
        gui_controller.Result(
            "start",
            True,
            {"running_model_id": "local-chat", "running_port": 8080},
        )
    )
    window._model_table.setCurrentCell(1, 0)
    assert window._start_btn.text() == "Switch"
    window._start_btn.click()
    assert fake.calls[-1] == ("request_start", "local-code", None)
    window._apply(
        gui_controller.Result(
            "start",
            False,
            {"running_model_id": "local-chat", "running_port": 8080},
            error="switch refused",
        )
    )
    assert window._model_status.text() == "switch refused"
    assert window._stop_btn.isEnabled()
    window._stop_btn.click()
    assert fake.calls[-1] == ("request_stop",)
    window._apply(
        gui_controller.Result(
            "stop",
            False,
            {"running_model_id": "local-chat", "running_port": 8080},
            error="stop refused",
        )
    )
    assert window._model_status.text() == "stop refused"
    assert window._stop_btn.isEnabled()
    window._stop_btn.click()
    window._apply(
        gui_controller.Result("stop", True, {"running_model_id": None, "running_port": None})
    )
    assert window._model_status.text() == "stopped"
    assert not window._stop_btn.isEnabled()

    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("terminal_kind", ["assistant_ended", "assistant_error"])
def test_assistant_terminal_results_restore_model_lifecycle(qapp, terminal_kind):
    """Assistant end and error release the model lock without inventing a snapshot."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    window._apply(
        gui_controller.Result(
            "start",
            True,
            {"running_model_id": "local-chat", "running_port": 8080},
        )
    )
    window._model_table.setCurrentCell(1, 0)
    snapshot = {
        "port": 8080,
        "running_model_id": "local-chat",
        "running_port": 8080,
    }
    window._apply(gui_controller.Result("assistant_started", True, snapshot))
    assert not window._start_btn.isEnabled()
    assert not window._stop_btn.isEnabled()

    if terminal_kind == "assistant_error":
        result = gui_controller.Result(terminal_kind, False, {}, error="assistant failed")
    else:
        result = gui_controller.Result(terminal_kind, True, {})
    window._apply(result)

    assert window._start_btn.text() == "Switch"
    assert window._start_btn.isEnabled()
    assert window._stop_btn.isEnabled()
    assert window._ui.running_model_id == "local-chat"
    assert window._ui.running_port == 8080
    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_voice_vision_memory_benchmark_and_settings_bindings(qapp):
    """Every non-dialog state-changing page routes clicks and renders outcomes."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")

    window._whisper_btn.click()
    window._listen_btn.click()
    window._speak_btn.click()
    window._audition_btn.click()
    assert ("request_whisper_toggle",) in fake.calls
    assert ("request_listen", 15.0) in fake.calls
    assert any(call[0] == "request_speak" for call in fake.calls)
    assert ("request_audition",) in fake.calls

    window._vision_path.setText("/locitize-test/real-user-selection.png")
    window._vision_prompt.setText("What is visible?")
    window._refresh_model_exclusive_buttons()
    window._vision_btn.click()
    assert fake.calls[-1] == (
        "request_describe",
        "/locitize-test/real-user-selection.png",
        "What is visible?",
    )
    window._apply(gui_controller.Result("describe", True, {"answer": "A local image."}))
    assert window._vision_result.toPlainText() == "A local image."
    window._apply(gui_controller.Result("describe", False, {}, error="vision unavailable"))
    assert window._vision_result.toPlainText() == "vision unavailable"

    window._memory_query.setText("owner")
    window._memory_btn.click()
    assert fake.calls[-1] == ("request_memory_search", "owner")
    window._apply(
        gui_controller.Result(
            "memory_search",
            True,
            {"query": "owner", "hits": [{"role": "user", "text": "hello"}]},
        )
    )
    assert window._memory_result.toPlainText() == "user: hello"
    window._apply(gui_controller.Result("memory_search", True, {"query": "missing", "hits": []}))
    assert "no stored conversation" in window._memory_result.toPlainText()
    window._apply(
        gui_controller.Result("memory_search", False, {}, error="memory index unavailable")
    )
    assert window._memory_result.toPlainText() == "memory index unavailable"

    # Benchmark has no button of its own any more (Auto-tune runs it), but its
    # results still update the row and status line.
    assert not hasattr(window, "_benchmark_btn")
    window._apply(
        gui_controller.Result(
            "benchmark",
            False,
            {"model_id": "local-chat"},
            error="benchmark busy",
        )
    )
    assert window._model_status.text() == "benchmark busy"
    window._apply(
        gui_controller.Result(
            "benchmark",
            True,
            {
                "model_id": "local-chat",
                "score": 55.0,
                "score_display": "55.0 tok/s",
                "detail": "55.0 tok/s",
            },
        )
    )
    assert window._model_status.text() == "benchmarked local-chat: 55.0 tok/s"

    window._gpu_edit.setText("61")
    assert window._save_btn.isEnabled()
    window._save_btn.click()
    assert fake.calls[-1] == (
        "save_model_edits",
        "local-chat",
        "61",
        "16384",
    )
    window._apply(
        gui_controller.Result(
            "save_edits",
            True,
            {"model_id": "local-chat", "gpu_layers": 61, "context_size": 16384},
        )
    )
    assert not window._save_btn.isEnabled()
    window._ctx_edit.setText("not-a-number")
    assert window._save_btn.isEnabled()
    assert window._edit_error.text()
    window._save_btn.click()
    assert fake.calls[-1] == (
        "save_model_edits",
        "local-chat",
        "61",
        "not-a-number",
    )
    window._apply(
        gui_controller.Result("save_edits", False, {}, error="context_size must be an integer")
    )
    assert window._edit_error.text() == "context_size must be an integer"

    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_assistant_and_chat_dialog_bindings(qapp, monkeypatch):
    """Talk controls and accepted chat dialogs delegate through controller intents."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")

    window._assistant_start_btn.click()
    assert fake.calls[-1] == ("request_start_assistant", "af_heart", True)
    window._apply(
        gui_controller.Result(
            "assistant_started",
            True,
            {
                "port": 8080,
                "running_model_id": "local-chat",
                "running_port": 8080,
            },
        )
    )
    window._talk_btn.click()
    window._interrupt_btn.click()
    window._assistant_end_btn.click()
    assert ("request_talk",) in fake.calls
    assert ("request_interrupt",) in fake.calls
    assert fake.calls[-1] == ("request_end_assistant",)

    window._apply(gui_controller.Result("assistant_ended", True, {}))
    window._chat_page_btn.click()
    assert fake.calls[-1] == ("request_chat",)
    monkeypatch.setattr(
        desktop.QtWidgets.QDialog,
        "exec",
        lambda _dialog: desktop.QtWidgets.QDialog.DialogCode.Accepted,
    )
    window._apply(gui_controller.Result("chat_ask", True, {}))
    assert fake.calls[-1] == ("open_chat", "llamacpp", False)
    window._apply(
        gui_controller.Result("chat_offer_start", True, {"reason": "Open WebUI is stopped"})
    )
    assert fake.calls[-1] == ("start_openwebui_and_open", False)
    window._apply(gui_controller.Result("chat", False, {}, error="no model is running"))
    assert window._chat_status.text() == "no model is running"
    window._apply(
        gui_controller.Result("chat", True, {"opened": True, "reason": "opened built-in chat"})
    )
    assert window._chat_status.text() == "opened built-in chat"

    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


# --------------------------------------------------------------------------- #
# Zero-friction harness onboarding (owner request 2026-08-21)
# --------------------------------------------------------------------------- #


def test_apply_install_harness_result_success_and_failure(qapp):
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")

    window._apply(
        gui_controller.Result(
            "install_harness", True, {"harness": "codex", "message": "Codex installed"}
        )
    )
    assert window._chat_status.text() == "Codex installed"

    window._apply(
        gui_controller.Result(
            "install_harness",
            False,
            {"harness": "opencode"},
            error="OpenCode install failed: exited 1",
        )
    )
    assert window._chat_status.text() == "OpenCode install failed: exited 1"

    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_confirm_and_install_harness_installs_only_on_yes(qapp, monkeypatch):
    # Security review 2026-08-21 fix: the picker's Install button used to
    # run the real installer on a single click with zero confirmation.
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    dummy_dialog = desktop.QtWidgets.QDialog(window)

    monkeypatch.setattr(
        desktop.QtWidgets.QMessageBox,
        "exec",
        lambda self: desktop.QtWidgets.QMessageBox.StandardButton.Yes,
    )
    window._confirm_and_install_harness(dummy_dialog, "codex", "Codex")
    assert [c[1] for c in fake.calls if c[0] == "request_install_harness"] == ["codex"]
    window.deleteLater()
    dummy_dialog.deleteLater()
    app.processEvents()


def test_confirm_and_install_harness_declines_by_default(qapp, monkeypatch):
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    dummy_dialog = desktop.QtWidgets.QDialog(window)

    # Simulates the default button (No) being activated without an
    # explicit Yes click.
    monkeypatch.setattr(
        desktop.QtWidgets.QMessageBox,
        "exec",
        lambda self: desktop.QtWidgets.QMessageBox.StandardButton.No,
    )
    window._confirm_and_install_harness(dummy_dialog, "codex", "Codex")
    assert [c for c in fake.calls if c[0] == "request_install_harness"] == []
    window.deleteLater()
    dummy_dialog.deleteLater()
    app.processEvents()


def test_show_harness_onboarding_skips_when_all_installed(qapp, monkeypatch):
    """0 missing -> no dialog at all, not even a constructed-but-unshown one."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    fake.detect_harnesses = lambda: {
        "claude": "/bin/claude",
        "codex": "/bin/codex",
        "opencode": "/bin/opencode",
    }
    window = desktop.MainWindow(fake, health="OK")

    def fail_if_shown(self):
        raise AssertionError("QDialog.exec() must not run when nothing is missing")

    monkeypatch.setattr(desktop.QtWidgets.QDialog, "exec", fail_if_shown)

    window.show_harness_onboarding_if_needed()  # must return quietly

    assert [c for c in fake.calls if c[0] == "request_install_harness"] == []
    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_show_harness_onboarding_accepting_with_nothing_checked_installs_nothing(
    qapp, monkeypatch
):
    # Security review 2026-08-21 fix: checkboxes default UNCHECKED (opt-in),
    # so clicking OK without checking anything must install nothing - a
    # reflexive OK must never run a remote installer script.
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    fake.detect_harnesses = lambda: {"claude": None, "codex": "/bin/codex", "opencode": None}
    window = desktop.MainWindow(fake, health="OK")

    monkeypatch.setattr(
        desktop.QtWidgets.QDialog,
        "exec",
        lambda self: desktop.QtWidgets.QDialog.DialogCode.Accepted,
    )
    window.show_harness_onboarding_if_needed()

    requested = [c[1] for c in fake.calls if c[0] == "request_install_harness"]
    assert requested == []
    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_show_harness_onboarding_offers_missing_and_installs_checked_ones(qapp, monkeypatch):
    """1-2 missing -> dialog shown; only explicitly-checked boxes install."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    fake.detect_harnesses = lambda: {"claude": None, "codex": "/bin/codex", "opencode": None}
    window = desktop.MainWindow(fake, health="OK")

    monkeypatch.setattr(
        desktop.QtWidgets.QDialog,
        "exec",
        lambda self: desktop.QtWidgets.QDialog.DialogCode.Accepted,
    )
    # Simulate the owner explicitly checking every offered box before OK.
    monkeypatch.setattr(desktop.QtWidgets.QCheckBox, "isChecked", lambda self: True)
    window.show_harness_onboarding_if_needed()

    requested = sorted(c[1] for c in fake.calls if c[0] == "request_install_harness")
    assert requested == ["claude", "opencode"]  # codex already installed, not offered
    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_show_harness_onboarding_installs_nothing_when_declined(qapp, monkeypatch):
    """3 missing -> declining the dialog requests zero installs."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    fake.detect_harnesses = lambda: {"claude": None, "codex": None, "opencode": None}
    window = desktop.MainWindow(fake, health="OK")

    monkeypatch.setattr(
        desktop.QtWidgets.QDialog,
        "exec",
        lambda self: desktop.QtWidgets.QDialog.DialogCode.Rejected,
    )
    window.show_harness_onboarding_if_needed()

    assert [c for c in fake.calls if c[0] == "request_install_harness"] == []
    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_show_harness_onboarding_survives_detection_failure(qapp, monkeypatch):
    """A detect_harnesses() exception means no offer, never a failed startup."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()

    def boom():
        raise RuntimeError("no PATH")

    fake.detect_harnesses = boom
    window = desktop.MainWindow(fake, health="OK")

    window.show_harness_onboarding_if_needed()  # must not raise

    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


# --------------------------------------------------------------------------- #
# Auto-tune context (owner request 2026-08-22)
# --------------------------------------------------------------------------- #


def _autotune_window(desktop, monkeypatch, answer):
    """Build a Models-page window whose auto-tune confirmation returns `answer`."""
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    monkeypatch.setattr(
        desktop.QtWidgets.QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: answer),
    )
    return fake, window


def test_autotune_button_asks_before_starting_and_queues_on_yes(qapp, monkeypatch):
    """The control is real: a Yes reaches the controller with the selected id."""
    app, desktop, qt_gui = qapp
    fake, window = _autotune_window(
        desktop, monkeypatch, desktop.QtWidgets.QMessageBox.StandardButton.Yes
    )
    window._autotune_btn.click()
    assert ("request_autotune_context", "local-chat") in fake.calls
    # Locked out for the duration so a second run cannot be queued on top.
    assert window._autotune_btn.isEnabled() is False
    window.deleteLater()
    app.processEvents()


def test_autotune_button_does_nothing_when_the_confirmation_is_declined(qapp, monkeypatch):
    """Several minutes of GPU time must never start from a misclick."""
    app, desktop, qt_gui = qapp
    fake, window = _autotune_window(
        desktop, monkeypatch, desktop.QtWidgets.QMessageBox.StandardButton.No
    )
    window._autotune_btn.click()
    assert [c for c in fake.calls if c[0] == "request_autotune_context"] == []
    window.deleteLater()
    app.processEvents()


def test_autotune_progress_lines_are_rendered_as_they_arrive(qapp):
    """The run is minutes long, so the owner must see which value is being tried."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    fake.result_q.put(
        gui_controller.Result(
            "autotune_progress",
            True,
            {"model_id": "local-chat", "line": "trial 2 of at most 5: starting at context 262144 ..."},
        )
    )
    window._drain()
    assert window._autotune_status.isVisible() or window._autotune_status.text()
    assert "262144" in window._autotune_status.text()
    window.deleteLater()
    app.processEvents()


def test_autotune_summary_reports_the_real_numbers(qapp):
    """The final readout names the native window, the ceiling and the YaRN choice."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    fake.result_q.put(
        gui_controller.Result(
            "autotune",
            True,
            {
                "model_id": "local-chat",
                "native_context": 131072,
                "previous_context": 16384,
                "chosen_context": 196608,
                "first_failure": 212992,
                "yarn_applied": True,
                "rope_scale": 2,
                "trials": [{}, {}, {}, {}, {}, {}],
                "tokens_per_second": 48.25,
            },
        )
    )
    window._drain()
    text = window._autotune_status.text()
    for expected in ("131072", "196608", "212992", "extended 2x", "16384", "48.2 tokens/s"):
        assert expected in text
    assert window._model_status.text() == "auto-tune finished: context 196608, 48.2 tokens/s"
    window.deleteLater()
    app.processEvents()


def test_autotune_failure_shows_the_reason_not_a_success(qapp):
    """A refusal or a rollback must read as a failure with its real cause."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    fake.result_q.put(
        gui_controller.Result(
            "autotune",
            False,
            {"model_id": "local-chat"},
            error="stop the running model before auto-tuning context",
        )
    )
    window._drain()
    assert "failed" in window._autotune_status.text()
    assert "stop the running model" in window._autotune_status.text()
    window.deleteLater()
    app.processEvents()


# ---- Stop interrupts a running auto-tune (owner request 2026-08-22) ------- #


def test_stop_cancels_a_running_autotune_instead_of_queueing_a_model_stop(
    qapp, monkeypatch
):
    """Stop must reach past the command queue while a tune owns the ops worker.

    A queued stop cannot be serviced until the multi-minute tune it was meant to
    interrupt has already finished, so the owner would press Stop and watch
    nothing happen. cancel_autotune() is a direct call for exactly that reason,
    and no request_stop must be raised (there is no model running to stop).
    """
    app, desktop, qt_gui = qapp
    fake, window = _autotune_window(
        desktop, monkeypatch, desktop.QtWidgets.QMessageBox.StandardButton.Yes
    )
    window._autotune_btn.click()
    assert window._stop_btn.isEnabled() is True, "Stop must be the way out of a tune"

    window._stop_btn.click()
    assert ("cancel_autotune",) in fake.calls
    assert [c for c in fake.calls if c[0] == "request_stop"] == []
    assert "cancel" in window._autotune_status.text().lower()
    window.deleteLater()
    app.processEvents()


def test_stop_is_live_immediately_after_the_autotune_click(qapp, monkeypatch):
    """Enabled in the same event as the click, not at the next refresh.

    The moment an owner most wants out is right after realising they clicked by
    mistake, which is before the ops worker has even picked the command up.
    """
    app, desktop, qt_gui = qapp
    fake, window = _autotune_window(
        desktop, monkeypatch, desktop.QtWidgets.QMessageBox.StandardButton.Yes
    )
    # No model is running, so Stop is normally disabled on this page.
    assert window._stop_btn.isEnabled() is False
    window._autotune_btn.click()
    assert window._stop_btn.isEnabled() is True
    window.deleteLater()
    app.processEvents()


def test_a_second_stop_click_is_refused_while_the_cancel_is_landing(qapp, monkeypatch):
    """Once asked, Stop goes flat rather than inviting a click that does nothing."""
    app, desktop, qt_gui = qapp
    fake, window = _autotune_window(
        desktop, monkeypatch, desktop.QtWidgets.QMessageBox.StandardButton.Yes
    )
    window._autotune_btn.click()
    window._stop_btn.click()
    assert window._stop_btn.isEnabled() is False
    window.deleteLater()
    app.processEvents()


def test_a_cancelled_autotune_is_reported_as_cancelled_not_as_a_failure(qapp):
    """Nothing went wrong, so the readout must not say it did."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    fake.result_q.put(
        gui_controller.Result(
            "autotune",
            False,
            {
                "model_id": "local-chat",
                "canceled": True,
                "detail": "auto-tune canceled.",
            },
            error=None,
        )
    )
    window._drain()
    text = window._autotune_status.text().lower()
    assert "cancel" in text
    assert "fail" not in text
    assert "unchanged" in text
    window.deleteLater()
    app.processEvents()


def test_stop_returns_to_its_ordinary_meaning_once_the_autotune_ends(qapp, monkeypatch):
    """The auto-tune override is scoped to the run, not latched forever."""
    app, desktop, qt_gui = qapp
    fake, window = _autotune_window(
        desktop, monkeypatch, desktop.QtWidgets.QMessageBox.StandardButton.Yes
    )
    window._autotune_btn.click()
    fake.autotune_running_model = None
    fake.result_q.put(
        gui_controller.Result(
            "autotune", False, {"model_id": "local-chat", "canceled": True}
        )
    )
    window._drain()
    # No model is running, so Stop is disabled again and Auto-tune is offered.
    assert window._stop_btn.isEnabled() is False
    assert window._autotune_btn.isEnabled() is True
    window.deleteLater()
    app.processEvents()


def test_thinking_box_defaults_to_no_override(qapp):
    """The box ships on 'as registered', so Start is unchanged until touched."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    assert window._thinking_combo.currentText() == desktop.THINKING_AS_REGISTERED
    window._start_btn.click()
    assert fake.calls[-1] == ("request_start", "local-chat", None)


def test_thinking_box_forwards_a_picked_level_and_budget(qapp):
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    window._thinking_combo.setCurrentText("low/2048")
    window._start_btn.click()
    assert fake.calls[-1] == (
        "request_start",
        "local-chat",
        {"effort": "low", "budget": 2048},
    )


def test_thinking_box_accepts_a_level_not_in_its_own_list(qapp):
    """The list is a convenience; the model's chat template owns the vocabulary.

    Qwen3.8 takes xhigh/medium/low while gpt-oss takes low/medium/high, so a
    fixed dropdown would be wrong for part of any real registry.
    """
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    window._thinking_combo.setCurrentText("some-future-level")
    window._start_btn.click()
    assert fake.calls[-1][2] == {"effort": "some-future-level"}


def test_thinking_box_refuses_to_start_on_an_unreadable_entry(qapp):
    """Same rule as the terminal menu: never start at a level nobody chose."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    before = list(fake.calls)
    window._thinking_combo.setCurrentText("low/abc")
    window._start_btn.click()
    assert fake.calls == before  # nothing was queued
    assert "integer" in window._model_status.text()


def test_an_externally_started_model_appears_in_the_window(qapp):
    """Owner-observed 2026-09-03: the router switched models from a browser
    picker and this window went on showing nothing running."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    assert window._ui.running_model_id is None
    window._apply(
        gui_controller.Result(
            "running_model",
            True,
            {"running_model_id": "local-code", "running_port": 8080},
        )
    )
    assert window._ui.running_model_id == "local-code"
    assert window._ui.running_port == 8080


def test_an_external_observation_does_not_clear_an_in_flight_start(qapp):
    """It is a passive observation. Clearing the flag would re-enable the
    buttons underneath a start the owner is still waiting on."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    window._ui.in_flight = True
    window._apply(
        gui_controller.Result(
            "running_model", True,
            {"running_model_id": "local-chat", "running_port": 8080},
        )
    )
    assert window._ui.in_flight is True


def test_an_external_stop_clears_the_stale_metrics(qapp):
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    window._apply(
        gui_controller.Result(
            "running_model", True,
            {"running_model_id": "local-chat", "running_port": 8080},
        )
    )
    window._ui.latest_metrics = object()
    window._apply(
        gui_controller.Result(
            "running_model", True,
            {"running_model_id": None, "running_port": None},
        )
    )
    assert window._ui.running_model_id is None
    assert window._ui.latest_metrics is None


def test_voice_display_name_uses_human_labels():
    import desktop

    assert desktop.voice_display_name("af_heart") == "Heart"
    assert desktop.voice_display_name("am_michael") == "Michael"
    assert desktop.voice_display_name("af_newvoice") == "Newvoice"


def test_talk_click_starts_the_assistant_when_not_live(qapp):
    """Talk is the first action: it starts the session using the on-disk voice id."""
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    assert window._talk_voice.currentText() == "Heart"
    window._talk_btn.click()
    assert fake.calls[-1] == ("request_start_assistant", "af_heart", True)
    window._apply(
        gui_controller.Result(
            "assistant_started",
            True,
            {
                "port": 8080,
                "running_model_id": "local-chat",
                "running_port": 8080,
            },
        )
    )
    assert ("request_talk",) in fake.calls
    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_talk_stays_clickable_while_speaking(qapp):
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    window._apply(
        gui_controller.Result(
            "assistant_started",
            True,
            {
                "port": 8080,
                "running_model_id": "local-chat",
                "running_port": 8080,
            },
        )
    )
    window._apply(
        gui_controller.Result("assistant_state", True, {"state": "speaking"})
    )
    assert window._talk_btn.text() == "Stop and listen"
    assert window._talk_btn.isEnabled()
    window._apply(
        gui_controller.Result("assistant_state", True, {"state": "idle"})
    )
    assert ("request_talk",) in fake.calls
    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_chat_page_shows_a_tailscale_phone_url_when_serve_is_configured(qapp):
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    fake._phone_access = SimpleNamespace(
        url="https://chat.example.ts.net/",
        proxy_target="http://127.0.0.1:8096",
        hostname="chat.example.ts.net",
        tailnet_only=True,
    )
    window = desktop.MainWindow(fake, health="OK")
    assert window._phone_url_edit is not None
    assert window._phone_url_edit.text() == "https://chat.example.ts.net/"
    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_chat_page_hides_the_phone_strip_without_tailscale_serve(qapp):
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    assert window._phone_url_edit is None
    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_memory_page_lists_recent_on_first_visit(qapp):
    app, desktop, qt_gui = qapp
    fake = FakeGuiController()
    window = desktop.MainWindow(fake, health="OK")
    assert ("request_memory_search", "") not in fake.calls
    window._sidebar.setCurrentRow(desktop.PAGE_NAMES.index("Memory"))
    assert ("request_memory_search", "") in fake.calls
    window.closeEvent(qt_gui.QCloseEvent())
    window.deleteLater()
    app.processEvents()
