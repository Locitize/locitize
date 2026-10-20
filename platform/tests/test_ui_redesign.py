"""The five-page navigation: where things live, and that disabled controls explain themselves."""

import os
from types import SimpleNamespace

import gui_controller
import pytest
from tests.test_desktop import FakeGuiController

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class RedesignFake(FakeGuiController):
    def __init__(self):
        super().__init__()
        self.ready = {"listen": True, "whisper_server": True, "speak": True}
        self.harnesses = {
            "codex": "/bin/codex",
            "claude": "/bin/claude",
            "opencode": None,
        }
        self.selected = []

    def voice_readiness(self):
        return self.ready

    def detect_harnesses(self):
        return self.harnesses

    def note_selected_model(self, model_id):
        self.selected.append(model_id)

    def request_release_and(self, then):
        self.calls.append(("request_release_and", then))

    def request_free_gpu(self):
        self.calls.append(("request_free_gpu",))

    def request_chat(self):
        self.calls.append(("request_chat",))


@pytest.fixture()
def win():
    pytest.importorskip("PySide6")
    import desktop
    from PySide6 import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fake = RedesignFake()
    window = desktop.MainWindow(fake, health="WARNING")
    yield window, fake, desktop
    window.closeEvent(QtGui.QCloseEvent())
    window.deleteLater()
    app.processEvents()


def test_navigation_is_six_pages_with_fine_tune_after_voice(win):
    window, _fake, desktop = win
    assert tuple(window.pages) == (
        "Start", "Models", "Code", "Voice & Vision", "Fine-tune", "Settings",
    )
    assert desktop.build_page_names(True) == (
        "Start", "Models", "Code", "Voice & Vision", "Fine-tune", "Settings",
    )
    for retired in ("Home", "Chat", "Memory", "Sessions", "Talk", "Voice Setup", "Vision"):
        assert retired not in window.pages
    assert window._sidebar.currentItem().text() == "Start"


def test_start_page_owns_start_stop_chat_and_the_picker(win):
    window, fake, _desktop = win
    start = window.pages["Start"]
    for button in (window._start_btn, window._stop_btn, window._chat_btn):
        assert start.isAncestorOf(button)
    assert window._start_model_combo.count() == 2
    # Picking a model on Start selects it everywhere and tells the controller.
    window._start_model_combo.setCurrentIndex(1)
    assert window._selected_id == "local-code"
    assert fake.selected[-1] == "local-code"
    assert window._settings_model.currentData() == "local-code"
    window._start_btn.click()
    assert fake.calls[-1] == ("request_start", "local-code", None)


def test_start_status_line_is_plain_language(win):
    window, _fake, _desktop = win
    assert window._start_status_line.text() == "No model running"
    window._apply(gui_controller.Result(
        "running_model", True, {"running_model_id": "local-chat", "running_port": 8080}
    ))
    assert window._start_status_line.text() == "Local Chat is running - ready to chat"
    assert "Local Chat running - port 8080" in window._header_status.text()
    sample = gui_controller.MetricsSample(
        gen_tokens_s=12.5, n_tokens_max=1024, n_ctx=16384
    )
    window._render_metrics(sample)
    assert "12.5 tokens per second" in window._header_status.text()
    assert "ctx 1024/16384 (6%)" in window._header_status.text()
    assert window._health_btn.text().startswith("Health:")


def test_fit_hint_uses_the_gpu_size(win):
    window, _fake, _desktop = win
    assert gui_controller.format_fit(10000, 16303) == "Yes"
    assert gui_controller.format_fit(20000, 16303) == "Partly"
    assert gui_controller.format_fit(None, 16303) == "-"
    assert gui_controller.format_fit(5000, None) == "-"
    assert "Fits your GPU" in window._start_fit_label.text()
    assert "Yes" in window._model_table.item(0, 4).text()


def test_models_page_lists_plain_columns_and_keeps_advanced_tools(win):
    window, _fake, _desktop = win
    headers = [
        window._model_table.horizontalHeaderItem(i).text()
        for i in range(window._model_table.columnCount())
    ]
    assert headers == [
        "Model", "Capabilities", "Source", "Size", "Fits your GPU?",
        "On GPU", "On CPU", "Speed", "Context", "Status",
    ]
    assert window._model_table.item(0, 1).text() == "Chat, Vision"
    assert window._model_table.item(0, 5).text() == "10.3 GB"
    assert window._model_table.item(0, 6).text() == "-"
    context = window._model_table.item(0, 8)
    assert context.text() == "16,384  !"
    assert "speed cliff" in context.toolTip()
    assert "speed cliff" not in window._model_table.item(0, 7).toolTip()

    for column, key in ((1, "capabilities_display"), (5, "vram"), (8, "context")):
        window._sort_col = None
        window._on_sort(column)
        assert window._sort_col == key
    models = window.pages["Models"]
    for widget in (
        window._autotune_btn, window._autotune_all_btn, window._thinking_combo,
        window._register_btn, window._delete_btn, window._refresh_models_btn,
        window._settings_model, window._gpu_edit, window._ctx_edit,
        window._id_edit, window._name_edit, window._capabilities_edit,
        window._hub_download_btn,
    ):
        assert models.isAncestorOf(widget)
    tabs = [window._models_tabs.tabText(i) for i in range(window._models_tabs.count())]
    assert tabs == ["My models", "Get models", "Advanced"]
    assert window._register_btn.text() == "Add to my models"


def test_code_page_has_launcher_sessions_memory_and_one_release_button(win):
    window, _fake, _desktop = win
    code = window.pages["Code"]
    for button in window._code_harness_buttons.values():
        assert code.isAncestorOf(button)
    assert code.isAncestorOf(window._sessions_page)
    assert code.isAncestorOf(window._memory_query)
    from PySide6 import QtWidgets

    release = [
        b for b in window.findChildren(QtWidgets.QPushButton) if b.text() == "Release model"
    ]
    assert len(release) == 1
    # A tool that is not installed is greyed out and says where to get it.
    assert window._code_harness_buttons["codex"].isEnabled()
    assert window._code_harness_buttons["claude"].text() == "Claude Code"
    assert window._code_harness_buttons["claude"].isEnabled()
    missing = window._code_harness_buttons["opencode"]
    assert not missing.isEnabled()
    assert "Settings" in missing.toolTip()


def test_code_tool_always_uses_session_dialog(win, monkeypatch):
    window, fake, _desktop = win
    launched = []
    monkeypatch.setattr(
        window._sessions_page, "local_launch", lambda session, tool=None: launched.append(tool)
    )
    window._on_code_tool("codex")
    assert launched == ["codex"]  # nothing running: the dialog starts a model
    window._ui.running_model_id = "local-chat"
    window._on_code_tool("codex")
    assert launched == ["codex", "codex"]


def test_old_entry_points_are_gone_from_the_chat_chooser(win):
    window, _fake, _desktop = win
    import inspect

    source = inspect.getsource(window._show_chat_chooser)
    assert "harness" not in source.lower()
    assert "Claude" not in source


def test_voice_and_vision_has_four_tabs_and_one_banner(win):
    window, fake, _desktop = win
    tabs = [window._vv_tabs.tabText(i) for i in range(window._vv_tabs.count())]
    assert tabs == ["Talk", "Voice and mic setup", "Ask about an image", "Watch my screen"]
    page = window.pages["Voice & Vision"]
    for widget in (window._talk_btn, window._whisper_btn, window._vision_btn,
                   window._second_eye_start_btn):
        assert page.isAncestorOf(widget)
    window._go_to("Voice & Vision")
    window.show()
    # No model running: the banner offers the Start page.
    assert window._vv_banner.isVisible()
    assert window._vv_banner_btn.text() == "Go to Start"
    window._vv_banner_btn.click()
    assert window._sidebar.currentItem().text() == "Start"
    # Voice not installed: the same banner now points at Settings.
    fake.ready = {"listen": False, "whisper_server": False, "speak": True}
    window._go_to("Voice & Vision")
    window._update_vv_banner()
    assert "voice input" in window._vv_banner_label.text()
    assert window._vv_banner_btn.text() == "Open Settings"
    window._vv_banner_btn.click()
    assert window._sidebar.currentItem().text() == "Settings"
    # Running model and voice installed: the banner steps aside.
    fake.ready = {"listen": True, "whisper_server": True, "speak": True}
    window._ui.running_model_id = "local-chat"
    window._update_vv_banner()
    assert not window._vv_banner.isVisible()


def test_settings_is_app_level_only(win):
    window, _fake, _desktop = win
    settings = window.pages["Settings"]
    for widget in (
        window._open_wizard_btn, window._uninstall_btn, window._noise_combo,
        window._about_btn, window._health_detail, window._phone_access_note,
        *window._cli_status_labels.values(),
    ):
        assert settings.isAncestorOf(widget)
    for widget in (window._gpu_edit, window._ctx_edit, window._id_edit):
        assert not settings.isAncestorOf(widget)
    assert "claude" in window._cli_status_labels
    assert settings.isAncestorOf(window._cli_status_labels["claude"])


def test_startup_harness_onboarding_offers_claude_code(win, monkeypatch):
    window, fake, desktop = win
    fake.harnesses = {
        "codex": "/bin/codex",
        "claude": None,
        "opencode": "/bin/opencode",
    }
    seen = []

    def inspect_and_decline(dialog):
        seen.extend(box.text() for box in dialog.findChildren(desktop.QtWidgets.QCheckBox))
        return desktop.QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(desktop.QtWidgets.QDialog, "exec", inspect_and_decline)
    window.show_harness_onboarding_if_needed()
    assert seen == ["Claude Code"]


def test_whisper_listen_speak_report_in_their_own_labels(win):
    window, _fake, _desktop = win
    window._model_status.setText("")
    window._apply(gui_controller.Result(
        "whisper", False, {"running": False},
        error="Voice input isn't installed. Add it in Settings > Features.",
    ))
    window._apply(gui_controller.Result(
        "listen", False, {}, error="Voice input isn't installed. Add it in Settings > Features.",
    ))
    window._apply(gui_controller.Result("speak", False, {}, error="Voice replies could not start."))
    assert "Settings > Features" in window._whisper_status.text()
    assert "Settings > Features" in window._voice_status.text()
    assert window._tts_status.text() == "Voice replies could not start."
    assert window._model_status.text() == ""


def test_reserved_stop_offers_release_and_stop(win, monkeypatch):
    window, fake, _desktop = win
    from PySide6 import QtWidgets

    asked = []
    answer = {"value": QtWidgets.QMessageBox.StandardButton.No}
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question",
        staticmethod(lambda *a, **k: (asked.append(a[1]), answer["value"])[1]),
    )
    refusal = gui_controller.Result(
        "stop", False, {"reserved": True, "then": "stop"},
        error=gui_controller.RESERVED_MODEL_MESSAGE,
    )
    window._apply(refusal)
    assert asked and "coding terminal" in asked[0]
    assert not any(c[0] == "request_release_and" for c in fake.calls)
    answer["value"] = QtWidgets.QMessageBox.StandardButton.Yes
    window._apply(refusal)
    assert fake.calls[-1] == ("request_release_and", "stop")
    window._apply(gui_controller.Result(
        "free_gpu", False, {"reserved": True, "then": "free_gpu"},
        error=gui_controller.RESERVED_MODEL_MESSAGE,
    ))
    assert fake.calls[-1] == ("request_release_and", "free_gpu")


def test_free_gpu_message_names_the_model_it_stopped(win):
    window, _fake, _desktop = win
    window._apply(gui_controller.Result("gpu_free", True, {
        "freed": 0, "stopped_model": "local-chat",
        "others": ["[Insufficient Permissions]", "game.exe"],
        "vram_free_mb": 9000.0, "running_model_id": None, "running_port": None,
    }))
    text = window._model_status.text()
    assert "Stopped Local Chat" in text and "game.exe" in text
    assert "Insufficient" not in text


def test_every_disabled_primary_control_explains_why(win):
    window, _fake, _desktop = win
    for button in (window._stop_btn, window._chat_btn, window._save_btn,
                   window._vision_btn, window._hub_download_btn, window._register_btn):
        assert not button.isEnabled()
        assert button.toolTip(), button.text()
    window._ui.running_model_id = "local-chat"
    window._refresh_buttons()
    assert window._stop_btn.isEnabled()
    assert window._stop_btn.toolTip() == "Stop the running model and free its memory"
    assert not window._start_btn.isEnabled()
    assert window._start_btn.toolTip() == "This model is already running."


def test_header_keeps_free_up_gpu_visible_when_the_status_is_long(win):
    window, _fake, _desktop = win
    window.resize(1366, 850)
    window.show()
    window._header_status.setText("An extremely long model name that keeps going " * 6)
    from PySide6 import QtWidgets

    QtWidgets.QApplication.processEvents()
    header = window._header_offload_btn.parentWidget()
    button = window._header_offload_btn
    assert button.text() == "Free up GPU"
    assert button.geometry().right() <= header.width()
    assert button.width() >= button.sizeHint().width()


def test_health_chip_explains_itself(win):
    window, _fake, _desktop = win

    def result(status, name, detail, remedy=None):
        return SimpleNamespace(
            status=SimpleNamespace(value=status), name=name, detail=detail, remedy=remedy
        )

    window._health_report = SimpleNamespace(results=[
        result("PASS", "python", "ok"),
        result("WARNING", "vram", "usable VRAM is tight", "close other GPU apps"),
    ])
    window._update_health_chip()
    assert window._health_btn.text() == "Health: Needs attention"
    assert "usable VRAM is tight" in window._health_btn.toolTip()
    assert "close other GPU apps" in window._health_details_text()
    assert "python" not in window._health_details_text()


def test_shared_status_label_mirrors_to_other_pages(win):
    window, _fake, _desktop = win
    window._model_status.setText("Stopping...")
    assert window._models_note.text() == "Stopping..."
    assert window._code_note.text() == "Stopping..."


def test_memory_search_runs_on_first_visit_to_its_code_tab(win):
    window, fake, _desktop = win
    assert ("request_memory_search", "") not in fake.calls
    window._go_to("Code", "Search past chats")
    assert ("request_memory_search", "") in fake.calls


def test_open_chat_works_without_a_model_when_the_router_can_load_one(win):
    window, fake, _desktop = win
    assert not window._chat_btn.isEnabled()
    assert "Start a model" in window._chat_btn.toolTip()
    fake.chat_needs_no_model = lambda: True
    window._refresh_buttons()
    assert window._chat_btn.isEnabled()
    assert "pick any model there" in window._chat_hint.text()
    window._chat_btn.click()
    assert fake.calls[-1] == ("request_chat",)


def test_controller_chat_needs_no_model_follows_router_and_webui(monkeypatch):
    import webui
    from tests.test_gui_controller import _controller

    gc = _controller()
    monkeypatch.setattr(webui, "webui_available", lambda settings: True)
    gc._settings.router.enabled = True
    gc._settings.chat.preferred_ui = "openwebui"
    assert gc.chat_needs_no_model() is True
    gc._settings.router.enabled = False
    assert gc.chat_needs_no_model() is False
    gc._settings.router.enabled = True
    monkeypatch.setattr(webui, "webui_available", lambda settings: False)
    assert gc.chat_needs_no_model() is False


def test_advanced_tab_has_one_shared_picker_that_stays_in_sync(win):
    window, fake, _desktop = win
    # My models selection drives the Advanced picker, summary and button names.
    window._model_table.selectRow(1)
    assert window._settings_model.currentData() == window._selected_id == "local-code"
    assert window._delete_btn.text() == f"Delete {window._settings_model.currentText()}..."
    assert window._autotune_btn.text() == "Auto-tune this model"
    assert window._adv_summary.text()
    # ... and the Advanced picker drives My models back.
    window._settings_model.setCurrentIndex(0)
    assert window._selected_id == "local-chat"
    assert window._model_table.item(window._model_table.currentRow(), 0).data(
        0x0100
    ) == "local-chat"
    # No second picker and no stray monitor line on this tab.
    assert len(window.pages["Models"].findChildren(type(window._settings_model))) == 2  # picker + hub filter
    assert not window._models_monitor.isVisible()
    # "Add to my models" only exists for discovered fine-tunes.
    assert not window._register_btn.isVisible()
    window._models[0]["source"] = "discovered"
    window._update_advanced_header()
    assert not window._register_btn.isHidden()
