"""Phase 1 fixes: Talk model choice, the reserved-model release path, plain
voice errors, and the fresh-data-dir fine-tune scan."""

import queue
from types import SimpleNamespace

import gui_controller
import launcher
from services import ServiceStatus
from tests.test_gui_controller import (
    FakeManager,
    FakeModelController,
    FakeRegistry,
    FakeWhisper,
    _controller,
    _dispatch_next,
    _model,
)


class _Registry:
    def __init__(self, models):
        self._models = {m.id: m for m in models}

    def get(self, model_id):
        return self._models.get(model_id)

    def launchable(self):
        # Mirrors ModelRegistry.launchable(): installed rows with a file.
        return [m for m in self._models.values() if m.location and m.status == "installed"]


def test_talk_model_prefers_running_then_selected_then_first_launchable():
    reg = _Registry([_model("a"), _model("b"), _model("c", location="")])
    assert launcher.resolve_assistant_model("b", "a", "", reg) == ("a", "")
    assert launcher.resolve_assistant_model("b", None, "", reg) == ("b", "")
    # A fresh install: no running model, no selection, no default.
    assert launcher.resolve_assistant_model(None, None, "", reg) == ("a", "")
    # A default that points at nothing is skipped, not fatal.
    assert launcher.resolve_assistant_model(None, None, "ghost", reg) == ("a", "")


def test_talk_model_skips_a_model_that_is_turned_off():
    import dataclasses

    off = dataclasses.replace(_model("off"), status="disabled")
    reg = _Registry([off, _model("a")])
    # Neither the selection nor the default may pick a row that cannot start.
    assert launcher.resolve_assistant_model("off", None, "", reg) == ("a", "")
    assert launcher.resolve_assistant_model(None, None, "off", reg) == ("a", "")


def test_talk_model_with_nothing_installed_gives_a_plain_message():
    model_id, problem = launcher.resolve_assistant_model(None, None, "", _Registry([]))
    assert model_id == ""
    assert "Models" in problem
    assert "launcher.default_model" not in problem


def test_start_assistant_forwards_the_selected_model_only_when_set():
    seen = []
    gc = _controller()
    gc._assistant_start_fn = lambda events, voice, speak, *rest: (
        seen.append(rest) or SimpleNamespace(port=1, stop=lambda: None)
    )
    gc.request_start_assistant("v", True)
    _dispatch_next(gc)
    gc.note_selected_model("qwen3-14b")
    gc._assistant_handle = None
    gc.request_start_assistant("v", True)
    _dispatch_next(gc)
    assert seen == [(), ("qwen3-14b",)]


def test_stop_refused_by_a_coding_terminal_can_be_released_and_stopped():
    ctrl = FakeModelController()
    ctrl.running_model_id = "qwen3-14b"
    ctrl._session_reserved_model = "qwen3-14b"
    gc = _controller(model_ctrl=ctrl)
    gc.request_stop()
    _dispatch_next(gc)
    refused = gc.result_q.get_nowait()
    assert refused.ok is False
    assert refused.payload["reserved"] is True
    assert refused.payload["then"] == "stop"
    assert "Sessions" not in refused.error
    assert ctrl.calls == []

    gc.request_release_and("stop")
    _dispatch_next(gc)
    assert ctrl._session_reserved_model is None
    assert ctrl.calls == [("stop",)]
    assert gc.result_q.get_nowait().ok


def test_free_gpu_refused_by_a_coding_terminal_names_the_next_step():
    ctrl = FakeModelController()
    ctrl._session_reserved_model = "qwen3-14b"
    gc = _controller(model_ctrl=ctrl)
    gc.request_free_gpu()
    _dispatch_next(gc)
    refused = gc.result_q.get_nowait()
    assert refused.kind == "free_gpu"
    assert refused.payload == {"reserved": True, "then": "free_gpu"}


def test_plain_voice_error_points_at_features_not_settings_files():
    msg = gui_controller.plain_voice_error(
        "whisper-server path is not configured; set paths.whisper in settings.yaml",
        "Voice input",
    )
    assert msg == "Voice input isn't installed. Add it in Settings > Features."
    assert "yaml" not in gui_controller.plain_voice_error("did not start", "Voice input")


def test_listen_failure_reports_the_real_reason_not_a_generic_one():
    gc = _controller()

    def listen(seconds, emit):
        emit("cannot start whisper-stream: whisper-stream path is not configured")
        return False

    gc._listen_fn = listen
    gc.request_listen(5.0)
    _dispatch_next(gc)
    results = []
    while True:
        try:
            results.append(gc.result_q.get_nowait())
        except queue.Empty:
            break
    final = results[-1]
    assert final.kind == "listen" and final.ok is False
    assert final.error == "Voice input isn't installed. Add it in Settings > Features."
    assert not any(r.kind == "transcript" for r in results)


def test_whisper_toggle_failure_is_plain():
    class Broken(FakeWhisper):
        def start(self):
            raise ValueError("whisper-server path is not configured")

    gc = _controller(whisper=Broken())
    gc.request_whisper_toggle()
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert result.ok is False
    assert "Settings > Features" in result.error


def test_voice_readiness_reads_the_configured_paths():
    gc = _controller()
    assert gc.voice_readiness() == {
        "listen": False, "whisper_server": False, "speak": False,
    }
    gc._settings.paths.whisper_stream = "x"
    gc._settings.paths.whisper_model = "y"
    assert gc.voice_readiness()["listen"] is True
