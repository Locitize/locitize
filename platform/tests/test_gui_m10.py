"""Headless tests for the M10 Command Center consolidation (Architecture M10).

These tests NEVER import gui.py and NEVER construct a Tk() root, so they run on a
headless build machine (G7). They drive the M10 seams with injected fakes:

- GuiSttSource talk-gate window semantics (keyword `gui_talk`): a Talk click opens
  ONE capture window over a REAL PushToTalkSttSource + a fake mic and returns the
  utterance; a STOP sentinel ends the loop; a no-speech window reloops and never
  fabricates a turn.
- GuiController assistant-session start/stop wiring (keyword `gui_assistant_session`):
  start marshals assistant_started, talk/interrupt route to the handle, end stops the
  handle, a factory error degrades to an error Result, and model-exclusive surfaces
  refuse while a session is live.
- No-orphan on window close covering the assistant (keyword `gui_shutdown`): shutdown
  stops the live session before stop_all, exactly once, idempotently.
"""

from __future__ import annotations

import queue as _queue

from config import Model, Settings
from services import ServiceStatus

from gui_controller import (
    GuiController,
    GuiSttSource,
    _STOP_SENTINEL,  # noqa: PLC2701 - the test drives the gate sentinels directly
    _TALK_TOKEN,  # noqa: PLC2701
)


# --------------------------------------------------------------------------- #
# Minimal fakes (no real process, no GPU, no socket, no Tk).
# --------------------------------------------------------------------------- #


class _FakeModelController:
    def __init__(
        self, running_model_id: str | None = None, running_port: int | None = None
    ) -> None:
        self.running_model_id = running_model_id
        self.running_port = running_port

    def snapshot(self):
        services = []
        if self.running_model_id is not None:
            services.append(
                {
                    "status": "RUNNING",
                    "model_id": self.running_model_id,
                    "port": self.running_port,
                }
            )
        return {"services": services}


class _FakeWhisper:
    def is_running(self) -> bool:
        return False


class _FakeManager:
    def __init__(self) -> None:
        self.stop_all_calls = 0

    def stop_all(self, grace_s=None) -> None:
        self.stop_all_calls += 1


class _FakeRegistry:
    def __init__(self, models) -> None:
        self._models = models

    def all(self):
        return list(self._models)

    def get(self, model_id):
        return next((m for m in self._models if m.id == model_id), None)


def _model():
    return Model(
        id="qwen3-14b",
        name="Qwen3 14B",
        description="",
        location="/locitize-test/models/x.gguf",
        context_size=32768,
        gpu_layers=-1,
    )


def _dispatch_next(gc: GuiController) -> None:
    """Run the ops-worker logic for one queued command synchronously (no thread)."""
    gc._dispatch(gc.command_q.get_nowait())


class _FakeMic:
    """A fake MicSegments seam: yields queued open-window segments, counts flushes."""

    def __init__(self, open_=None) -> None:
        self._open = list(open_ or [])
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1

    def poll_segment(self, timeout_s: float):
        return self._open.pop(0) if self._open else None


# --------------------------------------------------------------------------- #
# GuiSttSource talk-gate window semantics (keyword: gui_talk)
# --------------------------------------------------------------------------- #


def test_gui_talk_click_opens_one_window_and_returns_utterance():
    """gui_talk: a Talk click opens ONE capture window (mic flushed) and returns the
    captured utterance; the callbacks fire idle -> listening -> thinking and on_user
    carries the text (Architecture M10.3 a/b)."""
    states: list[str] = []
    users: list[str] = []
    gate: "_queue.Queue" = _queue.Queue()
    mic = _FakeMic(open_=["what is the capital of Kenya", None])
    src = GuiSttSource(
        mic, gate, on_state=states.append, on_user=users.append,
        settle_s=0.0, poll_s=0.0,
    )
    src.talk()  # preload one Talk click so next_utterance does not block
    utterance = src.next_utterance()
    assert utterance == "what is the capital of Kenya"
    assert mic.flush_calls == 1  # exactly one window opened, mic flushed
    assert states == ["idle", "listening", "thinking"]
    assert users == ["what is the capital of Kenya"]


def test_gui_talk_read_line_maps_talk_to_window_and_stop_to_none():
    """gui_talk (pure read_line): a Talk token -> open a window ("") + on_state
    listening; the STOP sentinel -> None (end the loop)."""
    states: list[str] = []
    gate: "_queue.Queue" = _queue.Queue()
    src = GuiSttSource(_FakeMic(), gate, on_state=states.append)
    gate.put(_TALK_TOKEN)
    assert src._read_line() == ""
    assert states == ["listening"]
    gate.put(_STOP_SENTINEL)
    assert src._read_line() is None


def test_gui_talk_stop_sentinel_ends_session_without_opening_a_window():
    """gui_talk: stop() ends the session (returns None) and never opens a window."""
    gate: "_queue.Queue" = _queue.Queue()
    mic = _FakeMic(open_=["ignored"])
    src = GuiSttSource(mic, gate, settle_s=0.0, poll_s=0.0)
    src.stop()
    assert src.next_utterance() is None
    assert mic.flush_calls == 0


def test_gui_talk_no_speech_window_reloops_no_fabricated_turn():
    """gui_talk: a no-speech window reloops INSIDE PushToTalkSttSource and never yields
    a fabricated turn; a second Talk click with real speech returns exactly one turn
    (Architecture M10.3 d). An injected clock advances on each silent poll so the
    silent window's max_capture expires instantly (no real-time wait)."""
    now = [0.0]
    users: list[str] = []
    gate: "_queue.Queue" = _queue.Queue()

    class _WindowMic:
        def __init__(self) -> None:
            self.flush_calls = 0
            self._said = False

        def flush(self) -> None:
            self.flush_calls += 1

        def poll_segment(self, timeout_s: float):
            now[0] += 0.5  # advance the injected clock so a silent window ends fast
            if self.flush_calls >= 2 and not self._said:
                self._said = True
                return "hello there"
            return None

    mic = _WindowMic()
    src = GuiSttSource(
        mic, gate, on_user=users.append,
        max_capture_s=1.0, settle_s=0.0, poll_s=0.0, clock=lambda: now[0],
    )
    src.talk()  # first window: silent -> reloop with the no-speech notice
    src.talk()  # second window: real speech
    assert src.next_utterance() == "hello there"
    assert users == ["hello there"]  # exactly one user turn, never fabricated
    assert mic.flush_calls == 2  # two windows opened


# --------------------------------------------------------------------------- #
# Controller assistant-session start/stop wiring (keyword: gui_assistant_session)
# --------------------------------------------------------------------------- #


class _FakeHandle:
    """Records the handle calls the controller makes on a live assistant session."""

    def __init__(self, port: int = 8080) -> None:
        self.port = port
        self.talk_calls = 0
        self.interrupt_calls = 0
        self.stop_calls = 0

    def talk(self) -> None:
        self.talk_calls += 1

    def interrupt(self) -> None:
        self.interrupt_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def _assistant_controller(
    start_fn=None, manager=None, model_controller=None, **kw
) -> GuiController:
    return GuiController(
        Settings(),
        _FakeRegistry([_model()]),
        model_controller or _FakeModelController(),
        _FakeWhisper(),
        manager or _FakeManager(),
        fetch=lambda port, path: None,
        assistant_start_fn=start_fn,
        **kw,
    )


def test_gui_assistant_session_start_marshals_started_and_records_port():
    """gui_assistant_session: start calls the factory with the voice/speak choice,
    records the resolved model port (for the monitor), and marshals assistant_started."""
    handle = _FakeHandle(port=9001)
    seen: dict = {}

    def start_fn(events, voice, speak):
        seen["voice"], seen["speak"] = voice, speak
        return handle

    model_controller = _FakeModelController("qwen3-14b", 9001)
    gc = _assistant_controller(start_fn=start_fn, model_controller=model_controller)
    gc.request_start_assistant("am_michael", True)
    _dispatch_next(gc)
    r = gc.result_q.get_nowait()
    assert r.kind == "assistant_started" and r.ok
    assert r.payload == {
        "port": 9001,
        "running_model_id": "qwen3-14b",
        "running_port": 9001,
    }
    assert gc._active_port == 9001
    assert gc._assistant_handle is handle
    assert seen == {"voice": "am_michael", "speak": True}


def test_gui_assistant_session_talk_and_interrupt_route_to_handle():
    """gui_assistant_session: request_talk -> handle.talk(); request_interrupt ->
    handle.interrupt() (both non-blocking, on the UI thread)."""
    handle = _FakeHandle()
    gc = _assistant_controller(start_fn=lambda e, v, s: handle)
    gc.request_start_assistant()
    _dispatch_next(gc)
    gc.result_q.get_nowait()
    gc.request_talk()
    gc.request_interrupt()
    assert handle.talk_calls == 1
    assert handle.interrupt_calls == 1


def test_gui_talk_while_speaking_barges_in_then_listens():
    """Talk during thinking/speaking interrupts first, then opens a capture window."""
    handle = _FakeHandle()
    gc = _assistant_controller(start_fn=lambda e, v, s: handle)
    gc.request_start_assistant()
    _dispatch_next(gc)
    gc.result_q.get_nowait()
    gc._assistant_ui_state = "speaking"
    gc.request_talk()
    assert handle.interrupt_calls == 1
    assert handle.talk_calls == 1
    gc._assistant_ui_state = "idle"
    gc.request_talk()
    assert handle.interrupt_calls == 1
    assert handle.talk_calls == 2


def test_gui_assistant_session_end_stops_handle_and_marshals_ended():
    """gui_assistant_session: End enqueues a stop that calls handle.stop() exactly once
    and marshals assistant_ended; the handle reference is cleared."""
    handle = _FakeHandle()
    gc = _assistant_controller(start_fn=lambda e, v, s: handle)
    gc.request_start_assistant()
    _dispatch_next(gc)
    gc.result_q.get_nowait()
    gc.request_end_assistant()
    _dispatch_next(gc)
    ended = gc.result_q.get_nowait()
    assert ended.kind == "assistant_ended" and ended.ok
    assert handle.stop_calls == 1
    assert gc._assistant_handle is None


def test_gui_assistant_session_factory_error_degrades_to_error_result():
    """gui_assistant_session: a factory error is turned into an honest assistant_error
    Result, never raised across the boundary, and leaves no live handle."""
    def boom(events, voice, speak):
        raise RuntimeError("chat model did not start")

    gc = _assistant_controller(start_fn=boom)
    gc.request_start_assistant()
    _dispatch_next(gc)  # must not raise
    r = gc.result_q.get_nowait()
    assert r.kind == "assistant_error" and not r.ok
    assert "chat model did not start" in (r.error or "")
    assert gc._assistant_handle is None


def test_gui_assistant_session_describe_and_benchmark_refuse_while_live():
    """gui_assistant_session: Vision and Benchmark are model-exclusive, so both refuse
    with an honest remedy Result (and enqueue nothing) while a Talk session is live."""
    gc = _assistant_controller(start_fn=lambda e, v, s: _FakeHandle())
    gc.request_start_assistant()
    _dispatch_next(gc)
    gc.result_q.get_nowait()
    gc.request_describe("/locitize-test/img.png", "what is this")
    gc.request_benchmark("qwen3-14b")
    assert gc.command_q.empty()  # neither was enqueued
    d = gc.result_q.get_nowait()
    b = gc.result_q.get_nowait()
    assert d.kind == "describe" and not d.ok
    assert b.kind == "benchmark" and not b.ok


def test_gui_assistant_session_memory_search_marshals_hits():
    """gui_assistant_session: the read-only memory search runs on the ops worker and
    marshals its hits (always available, no model, no write)."""
    gc = _assistant_controller(
        start_fn=None,
        memory_search_fn=lambda q: [{"role": "user", "text": "hi " + q}],
    )
    gc.request_memory_search("kenya")
    _dispatch_next(gc)
    r = gc.result_q.get_nowait()
    assert r.kind == "memory_search" and r.ok
    assert r.payload["hits"] == [{"role": "user", "text": "hi kenya"}]


def test_gui_assistant_session_benchmark_marshals_generation_tok_s():
    """A completed benchmark marshals measured throughput with an explicit unit."""
    gc = _assistant_controller(
        start_fn=None,
        benchmark_fn=lambda mid: (True, "1 ok, 0 failed of 1 scenarios", 87.5),
    )
    gc.request_benchmark("qwen3-14b")
    _dispatch_next(gc)
    r = gc.result_q.get_nowait()
    assert r.kind == "benchmark" and r.ok
    assert r.payload["model_id"] == "qwen3-14b"
    assert r.payload["score"] == 87.5
    assert r.payload["score_display"] == "87.5 tok/s"


# --------------------------------------------------------------------------- #
# No-orphan on window close covering the assistant session (keyword: gui_shutdown)
# --------------------------------------------------------------------------- #


def test_gui_shutdown_stops_live_assistant_then_stop_all_once():
    """gui_shutdown: a window close with a LIVE assistant session stops the session
    (handle.stop) BEFORE stop_all, and stop_all runs exactly once; a second shutdown
    is idempotent (the shared-manager no-orphan guarantee extended to the assistant)."""
    handle = _FakeHandle()
    manager = _FakeManager()
    gc = _assistant_controller(start_fn=lambda e, v, s: handle, manager=manager)
    gc.request_start_assistant()
    _dispatch_next(gc)
    gc.result_q.get_nowait()
    assert gc._assistant_handle is handle
    gc.shutdown()
    assert handle.stop_calls == 1
    assert manager.stop_all_calls == 1
    gc.shutdown()  # idempotent
    assert handle.stop_calls == 1
    assert manager.stop_all_calls == 1


def test_gui_shutdown_without_assistant_still_stops_services_once():
    """gui_shutdown: with no live session, shutdown still calls stop_all exactly once
    (the pre-M10 no-orphan behavior is preserved)."""
    manager = _FakeManager()
    gc = _assistant_controller(start_fn=None, manager=manager)
    gc.shutdown()
    assert manager.stop_all_calls == 1
