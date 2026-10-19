"""Voice input's real state reaches the view, whatever started it.

Owner-observed 2026-10-19: voice input was already running from startup while the
Voice page offered "Turn on voice input", so the click switched it off. The
monitor loop now publishes the state on every change.
"""

from __future__ import annotations

import queue

from config import Settings
from gui_controller import GuiController


class _Whisper:
    def __init__(self, running):
        self.running = running

    def is_running(self):
        return self.running

    def stop(self):
        self.running = False


class _Registry:
    def all(self):
        return []


def _controller(whisper) -> GuiController:
    return GuiController(Settings(), _Registry(), object(), whisper, object(),
                         fetch=lambda port, path: None)


def _drain(gc):
    out = []
    while True:
        try:
            out.append(gc.result_q.get_nowait())
        except queue.Empty:
            return out


def test_running_voice_input_is_published_once_and_then_on_change():
    whisper = _Whisper(running=True)
    gc = _controller(whisper)
    gc._publish_whisper_state()
    gc._publish_whisper_state()
    results = _drain(gc)
    assert [(r.kind, r.payload) for r in results] == [("whisper", {"running": True})]

    whisper.running = False  # stopped elsewhere, e.g. Talk ended
    gc._publish_whisper_state()
    assert [r.payload for r in _drain(gc)] == [{"running": False}]


def test_toggle_after_startup_turns_it_off_and_is_not_republished():
    whisper = _Whisper(running=True)
    gc = _controller(whisper)
    gc._publish_whisper_state()
    _drain(gc)
    gc._do_whisper_toggle()
    assert [r.payload for r in _drain(gc)] == [{"running": False}]
    gc._publish_whisper_state()
    assert _drain(gc) == []


def test_no_whisper_controller_is_not_an_error():
    gc = _controller(object())
    gc._publish_whisper_state()
    assert _drain(gc) == []
