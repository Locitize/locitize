"""Scripted Tk open/close smoke for the M10 Command Center (no real services).

Builds the REAL LocitizeGui over a fake GuiController (no model/mic/TTS/subprocess),
verifies the M10 Talk panel + Vision + Memory + Benchmark surfaces are present, drives
the pump once to render an assistant_started + a "You:"/"LOCITIZE:" turn, then closes the
window and asserts a clean, single-shot shutdown (no orphan). This exercises gui.py's
widget wiring and pump on a real Tk root without touching hardware.

Run: Codebase/.venv/Scripts/python Codebase/platform/scripts/m10_gui_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PLATFORM))

import queue

import tkinter as tk

import gui
import gui_controller
from gui_controller import Result


class _FakeController:
    """A stand-in GuiController: real queues, no threads, no services."""

    def __init__(self) -> None:
        self.result_q: "queue.Queue[Result]" = queue.Queue()
        self.command_q: "queue.Queue" = queue.Queue()
        self.calls: list[str] = []
        self._shutdown_calls = 0

    # read accessors used by gui.py at build time
    def list_models(self):
        return [
            {
                "id": "qwen3-14b",
                "name": "Qwen3 14B",
                "location": "/locitize-test/models/x.gguf",
                "status": "installed",
                "gpu_layers": -1,
                "context_size": 32768,
                "launchable": True,
                "size_bytes": None,
                "size_display": "-",
                "benchmark_score": None,
                "score_display": "-",
            }
        ]

    def available_voices(self):
        return ["am_michael", "af_bella"]

    def start_threads(self):
        self.calls.append("start_threads")

    def request_start_assistant(self, voice, speak):
        self.calls.append(f"start_assistant:{voice}:{speak}")

    def request_talk(self):
        self.calls.append("talk")

    def request_interrupt(self):
        self.calls.append("interrupt")

    def request_end_assistant(self):
        self.calls.append("end_assistant")

    def request_memory_search(self, q):
        self.calls.append(f"memory:{q}")

    def request_describe(self, path, prompt):
        self.calls.append(f"describe:{path}")

    def request_benchmark(self, model_id):
        self.calls.append(f"benchmark:{model_id}")

    def shutdown(self):
        self._shutdown_calls += 1
        self.calls.append("shutdown")


def main() -> int:
    root = tk.Tk()
    root.withdraw()  # do not flash a window during the smoke
    gc = _FakeController()
    app = gui.LocitizeGui(root, gc, health="PASS")

    # The M10 Talk panel + its controls exist.
    assert hasattr(app, "_talk_btn"), "Talk button missing"
    assert hasattr(app, "_conversation"), "conversation view missing"
    assert hasattr(app, "_assistant_start_btn"), "Start assistant missing"
    assert hasattr(app, "_assistant_end_btn"), "End assistant missing"
    assert hasattr(app, "_interrupt_btn"), "Interrupt missing"
    assert hasattr(app, "_vision_btn"), "Vision Describe missing"
    assert hasattr(app, "_memory_result"), "Memory results missing"
    assert hasattr(app, "_benchmark_btn"), "Benchmark button missing"
    # Talk starts disabled (no session yet).
    assert str(app._talk_btn["state"]) == "disabled"
    print("[1] M10 Talk/Vision/Memory/Benchmark widgets present; Talk starts disabled")

    # Simulate a Start assistant click -> the intent reaches the controller.
    app._on_start_assistant()
    assert any(c.startswith("start_assistant:") for c in gc.calls)

    # Marshal an assistant_started + one turn through the REAL pump/apply path.
    gc.result_q.put(Result("assistant_started", True, {"port": 8080}))
    gc.result_q.put(Result("assistant_state", True, {"state": "idle"}))
    gc.result_q.put(Result("assistant_user", True, {"text": "what is the capital of France"}))
    gc.result_q.put(Result("assistant_reply", True, {"line": "LOCITIZE: Paris."}))
    app._drain()  # the pump's drain applies every queued Result on the UI thread

    # Talk is now enabled (session live + idle); the conversation view shows the turn.
    assert app._assistant_live is True
    assert str(app._talk_btn["state"]) == "normal"
    text = app._conversation.get("1.0", "end")
    assert "You: what is the capital of Kenya" in text
    assert "LOCITIZE: Paris." in text
    print("[2] pump rendered a You/LOCITIZE turn; Talk enabled while live+idle")

    # Speaking state disables Talk; back to idle re-enables it (M10.3 state machine).
    gc.result_q.put(Result("assistant_state", True, {"state": "speaking"}))
    app._drain()
    assert str(app._talk_btn["state"]) == "disabled"
    gc.result_q.put(Result("assistant_state", True, {"state": "idle"}))
    app._drain()
    assert str(app._talk_btn["state"]) == "normal"
    print("[3] Talk state machine: speaking disables, idle re-enables the button")

    # Close the window -> gui_controller.shutdown() called exactly once, then destroy.
    app._on_close()
    assert gc._shutdown_calls == 1, "shutdown must run exactly once on window close"
    print("[4] window close called controller.shutdown() exactly once; clean teardown")

    print("GUI SMOKE OK: real Tk Talk panel present, pump renders turns, clean close.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
