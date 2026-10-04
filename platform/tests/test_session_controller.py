"""Readiness failures must never open a terminal against the wrong model."""
from types import SimpleNamespace
import queue

import gui_controller
import harness_launch
from session_store import SessionStore


def controller(tmp_path, running="chosen"):
    gc = object.__new__(gui_controller.GuiController)
    gc.result_q = queue.Queue()
    gc._settings = SimpleNamespace(data_dir=tmp_path)
    gc._registry = SimpleNamespace(get=lambda name: SimpleNamespace(context_size=8192) if name == "chosen" else None)
    gc._controller = SimpleNamespace(running_model_id=running)
    gc._active_port = 8080 if running else None
    return gc


def test_local_resume_records_profile_after_real_readiness(tmp_path, monkeypatch):
    gc = controller(tmp_path)
    calls = []
    monkeypatch.setattr(harness_launch, "detect_executable", lambda _: "codex")
    monkeypatch.setattr(harness_launch, "spawn_in_terminal", lambda *args, **kw: calls.append(args))
    gc._do_session_launch({"choice": "codex", "model_id": "chosen", "project_dir": str(tmp_path), "resume_id": "abc-123"})
    result = gc.result_q.get()
    assert result.ok and len(calls) == 1
    assert gc._controller._session_reserved_model == "chosen"
    assert SessionStore(tmp_path).all()[("codex", "abc-123")]["model_id"] == "chosen"


def test_failed_model_start_never_spawns_or_saves_profile(tmp_path, monkeypatch):
    gc = controller(tmp_path, running=None)
    gc._do_start = lambda _: None
    monkeypatch.setattr(harness_launch, "detect_executable", lambda _: "codex")
    monkeypatch.setattr(harness_launch, "spawn_in_terminal", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("spawned")))
    gc._do_session_launch({"choice": "codex", "model_id": "chosen", "project_dir": str(tmp_path), "resume_id": "abc-123"})
    assert not gc.result_q.get().ok
    assert SessionStore(tmp_path).all() == {}


def test_reserved_model_refuses_switch_before_stopping():
    import pytest
    from services import ModelController
    model = object.__new__(ModelController)
    model._session_reserved_model = "first"
    with pytest.raises(ValueError, match="coding session"):
        model.switch("second")


def test_reserved_model_blocks_stop_and_free_gpu(tmp_path):
    gc = controller(tmp_path)
    gc._controller._session_reserved_model = "chosen"
    gc._do_stop()
    assert not gc.result_q.get().ok
    gc._do_free_gpu()
    assert not gc.result_q.get().ok


def test_annotation_failure_after_spawn_reports_open_terminal(tmp_path, monkeypatch):
    import sqlite3
    gc = controller(tmp_path)
    monkeypatch.setattr(harness_launch, "detect_executable", lambda _: "codex")
    monkeypatch.setattr(harness_launch, "spawn_in_terminal", lambda *a, **kw: None)
    monkeypatch.setattr(SessionStore, "update", lambda *a, **kw: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
    gc._do_session_launch({"choice": "codex", "model_id": "chosen", "project_dir": str(tmp_path), "resume_id": "abc-123"})
    result = gc.result_q.get()
    assert result.ok and "Terminal opened" in result.payload["warning"]
    assert gc._controller._session_reserved_model == "chosen"
