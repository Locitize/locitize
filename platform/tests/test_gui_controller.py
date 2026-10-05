"""Headless tests for the GUI controller seam (Architecture G7).

These tests NEVER import gui.py and NEVER construct a Tk() root, so they run on a
headless build machine. They drive gui_controller with injected fakes (fake
ModelController / whisper controller / ServiceManager) and canned /metrics + /slots
payloads, exercising:

- AC10 (keyword `gui_command`): command delegation to the existing controllers and
  the gpu_layers/context_size validation table.
- AC11 (keyword `gui_metrics`): metrics/slots parsing that degrades honestly and
  never fabricates a number.
- AC14 (keyword `gui_shutdown`): shutdown joins both threads and calls stop_all
  exactly once (the no-orphan path).
"""

from __future__ import annotations

import threading

from config import Model, Settings
from services import ServiceStatus

import gui_controller
from gui_controller import GuiController, UiState


# --------------------------------------------------------------------------- #
# Fakes (no real process, no GPU, no socket).
# --------------------------------------------------------------------------- #


class FakeModelController:
    """Records start/switch/stop calls; reports a running model with a port."""

    def __init__(self, start_status: ServiceStatus = ServiceStatus.RUNNING) -> None:
        self.running_model_id: str | None = None
        self.calls: list[tuple] = []
        self.spec_kwargs: list[dict] = []
        self._start_status = start_status
        self._port = 8080

    def start(self, model_id, ctx=None, gpu=None, **spec_kwargs):
        self.calls.append(("start", model_id))
        # Recorded SEPARATELY from calls so the tuple shape every existing
        # assertion uses stays exactly as it was (2026-09-02, when the
        # one-launch reasoning override began flowing through here).
        self.spec_kwargs.append(spec_kwargs)
        if self._start_status is ServiceStatus.RUNNING:
            self.running_model_id = model_id
        return self._start_status

    def switch(self, model_id, ctx=None, gpu=None, **spec_kwargs):
        self.calls.append(("switch", model_id))
        self.spec_kwargs.append(spec_kwargs)
        self.running_model_id = model_id
        return ServiceStatus.RUNNING

    @property
    def running_port(self):
        # Mirrors SingleServiceController.running_port: the resolved port while
        # a model runs, None otherwise. Absent here until 2026-09-03, which let
        # _check_running_model's guard swallow an AttributeError in tests.
        return self._port if self.running_model_id else None

    def stop(self):
        self.calls.append(("stop",))
        self.running_model_id = None
        return ServiceStatus.STOPPED

    def snapshot(self):
        if self.running_model_id is None:
            return {"running_model": None, "services": []}
        return {
            "running_model": self.running_model_id,
            "services": [
                {"model_id": self.running_model_id, "status": "RUNNING", "port": self._port}
            ],
        }


class FakeWhisper:
    def __init__(self) -> None:
        self._running = False
        self.calls: list[str] = []

    def is_running(self) -> bool:
        return self._running

    def start(self) -> ServiceStatus:
        self.calls.append("start")
        self._running = True
        return ServiceStatus.RUNNING

    def stop(self) -> ServiceStatus:
        self.calls.append("stop")
        self._running = False
        return ServiceStatus.STOPPED


class FakeManager:
    def __init__(self) -> None:
        self.stop_all_calls = 0

    def stop_all(self, grace_s=None) -> None:
        self.stop_all_calls += 1


class FakeRegistry:
    def __init__(self, models: list[Model]) -> None:
        self._models = models

    def all(self) -> list[Model]:
        return list(self._models)

    def get(self, model_id: str) -> Model | None:
        return next((m for m in self._models if m.id == model_id), None)


def _model(
    model_id="qwen3-14b", location="/locitize-test/models/x.gguf", benchmark_score=None
) -> Model:
    return Model(
        id=model_id,
        name="Qwen3 14B",
        description="",
        location=location,
        context_size=32768,
        gpu_layers=-1,
        benchmark_score=benchmark_score,
    )


def _controller(model_ctrl=None, whisper=None, manager=None) -> GuiController:
    return GuiController(
        Settings(),
        FakeRegistry([_model()]),
        model_ctrl or FakeModelController(),
        whisper or FakeWhisper(),
        manager or FakeManager(),
        fetch=lambda port, path: None,  # never hit a socket in tests
    )


def _dispatch_next(gc: GuiController) -> None:
    """Synchronously run the ops-worker logic for one queued command (no thread)."""
    gc._dispatch(gc.command_q.get_nowait())


# --------------------------------------------------------------------------- #
# AC10 - command delegation (keyword: gui_command)
# --------------------------------------------------------------------------- #


def test_gui_command_start_delegates_to_controller_start():
    ctrl = FakeModelController()
    gc = _controller(model_ctrl=ctrl)
    gc.request_start("qwen3-14b")
    _dispatch_next(gc)
    assert ("start", "qwen3-14b") in ctrl.calls
    result = gc.result_q.get_nowait()
    assert result.kind == "start" and result.ok
    assert result.payload["port"] == 8080
    assert gc._active_port == 8080


def test_gui_command_start_uses_switch_when_a_model_already_runs():
    ctrl = FakeModelController()
    ctrl.running_model_id = "deepseek-r1-distill-14b"
    gc = _controller(model_ctrl=ctrl)
    gc.request_start("qwen3-14b")
    _dispatch_next(gc)
    assert ("switch", "qwen3-14b") in ctrl.calls
    assert not any(c[0] == "start" for c in ctrl.calls)


def test_gui_command_start_failure_marshals_remedy_not_a_crash():
    ctrl = FakeModelController(start_status=ServiceStatus.STOPPED_ERROR)
    gc = _controller(model_ctrl=ctrl)
    gc.request_start("qwen3-14b")
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert result.kind == "start" and not result.ok
    assert "STOPPED_ERROR" in (result.error or "")
    assert gc._active_port is None


def test_gui_command_stop_delegates_and_clears_active_port():
    ctrl = FakeModelController()
    ctrl.running_model_id = "qwen3-14b"
    gc = _controller(model_ctrl=ctrl)
    gc._active_port = 8080
    gc.request_stop()
    _dispatch_next(gc)
    assert ("stop",) in ctrl.calls
    assert gc._active_port is None
    assert gc.result_q.get_nowait().ok


def test_gui_command_whisper_toggle_starts_then_stops():
    whisper = FakeWhisper()
    gc = _controller(whisper=whisper)
    gc.request_whisper_toggle()
    _dispatch_next(gc)
    assert whisper.calls == ["start"]
    assert gc.result_q.get_nowait().payload["running"] is True
    gc.request_whisper_toggle()
    _dispatch_next(gc)
    assert whisper.calls == ["start", "stop"]
    assert gc.result_q.get_nowait().payload["running"] is False


def test_gui_command_validate_gpu_layers_table():
    ok_cases = ["-1", "0", "1", "32", "+8"]
    for text in ok_cases:
        ok, value, err = gui_controller.validate_gpu_layers(text)
        assert ok and err is None and value == int(text)
    bad_cases = ["-2", "1.5", "abc", "", "  ", "1e3", "0x10"]
    for text in bad_cases:
        ok, value, err = gui_controller.validate_gpu_layers(text)
        assert not ok and value is None and err


def test_gui_command_validate_context_size_table():
    for text in ["1", "4096", "32768"]:
        ok, value, err = gui_controller.validate_context_size(text)
        assert ok and value == int(text)
    for text in ["0", "-1", "3.5", "", "lots"]:
        ok, value, err = gui_controller.validate_context_size(text)
        assert not ok and err


def test_gui_command_save_edits_rejects_invalid_without_enqueue():
    gc = _controller()
    gc.save_model_edits("qwen3-14b", "not-int", "4096")
    # No command should have been enqueued; an error Result is pushed instead.
    assert gc.command_q.empty()
    result = gc.result_q.get_nowait()
    assert result.kind == "save_edits" and not result.ok and result.error


def test_gui_command_button_states_follow_running_and_inflight():
    idle = gui_controller.compute_button_states(UiState())
    assert idle["start_enabled"] and not idle["stop_enabled"] and not idle["chat_enabled"]
    running = gui_controller.compute_button_states(UiState(running_model_id="qwen3-14b", running_port=8080))
    assert running["stop_enabled"] and running["chat_enabled"]
    busy = gui_controller.compute_button_states(UiState(in_flight=True))
    assert not busy["start_enabled"] and not busy["stop_enabled"] and busy["working"]


# --------------------------------------------------------------------------- #
# AC11 - metrics/slots parsing degrades honestly (keyword: gui_metrics)
# --------------------------------------------------------------------------- #


_FULL_METRICS = """# HELP llamacpp:predicted_tokens_seconds gen speed
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 42.5
llamacpp:prompt_tokens_seconds 310.0
llamacpp:kv_cache_usage_ratio 0.25
llamacpp:kv_cache_tokens 4096
llamacpp:requests_processing 1
"""


def test_gui_metrics_parses_all_fields_present():
    sample = gui_controller.parse_metrics(_FULL_METRICS)
    assert sample.metrics_available
    assert sample.gen_tokens_s == 42.5
    assert sample.prompt_tokens_s == 310.0
    assert sample.kv_cache_usage_ratio == 0.25
    assert sample.kv_cache_tokens == 4096
    assert sample.requests_processing == 1


def test_gui_metrics_absent_field_is_none_never_zero():
    # Only generation speed present; every other field must be None (renders "-"),
    # never a fabricated 0.
    sample = gui_controller.parse_metrics("llamacpp:predicted_tokens_seconds 0\n")
    assert sample.gen_tokens_s == 0.0  # a real zero reading is preserved
    assert sample.prompt_tokens_s is None
    assert sample.kv_cache_usage_ratio is None
    assert sample.kv_cache_tokens is None
    assert sample.requests_processing is None


def test_gui_metrics_empty_body_flags_unavailable():
    for body in ("", "   ", "\n\n"):
        sample = gui_controller.parse_metrics(body)
        assert sample.metrics_available is False
        assert sample.gen_tokens_s is None


def test_gui_metrics_ignores_comments_and_unknown_lines():
    sample = gui_controller.parse_metrics(
        "# a comment\nsome:other_metric 99\nllamacpp:requests_processing 2\n"
    )
    assert sample.requests_processing == 2
    assert sample.gen_tokens_s is None


def test_gui_metrics_slots_list_and_disabled():
    present = gui_controller.parse_slots([{"n_ctx": 4096, "state": 1}])
    assert present.available and present.slots[0]["n_ctx"] == 4096
    disabled = gui_controller.parse_slots({"error": "slots disabled"})
    assert not disabled.available and disabled.slots == []


def test_gui_metrics_collect_sample_degrades_when_metrics_absent():
    # fetch returns None (endpoint absent): the sample is flagged unavailable and
    # the controller stops re-probing metrics (probe-once, G5).
    gc = _controller()
    gc._active_port = 8080
    sample = gc._collect_sample(8080)
    assert sample.metrics_available is False
    assert gc._metrics_available is False


# --------------------------------------------------------------------------- #
# AC14 - shutdown joins threads and calls stop_all once (keyword: gui_shutdown)
# --------------------------------------------------------------------------- #


def test_gui_shutdown_joins_threads_and_calls_stop_all_once():
    manager = FakeManager()
    gc = _controller(manager=manager)
    gc.start_threads()
    gc.shutdown()
    assert manager.stop_all_calls == 1
    assert gc._ops_thread is not None and not gc._ops_thread.is_alive()
    assert gc._monitor_thread is not None and not gc._monitor_thread.is_alive()


def test_gui_shutdown_is_idempotent_stop_all_still_once():
    manager = FakeManager()
    gc = _controller(manager=manager)
    gc.start_threads()
    gc.shutdown()
    gc.shutdown()  # a second close (or the atexit backstop) must not double-stop
    assert manager.stop_all_calls == 1


def test_gui_shutdown_without_threads_still_stops_services():
    # A shutdown before start_threads (e.g. GUI failed early) must still be safe.
    manager = FakeManager()
    gc = _controller(manager=manager)
    gc.shutdown()
    assert manager.stop_all_calls == 1


def test_gui_shutdown_during_in_flight_start_calls_stop_all_once():
    """M-1/H-1: closing the window WHILE a start command is running on the ops
    worker (not idle) must still join the wedged worker and call stop_all exactly
    once. The prior suite only exercised an idle shutdown, so H-1's in-flight orphan
    window went unproven. Here the model start blocks (standing in for the up-to-60s
    readiness wait) so the shutdown lands mid-flight."""
    entered = threading.Event()
    release = threading.Event()

    class _BlockingModelController(FakeModelController):
        def start(self, model_id, ctx=None, gpu=None):
            entered.set()  # the ops worker is now inside the in-flight start
            release.wait(5.0)  # hold the start open across the window close
            return super().start(model_id, ctx, gpu)

    manager = FakeManager()
    gc = _controller(model_ctrl=_BlockingModelController(), manager=manager)
    gc.start_threads()
    gc.request_start("qwen3-14b")
    assert entered.wait(5.0)  # confirm the command is genuinely in flight
    # Close the window from a helper thread; shutdown() joins the wedged ops worker.
    closer = threading.Thread(target=gc.shutdown)
    closer.start()
    # In production the 5s join times out and the daemon worker is killed at exit;
    # here we release the start so the join returns fast and the test is deterministic.
    release.set()
    closer.join(6.0)
    assert not closer.is_alive()
    assert manager.stop_all_calls == 1


# --------------------------------------------------------------------------- #
# Owner batch request 1 - Save-button state machine (keyword: gui_command)
# --------------------------------------------------------------------------- #


def test_gui_command_save_state_clean_when_editors_match_saved():
    # Editors equal the saved values -> Save DISABLED, no error (the greyed button
    # is the confirmation; there is no separate "saved" notice).
    state = gui_controller.compute_save_state("-1", "16384", saved_gpu=-1, saved_ctx=16384)
    assert state["enabled"] is False and state["error"] is None


def test_gui_command_save_state_dirty_when_a_field_differs():
    gpu_changed = gui_controller.compute_save_state("40", "16384", saved_gpu=-1, saved_ctx=16384)
    assert gpu_changed["enabled"] is True and gpu_changed["error"] is None
    ctx_changed = gui_controller.compute_save_state("-1", "8192", saved_gpu=-1, saved_ctx=16384)
    assert ctx_changed["enabled"] is True and ctx_changed["error"] is None


def test_gui_command_save_state_invalid_keeps_save_enabled_with_error():
    # Owner's rule: invalid input keeps Save ENABLED and surfaces the validation
    # error (invalid is a dirty state the owner can see and correct); a click is
    # still rejected by save_model_edits so nothing bad is written.
    bad_gpu = gui_controller.compute_save_state("-5", "16384", saved_gpu=-1, saved_ctx=16384)
    assert bad_gpu["enabled"] is True and bad_gpu["error"]
    bad_ctx = gui_controller.compute_save_state("-1", "0", saved_gpu=-1, saved_ctx=16384)
    assert bad_ctx["enabled"] is True and bad_ctx["error"]


def test_gui_command_save_state_greys_out_after_save_roundtrip():
    # Simulate the post-save recompute: after write_model_fields the in-memory saved
    # values become the edited ones, so the same editor text is now clean -> Save
    # greys out (the confirmation).
    before = gui_controller.compute_save_state("40", "16384", saved_gpu=-1, saved_ctx=16384)
    assert before["enabled"] is True
    after = gui_controller.compute_save_state("40", "16384", saved_gpu=40, saved_ctx=16384)
    assert after["enabled"] is False and after["error"] is None


# --------------------------------------------------------------------------- #
# Owner batch request 2 - Size (GB) column (keyword: gui_command)
# --------------------------------------------------------------------------- #


def test_gui_command_format_size_gb_decimal_and_missing():
    # 10,263,894,400 bytes is the real Qwen3-14B .gguf; decimal GB -> "10.3 GB"
    # (the owner's reference value). None (unset/missing location) -> "-".
    assert gui_controller.format_size_gb(10_263_894_400) == "10.3 GB"
    assert gui_controller.format_size_gb(None) == "-"
    assert gui_controller.format_size_gb(0) == "0.0 GB"


def test_gui_command_total_size_display_sums_known_sizes():
    rows = [
        {"size_bytes": 10_000_000_000},
        {"size_bytes": 5_000_000_000},
    ]
    assert gui_controller.total_size_display(rows) == "Total: 15.0 GB"


def test_gui_command_total_size_display_notes_unknown_rows():
    rows = [
        {"size_bytes": 10_000_000_000},
        {"size_bytes": None},
    ]
    assert gui_controller.total_size_display(rows) == "Total: 10.0 GB (1 unknown)"


def test_gui_command_total_size_display_all_unknown_is_dash():
    rows = [{"size_bytes": None}, {"size_bytes": None}]
    assert gui_controller.total_size_display(rows) == "Total: - (2 unknown)"


def test_gui_command_total_size_display_empty_registry():
    assert gui_controller.total_size_display([]) == "Total: -"


# --------------------------------------------------------------------------- #
# Owner request - replace the repeating id column with VRAM + spillage, and
# show host RAM/GPU specs
# --------------------------------------------------------------------------- #


def test_gui_command_format_gpu_portion_no_need_is_dash():
    assert gui_controller.format_gpu_portion_display(0, 16303.0) == "-"


def test_gui_command_format_gpu_portion_no_gpu_is_dash():
    # "How much fits on the GPU" is unanswerable without a detected card.
    assert gui_controller.format_gpu_portion_display(15000, None) == "-"


def test_gui_command_format_gpu_portion_fits_shows_the_full_need():
    assert gui_controller.format_gpu_portion_display(10000, 16303.0) == "10.0 GB"


def test_gui_command_format_gpu_portion_over_capacity_caps_at_the_card():
    # A 17100 MB need against a 16303 MiB card: the GPU can only ever hold its
    # own capacity, never more, regardless of what the model needs.
    assert gui_controller.format_gpu_portion_display(17100, 16303.0) == "16.3 GB"


def test_gui_command_format_cpu_portion_fits_is_dash():
    assert gui_controller.format_cpu_portion_display(10000, 16303.0) == "-"


def test_gui_command_format_cpu_portion_no_need_or_no_gpu_is_dash():
    assert gui_controller.format_cpu_portion_display(0, 16303.0) == "-"
    assert gui_controller.format_cpu_portion_display(15000, None) == "-"


def test_gui_command_format_cpu_portion_over_capacity_shows_the_spill():
    # 17100 MB need vs a 16303 MiB card: 797 MB (~0.8 GB) spills to CPU/RAM.
    assert gui_controller.format_cpu_portion_display(17100, 16303.0) == "0.8 GB"


def test_gui_command_format_cpu_portion_tiny_real_overage_still_shows():
    # 2026-08-16 owner fix: a real 17MB overage must never be hidden as "-" --
    # it rounds UP to the smallest displayable unit (0.1 GB), never down to 0.
    assert gui_controller.format_cpu_portion_display(16320, 16303.0) == "0.1 GB"


def test_gui_command_format_cpu_portion_exact_fit_is_dash():
    assert gui_controller.format_cpu_portion_display(16303, 16303.0) == "-"


class FakeGpuProvider:
    def __init__(self, gpus):
        self._gpus = gpus

    def gpus(self):
        return self._gpus


class FakeSysProvider:
    def __init__(self, ram_total_mb):
        self._ram = ram_total_mb

    def ram_total_mb(self):
        return self._ram


def test_gui_command_system_specs_detected_and_cached():
    from health import GpuInfo

    gc = GuiController(
        Settings(), FakeRegistry([_model()]), FakeModelController(), FakeWhisper(),
        FakeManager(), fetch=lambda port, path: None,
        gpu_provider=FakeGpuProvider([GpuInfo(name="Fake GPU", vram_total_mb=16303.0, vram_free_mb=16000.0)]),
        sys_provider=FakeSysProvider(32768.0),
    )
    specs = gc.system_specs()
    assert specs.gpu_name == "Fake GPU"
    assert specs.gpu_vram_total_mb == 16303.0
    assert specs.ram_total_mb == 32768.0
    # Cached: a second call returns the identical object without re-querying.
    assert gc.system_specs() is specs


def test_gui_command_system_specs_no_providers_is_honest_unknown():
    gc = _controller()
    specs = gc.system_specs()
    assert specs.ram_total_mb is None
    assert specs.gpu_name is None
    assert specs.gpu_vram_total_mb is None


def test_gui_command_format_system_specs_renders_ram_and_gpu():
    specs = gui_controller.SystemSpecs(
        ram_total_mb=32768.0, gpu_name="NVIDIA GeForce RTX 5070 Ti", gpu_vram_total_mb=16303.0
    )
    text = gui_controller.format_system_specs(specs)
    assert "32.0 GB RAM" in text
    assert "NVIDIA GeForce RTX 5070 Ti" in text
    assert "15.9 GB VRAM" in text


def test_gui_command_format_system_specs_degrades_honestly():
    text = gui_controller.format_system_specs(gui_controller.SystemSpecs())
    assert "RAM: -" in text
    assert "GPU: not detected" in text


def test_gui_command_list_models_includes_gpu_and_cpu_portion_display():
    from health import GpuInfo

    gc = GuiController(
        Settings(), FakeRegistry([_model()]), FakeModelController(), FakeWhisper(),
        FakeManager(), fetch=lambda port, path: None,
        gpu_provider=FakeGpuProvider([GpuInfo(name="Fake GPU", vram_total_mb=16303.0, vram_free_mb=16000.0)]),
        sys_provider=FakeSysProvider(32768.0),
    )
    rows = gc.list_models()
    assert rows[0]["vram_estimate_mb"] == 0  # _model() default has no vram estimate
    assert rows[0]["vram_need_mb"] == 0  # and no on-disk file (location doesn't exist)
    assert rows[0]["gpu_portion_display"] == "-"
    assert rows[0]["cpu_portion_display"] == "-"


def test_gui_command_compute_vram_need_mb_prefers_the_real_file_size():
    # A partial-offload-tuned estimate (16000) below a raw file that exceeds it
    # (17100): the raw file size wins, since that is the true full-GPU need.
    assert gui_controller.compute_vram_need_mb(16000, 17_100_000_000) == 17100.0
    # A vision model with a downloaded file: the real file size (6700) wins
    # even though the estimate (8500, includes the separate mmproj file's extra
    # VRAM) is larger -- "On GPU" must never exceed the Size column's own
    # number once Size has a real number to disagree with (2026-08-16 fix).
    assert gui_controller.compute_vram_need_mb(8500, 6_700_000_000) == 6700.0
    # No file on disk yet -> the estimate is used as a before-download preview.
    assert gui_controller.compute_vram_need_mb(12000, None) == 12000
    # Neither signal present -> 0 (renders "-").
    assert gui_controller.compute_vram_need_mb(0, None) == 0


def test_gui_command_sort_by_vram_is_numeric_missing_last():
    rows = [
        {"vram_need_mb": 0},
        {"vram_need_mb": 15000},
        {"vram_need_mb": 10000},
    ]
    ordered = gui_controller.sort_model_rows(rows, "vram", descending=False)
    assert [r["vram_need_mb"] for r in ordered] == [10000, 15000, 0]


# --------------------------------------------------------------------------- #
# Owner request - rename/re-id a model (id + name editors, Settings page)
# --------------------------------------------------------------------------- #


def test_gui_command_validate_model_id_rejects_blank_and_duplicate():
    ok, value, err = gui_controller.validate_model_id("qwen3-14b-v2", ["qwen3-14b", "other"], "qwen3-14b")
    assert ok and value == "qwen3-14b-v2" and err is None
    ok, value, err = gui_controller.validate_model_id("  ", ["qwen3-14b"], "qwen3-14b")
    assert not ok and err
    ok, value, err = gui_controller.validate_model_id("other", ["qwen3-14b", "other"], "qwen3-14b")
    assert not ok and "already used" in err
    # Renaming a model's id back to its own current id is not a collision.
    ok, value, err = gui_controller.validate_model_id("qwen3-14b", ["qwen3-14b", "other"], "qwen3-14b")
    assert ok and value == "qwen3-14b"


def test_gui_command_validate_model_name_rejects_blank():
    ok, value, err = gui_controller.validate_model_name("Qwen3 14B v2")
    assert ok and value == "Qwen3 14B v2" and err is None
    ok, value, err = gui_controller.validate_model_name("   ")
    assert not ok and err


def test_gui_command_identity_save_state_clean_dirty_invalid():
    clean = gui_controller.compute_identity_save_state(
        "qwen3-14b", "Qwen3 14B", "qwen3-14b", "Qwen3 14B", ["qwen3-14b"]
    )
    assert clean["enabled"] is False and clean["error"] is None
    dirty = gui_controller.compute_identity_save_state(
        "qwen3-14b-v2", "Qwen3 14B", "qwen3-14b", "Qwen3 14B", ["qwen3-14b"]
    )
    assert dirty["enabled"] is True and dirty["error"] is None
    invalid = gui_controller.compute_identity_save_state(
        "", "Qwen3 14B", "qwen3-14b", "Qwen3 14B", ["qwen3-14b"]
    )
    assert invalid["enabled"] is True and invalid["error"]


def test_gui_command_save_identity_rejects_invalid_without_enqueue():
    gc = _controller()
    gc.save_model_identity("qwen3-14b", "  ", "Qwen3 14B")
    assert gc.command_q.empty()
    result = gc.result_q.get_nowait()
    assert result.kind == "save_identity" and not result.ok and result.error


def test_gui_command_save_identity_rejects_duplicate_id_without_enqueue():
    registry = FakeRegistry([_model(model_id="qwen3-14b"), _model(model_id="other")])
    gc = GuiController(
        Settings(), registry, FakeModelController(), FakeWhisper(), FakeManager(),
        fetch=lambda port, path: None,
    )
    gc.save_model_identity("qwen3-14b", "other", "Qwen3 14B")
    assert gc.command_q.empty()
    result = gc.result_q.get_nowait()
    assert result.kind == "save_identity" and not result.ok and "already used" in result.error


def test_gui_command_find_id_change_blockers_draft_model_reference():
    blockers = gui_controller.find_id_change_blockers(
        "qwen3-14b", [("some-other-model", "qwen3-14b")], default_model="qwen3-8-27b"
    )
    assert len(blockers) == 1 and "some-other-model" in blockers[0]


def test_gui_command_find_id_change_blockers_default_model_reference():
    blockers = gui_controller.find_id_change_blockers(
        "qwen3-14b", [], default_model="qwen3-14b"
    )
    assert len(blockers) == 1 and "launcher.default_model" in blockers[0]


def test_gui_command_find_id_change_blockers_none_when_unreferenced():
    blockers = gui_controller.find_id_change_blockers(
        "qwen3-14b", [("other-model", "some-other-id")], default_model="other-model"
    )
    assert blockers == []


def test_gui_command_save_identity_rejects_id_change_that_orphans_draft_model():
    # 2026-08-16 incident: renaming a model referenced as another model's
    # draft_model (or settings.yaml's launcher.default_model) must be rejected,
    # not silently written and left to break at the next launch.
    settings = Settings()
    settings.launcher.default_model = "other-model"
    draft_referrer = Model(
        id="some-other-model", name="Other Model", description="", location="x.gguf",
        context_size=8192, gpu_layers=-1, draft_model="qwen3-14b",
    )
    gc = GuiController(
        settings, FakeRegistry([_model(), draft_referrer]), FakeModelController(),
        FakeWhisper(), FakeManager(), fetch=lambda port, path: None,
    )
    gc.save_model_identity("qwen3-14b", "qwen3-14b-v2", "Qwen3 14B")
    assert gc.command_q.empty()
    result = gc.result_q.get_nowait()
    assert result.kind == "save_identity" and not result.ok
    assert "some-other-model" in result.error


def test_gui_command_save_identity_rejects_id_change_that_orphans_default_model():
    settings = Settings()
    settings.launcher.default_model = "qwen3-14b"
    gc = GuiController(
        settings, FakeRegistry([_model()]), FakeModelController(), FakeWhisper(),
        FakeManager(), fetch=lambda port, path: None,
    )
    gc.save_model_identity("qwen3-14b", "qwen3-14b-v2", "Qwen3 14B")
    assert gc.command_q.empty()
    result = gc.result_q.get_nowait()
    assert result.kind == "save_identity" and not result.ok
    assert "launcher.default_model" in result.error


def test_gui_command_save_identity_rejects_id_change_while_running():
    ctrl = FakeModelController()
    ctrl.running_model_id = "qwen3-14b"
    gc = _controller(model_ctrl=ctrl)
    gc.save_model_identity("qwen3-14b", "qwen3-14b-v2", "Qwen3 14B")
    assert gc.command_q.empty()
    result = gc.result_q.get_nowait()
    assert result.kind == "save_identity" and not result.ok
    assert "stop" in result.error


def test_gui_command_save_identity_allows_name_only_change_while_running():
    # A pure rename (id unchanged) never touches the running-process link, so it
    # is allowed even while the model is running.
    ctrl = FakeModelController()
    ctrl.running_model_id = "qwen3-14b"
    gc = _controller(model_ctrl=ctrl)
    gc.save_model_identity("qwen3-14b", "qwen3-14b", "Qwen3 14B Renamed")
    assert not gc.command_q.empty()
    command = gc.command_q.get_nowait()
    assert command.kind == "save_identity"
    assert command.payload == {
        "model_id": "qwen3-14b",
        "new_id": "qwen3-14b",
        "new_name": "Qwen3 14B Renamed",
    }


def test_gui_command_list_models_reads_and_caches_gguf_size(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x" * 2_500_000_000)  # 2.5 GB decimal -> "2.5 GB"
    present = _model(location=str(gguf))
    missing = _model(model_id="no-file", location="/locitize-test/does/not/exist.gguf")
    unset = _model(model_id="unset", location="")
    gc = GuiController(
        Settings(),
        FakeRegistry([present, missing, unset]),
        FakeModelController(),
        FakeWhisper(),
        FakeManager(),
        fetch=lambda port, path: None,
    )
    rows = {r["id"]: r for r in gc.list_models()}
    assert rows["qwen3-14b"]["size_display"] == "2.5 GB"
    assert rows["qwen3-14b"]["size_bytes"] == 2_500_000_000
    assert rows["no-file"]["size_display"] == "-" and rows["no-file"]["size_bytes"] is None
    assert rows["unset"]["size_display"] == "-" and rows["unset"]["size_bytes"] is None
    # Cached per session: deleting the file does not change the cached size.
    gguf.unlink()
    assert gc.list_models()[0]["size_bytes"] == 2_500_000_000


def test_gui_command_list_models_keeps_quality_separate_from_generation_tok_s():
    # benchmark_score remains the quality percentage. The visible benchmark cell
    # comes from measured generation throughput and carries an explicit unit.
    scored = _model(model_id="scored", benchmark_score=73.9)
    unscored = _model(model_id="unscored", benchmark_score=None)
    gc = GuiController(
        Settings(),
        FakeRegistry([scored, unscored]),
        FakeModelController(),
        FakeWhisper(),
        FakeManager(),
        fetch=lambda port, path: None,
        benchmark_history_fn=lambda rows: {"scored": 42.5},
    )
    rows = {r["id"]: r for r in gc.list_models()}
    assert rows["scored"]["benchmark_score"] == 73.9
    assert rows["scored"]["benchmark_tok_s"] == 42.5
    assert rows["scored"]["score_display"] == "42.5 tok/s"
    assert rows["unscored"]["benchmark_score"] is None
    assert rows["unscored"]["benchmark_tok_s"] is None
    assert rows["unscored"]["score_display"] == "-"


# --------------------------------------------------------------------------- #
# Owner batch request 3 - sortable columns (keyword: gui_command)
# --------------------------------------------------------------------------- #


def _rows_for_sort():
    return [
        {"id": "b", "name": "Beta", "status": "STOPPED", "size_bytes": 3_000_000_000},
        {"id": "a", "name": "Alpha", "status": "RUNNING", "size_bytes": 1_000_000_000},
        {"id": "c", "name": "Gamma", "status": "STOPPED", "size_bytes": None},
        {"id": "d", "name": "delta", "status": "STOPPED", "size_bytes": 2_000_000_000},
    ]


def test_gui_command_sort_by_name_ascending_then_descending():
    rows = _rows_for_sort()
    asc = [r["name"] for r in gui_controller.sort_model_rows(rows, "name", False)]
    assert asc == ["Alpha", "Beta", "delta", "Gamma"]  # case-insensitive
    desc = [r["name"] for r in gui_controller.sort_model_rows(rows, "name", True)]
    assert desc == ["Gamma", "delta", "Beta", "Alpha"]


def test_gui_command_sort_by_size_is_numeric_missing_last():
    rows = _rows_for_sort()
    asc = gui_controller.sort_model_rows(rows, "size", False)
    assert [r["id"] for r in asc] == ["a", "d", "b", "c"]  # 1,2,3 GB then missing
    desc = gui_controller.sort_model_rows(rows, "size", True)
    assert [r["id"] for r in desc] == ["b", "d", "a", "c"]  # missing STILL last


def test_gui_command_sort_by_benchmark_tok_s_is_numeric_missing_last():
    rows = [
        {"id": "slow", "benchmark_tok_s": 8.4},
        {"id": "none", "benchmark_tok_s": None},
        {"id": "fast", "benchmark_tok_s": 120.7},
    ]
    asc = gui_controller.sort_model_rows(rows, "benchmark_tok_s", False)
    desc = gui_controller.sort_model_rows(rows, "benchmark_tok_s", True)
    assert [r["id"] for r in asc] == ["slow", "fast", "none"]
    assert [r["id"] for r in desc] == ["fast", "slow", "none"]


def test_gui_command_sort_is_stable_for_equal_keys():
    # Two STOPPED rows keep their input order (stability lets the GUI preserve the
    # selected row across a resort).
    rows = [
        {"id": "x", "name": "n", "status": "STOPPED", "size_bytes": 1},
        {"id": "y", "name": "n", "status": "STOPPED", "size_bytes": 1},
    ]
    out = gui_controller.sort_model_rows(rows, "status", False)
    assert [r["id"] for r in out] == ["x", "y"]


# --------------------------------------------------------------------------- #
# Owner batch request 4 - live monitor counts (keyword: gui_metrics)
# --------------------------------------------------------------------------- #

# Real field names from the owner's llama-server build (probed 2026-07-18): counts
# come from the cumulative counters, and this build exposes NEITHER
# kv_cache_usage_ratio NOR kv_cache_tokens, so n_tokens_max is the ctx-used source.
_REAL_BUILD_METRICS = """# HELP llamacpp:prompt_tokens_total prompt tokens
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 412
llamacpp:tokens_predicted_total 583
llamacpp:predicted_tokens_seconds 69.8
llamacpp:prompt_tokens_seconds 22.6
llamacpp:n_tokens_max 995
llamacpp:requests_processing 0
"""


def test_gui_metrics_parses_real_build_counters():
    sample = gui_controller.parse_metrics(_REAL_BUILD_METRICS)
    assert sample.prompt_tokens_total == 412
    assert sample.gen_tokens_total == 583
    assert sample.n_tokens_max == 995
    assert sample.kv_cache_usage_ratio is None  # absent in this build
    assert sample.kv_cache_tokens is None


def test_gui_metrics_monitor_line_matches_owner_example():
    # The exact line the owner asked for, reproduced from the real-build payload.
    sample = gui_controller.parse_metrics(_REAL_BUILD_METRICS)
    line = gui_controller.format_monitor_line(sample, context_size=16384)
    assert line == (
        "gen 69.8 tok/s | prompt 22.6 tok/s | prompt 412 tok | gen 583 tok | "
        "ctx 995/16384 (6%)"
    )


def test_gui_metrics_monitor_line_prefers_kv_tokens_and_slots_nctx():
    # A richer build with kv_cache_tokens + /slots n_ctx: used=kv tokens, total=n_ctx.
    sample = gui_controller.MetricsSample(
        gen_tokens_s=10.0,
        prompt_tokens_s=5.0,
        prompt_tokens_total=100,
        gen_tokens_total=200,
        kv_cache_tokens=300,
        n_ctx=4096,
    )
    line = gui_controller.format_monitor_line(sample, context_size=999)
    assert "ctx 300/4096 (7%)" in line  # n_ctx wins over the passed context_size


def test_gui_metrics_monitor_line_degrades_when_no_count_sources():
    # Only rates present, no counters, no ctx sources -> counts render "-" and ctx
    # is honest ("-"), never a fabricated number. requests_processing is None (no
    # active request), so the zero rate reads "idle" (AC13).
    sample = gui_controller.MetricsSample(gen_tokens_s=0.0, prompt_tokens_s=None)
    line = gui_controller.format_monitor_line(sample, context_size=None)
    assert line == "gen idle tok/s | prompt idle tok/s | prompt - tok | gen - tok | ctx -"


def test_gui_metrics_monitor_line_idle_only_when_not_processing():
    # AC13 (Architecture M5.12): "idle" is decided from requests_processing == 0, not
    # from the rate gauge reading 0. With no request in flight and a zero rate, the
    # line reads "idle".
    sample = gui_controller.MetricsSample(
        gen_tokens_s=0.0, prompt_tokens_s=0.0, requests_processing=0
    )
    line = gui_controller.format_monitor_line(sample, context_size=None)
    assert line.startswith("gen idle tok/s | prompt idle tok/s")


def test_gui_metrics_monitor_line_active_low_rate_is_not_idle():
    # AC13: a genuinely active request whose rate gauge momentarily reads 0 must NOT
    # be mislabeled "idle"; it reads "..." (working) instead.
    sample = gui_controller.MetricsSample(
        gen_tokens_s=0.0, prompt_tokens_s=0.0, requests_processing=1
    )
    line = gui_controller.format_monitor_line(sample, context_size=None)
    assert "gen ... tok/s" in line
    assert "idle" not in line


def test_gui_metrics_collect_sample_merges_slots_nctx(monkeypatch):
    # /slots supplies n_ctx (and, on richer builds, a token count) that the metrics
    # payload lacks; _collect_sample folds it into the sample.
    gc = _controller()
    gc._active_port = 8080

    def fake_fetch(port, path):
        if path == "/metrics":
            return _REAL_BUILD_METRICS
        if path == "/slots":
            return '[{"id": 0, "is_processing": false, "n_ctx": 8192}]'
        return None

    gc._fetch = fake_fetch
    sample = gc._collect_sample(8080)
    assert sample.n_ctx == 8192
    assert sample.prompt_tokens_total == 412


def test_gui_metrics_slots_extracts_nctx_and_tokens_used():
    rich = gui_controller.parse_slots(
        [{"id": 0, "n_ctx": 4096, "n_past": 512, "is_processing": True}]
    )
    assert rich.n_ctx == 4096 and rich.tokens_used == 512
    stripped = gui_controller.parse_slots(
        [{"id": 0, "n_ctx": 16384, "is_processing": False, "speculative": False}]
    )
    assert stripped.n_ctx == 16384 and stripped.tokens_used is None


# --------------------------------------------------------------------------- #
# Owner batch cosmetics - O-2 honest running state (keyword: gui_command)
# --------------------------------------------------------------------------- #


def test_gui_command_start_result_carries_authoritative_running_state():
    ctrl = FakeModelController()
    gc = _controller(model_ctrl=ctrl)
    gc.request_start("qwen3-14b")
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    # The Result stamps the controller's real running state so the GUI renders truth.
    assert result.payload["running_model_id"] == "qwen3-14b"
    assert result.payload["running_port"] == 8080


def test_gui_command_failed_switch_reports_no_stale_running_model():
    # A switch that fails to reach RUNNING must NOT leave the old model shown as
    # running: the snapshot (nothing RUNNING) is what the Result reports (O-2).
    class _FailingSwitch(FakeModelController):
        def switch(self, model_id, ctx=None, gpu=None):
            self.calls.append(("switch", model_id))
            self.running_model_id = None  # old model stopped, new one failed to start
            return ServiceStatus.STOPPED_ERROR

    ctrl = _FailingSwitch()
    ctrl.running_model_id = "deepseek-r1-distill-14b"
    gc = _controller(model_ctrl=ctrl)
    gc.request_start("qwen3-14b")
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert not result.ok
    assert result.payload["running_model_id"] is None  # no stale RUNNING chip
    assert result.payload["running_port"] is None


def test_gui_command_stop_result_reports_nothing_running():
    ctrl = FakeModelController()
    ctrl.running_model_id = "qwen3-14b"
    gc = _controller(model_ctrl=ctrl)
    gc._active_port = 8080
    gc.request_stop()
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert result.ok
    assert result.payload["running_model_id"] is None


def test_gui_shutdown_during_listen_calls_stop_all_once():
    """H-2/M-1: closing the window during a Listen -- the ops worker is inside
    listen_fn for the whole capture window and is not reading command_q -- must
    still join and call stop_all exactly once. With whisper-stream now registered on
    the shared, atexit-backstopped manager (launcher._gui_listen), that stop_all is
    the guarantee it cannot be orphaned."""
    entered = threading.Event()
    release = threading.Event()

    def blocking_listen(seconds, emit):
        entered.set()  # the ops worker is now inside the listen window
        release.wait(5.0)
        return True

    manager = FakeManager()
    gc = GuiController(
        Settings(),
        FakeRegistry([_model()]),
        FakeModelController(),
        FakeWhisper(),
        manager,
        fetch=lambda port, path: None,
        listen_fn=blocking_listen,
    )
    gc.start_threads()
    gc.request_listen(15.0)
    assert entered.wait(5.0)  # confirm the listen is genuinely in flight
    closer = threading.Thread(target=gc.shutdown)
    closer.start()
    release.set()
    closer.join(6.0)
    assert not closer.is_alive()
    assert manager.stop_all_calls == 1


# --------------------------------------------------------------------------- #
# M6 - Voice OUT (Kokoro TTS) command path (keyword: gui_command)
# --------------------------------------------------------------------------- #


def _voice_settings(tmp_path) -> Settings:
    """A Settings whose kokoro_voices dir holds a couple of fake .pt voices."""
    settings = Settings()
    (tmp_path / "am_michael.pt").write_bytes(b"x")
    (tmp_path / "af_bella.pt").write_bytes(b"x")
    settings.paths.kokoro_voices = str(tmp_path)
    return settings


def test_gui_command_speak_routes_through_injected_speak_fn(tmp_path):
    """request_speak dispatches to the injected speak_fn on the ops worker and
    marshals a success Result -- gui_controller itself does no TTS/HTTP/audio."""
    calls: list[tuple[str, str]] = []

    def fake_speak(text: str, voice: str):
        calls.append((text, voice))
        return True, "spoken"

    gc = GuiController(
        _voice_settings(tmp_path),
        FakeRegistry([_model()]),
        FakeModelController(),
        FakeWhisper(),
        FakeManager(),
        fetch=lambda port, path: None,
        speak_fn=fake_speak,
    )
    gc.request_speak("hello there", "am_michael")
    gc._dispatch(gc.command_q.get_nowait())
    assert calls == [("hello there", "am_michael")]
    result = gc.result_q.get_nowait()
    assert result.kind == "speak" and result.ok and result.payload["voice"] == "am_michael"


def test_gui_command_speak_empty_text_is_honest_error(tmp_path):
    """Empty text never reaches the engine; the worker returns an honest error."""
    gc = GuiController(
        _voice_settings(tmp_path),
        FakeRegistry([_model()]),
        FakeModelController(),
        FakeWhisper(),
        FakeManager(),
        fetch=lambda port, path: None,
        speak_fn=lambda text, voice: (True, "spoken"),
    )
    gc.request_speak("   ", "am_michael")
    gc._dispatch(gc.command_q.get_nowait())
    result = gc.result_q.get_nowait()
    assert result.kind == "speak" and not result.ok and result.error


def test_gui_command_speak_missing_engine_is_honest_error(tmp_path):
    """No speak_fn wired (TTS unavailable) -> honest error, never a dead button."""
    gc = GuiController(
        _voice_settings(tmp_path),
        FakeRegistry([_model()]),
        FakeModelController(),
        FakeWhisper(),
        FakeManager(),
        fetch=lambda port, path: None,
    )
    gc.request_speak("hi", "am_michael")
    gc._dispatch(gc.command_q.get_nowait())
    result = gc.result_q.get_nowait()
    assert result.kind == "speak" and not result.ok


def test_gui_command_audition_walks_every_on_disk_voice(tmp_path):
    """Audition speaks each on-disk voice once, emitting a per-voice progress Result
    then a final success Result carrying the count."""
    spoken: list[str] = []

    def fake_speak(text: str, voice: str):
        spoken.append(voice)
        return True, "spoken"

    settings = _voice_settings(tmp_path)
    gc = GuiController(
        settings,
        FakeRegistry([_model()]),
        FakeModelController(),
        FakeWhisper(),
        FakeManager(),
        fetch=lambda port, path: None,
        speak_fn=fake_speak,
    )
    assert gc.available_voices() == ["af_bella", "am_michael"]
    gc.request_audition()
    gc._dispatch(gc.command_q.get_nowait())
    assert spoken == ["af_bella", "am_michael"]  # every on-disk voice, in order
    kinds = []
    while True:
        try:
            kinds.append(gc.result_q.get_nowait())
        except Exception:
            break
    # Two per-voice progress Results, then one final ok audition Result.
    assert [r.kind for r in kinds] == ["audition_voice", "audition_voice", "audition"]
    assert kinds[-1].ok and kinds[-1].payload["count"] == 2


# --------------------------------------------------------------------------- #
# delete_model - Models page Delete button (owner request 2026-08-21).
# UI-thread refusals mirror save_model_identity/save_model_capabilities'
# existing coverage; the worker's real file-deletion behavior (including the
# shared-file-not-deleted rule) gets its own real-filesystem integration tests
# below, since a silent unlink() bug is a much worse failure mode than a
# malformed yaml edit.
# --------------------------------------------------------------------------- #


def test_gui_command_delete_model_refuses_a_discovered_row():
    gc = _controller(model_ctrl=FakeModelController())
    gc._registry = FakeRegistry([_model()])
    gc._is_discovered = lambda model_id: True  # noqa: ARG005 - test stub
    gc.delete_model("qwen3-14b")
    assert gc.command_q.empty()
    result = gc.result_q.get_nowait()
    assert result.kind == "delete_model" and not result.ok
    assert "discovered" in result.error.lower() or "register" in result.error.lower()


def test_gui_command_delete_model_refuses_the_running_model():
    model_ctrl = FakeModelController()
    model_ctrl.running_model_id = "qwen3-14b"
    gc = _controller(model_ctrl=model_ctrl)
    gc.delete_model("qwen3-14b")
    assert gc.command_q.empty()
    result = gc.result_q.get_nowait()
    assert result.kind == "delete_model" and not result.ok
    assert "stop" in result.error.lower()


def test_gui_command_delete_model_enqueues_when_stopped_and_registered():
    gc = _controller()
    gc.delete_model("qwen3-14b")
    assert gc.result_q.empty()
    command = gc.command_q.get_nowait()
    assert command.kind == "delete_model"
    assert command.payload == {"model_id": "qwen3-14b"}


def test_gui_command_autotune_refuses_a_discovered_row():
    """A discovered fine-tune has no models.yaml block to write the result into."""
    gc = _controller(model_ctrl=FakeModelController())
    gc._registry = FakeRegistry([_model()])
    gc._is_discovered = lambda model_id: True  # noqa: ARG005 - test stub
    gc.request_autotune_context("qwen3-14b")
    assert gc.command_q.empty()
    result = gc.result_q.get_nowait()
    assert result.kind == "autotune" and not result.ok
    assert "discovered" in result.error.lower() or "register" in result.error.lower()


def test_gui_command_autotune_refuses_while_a_model_is_running():
    """The tuner's own trial starts would collide with it on the reserved port."""
    model_ctrl = FakeModelController()
    model_ctrl.running_model_id = "qwen3-14b"
    gc = _controller(model_ctrl=model_ctrl)
    gc.request_autotune_context("qwen3-14b")
    assert gc.command_q.empty()
    result = gc.result_q.get_nowait()
    assert result.kind == "autotune" and not result.ok
    assert "stop" in result.error.lower()


def test_gui_command_autotune_enqueues_when_stopped_and_registered():
    """Opt-in only: nothing is queued until this request method is called."""
    gc = _controller()
    gc.request_autotune_context("qwen3-14b")
    assert gc.result_q.empty()
    command = gc.command_q.get_nowait()
    assert command.kind == "autotune"
    assert command.payload == {"model_id": "qwen3-14b"}


def test_autotune_worker_refuses_a_model_with_no_file_location(tmp_path):
    """Without a location there is no GGUF header to read, so it refuses early."""
    gc, _registry = _real_gui_controller(
        tmp_path,
        """version: 1
models:
  - id: model-a
    name: "Model A"
    location: ""
    context_size: 8192
    gpu_layers: -1
    status: installed
""",
    )
    gc._do_autotune_context("model-a")
    result = gc.result_q.get_nowait()
    assert result.kind == "autotune" and not result.ok
    assert "location" in result.error.lower()


def test_autotune_worker_publishes_progress_then_a_final_result(tmp_path):
    """A real (synthetic) GGUF plus a stubbed trial exercises the whole handler.

    The trial function is stubbed because the real one starts llama-server for
    25 seconds; everything else here - the GGUF read, the models.yaml writes, the
    registry reload, the progress publishing - runs for real against tmp_path.
    """
    import struct

    import autotune as autotune_module

    def gstr(text):
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    kvs = (
        gstr("general.architecture") + struct.pack("<I", 8) + gstr("qwen3")
        + gstr("qwen3.context_length") + struct.pack("<I", 4) + struct.pack("<I", 32768)
    )
    weights = tmp_path / "model-a.gguf"
    weights.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2)
        + kvs + b"\x00" * 32
    )

    import json as _json

    gc, registry = _real_gui_controller(
        tmp_path,
        f"""version: 1
models:
  - id: model-a
    name: "Model A"
    location: {_json.dumps(str(weights))}
    context_size: 8192
    gpu_layers: -1
    status: installed
    server_args: ["--parallel", "1"]
""",
    )

    # Every context loads, so the search stops at the 4x policy cap.
    def always_ok(model_id, context_size, **kwargs):
        return autotune_module.Trial(context_size, True)

    original = autotune_module.run_smoke_trial
    autotune_module.run_smoke_trial = always_ok
    try:
        gc._do_autotune_context("model-a")
    finally:
        autotune_module.run_smoke_trial = original

    kinds = []
    while not gc.result_q.empty():
        kinds.append(gc.result_q.get_nowait())
    progress = [r for r in kinds if r.kind == "autotune_progress"]
    final = [r for r in kinds if r.kind == "autotune"]
    assert progress, "the UI would look frozen without progress lines"
    assert len(final) == 1 and final[0].ok
    assert final[0].payload["native_context"] == 32768
    assert final[0].payload["chosen_context"] == 32768 * 4
    assert final[0].payload["yarn_applied"] is True
    assert final[0].payload["rope_scale"] == 4
    # The in-memory registry now matches what was really written to disk.
    assert registry.get("model-a").context_size == 32768 * 4


def test_autotune_then_benchmarks_at_the_tuned_context(tmp_path):
    """Auto-tune and benchmark are one action: the speed is measured AFTER the
    new context is written, so it is the speed at the setting the model will use."""
    import json as _json
    import struct

    import autotune as autotune_module

    def gstr(text):
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    kvs = (
        gstr("general.architecture") + struct.pack("<I", 8) + gstr("qwen3")
        + gstr("qwen3.context_length") + struct.pack("<I", 4) + struct.pack("<I", 32768)
    )
    weights = tmp_path / "model-a.gguf"
    weights.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2)
        + kvs + bytes(32)
    )
    gc, registry = _real_gui_controller(
        tmp_path,
        f"""version: 1
models:
  - id: model-a
    name: "Model A"
    location: {_json.dumps(str(weights))}
    context_size: 8192
    gpu_layers: -1
    status: installed
    server_args: ["--parallel", "1"]
""",
    )
    seen_context = []

    def benchmark(model_id):
        seen_context.append(registry.get(model_id).context_size)
        return True, "61.5 tok/s", 61.5

    gc._benchmark_fn = benchmark
    original = autotune_module.run_smoke_trial
    autotune_module.run_smoke_trial = lambda mid, ctx, **kw: autotune_module.Trial(ctx, True)
    try:
        gc._do_autotune_context("model-a")
    finally:
        autotune_module.run_smoke_trial = original

    results = []
    while not gc.result_q.empty():
        results.append(gc.result_q.get_nowait())
    kinds = [r.kind for r in results]
    assert seen_context == [32768 * 4]  # benchmarked at the tuned context
    assert kinds.index("benchmark") < kinds.index("autotune")
    final = next(r for r in results if r.kind == "autotune")
    assert final.ok and final.payload["tokens_per_second"] == 61.5
    assert not gc.autotune_in_progress()


def _real_gui_controller(tmp_path, yaml_text):
    """Build a GuiController over a REAL Config-loaded registry (not FakeRegistry),
    so _do_delete_model's actual config.remove_model_entry + filesystem calls run
    for real. Only the model/whisper process controllers stay fake."""
    from config import Config
    from models import ModelRegistry

    (tmp_path / "models.yaml").write_text(yaml_text, encoding="utf-8")
    settings, models_data, issues = Config.load(tmp_path)
    errors = [i for i in issues if i.level == "ERROR"]
    assert not errors, errors
    registry = ModelRegistry(models_data, settings)
    gc = GuiController(
        settings, registry, FakeModelController(), FakeWhisper(), FakeManager(),
        fetch=lambda port, path: None,
    )
    return gc, registry


def test_delete_model_worker_deletes_the_file_and_removes_the_row(tmp_path):
    weights = tmp_path / "model-a.gguf"
    weights.write_bytes(b"weights")
    import json as _json

    gc, registry = _real_gui_controller(
        tmp_path,
        f"""version: 1
models:
  - id: model-a
    name: "Model A"
    location: {_json.dumps(str(weights))}
    context_size: 8192
    gpu_layers: -1
    status: installed
""",
    )
    gc.delete_model("model-a")
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert result.kind == "delete_model" and result.ok
    assert result.payload["deleted_files"] == [str(weights)]
    assert not weights.exists()
    assert registry.get("model-a") is None


def test_delete_model_worker_never_deletes_a_file_another_row_still_uses(tmp_path):
    """Two vision models sharing one mmproj file: deleting one must not take the
    projector out from under the surviving row."""
    weights_a = tmp_path / "model-a.gguf"
    weights_a.write_bytes(b"weights a")
    weights_b = tmp_path / "model-b.gguf"
    weights_b.write_bytes(b"weights b")
    shared_mmproj = tmp_path / "shared-mmproj.gguf"
    shared_mmproj.write_bytes(b"mmproj")
    import json as _json

    gc, registry = _real_gui_controller(
        tmp_path,
        f"""version: 1
models:
  - id: model-a
    name: "Model A"
    location: {_json.dumps(str(weights_a))}
    mmproj: {_json.dumps(str(shared_mmproj))}
    context_size: 8192
    gpu_layers: -1
    status: installed

  - id: model-b
    name: "Model B"
    location: {_json.dumps(str(weights_b))}
    mmproj: {_json.dumps(str(shared_mmproj))}
    context_size: 8192
    gpu_layers: -1
    status: installed
""",
    )
    gc.delete_model("model-a")
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert result.kind == "delete_model" and result.ok
    assert result.payload["deleted_files"] == [str(weights_a)]
    assert result.payload["skipped_shared_files"] == [str(shared_mmproj)]
    assert not weights_a.exists()
    assert shared_mmproj.exists()  # still owned by model-b
    assert registry.get("model-a") is None
    assert registry.get("model-b") is not None
    assert registry.get("model-b").mmproj == str(shared_mmproj)


def test_delete_model_worker_leaves_the_row_in_place_when_the_file_delete_fails(tmp_path):
    """A location that cannot be unlinked (here: it is a directory, a portable
    stand-in for "permission denied" / "file in use") must leave the registry
    row untouched - a partially-applied delete is worse than an honest refusal."""
    undeleatable = tmp_path / "not-a-file.gguf"
    undeleatable.mkdir()
    import json as _json

    gc, registry = _real_gui_controller(
        tmp_path,
        f"""version: 1
models:
  - id: model-a
    name: "Model A"
    location: {_json.dumps(str(undeleatable))}
    context_size: 8192
    gpu_layers: -1
    status: installed
""",
    )
    gc.delete_model("model-a")
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert result.kind == "delete_model" and not result.ok
    assert str(undeleatable) in result.error
    assert undeleatable.exists()
    assert registry.get("model-a") is not None


def test_delete_model_worker_refuses_when_the_model_started_running_meanwhile(tmp_path):
    """The UI-thread guard in delete_model() can race a Start landing on the ops
    worker between the confirm click and this method running - the worker
    re-checks running state itself rather than trusting the earlier check."""
    weights = tmp_path / "model-a.gguf"
    weights.write_bytes(b"weights")
    import json as _json

    gc, registry = _real_gui_controller(
        tmp_path,
        f"""version: 1
models:
  - id: model-a
    name: "Model A"
    location: {_json.dumps(str(weights))}
    context_size: 8192
    gpu_layers: -1
    status: installed
""",
    )
    gc.command_q.put(gui_controller.Command("delete_model", {"model_id": "model-a"}))
    gc._controller.running_model_id = "model-a"  # started after the UI-thread guard ran
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert result.kind == "delete_model" and not result.ok
    assert "stop" in result.error.lower()
    assert weights.exists()
    assert registry.get("model-a") is not None


# --------------------------------------------------------------------------- #
# request_launch_harness / _do_launch_harness (owner request 2026-08-21:
# "Launch in Claude Code / Codex / OpenCode" from the Chat picker)
# --------------------------------------------------------------------------- #


def _minimal_models_yaml():
    return (
        "version: 1\nmodels:\n  - id: model-a\n    name: \"Model A\"\n"
        "    location: \"\"\n    context_size: 8192\n    gpu_layers: -1\n"
        "    status: installed\n"
    )


def test_launch_harness_refuses_when_no_model_running(tmp_path):
    gc, _registry = _real_gui_controller(tmp_path, _minimal_models_yaml())
    gc.request_launch_harness("codex", str(tmp_path))
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert result.kind == "launch_harness" and not result.ok
    assert "no model running" in result.error.lower()


def test_launch_harness_refuses_when_no_project_dir(tmp_path, monkeypatch):
    import harness_launch

    gc, _registry = _real_gui_controller(tmp_path, _minimal_models_yaml())
    gc._controller.running_model_id = "model-a"
    gc._active_port = 8080
    monkeypatch.setattr(harness_launch, "detect_executable", lambda h: "/bin/codex")
    gc.request_launch_harness("codex", "")
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert result.kind == "launch_harness" and not result.ok
    assert "project folder" in result.error.lower()


def test_launch_harness_refuses_when_executable_missing(tmp_path, monkeypatch):
    import harness_launch

    gc, _registry = _real_gui_controller(tmp_path, _minimal_models_yaml())
    gc._controller.running_model_id = "model-a"
    gc._active_port = 8080
    monkeypatch.setattr(harness_launch, "detect_executable", lambda h: None)
    gc.request_launch_harness("codex", str(tmp_path))
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()
    assert result.kind == "launch_harness" and not result.ok
    assert "not found on path" in result.error.lower()


def test_launch_harness_codex_wires_provider_and_spawns(tmp_path, monkeypatch):
    import harness_launch

    gc, _registry = _real_gui_controller(tmp_path, _minimal_models_yaml())
    gc._controller.running_model_id = "model-a"
    gc._active_port = 8080
    monkeypatch.setattr(harness_launch, "detect_executable", lambda h: "/bin/codex")

    upserts = []
    spawns = []
    monkeypatch.setattr(
        harness_launch, "upsert_codex_provider", lambda url: upserts.append(url)
    )
    monkeypatch.setattr(
        harness_launch,
        "spawn_in_terminal",
        lambda argv, cwd, env, title: spawns.append((argv, cwd, env, title)),
    )

    gc.request_launch_harness("codex", str(tmp_path), remember=False)
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()

    assert result.kind == "launch_harness" and result.ok
    assert upserts == []  # provider configuration belongs to this process only
    assert len(spawns) == 1
    argv, cwd, env, _title = spawns[0]
    assert argv[:3] == ["codex", "-C", str(tmp_path)]
    assert cwd == str(tmp_path)
    assert env == {"LOCITIZE_CODEX_API_KEY": "locitize-local"}


def test_launch_harness_claude_env_targets_model_port_with_no_v1_suffix(tmp_path, monkeypatch):
    import harness_launch

    gc, _registry = _real_gui_controller(tmp_path, _minimal_models_yaml())
    gc._controller.running_model_id = "model-a"
    gc._active_port = 8091
    monkeypatch.setattr(harness_launch, "detect_executable", lambda h: "/bin/claude")

    spawns = []
    monkeypatch.setattr(
        harness_launch,
        "spawn_in_terminal",
        lambda argv, cwd, env, title: spawns.append((argv, cwd, env, title)),
    )

    gc.request_launch_harness("claude", str(tmp_path), remember=False)
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()

    assert result.kind == "launch_harness" and result.ok
    assert result.payload["note"]  # the tool-use/llama.cpp-version caveat is surfaced
    _argv, _cwd, env, _title = spawns[0]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8091"


def test_launch_harness_remember_persists_via_write_chat_harness_dir(tmp_path, monkeypatch):
    import harness_launch

    (tmp_path / "settings.yaml").write_text(
        "version: 1\nchat_harness:\n  last_project_dir: \"\"\n", encoding="utf-8"
    )
    gc, _registry = _real_gui_controller(tmp_path, _minimal_models_yaml())
    gc._controller.running_model_id = "model-a"
    gc._active_port = 8080
    monkeypatch.setattr(harness_launch, "detect_executable", lambda h: "/bin/opencode")
    monkeypatch.setattr(harness_launch, "write_opencode_project_config", lambda *a: None)
    monkeypatch.setattr(
        harness_launch, "spawn_in_terminal", lambda argv, cwd, env, title: None
    )

    project_dir = str(tmp_path)
    gc.request_launch_harness("opencode", project_dir, remember=True)
    _dispatch_next(gc)
    result = gc.result_q.get_nowait()

    assert result.ok
    assert gc._settings.chat_harness.last_project_dir == project_dir
    text = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert project_dir in text


def test_detect_harnesses_delegates_to_harness_launch(tmp_path, monkeypatch):
    import harness_launch

    gc, _registry = _real_gui_controller(tmp_path, _minimal_models_yaml())
    monkeypatch.setattr(
        harness_launch, "detect_harnesses", lambda: {"claude": None, "codex": "/bin/codex", "opencode": None}
    )
    assert gc.detect_harnesses() == {"claude": None, "codex": "/bin/codex", "opencode": None}


def test_last_project_dir_reads_from_settings(tmp_path):
    gc, _registry = _real_gui_controller(tmp_path, _minimal_models_yaml())
    gc._settings.chat_harness.last_project_dir = "/locitize-test/somewhere"
    assert gc.last_project_dir() == "/locitize-test/somewhere"


# ---- auto-tune cancellation (owner request 2026-08-22) ------------------- #


def _autotune_gguf(tmp_path, context_length=32768):
    """Write a synthetic GGUF whose header carries a known native context."""
    import struct

    def gstr(text):
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    kvs = (
        gstr("general.architecture") + struct.pack("<I", 8) + gstr("qwen3")
        + gstr("qwen3.context_length")
        + struct.pack("<I", 4)
        + struct.pack("<I", context_length)
    )
    weights = tmp_path / "model-a.gguf"
    weights.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2)
        + kvs + b"\x00" * 32
    )
    return weights


def _autotune_controller(tmp_path):
    """A real controller over a real registry holding one tunable model."""
    import json as _json

    weights = _autotune_gguf(tmp_path)
    return _real_gui_controller(
        tmp_path,
        f"""version: 1
models:
  - id: model-a
    name: "Model A"
    location: {_json.dumps(str(weights))}
    context_size: 8192
    gpu_layers: -1
    status: installed
    server_args: ["--parallel", "1"]
""",
    )


def test_cancel_autotune_is_refused_when_no_tune_is_running(tmp_path):
    """Nothing to cancel means False, so the UI never claims it stopped something."""
    gc, _registry = _autotune_controller(tmp_path)
    assert gc.autotune_in_progress() is False
    assert gc.cancel_autotune() is False


def test_requesting_a_tune_clears_a_stale_cancellation(tmp_path):
    """A Stop pressed during the LAST run must not abort the next one.

    Without the clear, the event left set by a previous cancel would make the
    following tune abort on its first check - the owner would press Auto-tune
    and get an instant unexplained "canceled".
    """
    gc, _registry = _autotune_controller(tmp_path)
    gc._autotune_cancel.set()
    gc.request_autotune_context("model-a")
    assert gc._autotune_cancel.is_set() is False


def test_cancelling_mid_tune_stops_the_search_and_restores_the_settings(tmp_path):
    """The real handler, cancelled from another thread exactly as the UI does.

    The trial stub sets the event the way run_smoke_trial does when it sees the
    owner's Stop mid-load, so this drives _do_autotune_context's whole
    cancellation path: the search stops, models.yaml goes back to what it was,
    and the published Result says canceled rather than failed.
    """
    import autotune as autotune_module

    gc, registry = _autotune_controller(tmp_path)
    tried = []

    def cancel_on_second_trial(model_id, context_size, **kwargs):
        tried.append(context_size)
        if len(tried) >= 2:
            # What the real trial does: notice the event, kill the tree, report.
            gc._autotune_cancel.set()
            return autotune_module.Trial(
                context_size, False, "canceled by the owner", canceled=True
            )
        return autotune_module.Trial(context_size, True)

    original = autotune_module.run_smoke_trial
    autotune_module.run_smoke_trial = cancel_on_second_trial
    try:
        gc._do_autotune_context("model-a")
    finally:
        autotune_module.run_smoke_trial = original

    results = []
    while not gc.result_q.empty():
        results.append(gc.result_q.get_nowait())
    final = [r for r in results if r.kind == "autotune"]
    assert len(final) == 1
    assert final[0].payload["canceled"] is True
    # A cancellation is not an error: nothing downstream should paint it red.
    assert final[0].error is None
    assert len(tried) == 2, "no further model loads after the cancel"
    # models.yaml was put back exactly as it was found.
    assert registry.get("model-a").context_size == 8192
    assert list(registry.get("model-a").server_args) == ["--parallel", "1"]


def test_the_running_flag_is_cleared_even_when_the_handler_raises(tmp_path):
    """A crash must not leave the UI offering to cancel a run that has ended."""
    import autotune as autotune_module

    gc, _registry = _autotune_controller(tmp_path)

    def explode(model_id, context_size, **kwargs):
        raise ValueError("this model cannot be tuned")

    original = autotune_module.run_smoke_trial
    autotune_module.run_smoke_trial = explode
    try:
        gc._do_autotune_context("model-a")
    except ValueError:
        pass
    finally:
        autotune_module.run_smoke_trial = original
    assert gc.autotune_in_progress() is False


def test_shutdown_cancels_an_in_progress_autotune(tmp_path):
    """Closing the window must not leave a trial holding the GPU.

    Teardown step 0c sets the event before the bounded worker join, so a tune in
    flight kills its own trial tree instead of outliving the window.
    """
    gc, _registry = _autotune_controller(tmp_path)
    gc._shutdown_autotune()
    assert gc._autotune_cancel.is_set() is True


# --------------------------------------------------------------------------- #
# Offload GPU - free LOCITIZE's own, never a foreign app (M17.8)
# --------------------------------------------------------------------------- #


def test_free_gpu_stops_model_and_terminates_our_orphans(monkeypatch):
    import gpu_ledger

    ctrl = FakeModelController()
    ctrl.running_model_id = "qwen3-14b"
    gc = _controller(model_ctrl=ctrl)
    gc._active_port = 8080

    ours = gpu_ledger.GpuProcess(pid=999, name="bin/llama-server.exe",
                                 used_mb=8000, is_locitize=True)
    foreign = gpu_ledger.GpuProcess(pid=222, name="LM Studio.exe",
                                    used_mb=None, is_locitize=False)
    killed = {}
    monkeypatch.setattr(gpu_ledger, "query_compute_apps", lambda *a, **k: "raw")
    monkeypatch.setattr(gpu_ledger, "parse_compute_apps", lambda _t: [ours, foreign])
    monkeypatch.setattr(gpu_ledger, "mark_ours", lambda procs, **k: procs)
    monkeypatch.setattr(
        gpu_ledger, "terminate_pids",
        lambda pids, **k: killed.update({p: True for p in pids}) or {p: True for p in pids},
    )

    gc.request_free_gpu()
    _dispatch_next(gc)

    assert ("stop",) in ctrl.calls          # supervised model stopped
    assert killed == {999: True}            # our orphan killed, NOT the foreign pid
    assert gc._active_port is None
    result = gc.result_q.get_nowait()
    assert result.kind == "gpu_free" and result.ok
    assert result.payload["freed"] == 1
    assert "LM Studio.exe" in result.payload["others"]


def test_free_gpu_with_only_foreign_frees_nothing(monkeypatch):
    import gpu_ledger

    gc = _controller()
    foreign = gpu_ledger.GpuProcess(pid=222, name="LM Studio.exe",
                                    used_mb=None, is_locitize=False)
    monkeypatch.setattr(gpu_ledger, "query_compute_apps", lambda *a, **k: "raw")
    monkeypatch.setattr(gpu_ledger, "parse_compute_apps", lambda _t: [foreign])
    monkeypatch.setattr(gpu_ledger, "mark_ours", lambda procs, **k: procs)
    calls = []
    monkeypatch.setattr(gpu_ledger, "terminate_pids",
                        lambda pids, **k: calls.append(list(pids)) or {})

    gc.request_free_gpu()
    _dispatch_next(gc)

    assert calls == [[]]                    # terminate asked to kill nothing
    result = gc.result_q.get_nowait()
    assert result.kind == "gpu_free" and result.payload["freed"] == 0
    assert result.payload["others"] == ["LM Studio.exe"]


# --------------------------------------------------------------------------- #
# Derived tok/s from cumulative counters (M18.10, newer llama.cpp builds)
# --------------------------------------------------------------------------- #

_NEW_BUILD_METRICS = """# HELP h
llamacpp:prompt_tokens_total {pt}
llamacpp:prompt_seconds_total {ps}
llamacpp:tokens_predicted_total {gt}
llamacpp:tokens_predicted_seconds_total {gs}
llamacpp:n_tokens_max 25138
"""


def _derive_stub():
    class _Stub:
        _prev_metrics_sample = None
        _derive_rates = GuiController._derive_rates

    return _Stub()


def _new_build_sample(pt, ps, gt, gs):
    from gui_controller import parse_metrics

    return parse_metrics(_NEW_BUILD_METRICS.format(pt=pt, ps=ps, gt=gt, gs=gs))


def test_parse_reads_the_new_seconds_counters():
    sample = _new_build_sample(1000, 10.0, 620, 20.0)
    assert sample.gen_tokens_total == 620
    assert sample.gen_seconds_total == 20.0
    assert sample.prompt_seconds_total == 10.0
    # The old direct gauges are absent in this build -> None before derivation.
    assert sample.gen_tokens_s is None


def test_first_poll_derives_session_average():
    stub = _derive_stub()
    sample = _new_build_sample(1000, 10.0, 620, 20.0)
    stub._derive_rates(sample)
    assert sample.gen_tokens_s == 31.0     # 620 / 20.0
    assert sample.prompt_tokens_s == 100.0  # 1000 / 10.0


def test_second_poll_derives_the_delta_rate():
    stub = _derive_stub()
    first = _new_build_sample(1000, 10.0, 620, 20.0)
    stub._derive_rates(first)
    second = _new_build_sample(1000, 10.0, 720, 22.0)  # +100 tok in +2s
    stub._derive_rates(second)
    assert second.gen_tokens_s == 50.0  # the recent generation's true speed


def test_counter_restart_falls_back_to_session_average():
    """A server restart resets counters; a backwards delta must not go negative."""
    stub = _derive_stub()
    stub._derive_rates(_new_build_sample(1000, 10.0, 620, 20.0))
    after_restart = _new_build_sample(10, 0.5, 40, 1.0)
    stub._derive_rates(after_restart)
    assert after_restart.gen_tokens_s == 40.0  # 40/1.0, never negative


def test_old_build_direct_gauge_wins():
    """On an old build the direct gauge exists; derivation must not override it."""
    from gui_controller import parse_metrics

    stub = _derive_stub()
    sample = parse_metrics(
        "llamacpp:predicted_tokens_seconds 47.4\n"
        "llamacpp:tokens_predicted_total 620\n"
        "llamacpp:tokens_predicted_seconds_total 20.0\n"
    )
    stub._derive_rates(sample)
    assert sample.gen_tokens_s == 47.4


def test_no_counters_at_all_stays_none():
    stub = _derive_stub()
    from gui_controller import parse_metrics

    sample = parse_metrics("llamacpp:n_tokens_max 5\n")
    stub._derive_rates(sample)
    assert sample.gen_tokens_s is None  # honest "-", never fabricated


def test_hub_cancel_bypasses_the_ops_worker_queue():
    """Perf audit 2026-08-31: Cancel must work WHILE the ops worker is busy.

    request_hub_cancel sets the live download's cancel event directly on the
    calling thread (mirroring cancel_autotune) - a queued command would sit
    behind a running auto-tune or benchmark for minutes. The contract: the
    event is set immediately and NO command is enqueued for it.
    """
    gc = _controller()
    before = gc.command_q.qsize()
    with gc._hub_lock:
        cancel_event = gc._hub_cancel
    assert not cancel_event.is_set()
    gc.request_hub_cancel()
    assert cancel_event.is_set(), "cancel event must be set synchronously"
    assert gc.command_q.qsize() == before, "no queued command - direct set only"


def test_open_openwebui_ready_routes_through_the_ops_worker():
    """Perf audit 2026-08-31: opening Open WebUI can run secure-proxy work
    (winget/caddy/certutil/.local DNS) that blocks for minutes - request_chat
    must therefore ENQUEUE the open instead of doing it on the caller's
    (UI) thread, answering the click with an immediate status Result."""
    gc = _controller()
    gc._active_port = 8080  # a model is "running"
    gc._settings.chat.preferred_ui = "openwebui"

    import webui as webui_mod

    original_available = webui_mod.webui_available
    original_listening = gc._openwebui_listening
    webui_mod.webui_available = lambda _s: True
    gc._openwebui_listening = lambda: True
    try:
        gc.request_chat()
    finally:
        webui_mod.webui_available = original_available
        gc._openwebui_listening = original_listening

    kinds = []
    while not gc.command_q.empty():
        kinds.append(gc.command_q.get_nowait().kind)
    assert "open_openwebui_ready" in kinds, "the open must be a queued command"
    statuses = []
    while not gc.result_q.empty():
        statuses.append(gc.result_q.get_nowait().kind)
    assert "chat_status" in statuses, "the click gets an immediate status"


# --------------------------------------------------------------------------- #
# One-launch thinking override from the GUI (owner request 2026-09-02).
# --------------------------------------------------------------------------- #


def test_gui_start_forwards_no_reasoning_keyword_by_default():
    """The Thinking box's default must leave the argv exactly as it was.

    An explicit reasoning=None would NOT be equivalent: build_start_spec reads
    None as the deliberate 'force thinking off for this scenario', so forwarding
    it on every ordinary Start would silently disable thinking registry-wide.
    """
    ctrl = FakeModelController()
    gc = _controller(model_ctrl=ctrl)
    gc.request_start("qwen3-14b")
    _dispatch_next(gc)
    assert ctrl.spec_kwargs == [{}]


def test_gui_start_forwards_the_reasoning_override_when_one_is_picked():
    ctrl = FakeModelController()
    gc = _controller(model_ctrl=ctrl)
    gc.request_start("qwen3-14b", reasoning={"effort": "low", "budget": 2048})
    _dispatch_next(gc)
    assert ctrl.spec_kwargs == [{"reasoning": {"effort": "low", "budget": 2048}}]


def test_gui_switch_also_carries_the_reasoning_override():
    """Switching models must honour the picked level, not just a cold start."""
    ctrl = FakeModelController()
    ctrl.running_model_id = "deepseek-r1-distill-14b"
    gc = _controller(model_ctrl=ctrl)
    gc.request_start("qwen3-14b", reasoning={"enabled": False})
    _dispatch_next(gc)
    assert ("switch", "qwen3-14b") in ctrl.calls
    assert ctrl.spec_kwargs == [{"reasoning": {"enabled": False}}]


# --------------------------------------------------------------------------- #
# Live running-model tracking (owner-observed 2026-09-03).
#
# The window showed NO running model while one was running. Every path that
# repainted it went through a Result this worker had produced, so a model
# started by something else in the process - the model router, switching
# because a browser picker changed - was invisible. The controller is the
# truth; the monitor now asks it on the cadence that already exists.
# --------------------------------------------------------------------------- #


def _drain(gc, kind):
    """Every queued Result of one kind, oldest first."""
    found = []
    while True:
        try:
            result = gc.result_q.get_nowait()
        except Exception:  # noqa: BLE001 - queue.Empty
            break
        if result.kind == kind:
            found.append(result)
    return found


def test_the_first_check_publishes_even_with_nothing_running():
    """A window opened while a model already runs must paint the truth rather
    than sit blank waiting for a change - so the first observation always
    publishes. (None, None) is a legitimate state and cannot be the sentinel."""
    ctrl = FakeModelController()
    gc = _controller(model_ctrl=ctrl)
    gc._check_running_model()
    published = _drain(gc, "running_model")
    assert len(published) == 1
    assert published[0].payload["running_model_id"] is None


def test_a_model_started_outside_the_gui_is_published():
    """This is the reported defect: the router switches on the shared
    controller, and nothing repainted the window."""
    ctrl = FakeModelController()
    gc = _controller(model_ctrl=ctrl)
    gc._check_running_model()          # first observation: nothing running
    _drain(gc, "running_model")
    ctrl.start("qwen3-14b")            # as the router would, on this controller
    gc._check_running_model()
    published = _drain(gc, "running_model")
    assert len(published) == 1
    assert published[0].payload["running_model_id"] == "qwen3-14b"


def test_an_unchanged_model_publishes_nothing():
    """One check per monitor cycle must not queue a Result every 1.5 seconds."""
    ctrl = FakeModelController()
    ctrl.start("qwen3-14b")
    gc = _controller(model_ctrl=ctrl)
    gc._check_running_model()
    _drain(gc, "running_model")
    for _ in range(5):
        gc._check_running_model()
    assert _drain(gc, "running_model") == []


def test_the_metrics_port_follows_a_model_the_gui_did_not_start():
    """Without this the monitor keeps polling the old port, or idles forever
    because it never had one."""
    ctrl = FakeModelController()
    gc = _controller(model_ctrl=ctrl)
    gc._check_running_model()
    assert gc._active_port is None
    ctrl.start("qwen3-14b")
    gc._check_running_model()
    assert gc._active_port == 8080


def test_a_model_stopped_outside_the_gui_is_published_too():
    ctrl = FakeModelController()
    ctrl.start("qwen3-14b")
    gc = _controller(model_ctrl=ctrl)
    gc._check_running_model()
    _drain(gc, "running_model")
    ctrl.stop()
    gc._check_running_model()
    published = _drain(gc, "running_model")
    assert len(published) == 1
    assert published[0].payload["running_model_id"] is None
    assert gc._active_port is None


def test_a_failing_controller_read_is_not_a_crash():
    """A status read on a background thread must never take the worker down."""

    class Broken:
        @property
        def running_model_id(self):
            raise RuntimeError("controller went away")

    gc = _controller(model_ctrl=Broken())
    gc._check_running_model()  # must not raise
    assert _drain(gc, "running_model") == []


def test_chat_starts_open_webui_itself_instead_of_asking(monkeypatch):
    """Open WebUI is the default chat: installed but not running means start it,
    not a "Start Open WebUI?" dialog and never the llama.cpp page."""
    import webui as webui_mod

    gc = _controller()
    gc._active_port = 8080
    gc._settings.chat.preferred_ui = "openwebui"
    monkeypatch.setattr(webui_mod, "webui_available", lambda _s: True)
    monkeypatch.setattr(gc, "_openwebui_listening", lambda: False)
    gc.request_chat()

    commands = []
    while not gc.command_q.empty():
        commands.append(gc.command_q.get_nowait().kind)
    results = []
    while not gc.result_q.empty():
        results.append(gc.result_q.get_nowait().kind)
    assert commands == ["start_openwebui"]
    assert "chat_offer_start" not in results


def test_a_failed_open_webui_start_reports_why_and_does_not_open_llamacpp():
    gc = _controller()
    gc._active_port = 8080
    opened = []
    gc._open_llamacpp = lambda reason="": opened.append("llamacpp")

    def boom():
        raise RuntimeError("port 8096 is in use")

    gc._openwebui_start_fn = boom
    gc._do_start_openwebui()
    gc._openwebui_start_fn = lambda: False
    gc._do_start_openwebui()

    errors = []
    while not gc.result_q.empty():
        r = gc.result_q.get_nowait()
        if r.kind == "chat" and not r.ok:
            errors.append(r.error)
    assert opened == []
    assert len(errors) == 2
    assert "port 8096 is in use" in errors[0]
    assert "did not become ready" in errors[1]


def test_chat_never_opens_a_friendly_name_that_is_not_open_webui(monkeypatch):
    """locitize.local answering is not enough: another app (an Agent Portal's
    Caddy) can own the name. Chat opens it only when it serves Open WebUI."""
    import secure_proxy

    gc = _controller()
    opened = []
    gc._open_url = lambda url, reason="": opened.append(url)
    monkeypatch.setattr(secure_proxy, "ensure", lambda settings, notify=None: (True, "ok"))

    gc._serves_openwebui = lambda url: False
    gc._open_openwebui()
    gc._serves_openwebui = lambda url: True
    gc._open_openwebui()

    port = gc._settings.ports.openwebui
    hostname = gc._settings.secure_proxy.hostname
    assert opened == [f"http://127.0.0.1:{port}/", f"https://{hostname}/"]


def test_autotune_all_tunes_each_model_and_stops_when_canceled():
    gc = _controller()
    gc._is_discovered = lambda mid: False
    ran = []

    def fake_tune(model_id):
        ran.append(model_id)
        if model_id == "b":
            gc._autotune_cancel.set()  # Stop pressed during the second model
            return "canceled"
        return "ok"

    gc._do_autotune_context = fake_tune
    gc._autotune_cancel.clear()
    gc._do_autotune_all(["a", "b", "c"])
    assert ran == ["a", "b"]  # c is skipped after the cancel
    results = []
    while not gc.result_q.empty():
        results.append(gc.result_q.get_nowait())
    final = [r for r in results if r.kind == "autotune_all"]
    assert len(final) == 1
    assert final[0].payload == {"total": 3, "tuned": 1, "failed": 0, "skipped": 2, "canceled": True}
    assert not gc.autotune_in_progress()


def test_cancel_works_between_two_models_of_an_autotune_all_run():
    gc = _controller()
    gc._autotune_batch = True  # between models: no single tune running
    gc._autotune_running = False
    assert gc.autotune_in_progress() is True
    assert gc.cancel_autotune() is True
    assert gc._autotune_cancel.is_set()
