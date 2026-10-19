"""Boundary and recovery tests for the local adapter; model probes are mocked."""
import threading
from types import SimpleNamespace

import pytest
import config
from local_tuning import LocalTuning, candidates, choose_trial
from local_tuning import effective_context_from_log


def test_candidates_bounded_and_partial():
    model = SimpleNamespace(context_size=8192, gpu_layers=999)
    rows = candidates(model, 262144)
    assert len(rows) <= 5
    assert (8192, 32) in rows and (8192, 16) in rows
    assert all(ctx <= 262144 for ctx, _ in rows)


@pytest.mark.parametrize("saved,native,contexts", [
    (65536, 65536, {8192, 32768, 65536}),
    (65536, 262144, {8192, 32768, 65536}),
    (8192, 262144, {4096, 8192, 16384}),
    (16384, 262144, {8192, 16384, 32768}),
    (32768, 262144, {8192, 16384, 32768}),
    (131072, 65536, {8192, 32768, 65536}),
    (4096, 2048, {1024, 2048}),
    (2, 2, {1, 2}), (1, 1, {1}),
])
def test_context_comparison_survives_high_saved_and_native_limits(saved, native, contexts):
    rows = candidates(SimpleNamespace(context_size=saved, gpu_layers=999), native)
    assert rows[0] == (min(saved, native), 999)
    assert {ctx for ctx, _ in rows} == contexts
    assert 1 <= len(rows) <= 5
    assert len(rows) == len(set(rows))
    assert all(1 <= ctx <= native for ctx, _ in rows)
    assert (min(saved, native), 32) in rows
    assert (min(saved, native), 16) in rows


@pytest.mark.parametrize("layers", [16, 32, 999])
def test_duplicate_layer_and_context_candidates_stay_unique(layers):
    rows = candidates(SimpleNamespace(context_size=8192, gpu_layers=layers), 8192)
    assert len(rows) == len(set(rows)) <= 5
    assert len({ctx for ctx, _ in rows}) == 2


@pytest.mark.parametrize("native", [0, -1, True, None])
def test_invalid_native_context_refused(native):
    with pytest.raises(ValueError, match="positive integers"):
        candidates(SimpleNamespace(context_size=8192, gpu_layers=999), native)


def test_balanced_uses_measured_context_and_speed():
    rows = [{"context_effective": ctx, "tokens_per_second": speed}
            for ctx, speed in [(4096, 100), (8192, 86), (16384, 70)]]
    assert choose_trial(rows, "balanced")["context_effective"] == 8192
    assert choose_trial(rows, "speed")["context_effective"] == 4096


def test_balanced_middle_context_is_eligible_without_changing_policy():
    rows = candidates(SimpleNamespace(context_size=65536, gpu_layers=999), 262144)
    assert rows == [(65536, 999), (65536, 32), (65536, 16), (8192, 999), (32768, 999)]
    # Reproduce the established comparison as test data; it does not claim a
    # new benchmark or extend the chooser beyond its unchanged 85% contract.
    trials = [{"context_effective": ctx, "tokens_per_second": speed}
              for ctx, speed in [(65536, 50), (8192, 87.25), (32768, 86.99)]]
    assert choose_trial(trials, "balanced")["context_effective"] == 32768
    assert choose_trial(trials, "speed")["context_effective"] == 8192


def test_compare_and_write_preserves_unrelated_and_refuses_edit(tmp_path):
    target = tmp_path / "models.yaml"
    target.write_text("models:\n  - id: a\n    gpu_layers: 999 # preserve\n    context_size: 8192\n  - id: b\n    gpu_layers: 4\n    context_size: 4096\n")
    config.write_model_fields(tmp_path, "a", 32, 4096, expected_fields=(999, 8192))
    assert "gpu_layers: 4\n    context_size: 4096" in target.read_text()
    assert "# preserve" in target.read_text()
    before = target.read_bytes()
    with pytest.raises(ValueError, match="concurrently"):
        config.write_model_fields(tmp_path, "a", 16, 1024, expected_fields=(999, 8192))
    assert target.read_bytes() == before


def test_invalid_job_and_objective():
    helper = LocalTuning()
    with pytest.raises(ValueError):
        helper.tuning_status("no-such-job")
    with pytest.raises(ValueError):
        helper.start_tuning("anything", "quality")


@pytest.mark.parametrize("mode, expected", [("success", "completed"),
    ("cancel", "cancelled"), ("verify_fail", "failed"),
    ("restore_conflict", "recovery_failed"), ("baseline_fail", "failed")])
def test_job_recovery(monkeypatch, mode, expected):
    helper = LocalTuning()
    job = "job"
    helper._jobs[job] = {"objective": "speed", "trials": [], "state": "queued"}
    helper._cancel[job] = threading.Event()
    model = SimpleNamespace(id="a", context_size=4096, gpu_layers=999)
    manager = SimpleNamespace(stop_all=lambda: None)
    runtime = (SimpleNamespace(data_dir="unused"), None, manager, None, None, None)
    writes = []
    probes = []

    def probe(*args):
        probes.append(args)
        if mode == "cancel":
            helper._cancel[job].set()
        verify = len(probes) > len(candidates(model, 8192))
        if mode == "restore_conflict" and verify:
            helper._cancel[job].set()
        speed = None if mode == "baseline_fail" or (mode == "verify_fail" and verify) else 100
        return {"context_requested": args[-2], "gpu_layers_requested": args[-1],
                "context_effective": args[-2], "tokens_per_second": speed}

    def writer(*args, **kwargs):
        writes.append((args, kwargs))
        if mode == "restore_conflict" and len(writes) == 2:
            raise ValueError("concurrent edit")

    monkeypatch.setattr(helper, "_probe", probe)
    monkeypatch.setattr(config, "write_model_fields", writer)
    monkeypatch.setattr("local_tuning.Launcher._confirm_clean", lambda *args: True)
    lock = SimpleNamespace(__exit__=lambda: None)
    helper._run(job, runtime, model, 8192, lock)
    assert helper.tuning_status(job)["state"] == expected
    assert len(writes) == (0 if mode in ("cancel", "baseline_fail") else 1 if mode == "success" else 2)
    if len(writes) == 2:
        assert writes[1][0][2:] == (999, 4096)
    if mode == "restore_conflict":
        assert helper._recovery["applied"] is not None
        assert helper.stop_local_model()["cleanup_confirmed"] is True
        assert len(writes) == 3
        assert helper._recovery is None


def test_failed_apply_never_restores(monkeypatch):
    helper = LocalTuning()
    helper._jobs["j"] = {"objective": "speed", "trials": [], "state": "queued"}
    helper._cancel["j"] = threading.Event()
    model = SimpleNamespace(id="a", context_size=4096, gpu_layers=999)
    runtime = (SimpleNamespace(data_dir="unused"), None,
               SimpleNamespace(stop_all=lambda: None), None, None, None)
    monkeypatch.setattr(helper, "_probe", lambda *args:
        {"context_requested": args[-2], "gpu_layers_requested": args[-1],
         "context_effective": args[-2], "tokens_per_second": 100})
    writes = []
    def fail(*args, **kwargs):
        writes.append(args)
        raise ValueError("concurrent edit")
    monkeypatch.setattr(config, "write_model_fields", fail)
    monkeypatch.setattr("local_tuning.Launcher._confirm_clean", lambda *args: True)
    helper._run("j", runtime, model, 8192, SimpleNamespace(__exit__=lambda: None))
    assert len(writes) == 1
    assert helper.tuning_status("j")["state"] == "failed"


def test_owned_responses_router_lifecycle(monkeypatch):
    from services import ServiceStatus
    helper = LocalTuning()
    model = SimpleNamespace(id="a", location="registered", backend="")
    registry = SimpleNamespace(get=lambda _: model, launchable=lambda: [model])
    events = []
    manager = SimpleNamespace(stop_all=lambda: events.append("model-stop"))
    controller = SimpleNamespace(start=lambda _: ServiceStatus.RUNNING,
        running_port=8080, running_model_id="a")
    runner = SimpleNamespace(assert_can_run=lambda: None,
        _post_completion=lambda *args: {})
    settings = SimpleNamespace(ports=SimpleNamespace(router=8099))
    runtime = (settings, registry, manager, controller, runner, None)
    monkeypatch.setattr(helper, "_runtime", lambda: runtime)
    monkeypatch.setattr("local_tuning.resolve_log_dir", lambda _: __import__("pathlib").Path("."))
    monkeypatch.setattr("local_tuning.RunLock", lambda _: SimpleNamespace(
        __enter__=lambda: None, __exit__=lambda: events.append("unlock")))
    monkeypatch.setattr("local_tuning.parse_completion_timings", lambda _: SimpleNamespace(predicted_per_second=100))
    monkeypatch.setattr("local_tuning.Launcher._confirm_clean", lambda *args: True)
    captured = {}
    def router_factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(start=lambda: events.append("router-start"),
            stop=lambda: events.append("router-stop"),
            base_url=lambda: "http://127.0.0.1:8099/v1")
    monkeypatch.setattr("local_tuning.ModelRouter", router_factory)
    response = helper.start_local_model("a")
    assert response["codex_api_base_url"] == "http://127.0.0.1:8099/v1"
    assert response["router_port"] == 8099
    assert captured["registry_fn"]() == [model]
    assert captured["switch_fn"]("other")[0] is False
    assert helper.stop_local_model()["cleanup_confirmed"]
    assert events == ["router-start", "router-stop", "model-stop", "unlock"]


@pytest.mark.parametrize("field", ["n_ctx_slot", "n_ctx_seq", "n_ctx_per_seq"])
def test_probe_observes_actual_slot_capacity(monkeypatch, tmp_path, field):
    helper = LocalTuning()
    log = tmp_path / "trial.log"
    # Stale output must not become this trial's effective capacity.
    log.write_text("n_ctx_slot = 32768\n")
    def probe(*args, **kwargs):
        with log.open("a") as handle:
            handle.write(f"{field} = 2048\noffloaded 16/33 layers\n")
        return 20
    runner = SimpleNamespace(probe_context_throughput=probe)
    runtime = (None, None, None, None, runner, log)
    monkeypatch.setattr("local_tuning.Launcher._confirm_clean", lambda *args: True)
    monkeypatch.setattr(helper, "assess_computer", lambda: {"gpus": None})
    result = helper._probe(runtime, "a", 8192, 16)
    assert result["context_requested"] == 8192
    assert result["context_effective"] == 2048
    assert result["gpu_layers_effective"] == 16


@pytest.mark.parametrize("raises", [False, True])
def test_recovery_retains_owner_and_retries_until_clean(monkeypatch, raises):
    helper = LocalTuning()
    helper._jobs["j"] = {"objective": "speed", "trials": [], "state": "queued"}
    helper._cancel["j"] = threading.Event()
    model = SimpleNamespace(id="a", context_size=4096, gpu_layers=999)
    state = {"clean": False}
    events = []
    def stop():
        events.append("stop")
        if raises and not state["clean"]:
            raise RuntimeError("stop failed")
    manager = SimpleNamespace(stop_all=stop)
    runtime = (SimpleNamespace(data_dir="unused"), None, manager, None, None, None)
    def fail(*args):
        raise RuntimeError("probe failed")
    monkeypatch.setattr(helper, "_probe", fail)
    monkeypatch.setattr("local_tuning.Launcher._confirm_clean", lambda *args: state["clean"])
    lock = SimpleNamespace(__exit__=lambda: events.append("unlock"))
    helper._run("j", runtime, model, 8192, lock)
    assert helper._recovery["runtime"] is runtime
    assert helper.tuning_status("j")["state"] == "recovery_failed"
    assert helper.stop_local_model() == {"state": "recovery_failed", "cleanup_confirmed": False}
    assert helper.close()["cleanup_confirmed"] is False
    assert "unlock" not in events
    state["clean"] = True
    assert helper.close() == {"state": "stopped", "cleanup_confirmed": True}
    assert events.count("unlock") == 1
    assert helper._recovery is None
    assert helper.tuning_status("j")["recovery_completed"] is True


def test_router_shutdown_failure_still_stops_model(monkeypatch):
    helper = LocalTuning()
    events = []
    def broken_router():
        events.append("router")
        raise RuntimeError("router stop failed")
    helper._router = SimpleNamespace(stop=broken_router)
    helper._chat = (None, None, SimpleNamespace(stop_all=lambda: events.append("model")))
    helper._chat_lock = SimpleNamespace(__exit__=lambda: events.append("unlock"))
    monkeypatch.setattr("local_tuning.Launcher._confirm_clean", lambda *args: True)
    assert helper.stop_local_model()["cleanup_confirmed"] is False
    assert events == ["router", "model"]
    helper._router.stop = lambda: events.append("router-retry")
    assert helper.stop_local_model()["cleanup_confirmed"] is True
    assert events[-3:] == ["router-retry", "model", "unlock"]


def test_cpu_readings_unknown_explicitly(monkeypatch):
    helper = LocalTuning()
    def unknown(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr("local_tuning.host_platform.processor", unknown)
    monkeypatch.setattr("local_tuning.DefaultSystemInfoProvider", lambda: SimpleNamespace(
        ram_total_mb=lambda: 100, ram_available_mb=lambda: 50))
    monkeypatch.setattr("local_tuning.NvidiaSmiGpuInfoProvider", lambda: SimpleNamespace(gpus=lambda: None))
    result = helper.assess_computer()
    assert result["cpu"] == {"name": None, "logical_cores": None, "physical_cores": None}
    assert result["gpus"] is None


def test_start_rollback_router_error_preserves_owner(monkeypatch):
    from services import ServiceStatus
    helper = LocalTuning()
    model = SimpleNamespace(id="a", location="registered", backend="")
    events = []
    runtime = (SimpleNamespace(ports=SimpleNamespace(router=8099)),
        SimpleNamespace(get=lambda _: model, launchable=lambda: [model]),
        SimpleNamespace(stop_all=lambda: events.append("model-stop")),
        SimpleNamespace(start=lambda _: ServiceStatus.RUNNING, running_port=8080,
                        running_model_id="a"),
        SimpleNamespace(assert_can_run=lambda: None, _post_completion=lambda *args: {}), None)
    monkeypatch.setattr(helper, "_runtime", lambda: runtime)
    monkeypatch.setattr("local_tuning.resolve_log_dir", lambda _: __import__("pathlib").Path("."))
    monkeypatch.setattr("local_tuning.RunLock", lambda _: SimpleNamespace(
        __enter__=lambda: None, __exit__=lambda: events.append("unlock")))
    monkeypatch.setattr("local_tuning.parse_completion_timings", lambda _: SimpleNamespace(predicted_per_second=100))
    monkeypatch.setattr("local_tuning.Launcher._confirm_clean", lambda *args: True)
    def broken():
        raise RuntimeError("router failure")
    router = SimpleNamespace(start=broken, stop=broken)
    monkeypatch.setattr("local_tuning.ModelRouter", lambda **kwargs: router)
    with pytest.raises(RuntimeError):
        helper.start_local_model("a")
    assert events == ["model-stop"]
    assert helper._chat is runtime
    assert helper.stop_local_model()["cleanup_confirmed"] is False
    router.stop = lambda: events.append("router-retry")
    assert helper.close()["cleanup_confirmed"] is True
    assert events[-3:] == ["router-retry", "model-stop", "unlock"]


def test_actual_context_log_unknown_and_fresh_isolation(tmp_path):
    log = tmp_path / "start.log"
    assert effective_context_from_log(log, 0) is None
    log.write_text("n_ctx_slot = 32768\n")
    offset = log.stat().st_size
    assert effective_context_from_log(log, offset) is None
    with log.open("a") as handle:
        handle.write("n_ctx = 65536\n")
    assert effective_context_from_log(log, offset) is None
    with log.open("a") as handle:
        handle.write("n_ctx_slot = 8192\n")
    assert effective_context_from_log(log, offset) == 8192


def test_windows_empty_cpu_uses_bounded_registry(monkeypatch):
    import sys
    from local_tuning import cpu_name
    class Key:
        def __enter__(self): return self
        def __exit__(self, *_): pass
    seen = []
    registry = SimpleNamespace(HKEY_LOCAL_MACHINE=1,
        OpenKey=lambda *args: seen.append(args) or Key(),
        QueryValueEx=lambda *_: (" Intel(R) Core(TM) Ultra 7 265KF\n", 1))
    monkeypatch.setitem(sys.modules, "winreg", registry)
    monkeypatch.setattr("local_tuning.host_platform.processor", lambda: "")
    monkeypatch.setattr("local_tuning.host_platform.system", lambda: "Windows")
    assert cpu_name() == "Intel(R) Core(TM) Ultra 7 265KF"
    assert len(seen) == 1
    monkeypatch.setattr("local_tuning.host_platform.processor", lambda: "AMD64")
    assert cpu_name() == "Intel(R) Core(TM) Ultra 7 265KF"
    monkeypatch.setattr("local_tuning.host_platform.processor",
        lambda: "Intel64 Family 6 Model 198 Stepping 2, GenuineIntel")
    assert cpu_name() == "Intel(R) Core(TM) Ultra 7 265KF"
    for specific in ["Intel(R) Core(TM) Ultra 7 265KF", "AMD Ryzen 9 9950X"]:
        monkeypatch.setattr("local_tuning.host_platform.processor", lambda: specific)
        assert cpu_name() == specific
    monkeypatch.setattr("local_tuning.host_platform.processor",
        lambda: "Intel64 Family 6 Model 198 Stepping 2, GenuineIntel")
    registry.QueryValueEx = lambda *_: ("x" * 600, 1)
    assert len(cpu_name()) == 256
    def denied(*_): raise PermissionError("denied")
    registry.OpenKey = denied
    assert cpu_name() is None
    monkeypatch.setattr("local_tuning.host_platform.processor", lambda: "Primary CPU")
    assert cpu_name() == "Primary CPU"
