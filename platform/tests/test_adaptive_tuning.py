"""Adaptive decisions use injected measurements; these are not hardware benchmarks."""
import threading
from types import SimpleNamespace
import pytest
import config
from local_tuning import (LocalTuning, adaptive_trials, adaptive_choice, EXPLORATORY_BUDGET)


def trial(context, layers, speed=100):
    return {"context_requested": context, "context_effective": context,
            "gpu_layers_requested": layers, "gpu_layers_effective": None,
            "tokens_per_second": speed}


def explore(saved=4096, native=131072, blocks=48, speed=None, cancel=None):
    model = SimpleNamespace(context_size=saved, gpu_layers=999)
    return adaptive_trials(model, native, blocks,
        lambda ctx, layers: trial(ctx, layers, speed(ctx, layers) if speed else 100),
        cancel or (lambda: False), lambda *_: None)


def test_saved4k_grows_through32k_and_native_with_budget():
    rows = explore()
    assert rows[0]["context_requested"] == 4096
    assert {8192, 16384, 32768, 65536, 131072} <= {r["context_requested"] for r in rows}
    assert len(rows) <= EXPLORATORY_BUDGET
    assert adaptive_choice(rows)["context_effective"] == 131072
    assert 49 in {r["gpu_layers_requested"] for r in rows}
    assert all(r["gpu_layers_effective"] is None for r in rows)


def test_high_saved_context_keeps_lower_and_larger_alternatives():
    rows = explore(saved=65536, native=262144)
    assert {8192, 16384, 32768, 65536, 131072} <= {r["context_requested"] for r in rows}
    assert len(rows) <= 16
    assert len({(r["context_requested"], r["gpu_layers_requested"]) for r in rows}) == len(rows)


def test_layer_cliff_and_context_cliff_choose_measured_middle():
    def speed(context, layers):
        if context > 32768: return 40
        return {49: 30, 33: 100, 32: 5, 31: 125, 34: 80, 16: 70, 0: 5}.get(layers, 60)
    rows = explore(speed=speed)
    chosen = adaptive_choice(rows)
    assert (chosen["context_effective"], chosen["gpu_layers_requested"]) == (32768, 31)
    assert any(r["phase"] == "context_offload_recheck" for r in rows)
    assert len(rows) <= 16


def test_missing_layers_stays_bounded_and_missing_native_refused():
    rows = explore(blocks=None, native=32768)
    assert len(rows) <= 16
    assert adaptive_choice(rows)["context_effective"] == 32768
    assert explore(native=4096) == []


def test_cancellation_stops_between_probes():
    cancelled = threading.Event()
    calls = []
    def probe(ctx, layers):
        calls.append((ctx, layers)); cancelled.set()
        return trial(ctx, layers)
    rows = adaptive_trials(SimpleNamespace(context_size=4096, gpu_layers=999), 131072, 48,
                           probe, cancelled.is_set, lambda *_: None)
    assert len(calls) == len(rows) == 1


@pytest.mark.parametrize("context,speed", [(4096, 100), (8192, 19.9), (None, 100), (8192, float("nan"))])
def test_eligibility_refuses_unknown_capacity_and_unmet_floor(context, speed):
    row = trial(8192, 16, speed); row["context_effective"] = context
    assert adaptive_choice([row]) is None


def run_job(monkeypatch, speed=100, native=32768, final_speeds=None, baseline_failed=False,
            conflict=False, cancel_final=False):
    helper = LocalTuning(); job = "j"; event = threading.Event()
    helper._jobs[job] = {"objective": "balanced", "trials": [], "state": "queued",
                         "model_metadata": {"block_count": 48}}
    helper._cancel[job] = event
    model = SimpleNamespace(id="a", context_size=4096, gpu_layers=999)
    runtime = (SimpleNamespace(data_dir="unused"), None, SimpleNamespace(stop_all=lambda: None), None, None, None)
    order, writes, final = [], [], []
    def probe(_runtime, _id, ctx, layers):
        verifying = helper._jobs[job]["state"] == "verifying"
        order.append("verify" if verifying else "explore")
        if verifying:
            value = (final_speeds or [speed, speed])[len(final)]
            final.append(value)
            if cancel_final: event.set()
        else:
            value = None if baseline_failed and ctx == 4096 else speed
        return trial(ctx, layers, value)
    def writer(*args, **kwargs):
        order.append("write"); writes.append((args, kwargs))
        if conflict: raise ValueError("concurrent edit")
    monkeypatch.setattr(helper, "_probe", probe)
    monkeypatch.setattr(config, "write_model_fields", writer)
    monkeypatch.setattr("local_tuning.Launcher._confirm_clean", lambda *_: True)
    helper._run(job, runtime, model, native, SimpleNamespace(__exit__=lambda: None))
    return helper.tuning_status(job), order, writes


def test_repeated_verification_precedes_atomic_write_and_baseline_can_recover(monkeypatch):
    status, order, writes = run_job(monkeypatch, baseline_failed=True)
    assert status["state"] == "completed"
    assert order[-3:] == ["verify", "verify", "write"]
    assert len(writes) == 1 and writes[0][1]["expected_fields"] == (999, 4096)
    assert status["preferred_target_met"] is True
    assert len(status["trials"]) <= 16 and len(status["verifications"]) == 2
    assert status["cleanup_confirmed"] is True


@pytest.mark.parametrize("speed,native", [(10, 32768), (100, 4096)])
def test_unmet_target_is_explicit_and_registry_unchanged(monkeypatch, speed, native):
    status, _, writes = run_job(monkeypatch, speed=speed, native=native)
    assert status["state"] == "target_not_met"
    assert status["outcome"] == "target-not-met" and status["settings_applied"] is False
    assert not writes and status["cleanup_confirmed"]


def test_minimum_only_is_honest_about_preferred_target(monkeypatch):
    status, _, writes = run_job(monkeypatch, speed=25)
    assert status["state"] == "completed" and writes
    assert status["preferred_target_met"] is False


@pytest.mark.parametrize("final_speeds", [[100, 84], [100, None], [19, 100]])
def test_noisy_or_failing_verification_preserves_settings(monkeypatch, final_speeds):
    status, _, writes = run_job(monkeypatch, final_speeds=final_speeds)
    assert status["state"] == "target_not_met" and not writes
    assert len(status["verifications"]) == 2


def test_concurrent_edit_refuses_apply_after_verification(monkeypatch):
    status, order, writes = run_job(monkeypatch, conflict=True)
    assert status["state"] == "failed"
    assert len(writes) == 1 and order[-3:] == ["verify", "verify", "write"]


def test_cancel_during_verification_never_applies(monkeypatch):
    status, _, writes = run_job(monkeypatch, cancel_final=True)
    assert status["state"] == "cancelled" and not writes and status["cleanup_confirmed"]


def test_target_not_met_is_terminal_for_cancellation(monkeypatch):
    helper = LocalTuning()
    helper._jobs["done"] = {"state": "target_not_met", "cancel_requested": False}
    helper._cancel["done"] = threading.Event()
    assert helper.cancel_tuning("done")["state"] == "target_not_met"
    assert not helper._cancel["done"].is_set()
    assert helper.tuning_status("done")["cancel_requested"] is False


def test_resource_exhaustion_keeps_complete_registry_unchanged(monkeypatch):
    status, _, writes = run_job(monkeypatch, speed=None)
    assert status["state"] == "target_not_met" and not writes
    assert status["best_observed_result"] is None
    assert status["cleanup_confirmed"] and len(status["trials"]) <= 16


def test_cleanup_failure_retains_lock_and_can_restore_verified_apply(monkeypatch):
    helper = LocalTuning(); job = "cleanup"
    helper._jobs[job] = {"objective": "balanced", "trials": [], "state": "queued",
                         "model_metadata": {"block_count": 48}}
    helper._cancel[job] = threading.Event()
    def fail_stop(): raise RuntimeError("owned cleanup failed")
    manager = SimpleNamespace(stop_all=fail_stop)
    runtime = (SimpleNamespace(data_dir="unused"), None, manager, None, None, None)
    writes, releases = [], []
    monkeypatch.setattr(helper, "_probe", lambda _r, _id, ctx, layers: trial(ctx, layers))
    monkeypatch.setattr(config, "write_model_fields", lambda *args, **kwargs: writes.append((args, kwargs)))
    monkeypatch.setattr("local_tuning.Launcher._confirm_clean", lambda *_: True)
    lock = SimpleNamespace(__exit__=lambda: releases.append(True))
    helper._run(job, runtime, SimpleNamespace(id="a", context_size=4096, gpu_layers=999), 32768, lock)
    assert helper.tuning_status(job)["state"] == "recovery_failed" and not releases
    assert helper._recovery["applied"] is not None
    manager.stop_all = lambda: None
    assert helper.stop_local_model()["cleanup_confirmed"]
    assert len(writes) == 2 and writes[-1][0][2:] == (999, 4096)
    assert releases == [True] and helper.tuning_status(job)["settings_applied"] is False


def test_registered_quantization_and_layer_metadata_in_job(monkeypatch, tmp_path):
    helper = LocalTuning()
    model = config.Model(id="a", name="A", description="", location=str(tmp_path / "model.gguf"),
                         context_size=4096, gpu_layers=999, quantization="Q5_K_M")
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    registry = SimpleNamespace(get=lambda _: model, launchable=lambda: [model])
    runtime = (SimpleNamespace(data_dir=tmp_path), registry, None, None,
               SimpleNamespace(assert_can_run=lambda: None), None)
    monkeypatch.setattr(helper, "_runtime", lambda: runtime)
    monkeypatch.setattr(helper, "assess_computer", lambda: {"ram_available_mb": 12000})
    monkeypatch.setattr("local_tuning.gguf_meta.read_native_context_length", lambda _: 131072)
    monkeypatch.setattr("local_tuning.gguf_meta.read_gguf_header", lambda *_a, **_k:
        SimpleNamespace(architecture="test", block_count=48))
    monkeypatch.setattr("local_tuning.resolve_log_dir", lambda _: tmp_path)
    class Lock:
        def __init__(self, *_): pass
        def __enter__(self): pass
        def __exit__(self): pass
    monkeypatch.setattr("local_tuning.RunLock", Lock)
    monkeypatch.setattr("local_tuning.threading.Thread", lambda **_:
        SimpleNamespace(start=lambda: None, is_alive=lambda: False))
    status = helper.start_tuning("a", "balanced")
    assert status["model_metadata"]["quantization"] == "Q5_K_M"
    assert status["model_metadata"]["block_count"] == 48
    assert status["model_metadata"]["file_bytes"] == 4
    assert status["hardware_at_start"]["ram_available_mb"] == 12000
    assert status["remaining_exploratory_budget"] == 16


def test_missing_native_context_protocol_refuses_without_paths_or_side_effects(monkeypatch):
    import gguf_meta
    from local_tuning_mcp import _call
    helper = LocalTuning()
    model = SimpleNamespace(id="a", location="private-sensitive-model-file", backend="")
    registry = SimpleNamespace(get=lambda _: model, launchable=lambda: [model])
    monkeypatch.setattr(helper, "_runtime", lambda: (None, registry, None, None, None, None))
    def missing(_): raise gguf_meta.GgufError("private-sensitive-model-file missing context")
    monkeypatch.setattr("local_tuning.gguf_meta.read_native_context_length", missing)
    monkeypatch.setattr(helper, "_probe", lambda *_: pytest.fail("No native context may be probed"))
    monkeypatch.setattr(config, "write_model_fields", lambda *_a, **_k: pytest.fail("No settings may be written"))
    response = _call(helper.start_tuning, "a", "balanced")
    assert response == {"ok": False, "error": "Native GGUF context metadata unavailable; tuning refused"}
    assert not helper._jobs and helper._thread is None


@pytest.mark.parametrize("saved_layers", [999, -1])
def test_saved_full_offload_sentinel_uses_actual_layer_neighborhood(saved_layers):
    rows = adaptive_trials(SimpleNamespace(context_size=8192, gpu_layers=saved_layers), 131072, 48,
        lambda ctx, layers: trial(ctx, layers, 200 if layers == saved_layers or ctx > 8192 else 100),
        lambda: False, lambda *_: None)
    assert rows[0]["gpu_layers_requested"] == saved_layers
    assert rows[0]["phase"] == "baseline"
    neighbors = {row["gpu_layers_requested"] for row in rows if row["phase"] == "offload_neighbors"}
    assert {48, 47} <= neighbors
    growth = [row for row in rows if row["phase"] == "context_growth"]
    assert growth and all(row["gpu_layers_requested"] == 49 for row in growth)
    assert len(rows) <= 16


@pytest.mark.parametrize("saved_layers", [999, -1])
def test_auto_offload_uses_observed_partial_layer_region(saved_layers):
    def probe(ctx, layers):
        row = trial(ctx, layers, 200 if layers == saved_layers or ctx > 8192 else 100)
        row["gpu_layers_effective"] = 25 if layers == saved_layers else layers
        return row
    rows = adaptive_trials(SimpleNamespace(context_size=8192, gpu_layers=saved_layers), 131072, 48,
                           probe, lambda: False, lambda *_: None)
    assert rows[0]["gpu_layers_requested"] == saved_layers
    assert rows[0]["gpu_layers_effective"] == 25
    neighbors = {row["gpu_layers_requested"] for row in rows if row["phase"] == "offload_neighbors"}
    assert {24, 26, 23, 27} <= neighbors
    assert all(row["gpu_layers_requested"] == 25 for row in rows if row["phase"] == "context_growth")
    assert len(rows) <= 16
