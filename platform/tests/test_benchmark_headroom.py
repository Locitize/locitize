"""Tests for the M15.3 pre-run headroom guard and the throughput probe.

Both exist because of one measured incident: the same model and config
benchmarked at 16.6 and 214.3 tok/s ten minutes apart, the slow row taken
while system RAM sat at 98% - a machine-distress number recorded as if it
were the model's speed. The guard refuses that run before a model loads; the
probe is the measurement the load-only context auto-tune lacked.
"""

from __future__ import annotations

import pytest

from benchmark import BenchmarkConflictError, BenchmarkRunner
from config import Model, ModelRegistryData, Settings
from health import GpuInfo
from models import ModelRegistry

from fakes import FakeGpuInfoProvider, FakeSystemInfoProvider


def _model() -> Model:
    return Model(
        id="m", name="M", description="", location="/locitize-test/m.gguf",
        context_size=8192, gpu_layers=999,
    )


def _runner(sys_provider=None, gpu_provider=None, out=None, port_free=True):
    registry = ModelRegistry(ModelRegistryData(models=[_model()]), Settings())
    return BenchmarkRunner(
        _NullController(), registry, Settings(),
        gpu_provider=gpu_provider, sys_provider=sys_provider,
        out=out, port_is_free=lambda port: port_free,
    )


class _NullController:
    """Refuses to be started; headroom tests must never reach a start."""

    started = False

    def start(self, *a, **k):  # pragma: no cover - reaching this is the failure
        raise AssertionError("headroom guard must fire before any start")

    def stop(self) -> None:
        pass


# ---- RAM floor -------------------------------------------------------------- #


def test_ram_starvation_refuses_before_any_model_loads():
    runner = _runner(sys_provider=FakeSystemInfoProvider(ram_available=1500.0))
    with pytest.raises(BenchmarkConflictError) as excinfo:
        runner.assert_can_run()
    message = str(excinfo.value)
    assert "RAM" in message
    assert "1500" in message  # names the real number, not a vague complaint


def test_ram_at_the_floor_exactly_is_allowed():
    runner = _runner(
        sys_provider=FakeSystemInfoProvider(ram_available=BenchmarkRunner.RAM_FLOOR_MB)
    )
    runner.assert_can_run()  # no raise


def test_healthy_ram_passes():
    runner = _runner(sys_provider=FakeSystemInfoProvider(ram_available=16000.0))
    runner.assert_can_run()


def test_no_system_provider_means_no_ram_check_not_a_crash():
    _runner(sys_provider=None).assert_can_run()


# ---- VRAM warning ----------------------------------------------------------- #


def _gpu(free_mb: float) -> FakeGpuInfoProvider:
    return FakeGpuInfoProvider(
        [GpuInfo(name="FakeGPU", vram_total_mb=16303.0, vram_free_mb=free_mb)]
    )


def test_vram_held_by_other_apps_warns_but_does_not_refuse():
    lines: list[str] = []
    runner = _runner(gpu_provider=_gpu(free_mb=13000.0), out=lines.append)
    runner.assert_can_run()  # 3303 MB used by others: warn, never block
    assert any("WARNING" in line and "VRAM" in line for line in lines)


def test_normal_desktop_vram_use_stays_silent():
    lines: list[str] = []
    runner = _runner(gpu_provider=_gpu(free_mb=15600.0), out=lines.append)
    runner.assert_can_run()  # ~700 MB is a normal desktop; no noise
    assert lines == []


def test_no_gpu_at_all_is_not_an_error():
    _runner(gpu_provider=FakeGpuInfoProvider(None)).assert_can_run()


def test_port_conflict_still_fires_before_headroom():
    runner = _runner(
        sys_provider=FakeSystemInfoProvider(ram_available=1.0), port_free=False
    )
    with pytest.raises(BenchmarkConflictError) as excinfo:
        runner.assert_can_run()
    assert "port" in str(excinfo.value).lower()


# ---- probe_context_throughput ----------------------------------------------- #


class _FailingController:
    """start() reports a non-running status; stop() must still be called."""

    def __init__(self) -> None:
        self.stopped = False

    def start(self, *a, **k):
        from services import ServiceStatus

        return ServiceStatus.STOPPED_ERROR

    def stop(self) -> None:
        self.stopped = True


def test_probe_returns_none_and_stops_on_start_failure():
    registry = ModelRegistry(ModelRegistryData(models=[_model()]), Settings())
    controller = _FailingController()
    runner = BenchmarkRunner(controller, registry, Settings())
    assert runner.probe_context_throughput("m", 16384) is None
    assert controller.stopped is True


def test_probe_returns_none_for_an_unknown_model():
    registry = ModelRegistry(ModelRegistryData(models=[_model()]), Settings())
    runner = BenchmarkRunner(_FailingController(), registry, Settings())
    assert runner.probe_context_throughput("no-such-model", 16384) is None


def test_probe_returns_none_when_config_guard_raises():
    class _GuardController(_FailingController):
        def start(self, *a, **k):
            raise ValueError("missing location")

    registry = ModelRegistry(ModelRegistryData(models=[_model()]), Settings())
    runner = BenchmarkRunner(_GuardController(), registry, Settings())
    assert runner.probe_context_throughput("m", 16384) is None


# ---- RunLock stale reclaim (M15.4) ------------------------------------------ #


def test_stale_lock_with_dead_holder_is_reclaimed_and_announced(tmp_path):
    from benchmark import RunLock

    lock = tmp_path / "benchmark.lock"
    lock.write_text("999999999")  # a PID that cannot exist
    lines: list[str] = []
    with RunLock(lock, on_reclaim=lines.append):
        assert lock.exists()  # re-acquired by us
    assert not lock.exists()
    assert lines and "reclaim" in lines[0]


def test_lock_with_live_holder_is_still_refused(tmp_path):
    import os as _os

    from benchmark import RunLock

    lock = tmp_path / "benchmark.lock"
    lock.write_text(str(_os.getpid()))  # this very process: definitely alive
    with pytest.raises(BenchmarkConflictError):
        with RunLock(lock):
            pass
    assert lock.exists()  # never stolen from a live holder


def test_unreadable_lock_contents_are_respected_not_stolen(tmp_path):
    from benchmark import RunLock

    lock = tmp_path / "benchmark.lock"
    lock.write_text("not-a-pid")
    with pytest.raises(BenchmarkConflictError):
        with RunLock(lock):
            pass
    assert lock.exists()
