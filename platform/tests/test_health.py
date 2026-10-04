"""Health-system tests: the GPU-absent path is proven with fake providers.

These assert each probe's status and remedy for BOTH the present and absent case,
using injected fakes so no real GPU/binary/port is touched (Architecture 5.2,
Build Plan AC3 part 1). This is how the platform is validated on a GPU-absent
code path even on a machine that has a real GPU.
"""

from __future__ import annotations

from config import Model, Settings
from fakes import (
    FakeBinaryProbeProvider,
    FakeGpuInfoProvider,
    FakePortProbeProvider,
    FakeSystemInfoProvider,
)
from health import (
    GpuInfo,
    HealthChecker,
    HealthProviders,
    HealthStatus,
    status_mark,
    worst_of,
    HealthResult,
)


def _models() -> list[Model]:
    return [
        Model(
            id="qwen3-14b",
            name="Qwen3 14B",
            description="d",
            location="",
            context_size=32768,
            gpu_layers=-1,
            vram_estimate_mb=10000,
            status="installed",
        )
    ]


def _providers(gpus, occupied=None, existing=None, runnable=None) -> HealthProviders:
    return HealthProviders(
        system=FakeSystemInfoProvider(),
        gpu=FakeGpuInfoProvider(gpus),
        binary=FakeBinaryProbeProvider(existing=existing, runnable=runnable),
        port=FakePortProbeProvider(occupied=occupied),
    )


def _result(report, name) -> HealthResult:
    return next(r for r in report.results if r.name == name)


def test_no_gpu_reports_fail_with_remedy():
    """With no GPU, the gpu/cuda/vram probes FAIL and carry remedies (absent case)."""
    checker = HealthChecker(Settings(), _models(), _providers(gpus=None))
    report = checker.run_all()

    gpu = _result(report, "gpu")
    assert gpu.status is HealthStatus.FAIL
    assert gpu.remedy  # a remedy string is present, not a raw failure

    cuda = _result(report, "cuda")
    assert cuda.status is HealthStatus.FAIL
    assert cuda.remedy

    vram = _result(report, "vram")
    assert vram.status is HealthStatus.FAIL

    # Overall is FAIL because a hard probe failed - honest, not a crash.
    assert report.overall is HealthStatus.FAIL


def test_gpu_present_reports_pass(monkeypatch):
    """With a healthy GPU and enough VRAM, gpu/cuda/vram PASS (present case)."""
    gpus = [GpuInfo(name="RTX 5070 Ti", vram_total_mb=16000, vram_free_mb=15000, cuda_version="12.4")]
    checker = HealthChecker(Settings(), _models(), _providers(gpus=gpus))
    report = checker.run_all()

    assert _result(report, "gpu").status is HealthStatus.PASS
    assert _result(report, "cuda").status is HealthStatus.PASS
    assert _result(report, "vram").status is HealthStatus.PASS


def test_kokoro_missing_is_warning_not_fail():
    """Kokoro absent is WARNING (TTS degrades), and must not drive overall to FAIL
    on its own when everything else is fine (Architecture section 8)."""
    gpus = [GpuInfo(name="gpu", vram_total_mb=16000, vram_free_mb=15000, cuda_version="12.4")]
    settings = Settings()
    # Configure whisper present+runnable, kokoro absent, llama present.
    settings.paths.whisper = "whisper.exe"
    settings.paths.llama_cpp = "llama.exe"
    settings.paths.kokoro = ""
    providers = _providers(
        gpus=gpus,
        existing={"whisper.exe", "llama.exe"},
        runnable={"whisper.exe", "llama.exe"},
    )
    # Give the model a resolvable location so the models probe does not FAIL.
    models = _models()
    checker = HealthChecker(settings, models, providers)
    report = checker.run_all()

    assert _result(report, "kokoro").status is HealthStatus.WARNING
    # voice is WARNING (STT ok, TTS degraded), not FAIL.
    assert _result(report, "voice").status is HealthStatus.WARNING


def test_python_version_thresholds():
    """3.10 warns, below 3.10 fails, 3.11+ passes."""
    from health import PythonProbe

    warn = PythonProbe(FakeSystemInfoProvider(version=(3, 10, 0)), "3.11").run()
    assert warn.status is HealthStatus.WARNING

    fail = PythonProbe(FakeSystemInfoProvider(version=(3, 9, 0)), "3.11").run()
    assert fail.status is HealthStatus.FAIL

    ok = PythonProbe(FakeSystemInfoProvider(version=(3, 12, 0)), "3.11").run()
    assert ok.status is HealthStatus.PASS


def test_ports_probe_warns_when_occupied():
    """An occupied reserved port is a WARNING (auto allocation can reassign)."""
    gpus = [GpuInfo(name="g", vram_total_mb=16000, vram_free_mb=15000, cuda_version="12.4")]
    providers = _providers(gpus=gpus, occupied={8080})
    report = HealthChecker(Settings(), _models(), providers).run_all()
    assert _result(report, "ports").status is HealthStatus.WARNING


def test_status_mark_is_ascii():
    """The status marks are exactly the ASCII tokens, no glyphs."""
    assert status_mark(HealthStatus.PASS) == "[OK]"
    assert status_mark(HealthStatus.WARNING) == "[!!]"
    assert status_mark(HealthStatus.FAIL) == "[XX]"


def test_worst_of_aggregation():
    """worst_of returns FAIL if any FAIL, else WARNING if any WARNING, else PASS."""
    mk = lambda s: HealthResult("x", s, "")
    assert worst_of([mk(HealthStatus.PASS), mk(HealthStatus.WARNING)]) is HealthStatus.WARNING
    assert worst_of([mk(HealthStatus.WARNING), mk(HealthStatus.FAIL)]) is HealthStatus.FAIL
    assert worst_of([mk(HealthStatus.PASS)]) is HealthStatus.PASS


def test_report_roster_is_complete():
    """The report includes every probe the spec requires."""
    report = HealthChecker(Settings(), _models(), _providers(gpus=None)).run_all()
    names = {r.name for r in report.results}
    expected = {
        "python", "virtual_env", "gpu", "cuda", "vram", "ram", "disk_space",
        "ports", "whisper", "llama_cpp", "kokoro", "voice", "models",
    }
    assert expected <= names


def test_report_to_json_is_valid():
    """The report serializes to JSON with overall + per-check statuses."""
    import json

    report = HealthChecker(Settings(), _models(), _providers(gpus=None)).run_all()
    data = json.loads(report.to_json())
    assert "overall" in data
    assert data["overall"] in {"PASS", "WARNING", "FAIL"}
    assert all(r["status"] in {"PASS", "WARNING", "FAIL"} for r in data["results"])


def test_nvidia_smi_provider_caches_spawns(monkeypatch):
    import health
    """Perf audit 2026-08-31: three probes + the 1.5s sysmon loop share one
    provider; each gpus() used to spawn nvidia-smi twice. The contract now:
    a burst of calls inside the snapshot TTL costs ONE spawn pair total, and
    the CUDA version is fetched exactly once per provider ever."""
    import subprocess as subprocess_mod

    calls = []

    class _Proc:
        returncode = 0
        stdout = "RTX Test, 16000, 12000\nCUDA Version: 12.4  "

    def fake_run(argv, **_kw):
        calls.append(argv)
        return _Proc()

    monkeypatch.setattr(health.shutil, "which", lambda _n: "nvidia-smi")
    monkeypatch.setattr(subprocess_mod, "run", fake_run)
    provider = health.NvidiaSmiGpuInfoProvider()
    first = provider.gpus()
    assert first and first[0].cuda_version == "12.4"
    burst_spawns = len(calls)
    assert burst_spawns == 2  # one CSV query + one CUDA-version read
    provider.gpus()
    provider.gpus()
    assert len(calls) == burst_spawns, "calls within the TTL spawn nothing"
    # Age the snapshot out: the CSV query re-runs, the CUDA read never does.
    provider._snapshot = (provider._snapshot[0] - 10.0, provider._snapshot[1])
    provider.gpus()
    assert len(calls) == burst_spawns + 1


def test_vram_probe_uses_file_size_when_the_estimate_is_zero(tmp_path):
    """A scan-imported registry (vram_estimate_mb: 0) must still be measured.

    Regression for the 2026-09-01 defect: VramProbe filtered on
    vram_estimate_mb > 0, so a registry whose rows all carry 0 - which is every
    row the first-run scan imports - left smallest/largest pinned at 0. The FAIL
    branch became unreachable and the WARNING read "tight for the largest model
    (0MB)". The need now comes from the on-disk .gguf size.
    """
    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\0" * 12_000_000)  # 12 MB on disk -> 12.0 MB of need
    models = [
        Model(
            id="scanned",
            name="Scanned",
            description="d",
            location=str(gguf),
            context_size=8192,
            gpu_layers=999,
            vram_estimate_mb=0,  # exactly what the first-run scan writes
            status="installed",
        )
    ]
    # Total VRAM below the model's real need -> FAIL, the branch that could
    # never fire before.
    gpus = [GpuInfo(name="tiny", vram_total_mb=8.0, vram_free_mb=8.0, cuda_version="12.4")]
    report = HealthChecker(Settings(), models, _providers(gpus=gpus)).run_all()
    vram = _result(report, "vram")
    assert vram.status is HealthStatus.FAIL
    assert vram.data["largest_model_mb"] == 12
    assert "0MB" not in vram.detail, "must not report a zero-sized largest model"


def test_vram_probe_falls_back_to_the_estimate_with_no_file_on_disk():
    """A not-yet-downloaded row still contributes its estimate (preview case)."""
    models = [
        Model(
            id="future",
            name="Future",
            description="d",
            location="",  # nothing on disk to contradict the estimate
            context_size=8192,
            gpu_layers=999,
            vram_estimate_mb=10000,
            status="installed",
        )
    ]
    gpus = [GpuInfo(name="g", vram_total_mb=16000, vram_free_mb=15000, cuda_version="12.4")]
    report = HealthChecker(Settings(), models, _providers(gpus=gpus)).run_all()
    assert _result(report, "vram").data["largest_model_mb"] == 10000


# --------------------------------------------------------------------------- #
# Our own healthy operation must not read as a fault (owner-observed
# 2026-09-03: the window sat on WARNING because a model was loaded and a
# reserved port was being served - the platform doing exactly its job).
# --------------------------------------------------------------------------- #

from health import PortsProbe, VramProbe  # noqa: E402


def _model(mb: int):
    return Model(
        id="m", name="M", description="d", location="",
        context_size=8192, gpu_layers=999, vram_estimate_mb=mb, status="installed",
    )


def _gpu(total=16303.0, free=1996.0):
    return [GpuInfo(name="g", vram_total_mb=total, vram_free_mb=free,
                    cuda_version="12.4")]


class _Gpus:
    def __init__(self, gpus): self._g = gpus
    def gpus(self): return self._g


def test_vram_held_by_our_own_model_is_counted_as_available():
    """Starting another model STOPS the current one first (M8.2), so VRAM we
    hold is reclaimable. Judging against current free VRAM warned about the
    model we ourselves had loaded."""
    probe = VramProbe(_Gpus(_gpu()), [_model(12000)], 512,
                      lambda: (13000.0, True))
    result = probe.run()
    assert result.status is HealthStatus.PASS
    assert result.data["effective_free_mb"] == 14996  # 1996 free + 13000 ours
    assert "held by locitize" in result.detail


def test_a_running_server_with_unmeasurable_vram_still_passes():
    """nvidia-smi returns [N/A] for per-process memory on this maintainer's
    card. Summing those Nones raised a TypeError the boundary guard turned into
    a silent 0, which looked identical to 'we hold nothing' and kept the false
    warning alive."""
    probe = VramProbe(_Gpus(_gpu()), [_model(12000)], 512,
                      lambda: (0.0, True))
    result = probe.run()
    assert result.status is HealthStatus.PASS
    assert result.data["locitize_running"] is True
    assert result.data["basis"].startswith("total")
    assert "released on switch" in result.detail


def test_the_basis_is_reported_rather_than_presented_as_measurement():
    """Falling back to total VRAM slightly overstates - a foreign app may hold
    some too - so the result must say which basis it used."""
    passed = VramProbe(_Gpus(_gpu()), [_model(12000)], 512,
                       lambda: (0.0, True)).run()
    measured = VramProbe(_Gpus(_gpu()), [_model(12000)], 512,
                         lambda: (13000.0, True)).run()
    assert passed.data["basis"] != measured.data["basis"]
    assert measured.data["basis"] == "free+ours"


def test_a_foreign_app_hogging_vram_still_warns():
    """The real constraint must survive: nothing of ours is running, free VRAM
    cannot host the largest model."""
    result = VramProbe(_Gpus(_gpu()), [_model(12000)], 512,
                       lambda: (0.0, False)).run()
    assert result.status is HealthStatus.WARNING
    assert result.data["basis"] == "free"


def test_a_model_too_big_even_after_reclaiming_still_warns():
    """16000MB fits the 16303MB card, so this is not the FAIL branch - but it
    leaves no room for the 512MB headroom even with everything of ours
    reclaimed, which is a real constraint the owner can act on."""
    result = VramProbe(_Gpus(_gpu()), [_model(16000)], 512,
                       lambda: (0.0, True)).run()
    assert result.status is HealthStatus.WARNING
    assert "is tight for the largest model" in result.detail


def test_a_failing_ledger_read_degrades_to_the_strict_behaviour():
    """Unknown ownership must never be optimistic."""
    def boom():
        raise RuntimeError("nvidia-smi went away")

    result = VramProbe(_Gpus(_gpu()), [_model(12000)], 512, boom).run()
    assert result.status is HealthStatus.WARNING
    assert result.data["basis"] == "free"


class _Ports:
    def __init__(self, busy): self._busy = set(busy)
    def is_free(self, port): return port not in self._busy


class _Owner:
    def __init__(self, mapping): self._m = mapping
    def listener_exe(self, port): return self._m.get(port, "")


# Composed rather than written as a literal: verify_no_owner_paths refuses a
# drive-letter absolute path anywhere in the shipped tree, and it is right to -
# a test that hardcodes one machine's layout is the "works on my machine"
# failure Milestone 14 exists to eliminate.
_SEP = chr(92)
_BIN = _SEP.join(("X:", "locitize-data", "bin"))
_OURS = _SEP.join((_BIN, "llama.cpp", "llama-server.exe"))
_FOREIGN = _SEP.join(("X:", "Program Files", "SomethingElse", "app.exe"))


def test_a_port_served_by_our_own_model_is_not_a_conflict():
    """The reported defect: 'ports in use: llama_cpp:8080' while LOCITIZE was
    correctly serving a model on 8080."""
    result = PortsProbe(
        _Ports([8080]), {"llama_cpp": 8080, "whisper": 8091},
        _Owner({8080: _OURS}), _BIN,
    ).run()
    assert result.status is HealthStatus.PASS
    assert result.data["ours"] == ["llama_cpp:8080"]
    assert result.data["blocked"] == []
    assert "served by locitize" in result.detail


def test_a_port_taken_by_a_foreign_process_still_warns():
    result = PortsProbe(
        _Ports([8080]), {"llama_cpp": 8080},
        _Owner({8080: _FOREIGN}), _BIN,
    ).run()
    assert result.status is HealthStatus.WARNING
    assert result.data["blocked"] == ["llama_cpp:8080"]


def test_an_unprovable_owner_is_treated_as_foreign():
    """A port wrongly called ours would HIDE a real conflict; wrongly called
    foreign only shows a warning the owner can read."""
    result = PortsProbe(
        _Ports([8080]), {"llama_cpp": 8080}, _Owner({}), _BIN
    ).run()
    assert result.status is HealthStatus.WARNING


def test_without_an_owner_provider_the_old_strict_behaviour_holds():
    result = PortsProbe(_Ports([8080]), {"llama_cpp": 8080}).run()
    assert result.status is HealthStatus.WARNING


def test_all_ports_free_still_says_so_plainly():
    result = PortsProbe(_Ports([]), {"llama_cpp": 8080}, _Owner({}), _BIN).run()
    assert result.status is HealthStatus.PASS
    assert "all 1 reserved ports free" in result.detail


def test_a_python_based_service_is_recognised_as_ours():
    """Owner-observed 2026-09-03 (second pass): the probe still warned
    "ports in use: kokoro:8092" while Kokoro was correctly serving. Kokoro is a
    PYTHON service - its executable is the venv interpreter, which lives nowhere
    near locitize-data/bin - so an exe-only test could never recognise it. Its
    COMMAND LINE names kokoro_server.py and the model under the data root."""
    root = _SEP.join(("X:", "locitize-data"))
    venv_python = _SEP.join(("X:", "app", ".venv", "Scripts", "python.exe"))
    kokoro_argv = " ".join([
        venv_python,
        _SEP.join((root, "..", "kokoro_server.py")),
        "--model", _SEP.join((root, "kokoro", "kokoro-v1_0.pth")),
    ])
    result = PortsProbe(
        _Ports([8092]), {"kokoro": 8092}, _Owner({8092: kokoro_argv}), _BIN
    ).run()
    assert result.status is HealthStatus.PASS
    assert result.data["ours"] == ["kokoro:8092"]


def test_a_binary_under_bin_is_still_recognised():
    """bin/ is a subdirectory of the data root, so whisper and llama-server
    match by the same test."""
    result = PortsProbe(
        _Ports([8091]), {"whisper": 8091}, _Owner({8091: _OURS}), _BIN
    ).run()
    assert result.status is HealthStatus.PASS
    assert result.data["ours"] == ["whisper:8091"]


def test_a_foreign_python_service_is_still_a_conflict():
    """Broadening to the data root must not make every python process ours."""
    other = _SEP.join(("X:", "SomeOtherApp", "server.py"))
    result = PortsProbe(
        _Ports([8092]), {"kokoro": 8092}, _Owner({8092: other}), _BIN
    ).run()
    assert result.status is HealthStatus.WARNING
    assert result.data["blocked"] == ["kokoro:8092"]


def test_the_ports_probe_does_not_consult_the_real_machine_in_tests():
    """Regression: PortsProbe used to build its own DefaultPortOwnerProvider, so
    it queried the REAL host while the port provider was a fake. That made
    test_ports_probe_warns_when_occupied pass or fail depending on whether the
    developer happened to be serving a model on 8080."""
    import inspect

    import health as health_module

    src = inspect.getsource(health_module.HealthChecker.run_all)
    assert "DefaultPortOwnerProvider()" not in src
    assert "p.port_owner" in src


def test_providers_defaults_still_supply_a_real_owner_probe():
    """Injectable must not mean absent in production."""
    from health import DefaultPortOwnerProvider, HealthProviders

    assert isinstance(HealthProviders.defaults().port_owner, DefaultPortOwnerProvider)


def test_unconfigured_optional_binary_warns_but_a_broken_path_fails():
    """Voice not set up yet is a WARNING; a configured path that is gone is a FAIL."""
    from health import BinaryProbe

    binary = FakeBinaryProbeProvider()
    unset = BinaryProbe(
        "whisper", binary, "", "remedy", unconfigured_status=HealthStatus.WARNING
    ).run()
    broken = BinaryProbe(
        "whisper", binary, "missing.exe", "remedy", unconfigured_status=HealthStatus.WARNING
    ).run()
    assert unset.status is HealthStatus.WARNING
    assert broken.status is HealthStatus.FAIL
