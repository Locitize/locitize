"""Tests for the wizard's phased, parallel install loop (M18.17).

The refactor's contract, pinned here without Tk or the network:
- virtual environments run FIRST (everything else needs them);
- the model import runs LAST (it needs the platform venv's yaml);
- every pending step runs exactly once, and pips sharing one venv never
  overlap (base_pip strictly before gui_pip);
- downloads genuinely overlap (the whole point);
- an essential failure is recorded from any worker thread.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import setup_env
import setup_plan
import setup_wizard


def _step(key, kind, essential=False):
    req = SimpleNamespace(key=key, kind=kind, label=key, essential=essential)
    return SimpleNamespace(key=key, requirement=req)


def _wizard(monkeypatch, steps, fail_keys=(), slow_keys=()):
    """A SetupWizard shell (no Tk) wired to record execution order."""
    wiz = object.__new__(setup_wizard.SetupWizard)
    wiz._events = queue.Queue()
    wiz._state = {}
    wiz._machine = setup_env.Machine()
    wiz._vars = {}
    wiz._found_models = []
    wiz._unverified_ok = False

    record = {"order": [], "concurrent": 0, "max_concurrent": 0}
    lock = threading.Lock()

    def fake_run_step(step, venv_exe, say, discovered):
        with lock:
            record["concurrent"] += 1
            record["max_concurrent"] = max(record["max_concurrent"], record["concurrent"])
            record["order"].append(step.key)
        if step.key in slow_keys:
            time.sleep(0.15)
        with lock:
            record["concurrent"] -= 1
        ok = step.key not in fail_keys
        return setup_env.StepResult(step.key, ok, "done" if ok else "boom")

    wiz._run_step = fake_run_step
    # Neutralize every post-loop side effect.
    monkeypatch.setattr(setup_env, "detect_state", lambda m=None: ({}, setup_env.Machine()))
    monkeypatch.setattr(setup_env, "find_llama_server", lambda *a, **k: "")
    monkeypatch.setattr(setup_env, "create_desktop_shortcut",
                        lambda: setup_env.StepResult("shortcut", True, "", skipped=True))
    monkeypatch.setattr(setup_env, "venv_python_path", lambda *a, **k: setup_env.REPO_DIR / "no-such-venv" / "python.exe")
    plan = SimpleNamespace(pending=steps)
    return wiz, plan, record


def test_venvs_first_import_last_all_once(monkeypatch):
    steps = [
        _step("platform_venv", setup_plan.KIND_VENV),
        _step("base_pip", setup_plan.KIND_PIP),
        _step("gui_pip", setup_plan.KIND_PIP),
        _step("llama_cpp", setup_plan.KIND_BINARY),
        _step("whisper_model", setup_plan.KIND_WEIGHTS if hasattr(setup_plan, "KIND_WEIGHTS") else "weights"),
        _step("first_model", "model"),
    ]
    wiz, plan, record = _wizard(monkeypatch, steps)
    wiz._install(plan)
    order = record["order"]
    assert sorted(order) == sorted(s.key for s in steps)  # each exactly once
    assert order[0] == "platform_venv"
    assert order[-1] == "first_model"
    assert order.index("base_pip") < order.index("gui_pip")  # same-venv chain


def test_downloads_overlap(monkeypatch):
    steps = [
        _step("llama_cpp", setup_plan.KIND_BINARY),
        _step("whisper_bin", setup_plan.KIND_BINARY),
        _step("kokoro_weights", "weights"),
    ]
    wiz, plan, record = _wizard(monkeypatch, steps, slow_keys={s.key for s in steps})
    wiz._install(plan)
    assert record["max_concurrent"] >= 2, "downloads never overlapped"


def test_essential_failure_from_a_worker_is_recorded(monkeypatch):
    steps = [
        _step("llama_cpp", setup_plan.KIND_BINARY, essential=True),
        _step("whisper_bin", setup_plan.KIND_BINARY),
    ]
    wiz, plan, _record = _wizard(monkeypatch, steps, fail_keys={"llama_cpp"})
    wiz._install(plan)
    # The done event carries the essential-failure summary.
    messages = []
    while not wiz._events.empty():
        kind, payload = wiz._events.get_nowait()
        if kind == "done":
            messages.append(payload)
    assert messages and "could not finish" in messages[0]


def test_already_satisfied_steps_are_skipped(monkeypatch):
    steps = [_step("llama_cpp", setup_plan.KIND_BINARY)]
    wiz, plan, record = _wizard(monkeypatch, steps)
    wiz._state["llama_cpp"] = True
    wiz._install(plan)
    assert record["order"] == []  # never executed, only reported as present


def test_every_platform_venv_pip_runs_in_one_chain(monkeypatch):
    """The GPU Kokoro step replaces the torch the CPU step installed, and two
    pips resolving into one site-packages at once corrupt each other - so all
    five platform-venv pips are one ordered chain, whatever else overlaps."""
    steps = [
        _step("base_pip", setup_plan.KIND_PIP),
        _step("gui_pip", setup_plan.KIND_PIP),
        _step("pillow_pip", setup_plan.KIND_PIP),
        _step("kokoro_pip", setup_plan.KIND_PIP),
        _step("kokoro_pip_cuda", setup_plan.KIND_PIP),
        _step("llama_cpp", setup_plan.KIND_BINARY),
    ]
    pips = [s.key for s in steps if s.requirement.kind == setup_plan.KIND_PIP]
    wiz, plan, record = _wizard(monkeypatch, steps, slow_keys=set(pips))
    wiz._install(plan)
    order = record["order"]
    assert [k for k in order if k in pips] == pips
    # A pip and the binary download still overlap: chaining cost no wall time.
    assert record["max_concurrent"] >= 2


def _real_step_wizard(monkeypatch, wants_gpu, has_nvidia):
    wiz = object.__new__(setup_wizard.SetupWizard)
    wiz._state = {}
    wiz._machine = setup_env.Machine(has_nvidia=has_nvidia)
    wiz._vars = {"voice_out_gpu": SimpleNamespace(get=lambda: wants_gpu)}
    calls = []
    monkeypatch.setattr(
        setup_env, "pip_install",
        lambda target, packages, on_output=None: (
            calls.append(tuple(packages)) or setup_env.StepResult("pip", True, "installed")
        ),
    )
    return wiz, calls


def _real_step(key):
    return setup_plan.Step(requirement=setup_plan.REQUIREMENTS[key], wanted_by=())


def test_cpu_kokoro_pip_is_skipped_when_the_gpu_build_will_follow(monkeypatch):
    wiz, calls = _real_step_wizard(monkeypatch, wants_gpu=True, has_nvidia=True)
    result = wiz._run_step(_real_step("kokoro_pip"), setup_env.REPO_DIR / "x.exe", lambda t: None, {})
    assert result.ok and result.skipped and "GPU build" in result.message
    assert calls == []
    result = wiz._run_step(_real_step("kokoro_pip_cuda"), setup_env.REPO_DIR / "x.exe", lambda t: None, {})
    assert result.ok and calls == [setup_env.PIP_SETS["kokoro_pip_cuda"]]


def test_cpu_kokoro_pip_still_installs_without_the_gpu_feature(monkeypatch):
    wiz, calls = _real_step_wizard(monkeypatch, wants_gpu=False, has_nvidia=True)
    result = wiz._run_step(_real_step("kokoro_pip"), setup_env.REPO_DIR / "x.exe", lambda t: None, {})
    assert result.ok and not result.skipped
    assert calls == [setup_env.PIP_SETS["kokoro_pip"]]


def test_gpu_kokoro_refuses_honestly_without_an_nvidia_card(monkeypatch):
    """No card: the CPU engine goes in as usual and the GPU step fails with a
    reason, rather than downloading 2.8 GB of CUDA wheels that cannot run."""
    wiz, calls = _real_step_wizard(monkeypatch, wants_gpu=True, has_nvidia=False)
    cpu = wiz._run_step(_real_step("kokoro_pip"), setup_env.REPO_DIR / "x.exe", lambda t: None, {})
    assert cpu.ok and not cpu.skipped
    gpu = wiz._run_step(_real_step("kokoro_pip_cuda"), setup_env.REPO_DIR / "x.exe", lambda t: None, {})
    assert not gpu.ok and "no NVIDIA card" in gpu.message
    assert calls == [setup_env.PIP_SETS["kokoro_pip"]]
