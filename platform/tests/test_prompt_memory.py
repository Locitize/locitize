"""Prompt memory (prompt_memory.py): a model keeps the opening it has read across a model
switch or a restart. Headless: the server's /slots API is faked.

The contract:
  * a model's state file is named for the model and for every setting that shapes its
    state, and for nothing that only changes speed;
  * Locitize saves slot 0 just before a running model stops (only an idle slot holding a
    prompt worth keeping) and restores it as soon as the model is ready again;
  * no save, restore or hook failure can stop a model from starting or stopping.
"""

from __future__ import annotations

from pathlib import Path

import prompt_memory
from config import Model, ModelRegistryData, Settings
from models import ModelRegistry
from services import ManagedProcess, ServiceSpec, ServiceStatus

BASE = ["llama-server.exe", "--model", "models/a.gguf", "--host", "127.0.0.1", "--port", "8081", "--ctx-size", "32768"]


def test_the_file_name_follows_the_state_and_ignores_speed_settings():
    name = prompt_memory.file_name("gpt-oss-120b", BASE + ["--cache-type-k", "q8_0", "-t", "16"])
    assert name.startswith("gpt-oss-120b-") and name.endswith(".bin")
    faster = prompt_memory.file_name("gpt-oss-120b", BASE + ["--cache-type-k", "q8_0", "-t", "20",
                                                             "--n-cpu-moe", "29", "--port", "9000"])
    assert faster == name, "threads, expert placement and the port do not change the state"
    for changed in (BASE[:-1] + ["8192", "--cache-type-k", "q8_0"], BASE + ["--cache-type-k", "f16"],
                    BASE + ["--cache-type-k", "q8_0", "--lora", "x.gguf"]):
        assert prompt_memory.file_name("gpt-oss-120b", changed) != name
    assert prompt_memory.file_name("a b/c:d", BASE).startswith("a-b-c-d-")


def test_save_writes_an_idle_slot_holding_a_long_prompt():
    posted = []

    def post(url, body, timeout):
        posted.append((url, body))
        return {"n_saved": 9983, "timings": {"save_ms": 412.0}}
    lines = []
    get = lambda url, timeout: [{"id": 0, "is_processing": False, "n_prompt_tokens": 9967, "n_decoded": 16}]  # noqa: E731
    assert prompt_memory.save(8081, "m-1.bin", post=post, get=get, log=lines.append)
    assert posted == [("http://127.0.0.1:8081/slots/0?action=save", {"filename": "m-1.bin"})]
    assert "saved 9983 tokens" in lines[-1]


def test_save_skips_a_busy_or_short_slot_and_never_raises():
    def post(url, body, timeout):
        raise AssertionError("must not save")
    busy = lambda url, timeout: [{"is_processing": True, "n_prompt_tokens": 9000}]   # noqa: E731
    short = lambda url, timeout: [{"is_processing": False, "n_prompt_tokens": 200}]   # noqa: E731
    assert not prompt_memory.save(8081, "m.bin", post=post, get=busy)
    assert not prompt_memory.save(8081, "m.bin", post=post, get=short)

    def down(url, timeout):
        raise OSError("connection refused")
    assert not prompt_memory.save(8081, "m.bin", post=post, get=down)


def test_restore_loads_an_existing_file_and_survives_a_refusal(tmp_path: Path):
    calls = []

    def post(url, body, timeout):
        calls.append((url, body))
        return {"n_restored": 9983, "timings": {"restore_ms": 380.0}}
    assert not prompt_memory.restore(8081, "m.bin", tmp_path, post=post), "no file, no call"
    assert calls == []
    (tmp_path / "m.bin").write_bytes(b"state")
    assert prompt_memory.restore(8081, "m.bin", tmp_path, post=post)
    assert calls == [("http://127.0.0.1:8081/slots/0?action=restore", {"filename": "m.bin"})]

    def refuse(url, body, timeout):
        raise OSError("400 Bad Request")
    lines = []
    assert not prompt_memory.restore(8081, "m.bin", tmp_path, post=refuse, log=lines.append)
    assert "refused" in lines[-1]


def test_the_model_spec_carries_the_save_path_and_both_hooks(tmp_path: Path, monkeypatch):
    settings = Settings(base_dir=tmp_path)
    settings.paths.llama_cpp = "llama-server.exe"
    registry = ModelRegistry(ModelRegistryData(version=1, models=[Model(
        id="small", name="Small", description="d", location=str(tmp_path / "s.gguf"), context_size=8192,
        gpu_layers=20, status="installed", server_args=["--cache-type-k", "q8_0"])]), settings)
    spec = registry.build_start_spec("small")
    where = prompt_memory.folder(settings.data_dir)
    assert spec.command[spec.command.index("--slot-save-path") + 1] == str(where) + "\\"
    assert spec.command.index("--slot-save-path") < spec.command.index("--cache-type-k"), "owner flags still win"
    # llama-server refuses to start ("not a directory") unless the folder exists already.
    assert not where.exists()
    spec.before_start()
    assert where.is_dir()
    seen = []
    monkeypatch.setattr(prompt_memory, "save", lambda port, name, log: seen.append(("save", port, name)))
    monkeypatch.setattr(prompt_memory, "restore", lambda port, name, folder, log: seen.append(("restore", port, name)))
    spec.before_stop(8081)
    spec.on_ready(8081)
    name = prompt_memory.file_name("small", spec.command)
    assert seen == [("save", 8081, name), ("restore", 8081, name)] and where.is_dir()


class _Proc:
    pid = 4242

    def __init__(self):
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def send_signal(self, sig):
        self.alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.alive = False


def test_the_manager_restores_after_ready_and_saves_before_a_stop_even_when_hooks_fail():
    order = []
    proc = _Proc()

    def ready(port):
        order.append(("ready", port, proc.alive))

    def before(port):
        order.append(("before_stop", port, proc.alive))
        raise RuntimeError("server gone")
    def launch(*a, **k):
        order.append(("launch",))
        return proc
    spec = ServiceSpec(name="llama_cpp:m", command=["llama-server.exe", "--port", "8081"], port=8081,
                       health_path="/health", on_ready=ready, before_stop=before,
                       before_start=lambda: order.append(("before_start",)))
    manager = ManagedProcess(spec, launcher=launch, readiness=lambda s, h: True)
    assert manager.start() is ServiceStatus.RUNNING
    assert manager.stop() is ServiceStatus.STOPPED
    assert order == [("before_start",), ("launch",), ("ready", 8081, True), ("before_stop", 8081, True)], \
        "the folder exists before the launch; saved while the model still runs"
