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
    _state_file(tmp_path / "m.bin", 10, 100)
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

    class FakeWatcher:
        def __init__(self, port, name_fn, **kw):
            seen.append(("watch", port, name_fn()))

        def start(self):
            return self

        def stop(self):
            seen.append(("unwatch",))
    monkeypatch.setattr(prompt_memory, "Watcher", FakeWatcher)
    monkeypatch.setattr(prompt_memory, "save", lambda port, name, **kw: seen.append(("save", port, name)))
    monkeypatch.setattr(prompt_memory, "restore", lambda port, name, folder, **kw: seen.append(("restore", port, name)))
    spec.on_ready(8081)
    assert prompt_memory.for_port(8081) is not None, "the router can find the running model's slot"
    spec.before_stop(8081)
    assert prompt_memory.for_port(8081) is None
    name = prompt_memory.file_name("small", spec.command)
    # No app named yet: the model-wide file, as before; the watcher has nothing to name.
    assert seen == [("restore", 8081, name), ("watch", 8081, None), ("unwatch",), ("save", 8081, name)]
    assert where.is_dir()
    # The server's own output is kept (it went to the null device until 2026-11-06).
    assert spec.log_path and spec.log_path.endswith("llama-server.log") and spec.append_log


def _metrics(read: int) -> str:
    return f"# HELP llamacpp:prompt_tokens_total x\nllamacpp:prompt_tokens_total {read}\nllamacpp:n_decode_total 3\n"


def test_the_watcher_saves_after_a_long_new_read_once_the_model_is_idle():
    state = {"read": 0, "busy": False}
    saves = []
    now = [1000.0]

    def saver(port, name, where, post, get, log):
        saves.append((port, name, state["read"]))
        return True
    watcher = prompt_memory.Watcher(8081, lambda: "m.bin", get=lambda url, t: [{"is_processing": state["busy"]}],
                                    get_text=lambda url, t: _metrics(state["read"]), clock=lambda: now[0],
                                    saver=saver)
    assert not watcher.check() and watcher.baseline == 0, "the first look only sets the baseline"
    state.update(read=10046, busy=True)            # the portal's opening is being read
    assert not watcher.check()
    state["busy"] = False
    assert not watcher.check(), "idle once is not enough: a reply may follow at once"
    assert watcher.check() and saves == [(8081, "m.bin", 10046)]
    # More reading soon after: held back for WATCH_MIN_GAP_S, then saved.
    state["read"] = 10046 + 4000
    watcher.check(), watcher.check()
    assert len(saves) == 1
    now[0] += prompt_memory.WATCH_MIN_GAP_S
    assert watcher.check() and saves[-1][2] == 14046
    # A little new reading is not worth a 1 GB file.
    state["read"] += 300
    now[0] += prompt_memory.WATCH_MIN_GAP_S
    watcher.check(), watcher.check()
    assert len(saves) == 2


def test_the_watcher_skips_a_check_it_cannot_read_and_follows_a_restarted_counter():
    def down(url, t):
        raise OSError("refused")
    watcher = prompt_memory.Watcher(8081, lambda: "m.bin", get=down, get_text=down,
                                    saver=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no save")))
    assert not watcher.check() and watcher.baseline is None
    counts = iter([9000, 50])
    watcher = prompt_memory.Watcher(8081, lambda: "m.bin", get=lambda url, t: [{"is_processing": False}],
                                    get_text=lambda url, t: _metrics(next(counts)), saver=lambda *a, **k: True)
    watcher.check(), watcher.check()
    assert watcher.baseline == 50, "a server restarted under the watcher starts a new count"
    assert prompt_memory.prompt_tokens_read("nothing here") is None


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

    def ready(port, pid=None):
        order.append(("ready", port, proc.alive, pid))   # the pid lets the model's VRAM be measured

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
    assert order == [("before_start",), ("launch",), ("ready", 8081, True, 4242), ("before_stop", 8081, True)], \
        "the folder exists before the launch; saved while the model still runs"


# --- 2026-11-06: Flash-Next's saved state was refused twice ("HTTP Error 400") ---------------
# A save cut short (the model or Locitize stopped while llama-server was still writing) left a
# file whose header promised more than it held; llama-server refused it at once.

def _state_file(path: Path, n_tokens: int, payload: int) -> None:
    head = (prompt_memory.STATE_MAGIC.to_bytes(4, "little") + (4).to_bytes(4, "little")
            + n_tokens.to_bytes(4, "little"))
    path.write_bytes(head + b"\0" * (4 * n_tokens + payload))


def test_a_save_goes_to_a_part_file_and_only_a_finished_one_replaces_the_last_good(tmp_path: Path):
    _state_file(tmp_path / "m.bin", 10, 100)
    good = (tmp_path / "m.bin").read_bytes()
    get = lambda url, timeout: [{"is_processing": False, "n_prompt_tokens": 5000}]   # noqa: E731

    def server_writes(url, body, timeout):
        assert body == {"filename": "m.bin.part"}, "the server writes beside the real file"
        _state_file(tmp_path / body["filename"], 20, 100)
        return {"n_saved": 20}
    assert prompt_memory.save(8081, "m.bin", where=tmp_path, post=server_writes, get=get)
    assert not (tmp_path / "m.bin.part").exists() and prompt_memory.file_is_whole(tmp_path / "m.bin")
    assert (tmp_path / "m.bin").read_bytes() != good

    # Stopped part-way: the server dies mid-write. The last good file is untouched.
    good = (tmp_path / "m.bin").read_bytes()

    def killed(url, body, timeout):
        (tmp_path / body["filename"]).write_bytes(b"ggsq-half")
        raise OSError("connection reset")
    lines = []
    assert not prompt_memory.save(8081, "m.bin", where=tmp_path, post=killed, get=get, log=lines.append)
    assert (tmp_path / "m.bin").read_bytes() == good and "save failed" in lines[-1]


def test_restore_sets_a_cut_short_file_aside_without_asking_the_server(tmp_path: Path):
    _state_file(tmp_path / "m.bin", 3809, 0)
    with open(tmp_path / "m.bin", "r+b") as handle:   # cut it inside the token list
        handle.truncate(12 + 4 * 1000)
    assert not prompt_memory.file_is_whole(tmp_path / "m.bin")

    def post(url, body, timeout):
        raise AssertionError("a broken file is never sent")
    lines = []
    assert not prompt_memory.restore(8081, "m.bin", tmp_path, post=post, log=lines.append)
    assert not (tmp_path / "m.bin").exists() and (tmp_path / "m.bin.broken").is_file()
    assert "cut short" in lines[-1]
    _state_file(tmp_path / "w.bin", 3809, 5000)
    assert prompt_memory.file_is_whole(tmp_path / "w.bin")
    (tmp_path / "x.bin").write_bytes(b"not a state file at all")
    assert not prompt_memory.file_is_whole(tmp_path / "x.bin")


def test_a_refusal_is_logged_in_the_servers_own_words(tmp_path: Path):
    import io
    import json as _json
    import urllib.error
    _state_file(tmp_path / "m.bin", 10, 100)
    body = _json.dumps({"error": {"code": 400, "message": "Unable to restore slot: No available space in KV cache",
                                  "type": "invalid_request_error"}}).encode()

    def refuse(url, data, timeout):
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(body))
    lines = []
    assert not prompt_memory.restore(8081, "m.bin", tmp_path, post=refuse, log=lines.append)
    assert "HTTP 400: Unable to restore slot: No available space in KV cache" in lines[-1]


def test_stopping_the_watcher_waits_for_a_save_under_way():
    import threading
    import time as _time
    started, finished = threading.Event(), threading.Event()

    def slow_saver(port, name, where, post, get, log):
        started.set()
        _time.sleep(0.3)
        finished.set()
        return True
    reads = iter(range(0, 10**7, 5000))
    watcher = prompt_memory.Watcher(8081, lambda: "m.bin", get=lambda url, t: [{"is_processing": False}],
                                    get_text=lambda url, t: _metrics(next(reads)), saver=slow_saver,
                                    interval=0.01)
    watcher.start()
    assert started.wait(5), "the watcher began a save"
    watcher.stop()
    assert finished.is_set(), "stop() returned only after the save finished"


def test_the_server_log_rotates_once_it_is_too_big(tmp_path: Path, monkeypatch):
    import models
    log = tmp_path / "logs" / "llama-server.log"
    models._keep_server_log(log)
    assert log.parent.is_dir()
    log.write_bytes(b"x" * 10)
    monkeypatch.setattr(models, "SERVER_LOG_MAX_BYTES", 5)
    models._keep_server_log(log)
    assert not log.exists() and (tmp_path / "logs" / "llama-server.log.1").read_bytes() == b"x" * 10
    models._keep_server_log(None)


# --- One saved opening per app (2026-11-06) ---------------------------------------------------
# The portal and Open WebUI both used Flash-Next; the model's one slot held only the last app's
# opening, so the other app's first chat after a switch or restart re-read ~4,000 tokens (~60 s).

def _chat(system: str, tools: int = 0, user: str = "hi") -> bytes:
    import json as _json
    body = {"model": "m", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": f"tool_{i}", "parameters": {}}} for i in range(tools)]
    return _json.dumps(body).encode()


PORTAL = "You are the Agent Portal. " + "Rules and tools. " * 400
OWUI = "You are a helpful assistant in Open WebUI. " + "Knowledge. " * 500


def test_each_app_is_told_apart_by_the_start_of_its_opening():
    key = prompt_memory.opening_key
    portal_today = key(_chat(PORTAL + "\n\nToday: Friday 06 November 2026", tools=5))
    portal_tomorrow = key(_chat(PORTAL + "\n\nToday: Saturday 07 November 2026", tools=5, user="other"))
    assert portal_today and portal_today == portal_tomorrow, "the date at the end does not make a new app"
    assert key(_chat(OWUI)) not in (None, portal_today)
    assert key(_chat(PORTAL, tools=6)) != key(_chat(PORTAL, tools=5)), "a different tool set is another opening"
    assert key(_chat("Generate a concise title for this chat.")) is None, "short requests never swap"
    assert key(b"not json") is None and key(b'{"prompt": "x"}') is None
    parts = {"model": "m", "messages": [{"role": "system", "content": [{"type": "text", "text": OWUI}]}]}
    import json as _json
    assert key(_json.dumps(parts).encode()) == key(_chat(OWUI))


class _FakeSlotServer:
    """llama-server's slot endpoints over one slot: save writes what the slot holds, restore
    loads it back. `slot` is the app whose opening the slot holds now."""

    def __init__(self, where: Path):
        self.where, self.slot, self.calls = where, None, []

    def post(self, url, body, timeout):
        action = url.split("action=")[1]
        self.calls.append((action, body["filename"]))
        path = self.where / body["filename"]
        if action == "save":
            _state_file(path, 100, 50)
            path.write_bytes(path.read_bytes() + (self.slot or "-").encode())
            return {"n_saved": 100}
        self.slot = path.read_bytes()[12 + 400 + 50:].decode() or None
        return {"n_restored": 100}

    def get(self, url, timeout):
        return [{"is_processing": False, "n_prompt_tokens": 5000}]


def test_the_slot_swaps_to_each_apps_own_opening_and_keeps_what_the_other_read(tmp_path: Path):
    server = _FakeSlotServer(tmp_path)
    lines = []
    slots = prompt_memory.PromptSlots(8081, "m-abc.bin", tmp_path, post=server.post, get=server.get,
                                      log=lines.append)
    slots.recall()                                       # first start: nothing saved
    assert slots.key is None
    portal, owui = "11111111", "22222222"

    slots.before_request(portal)                         # the portal's first chat reads its opening once
    assert "read this once" in lines[-1]
    server.slot = portal
    slots.before_request(portal)                         # its next chat: nothing to do
    assert [c for c in server.calls if c[0] == "save"] == []

    slots.before_request(owui)                           # Open WebUI: the portal's opening is kept
    assert server.calls[-1] == ("save", "m-abc-11111111.bin.part")
    assert (tmp_path / "m-abc-11111111.bin").is_file()
    server.slot = owui

    slots.before_request(portal)                         # back to the portal: Open WebUI's kept, the portal's restored
    assert ("save", "m-abc-22222222.bin.part") in server.calls
    assert server.calls[-1] == ("restore", "m-abc-11111111.bin") and server.slot == portal

    slots.before_request(None)                           # a title request overwrites the slot
    assert slots.key is None and server.calls[-1] == ("save", "m-abc-11111111.bin.part")
    slots.before_request(portal)
    assert server.calls[-1] == ("restore", "m-abc-11111111.bin") and server.slot == portal

    slots.remember()                                     # the model stops: the portal is the last app
    assert (tmp_path / "m-abc.last").read_text() == portal

    server.slot = None                                   # a restart: the portal's opening comes back
    again = prompt_memory.PromptSlots(8081, "m-abc.bin", tmp_path, post=server.post, get=server.get)
    again.recall()
    assert again.key == portal and server.slot == portal


def test_no_more_than_three_saved_openings_are_kept_per_model(tmp_path: Path):
    import os as _os
    import time as _time
    server = _FakeSlotServer(tmp_path)
    slots = prompt_memory.PromptSlots(8081, "m-abc.bin", tmp_path, post=server.post, get=server.get)
    keys = ["aaaaaaa1", "aaaaaaa2", "aaaaaaa3", "aaaaaaa4", "aaaaaaa5"]
    for i, key in enumerate(keys):
        slots.before_request(key)
        server.slot = key
        slots.dirty = True
        slots.before_request(None)                       # forces a save of `key`
        _os.utime(tmp_path / f"m-abc-{key}.bin", (1000 + i, 1000 + i))
    _time.sleep(0)
    kept = sorted(f.name for f in tmp_path.glob("m-abc-*.bin"))
    assert len(kept) == prompt_memory.MAX_SAVED_OPENINGS
    assert "m-abc-aaaaaaa5.bin" in kept and "m-abc-aaaaaaa1.bin" not in kept
    assert (tmp_path / "m-abc.bin").exists() is False, "the model-wide file is not an app file"


def test_the_watcher_names_nothing_while_the_slot_holds_no_known_app():
    saves = []
    reads = iter([0, 9000, 9000, 9000])
    watcher = prompt_memory.Watcher(8081, lambda: None, get=lambda url, t: [{"is_processing": False}],
                                    get_text=lambda url, t: _metrics(next(reads)),
                                    saver=lambda *a, **k: saves.append(a) or True)
    for _ in range(4):
        watcher.check()
    assert saves == []


def test_the_router_swaps_the_apps_opening_in_before_it_forwards_a_chat(tmp_path: Path):
    """End to end over HTTP: the real router, a fake llama-server with slot endpoints."""
    import json as _json
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from config import Model as _Model
    from router import ModelRouter

    order = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self._send([{"id": 0, "is_processing": False, "n_prompt_tokens": 5000}])

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path.startswith("/slots/0"):
                action = self.path.split("action=")[1]
                name = _json.loads(body)["filename"]
                order.append((action, name))
                if action == "save":
                    _state_file(tmp_path / name, 10, 10)
                    self._send({"n_saved": 10})
                else:
                    self._send({"n_restored": 10})
                return
            order.append(("chat", prompt_memory.opening_key(body)))
            self._send({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

        def _send(self, payload):
            data = _json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    up_port = upstream.server_address[1]
    row = _Model(id="m", name="M", description="d", location="x.gguf", context_size=8192, gpu_layers=1,
                 status="installed")
    router = ModelRouter(port=0, registry_fn=lambda: [row], running_model_id_fn=lambda: "m",
                         port_provider=lambda: up_port, switch_fn=lambda model_id: (True, "RUNNING"))
    router.start()
    slots = prompt_memory.PromptSlots(up_port, "m-abc.bin", tmp_path)
    prompt_memory.register(up_port, slots)
    try:
        def chat(body):
            bound = router._server.server_address[1]
            req = urllib.request.Request(f"http://127.0.0.1:{bound}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return _json.load(r)
        portal, owui = _chat(PORTAL, tools=5), _chat(OWUI)
        assert chat(portal)["choices"][0]["message"]["content"]
        chat(owui)
        chat(portal)
        pk = prompt_memory.opening_key(portal)
        names = [entry for entry in order if entry[0] != "chat"]
        assert names[0] == ("save", f"m-abc-{pk}.bin.part"), names
        assert ("restore", f"m-abc-{pk}.bin") in names, names
        assert order[-1][0] == "chat" and order[-2] == ("restore", f"m-abc-{pk}.bin"), \
            "the portal's opening is restored before its chat is forwarded"
    finally:
        prompt_memory.unregister(up_port)
        router.stop()
        upstream.shutdown()
