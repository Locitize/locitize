"""Prompt memory: a model keeps what it has already read across a switch or a restart.

Why
---
A chat client sends the same long opening on every chat - the Agent Portal's system
text and tool list are about 10,000 tokens. llama-server reuses an opening it has
already read, but only while the model stays loaded: after a model switch or a
restart the first chat read it all again, about two minutes on Qwen3.8-Flash-Next
(85 tok/s reading a prompt, measured 2026-11-05) before the first word.

What it does
------------
llama-server can save a slot's state to a file and restore it (`--slot-save-path`,
`POST /slots/0?action=save|restore`); the WiFiBlu build also writes the context
checkpoints a hybrid model needs to resume part-way. Before Locitize stops a model it
saves slot 0 (when the slot holds a prompt worth keeping and is not busy); right after
the model is ready again it restores the file. The next chat then reads only what is
new, the shared opening coming from the file.

The file is named for the model and for the settings that shape its state (the model
file, context size, KV cache types, slots, adapters), so a changed setting starts
afresh instead of restoring a state that no longer fits. A restore that the server
refuses is logged and the file left for the next save to replace; nothing here can
stop a model from starting or stopping.

Saving only at a stop lost what a force-closed app had read, so while a model runs a
Watcher also saves once the server has read WATCH_MIN_NEW_TOKENS new prompt tokens and
sits idle (the first time after each load at once, then at most every WATCH_MIN_GAP_S).

The saved state holds the slot's last prompt, so it stays on this machine, in the data
folder, beside the chat history that holds the same text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

FOLDER_NAME = "prompt-memory"
SAVE_PATH_FLAG = "--slot-save-path"
# A prompt shorter than this is read again in a second or two: not worth a file.
MIN_TOKENS = 1024
# Flags whose values decide what a saved state means; anything else (threads, expert
# placement, batch sizes) changes speed, never the state.
STATE_FLAGS = ("--model", "-m", "--ctx-size", "-c", "--cache-type-k", "-ctk", "--cache-type-v", "-ctv",
               "--parallel", "-np", "--lora", "--lora-scaled", "--swa-full", "--kv-unified", "--no-kv-unified")
SAVE_TIMEOUT_S = 120.0
RESTORE_TIMEOUT_S = 120.0
# llama.cpp's sequence-state file starts with this magic ('ggsq'), then a version and the
# token count (LLAMA_STATE_SEQ_MAGIC in llama.h).
STATE_MAGIC = 0x67677371
# A save is written under this suffix and renamed when finished; a file found cut short at
# restore is set aside under the other.
PART_SUFFIX = ".part"
BROKEN_SUFFIX = ".broken"
# Saving only at a stop lost everything a force-closed Locitize had read. While a model
# runs, a watcher also saves once the server has read this many NEW prompt tokens (its own
# llamacpp:prompt_tokens_total, which leaves out what was restored or reused) and has been
# idle for two checks in a row...
WATCH_INTERVAL_S = 15.0
WATCH_MIN_NEW_TOKENS = 2048
# ... but after the first save of a load, no more often than this: a file is ~1 GB.
WATCH_MIN_GAP_S = 600.0


def folder(data_dir: str | Path) -> Path:
    return Path(data_dir) / FOLDER_NAME


def file_name(model_id: str, command: list[str]) -> str:
    """`<model id>-<10 hex>.bin`: the hex covers every STATE_FLAGS value in the argv."""
    picked = []
    for i, arg in enumerate(command):
        if arg in STATE_FLAGS:
            value = command[i + 1] if i + 1 < len(command) and not command[i + 1].startswith("-") else ""
            picked.append((arg, value))
    digest = hashlib.sha1(json.dumps(picked).encode("utf-8")).hexdigest()[:10]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-") or "model"
    return f"{safe}-{digest}.bin"


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 - loopback only
        return json.load(r)


def _get(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 - loopback only
        return json.load(r)


def _reason(exc: BaseException) -> str:
    """The server's own words for a refusal (llama-server answers {"error": {"message": ...}}),
    else the exception text. Before 2026-11-06 only "HTTP Error 400" was logged, which hid why."""
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace")
            error = json.loads(body).get("error")
            message = error.get("message") if isinstance(error, dict) else error
            return f"HTTP {exc.code}: {message or body[:300]}"
        except Exception:   # noqa: BLE001 - the reason is a nicety, never a new failure
            return f"HTTP {exc.code}"
    return str(exc)


def file_is_whole(path: Path) -> bool:
    """True when a saved state's header fits the file: the 'ggsq' magic, then a token count
    whose tokens the file really holds. A save cut short (the model or Locitize stopped while
    llama-server was still writing) fails this, and llama-server refuses such a file at once
    (Flash-Next, 2026-11-06 05:29 and 08:08: refused in ~0.1 s instead of the 3 s a restore takes)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            head = handle.read(12)
    except OSError:
        return False
    if len(head) < 12:
        return False
    magic, _version, n_tokens = (int.from_bytes(head[i:i + 4], "little") for i in (0, 4, 8))
    return magic == STATE_MAGIC and 12 + 4 * n_tokens < size


def save(port: int, name: str, *, where: Path | None = None, post: Callable[..., dict] = _post,
         get: Callable[..., Any] = _get, log: Callable[[str], None] = lambda text: None) -> bool:
    """Save slot 0 of the server on `port` as `name`, when it is idle and holds at least
    MIN_TOKENS of prompt. True when a file was written.

    With `where` (the --slot-save-path folder) the server writes `<name>.part` and only a
    finished file replaces `name`, so a save cut short can never spoil the last good one."""
    base = f"http://127.0.0.1:{port}"
    target = name + PART_SUFFIX if where is not None else name
    try:
        slots = get(base + "/slots", 5.0)
        slot = slots[0] if isinstance(slots, list) and slots else {}
        if slot.get("is_processing"):
            log("prompt memory: the model is busy, nothing saved")
            return False
        held = int(slot.get("n_prompt_tokens") or 0) + int(slot.get("n_decoded") or 0)
        if held < MIN_TOKENS:
            return False
        result = post(base + "/slots/0?action=save", {"filename": target}, SAVE_TIMEOUT_S)
        if where is not None:
            os.replace(where / target, where / name)
    except Exception as exc:   # noqa: BLE001 - a failed save must never hold up a stop
        log(f"prompt memory: save failed ({_reason(exc)})")
        return False
    saved = int(result.get("n_saved") or 0)
    ms = (result.get("timings") or {}).get("save_ms")
    log(f"prompt memory: saved {saved} tokens as {name}" + (f" in {ms:.0f} ms" if isinstance(ms, (int, float)) else ""))
    return saved > 0


def _get_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 - loopback only
        return r.read().decode("utf-8", "replace")


def prompt_tokens_read(metrics_text: str) -> int | None:
    """llama-server's count of prompt tokens it has actually read (Prometheus text)."""
    for line in metrics_text.splitlines():
        if line.startswith("llamacpp:prompt_tokens_total"):
            try:
                return int(float(line.split()[-1]))
            except ValueError:
                return None
    return None


def worth_saving(read_since_save: int, idle_checks: int, saved_this_load: bool, seconds_since_save: float) -> bool:
    """The watcher's rule: enough new reading, the model idle twice running, and (after the
    first save of this load) WATCH_MIN_GAP_S since the last one."""
    return (read_since_save >= WATCH_MIN_NEW_TOKENS and idle_checks >= 2
            and (not saved_this_load or seconds_since_save >= WATCH_MIN_GAP_S))


class Watcher:
    """Saves a running model's prompt memory as it goes (see worth_saving), on a daemon
    thread; stop() ends it. Every failure is a skipped check, never an error."""

    def __init__(self, port: int, name_fn: Callable[[], str | None], *, where: Path | None = None, get=_get,
                 get_text=_get_text, post=_post, log: Callable[[str], None] = lambda text: None,
                 interval: float = WATCH_INTERVAL_S, clock: Callable[[], float] = time.monotonic,
                 saver: Callable[..., bool] | None = None, guard: Any = None):
        self._port, self._name_fn, self._get, self._get_text, self._post = port, name_fn, get, get_text, post
        self._where = where
        # PromptSlots' lock: a save must not interleave with an app swap, or one app's state
        # could be written under another's name.
        self._guard = guard if guard is not None else threading.Lock()
        self._log, self._interval, self._clock = log, interval, clock
        self._save = saver or save
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline: int | None = None
        self.idle_checks = 0
        self.saved_this_load = False
        self.last_save = clock()

    def check(self) -> bool:
        """One look at the server; True when it saved."""
        base = f"http://127.0.0.1:{self._port}"
        try:
            read = prompt_tokens_read(self._get_text(base + "/metrics", 5.0))
            slots = self._get(base + "/slots", 5.0)
        except Exception:   # noqa: BLE001 - the server is loading, busy or gone: look again later
            return False
        if read is None:
            return False
        if self.baseline is None or read < self.baseline:   # first look, or a restarted counter
            self.baseline, self.idle_checks = read, 0
            return False
        busy = bool(slots[0].get("is_processing")) if isinstance(slots, list) and slots else True
        self.idle_checks = 0 if busy else self.idle_checks + 1
        if not worth_saving(read - self.baseline, self.idle_checks, self.saved_this_load,
                            self._clock() - self.last_save):
            return False
        with self._guard:
            name = self._name_fn()
            if not name:   # the slot holds no known app's opening: nothing worth naming
                return False
            saved = self._save(self._port, name, where=self._where, post=self._post, get=self._get, log=self._log)
        if saved:
            self.baseline, self.saved_this_load, self.last_save = read, True, self._clock()
            return True
        return False

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.check()

    def start(self) -> "Watcher":
        self._thread = threading.Thread(target=self._run, name=f"prompt-memory:{self._port}", daemon=True)
        self._thread.start()
        return self

    def stop(self, wait: float = SAVE_TIMEOUT_S) -> None:
        """End the watcher, waiting up to `wait` seconds for a save it has under way: the
        model is stopped right after this, and a server killed mid-save leaves a broken file."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(wait)


def restore(port: int, name: str, where: Path, *, post: Callable[..., dict] = _post,
            log: Callable[[str], None] = lambda text: None) -> bool:
    """Restore `name` into slot 0 of the server on `port` if the file exists in `where`.
    A file cut short is set aside as `<name>.broken` (the newest only) instead of sent."""
    path = where / name
    if not path.is_file():
        return False
    if not file_is_whole(path):
        try:
            os.replace(path, where / (name + BROKEN_SUFFIX))
        except OSError:
            pass
        log(f"prompt memory: {name} was cut short (a save stopped part-way) and is set aside as "
            f"{name}{BROKEN_SUFFIX}; the opening is read afresh and saved again")
        return False
    try:
        result = post(f"http://127.0.0.1:{port}/slots/0?action=restore", {"filename": name}, RESTORE_TIMEOUT_S)
    except Exception as exc:   # noqa: BLE001 - the model still serves, just without the head start
        log(f"prompt memory: restore of {name} refused ({_reason(exc)}); it is read afresh and saved again")
        return False
    restored = int(result.get("n_restored") or 0)
    ms = (result.get("timings") or {}).get("restore_ms")
    log(f"prompt memory: restored {restored} tokens from {name}"
        + (f" in {ms:.0f} ms" if isinstance(ms, (int, float)) else ""))
    return restored > 0


# --- One saved opening per app (2026-11-06) ------------------------------------------------------
# llama-server here runs one slot, so it holds one app's opening at a time. When the Agent Portal
# and, say, Open WebUI both use Flash-Next, each one's first chat after the other re-read its whole
# opening (~4,000 tokens, ~60 s on a model streamed from the SSD). Now each app's opening is kept in
# its own file and swapped in by the router before a chat is forwarded: save the slot (~0.5 s),
# restore the other app's (~1 s). An app is told apart by the start of its system text and its
# tool names (opening_key); short requests (titles, tags) carry no key and never cause a swap.

# The part of the system text that names an app. The Agent Portal keeps its opening word-for-word
# the same all day except the date at the very end, far past this many characters.
OPENING_CHARS = 4000
# An opening shorter than this (system text plus tool schemas, ~1,000 tokens) is re-read in a
# second or two: not worth a swap.
MIN_OPENING_CHARS = 4000
# Saved openings kept per model and settings; the least recently used beyond this are removed
# (each file is 0.5-3.5 GB).
MAX_SAVED_OPENINGS = 3
LAST_SUFFIX = ".last"
_KEY_RE = re.compile(r"^[0-9a-f]{8}$")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def opening_key(body: bytes) -> str | None:
    """8 hex naming the app a chat request comes from, or None for a request too short to matter
    (or not a chat). Built from the first system/developer message's start and the tool names."""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return None
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list) or not messages:
        return None
    first = messages[0] if isinstance(messages[0], dict) else {}
    text = _text_of(first.get("content")) if first.get("role") in ("system", "developer") else ""
    tools = data.get("tools") if isinstance(data.get("tools"), list) else []
    names = [str((tool.get("function") or tool).get("name", "")) for tool in tools if isinstance(tool, dict)]
    if len(text) + len(json.dumps(tools)) < MIN_OPENING_CHARS:
        return None
    return hashlib.sha1((text[:OPENING_CHARS] + "\0" + "\n".join(names)).encode("utf-8")).hexdigest()[:8]


class PromptSlots:
    """The one slot of a running model server, and which app's opening it holds.

    `base` is the model's file_name() (model and state settings); each app's opening is saved as
    `<base stem>-<key>.bin`, and `<base stem>.last` names the app the slot held at the last stop,
    so a restart restores that one. An unknown slot (key None: a keyless chat ran last) is saved
    under `base` itself, as before 2026-11-06."""

    def __init__(self, port: int, base: str, where: Path, *, post: Callable[..., dict] = _post,
                 get: Callable[..., Any] = _get, log: Callable[[str], None] = lambda text: None,
                 max_files: int = MAX_SAVED_OPENINGS):
        self.port, self.base, self.where = port, base, Path(where)
        self._post, self._get, self._log, self._max = post, get, log, max_files
        self.stem = base[:-4] if base.endswith(".bin") else base
        self.key: str | None = None    # whose opening slot 0 holds; None = unknown
        self.dirty = False             # the slot has read or written more since the last save/restore
        self.lock = threading.RLock()

    def name(self, key: str | None) -> str:
        return f"{self.stem}-{key}.bin" if key else self.base

    def current_name(self) -> str | None:
        """The file the watcher should save the slot under right now (None while unknown)."""
        return self.name(self.key) if self.key else None

    def _last_path(self) -> Path:
        return self.where / (self.stem + LAST_SUFFIX)

    def _write_last(self) -> None:
        try:
            self._last_path().write_text(self.key or "", encoding="utf-8")
        except OSError:
            pass

    def _read_last(self) -> str | None:
        try:
            text = self._last_path().read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return text if _KEY_RE.match(text) else None

    def _save(self, name: str) -> bool:
        ok = save(self.port, name, where=self.where, post=self._post, get=self._get, log=self._log)
        if ok:
            self.dirty = False
            self._evict()
        return ok

    def _evict(self) -> None:
        """Keep the MAX_SAVED_OPENINGS most recently used app files of this model; the rest go."""
        pattern = re.compile(re.escape(self.stem) + r"-[0-9a-f]{8}\.bin$")
        files = [f for f in self.where.glob(self.stem + "-*.bin") if pattern.match(f.name)]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        for old in files[self._max:]:
            try:
                old.unlink()
                self._log(f"prompt memory: removed {old.name}, the least recently used of "
                          f"{len(files)} saved openings (at most {self._max} are kept)")
            except OSError:
                pass

    def recall(self) -> None:
        """Right after the model is ready: restore the opening the slot held at the last stop."""
        with self.lock:
            key = self._read_last()
            if key and restore(self.port, self.name(key), self.where, post=self._post, log=self._log):
                self.key, self.dirty = key, False
                self._touch(self.name(key))
                return
            restore(self.port, self.base, self.where, post=self._post, log=self._log)
            self.key, self.dirty = None, False

    def remember(self) -> None:
        """Just before the model stops: save the slot under its app's name, note it as the last."""
        with self.lock:
            self._save(self.name(self.key))
            self._write_last()

    def before_request(self, key: str | None) -> None:
        """Called by the router before a chat is forwarded: make the slot hold this app's opening."""
        with self.lock:
            if key is not None and key == self.key:
                self.dirty = True
                return
            if self.key is not None and self.dirty:
                self._save(self.name(self.key))     # keep what this app has read so far
            if key is None:
                # A keyless chat (a title, a tag, a short prompt) overwrites the slot: nothing known.
                self.key, self.dirty = None, False
                return
            name = self.name(key)
            if restore(self.port, name, self.where, post=self._post, log=self._log):
                self._touch(name)
            else:
                self._log(f"prompt memory: no saved opening for app {key} yet; it is read this once")
            self.key, self.dirty = key, True
            self._write_last()

    def _touch(self, name: str) -> None:
        try:
            os.utime(self.where / name)   # used now: the last to be evicted
        except OSError:
            pass


_ACTIVE: dict[int, PromptSlots] = {}
_ACTIVE_LOCK = threading.Lock()


def register(port: int, slots: PromptSlots) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE[port] = slots


def unregister(port: int) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.pop(port, None)


def for_port(port: int | None) -> PromptSlots | None:
    """The running model's slot on `port` (the router asks before forwarding a chat)."""
    if port is None:
        return None
    with _ACTIVE_LOCK:
        return _ACTIVE.get(port)
