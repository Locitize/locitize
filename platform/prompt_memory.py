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

The saved state holds the slot's last prompt, so it stays on this machine, in the data
folder, beside the chat history that holds the same text.
"""

from __future__ import annotations

import hashlib
import json
import re
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


def save(port: int, name: str, *, post: Callable[..., dict] = _post, get: Callable[..., Any] = _get,
         log: Callable[[str], None] = lambda text: None) -> bool:
    """Save slot 0 of the server on `port` as `name`, when it is idle and holds at least
    MIN_TOKENS of prompt. True when a file was written."""
    base = f"http://127.0.0.1:{port}"
    try:
        slots = get(base + "/slots", 5.0)
        slot = slots[0] if isinstance(slots, list) and slots else {}
        if slot.get("is_processing"):
            log("prompt memory: the model is busy, nothing saved")
            return False
        held = int(slot.get("n_prompt_tokens") or 0) + int(slot.get("n_decoded") or 0)
        if held < MIN_TOKENS:
            return False
        result = post(base + "/slots/0?action=save", {"filename": name}, SAVE_TIMEOUT_S)
    except Exception as exc:   # noqa: BLE001 - a failed save must never hold up a stop
        log(f"prompt memory: save failed ({exc})")
        return False
    saved = int(result.get("n_saved") or 0)
    ms = (result.get("timings") or {}).get("save_ms")
    log(f"prompt memory: saved {saved} tokens as {name}" + (f" in {ms:.0f} ms" if isinstance(ms, (int, float)) else ""))
    return saved > 0


def restore(port: int, name: str, where: Path, *, post: Callable[..., dict] = _post,
            log: Callable[[str], None] = lambda text: None) -> bool:
    """Restore `name` into slot 0 of the server on `port` if the file exists in `where`."""
    if not (where / name).is_file():
        return False
    try:
        result = post(f"http://127.0.0.1:{port}/slots/0?action=restore", {"filename": name}, RESTORE_TIMEOUT_S)
    except Exception as exc:   # noqa: BLE001 - the model still serves, just without the head start
        log(f"prompt memory: restore of {name} refused ({exc}); it is read afresh and saved again")
        return False
    restored = int(result.get("n_restored") or 0)
    ms = (result.get("timings") or {}).get("restore_ms")
    log(f"prompt memory: restored {restored} tokens from {name}"
        + (f" in {ms:.0f} ms" if isinstance(ms, (int, float)) else ""))
    return restored > 0
