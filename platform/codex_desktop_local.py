"""Desktop local mode: point the Codex desktop app at the helper's local models.

The desktop app has no profiles and one flat model list whose provider comes
from config.toml, so local models can only appear there by switching the
user's config. This changes exactly three things and nothing else:

- top-level model_provider, model and model_catalog_json;
- apps = false in [features] (connector tools exceed any local context);
- one marked [model_providers.locitize_tuner] block.

Every edit is checked by parsing the result and comparing it with the
original: if anything other than those keys differs, nothing is written. The
original lines are saved so `off` restores exactly them; a key the user or
Codex changed in the meantime is left alone and reported.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import time
import tomllib

import codex_picker

BEGIN = "# >>> LOCITIZE Tuner desktop local mode: `server.py --desktop-local off` removes this block >>>"
END = "# <<< LOCITIZE Tuner desktop local mode <<<"
TOP_KEYS = ("model_provider", "model", "model_catalog_json")
_HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$")


def _key_line(key: str, value: str) -> str:
    return f"{key} = {json.dumps(value)}"


def _headers(lines):
    return [(index, match.group(1)) for index, match in
            ((i, _HEADER.match(line)) for i, line in enumerate(lines)) if match]


def _section(lines, name):
    """(first body line, end) of table `name`, or None."""
    headers = _headers(lines)
    for position, (index, header) in enumerate(headers):
        if header == name:
            end = headers[position + 1][0] if position + 1 < len(headers) else len(lines)
            return index + 1, end
    return None


def _find_key(lines, start, end, key):
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    return next((i for i in range(start, end) if pattern.match(lines[i])), None)


def _without(parsed: dict) -> dict:
    """Parsed config minus everything local mode may change."""
    rest = {k: v for k, v in parsed.items() if k not in TOP_KEYS}
    if isinstance(rest.get("features"), dict):
        rest["features"] = {k: v for k, v in rest["features"].items() if k != "apps"}
        if not rest["features"]:
            rest.pop("features")
    if isinstance(rest.get("model_providers"), dict):
        rest["model_providers"] = {k: v for k, v in rest["model_providers"].items()
                                   if k != codex_picker.PROVIDER}
        if not rest["model_providers"]:
            rest.pop("model_providers")
    return rest


def _write_checked(path: Path, original_text: str, lines, newline: str, expect) -> None:
    text = newline.join(lines) + newline
    try:
        before, after = tomllib.loads(original_text), tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError("config.toml edit refused: result would not parse") from exc
    if _without(before) != _without(after) or not expect(after):
        raise ValueError("config.toml edit refused: unexpected change outside local-mode keys")
    codex_picker._atomic_write(path, text)


def enable(home: Path, state_path: Path, router_port: int, default_model_id: str) -> dict:
    home = Path(home)
    config_path = home / "config.toml"
    catalog = home / codex_picker.CATALOG_FILE
    if state_path.exists():
        raise ValueError("desktop local mode is already on")
    if not catalog.exists() or not codex_picker._managed(catalog):
        raise ValueError("enable the Codex picker first")
    original = config_path.read_bytes().decode("utf-8") if config_path.exists() else ""
    tomllib.loads(original)  # refuse to touch a config that does not parse
    newline = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines()
    if BEGIN in lines or _section(lines, f"model_providers.{codex_picker.PROVIDER}"):
        raise ValueError("config.toml already defines the helper provider")
    backup = home / f"config.toml.before-locitize-local-{time.strftime('%Y%m%d-%H%M%S')}.bak"
    if config_path.exists():
        shutil.copy2(config_path, backup)
    values = {"model_provider": codex_picker.PROVIDER, "model": default_model_id,
              "model_catalog_json": catalog.as_posix()}
    saved = {}
    first_header = _headers(lines)[0][0] if _headers(lines) else len(lines)
    insert_at = first_header
    while insert_at > 0 and not lines[insert_at - 1].strip():
        insert_at -= 1
    for key in TOP_KEYS:
        index = _find_key(lines, 0, first_header, key)
        new = _key_line(key, values[key])
        if index is None:
            saved[key] = None
            lines.insert(insert_at, new)
            insert_at += 1
            first_header += 1
        else:
            saved[key] = lines[index]
            lines[index] = new
    section = _section(lines, "features")
    if section is None:
        saved["features.apps"] = None
        saved["features_table_added"] = True
        lines += ["", "[features]", "apps = false"]
    else:
        index = _find_key(lines, section[0], section[1], "apps")
        saved["features_table_added"] = False
        if index is None:
            saved["features.apps"] = None
            lines.insert(section[0], "apps = false")
        else:
            saved["features.apps"] = lines[index]
            lines[index] = "apps = false"
    lines += ["", BEGIN, f"[model_providers.{codex_picker.PROVIDER}]", 'name = "LOCITIZE (local)"',
              f'base_url = "http://127.0.0.1:{int(router_port)}/v1"', 'wire_api = "responses"', END]

    def expect(parsed):
        provider = parsed.get("model_providers", {}).get(codex_picker.PROVIDER, {})
        return (all(parsed.get(k) == v for k, v in values.items())
                and parsed.get("features", {}).get("apps") is False
                and provider.get("base_url") == f"http://127.0.0.1:{int(router_port)}/v1")
    _write_checked(config_path, original, lines, newline, expect)
    codex_picker._atomic_write(state_path, json.dumps({"saved": saved, "values": values,
                                                       "backup": str(backup)}, indent=2) + "\n")
    return {"desktop_local_mode": "on", "default_model_id": default_model_id,
            "backup": backup.name, "next_step": "Start the model (server.py --serve), then open the Codex desktop app."}


def disable(home: Path, state_path: Path) -> dict:
    home = Path(home)
    config_path = home / "config.toml"
    if not state_path.exists():
        return {"desktop_local_mode": "off", "changed": False}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    saved, values = state["saved"], state["values"]
    original = config_path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines()
    left_alone = []
    if BEGIN in lines and END in lines[lines.index(BEGIN):]:
        start, stop = lines.index(BEGIN), lines.index(END, lines.index(BEGIN))
        if start > 0 and not lines[start - 1].strip():
            start -= 1
        del lines[start:stop + 1]
    first_header = _headers(lines)[0][0] if _headers(lines) else len(lines)
    for key in TOP_KEYS:
        index = _find_key(lines, 0, first_header, key)
        ours = _key_line(key, values[key])
        if index is None or lines[index].strip() != ours:
            left_alone.append(key)
        elif saved[key] is None:
            del lines[index]
            first_header -= 1
        else:
            lines[index] = saved[key]
    section = _section(lines, "features")
    index = _find_key(lines, section[0], section[1], "apps") if section else None
    if index is None or lines[index].strip() != "apps = false":
        left_alone.append("features.apps")
    elif saved["features.apps"] is None:
        del lines[index]
        if saved.get("features_table_added"):
            section = _section(lines, "features")
            if section and not any(l.strip() for l in lines[section[0]:section[1]]):
                start = section[0] - 1
                if start > 0 and not lines[start - 1].strip():
                    start -= 1
                del lines[start:section[1]]
    else:
        lines[index] = saved["features.apps"]
    _write_checked(config_path, original, lines, newline, lambda parsed: codex_picker.PROVIDER
                   not in parsed.get("model_providers", {}))
    state_path.unlink()
    return {"desktop_local_mode": "off", "changed": True, "left_alone_changed_elsewhere": left_alone,
            "backup": Path(state["backup"]).name}
