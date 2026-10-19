"""Codex CLI model-picker entries for helper-served local models.

Codex lists picker models from a model catalog. A catalog replaces the whole
list, so it is written into a layered Codex profile rather than the user's
config: `codex --profile locitize` shows the local models, plain `codex` and
the desktop app keep their normal models. Only two helper-managed files in the
Codex home are ever written or removed; config.toml is never read or written.

Codex's own default instructions for an unknown model are captured once from
the installed CLI over loopback, so a picker entry behaves like the fallback
that was verified, without shipping or rewriting Codex's prompt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROFILE = "locitize"
# A provider id of its own: profiles merge with config.toml, so reusing a
# provider name the user already defined would inherit its keys (env_key...).
PROVIDER = "locitize_tuner"
PROFILE_FILE = PROFILE + ".config.toml"
CATALOG_FILE = PROFILE + "-models.json"
MARKER = "# Managed by LOCITIZE Tuner. disable_codex_picker removes this file."
CATALOG_MARKER = "LOCITIZE Tuner"
# Measured 2026-09-30 on a real Codex setup: with app connectors off (as the
# profile sets) a first request is still ~25k tokens once plugins, MCP servers
# and skills are declared; with them on it was ~201k. Codex's bare minimum is
# ~10k, so 32768 leaves room for the setup and a conversation.
MIN_CONTEXT = 32768
MAX_MODELS = 8
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    return Path(configured) if configured else Path.home() / ".codex"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _managed(path: Path) -> bool:
    """A missing file, or one this helper wrote; never a user-authored file."""
    if not path.exists():
        return True
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if path.suffix == ".json":
        try:
            return json.loads(text).get("managed_by") == CATALOG_MARKER
        except (ValueError, AttributeError):
            return False
    return text.startswith(MARKER)


def capture_base_instructions(executable: str | None = None, timeout: float = 90.0) -> str:
    """Record the instructions the installed Codex sends for an unknown model.

    Codex talks only to a loopback listener that answers every request with an
    error, inside a temporary Codex home that is deleted afterwards.
    """
    executable = executable or shutil.which("codex")
    if not executable:
        raise ValueError("Codex CLI is not installed")
    captured: list[str] = []

    class Capture(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if not captured and isinstance(body.get("instructions"), str):
                    captured.append(body["instructions"])
            except (ValueError, TypeError):
                pass
            raw = b'{"error":{"message":"instruction capture only","type":"invalid_request_error"}}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Capture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    home = tempfile.mkdtemp(prefix="locitize_codex_capture_")
    work = tempfile.mkdtemp(prefix="locitize_codex_capture_work_")
    env = dict(os.environ, CODEX_HOME=home)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        env.pop(key, None)
    argv = [executable, "exec", "-m", "locitize-instruction-capture",
            "-c", "model_provider=locitize",
            "-c", 'model_providers.locitize.name="LOCITIZE (capture)"',
            "-c", f'model_providers.locitize.base_url="http://127.0.0.1:{server.server_address[1]}/v1"',
            "-c", 'model_providers.locitize.wire_api="responses"',
            "-c", "features.plugins=false", "-c", "features.apps=false",
            "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
            "--sandbox", "read-only", "--json", "Reply with OK."]
    try:
        subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", stdin=subprocess.DEVNULL, timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Codex instruction capture failed") from exc
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    if not captured or not captured[0].strip() or len(captured[0]) > 200_000:
        raise ValueError("Codex instruction capture failed")
    return captured[0]


def catalog_entry(model, instructions: str, priority: int) -> dict:
    """Fields Codex requires, with measured context and configured modalities."""
    return {
        "slug": model.id,
        "display_name": f"{model.name or model.id} (local)",
        "description": "Local model served by LOCITIZE Tuner on this PC.",
        "supported_reasoning_levels": [],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": priority,
        "support_verbosity": False,
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "experimental_supported_tools": [],
        "base_instructions": instructions,
        "context_window": int(model.context_size),
        # A projector is configured, not proven; COMPATIBILITY.md lists what was verified.
        "input_modalities": ["text", "image"] if getattr(model, "mmproj", None) else ["text"],
    }


def build_catalog(models, instructions: str) -> dict:
    return {"managed_by": CATALOG_MARKER,
            "models": [catalog_entry(model, instructions, 100 + index)
                       for index, model in enumerate(models)]}


def profile_text(router_port: int, default_model_id: str, catalog_path: Path) -> str:
    if not isinstance(router_port, int) or not 1 <= router_port <= 65535:
        raise ValueError("invalid router port")
    if not _MODEL_ID.fullmatch(default_model_id):
        raise ValueError("invalid model ID")
    catalog = json.dumps(Path(catalog_path).as_posix())
    return (f"{MARKER}\n"
            "# Use with: codex --profile locitize   (then /model lists the local models)\n"
            f"model_provider = \"{PROVIDER}\"\n"
            f"model = \"{default_model_id}\"\n"
            f"model_catalog_json = {catalog}\n\n"
            f"[model_providers.{PROVIDER}]\n"
            "name = \"LOCITIZE (local)\"\n"
            f"base_url = \"http://127.0.0.1:{router_port}/v1\"\n"
            "wire_api = \"responses\"\n\n"
            "# ChatGPT app connectors declare hundreds of tools (~175k tokens on a\n"
            "# measured setup); no local model context holds them. This profile only.\n"
            "[features]\n"
            "apps = false\n")


def enable(models, default_model_id: str, router_port: int, instructions: str,
           home: Path | None = None) -> dict:
    """Write the catalog, then the profile that references it."""
    home = Path(home) if home is not None else codex_home()
    profile, catalog = home / PROFILE_FILE, home / CATALOG_FILE
    for path in (profile, catalog):
        if not _managed(path):
            raise ValueError("an existing Codex profile file was not created by this helper")
    ids = [model.id for model in models]
    if not ids or len(ids) > MAX_MODELS or default_model_id not in ids:
        raise ValueError("invalid Codex picker model list")
    _atomic_write(catalog, json.dumps(build_catalog(models, instructions), indent=2) + "\n")
    _atomic_write(profile, profile_text(router_port, default_model_id, catalog))
    return {"profile": PROFILE, "usage": "codex --profile locitize", "models": ids,
            "default_model_id": default_model_id,
            "files": [PROFILE_FILE, CATALOG_FILE]}


def disable(home: Path | None = None) -> dict:
    """Remove only the helper's own files; leave anything user-authored."""
    home = Path(home) if home is not None else codex_home()
    removed, kept = [], []
    for path in (home / PROFILE_FILE, home / CATALOG_FILE):
        if not path.exists():
            continue
        if _managed(path):
            path.unlink()
            removed.append(path.name)
        else:
            kept.append(path.name)
    return {"removed": removed, "kept_user_files": kept}
