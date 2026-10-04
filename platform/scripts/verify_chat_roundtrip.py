"""AC20 harness: prove a real chooser -> Open WebUI -> llama.cpp chat round-trip.

The chat-UI chooser's FUNCTIONAL proof for M9-lite (Architecture M9.4/M9.6) -- not a
page-load check. It drives the REAL production paths, with no fabrication:

  1. sets chat.preferred_ui = openwebui (in-memory for this run),
  2. starts a REAL chat model on llama.cpp via the real ModelController,
  3. starts the REAL Open WebUI managed service (build_openwebui_spec) on the SAME
     ServiceManager, wired to the running llama.cpp server as its OpenAI backend,
  4. drives the pure resolve_chat_choice and asserts it resolves to OPEN_OPENWEBUI,
  5. opens a real session against Open WebUI (WEBUI_AUTH disabled -> the auto
     admin sign-in returns a token), lists the models Open WebUI is proxying from
     llama.cpp, and sends ONE real chat message THROUGH Open WebUI's OpenAI proxy,
  6. asserts a NON-EMPTY reply actually round-trips back -- proving the backend wiring
     is live, never merely that the page loaded,
  7. stops BOTH services cleanly and confirms no orphan (open-webui / llama-server)
     survived.

The first-run embedding-model fetch stays disabled (the spec env), so no model
download happens. Run from Codebase/platform (needs a real llama.cpp binary + model
present):  python scripts/verify_chat_roundtrip.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PLATFORM_DIR = Path(__file__).resolve().parent.parent
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from config import Config  # noqa: E402
from launcher import Launcher, resolve_log_dir  # noqa: E402
from services import ServiceStatus  # noqa: E402
from webui import (  # noqa: E402
    ChatDecision,
    build_openwebui_spec,
    resolve_chat_choice,
    webui_available,
)

# A short, unambiguous prompt so a real model returns a real non-empty reply fast.
_PROMPT = "Reply with a short one-sentence greeting."
# HTTP timeout for the round-trip completion (cold model + one generation).
_CHAT_TIMEOUT_S = 180.0


def _http_json(
    url: str, payload: dict | None = None, token: str | None = None, timeout: float = 30.0
) -> tuple[int, Any]:
    """Minimal stdlib JSON request; returns (status_code, parsed_body_or_text).

    POST when `payload` is given, else GET. Adds a Bearer token when supplied. Never
    raises on an HTTP error status -- returns the code and body so the caller reports
    honestly instead of crashing.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            code = response.getcode()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        code = exc.code
    except (urllib.error.URLError, OSError) as exc:
        return 0, str(exc)
    try:
        return code, json.loads(body)
    except json.JSONDecodeError:
        return code, body


def _process_pids(image: str) -> set[int]:
    """PIDs of a running image (Windows tasklist); empty set on any failure."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    pids: set[int] = set()
    for line in out.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            pids.add(int(parts[1]))
    return pids


def _pick_model(models: Any) -> Any:
    """Choose a launchable chat model whose gguf actually exists on disk."""
    import os

    for model in models.models:
        if (
            model.status == "installed"
            and model.location
            and os.path.isfile(model.location)
            and model.mmproj is None  # a plain text chat model, not the vision one
        ):
            return model
    return None


def run_roundtrip() -> dict:
    """Drive the full real round trip and return a verdict summary dict."""
    summary: dict[str, Any] = {
        "outcome": "failed",
        "reason": "",
        "model_id": None,
        "llama_port": None,
        "openwebui_port": None,
        "chooser_decision": None,
        "openwebui_models": None,
        "reply": "",
        "reply_nonempty": None,
        "no_orphan": None,
    }

    settings, models, issues = Config.load()
    errors = [str(i) for i in issues if i.level == "ERROR"]
    if errors:
        summary["reason"] = f"config errors: {errors}"
        return summary
    # This run prefers Open WebUI (the chooser must resolve to it).
    settings.chat.preferred_ui = "openwebui"

    if not webui_available(settings):
        summary["reason"] = "Open WebUI is not installed in .webui-venv (build-time prerequisite)"
        return summary
    model = _pick_model(models)
    if model is None:
        summary["reason"] = "no launchable chat model with an on-disk gguf found"
        return summary
    summary["model_id"] = model.id

    llama_before = _process_pids("llama-server.exe")
    webui_before = _process_pids("open-webui.exe")

    launcher = Launcher()
    log_dir = resolve_log_dir(settings)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ONE shared ServiceManager supervises both services, so a single stop_all tears
    # both down with the no-orphan tree-kill contract.
    _registry, manager, controller = launcher._build_controller(
        settings, models, log_path=str(log_dir / "roundtrip_model.log")
    )
    import atexit

    atexit.register(manager.stop_all)

    ready = False
    try:
        # --- start the real llama.cpp chat model --- #
        status = controller.start(model.id)
        if status is not ServiceStatus.RUNNING:
            summary["reason"] = f"model {model.id} did not start ({status.value})"
            return summary
        llama_port = controller.running_port
        summary["llama_port"] = llama_port
        # Point Open WebUI at the model's ACTUAL resolved port (it may have
        # auto-incremented off 8080 if that port was busy), so the derived
        # OPENAI_API_BASE_URL reaches the live backend for this proof.
        if llama_port:
            settings.ports.llama_cpp = llama_port

        # --- start the real Open WebUI service on the SAME manager --- #
        _mgr, webui_controller = launcher._build_service_controller(
            settings,
            lambda: build_openwebui_spec(settings, str(log_dir / "roundtrip_openwebui.log")),
            manager=manager,
        )
        webui_status = webui_controller.start()
        if webui_status is not ServiceStatus.RUNNING:
            summary["reason"] = "Open WebUI service did not become ready"
            return summary
        webui_port = webui_controller.resolved_port or settings.ports.openwebui
        summary["openwebui_port"] = webui_port

        # --- the chooser must resolve to OPEN_OPENWEBUI --- #
        resolution = resolve_chat_choice(
            preferred=settings.chat.preferred_ui,
            cli_override=None,
            model_running=controller.running_model_id is not None,
            webui_installed=True,
            webui_ready=True,
        )
        summary["chooser_decision"] = resolution.decision.name
        if resolution.decision is not ChatDecision.OPEN_OPENWEBUI:
            summary["reason"] = f"chooser did not resolve to OPEN_OPENWEBUI ({resolution.decision.name})"
            return summary

        base = f"http://127.0.0.1:{webui_port}"

        # --- auto admin sign-in (WEBUI_AUTH disabled) returns a token --- #
        code, body = _http_json(
            f"{base}/api/v1/auths/signin", payload={"email": "", "password": ""}, timeout=60.0
        )
        token = body.get("token") if isinstance(body, dict) else None
        if not token:
            summary["reason"] = f"could not obtain an Open WebUI token (HTTP {code}: {body})"
            return summary

        # --- the models Open WebUI is proxying from llama.cpp --- #
        code, body = _http_json(f"{base}/openai/models", token=token, timeout=60.0)
        model_ids: list[str] = []
        if isinstance(body, dict):
            for entry in body.get("data", []):
                if isinstance(entry, dict) and entry.get("id"):
                    model_ids.append(str(entry["id"]))
        summary["openwebui_models"] = model_ids
        if not model_ids:
            summary["reason"] = f"Open WebUI proxied no models from llama.cpp (HTTP {code}: {body})"
            return summary

        # --- send ONE real chat message THROUGH Open WebUI to llama.cpp --- #
        chat_payload = {
            "model": model_ids[0],
            "messages": [{"role": "user", "content": _PROMPT}],
            "stream": False,
        }
        code, body = _http_json(
            f"{base}/openai/chat/completions",
            payload=chat_payload,
            token=token,
            timeout=_CHAT_TIMEOUT_S,
        )
        reply = ""
        if isinstance(body, dict):
            choices = body.get("choices") or []
            if choices and isinstance(choices[0], dict):
                reply = (choices[0].get("message") or {}).get("content", "") or ""
        summary["reply"] = reply.strip()[:400]
        summary["reply_nonempty"] = bool(reply.strip())
        if not reply.strip():
            summary["reason"] = f"no non-empty reply round-tripped (HTTP {code}: {str(body)[:200]})"
            return summary

        ready = True
    finally:
        manager.stop_all()

    # --- confirm no orphan process survived the run --- #
    time.sleep(1.0)
    llama_after = _process_pids("llama-server.exe")
    webui_after = _process_pids("open-webui.exe")
    new_llama = llama_after - llama_before
    new_webui = webui_after - webui_before
    summary["no_orphan"] = not new_llama and not new_webui

    if ready and summary["reply_nonempty"] and summary["no_orphan"]:
        summary["outcome"] = "verified"
        summary["reason"] = (
            "a real chat message round-tripped through Open WebUI to llama.cpp "
            "and both services stopped with no orphan"
        )
    elif not summary["no_orphan"]:
        summary["reason"] = (
            summary["reason"]
            or f"orphan process survived (llama={sorted(new_llama)}, webui={sorted(new_webui)})"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AC20 chat round-trip proof")
    parser.add_argument("--json", action="store_true", help="emit a JSON verdict")
    args = parser.parse_args(argv)

    summary = run_roundtrip()
    if args.json:
        print(json.dumps(summary))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0 if summary["outcome"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
