"""Local-only stdio MCP tools for the LOCITIZE tuning adapter.

No HTTP listener, arbitrary commands, caller-selected files, or remote transport.
"""
import atexit
from typing import Literal
from local_tuning import LocalTuning
from model_discovery import ModelDiscovery, DiscoveryError

adapter = LocalTuning()
discovery = ModelDiscovery(hardware=adapter.assess_computer)
atexit.register(adapter.close)
atexit.register(discovery.close)


def _call(method, *args):
    """Keep exception paths out of protocol responses (they can contain paths)."""
    try:
        return {"ok": True, "result": method(*args)}
    except DiscoveryError as exc:
        return {"ok": False, "error": str(exc)}
    except ValueError as exc:
        # Only known adapter messages cross the boundary: registry/GGUF
        # exceptions may contain private filesystem paths.
        message = str(exc)
        allowed = {"objective must be balanced or speed", "invalid model ID",
                   "unknown or unavailable model ID", "unknown job ID",
                   "unsupported backend; tuning requires llama-cpp",
                   "busy: stop the helper's model or wait for tuning",
                   "busy: helper resources already in use",
                   "local model did not become ready", "invalid local chat port",
                   "local model completion verification failed",
                   "Native GGUF context metadata unavailable; tuning refused",
                   "invalid Codex picker model list", "Codex CLI is not installed",
                   "Codex instruction capture failed",
                   "tune the model first: Codex needs a saved context of at least 32768",
                   "an existing Codex profile file was not created by this helper"}
        if message not in allowed:
            message = "Local operation refused; inspect the helper locally."
        return {"ok": False, "error": message}
    except Exception:
        return {"ok": False, "error": "Local operation unavailable or busy; inspect the helper locally."}


def assess_computer() -> dict:
    """Read local RAM and GPU availability; unknown readings remain explicit."""
    return _call(adapter.assess_computer)


def list_local_models() -> dict:
    """List registered launchable models without private filesystem paths."""
    return _call(adapter.list_local_models)


def start_tuning(model_id: str, objective: str = "balanced") -> dict:
    """Start bounded measured tuning of a registered model. Poll returned job ID."""
    return _call(adapter.start_tuning, model_id, objective)


def tuning_status(job_id: str) -> dict:
    """Read measured trials, progress, verification, and confirmed cleanup."""
    return _call(adapter.tuning_status, job_id)


def cancel_tuning(job_id: str) -> dict:
    """Cancel after the in-flight bounded probe cleans up owned processes."""
    return _call(adapter.cancel_tuning, job_id)


def start_local_chat(model_id: str) -> dict:
    """Start saved settings on loopback and request the local browser chat UI."""
    return _call(adapter.start_local_chat, model_id)


def start_local_model(model_id: str) -> dict:
    """Start a registered model, verify completion, and return its loopback API."""
    return _call(adapter.start_local_model, model_id)


def stop_local_model() -> dict:
    """Stop only this helper's owned local chat server."""
    return _call(adapter.stop_local_model)


def enable_codex_picker(model_ids: list[str], default_model_id: str = "") -> dict:
    """Show tuned local models in `codex --profile locitize` /model; config.toml untouched."""
    return _call(adapter.enable_codex_picker, model_ids, default_model_id or None)


def disable_codex_picker() -> dict:
    """Remove only the helper-managed Codex profile and catalog files."""
    return _call(adapter.disable_codex_picker)



def search_models(query: str, limit: int = 10, cursor: str = "") -> dict:
    """Search public HF metadata only; recommendation never downloads weights."""
    return _call(discovery.search_models, query, limit, cursor)


def list_model_files(repo_id: str) -> dict:
    """Inspect pinned GGUF sets and issue exact opaque download selections."""
    return _call(discovery.list_model_files, repo_id)


def recommend_model_candidates(selection_ids: list[str], use_case: Literal["general", "coding"] = "general", prefer_vision: bool = False) -> dict:
    """Estimate current RAM/VRAM/disk fit; capability and speed remain unmeasured."""
    return _call(discovery.recommend_model_candidates, selection_ids, use_case, prefer_vision)


def start_model_download(selection_id: str) -> dict:
    """Use only after explicit user download request. Does not register or tune."""
    return _call(discovery.start_model_download, selection_id)


def download_status(job_id: str) -> dict:
    """Read verified file-set progress and terminal cleanup, without private paths."""
    return _call(discovery.download_status, job_id)


def cancel_download(job_id: str) -> dict:
    """Cancel owned transfer; poll terminal cleanup before claiming cancellation."""
    return _call(discovery.cancel_download, job_id)


def resume_download(job_id: str) -> dict:
    """Restart interrupted file and reuse rehashed complete files in this session."""
    return _call(discovery.resume_download, job_id)


def register_download(job_id: str) -> dict:
    """Atomically register a complete reverified set; do not start inference/tuning."""
    return _call(discovery.register_download, job_id)

def build_server():
    """Load the optional isolated SDK only when starting an MCP transport."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    server = FastMCP("Locitize")
    # Explicit hints let plugin review and clients distinguish inspection from
    # operations that change saved settings or control local processes.
    read_only = {assess_computer, list_local_models, tuning_status, search_models, list_model_files, recommend_model_candidates, download_status}
    outbound = {search_models, list_model_files, start_model_download, resume_download}
    for tool in (assess_computer, list_local_models, start_tuning, tuning_status,
                 cancel_tuning, start_local_chat, start_local_model, stop_local_model,
                 enable_codex_picker, disable_codex_picker,
                 search_models, list_model_files, recommend_model_candidates, start_model_download,
                 download_status, cancel_download, resume_download, register_download):
        server.tool(annotations=ToolAnnotations(
            readOnlyHint=tool in read_only,
            destructiveHint=tool not in read_only,
            openWorldHint=tool in outbound,
        ))(tool)
    return server


def serve(model_id=None):
    """Keep one picker model and the router up for `codex --profile locitize`."""
    import json
    import signal
    import sys
    import threading

    model_id = model_id or adapter.picker_default_model()
    if not model_id:
        print("No Codex picker model: run enable_codex_picker first or pass a model ID.",
              file=sys.stderr)
        return 2
    started = adapter.start_local_model(model_id)
    print(json.dumps({"serving": started["model_id"], "codex": "codex --profile locitize",
                      "picker_models": started.get("codex_picker_models", []),
                      "router": started["codex_api_base_url"]}), flush=True)
    print("Serving until Ctrl+C.", file=sys.stderr, flush=True)
    # Ctrl+Break and termination would otherwise skip cleanup and orphan the
    # model server; every signal only sets the event, and the caller's finally
    # stops the owned processes.
    stop = threading.Event()
    for name in ("SIGINT", "SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: stop.set())
    while not stop.wait(0.5):
        pass
    return 0


def desktop_local(action):
    """Switch the Codex desktop app's config to/from the local picker models.

    Terminal-only: run with the desktop app closed, since it also writes
    config.toml. Output is one JSON line.
    """
    import json
    import sys
    from pathlib import Path
    import codex_desktop_local
    import codex_picker
    import config

    settings, _, _ = config.Config.load()
    state = Path(settings.data_dir) / "codex_desktop_local.json"
    try:
        if action == "on":
            default = adapter.picker_default_model()
            if not default:
                raise ValueError("enable the Codex picker first")
            result = codex_desktop_local.enable(codex_picker.codex_home(), state,
                                                settings.ports.router, default)
        else:
            result = codex_desktop_local.disable(codex_picker.codex_home(), state)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    import sys
    try:
        if sys.argv[1:2] == ["--desktop-local"] and sys.argv[2:3] in (["on"], ["off"]):
            raise SystemExit(desktop_local(sys.argv[2]))
        if sys.argv[1:2] == ["--serve"]:
            raise SystemExit(serve(sys.argv[2] if len(sys.argv) > 2 else None))
        build_server().run(transport="stdio")
    finally:
        discovery.close()
        adapter.close()
