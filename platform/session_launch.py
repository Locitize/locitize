"""Per-process local coding profiles; no global or project configuration edits."""
from __future__ import annotations

import json
import re
from pathlib import Path

import harness_launch

LOCAL_PROVIDERS = ("codex", "claude", "opencode")


def build_local_launch(provider, project, model_id, port, resume_id="", context_size=None):
    if provider not in LOCAL_PROVIDERS:
        raise ValueError("Local model routing is available for Codex, Claude Code and OpenCode")
    if not Path(project).is_dir():
        raise ValueError("Project folder is missing. Choose its new location before launching.")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("The local model endpoint is not ready")
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", model_id):
        raise ValueError("Invalid local model identity")
    if resume_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}", resume_id):
        raise ValueError("Unsupported session identity")
    if provider in ("codex", "claude") and context_size is not None and (
        not isinstance(context_size, int) or isinstance(context_size, bool)
        or context_size < 2
    ):
        raise ValueError("Measured local context must be an integer of at least 2 tokens")
    base = f"http://127.0.0.1:{port}"
    if provider == "codex":
        argv = ["codex"] + (["resume", resume_id] if resume_id else [])
        argv += ["-C", project, "-m", model_id]
        overrides = {
            "model_provider": "locitize",
            "model_providers.locitize.name": "locitize (local)",
            "model_providers.locitize.base_url": base + "/v1",
            "model_providers.locitize.env_key": "LOCITIZE_CODEX_API_KEY",
            "model_providers.locitize.wire_api": "responses",
        }
        if context_size is not None:
            # This is the caller-observed running capacity, never native model
            # capacity or a memory estimate. Keep 20% headroom for responses and
            # tool results; this conservative compaction policy is not a speed
            # measurement. CLI overrides affect only this launched process.
            overrides["model_context_window"] = context_size
            overrides["model_auto_compact_token_limit"] = max(1, context_size * 4 // 5)
        for key, value in overrides.items():
            argv += ["-c", f"{key}={json.dumps(value)}"]
        env = harness_launch.codex_launch_env()
    elif provider == "claude":
        argv = harness_launch.claude_launch_argv(model_id)
        if resume_id:
            argv += ["--resume", resume_id]
        env = harness_launch.claude_launch_env(base, context_size)
    else:
        argv = harness_launch.opencode_launch_argv(project, model_id)
        if resume_id:
            argv += ["--session", resume_id]
        env = {"OPENCODE_CONFIG_CONTENT": json.dumps(
            harness_launch.build_opencode_provider_config(base + "/v1", model_id))}
    return argv, env
