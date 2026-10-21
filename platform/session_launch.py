"""Per-process local coding profiles; no global or project configuration edits."""
from __future__ import annotations

import json
import re
import socket
import sys
import tomllib
from pathlib import Path

import harness_launch

LOCAL_PROVIDERS = ("codex", "claude", "opencode")

WEB_TOOL_NAME = "locitize_web"
WEB_TOOL_SCRIPT = Path(__file__).resolve().with_name("web_search_mcp.py")

# Owner report 2026-10-20: Codex on Huihui-Qwen3.8-27B (16,384 tokens) stopped
# with "exceeds the available context size (16384 tokens)". Its first request
# was 12,409 tokens before any work, mostly the owner's own Codex memory notes,
# skills list and plug-in tools, all written for cloud models. A local launch
# turns those off; measured the same day, the first request drops to 7,511
# tokens with the web search tool included. The owner's config.toml is not
# touched: these are -c overrides for the launched process only.
CODEX_LEAN_OVERRIDES = {
    "features.memories": False,
    "skills.include_instructions": False,
    "features.apps": False,
    "features.plugins": False,
    "features.multi_agent": False,
    # Codex's own web search runs on OpenAI's servers; locitize_web replaces it.
    "web_search": "disabled",
}

# Below this a coding tool summarizes its work every few steps: Codex alone
# starts at about 7,500 tokens.
ROOMY_CODING_CONTEXT = 32768


def context_warning(context_size) -> str:
    """A plain warning when a model's context is small for a coding tool."""
    if (not isinstance(context_size, int) or isinstance(context_size, bool)
            or context_size >= ROOMY_CODING_CONTEXT):
        return ""
    return (
        f"This model is set to {context_size:,} tokens of memory. The coding tool's "
        "own instructions use about 7,500 of them, so it will summarize its work "
        "often. For long tasks pick a model with more context, such as "
        "Qwen3.8-35B-A3B."
    )


def web_tool_command(data_root=None) -> list[str]:
    """The command that starts locitize's web search tool (web_search_mcp)."""
    python = Path(sys.executable)
    # pythonw has no console streams; the tool talks over stdin/stdout.
    if python.name.lower() == "pythonw.exe" and python.with_name("python.exe").exists():
        python = python.with_name("python.exe")
    command = [str(python), str(WEB_TOOL_SCRIPT)]
    if data_root:
        command += ["--data-root", str(data_root)]
    return command


def user_codex_mcp_servers(config_path=None) -> list[str]:
    """Names of the MCP servers in the owner's Codex config, to switch off for
    a local launch: their tools were written for cloud models and crowd a small
    context. An unreadable config just means none."""
    path = Path(config_path) if config_path else harness_launch.codex_config_path()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return []
    return [name for name in servers
            if re.fullmatch(r"[A-Za-z0-9_-]+", name) and name != WEB_TOOL_NAME]


def port_answers(port, timeout: float = 0.5) -> bool:
    """True when something accepts connections on 127.0.0.1:port."""
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        return False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def build_local_launch(provider, project, model_id, port, resume_id="", context_size=None,
                       data_root=None, codex_port=None, codex_config_path=None):
    """argv and environment for one coding terminal on a local model.

    Every launch gets locitize's web search tool (web_search_mcp). Codex gets
    it only when codex_port, the model router's port, is given: Codex sends MCP
    tools as Responses "namespace" tools, which llama-server drops (seen
    2026-10-20: no MCP tool reached the model), and the router flattens them
    into plain functions (responses_tools.flatten_request). Without it Codex
    still works straight against llama-server, just without web search.
    """
    if provider not in LOCAL_PROVIDERS:
        raise ValueError("Local model routing is available for Codex, Claude Code and OpenCode")
    project = str(project or "").strip()
    project_path = Path(project) if project else None
    if project_path is None or not project_path.is_absolute() or not project_path.is_dir():
        raise ValueError(
            "Choose an existing project folder using its full path before launching."
        )
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("The local model endpoint is not ready")
    if codex_port is not None and (
        not isinstance(codex_port, int) or not 1 <= codex_port <= 65535
    ):
        raise ValueError("The local model router port is not valid")
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
        codex_base = f"http://127.0.0.1:{codex_port}" if codex_port else base
        overrides = {
            "model_provider": "locitize",
            "model_providers.locitize.name": "locitize (local)",
            "model_providers.locitize.base_url": codex_base + "/v1",
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
        overrides.update(CODEX_LEAN_OVERRIDES)
        for name in user_codex_mcp_servers(codex_config_path):
            overrides[f"mcp_servers.{name}.enabled"] = False
        if codex_port:
            command = web_tool_command(data_root)
            overrides[f"mcp_servers.{WEB_TOOL_NAME}.command"] = command[0]
            overrides[f"mcp_servers.{WEB_TOOL_NAME}.args"] = command[1:]
        for key, value in overrides.items():
            argv += ["-c", f"{key}={json.dumps(value)}"]
        env = harness_launch.codex_launch_env()
    elif provider == "claude":
        argv = harness_launch.claude_launch_argv(model_id)
        if data_root:
            # A file, not inline JSON: the launch .bat doubles every quote,
            # which Codex reads back correctly but other programs may not.
            command = web_tool_command(data_root)
            mcp_file = Path(data_root) / "run" / "web-search-mcp.json"
            mcp_file.parent.mkdir(parents=True, exist_ok=True)
            mcp_file.write_text(json.dumps({"mcpServers": {WEB_TOOL_NAME: {
                "command": command[0], "args": command[1:]}}}, indent=2) + "\n",
                encoding="utf-8")
            argv += ["--mcp-config", str(mcp_file)]
        if resume_id:
            argv += ["--resume", resume_id]
        env = harness_launch.claude_launch_env(base, context_size)
    else:
        argv = harness_launch.opencode_launch_argv(project, model_id)
        if resume_id:
            argv += ["--session", resume_id]
        config = harness_launch.build_opencode_provider_config(base + "/v1", model_id)
        config["mcp"] = {WEB_TOOL_NAME: {
            "type": "local", "command": web_tool_command(data_root), "enabled": True}}
        env = {"OPENCODE_CONFIG_CONTENT": json.dumps(config)}
    return argv, env
