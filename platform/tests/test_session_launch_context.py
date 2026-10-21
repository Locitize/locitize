"""Process-specific Codex context policy without global configuration writes."""
import json
import os

import pytest
from session_launch import build_local_launch


def overrides(argv):
    """Read only CLI configuration pairs, preserving their typed JSON values."""
    return dict((argv[i + 1].split("=", 1)[0],
                 json.loads(argv[i + 1].split("=", 1)[1]))
                for i, value in enumerate(argv[:-1]) if value == "-c")


@pytest.mark.parametrize("resume", ["", "existing-session"])
def test_actual_context_process_override_preserves_files(tmp_path, resume):
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "user-default"\nmodel_context_window = 99999\n')
    before = config_path.read_bytes()
    environment = dict(os.environ)
    argv, _ = build_local_launch("codex", str(tmp_path), "registered", 8093,
                                 resume, context_size=16384)
    values = overrides(argv)
    assert values["model_context_window"] == 16384
    assert values["model_auto_compact_token_limit"] == 13107
    assert values["model_providers.locitize.wire_api"] == "responses"
    assert config_path.read_bytes() == before
    assert dict(os.environ) == environment


def test_unknown_context_leaves_model_defaults(tmp_path):
    argv, _ = build_local_launch("codex", str(tmp_path), "registered", 8093)
    assert "model_context_window" not in overrides(argv)
    assert "model_auto_compact_token_limit" not in overrides(argv)


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1, "8192", 8192.0])
def test_bad_context_rejected_without_coercion(tmp_path, invalid, provider):
    with pytest.raises(ValueError, match="Measured local context"):
        build_local_launch(provider, str(tmp_path), "registered", 8093,
                           context_size=invalid)


def _codex_config(tmp_path, text):
    path = tmp_path / "codex-config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_codex_local_launch_turns_off_cloud_only_extras(tmp_path):
    # Owner report 2026-10-20: the owner's memory notes, skills and plug-in
    # tools filled 12,409 of a 16,384-token model before any work.
    config = _codex_config(tmp_path, '[mcp_servers.cloudflare]\nurl = "https://x"\n'
                           '[mcp_servers.node_repl]\ncommand = "node"\n')
    before = config.read_bytes()
    argv, _ = build_local_launch("codex", str(tmp_path), "registered", 8080,
                                 context_size=16384, codex_config_path=config)
    values = overrides(argv)
    assert values["features.memories"] is False
    assert values["skills.include_instructions"] is False
    assert values["features.apps"] is False
    assert values["features.plugins"] is False
    assert values["features.multi_agent"] is False
    assert values["web_search"] == "disabled"
    assert values["mcp_servers.cloudflare.enabled"] is False
    assert values["mcp_servers.node_repl.enabled"] is False
    assert config.read_bytes() == before


@pytest.mark.parametrize("context_size, keeps_memory", [
    (None, False), (16384, False), (65535, False), (65536, True), (262144, True)])
def test_codex_memory_notes_follow_the_tuned_context(tmp_path, context_size, keeps_memory):
    # Owner decision 2026-10-20: memory notes and skills come back at 65,536
    # tokens; apps, plug-ins and personal MCP servers stay off at any size.
    config = _codex_config(tmp_path, '[mcp_servers.cloudflare]\nurl = "https://x"\n')
    argv, _ = build_local_launch("codex", str(tmp_path), "registered", 8080,
                                 context_size=context_size, codex_config_path=config)
    values = overrides(argv)
    for key in ("features.memories", "skills.include_instructions"):
        assert (key not in values) is keeps_memory
    assert values["features.apps"] is False
    assert values["features.plugins"] is False
    assert values["mcp_servers.cloudflare.enabled"] is False


def test_codex_gets_web_search_through_the_router_only(tmp_path):
    config = _codex_config(tmp_path, "")
    argv, _ = build_local_launch("codex", str(tmp_path), "registered", 8080,
                                 data_root=str(tmp_path), codex_port=8093,
                                 codex_config_path=config)
    values = overrides(argv)
    assert values["model_providers.locitize.base_url"] == "http://127.0.0.1:8093/v1"
    command = ([values["mcp_servers.locitize_web.command"]]
               + values["mcp_servers.locitize_web.args"])
    assert command[1].endswith("web_search_mcp.py")
    assert command[2:] == ["--data-root", str(tmp_path)]

    direct, _ = build_local_launch("codex", str(tmp_path), "registered", 8080,
                                   codex_config_path=config)
    values = overrides(direct)
    assert values["model_providers.locitize.base_url"] == "http://127.0.0.1:8080/v1"
    assert "mcp_servers.locitize_web.command" not in values


def test_bad_router_port_is_refused(tmp_path):
    with pytest.raises(ValueError, match="router port"):
        build_local_launch("codex", str(tmp_path), "registered", 8080, codex_port=0)


def test_claude_and_opencode_get_the_web_search_tool(tmp_path):
    argv, _ = build_local_launch("claude", str(tmp_path), "registered", 8080,
                                 data_root=str(tmp_path))
    mcp_file = argv[argv.index("--mcp-config") + 1]
    with open(mcp_file, encoding="utf-8") as handle:
        server = json.load(handle)["mcpServers"]["locitize_web"]
    assert server["args"][0].endswith("web_search_mcp.py")

    _, env = build_local_launch("opencode", str(tmp_path), "registered", 8080,
                                data_root=str(tmp_path))
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["mcp"]["locitize_web"]["type"] == "local"
    assert config["mcp"]["locitize_web"]["command"][1].endswith("web_search_mcp.py")
    assert "locitize" in config["provider"]


def test_a_small_context_gets_a_plain_warning():
    from session_launch import context_warning

    assert "16,384 tokens" in context_warning(16384)
    assert context_warning(262144) == ""
    assert context_warning(None) == ""
