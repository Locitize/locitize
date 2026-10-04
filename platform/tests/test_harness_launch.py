"""Tests for harness_launch.py's "Launch in Claude Code/Codex/OpenCode" wiring
(owner request 2026-08-21). Everything here is exercised without a real
model, a running llama-server, or the harness CLIs actually being installed:
detection is a monkeypatched shutil.which, config writers operate on
tmp_path, and the terminal-spawn batch-file text is asserted as pure data
(spawn_in_terminal itself, the line that calls Popen, is intentionally left
unexercised here -- it is a thin wrapper around write_launch_batch_file
and build_launch_batch_text, both tested directly).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import harness_launch as hl


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def test_detect_executable_found(monkeypatch):
    monkeypatch.setattr(hl.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert hl.detect_executable("codex") == "/usr/bin/codex"


def test_detect_executable_missing(monkeypatch):
    monkeypatch.setattr(hl.shutil, "which", lambda name: None)
    assert hl.detect_executable("claude") is None


def test_detect_executable_rejects_unknown_harness():
    with pytest.raises(ValueError):
        hl.detect_executable("not-a-real-harness")


def test_detect_harnesses_reports_all_three_independently(monkeypatch):
    resolved = {"claude": "/bin/claude", "codex": None, "opencode": "/bin/opencode"}
    monkeypatch.setattr(hl.shutil, "which", lambda name: resolved[name])
    assert hl.detect_harnesses() == resolved


# --------------------------------------------------------------------------- #
# Codex: ~/.codex/config.toml [model_providers.locitize]
# --------------------------------------------------------------------------- #


def test_upsert_codex_provider_creates_file_and_block(tmp_path):
    config_path = tmp_path / ".codex" / "config.toml"
    result_path = hl.upsert_codex_provider("http://127.0.0.1:8080/v1", config_path)
    assert result_path == config_path
    text = config_path.read_text(encoding="utf-8")
    assert "[model_providers.locitize]" in text
    assert 'base_url = "http://127.0.0.1:8080/v1"' in text
    assert 'env_key = "LOCITIZE_CODEX_API_KEY"' in text


def test_upsert_codex_provider_preserves_other_content(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "model = \"gpt-5\"\n\n[projects.'locitize-test/some/path']\ntrust_level = \"trusted\"\n",
        encoding="utf-8",
    )
    hl.upsert_codex_provider("http://127.0.0.1:8081/v1", config_path)
    text = config_path.read_text(encoding="utf-8")
    assert 'model = "gpt-5"' in text
    assert "[projects.'locitize-test/some/path']" in text
    assert 'trust_level = "trusted"' in text
    assert "[model_providers.locitize]" in text


def test_upsert_codex_provider_replaces_a_stale_block_in_place(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "model = \"gpt-5\"\n"
        "\n[model_providers.locitize]\n"
        'base_url = "http://127.0.0.1:9999/v1"\n'
        'env_key = "OLD"\n'
        "\n[model_providers.other]\n"
        'base_url = "https://example.com"\n',
        encoding="utf-8",
    )
    hl.upsert_codex_provider("http://127.0.0.1:8082/v1", config_path)
    text = config_path.read_text(encoding="utf-8")
    assert text.count("[model_providers.locitize]") == 1
    assert "http://127.0.0.1:8082/v1" in text
    assert "http://127.0.0.1:9999/v1" not in text
    # The unrelated provider section after it survives untouched.
    assert "[model_providers.other]" in text
    assert "https://example.com" in text


def test_upsert_codex_provider_matches_a_whitespace_variant_header(tmp_path):
    # Security review 2026-08-21, CONFIRMED MEDIUM: legal-TOML whitespace
    # inside the brackets (`[ model_providers.locitize ]`) failed the old
    # exact-string match, so the existing block was never found and a
    # SECOND `[model_providers.locitize]` got appended -- which is not
    # valid TOML (a table declared twice) and broke the owner's entire
    # config, not just this section. Real tomllib parse used as the check.
    import tomllib

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[ model_providers.locitize ]\n"
        'base_url = "http://127.0.0.1:9999/v1"\n'
        'env_key = "OLD"\n'
        "\n[model_providers.other]\n"
        'base_url = "https://example.com"\n',
        encoding="utf-8",
    )
    hl.upsert_codex_provider("http://127.0.0.1:8082/v1", config_path)
    text = config_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)  # raises if this regressed
    assert parsed["model_providers"]["locitize"]["base_url"] == "http://127.0.0.1:8082/v1"
    assert parsed["model_providers"]["other"]["base_url"] == "https://example.com"


def test_upsert_codex_provider_refuses_to_write_invalid_toml(tmp_path):
    # Security review 2026-08-21, CONFIRMED MEDIUM: no re-parse-before-commit
    # guard existed. This locks in config.py's own documented rule (see its
    # module docstring) that a malformed edit must leave the original file
    # untouched, by forcing the post-edit text to be unparseable.
    config_path = tmp_path / "config.toml"
    # The unterminated brace sits BEFORE the locitize header, so it survives
    # untouched by the section-replace and keeps the whole document
    # unparseable no matter how clean our own spliced-in block is.
    original = "broken = {\n[model_providers.locitize]\nbase_url = \"old\"\n"
    config_path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="not be valid TOML"):
        hl.upsert_codex_provider("http://127.0.0.1:8082/v1", config_path)
    # Original file untouched.
    assert config_path.read_text(encoding="utf-8") == original


def test_upsert_codex_provider_writes_atomically(tmp_path, monkeypatch):
    # Confirms this routes through config._atomic_write rather than a plain
    # path.write_text, per the security review's fix.
    calls = []
    import config as config_module

    real_atomic_write = config_module._atomic_write

    def spy(path, text):
        calls.append(path)
        real_atomic_write(path, text)

    monkeypatch.setattr(config_module, "_atomic_write", spy)
    config_path = tmp_path / "config.toml"
    hl.upsert_codex_provider("http://127.0.0.1:8082/v1", config_path)
    assert calls == [config_path]


def test_codex_launch_argv_and_env():
    argv = hl.codex_launch_argv("/locitize-test/proj", "qwen3-14b")
    assert argv[:3] == ["codex", "-C", "/locitize-test/proj"]
    assert "model_provider=locitize" in argv
    assert 'model="qwen3-14b"' in argv
    assert hl.codex_launch_env() == {"LOCITIZE_CODEX_API_KEY": "locitize-local"}


def test_prime_codex_update_noop_when_codex_not_on_path(monkeypatch):
    monkeypatch.setattr(hl.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(hl.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    hl.prime_codex_update()
    assert calls == []


def test_prime_codex_update_runs_hidden_version_check(monkeypatch):
    monkeypatch.setattr(hl.shutil, "which", lambda name: "/usr/bin/codex")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))

    monkeypatch.setattr(hl.subprocess, "run", fake_run)
    hl.prime_codex_update()
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == ["codex", "--version"]
    assert kwargs["capture_output"] is True


def test_prime_codex_update_swallows_failures(monkeypatch):
    monkeypatch.setattr(hl.shutil, "which", lambda name: "/usr/bin/codex")

    def raising_run(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(hl.subprocess, "run", raising_run)
    hl.prime_codex_update()  # must not raise

    def timeout_run(*a, **k):
        raise hl.subprocess.TimeoutExpired(cmd="codex", timeout=1)

    monkeypatch.setattr(hl.subprocess, "run", timeout_run)
    hl.prime_codex_update()  # must not raise


# --------------------------------------------------------------------------- #
# OpenCode: project-local opencode.json
# --------------------------------------------------------------------------- #


def test_write_opencode_project_config_creates_new_file(tmp_path):
    path = hl.write_opencode_project_config(
        str(tmp_path), "http://127.0.0.1:8080/v1", "qwen3-14b"
    )
    assert path == tmp_path / "opencode.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["provider"]["locitize"]["npm"] == "@ai-sdk/openai-compatible"
    assert data["provider"]["locitize"]["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert data["provider"]["locitize"]["models"] == {"qwen3-14b": {"name": "qwen3-14b"}}


def test_write_opencode_project_config_merges_existing_providers(tmp_path):
    existing = tmp_path / "opencode.json"
    existing.write_text(
        json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "provider": {"openai": {"npm": "@ai-sdk/openai"}},
                "agent": {"build": {"model": "openai/gpt-5"}},
            }
        ),
        encoding="utf-8",
    )
    hl.write_opencode_project_config(str(tmp_path), "http://127.0.0.1:8091/v1", "qwen3-14b")
    data = json.loads(existing.read_text(encoding="utf-8"))
    assert data["provider"]["openai"]["npm"] == "@ai-sdk/openai"
    assert data["provider"]["locitize"]["options"]["baseURL"] == "http://127.0.0.1:8091/v1"
    assert data["provider"]["locitize"]["models"] == {"qwen3-14b": {"name": "qwen3-14b"}}
    assert data["agent"]["build"]["model"] == "openai/gpt-5"


def test_write_opencode_project_config_accumulates_models_across_launches(tmp_path):
    hl.write_opencode_project_config(str(tmp_path), "http://127.0.0.1:8080/v1", "qwen3-14b")
    hl.write_opencode_project_config(str(tmp_path), "http://127.0.0.1:8080/v1", "demogpt")
    data = json.loads((tmp_path / "opencode.json").read_text(encoding="utf-8"))
    assert data["provider"]["locitize"]["models"] == {
        "qwen3-14b": {"name": "qwen3-14b"},
        "demogpt": {"name": "demogpt"},
    }


def test_write_opencode_project_config_prefers_existing_comment_free_jsonc(tmp_path):
    # A .jsonc file that happens to have no actual comments is safe to merge
    # into just like a plain .json - only real comment content refuses.
    (tmp_path / "opencode.jsonc").write_text(
        '{\n  "$schema": "https://opencode.ai/config.json"\n}\n',
        encoding="utf-8",
    )
    path = hl.write_opencode_project_config(
        str(tmp_path), "http://127.0.0.1:8080/v1", "qwen3-14b"
    )
    assert path.name == "opencode.jsonc"
    assert not (tmp_path / "opencode.json").exists()


def test_write_opencode_project_config_refuses_to_destroy_jsonc_comments(tmp_path):
    # Security review 2026-08-21, LOW: this used to silently json.dumps over
    # an existing opencode.jsonc, discarding every real comment with no
    # warning. A file with actual comment content is now refused loudly
    # (ValueError, caught by gui_controller.py's existing (OSError, ValueError)
    # handler and surfaced as a clear message) rather than destroyed quietly.
    jsonc_path = tmp_path / "opencode.jsonc"
    original = '{\n  // team convention: keep autoshare off\n  "$schema": "https://opencode.ai/config.json"\n}\n'
    jsonc_path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="comments"):
        hl.write_opencode_project_config(str(tmp_path), "http://127.0.0.1:8080/v1", "qwen3-14b")
    # Original file untouched.
    assert jsonc_path.read_text(encoding="utf-8") == original


def test_opencode_launch_argv():
    argv = hl.opencode_launch_argv("/locitize-test/proj", "qwen3-14b")
    assert argv == ["opencode", "/locitize-test/proj", "-m", "locitize/qwen3-14b"]


# --------------------------------------------------------------------------- #
# Claude Code: env pointed at the model's own llama-server (native /v1/messages)
# --------------------------------------------------------------------------- #


def test_claude_launch_argv_carries_the_model_id_and_scopes_the_tool_schema():
    argv = hl.claude_launch_argv("qwen3-14b")
    assert argv[:3] == ["claude", "--model", "qwen3-14b"]
    assert "--strict-mcp-config" in argv
    assert "--tools" in argv
    assert argv[argv.index("--tools") + 1] == hl.CLAUDE_LOCAL_SESSION_TOOLS
    # Empirically confirmed live 2026-08-21: this exact set is small enough
    # for llama-server's grammar compiler to accept (real "PONG" reply,
    # exit 0), where the full default tool+MCP surface hard-errored.
    assert hl.CLAUDE_LOCAL_SESSION_TOOLS == "Bash,Edit,Read,Write,Glob,Grep"


def test_claude_launch_env_points_at_model_server_no_v1_suffix():
    env = hl.claude_launch_env("http://127.0.0.1:8080")
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
    assert env["ANTHROPIC_API_KEY"]  # non-empty; claude CLI requires SOME value
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env


def test_claude_launch_env_sets_max_context_tokens_from_the_real_model():
    env = hl.claude_launch_env("http://127.0.0.1:8080", context_size=10000)
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "10000"


def test_claude_launch_env_omits_max_context_tokens_when_unknown():
    env = hl.claude_launch_env("http://127.0.0.1:8080", context_size=None)
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env
    env_zero = hl.claude_launch_env("http://127.0.0.1:8080", context_size=0)
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env_zero


# --------------------------------------------------------------------------- #
# Windows terminal spawn command (pure argv construction)
# --------------------------------------------------------------------------- #


def test_build_launch_batch_text_sets_env_and_quotes_argv():
    text = hl.build_launch_batch_text(
        ["codex", "-C", "/locitize-test/some project"],
        cwd="/locitize-test/some project",
        env_overrides={"FOO": "bar"},
    )
    assert text.startswith("@echo off\r\n")
    assert 'cd /d "/locitize-test/some project"' in text
    assert 'set "FOO=bar"' in text
    assert '"/locitize-test/some project"' in text


def test_build_launch_batch_text_no_env_overrides():
    text = hl.build_launch_batch_text(
        ["opencode", "/locitize-test/proj"], cwd="/locitize-test/proj", env_overrides={}
    )
    assert "set " not in text
    assert '"opencode" "/locitize-test/proj"' in text


def test_build_launch_batch_text_preserves_literal_quotes_in_argv():
    # The exact defect this format exists to avoid: Codex's `-c model="<id>"`
    # override contains a literal `"` inside one argv element. Doubled inside
    # the outer quotes (cmd's own escape for a literal quote), confirmed
    # empirically to round-trip back to `model="qwen3-14b"` in a real
    # argv-parsing child process.
    text = hl.build_launch_batch_text(
        ["codex", "-c", 'model="qwen3-14b"'], cwd="/locitize-test/proj", env_overrides={}
    )
    assert '"model=""qwen3-14b"""' in text


def test_build_launch_batch_text_quotes_defuse_ampersand_injection(tmp_path):
    # Security review 2026-08-21, CONFIRMED HIGH: an unquoted `&` in a
    # space-free folder name was an unconditional cmd.exe command separator.
    # Every interpolated value must be quoted (a real subprocess execution
    # of this exact shape was used to confirm the fix; this test locks the
    # generated TEXT so a future edit can't silently drop the quoting).
    payload_dir = "/locitize-test/proj&calc"
    text = hl.build_launch_batch_text(
        ["codex", "-C", payload_dir], cwd=payload_dir, env_overrides={}
    )
    assert f'cd /d "{payload_dir}"' in text
    assert f'"{payload_dir}"' in text
    # No unquoted `&` anywhere outside the quoted spans.
    import re

    unquoted = re.sub(r'"[^"]*"', "", text)
    assert "&" not in unquoted


def test_quote_arg_escapes_percent_and_doubles_quotes():
    assert hl._quote_arg("50%done") == '"50%%done"'
    assert hl._quote_arg('say "hi"') == '"say ""hi"""'


def test_quote_arg_rejects_newlines():
    with pytest.raises(ValueError):
        hl._quote_arg("line1\nline2")
    with pytest.raises(ValueError):
        hl._quote_arg("line1\rline2")


def test_write_launch_batch_file_creates_a_fresh_readable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(hl.tempfile, "gettempdir", lambda: str(tmp_path))
    path = hl.write_launch_batch_file(
        ["codex", "-c", 'model="qwen3-14b"'], cwd=str(tmp_path), env_overrides={"X": "y"}
    )
    assert path.exists()
    assert path.suffix == ".bat"
    text = path.read_text(encoding="utf-8")
    assert 'set "X=y"' in text
    assert '"model=""qwen3-14b"""' in text
    path.unlink()


def test_write_launch_batch_file_gives_each_launch_its_own_file(tmp_path, monkeypatch):
    monkeypatch.setattr(hl.tempfile, "gettempdir", lambda: str(tmp_path))
    first = hl.write_launch_batch_file(["codex"], cwd=str(tmp_path), env_overrides={})
    second = hl.write_launch_batch_file(["codex"], cwd=str(tmp_path), env_overrides={})
    assert first != second
    first.unlink()
    second.unlink()


def test_cleanup_stale_launch_batch_files_deletes_only_old_ones(tmp_path, monkeypatch):
    # Security review 2026-08-21, LOW: every launch left a .bat behind
    # forever. Age is simulated via os.utime rather than a real sleep.
    monkeypatch.setattr(hl.tempfile, "gettempdir", lambda: str(tmp_path))
    old = tmp_path / "locitize_harness_old.bat"
    fresh = tmp_path / "locitize_harness_fresh.bat"
    unrelated = tmp_path / "locitize_harness_keepme.txt"
    old.write_text("old", encoding="utf-8")
    fresh.write_text("fresh", encoding="utf-8")
    unrelated.write_text("not a batch file", encoding="utf-8")
    import os as os_module
    import time

    old_time = time.time() - 7200  # two hours ago
    os_module.utime(old, (old_time, old_time))

    deleted = hl.cleanup_stale_launch_batch_files(max_age_s=3600)
    assert deleted == 1
    assert not old.exists()
    assert fresh.exists()
    assert unrelated.exists()  # wrong suffix, never touched


def test_write_launch_batch_file_sweeps_stale_files_first(tmp_path, monkeypatch):
    monkeypatch.setattr(hl.tempfile, "gettempdir", lambda: str(tmp_path))
    stale = tmp_path / "locitize_harness_stale.bat"
    stale.write_text("stale", encoding="utf-8")
    import os as os_module
    import time

    old_time = time.time() - 7200
    os_module.utime(stale, (old_time, old_time))

    new_path = hl.write_launch_batch_file(["codex"], cwd=str(tmp_path), env_overrides={})
    assert not stale.exists()
    assert new_path.exists()
    new_path.unlink()


# --------------------------------------------------------------------------- #
# Zero-friction install (owner request 2026-08-21)
# --------------------------------------------------------------------------- #


def _completed(returncode=0, stdout="", stderr=""):
    import subprocess

    return subprocess.CompletedProcess(args=["x"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_detect_executable_fresh_searches_refreshed_path(monkeypatch):
    seen = {}

    def fake_which(name, path=None):
        seen["name"], seen["path"] = name, path
        return "/fake/claude"

    monkeypatch.setattr(hl, "refreshed_search_path", lambda: "/fake:/other")
    monkeypatch.setattr(hl.shutil, "which", fake_which)
    assert hl.detect_executable_fresh("claude") == "/fake/claude"
    assert seen == {"name": "claude", "path": "/fake:/other"}


def test_detect_executable_fresh_rejects_unknown_harness():
    with pytest.raises(ValueError):
        hl.detect_executable_fresh("not-a-real-harness")


def test_refreshed_search_path_is_plain_path_off_windows(monkeypatch):
    monkeypatch.setattr(hl.sys, "platform", "linux")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert hl.refreshed_search_path() == "/usr/bin:/bin"


def test_install_claude_code_success(monkeypatch):
    monkeypatch.setattr(hl.subprocess, "run", lambda *a, **k: _completed(0))
    monkeypatch.setattr(hl, "detect_executable_fresh", lambda h: "/fake/claude")
    ok, message = hl.install_claude_code()
    assert ok is True
    assert "Claude Code installed" in message


def test_install_claude_code_installer_failure_surfaces_stderr(monkeypatch):
    monkeypatch.setattr(hl.subprocess, "run", lambda *a, **k: _completed(1, stderr="network unreachable"))
    ok, message = hl.install_claude_code()
    assert ok is False
    assert "network unreachable" in message


def test_install_claude_code_not_found_after_install_says_restart(monkeypatch):
    monkeypatch.setattr(hl.subprocess, "run", lambda *a, **k: _completed(0))
    monkeypatch.setattr(hl, "detect_executable_fresh", lambda h: None)
    ok, message = hl.install_claude_code()
    assert ok is False
    assert "restart locitize" in message


def test_install_codex_success(monkeypatch):
    monkeypatch.setattr(hl.subprocess, "run", lambda *a, **k: _completed(0))
    monkeypatch.setattr(hl, "detect_executable_fresh", lambda h: "/fake/codex")
    ok, message = hl.install_codex()
    assert ok is True
    assert "Codex installed" in message


def test_install_codex_installer_failure(monkeypatch):
    monkeypatch.setattr(hl.subprocess, "run", lambda *a, **k: _completed(1, stderr="boom"))
    ok, message = hl.install_codex()
    assert ok is False
    assert "boom" in message


def test_install_opencode_skips_node_install_when_already_available(monkeypatch):
    calls = []
    monkeypatch.setattr(hl, "_node_available", lambda: True)
    monkeypatch.setattr(hl.shutil, "which", lambda name, path=None: f"/fake/{name}")
    monkeypatch.setattr(hl, "detect_executable_fresh", lambda h: "/fake/opencode")

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _completed(0)

    monkeypatch.setattr(hl.subprocess, "run", fake_run)
    ok, message = hl.install_opencode()
    assert ok is True
    assert calls == [["/fake/npm", "i", "-g", "opencode-ai"]]


def test_install_opencode_installs_node_first_when_missing(monkeypatch):
    node_installed = {"called": False}

    monkeypatch.setattr(hl, "_node_available", lambda: False)

    def fake_install_node():
        node_installed["called"] = True
        return True, "Node.js installed (winget, user scope)"

    monkeypatch.setattr(hl, "_install_node", fake_install_node)
    monkeypatch.setattr(hl.shutil, "which", lambda name, path=None: f"/fake/{name}")
    monkeypatch.setattr(hl, "detect_executable_fresh", lambda h: "/fake/opencode")
    monkeypatch.setattr(hl.subprocess, "run", lambda *a, **k: _completed(0))

    ok, message = hl.install_opencode()
    assert ok is True
    assert node_installed["called"] is True


def test_install_opencode_stops_early_when_node_install_fails(monkeypatch):
    monkeypatch.setattr(hl, "_node_available", lambda: False)
    monkeypatch.setattr(hl, "_install_node", lambda: (False, "winget exited 1"))
    ok, message = hl.install_opencode()
    assert ok is False
    assert "Node.js" in message


def test_install_opencode_npm_install_failure(monkeypatch):
    monkeypatch.setattr(hl, "_node_available", lambda: True)
    monkeypatch.setattr(hl.shutil, "which", lambda name, path=None: f"/fake/{name}")
    monkeypatch.setattr(hl.subprocess, "run", lambda *a, **k: _completed(1, stderr="EACCES"))
    ok, message = hl.install_opencode()
    assert ok is False
    assert "EACCES" in message


def test_install_harness_dispatches_by_key(monkeypatch):
    # INSTALL_FUNCTIONS captures the function object at import time, so the
    # dict entry (not the module-level name) is what install_harness() calls.
    monkeypatch.setitem(hl.INSTALL_FUNCTIONS, "codex", lambda: (True, "codex ok"))
    assert hl.install_harness("codex") == (True, "codex ok")


def test_install_harness_rejects_unknown_harness():
    with pytest.raises(ValueError):
        hl.install_harness("not-a-real-harness")
