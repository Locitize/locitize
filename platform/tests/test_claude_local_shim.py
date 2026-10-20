"""Tests for the claude-local shim installer."""

from pathlib import Path

import setup_env


def test_installs_both_files_beside_claude(tmp_path, monkeypatch):
    fake_claude = tmp_path / "bin" / "claude.exe"
    fake_claude.parent.mkdir(parents=True)
    fake_claude.write_bytes(b"x")
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: str(fake_claude))
    said = []
    result = setup_env.install_claude_local(said.append)
    assert result.ok and not result.skipped
    assert (tmp_path / "bin" / "claude-local.cmd").is_file()
    assert (tmp_path / "bin" / "claude-local.ps1").is_file()
    assert str(tmp_path / "bin") in result.message
    assert said


def test_honest_skip_when_claude_absent(monkeypatch):
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: None)
    result = setup_env.install_claude_local()
    assert result.ok and result.skipped
    assert "not installed" in result.message


def test_shipped_shim_sources_carry_the_contract():
    scripts = Path(setup_env.__file__).resolve().parent / "scripts"
    ps1 = (scripts / "claude-local.ps1").read_text(encoding="utf-8")
    cmd = (scripts / "claude-local.cmd").read_text(encoding="utf-8")
    assert "127.0.0.1" in ps1
    assert "ANTHROPIC_BASE_URL" in ps1
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in ps1
    assert "No locitize model is running" in ps1
    assert "--strict-mcp-config" in ps1 and "--tools" in ps1
    assert "claude-local.ps1" in cmd
