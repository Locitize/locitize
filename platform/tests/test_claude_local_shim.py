"""Tests for the claude-local shim installer."""

from pathlib import Path

import setup_env


def test_installs_both_files_beside_claude(tmp_path, monkeypatch):
    fake_claude = tmp_path / "bin" / "claude.exe"
    fake_claude.parent.mkdir(parents=True)
    fake_claude.write_bytes(b"x")
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: str(fake_claude))
    said = []
    data_root = tmp_path / "portable-data"
    result = setup_env.install_claude_local(said.append, data_root=data_root)
    assert result.ok and not result.skipped
    assert (tmp_path / "bin" / "claude-local.cmd").is_file()
    assert (tmp_path / "bin" / "claude-local.ps1").is_file()
    assert str(data_root.resolve()) in (
        tmp_path / "bin" / "claude-local.ps1"
    ).read_text(encoding="utf-8")
    assert str(tmp_path / "bin") in result.message
    assert said


def test_honest_skip_when_claude_absent(monkeypatch):
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: None)
    result = setup_env.install_claude_local()
    assert result.ok and result.skipped
    assert "not installed" in result.message


def test_existing_foreign_shim_pair_is_preserved_without_partial_write(
        tmp_path, monkeypatch):
    fake_claude = tmp_path / "bin" / "claude.exe"
    fake_claude.parent.mkdir(parents=True)
    fake_claude.write_bytes(b"x")
    cmd = fake_claude.parent / "claude-local.cmd"
    ps1 = fake_claude.parent / "claude-local.ps1"
    cmd.write_text("owner cmd\n", encoding="utf-8")
    ps1.write_text("owner ps1\n", encoding="utf-8")
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: str(fake_claude))

    result = setup_env.install_claude_local()

    assert result.ok and result.skipped
    assert "non-locitize" in result.message
    assert cmd.read_text(encoding="utf-8") == "owner cmd\n"
    assert ps1.read_text(encoding="utf-8") == "owner ps1\n"


def test_one_foreign_shim_blocks_updating_the_pair(tmp_path, monkeypatch):
    fake_claude = tmp_path / "bin" / "claude.exe"
    fake_claude.parent.mkdir(parents=True)
    fake_claude.write_bytes(b"x")
    cmd = fake_claude.parent / "claude-local.cmd"
    cmd.write_text("owner cmd\n", encoding="utf-8")
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: str(fake_claude))

    result = setup_env.install_claude_local()

    assert result.ok and result.skipped
    assert cmd.read_text(encoding="utf-8") == "owner cmd\n"
    assert not (fake_claude.parent / "claude-local.ps1").exists()


def test_existing_locitize_owned_shims_are_updated(tmp_path, monkeypatch):
    fake_claude = tmp_path / "bin" / "claude.exe"
    fake_claude.parent.mkdir(parents=True)
    fake_claude.write_bytes(b"x")
    header = "claude-local - run Claude Code against the model locitize is serving."
    (fake_claude.parent / "claude-local.cmd").write_text(
        f"rem {header}\nold\n", encoding="utf-8"
    )
    (fake_claude.parent / "claude-local.ps1").write_text(
        f"# {header}\nold\n", encoding="utf-8"
    )
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: str(fake_claude))

    result = setup_env.install_claude_local()

    scripts = Path(setup_env.__file__).resolve().parent / "scripts"
    assert result.ok and not result.skipped
    assert (fake_claude.parent / "claude-local.cmd").read_bytes() == (
        scripts / "claude-local.cmd"
    ).read_bytes()
    installed_ps1 = (fake_claude.parent / "claude-local.ps1").read_text(encoding="utf-8")
    assert "old" not in installed_ps1
    assert "$installedDataRoot = '" in installed_ps1


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
    assert "model-service.json" in ps1 and "LOCITIZE_DATA_DIR" in ps1
    assert "$installedDataRoot" in ps1
    assert ps1.index("/props") < ps1.index("/v1/models")
    assert "default_generation_settings.n_ctx" in ps1
    assert "$candidate -gt 65535" in ps1
    assert "foreach ($candidate in 8080..8099)" in ps1
    for name in (
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL",
        "DISABLE_TELEMETRY",
        "DISABLE_ERROR_REPORTING",
        "DISABLE_AUTOUPDATER",
    ):
        assert f'$env:{name} = "1"' in ps1
