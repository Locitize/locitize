"""Focused checks for the isolated, opt-in local Codex verifier."""
import json
from pathlib import Path
import subprocess

import pytest
from scripts import verify_codex_local_model as verifier


def test_command_uses_loopback_responses_and_read_only(tmp_path):
    argv, env = verifier.build_verification_command(str(tmp_path), "local-test", 8765)
    assert argv[:2] == ["codex", "exec"]
    assert 'model_providers.locitize.base_url="http://127.0.0.1:8765/v1"' in argv
    assert 'model_providers.locitize.wire_api="responses"' in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert env == {"LOCITIZE_CODEX_API_KEY": "locitize-local"}


@pytest.mark.parametrize("complete,expected", [(True, "PASS"), (False, "FAIL")])
def test_real_process_contract_is_isolated(monkeypatch, complete, expected):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("CODEX_HOME", "must-not-be-used")
    paths = []
    def run(argv, **kwargs):
        env = kwargs["env"]
        assert "OPENAI_API_KEY" not in env
        assert env["CODEX_HOME"] != "must-not-be-used"
        paths.append(Path(env["CODEX_HOME"]))
        assert paths[0].is_dir()
        events = [{"type": "item.completed", "item": {
            "type": "agent_message", "text": verifier.EXPECTED}}]
        if complete:
            events.append({"type": "turn.completed"})
        return subprocess.CompletedProcess(argv, 0, "\n".join(map(json.dumps, events)), "")
    monkeypatch.setattr(verifier.subprocess, "run", run)
    assert verifier.run_verification("local-test", 8765, executable="codex")["status"] == expected
    assert not paths[0].exists()


def test_timeout_is_failure(monkeypatch):
    def run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
    monkeypatch.setattr(verifier.subprocess, "run", run)
    assert verifier.run_verification("local-test", 8765, executable="codex")["status"] == "FAIL"


def test_extra_explanation_keeps_strict_failure():
    events = [{"type": "item.completed", "item": {"type": "agent_message",
               "text": "19 + 23 = 42\n\n" + verifier.EXPECTED}},
              {"type": "turn.completed", "usage": {"input_tokens": 7388}}]
    checks = verifier.assess_completion(0, events)
    assert checks["connectivity_and_completion"]
    assert checks["correct_result_marker"]
    assert not checks["exact_output_compliance"]
    assert not checks["strict_pass"]
    assert checks["usage"]["input_tokens"] == 7388


def test_tool_use_fails_no_tools_instruction():
    events = [{"type": "item.completed", "item": {"type": "agent_message",
               "text": verifier.EXPECTED}}, {"type": "turn.completed"},
              {"type": "item.completed", "item": {"type": "command_execution"}}]
    checks = verifier.assess_completion(0, events)
    assert checks["exact_output_compliance"]
    assert not checks["no_tools_used"]
    assert not checks["strict_pass"]


def test_measured_context_override(tmp_path):
    argv, _ = verifier.build_verification_command(str(tmp_path), "local", 8765,
                                                   context_size=16384)
    assert "model_context_window=16384" in argv
    with pytest.raises(ValueError):
        verifier.build_verification_command(str(tmp_path), "local", 8765, context_size=0)


def fixture_events():
    return [{"type": "item.completed", "item": {
        "type": "command_execution", "command": "Get-Content area.py",
        "exit_code": 0, "aggregated_output": verifier.FIXTURE}},
        {"type": "item.completed", "item": {"type": "agent_message",
         "text": verifier.FIXTURE_EXPECTED}}, {"type": "turn.completed"}]


def test_fixture_requires_real_read_and_unchanged_snapshot():
    events = fixture_events()
    assert verifier.assess_fixture(0, events, {"area.py": "a"}, {"area.py": "a"})["strict_pass"]
    assert not verifier.assess_fixture(0, events[1:], {}, {})["strict_pass"]
    assert not verifier.assess_fixture(0, events, {"area.py": "a"}, {"area.py": "b"})["strict_pass"]
    events[0]["item"]["exit_code"] = 1
    assert not verifier.assess_fixture(0, events, {}, {})["strict_pass"]


def test_snapshot_detects_new_and_deleted_files(tmp_path):
    fixture = tmp_path / "area.py"
    fixture.write_text(verifier.FIXTURE)
    before = verifier.snapshot_workspace(tmp_path)
    (tmp_path / "extra.txt").write_text("changed")
    assert before != verifier.snapshot_workspace(tmp_path)
    fixture.unlink()
    assert "area.py" not in verifier.snapshot_workspace(tmp_path)


def test_fixture_execution_contract(monkeypatch):
    monkeypatch.setattr(verifier, "sandbox_read_preflight", lambda *args: {"status": "PASS"})
    def run(argv, **kwargs):
        assert argv[-1] == verifier.FIXTURE_PROMPT
        assert "model_context_window=16384" in argv
        assert (kwargs["cwd"] / "area.py").read_text() == verifier.FIXTURE
        return subprocess.CompletedProcess(argv, 0, "\n".join(map(json.dumps, fixture_events())), "")
    monkeypatch.setattr(verifier.subprocess, "run", run)
    report = verifier.run_verification("local", 8765, executable="codex",
                                      context_size=16384, task="fixture")
    assert report["status"] == "PASS"
    assert report["checks"]["successful_fixture_read"]
    with pytest.raises(ValueError):
        verifier.run_verification("local", 8765, executable="codex", task="fixture")


def test_sandbox_probe_is_read_only_and_not_tool_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(verifier.sys, "platform", "win32")
    monkeypatch.setattr(verifier.shutil, "which", lambda name: "powershell.exe")
    def run(argv, **kwargs):
        assert argv[1:4] == ["sandbox", "-P", ":read-only"]
        assert argv[-1] == "Get-Content -LiteralPath area.py"
        return subprocess.CompletedProcess(argv, 0, verifier.FIXTURE, "")
    monkeypatch.setattr(verifier.subprocess, "run", run)
    result = verifier.sandbox_read_preflight("codex", tmp_path, {})
    assert result["status"] == "PASS"
    assert not result["model_tool_evidence"]
    assert not verifier.assess_fixture(0, fixture_events()[1:], {}, {})["strict_pass"]


def mcp_events(answer=None):
    answer = answer or {"defective_expression": "width * height",
                        "corrected_expression": "2 * (width + height)", "example_result": 16}
    return [{"type": "item.completed", "item": {"type": "mcp_tool_call",
             "server": "fixture_reader", "tool": "read_fixture", "status": "completed",
             "error": None, "result": {"content": [{"type": "text", "text": verifier.MCP_FIXTURE}]}}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(answer)}},
            {"type": "turn.completed"}]


def test_mcp_read_is_required_and_correct_expression_is_checked():
    events = mcp_events()
    assert verifier.assess_mcp_fixture(0, events, {}, {})["strict_pass"]
    assert not verifier.assess_mcp_fixture(0, events[1:], {}, {})["strict_pass"]
    events[0]["item"]["result"]["isError"] = True
    assert not verifier.assess_mcp_fixture(0, events, {}, {})["strict_pass"]
    assert not verifier.assess_mcp_fixture(0, mcp_events({"defective_expression": "width * height",
        "corrected_expression": "width + height", "example_result": 16}), {}, {})["strict_pass"]


def test_mcp_does_not_execute_arbitrary_expression():
    with pytest.raises(ValueError):
        verifier.expression_value("__import__('os').system('whoami')", 3, 5)
    assert "width * height" not in verifier.MCP_PROMPT
    assert "2 * (width + height)" not in verifier.MCP_PROMPT


def test_fixed_reader_takes_only_startup_path(tmp_path):
    from scripts.codex_fixture_reader import read_fixed_fixture
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fixture = workspace / "perimeter.py"
    fixture.write_text(verifier.MCP_FIXTURE)
    assert read_fixed_fixture(fixture) == verifier.MCP_FIXTURE
    with pytest.raises(ValueError):
        read_fixed_fixture(tmp_path / "anything.py")


def test_mcp_process_only_config_and_no_shell_preflight(monkeypatch, tmp_path):
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    def run(argv, **kwargs):
        assert argv[1] == "exec"
        assert "mcp_servers.fixture_reader.command=" + json.dumps(str(python)) in argv
        assert argv[-1] == verifier.MCP_PROMPT
        assert "--ignore-rules" not in argv
        assert "model_context_window=16384" in argv
        return subprocess.CompletedProcess(argv, 0, "\n".join(map(json.dumps, mcp_events())), "")
    monkeypatch.setattr(verifier.subprocess, "run", run)
    result = verifier.run_verification("local", 8765, executable="codex", context_size=16384,
                                      task="mcp-fixture", mcp_python=str(python))
    assert result["status"] == "PASS"
    assert result["sandbox_read_preflight"] is None


def test_vision_fixture_is_a_valid_png_with_the_requested_stripe_order():
    import struct
    import zlib
    png = verifier.stripe_png(["blue", "white", "red"], width=6, height=2)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    idat = png.index(b"IDAT")
    size = struct.unpack(">I", png[idat - 4:idat])[0]
    pixels = zlib.decompress(png[idat + 4:idat + 4 + size])
    assert (width, height) == (6, 2) and pixels[0] == 0
    row = pixels[1:1 + 3 * width]
    assert [tuple(row[i:i + 3]) for i in (0, 6, 12)] == [
        verifier.VISION_COLORS[c] for c in ("blue", "white", "red")]
    assert len(set(tuple(verifier.random_stripe_order()) for _ in range(40))) > 1


def test_vision_assessment_needs_exact_order_and_no_tools():
    def events(text, tool=False):
        found = [{"type": "turn.completed", "usage": {}},
                 {"type": "item.completed", "item": {"type": "agent_message", "text": text}}]
        if tool:
            found.append({"type": "item.completed", "item": {"type": "command_execution"}})
        return found
    assert verifier.assess_vision(0, events("red green blue"), ["red", "green", "blue"])["strict_pass"]
    loose = verifier.assess_vision(0, events("Red, green, blue."), ["red", "green", "blue"])
    assert loose["strict_pass"] and not loose["exact_output_compliance"]
    assert not verifier.assess_vision(0, events("green red blue"), ["red", "green", "blue"])["strict_pass"]
    assert not verifier.assess_vision(0, events("red green blue", tool=True), ["red", "green", "blue"])["strict_pass"]
    assert not verifier.assess_vision(1, events("red green blue"), ["red", "green", "blue"])["strict_pass"]


def test_vision_image_flag_cannot_swallow_the_prompt(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        # Shared subprocess module can also serve background hardware polling
        # from other lifecycle tests. Only the Codex request is this assertion's
        # subject; an unrelated reading must not replace the captured command.
        if argv[:2] != ["codex", "exec"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        seen["argv"] = argv
        image = Path(argv[argv.index("-i") + 1])
        seen["png"] = image.read_bytes()[:8]
        return subprocess.CompletedProcess(argv, 1, "", "")
    monkeypatch.setattr(verifier.subprocess, "run", run)
    verifier.run_verification("local-test", 8765, executable="codex", context_size=4096, task="vision")
    argv = seen["argv"]
    assert argv[:2] == ["codex", "exec"] and argv[-1] == verifier.VISION_PROMPT
    assert argv[argv.index("-i") + 2].startswith("-")
    assert seen["png"] == b"\x89PNG\r\n\x1a\n"


def test_mcp_fixture_fields_are_judged_independently():
    # Real Ministral answer shape: statements instead of expressions, correct result.
    answer = json.dumps({"defective_expression": "return width * height",
                         "corrected_expression": "return 2 * (width + height)", "example_result": 16})
    events = [{"type": "turn.completed", "usage": {}},
              {"type": "item.completed", "item": {"type": "agent_message", "text": answer}}]
    checks = verifier.assess_mcp_fixture(0, events, {"a": "1"}, {"a": "1"})
    assert checks["example_result_verified"] is True
    assert checks["correct_defect_identification"] is False
    assert checks["corrected_expression_verified"] is False
    assert checks["strict_pass"] is False
