"""Claude Code history provider regression tests."""

import json

import pytest

from session_core.models import Session
from session_core.providers import claude
from session_core.providers.base import model_group_label


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-4-1", "Claude Code / Opus"),
        ("claude-sonnet-4", "Claude Code / Sonnet"),
        ("claude-haiku-3-5", "Claude Code / Haiku"),
        ("custom-local", "Claude Code / custom-local"),
        ("", "Claude Code / Unknown"),
    ],
)
def test_claude_model_grouping(model, expected):
    assert model_group_label("claude", model) == expected


def test_claude_provider_loads_previews_and_threads(tmp_path, monkeypatch):
    claude_dir = tmp_path / ".claude"
    projects = claude_dir / "projects"
    project_history = projects / "C--work-demo"
    project_history.mkdir(parents=True)
    session_id = "12345678-1234-1234-1234-123456789abc"
    transcript = project_history / f"{session_id}.jsonl"
    records = [
        {"type": "ai-title", "aiTitle": "Fix the widget"},
        {
            "type": "user",
            "cwd": str(tmp_path),
            "message": {"role": "user", "content": "Please fix it"},
        },
        {
            "type": "assistant",
            "message": {
                "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": "Done"}],
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        },
    ]
    transcript.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    history = claude_dir / "history.jsonl"
    history.write_text(
        json.dumps({"sessionId": session_id, "timestamp": 1234, "display": "old title"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(claude, "CLAUDE_DIR", claude_dir)
    monkeypatch.setattr(claude, "PROJECTS_DIR", projects)
    monkeypatch.setattr(claude, "HISTORY_FILE", history)

    provider = claude.ClaudeProvider()
    sessions = provider.load_sessions()
    assert len(sessions) == 1
    session = sessions[0]
    assert session.provider == "claude"
    assert session.display == "Fix the widget"
    assert session.model_group == "Claude Code / Sonnet"
    assert session.project == str(tmp_path)
    preview = provider.preview(session)
    assert preview.first == "Please fix it"
    assert preview.tokens.input == 11
    assert preview.tokens.output == 7
    assert [(item.role, item.text) for item in provider.collect_thread(session)] == [
        ("user", "Please fix it"),
        ("assistant", "Done"),
    ]


def test_claude_provider_resume_uses_claude_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(claude, "find_claude_exe", lambda: "claude")
    session = Session("session-42", "claude", str(tmp_path))
    command = claude.ClaudeProvider().resume_command(session)
    assert command.cwd == str(tmp_path)
    assert command.shell_command == "claude --no-chrome --resume 'session-42'"
