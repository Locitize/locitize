"""Opt-in real Codex completion against an already-running LOCITIZE router.

No service is started and no user configuration is edited. The isolated
CLI invocation uses only a loopback Responses provider and a harmless prompt.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

PLATFORM_DIR = Path(__file__).resolve().parent.parent
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from session_launch import build_local_launch

PROMPT = "Compute 19 + 23. Reply with exactly LOCITIZE_LOCAL_OK 42. Do not use tools."
EXPECTED = "LOCITIZE_LOCAL_OK 42"
FIXTURE = "def rectangle_area(width, height):\n    return width + height\n"
FIXTURE_EXPECTED = "DEFECT: width + height; FIX: width * height"
FIXTURE_PROMPT = (
    "Read area.py by running exactly Get-Content -LiteralPath area.py in PowerShell. "
    "Use the default sandbox, do not request elevated permissions, and stop if the read is blocked. "
    "Its function must compute rectangle area. "
    "Identify the defect and corrected expression. Do not edit or create files. "
    "Reply with exactly DEFECT: width + height; FIX: width * height."
)
MCP_FIXTURE = "def rectangle_perimeter(width, height):\n    return width * height\n"
MCP_PROMPT = (
    "Use the fixture_reader read_fixture MCP tool to read the Python fixture. "
    "Do not use shell commands or any other tools. Do not change files. "
    "Identify the bug in rectangle_perimeter and its correction. "
    "Return JSON with defective_expression, corrected_expression, and example_result "
    "(a single integer) for the corrected function with width=3 and height=5. "
    "Reply with only that JSON object: no comments, explanation or other text. Stop if the tool read fails."
)


VISION_COLORS = {"red": (220, 30, 30), "green": (30, 160, 60), "blue": (30, 60, 220),
                 "yellow": (240, 220, 30), "black": (0, 0, 0), "white": (255, 255, 255)}
VISION_PROMPT = (
    "Look at the attached image. It shows three vertical color stripes. Do not use any tools. "
    "Reply with only the three color names from left to right, lowercase, separated by single "
    "spaces, choosing from: red, green, blue, yellow, black, white."
)


def stripe_png(order: list[str], width: int = 384, height: int = 128) -> bytes:
    """Stdlib-only PNG of equal vertical stripes; 120 orders make guessing unlikely."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    band = width // len(order)
    row = b"\x00" + b"".join(bytes(VISION_COLORS[order[min(x // band, len(order) - 1)]])
                             for x in range(width))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b""))


def random_stripe_order() -> list[str]:
    return secrets.SystemRandom().sample(sorted(VISION_COLORS), 3)


def build_verification_command(project: str, model_id: str, port: int,
                               executable: str = "codex", context_size: int | None = None
                               ) -> tuple[list[str], dict[str, str]]:
    argv, env = build_local_launch("codex", project, model_id, port)
    argv[0] = executable
    argv.insert(1, "exec")
    if context_size is not None:
        if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size < 1:
            raise ValueError("Context size must be a positive measured integer")
        argv += ["-c", f"model_context_window={context_size}"]
    argv += ["--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
             "--sandbox", "read-only", "--json", "--color", "never", PROMPT]
    return argv, env


def run_verification(model_id: str, port: int, timeout_s: float = 180.0,
                     executable: str | None = None, context_size: int | None = None,
                     task: str = "arithmetic", mcp_python: str | None = None) -> dict:
    """Run once after the caller has made the local Responses router ready."""
    executable = executable or shutil.which("codex")
    if not executable:
        return {"status": "FAIL", "reason": "Codex CLI is not installed"}
    if timeout_s <= 0:
        raise ValueError("Timeout must be positive")
    if task not in ("arithmetic", "fixture", "mcp-fixture", "vision"):
        raise ValueError("Unknown verification task")
    if task in ("fixture", "mcp-fixture") and context_size is None:
        raise ValueError("Fixture verification requires the measured running context")
    with tempfile.TemporaryDirectory(prefix="locitize_codex_check_") as temporary:
        root = Path(temporary)
        project = root / "workspace"
        project.mkdir()
        if task == "fixture":
            (project / "area.py").write_text(FIXTURE, encoding="ascii")
        if task == "mcp-fixture":
            (project / "perimeter.py").write_text(MCP_FIXTURE, encoding="ascii")
        expected_colors = None
        if task == "vision":
            # Outside the workspace: the image is Codex input, not a project file.
            expected_colors = random_stripe_order()
            image = root / "stripes.png"
            image.write_bytes(stripe_png(expected_colors))
        before = snapshot_workspace(project)
        codex_home = root / "codex-home"
        codex_home.mkdir()
        argv, overrides = build_verification_command(str(project), model_id, port,
                                                     executable, context_size)
        if task == "fixture":
            argv[-1] = FIXTURE_PROMPT
        if task == "mcp-fixture":
            mcp_python = mcp_python or str(PLATFORM_DIR.parent / ".plugin-venv" / "Scripts" / "python.exe")
            if not Path(mcp_python).is_file():
                return {"status": "FAIL", "reason": "MCP test runtime is not installed"}
            argv[-1] = MCP_PROMPT
            argv[2:2] = ["-c", "mcp_servers.fixture_reader.command=" + json.dumps(mcp_python),
                         "-c", "mcp_servers.fixture_reader.args=" + json.dumps([
                             str(Path(__file__).with_name("codex_fixture_reader.py")),
                             str(project / "perimeter.py")])]
        if task == "vision":
            # -i accepts several files; place it before a flag so the prompt
            # stays the positional argument instead of a second image path.
            argv[2:2] = ["-i", str(image)]
            argv[-1] = VISION_PROMPT
        env = os.environ.copy()
        env.update(overrides)
        env["CODEX_HOME"] = str(codex_home)
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            env.pop(key, None)
        preflight = (sandbox_read_preflight(executable, project, env)
                     if task == "fixture" else None)
        try:
            completed = subprocess.run(argv, cwd=project, env=env, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace",
                                       stdin=subprocess.DEVNULL,
                                       timeout=timeout_s,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return {"status": "FAIL", "reason": "Codex local completion timed out",
                    "model_id": model_id, "port": port}
        except OSError as exc:
            return {"status": "FAIL", "reason": str(exc)}
        events = []
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        answers = [event["item"].get("text", "") for event in events
                   if event.get("type") == "item.completed"
                   and isinstance(event.get("item"), dict)
                   and event["item"].get("type") == "agent_message"]
        turn_completed = any(event.get("type") == "turn.completed" for event in events)
        checks = (assess_fixture(completed.returncode, events, before, snapshot_workspace(project))
                  if task == "fixture" else assess_completion(completed.returncode, events))
        if task == "mcp-fixture":
            checks = assess_mcp_fixture(completed.returncode, events, before, snapshot_workspace(project))
        if task == "vision":
            checks = assess_vision(completed.returncode, events, expected_colors)
        return {"status": "PASS" if checks["strict_pass"] else "FAIL", "model_id": model_id,
                "port": port, "provider": "locitize", "wire_api": "responses",
                "returncode": completed.returncode, "turn_completed": turn_completed,
                "answers": answers, "events": events,
                "checks": checks, "context_size_override": context_size,
                "task": task,
                "sandbox_read_preflight": preflight,
                "command_policy_blocked": "blocked by policy" in completed.stderr.lower(),
                "stderr_tail": completed.stderr[-4000:],
                "isolated_config": True, "global_config_modified": False}


def sandbox_read_preflight(executable: str, project: Path, env: dict[str, str]) -> dict:
    """Check native sandbox read ACLs; this is not a model tool-read pass."""
    if sys.platform != "win32":
        return {"status": "SKIP", "reason": "Windows sandbox diagnostic only"}
    shell = shutil.which("powershell.exe")
    if not shell:
        return {"status": "SKIP", "reason": "PowerShell not found"}
    argv = [executable, "sandbox", "-P", ":read-only", "-C", str(project),
            shell, "-NoProfile", "-Command", "Get-Content -LiteralPath area.py"]
    try:
        result = subprocess.run(argv, env=env, cwd=project, capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                stdin=subprocess.DEVNULL, timeout=20,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "FAIL", "reason": str(exc)}
    success = result.returncode == 0 and FIXTURE.strip() in result.stdout.replace("\r\n", "\n")
    return {"status": "PASS" if success else "FAIL", "returncode": result.returncode,
            "fixture_read": success, "stderr_tail": result.stderr[-2000:],
            "model_tool_evidence": False}


def snapshot_workspace(project: Path) -> dict[str, str]:
    """Detect deleted, changed, or newly created files throughout the workspace."""
    return {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in project.rglob("*") if path.is_file()}


def assess_fixture(returncode: int, events: list[dict], before: dict, after: dict) -> dict:
    checks = assess_completion(returncode, events)
    answers = [event["item"].get("text", "") for event in events
               if event.get("type") == "item.completed"
               and isinstance(event.get("item"), dict)
               and event["item"].get("type") == "agent_message"]
    reads = [event["item"] for event in events
             if event.get("type") == "item.completed"
             and isinstance(event.get("item"), dict)
             and event["item"].get("type") == "command_execution"
             and event["item"].get("exit_code") == 0
             and "area.py" in event["item"].get("command", "")
             and FIXTURE.strip() in event["item"].get("aggregated_output", "").replace("\r\n", "\n")]
    identified = any(FIXTURE_EXPECTED in answer for answer in answers)
    exact = answers == [FIXTURE_EXPECTED]
    unchanged = before == after
    no_edit_events = "file_change" not in checks["tool_item_types"]
    checks.pop("correct_result_marker")
    checks.pop("no_tools_used")
    checks.update({"successful_fixture_read": bool(reads), "correct_defect_identification": identified,
                   "exact_output_compliance": exact, "workspace_unchanged": unchanged,
                   "no_file_change_events": no_edit_events,
                   "workspace_before": before, "workspace_after": after,
                   "strict_pass": checks["connectivity_and_completion"] and bool(reads)
                   and identified and exact and unchanged and no_edit_events})
    return checks


def expression_value(expression: str, width: int, height: int) -> int:
    """Interpret a tiny arithmetic AST without executing model-produced code."""
    def value(node):
        if isinstance(node, ast.Name) and node.id in ("width", "height"):
            return {"width": width, "height": height}[node.id]
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 100:
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            return left * right
        raise ValueError("Unsupported expression")
    if not isinstance(expression, str) or len(expression) > 100:
        raise ValueError("Invalid expression")
    return value(ast.parse(expression, mode="eval").body)


def assess_mcp_fixture(returncode: int, events: list[dict], before: dict, after: dict) -> dict:
    checks = assess_completion(returncode, events)
    items = [event["item"] for event in events if event.get("type") == "item.completed"
             and isinstance(event.get("item"), dict)]
    reads = [item for item in items if item.get("type") == "mcp_tool_call"
             and item.get("server") == "fixture_reader" and item.get("tool") == "read_fixture"
             and item.get("status") == "completed" and item.get("error") is None
             and isinstance(item.get("result"), dict) and not item["result"].get("isError", False)
             and any(isinstance(content, dict) and content.get("type") == "text"
                     and MCP_FIXTURE.strip() in content.get("text", "").replace("\r\n", "\n")
                     for content in item["result"].get("content", []))]
    answers = [item.get("text", "") for item in items if item.get("type") == "agent_message"]
    structured = None
    exact_json = False
    for answer in reversed(answers):
        candidate = answer.strip()
        if candidate.startswith("```json") and candidate.endswith("```"):
            candidate = candidate[7:-3].strip()
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            structured = parsed
            exact_json = candidate == answer.strip()
            break
    def field_check(check) -> bool:
        # Each field is judged on its own: one malformed field must not
        # hide whether the others were right.
        try:
            return bool(check())
        except (SyntaxError, ValueError, TypeError):
            return False

    defective = corrected = example = False
    if structured:
        defective = field_check(lambda: ast.dump(ast.parse(structured.get("defective_expression", ""), mode="eval"))
                                == ast.dump(ast.parse("width * height", mode="eval")))
        corrected = field_check(lambda: all(
            expression_value(structured.get("corrected_expression"), w, h) == 2 * (w + h)
            for w, h in ((3, 5), (2, 7), (4, 4))))
        example = type(structured.get("example_result")) is int and structured["example_result"] == 16
    forbidden = [item.get("type") for item in items if item.get("type") in (
        "command_execution", "file_change", "web_search") or
        (item.get("type") == "mcp_tool_call" and
         (item.get("server") != "fixture_reader" or item.get("tool") != "read_fixture"))]
    checks.pop("correct_result_marker")
    checks.pop("no_tools_used")
    checks.update({"successful_fixture_read": bool(reads), "correct_defect_identification": defective,
                   "corrected_expression_verified": corrected, "example_result_verified": example,
                   "json_only_compliance": exact_json, "parsed_answer": structured,
                   "exact_output_compliance": None, "workspace_unchanged": before == after,
                   "workspace_before": before, "workspace_after": after,
                   "forbidden_tool_events": forbidden,
                   "strict_pass": checks["connectivity_and_completion"] and bool(reads)
                   and defective and corrected and example and before == after and not forbidden})
    return checks


def assess_vision(returncode: int, events: list[dict], expected: list[str]) -> dict:
    """Exact left-to-right colors; content is checked separately from formatting."""
    checks = assess_completion(returncode, events)
    answers = [event["item"].get("text", "") for event in events
               if event.get("type") == "item.completed" and isinstance(event.get("item"), dict)
               and event["item"].get("type") == "agent_message"]
    final = answers[-1].strip().strip("`").strip().lower() if answers else ""
    words = final.replace(",", " ").replace(".", " ").split()
    correct = words == expected
    checks.pop("correct_result_marker")
    checks.update({"expected_order": expected, "answer_colors": words, "correct_colors": correct,
                   "exact_output_compliance": final == " ".join(expected),
                   "strict_pass": checks["connectivity_and_completion"] and correct
                   and checks["no_tools_used"]})
    return checks


def assess_completion(returncode: int, events: list[dict]) -> dict:
    """Keep protocol success separate from the model's output compliance."""
    answers = [event["item"].get("text", "") for event in events
               if event.get("type") == "item.completed"
               and isinstance(event.get("item"), dict)
               and event["item"].get("type") == "agent_message"]
    completed = [event for event in events if event.get("type") == "turn.completed"]
    transport = returncode == 0 and bool(completed) and any(answers)
    exact = answers == [EXPECTED]
    marker = any(EXPECTED in answer for answer in answers)
    tool_items = [event["item"].get("type") for event in events
                  if isinstance(event.get("item"), dict)
                  and event["item"].get("type") in (
                      "command_execution", "mcp_tool_call", "web_search", "file_change")]
    warnings = [event["item"].get("message", "") for event in events
                if isinstance(event.get("item"), dict)
                and event["item"].get("type") == "error"]
    return {"connectivity_and_completion": transport, "correct_result_marker": marker,
            "exact_output_compliance": exact, "no_tools_used": not tool_items,
            "strict_pass": transport and exact and not tool_items,
            "metadata_fallback": any("metadata" in message.lower() for message in warnings),
            "warnings": warnings, "tool_item_types": tool_items,
            "usage": completed[-1].get("usage", {}) if completed else {}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True, help="Running LOCITIZE router port")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--context-size", type=int, help="Measured running model context")
    parser.add_argument("--task", choices=("arithmetic", "fixture", "mcp-fixture", "vision"), default="arithmetic")
    parser.add_argument("--mcp-python", help="Python runtime with the MCP SDK installed")
    args = parser.parse_args()
    result = run_verification(args.model, args.port, args.timeout,
                              context_size=args.context_size, task=args.task, mcp_python=args.mcp_python)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
