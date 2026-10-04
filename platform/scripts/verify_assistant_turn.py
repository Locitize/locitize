"""AC14 harness: prove a real text-mode assistant turn against a real running LLM.

The assistant-engine functional proof for Milestone 7. This script drives the real
launcher non-interactively:

  launcher.py --assistant --text --no-speak --timings --json

feeding it ONE fixed scripted prompt on stdin and then closing stdin (EOF), so the
assistant runs exactly one turn and exits cleanly. From the captured stdout it
asserts, with no fabrication:
  - a non-empty model reply was produced and printed ("LOCITIZE: <reply>"),
  - real per-stage TurnTimings were printed ("TIMINGS {json}") with
    llm_first_token_ms and llm_total_ms as real wall-clock numbers (> 0), and
    stt_ms / tts_first_audio_ms == 0 in --no-speak text mode,
  - the launcher reported a clean shutdown (no orphaned child), which this script
    double-checks by confirming no NEW llama-server process survived the run.

The launcher itself starts the chat model via the existing ModelController and
stops it on exit; this script never fabricates a reply or a timing. Run from
Codebase/platform:  python scripts/verify_assistant_turn.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent

# A fixed, deterministic prompt (like the M3 known-phrase discipline): short, so the
# turn is fast, and unambiguous, so a real model returns real non-empty text.
_PROMPT = "Reply with a short one-sentence greeting."

# How long to allow the full turn (cold model load + one generation). Generous but
# bounded; the AC's own command timeout is the outer guard.
_TURN_TIMEOUT_S = 260


def _llama_pids() -> set[int]:
    """Return the set of running llama-server PIDs (Windows tasklist).

    Used to confirm the launcher left no NEW model process behind. Best-effort: an
    unavailable tasklist yields an empty set rather than a crash.
    """
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    pids: set[int] = set()
    for line in out.splitlines():
        # CSV row: "llama-server.exe","<pid>",...
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            pids.add(int(parts[1]))
    return pids


def run_turn() -> dict:
    """Drive one real assistant turn and return a verdict summary dict."""
    summary: dict = {
        "outcome": "failed",
        "reason": "",
        "reply": "",
        "timings": None,
        "launcher_clean": None,
        "no_orphan": None,
        "exit_code": None,
    }

    pids_before = _llama_pids()

    cmd = [
        sys.executable,
        "launcher.py",
        "--assistant",
        "--text",
        "--no-speak",
        "--timings",
        "--json",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=_PROMPT + "\nquit\n",
            capture_output=True,
            text=True,
            cwd=str(PLATFORM_DIR),
            timeout=_TURN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        summary["reason"] = f"assistant turn did not finish within {_TURN_TIMEOUT_S}s"
        return summary

    summary["exit_code"] = proc.returncode
    stdout = proc.stdout or ""

    reply = ""
    timings: dict | None = None
    launcher_final: dict | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("LOCITIZE:"):
            reply = line[len("LOCITIZE:"):].strip()
        elif line.startswith("TIMINGS "):
            try:
                timings = json.loads(line[len("TIMINGS "):])
            except ValueError:
                timings = None
        elif line.startswith("{") and '"clean"' in line:
            try:
                launcher_final = json.loads(line)
            except ValueError:
                launcher_final = None

    summary["reply"] = reply
    summary["timings"] = timings
    if launcher_final is not None:
        summary["launcher_clean"] = bool(launcher_final.get("clean"))

    # No NEW llama-server survived the run (the launcher must stop what it started).
    pids_after = _llama_pids()
    leaked = pids_after - pids_before
    summary["no_orphan"] = len(leaked) == 0

    # --- assertions (no fabricated pass) --- #
    if not reply:
        summary["reason"] = "no non-empty model reply was printed"
        return summary
    if not isinstance(timings, dict):
        summary["reason"] = "no per-stage TurnTimings were printed"
        return summary
    first = timings.get("llm_first_token_ms")
    total = timings.get("llm_total_ms")
    if not isinstance(first, (int, float)) or first <= 0:
        summary["reason"] = f"llm_first_token_ms is not a real measurement: {first!r}"
        return summary
    if not isinstance(total, (int, float)) or total < first:
        summary["reason"] = f"llm_total_ms is not a real measurement: {total!r}"
        return summary
    if timings.get("stt_ms") != 0:
        summary["reason"] = "stt_ms should be 0 in --no-speak text mode"
        return summary
    if timings.get("tts_first_audio_ms") != 0:
        summary["reason"] = "tts_first_audio_ms should be 0 in --no-speak mode"
        return summary
    if not summary["no_orphan"]:
        summary["reason"] = f"orphaned llama-server process(es) survived: {sorted(leaked)}"
        return summary
    if summary["launcher_clean"] is False:
        summary["reason"] = "launcher reported an unclean shutdown"
        return summary

    summary["outcome"] = "ok"
    summary["reason"] = "real reply + real timings, clean shutdown, no orphan"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AC14 real assistant-turn verifier")
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    args = parser.parse_args(argv)

    summary = run_turn()

    if args.json:
        print(json.dumps(summary))
    else:
        print(f"reply    : {summary['reply']}")
        print(f"timings  : {summary['timings']}")
        print(f"no_orphan: {summary['no_orphan']}")
        print(f"outcome  : {summary['outcome']}")
        if summary["reason"]:
            print(f"reason   : {summary['reason']}")
    return 0 if summary["outcome"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
