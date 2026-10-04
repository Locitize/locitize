"""AC15 harness: prove a real single-image vision Q&A against qwen2-5-vl + mmproj.

The vision-engine functional proof for Milestone 8. This script drives the real
launcher non-interactively:

  launcher.py --describe <fixture> --prompt "What color is the shape ...?" --json

against the Builder-generated, checked-in test image
tests/fixtures/vision_red_square.png (a bold RED square on white, drawn by a pure
stdlib PNG writer -- our own deterministic ground truth, not a downloaded asset).
From the captured JSON it asserts, with no fabrication, that the REAL model answer
text contains the expected keyword "red". The launcher switches to qwen2-5-vl WITH
--mmproj via the existing ModelController and stops it on exit; this script also
double-checks that no NEW llama-server process survived the run (VRAM returns to
baseline, no orphan).

Run from Codebase/platform:  python scripts/verify_vision_describe.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent
_FIXTURE = PLATFORM_DIR / "tests" / "fixtures" / "vision_red_square.png"

# The expected keyword the real model must produce for our deterministic image. The
# fixture is a saturated pure-red (255,0,0) square, which a VL model reliably names
# "red"; we match case-insensitively so "Red"/"RED" also count.
_EXPECTED_KEYWORD = "red"

# A prompt that steers the model to name the dominant color, keeping the check fair
# and unambiguous without leaking the answer.
_PROMPT = "What is the main color of the shape in this image? Answer in one short sentence."

# Generous but bounded: a cold VL model load + one multimodal generation.
_DESCRIBE_TIMEOUT_S = 290


def _llama_pids() -> set[int]:
    """Return running llama-server PIDs (Windows tasklist); best-effort empty set."""
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
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            pids.add(int(parts[1]))
    return pids


def run_describe() -> dict:
    """Drive one real vision describe and return a verdict summary dict."""
    summary: dict = {
        "outcome": "failed",
        "reason": "",
        "answer": "",
        "keyword": _EXPECTED_KEYWORD,
        "keyword_found": None,
        "no_orphan": None,
        "exit_code": None,
    }

    if not _FIXTURE.is_file():
        summary["reason"] = f"test fixture missing: {_FIXTURE}"
        return summary

    pids_before = _llama_pids()

    cmd = [
        sys.executable,
        "launcher.py",
        "--describe",
        str(_FIXTURE),
        "--prompt",
        _PROMPT,
        "--json",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PLATFORM_DIR),
            timeout=_DESCRIBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        summary["reason"] = f"describe did not finish within {_DESCRIBE_TIMEOUT_S}s"
        return summary

    summary["exit_code"] = proc.returncode

    # The launcher prints a single JSON record for --describe --json. Find it.
    record: dict | None = None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{") and '"answer"' in line:
            try:
                record = json.loads(line)
            except ValueError:
                record = None
    if record is None:
        summary["reason"] = "no vision JSON record was printed"
        summary["stderr_tail"] = (proc.stderr or "")[-400:]
        return summary

    answer = str(record.get("answer", ""))
    summary["answer"] = answer

    # No NEW llama-server survived (VRAM back to baseline; the launcher stops what it
    # started -- the M8.2 one-model-at-a-time discipline).
    leaked = _llama_pids() - pids_before
    summary["no_orphan"] = len(leaked) == 0

    # --- assertions (no fabricated pass) --- #
    if record.get("outcome") != "ok" or not answer:
        summary["reason"] = f"vision describe failed: {record.get('reason') or 'no answer'}"
        return summary
    found = _EXPECTED_KEYWORD in answer.casefold()
    summary["keyword_found"] = found
    if not found:
        summary["reason"] = (
            f"expected keyword '{_EXPECTED_KEYWORD}' not in real answer: {answer!r}"
        )
        return summary
    if not summary["no_orphan"]:
        summary["reason"] = f"orphaned llama-server survived: {sorted(leaked)}"
        return summary

    summary["outcome"] = "ok"
    summary["reason"] = "real model answer contains the expected keyword, no orphan"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AC15 real vision-describe verifier")
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    args = parser.parse_args(argv)

    summary = run_describe()

    if args.json:
        print(json.dumps(summary))
    else:
        print(f"answer       : {summary['answer']}")
        print(f"keyword_found: {summary['keyword_found']} ('{summary['keyword']}')")
        print(f"no_orphan    : {summary['no_orphan']}")
        print(f"outcome      : {summary['outcome']}")
        if summary["reason"]:
            print(f"reason       : {summary['reason']}")
    return 0 if summary["outcome"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
