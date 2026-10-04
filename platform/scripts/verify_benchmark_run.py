"""AC14 post-run verification: did a real single-model benchmark append a valid row?

After `launcher.py --benchmark --model <id> --runs N`, this scans
<data root>/reports/benchmark_results.jsonl BACKWARD for the TARGET model's most recent ok:true
record (F-1: not the last line overall - a later run for a different model must not
mask an earlier good row for <id>) and asserts it is a successful record: ok is
true and the speed fields are non-null (they always are when ok is true, because
speeds come only from real server timings, M5.2). It then confirms the human file
<data root>/reports/benchmark_results.md carries that model under the run section matching that
row's session (F-1: the winning row's own session, not blindly the newest section),
proving the markdown grew a matching record too.

This is the engine's end-to-end proof: a real model started, a real /completion
timing was measured, and both result files gained an honest record. It does NOT
judge the 27B performance target (that is AC15 / verify_27b_target.py).

Run from Codebase/platform:
  `python scripts/verify_benchmark_run.py --model <id> [--json]`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLATFORM_DIR))


def results_paths() -> tuple[Path, Path]:
    """The (jsonl, md) result files, resolved exactly as the runner writes them.

    DEC-M14-9 moved benchmark output from <install>/docs to <data root>/reports,
    so this gate resolves through benchmark.resolve_results_dir rather than
    rebuilding the path itself: two copies of a path is how a checker ends up
    reading a file the product no longer writes. Loaded lazily inside main() so
    importing this module has no side effects.
    """
    from benchmark import resolve_results_dir
    from config import Config

    settings, _models, _issues = Config.load()
    results = resolve_results_dir(settings)
    return results / "benchmark_results.jsonl", results / "benchmark_results.md"


def latest_ok_record_for_model(path: Path, model_id: str) -> dict | None:
    """The target model's most recent ok:true JSONL record, scanning backward.

    F-1: iterate the file's lines in reverse and return the first record whose
    model_id matches AND ok is true. A later run for a different model (or a later
    honest failure for the same model) therefore cannot mask an earlier good row.
    Malformed lines are skipped. Returns None if no such record exists.
    """
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("model_id") == model_id and record.get("ok") is True:
            return record
    return None


def md_section_for_session_has_model(path: Path, session_id: str, model_id: str) -> bool:
    """True if the md '## Run <session_id>' section names the model (F-1).

    Matches the markdown section belonging to the winning row's own session rather
    than blindly reading the newest section, so a later run's section cannot make an
    earlier model's verification pass or fail spuriously. If the session cannot be
    located (older row without a session header), falls back to any section that
    names the model.
    """
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if session_id:
        marker = f"\n## Run {session_id}"
        idx = text.find(marker)
        if idx != -1:
            # Bound the search to this section (up to the next '## Run ' or EOF).
            nxt = text.find("\n## Run ", idx + len(marker))
            section = text[idx:] if nxt == -1 else text[idx:nxt]
            return model_id in section
    # Fallback: the model appears in some run section at all.
    return ("\n## Run " in text) and (model_id in text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify_benchmark_run")
    parser.add_argument("--model", required=True, help="the benchmarked model id")
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args(argv)

    jsonl_path, md_path = results_paths()
    record = latest_ok_record_for_model(jsonl_path, args.model)
    speeds_present = bool(
        record
        and record.get("prompt_per_second") is not None
        and record.get("predicted_per_second") is not None
    )
    session_id = record.get("session_id", "") if record else ""
    md_ok = bool(record) and md_section_for_session_has_model(
        md_path, session_id, args.model
    )
    ok = bool(record) and speeds_present and md_ok

    report = {
        "ok": ok,
        "model": args.model,
        "found_ok_record": bool(record),
        "record_session_id": session_id or None,
        "record_timestamp": record.get("timestamp") if record else None,
        "predicted_per_second": record.get("predicted_per_second") if record else None,
        "prompt_per_second": record.get("prompt_per_second") if record else None,
        "markdown_section_has_model": md_ok,
    }
    if args.json:
        print(json.dumps(report))
    elif ok:
        print(
            f"PASS: latest ok benchmark row for {args.model} "
            f"(session {session_id}, {record.get('timestamp')}), "
            f"gen {record['predicted_per_second']} tok/s, prompt "
            f"{record['prompt_per_second']} tok/s; matching markdown section present."
        )
    else:
        print(f"FAIL: no verifiable ok benchmark record found for {args.model}:")
        print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
