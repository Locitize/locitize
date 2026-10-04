"""AC5 harness: prove real speech-to-text end to end against a known-phrase WAV.

The centerpiece functional proof for Milestone 3. This script:
  1. Ensures a deterministic ground-truth fixture WAV exists, generating it on
     demand with Windows SAPI text-to-speech (no external download) speaking a
     fixed phrase.
  2. Drives the real transcription path by shelling to
     `launcher.py --transcribe <wav> --json`, which starts whisper-server if
     needed, sends the audio to its /inference endpoint, and cleans up.
  3. Asserts the known phrase appears in the returned transcript under a
     punctuation/case-insensitive normalization, printing the actual transcript
     either way so a failure is diagnosable.

Exit 0 only if the phrase is found. Run from Codebase/platform:
`python scripts/verify_transcription.py --json`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent
FIXTURE_DIR = PLATFORM_DIR / "tests" / "fixtures"
FIXTURE_WAV = FIXTURE_DIR / "known_phrase.wav"

# The fixed ground-truth phrase: chosen for broad phonetic coverage and zero
# homophone ambiguity, so a correct transcript is unmistakable.
KNOWN_PHRASE = "the quick brown fox jumps over the lazy dog"


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, and collapse whitespace for a fuzzy compare.

    whisper capitalizes and adds a trailing period; normalizing both sides lets us
    check the phrase as a substring without being defeated by punctuation/casing.
    """
    lowered = text.lower()
    stripped = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def ensure_fixture() -> None:
    """Generate the known-phrase WAV via Windows SAPI if it is not already present.

    Uses System.Speech through PowerShell -- present on every Windows install, no
    download. Deterministic: the same phrase, same synthesizer, every run.
    """
    if FIXTURE_WAV.is_file() and FIXTURE_WAV.stat().st_size > 0:
        return
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    # Build the PowerShell one-liner that speaks KNOWN_PHRASE into the WAV file.
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 0; "
        f"$s.SetOutputToWaveFile('{FIXTURE_WAV.as_posix()}'); "
        f"$s.Speak('{KNOWN_PHRASE}'); "
        "$s.Dispose()"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not FIXTURE_WAV.is_file():
        raise RuntimeError(
            f"failed to generate fixture WAV via SAPI: {result.stderr.strip()}"
        )


def run_transcribe() -> dict:
    """Shell to launcher.py --transcribe and return the parsed JSON record.

    Uses the same interpreter running this script (the platform venv) so the CLI
    path is exercised exactly as a QA operator would run it.
    """
    proc = subprocess.run(
        [sys.executable, "launcher.py", "--transcribe", str(FIXTURE_WAV), "--json"],
        cwd=str(PLATFORM_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    # The JSON record is the last non-empty stdout line; anything else is noise.
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(
            f"--transcribe produced no output (stderr: {proc.stderr.strip()})"
        )
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"could not parse --transcribe JSON: {exc}; raw: {lines[-1]!r}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AC5 real-transcription harness")
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    args = parser.parse_args(argv)

    summary: dict = {
        "known_phrase": KNOWN_PHRASE,
        "transcript": "",
        "matched": False,
        "outcome": "failed",
        "reason": "",
    }
    try:
        ensure_fixture()
        record = run_transcribe()
    except RuntimeError as exc:
        summary["reason"] = str(exc)
        _emit(summary, args.json)
        return 1

    summary["transcript"] = record.get("transcript", "")
    if record.get("outcome") != "ok":
        summary["reason"] = record.get("reason", "transcription did not succeed")
        _emit(summary, args.json)
        return 1

    matched = _normalize(KNOWN_PHRASE) in _normalize(summary["transcript"])
    summary["matched"] = matched
    summary["outcome"] = "ok" if matched else "failed"
    if not matched:
        summary["reason"] = "known phrase not found in transcript"
    _emit(summary, args.json)
    return 0 if matched else 1


def _emit(summary: dict, as_json: bool) -> None:
    """Print the result, always showing the actual transcript for diagnosis."""
    if as_json:
        print(json.dumps(summary))
        return
    print(f"known phrase : {summary['known_phrase']}")
    print(f"transcript   : {summary['transcript']}")
    print(f"matched      : {summary['matched']}")
    if summary["reason"]:
        print(f"reason       : {summary['reason']}")


if __name__ == "__main__":
    sys.exit(main())
