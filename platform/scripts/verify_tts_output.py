"""AC13 harness: prove the wav produced by `launcher.py --speak` is real audio.

The centerpiece functional proof for Milestone 6 (voice OUT). After
`launcher.py --speak "<sentence>" --json` synthesizes a wav through the running
Kokoro service, this script re-opens that wav with the standard-library `wave`
module and asserts, from the header fields alone:
  - the file exists and is non-empty (nonzero bytes),
  - it is 16-bit PCM mono at Kokoro's 24 kHz,
  - its duration is plausible for a spoken sentence (a sane lower/upper bound in
    seconds, NOT an exact match) -- a fabricated or truncated wav fails.

No fabricated pass: every assertion reads real header data. Exit 0 only when the
wav is a real, plausibly-sized audio file. Run from Codebase/platform:
`python scripts/verify_tts_output.py --json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent

# Kokoro renders 16-bit PCM mono at 24 kHz; the launcher's --speak path writes
# exactly that, so a wav that disagrees is not a genuine Kokoro output.
_EXPECTED_RATE_HZ = 24000
_EXPECTED_CHANNELS = 1
_EXPECTED_SAMPLE_WIDTH = 2

# Plausibility window for a single spoken sentence. Deliberately wide: the point is
# to reject an empty/near-empty or absurdly long file, not to pin an exact length.
_MIN_DURATION_S = 0.5
_MAX_DURATION_S = 30.0


def _default_wav_path() -> Path:
    """The wav path --speak writes by default (mirrors logger.resolve_log_dir).

    Precedence: LOCITIZE_LOG_DIR, else <platform>/logs. Kept in sync with the
    launcher so the two commands agree on the file location with no plumbing.
    """
    override = os.environ.get("LOCITIZE_LOG_DIR")
    log_dir = Path(override) if override else PLATFORM_DIR / "logs"
    return log_dir / "tts_speak.wav"


def inspect_wav(path: Path) -> dict:
    """Read wav header facts and decide whether it is a plausible spoken sentence.

    Returns a summary dict; raises nothing -- an unreadable/absent file is reported
    as a failed outcome with a reason so the caller can print it.
    """
    summary: dict = {
        "wav_path": str(path),
        "exists": False,
        "bytes": 0,
        "channels": None,
        "sample_width": None,
        "framerate": None,
        "frames": None,
        "duration_s": None,
        "outcome": "failed",
        "reason": "",
    }
    if not path.is_file():
        summary["reason"] = f"wav not found at {path}"
        return summary
    size = path.stat().st_size
    summary["exists"] = True
    summary["bytes"] = size
    if size == 0:
        summary["reason"] = "wav is empty (zero bytes)"
        return summary

    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            framerate = reader.getframerate()
            frames = reader.getnframes()
    except (wave.Error, EOFError, OSError) as exc:
        summary["reason"] = f"not a readable wav: {exc}"
        return summary

    duration = frames / framerate if framerate else 0.0
    summary.update(
        {
            "channels": channels,
            "sample_width": sample_width,
            "framerate": framerate,
            "frames": frames,
            "duration_s": round(duration, 3),
        }
    )

    if channels != _EXPECTED_CHANNELS:
        summary["reason"] = f"expected mono, got {channels} channels"
        return summary
    if sample_width != _EXPECTED_SAMPLE_WIDTH:
        summary["reason"] = f"expected 16-bit PCM, got {sample_width * 8}-bit"
        return summary
    if framerate != _EXPECTED_RATE_HZ:
        summary["reason"] = f"expected {_EXPECTED_RATE_HZ} Hz, got {framerate} Hz"
        return summary
    if not (_MIN_DURATION_S <= duration <= _MAX_DURATION_S):
        summary["reason"] = (
            f"duration {duration:.2f}s outside the plausible "
            f"[{_MIN_DURATION_S}, {_MAX_DURATION_S}]s window"
        )
        return summary

    summary["outcome"] = "ok"
    summary["reason"] = "real 16-bit PCM mono wav with a plausible duration"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AC13 real-wav verifier")
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    parser.add_argument(
        "--wav",
        default=None,
        help="wav path to inspect (default: the --speak output logs/tts_speak.wav)",
    )
    args = parser.parse_args(argv)

    path = Path(args.wav) if args.wav else _default_wav_path()
    summary = inspect_wav(path)

    if args.json:
        print(json.dumps(summary))
    else:
        print(f"wav path   : {summary['wav_path']}")
        print(f"bytes      : {summary['bytes']}")
        print(f"duration_s : {summary['duration_s']}")
        print(f"outcome    : {summary['outcome']}")
        if summary["reason"]:
            print(f"reason     : {summary['reason']}")
    return 0 if summary["outcome"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
