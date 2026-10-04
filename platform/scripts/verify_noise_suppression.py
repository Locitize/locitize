"""Measure LOCITIZE's real FFmpeg noise preset with deterministic audio.

The verifier creates synthetic voice-band and steady-noise WAV fixtures using
only the standard library, runs the production AudioNoiseSuppressor, and reports
signal retention, noise reduction, and five-run p95 latency. It exits nonzero
when any approved quality or latency threshold is missed.
"""

from __future__ import annotations

import argparse
import array
import io
import json
import math
import random
import statistics
import sys
import tempfile
import wave
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLATFORM_DIR))

from audio_filter import AudioNoiseSuppressor  # noqa: E402

SAMPLE_RATE = 16000
DURATION_S = 4.0
MIN_NOISE_REDUCTION_DB = 6.0
MIN_VOICE_RETENTION = 0.50


def _envelope(second: float) -> float:
    """A syllable-like gate so spectral tracking does not see a steady tone."""
    windows = ((0.25, 0.85), (1.05, 1.65), (1.90, 2.55), (2.80, 3.65))
    for start, end in windows:
        if start <= second <= end:
            phase = (second - start) / (end - start)
            return math.sin(math.pi * phase) ** 0.6
    return 0.0


def _signals() -> tuple[list[float], list[float], list[float]]:
    """Return voice-like, steady-noise, and combined normalized sample lists."""
    rng = random.Random(20260904)
    voice: list[float] = []
    noise: list[float] = []
    combined: list[float] = []
    for index in range(int(SAMPLE_RATE * DURATION_S)):
        second = index / SAMPLE_RATE
        env = _envelope(second)
        # Harmonics plus two speech-formant-like components. The slowly varying
        # pitch keeps this unlike the stationary tones the denoiser targets.
        pitch = 175.0 + 18.0 * math.sin(2.0 * math.pi * 1.7 * second)
        spoken = env * (
            0.24 * math.sin(2.0 * math.pi * pitch * second)
            + 0.12 * math.sin(2.0 * math.pi * 700.0 * second)
            + 0.08 * math.sin(2.0 * math.pi * 1400.0 * second)
            + 0.05 * math.sin(2.0 * math.pi * 2400.0 * second)
        )
        # Deterministic fan/hiss plus mains-like rumble. The high-pass stage
        # should remove the rumble and afftdn should reduce the broadband bed.
        background = (
            0.10 * math.sin(2.0 * math.pi * 60.0 * second)
            + 0.075 * rng.uniform(-1.0, 1.0)
        )
        voice.append(spoken)
        noise.append(background)
        combined.append(max(-0.98, min(0.98, spoken + background)))
    return voice, noise, combined


def _wav(samples: list[float]) -> bytes:
    output = io.BytesIO()
    pcm = array.array("h", (int(max(-1.0, min(1.0, s)) * 32767) for s in samples))
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def _samples(payload: bytes) -> list[float]:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    pcm = array.array("h")
    pcm.frombytes(frames)
    return [value / 32768.0 for value in pcm]


def _rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def _process(suppressor: AudioNoiseSuppressor, payload: bytes) -> tuple[bytes, float]:
    result = suppressor.process(payload, "fixture.wav")
    if result.error:
        raise RuntimeError(result.error)
    if not result.applied:
        raise RuntimeError("balanced noise suppression was not applied")
    return result.audio, result.elapsed_ms


def verify(max_latency_ms: float) -> dict:
    voice, noise, combined = _signals()
    voice_wav, noise_wav, combined_wav = _wav(voice), _wav(noise), _wav(combined)
    with tempfile.TemporaryDirectory(prefix="locitize-noise-verify-") as root:
        suppressor = AudioNoiseSuppressor("balanced", Path(root) / "requests")
        filtered_voice, _ = _process(suppressor, voice_wav)
        filtered_noise, _ = _process(suppressor, noise_wav)
        # One warm-up before the five reported measurements keeps process and
        # filesystem cache startup from masquerading as steady-state latency.
        _process(suppressor, combined_wav)
        timings = [_process(suppressor, combined_wav)[1] for _ in range(5)]

    input_noise_rms = _rms(noise)
    output_noise_rms = _rms(_samples(filtered_noise))
    input_voice_rms = _rms(voice)
    output_voice_rms = _rms(_samples(filtered_voice))
    reduction_db = 20.0 * math.log10(
        input_noise_rms / max(output_noise_rms, 1e-12)
    )
    retention = output_voice_rms / max(input_voice_rms, 1e-12)
    ordered = sorted(timings)
    p95_ms = ordered[-1]  # conservative p95 for the approved five-run sample

    checks = {
        "noise_reduction": reduction_db >= MIN_NOISE_REDUCTION_DB,
        "voice_retention": retention >= MIN_VOICE_RETENTION,
        "latency": p95_ms < max_latency_ms,
    }
    return {
        "ok": all(checks.values()),
        "mode": "balanced",
        "noise_reduction_db": round(reduction_db, 2),
        "minimum_noise_reduction_db": MIN_NOISE_REDUCTION_DB,
        "voice_retention_ratio": round(retention, 3),
        "minimum_voice_retention_ratio": MIN_VOICE_RETENTION,
        "latency_ms": [round(value, 2) for value in timings],
        "latency_mean_ms": round(statistics.mean(timings), 2),
        "latency_p95_ms": round(p95_ms, 2),
        "maximum_latency_ms": max_latency_ms,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-latency-ms", type=float, default=500.0)
    args = parser.parse_args(argv)
    try:
        result = verify(args.max_latency_ms)
    except Exception as exc:  # noqa: BLE001 - CLI boundary prints honest failure
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
