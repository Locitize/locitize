"""Local microphone denoising before audio reaches whisper-server.

Open WebUI records browser audio and sends it to router.py. This module is the
single process boundary that turns that untrusted upload into a denoised 16 kHz
mono WAV. It invokes the owner's existing FFmpeg executable with fixed argument
presets; request data can never supply command syntax, paths, or a shell string.

Audio remains local. Request-scoped files live below the LOCITIZE data root and
are removed by TemporaryDirectory on success, failure, and timeout.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from logger import get_logger


# Filter graphs are constants selected by a validated enum. In particular, no
# filename, multipart value, or user-provided text is ever interpolated here.
# Balanced keeps the normal speech band, applies moderate adaptive spectral
# reduction, and closes a conservative gate during non-speech intervals. Strong
# is owner-selected for loud, steady backgrounds and trades more quiet-speech
# sensitivity for additional reduction.
FILTER_PRESETS = {
    "balanced": (
        "highpass=f=100,lowpass=f=8000,afftdn=nr=12:nf=-45:tn=1,"
        "agate=threshold=0.075:ratio=8:attack=5:release=250"
    ),
    "strong": (
        "highpass=f=120,lowpass=f=7500,afftdn=nr=20:nf=-40:tn=1,"
        "agate=threshold=0.09:ratio=12:attack=5:release=200"
    ),
}
VALID_NOISE_SUPPRESSION = ("off", "balanced", "strong")
DEFAULT_TIMEOUT_S = 15.0
_MAX_ERROR_CHARS = 240


@dataclass(frozen=True)
class AudioFilterResult:
    """One bounded processing outcome consumed by the router."""

    audio: bytes
    filename: str
    applied: bool
    processor: str
    elapsed_ms: float
    error: str = ""


class AudioNoiseSuppressor:
    """Apply one validated FFmpeg denoising preset to a recorded upload.

    ``runner`` and ``clock`` are seams for deterministic tests. Production uses
    subprocess.run and time.perf_counter. The result object carries errors rather
    than raising across the HTTP handler boundary, so one bad recording cannot
    terminate the router thread.
    """

    def __init__(
        self,
        mode: str,
        temp_root: Path | str,
        ffmpeg_path: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        runner: Callable[..., Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        normalized = str(mode or "").strip().lower()
        if normalized not in VALID_NOISE_SUPPRESSION:
            raise ValueError(
                "noise suppression must be one of "
                + ", ".join(VALID_NOISE_SUPPRESSION)
            )
        self.mode = normalized
        self.temp_root = Path(temp_root)
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg") or ""
        self.timeout_s = max(0.1, float(timeout_s))
        self._runner = runner or subprocess.run
        self._clock = clock or time.perf_counter
        self._log = get_logger("speech")

    def set_mode(self, mode: str) -> str:
        """Switch to another validated preset without reconstructing the processor.

        The live router holds this object for the process lifetime. A Voice Setup
        change must take effect on the next upload, so the enum is re-checked here
        the same way __init__ checks it. Returns the mode that is now active.
        """
        normalized = str(mode or "").strip().lower()
        if normalized not in VALID_NOISE_SUPPRESSION:
            raise ValueError(
                "noise suppression must be one of "
                + ", ".join(VALID_NOISE_SUPPRESSION)
            )
        self.mode = normalized
        return self.mode

    def process(self, audio: bytes, filename: str = "audio.wav") -> AudioFilterResult:
        """Return denoised WAV bytes, an exact off-mode pass-through, or an error."""
        started = self._clock()
        if not audio:
            return self._failure(started, "processing_failed: empty audio upload", 0)
        if self.mode == "off":
            return AudioFilterResult(
                audio=audio,
                filename=filename or "audio.wav",
                applied=False,
                processor="off",
                elapsed_ms=self._elapsed_ms(started),
            )
        if not self.ffmpeg_path:
            return self._failure(
                started,
                "unavailable: FFmpeg was not found; install FFmpeg or set "
                "speech.noise_suppression to off",
                len(audio),
            )

        try:
            self.temp_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="locitize-audio-", dir=str(self.temp_root)
            ) as private_dir:
                private = Path(private_dir)
                input_path = private / "input.audio"
                output_path = private / "filtered.wav"
                input_path.write_bytes(audio)
                command = self._command(input_path, output_path)
                try:
                    completed = self._runner(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.timeout_s,
                        check=False,
                        creationflags=_creation_flags(),
                    )
                except subprocess.TimeoutExpired:
                    return self._failure(
                        started,
                        f"timeout: noise suppression exceeded {self.timeout_s:g} seconds",
                        len(audio),
                    )
                except FileNotFoundError:
                    return self._failure(
                        started,
                        "unavailable: FFmpeg disappeared before processing",
                        len(audio),
                    )
                except PermissionError as exc:
                    return self._failure(
                        started,
                        "unavailable: FFmpeg could not be executed: "
                        + _safe_exception(exc),
                        len(audio),
                    )
                if int(getattr(completed, "returncode", 1)) != 0:
                    detail = _bounded_error(
                        getattr(completed, "stderr", b""), private
                    )
                    suffix = f": {detail}" if detail else ""
                    return self._failure(
                        started,
                        "processing_failed: FFmpeg rejected the audio" + suffix,
                        len(audio),
                    )
                try:
                    filtered = output_path.read_bytes()
                except OSError as exc:
                    return self._failure(
                        started,
                        "processing_failed: FFmpeg produced no readable output: "
                        + _safe_exception(exc),
                        len(audio),
                    )
                problem = _wav_problem(filtered)
                if problem:
                    return self._failure(
                        started,
                        "processing_failed: invalid filtered WAV: " + problem,
                        len(audio),
                    )
        except OSError as exc:
            return self._failure(
                started, "processing_failed: temporary audio storage unavailable: "
                + _safe_exception(exc),
                len(audio),
            )
        except Exception as exc:  # noqa: BLE001 - subprocess boundary containment
            return self._failure(
                started, "processing_failed: noise processor failed: "
                + _safe_exception(exc),
                len(audio),
            )

        elapsed = self._elapsed_ms(started)
        self._log.info(
            "noise suppression mode=%s input_bytes=%d output_bytes=%d "
            "elapsed_ms=%.1f result=pass",
            self.mode,
            len(audio),
            len(filtered),
            elapsed,
        )
        return AudioFilterResult(
            audio=filtered,
            filename="audio.wav",
            applied=True,
            processor="ffmpeg-afftdn",
            elapsed_ms=elapsed,
        )

    def _command(self, input_path: Path, output_path: Path) -> list[str]:
        """Build the fixed argv list; no shell or request text participates."""
        return [
            self.ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-af",
            FILTER_PRESETS[self.mode],
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

    def _failure(self, started: float, error: str, input_bytes: int) -> AudioFilterResult:
        elapsed = self._elapsed_ms(started)
        bounded = _ascii_bounded(error)
        self._log.warning(
            "noise suppression mode=%s input_bytes=%d output_bytes=0 "
            "elapsed_ms=%.1f result=fail category=%s",
            self.mode,
            max(0, int(input_bytes)),
            elapsed,
            bounded.split(":", 1)[0],
        )
        return AudioFilterResult(
            audio=b"",
            filename="audio.wav",
            applied=False,
            processor="ffmpeg-afftdn" if self.mode != "off" else "off",
            elapsed_ms=elapsed,
            error=bounded,
        )

    def _elapsed_ms(self, started: float) -> float:
        return max(0.0, (self._clock() - started) * 1000.0)


def _creation_flags() -> int:
    """Keep FFmpeg invisible under pythonw on Windows; zero elsewhere."""
    if os.name == "nt":
        return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return 0


def _wav_problem(payload: bytes) -> str:
    """Return why output is not the exact Whisper input contract, or empty."""
    if not payload:
        return "empty output"
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            if wav.getnchannels() != 1:
                return "expected mono output"
            if wav.getframerate() != 16000:
                return "expected 16000 Hz output"
            if wav.getsampwidth() != 2:
                return "expected signed 16-bit PCM output"
            if wav.getnframes() <= 0:
                return "output contained no audio frames"
            if wav.getcomptype() != "NONE":
                return "expected uncompressed PCM output"
    except (EOFError, OSError, wave.Error):
        return "unreadable WAV container"
    return ""


def _bounded_error(raw: Any, private_dir: Path) -> str:
    """Decode a bounded diagnostic and remove the request's private path."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", "replace")
    else:
        text = str(raw or "")
    text = text.replace(str(private_dir), "<temp>")
    return _ascii_bounded(" ".join(text.split()))


def _safe_exception(exc: BaseException) -> str:
    """Bound exception text and discard non-ASCII path/control artifacts."""
    return _ascii_bounded(" ".join(str(exc).split()))


def _ascii_bounded(text: str) -> str:
    cleaned = str(text).encode("ascii", "replace").decode("ascii")
    return cleaned[:_MAX_ERROR_CHARS]
