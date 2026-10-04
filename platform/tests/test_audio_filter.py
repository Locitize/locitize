"""Noise-filter unit tests with no microphone, model, or real FFmpeg process.

The runner seam records the exact argv and writes deterministic WAV output.
Separate functional verification invokes the installed FFmpeg binary.
"""

from __future__ import annotations

import io
import subprocess
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_filter import (
    AudioNoiseSuppressor,
    FILTER_PRESETS,
    VALID_NOISE_SUPPRESSION,
    _wav_problem,
)


def _wav_bytes(frames: int = 160) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x10\x00" * frames)
    return output.getvalue()


def _success_runner(calls: list | None = None):
    def run(command, **kwargs):
        if calls is not None:
            calls.append((command, kwargs))
        Path(command[-1]).write_bytes(_wav_bytes())
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    return run


def test_modes_are_a_closed_enum_and_invalid_syntax_is_refused(tmp_path):
    assert VALID_NOISE_SUPPRESSION == ("off", "balanced", "strong")
    assert set(FILTER_PRESETS) == {"balanced", "strong"}
    with pytest.raises(ValueError, match="must be one of"):
        AudioNoiseSuppressor("balanced,volume=99", tmp_path)


def test_set_mode_switches_a_live_processor_without_rebuild(tmp_path):
    suppressor = AudioNoiseSuppressor("balanced", tmp_path, ffmpeg_path="ffmpeg")
    assert suppressor.set_mode("strong") == "strong"
    assert suppressor.mode == "strong"
    assert suppressor.set_mode("off") == "off"
    with pytest.raises(ValueError, match="must be one of"):
        suppressor.set_mode("custom=unsafe")


def test_off_is_a_byte_identical_pass_through_without_a_process(tmp_path):
    called = []
    source = b"not-even-a-wav-is-fine-when-explicitly-off"
    result = AudioNoiseSuppressor(
        "off", tmp_path, runner=lambda *a, **k: called.append(a)
    ).process(source, "clip.webm")
    assert result.audio == source
    assert result.filename == "clip.webm"
    assert result.applied is False
    assert result.processor == "off"
    assert result.error == ""
    assert called == []


def test_enabled_mode_fails_closed_when_ffmpeg_is_missing(tmp_path):
    suppressor = AudioNoiseSuppressor("balanced", tmp_path, ffmpeg_path="ffmpeg")
    suppressor.ffmpeg_path = ""
    result = suppressor.process(b"audio", "clip.webm")
    assert result.audio == b""
    assert result.error.startswith("unavailable:")


def test_ffmpeg_disappearing_or_becoming_unexecutable_is_unavailable(tmp_path):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("gone")

    def denied(*_args, **_kwargs):
        raise PermissionError("denied")

    missing_result = AudioNoiseSuppressor(
        "balanced", tmp_path, ffmpeg_path="ffmpeg.exe", runner=missing
    ).process(b"audio", "recording.webm")
    denied_result = AudioNoiseSuppressor(
        "balanced", tmp_path, ffmpeg_path="ffmpeg.exe", runner=denied
    ).process(b"audio", "recording.webm")

    assert missing_result.error.startswith("unavailable:")
    assert denied_result.error.startswith("unavailable:")


def test_success_uses_fixed_argv_validates_wav_and_cleans_temp_files(tmp_path):
    calls = []
    suppressor = AudioNoiseSuppressor(
        "balanced",
        tmp_path,
        ffmpeg_path="ffmpeg-test.exe",
        runner=_success_runner(calls),
    )
    result = suppressor.process(b"browser audio", "hostile;name.webm")

    assert result.applied is True
    assert result.processor == "ffmpeg-afftdn"
    assert result.filename == "audio.wav"
    assert _wav_problem(result.audio) == ""
    command, kwargs = calls[0]
    assert command[0] == "ffmpeg-test.exe"
    assert command[command.index("-af") + 1] == FILTER_PRESETS["balanced"]
    assert "hostile;name.webm" not in command
    assert kwargs["check"] is False
    assert kwargs["timeout"] == 15.0
    assert "shell" not in kwargs
    assert list(tmp_path.iterdir()) == []


def test_timeout_is_typed_and_temp_files_are_removed(tmp_path):
    def timeout(_command, **_kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 0.1)

    result = AudioNoiseSuppressor(
        "balanced", tmp_path, ffmpeg_path="ffmpeg.exe", timeout_s=0.1,
        runner=timeout,
    ).process(b"audio")
    assert result.error.startswith("timeout:")
    assert list(tmp_path.iterdir()) == []


def test_nonzero_exit_is_bounded_ascii_and_hides_private_path(tmp_path):
    def fail(command, **_kwargs):
        private_path = command[-1]
        detail = ("bad input at " + private_path + " snowman=\u2603 " + "x" * 500)
        return SimpleNamespace(returncode=1, stdout=b"", stderr=detail.encode("utf-8"))

    result = AudioNoiseSuppressor(
        "strong", tmp_path, ffmpeg_path="ffmpeg.exe", runner=fail
    ).process(b"audio")
    assert result.error.startswith("processing_failed:")
    assert str(tmp_path) not in result.error
    assert len(result.error) <= 240
    assert result.error.isascii()
    assert list(tmp_path.iterdir()) == []


def test_malformed_output_is_refused(tmp_path):
    def malformed(command, **_kwargs):
        Path(command[-1]).write_bytes(b"not wav")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    result = AudioNoiseSuppressor(
        "balanced", tmp_path, ffmpeg_path="ffmpeg.exe", runner=malformed
    ).process(b"audio")
    assert "invalid filtered WAV" in result.error
    assert result.audio == b""


def test_unwritable_temp_root_is_an_explicit_failure(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x", encoding="ascii")
    result = AudioNoiseSuppressor(
        "balanced", blocked, ffmpeg_path="ffmpeg.exe", runner=_success_runner()
    ).process(b"audio")
    assert "temporary audio storage unavailable" in result.error


def test_concurrent_calls_use_distinct_private_directories(tmp_path):
    parents = []
    lock = threading.Lock()

    def run(command, **_kwargs):
        with lock:
            parents.append(Path(command[-1]).parent)
        Path(command[-1]).write_bytes(_wav_bytes())
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    suppressor = AudioNoiseSuppressor(
        "balanced", tmp_path, ffmpeg_path="ffmpeg.exe", runner=run
    )
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(suppressor.process(b"audio")))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(result.applied for result in results)
    assert len(set(parents)) == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "payload,problem",
    [
        (b"", "empty output"),
        (b"garbage", "unreadable WAV container"),
    ],
)
def test_wav_validation_rejects_invalid_outputs(payload, problem):
    assert _wav_problem(payload) == problem
