"""Kokoro TTS tests (M6, Architecture M6.6). Keyword: tts_client.

Proves the voice-OUT client and its service spec are correct with NO torch, NO
audio hardware, and NO real kokoro server: build_kokoro_server_spec is data-driven
and guards missing paths; KokoroClient.synthesize frames a correct request and
returns exactly the server's bytes against an injected fake opener; speak drives a
temp-wav-then-play sequence and degrades to TtsUnavailableError when the server is
unreachable (with a fake player, so real winsound is never called); and the kokoro
capability probe is correct across present/absent cases via injected find_spec /
file-exists facts. Every test name carries 'tts_client' so `pytest -k tts_client`
selects exactly this suite (AC2).
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import pytest

from config import Settings
from health import HealthStatus, KokoroProbe
from tts import (
    KOKORO_SERVER_NAME,
    KokoroClient,
    TtsUnavailableError,
    _pad_wav_leading_silence,
    build_kokoro_server_spec,
    concatenate_wavs,
    list_voices,
)


def _make_wav(sample_rate: int = 24000, seconds: float = 0.1) -> bytes:
    """Build a real mono 16-bit wav of non-silent samples for the pad tests.

    Kokoro produces 24kHz mono 16-bit audio; a small ramp of non-zero samples lets a
    test prove that leading silence was actually prepended in front of real audio.
    """
    n = int(sample_rate * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        # A simple non-zero waveform (value 1000) so the first real frame is not zero.
        wav.writeframes(b"\xe8\x03" * n)  # 1000 as little-endian int16, repeated
    return buf.getvalue()


def _tts_settings() -> Settings:
    """A Settings with both kokoro paths set (spec construction succeeds)."""
    settings = Settings()
    settings.paths.kokoro_model = "/locitize-test/kokoro/voices/kokoro-v1_0.pth"
    settings.paths.kokoro_voices = "/locitize-test/kokoro/voices"
    return settings


class _FakeResponse:
    """Context-manager stand-in for a urllib response returning canned bytes."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class _RecordingOpener:
    """Fake urlopen that records the request and returns canned wav bytes."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout=None):  # noqa: ANN001 - urllib Request
        self.request = request
        self.timeout = timeout
        return _FakeResponse(self._body)


# ---- build_kokoro_server_spec ------------------------------------------------


def test_tts_client_spec_is_data_driven():
    """The spec embeds the venv interpreter, the server script, the on-disk model
    and voices, the reserved kokoro port, and the /health readiness route."""
    spec = build_kokoro_server_spec(_tts_settings())
    assert spec.name == KOKORO_SERVER_NAME
    assert spec.command[0].lower().endswith("python.exe") or "python" in spec.command[0].lower()
    assert any(arg.endswith("kokoro_server.py") for arg in spec.command)
    assert "/locitize-test/kokoro/voices/kokoro-v1_0.pth" in spec.command
    assert "/locitize-test/kokoro/voices" in spec.command
    assert "127.0.0.1" in spec.command  # loopback host, never 0.0.0.0
    assert spec.port == 8092  # Data Model reserved kokoro port
    assert spec.health_path == "/health"  # readiness waits for model load
    assert spec.append_log is True  # a failed start keeps the prior trace


def test_tts_client_spec_guards_missing_model():
    """No kokoro model path -> a clear ValueError naming the remedy field."""
    settings = Settings()
    settings.paths.kokoro_voices = "/locitize-test/kokoro/voices"  # voices set, model unset
    with pytest.raises(ValueError) as exc:
        build_kokoro_server_spec(settings)
    assert "kokoro_model" in str(exc.value)


def test_tts_client_spec_guards_missing_voices():
    """No kokoro voices path -> ValueError naming the voices remedy field."""
    settings = Settings()
    settings.paths.kokoro_model = "kokoro-v1_0.pth"  # model set, voices unset
    with pytest.raises(ValueError) as exc:
        build_kokoro_server_spec(settings)
    assert "kokoro_voices" in str(exc.value)


# ---- KokoroClient.synthesize -------------------------------------------------


def test_tts_client_synthesize_frames_request_and_returns_bytes():
    """synthesize POSTs {text, voice, speed} as JSON to /synthesize on the resolved
    loopback port and returns exactly the server's wav bytes."""
    canned = b"RIFF....WAVEfake"
    opener = _RecordingOpener(canned)
    client = KokoroClient(8092, opener=opener)

    out = client.synthesize("hello world", voice="am_michael", speed=1.0)

    assert out == canned  # returns the server's bytes verbatim, never fabricated
    req = opener.request
    assert req.full_url == "http://127.0.0.1:8092/synthesize"
    assert req.get_method() == "POST"
    assert req.headers.get("Content-type") == "application/json"
    payload = json.loads(req.data.decode("utf-8"))
    assert payload == {"text": "hello world", "voice": "am_michael", "speed": 1.0}


def test_tts_client_synthesize_unreachable_raises_unavailable():
    """A transport error becomes TtsUnavailableError carrying a remedy, never a
    fabricated wav."""

    def _boom(request, timeout=None):  # noqa: ANN001
        raise OSError("connection refused")

    client = KokoroClient(8092, opener=_boom)
    with pytest.raises(TtsUnavailableError) as exc:
        client.synthesize("hi", voice="am_michael")
    assert exc.value.remedy  # a concrete owner-facing remedy is attached


# ---- KokoroClient.speak (temp-wav-then-play + degradation) --------------------


def test_tts_client_speak_writes_wav_then_plays(tmp_path):
    """speak synthesizes to a wav file, THEN plays that exact path (order matters:
    the file must exist before the player is handed it)."""
    canned = b"RIFF-canned-wav-bytes"
    events: list[tuple[str, object]] = []

    def _player(path: str) -> None:
        # Assert the file already exists with the synthesized bytes at play time.
        events.append(("play", path))
        assert Path(path).read_bytes() == canned

    out_path = tmp_path / "out.wav"
    client = KokoroClient(8092, opener=_RecordingOpener(canned), player=_player)
    returned = client.speak("hello", voice="am_michael", out_path=str(out_path))

    assert returned == str(out_path)
    assert out_path.read_bytes() == canned
    assert events == [("play", str(out_path))]  # played exactly once, after write


def test_tts_client_speak_deletes_one_off_temp_wav_after_play(tmp_path):
    """SEC-M6-1: a one-off speak (no out_path) writes its throwaway wav under the
    given temp_dir and DELETES it after playback, so no synthesized-speech residue is
    left on disk. The path must still exist at play time (played before deletion)."""
    canned = b"RIFF-canned-wav-bytes"
    played: list[str] = []

    def _player(path: str) -> None:
        played.append(path)
        assert Path(path).exists()  # played before it is removed

    client = KokoroClient(8092, opener=_RecordingOpener(canned), player=_player)
    returned = client.speak(
        "hello", voice="am_michael", temp_dir=str(tmp_path)
    )

    assert len(played) == 1
    assert Path(returned).parent == tmp_path  # landed in-tree, not the OS temp dir
    assert not Path(returned).exists()  # cleaned up after playback
    assert list(tmp_path.glob("*.wav")) == []  # no residue


def test_tts_client_speak_degrades_without_playing():
    """When the server is unreachable, speak raises TtsUnavailableError and the
    player is NEVER called (no fake silence, no real winsound)."""
    played: list[str] = []

    def _player(path: str) -> None:
        played.append(path)

    def _boom(request, timeout=None):  # noqa: ANN001
        raise OSError("no server")

    client = KokoroClient(8092, opener=_boom, player=_player)
    with pytest.raises(TtsUnavailableError):
        client.speak("hello", voice="am_michael")
    assert played == []  # degradation path never touches the player


# ---- D-M7-5: leading silence pad (keyword silence_pad) -----------------------


def test_tts_silence_pad_prepends_expected_leading_silence():
    """_pad_wav_leading_silence makes the wav longer by ~pad_ms of zero frames and
    the padded audio starts with silence, so device warm-up no longer clips the first
    phoneme (D-M7-5). Verified by decoding both wavs with the stdlib wave module."""
    sample_rate = 24000
    original = _make_wav(sample_rate=sample_rate, seconds=0.1)
    pad_ms = 200

    padded = _pad_wav_leading_silence(original, pad_ms)

    with wave.open(io.BytesIO(original), "rb") as src:
        src_frames = src.getnframes()
    with wave.open(io.BytesIO(padded), "rb") as dst:
        dst_frames = dst.getnframes()
        first = dst.readframes(dst.getnframes())

    expected_pad_frames = int(sample_rate * pad_ms / 1000.0)  # 4800 frames
    assert dst_frames == src_frames + expected_pad_frames
    # The leading pad_ms is pure silence: every byte of the first expected_pad_frames
    # (2 bytes/frame, mono) is zero before the real (non-zero) audio starts.
    lead_bytes = expected_pad_frames * 2
    assert first[:lead_bytes] == b"\x00" * lead_bytes
    assert first[lead_bytes:lead_bytes + 2] != b"\x00\x00"  # real audio resumes


def test_tts_silence_pad_leaves_non_wav_bytes_untouched():
    """Non-parseable bytes (a test's canned non-wav) are returned unchanged, so the
    pad never corrupts or fabricates audio and synthesize's byte contract holds."""
    canned = b"RIFF-not-a-real-wav"
    assert _pad_wav_leading_silence(canned, 200) == canned


def test_tts_silence_pad_disabled_when_pad_is_zero():
    """pad_ms <= 0 disables the pad entirely (identity)."""
    wav = _make_wav()
    assert _pad_wav_leading_silence(wav, 0) == wav


def test_tts_client_speak_pads_real_wav_before_writing(tmp_path):
    """speak() applies the leading-silence pad to a real synthesized wav before it is
    written/played, so the played file is longer than what the server returned by the
    pad (D-M7-5). A fake opener returns a real wav; a fake player avoids audio."""
    sample_rate = 24000
    server_wav = _make_wav(sample_rate=sample_rate, seconds=0.1)
    out_path = tmp_path / "spoken.wav"
    client = KokoroClient(
        8092,
        opener=_RecordingOpener(server_wav),
        player=lambda _p: None,
        lead_silence_ms=200,
    )

    client.speak("hello", voice="am_michael", out_path=str(out_path))

    with wave.open(io.BytesIO(server_wav), "rb") as src:
        src_frames = src.getnframes()
    with wave.open(str(out_path), "rb") as dst:
        written_frames = dst.getnframes()
        head = dst.readframes(dst.getnframes())
    expected_pad = int(sample_rate * 200 / 1000.0)
    assert written_frames == src_frames + expected_pad  # padded on disk
    assert head[: expected_pad * 2] == b"\x00" * (expected_pad * 2)  # starts silent


# ---- concatenate_wavs (D-M7-7 buffered whole-reply playback) ------------------


def test_tts_concatenate_wavs_builds_one_padded_gapped_reply():
    """concatenate_wavs joins per-sentence wavs into ONE wav = lead pad + s1 + gap + s2
    + gap + s3 (D-M7-7). Assert the exact frame count, a silent leading pad, silent
    inter-sentence gaps, and intact sentence audio -- proving the reply plays as a
    single clip with nothing clipped."""
    sr = 24000
    s1 = _make_wav(sample_rate=sr, seconds=0.10)
    s2 = _make_wav(sample_rate=sr, seconds=0.05)
    s3 = _make_wav(sample_rate=sr, seconds=0.08)
    lead_ms, gap_ms = 700, 150

    combined = concatenate_wavs([s1, s2, s3], lead_ms, gap_ms)

    with wave.open(io.BytesIO(combined), "rb") as dst:
        nframes = dst.getnframes()
        frame_size = dst.getsampwidth() * dst.getnchannels()
        raw = dst.readframes(nframes)
    lead = int(sr * lead_ms / 1000)
    gap = int(sr * gap_ms / 1000)
    n1, n2, n3 = int(sr * 0.10), int(sr * 0.05), int(sr * 0.08)
    # Two INTERNAL gaps only (between the three sentences), never a trailing one.
    assert nframes == lead + n1 + gap + n2 + gap + n3
    # Leading pad is pure silence; the first real phoneme starts right after it.
    assert raw[: lead * frame_size] == b"\x00" * (lead * frame_size)
    assert raw[lead * frame_size: lead * frame_size + 2] != b"\x00\x00"
    # The gap between sentence 1 and 2 is silence (a pause, not a clipped word).
    g1 = (lead + n1) * frame_size
    assert raw[g1: g1 + gap * frame_size] == b"\x00" * (gap * frame_size)


def test_tts_concatenate_wavs_skips_empty_and_returns_empty_when_none():
    """Empty/non-wav chunks are skipped; no usable chunk yields b'' (honest empty)."""
    assert concatenate_wavs([], 700, 150) == b""
    assert concatenate_wavs([b"", b"not-a-wav"], 700, 150) == b""
    # A single real wav still concatenates (just the lead pad + that sentence).
    one = _make_wav(seconds=0.05)
    out = concatenate_wavs([b"", one, b""], 200, 150)
    with wave.open(io.BytesIO(out), "rb") as dst:
        assert dst.getnframes() == int(24000 * 0.2) + int(24000 * 0.05)


def test_tts_concatenate_wavs_rejects_mismatched_format():
    """The 24kHz-mono-16-bit assumption is CHECKED: a differing sample rate raises
    rather than producing corrupt concatenated audio."""
    a = _make_wav(sample_rate=24000, seconds=0.05)
    b = _make_wav(sample_rate=16000, seconds=0.05)
    with pytest.raises(ValueError):
        concatenate_wavs([a, b], 700, 150)


# ---- list_voices -------------------------------------------------------------


def test_tts_client_list_voices_reads_on_disk_pt_files(tmp_path):
    """list_voices returns sorted stems of the .pt files in the voices dir."""
    for name in ("am_michael", "af_bella", "bm_george"):
        (tmp_path / f"{name}.pt").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignored")  # non-.pt ignored
    assert list_voices(str(tmp_path)) == ["af_bella", "am_michael", "bm_george"]


def test_tts_client_list_voices_missing_dir_is_empty():
    """An unset or missing voices dir yields [] (honest, not a guess)."""
    assert list_voices("") == []
    assert list_voices("/locitize-test/definitely/not/here") == []


# ---- KokoroProbe capability check (find_spec + file-exists) -------------------


def _exists_of(present: set[str]):
    """A path_exists provider that reports the given paths as on disk."""

    def _exists(path: str) -> bool:
        return path in present

    return _exists


def test_tts_client_capability_probe_pass_when_all_present():
    """Weights on disk AND the kokoro package importable AND enabled -> PASS."""
    probe = KokoroProbe(
        "kokoro-v1_0.pth",
        "voices",
        enabled=True,
        package_present=True,
        path_exists=_exists_of({"kokoro-v1_0.pth", "voices"}),
    )
    assert probe.run().status == HealthStatus.PASS


def test_tts_client_capability_probe_warns_when_package_absent():
    """Weights present but the kokoro package not installed -> WARNING + pip remedy."""
    result = KokoroProbe(
        "kokoro-v1_0.pth",
        "voices",
        enabled=True,
        package_present=False,
        path_exists=_exists_of({"kokoro-v1_0.pth", "voices"}),
    ).run()
    assert result.status == HealthStatus.WARNING
    assert "pip install" in (result.remedy or "")


def test_tts_client_capability_probe_warns_when_weights_missing():
    """Package installed but weights absent -> WARNING naming the paths remedy."""
    result = KokoroProbe(
        "kokoro-v1_0.pth",
        "voices",
        enabled=True,
        package_present=True,
        path_exists=_exists_of(set()),  # nothing on disk
    ).run()
    assert result.status == HealthStatus.WARNING
    assert "kokoro_model" in (result.remedy or "")


def test_tts_client_capability_probe_warns_when_disabled():
    """tts.enabled false -> WARNING even when everything else is present."""
    result = KokoroProbe(
        "kokoro-v1_0.pth",
        "voices",
        enabled=False,
        package_present=True,
        path_exists=_exists_of({"kokoro-v1_0.pth", "voices"}),
    ).run()
    assert result.status == HealthStatus.WARNING
    assert "enabled" in (result.detail or "").lower()


# ---- kokoro_server device choice (owner request 2026-09-03) -----------------
# The server moves the model to the card when torch was built with CUDA and one
# is present (measured 0.10s vs 0.78s per sentence). The choice is torch's own
# answer; a fake torch pins the mapping without a GPU in the test machine.


def _pick_device_with(monkeypatch, available: bool) -> str:
    import sys
    import types

    import kokoro_server

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return kokoro_server.pick_device()


def test_kokoro_server_picks_cuda_only_when_torch_sees_a_device(monkeypatch):
    assert _pick_device_with(monkeypatch, True) == "cuda"
    assert _pick_device_with(monkeypatch, False) == "cpu"


# --------------------------------------------------------------------------- #
# Video memory discipline in kokoro_server (2026-09-03): chunks are bounded so
# torch's allocator has a ceiling, and the engine warms up to that ceiling
# before /health says ready, so the footprint the launcher fits the model
# against is the footprint it keeps. Pure functions, tested without torch.
# --------------------------------------------------------------------------- #


def test_kokoro_split_keeps_whole_sentences_that_fit():
    import kokoro_server

    assert kokoro_server.split_utterance("Hello there. How are you?  Fine!\nNew line.") == [
        "Hello there.", "How are you?", "Fine!", "New line."
    ]
    assert kokoro_server.split_utterance("") == []
    assert kokoro_server.split_utterance("   \n ") == []


def test_kokoro_split_breaks_a_long_sentence_at_clauses_then_words():
    import kokoro_server

    clause = "the quick brown fox jumps over the lazy dog and keeps on running through the open fields, "
    long = clause * 4 + "and then it stops."
    chunks = kokoro_server.split_utterance(long)
    assert len(chunks) > 1
    assert all(len(c) <= kokoro_server._MAX_CHUNK_CHARS for c in chunks)
    # Clause boundaries are respected: every chunk but the last ends at a comma.
    assert all(c.endswith(",") for c in chunks[:-1])
    assert " ".join(chunks) == long.strip()

    words = "word " * 60  # no punctuation at all: word boundaries
    chunks = kokoro_server.split_utterance(words)
    assert all(len(c) <= kokoro_server._MAX_CHUNK_CHARS for c in chunks)
    assert " ".join(chunks) == words.strip()

    # A single oversized token is passed through rather than cut mid-word.
    assert kokoro_server.split_utterance("a" * 300) == ["a" * 300]


def test_kokoro_split_honours_a_custom_limit():
    import kokoro_server

    assert kokoro_server.split_utterance("one two three four", limit=9) == ["one two", "three", "four"]


def test_kokoro_warmup_text_is_exactly_the_chunk_bound_and_a_sentence():
    import kokoro_server

    text = kokoro_server.warmup_text()
    assert len(text) == kokoro_server._MAX_CHUNK_CHARS == 160
    assert text.endswith(".") and text.isascii()
    shorter = kokoro_server.warmup_text(limit=40)
    assert len(shorter) <= 40 and shorter.endswith(".") and not shorter.endswith(" .")
    # The bound the peak was measured at (155 chars -> 1192 MB) is the shipped one.
    assert kokoro_server.split_utterance(text) == [text]


def test_kokoro_engine_warms_up_with_the_first_voice_before_it_is_ready(tmp_path, monkeypatch):
    import kokoro_server

    (tmp_path / "bm_lewis.pt").write_bytes(b"")
    (tmp_path / "af_bella.pt").write_bytes(b"")
    calls: list = []

    engine = kokoro_server.KokoroEngine.__new__(kokoro_server.KokoroEngine)
    engine._voices_dir = tmp_path
    engine.device = "cuda"
    monkeypatch.setattr(
        engine, "synthesize", lambda text, voice, speed: calls.append((text, voice, speed)) or b"wav"
    )
    assert engine._warm_up() == "warmed up with af_bella"
    assert calls == [(kokoro_server.warmup_text(), "af_bella", 1.0)]


def test_kokoro_trim_silence_drops_quiet_edges():
    import numpy as np

    import kokoro_server

    # 50ms silence + 100ms tone + 50ms silence at 24 kHz.
    rate = kokoro_server._SAMPLE_RATE_HZ
    silence = np.zeros(int(rate * 0.05), dtype=np.float32)
    tone = np.full(int(rate * 0.10), 0.4, dtype=np.float32)
    trimmed = kokoro_server.trim_silence(np.concatenate([silence, tone, silence]))
    assert abs(len(trimmed) - len(tone)) <= 2
    assert float(np.min(np.abs(trimmed))) > kokoro_server._SILENCE_FLOOR


def test_kokoro_join_crossfades_instead_of_hard_gap():
    import numpy as np

    import kokoro_server

    rate = kokoro_server._SAMPLE_RATE_HZ
    fade = int(rate * kokoro_server._CROSSFADE_MS / 1000.0)
    # Two loud chunks with deliberate quiet pads that used to become a mid-phrase dip.
    # trim_silence strips those pads before the crossfade, so the seam is tone-on-tone.
    pad = np.zeros(fade, dtype=np.float32)
    tone = np.full(fade * 2, 0.5, dtype=np.float32)
    a = np.concatenate([tone, pad])
    b = np.concatenate([pad, tone])
    hard = np.concatenate([a, b])
    joined = kokoro_server.join_audio_chunks([a, b])
    # Crossfade + trim shortens the seam vs a hard concat of the padded chunks.
    assert len(joined) < len(hard)
    assert len(joined) == len(tone) + len(tone) - fade
    # No near-silent valley at the seam (the old choppy artefact).
    seam = len(tone) - fade
    window = joined[seam : seam + fade]
    assert float(np.min(np.abs(window))) > kokoro_server._SILENCE_FLOOR * 2


def test_kokoro_engine_warm_up_never_blocks_readiness(tmp_path, monkeypatch):
    import kokoro_server

    engine = kokoro_server.KokoroEngine.__new__(kokoro_server.KokoroEngine)
    engine._voices_dir = tmp_path
    engine.device = "cpu"
    assert engine._warm_up() == "no voice files to warm up with"

    (tmp_path / "am_michael.pt").write_bytes(b"")

    def boom(text, voice, speed):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(engine, "synthesize", boom)
    assert engine._warm_up() == "warm-up failed: cuda out of memory"
