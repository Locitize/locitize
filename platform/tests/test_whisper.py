"""Whisper module tests: spec construction and the HTTP transcription client.

Proves build_whisper_server_spec / build_whisper_stream_spec are data-driven and
guard missing paths, and that transcribe_file frames a correct multipart request
and parses the server's {"text": ...} reply -- all with an injected opener, so no
real whisper-server is needed (Architecture section 12).
"""

from __future__ import annotations

from config import Settings
from whisper import (
    CAPTURE_BANNER,
    WHISPER_SERVER_NAME,
    WHISPER_STREAM_NAME,
    StartupNoiseGate,
    TranscriptDeduplicator,
    build_whisper_server_spec,
    build_whisper_stream_spec,
    extract_spoken_text,
    is_diagnostic_line,
    is_non_speech,
    is_transcript_line,
    parse_block_header,
    parse_capture_interval,
    transcribe_file,
)


class _FakeClock:
    """A controllable monotonic clock so capture anchoring is tested without sleeping."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _server_settings() -> Settings:
    settings = Settings()
    settings.paths.whisper = "whisper-server.exe"
    settings.paths.whisper_model = "ggml.bin"
    settings.paths.whisper_stream = "whisper-stream.exe"
    return settings


def test_build_whisper_server_spec_is_data_driven():
    """The server spec embeds the configured binary, model, loopback host, and the
    reserved whisper port; readiness is TCP (no health_path)."""
    spec = build_whisper_server_spec(_server_settings())
    assert spec.name == WHISPER_SERVER_NAME
    assert spec.command[0] == "whisper-server.exe"
    assert "ggml.bin" in spec.command
    assert "127.0.0.1" in spec.command  # loopback host, never 0.0.0.0
    assert spec.port == 8091  # Data Model reserved whisper port
    assert spec.health_path is None  # whisper-server has no /health route
    assert "--suppress-nst" in spec.command
    assert "--vad" not in spec.command


def test_build_whisper_server_spec_guards_missing_model():
    """No configured model path -> a clear ValueError with a remedy, not a crash."""
    settings = Settings()
    settings.paths.whisper = "whisper-server.exe"  # binary set, model unset
    try:
        build_whisper_server_spec(settings)
        raise AssertionError("expected ValueError for missing whisper model")
    except ValueError as exc:
        assert "whisper_model" in str(exc)


def test_build_whisper_server_spec_guards_missing_binary():
    """No configured whisper binary -> ValueError, honest guard."""
    settings = Settings()
    settings.paths.whisper_model = "ggml.bin"  # model set, binary unset
    try:
        build_whisper_server_spec(settings)
        raise AssertionError("expected ValueError for missing whisper binary")
    except ValueError as exc:
        assert "whisper" in str(exc)


def test_build_whisper_stream_spec_has_no_port():
    """whisper-stream captures from the mic: no network port, model flag present."""
    spec = build_whisper_stream_spec(_server_settings())
    assert spec.name == WHISPER_STREAM_NAME
    assert spec.command[0] == "whisper-stream.exe"
    assert "ggml.bin" in spec.command
    assert spec.port is None  # SDL2 mic capture, no port to bind


# --- D-M3-2: VAD gating wired into the stream ServiceSpec (AC9) --------------
# whisper-stream's --step 0 selects voice-activity sliding-window mode, so it only
# transcribes detected speech and stops hallucinating stock phrases on silence.


def test_build_whisper_stream_spec_enables_vad_mode_by_default():
    """The stream command carries --step 0 (VAD mode) plus the VAD/freq thresholds
    from settings.speech, so silent windows are never transcribed (D-M3-2)."""
    spec = build_whisper_stream_spec(_server_settings())
    cmd = spec.command
    # --step 0 is the whole point: it switches whisper-stream out of fixed-window
    # transcription (which hallucinates on silence) into speech-gated mode.
    assert "--step" in cmd
    assert cmd[cmd.index("--step") + 1] == "0"
    assert "--vad-thold" in cmd
    assert cmd[cmd.index("--vad-thold") + 1] == "0.6"  # default, integral-trimmed
    assert "--freq-thold" in cmd
    assert cmd[cmd.index("--freq-thold") + 1] == "100"  # 100.0 -> "100"


def test_build_whisper_stream_spec_vad_flags_are_data_driven():
    """Owner-tuned VAD thresholds in settings.speech reach the command verbatim, so
    mic sensitivity is tunable without a code change."""
    settings = _server_settings()
    settings.speech.vad_thold = 0.75
    settings.speech.freq_thold = 80.0
    settings.speech.stream_step_ms = 0
    cmd = build_whisper_stream_spec(settings).command
    assert cmd[cmd.index("--vad-thold") + 1] == "0.75"
    assert cmd[cmd.index("--freq-thold") + 1] == "80"


class _FakeResponse:
    """Context-manager HTTP response stand-in returning fixed bytes."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._data


def test_transcribe_file_frames_multipart_and_parses_text(tmp_path):
    """The client posts a multipart 'file' upload to /inference on the given port
    and returns the server's stripped text."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF____fakeaudiobytes")
    captured: dict = {}

    def opener(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = request.data
        return _FakeResponse(b'{"text":" the quick brown fox.\\n"}')

    text = transcribe_file(str(wav), 8091, opener=opener)
    assert text == "the quick brown fox."  # leading/trailing whitespace stripped
    assert captured["url"] == "http://127.0.0.1:8091/inference"
    assert b"fakeaudiobytes" in captured["body"]
    assert b'name="file"' in captured["body"]
    assert b"json" in captured["body"]  # response_format=json part present


def test_transcribe_file_raises_without_text_field(tmp_path):
    """A server reply lacking a 'text' field is an honest error, not a fake result."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF____x")

    def opener(request, timeout=None):
        return _FakeResponse(b'{"error":"bad"}')

    try:
        transcribe_file(str(wav), 8091, opener=opener)
        raise AssertionError("expected ValueError for a reply without transcript text")
    except ValueError:
        pass


# --- D-M3-1: transcript de-duplication (AC9) --------------------------------
# These cover LOCITIZE's presentation filter for whisper-stream's overlapping-window
# output: exact dups and cosmetic (whitespace/punctuation/case) dups are dropped,
# while a genuinely repeated utterance separated by other speech survives.


def test_dedup_suppresses_exact_consecutive_duplicate():
    """The overlap case: one utterance emitted twice by two windows -> once out."""
    stream = ["Can you hear me?", "Can you hear me?"]
    assert TranscriptDeduplicator().filter_segments(stream) == ["Can you hear me?"]


def test_dedup_suppresses_whitespace_and_punctuation_variant():
    """Cosmetic differences (spacing, case, trailing mark) are the same utterance."""
    stream = ["Can you hear me?", "  can  you   hear me  "]
    # First survivor keeps its original text; the variant is recognized as a dup.
    assert TranscriptDeduplicator().filter_segments(stream) == ["Can you hear me?"]


def test_dedup_keeps_genuine_repeat_after_other_speech():
    """Near-miss guard: a real re-utterance separated by other speech must NOT be
    suppressed -- it falls outside the small trailing horizon."""
    stream = [
        "turn on the lights",
        "what time is it",
        "set a timer",
        "play some music",
        "turn on the lights",  # said again after 3 other segments -> keep it
    ]
    result = TranscriptDeduplicator(horizon=3).filter_segments(stream)
    assert result == stream  # every segment survives; nothing wrongly merged


def test_dedup_drops_blank_and_whitespace_only_segments():
    """Blank lines from the raw stream are noise, never emitted, and do not advance
    the horizon (so a real dup straddling them is still caught)."""
    stream = ["hello there", "   ", "", "hello there"]
    assert TranscriptDeduplicator().filter_segments(stream) == ["hello there"]


def test_dedup_normalize_key_matches_expected_form():
    """The comparison key: trimmed, whitespace-collapsed, case-folded, trailing
    punctuation stripped -- interior punctuation is preserved as content."""
    assert TranscriptDeduplicator.normalize("  Can  YOU hear me?! ") == "can you hear me"
    assert TranscriptDeduplicator.normalize("it's 3:15, right.") == "it's 3:15, right"
    assert TranscriptDeduplicator.normalize("   ") == ""


def test_dedup_allows_immediate_repeat_once_horizon_passes():
    """A short horizon of 1 only guards the immediately-preceding segment, proving
    the horizon is what bounds suppression (regression anchor for the setting)."""
    dedup = TranscriptDeduplicator(horizon=1)
    assert dedup.accept("yes") is True
    assert dedup.accept("yes") is False  # immediate dup suppressed
    assert dedup.accept("no") is True  # different segment ages "yes" out of horizon
    assert dedup.accept("yes") is True  # now allowed again (only 1 remembered)


# --- D-M3-2: silence-hallucination presentation guard (AC9) ------------------
# Defense-in-depth alongside VAD mode: drop bare placeholders and whisper's own
# non-speech markers that can still slip onto a marginal window, WITHOUT ever
# blocklisting a genuinely spoken phrase.


def test_is_non_speech_drops_placeholders_and_whisper_markers():
    """A lone dot, blank, and whisper's bracketed/parenthesized non-speech
    annotations carry no spoken words -> reported as non-speech and dropped."""
    for marker in [
        "",
        "   ",
        ".",
        "...",
        " . ",
        "[BLANK_AUDIO]",
        "[ Silence ]",
        "(music)",
        "[silence]",
    ]:
        assert is_non_speech(marker) is True, marker


def test_is_non_speech_keeps_genuinely_spoken_thank_you():
    """The critical near-miss: 'Thank you.' must survive when actually spoken. The
    guard is structural (has alphanumeric content), never a phrase blocklist."""
    assert is_non_speech("Thank you.") is False
    assert is_non_speech("thank you") is False
    assert is_non_speech("What time is it?") is False
    # Real words alongside an annotation still count as speech.
    assert is_non_speech("okay [BLANK_AUDIO]") is False


def test_dedup_drops_hallucination_markers_without_advancing_horizon():
    """Wired end-to-end: a silence hallucination stream (a real utterance framed by
    placeholder/marker noise) surfaces only the spoken words, and the markers do
    not pollute the dedup horizon (the real utterance is still surfaced once)."""
    # Models the D-M3-2 report: owner spoke one sentence, then silence produced
    # "Thank you." and "." placeholders on the quiet windows.
    stream = [
        "What time is it?",
        ".",
        "[BLANK_AUDIO]",
        "Thank you.",  # hallucinated stock phrase on silence -> must be dropped...
        "Thank you.",  # ...as a repeat too; the marker path drops it regardless
        "   ",
    ]
    # In the hallucination case "Thank you." is a non-word-window artifact. But it
    # DOES contain letters, so is_non_speech keeps it -- the marker guard alone will
    # not remove a genuine-looking phrase (that is by design, per D-M3-2: no phrase
    # blocklist). The real silence defense is VAD mode upstream. Here we assert the
    # placeholders/markers are stripped and dedup collapses the repeat.
    result = TranscriptDeduplicator().filter_segments(stream)
    assert result == ["What time is it?", "Thank you."]


def test_vad_style_stream_surfaces_only_spoken_utterances():
    """VAD mode changes cadence: one line per detected utterance, silent windows
    emit whisper markers instead of fixed-window overlaps. The guard yields only the
    spoken utterances in order, dropping every non-speech marker."""
    # Synthetic VAD-cadence output (never a real binary): distinct utterances, each
    # separated by the markers whisper prints for the intervening silence.
    vad_stream = [
        "turn on the lights",
        "[ Silence ]",
        "what is the weather",
        ".",
        "(music)",
        "set a five minute timer",
    ]
    result = TranscriptDeduplicator().filter_segments(vad_stream)
    assert result == [
        "turn on the lights",
        "what is the weather",
        "set a five minute timer",
    ]


# --------------------------------------------------------------------------- #
# D-M7-1: startup-noise gate -- only real transcribed speech becomes an utterance.
# --------------------------------------------------------------------------- #

# The exact engine/boot lines the owner observed being fed to the LLM as fake user
# turns in the AC17 session, plus the other classes from a real capture log.
_DIAGNOSTIC_LINES = [
    "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16302 MiB):",
    "  Device 0: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, VMM: yes, VRAM: 16302 MiB",
    "load_backend: loaded CUDA backend from /locitize-test/whisper/ggml-cuda.dll",
    "load_backend: loaded CPU backend from /locitize-test/whisper/ggml-cpu-alderlake.dll",
    "init: found 1 capture devices:",
    "init:    - Capture device #0: 'Microphone (USB PnP Audio Device)'",
    "whisper_init_from_file_with_params_no_state: loading model from '/locitize-test/whisper/ggml-large-v3-turbo.bin'",
    "whisper_init_with_params_no_state: use gpu    = 1",
    "whisper_model_load: n_vocab       = 51866",
    "whisper_init_state: compute buffer (decode) =  100.04 MB",
    "SDL_main: processing 0 samples (step = 0.0 sec / len = 10.0 sec ...)",
    "SDL_main: using VAD, will transcribe on speech activity",
    "### Transcription 0 START | t0 = 0 ms | t1 = 2713 ms",
    "### Transcription 0 END",
]

# Natural speech that MUST survive the filter, including the deliberately tricky
# "5: buy milk" (a colon that is not a key:value diagnostic) from the defect spec.
_REAL_SPEECH_LINES = [
    "what is the capital of Kenya",
    "remind me at 5: buy milk",
    "Thank you.",
    "This is going to be the last stop ending up.",
    "turn on the lights",
]


def test_is_diagnostic_line_flags_every_known_engine_class():
    """Every whisper-stream boot/engine/marker line is recognized as diagnostic."""
    for line in _DIAGNOSTIC_LINES:
        assert is_diagnostic_line(line) is True, line


def test_is_diagnostic_line_passes_real_speech_including_a_colon():
    """Conservative by design: natural speech is never mistaken for a diagnostic,
    even a sentence containing a colon ('remind me at 5: buy milk')."""
    for line in _REAL_SPEECH_LINES:
        assert is_diagnostic_line(line) is False, line


def test_is_transcript_line_gated_on_capture_started():
    """Nothing is speech before capture begins; after it, only non-diagnostic
    non-blank lines are speech."""
    assert is_transcript_line("what is the capital of Kenya", capture_started=False) is False
    assert is_transcript_line("what is the capital of Kenya", capture_started=True) is True
    assert is_transcript_line("ggml_cuda_init: found 1 CUDA devices", capture_started=True) is False
    assert is_transcript_line("   ", capture_started=True) is False


def test_extract_spoken_text_strips_the_timestamp_prefix():
    """A real timestamped transcription line is reduced to just its spoken words."""
    line = "[00:00:00.000 --> 00:00:05.920]   Thank you."
    assert extract_spoken_text(line) == "Thank you."
    # A plain line (no timestamp) is returned trimmed and unchanged.
    assert extract_spoken_text("  what is the capital of Kenya  ") == "what is the capital of Kenya"


def test_startup_noise_gate_emits_only_real_utterances():
    """startup_noise (D-M7-1): a synthetic whisper-stream log -- startup diagnostics,
    then the '[Start speaking]' banner, then two real utterances (one containing a
    colon to prove the filter is not dumb) -- surfaces ONLY the two utterances. Every
    engine line, and everything before the banner, is dropped."""
    log_lines = [
        "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16302 MiB):",
        "  Device 0: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, VMM: yes, VRAM: 16302 MiB",
        "load_backend: loaded CUDA backend from /locitize-test/whisper/ggml-cuda.dll",
        "SDL_main: using VAD, will transcribe on speech activity",
        "",
        CAPTURE_BANNER,
        "",
        "what is the capital of Kenya",
        "remind me at 5: buy milk",
    ]
    gate = StartupNoiseGate()
    surfaced = [gate.surfaced_text(line) for line in log_lines]
    emitted = [text for text in surfaced if text is not None]
    assert emitted == ["what is the capital of Kenya", "remind me at 5: buy milk"]
    # The three owner-observed CUDA lines were dropped, not surfaced.
    assert gate.capture_started is True


def test_startup_noise_gate_drops_everything_before_the_banner():
    """Even a natural-looking line printed BEFORE '[Start speaking]' is startup
    preamble, never speech, so it is dropped until the banner is crossed."""
    gate = StartupNoiseGate()
    assert gate.surfaced_text("hello there") is None  # pre-banner: dropped
    assert gate.capture_started is False
    assert gate.surfaced_text(CAPTURE_BANNER) is None  # the banner itself is consumed
    assert gate.capture_started is True
    assert gate.surfaced_text("hello there") == "hello there"  # now surfaced


def test_startup_noise_gate_extracts_words_from_timestamped_lines():
    """After the banner, a real '[timestamp]   words' transcription line is reduced
    to its spoken words; a lone '.' placeholder still comes through as '.' for the
    downstream is_non_speech guard to drop."""
    gate = StartupNoiseGate()
    gate.surfaced_text(CAPTURE_BANNER)
    assert gate.surfaced_text("[00:00:00.000 --> 00:00:05.920]   Thank you.") == "Thank you."
    assert gate.surfaced_text("[00:00:00.000 --> 00:00:10.000]   .") == "."


# --------------------------------------------------------------------------- #
# D-M7-3b: capture-time timestamp parsing + wall-clock anchoring (keyword
# capture_overlap / spoken_text). The words are unchanged; the CAPTURE interval is
# now recovered so the half-duplex gate can drop echoes by when they were spoken.
# --------------------------------------------------------------------------- #


def test_parse_capture_interval_reads_start_and_end_seconds():
    """capture_overlap: whisper's '[hh:mm:ss.mmm --> ...]' span parses to relative
    (start_s, end_s) capture seconds; a line without a timestamp returns None."""
    assert parse_capture_interval("[00:00:03.000 --> 00:00:05.920]   Thank you.") == (
        3.0,
        5.92,
    )
    # hours/minutes carry correctly (1h 2m 3.5s = 3723.5s).
    assert parse_capture_interval("[01:02:03.500 --> 01:02:04.000] hi") == (
        3723.5,
        3724.0,
    )
    assert parse_capture_interval("what time is it") is None


def test_startup_noise_gate_anchors_absolute_capture_interval():
    """capture_overlap / block_header: the gate stamps a wall-clock anchor at the
    banner, then anchors each segment on its BLOCK HEADER cumulative interval (D-M7-3c
    -- NOT the resetting per-line span), exposing the absolute window the mic
    half-duplex overlap check reads."""
    clock = _FakeClock(start=100.0)
    gate = StartupNoiseGate(clock=clock)
    # Before the banner there is no anchor and no interval.
    gate.surfaced_text("ggml_cuda_init: found 1 CUDA devices")
    assert gate.anchor is None
    assert gate.last_capture_interval is None
    # Banner at wall-clock 100.0 sets the anchor.
    gate.surfaced_text(CAPTURE_BANNER)
    assert gate.anchor == 100.0
    # The block header carries the real cumulative capture time (3.0s-5.92s); it is
    # consumed (never surfaced) and applied to the block's transcript line.
    assert gate.surfaced_text("### Transcription 3 START | t0 = 3000 ms | t1 = 5920 ms") is None
    # The per-line span RESETS to 00:00:00 but the words still extract, and the
    # capture interval comes from the header -> absolute 103.0 .. 105.92.
    text = gate.surfaced_text("[00:00:00.000 --> 00:00:05.920]   Thank you.")
    assert text == "Thank you."
    assert gate.last_capture_interval == (103.0, 105.92)


def test_parse_block_header_reads_cumulative_ms():
    """block_header: the '### Transcription N START | t0 = <ms> | t1 = <ms>' header
    parses to (t0_s, t1_s); a per-line span or plain line is not a header (None)."""
    # The real owner-log header format (see logs/assistant_whisper_stream.log).
    assert parse_block_header("### Transcription 0 START | t0 = 0 ms | t1 = 5536 ms") == (
        0.0,
        5.536,
    )
    # A block 90s into the stream (the D-M7-3c freeze case: header says 90443ms).
    assert parse_block_header(
        "### Transcription 36 START | t0 = 90443 ms | t1 = 100443 ms"
    ) == (90.443, 100.443)
    assert parse_block_header("[00:00:00.000 --> 00:00:10.000] hi") is None
    assert parse_block_header("### Transcription 0 END") is None


def test_startup_noise_gate_turn2_uses_block_header_not_resetting_per_line():
    """block_header / capture_overlap (THE D-M7-3c freeze): the per-line timestamp
    resets to 00:00:00 every block, but the header t0/t1 are cumulative. A turn-2
    utterance 90s into the stream must anchor at ~anchor+90s (from the header), NOT
    ~anchor+0s (from the resetting per-line span) -- otherwise it overlaps turn 1's
    speaking interval forever and every post-turn-1 utterance is dropped -> frozen."""
    clock = _FakeClock(start=0.0)
    gate = StartupNoiseGate(clock=clock)
    gate.surfaced_text(CAPTURE_BANNER)  # anchor = 0.0

    # Turn 1, captured early (header t0=0..5.536s), per-line resets to 00:00:00.
    assert gate.surfaced_text("### Transcription 0 START | t0 = 0 ms | t1 = 5536 ms") is None
    assert gate.surfaced_text("[00:00:00.000 --> 00:00:07.800]   Hey, how are you?") == (
        "Hey, how are you?"
    )
    assert gate.last_capture_interval == (0.0, 5.536)

    # Turn 2, captured 90s in. The per-line span STILL reads 00:00:00 (the reset that
    # broke D-M7-3b), but the header says t0 = 90443 ms -> absolute 90.443..100.443,
    # well clear of turn 1's early interval, so it will NOT be dropped as self-echo.
    assert gate.surfaced_text(
        "### Transcription 36 START | t0 = 90443 ms | t1 = 100443 ms"
    ) is None
    assert gate.surfaced_text("[00:00:00.000 --> 00:00:10.000]   What's your name?") == (
        "What's your name?"
    )
    assert gate.last_capture_interval == (90.443, 100.443)


def test_whisper_converts_non_wav_audio_when_ffmpeg_is_available(monkeypatch, tmp_path):
    """Owner-observed 2026-09-03: Open WebUI's voice mode records webm/opus (what
    a browser MediaRecorder produces) and whisper-server reads WAV, so every
    utterance failed. whisper-server's own --convert flag handles it."""
    import shutil as _shutil

    import whisper as whisper_module
    from config import Settings

    monkeypatch.setattr(_shutil, "which", lambda name: "ffmpeg.exe")
    settings = Settings()
    settings.data_dir = str(tmp_path)
    settings.paths.whisper = "whisper-server.exe"
    settings.paths.whisper_model = "model.bin"
    command = [str(a) for a in whisper_module.build_whisper_server_spec(settings).command]
    assert "--convert" in command
    # --tmp-dir defaults to "." - the service cwd - so transcoded files would
    # land in the install tree. It must point at the data root instead.
    assert "--tmp-dir" in command
    assert str(tmp_path) in command[command.index("--tmp-dir") + 1]


def test_whisper_does_not_promise_conversion_without_ffmpeg(monkeypatch, tmp_path):
    """Passing a flag whose external dependency is missing would turn a working
    WAV-only setup into a broken one."""
    import shutil as _shutil

    import whisper as whisper_module
    from config import Settings

    monkeypatch.setattr(_shutil, "which", lambda name: None)
    settings = Settings()
    settings.data_dir = str(tmp_path)
    settings.paths.whisper = "whisper-server.exe"
    settings.paths.whisper_model = "model.bin"
    command = [str(a) for a in whisper_module.build_whisper_server_spec(settings).command]
    assert "--convert" not in command


def test_whisper_server_threads_follow_the_machine(monkeypatch):
    """Owner request 2026-09-03 (voice latency): whisper-server defaults to four
    threads on any machine, and transcription was the largest cost in a spoken
    turn. Half the logical cores, capped at 8, floored at whisper's own 4."""
    import os as _os

    import whisper as whisper_module

    def threads_for(cores):
        monkeypatch.setattr(_os, "cpu_count", lambda: cores)
        command = [str(a) for a in whisper_module.build_whisper_server_spec(_server_settings()).command]
        return int(command[command.index("--threads") + 1])

    assert threads_for(20) == 8  # this machine: 10 would starve the LLM and Kokoro
    assert threads_for(12) == 6
    assert threads_for(4) == 4  # never below whisper's own default
    assert threads_for(None) == 4  # cpu_count unknown -> whisper's default
