"""Assistant-loop tests (AC3/AC4/AC5, Architecture M7.1/M7.3/M7.4/M7.5/M7.10).

Fully headless: fake STT (canned utterances), fake streaming LLM (canned deltas),
and fake TTS sink (records speak calls) -- no microphone, no GPU, no audio. Covers
the full-turn orchestration and interrupt (keyword 'assistant_loop'), context-budget
trimming (keyword 'assistant_history'), and sentence segmentation (keyword
'assistant_segment').
"""

from __future__ import annotations

from pathlib import Path
from threading import Event

from assistant import (
    PUSH_TO_TALK_NO_SPEECH,
    PUSH_TO_TALK_PROMPT,
    VOICE_MODE_SYSTEM_PROMPT,
    AssistantLoop,
    ConversationState,
    HalfDuplexGate,
    PushToTalkSttSource,
    SentenceSegmenter,
    clean_for_speech,
    drop_captured_segment,
    drop_if_gated,
    strip_markdown,
    strip_speech_glyphs,
    trim_history,
)
from fakes import FakeSttSource, FakeStreamingLlm, FakeTtsSink
from llm import ChatMessage
from memory import ConversationMemory


def _state_with_system() -> ConversationState:
    """A session seeded with a leading system message (M7.3)."""
    state = ConversationState(session_id="test-session")
    state.append("system", "You are locitize.")
    return state


def _make_wav(seconds: float = 0.1, sample_rate: int = 24000, value: int = 1000) -> bytes:
    """Build a real 24kHz mono 16-bit wav of non-silent samples (D-M7-7 sink tests).

    Kokoro produces this exact format; a constant non-zero sample (default 1000) lets a
    test prove real audio survived concatenation and where the silence pads/gaps sit.
    """
    import io
    import wave

    n = int(sample_rate * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(value.to_bytes(2, "little", signed=True) * n)
    return buf.getvalue()


def _wav_frames(wav_bytes: bytes) -> tuple[int, int, int, bytes]:
    """Decode a wav to (framerate, nframes, sampwidth*nchannels, raw frame bytes)."""
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as src:
        return (
            src.getframerate(),
            src.getnframes(),
            src.getsampwidth() * src.getnchannels(),
            src.readframes(src.getnframes()),
        )


# --------------------------------------------------------------------------- #
# D-M7-2: markdown stripping for speech/display, and the voice-mode system prompt.
# --------------------------------------------------------------------------- #


def test_markdown_stripper_produces_clean_plain_speech():
    """The canned defect-style markdown -- bold, ### headers, ```bash fences```, and
    bullets -- strips to plain spoken text: no markers survive, the words stay, and
    the fenced command text is kept (not read as 'backtick backtick backtick')."""
    markdown = (
        "### Setup Steps\n"
        "Here is **bold** advice and *emphasis* on the plan.\n"
        "- first install the tool\n"
        "- then run it\n"
        "Use the `pip install kokoro` command, like:\n"
        "```bash\n"
        "pip install kokoro\n"
        "```\n"
    )
    plain = strip_markdown(markdown)
    # No markdown syntax characters remain.
    for marker in ["**", "###", "```", "`"]:
        assert marker not in plain, marker
    # There is no leading "- " bullet marker left on any line.
    for line in plain.splitlines():
        assert not line.lstrip().startswith("- ")
    # The words survive, including the code the fence contained.
    assert "bold" in plain
    assert "emphasis" in plain
    assert "Setup Steps" in plain
    assert "first install the tool" in plain
    assert "pip install kokoro" in plain


def test_markdown_stripper_keeps_snake_case_and_plain_text_untouched():
    """Underscore identifiers must survive (underscore emphasis is not stripped), and
    already-plain prose passes through unchanged."""
    assert strip_markdown("call whisper_init_state before capture") == (
        "call whisper_init_state before capture"
    )
    assert strip_markdown("The capital of Kenya is Nairobi.") == (
        "The capital of Kenya is Nairobi."
    )


def test_assistant_loop_strips_markdown_before_speaking_and_display():
    """voice_prompt / D-M7-2 wiring: with strip_markdown injected as text_filter, the
    sink receives plain text (no '**') and the displayed reply is stripped too, while
    the stored history keeps the model's original markdown faithful."""
    stt = FakeSttSource(["explain"])
    # A one-sentence reply carrying bold markers the model should never have spoken.
    llm = FakeStreamingLlm(["This is **very** important.", ""], token_count=3)
    tts = FakeTtsSink()
    lines: list[str] = []
    state = _state_with_system()
    loop = AssistantLoop(
        stt,
        llm,
        tts,
        state,
        voice="am_michael",
        speaking=True,
        emit=lines.append,
        text_filter=strip_markdown,
    )

    loop.run()

    # Spoken text has no markdown markers.
    assert tts.spoken == [("This is very important.", "am_michael")]
    # Displayed reply is stripped.
    assert lines == ["locitize: This is very important."]
    # Stored assistant history keeps the original (unfiltered) reply.
    assert state.messages[-1].content == "This is **very** important."


def test_voice_mode_system_prompt_is_plain_text_and_concise_instruction():
    """voice_prompt content: the injected voice-mode instruction tells the model to
    avoid markdown and stay brief -- the conditioning half of D-M7-2."""
    lowered = VOICE_MODE_SYSTEM_PROMPT.lower()
    assert "markdown" in lowered
    assert "plain" in lowered
    assert "1-3 sentences" in lowered
    # ASCII-clean, per the character-hygiene rule.
    assert VOICE_MODE_SYSTEM_PROMPT.isascii()


# --------------------------------------------------------------------------- #
# assistant_loop: full-turn orchestration + interrupt
# --------------------------------------------------------------------------- #


def test_assistant_loop_full_turn_speaks_each_sentence():
    """One turn: reply text, per-sentence speak calls, and state append all correct."""
    stt = FakeSttSource(["Tell me a greeting."])
    # Deltas that resolve to two sentences once boundaries are confirmed.
    llm = FakeStreamingLlm(["Hello", " there.", " How", " are you?"], token_count=3)
    tts = FakeTtsSink()
    state = _state_with_system()
    loop = AssistantLoop(
        stt, llm, tts, state, voice="am_michael", speaking=True, context_size=4096
    )

    loop.run()

    # Full reply is the concatenation of all deltas.
    assert state.messages[-1].role == "assistant"
    assert state.messages[-1].content == "Hello there. How are you?"
    # The user turn was appended before the assistant reply.
    assert [m.role for m in state.messages] == ["system", "user", "assistant"]
    # Spoken per sentence, in order, in the chosen voice.
    assert tts.spoken == [
        ("Hello there.", "am_michael"),
        ("How are you?", "am_michael"),
    ]


def test_assistant_loop_records_real_timings():
    """A turn records real per-stage TurnTimings (llm first-token + total present)."""
    captured = []
    stt = FakeSttSource(["hi"])
    llm = FakeStreamingLlm(["Hi.", " Bye."], token_count=2)
    loop = AssistantLoop(
        stt,
        llm,
        FakeTtsSink(),
        _state_with_system(),
        speaking=True,
        on_timings=captured.append,
    )

    loop.run()

    assert len(captured) == 1
    t = captured[0]
    # Text mode leaves stt at 0; the LLM stages are real wall-clock (>= 0).
    assert t.stt_ms == 0.0
    assert t.llm_first_token_ms >= 0.0
    assert t.llm_total_ms >= t.llm_first_token_ms


def test_assistant_loop_no_speak_prints_reply_and_never_speaks():
    """--no-speak text mode: reply is produced and printed, TtsSink is untouched."""
    lines: list[str] = []
    stt = FakeSttSource(["hello"])
    llm = FakeStreamingLlm(["Response text."], token_count=2)
    tts = FakeTtsSink()
    loop = AssistantLoop(
        stt, llm, tts, _state_with_system(), speaking=False, emit=lines.append
    )

    loop.run()

    assert tts.spoken == []  # no audio when speaking is off
    assert any("Response text." in line for line in lines)


def test_assistant_loop_interrupt_stops_and_drains():
    """Setting the interrupt mid-stream stops consumption and drains pending audio."""
    stt = FakeSttSource(["go"])
    # stop_after=1: the fake sets the shared interrupt after the first delta.
    llm = FakeStreamingLlm(["First. ", "Second. ", "Third. "], stop_after=1)
    tts = FakeTtsSink()
    state = _state_with_system()
    interrupt = Event()
    loop = AssistantLoop(
        stt, llm, tts, state, speaking=True, interrupt=interrupt
    )

    reply, _timings = loop.run_turn("go")

    # Only the first delta was consumed before the interrupt fired.
    assert reply == "First. "
    # The one completed sentence was spoken; pending playback was drained.
    assert tts.spoken == [("First.", "")]
    assert tts.drained >= 1


def test_kokoro_sink_drain_purges_in_flight_reply_and_clears_buffer():
    """Review M-2 under the D-M7-7 buffered contract: drain() must drop the buffered
    reply AND purge the single clip already playing, so a barge-in interrupt stops
    audio instead of letting the whole concatenated reply finish. Uses a fake client
    (no server) and a fake purge (no winsound)."""
    from assistant import KokoroTtsSink

    purged: list[bool] = []

    class _FakeClient:
        def synthesize(self, *args, **kwargs):  # pragma: no cover - not driven here
            return b""

    sink = KokoroTtsSink(_FakeClient(), purge=lambda: purged.append(True))
    # Buffer a reply's sentences without finishing so the test stays synchronous.
    sink.speak("pending sentence", "am_michael")

    sink.drain()

    assert sink._reply_buffer == []  # not-yet-played reply dropped
    assert sink._queue == []
    assert purged == [True]  # in-flight clip purge issued exactly once


def test_kokoro_sink_drain_purge_failure_is_swallowed():
    """A purge that raises must not break interrupt handling (drain stays clean)."""
    from assistant import KokoroTtsSink

    class _FakeClient:
        def synthesize(self, *args, **kwargs):  # pragma: no cover - not driven here
            return b""

    def _boom() -> None:
        raise RuntimeError("no audio device")

    sink = KokoroTtsSink(_FakeClient(), purge=_boom)
    sink.speak("pending", "am_michael")

    sink.drain()  # must not raise despite the failing purge

    assert sink._reply_buffer == []


# --------------------------------------------------------------------------- #
# D-M7-3: half-duplex mic gate (keyword half_duplex / barge). The blocking AC17 fix:
# the assistant must never transcribe its own TTS back as a user turn.
# --------------------------------------------------------------------------- #


class _FakeClock:
    """A controllable monotonic clock so the tail window is tested without sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_half_duplex_gate_drops_segments_while_speaking_and_in_tail():
    """A segment captured while the gate is speaking, or within the tail after the
    last clip, is dropped; a segment after the tail passes (D-M7-3)."""
    clock = _FakeClock()
    gate = HalfDuplexGate(tail_s=0.8, clock=clock)

    # Not speaking yet: a real user segment passes.
    assert drop_if_gated("hello there", gate) is False

    # A clip starts playing -> gate is set -> the assistant's own audio is dropped.
    gate.begin()
    assert gate.is_gated() is True
    assert drop_if_gated("We face with smiling eyes", gate) is True  # the AC17 echo

    # Clip finishes; still inside the 0.8s tail -> whisper's fading-edge echo dropped.
    gate.end()
    clock.advance(0.5)
    assert gate.is_gated() is True
    assert drop_if_gated("For example, I can answer questions", gate) is True

    # Past the tail -> the mic reopens and a genuine next utterance passes through.
    clock.advance(0.4)  # 0.9s total since end() > 0.8s tail
    assert gate.is_gated() is False
    assert drop_if_gated("what time is it", gate) is False


def test_half_duplex_gate_interrupt_clears_the_gate():
    """A barge-in interrupt (clear) reopens the mic immediately, before the tail
    would have expired (D-M7-3: interrupt must clear the gate)."""
    clock = _FakeClock()
    gate = HalfDuplexGate(tail_s=5.0, clock=clock)
    gate.begin()
    assert gate.is_gated() is True

    gate.clear()  # barge-in

    assert gate.is_gated() is False
    assert drop_if_gated("stop and listen to me", gate) is False


def test_half_duplex_gate_stays_gated_across_a_multi_clip_burst():
    """Speaking several queued sentences keeps the gate continuously set until the
    LAST clip's tail expires, so no gap lets an echo through mid-reply."""
    clock = _FakeClock()
    gate = HalfDuplexGate(tail_s=0.8, clock=clock)
    gate.begin()  # sentence 1 starts
    gate.begin()  # sentence 2 starts before 1 ended (queued back-to-back)
    gate.end()    # sentence 1 done, sentence 2 still playing
    assert gate.is_gated() is True  # still one clip active
    gate.end()    # sentence 2 done; tail begins now
    clock.advance(0.5)
    assert gate.is_gated() is True
    clock.advance(0.4)
    assert gate.is_gated() is False


def test_half_duplex_none_gate_never_drops():
    """With no gate (text mode / no half-duplex) nothing is ever dropped."""
    assert drop_if_gated("anything at all", None) is False


def test_kokoro_sink_sets_and_clears_half_duplex_gate_around_playback():
    """barge / half_duplex wiring under D-M7-7: KokoroTtsSink begin's the gate before
    the single reply clip plays and end's it after, so the mic is gated for the WHOLE
    playback; drain (barge-in interrupt) clears it. A fake player blocks until released
    so the test observes the gate mid-playback deterministically."""
    from threading import Event as _Event

    from assistant import KokoroTtsSink

    clock = _FakeClock()
    gate = HalfDuplexGate(tail_s=0.8, clock=clock)
    playing = _Event()
    release = _Event()

    class _RealWavClient:
        def synthesize(self, *args, **kwargs):
            return _make_wav(seconds=0.05)  # a real 24kHz mono 16-bit clip

    def _blocking_player(_path):
        playing.set()          # signal playback has started
        release.wait(2.0)      # hold "playback" so the test inspects the gate

    sink = KokoroTtsSink(_RealWavClient(), speaking_gate=gate, player=_blocking_player)
    assert gate.is_gated() is False  # nothing playing yet

    sink.speak("hello there", "am_michael")
    assert gate.is_gated() is False  # buffered only -- no audio, no gate yet (D-M7-7)
    sink.finish()                    # flush the reply -> worker plays the single wav
    assert playing.wait(2.0)         # worker started the single reply playback
    assert gate.is_gated() is True   # mic gated for the whole reply (D-M7-3)

    release.set()                    # let the reply finish
    sink.close()                     # worker runs end() -> tail starts

    # Inside the tail the gate stays set; after it, the mic reopens.
    assert gate.is_gated() is True
    clock.advance(1.0)
    assert gate.is_gated() is False


def test_kokoro_sink_drain_clears_half_duplex_gate():
    """A barge-in drain() clears the gate at once (does not wait for the tail)."""
    from assistant import KokoroTtsSink

    clock = _FakeClock()
    gate = HalfDuplexGate(tail_s=10.0, clock=clock)

    class _FakeClient:
        def speak(self, *args, **kwargs):  # pragma: no cover - not driven here
            return None

    sink = KokoroTtsSink(_FakeClient(), speaking_gate=gate)
    gate.begin()  # simulate a clip in flight
    assert gate.is_gated() is True

    sink.drain()

    assert gate.is_gated() is False  # interrupt reopened the mic immediately


def test_assistant_loop_interrupt_clears_half_duplex_gate():
    """The loop owns the gate: an interrupt during a turn drains TTS and clears the
    gate so the next turn's mic is open (D-M7-3)."""
    clock = _FakeClock()
    gate = HalfDuplexGate(tail_s=10.0, clock=clock)
    gate.begin()  # pretend the sink was mid-clip when the interrupt fired
    stt = FakeSttSource(["go"])
    llm = FakeStreamingLlm(["First. ", "Second. "], stop_after=1)
    tts = FakeTtsSink()
    state = _state_with_system()
    interrupt = Event()
    loop = AssistantLoop(
        stt, llm, tts, state, speaking=True, interrupt=interrupt, half_duplex_gate=gate
    )

    loop.run_turn("go")

    assert gate.is_gated() is False  # interrupt path cleared the loop-owned gate


# --------------------------------------------------------------------------- #
# First-sentence playback (keyword buffered_playback / reply_audio).
# speak() queues each sentence immediately so the first sentence starts while
# the LLM is still generating. A short lead pad keeps the first phoneme intact.
# --------------------------------------------------------------------------- #


class _RecordingBufferSink:
    """Records speak() and finish() calls for the loop-integration test (no audio)."""

    def __init__(self) -> None:
        self.spoken: list[tuple[str, str]] = []
        self.finishes = 0

    def speak(self, text: str, voice: str) -> None:
        self.spoken.append((text, voice))

    def finish(self) -> None:
        self.finishes += 1


def test_buffered_playback_starts_on_first_sentence(tmp_path):
    """buffered_playback: the first speak() starts playback without waiting for
    finish(), so the owner hears the first sentence while later ones enqueue.
    Only the first clip of a burst carries the lead pad."""
    from assistant import KokoroTtsSink

    durations = {"one": 0.10, "two": 0.05, "three": 0.08}

    class _Client:
        def synthesize(self, text, voice, speed):
            return _make_wav(seconds=durations[text])

    played: list[bytes] = []

    def _player(path):
        played.append(Path(path).read_bytes())

    lead_ms, gap_ms = 120, 150
    sink = KokoroTtsSink(
        _Client(), wav_dir=tmp_path, player=_player,
        lead_silence_ms=lead_ms, gap_ms=gap_ms,
    )
    for sentence in ("one", "two", "three"):
        sink.speak(sentence, "am_michael")
    sink.finish()
    sink.close()

    assert len(played) == 3
    lead = int(24000 * lead_ms / 1000)
    first_rate, first_frames, frame_size, first_raw = _wav_frames(played[0])
    assert first_rate == 24000
    assert first_frames == lead + int(24000 * durations["one"])
    assert first_raw[: lead * frame_size] == b"\x00" * (lead * frame_size)
    assert first_raw[lead * frame_size: lead * frame_size + 2] != b"\x00\x00"
    second_rate, second_frames, _, second_raw = _wav_frames(played[1])
    assert second_rate == 24000
    # Later sentences do not repeat the device-spin-up pad.
    assert second_frames == int(24000 * durations["two"])
    assert second_raw[:2] != b"\x00\x00"
    assert sink.last_wav is not None and Path(sink.last_wav).parent == tmp_path


def test_reply_audio_interrupt_stops_single_playback():
    """reply_audio: a barge-in interrupt during playback purges it (stops audio)
    and clears the buffer/queue -- audio does not run to completion."""
    from threading import Event as _Event

    from assistant import KokoroTtsSink

    playing = _Event()
    release = _Event()

    class _Client:
        def synthesize(self, text, voice, speed):
            return _make_wav(seconds=0.05)

    def _player(_path):
        playing.set()
        release.wait(2.0)  # hold the single playback so the test can barge in

    purged: list[bool] = []
    sink = KokoroTtsSink(_Client(), player=_player, purge=lambda: purged.append(True))
    sink.speak("first sentence", "am_michael")

    assert playing.wait(2.0)  # first sentence is already playing
    sink.drain()              # barge-in interrupt mid-playback

    assert purged == [True]   # the in-flight single playback was purged (stopped)
    release.set()
    sink.close()
    assert sink._reply_buffer == []
    assert sink._queue == []


def test_buffered_playback_loop_finishes_reply_once():
    """buffered_playback: the loop speaks each sentence as it completes, then
    calls finish() once so any tail is flushed."""
    stt = FakeSttSource(["Tell me a greeting."])
    llm = FakeStreamingLlm(["Hello there.", " How are you?"], token_count=3)
    sink = _RecordingBufferSink()
    loop = AssistantLoop(
        stt, llm, sink, _state_with_system(), voice="am_michael", speaking=True
    )

    loop.run()

    assert sink.spoken == [
        ("Hello there.", "am_michael"),
        ("How are you?", "am_michael"),
    ]
    assert sink.finishes == 1


def test_buffered_playback_skips_finish_on_interrupt():
    """buffered_playback: an interrupted reply drains rather than finishing, so a
    half-produced reply is never flushed to a single playback (barge-in path)."""
    stt = FakeSttSource(["go"])
    # stop_after=1: the interrupt fires after the first delta.
    llm = FakeStreamingLlm(["First. ", "Second. "], stop_after=1)

    class _Sink(_RecordingBufferSink):
        def __init__(self):
            super().__init__()
            self.drained = 0

        def drain(self):
            self.drained += 1

    sink = _Sink()
    loop = AssistantLoop(
        stt, llm, sink, _state_with_system(), speaking=True, interrupt=Event()
    )

    loop.run_turn("go")

    assert sink.finishes == 0  # interrupted: no single-playback flush
    assert sink.drained >= 1   # audio was drained instead


# --------------------------------------------------------------------------- #
# D-M7-3b: DETERMINISTIC capture-time discard (keyword half_duplex / capture_overlap
# / barge). The reopener: the trailing self-spoken sentence is transcribed AFTER
# playback ends -- past the poll-time tail -- so a poll-time check reads it ungated.
# Correctness must come from CAPTURE-time interval overlap, race-free at any read lag.
# --------------------------------------------------------------------------- #


def test_capture_overlap_drops_late_read_echo_beyond_tail():
    """capture_overlap (THE race that blocked D-M7-3): a segment whose capture window
    overlaps a recorded speaking interval is dropped EVEN when read many polls later,
    well past the 0.8s tail -- the exact trailing self-echo that made it 'repeat'.
    The poll-time flag alone MISSES it (proving why capture-time is required)."""
    clock = _FakeClock()
    gate = HalfDuplexGate(tail_s=0.8, clock=clock)
    guard = 0.5

    # The assistant speaks from wall-clock t=10.0 to t=12.0 (recorded interval).
    clock.now = 10.0
    gate.begin()
    clock.now = 12.0
    gate.end()

    # Now jump far past the tail: it is t=20.0, 8s after playback ended.
    clock.now = 20.0
    assert gate.is_gated() is False  # poll-time flag has long since reopened the mic

    # whisper only NOW transcribes the assistant's trailing sentence -- but that audio
    # was CAPTURED at 10.5s-11.5s (during playback). Capture-time overlap drops it.
    echo_interval = (10.5, 11.5)
    # Poll-time filter fails to catch it (this is the reopener bug):
    assert drop_if_gated("we face with smiling eyes", gate) is False
    # Capture-time filter catches it deterministically, regardless of read lag:
    assert (
        drop_captured_segment("we face with smiling eyes", gate, echo_interval, guard)
        is True
    )

    # A segment CAPTURED strictly after the assistant finished (real user speech at
    # 13.0s-14.0s) does NOT overlap and passes -- the mic is genuinely open.
    assert (
        drop_captured_segment("what time is it", gate, (13.0, 14.0), guard) is False
    )


def test_capture_overlap_none_interval_falls_back_to_poll_flag():
    """capture_overlap: a segment with no timestamp (no capture interval) falls back
    to the cheap poll-time flag, preserving the legacy first-line behaviour."""
    clock = _FakeClock()
    gate = HalfDuplexGate(tail_s=0.8, clock=clock)
    gate.begin()  # speaking now -> flag set
    assert drop_captured_segment("untimed echo", gate, None, 0.5) is True
    gate.clear()  # barge / done
    assert drop_captured_segment("untimed real speech", gate, None, 0.5) is False
    # No gate at all -> never dropped.
    assert drop_captured_segment("anything", None, (0.0, 1.0), 0.5) is False


def test_capture_overlap_barge_drops_pre_barge_audio_read_after_clear():
    """barge / capture_overlap: on barge-in the gate closes the live interval at the
    barge moment and KEEPS it, so pre-barge assistant audio still buffered in whisper
    is dropped even when read AFTER clear(); genuine speech captured strictly after
    the barge passes and reopens the mic promptly (D-M7-3b point 4)."""
    clock = _FakeClock()
    gate = HalfDuplexGate(tail_s=10.0, clock=clock)
    guard = 0.5

    # Assistant is mid-clip; the owner barges in at wall-clock t=11.0.
    clock.now = 10.0
    gate.begin()
    clock.now = 11.0
    gate.clear()  # barge-in: closes the interval at 11.0, keeps the record

    # A pre-barge segment (captured 10.2s-10.8s) is only transcribed/read now, at t=15.
    clock.now = 15.0
    assert gate.is_gated() is False  # flag reopened immediately on barge
    assert drop_captured_segment("pre barge echo", gate, (10.2, 10.8), guard) is True

    # Real user speech captured strictly after the barge (11.6s-12.5s) passes.
    assert drop_captured_segment("stop and listen", gate, (11.6, 12.5), guard) is False


def test_capture_overlap_drops_segment_captured_during_live_clip():
    """capture_overlap: while a clip is still playing (open interval, not yet ended),
    a segment captured after it began overlaps and is dropped."""
    clock = _FakeClock()
    gate = HalfDuplexGate(tail_s=0.8, clock=clock)
    clock.now = 5.0
    gate.begin()  # clip still playing (no end yet)
    clock.now = 6.0
    # Segment captured 5.3s-5.8s, during the live clip -> dropped.
    assert drop_captured_segment("live echo", gate, (5.3, 5.8), 0.5) is True


# --------------------------------------------------------------------------- #
# D-M7-6: push-to-talk capture (keyword push_to_talk). The mic feeds the assistant
# ONLY inside an Enter-opened window; everything captured while it is closed (idle
# ambient noise, the assistant's own TTS) is discarded. Headless via a fake mic +
# scripted Enter input.
# --------------------------------------------------------------------------- #


class _FakeMic:
    """A fake MicSegments seam for the push-to-talk tests.

    `closed` are segments already captured while the window was closed (ambient
    noise / self-echo); flush() drops them, modelling the structural discard. `open_`
    are segments that arrive after the window opens; poll_segment() yields them one
    per call, then None (silence). A None in `open_` models a silent poll (no speech
    yet) so the settle/timeout logic is exercised.
    """

    def __init__(self, closed=None, open_=None) -> None:
        self._closed = list(closed or [])
        self._open = list(open_ or [])
        self.flush_calls = 0
        self.flushed = False

    def flush(self) -> None:
        self.flush_calls += 1
        self.flushed = True
        self._closed.clear()  # closed-window capture is discarded here

    def poll_segment(self, timeout_s: float) -> str | None:
        # Before the window opens, closed-window junk would be returned (the bug);
        # after flush() it is gone. Then open-window segments arrive in order.
        if not self.flushed and self._closed:
            return self._closed.pop(0)
        if self._open:
            return self._open.pop(0)
        return None


def _enter_reader(lines):
    """A read_line callable that returns successive scripted Enter-lines, then None
    (EOF) to end the session -- the injected stdin for push_to_talk tests."""
    queue = list(lines)

    def read() -> str | None:
        return queue.pop(0) if queue else None

    return read


def test_push_to_talk_drops_closed_window_segments_keeps_open_window_utterance():
    """push_to_talk: ambient/self-echo captured while the window was CLOSED is dropped
    on flush; only speech captured after Enter (window open) becomes the utterance."""
    mic = _FakeMic(
        closed=["Thanks for watching", "we face with smiling eyes"],  # junk + self-echo
        open_=["what is the capital of Kenya", None],
    )
    src = PushToTalkSttSource(
        mic, read_line=_enter_reader([""]), settle_s=0.0, poll_s=0.0
    )
    utterance = src.next_utterance()
    assert utterance == "what is the capital of Kenya"
    assert mic.flush_calls == 1  # the window was opened (closed backlog discarded)


def test_push_to_talk_self_echo_while_closed_is_dropped():
    """push_to_talk: a self-echo/ambient segment present while the window is closed is
    discarded by flush and never returned as the utterance, even if no real speech
    follows -- the turn ends with the no-speech notice instead (no real-time wait: the
    injected clock advances on each silent poll so max_capture_s expires instantly)."""
    emitted: list[str] = []
    clock = _FakeClock()

    class _SilentAfterFlush:
        # Holds only closed-window self-echo; after flush there is nothing to return,
        # and each silent poll advances the clock so the capture window ends at once.
        def __init__(self):
            self.flushed = False

        def flush(self):
            self.flushed = True  # drops the closed-window self-echo

        def poll_segment(self, timeout_s):
            clock.now += 0.5
            return None

    src = PushToTalkSttSource(
        _SilentAfterFlush(),
        read_line=_enter_reader(["", None]),  # Enter once, then EOF to end
        emit=emitted.append,
        max_capture_s=1.0,
        settle_s=0.0,
        poll_s=0.0,
        clock=clock,
    )
    assert src.next_utterance() is None  # no intelligible speech -> session ends on EOF
    # The closed-window self-echo was never surfaced; the owner saw the no-speech note.
    assert PUSH_TO_TALK_NO_SPEECH in emitted
    assert "thank you" not in emitted


def test_push_to_talk_prints_prompt_and_returns_to_it_after_a_turn():
    """push_to_talk: the prompt is printed each turn and the loop returns to it after
    a completed turn -- two Enter presses yield two utterances, two prompts."""
    prompts: list[str] = []
    mic = _FakeMic(open_=["first question", None, "second question", None])
    src = PushToTalkSttSource(
        mic,
        read_line=_enter_reader(["", ""]),
        emit=prompts.append,
        settle_s=0.0,
        poll_s=0.0,
    )
    assert src.next_utterance() == "first question"
    assert src.next_utterance() == "second question"
    assert prompts.count(PUSH_TO_TALK_PROMPT) == 2  # returned to the prompt each turn


def test_push_to_talk_quit_word_and_eof_end_the_session():
    """push_to_talk: typing 'quit' at the Enter prompt, or EOF (None), ends the
    session cleanly (returns None) without opening a capture window."""
    mic_quit = _FakeMic(open_=["ignored"])
    assert PushToTalkSttSource(mic_quit, read_line=_enter_reader(["quit"])).next_utterance() is None
    assert mic_quit.flush_calls == 0  # never opened a window

    mic_eof = _FakeMic(open_=["ignored"])
    assert PushToTalkSttSource(mic_eof, read_line=_enter_reader([])).next_utterance() is None
    assert mic_eof.flush_calls == 0


def test_push_to_talk_gathers_multi_segment_utterance_within_settle():
    """push_to_talk: once speech starts, segments arriving within the settle window are
    joined into one utterance; a settle-length silence ends it."""
    clock = _FakeClock()

    class _SettleMic:
        # Two segments arrive back-to-back, then silence; with settle_s > 0 they join.
        def __init__(self):
            self.flushed = False
            self._segs = ["remind me at five", "to buy milk"]

        def flush(self):
            self.flushed = True

        def poll_segment(self, timeout_s):
            if self._segs:
                return self._segs.pop(0)
            clock.now += 1.0  # advance so the settle window eventually closes
            return None

    src = PushToTalkSttSource(
        _SettleMic(), read_line=_enter_reader([""]), settle_s=0.5, poll_s=0.0, clock=clock
    )
    assert src.next_utterance() == "remind me at five to buy milk"


# --------------------------------------------------------------------------- #
# D-M7-4: emoji / non-speech glyph stripping before TTS (keyword emoji / strip).
# --------------------------------------------------------------------------- #


def test_strip_speech_glyphs_removes_emoji_keeps_words():
    """The AC17 offender -- a reply ending in a smiley -- loses only the emoji; the
    words and punctuation stay, so Kokoro never voices 'face with smiling eyes'."""
    reply = "Sure, I can help with that. \U0001f60a"
    assert strip_speech_glyphs(reply) == "Sure, I can help with that."


def test_strip_speech_glyphs_handles_emoji_mid_sentence_and_pictographs():
    """Emoji between words and assorted pictographs/dingbats are removed and the
    surrounding spacing is tidied to single spaces."""
    reply = "Great \U0001f44d idea \U00002b50 and here \U00002705 we go"
    assert strip_speech_glyphs(reply) == "Great idea and here we go"


def test_strip_speech_glyphs_keeps_legit_accents_and_punctuation():
    """Accented letters, currency and ordinary punctuation are legitimate speech
    content and must survive the glyph strip (only decorative symbols go)."""
    text = "Cafe creme, naive resume: it costs 5 euros -- ok?"
    assert strip_speech_glyphs(text) == text
    accented = "El nino de la cancion esta aqui."
    assert strip_speech_glyphs(accented) == accented


def test_clean_for_speech_strips_markdown_then_emoji():
    """The composite pre-TTS filter removes BOTH markdown and emoji in one pass, so a
    bold emoji-bearing reply becomes clean spoken prose (D-M7-2 + D-M7-4)."""
    reply = "**Done!** \U0001f389 See the `run()` function."
    assert clean_for_speech(reply) == "Done! See the run() function."


# --------------------------------------------------------------------------- #
# assistant_history: context-budget trimming (pure function)
# --------------------------------------------------------------------------- #


def _words(text: str) -> int:
    """Deterministic token counter for the trim tests: one token per whitespace word."""
    return len(text.split())


def test_assistant_history_fits_under_budget_keeps_all():
    msgs = [ChatMessage("system", "S"), ChatMessage("user", "hello world")]
    assert trim_history(msgs, budget=100, counter=_words) == msgs


def test_assistant_history_drops_oldest_pair_when_over_budget():
    msgs = [
        ChatMessage("system", "S"),          # 1
        ChatMessage("user", "a a a"),        # 3  <- oldest droppable pair
        ChatMessage("assistant", "b b b"),   # 3  <-
        ChatMessage("user", "c"),            # 1
        ChatMessage("assistant", "d"),       # 1
        ChatMessage("user", "e"),            # 1  (current turn)
    ]
    # total = 10; budget 7 forces dropping the oldest user/assistant pair (6 tokens).
    result = trim_history(msgs, budget=7, counter=_words)

    assert [m.content for m in result] == ["S", "c", "d", "e"]


def test_assistant_history_never_drops_system_or_current_turn():
    msgs = [ChatMessage("system", "s s s"), ChatMessage("user", "u u u")]
    # budget 1 is impossible to meet, but neither the system message nor the current
    # user turn may ever be dropped, so the list is returned intact.
    result = trim_history(msgs, budget=1, counter=_words)

    assert result == msgs


# --------------------------------------------------------------------------- #
# assistant_segment: sentence segmentation (pure)
# --------------------------------------------------------------------------- #


def test_assistant_segment_flushes_on_boundaries():
    seg = SentenceSegmenter()
    assert seg.feed("Hello world. ") == ["Hello world."]
    assert seg.feed("Second one! Third") == ["Second one!"]
    assert seg.flush() == "Third"


def test_assistant_segment_does_not_split_decimals():
    seg = SentenceSegmenter()
    # The '.' inside 3.14 is followed by a digit (not whitespace) so it is not a
    # boundary; only the sentence-ending '.' after 'today' flushes.
    sentences = seg.feed("Pi is 3.14 today. ")
    assert sentences == ["Pi is 3.14 today."]
    assert "3.14" in sentences[0]


def test_assistant_segment_trailing_flush_at_stream_end():
    seg = SentenceSegmenter()
    # A terminator at the very end (no following whitespace yet) is held, then
    # released by flush() at stream end -- the trailing-flush case.
    assert seg.feed("No trailing space.") == []
    assert seg.flush() == "No trailing space."


# --------------------------------------------------------------------------- #
# assistant_loop: /recall command injects stored context on demand (M8.3)
# --------------------------------------------------------------------------- #


def test_assistant_loop_recall_command_injects_memory(tmp_path):
    """A /recall <query> turn searches memory and injects a system preamble.

    The recall turn does NOT call the LLM (the fake would record a message); it
    returns the matched snippets and appends them to state as a system preamble so
    later turns can use them (M8.3: explicit, on-demand recall).
    """
    memory = ConversationMemory(tmp_path / "memory")
    memory.append("old-session", "user", "we decided to defer the 27B target")

    stt = FakeSttSource(["/recall 27B"])
    llm = FakeStreamingLlm(["should-not-run"])
    state = ConversationState(session_id="new-session")
    state.append("system", "You are locitize.")
    loop = AssistantLoop(stt, llm, None, state, memory=memory, recall_limit=5)

    reply, _ = loop.run_turn("/recall 27B")

    # The matched snippet is surfaced and injected, and the LLM was never called.
    assert "27B target" in reply
    assert llm.received == []
    assert any(
        m.role == "system" and "27B target" in m.content for m in state.messages
    )


def test_assistant_loop_recall_no_match_is_honest(tmp_path):
    """A /recall with no stored match says so honestly (never fabricates context)."""
    memory = ConversationMemory(tmp_path / "memory")
    stt = FakeSttSource([])
    llm = FakeStreamingLlm([])
    state = ConversationState(session_id="s")
    loop = AssistantLoop(stt, llm, None, state, memory=memory)

    reply, _ = loop.run_turn("/recall nonexistent")
    assert "no stored conversation matched" in reply


# --------------------------------------------------------------------------- #
# M10.3: the optional KokoroTtsSink on_speaking hook (keyword: assistant_gui).
# --------------------------------------------------------------------------- #


def test_kokoro_sink_on_speaking_hook_fires_true_then_false_assistant_gui():
    """assistant_gui: the optional on_speaking hook fires True at the reply clip's
    playback begin and False at its end (driving the GUI Talk button), in the same
    begin()/end() bracket the half-duplex gate uses -- no D-M7-7 logic duplicated."""
    from threading import Event as _Event

    from assistant import KokoroTtsSink

    events: list[bool] = []
    playing = _Event()
    release = _Event()

    class _WavClient:
        def synthesize(self, *args, **kwargs):
            return _make_wav(seconds=0.02)  # a real 24kHz mono 16-bit clip

    def _player(_path):
        playing.set()
        release.wait(2.0)  # hold "playback" so the test sees the True state

    sink = KokoroTtsSink(
        _WavClient(),
        player=_player,
        on_speaking=lambda active: events.append(active),
    )
    sink.speak("hello there", "am_michael")
    assert events == []  # buffered only -- no audio, no hook yet (D-M7-7)
    sink.finish()  # flush the reply -> worker plays the single wav
    assert playing.wait(2.0)
    assert events == [True]  # speaking began
    release.set()
    sink.close()  # worker finishes the clip -> end() -> on_speaking(False)
    assert events[-1] is False
    assert events.count(True) == 1  # exactly one begin per reply


def test_kokoro_sink_none_on_speaking_hook_is_noop_assistant_gui():
    """assistant_gui: a None on_speaking hook (the CLI default) leaves buffered
    playback byte-for-byte unchanged -- the single reply clip still plays once."""
    from assistant import KokoroTtsSink

    played: list[str] = []

    class _WavClient:
        def synthesize(self, *args, **kwargs):
            return _make_wav(seconds=0.02)

    sink = KokoroTtsSink(_WavClient(), player=lambda p: played.append(p))
    sink.speak("hello there", "am_michael")
    sink.finish()
    sink.close()
    assert len(played) == 1  # the single reply clip still played, no hook needed
