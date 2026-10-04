"""The LOCITIZE built-in voice-assistant loop (M7, Architecture M7.1-M7.10).

This is the origin-file loop, finally real and finally testable:

    mic/text utterance  ->  LLM chat (streaming, history-trimmed)  ->
    reply printed + spoken sentence-by-sentence through Kokoro

The whole orchestration is unit-testable headless by injecting fakes at three
seams (Architecture M7.8):
  - SttSource : next_utterance() -> str | None   (mic path, text path, or a fake)
  - LlmClient : chat_stream(messages) -> deltas   (llm.py; a fake streams canned text)
  - TtsSink   : speak(text, voice) -> None         (Kokoro behind a playback queue,
                                                     a printing sink, or a recorder)
No audio hardware and no GPU are needed to prove the loop, because every effect is
an injected collaborator. The real mic and Kokoro adapters are wired in launcher.py
(reusing the M3 whisper-stream listen seam and the M6 KokoroClient); this module
owns the pure orchestration plus the two pure helpers the plan calls out as test
seams: `trim_history` (M7.3) and `SentenceSegmenter` (M7.4).

The loop never spawns a process. It assumes a chat model is already RUNNING (the
launcher ensures that via the existing ModelController before the first turn) and
speaks to it only through the injected LlmClient.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from llm import ChatMessage, LlmClient, LlmError


# --------------------------------------------------------------------------- #
# Voice-mode conditioning and markdown stripping (defect D-M7-2).
#
# When the assistant is speaking, raw markdown replies read terribly aloud (Kokoro
# voices "asterisk asterisk", "backtick backtick backtick bash") and 200-400 word
# answers are unusable spoken. Two fixes work together: an extra system message
# that asks the model to answer conversationally and briefly in plain text, and a
# defensive stripper that removes any markdown the model still emits before it is
# spoken or shown in --assistant.
# --------------------------------------------------------------------------- #

# Appended as a second system message in voice mode. Deliberately concrete about
# format (plain text, no markdown, 1-3 sentences) because vague guidance ("be
# concise") does not reliably stop a coding-tuned model from formatting.
VOICE_MODE_SYSTEM_PROMPT = (
    "You are answering out loud through a text-to-speech voice. Reply in plain, "
    "conversational spoken English with no markdown, no asterisks, no code fences, "
    "no headers, and no bullet lists. Do not use emojis or symbols; write only plain "
    "spoken words, since a voice cannot pronounce them. Keep answers to 1-3 sentences "
    "unless the user explicitly asks for more detail or a step-by-step list."
)

# A fenced code block: ```lang\n ... \n```. The spoken/displayed form keeps the
# inner code text (the words) and drops the fence markers and language tag. DOTALL
# so the block can span lines; non-greedy so adjacent fences do not merge.
_MD_FENCE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
# Inline `code` -> code. Runs after fences so it never eats a fence delimiter.
_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
# [label](url) -> label (speak the words, not the URL).
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# **bold** and *italic* -> inner text. Only asterisk emphasis is stripped;
# underscores are left alone so snake_case identifiers in a reply survive intact.
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_ITALIC = re.compile(r"\*(?!\s)(.+?)(?<!\s)\*", re.DOTALL)
# Leading "#" ATX headers, and leading bullet/ordered list markers, per line.
_MD_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MD_ORDERED = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
# Collapse 3+ blank lines a fence removal may leave into a single paragraph break.
_MD_BLANKRUN = re.compile(r"\n{3,}")


def strip_markdown(text: str) -> str:
    """Reduce markdown to plain spoken/displayable text, keeping the words (D-M7-2).

    Removes fences (keeping their inner code text), inline code backticks, link
    URLs (keeping the label), asterisk bold/italic markers, ATX "#" headers, and
    bullet/ordered list markers. Underscore emphasis is intentionally NOT stripped
    so identifiers like whisper_init_state survive. Order matters: fences before
    inline code, bold before italic. Whitespace and blank-line runs are tidied so
    the result flows as sentences rather than markdown layout.
    """
    result = _MD_FENCE.sub(lambda m: m.group(1), text)
    result = _MD_INLINE_CODE.sub(r"\1", result)
    result = _MD_LINK.sub(r"\1", result)
    result = _MD_BOLD.sub(r"\1", result)
    result = _MD_ITALIC.sub(r"\1", result)
    result = _MD_HEADER.sub("", result)
    result = _MD_BULLET.sub("", result)
    result = _MD_ORDERED.sub("", result)
    result = _MD_BLANKRUN.sub("\n\n", result)
    return result.strip()


# --------------------------------------------------------------------------- #
# Emoji / non-speech glyph stripping (defect D-M7-4).
#
# Kokoro pronounces an emoji literally ("face with smiling eyes"), which reads as
# nonsense aloud and -- combined with the mic feedback loop -- was transcribed back
# as a fake user turn. The pre-TTS cleanup therefore also removes emoji, pictographs,
# symbols and dingbats before synthesis. Only decorative symbol code points are
# removed; accented letters (cafe, naive), currency, and ordinary punctuation are
# left untouched so legitimate non-English speech text still synthesizes correctly.
# --------------------------------------------------------------------------- #

# Decorative-glyph code point ranges. Deliberately scoped to emoji/pictograph/symbol/
# dingbat blocks; it does NOT touch Latin-1/extended-Latin letters (accents),
# General Punctuation quotes/dashes, or currency, so real words survive.
_SPEECH_GLYPHS = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoji, emoticons, pictographs, transport, supplemental
    "\U00002600-\U000027BF"  # miscellaneous symbols and dingbats
    "\U00002B00-\U00002BFF"  # miscellaneous symbols and arrows (stars, etc.)
    "\U00002190-\U000021FF"  # arrows (decorative, not spoken)
    "\U00002300-\U000023FF"  # miscellaneous technical (hourglass, keyboard glyphs)
    "\U0000FE00-\U0000FE0F"  # variation selectors (emoji presentation)
    "\U0000200D"             # zero-width joiner (binds emoji sequences)
    "\U000024C2"             # circled Latin M (metro symbol)
    "]+"
)
# Collapse the spaces/tabs a removed glyph leaves behind, but keep newlines so
# paragraph structure (used by the sentence segmenter) is preserved.
_SPACE_RUN = re.compile(r"[ \t]{2,}")


def strip_speech_glyphs(text: str) -> str:
    """Remove emoji and other non-speech glyphs before TTS, keeping real words (D-M7-4).

    Strips emoji, pictographs, symbols and dingbats (the ranges in _SPEECH_GLYPHS),
    then tidies the spacing a removed glyph left. Accented letters, currency and
    ordinary punctuation are preserved, so a reply like "Cafe au lait, please."
    is untouched while "Sure thing! [smiley]" loses only the emoji.
    """
    cleaned = _SPEECH_GLYPHS.sub("", text)
    cleaned = _SPACE_RUN.sub(" ", cleaned)
    # Trim any spaces a glyph removal left hugging a newline or the string ends.
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return cleaned.strip()


def clean_for_speech(text: str) -> str:
    """Full pre-TTS text cleanup: strip markdown then non-speech glyphs (D-M7-2/D-M7-4).

    This is the filter the launcher injects as the assistant loop's text_filter so
    every spoken (and --assistant-displayed) piece of reply text is plain, glyph-free
    prose. Markdown is removed first (D-M7-2), then emoji/symbols (D-M7-4).
    """
    return strip_speech_glyphs(strip_markdown(text))


# --------------------------------------------------------------------------- #
# Half-duplex mic gate (defect D-M7-3, the AC17 blocker).
#
# Without gating, whisper-stream hears Kokoro's own spoken reply through the
# speakers and transcribes it back as a "user" turn, so the assistant answers
# itself and loops forever ("it's repeating"). The gate is a small shared state the
# loop owns: the TTS sink SETs it speaking while a clip plays and CLEARs it a
# configurable tail after the last clip finishes; the mic source CHECKs it and drops
# every segment captured inside the speaking+tail window. Interrupt clears it so a
# barge-in immediately reopens the mic.
# --------------------------------------------------------------------------- #


class HalfDuplexGate:
    """Shared speaking-state gate: records WHEN the assistant's own audio played.

    Two layers, because D-M7-3's poll-time-only gate raced and was reopened by
    Reviewer delta H1 (defect D-M7-3b):

    1. Cheap first-line flag (is_gated): True while a clip is playing and for
       `tail_s` after the last end(). This is only a fast pre-filter now, NOT the
       correctness guarantee -- in VAD mode whisper transcribes the assistant's
       trailing sentence AFTER playback ends (silence window + inference + poll),
       which can exceed the tail, so a poll-time check reads that echo ungated.

    2. Deterministic capture-time overlap (overlaps_speaking): the gate records the
       wall-clock (start, end) interval of every clip it played. A mic segment is
       dropped when its OWN capture window (recovered from whisper's per-segment
       timestamps, see whisper.StartupNoiseGate) overlaps any recorded speaking
       interval -- regardless of when the poll happens to read it. That is race-free:
       the trailing echo was CAPTURED during playback even though it is READ late.

    A small guard margin is added to each side of a recorded interval to absorb
    anchor/timestamp jitter. Recorded intervals persist across clear() (barge-in), so
    pre-barge assistant audio still buffered in whisper is dropped even when read
    after the gate is cleared; only the not-yet-closed live interval is closed at the
    barge moment, letting genuine user speech captured AFTER the barge pass promptly.

    Thread-safe: begin()/end()/clear() run on the TTS worker (or loop) thread while
    is_gated()/overlaps_speaking() are polled on the mic thread, all under one lock.
    The clock is injectable so behaviour is unit-testable without real time.
    """

    # Keep at most this many recent speaking intervals so a long session does not
    # grow the list unbounded; far more than any plausible in-flight capture lag.
    _MAX_INTERVALS = 128

    def __init__(
        self, tail_s: float = 0.8, clock: Callable[[], float] | None = None
    ) -> None:
        import threading

        # Never let a negative/absent tail defeat the gate; 0 is allowed (gate only
        # while actively speaking) but a nonsensical value floors at 0.
        self._tail_s = max(0.0, tail_s)
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._active_clips = 0  # >0 while one or more clips are playing
        self._last_end: float | None = None  # monotonic mark of the last end()
        # Wall-clock start of the currently-open speaking span (set when the first
        # clip of a burst begins, cleared when the burst fully ends or on barge).
        self._open_start: float | None = None
        # Closed (start, end) speaking intervals, kept for capture-time overlap.
        self._intervals: list[tuple[float, float]] = []

    def begin(self) -> None:
        """Mark that a clip has started playing (called by the TTS sink)."""
        with self._lock:
            if self._active_clips == 0:
                # First clip of a (possibly bursty) speaking span: open the interval.
                self._open_start = self._clock()
            self._active_clips += 1

    def end(self) -> None:
        """Mark that a clip finished; the tail window starts now (TTS sink)."""
        with self._lock:
            if self._active_clips > 0:
                self._active_clips -= 1
            self._last_end = self._clock()
            if self._active_clips == 0 and self._open_start is not None:
                # The whole span (all queued clips) is done: record its interval.
                self._record_interval(self._open_start, self._last_end)
                self._open_start = None

    def clear(self) -> None:
        """Reset the flag immediately (barge-in reopens the mic, D-M7-3).

        The RECORDED intervals are kept (D-M7-3b): pre-barge assistant audio still
        buffered in whisper overlaps them and must stay dropped even when read after
        this clear. If a clip was mid-flight, its interval is closed at the barge
        moment so speech captured strictly after the barge does not overlap it.
        """
        with self._lock:
            if self._open_start is not None:
                self._record_interval(self._open_start, self._clock())
                self._open_start = None
            self._active_clips = 0
            self._last_end = None

    def _record_interval(self, start: float, end: float) -> None:
        """Append a speaking interval, pruning the oldest to stay bounded (locked)."""
        self._intervals.append((start, end))
        if len(self._intervals) > self._MAX_INTERVALS:
            # Drop the oldest; only recent intervals can overlap in-flight captures.
            del self._intervals[: -self._MAX_INTERVALS]

    def is_gated(self) -> bool:
        """True while speaking, or within the tail after the last clip finished."""
        with self._lock:
            if self._active_clips > 0:
                return True
            if self._last_end is None:
                return False
            return (self._clock() - self._last_end) < self._tail_s

    def overlaps_speaking(
        self, capture_start: float, capture_end: float, guard_s: float = 0.0
    ) -> bool:
        """True if [capture_start, capture_end] overlaps any speaking interval.

        This is the deterministic, race-free half-duplex test (D-M7-3b). Each
        recorded interval is expanded by guard_s on both sides to absorb anchor and
        timestamp jitter. The currently-open (still-playing) span is also checked,
        treated as running from its start through now -- anything captured after an
        active clip began is the assistant's own live audio.
        """
        guard = max(0.0, guard_s)
        with self._lock:
            # Still-speaking span: open_start .. now (nothing captured after a clip
            # began, while it is still playing, can be genuine user speech).
            if self._open_start is not None:
                if self._intervals_overlap(
                    capture_start, capture_end,
                    self._open_start - guard, self._clock() + guard,
                ):
                    return True
            for start, end in self._intervals:
                if self._intervals_overlap(
                    capture_start, capture_end, start - guard, end + guard
                ):
                    return True
        return False

    @staticmethod
    def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
        """Standard closed-interval overlap test: [a0,a1] intersects [b0,b1]."""
        return a0 <= b1 and b0 <= a1


def drop_if_gated(segment: str, gate: "HalfDuplexGate | None") -> bool:
    """Return True when a segment must be dropped by the poll-time flag (D-M7-3).

    The cheap first-line filter: a segment read while the gate flag is set is very
    likely the assistant's own TTS. Correctness for the racing trailing echo comes
    from drop_captured_segment's capture-time overlap; this is retained as the fast
    pre-check and the fallback when a segment carries no usable timestamp. With
    gate=None (text mode) nothing is ever dropped.
    """
    return gate is not None and gate.is_gated()


def drop_captured_segment(
    segment: str,
    gate: "HalfDuplexGate | None",
    capture_interval: "tuple[float, float] | None",
    guard_s: float,
) -> bool:
    """Return True when a captured mic segment must be DROPPED (half-duplex, D-M7-3b).

    Deterministic path: when the segment's absolute capture interval is known, drop
    it iff that interval OVERLAPS a recorded speaking interval -- independent of when
    the poll read it, so the late-transcribed trailing echo is caught while genuine
    speech captured strictly after the assistant finished passes through. When no
    interval is available (a line without a timestamp, or before an anchor exists),
    fall back to the cheap poll-time flag. With gate=None nothing is dropped.
    """
    if gate is None:
        return False
    if capture_interval is not None:
        # Race-free: overlap decides, regardless of read time. A non-overlapping
        # segment is real user speech and passes even if is_gated() is still True.
        return gate.overlaps_speaking(
            capture_interval[0], capture_interval[1], guard_s
        )
    # No usable timestamp: best-effort poll-time flag (unchanged legacy behaviour).
    return gate.is_gated()


# --------------------------------------------------------------------------- #
# Seams (Architecture M7.8): the three injected collaborator protocols.
# --------------------------------------------------------------------------- #


@runtime_checkable
class SttSource(Protocol):
    """Source of owner utterances. Returns None to end the session (M7.8)."""

    def next_utterance(self) -> str | None:
        """Return the next utterance text, or None when input has ended."""


@runtime_checkable
class TtsSink(Protocol):
    """Sink that voices a piece of reply text in the chosen voice (M7.8)."""

    def speak(self, text: str, voice: str) -> None:
        """Voice `text` (real impl plays audio; test/no-speak impls do not)."""


# --------------------------------------------------------------------------- #
# Per-turn timing record (M7.6): measured wall-clock only, never estimated.
# --------------------------------------------------------------------------- #


@dataclass
class TurnTimings:
    """Real per-stage timings for one turn, in milliseconds (Data Model 9.4).

    stt_ms is 0 in text mode (no speech recognition happens). tts_first_audio_ms is
    0 when speaking is disabled. Every value is a real wall-clock measurement of the
    actual turn; nothing here is guessed (M7.6).
    """

    stt_ms: float = 0.0
    llm_first_token_ms: float = 0.0
    llm_total_ms: float = 0.0
    tts_first_audio_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "stt_ms": round(self.stt_ms, 1),
            "llm_first_token_ms": round(self.llm_first_token_ms, 1),
            "llm_total_ms": round(self.llm_total_ms, 1),
            "tts_first_audio_ms": round(self.tts_first_audio_ms, 1),
        }


# --------------------------------------------------------------------------- #
# Pure helper 1: history trimming to a context budget (M7.3, keyword assistant_history)
# --------------------------------------------------------------------------- #


def trim_history(
    messages: Sequence[ChatMessage],
    budget: int,
    counter: Callable[[str], int],
) -> list[ChatMessage]:
    """Drop the oldest turn pairs until the messages fit under `budget` tokens.

    Reuses M4's ctx-accounting discipline: `counter` returns a token count for a
    string (the server's /tokenize count, or a labeled chars/4 fallback). The
    leading system message(s) and the final (current user) turn are NEVER dropped;
    only the oldest droppable user/assistant pair is removed each pass until the
    estimated total fits, or nothing droppable remains. Pure function (no I/O), so
    it is the M7.3 headless test seam.
    """
    msgs = list(messages)

    def total_tokens() -> int:
        return sum(counter(m.content) for m in msgs)

    while total_tokens() > budget:
        # Skip leading system messages -- they are never dropped.
        start = 0
        while start < len(msgs) and msgs[start].role == "system":
            start += 1
        # The last message is the current user turn -- never dropped. There must be
        # at least one droppable message strictly before it.
        if start >= len(msgs) - 1:
            break
        # Drop the oldest turn as a pair (a user turn and, if present, the assistant
        # reply that followed it), so history is removed in coherent exchanges.
        del msgs[start]
        if start < len(msgs) - 1 and msgs[start].role == "assistant":
            del msgs[start]
    return msgs


# --------------------------------------------------------------------------- #
# Pure helper 2: sentence segmentation for speak-per-sentence (M7.4, keyword assistant_segment)
# --------------------------------------------------------------------------- #


class SentenceSegmenter:
    """Accumulate streamed deltas and flush complete sentences on a boundary.

    A boundary is a newline, or one of . ! ? that is FOLLOWED by whitespace. The
    "followed by whitespace" rule is what keeps a decimal like "3.14" intact: the
    "." there is followed by a digit, not whitespace, so it is not a boundary. A
    terminator sitting at the very end of the buffer (nothing after it yet) is held
    until the next delta reveals what follows, or until flush() at stream end.

    Overlapping generation and TTS (M7.4) uses this: feed each delta, speak each
    returned sentence while the LLM keeps generating, then speak flush() at the end.
    Pure and I/O-free, so it is the M7.4 headless test seam.
    """

    _TERMINATORS = ".!?"

    def __init__(self, min_len: int = 1) -> None:
        # Sentences shorter than min_len (after stripping) are not flushed on their
        # own; they accumulate into the next one. Guards against a stray tiny
        # fragment being spoken as if it were a sentence.
        self._buf = ""
        self._min_len = max(1, min_len)

    def feed(self, delta: str) -> list[str]:
        """Add a delta and return every complete sentence it completed (in order)."""
        self._buf += delta
        sentences: list[str] = []
        while True:
            idx = self._find_boundary()
            if idx is None:
                break
            chunk = self._buf[: idx + 1].strip()
            self._buf = self._buf[idx + 1:].lstrip()
            if len(chunk) >= self._min_len:
                sentences.append(chunk)
            elif chunk:
                # Too short to stand alone: fold it back so it joins the next
                # sentence rather than being spoken as a fragment.
                self._buf = chunk + " " + self._buf
                break
        return sentences

    def flush(self) -> str | None:
        """Return any remaining buffered text as a final sentence, or None."""
        remainder = self._buf.strip()
        self._buf = ""
        return remainder or None

    def _find_boundary(self) -> int | None:
        """Index of the first confirmed sentence-ending char in the buffer, or None."""
        buf = self._buf
        for i, ch in enumerate(buf):
            if ch == "\n":
                return i
            if ch in self._TERMINATORS:
                nxt = buf[i + 1] if i + 1 < len(buf) else ""
                if nxt == "":
                    # Terminator at end of buffer: hold until we see what follows
                    # (could be a decimal digit, could be whitespace).
                    continue
                if nxt.isspace():
                    return i
                # Followed by a non-space (e.g. the "1" in "3.14"): not a boundary.
        return None


# --------------------------------------------------------------------------- #
# Conversation state (M7.3): the ordered history the assistant sends each turn.
# --------------------------------------------------------------------------- #


@dataclass
class ConversationState:
    """Ordered ChatMessage history for one assistant session (Data Model 9.4)."""

    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        """Append one message to the conversation."""
        self.messages.append(ChatMessage(role=role, content=content))


# --------------------------------------------------------------------------- #
# Concrete text-mode source and printing sink (the headless / no-hardware path).
# --------------------------------------------------------------------------- #


class TextSttSource:
    """Text-mode STT: read one line per turn from an injected input callable.

    `read_line` defaults to input(); the scripted verifier passes a callable that
    pops lines from a fixed list. A None/empty-on-EOF read ends the session, and a
    line equal to 'quit'/'exit' also ends it cleanly (M7.7). No microphone.
    """

    _QUIT_WORDS = {"quit", "exit"}

    def __init__(self, read_line: Callable[[], str] | None = None) -> None:
        self._read_line = read_line or self._default_read

    @staticmethod
    def _default_read() -> str:
        # A real EOF (Ctrl-Z / closed stdin) raises EOFError; treat it as end.
        try:
            return input()
        except EOFError:
            return ""

    def next_utterance(self) -> str | None:
        line = self._read_line()
        if line is None:
            return None
        line = line.strip()
        if not line or line.lower() in self._QUIT_WORDS:
            return None
        return line


# --------------------------------------------------------------------------- #
# Push-to-talk STT source (defect D-M7-6): the DEFAULT voice-mode capture.
#
# Always-listening voice mode is unusable in a real open-mic room: whisper
# hallucinates user turns from ambient noise, and the assistant's own TTS is
# re-transcribed and answered in a 28-turn self-echo runaway. Push-to-talk removes
# BOTH failure classes structurally: the mic feeds the assistant ONLY during a
# window the owner opens by pressing Enter. Everything captured while the window is
# closed -- idle ambient noise, and the assistant's own speech while it talks -- is
# discarded. The half-duplex capture-time gate becomes a secondary net, not the
# primary mechanism.
# --------------------------------------------------------------------------- #


@runtime_checkable
class MicSegments(Protocol):
    """Minimal mic seam the push-to-talk source drives (defect D-M7-6, M7.8).

    flush() discards everything captured so far (the closed-window junk), and
    poll_segment() returns the next captured transcript segment or None within a
    timeout. The real implementation is launcher._MicSttSource (which tails the
    whisper-stream log); tests inject a fake so the turn flow is exercised headless.
    """

    def flush(self) -> None:
        """Discard whatever was captured while the window was closed."""

    def poll_segment(self, timeout_s: float) -> str | None:
        """Return the next captured segment, or None if none within timeout_s."""


# Printed each turn so the owner knows the mic is waiting for them to open a window.
PUSH_TO_TALK_PROMPT = "[Press Enter, speak your question, then wait...]"
# Printed when a window opened but no intelligible speech was captured, so the owner
# is never left staring at a silent, seemingly-hung prompt.
PUSH_TO_TALK_NO_SPEECH = "(didn't catch that - press Enter to try again)"


class PushToTalkSttSource:
    """Turn-based STT where the mic only feeds the assistant inside an Enter-window.

    One next_utterance():
      1. print the prompt and BLOCK on read_line() for the owner to press Enter
         (blocking input is fine -- the loop is turn-based). EOF / 'quit' / 'exit'
         ends the session (returns None);
      2. open the window: flush the mic so nothing captured while it was closed
         (idle ambient noise, or the assistant's own prior TTS) can leak in;
      3. collect the next complete utterance: pull segments with a short poll
         timeout; once the first segment arrives, keep gathering while more arrive
         within settle_s (an inter-segment silence ends the utterance), bounded by
         max_capture_s so a silent room never hangs the turn;
      4. close the window and return the joined utterance. If nothing intelligible
         was captured, print the no-speech notice and reloop (never hang).

    Pure orchestration over the injected MicSegments + read_line seams, so it is
    unit-testable headless with a fake mic and scripted input.
    """

    _QUIT_WORDS = {"quit", "exit"}

    def __init__(
        self,
        mic: MicSegments,
        read_line: Callable[[], str] | None = None,
        emit: Callable[[str], None] | None = None,
        max_capture_s: float = 15.0,
        settle_s: float = 1.2,
        poll_s: float = 0.3,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._mic = mic
        self._read_line = read_line or self._default_read
        self._emit = emit or (lambda _line: None)
        # Hard cap on one capture window: if nothing intelligible arrives within this
        # many seconds the turn ends with the no-speech notice rather than hanging.
        self._max_capture_s = max(1.0, max_capture_s)
        # After the first segment, wait this long for a continuation before deciding
        # the utterance is complete (models the natural end-of-utterance silence).
        self._settle_s = max(0.0, settle_s)
        # Per-poll wait handed to the mic; small so stop()/timeout stay responsive.
        self._poll_s = max(0.01, poll_s)
        self._clock = clock or time.monotonic
        self._stopped = False

    @staticmethod
    def _default_read() -> str:
        # A real EOF (Ctrl-Z / closed stdin) raises EOFError; treat it as end.
        try:
            return input()
        except EOFError:
            return ""

    def stop(self) -> None:
        """Signal the source to end (next_utterance returns None)."""
        self._stopped = True

    def next_utterance(self) -> str | None:
        """Prompt for Enter, capture one utterance, and return it (or None to end)."""
        while not self._stopped:
            self._emit(PUSH_TO_TALK_PROMPT)
            line = self._read_line()
            if line is None:
                return None  # EOF / closed stdin ends the session
            if line.strip().lower() in self._QUIT_WORDS:
                return None
            utterance = self._capture_one()
            if utterance:
                return utterance
            # Nothing intelligible this window: tell the owner and loop back to Enter.
            self._emit(PUSH_TO_TALK_NO_SPEECH)
        return None

    def _capture_one(self) -> str:
        """Collect one complete utterance from the just-opened window.

        Discards the closed-window backlog first (flush), then gathers segments until
        an inter-segment silence (settle_s) marks end-of-utterance or max_capture_s
        caps the window. Returns the joined text, or "" when nothing was captured.
        """
        # Open the window: drop everything captured while it was closed (this is the
        # structural fix -- ambient hallucinations and self-echo are dropped here).
        self._mic.flush()
        deadline = self._clock() + self._max_capture_s
        collected: list[str] = []
        while self._clock() < deadline and not self._stopped:
            remaining = deadline - self._clock()
            segment = self._mic.poll_segment(min(self._poll_s, remaining))
            if not segment:
                continue
            collected.append(segment)
            # First speech arrived: keep gathering while more comes within settle_s;
            # a settle-length gap ends the utterance. DL-1: the settle gathering is
            # ALSO bounded by the outer max_capture deadline so a continuous stream of
            # segments (each re-extending settle) can never run the window past
            # max_capture_s -- the cap is a hard ceiling on the whole utterance.
            settle_deadline = self._clock() + self._settle_s
            while (
                self._clock() < settle_deadline
                and self._clock() < deadline
                and not self._stopped
            ):
                gap = min(settle_deadline, deadline) - self._clock()
                more = self._mic.poll_segment(min(self._poll_s, gap))
                if more:
                    collected.append(more)
                    settle_deadline = self._clock() + self._settle_s
            break
        # Close the window: anything captured after this point (idle/TTS) is ignored
        # until the next Enter reopens a window.
        return " ".join(part.strip() for part in collected if part.strip()).strip()


class PrintTtsSink:
    """No-audio TTS sink: 'speaks' by printing, for --no-speak and tests.

    Records nothing; it is the honest no-op voice used when the owner asked for text
    only. The real audible sink is KokoroTtsSink below.
    """

    def __init__(self, emit: Callable[[str], None] | None = None) -> None:
        self._emit = emit or (lambda _line: None)

    def speak(self, text: str, voice: str) -> None:
        self._emit(f"[tts:{voice}] {text}")


def _default_tts_purge() -> None:
    """Stop any wav currently playing in this process (barge-in interrupt, M7.5).

    winsound.PlaySound(None, SND_PURGE) halts a synchronous clip started on the
    playback worker thread, so an interrupt stops the sentence being voiced instead
    of letting it play to completion (Review M-2). Imported lazily and guarded so a
    non-Windows or headless test host without winsound degrades to a no-op rather
    than raising. winsound is stdlib, so importing it here does not pull in the heavy
    TTS path this module otherwise avoids.
    """
    try:
        import winsound
    except Exception:  # noqa: BLE001 - no winsound (non-Windows/test host): nothing to purge
        return
    winsound.PlaySound(None, winsound.SND_PURGE)


def _default_tts_play(path: str) -> None:
    """Play a wav file synchronously via the Windows-native winsound (barge-safe).

    Mirrors _default_tts_purge: imported lazily and guarded so a non-Windows or
    headless test host degrades to a no-op rather than raising. SND_FILENAME plays the
    file to completion (blocking), which is why the sink calls it on its worker thread.
    """
    try:
        import winsound
    except Exception:  # noqa: BLE001 - no winsound (non-Windows/test host): nothing to play
        return
    winsound.PlaySound(path, winsound.SND_FILENAME)


# --------------------------------------------------------------------------- #
# Buffered whole-reply playback (defect D-M7-7).
#
# Speak-per-sentence played every sentence as a SEPARATE winsound clip; each
# PlaySound pays an audio-device spin-up (~1s observed) that clipped the first few
# words of EVERY sentence ("i don't hear the first 4 words of each sentence"). The
# voice system prompt already keeps replies to 1-3 sentences, so the sink now BUFFERS
# the whole reply and plays it as ONE concatenated wav: a single device spin-up
# (covered by one leading pad), nothing clipped between sentences.
# --------------------------------------------------------------------------- #

# Single leading pad prepended to the whole concatenated reply. Sized to cover the
# one audio-device spin-up the reply now pays (the D-M7-5 200ms per-sentence pad was
# far too small for the real ~1s spin-up); because it is paid ONCE per reply rather
# than per sentence, a generous pad costs little latency and guarantees the first word
# is audible.
# Short pad so the audio device can spin up without clipping the first phoneme.
# Was 1500 ms, which made every spoken reply feel a beat late.
REPLY_LEAD_SILENCE_MS = 120
# Natural pause inserted between concatenated sentences (matches tts._SENTENCE_GAP_MS).
SENTENCE_GAP_MS = 150


class KokoroTtsSink:
    """Audible TTS sink: buffers a reply's sentences and plays it as ONE wav (D-M7-7).

    Wraps an injected tts.KokoroClient. speak() BUFFERS each sentence of the reply in
    progress (non-blocking, no audio yet); finish() -- called by the loop once the
    reply is complete -- hands the buffered sentences to a single background playback
    thread, which synthesizes each, concatenates them into one wav (single leading pad
    + sentences joined by a short gap), and plays it once. That means a SINGLE
    audio-device spin-up per reply, so nothing at the start of any sentence is clipped.
    drain() (interrupt) drops the buffer AND the queued/playing reply and purges the
    single clip so audio does not bleed over the next question.

    The client is injected (not imported here) to keep this module free of the heavy
    TTS path; the launcher builds the KokoroClient against the running service. The
    player and purge callables are injectable for the same reason (tests pass fakes).
    """

    def __init__(
        self,
        client: Any,
        speed: float = 1.0,
        wav_dir: Any = None,
        purge: Callable[[], None] | None = None,
        speaking_gate: "HalfDuplexGate | None" = None,
        player: Callable[[str], None] | None = None,
        lead_silence_ms: int = REPLY_LEAD_SILENCE_MS,
        gap_ms: int = SENTENCE_GAP_MS,
        on_speaking: Callable[[bool], None] | None = None,
    ) -> None:
        import threading

        self._client = client
        self._speed = speed
        self._purge = purge or _default_tts_purge
        self._player = player or _default_tts_play
        # M10.3: optional additive display hook. Fired True the instant a reply's
        # single buffered clip begins playing and False the instant it ends, in the
        # SAME begin()/end() bracket the half-duplex gate already uses. The GUI wires
        # it to drive the Talk button's "speaking" state; None (the CLI default)
        # leaves the loop byte-for-byte unchanged. It never duplicates the D-M7-7
        # buffered-playback logic - it only observes it.
        self._on_speaking = on_speaking
        self._lead_silence_ms = lead_silence_ms
        self._gap_ms = gap_ms
        # Half-duplex gate (D-M7-3): SET before the single reply clip plays, CLEARED
        # tail-seconds after it finishes, so the mic source can drop the assistant's
        # own audio instead of transcribing it as a user turn. None disables gating.
        # begin()/end() bracket the WHOLE single playback, keeping D-M7-3b/c correct.
        self._speaking_gate = speaking_gate
        # When set, each reply wav is written durably here (in addition to being
        # played) so a headless smoke can prove real audio bytes were synthesized.
        self._wav_dir = wav_dir
        self._clip_index = 0
        self.last_wav: str | None = None
        # Sentences of the reply currently being streamed (filled by speak(), moved to
        # the playback queue by finish()).
        self._reply_buffer: list[tuple[str, str]] = []
        # Queued playback jobs (each a list of sentences). speak() enqueues a
        # one-sentence job immediately so the first sentence starts while the
        # LLM is still generating the rest.
        self._queue: list[list[tuple[str, str]]] = []
        self._lock = Event()  # signaled when there is work; see _run
        self._stop = Event()
        self._playing = False  # True while a reply is being synthesized/played
        self._pending_lock = threading.Lock()
        self._worker: Thread | None = None
        # Lead pad only on the first clip of a speaking burst, not every sentence.
        self._apply_lead = True
        self._burst_active = False

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._stop.clear()
            self._worker = Thread(target=self._run, name="locitize-tts", daemon=True)
            self._worker.start()

    def _notify_speaking(self, active: bool) -> None:
        """Best-effort fire of the optional on_speaking display hook (M10.3)."""
        if self._on_speaking is None:
            return
        try:
            self._on_speaking(active)
        except Exception:  # noqa: BLE001 - a display hook fault must never break TTS
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._lock.wait(timeout=0.2)
            self._lock.clear()
            while True:
                with self._pending_lock:
                    if not self._queue:
                        break
                    job = self._queue.pop(0)
                    self._playing = True
                    start_burst = not self._burst_active
                    self._burst_active = True
                if self._stop.is_set():
                    self._playing = False
                    self._burst_active = False
                    break
                # Gate the mic before the first sample of a burst so no leading
                # audio leaks back as a user turn. Keep the gate and the speaking
                # display hook up across consecutive sentences so the Talk button
                # does not flicker idle between them.
                if start_burst:
                    if self._speaking_gate is not None:
                        self._speaking_gate.begin()
                    self._notify_speaking(True)
                try:
                    self._play_reply(job)
                except Exception:  # noqa: BLE001 - a failed reply must not kill the loop
                    pass
                finally:
                    with self._pending_lock:
                        more = bool(self._queue)
                        self._playing = False
                        if not more:
                            self._burst_active = False
                            self._apply_lead = True
                    if not more:
                        if self._speaking_gate is not None:
                            self._speaking_gate.end()
                        self._notify_speaking(False)

    def _play_reply(self, job: list[tuple[str, str]]) -> None:
        """Synthesize each buffered sentence, concatenate into ONE wav, play it once.

        Uses the client's synthesize() (raw bytes) rather than speak() so the reply is
        assembled into a single clip here. A sentence whose synthesis fails is skipped
        so the rest of the reply is not lost. Nothing is played (and the gate window is
        empty) when no audio results, so an all-failed reply degrades honestly.
        """
        from tts import concatenate_wavs

        chunks: list[bytes] = []
        for text, voice in job:
            if self._stop.is_set():
                return
            try:
                chunks.append(self._client.synthesize(text, voice, self._speed))
            except Exception:  # noqa: BLE001 - one bad sentence must not lose the reply
                continue
        if self._stop.is_set():
            return
        lead = self._lead_silence_ms if self._apply_lead else 0
        self._apply_lead = False
        combined = concatenate_wavs(chunks, lead, self._gap_ms)
        if not combined:
            return
        out_path, cleanup = self._reply_wav_path()
        try:
            with open(out_path, "wb") as handle:
                handle.write(combined)
            self.last_wav = out_path
            self._player(out_path)
        finally:
            if cleanup:
                # No durable wav_dir: the temp reply wav is playback-only, so remove it
                # afterwards and leave no synthesized-speech residue (SEC-M6-1 hygiene).
                from pathlib import Path

                try:
                    Path(out_path).unlink()
                except OSError:
                    pass

    def _reply_wav_path(self) -> tuple[str, bool]:
        """Return (path, cleanup) for the next concatenated reply wav.

        With a wav_dir the wav is durable and KEPT (cleanup=False) so a headless smoke
        can inspect the real audio bytes; without one a temp wav is written and DELETED
        after playback (cleanup=True).
        """
        from pathlib import Path

        if self._wav_dir is not None:
            self._clip_index += 1
            return str(Path(self._wav_dir) / f"assistant_reply_{self._clip_index}.wav"), False
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            return tmp.name, True

    def speak(self, text: str, voice: str) -> None:
        """Queue one sentence for playback immediately (non-blocking).

        The first sentence of a reply starts synthesizing as soon as it is
        ready, so the owner hears audio while the LLM is still generating.
        Later sentences enqueue behind it. finish() still flushes any tail
        left in the buffer (wait-for-complete mode).
        """
        with self._pending_lock:
            self._queue.append([(text, voice)])
        self._ensure_worker()
        self._lock.set()

    def finish(self) -> None:
        """Flush any remaining buffered sentences onto the playback queue.

        A no-op when speak() already queued every sentence (the usual path).
        Safe to call every turn.
        """
        with self._pending_lock:
            if not self._reply_buffer:
                return
            self._queue.append(self._reply_buffer)
            self._reply_buffer = []
        self._ensure_worker()
        self._lock.set()

    def drain(self) -> None:
        """Drop pending playback and stop the in-flight reply (interrupt path, M7.5).

        Clears the not-yet-played buffer AND queue, then purges the single reply clip
        already playing so a barge-in interrupt stops audio promptly (Review M-2)
        instead of letting the whole concatenated reply finish.
        """
        with self._pending_lock:
            self._reply_buffer = []
            self._queue.clear()
            self._burst_active = False
            self._apply_lead = True
        # A barge-in interrupt must reopen the mic at once (D-M7-3), so clear the
        # half-duplex gate here rather than waiting for the tail to expire.
        if self._speaking_gate is not None:
            self._speaking_gate.clear()
        # Best-effort; a purge failure must not break interrupt handling.
        try:
            self._purge()
        except Exception:  # noqa: BLE001 - purge is advisory, never fatal to the interrupt
            pass

    def close(self, finish_timeout: float = 60.0) -> None:
        """Let the queued reply finish playing, then stop the worker (session end).

        On a normal session end LOCITIZE should finish speaking its last reply, so this
        waits (bounded) for the queue to drain and the current reply to finish before
        stopping. After an interrupt, drain() has already cleared the queue, so this
        returns promptly.
        """
        import time

        deadline = time.monotonic() + finish_timeout
        while time.monotonic() < deadline:
            with self._pending_lock:
                idle = not self._queue and not self._playing
            if idle:
                break
            time.sleep(0.1)
        self._stop.set()
        self._lock.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# The loop itself (M7.1/M7.4/M7.5/M7.6/M7.10).
# --------------------------------------------------------------------------- #


class AssistantLoop:
    """Turn-based assistant orchestration over the three injected seams.

    Construction is pure dependency injection (Architecture M7.8): no globals, no
    process spawning. One turn:
      1. take an utterance from the SttSource (None ends the session),
      2. append it to state and to memory,
      3. trim history to the ctx budget,
      4. stream the reply from the LlmClient, segmenting into sentences and speaking
         each through the TtsSink while generation continues (speak-per-sentence),
      5. print the full reply, append it to state and memory, and record real
         per-stage timings (M7.6),
      6. honor an interrupt (Event) at any point, draining pending audio (M7.5).
    """

    def __init__(
        self,
        stt: SttSource,
        llm: LlmClient,
        tts: TtsSink | None,
        state: ConversationState,
        *,
        voice: str = "",
        speaking: bool = False,
        speak_per_sentence: bool = True,
        context_size: int = 4096,
        response_reserve_tokens: int = 512,
        max_history_turns: int = 20,
        measure_stt: bool = False,
        memory: Any = None,
        recall_limit: int = 5,
        interrupt: Event | None = None,
        emit: Callable[[str], None] | None = None,
        on_timings: Callable[[TurnTimings], None] | None = None,
        text_filter: Callable[[str], str] | None = None,
        half_duplex_gate: "HalfDuplexGate | None" = None,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._state = state
        self._voice = voice
        self._speaking = speaking and tts is not None
        self._speak_per_sentence = speak_per_sentence
        self._context_size = context_size
        self._reserve = response_reserve_tokens
        self._max_history_turns = max_history_turns
        self._measure_stt = measure_stt
        self._memory = memory
        self._recall_limit = recall_limit
        self._interrupt = interrupt or Event()
        self._emit = emit or (lambda _line: None)
        self._on_timings = on_timings
        # Applied to each sentence before it is spoken and to the reply before it is
        # displayed (D-M7-2: the launcher passes strip_markdown so speech/console
        # output is plain text). Defaults to identity so unit tests and the pure
        # loop are unchanged unless a filter is explicitly injected.
        self._text_filter = text_filter or (lambda text: text)
        # The loop owns the half-duplex gate (D-M7-3): the TTS sink sets/clears it as
        # it plays, the mic source checks it, and an interrupt here clears it so a
        # barge-in reopens the mic immediately. None in text mode (no self-hearing).
        self._half_duplex_gate = half_duplex_gate

    @property
    def half_duplex_gate(self) -> "HalfDuplexGate | None":
        return self._half_duplex_gate

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def interrupt(self) -> Event:
        return self._interrupt

    def _count_tokens(self, text: str) -> int:
        """Token count for `text`: the server's count, or a labeled chars/4 fallback.

        Reuses the M7.3 discipline -- prefer the model-correct /tokenize count; when
        the server cannot be reached, fall back to a rough chars/4 estimate (honest
        and clearly approximate) rather than guessing a precise-looking number.
        """
        exact = self._llm.count_tokens(text)
        if exact is not None:
            return exact
        # chars/4 is the standard rough English token heuristic; +1 so a short
        # non-empty string never counts as zero tokens.
        return (len(text) // 4) + 1

    def _budget(self) -> int:
        """The prompt token budget: ctx size minus the reply headroom (M7.3)."""
        return max(1, self._context_size - self._reserve)

    def _trimmed_messages(self) -> list[ChatMessage]:
        """History trimmed by turn cap then token budget for the next request."""
        msgs = self._state.messages
        # Secondary hard cap on retained turns before the token trim (Data Model
        # 9.1: max_history_turns), never dropping the leading system message(s).
        system = [m for m in msgs if m.role == "system"]
        rest = [m for m in msgs if m.role != "system"]
        if len(rest) > self._max_history_turns:
            rest = rest[-self._max_history_turns:]
        capped = system + rest
        return trim_history(capped, self._budget(), self._count_tokens)

    def run_turn(self, utterance: str, stt_ms: float = 0.0) -> tuple[str, TurnTimings]:
        """Run one full turn for a known utterance; return (reply, timings).

        Separated from run() so tests drive a single turn deterministically. Raises
        LlmError only if the LLM itself fails (the caller surfaces the remedy); a
        clean interrupt returns whatever partial reply was produced.
        """
        timings = TurnTimings(stt_ms=stt_ms)

        # M8.3: an explicit /recall command injects matched past context on demand
        # (never automatic). It searches the local memory store and folds the hits
        # into conversation state as a system preamble, so the NEXT turns can use
        # them; the command itself does not call the LLM. Recall is transparent and
        # cheap by design (substring scan, no embeddings).
        recall = self._maybe_recall(utterance)
        if recall is not None:
            self._emit(f"LOCITIZE: {recall}")
            return recall, timings

        self._state.append("user", utterance)
        self._append_memory("user", utterance)

        messages = self._trimmed_messages()
        turn_start = time.monotonic()
        first_token_at: float | None = None
        tts_first_at: float | None = None
        segmenter = SentenceSegmenter()
        reply_parts: list[str] = []

        for delta in self._llm.chat_stream(messages, interrupt=self._interrupt):
            if first_token_at is None:
                first_token_at = time.monotonic()
            reply_parts.append(delta)
            if self._speaking and self._speak_per_sentence:
                for sentence in segmenter.feed(delta):
                    if tts_first_at is None:
                        tts_first_at = time.monotonic()
                    # Strip markdown so Kokoro never voices "asterisk asterisk" etc.
                    self._speak(sentence)

        llm_done = time.monotonic()
        reply = "".join(reply_parts)

        interrupted = self._interrupt.is_set()
        if not interrupted and self._speaking:
            if self._speak_per_sentence:
                tail = segmenter.flush()
                if tail:
                    if tts_first_at is None:
                        tts_first_at = time.monotonic()
                    self._speak(tail)
            elif reply:
                # wait-for-complete mode: speak the whole reply once.
                tts_first_at = time.monotonic()
                self._speak(reply)
            # D-M7-7: the reply is complete -- flush the buffered sentences so the sink
            # plays the WHOLE reply as one wav (single device spin-up, no clipped starts).
            # Sinks without a finish() (PrintTtsSink / test fakes) are unaffected.
            self._finish_speaking()
        if interrupted:
            self._drain_tts()

        timings.llm_first_token_ms = self._ms(turn_start, first_token_at)
        timings.llm_total_ms = self._ms(turn_start, llm_done)
        timings.tts_first_audio_ms = self._ms(turn_start, tts_first_at)

        if reply:
            self._state.append("assistant", reply)
            self._append_memory("assistant", reply)
            # Display the plain-text form (D-M7-2) so --assistant shows clean prose,
            # not raw markdown. The unfiltered reply is what is kept in state/memory
            # so history and recall stay faithful to what the model actually produced.
            self._emit(f"LOCITIZE: {self._text_filter(reply)}")
        if self._on_timings is not None:
            self._on_timings(timings)
        return reply, timings

    def run(self) -> int:
        """Run turns until the source ends (None) or an interrupt closes the loop.

        Returns 0 on a clean end. Each turn measures STT wall-clock only when
        measure_stt is set (the mic path); in text mode stt_ms stays 0 (M7.6/AC14).
        """
        while True:
            if self._interrupt.is_set():
                # A prior turn's interrupt clears for the NEXT turn so the loop
                # continues normally (M7.5: interrupt returns control to next turn).
                self._interrupt.clear()
            stt_start = time.monotonic()
            utterance = self._stt.next_utterance()
            if utterance is None:
                break
            stt_ms = self._ms(stt_start, time.monotonic()) if self._measure_stt else 0.0
            try:
                self.run_turn(utterance, stt_ms=stt_ms)
            except LlmError as exc:
                # Honest surfaced failure -- never a fabricated reply (M7.2).
                remedy = f" ({exc.remedy})" if exc.remedy else ""
                self._emit(f"LOCITIZE: [llm unavailable: {exc}{remedy}]")
        return 0

    def _speak(self, text: str) -> None:
        """Voice one piece of reply text through the sink after markdown stripping.

        Centralizes the D-M7-2 stripping so every speak path (per-sentence, flushed
        tail, whole-reply) goes through the same plain-text filter. A sentence that
        is empty once its markdown markers are removed (e.g. a lone "**") is skipped
        so the sink is never handed a blank clip.
        """
        spoken = self._text_filter(text)
        if spoken:
            self._tts.speak(spoken, self._voice)  # type: ignore[union-attr]

    def _finish_speaking(self) -> None:
        """Tell the sink the reply is complete so any remaining buffer is flushed.

        KokoroTtsSink.speak() already queues each sentence; finish() is a no-op
        then. The printing sink and test fakes have no finish(); their speak()
        already acted.
        """
        finish = getattr(self._tts, "finish", None)
        if callable(finish):
            finish()

    def _drain_tts(self) -> None:
        """Drop pending audio on interrupt if the sink supports it (M7.5).

        Also clears the half-duplex gate (D-M7-3) so a barge-in interrupt reopens the
        mic at once, even if the injected sink is not the Kokoro sink that clears it
        itself (the two clears are idempotent and safe to both run).
        """
        drain = getattr(self._tts, "drain", None)
        if callable(drain):
            drain()
        if self._half_duplex_gate is not None:
            self._half_duplex_gate.clear()

    def _maybe_recall(self, utterance: str) -> str | None:
        """Handle a `/recall <query>` command; return the recall reply or None.

        Returns None for a normal utterance (so the caller runs a normal turn). For
        `/recall <query>` it searches the injected memory store for the query and,
        for a bare `/recall`, loads the most recent conversation tail. Matched
        snippets are both returned to the owner AND appended to conversation state as
        a single system preamble so subsequent turns can use them (M8.3: explicit,
        on-demand, bounded). With no memory store or no match it says so honestly --
        it never fabricates recalled content.
        """
        text = utterance.strip()
        if not (text == "/recall" or text.startswith("/recall ")):
            return None
        if self._memory is None:
            return "memory recall is disabled (no store configured)."
        query = text[len("/recall"):].strip()
        try:
            if query:
                hits = self._memory.search(query, self._recall_limit)
            else:
                # Bare /recall = "what were we just talking about": recent tail.
                hits = self._memory.recall(self._recall_limit)
        except Exception:  # noqa: BLE001 - recall must never break the session
            return "memory recall failed; continuing without it."
        if not hits:
            what = f"'{query}'" if query else "recent context"
            return f"no stored conversation matched {what}."
        # Format the matched entries compactly (role: text) for both display and the
        # injected preamble, keeping it bounded (recall_limit already caps the count).
        lines = [f"{h.role}: {h.text}" for h in hits]
        preamble = "Recalled earlier context:\n" + "\n".join(lines)
        # Inject as a system message so the model treats it as background, not as the
        # owner's current utterance. trim_history keeps it within the ctx budget.
        self._state.append("system", preamble)
        return preamble

    def _append_memory(self, role: str, text: str) -> None:
        """Persist one transcript entry if a memory store was injected (M8.3 hook)."""
        if self._memory is None:
            return
        try:
            self._memory.append(self._state.session_id, role, text)
        except Exception:  # noqa: BLE001 - memory persistence must not break a turn
            pass

    @staticmethod
    def _ms(start: float, end: float | None) -> float:
        """Milliseconds between two monotonic marks; 0.0 when end is unset."""
        if end is None:
            return 0.0
        return (end - start) * 1000.0
