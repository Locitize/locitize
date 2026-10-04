"""Whisper speech-to-text service specs and the real transcription client.

This is the whisper analog of models.py: it turns the platform Settings into
declarative ServiceSpecs for the two whisper binaries the platform supervises,
and it holds the real HTTP client that sends audio to a running whisper-server
and returns the transcript (Milestone 3, Architecture sections 4 and 10).

- build_whisper_server_spec  -> whisper-server.exe (HTTP /inference transcription)
- build_whisper_stream_spec  -> whisper-stream.exe (SDL2 live mic capture)
- transcribe_file            -> POST an audio file to a running server, get text

It never launches a process itself (services.py owns that) and never fabricates
a transcript (the client posts real audio to the real server and returns exactly
what the server reports). All machine-specific paths come from Settings
(settings.yaml / LOCITIZE_* env), so nothing here hardcodes a filesystem layout.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from config import Settings
from services import ServiceSpec, resolve_service_cwd

# Trailing punctuation stripped before comparing two transcript segments. whisper
# emits the same utterance from two overlapping windows with slightly different
# terminal punctuation (". " vs "?" vs no mark), so these must not defeat the
# dup check. Only trailing marks are stripped -- interior punctuation is content.
# The last two entries are the Unicode en-dash (U+2013) and em-dash (U+2014),
# spelled as escapes so this source file stays ASCII-clean while still stripping
# the fancy dashes whisper may append to an utterance.
_TRAILING_PUNCT = ".,!?;:- \t\u2013\u2014"

# Collapses any run of whitespace to a single space so "can  you" and "can you"
# compare equal. Precompiled once; used only for normalization, never on output.
_WHITESPACE_RUN = re.compile(r"\s+")

# whisper wraps its own non-speech annotations in brackets or parentheses, e.g.
# "[BLANK_AUDIO]", "[ Silence ]", "(music)". This matches such a span so it can be
# removed before deciding whether a segment carries any real spoken content. Used
# only by is_non_speech; never mutates surfaced text.
_BRACKETED_ANNOTATION = re.compile(r"[\[(][^\])]*[\])]")


# --------------------------------------------------------------------------- #
# Startup-noise gating for the mic/listen path (defect D-M7-1).
#
# whisper-stream writes its OWN engine boot diagnostics (ggml/CUDA init, the SDL
# capture-device probe, whisper model load, and per-utterance "### Transcription"
# block markers) to the same stdout stream it later prints transcriptions on.
# Before this gate, _MicSttSource fed every such line to the assistant LLM as a
# fabricated user turn (the owner's AC17 session "conversed" with the engine's
# own CUDA-init output). The fix has two layers:
#   1. a one-time capture gate on whisper-stream's "[Start speaking]" banner --
#      every line before that banner is boot noise and is ignored wholesale; and
#   2. a conservative diagnostic-line filter (is_diagnostic_line) as defense in
#      depth, so any engine line that slips through the banner (or a build that
#      omits the banner) is still rejected.
# Real transcribed speech must always pass -- the filter is deliberately narrow.
# --------------------------------------------------------------------------- #

# whisper-stream prints this banner exactly once, the moment live mic capture
# actually begins. Everything the tool wrote before it is startup/engine output,
# never speech, so the gate drops all of it (and the banner line itself).
CAPTURE_BANNER = "[Start speaking]"

# Anchored signatures of whisper-stream's known non-speech line classes, matched
# against the stripped line. These are the exact families seen in a real capture
# log: ggml_* / whisper_* engine lines, the SDL "init:" capture probe, the
# "SDL_main:" runtime banner, "main:" CLI echo, a "Device N:" GPU line, and the
# "###" per-utterance block markers whisper-stream frames each transcription with.
_DIAGNOSTIC_SIGNATURES = re.compile(
    r"^(?:ggml_|whisper_|load_backend\b|system_info\b|main\s*:|SDL_main\s*:"
    r"|init\s*:|Device\s+\d+\s*:|#{2,})"
)

# Keyword signatures that can appear mid-line in a GPU/VRAM diagnostic. Kept
# separate from the anchored set and intentionally specific (a compute-capability
# or "VRAM:" phrase never occurs in natural spoken English) so real speech is safe.
_DIAGNOSTIC_KEYWORDS = re.compile(r"compute\s+capability|VRAM\s*:", re.IGNORECASE)

# A generic "identifier: value" diagnostic: a single leading snake_case/dotted
# token immediately followed by a colon (as in "whisper_init_state: kv self size").
# The token must be a bare identifier with NO internal spaces, which is what keeps
# genuine speech that merely contains a colon -- e.g. "remind me at 5: buy milk" --
# from being misclassified (its pre-colon text "remind me at 5" has spaces).
_KEY_VALUE_DIAGNOSTIC = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*\s*:(?:\s|$)")

# whisper-stream tags each real transcription line with a "[hh:mm:ss.mmm --> ...]"
# timestamp span; the spoken words follow the closing bracket. This extracts just
# the words so the assistant hears "Thank you." rather than the timestamp noise.
_TIMESTAMP_PREFIX = re.compile(
    r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*(.*)$"
)

# Same span, but capturing the start and end hh:mm:ss.mmm fields. NOTE (defect
# D-M7-3c): whisper-stream's PER-LINE "[hh:mm:ss.mmm --> ...]" timestamp RESETS to
# 00:00:00 at the start of every transcription block, so it is NOT a usable stream
# clock. parse_capture_interval is retained (still unit-tested) but the half-duplex
# gate no longer anchors on it; the block HEADER cumulative ms is the real clock.
_TIMESTAMP_INTERVAL = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]"
)

# The per-utterance block HEADER whisper-stream prints before each transcription,
# e.g. "### Transcription 36 START | t0 = 90443 ms | t1 = 100443 ms". Unlike the
# per-line timestamp, t0/t1 here are CUMULATIVE milliseconds since capture start and
# never reset -- so they are the correct clock for the capture-time half-duplex gate
# (defect D-M7-3c). Confirmed against the real owner log
# (logs/assistant_whisper_stream.log), e.g. "### Transcription 0 START | t0 = 0 ms |
# t1 = 5536 ms" and later blocks with t0 = 3685/5719/... ms while the per-line span
# still reads "[00:00:00.000 --> ...]".
_BLOCK_HEADER = re.compile(
    r"^#{2,}\s*Transcription\s+\d+\s+START\s*\|\s*t0\s*=\s*(\d+)\s*ms"
    r"\s*\|\s*t1\s*=\s*(\d+)\s*ms"
)


def parse_capture_interval(line: str) -> tuple[float, float] | None:
    """Parse whisper-stream's per-line "[start --> end]" span into relative seconds.

    Returns (start_s, end_s) as seconds, or None when the line has no timestamp
    prefix. RETAINED for completeness/tests, but superseded for gating by
    parse_block_header (defect D-M7-3c): the per-line span resets each block and so
    cannot anchor an absolute capture time. Use the block header instead.
    """
    match = _TIMESTAMP_INTERVAL.match(line.strip())
    if not match:
        return None
    h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in match.groups())
    start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
    end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
    return (start, end)


def parse_block_header(line: str) -> tuple[float, float] | None:
    """Parse a "### Transcription N START | t0 = <ms> | t1 = <ms>" header (D-M7-3c).

    Returns (t0_s, t1_s) -- the block's CUMULATIVE capture interval in seconds since
    stream start -- or None when the line is not such a header. This is the clock the
    half-duplex gate anchors on, because it (unlike the per-line "[00:00:00 -->]"
    span) does not reset every block. The caller adds the wall-clock anchor recorded
    at banner time to get an absolute interval.
    """
    match = _BLOCK_HEADER.match(line.strip())
    if not match:
        return None
    t0_ms, t1_ms = int(match.group(1)), int(match.group(2))
    return (t0_ms / 1000.0, t1_ms / 1000.0)


def is_diagnostic_line(line: str) -> bool:
    """Return True if `line` is whisper-stream engine/boot output, not speech.

    Defense-in-depth companion to the "[Start speaking]" capture gate (defect
    D-M7-1). Conservative by design: it matches only the known non-speech line
    families (see the signature regexes above), so a CUDA-init line is rejected
    while a natural spoken sentence -- even one containing a colon -- passes.
    """
    stripped = line.strip()
    if not stripped:
        # A blank line carries nothing; treat as non-speech so it is never surfaced.
        return True
    if _DIAGNOSTIC_SIGNATURES.match(stripped):
        return True
    if _KEY_VALUE_DIAGNOSTIC.match(stripped):
        return True
    if _DIAGNOSTIC_KEYWORDS.search(stripped):
        return True
    return False


def is_transcript_line(line: str, capture_started: bool) -> bool:
    """Return True if `line` should be treated as real transcribed speech.

    The single reusable predicate the mic and listen paths gate on (defect
    D-M7-1). A line is speech only when capture has actually begun (the
    "[Start speaking]" banner was seen) AND the line is not a known engine
    diagnostic. Kept pure and side-effect-free so it is directly unit-testable.
    """
    if not capture_started:
        return False
    if not line.strip():
        return False
    return not is_diagnostic_line(line)


def extract_spoken_text(line: str) -> str:
    """Strip whisper-stream's "[timestamp -->]" prefix, returning just the words.

    A real transcription line is "[00:00:00.000 --> 00:00:05.920]   Thank you.";
    the assistant should hear "Thank you.", not the timestamp span. A plain line
    with no timestamp prefix is returned trimmed and unchanged.
    """
    stripped = line.strip()
    match = _TIMESTAMP_PREFIX.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


class StartupNoiseGate:
    """Stateful one-time capture gate over a whisper-stream log stream (D-M7-1).

    Tracks whether the "[Start speaking]" banner has been seen yet. Until it has,
    every line is boot/engine noise and is dropped; once it has, each line is run
    through is_transcript_line (diagnostic defense-in-depth) and, if it survives,
    reduced to its spoken words via extract_spoken_text. The gate is stateful
    because the banner appears exactly once per capture session, so a caller polling
    the log across many reads must remember it has been crossed.

    require_banner defaults True (the correct production behavior). It exists so a
    caller can start already-open when it is certain capture began -- not used in
    the shipped paths, but it keeps the class honest and testable.

    Capture-time anchoring (D-M7-3b, corrected in D-M7-3c): the gate records the
    WALL-CLOCK time at the moment it observes the "[Start speaking]" banner (capture
    begins essentially then) and uses it as the anchor: absolute_capture = anchor +
    block-header cumulative time. The block header "### Transcription N START | t0 =
    <ms> | t1 = <ms>" is the clock (its t0/t1 never reset), NOT the per-line
    "[00:00:00 -->]" span which resets every block. This is the
    best available reference -- its error is bounded by the log poll latency between
    whisper writing the banner and the caller reading it (~one poll interval), which
    the half-duplex overlap check absorbs with a guard margin. last_capture_interval
    exposes the most recently surfaced segment's absolute (start, end) wall-clock
    interval so the mic half-duplex gate can test it for overlap; it is None until a
    timestamped line is surfaced with an anchor in place. The clock is injectable so
    tests share the same fake clock as the HalfDuplexGate.
    """

    def __init__(
        self,
        require_banner: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._capture_started = not require_banner
        # monotonic so it is directly comparable with HalfDuplexGate's own default
        # clock; both must read the same clock for the interval math to line up.
        self._clock = clock or time.monotonic
        self._anchor: float | None = None
        # D-M7-3c: the most recently seen block header's (t0_s, t1_s) CUMULATIVE
        # capture interval. A header precedes the transcript line(s) of its block, so
        # this is stamped when the header is seen and applied to the following line.
        # None until the first parseable header, in which case a surfaced segment
        # gets no interval and the mic path falls back to the poll-time flag.
        self._block_interval: tuple[float, float] | None = None
        # Absolute (start, end) wall-clock capture interval of the last surfaced
        # segment, or None when it carried no timestamp / had no anchor yet. Set as a
        # side effect of surfaced_text and read immediately after by the mic path
        # within the same single-threaded poll, so there is no cross-thread race.
        self.last_capture_interval: tuple[float, float] | None = None

    @property
    def capture_started(self) -> bool:
        """True once the "[Start speaking]" banner has been observed."""
        return self._capture_started

    @property
    def anchor(self) -> float | None:
        """Wall-clock time recorded when capture began, or None before the banner."""
        return self._anchor

    def surfaced_text(self, line: str) -> str | None:
        """Map one raw log line to the speech to surface, or None to drop it.

        Before the banner: always None (and the banner line itself is consumed,
        never surfaced) -- the anchor wall-clock is stamped here. After the banner:
        None for a diagnostic/blank line, else the line's spoken words. A returned
        string may still be a non-speech placeholder like "." -- the downstream
        TranscriptDeduplicator.is_non_speech guard drops those, so this gate stays
        focused on startup/engine noise.

        Side effect: last_capture_interval is set to the surfaced segment's absolute
        (anchor + relative) capture window, or None when it has no timestamp.
        """
        self.last_capture_interval = None
        if not self._capture_started:
            if CAPTURE_BANNER in line:
                self._capture_started = True
                # Anchor the relative timestamps to now: capture (t=0) begins the
                # instant this banner is written, so this is the honest reference.
                self._anchor = self._clock()
            return None
        # D-M7-3c: a block header carries the real CUMULATIVE capture time. Record it
        # and consume the header (it is never surfaced as speech); the following
        # transcript line(s) of this block use it as their capture interval.
        header = parse_block_header(line)
        if header is not None:
            self._block_interval = header
            return None
        if not is_transcript_line(line, self._capture_started):
            return None
        # D-M7-3c: anchor on the BLOCK HEADER cumulative interval, NOT the per-line
        # "[00:00:00 -->]" span (which resets each block and, before this fix, mapped
        # every turn to anchor+0..10s -> froze the mic after turn 1). When no header
        # has been seen for this segment, leave the interval None so the mic path
        # falls back to the poll-time flag rather than dropping speech forever.
        if self._block_interval is not None and self._anchor is not None:
            self.last_capture_interval = (
                self._anchor + self._block_interval[0],
                self._anchor + self._block_interval[1],
            )
        return extract_spoken_text(line)


def is_non_speech(segment: str) -> bool:
    """Return True if `segment` carries no spoken words and must not be surfaced.

    Defense-in-depth for defect D-M3-2 (silence-hallucination). whisper-stream's
    VAD mode (--step 0, wired in build_whisper_stream_spec) is the primary silence
    gate, but on a marginal window the model can still emit a bare placeholder --
    a lone "." for a silent step, a blank line, or one of whisper's own bracketed
    non-speech annotations ("[BLANK_AUDIO]", "[ Silence ]", "(music)"). Those are
    artifacts, never words the owner spoke, so LOCITIZE drops them from the surfaced
    transcript.

    This is deliberately NOT a phrase blocklist: a genuinely spoken "Thank you."
    contains letters outside any annotation and returns False (survives). The test
    is purely structural -- strip whisper's bracketed/parenthesized annotations,
    then a segment is non-speech only if nothing alphanumeric remains.
    """
    without_annotations = _BRACKETED_ANNOTATION.sub(" ", segment)
    # Any surviving letter or digit means real content was spoken.
    return not any(ch.isalnum() for ch in without_annotations)


class TranscriptDeduplicator:
    """Suppress consecutive repeated transcript segments in LOCITIZE's listen output.

    Why this exists: whisper-stream transcribes overlapping audio windows (step
    3.0s / len 10.0s), so an utterance that spans a window boundary is emitted in
    both windows and LOCITIZE would surface it twice (defect D-M3-1, AC9). This
    filter sits on LOCITIZE's *presentation* of the stream: the raw child log is left
    untouched for debuggability, and only what LOCITIZE echoes/records is deduped.

    A segment is suppressed when its normalized form matches any of the last
    `horizon` emitted segments. Normalization is trim + whitespace-collapse +
    case-fold + trailing-punctuation-strip, so cosmetic variants of one utterance
    collapse together. The horizon is deliberately small: a genuinely repeated but
    distinct utterance separated by other speech falls outside the window and is
    correctly emitted again, rather than being silently swallowed.
    """

    def __init__(self, horizon: int = 3) -> None:
        # horizon = how many recently emitted segments a new one is compared
        # against. 3 covers the raw stream's overlap (a dup usually reappears
        # within one or two windows) while staying short enough that a real
        # re-utterance after other speech is not mistaken for an overlap echo.
        self._horizon = max(1, horizon)
        # Ring of the most recent emitted *normalized* segments, oldest first.
        self._recent: list[str] = []

    @staticmethod
    def normalize(segment: str) -> str:
        """Reduce a segment to its comparison key.

        Trim, collapse internal whitespace runs to one space, case-fold, and strip
        trailing punctuation/whitespace. Returns "" for a blank/whitespace-only
        segment so callers can treat it as "nothing meaningful to emit". Case-fold
        (not lower) is used so non-ASCII text folds correctly too.
        """
        collapsed = _WHITESPACE_RUN.sub(" ", segment).strip()
        folded = collapsed.casefold()
        return folded.rstrip(_TRAILING_PUNCT)

    def accept(self, segment: str) -> bool:
        """Return True if `segment` should be emitted, False if it is a duplicate.

        A non-speech marker (blank line, lone punctuation, or a whisper bracketed
        annotation -- see is_non_speech, defect D-M3-2) is never emitted and does
        not advance the horizon, so a real dup straddling such noise is still
        caught. A non-blank segment matching any of the last `horizon` emitted
        segments is suppressed; otherwise it is recorded as emitted and accepted.
        """
        # Silence/hallucination guard runs before dedup: drop artifacts outright so
        # they neither reach the owner nor pollute the dedup horizon.
        if is_non_speech(segment):
            return False
        key = self.normalize(segment)
        if not key:
            return False
        if key in self._recent:
            return False
        self._recent.append(key)
        # Bound the memory to the horizon so an old segment ages out and a later
        # genuine re-utterance is not suppressed.
        if len(self._recent) > self._horizon:
            self._recent.pop(0)
        return True

    def filter_segments(self, segments: Any) -> list[str]:
        """Convenience: return only the segments from an iterable that survive.

        Preserves order and original (un-normalized) text of the survivors; used by
        the listen path and directly exercised by the dedup unit tests.
        """
        return [seg for seg in segments if self.accept(seg)]

# Service names are stable identifiers used by ServiceManager registration and
# the status panel. Kept as module constants so the launcher, controllers, and
# tests all agree on the exact string.
WHISPER_SERVER_NAME = "whisper_server"
WHISPER_STREAM_NAME = "whisper_stream"

# whisper-server exposes no dedicated /health endpoint (confirmed against the
# real binary's --help), so readiness uses the platform's TCP port-connect
# fallback (services.py _default_readiness with health_path=None). Loading the
# large-v3-turbo model on GPU takes several seconds, so the readiness timeout is
# generous; the value is the platform default from settings unless overridden.
_STREAM_READY_TIMEOUT_S = 15.0


def build_whisper_server_spec(settings: Settings, log_path: str | None = None) -> ServiceSpec:
    """Build the whisper-server.exe ServiceSpec from settings.

    The command is fully data-driven: binary path (paths.whisper), model weights
    (paths.whisper_model), the reserved loopback port (ports.whisper=8091), and
    the hardcoded loopback host. health_path is deliberately None so readiness
    falls back to a TCP connect on the resolved port -- whisper-server has no
    /health route. Raises ValueError with a concrete remedy when the binary or
    model path is not configured, so a missing path is an honest failure, not a
    crash or a fabricated success.
    """
    binary = settings.paths.whisper
    if not binary:
        raise ValueError(
            "whisper-server path is not configured; set paths.whisper in "
            "settings.yaml or LOCITIZE_WHISPER_PATH"
        )
    model = settings.paths.whisper_model
    if not model:
        raise ValueError(
            "whisper model path is not configured; set paths.whisper_model in "
            "settings.yaml or LOCITIZE_WHISPER_MODEL_PATH"
        )
    port = settings.ports.whisper
    # Loopback host is fixed (never 0.0.0.0) so the service can never bind the LAN.
    command = [
        binary,
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        # Defense in depth after waveform filtering: prevent Whisper's
        # non-speech token class from becoming words. Native --vad remains off
        # because current Windows whisper-server builds can exit on a silent
        # request when their VAD model finds zero speech segments.
        "--suppress-nst",
    ]
    # Owner-observed 2026-09-03: Open WebUI's voice mode records webm/opus (what
    # a browser MediaRecorder produces), and whisper-server reads WAV. Every
    # utterance came back as a transcription failure. whisper-server's own
    # --convert flag fixes it - its help says "Convert audio to WAV, requires
    # ffmpeg on the server" - so the decode happens in the service that already
    # owns audio, not in a re-implementation of format conversion elsewhere.
    #
    # Added ONLY when ffmpeg is actually discoverable. Passing a flag whose
    # external dependency is missing would turn a working WAV-only setup into a
    # broken one, and "requires ffmpeg" is not a promise this can make for a
    # machine it cannot see.
    # Owner request 2026-09-03 (voice latency). whisper-server defaults to FOUR
    # threads regardless of the machine - its own log says "n_threads = 4 / 20"
    # on this 20-thread CPU - and transcription is the largest single cost in a
    # spoken turn (measured 0.62s of a 1.17s turn, against 0.16s for the LLM).
    #
    # HALF the logical cores, capped at 8. Not all of them: the LLM, Kokoro and
    # Open WebUI share this CPU, and whisper is bursty - taking every core for a
    # 3-second clip would stall the things it is meant to feed.
    threads = max(4, min((os.cpu_count() or 4) // 2, 8))
    command.extend(["--threads", str(threads)])

    if shutil.which("ffmpeg"):
        tmp_dir = Path(settings.data_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # --tmp-dir defaults to "." - the service cwd - so transcoded files
        # would land in the install tree. Point it at the data root instead.
        command.extend(["--convert", "--tmp-dir", str(tmp_dir)])
    return ServiceSpec(
        name=WHISPER_SERVER_NAME,
        command=command,
        # Never None - see resolve_service_cwd: an inherited cwd is the install
        # directory, and every relative file the child writes lands there.
        cwd=resolve_service_cwd(settings),
        env={},
        port=port,
        health_path=None,  # no /health route -> TCP port-connect readiness
        log_path=log_path,
        ready_timeout_s=settings.services.ready_timeout_s,
        stop_timeout_s=settings.services.stop_timeout_s,
        # D-M4-3: whisper_server.log is appended (dated separator per start) rather
        # than truncated, so a start no longer destroys the previous session's
        # failure trace -- exactly the evidence the D-M4-1 investigation lost.
        append_log=True,
    )


def _format_number(value: float) -> str:
    """Render a settings number for the whisper-stream command line.

    An integral float like 100.0 becomes "100" (not "100.0") to match how the tool
    prints its own defaults, while a real fraction like 0.6 is preserved. Uses
    repr-free formatting so no scientific notation or locale comma ever reaches the
    child process, which parses these as plain C floats/ints.
    """
    number = float(value)
    if number.is_integer():
        return str(int(number))
    # %g trims trailing zeros and avoids locale-specific separators.
    return f"{number:g}"


def build_whisper_stream_spec(settings: Settings, log_path: str | None = None) -> ServiceSpec:
    """Build the whisper-stream.exe ServiceSpec (live mic capture) from settings.

    whisper-stream has no network port and no self-terminating flag: it runs until
    it is signalled to stop, printing transcriptions to stdout. So the spec has
    port=None (readiness collapses to "process is alive after launch") and routes
    the child's stdout to log_path, which doubles as the captured-transcript file
    the --listen path tails for the owner. Raises ValueError with a remedy when the
    stream binary or model path is not configured.
    """
    binary = settings.paths.whisper_stream
    if not binary:
        raise ValueError(
            "whisper-stream path is not configured; set paths.whisper_stream in "
            "settings.yaml or LOCITIZE_WHISPER_STREAM_PATH"
        )
    model = settings.paths.whisper_model
    if not model:
        raise ValueError(
            "whisper model path is not configured; set paths.whisper_model in "
            "settings.yaml or LOCITIZE_WHISPER_MODEL_PATH"
        )
    # -m/--model is whisper-stream's model flag (confirmed against the binary's
    # --help). No --host/--port: it captures from the default mic device.
    #
    # VAD gating (defect D-M3-2): --step 0 selects whisper-stream's voice-activity
    # sliding-window mode. In that mode the tool does NOT transcribe every fixed
    # window; it waits for a speech-then-silence boundary and only then runs the
    # model on the detected utterance. That is what stops the model hallucinating
    # stock phrases ("Thank you.", "Thanks for watching.") on silent windows.
    # --vad-thold / --freq-thold are the detector's knobs (both confirmed present
    # in this build's --help). All three values come from settings.speech so an
    # owner can tune mic sensitivity without editing code. Numeric values are
    # rendered plainly (no locale formatting) since the child parses them as C
    # floats/ints.
    speech = settings.speech
    command = [
        binary,
        "--model",
        model,
        "--step",
        str(int(speech.stream_step_ms)),
        "--vad-thold",
        _format_number(speech.vad_thold),
        "--freq-thold",
        _format_number(speech.freq_thold),
    ]
    return ServiceSpec(
        name=WHISPER_STREAM_NAME,
        command=command,
        # Never None - see resolve_service_cwd (invariant W1).
        cwd=resolve_service_cwd(settings),
        env={},
        port=None,  # SDL2 mic capture, no network port
        health_path=None,
        log_path=log_path,
        # A short readiness window: "ready" just means the process launched and
        # stayed alive (readiness collapses to handle.poll() is None for a
        # port-less, health-less service).
        ready_timeout_s=_STREAM_READY_TIMEOUT_S,
        stop_timeout_s=settings.services.stop_timeout_s,
    )


def transcribe_file(
    audio_path: str,
    port: int,
    host: str = "127.0.0.1",
    timeout_s: float = 120.0,
    opener: Any = None,
) -> str:
    """POST an audio file to a running whisper-server /inference and return its text.

    Sends a real multipart/form-data upload (the field name whisper-server expects
    is 'file', with response_format=json) using only the standard library urllib,
    so no third-party HTTP dependency is added (Architecture section 13). The
    server responds with {"text": "..."}; this returns that text stripped of the
    leading/trailing whitespace whisper emits. `opener` is an injectable urlopen
    for tests (defaults to urllib.request.urlopen) so the client's request framing
    and response parsing are unit-tested without a real server.

    Raises OSError/urllib errors on a transport failure and ValueError if the
    server returns a body without a 'text' field -- callers convert these into an
    honest failure outcome rather than a fabricated transcript.
    """
    import urllib.request

    with open(audio_path, "rb") as handle:
        audio = handle.read()

    boundary = "----locitize" + uuid.uuid4().hex
    body = _multipart_body(boundary, audio, "audio.wav")
    headers = {"Content-Type": "multipart/form-data; boundary=" + boundary}
    url = f"http://{host}:{port}/inference"
    request = urllib.request.Request(url, data=body, headers=headers)

    # Injected opener for tests (inspects request.data / returns a fake response);
    # production uses the stdlib urlopen. Either way a real urllib Request is passed.
    call = opener if opener is not None else urllib.request.urlopen
    with call(request, timeout=timeout_s) as response:
        raw = response.read()
    parsed = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(parsed, dict) or "text" not in parsed:
        raise ValueError(f"whisper-server returned no transcript text: {parsed!r}")
    return str(parsed["text"]).strip()


def _multipart_body(boundary: str, audio: bytes, filename: str) -> bytes:
    """Frame a minimal multipart/form-data body: one file part plus response_format.

    Built by hand (rather than via a library) so the client stays dependency-free.
    The two parts are exactly what whisper-server's /inference route reads: the
    'file' upload and a 'response_format=json' field so the reply is JSON.
    """
    pre = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8")
    mid = (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="response_format"\r\n\r\n'
        f"json\r\n"
    ).encode("utf-8")
    end = f"--{boundary}--\r\n".encode("utf-8")
    return pre + audio + mid + end
