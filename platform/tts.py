"""Kokoro text-to-speech: service spec + real client (LOCITIZE M6, Architecture M6.3).

This is the whisper.py analog for voice OUT. It turns the platform Settings into a
declarative ServiceSpec for the managed kokoro_server.py child, and holds the real
client that talks to that running server over loopback HTTP and plays the returned
wav through a Windows-native path.

- build_kokoro_server_spec -> ServiceSpec for the kokoro_server.py loopback runner
- KokoroClient.synthesize   -> POST text to the running server, return wav bytes
- KokoroClient.speak        -> synthesize to a temp .wav and play it (winsound)
- audition                  -> speak one fixed sentence in every on-disk voice
- list_voices               -> the voice names discovered from the on-disk .pt set

It never imports torch or kokoro (the heavy engine lives only in kokoro_server.py,
run as a child). It never launches a process itself (services.py owns that) and
never fabricates audio: synthesize returns exactly the bytes the server produced,
and if the server is unreachable it raises TtsUnavailableError with a remedy rather
than emitting fake silence. All HTTP uses the standard library (urllib), so no new
HTTP dependency is added.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from config import Settings
from services import ServiceSpec, resolve_service_cwd

# Stable service identifier shared by the launcher, the controller, and tests.
KOKORO_SERVER_NAME = "kokoro_server"

# The single fixed sentence spoken by --audition in each voice, so the owner can
# compare voices on identical text. A module constant per Architecture M6.3; the
# launcher may override it from settings.tts.sample_sentence.
AUDITION_SENTENCE = "Hello, I am LOCITIZE. This is how this voice sounds."

# The kokoro server is a Python child that must run in the venv where torch/kokoro
# are installed. The heavy TTS dependencies live in a venv OUTSIDE the compiled
# source tree (so `compileall Codebase/platform` never chokes on torch's
# Python-3.12-only files); this module resolves that interpreter explicitly rather
# than assuming the launcher itself was started with it.
_SERVER_SCRIPT = Path(__file__).resolve().parent / "kokoro_server.py"


def resolve_venv_python(settings: Settings) -> str:
    """Return the interpreter that must run the kokoro child (has torch + kokoro).

    Priority: an explicit settings.paths.venv, then the conventional venv beside
    the platform dir (Codebase/.venv, where the M6 dependencies are installed),
    then an in-tree Codebase/platform/.venv, then the current interpreter as a
    last resort. Resolving this here means the TTS child always runs where kokoro
    is installed even when LOCITIZE itself was launched by a different interpreter.
    """
    candidates: list[Path] = []
    if settings.paths.venv:
        candidates.append(Path(settings.paths.venv) / "Scripts" / "python.exe")
    base = Path(settings.base_dir)
    candidates.append(base.parent / ".venv" / "Scripts" / "python.exe")
    candidates.append(base / ".venv" / "Scripts" / "python.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable

# Loading the ~330MB Kokoro checkpoint on CPU takes several seconds, so readiness
# needs a generous window; this is the floor applied on top of the global default.
_READY_TIMEOUT_FLOOR_S = 30.0


class TtsUnavailableError(RuntimeError):
    """Voice OUT could not be produced. Carries an owner-facing remedy.

    Raised when the kokoro server is unreachable or the weights/runtime are not in
    place. Callers convert it to an honest "text-to-speech is unavailable: <remedy>"
    line -- never a crash, never fabricated silence (Architecture M6.3).
    """

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.remedy = remedy


def list_voices(voices_dir: str) -> list[str]:
    """Return the sorted voice names discovered from the on-disk .pt files.

    A voice name is the stem of a `<name>.pt` file in the configured voices dir
    (e.g. am_michael.pt -> "am_michael"). Returns [] when the dir is unset or
    missing, so the caller can degrade honestly instead of guessing a voice list.
    """
    if not voices_dir:
        return []
    directory = Path(voices_dir)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.pt"))


def build_kokoro_server_spec(settings: Settings, log_path: str | None = None) -> ServiceSpec:
    """Build the kokoro_server.py ServiceSpec from settings (the whisper analog).

    The command is fully data-driven: the venv interpreter (sys.executable), the
    server script, the reserved loopback port (ports.kokoro=8092), and the on-disk
    model checkpoint / voices dir. health_path is "/health" so readiness waits for
    the model to finish loading. Raises ValueError with a concrete remedy when the
    kokoro model or voices path is not configured, so a missing path is an honest
    failure rather than a crash or a fabricated success.
    """
    model = settings.paths.kokoro_model
    if not model:
        raise ValueError(
            "kokoro model path is not configured; set paths.kokoro_model in "
            "settings.yaml or LOCITIZE_KOKORO_MODEL_PATH (the kokoro-v1_0.pth file)"
        )
    voices = settings.paths.kokoro_voices
    if not voices:
        raise ValueError(
            "kokoro voices path is not configured; set paths.kokoro_voices in "
            "settings.yaml or LOCITIZE_KOKORO_VOICES_PATH (the dir of voice .pt files)"
        )
    port = settings.ports.kokoro
    ready_timeout = max(settings.services.ready_timeout_s, _READY_TIMEOUT_FLOOR_S)
    # The child runs in the venv where torch/kokoro are installed (resolved
    # explicitly, not assumed from the parent). Loopback host is fixed (never
    # 0.0.0.0), matching every other LOCITIZE managed service.
    command = [
        resolve_venv_python(settings),
        str(_SERVER_SCRIPT),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        model,
        "--voices",
        voices,
    ]
    return ServiceSpec(
        name=KOKORO_SERVER_NAME,
        command=command,
        # Never None - see resolve_service_cwd (invariant W1).
        cwd=resolve_service_cwd(settings),
        env={},
        port=port,
        health_path="/health",  # 200 only once the model is loaded
        # A healthy kokoro_server already on this port is reused, never duplicated
        # onto another port (see ServiceSpec.reuse_marker).
        reuse_marker=_SERVER_SCRIPT.name,
        log_path=log_path,
        ready_timeout_s=ready_timeout,
        stop_timeout_s=settings.services.stop_timeout_s,
        # Append (dated separator per start) so a failed start does not destroy the
        # previous session's model-load failure trace, matching whisper (D-M4-3).
        append_log=True,
    )


# Leading-silence pad prepended to each played wav (defect D-M7-5). The audio device
# eats the first ~200ms while it warms up, so the first phoneme of a spoken reply was
# faint or clipped ("i do not hear the first words"). A short zero-sample pad gives the
# device time to spin up before real audio starts, at a negligible latency cost.
_LEADING_SILENCE_MS = 200


def _pad_wav_leading_silence(wav_bytes: bytes, pad_ms: int) -> bytes:
    """Return `wav_bytes` with `pad_ms` of leading silence prepended (D-M7-5).

    Parses the wav with the stdlib `wave` module, prepends pad_ms of zero samples at
    the file's own sample rate / width / channel count, and re-serializes. If the
    bytes are not a parseable wav (e.g. a test's canned non-wav bytes) or pad_ms <= 0,
    the input is returned unchanged -- the pad is best-effort and never corrupts or
    fabricates audio. Only real synthesized wavs are padded.
    """
    if pad_ms <= 0:
        return wav_bytes
    import io
    import wave

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as src:
            params = src.getparams()
            frames = src.readframes(src.getnframes())
    except (wave.Error, EOFError, OSError):
        # Not a real wav (or truncated): leave the bytes untouched.
        return wav_bytes
    # One silent frame is (sample width * channels) zero bytes; pad_ms of them at the
    # wav's frame rate is the leading silence.
    silent_frames = int(params.framerate * pad_ms / 1000.0)
    silence = b"\x00" * (silent_frames * params.sampwidth * params.nchannels)
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(silence + frames)
    return out.getvalue()


# Natural pause inserted between concatenated sentences when a whole reply is played
# as one wav (defect D-M7-7). Short enough to sound like ordinary sentence spacing,
# long enough to keep the sentences from running together.
_SENTENCE_GAP_MS = 150


def _silence_frames(framerate: int, sampwidth: int, nchannels: int, ms: int) -> bytes:
    """Return `ms` of zero-sample (silent) wav frame bytes for the given format.

    One silent frame is (sample width * channels) zero bytes; ms of them at the wav's
    frame rate is the silence. Shared by the leading pad and the inter-sentence gap.
    """
    if ms <= 0:
        return b""
    n = int(framerate * ms / 1000.0)
    return b"\x00" * (n * sampwidth * nchannels)


def concatenate_wavs(
    wav_chunks: Iterable[bytes], lead_silence_ms: int, gap_ms: int
) -> bytes:
    """Join same-format wavs into ONE wav: lead pad + chunk1 + gap + chunk2 + ... (D-M7-7).

    Speak-per-sentence played every sentence as a SEPARATE winsound clip, and each
    PlaySound pays an audio-device spin-up (~1s observed) that clipped the first few
    words of EVERY sentence. Buffering the whole reply and playing it as one wav means
    a SINGLE spin-up (covered by one leading pad) and nothing clipped between sentences.

    All Kokoro wavs are 24kHz mono 16-bit, so once every chunk is confirmed to share
    the same (framerate, sampwidth, nchannels) the raw frame bytes concatenate safely
    with zero-sample silence between them -- the concatenation assumption is CHECKED
    here, not assumed: a chunk whose format differs raises ValueError rather than
    producing corrupt audio. Empty or non-wav chunks are skipped. Returns b"" when
    there is nothing to join. A single leading pad of `lead_silence_ms` is prepended,
    and `gap_ms` of silence is inserted only BETWEEN sentences (n-1 gaps, never a
    trailing one).
    """
    import io
    import wave

    parsed: list[tuple[Any, bytes]] = []  # (params, frame bytes) per usable chunk
    for chunk in wav_chunks:
        if not chunk:
            continue
        try:
            with wave.open(io.BytesIO(chunk), "rb") as src:
                params = src.getparams()
                frames = src.readframes(src.getnframes())
        except (wave.Error, EOFError, OSError):
            # Not a real wav (or truncated): skip it rather than corrupt the reply.
            continue
        if not frames:
            continue
        parsed.append((params, frames))
    if not parsed:
        return b""
    base = parsed[0][0]
    fmt = (base.framerate, base.sampwidth, base.nchannels)
    for params, _frames in parsed[1:]:
        other = (params.framerate, params.sampwidth, params.nchannels)
        if other != fmt:
            # Raw-frame concatenation is only valid for identical formats; refuse to
            # silently mangle audio if the assumption ever breaks.
            raise ValueError(
                f"cannot concatenate wavs of differing format: {fmt} vs {other}"
            )
    framerate, sampwidth, nchannels = fmt
    lead = _silence_frames(framerate, sampwidth, nchannels, max(0, lead_silence_ms))
    gap = _silence_frames(framerate, sampwidth, nchannels, max(0, gap_ms))
    body = bytearray(lead)
    for i, (_params, frames) in enumerate(parsed):
        if i > 0:
            body += gap  # gap only BETWEEN sentences, never leading or trailing
        body += frames
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setnchannels(nchannels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)
        dst.writeframes(bytes(body))
    return out.getvalue()


def _winsound_play(path: str) -> None:
    """Play a wav file synchronously via the Windows-native winsound (no deps).

    SND_FILENAME plays the file to completion (blocking). Kept as a module function
    so KokoroClient can inject a fake player in tests and never touch real audio.
    """
    import winsound

    winsound.PlaySound(path, winsound.SND_FILENAME)


class KokoroClient:
    """Client for a running kokoro_server: synthesize wav bytes and play them.

    All process effects (starting the server) belong to the SingleServiceController;
    this client only talks to an already-running server on the resolved loopback
    port. `opener` (an injectable urlopen) and `player` (an injectable playback
    callable) are the two test seams, so request framing and the speak-then-play
    sequence are unit-tested with no real server and no audio hardware.
    """

    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        timeout_s: float = 120.0,
        opener: Any = None,
        player: Callable[[str], None] | None = None,
        lead_silence_ms: int = _LEADING_SILENCE_MS,
    ) -> None:
        self._port = port
        self._host = host
        self._timeout_s = timeout_s
        self._opener = opener
        self._player = player or _winsound_play
        # D-M7-5: leading silence prepended to each played/written wav so the audio
        # device warm-up does not eat the first phoneme. 0 disables the pad.
        self._lead_silence_ms = lead_silence_ms

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> bytes:
        """POST {text, voice, speed} to the running server and return wav bytes.

        Uses stdlib urllib only. Raises TtsUnavailableError (with a remedy) on any
        transport failure -- the caller treats that as "voice is unavailable", never
        as a crash. The returned bytes are exactly what the server produced.
        """
        import urllib.error
        import urllib.request

        body = json.dumps({"text": text, "voice": voice, "speed": speed}).encode("utf-8")
        url = f"http://{self._host}:{self._port}/synthesize"
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        call = self._opener if self._opener is not None else urllib.request.urlopen
        try:
            with call(request, timeout=self._timeout_s) as response:
                return response.read()
        except (urllib.error.URLError, OSError) as exc:
            raise TtsUnavailableError(
                f"kokoro server unreachable on {self._host}:{self._port}: {exc}",
                remedy="start the kokoro service (launcher --smoke-tts) and check "
                "paths.kokoro_model / paths.kokoro_voices",
            ) from exc

    def speak(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        out_path: str | None = None,
        play: bool = True,
        temp_dir: str | None = None,
    ) -> str:
        """Synthesize `text` to a wav file, then play it. Returns the wav path.

        When out_path is given the wav is written there and KEPT (used by
        --speak --wav and every smoke/verify path, which pass an explicit in-tree
        out_path and re-read the bytes).

        When out_path is None this is a one-off (--speak menu, --audition): a temp
        .wav is created under temp_dir when the caller supplies one (the platform
        logs/ dir) so the residue stays inside the gitignored platform tree, and it
        is DELETED after playback -- no caller re-reads a temp-path result, and
        leaving synthesized speech in the OS temp dir was a privacy-hygiene leak
        (SEC-M6-1). The temp-wav-then-play ordering is the exact sequence tests
        assert. Any synthesize failure propagates as TtsUnavailableError, so the
        player is never called on a failed synthesis.
        """
        wav_bytes = self.synthesize(text, voice, speed)
        # D-M7-5: prepend the leading-silence pad so the first phoneme is not clipped
        # by device warm-up. A no-op for non-wav bytes (see _pad_wav_leading_silence),
        # so synthesize's exact-bytes contract is unaffected -- only speak pads.
        wav_bytes = _pad_wav_leading_silence(wav_bytes, self._lead_silence_ms)
        if out_path is not None:
            target = out_path
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as handle:
                handle.write(wav_bytes)
            if play:
                self._player(target)
            return target
        # One-off temp wav: prefer the caller's in-tree dir, then remove the file
        # once it has been played so no derived-speech wav is left on disk (SEC-M6-1).
        if temp_dir is not None:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=temp_dir
        ) as tmp:
            tmp.write(wav_bytes)
            target = tmp.name
        try:
            if play:
                self._player(target)
        finally:
            # Best-effort cleanup; a failed unlink must not mask a playback result.
            try:
                Path(target).unlink()
            except OSError:
                pass
        return target


def audition(
    client: KokoroClient,
    voices: Iterable[str],
    sentence: str = AUDITION_SENTENCE,
    emit: Callable[[str], None] | None = None,
    temp_dir: str | None = None,
) -> None:
    """Speak one fixed sentence in each voice in turn (the file's origin task).

    Prints which voice is speaking before each clip so the owner can finally pick
    a preferred voice. `emit` is an injectable line sink (defaults to print) so the
    sequence is testable without capturing stdout. Playback goes through the
    client's speak(), so a fake player keeps this headless in tests. `temp_dir` is
    forwarded to speak() so each one-off clip lands in the platform tree and is
    cleaned up after playback (SEC-M6-1).
    """
    say = emit or print
    for voice in voices:
        say(f"voice: {voice}")
        client.speak(sentence, voice=voice, play=True, temp_dir=temp_dir)
