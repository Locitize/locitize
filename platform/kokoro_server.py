"""Managed loopback Kokoro text-to-speech server (LOCITIZE M6, Architecture M6.3).

Kokoro ships as a Python library, not a server binary, so LOCITIZE wraps it in this
small authored runner and supervises it exactly like llama.cpp / whisper: a
ServiceSpec on the reserved loopback port ports.kokoro (8092), started and torn
down by the shared ServiceManager (no-orphan guarantee inherited for free).

Why a child process rather than importing torch into the launcher: it keeps the
heavy torch + kokoro import out of the launcher/GUI process (fast startup, clean
crash isolation). The launcher never imports this module; it only spawns it with
the platform venv interpreter (sys.executable), which is where torch/kokoro live.

Contract (all on 127.0.0.1 only):
  GET  /health      -> 200 once the model is loaded (the readiness endpoint)
  POST /synthesize  {"text": str, "voice": str, "speed": float}
                    -> audio/wav bytes (16-bit PCM mono, Kokoro's 24 kHz), or a
                       4xx/5xx with a short JSON error body. Never fabricates audio.
  POST /synthesize/stream  same JSON body
                    -> raw s16le PCM (no wav container), flushed per text chunk with
                       X-Sample-Rate / X-Channels / X-Sample-Width headers. Clients
                       can start playback while later chunks (and the agent tool
                       loop) continue.

The model loads ONCE at startup from the paths passed on argv (the on-disk
kokoro-v1_0.pth plus the directory of voice .pt tensors); the small non-weight
architecture config.json is vendored at assets/kokoro/config.json, so no Hugging
Face hub weight download ever occurs. Wav is written with the stdlib wave module,
so there is no soundfile dependency.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from local_guard import foreign_request_reason

# Kokoro renders at a fixed 24 kHz mono; the wav header must match exactly or the
# audio plays back at the wrong pitch/speed. Single-sourced here.
_SAMPLE_RATE_HZ = 24000
_CHANNELS = 1
_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM

# Loopback host is hardcoded (never 0.0.0.0) so the TTS service can never bind the
# LAN, mirroring every other LOCITIZE managed service.
_LOOPBACK_HOST = "127.0.0.1"

# The Kokoro architecture descriptor (hyperparameters, not weights) vendored in the
# repo so the engine loads fully offline from the on-disk .pth without ever
# reaching the HF hub. Resolved relative to this file so cwd does not matter.
_CONFIG_PATH = Path(__file__).resolve().parent / "assets" / "kokoro" / "config.json"

# American English G2P. Kokoro's lang_code selects the misaki grapheme-to-phoneme
# pipeline; 'a' is American English (the owner's eight voices are en-US/en-GB, all
# served by the American pipeline here for a single, predictable phonemizer path).
_LANG_CODE = "a"

# --------------------------------------------------------------------------- #
# Video memory discipline (measured 2026-09-03, RTX 5070 Ti, CUDA torch).
#
# On the GPU build this server shares the card with the language model, and on
# Windows the driver never refuses either of them memory: when the two together
# exceed the card it pages one of them through system RAM, and the pages do not
# come back. A 27B that generated 29 tok/s beside an idle Kokoro generated 4.8
# after Kokoro spoke one six-sentence paragraph, and still 4.8 twenty-five
# seconds after Kokoro had shrunk again. The reason is torch's caching
# allocator: it keeps every block an utterance needed, sized by the longest
# chunk it ever rendered. This process measured 594 MB after loading, 1026 MB
# after an 80-character sentence, 1842 MB after a paragraph handed over as one
# chunk and 2422 MB after a 1000-character run-on sentence.
#
# Two rules make the footprint a constant the model can be fitted against:
#
# 1. Chunks are bounded. Text is split into sentences, and a sentence longer
#    than _MAX_CHUNK_CHARS is split again at its commas and clause marks, then
#    at word boundaries. Peak memory grew with chunk length (77 chars 1026 MB,
#    155 chars 1192, 200 chars 1426, 296 chars 1770), and 30 sentences rendered
#    one at a time peaked where one sentence did. Kokoro pauses between chunks
#    anyway (Open WebUI already sends one sentence per request), so a split at
#    a comma is a breath, not a glitch.
# 2. The engine warms up to that bound before the socket accepts, rendering a
#    _MAX_CHUNK_CHARS-long sentence once. After that the allocator already
#    holds the largest working set any request can ask for, the footprint the
#    launcher measures when it fits the model (gpu_ledger.fit_budget) is the
#    footprint it keeps, and the first spoken sentence of a call no longer
#    pays the 1s first-render cost either. Freeing the cache after each
#    utterance (torch.cuda.empty_cache) was measured and rejected: it lowers
#    the idle footprint to ~900 MB but every sentence then spikes back up, and
#    the model, fitted against the idle number, gets paged by the spike.
# --------------------------------------------------------------------------- #
_MAX_CHUNK_CHARS = 160

# Join discipline (Agent Portal voice, 2026-09-24): each VRAM-bounded chunk is a
# fresh Kokoro render, so a hard concatenate leaves a quiet dip and a prosody
# reset mid-phrase -- that is the choppy/robotic sound on :8443. Trim near-
# silence at each chunk edge, then crossfade a few tens of ms so the join is a
# breath, not a glitch. Fade length is short enough that two short sentences
# still sound distinct and long enough to hide the allocator seam.
_CROSSFADE_MS = 40
# Absolute float32 amplitude below which a sample counts as silence for trim.
_SILENCE_FLOOR = 0.008
# Keep at least this much audio after trim so a quiet consonant is not erased.
_MIN_KEEP_MS = 80

# The warm-up sentence: exactly _MAX_CHUNK_CHARS long, so the warm-up reserves
# what the largest allowed chunk needs (warmup_text() trims it if the bound
# is ever lowered).
_WARMUP_TEXT = (
    "The quick brown fox jumps over the lazy dog and keeps on running through "
    "the open fields while the weather stays calm and the light fades slowly "
    "over the hills."
)

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


def _pack(pieces: list[str], limit: int) -> list[str]:
    """Greedily join pieces with single spaces without exceeding limit."""
    out: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else f"{current} {piece}"
        if current and len(candidate) > limit:
            out.append(current)
            current = piece
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def split_utterance(text: str, limit: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks of at most `limit` characters for rendering.

    Sentences stay whole when they fit. A longer sentence breaks at commas,
    semicolons and colons, packing clauses back together up to the limit; a
    clause still longer than the limit breaks at word boundaries; a single
    word longer than the limit is left as is (Kokoro will render it). Empty
    input gives an empty list. Pure, so it is tested without torch.
    """
    chunks: list[str] = []
    for sentence in _SENTENCE_BREAK.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= limit:
            chunks.append(sentence)
            continue
        for clause in _pack(_CLAUSE_BREAK.split(sentence), limit):
            if len(clause) <= limit:
                chunks.append(clause)
            else:
                chunks.extend(_pack(clause.split(), limit))
    return chunks


def warmup_text(limit: int = _MAX_CHUNK_CHARS) -> str:
    """The warm-up sentence cut to the chunk bound at a word, ending in a period."""
    text = _WARMUP_TEXT
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return text


def trim_silence(
    audio: Any,
    floor: float = _SILENCE_FLOOR,
    sample_rate: int = _SAMPLE_RATE_HZ,
    min_keep_ms: int = _MIN_KEEP_MS,
) -> Any:
    """Drop near-silent edges so a hard join does not become a mid-phrase pause.

    Numpy is imported inside so split_utterance unit tests still load this module
    without torch/numpy. Empty or all-quiet input is returned unchanged.
    """
    import numpy as np

    if audio is None or len(audio) == 0:
        return audio
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    loud = np.where(np.abs(samples) > floor)[0]
    if loud.size == 0:
        return samples
    start = int(loud[0])
    end = int(loud[-1]) + 1
    min_keep = max(1, int(sample_rate * min_keep_ms / 1000.0))
    if end - start < min_keep:
        mid = (start + end) // 2
        start = max(0, mid - min_keep // 2)
        end = min(len(samples), start + min_keep)
    return samples[start:end]


def join_audio_chunks(
    chunks: list[Any],
    sample_rate: int = _SAMPLE_RATE_HZ,
    fade_ms: int = _CROSSFADE_MS,
) -> Any:
    """Concatenate float32 mono chunks with a linear crossfade at each seam.

    Each chunk is silence-trimmed first. A fade of fade_ms overlaps the tail of
    the running utterance with the head of the next chunk; when a chunk is shorter
    than the fade, they hard-join (rare for real speech). Returns an empty float32
    array when nothing usable remains.
    """
    import numpy as np

    cleaned: list[Any] = []
    for chunk in chunks:
        trimmed = trim_silence(chunk, sample_rate=sample_rate)
        if trimmed is None or len(trimmed) == 0:
            continue
        cleaned.append(np.asarray(trimmed, dtype=np.float32).reshape(-1))
    if not cleaned:
        return np.zeros(0, dtype=np.float32)
    if len(cleaned) == 1:
        return cleaned[0]

    fade = max(1, int(sample_rate * fade_ms / 1000.0))
    out = cleaned[0]
    for nxt in cleaned[1:]:
        n = min(fade, len(out), len(nxt))
        if n <= 0:
            out = np.concatenate([out, nxt])
            continue
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        mixed = out[-n:] * (1.0 - t) + nxt[:n] * t
        out = np.concatenate([out[:-n], mixed, nxt[n:]])
    return out


def pick_device() -> str:
    """"cuda" when torch was built with CUDA and a device is present, else "cpu".

    Kept a plain function so the choice is testable without a GPU: the
    decision is torch's own cuda.is_available(), nothing LOCITIZE-specific.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class KokoroEngine:
    """Loads the Kokoro model once and synthesizes wav bytes on demand.

    This is the only place that imports torch/kokoro. It is constructed once at
    server startup (blocking, several seconds on CPU) so that by the time the HTTP
    socket accepts connections the model is fully loaded -- which is exactly what
    makes GET /health a truthful readiness signal.
    """

    def __init__(self, model_path: str, voices_dir: str) -> None:
        # Imported lazily inside the child so the parent launcher never pays the
        # multi-second torch import cost.
        from kokoro import KModel, KPipeline

        self._voices_dir = Path(voices_dir)
        # config= points at the vendored descriptor; model= points at the on-disk
        # checkpoint. Both are local paths, so no hub download is attempted.
        model = KModel(config=str(_CONFIG_PATH), model=model_path)
        # Owner request 2026-09-03 (voice calls): synthesize on the GPU when the
        # installed torch can see one. Measured on the reference machine: a
        # 13-word sentence took 0.78s on CPU torch and 0.10s on CUDA, and the
        # first spoken sentence of a Call-mode turn waits on exactly this. The
        # CPU wheel stays the default install (an 82M model needs no GPU and
        # the CUDA build is a multi-gigabyte download), so this is a runtime
        # choice, not a requirement: the same file runs either way.
        self.device = pick_device()
        model = model.to(self.device).eval()
        # model=... hands the loaded KModel to the pipeline so it is not reloaded.
        self._pipeline = KPipeline(lang_code=_LANG_CODE, model=model)
        self.warmup_note = self._warm_up()

    def _warm_up(self) -> str:
        """Render one chunk of the maximum size before the socket opens.

        See the video memory notes above _MAX_CHUNK_CHARS: after this the
        allocator holds the largest working set a request can need, so the
        footprint stays put for the life of the process and the first real
        sentence does not pay the first-render cost (measured 1.0s cold,
        0.06-0.11s warm, either device). Uses the first voice on disk; with no
        voices installed there is nothing to warm and the note says so.
        """
        voices = sorted(self._voices_dir.glob("*.pt"))
        if not voices:
            return "no voice files to warm up with"
        try:
            self.synthesize(warmup_text(), voices[0].stem, 1.0)
        except Exception as exc:  # noqa: BLE001 - readiness must not hinge on it
            return f"warm-up failed: {exc}"
        return f"warmed up with {voices[0].stem}"

    def voice_path(self, voice: str) -> Path:
        """Resolve a voice NAME (e.g. 'am_michael') to its on-disk .pt file.

        Only a plain name is accepted (letters, digits, underscore). Anything
        else - '..', a drive letter, or a //host/share path that Windows would
        open over the network (leaking the user's login hash) - is refused
        before the filesystem is touched.
        """
        if not _VOICE_NAME.fullmatch(voice or ""):
            raise ValueError(f"invalid voice name: {voice!r}")
        return self._voices_dir / f"{voice}.pt"

    def _render_chunks(self, text: str, voice: str, speed: float):
        """Yield float32 mono numpy arrays, one per VRAM-bounded text piece.

        Shared by synthesize (joined wav) and synthesize_pcm_stream (progressive
        PCM). Raises ValueError if the voice .pt is missing.
        """
        voice_file = self.voice_path(voice)
        if not voice_file.is_file():
            raise ValueError(f"voice file not found: {voice_file}")

        for piece in split_utterance(text):
            piece_audio: list[Any] = []
            for result in self._pipeline(
                piece, voice=str(voice_file), speed=speed
            ):
                if result.audio is None:
                    continue
                piece_audio.append(result.audio.detach().cpu().numpy())
            if not piece_audio:
                continue
            # A single text piece may still yield multiple KPipeline Results
            # (phoneme waterfall); join those with the same crossfade so the
            # progressive stream does not reintroduce the seam.
            yield join_audio_chunks(piece_audio)

    def synthesize(self, text: str, voice: str, speed: float) -> bytes:
        """Render `text` in `voice` to 16-bit PCM mono wav bytes.

        Raises ValueError if the voice .pt is missing or the model produced no
        audio, so the HTTP layer can return an honest error instead of silence.
        Chunks are silence-trimmed and crossfaded (see join_audio_chunks) so a
        multi-chunk utterance does not sound choppy on the phone speaker.
        """
        import numpy as np

        chunks = list(self._render_chunks(text, voice, speed))
        if not chunks:
            raise ValueError("kokoro produced no audio for the given text")
        audio = join_audio_chunks(chunks)

        # Convert float32 [-1, 1] to 16-bit signed PCM. Clip first so a rare
        # out-of-range sample cannot wrap around into loud noise.
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2")
        return _encode_wav(pcm16.tobytes())

    def synthesize_pcm_stream(self, text: str, voice: str, speed: float):
        """Yield raw s16le PCM bytes per text chunk as soon as each is ready.

        Lets a client start playback (and lets the agent keep running tools)
        while later chunks are still synthesizing. A missing voice raises
        ValueError before any yield.
        """
        import numpy as np

        if not self.voice_path(voice).is_file():
            raise ValueError(f"voice file not found: {self.voice_path(voice)}")

        for chunk in self._render_chunks(text, voice, speed):
            clipped = np.clip(chunk, -1.0, 1.0)
            yield (clipped * 32767.0).astype("<i2").tobytes()


def _encode_wav(pcm_bytes: bytes) -> bytes:
    """Wrap raw 16-bit PCM mono samples in a 24 kHz wav container (stdlib only)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(_CHANNELS)
        writer.setsampwidth(_SAMPLE_WIDTH_BYTES)
        writer.setframerate(_SAMPLE_RATE_HZ)
        writer.writeframes(pcm_bytes)
    return buffer.getvalue()


_VOICE_NAME = re.compile(r"[A-Za-z0-9_]{1,64}")
_MAX_BODY_BYTES = 1_000_000


class _KokoroHandler(BaseHTTPRequestHandler):
    """HTTP handler for /health and /synthesize. The engine is on the server."""

    # Silence the default stderr access log; the child's stdout/stderr is captured
    # to a service log by the parent, and per-request noise is not useful there.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return None

    @property
    def _engine(self) -> KokoroEngine:
        # ThreadingHTTPServer stores the engine on the server instance at startup.
        return self.server.engine  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - required BaseHTTPRequestHandler name
        if self.path.split("?", 1)[0] == "/health":
            # Reaching here means the module imported and the engine constructed,
            # so the model is loaded and the service is genuinely ready.
            self._send_json(200, {
                "status": "ok",
                "sample_rate": _SAMPLE_RATE_HZ,
                "device": self._engine.device,
            })
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - required BaseHTTPRequestHandler name
        refused = foreign_request_reason(
            self.headers.get("Host"), self.headers.get("Origin"), self.server.server_address[1]
        )
        if refused:
            self._send_json(403, {"error": refused})
            return
        route = self.path.split("?", 1)[0]
        if route not in ("/synthesize", "/synthesize/stream"):
            self._send_json(404, {"error": "not found"})
            return
        parsed = self._read_synthesize_payload()
        if parsed is None:
            return
        text, voice, speed = parsed

        if route == "/synthesize/stream":
            self._stream_pcm(text, voice, speed)
            return

        try:
            wav = self._engine.synthesize(text, voice, speed)
        except ValueError as exc:
            # Honest client-side error (unknown voice / empty output).
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - report engine faults, never crash the server
            self._send_json(500, {"error": f"synthesis failed: {exc}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)

    def _read_synthesize_payload(self) -> tuple[str, str, float] | None:
        """Parse the shared synthesize JSON body, or send a 4xx and return None."""
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip()
        if content_type.lower() != "application/json":
            # A browser can only send a no-preflight cross-site POST as
            # text/plain or a form, so requiring JSON closes that door too.
            self._send_json(415, {"error": "Content-Type must be application/json"})
            return None
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length > _MAX_BODY_BYTES:
            self._send_json(413, {"error": "request body too large"})
            return None
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "body must be JSON"})
            return None

        text = str(payload.get("text", "")).strip()
        voice = str(payload.get("voice", "")).strip()
        speed = payload.get("speed", 1.0)
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        if not text:
            self._send_json(400, {"error": "text is required"})
            return None
        if not voice:
            self._send_json(400, {"error": "voice is required"})
            return None
        return text, voice, speed

    def _stream_pcm(self, text: str, voice: str, speed: float) -> None:
        """Flush raw s16le PCM per chunk so playback can start before the end."""
        try:
            stream = self._engine.synthesize_pcm_stream(text, voice, speed)
            first = next(stream, None)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"synthesis failed: {exc}"})
            return
        if first is None:
            self._send_json(400, {"error": "kokoro produced no audio for the given text"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Sample-Rate", str(_SAMPLE_RATE_HZ))
        self.send_header("X-Channels", str(_CHANNELS))
        self.send_header("X-Sample-Width", str(_SAMPLE_WIDTH_BYTES))
        self.send_header("X-Codec", "pcm_s16le")
        # No Content-Length: body is flushed chunk-by-chunk as synthesis proceeds.
        self.end_headers()
        try:
            self.wfile.write(first)
            self.wfile.flush()
            for pcm in stream:
                self.wfile.write(pcm)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client hung up mid-stream (user barge-in); abandon the rest.
            return

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kokoro_server", description="LOCITIZE Kokoro TTS runner")
    parser.add_argument("--host", default=_LOOPBACK_HOST)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True, help="path to kokoro-v1_0.pth")
    parser.add_argument("--voices", required=True, help="dir holding the voice .pt files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the model, then serve until terminated by the parent ServiceManager."""
    args = _parse_args(argv)
    # Host is forced to loopback regardless of any passed value (defense in depth;
    # the spec always passes 127.0.0.1 anyway).
    host = _LOOPBACK_HOST

    # Load the model BEFORE binding the socket so /health is never 200 until the
    # engine is actually ready (the readiness contract the ServiceSpec relies on).
    engine = KokoroEngine(args.model, args.voices)

    server = ThreadingHTTPServer((host, args.port), _KokoroHandler)
    server.engine = engine  # type: ignore[attr-defined]
    # Announce readiness on stdout so the service log shows the ready line; the
    # authoritative readiness signal is still the /health probe.
    print(
        f"kokoro_server ready on http://{host}:{args.port} ({engine.device}; "
        f"{engine.warmup_note})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
