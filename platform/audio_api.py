"""OpenAI-shaped audio endpoints over LOCITIZE's own speech services.

Owner request 2026-09-03: "Open WebUI needs to be my interface for everything."
Chat and vision already work through router.py, but the microphone and speaker
buttons did not, for a reason that is pure dialect - the two sides speak
different APIs for the same job:

    Open WebUI asks for   POST /v1/audio/transcriptions   (multipart, OpenAI)
                          POST /v1/audio/speech           (JSON, OpenAI)

    LOCITIZE serves       POST /inference                 (whisper.cpp's own)
                          POST /synthesize                (kokoro_server's own)

This module is the translation, and nothing else. It starts no process, opens
no socket, and holds no machine-specific path: router.py owns the transport and
hands these functions bytes. Everything here is a pure function, so the whole
mapping is unit-tested with no whisper, no Kokoro, and no audio hardware.

Why translate rather than point Open WebUI at the servers directly: whisper.cpp
returns {"text": ...} already, but wants its upload under a specific field name
with a response_format part; Kokoro wants {"text", "voice", "speed"} where
OpenAI sends {"input", "voice", "speed"}, and rejects a voice it does not have -
which is exactly what Open WebUI sends by default ("alloy").

The alternative was setting Open WebUI's STT engine to "web", the browser's own
recogniser. That works, needs no code, and ships every recorded utterance to
Google - which is a strange thing to do inside a platform whose whole premise is
that the models run on your machine. This keeps the audio local.
"""

from __future__ import annotations

import json
from email.parser import BytesParser
from email.policy import default as _email_policy
from typing import Any

# Answered here rather than forwarded to a model. Both spellings, because
# clients differ on whether they prefix /v1 (the same reason router.py accepts
# /models and /v1/models).
TRANSCRIPTION_PATHS = ("/v1/audio/transcriptions", "/audio/transcriptions")
SPEECH_PATHS = ("/v1/audio/speech", "/audio/speech")

# whisper.cpp's route and the field name its handler reads. Single-sourced here
# so a whisper build change is one edit, matching how models.py keeps the
# llama-server flag spellings.
WHISPER_PATH = "/inference"
WHISPER_FILE_FIELD = "file"

# kokoro_server.py's route (see its do_POST).
KOKORO_PATH = "/synthesize"


def extract_upload(body: bytes, content_type: str) -> tuple[bytes, str] | None:
    """The uploaded audio and its filename from a multipart body, or None.

    Parsed with the stdlib email parser rather than by hand: a multipart body
    can carry the boundary in quotes, parts in any order, and headers folded
    across lines, and re-implementing that correctly is not this module's job.

    Returns None - never raises - when the body is not multipart, carries no
    file part, or is malformed. The caller turns that into an honest 400 rather
    than forwarding something whisper cannot read.
    """
    if not body or "multipart/form-data" not in (content_type or "").lower():
        return None
    raw = (
        b"Content-Type: "
        + content_type.encode("utf-8", "replace")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    try:
        message = BytesParser(policy=_email_policy).parsebytes(raw)
    except Exception:  # noqa: BLE001 - a malformed upload is a 400, not a crash
        return None
    if not message.is_multipart():
        return None
    for part in message.iter_parts():
        filename = part.get_filename()
        name = part.get_param("name", header="content-disposition")
        # A file part is the one with a filename, or the one named as the API
        # says it should be. Either is enough; requiring both would reject
        # clients that send a nameless blob (Open WebUI's recorder does).
        if not filename and name != WHISPER_FILE_FIELD:
            continue
        try:
            payload = part.get_payload(decode=True)
        except Exception:  # noqa: BLE001
            return None
        if payload:
            return payload, (filename or "audio.wav")
    return None


def transcription_body(text: str, response_format: str = "json") -> tuple[bytes, str]:
    """(body, content-type) for a transcription reply in the format asked for.

    OpenAI's default is `json` -> {"text": ...}; `text` returns the bare string.
    Anything else is served as json rather than guessed at, because emitting a
    format this does not actually produce (srt, vtt, verbose_json) would be a
    fabricated response - the timing data needed for subtitles is not something
    whisper-server returns here.
    """
    if (response_format or "").strip().lower() == "text":
        return text.encode("utf-8"), "text/plain; charset=utf-8"
    return json.dumps({"text": text}).encode("utf-8"), "application/json"


def resolve_voice(requested: str, available: list[str], fallback: str) -> str:
    """A voice Kokoro actually has, given whatever the client asked for.

    Open WebUI's default voice is "alloy", one of OpenAI's names; Kokoro's are
    af_bella / am_adam / am_michael / bf_emma. A request for a voice Kokoro does
    not have falls back to the owner's configured tts.voice rather than being
    mapped onto some Kokoro voice by guesswork - "alloy sounds like am_michael"
    is an opinion this module has no basis for, and a wrong guess is worse than
    a consistent default the owner chose.

    Matching is case-insensitive; an exact Kokoro name always wins, so setting
    Open WebUI's voice field to "af_bella" works as expected.
    """
    wanted = (requested or "").strip()
    if wanted:
        for voice in available:
            if voice.lower() == wanted.lower():
                return voice
    return fallback


def speech_to_kokoro(
    body: bytes, available_voices: list[str], default_voice: str
) -> tuple[dict[str, Any] | None, str]:
    """Translate an OpenAI /v1/audio/speech body into a Kokoro /synthesize one.

    Returns (payload, "") on success or (None, reason) on a body this cannot
    honestly serve. The field names differ in exactly one place that matters -
    OpenAI's `input` is Kokoro's `text` - which is the whole reason Open WebUI
    pointed straight at kokoro_server would fail with "text is required".

    `model` is accepted and ignored: OpenAI requires it ("tts-1"), and LOCITIZE
    serves whatever Kokoro checkpoint the owner configured. Claiming to honour a
    model selection there would be a lie.
    """
    if not body:
        return None, "empty request body"
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, "body must be JSON"
    if not isinstance(payload, dict):
        return None, "body must be a JSON object"

    text = payload.get("input")
    if not isinstance(text, str) or not text.strip():
        return None, "input is required"

    speed = payload.get("speed", 1.0)
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = 1.0

    return {
        "text": text.strip(),
        "voice": resolve_voice(
            str(payload.get("voice", "") or ""), available_voices, default_voice
        ),
        "speed": speed,
    }, ""


def requested_response_format(body: bytes, content_type: str) -> str:
    """The `response_format` field of a multipart transcription request.

    Defaults to "json", which is what OpenAI defaults to and what Open WebUI
    expects. Never raises: an unreadable body simply means the default.
    """
    if not body or "multipart/form-data" not in (content_type or "").lower():
        return "json"
    raw = (
        b"Content-Type: "
        + content_type.encode("utf-8", "replace")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    try:
        message = BytesParser(policy=_email_policy).parsebytes(raw)
        if not message.is_multipart():
            return "json"
        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") == "response_format":
                value = part.get_payload(decode=True)
                if value:
                    return value.decode("utf-8", "replace").strip() or "json"
    except Exception:  # noqa: BLE001
        return "json"
    return "json"


def build_whisper_upload(audio: bytes, filename: str) -> tuple[bytes, str]:
    """(body, content-type) for whisper-server's /inference multipart upload.

    Deliberately the same framing whisper.transcribe_file already sends - the
    'file' part plus response_format=json - built here rather than imported so
    this module stays free of whisper.py's Settings-shaped surface. The boundary
    is fixed-prefix plus a random tail for the same reason that function uses
    one: it must not occur inside the audio.
    """
    import uuid

    boundary = "----locitize-audio-" + uuid.uuid4().hex
    pre = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{WHISPER_FILE_FIELD}"; '
        f'filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8")
    mid = (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    return pre + audio + mid, f"multipart/form-data; boundary={boundary}"


def transcript_from_whisper(raw: bytes) -> tuple[str | None, str]:
    """The transcript out of a whisper-server reply, or (None, reason).

    whisper-server answers {"text": "..."}; anything else is reported rather
    than turned into an empty transcript, because a silently-empty transcript
    reads to the owner as "it heard nothing" when the truth is "it failed".
    """
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return None, "whisper-server returned a non-JSON body"
    if not isinstance(parsed, dict) or "text" not in parsed:
        return None, "whisper-server returned no transcript text"
    text = str(parsed["text"]).strip()

    # Owner-observed 2026-09-03: the assistant kept replying "I did not hear
    # anything" to spoken turns. The audio was fine and TTS was fine - whisper
    # answers a silent window with the literal string "[BLANK_AUDIO]", and this
    # passed it through as if the owner had said those words. Open WebUI's voice
    # mode transcribes CONTINUOUSLY, so most chunks are pauses, and the model was
    # reading an annotation as speech.
    #
    # whisper.is_non_speech is the existing, tested predicate for exactly this
    # (built for the mic path, defect D-M3-2): strip whisper's bracketed
    # annotations and see whether anything alphanumeric survives. Reused rather
    # than re-implemented, so the two transcription paths cannot disagree about
    # what counts as silence. An annotation becomes "" - the honest answer to
    # "what was said" is nothing, not the name of the artifact.
    from whisper import is_non_speech

    if text and is_non_speech(text):
        return "", ""
    return text, ""
