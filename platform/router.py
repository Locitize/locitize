"""OpenAI-compatible model router: the whole registry behind one endpoint.

Owner request 2026-09-02. Open WebUI is pointed straight at the running
llama-server, which serves exactly ONE model and therefore advertises exactly
one on /v1/models. The picker showed a single entry, and choosing a different
LOCITIZE model meant leaving the browser, switching in the command center, and
coming back.

This module is the missing piece: a loopback front that answers /v1/models from
the REGISTRY (every launchable row, no GPU touched) and, when a chat request
names a model that is not the running one, asks the ModelController to switch
before forwarding. Selecting a model in Open WebUI's picker therefore loads it
on the GPU, through the same one-model-at-a-time discipline every other
LOCITIZE surface uses (M8.2) - the router starts nothing itself.

Relationship to proxy.py: that module exists to reach the ACTIVE model at a
friendly hostname and is deliberately transparent - it forwards everything and
decides nothing. This one decides which model should be running, so it is a
separate module rather than a flag on that one. The forwarding hop is
deliberately the same shape (http.client, no redirect following, chunked body
relay so SSE streaming survives) because that shape is already proven.

Design and safety, matching proxy.py's constraints:
- Binds 127.0.0.1 ONLY (hardcoded), never a routable interface.
- Reads the active port fresh from a callable per request, so it follows the
  model across a switch.
- Injectable `switch_fn` / `running_model_id_fn` / `port_provider` rather than a
  services import, so the whole decision path is unit-tested with no GPU, no
  subprocess, and no socket (the pure helpers below take no self at all).
- Switches are serialised behind one lock. Two browser tabs asking for two
  different models must not race a pair of 13GB loads onto one card.

Honest failure only: an unknown model name returns 404 listing what IS
available, and a failed switch returns 503 with a remedy. Outside tooling
traffic the router never answers with a different model than the one asked
for. Tooling requests (body carries tools=) that name a known-broken
tool-protocol model (obliterated/abliterated/heretic) are redirected to a
proven tool-capable row or refused with 422.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import audio_api
from audio_filter import AudioFilterResult

# The router only ever forwards to loopback; hardcoded so no config value can
# steer it off 127.0.0.1 (Permission Matrix section 7, same rule as proxy.py).
_UPSTREAM_HOST = "127.0.0.1"

# Requests larger than this are refused rather than buffered unbounded.
_MAX_BODY_BYTES = 32 * 1024 * 1024

# Cap on a single chunked-encoding header line, so a sender cannot make
# readline() buffer without limit before the size guard has seen a byte.
_MAX_CHUNK_HEADER_BYTES = 1024

# Answered from the registry and never forwarded: listing models must not load
# one. Open WebUI polls this on every page load and applies a SHORT timeout to
# it (AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST), so a listing that waited on a model
# load would time out and the picker would come back empty.
MODEL_LIST_PATHS = ("/v1/models", "/models")

# Paths whose JSON body names a model, and therefore may trigger a switch.
SWITCHING_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/chat/completions",
    "/completions",
)


def build_models_payload(
    models: list[Any], running_model_id: str | None = None
) -> dict[str, Any]:
    """The OpenAI /v1/models response for a list of registry rows.

    The RUNNING model is listed first. Owner-observed 2026-09-03: LOCITIZE was
    serving Kimi-VL while a new Open WebUI chat showed gemma-4-E2B. Open WebUI
    has no way to ask what is loaded - its ui.default_models is null on this
    install - so a new chat falls back to the FIRST entry of /v1/models, which
    was simply the first registry row. Two costs: the picker asserted a model
    that was not running, and typing into that chat would switch away from a
    model already loaded and ready, paying a reload nobody asked for.

    Ordering is the whole fix, and it is honest: the list still contains every
    launchable row, in registry order, with the one that is ready to answer
    right now at the top. The name is NOT decorated with "(running)" - clients
    send the advertised name back and resolve_requested_model matches it
    exactly, so a decorated name would stop resolving.

    `id` is the REGISTRY id, not the row's display name, because the id is the
    thing every other LOCITIZE surface addresses a model by and it is unique by
    construction (validated at registry load). The human name rides along as
    `name`, which recent Open WebUI builds show in the picker when present and
    older ones ignore harmlessly.

    `created` is 0 rather than a fabricated timestamp: LOCITIZE does not know
    when a model was made, and inventing a date to fill an OpenAI-shaped field
    would be a fabricated claim. Clients sort by it at worst.
    """
    ordered = list(models)
    if running_model_id:
        running = [m for m in ordered if m.id == running_model_id]
        if running:
            ordered = running + [m for m in ordered if m.id != running_model_id]
    return {
        "object": "list",
        "data": [
            {
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": "locitize",
                "name": model.name or model.id,
            }
            for model in ordered
        ],
    }


def resolve_requested_model(requested: str, models: list[Any]) -> str | None:
    """Map a request's `model` string to a registry id, or None if unknown.

    Matched against the id first, then the display name, then either
    case-insensitively. The name is accepted because llama-server is started
    with `--alias <name>` (models.py _ALIAS_FLAG), so a client that discovered
    the model from the RUNNING server rather than from this router will send the
    name back - and refusing that would break the exact flow this router exists
    to smooth.

    An empty request is not a match: OpenAI clients may omit `model`, and the
    caller treats that as "whatever is already running" rather than as an error.
    """
    wanted = (requested or "").strip()
    if not wanted:
        return None
    for model in models:
        if model.id == wanted:
            return model.id
    for model in models:
        if (model.name or "") == wanted:
            return model.id
    lowered = wanted.lower()
    for model in models:
        if model.id.lower() == lowered or (model.name or "").lower() == lowered:
            return model.id
    return None


def requested_model_from_body(body: bytes) -> str:
    """The `model` field of a JSON request body, or "" when there is not one.

    Deliberately total: a malformed or non-JSON body yields "", which the caller
    treats as "no switch requested" and forwards verbatim. Refusing to forward
    a body this function could not parse would break any endpoint whose payload
    shape it does not know about.
    """
    if not body:
        return ""
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    value = parsed.get("model")
    return value.strip() if isinstance(value, str) else ""


EARLIER_MODEL_TAG = "[earlier model] "

# Stable opening of the system note, used to recognise a note this module has
# already added. The tag on a turn is self-guarding (tag_assistant_turn checks
# for it), but the note lives in a DIFFERENT message, so without this a body
# passed through twice would carry two copies of it.
_NOTE_SENTINEL = "LOCITIZE note: this conversation has changed model."


def build_handoff_note(previous_model_name: str) -> str:
    """The system note added when a conversation changes model mid-flight.

    Owner-observed defect 2026-09-03: after switching Qwen2.5-VL -> gpt-oss-20b
    inside ONE Open WebUI conversation, gpt-oss answered "I am Qwen, a large
    language model created by Alibaba Cloud". The router was switching correctly
    - llama-server really was serving gpt-oss-20b-F16, confirmed from both
    /props and the process argv - and the transcript was the cause. Open WebUI
    keeps one conversation across a model change, so the incoming model reads a
    prior ASSISTANT turn saying "I am Qwen" and, with no way to know a different
    model wrote it, continues that persona.

    MEASURED on gpt-oss-20b-F16, 6 trials of "Who are you?" after a Qwen turn:
        nothing (baseline)     1/6 correct, 5/6 inherited the wrong identity
        this note ALONE        4/6 correct
        inline tags alone      5-6/6 correct
        note + inline tags     6/6 correct
    The note alone was not enough, which is why apply_handoff_note also tags the
    turns themselves - a system line loses to an explicit prior turn written in
    the model's own voice. None of these is deterministic: this is prompt-level
    mitigation of a model behaviour, not a guarantee.

    The note states only FACTS LOCITIZE actually has. It deliberately does NOT
    tell the model what it is - a GGUF header carries no vendor identity, and the
    registry id is a filename the owner chose - only that the marked turns may
    not be its own. It says "may have been" and "most recently" because the
    router knows the PREVIOUS model, not the author of every historic turn: a
    conversation that went A -> B -> A contains turns the incoming model really
    did write, and claiming otherwise would be a fabrication.
    """
    return (
        "LOCITIZE note: this conversation has changed model. Assistant turns "
        f"marked {EARLIER_MODEL_TAG.strip()} were written earlier, and may be "
        f"the words of a different local model (most recently "
        f"{previous_model_name}) rather than your own. Do not adopt another "
        "model's name, identity, or statements about itself as your own, and do "
        "not copy the marker into your reply. Answer as whichever model you "
        "actually are."
    )


def message_has_text(message: Any) -> bool:
    """True when a chat message carries real text, in either content shape.

    A string content is the ordinary case; a list is the multimodal shape Open
    WebUI sends once an image is attached, where text lives in {"type": "text",
    "text": ...} parts. An image-only assistant turn carries no persona to
    inherit, so it does not count.
    """
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and part["text"].strip()
            for part in content
        )
    return False


def tag_assistant_turn(message: dict) -> dict:
    """A copy of an assistant turn prefixed with the earlier-model marker.

    Returns the message unchanged when it already carries the marker, so a
    conversation that switches model several times does not accumulate a stack
    of them. Handles both content shapes; for a content array only the FIRST
    text part is prefixed, which is where the reader starts.
    """
    content = message.get("content")
    if isinstance(content, str):
        if content.lstrip().startswith(EARLIER_MODEL_TAG.strip()):
            return message
        tagged = dict(message)
        tagged["content"] = EARLIER_MODEL_TAG + content
        return tagged
    if isinstance(content, list):
        parts = list(content)
        for index, part in enumerate(parts):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                if part["text"].lstrip().startswith(EARLIER_MODEL_TAG.strip()):
                    return message
                new_part = dict(part)
                new_part["text"] = EARLIER_MODEL_TAG + part["text"]
                parts[index] = new_part
                tagged = dict(message)
                tagged["content"] = parts
                return tagged
    return message


def apply_handoff_note(body: bytes, previous_model_name: str) -> bytes:
    """Mark a chat request as having changed model, so the incoming model does
    not inherit the previous one's persona from the transcript.

    Does two things, because measurement showed one was not enough (see
    build_handoff_note for the trial counts):
      1. tags every prior assistant turn with EARLIER_MODEL_TAG, inline where
         the model actually reads it;
      2. adds a system note explaining the marker and naming the model that most
         recently answered.

    Returns the body UNCHANGED when the marking would be pointless or the
    payload is not a shape this understands - the router must never corrupt a
    request it could not fully parse. Skipped when no assistant turn with real
    text exists: on the first message there is no persona to inherit, and a note
    about turns that do not exist is noise in the prompt.

    Only the FORWARDED request is rewritten. Open WebUI's stored conversation is
    untouched, so the markers never appear in the owner's chat history.

    The system note is APPENDED to an existing leading system message rather
    than added as a second one: several chat templates render only the first
    system message and silently drop later ones.
    """
    if not body:
        return body
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return body
    if not any(
        isinstance(m, dict) and m.get("role") == "assistant" and message_has_text(m)
        for m in messages
    ):
        return body

    updated = [
        tag_assistant_turn(m)
        if isinstance(m, dict) and m.get("role") == "assistant" and message_has_text(m)
        else m
        for m in messages
    ]

    note = build_handoff_note(previous_model_name)
    first = updated[0] if isinstance(updated[0], dict) else None
    if first is not None and first.get("role") == "system" and isinstance(
        first.get("content"), str
    ):
        if _NOTE_SENTINEL in first["content"]:
            # Already marked; re-adding would stack duplicate notes.
            new_payload = dict(payload)
            new_payload["messages"] = updated
            return json.dumps(new_payload).encode("utf-8")
        merged = dict(first)
        merged["content"] = first["content"].rstrip() + "\n\n" + note
        updated[0] = merged
    else:
        # A non-string system content (a content array) is not ours to splice;
        # add the note as its own message rather than corrupt one.
        updated.insert(0, {"role": "system", "content": note})

    new_payload = dict(payload)
    new_payload["messages"] = updated
    return json.dumps(new_payload).encode("utf-8")


# ---------------------------------------------------------------------------
# Voice turns (owner request 2026-09-03: "talk to my models like ChatGPT")
# ---------------------------------------------------------------------------
# Open WebUI's Call overlay sends what whisper heard as an ordinary chat
# request, so on the wire a spoken turn is indistinguishable from a typed one.
# The router serves the transcription itself (M8.2), so it KNOWS the text it
# just returned: a chat request whose last user message is a transcript it
# produced within the last few seconds is a voice turn. Two things then change
# on the FORWARDED request only (the stored chat is untouched):
#   1. thinking is turned off for this turn. A reasoning model says nothing
#      until it has finished reasoning, and in a call that silence is the whole
#      answer. Measured on llama-server b10701 across every registered model:
#      chat_template_kwargs.enable_thinking=false is what Qwen3/Gemma-style
#      templates honour (1.77s -> 1.05s to the first token, zero reasoning),
#      gpt-oss ignores it but reasoning_effort="low" cuts its reasoning to a
#      sentence (0.79s -> 0.35s). No template raises on either field, so both
#      are sent, and only when the client did not choose a value itself.
#   2. a one-line system note says the answer will be read aloud, so the
#      model speaks in sentences instead of markdown tables and code fences.
# The window is generous because a transcript is returned BEFORE the browser
# posts the chat request, and a slow tab must not turn a call into text.
VOICE_TURN_WINDOW_S = 10.0
_VOICE_SENTINEL = "LOCITIZE voice turn:"
VOICE_TURN_NOTE = (
    f"{_VOICE_SENTINEL} the user is speaking to you and will hear this answer "
    "read aloud. Reply in plain conversational sentences, briefly, without "
    "markdown, lists, headings or code unless asked for them."
)


def normalize_utterance(text: str) -> str:
    """Whitespace-folded form used to match a transcript to a chat message."""
    return " ".join(text.split())


def message_text(message: Any) -> str:
    """The text of a chat message in either content shape ("" when none)."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def last_user_utterance(body: bytes) -> str | None:
    """Normalized text of the LAST user message in a chat body, or None."""
    if not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return None
    for message in reversed(payload["messages"]):
        if isinstance(message, dict) and message.get("role") == "user":
            text = normalize_utterance(message_text(message))
            return text or None
    return None


def apply_thinking_defaults(body: bytes) -> bytes:
    """Turn thinking off on chat completions unless the client already chose.

    Observed 2026-09-24 (phone OWUI "stuck"): a reasoning model with
    llama-server reasoning_format=deepseek (template auto) returns every token
    in reasoning_content and leaves content empty. Open WebUI then shows blank
    assistant turns even while the GPU is busy. Measured: chat_template_kwargs
    enable_thinking=false restores a normal content field (same fields the
    voice-turn path already set). Applies to typed and spoken turns; never
    overrides a value the client set.
    """
    if not body:
        return body
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return body

    new_payload = dict(payload)
    kwargs = new_payload.get("chat_template_kwargs")
    kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
    kwargs.setdefault("enable_thinking", False)
    new_payload["chat_template_kwargs"] = kwargs
    new_payload.setdefault("reasoning_effort", "low")
    return json.dumps(new_payload).encode("utf-8")


def apply_voice_turn(body: bytes) -> bytes:
    """Rewrite one chat request for a spoken turn (see the section note).

    Pure and idempotent: the sentinel guards the note, and a field the client
    already set (a chat whose owner chose an effort, a model row that fixes
    thinking) is never overridden. Returns the body unchanged when it is not a
    chat-completion payload this understands. Thinking defaults are applied
    first via apply_thinking_defaults (typed chats need the same fix).
    """
    body = apply_thinking_defaults(body)
    if not body:
        return body
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return body

    new_payload = dict(payload)
    updated = list(messages)
    first = updated[0] if isinstance(updated[0], dict) else None
    if first is not None and first.get("role") == "system" and isinstance(
        first.get("content"), str
    ):
        if _VOICE_SENTINEL not in first["content"]:
            merged = dict(first)
            merged["content"] = first["content"].rstrip() + "\n\n" + VOICE_TURN_NOTE
            updated[0] = merged
    else:
        updated.insert(0, {"role": "system", "content": VOICE_TURN_NOTE})
    new_payload["messages"] = updated
    return json.dumps(new_payload).encode("utf-8")



def _message_content_blank(message: dict) -> bool:
    """True when a chat message has no usable content (string or multimodal)."""
    content = message.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return not any(
            isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            and part["text"].strip()
            for part in content
        )
    return True


def promote_reasoning_to_content(payload: dict) -> dict:
    """Copy reasoning_content into content when content is empty.

    2026-09-24: any llama-server deepseek-format
    row) can still emit only reasoning_content even after thinking defaults.
    Open WebUI / Portal render content, so empty content looks stuck. Pure:
    returns a shallow-copied payload; never invents text when reasoning is
    also empty; never overrides non-blank content.
    """
    if not isinstance(payload, dict):
        return payload
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return payload
    new_choices: list = []
    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            new_choices.append(choice)
            continue
        choice = dict(choice)
        message = choice.get("message")
        delta = choice.get("delta")
        if isinstance(message, dict):
            message = dict(message)
            reasoning = message.get("reasoning_content")
            if (
                isinstance(reasoning, str)
                and reasoning.strip()
                and _message_content_blank(message)
            ):
                message["content"] = reasoning
                choice["message"] = message
                changed = True
            else:
                choice["message"] = message
        if isinstance(delta, dict):
            delta = dict(delta)
            reasoning = delta.get("reasoning_content")
            content = delta.get("content")
            content_blank = content is None or (
                isinstance(content, str) and not content.strip()
            )
            if isinstance(reasoning, str) and reasoning and content_blank:
                # Stream: promote this delta's reasoning token into content
                # so OWUI paints tokens live instead of a blank bubble.
                delta["content"] = reasoning
                choice["delta"] = delta
                changed = True
            else:
                choice["delta"] = delta
        new_choices.append(choice)
    if not changed:
        return payload
    out = dict(payload)
    out["choices"] = new_choices
    return out


def promote_completion_bytes(raw: bytes) -> bytes:
    """Rewrite one non-stream chat-completion JSON body (identity on failure)."""
    if not raw:
        return raw
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return raw
    if not isinstance(payload, dict):
        return raw
    promoted = promote_reasoning_to_content(payload)
    if promoted is payload:
        return raw
    return json.dumps(promoted, ensure_ascii=False).encode("utf-8")


def promote_sse_chunk(raw: bytes) -> bytes:
    """Rewrite one SSE frame (possibly multiple data: lines) for streaming."""
    if not raw or b"data:" not in raw:
        return raw
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    out_parts: list[str] = []
    changed = False
    for line in decoded.splitlines(keepends=True):
        if not line.startswith("data:"):
            out_parts.append(line)
            continue
        _prefix, _sep, rest = line.partition(":")
        if rest.startswith(" "):
            space, payload = " ", rest[1:]
        else:
            space, payload = "", rest
        if payload.endswith("\r\n"):
            core, nl = payload[:-2], "\r\n"
        elif payload.endswith("\n"):
            core, nl = payload[:-1], "\n"
        elif payload.endswith("\r"):
            core, nl = payload[:-1], "\r"
        else:
            core, nl = payload, ""
        if core.strip() == "[DONE]":
            out_parts.append(line)
            continue
        try:
            obj = json.loads(core)
        except ValueError:
            out_parts.append(line)
            continue
        if not isinstance(obj, dict):
            out_parts.append(line)
            continue
        promoted = promote_reasoning_to_content(obj)
        if promoted is obj:
            out_parts.append(line)
            continue
        new_core = json.dumps(promoted, ensure_ascii=False)
        out_parts.append(f"data:{space}{new_core}{nl}")
        changed = True
    if not changed:
        return raw
    return "".join(out_parts).encode("utf-8")


# ---------------------------------------------------------------------------
# Portal / tooling guard (2026-09-24)
# ---------------------------------------------------------------------------
# Agent Portal sends tools= on every turn. Obliterated/abliterated/heretic
# rows still advertise "tools" in GGUF metadata but break the tool protocol
# (raw XML, bare laya_route_* as chat). For tooling traffic the router must
# not happily serve them: redirect to a proven tool-capable row, or refuse.
# When the user ask clearly needs browse/list/read/desktop (or the model is
# tool-broken), answer the FIRST turn with a synthetic laya_route_tools call
# so Laya System-1 steers on CPU before any chat model burns a GPU turn.
# Subsequent turns (messages already carry tool / tool_calls) forward normally.
# Non-tooling clients (Open WebUI plain chat) are unchanged.

TOOL_BROKEN_MODEL_SUBSTR = ("obliterated", "abliterated", "heretic")

# The routing tool Agent Portal advertises; the pre-steer only fires for it.
LAYA_ROUTE_TOOL = "laya_route_tools"

# Preference order for redirect targets. Matched as id/name substrings against
# the live registry so a missing row falls through to the next.
TOOL_CAPABLE_PREFERENCE = (
    "qwen3-coder",
    "gpt-oss",
)

_TOOLS_NEEDED_RE = re.compile(
    r"(?:"
    r"\b(?:list_dir|read_file|browse_\w+|screenshot)\b"
    r"|\b(?:list|ls|dir)\b.{0,40}\b(?:file|folder|director(?:y|ies)|project|repo)s?\b"
    r"|\b(?:read|open|show|find|search|look\s+at)\b.{0,40}\b(?:file|folder|director(?:y|ies)|readme|project|repo)s?\b"
    r"|\b(?:browse|open)\b.{0,40}\b(?:web|website|site|url|page|http)\b"
    r"|\b(?:screenshot|snap|capture)\b"
    r"|\b(?:desktop|focused\s+window)\b"
    r"|\b(?:weather|sports|news|scores?|stocks?|prices?|flights?|traffic)\b"
    r"|what(?:'s|s|\s+is)\s+(?:this\s+)?(?:folder|project|directory|dir|repo)\s+about"
    r"|what(?:'s|s|\s+is)\s+(?:on\s+)?(?:my\s+)?screen"
    r"|(?:describe|summarize|explain)\s+(?:this\s+)?(?:folder|project|directory|repo)"
    r")",
    re.I | re.S,
)


def is_tool_broken_model(model_id: str) -> bool:
    """True when the id/name is a known-broken tool-protocol class."""
    mid = (model_id or "").strip().lower()
    if not mid:
        return False
    return any(bad in mid for bad in TOOL_BROKEN_MODEL_SUBSTR)


def request_has_tools(body: bytes) -> bool:
    """True when the JSON body carries a non-empty tools array (portal/tooling)."""
    if not body:
        return False
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False
    if not isinstance(parsed, dict):
        return False
    tools = parsed.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def ask_needs_tools(text: str) -> bool:
    """True when the user ask clearly needs browse/list/read/desktop tools."""
    return bool(_TOOLS_NEEDED_RE.search(text or ""))


def messages_have_tool_exchange(body: bytes) -> bool:
    """True when history already has tool results or assistant tool_calls.

    Used to run Laya pre-steer only on the first tooling turn, so later turns
    (after portal executed laya_route_tools) forward to the chat model.
    """
    if not body:
        return False
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False
    if not isinstance(parsed, dict):
        return False
    messages = parsed.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            return True
        if role == "assistant" and message.get("tool_calls"):
            return True
    return False


def pick_tool_capable_fallback(
    models: list[Any],
    *,
    running_model_id: str | None = None,
) -> str | None:
    """Registry id of a proven tool-capable model, or None if none available.

    Preference: qwen3-coder / gpt-oss (substring match on id),
    then the running model if it is not tool-broken, then any non-broken row.
    Never returns a tool-broken id.
    """
    rows = list(models or [])
    by_id = {getattr(m, "id", ""): m for m in rows if getattr(m, "id", None)}

    def _ok(model_id: str | None) -> str | None:
        if not model_id or model_id not in by_id:
            return None
        if is_tool_broken_model(model_id):
            return None
        name = (getattr(by_id[model_id], "name", None) or "").lower()
        if is_tool_broken_model(name):
            return None
        return model_id

    for needle in TOOL_CAPABLE_PREFERENCE:
        for model in rows:
            mid = getattr(model, "id", "") or ""
            name = (getattr(model, "name", None) or "").lower()
            if needle in mid.lower() or needle in name:
                picked = _ok(mid)
                if picked:
                    return picked
    picked = _ok(running_model_id)
    if picked:
        return picked
    for model in rows:
        picked = _ok(getattr(model, "id", None))
        if picked:
            return picked
    return None


def rewrite_body_model(body: bytes, model_id: str) -> bytes:
    """Return body with `model` set to model_id; untouched on parse failure."""
    if not body or not model_id:
        return body
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(parsed, dict):
        return body
    parsed = dict(parsed)
    parsed["model"] = model_id
    return json.dumps(parsed).encode("utf-8")


# OWUI native FC injects generate_image/search_web/etc. Portal uses agent tools.
# Never synthesize laya_route_tools for OWUI builtins — OWUI cannot execute that
# call and the chat hangs with empty assistant (done=false). Locitize still
# redirects tool-broken models to a capable fallback before forwarding.
OWUI_BUILTIN_TOOLS = frozenset({
    "generate_image",
    "edit_image",
    "search_web",
    "fetch_url",
    "execute_code",
    "search_memories",
    "list_memories",
    "add_memory",
    "update_memory",
    "delete_memory",
    "view_file",
    "query_chat_files",
    "list_chat_files",
    "search_chats",
    "view_chat",
    "ask_user",
    "delegate_task",
    "timer",
})


def request_tool_names(body: bytes) -> set[str]:
    """Names of the functions in the request's tools array (empty if none)."""
    if not body:
        return set()
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return set()
    if not isinstance(parsed, dict):
        return set()
    tools = parsed.get("tools")
    if not isinstance(tools, list):
        return set()
    names = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
        elif tool.get("name"):
            names.add(str(tool["name"]))
    return names


def request_has_owui_builtin_tools(body: bytes) -> bool:
    """True when the tools array is OWUI chat builtins (not Agent Portal)."""
    return bool(request_tool_names(body) & OWUI_BUILTIN_TOOLS)


def should_pre_steer_laya(body: bytes, requested_model: str) -> bool:
    """Whether this tooling request should get a synthetic laya_route_tools turn."""
    if not request_has_tools(body):
        return False
    # Only a client that offers laya_route_tools (Agent Portal) can execute the
    # synthetic call; any other tool-using client would be handed a call to a
    # tool it does not have.
    if LAYA_ROUTE_TOOL not in request_tool_names(body):
        return False
    if messages_have_tool_exchange(body):
        return False
    # OWUI image/chat feature path: rewrite-to-fallback is fine; Laya steer is not.
    if request_has_owui_builtin_tools(body):
        return False
    if is_tool_broken_model(requested_model):
        return True
    utterance = last_user_utterance(body) or ""
    # Portal user messages look like "Project folder: ...\n\nTask: ...".
    task = _portal_task_text(utterance) or utterance
    return ask_needs_tools(task)


def _portal_task_text(user_content: str) -> str:
    """Pull the Task: section from a portal user message, else the whole text.

    Handles both raw newlines and the whitespace-folded form last_user_utterance
    returns ("\\n\\nTask:" becomes " Task:").
    """
    text = user_content or ""
    match = re.search(r"(?:^|\s)Task:\s*(.*)\Z", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text


def build_laya_steer_completion(
    *,
    model_id: str,
    task: str,
    created: int | None = None,
) -> dict[str, Any]:
    """OpenAI-shaped chat.completion that forces laya_route_tools (no GPU).

    Portal's Engine parses tool_calls and the run loop executes laya_route_tools
    against the CPU sidecar, then continues. The chat model is never invoked
    for this turn.
    """
    stamp = int(time.time()) if created is None else int(created)
    task_text = (task or "").strip() or "route the current task"
    arguments = json.dumps({"task": task_text}, ensure_ascii=False)
    return {
        "id": "chatcmpl-laya-steer",
        "object": "chat.completion",
        "created": stamp,
        "model": model_id or "laya-steer",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_laya_steer_1",
                            "type": "function",
                            "function": {
                                "name": LAYA_ROUTE_TOOL,
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# Model identity (owner request 2026-09-24: "still not saying the model name")
# ---------------------------------------------------------------------------
# A local model cannot know which GGUF it was loaded from. The weights carry no
# dependable self-name, and llama-server puts none in the prompt, so asked "what
# model are you?" a row either guesses from its training data or recites what
# the client's system prompt claimed. The owner hit the second failure: a stale
# Open WebUI system prompt still said "MiniCPM-o 4.5", so every model, including
# a freshly registered one, answered with that name.
#
# The router is the one component that KNOWS, because it performed the switch.
# It states the running row's display name on each forwarded chat, appended to
# the first system message for the same reason the notes above are (templates
# render only the first one). The name is the registry's, not the model's guess,
# so it stays correct for every row without per-model prompt editing.
_IDENTITY_SENTINEL = "LOCITIZE model identity:"


def build_identity_note(model_name: str) -> str:
    """The one-line system note naming the model that is about to answer."""
    return (
        f"{_IDENTITY_SENTINEL} you are {model_name}, running locally on this "
        "computer through LOCITIZE. When asked which model or assistant you "
        "are, answer with that name."
    )


def apply_model_identity(body: bytes, model_name: str | None) -> bytes:
    """Name the running model in one chat request (see the section note).

    Pure and idempotent, like apply_voice_turn: the sentinel guards the note, so
    a body passed through twice carries one. Returns the body unchanged when the
    name is unknown or the payload is not a chat this understands - an unnamed
    model is better than a wrong name.
    """
    name = (model_name or "").strip()
    if not body or not name:
        return body
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return body

    note = build_identity_note(name)
    updated = list(messages)
    first = updated[0] if isinstance(updated[0], dict) else None
    if first is not None and first.get("role") == "system" and isinstance(
        first.get("content"), str
    ):
        if _IDENTITY_SENTINEL in first["content"]:
            return body
        merged = dict(first)
        merged["content"] = first["content"].rstrip() + "\n\n" + note
        updated[0] = merged
    else:
        # A non-string system content (a content array) is not ours to splice.
        updated.insert(0, {"role": "system", "content": note})

    new_payload = dict(payload)
    new_payload["messages"] = updated
    return json.dumps(new_payload).encode("utf-8")


def build_upstream_target(active_port: int | None) -> tuple[str, int] | None:
    """(host, port) for the forwarding hop, or None when no model is running."""
    if not active_port:
        return None
    return _UPSTREAM_HOST, int(active_port)


class _RouterHandler(BaseHTTPRequestHandler):
    """Answers model listings locally; forwards everything else, switching first."""

    # Quiet per-request logging, same reason as proxy.py: the platform logs
    # elsewhere and a stderr line per request would spam the console.
    def log_message(self, *args: Any) -> None:  # noqa: D401 - stdlib signature
        return

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_OPTIONS(self) -> None:
        self._handle("OPTIONS")

    def do_DELETE(self) -> None:
        self._handle("DELETE")

    # ---- routing ---------------------------------------------------------- #

    def _handle(self, method: str) -> None:
        router = self.server.router  # type: ignore[attr-defined]
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        if method == "GET" and path in MODEL_LIST_PATHS:
            self._send_json(
                200,
                build_models_payload(router.models(), router.running_model_id()),
            )
            return

        body = self._read_body()
        if body is None:
            self.send_error(413, "request body too large")
            return

        # Audio is answered from LOCITIZE's own speech services, never from a
        # model: whisper and Kokoro run beside the LLM on their own ports and
        # do not compete for the single model slot (M8.2). Checked BEFORE the
        # switching paths so an audio request can never trigger a model load.
        if method == "POST" and path in audio_api.TRANSCRIPTION_PATHS:
            self._handle_transcription(router, body)
            return
        if method == "POST" and path in audio_api.SPEECH_PATHS:
            self._handle_speech(router, body)
            return

        if method == "POST" and path in SWITCHING_PATHS:
            # Decided BEFORE the switch, applied after it: a model load can
            # take longer than the transcript window (measured 2026-09-03: a
            # 27B switch is 16s), and the turn was spoken either way.
            spoken = router.voice_turns_enabled and router.is_voice_turn(body)

            # Portal/tooling: refuse/redirect known-broken tool-protocol models
            # before any GPU switch, and let Laya steer the first tools-needed
            # turn so the chat model does not burn an empty/XML turn.
            if request_has_tools(body):
                requested = requested_model_from_body(body)
                original_requested = requested
                owui_builtins = request_has_owui_builtin_tools(body)
                if is_tool_broken_model(requested) and not owui_builtins:
                    # Agent Portal tooling: redirect off known-broken tool protocol.
                    fallback = pick_tool_capable_fallback(
                        router.models(),
                        running_model_id=router.running_model_id(),
                    )
                    if not fallback:
                        self._send_json(
                            422,
                            {
                                "error": {
                                    "message": (
                                        f"model {requested!r} is not tool-capable "
                                        "(obliterated/abliterated/heretic class); "
                                        "no proven tool-capable fallback is loaded"
                                    ),
                                    "type": "model_not_tool_capable",
                                    "remedy": (
                                        "pick a tool-capable model such as "
                                        "qwen3-coder or gpt-oss"
                                    ),
                                }
                            },
                        )
                        return
                    body = rewrite_body_model(body, fallback)
                    requested = fallback
                elif is_tool_broken_model(requested) and owui_builtins:
                    # OWUI chat features (generate_image etc.): do NOT Laya-steer.
                    # If a tool-capable model is already on GPU, retarget to it.
                    # If the GPU still holds a tool-broken row, leave the request
                    # alone — forcing a swap 503s; OWUI should pick a tool-capable model
                    # (or use /images/generations) rather than hang on laya_route_tools.
                    running = router.running_model_id()
                    if running and not is_tool_broken_model(running):
                        body = rewrite_body_model(body, running)
                        requested = running
                # Gate on the ORIGINAL ask/model: a redirected broken model still
                # earns a Laya-first turn even when the fallback id is clean.
                if should_pre_steer_laya(body, original_requested):
                    utterance = last_user_utterance(body) or ""
                    task = _portal_task_text(utterance) or utterance
                    model_for_reply = (
                        requested_model_from_body(body)
                        or router.running_model_id()
                        or "laya-steer"
                    )
                    self._send_json(
                        200,
                        build_laya_steer_completion(
                            model_id=model_for_reply,
                            task=task,
                        ),
                    )
                    return

            ok, body = self._ensure_model_for(router, body)
            if not ok:
                return  # _ensure_model_for already answered
            # After the switch, so the name is the model that will actually
            # answer rather than the one that was loaded when the request came.
            body = apply_model_identity(body, router.running_model_name())
            # Typed OWUI chats need thinking off too (reasoning-model empty-content stuck).
            body = apply_thinking_defaults(body)
            if spoken:
                body = apply_voice_turn(body)

        self._forward(method, body)

    def _ensure_model_for(self, router: Any, body: bytes) -> tuple[bool, bytes]:
        """Switch to the model the body names, if it is not already running.

        Returns (True, body) with the body to forward - possibly rewritten to
        carry the handoff note. On any refusal this method has already written
        the response, so the caller must return.
        """
        requested = requested_model_from_body(body)
        if not requested:
            # No model named: serve whatever is already running, exactly as a
            # direct llama-server connection would.
            return True, body

        models = router.models()
        model_id = resolve_requested_model(requested, models)
        if model_id is None:
            self._send_json(
                404,
                {
                    "error": {
                        "message": (
                            f"no model {requested!r} in this LOCITIZE registry"
                        ),
                        "type": "model_not_found",
                        "available": [m.id for m in models],
                    }
                },
            )
            return False, b""

        previous_id = router.running_model_id()
        if previous_id == model_id:
            return True, body

        ok, reason = router.switch(model_id)
        if not ok:
            self._send_json(
                503,
                {
                    "error": {
                        "message": f"could not start {model_id}: {reason}",
                        "type": "model_start_failed",
                        "remedy": (
                            "check logs/ for a VRAM or load error, free VRAM in "
                            "the LOCITIZE command center, then retry"
                        ),
                    }
                },
            )
            return False, b""
        # The conversation is changing hands. Tell the incoming model that the
        # earlier assistant turns are not its own words (see build_handoff_note
        # for the reproduction). Done AFTER the switch succeeded, so a refused
        # switch never alters the request that gets forwarded.
        if router.handoff_note_enabled and previous_id:
            previous_name = router.display_name(previous_id)
            body = apply_handoff_note(body, previous_name)
        return True, body

    # ---- audio (owner request 2026-09-03) -------------------------------- #

    def _handle_transcription(self, router: Any, body: bytes) -> None:
        """POST /v1/audio/transcriptions -> whisper-server /inference.

        Open WebUI's microphone button lands here. The reply is already the
        shape it wants ({"text": ...}), so the translation is entirely on the
        request side: pull the upload out of the browser's multipart body and
        re-frame it the way whisper.cpp's handler reads.
        """
        port = router.whisper_port()
        if not port:
            self._send_json(503, {"error": {
                "message": "the speech-to-text service is not running",
                "type": "stt_unavailable",
                "remedy": "start whisper in the LOCITIZE window, or set "
                          "router.audio true so it starts with the session",
            }})
            return

        upload = audio_api.extract_upload(body, self.headers.get("Content-Type", ""))
        if upload is None:
            self._send_json(400, {"error": {
                "message": "no audio file in the request",
                "type": "invalid_request_error",
            }})
            return
        audio, filename = upload
        filtered = router.filter_upload(audio, filename)
        if filtered.error:
            category = filtered.error.split(":", 1)[0]
            if category == "unavailable":
                status = 503
                error_type = "noise_suppression_unavailable"
                public_message = (
                    "Local noise suppression is unavailable. Check FFmpeg or set "
                    "speech.noise_suppression to off."
                )
            elif category == "timeout":
                status = 504
                error_type = "audio_processing_timeout"
                public_message = "Local noise suppression timed out."
            elif category == "unexpected":
                status = 502
                error_type = "audio_processing_failed"
                public_message = "The audio could not be processed."
            else:
                status = 422
                error_type = "audio_processing_failed"
                public_message = "The audio could not be processed."
            self._send_json(status, {"error": {
                # Processor diagnostics may contain host paths. Keep the public
                # contract useful but fixed so local details never cross HTTP.
                "message": public_message,
                "type": error_type,
            }})
            return
        payload, content_type = audio_api.build_whisper_upload(
            filtered.audio, filtered.filename
        )

        status, raw = self._post_to(port, audio_api.WHISPER_PATH, payload, content_type)
        if status is None:
            self._send_json(502, {"error": {
                "message": f"speech-to-text service not reachable: {raw}",
                "type": "stt_unreachable",
            }})
            return
        text, problem = audio_api.transcript_from_whisper(raw)
        if text is None:
            self._send_json(502, {"error": {
                "message": problem, "type": "stt_bad_response",
            }})
            return
        router.record_transcript(text)

        fmt = audio_api.requested_response_format(
            body, self.headers.get("Content-Type", "")
        )
        out, out_type = audio_api.transcription_body(text, fmt)
        self.send_response(200)
        self.send_header("Content-Type", out_type)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _handle_speech(self, router: Any, body: bytes) -> None:
        """POST /v1/audio/speech -> kokoro_server /synthesize.

        Open WebUI's read-aloud button lands here. Kokoro answers audio/wav,
        which is one of the formats OpenAI's API can return, so the response is
        relayed as-is; the translation is the field renaming plus resolving a
        voice Kokoro actually has.
        """
        port = router.kokoro_port()
        if not port:
            self._send_json(503, {"error": {
                "message": "the text-to-speech service is not running",
                "type": "tts_unavailable",
                "remedy": "start Kokoro in the LOCITIZE window, or set "
                          "router.audio true so it starts with the session",
            }})
            return

        payload, problem = audio_api.speech_to_kokoro(
            body, router.kokoro_voices(), router.default_voice()
        )
        if payload is None:
            self._send_json(400, {"error": {
                "message": problem, "type": "invalid_request_error",
            }})
            return

        status, raw = self._post_to(
            port, audio_api.KOKORO_PATH,
            json.dumps(payload).encode("utf-8"), "application/json",
        )
        if status is None:
            self._send_json(502, {"error": {
                "message": f"text-to-speech service not reachable: {raw}",
                "type": "tts_unreachable",
            }})
            return
        if status != 200:
            # Kokoro reports its own failures as JSON; relay rather than
            # inventing a reason.
            self._send_json(502, {"error": {
                "message": raw.decode("utf-8", "replace")[:300],
                "type": "tts_failed",
            }})
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _post_to(
        self, port: int, path: str, body: bytes, content_type: str
    ) -> tuple[int | None, bytes]:
        """POST to a loopback service. Returns (status, body) or (None, reason).

        Its own small helper rather than _forward, because these are not proxied
        requests: the body was rebuilt, the target path differs from the one the
        client asked for, and the reply is re-framed. Reusing the transparent
        forwarder would have meant lying about what it does.
        """
        import http.client

        conn = http.client.HTTPConnection(_UPSTREAM_HOST, port, timeout=300)
        try:
            conn.request(
                "POST", path, body=body,
                headers={"Content-Type": content_type,
                         "Content-Length": str(len(body))},
            )
            response = conn.getresponse()
            return response.status, response.read()
        except (OSError, http.client.HTTPException) as exc:
            return None, str(exc).encode("utf-8", "replace")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # ---- transport -------------------------------------------------------- #

    def _forward(self, method: str, body: bytes) -> None:
        import http.client

        router = self.server.router  # type: ignore[attr-defined]
        target = build_upstream_target(router.port())
        if target is None:
            self._send_json(
                503,
                {
                    "error": {
                        "message": "no model is running",
                        "type": "no_model_running",
                        "remedy": (
                            "pick a model in the chat UI, or start one in the "
                            "LOCITIZE command center"
                        ),
                    }
                },
            )
            return
        host, port = target

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in ("host", "connection", "proxy-connection")
        }
        headers["Host"] = f"{host}:{port}"
        # The client's Content-Length describes the body IT sent. The handoff
        # note makes the forwarded body longer, and a stale length would either
        # truncate the JSON upstream or hang waiting for bytes that never come.
        headers["Content-Length"] = str(len(body))

        upstream = http.client.HTTPConnection(host, port, timeout=300)
        try:
            upstream.request(method, self.path, body=body, headers=headers)
            response = upstream.getresponse()
            # Do NOT follow redirects: relay the 3xx and let the client decide
            # (the same SEC-M3-1 rule proxy.py follows).
            # Content-Type decides whether we may rewrite the body. Promote
            # (reasoning_content -> content) can GROW JSON / SSE bytes, so we
            # must not copy a stale upstream Content-Length before rewrite.
            # 2026-09-24: portal saw truncated mid-timings JSON
            # (~900B, ends_ok=False) on tools+tool_choice=auto turns.
            content_type = (response.getheader("Content-Type") or "").lower()
            is_sse = "text/event-stream" in content_type
            is_json = (not is_sse) and (
                "application/json" in content_type
                or self.path.rstrip("/").endswith("/chat/completions")
            )
            if is_json:
                # Read + promote BEFORE headers so Content-Length matches the
                # bytes we actually write.
                raw = promote_completion_bytes(response.read())
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() in (
                        "connection",
                        "transfer-encoding",
                        "content-length",
                    ):
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                self.wfile.flush()
            else:
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    # SSE promote_sse_chunk can also grow frames; drop CL so
                    # a rewritten stream is never truncated by a stale length.
                    skip = {"connection", "transfer-encoding"}
                    if is_sse:
                        skip.add("content-length")
                    if key.lower() in skip:
                        continue
                    self.send_header(key, value)
                self.end_headers()
                # Relay so llama.cpp's SSE token streaming survives. read1, not
                # read: on a chunked response read(n) BLOCKS until n bytes have
                # accumulated, and one token event is ~260 bytes, so read(8192)
                # released the answer in ~30-token bursts - a short reply arrived
                # all at once after a silence (owner-observed 2026-09-03: "my
                # models are not streaming words, they are blasting them").
                # read1 returns what the upstream has sent so far, one chunk at a
                # time. Measured: 87 writes, first at 0.03s, against 3 at 0.40s.
                carry = b""
                while True:
                    chunk = response.read1(8192)
                    if not chunk:
                        if carry:
                            self.wfile.write(
                                promote_sse_chunk(carry) if is_sse else carry
                            )
                            self.wfile.flush()
                        break
                    data = carry + chunk
                    if not is_sse:
                        self.wfile.write(data)
                        self.wfile.flush()
                        carry = b""
                        continue
                    if data.endswith((b"\n\n", b"\r\n\r\n")):
                        self.wfile.write(promote_sse_chunk(data))
                        self.wfile.flush()
                        carry = b""
                        continue
                    idx_n = data.rfind(b"\n\n")
                    idx_r = data.rfind(b"\r\n\r\n")
                    if idx_r > idx_n:
                        complete, carry = data[: idx_r + 4], data[idx_r + 4 :]
                        self.wfile.write(promote_sse_chunk(complete))
                        self.wfile.flush()
                    elif idx_n >= 0:
                        complete, carry = data[: idx_n + 2], data[idx_n + 2 :]
                        self.wfile.write(promote_sse_chunk(complete))
                        self.wfile.flush()
                    else:
                        carry = data
        except (OSError, http.client.HTTPException):
            self.send_error(502, "upstream model not reachable")
        finally:
            try:
                upstream.close()
            except OSError:
                pass

    def _read_body(self) -> bytes | None:
        """Read the request body. None when it exceeds the size guard.

        Handles BOTH framings. Owner-observed 2026-09-03: Open WebUI's voice
        mode reported "Error transcribing chunk: 400 Bad Request" from the
        phone, because it streams recorded audio with Transfer-Encoding:
        chunked and no Content-Length - aiohttp does that whenever the payload
        size is not known upfront, which for a live recording it never is.
        Reading only Content-Length yielded an empty body, so the upload
        "contained no audio file" and every utterance was rejected.
        """
        encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in encoding:
            return self._read_chunked_body()
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        try:
            size = int(length)
        except ValueError:
            return b""
        if size > _MAX_BODY_BYTES:
            return None
        return self.rfile.read(size) if size > 0 else b""

    def _read_chunked_body(self) -> bytes | None:
        """Reassemble a chunked body, honouring the same size guard.

        The size cap is enforced as chunks ARRIVE rather than afterwards, so a
        sender that never stops cannot make this buffer without limit - the
        whole point of the guard.
        """
        chunks: list[bytes] = []
        total = 0
        while True:
            line = self.rfile.readline(_MAX_CHUNK_HEADER_BYTES)
            if not line:
                return b"".join(chunks)  # connection ended; keep what arrived
            # A chunk header is the hex size, optionally followed by ";ext".
            try:
                size = int(line.split(b";", 1)[0].strip() or b"0", 16)
            except ValueError:
                return None
            if size == 0:
                # Consume the trailer section up to the closing blank line.
                while True:
                    trailer = self.rfile.readline(_MAX_CHUNK_HEADER_BYTES)
                    if not trailer or trailer in (b"\r\n", b"\n"):
                        break
                return b"".join(chunks)
            total += size
            if total > _MAX_BODY_BYTES:
                return None
            chunks.append(self.rfile.read(size))
            self.rfile.read(2)  # the CRLF that terminates each chunk

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ModelRouter:
    """In-process loopback router managed as a daemon thread.

    Not a ManagedProcess (that is for external binaries); this is Python code
    running inside whichever LOCITIZE process owns the ModelController, which is
    the only process that can legitimately switch models.

    The three callables are injected rather than imported so the decision path
    is testable headless:
      registry_fn        -> the launchable rows to advertise, read FRESH per
                            request so a registry reload is picked up without a
                            restart
      running_model_id_fn-> the id currently being served, or None
      port_provider      -> the resolved loopback port, or None
      switch_fn          -> (model_id) -> (ok, reason)
    """

    def __init__(
        self,
        port: int,
        registry_fn: Callable[[], list[Any]],
        running_model_id_fn: Callable[[], str | None],
        port_provider: Callable[[], int | None],
        switch_fn: Callable[[str], tuple[bool, str]],
        handoff_note: bool = True,
        audio: Any = None,
        voice_turns: bool = True,
    ) -> None:
        self._port = port
        self._registry_fn = registry_fn
        self._running_model_id_fn = running_model_id_fn
        self._port_provider = port_provider
        self._switch_fn = switch_fn
        # Whether a mid-conversation model change adds the handoff note. A
        # setting because it MODIFIES the owner's request: small and factual,
        # but anyone comparing models on identical prompts wants it off.
        self.handoff_note_enabled = bool(handoff_note)
        # Where the speech services are, and which voices exist. A mapping
        # rather than four more constructor arguments, and OPTIONAL: with no
        # audio config the /v1/audio/* routes answer an honest 503 instead of
        # failing to exist, which tells the owner what to turn on rather than
        # letting Open WebUI report a bare 404.
        self._audio = dict(audio or {})
        # Spoken turns (see apply_voice_turn): the transcripts this router
        # returned most recently, newest last, so a chat request can be
        # recognised as the one Open WebUI's Call overlay built from them. A
        # setting because it too MODIFIES the owner's request.
        self.voice_turns_enabled = bool(voice_turns)
        self._transcripts: deque[tuple[float, str]] = deque(maxlen=8)
        self._transcript_lock = threading.Lock()
        # One switch at a time. Without this, two tabs asking for two different
        # models would race a pair of multi-gigabyte loads onto one card.
        self._switch_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---- the handler's view of the world ---------------------------------- #

    def models(self) -> list[Any]:
        return list(self._registry_fn() or [])

    def running_model_id(self) -> str | None:
        return self._running_model_id_fn()

    def running_model_name(self) -> str | None:
        """Display name of the row that is loaded, or None when nothing is.

        Resolved through models(), which is read fresh, so a registry reload is
        picked up without a router restart - the same contract as the listing.
        A row with no name of its own answers by its id rather than nothing.
        """
        running = self.running_model_id()
        if not running:
            return None
        for model in self.models():
            if getattr(model, "id", None) == running:
                return getattr(model, "name", None) or running
        return running

    def port(self) -> int | None:
        return self._port_provider()

    def record_transcript(self, text: str, now: float | None = None) -> None:
        """Remember a transcript this router just returned (see is_voice_turn)."""
        utterance = normalize_utterance(text)
        if not utterance:
            return
        stamp = time.monotonic() if now is None else now
        with self._transcript_lock:
            self._transcripts.append((stamp, utterance))

    def is_voice_turn(self, body: bytes, now: float | None = None) -> bool:
        """True when the body's last user message is, verbatim, a transcript
        this router returned within VOICE_TURN_WINDOW_S."""
        utterance = last_user_utterance(body)
        if not utterance:
            return False
        stamp = time.monotonic() if now is None else now
        with self._transcript_lock:
            return any(
                text == utterance and stamp - when <= VOICE_TURN_WINDOW_S
                for when, text in self._transcripts
            )

    def whisper_port(self) -> int | None:
        """Port of a RUNNING whisper-server, or None. Read fresh per request
        so starting the service mid-session needs no router restart."""
        fn = self._audio.get("whisper_port")
        try:
            return fn() if callable(fn) else fn
        except Exception:  # noqa: BLE001 - an unavailable service is a 503
            return None

    def kokoro_port(self) -> int | None:
        """Port of a RUNNING kokoro server, or None (see whisper_port)."""
        fn = self._audio.get("kokoro_port")
        try:
            return fn() if callable(fn) else fn
        except Exception:  # noqa: BLE001
            return None

    def kokoro_voices(self) -> list[str]:
        """Voice names Kokoro actually has on disk (may be empty)."""
        fn = self._audio.get("voices")
        try:
            value = fn() if callable(fn) else fn
        except Exception:  # noqa: BLE001
            return []
        return [str(v) for v in (value or [])]

    def default_voice(self) -> str:
        """The owner's configured tts.voice, used when a client asks for a
        voice Kokoro does not have (Open WebUI defaults to "alloy")."""
        fn = self._audio.get("default_voice")
        try:
            value = fn() if callable(fn) else fn
        except Exception:  # noqa: BLE001
            return ""
        return str(value or "")

    def filter_upload(self, audio: bytes, filename: str) -> AudioFilterResult:
        """Apply the injected local processor without letting it kill a request.

        No callback means an explicitly unconfigured legacy router, which keeps
        the old byte-identical behavior. Normal launcher wiring always supplies
        the default-on processor from the loaded settings.
        """
        fn = self._audio.get("filter_upload")
        if not callable(fn):
            return AudioFilterResult(audio, filename, False, "off", 0.0)
        try:
            result = fn(audio, filename)
        except Exception:  # noqa: BLE001 - request boundary containment
            return AudioFilterResult(
                b"", "audio.wav", False, "unknown", 0.0,
                "unexpected: noise processor failed",
            )
        if not isinstance(result, AudioFilterResult):
            return AudioFilterResult(
                b"", "audio.wav", False, "unknown", 0.0,
                "unexpected: noise processor returned an invalid result",
            )
        return result

    def noise_suppression_mode(self) -> str:
        """Return the loaded diagnostic mode without exposing filter syntax."""
        fn = self._audio.get("noise_suppression_mode")
        try:
            value = fn() if callable(fn) else fn
        except Exception:  # noqa: BLE001 - diagnostics must remain safe
            return "off"
        return str(value or "off")

    def display_name(self, model_id: str) -> str:
        """The row's human name for `model_id`, falling back to the id itself.

        Used only to NAME the previous model in the handoff note, so a row that
        has vanished from the registry between the switch and the note degrades
        to its id rather than breaking the request.
        """
        for model in self.models():
            if model.id == model_id:
                return getattr(model, "name", "") or model_id
        return model_id

    def switch(self, model_id: str) -> tuple[bool, str]:
        """Serialised switch. Re-checks under the lock before doing the work.

        The re-check matters: two requests for the SAME model arriving together
        would otherwise both pass the caller's "is it already running" test and
        the second would pointlessly reload a model the first just started.
        """
        with self._switch_lock:
            if self._running_model_id_fn() == model_id:
                return True, "already running"
            try:
                return self._switch_fn(model_id)
            except Exception as exc:  # noqa: BLE001 - boundary guard
                return False, str(exc)

    @property
    def bind_port(self) -> int:
        return self._port

    def base_url(self) -> str:
        """The OpenAI-compatible base URL a client should be pointed at."""
        return f"http://{_UPSTREAM_HOST}:{self._port}/v1"

    # ---- lifecycle -------------------------------------------------------- #

    def start(self) -> None:
        """Bind 127.0.0.1:<port> and serve on a daemon thread."""
        server = ThreadingHTTPServer((_UPSTREAM_HOST, self._port), _RouterHandler)
        server.router = self  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="locitize-router", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and join the thread (idempotent)."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
