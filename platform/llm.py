"""Real llama.cpp chat client for the LOCITIZE assistant loop (M7, Architecture M7.2).

This module talks to an ALREADY-RUNNING llama.cpp server (started and owned by the
existing services.ModelController; this client never spawns a process). It sends a
conversation as a clean OpenAI-style `messages` list to the server's
`/v1/chat/completions` route -- which applies the model's own chat template
server-side, so LOCITIZE never hand-rolls per-model prompt templating -- and streams
the reply back delta by delta.

Wire format (confirmed live at Build time against llama-server build b10037,
the llama-server binary named by paths.llama_cpp, model Qwen3-14B-Q5_0,
on 2026-07-19):

  POST /v1/chat/completions  {"messages": [...], "stream": true}
  -> a text/event-stream of lines, each either blank or `data: <json>`, ending
     with the literal line `data: [DONE]`. Every json chunk carries
     choices[0].delta, whose fields observed were:
       - {"role": "assistant", "content": null}     (the opening frame)
       - {"reasoning_content": "..."}                (thinking; NOT the reply)
       - {"content": "..."}                          (the visible reply text)
       - {} with choices[0].finish_reason set        (the final frame)
  The spoken/printed reply is the concatenation of the `content` deltas only;
  `reasoning_content` (a reasoning model's private thinking) is deliberately NOT
  yielded as reply text, so LOCITIZE never speaks the model's scratchpad.

Token counting for the M7.3 context budget uses the same server's `POST /tokenize`
({"content": "..."} -> {"tokens": [...]}), which is the accurate, model-correct
count; count_tokens returns None when the endpoint is unavailable so the caller can
fall back to a labeled chars/4 heuristic rather than guessing silently.

Honest failure only: a transport error or an empty/malformed response raises
LlmError with a remedy. The client never fabricates a reply. All HTTP uses the
standard library (urllib) -- no `openai`/`sseclient` dependency is added.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Event
from typing import Any, Callable, Iterator, Sequence


@dataclass
class ChatMessage:
    """One turn of the conversation and one element of the request payload.

    role is "system", "user", or "assistant"; content is plain text. This is the
    Data Model 9.4 ChatMessage mirror -- the unit the assistant keeps in state and
    sends to /v1/chat/completions.
    """

    role: str
    content: str

    def as_payload(self) -> dict[str, str]:
        """Return the {role, content} mapping the chat endpoint expects."""
        return {"role": self.role, "content": self.content}


class LlmError(RuntimeError):
    """Honest LLM failure carrying an owner-facing remedy (Data Model 9.4).

    Raised on a transport error or an empty/malformed server response. The
    assistant surfaces the remedy to the owner; it never converts a failure into a
    fabricated reply.
    """

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.remedy = remedy


class LlmClient(ABC):
    """Injectable interface the assistant loop depends on (the M7.8 LLM seam).

    Concrete impl is LlamaCppClient; tests inject a fake that streams canned
    deltas, so the whole assistant orchestration is exercised with no real server.
    """

    @abstractmethod
    def chat_stream(
        self, messages: Sequence[ChatMessage], interrupt: Event | None = None
    ) -> Iterator[str]:
        """Yield reply text deltas as the model generates them."""

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Return the complete reply text (accumulates chat_stream)."""

    @abstractmethod
    def count_tokens(self, text: str) -> int | None:
        """Return the server's token count for text, or None if unavailable."""


class LlamaCppClient(LlmClient):
    """Real client against a running llama.cpp OpenAI-compatible server.

    The URL is composed from a fixed loopback host plus the ModelController-resolved
    port (SEC-1 discipline: LOCITIZE talks only to 127.0.0.1). `opener` is the single
    test seam -- an injectable urlopen-compatible callable -- so request framing and
    SSE parsing are unit-tested with no real server. When opener is None the stdlib
    urllib.request.urlopen is used.
    """

    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        timeout_s: float = 120.0,
        opener: Any = None,
    ) -> None:
        self._port = port
        self._host = host
        self._timeout_s = timeout_s
        self._opener = opener

    @property
    def base_url(self) -> str:
        """The loopback base URL this client posts to."""
        return f"http://{self._host}:{self._port}"

    def _open(self, request: Any) -> Any:
        """Open a request via the injected opener or the stdlib urlopen."""
        import urllib.request

        call = self._opener if self._opener is not None else urllib.request.urlopen
        return call(request, timeout=self._timeout_s)

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        """Build a JSON POST Request to `path` on this server and open it.

        Kept separate so both chat_stream and count_tokens share identical framing;
        transport errors are converted to LlmError by the callers that need it.
        """
        import urllib.request

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        return self._open(request)

    def chat_stream(
        self, messages: Sequence[ChatMessage], interrupt: Event | None = None
    ) -> Iterator[str]:
        """Stream the reply text deltas for `messages` from the running server.

        Yields only the `content` deltas (never `reasoning_content`), so a reasoning
        model's private thinking is never spoken or printed. When `interrupt` is set
        mid-stream, consumption stops and the underlying HTTP response is closed so
        llama-server stops generating (M7.5). Raises LlmError on a transport error
        or a stream that produced no content at all (never a fabricated reply).
        """
        import urllib.error

        payload = {
            "messages": [m.as_payload() for m in messages],
            "stream": True,
        }
        try:
            response = self._post("/v1/chat/completions", payload)
        except (urllib.error.URLError, OSError) as exc:
            raise LlmError(
                f"llama-server unreachable on {self._host}:{self._port}: {exc}",
                remedy="start a chat model (launcher menu or --smoke-start) and retry",
            ) from exc

        any_content = False
        try:
            for raw in response:
                if interrupt is not None and interrupt.is_set():
                    # Owner interrupted: stop consuming; the finally block closes the
                    # response so the server halts generation (no orphaned stream).
                    break
                delta = _parse_sse_delta(raw)
                if delta is None:
                    continue
                if delta == _DONE:
                    break
                if delta:
                    any_content = True
                    yield delta
        finally:
            _close(response)

        # An interrupted stream legitimately yields nothing; only a NON-interrupted
        # stream that produced zero content is a malformed/empty response.
        if not any_content and not (interrupt is not None and interrupt.is_set()):
            raise LlmError(
                "llama-server returned no reply content",
                remedy="check logs/ for a model-load or template error and retry",
            )

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Return the full reply by accumulating the streamed deltas."""
        return "".join(self.chat_stream(messages))

    def count_tokens(self, text: str) -> int | None:
        """Return the server's exact token count for text, or None on failure.

        Used by the M7.3 context-budget trimmer. A None return (endpoint missing or
        a transport error) tells the caller to fall back to the labeled chars/4
        heuristic rather than trusting a wrong number -- it is never fabricated.
        """
        import urllib.error

        if not text:
            return 0
        try:
            response = self._post("/tokenize", {"content": text})
            try:
                data = json.loads(response.read().decode("utf-8"))
            finally:
                _close(response)
        except (urllib.error.URLError, OSError, ValueError):
            return None
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if isinstance(tokens, list):
            return len(tokens)
        return None


# Sentinel yielded internally by the SSE parser to mark the terminal [DONE] line.
_DONE = "\x00__LOCITIZE_LLM_DONE__\x00"


def _parse_sse_delta(raw: bytes | str) -> str | None:
    """Turn one raw SSE line into a reply-content delta, or a control signal.

    Returns:
      - the sentinel _DONE when the line is `data: [DONE]`,
      - the content string when the chunk carries choices[0].delta.content,
      - "" when the chunk is valid but carries no reply content (role/reasoning/
        final frames), and
      - None when the line is blank or not a `data:` line (skip it).
    Kept as a module function so the parsing is unit-tested directly against the
    real wire shapes captured at Build time.
    """
    line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return _DONE
    try:
        obj = json.loads(data)
    except ValueError:
        # A malformed data line is skipped rather than crashing the stream; a
        # stream that is ENTIRELY malformed yields no content and is caught as an
        # empty response by chat_stream.
        return None
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _close(response: Any) -> None:
    """Best-effort close of an HTTP response / stream object (never raises)."""
    close: Callable[[], None] | None = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - closing must never mask the real result
            pass
