"""LLM chat-client tests (AC6, keyword 'llm_client', Architecture M7.2/M7.10).

Deterministic and offline: a fake opener stands in for urllib.request.urlopen, so
request framing to /v1/chat/completions and SSE `data:` parsing are proven against
the exact wire shapes captured live from llama-server at Build time -- with no real
server, no GPU. The malformed/empty paths prove the client raises LlmError and
never fabricates a reply.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from fakes import FakeHttpResponse
from llm import ChatMessage, LlamaCppClient, LlmError

# The real captured stream shape (llama-server b10037, 2026-07-19): an opening role
# frame with content:null, content deltas, a final frame with finish_reason, then
# the literal [DONE] line. Blank lines separate chunks (text/event-stream framing).
_REAL_STREAM = (
    b'data: {"choices":[{"finish_reason":null,"index":0,'
    b'"delta":{"role":"assistant","content":null}}]}\n'
    b"\n"
    b'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"Hello"}}]}\n'
    b"\n"
    b'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":" there"}}]}\n'
    b"\n"
    b'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}\n'
    b"\n"
    b"data: [DONE]\n"
)


class _CapturingOpener:
    """Fake urlopen: records the request and returns a scripted response per path."""

    def __init__(self, response_by_path: dict[str, object], raises: Exception | None = None):
        self._responses = response_by_path
        self._raises = raises
        self.requests: list[object] = []

    def __call__(self, request, timeout=None):  # noqa: ANN001 - urlopen signature
        self.requests.append(request)
        if self._raises is not None:
            raise self._raises
        # request.full_url is the composed URL; match on its path suffix.
        url = request.full_url
        for path, resp in self._responses.items():
            if url.endswith(path):
                return resp
        raise AssertionError(f"no scripted response for {url}")


def test_llm_client_builds_correct_post():
    """chat_stream POSTs messages + stream:true to /v1/chat/completions."""
    opener = _CapturingOpener({"/v1/chat/completions": FakeHttpResponse(_REAL_STREAM)})
    client = LlamaCppClient(port=8080, opener=opener)

    list(client.chat_stream([ChatMessage("user", "hi")]))

    req = opener.requests[0]
    assert req.full_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert req.get_method() == "POST"
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["stream"] is True
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_llm_client_parses_sse_content_deltas_only():
    """Only content deltas are yielded; role/reasoning/final frames are ignored."""
    opener = _CapturingOpener({"/v1/chat/completions": FakeHttpResponse(_REAL_STREAM)})
    client = LlamaCppClient(port=8080, opener=opener)

    deltas = list(client.chat_stream([ChatMessage("user", "hi")]))

    assert deltas == ["Hello", " there"]
    assert client.chat([ChatMessage("user", "hi")]) == "Hello there"


def test_llm_client_skips_reasoning_content():
    """A reasoning model's reasoning_content is never yielded as reply text."""
    stream = (
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n'
        b'data: {"choices":[{"delta":{"content":"PONG."}}]}\n'
        b"data: [DONE]\n"
    )
    opener = _CapturingOpener({"/v1/chat/completions": FakeHttpResponse(stream)})
    client = LlamaCppClient(port=8080, opener=opener)

    assert "".join(client.chat_stream([ChatMessage("user", "ping")])) == "PONG."


def test_llm_client_raises_on_transport_error():
    """A transport error becomes LlmError with a remedy, never a fabricated reply."""
    opener = _CapturingOpener({}, raises=urllib.error.URLError("connection refused"))
    client = LlamaCppClient(port=8080, opener=opener)

    with pytest.raises(LlmError) as excinfo:
        list(client.chat_stream([ChatMessage("user", "hi")]))
    assert excinfo.value.remedy


def test_llm_client_raises_on_empty_response():
    """A stream that carries no content at all is an honest empty-response error."""
    stream = b'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\ndata: [DONE]\n'
    opener = _CapturingOpener({"/v1/chat/completions": FakeHttpResponse(stream)})
    client = LlamaCppClient(port=8080, opener=opener)

    with pytest.raises(LlmError):
        list(client.chat_stream([ChatMessage("user", "hi")]))


def test_llm_client_count_tokens_reads_tokenize():
    """count_tokens returns the server's token count from /tokenize."""
    body = json.dumps({"tokens": [1, 2, 3, 4, 5, 6, 7]}).encode("utf-8")
    opener = _CapturingOpener({"/tokenize": FakeHttpResponse(body)})
    client = LlamaCppClient(port=8080, opener=opener)

    assert client.count_tokens("Hello world this is a test.") == 7


def test_llm_client_count_tokens_none_on_failure():
    """A transport error on /tokenize returns None so the caller uses its fallback."""
    opener = _CapturingOpener({}, raises=urllib.error.URLError("down"))
    client = LlamaCppClient(port=8080, opener=opener)

    assert client.count_tokens("some text") is None
