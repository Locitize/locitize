"""locitize's web search tool for local coding terminals (owner request 2026-10-20)."""
import io
import json
import socket

import pytest

import web_search_mcp as web

RESULTS_PAGE = """
<div class="result">
  <a rel="nofollow" class="result__a" href="https://example.org/a">First &amp; best</a>
  <a class="result__snippet" href="https://example.org/a">The <b>first</b> result.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.net%2Fb&amp;rut=x">Second</a>
  <a class="result__snippet" href="x">Wrapped link.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a" href="javascript:alert(1)">Not a page</a>
  <a class="result__snippet" href="y">Skipped.</a>
</div>
"""


def test_search_results_are_parsed_and_redirect_links_unwrapped():
    results = web.parse_search_results(RESULTS_PAGE, 10)
    assert results == [
        {"title": "First & best", "url": "https://example.org/a", "snippet": "The first result."},
        {"title": "Second", "url": "https://example.net/b", "snippet": "Wrapped link."},
    ]
    assert len(web.parse_search_results(RESULTS_PAGE, 1)) == 1


def _resolve_to(address):
    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port or 80))]
    return fake


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.1.1"])
def test_pages_on_this_computer_or_the_local_network_are_refused(monkeypatch, address):
    monkeypatch.setattr(web.socket, "getaddrinfo", _resolve_to(address))
    assert "local network" in web.refusal_for_url("http://router.example/admin")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "notaurl"])
def test_only_web_addresses_can_be_read(url):
    assert "Only http" in web.refusal_for_url(url)


def test_a_public_address_is_allowed(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo", _resolve_to("93.184.215.14"))
    assert web.refusal_for_url("https://example.com/") is None


def test_a_refused_page_comes_back_as_a_tool_error_not_a_crash(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo", _resolve_to("127.0.0.1"))
    answer = web.call_tool("fetch_page", {"url": "http://127.0.0.1:8080/props"})
    assert answer["isError"] is True
    assert "local network" in answer["content"][0]["text"]


def test_html_becomes_readable_text_without_scripts():
    page = "<html><head><title>T</title><script>evil()</script></head>" \
           "<body><h1>Heading</h1><p>One &amp; two</p><style>p{}</style></body></html>"
    text = web.html_to_text(page)
    assert "Heading" in text and "One & two" in text
    assert "evil" not in text and "p{}" not in text


def test_the_protocol_answers_initialize_list_and_call_and_ignores_notifications():
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "no_such_tool", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "made/up"},
    ]
    stdin = io.StringIO("\n".join(json.dumps(line) for line in lines) + "\nnot json\n")
    stdout = io.StringIO()
    web.serve(stdin, stdout)
    answers = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [a["id"] for a in answers] == [1, 2, 3, 4, None]
    assert answers[0]["result"]["capabilities"] == {"tools": {}}
    assert {t["name"] for t in answers[1]["result"]["tools"]} == {"web_search", "fetch_page"}
    assert answers[2]["result"]["isError"] is True
    assert answers[3]["error"]["code"] == -32601
    assert answers[4]["error"]["code"] == -32700


def test_an_empty_search_is_refused_before_any_request(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("no request should be made")
    monkeypatch.setattr(web.urllib.request, "urlopen", no_network)
    answer = web.call_tool("web_search", {"query": "  "})
    assert answer["isError"] is True
