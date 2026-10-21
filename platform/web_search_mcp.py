"""Web search for coding tools that run on a local model (an MCP server on stdio).

Why this file exists
--------------------
Owner request 2026-10-20: "how do I make sure that it has access to the
internet while using the terminal? I know I have access to the internet if I
use OpenWebUI". Open WebUI has its own keyless DuckDuckGo search (webui.py).
Codex's and Claude Code's built-in web search run on OpenAI's and Anthropic's
servers, so they do nothing when a launch points them at llama-server. A tool
the coding tool runs on this machine works with any model, so every local
launch (session_launch.build_local_launch) registers this server.

Two tools:
- web_search: DuckDuckGo's keyless HTML results page, the same engine Open
  WebUI uses here (probed 2026-10-20: html.duckduckgo.com/html/ answers a
  plain POST with result__a / result__snippet markup).
- fetch_page: one public http(s) page as plain text. Addresses on this machine
  or the local network are refused, so a page the model reads cannot steer it
  into llama-server, Open WebUI or a router admin page.

Every request is recorded in the privacy ledger (egress_log) when the launch
passes --data-root, so "nothing leaves unless you asked" stays checkable.

Protocol: newline-delimited JSON-RPC 2.0 (MCP stdio transport). Stdlib only.
"""
from __future__ import annotations

import argparse
import html
import ipaddress
import json
import re
import socket
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

import egress_log

SERVER_NAME = "locitize_web"
PROTOCOL_VERSION = "2025-06-18"
SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) locitize-web-search"
TIMEOUT_S = 20
MAX_RESULTS = 10
# A local model's context is small: a whole page would crowd out the work.
DEFAULT_PAGE_CHARS = 8000
MAX_PAGE_CHARS = 40000
MAX_DOWNLOAD_BYTES = 2_000_000

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the internet (DuckDuckGo). Returns titles, links and short "
            "snippets. Use fetch_page to read a result."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "max_results": {
                    "type": "integer", "minimum": 1, "maximum": MAX_RESULTS,
                    "description": f"How many results (default 5, at most {MAX_RESULTS})",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": "Read one public web page as plain text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "An http:// or https:// address"},
                "max_chars": {
                    "type": "integer", "minimum": 500, "maximum": MAX_PAGE_CHARS,
                    "description": f"Most characters to return (default {DEFAULT_PAGE_CHARS})",
                },
            },
            "required": ["url"],
        },
    },
]


class ToolError(Exception):
    """A failure the model should read as the tool's answer."""


def _clean(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _result_link(href: str) -> str:
    # Some result pages wrap links as //duckduckgo.com/l/?uddg=<target>.
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com") and parsed.path == "/l/":
        target = urllib.parse.parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


def parse_search_results(page: str, limit: int) -> list[dict[str, str]]:
    """Pull title, url and snippet out of DuckDuckGo's HTML results page."""
    titles = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', page, re.S
    )
    snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    results = []
    for index, (href, title) in enumerate(titles):
        url = _result_link(href)
        if not url.startswith(("http://", "https://")):
            continue
        snippet = _clean(snippets[index]) if index < len(snippets) else ""
        results.append({"title": _clean(title), "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def _open(request: urllib.request.Request, reason: str):
    with egress_log.reason(reason):
        egress_log.record(request.full_url)
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)


def web_search(query: str, max_results: int = 5) -> str:
    query = str(query or "").strip()
    if not query:
        raise ToolError("Give a search query.")
    limit = max(1, min(MAX_RESULTS, int(max_results or 5)))
    request = urllib.request.Request(
        SEARCH_URL,
        data=urllib.parse.urlencode({"q": query}).encode("utf-8"),
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with _open(request, "coding tool web search") as response:
            page = response.read(MAX_DOWNLOAD_BYTES).decode("utf-8", "replace")
    except OSError as exc:
        raise ToolError(f"The search did not answer: {exc}") from exc
    results = parse_search_results(page, limit)
    if not results:
        return f"No results for {query!r}."
    lines = []
    for number, item in enumerate(results, 1):
        lines.append(f"{number}. {item['title']}\n   {item['url']}")
        if item["snippet"]:
            lines.append(f"   {item['snippet']}")
    return "\n".join(lines)


def refusal_for_url(url: str) -> str | None:
    """Why `url` must not be fetched, or None when it is a public web address."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "Only http:// and https:// web addresses can be read."
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or None)
    except OSError:
        return f"{parsed.hostname} could not be found."
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            return "Addresses on this computer or the local network cannot be read."
    return None


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head", "template"}
    _BREAK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "pre", "section", "article", "table", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)


def html_to_text(page: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(page)
    text = "".join(extractor.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    # A public page must not redirect the fetch onto a local address.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        refusal = refusal_for_url(newurl)
        if refusal:
            raise ToolError(f"The page redirected somewhere that cannot be read. {refusal}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_page(url: str, max_chars: int = DEFAULT_PAGE_CHARS) -> str:
    url = str(url or "").strip()
    refusal = refusal_for_url(url)
    if refusal:
        raise ToolError(refusal)
    limit = max(500, min(MAX_PAGE_CHARS, int(max_chars or DEFAULT_PAGE_CHARS)))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(_CheckedRedirects)
    try:
        with egress_log.reason("coding tool page read"):
            egress_log.record(url)
        with opener.open(request, timeout=TIMEOUT_S) as response:
            kind = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read(MAX_DOWNLOAD_BYTES).decode(charset, "replace")
    except ToolError:
        raise
    except (OSError, LookupError) as exc:
        raise ToolError(f"The page could not be read: {exc}") from exc
    text = html_to_text(body) if kind in ("text/html", "application/xhtml+xml") else body
    if len(text) > limit:
        text = text[:limit] + f"\n\n[cut at {limit} characters; ask for more with max_chars]"
    return text or "The page has no readable text."


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        if name == "web_search":
            text = web_search(arguments.get("query", ""), arguments.get("max_results", 5))
        elif name == "fetch_page":
            text = fetch_page(arguments.get("url", ""),
                              arguments.get("max_chars", DEFAULT_PAGE_CHARS))
        else:
            raise ToolError(f"There is no tool called {name}.")
    except (ToolError, ValueError, TypeError) as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """The JSON-RPC answer to one message, or None for a notification."""
    method = message.get("method")
    if "id" not in message:
        return None
    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
    params = message.get("params") or {}
    if method == "initialize":
        reply["result"] = {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": "1"},
        }
    elif method == "tools/list":
        reply["result"] = {"tools": TOOLS}
    elif method == "tools/call":
        reply["result"] = call_tool(str(params.get("name", "")), params.get("arguments") or {})
    elif method == "ping":
        reply["result"] = {}
    else:
        reply["error"] = {"code": -32601, "message": f"Unknown method {method}"}
    return reply


def serve(stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            answer: dict[str, Any] | None = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "Not valid JSON"},
            }
        else:
            answer = handle(message) if isinstance(message, dict) else None
        if answer is not None:
            stdout.write(json.dumps(answer, ensure_ascii=True) + "\n")
            stdout.flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="", help="locitize data folder, for the privacy ledger")
    args = parser.parse_args(argv)
    if args.data_root:
        egress_log.configure(args.data_root)
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    serve()


if __name__ == "__main__":
    main()
