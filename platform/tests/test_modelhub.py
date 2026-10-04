"""Tests for modelhub.py - the model acquisition core (AC-M14-18..22, 24).

Every test here is headless and offline. The only "network" is a fake opener,
which is the point: the suite proves egress rule HF-1 by handing modelhub an
opener that raises the moment anything opens it, then asserting the raise never
happens except where a user press is being simulated.

FIXTURE PROVENANCE (the U2 obligation, discharged 2026-08-19)
-------------------------------------------------------------
tests/fixtures/hf_*.json are VERBATIM recordings of live HuggingFace responses
made on 2026-08-19 from this repository, not hand-written approximations:

  hf_search_response.json          GET /api/models?search=qwen3%20gguf&filter=gguf
                                   &sort=downloads&direction=-1&limit=5  -> 200
  hf_tree_response.json            GET /api/models/unsloth/Qwen3.8-27B-GGUF
                                   /tree/main?recursive=true             -> 200
  hf_resolve_redirect_headers.json HEAD .../resolve/main/
                                   Qwen3.8-27B-IQ4_NL.gguf, redirects not
                                   followed                              -> 302
  hf_resolve_final_headers.json    the same HEAD, redirects followed      -> 200
  hf_gated_or_absent_401.json      GET a repository an anonymous caller may not
                                   read                                  -> 401

What those recordings settled, all three of which are asserted below:

  * Rung V-API is real: a GGUF entry's `lfs.oid` is a 64-hex sha256.
  * Rung V-ETAG is real but ONLY on the 302 hop. `X-Linked-ETag` does not appear
    on the final CDN response, and its value equals the same file's `lfs.oid`.
  * The plain `etag` on the final CDN response is the Xet content hash - a
    DIFFERENT 64-hex value. Accepting a bare ETag would compare the file against
    the wrong digest while telling the user it was verified. It is rejected.

One recorded fact worth naming because it shapes a user-facing message: an
anonymous request for a repository you cannot read returns 401 "Invalid username
or password." whether the repository is gated or simply does not exist. LOCITIZE
therefore does not claim to know which.
"""

import email.message
import hashlib
import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import modelhub

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The one GGUF used across the checksum tests. Both values are copied out of the
# recorded fixtures, not invented.
REAL_FILENAME = "Qwen3.8-27B-IQ4_NL.gguf"
REAL_SHA256 = "466c6714b0eca21c032690c801391a3c1e8f464ef01bbf420b70840027590c38"
REAL_XET_HASH = "4c86fb6ff254d074f281ae9793cfe9da692263a486716d51e865a47981199be2"


def load_fixture(name):
    """Read one recorded HuggingFace response."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class ExplodingOpener:
    """An opener that fails the test if anything ever opens it.

    This is the mechanical form of egress rule HF-1: hand one of these to a
    Downloader and any unexpected network call becomes a loud test failure
    instead of a quiet packet.
    """

    def __init__(self):
        self.calls = 0

    def open(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError(
            "modelhub made a network call outside an explicit user action"
        )


def http_headers(mapping):
    """Build a CASE-INSENSITIVE header map, the way a real response carries one.

    Fidelity that matters (review MEDIUM-5): a real HTTPResponse.headers is an
    email.message.Message, so `headers["ETag"]` finds a header the server sent as
    `etag`. A plain dict does not - and with a plain dict here, code that wrongly
    read the final response's ETag would be INVISIBLE to this suite, because the
    recorded fixture spells the header in lower case. Message also returns None
    for a missing key instead of raising, exactly like the real object.
    """
    message = email.message.Message()
    for key, value in (mapping or {}).items():
        message[key] = str(value)
    return message


class FakeResponse:
    """Minimal stand-in for an http.client.HTTPResponse."""

    def __init__(self, body=b"", headers=None, status=200, url=""):
        self._body = body
        self._pos = 0
        self.headers = http_headers(headers)
        self.status = status
        self.url = url

    def read(self, size=-1):
        if size is None or size < 0:
            chunk, self._pos = self._body[self._pos:], len(self._body)
            return chunk
        chunk = self._body[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class CountingOpener:
    """An opener that counts calls and replays a scripted list of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.urls = []

    def open(self, request, timeout=None):
        self.calls += 1
        self.urls.append(getattr(request, "full_url", request))
        item = self._responses.pop(0) if self._responses else FakeResponse()
        if isinstance(item, Exception):
            raise item
        return item


class RedirectingOpener:
    """An opener that drives REAL redirect hops through the REAL handler.

    Why this exists rather than another header-dict assertion: the sha256
    HuggingFace publishes is visible only while a 302 is being followed, so the
    only way to test the checksum ladder on the path the product actually runs
    is to make the handler observe hops. This opener replays `hop_count` hops -
    handing the handler the header map it is given, which the tests fill from the
    recorded fixtures - and then serves the body, exactly as urllib would.
    """

    def __init__(self, handler, body, hop_headers=None, hop_count=1,
                 final_headers=None):
        self.handler = handler
        self.body = body
        self.hop_headers = dict(hop_headers or {})
        self.hop_count = hop_count
        self.calls = 0
        # One entry per request: how many hops the handler agreed to follow.
        self.hops_allowed = []
        # Headers the FINAL response carries. Previously the final response was
        # bare, so nothing in the suite ever handed the recorded Xet `etag` to
        # the object the download actually reads (review MEDIUM-5) - a rule that
        # could not be exercised is a rule that is not tested.
        self.final_headers = dict(final_headers or {})
        # Header maps actually served, so a test can prove what it exercised
        # instead of assuming it.
        self.final_served = []

    def open(self, target, timeout=None):
        self.calls += 1
        url = getattr(target, "full_url", None) or str(target)
        request = urllib.request.Request(url)
        allowed = 0
        for index in range(self.hop_count):
            outcome = self.handler.redirect_request(
                request, None, 302, "Found",
                # A case-INSENSITIVE map, as urllib really hands the handler:
                # with a plain dict, code that read `ETag` would silently miss a
                # header recorded as `etag` and look safe (review MEDIUM-5).
                http_headers(self.hop_headers),
                f"https://us.aws.cdn.hf.co/hop{index}",
            )
            if outcome is None:
                break
            allowed += 1
        self.hops_allowed.append(allowed)
        # Content-Length is the ONE recorded header that cannot be replayed
        # verbatim: the recording is the 16 GB real file and this body is a few
        # bytes. It is dropped case-insensitively first, because a header map is
        # case-insensitive and leaving the recorded `content-length` in place
        # would shadow the replacement. Everything else (including the Xet
        # `etag`) is served exactly as recorded.
        headers = {
            key: value for key, value in self.final_headers.items()
            if key.lower() != "content-length"
        }
        headers["Content-Length"] = str(len(self.body))
        self.final_served.append(headers)
        return FakeResponse(self.body, headers=headers)


class FakeGpu:
    """Stands in for health.GpuInfo without importing the real provider."""

    def __init__(self, name, vram_total_mb):
        self.name = name
        self.vram_total_mb = float(vram_total_mb)
        self.vram_free_mb = float(vram_total_mb)


class FakeGpuProvider:
    """Stands in for health.GpuInfoProvider - the ONE GPU fact source modelhub uses."""

    def __init__(self, gpus):
        self._gpus = gpus
        self.calls = 0

    def gpus(self):
        self.calls += 1
        return self._gpus


class FakeSystemProvider:
    """Stands in for health.SystemInfoProvider's disk_free_mb."""

    def __init__(self, free_mb):
        self._free_mb = free_mb

    def disk_free_mb(self, path):
        return self._free_mb


def make_downloader(opener, **kwargs):
    """Build a Downloader whose only network seam is `opener`."""
    config = kwargs.pop("config", None) or modelhub.HubConfig()
    return modelhub.Downloader(
        config, opener_factory=lambda hosts: (opener, None), **kwargs
    )


def gguf_bytes(payload=b"weights"):
    """A byte string that passes the GGUF magic check."""
    return modelhub.GGUF_MAGIC + payload


# =========================================================================== #
# AC-M14-18: modelhub_egress - network calls happen ONLY on explicit action
# =========================================================================== #


def test_modelhub_egress_import_makes_no_call():
    """Importing the module must not touch the network (nothing at startup)."""
    opener = ExplodingOpener()
    import importlib

    importlib.reload(modelhub)
    assert opener.calls == 0


def test_modelhub_egress_construction_makes_no_call():
    """Building a Downloader must not touch the network."""
    opener = ExplodingOpener()
    make_downloader(opener, models_dir="/nowhere")
    assert opener.calls == 0


def test_modelhub_egress_idle_page_makes_no_call(tmp_path):
    """The page-open path - catalog read plus fit estimate - makes no call.

    This is the state the Models page sits in from the moment it paints until
    the user presses something: catalog listed, fit estimated, no egress.
    """
    opener = ExplodingOpener()
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"models": [{"repo_id": "a/b", "display_name": "B", "files": []}]}),
        encoding="utf-8",
    )
    hub = make_downloader(
        opener,
        config=modelhub.HubConfig(catalog_path=str(catalog)),
        gpu_provider=FakeGpuProvider([FakeGpu("RTX 4080", 16384)]),
    )
    assert hub.catalog()["items"][0]["repo_id"] == "a/b"
    assert hub.fit_for(4_000_000_000)["band"] in ("fits", "tight", "exceeds")
    # Simulate keystrokes and repeated repaints, which must also be silent.
    for _ in range(20):
        hub.catalog()
        hub.fit_for(1_000_000)
    assert opener.calls == 0


def test_modelhub_egress_search_makes_exactly_one_call():
    """One explicit Search press produces exactly one network call."""
    opener = CountingOpener([FakeResponse(json.dumps([]).encode())])
    hub = make_downloader(opener)
    hub.search("qwen")
    assert opener.calls == 1


def _repo_entry(repo_id):
    return {"id": repo_id, "tags": [], "downloads": 1, "likes": 0}


NEXT_PAGE_URL = (
    "https://huggingface.co/api/models?search=qwen&filter=gguf"
    "&sort=downloads&direction=-1&limit=100&cursor=abc123"
)


def test_search_reports_has_more_when_a_next_link_is_present():
    """A Link: rel="next" header means there is a further page to load."""
    body = json.dumps([_repo_entry("unsloth/Qwen3-4B-GGUF")]).encode()
    headers = {"Link": f'<{NEXT_PAGE_URL}>; rel="next"'}
    opener = CountingOpener([FakeResponse(body, headers=headers)])
    hub = make_downloader(opener)
    result = hub.search("qwen")
    assert result["ok"] is True
    assert result["has_more"] is True
    assert len(result["items"]) == 1


def test_search_reports_no_more_on_the_last_page():
    """No Link header (or none with rel="next") means this was the last page."""
    body = json.dumps([_repo_entry("unsloth/Qwen3-4B-GGUF")]).encode()
    opener = CountingOpener([FakeResponse(body)])
    hub = make_downloader(opener)
    result = hub.search("qwen")
    assert result["has_more"] is False


def test_search_more_is_a_no_op_before_any_search():
    """Load more with no prior search (or no next page) makes no network call."""
    opener = CountingOpener([])
    hub = make_downloader(opener)
    result = hub.search_more()
    assert result["ok"] is True
    assert result["items"] == []
    assert result["has_more"] is False
    assert opener.calls == 0


def test_search_more_fetches_the_exact_cursor_url_and_appends_results():
    """Load more hits huggingface.co's own next-page cursor and merges results."""
    page1 = FakeResponse(
        json.dumps([_repo_entry("unsloth/Qwen3-4B-GGUF")]).encode(),
        headers={"Link": f'<{NEXT_PAGE_URL}>; rel="next"'},
    )
    page2 = FakeResponse(json.dumps([_repo_entry("bartowski/Qwen3-8B-GGUF")]).encode())
    opener = CountingOpener([page1, page2])
    hub = make_downloader(opener)
    hub.search("qwen")
    result = hub.search_more()
    assert opener.calls == 2
    assert opener.urls[1] == NEXT_PAGE_URL
    assert result["ok"] is True
    assert result["has_more"] is False
    assert [row["repo_id"] for row in result["items"]] == [
        "unsloth/Qwen3-4B-GGUF",
        "bartowski/Qwen3-8B-GGUF",
    ]


def test_search_more_reuses_the_original_query_in_its_result():
    """The query in a Load more result is the ORIGINAL search's, for the status line."""
    page1 = FakeResponse(
        json.dumps([]).encode(), headers={"Link": f'<{NEXT_PAGE_URL}>; rel="next"'}
    )
    page2 = FakeResponse(json.dumps([]).encode())
    hub = make_downloader(CountingOpener([page1, page2]))
    hub.search("qwen abliterated")
    result = hub.search_more()
    assert result["query"] == "qwen abliterated"


def test_modelhub_egress_list_files_makes_exactly_one_call():
    """One explicit repo selection produces exactly one network call."""
    opener = CountingOpener([FakeResponse(json.dumps([]).encode())])
    hub = make_downloader(opener)
    hub.list_files("unsloth/Qwen3.8-27B-GGUF")
    assert opener.calls == 1


def test_modelhub_egress_download_calls_only_when_pressed(tmp_path):
    """A download opens the network only when download() is actually invoked."""
    body = gguf_bytes(b"x" * 32)
    opener = CountingOpener(
        [FakeResponse(body, headers={"Content-Length": str(len(body))})]
    )
    hub = make_downloader(opener, models_dir=tmp_path)
    assert opener.calls == 0  # constructed, not pressed
    digest = __import__("hashlib").sha256(body).hexdigest()
    outcome = hub.download(
        "owner/repo", "m.gguf", expected_sha256=digest,
        verification=modelhub.V_API, size_bytes=len(body),
    )
    assert outcome.ok, outcome.error
    assert opener.calls == 1


def test_modelhub_egress_disabled_config_cannot_call():
    """models_hub.enabled false makes egress impossible, not merely hidden."""
    opener = ExplodingOpener()
    hub = make_downloader(opener, config=modelhub.HubConfig(enabled=False))
    assert hub.search("qwen")["ok"] is False
    assert hub.list_files("a/b")["ok"] is False
    assert hub.download("a/b", "m.gguf").ok is False
    assert opener.calls == 0


def test_modelhub_egress_off_allowlist_api_base_opens_nothing():
    """T2 on the FIRST request, not just on redirects (regression, review HIGH-1).

    `models_hub.api_base` is settable in the user's own settings.yaml (the
    environment override was removed by DEC-M14-10, so widening the egress
    boundary now takes two deliberate edits to a file the user owns), which
    still makes it mistake-reachable configuration. Every entry point must refuse it before a socket opens - the
    V-ETAG HEAD probe inside download() used to be the one that did not, and it
    leaked the repo id, file name and the user's IP to whatever host was named.
    """
    evil = modelhub.HubConfig(
        api_base="https://evil.example.com",
        allowed_hosts=("huggingface.co", "hf.co"),
    )
    for call in (
        lambda hub: hub.search("qwen"),
        lambda hub: hub.list_files("owner/repo"),
        lambda hub: hub.download("owner/repo", "m.gguf", confirm_unverified=True),
    ):
        opener = ExplodingOpener()
        hub = make_downloader(opener, config=evil, models_dir="/nowhere")
        try:
            result = call(hub)
        except modelhub.HubError as exc:
            result = exc  # search/list_files may surface it either way
        ok = getattr(result, "ok", None)
        if isinstance(result, dict):
            ok = result.get("ok")
        assert ok is not True, "an off-allowlist api_base must never succeed"
        assert opener.calls == 0, (
            "modelhub opened a connection to an off-allowlist host "
            "(ExplodingOpener was called)"
        )


def test_modelhub_egress_off_allowlist_api_base_never_even_builds_an_opener():
    """download() refuses before construction, so nothing can be opened at all.

    Stronger than counting opens: the factory itself is never called, which is
    what makes the refusal structural rather than a matter of the opener being
    polite.
    """
    built = []

    def factory(hosts):
        built.append(hosts)
        return ExplodingOpener(), None

    hub = modelhub.Downloader(
        modelhub.HubConfig(api_base="https://evil.example.com"),
        opener_factory=factory,
        models_dir="/nowhere",
    )
    outcome = hub.download("owner/repo", "m.gguf", confirm_unverified=True)
    assert outcome.ok is False
    assert "evil.example.com" in outcome.error
    assert built == [], "an opener was constructed for an off-allowlist host"


def test_modelhub_trust_open_checked_is_the_only_way_out_of_the_module():
    """Source guard: no call site may open a connection around the allowlist.

    This is the class-level fix for review HIGH-1. One bypass was a missing
    validate_url line at one call site; this test makes the next such line
    impossible to add unnoticed, by asserting `open_checked` (which validates
    then opens) holds the module's only `opener.open(` call.
    """
    source = Path(modelhub.__file__).read_text(encoding="utf-8")
    openings = [
        line.strip()
        for line in source.splitlines()
        if "opener.open(" in line and not line.strip().startswith("#")
    ]
    assert openings == ["return opener.open(target, timeout=timeout)"], (
        "modelhub opens a connection somewhere other than open_checked: "
        f"{openings}"
    )
    # And open_checked really does validate before it opens.
    with pytest.raises(modelhub.HubError):
        modelhub.open_checked(
            ExplodingOpener(),
            "https://evil.example.com/x.gguf",
            modelhub.DEFAULT_ALLOWED_HOSTS,
        )


# =========================================================================== #
# AC-M14-19: modelhub_checksum - the three-rung ladder, against real fixtures
# =========================================================================== #


def test_modelhub_checksum_v_api_reads_lfs_oid_from_the_real_tree_response():
    """V-API: the recorded tree response yields a real 64-hex sha256 per file."""
    rows = modelhub.parse_tree_files(load_fixture("hf_tree_response.json"))
    assert rows, "the recorded response contains top-level .gguf files"
    row = next(r for r in rows if r["filename"] == REAL_FILENAME)
    assert row["sha256"] == REAL_SHA256
    assert row["verification"] == modelhub.V_API
    assert modelhub.SHA256_RE.match(row["sha256"])


def test_modelhub_checksum_v_api_matches_the_real_x_linked_etag():
    """The two independent rungs agree on the same real file - a live cross-check."""
    tree_row = next(
        r for r in modelhub.parse_tree_files(load_fixture("hf_tree_response.json"))
        if r["filename"] == REAL_FILENAME
    )
    hop = load_fixture("hf_resolve_redirect_headers.json")
    linked = modelhub.normalize_etag_digest(hop["headers"]["X-Linked-ETag"])
    assert linked == tree_row["sha256"] == REAL_SHA256


def test_modelhub_checksum_v_api_mismatch_deletes_the_file(tmp_path):
    """A digest that does not match deletes the download and never renames it."""
    body = gguf_bytes(b"y" * 64)
    opener = CountingOpener(
        [FakeResponse(body, headers={"Content-Length": str(len(body))})]
    )
    dest = tmp_path / "m.gguf"
    wrong = "0" * 64
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf", dest, wrong, opener=opener
    )
    assert outcome.ok is False
    assert "checksum mismatch" in outcome.error
    assert wrong in outcome.error and outcome.sha256 in outcome.error
    assert not dest.exists()
    assert not dest.with_suffix(".gguf.partial").exists()


def test_modelhub_checksum_v_etag_accepts_only_exactly_64_hex():
    """V-ETAG's acceptance rule, exercised against the real header and near-misses."""
    hop = load_fixture("hf_resolve_redirect_headers.json")
    assert modelhub.normalize_etag_digest(hop["headers"]["X-Linked-ETag"]) == REAL_SHA256
    # Real quoted 64-hex accepted; everything else refused.
    assert modelhub.normalize_etag_digest('"' + REAL_SHA256 + '"') == REAL_SHA256
    assert modelhub.normalize_etag_digest(REAL_SHA256) == REAL_SHA256
    assert modelhub.normalize_etag_digest('W/"' + REAL_SHA256 + '"') is None
    assert modelhub.normalize_etag_digest('"abc123"') is None
    assert modelhub.normalize_etag_digest('"' + REAL_SHA256 + "aa" + '"') is None
    assert modelhub.normalize_etag_digest('"' + "z" * 64 + '"') is None
    assert modelhub.normalize_etag_digest(None) is None


def test_modelhub_checksum_plain_etag_is_never_used_as_a_digest():
    """The final CDN response's plain `etag` is the Xet hash, NOT the sha256.

    Recorded proof that leniency here would be a real, silent correctness bug:
    both values are 64 hex characters and they are different values.
    """
    final = load_fixture("hf_resolve_final_headers.json")
    plain = modelhub.normalize_etag_digest(final["headers"]["etag"])
    assert plain == REAL_XET_HASH
    assert plain != REAL_SHA256
    # And the header the ladder actually reads is absent from this response, so
    # a download that only looked at the final hop would find no digest at all.
    assert "X-Linked-ETag" not in final["headers"]
    assert "x-linked-etag" not in final["headers"]


def test_modelhub_checksum_v_etag_is_captured_from_the_redirect_hop():
    """The redirect handler records X-Linked-ETag as it passes the 302."""
    hop = load_fixture("hf_resolve_redirect_headers.json")
    handler = modelhub._AllowlistRedirectHandler(modelhub.DEFAULT_ALLOWED_HOSTS)
    handler._record_linked_headers(http_headers(hop["headers"]))
    assert handler.linked_etag == REAL_SHA256
    assert handler.linked_size == int(hop["headers"]["X-Linked-Size"])


def test_modelhub_checksum_a_file_with_no_lfs_oid_is_never_labelled_v_api():
    """No digest means rung V-NONE, never "verified against HuggingFace".

    Found by re-running the mutation harness: forcing this row to V_API left the
    whole suite green, so the rung's ONE precondition had no test. Small (under
    ~50 MB) files on HuggingFace are stored outside LFS and really do arrive
    without an `lfs.oid`, so this is a shape the product will meet.

    The entry is the RECORDED IQ4_NL row with its `lfs` block removed - a real
    row minus the field under test, not an invented one.
    """
    recorded = next(
        e for e in load_fixture("hf_tree_response.json")
        if e.get("path") == REAL_FILENAME
    )
    no_lfs = {k: v for k, v in recorded.items() if k != "lfs"}
    no_lfs["size"] = 1024
    rows = modelhub.parse_tree_files([no_lfs])
    assert len(rows) == 1
    assert rows[0]["sha256"] in (None, "")
    assert rows[0]["verification"] == modelhub.V_NONE
    assert modelhub.VERIFICATION_LABELS[rows[0]["verification"]] == (
        "Not verified - no publisher checksum available."
    )
    # The control: with the lfs block present the same row IS V-API.
    assert modelhub.parse_tree_files([recorded])[0]["verification"] == modelhub.V_API


def test_modelhub_checksum_recorder_ignores_a_plain_etag_on_the_real_path():
    """The recorder must read X-Linked-ETag and NOTHING else (review HIGH-2).

    Fed the RECORDED final-hop headers - whose plain `etag` is the 64-hex Xet
    content hash, not the sha256 - `_record_linked_headers` must come away with
    no digest at all. Widening it to fall back on `etag` makes this test fail,
    which is the whole point: the previous test only called the normaliser
    directly and so could not see the difference.

    The headers are handed over in a CASE-INSENSITIVE map, as urllib really
    hands them to a redirect handler. With a plain dict the fixture's lower-case
    `etag` was invisible to a `headers.get("ETag")` fallback, so the widening
    this test exists to forbid slipped through the whole suite untouched.
    """
    final = load_fixture("hf_resolve_final_headers.json")
    assert modelhub.normalize_etag_digest(final["headers"]["etag"]) == REAL_XET_HASH
    handler = modelhub._AllowlistRedirectHandler(modelhub.DEFAULT_ALLOWED_HOSTS)
    handler._record_linked_headers(http_headers(final["headers"]))
    assert handler.linked_etag is None, (
        "the plain ETag (the Xet content hash) was adopted as the file's sha256"
    )


def test_modelhub_checksum_x_linked_etag_wins_when_both_headers_are_present():
    """With both headers on one response, only the linked one is a digest."""
    handler = modelhub._AllowlistRedirectHandler(modelhub.DEFAULT_ALLOWED_HOSTS)
    handler._record_linked_headers(
        http_headers(
            {
                "X-Linked-ETag": '"' + REAL_SHA256 + '"',
                "etag": '"' + REAL_XET_HASH + '"',
            }
        )
    )
    assert handler.linked_etag == REAL_SHA256
    assert handler.linked_etag != REAL_XET_HASH


def test_modelhub_checksum_download_never_verifies_against_a_plain_etag(tmp_path):
    """End to end: a 302 carrying ONLY the Xet `etag` yields rung V-NONE.

    This drives download() -> the HEAD probe -> the real redirect handler ->
    rung selection with the recorded final-hop headers. If the recorder ever
    accepts a plain ETag, the download is compared against the Xet hash, fails
    with a checksum mismatch, and would otherwise have been labelled "verified
    against HuggingFace's linked file hash" - so this test turns that silent
    mislabelling into a red suite.

    The recorded headers are served BOTH on the 302 hop and on the final
    response (review MEDIUM-5). Serving them only on the hop left the other
    plausible mistake - reading the digest off the final response instead of the
    hop - completely untested, because the final response carried no `etag` at
    all for such code to find.
    """
    body = gguf_bytes(b"real bytes")
    handler = modelhub._AllowlistRedirectHandler(modelhub.DEFAULT_ALLOWED_HOSTS)
    final = load_fixture("hf_resolve_final_headers.json")
    opener = RedirectingOpener(
        handler, body,
        hop_headers=final["headers"],
        final_headers=final["headers"],
    )
    hub = modelhub.Downloader(
        modelhub.HubConfig(),
        opener_factory=lambda hosts: (opener, handler),
        models_dir=tmp_path,
    )
    outcome = hub.download("owner/repo", "m.gguf", confirm_unverified=True)
    assert outcome.ok is True, outcome.error
    # Prove the case under test was actually exercised: the final response the
    # download read really did carry a plain, well-formed, 64-hex ETag that is
    # NOT this file's sha256. Without this assertion the test could quietly stop
    # covering the rule the day the fixture or the double changes.
    assert opener.final_served, "no final response was served"
    for served in opener.final_served:
        assert modelhub.normalize_etag_digest(served["etag"]) == REAL_XET_HASH
        assert REAL_XET_HASH != REAL_SHA256
        assert "X-Linked-ETag" not in served and "x-linked-etag" not in served
    assert outcome.verification == modelhub.V_NONE
    assert outcome.expected_sha256 in ("", None)
    assert outcome.sha256 == hashlib.sha256(body).hexdigest()
    assert outcome.sha256 != REAL_XET_HASH
    # And the label the user sees makes no HuggingFace claim.
    label = modelhub.VERIFICATION_LABELS[outcome.verification]
    assert label == "Not verified - no publisher checksum available."


def test_modelhub_checksum_download_uses_x_linked_etag_when_it_is_present(tmp_path):
    """The positive control for the test above: a real hop digest reaches V-ETAG.

    Without this, "always end at V-NONE" would pass the trap test trivially.
    """
    body = gguf_bytes(b"real bytes")
    digest = hashlib.sha256(body).hexdigest()
    handler = modelhub._AllowlistRedirectHandler(modelhub.DEFAULT_ALLOWED_HOSTS)
    opener = RedirectingOpener(
        handler, body, hop_headers={"X-Linked-ETag": '"' + digest + '"'}
    )
    hub = modelhub.Downloader(
        modelhub.HubConfig(),
        opener_factory=lambda hosts: (opener, handler),
        models_dir=tmp_path,
    )
    # No confirm_unverified: reaching V-ETAG is what makes this legal.
    outcome = hub.download("owner/repo", "m.gguf")
    assert outcome.ok is True, outcome.error
    assert outcome.verification == modelhub.V_ETAG
    assert outcome.expected_sha256 == digest


def test_modelhub_checksum_v_none_requires_explicit_confirmation(tmp_path):
    """V-NONE: with no publisher digest, the download refuses without consent."""
    body = gguf_bytes()
    opener = CountingOpener([FakeResponse(body)])
    dest = tmp_path / "m.gguf"
    refused = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf", dest, None, opener=opener
    )
    assert refused.ok is False
    assert refused.verification == modelhub.V_NONE
    assert modelhub.UNVERIFIED_CONSENT_TEXT in refused.error
    assert not dest.exists()
    assert opener.calls == 0  # refused BEFORE any egress


def test_modelhub_checksum_v_none_proceeds_and_is_labelled_unverified(tmp_path):
    """With consent the file is kept, its digest recorded, and it stays unverified."""
    body = gguf_bytes(b"payload")
    opener = CountingOpener([FakeResponse(body)])
    dest = tmp_path / "m.gguf"
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf",
        dest,
        None,
        opener=opener,
        confirm_unverified=True,
    )
    assert outcome.ok is True
    assert dest.exists()
    assert outcome.verification == modelhub.V_NONE
    assert outcome.sha256 == __import__("hashlib").sha256(body).hexdigest()
    # The word "verified" must not appear in what the user is shown for V-NONE.
    label = modelhub.VERIFICATION_LABELS[modelhub.V_NONE]
    assert "verified" not in label.lower().replace("not verified", "")
    assert label.startswith("Not verified")


def test_modelhub_checksum_caller_cannot_upgrade_an_unverified_rung(tmp_path):
    """Passing verification='api' without a digest still records 'none'.

    The rung is a statement about what was compared, so it cannot be set by a
    caller's optimism.
    """
    opener = CountingOpener([FakeResponse(gguf_bytes())])
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf",
        tmp_path / "m.gguf",
        None,
        opener=opener,
        verification=modelhub.V_API,
        confirm_unverified=True,
    )
    assert outcome.ok is True
    assert outcome.verification == modelhub.V_NONE


def test_modelhub_checksum_v_none_registry_row_stays_unverified(tmp_path):
    """A V-NONE row re-read from models.yaml still reports verification=none."""
    import config

    models = tmp_path / "models.yaml"
    models.write_text("version: 1\nmodels: []\n", encoding="utf-8")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(gguf_bytes())
    digest = "a" * 64
    config.append_model_entry(
        tmp_path,
        "m",
        "M",
        str(gguf),
        notes=modelhub.registry_notes("o/r", "m.gguf", modelhub.V_NONE, "2026-08-19"),
        sha256=digest,
    )
    reread = config.Config.load(base_dir=tmp_path)[1]
    row = next(m for m in reread.models if m.id == "m")
    assert "verification=none" in row.notes
    assert row.sha256 == digest
    assert "verified" not in row.notes.lower()


# =========================================================================== #
# AC-M14-20: modelhub_integrity - magic bytes and size, on every rung
# =========================================================================== #


def test_modelhub_integrity_rejects_a_non_gguf_body(tmp_path):
    """An HTML sign-in page saved as .gguf is rejected and deleted."""
    body = b"<!DOCTYPE html><html><body>Sign in to continue</body></html>"
    opener = CountingOpener([FakeResponse(body)])
    dest = tmp_path / "m.gguf"
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf",
        dest,
        None,
        opener=opener,
        confirm_unverified=True,
    )
    assert outcome.ok is False
    assert "not a GGUF model" in outcome.error
    assert not dest.exists()
    assert not tmp_path.joinpath("m.gguf.partial").exists()


def test_modelhub_integrity_magic_check_runs_on_the_verified_rung_too(tmp_path):
    """The magic-byte check is not skipped just because a digest matched."""
    body = b"<html>not a model</html>"
    digest = __import__("hashlib").sha256(body).hexdigest()
    opener = CountingOpener([FakeResponse(body)])
    dest = tmp_path / "m.gguf"
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf", dest, digest, opener=opener,
        verification=modelhub.V_API,
    )
    assert outcome.ok is False
    assert "not a GGUF model" in outcome.error
    assert not dest.exists()


def test_modelhub_integrity_rejects_a_size_disagreement_before_downloading(tmp_path):
    """An API size that disagrees with Content-Length refuses before writing."""
    body = gguf_bytes(b"z" * 100)
    opener = CountingOpener(
        [FakeResponse(body, headers={"Content-Length": str(len(body))})]
    )
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf",
        tmp_path / "m.gguf",
        None,
        opener=opener,
        confirm_unverified=True,
        expected_size=len(body) + 1,
    )
    assert outcome.ok is False
    assert "size mismatch before download" in outcome.error
    assert not tmp_path.joinpath("m.gguf").exists()


def test_modelhub_integrity_rejects_a_truncated_transfer(tmp_path):
    """Fewer bytes than Content-Length promised is a refusal, not a warning."""
    body = gguf_bytes(b"short")
    opener = CountingOpener(
        [FakeResponse(body, headers={"Content-Length": str(len(body) + 500)})]
    )
    dest = tmp_path / "m.gguf"
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf", dest, None, opener=opener,
        confirm_unverified=True,
    )
    assert outcome.ok is False
    assert "size mismatch" in outcome.error
    assert not dest.exists()


def test_modelhub_integrity_rejects_an_empty_download(tmp_path):
    """A zero-byte body is never accepted as a model."""
    opener = CountingOpener([FakeResponse(b"")])
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf",
        tmp_path / "m.gguf",
        None,
        opener=opener,
        confirm_unverified=True,
    )
    assert outcome.ok is False
    assert "empty" in outcome.error


def test_modelhub_integrity_disk_precheck_refuses_before_the_first_byte(tmp_path):
    """Not enough free space is an honest refusal naming free, needed and path."""
    opener = ExplodingOpener()
    hub = make_downloader(
        opener,
        models_dir=tmp_path,
        system_provider=FakeSystemProvider(free_mb=100.0),
    )
    outcome = hub.download("o/r", "m.gguf", size_bytes=50 * 1024 * 1024 * 1024)
    assert outcome.ok is False
    assert "not enough free space" in outcome.error
    assert opener.calls == 0


# =========================================================================== #
# AC-M14-21: modelhub_trust - T1, T2, T3
# =========================================================================== #


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd",
        "owner/../../name",
        "https://huggingface.co/owner/name",
        "user@host/name",
        "owner/name/extra",
        "ownername",
        "/leading/slash",
        "owner/",
        "",
        "owner name/x",
        "%2e%2e/name",
    ],
)
def test_modelhub_trust_rejects_a_bad_repo_id(bad):
    """T1: an unsafe repo id is refused BEFORE any URL is built."""
    with pytest.raises(ValueError):
        modelhub.validate_repo_id(bad)
    with pytest.raises(ValueError):
        modelhub.build_tree_url("https://huggingface.co", bad)


@pytest.mark.parametrize(
    "bad",
    [
        "../m.gguf",
        "sub/dir/m.gguf",
        "m.gguf.exe",
        "m.bin",
        "models\\m.gguf",
        "m gguf.gguf",
        "",
        ".gguf",
        "m.GGUF/../x.gguf",
    ],
)
def test_modelhub_trust_rejects_a_bad_filename(bad):
    """T1: an unsafe file name is refused before it can reach a URL or a path."""
    with pytest.raises(ValueError):
        modelhub.validate_filename(bad)


def test_modelhub_trust_builds_urls_and_never_accepts_one():
    """T1: the built URL is exactly the documented shape, from validated parts."""
    assert modelhub.build_resolve_url(
        "https://huggingface.co", "unsloth/Qwen3.8-27B-GGUF", REAL_FILENAME
    ) == (
        "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/" + REAL_FILENAME
    )
    # And the Downloader's public download() surface takes repo/file, not a URL.
    import inspect

    params = inspect.signature(modelhub.Downloader.download).parameters
    assert "url" not in params
    assert "repo_id" in params and "filename" in params


@pytest.mark.parametrize(
    "host,expected",
    [
        ("huggingface.co", True),
        ("hf.co", True),
        ("us.aws.cdn.hf.co", True),  # the real CDN host a live download lands on
        ("cdn-lfs.huggingface.co", True),
        ("evil-hf.co", False),
        ("hf.co.evil.com", False),
        ("huggingface.co.attacker.net", False),
        ("localhost", False),
        ("", False),
    ],
)
def test_modelhub_trust_host_allowlist(host, expected):
    """T2: equal-or-subdomain, never substring."""
    assert modelhub.host_allowed(host, modelhub.DEFAULT_ALLOWED_HOSTS) is expected


def test_modelhub_trust_refuses_non_https_and_userinfo():
    """T2: scheme and userinfo are checked before any request is made."""
    for url in (
        "http://huggingface.co/a/b",
        "file:///etc/passwd",
        "ftp://huggingface.co/a",
        "https://user:pass@huggingface.co/a/b",
    ):
        with pytest.raises(modelhub.HubError):
            modelhub.validate_url(url, modelhub.DEFAULT_ALLOWED_HOSTS)


def test_modelhub_trust_revalidates_every_redirect_hop():
    """T2: five hops are re-validated; a sixth, and any bad host, are refused."""
    handler = modelhub._AllowlistRedirectHandler(modelhub.DEFAULT_ALLOWED_HOSTS)
    request = urllib.request.Request("https://huggingface.co/a/b")
    headers = {}
    for hop in range(1, 6):
        result = handler.redirect_request(
            request, None, 302, "Found", headers,
            f"https://us.aws.cdn.hf.co/hop{hop}",
        )
        assert result is not None, f"hop {hop} should be allowed"
    # The 6th hop exceeds the budget and is refused.
    assert handler.redirect_request(
        request, None, 302, "Found", headers, "https://hf.co/hop6"
    ) is None


@pytest.mark.parametrize(
    "target",
    [
        "http://huggingface.co/plain",           # downgraded scheme
        "https://evil.example.com/payload",      # off-allowlist host
        "https://hf.co.evil.example.com/x",      # lookalike host
        "https://user:pw@huggingface.co/x",      # userinfo smuggling
    ],
)
def test_modelhub_trust_refuses_a_bad_redirect_target(target):
    """T2: a hop that fails validation returns None, which urllib turns into an error."""
    handler = modelhub._AllowlistRedirectHandler(modelhub.DEFAULT_ALLOWED_HOSTS)
    request = urllib.request.Request("https://huggingface.co/a/b")
    assert handler.redirect_request(request, None, 302, "Found", {}, target) is None


def test_modelhub_trust_confines_the_destination(tmp_path):
    """T3: the resolved path must stay inside the models directory."""
    models = tmp_path / "models"
    models.mkdir()
    assert modelhub.resolve_destination(models, "m.gguf") == (models / "m.gguf").resolve()
    for attempt in ("../escape.gguf", "..\\escape.gguf", "sub/../../escape.gguf"):
        with pytest.raises(ValueError):
            modelhub.resolve_destination(models, attempt)


def test_modelhub_trust_confinement_survives_a_symlinked_models_dir(tmp_path):
    """T3: resolve() before comparing, so a symlinked models dir cannot escape."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform/user cannot create a directory symlink")
    resolved = modelhub.resolve_destination(link, "m.gguf")
    assert resolved == (real / "m.gguf").resolve()
    assert resolved.parent == real.resolve()


def _link_escaping_the_models_dir(link_path, target_dir):
    """Make `link_path` a link that resolves into `target_dir`, or return False.

    Tries a real symlink first, then a Windows directory junction (which needs
    no special privilege, unlike a symlink on a machine without developer mode).
    Returning False lets the caller skip rather than fail on a platform that
    forbids both.
    """
    try:
        os.symlink(target_dir / link_path.name, link_path)
        return True
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target_dir)],
            capture_output=True,
        )
        return link_path.exists()
    return False


def test_modelhub_trust_confinement_refuses_a_link_out_of_the_models_dir(tmp_path):
    """T3 gate 2 (AC-M14-21c) on a REAL crafted path (review HIGH-3).

    "x.gguf" passes the filename regex cleanly, so this is an input that gets
    PAST gate 1 and can only be stopped by the resolve()+is_relative_to check.
    That is what AC-M14-21(c) names, and what the previous traversal tests never
    reached: all three of their inputs died at the regex.
    """
    models = tmp_path / "models"
    models.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = models / "x.gguf"
    if not _link_escaping_the_models_dir(link, outside):
        pytest.skip("this platform/user can create neither a symlink nor a junction")
    assert modelhub.FILENAME_RE.match("x.gguf"), "the input must pass gate 1"
    assert not link.resolve().is_relative_to(models.resolve()), (
        "the crafted link does not actually escape - the test would prove nothing"
    )
    with pytest.raises(ValueError) as caught:
        modelhub.resolve_destination(models, "x.gguf")
    assert "outside the models directory" in str(caught.value)


def test_modelhub_trust_confinement_refuses_an_escaping_resolve(tmp_path, monkeypatch):
    """The same gate, proved on EVERY platform without needing link privilege.

    The test above is the realistic reproduction but can legitimately skip; a
    safety gate whose only coverage is skippable is not covered. This one
    substitutes a Path whose resolve() escapes exactly the way a symlinked
    <name>.gguf does, so `resolve_destination` and `_is_inside` run for real,
    everywhere, every run.
    """
    models = tmp_path / "models"
    models.mkdir()
    escape = tmp_path / "outside" / "x.gguf"
    real_resolve = Path.resolve

    def resolve(self, *args, **kwargs):
        # Only the candidate file escapes; the root must still resolve normally
        # or the comparison would be meaningless.
        if self.name == "x.gguf":
            return escape
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError) as caught:
        modelhub.resolve_destination(models, "x.gguf")
    assert "outside the models directory" in str(caught.value)
    # Control: a candidate that stays inside is still returned, so the test is
    # proving the comparison, not that everything raises.
    inside = modelhub.resolve_destination(models, "kept.gguf")
    assert inside.parent == real_resolve(models)


def test_modelhub_trust_hop_budget_is_per_request_not_cumulative(tmp_path):
    """The HEAD probe must not spend the GET's redirect budget.

    One handler instance serves both requests, and its counter only ever
    incremented, so a multi-hop probe used to leave the real transfer short and
    fail it with a refusal it had not earned. Five hops on each request is the
    documented budget; this asserts both requests get all five.
    """
    body = gguf_bytes(b"hops")
    handler = modelhub._AllowlistRedirectHandler(modelhub.DEFAULT_ALLOWED_HOSTS)
    opener = RedirectingOpener(handler, body, hop_count=5)
    hub = modelhub.Downloader(
        modelhub.HubConfig(),
        opener_factory=lambda hosts: (opener, handler),
        models_dir=tmp_path,
    )
    outcome = hub.download("owner/repo", "m.gguf", confirm_unverified=True)
    assert outcome.ok is True, outcome.error
    assert opener.calls == 2, "expected one HEAD probe and one GET"
    assert opener.hops_allowed == [5, 5], (
        f"the hop budget did not reset between requests: {opener.hops_allowed}"
    )


def test_modelhub_trust_refuses_a_silent_clobber(tmp_path):
    """T4: an existing destination is never overwritten."""
    dest = tmp_path / "m.gguf"
    dest.write_bytes(b"the user's own copy")
    opener = ExplodingOpener()
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf",
        dest,
        "a" * 64,
        opener=opener,
    )
    assert outcome.ok is False
    assert "already exists" in outcome.error
    assert dest.read_bytes() == b"the user's own copy"
    assert opener.calls == 0


# =========================================================================== #
# scripts/fetch_model.py - the owner-run CLI's printed provenance
# =========================================================================== #


def load_fetch_model():
    """Import scripts/fetch_model.py by path (scripts/ is not on sys.path)."""
    import importlib.util

    path = Path(modelhub.__file__).resolve().parent / "scripts" / "fetch_model.py"
    spec = importlib.util.spec_from_file_location("fetch_model_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modelhub_cli_url_form_never_claims_huggingface_verified_it(
    tmp_path, monkeypatch, capsys
):
    """--url + --sha256 checks the OPERATOR's digest, and must say so.

    The file really is verified against the digest given, so this is not a data
    problem - but printing "Checksum verified against HuggingFace's file
    listing." for a value
    that never came from HuggingFace is the exact overstatement rung V-NONE
    exists to prevent elsewhere. Runs the real CLI end to end over a fake opener.
    """
    cli = load_fetch_model()
    body = gguf_bytes(b"operator supplied")
    digest = hashlib.sha256(body).hexdigest()
    opener = CountingOpener(
        [FakeResponse(body, headers={"Content-Length": str(len(body))})]
    )
    monkeypatch.setattr(modelhub, "make_opener", lambda *a, **k: (opener, None))
    dest = tmp_path / "m.gguf"
    code = cli.main(
        [
            "--url", "https://huggingface.co/o/r/resolve/main/m.gguf",
            "--sha256", digest,
            "--dest", str(dest),
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert dest.exists()
    assert "verified against HuggingFace" not in printed
    assert "Checksum verified against the digest you supplied." in printed
    assert f"verification: {modelhub.V_OPERATOR}" in printed


def test_modelhub_cli_operator_rung_is_unreachable_from_the_gui():
    """V_OPERATOR exists only for the CLI; no GUI path can produce it."""
    gui_sources = [
        (Path(modelhub.__file__).resolve().parent / name).read_text(encoding="utf-8")
        for name in ("gui_controller.py", "desktop.py")
    ]
    for source in gui_sources:
        assert "V_OPERATOR" not in source
        assert '"operator"' not in source.replace('"operator_', "")
    # And the label itself never claims a HuggingFace provenance.
    assert "HuggingFace" not in modelhub.VERIFICATION_LABELS[modelhub.V_OPERATOR]


# =========================================================================== #
# AC-M14-22: modelhub_no_token - no HuggingFace credential, anywhere
# =========================================================================== #


def test_modelhub_no_token_source_carries_no_credential_literal():
    """The AC-M14-22 grep, run as a test so it cannot regress unnoticed."""
    source = (
        Path(modelhub.__file__).with_suffix(".py").read_text(encoding="utf-8").lower()
    )
    assert "token" not in source
    assert "authorization" not in source


def test_modelhub_no_token_settings_schema_has_no_credential_field():
    """The settings block accepts no credential field of any kind."""
    import dataclasses

    import config

    names = {f.name.lower() for f in dataclasses.fields(config.ModelsHubConfig)}
    for banned in ("token", "api_key", "apikey", "password", "secret", "credential",
                   "auth", "authorization", "hf_token"):
        assert banned not in names
    hub_names = {f.name.lower() for f in dataclasses.fields(modelhub.HubConfig)}
    assert not (hub_names & {"token", "api_key", "secret", "password", "auth"})


def test_modelhub_no_token_downloader_accepts_no_credential_argument():
    """Neither the constructor nor any egress method takes a credential."""
    import inspect

    for func in (
        modelhub.Downloader.__init__,
        modelhub.Downloader.search,
        modelhub.Downloader.list_files,
        modelhub.Downloader.download,
        modelhub.download_verified,
    ):
        params = {p.lower() for p in inspect.signature(func).parameters}
        assert not (params & {"token", "api_key", "auth", "password", "credential",
                              "headers", "authorization"})


def test_modelhub_no_token_401_is_reported_and_never_retried():
    """A 401 yields the honest message and exactly ONE request - no credential retry."""
    recorded = load_fixture("hf_gated_or_absent_401.json")
    assert recorded["status"] == 401  # recorded from a real anonymous request
    error = urllib.error.HTTPError(
        "https://huggingface.co/api/models/a/b/tree/main",
        401, "Unauthorized", {}, None,
    )
    opener = CountingOpener([error])
    hub = make_downloader(opener)
    result = hub.list_files("meta-llama/Some-Gated-Repo")
    assert result["ok"] is False
    assert result["kind"] == "gated"
    assert result["reason"] == modelhub.GATED_MESSAGE
    assert "Register" in result["reason"]  # the manual workaround is offered
    assert opener.calls == 1


def test_modelhub_no_token_403_uses_the_same_honest_message():
    """403 is treated exactly like 401: reported, never worked around."""
    error = urllib.error.HTTPError("https://huggingface.co/x", 403, "Forbidden", {}, None)
    hub = make_downloader(CountingOpener([error]))
    assert hub.list_files("a/b")["reason"] == modelhub.GATED_MESSAGE


def test_modelhub_no_token_429_is_never_auto_retried():
    """A rate limit is one request and one honest sentence - auto-retry earns a ban."""
    error = urllib.error.HTTPError("https://huggingface.co/x", 429, "Too Many", {}, None)
    opener = CountingOpener([error, error, error])
    hub = make_downloader(opener)
    result = hub.search("qwen")
    assert result["ok"] is False
    assert result["kind"] == "rate_limited"
    assert result["reason"] == modelhub.RATE_LIMIT_MESSAGE
    assert opener.calls == 1


def test_modelhub_no_token_offline_keeps_the_catalog_usable(tmp_path):
    """Offline is a first-class state: the reason renders and the catalog survives."""
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"models": [{"repo_id": "a/b"}]}), encoding="utf-8")
    opener = CountingOpener([urllib.error.URLError("getaddrinfo failed")])
    hub = make_downloader(opener, config=modelhub.HubConfig(catalog_path=str(catalog)))
    result = hub.search("qwen")
    assert result["ok"] is False
    assert result["kind"] == "offline"
    assert "Could not reach huggingface.co" in result["reason"]
    assert "built-in catalog" in result["reason"]
    assert hub.catalog()["items"][0]["repo_id"] == "a/b"


# =========================================================================== #
# AC-M14-24: modelhub_registry - one writer, two provenance values, additive field
# =========================================================================== #


def test_modelhub_registry_uses_append_model_entry(tmp_path, monkeypatch):
    """A completed download is written through M13's writer, spied not replaced."""
    import config
    import gui_controller

    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(config, "append_model_entry", spy)

    controller = gui_controller.GuiController.__new__(gui_controller.GuiController)
    controller._settings = type("S", (), {"data_dir": tmp_path})()
    controller.result_q = __import__("queue").Queue()
    controller.refresh_models = lambda: None
    done = {
        "filename": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "repo_id": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "path": str(tmp_path / "m.gguf"),
    }
    outcome = modelhub.DownloadOutcome(
        True, path=tmp_path / "m.gguf", sha256="b" * 64,
        verification=modelhub.V_API,
    )
    controller._hub_register(done, outcome)
    assert len(calls) == 1
    _args, kwargs = calls[0]
    assert kwargs["sha256"] == "b" * 64
    # DEFECT-QA-M14-6: the row must name the LITERAL provenance field, not the
    # internal rung key, per UX Spec section 14's engineer-facing rule.
    assert "verification=lfs.oid" in kwargs["notes"]
    assert "verification=api" not in kwargs["notes"]
    assert done["registered"] is True
    assert done["model_id"] == "qwen3-4b-instruct-2507-q4_k_m"


def test_modelhub_registry_writes_no_third_provenance_value(tmp_path):
    """A downloaded row is an ordinary source: 'registry' row."""
    import config

    (tmp_path / "models.yaml").write_text("version: 1\nmodels: []\n", encoding="utf-8")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(gguf_bytes())
    config.append_model_entry(tmp_path, "m", "M", str(gguf), sha256="c" * 64)
    models = config.Config.load(base_dir=tmp_path)[1]
    row = next(m for m in models.models if m.id == "m")
    assert row.source == "registry"
    assert {m.source for m in models.models} <= {"registry", "discovered"}
    text = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    assert "downloaded" not in text.lower().replace("downloaded from", "")


def test_modelhub_registry_failure_never_deletes_the_file(tmp_path, monkeypatch):
    """A registry write failure keeps the multi-gigabyte file the user waited for."""
    import config
    import gui_controller

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(gguf_bytes())

    def boom(*args, **kwargs):
        raise ValueError("an entry named 'm' already exists")

    monkeypatch.setattr(config, "append_model_entry", boom)
    controller = gui_controller.GuiController.__new__(gui_controller.GuiController)
    controller._settings = type("S", (), {"data_dir": tmp_path})()
    controller.result_q = __import__("queue").Queue()
    done = {"filename": "m.gguf", "repo_id": "o/r", "path": str(gguf)}
    controller._hub_register(
        done, modelhub.DownloadOutcome(True, path=gguf, sha256="d" * 64)
    )
    assert done["registered"] is False
    assert "register_error" in done
    assert gguf.exists(), "the downloaded file must survive a registry failure"


def test_modelhub_registry_sha256_field_is_additive(tmp_path):
    """An existing models.yaml with no sha256 key parses unchanged, defaulting to ''."""
    import config

    original = (
        "version: 1\n"
        "models:\n"
        "  - id: legacy\n"
        "    name: Legacy\n"
        "    description: \"\"\n"
        "    location: \"\"\n"
        "    context_size: 8192\n"
        "    gpu_layers: 999\n"
    )
    path = tmp_path / "models.yaml"
    path.write_text(original, encoding="utf-8")
    models = config.Config.load(base_dir=tmp_path)[1]
    row = next(m for m in models.models if m.id == "legacy")
    assert row.sha256 == ""
    # Reading must not rewrite: the file is byte-identical afterwards.
    assert path.read_text(encoding="utf-8") == original


def test_modelhub_registry_settings_version_is_unchanged():
    """The additive field required no schema bump, and none was made."""
    import config

    assert config.CURRENT_SETTINGS_VERSION == 2
    assert set(config.SUPPORTED_SETTINGS_VERSIONS) == {1, 2}


def test_modelhub_registry_catalog_is_never_registry_data():
    """model_catalog.json is not read by config.py and holds no location/sha256."""
    import config

    platform_dir = Path(config.__file__).resolve().parent
    catalog_path = platform_dir / "model_catalog.json"
    assert catalog_path.is_file(), "the shipped catalog must exist"
    raw = catalog_path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    for entry in parsed["models"]:
        assert "location" not in entry
        assert "sha256" not in entry
        for item in entry["files"]:
            assert "sha256" not in item
            assert "location" not in item
    # config.py must never READ it: it neither imports modelhub nor opens the
    # catalog, so no catalog row can reach a ModelRegistry by any route. (The
    # file name does appear once in a config docstring, describing what a blank
    # models_hub.catalog_path means - documentation, not a read.)
    config_source = Path(config.__file__).read_text(encoding="utf-8")
    assert "import modelhub" not in config_source
    assert "model_catalog.json" not in config_source.replace(
        "installed model_catalog.json", ""
    )
    catalog_repos = {e["repo_id"] for e in parsed["models"]}
    loaded = config.Config.load()[1]
    assert not ({m.id for m in loaded.models} & catalog_repos)
    assert not ({m.location for m in loaded.models} & {str(catalog_path)})


def test_modelhub_registry_default_models_yaml_still_parses_to_zero_rows():
    """AC-M14-3(a) still holds with the catalog present on disk."""
    import config
    import shutil

    import tempfile

    platform_dir = Path(config.__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copy(platform_dir / "models.default.yaml", root / "models.yaml")
        shutil.copy(platform_dir / "settings.default.yaml", root / "settings.yaml")
        shutil.copy(platform_dir / "model_catalog.json", root / "model_catalog.json")
        models = config.Config.load(base_dir=root)[1]
        assert models.models == []


def test_modelhub_registry_id_sanitiser_matches_the_documented_rule():
    """Ids are lowercased and reduced to [a-z0-9_-], collapsed, never empty."""
    assert modelhub.registry_id_for(REAL_FILENAME) == "qwen3-8-27b-iq4_nl"
    assert modelhub.registry_id_for("a  b!!c.gguf") == "a-b-c"
    assert modelhub.registry_id_for("!!!.gguf") == "model"


# =========================================================================== #
# Parsing and guidance against the recorded real payloads
# =========================================================================== #


def test_modelhub_parses_the_real_search_response():
    """The recorded search body yields usable rows, licence tag included."""
    rows = modelhub.parse_search_results(load_fixture("hf_search_response.json"))
    assert rows
    assert all(modelhub.REPO_ID_RE.match(r["repo_id"]) for r in rows)
    assert all(r["publisher"] == r["repo_id"].split("/")[0] for r in rows)
    # Recorded fact: the search API has no `license` field, only a tags list, so
    # this is where the licence has to come from.
    assert any(r["license_tag"] for r in rows)


def test_modelhub_skips_subdirectory_files_in_the_real_tree_response():
    """Sharded GGUFs under a subdirectory are not offered - a stated limitation."""
    payload = load_fixture("hf_tree_response.json")
    subdir_names = [
        e["path"] for e in payload
        if e.get("type") == "file" and e["path"].endswith(".gguf") and "/" in e["path"]
    ]
    assert subdir_names, "the recorded repository does contain subdirectory GGUFs"
    offered = {r["filename"] for r in modelhub.parse_tree_files(payload)}
    assert not any(name in offered for name in subdir_names)


def test_modelhub_quant_is_read_from_the_filename():
    """Quant labels are read off real file names, '' when there is none."""
    assert modelhub.quant_from_filename(REAL_FILENAME) == "IQ4_NL"
    assert modelhub.quant_from_filename("gemma-3-4b-it-Q4_K_M.gguf") == "Q4_K_M"
    assert modelhub.quant_from_filename("model.gguf") == ""


@pytest.mark.parametrize(
    "size_gb,vram_mb,band",
    [
        (2.5, 16384, "fits"),
        (13.0, 16384, "tight"),
        (30.0, 16384, "exceeds"),
    ],
)
def test_modelhub_fit_bands(size_gb, vram_mb, band):
    """The four bands, computed from the documented arithmetic."""
    fit = modelhub.estimate_fit(
        int(size_gb * 1024 * 1024 * 1024), [FakeGpu("RTX 4080", vram_mb)]
    )
    assert fit["band"] == band
    assert fit["disclaimer"] == modelhub.FIT_DISCLAIMER
    assert "GB weights" in fit["explanation"]


def test_modelhub_fit_unknown_makes_no_claim():
    """No GPU means no band, no number, no verdict - just the honest statement."""
    fit = modelhub.estimate_fit(9_000_000_000, None)
    assert fit["band"] == "unknown"
    assert fit["budget_mb"] is None
    assert fit["explanation"] == modelhub.FIT_UNKNOWN_TEXT
    assert "fit" not in fit["wording"].lower().replace("estimate the fit", "")


def test_modelhub_fit_uses_the_injected_provider_only(tmp_path):
    """modelhub asks health's provider for GPU facts and never probes itself."""
    provider = FakeGpuProvider([FakeGpu("A", 8192), FakeGpu("B", 24576)])
    hub = make_downloader(ExplodingOpener(), gpu_provider=provider)
    fit = hub.fit_for(4_000_000_000)
    assert provider.calls == 1
    assert fit["gpu_name"] == "B"  # largest SINGLE card, never the sum
    assert fit["multi_gpu_caveat"] == modelhub.MULTI_GPU_CAVEAT
    source = Path(modelhub.__file__).read_text(encoding="utf-8")
    assert "nvidia-smi" not in source


def test_modelhub_catalog_and_search_merge_with_live_winning():
    """The union keeps every row, and the live one wins a conflict."""
    merged = modelhub.merge_catalog_and_search(
        [{"repo_id": "a/b", "source": "catalog", "license_tag": "stale"},
         {"repo_id": "c/d", "source": "catalog"}],
        [{"repo_id": "a/b", "source": "api", "license_tag": "fresh"}],
    )
    by_id = {row["repo_id"]: row for row in merged}
    assert set(by_id) == {"a/b", "c/d"}
    assert by_id["a/b"]["source"] == "api"
    assert by_id["a/b"]["license_tag"] == "fresh"


def test_modelhub_catalog_missing_file_is_reported_not_raised(tmp_path):
    """A missing catalog degrades to an honest empty list."""
    result = modelhub.load_catalog(tmp_path / "nope.json")
    assert result["items"] == []
    assert "no catalog file" in result["reason"]


def test_modelhub_shipped_catalog_loads():
    """The shipped catalog parses cleanly and is EMPTY by decision.

    Owner decision 2026-08-29: LOCITIZE ships no model list - first-run setup
    imports the models a machine already has, and the hub searches in the
    user's own words. The file stays (a fork may curate for its audience), so
    this test now holds two lines: it must parse without a reason, and any
    entries a fork adds must still satisfy the validators below.
    """
    result = modelhub.load_catalog()
    assert result["reason"] == ""
    assert result["items"] == []  # ships empty; the loop below guards forks
    for item in result["items"]:
        assert modelhub.REPO_ID_RE.match(item["repo_id"])
        for entry in item["files"]:
            # Every entry, sharded or not, must pass the path validator.
            modelhub.validate_repo_path(entry["filename"])
            if entry.get("shard_parts"):
                # M15.2: a shard entry names part one inside a quant folder, so it
                # is NOT a plain file name - that is the whole reason it exists.
                parsed = modelhub.parse_shard(entry["filename"])
                assert parsed is not None, entry["filename"]
                assert parsed[1] == 1, "a catalog entry must name part one"
                assert parsed[2] == entry["shard_parts"]
            else:
                # Unsharded entries keep the original, stricter guarantee.
                assert modelhub.FILENAME_RE.match(entry["filename"])


def test_modelhub_cancel_stops_mid_transfer(tmp_path):
    """Cancellation is checked every chunk and leaves no file at the final name."""
    body = gguf_bytes(b"x" * (4 * 1024 * 1024))
    opener = CountingOpener([FakeResponse(body)])
    cancel = threading.Event()
    cancel.set()  # already cancelled: the very first chunk boundary must stop
    dest = tmp_path / "m.gguf"
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf",
        dest,
        None,
        opener=opener,
        confirm_unverified=True,
        cancel_event=cancel,
        chunk_bytes=1024,
    )
    assert outcome.cancelled is True
    assert outcome.ok is False
    assert not dest.exists()


def test_modelhub_progress_is_throttled_to_twice_a_second(tmp_path):
    """Progress is emitted at most every 500 ms, plus one per phase change.

    Without the throttle a 9 GB download at a 1 MiB chunk size would push ~9000
    objects through the queue and make the GUI thread the bottleneck.
    """
    body = gguf_bytes(b"y" * (200 * 1024))
    opener = CountingOpener([FakeResponse(body)])
    ticks = iter([0.0] + [0.01 * i for i in range(1, 500)])
    samples = []
    modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf",
        tmp_path / "m.gguf",
        None,
        opener=opener,
        confirm_unverified=True,
        cancel_event=None,
        progress=samples.append,
        clock=lambda: next(ticks),
        chunk_bytes=1024,
    )
    downloading = [s for s in samples if s.phase == "downloading"]
    # ~200 chunks over a simulated ~2 s: far fewer than one sample per chunk.
    assert len(downloading) < 20
    assert {"connecting", "verifying", "done"} <= {s.phase for s in samples}


# =========================================================================== #
# SEC-M14-1 / SEC-M14-2 regressions (security gate, 2026-08-19)
# =========================================================================== #


# The exact string Security's check-9b probe used. Kept verbatim so this test
# tracks the reported attack rather than a paraphrase of it.
SEC_M14_1_PAYLOAD = (
    "license:<br><br><b>Checksum verified against HuggingFace's "
    "file listing.</b><br>"
)


def test_modelhub_trust_license_tag_drops_publisher_markup():
    """A publisher-authored licence tag cannot carry markup out of the parser.

    SEC-M14-1, data half: `tags` is free text written by the repository owner, so
    the value must be reduced to the narrow shape a licence id has BEFORE any
    caller can render it.
    """
    tag = modelhub.license_tag_from_tags(["pytorch", SEC_M14_1_PAYLOAD])
    # Accept-or-drop: a value that is not licence-id-shaped is treated as if the
    # repo declared nothing, so none of the attacker's words survive either.
    assert tag == ""
    # Real licence ids are untouched.
    assert modelhub.license_tag_from_tags(["license:apache-2.0"]) == "apache-2.0"
    assert modelhub.license_tag_from_tags(["license:cc-by-nc-4.0"]) == "cc-by-nc-4.0"
    # A repo that declares nothing still reads as "not stated", never guessed.
    assert modelhub.license_tag_from_tags(["pytorch", "text-generation"]) == ""


def test_modelhub_trust_license_tag_is_length_capped():
    """A tag cannot be long enough to push a dialog's other lines off screen."""
    assert modelhub.license_tag_from_tags(["license:" + "a" * 5000]) == ""
    # The boundary itself is accepted, so the cap is not a hidden second rule.
    at_limit = "a" * modelhub.LICENSE_TAG_MAX_LEN
    assert modelhub.license_tag_from_tags(["license:" + at_limit]) == at_limit


def test_modelhub_trust_license_tag_refuses_a_newline_injection():
    """A tag that smuggles extra lines is dropped, not partially kept."""
    tag = modelhub.license_tag_from_tags(
        ["license:mit\n\nNot verified - no publisher checksum available."]
    )
    assert tag == ""


class EndlessResponse:
    """A response that never ends and declares no size (SEC-M14-2's shape).

    Mirrors the abuse Security measured: no `Content-Length`, and a tree entry
    that reported no size either, so nothing outside the read loop bounds it.

    `hard_stop` is a TEST safety net, not part of the scenario: without a cap in
    the product this response would feed the writer forever, so a regression must
    end as a loud failure rather than as a hung suite (and a full disk). It is set
    well above the ceiling under test, so it can only trigger if the ceiling did
    not.
    """

    def __init__(self, chunk=b"\0" * 4096, headers=None, hard_stop=8 * 1024 * 1024):
        self._chunk = chunk
        self.headers = http_headers(headers or {})
        self.status = 200
        self.url = ""
        self.reads = 0
        self.served = 0
        self._hard_stop = hard_stop
        self._first = True

    def read(self, size=-1):
        self.reads += 1
        if self.served > self._hard_stop:
            raise AssertionError(
                f"the read loop was still pulling bytes after {self.served}: "
                "the download has no working byte ceiling"
            )
        if self._first:
            # Start with the GGUF magic so the refusal under test is the byte
            # ceiling, not the magic check.
            self._first = False
            self.served += len(self._chunk) + len(modelhub.GGUF_MAGIC)
            return modelhub.GGUF_MAGIC + self._chunk
        self.served += len(self._chunk)
        return self._chunk

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_modelhub_integrity_caps_a_download_that_declares_no_size(tmp_path):
    """An endless, size-less transfer is stopped at the cap and the partial deleted.

    SEC-M14-2: before the fix this loop had no cumulative bound - Security wrote
    1.15 GB in 1.0 s with no size, no Content-Length and no disk check.
    """
    response = EndlessResponse()
    opener = CountingOpener([response])
    dest = tmp_path / "m.gguf"
    outcome = modelhub.download_verified(
        "https://huggingface.co/o/r/resolve/main/m.gguf",
        dest,
        None,
        opener=opener,
        confirm_unverified=True,
        expected_size=None,
        max_bytes=64 * 1024,
        chunk_bytes=4096,
    )
    assert outcome.ok is False
    assert "size limit" in outcome.error
    assert outcome.bytes_written <= 64 * 1024 + 8192  # one chunk of overshoot
    assert not dest.exists()
    assert not dest.with_suffix(".gguf.partial").exists()


def test_modelhub_integrity_has_a_ceiling_even_with_no_caller_budget():
    """There is no unbounded mode: the module backstop is a real number."""
    assert modelhub.MAX_UNDECLARED_DOWNLOAD_BYTES > 0
    hub = make_downloader(ExplodingOpener(), system_provider=None)
    # No declared size and no measurable free space still yields a finite cap.
    assert hub._write_ceiling(0, {"ok": True, "free_mb": None}) == (
        modelhub.MAX_UNDECLARED_DOWNLOAD_BYTES
    )


def test_modelhub_integrity_undeclared_size_still_consults_the_disk(tmp_path):
    """`size_bytes: 0` no longer skips the free-space check (SEC-M14-2).

    The pre-fix guard was `if size_bytes:`, so a tree entry with no size walked
    past the disk check entirely. Now a full volume refuses the download before
    a socket is opened - proven by the opener never being called.
    """
    opener = ExplodingOpener()
    hub = make_downloader(opener, system_provider=FakeSystemProvider(100.0))
    outcome = hub.download(
        "owner/repo", "m.gguf", size_bytes=0, confirm_unverified=True,
        models_dir=tmp_path,
    )
    assert outcome.ok is False
    assert "not enough free space" in outcome.error
    assert opener.calls == 0


def test_modelhub_integrity_free_space_becomes_the_ceiling_when_size_is_unknown():
    """With no declared size the write cap is what the volume can spare."""
    hub = make_downloader(
        ExplodingOpener(), system_provider=FakeSystemProvider(10_240.0)
    )
    disk = hub.check_disk(0, ".")
    ceiling = hub._write_ceiling(0, disk)
    headroom_mb = hub.config.min_free_disk_headroom_mb
    assert ceiling == int((10_240.0 - headroom_mb) * 1024 * 1024)
    # A declared size is tighter than free space, so it wins.
    assert hub._write_ceiling(1234, disk) == 1234


# =========================================================================== #
# DEFECT-QA-M14-1: a clean install has no models directory yet
#
# Every download test above hands the Downloader a directory that already
# exists - tmp_path, or a subdirectory a fixture created. That is why the first
# download of a real new install failed in QA's live run and the whole suite
# stayed green: shutil.disk_usage on a path that is not there raises
# FileNotFoundError (WinError 3) before a single byte moves. These tests use a
# path that does NOT exist, and the first one uses the REAL system provider so
# the real shutil call is the thing being exercised.
# =========================================================================== #


def test_modelhub_download_creates_the_models_directory_on_a_clean_install(tmp_path):
    """download() into a not-yet-created models directory succeeds end to end.

    Uses health.DefaultSystemInfoProvider, not FakeSystemProvider: the bug lived
    inside the real shutil.disk_usage call, so a fake disk provider here would
    reproduce nothing.
    """
    import health

    body = gguf_bytes(b"clean install")
    digest = hashlib.sha256(body).hexdigest()
    models_dir = tmp_path / "LOCITIZE" / "models"  # two levels, neither created
    assert not models_dir.exists()
    opener = CountingOpener([FakeResponse(body)])
    hub = make_downloader(
        opener,
        models_dir=models_dir,
        system_provider=health.DefaultSystemInfoProvider(),
    )
    outcome = hub.download(
        "owner/repo", "m.gguf", expected_sha256=digest,
        verification=modelhub.V_API, size_bytes=len(body),
    )
    assert outcome.ok is True, outcome.error
    assert models_dir.is_dir()
    assert (models_dir / "m.gguf").read_bytes() == body


def test_modelhub_download_on_a_clean_install_never_leaks_a_raw_oserror(tmp_path):
    """The failure QA saw - "unexpected error: [WinError 3]" - cannot recur.

    Belt and braces for the case above: even if the directory somehow cannot be
    created, download() must return a DownloadOutcome, never raise, so the GUI
    boundary never has to turn an OSError into "unexpected error".
    """
    import health

    # A FILE standing where the models directory should be: mkdir(exist_ok=True)
    # raises here on every platform, which is the closest portable stand-in for
    # "this location cannot be a directory".
    blocker = tmp_path / "models"
    blocker.write_text("not a directory", encoding="utf-8")
    opener = ExplodingOpener()
    hub = make_downloader(
        opener, models_dir=blocker,
        system_provider=health.DefaultSystemInfoProvider(),
    )
    outcome = hub.download(
        "owner/repo", "m.gguf", expected_sha256="a" * 64,
        verification=modelhub.V_API, size_bytes=10,
    )
    assert outcome.ok is False
    assert "WinError" not in outcome.error or "could not create" in outcome.error
    assert str(blocker) in outcome.error
    assert "settings.yaml" in outcome.error  # a next step, not just a diagnostic
    assert opener.calls == 0


def test_modelhub_check_disk_on_a_missing_path_refuses_with_a_next_step(tmp_path):
    """check_disk is public: an unmeasurable path refuses, it does not explode."""
    import health

    hub = make_downloader(
        ExplodingOpener(), system_provider=health.DefaultSystemInfoProvider()
    )
    missing = tmp_path / "does" / "not" / "exist"
    disk = hub.check_disk(1_000_000, missing)
    assert disk["ok"] is False
    assert str(missing) in disk["reason"]
    assert "could not check free space" in disk["reason"]
    assert "settings.yaml" in disk["reason"]


def test_modelhub_insufficient_space_message_ends_with_a_next_step():
    """H11: the full-disk refusal names the remedy, not only the arithmetic."""
    hub = make_downloader(ExplodingOpener(), system_provider=FakeSystemProvider(100.0))
    disk = hub.check_disk(50_000_000_000, ".")
    assert disk["ok"] is False
    assert "not enough free space" in disk["reason"]
    assert "Free up space" in disk["reason"]
    assert "models_hub.download_dir" in disk["reason"]


# =========================================================================== #
# DEFECT-QA-M14-6: the registry notes name the literal provenance field
# =========================================================================== #


@pytest.mark.parametrize(
    ("rung", "field"),
    [
        (modelhub.V_API, "lfs.oid"),
        (modelhub.V_ETAG, "X-Linked-ETag"),
        (modelhub.V_OPERATOR, "operator-supplied --url digest"),
        (modelhub.V_NONE, "none"),
    ],
)
def test_modelhub_registry_notes_name_the_literal_field(rung, field):
    """UX Spec section 14: `notes` records the field, never the rung key."""
    notes = modelhub.registry_notes("o/r", "m.gguf", rung, "2026-08-19")
    assert notes.endswith(f"verification={field}")
    # The written VALUE is never a bare rung key. Compared as the whole value
    # rather than as a substring, because "operator" is a prefix of the operator
    # rung's own field name.
    value = notes.rsplit("verification=", 1)[1]
    assert value not in (modelhub.V_API, modelhub.V_ETAG, modelhub.V_OPERATOR)


def test_modelhub_registry_notes_unknown_rung_under_claims():
    """An unrecognised rung records "none" - the failure mode is under-claiming."""
    assert "verification=none" in modelhub.registry_notes(
        "o/r", "m.gguf", "quantum-attested", "2026-08-19"
    )


def test_modelhub_hub_job_id_is_the_one_formula_both_sides_use():
    """The controller and the view must derive a job id from the same helper."""
    import inspect

    import desktop
    import gui_controller

    assert modelhub.hub_job_id("owner/repo", "m.gguf") == "owner/repo/m.gguf"
    # Neither side may reformat the id locally: that drift is what made the
    # terminal-state bug (DEFECT-QA-M14-2) possible to reintroduce quietly.
    for module in (desktop, gui_controller):
        source = inspect.getsource(module)
        assert "hub_job_id" in source


# =========================================================================== #
# DEFECT-QA-M14-4 (H11): a registry failure must end with a next step
# =========================================================================== #


def _registry_write_error(message):
    """Build a config.RegistryWriteError, imported lazily to keep this file flat."""
    import config

    return config.RegistryWriteError(message)


def register_failure_message(monkeypatch, tmp_path, exc):
    """Run the register step with a writer that fails, and return what the user reads."""
    import config
    import gui_controller

    def boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(config, "append_model_entry", boom)
    controller = gui_controller.GuiController.__new__(gui_controller.GuiController)
    controller._settings = type("S", (), {"data_dir": tmp_path})()
    controller.result_q = __import__("queue").Queue()
    controller.refresh_models = lambda: None
    done = {
        "filename": "m.gguf",
        "repo_id": "o/r",
        "path": str(tmp_path / "models" / "m.gguf"),
    }
    outcome = modelhub.DownloadOutcome(
        True, path=tmp_path / "models" / "m.gguf", sha256="b" * 64,
        verification=modelhub.V_API,
    )
    controller._hub_register(done, outcome)
    assert done["registered"] is False
    return done["register_error"]


@pytest.mark.parametrize(
    "exc",
    [
        # What the writer raises now for the live failure QA produced with a
        # read-only models.yaml: one typed error whose text is already finished
        # (DEC-M14-11). A raw OSError is deliberately NOT in this list - since
        # the chokepoint landed it cannot reach this call site, and the
        # registry_write_errors suite proves that against a real read-only file
        # rather than by injection.
        _registry_write_error(
            "LOCITIZE could not write your model list at models.yaml: Access is "
            "denied. Close any program that has the file open, check that it is "
            "not marked read-only, then try again."
        ),
        _registry_write_error(
            "LOCITIZE could not update your model list because models.yaml has no "
            "top-level 'models:' section to write into, and it refuses to guess "
            "where the list should start. Open that file, add a 'models:' line, "
            "then try again."
        ),
        # A caller-input refusal, which stays a plain ValueError: the id is
        # already taken, which no chokepoint can fix.
        ValueError(
            "an entry named 'm' already exists in models.yaml. Rename or remove "
            "the existing entry first."
        ),
    ],
    ids=["read-only", "no-section", "duplicate-id"],
)
def test_modelhub_register_failure_message_ends_with_a_next_step(
    monkeypatch, tmp_path, exc
):
    """Every register_error keeps the file AND tells the user what to do next.

    The diagnostic is still there - it is the only thing that says why - but a
    message that stops at "[WinError 5] Access is denied" leaves the user with a
    downloaded model and no idea what to press.
    """
    message = register_failure_message(monkeypatch, tmp_path, exc)
    assert "downloaded and kept" in message
    assert str(tmp_path / "models" / "m.gguf") in message, "the file's location"
    assert "Register on the Models page" in message, "the next step"
    assert message.rstrip().endswith(".")


def test_config_append_model_entry_failures_each_name_a_next_step(tmp_path):
    """The writer's OWN refusals must be actionable, not just diagnostic.

    The test above pins the sentence gui_controller wraps around whatever
    `append_model_entry` raised. This one pins the raised text itself, because
    the same string is what a CLI caller or a log line shows with no wrapper
    around it, and DEFECT-QA-M14-4 named both reachable refusal branches
    ("refusing to guess where to append", "post-write verification failed") as
    having the same no-next-step shape as the OSError branch.

    Updated for DEC-M14-11: the missing-'models:' and post-write branches are now
    raised by config's one registry-write chokepoint as RegistryWriteError rather
    than as a bare ValueError, and both name the real path. The caller-input
    branch (a duplicate id) is still the writer's own ValueError.

    Nothing is mocked: the first two branches are driven by writing a real
    models.yaml into a real directory; the third calls the post-write check with
    a parsed document that genuinely lacks the row.
    """
    import config

    gguf = tmp_path / "models" / "m.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"gguf")

    # Branch 1: no top-level "models:" key at all.
    (tmp_path / "models.yaml").write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(config.RegistryWriteError) as no_section:
        config.append_model_entry(tmp_path, "m", "M", str(gguf))
    text = str(no_section.value)
    assert "refuses to guess" in text, "the diagnostic is still there"
    assert "add a 'models:' line" in text, "the next step"
    assert "then try again" in text
    assert str(tmp_path / "models.yaml") in text, "the real path, not a bare name"

    # Branch 2: the id is already taken.
    (tmp_path / "models.yaml").write_text(
        "version: 1\nmodels:\n"
        '  - id: "m"\n'
        '    name: "M"\n'
        f'    location: "{gguf.as_posix()}"\n'
        "    context_size: 4096\n"
        "    gpu_layers: 999\n"
        "    status: installed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as clash:
        config.append_model_entry(tmp_path, "m", "M", str(gguf))
    assert "Rename or remove the existing entry first." in str(clash.value)

    # Branch 3: the re-parsed document does not carry the row that was rendered.
    with pytest.raises(config.RegistryWriteError) as post_write:
        config._confirm_appended(
            {"models": []}, "m", str(gguf), tmp_path / "models.yaml"
        )
    text = str(post_write.value)
    assert "post-write verification failed" in text
    assert "file left unchanged" in text, "the file is safe, and says so"
    assert "formatting problem" in text and "then try again" in text, "the next step"
