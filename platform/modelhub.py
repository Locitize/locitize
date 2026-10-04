"""Model acquisition: browse HuggingFace, verify a GGUF, put it on disk (M14.14).

WHERE THIS FITS
---------------
`modelhub` is a leaf module. It imports no Qt, spawns no process, and knows
nothing about the GUI's Command/Result vocabulary. `gui_controller` calls it from
a background thread; `scripts/fetch_model.py` calls it from the command line.
Everything it touches from the outside world arrives through an injectable seam
(`opener_factory`, `gpu_provider`, `system_provider`, `clock`), which is what lets
the whole feature be tested headless with no network, no GPU and no disk pressure
- the same discipline health.py established.

THE FOUR RULES THAT SHAPE THIS FILE
-----------------------------------
1. EGRESS RULE HF-1. A network call happens ONLY inside `Downloader.search`,
   `Downloader.list_files` and `Downloader.download` - the three methods that
   exist solely to serve an explicit user press. Importing this module, building
   a `Downloader`, reading the catalog, and estimating VRAM fit are all pure and
   offline. Nothing here runs on a timer, a keystroke, a repaint or a health
   probe. That is the mechanical form of the product's "no egress you did not
   ask for" promise, and tests/test_modelhub.py asserts it by handing the
   Downloader an opener that raises if it is ever opened.
2. NEVER INVENT A HASH. A digest is either obtained from HuggingFace and
   compared byte-for-byte, or it does not exist and the file is labelled
   unverified forever. There is no third option and no override button.
3. URLS ARE BUILT, NEVER ACCEPTED. Callers hand over a validated `repo_id` and
   `filename`; this module composes the URL. The one raw-URL entry point in the
   whole product is `scripts/fetch_model.py --url`, a deliberate manual act.
4. NO CREDENTIALS OF ANY KIND. Anonymous requests only. A gated repository is
   reported honestly with a manual workaround. This is what keeps the shipped
   product's "secrets: none" claim literally true.

WHAT A REAL HUGGINGFACE RESPONSE LOOKS LIKE (recorded 2026-08-19, live calls)
-----------------------------------------------------------------------------
The fixtures under tests/fixtures/hf_*.json are verbatim recordings, and they
settled three things the design could only guess at:

- The tree API's `lfs.oid` IS a sha256 for GGUF files (64 hex). Rung V-API is
  real.
- `X-Linked-ETag` appears ONLY on huggingface.co's 302 hop, never on the CDN
  response the redirect lands on, and its value equals that same `lfs.oid`. So
  the redirect handler is the only place that can see it - hence
  `_AllowlistRedirectHandler` records it on the way past.
- The plain `etag` on the final CDN response is the Xet content hash, which is a
  DIFFERENT 64-hex value from the sha256. Accepting a bare `etag` would silently
  compare a file against the wrong digest. This is exactly why the ladder accepts
  `X-Linked-ETag` and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Constants and validating patterns (T1)
# --------------------------------------------------------------------------- #

# A HuggingFace repository id: exactly one '/', each side starting with an
# alphanumeric. Written as a whitelist rather than a blacklist on purpose - it
# admits no scheme, no '..', no '@' userinfo, no percent-encoding and no second
# slash, so nothing that reaches URL construction can reshape the URL.
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# A downloadable file name. Because the pattern forbids '/' and '\' outright, a
# validated name cannot carry a directory traversal or a Windows drive letter.
# Consequence worth stating: GGUF files that live in a repo SUBDIRECTORY (real
# example, "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf") do not match and are
# therefore not offered for download. That is a deliberate, narrow first version,
# not a bug - sharded multi-part GGUFs need a different download story anyway.
FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.gguf$")

# M15.2: the "different download story" the paragraph above anticipated.
#
# A large MoE is published as a shard SET in a quant subdirectory - the real
# example this was built for is
# "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf". Supporting it needs
# a path with one folder part, which FILENAME_RE deliberately forbids.
#
# The rule that keeps this safe: FILENAME_RE IS NOT RELAXED. A second, equally
# strict validator is added beside it, and it is the only thing that accepts a
# folder part. Each segment must independently match the same conservative
# charset, must start alphanumeric (so "." and ".." cannot appear at all - they
# are rejected by the charset, not by a blacklist), and the depth is capped. A
# caller that wants a plain file keeps using validate_filename and is unaffected.
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# One quant folder plus the file. Deeper nesting is not a real publishing layout
# and every extra level is more surface for no gain.
MAX_REPO_PATH_DEPTH = 2
MAX_REPO_PATH_CHARS = 255

# "<stem>-00001-of-00003.gguf". Both counters are exactly five digits in every
# llama.cpp-produced split, and anchoring on that is what lets a member be
# recognised without trusting the folder name.
SHARD_RE = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$"
)

# A sha256 in hex. Length is the whole point: the ladder accepts a value from
# HuggingFace only when it is exactly 64 hex characters, so a weak/opaque ETag
# can never be mistaken for a digest.
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The first four bytes of every GGUF file. Checking them costs one read and
# catches the most common real-world failure: an HTML error or sign-in page
# saved under a .gguf name.
GGUF_MAGIC = b"GGUF"

DEFAULT_API_BASE = "https://huggingface.co"
DEFAULT_ALLOWED_HOSTS = ("huggingface.co", "hf.co")
MAX_REDIRECT_HOPS = 5
CHUNK_BYTES = 1024 * 1024  # 1 MiB, matching the pre-existing CLI downloader
PROGRESS_INTERVAL_S = 0.5  # HF-1 sibling rule: at most one progress Result / 500ms

# SEC-M14-2: a transfer whose size is declared nowhere (no tree size AND no
# Content-Length) used to be unbounded - it wrote until the disk filled. Every
# download now carries a ceiling. When the size is unknown and free space cannot
# be measured, this backstop applies: far above any single GGUF file offered
# today, far below "fill the volume". Exceeding it deletes the partial file.
MAX_UNDECLARED_DOWNLOAD_BYTES = 64 * 1024 * 1024 * 1024  # 64 GiB

# Verification rungs (M14.14.4). Stored in the registry row's notes and shown in
# the UI. "none" is a permanent label, never upgraded after the fact.
V_API = "api"
V_ETAG = "etag"
V_NONE = "none"
# A fourth rung that the GUI can never produce: it exists only for the
# owner-run CLI's --url form, where the digest comes from the operator's command
# line. The file really is checked against that digest, but calling it
# "verified against HuggingFace" would be a lie - LOCITIZE never obtained it from
# HuggingFace and cannot vouch for where the operator got it.
V_OPERATOR = "operator"

# The end-user labels for each rung (UX Spec section 14, final wording).
#
# Two audiences, two levels of precision, and both rules hold at once:
#   - Engineer-facing text (this comment, the module docstring, the registry's
#     notes field) names the literal field a claim rests on: `lfs.oid` for
#     V-API, the `X-Linked-ETag` header for V-ETAG, the digest passed to
#     `scripts/fetch_model.py --url` for V-OPERATOR.
#   - The user-facing strings below name the MECHANISM in plain language and
#     must never contain the bare word "ETag". HuggingFace overloads that word
#     to mean both the safe LFS value (X-Linked-ETag) and the Xet content hash
#     on the plain `etag` header, so a reader cannot tell which one LOCITIZE
#     checked - the ambiguity lives in the words, not in the code.
#
# These are the single source of truth for the completion line and the file-row
# label; views read this dict rather than restating the strings (a duplicated
# string is how the two surfaces drifted apart in the first place).
VERIFICATION_LABELS = {
    # Backed by the tree API's `lfs.oid` field.
    V_API: "Checksum verified against HuggingFace's file listing.",
    # Backed by the `X-Linked-ETag` header seen on huggingface.co's 302 hop.
    V_ETAG: "Checksum verified against HuggingFace's linked file hash.",
    # No publisher digest of any kind existed. The word "verified" appears only
    # as part of "Not verified" - never as a positive claim.
    V_NONE: "Not verified - no publisher checksum available.",
    # Backed by the digest the operator typed after `fetch_model.py --url`.
    # Never says "against HuggingFace": LOCITIZE did not obtain this digest from
    # HuggingFace and cannot vouch for where the operator got it.
    V_OPERATOR: "Checksum verified against the digest you supplied.",
}

# The V-NONE consent text. Kept as a constant so the GUI, the CLI and the tests
# all show the user the same sentence.
UNVERIFIED_CONSENT_TEXT = (
    "HuggingFace did not publish a checksum for this file. LOCITIZE will download "
    "it and record the checksum it computed, but it cannot verify the file is "
    "the one the publisher intended."
)

# T5. A 401 from an anonymous request means "gated OR does not exist" - recorded
# fact: HuggingFace answers a nonexistent repo with 401 "Invalid username or
# password." too, so LOCITIZE must not claim to know which. No credential support is
# offered, by design.
GATED_MESSAGE = (
    "HuggingFace refused this repository for an anonymous request. It is either "
    "gated behind a licence you must accept on huggingface.co, or it does not "
    "exist. LOCITIZE does not store HuggingFace credentials. If it is gated, "
    "download the file in your browser and point LOCITIZE at it with Register."
)

RATE_LIMIT_MESSAGE = (
    "HuggingFace is rate-limiting this connection. Try again in a minute."
)

FIT_DISCLAIMER = (
    "This is an estimate, not a guarantee. Actual use depends on context size, "
    "KV cache settings, and whatever else is using the GPU."
)

FIT_UNKNOWN_TEXT = "No NVIDIA GPU detected - LOCITIZE cannot estimate the fit."

MULTI_GPU_CAVEAT = (
    "You have more than one GPU. This estimate uses your largest single card, "
    "so it is conservative for a split setup."
)

FIT_WORDING = {
    "fits": "Should fit on your GPU",
    "tight": "Tight fit - may need a smaller context size",
    "exceeds": (
        "Larger than your VRAM. It will still run, partly on CPU, and will be slow."
    ),
    "unknown": FIT_UNKNOWN_TEXT,
}

# Registry defaults for a downloaded row, matching the values M13's Register uses.
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_GPU_LAYERS = 999


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class HubError(Exception):
    """A network/protocol failure, carrying a kind the UI renders differently.

    `kind` is one of: offline, rate_limited, gated, http, invalid, refused. The
    GUI maps it to a state (offline keeps the catalog browsable, rate_limited
    never auto-retries); `str(exc)` is the sentence shown to the user verbatim.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# --------------------------------------------------------------------------- #
# T1 - validation, then URL construction
# --------------------------------------------------------------------------- #


def validate_repo_id(repo_id: Any) -> str:
    """Return a repo id that is safe to interpolate, or raise ValueError."""
    text = str(repo_id or "").strip()
    if not REPO_ID_RE.match(text):
        raise ValueError(
            f"'{text}' is not a valid HuggingFace repository id; expected the "
            f"form owner/name using letters, digits, '.', '_' and '-' only"
        )
    return text


def validate_filename(filename: Any) -> str:
    """Return a file name that is safe as a URL segment and a path leaf, or raise."""
    text = str(filename or "").strip()
    if not FILENAME_RE.match(text):
        raise ValueError(
            f"'{text}' is not a downloadable model file name; LOCITIZE downloads "
            f"only plain .gguf names with no folder part"
        )
    return text


def validate_repo_path(path: Any) -> str:
    """Return a repo-relative path safe as URL segments and as a path tail, or raise.

    Accepts either a plain file name or exactly one folder part in front of it.
    Every segment is validated independently against the same charset
    validate_filename uses, so this is strictly a depth extension and never a
    charset relaxation. A backslash is REFUSED rather than normalised to '/':
    quietly rewriting a separator is how a validator ends up disagreeing with the
    filesystem that later opens the path.
    """
    text = str(path or "").strip()
    if not text or len(text) > MAX_REPO_PATH_CHARS:
        raise ValueError(f"'{text}' is not a usable model file path")
    if "\\" in text:
        raise ValueError(f"'{text}' contains a backslash; use '/' between folders")

    segments = text.split("/")
    if not 1 <= len(segments) <= MAX_REPO_PATH_DEPTH:
        raise ValueError(
            f"'{text}' must be a file name or '<folder>/<file>' "
            f"(at most {MAX_REPO_PATH_DEPTH} parts)"
        )
    for segment in segments[:-1]:
        if not _PATH_SEGMENT_RE.match(segment):
            raise ValueError(f"'{segment}' is not a usable folder name")
    # The leaf is held to the existing file rule, so a path can never smuggle in
    # a name that a plain download would have refused.
    validate_filename(segments[-1])
    return text


def parse_shard(name: str) -> tuple[str, int, int] | None:
    """Return (stem, index, total) for a shard member, or None when not sharded.

    Works on a bare file name or a path; only the leaf is inspected, because the
    folder tells you nothing reliable about membership.
    """
    leaf = str(name or "").rsplit("/", 1)[-1]
    match = SHARD_RE.match(leaf)
    if not match:
        return None
    index = int(match.group("index"))
    total = int(match.group("total"))
    if index < 1 or total < 1 or index > total:
        return None
    return match.group("stem"), index, total


def shard_set_for(paths: Iterable[str], member: str) -> list[str]:
    """Return every path belonging to `member`'s shard set, ordered by index.

    Pure: it filters a tree listing the caller already fetched. Returns [] when
    `member` is not a shard, and raises ValueError when the set is incomplete -
    a half-set is not a smaller download, it is an unloadable model, and finding
    that out before the first byte is the entire point of checking here.
    """
    parsed = parse_shard(member)
    if parsed is None:
        return []
    stem, _index, total = parsed
    folder = member.rsplit("/", 1)[0] if "/" in member else ""

    found: dict[int, str] = {}
    for candidate in paths:
        text = str(candidate)
        candidate_folder = text.rsplit("/", 1)[0] if "/" in text else ""
        if candidate_folder != folder:
            continue
        info = parse_shard(text)
        if info is None or info[0] != stem or info[2] != total:
            continue
        found[info[1]] = text

    missing = [n for n in range(1, total + 1) if n not in found]
    if missing:
        raise ValueError(
            f"shard set '{stem}' is incomplete: {len(found)} of {total} parts "
            f"present, missing index {missing[0]:05d}"
        )
    return [found[n] for n in range(1, total + 1)]


def shard_set_bytes(rows: Iterable[dict[str, Any]], members: Sequence[str]) -> int:
    """Total declared size of a shard set, so a caller can state the real number."""
    sizes = {
        str(row.get("filename", "")): int(row.get("size_bytes") or 0) for row in rows
    }
    return sum(sizes.get(name, 0) for name in members)


def _api_root(api_base: str) -> str:
    """Normalise the configured base URL and refuse anything but https."""
    base = str(api_base or DEFAULT_API_BASE).strip().rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(
            f"models_hub.api_base must be an https URL (got '{base or 'empty'}')"
        )
    return base


def build_search_url(api_base: str, query: str, limit: int = 25) -> str:
    """Build the model-search URL. The query is percent-encoded, never inlined."""
    root = _api_root(api_base)
    params = urllib.parse.urlencode(
        {
            "search": str(query or "").strip(),
            "filter": "gguf",
            "sort": "downloads",
            "direction": -1,
            "limit": max(1, int(limit)),
        }
    )
    return f"{root}/api/models?{params}"


_LINK_REL_RE = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?', re.IGNORECASE)


def _parse_link_next(link_header: str | None) -> str | None:
    """Extract the rel="next" URL from an RFC 5988 Link header, or None.

    huggingface.co's /api/models sends e.g.:
      Link: <https://huggingface.co/api/models?...&cursor=...>; rel="next"
    A header with no next relation (last page, or endpoint that ignores
    pagination entirely) returns None - "no more results", never an error.
    """
    if not link_header:
        return None
    match = _LINK_REL_RE.search(link_header)
    return match.group(1) if match else None


def build_tree_url(api_base: str, repo_id: str) -> str:
    """Build the file-listing URL for one validated repository."""
    return (
        f"{_api_root(api_base)}/api/models/{validate_repo_id(repo_id)}"
        f"/tree/main?recursive=true"
    )


def build_resolve_url(api_base: str, repo_id: str, filename: str) -> str:
    """Build the direct download URL for one validated file in one validated repo."""
    return (
        f"{_api_root(api_base)}/{validate_repo_id(repo_id)}"
        f"/resolve/main/{validate_filename(filename)}"
    )


def build_resolve_path_url(api_base: str, repo_id: str, path: str) -> str:
    """Download URL for a validated repo-relative PATH (M15.2, shard sets).

    Separate entry point rather than a looser build_resolve_url, so the plain
    single-file path keeps its narrower validator and nothing that reaches URL
    construction through the old door gains a folder part.
    """
    return (
        f"{_api_root(api_base)}/{validate_repo_id(repo_id)}"
        f"/resolve/main/{validate_repo_path(path)}"
    )


# --------------------------------------------------------------------------- #
# T2 - host allowlist, enforced on every hop
# --------------------------------------------------------------------------- #


def host_allowed(host: str, allowed_hosts: Iterable[str]) -> bool:
    """True when `host` equals, or is a subdomain of, an allowlisted host.

    The subdomain rule is load-bearing and was confirmed against a real download:
    huggingface.co redirects file downloads to `us.aws.cdn.hf.co`, a subdomain of
    the allowlisted `hf.co`. Matching bare equality would break every real
    download; matching a substring would let `evil-hf.co` through. Hence the
    explicit "equal, or ends with '.' + allowed" test.
    """
    name = (host or "").strip().lower().rstrip(".")
    if not name:
        return False
    for entry in allowed_hosts:
        allowed = (entry or "").strip().lower().rstrip(".")
        if not allowed:
            continue
        if name == allowed or name.endswith("." + allowed):
            return True
    return False


def validate_url(url: str, allowed_hosts: Iterable[str]) -> str:
    """Return `url` if it is https and on an allowlisted host; raise otherwise."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() != "https":
        raise HubError(
            "refused",
            f"refusing a non-https URL (scheme '{parsed.scheme or 'none'}'); "
            f"model files are only fetched over TLS",
        )
    if "@" in (parsed.netloc or ""):
        raise HubError("refused", "refusing a URL that carries userinfo in its host")
    if not host_allowed(parsed.hostname or "", allowed_hosts):
        raise HubError(
            "refused",
            f"refusing a request to '{parsed.hostname or 'unknown host'}': it is "
            f"not on the allowed host list ({', '.join(allowed_hosts)})",
        )
    return url


def open_checked(
    opener: Any,
    target: Any,
    allowed_hosts: Iterable[str],
    timeout: float | None = None,
    handler: Any = None,
) -> Any:
    """The ONE way this module is allowed to put a request on the wire.

    Every outbound request in modelhub goes through here, so the T2 allowlist is
    enforced by construction rather than by each call site remembering to ask.
    That matters: `_probe_linked_digest` previously opened its HEAD directly and
    so escaped the gate entirely whenever `models_hub.api_base` pointed off the
    allowlist (it is settable in the user's own settings.yaml; the environment
    override that also set it was removed by DEC-M14-10), which produced real
    egress to an unapproved host before anything refused. Validating inside the opening helper
    means a future call site cannot reintroduce that bypass by forgetting a line.

    `target` may be a URL string or a urllib Request; both are checked by their
    final URL. Raises HubError (never opens) when validation fails.
    """
    url = getattr(target, "full_url", None) or str(target)
    validate_url(url, allowed_hosts)
    # Each request gets its own redirect budget. The handler instance is reused
    # across the HEAD probe and the GET, and its counter only ever incremented,
    # so without this reset a 3-hop probe would eat most of the GET's budget and
    # a legitimate download would fail with a refusal it did not earn.
    reset = getattr(handler, "reset_hops", None)
    if callable(reset):
        reset()
    # M17.1 privacy ledger: this is THE chokepoint every outbound request in the
    # codebase passes through, so recording here captures all egress by
    # construction. Logged before the socket opens; never raises (see egress_log).
    try:
        import egress_log

        egress_log.record(url)
    except Exception:  # noqa: BLE001 - a witness, never a gate
        pass
    return opener.open(target, timeout=timeout)


class _AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate scheme and host on EVERY redirect hop, and cap the chain.

    urllib follows redirects transparently, so validating only the URL we asked
    for would leave the actual bytes coming from wherever the server pointed. We
    return None (which urllib turns into an HTTPError, not a silent follow) for a
    hop that fails validation or exceeds the hop budget.

    It also has a second job: recording `X-Linked-ETag` off the 302 response.
    That header carries the file's real sha256 and is present ONLY on the hop, so
    this handler is the single point in the process where rung V-ETAG's value can
    be observed at all.
    """

    def __init__(
        self, allowed_hosts: Sequence[str], max_hops: int = MAX_REDIRECT_HOPS
    ) -> None:
        super().__init__()
        self.allowed_hosts = tuple(allowed_hosts)
        self.max_hops = int(max_hops)
        self.hops = 0
        self.linked_etag: str | None = None
        self.linked_size: int | None = None

    def reset_hops(self) -> None:
        """Start a fresh hop budget. Called by `open_checked` per request.

        The recorded digest is deliberately NOT cleared: the HEAD probe exists
        precisely to capture it for the GET that follows on the same handler.
        """
        self.hops = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        self.hops += 1
        if self.hops > self.max_hops:
            # Refusing rather than following is the safe direction: a redirect
            # loop that ends somewhere unvalidated is exactly the risk here.
            return None
        self._record_linked_headers(headers)
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme.lower() != "https":
            return None
        if "@" in (parsed.netloc or ""):
            return None
        if not host_allowed(parsed.hostname or "", self.allowed_hosts):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def _record_linked_headers(self, headers: Any) -> None:
        """Stash the sha256 and size HuggingFace publishes on the redirect."""
        try:
            raw = headers.get("X-Linked-ETag")
            raw_size = headers.get("X-Linked-Size")
        except AttributeError:  # a mapping-like stub in a test
            raw = headers.get("X-Linked-ETag") if isinstance(headers, dict) else None
            raw_size = headers.get("X-Linked-Size") if isinstance(headers, dict) else None
        candidate = normalize_etag_digest(raw)
        if candidate:
            self.linked_etag = candidate
        if raw_size is not None:
            try:
                self.linked_size = int(str(raw_size).strip())
            except (TypeError, ValueError):
                self.linked_size = None


def normalize_etag_digest(raw: Any) -> str | None:
    """Return a 64-hex sha256 from an ETag-style header value, or None.

    Strict on purpose. A weak validator (`W/"..."`), a short opaque tag, or an
    upper-case-with-junk value all return None, because the alternative - being
    lenient - means comparing a downloaded file against a value that is not its
    digest and then telling the user it is "verified".
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower().startswith("w/"):
        return None  # weak validator: explicitly not a content digest
    text = text.strip('"').strip().lower()
    return text if SHA256_RE.match(text) else None


def make_opener(
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS,
    max_hops: int = MAX_REDIRECT_HOPS,
) -> tuple[urllib.request.OpenerDirector, _AllowlistRedirectHandler]:
    """Build a private opener plus the handler that guards and observes redirects.

    Deliberately NOT `urllib.request.urlopen`, which uses a shared global opener
    whose handlers any other module could have replaced. A private opener means
    the allowlist is guaranteed to be in the chain for every request this module
    makes.
    """
    handler = _AllowlistRedirectHandler(allowed_hosts, max_hops)
    return urllib.request.build_opener(handler), handler


# --------------------------------------------------------------------------- #
# Parsing real HuggingFace payloads
# --------------------------------------------------------------------------- #

# Quant labels as they appear in real GGUF file names, longest first so that
# "Q4_K_M" wins over a hypothetical "Q4" prefix match.
_QUANT_PATTERN = re.compile(
    r"(IQ\d+_[A-Z0-9_]+|Q\d+_K_[A-Z]+|Q\d+_K|Q\d+_\d+|Q\d+|BF16|F16|F32)",
    re.IGNORECASE,
)


def quant_from_filename(filename: str) -> str:
    """Pull the quantisation label out of a GGUF file name ('' when absent)."""
    match = _QUANT_PATTERN.search(str(filename or ""))
    return match.group(1).upper() if match else ""


# Long enough for every SPDX id and HuggingFace licence tag in use ("apache-2.0",
# "cc-by-nc-sa-4.0", "bigscience-openrail-m"); short enough that a tag cannot push
# the rest of a dialog off the screen.
LICENSE_TAG_MAX_LEN = 64

# A licence identifier has a narrow, known shape. This matches the WHOLE value,
# because the rule is accept-or-drop rather than repair: SEC-M14-1 showed that a
# publisher-authored tag reaching a renderer intact is enough to forge LOCITIZE's own
# verification wording, and a partially-scrubbed string still carries the
# attacker's words. Anything that is not licence-id-shaped is treated as if the
# repo declared nothing, which the UI already renders honestly as "not stated".
_LICENSE_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._+-]*\Z")


def sanitize_license_tag(raw: Any) -> str:
    """Return the licence id if it is licence-id-shaped, otherwise "".

    Untrusted input from the model card, so the check is an allowlist over the
    whole value: letters, digits, spaces and `. _ + -` only, no control
    characters or line breaks, and at most LICENSE_TAG_MAX_LEN characters.
    """
    text = str(raw or "").strip()
    if not text or len(text) > LICENSE_TAG_MAX_LEN:
        return ""
    return text if _LICENSE_TAG_RE.match(text) else ""


def license_tag_from_tags(tags: Iterable[Any]) -> str:
    """Extract the publisher's licence id from HuggingFace's flat `tags` list.

    Recorded fact: the search API has no `license` field. The licence arrives as
    a tag string like "license:apache-2.0", so this is where it has to come from.
    Returns "" when the repo declares none - shown as "not stated", never guessed.

    The value is publisher-controlled free text, so it is sanitised here (SEC-M14-1)
    before any caller can render it.
    """
    for tag in tags or []:
        text = str(tag)
        if text.lower().startswith("license:"):
            return sanitize_license_tag(text.split(":", 1)[1])
    return ""


def parse_search_results(payload: Any) -> list[dict[str, Any]]:
    """Turn a real /api/models response into the rows the UI lists."""
    rows: list[dict[str, Any]] = []
    for entry in payload or []:
        if not isinstance(entry, dict):
            continue
        repo_id = str(entry.get("id") or entry.get("modelId") or "").strip()
        if not REPO_ID_RE.match(repo_id):
            # Anything that would not survive validation is dropped here rather
            # than offered and refused later.
            continue
        rows.append(
            {
                "repo_id": repo_id,
                "display_name": repo_id.split("/", 1)[1],
                "publisher": repo_id.split("/", 1)[0],
                "license_tag": license_tag_from_tags(entry.get("tags")),
                "downloads": int(entry.get("downloads") or 0),
                "likes": int(entry.get("likes") or 0),
                "source": "api",
            }
        )
    return rows


def parse_tree_files(payload: Any) -> list[dict[str, Any]]:
    """Turn a real tree response into downloadable GGUF rows with their digests.

    Rung V-API lives here: `lfs.oid` on a GGUF entry is its sha256 (verified
    against a live response, and cross-checked against the same file's
    X-Linked-ETag, which matched exactly). An entry with no usable `lfs.oid`
    comes back as verification "none" and may still be upgraded to "etag" later,
    when the download's redirect exposes X-Linked-ETag.
    """
    rows: list[dict[str, Any]] = []
    for entry in payload or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type", "file")) != "file":
            continue
        path = str(entry.get("path") or "")
        if not FILENAME_RE.match(path):
            continue  # subdirectory or non-gguf; see FILENAME_RE's note
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
        digest = normalize_etag_digest(lfs.get("oid")) if lfs else None
        size = lfs.get("size") if lfs else None
        if size is None:
            size = entry.get("size")
        try:
            size_bytes = int(size)
        except (TypeError, ValueError):
            size_bytes = 0
        rows.append(
            {
                "filename": path,
                "quant": quant_from_filename(path),
                "size_bytes": size_bytes,
                "sha256": digest,
                "verification": V_API if digest else V_NONE,
            }
        )
    rows.sort(key=lambda row: row["size_bytes"])
    return rows


def parse_tree_paths(payload: Any) -> list[dict[str, Any]]:
    """Like parse_tree_files, but keeps GGUFs that live one folder deep (M15.2).

    Used only by the shard-set flow. Rows carry the same shape, so every
    downstream consumer (digest ladder, size, quant label) is unchanged; the one
    difference is that `filename` may contain a single validated folder part.
    """
    rows: list[dict[str, Any]] = []
    for entry in payload or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type", "file")) != "file":
            continue
        path = str(entry.get("path") or "")
        try:
            validate_repo_path(path)
        except ValueError:
            continue  # too deep, wrong charset, or not a .gguf
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
        digest = normalize_etag_digest(lfs.get("oid")) if lfs else None
        size = lfs.get("size") if lfs else None
        if size is None:
            size = entry.get("size")
        try:
            size_bytes = int(size)
        except (TypeError, ValueError):
            size_bytes = 0
        rows.append(
            {
                "filename": path,
                "quant": quant_from_filename(path),
                "size_bytes": size_bytes,
                "sha256": digest,
                "verification": V_API if digest else V_NONE,
            }
        )
    rows.sort(key=lambda row: row["size_bytes"])
    return rows


def collapse_shard_sets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold shard members into one logical row per set (M15.5, GUI support).

    Single files pass through untouched. A COMPLETE shard set becomes one row
    that looks like a file the rest of the hub already understands: filename is
    the part-one path (what llama.cpp opens and what download() dispatches on),
    size_bytes is the WHOLE set (the number every fit estimate and consent
    dialog must show), shard_parts is the member count, and verification is
    V_API only when every part carries a digest - one unverifiable part makes
    the whole set unverified, because the set is one model.

    Members of an INCOMPLETE set are dropped entirely, the same treatment
    subdirectory files got before M15.2: a set that cannot be downloaded is not
    offered for download. sha256 is None on a set row - per-part digests are
    re-resolved from the tree at download time and enforced per part.
    """
    singles: list[dict[str, Any]] = []
    names = [str(r.get("filename", "")) for r in rows]
    by_name = {str(r.get("filename", "")): r for r in rows}
    seen_sets: set[str] = set()
    sets: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("filename", ""))
        parsed = parse_shard(name)
        if parsed is None:
            singles.append(row)
            continue
        stem, _index, _total = parsed
        folder = name.rsplit("/", 1)[0] if "/" in name else ""
        key = folder + "//" + stem
        if key in seen_sets:
            continue
        seen_sets.add(key)
        try:
            members = shard_set_for(names, name)
        except ValueError:
            continue  # incomplete: not offered
        member_rows = [by_name[m] for m in members]
        all_verified = all(r.get("sha256") for r in member_rows)
        sets.append(
            {
                "filename": members[0],
                "quant": quant_from_filename(folder or members[0]) or folder,
                "size_bytes": sum(int(r.get("size_bytes") or 0) for r in member_rows),
                "sha256": None,
                "verification": V_API if all_verified else V_NONE,
                "shard_parts": len(members),
            }
        )
    out = singles + sets
    out.sort(key=lambda row: row["size_bytes"])
    return out


def resolve_shard_destination(models_dir: Path | str, path: str) -> Path:
    """The one legal on-disk location for a shard-set member.

    A set is kept in its own folder under the models directory, named for the
    quant folder it came from, because llama.cpp finds sibling parts by scanning
    the directory that holds part one. Flattening the set into the shared models
    directory would work for exactly one model and then collide.

    Both gates from resolve_destination still apply: the path is validated
    segment by segment before it is joined, and the joined result is re-checked
    against the resolved root so a symlinked models directory cannot redirect it.
    """
    safe = validate_repo_path(path)
    root = Path(models_dir).expanduser().resolve()
    candidate = (root / safe).resolve()
    if not _is_inside(candidate, root):
        raise ValueError(
            f"refusing to write outside the models directory: '{candidate}' is "
            f"not inside '{root}'"
        )
    return candidate


# --------------------------------------------------------------------------- #
# The shipped catalog (offline discovery, read from disk, never registry data)
# --------------------------------------------------------------------------- #


def default_catalog_path(install_dir: Path | str | None = None) -> Path:
    """Where model_catalog.json lives when models_hub.catalog_path is blank."""
    root = Path(install_dir) if install_dir else Path(__file__).resolve().parent
    return root / "model_catalog.json"


def load_catalog(path: Path | str | None = None) -> dict[str, Any]:
    """Read the shipped catalog from disk. Never a network call, never registry data.

    Returns {"items": [...], "reason": ""}. A missing or malformed file is a
    reported reason with an empty list - the Models page still works, it just has
    nothing curated to show, which is the truth in that situation.
    """
    target = Path(path) if path else default_catalog_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"items": [], "reason": f"no catalog file at {target}"}
    except (OSError, ValueError) as exc:
        return {"items": [], "reason": f"could not read {target.name}: {exc}"}
    entries = raw.get("models") if isinstance(raw, dict) else raw
    items: list[dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        repo_id = str(entry.get("repo_id", "")).strip()
        if not REPO_ID_RE.match(repo_id):
            continue
        items.append(
            {
                "repo_id": repo_id,
                "display_name": str(entry.get("display_name", repo_id)),
                "publisher": str(entry.get("publisher", repo_id.split("/", 1)[0])),
                "license_tag": str(entry.get("license_tag", "")),
                "params_b": entry.get("params_b"),
                "notes": str(entry.get("notes", "")),
                "files": [
                    {
                        "filename": str(f.get("filename", "")),
                        "quant": str(f.get("quant", "")),
                        # Explicitly a HINT. It is replaced by the live size
                        # before any confirm dialog shows a number the user acts
                        # on, so a stale catalog can never mis-state a download.
                        "size_bytes_hint": f.get("size_bytes_hint"),
                        # M15.2: number of parts when this row names a shard set,
                        # 0 for an ordinary single file. Carried through here
                        # because a consumer that loses it would treat part one
                        # as the whole model and fetch an unloadable fragment.
                        # For a set, size_bytes_hint is the WHOLE set, not part one.
                        "shard_parts": int(f.get("shard_parts") or 0),
                    }
                    for f in (entry.get("files") or [])
                    if isinstance(f, dict)
                ],
                "source": "catalog",
            }
        )
    return {"items": items, "reason": ""}


def merge_catalog_and_search(
    catalog_items: Sequence[dict[str, Any]], search_items: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Union the two discovery sources, with the LIVE source winning on conflict.

    A catalog row is a static claim that can go stale; a search row came off the
    Hub seconds ago. When both describe the same repo, the live one is what the
    user sees.
    """
    merged: dict[str, dict[str, Any]] = {}
    for item in catalog_items:
        merged[item["repo_id"]] = dict(item)
    for item in search_items:
        merged[item["repo_id"]] = dict(item)
    return sorted(merged.values(), key=lambda row: row["repo_id"].lower())


# --------------------------------------------------------------------------- #
# VRAM fit - guidance only, from the ONE existing GPU fact source
# --------------------------------------------------------------------------- #


def estimate_fit(size_bytes: int, gpus: Sequence[Any] | None) -> dict[str, Any]:
    """Estimate whether a GGUF of `size_bytes` fits, and show the arithmetic.

    `gpus` is exactly what health.GpuInfoProvider.gpus() returns - a list of
    GpuInfo or None. This module never runs a GPU query itself; there is one
    source of GPU truth in the product and it is health.py's provider.

    None (no GPU/driver) yields band "unknown" and makes NO numeric claim at all.
    Guessing a budget there would be inventing a fact about the user's machine.
    """
    weights_mb = float(size_bytes or 0) / 1024.0 / 1024.0
    # 12% of weights (floored at 512 MB) approximates KV cache plus compute
    # buffers at the 8192-wide context LOCITIZE registers a model with by default.
    # It is a rule of thumb, which is why the disclaimer below is mandatory.
    working_mb = max(512.0, weights_mb * 0.12)
    est_mb = weights_mb + working_mb
    if not gpus:
        return {
            "band": "unknown",
            "wording": FIT_WORDING["unknown"],
            "weights_mb": round(weights_mb, 1),
            "working_mb": round(working_mb, 1),
            "est_mb": round(est_mb, 1),
            "budget_mb": None,
            "gpu_name": "",
            "gpu_count": 0,
            "explanation": FIT_UNKNOWN_TEXT,
            "disclaimer": FIT_DISCLAIMER,
            "multi_gpu_caveat": "",
        }
    # Largest SINGLE card, never the sum: llama.cpp splits a model across cards
    # only when told to, so summing VRAM would over-promise on a default setup.
    best = max(gpus, key=lambda g: float(getattr(g, "vram_total_mb", 0.0)))
    budget_mb = float(getattr(best, "vram_total_mb", 0.0))
    if est_mb <= 0.85 * budget_mb:
        band = "fits"
    elif est_mb <= budget_mb:
        band = "tight"
    else:
        band = "exceeds"
    explanation = (
        f"{weights_mb / 1024:.1f} GB weights + ~{working_mb / 1024:.1f} GB working "
        f"memory vs {budget_mb / 1024:.1f} GB VRAM ({getattr(best, 'name', 'GPU')})"
    )
    return {
        "band": band,
        "wording": FIT_WORDING[band],
        "weights_mb": round(weights_mb, 1),
        "working_mb": round(working_mb, 1),
        "est_mb": round(est_mb, 1),
        "budget_mb": round(budget_mb, 1),
        "gpu_name": str(getattr(best, "name", "")),
        "gpu_count": len(gpus),
        "explanation": explanation,
        "disclaimer": FIT_DISCLAIMER,
        "multi_gpu_caveat": MULTI_GPU_CAVEAT if len(gpus) > 1 else "",
    }


# --------------------------------------------------------------------------- #
# T3 - destination confinement
# --------------------------------------------------------------------------- #


def resolve_destination(models_dir: Path | str, filename: str) -> Path:
    """Return the ONE legal destination path for `filename`, or raise ValueError.

    Two independent gates, because either alone is weaker than it looks: the
    filename regex (T1) stops traversal syntax getting in, and the
    resolve()-then-is_relative_to check stops a symlinked models directory
    landing the file somewhere else even when the name is innocent.
    """
    safe_name = validate_filename(filename)
    root = Path(models_dir).expanduser().resolve()
    candidate = (root / safe_name).resolve()
    if not _is_inside(candidate, root):
        raise ValueError(
            f"refusing to write outside the models directory: '{candidate}' is "
            f"not inside '{root}'"
        )
    return candidate


def _is_inside(candidate: Path, root: Path) -> bool:
    """True when `candidate` is at or below `root` (Python 3.9-safe)."""
    try:
        return candidate == root or candidate.is_relative_to(root)
    except AttributeError:  # pragma: no cover - Python < 3.9
        return str(candidate).startswith(str(root) + os.sep)


# --------------------------------------------------------------------------- #
# The verified download core (shared by the GUI and scripts/fetch_model.py)
# --------------------------------------------------------------------------- #


@dataclass
class DownloadOutcome:
    """The single result shape both callers of download_verified read."""

    ok: bool
    path: Path | None = None
    sha256: str = ""
    expected_sha256: str = ""
    verification: str = V_NONE
    bytes_written: int = 0
    cancelled: bool = False
    error: str = ""


@dataclass
class Progress:
    """One progress sample handed to the caller's callback."""

    phase: str  # connecting / downloading / hashing / verifying / registering / done
    bytes_done: int = 0
    bytes_total: int | None = None
    rate_bps: float = 0.0
    eta_s: float | None = None


def download_verified(
    url: str,
    dest: Path | str,
    expected_sha256: str | None,
    *,
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS,
    opener: Any = None,
    handler: Any = None,
    verification: str = V_NONE,
    confirm_unverified: bool = False,
    expected_size: int | None = None,
    max_bytes: int | None = None,
    require_gguf_magic: bool = True,
    cancel_event: threading.Event | None = None,
    progress: Callable[[Progress], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    chunk_bytes: int = CHUNK_BYTES,
) -> DownloadOutcome:
    """Stream `url` to `dest`, verify it, and keep it ONLY if every check passes.

    This is the one implementation of the safety contract the product had before
    (stream to .partial, hash while writing, compare before rename, delete on
    mismatch, never clobber) plus the checks M14.14.3 adds (GGUF magic bytes,
    size agreement). scripts/fetch_model.py and the GUI both call this; there is
    no second downloader for model weights anywhere in the tree.

    `expected_sha256` None/empty means rung V-NONE, which requires
    `confirm_unverified=True`. The refusal is deliberate: a caller must have
    asked the user first, and a caller that forgot gets an error rather than a
    quietly unverified multi-gigabyte file.

    `handler` is the redirect handler paired with `opener`, passed only so this
    request starts with a full redirect budget of its own (see `open_checked`).

    `max_bytes` is the hard ceiling on what may be written (SEC-M14-2). The caller
    passes the tightest number it knows - the declared file size, or the free-space
    budget when the size is undeclared. None means "use the backstop constant";
    there is no unbounded mode.
    """
    dest_path = Path(dest)
    expected = (expected_sha256 or "").strip().lower()
    if expected and not SHA256_RE.match(expected):
        return DownloadOutcome(
            False, error=f"expected sha256 '{expected}' is not 64 hex characters"
        )
    if not expected and not confirm_unverified:
        return DownloadOutcome(
            False,
            verification=V_NONE,
            error=(
                "no publisher checksum is available for this file and the "
                "download was not confirmed as unverified. " + UNVERIFIED_CONSENT_TEXT
            ),
        )
    try:
        validate_url(url, allowed_hosts)
    except HubError as exc:
        return DownloadOutcome(False, error=str(exc))
    # T4: never overwrite a file LOCITIZE did not create. The user may legitimately
    # already have these weights, and silently replacing several gigabytes is not
    # a decision a download button gets to make.
    if dest_path.exists():
        return DownloadOutcome(
            False,
            path=dest_path,
            error=(
                f"a file already exists at {dest_path}. LOCITIZE will not overwrite "
                f"it. Use the existing file (Register it), or move it aside first."
            ),
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    partial = dest_path.with_suffix(dest_path.suffix + ".partial")
    hasher = hashlib.sha256()
    written = 0
    started = clock()
    last_emit = 0.0
    active_opener = opener if opener is not None else make_opener(allowed_hosts)[0]

    def emit(phase: str, force: bool = False) -> None:
        """Publish progress at most every 500 ms, or immediately on a phase change."""
        nonlocal last_emit
        if progress is None:
            return
        now = clock()
        if not force and (now - last_emit) < PROGRESS_INTERVAL_S:
            return
        last_emit = now
        elapsed = max(1e-6, now - started)
        rate = written / elapsed
        eta = None
        if total_bytes and rate > 0 and total_bytes > written:
            eta = (total_bytes - written) / rate
        progress(Progress(phase, written, total_bytes, rate, eta))

    total_bytes: int | None = expected_size
    emit("connecting", force=True)
    try:
        # open_checked re-validates the URL immediately before the socket opens,
        # so the gate cannot be separated from the request by a later edit.
        response = open_checked(
            active_opener, url, allowed_hosts, timeout=60, handler=handler
        )
    except HubError as exc:
        return DownloadOutcome(False, error=str(exc))
    except urllib.error.HTTPError as exc:
        return DownloadOutcome(False, error=_http_error_message(exc))
    except (urllib.error.URLError, OSError) as exc:
        return DownloadOutcome(
            False, error=f"could not reach the download URL: {exc}"
        )

    try:
        header_total = response.headers.get("Content-Length")
        if header_total is not None:
            try:
                total_bytes = int(str(header_total).strip())
            except (TypeError, ValueError):
                total_bytes = expected_size
        # The API-reported size and the transport's Content-Length must agree.
        # A disagreement means the bytes on the wire are not the file that was
        # described, which is a refusal, not a warning.
        if (
            expected_size
            and total_bytes
            and int(expected_size) != int(total_bytes)
        ):
            response.close()
            return DownloadOutcome(
                False,
                error=(
                    f"size mismatch before download: HuggingFace listed "
                    f"{expected_size} bytes but the server offered {total_bytes}"
                ),
            )
        # SEC-M14-2: settle the byte ceiling now that every size source has been
        # consulted. The tightest KNOWN bound wins; when nothing is known at all
        # (no tree size, no Content-Length, no caller budget) the backstop applies
        # so the loop below can never be unbounded.
        bounds = [
            value
            for value in (max_bytes, total_bytes, expected_size)
            if value is not None and int(value) > 0
        ]
        ceiling = min(int(value) for value in bounds) if bounds else (
            MAX_UNDECLARED_DOWNLOAD_BYTES
        )
        emit("downloading", force=True)
        with partial.open("wb") as handle:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    handle.close()
                    _safe_unlink(partial)
                    return DownloadOutcome(
                        False,
                        cancelled=True,
                        bytes_written=written,
                        error="cancelled by the user",
                    )
                chunk = response.read(chunk_bytes)
                if not chunk:
                    break
                handle.write(chunk)
                hasher.update(chunk)
                written += len(chunk)
                # SEC-M14-2: the ceiling is checked INSIDE the loop, because the
                # post-transfer size check cannot run when nothing declared a
                # size - by then the bytes are already on the disk.
                if written > ceiling:
                    handle.close()
                    _safe_unlink(partial)
                    return DownloadOutcome(
                        False,
                        bytes_written=written,
                        error=(
                            f"the download exceeded its size limit of {ceiling} "
                            f"bytes and was stopped. The incomplete file was "
                            f"deleted."
                        ),
                    )
                emit("downloading")
    except (urllib.error.URLError, OSError) as exc:
        _safe_unlink(partial)
        return DownloadOutcome(False, error=f"download error: {exc}")
    finally:
        try:
            response.close()
        except Exception:  # noqa: BLE001 - closing must never mask the real result
            pass

    emit("verifying", force=True)
    actual = hasher.hexdigest()

    # Integrity checks that run on EVERY rung, including the unverified one.
    if written == 0:
        _safe_unlink(partial)
        return DownloadOutcome(False, error="the download was empty (0 bytes)")
    if total_bytes is not None and written != total_bytes:
        _safe_unlink(partial)
        return DownloadOutcome(
            False,
            error=(
                f"size mismatch: expected {total_bytes} bytes, received {written}. "
                f"The incomplete download was deleted."
            ),
        )
    if require_gguf_magic:
        magic = _read_magic(partial)
        if magic != GGUF_MAGIC:
            _safe_unlink(partial)
            return DownloadOutcome(
                False,
                sha256=actual,
                error=(
                    "this file is not a GGUF model: it starts with "
                    f"{magic!r} instead of {GGUF_MAGIC!r}. That usually means an "
                    "error or sign-in page was served instead of the model. The "
                    "download was deleted."
                ),
            )
    if expected and actual != expected:
        _safe_unlink(partial)
        return DownloadOutcome(
            False,
            sha256=actual,
            expected_sha256=expected,
            error=(
                "checksum mismatch - the download was deleted (never left in "
                f"place).\n  expected: {expected}\n  actual:   {actual}\n"
                "Try again; if it repeats, the published file or the mirror is "
                "wrong."
            ),
        )

    partial.rename(dest_path)
    emit("done", force=True)
    return DownloadOutcome(
        True,
        path=dest_path,
        sha256=actual,
        expected_sha256=expected,
        # Honesty rule: the rung is "api"/"etag" only when there was a real
        # digest to compare against. Without one it stays "none" no matter what
        # the caller passed in.
        verification=verification if expected else V_NONE,
        bytes_written=written,
    )


def _read_magic(path: Path) -> bytes:
    """Read the first four bytes of a file, tolerating a short read."""
    try:
        with path.open("rb") as handle:
            return handle.read(4)
    except OSError:
        return b""


def _safe_unlink(path: Path) -> None:
    """Delete a file, ignoring the case where it is already gone."""
    try:
        path.unlink()
    except OSError:
        pass


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    """Turn an HTTP status into the sentence the user should actually read."""
    if exc.code in (401, 403):
        return GATED_MESSAGE
    if exc.code == 429:
        return RATE_LIMIT_MESSAGE
    if exc.code == 404:
        return "HuggingFace has no such file or repository (404)."
    return f"HuggingFace returned HTTP {exc.code} ({exc.reason})."


# --------------------------------------------------------------------------- #
# Downloader - the object the GUI and CLI drive
# --------------------------------------------------------------------------- #


@dataclass
class HubConfig:
    """The models_hub settings block, flattened to what this module needs."""

    enabled: bool = True
    api_base: str = DEFAULT_API_BASE
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    search_limit: int = 100
    search_timeout_s: int = 15
    catalog_path: str = ""
    download_dir: str = ""
    min_free_disk_headroom_mb: int = 2048
    resume_enabled: bool = False
    register_on_complete: bool = True

    @classmethod
    def from_settings(cls, settings: Any) -> "HubConfig":
        """Read a config.Settings without importing config (avoids a cycle)."""
        block = getattr(settings, "models_hub", None)
        if block is None:
            return cls()
        hosts = tuple(getattr(block, "allowed_hosts", DEFAULT_ALLOWED_HOSTS) or ())
        return cls(
            enabled=bool(getattr(block, "enabled", True)),
            api_base=str(getattr(block, "api_base", DEFAULT_API_BASE)),
            allowed_hosts=hosts or DEFAULT_ALLOWED_HOSTS,
            search_limit=int(getattr(block, "search_limit", 100)),
            search_timeout_s=int(getattr(block, "search_timeout_s", 15)),
            catalog_path=str(getattr(block, "catalog_path", "")),
            download_dir=str(getattr(block, "download_dir", "")),
            min_free_disk_headroom_mb=int(
                getattr(block, "min_free_disk_headroom_mb", 2048)
            ),
            resume_enabled=bool(getattr(block, "resume_enabled", False)),
            register_on_complete=bool(getattr(block, "register_on_complete", True)),
        )


class Downloader:
    """Search, list and fetch models. Constructing one makes NO network call.

    Every collaborator is injected so the object is fully exercisable offline:
    `opener_factory` returns the urllib opener (a test hands one that raises if
    opened at all - that is how egress rule HF-1 is proven mechanically),
    `gpu_provider` is health.GpuInfoProvider, `system_provider` is
    health.SystemInfoProvider, and `clock` drives progress throttling.
    """

    def __init__(
        self,
        config: HubConfig | None = None,
        *,
        models_dir: Path | str | None = None,
        opener_factory: Callable[[Sequence[str]], Any] | None = None,
        gpu_provider: Any = None,
        system_provider: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or HubConfig()
        self.models_dir = Path(models_dir) if models_dir else None
        self._opener_factory = opener_factory
        self._gpu_provider = gpu_provider
        self._system_provider = system_provider
        self._clock = clock
        # Session-only memory. No disk cache by design: nothing to go stale,
        # nothing to invalidate, nothing new under the data root to back up.
        self._last_search: list[dict[str, Any]] = []
        self._last_query: str = ""
        # The huggingface.co-issued cursor URL (from the Link: rel="next"
        # response header) for the next page of the current search, or None
        # when there is no further page / no search has run yet.
        self._next_url: str | None = None

    # -- offline surface (never touches the network) ---------------------- #

    def catalog(self) -> dict[str, Any]:
        """Read the shipped catalog from disk. Called on page paint; no egress."""
        path = self.config.catalog_path or None
        return load_catalog(path)

    def fit_for(self, size_bytes: int) -> dict[str, Any]:
        """VRAM guidance for a file size, from the injected GPU provider only."""
        gpus = None
        if self._gpu_provider is not None:
            gpus = self._gpu_provider.gpus()
        return estimate_fit(size_bytes, gpus)

    def check_disk(self, size_bytes: int, target_dir: Path | str) -> dict[str, Any]:
        """Confirm there is room for the file plus headroom, before the first byte.

        Returning a refusal here is the difference between an honest "not enough
        space" and a half-download that fills the user's disk.
        """
        needed_mb = float(size_bytes) / 1024.0 / 1024.0
        headroom = float(self.config.min_free_disk_headroom_mb)
        if self._system_provider is None:
            return {"ok": True, "free_mb": None, "needed_mb": needed_mb + headroom,
                    "reason": ""}
        # Second rung of the DEFECT-QA-M14-1 fix. download() now creates the
        # directory before calling this, so the missing-path case should be
        # impossible - but this method is public and is also reachable from
        # scripts, and an OS error here used to escape as a raw WinError with no
        # path and no remedy. A refusal that names the path and a next step is
        # the honest answer for a location that genuinely cannot be measured.
        try:
            free_mb = float(self._system_provider.disk_free_mb(str(target_dir)))
        except OSError as exc:
            return {
                "ok": False,
                "free_mb": None,
                "needed_mb": round(needed_mb + headroom, 1),
                "reason": (
                    f"could not check free space for '{target_dir}': {exc}. "
                    f"Make sure that folder exists and you can write to it, or "
                    f"set models_hub.download_dir in settings.yaml to another "
                    f"location, then press Download again."
                ),
            }
        ok = free_mb >= needed_mb + headroom
        return {
            "ok": ok,
            "free_mb": round(free_mb, 1),
            "needed_mb": round(needed_mb + headroom, 1),
            "reason": (
                ""
                if ok
                else (
                    f"not enough free space in {target_dir}: "
                    f"{free_mb / 1024:.1f} GB free, "
                    f"{(needed_mb + headroom) / 1024:.1f} GB needed "
                    f"(the model plus {headroom / 1024:.1f} GB headroom). "
                    f"Free up space on that drive, or set "
                    f"models_hub.download_dir in settings.yaml to a drive with "
                    f"more room, then press Download again."
                )
            ),
        }

    def _write_ceiling(self, size_bytes: int | None, disk: dict[str, Any]) -> int:
        """How many bytes this download may write before it is stopped.

        A declared size is the tightest honest bound. Without one, the bound is
        what the volume can spare (free space minus the same headroom check_disk
        reserves), so an undeclared-size transfer can never fill the user's disk.
        When free space cannot be measured (no system provider, e.g. in tests)
        the module backstop applies rather than "no limit".
        """
        if size_bytes:
            return int(size_bytes)
        free_mb = disk.get("free_mb")
        if free_mb is None:
            return MAX_UNDECLARED_DOWNLOAD_BYTES
        budget = (
            float(free_mb) - float(self.config.min_free_disk_headroom_mb)
        ) * 1024.0 * 1024.0
        # A non-positive budget cannot happen here (check_disk already refused),
        # but clamping keeps the ceiling a positive number under any provider.
        return max(int(budget), 1)

    # -- the three explicit-action methods (the ONLY egress in this module) - #

    def search(self, query: str) -> dict[str, Any]:
        """ONE network call, only ever from an explicit Search press (HF-1)."""
        import egress_log as _el; _el._reason.label = "hub-search"
        if not self.config.enabled:
            return {"ok": False, "query": query, "items": [], "has_more": False,
                    "reason": "models_hub.enabled is false in settings.yaml"}
        url = build_search_url(self.config.api_base, query, self.config.search_limit)
        try:
            payload, next_url = self._get_json_with_link(url)
        except HubError as exc:
            # Offline and rate-limited are first-class states: the caller keeps
            # showing the catalog and renders this reason verbatim.
            return {"ok": False, "query": query, "items": [], "has_more": False,
                    "reason": str(exc), "kind": exc.kind}
        self._last_search = parse_search_results(payload)
        self._last_query = query
        self._next_url = next_url
        return {"ok": True, "query": query, "items": self._last_search,
                "reason": "", "source": "api", "has_more": bool(next_url)}

    def search_more(self) -> dict[str, Any]:
        """Fetch the next page of the CURRENT search and append it (Load more).

        ONE network call, only ever from an explicit Load more press (HF-1) -
        the same invariant search() carries. The next-page URL is the cursor
        huggingface.co itself returned on the prior page's Link header, and is
        re-validated against the host allowlist like any other request
        (open_checked does this by construction). A no-op, never an error,
        when there is no prior search or no further page.
        """
        import egress_log as _el; _el._reason.label = "hub-search"
        if not self.config.enabled:
            return {"ok": False, "query": self._last_query, "items": self._last_search,
                    "has_more": False,
                    "reason": "models_hub.enabled is false in settings.yaml"}
        if not self._next_url:
            return {"ok": True, "query": self._last_query, "items": self._last_search,
                    "reason": "", "has_more": False}
        try:
            payload, next_url = self._get_json_with_link(self._next_url)
        except HubError as exc:
            return {"ok": False, "query": self._last_query, "items": self._last_search,
                    "has_more": True, "reason": str(exc), "kind": exc.kind}
        self._last_search = self._last_search + parse_search_results(payload)
        self._next_url = next_url
        return {"ok": True, "query": self._last_query, "items": self._last_search,
                "reason": "", "source": "api", "has_more": bool(next_url)}

    def list_files(self, repo_id: str) -> dict[str, Any]:
        """ONE network call, only ever from an explicit repo selection (HF-1)."""
        import egress_log as _el; _el._reason.label = "hub-list-files"
        if not self.config.enabled:
            return {"ok": False, "repo_id": repo_id, "items": [],
                    "reason": "models_hub.enabled is false in settings.yaml"}
        try:
            # M15.5/M15.10: build_tree_url has been recursive since its first
            # commit - the old top-level-only behaviour lived in the FILENAME
            # filter, not the URL. M15.5 misread that and appended a second
            # "?recursive=true", producing ".../tree/main?recursive=true?
            # recursive=true" - HTTP 400 on every repository, which broke the
            # whole Get models panel until the full claims audit listed a live
            # repo. The unit fakes accept any URL, so only a live call could
            # catch it; test_tree_url_has_exactly_one_query_string now pins the
            # shape.
            url = build_tree_url(self.config.api_base, repo_id)
        except ValueError as exc:
            return {"ok": False, "repo_id": repo_id, "items": [], "reason": str(exc)}
        try:
            payload = self._get_json(url)
        except HubError as exc:
            return {"ok": False, "repo_id": repo_id, "items": [],
                    "reason": str(exc), "kind": exc.kind}
        items = collapse_shard_sets(parse_tree_paths(payload))
        for item in items:
            item["fit"] = self.fit_for(item["size_bytes"])
            item["verification_label"] = VERIFICATION_LABELS[item["verification"]]
        return {"ok": True, "repo_id": repo_id, "items": items, "reason": ""}

    def download(
        self,
        repo_id: str,
        filename: str,
        *,
        expected_sha256: str | None = None,
        verification: str = V_NONE,
        size_bytes: int | None = None,
        confirm_unverified: bool = False,
        confirmed_exceeds: bool = False,
        models_dir: Path | str | None = None,
        cancel_event: threading.Event | None = None,
        progress: Callable[[Progress], None] | None = None,
    ) -> DownloadOutcome:
        """ONE download, only ever from an explicit Download press (HF-1).

        `confirmed_exceeds` is carried for the caller's benefit: the fit band
        never blocks a download here, it only requires the GUI to have collected
        a second confirmation first. This module does not second-guess a user who
        knowingly wants a model bigger than their VRAM.
        """
        import egress_log as _el; _el._reason.label = "download"
        if not self.config.enabled:
            return DownloadOutcome(
                False, error="models_hub.enabled is false in settings.yaml"
            )
        if parse_shard(filename) is not None:
            return self._download_shard_set(
                repo_id,
                filename,
                confirm_unverified=confirm_unverified,
                models_dir=models_dir,
                cancel_event=cancel_event,
                progress=progress,
            )
        try:
            repo_id = validate_repo_id(repo_id)
            filename = validate_filename(filename)
            target_dir = Path(models_dir or self.models_dir or self.config.download_dir)
            if not str(target_dir):
                return DownloadOutcome(False, error="no models directory configured")
            dest = resolve_destination(target_dir, filename)
        except ValueError as exc:
            return DownloadOutcome(False, error=str(exc))

        # DEFECT-QA-M14-1: the models directory must EXIST before anything asks
        # the OS about it. On a clean install nothing has created <data root>/
        # models yet, and the free-space probe below is the first code to touch
        # that path - shutil.disk_usage on a missing directory raises
        # FileNotFoundError (WinError 3), which reached the user as a bare
        # "unexpected error" and stopped the very first download of a new
        # install before a single byte moved. download_verified also creates the
        # parent, but that runs far too late to help this check.
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return DownloadOutcome(
                False,
                error=(
                    f"could not create the models folder '{target_dir}': {exc}. "
                    f"Check that you can write to that location, or set "
                    f"models_hub.download_dir in settings.yaml to a folder you "
                    f"own, then press Download again."
                ),
            )

        # SEC-M14-2: this check used to be skipped whenever the tree API reported
        # no size (size_bytes == 0 is falsy), which also left the transfer with no
        # ceiling. It now runs on EVERY download: with a declared size it answers
        # "is there room for this file", and with no declared size it still
        # answers "is there room at all" and yields the free-space budget used as
        # the write ceiling below.
        disk = self.check_disk(size_bytes or 0, target_dir)
        if not disk["ok"]:
            return DownloadOutcome(False, error=disk["reason"])
        max_bytes = self._write_ceiling(size_bytes, disk)

        # T2, before anything can reach the wire. `api_base` is user-settable in
        # settings.yaml (and only there, per DEC-M14-10), so the URL this
        # method builds is NOT trusted just because the repo id and file name
        # were validated. Checking here means an off-allowlist api_base refuses
        # without even constructing an opener, let alone opening a connection.
        try:
            url = validate_url(
                build_resolve_url(self.config.api_base, repo_id, filename),
                self.config.allowed_hosts,
            )
        except (HubError, ValueError) as exc:
            return DownloadOutcome(False, error=str(exc))

        opener, handler = self._build_opener()
        expected = (expected_sha256 or "").strip().lower()
        rung = verification if expected else V_NONE

        if not expected:
            # Rung V-ETAG: HuggingFace publishes the sha256 as X-Linked-ETag on
            # the 302 hop only, so a cheap HEAD through the redirect-recording
            # handler is the only way to obtain it before the transfer starts.
            probe = self._probe_linked_digest(url, opener, handler)
            if probe:
                expected, rung = probe, V_ETAG

        if not expected and not confirm_unverified:
            return DownloadOutcome(
                False, verification=V_NONE,
                error="unverified download not confirmed. " + UNVERIFIED_CONSENT_TEXT,
            )

        return download_verified(
            url,
            dest,
            expected or None,
            allowed_hosts=self.config.allowed_hosts,
            opener=opener,
            handler=handler,
            verification=rung,
            confirm_unverified=confirm_unverified,
            expected_size=size_bytes,
            max_bytes=max_bytes,
            cancel_event=cancel_event,
            progress=progress,
            clock=self._clock,
        )

    # -- internals --------------------------------------------------------- #

    def _download_shard_set(
        self,
        repo_id: str,
        member: str,
        *,
        confirm_unverified: bool,
        models_dir: Path | str | None,
        cancel_event: threading.Event | None,
        progress: Callable[[Progress], None] | None,
    ) -> DownloadOutcome:
        """One shard SET, from one explicit Download press (M15.5).

        The tree is re-fetched here rather than trusted from a stale listing:
        the set membership and every per-part digest are resolved at the moment
        of download, and an incomplete set refuses before the first byte. All
        transfer-level safety lives in download_shard_set / download_verified;
        this method only translates the hub vocabulary into that call and its
        ShardSetOutcome back into the DownloadOutcome every caller reads.
        """
        import egress_log as _el; _el._reason.label = "download"
        try:
            repo_id = validate_repo_id(repo_id)
            member = validate_repo_path(member)
            target_dir = Path(models_dir or self.models_dir or self.config.download_dir)
            if not str(target_dir):
                return DownloadOutcome(False, error="no models directory configured")
        except ValueError as exc:
            return DownloadOutcome(False, error=str(exc))
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return DownloadOutcome(
                False,
                error=(
                    "could not create the models folder '" + str(target_dir) + "': "
                    + str(exc)
                ),
            )
        try:
            # build_tree_url is already recursive; see list_files (M15.10).
            url = build_tree_url(self.config.api_base, repo_id)
            payload = self._get_json(url)
        except (HubError, ValueError) as exc:
            return DownloadOutcome(False, error=str(exc))
        rows = parse_tree_paths(payload)
        try:
            members = shard_set_for([r["filename"] for r in rows], member)
        except ValueError as exc:
            return DownloadOutcome(False, error=str(exc))
        if not members:
            return DownloadOutcome(
                False, error=f"'{member}' is not part of a multi-part set"
            )

        wanted = set(members)
        member_rows = [r for r in rows if r["filename"] in wanted]
        total = shard_set_bytes(rows, members)
        unverified = [r["filename"] for r in member_rows if not r.get("sha256")]
        if unverified and not confirm_unverified:
            return DownloadOutcome(
                False,
                verification=V_NONE,
                error=(
                    f"{len(unverified)} of {len(members)} parts publish no "
                    f"checksum. " + UNVERIFIED_CONSENT_TEXT
                ),
            )
        disk = self.check_disk(total, target_dir)
        if not disk["ok"]:
            return DownloadOutcome(False, error=disk["reason"])

        outcome = download_shard_set(
            self.config.api_base,
            repo_id,
            members,
            member_rows,
            target_dir,
            allowed_hosts=self.config.allowed_hosts,
            confirm_unverified=confirm_unverified,
            cancel_event=cancel_event,
            progress=progress,
            # check_disk speaks MB; a None/unknown free space means no whole-set
            # pre-check here (each part still carries its own max_bytes ceiling).
            free_bytes=(
                int(disk["free_mb"] * 1024 * 1024)
                if disk.get("free_mb") is not None
                else None
            ),
        )
        return DownloadOutcome(
            ok=outcome.ok,
            path=outcome.load_path if outcome.ok else None,
            verification=V_API if not unverified else V_NONE,
            bytes_written=outcome.bytes_written,
            cancelled=outcome.cancelled,
            error=outcome.error,
        )

    def _build_opener(self) -> tuple[Any, Any]:
        """Return (opener, redirect handler). Building one opens no connection."""
        if self._opener_factory is not None:
            built = self._opener_factory(self.config.allowed_hosts)
            if isinstance(built, tuple):
                return built
            return built, None
        return make_opener(self.config.allowed_hosts)

    def _get_json(self, url: str) -> Any:
        """GET one JSON document, mapping every failure onto a HubError kind."""
        payload, _next_url = self._get_json_with_link(url)
        return payload

    def _get_json_with_link(self, url: str) -> tuple[Any, str | None]:
        """GET one JSON document plus the Link: rel="next" URL, if any.

        huggingface.co's /api/models paginates via a cursor URL in the Link
        response header (GitHub-style), not an offset/skip parameter - this is
        the one place that header is read. Every failure maps onto a HubError
        kind, same as _get_json.
        """
        opener, handler = self._build_opener()
        try:
            with open_checked(
                opener,
                url,
                self.config.allowed_hosts,
                timeout=self.config.search_timeout_s,
                handler=handler,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
                next_url = _parse_link_next(response.headers.get("Link"))
                return body, next_url
        except urllib.error.HTTPError as exc:
            kind = {401: "gated", 403: "gated", 429: "rate_limited"}.get(
                exc.code, "http"
            )
            # No retry on 429, ever: auto-retrying a rate limit is how a rate
            # limit becomes a ban.
            raise HubError(kind, _http_error_message(exc)) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise HubError(
                "offline",
                f"Could not reach huggingface.co: {exc}. The list below is "
                f"LOCITIZE's built-in catalog; downloads still need a connection.",
            ) from exc
        except ValueError as exc:
            raise HubError(
                "http", f"HuggingFace returned something that is not JSON: {exc}"
            ) from exc

    def _probe_linked_digest(self, url: str, opener: Any, handler: Any) -> str | None:
        """HEAD the resolve URL so the redirect handler can capture the sha256."""
        if handler is None:
            return None
        try:
            request = urllib.request.Request(url, method="HEAD")
            with open_checked(
                opener,
                request,
                self.config.allowed_hosts,
                timeout=self.config.search_timeout_s,
                handler=handler,
            ):
                pass
        except (HubError, urllib.error.URLError, OSError, ValueError):
            # A failed probe is not a failed download - it just means this file
            # stays on a lower rung. The real transfer reports its own errors.
            return None
        return getattr(handler, "linked_etag", None)


# --------------------------------------------------------------------------- #
# Registry hand-off (no third provenance concept)
# --------------------------------------------------------------------------- #


@dataclass
class ShardSetOutcome:
    """The result of fetching a whole shard set. `load_path` is what to register."""

    ok: bool
    load_path: Path | None = None
    parts_done: int = 0
    parts_total: int = 0
    bytes_written: int = 0
    cancelled: bool = False
    error: str = ""


def download_shard_set(
    api_base: str,
    repo_id: str,
    members: Sequence[str],
    rows: Sequence[dict[str, Any]],
    models_dir: Path | str,
    *,
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS,
    confirm_unverified: bool = False,
    cancel_event: threading.Event | None = None,
    progress: Callable[[Progress], None] | None = None,
    free_bytes: int | None = None,
) -> ShardSetOutcome:
    """Fetch every part of a shard set, then return the path llama.cpp should open.

    Each part goes through download_verified, so the safety contract is per-part
    and unchanged: streamed to .partial, hashed while writing, compared before
    rename, deleted on mismatch, allowlist re-checked on every redirect hop.

    Three things this adds on top of a loop:

    - it refuses to start unless the WHOLE set fits, because a 67 GB download
      that dies on the last part at 96% has cost the user hours for nothing;
    - an already-complete part is skipped, so an interrupted set resumes at part
      boundaries instead of restarting;
    - it returns part one's path. llama.cpp opens the set by that file and finds
      its siblings in the same directory, which is why resolve_shard_destination
      keeps them together.
    """
    if not members:
        return ShardSetOutcome(False, error="no shard members were given")

    sizes = {str(r.get("filename", "")): int(r.get("size_bytes") or 0) for r in rows}
    digests = {str(r.get("filename", "")): r.get("sha256") for r in rows}
    total_bytes = sum(sizes.get(name, 0) for name in members)
    if free_bytes is not None and total_bytes and total_bytes > free_bytes:
        return ShardSetOutcome(
            False,
            parts_total=len(members),
            error=(
                f"the set needs {total_bytes / 2**30:.1f} GB but only "
                f"{free_bytes / 2**30:.1f} GB is free; nothing was downloaded"
            ),
        )

    first_path: Path | None = None
    written = 0
    for index, member in enumerate(members, start=1):
        if cancel_event is not None and cancel_event.is_set():
            return ShardSetOutcome(
                False, first_path, index - 1, len(members), written, cancelled=True
            )
        try:
            dest = resolve_shard_destination(models_dir, member)
        except ValueError as exc:
            return ShardSetOutcome(False, first_path, index - 1, len(members), written,
                                   error=str(exc))
        if index == 1:
            first_path = dest
        dest.parent.mkdir(parents=True, exist_ok=True)

        expected = sizes.get(member, 0)
        if dest.is_file() and expected and dest.stat().st_size == expected:
            continue  # already complete; resume at the part boundary

        outcome = download_verified(
            build_resolve_path_url(api_base, repo_id, member),
            dest,
            digests.get(member),
            allowed_hosts=allowed_hosts,
            confirm_unverified=confirm_unverified,
            expected_size=expected or None,
            max_bytes=expected or None,
            require_gguf_magic=True,
            cancel_event=cancel_event,
            progress=progress,
        )
        if not outcome.ok:
            return ShardSetOutcome(
                False, first_path, index - 1, len(members), written,
                cancelled=outcome.cancelled,
                error=f"part {index} of {len(members)}: {outcome.error}",
            )
        written += outcome.bytes_written

    return ShardSetOutcome(True, first_path, len(members), len(members), written)


def registry_id_for(filename: str) -> str:
    """Derive a models.yaml id from a file name, using M13's sanitising rule."""
    # M15.5: a shard set registers under its stem, not under
    # '...-00001-of-00003' - the part counter is transport detail, not
    # identity - and never under its quant folder.
    parsed = parse_shard(filename)
    if parsed is not None:
        filename = parsed[0] + ".gguf"
    filename = filename.rsplit("/", 1)[-1]
    stem = Path(str(filename)).stem.lower()
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned or "model"


# UX Spec section 14's engineer-facing rule: the recorded provenance must name
# the LITERAL field the digest came from, not the internal rung key. "api" tells
# a maintainer nothing; "lfs.oid" tells them exactly which HuggingFace field to
# go and re-check. An unknown rung maps to "none" so a future typo under-claims.
VERIFICATION_FIELD_NAMES = {
    V_API: "lfs.oid",
    V_ETAG: "X-Linked-ETag",
    V_OPERATOR: "operator-supplied --url digest",
    V_NONE: "none",
}


def verification_field_name(verification: str) -> str:
    """The literal provenance field behind a rung, for notes and log lines."""
    return VERIFICATION_FIELD_NAMES.get(verification, VERIFICATION_FIELD_NAMES[V_NONE])


def registry_notes(repo_id: str, filename: str, verification: str, when: str) -> str:
    """The provenance sentence stored in the row's notes field.

    Provenance lives in `notes` rather than three new columns because the row is
    an ordinary registered model - the whole point of M14.14.5 is that a
    downloaded model is not a new kind of thing.

    DEFECT-QA-M14-6: this used to write `verification=api`, the rung key. The
    spec asks for the field itself, so the written row now says
    `verification=lfs.oid` - a name someone can look up in HuggingFace's API
    response months later.
    """
    return (
        f"downloaded from {repo_id}/{filename} on {when}; "
        f"verification={verification_field_name(verification)}"
    )


def hub_job_id(repo_id: str, filename: str) -> str:
    """The identifier that names ONE download job, shared by controller and view.

    DEFECT-QA-M14-2: the view has to be able to tell "this terminal result is
    about the download I am showing" from "this is about a request that was
    refused while that download kept running". Both sides therefore derive the
    id here rather than each formatting their own string, because two copies of
    this formula silently drifting apart is exactly how the terminal-state bug
    would come back.
    """
    return f"{repo_id}/{filename}"
