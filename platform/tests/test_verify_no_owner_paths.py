"""Path-hygiene enforcement tests (AC-M14-2, Architecture M14.2.5, DEC-M14-4).

This wires scripts/verify_no_owner_paths.py into the pytest suite. That wiring is
the whole point of the M14 config work: once these tests exist, a machine-specific
path cannot pass the suite, so it cannot pass review or the release checklist
either. Nobody has to remember the rule.

Three things are proved here:

1. the shipped tree is clean right now;
2. the rules actually fire (a scanner that always says OK proves nothing);
3. the script still has no exemption mechanism - the structural guarantee. An
   escape hatch would become the door the next machine-specific path walks
   through, so its absence is itself a test.

The bad-path samples below are assembled from pieces at runtime on purpose: a
literal machine-specific path written out in this file would be found by the very
scanner these tests exercise, because this file is scanned too. No exceptions,
including for the tests.
"""

from __future__ import annotations

import codecs
import subprocess
import sys
from pathlib import Path

import pytest

PLATFORM_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PLATFORM_DIR / "scripts" / "verify_no_owner_paths.py"

sys.path.insert(0, str(PLATFORM_DIR / "scripts"))

import verify_no_owner_paths as scanner  # noqa: E402


def test_verify_no_owner_paths_finds_nothing_in_the_shipped_tree():
    """Every copy of every file that would ship is free of machine-specific paths.

    "Would ship" means the content git holds, not only the copy on this disk
    (Review H-6). A finding whose name carries [index] or [HEAD] is not in the file
    on disk - that copy is already clean - it is in what a clone or `git archive`
    delivers, and the remedy is to COMMIT the clean file rather than to edit it.
    That distinction is in the message because the first person to see this fail
    will otherwise open a file that does not contain the reported line.
    """
    findings = scanner.scan(PLATFORM_DIR)
    rendered = "\n".join(f.render() for f in findings)
    from_git = sorted({f.path for f in findings if f.path.endswith("]")})
    assert findings == [], (
        f"machine-specific paths found:\n{rendered}\n\n"
        f"{len(from_git)} name(s) are clean on disk and dirty in git - commit them: "
        f"{from_git}"
    )


def test_verify_no_owner_paths_script_runs_clean_as_a_command():
    """The script is usable on its own, not only through pytest.

    The release checklist and a human debugging a failure both run it directly,
    so the standalone entry point is covered rather than assumed.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(PLATFORM_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK:" in result.stdout


@pytest.mark.parametrize(
    "sample, expected_rule",
    [
        # A drive-letter binary path - the classic leak.
        ("llama_cpp: " + '"' + "D:" + "/tools/llama-server.exe" + '"',
         "drive-letter absolute path"),
        # A home directory, in Windows separator form.
        ("caddyfile: " + '"' + "C:" + chr(92) + "Users" + chr(92) + "someone"
         + chr(92) + "Caddyfile" + '"',
         "home-directory path segment"),
        # The shared model tree, lowercase and forward-slashed.
        ("location: " + '"' + "c:" + "/ai/models/model.gguf" + '"',
         "machine-specific model tree root"),
    ],
)
def test_verify_no_owner_paths_rules_actually_fire(sample, expected_rule):
    """Positive control: each rule catches the shape it exists to catch."""
    findings = scanner.scan_text(sample, "sample.yaml")
    assert expected_rule in {f.rule for f in findings}, sample


def test_verify_no_owner_paths_ignores_urls_and_environment_names():
    """Legitimate lines are not flagged, so the check stays credible.

    A scanner that cries wolf gets an exemption bolted onto it within a week,
    which is exactly the outcome the no-exemption rule exists to prevent. Loopback
    URLs, environment variable NAMES and relative paths must all pass cleanly.
    """
    clean = "\n".join(
        [
            "backend_base_url: http://127.0.0.1:8080/v1",
            "see https://download.pytorch.org/whl/cpu for the wheel index",
            "the LOCALAPPDATA environment variable names the data root",
            "data_dir: webui-data",
            "location: /locitize-test/models/example.gguf",
        ]
    )
    assert scanner.scan_text(clean, "clean.yaml") == []


def test_verify_no_owner_paths_has_no_exemption_mechanism():
    """The structural rule: no skip list, no opt-out, no magic comment.

    AC-M14-2 checks this from the outside with a grep for "allow" in the script's
    source. The same check lives here so a future edit that quietly adds a skip
    list fails in the suite, at the moment it is written, with a message that says
    why. The forbidden words are assembled from pieces so this test file does not
    trip its own check.
    """
    source = SCRIPT_PATH.read_text(encoding="utf-8").lower()
    forbidden = [
        "all" + "ow",  # the exact substring AC-M14-2 greps for
        "skip" + "list",
        "exempt",
        "whitelist",
        "ignore_list",
    ]
    hits = [word for word in forbidden if word in source]
    assert hits == [], f"exemption mechanism introduced: {hits}"


# ===========================================================================
# Reading regression matrix (Review H-1, H-3, H-5), organised by FAILURE MODE.
#
# Why not by encoding: this matrix used to be a list of encodings, one row per
# codec and byte order, and every row wrote a file whose byte-order mark agreed
# with its body. Three separate reviews found three separate bypasses that the
# list could not have caught, because each was a NEW WAY FOR A FILE TO LIE, not
# a new codec. Growing the list from five encodings to ten changed nothing.
#
# So the rows below are grouped by the way a file can defeat the scanner, and a
# new codec is just another row inside an existing group:
#
#   1. LABEL DISAGREES WITH CONTENT - the file claims one thing and is another
#      (a byte-order mark that lies, a suffix that lies). Required outcome: the
#      real content is read and its owner path is REPORTED.
#   2. NOT TEXT AT ALL - binary noise, truncated sequences, control bytes, with
#      and without a mark in front. Required outcome: reported UNREADABLE, which
#      fails the gate, because a file the scanner cannot read is one it cannot
#      clear.
#   3. LEGITIMATE BUT UNUSUAL - real files that are merely unfamiliar. Required
#      outcome: NO findings. This is the false-positive guard; strictness that
#      flags valid files gets an exemption bolted onto it within a week.
#   4. PLAIN - the straightforward cases, one per encoding, mark agreeing with
#      body, plus every way a home directory can be spelled (Review M-9).
#      Required outcome: the owner path is reported.
#   5. GIT DISAGREES WITH THE DISK - the payload is in HEAD, or in the index, or
#      in a blob whose file was deleted, or in a symlink entry (Review H-6).
#      Required outcome: reported, naming the copy it lives in, even though no
#      file on disk contains it.
#
# When the next variant of this bug is imagined, it belongs in one of these five
# groups before it is fixed. Group 1 also has a GENERATED row per byte-order
# mark (see the exhaustive test at the end of group 1), so a mark added to the
# scanner cannot arrive without a lying-label case.
#
# Group 5 carries the one check that does not depend on anybody imagining the next
# variant at all: the COUNT INVARIANT. Four of the five bugs above were a silent
# narrowing of scope, so the group ends by requiring the number of names git lists
# to equal the number the scan accounted for, and by proving that requirement
# raises when it is broken.
#
# Each case is built in a throwaway git repository rather than in the real
# platform directory, so a failing test can never leave a probe file behind in
# the tree the scanner guards.
# ===========================================================================

# The Reviewer's probe line, assembled from pieces so this test file does not
# itself trip the scanner that reads it. The user name is a neutral placeholder:
# the SHAPE of the path is what the rule matches, and a real person's username
# has no business being committed to the repository this check protects.
OWNER_PATH_PROBE = (
    "$caddy = " + '"' + "C:" + chr(92) + "Users" + chr(92) + "someone"
    + chr(92) + "Caddyfile" + '"' + "\n"
)

# Every byte value twice: valid in no text encoding the scanner tries.
BINARY_NOISE = bytes(range(256)) * 2

# Real PNG magic followed by bytes no codec reads as text.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes([0x00, 0xFF, 0xC3, 0x28]) * 8


def _git_in(root: Path, *args: str) -> str:
    """Run one git command inside a throwaway repository, returning its stdout.

    Identity is passed per-command so these fixtures do not depend on - or care
    about - whatever git identity the machine running the suite has configured.
    """
    result = subprocess.run(
        ["git", "-c", "user.name=probe", "-c", "user.email=probe@example.invalid", *args],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _new_repo(tmp_path: Path, name: str) -> Path:
    """Create an empty git repository under a fresh directory."""
    root = tmp_path / name
    root.mkdir(parents=True)
    _git_in(root, "init", "-q")
    return root


def _probe_repo(tmp_path: Path, filename: str, data: bytes) -> Path:
    """Make a one-file git repository whose single file has exactly `data`.

    `ship_set()` derives the ship set from git, so a real repository is the
    only honest way to exercise it. The file is left untracked-but-not-ignored,
    which is precisely the state the Reviewer's probe was in.
    """
    root = tmp_path / "probe_repo"
    root.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q"], cwd=str(root), check=True, capture_output=True
    )
    (root / filename).write_bytes(data)
    return root


def _assert_owner_path_reported(tmp_path: Path, filename: str, data: bytes) -> None:
    """The file is read as what it really is and both path rules fire."""
    findings = scanner.scan(_probe_repo(tmp_path, filename, data))
    rules = {f.rule for f in findings}
    assert "drive-letter absolute path" in rules, findings
    assert "home-directory path segment" in rules, findings


def _assert_reported_unreadable(tmp_path: Path, filename: str, data: bytes) -> None:
    """The file cannot be read, so it becomes a finding naming itself."""
    findings = scanner.scan(_probe_repo(tmp_path, filename, data))
    assert [f.path for f in findings] == [filename], findings
    assert findings[0].rule == scanner.UNREADABLE_RULE, findings


def _assert_clean(tmp_path: Path, filename: str, data: bytes) -> None:
    """A legitimate file produces no findings of any kind."""
    findings = scanner.scan(_probe_repo(tmp_path, filename, data))
    assert findings == [], [f.render() for f in findings]


# ---------------------------------------------------------------------------
# Group 1: THE LABEL DISAGREES WITH THE CONTENT.
#
# The failure mode behind Review H-3 and H-5. A file states its encoding with a
# byte-order mark, or its format with a suffix, and the statement is false. The
# scanner must believe the BYTES, not the label, and still find the path.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename, data",
    [
        # A little-endian mark in front of a big-endian body (Review H-5). The
        # declared decode succeeds and yields byte-swapped CJK mojibake with no
        # NUL in it, which every path rule then misses.
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF16_LE + OWNER_PATH_PROBE.encode("utf-16-be"),
            id="mark-says-utf16le-body-is-utf16be",
        ),
        # The same lie the other way round.
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF16_BE + OWNER_PATH_PROBE.encode("utf-16-le"),
            id="mark-says-utf16be-body-is-utf16le",
        ),
        # The lie one width up.
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF32_LE + OWNER_PATH_PROBE.encode("utf-32-be"),
            id="mark-says-utf32le-body-is-utf32be",
        ),
        # The mark lies about the WIDTH rather than the byte order.
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF16_BE + OWNER_PATH_PROBE.encode("utf-32-le"),
            id="mark-says-utf16be-body-is-utf32le",
        ),
        # A UTF-8 mark in front of a wide body - what an editor that "adds a
        # BOM" to a PowerShell file produces.
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF8 + OWNER_PATH_PROBE.encode("utf-16-le"),
            id="mark-says-utf8-body-is-utf16le",
        ),
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF8 + OWNER_PATH_PROBE.encode("utf-16-be"),
            id="mark-says-utf8-body-is-utf16be",
        ),
        # The other label a file makes: its SUFFIX. Plain text carrying an owner
        # path, named as an image, must not escape the scan on the strength of
        # its name.
        pytest.param(
            "notes.png",
            OWNER_PATH_PROBE.encode("utf-8"),
            id="suffix-says-png-content-is-utf8-text",
        ),
        # The same lie with a model-weight suffix and a wide encoding, so both
        # claims are false at once.
        pytest.param(
            "notes.gguf",
            OWNER_PATH_PROBE.encode("utf-16-le"),
            id="suffix-says-gguf-content-is-utf16le-text",
        ),
    ],
)
def test_a_lying_label_does_not_hide_an_owner_path(tmp_path, filename, data):
    """Review H-5: what a file says about itself is a claim, not evidence."""
    _assert_owner_path_reported(tmp_path, filename, data)


@pytest.mark.parametrize("mark, declared, _self_validating", scanner._BOMS)
def test_every_byte_order_mark_is_checked_against_its_body(
    tmp_path, mark, declared, _self_validating
):
    """Generated coverage: no mark in the scanner may go untested.

    The list above is hand-written and therefore only as complete as whoever
    last thought about it - which is exactly how H-5 survived. This walks the
    scanner's own table of marks instead: for every mark, a body written in a
    DIFFERENT encoding of the same family must never end up silently clean.
    Adding a mark to the scanner adds rows here automatically.

    Two outcomes are correct - the path is found (the body was re-read properly)
    or the file is reported unreadable (nothing read it, so nobody cleared it).
    Only the third outcome, "read into something harmless-looking and passed",
    is the bug.
    """
    others = [
        encoding
        for encoding in ("utf-8", "utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be")
        if encoding != declared
    ]
    for encoding in others:
        data = mark + OWNER_PATH_PROBE.encode(encoding)
        findings = scanner.scan(_probe_repo(tmp_path / encoding, "setup.ps1", data))
        rules = {f.rule for f in findings}
        assert rules, f"{declared} mark over a {encoding} body scanned clean"
        assert (
            "drive-letter absolute path" in rules or scanner.UNREADABLE_RULE in rules
        ), f"{declared} mark over a {encoding} body: {findings}"


def test_a_decode_that_succeeds_into_nonsense_is_rejected():
    """Review H-3, at the level of the rule rather than one file shape.

    Reading big-endian bytes as little-endian is the whole class of bug in one
    line: it does not raise, it returns characters in the CJK range built from
    swapped ASCII pairs. The scanner must reject that result and keep looking,
    which is what makes the cases above find the path instead of nothing.
    """
    swapped = OWNER_PATH_PROBE.encode("utf-16-be").decode("utf-16-le")
    assert "\x00" not in swapped, "the wrong-order decode really does contain no NUL"
    assert scanner._plain_text_share(swapped) < scanner._PLAUSIBLE_TEXT_SHARE
    assert scanner.decode_bytes(OWNER_PATH_PROBE.encode("utf-16-be")) == OWNER_PATH_PROBE


def test_a_lying_mark_is_discarded_rather_than_trusted_at_the_unit_level():
    """The unit-level statement of the H-5 fix, independent of file layout.

    A little-endian mark over a big-endian body must decode to the TRUE text,
    not to the mark's version of it, so the path rules see the real line.
    """
    data = codecs.BOM_UTF16_LE + OWNER_PATH_PROBE.encode("utf-16-be")
    assert scanner.decode_bytes(data) == OWNER_PATH_PROBE


# ---------------------------------------------------------------------------
# Group 2: THE CONTENT IS NOT TEXT AT ALL.
#
# Rule 4 of the scanner: a file it cannot decode is a file it cannot clear, so
# it is reported by name and the gate fails. The cases with a mark in front are
# the ones that mattered - a mark used to be enough to get 6 bytes of noise
# accepted as text, which defeated this rule entirely (Review H-5).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename, data",
    [
        pytest.param("mystery.ps1", BINARY_NOISE, id="noise-no-mark"),
        pytest.param(
            "mystery.ps1",
            bytes([0xC3, 0x28, 0xA0, 0xA1, 0x80]),
            id="invalid-utf8-no-mark",
        ),
        # Six bytes of noise behind a little-endian mark - the Reviewer's fourth
        # probe. The mark used to buy this file a free pass as "text".
        pytest.param(
            "mystery.ps1",
            codecs.BOM_UTF16_LE + bytes([0x01, 0x02, 0x7F, 0xFF, 0x00, 0x1B]),
            id="noise-behind-utf16le-mark",
        ),
        pytest.param(
            "mystery.ps1",
            codecs.BOM_UTF16_BE + bytes([0x01, 0x02, 0x7F, 0xFF, 0x00, 0x1B]),
            id="noise-behind-utf16be-mark",
        ),
        pytest.param(
            "mystery.ps1",
            codecs.BOM_UTF8 + BINARY_NOISE,
            id="noise-behind-utf8-mark",
        ),
        # Valid UTF-8 that is nonetheless not text: a run of control bytes. The
        # NUL test alone cannot see this; the control-character test can.
        pytest.param(
            "mystery.ps1",
            bytes(range(0x01, 0x09)) * 4,
            id="control-bytes-valid-as-utf8",
        ),
        # A multibyte sequence cut in half - what a truncated download or a
        # botched byte-range copy produces.
        pytest.param("mystery.ps1", b"abc" + b"\xe3\x81", id="truncated-utf8-sequence"),
        pytest.param(
            "mystery.ps1",
            codecs.BOM_UTF16_LE + OWNER_PATH_PROBE.encode("utf-16-le")[:-1],
            id="truncated-utf16-odd-byte-count",
        ),
    ],
)
def test_content_that_is_not_text_is_a_finding_not_a_silent_pass(
    tmp_path, filename, data
):
    """Fail closed: unreadable means uncleared, mark or no mark."""
    _assert_reported_unreadable(tmp_path, filename, data)


def test_binary_noise_is_never_believed_as_wide_text():
    """The other side of the same rule: nonsense must not become "text".

    Being willing to try four encodings is only safe because the result is
    judged. Without that, almost any byte string decodes as UTF-16 and the gate
    would report a clean scan of a file it never actually read.
    """
    assert scanner.decode_bytes(BINARY_NOISE) is None


# ---------------------------------------------------------------------------
# Group 3: LEGITIMATE BUT UNUSUAL - the false-positive guard.
#
# Everything here is a real file a real contributor might commit. None of it
# carries an owner path, so none of it may produce a finding. A scanner that
# flags valid files loses its credibility and then its teeth.
# ---------------------------------------------------------------------------

# Non-ASCII written as escapes so this source file itself stays plain ASCII;
# what the cases are about is the decoded string, not the bytes of this file.
UNIT_SYMBOLS = "threshold: 12 \u00b5g/m\u00b3 and a name like Bj\u00f6rn\n"
NON_LATIN_TEXT = "\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059\n"


@pytest.mark.parametrize(
    "filename, data",
    [
        pytest.param("notes.txt", b"", id="empty-file"),
        pytest.param("notes.txt", b"plain ascii, nothing unusual\n", id="plain-ascii"),
        pytest.param(
            "notes.txt", b"crlf line ends\r\nand\ta tab\r\n", id="crlf-and-tabs"
        ),
        # A real UTF-8 document with non-ASCII content. UTF-8 validates itself,
        # so it is read rather than doubted, whatever script it is written in.
        pytest.param("notes.txt", UNIT_SYMBOLS.encode("utf-8"), id="utf8-unit-symbols"),
        pytest.param(
            "notes.txt",
            codecs.BOM_UTF8 + UNIT_SYMBOLS.encode("utf-8"),
            id="utf8-with-mark-unit-symbols",
        ),
        pytest.param(
            "notes.txt", NON_LATIN_TEXT.encode("utf-8"), id="utf8-non-latin-script"
        ),
        # Real wide files whose mark tells the truth about an ASCII body: the
        # ordinary output of PowerShell's Out-File.
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF16_LE + "Write-Host ok\n".encode("utf-16-le"),
            id="truthful-utf16le-mark",
        ),
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF16_BE + "Write-Host ok\n".encode("utf-16-be"),
            id="truthful-utf16be-mark",
        ),
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF32_LE + "Write-Host ok\n".encode("utf-32-le"),
            id="truthful-utf32le-mark",
        ),
        pytest.param(
            "setup.ps1", "Write-Host ok\n".encode("utf-16-le"), id="markless-utf16le"
        ),
        # A genuine image: its suffix and its bytes agree, so it is left alone
        # rather than reported unreadable.
        pytest.param("capture.png", PNG_BYTES, id="real-png-with-png-suffix"),
        # Review M-9 bought its case-insensitive rule 2 with a false-positive
        # risk, so the price is pinned here. "users" is an ordinary segment in a
        # web path, and a gate that flags documentation links gets switched off.
        # The lowercase row is the one the widened rule could break; the
        # capitalised row is a false positive the OLD rule actually had.
        pytest.param(
            "README.md",
            b"see https://example.com/users/me for the account page\n",
            id="url-with-lowercase-users-segment",
        ),
        pytest.param(
            "README.md",
            b"see https://example.com/Users/someone for the account page\n",
            id="url-with-capitalised-users-segment",
        ),
        # Environment-variable indirection is the RECOMMENDED repair for a
        # violation, so it must never be one. A gate that rejects its own advice
        # is a gate nobody keeps.
        pytest.param(
            "settings.default.yaml",
            b"data_dir: %LOCALAPPDATA%/LOCITIZE\nmodels: %USERPROFILE%/ai/models\n",
            id="windows-environment-variable-names",
        ),
        pytest.param(
            "setup.ps1",
            b"$root = $env:USERPROFILE\n$data = $env:LOCALAPPDATA\n",
            id="powershell-environment-variable-names",
        ),
    ],
)
def test_legitimate_files_produce_no_findings(tmp_path, filename, data):
    """Strictness must not be bought with noise."""
    _assert_clean(tmp_path, filename, data)


def test_ordinary_utf8_text_is_not_rejected_by_the_plausibility_rule():
    """UTF-8 is read, not scored, so non-ASCII content survives the strictness.

    UTF-8 validates itself - wrong bytes raise rather than decode - so its
    result only has to be free of control characters. Only a guessed or merely
    claimed width and byte order gets the plain-text score.
    """
    assert scanner.decode_bytes(UNIT_SYMBOLS.encode("utf-8")) == UNIT_SYMBOLS
    assert scanner.decode_bytes(NON_LATIN_TEXT.encode("utf-8")) == NON_LATIN_TEXT
    assert (
        scanner.decode_bytes(codecs.BOM_UTF8 + UNIT_SYMBOLS.encode("utf-8"))
        == UNIT_SYMBOLS
    )


def test_wide_non_latin_text_is_reported_rather_than_quietly_cleared(tmp_path):
    """The known cost of the strict rule, pinned so it is a decision not a bug.

    A UTF-16 file written in a non-Latin script scores below the plain-text
    threshold, and nothing distinguishes it from a mark that lied - both are
    "the claimed encoding produced very little plain text". The scanner errs
    toward asking a human to look: the file is reported UNREADABLE and the gate
    fails, rather than being cleared by a decode nobody checked. The equivalent
    UTF-8 file (the realistic case, covered above) is read normally.
    """
    _assert_reported_unreadable(
        tmp_path,
        "notes.ps1",
        codecs.BOM_UTF16_LE + NON_LATIN_TEXT.encode("utf-16-le"),
    )


def test_binary_ship_set_files_are_read_as_bytes_not_reported(tmp_path):
    """Images are not text, so they are neither scanned nor reported unreadable.

    Without this, the fail-closed rule would report every PNG in the tree and
    the gate would be useless. The distinction is the file FORMAT - declared in
    BINARY_SUFFIXES and confirmed against the bytes - never a file name or path.
    """
    root = _probe_repo(tmp_path, "capture.png", PNG_BYTES)
    payload = scanner.ship_set(root)
    assert scanner.scan_ship_set(payload) == []
    assert payload.entries == []
    # Not scanned is not the same as not accounted for (Review H-6): the name is
    # still listed, and its disposition has to say why nothing was read.
    assert payload.listed == ["capture.png"]
    assert "binary content" in payload.accounted["capture.png"]


# ---------------------------------------------------------------------------
# Group 4: PLAIN - the straightforward cases.
#
# One row per encoding, each written the way the tool that produces it writes
# it, mark and body agreeing. These are the cases the original bypass (Review
# H-1) was about: the scanner used to catch UnicodeDecodeError and move on, so
# any ship-set file that was not valid UTF-8 was passed over with no finding and
# no effect on the exit code. Windows PowerShell 5.1 writes UTF-16LE by default
# and .ps1 is in the ship set, so setup.ps1 was the file most likely to walk
# through that door carrying an owner path.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename, data",
    [
        # PowerShell 5.1's default output encoding, written without a mark. The
        # sharpest plain case: mark-less UTF-16LE ASCII is also VALID UTF-8, so
        # a naive utf-8 decode "succeeds" and yields NUL-separated characters
        # that no path regex can match.
        pytest.param(
            "setup.ps1", OWNER_PATH_PROBE.encode("utf-16-le"), id="utf16le-no-mark"
        ),
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF16_LE + OWNER_PATH_PROBE.encode("utf-16-le"),
            id="utf16le-with-mark",
        ),
        # Mark-less BIG-endian - the case that got through the first fix (Review
        # H-3). Decoding these bytes as little-endian raises nothing.
        pytest.param(
            "setup.ps1", OWNER_PATH_PROBE.encode("utf-16-be"), id="utf16be-no-mark"
        ),
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF16_BE + OWNER_PATH_PROBE.encode("utf-16-be"),
            id="utf16be-with-mark",
        ),
        # UTF-32, both byte orders, with and without a mark. Rarer than UTF-16,
        # but the same trap one width up, so it is pinned rather than argued
        # about.
        pytest.param(
            "setup.ps1", OWNER_PATH_PROBE.encode("utf-32-le"), id="utf32le-no-mark"
        ),
        pytest.param(
            "setup.ps1", OWNER_PATH_PROBE.encode("utf-32-be"), id="utf32be-no-mark"
        ),
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF32_LE + OWNER_PATH_PROBE.encode("utf-32-le"),
            id="utf32le-with-mark",
        ),
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF32_BE + OWNER_PATH_PROBE.encode("utf-32-be"),
            id="utf32be-with-mark",
        ),
        pytest.param(
            "setup.ps1", OWNER_PATH_PROBE.encode("utf-8"), id="utf8-no-mark"
        ),
        pytest.param(
            "setup.ps1",
            codecs.BOM_UTF8 + OWNER_PATH_PROBE.encode("utf-8"),
            id="utf8-with-mark",
        ),
        # Not a .py/.yaml file, to prove the widened suffix coverage (Review
        # M-2). A local editable-install line pointing at one developer's drive
        # is exactly the leak this milestone exists to stop.
        pytest.param(
            "requirements.txt",
            OWNER_PATH_PROBE.encode("utf-16-le"),
            id="requirements-txt-utf16le",
        ),
    ],
)
def test_owner_path_is_found_whatever_the_text_encoding(tmp_path, filename, data):
    """A machine-specific path is reported no matter how the file was saved."""
    _assert_owner_path_reported(tmp_path, filename, data)


def test_requirements_and_plain_text_files_are_in_the_scan_set():
    """Review M-2: the real ship set is scanned, not a narrow suffix selection."""
    scanned = {entry.name for entry in scanner.ship_set().entries}
    for expected in ("requirements.txt", "finetune_warning.txt", ".gitignore"):
        assert expected in scanned, f"{expected} is not being scanned"
    # Real binaries stay out: reading them for lines is meaningless.
    assert not [name for name in scanned if name.endswith(".png")]


# The home-directory shapes rule 2 could not see before Review M-9. Each is the
# SAME directory, written the way one real tool writes it, and each is assembled
# from pieces so this file does not trip the scanner that reads it.
_SEP = chr(92)
_HOME_DIR_SHAPES = {
    # A Windows path inside any Python or JSON string literal. The doubled
    # separator is the normal form there, and the old character class could not
    # match a second separator, so this was invisible.
    "escaped-doubled-separator": '{"dir": "' + _SEP * 4 + "nas" + _SEP * 2
    + "Users" + _SEP * 2 + "someone" + _SEP * 2 + 'ai"}',
    # Lowercase, which every Windows filesystem treats as the same path.
    "lowercase-windows": "path = " + _SEP + "users" + _SEP + "someone" + _SEP + "x",
    # WSL: lowercase, forward slashes, and no drive letter for rule 1 to catch.
    # Assembled rather than written out, like every other case in this file: a
    # literal here would be found by the scanner reading its own test suite.
    "wsl-lowercase-forward": "path = /mnt/c/" + "users" + "/someone/x",
    # A UNC share: again no drive letter, so rule 2 is the only rule that can fire.
    "unc-share-lowercase": "path = " + _SEP * 2 + "mypc" + _SEP + "users" + _SEP
    + "someone",
    # The long-path prefix Windows APIs use.
    "unc-long-path-prefix": "path = " + _SEP * 2 + "?" + _SEP + "C:" + _SEP
    + "Users" + _SEP + "someone",
}


@pytest.mark.parametrize("shape", sorted(_HOME_DIR_SHAPES), ids=sorted(_HOME_DIR_SHAPES))
def test_rule_two_sees_every_way_a_home_directory_is_written(shape):
    """Review M-9: rule 2 must fire on its own, not only where rule 1 already does.

    Three of these five carry no drive letter at all, so before this widening the
    entire gate passed them: a JSON config value naming the owner's home directory
    on a NAS was clean. The assertion is deliberately on rule 2 specifically rather
    than on "some finding", because a pass carried entirely by rule 1 is what made
    the hole invisible in the first place.
    """
    findings = scanner.scan_text(_HOME_DIR_SHAPES[shape], "sample.json")
    assert "home-directory path segment" in {f.rule for f in findings}, findings


def test_the_model_tree_root_rule_survives_an_escaped_separator():
    """Review M-9, same class one rule over: rule 3 doubled its separator too.

    Rule 3 exists so the failure message names the real cause rather than saying
    only "drive-letter path". In a string literal - the form it is most likely to
    appear in - the separator is doubled, and the un-widened rule fell back to
    rule 1's less specific message.
    """
    sample = 'model = "C:' + _SEP * 2 + 'ai' + _SEP * 2 + 'models' + _SEP * 2 + 'm.gguf"'
    assert "machine-specific model tree root" in {
        f.rule for f in scanner.scan_text(sample, "sample.py")
    }


# ---------------------------------------------------------------------------
# Group 5: GIT DISAGREES WITH THE DISK.
#
# The failure mode behind Review H-6, and the reason it is a group rather than
# four tests: the scanner took its file LIST from git and its file CONTENT from
# the working tree, which are two different artifacts. Every case below is a way
# for those two to differ, and in every one of them the copy that SHIPS is the one
# git holds - a clone, a CI checkout and `git archive` all deliver HEAD.
#
# Required outcome for each: the owner path is REPORTED, even though no file on
# disk contains it.
#
# The last two cases are not attacks. One pins the deliberate decision about a
# tracked file that is absent from disk; the other pins the count invariant, which
# is the part of this fix that catches the shape nobody has imagined yet.
# ---------------------------------------------------------------------------

CLEAN_TEXT = "$caddy = " + '"' + "./Caddyfile" + '"' + "\n"


def _assert_reported_from(root: Path, expected_origin: str) -> None:
    """Both path rules fire, and the finding names the copy the path lives in."""
    findings = scanner.scan(root)
    rules = {f.rule for f in findings}
    assert "drive-letter absolute path" in rules, [f.render() for f in findings]
    assert "home-directory path segment" in rules, [f.render() for f in findings]
    labels = {f.path for f in findings}
    assert any(f"[{expected_origin}]" in label for label in labels), labels


def test_an_owner_path_committed_to_head_is_found_with_a_clean_working_tree(tmp_path):
    """Review H-6, case A2: `git archive HEAD` ships it, so the gate must see it.

    HEAD carries the payload; the index and the working tree have both been
    cleaned. Every check that reads the disk reports success, and every consumer of
    the release gets the owner path.
    """
    root = _new_repo(tmp_path, "head_payload")
    (root / "setup.ps1").write_text(OWNER_PATH_PROBE, encoding="utf-8")
    _git_in(root, "add", "setup.ps1")
    _git_in(root, "commit", "-q", "-m", "payload")
    (root / "setup.ps1").write_text(CLEAN_TEXT, encoding="utf-8")
    _git_in(root, "add", "setup.ps1")  # index clean too: HEAD is the only carrier

    _assert_reported_from(root, "HEAD")


def test_an_owner_path_staged_in_the_index_is_found_with_a_clean_working_tree(tmp_path):
    """Review H-6, case A3: the next commit will carry it.

    HEAD is clean, the index holds the payload, and the working tree has been
    cleaned since staging. `git commit` at this point publishes the path.
    """
    root = _new_repo(tmp_path, "index_payload")
    (root / "setup.ps1").write_text(CLEAN_TEXT, encoding="utf-8")
    _git_in(root, "add", "setup.ps1")
    _git_in(root, "commit", "-q", "-m", "clean")
    (root / "setup.ps1").write_text(OWNER_PATH_PROBE, encoding="utf-8")
    _git_in(root, "add", "setup.ps1")
    (root / "setup.ps1").write_text(CLEAN_TEXT, encoding="utf-8")

    _assert_reported_from(root, "index")


def test_a_tracked_file_deleted_from_disk_is_still_read_from_git(tmp_path):
    """Review H-6, case A1: the shape that was live in the real tree.

    Four tracked wrapper scripts had been deleted from the working tree, so
    `path.is_file()` dropped them - with no finding of any kind, not even the
    "unreadable" one the design promises for anything it cannot clear. git listed
    96 names, 91 were read, and the gate printed OK.
    """
    root = _new_repo(tmp_path, "deleted_on_disk")
    (root / "wrapper.bat").write_text(OWNER_PATH_PROBE, encoding="utf-8")
    _git_in(root, "add", "wrapper.bat")
    _git_in(root, "commit", "-q", "-m", "payload")
    (root / "wrapper.bat").unlink()

    _assert_reported_from(root, "index+HEAD")


def test_a_symlink_index_entry_whose_blob_is_an_owner_path_is_found(tmp_path):
    """Review H-6, case E2: the shape a Linux commit takes on a Windows checkout.

    A mode-120000 entry's blob content IS the link target, stored as text. Nothing
    is created on disk here - the entry is written straight into the index, which
    is exactly the state a Windows checkout of a symlink-carrying commit can be in.
    The scanner reads the blob like any other, which is why it does not have to
    know anything about symbolic links to catch this.
    """
    root = _new_repo(tmp_path, "symlink_entry")
    target = OWNER_PATH_PROBE.strip()
    # The blob has to exist in the object store before the index can reference it.
    # It is written from a scratch file that is then removed, so nothing on disk
    # carries the payload and only the index entry does.
    (root / "_payload").write_text(target, encoding="utf-8")
    object_id = _git_in(root, "hash-object", "-w", "_payload").strip()
    (root / "_payload").unlink()
    _git_in(root, "update-index", "--add", "--cacheinfo", f"120000,{object_id},caddy_link")

    findings = scanner.scan(root)
    rendered = [f.render() for f in findings]
    assert any("caddy_link" in f.path for f in findings), rendered
    assert "home-directory path segment" in {f.rule for f in findings}, rendered


def test_a_tracked_file_absent_from_disk_is_accounted_for_not_dropped(tmp_path):
    """The deliberate decision behind case A1, pinned so it is not read as an oversight.

    A tracked file missing from the working tree is NOT itself a finding here.
    Review prescribed making it one; the reason it is not is that the premise
    changed with the fix. The finding was to exist because the gate could not clear
    such a file - and now it can, because it reads the blob that ships. What must
    never happen again is the SILENT part, so the name is required to carry a
    disposition that says where its content came from, and the count invariant
    makes that mandatory rather than best-effort.

    The four deleted LOCITIZE *.bat wrappers in the real tree are in exactly this
    state, deliberately, pending a separate reconciliation (M13 AC-M13-6). Their
    blobs are clean, and this test is why that is a pass rather than an omission.
    """
    root = _new_repo(tmp_path, "absent_but_clean")
    (root / "wrapper.bat").write_text(CLEAN_TEXT, encoding="utf-8")
    _git_in(root, "add", "wrapper.bat")
    _git_in(root, "commit", "-q", "-m", "clean")
    (root / "wrapper.bat").unlink()

    payload = scanner.ship_set(root)
    assert scanner.scan_ship_set(payload) == []
    assert "wrapper.bat" in payload.accounted
    disposition = payload.accounted["wrapper.bat"]
    assert "absent from the working tree" in disposition, disposition
    assert "scanned" in disposition, disposition
    # The content really was read, from git rather than from a file that is gone.
    assert [e.origins for e in payload.entries if e.name == "wrapper.bat"] == [
        ("index", "HEAD")
    ]


def test_content_git_no_longer_tracks_is_read_from_head_but_not_from_disk(tmp_path):
    """The one case where reading the disk copy would report the USER to themselves.

    A file HEAD still carries and the index has dropped is the shape of this
    milestone's own work: settings.yaml and models.yaml were committed once, are
    now gitignored, and still sit on disk as the user's live configuration full of
    their own paths. The committed blob still ships and must be scanned; the
    working copy must not be, or the gate reports a user's own machine to them as a
    defect.
    """
    root = _new_repo(tmp_path, "untracked_now")
    (root / "settings.yaml").write_text(OWNER_PATH_PROBE, encoding="utf-8")
    _git_in(root, "add", "settings.yaml")
    _git_in(root, "commit", "-q", "-m", "committed user data")
    (root / ".gitignore").write_text("settings.yaml\n", encoding="utf-8")
    _git_in(root, "rm", "-q", "--cached", "settings.yaml")
    # The disk copy now holds a DIFFERENT machine-specific path, so the two copies
    # can be told apart in the output.
    (root / "settings.yaml").write_text(
        "kokoro: " + '"' + "Z:" + chr(92) + "private" + chr(92) + "voices" + '"' + "\n",
        encoding="utf-8",
    )

    findings = scanner.scan(root)
    rendered = "\n".join(f.render() for f in findings)
    assert "settings.yaml [HEAD]" in {f.path for f in findings}, rendered
    assert "Z:" not in rendered, "the user's own gitignored data must not be scanned"
    disposition = scanner.ship_set(root).accounted["settings.yaml"]
    assert "no longer in the ship set" in disposition, disposition


def test_every_name_git_lists_is_accounted_for_in_the_real_tree():
    """The count invariant, on the real tree (Review H-6, missing test 2).

    This is the assertion that generalises past the variant that prompted it. H-6
    was not one bug about symlinks or one bug about deleted files - it was a scope
    that narrowed silently, and four different shapes fell through the same gap. A
    scanner cannot be made safe by imagining more shapes; it can be made safe by
    refusing to report success over a name it never accounted for.
    """
    payload = scanner.ship_set(PLATFORM_DIR)
    listed = set(payload.listed)

    tracked = set(scanner._git_names(PLATFORM_DIR, []))
    committed = set(scanner._head_blobs(PLATFORM_DIR))
    untracked = set(scanner._git_names(PLATFORM_DIR, ["--others", "--exclude-standard"]))
    assert listed == tracked | committed | untracked

    # Nothing listed and forgotten, and no disposition left blank.
    assert set(payload.accounted) == listed
    assert [name for name, why in payload.accounted.items() if not why.strip()] == []
    # Non-vacuity: a scope that had collapsed to nothing would satisfy the above.
    assert len(listed) >= 90, len(listed)
    # Every listed name either had its content scanned or carries an explicit
    # no-content disposition (binary asset, or a name HEAD ships that the tree
    # no longer holds). The old form hardcoded an allowance of exactly ONE
    # unscanned name - true when the repo had one binary, false the day it had
    # five, and masked locally by untracked files padding the entry count; the
    # first clean-checkout CI run exposed it. Deriving the allowance from the
    # accounting keeps the invariant true at any binary count while still
    # refusing a silent scan-scope collapse.
    scanned = {entry.name for entry in payload.entries}
    unscanned = listed - scanned
    for name in sorted(unscanned):
        why = payload.accounted[name]
        assert "binary content" in why or "no longer" in why or "absent" in why, (
            name,
            why,
        )
    assert len(payload.entries) >= len(listed) - len(unscanned)


def test_the_count_invariant_raises_rather_than_reporting_success():
    """The invariant must FAIL, loudly, when a name goes unaccounted for.

    Proving the invariant holds today proves nothing about the case it exists for,
    so a name is dropped from the accounting on purpose. RuntimeError is the right
    failure: main() turns it into exit code 2, "the scan could not run", which is
    the only honest answer when the gate cannot say what happened to its own scope.
    """
    payload = scanner.ship_set(PLATFORM_DIR)
    dropped = dict(payload.accounted)
    victim = sorted(dropped)[0]
    del dropped[victim]
    narrowed = scanner.ShipSet(
        entries=payload.entries,
        listed=payload.listed,
        accounted=dropped,
        unreadable=payload.unreadable,
    )
    with pytest.raises(RuntimeError) as raised:
        narrowed.check_invariant()
    assert victim in str(raised.value)
    assert "invariant violated" in str(raised.value)

    blanked = dict(payload.accounted)
    blanked[victim] = ""
    with pytest.raises(RuntimeError):
        scanner.ShipSet(
            entries=payload.entries,
            listed=payload.listed,
            accounted=blanked,
            unreadable=payload.unreadable,
        ).check_invariant()


def test_a_blob_git_cannot_produce_is_a_finding_not_a_silent_skip(tmp_path, monkeypatch):
    """Fail closed when the object store cannot hand over content it listed.

    A damaged or partially fetched repository is the realistic cause. The gate must
    say it could not read the file rather than pass over it, which is the same rule
    an undecodable file gets (Review H-1) applied one layer down.
    """
    root = _new_repo(tmp_path, "missing_object")
    (root / "wrapper.bat").write_text(CLEAN_TEXT, encoding="utf-8")
    _git_in(root, "add", "wrapper.bat")
    monkeypatch.setattr(scanner, "_read_blobs", lambda *_args, **_kwargs: {})

    findings = scanner.scan(root)
    assert [f.rule for f in findings] == [scanner.UNREADABLE_RULE], findings
    assert findings[0].path == "wrapper.bat"


def test_finetune_studio_ships_no_owner_paths():
    """The owner-paths gate scans platform/, but finetune-studio/ is a sibling
    that also ships to users - and a hardcoded absolute models.yaml path
    hid there until the pre-launch audit (M17.5). This guards that whole
    directory's git-tracked Python against drive-letter absolute paths and the
    owner's home, so the class cannot recur outside platform/."""
    import re
    import subprocess

    studio = PLATFORM_DIR.parent / "finetune-studio"
    if not studio.is_dir():
        return
    out = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=str(studio),
        capture_output=True, text=True,
    ).stdout
    offenders = []
    # A drive-letter absolute path, or the owner's user dir, baked into shipped
    # code. Comments count too - a copy-pasted example path misleads just as much.
    # The first alternative is the original dev machine's root directory name,
    # assembled so this scanner's own source never carries it as prose.
    _dev_root = "".join(("O", "perynth"))
    pattern = re.compile(r"[A-Za-z]:[\\/](?:" + _dev_root + r"|Users)")
    for rel in out.splitlines():
        f = studio / rel
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()[:70]}")
    assert offenders == [], "machine-specific paths in finetune-studio:\n" + "\n".join(offenders)
