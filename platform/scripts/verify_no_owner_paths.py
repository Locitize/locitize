"""Path-hygiene scanner: no machine-specific path may exist in a tracked file.

Where this fits (Architecture.md M14.2.5, DEC-M14-4): LOCITIZE is being turned into
something a stranger can install. The failure mode that keeps recurring is a
developer's own filesystem layout leaking into a committed file - a binary path
under one person's drive, a model registry row pointing at a directory nobody
else has, a home-directory literal in a docstring. Every such value is a
guaranteed first-run error on any other machine.

This script is the structural fix. It scans every file git tracks under the
platform directory - the content git HOLDS, not merely the copy on this disk, see
"What ships" below - and fails the build if any of them contains a path that could
only be true on one person's computer. It is wired into the pytest suite
(tests/test_verify_no_owner_paths.py), so such a value cannot pass the suite,
which means it cannot pass review or the release checklist either.

Design rule, deliberately absolute: there is no way to excuse a file from this
check - no skip list, no per-file opt-out, no magic comment. One rule with no
exceptions is the
entire point - any escape hatch becomes the door the next machine-specific path
walks through. Test fixtures that need a path use a neutral synthetic root such
as /locitize-test/models/x.gguf, or pytest's tmp_path, neither of which is tied to
a drive letter.

What counts as a violation, per Architecture.md M14.2.5:

1. A drive-letter absolute path (a single letter, a colon, then a separator),
   in any scanned file type. Windows environment-variable NAMES such as
   LOCALAPPDATA are fine; an expanded value is not.
2. A "Users" path segment followed by a name - the shape of a home directory - in
   ANY case, with a forward or backward separator, single or doubled (Review
   M-9). All four dimensions are needed: Windows paths are case-insensitive, WSL
   writes them lowercase with forward slashes, and every Python and JSON string
   literal doubles the backslash. Before this widening, rule 2 only fired where
   rule 1 already had, so a home directory on a UNC share - which has no drive
   letter for rule 1 to catch - passed the whole gate.
3. The shared model-tree root used on the maintainer's own box, in any case or
   separator form, single or doubled. This is a subset of rule 1 and is kept as
   its own rule so the failure message names the real cause.
4. A file the scanner cannot decode at all. Not a path rule, but the same
   guarantee: unreadable means uncleared (see Encoding below).

What is deliberately NOT a violation, and why the distinction is the point of the
whole check: an environment-variable NAME. A value like %USERPROFILE%\\ai\\models
or $env:LOCALAPPDATA names the owner's home directory in effect, but it resolves
correctly on every machine, which makes it machine-INDEPENDENT - it is the
recommended repair for a violation, not a violation. Flagging it would mean the
gate rejects its own advice, and a gate that rejects the fix it recommends is a
gate that gets switched off. The rule is about literal paths that are true on one
computer, never about which directory a value eventually points at.

Scanned file types: EVERY file in the ship set except the binary suffixes listed
in BINARY_SUFFIXES below - and even those are scanned when their bytes turn out
to be text after all, because a suffix is only a claim (see is_scannable). The
suffix rule is inverted deliberately (Review M-2): a
suffix list of "things we look at" silently narrows the guarantee every time
someone adds a new kind of text file - requirements.txt, an .ini, a .cmd
wrapper. Declaring instead which bytes cannot possibly be text excuses no file
from the rule; it only says where reading lines is meaningless.

Encoding, and why an unreadable file is a failure (Review H-1, H-3): a file this
script cannot decode is a file it cannot clear, so it is reported as a finding
rather than passed over. Windows PowerShell 5.1 writes UTF-16LE by default, so
a .ps1 carrying an owner path is the realistic case; UTF-16 and UTF-32, with or
without a byte-order mark, are therefore decoded and scanned properly.

The subtle half of that guarantee is that a decode which SUCCEEDS can still be
wrong. Reading big-endian bytes as little-endian raises nothing - it returns
byte-swapped nonsense that no path regex matches - so this scanner judges every
decode by its result rather than by the absence of an exception. See
decode_bytes. Anything that survives neither the codecs nor that judgement fails
the scan by name. Failing closed is the only behaviour consistent with "one rule
with no exceptions".

The general form of that lesson, learned four times (Review H-1, H-3, H-5)
and now written down once: NOTHING A FILE SAYS ABOUT ITSELF IS EVIDENCE. A
byte-order mark is a claim by whoever wrote the file, not a fact about the bytes
after it; a suffix is a claim about format made by whoever named the file. Both
claims are useful hints and both are checked against the actual bytes before
they are believed - see decode_bytes for the mark and is_scannable for the
suffix. A claim that the content contradicts is discarded, and the scanner falls
back to reading the bytes for what they are.

And the fifth time (Review H-6) that lesson turned out to stop one word short:
NOTHING GIT SAYS ABOUT A FILE IS EVIDENCE EITHER, UNTIL THE BLOB IS READ. This
scanner used to take its file LIST from git and its file CONTENT from the working
tree, which are two different artifacts. Where they differed it reported on the
copy that does not ship: 96 names listed, 91 files read, "OK" printed, and four
names never read at all - not even as an unreadable finding.

What ships, therefore, is read from git's object store (see ship_set):

- the HEAD copy, because that is what a clone, a CI checkout and `git archive`
  hand to a stranger;
- the index copy, because that is what the next commit will contain;
- the working-tree copy, because that is what the developer is about to commit,
  and for an untracked-but-not-ignored file it is the only copy there is.

Identical copies collapse into one scan. Where they differ, all of them are
scanned and the finding names which copy it came from, because "this line is not
in your file, it is in your commit" is a different instruction to a human than
"fix this line".

The part of that fix which generalises is not the blob reading - it is the COUNT
INVARIANT in ShipSet.check_invariant. Four of the five bugs above were a silent
narrowing of scope, and no amount of imagining new attack shapes finds the next
one. Comparing the number of names git listed against the number of names the
scan accounted for finds all of them, including the shapes nobody has thought of:
a name that is listed and not dispositioned raises, and raising means exit code 2,
scan-could-not-run. There is no arrangement of files, modes, encodings or commits
under which this scanner can report success over a name it never read.

Usage:
    python scripts/verify_no_owner_paths.py          # scan, print findings
    python scripts/verify_no_owner_paths.py --json   # machine-readable findings

Exit code 0 means clean; 1 means findings were printed; 2 means the scan itself
could not run (git missing, not a repository).
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# The platform directory (the parent of scripts/). Every scan is rooted here so
# the check has exactly one, predictable scope.
PLATFORM_DIR = Path(__file__).resolve().parent.parent

# Suffixes whose contents are bytes, not lines: images, audio, model weights,
# archives, compiled objects and fonts. Reading these for "paths" is meaningless,
# so they are not read - unless their content contradicts the suffix, which
# is_scannable checks. This is the ONLY thing the scanner declines to look at,
# and it is a statement about file format rather than about any particular file:
# no path, name or directory can put itself on this list.
BINARY_SUFFIXES = frozenset(
    {
        # images
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tif", ".tiff",
        # audio and video
        ".wav", ".mp3", ".ogg", ".flac", ".m4a", ".mp4", ".webm",
        # model weights and other large binary payloads
        ".gguf", ".bin", ".safetensors", ".pt", ".pth", ".onnx", ".npy", ".npz",
        # archives
        ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".tar", ".whl", ".jar",
        # compiled or linked objects
        ".exe", ".dll", ".so", ".dylib", ".lib", ".obj", ".pyc", ".pyd", ".msi",
        # documents, fonts, databases
        ".pdf", ".ttf", ".otf", ".woff", ".woff2", ".db", ".sqlite", ".sqlite3",
    }
)

# Byte-order marks, each with the encoding it CLAIMS and whether the body that
# follows it validates itself.
#
# The claim is a starting guess, never a verdict (Review H-5): a file can carry a
# little-endian mark and a big-endian body, and the mark is believed only if the
# text it produces survives the same inspection every other decode gets.
#
# The third field says which inspection applies. UTF-8 is structurally
# self-validating - wrong bytes raise rather than decode - so a UTF-8 body only
# has to look like text, and a legitimate UTF-8 document written in Japanese
# still reads normally. The UTF-16 and UTF-32 marks name a WIDTH and a BYTE
# ORDER, which nothing in the bytes confirms, so those bodies must also score as
# plausible text or the mark is treated as a lie.
_BOMS = (
    (codecs.BOM_UTF8, "utf-8", True),
    (codecs.BOM_UTF32_LE, "utf-32-le", False),  # must precede UTF-16LE: BOM_UTF32_LE
    (codecs.BOM_UTF32_BE, "utf-32-be", False),  # starts with the same two bytes.
    (codecs.BOM_UTF16_LE, "utf-16-le", False),
    (codecs.BOM_UTF16_BE, "utf-16-be", False),
)

# The encodings that cannot be told apart from the bytes alone once the
# byte-order mark is gone. They differ only in code-unit WIDTH (2 vs 4 bytes) and
# BYTE ORDER, and every one of them will happily "succeed" on another one's bytes
# while producing nonsense - see _decode_ambiguous below for why that matters.
_AMBIGUOUS_ENCODINGS = ("utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be")

# The characters a source or configuration file in this project is made of:
# printable ASCII plus the usual whitespace. Used as a plausibility measure, not
# as a rule about content - see _plain_text_share.
_PLAIN_TEXT_CHARS = frozenset(chr(code) for code in range(0x20, 0x7F)) | set("\t\n\r\f\v")

# Minimum share of plain ASCII characters before a byte-order-ambiguous decode is
# believed. Real files here are ASCII with at most a stray unit symbol; a decode
# that came out the wrong way round scores near zero, because byte-swapped ASCII
# lands in the CJK code-point range. Sixty percent sits far from both.
_PLAUSIBLE_TEXT_SHARE = 0.60

# Control characters no text file contains: the C0 block and DEL, minus the
# whitespace a real file legitimately uses. Their presence means the bytes are
# not text at all, whatever encoding was claimed. This test is language-neutral -
# it does not punish a document written in a non-Latin script, which is why it,
# rather than the plain-text share, is what a self-validating UTF-8 body has to
# pass.
_CONTROL_CHARS = (
    frozenset(chr(code) for code in range(0x00, 0x20))
    | {chr(0x7F)}
) - set("\t\n\r\f\v")

# How many leading bytes are read to test a binary suffix's claim against the
# actual content (see is_scannable). A prefix is enough to tell text from a
# format, and a fixed ceiling keeps the cost of the check independent of the size
# of a model file.
_SUFFIX_PROBE_BYTES = 4096


# Rule 1: a drive-letter absolute path. The leading (?<![A-Za-z0-9_]) guard is
# what keeps URL schemes out of the results: in "http://x" the character before
# the colon is "p", which is itself preceded by a letter, so it is not a lone
# drive letter. When the letter before the colon does stand alone and a
# separator follows, the line is reported.
_DRIVE_LETTER = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")

# Rule 2: a home-directory shape - a "Users" segment followed by a name.
#
# Case and separator forms are all in scope (Review M-9). Windows paths are
# case-insensitive, and the SAME home directory is written several different ways
# by the several different tools that touch this project. Writing <sep> for a
# separator, because a literal example here would be found by the scanner reading
# its own source:
#
#   C:<sep>Users<sep>someone          a shell transcript or a .bat wrapper
#   Users<sep><sep>someone            every Python and JSON string literal - the
#                                     DOUBLED separator is the normal form there,
#                                     not an exotic one
#   /mnt/c/users<sep>someone          WSL, lowercase, and no drive letter for
#                                     rule 1 to catch
#   <sep><sep>host<sep>users<sep>x    a UNC share, likewise no drive letter
#
# Only the first of those matched before this rule became case-insensitive and
# tolerant of a doubled separator, which meant rule 2 only ever fired where rule 1
# already had. The cost of case-insensitivity is that "users" is also an ordinary
# word in a web path, so the match is filtered by _HomeDirectoryRule below.
_HOME_DIR = re.compile(r"(?i)users[\\/]{1,2}[A-Za-z0-9._-]+")

# Rule 3: a common Windows model-tree root a dev machine once used, any case, either separator,
# single or doubled (the doubled form for the same string-literal reason as rule
# 2). Written as a regex rather than a literal for the same self-match reason.
_MODEL_TREE_ROOT = re.compile(r"(?i)(?<![A-Za-z0-9_])c:[\\/]{1,2}ai(?![A-Za-z0-9])")

# Characters that cannot appear inside a URL as it is written in running text or
# in a configuration value. Used to isolate the token a match sits in - see
# _HomeDirectoryRule.
# A colon is deliberately NOT a break character: the marker being looked for is
# "://", so breaking on the colon would destroy the very evidence of a scheme.
_TOKEN_BREAK = re.compile(r"""[\s"'`<>,;()\[\]{}=|]""")


class _HomeDirectoryRule:
    """Rule 2 plus the one context where a "users" segment is not a home path.

    Making rule 2 case-insensitive was necessary (see _HOME_DIR) and it widens
    matching onto ordinary web paths: a documentation link whose last two segments
    are a lowercase "users" and a name is not a machine-specific path, and a gate
    that flags documentation links is a gate somebody switches off. So a candidate
    match is discarded when it sits inside a URL, judged by the only structural
    marker a URL has: the scheme separator earlier in the same unbroken token.

    The example is described rather than written out for the reason the whole file
    is written this way - a literal one here would be found by the scanner reading
    its own source. The real pin for this behaviour is a test, not a comment: see
    test_verify_no_owner_paths.py, group 3.

    Two deliberate limits, both stated so they are boundaries rather than
    surprises:

    - A match containing a BACKSLASH is never discarded. No URL path contains
      one, so the URL context cannot be used to hide a Windows path.
    - A URL that genuinely embeds a local path (a file:// link to a home
      directory) is discarded by this rule and caught by rule 1 instead, because
      such a form carries a drive letter.

    Exposes .search(line) so it is interchangeable with a compiled pattern in
    _RULES; the caller does not need to know which of the two it holds.
    """

    def search(self, line: str) -> re.Match[str] | None:
        """Return the first match that is not part of a URL, or None."""
        for match in _HOME_DIR.finditer(line):
            if "\\" in match.group(0):
                return match
            token = _TOKEN_BREAK.split(line[: match.start()])[-1]
            if "://" not in token:
                return match
        return None


# Rule 4 is not a regex: a file whose bytes cannot be decoded is reported under
# this name so the failure message says why the gate could not clear it.
UNREADABLE_RULE = "unreadable file (hygiene gate cannot decode it)"

_RULES = (
    (_DRIVE_LETTER, "drive-letter absolute path"),
    (_HomeDirectoryRule(), "home-directory path segment"),
    (_MODEL_TREE_ROOT, "machine-specific model tree root"),
)


@dataclass(frozen=True)
class Finding:
    """One rule violation: which file, which line, which rule, and the text."""

    path: str
    line_no: int
    rule: str
    text: str

    def render(self) -> str:
        """One grep-style line a human can paste straight into an editor."""
        return f"{self.path}:{self.line_no}: {self.rule}: {self.text.strip()}"


def _git(root: Path, args: list[str], stdin: bytes | None = None) -> bytes:
    """Run one git command in `root` and return its raw stdout BYTES.

    Bytes, not text, because some callers here read file CONTENT out of git. A
    blob is not necessarily UTF-8 - it can be UTF-16, or a real binary - and
    decoding it at the subprocess boundary would destroy the exact bytes this gate
    exists to inspect.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        input=stdin,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip() or "unknown error"
        raise RuntimeError(f"git {' '.join(args)} failed in {root}: {detail}")
    return result.stdout


def _git_records(root: Path, args: list[str]) -> list[str]:
    """Run a NUL-terminated git listing and return its non-empty records.

    Names are decoded with surrogateescape so a filename that is not valid UTF-8
    survives the round trip into a Path instead of raising - an unreadable NAME
    must not be a way to leave a file unscanned.
    """
    output = _git(root, args).decode("utf-8", "surrogateescape")
    return [record for record in output.split("\0") if record]


def _git_names(root: Path, args: list[str]) -> list[str]:
    """Run one `git ls-files` variant in `root` and return its NUL-split names."""
    return _git_records(root, ["ls-files", "-z", *args])


def _index_blobs(root: Path) -> dict[str, list[str]]:
    """Map each tracked name to the object id(s) of the content git has STAGED.

    `git ls-files -s` is the only enumeration that reports an object id, which is
    what makes the staged content readable without touching the disk. Record
    format: "<mode> <object-id> <stage>\\t<name>".

    A name can appear more than once - an unresolved merge keeps stages 1, 2 and 3
    of the same path - so the object ids are collected in a list and every one of
    them is scanned. A conflicted checkout is exactly the situation where the disk
    copy and the copies git holds are most likely to differ.

    The MODE is deliberately not consulted. A mode-120000 entry is a symbolic
    link, and its blob content is the link TARGET as text; reading that blob like
    any other is what makes an owner path stored as a link target visible
    (Review H-6, probe E2). Special-casing the mode would put it back out of
    reach.
    """
    blobs: dict[str, list[str]] = {}
    for record in _git_records(root, ["ls-files", "-s", "-z"]):
        meta, name = record.split("\t", 1)
        _mode, object_id, _stage = meta.split()
        blobs.setdefault(name, []).append(object_id)
    return blobs


def _head_blobs(root: Path) -> dict[str, list[str]]:
    """Map each committed name to the object id of the content HEAD holds.

    This is the copy a release actually ships: a clone, a CI checkout and
    `git archive HEAD` all deliver HEAD, which is stated verbatim in
    verify_ship_set.py's own docstring. Scanning it is the whole point of Review
    H-6 - the index can be clean while HEAD is not.

    Run with `root` as the working directory, `git ls-tree -r HEAD` reports names
    relative to that directory and restricted to that subtree, so its names line
    up with `git ls-files` without any prefix arithmetic.

    A repository with no commits has no HEAD. That is not an error - there is
    simply nothing committed to scan - and it is the state every throwaway test
    fixture starts in, so it returns an empty mapping rather than raising.
    """
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "HEAD"],
        cwd=str(root),
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return {}
    blobs: dict[str, list[str]] = {}
    for record in _git_records(root, ["ls-tree", "-r", "-z", "HEAD"]):
        meta, name = record.split("\t", 1)
        _mode, kind, object_id = meta.split()
        # -r flattens trees, so every record is a blob or a submodule commit. A
        # submodule has no content in this repository and nothing to scan.
        if kind == "blob":
            blobs.setdefault(name, []).append(object_id)
    return blobs


def _read_blobs(root: Path, object_ids: set[str]) -> dict[str, bytes]:
    """Read many blobs in ONE git process and return {object id: bytes}.

    `git cat-file --batch` takes object ids on stdin and answers each with a
    header line "<id> <type> <size>" followed by exactly <size> bytes and a
    newline. One process for the whole tree instead of one per file is what keeps
    reading the object store as cheap as reading the disk was.

    The whole tree is held in memory at once. That is a deliberate choice for a
    source repository of this size and it is bounded by the same rule that bounds
    the disk scan: real binaries are set aside by their content, and nothing here
    is larger than a source file. If that stops being true, this is the function
    to stream.
    """
    if not object_ids:
        return {}
    ordered = sorted(object_ids)
    payload = _git(root, ["cat-file", "--batch"], stdin=("\n".join(ordered) + "\n").encode())
    blobs: dict[str, bytes] = {}
    offset = 0
    while offset < len(payload):
        end_of_header = payload.index(b"\n", offset)
        header = payload[offset:end_of_header].decode("utf-8", "replace").split()
        offset = end_of_header + 1
        if len(header) < 3:
            # "<id> missing" - an object git listed but cannot produce. Fail
            # closed: the caller reports the name as unreadable rather than
            # passing over it.
            continue
        object_id, _kind, size = header[0], header[1], int(header[2])
        blobs[object_id] = payload[offset : offset + size]
        offset += size + 1  # +1 for the newline git writes after the content
    return blobs


def is_scannable(name: str, data: bytes) -> bool:
    """True when this content can be read as text, so it gets scanned.

    Two questions, and the second one exists because the first is a CLAIM. A
    suffix is chosen by whoever named the file; it is not a fact about the
    content. So:

    1. Is the suffix a known binary format? If not - .py, .txt, .cfg, .cmd, a
       build file with no suffix at all - it is text and gets scanned.
    2. If it is, do the bytes agree? The leading bytes are decoded. A real image
       or model file produces nothing believable and is left alone, as intended.
       Content NAMED like a binary that is plainly text has been mislabelled, and
       it is scanned like the text it is - the same treatment a lying byte-order
       mark gets in decode_bytes, applied to the other claim a file makes about
       itself.

    Takes a name and BYTES rather than a path (Review H-6): the same decision has
    to be made about content that has no file on disk at all, because it came out
    of git's object store.
    """
    if PurePosixPath(name).suffix.lower() not in BINARY_SUFFIXES:
        return True
    return _binary_suffix_is_contradicted(data)


def _binary_suffix_is_contradicted(data: bytes) -> bool:
    """True when content with a binary suffix actually holds plain text.

    Inspects at most _SUFFIX_PROBE_BYTES and applies the strict test: the prefix
    has to decode into believable plain text. A prefix cut mid-character simply
    fails to decode, which lands on the safe side - the content keeps its declared
    binary format and is left unscanned, exactly as before this check existed.

    Only a prefix is inspected, so the cost does not grow with the size of a model
    file.
    """
    prefix = data[:_SUFFIX_PROBE_BYTES]
    if not prefix:
        return False
    text = decode_bytes(prefix)
    return text is not None and _is_believable(text)


def _plain_text_share(text: str) -> float:
    """Share of `text` that is printable ASCII or ordinary whitespace (0.0-1.0).

    This is a plausibility measure for a decode, not a judgement about content.
    A file decoded with the right codec is nearly all plain characters; the same
    bytes decoded with the wrong byte order come out as a wall of code points in
    the CJK range, which scores about zero. An empty file scores 1.0 - there is
    nothing implausible about it.
    """
    if not text:
        return 1.0
    plain = sum(1 for character in text if character in _PLAIN_TEXT_CHARS)
    return plain / len(text)


def _reads_as_text(text: str) -> bool:
    """True when a decode contains nothing that text files never contain.

    The weaker of the two inspections, and the one that applies to a
    self-validating UTF-8 body. It asks only: are there control characters in
    here? A NUL means the codec read one byte where the file meant two or four
    (this alone closed the BOM-less UTF-16LE bypass, because such a file is
    byte-for-byte VALID UTF-8 and decodes "successfully" into
    C-NUL-:-NUL-backslash-NUL). Any other C0 character or DEL means the bytes
    were never text - a binary file that happened to decode.

    Deliberately says nothing about which script or language the text is in, so a
    legitimate non-English UTF-8 file passes it unharmed.
    """
    return not any(character in _CONTROL_CHARS for character in text)


def _is_believable(text: str) -> bool:
    """True when a decode looks like text a person wrote rather than noise.

    The stronger inspection, applied wherever the WIDTH or BYTE ORDER of the
    bytes was guessed or merely claimed. Two questions:

    1. Does it read as text at all (no control characters)?
    2. Is the result mostly plain characters? This catches what the control test
       cannot: a decode that came out the wrong way round contains no control
       characters at all, just a wall of CJK-range code points built from swapped
       ASCII pairs.
    """
    return _reads_as_text(text) and _plain_text_share(text) >= _PLAUSIBLE_TEXT_SHARE


def _try_decode(data: bytes, encoding: str) -> str | None:
    """Decode with one codec, returning None instead of raising.

    A raise and an implausible result are the same thing to every caller here -
    this codec did not read the file - so they are collapsed into one value.
    """
    try:
        return data.decode(encoding)
    except (UnicodeDecodeError, ValueError):
        return None


def _decode_ambiguous(data: bytes) -> str | None:
    """Decode bytes whose width and byte order are unknown, or return None.

    Why every candidate is tried and then SCORED rather than returned in order
    (Review H-3): decoding is not a test of correctness. UTF-16LE reading
    big-endian bytes does not raise - it returns byte-swapped mojibake with no
    NUL in it, which the first-match-wins loop this replaced accepted as the
    file's text. Every path regex then missed, and a machine-specific path
    sailed through the gate reporting success.

    The same trap exists for every pair in this family, in both directions and
    at both widths, so the fix is aimed at the class: decode with all four,
    discard the ones that raise or come out implausible, and keep the most
    plausible survivor. If nothing survives, the caller reports the file as
    unreadable, which fails the gate. Wrong-but-quiet is the one outcome that
    must be impossible.

    The cost of being strict: a genuinely non-English UTF-16 file (say a script
    written in Japanese) scores below the threshold and is reported unreadable
    rather than scanned. That is the deliberate direction to be wrong in - it
    asks a human to look, instead of quietly clearing a file nobody read.
    """
    candidates: list[tuple[float, str]] = []
    for encoding in _AMBIGUOUS_ENCODINGS:
        text = _try_decode(data, encoding)
        if text is not None and _is_believable(text):
            candidates.append((_plain_text_share(text), text))
    if not candidates:
        return None
    return max(candidates, key=lambda scored: scored[0])[1]


def decode_bytes(data: bytes) -> str | None:
    """Decode a ship-set file to text, or return None if it cannot be read.

    Returning None is a failure signal, not a pass: `scan()` turns it into a
    finding (Review H-1). Three steps, in decreasing order of how much evidence
    there is for the encoding - and NO step is trusted on its say-so. Every
    candidate result, from every branch, goes through the same acceptance test
    before it is returned, so a branch cannot forget to look at what it decoded
    (Review H-5):

    1. A byte-order mark CLAIMS an encoding. It is tried first because it is
       usually right, and its result is inspected because it can be wrong: a
       file written with a little-endian mark and a big-endian body decodes into
       byte-swapped nonsense that no path regex matches. When the mark's claim
       fails inspection the mark is discarded and the body is re-read by step 3,
       which reads it correctly - so a mislabelled file is scanned properly
       rather than merely reported.
    2. UTF-8 is what everything in this project is written as, and unlike a mark
       it validates itself: wrong bytes usually raise rather than decode. Its
       result only has to read as text (no control characters), so a legitimate
       UTF-8 document full of non-ASCII characters is read, not doubted.
    3. Otherwise the file may be BOM-less UTF-16 or UTF-32 - what Windows
       PowerShell 5.1 writes by default, and the realistic way a .ps1 carrying an
       owner path enters the tree. Nothing names the width or byte order there,
       so _decode_ambiguous tries all four and scores the results.
    """
    for bom, encoding, self_validating in _BOMS:
        if data.startswith(bom):
            body = data[len(bom):]
            text = _try_decode(body, encoding)
            inspect = _reads_as_text if self_validating else _is_believable
            if text is not None and inspect(text):
                return text
            # The mark lied, or the body is not text. Re-read the body with no
            # regard for what it claimed; if some codec produces believable text,
            # THAT is the file's real content and it gets scanned. If none does,
            # the file is unreadable and becomes a finding.
            return _decode_ambiguous(body)

    text = _try_decode(data, "utf-8")
    if text is not None and _reads_as_text(text):
        return text

    return _decode_ambiguous(data)


@dataclass(frozen=True)
class ScanEntry:
    """One unit of content to scan: where it came from, and its bytes.

    `origins` names every copy that holds these exact bytes, in the order
    worktree, index, HEAD. The ordinary case - a file whose three copies agree -
    is one entry with all three origins, which is why the common output is
    unchanged by Review H-6's fix.
    """

    name: str
    origins: tuple[str, ...]
    data: bytes

    @property
    def label(self) -> str:
        """The name a finding reports.

        A copy present in the working tree is reported by its plain name, because
        that is the file a person opens. Content that exists ONLY inside git gets
        its origin appended, because "the line is not in your file, it is in your
        commit" is the single most important thing such a finding has to say.
        """
        if "worktree" in self.origins:
            return self.name
        return f"{self.name} [{'+'.join(self.origins)}]"


@dataclass(frozen=True)
class ShipSet:
    """Everything that would ship, plus a full account of what happened to it.

    Three fields, and the third exists because Review H-6 was a SILENT DROP: the
    gate listed 96 names, scanned 91, and said OK. Nothing in the code compared
    those two numbers, so nothing could notice.

    - `entries`   - the content actually scanned.
    - `listed`    - every name git named, from any of the queries.
    - `accounted` - one disposition sentence per listed name, saying what became
      of it. Empty is impossible: see check_invariant.
    - `unreadable`- listed names whose content could not be obtained at all,
      mapped to why. These become findings, never silent skips.
    """

    entries: list[ScanEntry]
    listed: list[str]
    accounted: dict[str, str]
    unreadable: dict[str, str]

    def check_invariant(self) -> None:
        """Raise unless every listed name was accounted for. The H-6 guard.

        This is the assertion that generalises past the variant that prompted it.
        The four shapes Review found - content in HEAD, content in the index, a
        tracked file deleted from disk, a symlink entry - are four ways for a name
        to be listed and never read. Comparing the count of names listed against
        the count of names dispositioned catches ALL of them, including the fifth
        nobody has thought of yet.

        It raises rather than asserting, because `python -O` strips assertions and
        a guarantee that evaporates under a flag is not a guarantee. RuntimeError
        reaches main() as exit code 2, "the scan could not run" - the correct
        fail-closed answer when the gate cannot account for its own scope.
        """
        listed = set(self.listed)
        dispositioned = set(self.accounted)
        if listed != dispositioned:
            missing = sorted(listed - dispositioned)
            extra = sorted(dispositioned - listed)
            raise RuntimeError(
                "path-hygiene scope invariant violated: "
                f"{len(listed)} name(s) listed by git, {len(dispositioned)} "
                f"accounted for. Never read: {missing}. Not listed: {extra}"
            )
        blank = sorted(name for name, why in self.accounted.items() if not why)
        if blank:
            raise RuntimeError(
                f"path-hygiene scope invariant violated: no disposition for {blank}"
            )
        unlisted = sorted({entry.name for entry in self.entries} - listed)
        if unlisted:
            raise RuntimeError(
                f"path-hygiene scope invariant violated: scanned but not listed: {unlisted}"
            )


def _candidate_contents(
    root: Path,
    name: str,
    index_ids: list[str],
    head_ids: list[str],
    blobs: dict[str, bytes],
    read_worktree: bool,
) -> tuple[list[tuple[str, bytes]], str | None]:
    """Collect every distinct copy of one name, with the origin(s) of each.

    Returns (copies, failure). `copies` is a list of (origin-label, bytes) with
    identical bytes collapsed into a single copy whose origin label names all the
    places that hold it. `failure` is set only when NO copy could be obtained at
    all, which is the fail-closed case.

    Why the disk copy and the git copies are all collected (Review H-6): they
    answer different questions. The disk copy is what the developer is about to
    commit. The index copy is what the next commit will contain. The HEAD copy is
    what a clone, a checkout or `git archive` hands to a stranger today. The gate
    promises that no machine-specific path SHIPS, so the copy that ships has to be
    the copy that is read.

    `read_worktree` is what keeps the user's own data out of the scan, and it is
    not a detail. A file that HEAD still carries but the index has dropped - a
    committed settings.yaml that this milestone is in the middle of moving out of
    the tree and into .gitignore - is exactly that shape. Its committed blob still
    ships and must be scanned; its working copy is now the user's live
    configuration, legitimately full of their own paths, and reading it would
    report the user's machine to the user as a defect. So the working copy is read
    only when git still counts it as part of the ship set: tracked in the index, or
    untracked and not ignored. The caller decides; see ship_set.
    """
    by_content: dict[bytes, list[str]] = {}
    contents: dict[bytes, bytes] = {}

    def remember(origin: str, data: bytes) -> None:
        # Keyed by hash so three identical copies collapse into one scan and one
        # finding rather than three of each.
        digest = hashlib.sha256(data).digest()
        by_content.setdefault(digest, []).append(origin)
        contents[digest] = data

    path = root / name
    if read_worktree and path.is_file():
        try:
            remember("worktree", path.read_bytes())
        except OSError as exc:
            return [], f"cannot read the working-tree copy: {exc}"

    for origin, object_ids in (("index", index_ids), ("HEAD", head_ids)):
        for object_id in object_ids:
            if object_id not in blobs:
                # git listed the object and then could not produce it - a damaged
                # or pruned object store. Fail closed: a copy that cannot be read
                # is a copy that cannot be cleared, which is the same rule an
                # undecodable file gets.
                return [], (
                    f"git names object {object_id} as the {origin} copy of this "
                    "file, but the object store could not produce it"
                )
            remember(origin, blobs[object_id])

    if not by_content:
        return [], (
            "git tracks this name but no copy of its content could be obtained "
            "from the working tree, the index or HEAD"
        )
    copies = [
        ("+".join(origins), contents[digest]) for digest, origins in by_content.items()
    ]
    return copies, None


def ship_set(root: Path = PLATFORM_DIR) -> ShipSet:
    """Return everything that would ship from `root`, with content and accounting.

    "Would ship" is derived from git, deliberately not from a hand-maintained
    list, and it is now derived in BOTH directions - names and content:

    - Names come from two git queries: files git already tracks, plus files that
      are untracked but NOT ignored (`--others --exclude-standard`), so a
      brand-new source file is checked the moment it is written, before anyone
      remembers to add it.
    - Content for a TRACKED name comes from git's object store, and from the
      working tree when a file is there too. Both are scanned when they differ
      (Review H-6). Reading only the disk was a hole: a tracked name with no file
      on disk was dropped with no finding of any kind, and content committed and
      then cleaned in the working copy scanned green while `git archive HEAD`
      still shipped it.
    - Content for an UNTRACKED name can only come from the disk - it has no blob -
      and that path is preserved exactly, because it is what catches brand-new
      work.

    What the union excludes is exactly what should be excluded: gitignored files
    on disk. Those are the user's own generated data - the live settings.yaml and
    models.yaml that M14 moves into the data root, logs, chat memory, fetched
    binaries. Their contents are legitimately machine-specific and never ship. The
    exclusion is a property of where a file sits in git, never a name this script
    knows about.

    Those two rules meet in one case worth stating plainly, because getting it
    wrong reports a user's own configuration to them as a defect: a file that HEAD
    still carries and the index has dropped. Its committed blob IS scanned, because
    a clone of HEAD still delivers it. Its working copy is NOT, because git no
    longer counts that copy as part of the ship set. See _candidate_contents.

    Every listed name leaves this function with a disposition recorded against it,
    and the invariant that says so is checked here rather than left to a caller.
    """
    index_blobs = _index_blobs(root)
    head_blobs = _head_blobs(root)
    untracked = _git_names(root, ["--others", "--exclude-standard"])

    listed = sorted(set(index_blobs) | set(head_blobs) | set(untracked))
    blobs = _read_blobs(
        root,
        {oid for ids in index_blobs.values() for oid in ids}
        | {oid for ids in head_blobs.values() for oid in ids},
    )

    entries: list[ScanEntry] = []
    accounted: dict[str, str] = {}
    unreadable: dict[str, str] = {}
    in_ship_set_now = set(index_blobs) | set(untracked)

    for name in listed:
        copies, failure = _candidate_contents(
            root,
            name,
            index_blobs.get(name, []),
            head_blobs.get(name, []),
            blobs,
            read_worktree=name in in_ship_set_now,
        )
        if failure is not None:
            unreadable[name] = failure
            accounted[name] = f"NOT READ: {failure}"
            continue

        scanned: list[str] = []
        set_aside: list[str] = []
        for origins, data in copies:
            if is_scannable(name, data):
                entries.append(ScanEntry(name, tuple(origins.split("+")), data))
                scanned.append(origins)
            else:
                set_aside.append(origins)

        parts = []
        if scanned:
            parts.append("scanned: " + ", ".join(scanned))
        if set_aside:
            parts.append("binary content, not read as text: " + ", ".join(set_aside))
        # These notes are why the accounting is worth keeping rather than just
        # counting: each one names a state in which the disk and git disagree, and
        # every one of them used to be invisible.
        if name not in index_blobs and name not in head_blobs:
            parts.append("untracked but not ignored, disk copy only")
        elif name not in in_ship_set_now:
            parts.append(
                "no longer in the ship set (still in HEAD); the working copy is "
                "the user's own data and is out of scope"
            )
        elif not (root / name).is_file():
            # Visible rather than dropped. This is the state the four deleted
            # LOCITIZE *.bat wrappers are in, and the state Review H-6 measured as a
            # silent skip of four names.
            parts.append("absent from the working tree, read from the object store")
        accounted[name] = "; ".join(parts)

    result = ShipSet(
        entries=sorted(entries, key=lambda e: (e.name, e.origins)),
        listed=listed,
        accounted=accounted,
        unreadable=unreadable,
    )
    result.check_invariant()
    return result


def scan_text(text: str, label: str) -> list[Finding]:
    """Apply every rule to one file's text, returning all violations.

    Kept separate from file I/O so the tests can exercise the rules against
    in-memory strings without writing anything to disk.
    """
    findings: list[Finding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern, rule in _RULES:
            if pattern.search(line):
                findings.append(Finding(label, line_no, rule, line))
    return findings


def scan(root: Path = PLATFORM_DIR) -> list[Finding]:
    """Scan every copy of every in-scope name under `root`."""
    return scan_ship_set(ship_set(root))


def scan_ship_set(payload: ShipSet) -> list[Finding]:
    """Apply every rule to an already-enumerated ship set.

    Split from scan() so a caller that needs the accounting as well as the
    findings - main(), and the tests - enumerates git ONCE instead of twice.
    """
    findings: list[Finding] = []

    for name, why in sorted(payload.unreadable.items()):
        findings.append(Finding(name, 0, UNREADABLE_RULE, why))

    for entry in payload.entries:
        label = entry.label
        text = decode_bytes(entry.data)
        if text is None:
            # Fail closed. A file the gate cannot read is a file it cannot
            # clear, so it becomes a finding naming itself. The alternative -
            # passing over it quietly - was a working bypass: a UTF-16LE .ps1
            # carrying an owner path scanned clean (Review H-1).
            findings.append(
                Finding(
                    label,
                    0,
                    UNREADABLE_RULE,
                    "no encoding this gate tries produced believable text; "
                    "re-save it as UTF-8, or "
                    "add its suffix to BINARY_SUFFIXES if it is a binary format",
                )
            )
            continue

        findings.extend(scan_text(text, label))
    return findings


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Fail if any tracked platform file carries a machine-specific path."
    )
    parser.add_argument(
        "--json", action="store_true", help="emit findings as JSON on stdout"
    )
    args = parser.parse_args(argv)

    try:
        payload = ship_set()
        findings = scan_ship_set(payload)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"path-hygiene scan could not run: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "path": f.path,
                        "line": f.line_no,
                        "rule": f.rule,
                        "text": f.text.strip(),
                    }
                    for f in findings
                ],
                indent=2,
            )
        )
    else:
        for finding in findings:
            print(finding.render())

    if findings:
        print(
            f"\nFAIL: {len(findings)} hygiene finding(s) in tracked files.\n"
            "Replace each machine-specific path with an empty default, a relative "
            "path, an environment variable name, or a neutral synthetic test root. "
            "Re-save any unreadable file as UTF-8.\n"
            "A finding whose name ends in [index] or [HEAD] is NOT in the file on "
            "disk - that copy is already clean. It is in the content git holds, "
            "which is what a clone or `git archive` delivers, so it is fixed by "
            "committing the clean file rather than by editing it.",
            file=sys.stderr,
        )
        return 1

    # With --json, stdout must contain the JSON document and nothing else, so a
    # caller can pipe it straight into a parser (Review L-4). The human summary
    # still gets printed, on stderr, where it cannot corrupt the payload.
    #
    # The summary reports BOTH counts on purpose (Review H-6): the old message
    # named only the number of files read, which is exactly how a scan of 91 files
    # could report success over a list of 96 names without anyone noticing the
    # difference.
    summary = (
        f"OK: no machine-specific paths in {len(payload.entries)} content "
        f"copies covering all {len(payload.listed)} name(s) git lists."
    )
    print(summary, file=sys.stderr if args.json else sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
