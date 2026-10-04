"""The W1 fence's own rules, shape by shape - what fires, and what stays silent.

WHY THIS FILE EXISTS
--------------------
`scripts/verify_write_fence.py` is the structural half of invariant W1 (DEC-M14-9:
nothing LOCITIZE writes at runtime lands in the install tree). Every previous round
of review found the fence's DISCLOSED reach wider than its real reach - three
times for W1-S4 alone - because the only thing exercising the rules was the
product's own clean source, which by definition fires nothing. A rule that never
fires in the suite is a rule nobody has seen work.

So each shape below is asserted twice over: the ones the docstrings promise to
catch must produce a finding, and the ones they promise to leave alone must
produce none. When the disclosed boundary moves, this table moves with it, and a
disclosure narrower than the behaviour fails here rather than in review.

Keyword: write_fence_rules
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLATFORM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLATFORM_DIR / "scripts"))

import verify_write_fence  # noqa: E402


def _rules(source: str) -> set[str]:
    """The rule ids the fence reports for a synthetic module."""
    return {finding.rule for finding in verify_write_fence.scan_source(source, "probe.py")}


# --------------------------------------------------------------------------- #
# W1-S4: writes to a path with no absolute anchor
# --------------------------------------------------------------------------- #

# A drive-anchored root, assembled rather than spelled out. The repository's
# machine-path hygiene gate (scripts/verify_no_owner_paths.py) has no exceptions
# by design, so a literal drive letter here - even inside a synthetic fixture -
# would fail it. The fence sees the assembled string exactly as it would see a
# literal one.
_DRIVE = "C" + ":/"

# Shapes the rule claims it can PROVE are unanchored. LOCITIZE.bat does
# `cd /d "%~dp0"`, so every one of these lands in the install directory.
UNANCHORED = {
    "bare-literal": 'def g():\n    open("crash.txt", "w")\n',
    "path-literal": 'def g():\n    Path("crash.txt").write_text("x")\n',
    "path-div": 'def g():\n    (Path("logs") / "c.txt").write_text("x")\n',
    "local-name": 'def g():\n    p = Path("crash.txt")\n    p.write_text("x")\n',
    "module-constant": 'C = Path("crash.txt")\ndef g():\n    C.write_text("x")\n',
    "os-makedirs": 'def g():\n    os.makedirs("logs")\n',
    "in-a-lambda": 'g = lambda: Path("c.txt").write_text("x")\n',
    # The six shapes review round 7 (MEDIUM-2r) found missed AND undisclosed.
    "os-path-join": 'def g():\n    open(os.path.join("logs", "c.txt"), "w")\n',
    "string-concat": 'def g():\n    open("logs" + "/c.txt", "w")\n',
    "pure-literal-fstring": 'def g():\n    open(f"logs/crash.txt", "w")\n',
    "literal-format": 'def g(n):\n    open("logs/{}.txt".format(n), "w")\n',
    "str-wrapper": 'def g():\n    open(str("logs/c.txt"), "w")\n',
    "comprehension-target": 'def g():\n    [p.write_text("x") for p in [Path("c.txt")]]\n',
    # Provable even with interpolation, because the LEADING part is a literal.
    "fstring-literal-lead": 'def g(n):\n    open(f"logs/{n}.txt", "w")\n',
    "join-through-a-name": 'def g():\n    p = os.path.join("logs", "c.txt")\n    open(p, "w")\n',
    # Module-qualified opens, which MEDIUM-3r found were scored as reads.
    "io-open": 'def g():\n    io.open("crash.txt", "w")\n',
    "codecs-open": 'def g():\n    codecs.open("crash.txt", "w")\n',
    "gzip-open-binary": 'def g():\n    gzip.open("crash.gz", "wb")\n',
    # The four shapes still missed AND still undisclosed after round 7's first
    # pass at MEDIUM-2r, measured rather than assumed this time.
    "percent-format": 'def g(n):\n    open("logs/%s.txt" % n, "w")\n',
    "path-joinpath": 'def g():\n    Path("logs").joinpath("c.txt").write_text("x")\n',
    "posixpath-join": 'def g():\n    open(posixpath.join("logs", "c.txt"), "w")\n',
    "ntpath-join": 'def g():\n    open(ntpath.join("logs", "c.txt"), "w")\n',
    "conditional-both-relative": 'def g(f):\n    open("a.txt" if f else "b.txt", "w")\n',
    # Module-qualified writes, whose DESTINATION (not just whose mode) was being
    # resolved by node type: `os.replace` reported the module `os` as its own
    # write target, so W1-S4 could not see where the rename landed. That is the
    # atomic-write idiom migration.py and config.py both use.
    "os-replace-destination": 'def g(tmp):\n    os.replace(tmp, "models.yaml")\n',
    "os-rename-destination": 'def g(tmp):\n    os.rename(tmp, "models.yaml")\n',
    "os-mkdir": 'def g():\n    os.mkdir("logs")\n',
    "os-unlink": 'def g():\n    os.unlink("c.txt")\n',
    "bound-path-replace": 'def g():\n    Path("staging").replace("c.txt")\n',
    # Round 8: the shapes the docstring had spent four rounds neither covering
    # nor excluding. `.resolve()` is the sharpest of them - it makes a relative
    # path look absolute while anchoring it to the working directory, which is
    # the install tree, so it is the most plausible way this fence gets walked
    # straight past by someone acting in good faith.
    "resolve-wrapper": 'def g():\n    Path("crash.txt").resolve().write_text("x")\n',
    "absolute-wrapper": 'def g():\n    Path("crash.txt").absolute().write_text("x")\n',
    "expanduser-wrapper": (
        'def g():\n    Path("crash.txt").expanduser().write_text("x")\n'
    ),
    "with-suffix-wrapper": (
        'def g():\n    Path("crash").with_suffix(".txt").write_text("x")\n'
    ),
    "resolve-then-open": 'def g():\n    open(Path("crash.txt").resolve(), "w")\n',
    "walrus-target": 'def g():\n    (p := Path("c.txt")).write_text("x")\n',
    "walrus-binding": (
        'def g():\n    if (p := Path("c.txt")).name:\n        p.write_text("x")\n'
    ),
    "tuple-unpack": (
        'def g():\n    a, b = "x.txt", "y.txt"\n    open(a, "w")\n'
    ),
    "tuple-unpack-second-slot": (
        'def g():\n    a, b = "x.txt", "y.txt"\n    open(b, "w")\n'
    ),
}

# Shapes that must stay silent: either genuinely anchored, or not a write, or
# beyond what a static rule can prove. A fence that cries wolf gets switched off,
# so this half is asserted as hard as the other.
ANCHORED_OR_NOT_A_WRITE = {
    "absolute-posix": 'def g():\n    open("/var/log/c.txt", "w")\n',
    "absolute-drive": f'def g():\n    open("{_DRIVE}logs/c.txt", "w")\n',
    "home-tilde": 'def g():\n    open("~/c.txt", "w")\n',
    "absolute-join": f'def g():\n    open(os.path.join("{_DRIVE}logs", "c.txt"), "w")\n',
    "absolute-concat": f'def g():\n    open("{_DRIVE}logs" + "/c.txt", "w")\n',
    "absolute-fstring": 'def g(n):\n    open(f"/var/{n}.txt", "w")\n',
    "read-no-mode": 'def g():\n    open("c.txt")\n',
    "read-mode-r": 'def g():\n    open("c.txt", "r")\n',
    "io-open-read": 'def g():\n    io.open("c.txt")\n',
    "io-open-mode-r": 'def g():\n    io.open("c.txt", "r")\n',
    "bound-open-read": 'def g(p):\n    p.open()\n',
    # An argument is the disclosed blind spot: nothing here says where p points.
    "argument-write": 'def g(p):\n    p.open("w")\n',
    "unknown-name": 'def g(n):\n    open(n, "w")\n',
    # An f-string that STARTS with an interpolation is genuinely unknowable.
    "fstring-interp-lead": 'def g(n):\n    open(f"{n}/c.txt", "w")\n',
    "empty-literal": 'def g():\n    open("", "w")\n',
    # str.join is a different function that happens to share a name.
    "str-join": 'def g(parts):\n    open(", ".join(parts), "w")\n',
    # A conditional needs BOTH branches provably relative; one anchored
    # fallback and the write may well land outside the install tree.
    "conditional-one-absolute": (
        f'def g(f):\n    open("a.txt" if f else "{_DRIVE}b.txt", "w")\n'
    ),
    # str.replace shares its name with Path.replace but is not a filesystem
    # call at all. Told apart by arity: Path.replace takes exactly one
    # argument, str.replace two or three. Without that, every string operation
    # on a relative-looking literal was reported as a write.
    "str-replace-two-args": 'def g():\n    return "logs/x".replace("a", "b")\n',
    "str-replace-three-args": 'def g():\n    return "logs/x".replace("a", "b", 1)\n',
    # The leading component decides, so a non-literal lead stays silent in
    # every join and format spelling.
    "join-non-literal-lead": 'def g(root):\n    open(os.path.join(root, "c.txt"), "w")\n',
    "concat-non-literal-lead": 'def g(root):\n    open(root + "/c.txt", "w")\n',
    "percent-non-literal-lead": 'def g(root, n):\n    open(root % n, "w")\n',
    "joinpath-non-literal-lead": (
        'def g(root):\n    Path(root).joinpath("c.txt").write_text("x")\n'
    ),
    # Round 8: every remaining line of the disclosed boundary, pinned. A gap that
    # is written down but never exercised drifts into being read as covered -
    # which is precisely how W1-S4 collected four rounds of the same finding.
    "wrapper-on-an-argument": 'def g(p):\n    Path(p).resolve().write_text("x")\n',
    "attribute-receiver": 'def g(o):\n    o.path.open("w")\n',
    "subscript-lookup": 'def g(paths):\n    open(paths["log"], "w")\n',
    "parent-attribute": 'def g():\n    Path("logs").parent.mkdir()\n',
    "helper-call-return": (
        'def name():\n    return "c.txt"\ndef g():\n    open(name(), "w")\n'
    ),
    "unpack-from-a-call": (
        'def paths():\n    return ("a.txt", "b.txt")\n'
        'def g():\n    a, b = paths()\n    open(a, "w")\n'
    ),
    "unpack-with-a-star": (
        'def g():\n    a, *rest = ["x.txt", "y.txt"]\n    open(a, "w")\n'
    ),
    "augmented-from-an-argument": (
        'def g(root):\n    p = root\n    p += "/c.txt"\n    open(p, "w")\n'
    ),
    "name-from-a-sibling-scope": (
        'def h():\n    p = Path("c.txt")\ndef g(p):\n    p.write_text("x")\n'
    ),
    "walrus-on-an-argument": 'def g(root):\n    (p := Path(root)).write_text("x")\n',
}


@pytest.mark.parametrize("source", list(UNANCHORED.values()), ids=list(UNANCHORED))
def test_write_fence_rules_w1_s4_catches_every_disclosed_unanchored_shape(source):
    """Each shape the rule claims must actually produce a W1-S4 finding."""
    assert "W1-S4" in _rules(source), (
        "this write lands in the install directory and W1-S4 does not see it"
    )


@pytest.mark.parametrize(
    "source", list(ANCHORED_OR_NOT_A_WRITE.values()), ids=list(ANCHORED_OR_NOT_A_WRITE)
)
def test_write_fence_rules_w1_s4_stays_silent_on_anchored_and_non_writes(source):
    """The false-positive half. A noisy fence is a deleted fence."""
    assert _rules(source) == set(), "the fence reported a write that is not a finding"


# --------------------------------------------------------------------------- #
# The shared write table: which calls write, and what they write TO
# --------------------------------------------------------------------------- #


def _one_call(source: str):
    """The single Call node in a one-line synthetic module."""
    import ast

    return next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Path-first spellings: the destination is the FIRST argument and the
        # mode is the second, whatever module the name is reached through.
        ('open(target, "w")', "target"),
        ('open(target, mode="w")', "target"),
        ('io.open(target, "w")', "target"),
        ('codecs.open(target, "w")', "target"),
        ('gzip.open(target, "wb")', "target"),
        ('lzma.open(target, "w")', "target"),
        # Mode-first: a BOUND open, where the path is the object itself.
        ('target.open("w")', "target"),
        ('target.open(mode="w")', "target"),
        ('self.registry.open("a")', "self.registry"),
    ],
    ids=[
        "builtin-positional",
        "builtin-keyword",
        "io-open",
        "codecs-open",
        "gzip-open",
        "lzma-open",
        "bound-positional",
        "bound-keyword",
        "bound-attribute-chain",
    ],
)
def test_write_fence_rules_write_targets_resolves_the_open_destination(source, expected):
    """MEDIUM-3r: the mode and the path are found by call SHAPE, not node type.

    `io.open(path, "w")` is attribute-form AND path-first. A detector that read
    "attribute-form" as "bound Path.open, mode first" pulled the mode out of the
    path expression, found no constant, and scored the call a READ - which
    silently weakened both W1-S1 and the AC-M14-28 registry chokepoint test.
    """
    import ast

    targets = verify_write_fence.write_targets(_one_call(source))
    assert [ast.unparse(t) for t in targets] == [expected], (
        f"{source} is a write and its destination must be resolved exactly once"
    )


@pytest.mark.parametrize(
    "source",
    [
        "open(target)",
        'open(target, "r")',
        'io.open(target, "r")',
        "io.open(target)",
        "target.open()",
        'target.open("r")',
        'codecs.open(target, "r")',
    ],
    ids=[
        "builtin-default",
        "builtin-r",
        "io-r",
        "io-default",
        "bound-default",
        "bound-r",
        "codecs-r",
    ],
)
def test_write_fence_rules_write_targets_ignores_reads(source):
    """A read of the install tree is the whole point of an install tree."""
    assert verify_write_fence.write_targets(_one_call(source)) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Module functions: the destination is an ARGUMENT. Five names -
        # mkdir, unlink, rmdir, rename, replace - appear in both write tables,
        # and resolving them by node type made every one of these report the
        # module `os` as the thing being written to.
        ("os.replace(staging, destination)", ["staging", "destination"]),
        ("os.rename(staging, destination)", ["staging", "destination"]),
        ("os.mkdir(destination)", ["destination"]),
        ("os.unlink(destination)", ["destination"]),
        ("os.makedirs(destination)", ["destination"]),
        ("shutil.copy2(source, destination)", ["source", "destination"]),
        ("shutil.move(source, destination)", ["source", "destination"]),
        # Bound methods: the destination is the OBJECT before the dot.
        ("staging.replace(destination)", ["staging"]),
        ("staging.rename(destination)", ["staging"]),
        ("destination.mkdir(parents=True)", ["destination"]),
        ("destination.write_text(body)", ["destination"]),
        # str.replace is not a filesystem call, whatever it shares its name
        # with. Two or three arguments is the tell.
        ('text.replace("a", "b")', []),
        ('text.replace("a", "b", 1)', []),
    ],
    ids=[
        "os-replace",
        "os-rename",
        "os-mkdir",
        "os-unlink",
        "os-makedirs",
        "shutil-copy2",
        "shutil-move",
        "bound-replace",
        "bound-rename",
        "bound-mkdir",
        "bound-write-text",
        "str-replace-two-args",
        "str-replace-three-args",
    ],
)
def test_write_fence_rules_write_targets_resolves_module_calls_from_the_call_shape(
    source, expected
):
    """MEDIUM-3r's second half, asked of the DESTINATION rather than the mode.

    Round 7 fixed the open() MODE to be resolved by call shape and left the
    write DESTINATION resolved by node type, so `os.replace(staging, dest)` -
    the atomic-rename idiom migration.py and config.py both use - named `os`
    itself as its write target. The registry chokepoint test happened to still
    catch that shape, but only because a non-empty target list was enough for
    it; W1-S4, which asks WHERE the write lands, could not see it at all.

    Asserted as the exact target list, so "it returned something" can never
    again be mistaken for "it returned the right thing".
    """
    import ast

    targets = verify_write_fence.write_targets(_one_call(source))
    assert [ast.unparse(t) for t in targets] == expected
