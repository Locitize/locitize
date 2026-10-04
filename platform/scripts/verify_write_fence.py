"""Invariant W1: at runtime, locitize writes nothing beneath the install directory.

WHAT THIS IS (DEC-M14-9, Architecture M14.2.3.1)
------------------------------------------------
Every byte the user owns - settings, models, transcripts, logs, the Open WebUI
chat database, reports, the generated Caddyfile - lives under the data root. The
install tree holds the program and nothing else, and is read-only once locitize is
running. That is what makes "back up one folder" true, a `Program Files` install
possible, and a reinstall safe.

This scanner is the structural half of proving it. It reads the runtime modules
and reports any place where a path derived from `settings.base_dir` (the INSTALL
directory) reaches a write. The behavioural half - running the product against a
watched install tree and diffing it afterwards - lives in
tests/test_data_root_ownership.py, because a static rule alone would never have
caught the shipped defect either.

FOUR RULES, and why each exists
-------------------------------
W1-S1  A path derived from `.base_dir` must not reach a write call (mkdir,
       write_text, open("w"/"a"), a rotating log handler, shutil.copy*, ...).
       Taint follows local assignments AND function returns: the shipped defect
       (NEW-QA-M14-8) was exactly a base_dir-derived path RETURNED by
       logger.resolve_log_dir and written by a handler in another function, so a
       rule that stopped at the function boundary would have scored it clean.

W1-S2  Directly writing `settings.base_dir / "something"` at the call site,
       which is the shorter shape of the same mistake.

W1-S3  `.base_dir` must not be passed into a directory resolver (a callee whose
       name looks like *_dir / *_path / *_root / resolve_*). This is the one
       that catches memory.resolve_memory_dir(settings.base_dir, ...): the write
       happens deep inside a class the caller never names, so no reachable
       write-call rule can see it, but the seeding of a data-directory resolver
       with the install directory is itself the defect.

W1-S4  A write to a RELATIVE path - one whose leading component is a string
       literal with no absolute anchor. This rule exists because locitize.bat line 9
       does `cd /d "%~dp0"`, so the process working directory IS the install
       directory: a relative write mentions no `base_dir` and would sail past
       S1-S3 while landing squarely in the install tree (review round 6,
       MEDIUM-1). The same reasoning is why every ServiceSpec now sets an
       explicit `cwd` - see tests/test_data_root_ownership.py for the
       child-process half, which a static rule cannot see.

       Its exact reach - which shapes fire and which are the measured blind
       spots - is documented on `_is_relative_literal` and pinned shape by shape
       in tests/test_write_fence_rules.py, so the disclosed limit can never
       again be narrower than the behaviour (review round 7, MEDIUM-2r).

WHAT IS NOT A FINDING
---------------------
Reading from the install tree is the whole point of an install tree: the shipped
docs path the launcher prints, the free-space probe on the install drive, the
venv interpreters locitize launches children from, and config's copy of a shipped
*.default.yaml template all compose install paths and none of them writes there.
Install-time code (scripts/, setup) is excluded by path, per M14.2.3.1.

USAGE
-----
  python scripts/verify_write_fence.py          # exit 0 clean, 1 with findings
  python scripts/verify_write_fence.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent

# The attribute that names the INSTALL directory. `_base_dir` is the launcher's
# own private copy of the same value, which is passed to Config.load.
INSTALL_ATTRS = ("base_dir", "_base_dir")

# Method names that write through a path object.
WRITE_METHODS = frozenset(
    {
        "mkdir",
        "write_text",
        "write_bytes",
        "touch",
        "rename",
        "replace",
        "unlink",
        "rmdir",
    }
)

# Bare callables that write, and the position of their destination argument.
# open()/Path.open() are handled separately because only a writing MODE counts.
WRITE_FUNCTIONS: dict[str, tuple[int, ...]] = {
    "makedirs": (0,),
    "mkdir": (0,),
    "remove": (0,),
    "unlink": (0,),
    "rmtree": (0,),
    "copy": (0, 1),
    "copy2": (0, 1),
    "copyfile": (0, 1),
    "copytree": (0, 1),
    "move": (0, 1),
    "replace": (0, 1),
    "rename": (0, 1),
    # A rotating/plain file handler opens its path for append the moment it is
    # constructed, which is how logs land on disk.
    "RotatingFileHandler": (0,),
    "FileHandler": (0,),
    "TimedRotatingFileHandler": (0,),
}

# Callee names that mean "resolve a directory or file location" (W1-S3).
RESOLVER_NAME = re.compile(r"(^resolve_|_dir$|_path$|_root$|_file$)")

# Modules whose `open` is the builtin under another spelling: the PATH is the
# first argument and the MODE is the second, exactly as for the builtin. They are
# listed by name because `io.open(path, "w")` is attribute-form, and a detector
# that decided "attribute-form means a bound Path.open, so the mode is first"
# read the mode out of the path expression and scored the write a READ - blinding
# both W1-S1 and the AC-M14-28 registry test (review round 7, MEDIUM-3r).
PATH_FIRST_OPEN_MODULES = frozenset({"io", "codecs", "gzip", "bz2", "lzma"})

# Dotted owners that are MODULES, so `<owner>.<name>(...)` is a plain function
# call and its destination is an ARGUMENT - never the thing before the dot. Five
# names appear in both write tables (mkdir, unlink, rmdir, rename, replace), and
# without this set the bound-method branch claimed `os.replace(staging, dest)`
# and reported the write target as `os` itself. That is the same "decide from the
# call shape, not the node type" defect MEDIUM-3r fixed for the open() MODE, left
# unfixed for the open() DESTINATION - and `os.replace` is the atomic-rename
# idiom migration.py and config.py both use, so the shape most likely to be a
# second registry writer was the shape the shared table could not see.
FUNCTION_OWNER_MODULES = frozenset(
    {"os", "os.path", "shutil", "io", "codecs", "gzip", "bz2", "lzma", "logging",
     "logging.handlers", "pathlib", "tempfile", "posixpath", "ntpath"}
)

# Callees that hand back their argument unchanged as a path/string. Unwrapped by
# W1-S4 so `open(str("logs/c.txt"), "w")` is not laundered into invisibility.
PATH_PASSTHROUGH = frozenset({"str", "fspath", "os.fspath"})

# Path METHODS that hand back the same location the receiver named, so the
# receiver decides whether the result is anchored. `.resolve()` is the one that
# matters most: it turns a relative path into an absolute one by anchoring it to
# the WORKING DIRECTORY, which for LOCITIZE is the install tree - so
# `Path("crash.txt").resolve().write_text(...)` writes exactly where W1-S4
# exists to stop, while reading as absolute to a human. These four were disclosed
# as unhandled for three review rounds without being handled (round 8, MEDIUM).
# `.with_suffix`/`.with_name` change the last component only; `.expanduser()` on
# an unanchored literal has nothing to expand and returns it unchanged.
PATH_PRESERVING_METHODS = frozenset(
    {"resolve", "absolute", "expanduser", "with_suffix", "with_name", "with_stem"}
)

# Dotted callees that join path components, where the FIRST component decides
# whether the result is anchored. Matched on the dotted prefix so that
# `",".join(parts)` - str.join, a completely different function - is not treated
# as a path join.
PATH_JOIN_PREFIXES = frozenset({"os.path", "posixpath", "ntpath", "path"})

# Modes that mean the file is opened for writing.
WRITE_MODE = re.compile(r"[waxWAX+]")


@dataclass(frozen=True)
class Finding:
    """One place where an install-directory path could be written."""

    rule: str
    module: str
    line: int
    detail: str

    def render(self) -> str:
        return f"{self.module}:{self.line}: {self.rule}: {self.detail}"


def runtime_modules(root: Path = PLATFORM_DIR) -> list[Path]:
    """The modules that run while LOCITIZE is running.

    Top-level *.py only: tests/ is not the product, and scripts/ is install-time
    tooling that is ALLOWED to write into the install tree (fetch_binaries.py
    puts the runtime binaries there), which M14.2.3.1 excludes by path.
    """
    return sorted(p for p in root.glob("*.py") if p.name != "conftest.py")


def _is_install_attr(node: ast.AST) -> bool:
    """True for `<anything>.base_dir` / `<anything>._base_dir`."""
    return isinstance(node, ast.Attribute) and node.attr in INSTALL_ATTRS


def _mentions_install(node: ast.AST, tainted_names: set[str], tainted_calls: set[str]) -> bool:
    """True when this expression derives from the install directory.

    Derivation is deliberately generous - any subexpression is enough - because
    every real shape (`Path(s.base_dir) / "logs"`, `str(base)`, `base.parent`)
    is a wrapper around the same value, and a rule that tried to enumerate the
    legal wrappers would be a list to forget an entry from.
    """
    for child in ast.walk(node):
        if _is_install_attr(child):
            return True
        if isinstance(child, ast.Name) and child.id in tainted_names:
            return True
        if isinstance(child, ast.Call):
            name = _callee_name(child.func)
            if name and name in tainted_calls:
                return True
    return False


def _callee_name(func: ast.AST) -> str:
    """The simple name of a callee: `f`, `mod.f` and `obj.f` all give "f"."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _dotted_name(node: ast.AST) -> str:
    """The dotted source spelling of a name/attribute chain, or "".

    `os` -> "os", `os.path` -> "os.path", `self.settings` -> "self.settings".
    Anything that is not a plain chain of names (a call, a subscript) gives "",
    which every caller reads as "unknown", never as a match.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _is_bound_method(call: ast.Call) -> bool:
    """True when `<expr>.name(...)` hangs off an OBJECT rather than a module.

    The distinction decides where the destination of a write lives. For a bound
    method - `path.mkdir()`, `staging.replace(dest)` - the path is the object
    before the dot. For a module function - `os.mkdir(path)`,
    `os.replace(staging, dest)` - the path is an argument and the name before the
    dot is just the module. Both spellings share five names, so resolving this
    by "is the callee an attribute" made every `os.*` write report the module
    `os` as its own destination (round 7, MEDIUM-3r's second half).

    An unknown owner (a call, a subscript, an unrecognised name) is treated as an
    object, which is the conservative direction: it keeps reporting a target
    rather than silently dropping the write from the table.
    """
    if not isinstance(call.func, ast.Attribute):
        return False
    return _dotted_name(call.func.value) not in FUNCTION_OWNER_MODULES


def _open_takes_path_first(call: ast.Call) -> bool:
    """True when this `open`-shaped call's FIRST argument is the path.

    Resolved from the call SHAPE, not from the node type. `open(...)`,
    `io.open(...)`, `codecs.open(...)` and `gzip.open(...)` are all path-first
    even though only the first is a bare name; only a bound `<expr>.open(...)`
    is mode-first, because there the path is the object the method hangs off.
    Deciding this by "is the callee an attribute" made `io.open(path, "w")` read
    its mode out of the path expression and score as a read (round 7, MEDIUM-3r).
    """
    if not isinstance(call.func, ast.Attribute):
        return True  # a bare open(path, mode)
    return _dotted_name(call.func.value) in PATH_FIRST_OPEN_MODULES


def _opens_for_write(call: ast.Call) -> bool:
    """True when this open()/Path.open()/io.open() call uses a writing mode.

    The mode is read from the positional argument OR the `mode=` keyword. Both
    matter: `path.open("w", encoding="utf-8")` has the mode positionally while
    `open(path, mode="w")` has it by keyword, and a detector that counted
    positional arguments instead of resolving the mode was blind to the first of
    those - the idiomatic shape used throughout this codebase (review round 6,
    HIGH-3).
    """
    mode: str | None = None
    # open(path, mode) has the mode second; Path.open(mode) has it first,
    # because the path is the bound object rather than an argument.
    mode_index = 1 if _open_takes_path_first(call) else 0
    if len(call.args) > mode_index and isinstance(call.args[mode_index], ast.Constant):
        mode = str(call.args[mode_index].value)
    for keyword in call.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = str(keyword.value.value)
    if mode is None:
        # open() with no mode is a read; Path.open() likewise defaults to "r".
        return False
    return bool(WRITE_MODE.search(mode))


def write_targets(call: ast.Call) -> list[ast.expr]:
    """Every expression this call writes TO, or [] when it is not a write.

    The single table of "what counts as a write" in this repository. It exists as
    a public function because the AC-M14-28 registry test needs exactly the same
    answer, and the previous separate, weaker copy of this question in that test
    could not see `path.open("w")`, `shutil.copy2(...)` or `Path.write_text(...)`
    (review round 6, HIGH-3). One table, two callers, no drift.

    Returns expressions rather than a bool so a caller can ask a further question
    about the destination - is it install-derived (W1-S1), is it a bare relative
    literal (W1-S4)?
    """
    name = _callee_name(call.func)
    if _is_bound_method(call) and name in WRITE_METHODS:
        if name in ("replace", "rename") and len(call.args) != 1:
            # `Path.replace(target)` takes exactly one argument; `str.replace`
            # takes two or three and is not a filesystem call at all. Telling
            # them apart by arity keeps `"logs/x".replace("a", "b")` - an
            # ordinary string operation on a relative-looking literal - from
            # being reported as a write into the install tree.
            return []
        return [call.func.value]
    if name == "open":
        if not _opens_for_write(call):
            return []
        # The destination follows the same call shape as the mode: path-first
        # for open/io.open/codecs.open, the bound object for `<expr>.open`.
        if _open_takes_path_first(call):
            return [call.args[0]] if call.args else []
        return [call.func.value]
    if name in WRITE_FUNCTIONS:
        return [call.args[i] for i in WRITE_FUNCTIONS[name] if i < len(call.args)]
    return []


def _is_unanchored_text(text: str) -> bool:
    """True when this literal string names a path with no absolute anchor.

    A leading slash, backslash or tilde, or a drive letter followed by a colon,
    anchors the path somewhere the working directory cannot change. Anything
    else resolves against the process working directory, which for LOCITIZE is the
    install tree.
    """
    if not text:
        return False
    if text.startswith(("/", "\\", "~")):
        return False
    return not (len(text) > 1 and text[1] == ":")


def _is_relative_literal(node: ast.AST, relative_names: set[str] | None = None) -> bool:
    """True for a path expression built from bare relative string literals.

    Recognised shapes, all of which the rule can PROVE are unanchored because
    their leading component is a literal with no absolute anchor:

      "crash.txt"                   a plain literal handed to open()/os.remove()
      Path("crash.txt")             the same, wrapped
      Path("logs") / "crash.txt"    a literal root joined with more literals
      os.path.join("logs", name)    the os.path spelling of the same join
        (also posixpath.join / ntpath.join)
      Path("logs").joinpath("c")    the method spelling of "/"
      "logs" + "/crash.txt"         string concatenation
      f"logs/{name}.txt"            an f-string whose LEADING part is a literal
      "logs/{}.txt".format(name)    .format() on a literal template
      "logs/%s.txt" % name          the % spelling of the same
      "a.txt" if flag else "b.txt"  a conditional where BOTH branches are relative
      str(...) / os.fspath(...)     any of the above, laundered through a wrapper
      Path("c.txt").resolve()       and .absolute() / .expanduser() /
                                    .with_suffix() / .with_name() / .with_stem():
                                    the receiver decides, because these name the
                                    same place - .resolve() in particular anchors
                                    it to the WORKING directory, i.e. the install
                                    tree
      (p := Path("c.txt"))          the walrus, both as the write target itself
                                    and as a binding of `p`
      a, b = "x.txt", "y.txt"       tuple unpacking, paired positionally against
                                    a literal sequence of the same length
      p, where p was bound to any of the above earlier in the same scope
        (by assignment, by a walrus, by tuple unpacking, by a `for` target, or by
        a comprehension target)

    THE MEASURED BOUNDARY - what this rule does NOT see. Each line was measured
    against the shipped code, not reasoned about, and each is pinned as a
    must-stay-silent case in tests/test_write_fence_rules.py:

      * a path arriving as a function argument (`open(p, "w")`), or read from ANY
        attribute or subscript (`self.path.open("w")`, `PATHS["log"]`,
        `Path("logs").parent`) - nothing at the write site says where it points.
        Attribute access is W1-S1's business, not W1-S4's;
      * the return value of any call other than the wrappers listed above -
        including a helper in this same module that returns a bare literal;
      * a name bound by unpacking a non-literal sequence (`a, b = _paths()`), by
        unpacking with a starred target (`a, *rest = ...`), or by an augmented
        assignment from an untracked name (`p = root` then `p += "/c.txt"`);
      * a name bound in a SIBLING scope: bindings are collected per scope (see
        _scope_body), so `p = Path("c.txt")` in one function says nothing about a
        `p` written in another;
      * an f-string or join whose LEADING component is not a literal
        (`f"{root}/c.txt"`, `os.path.join(root, "c.txt")`, `root + "/c.txt"`);
      * `.join()` on a non-literal separator, which is str.join, a different
        function that happens to share the name;
      * the empty literal `""`;
      * a write reached through `os.open` + `os.write`, through an alias bound to
        a write method, or performed by a child process - all beyond a static
        AST rule, and covered instead by the behavioural watched-tree half in
        tests/test_data_root_ownership.py.

    And the measured limits in the NOISY direction, disclosed because a boundary
    stated in only one direction is how this rule got four review findings:

      * name tracking is flow-INSENSITIVE, so a name first bound to a relative
        literal and later rebound to an argument (`p = Path("c.txt")` then
        `p = root`) still reports;
      * a binding is visible to every write NESTED inside the scope that made it,
        so a module-level `C = Path("c.txt")` reports a `C.write_text(...)` in any
        function in the file - which is what the module-constant case wants, and
        what a closure over an unrelated `p` would get too.

    Both err toward a false finding rather than a missed write, which is the
    survivable direction for a fence.

    The rule reports only what it can prove is unanchored, because a false
    finding in a fence nobody can silence is a fence people delete - but the
    disclosed limit is kept no narrower than the measured one (review round 7,
    MEDIUM-2r).
    """
    relative_names = relative_names or set()
    if isinstance(node, ast.Name):
        return node.id in relative_names
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and _is_unanchored_text(node.value)
    if isinstance(node, ast.JoinedStr):
        # An f-string is unanchored when its first piece is a literal that is.
        # `f"logs/{name}.txt"` is provably relative whatever `name` holds; a
        # leading `{root}` is not, and is left alone.
        first = node.values[0] if node.values else None
        return isinstance(first, ast.Constant) and _is_unanchored_text(
            str(first.value)
        )
    if isinstance(node, ast.Call):
        name = _callee_name(node.func)
        if name in ("Path", "PurePath") or name in PATH_PASSTHROUGH:
            # Path(x) / str(x) / os.fspath(x): the argument decides.
            return bool(node.args) and _is_relative_literal(node.args[0], relative_names)
        if name == "join" and isinstance(node.func, ast.Attribute):
            owner = _dotted_name(node.func.value)
            if owner in PATH_JOIN_PREFIXES:
                # os.path.join(a, b): the FIRST component decides, exactly as
                # for "/" - a later drive-anchored component would win, but
                # joining a relative root onto an absolute tail is not a shape
                # anyone writes.
                return bool(node.args) and _is_relative_literal(
                    node.args[0], relative_names
                )
            return False
        if name == "format" and isinstance(node.func, ast.Attribute):
            # "logs/{}.txt".format(x): the template is the anchor question.
            return _is_relative_literal(node.func.value, relative_names)
        if name == "joinpath" and isinstance(node.func, ast.Attribute):
            # Path("logs").joinpath("c.txt") is the method spelling of "/", and
            # the same reasoning applies: the receiver decides.
            return _is_relative_literal(node.func.value, relative_names)
        if name in PATH_PRESERVING_METHODS and isinstance(node.func, ast.Attribute):
            # .resolve() / .absolute() / .expanduser() / .with_suffix(): the
            # result names the same place the receiver did, anchored at the
            # working directory - which is the install tree.
            return _is_relative_literal(node.func.value, relative_names)
        return False
    if isinstance(node, ast.NamedExpr):
        # `(p := Path("c.txt")).write_text(...)`: the walrus writes to whatever
        # its value is, and binds it too (see _local_relative_names).
        return _is_relative_literal(node.value, relative_names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add, ast.Mod)):
        # A "/" join, a "+" concatenation or a "%" format is relative exactly
        # when its leftmost element is. "%" is included because
        # "logs/%s.txt" % name is the same provable shape as .format() and was
        # the last formatting spelling still slipping through.
        return _is_relative_literal(node.left, relative_names)
    if isinstance(node, ast.IfExp):
        # `"a.txt" if flag else "b.txt"` writes to one of two paths and both
        # are unanchored, so the write lands in the install tree either way.
        # Requiring BOTH branches keeps a genuine absolute fallback silent.
        return _is_relative_literal(node.body, relative_names) and _is_relative_literal(
            node.orelse, relative_names
        )
    return False


def _scope_body(node: ast.AST) -> list[ast.AST]:
    """Every node in THIS scope, not descending into nested function bodies.

    W1-S4's name tracking has to be scope-accurate in a way W1-S1's does not.
    `.base_dir` is an attribute that means the same thing everywhere, but a bare
    local name does not: `data = Path(data_dir)` in one function and an unrelated
    `data_dir = "docs"` in another would, under a whole-module walk, make the
    first look like a relative write. That false positive is exactly the kind
    that gets a fence switched off, so scopes are kept apart.
    """
    scoped: list[ast.AST] = []
    stack: list[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        current = stack.pop()
        scoped.append(current)
        if isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(ast.iter_child_nodes(current))
    return scoped


def _unpacked_relative_names(node: ast.Assign, names: set[str]) -> set[str]:
    """Names bound to a relative literal by TUPLE UNPACKING, if any.

    Handles the one shape a static rule can honestly resolve: a tuple/list target
    paired element-for-element with a tuple/list value of the same length, with
    no starred target to shift the positions. `a, b = _paths()` is not resolvable
    and is left alone, and that limit is disclosed in _is_relative_literal.
    """
    found: set[str] = set()
    for target in node.targets:
        if not isinstance(target, (ast.Tuple, ast.List)):
            continue
        if not isinstance(node.value, (ast.Tuple, ast.List)):
            continue
        if len(target.elts) != len(node.value.elts):
            continue
        if any(isinstance(element, ast.Starred) for element in target.elts):
            # `a, *rest = ...` breaks the positional pairing this relies on.
            continue
        # strict=True is safe: the length check above already refused any pair
        # whose sides differ, and a silent truncation would drop a tracked name.
        for slot, value in zip(target.elts, node.value.elts, strict=True):
            if isinstance(slot, ast.Name) and _is_relative_literal(value, names):
                found.add(slot.id)
    return found


def _local_relative_names(func: ast.AST) -> set[str]:
    """Names inside one scope that hold a bare relative path.

    The mirror of _local_taint, for W1-S4. It exists because the idiomatic shape
    is two statements - `p = Path("crash.txt")` then `p.open("w")` - and a rule
    that only looked at the write's own expression would score that clean.
    """
    names: set[str] = set()
    scoped = _scope_body(func)
    for _ in range(2):
        for node in scoped:
            if isinstance(node, ast.Assign):
                if _is_relative_literal(node.value, names):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
                else:
                    # `a, b = "x.txt", "y.txt"` - an assignment whose VALUE is
                    # not one path but a sequence of them. Paired positionally
                    # and only for a literal sequence of the same length, so
                    # unpacking a function result (which says nothing about
                    # where its elements point) stays silent (round 8, MEDIUM).
                    names.update(_unpacked_relative_names(node, names))
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                # The walrus: `if (p := Path("c.txt")).exists(): p.write_text(x)`
                # binds a name exactly as `=` does, and was neither tracked nor
                # excluded for four review rounds (round 8, MEDIUM).
                if _is_relative_literal(node.value, names):
                    names.add(node.target.id)
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and _is_relative_literal(node.value, names)
            ):
                names.add(node.target.id)
            elif isinstance(node, (ast.For, ast.comprehension)):
                # `for p in [Path("c.txt")]: p.write_text(...)` and its
                # comprehension spelling. Only a literal sequence of literal
                # paths counts - iterating anything else says nothing about
                # where its elements point (round 7, MEDIUM-2r).
                if (
                    isinstance(node.target, ast.Name)
                    and isinstance(node.iter, (ast.List, ast.Tuple, ast.Set))
                    and node.iter.elts
                    and all(
                        _is_relative_literal(element, names)
                        for element in node.iter.elts
                    )
                ):
                    names.add(node.target.id)
    return names


def _install_returning_functions(tree: ast.AST) -> set[str]:
    """Functions in this module whose return value derives from the install dir.

    These are what make the scan cross function boundaries: a resolver that
    hands back <install>/logs is the first half of the shipped defect, and the
    handler that writes to it is the second half, in a different function.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = _local_taint(node, set())
        for statement in ast.walk(node):
            if isinstance(statement, ast.Return) and statement.value is not None:
                if _mentions_install(statement.value, local, set()):
                    found.add(node.name)
                    break
    return found


def _local_taint(func: ast.AST, tainted_calls: set[str]) -> set[str]:
    """Names inside one function that hold an install-derived path.

    Two passes, because a function may assign from a resolver defined later in
    the file; a fixed point is reached quickly since assignments are finite.
    """
    tainted: set[str] = set()
    for _ in range(2):
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and _mentions_install(
                node.value, tainted, tainted_calls
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        tainted.add(target.id)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                if _mentions_install(node.value, tainted, tainted_calls) and isinstance(
                    node.target, ast.Name
                ):
                    tainted.add(node.target.id)
    return tainted


def scan_module(path: Path, tainted_calls: set[str]) -> list[Finding]:
    """Apply the four rules to one module on disk."""
    return scan_source(path.read_text(encoding="utf-8"), path.name, tainted_calls)


def scan_source(source: str, module: str, tainted_calls: set[str] | None = None) -> list[Finding]:
    """Apply the four rules to one module's SOURCE TEXT.

    Split out from scan_module so the rules can be tested against short synthetic
    modules that contain exactly the shape under test. Without this, the only way
    to prove a rule can fire was to edit a real product file - which is why the
    fence's blind spots were found by a reviewer rather than by its own tests.
    """
    tainted_calls = set(tainted_calls or ())
    tree = ast.parse(source, filename=module)
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            continue
        local = _local_taint(node, tainted_calls)
        relative_names = _local_relative_names(node)

        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            name = _callee_name(inner.func)

            # W1-S4 first, because it is the rule that needs no taint: a write to
            # a bare relative path lands wherever the process happens to be, and
            # locitize.bat cd's into the install directory. Checked before the
            # taint-based rules because those `continue` past the rest of the
            # loop body for the shapes they recognise.
            for target in write_targets(inner):
                if _is_relative_literal(target, relative_names):
                    findings.append(
                        Finding(
                            "W1-S4",
                            module,
                            inner.lineno,
                            f"{name}() writes to the relative path "
                            f"{ast.unparse(target)}; locitize's working directory "
                            f"is the install directory, so this lands in the "
                            f"install tree - anchor it to settings.data_dir",
                        )
                    )
                    break

            # W1-S1 / W1-S2: a write whose target derives from the install dir.
            if isinstance(inner.func, ast.Attribute) and name in WRITE_METHODS:
                if _mentions_install(inner.func.value, local, tainted_calls):
                    findings.append(
                        Finding(
                            "W1-S1",
                            module,
                            inner.lineno,
                            f"{name}() on a path derived from the install directory",
                        )
                    )
                    continue
            if name == "open":
                # Asked of the shared table rather than re-derived here, so the
                # module-qualified shapes (io.open, codecs.open) resolve their
                # path the same way for W1-S1 as they do for W1-S4 and for the
                # AC-M14-28 registry test (round 7, MEDIUM-3r).
                opened = write_targets(inner)
                if any(
                    _mentions_install(target, local, tainted_calls)
                    for target in opened
                ):
                    findings.append(
                        Finding(
                            "W1-S1",
                            module,
                            inner.lineno,
                            "open() for writing on a path derived from the install directory",
                        )
                    )
                    continue
            if name in WRITE_FUNCTIONS:
                for index in WRITE_FUNCTIONS[name]:
                    if index < len(inner.args) and _mentions_install(
                        inner.args[index], local, tainted_calls
                    ):
                        findings.append(
                            Finding(
                                "W1-S1",
                                module,
                                inner.lineno,
                                f"{name}() writes to a path derived from the install directory",
                            )
                        )
                        break
                continue

            # W1-S3: seeding a directory resolver with the install directory.
            if name and RESOLVER_NAME.search(name):
                for argument in list(inner.args) + [k.value for k in inner.keywords]:
                    if any(_is_install_attr(child) for child in ast.walk(argument)):
                        findings.append(
                            Finding(
                                "W1-S3",
                                module,
                                inner.lineno,
                                f"{name}() is given the install directory as its root; "
                                f"user data resolvers take settings.data_dir",
                            )
                        )
                        break
    # A call inside a function is visited twice - once through its own scope and
    # once through the enclosing Module node - so the same site would otherwise
    # be reported twice. One site reported twice is noise, not two problems.
    return sorted(set(findings), key=lambda f: (f.module, f.line, f.rule))


def scan(root: Path = PLATFORM_DIR) -> list[Finding]:
    """Scan every runtime module and return the findings, sorted for stability."""
    modules = runtime_modules(root)

    # Pass one: which function names hand back an install-derived path anywhere
    # in the product. Names are global here because the platform modules are
    # flat and import each other by bare name (from logger import resolve_log_dir).
    tainted_calls: set[str] = set()
    trees: dict[Path, ast.AST] = {}
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        trees[path] = tree
        tainted_calls |= _install_returning_functions(tree)

    findings: set[Finding] = set()
    for path in modules:
        # A set, because a call inside a nested function is visited once through
        # its own scope and once through the enclosing one; the same site
        # reported twice is noise, not two problems.
        findings.update(scan_module(path, tainted_calls))
    return sorted(findings, key=lambda f: (f.module, f.line, f.rule))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify_write_fence", description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args(argv)

    findings = scan()
    if args.json:
        print(
            json.dumps(
                {
                    "ok": not findings,
                    "findings": [
                        {
                            "rule": f.rule,
                            "module": f.module,
                            "line": f.line,
                            "detail": f.detail,
                        }
                        for f in findings
                    ],
                }
            )
        )
    elif findings:
        for finding in findings:
            print(finding.render())
        print(
            f"\nFAIL: {len(findings)} runtime write(s) could land in the install "
            f"directory (invariant W1, DEC-M14-9).\n"
            f"User data belongs under settings.data_dir. The install tree holds "
            f"the program only and is read-only at runtime."
        )
    else:
        print(
            f"OK: {len(runtime_modules())} runtime modules scanned; no writable "
            f"path is composed from the install directory (invariant W1)."
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
