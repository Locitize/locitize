"""Ship-set guard: what git tracks must be a complete, runnable product.

Where this fits (Architecture.md M14, Review H-2): Milestone 14 exists to turn
LOCITIZE into something a stranger can install. The release payload is built from
what git tracks - a clean clone, a CI checkout, `git archive`. So a source file
that was written but never `git add`ed is invisible during development (it sits
right there on the developer's disk and every import works) and fatal on
delivery.

That is not hypothetical. Review found `models.py` importing `finetune`,
`launcher.py` importing `desktop` and `gui_controller.py` importing
`secure_proxy`, with none of those three modules in the index: a release built
from git died at import with `ModuleNotFoundError: No module named 'finetune'`.
No gitignore rule was responsible - they had simply never been added.

This script is the standing guard against that returning. It builds a tree
containing ONLY tracked files, then proves the product imports and runs from it.
A one-time fix would have been fixed once; this fails the suite the next time it
happens, on the day it happens.

Running the tree is not enough on its own (Review H-4). Two of those three
modules are imported lazily, inside the function that needs them, so no import
probe ever executes those lines and their absence would go unreported. So the
tracked source is also READ: every tracked .py is parsed, and any local module it
names anywhere - module scope, inside a function, inside an `if` - must be in the
ship set. Reading catches what running cannot.

Why tracked content is copied from the working tree rather than extracted from a
commit: the gap must be catchable BEFORE a commit exists, otherwise the check
only ever reports a mistake that has already been published. Staging a file is
enough to satisfy this guard; forgetting to stage it is exactly the failure it
reports.

What this guard proves, stated honestly (Review M-8). It proves three things:

1. every local module that tracked Python code IMPORTS - by an import
   statement, or by importlib.import_module("x") / __import__("x") with a
   literal name - is itself tracked;
2. every non-Python file named in SHIP_CRITICAL_ASSETS below is tracked;
3. the tracked-only tree imports and its entry point runs.

What it does NOT prove, and deliberately does not claim to: that every data file
the product opens at runtime is tracked. Only the assets declared in the
manifest are checked, because a file opened through a computed path cannot be
resolved by reading the source. A module named only by a computed string, or
referenced only from a .bat wrapper, is invisible to this guard for the same
reason. The manifest exists so the assets that would actually break the product
are covered by a list a human maintains, rather than by a claim nobody can back.

Each of those three gaps has a test of its own, which is a stranger thing than it
sounds and worth explaining (Review M-10). A test that proves the guard says
NOTHING looks pointless until the failure mode is named: a documented gap with no
test behind it drifts, and a later reader takes the coverage as broader than it is.
So tests/test_ship_set.py pins all three - a computed import name, a module named
only from a .bat wrapper, and an undeclared data asset - and each one is paired
with a case proving the same fixture DOES fail for a dependency this guard can see,
so silence about the gap cannot be confused with a fixture that never fails.

Usage:
    python scripts/verify_ship_set.py            # human-readable report
    python scripts/verify_ship_set.py --json     # machine-readable result

Exit code 0 means the tracked tree is complete and runnable; 1 means it is not;
2 means the check itself could not run (git missing, not a repository).
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# The platform directory (the parent of scripts/), the root of the ship set.
PLATFORM_DIR = Path(__file__).resolve().parent.parent

# Directories in the materialised tree whose modules are not product runtime and
# are therefore not import-probed: tests are exercised by pytest itself, and
# scripts/ are standalone tools run with an explicit path.
NON_RUNTIME_DIRS = frozenset({"tests", "scripts"})

# Non-Python files the product reads by name at runtime, so a release without
# them is broken even though every import still resolves (Review M-8). The AST
# walk below cannot find these - they are opened through a path built at runtime,
# not imported - so they are declared here and checked against the tracked set.
#
# The rule for adding one: if a clean clone missing this file would misbehave,
# it belongs here. Each entry is a path relative to the platform directory, and
# each is verified to exist on disk as well as in git, so a stale entry after a
# rename is reported rather than silently satisfied.
SHIP_CRITICAL_ASSETS = (
    # The fine-tuning warning shown in the GUI (finetune.py reads it by name).
    "finetune_warning.txt",
    # The configuration templates a first run copies into the data root.
    "settings.default.yaml",
    "models.default.yaml",
    # The vendored Kokoro architecture config, so speech works with no download.
    "assets/kokoro/config.json",
    # The dependency list every install path reads.
    "requirements.txt",
)

# How long the child process gets to import every module or print --help. These
# are pure-import operations that take well under a second in practice; the
# ceiling only exists so a hung import fails loudly instead of blocking the suite.
SUBPROCESS_TIMEOUT_S = 180


@dataclass
class ShipSetResult:
    """The outcome of one ship-set check, in a form both humans and JSON accept."""

    tracked_count: int = 0
    modules: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the tracked-only tree imported and ran without complaint.

        Warnings deliberately do not fail the check - see `materialise` for the
        one thing that is reported without failing and why.
        """
        return not self.problems


def tracked_paths(root: Path = PLATFORM_DIR) -> list[str]:
    """Return every path git TRACKS under `root`, relative and POSIX-style.

    Deliberately different from the hygiene scanner's `ship_set()`, which also
    unions in untracked-but-not-ignored files so that brand-new work is checked
    before anyone remembers to add it. Here the untracked files are the very thing
    being tested for, so including them would defeat the check.

    Also different in what it reads. That scanner takes content from git's object
    store as well as from the disk, because its question is what a clone receives
    (Review H-6). This guard deliberately copies content from the working tree, so
    the gap it looks for is catchable BEFORE a commit exists - see the module
    docstring. Two questions, two sources, and neither is the other's bug.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed in {root}: {result.stderr.strip() or 'unknown error'}"
        )
    return [name for name in result.stdout.split("\0") if name]


def materialise(root: Path, destination: Path) -> tuple[int, list[str]]:
    """Copy the tracked files (and only those) from `root` into `destination`.

    Returns the number of files copied and the names of tracked paths that are
    missing from the working tree.

    A missing tracked path is reported but does NOT fail the check, and the
    distinction matters. It means the file was deleted on disk without the
    deletion being staged - so a real clean clone still receives it from the
    commit, and no release can break because of it. What it does signal is that
    the index and the disk disagree, which is worth a human's attention. The
    failure this script exists to catch is the opposite direction: a file on disk
    that git does not have, which a clone genuinely will not receive. That one is
    caught by the import probe, where it belongs.
    """
    copied = 0
    missing: list[str] = []
    for name in tracked_paths(root):
        source = root / name
        if not source.is_file():
            missing.append(name)
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied, missing


def _local_module_exists(root: Path, name: str) -> bool:
    """True when `name` names a module that lives in `root` itself.

    A top-level import is either of something installed (a standard-library or
    third-party package, which is not this script's business) or of a file
    sitting next to the importer. Only the second kind can go missing from a
    release because someone forgot to stage it, so only the second kind is
    checked. Both file layouts count: a plain module and a package directory.
    """
    return (root / f"{name}.py").is_file() or (root / name / "__init__.py").is_file()


def _local_module_location(root: Path, base: Path, name: str) -> str:
    """The path that actually resolved for `name`, for the failure message.

    Review L-7: the message used to be hardcoded to "<name>.py", so un-staging a
    package directory sent the reader looking for a file that does not exist.
    Report what was found instead - the module file, or the package's __init__.
    """
    if (root / f"{name}.py").is_file():
        return f"{(base / name).as_posix()}.py"
    return f"{(base / name / '__init__.py').as_posix()}"


def _dynamic_import_name(node: ast.Call) -> str | None:
    """The module a dynamic-import call names, when it names one literally.

    Covers the two ways Python code imports without an import statement:
    `importlib.import_module("x")` (also `import_module("x")` when imported
    directly) and `__import__("x")`. Only a literal string argument can be
    resolved by reading the source; a computed name is out of reach and is
    reported as a known gap in the module docstring rather than pretended away.
    """
    target = node.func
    called = target.attr if isinstance(target, ast.Attribute) else (
        target.id if isinstance(target, ast.Name) else None
    )
    if called not in {"import_module", "__import__"}:
        return None
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def imported_local_names(source: str) -> set[str]:
    """Every top-level module name a Python source file imports, anywhere in it.

    `ast.walk` rather than a scan of the module body on purpose: this exists to
    see imports that a runtime probe cannot, and those are exactly the ones
    hidden inside a function or an `if` branch. Relative imports are skipped -
    they resolve inside their own package and cannot name a top-level module.

    Dynamic imports with a literal name count too (Review M-8): to the release
    payload, `importlib.import_module("plugins")` is every bit as much a
    dependency as `import plugins`, and it is even more invisible because no
    import statement exists to read.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            dynamic = _dynamic_import_name(node)
            if dynamic:
                names.add(dynamic.split(".")[0])
    return names


def untracked_ship_assets(root: Path, tree: Path) -> list[str]:
    """Report declared ship-critical assets that the tracked tree does not have.

    Why a declared list rather than a derivation (Review M-8): the guard's import
    walk sees dependencies expressed as imports, and a data file the product
    opens by name is not one. `finetune_warning.txt` is the live example - the
    GUI reads it to show the fine-tuning warning, and un-staging it broke nothing
    the other probes could see. A short manifest a human maintains covers the
    files whose absence actually breaks the product.

    Two failures are reported. An asset on disk but missing from the tracked tree
    is the release bug this exists to catch. An asset missing from disk as well
    means the manifest itself has gone stale - a rename that left this list
    behind - which would otherwise turn the check into a no-op. That second
    report is made only when the tree being checked IS this project: the guard is
    also run against throwaway fixture trees, and those have no reason to contain
    LOCITIZE assets.
    """
    problems: list[str] = []
    is_this_project = root.resolve() == PLATFORM_DIR
    for name in SHIP_CRITICAL_ASSETS:
        if not (root / name).is_file():
            if is_this_project:
                problems.append(
                    f"ship-critical asset '{name}' is declared in "
                    "SHIP_CRITICAL_ASSETS but does not exist on disk - update the "
                    "manifest if the file was renamed or removed"
                )
            continue
        if not (tree / name).is_file():
            problems.append(
                f"ship-critical asset '{name}' exists on disk but is NOT tracked by "
                "git - a clean clone would not receive it, and the product reads it "
                "by name at runtime"
            )
    return problems


def unstaged_imported_modules(root: Path, tree: Path) -> list[str]:
    """Report local modules that tracked code imports but the ship set lacks.

    Why this exists as well as the import probe (Review H-4): the probe can only
    exercise imports that actually execute, and a lazy import - `import desktop`
    inside the function that opens the desktop shell - executes nowhere during a
    probe. `desktop` and `secure_proxy` are both imported that way, and both were
    among the modules that had never been staged, so the probe alone would have
    reported a clean ship set for two of the three files the defect was about.

    Reading the source instead of running it removes that blind spot: every
    tracked .py is parsed, every import it names anywhere is resolved against the
    modules on disk, and a name that resolves to a local module missing from the
    tracked tree is a release that will die the first time a user reaches that
    code path.
    """
    problems: list[str] = []
    for path in sorted(tree.rglob("*.py")):
        relative = path.relative_to(tree).as_posix()
        try:
            names = imported_local_names(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            problems.append(f"tracked file {relative} could not be parsed: {exc}")
            continue

        # Where a bare `import x` can resolve to a file of this project: the
        # platform root (how the product runs) and the importing file's own
        # directory (how pytest runs a test and how a script run by path
        # resolves its neighbours). Anything found in neither is installed
        # software, which git was never going to ship anyway.
        directory = path.parent.relative_to(tree)
        for name in sorted(names):
            for base in (Path("."), directory):
                if _local_module_exists(root / base, name) and not _local_module_exists(
                    tree / base, name
                ):
                    problems.append(
                        f"{relative} imports local module '{name}' "
                        f"({_local_module_location(root / base, base, name)}), which "
                        f"exists on disk but is NOT tracked by git - a clean clone "
                        f"would not receive it"
                    )
    return problems


def runtime_modules(tree: Path) -> list[str]:
    """List the importable product modules present in a materialised tree.

    Derived from the tree rather than hardcoded so a module added later is
    covered without anyone editing this file. What this derivation cannot see -
    a module that was never staged is simply absent from the list - is covered by
    `unstaged_imported_modules`, which reads the imports out of the tracked
    source rather than out of the tree.
    """
    names = []
    for path in sorted(tree.glob("*.py")):
        if path.stem == "__init__" or path.parent.name in NON_RUNTIME_DIRS:
            continue
        names.append(path.stem)
    return names


def _run_in_tree(tree: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a python command with `tree` as the working directory.

    `-E` and `-s` strip PYTHONPATH and the user site directory so the child
    cannot accidentally import the untracked originals sitting in the real
    platform directory - which would make this whole check pass vacuously.
    """
    return subprocess.run(
        [sys.executable, "-E", "-s", *args],
        cwd=str(tree),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_S,
    )


def check_ship_set(root: Path = PLATFORM_DIR) -> ShipSetResult:
    """Materialise a tracked-files-only tree and prove the product runs from it."""
    result = ShipSetResult()

    with tempfile.TemporaryDirectory(prefix="locitize-shipset-") as tmp:
        tree = Path(tmp) / "platform"
        tree.mkdir(parents=True)

        copied, missing = materialise(root, tree)
        result.tracked_count = copied
        for name in missing:
            result.warnings.append(
                f"WARNING: tracked but deleted on disk (deletion not staged): {name}"
            )

        # Probe 0: source-level completeness. Runs first because it is the only
        # check that can see a lazily imported module, and its message names the
        # missing file directly instead of leaving a human to read a traceback.
        result.problems.extend(unstaged_imported_modules(root, tree))

        # Probe 0b: the declared non-Python payload. An import walk cannot see a
        # data file the product opens by name, so the ones that matter are listed
        # and checked (Review M-8).
        result.problems.extend(untracked_ship_assets(root, tree))

        modules = runtime_modules(tree)
        result.modules = modules
        if not modules:
            result.problems.append(
                "no runtime modules found in the tracked tree - nothing would ship"
            )
            return result

        # Probe 1: does every product module import? This is the exact failure
        # Review reproduced, and it catches a missing module wherever it is
        # imported from.
        imported = _run_in_tree(tree, ["-c", "import " + ", ".join(modules)])
        if imported.returncode != 0:
            result.problems.append(
                "the tracked-only tree does not import:\n"
                + (imported.stderr.strip() or imported.stdout.strip())
            )

        # Probe 2: does it RUN? Importing proves the module graph is complete;
        # --help proves the entry point still reaches argument parsing, which is
        # a materially stronger claim and costs one more subprocess.
        launcher = tree / "launcher.py"
        if not launcher.is_file():
            result.problems.append("launcher.py is not tracked - there is no entry point")
        else:
            helped = _run_in_tree(tree, ["launcher.py", "--help"])
            if helped.returncode != 0:
                result.problems.append(
                    "`launcher.py --help` fails in the tracked-only tree:\n"
                    + (helped.stderr.strip() or helped.stdout.strip())
                )
            elif "LOCITIZE" not in helped.stdout:
                result.problems.append(
                    "`launcher.py --help` ran but printed no LOCITIZE usage text"
                )

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Fail if the git-tracked tree is not a complete, runnable product."
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the result as JSON on stdout"
    )
    args = parser.parse_args(argv)

    try:
        result = check_ship_set()
    except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"ship-set check could not run: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "tracked_files": result.tracked_count,
                    "modules": result.modules,
                    "problems": result.problems,
                    "warnings": result.warnings,
                },
                indent=2,
            )
        )
    else:
        for warning in result.warnings:
            print(warning)
        for problem in result.problems:
            print(problem)

    if not result.ok:
        print(
            f"\nFAIL: the tree built from `git ls-files` ({result.tracked_count} files) "
            "is not a runnable product.\n"
            "A release built from a clean clone would break the same way. Run "
            "`git status` and stage the source files that were never added.",
            file=sys.stderr,
        )
        return 1

    # With --json, stdout carries the JSON document alone so a caller can parse
    # it directly; the human summary goes to stderr instead (Review L-4).
    print(
        f"OK: {result.tracked_count} tracked files import and run "
        f"({len(result.modules)} runtime modules).",
        file=sys.stderr if args.json else sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
