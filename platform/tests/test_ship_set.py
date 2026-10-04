"""Ship-set guard tests (Review H-2, MT-2).

This wires scripts/verify_ship_set.py into the pytest suite. The defect it
guards against was invisible to every other check in the project: three modules
that the product imports at module scope had never been `git add`ed, so
everything worked on the developer's machine and a release built from a clean
clone died at `import finetune`.

Five things are proved here:

1. the real tree, built from `git ls-files` alone, imports and runs;
2. the guard actually fails when a module is missing - a check that always says
   OK is how the hole survived in the first place;
3. it fails for a LAZILY imported module too (Review H-4). Two of the three
   modules in the original defect are imported inside a function, so no import
   probe can reach them; only reading the tracked source finds them missing;
4. it fails for a module imported dynamically by a literal name, and it is
   pinned as NOT seeing a name computed at runtime, so the guard's claim stays
   honest rather than overstated (Review M-8);
5. it fails for a declared ship-critical data file that is not tracked - the
   finetune_warning.txt shape, which no import walk can see (Review M-8);
6. it stays SILENT about each of the three gaps its docstring names, and each of
   those silences is paired with a case proving the same fixture can fail
   (Review M-10). A documented gap with no test behind it drifts into being read
   as covered.

The negative control below builds its own throwaway git repository rather than
touching the real one, so a failing test can never leave the project's index in
a modified state.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PLATFORM_DIR / "scripts" / "verify_ship_set.py"

sys.path.insert(0, str(PLATFORM_DIR / "scripts"))

import verify_ship_set as guard  # noqa: E402


def test_the_git_tracked_tree_imports_and_runs():
    """A release built from what git tracks is a working product.

    This is the exact reproduction Review used, turned into a standing check:
    materialise a tree from `git ls-files` only, import every runtime module,
    then run the entry point. Failure here means a clean clone is broken.
    """
    result = guard.check_ship_set(PLATFORM_DIR)
    assert result.problems == [], "\n".join(result.problems)
    assert result.tracked_count > 0
    # Guard against a vacuous pass: an empty or near-empty module list would
    # satisfy "nothing failed to import" while proving nothing at all.
    assert len(result.modules) >= 20, result.modules


def test_ship_set_script_runs_clean_as_a_command():
    """The guard is usable standalone, as the release checklist will run it."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(PLATFORM_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK:" in result.stdout


def test_ship_set_guard_reports_a_module_that_was_never_staged(tmp_path):
    """Negative control: reproduce H-2 in miniature and require a failure.

    `launcher.py` imports `helper`, but only `launcher.py` is added to the index.
    That is precisely the shape of the real defect, and the guard must catch it.
    """
    root = tmp_path / "fake_platform"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True)

    (root / "launcher.py").write_text(
        "import helper\n"
        "if __name__ == '__main__':\n"
        "    print('locitize usage')\n",
        encoding="utf-8",
    )
    (root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    # Only the importer is staged. The imported module stays untracked - exactly
    # the state the real tree was in.
    subprocess.run(
        ["git", "add", "launcher.py"], cwd=str(root), check=True, capture_output=True
    )

    result = guard.check_ship_set(root)
    assert not result.ok
    assert any("does not import" in p for p in result.problems), result.problems
    assert any("No module named 'helper'" in p for p in result.problems), result.problems


def test_ship_set_guard_reports_a_lazily_imported_unstaged_module(tmp_path):
    """Review H-4: the shape the import probe structurally cannot see.

    `import shell` sits inside a function, so nothing imports it when the guard
    imports every module, and the missing file leaves no trace in the tree the
    probe list is derived from. Running the tree therefore reports OK while a
    clean clone crashes the first time a user opens that feature. Reading the
    source is what closes it, and `desktop` and `secure_proxy` - two of the three
    modules the original defect was about - are imported exactly this way.
    """
    root = tmp_path / "lazy_platform"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True)

    (root / "launcher.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser(prog='locitize')\n"
        "def open_shell():\n"
        "    import shell\n"
        "    return shell.VALUE\n"
        "if __name__ == '__main__':\n"
        "    parser.parse_args()\n",
        encoding="utf-8",
    )
    (root / "shell.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "launcher.py"], cwd=str(root), check=True, capture_output=True
    )

    result = guard.check_ship_set(root)
    assert not result.ok, "a lazily imported unstaged module must fail the guard"
    assert any("'shell'" in p and "NOT tracked" in p for p in result.problems), (
        result.problems
    )


def test_ship_set_guard_ignores_installed_packages(tmp_path):
    """The source check must only speak about files of this project.

    Every module imports third-party and standard-library names; none of them
    are git's job. Flagging those would make the guard noise, and a noisy guard
    gets switched off - the same fate an over-eager hygiene scanner would meet.
    """
    root = tmp_path / "thirdparty_platform"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True)

    (root / "launcher.py").write_text(
        "import argparse, json, sys\n"
        "def later():\n"
        "    import subprocess\n"
        "    return subprocess\n"
        "parser = argparse.ArgumentParser(prog='locitize')\n"
        "if __name__ == '__main__':\n"
        "    parser.parse_args()\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "launcher.py"], cwd=str(root), check=True, capture_output=True
    )

    result = guard.check_ship_set(root)
    assert result.ok, result.problems


def test_imported_local_names_sees_every_import_position():
    """Unit-level proof of the rule the guard depends on.

    Module scope, function scope, inside a conditional, and `from x import y`
    all count; a relative import does not, because it names something inside its
    own package rather than a top-level module.
    """
    names = guard.imported_local_names(
        "import alpha\n"
        "from beta import thing\n"
        "from . import sibling\n"
        "def f():\n"
        "    import gamma.sub\n"
        "if True:\n"
        "    from delta.deep import x\n"
    )
    assert names == {"alpha", "beta", "gamma", "delta"}


def test_ship_set_guard_passes_when_every_import_is_staged(tmp_path):
    """The same fixture, fully staged, must pass - so the failure above is real.

    Without this pair, the negative control could be passing for an unrelated
    reason (a broken fixture, a git error) and still look like proof.
    """
    root = tmp_path / "fixed_platform"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True)

    (root / "launcher.py").write_text(
        "import helper\n"
        "import argparse\n"
        "parser = argparse.ArgumentParser(prog='locitize')\n"
        "if __name__ == '__main__':\n"
        "    parser.parse_args()\n",
        encoding="utf-8",
    )
    (root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "launcher.py", "helper.py"],
        cwd=str(root), check=True, capture_output=True,
    )

    result = guard.check_ship_set(root)
    assert result.ok, result.problems


def test_imported_local_names_sees_dynamic_imports_with_a_literal_name():
    """Review M-8: an import does not have to be an import statement.

    `importlib.import_module("x")` makes `x` part of the release payload just as
    surely as `import x`, and it is harder to notice because no import statement
    exists to read. Both spellings of the dynamic form are covered.
    """
    names = guard.imported_local_names(
        "import importlib\n"
        "def load():\n"
        "    a = importlib.import_module('alpha')\n"
        "    b = __import__('beta')\n"
        "    return a, b\n"
        "from importlib import import_module\n"
        "def load2():\n"
        "    return import_module('gamma.sub')\n"
    )
    assert {"alpha", "beta", "gamma"} <= names


def test_a_computed_import_name_is_a_known_gap_not_a_silent_claim():
    """The honest boundary of the check above (Review M-8).

    A module name assembled at runtime cannot be resolved by reading the source,
    so the guard does not see it. That gap is stated in the module docstring; it
    is pinned here so nobody later reads the dynamic-import support as covering
    more than it does.
    """
    names = guard.imported_local_names(
        "import importlib\n"
        "def load(part):\n"
        "    return importlib.import_module('plug' + part)\n"
    )
    assert names == {"importlib"}


def test_ship_set_guard_reports_a_dynamically_imported_unstaged_module(tmp_path):
    """The dynamic form of the H-2 defect must fail the guard too."""
    root = tmp_path / "dynamic_platform"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True)

    (root / "launcher.py").write_text(
        "import argparse\n"
        "import importlib\n"
        "parser = argparse.ArgumentParser(prog='locitize')\n"
        "def load_plugin():\n"
        "    return importlib.import_module('plugin')\n"
        "if __name__ == '__main__':\n"
        "    parser.parse_args()\n",
        encoding="utf-8",
    )
    (root / "plugin.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "launcher.py"], cwd=str(root), check=True, capture_output=True
    )

    result = guard.check_ship_set(root)
    assert not result.ok, "a dynamically imported unstaged module must fail the guard"
    assert any("'plugin'" in p and "NOT tracked" in p for p in result.problems), (
        result.problems
    )


def test_ship_set_guard_names_the_path_that_actually_resolved(tmp_path):
    """Review L-7: the message must point at a file that exists.

    An unstaged PACKAGE resolves through its __init__.py, and the failure message
    used to say "<name>.py" regardless, sending the reader to a path that is not
    there.
    """
    root = tmp_path / "package_platform"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True)

    (root / "launcher.py").write_text(
        "import argparse\n"
        "import plugins\n"
        "parser = argparse.ArgumentParser(prog='locitize')\n"
        "if __name__ == '__main__':\n"
        "    parser.parse_args()\n",
        encoding="utf-8",
    )
    (root / "plugins").mkdir()
    (root / "plugins" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "launcher.py"], cwd=str(root), check=True, capture_output=True
    )

    result = guard.check_ship_set(root)
    assert not result.ok
    message = "\n".join(result.problems)
    assert "plugins/__init__.py" in message, message
    assert "plugins.py" not in message, message


def test_ship_set_guard_reports_an_untracked_ship_critical_asset(tmp_path, monkeypatch):
    """Review M-8: a data file the product reads by name must be tracked.

    The live example is finetune_warning.txt - the GUI reads it to show the
    fine-tuning warning, and un-staging it broke nothing that the import walk or
    the run probe could see. The manifest closes that, and this proves the
    manifest is enforced rather than decorative.
    """
    root = tmp_path / "asset_platform"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True)

    (root / "launcher.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser(prog='locitize')\n"
        "if __name__ == '__main__':\n"
        "    parser.parse_args()\n",
        encoding="utf-8",
    )
    (root / "warning.txt").write_text("read at runtime\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "launcher.py"], cwd=str(root), check=True, capture_output=True
    )
    monkeypatch.setattr(guard, "SHIP_CRITICAL_ASSETS", ("warning.txt",))

    result = guard.check_ship_set(root)
    assert not result.ok, "an untracked ship-critical asset must fail the guard"
    assert any("warning.txt" in p and "NOT tracked" in p for p in result.problems), (
        result.problems
    )

    # And the positive half: staged, the same tree passes. Without this the
    # failure above could be coming from anywhere.
    subprocess.run(
        ["git", "add", "warning.txt"], cwd=str(root), check=True, capture_output=True
    )
    assert guard.check_ship_set(root).ok


def test_every_declared_ship_critical_asset_is_tracked_and_present():
    """The manifest must be true of the real tree, and must not be empty.

    An empty or stale manifest would satisfy the check above while proving
    nothing about this project, so both are asserted here. finetune_warning.txt
    is named explicitly because it is the file the Review found exposed.
    """
    assert guard.SHIP_CRITICAL_ASSETS, "the ship-critical asset manifest is empty"
    assert "finetune_warning.txt" in guard.SHIP_CRITICAL_ASSETS
    tracked = set(guard.tracked_paths(PLATFORM_DIR))
    for name in guard.SHIP_CRITICAL_ASSETS:
        assert (PLATFORM_DIR / name).is_file(), f"{name} is declared but not on disk"
        assert name in tracked, f"{name} is declared ship-critical but not tracked"


def test_unstaged_deletion_is_a_warning_not_a_failure(tmp_path):
    """A file deleted on disk but still in the index must not fail the guard.

    A clean clone receives that file from the commit, so no release can break
    because of it. It is still reported, because the index and the disk
    disagreeing is worth a human's attention.
    """
    root = tmp_path / "deleted_platform"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True)

    (root / "launcher.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser(prog='locitize')\n"
        "if __name__ == '__main__':\n"
        "    parser.parse_args()\n",
        encoding="utf-8",
    )
    (root / "old_wrapper.bat").write_text("@echo off\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "launcher.py", "old_wrapper.bat"],
        cwd=str(root), check=True, capture_output=True,
    )
    (root / "old_wrapper.bat").unlink()

    result = guard.check_ship_set(root)
    assert result.ok, result.problems
    assert any("old_wrapper.bat" in w for w in result.warnings), result.warnings


# ---------------------------------------------------------------------------
# The guard's stated GAPS, each pinned by a test (Review M-10).
#
# The module docstring names three things this guard does not see. Round 5 found
# that only one of the three had a test behind it, which is the second consecutive
# round in which a coverage claim was broader than its tests. So each named gap now
# has a case that proves the guard is silent about it.
#
# A test that asserts a gap looks strange - it passes when nothing is found - so
# what each one is FOR is worth stating: it fails the day somebody closes the gap
# without updating the docstring, and it stops a reader from taking the guard's
# coverage as broader than it is. Every one of them is written so it cannot pass
# vacuously: the same fixture is checked to fail for a dependency the guard DOES
# see, so a case that reports nothing at all cannot be mistaken for a case that
# correctly reports nothing.
# ---------------------------------------------------------------------------

_RUNNABLE_LAUNCHER = (
    "import argparse\n"
    "parser = argparse.ArgumentParser(prog='locitize')\n"
    "if __name__ == '__main__':\n"
    "    parser.parse_args()\n"
)


def _tracked_repo(tmp_path: Path, name: str) -> Path:
    """A throwaway repository with a tracked, runnable launcher and nothing else."""
    root = tmp_path / name
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True)
    (root / "launcher.py").write_text(_RUNNABLE_LAUNCHER, encoding="utf-8")
    subprocess.run(
        ["git", "add", "launcher.py"], cwd=str(root), check=True, capture_output=True
    )
    return root


def test_a_module_named_only_from_a_bat_wrapper_is_a_known_gap(tmp_path):
    """Gap 2 of 3: a .bat wrapper is not Python, so no import walk can read it.

    The real shape is one of the LOCITIZE *.bat launchers naming a module that was
    never staged. The guard reads Python; a batch file calling `python helper.py`
    expresses the same dependency in a language nothing here parses, so the module
    can be missing from the ship set and the guard stays silent.

    Proved non-vacuous below: the identical fixture DOES fail once the same module
    is named by an import statement instead, so the silence is about the .bat form
    and not about a fixture that cannot fail.
    """
    root = _tracked_repo(tmp_path, "bat_reference")
    (root / "locitize Helper.bat").write_text(
        "@echo off\r\npython helper.py %*\r\n", encoding="utf-8"
    )
    (root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "locitize Helper.bat"], cwd=str(root), check=True, capture_output=True
    )

    result = guard.check_ship_set(root)
    assert result.ok, result.problems
    assert not any("helper" in problem for problem in result.problems), result.problems

    # Non-vacuity: the same untracked module, named by an import, IS reported.
    (root / "launcher.py").write_text(
        "import helper\n" + _RUNNABLE_LAUNCHER, encoding="utf-8"
    )
    subprocess.run(
        ["git", "add", "launcher.py"], cwd=str(root), check=True, capture_output=True
    )
    seen = guard.check_ship_set(root)
    assert not seen.ok, "the fixture cannot fail, so the gap test proves nothing"
    assert any("helper" in problem for problem in seen.problems), seen.problems


def test_an_undeclared_data_asset_is_a_known_gap(tmp_path):
    """Gap 3 of 3: a data file opened by name and not in the manifest.

    `open("notes.txt")` is a dependency of the running product exactly as much as
    an import is, and the guard cannot see it: reading source finds imports, and
    the manifest covers only what a human wrote down. This is the finetune_warning
    shape, one step earlier - before anybody added it to SHIP_CRITICAL_ASSETS.

    Proved non-vacuous below: declaring the same file in the manifest makes the
    guard report it immediately, so the silence is about the manifest's boundary and
    not about a fixture that cannot fail.
    """
    root = _tracked_repo(tmp_path, "undeclared_asset")
    (root / "launcher.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser(prog='locitize')\n"
        "def load_notes():\n"
        "    with open('notes.txt', encoding='utf-8') as handle:\n"
        "        return handle.read()\n"
        "if __name__ == '__main__':\n"
        "    parser.parse_args()\n",
        encoding="utf-8",
    )
    (root / "notes.txt").write_text("shipped text\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "launcher.py"], cwd=str(root), check=True, capture_output=True
    )

    result = guard.check_ship_set(root)
    assert result.ok, result.problems
    assert not any("notes.txt" in problem for problem in result.problems), result.problems


def test_declaring_an_asset_is_what_makes_it_visible(monkeypatch, tmp_path):
    """The other half of the gap above: the manifest is the whole mechanism.

    Same fixture, same untracked data file, one difference - it is declared. The
    guard then reports it. This is what makes the gap test above a statement about
    the manifest's boundary rather than about a fixture that cannot fail.
    """
    root = _tracked_repo(tmp_path, "declared_asset")
    (root / "notes.txt").write_text("shipped text\n", encoding="utf-8")
    monkeypatch.setattr(guard, "SHIP_CRITICAL_ASSETS", ("notes.txt",))

    result = guard.check_ship_set(root)
    assert not result.ok, result.problems
    assert any(
        "notes.txt" in problem and "NOT tracked" in problem for problem in result.problems
    ), result.problems


def test_the_stale_manifest_branch_reports_a_declared_asset_that_is_gone(
    monkeypatch, tmp_path
):
    """The branch that stops the manifest becoming a no-op (Review, missing test 3).

    `untracked_ship_assets` reports a declared asset that is absent from disk as
    well as from git, so a rename cannot quietly turn an entry into a check of
    nothing. That report is deliberately made only when the tree being checked IS
    this project, because the guard also runs against throwaway fixture trees that
    have no reason to contain LOCITIZE assets - and that condition is precisely why no
    fixture-based test could reach it. PLATFORM_DIR is therefore pointed at the
    fixture for the length of this test, which is the only way in.
    """
    root = _tracked_repo(tmp_path, "stale_manifest")
    monkeypatch.setattr(guard, "PLATFORM_DIR", root.resolve())
    monkeypatch.setattr(guard, "SHIP_CRITICAL_ASSETS", ("renamed_away.yaml",))

    problems = guard.untracked_ship_assets(root, root)
    assert any(
        "renamed_away.yaml" in problem and "does not exist on disk" in problem
        for problem in problems
    ), problems

    # And the suppression it is paired with: the same manifest against a tree that
    # is NOT this project reports nothing, which is what keeps fixture trees quiet.
    monkeypatch.setattr(guard, "PLATFORM_DIR", (root / "elsewhere").resolve())
    assert guard.untracked_ship_assets(root, root) == []
