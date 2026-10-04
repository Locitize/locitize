"""AC-M14-28: one registry-write chokepoint, one typed error, and one removed key.

Two decisions meet in this file because one criterion grades both.

DEC-M14-11 (NEW-QA-M14-9). QA watched the real Save-edits and Rename surfaces
print `[Errno 2] No such file or directory:` followed by a doubled-backslash
Python repr of models.yaml, with no next step. All four writers raised the raw
error. Now every models.yaml mutation goes through ONE private write-open in
config.py, which raises ONE typed RegistryWriteError whose str() is a finished
sentence: what happened, the real path via str(path), and what to do next.

DEC-M14-10 (SEC-M14-3). `LOCITIZE_MODELS_HUB_API_BASE` is gone from the product.
A control that decides WHO LOCITIZE talks to may be widened only from a file the
user owns and can audit. The grep below is this criterion's own test, and is the
only place in Codebase/platform where that name may still appear.

Keyword: registry_write_errors (see AC-M14-28's verification command).
"""

from __future__ import annotations

import ast
import os
import stat
import sys
from pathlib import Path

import config
import pytest
from config import RegistryWriteError

PLATFORM_DIR = Path(__file__).resolve().parent.parent

# scripts/ is not a package, so the W1 fence is reached the same way
# test_data_root_ownership.py reaches it. The structural check below borrows the
# fence's write-primitive table rather than keeping a second, weaker copy of it.
if str(PLATFORM_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR / "scripts"))

# The registry as it exists on a working install: one row, all the keys the
# writers edit.
GOOD_REGISTRY = (
    "version: 1\n"
    "models:\n"
    '  - id: "m"\n'
    '    name: "M"\n'
    '    description: ""\n'
    '    location: "/locitize-test/models/m.gguf"\n'
    "    context_size: 4096\n"
    "    gpu_layers: 999\n"
    "    benchmark_score: null\n"
    "    status: installed\n"
)


def all_four_writers(tmp_path: Path):
    """Every public models.yaml writer, as a zero-argument call against tmp_path.

    Parametrising over the four is the point: the defect was that each writer
    carried its own copy of the failure behaviour, so a test that checked one
    proved nothing about the other three.
    """
    gguf = tmp_path / "models" / "new.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    if not gguf.exists():
        gguf.write_bytes(b"gguf")
    return {
        "write_model_fields": lambda: config.write_model_fields(tmp_path, "m", 10, 2048),
        "write_model_identity": lambda: config.write_model_identity(
            tmp_path, "m", "m2", "M2"
        ),
        "write_model_score": lambda: config.write_model_score(tmp_path, "m", 42.0),
        "append_model_entry": lambda: config.append_model_entry(
            tmp_path, "new", "New", str(gguf)
        ),
    }


WRITER_NAMES = (
    "write_model_fields",
    "write_model_identity",
    "write_model_score",
    "append_model_entry",
)


def assert_finished_sentence(message: str, path: Path) -> None:
    """The shared shape every RegistryWriteError message must have."""
    assert str(path) in message, "the message must name the real file"
    assert "\\\\" not in message, (
        "the path was rendered as a repr (doubled separators) - the exact thing "
        "NEW-QA-M14-9 was raised about"
    )
    assert "Errno" not in message and "WinError" not in message, (
        "a bare errno is a diagnostic, not something a user can act on"
    )
    assert "try again" in message or "then " in message, "a next step is required"
    assert message.rstrip().endswith("."), "a finished sentence ends in a full stop"


# --------------------------------------------------------------------------- #
# The three failure conditions, against all four writers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("writer", WRITER_NAMES)
def test_registry_write_errors_absent_file_is_typed_and_actionable(tmp_path, writer):
    """models.yaml deleted or renamed away while LOCITIZE is open - QA's repro."""
    with pytest.raises(RegistryWriteError) as caught:
        all_four_writers(tmp_path)[writer]()
    message = str(caught.value)
    assert_finished_sentence(message, tmp_path / "models.yaml")
    assert "missing" in message


@pytest.mark.parametrize("writer", WRITER_NAMES)
def test_registry_write_errors_read_only_file_is_typed_and_actionable(tmp_path, writer):
    """A read-only models.yaml - the live failure QA produced with the downloader.

    The file is really made read-only and really written to; nothing is mocked,
    because the OSError being translated is the whole subject of the test.
    """
    registry = tmp_path / "models.yaml"
    registry.write_text(GOOD_REGISTRY, encoding="utf-8")
    os.chmod(registry, stat.S_IREAD)
    try:
        with pytest.raises(RegistryWriteError) as caught:
            all_four_writers(tmp_path)[writer]()
    finally:
        os.chmod(registry, stat.S_IWRITE | stat.S_IREAD)
    message = str(caught.value)
    assert_finished_sentence(message, registry)
    assert "read-only" in message
    # The temp file the atomic write used is an implementation detail the user
    # has never heard of, and it is what made QA's message unreadable.
    assert ".tmp" not in message
    # The file itself is untouched by the failed write.
    assert registry.read_text(encoding="utf-8") == GOOD_REGISTRY


@pytest.mark.parametrize("writer", WRITER_NAMES)
def test_registry_write_errors_missing_models_section_is_typed_and_actionable(
    tmp_path, writer
):
    """A registry with no top-level `models:` key: refuse, do not guess."""
    registry = tmp_path / "models.yaml"
    registry.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(RegistryWriteError) as caught:
        all_four_writers(tmp_path)[writer]()
    message = str(caught.value)
    assert_finished_sentence(message, registry)
    assert "models:" in message and "refuses to guess" in message
    assert registry.read_text(encoding="utf-8") == "version: 1\n"


@pytest.mark.parametrize("writer", WRITER_NAMES)
def test_registry_write_errors_never_leak_a_bare_builtin_exception(tmp_path, writer):
    """The raised type is RegistryWriteError, never FileNotFoundError/OSError/ValueError.

    Checked as an exact type rather than with isinstance, because a
    RegistryWriteError that quietly subclassed ValueError would satisfy every
    other assertion in this file while leaving the caller's `except ValueError`
    doing the same undifferentiated thing it did before.
    """
    with pytest.raises(BaseException) as caught:
        all_four_writers(tmp_path)[writer]()
    assert type(caught.value) is RegistryWriteError
    assert not isinstance(caught.value, (OSError, ValueError))


def test_registry_write_errors_post_write_verification_failure_is_typed(tmp_path):
    """The fourth cause: the re-parsed document does not carry what was written."""
    registry = tmp_path / "models.yaml"
    with pytest.raises(RegistryWriteError) as caught:
        config._confirm_appended({"models": []}, "m", "/locitize-test/x.gguf", registry)
    message = str(caught.value)
    assert_finished_sentence(message, registry)
    assert "post-write verification failed" in message
    assert "file left unchanged" in message


def test_registry_write_errors_a_good_write_still_succeeds(tmp_path):
    """The chokepoint is a guard, not a wall: the happy path is unchanged."""
    registry = tmp_path / "models.yaml"
    registry.write_text(GOOD_REGISTRY, encoding="utf-8")
    config.write_model_fields(tmp_path, "m", 10, 2048)
    config.write_model_score(tmp_path, "m", 42.0)
    config.write_model_identity(tmp_path, "m", "m2", "M2")
    text = registry.read_text(encoding="utf-8")
    assert "gpu_layers: 10" in text
    assert "context_size: 2048" in text
    assert "benchmark_score: 42.0" in text
    assert 'id: m2' in text and 'name: "M2"' in text


def test_registry_write_errors_caller_input_refusals_stay_value_errors(tmp_path):
    """A bad ARGUMENT is still a ValueError: only file-level faults are typed.

    The distinction is load-bearing. RegistryWriteError says "something about
    the file stopped this"; ValueError says "what you asked for is not allowed".
    Collapsing the two would tell a user to check their file permissions when
    they typed a negative context size.
    """
    (tmp_path / "models.yaml").write_text(GOOD_REGISTRY, encoding="utf-8")
    with pytest.raises(ValueError) as bad_value:
        config.write_model_fields(tmp_path, "m", 10, -1)
    assert type(bad_value.value) is ValueError

    with pytest.raises(ValueError) as unknown_id:
        config.write_model_fields(tmp_path, "nope", 10, 2048)
    assert "not found" in str(unknown_id.value)


# --------------------------------------------------------------------------- #
# The structural half: exactly one write-open of the registry path
# --------------------------------------------------------------------------- #


# config.py's own atomic-write helper. Every other write primitive - open() in a
# writing mode, Path.write_text/write_bytes, shutil.copy*/move, os.replace/rename,
# mkdir - is recognised by verify_write_fence.write_targets, which is the single
# table of "what counts as a write" in this repository.
_PROJECT_WRITE_CALLS = frozenset({"_atomic_write"})

# Functions that write models.yaml but are NOT registry EDITS, each with the
# guard that makes it safe. ensure_user_config seeds a missing registry with
# shutil.copy2 and returns early when the file already exists, so it can create
# but never modify or destroy a user's model list (review round 6, LOW-3). Its
# create-only guard is asserted below rather than trusted.
_CREATE_ONLY_WRITERS = ("ensure_user_config",)


def _writes_the_registry(node: ast.FunctionDef) -> bool:
    """True when this function names models.yaml AND invokes a write primitive.

    The detector this replaces counted `open` calls with more than one POSITIONAL
    argument, which made `path.open("w", encoding="utf-8")` - the idiomatic shape
    used throughout this codebase - completely invisible; a rogue writer in that
    form was appended to config.py during review and the suite stayed green
    (round 6, HIGH-3). Rather than fix that copy of the question, the check now
    asks the fence's own table, so the two can never disagree again.
    """
    import verify_write_fence

    names_registry = any(
        isinstance(child, ast.Name) and child.id == "MODELS_FILE"
        for child in ast.walk(node)
    )
    if not names_registry:
        return False
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        callee = getattr(child.func, "id", getattr(child.func, "attr", ""))
        if callee in _PROJECT_WRITE_CALLS or verify_write_fence.write_targets(child):
            return True
    return False


def _functions_writing_the_registry(source: str) -> list[str]:
    """Names of functions in config.py that both name models.yaml AND write."""
    tree = ast.parse(source)
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and _writes_the_registry(node)
    ]


def test_registry_write_errors_exactly_one_write_open_of_the_registry_exists():
    """DEC-M14-11's structure: one EDITING chokepoint, four thin callers.

    "Exactly one" is asserted against the full set of write primitives, not
    against one spelling of open(). Any writer beyond the chokepoint must be on
    the create-only list, and the next test proves each of those really is
    create-only rather than merely declared so.
    """
    source = (PLATFORM_DIR / "config.py").read_text(encoding="utf-8")
    writers = _functions_writing_the_registry(source)
    expected = ["_edit_registry", *_CREATE_ONLY_WRITERS]
    assert sorted(writers) == sorted(expected), (
        f"expected exactly one registry EDIT chokepoint (_edit_registry) plus the "
        f"declared create-only seeders {list(_CREATE_ONLY_WRITERS)}, found: {writers}"
    )


def test_registry_write_errors_the_declared_create_only_writers_really_are_create_only():
    """A named exception has to earn its exemption, not just be listed.

    ensure_user_config is allowed to write models.yaml because it refuses to
    touch a file that already exists. That refusal is the whole exemption, so it
    is read out of the source: an `exists()` test whose body skips the copy.
    """
    tree = ast.parse((PLATFORM_DIR / "config.py").read_text(encoding="utf-8"))
    for name in _CREATE_ONLY_WRITERS:
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        guards = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and any(
                getattr(call.func, "attr", "") in ("exists", "is_file")
                for call in ast.walk(node.test)
                if isinstance(call, ast.Call)
            )
            and any(isinstance(stmt, (ast.Continue, ast.Return)) for stmt in node.body)
        ]
        assert guards, (
            f"{name} is exempted from the one-chokepoint rule as create-only, but "
            f"has no exists()/is_file() guard that skips an existing registry"
        )


@pytest.mark.parametrize(
    "rogue",
    [
        # The exact shape Reviewer appended to config.py in round 6: one
        # positional argument, so the old detector never saw it.
        'def rogue(base_dir):\n'
        '    path = Path(base_dir) / MODELS_FILE\n'
        '    with path.open("w", encoding="utf-8") as handle:\n'
        '        handle.write("models: []")\n',
        # The mode by keyword instead of by position.
        'def rogue(base_dir):\n'
        '    open(Path(base_dir) / MODELS_FILE, mode="w").write("models: []")\n',
        # A copy over the registry, which is how ensure_user_config writes it.
        'import shutil\n'
        'def rogue(base_dir, src):\n'
        '    shutil.copy2(src, Path(base_dir) / MODELS_FILE)\n',
        # Path.write_text, the shortest spelling of the same mistake.
        'def rogue(base_dir):\n'
        '    (Path(base_dir) / MODELS_FILE).write_text("models: []")\n',
        # An atomic rename over the registry.
        'import os\n'
        'def rogue(base_dir, tmp):\n'
        '    os.replace(tmp, Path(base_dir) / MODELS_FILE)\n',
        # io.open: the builtin under another spelling, and the one shape that
        # was still invisible in round 7 (MEDIUM-3r) because the detector chose
        # the mode argument by node type instead of by call shape.
        'import io\n'
        'def rogue(base_dir):\n'
        '    io.open(Path(base_dir) / MODELS_FILE, "w").write("models: []")\n',
    ],
    ids=[
        "path-open-w",
        "open-mode-kw",
        "shutil-copy2",
        "write-text",
        "os-replace",
        "io-open",
    ],
)
def test_registry_write_errors_the_structural_check_catches_a_second_writer(rogue):
    """The negative half: prove the detector above CAN fail, for each shape.

    Eight tests in this milestone turned out to be unable to fail for the
    property their name claimed. The only defence is to make each structural
    check demonstrate its own failure, permanently and in the suite - so every
    write primitive a second registry writer could plausibly use is fed to the
    detector here and must be reported.
    """
    assert _functions_writing_the_registry(rogue) == ["rogue"], (
        "a second registry writer in this form is invisible to the AC-M14-28 "
        "structural check"
    )


def test_registry_write_errors_no_call_site_catches_oserror_from_a_registry_write():
    """DEC-M14-11 rule 5: no caller may compose its own message from an OSError."""
    for module in ("gui_controller.py", "benchmark.py"):
        source = (PLATFORM_DIR / module).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            calls = {
                getattr(child.func, "id", getattr(child.func, "attr", ""))
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
            }
            if not calls & {
                "write_model_fields",
                "write_model_identity",
                "write_model_score",
                "append_model_entry",
            }:
                continue
            caught = set()
            for handler in node.handlers:
                for name in ast.walk(handler.type) if handler.type else []:
                    if isinstance(name, ast.Name):
                        caught.add(name.id)
            assert "OSError" not in caught, (
                f"{module}:{node.lineno} catches OSError around a registry write; "
                f"the chokepoint owns that translation (DEC-M14-11 rule 5)"
            )
            assert "RegistryWriteError" in caught, (
                f"{module}:{node.lineno} guards a registry write without handling "
                f"RegistryWriteError"
            )


# --------------------------------------------------------------------------- #
# DEC-M14-10: the removed environment override
# --------------------------------------------------------------------------- #


def test_registry_write_errors_api_base_env_override_is_gone_from_the_product():
    """AC-M14-28 (a): the name appears nowhere in the tree except in this file.

    Scanned over the source rather than by behaviour, because "the product no
    longer reads this variable" is a claim about the absence of code, and the
    only honest way to check an absence is to look everywhere.
    """
    banned = "LOCITIZE_MODELS_HUB" + "_API_BASE"  # split so the grep below is exact
    this_file = Path(__file__).resolve()
    offenders = []
    for path in PLATFORM_DIR.rglob("*"):
        if not path.is_file() or path.resolve() == this_file:
            continue
        if any(part in {"__pycache__", ".venv", ".git"} for part in path.parts):
            continue
        if path.suffix.lower() not in {
            ".py", ".yaml", ".yml", ".json", ".md", ".txt", ".bat", ".ps1", ".cfg", ".ini"
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if banned in text:
            offenders.append(str(path.relative_to(PLATFORM_DIR)))
    assert offenders == [], f"{banned} still appears in: {offenders}"


def test_registry_write_errors_api_base_is_settings_only_and_env_cannot_widen_it(
    tmp_path
):
    """The behaviour behind the removal: the environment no longer moves the boundary."""
    (tmp_path / "settings.yaml").write_text(
        "version: 2\nmodels_hub:\n  api_base: https://huggingface.co\n", encoding="utf-8"
    )
    (tmp_path / "models.yaml").write_text(GOOD_REGISTRY, encoding="utf-8")

    settings, _models, _issues = config.Config.load(
        tmp_path,
        env={
            "LOCITIZE_MODELS_HUB" + "_API_BASE": "https://evil.example.com",
            # The two keys that DO survive still work, so this is a targeted
            # removal rather than the env chain going quiet.
            "LOCITIZE_MODELS_HUB_DOWNLOAD_DIR": str(tmp_path / "elsewhere"),
            "LOCITIZE_MODELS_HUB_ENABLED": "false",
        },
    )
    assert settings.models_hub.api_base == "https://huggingface.co"
    assert settings.models_hub.download_dir == str(tmp_path / "elsewhere")
    assert settings.models_hub.enabled is False


def test_registry_write_errors_hub_config_api_base_remains_the_test_seam():
    """DEC-M14-10 item 3: configurability moves to the constructor, not the env."""
    import modelhub

    hub = modelhub.HubConfig(api_base="https://mirror.example.com")
    assert hub.api_base == "https://mirror.example.com"
