"""Config architecture tests: shipped templates, data root, migration (AC-M14-3).

Covers Architecture M14.2.2 / M14.2.3 / DEC-M14-4 - the change that makes a
maintainer's own machine layout structurally unshippable:

- the repo ships TEMPLATES only (settings.default.yaml, models.default.yaml),
  with every machine-specific value empty and zero model rows;
- the live settings.yaml / models.yaml belong to the user and live in one data
  root, resolved from the environment, a portable marker folder, or LOCALAPPDATA;
- moving to that layout is a COPY. The user's existing file is never moved,
  never edited and never deleted, because getting this wrong would cost somebody
  their working configuration - and a file that is only ever copied cannot.

The version bump 1 -> 2 is deliberately not a rewrite: a version 1 file parses
unchanged under the version 2 reader, so downgrading to an older LOCITIZE still
works. That is asserted here too.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from config import (
    CURRENT_SETTINGS_VERSION,
    DEFAULT_MODELS_FILE,
    DEFAULT_SETTINGS_FILE,
    MODELS_FILE,
    SETTINGS_FILE,
    Config,
    ensure_user_config,
    resolve_data_dir,
)

PLATFORM_DIR = Path(__file__).resolve().parent.parent

# A realistic pre-M14 settings.yaml: version 1, and carrying the two values the
# scrub targets (a live secure_proxy with a hand-written file path, and a default
# model id). It stands in for the maintainer's real file without being it.
LEGACY_SETTINGS_V1 = """# a user's own settings, hand-maintained
version: 1
paths:
  llama_cpp: "/somewhere/llama-server.exe"
ports:
  llama_cpp: 8080
secure_proxy:
  enabled: true
  caddyfile: "/somewhere/Caddyfile"
launcher:
  default_model: their-own-model
"""

LEGACY_MODELS_V1 = """version: 1
models:
  - id: their-own-model
    name: "Their Own Model"
    description: "a model only this machine has"
    location: "/somewhere/models/theirs.gguf"
    context_size: 8192
    gpu_layers: 999
    status: installed
"""


# --------------------------------------------------------------------------- #
# (a) the shipped model registry template carries zero rows
# --------------------------------------------------------------------------- #


def test_config_migration_models_template_ships_zero_rows():
    """models.default.yaml parses to an empty registry.

    A shipped row is a claim that a specific file exists on this machine, which
    is false everywhere but the machine it was written on - fabricated data
    reachable from the running app. Zero rows is the honest state.
    """
    raw = yaml.safe_load(
        (PLATFORM_DIR / DEFAULT_MODELS_FILE).read_text(encoding="utf-8")
    )
    assert raw["models"] == [], "the shipped model registry template must be empty"


def test_config_migration_models_template_example_is_never_parsed(tmp_path):
    """The commented example block is documentation, not a phantom model.

    Loading the template as if it were the live registry must still yield no
    models, which is what proves the example cannot leak into the app.
    """
    (tmp_path / MODELS_FILE).write_text(
        (PLATFORM_DIR / DEFAULT_MODELS_FILE).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / SETTINGS_FILE).write_text(
        (PLATFORM_DIR / DEFAULT_SETTINGS_FILE).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _settings, models, _issues = Config.load(tmp_path)
    assert models.models == []


# --------------------------------------------------------------------------- #
# (b) the shipped settings template carries no machine-specific default
# --------------------------------------------------------------------------- #


def test_config_migration_settings_template_has_no_machine_specific_default(tmp_path):
    """Every filesystem-shaped default in settings.default.yaml is empty.

    Checked through the real loader rather than by reading the text, so this
    asserts what the application would actually receive.
    """
    (tmp_path / SETTINGS_FILE).write_text(
        (PLATFORM_DIR / DEFAULT_SETTINGS_FILE).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / MODELS_FILE).write_text("version: 1\nmodels: []\n", encoding="utf-8")
    settings, _models, _issues = Config.load(tmp_path)

    for name in (
        "llama_cpp",
        "whisper",
        "whisper_model",
        "whisper_stream",
        "kokoro",
        "kokoro_model",
        "kokoro_voices",
        "venv",
    ):
        assert getattr(settings.paths, name) == "", f"paths.{name} must ship empty"

    assert settings.secure_proxy.caddyfile == ""
    assert settings.secure_proxy.caddy_path == ""
    assert settings.finetune.studio_dir == ""
    assert settings.finetune.outputs_dir == ""
    assert settings.finetune.python == ""
    assert settings.openwebui.backend_base_url == ""
    # A default naming a model id is the same class of defect as a default naming
    # a path: it is a claim about one machine's contents (M14.2.6).
    assert settings.launcher.default_model == ""


def test_config_migration_settings_template_defaults_are_safe(tmp_path):
    """The two consent-sensitive features ship OFF (AC-M14-9, M14.2.6).

    secure_proxy installs software with winget, binds 443 and writes to the
    current-user certificate store. None of that may happen to somebody who
    merely started the app, so it is opt-in.
    """
    (tmp_path / SETTINGS_FILE).write_text(
        (PLATFORM_DIR / DEFAULT_SETTINGS_FILE).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / MODELS_FILE).write_text("version: 1\nmodels: []\n", encoding="utf-8")
    settings, _models, _issues = Config.load(tmp_path)

    assert settings.secure_proxy.enabled is False
    assert settings.proxy.enabled is False
    assert settings.version == CURRENT_SETTINGS_VERSION


def test_config_migration_code_defaults_carry_no_machine_specific_value():
    """Even with no config file at all, no default names a machine's contents.

    Tier 1 of M14.2.1: the code itself must be clean, so a missing or unreadable
    settings.yaml can never fall back onto somebody else's layout.
    """
    from config import LauncherConfig, PathsConfig, SecureProxyConfig

    paths = PathsConfig()
    for f in paths.__dataclass_fields__:
        assert getattr(paths, f) == "", f"PathsConfig.{f} default must be empty"
    assert SecureProxyConfig().enabled is False
    assert SecureProxyConfig().caddyfile == ""
    assert LauncherConfig().default_model == ""


# --------------------------------------------------------------------------- #
# (c) the migration copies; it never moves and never edits
# --------------------------------------------------------------------------- #


def test_config_migration_copies_legacy_settings_leaving_original_identical(tmp_path):
    """A pre-M14 file beside the code is COPIED into the data root, untouched.

    This is the assertion that protects a real user's working configuration: the
    original must still be there, byte for byte, after the migration has run.
    """
    install = tmp_path / "install"
    install.mkdir()
    data = tmp_path / "data-root"
    legacy = install / SETTINGS_FILE
    legacy.write_text(LEGACY_SETTINGS_V1, encoding="utf-8")
    (install / MODELS_FILE).write_text(LEGACY_MODELS_V1, encoding="utf-8")
    before = legacy.read_bytes()

    actions = ensure_user_config(data, install)

    # The original is still where it was, with exactly the same bytes.
    assert legacy.exists(), "the migration must never move the user's file"
    assert legacy.read_bytes() == before, "the migration must never edit it either"
    # And the copy landed in the data root with identical content.
    assert (data / SETTINGS_FILE).read_bytes() == before
    assert (data / MODELS_FILE).read_text(encoding="utf-8") == LEGACY_MODELS_V1
    assert {a.filename: a.kind for a in actions} == {
        SETTINGS_FILE: "copied-legacy",
        MODELS_FILE: "copied-legacy",
    }


def test_config_migration_never_overwrites_an_existing_live_file(tmp_path):
    """Running the migration again leaves the live file alone.

    Startup calls this on every launch, so a second run must be a no-op. If it
    were not, every restart would silently revert the user's edits.
    """
    install = tmp_path / "install"
    install.mkdir()
    (install / SETTINGS_FILE).write_text(LEGACY_SETTINGS_V1, encoding="utf-8")
    (install / MODELS_FILE).write_text(LEGACY_MODELS_V1, encoding="utf-8")
    data = tmp_path / "data-root"
    data.mkdir()
    live = data / SETTINGS_FILE
    live.write_text("version: 2\nports:\n  llama_cpp: 8088\n", encoding="utf-8")
    mine = live.read_bytes()

    actions = ensure_user_config(data, install)

    assert live.read_bytes() == mine
    assert {a.kind for a in actions if a.filename == SETTINGS_FILE} == {"kept"}


def test_config_migration_seeds_from_the_template_on_a_clean_machine(tmp_path):
    """With no legacy file, the shipped templates seed the data root.

    This is the first-run path on a machine that has never had LOCITIZE: the user
    gets a full commented settings file and an empty model registry.
    """
    install = tmp_path / "install"
    install.mkdir()
    for name in (DEFAULT_SETTINGS_FILE, DEFAULT_MODELS_FILE):
        (install / name).write_text(
            (PLATFORM_DIR / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    data = tmp_path / "data-root"

    actions = ensure_user_config(data, install)

    assert {a.filename: a.kind for a in actions} == {
        SETTINGS_FILE: "copied-template",
        MODELS_FILE: "copied-template",
    }
    settings, models, issues = Config.load(data)
    assert models.models == []
    assert settings.paths.llama_cpp == ""
    assert not [i for i in issues if i.level == "ERROR"]


# --------------------------------------------------------------------------- #
# (d) both schema versions parse - forwards and backwards
# --------------------------------------------------------------------------- #


def test_config_migration_copied_v1_file_parses_under_the_v2_reader(tmp_path):
    """The copied version 1 file loads, unmigrated, with its values intact.

    The migration deliberately does not rewrite the version line. This is why
    that is safe: the current reader parses a version 1 file and every value the
    user set still takes effect.
    """
    install = tmp_path / "install"
    install.mkdir()
    (install / SETTINGS_FILE).write_text(LEGACY_SETTINGS_V1, encoding="utf-8")
    (install / MODELS_FILE).write_text(LEGACY_MODELS_V1, encoding="utf-8")
    data = tmp_path / "data-root"

    ensure_user_config(data, install)
    settings, models, issues = Config.load(data)

    assert settings.version == 1  # not rewritten
    assert settings.paths.llama_cpp == "/somewhere/llama-server.exe"
    assert settings.secure_proxy.enabled is True  # the user's choice survives
    assert settings.secure_proxy.caddyfile == "/somewhere/Caddyfile"
    assert settings.launcher.default_model == "their-own-model"
    assert [m.id for m in models.models] == ["their-own-model"]
    assert not [i for i in issues if i.level == "ERROR"]


def test_config_migration_v2_file_parses_and_reports_no_version_warning(tmp_path):
    """A version 2 file is the current shape and loads without complaint."""
    (tmp_path / SETTINGS_FILE).write_text(
        "version: 2\nports:\n  llama_cpp: 8080\n", encoding="utf-8"
    )
    (tmp_path / MODELS_FILE).write_text("version: 1\nmodels: []\n", encoding="utf-8")
    settings, _models, issues = Config.load(tmp_path)
    assert settings.version == 2
    assert not [i for i in issues if "settings version" in i.message]


def test_config_migration_unknown_version_warns_but_still_parses(tmp_path):
    """A file from a newer LOCITIZE warns and loads anyway.

    Refusing to start would lock a user out of their own configuration over a
    number, which is never the right trade for a local tool.
    """
    (tmp_path / SETTINGS_FILE).write_text(
        "version: 99\nports:\n  llama_cpp: 8087\n", encoding="utf-8"
    )
    (tmp_path / MODELS_FILE).write_text("version: 1\nmodels: []\n", encoding="utf-8")
    settings, _models, issues = Config.load(tmp_path)
    assert settings.ports.llama_cpp == 8087
    assert any("settings version 99" in i.message for i in issues)
    assert not [i for i in issues if i.level == "ERROR"]


# --------------------------------------------------------------------------- #
# Data root resolution (Architecture M14.2.3)
# --------------------------------------------------------------------------- #


def test_config_migration_data_root_prefers_the_environment_variable(tmp_path):
    """LOCITIZE_DATA_DIR wins over everything - the explicit user override."""
    chosen = tmp_path / "elsewhere"
    env = {"LOCITIZE_DATA_DIR": str(chosen), "LOCALAPPDATA": str(tmp_path / "local")}
    assert resolve_data_dir(env, tmp_path) == chosen


def test_config_migration_data_root_uses_the_portable_marker_when_present(tmp_path):
    """An existing locitize-data/ folder beside the install selects portable mode.

    Creating the folder is the entire opt-in, which is what makes a dev checkout
    or a USB-stick install one mkdir away.
    """
    install = tmp_path / "install"
    (install / "locitize-data").mkdir(parents=True)
    env = {"LOCALAPPDATA": str(tmp_path / "local")}
    assert resolve_data_dir(env, install) == install / "locitize-data"


def test_config_migration_data_root_falls_back_to_local_appdata(tmp_path):
    """With no override and no marker folder, LOCALAPPDATA wins.

    LOCALAPPDATA and not APPDATA on purpose: this tree holds logs, transcripts,
    chat databases, fetched binaries and GGUF weights - tens of gigabytes that
    must never be roamed onto an enterprise profile.
    """
    install = tmp_path / "install"
    install.mkdir()
    local = tmp_path / "local"
    env = {"LOCALAPPDATA": str(local), "APPDATA": str(tmp_path / "roaming")}
    resolved = resolve_data_dir(env, install)
    assert resolved == local / "LOCITIZE"
    assert str(tmp_path / "roaming") not in str(resolved)


def test_config_migration_data_root_resolution_creates_nothing(tmp_path):
    """Resolving is a pure calculation; only ensure_user_config touches disk."""
    install = tmp_path / "install"
    install.mkdir()
    env = {"LOCALAPPDATA": str(tmp_path / "local")}
    resolved = resolve_data_dir(env, install)
    assert not resolved.exists()


def test_config_migration_explicit_base_dir_keeps_config_self_contained(tmp_path):
    """Config.load(base_dir) reads and writes config in that one directory.

    Tests and tools point the loader at a scratch directory; if data_dir did not
    follow, a save action in one of them would write into the real, live
    models.yaml on the developer's machine.
    """
    (tmp_path / SETTINGS_FILE).write_text("version: 2\n", encoding="utf-8")
    (tmp_path / MODELS_FILE).write_text("version: 1\nmodels: []\n", encoding="utf-8")
    settings, _models, _issues = Config.load(tmp_path)
    assert settings.base_dir == tmp_path
    assert settings.data_dir == tmp_path
