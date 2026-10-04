"""Config-loading tests: valid, partial, and malformed fixtures plus env layering.

Points Config.load() at temp fixture directories so the typed output, defaults,
env overrides, and the parse report are all asserted without touching the shipped
settings.yaml/models.yaml (Architecture section 12).
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from config import (
    Config,
    _is_safe_health_path,
    clear_launcher_default_model,
    settings_to_dict,
    write_chat_harness_dir,
    write_speech_noise_suppression,
)


def _write(base: Path, name: str, text: str) -> None:
    (base / name).write_text(text, encoding="utf-8")


VALID_SETTINGS = """
version: 1
paths:
  llama_cpp: ""
ports:
  allocation: auto
  range_start: 8080
  range_end: 8099
  llama_cpp: 8080
launcher:
  default_model: qwen3-14b
"""

VALID_MODELS = """
version: 1
models:
  - id: qwen3-14b
    name: Qwen3 14B
    description: general
    location: ""
    context_size: 32768
    gpu_layers: -1
    status: installed
    vram_estimate_mb: 10000
"""


def test_valid_config_parses_typed(tmp_path):
    _write(tmp_path, "settings.yaml", VALID_SETTINGS)
    _write(tmp_path, "models.yaml", VALID_MODELS)
    settings, models, issues = Config.load(tmp_path)

    assert settings.version == 1
    assert settings.ports.allocation == "auto"
    assert settings.launcher.default_model == "qwen3-14b"
    assert len(models.models) == 1
    assert models.models[0].id == "qwen3-14b"
    assert models.models[0].context_size == 32768
    # No ERROR-level issues for a clean config.
    assert not [i for i in issues if i.level == "ERROR"]


def test_missing_settings_falls_back_to_defaults(tmp_path):
    """Absent settings.yaml -> code defaults + a WARNING issue (not a crash)."""
    _write(tmp_path, "models.yaml", VALID_MODELS)
    settings, models, issues = Config.load(tmp_path)
    assert settings.ports.range_start == 8080  # default applied
    assert any(i.source == "settings.yaml" and i.level == "WARNING" for i in issues)


def test_missing_models_reports_error(tmp_path):
    """Absent models.yaml is an ERROR-level issue the launcher treats as fatal."""
    _write(tmp_path, "settings.yaml", VALID_SETTINGS)
    _settings, _models, issues = Config.load(tmp_path)
    assert any(i.source == "models.yaml" and i.level == "ERROR" for i in issues)


def test_malformed_yaml_reports_error(tmp_path):
    """A malformed yaml file yields an ERROR issue rather than raising."""
    _write(tmp_path, "settings.yaml", "paths: [unterminated\n")
    _write(tmp_path, "models.yaml", VALID_MODELS)
    _settings, _models, issues = Config.load(tmp_path)
    assert any(i.source == "settings.yaml" and i.level == "ERROR" for i in issues)


def test_partial_config_fills_defaults(tmp_path):
    """A partial settings.yaml keeps code defaults for unspecified fields."""
    _write(tmp_path, "settings.yaml", "version: 1\nports:\n  llama_cpp: 8085\n")
    _write(tmp_path, "models.yaml", VALID_MODELS)
    settings, _models, _issues = Config.load(tmp_path)
    assert settings.ports.llama_cpp == 8085  # from file
    assert settings.ports.whisper == 8091  # default retained
    assert settings.speech.noise_suppression == "balanced"


def test_noise_suppression_enum_is_validated(tmp_path):
    _write(
        tmp_path,
        "settings.yaml",
        "version: 1\nspeech:\n  noise_suppression: arbitrary-filter\n",
    )
    _write(tmp_path, "models.yaml", VALID_MODELS)
    settings, _models, issues = Config.load(tmp_path)
    assert settings.speech.noise_suppression == "balanced"
    assert any("speech.noise_suppression" in issue.message for issue in issues)


def test_noise_suppression_writer_changes_only_its_scalar(tmp_path):
    original = """\
version: 2
speech:
  stream_step_ms: 0
  vad_thold: 0.60  # owner sensitivity
  freq_thold: 100.0
  noise_suppression: balanced  # owner mode
tts:
  enabled: true
"""
    _write(tmp_path, "settings.yaml", original)
    _write(tmp_path, "models.yaml", VALID_MODELS)

    write_speech_noise_suppression(tmp_path, "STRONG")

    text = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert 'noise_suppression: "strong"  # owner mode' in text
    assert "vad_thold: 0.60  # owner sensitivity" in text
    assert "tts:\n  enabled: true" in text
    assert Config.load(tmp_path)[0].speech.noise_suppression == "strong"


def test_noise_suppression_writer_inserts_for_an_older_settings_file(tmp_path):
    _write(
        tmp_path,
        "settings.yaml",
        "version: 1\nspeech:\n  stream_step_ms: 0\n  vad_thold: 0.6\n",
    )
    _write(tmp_path, "models.yaml", VALID_MODELS)
    write_speech_noise_suppression(tmp_path, "off")
    text = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert text.count("noise_suppression:") == 1
    assert 'noise_suppression: "off"' in text


def test_noise_suppression_writer_refuses_invalid_or_missing_section(tmp_path):
    _write(tmp_path, "settings.yaml", "version: 1\n")
    with pytest.raises(ValueError, match="must be one of"):
        write_speech_noise_suppression(tmp_path, "custom=unsafe")
    with pytest.raises(ValueError, match="speech"):
        write_speech_noise_suppression(tmp_path, "balanced")


def test_clear_launcher_default_model_changes_only_that_scalar(tmp_path):
    original = """\
version: 1
launcher:
  auto_journal: true
  default_model: qwen3-14b  # owner default
chat:
  preferred_ui: openwebui
"""
    _write(tmp_path, "settings.yaml", original)

    clear_launcher_default_model(tmp_path)

    text = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert 'default_model: ""  # owner default' in text
    assert "auto_journal: true" in text
    assert "preferred_ui: openwebui" in text
    assert Config.load(tmp_path)[0].launcher.default_model == ""


def test_clear_launcher_default_model_refuses_to_guess(tmp_path):
    _write(tmp_path, "settings.yaml", "version: 1\nchat:\n  preferred_ui: openwebui\n")

    with pytest.raises(ValueError, match="launcher"):
        clear_launcher_default_model(tmp_path)


def test_env_override_beats_file(tmp_path):
    """LOCITIZE_* env vars override both file and defaults (highest precedence)."""
    _write(tmp_path, "settings.yaml", VALID_SETTINGS)
    _write(tmp_path, "models.yaml", VALID_MODELS)
    env = {
        "LOCITIZE_LLAMACPP_PATH": "/locitize-test/tools/llama-server.exe",
        "LOCITIZE_PORTS_LLAMACPP": "8090",
        "LOCITIZE_MODEL_QWEN3_14B": "/locitize-test/models/qwen3-14b.gguf",
    }
    settings, models, _issues = Config.load(tmp_path, env=env)
    assert settings.paths.llama_cpp == "/locitize-test/tools/llama-server.exe"
    assert settings.ports.llama_cpp == 8090
    assert models.models[0].location == "/locitize-test/models/qwen3-14b.gguf"


def test_unknown_key_is_warning_not_error(tmp_path):
    """An unknown settings key is ignored with a WARNING, not a failure."""
    _write(tmp_path, "settings.yaml", "version: 1\nports:\n  bogus: 1\n")
    _write(tmp_path, "models.yaml", VALID_MODELS)
    _settings, _models, issues = Config.load(tmp_path)
    assert any("unknown key 'bogus'" in i.message for i in issues)


def test_settings_dump_redacts_paths(tmp_path):
    """A settings dump masks path values so a layout/secret never leaks."""
    _write(tmp_path, "settings.yaml", VALID_SETTINGS)
    _write(tmp_path, "models.yaml", VALID_MODELS)
    settings, _models, _issues = Config.load(
        tmp_path, env={"LOCITIZE_LLAMACPP_PATH": "/locitize-test/secret/path.exe"}
    )
    dump = settings_to_dict(settings, redact=True)
    assert dump["paths"]["llama_cpp"] == "<set>"  # value masked, presence shown


def test_settings_dump_carries_every_settings_block(tmp_path):
    """Every settings block reaches the dump, including M13's finetune block.

    The dump is the one surface that tells the owner whether a setting or env
    override actually took, so a silently missing block turns diagnosis into
    guesswork. The finetune block's three path-shaped keys are masked on the
    same rule as the paths block.
    """
    _write(tmp_path, "settings.yaml", VALID_SETTINGS)
    _write(tmp_path, "models.yaml", VALID_MODELS)
    settings, _models, _issues = Config.load(
        tmp_path, env={"LOCITIZE_FINETUNE_STUDIO_DIR": "/locitize-test/secret/studio"}
    )
    dump = settings_to_dict(settings, redact=True)

    for block in fields(settings):
        # version, base_dir and data_dir are not settings blocks: they are the
        # schema number and the two resolved directories (code root, data root).
        if block.name in ("version", "base_dir", "data_dir"):
            continue
        assert block.name in dump, f"settings block '{block.name}' missing from dump"

    assert dump["finetune"]["studio_dir"] == "<set>"  # masked, presence shown
    assert dump["finetune"]["outputs_dir"] == ""  # unset stays visibly unset
    assert dump["finetune"]["enabled"] is False  # plain flags pass through
    assert dump["finetune"]["port"] == 8501

    # Unredacted, the owner sees the real value they need to confirm.
    raw = settings_to_dict(settings, redact=False)
    assert raw["finetune"]["studio_dir"] == "/locitize-test/secret/studio"


# --------------------------------------------------------------------------- #
# chat_harness.last_project_dir - "Launch in..." picker's remembered folder
# (owner request 2026-08-21, direct sibling of write_chat_ui / chat_persist)
# --------------------------------------------------------------------------- #

_SETTINGS_WITH_CHAT_HARNESS = """\
version: 1

chat_harness:
  # this comment must survive the write
  last_project_dir: ""

launcher:
  default_model: qwen3-14b
"""


def test_chat_harness_default_is_empty_string():
    from config import Settings

    assert Settings().chat_harness.last_project_dir == ""


def test_write_chat_harness_dir_updates_only_the_one_scalar(tmp_path):
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_CHAT_HARNESS, encoding="utf-8")
    write_chat_harness_dir(tmp_path, r"/locitize-test/My Project")
    text = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert r"/locitize-test/My Project" in text
    assert "# this comment must survive the write" in text
    assert "default_model: qwen3-14b" in text


def test_write_chat_harness_dir_round_trips_a_path_with_spaces_and_backslashes(tmp_path):
    """A backslash-and-space-bearing path (the shape a real Windows project
    folder takes) is the whole reason this writer exists instead of reusing
    write_chat_ui's bare-token _replace_scalar (D-M harness fix). No drive
    letter here on purpose - a relative-looking backslash path still exercises
    the same YAML-escaping hazard without tripping the owner-path hygiene
    scan this repo runs over its own tracked tests."""
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_CHAT_HARNESS, encoding="utf-8")
    _write(tmp_path, "models.yaml", VALID_MODELS)
    tricky = r"locitize-test\Projects\my app"
    write_chat_harness_dir(tmp_path, tricky)
    settings, _models, issues = Config.load(tmp_path)
    assert settings.chat_harness.last_project_dir == tricky
    assert not any(i.level == "ERROR" for i in issues)


def test_write_chat_harness_dir_missing_section_raises(tmp_path):
    (tmp_path / "settings.yaml").write_text(
        "version: 1\nports:\n  llama_cpp: 8080\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        write_chat_harness_dir(tmp_path, "/locitize-test/proj")


def test_write_chat_harness_dir_is_atomic_no_temp_left_behind(tmp_path):
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_CHAT_HARNESS, encoding="utf-8")
    write_chat_harness_dir(tmp_path, "/locitize-test/proj")
    assert list(tmp_path.glob("settings.yaml.*")) == []


def test_health_path_validation_rejects_authority(tmp_path):
    """SEC-1: a crafted services.llama_cpp_health_path with an authority/scheme is
    rejected at load and replaced with the safe default, so it can never reshape
    the readiness URL host."""
    _write(
        tmp_path,
        "settings.yaml",
        "version: 1\nservices:\n  llama_cpp_health_path: '//evil.example.com'\n",
    )
    _write(tmp_path, "models.yaml", VALID_MODELS)
    settings, _models, issues = Config.load(tmp_path)
    # The dangerous value is not honored; it is reset to the safe default.
    assert settings.services.llama_cpp_health_path == "/health"
    assert any("llama_cpp_health_path" in i.message for i in issues)


def test_health_path_predicate_accepts_and_rejects():
    """The SEC-1 predicate accepts rooted paths and rejects authority/scheme/ctrl
    forms (unit-level proof independent of the loader)."""
    assert _is_safe_health_path("/health")
    assert _is_safe_health_path("/v1/health")
    assert not _is_safe_health_path("//evil.example.com")
    assert not _is_safe_health_path("http://evil.example.com")
    assert not _is_safe_health_path("health")  # not rooted
    assert not _is_safe_health_path("/has space")
    assert not _is_safe_health_path("/user@host")


def test_bare_list_models_accepted(tmp_path):
    """A bare list models.yaml (loose form) still parses (AC4 tolerance)."""
    _write(tmp_path, "settings.yaml", VALID_SETTINGS)
    _write(
        tmp_path,
        "models.yaml",
        "- id: m1\n  name: M1\n  description: d\n  location: ''\n"
        "  context_size: 8192\n  gpu_layers: -1\n  status: installed\n",
    )
    _settings, models, _issues = Config.load(tmp_path)
    assert len(models.models) == 1
    assert models.models[0].id == "m1"


# --------------------------------------------------------------------------- #
# write_model_fields - targeted, comment-preserving, atomic write-back (AC12,
# keyword: write_model_fields; Architecture G6).
# --------------------------------------------------------------------------- #

from config import write_model_fields  # noqa: E402


# A realistic two-model fixture with comments and quoted values on the exact
# lines write_model_fields must edit, so the comment-preservation claim is tested
# against a file shaped like the shipped models.yaml.
_MODELS_WITH_COMMENTS = """# locitize model registry (owner-maintained).
version: 1
models:
  - id: qwen3-14b
    name: "Qwen3 14B"
    description: "General model."
    location: "/locitize-test/models/Qwen3-14B.gguf"   # or LOCITIZE_MODEL_QWEN3_14B
    context_size: 32768                        # trailing comment must survive
    gpu_layers: -1                             # -1 = all layers
    status: installed
    server_args: []

  - id: deepseek-r1-distill-14b
    name: "DeepSeek R1 Distill 14B"
    description: "Reasoning model."
    location: "/locitize-test/models/DeepSeek.gguf"
    context_size: 16384
    gpu_layers: 20
    status: installed
    server_args: []
"""


def test_write_model_fields_changes_only_the_two_target_lines(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    before = (tmp_path / "models.yaml").read_text(encoding="utf-8").splitlines()

    write_model_fields(tmp_path, "qwen3-14b", gpu_layers=10, context_size=8192)

    after = (tmp_path / "models.yaml").read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    # Exactly two lines differ, and they are the target model's scalar lines.
    diffs = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(diffs) == 2
    changed = "\n".join(a for _b, a in diffs)
    assert "context_size: 8192" in changed
    assert "gpu_layers: 10" in changed


def test_write_model_fields_preserves_inline_comments(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    write_model_fields(tmp_path, "qwen3-14b", gpu_layers=5, context_size=4096)
    text = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    # The owner's trailing comments on the edited lines are retained verbatim.
    assert "context_size: 4096                        # trailing comment must survive" in text
    assert "gpu_layers: 5                             # -1 = all layers" in text
    # The header comment and the other model are untouched.
    assert "# locitize model registry (owner-maintained)." in text
    assert "gpu_layers: 20" in text  # second model unchanged


def test_write_model_fields_scoped_to_the_named_model(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    write_model_fields(tmp_path, "deepseek-r1-distill-14b", gpu_layers=0, context_size=2048)
    _settings = None
    _s, models, _issues = Config.load(tmp_path)
    by_id = {m.id: m for m in models.models}
    # Only the second model changed; the first keeps its original values.
    assert by_id["qwen3-14b"].gpu_layers == -1
    assert by_id["qwen3-14b"].context_size == 32768
    assert by_id["deepseek-r1-distill-14b"].gpu_layers == 0
    assert by_id["deepseek-r1-distill-14b"].context_size == 2048


def test_write_model_fields_round_trips_through_safe_load(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    write_model_fields(tmp_path, "qwen3-14b", gpu_layers=-1, context_size=65536)
    # The file still parses and carries the new value (the RG3 guard proves this
    # before committing; here we confirm the committed file loads).
    _s, models, _issues = Config.load(tmp_path)
    model = next(m for m in models.models if m.id == "qwen3-14b")
    assert model.context_size == 65536


def test_write_model_fields_rejects_invalid_invariants(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    original = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        write_model_fields(tmp_path, "qwen3-14b", gpu_layers=-2, context_size=4096)
    with pytest.raises(ValueError):
        write_model_fields(tmp_path, "qwen3-14b", gpu_layers=0, context_size=0)
    with pytest.raises(ValueError):
        write_model_fields(tmp_path, "qwen3-14b", gpu_layers=True, context_size=4096)
    # A rejected write leaves the file byte-for-byte unchanged.
    assert (tmp_path / "models.yaml").read_text(encoding="utf-8") == original


def test_write_model_fields_raises_for_unknown_model(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        write_model_fields(tmp_path, "no-such-model", gpu_layers=1, context_size=4096)


def test_write_model_fields_atomic_leaves_no_temp_file(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    write_model_fields(tmp_path, "qwen3-14b", gpu_layers=8, context_size=8192)
    # os.replace is atomic; no sibling temp file should linger after a clean write.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "models.yaml"]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# write_model_identity - targeted, comment-preserving, atomic rename/re-id
# write-back (sibling of write_model_fields, keyword: write_model_identity).
# --------------------------------------------------------------------------- #

from config import write_model_identity  # noqa: E402


def test_write_model_identity_renames_id_and_name(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    write_model_identity(tmp_path, "qwen3-14b", "qwen3-14b-v2", "Qwen3 14B v2")
    _s, models, _issues = Config.load(tmp_path)
    by_id = {m.id: m for m in models.models}
    assert "qwen3-14b" not in by_id
    assert by_id["qwen3-14b-v2"].name == "Qwen3 14B v2"
    # The other model and the rest of the file are untouched.
    assert by_id["deepseek-r1-distill-14b"].name == "DeepSeek R1 Distill 14B"


def test_write_model_identity_preserves_comments_and_other_lines(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    before = (tmp_path / "models.yaml").read_text(encoding="utf-8").splitlines()
    write_model_identity(tmp_path, "qwen3-14b", "qwen3-14b", "Qwen3 14B Renamed")
    after = (tmp_path / "models.yaml").read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    diffs = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(diffs) == 1
    assert diffs[0][1].strip() == 'name: "Qwen3 14B Renamed"'
    text = "\n".join(after)
    assert 'location: "/locitize-test/models/Qwen3-14B.gguf"   # or LOCITIZE_MODEL_QWEN3_14B' in text


def test_write_model_identity_rejects_duplicate_id(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    original = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        write_model_identity(
            tmp_path, "qwen3-14b", "deepseek-r1-distill-14b", "Qwen3 14B"
        )
    assert (tmp_path / "models.yaml").read_text(encoding="utf-8") == original


def test_write_model_identity_rejects_blank_id_or_name(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    original = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        write_model_identity(tmp_path, "qwen3-14b", "  ", "Qwen3 14B")
    with pytest.raises(ValueError):
        write_model_identity(tmp_path, "qwen3-14b", "qwen3-14b", "  ")
    assert (tmp_path / "models.yaml").read_text(encoding="utf-8") == original


def test_write_model_identity_raises_for_unknown_model(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        write_model_identity(tmp_path, "no-such-model", "new-id", "New Name")


# --------------------------------------------------------------------------- #
# write_model_capabilities - targeted, atomic write-back that must both
# REPLACE an existing capabilities: line and INSERT a new one when the field
# is absent (owner request 2026-08-21, keyword: write_model_capabilities).
# --------------------------------------------------------------------------- #

from config import write_model_capabilities  # noqa: E402

# Has a nested benchmark_sweep block under the FIRST model, so its own
# gpu_layers sub-key is the block's last line - a real registry shape that
# broke the original "insert after the last line" implementation (it attached
# capabilities as a child of benchmark_sweep instead of a sibling of status).
_MODELS_WITH_NESTED_SWEEP = """version: 1
models:
  - id: qwen3-14b
    name: "Qwen3 14B"
    location: "/locitize-test/models/Qwen3-14B.gguf"
    context_size: 32768
    gpu_layers: -1
    status: installed
    server_args: []
    benchmark_sweep:
      gpu_layers: [30, 35, 999]

  - id: qwen2-5-vl
    name: "Qwen2.5 VL"
    location: "/locitize-test/models/Qwen2.5-VL.gguf"
    context_size: 16384
    gpu_layers: -1
    status: installed
    capabilities: ["vision"]
    server_args: []
"""


def test_write_model_capabilities_inserts_after_status_when_absent(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_NESTED_SWEEP, encoding="utf-8")
    write_model_capabilities(tmp_path, "qwen3-14b", ["chat", "coding"])
    _s, models, _issues = Config.load(tmp_path)
    by_id = {m.id: m for m in models.models}
    assert by_id["qwen3-14b"].capabilities == ["chat", "coding"]
    # The nested benchmark_sweep survives untouched - the regression this test
    # guards against attached capabilities to IT instead of the model row.
    assert by_id["qwen3-14b"].benchmark_sweep == {"gpu_layers": [30, 35, 999]}


def test_write_model_capabilities_replaces_an_existing_line(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_NESTED_SWEEP, encoding="utf-8")
    before = (tmp_path / "models.yaml").read_text(encoding="utf-8").splitlines()
    write_model_capabilities(tmp_path, "qwen2-5-vl", ["chat", "vision"])
    after = (tmp_path / "models.yaml").read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    diffs = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(diffs) == 1
    assert diffs[0][1].strip() == 'capabilities: ["chat", "vision"]'
    _s, models, _issues = Config.load(tmp_path)
    by_id = {m.id: m for m in models.models}
    assert by_id["qwen2-5-vl"].capabilities == ["chat", "vision"]


def test_write_model_capabilities_can_clear_to_empty(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_NESTED_SWEEP, encoding="utf-8")
    write_model_capabilities(tmp_path, "qwen2-5-vl", [])
    _s, models, _issues = Config.load(tmp_path)
    by_id = {m.id: m for m in models.models}
    assert by_id["qwen2-5-vl"].capabilities == []


def test_write_model_capabilities_does_not_disturb_the_other_model(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_NESTED_SWEEP, encoding="utf-8")
    write_model_capabilities(tmp_path, "qwen3-14b", ["reasoning"])
    _s, models, _issues = Config.load(tmp_path)
    by_id = {m.id: m for m in models.models}
    assert by_id["qwen2-5-vl"].capabilities == ["vision"]
    assert by_id["qwen2-5-vl"].name == "Qwen2.5 VL"


def test_write_model_capabilities_raises_for_unknown_model(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_NESTED_SWEEP, encoding="utf-8")
    original = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        write_model_capabilities(tmp_path, "no-such-model", ["chat"])
    assert (tmp_path / "models.yaml").read_text(encoding="utf-8") == original


def test_write_model_capabilities_atomic_leaves_no_temp_file(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_NESTED_SWEEP, encoding="utf-8")
    write_model_capabilities(tmp_path, "qwen3-14b", ["chat"])
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "models.yaml"]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# remove_model_entry - deletes a whole model block (owner request 2026-08-21:
# Models page Delete button, keyword: remove_model_entry).
# --------------------------------------------------------------------------- #

from config import remove_model_entry  # noqa: E402

# Three entries so a middle-row delete can be checked against BOTH neighbors:
# does X's own trailing blank survive, and does B end up separated from X by
# exactly one blank line (not zero, not two) once A is gone.
_THREE_MODELS = """version: 1
models:
  - id: model-x
    name: "Model X"
    location: "/locitize-test/models/x.gguf"
    context_size: 8192
    gpu_layers: -1
    status: installed

  - id: model-a
    name: "Model A"
    location: "/locitize-test/models/a.gguf"
    context_size: 8192
    gpu_layers: -1
    status: installed

  - id: model-b
    name: "Model B"
    location: "/locitize-test/models/b.gguf"
    context_size: 8192
    gpu_layers: -1
    status: installed
"""


def test_remove_model_entry_deletes_only_the_target_row(tmp_path):
    (tmp_path / "models.yaml").write_text(_THREE_MODELS, encoding="utf-8")
    remove_model_entry(tmp_path, "model-a")
    _s, models, _issues = Config.load(tmp_path)
    ids = [m.id for m in models.models]
    assert ids == ["model-x", "model-b"]


def test_remove_model_entry_leaves_a_clean_blank_line(tmp_path):
    """Deleting a middle row must not leave a double blank or no blank at all
    between its neighbors - _locate_model_block folds the trailing separator
    into the DELETED row's own range, so X's own separator (not A's) is what
    is left standing between X and B."""
    (tmp_path / "models.yaml").write_text(_THREE_MODELS, encoding="utf-8")
    remove_model_entry(tmp_path, "model-a")
    text = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    assert "\n\n  - id: model-b\n" in text
    assert "\n\n\n" not in text
    assert "model-a" not in text


def test_remove_model_entry_last_row_leaves_file_parseable(tmp_path):
    (tmp_path / "models.yaml").write_text(_THREE_MODELS, encoding="utf-8")
    remove_model_entry(tmp_path, "model-b")
    _s, models, issues = Config.load(tmp_path)
    errors = [i for i in issues if i.level == "ERROR"]
    assert errors == []
    assert [m.id for m in models.models] == ["model-x", "model-a"]


def test_remove_model_entry_only_row_leaves_an_honestly_empty_registry(tmp_path):
    only_one = """version: 1
models:
  - id: model-x
    name: "Model X"
    location: "/locitize-test/models/x.gguf"
    context_size: 8192
    gpu_layers: -1
    status: installed
"""
    (tmp_path / "models.yaml").write_text(only_one, encoding="utf-8")
    remove_model_entry(tmp_path, "model-x")
    _s, models, issues = Config.load(tmp_path)
    errors = [i for i in issues if i.level == "ERROR"]
    assert errors == []
    assert models.models == []


def test_remove_model_entry_raises_for_unknown_model(tmp_path):
    (tmp_path / "models.yaml").write_text(_THREE_MODELS, encoding="utf-8")
    original = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        remove_model_entry(tmp_path, "no-such-model")
    assert (tmp_path / "models.yaml").read_text(encoding="utf-8") == original


def test_remove_model_entry_preserves_the_untouched_rows_byte_for_byte(tmp_path):
    (tmp_path / "models.yaml").write_text(_THREE_MODELS, encoding="utf-8")
    remove_model_entry(tmp_path, "model-a")
    text = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    assert '  - id: model-x\n    name: "Model X"' in text
    assert '  - id: model-b\n    name: "Model B"' in text


def test_remove_model_entry_atomic_leaves_no_temp_file(tmp_path):
    (tmp_path / "models.yaml").write_text(_THREE_MODELS, encoding="utf-8")
    remove_model_entry(tmp_path, "model-a")
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "models.yaml"]
    assert leftovers == []


def test_fresh_data_root_seeds_then_accepts_a_model(tmp_path):
    """Regression (M17.6): the wizard runs before the launcher, so it must
    seed the data root itself before registering a model. This codifies the
    sequence that failed on a real first run - an unseeded root has no
    models.yaml to append to, so ensure_user_config MUST create it first."""
    import config

    data = tmp_path / "locitize-data"
    # An unseeded root: appending a model must be impossible (no models.yaml).
    fake = tmp_path / "m.gguf"
    fake.write_bytes(b"GGUF" + b"0" * 100)
    with pytest.raises(Exception):
        config.append_model_entry(data, "m", "M", str(fake))

    # Seed it the way the wizard now does, then the same append succeeds.
    actions = config.ensure_user_config(data, install_dir=config.BASE_DIR)
    assert (data / "settings.yaml").is_file()
    assert (data / "models.yaml").is_file()
    config.append_model_entry(data, "m", "M", str(fake))
    text = (data / "models.yaml").read_text(encoding="utf-8")
    assert "id: \"m\"" in text or "id: m" in text


# --------------------------------------------------------------------------- #
# reasoning - shape validation and the targeted write-back (owner request
# 2026-09-02, keywords: parse_reasoning, write_model_reasoning).
# --------------------------------------------------------------------------- #

from config import parse_reasoning, write_model_reasoning  # noqa: E402


def test_parse_reasoning_accepts_the_three_documented_keys():
    cleaned, problems = parse_reasoning(
        {"enabled": True, "effort": "low", "budget": 2048}, "m"
    )
    assert cleaned == {"enabled": True, "effort": "low", "budget": 2048}
    assert problems == []


def test_parse_reasoning_does_not_police_the_effort_vocabulary():
    """The accepted levels belong to the model's chat template, not to LOCITIZE.

    Measured 2026-09-02: Qwen3.8-27B-UD-IQ4_XS's template accepts only
    xhigh/medium/low, while llama-server's own --help advertises minimal/low/
    medium/high/xhigh. A hardcoded allowlist here would reject levels other
    templates require, so any non-empty string passes through.
    """
    cleaned, problems = parse_reasoning({"effort": "xhigh"}, "m")
    assert cleaned == {"effort": "xhigh"} and problems == []
    cleaned, problems = parse_reasoning({"effort": "some-future-level"}, "m")
    assert cleaned == {"effort": "some-future-level"} and problems == []


def test_parse_reasoning_reports_an_unknown_key_rather_than_dropping_it():
    """Unlike spec_config, a misspelled key here is REPORTED.

    A misspelled speculation knob only costs throughput; a misspelled reasoning
    knob silently leaves the model thinking at its template default when the
    owner believed they had turned it down.
    """
    cleaned, problems = parse_reasoning({"efort": "low"}, "m")
    assert cleaned is None
    assert len(problems) == 1 and "efort" in problems[0]


def test_parse_reasoning_rejects_bad_shapes_and_keeps_the_good_ones():
    cleaned, problems = parse_reasoning(
        {"enabled": "yes", "effort": "  ", "budget": -5}, "m"
    )
    assert cleaned is None
    assert len(problems) == 3
    cleaned, problems = parse_reasoning({"enabled": "yes", "budget": 512}, "m")
    assert cleaned == {"budget": 512} and len(problems) == 1


def test_parse_reasoning_allows_the_documented_budget_sentinels():
    """-1 is unrestricted and 0 is 'end thinking immediately' - both are legal."""
    assert parse_reasoning({"budget": -1}, "m")[0] == {"budget": -1}
    assert parse_reasoning({"budget": 0}, "m")[0] == {"budget": 0}


def test_parse_reasoning_treats_absent_and_empty_as_off():
    assert parse_reasoning(None, "m") == (None, [])
    assert parse_reasoning({}, "m") == (None, [])


def test_write_model_reasoning_inserts_after_status_when_absent(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_NESTED_SWEEP, encoding="utf-8")
    write_model_reasoning(tmp_path, "qwen3-14b", {"effort": "low", "budget": 2048})
    _s, models, _issues = Config.load(tmp_path)
    by_id = {m.id: m for m in models.models}
    assert by_id["qwen3-14b"].reasoning == {"effort": "low", "budget": 2048}
    # The nested benchmark_sweep and the sibling row survive untouched.
    assert by_id["qwen3-14b"].benchmark_sweep == {"gpu_layers": [30, 35, 999]}
    assert by_id["qwen2-5-vl"].reasoning is None


def test_write_model_reasoning_clears_with_an_empty_mapping(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_NESTED_SWEEP, encoding="utf-8")
    write_model_reasoning(tmp_path, "qwen3-14b", {"effort": "low"})
    write_model_reasoning(tmp_path, "qwen3-14b", {})
    _s, models, _issues = Config.load(tmp_path)
    by_id = {m.id: m for m in models.models}
    assert by_id["qwen3-14b"].reasoning is None


def test_write_model_reasoning_refuses_a_bad_value_before_touching_the_file(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_NESTED_SWEEP, encoding="utf-8")
    original = (tmp_path / "models.yaml").read_text(encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        write_model_reasoning(tmp_path, "qwen3-14b", {"budget": -5})
    with pytest.raises(ValueError):
        write_model_reasoning(tmp_path, "qwen3-14b", {"efort": "low"})
    assert (tmp_path / "models.yaml").read_text(encoding="utf-8") == original


def test_write_model_reasoning_raises_for_unknown_model(tmp_path):
    (tmp_path / "models.yaml").write_text(_MODELS_WITH_NESTED_SWEEP, encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        write_model_reasoning(tmp_path, "no-such-model", {"effort": "low"})


def test_loader_warns_and_drops_a_malformed_reasoning_block(tmp_path):
    """A bad row must degrade to llama-server's defaults with a WARNING, never
    break registry load."""
    (tmp_path / "models.yaml").write_text(
        _MODELS_WITH_NESTED_SWEEP.replace(
            "    status: installed\n    server_args: []\n    benchmark_sweep:",
            "    status: installed\n    reasoning: {budget: -5}\n"
            "    server_args: []\n    benchmark_sweep:",
        ),
        encoding="utf-8",
    )
    _s, models, issues = Config.load(tmp_path)
    by_id = {m.id: m for m in models.models}
    assert by_id["qwen3-14b"].reasoning is None
    assert any(
        i.level == "WARNING" and "reasoning.budget" in i.message for i in issues
    )


from config import parse_reasoning_choice  # noqa: E402


def test_parse_reasoning_choice_covers_the_documented_grammar():
    """One grammar shared by the menu suffix and the --reasoning flag."""
    assert parse_reasoning_choice("", "m") == (None, [])
    assert parse_reasoning_choice("off", "m")[0] == {"enabled": False}
    assert parse_reasoning_choice("on", "m")[0] == {"enabled": True}
    assert parse_reasoning_choice("low", "m")[0] == {"effort": "low"}
    assert parse_reasoning_choice("low/2048", "m")[0] == {
        "effort": "low",
        "budget": 2048,
    }
    # Budget alone leaves effort at the row/template value.
    assert parse_reasoning_choice("/0", "m")[0] == {"budget": 0}


def test_parse_reasoning_choice_reports_a_bad_budget_instead_of_guessing():
    picked, problems = parse_reasoning_choice("low/abc", "m")
    assert picked is None and len(problems) == 1
    assert "integer" in problems[0]


def test_parse_reasoning_choice_does_not_police_the_level():
    """Same reason as parse_reasoning: the vocabulary is the template's."""
    assert parse_reasoning_choice("xhigh", "m")[0] == {"effort": "xhigh"}
    assert parse_reasoning_choice("some-future-level", "m")[0] == {
        "effort": "some-future-level"
    }
