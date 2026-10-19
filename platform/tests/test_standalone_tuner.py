"""Standalone source allowlist and isolated configuration/discovery boundaries."""
from pathlib import Path
import importlib.util
import zipfile

import config
import setup_env


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]


def test_archive_has_explicit_minimal_source_only(tmp_path):
    builder = load_module("standalone_builder", ROOT / "platform/scripts/build_standalone_tuner.py")
    archive = tmp_path / "helper.zip"
    result = builder.build(ROOT, archive)
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        assert len(names) == len(builder.PACKAGE_FILES) + len(builder.ROOT_FILES) + len(builder.MODULES) + 2
        assert package.read("assets/locitize.png") == (ROOT / "platform/assets/locitize_icon_1024.png").read_bytes()
        assert package.read("LICENSE") == (ROOT / "LICENSE").read_bytes()
        assert package.read("runtime/models.default.yaml") == b"version: 2\nmodels: []\n"
        assert all(not name.endswith((".exe", ".dll", ".gguf", ".env")) for name in names)
        assert "runtime/router.py" in names and "runtime/responses_tools.py" in names
    assert result["runtime_binary_bundled"] is False


def test_setup_registration_preserves_original_weights(monkeypatch, tmp_path):
    bootstrap = load_module("standalone_bootstrap", ROOT / "plugins/locitize-tuner/bootstrap.py")
    package = tmp_path / "package"
    runtime = package / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "models.default.yaml").write_text("version: 2\nmodels: []\n")
    (runtime / "settings.default.yaml").write_text("version: 2\n")
    model = tmp_path / "original.gguf"
    model.write_bytes(b"original model remains here")
    before = model.read_bytes()
    monkeypatch.setattr(setup_env, "find_local_models", lambda **kwargs:
        [{"path": str(model), "name": model.name, "size": model.stat().st_size}])
    monkeypatch.setenv("LOCITIZE_DATA_DIR", str(tmp_path / "irrelevant"))
    root = tmp_path / "private-data"
    first = bootstrap.prepare(root, package, skip_runtime=True)
    second = bootstrap.prepare(root, package, skip_runtime=True)
    assert first["registered_models"] == 1 and second["registered_models"] == 0
    assert model.read_bytes() == before
    _, rows, _ = config.Config.load(root)
    assert rows.models[0].location == str(model)
    assert not (root / "models" / model.name).exists()


def test_setup_pairs_vision_projector_in_place_and_keeps_explicit_choice(monkeypatch, tmp_path):
    bootstrap = load_module("standalone_bootstrap", ROOT / "plugins/locitize-tuner/bootstrap.py")
    package = tmp_path / "package"
    runtime = package / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "models.default.yaml").write_text("version: 2\nmodels: []\n")
    (runtime / "settings.default.yaml").write_text("version: 2\n")
    weights = tmp_path / "weights"
    weights.mkdir()
    model = weights / "Vision-Model-Q8_0.gguf"
    model.write_bytes(b"model")
    projector = weights / "Vision-Model-Q8_0.mmproj-F16.gguf"
    projector.write_bytes(b"projector")
    found = [{"path": str(model), "name": model.name, "size": 5}]
    monkeypatch.setattr(setup_env, "find_local_models", lambda **kwargs: found)
    monkeypatch.setattr(setup_env, "pair_mmproj", lambda path: "")
    root = tmp_path / "private-data"
    assert bootstrap.prepare(root, package, skip_runtime=True)["vision_projectors_paired"] == 0
    # A projector found on a later run is attached to the existing row, in place.
    monkeypatch.setattr(setup_env, "pair_mmproj", lambda path: str(projector))
    upgraded = bootstrap.prepare(root, package, skip_runtime=True)
    assert upgraded["registered_models"] == 0 and upgraded["vision_projectors_paired"] == 1
    _, rows, _ = config.Config.load(root)
    assert rows.models[0].mmproj.replace("/", "\\") == str(projector).replace("/", "\\")
    assert not (root / "models").exists()
    # An explicitly configured projector is never replaced.
    other = weights / "other.mmproj-F16.gguf"
    other.write_bytes(b"other")
    config.write_model_mmproj(root, rows.models[0].id, str(other))
    assert bootstrap.prepare(root, package, skip_runtime=True)["vision_projectors_paired"] == 0
    _, rows, _ = config.Config.load(root)
    assert rows.models[0].mmproj.replace("/", "\\") == str(other).replace("/", "\\")


def test_runtime_local_only_does_not_adopt_machine_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_env, "find_llama_server", lambda *args: "external-binary")
    monkeypatch.setattr(setup_env, "resolve_llama_release", lambda *args, **kwargs: (None, "offline"))
    result, server = setup_env.install_llama_cpp(tmp_path / "bin", False, local_only=True)
    assert not result.ok and server == ""


def test_default_root_is_outside_redirected_appdata(monkeypatch, tmp_path):
    bootstrap = load_module("standalone_bootstrap", ROOT / "plugins/locitize-tuner/bootstrap.py")
    monkeypatch.delenv("LOCITIZE_TUNER_DATA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "profile"))
    assert bootstrap.data_dir() == tmp_path / "profile" / ".locitize-tuner"
    monkeypatch.setenv("LOCITIZE_TUNER_DATA_DIR", str(tmp_path / "chosen"))
    assert bootstrap.data_dir() == (tmp_path / "chosen").resolve()


def test_codex_store_app_install_is_migrated_once_and_kept(monkeypatch, tmp_path):
    bootstrap = load_module("standalone_bootstrap", ROOT / "plugins/locitize-tuner/bootstrap.py")
    monkeypatch.delenv("LOCITIZE_TUNER_DATA_DIR", raising=False)
    local = tmp_path / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    legacy = local / "Packages" / "OpenAI.Codex_abc" / "LocalCache" / "Local" / "LOCITIZE-Tuner"
    (legacy / "environment").mkdir(parents=True)
    (legacy / "models.yaml").write_text("version: 2\nmodels: []\n")
    (legacy / "logs").mkdir()
    (legacy / "logs" / "benchmark.lock").write_text("pid")
    (legacy / "logs" / "llama.log").write_text("kept")
    root = tmp_path / "profile" / ".locitize-tuner"
    assert bootstrap.migrate_legacy(root) == legacy
    assert (root / "models.yaml").read_text() == "version: 2\nmodels: []\n"
    assert (root / "logs" / "llama.log").read_text() == "kept"
    assert not (root / "logs" / "benchmark.lock").exists()
    assert not root.with_name(".locitize-tuner.migrating").exists()
    assert (legacy / "models.yaml").exists()
    (legacy / "models.yaml").write_text("changed later")
    assert bootstrap.migrate_legacy(root) is None
    assert (root / "models.yaml").read_text() == "version: 2\nmodels: []\n"


def test_migration_skipped_without_install_or_with_explicit_root(monkeypatch, tmp_path):
    bootstrap = load_module("standalone_bootstrap", ROOT / "plugins/locitize-tuner/bootstrap.py")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.delenv("LOCITIZE_TUNER_DATA_DIR", raising=False)
    assert bootstrap.migrate_legacy(tmp_path / "root") is None
    legacy = tmp_path / "Local" / "LOCITIZE-Tuner"
    (legacy / "environment").mkdir(parents=True)
    (legacy / "models.yaml").write_text("models: []\n")
    monkeypatch.setenv("LOCITIZE_TUNER_DATA_DIR", str(tmp_path / "explicit"))
    assert bootstrap.migrate_legacy(tmp_path / "explicit") is None
    assert not (tmp_path / "explicit").exists()
