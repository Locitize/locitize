"""Public-upload staging remains separate, bounded, and reproducible."""
from pathlib import Path
import importlib.util
import json
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "platform/scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_upload_is_separate_and_has_one_plugin_root(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    destination = tmp_path / "stage"
    archive = tmp_path / "public.zip"
    report = load("build_tuner_public_upload").build_public_upload(ROOT, destination, archive)
    assert report["private_app_bindings_included"] is False
    assert (destination / "locitize-tuner/runtime/local_tuning_mcp.py").is_file()
    with zipfile.ZipFile(archive) as package:
        assert all(name.startswith("locitize-tuner/") for name in package.namelist())
        assert not any(name.endswith(".app.json") for name in package.namelist())
        manifest = json.loads(package.read("locitize-tuner/plugin.json"))
        assert manifest["name"] == "locitize-tuner"
        assert "apps" not in manifest
        assert "apps" not in manifest["extensions"]["com.openai"]
    checked = load("verify_standalone_tuner").check_public_archive(archive)
    assert checked["app_bindings_absent"] is True
    assert checked["submission_metadata_complete"] is False
    assert "interface.privacyPolicyURL" in checked["submission_metadata_gaps"]
