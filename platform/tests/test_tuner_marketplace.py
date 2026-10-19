"""Catalog payload, portability and incomplete-package refusal regressions."""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]


def builder(monkeypatch):
    scripts = ROOT / "platform/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("tuner_marketplace", scripts / "build_tuner_marketplace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_movable_catalog_resolves_complete_payload(monkeypatch, tmp_path):
    output, archive = tmp_path / "market", tmp_path / "standalone.zip"
    builder(monkeypatch).build_marketplace(ROOT, output, archive)
    moved = tmp_path / "outside-checkout"
    shutil.move(str(output), moved)
    catalog = json.loads((moved / ".agents/plugins/marketplace.json").read_text())
    assert catalog["name"] == "locitize-local"
    entry = catalog["plugins"][0]
    assert entry["name"] == "locitize-tuner"
    package = moved / entry["source"]["path"]
    assert json.loads((package / "plugin.json").read_text())["version"] == "0.3.5"
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            assert (package / name).read_bytes() == source.read(name)
    assert (package / "runtime/local_tuning_mcp.py").is_file()
    assert (package / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    repository = json.loads((ROOT / ".agents/plugins/marketplace.json").read_text())
    assert repository["plugins"][0]["source"]["path"] == "./dist/locitize-tuner-marketplace-0.3.5-desktop/plugins/locitize-tuner"


def test_build_preserves_existing_marketplace(monkeypatch, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="empty marketplace"):
        builder(monkeypatch).build_marketplace(ROOT, destination, tmp_path / "archive.zip")
    assert marker.read_text() == "preserve"


def test_incomplete_package_refuses_checkout_fallback(tmp_path, monkeypatch):
    package = tmp_path / "plugin"
    package.mkdir()
    shutil.copy2(ROOT / "plugins/locitize-tuner/server.py", package / "server.py")
    # Even an available development runtime must not mask a missing payload.
    monkeypatch.setenv("LOCITIZE_INSTALL_DIR", str(ROOT))
    process = subprocess.run([sys.executable, str(package / "server.py")],
                             cwd=package, capture_output=True, text=True, timeout=10)
    assert process.returncode == 2
    assert process.stdout == ""
    assert "bundled runtime is missing" in process.stderr


def test_legacy_verifier_resolves_catalog_correction_payload(tmp_path):
    spec = importlib.util.spec_from_file_location("legacy_tuner_verifier",
        ROOT / "platform/scripts/verify_local_tuning_plugin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = tmp_path / ".agents/plugins/marketplace.json"
    catalog.parent.mkdir(parents=True)
    payload = "./dist/corrected-payload/plugins/locitize-tuner"
    catalog.write_text(json.dumps({"plugins": [{"name": "locitize-tuner",
        "source": {"source": "local", "path": payload}}]}))
    assert module.catalog_package(tmp_path) == (tmp_path / payload).resolve()
    current = json.loads((ROOT / ".agents/plugins/marketplace.json").read_text())
    assert module.PACKAGE == (ROOT / current["plugins"][0]["source"]["path"]).resolve()
    catalog.write_text(json.dumps({"plugins": [{"name": "locitize-tuner",
        "source": {"source": "local", "path": "../outside"}}]}))
    with pytest.raises(ValueError):
        module.catalog_package(tmp_path)


def test_fresh_checkout_defaults_build_the_owning_catalog_payload(monkeypatch, tmp_path):
    module = builder(monkeypatch)
    from build_standalone_tuner import MODULES, PACKAGE_FILES, ROOT_FILES
    fresh = tmp_path / "fresh-checkout"
    files = ([Path("platform") / (name + ".py") for name in MODULES] +
             [Path("plugins/locitize-tuner") / name for name in PACKAGE_FILES] +
             [Path(name) for name in ROOT_FILES] + [Path(".agents/plugins/marketplace.json")])
    for relative in files:
        target = fresh / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    destination, archive = module.default_paths(fresh)
    assert destination.name == "locitize-tuner-marketplace-0.3.5-desktop"
    assert archive.name == "locitize-tuner-standalone-0.3.5-desktop.zip"
    module.build_marketplace(fresh, destination, archive)
    catalog = json.loads((fresh / ".agents/plugins/marketplace.json").read_text())
    payload = fresh / catalog["plugins"][0]["source"]["path"]
    assert (payload / "runtime/local_tuning_mcp.py").is_file()
    assert (payload / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()


@pytest.mark.parametrize("path", ["../outside/plugins/locitize-tuner", "/outside/plugins/locitize-tuner",
    "./plugins/locitize-tuner", "./dist/invalid/elsewhere"])
def test_build_defaults_refuse_unsafe_catalog(monkeypatch, tmp_path, path):
    catalog = tmp_path / ".agents/plugins/marketplace.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({"name": "locitize-local", "plugins": [{"name": "locitize-tuner",
        "source": {"source": "local", "path": path}}]}))
    with pytest.raises(ValueError):
        builder(monkeypatch).default_paths(tmp_path)


def test_build_defaults_refuse_ambiguous_catalog(monkeypatch, tmp_path):
    catalog = tmp_path / ".agents/plugins/marketplace.json"
    catalog.parent.mkdir(parents=True)
    row = {"name": "locitize-tuner", "source": {"source": "local", "path": "./dist/a/plugins/locitize-tuner"}}
    catalog.write_text(json.dumps({"name": "locitize-local", "plugins": [row, row]}))
    with pytest.raises(ValueError):
        builder(monkeypatch).default_paths(tmp_path)


def test_existing_archive_never_overwritten(monkeypatch, tmp_path):
    archive = tmp_path / "frozen.zip"
    archive.write_bytes(b"immutable")
    with pytest.raises(ValueError, match="new archive"):
        builder(monkeypatch).build_marketplace(ROOT, tmp_path / "new-destination", archive)
    assert archive.read_bytes() == b"immutable"
    assert not (tmp_path / "new-destination").exists()
