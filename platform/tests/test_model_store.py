"""Tests for the shared model store (setup_env, M22).

The contract: LOCITIZE creates the store when it is missing (with a README
explaining the folder to a human who finds it), leaves an existing store
byte-for-byte alone, honors the LOCITIZE_MODEL_STORE override, and the
discovery scan includes the store so anything dropped there is importable.
"""

from __future__ import annotations

import setup_env


def test_creates_store_with_readme(tmp_path, monkeypatch):
    store = tmp_path / "AI" / "Models"
    monkeypatch.setenv("LOCITIZE_MODEL_STORE", str(store))
    said = []
    result = setup_env.ensure_model_store(said.append)
    assert result.ok and not result.skipped
    assert store.is_dir()
    readme = (store / "README.txt").read_text(encoding="utf-8")
    assert "shared model store" in readme and "GGUF" in readme
    assert said, "creation should be narrated"


def test_existing_store_left_alone(tmp_path, monkeypatch):
    store = tmp_path / "Models"
    store.mkdir()
    marker = store / "my-notes.txt"
    marker.write_text("mine", encoding="utf-8")
    monkeypatch.setenv("LOCITIZE_MODEL_STORE", str(store))
    result = setup_env.ensure_model_store()
    assert result.ok and result.skipped
    assert marker.read_text(encoding="utf-8") == "mine"
    # No README injected into a folder the user already owns.
    assert not (store / "README.txt").exists()


def test_env_override_and_default(tmp_path, monkeypatch):
    import os as _os

    override = str(tmp_path / "Elsewhere" / "Store")
    monkeypatch.setenv("LOCITIZE_MODEL_STORE", override)
    assert str(setup_env.model_store_path()) == override
    monkeypatch.delenv("LOCITIZE_MODEL_STORE", raising=False)
    default = str(setup_env.model_store_path())
    assert default == setup_env.MODEL_STORE_DEFAULT
    # Environment-derived: system drive root, then the two name parts.
    assert default.endswith(_os.path.join("AI", "Models"))


def test_discovery_scans_the_store(tmp_path, monkeypatch):
    store = tmp_path / "Models"
    store.mkdir()
    big = store / "SomeModel.Q4_K_M.gguf"
    big.write_bytes(b"g" * (setup_env.DISCOVERY_MIN_BYTES + 1))
    monkeypatch.setenv("LOCITIZE_MODEL_STORE", str(store))
    roots = setup_env.discovery_roots()
    assert store in roots and roots[0] == store  # the store scans first
    found = setup_env.find_local_models(roots=[store])
    assert [f["name"] for f in found] == ["SomeModel.Q4_K_M.gguf"]
