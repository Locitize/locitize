"""A deleted hardlink in the models folder is restored from the Models store."""

import os
from types import SimpleNamespace

import setup_env


def _row(model_id, location, mmproj=None, sha256=""):
    return SimpleNamespace(id=model_id, location=str(location), mmproj=mmproj, sha256=sha256)


def _setup(tmp_path):
    store = tmp_path / "AI" / "Models"
    models = tmp_path / "data" / "models"
    store.mkdir(parents=True)
    models.mkdir(parents=True)
    return store, models


def test_missing_link_is_restored_without_copying(tmp_path):
    store, models = _setup(tmp_path)
    (store / "A.gguf").write_bytes(b"weights")
    (store / "A.mmproj.gguf").write_bytes(b"proj")
    row = _row("a", models / "A.gguf", mmproj=str(models / "A.mmproj.gguf"))

    done = setup_env.relink_missing_model_files([row], store, models)

    assert done == [("a", str(models / "A.gguf")), ("a", str(models / "A.mmproj.gguf"))]
    assert os.stat(models / "A.gguf").st_ino == os.stat(store / "A.gguf").st_ino


def test_present_file_and_rows_outside_the_models_folder_are_left_alone(tmp_path):
    store, models = _setup(tmp_path)
    (store / "A.gguf").write_bytes(b"store copy")
    (models / "A.gguf").write_bytes(b"different file")
    elsewhere = tmp_path / "elsewhere" / "B.gguf"
    (store / "B.gguf").write_bytes(b"b")

    done = setup_env.relink_missing_model_files(
        [_row("a", models / "A.gguf"), _row("b", elsewhere)], store, models
    )

    assert done == []
    assert (models / "A.gguf").read_bytes() == b"different file"
    assert not elsewhere.exists()


def test_downloaded_rows_and_non_gguf_names_are_never_relinked(tmp_path):
    store, models = _setup(tmp_path)
    (store / "A.gguf").write_bytes(b"a")
    (store / "notes.txt").write_bytes(b"x")

    done = setup_env.relink_missing_model_files(
        [_row("a", models / "A.gguf", sha256="ab" * 32), _row("n", models / "notes.txt")],
        store,
        models,
    )

    assert done == []
    assert not (models / "A.gguf").exists()


def test_nothing_in_the_store_leaves_the_model_missing(tmp_path):
    store, models = _setup(tmp_path)
    assert setup_env.relink_missing_model_files([_row("a", models / "A.gguf")], store, models) == []
