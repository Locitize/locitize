"""Meaningful standalone archive leak, allowlist and metadata rejection checks."""
import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest
pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "platform/scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("defect", ["extra", "secret", "machine_path", "bad_manifest", "unpinned", "missing_license", "changed_license", "changed_icon"])
def test_rejects_bad_archive(monkeypatch, tmp_path, defect):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    archive = tmp_path / "valid.zip"
    load("build_standalone_tuner").build(ROOT, archive)
    checker = load("verify_standalone_tuner")
    assert checker.check_archive(archive)["compatibility_document_included"]
    with zipfile.ZipFile(archive) as source:
        files = {name: source.read(name) for name in source.namelist()}
    if defect == "extra":
        files["private.env"] = b"private"
    elif defect == "secret":
        files["README.md"] += b"\nsk-" + b"x" * 25
    elif defect == "machine_path":
        files["README.md"] += b"\nC:\\\\Users\\\\owner\\\\private"
    elif defect == "bad_manifest":
        files["plugin.json"] = b"{}"
    elif defect == "missing_license":
        del files["LICENSE"]
    elif defect == "changed_icon":
        files["assets/locitize.png"] += b"unexpected content"
    elif defect == "changed_license":
        files["LICENSE"] = b"MIT License without redistribution notice"
    else:
        files["requirements-helper.txt"] = b"mcp>=1\n"
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as target:
        for name, data in files.items():
            target.writestr(name, data)
    with pytest.raises(Exception):
        checker.check_archive(bad)
