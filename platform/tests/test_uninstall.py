"""Tests for the safe uninstaller's planner (uninstall.py, M18.16).

The plan decides what gets deleted, so it is the part that must be exactly
right: keep-models must preserve the models and fine-tune folders in place,
nothing outside the install's own paths may ever appear in the delete list,
and missing pieces simply do not show up (the plan is what will really happen).
"""

from __future__ import annotations

import uninstall
from uninstall import uninstall_plan


def _install(tmp_path, with_models=True):
    """A fake install: repo venvs + a data root with typical children."""
    repo = tmp_path / "repo"
    (repo / ".venv").mkdir(parents=True)
    (repo / ".webui-venv").mkdir()
    (repo / "finetune-studio" / ".venv").mkdir(parents=True)
    root = tmp_path / "data"
    for child in ("bin", "logs", "reports", "webui-data", "memory"):
        (root / child).mkdir(parents=True)
    (root / "settings.yaml").write_text("x", encoding="utf-8")
    (root / "models.yaml").write_text("x", encoding="utf-8")
    if with_models:
        (root / "models").mkdir()
        (root / "models" / "m.gguf").write_bytes(b"g")
        (root / "finetune" / "outputs" / "run1").mkdir(parents=True)
    return repo, root


def test_keep_models_preserves_models_and_finetune(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    repo, root = _install(tmp_path)
    delete, keep = uninstall_plan(repo, root, keep_models=True)
    kept_names = {p.name for p in keep}
    assert kept_names == {"models", "finetune", "webui-data", "memory"}
    deleted_names = {p.name for p in delete}
    # The re-creatable pieces go; the sacred ones never appear in delete.
    assert {"bin", "logs", "settings.yaml", "models.yaml"} <= deleted_names
    assert "models" not in deleted_names and "finetune" not in deleted_names
    # All three venvs are planned.
    assert sum(1 for p in delete if p.name in (".venv", ".webui-venv")) == 3


def test_a_folder_that_is_not_a_locitize_data_root_is_never_deleted(tmp_path, monkeypatch):
    """LOCITIZE_DATA_DIR pointing at a drive root or Documents must not empty it."""
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(uninstall, "desktop_shortcut", lambda: None)
    repo, _root = _install(tmp_path)
    documents = tmp_path / "Documents"
    (documents / "taxes").mkdir(parents=True)
    for keep_models in (True, False):
        delete, keep = uninstall_plan(repo, documents, keep_models=keep_models)
        assert documents in keep
        assert not any(str(p).startswith(str(documents)) for p in delete)


def test_full_uninstall_takes_the_whole_root(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    repo, root = _install(tmp_path)
    delete, keep = uninstall_plan(repo, root, keep_models=False)
    assert keep == []
    assert root in delete  # one target: the root itself


def test_missing_pieces_do_not_appear(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    root = tmp_path / "nonexistent-data"
    # Isolate from the REAL machine: a genuine Desktop LOCITIZE.lnk (this dev
    # box has one) must not leak into the honest-empty-plan assertion.
    monkeypatch.setattr(uninstall, "desktop_shortcut", lambda: None)
    delete, keep = uninstall_plan(repo, root, keep_models=True)
    assert delete == [] and keep == []  # honest empty plan, nothing invented


def test_localappdata_fallback_included_once(tmp_path, monkeypatch):
    repo, root = _install(tmp_path)
    fallback = tmp_path / "lad" / "LOCITIZE"
    fallback.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    delete, _keep = uninstall_plan(repo, root, keep_models=True)
    assert fallback in delete
    # And when the data root IS the fallback, it is not listed twice.
    delete2, _ = uninstall_plan(repo, fallback, keep_models=False)
    assert delete2.count(fallback) == 1


def test_nothing_outside_install_paths(tmp_path, monkeypatch):
    """Every planned deletion lives under the repo, the data root, or the
    LOCALAPPDATA fallback - the guarantee that makes the uninstaller safe."""
    repo, root = _install(tmp_path)
    lad = tmp_path / "lad"
    (lad / "LOCITIZE").mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(lad))
    delete, _keep = uninstall_plan(repo, root, keep_models=True)
    allowed = (str(repo), str(root), str(lad / "LOCITIZE"))
    for target in delete:
        if target.name == "LOCITIZE.lnk":
            continue  # the Desktop shortcut is the one path outside the trio
        assert str(target).startswith(allowed), target


def test_execute_reports_failures_and_continues(tmp_path):
    good = tmp_path / "a"
    good.mkdir()
    missing_parent_ok = tmp_path / "gone.txt"  # unlink(missing_ok) succeeds
    failures = uninstall.execute([good, missing_parent_ok], say=lambda _t: None)
    assert failures == []
    assert not good.exists()


def test_keep_models_also_preserves_session_notes_in_both_data_roots(tmp_path, monkeypatch):
    repo, root = _install(tmp_path)
    (root / "sessions").mkdir()
    fallback = tmp_path / "local" / "LOCITIZE"
    (fallback / "sessions").mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(fallback.parent))
    delete, keep = uninstall_plan(repo, root)
    assert root / "sessions" in keep
    assert fallback / "sessions" in keep
    assert fallback not in delete
