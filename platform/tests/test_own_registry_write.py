"""The on-disk notice ignores this app's own models.yaml saves."""

import os
import queue
from types import SimpleNamespace

import config
import gui_controller
from tests.test_config import _MODELS_WITH_COMMENTS


def _watcher(tmp_path):
    return SimpleNamespace(
        _settings=SimpleNamespace(data_dir=tmp_path), result_q=queue.Queue()
    )


_TICK = [1_000_000_000]


def _bump_mtime(path):
    # A fresh, distinct mtime per call: two quick writes can share one on disk.
    _TICK[0] += 10
    os.utime(path, (_TICK[0], _TICK[0]))


def test_own_write_is_recognised_and_an_outside_edit_is_not(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    assert not config.registry_is_own_write(path)

    config.write_model_fields(tmp_path, "qwen3-14b", gpu_layers=10, context_size=8192)
    assert config.registry_is_own_write(path)

    path.write_text(path.read_text(encoding="utf-8") + "# edited by hand\n", encoding="utf-8")
    assert not config.registry_is_own_write(path)


def test_watcher_stays_quiet_for_an_own_save_and_speaks_for_an_outside_edit(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    watcher = _watcher(tmp_path)
    check = gui_controller.GuiController._check_registry_mtime

    check(watcher)  # first look only records the mtime
    config.write_model_fields(tmp_path, "qwen3-14b", gpu_layers=10, context_size=8192)
    _bump_mtime(path)
    check(watcher)
    assert watcher.result_q.empty()

    path.write_text(path.read_text(encoding="utf-8") + "# edited by hand\n", encoding="utf-8")
    _bump_mtime(path)
    check(watcher)
    notice = watcher.result_q.get_nowait()
    assert notice.kind == "status_line"
    assert "changed on disk" in notice.payload["text"]


def test_an_outside_edit_just_before_an_own_save_is_still_reported(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    watcher = _watcher(tmp_path)
    check = gui_controller.GuiController._check_registry_mtime

    check(watcher)
    path.write_text(path.read_text(encoding="utf-8") + "# edited by hand\n", encoding="utf-8")
    config.write_model_fields(tmp_path, "qwen3-14b", gpu_layers=10, context_size=8192)
    _bump_mtime(path)
    check(watcher)

    assert watcher.result_q.get_nowait().kind == "status_line"
    # Reported once: the next look at the same file is quiet.
    _bump_mtime(path)
    check(watcher)
    assert watcher.result_q.empty()


def test_a_failed_save_does_not_claim_the_file(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    path.write_text(_MODELS_WITH_COMMENTS, encoding="utf-8")
    config.note_registry_baseline(path)

    def boom(*_a, **_k):
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(config, "_atomic_write", boom)
    try:
        config.write_model_fields(tmp_path, "qwen3-14b", gpu_layers=10, context_size=8192)
    except config.RegistryWriteError:
        pass
    assert config.registry_is_own_write(path)  # still the untouched baseline
