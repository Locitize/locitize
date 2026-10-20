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


def _bump_mtime(path):
    st = os.stat(path)
    os.utime(path, (st.st_atime, st.st_mtime + 5))


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
