"""recycle.py: every hard-link name of a model file, and the Recycle Bin move."""
import os
import sys

import pytest

import recycle


@pytest.mark.skipif(sys.platform != "win32", reason="hard-link names are read through the Windows API")
def test_file_names_lists_every_hard_link_name_with_the_given_path_first(tmp_path):
    first = tmp_path / "a" / "model.gguf"
    second = tmp_path / "b" / "model.gguf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"GGUF")
    os.link(first, second)

    names = recycle.file_names(first)

    assert names[0] == str(first)
    assert len(names) == 2
    assert os.path.samefile(names[1], second)


def test_file_names_answers_with_the_given_path_when_the_api_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(recycle.sys, "platform", "linux")
    assert recycle.file_names(tmp_path / "model.gguf") == [str(tmp_path / "model.gguf")]


def test_file_names_answers_with_the_given_path_for_a_missing_file(tmp_path):
    missing = tmp_path / "gone.gguf"
    assert recycle.file_names(missing) == [str(missing)]


def test_the_real_move_refuses_off_windows_instead_of_deleting(tmp_path, monkeypatch):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"GGUF")
    monkeypatch.setattr(recycle.sys, "platform", "linux")
    with pytest.raises(OSError):
        recycle._real_send_to_recycle_bin([str(target)])
    assert target.exists()


def test_the_real_move_refuses_a_folder_before_anything_moves(tmp_path):
    keep = tmp_path / "model.gguf"
    keep.write_bytes(b"GGUF")
    folder = tmp_path / "folder.gguf"
    folder.mkdir()
    with pytest.raises(OSError):
        recycle._real_send_to_recycle_bin([str(keep), str(folder)])
    assert keep.exists() and folder.exists()
