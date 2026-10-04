"""Setup installs plain PyPI sets with uv, and falls back to pip."""

from pathlib import Path

import setup_env


def _fake_python(tmp_path):
    exe = tmp_path / "python.exe"
    exe.write_text("", encoding="utf-8")
    return exe


def test_plain_sets_use_uv(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(setup_env, "_uv_exe", lambda: Path("uv.exe"))
    monkeypatch.setattr(setup_env, "_run_streamed", lambda argv, out, t: (calls.append(argv), (0, []))[1])
    result = setup_env.pip_install(_fake_python(tmp_path), ["open-webui==0.11.4"])
    assert result.ok
    assert calls[0][:3] == ["uv.exe", "pip", "install"]
    assert len(calls) == 1


def test_uv_failure_falls_back_to_pip(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(setup_env, "_uv_exe", lambda: Path("uv.exe"))

    def run(argv, out, timeout):
        calls.append(argv)
        return (1, ["boom"]) if argv[0] == "uv.exe" else (0, [])

    monkeypatch.setattr(setup_env, "_run_streamed", run)
    assert setup_env.pip_install(_fake_python(tmp_path), ["streamlit>=1.30"]).ok
    assert [c[0] for c in calls] == ["uv.exe", str(tmp_path / "python.exe")]
    assert calls[1][1:4] == ["-m", "pip", "install"]


def test_sets_with_their_own_index_flags_go_straight_to_pip(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(setup_env, "_uv_exe", lambda: Path("uv.exe"))
    monkeypatch.setattr(setup_env, "_run_streamed", lambda argv, out, t: (calls.append(argv), (0, []))[1])
    setup_env.pip_install(_fake_python(tmp_path), list(setup_env.PIP_SETS["kokoro_pip_cuda"]))
    assert len(calls) == 1 and calls[0][1:3] == ["-m", "pip"]


def test_no_uv_means_pip(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(setup_env, "_uv_exe", lambda: None)
    monkeypatch.setattr(setup_env, "_run_streamed", lambda argv, out, t: (calls.append(argv), (0, []))[1])
    assert setup_env.pip_install(_fake_python(tmp_path), ["PyYAML>=6.0"]).ok
    assert calls[0][1:3] == ["-m", "pip"]
