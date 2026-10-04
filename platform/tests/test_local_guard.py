"""Requests from web pages and DNS-rebinding hosts are refused by local services."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from local_guard import foreign_request_reason


def test_local_callers_are_accepted():
    assert foreign_request_reason("127.0.0.1:8092", None, 8092) is None
    assert foreign_request_reason("localhost:8092", None, 8092) is None
    assert foreign_request_reason("127.0.0.1:8092", "http://localhost:8096", 8092) is None


def test_websites_and_rebinding_are_refused():
    assert foreign_request_reason("127.0.0.1:8092", "https://evil.example", 8092)
    assert foreign_request_reason("evil.example:8092", None, 8092)
    assert foreign_request_reason("127.0.0.1:1", None, 8092)
    assert foreign_request_reason(None, None, 8092)
    assert foreign_request_reason("127.0.0.1:8092", "null", 8092)


@pytest.mark.parametrize(
    "voice", ["../x", r"..\x", "//attacker/share/x", "D:x", "a b", "", "x" * 65]
)
def test_kokoro_voice_names_cannot_become_paths(voice):
    from kokoro_server import KokoroEngine

    engine = SimpleNamespace(_voices_dir=Path("voices"))
    with pytest.raises(ValueError):
        KokoroEngine.voice_path(engine, voice)


def test_kokoro_plain_voice_name_resolves_inside_voices_dir():
    from kokoro_server import KokoroEngine

    engine = SimpleNamespace(_voices_dir=Path("voices"))
    assert KokoroEngine.voice_path(engine, "am_michael") == Path("voices") / "am_michael.pt"
