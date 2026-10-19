"""Provider retirement regression: keep supported local launches and history adapters."""
import json

import pytest

import harness_launch
from session_core.config import PROVIDER_OPTIONS
from session_core.providers.registry import PROVIDERS, get_provider
from session_launch import LOCAL_PROVIDERS, build_local_launch


def test_local_launch_catalog_and_installers_agree():
    assert LOCAL_PROVIDERS == ("codex", "opencode")
    assert harness_launch.HARNESSES == LOCAL_PROVIDERS
    assert set(harness_launch.INSTALL_FUNCTIONS) == set(LOCAL_PROVIDERS)


def test_surviving_history_adapters_remain_registered():
    expected = {"codex", "grok", "copilot", "gemini", "qwen", "opencode"}
    assert {provider.key for provider in PROVIDERS} == expected
    assert expected <= set(PROVIDER_OPTIONS)
    assert all(get_provider(key) is not None for key in expected)
    assert get_provider("retired-provider") is None


@pytest.mark.parametrize("provider", LOCAL_PROVIDERS)
def test_fresh_and_resume_local_launch_keep_scoped_routing(tmp_path, provider):
    for resume in ("", "session-42"):
        argv, env = build_local_launch(provider, str(tmp_path), "local-model", 8099,
                                       resume_id=resume, context_size=32768)
        assert argv[0] == provider
        if resume:
            assert resume in argv
        if provider == "codex":
            assert 'model_context_window=32768' in argv
            assert 'model_auto_compact_token_limit=26214' in argv
            assert any('http://127.0.0.1:8099/v1' in arg for arg in argv)
            assert env["LOCITIZE_CODEX_API_KEY"]
        else:
            config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
            assert config["provider"]["locitize"]["options"]["baseURL"] == "http://127.0.0.1:8099/v1"


def test_unregistered_local_provider_rejected_before_launch(tmp_path):
    with pytest.raises(ValueError, match="Codex.*OpenCode"):
        build_local_launch("retired-provider", str(tmp_path), "local-model", 8099)
