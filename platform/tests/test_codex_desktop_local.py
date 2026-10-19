"""Desktop local mode edits exactly its keys in config.toml and restores them."""
import json
import tomllib

import pytest

import codex_desktop_local as local
import codex_picker

REALISTIC = """model = "gpt-6.1-sol"
model_reasoning_effort = "low"

[projects.'work-repo']
trust_level = "trusted"

[features]
apps = true
multi_agent = true

[model_providers.locitize]
name = "LOCITIZE (local)"
base_url = "http://127.0.0.1:8080/v1"
env_key = "LOCITIZE_CODEX_API_KEY"

[desktop]
enabled-reasoning-efforts = ["low", "medium"]
"browser-family-enabled:chrome" = false

[tui.model_availability_nux]
"gpt-6.1-sol" = 1
"""


@pytest.fixture
def home(tmp_path):
    (tmp_path / codex_picker.CATALOG_FILE).write_text(json.dumps({"managed_by": codex_picker.CATALOG_MARKER,
                                                                  "models": []}))
    return tmp_path


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_on_then_off_restores_the_file_byte_for_byte(home, newline):
    config = home / "config.toml"
    config.write_bytes(REALISTIC.replace("\n", newline).encode())
    before = config.read_bytes()
    state = home / "state.json"
    result = local.enable(home, state, 8093, "qwen-local")
    parsed = tomllib.loads(config.read_text())
    assert parsed["model"] == "qwen-local" and parsed["model_provider"] == codex_picker.PROVIDER
    assert parsed["model_catalog_json"].endswith(codex_picker.CATALOG_FILE)
    assert parsed["features"] == {"apps": False, "multi_agent": True}
    assert parsed["model_providers"][codex_picker.PROVIDER]["base_url"] == "http://127.0.0.1:8093/v1"
    assert parsed["model_providers"]["locitize"]["env_key"] == "LOCITIZE_CODEX_API_KEY"
    assert parsed["desktop"] == tomllib.loads(REALISTIC)["desktop"]
    assert (home / result["backup"]).read_bytes() == before
    assert local.disable(home, state)["left_alone_changed_elsewhere"] == []
    assert config.read_bytes() == before
    assert not state.exists()


def test_missing_keys_and_features_table_are_added_then_removed(home):
    config = home / "config.toml"
    original = '[desktop]\nappearanceTheme = "system"\n'
    config.write_text(original)
    local.enable(home, home / "s.json", 8093, "m")
    parsed = tomllib.loads(config.read_text())
    assert parsed["model"] == "m" and parsed["features"] == {"apps": False}
    local.disable(home, home / "s.json")
    assert config.read_text() == original


def test_edits_made_while_on_are_kept_and_user_changes_are_reported(home):
    config = home / "config.toml"
    config.write_text(REALISTIC)
    local.enable(home, home / "s.json", 8093, "qwen-local")
    text = config.read_text()
    # The desktop app adds a preference; the user picks another model by hand.
    text = text.replace('"browser-family-enabled:chrome" = false',
                        '"browser-family-enabled:chrome" = false\nnewPref = 1')
    text = text.replace('model = "qwen-local"', 'model = "hand-picked"')
    config.write_text(text)
    result = local.disable(home, home / "s.json")
    parsed = tomllib.loads(config.read_text())
    assert result["left_alone_changed_elsewhere"] == ["model"]
    assert parsed["model"] == "hand-picked" and parsed["desktop"]["newPref"] == 1
    assert "model_provider" not in parsed and parsed["features"]["apps"] is True
    assert codex_picker.PROVIDER not in parsed["model_providers"]


def test_refusals_never_write(home, tmp_path):
    config = home / "config.toml"
    config.write_text(REALISTIC)
    before = config.read_bytes()
    local.enable(home, home / "s.json", 8093, "m")
    with pytest.raises(ValueError, match="already on"):
        local.enable(home, home / "s.json", 8093, "m")
    local.disable(home, home / "s.json")
    assert local.disable(home, home / "s.json") == {"desktop_local_mode": "off", "changed": False}
    (home / codex_picker.CATALOG_FILE).unlink()
    with pytest.raises(ValueError, match="picker first"):
        local.enable(home, home / "s.json", 8093, "m")
    assert config.read_bytes() == before
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / codex_picker.CATALOG_FILE).write_text(json.dumps({"managed_by": codex_picker.CATALOG_MARKER}))
    (broken / "config.toml").write_text("model = \n")
    with pytest.raises(tomllib.TOMLDecodeError):
        local.enable(broken, broken / "s.json", 8093, "m")
    assert (broken / "config.toml").read_text() == "model = \n"
    assert not (broken / "s.json").exists()
