"""Chat-UI chooser + Open WebUI spec + preference persistence tests (M9-lite, AC18).

Three headless suites, no browser, no Open WebUI install, no real process, no
network:

- keyword `chat_chooser`  : resolve_chat_choice across the full decision matrix
  (preferred x cli_override x model_running x webui_installed x webui_ready),
  asserting every ChatDecision and its reason string. The function is pure, so this
  runs with plain values and no fakes.
- keyword `openwebui_spec`: build_openwebui_spec asserts the dedicated .webui-venv
  console script (NOT sys.executable), the confirmed `serve` argv, loopback host,
  ports.openwebui, health_path, the DATA_DIR / OPENAI_API_BASE_URL(S) / offline env,
  ready_timeout_s=300, and the ".webui-venv missing -> ValueError with remedy" guard.
  The venv presence is faked with a temp Codebase tree, so no real install is needed.
- keyword `chat_persist` : config.write_chat_ui is the targeted atomic sibling of
  write_model_fields; it changes only the one scalar, preserves comments/other keys,
  and rejects an invalid value.

Every test name carries one of those keywords so `pytest -k "chat_chooser or
openwebui_spec"` selects exactly the AC18 headless proof. ASCII only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from config import Config, Settings, write_chat_ui, write_openwebui_enabled
from webui import (
    OPENWEBUI_HEALTH_PATH,
    OPENWEBUI_NAME,
    ChatDecision,
    build_openwebui_spec,
    resolve_chat_choice,
    webui_available,
)


# --------------------------------------------------------------------------- #
# resolve_chat_choice - the full decision matrix (keyword chat_chooser)
# --------------------------------------------------------------------------- #


def test_chat_chooser_no_model_short_circuits_regardless_of_preference():
    """No running model -> NO_MODEL for every preference (never open a dead port)."""
    for preferred in ("ask", "llamacpp", "openwebui"):
        for override in (None, "ask", "llamacpp", "openwebui"):
            result = resolve_chat_choice(
                preferred=preferred,
                cli_override=override,
                model_running=False,
                webui_installed=True,
                webui_ready=True,
            )
            assert result.decision is ChatDecision.NO_MODEL
            assert result.reason == "no model running - start one first"


def test_chat_chooser_llamacpp_opens_builtin():
    """preferred=llamacpp with a model running -> OPEN_LLAMACPP, no reason."""
    result = resolve_chat_choice("llamacpp", None, True, False, False)
    assert result.decision is ChatDecision.OPEN_LLAMACPP
    assert result.reason == ""


def test_chat_chooser_openwebui_ready_opens_openwebui():
    """preferred=openwebui, installed AND ready -> OPEN_OPENWEBUI."""
    result = resolve_chat_choice("openwebui", None, True, True, True)
    assert result.decision is ChatDecision.OPEN_OPENWEBUI
    assert result.reason == ""


def test_chat_chooser_openwebui_installed_not_ready_offers_start():
    """preferred=openwebui, installed but not running -> OFFER_START_OPENWEBUI."""
    result = resolve_chat_choice("openwebui", None, True, True, False)
    assert result.decision is ChatDecision.OFFER_START_OPENWEBUI
    assert result.reason == "Open WebUI is installed but not running"


def test_chat_chooser_openwebui_not_installed_degrades_with_remedy():
    """preferred=openwebui but not installed -> DEGRADE_TO_LLAMACPP with a remedy."""
    result = resolve_chat_choice("openwebui", None, True, False, False)
    assert result.decision is ChatDecision.DEGRADE_TO_LLAMACPP
    assert "not installed" in result.reason
    assert "docs/chat.md" in result.reason


def test_chat_chooser_ask_presents_choice_when_openwebui_installed():
    """preferred=ask with Open WebUI installed -> ASK (a real second choice exists)."""
    # installed and ready, or installed but not ready, both still ASK.
    for ready in (True, False):
        result = resolve_chat_choice("ask", None, True, True, ready)
        assert result.decision is ChatDecision.ASK
        assert result.reason == ""


def test_chat_chooser_ask_collapses_to_llamacpp_when_openwebui_absent():
    """preferred=ask but Open WebUI not installed -> OPEN_LLAMACPP (nothing to ask)."""
    result = resolve_chat_choice("ask", None, True, False, False)
    assert result.decision is ChatDecision.OPEN_LLAMACPP
    assert "not installed" in result.reason


def test_chat_chooser_cli_override_beats_persisted_preference():
    """--chat-ui overrides the persisted preference (both directions)."""
    # Persisted llamacpp, override openwebui -> the openwebui branch wins.
    result = resolve_chat_choice("llamacpp", "openwebui", True, True, True)
    assert result.decision is ChatDecision.OPEN_OPENWEBUI
    # Persisted openwebui, override llamacpp -> the llamacpp branch wins.
    result = resolve_chat_choice("openwebui", "llamacpp", True, True, True)
    assert result.decision is ChatDecision.OPEN_LLAMACPP


def test_chat_chooser_unknown_preference_is_treated_as_ask():
    """A stray persisted value falls back to ask, never crashes the chooser."""
    result = resolve_chat_choice("bogus", None, True, True, True)
    assert result.decision is ChatDecision.ASK
    # And an unknown override likewise degrades to ask.
    result = resolve_chat_choice("llamacpp", "nonsense", True, True, True)
    assert result.decision is ChatDecision.ASK


def test_chat_chooser_empty_override_uses_persisted_preference():
    """An empty/None override does NOT wipe the persisted preference."""
    result = resolve_chat_choice("llamacpp", None, True, True, True)
    assert result.decision is ChatDecision.OPEN_LLAMACPP
    result = resolve_chat_choice("llamacpp", "", True, True, True)
    assert result.decision is ChatDecision.OPEN_LLAMACPP


def test_chat_chooser_full_matrix_returns_a_valid_decision():
    """Exhaustive sweep: every combination yields exactly one known ChatDecision."""
    valid = set(ChatDecision)
    prefs = ("ask", "llamacpp", "openwebui")
    overrides = (None, "ask", "llamacpp", "openwebui")
    for preferred in prefs:
        for override in overrides:
            for model_running in (True, False):
                for installed in (True, False):
                    for ready in (True, False):
                        result = resolve_chat_choice(
                            preferred, override, model_running, installed, ready
                        )
                        assert result.decision in valid
                        # OPEN_OPENWEBUI only ever with a running model + installed + ready.
                        if result.decision is ChatDecision.OPEN_OPENWEBUI:
                            assert model_running and installed and ready


# --------------------------------------------------------------------------- #
# build_openwebui_spec - the data-driven service spec (keyword openwebui_spec)
# --------------------------------------------------------------------------- #


def _fake_codebase(tmp_path: Path, *, install: bool = True) -> Settings:
    """Build a Settings whose base_dir is a fake Codebase/platform tree.

    When install=True a fake .webui-venv/Scripts/open-webui.exe is created so
    build_openwebui_spec / webui_available see an "installed" Open WebUI without a
    real 2.6GB install. base_dir is <tmp>/Codebase/platform so venv resolution
    (base_dir.parent / .webui-venv) lands under <tmp>/Codebase.

    data_dir is a SEPARATE directory, deliberately (DEC-M14-9): the chat store
    belongs to the user's data root, and a fixture that let the two paths be
    equal could not tell the two apart - which is how the shipped defect
    (NEW-QA-M14-8, webui-data written into the install tree) went unnoticed.
    """
    platform_dir = tmp_path / "Codebase" / "platform"
    platform_dir.mkdir(parents=True)
    settings = Settings()
    object.__setattr__(settings, "base_dir", platform_dir)
    object.__setattr__(settings, "data_dir", tmp_path / "locitize-data")
    if install:
        scripts = tmp_path / "Codebase" / ".webui-venv" / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "open-webui.exe").write_bytes(b"fake console script")
    return settings


def test_openwebui_spec_uses_dedicated_venv_console_not_sys_executable(tmp_path):
    """argv[0] is the .webui-venv console script, never the platform interpreter."""
    settings = _fake_codebase(tmp_path)
    spec = build_openwebui_spec(settings, "x.log")
    assert spec.command[0].replace("\\", "/").endswith(
        "Codebase/.webui-venv/Scripts/open-webui.exe"
    )
    assert spec.command[0] != sys.executable


def test_openwebui_spec_serve_argv_and_loopback_host(tmp_path):
    """The confirmed `serve` argv binds 127.0.0.1 on ports.openwebui only."""
    settings = _fake_codebase(tmp_path)
    settings.ports.openwebui = 8096
    spec = build_openwebui_spec(settings, "x.log")
    assert spec.command[1:] == ["serve", "--host", "127.0.0.1", "--port", "8096"]
    assert spec.name == OPENWEBUI_NAME
    assert spec.port == 8096
    # Never a wildcard bind.
    assert "0.0.0.0" not in spec.command


def test_openwebui_spec_health_and_timeout(tmp_path):
    """health_path is /health and the readiness window is the 300s first-boot value."""
    settings = _fake_codebase(tmp_path)
    settings.openwebui.ready_timeout_s = 300
    spec = build_openwebui_spec(settings, "x.log")
    assert spec.health_path == OPENWEBUI_HEALTH_PATH == "/health"
    assert spec.ready_timeout_s == 300.0
    assert spec.append_log is True


def test_openwebui_spec_backend_and_data_env(tmp_path):
    """The env wires llama.cpp as the OpenAI backend and DATA_DIR under the data root."""
    settings = _fake_codebase(tmp_path)
    settings.ports.llama_cpp = 8080
    spec = build_openwebui_spec(settings, "x.log")
    assert spec.env["OPENAI_API_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert spec.env["OPENAI_API_BASE_URLS"] == "http://127.0.0.1:8080/v1"
    # A non-empty placeholder key (llama.cpp ignores it; Open WebUI requires it set).
    assert spec.env["OPENAI_API_KEY"]
    # Auth disabled for local single-user use (Tailscale Serve path).
    assert spec.env["WEBUI_AUTH"] == "False"
    assert spec.env["ENABLE_LOGIN_FORM"] == "False"
    # WEBUI_URL is the Serve ROOT host (no :4443), or loopback without Tailscale.
    assert spec.env["WEBUI_URL"].startswith(("https://", "http://127.0.0.1:"))
    assert not spec.env["WEBUI_URL"].endswith("/")
    assert ":4443" not in spec.env["WEBUI_URL"]
    assert spec.env["WEBUI_SESSION_COOKIE_SAME_SITE"] == "lax"
    assert spec.env["WEBUI_AUTH_COOKIE_SAME_SITE"] == "lax"
    assert spec.env["WEBUI_SESSION_COOKIE_SECURE"] == "True"
    assert spec.env["WEBUI_AUTH_COOKIE_SECURE"] == "True"
    # Only loopback (Tailscale Serve) may forward headers, and browsers may call
    # Open WebUI only from its own pages - never "*" while auth is off.
    assert spec.env["FORWARDED_ALLOW_IPS"] == "127.0.0.1"
    origins = spec.env["CORS_ALLOW_ORIGIN"].split(";")
    assert "*" not in origins
    assert spec.env["WEBUI_URL"] in origins
    assert spec.env["ENABLE_COMMUNITY_SHARING"] == "False"
    assert spec.env["ENABLE_VERSION_UPDATE_CHECK"] == "False"
    # DATA_DIR stays under the user's data root and never escapes it, and it is
    # NOT under the install tree (DEC-M14-9: it is the user's chat database).
    data_dir = Path(spec.env["DATA_DIR"]).resolve()
    assert Path(settings.data_dir).resolve() in data_dir.parents
    assert Path(settings.base_dir).resolve() not in data_dir.parents
    assert data_dir.name == "webui-data"


def test_openwebui_spec_disables_embedding_fetch_by_default(tmp_path):
    """disable_embedding_fetch=true (default) sets OFFLINE_MODE=true, no model download."""
    settings = _fake_codebase(tmp_path)
    assert settings.openwebui.disable_embedding_fetch is True
    spec = build_openwebui_spec(settings, "x.log")
    # OFFLINE_MODE must be the string "true" (Open WebUI lowercases and compares to
    # "true"; "1" would read as false and let the download proceed).
    assert spec.env["OFFLINE_MODE"] == "true"
    assert spec.env["HF_HUB_OFFLINE"] == "1"
    assert spec.env["RAG_EMBEDDING_MODEL_AUTO_UPDATE"] == "False"


def test_openwebui_spec_embedding_env_absent_when_fetch_enabled(tmp_path):
    """disable_embedding_fetch=false leaves the offline env off (owner-approved fetch)."""
    settings = _fake_codebase(tmp_path)
    settings.openwebui.disable_embedding_fetch = False
    spec = build_openwebui_spec(settings, "x.log")
    assert "OFFLINE_MODE" not in spec.env
    assert "HF_HUB_OFFLINE" not in spec.env


def test_openwebui_spec_custom_backend_url_used(tmp_path):
    """A configured loopback backend_base_url overrides the derived one."""
    settings = _fake_codebase(tmp_path)
    settings.openwebui.backend_base_url = "http://127.0.0.1:9001/v1"
    spec = build_openwebui_spec(settings, "x.log")
    assert spec.env["OPENAI_API_BASE_URL"] == "http://127.0.0.1:9001/v1"


def test_openwebui_spec_missing_venv_raises_with_remedy(tmp_path):
    """No .webui-venv console script -> ValueError carrying an install remedy."""
    settings = _fake_codebase(tmp_path, install=False)
    with pytest.raises(ValueError) as excinfo:
        build_openwebui_spec(settings, "x.log")
    message = str(excinfo.value)
    assert "not installed" in message
    assert "pip install open-webui" in message


def test_openwebui_spec_venv_escape_is_rejected(tmp_path):
    """A venv name that resolves outside Codebase/ is refused (path safety)."""
    settings = _fake_codebase(tmp_path)
    settings.openwebui.venv = "../../escape-venv"
    with pytest.raises(ValueError):
        build_openwebui_spec(settings, "x.log")


def test_openwebui_spec_webui_available_reflects_install_and_enabled(tmp_path):
    """webui_available is True only when enabled AND the console script exists."""
    installed = _fake_codebase(tmp_path)
    # Open WebUI is OFF by default (DEC-M14-1's demotion to opt-in), so being
    # installed is not enough on its own. Asserting that first, then opting in
    # explicitly, keeps this test about the AND condition it is named for rather
    # than about whichever way the default happens to point.
    assert webui_available(installed) is False
    installed.openwebui.enabled = True
    assert webui_available(installed) is True
    installed.openwebui.enabled = False
    assert webui_available(installed) is False

    missing = _fake_codebase(tmp_path / "other", install=False)
    assert webui_available(missing) is False


# --------------------------------------------------------------------------- #
# config.write_chat_ui - targeted atomic preference write (keyword chat_persist)
# --------------------------------------------------------------------------- #


_SETTINGS_WITH_CHAT = """\
version: 1

ports:
  llama_cpp: 8080   # keep this comment
  openwebui: 8096

chat:
  # the chooser default (a comment that must survive the write)
  preferred_ui: ask

launcher:
  default_model: qwen3-14b
"""


def _models_yaml(base: Path) -> None:
    (base / "models.yaml").write_text(
        "version: 1\nmodels:\n  - id: m\n    name: M\n    location: \"\"\n"
        "    context_size: 4096\n    gpu_layers: -1\n    status: installed\n",
        encoding="utf-8",
    )


def test_chat_persist_updates_only_the_one_scalar(tmp_path):
    """write_chat_ui changes preferred_ui only; comments and other keys are preserved."""
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_CHAT, encoding="utf-8")
    write_chat_ui(tmp_path, "openwebui")
    text = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    # The scalar changed.
    assert "preferred_ui: openwebui" in text
    assert "preferred_ui: ask" not in text
    # Every comment and unrelated key survived byte-for-byte.
    assert "# keep this comment" in text
    assert "# the chooser default (a comment that must survive the write)" in text
    assert "llama_cpp: 8080" in text
    assert "default_model: qwen3-14b" in text


def test_chat_persist_round_trips_through_the_loader(tmp_path):
    """After write_chat_ui the loader reads back the new preference."""
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_CHAT, encoding="utf-8")
    _models_yaml(tmp_path)
    write_chat_ui(tmp_path, "llamacpp")
    settings, _models, _issues = Config.load(tmp_path)
    assert settings.chat.preferred_ui == "llamacpp"


def test_chat_persist_rejects_invalid_value(tmp_path):
    """An invalid UI value is refused before the file is touched."""
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_CHAT, encoding="utf-8")
    original = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        write_chat_ui(tmp_path, "not-a-ui")
    # File left untouched.
    assert (tmp_path / "settings.yaml").read_text(encoding="utf-8") == original


def test_chat_persist_missing_section_raises(tmp_path):
    """No chat: section -> refuse to guess where to write (raise, do not corrupt)."""
    (tmp_path / "settings.yaml").write_text(
        "version: 1\nports:\n  llama_cpp: 8080\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        write_chat_ui(tmp_path, "openwebui")


def test_chat_persist_is_atomic_no_temp_left_behind(tmp_path):
    """A successful write leaves no *.tmp sibling (atomic replace cleaned up)."""
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_CHAT, encoding="utf-8")
    write_chat_ui(tmp_path, "openwebui")
    leftovers = list(tmp_path.glob("settings.yaml.*"))
    assert leftovers == []


# --------------------------------------------------------------------------- #
# config.write_openwebui_enabled - targeted atomic feature-flag write (M17.11)
# --------------------------------------------------------------------------- #

_SETTINGS_WITH_OPENWEBUI = """\
version: 1

openwebui:
  # opt-in, ships off (a comment that must survive the write)
  enabled: false
  data_dir: webui-data

chat:
  preferred_ui: ask
"""


def test_openwebui_enable_flips_only_that_flag(tmp_path):
    """Installing the venv is not enough; selecting the feature must enable it.
    The write flips openwebui.enabled and preserves comments and other keys."""
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_OPENWEBUI, encoding="utf-8")
    write_openwebui_enabled(tmp_path, True)
    text = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert "enabled: true" in text
    assert "enabled: false" not in text
    assert "opt-in, ships off (a comment that must survive the write)" in text
    assert "preferred_ui: ask" in text  # unrelated section untouched


def test_openwebui_enable_round_trips_through_loader(tmp_path):
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_OPENWEBUI, encoding="utf-8")
    _models_yaml(tmp_path)
    write_openwebui_enabled(tmp_path, True)
    settings, _m, _i = Config.load(tmp_path)
    assert settings.openwebui.enabled is True


def test_openwebui_enable_can_also_disable(tmp_path):
    on = _SETTINGS_WITH_OPENWEBUI.replace("enabled: false", "enabled: true")
    (tmp_path / "settings.yaml").write_text(on, encoding="utf-8")
    write_openwebui_enabled(tmp_path, False)
    assert "enabled: false" in (tmp_path / "settings.yaml").read_text(encoding="utf-8")


def test_openwebui_enable_missing_section_raises(tmp_path):
    (tmp_path / "settings.yaml").write_text(
        "version: 1\nports:\n  llama_cpp: 8080\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        write_openwebui_enabled(tmp_path, True)


def test_openwebui_enable_is_atomic_no_temp_left(tmp_path):
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_OPENWEBUI, encoding="utf-8")
    write_openwebui_enabled(tmp_path, True)
    assert list(tmp_path.glob("settings.yaml.*")) == []


# --------------------------------------------------------------------------- #
# Persisted backend reconciliation (owner request 2026-09-02).
#
# Defect found live: with the router enabled, Open WebUI still showed "No models
# available". OPENAI_API_BASE_URLS is a SEED DEFAULT only - on an initialized
# DATA_DIR the row in webui.db wins, exactly as this module already documented
# for the web-search flags. The persisted row still read
# ["http://127.0.0.1:8080/v1"], so the picker was asking a llama-server that was
# not running.
# --------------------------------------------------------------------------- #

from webui import (  # noqa: E402
    OPENAI_PLACEHOLDER_KEY,
    plan_backend_urls,
    reconcile_persisted_backend,
)

_ROUTER_URL = "http://127.0.0.1:8093/v1"
_PORTS = [8080, 8093]


def test_a_stale_locitize_url_is_migrated_to_the_intended_one():
    """The exact defect: a row written when the backend was llama-server."""
    assert plan_backend_urls(
        ["http://127.0.0.1:8080/v1"], _ROUTER_URL, _PORTS
    ) == [_ROUTER_URL]


def test_the_migration_runs_in_both_directions():
    """Turning the router back OFF must return the row to llama-server."""
    assert plan_backend_urls([_ROUTER_URL], "http://127.0.0.1:8080/v1", _PORTS) == [
        "http://127.0.0.1:8080/v1"
    ]


def test_a_foreign_provider_is_never_touched():
    """A URL the owner added for their own provider is not ours to rewrite."""
    planned = plan_backend_urls(
        ["https://api.openai.com/v1"], _ROUTER_URL, _PORTS
    )
    assert planned == ["https://api.openai.com/v1", _ROUTER_URL]


def test_ours_is_appended_not_substituted_when_nothing_matched():
    """Enabling the router on an install pointing elsewhere adds a choice
    rather than hijacking one."""
    planned = plan_backend_urls(
        ["https://api.example.com/v1", "http://192.168.1.5:1234/v1"],
        _ROUTER_URL,
        _PORTS,
    )
    assert planned[-1] == _ROUTER_URL
    assert planned[:2] == ["https://api.example.com/v1", "http://192.168.1.5:1234/v1"]


def test_a_url_on_a_non_locitize_loopback_port_is_left_alone():
    """localhost:11434 is Ollama, not ours. Only OUR reserved ports migrate."""
    planned = plan_backend_urls(
        ["http://localhost:11434/v1"], _ROUTER_URL, _PORTS
    )
    assert "http://localhost:11434/v1" in planned


def test_localhost_spelling_of_our_own_port_is_recognised_as_ours():
    assert plan_backend_urls(
        ["http://localhost:8080/v1"], _ROUTER_URL, _PORTS
    ) == [_ROUTER_URL]


def test_an_already_correct_list_is_unchanged():
    assert plan_backend_urls([_ROUTER_URL], _ROUTER_URL, _PORTS) == [_ROUTER_URL]


def test_duplicates_collapse_and_empty_entries_drop():
    planned = plan_backend_urls(
        ["http://127.0.0.1:8080/v1", "  ", _ROUTER_URL], _ROUTER_URL, _PORTS
    )
    assert planned == [_ROUTER_URL]


def test_an_empty_list_gets_the_intended_url():
    assert plan_backend_urls([], _ROUTER_URL, _PORTS) == [_ROUTER_URL]


def _webui_db(tmp_path, base_urls, api_keys=None):
    """A minimal webui.db carrying just the config rows this reads."""
    import json
    import sqlite3

    data_dir = tmp_path / "webui-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(data_dir / "webui.db"))
    conn.execute(
        "create table config (key text primary key, value text, updated_at integer)"
    )
    conn.execute(
        "insert into config values (?, ?, ?)",
        ("openai.api_base_urls", json.dumps(base_urls), 0),
    )
    if api_keys is not None:
        conn.execute(
            "insert into config values (?, ?, ?)",
            ("openai.api_keys", json.dumps(api_keys), 0),
        )
    conn.commit()
    conn.close()
    return data_dir


def _settings_with_router(tmp_path, enabled=True):
    settings = Settings()
    settings.data_dir = str(tmp_path)
    settings.router.enabled = enabled
    return settings


def test_reconcile_rewrites_the_persisted_row(tmp_path):
    import json
    import sqlite3

    _webui_db(tmp_path, ["http://127.0.0.1:8080/v1"], [OPENAI_PLACEHOLDER_KEY])
    settings = _settings_with_router(tmp_path)
    message = reconcile_persisted_backend(settings)
    assert str(settings.ports.router) in message

    conn = sqlite3.connect(str(tmp_path / "webui-data" / "webui.db"))
    value = conn.execute(
        "select value from config where key = 'openai.api_base_urls'"
    ).fetchone()[0]
    conn.close()
    assert json.loads(value) == [f"http://127.0.0.1:{settings.ports.router}/v1"]


def test_reconcile_pads_api_keys_to_match_the_url_count(tmp_path):
    """api_keys pairs with api_base_urls BY INDEX; a short list leaves later
    connections keyless and Open WebUI drops them from the picker."""
    import json
    import sqlite3

    _webui_db(tmp_path, ["https://api.example.com/v1"], ["sk-owner-key"])
    settings = _settings_with_router(tmp_path)
    reconcile_persisted_backend(settings)

    conn = sqlite3.connect(str(tmp_path / "webui-data" / "webui.db"))
    urls = json.loads(
        conn.execute(
            "select value from config where key = 'openai.api_base_urls'"
        ).fetchone()[0]
    )
    keys = json.loads(
        conn.execute(
            "select value from config where key = 'openai.api_keys'"
        ).fetchone()[0]
    )
    conn.close()
    assert len(keys) == len(urls)
    assert keys[0] == "sk-owner-key"  # the owner's key kept, in place


def test_reconcile_is_idempotent(tmp_path):
    _webui_db(tmp_path, ["http://127.0.0.1:8080/v1"])
    settings = _settings_with_router(tmp_path)
    reconcile_persisted_backend(settings)
    second = reconcile_persisted_backend(settings)
    assert "already points at" in second


def test_reconcile_reports_rather_than_raising_with_no_database(tmp_path):
    """A first run has no database; the env seed applies and needs no help."""
    settings = _settings_with_router(tmp_path)
    assert "no database yet" in reconcile_persisted_backend(settings)


def test_reconcile_degrades_honestly_on_a_broken_database(tmp_path):
    """Failing to reconfigure Open WebUI must be a message, never a crash that
    stops a chat UI from starting."""
    data_dir = tmp_path / "webui-data"
    data_dir.mkdir(parents=True)
    (data_dir / "webui.db").write_bytes(b"this is not a sqlite database")
    settings = _settings_with_router(tmp_path)
    message = reconcile_persisted_backend(settings)
    assert "could not update" in message


# --------------------------------------------------------------------------- #
# Audio config reconciliation (owner request 2026-09-03).
#
# The audio ENGINE cannot be set by env at all in this build, so without this
# the microphone button falls through to Open WebUI's own faster-whisper -
# which the OFFLINE_MODE this module sets deliberately stops from downloading a
# model. The button then does nothing, with no explanation.
# --------------------------------------------------------------------------- #

from webui import audio_config_rows, reconcile_persisted_audio  # noqa: E402


def test_audio_points_at_the_router_not_the_browser():
    """Open WebUI's other option is the browser's speech engine, which ships
    every recorded utterance to Google - a strange thing to do inside a
    local-first platform."""
    settings = _settings_with_router(Path("."))
    rows = audio_config_rows(settings)
    assert rows["audio.stt.engine"] == "openai"
    assert rows["audio.tts.engine"] == "openai"
    for key in ("audio.stt.openai.api_base_url", "audio.tts.openai.api_base_url"):
        assert rows[key] == f"http://127.0.0.1:{settings.ports.router}/v1"


def test_the_seeded_voice_is_one_kokoro_actually_has():
    """Open WebUI ships 'alloy', which Kokoro rejects."""
    settings = _settings_with_router(Path("."))
    settings.tts.voice = "af_bella"
    assert audio_config_rows(settings)["audio.tts.voice"] == "af_bella"


def test_the_api_keys_are_non_empty_placeholders():
    """Open WebUI refuses to call an endpoint with an empty key; the router does
    not check it."""
    rows = audio_config_rows(_settings_with_router(Path(".")))
    assert rows["audio.stt.openai.api_key"] == OPENAI_PLACEHOLDER_KEY
    assert rows["audio.tts.openai.api_key"] == OPENAI_PLACEHOLDER_KEY
    assert "audio.tts.api_key" not in rows


def _audio_db(tmp_path):
    import sqlite3

    data_dir = tmp_path / "webui-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(data_dir / "webui.db"))
    conn.execute(
        "create table config (key text primary key, value text, updated_at integer)"
    )
    conn.execute("insert into config values ('audio.stt.engine', '\"\"', 0)")
    conn.commit()
    conn.close()
    return data_dir


def test_reconcile_writes_the_audio_rows(tmp_path):
    import json
    import sqlite3

    _audio_db(tmp_path)
    settings = _settings_with_router(tmp_path)
    message = reconcile_persisted_audio(settings)
    assert "LOCITIZE speech services" in message

    conn = sqlite3.connect(str(tmp_path / "webui-data" / "webui.db"))
    got = dict(conn.execute("select key, value from config"))
    conn.close()
    assert json.loads(got["audio.stt.engine"]) == "openai"
    assert json.loads(got["audio.tts.engine"]) == "openai"
    assert json.loads(got["audio.tts.openai.api_key"]) == OPENAI_PLACEHOLDER_KEY


def test_audio_reconcile_is_idempotent(tmp_path):
    _audio_db(tmp_path)
    settings = _settings_with_router(tmp_path)
    reconcile_persisted_audio(settings)
    assert "already points at" in reconcile_persisted_audio(settings)


def test_audio_disabled_leaves_open_webui_alone(tmp_path):
    """An owner who turned the audio routes off must not get Open WebUI pointed
    at a 503."""
    _audio_db(tmp_path)
    settings = _settings_with_router(tmp_path)
    settings.router.audio = False
    assert "left alone" in reconcile_persisted_audio(settings)


def test_router_disabled_leaves_open_webui_alone(tmp_path):
    _audio_db(tmp_path)
    settings = _settings_with_router(tmp_path, enabled=False)
    assert "left alone" in reconcile_persisted_audio(settings)


def test_audio_reconcile_degrades_honestly_on_a_broken_database(tmp_path):
    """A chat UI that cannot do voice is worth starting anyway."""
    data_dir = tmp_path / "webui-data"
    data_dir.mkdir(parents=True)
    (data_dir / "webui.db").write_bytes(b"not a sqlite database")
    message = reconcile_persisted_audio(_settings_with_router(tmp_path))
    assert "could not update" in message


# --------------------------------------------------------------------------- #
# Voice call defaults (owner request 2026-09-03, "talk to my models like
# ChatGPT"): reliable phone turn-taking, plus the Call overlay's end-of-speech
# wait under the owner's control instead of a literal in the compiled bundle.
# --------------------------------------------------------------------------- #

from config import (  # noqa: E402
    CALL_SILENCE_UPSTREAM_MS,
    write_openwebui_call_silence,
    write_tts_voice,
)
from webui import (  # noqa: E402
    find_call_audio_context_chunk,
    find_call_audio_chunk,
    find_call_barge_in_chunk,
    find_call_silence_chunk,
    frontend_build_dir,
    patch_call_audio_context,
    patch_call_audio_playback,
    patch_call_barge_in,
    patch_call_silence,
    reconcile_call_audio_context,
    reconcile_call_audio_playback,
    reconcile_call_barge_in,
    reconcile_call_silence,
    reconcile_frontend_version,
    reconcile_voice_interruption,
)

# The closure as open-webui 0.11.1 ships it (CallOverlay.svelte compiled), with
# the interruption check ~150 bytes before the silence literal. Identifiers are
# minifier output and change per release; the shape is what the patch keys on.
_BUNDLE_CLOSURE = (
    'te=ve=>{const de=new AudioContext,ze=de.createMediaStreamSource(ve),'
    'oe=de.createAnalyser();analyser.getByteFrequencyData(Xe),c(T,we(gt)),'
    '(r(V)||r(W)&&!(((ut=s())==null'
    '?void 0:ut.voiceInterruption)??!1))&&c(T,0),Xe.some(xt=>xt>0)&&(y&&y.state'
    '!=="recording"&&y.start(),M||(M=!0,le()),Ct=Date.now()),M&&Date.now()-Ct>2e3'
    '&&(U=!0,y)){y.stop();return}window.requestAnimationFrame(xe)}'
)
_BUNDLE_AUDIO = (
    'Ee=ve=>i()?new Promise(de=>{var Xe,gt;const ze=document.getElementById('
    '"audioElement");if(!ze){de(null);return}let oe=!1;const We=async(Ct=null)=>'
    '{oe||(oe=!0,ze.onended=null,ze.onerror=null,ze.onpause=null,await new Promise('
    'Jt=>setTimeout(Jt,100)),de(Ct))};ze.src=ve.src,ze.muted=!0,ze.playbackRate='
    '((gt=(Xe=s().audio)==null?void 0:Xe.tts)==null?void 0:gt.playbackRate)??1,'
    'ze.onended=We,ze.onerror=()=>We(),ze.onpause=We,ze.play().then(()=>{ze.muted='
    '!1}).catch(Ct=>{We(Ct)})}):Promise.resolve(),le=async()=>{c(W,!1),D&&k()(),'
    '_e&&(speechSynthesis.cancel(),_e=null);const ve=document.getElementById('
    '"audioElement");ve&&(ve.muted=!0,ve.pause(),ve.currentTime=0)}'
)
_BUNDLE = "const a=1;" + _BUNDLE_CLOSURE + ";" + _BUNDLE_AUDIO + ";setTimeout(f,2e3);"


def test_patch_rewrites_only_the_call_overlay_literal():
    new, previous = patch_call_silence(_BUNDLE, 1000)
    assert previous == 2000
    assert "Date.now()-Ct>1000&&(U=!0,y)" in new
    # the unrelated 2e3 later in the file is not touched
    assert new.endswith(";setTimeout(f,2e3);")
    assert len(new) == len(_BUNDLE) + 1


def test_patch_is_symmetric_and_reports_already_right():
    patched, _ = patch_call_silence(_BUNDLE, 1000)
    again, previous = patch_call_silence(patched, 1000)
    assert again == patched and previous == 1000
    restored, previous = patch_call_silence(patched, CALL_SILENCE_UPSTREAM_MS)
    assert previous == 1000
    assert "Date.now()-Ct>2000&&(U=!0,y)" in restored


def test_patch_leaves_an_unknown_bundle_alone():
    """A future Open WebUI that restructured the overlay is reported, not guessed at."""
    text = "voiceInterruption" + "x" * 700 + "Date.now()-Ct>2e3&&(U=!0,y)"
    assert patch_call_silence(text, 1000) == (text, None)
    assert patch_call_silence("Date.now()-Ct>2e3&&(U=!0,y)", 1000)[1] is None


def test_barge_in_guard_can_be_added_for_recovery_compatibility():
    patched, previous = patch_call_barge_in(_BUNDLE, enabled=True)
    assert previous is False
    assert (
        'Xe.some(xt=>xt>0)&&(!r(W)||speechSynthesis.speaking||'
        '!document.getElementById("audioElement")?.paused)&&('
    ) in patched


def test_call_audio_context_is_closed_before_every_recorder_rearm():
    patched, previous = patch_call_audio_context(_BUNDLE)
    assert previous is False
    assert "te=async ve=>" in patched
    assert "await globalThis.__locitizeCallAudioContext?.close()" in patched
    assert (
        "const de=globalThis.__locitizeCallAudioContext=new AudioContext"
        in patched
    )


def test_call_audio_context_patch_is_idempotent():
    patched, _ = patch_call_audio_context(_BUNDLE)
    again, previous = patch_call_audio_context(patched)
    assert previous is True
    assert again == patched


def test_call_audio_context_leaves_an_unknown_bundle_alone():
    text = "voiceInterruption;const unrelated=new AudioContext"
    assert patch_call_audio_context(text) == (text, None)


def test_barge_in_guard_is_removed_for_reliable_phone_audio():
    patched, _ = patch_call_barge_in(_BUNDLE, enabled=True)
    again, previous = patch_call_barge_in(patched, enabled=False)
    assert previous is True
    assert again == _BUNDLE
    unchanged, previous = patch_call_barge_in(again, enabled=False)
    assert previous is False
    assert unchanged == _BUNDLE


def test_barge_in_patch_leaves_an_unknown_bundle_alone():
    text = "voiceInterruption;const unrelated=1"
    assert patch_call_barge_in(text) == (text, None)


def test_mobile_call_audio_uses_the_fetched_clip_and_retries_on_a_gesture():
    patched, previous = patch_call_audio_playback(_BUNDLE)
    assert previous is False
    assert "const ze=ve;if(!ze)" in patched
    assert "ze.src=ve.src" not in patched
    assert "ze.play().then" not in patched
    assert "globalThis.__locitizeCallAudio=ze" in patched
    assert 'document.addEventListener("pointerdown",locitizeRetry' in patched
    assert "const ve=globalThis.__locitizeCallAudio??" in patched
    assert "globalThis.__locitizeStopCallAudio?.()" in patched


def test_mobile_call_audio_patch_is_idempotent():
    patched, _ = patch_call_audio_playback(_BUNDLE)
    again, previous = patch_call_audio_playback(patched)
    assert previous is True
    assert again == patched


def test_mobile_call_audio_leaves_an_unknown_bundle_alone():
    text = 'voiceInterruption;document.getElementById("audioElement")'
    assert patch_call_audio_playback(text) == (text, None)


def _fake_frontend(tmp_path, bundle=_BUNDLE):
    settings = _fake_codebase(tmp_path)
    chunks = (
        tmp_path / "Codebase" / ".webui-venv" / "Lib" / "site-packages"
        / "open_webui" / "frontend" / "_app" / "immutable" / "chunks"
    )
    chunks.mkdir(parents=True)
    (chunks / "Aaaa.js").write_text("nothing here", encoding="utf-8")
    (chunks / "COZ9VdsL.js").write_text(bundle, encoding="utf-8")
    return settings, chunks / "COZ9VdsL.js"


def test_chunk_is_found_by_content_not_name(tmp_path):
    settings, chunk = _fake_frontend(tmp_path)
    assert find_call_silence_chunk(frontend_build_dir(settings)) == chunk
    assert find_call_audio_context_chunk(frontend_build_dir(settings)) == chunk
    assert find_call_barge_in_chunk(frontend_build_dir(settings)) == chunk
    assert find_call_audio_chunk(frontend_build_dir(settings)) == chunk


def test_reconcile_rewrites_the_bundle_and_is_idempotent(tmp_path):
    settings, chunk = _fake_frontend(tmp_path)
    settings.openwebui.call_silence_ms = 1000
    assert "set to 1000ms (was 2000ms)" in reconcile_call_silence(settings)
    assert "Date.now()-Ct>1000&&" in chunk.read_text(encoding="utf-8")
    assert "already 1000ms" in reconcile_call_silence(settings)
    assert list(chunk.parent.glob("*.locitize-tmp")) == []


def test_reconcile_removes_feedback_prone_call_guard(tmp_path):
    settings, chunk = _fake_frontend(tmp_path)
    guarded, _ = patch_call_barge_in(chunk.read_text(encoding="utf-8"), enabled=True)
    chunk.write_text(guarded, encoding="utf-8")
    assert "guard removed" in reconcile_call_barge_in(settings)
    assert 'speechSynthesis.speaking' not in chunk.read_text(encoding="utf-8")
    assert "already removed" in reconcile_call_barge_in(settings)
    assert list(chunk.parent.glob("*.locitize-tmp")) == []


def test_reconcile_repairs_mobile_call_audio_and_is_idempotent(tmp_path):
    settings, chunk = _fake_frontend(tmp_path)
    assert "audio fixed" in reconcile_call_audio_playback(settings)
    assert "globalThis.__locitizeCallAudio" in chunk.read_text(encoding="utf-8")
    assert "already applied" in reconcile_call_audio_playback(settings)
    assert list(chunk.parent.glob("*.locitize-tmp")) == []


def test_reconcile_repairs_call_microphone_lifecycle_and_is_idempotent(tmp_path):
    settings, chunk = _fake_frontend(tmp_path)
    assert "lifecycle fixed" in reconcile_call_audio_context(settings)
    assert "__locitizeCallAudioContext" in chunk.read_text(encoding="utf-8")
    assert "already applied" in reconcile_call_audio_context(settings)
    assert list(chunk.parent.glob("*.locitize-tmp")) == []


def _add_frontend_version_files(settings, version="0.11.1"):
    import json

    frontend = frontend_build_dir(settings)
    version_path = frontend / "_app" / "version.json"
    version_path.write_text(json.dumps({"version": version}), encoding="utf-8")
    runtime = frontend / "_app" / "immutable" / "chunks" / "Runtime.js"
    runtime.write_text(
        f'const appVersion={json.dumps(version)};fetch("/_app/version.json");',
        encoding="utf-8",
    )
    return version_path, runtime


def test_frontend_version_busts_stale_pwa_cache_and_is_idempotent(tmp_path):
    import json

    settings, chunk = _fake_frontend(tmp_path)
    settings.openwebui.call_silence_ms = 500
    reconcile_call_silence(settings)
    reconcile_call_barge_in(settings)
    version_path, runtime = _add_frontend_version_files(settings)

    message = reconcile_frontend_version(settings)
    version = json.loads(version_path.read_text(encoding="utf-8"))["version"]
    assert message == f"Open WebUI PWA cache identity set to {version}"
    assert version.startswith("0.11.1+locitize.")
    assert len(version.rsplit(".", 1)[1]) == 12
    assert f'"{version}"' in runtime.read_text(encoding="utf-8")
    assert list(runtime.parent.glob("*.locitize-version-*")) == []

    before = (version_path.stat().st_mtime_ns, runtime.stat().st_mtime_ns)
    assert "already" in reconcile_frontend_version(settings)
    assert (version_path.stat().st_mtime_ns, runtime.stat().st_mtime_ns) == before


def test_frontend_version_changes_when_the_call_bundle_changes(tmp_path):
    import json

    settings, chunk = _fake_frontend(tmp_path)
    version_path, runtime = _add_frontend_version_files(settings)
    reconcile_frontend_version(settings)
    first = json.loads(version_path.read_text(encoding="utf-8"))["version"]

    settings.openwebui.call_silence_ms = 500
    reconcile_call_silence(settings)
    assert "set to" in reconcile_frontend_version(settings)
    second = json.loads(version_path.read_text(encoding="utf-8"))["version"]
    assert second != first
    assert f'"{second}"' in runtime.read_text(encoding="utf-8")


def test_frontend_version_leaves_an_unknown_runtime_alone(tmp_path):
    import json

    settings, _chunk = _fake_frontend(tmp_path)
    version_path = frontend_build_dir(settings) / "_app" / "version.json"
    version_path.write_text(json.dumps({"version": "0.11.1"}), encoding="utf-8")
    assert "not recognized" in reconcile_frontend_version(settings)
    assert json.loads(version_path.read_text(encoding="utf-8"))["version"] == "0.11.1"


def test_reconcile_at_upstream_value_does_not_write(tmp_path):
    settings, chunk = _fake_frontend(tmp_path)
    before = chunk.stat().st_mtime_ns
    assert "default 2000ms" in reconcile_call_silence(settings)
    assert chunk.stat().st_mtime_ns == before


def test_reconcile_restores_upstream_when_the_owner_sets_2000(tmp_path):
    settings, chunk = _fake_frontend(tmp_path)
    settings.openwebui.call_silence_ms = 700
    reconcile_call_silence(settings)
    settings.openwebui.call_silence_ms = CALL_SILENCE_UPSTREAM_MS
    assert "(was 700ms)" in reconcile_call_silence(settings)
    assert "Date.now()-Ct>2000&&" in chunk.read_text(encoding="utf-8")


def test_reconcile_is_honest_about_a_bundle_it_does_not_understand(tmp_path):
    settings, _chunk = _fake_frontend(tmp_path, bundle="const overlay = 'rewritten';")
    settings.openwebui.call_silence_ms = 1000
    assert "has no effect on this build" in reconcile_call_silence(settings)


def test_reconcile_without_a_frontend_is_a_no_op(tmp_path):
    settings = _fake_codebase(tmp_path)
    settings.openwebui.call_silence_ms = 1000
    assert "left alone" in reconcile_call_silence(settings)


def test_call_silence_out_of_range_is_reported_and_reset(tmp_path):
    (tmp_path / "settings.yaml").write_text(
        "version: 1\nopenwebui:\n  enabled: true\n  call_silence_ms: 50\n", encoding="utf-8"
    )
    _models_yaml(tmp_path)
    settings, _m, issues = Config.load(tmp_path)
    assert settings.openwebui.call_silence_ms == CALL_SILENCE_UPSTREAM_MS
    assert any("call_silence_ms" in issue.message for issue in issues)


def test_call_silence_writer_inserts_then_rewrites(tmp_path):
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_OPENWEBUI, encoding="utf-8")
    _models_yaml(tmp_path)
    write_openwebui_call_silence(tmp_path, 1000)
    text = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert "  call_silence_ms: 1000\n" in text
    assert "opt-in, ships off (a comment that must survive the write)" in text
    assert text.index("call_silence_ms") < text.index("chat:")  # inside the block
    write_openwebui_call_silence(tmp_path, 800)
    settings, _m, _i = Config.load(tmp_path)
    assert settings.openwebui.call_silence_ms == 800
    assert (tmp_path / "settings.yaml").read_text(encoding="utf-8").count("call_silence_ms") == 1


def test_call_silence_writer_rejects_out_of_range(tmp_path):
    (tmp_path / "settings.yaml").write_text(_SETTINGS_WITH_OPENWEBUI, encoding="utf-8")
    with pytest.raises(ValueError):
        write_openwebui_call_silence(tmp_path, 10)
    with pytest.raises(ValueError):
        write_openwebui_call_silence(tmp_path, True)


def test_tts_voice_writer_preserves_neighbors_and_loads(tmp_path):
    original = (
        "version: 1\n"
        "tts:\n"
        "  enabled: true\n"
        "  voice: am_michael  # keep this comment\n"
        "  speed: 1.0\n"
    )
    (tmp_path / "settings.yaml").write_text(original, encoding="utf-8")
    _models_yaml(tmp_path)

    write_tts_voice(tmp_path, "af_heart")

    text = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert "voice: af_heart  # keep this comment" in text
    assert "  speed: 1.0\n" in text
    settings, _models, _issues = Config.load(tmp_path)
    assert settings.tts.voice == "af_heart"


@pytest.mark.parametrize("voice", ["", "../voice", "af heart"])
def test_tts_voice_writer_rejects_non_voice_names(tmp_path, voice):
    (tmp_path / "settings.yaml").write_text(
        "version: 1\ntts:\n  voice: am_michael\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        write_tts_voice(tmp_path, voice)


def _defaults_row(tmp_path):
    import json
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "webui-data" / "webui.db"))
    row = conn.execute(
        "select value from config where key = 'ui.default_interface_settings'"
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def test_voice_interruption_is_defaulted_on(tmp_path):
    """Call mode keeps the mic open so a real utterance can stop the reply."""
    _audio_db(tmp_path)
    settings = _settings_with_router(tmp_path)
    assert "set on" in reconcile_voice_interruption(settings)
    assert _defaults_row(tmp_path) == {"voiceInterruption": True}
    assert "already on" in reconcile_voice_interruption(settings)


def test_voice_interruption_turns_old_default_on_and_preserves_other_defaults(tmp_path):
    import json
    import sqlite3

    _audio_db(tmp_path)
    conn = sqlite3.connect(str(tmp_path / "webui-data" / "webui.db"))
    conn.execute(
        "insert into config values ('ui.default_interface_settings', ?, 0)",
        (json.dumps({"voiceInterruption": False, "widescreenMode": True}),),
    )
    conn.commit()
    conn.close()
    assert "set on" in reconcile_voice_interruption(_settings_with_router(tmp_path))
    assert _defaults_row(tmp_path) == {"voiceInterruption": True, "widescreenMode": True}


def test_voice_interruption_can_be_forced_off(tmp_path):
    _audio_db(tmp_path)
    settings = _settings_with_router(tmp_path)
    settings.openwebui.voice_interruption = False
    assert "set off" in reconcile_voice_interruption(settings)
    assert _defaults_row(tmp_path) == {"voiceInterruption": False}


def test_voice_interruption_leaves_a_disabled_router_alone(tmp_path):
    _audio_db(tmp_path)
    assert "left alone" in reconcile_voice_interruption(
        _settings_with_router(tmp_path, enabled=False)
    )
    assert _defaults_row(tmp_path) is None


def test_call_defaults_are_reconciled_before_open_webui_starts():
    """The persisted voice/backend edits and model-default clearing stay
    together ahead of the start and post-start /health confirm."""
    import inspect

    import launcher

    src = inspect.getsource(launcher.Launcher._start_openwebui)
    for call in ("reconcile_voice_interruption(settings)", "reconcile_call_silence(settings)",
                 "reconcile_call_audio_playback(settings)",
                 "reconcile_frontend_version(settings)", "reconcile_default_model(settings)"):
        assert src.index(call) < src.index("controller.start()")
        assert src.index(call) < src.index("wait_openwebui_healthy(port")


# --------------------------------------------------------------------------- #
# Model authority (owner correction 2026-09-03): no global model default is
# saved. The model picker on each Open WebUI request is authoritative.
# --------------------------------------------------------------------------- #

from webui import reconcile_default_model  # noqa: E402


def _default_models_row(tmp_path):
    import json
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "webui-data" / "webui.db"))
    row = conn.execute("select value from config where key = 'ui.default_models'").fetchone()
    conn.close()
    return json.loads(row[0]) if row else "absent"


def _seed_default_models(tmp_path, raw):
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "webui-data" / "webui.db"))
    conn.execute("insert into config values ('ui.default_models', ?, 0)", (raw,))
    conn.commit()
    conn.close()


def test_reconcile_clears_saved_openwebui_model_default(tmp_path):
    _audio_db(tmp_path)
    _seed_default_models(tmp_path, '"big-model, other"')
    settings = _settings_with_router(tmp_path)
    settings.launcher.default_model = "locitize-choice"

    message = reconcile_default_model(settings)

    assert "default cleared" in message
    assert _default_models_row(tmp_path) == "absent"


def test_absent_openwebui_default_stays_unset(tmp_path):
    _audio_db(tmp_path)
    settings = _settings_with_router(tmp_path)

    assert "no saved model default" in reconcile_default_model(settings)
    assert _default_models_row(tmp_path) == "absent"

def test_reconcile_reports_a_missing_database_without_creating_one(tmp_path):
    settings = _settings_with_router(tmp_path)

    assert "no database yet" in reconcile_default_model(settings)
    assert not (tmp_path / "webui-data" / "webui.db").exists()



def test_start_openwebui_never_upgrades_and_always_health_checks():
    """A normal start installs nothing (no unpinned pip -U on every launch).

    A TCP-ready early-return after a kill races a dying :8096 listener and
    leaves Serve 502. Source contract: start -> /health FAIL log, no upgrade.
    """
    import inspect

    import launcher

    src = inspect.getsource(launcher.Launcher._start_openwebui)
    assert "upgrade_openwebui" not in src
    assert "pip" not in src
    assert "controller.start()" in src
    assert "wait_openwebui_healthy(port" in src
    assert "Open WebUI FAIL" in src
    # Must not skip start on a brief post-kill TCP answer.
    assert "if self._openwebui_ready(settings):" not in src
    assert src.index("controller.start()") < src.index("wait_openwebui_healthy(port")


def test_openwebui_process_check_only_targets_locitize_venv(tmp_path, monkeypatch):
    """Another Open WebUI install (or an editor on a checkout) is never killed."""
    import psutil

    from webui import ensure_single_openwebui_processes

    ours = tmp_path / ".webui-venv"
    killed = []

    class FakeProc:
        def __init__(self, pid, exe, cmdline):
            self.pid = pid
            self.info = {"pid": pid, "name": "python.exe", "exe": exe, "cmdline": cmdline}

        def terminate(self):
            killed.append(self.pid)

        def wait(self, timeout=None):
            return 0

    procs = [
        FakeProc(1, str(ours / "Scripts" / "python.exe"),
                 [str(ours / "Scripts" / "open-webui.exe"), "serve"]),
        FakeProc(2, str(tmp_path / "other" / "python.exe"), ["open-webui", "serve"]),
        FakeProc(3, str(tmp_path / "Code.exe"), ["code", str(tmp_path / "open-webui")]),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(procs))
    ensure_single_openwebui_processes(8096, force=True, venv_dir=ours)
    assert killed == [1]
    killed.clear()
    assert "skipped" in ensure_single_openwebui_processes(8096, force=True)
    assert killed == []


def test_wait_openwebui_healthy_false_when_nothing_listens():
    """Health helper times out cleanly on a closed loopback port."""
    from webui import wait_openwebui_healthy

    assert wait_openwebui_healthy(1, timeout_s=0.3, poll_s=0.1) is False


def test_wait_loopback_port_free_true_when_closed():
    from webui import wait_loopback_port_free

    assert wait_loopback_port_free(1, timeout_s=0.5) is True


def test_upgrade_openwebui_fail_soft_when_venv_missing(tmp_path):
    """Missing .webui-venv python -> clear skip line, no raise."""
    from webui import upgrade_openwebui
    settings = _fake_codebase(tmp_path, install=False)
    msg = upgrade_openwebui(settings)
    assert "auto-update skipped" in msg.lower() or "missing" in msg.lower()
