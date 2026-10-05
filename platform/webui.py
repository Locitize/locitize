"""Open WebUI managed chat service + the chat-UI chooser (LOCITIZE M9-lite).

This is the tts.py analog for the chat application. It does two small things and
nothing else, because LOCITIZE manages chat applications, it never rebuilds chat:

- build_openwebui_spec -> a declarative ServiceSpec for the Open WebUI child, run
  from the DEDICATED .webui-venv interpreter (never the platform venv) on a loopback
  port, wired to the running llama.cpp server as its OpenAI-compatible backend, with
  the first-run RAG embedding-model network fetch disabled by default.
- resolve_chat_choice -> a PURE decision function (no tkinter, no I/O, no network)
  that maps (persisted preference x CLI override x model-running x Open-WebUI-installed
  x Open-WebUI-ready) onto one ChatDecision, so the GUI dialog and the terminal menu
  share a single, exhaustively unit-tested decision path.

It never launches a process itself (services.py owns that via ServiceManager /
SingleServiceController, the same lifecycle that runs llama.cpp, whisper, and the M6
kokoro_server) and never opens a browser (gui_controller / the launcher reuse the
existing open_chat path). Open WebUI owns its own chat surface, history store, and
model serving; LOCITIZE only installs it, points it at llama.cpp, starts it, and stops
it with no orphan. ASCII only.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from config import CALL_SILENCE_UPSTREAM_MS, Settings
from services import ServiceSpec, resolve_service_cwd

# Non-empty because Open WebUI requires a value; llama.cpp ignores it. A
# placeholder, not a credential (Data Model section 10). Single-sourced so the
# spec env and the persisted-config reconciler cannot drift apart - they are
# paired BY INDEX with api_base_urls, and a mismatch silently drops a
# connection from the picker.
OPENAI_PLACEHOLDER_KEY = "sk-locitize-local-placeholder"

# Stable service identifier shared by the launcher, the controller, and tests.
OPENWEBUI_NAME = "openwebui"

# Open WebUI exposes an HTTP health endpoint that returns 200 once the app has
# finished its first-run database migration and is ready to serve. Confirmed against
# the installed release at build time (R1 discipline). The loopback host in the
# composed probe URL is a hardcoded constant in services.build_readiness_url, so this
# path can never steer the probe off 127.0.0.1 (Security SEC-1).
OPENWEBUI_HEALTH_PATH = "/health"


class ChatDecision(Enum):
    """The result of resolve_chat_choice (Architecture M9.3, Data Model 10.4).

    OPEN_LLAMACPP        - open the built-in llama.cpp web UI (zero setup)
    OPEN_OPENWEBUI       - open the running Open WebUI service (rich chat, history)
    ASK                  - present the choice to the owner (both UIs are available)
    OFFER_START_OPENWEBUI- Open WebUI is installed but not running; offer to start it
    DEGRADE_TO_LLAMACPP  - Open WebUI was requested but is not installed/broken;
                           fall back to the built-in UI (carries the remedy)
    NO_MODEL             - no chat backend is running; open neither UI onto a dead port
    """

    OPEN_LLAMACPP = "open_llamacpp"
    OPEN_OPENWEBUI = "open_openwebui"
    ASK = "ask"
    OFFER_START_OPENWEBUI = "offer_start_openwebui"
    DEGRADE_TO_LLAMACPP = "degrade_to_llamacpp"
    NO_MODEL = "no_model"


@dataclass(frozen=True)
class ChatResolution:
    """One chooser outcome: a decision plus an owner-facing reason string.

    The reason is empty for the plain "just open it" decisions (OPEN_LLAMACPP,
    OPEN_OPENWEBUI, ASK) and carries an honest one-line explanation for the
    degrade/offer/no-model cases so no surface ever needs to invent its own text or
    print a traceback (Architecture M9.5).
    """

    decision: ChatDecision
    reason: str = ""


# Owner-facing reason strings, single-sourced here so the GUI dialog, the terminal
# menu, and the tests all assert the exact same honest text (no drift between
# surfaces). docs/chat.md is the install/troubleshoot reference.
_REASON_NO_MODEL = "no model running - start one first"
_REASON_NOT_INSTALLED = (
    "Open WebUI is not installed; opening the built-in llama.cpp UI. "
    "To install it, see docs/chat.md"
)
_REASON_NOT_RUNNING = "Open WebUI is installed but not running"


def resolve_chat_choice(
    preferred: str,
    cli_override: str | None,
    model_running: bool,
    webui_installed: bool,
    webui_ready: bool,
) -> ChatResolution:
    """Pure chat-UI decision (Architecture M9.3). No tkinter, no I/O, no network.

    Precedence (single decision path shared by the GUI and the terminal):

    1. cli_override (--chat-ui) beats the persisted `preferred`; either value is one
       of ask | llamacpp | openwebui, and anything else is treated as 'ask' so a
       stray value can never crash the chooser.
    2. A missing running model short-circuits to NO_MODEL: chat needs a backend, and
       neither UI is opened onto a dead port (the same guard the existing open_chat
       already enforces).
    3. 'llamacpp' -> OPEN_LLAMACPP (the built-in UI is always available when a model
       is running).
    4. 'openwebui' -> OPEN_OPENWEBUI when installed AND ready; OFFER_START_OPENWEBUI
       when installed but not running (an honest offer, never an auto-launch
       surprise); DEGRADE_TO_LLAMACPP with the remedy when not installed/broken.
    5. 'ask' -> ASK when Open WebUI is installed (a real second choice exists),
       otherwise OPEN_LLAMACPP (there is nothing to ask about when only the built-in
       UI exists - the same collapse as openwebui.enabled=false).

    Returns a ChatResolution (decision + reason). Exhaustively unit-tested across the
    whole matrix (keyword chat_chooser).
    """
    # cli_override wins over the persisted preference; normalize unknowns to 'ask'.
    effective = (cli_override or preferred or "ask").strip().lower()
    if effective not in ("ask", "llamacpp", "openwebui"):
        effective = "ask"

    # A backend must be running before either UI is worth opening (guard first).
    if not model_running:
        return ChatResolution(ChatDecision.NO_MODEL, _REASON_NO_MODEL)

    if effective == "llamacpp":
        return ChatResolution(ChatDecision.OPEN_LLAMACPP)

    if effective == "openwebui":
        if not webui_installed:
            return ChatResolution(
                ChatDecision.DEGRADE_TO_LLAMACPP, _REASON_NOT_INSTALLED
            )
        if not webui_ready:
            return ChatResolution(
                ChatDecision.OFFER_START_OPENWEBUI, _REASON_NOT_RUNNING
            )
        return ChatResolution(ChatDecision.OPEN_OPENWEBUI)

    # effective == 'ask': only a real choice when Open WebUI is actually available.
    if not webui_installed:
        return ChatResolution(ChatDecision.OPEN_LLAMACPP, _REASON_NOT_INSTALLED)
    return ChatResolution(ChatDecision.ASK)


def _webui_venv_scripts(settings: Settings) -> Path:
    """Return the dedicated .webui-venv Scripts dir (NEVER the platform venv).

    Open WebUI lives in its own isolated venv (Architecture M9.2) whose pins would
    clash with the platform venv's torch/kokoro pins, so its child must run from that
    venv, not the platform interpreter. The venv dir name comes from
    settings.openwebui.venv and is resolved UNDER Codebase/ (base_dir.parent),
    path-safe: a name that tried to escape (e.g. an absolute path or ..) is rejected
    so the child can only ever be launched from inside the Codebase tree.
    """
    from runtime_layout import bundled_python, packaged_environment_root
    codebase = (packaged_environment_root(settings.data_dir) if bundled_python(settings.base_dir)
                else settings.base_dir.parent)
    venv_name = settings.openwebui.venv or ".webui-venv"
    venv_dir = (codebase / venv_name).resolve()
    # Path-safety: the resolved venv dir must stay under Codebase/ (Permission Matrix).
    codebase_resolved = codebase.resolve()
    if codebase_resolved != venv_dir and codebase_resolved not in venv_dir.parents:
        raise ValueError(
            f"openwebui.venv '{venv_name}' resolves outside Codebase/; refusing to "
            f"run a child from an out-of-tree venv"
        )
    return venv_dir / "Scripts"


def webui_venv_python(settings: Settings) -> Path:
    """Return the dedicated .webui-venv interpreter path (NEVER the platform venv)."""
    return _webui_venv_scripts(settings) / "python.exe"


def webui_console_script(settings: Settings) -> Path:
    """Return the dedicated .webui-venv `open-webui` console script path.

    R1 confirmation (open-webui 0.10.2): the working, stable invocation is this
    console entry point's `serve` subcommand -- `python -m open_webui serve` is NOT
    available in this release (the package ships no __main__), exactly the release-
    to-release CLI variance RC1 anticipated. Launching this .exe still runs the child
    entirely from the dedicated venv (it is a sibling of that venv's python.exe),
    never the platform venv, satisfying the isolation requirement.
    """
    return _webui_venv_scripts(settings) / "open-webui.exe"


def webui_data_dir(settings: Settings) -> Path:
    """Return the Open WebUI DATA_DIR under the user data root (path-safe).

    Its sqlite DB, chat history, uploads, and config live here. The dir name comes
    from settings.openwebui.data_dir and is resolved UNDER settings.data_dir and
    forced to stay there, so a crafted value cannot point Open WebUI's private
    store somewhere else (Data Model section 10, Permission Matrix).

    The data root, never the install directory (DEC-M14-9, defect NEW-QA-M14-8):
    this is the user's chat database, so a reinstall must not be able to delete
    it and a backup of the data folder must contain it. Moving it here also
    carries Open WebUI's generated .webui_secret_key out of the tracked source
    tree structurally, rather than relying on a .gitignore line.
    """
    base = Path(settings.data_dir).resolve()
    name = settings.openwebui.data_dir or "webui-data"
    data_dir = (base / name).resolve()
    if base != data_dir and base not in data_dir.parents:
        raise ValueError(
            f"openwebui.data_dir '{name}' resolves outside the locitize data "
            f"folder; refusing to point the chat data store out of tree"
        )
    return data_dir


def webui_available(settings: Settings) -> bool:
    """Cheap check: is Open WebUI usable (enabled, venv present, console script there)?

    Used by the chooser to compute `webui_installed` without importing anything heavy
    from the Open WebUI package. False when the service is disabled in settings, the
    dedicated venv is absent, or its `open-webui` console entry point is missing (a
    half-installed venv). A False here makes the chooser degrade honestly to the
    built-in UI rather than trying to start a broken service (Architecture M9.5).
    """
    if not settings.openwebui.enabled:
        return False
    try:
        entry = webui_console_script(settings)
    except ValueError:
        return False
    # The console entry point Open WebUI installs into the venv Scripts dir. Its
    # presence is a reliable, import-free signal that `pip install open-webui`
    # completed in this venv.
    return entry.is_file()


def backend_base_url(settings: Settings) -> str:
    """The OpenAI-compatible base URL Open WebUI talks to (the running llama.cpp).

    Empty settings.openwebui.backend_base_url -> derive http://127.0.0.1:<llama_cpp
    port>/v1 (the fixed model port; a switched model normally re-binds it, so Open
    WebUI keeps working across switches - Architecture M9.2). A non-empty value has
    already been validated as a credential-free loopback URL by config._validate.
    """
    configured = settings.openwebui.backend_base_url
    if configured:
        return configured
    # Owner request 2026-09-02: with the router enabled, point Open WebUI at it
    # instead of straight at llama-server. The direct URL advertises only the ONE
    # model llama-server has loaded, so the picker showed a single entry and
    # changing models meant leaving the browser. The router answers /v1/models
    # from the registry and switches what is served when the picker changes.
    # An explicit backend_base_url still wins over both - it is the owner
    # pointing at something deliberately.
    if getattr(settings, "router", None) is not None and settings.router.enabled:
        return f"http://127.0.0.1:{settings.ports.router}/v1"
    return f"http://127.0.0.1:{settings.ports.llama_cpp}/v1"


# Hosts that mean "this machine". A persisted backend URL on one of these, at a
# port LOCITIZE reserves, is a URL LOCITIZE itself wrote - and therefore one it
# may rewrite. Anything else is the owner pointing at their own provider and is
# left completely alone.
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def plan_backend_urls(
    existing: list[str], intended: str, locitize_ports: list[int]
) -> list[str]:
    """The api_base_urls list Open WebUI should persist. Pure; no I/O.

    Rules, in order:
    - Every entry that is a LOCITIZE-owned loopback URL (one of our reserved
      ports on this machine) is REPLACED by `intended`. That is the migration:
      a row written when the backend was llama-server directly now points at
      the router, and vice versa when the router is turned off.
    - Every other entry is preserved verbatim, in place. A URL the owner added
      for their own provider is not ours to touch.
    - If nothing was replaced and `intended` is not already present, it is
      APPENDED rather than substituted, so enabling the router on an install
      that points somewhere else adds a choice instead of hijacking one.
    - Duplicates are collapsed, keeping first occurrence order.
    """
    owned = {
        f"http://{host}:{port}/v1"
        for host in _LOOPBACK_HOSTS
        for port in locitize_ports
    }

    def is_ours(url: str) -> bool:
        return url.rstrip("/") in {u.rstrip("/") for u in owned}

    planned: list[str] = []
    replaced = False
    for url in existing:
        text = str(url).strip()
        if not text:
            continue
        if is_ours(text):
            planned.append(intended)
            replaced = True
        else:
            planned.append(text)
    if not replaced and intended not in planned:
        planned.append(intended)
    seen: set[str] = set()
    deduped: list[str] = []
    for url in planned:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def reconcile_persisted_backend(settings: Settings) -> str:
    """Point the PERSISTED Open WebUI backend at what settings now say.

    Defect found 2026-09-02: with the model router enabled, Open WebUI still
    showed "No models available". Its OPENAI_API_BASE_URLS env var is a SEED
    DEFAULT only - on an already-initialized DATA_DIR the row in webui.db wins,
    exactly as this module already documented for the web-search flags. The
    persisted row still read ["http://127.0.0.1:8080/v1"], so the picker was
    asking a llama-server that was not running.

    Returns a human line describing what happened (never raises): Open WebUI
    failing to be reconfigured must degrade to a message, not stop a chat UI
    from starting.

    MUST run while Open WebUI is STOPPED. It caches this config in memory and
    writes it back on shutdown, so a live process would overwrite the edit.
    """
    import json
    import sqlite3

    db = webui_data_dir(settings) / "webui.db"
    if not db.is_file():
        # A first run has no database yet; the env seed will be used, which is
        # exactly the right thing and needs no help from here.
        return "Open WebUI has no database yet; env defaults apply"

    intended = backend_base_url(settings)
    ports = [settings.ports.llama_cpp, settings.ports.router]
    try:
        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute(
                "select value from config where key = ?", ("openai.api_base_urls",)
            ).fetchone()
            existing = json.loads(row[0]) if row and row[0] else []
            if not isinstance(existing, list):
                existing = []
            planned = plan_backend_urls(existing, intended, ports)
            if planned == existing:
                return f"Open WebUI already points at {intended}"
            conn.execute(
                "insert into config (key, value, updated_at) values (?, ?, ?) "
                "on conflict(key) do update set value = excluded.value, "
                "updated_at = excluded.updated_at",
                ("openai.api_base_urls", json.dumps(planned), int(time.time())),
            )
            # api_keys is paired with api_base_urls BY INDEX, so a list that is
            # shorter than the URLs leaves later connections keyless and Open
            # WebUI drops them. Pad with the same placeholder the env uses.
            key_row = conn.execute(
                "select value from config where key = ?", ("openai.api_keys",)
            ).fetchone()
            keys = json.loads(key_row[0]) if key_row and key_row[0] else []
            if not isinstance(keys, list):
                keys = []
            while len(keys) < len(planned):
                keys.append(OPENAI_PLACEHOLDER_KEY)
            conn.execute(
                "insert into config (key, value, updated_at) values (?, ?, ?) "
                "on conflict(key) do update set value = excluded.value, "
                "updated_at = excluded.updated_at",
                ("openai.api_keys", json.dumps(keys[: len(planned)]), int(time.time())),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - honest degrade, never block start
        return f"could not update the Open WebUI backend URL: {exc}"
    return f"Open WebUI backend set to {intended}"


def audio_config_rows(settings: Settings) -> dict[str, Any]:
    """The webui.db config rows that point Open WebUI's audio at the router.

    Pure; no I/O. Open WebUI names both engines "openai" because that is the
    API SHAPE, not the vendor - router.py serves /v1/audio/transcriptions and
    /v1/audio/speech itself, translating onto whisper-server and kokoro_server
    (see audio_api). No request leaves this machine.

    The alternative Open WebUI offers is the browser's own speech engine, which
    would send every recorded utterance to Google. That is a strange thing to do
    inside a platform whose premise is that the models run on your machine, so
    it is not what LOCITIZE configures.

    The API keys are the same non-empty placeholder the chat backend uses: Open
    WebUI refuses to call an endpoint with an empty key, and the router does not
    check it. A placeholder, not a credential (Data Model section 10).
    """
    base = backend_base_url(settings)
    return {
        "audio.stt.engine": "openai",
        "audio.stt.openai.api_base_url": base,
        "audio.stt.openai.api_key": OPENAI_PLACEHOLDER_KEY,
        # whisper-server has one loaded model; naming another would be a claim
        # LOCITIZE cannot honour, so the OpenAI-conventional name is sent and
        # audio_api ignores it.
        "audio.stt.model": "whisper-1",
        "audio.tts.engine": "openai",
        "audio.tts.openai.api_base_url": base,
        # Open WebUI 0.11.1's OpenAI TTS path reads this provider-specific
        # key.  ``audio.tts.api_key`` belongs to the ElevenLabs/Azure paths;
        # populating it leaves OpenAI TTS with an empty bearer token.
        "audio.tts.openai.api_key": OPENAI_PLACEHOLDER_KEY,
        "audio.tts.model": "tts-1",
        # A voice Kokoro actually has. Open WebUI ships "alloy", which Kokoro
        # would reject; audio_api.resolve_voice falls back safely either way,
        # but seeding the real name means the UI shows the truth.
        "audio.tts.voice": settings.tts.voice,
    }


def reconcile_persisted_audio(settings: Settings) -> str:
    """Point the PERSISTED Open WebUI audio config at the router's endpoints.

    Same seed-vs-database trap as reconcile_persisted_backend: the audio engine
    cannot be set by env at all in this build, so without this the microphone
    button falls through to Open WebUI's own faster-whisper - which the
    OFFLINE_MODE this module sets deliberately prevents from downloading a
    model. The button then does nothing, with no explanation.

    Returns a human line, never raises: a chat UI that cannot do voice is worth
    starting anyway. MUST run while Open WebUI is STOPPED, for the same reason
    as its sibling - it caches config and writes it back on shutdown.

    Does nothing when router.audio is off, so an owner who turned the audio
    routes off does not get Open WebUI pointed at a 503.
    """
    import json
    import sqlite3

    if not getattr(settings, "router", None) or not settings.router.enabled:
        return "router disabled; Open WebUI audio left alone"
    if not settings.router.audio:
        return "router.audio disabled; Open WebUI audio left alone"

    db = webui_data_dir(settings) / "webui.db"
    if not db.is_file():
        return "Open WebUI has no database yet; audio defaults apply"

    rows = audio_config_rows(settings)
    try:
        conn = sqlite3.connect(str(db))
        try:
            changed = []
            for key, value in rows.items():
                encoded = json.dumps(value)
                current = conn.execute(
                    "select value from config where key = ?", (key,)
                ).fetchone()
                if current and current[0] == encoded:
                    continue
                conn.execute(
                    "insert into config (key, value, updated_at) values (?, ?, ?) "
                    "on conflict(key) do update set value = excluded.value, "
                    "updated_at = excluded.updated_at",
                    (key, encoded, int(time.time())),
                )
                changed.append(key)
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - never block the chat UI starting
        return f"could not update the Open WebUI audio config: {exc}"
    if not changed:
        return "Open WebUI audio already points at locitize"
    return f"Open WebUI audio set to locitize speech services ({len(changed)} setting(s))"


def reconcile_voice_interruption(settings: Settings) -> str:
    """Set Open WebUI's 'Allow Voice Interruption in Call' from LOCITIZE settings.

    True (the default): the Call overlay keeps the microphone open while Kokoro
    speaks, so a real utterance stops the reply. Open WebUI already asks the
    browser for echoCancellation. Speaker-phone bleed can still false-trigger;
    a headset is the reliable path. False restores half-duplex.

    Writes the ADMIN default (ui.default_interface_settings), which Open WebUI
    merges under any per-user setting that does not already name the key.

    Same seed-vs-database rule as its siblings: a persisted row, run while Open
    WebUI is stopped. Returns a human line, never raises.
    """
    import json
    import sqlite3

    if not getattr(settings, "router", None) or not settings.router.enabled:
        return "router disabled; Open WebUI call defaults left alone"
    if not settings.router.audio:
        return "router.audio disabled; Open WebUI call defaults left alone"

    db = webui_data_dir(settings) / "webui.db"
    if not db.is_file():
        return "Open WebUI has no database yet; call defaults apply on first run"

    wanted = bool(getattr(settings.openwebui, "voice_interruption", True))
    key = "ui.default_interface_settings"
    try:
        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute("select value from config where key = ?", (key,)).fetchone()
            current: dict = {}
            if row and row[0]:
                try:
                    loaded = json.loads(row[0])
                except ValueError:
                    loaded = None
                if isinstance(loaded, dict):
                    current = loaded
            if current.get("voiceInterruption") is wanted:
                return (
                    "Open WebUI voice interruption already on"
                    if wanted
                    else "Open WebUI voice interruption already off"
                )
            current["voiceInterruption"] = wanted
            conn.execute(
                "insert into config (key, value, updated_at) values (?, ?, ?) "
                "on conflict(key) do update set value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key, json.dumps(current), int(time.time())),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - never block the chat UI starting
        return f"could not set the Open WebUI voice interruption default: {exc}"
    return (
        "Open WebUI voice interruption set on (speak to cut in)"
        if wanted
        else "Open WebUI voice interruption set off (prevents phone speaker feedback)"
    )


def reconcile_default_model(settings: Settings) -> str:
    """Ensure Open WebUI has no persisted global model default.

    The model in each Open WebUI request is authoritative and the router starts
    or switches to that model.  A saved ``ui.default_models`` value bypasses that
    per-chat choice at launch, so LOCITIZE removes the row instead of seeding or
    activating it.
    """
    import sqlite3

    db = webui_data_dir(settings) / "webui.db"
    if not db.is_file():
        return "Open WebUI has no database yet; no model default saved"
    try:
        conn = sqlite3.connect(str(db))
        try:
            cursor = conn.execute(
                "delete from config where key = ?", ("ui.default_models",)
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - never block the chat UI starting
        return f"could not clear the Open WebUI model default: {exc}"
    if cursor.rowcount:
        return "Open WebUI saved model default cleared; picker choice applies per request"
    return "Open WebUI has no saved model default; picker choice applies per request"


# --------------------------------------------------------------------------- #
# Call-mode end-of-speech wait (owner request 2026-09-03, "talk to my models
# like ChatGPT").
#
# Measured on this machine, a warm voice turn is 0.9s of work: 0.36s whisper,
# 0.12s to the model's first sentence, 0.4s Kokoro. What the owner FEELS is
# ~3s, because Open WebUI's Call overlay waits a fixed 2000ms of silence after
# the last sound before it sends the recording - two-thirds of every turn is
# nobody doing anything. The constant is not a setting; it is a literal in the
# compiled frontend bundle (CallOverlay.svelte: `Date.now() - lastSoundTime >
# 2000`).
#
# So this rewrites that one literal, in place, at Open WebUI start, when the
# owner has set openwebui.call_silence_ms to anything but the upstream 2000.
# Deliberately narrow:
#   - the match is anchored on the neighbouring `voiceInterruption` check so a
#     `2e3` elsewhere in a 900 KB bundle cannot be hit;
#   - it is a symmetric substitution, so setting 2000 again restores the
#     upstream behaviour (no backup needed, and none is kept);
#   - a bundle that does not contain the pattern (a future Open WebUI that
#     restructured the overlay) is reported and left untouched, never guessed at;
#   - a pip reinstall of Open WebUI silently restores 2000, which is why this
#     runs at every start and not once.
# Open WebUI serves _app/immutable with an ETag and no Cache-Control. Desktop
# browsers normally revalidate it on reload, but an installed mobile PWA can
# keep the old bytes because the content-hashed URL and SvelteKit app version
# did not change. reconcile_frontend_version below publishes a content-derived
# version after these patches land, using SvelteKit's own update path.
# --------------------------------------------------------------------------- #

# The literal as the bundle carries it, and the anchor that makes the match
# safe: the interruption check sits ~150 bytes before it in the same closure.
_CALL_SILENCE_ANCHOR = "voiceInterruption"
_CALL_SILENCE_PATTERN = re.compile(
    r"(Date\.now\(\)-[A-Za-z_$][\w$]*>)(\d+(?:e\d+)?)(&&\([A-Za-z_$][\w$]*=!0,)"
)
_CALL_SILENCE_WINDOW = 600


def find_call_silence_chunk(frontend_dir: Path) -> Path | None:
    """The one bundle chunk that carries the Call overlay, or None.

    Located by content (the anchor plus the pattern) rather than by name,
    because the chunk's filename is a content hash that changes every release.
    """
    chunks = frontend_dir / "_app" / "immutable" / "chunks"
    if not chunks.is_dir():
        return None
    for path in sorted(chunks.glob("*.js")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if patch_call_silence(text, CALL_SILENCE_UPSTREAM_MS)[1] is not None:
            return path
    return None


def patch_call_silence(text: str, silence_ms: int) -> tuple[str, int | None]:
    """(new_text, previous_ms) with the Call overlay's silence literal rewritten.

    Pure. previous_ms is None when the pattern is not present, in which case
    new_text is text unchanged. When the literal already equals silence_ms the
    text is returned unchanged with previous_ms == silence_ms, so callers can
    tell "already right" from "not found".
    """
    anchor = text.find(_CALL_SILENCE_ANCHOR)
    while anchor != -1:
        window = text[anchor : anchor + _CALL_SILENCE_WINDOW]
        match = _CALL_SILENCE_PATTERN.search(window)
        if match:
            previous = int(float(match.group(2)))
            if previous == silence_ms:
                return text, previous
            start = anchor + match.start(2)
            end = anchor + match.end(2)
            return text[:start] + str(silence_ms) + text[end:], previous
        anchor = text.find(_CALL_SILENCE_ANCHOR, anchor + 1)
    return text, None


def frontend_build_dir(settings: Settings) -> Path:
    """Open WebUI's compiled frontend inside its dedicated venv."""
    return (
        _webui_venv_scripts(settings).parent
        / "Lib"
        / "site-packages"
        / "open_webui"
        / "frontend"
    )


def reconcile_call_silence(settings: Settings) -> str:
    """Make the installed bundle's Call-mode wait match openwebui.call_silence_ms.

    Returns a human line, never raises. Runs while Open WebUI is stopped for
    consistency with its siblings, though the file is static and a running
    server would serve the new bytes on the next reload anyway.
    """
    wanted = settings.openwebui.call_silence_ms
    frontend = frontend_build_dir(settings)
    if not frontend.is_dir():
        return "Open WebUI frontend not found; Call-mode wait left alone"
    chunk = find_call_silence_chunk(frontend)
    if chunk is None:
        # Nothing to restore either: if a previous LOCITIZE rewrote the literal
        # the pattern would still match, so absence means a bundle this does not
        # understand, and the honest move is to say so.
        return (
            "Open WebUI bundle does not carry the Call-mode wait where expected; "
            "left alone (openwebui.call_silence_ms has no effect on this build)"
        )
    try:
        text = chunk.read_text(encoding="utf-8")
        new_text, previous = patch_call_silence(text, wanted)
        if previous == wanted:
            if wanted == CALL_SILENCE_UPSTREAM_MS:
                return "Open WebUI Call-mode wait at its default 2000ms"
            return f"Open WebUI Call-mode wait already {wanted}ms"
        tmp = chunk.with_suffix(chunk.suffix + ".locitize-tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, chunk)
    except OSError as exc:
        return f"could not rewrite the Open WebUI Call-mode wait: {exc}"
    return f"Open WebUI Call-mode wait set to {wanted}ms (was {previous}ms)"


# Open WebUI 0.11.1 creates a fresh Web Audio ``AudioContext`` every time Call
# mode re-arms its ``MediaRecorder``. The old context is never closed. Mobile
# browsers enforce a small limit on live AudioContexts, so a call works for a
# handful of turns and then its analyser silently stops seeing microphone
# samples even though the MediaStream, Whisper, and the rest of the backend are
# still healthy. Close the previous analyser context before creating the next
# one. The async wait matters: creating the replacement before ``close()`` has
# released the browser's audio resource can hit the same limit again.
_CALL_AUDIO_CONTEXT_MARKER = "__locitizeCallAudioContext"
_CALL_AUDIO_CONTEXT_PATTERN = re.compile(
    r"(?P<fn>[A-Za-z_$][\w$]*)=(?P<stream>[A-Za-z_$][\w$]*)=>\{"
    r"const (?P<context>[A-Za-z_$][\w$]*)=new AudioContext,"
    r"(?P<source>[A-Za-z_$][\w$]*)=(?P=context)\.createMediaStreamSource\("
    r"(?P=stream)\)"
)
_CALL_AUDIO_CONTEXT_BEFORE = 1800


def patch_call_audio_context(text: str) -> tuple[str, bool | None]:
    """Close Call mode's previous analyser AudioContext before re-arming.

    Returns ``(new_text, previous)``. ``previous`` is False for the recognized
    upstream leak, True when the repair is already present, and None when the
    Call overlay shape is not recognized.
    """
    marker = f"globalThis.{_CALL_AUDIO_CONTEXT_MARKER}"
    if f"await {marker}?.close()" in text and f"{marker}=new AudioContext" in text:
        return text, True

    anchor = text.find(_CALL_SILENCE_ANCHOR)
    while anchor != -1:
        start = max(0, anchor - _CALL_AUDIO_CONTEXT_BEFORE)
        window = text[start:anchor]
        matches = list(_CALL_AUDIO_CONTEXT_PATTERN.finditer(window))
        if len(matches) == 1:
            match = matches[0]
            fn = match.group("fn")
            stream = match.group("stream")
            context = match.group("context")
            source = match.group("source")
            replacement = (
                f"{fn}=async {stream}=>{{await {marker}?.close();"
                f"const {context}={marker}=new AudioContext,"
                f"{source}={context}.createMediaStreamSource({stream})"
            )
            at = start + match.start()
            end = start + match.end()
            return text[:at] + replacement + text[end:], False
        anchor = text.find(_CALL_SILENCE_ANCHOR, anchor + 1)
    return text, None


def find_call_audio_context_chunk(frontend_dir: Path) -> Path | None:
    """The compiled Call overlay chunk carrying the analyser AudioContext."""
    chunks = frontend_dir / "_app" / "immutable" / "chunks"
    if not chunks.is_dir():
        return None
    for path in sorted(chunks.glob("*.js")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if patch_call_audio_context(text)[1] is not None:
            return path
    return None


def reconcile_call_audio_context(settings: Settings) -> str:
    """Prevent repeated Open WebUI Call turns exhausting browser audio input."""
    frontend = frontend_build_dir(settings)
    if not frontend.is_dir():
        return "Open WebUI frontend not found; Call microphone lifecycle left alone"
    chunk = find_call_audio_context_chunk(frontend)
    if chunk is None:
        return (
            "Open WebUI bundle does not carry the Call microphone lifecycle "
            "where expected; left alone"
        )
    try:
        text = chunk.read_text(encoding="utf-8")
        new_text, previous = patch_call_audio_context(text)
        if previous is True:
            return "Open WebUI Call microphone lifecycle fix already applied"
        if previous is None:
            return "Open WebUI Call microphone lifecycle changed; left alone"
        tmp = chunk.with_suffix(chunk.suffix + ".locitize-tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, chunk)
    except OSError as exc:
        return f"could not repair Open WebUI Call microphone lifecycle: {exc}"
    return "Open WebUI Call microphone lifecycle fixed (one AudioContext at a time)"


# Open WebUI 0.11.1 marks the assistant as "speaking" at chat:start, before
# the model has emitted a token or TTS has audio to play. With voice
# interruption enabled, the Call overlay immediately reopens the microphone;
# its first detected room sound calls stopResponse() because chatStreaming is
# already true. A cold model switch therefore gets cancelled during its load
# (live reproduction: 0-3 generated tokens and an assistant row left
# done=false).
#
# An attempted guard armed barge-in only once audio existed, but that still let
# the phone's own speaker output trip the microphone and stop TTS. Keep the
# narrow matcher so installations carrying that LOCITIZE patch can be restored
# to Open WebUI's upstream expression. With voiceInterruption defaulted off,
# upstream Call mode suppresses the mic while the assistant is speaking.
_CALL_BARGE_IN_SPEAKING_PATTERN = re.compile(
    r"r\(([A-Za-z_$][\w$]*)\)\|\|r\(([A-Za-z_$][\w$]*)\)&&!\(\(\("
)
_CALL_BARGE_IN_SOUND_PATTERN = re.compile(
    r"([A-Za-z_$][\w$]*\.some\([A-Za-z_$][\w$]*=>"
    r"[A-Za-z_$][\w$]*>0\))"
)
_CALL_BARGE_IN_BEFORE = 220
_CALL_BARGE_IN_AFTER = 800


def patch_call_barge_in(text: str, enabled: bool = False) -> tuple[str, bool | None]:
    """Enable or remove LOCITIZE's experimental Call-mode barge-in guard.

    ``enabled=False`` is the production path and restores the upstream bundle.
    Returns ``(new_text, previous)`` where previous reports whether the guard
    was present, or None when the bundle shape is not recognized.
    """
    anchor = text.find(_CALL_SILENCE_ANCHOR)
    while anchor != -1:
        start = max(0, anchor - _CALL_BARGE_IN_BEFORE)
        end = min(len(text), anchor + _CALL_BARGE_IN_AFTER)
        window = text[start:end]
        speaking = _CALL_BARGE_IN_SPEAKING_PATTERN.search(window)
        sound = _CALL_BARGE_IN_SOUND_PATTERN.search(window)
        if speaking and sound and speaking.start() < anchor - start < sound.start():
            assistant_speaking = speaking.group(2)
            guard = (
                f'(!r({assistant_speaking})||speechSynthesis.speaking||'
                '!document.getElementById("audioElement")?.paused)'
            )
            suffix = window[sound.end() :]
            if suffix.startswith(f"&&{guard}&&("):
                if enabled:
                    return text, True
                remove_at = start + sound.end()
                return text[:remove_at] + text[remove_at + len(f"&&{guard}") :], True
            if suffix.startswith("&&("):
                if not enabled:
                    return text, False
                insert_at = start + sound.end()
                return text[:insert_at] + f"&&{guard}" + text[insert_at:], False
        anchor = text.find(_CALL_SILENCE_ANCHOR, anchor + 1)
    return text, None


def find_call_barge_in_chunk(frontend_dir: Path) -> Path | None:
    """The compiled Call overlay chunk carrying the false-interrupt path."""
    chunks = frontend_dir / "_app" / "immutable" / "chunks"
    if not chunks.is_dir():
        return None
    for path in sorted(chunks.glob("*.js")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if patch_call_barge_in(text)[1] is not None:
            return path
    return None


def reconcile_call_barge_in(settings: Settings) -> str:
    """Remove LOCITIZE's feedback-prone experimental barge-in guard."""
    frontend = frontend_build_dir(settings)
    if not frontend.is_dir():
        return "Open WebUI frontend not found; Call-mode interruption left alone"
    chunk = find_call_barge_in_chunk(frontend)
    if chunk is None:
        return (
            "Open WebUI bundle does not carry the Call-mode interruption path "
            "where expected; left alone"
        )
    try:
        text = chunk.read_text(encoding="utf-8")
        new_text, previous = patch_call_barge_in(text, enabled=False)
        if previous is False:
            return "Open WebUI Call-mode interruption guard already removed"
        if previous is None:
            return "Open WebUI Call-mode interruption path changed; left alone"
        tmp = chunk.with_suffix(chunk.suffix + ".locitize-tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, chunk)
    except OSError as exc:
        return f"could not restore Open WebUI Call-mode interruption path: {exc}"
    return "Open WebUI Call-mode interruption guard removed"


# Open WebUI 0.11.1 creates each fetched TTS clip as its own ``Audio`` object,
# but CallOverlay copies only that object's URL into a shared ``#audioElement``
# and asks the shared element to play while muted. Mobile autoplay policy can
# reject that play (or pause it when the overlay unmutes), even though the
# already-created clip is playable. The rejected play used to leave Call mode
# silent while STT, inference, and TTS all continued to return HTTP 200.
#
# Use the fetched Audio object directly. If a browser still requires activation,
# retry synchronously inside the next pointer/key gesture. Track the active clip
# and its completion callback so ending/interruption cannot wedge the speaking
# loop. This matcher is intentionally limited to the one CallOverlay closure and
# leaves an unknown future Open WebUI build untouched.
_CALL_AUDIO_MARKER = "__locitizeCallAudio"
_CALL_AUDIO_ID = r"[A-Za-z_$][\w$]*"
_CALL_AUDIO_PLAY_PATTERN = re.compile(
    rf'const (?P<audio>{_CALL_AUDIO_ID})=document\.getElementById\("audioElement"\);'
    rf'if\(!(?P=audio)\)\{{(?P<resolve>{_CALL_AUDIO_ID})\(null\);return\}}'
    rf'(?P<middle>.{{0,900}}?);(?P=audio)\.src=(?P<clip>{_CALL_AUDIO_ID})\.src,'
    rf'(?P=audio)\.muted=!0,(?P<setup>.{{0,900}}?),(?P=audio)\.play\(\)'
    rf'\.then\(\(\)=>\{{(?P=audio)\.muted=!1\}}\)'
    rf'\.catch\((?P<error>{_CALL_AUDIO_ID})=>\{{'
    rf'(?P<settle>{_CALL_AUDIO_ID})\((?P=error)\)\}}\)'
)
_CALL_AUDIO_STOP_PATTERN = re.compile(
    rf'const (?P<audio>{_CALL_AUDIO_ID})=document\.getElementById\("audioElement"\);'
    rf'(?P=audio)&&\((?P=audio)\.muted=!0,(?P=audio)\.pause\(\),'
    rf'(?P=audio)\.currentTime=0\)'
)


def patch_call_audio_playback(text: str) -> tuple[str, bool | None]:
    """Route Call-mode TTS through its fetched Audio object.

    Returns ``(new_text, previous)``. ``previous`` is False for the recognized
    upstream shared-element path, True when this patch is already present, and
    None when the bundle is not recognized.
    """
    if (
        f"globalThis.{_CALL_AUDIO_MARKER}=" in text
        and f"globalThis.{_CALL_AUDIO_MARKER}??" in text
    ):
        return text, True

    matches = list(_CALL_AUDIO_PLAY_PATTERN.finditer(text))
    if len(matches) != 1:
        return text, None
    play = matches[0]
    anchor = text.rfind(_CALL_SILENCE_ANCHOR, max(0, play.start() - 10000), play.start())
    if anchor == -1:
        return text, None

    stop_matches = list(
        _CALL_AUDIO_STOP_PATTERN.finditer(
            text, play.end(), min(len(text), play.end() + 1600)
        )
    )
    if len(stop_matches) != 1:
        return text, None
    stop = stop_matches[0]

    audio = play.group("audio")
    resolve = play.group("resolve")
    clip = play.group("clip")
    error = play.group("error")
    settle = play.group("settle")
    retry = "locitizeRetry"
    play_replacement = (
        f"const {audio}={clip};if(!{audio}){{{resolve}(null);return}}"
        f'{play.group("middle")};'
        f"globalThis.{_CALL_AUDIO_MARKER}={audio},"
        f"globalThis.__locitizeStopCallAudio={settle},{audio}.muted=!1,"
        f'{play.group("setup")},{audio}.play().catch({error}=>{{'
        f'if({error}.name!=="NotAllowedError"){{{settle}({error});return}}'
        f'const {retry}=()=>{{document.removeEventListener("pointerdown",{retry}),'
        f'document.removeEventListener("keydown",{retry}),'
        f"globalThis.__locitizeRetryCallAudio=null,{audio}.play().catch({settle})}};"
        f"globalThis.__locitizeRetryCallAudio={retry},"
        f'document.addEventListener("pointerdown",{retry},{{once:!0}}),'
        f'document.addEventListener("keydown",{retry},{{once:!0}})}})'
    )

    stop_audio = stop.group("audio")
    stop_replacement = (
        f"const {stop_audio}=globalThis.{_CALL_AUDIO_MARKER}??"
        'document.getElementById("audioElement");'
        "globalThis.__locitizeRetryCallAudio&&("
        'document.removeEventListener("pointerdown",'
        "globalThis.__locitizeRetryCallAudio),"
        'document.removeEventListener("keydown",'
        "globalThis.__locitizeRetryCallAudio)),"
        "globalThis.__locitizeRetryCallAudio=null,"
        "globalThis.__locitizeStopCallAudio?.(),"
        f"{stop_audio}&&({stop_audio}.muted=!0,{stop_audio}.pause(),"
        f"{stop_audio}.currentTime=0),"
        f"globalThis.{_CALL_AUDIO_MARKER}=null,"
        "globalThis.__locitizeStopCallAudio=null"
    )

    # Later replacement first keeps the earlier match offsets valid.
    new_text = text[: stop.start()] + stop_replacement + text[stop.end() :]
    new_text = new_text[: play.start()] + play_replacement + new_text[play.end() :]
    return new_text, False


def find_call_audio_chunk(frontend_dir: Path) -> Path | None:
    """The compiled Call overlay chunk carrying the TTS playback path."""
    chunks = frontend_dir / "_app" / "immutable" / "chunks"
    if not chunks.is_dir():
        return None
    for path in sorted(chunks.glob("*.js")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if patch_call_audio_playback(text)[1] is not None:
            return path
    return None


def reconcile_call_audio_playback(settings: Settings) -> str:
    """Make Open WebUI Call-mode TTS audible under mobile autoplay policy."""
    frontend = frontend_build_dir(settings)
    if not frontend.is_dir():
        return "Open WebUI frontend not found; mobile Call audio left alone"
    chunk = find_call_audio_chunk(frontend)
    if chunk is None:
        return (
            "Open WebUI bundle does not carry the mobile Call audio path "
            "where expected; left alone"
        )
    try:
        text = chunk.read_text(encoding="utf-8")
        new_text, previous = patch_call_audio_playback(text)
        if previous is True:
            return "Open WebUI mobile Call audio fix already applied"
        if previous is None:
            return "Open WebUI mobile Call audio path changed; left alone"
        tmp = chunk.with_suffix(chunk.suffix + ".locitize-tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, chunk)
    except OSError as exc:
        return f"could not repair Open WebUI mobile Call audio: {exc}"
    return "Open WebUI mobile Call audio fixed (fetched clip plays directly)"


_LOCITIZE_FRONTEND_VERSION = re.compile(r"\+locitize\.[0-9a-f]{12}$")


def _version_runtime_chunk(frontend: Path) -> Path | None:
    """Return SvelteKit's version-polling runtime chunk, when unambiguous."""
    chunks = frontend / "_app" / "immutable" / "chunks"
    if not chunks.is_dir():
        return None
    matches: list[Path] = []
    for path in sorted(chunks.glob("*.js")):
        try:
            if "_app/version.json" in path.read_text(
                encoding="utf-8", errors="replace"
            ):
                matches.append(path)
        except OSError:
            continue
    return matches[0] if len(matches) == 1 else None


def reconcile_frontend_version(settings: Settings) -> str:
    """Publish a cache identity for LOCITIZE's patched Open WebUI frontend.

    SvelteKit polls ``_app/version.json`` with ``no-cache`` once per minute.
    A normal build changes both that value and hashed asset names, but LOCITIZE
    deliberately makes surgical post-install edits to one compiled chunk.
    Give those bytes a content-derived app version and update the matching
    compiled runtime constant. A stale PWA then detects the version mismatch,
    refreshes, and revalidates the changed ETag instead of serving its old Call
    overlay forever.

    The runtime is replaced first and version.json last: the externally visible
    update marker is never published before the fresh runtime is on disk. Both
    writes are atomic, and a failed marker write restores the runtime.
    """
    import hashlib
    import json

    frontend = frontend_build_dir(settings)
    version_path = frontend / "_app" / "version.json"
    if not version_path.is_file():
        return "Open WebUI app version file not found; PWA cache identity left alone"
    call_chunk = find_call_silence_chunk(frontend)
    runtime = _version_runtime_chunk(frontend)
    if call_chunk is None or runtime is None:
        return (
            "Open WebUI frontend version layout was not recognized; "
            "PWA cache identity left alone"
        )

    runtime_original = ""
    runtime_changed = False
    try:
        version_raw = version_path.read_text(encoding="utf-8")
        payload = json.loads(version_raw)
        current = str(payload.get("version", "")).strip()
        if not current:
            return "Open WebUI app version is empty; PWA cache identity left alone"
        base = _LOCITIZE_FRONTEND_VERSION.sub("", current)
        digest = hashlib.sha256(call_chunk.read_bytes()).hexdigest()[:12]
        wanted = f"{base}+locitize.{digest}"

        runtime_original = runtime.read_text(encoding="utf-8")
        runtime_new = runtime_original
        wanted_literal = json.dumps(wanted)
        if wanted_literal not in runtime_original:
            candidates = []
            for value in dict.fromkeys((current, base)):
                literal = json.dumps(value)
                if runtime_original.count(literal) == 1:
                    candidates.append(literal)
            if len(candidates) != 1:
                return (
                    "Open WebUI runtime version constant was not recognized; "
                    "PWA cache identity left alone"
                )
            runtime_new = runtime_original.replace(
                candidates[0], wanted_literal, 1
            )

        payload["version"] = wanted
        version_new = json.dumps(payload, separators=(",", ":"))
        if runtime_new == runtime_original and version_new == version_raw:
            return f"Open WebUI PWA cache identity already {wanted}"

        if runtime_new != runtime_original:
            runtime_tmp = runtime.with_suffix(runtime.suffix + ".locitize-version-tmp")
            runtime_tmp.write_text(runtime_new, encoding="utf-8")
            os.replace(runtime_tmp, runtime)
            runtime_changed = True

        try:
            version_tmp = version_path.with_suffix(
                version_path.suffix + ".locitize-version-tmp"
            )
            version_tmp.write_text(version_new, encoding="utf-8")
            os.replace(version_tmp, version_path)
        except OSError:
            if runtime_changed:
                rollback = runtime.with_suffix(
                    runtime.suffix + ".locitize-version-rollback"
                )
                rollback.write_text(runtime_original, encoding="utf-8")
                os.replace(rollback, runtime)
            raise
    except (OSError, ValueError, TypeError) as exc:
        return f"could not update the Open WebUI PWA cache identity: {exc}"
    return f"Open WebUI PWA cache identity set to {wanted}"




def openwebui_package_version(settings: Settings) -> str | None:
    """Installed open-webui version in the launcher .webui-venv, or None."""
    import subprocess
    import sys

    py = webui_venv_python(settings)
    if not py.is_file():
        return None
    try:
        proc = subprocess.run(
            [
                str(py),
                "-c",
                "import importlib.metadata as m; print(m.version('open-webui'))",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    ver = (proc.stdout or "").strip().splitlines()
    return ver[-1].strip() if ver else None


def upgrade_openwebui(settings: Settings, *, timeout_s: int = 600) -> str:
    """Install the pinned open-webui version in the venv the launcher runs.

    Explicit action only - never called on a normal start. Installs
    setup_env.OPENWEBUI_VERSION (a reviewed pin), not whatever PyPI calls latest.

    Uses that venv's `python -m pip install -U open-webui` so a checkout and
    any system-drive junction to it resolve to the same site-packages the child uses.
    Stops open-webui processes first so Windows does not lock `open-webui.exe`.
    Fail-soft: on any error return a clear log line and leave whatever install
    is present so boot can still start Open WebUI.
    """
    import subprocess
    import sys

    py = webui_venv_python(settings)
    if not py.is_file():
        return (
            f"Open WebUI pinned install skipped: venv python missing at {py}"
        )
    # Unlock Scripts/open-webui.exe before pip rewrites it.
    venv_root = py.parent.parent
    ensure_single_openwebui_processes(
        settings.ports.openwebui, force=True, venv_dir=venv_root
    )
    before = openwebui_package_version(settings) or "unknown"
    from setup_env import OPENWEBUI_VERSION

    try:
        proc = subprocess.run(
            [str(py), "-m", "pip", "install", f"open-webui=={OPENWEBUI_VERSION}"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Open WebUI pinned install timed out after {timeout_s}s "
            f"(was {before}); starting previous install from {venv_root}"
        )
    except OSError as exc:
        return (
            f"Open WebUI pinned install failed to launch pip ({exc}); "
            f"starting previous install {before} from {venv_root}"
        )
    after = openwebui_package_version(settings) or before
    # Pip may return non-zero when only the console .exe rewrite is locked, even
    # after the wheel itself landed. Prefer the installed version as truth.
    if after != before and after != "unknown":
        note = ""
        if proc.returncode != 0:
            note = " (pip warned; package version advanced)"
        return (
            f"Open WebUI pinned install: {before} -> {after} in {venv_root}{note}"
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = next(
            (line for line in reversed(tail) if line and "notice" not in line.lower()),
            f"exit {proc.returncode}",
        )
        return (
            f"Open WebUI pinned install FAILED ({detail}); "
            f"keeping {before} at {venv_root}"
        )
    return f"Open WebUI pinned install: already latest ({after}) in {venv_root}"



def wait_loopback_port_free(port: int, *, timeout_s: float = 15.0, poll_s: float = 0.2) -> bool:
    """True once nothing is accepting TCP on 127.0.0.1:port (post force-kill)."""
    import socket
    import time

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            try:
                sock.connect(("127.0.0.1", int(port)))
                # Still accepting — keep waiting for the dying listener to exit.
            except OSError:
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)


def wait_openwebui_healthy(
    port: int,
    *,
    timeout_s: float = 60.0,
    poll_s: float = 0.5,
    path: str = OPENWEBUI_HEALTH_PATH,
) -> bool:
    """Poll http://127.0.0.1:port{path} until HTTP 200 or timeout.

    Used after upgrade/start so boot never quietly leaves Serve pointed at a
    dead :8096. Prefer /health; callers may pass /api/config as a fallback path.
    """
    import time
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{int(port)}{path}"
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if int(getattr(resp, "status", 200) or 200) == 200:
                    return True
        except (OSError, urllib.error.HTTPError, urllib.error.URLError, ValueError):
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)


def ensure_single_openwebui_processes(
    port: int, *, force: bool = False, venv_dir: Path | str | None = None
) -> str:
    """Kill duplicate Open WebUI processes so only one can own the loopback port.

    Returns a short human line for the launcher log. Uses psutil (already a
    platform dependency). Only processes running from LOCITIZE's own
    ``venv_dir`` (the .webui-venv) are candidates, so a user's separate Open
    WebUI install, a Docker container or an editor open on an open-webui
    checkout is never touched. Without ``venv_dir`` nothing is killed.
    ``port`` is logged for operators only.
    """
    if venv_dir is None:
        return "Open WebUI process check skipped (no locitize venv given)"
    try:
        import psutil
    except ImportError:
        return "Open WebUI process check skipped (psutil unavailable)"

    root = os.path.normcase(os.path.abspath(str(venv_dir))).rstrip("\/") + os.sep
    markers = ("open-webui", "open_webui", "open-webui.exe")
    matched: list = []
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            argv = proc.info.get("cmdline") or []
            cmdline = " ".join(argv).lower()
            if not (any(m in name for m in markers) or any(m in cmdline for m in markers)):
                continue
            paths = [proc.info.get("exe") or ""] + list(argv[:2])
            if any(
                p and os.path.normcase(os.path.abspath(p)).startswith(root)
                for p in paths
            ):
                matched.append(proc)
        except (psutil.Error, PermissionError):
            continue

    if len(matched) == 0:
        return (
            f"Open WebUI process check: 0 instance(s) on/around :{port}"
        )
    if len(matched) == 1 and not force:
        return (
            f"Open WebUI process check: 1 instance(s) on/around :{port}"
        )

    killed: list[int] = []
    for proc in matched:
        try:
            pid = int(proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                proc.kill()
            killed.append(pid)
        except (psutil.Error, ProcessLookupError, PermissionError):
            continue
    return (
        f"Open WebUI process check: killed {len(killed)} duplicate PID(s) "
        f"{killed} so :{port} has a single listener"
    )


def build_openwebui_spec(settings: Settings, log_path: str | None = None) -> ServiceSpec:
    """Build the Open WebUI ServiceSpec from settings (the kokoro_server analog).

    The command is fully data-driven: the DEDICATED .webui-venv interpreter (never
    sys.executable), the `open_webui serve` entry, the loopback host, and the
    reserved openwebui port. The env wires Open WebUI to the running llama.cpp
    OpenAI-compatible backend, points its private DATA_DIR under the platform tree,
    and DISABLES the first-run RAG embedding-model network download by default
    (Architecture M9.2, Permission Matrix section 9). Readiness uses the per-service
    openwebui.ready_timeout_s (300s) because the first boot runs database migrations.

    Raises ValueError with a concrete remedy when the .webui-venv interpreter is
    missing, so an uninstalled Open WebUI is an honest failure the chooser degrades
    on, never a crash.
    """
    console = webui_console_script(settings)
    if not console.is_file():
        raise ValueError(
            "Open WebUI is not installed: the dedicated venv console script "
            f"'{console}' does not exist. Create it and install Open WebUI "
            "(see docs/chat.md): python -m venv Codebase/.webui-venv && "
            "Codebase/.webui-venv/Scripts/pip install open-webui"
        )

    port = settings.ports.openwebui
    data_dir = webui_data_dir(settings)
    # The child creates its own store on first run; ensure the parent tree exists so
    # DATA_DIR points at a real, writable, in-tree location.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Confirmed R1-style against open-webui 0.10.2 (`open-webui serve --help`): the
    # serve subcommand accepts --host and --port. Loopback host is fixed (never
    # 0.0.0.0), matching every other LOCITIZE service. The console script is the
    # dedicated venv's, so the child never runs in the platform venv.
    command = [
        str(console),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    env = _build_openwebui_env(settings, data_dir)

    return ServiceSpec(
        name=OPENWEBUI_NAME,
        command=command,
        # Never None: a None cwd makes the child inherit LOCITIZE's own working
        # directory, which locitize.bat sets to the install tree (invariant W1).
        cwd=resolve_service_cwd(settings),
        env=env,
        port=port,
        health_path=OPENWEBUI_HEALTH_PATH,  # 200 once first-run migration completes
        log_path=log_path,
        ready_timeout_s=float(settings.openwebui.ready_timeout_s),
        stop_timeout_s=settings.services.stop_timeout_s,
        # Append (dated separator per start) so a failed first boot does not destroy
        # the previous session's migration/error trace, matching kokoro/whisper.
        append_log=True,
        # M17.10: Open WebUI's uvicorn startup spawns short-lived console child
        # processes; under the default CREATE_NO_WINDOW each would flash its own
        # console window on first launch. A hidden console lets them attach to it
        # instead, killing the black flash the owner saw on first "Open Chat".
        hidden_console=True,
    )


def _build_openwebui_env(settings: Settings, data_dir: Path) -> dict[str, str]:
    """Assemble the child-process environment for Open WebUI (confirmed R1-style).

    Every value here was confirmed against the installed Open WebUI release at build
    time (recorded in Builder Verification.md), not guessed:

    - DATA_DIR                 : its private store, under the gitignored platform tree.
    - OPENAI_API_BASE_URL/URLS : the running llama.cpp OpenAI-compatible endpoint.
      Both the singular (older) and plural (newer) names are set so the wiring holds
      across releases without guessing which one this build reads.
    - OPENAI_API_KEY           : a NON-EMPTY dummy. llama.cpp ignores the key but Open
      WebUI requires a non-empty value; it is a placeholder, not a secret (Data Model
      section 10), so it may live in the spec env.
    - WEBUI_AUTH=False         : disable the account/sign-up wall for local single-user
      use (Tailscale Serve-path boot default). Auth-off signin still uses
      admin@localhost with password literal 'admin' — webui.db must store a
      bcrypt hash of that literal string.
    - WEBUI_URL                : Tailscale Serve root HTTPS URL (https://<magicdns>,
      no port). LOCITIZE_WEBUI_URL if set, else MagicDNS, else the loopback Open
      WebUI address when Tailscale is unavailable. :4443 remains an alt Serve URL but is not WEBUI_URL.
    - ENABLE_LOGIN_FORM=False  : hide login form (pairs with WEBUI_AUTH=False).
    - WEBUI_*_COOKIE_SECURE=True + SameSite=lax : required behind Serve HTTPS.
    - FORWARDED_ALLOW_IPS      : 127.0.0.1 - Tailscale Serve proxies from loopback,
                                 so nothing else is trusted to forward headers.
    - CORS_ALLOW_ORIGIN        : only Open WebUI's own addresses. Its default is
                                 "*" with credentials, which with auth off would
                                 let any website a user visits sign in as admin.
    - ENABLE_COMMUNITY_SHARING / ENABLE_VERSION_UPDATE_CHECK = False: no chat
                                 sharing to openwebui.com, no update ping.
    - RAG_EMBEDDING_ENGINE / *_AUTOMATIC_UPDATE / OFFLINE : disable the first-run
      sentence-transformers embedding download when disable_embedding_fetch is true
      (the default), so no model is fetched over the network without owner approval.
      HF_HUB_OFFLINE hard-stops any Hugging Face fetch as a belt-and-suspenders guard.
    """
    base_url = backend_base_url(settings)
    env: dict[str, str] = {
        "DATA_DIR": str(data_dir),
        "OPENAI_API_BASE_URL": base_url,
        "OPENAI_API_BASE_URLS": base_url,
        # Placeholder, not a credential (llama.cpp ignores it; Open WebUI needs it set).
        "OPENAI_API_KEY": OPENAI_PLACEHOLDER_KEY,
        # Single-user local use: do not force account creation to chat locally.
        "WEBUI_AUTH": "False",
        # Internet access (owner request 2026-08-13): built-in web search with the
        # keyless DuckDuckGo engine so chats can pull live web context. Both the
        # current and legacy spellings are set (same both-names pattern as
        # OPENAI_API_BASE_URL/S above). NOTE: on an already-initialized DATA_DIR
        # these are seed defaults only - the persisted web.search.* rows in
        # webui.db win, so the 2026-08-13 change also flipped those rows directly.
        # Owner request 2026-09-02: picking a model in the picker can now trigger
        # a real multi-gigabyte load behind this request (see router.py), and a
        # cold 27B legitimately exceeds Open WebUI's shorter default. Sized off
        # services.ready_timeout_s plus the generation itself rather than a round
        # number. The model LIST keeps its own short timeout: router.py answers
        # /v1/models from the registry and never loads anything, so a slow
        # listing would mean something is wrong, not something is loading.
        "AIOHTTP_CLIENT_TIMEOUT": "600",
        "AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST": "10",
        "ENABLE_WEB_SEARCH": "True",
        "WEB_SEARCH_ENGINE": "duckduckgo",
        "ENABLE_RAG_WEB_SEARCH": "True",
        "RAG_WEB_SEARCH_ENGINE": "duckduckgo",
    }
    # WEBUI_URL is the Serve ROOT host (no :4443). Always set - see
    # resolve_webui_serve_url for the override / MagicDNS / loopback order.
    from tailscale_phone import resolve_webui_serve_url

    env["WEBUI_URL"] = resolve_webui_serve_url(openwebui_port=settings.ports.openwebui)
    # Auth-off + no login form; Secure cookies + forwarded headers for Serve HTTPS.
    env["ENABLE_LOGIN_FORM"] = "False"
    env["WEBUI_SESSION_COOKIE_SAME_SITE"] = "lax"
    env["WEBUI_AUTH_COOKIE_SAME_SITE"] = "lax"
    env["WEBUI_SESSION_COOKIE_SECURE"] = "True"
    env["WEBUI_AUTH_COOKIE_SECURE"] = "True"
    env["FORWARDED_ALLOW_IPS"] = "127.0.0.1"
    # Browsers may only call Open WebUI from Open WebUI's own pages. With auth
    # off, the upstream default ("*" plus credentials) would let any website
    # obtain an admin session from a page the user merely visits.
    owui_port = int(settings.ports.openwebui)
    origins = [
        env["WEBUI_URL"],
        f"http://127.0.0.1:{owui_port}",
        f"http://localhost:{owui_port}",
    ]
    # Chat opens https://<secure_proxy.hostname> (locitize.local) when that
    # name serves Open WebUI. Its live-update socket checks this same list, so
    # without the name a page opened there sends messages but never receives
    # a reply.
    hostname = (getattr(getattr(settings, "secure_proxy", None), "hostname", "") or "").strip()
    if hostname:
        origins.append(f"https://{hostname}")
    env["CORS_ALLOW_ORIGIN"] = ";".join(dict.fromkeys(origins))
    env["ENABLE_COMMUNITY_SHARING"] = "False"
    env["ENABLE_VERSION_UPDATE_CHECK"] = "False"
    if settings.openwebui.disable_embedding_fetch:
        # Suppress the first-run RAG embedding-model network download (Permission
        # Matrix section 9). Confirmed against open-webui 0.10.2 env.py/config.py:
        # OFFLINE_MODE is the master switch (parsed as a lowercased == "true"
        # comparison, so it MUST be the string "true", not "1"). When true it forces
        # HF_HUB_OFFLINE=1 and disables RAG_EMBEDDING_MODEL_AUTO_UPDATE /
        # RAG_RERANKING_MODEL_AUTO_UPDATE (config.py lines 989/1016), so no
        # sentence-transformers weights are fetched over the network. The explicit
        # AUTO_UPDATE=False and HF_HUB_OFFLINE=1 are belt-and-suspenders in case a
        # future release changes how OFFLINE_MODE cascades.
        env["OFFLINE_MODE"] = "true"
        env["HF_HUB_OFFLINE"] = "1"
        env["RAG_EMBEDDING_MODEL_AUTO_UPDATE"] = "False"
        env["RAG_RERANKING_MODEL_AUTO_UPDATE"] = "False"
    return env
