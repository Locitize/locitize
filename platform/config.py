"""Typed configuration loading for the LOCITIZE platform.

This module parses settings.yaml and models.yaml into typed dataclasses and
applies the three-layer override chain (Architecture section 7, Data Model
sections 1-2):

    code defaults  <  settings.yaml / models.yaml  <  LOCITIZE_* environment vars

It imports nothing from the rest of the platform so it sits at the bottom of the
dependency graph.

Where config lives (Architecture M14.2, DEC-M14-4). The repo ships only
templates - settings.default.yaml and models.default.yaml - and never reads them
at runtime. The live files belong to the user and live in the data root that
resolve_data_dir() finds (LOCITIZE_DATA_DIR, then a portable locitize-data/ folder,
then %LOCALAPPDATA%\\LOCITIZE). Calling Config.load() with no base_dir seeds that
root on first run via ensure_user_config(), which only ever copies: it never
moves, edits or deletes a file the user already has.

Security notes (Permission Matrix sections 1-2):
- Config file paths resolve from the data root (or an explicitly supplied
  directory); code and user data stay separated.
- yaml.safe_load only (never yaml.load) so no arbitrary object construction.
- Secrets never come from yaml; only LOCITIZE_* env vars carry machine-specific
  paths and any future tokens. settings_to_dict(redact=True) masks them in any
  dump.
- services.llama_cpp_health_path is validated to be a rooted path with no scheme
  or authority component so the readiness probe can never leave loopback
  (Security SEC-1); the loopback host itself is hardcoded in services.py.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable

import yaml

# The platform directory: the CODE root. It stays the code root and nothing
# user-owned is written here from M14 onward (Architecture M14.2.3).
BASE_DIR = Path(__file__).resolve().parent

SETTINGS_FILE = "settings.yaml"
MODELS_FILE = "models.yaml"

# The tracked templates that ship in the repo. They are copied into the data
# root on first run and are never read as live config (Architecture M14.2.2).
DEFAULT_SETTINGS_FILE = "settings.default.yaml"
DEFAULT_MODELS_FILE = "models.default.yaml"

# Data-root resolution (Architecture M14.2.3). See resolve_data_dir().
DATA_DIR_ENV = "LOCITIZE_DATA_DIR"
PORTABLE_DIR_NAME = "locitize-data"
# Windows environment variable NAME only - never an expanded value, so no
# machine-specific path ever enters the source (Architecture M14.2.1).
LOCAL_APPDATA_ENV = "LOCALAPPDATA"
APP_DIR_NAME = "LOCITIZE"

# settings.yaml schema versions this reader understands. 2 is current (it is the
# version the data-root layout ships with); 1 is the pre-M14 layout and still
# parses unchanged, so downgrading to an older LOCITIZE stays survivable.
CURRENT_SETTINGS_VERSION = 2
SUPPORTED_SETTINGS_VERSIONS = (1, 2)

# Valid enumerations, single-sourced for validation.
VALID_MODEL_STATUS = ("installed", "future", "disabled")
VALID_PORT_ALLOCATION = ("auto", "strict")
VALID_NOISE_SUPPRESSION = ("off", "balanced", "strong")


# --------------------------------------------------------------------------- #
# Load report (a lightweight HealthReport-style list of parse issues).
# Kept independent of health.py to avoid a circular import; the launcher folds
# these into the status panel.
# --------------------------------------------------------------------------- #


@dataclass
class ConfigIssue:
    """One configuration parse warning or error."""

    level: str  # "WARNING" or "ERROR"
    source: str  # file name the issue came from
    message: str


# --------------------------------------------------------------------------- #
# settings.yaml typed structure
# --------------------------------------------------------------------------- #


@dataclass
class PathsConfig:
    """Binary/venv paths. Empty string means "not configured yet" (legal).

    whisper points at whisper-server.exe (the HTTP transcription service);
    whisper_model is the ggml .bin weights that server loads; whisper_stream is
    the SDL2 mic-streaming binary (whisper-stream.exe) used by the --listen path.
    Each is an absolute path or empty string; the matching LOCITIZE_* env var
    overrides the file (Data Model section 2), and a missing path degrades to a
    health/service WARNING with a remedy, never a crash.
    """

    llama_cpp: str = ""
    whisper: str = ""
    whisper_model: str = ""
    whisper_stream: str = ""
    kokoro: str = ""
    # M6 (voice OUT): the on-disk Kokoro PyTorch checkpoint and the directory
    # holding the eight voice .pt tensors. Empty string means "not configured"
    # (legal; the kokoro health probe degrades to a WARNING, not a crash). The
    # matching LOCITIZE_KOKORO_MODEL_PATH / LOCITIZE_KOKORO_VOICES_PATH env vars
    # override the file value. Never a secret -- just filesystem paths.
    kokoro_model: str = ""
    kokoro_voices: str = ""
    venv: str = ""
    # M22: the shared, tool-agnostic model store LOCITIZE creates at setup,
    # scans for imports, and links downloads into. Deliberately outside any
    # one app's tree so an uninstall never takes the user's models along.
    model_store: str = ""  # "" = the standard shared store (see setup_env.model_store_path)


@dataclass
class PortsConfig:
    """Loopback port assignments; all must stay within [range_start, range_end]."""

    allocation: str = "auto"
    range_start: int = 8080
    range_end: int = 8099
    llama_cpp: int = 8080
    whisper: int = 8091
    kokoro: int = 8092
    vision: int = 8094  # future service
    scheduler: int = 8095  # future service
    # M9-lite (Data Model section 10): the Open WebUI managed chat service. 8094 is
    # nominally reserved for the future vision service, so 8096 is the next genuinely
    # free slot inside the reserved 8080-8099 range. Bound to 127.0.0.1 only.
    openwebui: int = 8096
    # Owner request 2026-09-02: the model router (router.py), the
    # OpenAI-compatible front that lists the whole registry and switches the
    # served model on demand. 8093 is the first genuinely free slot inside the
    # reserved 8080-8099 range (8091/8092 are whisper/kokoro, 8094/8095 are held
    # for the vision and scheduler services). Bound to 127.0.0.1 only.
    router: int = 8093


@dataclass
class ThresholdsConfig:
    """Health-probe thresholds. hard_min values are the FAIL floor."""

    ram_floor_mb: int = 8192
    ram_hard_min_mb: int = 4096
    disk_floor_mb: int = 20480
    disk_hard_min_mb: int = 2048
    vram_headroom_mb: int = 512
    python_min: str = "3.11"


@dataclass
class ServicesConfig:
    """Service lifecycle timeouts and the llama.cpp readiness endpoint."""

    ready_timeout_s: float = 60.0
    stop_timeout_s: float = 10.0
    llama_cpp_health_path: str = "/health"
    # M4/GUI (Data Model 6.1): when true, models.build_start_spec appends
    # --metrics so the GUI live monitor can read llama.cpp's Prometheus /metrics
    # endpoint. An owner whose llama-server build rejects the flag sets this false
    # and the monitor degrades honestly (Architecture G5) instead of the model
    # failing to start.
    llama_cpp_metrics: bool = True


@dataclass
class LoggingConfig:
    """Logging levels, rotation sizes, and the required subsystem channels."""

    level_console: str = "INFO"
    level_file: str = "DEBUG"
    max_bytes: int = 5_000_000
    backup_count: int = 5
    channels: list[str] = field(
        default_factory=lambda: [
            "launcher",
            "assistant",
            "speech",
            "llm",
            "tts",
            "benchmark",
            "errors",
        ]
    )


@dataclass
class SpeechConfig:
    """whisper-stream live-capture tuning (Milestone 3, defect D-M3-2).

    stream_step_ms=0 selects whisper-stream's VAD sliding-window mode: instead of
    transcribing every fixed 3s window (which makes the model hallucinate stock
    phrases such as "Thank you." on pure-silence windows -- an artifact of its
    video training data), it waits for a speech-then-silence boundary and
    transcribes only the detected utterance. That is the primary silence gate for
    D-M3-2. vad_thold / freq_thold are the tool's voice-activity detector knobs
    (higher vad_thold = stricter speech detection; freq_thold is the high-pass
    cutoff in Hz that rejects low-frequency rumble). The defaults match the
    whisper-stream binary's own defaults, verified via its --help, and are exposed
    here so an owner can tune sensitivity for their mic without a code change.
    noise_suppression selects the router-side FFmpeg filter used for Open WebUI
    recordings. It is deliberately an enum, never arbitrary filter syntax.
    """

    stream_step_ms: int = 0
    vad_thold: float = 0.60
    freq_thold: float = 100.0
    noise_suppression: str = "balanced"


@dataclass
class TtsConfig:
    """Kokoro text-to-speech (voice OUT) settings (Data Model 9.1, M6).

    voice must name an on-disk voice .pt (the owner's six plus af_nicole/af_sky);
    an unknown voice degrades to the default with a WARNING log rather than
    failing. speed is Kokoro's speech-rate multiplier (must be > 0).
    sample_sentence is the fixed line --audition speaks in each voice. autoplay
    false makes speak() synthesize a wav without playing it (used by headless
    tests and synthesize-only flows), so no audio hardware is ever required.
    None of these hold secrets.
    """

    enabled: bool = True
    voice: str = "am_michael"
    speed: float = 1.0
    sample_sentence: str = "Hello, I am locitize. This is how this voice sounds."
    autoplay: bool = True
    # Leading silence (ms) prepended to the FIRST clip of a spoken reply so the
    # audio device can spin up without clipping the first phoneme. 120 ms is
    # enough for wired/USB output. Bluetooth speakers that sleep between sounds
    # may need 800-1500; raise this setting rather than waiting on every reply.
    reply_lead_silence_ms: int = 120


@dataclass
class AssistantConfig:
    """Built-in voice-assistant loop settings (Data Model 9.1, M7).

    system_prompt is the leading `system` ChatMessage. response_reserve_tokens is
    the ctx headroom kept for the reply when trimming history (M7.3; must be > 0).
    max_history_turns is a secondary hard cap on retained turns before the token
    trim (>= 1). speak_per_sentence true streams and speaks each sentence as it is
    generated (M7.4); false synthesizes the whole reply then speaks. capture_mode
    (D-M7-6) is 'push_to_talk' (DEFAULT: mic feeds the assistant only inside an
    Enter-opened window) or 'continuous' (legacy always-listening, fragile).
    max_capture_s / utterance_settle_s bound one push_to_talk window. stt_mode and
    listen_window_s are legacy tuning fields kept for compatibility.
    half_duplex_tail_s is the half-duplex mic-gate tail (D-M7-3): while the assistant
    is speaking AND for this many seconds after the last audio finishes, the mic
    source discards captured segments so the assistant never re-hears its own TTS
    through the speakers and converses with itself. It is only the cheap first-line
    (poll-time) filter now; correctness comes from capture-time interval overlap.
    half_duplex_guard_s is the +/- jitter margin (D-M7-3b) added to each recorded
    speaking interval when the mic source tests whether a segment's absolute capture
    window overlaps it, absorbing timestamp/anchor read-latency jitter (see
    assistant.HalfDuplexGate.overlaps_speaking). None hold secrets.
    """

    system_prompt: str = "You are locitize, a concise local voice assistant."
    response_reserve_tokens: int = 512
    max_history_turns: int = 20
    speak_per_sentence: bool = True
    stt_mode: str = "vad"
    listen_window_s: float = 15.0
    # D-M7-6: voice-mode mic capture strategy. 'push_to_talk' (the DEFAULT) feeds the
    # assistant only during a window the owner opens with Enter -- this structurally
    # removes the always-listening failures (ambient hallucination and self-echo
    # runaway) the owner hit in AC17. 'continuous' preserves the old always-listening
    # behaviour and is documented as fragile (--listen-continuous overrides this).
    capture_mode: str = "push_to_talk"
    # push_to_talk: hard cap on one capture window in seconds (a silent room ends the
    # turn with a notice instead of hanging), and the inter-segment silence that marks
    # end-of-utterance once the owner has started speaking.
    max_capture_s: float = 15.0
    utterance_settle_s: float = 0.7
    half_duplex_tail_s: float = 0.8
    # Guard margin (seconds) added to each side of a recorded speaking interval when
    # deciding capture-time overlap (D-M7-3b). Default 0.5s comfortably exceeds the
    # ~0.3s log poll interval, the dominant source of anchor read-latency jitter.
    half_duplex_guard_s: float = 0.5


@dataclass
class MemoryConfig:
    """Conversation-memory store settings (Data Model 9.1/9.3, M8.3).

    enabled false disables all append/recall. dir is resolved UNDER the platform
    base_dir (like logs/ and docs/) and can never escape it (memory.resolve_memory_dir
    enforces this). auto_recall true injects the recent-N transcript into each turn's
    context; recall_recent_n / search_limit bound how much is pulled. None hold
    secrets -- transcripts are local personal data, never transmitted.
    """

    enabled: bool = True
    dir: str = "memory"
    auto_recall: bool = False
    recall_recent_n: int = 6
    search_limit: int = 5


@dataclass
class VisionConfig:
    """Single-image vision Q&A settings (Data Model 9.1, M8.1).

    prompt is the default query sent to the vision model when `--describe` is given
    no explicit question. Vision is single-image Q&A only (not video); the flag path
    is data-driven from the model row's optional mmproj field. No secrets here.

    model pins WHICH registry row answers --describe. Empty (the default)
    means "resolve it" - see vision.resolve_vision_model_id, which falls back
    to the rows declaring the vision capability. Added 2026-09-02 after the
    hardcoded vision.VISION_MODEL_ID failed to match any row in a
    scan-imported registry; an owner holding several vision models pins one
    here instead of passing --model every time.
    """

    prompt: str = "Describe this image accurately."
    model: str = ""


@dataclass
class RouterConfig:
    """Model-router settings (owner request 2026-09-02).

    When enabled, webui.backend_base_url points Open WebUI at router.py instead
    of straight at llama-server, so its picker lists every registered model and
    choosing one switches what LOCITIZE serves on the GPU.

    OFF by default on purpose. The router runs INSIDE the process that owns the
    ModelController, so a chat UI pointed at it while no LOCITIZE session is up
    would find nothing listening - whereas the direct llama-server URL keeps
    working for anyone who starts a model by other means. Enabling it is a
    deliberate choice to route chat through LOCITIZE.
    """

    enabled: bool = False
    # Owner request 2026-09-03: add a short system note when a conversation
    # changes model mid-flight, telling the incoming model that the earlier
    # assistant turns are another model's words. Without it a switched-to model
    # inherits the previous one's persona from the transcript (see
    # router.build_handoff_note for the reproduction). On by default because the
    # inherited answer is simply WRONG; off for anyone comparing models on
    # byte-identical prompts.
    handoff_note: bool = True
    # Owner request 2026-09-03: serve /v1/audio/transcriptions and
    # /v1/audio/speech from the router, translated onto whisper-server and
    # kokoro_server, so Open WebUI's microphone and read-aloud buttons work
    # against LOCAL speech services instead of the browser's (which ships the
    # audio to Google). Starts both services with the session so the first mic
    # press does not pay a service start. On by default because the router is
    # itself opt-in; false leaves the audio routes answering an honest 503.
    audio: bool = True
    # Owner request 2026-09-03: recognise a chat request that Open WebUI's Call
    # overlay built from a transcript the router itself just returned, and for
    # that one turn switch thinking off and ask for a spoken register (see
    # router.apply_voice_turn). A reasoning model otherwise sits silent for its
    # whole reasoning before the first word is heard. On by default because it
    # touches only turns that came in by voice; false forwards them untouched.
    voice_turns: bool = True


@dataclass
class GuiConfig:
    """Tkinter command-center behaviour (Data Model 6.1, Architecture G3/G5).

    monitor_interval_s is the live-monitor poll cadence (the monitor thread's
    wait() interval); the owner-sensible range is 1.0-2.0 seconds. monitor_enabled
    false hides the live-monitor panel entirely. chat_open_browser false makes the
    Chat button surface the loopback URL as text instead of launching a browser.
    None of these hold secrets.
    """

    monitor_interval_s: float = 1.5
    monitor_enabled: bool = True
    chat_open_browser: bool = True


@dataclass
class ProxyConfig:
    """locitize.local reverse-proxy settings (Data Model 6.1, Architecture G4).

    The proxy is OFF by default. When enabled it binds 127.0.0.1:<port> only
    (loopback, no elevation for the default 8085) and forwards to the active
    model's port. bind_port_80 is an explicit opt-in that STILL requires the owner
    to run LOCITIZE elevated; a non-elevated port-80 bind fails with a remedy and is
    never silently downgraded. hostname requires a manual/elevated hosts-file line
    that the platform never writes itself (Permission Matrix section 7).
    """

    enabled: bool = False
    port: int = 8085
    hostname: str = "locitize.local"
    bind_port_80: bool = False


@dataclass
class SecureProxyConfig:
    """Caddy TLS reverse-proxy settings for https://locitize.local (owner request
    2026-08-14).

    Unlike ProxyConfig (the built-in llama.cpp forwarder), this supervises an
    external Caddy binary that terminates TLS on 127.0.0.1:443 with Caddy's
    local CA and proxies to the Open WebUI port, so the browser shows a padlock
    instead of "Not secure". LOCITIZE ensures the whole chain on chat launch:
    discover (or winget-install, user scope, no elevation) caddy.exe, write the
    Caddyfile if missing, start caddy if 443 is closed, and trust the local root
    CA in the current-user store via certutil (which may show the owner one
    Windows consent dialog - never silent, never machine-wide). The hosts-file
    line (locitize.local -> 127.0.0.1) still requires a one-time elevated write the
    platform never performs itself (Permission Matrix section 7); ensure reports
    it as a remedy when missing.

    - caddy_path: explicit caddy.exe path; empty = auto-discover (PATH, then
      the winget user-scope package directory)
    - caddyfile: the config file LOCITIZE materializes and points caddy at
    - auto_install: allow a user-scope `winget install CaddyServer.Caddy` when
      caddy is not found (one-time, no elevation)
    """

    # OFF by default (M14.2.6). Enabling this lets LOCITIZE winget-install software,
    # bind 443, and add a root certificate to the current-user store. None of
    # that may be a side effect of merely starting the app for someone who never
    # asked for TLS, so it is opt-in and stays opt-in.
    enabled: bool = False
    hostname: str = "locitize.local"
    caddy_path: str = ""
    caddyfile: str = ""
    auto_install: bool = True


@dataclass
class LauncherConfig:
    """Launcher behaviour flags.

    default_model is EMPTY by default (M14.2.6): a code default naming a model
    id is a claim that a specific file exists on this machine, which is false on
    every machine but the one it was written on. Empty means "no default yet",
    which the launcher reports honestly instead of failing to start a model
    nobody has.
    """

    auto_journal: bool = True
    default_model: str = ""


@dataclass
class ChatConfig:
    """Chat-UI chooser preference (Data Model section 10, M9-lite).

    preferred_ui is the chooser's persisted default: 'openwebui' (the DEFAULT -
    open the rich Open WebUI chat, degrading honestly to the built-in UI when it
    is not installed), 'llamacpp' (skip straight to the built-in llama.cpp web
    UI), or 'ask' (present the choice each time). An unknown value falls
    back to 'ask' with a WARNING at load. Written back by config.write_chat_ui (a
    targeted atomic write, the sibling of write_model_fields). No secrets.
    """

    preferred_ui: str = "openwebui"


@dataclass
class ChatHarnessConfig:
    """"Launch in..." coding-harness picker settings (owner request 2026-08-21).

    last_project_dir is the folder the Claude Code / Codex / OpenCode launch
    options operate on, remembered across launches so the folder picker
    defaults to it instead of asking every time (the owner can still browse
    to a different folder on any given launch). Empty means "never launched
    one yet" -- the picker's folder dialog opens with no default. Written
    back by config.write_chat_harness_dir, the direct sibling of
    write_chat_ui.
    """

    last_project_dir: str = ""


@dataclass
class OpenWebUIConfig:
    """Open WebUI managed-service settings (Data Model section 10, M9-lite).

    Open WebUI is an external chat application LOCITIZE installs into an ISOLATED venv
    and supervises with the same SingleServiceController lifecycle as every other
    managed service. None of these hold secrets: the placeholder OPENAI_API_KEY the
    child receives is a non-empty dummy the loopback llama.cpp server ignores.

    - enabled: false (the DEFAULT) removes Open WebUI from the chooser, so only the
      built-in llama.cpp UI is offered until the owner opts in
    - venv: dedicated venv dir under Codebase/ whose interpreter runs the child
      (NEVER the platform venv, whose pins Open WebUI's deps would clash with)
    - data_dir: DATA_DIR under Codebase/platform/ for its sqlite DB, chat history,
      uploads; gitignored and private exactly like memory/ (owner chat data)
    - ready_timeout_s: readiness window for the slow first boot (DB migration), longer
      than the global 60s (cf. the 27B model's 240s cold-load window)
    - backend_base_url: "" derives http://127.0.0.1:<ports.llama_cpp>/v1 at start; a
      non-empty value overrides (advanced), but only ever a loopback URL, never a token
    - disable_embedding_fetch: true (default-safe) suppresses the first-run RAG
      embedding-model network download; false is the owner-approved fetch
    - call_silence_ms: how long Call mode waits after you stop talking before it
      sends what it heard (2000 = Open WebUI's own hard-coded value = no change).
      Owner request 2026-09-03 ("talk to my models like ChatGPT"): every voice
      turn measured 0.9s of work behind a fixed 2.0s of waiting, so the wait was
      the latency. A value other than 2000 rewrites that constant in the
      installed bundle at Open WebUI start (webui.reconcile_call_silence);
      300..5000
    - voice_interruption: Call mode keeps the mic open while the reply plays so
      speaking over it stops the audio. True is the default. False is half-duplex.
    """

    # Default false, matching settings.default.yaml (Review L-3, DEC-M14-1's
    # demotion of Open WebUI to opt-in). It matters that the CODE default agrees
    # with the shipped template and not just the template: a user whose
    # settings.yaml has no openwebui: block at all gets this value, and the old
    # True meant LOCITIZE would offer to install separately licensed third-party
    # software to someone who never asked for it - the opposite of the shipped
    # intent, reached by the one path nobody looks at.
    enabled: bool = False
    venv: str = ".webui-venv"
    data_dir: str = "webui-data"
    ready_timeout_s: int = 300
    backend_base_url: str = ""
    disable_embedding_fetch: bool = True
    call_silence_ms: int = 2000
    # Call-mode barge-in: the microphone stays open while Kokoro speaks so a
    # real utterance can stop the reply. Open WebUI already requests
    # echoCancellation on getUserMedia. Speaker-phone echo can still false-
    # trigger; a headset is the reliable path. False restores half-duplex.
    voice_interruption: bool = True


# Open WebUI's own Call-mode end-of-speech wait, and the range LOCITIZE will
# rewrite it within. Below 300 the recorder cuts utterances at every breath;
# above 5000 nothing is gained over leaving the bundle alone.
CALL_SILENCE_UPSTREAM_MS = 2000
CALL_SILENCE_MIN_MS = 300
CALL_SILENCE_MAX_MS = 5000


@dataclass
class FineTuneConfig:
    """Fine-tune studio + model-discovery settings (M13, Data Model section 13).

    LOCITIZE supervises an EXTERNAL fine-tuning studio (a Streamlit app in its own
    checkout, running on its own interpreter) the same way it supervises Open
    WebUI, and separately scans that studio's outputs/ tree for finished .gguf
    models. Every default here is safe and off, and no machine-specific absolute
    path is committed: the owner sets studio_dir locally or by env var.

    - enabled: false -> the Fine-tune page renders an honest disabled state and no
      child process can be started
    - studio_dir: absolute path to the llm-finetune-studio checkout (READ-ONLY to
      LOCITIZE: it is only read and launched, never written to)
    - outputs_dir: blank -> <studio_dir>/outputs
    - python: blank -> autodetect the studio's own venv interpreter
    - app_path: the Streamlit entry script, relative to studio_dir
    - port / ready_timeout_s: the loopback port and its readiness window
    - autostart: RESERVED, NOT IMPLEMENTED. No module reads this key; LOCITIZE never
      starts the studio merely because LOCITIZE started, and setting it true changes
      nothing today. Kept as a declared key so a future implementation does not
      have to introduce it as a breaking settings change.
    - open_browser: open the studio URL in the owner's browser on the Open action
    - discovery_enabled: independent of `enabled`, so the owner can list and serve
      their fine-tunes without ever starting Streamlit (the common case)
    - default_context_size / default_gpu_layers: the serving defaults a discovered
      model gets, since it has no models.yaml row to read them from
    - active_run_window_s: how recently a run's train.log must have been written
      for LOCITIZE to treat that training run as live (gates the honest
      container-runtime limitation warning on stop/close)

    No secrets: nothing in this block is or carries a credential.
    """

    enabled: bool = False
    studio_dir: str = ""
    outputs_dir: str = ""
    python: str = ""
    app_path: str = "app/app.py"
    port: int = 8501
    ready_timeout_s: int = 60
    autostart: bool = False
    open_browser: bool = True
    discovery_enabled: bool = True
    default_context_size: int = 8192
    default_gpu_layers: int = 999
    active_run_window_s: int = 120


@dataclass
class ModelsHubConfig:
    """Model acquisition settings (M14.14.8, Data Model section 14).

    Drives the Models page's "Get models" section: browsing a shipped catalog,
    searching HuggingFace on an explicit press, and downloading one verified
    GGUF into the data root's models/ directory.

    Every default here is safe and machine-independent, so M14.2's "no
    machine-specific default ships" rule is satisfied by construction:
    download_dir ships blank (meaning <data root>/models) and catalog_path ships
    blank (meaning the installed model_catalog.json).

    - enabled: false -> the Get models section renders a disabled state and no
      network call is reachable at all
    - api_base / allowed_hosts: the one host family LOCITIZE will talk to. Every
      redirect hop is re-validated against allowed_hosts, and a host is accepted
      only if it equals or is a subdomain of an entry (HuggingFace redirects real
      downloads to *.hf.co CDN hosts, so the subdomain rule is required).
      allowed_hosts is deliberately NOT env-overridable: an environment variable
      that widens a security allowlist is a worse mechanism than editing a file
      the user owns.
    - min_free_disk_headroom_mb: free space required BEYOND the model's own size
      before the first byte is written, so a download can never fill the disk.
    - register_on_complete: the default state of the confirm dialog's "add to my
      model list when finished" checkbox. The write is still one explicit user
      action - the user pressed Download with the box ticked.

    No secrets: this block holds no credential, and the product supports no
    HuggingFace account of any kind (M14.14.6 T5).
    """

    enabled: bool = True
    api_base: str = "https://huggingface.co"
    allowed_hosts: list[str] = field(
        default_factory=lambda: ["huggingface.co", "hf.co"]
    )
    search_limit: int = 100
    search_timeout_s: int = 15
    catalog_path: str = ""
    download_dir: str = ""
    min_free_disk_headroom_mb: int = 2048
    # RESERVED, NOT IMPLEMENTED: resuming a cancelled download is designed
    # (Architecture M14.14.3) but not built, so nothing reads this key yet.
    # Ships false so the declared default matches the real behaviour.
    resume_enabled: bool = False
    register_on_complete: bool = True


@dataclass
class Settings:
    """The whole platform settings tree plus the resolved base directory."""

    version: int = CURRENT_SETTINGS_VERSION
    paths: PathsConfig = field(default_factory=PathsConfig)
    ports: PortsConfig = field(default_factory=PortsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    secure_proxy: SecureProxyConfig = field(default_factory=SecureProxyConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    chat_harness: ChatHarnessConfig = field(default_factory=ChatHarnessConfig)
    openwebui: OpenWebUIConfig = field(default_factory=OpenWebUIConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    models_hub: ModelsHubConfig = field(default_factory=ModelsHubConfig)
    # base_dir is not read from yaml; it is the platform (code) directory used to
    # resolve logs/ and docs/. Excluded from any config dump.
    base_dir: Path = field(default=BASE_DIR)
    # data_dir is the resolved user data root (Architecture M14.2.3): where the
    # live settings.yaml and models.yaml were read from and are written back to.
    # Not read from yaml either. Left as None it follows base_dir (see
    # __post_init__), so a caller that points Settings at one self-contained
    # directory - every test, and Config.load(base_dir=...) - gets config read and
    # written in that same directory, never in the real user data root.
    data_dir: Path | None = None

    def __post_init__(self) -> None:
        """Default data_dir to base_dir so the two can never silently diverge.

        This matters for safety, not tidiness: if data_dir fell back to the
        installed platform directory, a test that built Settings(base_dir=tmp)
        and then saved a model edit would write into the real, live models.yaml.
        """
        if self.data_dir is None:
            self.data_dir = self.base_dir
        elif self.data_dir == BASE_DIR and self.base_dir != BASE_DIR:
            # dataclasses.replace(settings, base_dir=other) copies the OLD
            # data_dir across, so a Settings redirected at a temp directory would
            # otherwise keep pointing config writes at the installed platform
            # directory. A data_dir still sitting on the installed default was
            # never chosen deliberately, so it follows base_dir too. An explicit
            # data_dir (the real data root, or any other directory) is untouched.
            self.data_dir = self.base_dir


# --------------------------------------------------------------------------- #
# models.yaml typed structure
# --------------------------------------------------------------------------- #


@dataclass
class Model:
    """One row of the model registry (Data Model section 1.1)."""

    id: str
    name: str
    description: str
    location: str
    context_size: int
    gpu_layers: int
    recommended_prompt: str = ""
    benchmark_score: float | None = None
    notes: str = ""
    status: str = "installed"
    quantization: str = ""
    vram_estimate_mb: int = 0
    server_args: list[str] = field(default_factory=list)
    # M8.1 (vision): optional absolute path to a multimodal projector gguf (e.g.
    # mmproj-BF16.gguf). When set, models.build_start_spec appends `--mmproj <path>`
    # to the llama-server argv (flag spelling confirmed against the installed binary
    # --help at Build time: `-mm, --mmproj FILE`), enabling multimodal serving for
    # that model. None -> a text-only chat model exactly as before. Never a URL/token.
    mmproj: str | None = None
    # M18.1 (backend seam): which inference engine serves this model. Absent ->
    # "llama-cpp" (the built-in), so every existing registry row is unchanged.
    # Resolved at spec-build time via backends.get_backend, which refuses an
    # unknown name with the registered list rather than guessing flags.
    backend: str = ""
    # D-M4-2: optional per-model readiness timeout in seconds. A large model on a
    # cold load legitimately needs longer than the global services.ready_timeout_s
    # (the 27B exceeded 60s cold, so it false-failed). None means "fall back to the
    # global default"; a set value must be a positive integer (validated at load).
    ready_timeout_s: int | None = None
    # M5.6 lever (b): optional speculative-decoding draft model. Either the id of
    # another registry model (resolved to its location at spec-build time) or a
    # direct filesystem path to a small draft gguf. None -> no draft speculation.
    draft_model: str | None = None
    # M5.6: optional speculation knobs (spec_type, draft_max, draft_min, ngram_*).
    # A mapping of scalar values; the flag SPELLINGS live single-sourced in
    # models.py (confirmed against the binary --help), never in the yaml, so a
    # build variance is a code-constant edit. None -> speculation off.
    spec_config: dict[str, Any] | None = None
    # Owner request 2026-09-02: thinking/reasoning control, first-classed for the
    # same reason spec_config was - the flags are real, but leaving them to raw
    # server_args means a typo is discovered by a jinja exception at REQUEST time
    # rather than at load. Optional mapping with three scalar keys (effort,
    # budget, enabled); flag SPELLINGS live single-sourced in models.py, never
    # in the yaml. None -> nothing appended, so llama-server keeps its own
    # defaults and every existing row starts byte-identically.
    #
    # The EFFORT VOCABULARY IS DELIBERATELY NOT VALIDATED HERE, because it is a
    # property of the model's own chat template, not of llama-server. Verified
    # 2026-09-02 on this machine: Qwen3.8-27B-UD-IQ4_XS's template accepts only
    # ('xhigh', 'medium', 'low') and raise_exception()s on anything else, while
    # silently aliasing 'high' UP to 'xhigh' - so a hardcoded allowlist here
    # would reject levels that other templates require. LOCITIZE checks the
    # SHAPE and lets the template be the authority on the value.
    reasoning: dict[str, Any] | None = None
    # M5.4: optional declared sweep axes for this model (gpu_layers/context_size/
    # server_args/spec lists). Absent -> the benchmark runs only the current-config
    # scenario. Consumed by benchmark.SweepPlan; not used at model-start time.
    benchmark_sweep: dict[str, Any] | None = None
    # M13: where this row came from. "registry" for every row parsed out of
    # models.yaml (the loader never reads this field from yaml, so a hand-written
    # row cannot claim to be discovered), "discovered" for a row synthesized by
    # finetune.to_model from a .gguf found under the studio's outputs/ tree. The
    # UI shows it as the Source column; the controller uses it to refuse edits
    # that would need a models.yaml block the discovered row does not have.
    source: str = "registry"
    # The run folder a discovered model came from (None for a manual row).
    source_run: str | None = None
    # M14.14.5: the sha256 LOCITIZE computed when it downloaded this file. Empty for
    # every hand-added row and every row that predates the downloader, which is
    # what makes this purely additive - no migration, no settings.version bump,
    # and every existing models.yaml keeps parsing byte-identically. It exists so
    # a later verify-only pass can re-check a file the user still has. Provenance
    # (repo, filename, date, verification rung) stays in `notes` rather than
    # becoming three more columns.
    sha256: str = ""
    # Owner request 2026-08-21: what this model is actually good for, so the
    # owner can tell at a glance which one to switch to. Free-form tags (e.g.
    # "vision", "reasoning", "coding") set explicitly in the row - never
    # inferred/guessed here, so the column never claims a capability the owner
    # did not assert. "voice" is deliberately NOT a tag: every chat model works
    # with Talk (speech in/out is a separate pipeline stage - Whisper/Kokoro -
    # not a property of the LLM), so tagging some models "voice" and not others
    # would be a fabricated distinction. Empty -> the UI shows a plain "Chat"
    # default rather than a blank cell.
    capabilities: list[str] = field(default_factory=list)


def compute_vram_need_mb(vram_estimate_mb: int, size_bytes: int | None) -> float:
    """The basis "On GPU"/"On CPU" judge fit against: the on-disk .gguf file
    size when it is known, else the owner's estimate as a before-download
    preview (owner fix, 2026-08-16, second pass: "On GPU" showing a bigger
    number than "Size" on the same row read as a flat contradiction -- a
    vision model's vram_estimate_mb includes its separate mmproj file plus
    KV-cache overhead, which is real VRAM usage but not part of "the model" in
    the sense the Size column already answers, so this must never exceed Size
    once Size has a real number to disagree with).

    A first pass took max(vram_estimate_mb, size_bytes) so a model whose
    raw weights exceed the card (qwen3-8-27b-q4km, tuned to gpu_layers < full
    specifically so the estimate undersold the true weight size) still showed
    real spillage instead of a tautological "(fits)". That case is still
    caught here: size_bytes is used whenever a file exists, since it is always
    >= the estimate for a model where the OWNER'S TUNING, not extra overhead,
    is what shrank the estimate. The estimate is now used ONLY when there is
    no file on disk yet (nothing for it to contradict).

    Lives here (not in gui_controller) so the GUI's fit columns and health.py's
    VramProbe answer "how much VRAM does this model need" the same way. A
    scan-imported registry carries vram_estimate_mb: 0 on every row, so a probe
    reading the estimate alone sees nothing at all (defect found 2026-09-01:
    VramProbe reported "tight for the largest model (0MB)" on a 20-model
    registry, its FAIL branch structurally unreachable).
    """
    if size_bytes:
        return size_bytes / 1_000_000
    return vram_estimate_mb or 0


def model_vram_need_mb(model: "Model") -> float:
    """compute_vram_need_mb for a registry row, reading the file if it exists.

    A missing/unreadable location is not an error here - it falls back to the
    row's estimate, exactly as a not-yet-downloaded model does.
    """
    size_bytes: int | None = None
    location = getattr(model, "location", "") or ""
    if location:
        try:
            size_bytes = Path(location).stat().st_size
        except OSError:
            size_bytes = None
    return compute_vram_need_mb(model.vram_estimate_mb, size_bytes)


@dataclass
class ModelRegistryData:
    """Parsed models.yaml: a schema version plus the list of models."""

    version: int = 1
    models: list[Model] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Data root resolution and first-run config seeding (Architecture M14.2.3)
# --------------------------------------------------------------------------- #


def resolve_data_dir(
    env: dict[str, str] | None = None, install_dir: Path | str | None = None
) -> Path:
    """Return the single directory that holds everything the user owns.

    Resolution order, first hit wins (Architecture M14.2.3):

    1. the LOCITIZE_DATA_DIR environment variable, if set to a non-empty value -
       the explicit override, for a user who wants their data on another drive;
    2. <install dir>/locitize-data/, but only IF that directory already exists -
       the portable / dev-checkout mode. Creating the folder is the entire
       opt-in, which makes it one mkdir to keep everything inside a checkout or
       on a USB stick;
    3. %LOCALAPPDATA%\\LOCITIZE - the default for a normal install.

    LOCALAPPDATA, deliberately not APPDATA: this tree holds logs, transcripts,
    chat databases, fetched binaries, GGUF weights and training outputs - tens of
    gigabytes of machine-local data. APPDATA (Roaming) is synchronised on
    domain/enterprise profiles, so putting it there would be actively harmful.

    Nothing is created here; this function only computes a path. If LOCALAPPDATA
    is not set (a non-Windows host, or a stripped environment), the last resort
    is <install dir>/locitize-data so the caller always gets a usable, writable-by-
    intent location rather than an exception.
    """
    environ = env if env is not None else dict(os.environ)
    root = Path(install_dir) if install_dir is not None else BASE_DIR

    explicit = (environ.get(DATA_DIR_ENV) or "").strip()
    if explicit:
        return Path(explicit).expanduser()

    portable = root / PORTABLE_DIR_NAME
    if portable.is_dir():
        return portable

    local_appdata = (environ.get(LOCAL_APPDATA_ENV) or "").strip()
    if local_appdata:
        return Path(local_appdata) / APP_DIR_NAME

    return portable


def resolve_models_dir(settings: "Settings") -> Path:
    """Return the one directory downloaded model files may be written into.

    Blank models_hub.download_dir (the shipped default) means <data root>/models,
    which keeps every downloaded byte inside the single folder M14.2.3 made the
    user's whole backup story. An explicitly configured download_dir is honoured
    as-is so a user can put tens of gigabytes on another drive.

    This resolves a location; it creates nothing and validates nothing. The
    caller confines the final file path with modelhub.resolve_destination, which
    is where the escape-proofing actually lives.
    """
    configured = str(getattr(settings.models_hub, "download_dir", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(settings.data_dir) / "models"


@dataclass(frozen=True)
class SeedAction:
    """One record of what ensure_user_config() did, for logging and tests.

    kind is "copied-legacy" (an existing pre-M14 file beside the code was copied
    into the data root), "copied-template" (the shipped .default.yaml was used as
    the seed), or "kept" (a live file was already there and was left alone).
    """

    filename: str
    kind: str
    source: str


def ensure_user_config(
    data_dir: Path | str, install_dir: Path | str | None = None
) -> list[SeedAction]:
    """Make sure the data root holds a live settings.yaml and models.yaml.

    This is the M14 migration, and it is non-destructive by construction
    (Architecture M14.2.2). For each of the two files:

    - if the data root already has it, nothing happens at all;
    - else if a pre-M14 file sits beside the code (the old location), it is
      COPIED into the data root. The original is not moved, not deleted and not
      edited - it is left byte-identical, so an older LOCITIZE on the same machine
      keeps working and a mistake here costs the user nothing;
    - else the shipped .default.yaml template is copied in as the seed.

    Nothing is rewritten during the copy, including the `version:` line: the
    version 2 reader parses a version 1 file unchanged, so there is no need to
    touch a file the user hand-maintains. Returns what it did, so a caller can
    log it honestly.
    """
    data = Path(data_dir)
    code = Path(install_dir) if install_dir is not None else BASE_DIR
    data.mkdir(parents=True, exist_ok=True)

    actions: list[SeedAction] = []
    for live_name, template_name in (
        (SETTINGS_FILE, DEFAULT_SETTINGS_FILE),
        (MODELS_FILE, DEFAULT_MODELS_FILE),
    ):
        target = data / live_name
        if target.exists():
            actions.append(SeedAction(live_name, "kept", str(target)))
            continue
        legacy = code / live_name
        template = code / template_name
        if legacy.is_file():
            shutil.copy2(legacy, target)
            actions.append(SeedAction(live_name, "copied-legacy", str(legacy)))
        elif template.is_file():
            shutil.copy2(template, target)
            actions.append(SeedAction(live_name, "copied-template", str(template)))
    return actions


class Config:
    """Namespace for the load entry points. No instances, no mutable state."""

    @staticmethod
    def load(
        base_dir: Path | str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[Settings, ModelRegistryData, list[ConfigIssue]]:
        """Load settings + models and apply the override layering.

        Two modes, on purpose:

        - `base_dir` given (every test, and any caller that wants a
          self-contained directory): that directory IS both the code root and
          the config location. Nothing is resolved, nothing is seeded, nothing
          outside it is touched.
        - `base_dir` omitted (the real application): the code root is this
          module's directory and the config comes from the resolved data root
          (Architecture M14.2.3), which is seeded non-destructively on first run
          by ensure_user_config().

        `env` lets tests inject a fake environment instead of os.environ, keeping
        both the data-root resolution and the override logic unit-testable.
        Returns (settings, models, issues) where issues is the parse report.
        """
        environ = env if env is not None else dict(os.environ)
        issues: list[ConfigIssue] = []

        if base_dir is not None:
            base = Path(base_dir)
            data = base
        else:
            base = BASE_DIR
            data = resolve_data_dir(environ, base)
            try:
                ensure_user_config(data, base)
            except OSError as exc:
                # An unwritable data root is a real, reportable condition, not a
                # crash: the loader falls back to the code defaults and says why.
                issues.append(
                    ConfigIssue(
                        "ERROR",
                        SETTINGS_FILE,
                        f"could not prepare the data directory: {exc}",
                    )
                )

        settings = _load_settings(data, issues)
        # Preserve the code root (logs/docs) and record where config came from.
        settings = replace(settings, base_dir=base, data_dir=data)
        models = _load_models(data, issues)

        _apply_env_overrides(settings, models, environ, issues)
        _validate(settings, models, issues)
        return settings, models, issues


def _load_settings(base: Path, issues: list[ConfigIssue]) -> Settings:
    """Parse settings.yaml over code defaults; missing file is a WARNING only."""
    path = base / SETTINGS_FILE
    defaults = Settings()
    if not path.exists():
        issues.append(
            ConfigIssue("WARNING", SETTINGS_FILE, "not found; using code defaults")
        )
        return defaults
    raw = _safe_load(path, issues)
    if not isinstance(raw, dict):
        if raw is not None:
            issues.append(
                ConfigIssue("ERROR", SETTINGS_FILE, "top level is not a mapping")
            )
        return defaults

    return Settings(
        version=_int(raw.get("version"), defaults.version),
        paths=_build(PathsConfig, raw.get("paths"), SETTINGS_FILE, issues),
        ports=_build(PortsConfig, raw.get("ports"), SETTINGS_FILE, issues),
        thresholds=_build(
            ThresholdsConfig, raw.get("thresholds"), SETTINGS_FILE, issues
        ),
        services=_build(ServicesConfig, raw.get("services"), SETTINGS_FILE, issues),
        logging=_build(LoggingConfig, raw.get("logging"), SETTINGS_FILE, issues),
        speech=_build(SpeechConfig, raw.get("speech"), SETTINGS_FILE, issues),
        tts=_build(TtsConfig, raw.get("tts"), SETTINGS_FILE, issues),
        assistant=_build(
            AssistantConfig, raw.get("assistant"), SETTINGS_FILE, issues
        ),
        memory=_build(MemoryConfig, raw.get("memory"), SETTINGS_FILE, issues),
        vision=_build(VisionConfig, raw.get("vision"), SETTINGS_FILE, issues),
        gui=_build(GuiConfig, raw.get("gui"), SETTINGS_FILE, issues),
        proxy=_build(ProxyConfig, raw.get("proxy"), SETTINGS_FILE, issues),
        router=_build(RouterConfig, raw.get("router"), SETTINGS_FILE, issues),
        secure_proxy=_build(
            SecureProxyConfig, raw.get("secure_proxy"), SETTINGS_FILE, issues
        ),
        launcher=_build(LauncherConfig, raw.get("launcher"), SETTINGS_FILE, issues),
        chat=_build(ChatConfig, raw.get("chat"), SETTINGS_FILE, issues),
        chat_harness=_build(
            ChatHarnessConfig, raw.get("chat_harness"), SETTINGS_FILE, issues
        ),
        openwebui=_build(OpenWebUIConfig, raw.get("openwebui"), SETTINGS_FILE, issues),
        finetune=_build(FineTuneConfig, raw.get("finetune"), SETTINGS_FILE, issues),
        models_hub=_build(
            ModelsHubConfig, raw.get("models_hub"), SETTINGS_FILE, issues
        ),
    )


def _load_models(base: Path, issues: list[ConfigIssue]) -> ModelRegistryData:
    """Parse models.yaml into typed Model rows.

    Accepts both the canonical {version, models: [...]} mapping and a bare list
    (Data Model section 1.1's loose form) so AC4's simple check and the richer
    loader agree.
    """
    path = base / MODELS_FILE
    if not path.exists():
        issues.append(ConfigIssue("ERROR", MODELS_FILE, "not found"))
        return ModelRegistryData()
    raw = _safe_load(path, issues)

    if isinstance(raw, dict):
        version = _int(raw.get("version"), 1)
        rows = raw.get("models", [])
    elif isinstance(raw, list):
        version = 1
        rows = raw
    else:
        issues.append(
            ConfigIssue("ERROR", MODELS_FILE, "expected a mapping or list")
        )
        return ModelRegistryData()

    models: list[Model] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows or []):
        model = _build_model(row, index, issues)
        if model is None:
            continue
        # M13: "ft:" is the reserved namespace for ids GENERATED by fine-tune
        # discovery. Excluding a hand-written row that claims one guarantees a
        # manual id and a generated id can never collide by construction.
        if model.id.startswith("ft:"):
            issues.append(
                ConfigIssue(
                    "WARNING",
                    MODELS_FILE,
                    f"model id '{model.id}' uses the reserved 'ft:' prefix "
                    f"(generated by fine-tune discovery); row ignored",
                )
            )
            continue
        if model.id in seen_ids:
            issues.append(
                ConfigIssue(
                    "WARNING", MODELS_FILE, f"duplicate model id '{model.id}' ignored"
                )
            )
            continue
        seen_ids.add(model.id)
        models.append(model)
    return ModelRegistryData(version=version, models=models)


def _build_model(row: Any, index: int, issues: list[ConfigIssue]) -> Model | None:
    """Turn one raw yaml row into a Model, reporting missing required fields."""
    if not isinstance(row, dict):
        issues.append(
            ConfigIssue("WARNING", MODELS_FILE, f"model #{index} is not a mapping")
        )
        return None
    model_id = str(row.get("id", "")).strip()
    if not model_id:
        issues.append(
            ConfigIssue("WARNING", MODELS_FILE, f"model #{index} missing 'id'")
        )
        return None
    # D-M4-2: a present-but-invalid ready_timeout_s is reported and dropped (falls
    # back to the global default) rather than silently accepted, so a typo like a
    # zero or a string cannot shorten a large model's cold-load window.
    raw_timeout = row.get("ready_timeout_s")
    if raw_timeout is not None and _opt_positive_int(raw_timeout) is None:
        issues.append(
            ConfigIssue(
                "WARNING",
                MODELS_FILE,
                f"model '{model_id}' ready_timeout_s must be a positive integer; "
                f"ignoring and using the global services.ready_timeout_s",
            )
        )
    # Owner request 2026-09-02: reasoning problems are surfaced as WARNINGs and
    # the bad key dropped, matching the ready_timeout_s precedent directly above -
    # a typo must never quietly leave the model at a setting the owner thought
    # they had changed.
    parsed_reasoning, reasoning_problems = parse_reasoning(
        row.get("reasoning"), model_id
    )
    for problem in reasoning_problems:
        issues.append(ConfigIssue("WARNING", MODELS_FILE, problem))
    return Model(
        id=model_id,
        name=str(row.get("name", model_id)),
        description=str(row.get("description", "")),
        location=str(row.get("location", "") or ""),
        context_size=_int(row.get("context_size"), 0),
        gpu_layers=_int(row.get("gpu_layers"), 0),
        recommended_prompt=str(row.get("recommended_prompt", "")),
        benchmark_score=_opt_float(row.get("benchmark_score")),
        notes=str(row.get("notes", "")),
        status=str(row.get("status", "installed")),
        quantization=str(row.get("quantization", "")),
        vram_estimate_mb=_int(row.get("vram_estimate_mb"), 0),
        server_args=[str(a) for a in (row.get("server_args") or [])],
        mmproj=_opt_str(row.get("mmproj")),
        # M18.1: optional engine selector; absent in every existing row.
        backend=str(row.get("backend", "") or "").strip().lower(),
        ready_timeout_s=_opt_positive_int(row.get("ready_timeout_s")),
        draft_model=_opt_str(row.get("draft_model")),
        spec_config=_opt_mapping(row.get("spec_config")),
        benchmark_sweep=_opt_mapping(row.get("benchmark_sweep")),
        reasoning=parsed_reasoning,
        # Absent in every pre-M14 row, which is exactly the additive contract.
        sha256=str(row.get("sha256", "") or ""),
        capabilities=[str(c).strip() for c in (row.get("capabilities") or []) if str(c).strip()],
    )


# --------------------------------------------------------------------------- #
# Environment overrides (highest precedence)
# --------------------------------------------------------------------------- #

# Explicit, documented mapping of env var -> settings path field. Keeping it
# explicit (rather than magic name-mangling) makes the override surface auditable.
_PATH_ENV = {
    "LOCITIZE_LLAMACPP_PATH": "llama_cpp",
    "LOCITIZE_WHISPER_PATH": "whisper",
    "LOCITIZE_WHISPER_MODEL_PATH": "whisper_model",
    "LOCITIZE_WHISPER_STREAM_PATH": "whisper_stream",
    "LOCITIZE_KOKORO_PATH": "kokoro",
    "LOCITIZE_KOKORO_MODEL_PATH": "kokoro_model",
    "LOCITIZE_KOKORO_VOICES_PATH": "kokoro_voices",
    "LOCITIZE_VENV_PATH": "venv",
}

# Env var -> ports field.
_PORT_ENV = {
    "LOCITIZE_PORTS_LLAMACPP": "llama_cpp",
    "LOCITIZE_PORTS_WHISPER": "whisper",
    "LOCITIZE_PORTS_KOKORO": "kokoro",
    "LOCITIZE_PORTS_VISION": "vision",
    "LOCITIZE_PORTS_SCHEDULER": "scheduler",
}


def _apply_env_overrides(
    settings: Settings,
    models: ModelRegistryData,
    environ: dict[str, str],
    issues: list[ConfigIssue],
) -> None:
    """Apply LOCITIZE_* environment overrides in place (env beats file)."""
    for env_key, attr in _PATH_ENV.items():
        if environ.get(env_key):
            setattr(settings.paths, attr, environ[env_key])

    for env_key, attr in _PORT_ENV.items():
        if env_key in environ:
            setattr(settings.ports, attr, _int(environ[env_key], getattr(settings.ports, attr)))

    # M13 fine-tune overrides. Kept explicit (like every other override above) so
    # the whole env surface stays auditable. A malformed port is ignored rather
    # than crashing startup over an env typo; the configured default stands.
    if environ.get("LOCITIZE_FINETUNE_STUDIO_DIR"):
        settings.finetune.studio_dir = environ["LOCITIZE_FINETUNE_STUDIO_DIR"]
    if environ.get("LOCITIZE_FINETUNE_OUTPUTS_DIR"):
        settings.finetune.outputs_dir = environ["LOCITIZE_FINETUNE_OUTPUTS_DIR"]
    if "LOCITIZE_PORTS_FINETUNE" in environ:
        settings.finetune.port = _int(
            environ["LOCITIZE_PORTS_FINETUNE"], settings.finetune.port
        )

    # M14 model-acquisition overrides. Exactly TWO keys (DEC-M14-10), and
    # deliberately neither allowed_hosts nor api_base: a control that decides
    # WHO LOCITIZE will talk to may be widened only from a file the user owns and
    # can audit, never from the environment. An environment variable can be set
    # by a parent shell, a shortcut, a scheduled task or another process, leaves
    # no artefact to inspect afterwards, and is invisible in a support
    # conversation. models_hub.api_base remains a settings key (and is still
    # refused unless allowed_hosts names its host), and HubConfig's api_base
    # constructor argument remains the test seam.
    #
    # The two that stay select a LOCATION rather than a counterparty:
    # download_dir relocates a confinement root whose destination is still
    # confined by resolve_destination, and enabled turns the feature off.
    if "LOCITIZE_MODELS_HUB_ENABLED" in environ:
        settings.models_hub.enabled = _env_flag(
            environ["LOCITIZE_MODELS_HUB_ENABLED"], settings.models_hub.enabled
        )
    if environ.get("LOCITIZE_MODELS_HUB_DOWNLOAD_DIR"):
        settings.models_hub.download_dir = environ["LOCITIZE_MODELS_HUB_DOWNLOAD_DIR"]

    # Per-model location override: LOCITIZE_MODEL_<ID> where <ID> is the model id
    # uppercased with non-alphanumeric characters turned into underscores.
    for model in models.models:
        env_key = "LOCITIZE_MODEL_" + _env_id(model.id)
        if environ.get(env_key):
            model.location = environ[env_key]


def _env_id(model_id: str) -> str:
    """Normalise a model id into the tail of its location env var name."""
    return "".join(c if c.isalnum() else "_" for c in model_id).upper()


def _is_safe_health_path(path: str) -> bool:
    """True if `path` is a rooted URL path that cannot reshape the URL authority.

    Security SEC-1: the readiness URL is built as
    "http://127.0.0.1:{port}{path}". A safe path starts with a single '/', so it
    is unambiguously a path component. It is rejected if it:
      - does not start with '/'                (relative -> could merge with host)
      - starts with '//'                       (an RFC 3986 authority)
      - contains '://' or a ':' before the first '/'  (embeds a scheme/host)
      - contains a '@', which can carry userinfo@host into the authority
      - contains whitespace or control characters.
    The loopback host in the built URL is a hardcoded constant regardless, so this
    is defense in depth, not the sole control.
    """
    if not path.startswith("/"):
        return False
    if path.startswith("//"):
        return False
    if "://" in path or "@" in path:
        return False
    if any(ch.isspace() or ord(ch) < 0x20 for ch in path):
        return False
    return True


def _is_safe_backend_url(url: str) -> bool:
    """True if `url` is a credential-free http(s) loopback URL (M9-lite).

    The Open WebUI backend base URL is handed to the child as OPENAI_API_BASE_URL(S).
    Data Model section 10 restricts it to a loopback endpoint carrying no token: it
    must use http/https, target 127.0.0.1 or localhost, and contain no '@' userinfo
    (which could smuggle credentials or a different host into the authority). A
    remote host or an embedded credential is rejected and the derived loopback URL
    is used instead -- the same "never leave loopback, never carry a secret in yaml"
    discipline as the health-path guard above.
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    if "@" in (parsed.netloc or ""):
        return False
    host = parsed.hostname or ""
    return host in ("127.0.0.1", "localhost", "::1")


# --------------------------------------------------------------------------- #
# Validation (invariants from the Data Model)
# --------------------------------------------------------------------------- #


def _validate(
    settings: Settings, models: ModelRegistryData, issues: list[ConfigIssue]
) -> None:
    """Check the Data Model invariants, appending WARNING/ERROR issues."""
    # Schema version. An unknown version is a WARNING, never a failure: a file
    # written by a newer LOCITIZE must still start an older one rather than lock the
    # user out of their own configuration.
    if settings.version not in SUPPORTED_SETTINGS_VERSIONS:
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"settings version {settings.version} is not one of "
                f"{list(SUPPORTED_SETTINGS_VERSIONS)}; parsing it anyway",
            )
        )
    ports = settings.ports
    if ports.allocation not in VALID_PORT_ALLOCATION:
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"ports.allocation '{ports.allocation}' invalid; using 'auto'",
            )
        )
        ports.allocation = "auto"
    if not (1024 <= ports.range_start < ports.range_end <= 65535):
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                "ports range must satisfy 1024 <= start < end <= 65535",
            )
        )
    for attr in (
        "llama_cpp",
        "whisper",
        "kokoro",
        "vision",
        "scheduler",
        "openwebui",
    ):
        port = getattr(ports, attr)
        if not (ports.range_start <= port <= ports.range_end):
            issues.append(
                ConfigIssue(
                    "WARNING",
                    SETTINGS_FILE,
                    f"port {attr}={port} is outside the reserved range",
                )
            )

    # M9-lite (Data Model section 10): the chat-UI preference and Open WebUI knobs.
    # An unknown preferred_ui is corrected to the safe default 'ask' (so a typo can
    # never silently pick a UI), and a non-positive ready_timeout_s is corrected to
    # the 300s default (so the slow first boot is never starved of its readiness
    # window). backend_base_url, if set, must be a loopback http(s) URL and must not
    # carry credentials, matching the SEC-1 loopback discipline of every other
    # service endpoint.
    if settings.chat.preferred_ui not in ("ask", "llamacpp", "openwebui"):
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"chat.preferred_ui '{settings.chat.preferred_ui}' invalid; using 'ask'",
            )
        )
        settings.chat.preferred_ui = "ask"
    owui = settings.openwebui
    if not isinstance(owui.ready_timeout_s, int) or owui.ready_timeout_s <= 0:
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"openwebui.ready_timeout_s '{owui.ready_timeout_s}' must be a "
                f"positive integer; using 300",
            )
        )
        owui.ready_timeout_s = 300
    if owui.backend_base_url and not _is_safe_backend_url(owui.backend_base_url):
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"openwebui.backend_base_url '{owui.backend_base_url}' is not a "
                f"credential-free loopback URL; deriving from ports.llama_cpp instead",
            )
        )
        owui.backend_base_url = ""
    silence = owui.call_silence_ms
    if (
        not isinstance(silence, int)
        or isinstance(silence, bool)
        or not CALL_SILENCE_MIN_MS <= silence <= CALL_SILENCE_MAX_MS
    ):
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"openwebui.call_silence_ms '{silence}' must be an integer "
                f"{CALL_SILENCE_MIN_MS}..{CALL_SILENCE_MAX_MS}; using "
                f"{CALL_SILENCE_UPSTREAM_MS}",
            )
        )
        owui.call_silence_ms = CALL_SILENCE_UPSTREAM_MS

    noise_mode = settings.speech.noise_suppression
    if noise_mode not in VALID_NOISE_SUPPRESSION:
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"speech.noise_suppression '{noise_mode}' invalid; using 'balanced'",
            )
        )
        settings.speech.noise_suppression = "balanced"

    thr = settings.thresholds
    if thr.ram_hard_min_mb >= thr.ram_floor_mb:
        issues.append(
            ConfigIssue("WARNING", SETTINGS_FILE, "ram hard_min should be < floor")
        )
    if thr.disk_hard_min_mb >= thr.disk_floor_mb:
        issues.append(
            ConfigIssue("WARNING", SETTINGS_FILE, "disk hard_min should be < floor")
        )

    for model in models.models:
        if model.status not in VALID_MODEL_STATUS:
            issues.append(
                ConfigIssue(
                    "WARNING",
                    MODELS_FILE,
                    f"model '{model.id}' status '{model.status}' invalid",
                )
            )
        if model.context_size <= 0:
            issues.append(
                ConfigIssue(
                    "WARNING", MODELS_FILE, f"model '{model.id}' context_size must be > 0"
                )
            )
        if model.gpu_layers < -1:
            issues.append(
                ConfigIssue(
                    "WARNING", MODELS_FILE, f"model '{model.id}' gpu_layers must be >= -1"
                )
            )
        if model.benchmark_score is not None and not (
            0.0 <= model.benchmark_score <= 100.0
        ):
            issues.append(
                ConfigIssue(
                    "WARNING",
                    MODELS_FILE,
                    f"model '{model.id}' benchmark_score must be null or 0-100",
                )
            )

    # SEC-1: the llama.cpp readiness path is appended to a loopback URL. Reject
    # any value that could reshape the URL's authority (a leading '//' is an
    # authority per RFC 3986; a scheme like 'http:' or a bare '@' can steer the
    # host). An invalid value is replaced with the safe default rather than
    # allowed through. Empty string is legal (means: fall back to TCP/poll).
    health_path = settings.services.llama_cpp_health_path
    if health_path and not _is_safe_health_path(health_path):
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"services.llama_cpp_health_path '{health_path}' is not a rooted "
                f"loopback path; using '/health'",
            )
        )
        settings.services.llama_cpp_health_path = "/health"

    # M4/GUI invariants (Data Model 6.2). Bad values degrade to the safe default
    # with a WARNING rather than breaking GUI startup.
    if settings.gui.monitor_interval_s <= 0:
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"gui.monitor_interval_s {settings.gui.monitor_interval_s} must be "
                f"> 0; using 1.5",
            )
        )
        settings.gui.monitor_interval_s = 1.5
    proxy_port = settings.proxy.port
    # bind_port_80 legitimately selects port 80 (an elevated opt-in), so only the
    # non-privileged proxy.port field is held to the loopback reserved range.
    if not (1024 <= proxy_port <= 65535):
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"proxy.port {proxy_port} outside 1024-65535; using 8085",
            )
        )
        settings.proxy.port = 8085

    default_model = settings.launcher.default_model
    if default_model and default_model not in {m.id for m in models.models}:
        issues.append(
            ConfigIssue(
                "WARNING",
                SETTINGS_FILE,
                f"launcher.default_model '{default_model}' matches no model id",
            )
        )


# --------------------------------------------------------------------------- #
# Dump / redaction (used by the Settings menu action)
# --------------------------------------------------------------------------- #


def settings_to_dict(settings: Settings, redact: bool = True) -> dict[str, Any]:
    """Render settings as a plain dict for display, redacting env-sourced paths.

    Any non-empty path value is treated as potentially machine-specific/secret
    and masked when redact=True (Permission Matrix section 2), so a Settings dump
    never leaks a filesystem layout or a future token.
    """
    def dump(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {f.name: dump(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, Path):
            return str(obj)
        return obj

    data = {
        "version": settings.version,
        "paths": dump(settings.paths),
        "ports": dump(settings.ports),
        "thresholds": dump(settings.thresholds),
        "services": dump(settings.services),
        "logging": dump(settings.logging),
        "speech": dump(settings.speech),
        "tts": dump(settings.tts),
        "assistant": dump(settings.assistant),
        "memory": dump(settings.memory),
        "vision": dump(settings.vision),
        "gui": dump(settings.gui),
        "proxy": dump(settings.proxy),
        "router": dump(settings.router),
        "secure_proxy": dump(settings.secure_proxy),
        "launcher": dump(settings.launcher),
        "chat": dump(settings.chat),
        "chat_harness": dump(settings.chat_harness),
        "openwebui": dump(settings.openwebui),
        "finetune": dump(settings.finetune),
        "models_hub": dump(settings.models_hub),
    }
    if redact:
        data["paths"] = {
            key: ("<set>" if value else "") for key, value in data["paths"].items()
        }
        # The finetune block carries three machine-specific filesystem locations
        # alongside plain flags. Mask only those three, on the same "<set>" vs ""
        # rule as the paths block, so the owner can still confirm from the dump
        # that an override actually took without the dump leaking their layout.
        for key in ("studio_dir", "outputs_dir", "python"):
            value = data["finetune"].get(key, "")
            data["finetune"][key] = "<set>" if value else ""
    return data


# --------------------------------------------------------------------------- #
# Targeted models.yaml write-back (M4/GUI, Architecture G6)
# --------------------------------------------------------------------------- #


# Matches a "key: value  # optional comment" line, capturing the prefix (indent +
# key + separator), the current scalar value, and any trailing remainder (spaces
# and/or a comment). Rewriting only group 2 preserves indentation, key casing, and
# the owner's inline comment exactly (RG3 mitigation).
_SCALAR_LINE = None  # lazily compiled in _replace_scalar to keep import cheap


# --------------------------------------------------------------------------- #
# The one registry write chokepoint (DEC-M14-11)
# --------------------------------------------------------------------------- #


class RegistryWriteError(Exception):
    """The single typed failure every models.yaml write can raise.

    QA (NEW-QA-M14-9) watched the Save-edits and Rename surfaces print a bare
    "[Errno 2] No such file or directory:" followed by the Python repr of the
    models.yaml path, in which every path separator was doubled: an errno the
    user cannot act on, a path they have to mentally un-escape, and no next
    step at all. Every UI and CLI caller
    renders a failure with str(exc), so the fix is that str(exc) is ALWAYS a
    finished sentence: what happened, the real path (rendered with str(), never
    repr()), and what to do next.

    One type, not one per cause, because the callers do not branch on the cause -
    they show it. Raised only by _edit_registry below, which is the only code
    that opens models.yaml for writing.
    """


def _edit_registry(
    base_dir: Path | str,
    transform: "Callable[[list[str], Path], str]",
) -> None:
    """Open, edit and atomically rewrite models.yaml - the ONLY writer of it.

    DEC-M14-11: a chokepoint rather than a rule each of the four public writers
    has to remember. It owns exactly three things, so no caller has to:

    1. the pre-write checks that produced QA's unreadable errors - the file must
       exist, must be readable, and must actually have a top-level `models:`
       section to write into;
    2. the atomic temp-file write plus os.replace (via _atomic_write), so a
       crash mid-write cannot leave a half-written registry;
    3. the error translation - every OSError becomes a RegistryWriteError whose
       message names the real path and ends with a next step.

    `transform` is the writer-specific part: it receives the file's lines (with
    line endings kept) and the resolved path, and returns the new full text. It
    may raise ValueError for a caller-input refusal (an unknown model id, a
    duplicate id, an out-of-range value) - those are the caller's own contract
    and pass through untouched - or RegistryWriteError for a post-write
    verification miss.

    Note the deliberate use of exc.strerror rather than str(exc) when
    translating an OSError: str(OSError) embeds the errno AND the offending
    filename as a repr, which is exactly the "[WinError 5] ... models.yaml.
    1yohmr7c.tmp" noise QA read on screen. strerror is the plain-language half
    ("Access is denied"), and the path this function names is the file the user
    actually knows about, not the temp file.
    """
    path = Path(base_dir) / MODELS_FILE

    if not path.is_file():
        raise RegistryWriteError(
            f"locitize could not update your model list because the file "
            f"{path} is missing. Restart locitize to have it recreated from the "
            f"shipped template, or restore it from your backup, then try again."
        )

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryWriteError(_registry_os_message(path, exc, "read")) from exc

    lines = text.splitlines(keepends=True)
    section_start, _section_end = _locate_top_section(lines, "models")
    if section_start is None:
        raise RegistryWriteError(
            f"locitize could not update your model list because {path} has no "
            f"top-level 'models:' section to write into, and it refuses to "
            f"guess where the list should start. Open that file, add a "
            f"'models:' line, then try again."
        )

    new_text = transform(lines, path)

    try:
        _atomic_write(path, new_text)
    except OSError as exc:
        raise RegistryWriteError(_registry_os_message(path, exc, "write")) from exc


def _registry_os_message(path: Path, exc: OSError, verb: str) -> str:
    """Render one OSError as a finished sentence naming the real path.

    Kept beside the chokepoint because it is the half of the translation that
    decides what the user reads. The cause is the OS's own plain-language
    strerror ("Access is denied"), never the errno-plus-repr form.
    """
    cause = (exc.strerror or "the file could not be opened").strip().rstrip(".")
    return (
        f"locitize could not {verb} your model list at {path}: {cause}. Close any "
        f"program that has the file open, check that it is not marked "
        f"read-only, then try again."
    )


def write_model_fields(
    base_dir: Path | str,
    model_id: str,
    gpu_layers: int,
    context_size: int,
) -> None:
    """Rewrite ONLY a model's gpu_layers and context_size lines in models.yaml.

    This is the single new write path the GUI's Panel 3 editors use (Architecture
    G6, Permission Matrix section 7). It is deliberately a targeted line edit, not
    a yaml.safe_dump of the whole document, because a full re-dump would strip the
    owner's comments and reorder keys. Contract:

    - Re-validate the invariants first (gpu_layers >= -1, context_size > 0) and
      raise ValueError with a remedy on violation. The GUI validates too; this is
      the second, authoritative gate so a bad value can never reach the file.
    - Locate the target model's block by its `id:` line within the models list,
      then rewrite only the gpu_layers and context_size scalar values on their
      existing lines, preserving every other byte (indentation, key order,
      comments). If either line cannot be located, raise rather than guess (RG3).
    - Re-parse the rewritten text through yaml.safe_load and confirm the target
      model now carries the new values BEFORE committing, so a malformed edit
      leaves the original file untouched.
    - Write atomically: a sibling temp file plus os.replace, so a crash mid-write
      cannot leave a half-written models.yaml. Since DEC-M14-11 the opening,
      the atomic write and the error translation all live in _edit_registry;
      this function only supplies the edit.

    Path safety (Permission Matrix section 7): the file is always base_dir/
    models.yaml resolved from the data root, so the write cannot escape it.
    """
    # Second-gate invariant check (Data Model 1.1). Raise a remedy, not a bare error.
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool) or gpu_layers < -1:
        raise ValueError(
            f"gpu_layers must be an integer >= -1 (got {gpu_layers!r}); "
            f"-1 means all layers on GPU"
        )
    if (
        not isinstance(context_size, int)
        or isinstance(context_size, bool)
        or context_size <= 0
    ):
        raise ValueError(
            f"context_size must be a positive integer (got {context_size!r})"
        )

    def edit(lines: list[str], path: Path) -> str:
        start, end = _locate_model_block(lines, model_id)
        if start is None:
            raise ValueError(
                f"model id '{model_id}' not found in {MODELS_FILE}; cannot write fields"
            )

        gpu_idx = _find_key_line(lines, start, end, "gpu_layers")
        ctx_idx = _find_key_line(lines, start, end, "context_size")
        if gpu_idx is None or ctx_idx is None:
            missing = "gpu_layers" if gpu_idx is None else "context_size"
            raise ValueError(
                f"could not locate '{missing}:' line for model '{model_id}' in "
                f"{MODELS_FILE}; refusing to guess"
            )

        lines[gpu_idx] = _replace_scalar(lines[gpu_idx], gpu_layers)
        lines[ctx_idx] = _replace_scalar(lines[ctx_idx], context_size)
        new_text = "".join(lines)

        # Round-trip guard (RG3): confirm the edited text still parses AND the
        # target model now carries exactly the requested values before the
        # chokepoint touches disk.
        parsed = yaml.safe_load(new_text)
        _confirm_written(parsed, model_id, gpu_layers, context_size, path)
        return new_text

    _edit_registry(base_dir, edit)


def write_model_identity(
    base_dir: Path | str,
    model_id: str,
    new_id: str,
    new_name: str,
) -> None:
    """Rewrite ONLY a model's id and name lines in models.yaml (rename / re-id).

    A sibling of write_model_fields, reusing the same targeted, atomic,
    comment-and-key-order-preserving line-rewrite mechanism, extended to values
    that may contain spaces (a model name), which write_model_fields' single-token
    _replace_scalar cannot handle. Contract:

    - new_id and new_name must both be non-empty after stripping; raise ValueError
      with a remedy otherwise, so a blank identity can never reach the file.
    - When new_id differs from model_id, it must not already belong to another
      model in the registry (ids are the lookup key everywhere: running-process
      tracking, benchmark results, settings selection); raise ValueError rather
      than silently creating a duplicate.
    - Locate the target model's block by its current `id:` line, then rewrite only
      the `id:` and `name:` scalar values on their existing lines, preserving every
      other byte (indentation, key order, comments, and the other model's blocks).
      If the `name:` line cannot be located, raise rather than guess (RG3).
    - Re-parse the rewritten text through yaml.safe_load and confirm a model now
      carries new_id/new_name BEFORE committing, so a malformed edit leaves the
      original file untouched.
    - Write atomically: a sibling temp file plus os.replace.

    Path safety (Permission Matrix section 7): the file is always base_dir/
    models.yaml resolved from the platform directory, so the write cannot escape
    Codebase/platform/.
    """
    new_id = str(new_id).strip()
    new_name = str(new_name).strip()
    if not new_id:
        raise ValueError("model id must not be empty")
    if not new_name:
        raise ValueError("model name must not be empty")

    def edit(lines: list[str], path: Path) -> str:
        start, end = _locate_model_block(lines, model_id)
        if start is None:
            raise ValueError(
                f"model id '{model_id}' not found in {MODELS_FILE}; cannot rename"
            )

        if new_id != model_id:
            other_start, _ = _locate_model_block(lines, new_id)
            if other_start is not None:
                raise ValueError(
                    f"model id '{new_id}' already exists in {MODELS_FILE}; "
                    f"ids must be unique"
                )

        name_idx = _find_key_line(lines, start, end, "name")
        if name_idx is None:
            raise ValueError(
                f"could not locate 'name:' line for model '{model_id}' in "
                f"{MODELS_FILE}; refusing to guess"
            )

        lines[start] = _replace_value_line(lines[start], new_id, quote=False)
        lines[name_idx] = _replace_value_line(lines[name_idx], new_name, quote=True)
        new_text = "".join(lines)

        # Round-trip guard (RG3): confirm the edited text still parses AND a model
        # now carries exactly the requested id/name before the chokepoint writes.
        parsed = yaml.safe_load(new_text)
        _confirm_identity_written(parsed, new_id, new_name, path)
        return new_text

    _edit_registry(base_dir, edit)


def write_model_capabilities(
    base_dir: Path | str,
    model_id: str,
    capabilities: list[str],
) -> None:
    """Rewrite (or insert) ONLY a model's capabilities line in models.yaml.

    Owner request 2026-08-21: a sibling of write_model_fields/write_model_identity,
    reusing the identical targeted, atomic, comment-and-key-order-preserving
    line-rewrite mechanism. Differs from those two in one way: `capabilities` is
    an ADDITIVE field (config.py's default is []), so most pre-existing rows have
    no `capabilities:` line at all to rewrite - this is the first registry writer
    that has to INSERT a line rather than only ever replacing one.

    Contract:
    - Locate the target model's block by its `id:` line.
    - If a `capabilities:` line already exists in the block, rewrite its value
      in place (via _replace_value_line, same as a multi-word name) - every
      other byte (indentation, comments, key order) is preserved.
    - Otherwise, insert a new `capabilities: [...]` line immediately after the
      block's last real field line, at that line's own indentation, so it lands
      inside the model's block rather than after the trailing blank line that
      separates registry entries.
    - Re-parse through yaml.safe_load and confirm the target model now carries
      exactly the requested list BEFORE committing, so a malformed edit leaves
      the original file untouched.
    - Write atomically (sibling temp file + os.replace).

    Path safety (Permission Matrix section 7): the file is always base_dir/
    models.yaml resolved from the data root, so the write cannot escape it.
    """
    cleaned = [str(c).strip() for c in (capabilities or []) if str(c).strip()]
    rendered = "[" + ", ".join(f'"{c}"' for c in cleaned) + "]"

    def edit(lines: list[str], path: Path) -> str:
        start, end = _locate_model_block(lines, model_id)
        if start is None:
            raise ValueError(
                f"model id '{model_id}' not found in {MODELS_FILE}; "
                f"cannot write capabilities"
            )

        cap_idx = _find_key_line(lines, start, end, "capabilities")
        if cap_idx is not None:
            lines[cap_idx] = _replace_value_line(lines[cap_idx], rendered, quote=False)
        else:
            # No existing line - insert one right after status:, a required
            # top-level field on every row (never absent, never nested), so its
            # indentation is always the block's real top-level field indent.
            # The block's LAST line is not a safe anchor: it can be a nested
            # sub-key (e.g. benchmark_sweep's own gpu_layers list, indented one
            # level deeper), which would attach capabilities to the wrong
            # mapping - exactly what the round-trip guard below exists to catch.
            status_idx = _find_key_line(lines, start, end, "status")
            if status_idx is None:
                raise ValueError(
                    f"could not locate 'status:' line for model '{model_id}' in "
                    f"{MODELS_FILE}; refusing to guess where to insert capabilities"
                )
            indent = len(lines[status_idx]) - len(lines[status_idx].lstrip())
            newline = "\r\n" if lines[status_idx].endswith("\r\n") else "\n"
            lines.insert(status_idx + 1, f"{' ' * indent}capabilities: {rendered}{newline}")

        new_text = "".join(lines)

        # Round-trip guard (RG3): confirm the edited text still parses AND the
        # target model now carries exactly the requested list before the
        # chokepoint writes.
        parsed = yaml.safe_load(new_text)
        _confirm_capabilities_written(parsed, model_id, cleaned, path)
        return new_text

    _edit_registry(base_dir, edit)


def write_model_mmproj(base_dir: Path | str, model_id: str, mmproj: str) -> None:
    """Rewrite (or insert) ONLY a model's mmproj line in models.yaml (M18.18).

    Sibling of write_model_capabilities with the identical targeted, atomic,
    comment-preserving mechanism. Written for projector auto-pairing: a vision
    model registered without its mmproj can have the projector attached later
    without touching any other byte of the owner's file. The path must exist -
    a projector line pointing at nothing would advertise sight the model does
    not have.
    """
    cleaned = str(mmproj or "").strip()
    if cleaned and not Path(cleaned).is_file():
        raise ValueError(
            f"mmproj must be an existing projector file (got {cleaned!r})"
        )
    # An empty value is the deliberate UNPAIR: it clears a projector a past
    # (looser) rule attached wrongly. Only a non-empty path must exist.
    rendered = json.dumps(cleaned.replace("\\", "/"))

    def edit(lines: list[str], path: Path) -> str:
        start, end = _locate_model_block(lines, model_id)
        if start is None:
            raise ValueError(
                f"model id '{model_id}' not found in {MODELS_FILE}; "
                f"cannot write mmproj"
            )
        idx = _find_key_line(lines, start, end, "mmproj")
        if idx is not None:
            lines[idx] = _replace_value_line(lines[idx], rendered, quote=False)
        else:
            status_idx = _find_key_line(lines, start, end, "status")
            if status_idx is None:
                raise ValueError(
                    f"could not locate 'status:' line for model '{model_id}' in "
                    f"{MODELS_FILE}; refusing to guess where to insert mmproj"
                )
            indent = len(lines[status_idx]) - len(lines[status_idx].lstrip())
            newline = "\r\n" if lines[status_idx].endswith("\r\n") else "\n"
            lines.insert(status_idx + 1, f"{' ' * indent}mmproj: {rendered}{newline}")
        new_text = "".join(lines)
        parsed = yaml.safe_load(new_text)
        rows = (parsed or {}).get("models") or []
        row = next((r for r in rows if isinstance(r, dict) and r.get("id") == model_id), None)
        if row is None or str(row.get("mmproj", "")) != cleaned.replace("\\", "/"):
            raise ValueError(
                f"post-write verification failed for mmproj on '{model_id}'; "
                f"file left unchanged"
            )
        return new_text

    _edit_registry(base_dir, edit)


def write_model_reasoning(
    base_dir: Path | str, model_id: str, reasoning: dict[str, Any] | None
) -> None:
    """Rewrite (or insert) ONLY a model's reasoning line in models.yaml.

    Sibling of write_model_mmproj / write_model_capabilities, with the identical
    targeted, atomic, comment-and-key-order-preserving mechanism. Owner request
    2026-09-02, after finding that every model on this machine was running at its
    template's default effort (xhigh on Qwen3.8-27B-UD-IQ4_XS) with
    --reasoning-budget at its own default of -1, unrestricted - a combination
    nobody chose, because nothing in LOCITIZE ever surfaced it.

    The value is validated through parse_reasoning first, so this writer cannot
    put a shape on disk that the loader would then warn about and drop. An empty
    mapping (or None) is the deliberate CLEAR: it writes `reasoning: {}`, which
    the loader reads as absent, returning the model to llama-server's defaults.

    Rendered as an inline YAML flow mapping on one line, which keeps this inside
    the single-line rewrite mechanism the other registry writers use rather than
    needing a block-structure editor.
    """
    cleaned, problems = parse_reasoning(reasoning or None, model_id)
    if problems:
        raise ValueError("; ".join(problems))
    rendered = json.dumps(cleaned or {}, sort_keys=True)

    def edit(lines: list[str], path: Path) -> str:
        start, end = _locate_model_block(lines, model_id)
        if start is None:
            raise ValueError(
                f"model id '{model_id}' not found in {MODELS_FILE}; "
                f"cannot write reasoning"
            )
        idx = _find_key_line(lines, start, end, "reasoning")
        if idx is not None:
            lines[idx] = _replace_value_line(lines[idx], rendered, quote=False)
        else:
            status_idx = _find_key_line(lines, start, end, "status")
            if status_idx is None:
                raise ValueError(
                    f"could not locate 'status:' line for model '{model_id}' in "
                    f"{MODELS_FILE}; refusing to guess where to insert reasoning"
                )
            indent = len(lines[status_idx]) - len(lines[status_idx].lstrip())
            newline = "\r\n" if lines[status_idx].endswith("\r\n") else "\n"
            lines.insert(
                status_idx + 1, f"{' ' * indent}reasoning: {rendered}{newline}"
            )
        new_text = "".join(lines)
        parsed = yaml.safe_load(new_text)
        rows = (parsed or {}).get("models") or []
        row = next(
            (r for r in rows if isinstance(r, dict) and r.get("id") == model_id), None
        )
        got, _ = parse_reasoning(row.get("reasoning") if row else None, model_id)
        if row is None or got != cleaned:
            raise ValueError(
                f"post-write verification failed for reasoning on '{model_id}'; "
                f"file left unchanged"
            )
        return new_text

    _edit_registry(base_dir, edit)


def write_model_score(
    base_dir: Path | str,
    model_id: str,
    score: float,
) -> None:
    """Rewrite ONLY a model's benchmark_score line in models.yaml (M5.7).

    A sibling of write_model_fields that reuses the identical targeted, atomic,
    comment-and-key-order-preserving line-rewrite mechanism (_locate_model_block /
    _find_key_line / _replace_scalar / round-trip safe_load / temp-file +
    os.replace). Kept a separate function rather than extending write_model_fields
    so that M4-tested signature stays frozen (Data Model section 7.2).

    Contract:
    - `score` must be a real number in [0, 100] (a benchmark_score invariant);
      raise ValueError otherwise so a bad value never reaches the file.
    - Locate the target model's block, then rewrite only its benchmark_score
      scalar value on its existing line, replacing the literal `null` default with
      the float and preserving every other byte (indentation, comments, key order).
    - Re-parse through yaml.safe_load and confirm the target model now carries the
      new score BEFORE committing, so a malformed edit leaves the file untouched.
    - Write atomically (sibling temp file + os.replace).

    Path safety (Permission Matrix section 8): the file is always base_dir/
    models.yaml resolved from the platform directory, so the write cannot escape
    Codebase/platform/.
    """
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"benchmark_score must be a number (got {score!r})")
    score = float(score)
    if not (0.0 <= score <= 100.0):
        raise ValueError(f"benchmark_score must be within 0-100 (got {score})")

    def edit(lines: list[str], path: Path) -> str:
        start, end = _locate_model_block(lines, model_id)
        if start is None:
            raise ValueError(
                f"model id '{model_id}' not found in {MODELS_FILE}; cannot write score"
            )

        score_idx = _find_key_line(lines, start, end, "benchmark_score")
        if score_idx is None:
            raise ValueError(
                f"could not locate 'benchmark_score:' line for model '{model_id}' in "
                f"{MODELS_FILE}; refusing to guess"
            )

        # Render as a compact float so the file never gains a noisy 73.90000001;
        # the value is already rounded upstream but this guards against a raw float.
        lines[score_idx] = _replace_scalar(lines[score_idx], round(score, 1))
        new_text = "".join(lines)

        parsed = yaml.safe_load(new_text)
        _confirm_score_written(parsed, model_id, round(score, 1), path)
        return new_text

    _edit_registry(base_dir, edit)


def write_model_tuning(
    base_dir: Path | str,
    model_id: str,
    context_size: int,
    server_args: list[Any],
    note: str = "",
) -> None:
    """Rewrite ONLY a model's context_size and server_args lines in models.yaml.

    Owner request 2026-08-22, written for the context auto-tuner (autotune.py).
    A fifth sibling of write_model_fields/write_model_identity/
    write_model_capabilities/write_model_score, reusing the identical targeted,
    atomic, comment-and-key-order-preserving mechanism through _edit_registry.

    Why a new function rather than extending write_model_fields: that one also
    writes gpu_layers, and the auto-tuner must not touch gpu_layers. gpu_layers
    is the tokens-per-second lever the owner swept by hand per model; a context
    tuner silently rewriting it would undo that work. Same reasoning for every
    other field on the row (notes, benchmark_score, ready_timeout_s): this writer
    can only ever change the two lines named in its own name.

    Contract:
    - context_size must be a positive integer; server_args must be a list whose
      items all render as single YAML-safe tokens. Raise ValueError with a
      remedy otherwise, so a bad value never reaches the file.
    - Locate the target model's block by its `id:` line, rewrite the
      context_size scalar and the server_args value in place. Like
      write_model_capabilities, server_args may legitimately be ABSENT from a
      row, in which case a new line is inserted after `status:` (the one field
      every row carries at the block's real top-level indent).
    - Re-parse through yaml.safe_load and confirm the target model now carries
      exactly the requested values BEFORE committing, so a malformed edit leaves
      the original file untouched.
    - Write atomically (sibling temp file + os.replace), via _edit_registry.

    The one place this writer does NOT preserve every other byte is the inline
    comment on the context_size line, which `note` replaces when supplied. Those
    comments are dated explanations of why the value is what it is ("raised
    10000->65536 2026-08-21 because ..."); leaving one standing next to a value
    it no longer describes would be worse than losing it, so the auto-tuner
    passes a fresh one stating what it actually measured.

    Path safety (Permission Matrix section 7): the file is always base_dir/
    models.yaml resolved from the data root, so the write cannot escape it.
    """
    if (
        not isinstance(context_size, int)
        or isinstance(context_size, bool)
        or context_size <= 0
    ):
        raise ValueError(
            f"context_size must be a positive integer (got {context_size!r})"
        )
    if not isinstance(server_args, (list, tuple)):
        raise ValueError(
            f"server_args must be a list of command-line arguments (got "
            f"{type(server_args).__name__})"
        )
    cleaned = [str(a) for a in server_args]
    # Guard the flow-sequence rendering below: a value containing a quote, a
    # comma or a bracket could not be written as `["a", "b"]` without escaping,
    # and no real llama-server flag needs one. Refuse rather than emit YAML that
    # would re-parse into something different from what was asked for.
    for arg in cleaned:
        if not arg or any(ch in arg for ch in '"\n#,[]'):
            raise ValueError(
                f"server_args entry {arg!r} contains a character locitize will "
                f"not write into models.yaml; edit this model's server_args by "
                f"hand instead"
            )
    rendered = "[" + ", ".join(f'"{a}"' for a in cleaned) + "]"

    def edit(lines: list[str], path: Path) -> str:
        start, end = _locate_model_block(lines, model_id)
        if start is None:
            raise ValueError(
                f"model id '{model_id}' not found in {MODELS_FILE}; "
                f"cannot write tuning"
            )

        ctx_idx = _find_key_line(lines, start, end, "context_size")
        if ctx_idx is None:
            raise ValueError(
                f"could not locate 'context_size:' line for model '{model_id}' "
                f"in {MODELS_FILE}; refusing to guess"
            )
        lines[ctx_idx] = _replace_scalar(lines[ctx_idx], context_size)
        if note:
            lines[ctx_idx] = _set_inline_comment(lines[ctx_idx], note)

        args_idx = _find_key_line(lines, start, end, "server_args")
        if args_idx is not None:
            lines[args_idx] = _replace_value_line(lines[args_idx], rendered, quote=False)
        else:
            # Same anchor and reasoning as write_model_capabilities: status: is
            # the one required, never-nested field on every row, so its
            # indentation is reliably the block's own top-level field indent.
            status_idx = _find_key_line(lines, start, end, "status")
            if status_idx is None:
                raise ValueError(
                    f"could not locate 'status:' line for model '{model_id}' in "
                    f"{MODELS_FILE}; refusing to guess where to insert server_args"
                )
            indent = len(lines[status_idx]) - len(lines[status_idx].lstrip())
            newline = "\r\n" if lines[status_idx].endswith("\r\n") else "\n"
            lines.insert(
                status_idx + 1, f"{' ' * indent}server_args: {rendered}{newline}"
            )

        new_text = "".join(lines)

        # Round-trip guard (RG3): confirm the edited text still parses AND the
        # target model now carries exactly the requested values before the
        # chokepoint writes.
        parsed = yaml.safe_load(new_text)
        _confirm_tuning_written(parsed, model_id, context_size, cleaned, path)
        return new_text

    _edit_registry(base_dir, edit)


def _set_inline_comment(line: str, note: str) -> str:
    """Replace (or add) the trailing `# ...` comment on one already-edited line.

    Only ever called on a line whose value has just been rewritten, so the
    comment being discarded is by definition a comment about the OLD value. The
    note is flattened to a single line and stripped of any '#' of its own, since
    a comment cannot contain a newline and a second '#' would just read as noise.
    """
    import re

    newline = ""
    body = line
    if body.endswith("\r\n"):
        body, newline = body[:-2], "\r\n"
    elif body.endswith("\n"):
        body, newline = body[:-1], "\n"

    body = re.sub(r"\s+#.*$", "", body)
    flat = " ".join(str(note).replace("#", "").split())
    return f"{body}   # {flat}{newline}"


def _confirm_tuning_written(
    parsed: Any,
    model_id: str,
    context_size: int,
    server_args: list[str],
    path: Path,
) -> None:
    """Assert the re-parsed document carries the new context_size/server_args."""
    rows = parsed.get("models", []) if isinstance(parsed, dict) else parsed
    for row in rows or []:
        if isinstance(row, dict) and str(row.get("id", "")).strip() == model_id:
            if int(row.get("context_size", 0)) != context_size:
                raise _post_write_failure(f"for '{model_id}' context_size", path)
            written = [str(a) for a in (row.get("server_args") or [])]
            if written != server_args:
                raise _post_write_failure(f"for '{model_id}' server_args", path)
            return
    raise _post_write_failure(f"- could not find '{model_id}'", path)


def append_model_entry(
    base_dir: Path | str,
    model_id: str,
    name: str,
    location: str,
    *,
    description: str = "",
    context_size: int = 8192,
    gpu_layers: int = 999,
    recommended_prompt: str = "",
    quantization: str = "",
    notes: str = "",
    sha256: str = "",
    mmproj: str = "",
) -> None:
    """Append ONE new model block to models.yaml (M13's Register action).

    This is the only writer discovery ever reaches, and it is reachable only from
    an explicit owner click - a background scan must never mutate an
    owner-maintained, comment-rich file, and never race the studio's own deploy
    form (Architecture M13.6). It follows the same discipline as its siblings
    write_model_fields / write_model_score:

    - Refuse an empty id/name/location, a non-existent location, an id already
      present in the file, or a non-positive context_size / gpu_layers < -1, each
      with a ValueError carrying a remedy the UI shows verbatim.
    - Insert the new block at the END of the existing top-level `models:` list
      rather than re-dumping the document, so every comment, key order, and byte
      of the owner's file is preserved.
    - Values are emitted with json.dumps, which produces valid YAML double-quoted
      scalars, so a Windows path or an apostrophe in a name cannot break the file.
    - Re-parse through yaml.safe_load and confirm the new row is present and
      carries the requested location BEFORE committing, then write atomically.

    Path safety (Permission Matrix section 7): the file is always base_dir/
    models.yaml, so the write cannot escape Codebase/platform/.
    """
    model_id = str(model_id).strip()
    name = str(name).strip()
    location = str(location).strip()
    if not model_id:
        raise ValueError("model id must not be empty")
    if not name:
        raise ValueError("model name must not be empty")
    if not location:
        raise ValueError("model location must not be empty")
    if not Path(location).expanduser().is_file():
        raise ValueError(
            f"model file '{location}' does not exist; refusing to register a row "
            f"that points at nothing"
        )
    if not isinstance(context_size, int) or isinstance(context_size, bool) or context_size <= 0:
        raise ValueError(f"context_size must be a positive integer (got {context_size!r})")
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool) or gpu_layers < -1:
        raise ValueError(f"gpu_layers must be an integer >= -1 (got {gpu_layers!r})")

    def edit(lines: list[str], path: Path) -> str:
        existing_start, _existing_end = _locate_model_block(lines, model_id)
        if existing_start is not None:
            raise ValueError(
                f"an entry named '{model_id}' already exists in {MODELS_FILE}. "
                f"Rename or remove the existing entry first."
            )

        # The chokepoint already refused a file with no top-level 'models:'
        # section, so this lookup cannot come back empty here.
        section_start, section_end = _locate_top_section(lines, "models")

        # An empty registry may be written as the inline "models: []". Appending a
        # block item under that flow-style value would produce invalid YAML, so the
        # header is normalized to a plain "models:" first (the only case where this
        # writer changes an existing line at all).
        header = lines[section_start].rstrip("\n")
        if header.split(":", 1)[1].strip() in ("[]", "~", "null"):
            lines[section_start] = "models:\n"

        block = _render_model_block(
            model_id,
            name,
            location,
            description=description,
            context_size=context_size,
            gpu_layers=gpu_layers,
            recommended_prompt=recommended_prompt,
            quantization=quantization,
            notes=notes,
            sha256=sha256,
            mmproj=mmproj,
        )
        insert_at = section_end if section_end is not None else len(lines)
        # Make sure the preceding line ends cleanly so the appended block cannot be
        # glued onto a final line that lacks a newline.
        if insert_at > 0 and lines[insert_at - 1] and not lines[insert_at - 1].endswith("\n"):
            lines[insert_at - 1] = lines[insert_at - 1] + "\n"
        lines[insert_at:insert_at] = block
        new_text = "".join(lines)

        parsed = yaml.safe_load(new_text)
        _confirm_appended(parsed, model_id, location, path)
        return new_text

    _edit_registry(base_dir, edit)


def _render_model_block(
    model_id: str,
    name: str,
    location: str,
    *,
    description: str,
    context_size: int,
    gpu_layers: int,
    recommended_prompt: str,
    quantization: str,
    notes: str,
    sha256: str = "",
    mmproj: str = "",
) -> list[str]:
    """Render one models.yaml list item as lines, in the file's existing key order."""
    quoted = json.dumps  # valid YAML double-quoted scalars, with escaping
    # M14.14.5: the sha256 line is emitted ONLY when there is a digest to record,
    # so a hand-registered row's block stays byte-for-byte what M13 produced and
    # no row ever carries an empty key implying a hash was expected.
    digest_lines = [f"    sha256: {quoted(sha256)}\n"] if sha256 else []
    # M15.6: same only-when-present rule for the vision projector path - a
    # text-only model's block must stay byte-for-byte what M13 produced.
    if mmproj:
        digest_lines.append(f"    mmproj: {quoted(mmproj)}\n")
    return [
        "\n",
        # SEC-M13-5: the id goes through the SAME quoting as every other string
        # scalar here. Today's only caller hands us a sanitized id, but this
        # writer is public API - a future caller must not be able to restructure
        # the emitted YAML with a colon, a '#', or a leading '-' in the id.
        f"  - id: {quoted(model_id)}\n",
        f"    name: {quoted(name)}\n",
        f"    description: {quoted(description)}\n",
        f"    location: {quoted(location)}\n",
        f"    context_size: {context_size}\n",
        f"    gpu_layers: {gpu_layers}\n",
        f"    recommended_prompt: {quoted(recommended_prompt)}\n",
        "    benchmark_score: null\n",
        f"    notes: {quoted(notes)}\n",
        "    status: installed\n",
        f"    quantization: {quoted(quantization)}\n",
        "    vram_estimate_mb: 0\n",
        '    server_args: ["--parallel", "1"]\n',
    ] + digest_lines


def _confirm_appended(
    parsed: Any, model_id: str, location: str, path: Path
) -> None:
    """Assert the re-parsed document now carries the new row with its location."""
    rows = parsed.get("models", []) if isinstance(parsed, dict) else parsed
    for row in rows or []:
        if isinstance(row, dict) and str(row.get("id", "")).strip() == model_id:
            if str(row.get("location", "")).strip() != location:
                break
            return
    raise RegistryWriteError(
        f"post-write verification failed for new model '{model_id}'; "
        f"file left unchanged. Check {path} for a formatting problem "
        f"near the end of the 'models:' list, then try again."
    )


def remove_model_entry(base_dir: Path | str, model_id: str) -> None:
    """Delete ONE model's block entirely from models.yaml (owner request
    2026-08-21: a Delete action for the Models page).

    The mirror image of append_model_entry, reusing the identical targeted,
    atomic, comment-and-key-order-preserving edit mechanism to remove a whole
    `- id: ...` block instead of adding one. Every other row's bytes -
    indentation, comments, key order - are untouched.

    _locate_model_block's forward scan already folds a block's trailing blank
    separator line into its own [start, end) range (it does not stop at a
    blank line, only at the next `- ` item or a top-level key), so deleting
    exactly lines[start:end] removes the row AND the blank line that used to
    separate it from the next entry, without leaving a double blank or a
    missing one - verified in test_remove_model_entry_leaves_a_clean_blank_line.

    Contract:
    - Raise ValueError if model_id is not found, rather than silently no-op-ing
      (the caller - a Delete button press - must know whether it did anything).
    - Re-parse through yaml.safe_load and confirm the id is GONE before
      committing, so a malformed edit leaves the original file untouched.
    - Write atomically (sibling temp file + os.replace).

    This writer only edits models.yaml. It never touches the filesystem
    location a row pointed at - actually deleting the model file from disk is
    the caller's separate, explicit step (gui_controller.delete_model), kept
    apart so a registry-only cleanup (the owner moved or renamed the file by
    hand) never forces a file deletion alongside it.
    """

    def edit(lines: list[str], path: Path) -> str:
        start, end = _locate_model_block(lines, model_id)
        if start is None:
            raise ValueError(
                f"model id '{model_id}' not found in {MODELS_FILE}; cannot delete"
            )
        del lines[start:end]
        new_text = "".join(lines)

        parsed = yaml.safe_load(new_text)
        _confirm_removed(parsed, model_id, path)
        return new_text

    _edit_registry(base_dir, edit)


def _confirm_removed(parsed: Any, model_id: str, path: Path) -> None:
    """Assert the re-parsed document no longer carries a row for model_id."""
    rows = parsed.get("models", []) if isinstance(parsed, dict) else parsed
    for row in rows or []:
        if isinstance(row, dict) and str(row.get("id", "")).strip() == model_id:
            raise _post_write_failure(
                f"- '{model_id}' is still present after delete", path
            )


def write_chat_ui(base_dir: Path | str, value: str) -> None:
    """Rewrite ONLY the chat.preferred_ui scalar in settings.yaml (M9-lite, G6 sibling).

    This is the "remember my choice" persistence for the chat-UI chooser
    (Architecture M9.3). It is the direct sibling of write_model_fields /
    write_model_score, reusing the identical targeted, atomic, comment-and-key-order-
    preserving line-rewrite mechanism, but against settings.yaml's `chat:` section
    instead of a models.yaml list item. It is deliberately NOT a yaml.safe_dump of
    the whole document, because a full re-dump would strip the owner's comments and
    reorder every key. Contract:

    - `value` must be one of 'ask', 'llamacpp', 'openwebui' (the enum invariant,
      Data Model section 10); raise ValueError with a remedy otherwise, so a bad
      value can never reach the file.
    - Locate the top-level `chat:` block, then rewrite only its `preferred_ui:`
      scalar on its existing line, preserving every other byte (indentation, key
      order, comments). If the section or key cannot be located, raise rather than
      guess (the same RG3 discipline as write_model_fields).
    - Re-parse the rewritten text through yaml.safe_load and confirm chat.preferred_ui
      now equals `value` BEFORE committing, so a malformed edit leaves the file
      untouched.
    - Write atomically (sibling temp file + os.replace).

    Path safety (Permission Matrix section 7/8): the file is always base_dir/
    settings.yaml resolved from the platform directory, so the write cannot escape
    Codebase/platform/.
    """
    if value not in ("ask", "llamacpp", "openwebui"):
        raise ValueError(
            f"chat.preferred_ui must be one of ask, llamacpp, openwebui "
            f"(got {value!r})"
        )

    base = Path(base_dir)
    path = base / SETTINGS_FILE
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    start, end = _locate_top_section(lines, "chat")
    if start is None:
        raise ValueError(
            f"'chat:' section not found in {SETTINGS_FILE}; cannot write "
            f"preferred_ui (refusing to guess)"
        )
    key_idx = _find_key_line(lines, start + 1, end, "preferred_ui")
    if key_idx is None:
        raise ValueError(
            f"could not locate 'preferred_ui:' line under 'chat:' in "
            f"{SETTINGS_FILE}; refusing to guess"
        )

    lines[key_idx] = _replace_scalar(lines[key_idx], value)
    new_text = "".join(lines)

    # Round-trip guard: confirm the edited text still parses AND chat.preferred_ui
    # now carries exactly the requested value before touching disk.
    parsed = yaml.safe_load(new_text)
    chat_section = parsed.get("chat", {}) if isinstance(parsed, dict) else {}
    if not isinstance(chat_section, dict) or str(
        chat_section.get("preferred_ui", "")
    ).strip() != value:
        raise ValueError(
            f"post-write verification failed for chat.preferred_ui; "
            f"file left unchanged"
        )

    _atomic_write(path, new_text)


def clear_launcher_default_model(base_dir: Path | str) -> None:
    """Clear ONLY ``launcher.default_model`` in settings.yaml.

    Model selection belongs to the chat request, not to persisted launcher
    state.  Keep this owner-data migration behind the same targeted, atomic,
    comment-preserving writer used for the other settings scalars.
    """
    base = Path(base_dir)
    path = base / SETTINGS_FILE
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    start, end = _locate_top_section(lines, "launcher")
    if start is None:
        raise ValueError(
            f"'launcher:' section not found in {SETTINGS_FILE}; cannot clear "
            "default_model (refusing to guess)"
        )
    key_idx = _find_key_line(lines, start + 1, end, "default_model")
    if key_idx is None:
        raise ValueError(
            f"could not locate 'default_model:' line under 'launcher:' in "
            f"{SETTINGS_FILE}; refusing to guess"
        )

    lines[key_idx] = _replace_value_line(lines[key_idx], "", quote=True)
    new_text = "".join(lines)

    parsed = yaml.safe_load(new_text)
    section = parsed.get("launcher", {}) if isinstance(parsed, dict) else {}
    if not isinstance(section, dict) or section.get("default_model") != "":
        raise ValueError(
            "post-write verification failed for launcher.default_model; "
            "file left unchanged"
        )

    _atomic_write(path, new_text)


def write_tts_voice(base_dir: Path | str, voice: str) -> None:
    """Rewrite ONLY the owner's ``tts.voice`` setting.

    Voice selection is user data, so keep it behind the same targeted,
    comment-preserving, atomic settings writer used by the chat and Open WebUI
    preferences.  The writer accepts a Kokoro voice stem (for example
    ``af_heart``), never a path; availability is checked by the caller against
    the configured voices directory before choosing it.
    """
    cleaned = str(voice or "").strip()
    if not cleaned or not all(ch.isalnum() or ch in "_-" for ch in cleaned):
        raise ValueError(
            "tts.voice must be a non-empty Kokoro voice name containing only "
            "letters, numbers, underscores, or hyphens"
        )

    base = Path(base_dir)
    path = base / SETTINGS_FILE
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    start, end = _locate_top_section(lines, "tts")
    if start is None:
        raise ValueError(
            f"'tts:' section not found in {SETTINGS_FILE}; cannot write voice "
            "(refusing to guess)"
        )
    key_idx = _find_key_line(lines, start + 1, end, "voice")
    if key_idx is None:
        raise ValueError(
            f"could not locate 'voice:' line under 'tts:' in {SETTINGS_FILE}; "
            "refusing to guess"
        )

    rendered = yaml.safe_dump(cleaned).split("\n", 1)[0]
    lines[key_idx] = _replace_value_line(lines[key_idx], rendered, quote=False)
    new_text = "".join(lines)

    parsed = yaml.safe_load(new_text)
    section = parsed.get("tts", {}) if isinstance(parsed, dict) else {}
    if not isinstance(section, dict) or str(section.get("voice", "")) != cleaned:
        raise ValueError(
            "post-write verification failed for tts.voice; file left unchanged"
        )

    _atomic_write(path, new_text)


def write_openwebui_enabled(base_dir: Path | str, enabled: bool) -> None:
    """Rewrite ONLY the openwebui.enabled scalar in settings.yaml (M17.11).

    Sibling of write_chat_ui, same targeted/atomic/comment-preserving mechanism.
    Why it exists: installing Open WebUI's venv is not enough for the chat chooser
    to use it - webui_available() also requires openwebui.enabled. So selecting the
    Open WebUI feature in setup must flip this flag on, or a freshly installed Open
    WebUI is silently never reached and chat falls back to the built-in llama.cpp
    UI (the exact defect an owner hit: venv present, enabled false, chat stuck on
    :8080). Never a full re-dump, which would strip the owner's comments.
    """
    base = Path(base_dir)
    path = base / SETTINGS_FILE
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    start, end = _locate_top_section(lines, "openwebui")
    if start is None:
        raise ValueError(
            f"'openwebui:' section not found in {SETTINGS_FILE}; cannot write "
            f"enabled (refusing to guess)"
        )
    key_idx = _find_key_line(lines, start + 1, end, "enabled")
    if key_idx is None:
        raise ValueError(
            f"could not locate 'enabled:' line under 'openwebui:' in "
            f"{SETTINGS_FILE}; refusing to guess"
        )

    lines[key_idx] = _replace_scalar(lines[key_idx], "true" if enabled else "false")
    new_text = "".join(lines)

    parsed = yaml.safe_load(new_text)
    section = parsed.get("openwebui", {}) if isinstance(parsed, dict) else {}
    if not isinstance(section, dict) or bool(section.get("enabled")) != enabled:
        raise ValueError(
            "post-write verification failed for openwebui.enabled; "
            "file left unchanged"
        )

    _atomic_write(path, new_text)


def write_openwebui_call_silence(base_dir: Path | str, silence_ms: int) -> None:
    """Rewrite (or insert) ONLY openwebui.call_silence_ms in settings.yaml.

    Sibling of write_openwebui_enabled, same targeted/atomic/comment-preserving
    mechanism. Inserts the line at the end of the openwebui: block when it is
    absent, which it is in every settings.yaml written before 2026-09-03 - the
    loader's default already covers those files, so nothing is lost by not
    touching them until the owner chooses a value.

    Validated against the same range the loader enforces, so this cannot put a
    value on disk that startup would then warn about and discard.
    """
    if (
        not isinstance(silence_ms, int)
        or isinstance(silence_ms, bool)
        or not CALL_SILENCE_MIN_MS <= silence_ms <= CALL_SILENCE_MAX_MS
    ):
        raise ValueError(
            f"openwebui.call_silence_ms must be an integer "
            f"{CALL_SILENCE_MIN_MS}..{CALL_SILENCE_MAX_MS}, got {silence_ms!r}"
        )
    base = Path(base_dir)
    path = base / SETTINGS_FILE
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    start, end = _locate_top_section(lines, "openwebui")
    if start is None:
        raise ValueError(
            f"'openwebui:' section not found in {SETTINGS_FILE}; cannot write "
            f"call_silence_ms (refusing to guess)"
        )
    key_idx = _find_key_line(lines, start + 1, end, "call_silence_ms")
    if key_idx is not None:
        lines[key_idx] = _replace_scalar(lines[key_idx], str(silence_ms))
    else:
        # Insert after the last real key of the block, matching its indent and
        # line ending, so trailing blank lines stay where they were.
        last_key = None
        for idx in range(start + 1, end):
            stripped = lines[idx].lstrip()
            if stripped and not stripped.startswith("#"):
                last_key = idx
        if last_key is None:
            raise ValueError(
                f"'openwebui:' section in {SETTINGS_FILE} has no keys; refusing "
                f"to guess its indentation"
            )
        indent = len(lines[last_key]) - len(lines[last_key].lstrip())
        newline = "\r\n" if lines[last_key].endswith("\r\n") else "\n"
        if not lines[last_key].endswith("\n"):
            lines[last_key] += newline
        lines.insert(last_key + 1, f"{' ' * indent}call_silence_ms: {silence_ms}{newline}")
    new_text = "".join(lines)

    parsed = yaml.safe_load(new_text)
    section = parsed.get("openwebui", {}) if isinstance(parsed, dict) else {}
    if not isinstance(section, dict) or section.get("call_silence_ms") != silence_ms:
        raise ValueError(
            "post-write verification failed for openwebui.call_silence_ms; "
            "file left unchanged"
        )

    _atomic_write(path, new_text)


def write_speech_noise_suppression(base_dir: Path | str, mode: str) -> None:
    """Rewrite or insert only speech.noise_suppression in settings.yaml.

    The writer follows the same targeted, atomic, comment-preserving contract as
    the Open WebUI settings writers. Arbitrary FFmpeg graphs are impossible: only
    the three validated preset names can reach disk.
    """
    normalized = str(mode or "").strip().lower()
    if normalized not in VALID_NOISE_SUPPRESSION:
        raise ValueError(
            "speech.noise_suppression must be one of "
            + ", ".join(VALID_NOISE_SUPPRESSION)
            + f", got {mode!r}"
        )
    base = Path(base_dir)
    path = base / SETTINGS_FILE
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    start, end = _locate_top_section(lines, "speech")
    if start is None:
        raise ValueError(
            f"'speech:' section not found in {SETTINGS_FILE}; cannot write "
            "noise_suppression (refusing to guess)"
        )
    key_idx = _find_key_line(lines, start + 1, end, "noise_suppression")
    if key_idx is not None:
        # Quote every enum value because YAML 1.1 parses the bare word "off" as
        # boolean false. The loaded type must remain a string in all modes.
        lines[key_idx] = _replace_value_line(lines[key_idx], normalized, quote=True)
    else:
        last_key = None
        for idx in range(start + 1, end):
            stripped = lines[idx].lstrip()
            if stripped and not stripped.startswith("#"):
                last_key = idx
        if last_key is None:
            raise ValueError(
                f"'speech:' section in {SETTINGS_FILE} has no keys; refusing "
                "to guess its indentation"
            )
        indent = len(lines[last_key]) - len(lines[last_key].lstrip())
        newline = "\r\n" if lines[last_key].endswith("\r\n") else "\n"
        if not lines[last_key].endswith("\n"):
            lines[last_key] += newline
        lines.insert(
            last_key + 1,
            f"{' ' * indent}noise_suppression: \"{normalized}\"{newline}",
        )
    new_text = "".join(lines)

    parsed = yaml.safe_load(new_text)
    section = parsed.get("speech", {}) if isinstance(parsed, dict) else {}
    if not isinstance(section, dict) or section.get("noise_suppression") != normalized:
        raise ValueError(
            "post-write verification failed for speech.noise_suppression; "
            "file left unchanged"
        )
    _atomic_write(path, new_text)


def write_chat_harness_dir(base_dir: Path | str, project_dir: str) -> None:
    """Rewrite ONLY chat_harness.last_project_dir in settings.yaml.

    Direct sibling of write_chat_ui: same targeted, atomic, comment-and-key-
    order-preserving line rewrite, against the chat_harness: section instead
    of chat:. Raises rather than guesses if the section/key is missing (RG3
    discipline) -- settings.default.yaml always seeds a chat_harness: block,
    so a real install's settings.yaml always has one; a test fixture that
    omits it is expected to hit this raise.
    """
    base = Path(base_dir)
    path = base / SETTINGS_FILE
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    start, end = _locate_top_section(lines, "chat_harness")
    if start is None:
        raise ValueError(
            f"'chat_harness:' section not found in {SETTINGS_FILE}; cannot write "
            f"last_project_dir (refusing to guess)"
        )
    key_idx = _find_key_line(lines, start + 1, end, "last_project_dir")
    if key_idx is None:
        raise ValueError(
            f"could not locate 'last_project_dir:' line under 'chat_harness:' in "
            f"{SETTINGS_FILE}; refusing to guess"
        )

    # A Windows project path routinely contains backslashes and spaces, so
    # _replace_scalar's bare-token substitution (and a naive '"value"' wrap)
    # are both unsafe here: a raw double-quoted YAML scalar interprets a
    # backslash as an escape sequence, and an arbitrary drive-letter path can
    # start with a sequence PyYAML rejects as invalid. yaml.safe_dump renders
    # a correctly quoted-or-plain scalar for any input;
    # its first line is exactly the value token _replace_value_line expects.
    safe_value = yaml.safe_dump(project_dir).split("\n", 1)[0]
    lines[key_idx] = _replace_value_line(lines[key_idx], safe_value, quote=False)
    new_text = "".join(lines)

    parsed = yaml.safe_load(new_text)
    section = parsed.get("chat_harness", {}) if isinstance(parsed, dict) else {}
    if not isinstance(section, dict) or str(
        section.get("last_project_dir", "")
    ) != project_dir:
        raise ValueError(
            f"post-write verification failed for chat_harness.last_project_dir; "
            f"file left unchanged"
        )

    _atomic_write(path, new_text)


def _locate_top_section(
    lines: list[str], section: str
) -> tuple[int | None, int | None]:
    """Return [start, end) line indices spanning a top-level `<section>:` block.

    `start` is the index of the `<section>:` header line (indent 0); `end` is the
    index of the next top-level key (indent 0, non-comment/blank) or end-of-file.
    Used to bound the search for a nested scalar so write_chat_ui edits only the key
    inside the intended section and never a same-named key elsewhere. Returns
    (None, None) when the section header is absent.
    """
    start = None
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent == 0 and stripped.startswith(f"{section}:"):
            start = idx
            break
    if start is None:
        return None, None
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[idx]) - len(lines[idx].lstrip())
        if indent == 0:
            end = idx
            break
    return start, end


def _confirm_score_written(
    parsed: Any, model_id: str, score: float, path: Path
) -> None:
    """Assert the re-parsed document carries the new benchmark_score for model_id."""
    rows = parsed.get("models", []) if isinstance(parsed, dict) else parsed
    for row in rows or []:
        if isinstance(row, dict) and str(row.get("id", "")).strip() == model_id:
            written = row.get("benchmark_score")
            if written is None or abs(float(written) - score) > 1e-6:
                raise _post_write_failure(
                    f"for '{model_id}' benchmark_score", path
                )
            return
    raise _post_write_failure(f"- could not find '{model_id}'", path)


def _locate_model_block(
    lines: list[str], model_id: str
) -> tuple[int | None, int | None]:
    """Return [start, end) line indices of the list item whose id is model_id.

    A model list item begins with a `- id: <value>` line. The block runs until the
    next line that starts a new list item at the same indentation (`- `) or a
    dedent to a top-level key, whichever comes first.
    """
    id_line_idx = None
    item_indent = 0
    for idx, line in enumerate(lines):
        parsed_id, indent = _parse_id_line(line)
        if parsed_id is not None and parsed_id == model_id:
            id_line_idx = idx
            item_indent = indent
            break
    if id_line_idx is None:
        return None, None

    # Walk forward to the start of the next list item (a '- ' at <= item_indent)
    # or a top-level key (indent 0 that is not a comment/blank), which ends this
    # block.
    end = len(lines)
    for idx in range(id_line_idx + 1, len(lines)):
        stripped = lines[idx].lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[idx]) - len(lines[idx].lstrip())
        if stripped.startswith("- ") and indent <= item_indent:
            end = idx
            break
        if indent == 0:
            end = idx
            break
    return id_line_idx, end


def _parse_id_line(line: str) -> tuple[str | None, int]:
    """If `line` is a `- id: <value>` entry, return (value, indent); else (None, 0)."""
    stripped = line.lstrip()
    indent = len(line) - len(stripped)
    if not stripped.startswith("- "):
        return None, indent
    rest = stripped[2:].lstrip()
    if not rest.startswith("id:"):
        return None, indent
    value = rest[len("id:") :].strip()
    # Strip an inline comment and surrounding quotes so 'qwen3-14b' == "qwen3-14b".
    value = value.split("#", 1)[0].strip().strip("'\"")
    return value, indent


def _find_key_line(
    lines: list[str], start: int, end: int, key: str
) -> int | None:
    """Index of the first `<key>:` line within [start, end), or None."""
    for idx in range(start, end):
        stripped = lines[idx].lstrip()
        if stripped.startswith(f"{key}:"):
            return idx
    return None


def _replace_scalar(line: str, new_value: Any) -> str:
    """Rewrite the scalar value on a `key: value  # comment` line, keeping the rest.

    Preserves the line's newline, indentation, key, separator, and any trailing
    inline comment; only the value token is replaced.
    """
    import re

    global _SCALAR_LINE
    if _SCALAR_LINE is None:
        # prefix = indent + key + ':' + spaces ; value = the scalar ; suffix = rest.
        _SCALAR_LINE = re.compile(r"^(\s*[^:\s]+:\s*)(\S+)(.*)$")
    # Split off the line ending first: the lines come from splitlines(keepends=True)
    # so each carries its own newline, but the regex '$' stops before it. Preserving
    # the exact terminator (LF or CRLF) is what keeps the rest of the file intact.
    newline = ""
    body = line
    if body.endswith("\r\n"):
        body, newline = body[:-2], "\r\n"
    elif body.endswith("\n"):
        body, newline = body[:-1], "\n"
    match = _SCALAR_LINE.match(body)
    if match is None:  # pragma: no cover - callers only pass matched key lines
        raise ValueError(f"cannot parse scalar line: {line!r}")
    return f"{match.group(1)}{new_value}{match.group(3)}{newline}"


def _replace_value_line(line: str, new_value: str, quote: bool) -> str:
    """Rewrite the value of a `[- ]key: <value>[  # comment]` line, spaces allowed.

    _replace_scalar only handles single-token values (its (\\S+) capture stops at
    the first space), which breaks on a multi-word model name. This variant treats
    everything between the key's colon and an optional trailing inline comment as
    the value, so "Qwen3 14B" round-trips intact. Preserves the line's leading
    list-item marker/indentation, the key, and any trailing comment verbatim;
    `quote` wraps the new value in double quotes to match this file's convention
    for name: values (id: values are written bare).
    """
    import re

    newline = ""
    body = line
    if body.endswith("\r\n"):
        body, newline = body[:-2], "\r\n"
    elif body.endswith("\n"):
        body, newline = body[:-1], "\n"

    key_match = re.match(r"^(\s*(?:-\s*)?[^:\s]+:\s*)", body)
    if key_match is None:  # pragma: no cover - callers only pass matched key lines
        raise ValueError(f"cannot parse key/value line: {line!r}")
    prefix = key_match.group(1)
    rest = body[len(prefix):]

    comment_match = re.search(r"(\s+#.*)$", rest)
    comment = comment_match.group(1) if comment_match else ""

    rendered = f'"{new_value}"' if quote else new_value
    return f"{prefix}{rendered}{comment}{newline}"


def _post_write_failure(detail: str, path: Path) -> "RegistryWriteError":
    """Build the one post-write-verification sentence, with the real path.

    DEC-M14-11's fourth cause. The three sibling confirm helpers below all end
    the same way, so the wording lives once: what failed, that the file was NOT
    touched, and the next step.
    """
    return RegistryWriteError(
        f"post-write verification failed {detail}; file left unchanged. "
        f"Check {path} for a formatting problem, then try again."
    )


def _confirm_identity_written(
    parsed: Any, model_id: str, name: str, path: Path
) -> None:
    """Assert the re-parsed document carries the new id/name for model_id."""
    rows = parsed.get("models", []) if isinstance(parsed, dict) else parsed
    for row in rows or []:
        if isinstance(row, dict) and str(row.get("id", "")).strip() == model_id:
            if str(row.get("name", "")).strip() != name:
                raise _post_write_failure(f"for '{model_id}' name", path)
            return
    raise _post_write_failure(f"- could not find '{model_id}'", path)


def _confirm_capabilities_written(
    parsed: Any, model_id: str, capabilities: list[str], path: Path
) -> None:
    """Assert the re-parsed document carries the new capabilities for model_id."""
    rows = parsed.get("models", []) if isinstance(parsed, dict) else parsed
    for row in rows or []:
        if isinstance(row, dict) and str(row.get("id", "")).strip() == model_id:
            got = [str(c).strip() for c in (row.get("capabilities") or [])]
            if got != capabilities:
                raise _post_write_failure(f"for '{model_id}' capabilities", path)
            return
    raise _post_write_failure(f"- could not find '{model_id}'", path)


def _confirm_written(
    parsed: Any, model_id: str, gpu_layers: int, context_size: int, path: Path
) -> None:
    """Assert the re-parsed document carries the new values for model_id."""
    rows = parsed.get("models", []) if isinstance(parsed, dict) else parsed
    for row in rows or []:
        if isinstance(row, dict) and str(row.get("id", "")).strip() == model_id:
            if int(row.get("gpu_layers")) != gpu_layers or int(
                row.get("context_size")
            ) != context_size:
                raise _post_write_failure(f"for '{model_id}'", path)
            return
    raise _post_write_failure(f"- could not find '{model_id}'", path)


def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to a sibling temp file, then os.replace over `path`."""
    import tempfile

    directory = path.parent
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(directory),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
        newline="",
    )
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, path)
    except BaseException:
        # Never leave the temp file behind if the replace failed.
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #


def _safe_load(path: Path, issues: list[ConfigIssue]) -> Any:
    """yaml.safe_load a file, turning a parse error into a reported issue."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        issues.append(ConfigIssue("ERROR", path.name, f"YAML parse error: {exc}"))
        return None
    except OSError as exc:
        issues.append(ConfigIssue("ERROR", path.name, f"could not read: {exc}"))
        return None


def _build(cls: type, raw: Any, source: str, issues: list[ConfigIssue]) -> Any:
    """Construct a dataclass from a raw mapping, ignoring unknown keys.

    Unknown keys are reported as WARNING (Architecture section 7) rather than
    raising, so an owner typo does not break startup.
    """
    instance = cls()
    if raw is None:
        return instance
    if not isinstance(raw, dict):
        issues.append(ConfigIssue("WARNING", source, f"{cls.__name__} is not a mapping"))
        return instance
    known = {f.name: f for f in fields(cls)}
    for key, value in raw.items():
        if key not in known:
            issues.append(
                ConfigIssue("WARNING", source, f"unknown key '{key}' in {cls.__name__}")
            )
            continue
        setattr(instance, key, value)
    return instance


def _env_flag(raw: str, default: bool) -> bool:
    """Read a boolean from an environment variable, ignoring a typo.

    Only the four unambiguous spellings each way are honoured. Anything else
    keeps the configured value rather than guessing, because silently reading
    "flase" as false would turn a typo into a disabled feature the user cannot
    explain.
    """
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _int(value: Any, default: int) -> int:
    """Coerce to int, falling back to default on None or bad input."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _opt_float(value: Any) -> float | None:
    """Coerce to float or None (for benchmark_score which is nullable)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_str(value: Any) -> str | None:
    """Coerce a present scalar to a non-empty string, else None (M5 optional field).

    An absent/empty/null value is None (feature off); a set value becomes its
    string form. Used for the optional draft_model reference.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_mapping(value: Any) -> dict[str, Any] | None:
    """Return value if it is a non-empty mapping, else None (M5 optional field).

    Used for spec_config / benchmark_sweep, which are optional mappings of scalar
    knobs. A non-mapping or empty value is treated as absent (feature off) rather
    than raising, so a malformed entry degrades to the default single-config
    behaviour instead of breaking registry load.
    """
    if isinstance(value, dict) and value:
        return value
    return None


# The three keys models.py knows how to turn into llama-server reasoning flags.
# Named here (not in models.py) because this is where a row is VALIDATED; the
# flag spellings stay single-sourced in models.py, per the spec_config precedent.
REASONING_KEYS = ("enabled", "effort", "budget")


def parse_reasoning(value: Any, model_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate a row's `reasoning:` mapping, returning (cleaned, problems).

    Shape only - see Model.reasoning for why the effort VOCABULARY is the chat
    template's business and not LOCITIZE's. What is checked here is what LOCITIZE
    can be certain about from llama-server's own --help (build b10701, confirmed
    2026-09-02 by running the installed binary):

      -rea, --reasoning [on|off|auto]   default 'auto' (detect from template)
      --reasoning-effort LEVEL          'default' keeps the template default
      --reasoning-budget N              -1 unrestricted, 0 immediate end, N>0 cap

    So: `enabled` must be a bool, `effort` a non-empty string, `budget` an int
    >= -1. An unknown key is REPORTED rather than dropped silently - unlike
    spec_config, where an unrecognised key is ignored. The difference is
    deliberate: a misspelled speculation knob costs throughput, while a
    misspelled reasoning knob silently leaves a model thinking at its template
    default (xhigh, unbudgeted, on this machine) when the owner believed they
    had turned it down.

    Returns (None, problems) when nothing usable survives, so a bad mapping
    degrades to llama-server's defaults instead of breaking registry load.
    """
    problems: list[str] = []
    if value is None:
        return None, problems
    if not isinstance(value, dict) or not value:
        if value:
            problems.append(
                f"model '{model_id}' reasoning must be a mapping of "
                f"{'/'.join(REASONING_KEYS)}; ignoring it"
            )
        return None, problems

    cleaned: dict[str, Any] = {}
    for key, raw in value.items():
        name = str(key).strip()
        if name not in REASONING_KEYS:
            problems.append(
                f"model '{model_id}' reasoning has unknown key '{name}'; "
                f"supported keys are {', '.join(REASONING_KEYS)}"
            )
            continue
        if raw is None:
            continue
        if name == "enabled":
            if not isinstance(raw, bool):
                problems.append(
                    f"model '{model_id}' reasoning.enabled must be true or false "
                    f"(got {raw!r}); ignoring it"
                )
                continue
            cleaned["enabled"] = raw
        elif name == "effort":
            text = str(raw).strip()
            if not text:
                problems.append(
                    f"model '{model_id}' reasoning.effort must be a non-empty "
                    f"level accepted by this model's chat template; ignoring it"
                )
                continue
            cleaned["effort"] = text
        else:  # budget
            if isinstance(raw, bool):
                problems.append(
                    f"model '{model_id}' reasoning.budget must be an integer "
                    f">= -1 (got {raw!r}); ignoring it"
                )
                continue
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                problems.append(
                    f"model '{model_id}' reasoning.budget must be an integer "
                    f">= -1 (got {raw!r}); ignoring it"
                )
                continue
            if parsed < -1:
                problems.append(
                    f"model '{model_id}' reasoning.budget must be -1 "
                    f"(unrestricted), 0 (no thinking), or a positive token cap; "
                    f"got {parsed}, ignoring it"
                )
                continue
            cleaned["budget"] = parsed
    return (cleaned or None), problems


def parse_reasoning_choice(
    text: str, model_id: str = ""
) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse the compact launch-time thinking selector into a reasoning mapping.

    One grammar, shared by every place an owner PICKS a level rather than
    writing one into models.yaml (the interactive menu's "5 low" suffix and the
    --reasoning CLI flag), so the two can never drift:

        ""            -> None          use the row's own reasoning:, unchanged
        "off"         -> enabled False turn thinking off for this launch
        "on"          -> enabled True  force thinking on
        "low"         -> effort low    any level the model's template accepts
        "low/2048"    -> effort low + budget 2048
        "/2048"       -> budget 2048 only, leaving effort at the row/template value

    The level is NOT checked against a list here for the same reason
    parse_reasoning does not check it: the vocabulary belongs to the model's
    chat template. A budget that is not an integer >= -1 is reported, not
    guessed at.

    Returns (None, problems) when nothing usable parsed, so a typo at the menu
    prompt reports itself instead of silently starting the model at a setting
    the owner did not choose.
    """
    raw = (text or "").strip()
    if not raw:
        return None, []
    effort_text, _, budget_text = raw.partition("/")
    effort_text = effort_text.strip()
    budget_text = budget_text.strip()

    choice: dict[str, Any] = {}
    if effort_text.lower() in ("off", "on"):
        choice["enabled"] = effort_text.lower() == "on"
    elif effort_text:
        choice["effort"] = effort_text
    if budget_text:
        try:
            choice["budget"] = int(budget_text)
        except ValueError:
            return None, [
                f"reasoning budget must be an integer (got {budget_text!r}); "
                f"use for example 'low/2048', 'off', or '/0'"
            ]
    if not choice:
        return None, [f"could not read a thinking level from {raw!r}"]
    return parse_reasoning(choice, model_id or "selection")


def _opt_positive_int(value: Any) -> int | None:
    """Coerce to a strictly-positive int, or None for absent/invalid (D-M4-2).

    Used for the optional per-model ready_timeout_s. A bool is rejected (True/False
    would coerce to 1/0 and mask a config mistake); zero and negatives are rejected
    so a bad value never yields a non-positive timeout.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
