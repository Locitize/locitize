"""First-run setup: the feature catalogue and the pure planning logic (M15).

This is the setup wizard's decision layer, and it is the webui.resolve_chat_choice
analog for onboarding: no tkinter, no subprocess, no network, no filesystem. It maps
(features the user ticked) x (what this machine already has) onto one ordered
InstallPlan, so the Tkinter bootstrap wizard and any future Qt/CLI front end share a
single, exhaustively unit-tested path instead of each growing their own rules.

Three ideas carry the whole module:

- A Feature is what a person wants ("talk to it", "hear it back"). It is the only
  vocabulary the wizard shows.
- A Requirement is what that costs (a pip set, a binary, a weights file). Features
  name requirements; requirements are deduplicated across features, so ticking both
  Vision and Second eye plans one projector download, not two.
- A Step is one unit of work with a size, a reason, and an idempotent skip rule.
  plan_for() emits them already ordered and already filtered against detected state,
  so the executor stays a dumb loop and the UI can total the bytes before anything
  touches the network.

Feature closure is explicit rather than implied: second_eye is useless without vision
and voice_out, so selecting it selects those, visibly, in the checklist. The user is
never silently charged for a download they did not see ticked.

Nothing here decides WHETHER to install; the user does that, and every network step
carries its own size so the consent screen can state the real number. ASCII only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

# Requirement kinds. The executor dispatches on these; the planner only groups them.
KIND_PIP = "pip"          # packages into the platform venv
KIND_BINARY = "binary"    # a fetched, verified executable under <data root>/bin
KIND_WEIGHTS = "weights"  # a fetched, verified non-GGUF model file (whisper/kokoro)
KIND_MODEL = "model"      # a GGUF chosen through the existing Get-models flow
KIND_WINGET = "winget"    # a user-scope winget package (never elevated)
KIND_VENV = "venv"        # a separate, isolated virtual environment

# Ordering weight per kind. A venv must exist before pip runs into it; binaries and
# weights are independent; the heavy optional venvs go last so a failure there never
# blocks an otherwise working chat install.
_KIND_ORDER = {
    KIND_VENV: 0,
    KIND_PIP: 1,
    KIND_BINARY: 2,
    KIND_WEIGHTS: 3,
    KIND_MODEL: 4,
    KIND_WINGET: 5,
}


@dataclass(frozen=True)
class Requirement:
    """One installable thing, named once and shared by every feature that needs it."""

    key: str
    kind: str
    label: str
    detail: str
    # Download/install size in MB. None means "not a download" (a venv, a local scan).
    size_mb: int | None = None
    # True when a failure here leaves the whole product unusable. Optional
    # requirements degrade to a warning and the wizard still finishes.
    essential: bool = False


@dataclass(frozen=True)
class Feature:
    """One capability as the user thinks about it, not as the installer sees it."""

    key: str
    label: str
    summary: str
    requires: tuple[str, ...]
    # Features this one is useless without. Selecting this selects those too.
    implies: tuple[str, ...] = ()
    default_on: bool = False
    # core features cannot be unticked: without them there is no product to run.
    core: bool = False


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------
# Sizes are the honest download cost measured at build time, rounded up. They exist
# so the consent screen can say "this will download 1.4 GB" instead of "this may take
# a while", which is the difference between a choice and a surprise.

REQUIREMENTS: dict[str, Requirement] = {
    r.key: r
    for r in (
        Requirement(
            "platform_venv", KIND_VENV, "Python environment",
            "A private .venv for LOCITIZE, so nothing is installed system-wide.",
            None, essential=True,
        ),
        Requirement(
            "base_pip", KIND_PIP, "Core Python packages",
            "PyYAML and psutil - config parsing and the health probes.",
            15, essential=True,
        ),
        Requirement(
            "gui_pip", KIND_PIP, "Desktop UI (PySide6)",
            "The Qt toolkit the main LOCITIZE window is built on.",
            95, essential=True,
        ),
        Requirement(
            "llama_cpp", KIND_BINARY, "llama.cpp server",
            "The inference engine that actually runs your models. The build is "
            "chosen from the detected GPU: CUDA when an NVIDIA card is present, "
            "CPU otherwise.",
            360, essential=True,
        ),
        Requirement(
            "first_model", KIND_MODEL, "A model to talk to",
            "Models already on this machine are found and imported - LOCITIZE "
            "ships no model list and copies nothing. If you have none, the "
            "Models page will help you find one.",
            None, essential=True,
        ),
        Requirement(
            "whisper_bin", KIND_BINARY, "whisper.cpp server",
            "Speech-to-text engine plus the microphone streaming binary.",
            45,
        ),
        Requirement(
            "whisper_model", KIND_WEIGHTS, "Whisper weights",
            "The ggml speech model whisper.cpp transcribes with.",
            150,
        ),
        Requirement(
            "kokoro_pip", KIND_PIP, "Kokoro TTS engine",
            "The kokoro package on CPU torch. The CPU wheel is deliberate: an 82M "
            "voice model needs no GPU and must not contend for VRAM with the LLM.",
            250,
        ),
        Requirement(
            "kokoro_pip_cuda", KIND_PIP, "Kokoro TTS engine on the GPU",
            "Swaps the CPU torch for the CUDA build so a spoken sentence renders "
            "in about a tenth of a second instead of most of one (measured 0.10s "
            "vs 0.78s). Costs a 2.8 GB download and about 0.5 GB of VRAM the "
            "model no longer gets. Needs an NVIDIA card with driver 570 or newer.",
            2800,
        ),
        Requirement(
            "kokoro_weights", KIND_WEIGHTS, "Kokoro voice weights",
            "The kokoro-v1_0 checkpoint and its voice tensors.",
            350,
        ),
        Requirement(
            "pillow_pip", KIND_PIP, "Screen capture (Pillow)",
            "Grabs and downscales frames for the second-eye change detector.",
            5,
        ),
        Requirement(
            "vision_projector", KIND_WEIGHTS, "Vision projector",
            "The mmproj file a vision model needs alongside its GGUF.",
            600,
        ),
        Requirement(
            "node_runtime", KIND_WINGET, "Node.js LTS",
            "Required by two of the three coding harnesses. User-scope winget "
            "install, no elevation.",
            30,
        ),
        Requirement(
            "harness_clis", KIND_WINGET, "Coding harness CLIs",
            "Claude Code, Codex, and OpenCode, pointed at your local model.",
            120,
        ),
        Requirement(
            "caddy", KIND_WINGET, "Caddy",
            "Serves the running model over https://locitize.local. User-scope "
            "winget install; the certificate trust step stays current-user.",
            20,
        ),
        Requirement(
            "webui_venv", KIND_VENV, "Open WebUI environment",
            "An isolated venv so Open WebUI's heavy dependency tree never touches "
            "the platform venv.",
            None,
        ),
        Requirement(
            "webui_pip", KIND_PIP, "Open WebUI",
            "Rich chat UI with history and uploads. Large: it pulls its own stack.",
            2600,
        ),
        Requirement(
            "finetune_venv", KIND_VENV, "Fine-tune studio environment",
            "An isolated venv for the bundled QLoRA studio.",
            None,
        ),
        Requirement(
            "finetune_pip", KIND_PIP, "Fine-tune studio",
            "Streamlit and the trainer front end. Training itself needs Docker, "
            "which LOCITIZE does not install for you.",
            400,
        ),
    )
}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

FEATURES: tuple[Feature, ...] = (
    Feature(
        "core", "Run and chat with local models",
        "The engine, the desktop window, and one model to talk to. This is "
        "LOCITIZE; everything else is optional.",
        requires=("platform_venv", "base_pip", "gui_pip", "llama_cpp", "first_model"),
        default_on=True, core=True,
    ),
    Feature(
        "voice_in", "Talk to it",
        "Speak instead of typing. Microphone capture and transcription, on-device.",
        requires=("whisper_bin", "whisper_model"),
        default_on=True,
    ),
    Feature(
        "voice_out", "Hear it back",
        "Spoken replies in one of eight voices, synthesized on CPU so the "
        "model keeps the whole GPU.",
        requires=("kokoro_pip", "kokoro_weights"),
        default_on=True,
    ),
    Feature(
        "voice_out_gpu", "Hear it back, faster",
        "The same voices rendered on the NVIDIA card: the first spoken word "
        "lands about 0.7s sooner on every turn of a voice call. Off by default "
        "because it takes VRAM from the model and the download is large.",
        requires=("kokoro_pip_cuda",),
        implies=("voice_out",),
    ),
    Feature(
        "vision", "Look at images",
        "Ask questions about a screenshot or a photo with a vision model.",
        requires=("vision_projector",),
    ),
    Feature(
        "second_eye", "Watch my screen",
        "A continuous second eye: it judges what changed on screen against a goal "
        "you state, and speaks a correction. Needs vision and a voice.",
        requires=("pillow_pip",),
        implies=("vision", "voice_out"),
    ),
    Feature(
        "harnesses", "Code with a local model",
        "Open Claude Code, Codex or OpenCode in a terminal, pointed at your own "
        "model instead of a cloud API.",
        requires=("node_runtime", "harness_clis"),
    ),
    Feature(
        "openwebui", "Open WebUI (rich chat UI)",
        "The full Open WebUI chat app as a managed local service: conversation "
        "history, uploads, web search, multi-model. Without this, chat uses "
        "llama.cpp's built-in minimal UI. Large: it pulls its own stack. "
        "Separately licensed third-party software.",
        requires=("webui_venv", "webui_pip"),
        default_on=True,
    ),
    Feature(
        "finetune", "Fine-tune models",
        "The bundled QLoRA studio for training on your own data. Training runs "
        "need Docker Desktop, installed separately.",
        requires=("finetune_venv", "finetune_pip"),
    ),
    Feature(
        "secure_proxy", "https://locitize.local",
        "Serve the running model under a friendly trusted name instead of a "
        "loopback port number.",
        requires=("caddy",),
    ),
)

FEATURES_BY_KEY: dict[str, Feature] = {f.key: f for f in FEATURES}
CORE_KEYS: tuple[str, ...] = tuple(f.key for f in FEATURES if f.core)
DEFAULT_SELECTION: tuple[str, ...] = tuple(f.key for f in FEATURES if f.default_on)


@dataclass(frozen=True)
class Step:
    """One planned unit of work, already checked against what the machine has."""

    requirement: Requirement
    # Features that asked for this step, for the "why am I downloading this" line.
    wanted_by: tuple[str, ...]
    satisfied: bool = False

    @property
    def key(self) -> str:
        return self.requirement.key

    @property
    def size_mb(self) -> int:
        """What this step will actually pull. A satisfied step costs nothing."""
        if self.satisfied or self.requirement.size_mb is None:
            return 0
        return self.requirement.size_mb


@dataclass
class InstallPlan:
    """The ordered work list plus everything the consent screen needs to say."""

    selected: tuple[str, ...] = ()
    steps: tuple[Step, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pending(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if not s.satisfied)

    @property
    def satisfied(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if s.satisfied)

    @property
    def total_mb(self) -> int:
        return sum(s.size_mb for s in self.steps)

    @property
    def nothing_to_do(self) -> bool:
        return not self.pending


def expand_selection(selected: Iterable[str]) -> tuple[str, ...]:
    """Return the selection plus core plus every implied feature, in catalogue order.

    Closure is transitive and terminates because `implies` is a DAG over a fixed
    catalogue. Unknown keys are dropped rather than raising: the wizard must never
    fail to draw because a stale saved selection names a feature since renamed.
    """
    chosen = {key for key in selected if key in FEATURES_BY_KEY}
    chosen.update(CORE_KEYS)

    changed = True
    while changed:
        changed = False
        for key in tuple(chosen):
            for implied in FEATURES_BY_KEY[key].implies:
                if implied in FEATURES_BY_KEY and implied not in chosen:
                    chosen.add(implied)
                    changed = True

    return tuple(f.key for f in FEATURES if f.key in chosen)


def requirements_for(
    selected: Iterable[str],
) -> tuple[tuple[Requirement, tuple[str, ...]], ...]:
    """Map an expanded selection onto deduplicated (requirement, wanted_by) pairs."""
    wanted: dict[str, list[str]] = {}
    for feature_key in expand_selection(selected):
        for req_key in FEATURES_BY_KEY[feature_key].requires:
            if req_key in REQUIREMENTS:
                wanted.setdefault(req_key, []).append(feature_key)

    pairs = [(REQUIREMENTS[key], tuple(names)) for key, names in wanted.items()]
    pairs.sort(key=lambda pair: (_KIND_ORDER.get(pair[0].kind, 99), pair[0].key))
    return tuple(pairs)


def plan_for(
    selected: Iterable[str],
    state: Mapping[str, bool] | None = None,
    warnings: Iterable[str] = (),
) -> InstallPlan:
    """Build the ordered plan for `selected` against detected `state`.

    `state` maps a requirement key to True when this machine already satisfies it.
    A missing key means "not satisfied", so a caller that can only detect some of
    them still gets a correct, if pessimistic, plan rather than a wrong one.
    """
    facts = dict(state or {})
    expanded = expand_selection(selected)
    steps = tuple(
        Step(requirement=req, wanted_by=names, satisfied=bool(facts.get(req.key, False)))
        for req, names in requirements_for(expanded)
    )
    return InstallPlan(selected=expanded, steps=steps, warnings=tuple(warnings))


def format_size(total_mb: int) -> str:
    """Render a download total the way a person reads it. Never 'unknown'."""
    if total_mb <= 0:
        return "nothing to download"
    if total_mb < 1024:
        return f"{total_mb} MB"
    return f"{total_mb / 1024:.1f} GB"


def summarize(plan: InstallPlan) -> str:
    """One honest line for the consent screen. ASCII, no emoji, no hype."""
    if plan.nothing_to_do:
        return "Everything the selected features need is already installed."
    count = len(plan.pending)
    noun = "item" if count == 1 else "items"
    already = len(plan.satisfied)
    tail = f" {already} already present." if already else ""
    return f"{count} {noun} to install, {format_size(plan.total_mb)} to download.{tail}"
