"""Local provider paths and labels adapted from Session Portal."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# -- User / provider paths ---------------------------------------------------
CLAUDE_DIR = Path.home() / ".claude"
HISTORY_FILE = CLAUDE_DIR / "history.jsonl"
PROJECTS_DIR = CLAUDE_DIR / "projects"

CODEX_DIR = Path.home() / ".codex"
CODEX_INDEX_FILE = CODEX_DIR / "session_index.jsonl"
CODEX_SESSIONS_DIR = CODEX_DIR / "sessions"
CODEX_EXE_DIR = Path.home() / "AppData" / "Local" / "OpenAI" / "Codex" / "bin"
CODEX_PROGRAMS_EXE_DIR = Path.home() / "AppData" / "Local" / "Programs" / "OpenAI" / "Codex" / "bin"

GROK_DIR = Path.home() / ".grok"
GROK_SESSIONS_DIR = GROK_DIR / "sessions"
GROK_MODELS_FILE = GROK_DIR / "models_cache.json"
GROK_EXE = GROK_DIR / "bin" / "grok.exe"

APPDATA_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
LOCALAPPDATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))

# Gemini CLI: chats live under ~/.gemini/tmp/<project-slug-or-sha256>/chats/.
GEMINI_DIR = Path.home() / ".gemini"
GEMINI_TMP_DIR = GEMINI_DIR / "tmp"
GEMINI_SETTINGS_FILE = GEMINI_DIR / "settings.json"

# Qwen Code: current builds record Claude-style JSONL under
# ~/.qwen/projects/<sanitized-cwd>/chats/; older builds used ~/.qwen/tmp/<hash>/chats/.
QWEN_DIR = Path.home() / ".qwen"
QWEN_PROJECTS_DIR = QWEN_DIR / "projects"
QWEN_TMP_DIR = QWEN_DIR / "tmp"

# opencode: xdg-basedir resolves ~/.local/share even on Windows. Current
# releases store sessions in opencode.db (SQLite); older installs used JSON
# trees under project/<slug>/storage/ or storage/.
OPENCODE_DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "opencode"
OPENCODE_DB_FILE = OPENCODE_DATA_DIR / "opencode.db"

COPILOT_DIR = Path.home() / ".copilot"
COPILOT_SESSIONS_DIR = COPILOT_DIR / "session-state"

AMP_DIR = Path.home() / ".amp"
AMP_CONFIG_DIR = Path.home() / ".config" / "amp"
AMP_DATA_DIR = Path.home() / ".local" / "share" / "amp"
AMP_LOCALAPPDATA_DIR = LOCALAPPDATA_DIR / "amp"

# LOCITIZE owns annotations; provider histories remain at their original paths.
CREATE_NO_WINDOW = getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0)
MAX_METADATA_SCAN_BYTES = 2 * 1024 * 1024
MAX_JSON_LINE_CHARS = 400_000
MAX_PREVIEW_MESSAGE_CHARS = 600

# -- Provider catalog (resumable sources) -----------------------------------
PROVIDER_OPTIONS: dict[str, dict] = {
    "claude": {
        "label": "Claude Code",
        "description": "Claude Code sessions and history",
        "path": str(CLAUDE_DIR),
        "paths": [CLAUDE_DIR],
        "commands": ["claude"],
    },
    "codex": {
        "label": "Codex",
        "description": "Codex sessions",
        "path": str(CODEX_DIR),
        "paths": [CODEX_DIR, CODEX_EXE_DIR, CODEX_PROGRAMS_EXE_DIR],
        "commands": ["codex"],
    },
    "grok": {
        "label": "Grok",
        "description": "Grok CLI sessions",
        "path": str(GROK_DIR),
        "paths": [GROK_DIR, GROK_EXE],
        "commands": ["grok"],
    },
    "copilot": {
        "label": "Copilot",
        "description": "GitHub Copilot CLI sessions",
        "path": str(COPILOT_DIR),
        "paths": [COPILOT_DIR, COPILOT_SESSIONS_DIR, LOCALAPPDATA_DIR / "copilot",
                  LOCALAPPDATA_DIR / "GitHub CLI" / "copilot"],
        "commands": ["gh"],
    },
    "amp": {
        "label": "AMP",
        "description": "AMP CLI threads",
        "path": str(AMP_DATA_DIR),
        "paths": [AMP_DIR, AMP_CONFIG_DIR, AMP_DATA_DIR, AMP_LOCALAPPDATA_DIR],
        "commands": ["amp"],
    },
    "gemini": {
        "label": "Gemini CLI",
        "description": "Gemini CLI sessions",
        "path": str(GEMINI_DIR),
        "paths": [GEMINI_TMP_DIR, GEMINI_DIR],
        "commands": ["gemini"],
    },
    "qwen": {
        "label": "Qwen Code",
        "description": "Qwen Code sessions",
        "path": str(QWEN_DIR),
        "paths": [QWEN_PROJECTS_DIR, QWEN_TMP_DIR, QWEN_DIR],
        "commands": ["qwen"],
    },
    "opencode": {
        "label": "OpenCode",
        "description": "opencode sessions",
        "path": str(OPENCODE_DATA_DIR),
        "paths": [OPENCODE_DATA_DIR],
        "commands": ["opencode"],
    },
}

# -- Other local AI tools (detected, listed, but not resumable) --------------
OTHER_AI_TOOLS: dict[str, dict] = {
    "cursor": {
        "label": "Cursor",
        "paths": [APPDATA_DIR / "Cursor", LOCALAPPDATA_DIR / "Programs" / "Cursor"],
        "commands": ["cursor"],
    },
    "windsurf": {
        "label": "Windsurf",
        "paths": [APPDATA_DIR / "Windsurf", LOCALAPPDATA_DIR / "Programs" / "Windsurf"],
        "commands": ["windsurf"],
    },
    "continue": {
        "label": "Continue",
        "paths": [Path.home() / ".continue",
                  APPDATA_DIR / "Code" / "User" / "globalStorage" / "continue.continue"],
        "commands": [],
    },
    "aider": {
        "label": "Aider",
        "paths": [Path.home() / ".aider.conf.yml", Path.home() / ".aider.model.settings.yml"],
        "commands": ["aider"],
    },
    "ollama": {
        "label": "Ollama",
        "paths": [Path.home() / ".ollama"],
        "commands": ["ollama"],
    },
    "lmstudio": {
        "label": "LM Studio",
        "paths": [Path.home() / ".lmstudio", APPDATA_DIR / "LM Studio"],
        "commands": [],
    },
}

DEFAULT_SETTINGS = {
    "onboarding_complete": False,
    "providers": {key: True for key in PROVIDER_OPTIONS},
    "auto_scan_enabled": True,
    "auto_scan_interval_ms": 60000,
    "egress_watch_enabled": True,
    "egress_watch_interval_ms": 300000,
}

# -- Theme ------------------------------------------------------------------
APP_PALETTE = {
    "bg": "#11111b",
    "bg_deep": "#090910",
    "surface": "#242438",
    "surface_2": "#1b1b2b",
    "overlay": "#5f6682",
    "select": "#3d55a6",
    "bar": "#090910",
    "muted": "#d4dcf5",
    "text": "#ffffff",
}

ACCENT = {
    "blue": "#9fc5ff",
    "green": "#b8f7b3",
    "yellow": "#ffe7a3",
    "pink": "#ff7ad9",
    "purple": "#d8b4ff",
    "orange": "#f9c784",
    "danger": "#ff4d6d",
    "teal": "#8fe8dc",
    "periwinkle": "#b0c6ff",
}

PROVIDER_COLORS = {
    "amp": ACCENT["blue"],
    "claude": ACCENT["orange"],
    "codex": ACCENT["yellow"],
    "copilot": ACCENT["purple"],
    "grok": ACCENT["pink"],
    "gemini": ACCENT["green"],
    "qwen": ACCENT["periwinkle"],
    "opencode": ACCENT["teal"],
}


# -- Detection helpers (shared with v1 behavior) -----------------------------
def candidate_detected(info: dict) -> bool:
    for path in info.get("paths", []):
        try:
            if Path(path).exists():
                return True
        except OSError:
            continue
    return any(shutil.which(command) for command in info.get("commands", []))


def provider_detected(key: str) -> bool:
    info = PROVIDER_OPTIONS.get(key)
    return bool(info and candidate_detected(info))


def discover_other_ai_tools() -> list[dict]:
    found = []
    for key, info in OTHER_AI_TOOLS.items():
        if candidate_detected(info):
            found.append({"key": key, "label": info["label"]})
    return found


def provider_label(key: str) -> str:
    return PROVIDER_OPTIONS.get(key, {}).get("label", key.title())


def provider_key_for_label(label: str) -> str:
    for key, info in PROVIDER_OPTIONS.items():
        if info.get("label") == label:
            return key
    return ""


def provider_color(key: str) -> str:
    return PROVIDER_COLORS.get(key, APP_PALETTE["text"])
