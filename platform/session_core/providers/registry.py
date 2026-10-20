"""Provider registry - the single place that lists resumable providers.

Adding a 5th provider means: write a ``Provider`` implementation and append it
to :data:`PROVIDERS`. Nothing else in the app changes.
"""
from __future__ import annotations

from .base import Provider
from .claude import ClaudeProvider
from .codex import CodexProvider
from .copilot import CopilotProvider
from .gemini import GeminiProvider
from .grok import GrokProvider
from .opencode import OpencodeProvider
from .qwen import QwenProvider

PROVIDERS: list[Provider] = [
    ClaudeProvider(),
    CodexProvider(),
    GrokProvider(),
    CopilotProvider(),
    GeminiProvider(),
    QwenProvider(),
    OpencodeProvider(),
]

_BY_KEY = {p.key: p for p in PROVIDERS}

# Providers whose sessions live in an external store (a server, another
# tool's database): Session Portal never deletes those - it hides the row
# locally instead.
HIDE_ONLY_KEYS: frozenset[str] = frozenset(
    p.key for p in PROVIDERS if getattr(p, "hide_only", False)
)


def get_provider(key: str) -> Provider | None:
    return _BY_KEY.get(key)


def is_hide_only(key: str) -> bool:
    return key in HIDE_ONLY_KEYS


def detected_provider_keys() -> list[str]:
    return [p.key for p in PROVIDERS if p.detected()]
