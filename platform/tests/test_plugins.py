"""Plugin-registry tests (keyword 'plugins_registry', Architecture M8.4/M8.5, AC9).

Deterministic, no real service start: the registry auto-registers the built-in
applications at construction; these tests confirm the M8 matured surface -- Voice
Assistant, Benchmark Suite, and the new Vision plugin are all available=True, a
still-future plugin registered available=False stays unavailable, and launch()
dispatch routes correctly (0 launched, 2 unknown, 3 unavailable).
"""

from __future__ import annotations

from config import ModelRegistryData, Settings
from logger import get_logger
from models import ModelRegistry
from plugins import PluginContext, PluginRegistry, _BuiltinPlugin
from services import ServiceManager


def _registry() -> PluginRegistry:
    settings = Settings()
    context = PluginContext(
        config=settings,
        logger_factory=get_logger,
        service_manager=ServiceManager(),
        model_registry=ModelRegistry(ModelRegistryData(), settings),
    )
    return PluginRegistry(context)


def test_plugins_registry_registers_all_builtins():
    """All seven built-in applications auto-register at construction (M8: +Vision)."""
    apps = {p.id for p in _registry().applications()}
    assert apps == {
        "model-manager",
        "documentation",
        "health-checks",
        "settings",
        "voice-assistant",
        "benchmark-suite",
        "vision",
    }


def test_plugins_registry_live_surfaces_available():
    """Voice Assistant (M7), Benchmark Suite (M5), and Vision (M8) are available."""
    reg = _registry()
    assert reg.get("voice-assistant").available is True
    assert reg.get("benchmark-suite").available is True
    assert reg.get("vision").available is True
    assert reg.get("health-checks").available is True


def test_plugins_registry_vision_is_an_application():
    """The Vision plugin is an application (so it shows in the menu) and is real."""
    reg = _registry()
    vision = reg.get("vision")
    assert vision is not None
    assert vision.kind == "application"
    assert vision.name == "Vision"


def test_plugins_registry_launch_unavailable_returns_code_3():
    """Launching an unavailable (future) plugin returns code 3 and a reason.

    Every built-in is now live, so this exercises the degraded path with an ad-hoc
    still-future plugin registered available=False -- the same contract a real future
    plugin (e.g. a scheduling app) will use.
    """
    reg = _registry()
    reg.register(
        _BuiltinPlugin(
            "future-scheduler",
            "Scheduler",
            "0.0.0",
            False,
            unavailable_reason="scheduling is planned for a later milestone",
        )
    )
    assert reg.launch("future-scheduler") == 3
    msg = reg.unavailable_message("future-scheduler")
    assert "not available" in msg
    assert "later milestone" in msg


def test_plugins_registry_launch_unknown_returns_code_2():
    reg = _registry()
    assert reg.launch("does-not-exist") == 2


def test_plugins_registry_launch_available_returns_zero():
    reg = _registry()
    assert reg.launch("health-checks") == 0
    assert reg.launch("vision") == 0
