"""Plugin / application registry for the LOCITIZE platform (Architecture 6 + M8.4).

A working registry the launcher uses to list and launch applications. Built-in
applications register automatically at construction. An application not yet built
registers with available=False and, when launched, returns a clear "not available"
code + reason instead of crashing or faking behavior.

M8.4 matured (did not rebuild) this registry: the Plugin ABC and PluginRegistry API
are unchanged in SHAPE (the M1 promise), but the built-in surfaces are now real,
auto-registered, available plugins -- Voice Assistant (M7), Benchmark Suite (M5),
and the new Vision plugin (M8). The third-party contract below is now FROZEN and
documented (see also docs/plugins.md).

THIRD-PARTY PLUGIN CONTRACT (frozen at M8; the future directory scan drops in
without changing it):
- A plugin is a module exposing a concrete subclass of `Plugin`.
- It declares: `id` (unique str), `name` (str), `version` (str), `kind` (one of
  "application" | "service" | "capability"), `available` (bool), and an optional
  `unavailable_reason` (str, shown when an unavailable plugin is launched).
- It implements: `activate(context)` (prepare, called once before first launch),
  `deactivate()` (release resources), and `launch(context) -> int` (run; return a
  process/exit code). launch() must return, not block forever, so the menu regains
  control.
- It receives a `PluginContext` via dependency injection (config, logger_factory,
  service_manager, model_registry). It MUST NOT use global mutable state and MUST
  NOT spawn OS processes directly -- any managed child process goes through the
  injected `service_manager` (Permission Matrix M8), which owns lifecycle and the
  no-orphan cleanup. All network I/O stays loopback (SEC-1).
- Discovery: built-ins register automatically here. Directory-scan discovery of
  `platform/plugins_installed/*/plugin.py` remains the documented FUTURE mechanism
  (no external plugin ships this milestone); PluginRegistry.register() already
  accepts a dynamically-discovered plugin, so the scan is additive later.

No global mutable state: the registry is constructed with a PluginContext of
injected collaborators (config, logger factory, service manager, model registry).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Literal

PluginKind = Literal["application", "service", "capability"]


@dataclass
class PluginContext:
    """Injected collaborators a plugin may use. Dependency injection, no globals.

    Fields are typed as Any to avoid import cycles with the concrete platform
    modules; the launcher passes the real objects at runtime.
    """

    config: Any  # Settings
    logger_factory: Callable[[str], Any]  # get_logger
    service_manager: Any  # ServiceManager
    model_registry: Any  # ModelRegistry


class Plugin(ABC):
    """Abstract application/plugin interface (Architecture section 6)."""

    id: str
    name: str
    version: str
    kind: PluginKind
    available: bool
    # Milestone note shown when an unavailable plugin is launched.
    unavailable_reason: str = ""

    @abstractmethod
    def activate(self, context: PluginContext) -> None:
        """Prepare the plugin for use (called once before first launch)."""

    @abstractmethod
    def deactivate(self) -> None:
        """Release any resources the plugin acquired."""

    @abstractmethod
    def launch(self, context: PluginContext) -> int:
        """Run the application; return a process/exit code."""


class _BuiltinPlugin(Plugin):
    """Base for the built-in applications registered at construction time.

    Available built-ins implement launch(); the platform itself provides their
    behavior via the launcher (e.g. Health Checks re-runs the ladder). Their
    launch() here returns 0 as a no-op hook the launcher wires to real actions,
    keeping this module free of launcher imports (avoids a cycle).
    """

    def __init__(
        self,
        plugin_id: str,
        name: str,
        version: str,
        available: bool,
        kind: PluginKind = "application",
        unavailable_reason: str = "",
    ) -> None:
        self.id = plugin_id
        self.name = name
        self.version = version
        self.kind = kind
        self.available = available
        self.unavailable_reason = unavailable_reason

    def activate(self, context: PluginContext) -> None:
        # Built-ins need no activation resources in M1.
        return None

    def deactivate(self) -> None:
        return None

    def launch(self, context: PluginContext) -> int:
        # Available built-ins are dispatched by the launcher, not here; this hook
        # returns success. Unavailable ones never reach here (guarded by launch()).
        return 0


class PluginRegistry:
    """Registers built-in applications and mediates launching them."""

    def __init__(self, context: PluginContext) -> None:
        self._context = context
        self._plugins: dict[str, Plugin] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register the built-in applications automatically (spec requirement).

        Model Manager, Documentation, Health Checks, Settings are platform-provided
        (available=True). As of M8 every built-in application is live: Voice
        Assistant (M7), Benchmark Suite (M5), and the new Vision plugin (M8) are all
        available=True and auto-registered here with no manual wiring.
        """
        builtins = [
            _BuiltinPlugin("model-manager", "Model Manager", "1.0.0", True),
            _BuiltinPlugin("documentation", "Documentation", "1.0.0", True),
            _BuiltinPlugin("health-checks", "Health Checks", "1.0.0", True),
            _BuiltinPlugin("settings", "Settings", "1.0.0", True),
            _BuiltinPlugin(
                "voice-assistant",
                "Voice Assistant",
                "1.0.0",
                # M7: the built-in assistant loop is real (mic/text -> LLM -> Kokoro,
                # streaming + speak-per-sentence + interrupt). The launcher menu
                # special-cases its selection to run AssistantLoop, and the CLI drives
                # it via `launcher.py --assistant`. Its label is finally true.
                True,
            ),
            _BuiltinPlugin(
                "benchmark-suite",
                "Benchmark Suite",
                "1.0.0",
                # M5: the benchmark engine is real. Available now; the launcher menu
                # special-cases its selection to run the suite over the launchable
                # models (Architecture M5.11), and the CLI drives it via
                # `launcher.py --benchmark`. The GUI shows scores but does not kick
                # off runs (M5.11, a long blocking multi-model run does not belong on
                # the interactive controller).
                True,
            ),
            _BuiltinPlugin(
                "vision",
                "Vision",
                "1.0.0",
                # M8: single-image Q&A is real. Available now; the launcher menu
                # special-cases its selection to prompt for an image path and run the
                # describe path (qwen2-5-vl + --mmproj via the ModelController), and
                # the CLI drives it via `launcher.py --describe <image>`.
                True,
            ),
        ]
        for plugin in builtins:
            self.register(plugin)

    def register(self, plugin: Plugin) -> None:
        """Register a plugin (built-in now, discovered externally in M4)."""
        self._plugins[plugin.id] = plugin
        # Activate available plugins immediately so they are ready to launch.
        if plugin.available:
            plugin.activate(self._context)

    def applications(self) -> list[Plugin]:
        """All registered plugins of kind 'application', for the menu."""
        return [p for p in self._plugins.values() if p.kind == "application"]

    def get(self, app_id: str) -> Plugin | None:
        return self._plugins.get(app_id)

    def launch(self, app_id: str) -> int:
        """Launch an application by id.

        Unknown id or an unavailable plugin returns a non-crash code and a clear
        message via the return value semantics (the launcher prints the reason).
        Return codes: 0 launched ok, 2 unknown, 3 unavailable.
        """
        plugin = self._plugins.get(app_id)
        if plugin is None:
            return 2
        if not plugin.available:
            # Degraded mode: never call into an unbuilt application.
            return 3
        return plugin.launch(self._context)

    def unavailable_message(self, app_id: str) -> str:
        """Human-readable reason an application cannot be launched yet."""
        plugin = self._plugins.get(app_id)
        if plugin is None:
            return f"no application registered under '{app_id}'"
        if not plugin.available:
            reason = plugin.unavailable_reason or "not yet implemented"
            return f"{plugin.name} is not available: {reason}"
        return ""
