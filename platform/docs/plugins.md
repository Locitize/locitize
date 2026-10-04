# locitize Plugin Contract (frozen at M8)

locitize applications are plugins registered in a small in-process registry
(`plugins.py`, Architecture section 6 and M8.4). This document is the frozen
third-party contract: what a plugin module must expose so a future
directory-scan discovery mechanism can load it without any interface change.

The built-in applications (Model Manager, Documentation, Health Checks,
Settings, Voice Assistant, Benchmark Suite, Vision) all implement this same
contract; they are auto-registered at construction. No third-party plugin ships
this milestone -- the discovery scan is still future -- but the contract below is
final.

## What a plugin is

A plugin is a Python module exposing a concrete subclass of `plugins.Plugin`.

### Declared attributes

| Attribute | Type | Meaning |
|-----------|------|---------|
| `id` | str | Unique registry id (kebab-case, e.g. `my-app`). Two plugins may not share an id. |
| `name` | str | Human-readable name shown in the menu. |
| `version` | str | Plugin version string (e.g. `1.0.0`). |
| `kind` | str | One of `application`, `service`, `capability`. Only `application` plugins show in the launcher applications menu. |
| `available` | bool | `True` if the plugin can run now; `False` marks a declared-but-unbuilt surface. |
| `unavailable_reason` | str | Optional. Shown when an unavailable plugin is launched. |

### Required methods

| Method | Contract |
|--------|----------|
| `activate(context) -> None` | Prepare the plugin (called once before first launch). Available plugins are activated at registration. |
| `deactivate() -> None` | Release any resources the plugin acquired. |
| `launch(context) -> int` | Run the application; return a process/exit code (0 = ok). Must RETURN (not block forever) so the menu regains control. |

## The injected context

Every method receives a `plugins.PluginContext` by dependency injection -- there is
no global state. Its fields:

| Field | What it provides |
|-------|------------------|
| `config` | The loaded `Settings` (paths, ports, thresholds, feature config). |
| `logger_factory` | `get_logger(channel)` -> a logger writing to the platform log tree. |
| `service_manager` | The `ServiceManager` that owns managed child processes and their no-orphan cleanup. |
| `model_registry` | The `ModelRegistry` for model lookup and start-spec construction. |

## Rules a plugin MUST follow

1. **No global mutable state.** Use only the injected context; do not reach for
   module-level singletons.
2. **No direct process spawning.** Any managed child process goes through the
   injected `service_manager`, which owns lifecycle and guarantees no orphan is
   left on exit. Do not call `subprocess`/`Popen` yourself.
3. **Loopback-only networking.** Any HTTP a plugin makes stays on `127.0.0.1`
   (SEC-1). locitize exposes nothing beyond loopback.
4. **No secrets in code or config artifacts.** Secrets, if ever needed, come from
   environment variables, never committed files.
5. **`launch()` returns.** A long-running interactive loop must still return
   control to the menu on exit (EOF/quit/interrupt).

## Registration

Built-in plugins register automatically in `PluginRegistry._register_builtins`.
`PluginRegistry.register(plugin)` also accepts a dynamically discovered plugin, so
the future directory-scan of `platform/plugins_installed/*/plugin.py` (documented
as future in `docs/roadmap.md`) is purely additive -- it will call `register()` for
each discovered `Plugin` subclass. Launch dispatch is by id via
`PluginRegistry.launch(id)`: return code 0 = launched, 2 = unknown id, 3 =
unavailable (with a reason from `unavailable_message(id)`).
