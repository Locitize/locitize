"""Launch external coding harnesses (Claude Code, Codex, OpenCode) against a
running LOCITIZE model (owner request 2026-08-21: "Launch in Claude Code /
Codex / OpenCode" from the Chat picker, alongside the existing Open WebUI
option).

Split from gui_controller.py the same way webui.py and config.py are split
out: everything here is pure or narrowly side-effecting (file writes, one
subprocess spawn, one shutil.which probe) and independently testable without
a GUI, a running model, or the harness CLIs actually being installed.

None of this touches a model's weights or LOCITIZE's own service lifecycle --
it only points a THIRD-PARTY CLI at an OpenAI/Anthropic-compatible endpoint
the running llama.cpp server already exposes natively (chat completions and
responses for Codex/OpenCode; the Anthropic Messages API for Claude Code --
no translation proxy needed, see the Claude Code section below). Every
base_url used here is loopback-only, matching the rest of the platform's
Permission Matrix section 7 discipline.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import winreg
except ImportError:  # non-Windows (test collection on CI, etc.)
    winreg = None  # type: ignore[assignment]

# The four surfaces the Chat picker offers. "openwebui"/"llamacpp" are the
# pre-existing built-in choices (webui.py); these three are the new harnesses.
HARNESS_CLAUDE = "claude"
HARNESS_CODEX = "codex"
HARNESS_OPENCODE = "opencode"
HARNESSES = (HARNESS_CLAUDE, HARNESS_CODEX, HARNESS_OPENCODE)

_EXECUTABLE_NAMES = {
    HARNESS_CLAUDE: "claude",
    HARNESS_CODEX: "codex",
    HARNESS_OPENCODE: "opencode",
}

_CODEX_PROVIDER_NAME = "locitize"
_OPENCODE_PROVIDER_NAME = "locitize"
_CODEX_CONFIG_RELATIVE = Path(".codex") / "config.toml"


def detect_executable(harness: str) -> str | None:
    """Return the resolved path to `harness`'s CLI, or None if not on PATH."""
    name = _EXECUTABLE_NAMES.get(harness)
    if name is None:
        raise ValueError(f"unknown harness {harness!r}; expected one of {HARNESSES}")
    return shutil.which(name)


def detect_harnesses() -> dict[str, str | None]:
    """Resolve all three harness executables in one pass, for the picker UI."""
    return {name: detect_executable(name) for name in HARNESSES}


# --------------------------------------------------------------------------- #
# Codex: ~/.codex/config.toml [model_providers.locitize]
# --------------------------------------------------------------------------- #


def codex_config_path(home: Path | None = None) -> Path:
    """Resolve ~/.codex/config.toml (or under an injected home, for tests)."""
    base = home if home is not None else Path.home()
    return base / _CODEX_CONFIG_RELATIVE


def build_codex_provider_block(base_url: str) -> str:
    """The [model_providers.locitize] TOML block Codex reads at startup.

    wire_api = "responses" (found empirically 2026-08-21: installed codex-cli
    0.148.0 rejects "chat" outright -- "wire_api = 'chat' is no longer
    supported", https://github.com/openai/codex/discussions/7782 -- current
    Codex only speaks OpenAI's Responses API). llama-server (build 10037+)
    natively exposes POST /v1/responses alongside /v1/chat/completions, so
    "responses" targets that route directly; confirmed with a real curl
    round-trip against a running local model before wiring this in. env_key
    names an environment variable Codex will read for the bearer token. The
    local server does not check it, but Codex requires SOME non-empty value
    be resolvable, so LOCITIZE_CODEX_API_KEY is set to a fixed dummy by the
    launch command line (see codex_launch_env below) rather than left unset.
    """
    return (
        f"\n[model_providers.{_CODEX_PROVIDER_NAME}]\n"
        f'name = "LOCITIZE (local)"\n'
        f'base_url = "{base_url}"\n'
        f'env_key = "LOCITIZE_CODEX_API_KEY"\n'
        f'wire_api = "responses"\n'
    )


def _normalized_header(line: str) -> str:
    """Collapse a TOML table-header line to a whitespace-insensitive form.

    Security review 2026-08-21 (CONFIRMED MEDIUM, fixed here): the previous
    exact-string match (`line.strip() == "[model_providers.locitize]"`)
    missed a legal-TOML variant like `[ model_providers.locitize ]`, so a
    second `[model_providers.locitize]` block got appended instead of the
    existing one being replaced -- reproduced live: `tomllib.loads` on the
    result raised "Cannot declare ('model_providers', 'locitize') twice",
    meaning Codex would fail to load ALL of the owner's providers and
    settings, not just this one. Stripping every space inside (not just
    around) the brackets makes both spellings compare equal.
    """
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return "[" + stripped[1:-1].replace(" ", "").replace("\t", "") + "]"
    return stripped


def upsert_codex_provider(base_url: str, config_path: Path | None = None) -> Path:
    """Write/replace ONLY the [model_providers.locitize] section of config.toml.

    A full TOML re-dump would risk reordering or losing the owner's existing
    Codex settings and other providers (the same reasoning config.py's
    targeted-edit writers use for settings.yaml/models.yaml). Instead this
    does a textual section replace: find the `[model_providers.locitize]`
    header if present and replace through the next top-level `[` header (or
    end of file); otherwise append the block. Every other byte of the file
    is untouched. Creates the file (and ~/.codex/) if it does not exist yet.

    Security review 2026-08-21 (CONFIRMED MEDIUM, fixed here): this is a
    file LOCITIZE did not create, holding the owner's real Codex settings
    and every other configured provider -- config.py's own documented rule
    for exactly this situation (see its module docstring) is "re-parse the
    rewritten text before committing, so a malformed edit leaves the
    original file untouched," via an atomic temp-file-plus-replace write.
    Neither guard was present here before this fix; both are now, reusing
    config.py's own `_atomic_write` rather than a second implementation.
    """
    import tomllib

    from config import _atomic_write

    path = config_path if config_path is not None else codex_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    block = build_codex_provider_block(base_url)

    header = _normalized_header(f"[model_providers.{_CODEX_PROVIDER_NAME}]")
    lines = existing.splitlines(keepends=True)
    start_idx = None
    for idx, line in enumerate(lines):
        if _normalized_header(line) == header:
            start_idx = idx
            break

    if start_idx is None:
        new_text = existing
        if new_text and not new_text.endswith("\n"):
            new_text += "\n"
        new_text += block
    else:
        end_idx = len(lines)
        for idx in range(start_idx + 1, len(lines)):
            candidate = lines[idx].lstrip()
            if candidate.startswith("[") and _normalized_header(lines[idx]) != header:
                end_idx = idx
                break
        new_text = "".join(lines[:start_idx]) + block.lstrip("\n") + "\n" + "".join(
            lines[end_idx:]
        )

    try:
        tomllib.loads(new_text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"refusing to write {path}: the result would not be valid TOML ({exc})"
        ) from exc

    _atomic_write(path, new_text)
    return path


def codex_launch_argv(project_dir: str, model_id: str) -> list[str]:
    """codex CLI invocation: cd into project_dir, use the locitize provider."""
    return [
        "codex",
        "-C",
        project_dir,
        "-c",
        f"model_provider={_CODEX_PROVIDER_NAME}",
        "-c",
        f'model="{model_id}"',
    ]


def codex_launch_env() -> dict[str, str]:
    """The dummy bearer token Codex's env_key setting expects to find set."""
    return {"LOCITIZE_CODEX_API_KEY": "locitize-local"}


def prime_codex_update(timeout_s: float = 60.0) -> None:
    """Run codex's own self-update out-of-band before opening the visible
    terminal (owner-observed defect 2026-08-21): codex's first invocation
    after a new release silently self-updates, prints "Please restart
    Codex.", and drops back to a bare shell prompt instead of starting the
    interactive session -- so the owner's visible terminal opened correctly
    but never actually launched a usable Codex session. codex has no
    documented flag to disable this (openai/codex#3855, #4375, both open,
    unresolved as of this writing), so instead of fighting it, this runs the
    same self-update check hidden (CREATE_NO_WINDOW) immediately before
    spawning the real terminal, so that visible invocation is (best-effort)
    already current and proceeds straight to the interactive session.
    Best-effort only: swallows every failure (missing binary, timeout, no
    network) because a failed priming call must never block the real launch
    that follows it -- worst case codex updates itself in the visible
    terminal exactly as it did before this fix existed.
    """
    if shutil.which("codex") is None:
        return
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            timeout=timeout_s,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


# --------------------------------------------------------------------------- #
# OpenCode: project-local opencode.json ["provider"]["locitize"]
# --------------------------------------------------------------------------- #

def _strip_jsonc_comments(text: str) -> str:
    """Strip // and /* */ comments so json.loads can parse a .jsonc file.

    A regex alone is unsafe here: opencode's own "$schema":
    "https://opencode.ai/config.json" contains a bare "//" inside a string
    literal, which a naive `//.*` pattern would truncate mid-value (this was
    caught by test_write_opencode_project_config_merges_existing_providers).
    This is instead a small character-by-character scanner that tracks
    whether it is inside a JSON string (respecting \\" escapes) and only
    treats // or /* as a comment start OUTSIDE a string.
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            end = text.find("\n", i)
            i = end if end != -1 else n
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = end + 2 if end != -1 else n
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def build_opencode_provider_config(base_url: str, model_id: str) -> dict[str, Any]:
    """The {"provider": {"locitize": {...}}} fragment opencode's config schema
    expects for a custom OpenAI-compatible endpoint (opencode.ai/docs/providers:
    npm '@ai-sdk/openai-compatible' targets /v1/chat/completions).

    A custom provider's endpoint is never auto-probed for its model list --
    opencode requires each servable model declared explicitly under `models`,
    keyed by the id `-m locitize/<model_id>` will request (found empirically
    2026-08-21: omitting this produced a real ProviderModelNotFoundError at
    prompt time even though the provider/endpoint itself was reachable).
    """
    return {
        "provider": {
            _OPENCODE_PROVIDER_NAME: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "LOCITIZE (local)",
                "options": {"baseURL": base_url},
                "models": {model_id: {"name": model_id}},
            }
        }
    }


def opencode_config_path(project_dir: str) -> Path:
    """Prefer an existing opencode.json/opencode.jsonc in the project; default
    to opencode.json (no comments needed) for a fresh one."""
    root = Path(project_dir)
    for name in ("opencode.json", "opencode.jsonc"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / "opencode.json"


def write_opencode_project_config(project_dir: str, base_url: str, model_id: str) -> Path:
    """Merge the locitize provider into the project's opencode config, leaving
    every other key (other providers, $schema, agent settings, ...) intact.

    Merges into any `models` map already present under the locitize provider
    (e.g. from a previous launch against a different model) rather than
    replacing it, so switching models across launches accumulates a usable
    model list instead of only ever remembering the most recent one.
    """
    path = opencode_config_path(project_dir)
    data: dict[str, Any] = {"$schema": "https://opencode.ai/config.json"}
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        stripped = _strip_jsonc_comments(raw)
        # Security review 2026-08-21 (LOW, fixed here): this used to
        # json.dumps the parsed result straight back over the original file
        # -- every real // or /* */ comment in an existing opencode.jsonc
        # silently vanished, an unannounced destructive edit to a file
        # that's usually committed to the owner's own repo. `stripped !=
        # raw` is true only when actual comment text (not just whitespace)
        # was removed, so a plain opencode.json (no comments possible) is
        # never affected. Refusing here, loudly, beats guessing at a
        # comment-preserving merge for a LOW-severity, non-security issue.
        if path.suffix == ".jsonc" and stripped != raw:
            raise ValueError(
                f"{path} has comments LOCITIZE cannot safely preserve while merging - "
                "add the locitize provider to it by hand, or delete opencode.jsonc "
                "and relaunch so LOCITIZE writes a fresh opencode.json instead"
            )
        parsed = json.loads(stripped) if raw.strip() else {}
        if isinstance(parsed, dict):
            data = parsed

    provider = data.get("provider")
    if not isinstance(provider, dict):
        provider = {}
    new_entry = build_opencode_provider_config(base_url, model_id)["provider"][
        _OPENCODE_PROVIDER_NAME
    ]
    existing_entry = provider.get(_OPENCODE_PROVIDER_NAME)
    if isinstance(existing_entry, dict):
        existing_models = existing_entry.get("models")
        if isinstance(existing_models, dict):
            new_entry["models"] = {**existing_models, **new_entry["models"]}
    provider[_OPENCODE_PROVIDER_NAME] = new_entry
    data["provider"] = provider

    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def opencode_launch_argv(project_dir: str, model_id: str) -> list[str]:
    """opencode CLI invocation: positional project dir, -m provider/model."""
    return ["opencode", project_dir, "-m", f"{_OPENCODE_PROVIDER_NAME}/{model_id}"]


# --------------------------------------------------------------------------- #
# Claude Code: env-pointed straight at the running llama-server's native
# Anthropic Messages API (llama.cpp PR #17570, merged 2025-11-28: llama-server
# now serves POST /v1/messages itself). Confirmed empirically against a local
# llama.cpp install new enough to include the PR: both the plain and
# streaming forms returned correctly shaped Anthropic responses with no auth
# header required. No translation proxy needed -- the earlier plan to write
# one (claude_bridge.py) turned out to be solving an already-solved problem;
# it added a whole extra HTTP-server subsystem, a second port, and a
# tool-use-translation gap for zero benefit once llama-server's native
# support was found and verified.
#
# This DOES depend on the owner's llama.cpp being new enough. A build older
# than the PR's merge would 404 on /v1/messages; LOCITIZE has no reliable way
# to introspect a third-party binary's exact build date, so this is not
# probed before offering "Claude Code" in the picker. If claude CLI reports a
# connection/route error, the fix is upgrading llama.cpp, not LOCITIZE code.
# --------------------------------------------------------------------------- #


# Owner-observed defect 2026-08-21: Claude Code's full tool surface (its own
# built-ins plus every configured MCP server) produces a JSON-schema/tool
# definition too large or too irregular for llama-server's grammar compiler
# -- confirmed against the owner's real config: "API Error: 400 ... Failed to
# initialize samplers: failed to parse grammar" and, separately, "Pattern
# must start with '^' and end with '$'" (an MCP tool's own unanchored regex,
# which upstream llama.cpp confirms it cannot relax -- github.com/ggml-org/
# llama.cpp maintainers state there is no server-side flag to disable this).
# Neither is fixable in LOCITIZE's own code. What IS fixable: a local coding
# session never needed the owner's MCP servers (trading, BI, browser
# control, ...) in the first place, and Claude Code exposes exactly the
# scoping needed to drop them -- confirmed live: --strict-mcp-config (loads
# zero MCP servers) plus a --tools allowlist limited to the handful of
# built-ins a coding session actually uses got a real "PONG" back from the
# owner's own local model, exit 0. Kept as a named constant, not inlined,
# so if a future coding workflow genuinely needs one more built-in tool
# (e.g. TodoWrite), it is a one-line change here rather than a hunt through
# argv-building code.
CLAUDE_LOCAL_SESSION_TOOLS = "Bash,Edit,Read,Write,Glob,Grep"


def claude_launch_argv(model_id: str) -> list[str]:
    """claude CLI invocation: ANTHROPIC_BASE_URL does the routing, --model
    carries the running LOCITIZE model's real id, --strict-mcp-config and
    --tools keep the request's tool schema inside llama-server's grammar
    engine limits (see CLAUDE_LOCAL_SESSION_TOOLS above for why).

    Owner-observed defect 2026-08-21: with no --model, claude's status line
    displays its settings.json default ("sonnet") regardless of which
    backend actually serves the session -- correct routing looked
    indistinguishable from a subscription fallback. Passing --model here
    fixes both the cosmetic label and sends the honest model name in the
    request body (llama-server does not validate it, but "sonnet" was
    always wrong to send).
    """
    return [
        "claude",
        "--model",
        model_id,
        "--strict-mcp-config",
        "--tools",
        CLAUDE_LOCAL_SESSION_TOOLS,
    ]


def claude_launch_env(model_base_url: str, context_size: int | None = None) -> dict[str, str]:
    """ANTHROPIC_BASE_URL is the documented override the claude CLI (and the
    anthropic-sdk it embeds) reads before falling back to api.anthropic.com.
    Points straight at the running model's llama-server (e.g.
    "http://127.0.0.1:8080"), the SAME server Codex/OpenCode already target
    under /v1/chat/completions -- llama-server now answers both APIs itself.
    ANTHROPIC_API_KEY must be non-empty for the CLI to skip its login flow;
    llama-server does not check its value.

    Owner-observed defect 2026-08-21: an unrecognized --model makes claude
    assume a 200k-token window for its own auto-compact bookkeeping,
    regardless of the running model's REAL context_size (models.yaml entries
    range from 8192 to 32768 in this registry) -- claude's own warning names
    the exact fix, CLAUDE_CODE_MAX_CONTEXT_TOKENS, so this sets it from the
    model's actual registered context_size whenever the caller has it,
    letting claude compact conversation history before it ever sends more
    than the local model can actually accept.
    """
    env = {
        "ANTHROPIC_BASE_URL": model_base_url,
        "ANTHROPIC_API_KEY": "locitize-local",
    }
    if context_size:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(context_size)
    return env


# --------------------------------------------------------------------------- #
# Spawning a visible, interactive terminal (the opposite of the CREATE_NO_WINDOW
# console-flash fixes elsewhere in this codebase: these three ARE meant to be
# seen and typed into).
# --------------------------------------------------------------------------- #


def build_launch_batch_text(
    argv: list[str], cwd: str, env_overrides: dict[str, str]
) -> str:
    """The literal .bat file content that runs `argv` in `cwd` with
    `env_overrides` set for that window only.

    Owner-observed defect 2026-08-21: the previous design packed this same
    command line into ONE element of the argv list handed to
    `subprocess.Popen(["cmd", "/c", "start", title, "cmd", "/k", inner], ...)`.
    Windows Popen re-quotes list elements with `list2cmdline` (C-runtime
    argv-escaping: an embedded `"` becomes `\\"`), but `cmd.exe`'s own
    parser does not understand backslash-escaped quotes -- so the moment an
    argv element contained a literal `"` (Codex's own `-c model="<id>"` TOML
    override does, always), `cmd /k "...\\"..."` silently mis-parsed and
    dropped to a bare prompt instead of running the command. A codex
    terminal opening to an empty prompt (never a crash, never an error) was
    the visible symptom.

    Writing the command as a real .bat file sidesteps the problem entirely:
    the file's bytes are never re-escaped by anything, they are read by
    cmd.exe exactly as written, using ordinary batch-file quoting (the same
    rules a person would use typing this by hand). Pure and unit-testable.

    Security review 2026-08-21 (CONFIRMED HIGH, fixed here): the previous
    `_quote_arg` only quoted a value containing a space or tab, so a
    space-free project folder name containing `&`, `|`, or `^` -- all legal
    in a Windows directory name -- reached the `cd /d` and argv lines
    UNQUOTED, and cmd.exe treats an unquoted `&` as an unconditional command
    separator. Reproduced live: a folder named `proj&calc` made the
    generated .bat run an attacker-chosen second command after `cd`-ing into
    it, in the exact console the owner believes is their coding harness
    (this is precisely the "point a harness at a downloaded project"
    workflow this feature exists for). Every interpolated value (cwd AND
    each argv element) is now unconditionally quoted via `_quote_arg`,
    which also escapes `%` (bare `%` still expands inside a quoted batch
    argument -- `%TEMP%` in a folder name would otherwise substitute a
    different path) and rejects CR/LF (unrepresentable on one batch line;
    failing loudly here beats silently truncating or corrupting the launch).
    """
    lines = ["@echo off", f"cd /d {_quote_arg(cwd)}"]
    lines.extend(f'set "{key}={_escape_percent(value)}"' for key, value in env_overrides.items())
    lines.append(" ".join(_quote_arg(a) for a in argv))
    return "\r\n".join(lines) + "\r\n"


def _escape_percent(value: str) -> str:
    """Defuse `%VAR%` expansion inside a `set "KEY=value"` line.

    `set "KEY=VALUE"` (the whole assignment quoted, NOT `set KEY="VALUE"`,
    which sets the variable's value to literally include the quote
    characters -- confirmed empirically) does not need `"` doubled the way
    an argv element does, since there is exactly one quoted span and no
    child-process argv parser re-reads it. `%` still expands here though.
    """
    if "\r" in value or "\n" in value:
        raise ValueError("harness launch env value cannot contain a newline")
    return value.replace("%", "%%")


def _quote_arg(arg: str) -> str:
    """Unconditionally cmd.exe/batch-quote one value.

    `"` -> `""` (cmd's own doubled-quote escape for a literal quote inside a
    quoted argument -- confirmed empirically against a real argv-parsing
    child process, not assumed) and `%` -> `%%` (defuses variable expansion
    inside the quotes). CR/LF cannot be represented on a single batch line
    at all, so those raise rather than silently mis-launching.
    """
    if "\r" in arg or "\n" in arg:
        raise ValueError("harness launch argument cannot contain a newline")
    return '"' + arg.replace('"', '""').replace("%", "%%") + '"'


_LAUNCH_BATCH_MAX_AGE_S = 3600  # an hour is generously longer than any real coding session's startup


def cleanup_stale_launch_batch_files(max_age_s: float = _LAUNCH_BATCH_MAX_AGE_S) -> int:
    """Delete `locitize_harness_*.bat` temp files older than `max_age_s`.

    Security review 2026-08-21 (LOW, fixed here): `write_launch_batch_file`
    never deleted the file it wrote -- each harness launch left one more
    .bat behind in %TEMP% forever. Not a secrets-disclosure issue (the only
    content is a loopback URL, a dummy placeholder token, and an integer,
    all meaningless outside a llama-server that never validates them), but
    unbounded per-user litter is still worth sweeping. Called opportunistically
    from write_launch_batch_file itself (every real launch is a natural,
    already-happening moment to do this) rather than needing a separate
    startup hook. Best-effort: a file that is still open (mid-launch on
    another thread, or the owner has it open in an editor) is skipped, not
    an error - deletion failures here must never block the real launch.
    """
    deleted = 0
    import time

    cutoff = time.time() - max_age_s
    temp_dir = Path(tempfile.gettempdir())
    for candidate in temp_dir.glob("locitize_harness_*.bat"):
        try:
            if candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                deleted += 1
        except OSError:
            continue
    return deleted


def write_launch_batch_file(
    argv: list[str], cwd: str, env_overrides: dict[str, str]
) -> Path:
    """Write `build_launch_batch_text`'s output to a fresh temp .bat file and
    return its path. A new file per launch (never reused/overwritten) so two
    harness launches in flight at once can never race on the same file.

    Sweeps old batch files from prior launches first (see
    cleanup_stale_launch_batch_files) -- best-effort, never blocks this
    launch if the sweep itself fails.
    """
    try:
        cleanup_stale_launch_batch_files()
    except OSError:
        pass
    fd, raw_path = tempfile.mkstemp(prefix="locitize_harness_", suffix=".bat")
    path = Path(raw_path)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
        handle.write(build_launch_batch_text(argv, cwd, env_overrides))
    return path


def spawn_in_terminal(
    argv: list[str], cwd: str, env_overrides: dict[str, str], title: str
) -> subprocess.Popen:
    """Launch argv in a new, visible, interactive console window rooted at cwd.

    CREATE_NEW_CONSOLE (not CREATE_NO_WINDOW -- these three launches are the
    one place in the app that WANTS a console the owner can see and type
    into). Windows-only, matching every other harness/service launcher in
    this platform. Routes through a temp .bat file (see
    `build_launch_batch_text`) rather than an inline command string, so
    argv elements containing literal quotes (Codex's `-c model="<id>"`)
    cannot be mangled by nested cmd.exe/Popen re-quoting.
    """
    bat_path = write_launch_batch_file(argv, cwd, env_overrides)
    command = ["cmd", "/c", "start", title, "cmd", "/k", str(bat_path)]
    creationflags = (
        subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    )
    return subprocess.Popen(command, cwd=cwd, creationflags=creationflags)


# --------------------------------------------------------------------------- #
# Zero-friction install (owner request 2026-08-21: "the user should not have
# to do anything" -- if a harness isn't found, LOCITIZE offers to install and
# configure it itself, not just grey out the picker row). Every command below
# is the real, current (2026) vendor-official installer for that CLI -- never
# an unofficial mirror or a curl-pipe-to-bash of unknown provenance.
# --------------------------------------------------------------------------- #

_NODE_WINGET_ID = "OpenJS.NodeJS.LTS"
_INSTALL_TIMEOUT_S = 300


def _read_registry_path(hive: int, subkey: str) -> str:
    """Best-effort read of one registry Path value; "" on any failure."""
    if winreg is None:
        return ""
    try:
        with winreg.OpenKey(hive, subkey) as key:
            value, _ = winreg.QueryValueEx(key, "Path")
            return value or ""
    except OSError:
        return ""


def refreshed_search_path() -> str:
    """PATH merged with the live HKCU/HKLM registry Path values.

    A native installer (Claude Code, Codex) or `npm install -g` updates the
    registry immediately, but this already-running process's own os.environ
    PATH snapshot was taken at process start -- shutil.which(name) alone
    would report "not found" for a harness that was just installed a moment
    ago, until LOCITIZE itself is restarted. Re-reading the registry lets a
    same-session re-detect succeed instead. Windows-only; returns the plain
    process PATH unchanged elsewhere (matching every other Windows-only
    codepath in this module, e.g. spawn_in_terminal's CREATE_NEW_CONSOLE).
    """
    if sys.platform != "win32":
        return os.environ.get("PATH", "")
    parts = [os.environ.get("PATH", "")]
    parts.append(_read_registry_path(winreg.HKEY_CURRENT_USER, "Environment"))
    parts.append(
        _read_registry_path(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        )
    )
    return os.pathsep.join(p for p in parts if p)


def detect_executable_fresh(harness: str) -> str | None:
    """Like detect_executable, but searches refreshed_search_path() too.

    Used only right after an install completes, so a just-installed harness
    is found in the same LOCITIZE session instead of requiring a restart.
    detect_executable() itself is left untouched (still a plain
    shutil.which(name) call) so the picker's ordinary detection stays cheap
    and its existing tests keep asserting a single-argument shutil.which.
    """
    name = _EXECUTABLE_NAMES.get(harness)
    if name is None:
        raise ValueError(f"unknown harness {harness!r}; expected one of {HARNESSES}")
    return shutil.which(name, path=refreshed_search_path())


def _run_powershell(script: str, timeout: int = _INSTALL_TIMEOUT_S) -> tuple[bool, str]:
    """Run one PowerShell command with no visible window.

    The Claude Code and Codex native installers are silent, non-interactive
    single-binary drops (irm | iex) -- CREATE_NO_WINDOW here is the same
    "the owner didn't ask to watch a console" reasoning as every other
    background subprocess in this platform (see health.py, services.py).
    """
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"failed to run: {exc}"
    if proc.returncode != 0:
        tail = (proc.stdout or proc.stderr or "").strip()[-300:]
        return False, f"exited {proc.returncode}: {tail}"
    return True, "ok"


def install_claude_code() -> tuple[bool, str]:
    """Official native installer, no Node.js required.

    https://claude.ai/install.ps1 (verified current 2026 install path). Drops
    a single binary and updates the user's PATH itself -- LOCITIZE writes
    nothing extra; Claude Code needs no config file, only the
    ANTHROPIC_BASE_URL set at launch time (see claude_launch_env above).
    """
    ok, detail = _run_powershell("irm https://claude.ai/install.ps1 | iex")
    if not ok:
        return False, f"Claude Code install failed: {detail}"
    if detect_executable_fresh(HARNESS_CLAUDE) is None:
        return False, (
            "the installer ran but claude was not found afterward - "
            "restart LOCITIZE and it should be picked up"
        )
    return True, "Claude Code installed"


def install_codex() -> tuple[bool, str]:
    """Official native installer, no Node.js required.

    https://chatgpt.com/codex/install.ps1 (verified current 2026 install
    path). Codex's own [model_providers.locitize] config.toml section is
    written fresh on every real launch (see _do_launch_harness in
    gui_controller.py), so nothing further needs writing here.
    """
    ok, detail = _run_powershell("irm https://chatgpt.com/codex/install.ps1 | iex")
    if not ok:
        return False, f"Codex install failed: {detail}"
    if detect_executable_fresh(HARNESS_CODEX) is None:
        return False, (
            "the installer ran but codex was not found afterward - "
            "restart LOCITIZE and it should be picked up"
        )
    return True, "Codex installed"


def _node_available() -> bool:
    path = refreshed_search_path()
    return (
        shutil.which("node", path=path) is not None
        and shutil.which("npm", path=path) is not None
    )


def _install_node() -> tuple[bool, str]:
    """Winget install of the Node.js LTS.

    Security review 2026-08-21 (correction, not a vulnerability): this
    docstring and the earlier success message both used to claim "no
    elevation" by analogy with secure_proxy.py's _install_caddy(), but that
    was never actually verified for THIS package. Checked directly:
    `winget show --id OpenJS.NodeJS.LTS` declares no Scope override and an
    "Installer Type: wix" (MSI) -- winget's default for an MSI with no
    declared user-scope support is a machine-wide install, which prompts a
    real Windows UAC consent dialog. `--scope user` is NOT passed here
    because passing it against a package that does not support user scope
    would make winget reject the install outright rather than silently
    downgrade it. The caller (the onboarding dialog in desktop.py) says so
    up front so a UAC prompt is expected, not a surprise; this function
    itself just runs the install and reports the real outcome, with
    CREATE_NO_WINDOW hiding only the console flash, never the UAC dialog
    itself (Windows renders that on the secure desktop regardless).
    """
    winget = Path.home() / "AppData/Local/Microsoft/WindowsApps/winget.exe"
    exe = str(winget) if winget.exists() else "winget"
    try:
        proc = subprocess.run(
            [
                exe,
                "install",
                "--id",
                _NODE_WINGET_ID,
                "--accept-source-agreements",
                "--accept-package-agreements",
            ],
            capture_output=True,
            text=True,
            timeout=_INSTALL_TIMEOUT_S,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Node.js install failed to run: {exc}"
    if proc.returncode != 0:
        tail = (proc.stdout or proc.stderr or "").strip()[-300:]
        return False, f"Node.js install exited {proc.returncode}: {tail}"
    return True, "Node.js installed"


def install_opencode() -> tuple[bool, str]:
    """OpenCode has no native Windows installer; it ships as an npm package
    (opencode-ai). Installs Node.js LTS via winget first when neither node
    nor npm is already on PATH, then `npm i -g opencode-ai`. OpenCode's own
    per-project opencode.json is written fresh on every real launch (see
    _do_launch_harness), so nothing further needs writing here.
    """
    if not _node_available():
        ok, detail = _install_node()
        if not ok:
            return False, f"OpenCode needs Node.js and it could not be installed: {detail}"
    npm_path = shutil.which("npm", path=refreshed_search_path())
    if npm_path is None:
        return False, (
            "Node.js was installed but npm could not be found yet - "
            "restart LOCITIZE and try again"
        )
    try:
        proc = subprocess.run(
            [npm_path, "i", "-g", "opencode-ai"],
            capture_output=True,
            text=True,
            timeout=_INSTALL_TIMEOUT_S,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"OpenCode install failed to run: {exc}"
    if proc.returncode != 0:
        tail = (proc.stdout or proc.stderr or "").strip()[-300:]
        return False, f"OpenCode install exited {proc.returncode}: {tail}"
    if detect_executable_fresh(HARNESS_OPENCODE) is None:
        return False, (
            "the installer ran but opencode was not found afterward - "
            "restart LOCITIZE and it should be picked up"
        )
    return True, "OpenCode installed"


INSTALL_FUNCTIONS = {
    HARNESS_CLAUDE: install_claude_code,
    HARNESS_CODEX: install_codex,
    HARNESS_OPENCODE: install_opencode,
}


def install_harness(harness: str) -> tuple[bool, str]:
    """Dispatch to the right installer by harness key (picker-facing entry
    point, mirrors detect_executable's own name-validation discipline)."""
    fn = INSTALL_FUNCTIONS.get(harness)
    if fn is None:
        raise ValueError(f"unknown harness {harness!r}; expected one of {HARNESSES}")
    return fn()
