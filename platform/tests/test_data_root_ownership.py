"""AC-M14-27: the directory ownership contract and the W1 write fence.

DEC-M14-9 / Architecture M14.2.3.1. Three layers, because the shipped defect
(NEW-QA-M14-8: transcripts, logs and the Open WebUI store written into the
INSTALL directory) would have slipped past any one of them alone:

1. RESOLVERS - given a Settings whose data_dir differs from base_dir, every
   resolver named in the contract returns a path under data_dir. This is the
   layer the defect broke.
2. CALL SITES - the three resolve_memory_dir calls in launcher.py are read out
   of the source and asserted to pass settings.data_dir. The resolver itself was
   never wrong; its callers were, so a resolver-only test proves nothing here.
3. THE FENCE - the product actually runs, writing logs, a transcript, a chat
   store, a benchmark report, the journal, the Caddyfile and a registry edit,
   against a REAL install tree that is hashed before and after. Nothing may
   change beneath it. This is a watched tree, not an assertion about intent: it
   is the only layer that would catch a write through a path the static rules
   cannot follow.

Keyword: data_root_ownership (see AC-M14-27's verification command).
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import benchmark
import config
import finetune
import logger
import memory
import secure_proxy
import webui
from config import Settings

PLATFORM_DIR = Path(__file__).resolve().parent.parent
FENCE_SCRIPT = PLATFORM_DIR / "scripts" / "verify_write_fence.py"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_settings(tmp_path: Path) -> tuple[Settings, Path, Path]:
    """A Settings whose install tree and data root are DIFFERENT directories.

    Different on purpose: a fixture that let them coincide could not tell a
    correct resolver from the broken one, which is how the defect survived a
    green suite.
    """
    install = tmp_path / "install"
    data = tmp_path / "locitize-data"
    install.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return Settings(base_dir=install, data_dir=data), install, data


def snapshot(root: Path) -> dict[str, str]:
    """Map every file under `root` to the sha256 of its bytes."""
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digests


# --------------------------------------------------------------------------- #
# Layer 1: the resolver contract (Architecture M14.2.3.1 table)
# --------------------------------------------------------------------------- #


def test_data_root_ownership_every_resolver_returns_a_path_under_the_data_root(
    tmp_path, monkeypatch
):
    """Every resolver in the M14.2.3.1 table resolves under data_dir, not base_dir."""
    # The suite-wide LOCITIZE_LOG_DIR override outranks the resolver by design,
    # so it is removed here to test the resolver rather than the override.
    monkeypatch.delenv("LOCITIZE_LOG_DIR", raising=False)
    settings, install, data = make_settings(tmp_path)

    resolved = {
        "logs": logger.resolve_log_dir(settings),
        "memory": memory.resolve_memory_dir(settings.data_dir, settings.memory.dir),
        "webui-data": webui.webui_data_dir(settings),
        "reports": benchmark.resolve_results_dir(settings),
        "Caddyfile": secure_proxy.resolve_caddyfile_path(settings),
        "finetune outputs": finetune.outputs_root(settings),
        "finetune datasets": finetune.datasets_root(settings),
        "models": config.resolve_models_dir(settings),
    }

    for label, path in resolved.items():
        assert path is not None, f"{label} did not resolve"
        resolved_path = Path(path).resolve()
        assert data.resolve() in resolved_path.parents or resolved_path == data.resolve(), (
            f"{label} resolved to {resolved_path}, which is not under the data root"
        )
        assert install.resolve() not in resolved_path.parents, (
            f"{label} resolved into the INSTALL tree ({resolved_path}) - "
            f"this is the shipped defect NEW-QA-M14-8"
        )

    # The exact locations the contract names, so a resolver cannot satisfy the
    # rule above by inventing a different folder.
    assert resolved["logs"] == data / "logs"
    assert resolved["memory"] == (data / "memory").resolve()
    assert resolved["webui-data"] == (data / "webui-data").resolve()
    assert resolved["reports"] == data / "reports"
    assert resolved["Caddyfile"] == data / "Caddyfile"
    assert resolved["finetune outputs"] == data / "finetune" / "outputs"
    assert resolved["finetune datasets"] == data / "finetune" / "datasets"
    assert resolved["models"] == data / "models"


def test_data_root_ownership_memory_escape_refusal_is_anchored_on_the_data_root(
    tmp_path
):
    """A memory.dir that tries to escape lands back inside the DATA root."""
    settings, install, data = make_settings(tmp_path)
    escaped = memory.resolve_memory_dir(settings.data_dir, "../../elsewhere")
    assert escaped == data / "memory"


def test_data_root_ownership_an_explicit_relocation_still_wins(tmp_path):
    """The two documented exceptions stay honoured exactly as configured.

    INSTALL.md names models_hub.download_dir and finetune.outputs_dir as the only
    two locations outside the data folder, so they must NOT be forced back into
    it - the backup claim names them instead.
    """
    settings, _install, _data = make_settings(tmp_path)
    elsewhere = tmp_path / "other-drive"
    settings.models_hub.download_dir = str(elsewhere / "models")
    settings.finetune.outputs_dir = str(elsewhere / "outputs")
    assert config.resolve_models_dir(settings) == elsewhere / "models"
    assert finetune.outputs_root(settings) == elsewhere / "outputs"


# --------------------------------------------------------------------------- #
# Layer 2: the call sites (the half the resolver test cannot see)
# --------------------------------------------------------------------------- #


def test_data_root_ownership_launcher_memory_call_sites_pass_the_data_root():
    """All three resolve_memory_dir calls in launcher.py pass settings.data_dir.

    Read out of the source rather than exercised, because the defect was three
    identical arguments in three different flows (chat, memory search, the voice
    session) and a behavioural test would have to reach all three to see it.
    """
    tree = ast.parse((PLATFORM_DIR / "launcher.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", "")) == "resolve_memory_dir"
    ]
    assert len(calls) == 3, f"expected 3 resolve_memory_dir call sites, found {len(calls)}"
    for call in calls:
        first = call.args[0]
        assert isinstance(first, ast.Attribute) and first.attr == "data_dir", (
            f"launcher.py:{call.lineno} passes "
            f"{ast.unparse(first)} to resolve_memory_dir; the contract is "
            f"settings.data_dir (DEC-M14-9)"
        )


def test_data_root_ownership_journal_resolves_through_the_reports_resolver():
    """_append_journal writes under the same reports root the benchmark uses."""
    source = (PLATFORM_DIR / "launcher.py").read_text(encoding="utf-8")
    body = source.split("def _append_journal", 1)[1].split("\n    def ", 1)[0]
    assert "resolve_results_dir" in body
    assert 'base_dir / "docs"' not in body


# --------------------------------------------------------------------------- #
# Layer 3a: the structural W1 fence
# --------------------------------------------------------------------------- #


def test_data_root_ownership_w1_structural_fence_reports_nothing():
    """No runtime module composes a writable path from settings.base_dir."""
    sys.path.insert(0, str(PLATFORM_DIR / "scripts"))
    import verify_write_fence

    findings = verify_write_fence.scan()
    assert findings == [], "\n".join(f.render() for f in findings)


def test_data_root_ownership_w1_fence_script_runs_clean_as_a_command():
    """The fence is a command anyone can run, not only an importable function."""
    result = subprocess.run(
        [sys.executable, str(FENCE_SCRIPT), "--json"],
        cwd=str(PLATFORM_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True and report["findings"] == []


# --------------------------------------------------------------------------- #
# Layer 3b: the real write fence - a watched install tree
# --------------------------------------------------------------------------- #


def _fake_install_tree(tmp_path: Path) -> tuple[Settings, Path, Path]:
    """Build an install tree that looks like a real one, plus a separate data root.

    The tree carries the shapes a real install has and that the shipped defect
    grew into: docs/, a shipped template, and the code marker file. Anything
    appearing beneath it during the run is a W1 violation.
    """
    install = tmp_path / "Codebase" / "platform"
    (install / "docs").mkdir(parents=True)
    (install / "docs" / "chat.md").write_text("shipped documentation\n", encoding="utf-8")
    (install / "settings.default.yaml").write_text("version: 2\n", encoding="utf-8")
    (install / "launcher.py").write_text("# shipped code\n", encoding="utf-8")
    # The Open WebUI venv lives BESIDE the install (install-scoped cache, created
    # at install time), so creating it here is not a runtime write.
    scripts = tmp_path / "Codebase" / ".webui-venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "open-webui.exe").write_bytes(b"fake console script")

    data = tmp_path / "locitize-data"
    data.mkdir()
    (data / "models.yaml").write_text(
        "version: 1\nmodels:\n"
        '  - id: "m"\n'
        '    name: "M"\n'
        '    description: ""\n'
        '    location: "/locitize-test/models/m.gguf"\n'
        "    context_size: 4096\n"
        "    gpu_layers: 999\n"
        "    benchmark_score: null\n"
        "    status: installed\n",
        encoding="utf-8",
    )
    return Settings(base_dir=install, data_dir=data), install, data


def _drive_the_product(settings: Settings) -> None:
    """Exercise every writer the ownership contract names, for real.

    Each call here is the product's own writer, reached the way the running app
    reaches it. configure_logging touches process-global logging state, so the
    handlers it adds are removed again in the caller's finally block.
    """
    # Logs (logger.configure_logging -> rotating handlers on disk).
    logger.configure_logging(settings)
    logging.getLogger("launcher").warning("fence test log line")

    # A conversation transcript, through the product's own store.
    store = memory.ConversationMemory(
        memory.resolve_memory_dir(settings.data_dir, settings.memory.dir),
        enabled=True,
    )
    store.append("fence-session", "user", "hello from the fence test")

    # The Open WebUI data dir (build_openwebui_spec creates it).
    webui.build_openwebui_spec(settings, "openwebui.log")

    # A benchmark report and its markdown section.
    runner = benchmark.BenchmarkRunner(
        controller=None, registry=None, settings=settings
    )
    result = benchmark.BenchmarkResult(
        model_id="m",
        session_id="fence",
        scenario_key="fence",
        timestamp="2026-08-19T00:00:00",
        ok=False,
        reason="fence test row, never measured",
        gpu_layers=999,
        context_size=4096,
    )
    runner._append_markdown("fence", "Fence GPU", [result])
    jsonl = runner._results_dir / "benchmark_results.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a", encoding="utf-8") as handle:
        handle.write(benchmark.format_jsonl_line(result))

    # The development journal, through the launcher's own writer.
    import launcher

    class _Status:
        value = "PASS"

    class _Report:
        results: list = []
        overall = _Status()

    launcher.Launcher()._append_journal(settings, _Report())

    # The generated Caddyfile: the location decision is resolve_caddyfile_path;
    # secure_proxy.ensure() itself cannot run here (it needs a hosts entry, a
    # caddy binary and Windows trust prompts), so the write it performs is
    # reproduced against the resolved path.
    caddyfile = secure_proxy.resolve_caddyfile_path(settings)
    caddyfile.parent.mkdir(parents=True, exist_ok=True)
    caddyfile.write_text(secure_proxy._caddyfile_body("locitize.local", 8081), encoding="utf-8")

    # Fine-tune locations.
    finetune.outputs_root(settings).mkdir(parents=True, exist_ok=True)
    finetune.datasets_root(settings).mkdir(parents=True, exist_ok=True)

    # A registry edit through the one chokepoint.
    config.write_model_fields(settings.data_dir, "m", 10, 2048)


def test_data_root_ownership_running_the_product_writes_nothing_into_the_install_tree(
    tmp_path, monkeypatch
):
    """Invariant W1, measured: the install tree is byte-identical afterwards.

    This is the layer that does not care HOW a write was composed. It runs the
    real writers - logging, the transcript store, the Open WebUI spec builder,
    the benchmark report writer, the journal, the Caddyfile, the registry - and
    then diffs the install tree file by file.

    The process working directory is moved INTO the watched install tree for the
    duration, because that is what LOCITIZE.bat line 9 (`cd /d "%~dp0"`) does on a
    real machine. Without it, a relative-path write went to whatever directory
    pytest was started from and this test scored it clean - the hole a reviewer
    demonstrated with a one-line `Path("locitize-crash.txt").write_text("boom")` in
    logger.py (round 6, MEDIUM-1).
    """
    monkeypatch.delenv("LOCITIZE_LOG_DIR", raising=False)
    settings, install, data = _fake_install_tree(tmp_path)
    monkeypatch.chdir(install)

    before = snapshot(install)
    root_logger = logging.getLogger()
    existing_root = list(root_logger.handlers)
    channels = list(settings.logging.channels)
    existing_channels = {name: list(logging.getLogger(name).handlers) for name in channels}
    try:
        _drive_the_product(settings)
    finally:
        # Put process-global logging back exactly as it was, and close the files
        # so the tmp tree can be removed on Windows.
        for handler in list(root_logger.handlers):
            if handler not in existing_root:
                handler.close()
                root_logger.removeHandler(handler)
        for name in channels:
            channel_logger = logging.getLogger(name)
            for handler in list(channel_logger.handlers):
                if handler not in existing_channels.get(name, []):
                    handler.close()
                    channel_logger.removeHandler(handler)

    after = snapshot(install)
    added = sorted(set(after) - set(before))
    changed = sorted(k for k in set(after) & set(before) if after[k] != before[k])
    removed = sorted(set(before) - set(after))
    assert not added, f"runtime wrote NEW files into the install tree: {added}"
    assert not changed, f"runtime modified install-tree files: {changed}"
    assert not removed, f"runtime deleted install-tree files: {removed}"

    # And the same run really did produce the user's data, in the data root -
    # otherwise "nothing was written to the install tree" would be trivially
    # true because nothing was written anywhere.
    assert (data / "logs" / "launcher.log").is_file()
    assert list((data / "memory").glob("*.jsonl"))
    assert (data / "webui-data").is_dir()
    assert (data / "reports" / "benchmark_results.md").is_file()
    assert (data / "reports" / "benchmark_results.jsonl").is_file()
    assert (data / "reports" / "development_journal.md").is_file()
    assert (data / "Caddyfile").is_file()
    assert (data / "finetune" / "outputs").is_dir()
    assert (data / "finetune" / "datasets").is_dir()
    assert "gpu_layers: 10" in (data / "models.yaml").read_text(encoding="utf-8")


def test_data_root_ownership_portable_mode_puts_the_data_root_inside_the_install(
    tmp_path
):
    """Portable mode: <install>/locitize-data IS the data root, and W1 allows it.

    The one benign overlap the invariant names. Creating the folder is the whole
    opt-in, so this also pins that an install WITHOUT the folder does not
    accidentally become portable.
    """
    install = tmp_path / "install"
    install.mkdir()
    env: dict[str, str] = {"LOCALAPPDATA": str(tmp_path / "AppData")}

    assert config.resolve_data_dir(env, install) == tmp_path / "AppData" / "LOCITIZE"
    (install / "locitize-data").mkdir()
    portable = config.resolve_data_dir(env, install)
    assert portable == install / "locitize-data"

    # Every resolver then lands inside that subtree - which is data-root
    # territory, not install territory.
    settings = Settings(base_dir=install, data_dir=portable)
    assert benchmark.resolve_results_dir(settings) == portable / "reports"
    # LOCITIZE_LOG_DIR still outranks the data root (Architecture M5.12), and the
    # suite sets it, so the log dir is compared through the same override-free
    # path the previous test uses rather than re-asserting the override here.
    assert config.resolve_models_dir(settings) == portable / "models"


# --------------------------------------------------------------------------- #
# Layer 3c: the child processes' working directory
# --------------------------------------------------------------------------- #


def _spec_builders(settings: Settings) -> dict:
    """Every ServiceSpec a running LOCITIZE builds, keyed by what launches it.

    Built here rather than asserted from source because the value under test is
    the one Popen receives, and only a real builder produces that.
    """
    import tts
    import whisper
    from models import Model, ModelRegistry, ModelRegistryData

    registry = ModelRegistry(
        ModelRegistryData(
            version=1,
            models=[
                Model(
                    id="m",
                    name="M",
                    description="d",
                    location="/locitize-test/models/m.gguf",
                    context_size=4096,
                    gpu_layers=-1,
                    status="installed",
                )
            ],
        ),
        settings,
    )
    return {
        "llama.cpp": registry.build_start_spec("m"),
        "Open WebUI": webui.build_openwebui_spec(settings, "openwebui.log"),
        "whisper-server": whisper.build_whisper_server_spec(settings),
        "whisper-stream": whisper.build_whisper_stream_spec(settings),
        "kokoro": tts.build_kokoro_server_spec(settings),
    }


def test_data_root_ownership_every_service_spec_names_a_working_directory(tmp_path):
    """Invariant W1 for CHILD processes, which the fence's two layers cannot see.

    A ServiceSpec with `cwd=None` makes the child inherit LOCITIZE's own working
    directory, and LOCITIZE.bat line 9 (`cd /d "%~dp0"`) makes that the install
    directory - so every relative file llama-server, Open WebUI, whisper or
    kokoro wrote landed in the install tree. Neither the static fence (a relative
    path names no base_dir) nor the watched-tree test (it watches a synthetic
    tree) could observe it (round 6, MEDIUM-1).
    """
    settings, install, data = _fake_install_tree(tmp_path)
    settings.paths.llama_cpp = str(tmp_path / "bin" / "llama-server.exe")
    settings.paths.whisper = str(tmp_path / "bin" / "whisper-server.exe")
    settings.paths.whisper_model = str(tmp_path / "bin" / "ggml.bin")
    settings.paths.whisper_stream = str(tmp_path / "bin" / "whisper-stream.exe")
    settings.paths.kokoro_model = str(tmp_path / "bin" / "kokoro.pth")
    settings.paths.kokoro_voices = str(tmp_path / "bin" / "voices")

    for label, spec in _spec_builders(settings).items():
        assert spec.cwd, f"{label} inherits LOCITIZE's working directory (cwd=None)"
        resolved = Path(spec.cwd).resolve()
        assert install.resolve() not in [resolved, *resolved.parents], (
            f"{label} runs inside the install tree ({spec.cwd}); a relative file "
            f"it writes lands in the install directory (invariant W1)"
        )
        assert data.resolve() in [resolved, *resolved.parents], (
            f"{label} runs at {spec.cwd}, which is outside the data root"
        )


def test_data_root_ownership_a_service_working_directory_is_created_before_launch(
    tmp_path,
):
    """Popen fails outright on a missing cwd, so the scratch folder is made first.

    The behavioural half of the previous test: a spec that names a directory
    nobody creates would trade a silent write into the install tree for a
    service that cannot start at all.
    """
    import services

    settings, _install, data = _fake_install_tree(tmp_path)
    target = services.resolve_service_cwd(settings)
    assert not Path(target).exists()
    assert services._ensure_working_dir(target) == target
    assert Path(target).is_dir()
    assert Path(target).resolve().parent == data.resolve()
