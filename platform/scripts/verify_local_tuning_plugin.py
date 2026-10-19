"""Exercise the local plugin through a real MCP client and inspect its test archive."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = json.loads((ROOT / "plugins/locitize-tuner/plugin.json").read_text())["version"]

def catalog_package(root: Path) -> Path:
    """Verify the owning catalog payload, including immutable correction suffixes."""
    catalog = json.loads((root / ".agents/plugins/marketplace.json").read_text())
    entries = [row for row in catalog.get("plugins", []) if row.get("name") == "locitize-tuner"]
    if len(entries) != 1:
        raise ValueError("Catalog must contain exactly one owning plugin")
    source = entries[0].get("source") or {}
    if source.get("source") != "local" or not isinstance(source.get("path"), str):
        raise ValueError("Owning plugin must have a local catalog path")
    package = (root / source["path"]).resolve()
    package.relative_to(root.resolve())
    return package


PACKAGE = catalog_package(ROOT)
TOOLS = {
    "assess_computer", "list_local_models", "start_tuning", "tuning_status",
    "cancel_tuning", "start_local_model", "stop_local_model", "start_local_chat",
    "search_models", "list_model_files", "recommend_model_candidates", "start_model_download",
        "download_status", "cancel_download", "resume_download", "register_download",
}


def helper_python() -> Path:
    """Keep the optional SDK separate from the desktop and Open WebUI stacks."""
    return ROOT / ".plugin-venv" / "Scripts" / "python.exe"


def unpack_result(result):
    """Prefer structured protocol content, with a JSON text compatibility fallback."""
    if result.isError:
        raise RuntimeError("MCP call returned an error")
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return json.loads(block.text)
    raise RuntimeError("MCP call returned no structured result")


async def protocol(package: Path = PACKAGE, real: bool = False, model_id: str | None = None) -> dict:
    """Start the packaged entrypoint and perform genuine protocol calls."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ)
    env.pop("LOCITIZE_INSTALL_DIR", None)
    env.pop("LOCITIZE_DATA_DIR", None)
    assert (package / "runtime/local_tuning_mcp.py").is_file(), "Build the complete marketplace first"
    parameters = StdioServerParameters(
        command="python", args=[str(package / "server.py")],
        cwd=str(package), env=env,
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            discovered = {tool.name for tool in (await session.list_tools()).tools}
            assert TOOLS <= discovered, "Required tools were not discovered"
            hardware = unpack_result(await session.call_tool("assess_computer", {}))
            models = unpack_result(await session.call_tool("list_local_models", {}))
            assert hardware.get("ok") and models.get("ok"), "Real assessment failed"
            invalid = await session.call_tool("tuning_status", {"job_id": "missing-job"})
            # Invalid identities may be represented as an MCP error or a typed result.
            assert invalid.isError or unpack_result(invalid).get("ok") is False
            report = {"tools": sorted(discovered), "hardware": hardware, "models": models}
            if real:
                report["real"] = await real_checks(session, model_id, package)
            return report


async def call(session, name: str, **arguments):
    """Unwrap the adapter's typed success envelope without hiding failures."""
    envelope = unpack_result(await session.call_tool(name, arguments))
    assert envelope.get("ok"), envelope.get("error", "operation failed")
    return envelope["result"]


async def wait_job(session, job_id: str, timeout: float = 5400) -> dict:
    """Poll actual job state; a cancelled request is not itself completed cleanup."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        status = await call(session, "tuning_status", job_id=job_id)
        if status["state"] in {"completed", "cancelled", "failed", "recovery_failed", "target_not_met"}:
            return status
        await asyncio.sleep(1)
    await call(session, "cancel_tuning", job_id=job_id)
    raise RuntimeError("Tuning exceeded verifier budget; cancellation requested")


async def real_checks(session, model_id: str | None, package: Path) -> dict:
    """Exercise real cancellation, saved tuning and local Codex inference in sequence."""
    # These imports occur only after SDK initialization, avoiding stdlib name shadowing.
    platform_dir = package / "runtime"
    if str(platform_dir) not in sys.path:
        sys.path.insert(0, str(platform_dir))
    import config
    from verify_codex_local_model import run_verification

    data = Path(os.environ.get("LOCITIZE_TUNER_DATA_DIR") or
                str(Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local"))) / "LOCITIZE-Tuner"))
    _, registered, _ = config.Config.load(data)
    available = [m for m in registered.models if m.status == "installed"
                 and m.location and Path(m.location).is_file()]
    if model_id is None:
        bounded = [m for m in available if m.context_size <= 8192]
        selected = min(bounded or available, key=lambda m: Path(m.location).stat().st_size)
    else:
        selected = next(m for m in available if m.id == model_id)
    model_id = selected.id
    before = (selected.gpu_layers, selected.context_size)
    cancelled = await call(session, "start_tuning", model_id=model_id, objective="speed")
    # Allow the real trial to begin, so cancellation covers cleanup of a live model.
    await asyncio.sleep(2)
    await call(session, "cancel_tuning", job_id=cancelled["job_id"])
    cancelled = await wait_job(session, cancelled["job_id"])
    assert cancelled["state"] == "cancelled" and cancelled["cleanup_confirmed"]
    _, refreshed, _ = config.Config.load(data)
    row = next(m for m in refreshed.models if m.id == model_id)
    assert (row.gpu_layers, row.context_size) == before, "Cancellation changed saved settings"
    job = await call(session, "start_tuning", model_id=model_id, objective="speed")
    completed = await wait_job(session, job["job_id"])
    assert completed["state"] == "completed" and completed["cleanup_confirmed"], completed
    assert completed["verification"]["tokens_per_second"] > 0
    assert any(t["gpu_layers_requested"] in (16, 32) for t in completed["trials"])
    try:
        started = await call(session, "start_local_model", model_id=model_id)
        assert started["completion_verified"]
        assert started["context_effective"] is not None, "Running context was not observed"
        codex = await asyncio.to_thread(
            run_verification, model_id, started["router_port"],
            context_size=started["context_effective"], task="mcp-fixture")
    finally:
        stopped = await call(session, "stop_local_model")
        assert stopped["cleanup_confirmed"]
    return {"status": codex["status"], "model_id": model_id, "cancellation": cancelled,
            "tuning": completed, "codex": codex, "stop": stopped}


def package_archive() -> Path:
    """Build and validate the full standalone payload, including its runtime/license."""
    from build_standalone_tuner import build
    from verify_standalone_tuner import check_archive
    version = json.loads((ROOT / "plugins/locitize-tuner/plugin.json").read_text())["version"]
    destination = ROOT / "dist" / ("locitize-tuner-local-test-" + version + ".zip")
    build(ROOT, destination)
    check_archive(destination)
    return destination


def main() -> int:
    """Re-execute the verifier in its optional SDK environment when necessary."""
    if importlib.util.find_spec("mcp") is None:
        python = helper_python()
        if not python.is_file():
            raise RuntimeError("Run the helper environment setup first")
        return subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
                               cwd=ROOT / "platform")
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", action="store_true")
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.real:
        report = asyncio.run(protocol(real=True, model_id=args.model))
        print(json.dumps(report, indent=2))
        return 0 if report["real"]["status"] == "PASS" else 1
    elif args.package:
        archive = package_archive()
        from verify_standalone_tuner import installation
        result = installation(archive)
        print(json.dumps({"package": str(archive), "protocol": result}, indent=2))
    else:
        print(json.dumps(asyncio.run(protocol()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
