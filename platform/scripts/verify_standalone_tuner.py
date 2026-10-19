"""Check standalone archive boundaries and exercise extracted MCP installation.

This harness never starts a model or allocates GPU memory. Acquisition/inference
remain separately recorded gates; inspection-only setup is labeled honestly.
"""
from __future__ import annotations
import argparse
import asyncio
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET


def check_archive(archive):
    """Reject non-allowlisted files, secret literals, private paths and bad metadata."""
    from build_standalone_tuner import MODULES, PACKAGE_FILES, ROOT_FILES, LICENSE_SHA256, ICON_SHA256
    import jsonschema
    expected = set(PACKAGE_FILES) | set(ROOT_FILES) | {"runtime/" + name + ".py" for name in MODULES} | {
        "runtime/models.default.yaml", "runtime/settings.default.yaml"}
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Archive file allowlist mismatch or duplicate member")
        import hashlib
        if hashlib.sha256(package.read("LICENSE")).hexdigest() != LICENSE_SHA256:
            raise ValueError("Redistribution license differs from the unchanged notice")
        for name in names:
            if name == "assets/locitize.png":
                continue  # Binary icon integrity and dimensions are checked below.
            text = package.read(name).decode("utf-8-sig")
            if re.search(r"(?i)[a-z]:[\\/](?:users|workspace|locitize|operynth)[\\/]", text.replace("\\\\", "\\")):
                raise ValueError("Private machine path in " + name)
            if re.search(r"(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)", text):
                raise ValueError("Secret-shaped literal in " + name)
            if name.endswith(".py"):
                ast.parse(text, filename=name)
        manifest = json.loads(package.read("plugin.json"))
        # Validate this package's required manifest contract. This is a local
        # structural gate, not marketplace's independent schema/review approval.
        jsonschema.validate(manifest, {"type": "object", "required": ["name", "version", "description", "extensions"],
            "properties": {"name": {"const": "locitize-tuner"}, "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
            "description": {"type": "string", "minLength": 1}, "extensions": {"type": "object"}}})
        extension = manifest["extensions"]["com.openai"]
        interface = extension["interface"]
        if len(interface["displayName"]) > 30 or len(interface["shortDescription"]) > 30:
            raise ValueError("Listing name or subtitle exceeds 30 characters")
        if len(interface["longDescription"]) > 4000:
            raise ValueError("Long description exceeds 4000 characters")
        for key in ("logo", "composerIcon"):
            if interface[key] != "./assets/locitize.png":
                raise ValueError("Unexpected icon reference")
        icon = package.read("assets/locitize.png")
        import struct
        if hashlib.sha256(icon).hexdigest() != ICON_SHA256:
            raise ValueError("Icon differs from the unchanged approved product asset")
        if icon[:8] != b"\x89PNG\r\n\x1a\n" or icon[12:16] != b"IHDR":
            raise ValueError("Icon must be a PNG image")
        width, height = struct.unpack(">II", icon[16:24])
        if width != height or width < 48:
            raise ValueError("Icon must be square and at least 48 pixels")
        if interface["displayName"] != "LoCiTiZe" or "LOCK-ih-tize" not in interface["longDescription"]:
            raise ValueError("Product name or pronunciation differs from approved listing")
        cases = extension["review"]["test_cases"]
        if len(cases["positive"]) != 5 or len(cases["negative"]) != 3:
            raise ValueError("Expected exactly five positive and three negative review cases")
        if not extension["publication"]["release_notes"]:
            raise ValueError("Missing release notes")
        transport = json.loads(package.read("mcp.json"))
        jsonschema.validate(transport, {"type": "object", "required": ["mcpServers"],
            "properties": {"mcpServers": {"type": "object", "required": ["locitize-tuner"]}}})
        server = transport["mcpServers"]["locitize-tuner"]
        if server != {"type": "stdio", "command": "python", "args": ["${PLUGIN_ROOT}/server.py"], "cwd": "${PLUGIN_ROOT}"}:
            raise ValueError("Unexpected transport or startup command")
        requirements = package.read("requirements-helper.txt").decode("utf-8-sig")
        pins = [line for line in requirements.splitlines() if line and not line.startswith("#")]
        if not pins or any(not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", line) for line in pins):
            raise ValueError("Dependencies must be exact version pins")
        banned = {"pyside6", "torch", "pillow", "open-webui", "streamlit"}
        if any(line.split("==")[0].lower() in banned for line in pins):
            raise ValueError("Optional heavyweight dependency included")
        compatibility = package.read("COMPATIBILITY.md").decode("utf-8-sig")
        if "Measured results" not in compatibility or "untested" not in compatibility:
            raise ValueError("Compatibility disclosure missing")
    return {"archive_bytes": Path(archive).stat().st_size, "members": len(names),
            "dependency_pins": len(pins), "local_schema_valid": True,
            "compatibility_document_included": True}


def check_public_archive(archive):
    """Inspect the author-supplied public ZIP and report truthful submission gaps."""
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        prefix = "locitize-tuner/"
        if not names or any(not name.startswith(prefix) for name in names):
            raise ValueError("Public ZIP must contain one locitize-tuner directory")
        if any(name.endswith("/.app.json") or "/.app.json" in name for name in names):
            raise ValueError("Public ZIP contains .app.json")
        with tempfile.TemporaryDirectory(prefix="locitize-public-verify-") as temporary:
            flat = Path(temporary) / "flat.zip"
            with zipfile.ZipFile(flat, "w", zipfile.ZIP_DEFLATED) as target:
                for name in names:
                    relative = name[len(prefix):]
                    if relative:
                        target.writestr(relative, package.read(name))
            report = check_archive(flat)
        manifest = json.loads(package.read(prefix + "plugin.json"))
        if manifest.get("apps") is not None or manifest.get("extensions", {}).get("com.openai", {}).get("apps") is not None:
            raise ValueError("Public ZIP contains an app binding")
        interface = manifest["extensions"]["com.openai"]["interface"]
        required_listing = ("developerName", "category", "websiteURL", "supportURL",
                            "privacyPolicyURL", "termsOfServiceURL")
        gaps = ["interface." + key for key in required_listing if not interface.get(key)]
        review = manifest["extensions"]["com.openai"].get("review", {})
        publication = manifest["extensions"]["com.openai"].get("publication", {})
        if not review.get("demo_recording_url"):
            gaps.append("review.demo_recording_url")
        if "commerce" not in review:
            gaps.append("review.commerce")
        if "countries" not in publication:
            gaps.append("publication.countries")
        report["payload_archive_bytes"] = report.pop("archive_bytes")
        report.update({"archive_bytes": Path(archive).stat().st_size,
                       "public_root": "locitize-tuner", "app_bindings_absent": True,
                       "submission_metadata_gaps": gaps,
                       "submission_metadata_complete": not gaps})
        return report


async def protocol(package, direct_runtime=False):
    """Exercise actual startup/discovery and path-free real assessment responses."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    env = dict(os.environ)
    env.pop("LOCITIZE_INSTALL_DIR", None)
    temporary = None
    if direct_runtime:
        temporary = tempfile.TemporaryDirectory(prefix="locitize-tuner-protocol-data-")
        env["LOCITIZE_DATA_DIR"] = temporary.name
        env["LOCITIZE_TUNER_DATA_DIR"] = temporary.name
        command = sys.executable
        arguments = [str(package / "runtime/local_tuning_mcp.py")]
    else:
        env.pop("LOCITIZE_DATA_DIR", None)
        command = "python"
        arguments = [str(package / "server.py")]
    # Match the installed plugin's actual host command instead of silently
    # substituting the test client's already-prepared interpreter.
    try:
        async with stdio_client(StdioServerParameters(command=command,
            args=arguments, cwd=str(package), env=env)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                names = {tool.name for tool in tools}
                required = {"assess_computer", "list_local_models", "start_tuning", "tuning_status",
                            "cancel_tuning", "start_local_model", "start_local_chat", "stop_local_model",
                            "enable_codex_picker", "disable_codex_picker", "search_models",
                            "list_model_files", "recommend_model_candidates", "start_model_download",
                            "download_status", "cancel_download", "resume_download", "register_download"}
                if names != required:
                    raise ValueError("MCP discovery mismatch")
                read_only = {"assess_computer", "list_local_models", "tuning_status", "search_models",
                             "list_model_files", "recommend_model_candidates", "download_status"}
                outbound = {"search_models", "list_model_files", "start_model_download", "resume_download"}
                for tool in tools:
                    annotation = tool.annotations
                    if annotation is None or annotation.readOnlyHint is not (tool.name in read_only):
                        raise ValueError("Incorrect read-only annotation: " + tool.name)
                    if annotation.destructiveHint is not (tool.name not in read_only):
                        raise ValueError("Incorrect destructive annotation: " + tool.name)
                    if annotation.openWorldHint is not (tool.name in outbound):
                        raise ValueError("Incorrect open-world annotation: " + tool.name)
                outcomes = {}
                for name in ("assess_computer", "list_local_models", "stop_local_model"):
                    response = await session.call_tool(name, {})
                    payload = json.loads(next(block.text for block in response.content if hasattr(block, "text")))
                    if not payload["ok"]:
                        raise ValueError("MCP call failed: " + name)
                    outcomes[name] = payload["result"]
                invalid = await session.call_tool("start_tuning", {"model_id": "../../not-a-model", "objective": "invalid"})
                payload = json.loads(next(block.text for block in invalid.content if hasattr(block, "text")))
                if payload["ok"]:
                    raise ValueError("Invalid tuning request accepted")
                if outcomes["list_local_models"]["models"] != []:
                    raise ValueError("Fresh registry unexpectedly contains models")
                return {"tools": sorted(names), "hardware": outcomes["assess_computer"],
                        "fresh_models_empty": True, "invalid_request_rejected": True,
                        "owned_stop_confirmed": outcomes["stop_local_model"]["cleanup_confirmed"],
                        "startup": "staged runtime with existing dependencies" if direct_runtime else "packaged bootstrap entry point"}
    finally:
        if temporary is not None:
            temporary.cleanup()


def installation(archive):
    """Extract into a new root and bootstrap a fresh isolated environment."""
    package = Path(tempfile.mkdtemp(prefix="locitize-tuner-install-check-"))
    with zipfile.ZipFile(archive) as source:
        source.extractall(package)  # Safe after exact allowlist check_archive.
    data = package / "private-data"
    env = dict(os.environ)
    env.pop("LOCITIZE_INSTALL_DIR", None)
    env.pop("LOCITIZE_DATA_DIR", None)
    env["LOCITIZE_TUNER_DATA_DIR"] = str(data)
    subprocess.run([sys.executable, str(package / "bootstrap.py"), "--skip-runtime", "--skip-discovery"],
                   cwd=package, env=env, check=True, timeout=600)
    python = data / "environment" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "check"], cwd=package, env=env,
                   check=True, timeout=90)
    # Run the harness inside the private environment without checkout imports.
    copied = package / "verify_install.py"
    shutil.copy2(__file__, copied)
    result = subprocess.run([str(python), str(copied), "--protocol-only", "--package", str(package)],
                            cwd=package, env=env, text=True, capture_output=True, check=True, timeout=90)
    report = json.loads(result.stdout)
    report.update({"extracted_package": str(package), "dependency_environment_bytes":
        sum(path.stat().st_size for path in (data / "environment").rglob("*") if path.is_file()),
        "runtime_acquisition": "deliberately skipped for inspection-only installation test"})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--direct-runtime", action="store_true")
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    if args.protocol_only:
        report = asyncio.run(protocol(args.package, direct_runtime=args.direct_runtime))
    else:
        report = check_public_archive(args.archive) if args.public else check_archive(args.archive)
        if args.install:
            report["installation"] = installation(args.archive)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
