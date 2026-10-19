"""Build an explicit source-only standalone helper archive without private data."""
from pathlib import Path
import argparse
import json
import zipfile

# Executable closure audited through real provider/registry imports. Shared
# protocol modules remain included; optional engines/assets/dependencies do not.
MODULES = (
    "audio_api", "audio_filter", "backends", "benchmark", "benchmark_tasks", "codex_desktop_local", "codex_picker",
    "config", "egress_log", "finetune", "gguf_meta", "gpu_ledger", "health",
    "launcher", "local_tuning", "local_tuning_mcp", "logger", "modelhub",
    "models", "model_discovery", "registry_lock", "responses_tools", "router", "runtime_layout",
    "services", "setup_env",
)
PACKAGE_FILES = ("plugin.json", "mcp.json", "server.py", "bootstrap.py", "README.md",
                 "COMPATIBILITY.md", "requirements-helper.txt", "skills/tune-local-model/SKILL.md")
PACKAGE_FILES += ("assets/locitize.png",)
ICON_SHA256 = "1c09c2799718cccc5962e2613fbc2a6ac4d4a669a59fd3249951308bb0c9e137"
ROOT_FILES = ("LICENSE",)
# The redistribution notice is bundled unchanged, alongside the reviewed source.
LICENSE_SHA256 = "acbc3211bef31b91c93ee07f7b8268b451464371a8a7467d6366f500300858f7"


def build(root, output):
    """Never glob private runtime trees; only reviewed source enters the archive."""
    package = root / "plugins" / "locitize-tuner"
    output.parent.mkdir(parents=True, exist_ok=True)
    source_size = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in PACKAGE_FILES:
            path = package / relative
            archive.write(path, relative)
        for relative in ROOT_FILES:
            archive.write(root / relative, relative)
        for module in MODULES:
            path = root / "platform" / (module + ".py")
            source_size += path.stat().st_size
            archive.write(path, "runtime/" + path.name)
        archive.writestr("runtime/settings.default.yaml", "version: 2\npaths:\n  llama_cpp: ''\nrouter:\n  enabled: true\n")
        archive.writestr("runtime/models.default.yaml", "version: 2\nmodels: []\n")
    return {"archive": str(output), "archive_bytes": output.stat().st_size,
            "source_bytes": source_size, "source_modules": len(MODULES),
            "dependencies_bundled": False, "runtime_binary_bundled": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(build(root, args.output or root / "dist" / "locitize-tuner-standalone.zip")))
