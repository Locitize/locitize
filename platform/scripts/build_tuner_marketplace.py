"""Build a movable local marketplace whose catalog contains a complete helper."""
from pathlib import Path
import argparse
import json
import zipfile

from build_standalone_tuner import build


def default_paths(root):
    """Derive fresh build destinations from the owning, bounded local catalog."""
    root = root.resolve()
    catalog = json.loads((root / ".agents/plugins/marketplace.json").read_text())
    entries = catalog.get("plugins")
    if catalog.get("name") != "locitize-local" or not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("Unexpected marketplace identity or entries")
    entry = entries[0]
    source = entry.get("source") or {}
    relative = source.get("path")
    if entry.get("name") != "locitize-tuner" or source.get("source") != "local" or not isinstance(relative, str) or not relative:
        raise ValueError("Expected the owning local plugin catalog path")
    path = Path(relative)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError("Catalog payload must be a relative path within dist")
    package = (root / path).resolve()
    if package.name != "locitize-tuner" or package.parent.name != "plugins":
        raise ValueError("Catalog path must identify plugins/locitize-tuner")
    destination = package.parent.parent
    destination.relative_to(root / "dist")
    if not destination.name.startswith("locitize-tuner-marketplace-"):
        raise ValueError("Catalog path must identify a versioned marketplace")
    archive = root / "dist" / (destination.name.replace("locitize-tuner-marketplace-", "locitize-tuner-standalone-", 1) + ".zip")
    return destination, archive


def build_marketplace(root, destination, archive):
    """Generate from the explicit archive, never from private installed caches."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty marketplace destination to preserve existing artifacts")
    if archive.exists():
        raise ValueError("Use a new archive path to preserve existing artifacts")
    report = build(root, archive)
    package = destination / "plugins" / "locitize-tuner"
    package.mkdir(parents=True, exist_ok=True)
    # All members originate from the source allowlist; still reject path escape
    # so future builder changes cannot turn extraction into arbitrary writes.
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (package / member.filename).resolve()
            target.relative_to(package.resolve())
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(member))
    catalog = json.loads((root / ".agents/plugins/marketplace.json").read_text())
    if catalog["name"] != "locitize-local" or len(catalog["plugins"]) != 1:
        raise ValueError("Unexpected marketplace identity or entries")
    entry = catalog["plugins"][0]
    if entry["name"] != "locitize-tuner":
        raise ValueError("Unexpected plugin identity")
    entry["source"] = {"source": "local", "path": "./plugins/locitize-tuner"}
    metadata = destination / ".agents/plugins/marketplace.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    report.update({"marketplace": str(destination), "plugin": str(package),
                   "version": json.loads((package / "plugin.json").read_text())["version"]})
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    destination, archive = default_paths(root)
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path,
                        default=destination)
    parser.add_argument("--archive", type=Path,
                        default=archive)
    args = parser.parse_args()
    print(json.dumps(build_marketplace(root, args.destination, args.archive)))
