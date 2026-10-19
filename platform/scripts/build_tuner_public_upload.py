"""Build a separate public-upload copy of the standalone local plugin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import zipfile

from build_standalone_tuner import build


def _public_manifest(raw: bytes) -> bytes:
    """Strip private app bindings from the upload copy, never from source."""
    manifest = json.loads(raw)
    manifest.pop("apps", None)
    extension = manifest.get("extensions", {}).get("com.openai", {})
    extension.pop("apps", None)
    return (json.dumps(manifest, indent=2) + "\n").encode("utf-8")


def build_public_upload(root: Path, destination: Path, archive: Path) -> dict:
    """Stage one self-contained plugin directory and ZIP it without private state."""
    root = root.resolve()
    destination = destination.resolve()
    archive = archive.resolve()
    if destination.exists():
        raise ValueError("Use a new public-upload staging destination")
    if archive.exists():
        raise ValueError("Use a new public-upload archive path")
    package = destination / "locitize-tuner"
    package.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="locitize-tuner-public-build-") as temporary:
        source_archive = Path(temporary) / "standalone.zip"
        report = build(root, source_archive)
        with zipfile.ZipFile(source_archive) as source:
            for member in source.infolist():
                if member.is_dir() or member.filename == ".app.json":
                    continue
                target = (package / member.filename).resolve()
                target.relative_to(package)
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = source.read(member)
                if member.filename == "plugin.json":
                    payload = _public_manifest(payload)
                target.write_bytes(payload)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(package.rglob("*")):
            if path.is_symlink():
                raise ValueError("Public upload cannot contain symlinks")
            if path.is_file():
                output.write(path, path.relative_to(destination).as_posix())
    report.update({"staging": str(destination), "package": str(package),
                   "public_archive": str(archive), "public_archive_bytes": archive.stat().st_size,
                   "root_directory": "locitize-tuner", "private_app_bindings_included": False})
    return report


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[2]
    version = json.loads((repository / "plugins/locitize-tuner/plugin.json").read_text())["version"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path,
                        default=repository / "dist" / f"locitize-tuner-public-upload-{version}")
    parser.add_argument("--archive", type=Path,
                        default=repository / "dist" / f"locitize-tuner-public-upload-{version}.zip")
    arguments = parser.parse_args()
    print(json.dumps(build_public_upload(repository, arguments.destination, arguments.archive)))
