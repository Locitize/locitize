"""Detect the bundled Python layout without changing source-checkout behavior."""
from pathlib import Path


def bundled_python(code_dir=None):
    root = Path(code_dir) if code_dir is not None else Path(__file__).resolve().parent
    exe = root.parent / "runtime" / "python.exe"
    return exe if exe.is_file() else None


def packaged_environment_root(data_dir):
    from release_info import VERSION
    return Path(data_dir) / "environments" / VERSION


def environment_root(code_dir=None):
    root = Path(code_dir) if code_dir is not None else Path(__file__).resolve().parent
    if bundled_python(root):
        from config import resolve_data_dir
        return packaged_environment_root(resolve_data_dir(install_dir=root))
    return root.parent
