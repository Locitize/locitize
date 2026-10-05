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


def default_data_root(code_dir=None):
    """Where setup should put user data: the real data directory when packaged,
    the in-checkout locitize-data otherwise (source-checkout behavior unchanged)."""
    root = Path(code_dir) if code_dir is not None else Path(__file__).resolve().parent
    if bundled_python(root):
        from config import resolve_data_dir
        return resolve_data_dir(install_dir=root)
    return root / "locitize-data"


def ensure_platform_venv(code_dir=None):
    """Create the packaged app's platform venv (system-site-packages of the
    bundled runtime) if missing. Returns its python.exe, or None when not packaged."""
    import subprocess
    root = Path(code_dir) if code_dir is not None else Path(__file__).resolve().parent
    bundled = bundled_python(root)
    if not bundled:
        return None
    target = environment_root(root) / ".venv"
    interpreter = target / "Scripts" / "python.exe"
    if not interpreter.is_file():
        subprocess.run([str(bundled), "-B", "-E", "-s", "-m", "venv", "--system-site-packages", str(target)],
                       check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return interpreter
