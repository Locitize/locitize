"""First-run setup: machine detection and step execution (M15).

The executor half of the wizard. setup_plan.py decides WHAT to do; this module
finds out what the machine already has and then does it.

ONE HARD CONSTRAINT SHAPES THIS FILE: it runs on a bare `python` before anything
is installed, which is the whole point of a bootstrap. So it imports nothing but
the standard library at module scope - no yaml, no psutil, no PySide6, and not
config.py either (config imports yaml). Anything richer is probed lazily inside a
function and degrades to a best-effort answer when absent, exactly the way the
health probes report an unconfigured path instead of raising.

Detection is therefore deliberately pessimistic. It answers "is this definitely
already here?" and returns False when it cannot tell. A False costs the user a
re-install of something they had; a wrong True would leave a broken install
claiming to be finished, which is the failure this project does not accept.

Execution rules, inherited from the rest of the platform rather than invented here:

- every download goes through modelhub.download_verified, the one downloader in
  the tree: streamed to .partial, hashed while writing, compared before rename,
  deleted on mismatch, host allowlist re-checked on every redirect hop;
- fetched binaries land under <data root>/bin, never the install directory
  (write fence, DEC-M14-9);
- winget installs are user-scope, never elevated;
- no step ever runs without the caller having shown its size and taken consent.

ASCII only.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parent
from runtime_layout import environment_root
REPO_DIR = environment_root(BASE_DIR)

# Hosts the setup fetcher may reach, re-checked on every redirect hop by
# modelhub.open_checked. Deliberately separate from models_hub.allowed_hosts: a
# release binary and a model weight are different trust decisions, and widening
# one must not silently widen the other.
RELEASE_HOSTS: tuple[str, ...] = (
    "github.com",
    "api.github.com",
    "githubusercontent.com",  # objects.* and release-assets.* live under this
)

# Windows subprocess flag: never flash a console window from a GUI wizard. Mirrors
# the console-flash fix already applied across the platform (commit 5617bd6).
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class StepResult:
    """What one executed step actually did. `ok=False` is never a raised traceback."""

    key: str
    ok: bool
    message: str = ""
    skipped: bool = False


@dataclass
class Machine:
    """The facts detection gathered, and the paths the executor will write to."""

    venv_python: Path | None = None
    data_root: Path = field(default_factory=lambda: BASE_DIR / "locitize-data")
    has_nvidia: bool = False
    gpu_name: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def bin_dir(self) -> Path:
        return self.data_root / "bin"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def venv_python_path(repo_dir: Path | str | None = None) -> Path:
    """The interpreter the platform venv would expose. Not a claim that it exists."""
    root = Path(repo_dir) if repo_dir is not None else REPO_DIR
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _run(
    argv: Sequence[str],
    timeout: int = 60,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a command, never raise, never flash a console. Returns (code, output)."""
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            env=env,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def torch_sees_cuda(python_exe: Path | str) -> bool:
    """True when the venv's torch is a CUDA build that can see a device.

    torch's own answer, asked of the target interpreter: a CPU wheel says no,
    and so does a CUDA wheel on a machine whose driver is too old for it -
    which is exactly the case where the GPU step must be offered again.
    """
    exe = Path(python_exe)
    if not exe.is_file():
        return False
    code = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    rc, _ = _run([str(exe), "-c", code], timeout=90)
    return rc == 0


def modules_present(python_exe: Path | str, modules: Iterable[str]) -> bool:
    """True only when `python_exe` can import every module named.

    Asking the target interpreter is the only answer that is not a guess: the
    wizard's own interpreter is frequently NOT the venv it is building.
    """
    names = [m for m in modules if m]
    if not names:
        return True
    exe = Path(python_exe)
    if not exe.is_file():
        return False
    code = "import " + ", ".join(names)
    rc, _ = _run([str(exe), "-c", code], timeout=90)
    return rc == 0


def detect_nvidia() -> tuple[bool, str]:
    """Ask nvidia-smi whether a usable NVIDIA GPU is present.

    Reuses the platform's existing decision that nvidia-smi (shipped with the
    driver) is the GPU source of truth, so no new dependency appears here.
    """
    rc, out = _run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=20
    )
    if rc != 0:
        return False, ""
    first = (out or "").strip().splitlines()
    if not first or not first[0].strip():
        return False, ""
    return True, first[0].strip()


def _which(name: str) -> str:
    return shutil.which(name) or ""


def _settings_paths(data_root: Path) -> dict[str, str]:
    """Read paths.* out of the live settings.yaml, or {} when that is impossible.

    yaml is a dependency the bootstrap may not have yet, so this is best-effort by
    construction. A {} result makes detection pessimistic, which is the safe way
    to be wrong here.
    """
    try:
        import yaml  # noqa: PLC0415 - deliberately lazy; may not be installed yet
    except ImportError:
        return {}
    target = data_root / "settings.yaml"
    if not target.is_file():
        return {}
    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    paths = loaded.get("paths") if isinstance(loaded, dict) else None
    if not isinstance(paths, dict):
        return {}
    return {str(k): str(v or "") for k, v in paths.items()}


def _configured_file(paths: dict[str, str], key: str) -> bool:
    value = (paths.get(key) or "").strip()
    return bool(value) and Path(value).exists()


def find_llama_server(extra_dirs: Iterable[Path | str] = ()) -> str:
    """Locate an existing llama-server.exe without downloading one.

    Checked in order: PATH, then the directories a user most plausibly already
    unpacked it into. Finding one turns a 360 MB download into a config write.
    """
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    on_path = _which(exe)
    if on_path:
        return on_path

    # Every candidate is derived from the environment or from this checkout.
    # Hardcoding a drive-letter path here would ship one machine's disk layout to
    # every other machine, which is what scripts/verify_no_owner_paths.py exists
    # to refuse - it caught exactly that in the first draft of this function.
    candidates: list[Path] = [Path(d) for d in extra_dirs]
    home = Path.home()
    candidates.extend(
        [
            REPO_DIR / "locitize-data" / "bin",
            home / ".lmstudio" / "bin",
            home / "llama.cpp",
        ]
    )
    for env_key, tail in (
        ("LOCALAPPDATA", ("Programs", "llama.cpp")),
        ("LOCALAPPDATA", ("llama.cpp",)),
        ("ProgramFiles", ("llama.cpp",)),
        ("LOCITIZE_LLAMACPP_DIR", ()),
    ):
        root = (os.environ.get(env_key) or "").strip()
        if root:
            candidates.append(Path(root).joinpath(*tail))
    for folder in candidates:
        try:
            if not folder.is_dir():
                continue
        except OSError:
            continue
        direct = folder / exe
        if direct.is_file():
            return str(direct)
        try:
            for found in folder.rglob(exe):
                if found.is_file():
                    return str(found)
        except OSError:
            continue
    return ""


def detect_state(machine: Machine | None = None) -> tuple[dict[str, bool], Machine]:
    """Return (requirement_key -> already satisfied, the Machine facts used).

    Every unknown answers False. See the module docstring for why that direction
    is the only safe one.
    """
    info = machine or Machine()
    info.has_nvidia, info.gpu_name = detect_nvidia()

    venv_exe = venv_python_path()
    has_venv = venv_exe.is_file()
    info.venv_python = venv_exe if has_venv else None

    paths = _settings_paths(info.data_root)
    state: dict[str, bool] = {
        "platform_venv": has_venv,
        "base_pip": has_venv and modules_present(venv_exe, ("yaml", "psutil")),
        "gui_pip": has_venv and modules_present(venv_exe, ("PySide6",)),
        "pillow_pip": has_venv and modules_present(venv_exe, ("PIL",)),
        "kokoro_pip": has_venv and modules_present(venv_exe, ("torch", "kokoro")),
        "kokoro_pip_cuda": has_venv and torch_sees_cuda(venv_exe),
        "llama_cpp": _configured_file(paths, "llama_cpp") or bool(find_llama_server()),
        "whisper_bin": _configured_file(paths, "whisper"),
        "whisper_model": _configured_file(paths, "whisper_model"),
        "kokoro_weights": _configured_file(paths, "kokoro_model"),
        "vision_projector": False,
        "first_model": _has_registered_model(info.data_root),
        "node_runtime": bool(_which("node")),
        "harness_clis": any(
            _which(name) for name in ("claude", "codex", "opencode")
        ),
        "caddy": bool(_which("caddy")),
        "webui_venv": (REPO_DIR / ".webui-venv").is_dir(),
        "webui_pip": _webui_installed(),
        "finetune_venv": (REPO_DIR / "finetune-studio" / ".venv").is_dir(),
        "finetune_pip": False,
    }

    if not info.has_nvidia:
        info.notes.append(
            "No NVIDIA GPU detected. The CPU build of llama.cpp will be used; "
            "models will run, more slowly."
        )
    return state, info


def _webui_installed() -> bool:
    if os.name == "nt":
        return (REPO_DIR / ".webui-venv" / "Scripts" / "open-webui.exe").is_file()
    return (REPO_DIR / ".webui-venv" / "bin" / "open-webui").is_file()


def _has_registered_model(data_root: Path) -> bool:
    """True when models.yaml already names at least one model file that exists."""
    try:
        import yaml  # noqa: PLC0415 - lazy for the same reason as _settings_paths
    except ImportError:
        return False
    target = data_root / "models.yaml"
    if not target.is_file():
        return False
    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return False
    rows = loaded.get("models") if isinstance(loaded, dict) else None
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        location = str(row.get("location") or "").strip()
        if location and Path(location).is_file():
            return True
    return False


# ---------------------------------------------------------------------------
# Asset selection (pure)
# ---------------------------------------------------------------------------


def score_asset(name: str, want_cuda: bool) -> int:
    """Rank one llama.cpp release asset filename. Higher is better, <=0 rejects.

    Scored rather than pattern-matched on purpose: the upstream naming scheme has
    changed more than once (cu12.4 / cuda-12.4 / cuda), and a hardcoded pattern
    turns a routine upstream rename into a broken installer on a stranger's
    machine. Scoring degrades to "best available" instead of to nothing.
    """
    low = name.lower()
    if not low.endswith(".zip"):
        return 0
    if "win" not in low:
        return 0
    # Runtime-only bundles. Found by the first LIVE exercise of this fetch
    # (2026-08-29): upstream ships "cudart-llama-bin-win-cuda-12.4-x64.zip",
    # which contains CUDA runtime DLLs and NO llama-server, and it matched
    # every positive rule below. It is paired WITH a server build by the
    # installer, never chosen AS the build.
    if low.startswith("cudart"):
        return 0
    # Wrong architecture is a hard reject, not a low score.
    if any(tag in low for tag in ("arm64", "x86-32", "-x86.")):
        return 0
    # Vendor-specific builds we must not hand to an NVIDIA or CPU user.
    if any(
        tag in low
        for tag in ("hip", "rocm", "sycl", "vulkan", "musa", "cann",
                    "openvino", "opencl")
    ):
        return 0

    score = 10
    is_cuda = "cuda" in low or "cu1" in low or "cu2" in low
    if want_cuda and is_cuda:
        score += 100
        # Compatibility-first among CUDA builds: a cuda-13.x binary refuses to
        # run on a 12.x driver, while a cuda-12.x binary runs on both. Prefer
        # the LOWER major version rather than guessing the driver correctly.
        version = re.search(r"cuda-(\d+)", low)
        if version and int(version.group(1)) <= 12:
            score += 8
    elif want_cuda and not is_cuda:
        score += 5   # usable fallback: a CPU build still runs on a GPU machine
    elif not want_cuda and is_cuda:
        return 0     # never hand a CUDA build to a machine with no NVIDIA driver
    else:
        score += 50
        if "avx2" in low:
            score += 10
    if "x64" in low:
        score += 5
    return score


def choose_asset(names: Iterable[str], want_cuda: bool) -> str:
    """Pick the best-scoring asset name, or "" when none is usable."""
    best, best_score = "", 0
    for name in names:
        value = score_asset(str(name), want_cuda)
        if value > best_score:
            best, best_score = str(name), value
    return best


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

# Open WebUI is installed at a reviewed version, never "whatever is latest":
# it is separately licensed third-party software with a large dependency tree,
# and a new release reaches users only when this pin is bumped deliberately.
OPENWEBUI_VERSION = "0.11.4"

# Pip sets per requirement key. Kept here, next to the executor that installs
# them, rather than in requirements.txt: the wizard installs a SUBSET chosen by
# feature, which is the entire difference from `pip install -r requirements.txt`.
PIP_SETS: dict[str, tuple[str, ...]] = {
    "base_pip": ("PyYAML>=6.0", "psutil>=5.9"),
    "gui_pip": ("PySide6>=6.6,<7",),
    "pillow_pip": ("Pillow>=10.0",),
    # The CPU index and the exact pin are load-bearing; see requirements.txt for
    # the full reasoning. Reproduced faithfully rather than referenced, because a
    # resolver that silently picks the CUDA build costs a stranger gigabytes.
    "kokoro_pip": (
        "--extra-index-url", "https://download.pytorch.org/whl/cpu",
        "torch==2.13.0+cpu", "kokoro>=0.9.4",
    ),
    # The opt-in GPU build (owner request 2026-09-03: a Kokoro sentence took
    # 0.78s on CPU, 0.10s on the card). cu128 is the primary index so pip
    # cannot resolve the CPU wheel back in; PyPI stays reachable for kokoro
    # itself. The version differs from the CPU pin on purpose: the cu128 index
    # carries no 2.13 Windows wheel, cu126 predates Blackwell (sm_120) and
    # cu130 needs a 580+ driver - 2.11.0+cu128 is the one build that ran on the
    # reference RTX 5070 Ti with driver 576.88.
    "kokoro_pip_cuda": (
        "--index-url", "https://download.pytorch.org/whl/cu128",
        "--extra-index-url", "https://pypi.org/simple",
        "torch==2.11.0+cu128", "kokoro>=0.9.4",
    ),
    "webui_pip": (f"open-webui=={OPENWEBUI_VERSION}",),
    "finetune_pip": ("streamlit>=1.30", "pypdf>=4.0"),
}


def create_venv(target: Path | str, base_python: str | None = None) -> StepResult:
    """Create a virtual environment. Idempotent: an existing one is left alone."""
    path = Path(target)
    exe = (
        path / "Scripts" / "python.exe"
        if os.name == "nt"
        else path / "bin" / "python"
    )
    if exe.is_file():
        return StepResult("venv", True, f"already present at {path}", skipped=True)

    python = base_python or sys.executable
    rc, out = _run([python, "-m", "venv", str(path)], timeout=300)
    if rc != 0 or not exe.is_file():
        return StepResult("venv", False, f"could not create venv at {path}: {out[-400:]}")
    return StepResult("venv", True, f"created {path}")


def pip_install(
    python_exe: Path | str,
    packages: Sequence[str],
    on_output: Callable[[str], None] | None = None,
    timeout: int = 3600,
) -> StepResult:
    """Install `packages` into the interpreter at `python_exe`, streaming output.

    Streamed rather than captured: a 2.6 GB Open WebUI resolve is silent for
    minutes, and a progress window with nothing in it reads as a hang.
    """
    exe = Path(python_exe)
    if not exe.is_file():
        return StepResult("pip", False, f"interpreter not found: {exe}")
    argv = [str(exe), "-m", "pip", "install", "--disable-pip-version-check", *packages]
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return StepResult("pip", False, f"could not start pip: {exc}")

    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip()
        tail.append(text)
        del tail[:-40]
        if on_output:
            on_output(text)
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return StepResult("pip", False, "pip timed out")
    if code != 0:
        return StepResult("pip", False, "pip failed: " + " | ".join(tail[-6:]))
    return StepResult("pip", True, f"installed {len(packages)} requirement(s)")


def winget_install(package_id: str, timeout: int = 900) -> StepResult:
    """User-scope winget install. Never elevated; an existing package is a no-op."""
    if not _which("winget"):
        return StepResult(
            package_id,
            False,
            "winget is not available on this machine; install this one by hand",
        )
    rc, out = _run(
        [
            "winget", "install", "--id", package_id, "--exact",
            "--scope", "user", "--silent",
            "--accept-package-agreements", "--accept-source-agreements",
        ],
        timeout=timeout,
    )
    lowered = (out or "").lower()
    if rc == 0 or "already installed" in lowered:
        return StepResult(package_id, True, "installed")
    return StepResult(package_id, False, f"winget failed: {out[-300:].strip()}")


def unpack_zip(archive: Path | str, dest: Path | str) -> StepResult:
    """Extract `archive` into `dest`, refusing any member that escapes the root.

    zipfile.extractall does not protect against absolute or ../ member names on
    every Python version, and this archive arrived over the network, so the check
    is made here rather than assumed.
    """
    src, out_dir = Path(archive), Path(dest)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(src) as bundle:
            root = out_dir.resolve()
            for member in bundle.namelist():
                target = (out_dir / member).resolve()
                if root != target and root not in target.parents:
                    return StepResult(
                        "unpack", False, f"archive member escapes destination: {member}"
                    )
            bundle.extractall(out_dir)
    except (OSError, zipfile.BadZipFile) as exc:
        return StepResult("unpack", False, f"could not unpack {src.name}: {exc}")
    return StepResult("unpack", True, f"unpacked into {out_dir}")


@dataclass(frozen=True)
class ReleaseAsset:
    """One resolved, downloadable release file."""

    name: str
    url: str
    size: int = 0
    # "sha256:..." when the publisher declares one. Empty means the download can
    # only proceed on the explicit unverified rung, with the user having said so.
    digest: str = ""

    # Additional archives that must land in the same folder for the main one
    # to run (M15.6: the CUDA server build needs the separately-shipped cudart
    # DLL bundle of the SAME cuda version - upstream splits them to dedupe).
    companions: tuple["ReleaseAsset", ...] = ()

    @property
    def sha256(self) -> str:
        value = (self.digest or "").strip().lower()
        return value[7:] if value.startswith("sha256:") else ""


def _is_cudart_companion(name: str, chosen: str) -> bool:
    """True when `name` is the cudart bundle matching the chosen CUDA build."""
    low, chosen_low = name.lower(), chosen.lower()
    if not low.startswith("cudart") or "cuda" not in chosen_low:
        return False
    version = re.search(r"cuda-([\d.]+)", chosen_low)
    return bool(version) and f"cuda-{version.group(1)}" in low and "arm64" not in low


LLAMA_REPO = "ggml-org/llama.cpp"


def resolve_llama_release(
    want_cuda: bool,
    repo: str = LLAMA_REPO,
    tag: str = "",
    timeout: float = 30.0,
) -> tuple[ReleaseAsset | None, str]:
    """Ask GitHub which llama.cpp build fits this machine. Returns (asset, error).

    The asset is chosen by score_asset from the release's ACTUAL file list rather
    than by composing a filename from a template: upstream has renamed these more
    than once, and a template turns a routine rename into a broken installer on a
    stranger's machine.

    `tag` empty means the latest release. A pinned tag is honoured so a user (or a
    future settings key) can hold a known-good build.

    The publisher digest is used when GitHub declares one. When it does not, this
    returns the asset with an empty sha256 and the CALLER must obtain explicit
    unverified consent - download_verified refuses the unverified rung otherwise,
    which is the behaviour that keeps an unchecked binary from landing quietly.
    """
    try:
        import json  # noqa: PLC0415 - stdlib-only at module scope by design
        import modelhub  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - modelhub is in-tree
        return None, f"downloader unavailable: {exc}"

    # NOT /releases/latest: the live exercise of 2026-08-29 found that
    # endpoint pointing at a "v0.3.0" release carrying no Windows builds at
    # all, while the b-numbered builds sat one page-listing away. The list is
    # walked newest-first and the first release with a usable asset wins.
    if tag:
        url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    else:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=10"
    opener, handler = modelhub.make_opener(RELEASE_HOSTS)
    try:
        with modelhub.open_checked(
            opener, url, RELEASE_HOSTS, timeout=timeout, handler=handler
        ) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001 - any network failure is one message
        return None, f"could not reach the llama.cpp release list: {exc}"

    releases = [payload] if isinstance(payload, dict) else list(payload or [])
    last_tag = ""
    for release in releases:
        if not isinstance(release, dict):
            continue
        last_tag = str(release.get("tag_name", "")) or last_tag
        assets = release.get("assets")
        if not isinstance(assets, list) or not assets:
            continue
        by_name = {str(a.get("name", "")): a for a in assets if isinstance(a, dict)}
        picked = choose_asset(by_name.keys(), want_cuda)
        if not picked:
            continue
        row = by_name[picked]
        return (
            ReleaseAsset(
                name=picked,
                url=str(row.get("browser_download_url", "")),
                size=int(row.get("size") or 0),
                digest=str(row.get("digest") or ""),
                companions=tuple(
                    ReleaseAsset(
                        name=str(c.get("name", "")),
                        url=str(c.get("browser_download_url", "")),
                        size=int(c.get("size") or 0),
                        digest=str(c.get("digest") or ""),
                    )
                    for c in assets
                    if isinstance(c, dict)
                    and _is_cudart_companion(str(c.get("name", "")), picked)
                ),
            ),
            "",
        )
    kind = "CUDA" if want_cuda else "CPU"
    return None, (
        f"no Windows {kind} build in the last {len(releases)} release(s) "
        f"(newest checked: {last_tag or 'none'})"
    )


def install_llama_cpp(
    bin_dir: Path | str,
    want_cuda: bool,
    *,
    confirm_unverified: bool = False,
    say: Callable[[str], None] | None = None,
    tag: str = "",
) -> tuple[StepResult, str]:
    """Fetch, verify, and unpack llama.cpp. Returns (result, llama-server path).

    Idempotent: an existing llama-server under `bin_dir` short-circuits the whole
    thing, so re-running setup after a partial install costs one directory scan.
    """
    note = say or (lambda _text: None)
    target = Path(bin_dir) / "llama.cpp"
    existing = find_llama_server([target])
    if existing:
        return StepResult("llama_cpp", True, f"already present: {existing}", skipped=True), existing

    asset, error = resolve_llama_release(want_cuda, tag=tag)
    if asset is None:
        return StepResult("llama_cpp", False, error), ""

    if not asset.sha256 and not confirm_unverified:
        return (
            StepResult(
                "llama_cpp",
                False,
                f"{asset.name} publishes no checksum; re-run with the unverified "
                "download confirmed if you accept that",
            ),
            "",
        )

    note(f"   {asset.name} ({asset.size // (1024 * 1024)} MB)")
    target.mkdir(parents=True, exist_ok=True)
    archive = target / asset.name
    fetched = fetch_release_zip(
        asset.url,
        archive,
        asset.sha256 or None,
        confirm_unverified=confirm_unverified and not asset.sha256,
    )
    if not fetched.ok:
        return StepResult("llama_cpp", False, fetched.message), ""

    unpacked = unpack_zip(archive, target)
    if not unpacked.ok:
        return StepResult("llama_cpp", False, unpacked.message), ""
    try:
        archive.unlink()
    except OSError:
        pass

    # The CUDA server build cannot start without its matching cudart DLLs,
    # which upstream ships as a separate archive. Fetch them into the SAME
    # folder; a failure here fails the whole step honestly rather than leaving
    # a server binary that dies with a missing-DLL dialog on first launch.
    for companion in asset.companions:
        note(f"   companion {companion.name} ({companion.size // (1024 * 1024)} MB)")
        companion_archive = target / companion.name
        fetched_companion = fetch_release_zip(
            companion.url,
            companion_archive,
            companion.sha256 or None,
            confirm_unverified=confirm_unverified and not companion.sha256,
        )
        if not fetched_companion.ok:
            return (
                StepResult(
                    "llama_cpp",
                    False,
                    f"companion {companion.name}: {fetched_companion.message}",
                ),
                "",
            )
        unpacked_companion = unpack_zip(companion_archive, target)
        if not unpacked_companion.ok:
            return StepResult("llama_cpp", False, unpacked_companion.message), ""
        try:
            companion_archive.unlink()
        except OSError:
            pass

    server = find_llama_server([target])
    if not server:
        return (
            StepResult("llama_cpp", False, "unpacked archive contained no llama-server"),
            "",
        )
    verified = "publisher checksum" if asset.sha256 else "size only (unverified)"
    return StepResult("llama_cpp", True, f"installed ({verified})"), server


_WRITE_PATHS_SNIPPET = """
import sys, pathlib, yaml
data = pathlib.Path(sys.argv[1])
updates = dict(pair.split("=", 1) for pair in sys.argv[2:])
target = data / "settings.yaml"
loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
paths = loaded.setdefault("paths", {})
changed = [k for k, v in updates.items() if str(paths.get(k) or "") != v]
paths.update(updates)
target.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")
print("updated:", ",".join(changed) if changed else "(no change)")
"""


def write_settings_paths(
    python_exe: Path | str, data_root: Path | str, updates: dict[str, str]
) -> StepResult:
    """Persist discovered binary paths into the user's live settings.yaml.

    Run through the VENV interpreter, not this one: the bootstrap process may have
    no yaml, and hand-editing YAML with string surgery is how a user's hand-
    maintained config file gets corrupted. A round-trip through safe_load/safe_dump
    is the same treatment models.yaml already gets from the registry writer.

    This is the step that closes the gap the wizard exists for: a downloaded
    llama-server is useless until paths.llama_cpp names it.
    """
    if not updates:
        return StepResult("settings", True, "no paths to write", skipped=True)
    exe = Path(python_exe)
    if not exe.is_file():
        return StepResult("settings", False, f"interpreter not found: {exe}")
    target = Path(data_root) / "settings.yaml"
    if not target.is_file():
        return StepResult("settings", False, f"no settings.yaml at {target}")

    argv = [str(exe), "-c", _WRITE_PATHS_SNIPPET, str(data_root)]
    argv.extend(f"{key}={value}" for key, value in updates.items())
    rc, out = _run(argv, timeout=60)
    if rc != 0:
        return StepResult("settings", False, f"could not write settings.yaml: {out[-300:]}")
    return StepResult("settings", True, out.strip() or "settings.yaml updated")


def fetch_release_zip(
    url: str,
    dest: Path | str,
    expected_sha256: str | None,
    *,
    confirm_unverified: bool = False,
    progress: Callable[[Any], None] | None = None,
    allowed_hosts: Sequence[str] = RELEASE_HOSTS,
) -> StepResult:
    """Download one release archive through the platform's single verified path.

    Delegates to modelhub.download_verified rather than opening a socket here:
    that function owns the streamed-hash / verify-before-rename / delete-on-
    mismatch contract and the per-hop host allowlist, and a second downloader in
    this tree is exactly the duplication the M14 work removed.

    require_gguf_magic is False because this is a zip, not a GGUF; every other
    check the model path performs still applies.
    """
    try:
        import modelhub  # noqa: PLC0415 - stdlib-only at module scope by design
    except ImportError as exc:
        return StepResult("fetch", False, f"downloader unavailable: {exc}")

    outcome = modelhub.download_verified(
        url,
        dest,
        expected_sha256,
        allowed_hosts=list(allowed_hosts),
        require_gguf_magic=False,
        confirm_unverified=confirm_unverified,
        progress=progress,
    )
    if not outcome.ok:
        return StepResult("fetch", False, outcome.error or "download failed")
    return StepResult("fetch", True, f"downloaded {outcome.bytes_written} bytes")


# ---------------------------------------------------------------------------
# Weight and speech installers (M15.6)
# ---------------------------------------------------------------------------
# Sources are pinned by (repo, path) only. Digests are NEVER stored here - they
# are resolved live from the HuggingFace tree at install time (lfs.oid, rung
# V-API), the same catalog rule model_catalog.json documents: a stale stored
# hash is worse than no hash. Every source below was verified live 2026-08-29:
# repo exists, file exists at the recorded path, digest present.

WHISPER_WEIGHTS = ("ggerganov/whisper.cpp", "ggml-base.en.bin")
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_CHECKPOINT = "kokoro-v1_0.pth"
KOKORO_VOICES = (
    "voices/am_michael.pt",
    "voices/af_bella.pt",
    "voices/am_adam.pt",
    "voices/bf_emma.pt",
)
# Owner decision 2026-08-29: LOCITIZE ships NO model names. There used to be a
# pinned vision model pair here; which chat or vision model a person runs is
# their choice, made from their own disk (find_local_models) or their own
# search. Whisper and Kokoro weights above are feature INFRASTRUCTURE (the
# speech-to-text and text-to-speech engines' own files), not model taste, and
# stay pinned.


def hf_tree_digest(repo: str, path: str, timeout: float = 60.0) -> tuple[str, int]:
    """Return (sha256, size) for one file, resolved LIVE from the HF tree.

    Empty sha256 means the API declared none; the caller must then take the
    explicit unverified rung or refuse - never silently continue.
    """
    try:
        import json  # noqa: PLC0415 - stdlib-only at module scope by design
        import modelhub  # noqa: PLC0415
    except ImportError:
        return "", 0
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    opener, handler = modelhub.make_opener(modelhub.DEFAULT_ALLOWED_HOSTS)
    try:
        with modelhub.open_checked(
            opener, url, modelhub.DEFAULT_ALLOWED_HOSTS, timeout=timeout, handler=handler
        ) as response:
            rows = json.loads(response.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - the caller shows one honest message
        return "", 0
    for row in rows or []:
        if str(row.get("path", "")) == path:
            lfs = row.get("lfs") if isinstance(row.get("lfs"), dict) else {}
            digest = modelhub.normalize_etag_digest(lfs.get("oid")) or ""
            size = int(lfs.get("size") or row.get("size") or 0)
            return digest, size
    return "", 0


def fetch_hf_file(
    repo: str,
    path: str,
    dest: Path | str,
    say: Callable[[str], None] | None = None,
) -> StepResult:
    """One HuggingFace file through the platform's single verified path.

    Refuses when the tree declares no digest: every source this module pins
    carried one at verification time, so a missing digest means the repo
    changed and a human should look, not that the download should proceed.
    """
    note = say or (lambda _t: None)
    try:
        import modelhub  # noqa: PLC0415
    except ImportError as exc:
        return StepResult("fetch", False, f"downloader unavailable: {exc}")
    target = Path(dest)
    digest, size = hf_tree_digest(repo, path)
    if target.is_file() and size and target.stat().st_size == size:
        return StepResult("fetch", True, f"already present: {target.name}", skipped=True)
    if not digest:
        return StepResult(
            "fetch",
            False,
            f"{repo}/{path} no longer declares a checksum; refusing to fetch it "
            f"unverified - check the repository",
        )
    note(f"   {path} ({size // (1024 * 1024)} MB)")
    target.parent.mkdir(parents=True, exist_ok=True)
    outcome = modelhub.download_verified(
        f"https://huggingface.co/{repo}/resolve/main/{path}",
        target,
        digest,
        verification=modelhub.V_API,
        expected_size=size or None,
        max_bytes=size or None,
        require_gguf_magic=path.lower().endswith(".gguf"),
    )
    if not outcome.ok:
        return StepResult("fetch", False, outcome.error or "download failed")
    return StepResult("fetch", True, f"verified {target.name}")


def install_whisper(
    bin_dir: Path | str,
    say: Callable[[str], None] | None = None,
    confirm_unverified: bool = False,
) -> tuple[StepResult, dict[str, str]]:
    """whisper.cpp server binary from GitHub releases. Returns (result, paths).

    Walks the release list newest-first exactly like the llama.cpp fetch (the
    newest whisper.cpp tag, v1.9.3, ships no binaries at all - verified live
    2026-08-29). The plain x64 build is chosen over the BLAS/CUDA variants:
    speech-to-text of one microphone is not the bottleneck, and the plain build
    has no extra DLL dependencies to go wrong on a fresh machine.
    """
    note = say or (lambda _t: None)
    try:
        import json  # noqa: PLC0415
        import modelhub  # noqa: PLC0415
    except ImportError as exc:
        return StepResult("whisper_bin", False, f"downloader unavailable: {exc}"), {}
    target = Path(bin_dir) / "whisper.cpp"
    existing = _find_in(target, "whisper-server.exe")
    if existing:
        return (
            StepResult("whisper_bin", True, f"already present: {existing}", skipped=True),
            {"whisper": existing},
        )
    url = "https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=10"
    opener, handler = modelhub.make_opener(RELEASE_HOSTS)
    try:
        with modelhub.open_checked(
            opener, url, RELEASE_HOSTS, timeout=30, handler=handler
        ) as response:
            releases = json.loads(response.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        return StepResult("whisper_bin", False, f"could not reach releases: {exc}"), {}
    chosen = None
    for release in releases or []:
        for asset in release.get("assets", []) or []:
            if str(asset.get("name", "")).lower() == "whisper-bin-x64.zip":
                chosen = asset
                break
        if chosen:
            break
    if chosen is None:
        return StepResult(
            "whisper_bin", False,
            "no whisper-bin-x64.zip in the last 10 releases; install by hand",
        ), {}
    digest = str(chosen.get("digest") or "")
    sha = digest[7:] if digest.lower().startswith("sha256:") else ""
    note(f"   {chosen['name']} ({int(chosen.get('size') or 0) // (1024 * 1024)} MB)")
    target.mkdir(parents=True, exist_ok=True)
    archive = target / str(chosen["name"])
    fetched = fetch_release_zip(
        str(chosen.get("browser_download_url", "")), archive, sha or None,
        # No published digest means no check at all: that needs the user's
        # explicit consent, exactly like llama.cpp - never implied.
        confirm_unverified=confirm_unverified,
    )
    if not fetched.ok:
        return StepResult("whisper_bin", False, fetched.message), {}
    unpacked = unpack_zip(archive, target)
    if not unpacked.ok:
        return StepResult("whisper_bin", False, unpacked.message), {}
    try:
        archive.unlink()
    except OSError:
        pass
    server = _find_in(target, "whisper-server.exe")
    if not server:
        return StepResult(
            "whisper_bin", False, "archive contained no whisper-server.exe"
        ), {}
    paths = {"whisper": server}
    stream = _find_in(target, "whisper-stream.exe") or _find_in(target, "stream.exe")
    if stream:
        paths["whisper_stream"] = stream
    else:
        note("   (no whisper-stream in this build; live microphone mode stays off)")
    return StepResult("whisper_bin", True, f"installed {Path(server).name}"), paths


def install_whisper_weights(
    weights_dir: Path | str, say: Callable[[str], None] | None = None
) -> tuple[StepResult, dict[str, str]]:
    """The ggml speech model, digest-verified from the HF tree."""
    dest = Path(weights_dir) / WHISPER_WEIGHTS[1]
    result = fetch_hf_file(WHISPER_WEIGHTS[0], WHISPER_WEIGHTS[1], dest, say)
    return (
        StepResult("whisper_model", result.ok, result.message, result.skipped),
        {"whisper_model": str(dest)} if result.ok else {},
    )


def install_kokoro_weights(
    kokoro_dir: Path | str, say: Callable[[str], None] | None = None
) -> tuple[StepResult, dict[str, str]]:
    """Kokoro checkpoint plus the four documented voices, each digest-verified.

    All-or-nothing on the checkpoint, best-effort on voices: a missing voice
    degrades to fewer choices in the audition list, but a missing checkpoint is
    no TTS at all, so only that failure fails the step.
    """
    root = Path(kokoro_dir)
    checkpoint = root / KOKORO_CHECKPOINT
    result = fetch_hf_file(KOKORO_REPO, KOKORO_CHECKPOINT, checkpoint, say)
    if not result.ok:
        return StepResult("kokoro_weights", False, result.message), {}
    voices_dir = root / "voices"
    fetched = 0
    for voice in KOKORO_VOICES:
        voice_result = fetch_hf_file(KOKORO_REPO, voice, root / voice, say)
        if voice_result.ok:
            fetched += 1
    if fetched == 0:
        return StepResult(
            "kokoro_weights", False,
            "checkpoint fetched but no voice file could be verified",
        ), {}
    return (
        StepResult(
            "kokoro_weights", True,
            f"checkpoint + {fetched}/{len(KOKORO_VOICES)} voices",
        ),
        {"kokoro_model": str(checkpoint), "kokoro_voices": str(voices_dir)},
    )


def _find_in(folder: Path, exe: str) -> str:
    try:
        if not folder.is_dir():
            return ""
        direct = folder / exe
        if direct.is_file():
            return str(direct)
        for found in folder.rglob(exe):
            if found.is_file():
                return str(found)
    except OSError:
        pass
    return ""


_REGISTER_SNIPPET = """
import sys, json
import config
spec = json.loads(sys.argv[1])
config.append_model_entry(
    spec["data_root"], spec["model_id"], spec["name"], spec["location"],
    description=spec.get("description", ""),
    context_size=int(spec.get("context_size", 8192)),
    gpu_layers=int(spec.get("gpu_layers", 999)),
    quantization=spec.get("quantization", ""),
    notes=spec.get("notes", ""),
    sha256=spec.get("sha256", ""),
    mmproj=spec.get("mmproj", ""),
)
# M18.15 (owner report: every model showed plain "Chat"): fill the true
# capabilities from the model's own GGUF right at registration, so the
# Capabilities column is honest from the first paint. Best-effort by rule:
# a detection failure must never fail the registration it decorates.
try:
    import gguf_meta
    caps = gguf_meta.detect_capabilities(spec["location"])
    # Same rule as launcher --detect-capabilities: "vision" only with a
    # projector on the row, so the flag never overpromises.
    if "vision" in caps and not (spec.get("mmproj") or "").strip():
        caps = [c for c in caps if c != "vision"]
    if caps:
        config.write_model_capabilities(spec["data_root"], spec["model_id"], caps)
except Exception:
    pass
print("registered", spec["model_id"])
"""


_SEED_SNIPPET = """
import sys
import config
data, install = sys.argv[1], sys.argv[2]
actions = config.ensure_user_config(data, install)
print("seeded:", ",".join(a.kind for a in actions))
"""


_ENABLE_OPENWEBUI_SNIPPET = """
import sys
import config
config.write_openwebui_enabled(sys.argv[1], True)
print("openwebui enabled")
"""


def enable_openwebui_via_venv(
    python_exe: Path | str, data_root: Path | str
) -> StepResult:
    """Flip openwebui.enabled true in settings.yaml, in the venv interpreter.

    Installing Open WebUI's venv + package does not make the chat chooser use it:
    webui_available() also checks openwebui.enabled, which ships false (opt-in).
    So when the user selects the Open WebUI feature, setup must enable it, or a
    freshly installed Open WebUI is never reached and chat stays on the built-in
    llama.cpp UI. Runs in the venv because config imports yaml, which the wizard's
    bootstrap interpreter may lack.
    """
    exe = Path(python_exe)
    if not exe.is_file():
        return StepResult("openwebui_enable", False, f"interpreter not found: {exe}")
    rc, out = _run(
        [str(exe), "-c", _ENABLE_OPENWEBUI_SNIPPET, str(data_root)],
        timeout=30,
        cwd=BASE_DIR,
    )
    if rc != 0:
        return StepResult(
            "openwebui_enable", False, f"could not enable Open WebUI: {out[-200:]}"
        )
    return StepResult("openwebui_enable", True, out.strip() or "Open WebUI enabled")


def create_desktop_shortcut() -> StepResult:
    """Create (or refresh) a Desktop shortcut to LOCITIZE.vbs (M17.12).

    LOCITIZE.vbs is the official no-console launcher; a Desktop shortcut to it
    means the everyday way in never flashes a black console. Built via PowerShell
    + WScript.Shell.CreateShortcut (no third-party dependency). The shortcut
    targets wscript.exe with the .vbs as its argument - the robust form - so it
    launches exactly as a double-click on the .vbs would. Best-effort: any
    failure (no PowerShell, a locked Desktop) is a non-fatal skip, never a setup
    failure, because a missing shortcut costs nothing (LOCITIZE.vbs still works).
    """
    vbs = BASE_DIR / "LOCITIZE.vbs"
    from runtime_layout import bundled_python
    native = BASE_DIR.parent / "LOCITIZE.exe"
    packaged = bundled_python(BASE_DIR) is not None and native.is_file()
    if not packaged and not vbs.is_file():
        return StepResult(
            "shortcut", True, "no LOCITIZE.vbs to link (skipped)", skipped=True
        )
    # Single-quoted PS strings: backslashes are literal, so a Windows path needs
    # no escaping; a stray single quote is doubled to stay inside the literal.
    vbs_ps = str(vbs).replace("'", "''")
    dir_ps = str(BASE_DIR).replace("'", "''")
    # The launcher argument is the .vbs path wrapped in double quotes (so a path
    # with spaces survives), and that whole thing lives inside a PS '...' literal.
    quoted_vbs = '"' + vbs_ps + '"'
    target_lines = "$s.TargetPath = 'wscript.exe'; $s.Arguments = '" + quoted_vbs + "'; "
    if packaged:
        target_lines = "$s.TargetPath = '" + str(native).replace("'", "''") + "'; $s.Arguments = ''; "
        dir_ps = str(native.parent).replace("'", "''")
    # M18.3: the shortcut carries LOCITIZE's own icon (the multi-res .ico that
    # ships in assets), not wscript.exe's - the first thing a new user sees on
    # their Desktop is the product's mark. Degrades to the default icon if the
    # asset is ever absent (IconLocation is only set when the file exists).
    ico = BASE_DIR / "assets" / "locitize_launcher.ico"
    icon_line = ""
    if ico.is_file():
        ico_ps = str(ico).replace("'", "''")
        icon_line = "$s.IconLocation = '" + ico_ps + ",0'; "
    script = (
        "$w = New-Object -ComObject WScript.Shell; "
        "$lnk = Join-Path $w.SpecialFolders('Desktop') 'LOCITIZE.lnk'; "
        "$s = $w.CreateShortcut($lnk); "
        + target_lines +
        "$s.WorkingDirectory = '" + dir_ps + "'; "
        + icon_line +
        "$s.Description = 'Launch LOCITIZE'; "
        "$s.Save()"
    )
    rc, out = _run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        timeout=30,
    )
    if rc != 0:
        return StepResult(
            "shortcut", False,
            f"could not create desktop shortcut (LOCITIZE.vbs still works): "
            f"{out[-160:].strip()}",
        )
    return StepResult("shortcut", True, "desktop shortcut to LOCITIZE created")


def seed_data_root(
    python_exe: Path | str, data_root: Path | str, install_dir: Path | str
) -> StepResult:
    """Create the data root's settings.yaml + models.yaml before anything writes.

    config.ensure_user_config copies the shipped *.default.yaml templates into
    the data root (non-destructive: it never overwrites a live file). It must
    run BEFORE the wizard registers a model or writes a discovered path -
    append_model_entry needs a models.yaml to append to, and write_settings_paths
    needs a settings.yaml to edit. Runs in the VENV interpreter because config
    imports yaml, which the bootstrap python may lack; the venv has it by the
    time any model step runs (base_pip is installed first).
    """
    exe = Path(python_exe)
    if not exe.is_file():
        return StepResult("seed", False, f"interpreter not found: {exe}")
    rc, out = _run(
        [str(exe), "-c", _SEED_SNIPPET, str(data_root), str(install_dir)],
        timeout=60,
        cwd=BASE_DIR,
    )
    if rc != 0:
        return StepResult("seed", False, f"could not seed config: {out[-300:]}")
    return StepResult("seed", True, out.strip() or "config seeded")


def register_model_via_venv(
    python_exe: Path | str, spec: dict[str, str]
) -> StepResult:
    """Register a downloaded model through config.append_model_entry.

    Run in the VENV interpreter for the same reason write_settings_paths is:
    the wizard's own interpreter has no yaml, and models.yaml only ever gets
    written through config.py's chokepoint - the wizard is not a second writer.
    """
    import json  # noqa: PLC0415

    exe = Path(python_exe)
    if not exe.is_file():
        return StepResult("register", False, f"interpreter not found: {exe}")
    rc, out = _run(
        [str(exe), "-c", _REGISTER_SNIPPET, json.dumps(spec)],
        timeout=120,
        cwd=BASE_DIR,
    )
    if rc != 0:
        return StepResult("register", False, f"could not register: {out[-300:]}")
    return StepResult("register", True, out.strip() or "registered")


def measure_contexts(
    python_exe: "Path | str",
    say: "Callable[[str], None] | None" = None,
    budget_s: int = 24 * 3600,
) -> StepResult:
    """Measure each registered model's real context ceiling and apply it.

    Delegates to scripts/measure_context_ceilings.py - the MEASURED finder that
    probes real llama-server throughput at each context rung and writes only what
    it measured, never a computed guess. (The project already learned that a
    KV-arithmetic estimate mispredicts across architectures; that is why this
    measures instead.) Runs in the venv because it needs config + the server
    controller, streaming each probe line so the wizard shows live progress
    rather than a silent multi-minute pause.

    Resumable: the finder persists its state after every probe, so an interrupted
    run continues where it stopped. The large default budget lets a first run
    measure every model in one pass; that number is only a resumable-stop guard -
    nothing about the chosen context is hardcoded, each is measured on THIS
    machine and GPU. Degrades honestly: no GPU, no llama.cpp, or no models leaves
    every model at its safe default and returns a non-fatal result.
    """
    emit = say or (lambda _t: None)
    exe = Path(python_exe)
    if not exe.is_file():
        return StepResult("measure", False, f"interpreter not found: {exe}")
    script = BASE_DIR / "scripts" / "measure_context_ceilings.py"
    if not script.is_file():
        return StepResult(
            "measure", True,
            "measured tuner not present; models keep their safe defaults",
            skipped=True,
        )
    try:
        proc = subprocess.Popen(
            [str(exe), str(script), "--apply", "--budget", str(budget_s)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=BASE_DIR,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return StepResult("measure", False, f"could not start the measured tuner: {exc}")

    tail: list[str] = []
    applied = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip()
        tail.append(text)
        del tail[:-60]
        if text.strip():
            emit("   " + text)
        if text.strip().startswith("APPLIED"):
            applied += 1
    code = proc.wait()
    if code != 0:
        return StepResult(
            "measure", False,
            "measured tuner could not finish (models keep safe defaults): "
            + " | ".join(t for t in tail[-4:] if t.strip()),
        )
    if applied == 0:
        return StepResult(
            "measure", True,
            "no context change measured; models keep their current values",
            skipped=True,
        )
    return StepResult("measure", True, f"measured and applied context for {applied} model(s)")


# ---------------------------------------------------------------------------
# Local model discovery (M15.11)
# ---------------------------------------------------------------------------
# Owner decision 2026-08-29: the wizard suggests no model names. Instead it
# finds the GGUFs a person already has - most people trying a local-AI app
# have some - and imports them. Nothing here is copied: a file on the same
# volume is HARDLINKED into LOCITIZE's models folder (zero extra bytes, and
# deleting either name leaves the other intact); a file on another volume is
# registered where it lies. Both mirror the approach the owner's own registry
# was built with.

DISCOVERY_MIN_BYTES = 100 * 1024 * 1024  # excludes tokenizer fixtures and stubs


# The conventional shared store lives at the root of the SYSTEM drive, in
# an AI folder's Models subfolder. Assembled from the environment plus
# separate name components on purpose: the shipped source carries no
# machine-specific path literal (verify_no_owner_paths gates on that),
# yet virtually every Windows machine resolves this to the same place.
MODEL_STORE_DEFAULT = str(
    Path((os.environ.get("SystemDrive") or "C:") + os.sep) / "AI" / "Models"
)


def model_store_path() -> Path:
    """The shared, tool-agnostic model store (M22).

    One folder every local-AI tool on the machine can feed from - LOCITIZE
    creates it at setup, scans it for imports, and links finished
    downloads into it. LOCITIZE_MODEL_STORE overrides the default
    location; the default is deliberately outside any one app's tree so
    no uninstall ever takes the user's models with it."""
    override = (os.environ.get("LOCITIZE_MODEL_STORE") or "").strip()
    return Path(override) if override else Path(MODEL_STORE_DEFAULT)


def ensure_model_store(say: Callable[[str], None] | None = None) -> StepResult:
    """Create the shared model store if missing (M22). Best-effort.

    Drops a one-paragraph README into a store it CREATES so a user who
    finds the folder knows what it is and what to put there. An existing
    folder is left byte-for-byte alone."""
    note = say or (lambda _t: None)
    store = model_store_path()
    try:
        if store.is_dir():
            return StepResult("model_store", True, f"model store present: {store}", skipped=True)
        store.mkdir(parents=True, exist_ok=True)
        readme = store / "README.txt"
        if not readme.exists():
            readme.write_text(
                "This folder is your machine's shared model store, created by "
                "LOCITIZE.\n\nPut GGUF model files here and every local-AI tool "
                "on this machine can use them from one place. LOCITIZE scans "
                "this folder when importing models and links its downloads "
                "into it (links share bytes - nothing is ever duplicated).\n",
                encoding="utf-8",
            )
    except OSError as exc:
        return StepResult("model_store", False, f"could not create {store}: {exc}")
    note(f"   created model store: {store}")
    return StepResult("model_store", True, f"created model store: {store}")


def discovery_roots() -> list[Path]:
    """Where local models plausibly already live. Environment-derived only."""
    home = Path.home()
    roots = [
        model_store_path(),  # M22: the shared store scans first
        home / "Downloads",
        home / ".lmstudio" / "models",
        home / ".cache" / "huggingface" / "hub",
        home / ".ollama" / "models",
        home / "Documents",
    ]
    local = (os.environ.get("LOCALAPPDATA") or "").strip()
    if local:
        roots.append(Path(local) / "nomic.ai")  # GPT4All's default store
    return [r for r in roots if r.is_dir()]


def pair_mmproj(model_path: Path | str, allow_generic: bool = True) -> str:
    """The projector file that belongs to `model_path`, or "" (M18.18).

    Vision GGUFs ship as a PAIR - the language model plus an mmproj projector -
    and people keep them side by side (LM Studio's per-model folders, a
    download directory, an HF snapshot). The scan skips projectors as models;
    this finds the one that belongs to a given model so import can carry it.

    Pairing rules, strict to avoid a wrong pair (worse than none):
    1. Only files in the SAME directory as the model are candidates.
    2. A candidate whose name starts with the model's stem wins outright
       (LM Studio's "<model>.mmproj-Q8_0.gguf" convention).
    3. A generically-named candidate (e.g. "mmproj-F16.gguf") is accepted only
       when it is the directory's ONLY projector AND the model's own GGUF
       header says it is a vision architecture - the header is the authority,
       so a text model never gets a projector bolted on.
    """
    try:
        model = Path(model_path)
        directory = model.parent
        stem = model.stem.lower().replace(".gguf", "")
        family = stem.split(".")[0][:12]
        # Same dir first; then SIBLING dirs that share the family name prefix
        # (LM Studio splits "Model.Q6_K/" and "Model/" into two folders, with
        # the projector in the plain one). Siblings only count for the strict
        # stem-match rule below - never the generic single-candidate rule.
        search_dirs = [directory]
        try:
            for sibling in directory.parent.iterdir():
                if (sibling.is_dir() and sibling != directory
                        and sibling.name.lower().startswith(family)):
                    search_dirs.append(sibling)
        except OSError:
            pass
        candidates = []
        own_dir_candidates = []
        for d in search_dirs:
            for p in d.glob("*.gguf"):
                if "mmproj" in p.name.lower() and p.is_file():
                    candidates.append(p)
                    if d == directory:
                        own_dir_candidates.append(p)
    except OSError:
        return ""
    if not candidates:
        return ""
    model_stem = model.name.lower()
    for candidate in candidates:
        # Strict stem rule: the candidate must be "<base>.mmproj-*" (or
        # "<base>-mmproj-*") where <base> is a real prefix of the model's own
        # file name. "qwen3" alone must never match "Qwen3-VL-..."'s projector.
        cname = candidate.name.lower()
        for marker in (".mmproj", "-mmproj"):
            base = cname.split(marker)[0]
            if base != cname and len(base) >= 6 and model_stem.startswith(base):
                return str(candidate)
    if allow_generic and len(own_dir_candidates) == 1:
        try:
            import gguf_meta  # noqa: PLC0415 - stdlib-only, lazy like the rest

            header = gguf_meta.read_gguf_header(str(model))
            arch = (header.architecture or "").lower()
            if any(tag in arch for tag in ("vl", "vision", "clip", "mllama", "gemma3", "gemma4")):
                return str(own_dir_candidates[0])
        except Exception:  # noqa: BLE001 - unsure means no pair, never a guess
            return ""
    return ""


def install_claude_local(say: Callable[[str], None] | None = None) -> StepResult:
    """Put the claude-local shim beside the claude CLI (M21).

    claude-local runs Claude Code against the model LOCITIZE is serving: it
    finds the running llama-server on the loopback port range, reads the
    loaded model's id and REAL context size from the server itself, and
    launches claude with the Anthropic overrides scoped to that single
    process - plain `claude` elsewhere still talks to Anthropic. The shim
    lands in the directory that already holds claude.exe (which is on PATH
    by construction), so no PATH edit, no settings.json change, nothing
    global. Honest no-op when the claude CLI is not installed.
    """
    note = say or (lambda _t: None)
    claude = shutil.which("claude")
    if not claude:
        return StepResult(
            "claude_local", True,
            "claude CLI not installed - claude-local shim skipped",
            skipped=True,
        )
    target_dir = Path(claude).resolve().parent
    source_dir = Path(__file__).resolve().parent / "scripts"
    try:
        for name in ("claude-local.cmd", "claude-local.ps1"):
            source = source_dir / name
            if not source.is_file():
                return StepResult(
                    "claude_local", False, f"shim source missing: {source}"
                )
            shutil.copyfile(source, target_dir / name)
    except OSError as exc:
        return StepResult(
            "claude_local", False, f"could not install claude-local: {exc}"
        )
    note(f"   claude-local installed to {target_dir}")
    return StepResult("claude_local", True, f"claude-local -> {target_dir}")


def find_local_models(
    roots: Iterable[Path | str] | None = None,
    say: Callable[[str], None] | None = None,
) -> list[dict]:
    """Scan for GGUF model files the user already has. Read-only, best-effort.

    Returns [{"path", "name", "size"}] sorted smallest-first. Excludes
    projector (mmproj) files and anything under DISCOVERY_MIN_BYTES, and
    dedupes hardlinked copies by (st_dev, st_ino) - an LM Studio library built
    from hardlinks would otherwise import every model twice. Unreadable
    directories are skipped silently: a permission error in one corner of a
    disk must not cost the user the rest of the scan.
    """
    note = say or (lambda _t: None)
    seen: set[tuple[int, int]] = set()
    by_identity: dict[tuple[str, int], dict] = {}
    scan_roots = [Path(r) for r in roots] if roots is not None else discovery_roots()
    for root in scan_roots:
        note(f"   scanning {root} ...")
        try:
            candidates = root.rglob("*.gguf")
        except OSError:
            continue
        while True:
            try:
                path = next(candidates)
            except StopIteration:
                break
            except OSError:
                continue
            name = path.name
            if "mmproj" in name.lower() or "imatrix" in name.lower():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size < DISCOVERY_MIN_BYTES:
                continue
            hard_key = (stat.st_dev, stat.st_ino)
            if hard_key in seen:
                continue
            seen.add(hard_key)
            # Same name AND same size in two places is the same model copied;
            # keep the first sighting rather than importing twins.
            identity = (name.lower(), stat.st_size)
            if identity in by_identity:
                continue
            by_identity[identity] = {
                "path": str(path), "name": name, "size": stat.st_size,
            }
    found = sorted(by_identity.values(), key=lambda r: r["size"])
    note(f"   found {len(found)} model file(s)")
    return found


def place_into_models_dir(source: Path | str, models_dir: Path | str) -> str:
    """Give LOCITIZE's models folder a name for `source` without copying it.

    Same volume: a hardlink (the file is genuinely IN the folder, zero bytes
    spent, and the original tool keeps working). Different volume, or any
    link failure: the original path is used as-is - registration by absolute
    path is how the owner's own registry references three different stores.
    Never copies: duplicating a 15 GB weight file to satisfy a folder
    convention is not a favour to anyone's disk.
    """
    src = Path(source)
    target_dir = Path(models_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name
        if target.exists():
            try:
                if target.stat().st_ino == src.stat().st_ino:
                    return str(target)  # already linked
            except OSError:
                pass
            return str(src)  # a DIFFERENT file already owns that name
        os.link(src, target)
        return str(target)
    except OSError:
        return str(src)


# --------------------------------------------------------------------------- #
# Update check (M17.4)
# --------------------------------------------------------------------------- #
# Local software that keeps itself current without phoning home except when you
# ask. This checks the installed llama.cpp build against the latest release and
# reports whether an update exists - it never downloads on its own. The one
# network call flows through the same allowlisted opener as everything else, so
# it shows up in the privacy ledger like any other egress: honest by
# construction.

def installed_llama_build(server_path: str | Path) -> int | None:
    """Parse the build number from `llama-server --version`, or None.

    llama-server prints e.g. 'version: 0.3.0-dev (build 10684, commit ...)'.
    The build number is the monotonic integer that actually orders releases.
    """
    import re

    exe = Path(server_path)
    if not exe.is_file():
        return None
    rc, out = _run([str(exe), "--version"], timeout=30)
    # Two known formats: newer builds print "build 10684, commit ..."; older
    # ones print "version: 10037 (hash)" where the version IS the build number.
    match = re.search(r"build (\d+)", out) or re.search(r"version:\s*(\d{3,})", out)
    return int(match.group(1)) if match else None


def latest_llama_build(want_cuda: bool) -> tuple[int | None, str]:
    """The newest available build number, from the release the fetcher would
    pick. Returns (build, error); build is None on any failure."""
    import re

    asset, error = resolve_llama_release(want_cuda)
    if asset is None:
        return None, error
    match = re.search(r"-b(\d+)-", asset.name) or re.search(r"(\d{4,})", asset.name)
    if not match:
        return None, f"could not read a build number from '{asset.name}'"
    return int(match.group(1)), ""


def check_llama_update(
    server_path: str | Path, want_cuda: bool
) -> dict[str, object]:
    """Compare installed vs latest llama.cpp. Pure result dict, never raises.

    {"ok", "installed", "latest", "update_available", "detail"} - so a caller
    (CLI or GUI) renders one honest line without re-deriving anything.
    """
    installed = installed_llama_build(server_path)
    if installed is None:
        return {"ok": False, "installed": None, "latest": None,
                "update_available": False,
                "detail": "no installed llama-server found to check"}
    latest, error = latest_llama_build(want_cuda)
    if latest is None:
        return {"ok": False, "installed": installed, "latest": None,
                "update_available": False, "detail": error}
    return {
        "ok": True, "installed": installed, "latest": latest,
        "update_available": latest > installed,
        "detail": (f"update available: b{installed} -> b{latest}"
                   if latest > installed else f"up to date (b{installed})"),
    }
