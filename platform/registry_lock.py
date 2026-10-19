"""Cross-process lock shared by the atomic model registry writers."""
from contextlib import contextmanager
import os
import threading
import time
import hashlib
import tempfile
from pathlib import Path
from functools import wraps

_mutex = threading.RLock()


def registry_locked(function):
    """Keep the registry's single editing chokepoint inside a shared OS lock."""
    @wraps(function)
    def wrapped(base_dir, *args, **kwargs):
        with registry_write_lock(Path(base_dir) / "models.yaml"):
            return function(base_dir, *args, **kwargs)
    return wrapped


@contextmanager
def registry_write_lock(path):
    """Lock a stable sidecar, because atomic replacement changes the data inode."""
    with _mutex:
        # A per-user temporary lock avoids leaving runtime files in the install
        # or portable data tree. Resolve/case-fold so all writers share one key.
        identity = os.path.normcase(str(Path(path).resolve())).encode("utf-8")
        lock_path = Path(tempfile.gettempdir()) / ("locitize-registry-" + hashlib.sha256(identity).hexdigest() + ".lock")
        with open(lock_path, "a+b") as handle:
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            deadline = time.monotonic() + 5
            while True:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("model registry is busy; retry later")
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
