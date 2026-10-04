"""Capture every desktop page against the REAL registry (M18.13).

The README's visual tour must show real pixels: the machine's actual models,
the real GPU name in the header, the shipped dark theme - never the fake test
providers (the old assets showed "Fake GPU" and invented model rows, which is
exactly what a measured-not-assumed product cannot put on its front page).

Read-only by construction, same discipline as ui_screenshot_real.py: the
GuiController is built with the real config and registry, but start_threads()
is never called and no request_* fires, so no model process, port, or GPU
allocation happens. The desktop's own background RAM/VRAM sampler does run -
that is a read-only nvidia-smi/psutil poll and gives the System memory meters
live, honest numbers in the shot.

Usage:  python scripts/ui_screenshot_all_pages.py <output-dir>
Writes one PNG per page, named after the sidebar entry.
"""
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("UI_SHOTS_PLATFORM", "windows"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

import config
import desktop
from launcher import Launcher
from logger import resolve_log_dir
from gui_controller import GuiController
from health import DefaultSystemInfoProvider, NvidiaSmiGpuInfoProvider
from PySide6 import QtWidgets

settings, models, _ = config.Config.load()
lch = Launcher()
log_path = resolve_log_dir(settings) / "shot_llama.log"
registry, manager, controller = lch._build_controller(settings, models, log_path=log_path)
whisper_controller = lch._build_whisper_controller(settings, manager)

gc = GuiController(
    settings,
    registry,
    controller,
    whisper_controller,
    manager,
    listen_fn=lambda seconds, emit: None,
    speak_fn=lambda text, voice: None,
    benchmark_history_fn=lambda rows: lch._gui_benchmark_history(settings, rows),
    gpu_provider=NvidiaSmiGpuInfoProvider(),
    sys_provider=DefaultSystemInfoProvider(),
)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
app.setStyleSheet(desktop.APP_STYLE)
window = desktop.MainWindow(gc, health="OK")
window.resize(1400, 860)
window.move(20, 20)
window.show()
app.processEvents()

# Let the live RAM/VRAM sampler post a few readings so the System memory meters
# in the Get models panel show real numbers rather than "-".
deadline = time.time() + 5.0
while time.time() < deadline:
    app.processEvents()
    time.sleep(0.05)

for row_index, name in enumerate(desktop.PAGE_NAMES):
    window._sidebar.setCurrentRow(row_index)
    for _ in range(12):
        app.processEvents()
        time.sleep(0.03)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    target = out / f"{slug}.png"
    window.grab().save(str(target))
    print(f"saved {target.name}")

window.close()
app.processEvents()
print("done")
