"""Render the desktop's Models page against the REAL registry.

Read-only by construction: the GuiController is built with the real config and
registry (so the window shows the machine's actual 33 models and benchmark
history) but start_threads() is never called and no request_* method fires, so
no process, port, or GPU is touched. The single-instance lock is taken by
run(), not by MainWindow, so this coexists with a running desktop.
"""
import os
import sys
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
window.resize(1196, 819)
window.move(20, 20)
window.show()
app.processEvents()
window._sidebar.setCurrentRow(0)
app.processEvents()
window.grab().save(str(out / "models-real.png"))
print("saved models-real.png")
window.close()
