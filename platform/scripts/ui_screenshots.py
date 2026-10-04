"""Offscreen/native screenshot harness: render every desktop page to PNG (M15.9).

Design changes to desktop.py are verified by LOOKING at them, not by imagining
them: this renders the real MainWindow against the test suite's own
FakeGuiController (no model, GPU, or service is touched) and grabs one PNG per
sidebar page. The M15.9 UI pass was reviewed as before/after pairs from exactly
this harness, which is why it is kept as a script rather than a scratch file.

Usage:
    python scripts/ui_screenshots.py <out_dir>

Set UI_SHOTS_PLATFORM=windows to render with the native platform plugin (real
font rasterisation; the window appears briefly). The default offscreen platform
is fully headless but its font database may draw placeholder glyphs - geometry
is faithful either way.
"""
import os
import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_DIR))
sys.path.insert(0, str(PLATFORM_DIR / "tests"))
os.environ.setdefault(
    "QT_QPA_PLATFORM", os.environ.get("UI_SHOTS_PLATFORM", "offscreen")
)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "ui_shots")
out.mkdir(parents=True, exist_ok=True)

from test_desktop import FakeGuiController  # noqa: E402
import desktop  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
app.setStyleSheet(desktop.APP_STYLE)
window = desktop.MainWindow(FakeGuiController(), health="OK")
window.resize(1196, 819)
window.move(20, 20)
window.show()
app.processEvents()

for index in range(window._sidebar.count()):
    window._sidebar.setCurrentRow(index)
    app.processEvents()
    name = window._sidebar.item(index).text().strip().lower().replace(" ", "-")
    window.grab().save(str(out / f"{index}-{name}.png"))
    print(f"saved {index}-{name}.png")

window.close()
print(f"done -> {out}")
