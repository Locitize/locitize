"""Capture the Fine-tune page's four steps (read-only, like ui_screenshot_all_pages.py).

Uses the real config and registry but starts nothing: the steps are filled with
a small sample dataset and a stand-in finished run in a temporary folder, so no
training, GPU work or write to the data root happens.

Usage:  python scripts/ui_screenshot_finetune_flow.py <output-dir>
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("UI_SHOTS_PLATFORM", "windows"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

import config
import desktop
from gui_controller import GuiController
from health import DefaultSystemInfoProvider, NvidiaSmiGpuInfoProvider
from launcher import Launcher
from logger import resolve_log_dir
from PySide6 import QtWidgets

settings, models, _ = config.Config.load()
lch = Launcher()
registry, manager, controller = lch._build_controller(
    settings, models, log_path=resolve_log_dir(settings) / "shot_llama.log")
gc = GuiController(
    settings, registry, controller, lch._build_whisper_controller(settings, manager), manager,
    listen_fn=lambda seconds, emit: None, speak_fn=lambda text, voice: None,
    benchmark_history_fn=lambda rows: lch._gui_benchmark_history(settings, rows),
    gpu_provider=NvidiaSmiGpuInfoProvider(), sys_provider=DefaultSystemInfoProvider(),
)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
app.setStyleSheet(desktop.APP_STYLE)
window = desktop.MainWindow(gc, health="OK")
window.resize(1400, 900)
window.show()
window._go_to("Fine-tune")


def settle(seconds=0.6):
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.03)


def shot(name):
    settle()
    page = window._ft_flow.parentWidget()
    page.grab().save(str(out / f"{name}.png"))
    print("saved", name)


flow = window._ft_flow
shot("1-model")

flow.custom_model.setText("")
flow.model = flow.chosen_model()
rows = [{"messages": [{"role": "user", "content": f"What are your opening hours on day {i}?"},
                      {"role": "assistant", "content": f"We are open 9am to 5pm on day {i}."}]} for i in range(1, 9)]
rows += [rows[0], {"instruction": "Say hello", "output": "Hello! How can I help?"},
         {"messages": [{"role": "user", "content": "A question with no answer"}]}]
flow.go(2)
flow.show_rows(rows, "file shop-faq.jsonl", "shop-faq")
shot("2-dataset")

tmp = Path(tempfile.mkdtemp(prefix="ft-shot-"))
dataset = tmp / "shop-faq.jsonl"
dataset.write_text("\n".join(json.dumps(r) for r in flow.final_rows()), encoding="utf-8")
flow.dataset_path, flow.dataset_rows = str(dataset), len(flow.final_rows())
flow.train_name.setText("shop-faq")
flow._show_train_form(True)
flow.go(3)
shot("3-train")

run = tmp / "shop-faq-20261019-120000"
(run / "merged_model").mkdir(parents=True)
(run / "lora_adapter").mkdir()
(run / "shop-faq-20261019-120000.q4_k_m.gguf").write_bytes(b"GGUF" + b"0" * 2048)
(run / "run_meta.json").write_text(json.dumps({"base_model": flow.model, "default_system": ""}), encoding="utf-8")
flow.open_run(run)
shot("4-save")

window.close()
app.processEvents()
print("done")
