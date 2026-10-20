"""The guided fine-tune flow on the desktop Fine-tune page.

Four steps, one screen each: choose a base model, prepare the dataset (get rows
from a file, documents, a word table or a bigger model, then review and clean
them), train, and save the result to a folder of the owner's choosing - and onto
the model list. It replaces the separate browser studio.

The heavy lifting is finetune_jobs (trainer children in the fine-tune
environment) and finetune_dataset (pure review logic). Anything that can take
more than a moment runs on a QThreadPool worker, like the Sessions page; a
running training or distillation is followed by polling its log file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

import finetune_dataset as ftd
import finetune_jobs as jobs

STEPS = ("Choose a model", "Prepare the dataset", "Train", "Save your model")
SOURCES = ("I have a training file", "Build it from my documents",
           "Build it from a word table", "Have a bigger model write it")
REVIEW_TABLE_LIMIT = 2000
POLL_MS = 1500
DISTILL_LOG_MEMORY_CHARS = 128 * 1024
DISTILL_LOG_VIEW_CHARS = 16 * 1024
_KEEP, _USER, _ASSISTANT, _PROBLEM, _TOKENS = range(5)
_EDITABLE = QtCore.Qt.ItemFlag.ItemIsEditable


class _Signals(QtCore.QObject):
    done = QtCore.Signal(object, object)


class _Task(QtCore.QRunnable):
    """Run fn() on the pool; done(result, error) arrives on the GUI thread."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.fn(), None)
        except Exception as exc:  # noqa: BLE001 - shown to the owner as text
            self.signals.done.emit(None, str(exc) or exc.__class__.__name__)


def _label(text="", muted=False, chip=False, error=False):
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    if chip:
        label.setObjectName("statusChip")
    elif error:
        label.setObjectName("error")
    elif muted:
        label.setObjectName("muted")
    return label


def _row(*widgets, stretch=True):
    row = QtWidgets.QHBoxLayout()
    for widget in widgets:
        row.addWidget(widget)
    if stretch:
        row.addStretch(1)
    return row


def _form(*pairs):
    form = QtWidgets.QFormLayout()
    form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
    for text, widget in pairs:
        form.addRow(text, widget)
    return form


class _Pages(QtWidgets.QWidget):
    """Show one child at a time, sized to that child.

    A QStackedWidget reserves the height of its tallest page, which left a
    screen of empty panel under the short steps; hidden children here take no
    space at all.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("plainGroup")
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._children = []
        self._current = -1

    def addWidget(self, widget):
        self._children.append(widget)
        self._layout.addWidget(widget)
        widget.setVisible(len(self._children) - 1 == self._current)

    def count(self):
        return len(self._children)

    def widget(self, index):
        return self._children[index]

    def currentIndex(self):
        return self._current

    def setCurrentIndex(self, index):
        self._current = index
        for i, child in enumerate(self._children):
            child.setVisible(i == index)


def _primary(text):
    button = QtWidgets.QPushButton(text)
    button.setObjectName("primaryButton")
    return button


class FineTuneFlow(QtWidgets.QWidget):
    """Stepper plus one panel per step. `window` supplies the shared panel helpers."""

    def __init__(self, window, controller, sparkline_cls, on_model_saved=None, parent=None):
        super().__init__(parent)
        self._window = window
        self._gc = controller
        self._settings = getattr(controller, "_settings", None)
        self._on_model_saved = on_model_saved or (lambda: None)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(2)
        self._tasks = set()

        self.step = 1
        self.model = None
        self.records = None
        self.source_label = ""
        self.dataset_path = None
        self.dataset_rows = 0
        self.default_system = ""
        self.train_job = None
        self.run_dir = None
        self.run_done = False
        self.distill_job = None
        self._log_offset = 0
        self._log_text = ""
        self._loading_table = False
        self._fit_for = None
        self._shown_losses = 0
        self._distill_offset, self._distill_text = 0, ""
        self._distill_log_path = None
        self._distill_run_dir = None
        self._history_runs = []
        self._history_selected_path = None
        self._sparkline_cls = sparkline_cls

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        stepper = QtWidgets.QHBoxLayout()
        self.step_buttons = []
        for number, title in enumerate(STEPS, 1):
            button = QtWidgets.QPushButton(f"{number}. {title}")
            button.clicked.connect(lambda _c=False, n=number: self.go(n))
            stepper.addWidget(button, 1)
            self.step_buttons.append(button)
        layout.addLayout(stepper)

        self.unavailable = _label("", error=True)
        self.unavailable.setVisible(False)
        layout.addWidget(self.unavailable)

        self.pages = _Pages()
        for builder in (self._build_model_step, self._build_dataset_step,
                        self._build_train_step, self._build_save_step):
            self.pages.addWidget(builder())
        layout.addWidget(self.pages)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

        self._check_environment()
        self._on_model_choice()
        self.go(1)

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #

    def _run(self, fn, done):
        task = _Task(fn)
        self._tasks.add(task)

        def finished(result, error, task=task):
            self._tasks.discard(task)
            done(result, error)

        task.signals.done.connect(finished)
        self._pool.start(task)

    def _check_environment(self):
        try:
            jobs.environment(self._settings)
        except jobs.FineTuneUnavailable as exc:
            self.unavailable.setText(str(exc))
            self.unavailable.setVisible(True)
            return False
        self.unavailable.setVisible(False)
        return True

    def reachable(self, number):
        return {
            1: True,
            2: bool(self.model),
            3: bool(self.model and self.dataset_path),
            4: bool(self.run_dir and self.run_done),
        }[number]

    def go(self, number):
        if not self.reachable(number):
            return
        self.step = number
        self.pages.setCurrentIndex(number - 1)
        if number == 3:
            self._refresh_train_summary()
        if number == 4:
            self._refresh_save_step()
        if number == 2:
            self._refresh_doc_qa_gate()
        self._refresh_stepper()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_doc_qa_gate()

    def _refresh_stepper(self):
        for number, button in enumerate(self.step_buttons, 1):
            button.setObjectName("primaryButton" if number == self.step else "")
            button.setEnabled(self.reachable(number))
            button.style().unpolish(button)
            button.style().polish(button)

    def shutdown(self):
        """Window closing: stop polling. A live run keeps going (the page warns)."""
        self._timer.stop()

    def run_active(self):
        return bool(self.train_job and self.train_job.running()) or bool(
            self.distill_job and self.distill_job.running())

    # ------------------------------------------------------------------ #
    # Step 1: model
    # ------------------------------------------------------------------ #

    def _build_model_step(self):
        frame, body = self._window._panel()
        self._window._section(
            body, "Step 1 - Choose the model to fine-tune",
            "Fine-tuning teaches an existing model new behaviour from your examples. "
            "Smaller models train faster; bigger ones are smarter. If you are unsure, "
            "start with the first one.")
        self.model_combo = QtWidgets.QComboBox()
        for model_id, _note in jobs.MODEL_CHOICES:
            self.model_combo.addItem(model_id.split("/")[-1], model_id)
        self.model_combo.currentIndexChanged.connect(self._on_model_choice)
        body.addLayout(_form(("Base model", self.model_combo)))
        self.model_note = _label(muted=True)
        body.addWidget(self.model_note)
        self.custom_model = QtWidgets.QLineEdit()
        self.custom_model.setPlaceholderText("Optional: any Hugging Face model id, e.g. org/Model-7B-Instruct")
        self.custom_model.editingFinished.connect(self._on_model_choice)
        body.addLayout(_form(("Or type a model id", self.custom_model)))
        self.fit_chip = _label("", chip=True)
        body.addWidget(self.fit_chip)
        self.model_next = _primary("Next: prepare the dataset")
        self.model_next.clicked.connect(self._on_model_next)
        body.addLayout(_row(self.model_next))
        body.addStretch(1)
        return frame

    def chosen_model(self):
        return self.custom_model.text().strip() or self.model_combo.currentData()

    def _on_model_choice(self, *_args):
        model = self.chosen_model()
        note = dict(jobs.MODEL_CHOICES).get(model, "A model of your choice from Hugging Face.")
        self.model_note.setText(note)
        if model == self._fit_for or self.unavailable.isVisible():
            return
        self._fit_for = model
        self.fit_chip.setText("Checking whether it fits your graphics card...")
        settings = self._settings
        self._run(lambda: jobs.preflight(settings, model),
                  lambda result, error, model=model: self._on_fit(model, result, error))

    def _on_fit(self, model, result, error):
        if model != self.chosen_model():
            return  # the owner moved on; a newer check is on its way
        if error or not result:
            self.fit_chip.setText(f"Could not check the graphics card: {error}")
            return
        if result.get("ok"):
            self.fit_chip.setText(
                f"Fits: needs about {result.get('need_mb', 0) / 1024:.1f} GB, "
                f"{result.get('free_mb', 0) / 1024:.1f} GB free on {result.get('gpu_name', 'your GPU')}.")
        else:
            self.fit_chip.setText(" ".join(result.get("problems", [])))

    def _on_model_next(self):
        model = self.chosen_model()
        if not model:
            return
        if model != self.model:
            self.run_dir, self.run_done, self.train_job = None, False, None
        self.model = model
        self.go(2)

    # ------------------------------------------------------------------ #
    # Step 2: dataset
    # ------------------------------------------------------------------ #

    def _build_dataset_step(self):
        frame, body = self._window._panel()
        self._window._section(
            body, "Step 2 - Prepare the dataset",
            "A dataset is a list of example conversations: a question and the answer you "
            "want the model to give. Get one in, then review it below before training.")
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(SOURCES)
        body.addLayout(_form(("Where does your data come from?", self.source_combo)))
        self.source_pages = _Pages()
        for builder in (self._build_source_file, self._build_source_documents,
                        self._build_source_table, self._build_source_distill):
            self.source_pages.addWidget(builder())
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.source_pages.setCurrentIndex(0)
        body.addWidget(self.source_pages)
        self.source_status = _label(muted=True)
        body.addWidget(self.source_status)

        self.review = QtWidgets.QWidget()
        self.review.setObjectName("plainGroup")
        review = QtWidgets.QVBoxLayout(self.review)
        review.setContentsMargins(0, 8, 0, 0)
        self.review_title = QtWidgets.QLabel("Review your dataset")
        self.review_title.setObjectName("sectionTitle")
        review.addWidget(self.review_title)
        self.review_summary = _label(chip=True)
        review.addWidget(self.review_summary)
        self.review_notes = _label(muted=True)
        review.addWidget(self.review_notes)
        self.drop_duplicates = QtWidgets.QCheckBox("Leave out duplicate rows")
        self.drop_duplicates.setChecked(True)
        self.drop_duplicates.toggled.connect(self._update_final_count)
        review.addWidget(self.drop_duplicates)
        self.system_prompt = QtWidgets.QLineEdit()
        self.system_prompt.setPlaceholderText("Optional: one instruction for every row, e.g. You are a friendly support agent.")
        self.system_prompt.textChanged.connect(self._update_final_count)
        review.addLayout(_form(("System prompt", self.system_prompt)))
        review.addWidget(_label(
            "Untick Keep to leave a row out. You can fix the text of simple question and "
            "answer rows by double-clicking a cell.", muted=True))
        self.review_table = QtWidgets.QTableWidget(0, 5)
        self.review_table.setHorizontalHeaderLabels(["Keep", "User", "Assistant", "Problem", "Tokens"])
        self.review_table.verticalHeader().setVisible(False)
        self.review_table.setAlternatingRowColors(True)
        self.review_table.setMinimumHeight(260)
        header = self.review_table.horizontalHeader()
        header.setSectionResizeMode(_USER, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_ASSISTANT, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in (_KEEP, _PROBLEM, _TOKENS):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.review_table.itemChanged.connect(self._update_final_count)
        review.addWidget(self.review_table)
        self.dataset_name = QtWidgets.QLineEdit()
        self.dataset_name.textChanged.connect(self._update_final_count)
        review.addLayout(_form(("Dataset name", self.dataset_name)))
        self.final_count = _label(chip=True)
        review.addWidget(self.final_count)
        self.review.setVisible(False)
        body.addWidget(self.review)

        back = QtWidgets.QPushButton("Back: change the model")
        back.clicked.connect(lambda: self.go(1))
        self.dataset_next = _primary("Next: train")
        self.dataset_next.clicked.connect(self._on_dataset_next)
        self.dataset_next.setEnabled(False)
        body.addLayout(_row(self.dataset_next, back))
        return frame

    def _on_source_changed(self, index):
        self.source_pages.setCurrentIndex(index)
        if index == 1:
            self._refresh_doc_qa_gate()

    def _source_box(self, hint):
        box = QtWidgets.QWidget()
        box.setObjectName("plainGroup")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_label(hint, muted=True))
        return box, layout

    def _build_source_file(self):
        box, layout = self._source_box(
            'A .jsonl file (one example per line) or a .json list. Each example is '
            '{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} or '
            '{"instruction": ..., "output": ...}.')
        pick = _primary("Choose a training file...")
        pick.clicked.connect(self._on_pick_file)
        layout.addLayout(_row(pick))
        return box

    def _build_source_documents(self):
        box, layout = self._source_box(
            "Notes, manuals or exports (.txt .md .csv .docx .pdf) are cut into training rows. "
            "If a model is running in LoCiTiZe, it also writes question and answer pairs "
            "about them - all on this computer.")
        self.doc_files = []
        pick = QtWidgets.QPushButton("Choose documents...")
        pick.clicked.connect(self._on_pick_documents)
        self.doc_label = _label("No documents chosen.", muted=True)
        layout.addLayout(_row(pick, self.doc_label))
        self.doc_qa = QtWidgets.QCheckBox("Also write question and answer pairs with the running model")
        self.doc_qa.setChecked(False)
        self.doc_qa.toggled.connect(self._update_doc_pairs_enabled)
        self.doc_qa_hint = _label("", muted=True)
        self.doc_pairs = QtWidgets.QSpinBox()
        self.doc_pairs.setRange(1, 6)
        self.doc_pairs.setValue(3)
        self.doc_name = QtWidgets.QLineEdit()
        self.doc_name.setPlaceholderText("e.g. my-handbook")
        layout.addWidget(self.doc_qa)
        layout.addWidget(self.doc_qa_hint)
        layout.addLayout(_form(("Pairs per passage", self.doc_pairs), ("Dataset name", self.doc_name)))
        self.doc_build = _primary("Build dataset from documents")
        self.doc_build.clicked.connect(self._on_build_documents)
        layout.addLayout(_row(self.doc_build))
        self._refresh_doc_qa_gate()
        return box

    def _model_is_ready(self):
        reader = getattr(self._gc, "running_model_id", None)
        if not callable(reader):
            return False
        try:
            return bool(reader())
        except Exception:  # noqa: BLE001 - a missing controller snapshot means unavailable
            return False

    def _refresh_doc_qa_gate(self, ready=None):
        if not hasattr(self, "doc_qa"):
            return
        ready = self._model_is_ready() if ready is None else bool(ready)
        if not ready:
            self.doc_qa.setChecked(False)
        self.doc_qa.setEnabled(ready)
        self.doc_qa_hint.setText(
            "Start a model in LoCiTiZe to add local Q&A pairs."
            if not ready else "A running model is ready; select this if you want Q&A pairs."
        )
        self._update_doc_pairs_enabled()

    def _update_doc_pairs_enabled(self, *_args):
        if hasattr(self, "doc_pairs"):
            self.doc_pairs.setEnabled(self.doc_qa.isEnabled() and self.doc_qa.isChecked())

    def _build_source_table(self):
        box, layout = self._source_box(
            "A reviewed word table in markdown (| # | Word | Meaning | Example sentence |, "
            "example cell 'sentence - English translation'). Each word becomes several "
            "differently phrased rows, mixed with optional persona examples.")
        self.table_file = ""
        self.identity_file = ""
        pick = QtWidgets.QPushButton("Choose word table (.md)...")
        pick.clicked.connect(self._on_pick_table)
        self.table_label = _label("No table chosen.", muted=True)
        layout.addLayout(_row(pick, self.table_label))
        pick_id = QtWidgets.QPushButton("Choose persona examples (.jsonl, optional)...")
        pick_id.clicked.connect(self._on_pick_identity)
        self.identity_label = _label("", muted=True)
        layout.addLayout(_row(pick_id, self.identity_label))
        self.table_templates = QtWidgets.QSpinBox()
        self.table_templates.setRange(1, 16)
        self.table_templates.setValue(16)
        self.table_oversample = QtWidgets.QSpinBox()
        self.table_oversample.setRange(0, 30)
        self.table_oversample.setValue(8)
        self.table_oversample.setToolTip("Aim for persona rows to be about 5-10% of the total.")
        self.table_name = QtWidgets.QLineEdit()
        self.table_name.setPlaceholderText("e.g. my-assistant-v1")
        layout.addLayout(_form(("Phrasings per word", self.table_templates),
                               ("Repeat persona examples", self.table_oversample),
                               ("Dataset name", self.table_name)))
        self.table_build = _primary("Build dataset from the table")
        self.table_build.clicked.connect(self._on_build_table)
        layout.addLayout(_row(self.table_build))
        return box

    def _build_source_distill(self):
        box, layout = self._source_box(
            "A bigger model answers a list of your prompts, and the prompt and answer pairs "
            "become the dataset (knowledge distillation). Runs on your graphics card.")
        self.teacher = QtWidgets.QComboBox()
        for model_id, _note in jobs.MODEL_CHOICES:
            self.teacher.addItem(model_id.split("/")[-1], model_id)
        self.teacher.setCurrentIndex(3)
        self.distill_system = QtWidgets.QLineEdit("You are a helpful assistant.")
        self.distill_prompts = QtWidgets.QPlainTextEdit()
        self.distill_prompts.setPlaceholderText("One prompt per line\nHow do I reset my password?\nWhat is your return policy?")
        self.distill_prompts.setMinimumHeight(110)
        self.distill_load_prompts = QtWidgets.QPushButton("Load prompts from file...")
        self.distill_load_prompts.clicked.connect(self._on_load_prompts)
        self.distill_tokens = QtWidgets.QSpinBox()
        self.distill_tokens.setRange(32, 512)
        self.distill_tokens.setSingleStep(16)
        self.distill_tokens.setValue(200)
        self.distill_temp = QtWidgets.QDoubleSpinBox()
        self.distill_temp.setRange(0.0, 1.5)
        self.distill_temp.setSingleStep(0.1)
        self.distill_temp.setValue(0.7)
        self.distill_name = QtWidgets.QLineEdit()
        self.distill_name.setPlaceholderText("e.g. handbook-distill-v1")
        layout.addLayout(_form(("Teacher model", self.teacher), ("System prompt", self.distill_system),
                               ("Prompts", self.distill_prompts), ("Answer length (tokens)", self.distill_tokens),
                               ("Temperature", self.distill_temp), ("Dataset name", self.distill_name)))
        layout.addLayout(_row(self.distill_load_prompts))
        self.distill_start = _primary("Start writing answers")
        self.distill_start.clicked.connect(self._on_distill_start)
        self.distill_stop = QtWidgets.QPushButton("Stop")
        self.distill_stop.setEnabled(False)
        self.distill_stop.clicked.connect(lambda: jobs.stop(self.distill_job))
        layout.addLayout(_row(self.distill_start, self.distill_stop))
        self.distill_progress = QtWidgets.QProgressBar()
        self.distill_progress.setVisible(False)
        layout.addWidget(self.distill_progress)
        self.distill_log_toggle = QtWidgets.QPushButton("Show log")
        self.distill_log_toggle.setCheckable(True)
        self.distill_log_view = QtWidgets.QPlainTextEdit()
        self.distill_log_view.setReadOnly(True)
        self.distill_log_view.setMinimumHeight(160)
        self.distill_log_view.setVisible(False)
        self.distill_log_toggle.toggled.connect(self.distill_log_view.setVisible)
        self.distill_open_log = QtWidgets.QPushButton("Open log")
        self.distill_open_log.setEnabled(False)
        self.distill_open_log.clicked.connect(lambda: self._open_path(self._distill_log_path))
        self.distill_open_folder = QtWidgets.QPushButton("Open folder")
        self.distill_open_folder.setEnabled(False)
        self.distill_open_folder.clicked.connect(lambda: self._open_folder(self._distill_run_dir))
        layout.addLayout(_row(self.distill_log_toggle, self.distill_open_log, self.distill_open_folder))
        layout.addWidget(self.distill_log_view)
        return box

    # -- sources ------------------------------------------------------- #

    def _busy(self, text):
        self.source_status.setObjectName("muted")
        self.source_status.setText(text)

    def _fail(self, text):
        self.source_status.setText(text)

    def _on_pick_file(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose a training file", "", "Datasets (*.jsonl *.json)")
        if path:
            self.load_file(path, f"file {Path(path).name}")

    def load_file(self, path, source_label):
        """Read a dataset file off the GUI thread and show it for review."""
        self._busy(f"Reading {Path(path).name}...")

        def read():
            data = Path(path).read_bytes()
            rows, notes = ftd.parse_dataset(data, str(path))
            return rows, notes

        def done(result, error):
            if error:
                self._fail(f"Could not read {Path(path).name}: {error}")
                return
            rows, notes = result
            self.show_rows(rows, source_label, Path(path).stem.replace("-source", ""), notes)

        self._run(read, done)

    def _on_pick_documents(self):
        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Choose documents", "", "Documents (*.txt *.md *.markdown *.csv *.docx *.pdf)")
        if paths:
            self.doc_files = paths
            self.doc_label.setText(f"{len(paths)} document(s): " + ", ".join(Path(p).name for p in paths[:4])
                                   + (" ..." if len(paths) > 4 else ""))

    def _on_build_documents(self):
        if not self.doc_files or not self._check_environment():
            self._fail("Choose one or more documents first." if not self.doc_files else self.unavailable.text())
            return
        files, pairs = list(self.doc_files), self.doc_pairs.value() if self.doc_qa.isChecked() else 0
        name, settings = self.doc_name.text() or "documents", self._settings
        self.doc_build.setEnabled(False)
        self._busy("Building the dataset" + (" - the running model is writing questions, this can take a while"
                                             if pairs else "") + "...")

        def done(result, error):
            self.doc_build.setEnabled(True)
            if error:
                self._fail(f"No dataset was built: {error}")
                return
            path, _rows, notes = result
            self.load_file(path, f"{len(files)} document(s)")
            if notes:
                self.review_notes.setText("; ".join(notes))

        self._run(lambda: jobs.build_from_documents(settings, files, pairs, name), done)

    def _on_pick_table(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a word table", "", "Markdown (*.md)")
        if path:
            self.table_file = path
            self.table_label.setText(Path(path).name)

    def _on_pick_identity(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Choose persona examples", "", "JSONL (*.jsonl)")
        if path:
            self.identity_file = path
            self.identity_label.setText(Path(path).name)

    def _on_load_prompts(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load distillation prompts", "", "Prompt files (*.txt *.jsonl)")
        if not path:
            return
        if self.distill_prompts.toPlainText().strip():
            answer = QtWidgets.QMessageBox.question(
                self, "Replace prompts?",
                "The prompts box already has text. Replace it with prompts from this file?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.distill_load_prompts.setEnabled(False)
        self._busy(f"Reading {Path(path).name}...")

        def done(result, error):
            self.distill_load_prompts.setEnabled(True)
            if error:
                self._fail(f"Could not load prompts: {error}")
                return
            prompts, malformed = result
            if malformed:
                summary = ftd.format_line_numbers(malformed)
                answer = QtWidgets.QMessageBox.question(
                    self, "Some prompt lines are malformed",
                    f"{summary}. Load the {len(prompts)} valid prompt(s) and skip those lines?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    self._fail("The prompt file was not loaded.")
                    return
            self.distill_prompts.setPlainText("\n".join(prompts))
            self.source_status.setText(f"Loaded {len(prompts)} prompt(s) from {Path(path).name}.")

        self._run(lambda: ftd.load_prompt_file(path), done)

    def _on_build_table(self):
        if not self.table_file or not self._check_environment():
            self._fail("Choose a word table first." if not self.table_file else self.unavailable.text())
            return
        args = (self._settings, self.table_file, self.identity_file, self.table_templates.value(),
                self.table_oversample.value(), self.table_name.text() or "word-table")
        self.table_build.setEnabled(False)
        self._busy("Building the dataset...")

        def done(result, error):
            self.table_build.setEnabled(True)
            if error:
                self._fail(f"No dataset was built: {error}")
                return
            path, _rows, notes = result
            self.load_file(path, "word table")
            if notes:
                self.review_notes.setText("; ".join(notes))

        self._run(lambda: jobs.build_from_word_table(*args), done)

    def _on_distill_start(self):
        prompts = [p for p in self.distill_prompts.toPlainText().splitlines() if p.strip()]
        if not prompts:
            self._fail("Type at least one prompt.")
            return
        if not self._check_environment():
            self._fail(self.unavailable.text())
            return
        try:
            self.distill_job = jobs.start_distill(
                self._settings, teacher=self.teacher.currentData(), system=self.distill_system.text(),
                prompts=prompts, max_new_tokens=self.distill_tokens.value(),
                temperature=self.distill_temp.value(), label=self.distill_name.text())
        except (OSError, jobs.FineTuneUnavailable) as exc:
            self._fail(f"Could not start: {exc}")
            return
        self._busy(f"The teacher model is answering {len(prompts)} prompt(s)...")
        self.distill_start.setEnabled(False)
        self.distill_stop.setEnabled(True)
        self.distill_progress.setRange(0, len(prompts))
        self.distill_progress.setValue(0)
        self.distill_progress.setVisible(True)
        self._distill_offset, self._distill_text = 0, ""
        self._distill_log_path = self.distill_job.log_path
        self._distill_run_dir = self.distill_job.log_path.parent
        self.distill_log_view.clear()
        self.distill_open_log.setEnabled(True)
        self.distill_open_folder.setEnabled(True)
        self._timer.start()

    # -- review -------------------------------------------------------- #

    def show_rows(self, rows, source_label, name, notes=()):
        """Hand rows from any source to the review table."""
        self.records = ftd.review(rows)
        self.source_label = source_label
        self.dataset_path = None
        stats = ftd.summarize(self.records)
        self.review_title.setText(f"Review your dataset ({source_label})")
        self.review_summary.setText(
            f"{stats['total']} rows: {stats['usable']} usable, {stats['problems']} with problems, "
            f"{stats['duplicates']} duplicates")
        lines = list(notes)
        if stats["problems"]:
            lines.append(f"{stats['problems']} row(s) cannot be trained on and will be left out (see Problem).")
        if stats["too_long"]:
            lines.append(f"{stats['too_long']} row(s) are longer than about {ftd.MAX_SEQ_TOKENS} tokens; "
                         "the trainer cuts them off at that length.")
        if stats["usable"] and stats["usable"] < ftd.SMALL_DATASET_ROWS:
            lines.append(f"Only {stats['usable']} usable rows: fine for a test, but real fine-tunes usually "
                         "need a few hundred examples.")
        if len(self.records) > REVIEW_TABLE_LIMIT:
            lines.append(f"The table shows the first {REVIEW_TABLE_LIMIT} rows; the rest are kept as they are.")
        self.review_notes.setText(" ".join(lines))
        self._fill_table()
        self.dataset_name.setText(ftd.safe_name(name, "dataset"))
        self.source_status.setText("")
        self.review.setVisible(True)
        self._update_final_count()
        self._refresh_stepper()

    def _fill_table(self):
        self._loading_table = True
        shown = self.records[:REVIEW_TABLE_LIMIT]
        table = self.review_table
        table.setRowCount(len(shown))
        for row, record in enumerate(shown):
            keep = QtWidgets.QTableWidgetItem()
            keep.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            usable = not record["problem"]
            keep.setCheckState(QtCore.Qt.CheckState.Checked if usable else QtCore.Qt.CheckState.Unchecked)
            if not usable:
                keep.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            keep.setData(QtCore.Qt.ItemDataRole.UserRole, record["index"])
            table.setItem(row, _KEEP, keep)
            editable = usable and ftd.is_single_turn(record["messages"])
            for column, text in ((_USER, ftd.first_text(record["messages"], "user")),
                                 (_ASSISTANT, ftd.last_text(record["messages"], "assistant"))):
                item = QtWidgets.QTableWidgetItem(text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, text)
                flags = item.flags() | _EDITABLE if editable else item.flags() & ~_EDITABLE
                item.setFlags(flags)
                table.setItem(row, column, item)
            problem = record["problem"] or ("duplicate" if record["duplicate"] else "")
            for column, text in ((_PROBLEM, problem), (_TOKENS, str(record["tokens"]))):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~_EDITABLE)
                table.setItem(row, column, item)
        self._loading_table = False

    def table_choices(self):
        """(keep, edits) from the review table, keyed by record index."""
        keep, edits = {}, {}
        table = self.review_table
        for row in range(table.rowCount()):
            keep_item = table.item(row, _KEEP)
            index = keep_item.data(QtCore.Qt.ItemDataRole.UserRole)
            keep[index] = keep_item.checkState() == QtCore.Qt.CheckState.Checked
            user, assistant = table.item(row, _USER), table.item(row, _ASSISTANT)
            if (user.text() != user.data(QtCore.Qt.ItemDataRole.UserRole)
                    or assistant.text() != assistant.data(QtCore.Qt.ItemDataRole.UserRole)):
                edits[index] = (user.text(), assistant.text())
        return keep, edits

    def final_rows(self):
        if not self.records:
            return []
        keep, edits = self.table_choices()
        return ftd.curated_rows(self.records, keep=keep, edits=edits,
                                drop_duplicates=self.drop_duplicates.isChecked(),
                                system_prompt=self.system_prompt.text())

    def _final_path(self):
        try:
            folder = jobs.datasets_dir(self._settings)
        except (jobs.FineTuneUnavailable, AttributeError, TypeError):
            return None
        return folder / f"{ftd.safe_name(self.dataset_name.text(), 'dataset')}.jsonl"

    def _update_final_count(self, *_args):
        if self._loading_table or not self.records:
            return
        count = len(self.final_rows())
        path = self._final_path()
        where = f" They will be saved to {path}" if path else ""
        self.final_count.setText(f"{count} rows will be used for training.{where}")
        self.dataset_next.setEnabled(count > 0 and path is not None)

    def _on_dataset_next(self):
        rows = self.final_rows()
        path = self._final_path()
        if not rows or path is None:
            return
        try:
            ftd.write_jsonl(rows, str(path))
        except OSError as exc:
            self._fail(f"Could not save the dataset: {exc}")
            return
        self.dataset_path = str(path)
        self.dataset_rows = len(rows)
        self.default_system = ftd.first_text(rows[0]["messages"], "system")
        self.run_dir, self.run_done, self.train_job = None, False, None
        self.train_name.setText(ftd.safe_name(self.dataset_name.text(), "model"))
        self._show_train_form(True)
        self.go(3)

    # ------------------------------------------------------------------ #
    # Step 3: train
    # ------------------------------------------------------------------ #

    def _build_train_step(self):
        frame, body = self._window._panel()
        self._window._section(body, "Step 3 - Train",
                              "Training runs on your graphics card. You can keep using the rest of LoCiTiZe.")
        self.train_summary = _label(chip=True)
        body.addWidget(self.train_summary)

        self.train_form = QtWidgets.QWidget()
        self.train_form.setObjectName("plainGroup")
        form_layout = QtWidgets.QVBoxLayout(self.train_form)
        form_layout.setContentsMargins(0, 0, 0, 0)
        self.train_name = QtWidgets.QLineEdit()
        self.train_name.setToolTip("Used for the output folder and the model file. A date and time is added.")
        self.train_length = QtWidgets.QComboBox()
        self.train_length.addItems(list(ftd.LENGTH_PRESETS))
        self.train_length.setCurrentIndex(1)
        self.train_length.setToolTip("One pass means the model sees every row once. Too many passes can make it "
                                     "repeat your examples word for word.")
        form_layout.addLayout(_form(("Name for your model", self.train_name), ("How long to train", self.train_length)))

        self.advanced_toggle = QtWidgets.QPushButton("Show advanced settings")
        self.advanced_toggle.setCheckable(True)
        form_layout.addLayout(_row(self.advanced_toggle))
        self.advanced = QtWidgets.QWidget()
        self.advanced.setObjectName("plainGroup")
        self.lr = QtWidgets.QComboBox()
        for value in ("1e-05", "5e-05", "0.0001", "0.0002", "0.0003", "0.0005"):
            self.lr.addItem(value, float(value))
        self.lr.setCurrentIndex(3)
        self.rank = QtWidgets.QComboBox()
        for value in (4, 8, 16, 32, 64):
            self.rank.addItem(str(value), value)
        self.rank.setCurrentIndex(2)
        self.batch = QtWidgets.QComboBox()
        for value in (1, 2, 4, 8):
            self.batch.addItem(str(value), value)
        self.batch.setCurrentIndex(1)
        self.accum = QtWidgets.QComboBox()
        for value in (1, 2, 4, 8):
            self.accum.addItem(str(value), value)
        self.accum.setCurrentIndex(2)
        self.steps = QtWidgets.QSpinBox()
        self.steps.setRange(ftd.MIN_STEPS, ftd.MAX_STEPS)
        self.steps.setSingleStep(10)
        self.quant = QtWidgets.QComboBox()
        for value in jobs.QUANTS:
            self.quant.addItem(f"{value} - {jobs.QUANT_NOTES[value]}", value)
        self.advanced.setLayout(_form(("Learning rate", self.lr), ("LoRA rank", self.rank), ("Batch size", self.batch),
                                      ("Gradient accumulation", self.accum), ("Training steps", self.steps),
                                      ("Model file quality", self.quant)))
        self.advanced.setVisible(False)
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        form_layout.addWidget(self.advanced)
        for combo in (self.train_length, self.batch, self.accum):
            combo.currentIndexChanged.connect(self._recommend_steps)
        self.steps_note = _label(muted=True)
        self.steps.valueChanged.connect(self._update_steps_note)
        form_layout.addWidget(self.steps_note)
        body.addWidget(self.train_form)

        self.train_stage = _label(chip=True)
        self.train_bar = QtWidgets.QProgressBar()
        self.loss_title = _label("Training loss (lower is better; it should drop, then level off)", muted=True)
        self.loss_graph = self._window_sparkline()
        self.loss_values = _label(muted=True)
        self.train_error = _label(error=True)
        self.log_toggle = QtWidgets.QPushButton("Show log")
        self.log_toggle.setCheckable(True)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(180)
        self.log_view.setVisible(False)
        self.log_toggle.toggled.connect(self.log_view.setVisible)
        self.progress_box = QtWidgets.QWidget()
        self.progress_box.setObjectName("plainGroup")
        progress_layout = QtWidgets.QVBoxLayout(self.progress_box)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.train_stage, self.train_bar, self.loss_title, self.loss_graph, self.loss_values,
                       self.train_error):
            progress_layout.addWidget(widget)
        progress_layout.addLayout(_row(self.log_toggle))
        progress_layout.addWidget(self.log_view)
        self.progress_box.setVisible(False)
        body.addWidget(self.progress_box)

        self.train_start = _primary("Start training")
        self.train_start.clicked.connect(self._on_train_start)
        self.train_stop = QtWidgets.QPushButton("Stop training")
        self.train_stop.clicked.connect(self._on_train_stop)
        self.train_back = QtWidgets.QPushButton("Back: change the dataset")
        self.train_back.clicked.connect(lambda: self.go(2))
        self.train_again = QtWidgets.QPushButton("Change settings and train again")
        self.train_again.clicked.connect(lambda: self._show_train_form(True))
        self.train_next = _primary("Next: save your model")
        self.train_next.clicked.connect(lambda: self.go(4))
        body.addLayout(_row(self.train_start, self.train_stop, self.train_next, self.train_again, self.train_back))
        self.history_toggle = QtWidgets.QPushButton("Run history")
        self.history_toggle.setCheckable(True)
        self.history_toggle.toggled.connect(self._toggle_run_history)
        body.addLayout(_row(self.history_toggle))
        self.history_box = QtWidgets.QWidget()
        self.history_box.setObjectName("plainGroup")
        history_layout = QtWidgets.QVBoxLayout(self.history_box)
        history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_refresh = QtWidgets.QPushButton("Refresh")
        self.history_refresh.clicked.connect(self._refresh_run_history)
        self.history_status = _label("", muted=True)
        history_layout.addLayout(_row(self.history_refresh, self.history_status))
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setMinimumHeight(120)
        self.history_list.currentRowChanged.connect(self._show_history_run)
        history_layout.addWidget(self.history_list)
        self.history_details = _label("", muted=True)
        history_layout.addWidget(self.history_details)
        self.history_loss_title = _label("Loss history", muted=True)
        self.history_loss_graph = self._window_sparkline()
        history_layout.addWidget(self.history_loss_title)
        history_layout.addWidget(self.history_loss_graph)
        self.history_log = QtWidgets.QPlainTextEdit()
        self.history_log.setReadOnly(True)
        self.history_log.setMinimumHeight(150)
        history_layout.addWidget(self.history_log)
        self.history_open_folder = QtWidgets.QPushButton("Open folder")
        self.history_open_folder.setEnabled(False)
        self.history_open_folder.clicked.connect(lambda: self._open_folder(self._history_selected_path))
        history_layout.addLayout(_row(self.history_open_folder))
        self.history_box.setVisible(False)
        body.addWidget(self.history_box)
        self._show_train_form(True)
        body.addStretch(1)
        return frame

    def _window_sparkline(self):
        cls = self._sparkline_cls
        return cls("#4f8cff") if cls else QtWidgets.QWidget()

    def _toggle_run_history(self, expanded):
        self.history_box.setVisible(expanded)
        if expanded:
            self._refresh_run_history()

    def _refresh_run_history(self):
        self.history_refresh.setEnabled(False)
        self.history_status.setText("Scanning runs...")
        settings, live_job = self._settings, self.train_job

        def done(result, error):
            self.history_refresh.setEnabled(True)
            if error:
                self._history_runs = []
                self.history_list.clear()
                self.history_status.setText(f"Could not scan run history: {error}")
                return
            self._history_runs = list(result or [])
            self.history_list.clear()
            for run in self._history_runs:
                self.history_list.addItem(
                    f"{run['name']}  |  {run['status']}  |  {run['modified_display']}"
                )
            self.history_status.setText(
                f"{len(self._history_runs)} run(s)" if self._history_runs else "No training runs yet."
            )
            if self._history_runs:
                self.history_list.setCurrentRow(0)
            else:
                self._show_history_run(-1)

        self._run(lambda: jobs.scan_training_runs(settings, live_job=live_job), done)

    def _reset_history_sparkline(self, losses):
        old = self.history_loss_graph
        self.history_loss_graph = self._window_sparkline()
        old.parentWidget().layout().replaceWidget(old, self.history_loss_graph)
        old.deleteLater()
        top = max(losses, default=0.0) or 1.0
        for value in losses:
            if hasattr(self.history_loss_graph, "push"):
                self.history_loss_graph.push(value / top)

    def _show_history_run(self, row):
        if row < 0 or row >= len(self._history_runs):
            self._history_selected_path = None
            self.history_details.setText("")
            self.history_log.clear()
            self.history_open_folder.setEnabled(False)
            self.history_loss_title.setText("Loss history")
            self._reset_history_sparkline([])
            return
        run = self._history_runs[row]
        self._history_selected_path = run["path"]
        lines = [
            f"Status: {run['status']}",
            f"Modified: {run['modified_display']}",
            f"Base model: {run['base_model'] or 'unknown'}",
            f"Dataset: {run['dataset_path'] or 'unknown'}",
        ]
        if run["meta"]:
            rendered = json.dumps(run["meta"], indent=2, ensure_ascii=True)
            if len(rendered) > 8000:
                rendered = rendered[:8000] + "\n... metadata display truncated"
            lines.append("Metadata:\n" + rendered)
        elif run["meta_error"]:
            lines.append(run["meta_error"])
        self.history_details.setText("\n".join(lines))
        self.history_log.setPlainText(run["log_tail"] or "(no training log)")
        losses = run["losses"]
        self.history_loss_title.setText(
            f"Loss history ({len(losses)} points; {losses[-1]:.3g} latest)"
            if losses else "Loss history (none found)"
        )
        self._reset_history_sparkline(losses)
        self.history_open_folder.setEnabled(Path(run["path"]).is_dir())

    def _show_train_form(self, editing):
        self.train_form.setVisible(editing)
        self.progress_box.setVisible(not editing)
        running = self.run_active() and self.train_job is not None
        self.train_start.setVisible(editing)
        self.train_back.setVisible(editing)
        self.train_stop.setVisible(not editing and running)
        self.train_next.setVisible(not editing and self.run_done)
        self.train_again.setVisible(not editing and not running)
        if editing:
            self._recommend_steps()

    def _refresh_train_summary(self):
        self.train_summary.setText(
            f"Model: {self.model}\nDataset: {self.dataset_rows} rows from {self.dataset_path}")

    def _recommend_steps(self, *_args):
        passes = ftd.LENGTH_PRESETS[self.train_length.currentText()]
        self.steps.setValue(ftd.recommended_steps(max(self.dataset_rows, 1), passes,
                                                  self.batch.currentData(), self.accum.currentData()))
        self._update_steps_note()

    def _update_steps_note(self, *_args):
        self.steps_note.setText(f"{self.steps.value()} training steps. The first run with a model downloads it "
                                "first, which can take a few minutes.")

    def _on_train_start(self):
        if not self._check_environment():
            return
        self.train_start.setEnabled(False)
        self.train_stage.setText("Checking your graphics card...")
        model, batch, settings = self.model, self.batch.currentData(), self._settings

        def done(result, error):
            self.train_start.setEnabled(True)
            if error or not result or not result.get("ok"):
                problems = [error] if error else (result or {}).get("problems", ["The check did not finish."])
                self.steps_note.setText(" ".join(str(p) for p in problems))
                return
            self._start_training()

        self._run(lambda: jobs.preflight(settings, model, batch), done)

    def _start_training(self):
        try:
            self.train_job = jobs.start_training(
                self._settings, model=self.model, dataset_path=self.dataset_path,
                label=self.train_name.text(), steps=self.steps.value(), learning_rate=self.lr.currentData(),
                lora_r=self.rank.currentData(), batch_size=self.batch.currentData(),
                grad_accum=self.accum.currentData(), quant=self.quant.currentData(),
                default_system=self.default_system, dataset_rows=self.dataset_rows)
        except (OSError, jobs.FineTuneUnavailable) as exc:
            self.steps_note.setText(f"Could not start training: {exc}")
            return
        self.run_dir = self.train_job.log_path.parent
        self.run_done = False
        self._log_offset, self._log_text = 0, ""
        self.loss_graph_reset()
        self.train_error.setText("")
        self.train_stage.setText("Starting up")
        self.train_bar.setRange(0, 0)
        self._show_train_form(False)
        self._timer.start()

    def loss_graph_reset(self):
        old = self.loss_graph
        self.loss_graph = self._window_sparkline()
        old.parentWidget().layout().replaceWidget(old, self.loss_graph)
        old.deleteLater()
        self.loss_values.setText("")

    def _on_train_stop(self):
        jobs.stop(self.train_job)
        self.train_stage.setText("Stopping...")

    # -- polling --------------------------------------------------------- #

    def _poll(self):
        busy = False
        if self.train_job is not None:
            busy |= self._poll_training()
        if self.distill_job is not None:
            busy |= self._poll_distill()
        if not busy:
            self._timer.stop()

    def _poll_training(self):
        job = self.train_job
        text, self._log_offset = jobs.read_new(job.log_path, self._log_offset)
        if text:
            self._log_text = (self._log_text + text)[-400_000:]
            self.log_view.setPlainText(self._log_text[-8000:])
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
        state = jobs.progress(self._log_text)
        self.train_stage.setText(state["stage"])
        if state["total"]:
            self.train_bar.setRange(0, state["total"])
            self.train_bar.setValue(state["current"])
        losses = state["losses"]
        if losses and len(losses) != self._shown_losses:
            top = max(losses) or 1.0
            for value in losses[self._shown_losses:]:
                if hasattr(self.loss_graph, "push"):
                    self.loss_graph.push(value / top)
            self._shown_losses = len(losses)
            self.loss_values.setText(f"loss {losses[0]:.2f} at the start, {losses[-1]:.2f} now")
        if job.running():
            return True
        self._shown_losses = 0
        self.run_done = state["done"]
        if state["done"]:
            # The finished run now belongs in the fine-tuned models table below.
            self._on_model_saved()
            self.train_stage.setText("Done - your model is ready to save.")
            self.train_bar.setRange(0, 1)
            self.train_bar.setValue(1)
        else:
            code = job.exit_code()
            self.train_stage.setText("Stopped before finishing" + (f" (exit code {code})" if code else ""))
            self.train_bar.setRange(0, 1)
            self.train_bar.setValue(0)
            settings, log_path = self._settings, job.log_path
            self._run(lambda: jobs.explain_failure(settings, log_path),
                      lambda reason, _e: self.train_error.setText(
                          reason or "See the log for details, then change settings and try again."))
            self.log_toggle.setChecked(True)
        self._show_train_form(False)
        self._refresh_stepper()
        if self.history_toggle.isChecked():
            self._refresh_run_history()
        return False

    def _poll_distill(self):
        job = self.distill_job
        text, self._distill_offset = jobs.read_new(
            job.log_path, self._distill_offset, max_bytes=DISTILL_LOG_MEMORY_CHARS
        )
        self._distill_text = (self._distill_text + text)[-DISTILL_LOG_MEMORY_CHARS:]
        if text:
            self.distill_log_view.setPlainText(self._distill_text[-DISTILL_LOG_VIEW_CHARS:])
            bar = self.distill_log_view.verticalScrollBar()
            bar.setValue(bar.maximum())
        state = jobs.progress(self._distill_text)
        if state["total"]:
            self.distill_progress.setValue(state["current"])
        if job.running():
            return True
        self.distill_start.setEnabled(True)
        self.distill_stop.setEnabled(False)
        self.distill_progress.setVisible(False)
        if state["done"] and job.out_path and job.out_path.is_file():
            self.load_file(str(job.out_path), "distillation")
        else:
            self._fail("The teacher model stopped before finishing. Log: " + str(job.log_path))
            self.distill_log_toggle.setChecked(True)
        self.distill_job = None
        return False

    # ------------------------------------------------------------------ #
    # Step 4: save
    # ------------------------------------------------------------------ #

    def _build_save_step(self):
        frame, body = self._window._panel()
        self._window._section(body, "Step 4 - Save your model",
                              "Pick a name and a folder. The model is also added to your model list so you "
                              "can run it from the Start page.")
        self.saved_where = _label(chip=True)
        body.addWidget(self.saved_where)
        self.open_run_btn = QtWidgets.QPushButton("Open the run folder")
        self.open_run_btn.clicked.connect(lambda: self._open_folder(self.run_dir))
        body.addLayout(_row(self.open_run_btn))

        self.save_name = QtWidgets.QLineEdit()
        self.save_folder = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._on_browse_folder)
        folder_row = QtWidgets.QHBoxLayout()
        folder_row.addWidget(self.save_folder, 1)
        folder_row.addWidget(browse)
        self.save_register = QtWidgets.QCheckBox("Add it to my model list")
        self.save_register.setChecked(True)
        self.save_prompt = QtWidgets.QLineEdit()
        self.save_context = QtWidgets.QSpinBox()
        self.save_context.setRange(512, 262144)
        self.save_context.setSingleStep(1024)
        self.save_context.setValue(getattr(getattr(self._settings, "finetune", None), "default_context_size", 8192))
        self.save_id = QtWidgets.QLineEdit()
        self.save_id.setToolTip("The key the model list uses for this model. Must be new.")
        self.save_gguf = QtWidgets.QComboBox()
        form = _form(("Model name", self.save_name))
        form.addRow("Save the model file to", folder_row)
        form.addRow("", self.save_register)
        form.addRow("System prompt to use with it", self.save_prompt)
        form.addRow("Context size", self.save_context)
        form.addRow("Model id", self.save_id)
        form.addRow("Model file", self.save_gguf)
        body.addLayout(form)
        self.save_name.textChanged.connect(self._update_save_preview)
        self.save_folder.textChanged.connect(self._update_save_preview)
        self.save_preview = _label(muted=True)
        body.addWidget(self.save_preview)
        self.save_button = _primary("Save model")
        self.save_button.clicked.connect(self._on_save)
        body.addLayout(_row(self.save_button))
        self.save_result = _label()
        body.addWidget(self.save_result)

        compare_title = QtWidgets.QLabel("Optional: compare it with the original model")
        compare_title.setObjectName("sectionTitle")
        body.addWidget(compare_title)
        self.compare_prompt = QtWidgets.QLineEdit()
        self.compare_prompt.setPlaceholderText("Ask something your dataset should have taught it...")
        self.compare_button = QtWidgets.QPushButton("Compare replies")
        self.compare_button.clicked.connect(self._on_compare)
        compare_row = QtWidgets.QHBoxLayout()
        compare_row.addWidget(self.compare_prompt, 1)
        compare_row.addWidget(self.compare_button)
        body.addLayout(compare_row)
        self.compare_before = _label()
        self.compare_after = _label()
        replies = QtWidgets.QHBoxLayout()
        for title, label in (("Before (original)", self.compare_before), ("After (fine-tuned)", self.compare_after)):
            column = QtWidgets.QVBoxLayout()
            column.addWidget(_label(title, muted=True))
            column.addWidget(label)
            replies.addLayout(column, 1)
        body.addLayout(replies)

        new = QtWidgets.QPushButton("Start a new fine-tune")
        new.clicked.connect(self._on_new)
        body.addLayout(_row(new))
        body.addStretch(1)
        return frame

    def open_run(self, run_dir):
        """Jump straight to step 4 for a finished run (used by the models table)."""
        meta = jobs.read_meta(run_dir)
        self.run_dir = Path(run_dir)
        self.run_done = True
        self.model = self.model or meta.get("base_model")
        self.default_system = meta.get("default_system", "")
        self.go(4)

    def _default_models_folder(self):
        data_dir = getattr(self._settings, "data_dir", None)
        return str(Path(data_dir) / "models") if data_dir else ""

    def _refresh_save_step(self):
        run_dir = Path(self.run_dir)
        files = jobs.run_files(run_dir)
        lines = [f"{run_dir.name} is saved in {run_dir}"]
        for name in files["gguf"]:
            size = (run_dir / name).stat().st_size / 1024 ** 2
            lines.append(f"- {name} ({size:,.0f} MB): the model file to chat with")
        if files["merged"]:
            lines.append(f"- merged_model ({jobs.folder_size(files['merged']) / 1024 ** 2:,.0f} MB): "
                         "full model, for further training")
        if files["lora"]:
            lines.append(f"- lora_adapter ({jobs.folder_size(files['lora']) / 1024 ** 2:,.0f} MB): "
                         "just the changes, to apply on top of the original")
        self.saved_where.setText("\n".join(lines))
        base = run_dir.name.rsplit("-", 2)[0] if run_dir.name.count("-") >= 2 else run_dir.name
        self.save_name.setText(base)
        self.save_id.setText(ftd.safe_name(base, "model").lower())
        if not self.save_folder.text():
            self.save_folder.setText(self._default_models_folder())
        self.save_prompt.setText(self.default_system)
        self.save_gguf.clear()
        self.save_gguf.addItems(files["gguf"])
        self.save_button.setEnabled(bool(files["gguf"]))
        self.compare_button.setEnabled(bool(files["merged"]))
        self.save_result.setText("" if files["gguf"] else "This run has no model file (.gguf) to save.")
        self._update_save_preview()

    def _update_save_preview(self, *_args):
        gguf = self.save_gguf.currentText()
        quant = jobs.quant_of(gguf) if gguf else ""
        name = ftd.safe_name(self.save_name.text(), "model")
        filename = f"{name}.{quant}.gguf" if quant else f"{name}.gguf"
        folder = self.save_folder.text().strip()
        self.save_preview.setText(f"Will save: {os.path.join(folder, filename)}" if folder else "")

    def _on_browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save the model file to", self.save_folder.text())
        if folder:
            self.save_folder.setText(os.path.normpath(folder))

    def _on_save(self):
        try:
            _dest, message = jobs.save_model(
                self._settings, run_dir=self.run_dir, gguf_name=self.save_gguf.currentText(),
                folder=self.save_folder.text(), name=self.save_name.text(),
                register=self.save_register.isChecked(), model_id=self.save_id.text(),
                context_size=self.save_context.value(), prompt=self.save_prompt.text())
        except (ValueError, OSError) as exc:
            self.save_result.setObjectName("error")
            self.save_result.setText(str(exc))
        else:
            self.save_result.setObjectName("")
            self.save_result.setText(message)
            if self.save_register.isChecked():
                self._on_model_saved()
        self.save_result.style().unpolish(self.save_result)
        self.save_result.style().polish(self.save_result)

    def _on_compare(self):
        prompt = self.compare_prompt.text().strip()
        merged = jobs.run_files(Path(self.run_dir))["merged"]
        base = jobs.read_meta(self.run_dir).get("base_model") or self.model
        if not prompt or merged is None or not base:
            return
        self.compare_button.setEnabled(False)
        self.compare_before.setText("Running the original model...")
        self.compare_after.setText("Then your fine-tuned model...")
        settings, system = self._settings, self.save_prompt.text()

        def done(result, error):
            self.compare_button.setEnabled(True)
            if error:
                self.compare_before.setText(f"Could not compare: {error}")
                self.compare_after.setText("")
                return
            self.compare_before.setText(result[0])
            self.compare_after.setText(result[1])

        self._run(lambda: jobs.compare(settings, base, str(merged), system, prompt), done)

    def _open_path(self, path):
        if path and Path(path).exists():
            return QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(path))))
        return False

    def _open_folder(self, folder):
        if folder and Path(folder).is_dir():
            return self._open_path(folder)
        return False

    def _on_new(self):
        self.records, self.dataset_path, self.dataset_rows = None, None, 0
        self.run_dir, self.run_done, self.train_job = None, False, None
        self.review.setVisible(False)
        self.save_result.setText("")
        self.compare_before.setText("")
        self.compare_after.setText("")
        self._show_train_form(True)
        self.go(1)
