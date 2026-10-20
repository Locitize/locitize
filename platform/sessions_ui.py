"""Native Qt Sessions page. Disk reads run off the GUI thread."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from session_core.config import provider_label
from session_service import SessionService
from session_launch import LOCAL_PROVIDERS


class JobSignals(QtCore.QObject):
    done = QtCore.Signal(object, object)


class Job(QtCore.QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = JobSignals()

    def run(self):
        try:
            self.signals.done.emit(self.fn(), None)
        except Exception as exc:
            self.signals.done.emit(None, str(exc))


class SessionsPage(QtWidgets.QWidget):
    def __init__(self, controller, service=None, parent=None):
        super().__init__(parent)
        self.controller = controller
        if service is None:
            from config import resolve_data_dir

            data = getattr(getattr(controller, "_settings", None), "data_dir", None)
            service = SessionService(data or resolve_data_dir())
        self.service = service
        self.sessions = []
        self.metadata = {}
        self._visible = []
        self._jobs = set()
        self._generation = 0
        self._scanning = False
        self._content_matches = None
        self._search_cancel = None
        self._visited = False
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        # The Code page that hosts this widget supplies the title and the one
        # "Release model" button, so neither is repeated here.
        hint = QtWidgets.QLabel("Find past coding sessions, keep notes, and continue in the right project.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        tools = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search titles, projects, models, tags and notes")
        self.search.setAccessibleName("Search sessions")
        self.search.textChanged.connect(self.query_changed)
        tools.addWidget(self.search, 1)
        self.content_btn = QtWidgets.QPushButton("Search transcripts")
        self.content_btn.clicked.connect(self.search_transcripts)
        tools.addWidget(self.content_btn)
        self.provider = QtWidgets.QComboBox()
        self.provider.addItem("All tools", "")
        for p in self.service.providers:
            self.provider.addItem(p.label, p.key)
        self.provider.currentIndexChanged.connect(self.render)
        tools.addWidget(self.provider)
        self.archived = QtWidgets.QCheckBox("Archived")
        self.archived.toggled.connect(self.render)
        tools.addWidget(self.archived)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        tools.addWidget(self.refresh_btn)
        self.new_btn = QtWidgets.QPushButton("New coding session")
        self.new_btn.setToolTip("Pick a coding tool, a model and a project folder, then open a terminal")
        self.new_btn.setObjectName("primaryButton")
        self.new_btn.clicked.connect(lambda: self.local_launch(None))
        tools.addWidget(self.new_btn)
        layout.addLayout(tools)
        split = QtWidgets.QSplitter()
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Session", "Tool", "Project", "Updated"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(1, 105)
        self.table.setColumnWidth(2, 160)
        self.table.setColumnWidth(3, 125)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        split.addWidget(self.table)
        inspector = QtWidgets.QWidget()
        details = QtWidgets.QVBoxLayout(inspector)
        self.detail = QtWidgets.QLabel("Select a session to preview its transcript.")
        self.detail.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.detail.setWordWrap(True)
        details.addWidget(self.detail)
        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Transcripts stay on this computer.")
        self.preview.setAccessibleName("Session transcript")
        details.addWidget(self.preview, 1)
        self.title = QtWidgets.QLineEdit()
        self.tags = QtWidgets.QLineEdit()
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setMaximumHeight(80)
        form = QtWidgets.QFormLayout()
        form.addRow("Title", self.title)
        form.addRow("Tags", self.tags)
        form.addRow("Notes", self.notes)
        details.addLayout(form)
        self.pinned = QtWidgets.QCheckBox("Pin session")
        details.addWidget(self.pinned)
        self.save_btn = QtWidgets.QPushButton("Save details")
        self.save_btn.clicked.connect(self.save_details)
        details.addWidget(self.save_btn)
        split.addWidget(inspector)
        split.setSizes([650, 450])
        layout.addWidget(split, 1)
        actions = QtWidgets.QHBoxLayout()
        self.resume_btn = QtWidgets.QPushButton("Resume original")
        self.local_btn = QtWidgets.QPushButton("Resume with local model")
        self.export_btn = QtWidgets.QPushButton("Export transcript")
        self.archive_btn = QtWidgets.QPushButton("Archive")
        for button in (self.resume_btn, self.local_btn, self.export_btn, self.archive_btn):
            actions.addWidget(button)
        self.resume_btn.clicked.connect(self.resume_original)
        self.local_btn.clicked.connect(lambda: self.local_launch(self.selected()))
        self.export_btn.clicked.connect(self.export_selected)
        self.archive_btn.clicked.connect(self.archive_selected)
        actions.addStretch(1)
        more = QtWidgets.QPushButton("Import / backup")
        menu = QtWidgets.QMenu(more)
        menu.addAction("Import Session Portal details", self.import_portal)
        menu.addAction("Back up session details", self.backup)
        menu.addAction("Restore session details", self.restore)
        more.setMenu(menu)
        actions.addWidget(more)
        layout.addLayout(actions)
        self.status = QtWidgets.QLabel("Your past coding sessions are listed here once they have been read.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.selection_changed()

    def submit(self, fn, callback):
        job = Job(fn)
        self._jobs.add(job)
        def done(value, error):
            self._jobs.discard(job)
            if error:
                self.status.setText(f"Could not complete this action: {error}")
            callback(value, error)
        job.signals.done.connect(done)
        self.pool.start(job)

    def visit(self):
        if not self._visited:
            self._visited = True
            self.refresh()

    def refresh(self):
        if self._scanning:
            return
        self._scanning = True
        self.refresh_btn.setEnabled(False)
        self.status.setText("Reading your past coding sessions...")
        def read():
            sessions, errors = self.service.scan()
            return sessions, errors, self.service.store.all()
        def finished(value, error):
            self._scanning = False
            self.refresh_btn.setEnabled(True)
            if not error:
                self.sessions, errors, self.metadata = value
                self.render()
                self.status.setText(f"{len(self.sessions)} local sessions. " + " ".join(errors))
        self.submit(read, finished)

    def selected(self):
        row = self.table.currentRow()
        return self._visible[row] if 0 <= row < len(self._visible) else None

    def render(self, *_):
        selected = self.selected()
        key = (selected.provider, selected.id) if selected else None
        query = self.search.text().casefold()
        visible = []
        for s in self.sessions:
            m = self.metadata.get((s.provider, s.id), {})
            if bool(m.get("archived")) != self.archived.isChecked():
                continue
            if self.provider.currentData() and s.provider != self.provider.currentData():
                continue
            blob = " ".join([s.display, s.project, s.model, *[str(v) for v in m.values()]]).casefold()
            if self._content_matches is not None:
                if (s.provider, s.id) not in self._content_matches:
                    continue
            elif query and query not in blob:
                continue
            visible.append(s)
        visible.sort(key=lambda s: (bool(self.metadata.get((s.provider, s.id), {}).get("pinned")), s.timestamp), reverse=True)
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self._visible = visible
        for row, s in enumerate(visible):
            self.table.insertRow(row)
            meta = self.metadata.get((s.provider, s.id), {})
            title = meta.get("title") or s.display or s.id
            if meta.get("pinned"):
                title = "[Pinned] " + title
            try:
                stamp = datetime.fromtimestamp(s.timestamp / 1000).strftime("%Y-%m-%d %H:%M")
            except (ValueError, OSError, OverflowError):
                stamp = "Unknown"
            for col, value in enumerate((title, provider_label(s.provider), meta.get("project") or s.project, stamp)):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
            if (s.provider, s.id) == key:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        self.selection_changed()

    def query_changed(self, *_):
        self._content_matches = None
        if self._search_cancel:
            self._search_cancel.set()
        self.render()

    def search_transcripts(self):
        import threading
        if self._search_cancel is not None:
            self._search_cancel.set()
            self.status.setText("Cancelling transcript search...")
            return
        query = self.search.text().strip().casefold()
        if not query:
            self.status.setText("Enter a phrase before searching transcripts.")
            return
        cancel = threading.Event()
        self._search_cancel = cancel
        self.content_btn.setText("Cancel search")
        self.status.setText("Searching your transcripts...")
        candidates = list(self.sessions)
        def find():
            matches, unavailable = set(), 0
            for s in candidates:
                if cancel.is_set():
                    break
                try:
                    if query in self.service.transcript(s).casefold():
                        matches.add((s.provider, s.id))
                except Exception:
                    unavailable += 1
            return matches, unavailable
        def finished(value, error):
            self.content_btn.setText("Search transcripts")
            if not cancel.is_set() and not error and self.search.text().strip().casefold() == query:
                self._content_matches, unavailable = value
                self.render()
                self.status.setText(f"{len(self._content_matches)} transcript matches ({unavailable} could not be read). Very long histories are only searched in part.")
            elif cancel.is_set():
                self.status.setText("Transcript search cancelled.")
            self._search_cancel = None
        self.submit(find, finished)

    def selection_changed(self):
        self._generation += 1
        generation = self._generation
        s = self.selected()
        pick = "Select a session in the list first."
        for button in (self.resume_btn, self.local_btn, self.export_btn, self.archive_btn, self.save_btn):
            button.setEnabled(s is not None)
            button.setToolTip("" if s is not None else pick)
        self.preview.clear()
        if s is None:
            self.title.clear()
            self.tags.clear()
            self.notes.clear()
            self.pinned.setChecked(False)
            self.detail.setText("Select a session to preview its transcript.")
            return
        meta = self.metadata.get((s.provider, s.id), {})
        self.title.setText(meta.get("title") or s.display)
        self.tags.setText(meta.get("tags", ""))
        self.notes.setPlainText(meta.get("notes", ""))
        self.pinned.setChecked(bool(meta.get("pinned")))
        self.archive_btn.setText("Restore from archive" if meta.get("archived") else "Archive")
        self.local_btn.setEnabled(s.provider in LOCAL_PROVIDERS and s.resumable)
        self.local_btn.setToolTip(
            "" if self.local_btn.isEnabled()
            else "This session cannot be continued with a local model."
        )
        self.resume_btn.setEnabled(s.resumable)
        self.resume_btn.setToolTip(
            "" if s.resumable else "This session has no saved state to resume."
        )
        self.detail.setText(f"{provider_label(s.provider)} | {meta.get('model_id') or s.model or 'Model not recorded'}\n{meta.get('project') or s.project}")
        self.preview.setPlainText("Loading transcript...")
        def finished(text, error):
            if generation == self._generation:
                self.preview.setPlainText(text if not error else "Transcript unavailable. Original files are unchanged.")
        self.submit(lambda: self.service.transcript(s), finished)

    def change(self, **changes):
        s = self.selected()
        if not s:
            return
        try:
            self.metadata[(s.provider, s.id)] = self.service.store.update(s.provider, s.id, **changes)
            self.render()
            self.status.setText("Session details saved. Original history is unchanged.")
        except Exception as exc:
            self.status.setText(f"Could not save details: {exc}")

    def save_details(self):
        self.change(title=self.title.text(), tags=self.tags.text(), notes=self.notes.toPlainText(), pinned=self.pinned.isChecked())

    def archive_selected(self):
        s = self.selected()
        if s:
            self.change(archived=not self.metadata.get((s.provider, s.id), {}).get("archived", False))

    def local_launch(self, session, tool=None):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Continue with locitize" if session else "New coding session")
        form = QtWidgets.QFormLayout(dialog)
        provider = QtWidgets.QComboBox()
        for key in LOCAL_PROVIDERS:
            provider.addItem(provider_label(key), key)
        if session:
            provider.setCurrentIndex(provider.findData(session.provider))
            provider.setEnabled(False)
        elif tool and provider.findData(tool) >= 0:
            provider.setCurrentIndex(provider.findData(tool))
        models = QtWidgets.QComboBox()
        models.setObjectName("localLaunchModel")
        for row in self.controller.list_models():
            if row.get("launchable"):
                models.addItem(row.get("name", row["id"]), row["id"])
        meta = self.metadata.get((session.provider, session.id), {}) if session else {}
        running_model = ""
        running_model_id = getattr(self.controller, "running_model_id", None)
        if callable(running_model_id):
            running_model = running_model_id() or ""
        saved = models.findData(
            meta.get("model_id")
            or (session.model if session else "")
            or running_model
        )
        if saved >= 0:
            models.setCurrentIndex(saved)
        last_project = getattr(self.controller, "last_project_dir", lambda: "")()
        folder = QtWidgets.QLineEdit(
            meta.get("project") or (session.project if session else "") or last_project
        )
        folder.setObjectName("localLaunchProject")
        browse = QtWidgets.QPushButton("Choose folder")
        def pick():
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                dialog, "Project folder", last_project
            )
            if chosen:
                folder.setText(chosen)
        browse.clicked.connect(pick)
        form.addRow("Coding tool", provider)
        form.addRow("Local model", models)
        form.addRow("Project", folder)
        form.addRow("", browse)
        remember = QtWidgets.QCheckBox("Remember this project folder")
        remember.setObjectName("localLaunchRemember")
        remember.setChecked(True)
        form.addRow("", remember)
        notice = QtWidgets.QLabel("The selected model will start before the terminal opens. The coding tool retains its own permissions and network behavior. Keep locitize open while working.")
        notice.setWordWrap(True)
        form.addRow(notice)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        ok_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)

        def refresh_ok(_text=""):
            raw = folder.text().strip()
            path = Path(raw) if raw else None
            valid = bool(path is not None and path.is_absolute() and path.is_dir())
            ok_button.setEnabled(models.count() > 0 and valid)
            ok_button.setToolTip(
                "" if valid else "Choose an existing project folder using its full path."
            )

        folder.textChanged.connect(refresh_ok)
        refresh_ok()
        form.addRow(buttons)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            payload = {"choice": provider.currentData(), "model_id": models.currentData(),
                       "project_dir": folder.text(), "resume_id": session.id if session else "",
                       "remember": remember.isChecked()}
            self.controller.request_session_launch(payload)
            self.status.setText("Preparing the local model and coding terminal...")

    def resume_original(self):
        s = self.selected()
        if not s:
            return
        from session_core.resume import launch
        folder = self.metadata.get((s.provider, s.id), {}).get("project", "")
        self.submit(lambda: launch(self.service.original_command(s, folder)),
                    lambda _, err: self.status.setText("Terminal opened using the tool's original backend settings.") if not err else None)

    def export_selected(self):
        s = self.selected()
        if not s:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export transcript", "session.md", "Markdown (*.md)")
        if path:
            self.submit(lambda: self.service.export(s, Path(path)),
                        lambda _, err: self.status.setText("Transcript exported.") if not err else None)

    def backup(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Back up details", "sessions.json", "JSON (*.json)")
        if path:
            self.submit(lambda: self.service.store.backup(Path(path)),
                        lambda n, err: self.status.setText(f"Backed up {n} session annotations.") if not err else None)

    def restore(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Restore session details", "", "JSON (*.json)")
        if path:
            self.submit(lambda: self.service.store.restore(Path(path)), lambda _, err: self.refresh() if not err else None)

    def import_portal(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Session Portal's data folder")
        if folder:
            self.submit(lambda: self.service.store.import_portal(Path(folder), self.sessions), lambda _, err: self.refresh() if not err else None)

    def apply_launch_result(self, result):
        if result.ok and result.payload.get("released"):
            self.status.setText("Model reservation released. Existing terminals are still open; their model can now be switched.")
            return
        self.status.setText(result.payload.get("warning") or "Coding terminal opened. Keep locitize running while this session uses its model." if result.ok else result.error or "Launch failed")

    def release_model(self):
        from gui_controller import Command

        self.controller.command_q.put(Command("release_sessions"))
