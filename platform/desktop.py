"""PySide6 presentation shell for the LOCITIZE desktop command center.

This module is the M12 view layer. It builds a dark, sidebar-driven native
window and binds every control to the existing gui_controller intents. The
controller owns all work, queues, validation, and lifecycle behavior; this file
only renders its Result objects on the Qt GUI thread through one QTimer drain.
"""

import html
import sys
import tempfile
import threading
import time as _time
from pathlib import Path

import gui_controller
import setup_plan
import sysmon
from PySide6 import QtCore, QtGui, QtWidgets

# Owner request 2026-08-21: the real LOCITIZE mark, not a plain-text stand-in.
# Ships inside Codebase/platform (not the vault-level Assets/ folder) so the
# app stays self-contained if someone clones just Codebase/ - the same reason
# every other static resource here is loaded relative to this file.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_ICON_PATH = ASSETS_DIR / "locitize_icon_1024.png"
LOGO_ICO_PATH = ASSETS_DIR / "locitize_launcher.ico"
# Owner request 2026-08-23: the square icon (2-row "LoCi/TiZe", needed for
# small-size legibility as an app/taskbar icon) reads as visually cramped when
# used in the header, which has no square constraint and real horizontal room.
# This is a separate, wide, single-row rendering of the same wordmark just for
# the header - the app/taskbar icon (LOGO_ICON_PATH/LOGO_ICO_PATH) is untouched.
HEADER_LOGO_PATH = ASSETS_DIR / "locitize_header_wordmark.png"

# Security review 2026-08-21: installing and running a remote script is a
# real, security-relevant action, so every place that offers to install a
# harness (the first-run onboarding dialog and the Chat picker's own
# "Install" button) shows the SAME exact command before the owner opts in,
# rather than a vague "we'll set it up" - one source so the two dialogs
# can't drift apart and say different things about what actually runs.
HARNESS_INSTALL_SUMMARIES = {
    "claude": "irm https://claude.ai/install.ps1 | iex  (official Claude Code installer)",
    "codex": "irm https://chatgpt.com/codex/install.ps1 | iex  (official Codex installer)",
    "opencode": (
        "npm i -g opencode-ai  (installs Node.js LTS via winget first if "
        "missing - that step may show a Windows admin consent (UAC) "
        "prompt; this is the ONLY one of the three that can)"
    ),
}

# Owner request 2026-08-21: a real timestamped record of one launch's stages
# (process start, window construction, first show, first actual paint), so an
# intermittent "black flash on launch" report can be diagnosed from hard
# timing data instead of trying to catch a sub-second artifact in a manual
# screenshot. Overwritten each run (mode "w" on first write) so it always
# reflects the MOST RECENT launch, never a stale one from days ago. Lives
# beside the single-instance lock file - same rationale (OS temp dir, no
# dependency on the data root being resolved yet).
LAUNCH_LOG_PATH = Path(tempfile.gettempdir()) / "locitize_desktop_launch.log"
_launch_log_started = False


def _log_launch(message: str) -> None:
    """Append one timestamped line to LAUNCH_LOG_PATH. Never raises - a
    diagnostic log that could crash the app it is diagnosing would defeat
    its own purpose."""
    global _launch_log_started
    try:
        mode = "w" if not _launch_log_started else "a"
        with open(LAUNCH_LOG_PATH, mode, encoding="utf-8") as handle:
            handle.write(f"{_time.monotonic():.4f}  {message}\n")
        _launch_log_started = True
    except OSError:
        pass

# Single-sourced from modelhub so the sentence the user consents to in the GUI is
# byte-for-byte the sentence the download core enforces (M14.14.4). Importing the
# constant costs nothing and opens no connection - modelhub makes network calls
# only inside its three explicit-action methods.
from modelhub import UNVERIFIED_CONSENT_TEXT, VERIFICATION_LABELS, hub_job_id

# DEFECT-QA-M14-3 sizing floors, in device-independent pixels.
#
# Qt gives a QTableWidget no meaningful minimum height of its own, so a table in
# a space-starved layout shrinks until its viewport is zero pixels tall - rows
# present, rows unreachable. These two numbers are the floors that stop that.
# A hub row renders at 30 px and the header at about 26 px, so the header plus
# three rows (the size of the shipped built-in catalog) needs ~116 px; 124
# leaves room for the frame.
HUB_TABLE_MIN_HEIGHT = 124
# Owner request 2026-08-21: a floor of only four rows forced a scrollbar over
# the model inventory on every real registry (the owner's machine has a dozen+
# rows) even in a maximized window, because the surrounding layout's stretch
# distribution was not reliably handing the table the extra space a big window
# has available. The table has vertical stretch, so on a large/maximized window
# it still grows to show the whole inventory with no inner scrollbar. This floor
# only sets how tall it stays on a SMALL window: owner request 2026-08-29, it was
# a 14-row floor, which - added to the controls, monitor, and the "Get models"
# strip below - overflowed a normal (non-maximized) window and forced the whole
# page into an outer scrollbar (the "long page"). An 8-row floor lets the page
# fit a normal window, with the table's own inner scrollbar handling overflow
# there, which is the right place for the scroll rather than the whole page.
TABLE_ROW_HEIGHT_PX = 30
MODEL_TABLE_MIN_HEIGHT = 26 + TABLE_ROW_HEIGHT_PX * 8

# The Thinking box's do-nothing first item (owner request 2026-09-02). A named
# constant, not a literal in two places, because it is compared against as well
# as displayed - a drifting copy would silently turn "as registered" into an
# effort level and get rejected by the model's chat template at request time.
THINKING_AS_REGISTERED = "as registered"

# Human labels for on-disk Kokoro voice ids. Unknown ids fall back to the
# suffix after af_/am_/bf_/bm_. The combo stores the id as userData.
VOICE_DISPLAY_NAMES = {
    "af_heart": "Heart",
    "af_bella": "Bella",
    "af_nicole": "Nicole",
    "af_sarah": "Sarah",
    "af_sky": "Sky",
    "am_adam": "Adam",
    "am_fenrir": "Fenrir",
    "am_michael": "Michael",
    "am_puck": "Puck",
    "bf_emma": "Emma",
    "bf_isabella": "Isabella",
    "bm_george": "George",
}


def voice_display_name(voice_id):
    """Return the human label for an on-disk Kokoro voice id."""
    key = str(voice_id or "").strip()
    if not key:
        return ""
    if key in VOICE_DISPLAY_NAMES:
        return VOICE_DISPLAY_NAMES[key]
    if "_" in key:
        return key.split("_", 1)[1].replace("_", " ").title()
    return key


def selected_voice_id(combo):
    """On-disk voice id from a combo that displays human names."""
    data = combo.currentData()
    if data:
        return str(data)
    return combo.currentText()

# "Fine-tune" sits immediately after "Models" because both are about the model
# registry and lifecycle; the sidebar remains the only mode selector (M13.2).
PAGE_NAMES = (
    "Home",
    "Models",
    "Sessions",
    "Fine-tune",
    "Talk",
    "Voice Setup",
    "Vision",
    "Memory",
    "Chat",
    "Settings",
)


APP_STYLE = """
QWidget {
    background: #1e1f22;
    color: #e8eaed;
    /* M17.15 (owner request "fonts way better"): prefer Segoe UI Variable, the
       refined Windows 11 system face - crisper hinting and a truer weight range
       than plain Segoe UI - and fall back through it on older Windows. */
    font-family: "Segoe UI Variable Text", "Segoe UI", "Segoe UI Symbol", Arial, sans-serif;
    font-size: 13px;
}
QMainWindow, QWidget#shell { background: #1e1f22; }
/* M15.9 (owner request 2026-08-29: "cleaner, fonts to match, empty spaces").
   The global QWidget rule above paints EVERY widget's background - including
   labels and checkboxes sitting on #2b2d30 panels, which rendered as darker
   bars behind their own text on every page (most visible behind headings and
   status lines). Text-bearing widgets carry no background of their own; the
   chip styles below still win via their objectName selectors. */
QLabel, QCheckBox, QRadioButton { background: transparent; }
QFrame#header, QFrame#sidebar, QFrame#panel {
    background: #2b2d30;
    border: 1px solid #35383d;
    border-radius: 8px;
}
/* M15.9: the header wordmark is TEXT in the app's own face again, not the
   italic script PNG - one typeface everywhere was the explicit ask. Tracking
   is set in code (QSS has no letter-spacing). */
/* M17.15: titles use the Display cut of Segoe UI Variable, drawn for large text
   (tighter, more even than the Text cut at big sizes), with heavier weights for
   a clearer type ramp. */
QLabel#brand { font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 18px; font-weight: 800; color: #ffffff; }
QLabel#pageTitle { font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 24px; font-weight: 700; color: #ffffff; }
/* A heading INSIDE a page (e.g. "Get models") used pageTitle's 24px, giving
   two competing page titles on one screen. Section headings get their own
   step on the type ramp: clearly a heading, clearly subordinate. */
QLabel#sectionTitle { font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 18px; font-weight: 700; color: #ffffff; }
/* M15.9: the Chat page's centered action card - a deliberately framed focal
   surface, not a stray global-background rectangle. */
QWidget#focusCard {
    background: #26282c;
    border: 1px solid #35383d;
    border-radius: 10px;
}
QLabel#muted { color: #a7aeb8; }
QLabel#error { color: #ff7b72; }
/* Owner request 2026-08-21: was a flat #202329 fill on a #2b2d30 panel - darker
   than its own background, which reads as a sunken/pressed well, not a badge.
   A top-lighter-to-bottom-darker gradient plus a lighter top border sells a
   raised, button-like bevel instead, without adding a hover/pressed state
   (these are never clickable - only real QPushButtons get those). Padding
   widened from 4px/10px so longer status strings (health/spec lines) get
   breathing room instead of crowding the border. */
QLabel#statusChip {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #40454e, stop:1 #33373e);
    color: #dce3f2;
    border: 1px solid #4c5364;
    border-top-color: #5a6275;
    border-radius: 12px;
    padding: 6px 14px;
    font-weight: 600;
}
/* Same raised look as statusChip, but the sidebar's fixed 176px width has no
   room for that chip's 14px side padding plus "COMMAND CENTER" - it clipped
   to "COMMAND CENTE". Smaller font and tighter padding fit it instead of
   widening the whole sidebar for one caption. */
QLabel#sidebarChip {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #40454e, stop:1 #33373e);
    color: #dce3f2;
    border: 1px solid #4c5364;
    border-top-color: #5a6275;
    border-radius: 10px;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 600;
}
QListWidget#sidebarList {
    background: transparent;
    border: 0;
    outline: 0;
    padding: 6px;
}
QListWidget#sidebarList::item {
    border-radius: 6px;
    margin: 2px 0;
    padding: 11px 14px;
}
QListWidget#sidebarList::item:hover { background: #34373c; }
QListWidget#sidebarList::item:selected {
    background: #284b7a;
    color: #ffffff;
    font-weight: 600;
}
/* Owner request 2026-08-21: every secondary button (Search, Download, Register,
   Refresh, Stop, ...) now carries the same raised gradient/pill look the header's
   Offload GPU button and the statusChip badges use - one consistent button
   language app-wide instead of Offload GPU being a one-off exception. */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #40454e, stop:1 #33373e);
    border: 1px solid #4c5364;
    border-top-color: #5a6275;
    border-radius: 12px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a5058, stop:1 #3b3f47);
    border-color: #5b6377;
}
QPushButton:pressed { background: #33373e; }
QPushButton:disabled { background: #292b2f; color: #6f7379; border-color: #323438; }
/* Owner request 2026-08-21: the Get models collapse toggle was autoRaise (no
   visible chrome until hovered) - the same raised gradient as every button,
   sized as a small square badge since it holds only an arrow glyph. */
QToolButton#collapseToggle {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #40454e, stop:1 #33373e);
    border: 1px solid #4c5364;
    border-top-color: #5a6275;
    border-radius: 8px;
    padding: 4px;
}
QToolButton#collapseToggle:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a5058, stop:1 #3b3f47);
    border-color: #5b6377;
}
QToolButton#collapseToggle:pressed { background: #33373e; }
QPushButton#primaryButton { background: #4f8cff; border-color: #4f8cff; color: #ffffff; }
QPushButton#primaryButton:hover { background: #67a0ff; }
/* Primary specificity otherwise overrides the general disabled selector. */
QPushButton#primaryButton:disabled {
    background: #292b2f;
    color: #6f7379;
    border-color: #323438;
}
/* Owner request 2026-08-21: the Delete button (permanently removes a model's
   file from disk) gets a visibly distinct red-tinted look, not the same
   gradient as every reversible action beside it. */
QPushButton#dangerButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5a3438, stop:1 #472a2d);
    border: 1px solid #7a4448;
    border-top-color: #8f5155;
    color: #ffd6d6;
}
QPushButton#dangerButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6b3c40, stop:1 #523033);
    border-color: #925157;
}
QPushButton#dangerButton:pressed { background: #472a2d; }
QPushButton#dangerButton:disabled {
    background: #292b2f;
    color: #6f7379;
    border-color: #323438;
}
QPushButton#talkButton {
    background: #4f8cff;
    border: 2px solid #73a6ff;
    border-radius: 16px;
    color: #ffffff;
    font-size: 19px;
    font-weight: 700;
    min-height: 88px;
}
QPushButton#talkButton:disabled { background: #303b4b; color: #8894a8; border-color: #3d4859; }
QPushButton#talkButton[talkState="listening"] {
    background: #3d8f6b;
    border-color: #5fd4a0;
}
QPushButton#talkButton[talkState="thinking"] {
    background: #5a6275;
    border-color: #7b8496;
}
QPushButton#talkButton[talkState="speaking"] {
    background: #c47a2c;
    border-color: #e8a85a;
}
QFrame#talkPulse {
    background: #4f8cff;
    border: none;
    border-radius: 3px;
    min-height: 6px;
    max-height: 6px;
}
QFrame#talkPulse[talkState="listening"] { background: #5fd4a0; }
QFrame#talkPulse[talkState="thinking"] { background: #7b8496; }
QFrame#talkPulse[talkState="speaking"] { background: #e8a85a; }
QFrame#talkPulse[talkState="idle"] { background: #35383d; }
QLabel#visionPreview {
    background: #232529;
    border: 1px solid #43474e;
    border-radius: 8px;
    min-height: 180px;
    max-height: 240px;
}
QLineEdit, QSpinBox, QComboBox, QTextEdit, QTableWidget {
    background: #232529;
    border: 1px solid #43474e;
    border-radius: 6px;
    padding: 7px;
    selection-background-color: #3c6fb3;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QTextEdit:focus, QTableWidget:focus {
    border-color: #4f8cff;
}
/* M15.9 pass 3 (owner request 2026-08-29: "no rectangular shapes anywhere").
   QComboBox and QSpinBox are COMPLEX controls: styling only their frame makes
   Qt keep the native sub-controls - a sharp-cornered drop-down section and a
   boxed platform arrow inside the themed pill, which is exactly the artifact
   the owner screenshotted on the Voice combo. Every sub-control is therefore
   specified; the arrows are drawn with the CSS border-triangle technique so no
   image asset is involved. */
QComboBox { padding: 6px 30px 6px 10px; }
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
    background: transparent;
}
QComboBox::down-arrow {
    image: url("@UI@/chevron-down.png");
    width: 10px;
    height: 6px;
    margin-right: 8px;
}
QComboBox::down-arrow:on { image: url("@UI@/chevron-down-bright.png"); }
QComboBox QAbstractItemView {
    background: #26282c;
    color: #e8eaed;
    border: 1px solid #43474e;
    border-radius: 6px;
    selection-background-color: #284b7a;
    selection-color: #ffffff;
    outline: 0;
    padding: 4px;
}
QSpinBox { padding-right: 24px; }
QSpinBox::up-button, QSpinBox::down-button {
    subcontrol-origin: border;
    width: 20px;
    border: none;
    background: transparent;
}
QSpinBox::up-button { subcontrol-position: top right; margin-top: 2px; }
QSpinBox::down-button { subcontrol-position: bottom right; margin-bottom: 2px; }
QSpinBox::up-arrow { image: url("@UI@/chevron-up.png"); width: 10px; height: 6px; }
QSpinBox::down-arrow { image: url("@UI@/chevron-down.png"); width: 10px; height: 6px; }
/* The checkbox indicator was the native platform square for the same reason. */
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #4c5364;
    border-radius: 4px;
    background: #232529;
}
QCheckBox::indicator:hover { border-color: #5b6377; }
QCheckBox::indicator:checked {
    background: #4f8cff;
    border-color: #4f8cff;
    image: url("@UI@/check.png");
}
QCheckBox::indicator:disabled { background: #292b2f; border-color: #323438; }
/* The square button where a table's header row meets its vertical header. */
QTableCornerButton::section { background: #303238; border: 0; }
QHeaderView::section {
    background: #303238;
    /* M17.15: brighter header text and a touch more padding so column titles
       (Repository, Publisher, Fits your GPU?, ...) read as clear labels. */
    color: #e3e7ee;
    border: 0;
    border-right: 1px solid #41444a;
    padding: 8px 10px;
    font-weight: 650;
}
/* M17.15: breathing room in every table cell so rows are not cramped. */
QTableWidget::item { padding: 4px 8px; }
QTableWidget { gridline-color: #34373c; alternate-background-color: #26282c; }
/* Owner request 2026-08-21: 7px all around read as too much air once the
   Models table grew to 8 ResizeToContents columns - padding compounds into
   column WIDTH (ResizeToContents measures content + this padding), so a flat
   cut here tightens every column's spacing, not just row height. */
QTableWidget::item { padding: 4px 8px; }
QCheckBox { spacing: 7px; }
QScrollBar:vertical { background: #24262a; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #484c53; border-radius: 5px; min-height: 28px; }
QScrollBar:horizontal { background: #24262a; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: #484c53; border-radius: 5px; min-width: 28px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
/* M15.9: QProgressBar had NO rule, so the hub download bar rendered in the
   platform's native light chrome - a white-bordered box on a dark panel. */
QProgressBar {
    background: #232529;
    border: 1px solid #43474e;
    border-radius: 6px;
    text-align: center;
    color: #cfd4dc;
    min-height: 16px;
}
QProgressBar::chunk { background: #4f8cff; border-radius: 5px; }
QToolTip { background: #2b2d30; color: #ffffff; border: 1px solid #555a63; }
"""

# M15.9 pass 3: the arrow/check glyphs are image assets (assets/ui/*,
# committed) because Qt draws QSS border-triangle "arrows" as their element box
# - a solid rectangle, the exact artifact this pass exists to remove. The @UI@
# token becomes this file's own assets/ui directory, so the stylesheet works
# from any install location.
APP_STYLE = APP_STYLE.replace(
    "@UI@", (Path(__file__).resolve().parent / "assets" / "ui").as_posix()
)


# --------------------------------------------------------------------------- #
# Text-rendering policy (SEC-M14-1)
#
# Qt's default text format is AutoText: a widget SNIFFS its own content and, if
# it looks like HTML, renders the WHOLE string as rich text. LOCITIZE displays text
# it did not write - HuggingFace repo ids, licence tags, model file names,
# server error messages, model output, run metadata - so AutoText hands a third
# party control of the layout and emphasis of LOCITIZE's own dialogs. That was
# SEC-M14-1: a licence tag could bury the "not verified" line and bold a forged
# "Checksum verified..." claim in the download consent dialog.
#
# The rule is therefore a POLICY applied to every text widget, not a fix at the
# one site that was reported: nothing in this view renders rich text. Nothing in
# LOCITIZE's own copy uses markup, so this costs no formatting.
# --------------------------------------------------------------------------- #

PLAIN_TEXT = QtCore.Qt.TextFormat.PlainText


def plain_message_box(
    parent,
    title,
    text,
    *,
    icon=QtWidgets.QMessageBox.Icon.NoIcon,
    buttons=QtWidgets.QMessageBox.StandardButton.Ok,
    default=None,
):
    """Build a QMessageBox whose body is rendered literally, never as markup.

    Constructed rather than using the `QMessageBox.question/information`
    convenience calls, because those give no opportunity to set the text format
    before the dialog is shown.
    """
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    # Order matters only for readability; setTextFormat applies at paint time.
    box.setTextFormat(PLAIN_TEXT)
    box.setText(text)
    box.setStandardButtons(buttons)
    if default is not None:
        box.setDefaultButton(default)
    return box


def harden_text_rendering(root):
    """Force every text widget under `root` to plain-text rendering.

    Applied to the main window after it is built and to every dialog this view
    creates, so a widget added later inherits the policy by construction instead
    of by the author remembering. QLabel and QMessageBox honour setTextFormat;
    QTextEdit has no text-format property, so its rich-text acceptance is turned
    off and callers use `append_plain` / `setPlainText` to write into it.
    """
    for label in root.findChildren(QtWidgets.QLabel):
        label.setTextFormat(PLAIN_TEXT)
    for box in root.findChildren(QtWidgets.QMessageBox):
        box.setTextFormat(PLAIN_TEXT)
    for edit in root.findChildren(QtWidgets.QTextEdit):
        edit.setAcceptRichText(False)
    if isinstance(root, QtWidgets.QLabel):
        root.setTextFormat(PLAIN_TEXT)
    elif isinstance(root, QtWidgets.QMessageBox):
        root.setTextFormat(PLAIN_TEXT)
    elif isinstance(root, QtWidgets.QTextEdit):
        root.setAcceptRichText(False)
    return root


def plain_tooltip_text(text):
    """Escape text so a tooltip shows it literally.

    Tooltips are the one surface with no text-format property: QToolTip always
    sniffs. Escaping is the equivalent defence - if Qt then treats the string as
    rich text it decodes the entities back to the exact characters, and if it
    does not, the escaped form has nothing to interpret either.
    """
    return html.escape(str(text or ""), quote=False)


def append_plain(edit, text):
    """Append one line to a QTextEdit without rich-text auto-detection.

    QTextEdit.append() runs the same AutoText sniff as QLabel, so appending a
    transcript segment or a model reply containing "<b>" would render it as
    formatting. Inserting at the end as plain text is the same visible result
    for LOCITIZE's own copy and literal for everyone else's.
    """
    edit.moveCursor(QtGui.QTextCursor.MoveOperation.End)
    if edit.toPlainText():
        edit.insertPlainText("\n")
    edit.insertPlainText(str(text))
    edit.moveCursor(QtGui.QTextCursor.MoveOperation.End)


class _Sparkline(QtWidgets.QWidget):
    """A small live area-graph of a 0..1 series (M17.18), custom-painted.

    Holds a rolling window of fractions (0 = empty, 1 = full) and draws a filled
    line rising left-to-right, newest on the right - the shape a task manager's
    usage graph has. Custom QPainter rather than a chart library so the look is
    fully controlled and it carries no extra dependency.
    """

    _MAX_POINTS = 180

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._values: list[float] = []
        self._color = QtGui.QColor(color)
        self.setMinimumHeight(96)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def push(self, fraction: float) -> None:
        self._values.append(max(0.0, min(1.0, float(fraction))))
        if len(self._values) > self._MAX_POINTS:
            self._values = self._values[-self._MAX_POINTS :]
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        # Rounded inset panel behind the graph, matching the app's card surfaces.
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#26282c"))
        painter.drawRoundedRect(rect, 8, 8)

        width = float(rect.width())
        height = float(rect.height())
        # Faint 25/50/75% gridlines so the level is readable at a glance.
        grid = QtGui.QColor("#34373c")
        painter.setPen(QtGui.QPen(grid, 1))
        for frac in (0.25, 0.5, 0.75):
            y = height - frac * height
            painter.drawLine(0, int(y), int(width), int(y))

        values = self._values
        if len(values) < 2:
            painter.end()
            return

        count = len(values)
        points = [
            QtCore.QPointF(width * i / (count - 1), height - v * height)
            for i, v in enumerate(values)
        ]
        # Filled area under the line.
        area = QtGui.QPainterPath()
        area.moveTo(0.0, height)
        for point in points:
            area.lineTo(point)
        area.lineTo(width, height)
        area.closeSubpath()
        fill = QtGui.QColor(self._color)
        fill.setAlpha(64)
        painter.fillPath(area, fill)
        # The line itself.
        line = QtGui.QPainterPath()
        line.moveTo(points[0])
        for point in points[1:]:
            line.lineTo(point)
        painter.setPen(QtGui.QPen(self._color, 2))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(line)
        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    """One LOCITIZE desktop window backed by an injected GuiController."""

    def __init__(self, controller, health="unknown"):
        super().__init__()
        self._gc = controller
        self._health = health
        self._ui = gui_controller.UiState()
        self._models = self._load_model_rows()
        self._selected_id = None
        # Owner preference: make the fastest measured model visible first.
        # The shared sorter always keeps unbenchmarked rows at the bottom.
        self._sort_col = "benchmark_tok_s"
        self._sort_desc = True
        self._assistant_live = False
        self._assistant_port = None
        self._talk_state = "idle"
        self._listen_when_ready = False
        self._memory_visited = False
        self._closed = False
        # The download job this panel is currently showing, or None when nothing
        # is in flight. Set before the widgets exist because table signals fire
        # during page construction (DEFECT-QA-M14-2).
        #
        # NEW-QA-M14-7: "active" means CONFIRMED RUNNING - a job id is adopted
        # here only once the controller has published progress for it. A request
        # that has been sent but not yet answered sits in _hub_pending_job
        # instead. The two were one field before, and that is precisely why the
        # job-id guard in _apply_hub_done could not fire: a second request
        # overwrote the live job's id with its own before the controller had
        # decided anything, so the refusal that came back compared EQUAL to the
        # "active" job and the guard was skipped - killing Cancel and the
        # progress bar of a transfer that was still moving bytes.
        self._hub_active_job = None
        self._hub_pending_job = None
        # Owner request 2026-08-21 (black-flash-on-launch diagnostic): logged
        # once each, the first time this window is actually shown/painted -
        # see showEvent/paintEvent below and LAUNCH_LOG_PATH.
        self._logged_first_show = False
        self._logged_first_paint = False

        _log_launch("MainWindow.__init__ start")
        self.setWindowTitle("locitize Desktop")
        self.setMinimumSize(960, 680)
        self.resize(1180, 780)
        self._build_shell()
        # SEC-M14-1: apply the plain-text policy once, to the finished tree, so
        # every label and text view in the window is covered whether or not its
        # author remembered - including the ones that display HuggingFace text.
        harden_text_rendering(self)
        self._refresh_model_rows()
        self._select_initial_model()
        self._refresh_buttons()

        # The timer is the only consumer of controller outcomes. Since it belongs
        # to this window, timeout and every widget update run on the Qt GUI thread.
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._drain)
        self._timer.start()

        # M17.18: background RAM/VRAM sampler feeding the System monitor page. A
        # daemon thread (nvidia-smi/psutil off the UI thread); it posts samples
        # onto result_q, drained like every other Result. Started here, stopped
        # in closeEvent.
        self._sysmon_stop = threading.Event()
        self._sysmon_thread = threading.Thread(
            target=self._sysmon_loop, name="locitize-sysmon", daemon=True
        )
        self._sysmon_thread.start()

        # First paint of the Fine-tune page. Both the state snapshot and the scan
        # touch the filesystem (the studio checkout and the outputs tree, either of
        # which may sit on a disconnected drive), so both go through the queue and
        # arrive as Results. The page already shows its neutral Stopped state from
        # _build_finetune_page, so nothing here blocks the Qt thread waiting.
        if hasattr(self._gc, "request_finetune_state"):
            self._gc.request_finetune_state()
        if hasattr(self._gc, "request_scan_finetunes"):
            self._on_finetune_rescan()

        # First paint of the Get models section. This asks for the SHIPPED
        # CATALOG, which is a disk read - it is the only hub request the window
        # ever makes on its own, and it cannot reach the network (egress rule
        # HF-1). Search, repo selection and Download are the only three paths to
        # huggingface.co, and all three require a press.
        if self._hub_enabled and hasattr(self._gc, "request_hub_catalog"):
            self._gc.request_hub_catalog()
        _log_launch("MainWindow.__init__ done (widget tree built, not shown yet)")

    def showEvent(self, event):
        """Log the first real show (owner request 2026-08-21 diagnostic)."""
        super().showEvent(event)
        if not self._logged_first_show:
            self._logged_first_show = True
            _log_launch("MainWindow.showEvent (first)")

    def paintEvent(self, event):
        """Log the first real paint - the moment actual pixels land on
        screen, which is what a "black flash" is a gap BEFORE (owner request
        2026-08-21 diagnostic)."""
        super().paintEvent(event)
        if not self._logged_first_paint:
            self._logged_first_paint = True
            _log_launch("MainWindow.paintEvent (first)")

    def _load_model_rows(self):
        """Model rows for the table: the merged view when the controller has one.

        list_models_merged() is the M13 addition that appends discovered
        fine-tunes after the manual rows. Falling back to list_models() keeps this
        view working against any controller that predates M13.
        """
        loader = getattr(self._gc, "list_models_merged", None) or self._gc.list_models
        return loader()

    def _build_shell(self):
        """Build the fixed header, sidebar navigation, and page stack."""
        shell = QtWidgets.QWidget()
        shell.setObjectName("shell")
        root = QtWidgets.QVBoxLayout(shell)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(self._build_header())

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(12)
        content.addWidget(self._build_sidebar())

        self._stack = QtWidgets.QStackedWidget()
        self._stack.setObjectName("pageStack")
        self.pages = {}
        builders = (
            self._build_home_page,
            self._build_models_page,
            self._build_sessions_page,
            self._build_finetune_page,
            self._build_talk_page,
            self._build_voice_page,
            self._build_vision_page,
            self._build_memory_page,
            self._build_chat_page,
            self._build_settings_page,
        )
        for name, builder in zip(PAGE_NAMES, builders, strict=True):
            page = builder()
            page.setObjectName(f"page{name}")
            self.pages[name] = page
            self._stack.addWidget(page)
        content.addWidget(self._stack, 1)
        root.addLayout(content, 1)
        self.setCentralWidget(shell)

        self._sidebar.currentRowChanged.connect(self._on_sidebar_changed)
        # Home connects model setup, existing work and chat.
        self._sidebar.setCurrentRow(PAGE_NAMES.index("Home"))

    def _build_home_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(28, 28, 28, 28)
        title = QtWidgets.QLabel("Your AI. Your models. Your work.")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        description = QtWidgets.QLabel("Run a local model, start a conversation, or return to a coding session.")
        description.setWordWrap(True)
        layout.addWidget(description)
        for heading, text, target in (
            ("Choose a model", "Download, start and measure models on this computer.", "Models"),
            ("Continue your work", "Find local coding sessions, preview transcripts and resume a project.", "Sessions"),
            ("Start a conversation", "Open chat or launch a coding tool with the running model.", "Chat"),
        ):
            card = QtWidgets.QGroupBox(heading)
            box = QtWidgets.QHBoxLayout(card)
            label = QtWidgets.QLabel(text)
            label.setWordWrap(True)
            box.addWidget(label, 1)
            button = QtWidgets.QPushButton("Open " + target)
            button.clicked.connect(lambda _checked=False, name=target: self._sidebar.setCurrentRow(PAGE_NAMES.index(name)))
            box.addWidget(button)
            layout.addWidget(card)
        info = QtWidgets.QLabel("Local inference stays on this computer. Downloads and external coding tools have their own network behavior. Voice, vision and fine-tuning are optional.")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch(1)
        about = QtWidgets.QPushButton("About and diagnostics")
        about.clicked.connect(self._show_release_diagnostics)
        layout.addWidget(about)
        return page

    def _build_sessions_page(self):
        from sessions_ui import SessionsPage

        self._sessions_page = SessionsPage(self._gc)
        return self._sessions_page

    def _show_release_diagnostics(self):
        from release_info import VERSION, diagnostics

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("locitize " + VERSION)
        layout = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(diagnostics())
        layout.addWidget(text)
        copy = QtWidgets.QPushButton("Copy diagnostics")
        copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(text.toPlainText()))
        layout.addWidget(copy)
        dialog.resize(600, 400)
        dialog.exec()

    def _build_header(self):
        """Build the always-visible brand and live platform status line."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("header")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(18, 12, 18, 12)
        # Owner decision 2026-08-29: the LOGO IMAGE stays. The M15.9 pass
        # briefly replaced it with a text mark, reading "fonts to match" as
        # covering the wordmark - the owner corrected that: a logo is brand
        # identity, not a UI font, and the 2026-08-23 image-only request
        # stands. Degrades honestly (nothing shown, not a broken-image icon)
        # if the asset is ever missing.
        if HEADER_LOGO_PATH.is_file():
            logo_pixmap = QtGui.QPixmap(str(HEADER_LOGO_PATH)).scaledToHeight(
                32, QtCore.Qt.TransformationMode.SmoothTransformation
            )
            logo_label = QtWidgets.QLabel()
            logo_label.setPixmap(logo_pixmap)
            layout.addWidget(logo_label)
        layout.addStretch(1)
        # Host RAM/GPU specs (owner request): static hardware facts, so this reads
        # the controller's cached system_specs() once at build time rather than on
        # every pump tick.
        self._specs_label = QtWidgets.QLabel(
            gui_controller.format_system_specs(self._gc.system_specs())
        )
        self._specs_label.setObjectName("statusChip")
        layout.addWidget(self._specs_label)
        self._header_status = QtWidgets.QLabel()
        self._header_status.setObjectName("statusChip")
        self._header_status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._header_status)
        # Global quick-unload: frees GPU VRAM from any page without navigating to
        # Models first. Reuses the same stop path/state as the Models page Stop
        # button (request_stop() always targets whatever model is running, not
        # whatever is merely selected), so behavior stays identical either way.
        self._header_offload_btn = QtWidgets.QPushButton("Offload GPU")
        self._header_offload_btn.setObjectName("headerOffloadButton")
        self._header_offload_btn.setToolTip(
            "Free GPU memory held by locitize: stop the running model and clear "
            "any leftover locitize servers (a crashed session or a measurement "
            "probe). Other apps are shown in the result but never touched."
        )
        self._header_offload_btn.clicked.connect(self._on_offload_gpu)
        layout.addWidget(self._header_offload_btn)
        return frame

    def _build_sidebar(self):
        """Build the compact navigation rail; selection drives the page stack."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("sidebar")
        frame.setFixedWidth(176)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(8, 12, 8, 12)
        # Owner request 2026-08-21: was plain "muted" caption text; the same
        # raised-chip badge as the header's status pills, so the sidebar picks
        # up the same visual language instead of looking bare above the nav list.
        caption = QtWidgets.QLabel("locitize")
        caption.setObjectName("sidebarChip")
        caption.setContentsMargins(0, 0, 0, 0)
        caption_row = QtWidgets.QHBoxLayout()
        caption_row.setContentsMargins(4, 0, 4, 6)
        caption_row.addWidget(caption)
        caption_row.addStretch(1)
        layout.addLayout(caption_row)
        self._sidebar = QtWidgets.QListWidget()
        self._sidebar.setObjectName("sidebarList")
        self._sidebar.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        for name in PAGE_NAMES:
            self._sidebar.addItem(name)
        layout.addWidget(self._sidebar, 1)
        # Owner request 2026-08-21: "Local. Private. Yours." removed - a bare
        # tagline with no context of its own at the bottom of the nav rail.
        return frame

    @staticmethod
    def _attach_empty_state(table, text):
        """Overlay a centered hint on `table` whenever it has no rows (M15.9).

        An empty inventory used to render as a bare dark void under its own
        headers - technically honest, visually unfinished, and the largest
        remaining "empty space" after the first UI pass. The label lives on the
        table's viewport, re-centers itself on resize via an event filter, and
        tracks emptiness through the table model's OWN signals, so no
        population site anywhere has to remember to toggle it.
        """
        label = QtWidgets.QLabel(text, table.viewport())
        label.setObjectName("muted")
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setTextFormat(PLAIN_TEXT)
        # The viewport paints the table background itself; the label must not
        # re-paint the global widget background over it.
        label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        def _recenter(*_args):
            # RuntimeError guard: Qt destroys the C++ table before the Python
            # wrapper during window teardown, and modelReset fires one last
            # time into these slots. A dying widget needs no re-layout.
            try:
                area = table.viewport().rect()
                width = min(max(area.width() - 48, 120), 560)
                label.resize(width, label.heightForWidth(width) + 8)
                label.move(
                    (area.width() - label.width()) // 2,
                    (area.height() - label.height()) // 2,
                )
            except RuntimeError:
                pass

        def _update(*_args):
            try:
                label.setVisible(table.rowCount() == 0)
            except RuntimeError:
                return
            _recenter()

        class _ResizeWatch(QtCore.QObject):
            def eventFilter(self, _obj, event):
                if event.type() == QtCore.QEvent.Type.Resize:
                    _recenter()
                return False

        watch = _ResizeWatch(table)
        table.viewport().installEventFilter(watch)
        model = table.model()
        model.rowsInserted.connect(_update)
        model.rowsRemoved.connect(_update)
        model.modelReset.connect(_update)
        _update()
        return label

    @staticmethod
    def _page(title, subtitle):
        """Return a consistently styled page and its body layout."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(12)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("pageTitle")
        layout.addWidget(title_label)
        subtitle_label = QtWidgets.QLabel(subtitle)
        subtitle_label.setObjectName("muted")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)
        return page, layout

    def _titled_panel(self, title, subtitle):
        """A page-style column: a pageTitle and its muted description ABOVE a
        panel box (M17.17). Every column on the split Models page - the inventory
        and Get models - is built this way, so both read identically: title on
        top, one-line description under it, then the bordered box of content.
        Returns the column widget (to drop into the splitter) and the panel's
        body layout (to fill with that column's content)."""
        column = QtWidgets.QWidget()
        stack = QtWidgets.QVBoxLayout(column)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(12)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("pageTitle")
        stack.addWidget(title_label)
        subtitle_label = QtWidgets.QLabel(subtitle)
        subtitle_label.setObjectName("muted")
        subtitle_label.setWordWrap(True)
        stack.addWidget(subtitle_label)
        panel, body = self._panel()
        stack.addWidget(panel, 1)
        return column, body

    @staticmethod
    def _scrollable(page):
        """Wrap a page so its content stays reachable in a short window.

        DEFECT-QA-M14-3: the Models page carries two panels whose combined
        natural height exceeds the app's own default window at 1180x780. With no
        scroll area anywhere on the page (QA measured `scroll_areas: 0`) the only
        way to reach the lower panel was to resize the window - so a new user's
        first screen showed an empty inventory above two clipped, empty-looking
        tables. Wrapping the page means the layout can honour the minimum heights
        above and hand the overflow to a scrollbar instead of to the user.
        """
        area = QtWidgets.QScrollArea()
        area.setObjectName("pageScroll")
        area.setWidgetResizable(True)  # the page still stretches to fill a big window
        area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        area.setWidget(page)
        return area

    @staticmethod
    def _panel():
        """Create a card-like frame and inner vertical layout."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("panel")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        return frame, layout

    def _build_models_page(self):
        """Build model inventory, lifecycle controls, benchmark, and monitor."""
        # M17.17: no single page-level title. Each column (the inventory and Get
        # models) carries its OWN title + description above its own box, so the
        # two halves read identically (owner request) instead of one page title
        # over a section title buried inside the other box.
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(12)
        left_col, body = self._titled_panel(
            "Models",
            "Start, switch, stop, benchmark, and inspect the models already "
            "registered on this machine.",
        )
        # M13: Source sits right after Model so the owner can tell a manually
        # registered model from a discovered fine-tune without scrolling.
        # Capabilities (owner request 2026-08-21) sits right next to Model - the
        # "which one do I use" question is the whole point of this column, so it
        # belongs beside the name, not buried after Status.
        self._model_table = QtWidgets.QTableWidget(0, 9)
        self._model_table.setHorizontalHeaderLabels(
            [
                "Model",
                "Capabilities",
                "Source",
                "Size",
                "On GPU",
                "On CPU",
                "Last gen tok/s",
                "Context",
                "Status",
            ]
        )
        self._model_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._model_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._model_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._model_table.setAlternatingRowColors(True)
        self._model_table.verticalHeader().setVisible(False)
        header = self._model_table.horizontalHeader()
        # M17.19 (owner request): columns are user-RESIZABLE (Interactive), each
        # with a sensible default width - the Model column wide enough to read a
        # full name - and no maximum cap, so a name can be dragged wider as far as
        # wanted. The last section (Status) still stretches to absorb any leftover
        # width on a wide window; if the chosen widths exceed the panel the table
        # gets its OWN horizontal scrollbar (below), so a squeezed name is now the
        # owner's choice to widen, not a hard clip.
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        default_widths = {
            0: 240,  # Model - wide enough for a full name by default
            1: 110,  # Capabilities
            2: 96,   # Source
            3: 80,   # Size
            4: 92,   # On GPU
            5: 92,   # On CPU
            6: 104,  # Last gen tok/s
            7: 84,   # Context
        }
        for column, width in default_widths.items():
            if column < self._model_table.columnCount():
                header.resizeSection(column, width)
        # With resizable columns the total can exceed the panel; let the table
        # scroll horizontally on demand rather than clip.
        self._model_table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        header.sectionClicked.connect(self._on_sort)
        self._model_table.currentCellChanged.connect(self._on_select_model)
        # DEFECT-QA-M14-3: a floor of roughly a header plus four rows. Without
        # one this table would happily shrink to nothing inside the scroll area
        # below, and an inventory with no visible rows is not an inventory.
        self._model_table.setMinimumHeight(MODEL_TABLE_MIN_HEIGHT)
        self._attach_empty_state(
            self._model_table,
            "No models registered yet.\n"
            "Use Get models below to download one, or Register a file you "
            "already have.",
        )
        body.addWidget(self._model_table, 1)

        controls = QtWidgets.QHBoxLayout()
        # Owner request 2026-09-02: pick a thinking level for THIS launch, the
        # same one-launch override the terminal menu takes as "3 low". Editable
        # on purpose: the accepted levels belong to the model's own chat
        # template (Qwen3.8 takes xhigh/medium/low, gpt-oss takes low/medium/
        # high), so a fixed dropdown would be wrong for half the registry. The
        # listed items are the common cases; anything the grammar accepts can be
        # typed. Index 0 is the do-nothing default, so Start behaves exactly as
        # it always has unless the owner changes this.
        self._thinking_combo = QtWidgets.QComboBox()
        self._thinking_combo.setEditable(True)
        self._thinking_combo.addItems(
            [THINKING_AS_REGISTERED, "off", "low", "medium", "xhigh", "low/2048"]
        )
        self._thinking_combo.setToolTip(
            "Thinking level for the next Start, without changing your model "
            "list. 'off' disables thinking, a level such as 'low' sets the "
            "effort, and 'low/2048' also caps thinking at 2048 tokens. Levels "
            "come from the model's own chat template."
        )
        controls.addWidget(QtWidgets.QLabel("Thinking:"))
        controls.addWidget(self._thinking_combo)
        self._start_btn = QtWidgets.QPushButton("Start")
        self._start_btn.setObjectName("primaryButton")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QtWidgets.QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._chat_btn = QtWidgets.QPushButton("Chat")
        self._chat_btn.setToolTip("Open the chat UI for the running model")
        self._chat_btn.clicked.connect(self._on_chat)
        self._refresh_models_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_models_btn.setToolTip(
            "Reload your model list so new downloads appear"
        )
        self._refresh_models_btn.clicked.connect(self._on_refresh_models)
        # M13: promote the selected DISCOVERED fine-tune into models.yaml. Only
        # enabled for a discovered row - a registered row has nothing to promote.
        self._register_btn = QtWidgets.QPushButton("Register")
        self._register_btn.setToolTip(
            "Add the selected fine-tune to your model list"
        )
        self._register_btn.clicked.connect(self._on_register_selected)
        # Owner request 2026-08-21: permanently delete a model's .gguf (and
        # mmproj, if it has one) from disk, not just drop it from the list.
        # #dangerButton gives it a visibly different (red-tinted) look from
        # every other button here on purpose - the one control on this page
        # that cannot be undone with another click.
        self._delete_btn = QtWidgets.QPushButton("Delete")
        self._delete_btn.setObjectName("dangerButton")
        self._delete_btn.setToolTip(
            "Permanently delete this model's file(s) from disk and remove it "
            "from the list"
        )
        self._delete_btn.clicked.connect(self._on_delete_model)
        # Owner request 2026-08-22: work out this model's largest usable context
        # window automatically instead of hand-editing models.yaml per model.
        # Opt-in per model on purpose - it performs several real model loads, so
        # it must be something the owner asks for, never a side effect of
        # selecting a row.
        # Auto-tune and Benchmark are one action: tuning finds the context,
        # then benchmarks the model at it, so the speed shown is the speed at
        # the setting the model will actually run with.
        self._autotune_btn = QtWidgets.QPushButton("Auto-tune")
        self._autotune_btn.setToolTip(
            "Find the largest context that really loads on this machine, write "
            "it back with the right rope-scaling flags, then benchmark the model "
            "at that context to measure its real tokens per second. Takes a few "
            "minutes and starts the model several times."
        )
        self._autotune_btn.clicked.connect(self._on_autotune_context)
        self._autotune_all_btn = QtWidgets.QPushButton("Auto-tune all")
        self._autotune_all_btn.setToolTip(
            "Auto-tune every model on your list, one after another: find each "
            "one's largest context and measure its speed. Takes a few minutes "
            "per model; Stop cancels the rest."
        )
        self._autotune_all_btn.clicked.connect(self._on_autotune_all)
        for button in (
            self._start_btn,
            self._stop_btn,
            self._chat_btn,
            self._autotune_btn,
            self._autotune_all_btn,
            self._register_btn,
            self._delete_btn,
            self._refresh_models_btn,
        ):
            controls.addWidget(button)
        controls.addStretch(1)
        body.addLayout(controls)
        # M15.9: these two chips lived at the end of the seven-button row above,
        # and their combined minimum width pushed the page past the viewport -
        # the ONLY horizontal scrollbar in the app. A status row of their own
        # keeps the raised-chip look (owner request 2026-08-21) and lets the
        # button row fit at every window width.
        status_row = QtWidgets.QHBoxLayout()
        self._models_total = QtWidgets.QLabel("Total: -")
        self._models_total.setObjectName("statusChip")
        status_row.addWidget(self._models_total)
        self._model_status = QtWidgets.QLabel("ready")
        self._model_status.setObjectName("statusChip")
        # M18.8 (owner report: clicking Offload GPU resized the window): a long
        # status line ("freed 1 server; ...; still held by other apps: ...")
        # raised this label's preferred width, and Qt grew the WINDOW to honour
        # it. Ignored horizontal policy means the chip takes whatever width the
        # row gives it and never demands more - long text clips inside the chip
        # (the full line is set as a tooltip where it matters) and the window
        # geometry never moves because of a status message.
        self._model_status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        status_row.addWidget(self._model_status, 1)
        body.addLayout(status_row)

        self._models_monitor = QtWidgets.QLabel("idle")
        self._models_monitor.setObjectName("statusChip")
        self._models_monitor.setWordWrap(True)
        body.addWidget(self._models_monitor)

        # Auto-tune's own readout (owner request 2026-08-22). A separate label
        # from _model_status and _models_monitor because it is the only thing on
        # this page that runs for minutes: it needs room for a running trial log
        # and a multi-part final summary, and it must not be overwritten by the
        # monitor's once-per-tick metrics line. Hidden until a run starts, so it
        # costs nothing visually the rest of the time.
        self._autotune_status = QtWidgets.QLabel("")
        self._autotune_status.setObjectName("statusChip")
        self._autotune_status.setWordWrap(True)
        self._autotune_status.setVisible(False)
        body.addWidget(self._autotune_status)
        # Two UI-thread-only flags backing the Stop button's auto-tune meaning.
        # _autotune_pending covers the gap between the click and the ops worker
        # actually picking the command up - without it Stop would be dead for
        # that first moment, which is precisely when an owner who clicked by
        # mistake wants it. _autotune_canceling latches the "I have asked" state
        # so the button and the status line stop inviting a second click.
        self._autotune_pending = False
        self._autotune_canceling = False
        # True from an "Auto-tune all" click until its final result: each
        # model's own result must not end the run in the UI.
        self._autotune_batch = False
        # M17.14 (owner request): the inventory and "Get models" sit SIDE BY SIDE,
        # each claiming half the width, instead of stacked one above the other -
        # so the page is one screen wide instead of one long vertical scroll. A
        # QSplitter lets the owner drag the divider to give either half more room;
        # setChildrenCollapsible(False) keeps both halves always visible. The page
        # stays wrapped in the scroll area (DEFECT-QA-M14-3): on a short window the
        # split content is still reachable via the page scrollbar rather than
        # clipping the catalog out of existence.
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setObjectName("modelsSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left_col)
        splitter.addWidget(self._build_get_models_panel())
        # M17.20: near-even split (owner request - the 60/40 lean left the
        # inventory over-wide and squeezed Get models). Columns are resizable now,
        # so the table no longer needs the extra room; a slight lean toward the
        # inventory remains, and both halves grow equally on window resize. Still
        # draggable.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1050, 950])  # ~52/48 toward the inventory; draggable
        layout.addWidget(splitter, 1)
        return self._scrollable(page)

    def _build_get_models_panel(self):
        """Build the M14.14 "Get models" section that lives under the model list.

        Deliberately a second panel on the EXISTING Models page rather than a new
        nav entry: finding a model and running a model are the same task, and a
        stranger who has just installed LOCITIZE lands here with an empty list and
        needs the next step in front of them.

        Nothing in this method touches the network. The page paints from the
        shipped catalog on disk; the only three controls that can cause egress
        are Search, selecting a search result, and Download (egress rule HF-1).
        """
        # M17.17: same column shape as the inventory - "Get models" title and a
        # one-line description ABOVE the box, then the box of controls - so the
        # two halves of the page read identically. The title/description now live
        # outside the panel (was a sectionTitle + blurb inside it).
        col, body = self._titled_panel(
            "Get models",
            "Search huggingface.co for a model. locitize "
            "contacts huggingface.co only when you press Search, pick a "
            "repository, or confirm a download - never on its own.",
        )
        self._hub_enabled = bool(getattr(self._gc, "hub_enabled", lambda: True)())

        search_row = QtWidgets.QHBoxLayout()
        self._hub_query = QtWidgets.QLineEdit()
        self._hub_query.setPlaceholderText("Search huggingface.co for a GGUF model")
        # Enter is wired to the same handler as the button, not to a live-search
        # signal: textChanged must never reach the network (HF-1).
        self._hub_query.returnPressed.connect(self._on_hub_search)
        self._hub_search_btn = QtWidgets.QPushButton("Search")
        self._hub_search_btn.clicked.connect(self._on_hub_search)
        search_row.addWidget(self._hub_query, 1)
        search_row.addWidget(self._hub_search_btn)
        body.addLayout(search_row)

        self._hub_repo_table = QtWidgets.QTableWidget(0, 4)
        self._attach_empty_state(
            self._hub_repo_table,
            "Search huggingface.co above.",
        )
        self._hub_repo_table.setHorizontalHeaderLabels(
            ["Repository", "Publisher", "Licence", "Where from"]
        )
        self._prepare_hub_table(self._hub_repo_table)
        self._hub_repo_table.currentCellChanged.connect(self._on_hub_repo_selected)
        body.addWidget(self._hub_repo_table)

        # Owner request 2026-08-21: a search is capped per page (settings.yaml's
        # models_hub.search_limit); Load more fetches huggingface.co's own next-page
        # cursor (the Link: rel="next" response header) rather than raising the cap
        # unboundedly. Hidden until a search reports a further page exists.
        self._hub_has_more = False
        load_more_row = QtWidgets.QHBoxLayout()
        self._hub_load_more_btn = QtWidgets.QPushButton("Load more results")
        self._hub_load_more_btn.setVisible(False)
        self._hub_load_more_btn.clicked.connect(self._on_hub_search_more)
        load_more_row.addWidget(self._hub_load_more_btn)
        load_more_row.addStretch(1)
        body.addLayout(load_more_row)

        self._hub_file_table = QtWidgets.QTableWidget(0, 5)
        self._attach_empty_state(
            self._hub_file_table,
            "Select a repository to list its downloadable files.",
        )
        self._hub_file_table.setHorizontalHeaderLabels(
            ["File", "Quant", "Size", "Fits your GPU?", "Checksum"]
        )
        self._prepare_hub_table(self._hub_file_table)
        self._hub_file_table.currentCellChanged.connect(self._on_hub_file_selected)
        body.addWidget(self._hub_file_table)

        self._hub_fit = QtWidgets.QLabel("")
        self._hub_fit.setObjectName("muted")
        self._hub_fit.setWordWrap(True)
        body.addWidget(self._hub_fit)

        # M17.15: one tidy row - the two buttons grouped together, then a clear
        # gap, then the checkbox, then stretch pushes the lot to the left so it
        # reads as one organized line instead of scattered controls.
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        self._hub_download_btn = QtWidgets.QPushButton("Download")
        self._hub_download_btn.setObjectName("primaryButton")
        self._hub_download_btn.setEnabled(False)
        self._hub_download_btn.clicked.connect(self._on_hub_download)
        self._hub_cancel_btn = QtWidgets.QPushButton("Cancel")
        self._hub_cancel_btn.setToolTip("Cancel the download in progress")
        self._hub_cancel_btn.setEnabled(False)
        self._hub_cancel_btn.clicked.connect(self._on_hub_cancel)
        self._hub_register_box = QtWidgets.QCheckBox("Add to my model list when finished")
        self._hub_register_box.setChecked(True)
        controls.addWidget(self._hub_download_btn)
        controls.addWidget(self._hub_cancel_btn)
        controls.addSpacing(16)
        controls.addWidget(self._hub_register_box)
        controls.addStretch(1)
        body.addLayout(controls)

        self._hub_progress = QtWidgets.QProgressBar()
        self._hub_progress.setRange(0, 100)
        self._hub_progress.setValue(0)
        self._hub_progress.setVisible(False)
        body.addWidget(self._hub_progress)

        self._hub_status = QtWidgets.QLabel("ready")
        self._hub_status.setObjectName("muted")
        self._hub_status.setWordWrap(True)
        body.addWidget(self._hub_status)

        # M17.18 (owner request): the live RAM/VRAM monitor lives HERE, filling
        # the empty lower area of the Get models box, instead of a separate page.
        self._add_system_meters(body)

        if not self._hub_enabled:
            # The honest disabled state: say which setting turns it back on, and
            # make every control that could reach the network unusable.
            self._hub_status.setText(
                "Model downloading is off. Re-run setup (locitize.vbs --setup) "
                "or enable it in your settings file to turn it on."
            )
            for widget in (
                self._hub_query,
                self._hub_search_btn,
                self._hub_load_more_btn,
                self._hub_download_btn,
                self._hub_cancel_btn,
            ):
                widget.setEnabled(False)
        return col

    @staticmethod
    def _prepare_hub_table(table):
        """Apply the shared read-only single-selection table styling."""
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        # DEFECT-QA-M14-3: the maximum alone let this table collapse to a zero
        # height viewport whenever the page was short of space, which is what
        # happened at the app's own default window size: the three built-in
        # catalog rows existed, were counted in the status line, and could not be
        # seen or reached. The minimum is the header plus three rows, so the
        # built-in catalog is always visible; the page scrolls when the window
        # cannot fit everything (see _scrollable).
        table.setMinimumHeight(HUB_TABLE_MIN_HEIGHT)
        table.setMaximumHeight(150)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in range(1, table.columnCount()):
            header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )

    def _add_system_meters(self, body):
        """Append the live RAM + VRAM monitor to `body` (M17.18).

        Owner request: no separate page - the monitor fills the empty lower area
        of the Get models box. Two meters (RAM, GPU VRAM), each a label + value
        chip, a fill bar, and a rolling area graph. The numbers arrive as
        'system_sample' Results posted by a background sampler thread (nvidia-smi
        + psutil off the UI thread), so the graphs update live as a model loads
        without stuttering the window.
        """
        divider = QtWidgets.QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #35383d;")
        body.addWidget(divider)
        heading = QtWidgets.QLabel("System memory")
        heading.setObjectName("sectionTitle")
        body.addWidget(heading)

        def meter(title, color):
            head = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(title)
            head.addWidget(label)
            head.addStretch(1)
            value = QtWidgets.QLabel("-")
            value.setObjectName("statusChip")
            head.addWidget(value)
            body.addLayout(head)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            body.addWidget(bar)
            spark = _Sparkline(color)
            body.addWidget(spark)
            return value, bar, spark

        self._sys_ram_value, self._sys_ram_bar, self._sys_ram_spark = meter(
            "RAM", "#4f8cff"
        )
        self._sys_vram_value, self._sys_vram_bar, self._sys_vram_spark = meter(
            "GPU VRAM", "#43d17a"
        )
        self._sys_vram_caption = QtWidgets.QLabel(
            "VRAM in use includes the running model plus anything else on the GPU. "
            "Load a model on the Models page and watch this climb."
        )
        self._sys_vram_caption.setObjectName("muted")
        self._sys_vram_caption.setWordWrap(True)
        body.addWidget(self._sys_vram_caption)

    def _apply_system_sample(self, payload):
        """Render one live RAM/VRAM sample onto the System page (GUI thread)."""
        sample = sysmon.SystemSample(
            ram_used_mb=payload.get("ram_used_mb", 0.0),
            ram_total_mb=payload.get("ram_total_mb", 0.0),
            vram_used_mb=payload.get("vram_used_mb"),
            vram_total_mb=payload.get("vram_total_mb"),
        )
        self._sys_ram_value.setText(
            f"{sysmon.format_gb(sample.ram_used_mb)} / "
            f"{sysmon.format_gb(sample.ram_total_mb)}  ({sample.ram_pct:.0f}%)"
        )
        self._sys_ram_bar.setValue(int(sample.ram_pct))
        self._sys_ram_spark.push(sample.ram_pct / 100.0)
        if sample.has_gpu:
            self._sys_vram_value.setText(
                f"{sysmon.format_gb(sample.vram_used_mb)} / "
                f"{sysmon.format_gb(sample.vram_total_mb)}  ({sample.vram_pct:.0f}%)"
            )
            self._sys_vram_bar.setValue(int(sample.vram_pct))
            self._sys_vram_spark.push(sample.vram_pct / 100.0)
        else:
            self._sys_vram_value.setText("no NVIDIA GPU")

    def _sysmon_loop(self):
        """Background sampler: every ~1.5s post a live RAM/VRAM sample (M17.18).

        Runs off the UI thread because nvidia-smi is a subprocess; results reach
        the UI through the same result_q the rest of the app drains. Waits BEFORE
        the first sample so a short-lived test window never posts, and stops
        promptly when the window closes."""
        import health

        gpu = health.NvidiaSmiGpuInfoProvider()
        sys_provider = health.DefaultSystemInfoProvider()
        while not self._sysmon_stop.wait(1.5):
            try:
                sample = sysmon.gather(gpu, sys_provider)
                self._gc.result_q.put(
                    gui_controller.Result(
                        "system_sample",
                        True,
                        {
                            "ram_used_mb": sample.ram_used_mb,
                            "ram_total_mb": sample.ram_total_mb,
                            "vram_used_mb": sample.vram_used_mb,
                            "vram_total_mb": sample.vram_total_mb,
                        },
                    )
                )
            except Exception:  # noqa: BLE001 - a sampling glitch must not kill the thread
                continue

    def _build_finetune_page(self):
        """Build the M13 Fine-tune page: studio lifecycle plus discovered models.

        Two panels, both assembled from the existing shell primitives (no new QSS
        class, color, or widget type is introduced): Panel A owns the studio's
        start/stop/open lifecycle and its status chip, Panel B lists the fine-tuned
        .gguf files discovered under the studio's outputs/ folder. Discovery works
        whether or not the studio is running, which is the common case.
        """
        page, layout = self._page(
            "Fine-tune",
            "Run the fine-tuning studio and serve the models it has already "
            "produced. The studio opens in your normal browser.",
        )

        # --- Panel A: studio lifecycle ---
        studio, studio_body = self._panel()
        studio_body.addWidget(QtWidgets.QLabel("Fine-tune studio"))
        self._ft_chip = QtWidgets.QLabel("Stopped")
        self._ft_chip.setObjectName("statusChip")
        self._ft_chip.setWordWrap(True)
        self._ft_chip.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        studio_body.addWidget(self._ft_chip)

        controls = QtWidgets.QHBoxLayout()
        self._ft_start_btn = QtWidgets.QPushButton("Start")
        self._ft_start_btn.setObjectName("primaryButton")
        self._ft_start_btn.clicked.connect(self._on_finetune_start)
        self._ft_stop_btn = QtWidgets.QPushButton("Stop")
        self._ft_stop_btn.clicked.connect(self._on_finetune_stop)
        self._ft_open_btn = QtWidgets.QPushButton("Open in browser")
        self._ft_open_btn.clicked.connect(self._on_finetune_open)
        for button in (self._ft_start_btn, self._ft_stop_btn, self._ft_open_btn):
            controls.addWidget(button)
        controls.addStretch(1)
        studio_body.addLayout(controls)

        # The honesty warning lives in its own label so it can be shown on its own
        # line without ever replacing the lifecycle state text.
        self._ft_warning = QtWidgets.QLabel("")
        self._ft_warning.setObjectName("error")
        self._ft_warning.setWordWrap(True)
        self._ft_warning.setVisible(False)
        studio_body.addWidget(self._ft_warning)

        self._ft_log_label = QtWidgets.QLabel("")
        self._ft_log_label.setObjectName("muted")
        self._ft_log_label.setWordWrap(True)
        self._ft_log_label.setVisible(False)
        studio_body.addWidget(self._ft_log_label)
        layout.addWidget(studio)

        # --- Panel B: discovered models ---
        found, found_body = self._panel()
        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(QtWidgets.QLabel("Fine-tuned models"))
        header_row.addStretch(1)
        self._ft_rescan_btn = QtWidgets.QPushButton("Rescan")
        self._ft_rescan_btn.clicked.connect(self._on_finetune_rescan)
        header_row.addWidget(self._ft_rescan_btn)
        found_body.addLayout(header_row)

        self._ft_table = QtWidgets.QTableWidget(0, 6)
        self._ft_table.setHorizontalHeaderLabels(
            ["Name", "Source", "Quant", "Size", "Modified", "Status"]
        )
        self._ft_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._ft_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._ft_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._ft_table.setAlternatingRowColors(True)
        self._ft_table.verticalHeader().setVisible(False)
        ft_header = self._ft_table.horizontalHeader()
        self._attach_empty_state(
            self._ft_table,
            "No fine-tuned models discovered yet.\n"
            "Start the studio and train one, then Rescan.",
        )
        ft_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in range(1, self._ft_table.columnCount()):
            ft_header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        self._ft_table.currentCellChanged.connect(self._on_select_finetune)
        found_body.addWidget(self._ft_table, 1)

        row_actions = QtWidgets.QHBoxLayout()
        self._ft_serve_btn = QtWidgets.QPushButton("Serve")
        self._ft_serve_btn.setToolTip(
            "Start the selected fine-tune through the normal model-serving path"
        )
        self._ft_serve_btn.clicked.connect(self._on_finetune_serve)
        self._ft_register_btn = QtWidgets.QPushButton("Register")
        self._ft_register_btn.setToolTip("Add the selected fine-tune to your model list")
        self._ft_register_btn.clicked.connect(self._on_finetune_register)
        self._ft_folder_btn = QtWidgets.QPushButton("Open folder")
        self._ft_folder_btn.clicked.connect(self._on_finetune_open_folder)
        # M18.7 (owner request): a run is deletable from the page it lives on.
        # dangerButton styling + a blocking confirmation, because a run is hours
        # of GPU time and this removes the whole folder from disk.
        self._ft_delete_btn = QtWidgets.QPushButton("Delete")
        self._ft_delete_btn.setObjectName("dangerButton")
        self._ft_delete_btn.setToolTip(
            "Permanently delete this run's folder (adapter, merged model, GGUF) "
            "from disk"
        )
        self._ft_delete_btn.clicked.connect(self._on_finetune_delete)
        for button in (self._ft_serve_btn, self._ft_register_btn,
                       self._ft_delete_btn, self._ft_folder_btn):
            row_actions.addWidget(button)
        row_actions.addStretch(1)
        self._ft_status = QtWidgets.QLabel("")
        self._ft_status.setObjectName("muted")
        self._ft_status.setWordWrap(True)
        row_actions.addWidget(self._ft_status)
        found_body.addLayout(row_actions)
        layout.addWidget(found, 1)

        # First paint: render the lifecycle state synchronously, then ask the ops
        # worker for the scan so the filesystem walk never runs on the GUI thread.
        self._finetunes = []
        self._selected_finetune_id = None
        self._apply_finetune_state({"status": "stopped", "reason": "", "url": None})
        self._refresh_finetune_buttons()
        return page

    def _build_talk_page(self):
        """Build the assistant page around one mic control."""
        page, layout = self._page(
            "Talk",
            "Click Talk and speak. Talking over the reply also cuts in. "
            "After each answer it listens again.",
        )

        session, session_layout = self._panel()
        row = QtWidgets.QHBoxLayout()
        self._assistant_start_btn = QtWidgets.QPushButton("Start assistant")
        self._assistant_start_btn.setObjectName("primaryButton")
        self._assistant_start_btn.clicked.connect(self._on_start_assistant)
        self._assistant_end_btn = QtWidgets.QPushButton("End")
        self._assistant_end_btn.clicked.connect(self._on_end_assistant)
        self._assistant_end_btn.setEnabled(False)
        self._interrupt_btn = QtWidgets.QPushButton("Interrupt")
        self._interrupt_btn.clicked.connect(self._on_interrupt)
        self._interrupt_btn.setEnabled(False)
        self._assistant_speak = QtWidgets.QCheckBox("Speak replies")
        self._assistant_speak.setChecked(True)
        self._talk_voice = QtWidgets.QComboBox()
        self._populate_voice_combo(self._talk_voice)
        self._talk_voice.currentIndexChanged.connect(self._sync_voice_from_talk)
        row.addWidget(self._assistant_start_btn)
        row.addWidget(self._assistant_end_btn)
        row.addWidget(self._interrupt_btn)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Voice"))
        row.addWidget(self._talk_voice)
        row.addWidget(self._assistant_speak)
        session_layout.addLayout(row)

        self._assistant_status = QtWidgets.QLabel("Not running. Click Talk to start.")
        self._assistant_status.setObjectName("muted")
        session_layout.addWidget(self._assistant_status)
        self._talk_btn = QtWidgets.QPushButton("Talk")
        self._talk_btn.setObjectName("talkButton")
        self._talk_btn.setEnabled(False)
        self._talk_btn.clicked.connect(self._on_talk)
        session_layout.addWidget(self._talk_btn)
        self._talk_pulse = QtWidgets.QFrame()
        self._talk_pulse.setObjectName("talkPulse")
        self._talk_pulse.setProperty("talkState", "idle")
        session_layout.addWidget(self._talk_pulse)

        self._talk_monitor = QtWidgets.QLabel("idle")
        self._talk_monitor.setObjectName("statusChip")
        self._talk_monitor.setWordWrap(True)
        session_layout.addWidget(self._talk_monitor)
        voice_setup_btn = QtWidgets.QPushButton("Microphone and voices")
        voice_setup_btn.clicked.connect(
            lambda: self._sidebar.setCurrentRow(PAGE_NAMES.index("Voice Setup"))
        )
        session_layout.addWidget(voice_setup_btn)
        layout.addWidget(session)

        conversation, conversation_layout = self._panel()
        conversation_layout.addWidget(QtWidgets.QLabel("Conversation"))
        self._conversation = QtWidgets.QTextEdit()
        self._conversation.setReadOnly(True)
        self._conversation.setPlaceholderText(
            "Click Talk and speak. Turns show up here."
        )
        conversation_layout.addWidget(self._conversation, 1)
        layout.addWidget(conversation, 1)

        shortcut = QtGui.QShortcut(QtGui.QKeySequence("Space"), page)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self._on_talk_shortcut)
        return page

    def _build_voice_page(self):
        """Build speech-to-text capture and Kokoro voice audition controls."""
        page, layout = self._page(
            "Voice Setup",
            "This is what Talk hears and speaks with. Test the microphone, pick "
            "a voice, and set noise suppression for Open WebUI recordings.",
        )
        stt, stt_layout = self._panel()
        stt_layout.addWidget(QtWidgets.QLabel("Speech to text"))
        controls = QtWidgets.QHBoxLayout()
        self._whisper_btn = QtWidgets.QPushButton("Start whisper server")
        self._whisper_btn.clicked.connect(self._on_whisper)
        self._listen_btn = QtWidgets.QPushButton("Listen (15s)")
        self._listen_btn.clicked.connect(self._on_listen)
        controls.addWidget(self._whisper_btn)
        controls.addWidget(self._listen_btn)
        controls.addStretch(1)
        self._voice_status = QtWidgets.QLabel("idle")
        self._voice_status.setObjectName("muted")
        controls.addWidget(self._voice_status)
        stt_layout.addLayout(controls)
        self._transcript = QtWidgets.QTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setPlaceholderText("Accepted transcript segments appear here.")
        stt_layout.addWidget(self._transcript)
        layout.addWidget(stt, 1)

        tts, tts_layout = self._panel()
        tts_layout.addWidget(QtWidgets.QLabel("Text to speech"))
        voice_row = QtWidgets.QHBoxLayout()
        voice_row.addWidget(QtWidgets.QLabel("Voice"))
        self._voice_combo = QtWidgets.QComboBox()
        self._populate_voice_combo(self._voice_combo)
        self._voice_combo.currentIndexChanged.connect(self._sync_voice_from_voice_page)
        self._speak_btn = QtWidgets.QPushButton("Speak test")
        self._speak_btn.clicked.connect(self._on_speak)
        self._audition_btn = QtWidgets.QPushButton("Audition all")
        self._audition_btn.clicked.connect(self._on_audition)
        voices_available = self._voice_combo.count() > 0
        self._speak_btn.setEnabled(voices_available)
        self._audition_btn.setEnabled(voices_available)
        voice_row.addWidget(self._voice_combo)
        voice_row.addWidget(self._speak_btn)
        voice_row.addWidget(self._audition_btn)
        voice_row.addStretch(1)
        self._tts_status = QtWidgets.QLabel(
            "ready" if voices_available else "no voices configured"
        )
        self._tts_status.setObjectName("muted")
        voice_row.addWidget(self._tts_status)
        tts_layout.addLayout(voice_row)
        layout.addWidget(tts)

        noise, noise_layout = self._panel()
        noise_title = QtWidgets.QLabel("Noise suppression")
        noise_title.setObjectName("sectionTitle")
        noise_layout.addWidget(noise_title)
        noise_hint = QtWidgets.QLabel(
            "Filters Open WebUI microphone uploads on this machine. Balanced "
            "is the default. Strong removes more fan/rumble. Off forwards the "
            "recording unchanged. Native Talk uses the whisper-stream gate."
        )
        noise_hint.setObjectName("muted")
        noise_hint.setWordWrap(True)
        noise_layout.addWidget(noise_hint)
        noise_row = QtWidgets.QHBoxLayout()
        noise_row.addWidget(QtWidgets.QLabel("Mode"))
        self._noise_combo = QtWidgets.QComboBox()
        for mode, label in (
            ("off", "Off"),
            ("balanced", "Balanced"),
            ("strong", "Strong"),
        ):
            self._noise_combo.addItem(label, mode)
        current_mode = "balanced"
        if hasattr(self._gc, "noise_suppression_mode"):
            current_mode = self._gc.noise_suppression_mode() or "balanced"
        index = self._noise_combo.findData(current_mode)
        if index >= 0:
            self._noise_combo.setCurrentIndex(index)
        self._noise_combo.currentIndexChanged.connect(self._on_noise_mode_changed)
        noise_row.addWidget(self._noise_combo)
        noise_row.addStretch(1)
        self._noise_status = QtWidgets.QLabel("")
        self._noise_status.setObjectName("muted")
        noise_row.addWidget(self._noise_status)
        noise_layout.addLayout(noise_row)
        layout.addWidget(noise)
        return page

    def _build_vision_page(self):
        """Build the owner-picked image path and read-only description surface."""
        page, layout = self._page(
            "Vision",
            "Pick or drop an image, ask a question, or watch the screen against a goal.",
        )
        panel, body = self._panel()
        panel.setAcceptDrops(True)
        pick_row = QtWidgets.QHBoxLayout()
        self._vision_pick_btn = QtWidgets.QPushButton("Pick image...")
        self._vision_pick_btn.clicked.connect(self._on_pick_image)
        self._vision_path = QtWidgets.QLineEdit()
        self._vision_path.setReadOnly(True)
        self._vision_path.setPlaceholderText("Pick or drop an image")
        self._vision_path.textChanged.connect(self._update_vision_preview)
        pick_row.addWidget(self._vision_pick_btn)
        pick_row.addWidget(self._vision_path, 1)
        body.addLayout(pick_row)
        self._vision_preview = QtWidgets.QLabel("No image yet")
        self._vision_preview.setObjectName("visionPreview")
        self._vision_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._vision_preview.setScaledContents(False)
        body.addWidget(self._vision_preview)
        prompt_row = QtWidgets.QHBoxLayout()
        self._vision_prompt = QtWidgets.QLineEdit()
        self._vision_prompt.setPlaceholderText("Optional question about the image")
        self._vision_btn = QtWidgets.QPushButton("Describe")
        self._vision_btn.setObjectName("primaryButton")
        self._vision_btn.setEnabled(False)
        self._vision_btn.clicked.connect(self._on_describe)
        prompt_row.addWidget(self._vision_prompt, 1)
        prompt_row.addWidget(self._vision_btn)
        body.addLayout(prompt_row)
        self._vision_result = QtWidgets.QTextEdit()
        self._vision_result.setReadOnly(True)
        self._vision_result.setPlaceholderText(
            "Pick or drop an image, then Describe."
        )
        body.addWidget(self._vision_result, 1)
        layout.addWidget(panel, 1)

        eye, eye_layout = self._panel()
        eye_title = QtWidgets.QLabel("Watch my screen")
        eye_title.setObjectName("sectionTitle")
        eye_layout.addWidget(eye_title)
        eye_hint = QtWidgets.QLabel(
            "Keeps a vision model warm and speaks when the screen drifts from "
            "the goal you type. Stops Talk first because both need the GPU."
        )
        eye_hint.setObjectName("muted")
        eye_hint.setWordWrap(True)
        eye_layout.addWidget(eye_hint)
        self._second_eye_goal = QtWidgets.QLineEdit()
        self._second_eye_goal.setPlaceholderText(
            "What you are doing, e.g. learning Python"
        )
        eye_layout.addWidget(self._second_eye_goal)
        eye_row = QtWidgets.QHBoxLayout()
        self._second_eye_start_btn = QtWidgets.QPushButton("Start watching")
        self._second_eye_start_btn.setObjectName("primaryButton")
        self._second_eye_start_btn.clicked.connect(self._on_second_eye_start)
        self._second_eye_stop_btn = QtWidgets.QPushButton("Stop")
        self._second_eye_stop_btn.setEnabled(False)
        self._second_eye_stop_btn.clicked.connect(self._on_second_eye_stop)
        eye_row.addWidget(self._second_eye_start_btn)
        eye_row.addWidget(self._second_eye_stop_btn)
        eye_row.addStretch(1)
        self._second_eye_status = QtWidgets.QLabel("not watching")
        self._second_eye_status.setObjectName("muted")
        eye_row.addWidget(self._second_eye_status)
        eye_layout.addLayout(eye_row)
        layout.addWidget(eye)
        self._install_vision_drop(panel)
        return page

    def _build_chat_page(self):
        """Build the dedicated Chat destination for the running model's web UI.

        The chooser, the fallback logic, and the browser open all live in the
        controller (request_chat); this page only offers the action and stays
        disabled while no model is running, so the button can never open a dead
        URL. It is the same intent the Models page's Open Chat button raises.
        """
        page, layout = self._page(
            "Chat",
            "Open a full chat window for the running model in your browser.",
        )
        panel, body = self._panel()
        # M15.9: this page holds one action and two lines of text, and before
        # this pass they sat in a strip at the top of a full-height empty panel
        # - the largest single void in the app. The content now lives in a
        # width-capped column, vertically centered by stretch on both sides, so
        # the page reads as a deliberate focal card rather than an unfinished
        # form.
        body.addStretch(2)
        column = QtWidgets.QVBoxLayout()
        column.setSpacing(12)
        self._chat_page_btn = QtWidgets.QPushButton("Open Chat")
        self._chat_page_btn.setObjectName("primaryButton")
        self._chat_page_btn.setMinimumHeight(44)
        self._chat_page_btn.clicked.connect(self._on_chat)
        column.addWidget(self._chat_page_btn)
        hint = QtWidgets.QLabel(
            "Start a model on Models first. Chat then opens the rich chat "
            "application when it is installed, or the built-in model UI."
        )
        self._chat_hint = hint
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        column.addWidget(hint)
        self._build_phone_strip(column)

        # M18.9 (owner report: "why don't I have my options anymore to start
        # Claude Code / Codex / OpenCode"): the coding harnesses lived ONLY
        # inside the chat-chooser dialog, so remembering a chat UI silently
        # removed them. They now have a permanent home here - always visible,
        # regardless of the remembered chat preference. A harness missing from
        # PATH shows disabled with the exact install command in its tooltip,
        # the same honesty rule as the chooser.
        code_title = QtWidgets.QLabel("Code with your local model")
        code_title.setObjectName("sectionTitle")
        code_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        column.addSpacing(16)
        column.addWidget(code_title)
        code_hint = QtWidgets.QLabel(
            "Opens a terminal running the harness against the running model, "
            "working on a project folder you pick."
        )
        code_hint.setObjectName("muted")
        code_hint.setWordWrap(True)
        code_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        column.addWidget(code_hint)
        harness_row = QtWidgets.QHBoxLayout()
        harness_row.setSpacing(8)
        harness_row.addStretch(1)
        self._chat_harness_buttons = {}
        for key, label in (
            ("claude", "Claude Code"),
            ("codex", "Codex"),
            ("opencode", "OpenCode"),
        ):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, k=key: self._launch_harness_with_folder(k, False)
            )
            harness_row.addWidget(button)
            self._chat_harness_buttons[key] = button
        harness_row.addStretch(1)
        column.addLayout(harness_row)
        self._refresh_chat_harness_buttons()
        # Chat outcomes (a fallback explanation, or an honest refusal such as "no
        # model is running") land here as well as on the Models page's status line,
        # so the owner sees them on whichever page they raised the action from.
        self._chat_status = QtWidgets.QLabel("")
        self._chat_status.setObjectName("muted")
        self._chat_status.setWordWrap(True)
        self._chat_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        column.addWidget(self._chat_status)
        centered = QtWidgets.QHBoxLayout()
        centered.addStretch(1)
        holder = QtWidgets.QWidget()
        holder.setObjectName("focusCard")
        column.setContentsMargins(28, 24, 28, 24)
        holder.setLayout(column)
        holder.setMaximumWidth(560)
        holder.setMinimumWidth(360)
        centered.addWidget(holder)
        centered.addStretch(1)
        body.addLayout(centered)
        body.addStretch(3)
        body.addStretch(1)
        layout.addWidget(panel, 1)
        return page

    def _build_phone_strip(self, column):
        """Show the Tailscale Serve URL when one already fronts Open WebUI."""
        self._phone_url_edit = None
        self._phone_copy_btn = None
        self._phone_hint = None
        reader = getattr(self._gc, "phone_access", None)
        access = reader() if callable(reader) else None
        if access is None:
            return
        url = str(getattr(access, "url", "") or "")
        if not url:
            return
        title = QtWidgets.QLabel("On your phone")
        title.setObjectName("sectionTitle")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        column.addSpacing(16)
        column.addWidget(title)
        row = QtWidgets.QHBoxLayout()
        self._phone_url_edit = QtWidgets.QLineEdit(url)
        self._phone_url_edit.setReadOnly(True)
        self._phone_url_edit.setCursorPosition(0)
        row.addWidget(self._phone_url_edit, 1)
        self._phone_copy_btn = QtWidgets.QPushButton("Copy")
        self._phone_copy_btn.clicked.connect(self._on_copy_phone_url)
        row.addWidget(self._phone_copy_btn)
        column.addLayout(row)
        self._phone_hint = QtWidgets.QLabel(
            "Same tailnet only. Open the Tailscale app on the phone, then this "
            "URL. Speak to cut in while it is talking. A headset is more reliable "
            "than the phone speaker."
        )
        self._phone_hint.setObjectName("muted")
        self._phone_hint.setWordWrap(True)
        self._phone_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        column.addWidget(self._phone_hint)

    def _on_copy_phone_url(self):
        """Copy the Tailscale phone URL to the clipboard as plain text."""
        if self._phone_url_edit is None:
            return
        QtWidgets.QApplication.clipboard().setText(self._phone_url_edit.text())
        if self._phone_hint is not None:
            self._phone_hint.setText("Copied. Open it on the phone with Tailscale on.")

    def _build_memory_page(self):
        """Build read-only conversation-memory search with an honest empty state."""
        page, layout = self._page(
            "Memory",
            "Search past local conversations. Leave the query blank to show recent conversations.",
        )
        panel, body = self._panel()
        row = QtWidgets.QHBoxLayout()
        self._memory_query = QtWidgets.QLineEdit()
        self._memory_query.setPlaceholderText("Search words, or leave blank for recent")
        self._memory_query.returnPressed.connect(self._on_memory_search)
        self._memory_btn = QtWidgets.QPushButton("Search")
        self._memory_btn.setObjectName("primaryButton")
        self._memory_btn.clicked.connect(self._on_memory_search)
        row.addWidget(self._memory_query, 1)
        row.addWidget(self._memory_btn)
        body.addLayout(row)
        self._memory_result = QtWidgets.QTextEdit()
        self._memory_result.setReadOnly(True)
        self._memory_result.setPlaceholderText(
            "Recent conversations will appear here."
        )
        body.addWidget(self._memory_result, 1)
        layout.addWidget(panel, 1)
        return page

    def _build_settings_page(self):
        """Build per-model GPU layer and context editors over controller validation."""
        page, layout = self._page(
            "Settings",
            "Launch settings apply the next time this model starts.",
        )
        panel, body = self._panel()
        # M15.9 pass 2: the three Settings panels were anonymous boxes whose
        # only identity was their first field label. Each now opens with a
        # sectionTitle, so the page reads as three named sections of one form.
        launch_title = QtWidgets.QLabel("Launch")
        launch_title.setObjectName("sectionTitle")
        body.addWidget(launch_title)
        form = QtWidgets.QGridLayout()
        form.addWidget(QtWidgets.QLabel("Model"), 0, 0)
        self._settings_model = QtWidgets.QComboBox()
        for model in self._models:
            self._settings_model.addItem(model["name"], model["id"])
        self._settings_model.currentIndexChanged.connect(self._on_settings_model_changed)
        form.addWidget(self._settings_model, 0, 1, 1, 3)
        gpu_label = QtWidgets.QLabel("GPU layers")
        gpu_label.setToolTip(
            "How many layers to keep on the GPU. 999 or -1 means fit as many "
            "as this card can hold."
        )
        form.addWidget(gpu_label, 1, 0)
        self._gpu_edit = QtWidgets.QLineEdit()
        self._gpu_edit.textChanged.connect(self._on_edit_change)
        form.addWidget(self._gpu_edit, 1, 1)
        form.addWidget(QtWidgets.QLabel("Context size"), 1, 2)
        self._ctx_edit = QtWidgets.QLineEdit()
        self._ctx_edit.textChanged.connect(self._on_edit_change)
        form.addWidget(self._ctx_edit, 1, 3)
        body.addLayout(form)
        action_row = QtWidgets.QHBoxLayout()
        self._edit_error = QtWidgets.QLabel("")
        self._edit_error.setObjectName("error")
        self._edit_error.setWordWrap(True)
        action_row.addWidget(self._edit_error, 1)
        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        action_row.addWidget(self._save_btn)
        body.addLayout(action_row)

        identity_panel, identity_body = self._panel()
        identity_title = QtWidgets.QLabel("Identity")
        identity_title.setObjectName("sectionTitle")
        identity_body.addWidget(identity_title)
        identity_form = QtWidgets.QGridLayout()
        identity_form.addWidget(QtWidgets.QLabel("id"), 0, 0)
        self._id_edit = QtWidgets.QLineEdit()
        self._id_edit.textChanged.connect(self._on_identity_change)
        identity_form.addWidget(self._id_edit, 0, 1)
        identity_form.addWidget(QtWidgets.QLabel("name"), 0, 2)
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.textChanged.connect(self._on_identity_change)
        identity_form.addWidget(self._name_edit, 0, 3)
        identity_body.addLayout(identity_form)
        identity_action_row = QtWidgets.QHBoxLayout()
        self._identity_error = QtWidgets.QLabel("")
        self._identity_error.setObjectName("error")
        self._identity_error.setWordWrap(True)
        identity_action_row.addWidget(self._identity_error, 1)
        self._identity_save_btn = QtWidgets.QPushButton("Rename")
        self._identity_save_btn.setObjectName("primaryButton")
        self._identity_save_btn.setEnabled(False)
        self._identity_save_btn.clicked.connect(self._on_save_identity)
        identity_action_row.addWidget(self._identity_save_btn)
        identity_body.addLayout(identity_action_row)

        # Owner request 2026-08-21: what a model is good for, so the Models page
        # Capabilities column has something to show beyond the default "Chat". A
        # separate panel/Save button (not folded into Rename above) because it
        # carries none of the id-collision/draft-model-reference validation a
        # rename does - any text is acceptable, including blank (plain chat).
        capabilities_panel, capabilities_body = self._panel()
        capabilities_title = QtWidgets.QLabel("Capabilities")
        capabilities_title.setObjectName("sectionTitle")
        capabilities_body.addWidget(capabilities_title)
        capabilities_form = QtWidgets.QGridLayout()
        capabilities_form.addWidget(
            QtWidgets.QLabel("What this model can do (vision, reasoning, tools)"),
            0, 0,
        )
        self._capabilities_edit = QtWidgets.QLineEdit()
        self._capabilities_edit.textChanged.connect(self._on_capabilities_change)
        capabilities_form.addWidget(self._capabilities_edit, 1, 0)
        capabilities_body.addLayout(capabilities_form)
        capabilities_action_row = QtWidgets.QHBoxLayout()
        self._capabilities_error = QtWidgets.QLabel("")
        self._capabilities_error.setObjectName("error")
        self._capabilities_error.setWordWrap(True)
        capabilities_action_row.addWidget(self._capabilities_error, 1)
        self._capabilities_save_btn = QtWidgets.QPushButton("Save capabilities")
        self._capabilities_save_btn.setObjectName("primaryButton")
        self._capabilities_save_btn.setEnabled(False)
        self._capabilities_save_btn.clicked.connect(self._on_save_capabilities)
        capabilities_action_row.addWidget(self._capabilities_save_btn)
        capabilities_body.addLayout(capabilities_action_row)

        # M18.12 (owner request): features are manageable AFTER install. This
        # panel shows what the machine has (the wizard's own detection, run on
        # the ops worker), opens the setup wizard to add anything missed, and
        # installs the coding CLIs directly - so forgetting a checkbox at
        # install time never means living without the feature.
        features_panel, features_body = self._panel()
        features_title = QtWidgets.QLabel("Features")
        features_title.setObjectName("sectionTitle")
        features_body.addWidget(features_title)
        features_hint = QtWidgets.QLabel(
            "What this machine has installed. Add anything missing with the "
            "setup wizard - already-installed pieces are never downloaded twice."
        )
        features_hint.setObjectName("muted")
        features_hint.setWordWrap(True)
        features_body.addWidget(features_hint)
        self._feature_rows_grid = QtWidgets.QGridLayout()
        self._feature_rows_grid.setColumnStretch(0, 1)
        features_body.addLayout(self._feature_rows_grid)
        self._feature_state_note = QtWidgets.QLabel("checking this machine...")
        self._feature_state_note.setObjectName("muted")
        features_body.addWidget(self._feature_state_note)
        wizard_row = QtWidgets.QHBoxLayout()
        self._open_wizard_btn = QtWidgets.QPushButton("Add features (open setup wizard)")
        self._open_wizard_btn.clicked.connect(self._on_open_setup_wizard)
        wizard_row.addWidget(self._open_wizard_btn)
        # M18.16 (owner request): uninstalling must be as easy as installing.
        # Opens the stdlib uninstaller in its own window; it offers to keep the
        # user's model files and stops LOCITIZE's own services first.
        self._uninstall_btn = QtWidgets.QPushButton("Uninstall locitize...")
        self._uninstall_btn.setObjectName("dangerButton")
        self._uninstall_btn.setToolTip(
            "Remove what setup installed (with the option to keep your model "
            "files). Opens a separate window; locitize will close."
        )
        self._uninstall_btn.clicked.connect(self._on_uninstall)
        wizard_row.addWidget(self._uninstall_btn)
        wizard_row.addStretch(1)
        features_body.addLayout(wizard_row)

        clis_title = QtWidgets.QLabel("Coding CLIs")
        clis_title.setObjectName("sectionTitle")
        features_body.addWidget(clis_title)
        clis_hint = QtWidgets.QLabel(
            "Terminal harnesses locitize can launch against your running model. "
            "Install any of them right here."
        )
        clis_hint.setObjectName("muted")
        clis_hint.setWordWrap(True)
        features_body.addWidget(clis_hint)
        clis_grid = QtWidgets.QGridLayout()
        clis_grid.setColumnStretch(1, 1)
        self._cli_status_labels = {}
        for row_index, (key, label) in enumerate(
            (("claude", "Claude Code"), ("codex", "Codex"), ("opencode", "OpenCode"))
        ):
            clis_grid.addWidget(QtWidgets.QLabel(label), row_index, 0)
            status = QtWidgets.QLabel("checking...")
            status.setObjectName("muted")
            self._cli_status_labels[key] = status
            clis_grid.addWidget(status, row_index, 1)
            install_btn = QtWidgets.QPushButton("Install")
            install_btn.clicked.connect(
                lambda _checked=False, k=key, lb=label: self._confirm_and_install_harness(
                    None, k, lb
                )
            )
            clis_grid.addWidget(install_btn, row_index, 2)
        features_body.addLayout(clis_grid)
        self._refresh_cli_status_rows()

        layout.addWidget(panel)
        layout.addWidget(identity_panel)
        layout.addWidget(capabilities_panel)
        layout.addWidget(features_panel)
        # Detection shells out; ask the ops worker once the page exists.
        if hasattr(self._gc, "request_feature_state"):
            self._gc.request_feature_state()
        layout.addStretch(1)
        # M18.12: four stacked panels exceed a normal window; without a scroll
        # area the Features/Coding CLIs panel clipped at the bottom edge
        # (caught by the real-screenshot pass).
        return self._scrollable(page)

    def _populate_voice_combo(self, combo):
        """Fill a voice picker from on-disk ids, showing human names."""
        combo.clear()
        for voice_id in self._gc.available_voices():
            combo.addItem(voice_display_name(voice_id), voice_id)
        combo.setEnabled(combo.count() > 0)

    def _set_combo_voice_id(self, combo, voice_id):
        """Select the combo row whose userData is this on-disk voice id."""
        index = combo.findData(voice_id)
        if index < 0 and voice_id:
            index = combo.findText(str(voice_id))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _sync_voice_from_talk(self, _index=None):
        """Keep the Voice page and Talk page on one chosen voice."""
        if not hasattr(self, "_voice_combo"):
            return
        voice_id = selected_voice_id(self._talk_voice)
        if selected_voice_id(self._voice_combo) != voice_id:
            self._voice_combo.blockSignals(True)
            self._set_combo_voice_id(self._voice_combo, voice_id)
            self._voice_combo.blockSignals(False)

    def _sync_voice_from_voice_page(self, _index=None):
        """Mirror a Voice-page choice into the Talk session control."""
        voice_id = selected_voice_id(self._voice_combo)
        if selected_voice_id(self._talk_voice) != voice_id:
            self._talk_voice.blockSignals(True)
            self._set_combo_voice_id(self._talk_voice, voice_id)
            self._talk_voice.blockSignals(False)

    def _select_initial_model(self):
        """Select the first real row so lifecycle and Settings controls are usable."""
        if self._model_table.rowCount() > 0:
            self._model_table.selectRow(0)
            self._on_select_model(0, 0, -1, -1)
        elif self._settings_model.count() == 0:
            self._gpu_edit.setEnabled(False)
            self._ctx_edit.setEnabled(False)
            self._save_btn.setEnabled(False)
            self._id_edit.setEnabled(False)
            self._name_edit.setEnabled(False)
            self._identity_save_btn.setEnabled(False)
            self._capabilities_edit.setEnabled(False)
            self._capabilities_save_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            self._autotune_btn.setEnabled(False)
            self._autotune_all_btn.setEnabled(False)

    def _selected_model(self):
        """Return the cached model matching the selected row, if any."""
        for model in self._models:
            if model["id"] == self._selected_id:
                return model
        return None

    def _settings_selected_model(self):
        """Return the cached model selected on the Settings page."""
        model_id = self._settings_model.currentData()
        for model in self._models:
            if model["id"] == model_id:
                return model
        return None

    def _on_refresh_models(self):
        """Re-read models.yaml on demand (owner request 2026-08-13) so a model
        registered while the desktop is open appears without a restart."""
        try:
            self._gc.refresh_models()
            # refresh_models() returns the manual rows; re-read through the merged
            # loader so discovered fine-tunes survive a Refresh click.
            self._models = self._load_model_rows()
        except Exception as exc:
            self._models_monitor.setText(f"refresh failed: {exc}")
            return
        self._refresh_model_rows()
        self._restore_model_selection()
        self._models_monitor.setText(
            f"model list refreshed ({len(self._models)} models)"
        )

    def _refresh_model_rows(self):
        """Render cached model rows in the controller helper's current sort order."""
        rows = self._models
        if self._sort_col is not None:
            rows = gui_controller.sort_model_rows(rows, self._sort_col, self._sort_desc)
        self._model_table.blockSignals(True)
        self._model_table.setRowCount(len(rows))
        for row_index, model in enumerate(rows):
            status = self._status_for(model)
            values = (
                model["name"],
                # Owner request 2026-08-21: "-" for a controller/test double that
                # predates the field, same fallback style as source_display below.
                model.get("capabilities_display", "Chat"),
                # Rows from a controller that predates M13 (or a test double) have
                # no source key; they are manual registry rows by definition.
                model.get("source_display", "Registered"),
                model["size_display"],
                model["gpu_portion_display"],
                model["cpu_portion_display"],
                model["score_display"],
                model.get("context_display", str(model.get("context_size", "-"))),
                status,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, model["id"])
                if column in (3, 4, 5, 6, 7):
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                if column == 8:
                    self._style_status_item(item, status)
                # Benchmark column: any explanatory note remains attached as a
                # tooltip so it survives a narrow/elided cell.
                if column == 6 and model.get("score_note"):
                    # Tooltips always sniff for markup (SEC-M14-1 policy).
                    item.setToolTip(plain_tooltip_text(model["score_note"]))
                if column == 7 and model.get("ctx_warning"):
                    # M17.3: flag a context past the measured cliff - amber text
                    # plus the full warning on hover, so a slow config is visible
                    # in the inventory, not only after it is running.
                    item.setForeground(QtGui.QColor("#ffd166"))
                    item.setText(str(value) + "  !")
                    item.setToolTip(plain_tooltip_text(model["ctx_warning"]))
                self._model_table.setItem(row_index, column, item)
        self._model_table.blockSignals(False)
        self._restore_model_selection()
        self._models_total.setText(gui_controller.total_size_display(self._models))

    @staticmethod
    def _style_status_item(item, status):
        """Use color as a secondary status cue while preserving explicit text."""
        if status.startswith("RUNNING"):
            item.setForeground(QtGui.QColor("#6ee7a8"))
        elif status in ("STARTING",):
            item.setForeground(QtGui.QColor("#ffd166"))
        elif status in ("future", "location not set"):
            item.setForeground(QtGui.QColor("#8b9098"))

    def _restore_model_selection(self):
        """Re-select the same model after a sort or outcome-driven refresh.

        Owner request 2026-08-21 (flashing on model select): selectRow() always
        emits currentCellChanged even when the row is already current, and this
        method is itself called FROM that signal's handler chain (_on_select_model
        -> _on_settings_model_changed -> here) - re-selecting an already-selected
        row fired a second full _on_select_model cycle, doubling every editor
        repaint on a single click. Skipping the no-op reselect breaks that loop.
        """
        if not self._selected_id:
            return
        for row in range(self._model_table.rowCount()):
            item = self._model_table.item(row, 0)
            if item and item.data(QtCore.Qt.ItemDataRole.UserRole) == self._selected_id:
                if self._model_table.currentRow() != row:
                    self._model_table.selectRow(row)
                return

    def _on_sort(self, column_index):
        """Sort headers through the controller's stable model-row helper."""
        # One key per column; None marks a column with no defined sort order.
        # Source shifted every following column right by one in M13; Capabilities
        # (owner request 2026-08-21) shifted every following column right again.
        keys = (
            "name",
            "capabilities_display",
            None,
            "size",
            "vram",
            None,
            "benchmark_tok_s",
            "context",
            "status",
        )
        column = keys[column_index]
        if column is None:
            return
        if self._sort_col == column:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = column
            self._sort_desc = False
        self._refresh_model_rows()

    def _status_for(self, model):
        """Render a textual model status from the pump-owned UI snapshot."""
        if not model["launchable"]:
            return "location not set" if not model["location"] else "future"
        if self._ui.running_model_id == model["id"]:
            return (
                f"RUNNING (port {self._ui.running_port})"
                if self._ui.running_port
                else "RUNNING"
            )
        if self._ui.in_flight and self._selected_id == model["id"]:
            return "STARTING"
        return "STOPPED"

    def _on_select_model(self, row, _column, _previous_row, _previous_column):
        """Use the selected row for lifecycle controls and sync Settings.

        Owner request 2026-08-21 (flashing on model select): setCurrentIndex
        below used to fire Qt's currentIndexChanged synchronously into
        _on_settings_model_changed, which re-ran _load_settings_model and
        _refresh_buttons a SECOND time before this method's own calls to them
        - every editor field and the button row visibly repainted twice per
        click. Signals are blocked around the combo update (this method already
        does the identical work itself right after), so the pair now runs once.
        """
        if row < 0:
            return
        item = self._model_table.item(row, 0)
        if item is None:
            return
        self._selected_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        settings_index = self._settings_model.findData(self._selected_id)
        if settings_index >= 0 and self._settings_model.currentIndex() != settings_index:
            self._settings_model.blockSignals(True)
            self._settings_model.setCurrentIndex(settings_index)
            self._settings_model.blockSignals(False)
        self._load_settings_model()
        self._refresh_buttons()

    def _on_settings_model_changed(self, _index):
        """Make Settings selection authoritative and mirror it to the model table."""
        model = self._settings_selected_model()
        if model is None:
            return
        self._selected_id = model["id"]
        self._restore_model_selection()
        self._load_settings_model()
        self._refresh_buttons()

    def _load_settings_model(self):
        """Prefill editors from persisted values without inventing dirty state."""
        model = self._settings_selected_model()
        if model is None:
            return
        self._gpu_edit.blockSignals(True)
        self._ctx_edit.blockSignals(True)
        self._gpu_edit.setText(str(model["gpu_layers"]))
        self._ctx_edit.setText(str(model["context_size"]))
        self._gpu_edit.blockSignals(False)
        self._ctx_edit.blockSignals(False)
        self._on_edit_change()
        self._id_edit.blockSignals(True)
        self._name_edit.blockSignals(True)
        self._id_edit.setText(str(model["id"]))
        self._name_edit.setText(str(model["name"]))
        self._id_edit.blockSignals(False)
        self._name_edit.blockSignals(False)
        self._on_identity_change()
        self._capabilities_edit.blockSignals(True)
        self._capabilities_edit.setText(", ".join(model.get("capabilities", [])))
        self._capabilities_edit.blockSignals(False)
        self._on_capabilities_change()

    def _autotune_active(self):
        """True from the Auto-tune click until its result lands.

        Two sources because they cover different halves of the run: the local
        flag covers the queued-but-not-yet-started window, the controller's own
        flag covers the run itself (and survives a page refresh, which the local
        flag would too, but the controller is the authority once it has started).
        """
        return bool(self._autotune_pending or self._gc.autotune_in_progress())

    def _refresh_buttons(self):
        """Apply controller-defined lifecycle states across both relevant pages."""
        states = gui_controller.compute_button_states(self._ui)
        model = self._selected_model()
        start_enabled = bool(states["start_enabled"] and model and model["launchable"])
        start_label = "Start"
        if model is not None:
            start_label = gui_controller.start_label_for(model["id"], self._ui)
            if start_label == "Running":
                start_enabled = False
        self._start_btn.setText(start_label)
        self._start_btn.setEnabled(start_enabled)
        self._stop_btn.setEnabled(bool(states["stop_enabled"]))
        # Offload GPU stays available even with no supervised model running (M17.8):
        # it also clears LOCITIZE's OWN orphaned servers - a crashed session or a
        # measurement probe - which the plain model-stop path cannot reach. The
        # autotune/assistant-live rules below still switch it off where offloading
        # would fight a run LOCITIZE is deliberately managing.
        self._header_offload_btn.setEnabled(True)
        chat_enabled = bool(states["chat_enabled"] or self._assistant_port)
        self._chat_btn.setEnabled(chat_enabled)
        # The Chat page's own button raises the identical intent, so it shares
        # exactly one enablement rule with the Models page control.
        self._chat_page_btn.setEnabled(chat_enabled)
        self._update_chat_hint()
        if not self._assistant_live:
            can_start_talk = bool(model and model.get("launchable"))
            self._talk_btn.setEnabled(can_start_talk)
        if self._assistant_live:
            # A live session owns its chat model; the owner ends the session before
            # changing model lifecycle, while Chat remains safe and reachable.
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            self._header_offload_btn.setEnabled(False)
        if self._autotune_active():
            # Owner request 2026-08-22: while an auto-tune holds the ops worker,
            # Stop is the owner's only way out of a multi-minute run, so it is
            # enabled here LAST - after every rule that would otherwise have
            # switched it off. Start stays disabled: the tuner is already using
            # the GPU. Once cancellation has been asked for, Stop goes flat so a
            # second click cannot read as "it ignored me".
            self._stop_btn.setEnabled(not self._autotune_canceling)
            self._start_btn.setEnabled(False)
            self._header_offload_btn.setEnabled(False)
        self._refresh_model_exclusive_buttons()
        self._header_status.setText(self._health_line())

    def _refresh_model_exclusive_buttons(self):
        """Protect assistant-held models from Vision and benchmark switching."""
        model = self._selected_model()
        # Register only applies to a discovered row; a registered row has nothing
        # to promote, so the control stays visibly disabled rather than dead.
        self._register_btn.setEnabled(
            model is not None and model.get("source") == "discovered"
        )
        # Delete (owner request 2026-08-21): only a registered row has a
        # models.yaml block to remove (mirrors the identity/capabilities
        # editors' discovered-row refusal), and never the running model - its
        # .gguf is an open file handle, so disabling here gives the honest
        # reason up front instead of a failed unlink after a confirm click.
        self._delete_btn.setEnabled(
            model is not None
            and model.get("source") != "discovered"
            and self._ui.running_model_id != model.get("id")
        )
        # Auto-tune context (owner request 2026-08-22): same two structural
        # refusals as Delete - a discovered row has no models.yaml block to write
        # back into - plus one of its own: nothing may be running, because the
        # tuner's own trial starts would collide with it on the reserved port.
        self._autotune_btn.setEnabled(
            not self._assistant_live
            and model is not None
            and model.get("source") != "discovered"
            and self._ui.running_model_id is None
            # ...and never while one is already running. Without this clause any
            # button refresh during a tune (a monitor tick, a selection change)
            # would hand the owner a second Auto-tune click that would queue a
            # run behind the live one.
            and not self._autotune_active()
        )
        self._autotune_all_btn.setEnabled(
            not self._assistant_live
            and self._ui.running_model_id is None
            and not self._autotune_active()
            and any(m.get("source") != "discovered" for m in self._models)
        )
        self._vision_btn.setEnabled(
            not self._assistant_live and bool(self._vision_path.text())
        )

    def _selected_context_size(self):
        """Return context size for the model represented by the live metrics."""
        target = self._ui.running_model_id or self._selected_id
        for model in self._models:
            if model["id"] == target:
                return model.get("context_size")
        return None

    def _health_line(self):
        """Build the compact header from the same honest state as the monitor."""
        running = self._ui.running_model_id or (
            "assistant model" if self._assistant_port else "(none)"
        )
        visible_port = self._ui.running_port or self._assistant_port
        port_text = f" port {visible_port}" if visible_port else ""
        rate = "-"
        context = "-"
        sample = self._ui.latest_metrics
        if sample is not None and sample.metrics_available:
            rate = "idle" if not sample.gen_tokens_s else f"{sample.gen_tokens_s:.1f} tok/s"
            context = gui_controller._format_ctx(sample, self._selected_context_size())
        return f"Health: {self._health} | model: {running}{port_text} | {rate} | ctx {context}"

    def _on_start(self):
        """Queue a start or switch and immediately render an honest working state."""
        model = self._selected_model()
        if model is None:
            self._model_status.setText("select a model first")
            return
        if not model["launchable"]:
            self._model_status.setText(
                "that model has no file to launch - use Register to point "
                "at its .gguf"
            )
            return
        picked, problems = self._selected_thinking(model["id"])
        if problems:
            # Refuse rather than start at a level the owner did not choose - the
            # same rule the terminal menu applies to "3 low/abc".
            self._model_status.setText(problems[0])
            return
        self._ui.in_flight = True
        detail = f" ({picked})" if picked else ""
        self._model_status.setText(
            f"working... starting {model['name']}{detail}"
        )
        self._refresh_buttons()
        self._gc.request_start(model["id"], reasoning=picked)

    def _selected_thinking(self, model_id: str):
        """The Thinking box as a validated reasoning mapping, or (None, []).

        Returns (None, []) for the do-nothing default so Start forwards no
        override at all, and (None, problems) for an unreadable entry so the
        caller can refuse instead of guessing.
        """
        from config import parse_reasoning_choice

        text = self._thinking_combo.currentText().strip()
        if not text or text == THINKING_AS_REGISTERED:
            return None, []
        return parse_reasoning_choice(text, model_id)

    def _on_offload_gpu(self):
        """Offload GPU (M17.8): free the VRAM LOCITIZE is holding.

        While an auto-tune owns the GPU, this defers to the same cancel path Stop
        uses - killing the tune's server out from under it would leave a
        half-probed config, so the tune is asked to stop cleanly instead.
        Otherwise it asks the controller to stop the running model AND sweep any
        of LOCITIZE's own orphaned servers, then reports what it freed and what
        (foreign) processes remain - the visibility the plain stop never gave.
        """
        if self._gc.autotune_in_progress():
            self._on_stop()
            return
        self._model_status.setText(
            "offloading GPU: stopping locitize's own servers ..."
        )
        self._gc.request_free_gpu()

    def _apply_gpu_free(self, result):
        """Render the outcome of an Offload GPU sweep."""
        self._ui.in_flight = False
        payload = result.payload or {}
        self._apply_running_snapshot(payload)
        self._assistant_port = None
        self._ui.latest_metrics = None
        self._render_monitor_text("idle")
        freed = int(payload.get("freed", 0))
        others = payload.get("others") or []
        vram = payload.get("vram_free_mb")
        parts = [
            f"freed {freed} locitize server(s)"
            if freed
            else "no locitize servers were holding the GPU"
        ]
        if vram is not None:
            parts.append(f"{vram:.0f} MB VRAM free")
        if others:
            shown = ", ".join(others[:4]) + ("..." if len(others) > 4 else "")
            parts.append(f"still held by other apps: {shown}")
        message = "; ".join(parts) + "."
        self._model_status.setText(message)
        # The chip clips rather than resizing the window (M18.8); hover shows all.
        self._model_status.setToolTip(message)
        self._refresh_model_rows()
        self._refresh_buttons()

    def _on_stop(self):
        """Stop: cancel a running auto-tune if there is one, else stop the model.

        The auto-tune branch comes first and does NOT go through the command
        queue (owner request 2026-08-22). An auto-tune holds the single ops
        worker for several minutes, so a queued stop would sit unread until the
        tune it was meant to interrupt had already finished - the owner would
        press Stop and watch nothing happen. cancel_autotune() sets an event the
        tune's trial loop polls directly, which is why this one control reaches
        past the queue that everything else uses.

        There is no ambiguity between the two meanings: an auto-tune only runs
        when no model is running (request_autotune_context refuses otherwise),
        so at most one of these branches is ever applicable.
        """
        if self._gc.autotune_in_progress():
            model_id = self._gc.autotune_model_id() or ""
            self._gc.cancel_autotune()
            self._autotune_canceling = True
            self._autotune_status.setVisible(True)
            self._autotune_status.setText(
                f"canceling the auto-tune of {model_id}: stopping the trial that "
                f"is running now ..."
            )
            self._model_status.setText("canceling auto-tune...")
            self._refresh_buttons()
            return
        self._ui.in_flight = True
        self._model_status.setText("working... stopping")
        self._refresh_buttons()
        self._gc.request_stop()

    def _on_chat(self):
        """Ask the controller to resolve and open the configured chat surface."""
        self._model_status.setText("choosing an available chat interface...")
        self._gc.request_chat()

    def _on_autotune_context(self):
        """Ask for confirmation, then queue the selected model's context auto-tune.

        A confirmation dialog because this is not a click with an instant result:
        it loads the model for real several times over several minutes and
        rewrites its models.yaml row. The dialog names the model and says what
        will happen, so nobody starts it by accident while the GPU is wanted for
        something else.
        """
        model = self._selected_model()
        if model is None:
            self._model_status.setText("select a model to auto-tune")
            return
        if not self._autotune_btn.isEnabled():
            # Same belt-and-braces guard Delete uses: a keyboard activation can
            # reach a control the enablement rule has just turned off.
            return
        confirmed = QtWidgets.QMessageBox.question(
            self,
            "Auto-tune context",
            f"Auto-tune the context window for '{model['id']}'?\n\n"
            f"locitize will read the model's native context from its file, then "
            f"start it several times at different context sizes to find the "
            f"largest one this machine can actually load, and finally benchmark "
            f"it at that size to measure its real speed (tokens per second). "
            f"This takes a few minutes and uses the GPU throughout. The context "
            f"is written back to models.yaml for this model only.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirmed != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._autotune_btn.setEnabled(False)
        self._autotune_pending = True
        self._autotune_canceling = False
        self._autotune_status.setVisible(True)
        self._autotune_status.setText(
            f"auto-tuning {model['id']}: starting ... (press Stop to cancel)"
        )
        self._model_status.setText(f"auto-tuning {model['id']}...")
        self._gc.request_autotune_context(model["id"])
        # Immediately, so Stop becomes live in the same event as the click rather
        # than at whenever the next refresh happens to be.
        self._refresh_buttons()

    def _on_autotune_all(self):
        """Confirm, then queue an auto-tune of every registered model."""
        if not self._autotune_all_btn.isEnabled():
            return
        count = sum(1 for m in self._models if m.get("source") != "discovered")
        confirmed = QtWidgets.QMessageBox.question(
            self,
            "Auto-tune all models",
            f"Auto-tune all {count} models?\n\n"
            f"Each model is started several times to find the largest context "
            f"this machine can load, then benchmarked at that size for its real "
            f"speed. That takes a few minutes per model - roughly "
            f"{max(1, round(count * 4 / 60))} hour(s) for {count} - and uses the "
            f"GPU throughout. Press Stop at any time: the model in progress is "
            f"left unchanged and the rest are skipped.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirmed != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._autotune_batch = True
        self._autotune_pending = True
        self._autotune_canceling = False
        self._autotune_status.setVisible(True)
        self._autotune_status.setText(
            f"auto-tuning all {count} models: starting ... (press Stop to cancel)"
        )
        self._model_status.setText(f"auto-tuning all {count} models...")
        self._gc.request_autotune_all()
        self._refresh_buttons()

    def _apply_autotune_all_result(self, result):
        """Close an "Auto-tune all" run with its tally."""
        self._autotune_batch = False
        self._autotune_pending = False
        self._autotune_canceling = False
        self._refresh_buttons()
        self._autotune_status.setVisible(True)
        if not result.ok:
            message = result.error or "auto-tune all failed"
            self._autotune_status.setText(f"auto-tune all: {message}")
            self._model_status.setText(f"auto-tune all: {message}")
            return
        p = result.payload
        text = (
            f"auto-tune all finished: {p.get('tuned', 0)} of {p.get('total', 0)} "
            f"models tuned"
        )
        if p.get("failed"):
            text += f", {p['failed']} failed"
        if p.get("canceled"):
            text += f", canceled ({p.get('skipped', 0)} not tuned)"
        self._autotune_status.setText(text + ".")
        self._model_status.setText(text)
        self._on_refresh_models()

    def _apply_autotune_progress(self, result):
        """Render one live progress line from the running auto-tune.

        The whole point of this handler: these runs take minutes, so the owner
        must be able to see WHICH context size is being tried right now rather
        than a spinner that is indistinguishable from a hung window.
        """
        line = result.payload.get("line", "")
        if not line:
            return
        self._autotune_status.setVisible(True)
        if self._autotune_canceling:
            # A cancel has been asked for; do not overwrite the "canceling" line
            # with trial chatter that would read as though Stop was ignored.
            return
        self._autotune_status.setText(
            f"auto-tuning {result.payload.get('model_id', '')}: {line} "
            f"(press Stop to cancel)"
        )

    def _apply_autotune_result(self, result):
        """Render the finished auto-tune: the real numbers, or the honest reason."""
        # The run is over however it ended, so Stop stops meaning "cancel" again
        # before any of the enablement rules below are re-derived - unless this
        # is one model of an "Auto-tune all" run, which ends with its own result.
        if not self._autotune_batch:
            self._autotune_pending = False
            self._autotune_canceling = False
        # Re-derive rather than force-enable: the selection or running state may
        # have changed while the tune ran, and this one rule owns the answer.
        # _refresh_buttons covers the model-exclusive controls too, so it also
        # restores Stop to its ordinary "stop the running model" enablement.
        self._refresh_buttons()
        model_id = result.payload.get("model_id", "")
        self._autotune_status.setVisible(True)
        if result.payload.get("canceled"):
            # Deliberately NOT phrased as a failure. Nothing went wrong: the
            # owner asked it to stop and their model was left exactly as it was.
            self._autotune_status.setText(
                f"auto-tune of {model_id} canceled. Its context size and "
                f"launch settings were left unchanged."
            )
            self._model_status.setText("auto-tune canceled")
            return
        if not result.ok:
            message = result.error or "auto-tune failed"
            self._autotune_status.setText(f"auto-tune of {model_id} failed: {message}")
            self._model_status.setText(f"auto-tune failed: {message}")
            return
        native = result.payload.get("native_context")
        chosen = result.payload.get("chosen_context")
        previous = result.payload.get("previous_context")
        failure = result.payload.get("first_failure")
        trials = result.payload.get("trials") or []
        yarn = (
            f"context extended {result.payload.get('rope_scale')}x beyond "
            "the trained window"
            if result.payload.get("yarn_applied")
            else "no context extension needed (within the trained window)"
        )
        ceiling = (
            f"largest that loaded: {chosen}"
            + (f", smallest that failed: {failure}" if failure else "")
        )
        speed = result.payload.get("tokens_per_second")
        if speed is not None:
            measured = f" Measured speed at that context: {float(speed):.1f} tokens/s."
            headline = f"context {chosen}, {float(speed):.1f} tokens/s"
        else:
            reason = result.payload.get("benchmark_error") or "not measured"
            measured = f" Speed not measured ({reason})."
            headline = f"context size now {chosen}"
        self._autotune_status.setText(
            f"auto-tune of {model_id} finished. Native context: {native}. "
            f"{ceiling}. {yarn}. context_size {previous} -> {chosen}, written to "
            f"models.yaml after {len(trials)} real starts.{measured} Applies the "
            f"next time this model starts."
        )
        self._model_status.setText(f"auto-tune finished: {headline}")
        # The row's cached context_size is now stale; re-read the registry so the
        # page shows what is really on disk rather than what it showed before.
        self._on_refresh_models()

    def _on_start_assistant(self):
        """Queue assistant startup with the chosen voice and output preference."""
        self._assistant_status.setText("Starting...")
        self._assistant_start_btn.setEnabled(False)
        self._talk_btn.setEnabled(False)
        self._gc.request_start_assistant(
            selected_voice_id(self._talk_voice), self._assistant_speak.isChecked()
        )

    def _on_end_assistant(self):
        """Queue session teardown and lock controls until the outcome arrives."""
        self._assistant_status.setText("ending session...")
        self._assistant_end_btn.setEnabled(False)
        self._talk_btn.setEnabled(False)
        self._interrupt_btn.setEnabled(False)
        self._gc.request_end_assistant()

    def _on_talk(self):
        """Start the session if needed, barge in if speaking, else open capture."""
        if not self._assistant_live:
            self._listen_when_ready = True
            self._on_start_assistant()
            return
        self._gc.request_talk()

    def _on_talk_shortcut(self):
        """Space on the Talk page is the same as the Talk button, when it is live."""
        if self._talk_btn.isEnabled() or (
            self._assistant_live and self._talk_state in ("thinking", "speaking")
        ):
            self._on_talk()

    def _on_interrupt(self):
        """Request a non-blocking generation and audio interruption."""
        self._gc.request_interrupt()

    def _on_whisper(self):
        """Queue the existing whisper service toggle."""
        self._model_status.setText("working... toggling whisper server")
        self._gc.request_whisper_toggle()

    def _on_listen(self):
        """Queue the existing fifteen-second microphone capture path."""
        self._model_status.setText("listening for 15s - speak now")
        self._voice_status.setText("listening for 15s - speak now")
        self._transcript.clear()
        self._gc.request_listen(15.0)

    def _on_speak(self):
        """Queue the fixed voice sample in the selected on-disk voice."""
        voice = selected_voice_id(self._voice_combo)
        self._tts_status.setText(f"speaking test in {voice_display_name(voice)} ...")
        self._gc.request_speak("locitize voice test. This is the selected voice.", voice)

    def _on_audition(self):
        """Queue an audition of every voice known to the controller."""
        self._tts_status.setText("auditioning all voices...")
        self._gc.request_audition()

    def _on_pick_image(self):
        """Ask the owner for one local image path and retain only the selection."""
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Pick an image to describe",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;All files (*.*)",
        )
        if path:
            self._vision_path.setText(path)
            self._refresh_model_exclusive_buttons()

    def _update_vision_preview(self, path=""):
        """Show a scaled pixmap for the selected image, or an honest empty label."""
        chosen = str(path or self._vision_path.text() or "")
        if not chosen:
            self._vision_preview.setPixmap(QtGui.QPixmap())
            self._vision_preview.setText("No image yet")
            return
        pixmap = QtGui.QPixmap(chosen)
        if pixmap.isNull():
            self._vision_preview.setPixmap(QtGui.QPixmap())
            self._vision_preview.setText("Could not read that image")
            return
        fitted = pixmap.scaled(
            self._vision_preview.size()
            if self._vision_preview.width() > 40
            else QtCore.QSize(640, 180),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._vision_preview.setText("")
        self._vision_preview.setPixmap(fitted)

    def _install_vision_drop(self, panel):
        """Accept one local image dropped onto the Vision panel."""
        window = self
        original_drag = panel.dragEnterEvent
        original_drop = panel.dropEvent

        def drag_enter(event):
            mime = event.mimeData()
            if mime.hasUrls() or mime.hasImage():
                event.acceptProposedAction()
            elif original_drag is not None:
                original_drag(event)

        def drop(event):
            mime = event.mimeData()
            path = ""
            if mime.hasUrls():
                for url in mime.urls():
                    local = url.toLocalFile()
                    if local:
                        path = local
                        break
            if path:
                window._vision_path.setText(path)
                window._refresh_model_exclusive_buttons()
                event.acceptProposedAction()
            elif original_drop is not None:
                original_drop(event)

        panel.dragEnterEvent = drag_enter
        panel.dropEvent = drop

    def _on_describe(self):
        """Queue vision description for the chosen path and optional prompt."""
        path = self._vision_path.text()
        if not path:
            self._vision_result.setPlainText("pick an image first")
            return
        self._vision_result.setPlainText("describing (switching to the vision model)...")
        self._vision_btn.setEnabled(False)
        self._gc.request_describe(path, self._vision_prompt.text())

    def _on_memory_search(self):
        """Queue a read-only memory query; blank intentionally means recent."""
        self._memory_result.setPlainText("searching...")
        self._gc.request_memory_search(self._memory_query.text())

    def _on_edit_change(self):
        """Drive Save enablement and validation from the controller's pure helper."""
        model = self._settings_selected_model()
        if model is None:
            self._save_btn.setEnabled(False)
            self._edit_error.clear()
            return
        state = gui_controller.compute_save_state(
            self._gpu_edit.text(),
            self._ctx_edit.text(),
            model["gpu_layers"],
            model["context_size"],
        )
        self._save_btn.setEnabled(bool(state["enabled"]))
        self._edit_error.setText(state["error"] or "")

    def _on_save(self):
        """Validate and queue the selected model's settings through the controller."""
        model = self._settings_selected_model()
        if model is not None:
            self._gc.save_model_edits(
                model["id"], self._gpu_edit.text(), self._ctx_edit.text()
            )

    def _on_identity_change(self):
        """Drive Rename enablement and validation from the controller's pure helper."""
        model = self._settings_selected_model()
        if model is None:
            self._identity_save_btn.setEnabled(False)
            self._identity_error.clear()
            return
        existing_ids = [m["id"] for m in self._models]
        state = gui_controller.compute_identity_save_state(
            self._id_edit.text(),
            self._name_edit.text(),
            model["id"],
            model["name"],
            existing_ids,
        )
        self._identity_save_btn.setEnabled(bool(state["enabled"]))
        self._identity_error.setText(state["error"] or "")

    def _on_save_identity(self):
        """Validate and queue the selected model's id/name rename through the controller."""
        model = self._settings_selected_model()
        if model is not None:
            self._gc.save_model_identity(
                model["id"], self._id_edit.text(), self._name_edit.text()
            )

    def _on_capabilities_change(self):
        """Enable Save capabilities only when the parsed tag list actually differs."""
        model = self._settings_selected_model()
        if model is None:
            self._capabilities_save_btn.setEnabled(False)
            self._capabilities_error.clear()
            return
        typed = gui_controller.parse_capabilities_text(self._capabilities_edit.text())
        self._capabilities_save_btn.setEnabled(typed != list(model.get("capabilities", [])))
        self._capabilities_error.clear()

    def _on_save_capabilities(self):
        """Queue the selected model's capabilities write through the controller."""
        model = self._settings_selected_model()
        if model is not None:
            self._gc.save_model_capabilities(model["id"], self._capabilities_edit.text())

    # ---- M13 Fine-tune page ---------------------------------------------- #

    def _on_finetune_start(self):
        """Ask the controller to start the studio; buttons lock until it answers."""
        self._ft_warning.setVisible(False)
        self._apply_finetune_state({"status": "starting", "reason": "", "url": None})
        self._gc.request_finetune_start()

    def _on_finetune_stop(self):
        """Ask the controller to stop the studio."""
        self._set_finetune_buttons(False, False, False)
        self._gc.request_finetune_stop()

    def _on_finetune_open(self):
        """Open the running studio in the owner's default browser."""
        self._gc.request_finetune_open()

    def _on_finetune_rescan(self):
        """Re-run the discovery scan; the table keeps its rows until new ones land."""
        self._ft_rescan_btn.setEnabled(False)
        self._ft_status.setText("Scanning...")
        self._gc.request_scan_finetunes()

    def _on_select_finetune(self, row, _column, _previous_row, _previous_column):
        """Track the selected discovered row so the row actions know their target."""
        if row < 0:
            return
        item = self._ft_table.item(row, 0)
        if item is None:
            return
        self._selected_finetune_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._refresh_finetune_buttons()

    def _selected_finetune(self):
        """Return the cached discovered row matching the current selection."""
        for row in self._finetunes:
            if row["id"] == self._selected_finetune_id:
                return row
        return None

    def _on_finetune_serve(self):
        """Serve the selected discovered model through the normal start path."""
        row = self._selected_finetune()
        if row is None:
            return
        self._ft_status.setText(f"starting {row['name']}...")
        self._gc.request_start(row["id"])

    def _on_finetune_delete(self):
        """Confirm, then ask the worker to delete the selected run (M18.7)."""
        row = self._selected_finetune()
        if row is None:
            return
        answer = QtWidgets.QMessageBox.warning(
            self,
            "Delete fine-tune run",
            f"Permanently delete the run '{row.get('run', row.get('name', '?'))}' "
            "and every file in its folder (adapter, merged model, GGUF)?\n\n"
            "A training run cannot be recovered without re-training.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._ft_status.setText(f"deleting {row.get('run', '')} ...")
        self._gc.request_finetune_delete(row["id"])

    def _on_finetune_register(self):
        """Promote the selected discovered model into models.yaml (owner action)."""
        row = self._selected_finetune()
        if row is None:
            return
        self._ft_register_btn.setEnabled(False)
        self._ft_status.setText(f"Registering {row['name']}...")
        self._gc.request_register_discovered(row["id"])

    def _on_register_selected(self):
        """Models-page Register: same intent, targeting the selected model row."""
        model = self._selected_model()
        if model is None or model.get("source") != "discovered":
            return
        self._register_btn.setEnabled(False)
        self._model_status.setText(f"Registering {model['name']}...")
        self._gc.request_register_discovered(model["id"])

    def _on_delete_model(self):
        """Confirm, then queue a permanent Delete (owner request 2026-08-21).

        The confirmation dialog is built here, in the UI layer, and is the
        ONLY thing standing between a click and an irreversible unlink() -
        gui_controller.delete_model() only validates state (discovered row,
        running model), it never asks the owner anything. Names every file
        that will actually be removed rather than a generic "are you sure",
        so the owner is confirming a specific, visible list of paths.
        """
        model = self._selected_model()
        if model is None or not self._delete_btn.isEnabled():
            return
        paths = [p for p in (model.get("location"), model.get("mmproj")) if p]
        file_list = "\n".join(f"  - {p}" for p in paths) or "  (no file on disk recorded)"
        confirmed = QtWidgets.QMessageBox.warning(
            self,
            "Delete model",
            f'Permanently delete "{model["name"]}"?\n\n'
            f"This removes it from the model list AND deletes this file from "
            f"disk:\n{file_list}\n\n"
            f"This cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if confirmed != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._delete_btn.setEnabled(False)
        self._model_status.setText(f"Deleting {model['name']}...")
        self._gc.delete_model(model["id"])

    # ---- M14.14 "Get models" handlers ----------------------------------- #

    def _on_hub_search(self):
        """Search press: the FIRST of the three controls allowed to cause egress."""
        if not self._hub_enabled:
            return
        query = self._hub_query.text().strip()
        if not query:
            self._hub_status.setText("Enter a search term first.")
            return
        # Disabled until the answer comes back, so a repeated press cannot queue
        # a second call to the same rate-limited endpoint.
        self._hub_search_btn.setEnabled(False)
        # A fresh search starts a new result set - the prior query's Load more
        # cursor no longer applies, so hide it until this search's own answer
        # says whether a further page exists.
        self._hub_has_more = False
        self._hub_load_more_btn.setVisible(False)
        self._hub_status.setText(f"Searching huggingface.co for '{query}'...")
        self._gc.request_hub_search(query)

    def _on_hub_search_more(self):
        """Load more press: the same egress-on-explicit-action rule as Search."""
        if not self._hub_enabled or not self._hub_has_more:
            return
        self._hub_load_more_btn.setEnabled(False)
        self._hub_load_more_btn.setText("Loading more...")
        self._gc.request_hub_search_more()

    def _on_hub_repo_selected(self, row=-1, *_args):
        """Repo selection: the SECOND egress control - it lists that repo's files."""
        entry = self._selected_hub_repo()
        if entry is None or not self._hub_enabled:
            return
        self._hub_file_table.setRowCount(0)
        self._hub_download_btn.setEnabled(False)
        self._hub_status.setText(f"Listing files in {entry['repo_id']}...")
        self._gc.request_hub_files(entry["repo_id"])

    def _on_hub_file_selected(self, row=-1, *_args):
        """Show the fit arithmetic for the selected file. Pure local computation."""
        entry = self._selected_hub_file()
        if entry is None:
            self._hub_download_btn.setEnabled(False)
            self._hub_fit.setText("")
            return
        # DEFECT-QA-M14-2: browsing the file list during a download must not
        # re-arm Download. One transfer at a time is the controller's rule, and
        # a button that looks pressable but answers "A download is already
        # running." is a dead control by any other name. A request that is sent
        # but not yet confirmed counts as busy too (NEW-QA-M14-7), so the gap
        # between pressing Download and the first progress line is not a window
        # in which the button comes back to life.
        self._hub_download_btn.setEnabled(self._hub_busy_job() is None)
        fit = entry.get("fit") or {}
        lines = [fit.get("wording", "")]
        if fit.get("explanation") and fit.get("band") != "unknown":
            lines.append(fit["explanation"])
        # The disclaimer is not optional decoration: an estimate presented as a
        # verdict is the thing that would make this feature dishonest.
        if fit.get("band") != "unknown":
            lines.append(fit.get("disclaimer", ""))
        if fit.get("multi_gpu_caveat"):
            lines.append(fit["multi_gpu_caveat"])
        lines.append(entry.get("verification_label", ""))
        self._hub_fit.setText("  ".join(x for x in lines if x))

    def _selected_hub_repo(self):
        """The currently selected discovery row, or None."""
        index = self._hub_repo_table.currentRow()
        rows = getattr(self, "_hub_repo_rows", [])
        return rows[index] if 0 <= index < len(rows) else None

    def _selected_hub_file(self):
        """The currently selected file row, or None."""
        index = self._hub_file_table.currentRow()
        rows = getattr(self, "_hub_file_rows", [])
        return rows[index] if 0 <= index < len(rows) else None

    def _on_hub_download(self):
        """Download press: the THIRD and last egress control, and the only write.

        Everything the user is agreeing to is put in front of them BEFORE the
        first byte moves: the size, the destination, the licence position, and -
        when there is no publisher checksum - the fact that LOCITIZE cannot verify
        what it is about to save.
        """
        entry = self._selected_hub_file()
        repo = self._selected_hub_repo()
        if entry is None or repo is None or not self._hub_enabled:
            return
        size_gb = (entry.get("size_bytes") or 0) / 1_000_000_000
        fit = entry.get("fit") or {}
        lines = [
            f"Download {entry['filename']}",
            f"from {repo['repo_id']}",
            f"Size: {size_gb:.1f} GB",
            entry.get("verification_label", ""),
            "",
            "This model's license is an agreement between you and its "
            "publisher. locitize does not redistribute it."
            + (f" Publisher's license: {repo.get('license_tag')}."
               if repo.get("license_tag") else ""),
        ]
        if not self._confirm("Download this model?", "\n".join(lines)):
            return
        # V-NONE: a second, separate consent whose text says plainly that LOCITIZE
        # cannot verify the file. Never folded into the first dialog - it is a
        # different question from "do you want this model".
        if entry.get("verification") == "none":
            if not self._confirm("No publisher checksum", UNVERIFIED_CONSENT_TEXT):
                return
        # `exceeds` advises, it never vetoes: users with plenty of system RAM do
        # this on purpose. The second button names the consequence.
        if fit.get("band") == "exceeds":
            if not self._confirm(
                "Larger than your VRAM",
                f"{fit.get('explanation', '')}\n\n{fit.get('disclaimer', '')}\n\n"
                "Download anyway (will be slow)?",
            ):
                return
        # The id of the job now being REQUESTED, derived through modelhub's
        # shared helper so it is byte-identical to the one the controller stamps
        # on every progress and terminal Result (DEFECT-QA-M14-2).
        #
        # It goes into _hub_pending_job, never straight into _hub_active_job
        # (NEW-QA-M14-7): the controller may refuse this request, and a refusal
        # carries this very id. If it had already displaced a live job's id, the
        # guard in _apply_hub_done would compare the refusal against itself,
        # find them equal, and apply terminal UI to the transfer still running.
        # The id is adopted as "active" in _apply_hub_progress, i.e. only once
        # the controller has proved the transfer started.
        self._hub_pending_job = hub_job_id(repo["repo_id"], entry["filename"])
        self._hub_download_btn.setEnabled(False)
        self._hub_cancel_btn.setEnabled(True)
        self._hub_progress.setValue(0)
        self._hub_progress.setVisible(True)
        self._hub_status.setText(f"Starting download of {entry['filename']}...")
        self._gc.request_hub_download(
            {
                "repo_id": repo["repo_id"],
                "filename": entry["filename"],
                "size_bytes": entry.get("size_bytes"),
                "sha256": entry.get("sha256"),
                "verification": entry.get("verification", "none"),
                "confirm_unverified": entry.get("verification") == "none",
                "confirmed_exceeds": fit.get("band") == "exceeds",
                "register": self._hub_register_box.isChecked(),
            }
        )

    def _on_hub_cancel(self):
        """Ask the running download to stop. The partial file is DELETED, not kept.

        There is no resume (models_hub.resume_enabled is RESERVED/NOT
        IMPLEMENTED); modelhub deletes the .partial on cancel, which is what the
        confirmation text the user just accepted says.
        """
        self._hub_cancel_btn.setEnabled(False)
        self._hub_status.setText("Canceling...")
        self._gc.request_hub_cancel()

    def _confirm_box(self, title, text):
        """Build the yes/no dialog without showing it. Seam for headless tests.

        Separate from `_confirm` so a test can inspect exactly what the user
        would see - the rendered text and its format - without executing a modal
        loop. SEC-M14-1: this dialog carries HuggingFace-authored text (repo id,
        file name, licence tag), so its body must never be interpreted as markup.
        """
        return plain_message_box(
            self,
            title,
            text,
            icon=QtWidgets.QMessageBox.Icon.Question,
            buttons=QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            default=QtWidgets.QMessageBox.StandardButton.No,
        )

    def _confirm(self, title, text):
        """Show a modal yes/no whose default is No. Seam for headless tests."""
        answer = self._confirm_box(title, text).exec()
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _apply_hub_catalog(self, result):
        """Render the shipped catalog. Reached on page paint; no network involved."""
        items = result.payload.get("items", [])
        self._fill_hub_repos(items)
        reason = result.payload.get("reason", "")
        self._hub_status.setText(
            reason or f"{len(items)} catalog entries."
        )

    def _apply_hub_search(self, result):
        """Render search results, or the honest offline/rate-limit line."""
        self._hub_search_btn.setEnabled(self._hub_enabled)
        if not result.ok:
            # The catalog rows stay exactly where they are: being offline must
            # not empty the page the user was already browsing.
            self._hub_status.setText(
                result.payload.get("reason") or result.error or "search failed"
            )
            return
        items = result.payload.get("items", [])
        self._fill_hub_repos(items)
        self._set_hub_has_more(result.payload.get("has_more", False))
        self._hub_status.setText(
            f"{len(items)} results for '{result.payload.get('query', '')}'."
            if items
            else "No GGUF repositories matched that search."
        )

    def _apply_hub_search_more(self, result):
        """Render an appended page of results (already merged by the worker)."""
        self._hub_load_more_btn.setText("Load more results")
        if not result.ok:
            # The rows already on screen (from the prior page) stay put; a
            # failed Load more press must not lose what was already found.
            self._hub_load_more_btn.setEnabled(self._hub_has_more)
            self._hub_status.setText(
                result.payload.get("reason") or result.error or "could not load more"
            )
            return
        items = result.payload.get("items", [])
        self._fill_hub_repos(items)
        self._set_hub_has_more(result.payload.get("has_more", False))
        self._hub_status.setText(
            f"{len(items)} results for '{result.payload.get('query', '')}'."
        )

    def _set_hub_has_more(self, has_more):
        """Show/hide and (re)enable the Load more button to match server state."""
        self._hub_has_more = bool(has_more)
        self._hub_load_more_btn.setVisible(self._hub_has_more)
        self._hub_load_more_btn.setEnabled(self._hub_has_more and self._hub_enabled)

    def _apply_hub_files(self, result):
        """Render one repository's downloadable files."""
        self._hub_file_table.setRowCount(0)
        self._hub_file_rows = []
        if not result.ok:
            self._hub_status.setText(
                result.payload.get("reason") or result.error or "could not list files"
            )
            return
        rows = result.payload.get("items", [])
        self._hub_file_rows = rows
        self._hub_file_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            size_gb = (row.get("size_bytes") or 0) / 1_000_000_000
            fit = row.get("fit") or {}
            for column, value in enumerate(
                [
                    row.get("filename", ""),
                    row.get("quant", ""),
                    f"{size_gb:.1f} GB",
                    fit.get("wording", ""),
                    row.get("verification", ""),
                ]
            ):
                self._hub_file_table.setItem(
                    index, column, QtWidgets.QTableWidgetItem(str(value))
                )
        self._hub_status.setText(
            f"{len(rows)} downloadable files in {result.payload.get('repo_id', '')}."
            if rows
            else "That repository has no single-file .gguf locitize can download."
        )

    def _fill_hub_repos(self, items):
        """Populate the discovery table from catalog or search rows."""
        self._hub_repo_rows = list(items)
        self._hub_repo_table.setRowCount(len(items))
        for index, row in enumerate(items):
            for column, value in enumerate(
                [
                    row.get("repo_id", ""),
                    row.get("publisher", ""),
                    row.get("license_tag", "") or "not stated",
                    "huggingface.co" if row.get("source") == "api" else "built-in list",
                ]
            ):
                self._hub_repo_table.setItem(
                    index, column, QtWidgets.QTableWidgetItem(str(value))
                )

    def _hub_busy_job(self):
        """The job this panel is committed to: the confirmed one, else the pending one.

        One helper so every "is a download in the way?" question has a single
        answer. Active wins over pending because a confirmed transfer is the
        thing the live controls belong to.
        """
        return self._hub_active_job or self._hub_pending_job

    def _apply_hub_progress(self, result):
        """Show phase, bytes and ETA. Arrives at most twice a second by design.

        This is also where a requested job becomes the ACTIVE one (NEW-QA-M14-7):
        progress for a job id is the controller's proof that it accepted the
        request and bytes are moving, which is exactly the moment the panel's
        live controls start belonging to that job. Progress for any other job is
        ignored rather than painted over the running one.
        """
        payload = result.payload
        job = payload.get("job_id") or ""
        if job:
            if self._hub_active_job is None and job == self._hub_pending_job:
                self._hub_active_job = job
                self._hub_pending_job = None
            elif self._hub_active_job is not None and job != self._hub_active_job:
                return
        done = payload.get("bytes_done") or 0
        total = payload.get("bytes_total")
        if total:
            self._hub_progress.setValue(int(100 * done / total))
        rate = (payload.get("rate_bps") or 0) / 1_000_000
        eta = payload.get("eta_s")
        # "Verifying" is called out because hashing several gigabytes takes real
        # time and a silent progress bar reads as a hang.
        phase = payload.get("phase", "")
        text = f"{phase}: {done / 1_000_000_000:.2f} GB"
        if total:
            text += f" of {total / 1_000_000_000:.2f} GB"
        if rate:
            text += f" at {rate:.1f} MB/s"
        if eta:
            text += f", about {int(eta // 60)} min left"
        self._hub_status.setText(text)

    def _apply_hub_done(self, result):
        """Render the terminal outcome: success, cancellation, or an honest failure.

        Terminal UI (Cancel off, progress bar hidden, Download re-armed) is
        applied ONLY for the job this panel is currently showing. A result about
        any other job - in practice a request refused with "A download is
        already running." - is a status line and nothing more. Applying it
        unconditionally is DEFECT-QA-M14-2: it disabled Cancel and hid the
        progress bar of a transfer that was still running, so the user could
        neither watch nor stop a 1.5 GB download.

        The guard is load-bearing again since NEW-QA-M14-7: because a requested
        job is no longer written over the live one, the refusal of a second
        request now genuinely carries a DIFFERENT id from the running transfer,
        so this comparison is false where it used to be true.
        """
        payload = result.payload
        job = payload.get("job_id") or ""
        active = self._hub_active_job
        if active is not None and job and job != active:
            self._hub_status.setText(
                payload.get("message") or result.error or "the download failed"
            )
            # A refused request is over: it never becomes the active job, and it
            # must not leave the Download button disabled forever.
            if job == self._hub_pending_job:
                self._hub_pending_job = None
            return
        self._hub_active_job = None
        self._hub_pending_job = None
        self._hub_cancel_btn.setEnabled(False)
        self._hub_progress.setVisible(False)
        self._hub_download_btn.setEnabled(self._selected_hub_file() is not None)
        if not result.ok:
            self._hub_status.setText(
                payload.get("message") or result.error or "the download failed"
            )
            return
        parts = [f"Saved to {payload.get('path', '')}."]
        # The rung label IS the whole provenance claim on this line, and it is
        # read from modelhub rather than restated here so the completion line
        # and the file-row label can never drift apart (UX Spec section 14).
        #
        # No hex excerpt of any length belongs here: 16 characters of a sha256
        # cannot be checked by eye against anything on screen, and hash-shaped
        # text sitting exactly where the reader looks for provenance reads as
        # proof regardless of which rung produced it. The full digests still
        # appear on a checksum MISMATCH, where they are diagnostics to paste
        # into a bug report rather than a claim about what was verified.
        rung = payload.get("verification") or "none"
        # An unrecognised rung falls back to the unverified label: the failure
        # mode of a future typo must be under-claiming, never over-claiming.
        parts.append(VERIFICATION_LABELS.get(rung, VERIFICATION_LABELS["none"]))
        if payload.get("registered"):
            parts.append(f"Registered as '{payload.get('model_id')}'.")
        elif payload.get("register_error"):
            parts.append(payload["register_error"])
        self._hub_status.setText(" ".join(parts))
        # The new row exists in models.yaml but not yet in the table above, so
        # re-read it here rather than making the user find the Refresh button.
        if payload.get("registered"):
            self._on_refresh_models()

    def _on_finetune_open_folder(self):
        """Open the selected run's folder in the OS file browser."""
        row = self._selected_finetune()
        if row is None:
            return
        folder = QtCore.QFileInfo(row["path"]).absolutePath()
        url = QtCore.QUrl.fromLocalFile(folder)
        if not QtGui.QDesktopServices.openUrl(url):
            # Deleted between scan and click: an inline message, never a crash.
            self._ft_status.setText(f"could not open {folder}")

    def _set_finetune_buttons(self, start, stop, open_browser):
        """Set the three lifecycle buttons together (they are never mixed states)."""
        self._ft_start_btn.setEnabled(bool(start))
        self._ft_stop_btn.setEnabled(bool(stop))
        self._ft_open_btn.setEnabled(bool(open_browser))

    def _refresh_finetune_buttons(self):
        """Enable the row actions only when a usable row is selected."""
        row = self._selected_finetune()
        has_row = row is not None
        self._ft_serve_btn.setEnabled(has_row)
        self._ft_register_btn.setEnabled(has_row and not row.get("registered"))
        self._ft_delete_btn.setEnabled(has_row)
        self._ft_folder_btn.setEnabled(has_row)

    def _apply_finetune_state(self, payload):
        """Render one studio lifecycle snapshot (the exact state matrix, UX 3.1)."""
        status = payload.get("status", "stopped")
        reason = payload.get("reason", "") or ""
        url = payload.get("url")
        if status == "disabled":
            # Not configured / unavailable: the reason string names the exact
            # setting or env var that fixes it.
            self._ft_chip.setText(reason or "Fine-tune studio is not configured.")
            self._set_finetune_buttons(False, False, False)
        elif status == "starting":
            self._ft_chip.setText("Starting studio...")
            self._set_finetune_buttons(False, False, False)
        elif status == "running":
            self._ft_chip.setText(f"Running - {url}")
            self._set_finetune_buttons(False, True, True)
        elif status == "error":
            self._ft_chip.setText(reason or "Did not start - see logs/finetune_studio.log")
            self._set_finetune_buttons(True, False, False)
        else:
            self._ft_chip.setText("Stopped")
            self._set_finetune_buttons(True, False, False)
        log_path = payload.get("log_path") or ""
        self._ft_log_label.setText(f"log: {log_path}" if log_path else "")
        self._ft_log_label.setVisible(bool(log_path))

    def _apply_finetune_result(self, result):
        """Apply a finetune_state Result, including the honest stop-time warning."""
        payload = dict(result.payload)
        if not result.ok and result.error:
            payload["status"] = "error"
            payload["reason"] = result.error
        self._apply_finetune_state(payload)
        # An informational message (e.g. the Open action when open_browser is off)
        # goes to the page's status line so no click ever looks like it did nothing.
        message = payload.get("message", "")
        if message:
            self._ft_status.setText(message)
        warning = payload.get("warning", "")
        self._ft_warning.setText(warning)
        self._ft_warning.setVisible(bool(warning))

    def _apply_finetune_models(self, result):
        """Render the discovered-models table (empty / error / populated states)."""
        self._ft_rescan_btn.setEnabled(True)
        payload = result.payload
        self._finetunes = payload.get("items", [])
        reason = payload.get("reason", "") or ""
        self._ft_table.blockSignals(True)
        self._ft_table.setRowCount(len(self._finetunes))
        for row_index, row in enumerate(self._finetunes):
            values = (
                row["name"],
                "Discovered",
                row["quant"],
                row["size_display"],
                row["modified_display"],
                "Registered" if row.get("registered") else "Not registered",
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, row["id"])
                if column in (3, 4):
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                self._ft_table.setItem(row_index, column, item)
        self._ft_table.blockSignals(False)
        if self._finetunes:
            self._ft_status.setText(f"{len(self._finetunes)} fine-tuned models found")
        else:
            # The controller's own reason string is rendered verbatim so the owner
            # always learns why the list is empty.
            self._ft_status.setText(reason or "No fine-tuned models found.")
        self._refresh_finetune_buttons()

    # Column index of the Fine-tune table's Status/Actions cell, kept next to the
    # one place that repaints it out of band (see _apply_finetune_register_result).
    _FT_STATUS_COLUMN = 5

    def _mark_finetune_registered(self, key):
        """Flip one cached fine-tune row to registered and repaint its Status cell.

        The table is built row-for-row from self._finetunes and is never sorted,
        so the list index is the table row index. Repainting the single cell here
        (rather than re-running a whole scan) keeps the confirmation instant and
        leaves the discovered duplicate in place until the owner clicks Rescan,
        exactly as the status message promises.
        """
        if not key:
            return
        for index, row in enumerate(self._finetunes):
            if row.get("id") != key:
                continue
            row["registered"] = True
            item = self._ft_table.item(index, self._FT_STATUS_COLUMN)
            if item is not None:
                item.setText("Registered")
            return

    def _apply_finetune_register_result(self, result):
        """Confirm or honestly refuse one Register action."""
        if not result.ok:
            # Only the failure path hands the controls back: the owner may want to
            # retry after fixing whatever the refusal named.
            self._refresh_finetune_buttons()
            self._register_btn.setEnabled(True)
            message = result.error or "could not register"
            self._ft_status.setText(message)
            self._model_status.setText(message)
            return
        model_id = result.payload.get("model_id", "")
        # UX Spec section 4 step 3: the row itself must show the new state now.
        # Leaving it on "Not registered" with a live Register button invites a
        # second click, whose duplicate-id refusal reads as a failure of an action
        # that actually succeeded. The discovered duplicate still only disappears
        # on the next Rescan, which is what the status message below says.
        self._mark_finetune_registered(result.payload.get("key", ""))
        self._refresh_finetune_buttons()
        message = (
            f"Registered as '{model_id}'. "
            f"Click Rescan to refresh this list."
        )
        self._ft_status.setText(message)
        self._model_status.setText(message)
        self._on_refresh_models()

    def _drain(self):
        """Apply every Result from the controller's one existing result queue."""
        while True:
            try:
                result = self._gc.result_q.get_nowait()
            except gui_controller.queue.Empty:
                break
            self._apply(result)

    def _apply(self, result):
        """Render one controller Result; called only by the GUI-thread timer."""
        if result.kind == "session_launch":
            self._sessions_page.apply_launch_result(result)
            return
        kind = result.kind
        if kind == "status_line":
            # M15.7: a passive notice (e.g. models.yaml changed on disk).
            # Status text only - never a dialog, never an automatic reload.
            self._model_status.setText(str(result.payload.get("text", "")))
            return
        if kind == "running_model":
            # Owner-observed 2026-09-03: a model started by the model router
            # (a browser picker change) left this window showing no running
            # model at all. The worker now polls the controller and publishes
            # this whenever the answer changes.
            #
            # Deliberately does NOT touch in_flight: this is a passive
            # observation, and clearing the flag would re-enable the buttons
            # underneath a start the OWNER is still waiting on.
            previous = self._ui.running_model_id
            self._apply_running_snapshot(result.payload)
            if previous != self._ui.running_model_id:
                self._ui.latest_metrics = None
                if self._ui.running_model_id is None:
                    self._render_monitor_text("idle")
            self._refresh_model_rows()
            self._refresh_buttons()
            return
        if kind == "start":
            self._ui.in_flight = False
            self._apply_running_snapshot(result.payload)
            # On success, say nothing here -- the header status chip already
            # shows "model: <id> port <port>", so repeating it was redundant.
            self._model_status.setText(
                "" if result.ok else (result.error or "start failed")
            )
            self._refresh_model_rows()
            self._refresh_buttons()
        elif kind == "stop":
            self._ui.in_flight = False
            self._apply_running_snapshot(result.payload)
            self._assistant_port = None
            self._ui.latest_metrics = None
            self._render_monitor_text("idle")
            self._model_status.setText(
                "stopped" if result.ok else (result.error or "stop failed")
            )
            self._refresh_model_rows()
            self._refresh_buttons()
        elif kind == "gpu_free":
            self._apply_gpu_free(result)
        elif kind == "system_sample":
            self._apply_system_sample(result.payload)
        elif kind == "metrics":
            self._render_metrics(result.payload.get("sample"))
        elif kind == "whisper":
            self._ui.whisper_running = bool(result.payload.get("running"))
            self._whisper_btn.setText(
                "Stop whisper server"
                if self._ui.whisper_running
                else "Start whisper server"
            )
            self._model_status.setText(
                result.error
                or ("whisper running" if self._ui.whisper_running else "whisper stopped")
            )
        elif kind == "transcript":
            line = result.payload.get("line", "")
            if line:
                self._voice_status.setText(line)
                append_plain(self._transcript, line)
        elif kind == "listen":
            self._model_status.setText(
                "listen finished" if result.ok else (result.error or "listen failed")
            )
            if not result.ok:
                self._voice_status.setText(result.error or "listen failed")
        elif kind == "speak":
            voice = result.payload.get("voice", "")
            self._tts_status.setText(
                f"spoke in {voice}" if result.ok else (result.error or "speak failed")
            )
        elif kind == "audition_voice":
            self._tts_status.setText(f"speaking: {result.payload.get('voice', '')}")
        elif kind == "audition":
            self._tts_status.setText(
                f"auditioned {result.payload.get('count', 0)} voices"
                if result.ok
                else (result.error or "audition failed")
            )
        elif kind == "save_edits":
            self._apply_save_result(result)
        elif kind == "save_identity":
            self._apply_identity_result(result)
        elif kind == "save_capabilities":
            self._apply_capabilities_result(result)
        elif kind == "delete_model":
            self._apply_delete_model_result(result)
        elif kind == "autotune_progress":
            self._apply_autotune_progress(result)
        elif kind == "autotune":
            self._apply_autotune_result(result)
        elif kind == "autotune_all":
            self._apply_autotune_all_result(result)
        elif kind == "finetune_state":
            self._apply_finetune_result(result)
        elif kind == "finetune_models":
            self._apply_finetune_models(result)
        elif kind == "finetune_register_result":
            self._apply_finetune_register_result(result)
        elif kind == "feature_state":
            self._apply_feature_state(result)
        elif kind == "finetune_delete_result":
            # M18.7: one status line; the table itself refreshes via the
            # finetune_models scan the worker emits right after the delete.
            message = result.payload.get("message", "") or result.error or ""
            self._ft_status.setText(message)
        elif kind == "hub_catalog":
            self._apply_hub_catalog(result)
        elif kind == "hub_search":
            self._apply_hub_search(result)
        elif kind == "hub_search_more":
            self._apply_hub_search_more(result)
        elif kind == "hub_files":
            self._apply_hub_files(result)
        elif kind == "hub_download_progress":
            self._apply_hub_progress(result)
        elif kind == "hub_download_done":
            self._apply_hub_done(result)
        elif kind == "chat":
            self._apply_chat_result(result)
        elif kind == "chat_status":
            message = result.payload.get("message", "")
            self._model_status.setText(message)
        elif kind == "chat_ask":
            self._show_chat_chooser()
        elif kind == "chat_offer_start":
            self._show_openwebui_start_offer(result.payload.get("reason", ""))
        elif kind == "launch_harness":
            self._apply_launch_harness_result(result)
        elif kind == "install_harness":
            self._apply_install_harness_result(result)
        elif kind == "assistant_started":
            self._on_assistant_started(result.payload)
        elif kind == "assistant_error":
            self._on_assistant_error(result.error or "assistant error")
        elif kind == "assistant_ended":
            self._on_assistant_ended()
        elif kind == "assistant_state":
            self._set_talk_state(result.payload.get("state", "idle"))
        elif kind == "assistant_user":
            self._append_conversation("You: ", result.payload.get("text", ""))
        elif kind == "assistant_reply":
            self._append_conversation("", result.payload.get("line", ""))
        elif kind == "describe":
            self._vision_result.setPlainText(
                result.payload.get("answer", "")
                if result.ok
                else (result.error or "describe failed")
            )
            self._refresh_model_exclusive_buttons()
        elif kind == "memory_search":
            self._render_memory_hits(result)
        elif kind == "noise_suppression":
            mode = result.payload.get("mode", "")
            self._noise_status.setText(
                f"Using {mode}" if result.ok else (result.error or "could not save")
            )
        elif kind == "second_eye":
            running = bool(result.payload.get("running"))
            self._second_eye_start_btn.setEnabled(not running)
            self._second_eye_stop_btn.setEnabled(running)
            if not result.ok:
                self._second_eye_status.setText(result.error or "watch failed")
            elif running:
                goal = result.payload.get("goal", "")
                self._second_eye_status.setText(
                    f"watching: {goal}" if goal else "watching"
                )
            else:
                self._second_eye_status.setText("not watching")
        elif kind == "benchmark":
            self._apply_benchmark_result(result)

    def _apply_running_snapshot(self, payload):
        """Trust the controller's post-operation snapshot instead of guessing."""
        self._ui.running_model_id = payload.get("running_model_id")
        self._ui.running_port = payload.get("running_port")
        self._update_chat_hint()

    def _update_chat_hint(self):
        """Keep the Chat page hint honest about whether a model is running."""
        if not hasattr(self, "_chat_hint"):
            return
        running = self._ui.running_model_id
        if running:
            self._chat_hint.setText(
                f"{running} is running. Open Chat uses the rich UI when it is "
                "installed, or the built-in model UI."
            )
        else:
            self._chat_hint.setText(
                "Start a model on Models first. Chat then opens the rich chat "
                "application when it is installed, or the built-in model UI."
            )

    def _on_sidebar_changed(self, index):
        """Switch pages and run first-visit actions (Memory search, header)."""
        self._stack.setCurrentIndex(index)
        name = PAGE_NAMES[index] if 0 <= index < len(PAGE_NAMES) else ""
        if name == "Sessions":
            self._sessions_page.visit()
        if hasattr(self, "_header_offload_btn"):
            self._header_offload_btn.setVisible(name != "Talk")
        if name == "Memory" and not self._memory_visited:
            self._memory_visited = True
            self._on_memory_search()

    def _on_noise_mode_changed(self, _index=None):
        """Persist the Voice Setup noise-suppression preset."""
        mode = self._noise_combo.currentData() or "balanced"
        setter = getattr(self._gc, "set_noise_suppression", None)
        if not callable(setter):
            self._noise_status.setText("this session cannot change the preset")
            return
        try:
            setter(str(mode))
            self._noise_status.setText(f"Using {mode}")
        except Exception as exc:  # noqa: BLE001 - a settings write must not kill the UI
            self._noise_status.setText(str(exc))

    def _on_second_eye_start(self):
        """Start the screen watcher with the typed goal."""
        starter = getattr(self._gc, "request_second_eye_start", None)
        if not callable(starter):
            self._second_eye_status.setText("Watch my screen is not available")
            return
        self._second_eye_status.setText("starting...")
        starter(self._second_eye_goal.text(), 1.0)

    def _on_second_eye_stop(self):
        """Ask the screen watcher to exit."""
        stopper = getattr(self._gc, "request_second_eye_stop", None)
        if callable(stopper):
            stopper()
            self._second_eye_status.setText("stopping...")

    def _render_metrics(self, sample):
        """Render real metrics or an explicit unavailable message on all surfaces."""
        if sample is None:
            return
        self._ui.latest_metrics = sample
        if (
            not getattr(sample, "metrics_available", False)
            and getattr(sample, "slots", None) is None
        ):
            text = (
                "this llama.cpp build does not report live speed; "
                "Auto-tune still measures it"
            )
        else:
            text = gui_controller.format_monitor_line(
                sample, self._selected_context_size()
            )
        self._render_monitor_text(text)
        self._header_status.setText(self._health_line())

    def _render_monitor_text(self, text):
        """Keep the expanded Models and Talk monitor lines synchronized."""
        self._models_monitor.setText(text)
        self._talk_monitor.setText(text)

    def _apply_save_result(self, result):
        """Update the cached persisted values so the grey Save button confirms success."""
        if not result.ok:
            self._edit_error.setText(result.error or "save failed")
            return
        model_id = result.payload.get("model_id")
        for model in self._models:
            if model["id"] == model_id:
                model["gpu_layers"] = result.payload.get(
                    "gpu_layers", model["gpu_layers"]
                )
                model["context_size"] = result.payload.get(
                    "context_size", model["context_size"]
                )
                break
        self._edit_error.clear()
        self._on_edit_change()

    def _apply_identity_result(self, result):
        """Update cached model rows and the Settings picker after an id/name rename."""
        if not result.ok:
            self._identity_error.setText(result.error or "rename failed")
            return
        old_id = result.payload.get("model_id")
        new_id = result.payload.get("new_id", old_id)
        new_name = result.payload.get("new_name", "")
        for model in self._models:
            if model["id"] == old_id:
                model["id"] = new_id
                model["name"] = new_name
                break
        if self._selected_id == old_id:
            self._selected_id = new_id
        combo_index = self._settings_model.findData(old_id)
        if combo_index != -1:
            self._settings_model.setItemData(combo_index, new_id)
            self._settings_model.setItemText(combo_index, new_name)
        self._identity_error.clear()
        self._refresh_model_rows()
        self._on_identity_change()

    def _apply_capabilities_result(self, result):
        """Update the cached model row and table after a capabilities write."""
        if not result.ok:
            self._capabilities_error.setText(result.error or "save failed")
            return
        model_id = result.payload.get("model_id")
        capabilities = result.payload.get("capabilities", [])
        for model in self._models:
            if model["id"] == model_id:
                model["capabilities"] = list(capabilities)
                model["capabilities_display"] = gui_controller.format_capabilities(capabilities)
                break
        self._capabilities_error.clear()
        self._refresh_model_rows()
        self._on_capabilities_change()

    def _apply_delete_model_result(self, result):
        """Remove the model from every UI surface after a confirmed Delete."""
        if not result.ok:
            self._model_status.setText(result.error or "delete failed")
            self._refresh_buttons()
            return
        model_id = result.payload.get("model_id")
        deleted = result.payload.get("deleted_files", [])
        skipped = result.payload.get("skipped_shared_files", [])
        self._models = [m for m in self._models if m["id"] != model_id]
        combo_index = self._settings_model.findData(model_id)
        if combo_index != -1:
            self._settings_model.removeItem(combo_index)
        if self._selected_id == model_id:
            self._selected_id = None
        self._refresh_model_rows()
        self._select_initial_model()
        summary = f"Deleted ({len(deleted)} file(s) removed"
        if skipped:
            summary += f", {len(skipped)} kept - still used by another model"
        summary += ")"
        self._model_status.setText(summary)
        self._models_total.setText(gui_controller.total_size_display(self._models))

    def _apply_chat_result(self, result):
        """Surface chat success, fallback, or a copyable URL without hiding errors."""
        # M18.14 (fresh-clone report): when chat falls back to the built-in UI
        # because Open WebUI was never installed, the browser tab gives no hint
        # that a richer chat exists - a fresh user who skipped the wizard
        # checkbox silently loses the flagship experience. Offer the install
        # path ONCE per session: a small dialog naming what is missing and the
        # exact way to add it, never repeated, never blocking the chat that
        # already opened.
        reason_text = (result.payload or {}).get("reason", "") if result.ok else ""
        if (
            "not installed" in reason_text
            and "Open WebUI" in reason_text
            and not getattr(self, "_webui_install_hint_shown", False)
        ):
            self._webui_install_hint_shown = True
            offer = QtWidgets.QMessageBox(self)
            offer.setWindowTitle("Rich chat UI available")
            offer.setIcon(QtWidgets.QMessageBox.Icon.Information)
            offer.setText(
                "The built-in chat just opened. locitize also supports Open "
                "WebUI - conversation history, uploads, web search - but it is "
                "not installed yet.\n\nInstall it from the setup wizard? (One "
                "checkbox; about 2.6 GB. Chat keeps working either way.)"
            )
            offer.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
            )
            offer.button(QtWidgets.QMessageBox.StandardButton.Yes).setText("Open setup wizard")
            offer.button(QtWidgets.QMessageBox.StandardButton.No).setText("Not now")
            offer.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
            if harden_text_rendering(offer).exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                self._on_open_setup_wizard()
        if not result.ok:
            message = result.error or "cannot open chat"
            self._model_status.setText(message)
            self._chat_status.setText(message)
            return
        url = result.payload.get("url", "")
        reason = result.payload.get("reason", "")
        if result.payload.get("opened", True) is False:
            # The URL is controller-supplied; plain text keeps it literal.
            plain_message_box(
                self,
                "locitize chat",
                f"Open this in your browser:\n{url}",
                icon=QtWidgets.QMessageBox.Icon.Information,
            ).exec()
            self._chat_status.setText(f"open this in your browser: {url}")
        elif reason:
            # An actual fallback explanation is worth surfacing; a plain
            # "chat opened" echo is not -- the browser/window opening is
            # already visible, so that case is silent on purpose.
            self._model_status.setText(reason)
            self._chat_status.setText(reason)

    def _show_chat_chooser(self):
        """Render the controller-requested chat/harness choice as a modal dialog.

        Owner request 2026-08-21: alongside the two built-in chat surfaces,
        offer "Launch in Claude Code / Codex / OpenCode" - each opens a real
        terminal running that CLI, pointed at the running model, working on a
        folder the owner picks (the same "walk a project in a folder" the
        harnesses already do against the real APIs). A harness not found on
        PATH is offered disabled rather than hidden, so its absence is visible
        and explained instead of silently missing.
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Choose a chat interface")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Which chat interface would you like to open?"))
        built_in = QtWidgets.QRadioButton("llama.cpp web UI (built-in, zero setup)")
        rich = QtWidgets.QRadioButton("Open WebUI (rich chat, history)")
        built_in.setChecked(True)
        layout.addWidget(built_in)
        layout.addWidget(rich)

        layout.addWidget(QtWidgets.QLabel("Launch in a coding harness (works on a project folder):"))
        installed = self._gc.detect_harnesses()
        harness_labels = {
            "claude": "Claude Code",
            "codex": "Codex",
            "opencode": "OpenCode",
        }
        harness_buttons: dict[str, QtWidgets.QRadioButton] = {}
        for key, label in harness_labels.items():
            found = installed.get(key)
            row = QtWidgets.QHBoxLayout()
            button = QtWidgets.QRadioButton(label if found else f"{label} (not found on PATH)")
            button.setEnabled(bool(found))
            row.addWidget(button)
            if not found:
                # Owner request 2026-08-21: "the user should not have to do
                # anything" - a missing harness gets an install path right
                # here instead of a dead end. dialog.accept() first so the
                # background install (harness_launch.install_harness, on the
                # ops worker) is never blocked behind this modal dialog.
                install_btn = QtWidgets.QPushButton("Install")
                install_btn.clicked.connect(
                    lambda _checked=False, k=key, lbl=label: self._confirm_and_install_harness(
                        dialog, k, lbl
                    )
                )
                row.addWidget(install_btn)
            row.addStretch()
            layout.addLayout(row)
            harness_buttons[key] = button

        remember = QtWidgets.QCheckBox("Remember my choice")
        layout.addWidget(remember)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Open
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if harden_text_rendering(dialog).exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        for key, button in harness_buttons.items():
            if button.isChecked():
                self._launch_harness_with_folder(key, remember.isChecked())
                return
        choice = "openwebui" if rich.isChecked() else "llamacpp"
        self._gc.open_chat(choice, remember.isChecked())

    def _launch_harness_with_folder(self, choice, remember):
        """Ask for the project folder (defaulting to the remembered one), then
        queue the launch. The folder picker is per-launch on purpose - the
        remembered path is only a default, never silently reused without the
        owner seeing which folder is about to be opened."""
        default_dir = self._gc.last_project_dir() or str(Path.home())
        project_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a project folder", default_dir
        )
        if not project_dir:
            return
        self._model_status.setText(f"launching {choice}...")
        self._gc.request_launch_harness(choice, project_dir, remember)

    def _apply_launch_harness_result(self, result):
        """Surface a harness launch's outcome, including the Claude-bridge's
        stated tool-use limitation, without hiding a failure."""
        if not result.ok:
            message = result.error or "could not launch harness"
            self._model_status.setText(message)
            self._chat_status.setText(message)
            return
        choice = result.payload.get("choice", "")
        note = result.payload.get("note", "")
        message = f"{choice} launched in a new terminal"
        self._model_status.setText(message)
        self._chat_status.setText(note or message)

    def _refresh_chat_harness_buttons(self):
        """Enable each Chat-page harness button per a fresh PATH detection.

        Defensive by the same rule as harness onboarding: a detection failure
        (or a controller without the probe at all) must never break the window -
        the buttons just stay disabled with the install hint.
        """
        if not getattr(self, "_chat_harness_buttons", None):
            return
        try:
            installed = self._gc.detect_harnesses()
        except Exception:  # noqa: BLE001 - degrade to disabled, never crash
            installed = {}
        if not isinstance(installed, dict):
            installed = {}
        for key, button in self._chat_harness_buttons.items():
            found = installed.get(key)
            button.setEnabled(bool(found))
            if found:
                button.setToolTip(f"found at {found}")
            else:
                button.setToolTip(
                    "Not found on PATH. Install with: "
                    + HARNESS_INSTALL_SUMMARIES.get(key, "")
                )

    def _apply_install_harness_result(self, result):
        """Surface an "Install <harness>" outcome (owner request 2026-08-21:
        zero-friction onboarding). A failure always carries a concrete remedy
        string from harness_launch.py, never a bare "failed" - shown as-is.
        """
        harness = result.payload.get("harness", "")
        if not result.ok:
            message = result.error or f"could not install {harness}"
            self._model_status.setText(message)
            self._chat_status.setText(message)
            return
        message = result.payload.get("message", f"{harness} installed")
        self._model_status.setText(message)
        self._chat_status.setText(message)
        self._refresh_chat_harness_buttons()
        self._refresh_cli_status_rows()

    def _apply_feature_state(self, result):
        """Render the ops worker's feature detection into the Settings rows."""
        if not hasattr(self, "_feature_rows_grid"):
            return
        if not result.ok:
            self._feature_state_note.setText(result.error or "detection failed")
            return
        while self._feature_rows_grid.count():
            item = self._feature_rows_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for row_index, feature in enumerate(result.payload.get("features", [])):
            name = QtWidgets.QLabel(feature.get("label", feature.get("key", "?")))
            self._feature_rows_grid.addWidget(name, row_index, 0)
            installed = bool(feature.get("installed"))
            status = QtWidgets.QLabel("Installed" if installed else "Not installed")
            status.setObjectName("statusChip")
            if not installed:
                status.setStyleSheet("color: #ffd166;")
            self._feature_rows_grid.addWidget(status, row_index, 1)
        self._feature_state_note.setText("")

    def _refresh_cli_status_rows(self):
        """Fill the Settings CLI rows from a fresh PATH detection (defensive)."""
        if not getattr(self, "_cli_status_labels", None):
            return
        try:
            installed = self._gc.detect_harnesses()
        except Exception:  # noqa: BLE001 - detection failure shows as unknown
            installed = {}
        for key, label in self._cli_status_labels.items():
            found = (installed or {}).get(key)
            label.setText(found if found else "not installed")

    def _on_uninstall(self):
        """Launch the stdlib uninstaller and close the desktop (M18.16).

        The uninstaller runs under the bootstrap python - never this venv,
        which is one of the things it deletes - and this window closes so no
        running process locks the folders it is about to remove.
        """
        import subprocess  # noqa: PLC0415

        from runtime_layout import bundled_python
        if bundled_python():
            QtWidgets.QMessageBox.information(
                self, "Remove this beta",
                "Close locitize, then remove this version's installation folder and its shortcuts. "
                "Keep the separate locitize data folder to preserve models, histories and session notes. "
                "A registered Windows uninstaller is planned for the production release.")
            return
        script = Path(__file__).resolve().parent / "uninstall.py"
        if not script.is_file():
            self._edit_error.setText(f"uninstaller not found at {script}")
            return
        answer = QtWidgets.QMessageBox.warning(
            self,
            "Uninstall locitize",
            "The uninstall window will open next, and locitize will close so "
            "its files can be removed. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            subprocess.Popen(
                ["python", str(script)],
                cwd=str(script.parent),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self._edit_error.setText(f"could not start the uninstaller: {exc}")
            return
        self.close()

    def _on_open_setup_wizard(self):
        """Open the setup wizard beside the running desktop (M18.12).

        The wizard runs in its own console python (it may need to install into
        the venv this desktop is running from), so it is launched detached with
        a visible console; installed features appear after the next LOCITIZE
        restart, and the status text says exactly that.
        """
        import subprocess  # noqa: PLC0415 - only needed for this launch

        bat = Path(__file__).resolve().parent / "locitize.bat"
        from runtime_layout import bundled_python
        if bundled_python():
            try:
                subprocess.Popen([sys.executable, str(bat.with_name("setup_wizard.py"))],
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self._feature_state_note.setText("Setup wizard opened. Restart locitize after adding features.")
            except OSError as exc:
                self._edit_error.setText(f"Could not open setup: {exc}")
            return
        if not bat.is_file():
            self._edit_error.setText(f"setup launcher not found at {bat}")
            return
        try:
            # No console window: the wizard is a window of its own.
            subprocess.Popen(
                ["cmd", "/c", str(bat), "--setup"],
                cwd=str(bat.parent),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._feature_state_note.setText(
                "setup wizard opened in its own window; restart locitize after "
                "installing to use new features"
            )
        except OSError as exc:
            self._edit_error.setText(f"could not open the setup wizard: {exc}")

    def _confirm_and_install_harness(self, source_dialog, key, label):
        """One explicit confirmation naming the exact command, THEN install.

        Security review 2026-08-21 (fix): the picker's "Install" button used
        to fire the real installer on a single click with no confirmation at
        all - less visibility than the onboarding dialog had even before ITS
        own fix. Same command text as HARNESS_INSTALL_SUMMARIES, same
        default-to-declining posture (QMessageBox.No is the default focused
        button).
        """
        summary = HARNESS_INSTALL_SUMMARIES.get(key, "")
        confirm = QtWidgets.QMessageBox(self)
        confirm.setWindowTitle(f"Install {label}?")
        confirm.setIcon(QtWidgets.QMessageBox.Icon.Question)
        confirm.setText(f"This will run:\n\n{summary}")
        confirm.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        confirm.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        if harden_text_rendering(confirm).exec() != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        # reject(), not accept(): closes the picker without falling into the
        # "Open" path's chat-surface logic (which would otherwise open
        # llama.cpp/Open WebUI as an unwanted side effect of Install).
        if source_dialog is not None:
            source_dialog.reject()
        self._model_status.setText(f"installing {key}...")
        self._gc.request_install_harness(key)

    def _show_openwebui_start_offer(self, reason):
        """Offer the controller's optional rich-chat service start without blocking."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Start Open WebUI?")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        message = QtWidgets.QLabel(reason or "Open WebUI is installed but not running.")
        message.setWordWrap(True)
        layout.addWidget(message)
        detail = QtWidgets.QLabel(
            "First launch sets up its database and can take up to 5 minutes."
        )
        detail.setObjectName("muted")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        # M18.11 (owner request): every road to chat also shows the coding
        # harnesses. This offer used to be a dead end - Start or Cancel - even
        # though the model is running and a terminal harness needs nothing else.
        # Same honesty rules as the chooser: missing-from-PATH shows disabled
        # with the install command in the tooltip.
        code_label = QtWidgets.QLabel("Or code in a terminal instead:")
        code_label.setObjectName("muted")
        layout.addWidget(code_label)
        harness_row = QtWidgets.QHBoxLayout()
        chosen: dict[str, str] = {}
        try:
            installed = self._gc.detect_harnesses()
        except Exception:  # noqa: BLE001 - detection failure never breaks the offer
            installed = {}
        for key, label in (
            ("claude", "Claude Code"),
            ("codex", "Codex"),
            ("opencode", "OpenCode"),
        ):
            button = QtWidgets.QPushButton(label)
            found = (installed or {}).get(key)
            button.setEnabled(bool(found))
            button.setToolTip(
                f"found at {found}" if found else
                "Not found on PATH. Install with: "
                + HARNESS_INSTALL_SUMMARIES.get(key, "")
            )

            def pick(_checked=False, k=key):
                chosen["harness"] = k
                dialog.reject()  # close WITHOUT starting Open WebUI

            button.clicked.connect(pick)
            harness_row.addWidget(button)
        harness_row.addStretch(1)
        layout.addLayout(harness_row)

        remember = QtWidgets.QCheckBox("Remember my choice")
        layout.addWidget(remember)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(
            "Start Open WebUI"
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if harden_text_rendering(dialog).exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._model_status.setText("starting Open WebUI; first launch can take up to 5 minutes...")
            self._gc.start_openwebui_and_open(remember.isChecked())
        elif chosen.get("harness"):
            self._launch_harness_with_folder(chosen["harness"], False)

    def _on_assistant_started(self, payload):
        """Unlock session controls only after the controller reports readiness."""
        self._assistant_live = True
        self._assistant_port = payload.get("port")
        self._apply_running_snapshot(payload)
        self._assistant_status.setText("Ready to listen")
        self._assistant_end_btn.setEnabled(True)
        self._interrupt_btn.setEnabled(True)
        self._set_talk_state("idle")
        self._refresh_model_rows()
        self._refresh_buttons()
        if self._listen_when_ready:
            self._listen_when_ready = False
            self._gc.request_talk()

    def _on_assistant_error(self, message):
        """Return the Talk page to a truthful stopped state after any session error."""
        self._assistant_live = False
        self._assistant_status.setText(message)
        self._listen_when_ready = False
        self._assistant_start_btn.setEnabled(True)
        self._assistant_end_btn.setEnabled(False)
        self._interrupt_btn.setEnabled(False)
        self._talk_btn.setEnabled(False)
        # The assistant temporarily disables model lifecycle controls. Recompute
        # the full button set here so Start/Switch and Stop recover after failure.
        self._refresh_buttons()

    def _on_assistant_ended(self):
        """Reset every assistant control after a clean end result."""
        self._assistant_live = False
        self._assistant_status.setText("Not running. Click Talk to start.")
        self._assistant_start_btn.setEnabled(True)
        self._assistant_end_btn.setEnabled(False)
        self._interrupt_btn.setEnabled(False)
        self._listen_when_ready = False
        self._talk_btn.setText("Talk")
        self._talk_btn.setEnabled(False)
        # Ending a session releases its model lock. The complete refresh restores
        # every lifecycle control from the last controller-owned snapshot.
        self._refresh_buttons()

    def _set_talk_state(self, state):
        """Map the existing assistant states to accessible text and enablement."""
        previous = self._talk_state
        self._talk_state = state or "idle"
        labels = {
            "idle": "Talk",
            "listening": "Listening... speak now",
            "thinking": "Thinking...",
            "speaking": "Stop and listen",
        }
        status = {
            "idle": "Ready to listen",
            "listening": "Listening... speak now",
            "thinking": "Thinking...",
            "speaking": "Speaking... click Talk to cut in, or talk over it on the phone",
        }
        self._talk_btn.setText(labels.get(self._talk_state, "Talk"))
        if self._assistant_live:
            self._assistant_status.setText(status.get(self._talk_state, "Ready to listen"))
        self._talk_btn.setEnabled(
            self._assistant_live and self._talk_state in ("idle", "thinking", "speaking")
        )
        self._talk_btn.setProperty("talkState", self._talk_state)
        self._talk_pulse.setProperty("talkState", self._talk_state)
        self._talk_btn.style().unpolish(self._talk_btn)
        self._talk_btn.style().polish(self._talk_btn)
        self._talk_pulse.style().unpolish(self._talk_pulse)
        self._talk_pulse.style().polish(self._talk_pulse)
        # After a reply, listen again so the next turn does not need a tap.
        if (
            self._assistant_live
            and self._talk_state == "idle"
            and previous in ("thinking", "speaking")
        ):
            self._gc.request_talk()

    def _append_conversation(self, prefix, text):
        """Append one completed turn and keep the newest line visible."""
        # Model replies are third-party text: appended literally, never sniffed.
        append_plain(self._conversation, f"{prefix}{text}")
        self._conversation.ensureCursorVisible()

    def _render_memory_hits(self, result):
        """Render real local hits or a query-specific honest empty state."""
        if not result.ok:
            self._memory_result.setPlainText(result.error or "memory search failed")
            return
        hits = result.payload.get("hits", [])
        if not hits:
            query = result.payload.get("query", "")
            target = f"'{query}'" if query else "recent conversations"
            self._memory_result.setPlainText(
                f"no stored conversation matched {target}."
            )
            return
        self._memory_result.setPlainText(
            "\n".join(f"{hit.get('role', '?')}: {hit.get('text', '')}" for hit in hits)
        )

    def _apply_benchmark_result(self, result):
        """Refresh the selected row's measured generation throughput."""
        self._refresh_model_exclusive_buttons()
        model_id = result.payload.get("model_id", "")
        if not result.ok:
            self._model_status.setText(result.error or "benchmark failed")
            return
        # A discovered model has no models.yaml row, so its score is real but
        # session-only. UX Spec 3.4 requires the Score CELL itself to carry the
        # note - a status line alone scrolls away and leaves a normal-looking
        # score on a row that will silently lose it on the next rescan.
        note = result.payload.get("note", "")
        persisted = result.payload.get("score_persisted", True)
        score_display = result.payload.get("score_display", "-")
        cell_text = score_display
        if not persisted and note:
            cell_text = f"{score_display} - {note}"
        for model in self._models:
            if model["id"] == model_id:
                model["benchmark_tok_s"] = result.payload.get("score")
                model["score_display"] = cell_text
                # Consumed by _refresh_model_rows to tooltip the cell; keeps the
                # honesty attached to the row even when the column is narrow.
                model["score_note"] = "" if persisted else note
                break
        self._refresh_model_rows()
        detail = result.payload.get("detail", "")
        self._model_status.setText(
            f"benchmarked {model_id}: {detail}"
            + (f" {note}" if note and note not in detail else "")
        )

    def start_controller(self):
        """Start the existing controller workers immediately before the event loop."""
        self._gc.start_threads()

    def closeEvent(self, event):
        """Route window close through the controller's idempotent full shutdown.

        M13: when a training run looks live at this moment, the owner is shown a
        blocking, acknowledgeable dialog BEFORE the window finishes closing. A
        status chip would vanish with the window, and LOCITIZE must never let a close
        read as a clean, complete stop when it cannot stop a container it did not
        start.
        """
        if not self._closed:
            if getattr(self._gc, "_local_session_launches", 0):
                answer = plain_message_box(
                    self, "Close locitize?",
                    "Coding terminals launched with locitize may still use its model. Closing stops the model server. Keep locitize open to continue working.",
                    icon=QtWidgets.QMessageBox.Icon.Warning,
                    buttons=QtWidgets.QMessageBox.StandardButton.Close | QtWidgets.QMessageBox.StandardButton.Cancel,
                    default=QtWidgets.QMessageBox.StandardButton.Cancel,
                ).exec()
                if answer != QtWidgets.QMessageBox.StandardButton.Close:
                    event.ignore()
                    return
            self._warn_if_run_active()
            self._closed = True
            # The window goes the moment X is clicked; the cleanup below (stop
            # the model, Open WebUI and voice services) finishes behind it.
            self.hide()
            QtWidgets.QApplication.processEvents()
            self._timer.stop()
            self._sysmon_stop.set()  # M17.18: stop the RAM/VRAM sampler thread
            self._gc.shutdown()
        event.accept()

    def show_data_migration_notice(self):
        """Show the one-time "your data was copied" notice, if one is owed.

        DEC-M14-9 item 7. The wording and the "has it been shown yet" flag both
        live in migration.py beside the copy itself, because the sentence is a
        claim about what that code did: if the two ever drifted, LOCITIZE would be
        telling the user something that was not true. Returns the text shown (or
        an empty string), so the behaviour is testable without a screenshot.

        Never raises: a missing settings object or an unreadable marker means no
        notice, not a failed start.
        """
        import migration

        settings = getattr(self._gc, "_settings", None)
        data_dir = getattr(settings, "data_dir", None)
        if data_dir is None:
            return ""
        try:
            text = migration.pending_notice(data_dir)
        except Exception:  # noqa: BLE001 - a notice must never block startup
            return ""
        if not text:
            return ""
        # Plain text, like every other externally-sourced string in this view:
        # the message embeds two filesystem paths, which must never be parsed
        # as markup.
        plain_message_box(
            self,
            "Your locitize data moved to one folder",
            text,
            icon=QtWidgets.QMessageBox.Icon.Information,
            buttons=QtWidgets.QMessageBox.StandardButton.Ok,
            default=QtWidgets.QMessageBox.StandardButton.Ok,
        ).exec()
        # Marked shown only AFTER the dialog closed: a crash while it is open
        # means the user has not read it, so it is still owed next start.
        migration.mark_notice_shown(data_dir)
        return text

    def show_harness_onboarding_if_needed(self):
        """Offer to install any missing coding harness at startup.

        Owner request 2026-08-21: "during onboarding, it should scan. And if
        it doesn't find the configuration of any of those harnesses, it
        should suggest that the user select, and then it will go ahead and
        download, configure. And that way the user can just start using it."
        Shown every launch while any of the three is missing (not a
        one-time-ever notice like show_data_migration_notice - there is
        nothing to "mark shown" here, since the thing being reported, a
        missing CLI, is still true until the owner installs it or dismisses
        this particular prompt). Declining just closes the dialog; the same
        offer is always still reachable later via the Chat picker's own
        "Install" button next to a harness not found on PATH (see
        _show_chat_chooser), so skipping this once is never a dead end.

        Never raises: a detection failure means no offer, not a failed start.
        """
        try:
            installed = self._gc.detect_harnesses()
        except Exception:  # noqa: BLE001 - onboarding must never block startup
            return
        missing = [key for key, path in installed.items() if not path]
        if not missing:
            return

        labels = {"claude": "Claude Code", "codex": "Codex", "opencode": "OpenCode"}
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Set up coding harnesses?")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(
            QtWidgets.QLabel(
                "locitize can launch your local models inside these coding tools.\n"
                "The ones below aren't installed yet. Nothing is installed until\n"
                "you check a box AND click OK - each one runs the exact command\n"
                "shown beneath it:"
            )
        )
        checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        for key in missing:
            box = QtWidgets.QCheckBox(labels.get(key, key))
            box.setChecked(False)  # opt-in, never pre-selected (owner must choose)
            layout.addWidget(box)
            summary = QtWidgets.QLabel(HARNESS_INSTALL_SUMMARIES.get(key, ""))
            summary.setWordWrap(True)
            summary.setObjectName("statusChip")
            layout.addWidget(summary)
            checkboxes[key] = box
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if harden_text_rendering(dialog).exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        selected = [key for key, box in checkboxes.items() if box.isChecked()]
        for key in selected:
            self._model_status.setText(f"installing {key}...")
            self._gc.request_install_harness(key)

    def _warn_if_run_active(self):
        """Show the limitation notice if (and only if) a run is live right now.

        finetune_run_active() is the controller's cached answer, refreshed by the
        background monitor loop: close must never stat a possibly-dead drive on
        the Qt thread, so it reads a value that is at most one monitor interval
        old rather than re-walking the outputs tree here.
        """
        checker = getattr(self._gc, "finetune_run_active", None)
        if checker is None:
            return
        try:
            if not checker():
                return
            message = self._gc.finetune_warning_text()
        except Exception:  # noqa: BLE001 - a warning check must never block close
            return
        # The warning text names run folders read from disk, so it is plain text
        # like every other externally-sourced string in this view.
        plain_message_box(
            self,
            "Training may still be running",
            message,
            icon=QtWidgets.QMessageBox.Icon.Warning,
            buttons=QtWidgets.QMessageBox.StandardButton.Ok,
            default=QtWidgets.QMessageBox.StandardButton.Ok,
        ).exec()


# Both names are documented runtime types; one implementation keeps the view small.
LocitizeDesktop = MainWindow


def _raise_existing_window():
    """Best-effort: bring an already-running LOCITIZE Desktop window to the front.

    Owner request 2026-08-21 (multiple flashing windows on a repeated
    double-click): a launch with no visible feedback for a second or two while
    the app bootstraps invites an impatient second click, which used to spawn
    a whole SECOND full instance - each one popping onto the screen read as
    "the screen flashes several times." Windows-only (ctypes), and silently a
    no-op if the window cannot be found or the OS refuses the focus request
    (Windows' focus-stealing prevention sometimes blocks a background
    process's SetForegroundWindow call) - never worth crashing the refusal to
    launch a duplicate over.
    """
    try:
        import ctypes

        hwnd = ctypes.windll.user32.FindWindowW(None, "locitize Desktop")
        if hwnd:
            # SW_RESTORE unconditionally would un-maximize a window that was
            # already maximized (confirmed live: it dropped a maximized
            # window back to its small default geometry). Only restore when
            # the window is actually minimized (IsIconic); otherwise leave
            # its current maximized/normal state alone and just raise it.
            if ctypes.windll.user32.IsIconic(hwnd):
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.FlashWindow(hwnd, True)
    except Exception:  # noqa: BLE001 - best-effort only, never fatal
        pass


def run(controller, health="unknown"):
    """Build and run the native desktop, returning its event-loop exit code."""
    _log_launch("run() start")
    # Owner request 2026-08-21: a lock file (not a port/process check) is the
    # standard single-instance guard - held for the whole process lifetime,
    # released automatically on a crash (unlike a PID file, which can go
    # stale). A second launch while one is already running raises the
    # existing window instead of building a second full instance (duplicate
    # model registry load, duplicate llama-server start attempt, and the
    # flashing this whole guard exists to prevent).
    lock = QtCore.QLockFile(str(Path(tempfile.gettempdir()) / "locitize_desktop.lock"))
    lock.setStaleLockTime(30_000)
    if not lock.tryLock(100):
        _log_launch("run() refused - another instance already holds the lock")
        _raise_existing_window()
        return 0
    _log_launch("lock acquired")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("locitize Desktop")
    app.setStyleSheet(APP_STYLE)
    _log_launch("QApplication ready, stylesheet applied")
    # Owner request 2026-08-21: the real mark for the title bar and taskbar,
    # not Qt's generic default window icon. Set on the app (not just the
    # window) so it also covers the taskbar/Alt-Tab thumbnail.
    #
    # M18.3 (owner report: taskbar still showed the Python icon): setWindowIcon
    # alone is not enough on Windows. The taskbar groups windows by the
    # process's AppUserModelID, and a pythonw.exe-hosted app inherits PYTHON's
    # AUMID - so the taskbar shows python's icon and pins group under Python no
    # matter what the window declares. Claiming an explicit AUMID of our own,
    # BEFORE the first window shows, makes Windows treat LOCITIZE as its own
    # application: our icon on the taskbar, our identity when pinned.
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "Locitize.LOCITIZE.Desktop"
            )
        except (OSError, AttributeError):
            pass  # icon degrades to the host interpreter's; never block launch
    if LOGO_ICO_PATH.is_file():
        app_icon = QtGui.QIcon(str(LOGO_ICO_PATH))
        app.setWindowIcon(app_icon)
    window = MainWindow(controller, health)
    if LOGO_ICO_PATH.is_file():
        window.setWindowIcon(app_icon)
    window.start_controller()
    _log_launch("start_controller() done")
    # Owner request 2026-08-21 (black-flash-on-launch diagnostic): calling
    # showMaximized() directly on a window that has never been shown asks
    # Windows' compositor to allocate and display a FULL-SCREEN surface
    # before Qt has painted a single frame into it - a documented Qt-on-
    # Windows cause of a black flash, and much more visible full-screen than
    # the small fixed-size window this app used before the M14 "launch
    # maximized" change. show() first lets the real first paint land on a
    # small, fast-to-composite surface; showMaximized() is then deferred one
    # event-loop tick (QTimer.singleShot(0, ...)) so it runs AFTER that first
    # frame already exists, giving DWM real pixels to scale up instead of an
    # empty surface to fill with black.
    window.show()
    QtCore.QTimer.singleShot(0, window.showMaximized)
    _log_launch("window.show() called; showMaximized() queued for next tick")
    # DEC-M14-9 item 7: if this start copied the user's data out of the install
    # tree, say so once, in the app, naming both real folders. Shown after the
    # window exists so the notice has a parent and cannot appear behind it.
    window.show_data_migration_notice()
    # Owner request 2026-08-21: zero-friction harness onboarding, shown after
    # the data-migration notice so first-run messages appear in a stable,
    # predictable order.
    window.show_harness_onboarding_if_needed()
    _log_launch("entering app.exec()")
    try:
        return app.exec()
    finally:
        # Release explicitly rather than waiting for process exit / GC, so a
        # restart from within the same interpreter (tests, --terminal
        # handing off to --desktop) never sees a stale lock from this run.
        _log_launch("app.exec() returned, releasing lock")
        lock.unlock()
