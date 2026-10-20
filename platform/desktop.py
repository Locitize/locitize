"""PySide6 presentation shell for the LOCITIZE desktop command center.

This module is the M12 view layer. It builds a dark, sidebar-driven native
window and binds every control to the existing gui_controller intents. The
controller owns all work, queues, validation, and lifecycle behavior; this file
only renders its Result objects on the Qt GUI thread through one QTimer drain.
"""

import html
import os
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
THINKING_AS_REGISTERED = "Model's default"

# Plain thinking choices: (label, value handed to parse_reasoning_choice).
# An empty value means "send no override at all".
THINKING_CHOICES = (
    (THINKING_AS_REGISTERED, ""),
    ("Off", "off"),
    ("Light", "low"),
    ("Medium", "medium"),
    ("Deep", "xhigh"),
)

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

HUB_DOWNLOAD_REASON = (
    "Pick a file from the list first, and wait for any download in progress to finish."
)

# Fine-tuning runs natively now (no Docker), so its page is a normal sidebar
# page. LOCITIZE_SHOW_FINETUNE=0 hides it.
SHOW_FINETUNE_TAB = os.environ.get("LOCITIZE_SHOW_FINETUNE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)


def build_page_names(show_finetune=True):
    """The sidebar pages, in order. Fine-tune follows Voice & Vision."""
    names = ["Start", "Models", "Code", "Voice & Vision"]
    if show_finetune:
        names.append("Fine-tune")
    names.append("Settings")
    return tuple(names)


PAGE_NAMES = build_page_names(SHOW_FINETUNE_TAB)

# One line for each page, shown when you hover its name in the sidebar.
PAGE_TOOLTIPS = {
    "Start": "Pick a model, start it, and open the chat",
    "Models": "See your models, download new ones, and tune them",
    "Code": "Use your model with coding tools, resume sessions, search old chats",
    "Voice & Vision": "Talk out loud, ask about pictures, watch your screen",
    "Fine-tune": "Train a model on your own data",
    "Settings": "Features, coding tools, phone access and about",
}


class _HideWhenEmptyLabel(QtWidgets.QLabel):
    """A status chip that takes no room while it has nothing to say."""

    def __init__(self, text=""):
        super().__init__(text)
        self.setVisible(bool(text))

    def setText(self, text):
        super().setText(text)
        self.setVisible(bool(text))


class _SharedLabel(QtWidgets.QLabel):
    """A status line that other pages can mirror.

    Every page that can start an action shows the same outcome message, so a
    failure is never hidden on a page the person is not looking at. A widget can
    only live in one place, so each extra page asks for a mirror() copy and this
    label keeps all of them in step.
    """

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._mirrors = []

    def mirror(self):
        label = QtWidgets.QLabel(self.text())
        label.setObjectName(self.objectName() or "muted")
        label.setWordWrap(True)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Minimum
        )
        label.setToolTip(self.toolTip())
        self._mirrors.append(label)
        return label

    def _each_mirror(self, action):
        for label in list(self._mirrors):
            try:
                action(label)
            except RuntimeError:  # the mirror's C++ widget is already gone
                self._mirrors.remove(label)

    def setText(self, text):  # noqa: N802 - Qt naming
        super().setText(text)
        self._each_mirror(lambda label: label.setText(text))

    def setToolTip(self, text):  # noqa: N802 - Qt naming
        super().setToolTip(text)
        self._each_mirror(lambda label: label.setToolTip(text))


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
/* Plain grouping widgets inside a panel (the Fine-tune steps) take the panel's colour. */
QWidget#plainGroup { background: transparent; }
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
/* Owner request 2026-10-20: with no rule of its own a right-click menu was
   drawn by Windows in the system menu font, unlike the rest of the app. A
   full rule makes Qt draw it with the app's face, size and colours. */
QMenu {
    background: #2b2d30;
    color: #e8eaed;
    border: 1px solid #4c5364;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item { padding: 6px 18px; border-radius: 4px; background: transparent; }
QMenu::item:selected { background: #284b7a; color: #ffffff; }
QMenu::item:disabled { color: #6f7379; }
QMenu::separator { height: 1px; background: #35383d; margin: 4px 6px; }
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
/* Sub-tabs inside a page (Models, Code, Voice & Vision). */
QTabWidget#subTabs::pane { border: 0; top: -1px; }
QTabWidget#subTabs > QTabBar::tab {
    background: transparent;
    color: #a7aeb8;
    padding: 9px 18px;
    margin-right: 4px;
    border: 0;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}
QTabWidget#subTabs > QTabBar::tab:hover { color: #ffffff; }
QTabWidget#subTabs > QTabBar::tab:selected {
    color: #ffffff;
    border-bottom: 2px solid #4f8cff;
}
/* The "what is needed" banner on Voice & Vision. */
QFrame#banner {
    background: #3a3322;
    border: 1px solid #6b5a2c;
    border-radius: 8px;
}
QFrame#banner QLabel { color: #ffe3a1; }
QPushButton#healthChip { padding: 6px 14px; }
QPushButton#healthChip[healthState="PASS"] { color: #6ee7a8; }
QPushButton#healthChip[healthState="WARNING"] { color: #ffd166; }
QPushButton#healthChip[healthState="FAIL"] { color: #ff7b72; }
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


def _model_file_present(model):
    """True when a model row's file is on disk, so Auto-tune can load it.

    A row left behind after its .gguf was moved or deleted used to offer
    Auto-tune, which then failed with "could not read the model file". The
    controller checks the disk when it lists the models; a row without the
    flag is treated as present.
    """
    return bool(model.get("file_present", True))


def _model_turned_off(model):
    """True for a row its owner switched off (status disabled or future).

    The controller refuses to start or tune such a row ("not installed"), so
    offering Auto-tune for it only produced a confusing failure. A row without
    the field is treated as installed.
    """
    return model.get("status", "installed") != "installed"


def _model_tunable(model):
    """True when Auto-tune can run this row: on the list, switched on, file on disk."""
    return (
        model.get("source", "registry") == "registry"
        and not _model_turned_off(model)
        and _model_file_present(model)
    )


def plain_speed(score_display):
    """Speed cell, kept short as "31.5 tok/s" (owner request 2026-10-20)."""
    text = str(score_display or "-").strip()
    if text in ("", "-"):
        return "Not measured yet"
    return text


def plain_total(text):
    """The model-list footer in everyday words."""
    return str(text or "").replace("Total:", "Disk space used:", 1)


def plain_monitor_line(line):
    """Rewrite the controller's compact monitor line without abbreviations."""
    words = []
    for part in str(line or "").split(" | "):
        part = part.strip()
        if part.startswith("gen ") and part.endswith(" tok/s"):
            words.append("Writing " + part[4:-6] + " tokens per second")
        elif part.startswith("prompt ") and part.endswith(" tok/s"):
            words.append("Reading " + part[7:-6] + " tokens per second")
        elif part.startswith("prompt ") and part.endswith(" tok"):
            words.append("Read " + part[7:-4] + " tokens")
        elif part.startswith("gen ") and part.endswith(" tok"):
            words.append("Wrote " + part[4:-4] + " tokens")
        elif part.startswith("ctx "):
            words.append("Working memory " + part[4:])
        else:
            words.append(part)
    return "  |  ".join(words)


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

    def __init__(self, controller, health="unknown", health_report=None):
        super().__init__()
        self._gc = controller
        self._health = health
        self._health_report = health_report
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
        # touch the filesystem (the trainer folder and the outputs tree, either of
        # which may sit on a disconnected drive), so both go through the queue and
        # arrive as Results. The page already shows its neutral ready state from
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
        builders = {
            "Start": self._build_start_page,
            "Models": self._build_models_page,
            "Code": self._build_code_page,
            "Voice & Vision": self._build_voice_vision_page,
            "Fine-tune": self._build_finetune_page,
            "Settings": self._build_settings_page,
        }
        # The Fine-tune widgets are always built because the controller's
        # fine-tune results update them by name; the page is only reachable from
        # the sidebar when the flag is on (see SHOW_FINETUNE_TAB).
        for name in builders:
            if name not in PAGE_NAMES and name != "Fine-tune":
                continue
            page = builders[name]()
            page.setObjectName(f"page{name}")
            if name in PAGE_NAMES:
                self.pages[name] = page
                self._stack.addWidget(page)
            else:
                page.setParent(self._stack)
                page.hide()
                self._hidden_finetune_page = page
        content.addWidget(self._stack, 1)
        root.addLayout(content, 1)
        self.setCentralWidget(shell)

        self._sidebar.currentRowChanged.connect(self._on_sidebar_changed)
        self._sidebar.setCurrentRow(PAGE_NAMES.index("Start"))

    def _build_sessions_page(self):
        from sessions_ui import SessionsPage

        self._sessions_page = SessionsPage(self._gc)
        return self._sessions_page

    def _show_release_diagnostics(self):
        from release_info import VERSION, diagnostics

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("About locitize " + VERSION)
        layout = QtWidgets.QVBoxLayout(dialog)
        intro = QtWidgets.QLabel(
            "This is the technical detail behind the app. If something is not "
            "working, copy it and send it along with your question."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(diagnostics())
        layout.addWidget(text)
        copy = QtWidgets.QPushButton("Copy to clipboard")
        copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(text.toPlainText()))
        layout.addWidget(copy)
        dialog.resize(600, 400)
        dialog.exec()

    def _build_header(self):
        """Build the always-visible brand, hardware, health and model status line."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("header")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(10)
        # Owner decision 2026-08-29: the LOGO IMAGE stays (a logo is brand
        # identity, not a UI font). Degrades honestly if the asset is missing.
        if HEADER_LOGO_PATH.is_file():
            logo_pixmap = QtGui.QPixmap(str(HEADER_LOGO_PATH)).scaledToHeight(
                32, QtCore.Qt.TransformationMode.SmoothTransformation
            )
            logo_label = QtWidgets.QLabel()
            logo_label.setPixmap(logo_pixmap)
            layout.addWidget(logo_label)
        layout.addStretch(1)
        # The chips may shrink (and clip) when the window is narrow, but the
        # button at the end never does: a widening status line used to push
        # "Offload GPU" off the right edge (audit bug 5).
        specs_text = gui_controller.format_system_specs(self._gc.system_specs())
        self._specs_label = QtWidgets.QLabel(specs_text)
        self._specs_label.setObjectName("statusChip")
        self._specs_label.setMinimumWidth(40)
        self._specs_label.setToolTip("Your computer: memory and graphics card")
        layout.addWidget(self._specs_label)
        # Health: a button so the reasons are one click away.
        self._health_btn = QtWidgets.QPushButton()
        self._health_btn.setObjectName("healthChip")
        self._health_btn.setMinimumWidth(40)
        self._health_btn.clicked.connect(self._show_health_details)
        layout.addWidget(self._health_btn)
        self._header_status = QtWidgets.QLabel()
        self._header_status.setObjectName("statusChip")
        self._header_status.setMinimumWidth(40)
        self._header_status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._header_status)
        # Global quick-unload: frees GPU memory from any page. Same stop path as
        # the Stop button (request_free_gpu always targets what is running).
        self._header_offload_btn = QtWidgets.QPushButton("Free up GPU")
        self._header_offload_btn.setObjectName("headerOffloadButton")
        self._header_offload_btn.setToolTip(
            "Free the graphics-card memory locitize is using: stop the running "
            "model and clear any leftover locitize helpers. Other apps are "
            "listed in the result but never touched."
        )
        self._header_offload_btn.clicked.connect(self._on_offload_gpu)
        self._header_offload_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self._header_offload_btn.setMinimumWidth(
            self._header_offload_btn.sizeHint().width() + 10
        )
        layout.addWidget(self._header_offload_btn)
        self._update_health_chip()
        return frame

    def _build_sidebar(self):
        """Build the compact navigation rail; selection drives the page stack."""
        frame = QtWidgets.QFrame()
        frame.setObjectName("sidebar")
        frame.setFixedWidth(176)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(8, 12, 8, 12)
        self._sidebar = QtWidgets.QListWidget()
        self._sidebar.setObjectName("sidebarList")
        self._sidebar.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        for name in PAGE_NAMES:
            self._sidebar.addItem(name)
            self._sidebar.item(self._sidebar.count() - 1).setToolTip(
                PAGE_TOOLTIPS.get(name, "")
            )
        layout.addWidget(self._sidebar, 1)
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

    @staticmethod
    def _tab_body(hint=""):
        """A sub-tab's content column: an optional muted hint, then the body."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)
        if hint:
            label = QtWidgets.QLabel(hint)
            label.setObjectName("muted")
            label.setWordWrap(True)
            layout.addWidget(label)
        return tab, layout

    @staticmethod
    def _section(layout, title, hint=""):
        """Add a bold section heading (and optional muted hint) to `layout`."""
        label = QtWidgets.QLabel(title)
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        if hint:
            note = QtWidgets.QLabel(hint)
            note.setObjectName("muted")
            note.setWordWrap(True)
            layout.addWidget(note)

    @staticmethod
    def _gate(widget, enabled, reason=""):
        """Enable or disable a control and say WHY on hover when it is disabled.

        A greyed-out button with no explanation reads as broken. The control's
        normal tooltip is remembered on first use and restored when it re-enables.
        """
        base = widget.property("baseTip")
        if base is None:
            base = widget.toolTip()
            widget.setProperty("baseTip", base)
        widget.setEnabled(bool(enabled))
        widget.setToolTip(base if enabled else (reason or base))

    def _go_to(self, name, tab=None):
        """Switch to a page (and optionally one of its sub-tabs) by name."""
        if name in PAGE_NAMES:
            self._sidebar.setCurrentRow(PAGE_NAMES.index(name))
        if tab is not None:
            widget = getattr(self, {
                "Models": "_models_tabs",
                "Code": "_code_tabs",
                "Voice & Vision": "_vv_tabs",
            }.get(name, "_unused"), None)
            if widget is not None:
                for index in range(widget.count()):
                    if widget.tabText(index) == tab:
                        widget.setCurrentIndex(index)
                        break

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
        """Models: a simple list, a download panel, and an Advanced tab.

        Three sub-tabs keep every screenful short enough for a 1366x768 window
        without anything clipping: "My models" (what you have), "Get models"
        (find and download more), and "Advanced" (speed test, thinking level,
        add/delete/refresh, and the per-model editor).
        """
        page, layout = self._page(
            "Models",
            "The AI models on this computer. Pick one on the Start page to run it.",
        )
        self._models_tabs = QtWidgets.QTabWidget()
        self._models_tabs.setObjectName("subTabs")
        self._models_tabs.setDocumentMode(True)
        self._models_tabs.tabBar().setDrawBase(False)

        # ---- My models -------------------------------------------------
        mine, mine_layout = self._tab_body()
        panel, body = self._panel()
        self._model_table = QtWidgets.QTableWidget(0, 10)
        self._model_table.setHorizontalHeaderLabels(
            [
                "Model",
                "Capabilities",
                "Source",
                "Size",
                "Fits your GPU?",
                "On GPU",
                "On CPU",
                "Speed",
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
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        for column, width in {
            0: 260,
            1: 170,
            2: 110,
            3: 90,
            4: 135,
            5: 90,
            6: 90,
            7: 120,
            8: 120,
        }.items():
            header.resizeSection(column, width)
        self._model_table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        header.sectionClicked.connect(self._on_sort)
        self._model_table.currentCellChanged.connect(self._on_select_model)
        self._model_table.setMinimumHeight(MODEL_TABLE_MIN_HEIGHT)
        self._attach_empty_state(
            self._model_table,
            "No models yet.\nOpen the Get models tab to download one.",
        )
        body.addWidget(self._model_table, 1)
        footer = QtWidgets.QHBoxLayout()
        self._models_total = QtWidgets.QLabel("Total: -")
        self._models_total.setObjectName("statusChip")
        footer.addWidget(self._models_total)
        self._use_model_btn = QtWidgets.QPushButton("Use this model")
        self._use_model_btn.setObjectName("primaryButton")
        self._use_model_btn.setToolTip("Go to the Start page with this model selected")
        self._use_model_btn.clicked.connect(lambda: self._go_to("Start"))
        footer.addWidget(self._use_model_btn)
        self._mine_add_file_btn = QtWidgets.QPushButton("Add model file...")
        self._mine_add_file_btn.setToolTip(
            "Add a GGUF already on this computer. This does not use the network."
        )
        self._mine_add_file_btn.clicked.connect(self._on_add_model_file)
        footer.addWidget(self._mine_add_file_btn)
        self._mine_register_btn = QtWidgets.QPushButton("Add discovered model")
        self._mine_register_btn.setToolTip(
            "Add the selected GGUF from the Models folder to your model list"
        )
        self._mine_register_btn.clicked.connect(self._on_register_selected)
        self._mine_register_btn.setVisible(False)
        footer.addWidget(self._mine_register_btn)
        # The same permanent delete as Advanced > Manage this model, here where
        # the list is: the old desktop kept Delete beside the table, and the
        # owner could not find it at the bottom of Advanced.
        self._mine_delete_btn = QtWidgets.QPushButton("Delete...")
        self._mine_delete_btn.setObjectName("dangerButton")
        self._mine_delete_btn.setToolTip(
            "Permanently delete the selected model's file(s) from disk and "
            "remove it from the list (asks first)"
        )
        self._mine_delete_btn.clicked.connect(self._on_delete_model)
        footer.addWidget(self._mine_delete_btn)
        footer.addStretch(1)
        body.addLayout(footer)
        self._model_table.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._model_table.customContextMenuRequested.connect(
            self._on_model_table_menu
        )
        self._models_note = self._model_status.mirror()
        body.addWidget(self._models_note)
        mine_layout.addWidget(panel, 1)
        self._models_tabs.addTab(mine, "My models")

        # ---- Get models -------------------------------------------------
        get_tab, get_layout = self._tab_body()
        get_body = self._build_get_models_panel()
        get_layout.addWidget(self._scrollable(get_body), 1)
        self._models_tabs.addTab(get_tab, "Get models")

        # ---- Advanced ---------------------------------------------------
        self._models_tabs.addTab(self._build_models_advanced(), "Advanced")

        layout.addWidget(self._models_tabs, 1)
        return page

    def _build_models_advanced(self):
        """The Advanced tab: one model picker drives every section below it."""
        tab, tab_layout = self._tab_body()
        scroll_body = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(scroll_body)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)

        # -- Picker and summary -------------------------------------------
        top, top_body = self._panel()
        top_body.setContentsMargins(14, 10, 14, 10)
        top_body.setSpacing(6)
        pick_row = QtWidgets.QHBoxLayout()
        pick_label = QtWidgets.QLabel("Model:")
        pick_label.setObjectName("sectionTitle")
        pick_row.addWidget(pick_label)
        # The shared picker. It mirrors the My models selection both ways and
        # every control on this tab acts on the model it shows.
        self._settings_model = QtWidgets.QComboBox()
        self._settings_model.setToolTip(
            "Everything on this tab applies to this model. It follows the "
            "selection on the My models tab."
        )
        for model in self._models:
            self._settings_model.addItem(model["name"], model["id"])
        self._settings_model.currentIndexChanged.connect(self._on_settings_model_changed)
        pick_row.addWidget(self._settings_model, 1)
        self._refresh_models_btn = QtWidgets.QPushButton("Refresh list")
        self._refresh_models_btn.setToolTip(
            "Reload your model list so new downloads appear"
        )
        self._refresh_models_btn.clicked.connect(self._on_refresh_models)
        pick_row.addWidget(self._refresh_models_btn)
        top_body.addLayout(pick_row)
        self._adv_summary = QtWidgets.QLabel("")
        self._adv_summary.setObjectName("muted")
        self._adv_summary.setWordWrap(True)
        top_body.addWidget(self._adv_summary)
        column.addWidget(top)

        # -- Auto-tune ------------------------------------------------------
        speed, speed_body = self._panel()
        speed_body.setContentsMargins(14, 10, 14, 10)
        speed_body.setSpacing(6)
        self._section(
            speed_body,
            "Find the best settings",
            "Tries this model at several working-memory sizes, keeps the "
            "biggest your computer can hold, and measures its speed. Takes a "
            "few minutes and uses the GPU.",
        )
        speed_row = QtWidgets.QHBoxLayout()
        self._autotune_btn = QtWidgets.QPushButton("Auto-tune this model")
        self._autotune_btn.setToolTip(
            "Find the biggest working memory that really loads on this computer "
            "for the model above, then measure its speed in tokens per "
            "second. Takes a few minutes and starts the model several times."
        )
        self._autotune_btn.clicked.connect(self._on_autotune_context)
        self._autotune_all_btn = QtWidgets.QPushButton("Auto-tune all")
        self._autotune_all_btn.setToolTip(
            "Auto-tune every model on your list, one after another. Takes a few "
            "minutes per model; Stop cancels the rest."
        )
        self._autotune_all_btn.clicked.connect(self._on_autotune_all)
        speed_row.addWidget(self._autotune_btn)
        speed_row.addWidget(self._autotune_all_btn)
        speed_row.addStretch(1)
        speed_body.addLayout(speed_row)
        # Why the buttons are grey, on the page rather than only in a tooltip: a
        # disabled Auto-tune with a model running read as a broken button.
        self._autotune_hint = QtWidgets.QLabel("")
        self._autotune_hint.setObjectName("muted")
        self._autotune_hint.setWordWrap(True)
        self._autotune_hint.setVisible(False)
        speed_body.addWidget(self._autotune_hint)
        self._autotune_status = QtWidgets.QLabel("")
        self._autotune_status.setObjectName("statusChip")
        self._autotune_status.setWordWrap(True)
        self._autotune_status.setVisible(False)
        speed_body.addWidget(self._autotune_status)
        self._autotune_pending = False
        self._autotune_canceling = False
        self._autotune_batch = False
        self._models_monitor = _HideWhenEmptyLabel("")
        self._models_monitor.setObjectName("statusChip")
        self._models_monitor.setWordWrap(True)
        self._models_monitor.setToolTip(
            "Live numbers from the running model: how fast it writes (tokens "
            "per second) and how full its working memory is."
        )
        speed_body.addWidget(self._models_monitor)
        column.addWidget(speed)

        # -- Thinking level ------------------------------------------------
        think, think_body = self._panel()
        think_body.setContentsMargins(14, 10, 14, 10)
        think_body.setSpacing(6)
        think_row = QtWidgets.QHBoxLayout()
        think_title = QtWidgets.QLabel("Thinking level")
        think_title.setObjectName("sectionTitle")
        think_row.addWidget(think_title)
        # A fixed list (owner request 2026-10-20): free text let any typo
        # through to the model's chat template.
        self._thinking_combo = QtWidgets.QComboBox()
        for label, value in THINKING_CHOICES:
            self._thinking_combo.addItem(label, value)
        self._thinking_combo.setToolTip(
            "How hard the model thinks before it answers. More thinking gives "
            "better answers on hard questions but takes longer."
        )
        think_row.addWidget(self._thinking_combo, 1)
        think_body.addLayout(think_row)
        think_note = QtWidgets.QLabel(
            "Applies the next time you press Start. Not every model can think."
        )
        think_note.setObjectName("muted")
        think_note.setWordWrap(True)
        think_body.addWidget(think_note)
        column.addWidget(think)

        # -- Edit this model -------------------------------------------------------
        edit, edit_body = self._panel()
        edit_body.setContentsMargins(14, 10, 14, 10)
        edit_body.setSpacing(6)
        self._edit_title = QtWidgets.QLabel("Edit this model")
        self._edit_title.setObjectName("sectionTitle")
        edit_body.addWidget(self._edit_title)
        edit_hint = QtWidgets.QLabel("Changes apply the next time the model starts.")
        edit_hint.setObjectName("muted")
        edit_body.addWidget(edit_hint)
        form = QtWidgets.QGridLayout()
        form.setColumnStretch(0, 1)
        gpu_label = QtWidgets.QLabel(
            "GPU offload (more = faster, needs more graphics memory)"
        )
        gpu_label.setWordWrap(True)
        gpu_label.setToolTip(
            "How many layers of the model to keep on the graphics card (the "
            "rest runs on the processor, more slowly). 999 or -1 means as "
            "much as the card can hold."
        )
        form.addWidget(gpu_label, 0, 0)
        self._gpu_edit = QtWidgets.QLineEdit()
        self._gpu_edit.setMaximumWidth(140)
        self._gpu_edit.setToolTip(gpu_label.toolTip())
        self._gpu_edit.textChanged.connect(self._on_edit_change)
        form.addWidget(self._gpu_edit, 0, 1)
        ctx_label = QtWidgets.QLabel(
            "Context size (how much text it can keep in mind)"
        )
        ctx_label.setWordWrap(True)
        ctx_label.setToolTip(
            "How much of a conversation the model can keep in mind at once, in "
            "tokens (small pieces of words). Bigger remembers more but uses more "
            "memory."
        )
        form.addWidget(ctx_label, 1, 0)
        self._ctx_edit = QtWidgets.QLineEdit()
        self._ctx_edit.setMaximumWidth(140)
        self._ctx_edit.setToolTip(ctx_label.toolTip())
        self._ctx_edit.textChanged.connect(self._on_edit_change)
        form.addWidget(self._ctx_edit, 1, 1)
        edit_body.addLayout(form)
        action_row = QtWidgets.QHBoxLayout()
        self._edit_error = QtWidgets.QLabel("")
        self._edit_error.setObjectName("error")
        self._edit_error.setWordWrap(True)
        action_row.addWidget(self._edit_error, 1)
        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.setObjectName("primaryButton")
        self._gate(self._save_btn, False, 'Nothing to save yet - change a value first, or fix the problem shown.')
        self._save_btn.clicked.connect(self._on_save)
        action_row.addWidget(self._save_btn)
        edit_body.addLayout(action_row)

        rename_form = QtWidgets.QGridLayout()
        rename_form.setColumnStretch(1, 1)
        rename_form.setColumnStretch(3, 2)
        rename_form.addWidget(QtWidgets.QLabel("Short id"), 0, 0)
        self._id_edit = QtWidgets.QLineEdit()
        self._id_edit.setToolTip("A short name used inside locitize. Letters, numbers and dashes.")
        self._id_edit.textChanged.connect(self._on_identity_change)
        rename_form.addWidget(self._id_edit, 0, 1)
        rename_form.addWidget(QtWidgets.QLabel("Display name"), 0, 2)
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.textChanged.connect(self._on_identity_change)
        rename_form.addWidget(self._name_edit, 0, 3)
        edit_body.addLayout(rename_form)
        identity_row = QtWidgets.QHBoxLayout()
        self._identity_error = QtWidgets.QLabel("")
        self._identity_error.setObjectName("error")
        self._identity_error.setWordWrap(True)
        identity_row.addWidget(self._identity_error, 1)
        self._identity_save_btn = QtWidgets.QPushButton("Rename")
        self._identity_save_btn.setObjectName("primaryButton")
        self._gate(self._identity_save_btn, False, 'Nothing to rename yet - change the id or name first, or fix the problem shown.')
        self._identity_save_btn.clicked.connect(self._on_save_identity)
        identity_row.addWidget(self._identity_save_btn)
        edit_body.addLayout(identity_row)

        edit_body.addWidget(
            QtWidgets.QLabel("What this model is good at (for example: vision, reasoning, tools)")
        )
        self._capabilities_edit = QtWidgets.QLineEdit()
        self._capabilities_edit.textChanged.connect(self._on_capabilities_change)
        edit_body.addWidget(self._capabilities_edit)
        capabilities_row = QtWidgets.QHBoxLayout()
        self._capabilities_error = QtWidgets.QLabel("")
        self._capabilities_error.setObjectName("error")
        self._capabilities_error.setWordWrap(True)
        capabilities_row.addWidget(self._capabilities_error, 1)
        self._capabilities_save_btn = QtWidgets.QPushButton("Save")
        self._capabilities_save_btn.setObjectName("primaryButton")
        self._gate(self._capabilities_save_btn, False, 'Nothing to save yet - change the list first.')
        self._capabilities_save_btn.clicked.connect(self._on_save_capabilities)
        capabilities_row.addWidget(self._capabilities_save_btn)
        edit_body.addLayout(capabilities_row)
        column.addWidget(edit)

        # -- Manage -----------------------------------------------------------
        manage, manage_body = self._panel()
        manage_body.setContentsMargins(14, 10, 14, 10)
        manage_body.setSpacing(6)
        self._section(manage_body, "Manage this model")
        manage_row = QtWidgets.QHBoxLayout()
        self._register_btn = QtWidgets.QPushButton("Add to my models")
        self._register_btn.setToolTip(
            "Add this fine-tuned model to your model list"
        )
        self._register_btn.clicked.connect(self._on_register_selected)
        self._register_btn.setVisible(False)
        self._delete_btn = QtWidgets.QPushButton("Delete this model...")
        self._delete_btn.setObjectName("dangerButton")
        self._delete_btn.setToolTip(
            "Permanently delete this model's file(s) from disk and remove it "
            "from the list"
        )
        self._delete_btn.clicked.connect(self._on_delete_model)
        manage_row.addWidget(self._register_btn)
        manage_row.addWidget(self._delete_btn)
        manage_row.addStretch(1)
        manage_body.addLayout(manage_row)
        column.addWidget(manage)

        # -- Memory use ---------------------------------------------------------
        memory, memory_body = self._panel()
        self._add_system_meters(memory_body)
        column.addWidget(memory)
        column.addStretch(1)

        scroll = self._scrollable(scroll_body)
        tab_layout.addWidget(scroll, 1)
        return tab

    def _build_get_models_panel(self):
        """Build the "Get models" panel (a tab of the Models page).

        Nothing in this method touches the network. The page paints from the
        shipped catalog on disk; the only three controls that can cause egress
        are Search, selecting a search result, and Download (egress rule HF-1).
        """
        panel, body = self._panel()
        self._section(
            body,
            "Find a model to download",
            "locitize only contacts huggingface.co when you press Search, pick "
            "a result, or confirm a download - never on its own.",
        )
        self._hub_enabled = bool(getattr(self._gc, "hub_enabled", lambda: True)())

        search_row = QtWidgets.QHBoxLayout()
        self._hub_query = QtWidgets.QLineEdit()
        self._hub_query.setPlaceholderText("Search huggingface.co for a model (GGUF)")
        self._hub_query.returnPressed.connect(self._on_hub_search)
        self._hub_search_btn = QtWidgets.QPushButton("Search")
        self._hub_search_btn.clicked.connect(self._on_hub_search)
        search_row.addWidget(self._hub_query, 1)
        search_row.addWidget(self._hub_search_btn)
        body.addLayout(search_row)

        self._hub_repo_table = QtWidgets.QTableWidget(0, 4)
        self._attach_empty_state(self._hub_repo_table, "Search huggingface.co above.")
        self._hub_repo_table.setHorizontalHeaderLabels(
            ["Model page", "Publisher", "Licence", "Where from"]
        )
        self._prepare_hub_table(self._hub_repo_table)
        self._hub_repo_table.currentCellChanged.connect(self._on_hub_repo_selected)
        body.addWidget(self._hub_repo_table)

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
            "Select a result to list its downloadable files.",
        )
        self._hub_file_table.setHorizontalHeaderLabels(
            ["File", "Quality level", "Size", "Fits your GPU?", "Checksum"]
        )
        self._prepare_hub_table(self._hub_file_table)
        self._hub_file_table.currentCellChanged.connect(self._on_hub_file_selected)
        body.addWidget(self._hub_file_table)

        self._hub_fit = QtWidgets.QLabel("")
        self._hub_fit.setObjectName("muted")
        self._hub_fit.setWordWrap(True)
        body.addWidget(self._hub_fit)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        self._hub_download_btn = QtWidgets.QPushButton("Download")
        self._hub_download_btn.setObjectName("primaryButton")
        self._hub_download_btn.setToolTip("Download the selected file")
        self._gate(self._hub_download_btn, False, "Pick a file from the list first.")
        self._hub_download_btn.clicked.connect(self._on_hub_download)
        self._hub_cancel_btn = QtWidgets.QPushButton("Cancel")
        self._hub_cancel_btn.setToolTip("Cancel the download in progress")
        self._gate(self._hub_cancel_btn, False, "No download is running.")
        self._hub_cancel_btn.clicked.connect(self._on_hub_cancel)
        self._hub_register_box = QtWidgets.QCheckBox("Add to my models when finished")
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

        if not self._hub_enabled:
            self._hub_status.setText(
                "Downloading models is switched off. Turn it on in Settings > "
                "Features, or re-run setup."
            )
            for widget in (
                self._hub_query,
                self._hub_search_btn,
                self._hub_load_more_btn,
                self._hub_download_btn,
                self._hub_cancel_btn,
            ):
                self._gate(widget, False, "Downloading models is switched off.")
        return panel

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
        # A header plus about four rows: tall enough to read, short enough that
        # both tables, the buttons and the status line fit a 1366x768 window.
        table.setMinimumHeight(HUB_TABLE_MIN_HEIGHT)
        table.setMaximumHeight(170)
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
        """Build the Fine-tune page: the guided four-step flow plus discovered models.

        The flow (finetune_page.FineTuneFlow) picks a model, prepares and reviews a
        dataset, trains and saves. Below it, the fine-tuned .gguf files discovered
        in the outputs folder can be served, registered, saved or deleted.
        """
        page, layout = self._page(
            "Fine-tune",
            "Teach a model your own material. You give it examples, it trains "
            "a small add-on on this computer, and you get a new model to run "
            "here. Nothing is uploaded.",
        )

        # --- What it needs ---
        needs, needs_body = self._panel()
        self._section(
            needs_body,
            "What you need",
            "An NVIDIA graphics card with enough memory for the model you "
            "train (a 3 to 4 billion parameter model needs about 6 to 8 GB "
            "free), and the fine-tuning feature installed, which you add under "
            "Settings > Features. Training takes from several minutes to a few "
            "hours, and LoCiTiZe checks your graphics card before it starts.",
        )
        layout.addWidget(needs)

        # --- Install path (shown only while fine-tuning is not installed) ---
        self._ft_install_panel, install_body = self._panel()
        self._section(
            install_body,
            "Fine-tuning is not installed yet",
            "Add it once and this page unlocks. The install downloads the "
            "training tools (several gigabytes) and needs an NVIDIA graphics "
            "card.",
        )
        install_row = QtWidgets.QHBoxLayout()
        self._ft_install_btn = QtWidgets.QPushButton("Install fine-tuning")
        self._ft_install_btn.setObjectName("primaryButton")
        self._ft_install_btn.setToolTip(
            "Opens the setup window so you can add the fine-tuning feature"
        )
        self._ft_install_btn.clicked.connect(self._on_open_setup_wizard)
        self._ft_features_btn = QtWidgets.QPushButton("Open Settings > Features")
        self._ft_features_btn.clicked.connect(lambda: self._go_to("Settings"))
        install_row.addWidget(self._ft_install_btn)
        install_row.addWidget(self._ft_features_btn)
        install_row.addStretch(1)
        self._ft_install_reason = QtWidgets.QLabel("")
        self._ft_install_reason.setObjectName("muted")
        self._ft_install_reason.setWordWrap(True)
        install_body.addWidget(self._ft_install_reason)
        install_body.addLayout(install_row)
        self._ft_install_panel.setVisible(False)
        layout.addWidget(self._ft_install_panel)

        # --- The guided flow: model, dataset, train, save ---
        from finetune_page import FineTuneFlow

        self._ft_flow = FineTuneFlow(
            self, self._gc, _Sparkline, on_model_saved=self._on_finetune_flow_saved
        )
        layout.addWidget(self._ft_flow)

        # --- Panel B: discovered models ---
        found, found_body = self._panel()
        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(QtWidgets.QLabel("Your fine-tuned models"))
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
            "No fine-tuned models yet.\n"
            "Train one with the steps above; finished runs appear here.",
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
        self._ft_save_btn = QtWidgets.QPushButton("Save a copy...")
        self._ft_save_btn.setToolTip(
            "Open the selected run in step 4 to save it to a folder of your choice"
        )
        self._ft_save_btn.clicked.connect(self._on_finetune_save_copy)
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
        for button in (self._ft_serve_btn, self._ft_register_btn, self._ft_save_btn,
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
        self._apply_finetune_state({"status": "ready", "reason": ""})
        self._refresh_finetune_buttons()
        # Four steps plus the run list are taller than a laptop screen.
        self._ft_table.setMinimumHeight(220)
        return self._scrollable(page)

    def _build_code_page(self):
        """Code: launch a coding tool on your model, resume sessions, search chats."""
        page, layout = self._page(
            "Code",
            "Code with your local model, pick up a past coding session, or "
            "search your earlier chats.",
        )
        card, body = self._panel()
        self._section(
            body,
            "Code with your local model",
            "Opens a terminal running a coding tool on your model, working in a "
            "project folder you pick. If no model is running yet, the one "
            "chosen on the Start page is started for you.",
        )
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self._code_harness_buttons = {}
        for key, label in (
            ("codex", "Codex"),
            ("claude", "Claude Code"),
            ("opencode", "OpenCode"),
        ):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _checked=False, k=key: self._on_code_tool(k))
            row.addWidget(button)
            self._code_harness_buttons[key] = button
        row.addStretch(1)
        body.addLayout(row)
        self._refresh_code_harness_buttons()
        self._code_note = self._model_status.mirror()
        body.addWidget(self._code_note)

        release_row = QtWidgets.QHBoxLayout()
        self._release_btn = QtWidgets.QPushButton("Release model")
        self._release_btn.setToolTip(
            "Tell locitize you are done with your coding terminals so the model "
            "can be switched or stopped again."
        )
        self._release_btn.clicked.connect(self._on_release_model)
        release_row.addWidget(self._release_btn)
        release_hint = QtWidgets.QLabel(
            "While a coding terminal is open, locitize keeps its model loaded so "
            "it is not switched out from under you. Press Release model when you "
            "are finished. Open terminals keep running but lose their model."
        )
        release_hint.setObjectName("muted")
        release_hint.setWordWrap(True)
        release_row.addWidget(release_hint, 1)
        body.addLayout(release_row)
        layout.addWidget(card)

        self._code_tabs = QtWidgets.QTabWidget()
        self._code_tabs.setObjectName("subTabs")
        self._code_tabs.setDocumentMode(True)
        self._code_tabs.tabBar().setDrawBase(False)
        sessions = self._build_sessions_page()
        sessions.layout().setContentsMargins(0, 12, 0, 0)
        self._code_tabs.addTab(sessions, "Past sessions")

        memory_tab, memory_layout = self._tab_body(
            "Search what you have said to your local models. Leave the box "
            "empty to see your most recent chats."
        )
        panel, mem_body = self._panel()
        mem_row = QtWidgets.QHBoxLayout()
        self._memory_query = QtWidgets.QLineEdit()
        self._memory_query.setPlaceholderText("Search words, or leave blank for recent chats")
        self._memory_query.returnPressed.connect(self._on_memory_search)
        self._memory_btn = QtWidgets.QPushButton("Search")
        self._memory_btn.setObjectName("primaryButton")
        self._memory_btn.clicked.connect(self._on_memory_search)
        mem_row.addWidget(self._memory_query, 1)
        mem_row.addWidget(self._memory_btn)
        mem_body.addLayout(mem_row)
        self._memory_result = QtWidgets.QTextEdit()
        self._memory_result.setReadOnly(True)
        self._memory_result.setPlaceholderText("Your recent chats will appear here.")
        mem_body.addWidget(self._memory_result, 1)
        memory_layout.addWidget(panel, 1)
        self._code_tabs.addTab(memory_tab, "Search past chats")
        self._code_tabs.currentChanged.connect(self._on_code_tab_changed)
        layout.addWidget(self._code_tabs, 1)
        return page

    def _on_code_tab_changed(self, index):
        """First visit to "Search past chats" shows the recent chats."""
        if self._code_tabs.tabText(index) == "Search past chats" and not self._memory_visited:
            self._memory_visited = True
            self._on_memory_search()

    def _on_code_tool(self, key):
        """Open the one coding-session dialog for every launch.

        The dialog shows the selected model and project before anything starts,
        including when a model is already running.
        """
        self._sessions_page.local_launch(None, tool=key)

    def _on_release_model(self):
        """Release the coding-terminal reservation (the one Release button)."""
        self._model_status.setText("Releasing the model...")
        self._sessions_page.release_model()

    def _refresh_code_harness_buttons(self):
        """Enable each coding-tool button per a fresh PATH detection.

        A detection failure must never break the window: the buttons then stay
        disabled and say where to install the tool.
        """
        if not getattr(self, "_code_harness_buttons", None):
            return
        try:
            installed = self._gc.detect_harnesses()
        except Exception:  # noqa: BLE001 - degrade to disabled, never crash
            installed = {}
        if not isinstance(installed, dict):
            installed = {}
        for key, button in self._code_harness_buttons.items():
            found = installed.get(key)
            self._gate(
                button,
                bool(found),
                "Not installed yet. Install it in Settings > Coding tools.",
            )
            if found:
                button.setToolTip(f"Open {button.text()} on your local model")

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
        title = QtWidgets.QLabel("Chat from your phone")
        title.setObjectName("sectionTitle")
        column.addWidget(title)
        row = QtWidgets.QHBoxLayout()
        self._phone_url_edit = QtWidgets.QLineEdit(url)
        self._phone_url_edit.setReadOnly(True)
        self._phone_url_edit.setCursorPosition(0)
        row.addWidget(self._phone_url_edit, 1)
        self._phone_copy_btn = QtWidgets.QPushButton("Copy link")
        self._phone_copy_btn.clicked.connect(self._on_copy_phone_url)
        row.addWidget(self._phone_copy_btn)
        column.addLayout(row)
        self._phone_hint = QtWidgets.QLabel(
            "Works on your own Tailscale network only. Open the Tailscale app on "
            "the phone, then this link. A headset is more reliable than the "
            "phone speaker for voice."
        )
        self._phone_hint.setObjectName("muted")
        self._phone_hint.setWordWrap(True)
        column.addWidget(self._phone_hint)

    def _on_copy_phone_url(self):
        """Copy the Tailscale phone URL to the clipboard as plain text."""
        if self._phone_url_edit is None:
            return
        QtWidgets.QApplication.clipboard().setText(self._phone_url_edit.text())
        if self._phone_hint is not None:
            self._phone_hint.setText("Copied. Open it on the phone with Tailscale on.")

    def _build_settings_page(self):
        """Settings: app-level options only. Model options live on Models."""
        page, layout = self._page(
            "Settings",
            "What is installed, coding tools, phone access, microphone cleanup "
            "and the app itself. Options for a single model are on the Models "
            "page, under Advanced.",
        )

        # ---- Features -------------------------------------------------
        features_panel, features_body = self._panel()
        self._section(
            features_body,
            "Features",
            "What this computer has installed. Add anything missing with setup - "
            "pieces that are already installed are never downloaded twice.",
        )
        self._feature_rows_grid = QtWidgets.QGridLayout()
        self._feature_rows_grid.setColumnStretch(0, 1)
        features_body.addLayout(self._feature_rows_grid)
        self._feature_state_note = QtWidgets.QLabel("checking this computer...")
        self._feature_state_note.setObjectName("muted")
        self._feature_state_note.setWordWrap(True)
        features_body.addWidget(self._feature_state_note)
        wizard_row = QtWidgets.QHBoxLayout()
        self._open_wizard_btn = QtWidgets.QPushButton("Add features (setup)")
        self._open_wizard_btn.setToolTip(
            "Opens the setup window so you can add voice, chat or other pieces"
        )
        self._open_wizard_btn.clicked.connect(self._on_open_setup_wizard)
        wizard_row.addWidget(self._open_wizard_btn)
        wizard_row.addStretch(1)
        features_body.addLayout(wizard_row)
        layout.addWidget(features_panel)

        # ---- Models folder --------------------------------------------
        models_folder, models_folder_body = self._panel()
        self._section(
            models_folder_body,
            "Models folder",
            "New model downloads and setup imports go here. Existing model "
            "list entries stay where they are.",
        )
        folder_row = QtWidgets.QHBoxLayout()
        current_store = ""
        if hasattr(self._gc, "model_store_path"):
            current_store = self._gc.model_store_path()
        self._model_store_edit = QtWidgets.QLineEdit(current_store)
        self._model_store_edit.setReadOnly(True)
        self._model_store_edit.setAccessibleName("Models folder")
        folder_row.addWidget(self._model_store_edit, 1)
        self._model_store_change = QtWidgets.QPushButton("Change...")
        self._model_store_change.clicked.connect(self._on_change_model_store)
        folder_row.addWidget(self._model_store_change)
        models_folder_body.addLayout(folder_row)
        self._model_store_note = QtWidgets.QLabel("")
        self._model_store_note.setObjectName("muted")
        self._model_store_note.setWordWrap(True)
        models_folder_body.addWidget(self._model_store_note)
        layout.addWidget(models_folder)

        # ---- Coding tools -------------------------------------------------
        clis_panel, clis_body = self._panel()
        self._section(
            clis_body,
            "Coding tools",
            "Programs locitize can open on your model from the Code page. "
            "Install any of them here.",
        )
        clis_grid = QtWidgets.QGridLayout()
        clis_grid.setColumnStretch(1, 1)
        self._cli_status_labels = {}
        for row_index, (key, label) in enumerate(
            (("codex", "Codex"), ("claude", "Claude Code"), ("opencode", "OpenCode"))
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
        clis_body.addLayout(clis_grid)
        self._refresh_cli_status_rows()
        layout.addWidget(clis_panel)

        # ---- Phone access -----------------------------------------------------
        phone_panel, phone_body = self._panel()
        self._section(phone_body, "Phone access")
        self._phone_access_note = QtWidgets.QLabel(self._phone_access_text())
        self._phone_access_note.setObjectName("muted")
        self._phone_access_note.setWordWrap(True)
        phone_body.addWidget(self._phone_access_note)
        layout.addWidget(phone_panel)

        # ---- Noise suppression ----------------------------------------------------
        noise, noise_layout = self._panel()
        self._section(
            noise_layout,
            "Microphone cleanup (noise suppression)",
            "Cleans up recordings from the chat page's microphone on this "
            "computer. Balanced is the default. Strong removes more fan and "
            "rumble. Off sends the recording unchanged. Talk uses its own "
            "built-in cleanup.",
        )
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

        # ---- Health ----------------------------------------------------------------
        health_panel, health_body = self._panel()
        self._section(
            health_body,
            "Health",
            "The checks locitize ran when it started. Click the Health chip at "
            "the top of the window any time to see this again.",
        )
        self._health_detail = QtWidgets.QPlainTextEdit()
        self._health_detail.setReadOnly(True)
        self._health_detail.setMaximumHeight(150)
        self._health_detail.setPlainText(self._health_details_text())
        health_body.addWidget(self._health_detail)
        layout.addWidget(health_panel)

        # ---- About ----------------------------------------------------------------
        about_panel, about_body = self._panel()
        self._section(about_body, "About")
        try:
            from release_info import VERSION
        except Exception:  # noqa: BLE001 - the page must paint without it
            VERSION = ""
        about_row = QtWidgets.QHBoxLayout()
        about_label = QtWidgets.QLabel(
            f"locitize {VERSION}".strip() + " - your AI, running on this computer."
        )
        about_label.setObjectName("muted")
        about_row.addWidget(about_label, 1)
        self._about_btn = QtWidgets.QPushButton("About and diagnostics")
        self._about_btn.setToolTip("Version and technical details you can copy for support")
        self._about_btn.clicked.connect(self._show_release_diagnostics)
        about_row.addWidget(self._about_btn)
        about_body.addLayout(about_row)
        layout.addWidget(about_panel)

        # ---- Uninstall ---------------------------------------------------------------
        uninstall_panel, uninstall_body = self._panel()
        self._section(uninstall_body, "Uninstall")
        uninstall_row = QtWidgets.QHBoxLayout()
        uninstall_hint = QtWidgets.QLabel(
            "Remove locitize from this computer. You can keep your model files."
        )
        uninstall_hint.setObjectName("muted")
        uninstall_hint.setWordWrap(True)
        uninstall_row.addWidget(uninstall_hint, 1)
        self._uninstall_btn = QtWidgets.QPushButton("Uninstall locitize...")
        self._uninstall_btn.setObjectName("dangerButton")
        self._uninstall_btn.setToolTip(
            "Remove what setup installed (with the option to keep your model "
            "files). Opens a separate window; locitize will close."
        )
        self._uninstall_btn.clicked.connect(self._on_uninstall)
        uninstall_row.addWidget(self._uninstall_btn)
        uninstall_body.addLayout(uninstall_row)
        layout.addWidget(uninstall_panel)

        if hasattr(self._gc, "request_feature_state"):
            self._gc.request_feature_state()
        layout.addStretch(1)
        return self._scrollable(page)

    def _on_change_model_store(self):
        current = self._model_store_edit.text().strip()
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose your Models folder", current
        )
        if not chosen:
            return
        self._model_store_change.setEnabled(False)
        self._model_store_note.setText("Checking the folder...")
        self._gc.request_model_store_change(chosen)

    def _apply_model_store_change(self, result):
        self._model_store_change.setEnabled(True)
        if not result.ok:
            self._model_store_note.setText(result.error or "Could not save the Models folder.")
            return
        self._model_store_edit.setText(result.payload.get("path", ""))
        self._model_store_note.setText(result.payload.get("message", "Models folder saved."))

    def _phone_access_text(self):
        """Plain status for the Settings > Phone access section."""
        reader = getattr(self._gc, "phone_access", None)
        access = reader() if callable(reader) else None
        url = str(getattr(access, "url", "") or "") if access is not None else ""
        if url:
            return (
                "Your phone link is ready and shown on the Start page with a "
                "Copy button. It works on your own Tailscale network only."
            )
        return (
            "Not set up. To chat from your phone, install Tailscale on this "
            "computer and on the phone, sign in to both, and turn on Tailscale "
            "Serve for locitize. The link then appears on the Start page."
        )

    def _health_label(self):
        """Short word for the header chip."""
        return {
            "PASS": "Good",
            "WARNING": "Needs attention",
            "FAIL": "Problem",
        }.get(str(self._health).upper(), "Not checked")

    def _health_reasons(self):
        """The startup checks that were not clean, as plain lines."""
        report = self._health_report
        results = getattr(report, "results", None) or []
        lines = []
        for result in results:
            status = getattr(getattr(result, "status", None), "value", "")
            if status == "PASS":
                continue
            line = f"{getattr(result, 'name', 'check')}: {getattr(result, 'detail', '')}"
            remedy = getattr(result, "remedy", None)
            if remedy:
                line += f"  Fix: {remedy}"
            lines.append(line)
        return lines

    def _health_details_text(self):
        reasons = self._health_reasons()
        if reasons:
            return "\n\n".join(reasons)
        if getattr(self._health_report, "results", None):
            return "Every startup check passed."
        return (
            "No startup check details are available in this session. Open "
            "About and diagnostics in Settings for the full technical report."
        )

    def _update_health_chip(self):
        """Paint the header Health chip and its tooltip with the real reasons."""
        label = self._health_label()
        self._health_btn.setText(f"Health: {label}")
        self._health_btn.setProperty("healthState", str(self._health).upper())
        reasons = self._health_reasons()
        if reasons:
            tip = "Click for details.\n\n" + "\n".join(reasons[:4])
        elif getattr(self._health_report, "results", None):
            tip = "Every startup check passed. Click for details."
        else:
            tip = "Click for details about the startup checks."
        self._health_btn.setToolTip(plain_tooltip_text(tip))
        style = self._health_btn.style()
        style.unpolish(self._health_btn)
        style.polish(self._health_btn)

    def _show_health_details(self):
        """Explain the Health chip in a dialog: each check that needs attention."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Health")
        layout = QtWidgets.QVBoxLayout(dialog)
        headline = QtWidgets.QLabel(f"Health: {self._health_label()}")
        headline.setObjectName("sectionTitle")
        layout.addWidget(headline)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(self._health_details_text())
        layout.addWidget(text)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(560, 360)
        harden_text_rendering(dialog).exec()

    def _build_start_page(self):
        """Start: pick a model, press one big button, then chat.

        This replaces the old Home, Chat page and the model-start controls that
        lived on Models. Everything a first-time user needs is on this screen.
        """
        page, layout = self._page(
            "Start",
            "Pick a model, press Start, then open the chat. Everything runs on "
            "this computer.",
        )
        card, body = self._panel()
        self._section(body, "1. Choose a model")
        self._start_model_combo = QtWidgets.QComboBox()
        self._start_model_combo.setAccessibleName("Model to run")
        self._start_model_combo.currentIndexChanged.connect(self._on_start_combo_changed)
        body.addWidget(self._start_model_combo)
        self._start_fit_label = QtWidgets.QLabel("")
        self._start_fit_label.setObjectName("muted")
        self._start_fit_label.setWordWrap(True)
        body.addWidget(self._start_fit_label)
        self._start_empty_row = QtWidgets.QWidget()
        empty_layout = QtWidgets.QHBoxLayout(self._start_empty_row)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_label = QtWidgets.QLabel("You do not have a model yet.")
        empty_label.setObjectName("muted")
        empty_layout.addWidget(empty_label)
        get_btn = QtWidgets.QPushButton("Get a model")
        get_btn.setObjectName("primaryButton")
        get_btn.clicked.connect(lambda: self._go_to("Models", "Get models"))
        empty_layout.addWidget(get_btn)
        empty_layout.addStretch(1)
        body.addWidget(self._start_empty_row)

        buttons = QtWidgets.QHBoxLayout()
        self._start_btn = QtWidgets.QPushButton("Start")
        self._start_btn.setObjectName("primaryButton")
        self._start_btn.setMinimumHeight(48)
        self._start_btn.setMinimumWidth(200)
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QtWidgets.QPushButton("Stop")
        self._stop_btn.setMinimumHeight(48)
        self._stop_btn.setToolTip("Stop the running model and free its memory")
        self._stop_btn.clicked.connect(self._on_stop)
        buttons.addWidget(self._start_btn)
        buttons.addWidget(self._stop_btn)
        buttons.addStretch(1)
        body.addLayout(buttons)
        self._start_status_line = QtWidgets.QLabel("No model running")
        self._start_status_line.setObjectName("statusChip")
        self._start_status_line.setWordWrap(True)
        body.addWidget(self._start_status_line)
        # The one shared message line (errors, progress): every page that can
        # raise an action shows a mirror of it so no outcome is hidden.
        self._model_status = _SharedLabel("")
        self._model_status.setObjectName("muted")
        self._model_status.setWordWrap(True)
        self._model_status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self._chat_status = self._model_status
        body.addWidget(self._model_status)
        layout.addWidget(card)

        chat_card, chat_body = self._panel()
        self._section(chat_body, "2. Chat")
        self._openwebui_banner = QtWidgets.QFrame()
        banner_layout = QtWidgets.QHBoxLayout(self._openwebui_banner)
        banner_layout.setContentsMargins(0, 0, 0, 8)
        self._openwebui_banner_text = QtWidgets.QLabel(
            "Open WebUI is not installed yet. Add richer chat with history and uploads."
        )
        self._openwebui_banner_text.setWordWrap(True)
        banner_layout.addWidget(self._openwebui_banner_text, 1)
        self._openwebui_install_btn = QtWidgets.QPushButton("Install Open WebUI")
        self._openwebui_install_btn.clicked.connect(self._on_openwebui_banner_install)
        banner_layout.addWidget(self._openwebui_install_btn)
        self._openwebui_cancel_btn = QtWidgets.QPushButton("Cancel")
        self._openwebui_cancel_btn.clicked.connect(self._on_openwebui_install_cancel)
        self._openwebui_cancel_btn.setVisible(False)
        banner_layout.addWidget(self._openwebui_cancel_btn)
        chat_body.addWidget(self._openwebui_banner)
        self._refresh_openwebui_banner()
        self._chat_hint = QtWidgets.QLabel("")
        self._chat_hint.setObjectName("muted")
        self._chat_hint.setWordWrap(True)
        chat_body.addWidget(self._chat_hint)
        chat_row = QtWidgets.QHBoxLayout()
        self._chat_btn = QtWidgets.QPushButton("Open chat")
        self._chat_btn.setObjectName("primaryButton")
        self._chat_btn.setMinimumHeight(40)
        self._chat_btn.setToolTip("Open the chat for the running model in your browser")
        self._chat_btn.clicked.connect(self._on_chat)
        chat_row.addWidget(self._chat_btn)
        chat_row.addStretch(1)
        chat_body.addLayout(chat_row)
        self._build_phone_strip(chat_body)
        layout.addWidget(chat_card)
        layout.addStretch(1)
        return self._scrollable(page)

    def _model_name(self, model_id):
        """Plain display name for a model id, falling back to the id."""
        for model in self._models:
            if model["id"] == model_id:
                return str(model.get("name") or model_id)
        return str(model_id or "")

    def _gpu_total_mb(self):
        try:
            return self._gc.system_specs().gpu_vram_total_mb
        except Exception:  # noqa: BLE001 - hardware facts must never break a page
            return None

    def _gpu_usable_mb(self):
        """The VRAM a model start can use, from the startup Health probe."""
        for result in getattr(self._health_report, "results", None) or []:
            if getattr(result, "name", "") != "vram":
                continue
            value = (getattr(result, "data", None) or {}).get("effective_free_mb")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return self._gpu_total_mb()

    def _populate_start_combo(self):
        """Rebuild the Start page's model picker from the cached model rows."""
        combo = self._start_model_combo
        combo.blockSignals(True)
        combo.clear()
        gpu_usable = self._gpu_usable_mb()
        for model in self._models:
            if _model_turned_off(model):
                continue
            fit = gui_controller.format_fit(model.get("vram_need_mb"), gpu_usable)
            label = f"{model['name']}  -  {model.get('size_display', '-')}"
            if fit == "Yes":
                label += "  -  fits your GPU"
            elif fit == "Partly":
                label += "  -  partly on your GPU"
            combo.addItem(label, model["id"])
        index = combo.findData(self._selected_id)
        if index < 0 and combo.count():
            index = 0
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self._start_empty_row.setVisible(combo.count() == 0)
        combo.setVisible(combo.count() > 0)
        self._update_start_fit()

    def _update_start_fit(self):
        model = self._selected_model()
        if model is None:
            self._start_fit_label.setText("")
            return
        self._start_fit_label.setText(
            gui_controller.describe_fit(model.get("vram_need_mb"), self._gpu_usable_mb())
        )

    def _on_start_combo_changed(self, _index):
        """Picking a model on Start selects the same row on Models."""
        model_id = self._start_model_combo.currentData()
        if model_id and model_id != self._selected_id:
            self._selected_id = model_id
            self._restore_model_selection()
            self._note_selected_model()
            settings_index = self._settings_model.findData(model_id)
            if settings_index >= 0 and self._settings_model.currentIndex() != settings_index:
                self._settings_model.blockSignals(True)
                self._settings_model.setCurrentIndex(settings_index)
                self._settings_model.blockSignals(False)
            self._load_settings_model()
            self._refresh_buttons()
        self._update_start_fit()

    def _build_voice_vision_page(self):
        """Voice & Vision: talk, mic and voice setup, images and screen watching."""
        page, layout = self._page(
            "Voice & Vision",
            "Talk to your model out loud, test your microphone and voices, ask "
            "about a picture, or let it watch your screen.",
        )
        banner = QtWidgets.QFrame()
        banner.setObjectName("banner")
        banner_row = QtWidgets.QHBoxLayout(banner)
        banner_row.setContentsMargins(14, 10, 14, 10)
        self._vv_banner_label = QtWidgets.QLabel("")
        self._vv_banner_label.setWordWrap(True)
        banner_row.addWidget(self._vv_banner_label, 1)
        self._vv_banner_btn = QtWidgets.QPushButton("")
        self._vv_banner_btn.clicked.connect(self._on_vv_banner_fix)
        banner_row.addWidget(self._vv_banner_btn)
        self._vv_banner = banner
        self._vv_banner_target = "Start"
        layout.addWidget(banner)

        self._vv_tabs = QtWidgets.QTabWidget()
        self._vv_tabs.setObjectName("subTabs")
        self._vv_tabs.setDocumentMode(True)
        self._vv_tabs.tabBar().setDrawBase(False)
        self._vv_tabs.addTab(self._build_talk_tab(), "Talk")
        self._vv_tabs.addTab(self._build_voice_setup_tab(), "Voice and mic setup")
        self._vv_tabs.addTab(self._build_vision_tab(), "Ask about an image")
        self._vv_tabs.addTab(self._build_watch_tab(), "Watch my screen")
        layout.addWidget(self._vv_tabs, 1)
        return page

    def _vv_missing_voice(self):
        """Plain names of the voice pieces that are not installed (may be empty)."""
        reader = getattr(self._gc, "voice_readiness", None)
        if not callable(reader):
            return []
        try:
            ready = reader() or {}
        except Exception:  # noqa: BLE001 - never block a page on a probe
            return []
        missing = []
        if not ready.get("listen", True):
            missing.append("voice input")
        if not ready.get("speak", True):
            missing.append("voice replies")
        return missing

    def _update_vv_banner(self):
        """One shared banner: what Voice & Vision still needs, with a fix button."""
        if not hasattr(self, "_vv_banner"):
            return
        missing = self._vv_missing_voice()
        if missing:
            self._vv_banner_label.setText(
                "Not installed yet: " + " and ".join(missing) + ". Add it in "
                "Settings > Features, then restart locitize."
            )
            self._vv_banner_btn.setText("Open Settings")
            self._vv_banner_target = "Settings"
            self._vv_banner.setVisible(True)
        elif not self._ui.running_model_id and not self._assistant_live:
            self._vv_banner_label.setText(
                "No model is running. Talk starts the model chosen on the Start "
                "page for you; asking about an image or watching your screen "
                "needs one."
            )
            self._vv_banner_btn.setText("Go to Start")
            self._vv_banner_target = "Start"
            self._vv_banner.setVisible(True)
        else:
            self._vv_banner.setVisible(False)

    def _on_vv_banner_fix(self):
        self._go_to(self._vv_banner_target)

    def _build_talk_tab(self):
        """Talk: one big mic control and the running conversation."""
        tab, layout = self._tab_body(
            "Click Talk and speak. Talking over the reply also cuts in. After "
            "each answer it listens again."
        )
        session, session_layout = self._panel()
        row = QtWidgets.QHBoxLayout()
        self._assistant_start_btn = QtWidgets.QPushButton("Start voice chat")
        self._assistant_start_btn.setObjectName("primaryButton")
        self._assistant_start_btn.clicked.connect(self._on_start_assistant)
        self._assistant_end_btn = QtWidgets.QPushButton("End")
        self._assistant_end_btn.setToolTip("End the voice chat and release the model")
        self._assistant_end_btn.clicked.connect(self._on_end_assistant)
        self._gate(self._assistant_end_btn, False, 'Voice chat is not running.')
        self._interrupt_btn = QtWidgets.QPushButton("Interrupt")
        self._interrupt_btn.setToolTip("Stop the reply that is being spoken")
        self._interrupt_btn.clicked.connect(self._on_interrupt)
        self._gate(self._interrupt_btn, False, 'Nothing is being spoken.')
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
        self._assistant_status.setWordWrap(True)
        session_layout.addWidget(self._assistant_status)
        self._talk_btn = QtWidgets.QPushButton("Talk")
        self._talk_btn.setObjectName("talkButton")
        self._gate(self._talk_btn, False, 'Press Start voice chat first.')
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
            lambda: self._go_to("Voice & Vision", "Voice and mic setup")
        )
        session_layout.addWidget(voice_setup_btn)
        layout.addWidget(session)

        conversation, conversation_layout = self._panel()
        conversation_layout.addWidget(QtWidgets.QLabel("Conversation"))
        self._conversation = QtWidgets.QTextEdit()
        self._conversation.setReadOnly(True)
        self._conversation.setPlaceholderText("Click Talk and speak. Turns show up here.")
        conversation_layout.addWidget(self._conversation, 1)
        layout.addWidget(conversation, 1)

        shortcut = QtGui.QShortcut(QtGui.QKeySequence("Space"), tab)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self._on_talk_shortcut)
        return tab

    def _build_voice_setup_tab(self):
        """Voice and mic setup: test the microphone and pick a voice."""
        tab, layout = self._tab_body(
            "This is what Talk hears and speaks with. Test the microphone and "
            "pick a voice."
        )
        stt, stt_layout = self._panel()
        self._section(stt_layout, "Microphone (speech to text)")
        controls = QtWidgets.QHBoxLayout()
        self._whisper_btn = QtWidgets.QPushButton("Turn on voice input")
        self._whisper_btn.setToolTip(
            "Start or stop the part of locitize that turns your speech into text"
        )
        self._whisper_btn.clicked.connect(self._on_whisper)
        self._listen_btn = QtWidgets.QPushButton("Test microphone (15 s)")
        self._listen_btn.setToolTip("Listen for 15 seconds and show what it heard")
        self._listen_btn.clicked.connect(self._on_listen)
        controls.addWidget(self._whisper_btn)
        controls.addWidget(self._listen_btn)
        controls.addStretch(1)
        stt_layout.addLayout(controls)
        self._whisper_status = QtWidgets.QLabel("")
        self._whisper_status.setObjectName("muted")
        self._whisper_status.setWordWrap(True)
        stt_layout.addWidget(self._whisper_status)
        self._voice_status = QtWidgets.QLabel("")
        self._voice_status.setObjectName("muted")
        self._voice_status.setWordWrap(True)
        stt_layout.addWidget(self._voice_status)
        self._transcript = QtWidgets.QTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setPlaceholderText("What it heard appears here.")
        stt_layout.addWidget(self._transcript)
        layout.addWidget(stt, 1)

        tts, tts_layout = self._panel()
        self._section(tts_layout, "Voice (text to speech)")
        voice_row = QtWidgets.QHBoxLayout()
        voice_row.addWidget(QtWidgets.QLabel("Voice"))
        self._voice_combo = QtWidgets.QComboBox()
        self._populate_voice_combo(self._voice_combo)
        self._voice_combo.currentIndexChanged.connect(self._sync_voice_from_voice_page)
        self._speak_btn = QtWidgets.QPushButton("Hear a sample")
        self._speak_btn.clicked.connect(self._on_speak)
        self._audition_btn = QtWidgets.QPushButton("Hear every voice")
        self._audition_btn.clicked.connect(self._on_audition)
        voices_available = self._voice_combo.count() > 0
        no_voices = "No voices are installed. Add voice replies in Settings > Features."
        self._speak_btn.setToolTip("Say a short sentence in the chosen voice")
        self._audition_btn.setToolTip("Say the same sentence in every voice, one after another")
        self._gate(self._speak_btn, voices_available, no_voices)
        self._gate(self._audition_btn, voices_available, no_voices)
        voice_row.addWidget(self._voice_combo)
        voice_row.addWidget(self._speak_btn)
        voice_row.addWidget(self._audition_btn)
        voice_row.addStretch(1)
        tts_layout.addLayout(voice_row)
        self._tts_status = QtWidgets.QLabel(
            "" if voices_available else "No voices installed yet."
        )
        self._tts_status.setObjectName("muted")
        self._tts_status.setWordWrap(True)
        tts_layout.addWidget(self._tts_status)
        layout.addWidget(tts)
        layout.addStretch(0)
        return tab

    def _build_vision_tab(self):
        """Ask about an image: pick or drop a picture, then ask a question."""
        tab, layout = self._tab_body(
            "Pick or drop a picture and ask a question about it. locitize "
            "switches to a model that can see images for this, so a model you "
            "are chatting with may be swapped out."
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
        self._vision_btn.setToolTip("Ask the model about this picture")
        self._gate(self._vision_btn, False, "Pick or drop an image first.")
        self._vision_btn.clicked.connect(self._on_describe)
        prompt_row.addWidget(self._vision_prompt, 1)
        prompt_row.addWidget(self._vision_btn)
        body.addLayout(prompt_row)
        self._vision_result = QtWidgets.QTextEdit()
        self._vision_result.setReadOnly(True)
        self._vision_result.setPlaceholderText("Pick or drop an image, then press Describe.")
        body.addWidget(self._vision_result, 1)
        layout.addWidget(panel, 1)
        self._install_vision_drop(panel)
        return tab

    def _build_watch_tab(self):
        """Watch my screen: speak up when the screen drifts from a stated goal."""
        tab, layout = self._tab_body(
            "Keeps a model that can see images ready and speaks up when your "
            "screen drifts from the goal you type. It ends voice chat first, "
            "because both need the GPU."
        )
        eye, eye_layout = self._panel()
        self._second_eye_goal = QtWidgets.QLineEdit()
        self._second_eye_goal.setPlaceholderText("What you are doing, e.g. learning Python")
        eye_layout.addWidget(self._second_eye_goal)
        eye_row = QtWidgets.QHBoxLayout()
        self._second_eye_start_btn = QtWidgets.QPushButton("Start watching")
        self._second_eye_start_btn.setObjectName("primaryButton")
        self._second_eye_start_btn.clicked.connect(self._on_second_eye_start)
        self._second_eye_stop_btn = QtWidgets.QPushButton("Stop")
        self._second_eye_stop_btn.setToolTip("Stop watching your screen")
        self._gate(self._second_eye_stop_btn, False, "Not watching right now.")
        self._second_eye_stop_btn.clicked.connect(self._on_second_eye_stop)
        eye_row.addWidget(self._second_eye_start_btn)
        eye_row.addWidget(self._second_eye_stop_btn)
        eye_row.addStretch(1)
        eye_layout.addLayout(eye_row)
        self._second_eye_status = QtWidgets.QLabel("Not watching")
        self._second_eye_status.setObjectName("muted")
        self._second_eye_status.setWordWrap(True)
        eye_layout.addWidget(self._second_eye_status)
        layout.addWidget(eye)
        layout.addStretch(1)
        return tab

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
            self._gate(self._save_btn, False, 'Nothing to save yet - change a value first, or fix the problem shown.')
            self._id_edit.setEnabled(False)
            self._name_edit.setEnabled(False)
            self._gate(self._identity_save_btn, False, 'Nothing to rename yet - change the id or name first, or fix the problem shown.')
            self._capabilities_edit.setEnabled(False)
            self._gate(self._capabilities_save_btn, False, 'Nothing to save yet - change the list first.')
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
        """Reload the model list on demand (owner request 2026-08-13) so a model
        added while the desktop is open appears without a restart."""
        try:
            self._gc.refresh_models()
            # refresh_models() returns the manual rows; re-read through the merged
            # loader so discovered fine-tunes survive a Refresh click.
            self._models = self._load_model_rows()
        except Exception as exc:
            self._models_monitor.setText(f"Could not reload the list: {exc}")
            return
        self._refresh_model_rows()
        self._restore_model_selection()
        self._models_monitor.setText(
            f"Model list reloaded ({len(self._models)} models)"
        )

    def _refresh_model_rows(self):
        """Render cached model rows in the controller helper's current sort order."""
        rows = self._models
        if self._sort_col is not None:
            rows = gui_controller.sort_model_rows(rows, self._sort_col, self._sort_desc)
        gpu_total = self._gpu_total_mb()
        gpu_usable = self._gpu_usable_mb()
        self._model_table.blockSignals(True)
        self._model_table.setRowCount(len(rows))
        for row_index, model in enumerate(rows):
            status = self._status_for(model)
            fit = gui_controller.format_fit(model.get("vram_need_mb"), gpu_usable)
            fit_text = {"Yes": "Yes", "Partly": "Partly (slower)"}.get(fit, "-")
            values = (
                model["name"],
                model.get("capabilities_display", "Chat"),
                model.get("source_display", "Registered"),
                model["size_display"],
                fit_text,
                model.get("gpu_portion_display", "-"),
                model.get("cpu_portion_display", "-"),
                plain_speed(model.get("score_display", "-")),
                model.get("context_display", str(model.get("context_size", "-"))),
                status,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, model["id"])
                if column in (3, 5, 6, 7, 8):
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                if column == 4 and fit == "Partly":
                    item.setForeground(QtGui.QColor("#ffd166"))
                if column == 9:
                    self._style_status_item(item, status)
                if column == 7 and model.get("score_note"):
                    item.setToolTip(plain_tooltip_text(model["score_note"]))
                if column == 8 and model.get("ctx_warning"):
                    # A context past the measured cliff makes this model slow.
                    item.setForeground(QtGui.QColor("#ffd166"))
                    item.setText(str(value) + "  !")
                    item.setToolTip(plain_tooltip_text(model["ctx_warning"]))
                self._model_table.setItem(row_index, column, item)
        self._model_table.blockSignals(False)
        self._restore_model_selection()
        self._models_total.setText(
            plain_total(gui_controller.total_size_display(self._models))
        )
        if hasattr(self, "_start_model_combo"):
            self._populate_start_combo()
        if hasattr(self, "_adv_summary"):
            self._repopulate_advanced_picker()
            self._update_advanced_header()

    @staticmethod
    def _style_status_item(item, status):
        """Use color as a secondary status cue while preserving explicit text."""
        if status == "Running":
            item.setForeground(QtGui.QColor("#6ee7a8"))
        elif status == "Starting":
            item.setForeground(QtGui.QColor("#ffd166"))
        elif status in ("Turned off on your list", "No file yet", "Model file is missing"):
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
        keys = (
            "name",
            "capabilities_display",
            None,
            "size",
            "vram",
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
        if model.get("source") == "store-discovered":
            return "Discovered - add to use"
        if _model_turned_off(model):
            return "Turned off on your list"
        if not model.get("location"):
            return "No file yet"
        if not _model_file_present(model):
            return "Model file is missing"
        if self._ui.running_model_id == model["id"]:
            return "Running"
        if self._ui.in_flight and self._selected_id == model["id"]:
            return "Starting"
        return "Stopped"

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
        self._note_selected_model()
        settings_index = self._settings_model.findData(self._selected_id)
        if settings_index >= 0 and self._settings_model.currentIndex() != settings_index:
            self._settings_model.blockSignals(True)
            self._settings_model.setCurrentIndex(settings_index)
            self._settings_model.blockSignals(False)
        self._load_settings_model()
        self._sync_start_combo()
        self._refresh_buttons()

    def _sync_start_combo(self):
        """Show the selected model in the Start page's picker."""
        combo = getattr(self, "_start_model_combo", None)
        if combo is None:
            return
        index = combo.findData(self._selected_id)
        if index >= 0 and combo.currentIndex() != index:
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self._update_start_fit()

    def _note_selected_model(self):
        """Tell the controller which model is picked so Talk can start it."""
        notify = getattr(self._gc, "note_selected_model", None)
        if callable(notify):
            notify(self._selected_id)

    def _on_settings_model_changed(self, _index):
        """Make Settings selection authoritative and mirror it to the model table."""
        model = self._settings_selected_model()
        if model is None:
            return
        self._selected_id = model["id"]
        self._note_selected_model()
        self._restore_model_selection()
        self._load_settings_model()
        self._refresh_buttons()

    def _advanced_summary_text(self, model):
        """One plain line: size, context, measured speed and GPU fit."""
        if model is None:
            return "No model picked."
        parts = [str(model.get("size_display") or "-")]
        context = model.get("context_size")
        if context:
            parts.append(f"{int(context):,} tokens of context")
        speed = plain_speed(model.get("score_display", "-"))
        parts.append(
            speed.lower() if speed and speed not in ("-", "Not measured yet") else "speed not measured yet"
        )
        fit = gui_controller.format_fit(model.get("vram_need_mb"), self._gpu_usable_mb())
        if fit == "Yes":
            parts.append("fits your GPU")
        elif fit == "Partly":
            parts.append("too big for your GPU alone (slower)")
        if model.get("source") == "discovered":
            parts.append("fine-tuned, not on your list yet")
        elif model.get("source") == "store-discovered":
            parts.append("discovered in the Models folder, not on your list yet")
        return "  |  ".join(parts)

    def _update_advanced_header(self):
        """Refresh the summary line and the buttons that name the target model."""
        model = self._settings_selected_model()
        self._adv_summary.setText(self._advanced_summary_text(model))
        name = model["name"] if model else ""
        self._edit_title.setText(f"Edit {name}" if name else "Edit this model")
        self._delete_btn.setText(f"Delete {name}..." if name else "Delete this model...")
        self._register_btn.setVisible(
            bool(model and model.get("source") in ("discovered", "store-discovered"))
        )
        self._register_btn.setText(
            "Add discovered model" if model and model.get("source") == "store-discovered"
            else "Add to my models"
        )

    def _repopulate_advanced_picker(self):
        """Rebuild the shared picker from the current model rows."""
        combo = self._settings_model
        combo.blockSignals(True)
        combo.clear()
        for model in self._models:
            combo.addItem(model["name"], model["id"])
        index = combo.findData(self._selected_id)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _load_settings_model(self):
        """Prefill editors from persisted values without inventing dirty state."""
        model = self._settings_selected_model()
        self._update_advanced_header()
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
        editable = model.get("source", "registry") == "registry"
        for widget in (
            self._gpu_edit,
            self._ctx_edit,
            self._id_edit,
            self._name_edit,
            self._capabilities_edit,
        ):
            widget.setEnabled(editable)
        if not editable:
            reason = "Add this discovered model to your list before editing it."
            self._gate(self._save_btn, False, reason)
            self._gate(self._identity_save_btn, False, reason)
            self._gate(self._capabilities_save_btn, False, reason)

    def _autotune_active(self):
        """True from the Auto-tune click until its result lands.

        Two sources because they cover different halves of the run: the local
        flag covers the queued-but-not-yet-started window, the controller's own
        flag covers the run itself (and survives a page refresh, which the local
        flag would too, but the controller is the authority once it has started).
        """
        return bool(self._autotune_pending or self._gc.autotune_in_progress())

    def _refresh_buttons(self):
        """Apply controller-defined lifecycle states across every page.

        Every disabled control gets a hover explanation (see _gate): a greyed
        button with no reason reads as broken.
        """
        states = gui_controller.compute_button_states(self._ui)
        model = self._selected_model()
        running = self._ui.running_model_id
        autotune = self._autotune_active()

        start_label = "Start"
        start_reason = ""
        start_enabled = bool(states["start_enabled"])
        if model is None:
            start_enabled = False
            start_reason = "Pick a model first."
        elif model.get("source") == "store-discovered":
            start_enabled = False
            start_reason = "Add this discovered model to your list before starting it."
        elif _model_turned_off(model):
            start_enabled = False
            start_reason = "Turned off on your list"
        elif not model.get("location"):
            start_enabled = False
            start_reason = "This model has no file yet. Download it on the Models page."
        elif not _model_file_present(model):
            start_enabled = False
            start_reason = f"The model file is missing: {model['location']}"
        else:
            start_label = gui_controller.start_label_for(model["id"], self._ui)
            if start_label == "Running":
                start_enabled = False
                start_reason = "This model is already running."
        if not states["start_enabled"] and not start_reason:
            start_reason = "Working - please wait a moment."
        self._start_btn.setText(start_label)
        stop_enabled = bool(states["stop_enabled"])
        stop_reason = (
            "Working - please wait a moment." if self._ui.in_flight
            else "No model is running."
        )
        offload_enabled = True
        offload_reason = ""
        probe = getattr(self._gc, "chat_needs_no_model", None)
        no_model_chat = bool(callable(probe) and probe())
        chat_enabled = bool(
            states["chat_enabled"] or self._assistant_port
            or (no_model_chat and not self._ui.in_flight)
        )
        chat_reason = (
            "Working - please wait a moment." if self._ui.in_flight
            else "Start a model first - chat needs a running model here."
        )
        if not self._assistant_live:
            self._gate(
                self._talk_btn,
                bool(model and model.get("launchable")),
                "Pick a model on the Start page first.",
            )
        if self._assistant_live:
            # A live voice chat owns its model; end it before changing the model.
            live = "A voice chat is using the model. End it on Voice & Vision first."
            start_enabled, start_reason = False, live
            stop_enabled, stop_reason = False, live
            offload_enabled, offload_reason = False, live
        if autotune:
            # Stop is the only way out of a multi-minute speed test, so it is
            # enabled LAST, after every rule that would switch it off.
            busy = "A speed test is running. Press Stop to cancel it."
            stop_enabled = not self._autotune_canceling
            stop_reason = "Cancelling the auto-tune..."
            start_enabled, start_reason = False, busy
            offload_enabled, offload_reason = False, busy
            chat_enabled, chat_reason = False, "Wait for the auto-tune to finish or stop it."
        self._gate(self._start_btn, start_enabled, start_reason)
        self._gate(self._stop_btn, stop_enabled, stop_reason)
        if autotune and stop_enabled:
            self._stop_btn.setToolTip("Stop the auto-tune")
        self._gate(self._header_offload_btn, offload_enabled, offload_reason)
        self._chat_no_model_ok = no_model_chat
        self._gate(self._chat_btn, chat_enabled, chat_reason)
        self._update_chat_hint()
        self._refresh_model_exclusive_buttons()
        self._header_status.setText(self._health_line())
        self._header_status.setToolTip(self._header_status.text())
        self._update_vv_banner()

    def _refresh_model_exclusive_buttons(self):
        """Protect assistant-held models from Vision and benchmark switching."""
        model = self._selected_model()
        busy_voice = "A voice chat is using the model. End it on Voice & Vision first."
        self._gate(
            self._register_btn,
            model is not None and model.get("source") in ("discovered", "store-discovered"),
            "Only a discovered model that is not on your list yet can be added.",
        )
        store_discovered = bool(model and model.get("source") == "store-discovered")
        self._mine_register_btn.setVisible(store_discovered)
        self._gate(
            self._mine_register_btn,
            store_discovered,
            "Select a discovered GGUF from the Models folder first.",
        )
        # Delete: a registered row (list entry and file) or a GGUF found in the
        # Models folder (file only), and never the running model, whose file
        # is open. A discovered fine-tune is deleted from the Fine-tune page.
        source = model.get("source", "registry") if model else None
        if model is None:
            delete_reason = "Pick a model first."
        elif source not in ("registry", "store-discovered"):
            delete_reason = "Delete a fine-tune run from the Fine-tune page."
        else:
            delete_reason = "Stop the model before deleting it."
        self._gate(
            self._delete_btn,
            source in ("registry", "store-discovered")
            and self._ui.running_model_id != model.get("id"),
            delete_reason,
        )
        self._gate(self._mine_delete_btn, self._delete_btn.isEnabled(), delete_reason)
        if self._assistant_live:
            tune_reason = busy_voice
        elif self._autotune_active():
            tune_reason = "A speed test is already running."
        elif self._ui.running_model_id is not None:
            tune_reason = "Stop the running model first - the test needs the GPU to itself."
        elif model is None:
            tune_reason = "Pick a model first."
        elif model.get("source", "registry") != "registry":
            tune_reason = "This discovered model is not on your list yet. Add it first."
        elif _model_turned_off(model):
            tune_reason = (
                f"This model is turned off on your list (status: {model.get('status')}), "
                "so it cannot be started or tuned."
            )
        else:
            tune_reason = (
                f"This model's file is missing ({model.get('location') or 'no file set'}). "
                "Point it at the right file or remove it from your list."
            )
        self._gate(
            self._autotune_btn,
            not self._assistant_live
            and model is not None
            and _model_tunable(model)
            and self._ui.running_model_id is None
            and not self._autotune_active(),
            tune_reason,
        )
        hint = getattr(self, "_autotune_hint", None)
        if hint is not None:
            # A running tune shows its own progress; every other reason is shown.
            text = "" if self._autotune_btn.isEnabled() or self._autotune_active() else tune_reason
            hint.setText(text)
            hint.setVisible(bool(text))
        self._gate(
            self._autotune_all_btn,
            not self._assistant_live
            and self._ui.running_model_id is None
            and not self._autotune_active()
            and any(_model_tunable(m) for m in self._models),
            busy_voice if self._assistant_live else (
                "A speed test is already running." if self._autotune_active()
                else "Stop the running model first - the test needs the GPU to itself."
                if self._ui.running_model_id is not None
                else "You have no models to test yet."
            ),
        )
        self._gate(
            self._vision_btn,
            not self._assistant_live and bool(self._vision_path.text()),
            busy_voice if self._assistant_live else "Pick or drop an image first.",
        )

    def _selected_context_size(self):
        """Return context size for the model represented by the live metrics."""
        target = self._ui.running_model_id or self._selected_id
        for model in self._models:
            if model["id"] == target:
                return model.get("context_size")
        return None

    def _health_line(self):
        """The running model, port, speed and context beside the health chip."""
        if self._autotune_active():
            model_id = self._gc.autotune_model_id() or self._selected_id or "model"
            return f"Auto-tuning {self._model_name(model_id)}..."
        running = self._ui.running_model_id
        if not running and not self._assistant_port:
            return "No model running"
        name = self._model_name(running) if running else "Voice chat model"
        visible_port = self._ui.running_port or self._assistant_port
        port_text = f" - port {visible_port}" if visible_port else ""
        sample = self._ui.latest_metrics
        rate = "-"
        context = "-"
        if sample is not None and sample.metrics_available:
            rate = (
                "idle"
                if not sample.gen_tokens_s
                else f"{sample.gen_tokens_s:.1f} tokens per second"
            )
            context = gui_controller._format_ctx(sample, self._selected_context_size())
        return f"{name} running{port_text} - {rate} - ctx {context}"

    def _on_start(self):
        """Queue a start or switch and immediately render an honest working state."""
        model = self._selected_model()
        if model is None:
            self._model_status.setText("Pick a model first.")
            return
        if _model_turned_off(model):
            self._model_status.setText("Turned off on your list")
            return
        if not model.get("location"):
            self._model_status.setText("That model has no file yet. Download it on the Models page.")
            return
        if not _model_file_present(model):
            self._model_status.setText(f"The model file is missing: {model['location']}")
            return
        picked, problems = self._selected_thinking(model["id"])
        if problems:
            # Refuse rather than start at a level the person did not choose.
            self._model_status.setText(problems[0])
            return
        self._ui.in_flight = True
        detail = f" (thinking: {picked})" if picked else ""
        self._model_status.setText(
            f"Starting {model['name']}{detail}... the first start can take a minute."
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

        text = str(self._thinking_combo.currentData() or "").strip()
        if not text:
            return None, []
        return parse_reasoning_choice(text, model_id)

    def _on_offload_gpu(self):
        """Free up GPU (M17.8): free the graphics memory locitize is holding.

        While a speed test owns the GPU, this defers to the same cancel path Stop
        uses - killing the test's server out from under it would leave a
        half-measured setting, so it is asked to stop cleanly instead.
        Otherwise the controller stops the running model AND clears any of
        locitize's own leftover servers, then reports what it freed and which
        other apps still hold the card.
        """
        if self._gc.autotune_in_progress():
            self._on_stop()
            return
        self._model_status.setText("Freeing up the GPU...")
        self._gc.request_free_gpu()

    def _apply_gpu_free(self, result):
        """Render the outcome of a Free up GPU sweep."""
        self._ui.in_flight = False
        payload = result.payload or {}
        self._apply_running_snapshot(payload)
        self._assistant_port = None
        self._ui.latest_metrics = None
        self._render_monitor_text("idle")
        freed = int(payload.get("freed", 0))
        stopped = payload.get("stopped_model")
        others = sorted({o for o in (payload.get("others") or []) if not str(o).startswith("[")})
        vram = payload.get("vram_free_mb")
        parts = []
        if stopped:
            parts.append(f"Stopped {self._model_name(stopped)}")
        if freed:
            parts.append(f"Cleared {freed} leftover locitize helper(s)")
        if not parts:
            parts.append("Nothing from locitize was using the GPU")
        if vram is not None:
            parts.append(f"{vram / 1000:.1f} GB of GPU memory is free")
        if others:
            shown = ", ".join(others[:4]) + (" and more" if len(others) > 4 else "")
            parts.append(f"other apps are still using it: {shown}")
        message = "; ".join(parts) + "."
        self._model_status.setText(message)
        # The line clips rather than resizing the window; hover shows all of it.
        self._model_status.setToolTip(message)
        self._refresh_model_rows()
        self._refresh_buttons()

    def _on_stop(self):
        """Stop: cancel a running speed test if there is one, else stop the model.

        The speed-test branch comes first and does NOT go through the command
        queue: a test holds the single ops worker for several minutes, so a
        queued stop would sit unread until the test it was meant to interrupt had
        finished. cancel_autotune() sets an event the test polls directly.

        There is no ambiguity between the two meanings: a speed test only runs
        when no model is running (request_autotune_context refuses otherwise),
        so at most one of these branches is ever applicable.
        """
        if self._gc.autotune_in_progress():
            model_id = self._gc.autotune_model_id() or ""
            self._gc.cancel_autotune()
            self._autotune_canceling = True
            self._autotune_status.setVisible(True)
            self._autotune_status.setText(
                f"Cancelling the speed test of {model_id}: finishing the step "
                f"that is running now..."
            )
            self._model_status.setText("Cancelling the speed test...")
            self._refresh_buttons()
            return
        self._ui.in_flight = True
        self._model_status.setText("Stopping...")
        self._refresh_buttons()
        self._gc.request_stop()

    def _offer_release_and(self, then, model_id="", reasoning=None):
        """Offer "Release and stop" when a coding terminal still holds the model.

        Stop and Free up GPU refuse while a terminal launched from Code is using
        the model; this is the one place that explains why and lets the person
        continue, instead of sending them to another page to find Release.
        """
        action = {
            "free_gpu": "free the GPU",
            "start": "switch models",
            "release": "continue",
        }.get(then, "stop the model")
        confirmed = QtWidgets.QMessageBox.question(
            self,
            "A coding terminal is using this model",
            "You opened a coding tool (Codex or OpenCode) with this "
            f"model, so locitize kept it loaded. Release it and {action}? "
            "Terminals that are still open will lose their model.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirmed != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        self._ui.in_flight = then != "release"
        self._model_status.setText("Releasing the model...")
        self._refresh_buttons()
        if then == "start":
            self._gc.request_release_and(then, model_id, reasoning)
        else:
            self._gc.request_release_and(then)
        return True

    def _on_chat(self):
        """Ask the controller to resolve and open the configured chat surface."""
        self._model_status.setText("Opening the chat...")
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
            self._model_status.setText("Pick a model to auto-tune first.")
            return
        if not self._autotune_btn.isEnabled():
            # Same belt-and-braces guard Delete uses: a keyboard activation can
            # reach a control the enablement rule has just turned off.
            return
        confirmed = QtWidgets.QMessageBox.question(
            self,
            "Auto-tune",
            f"Auto-tune '{model['name']}'?\n\n"
            f"locitize will start the model several times with different "
            f"amounts of working memory to find the biggest amount this computer "
            f"can really hold, then measure how fast it answers (tokens per "
            f"second). This takes a few minutes and uses the GPU the whole time. "
            f"The best setting is saved for this model only.",
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
            f"Auto-tuning {model['name']}: starting... (press Stop to cancel)"
        )
        self._model_status.setText(f"Auto-tuning {model['name']}...")
        self._gc.request_autotune_context(model["id"])
        # Immediately, so Stop becomes live in the same event as the click rather
        # than at whenever the next refresh happens to be.
        self._refresh_buttons()

    def _on_autotune_all(self):
        """Confirm, then queue an auto-tune of every registered model."""
        if not self._autotune_all_btn.isEnabled():
            return
        # Same set the controller tunes: switched-on rows (launchable), and of
        # those only the ones whose file is on disk.
        registered = [
            m for m in self._models
            if m.get("source", "registry") == "registry" and not _model_turned_off(m)
        ]
        count = sum(1 for m in registered if _model_file_present(m))
        missing = len(registered) - count
        turned_off = sum(
            1 for m in self._models
            if m.get("source", "registry") == "registry" and _model_turned_off(m)
        )
        missing_note = (
            f"{missing} model(s) on your list are skipped because their file is "
            f"missing.\n\n" if missing else ""
        )
        if turned_off:
            missing_note += (
                f"{turned_off} model(s) turned off on your list are skipped.\n\n"
            )
        confirmed = QtWidgets.QMessageBox.question(
            self,
            "Auto-tune all models",
            f"Auto-tune all {count} models?\n\n"
            f"{missing_note}"
            f"Each model is started several times to find the biggest working "
            f"memory this computer can hold, then its speed is measured. That "
            f"takes a few minutes per model - roughly "
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
            f"Auto-tuning all {count} models: starting... (press Stop to cancel)"
        )
        self._model_status.setText(f"Auto-tuning all {count} models...")
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
            message = result.error or "it did not finish"
            self._autotune_status.setText(f"Auto-tune all: {message}")
            self._model_status.setText(f"Auto-tune all: {message}")
            return
        p = result.payload
        text = (
            f"Auto-tune all finished: {p.get('tuned', 0)} of {p.get('total', 0)} "
            f"models tuned"
        )
        if p.get("failed"):
            text += f", {p['failed']} failed"
        if p.get("canceled"):
            text += f", canceled ({p.get('skipped', 0)} not tuned)"
        if p.get("missing"):
            text += f"; {p['missing']} skipped because their file is missing"
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
            f"Auto-tuning {result.payload.get('model_id', '')}: {line} "
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
                f"Auto-tune of {model_id} cancelled. Its settings were left "
                f"unchanged."
            )
            self._model_status.setText("Auto-tune cancelled.")
            return
        if not result.ok:
            message = result.error or "it did not finish"
            self._autotune_status.setText(f"Auto-tune of {model_id} failed: {message}")
            self._model_status.setText(f"Auto-tune failed: {message}")
            return
        native = result.payload.get("native_context")
        chosen = result.payload.get("chosen_context")
        previous = result.payload.get("previous_context")
        failure = result.payload.get("first_failure")
        trials = result.payload.get("trials") or []
        stretch = (
            f"stretched {result.payload.get('rope_scale')}x past what the model "
            "was trained on"
            if result.payload.get("yarn_applied")
            else "within what the model was trained on"
        )
        ceiling = (
            f"biggest that loaded: {chosen}"
            + (f", smallest that failed: {failure}" if failure else "")
        )
        speed = result.payload.get("tokens_per_second")
        if speed is not None:
            measured = f" Speed: {float(speed):.1f} tokens per second."
            headline = f"working memory {chosen}, {float(speed):.1f} tokens per second"
        else:
            reason = result.payload.get("benchmark_error") or "not measured"
            measured = f" Speed was not measured ({reason})."
            headline = f"working memory now {chosen}"
        self._autotune_status.setText(
            f"Auto-tune of {model_id} finished. The model was trained for {native}. "
            f"{ceiling} ({stretch}). Working memory {previous} -> {chosen}, saved "
            f"after {len(trials)} test starts.{measured} It applies the next time "
            f"this model starts."
        )
        self._model_status.setText(f"Auto-tune finished: {headline}")
        # The row's cached context_size is now stale; re-read the registry so the
        # page shows what is really on disk rather than what it showed before.
        self._on_refresh_models()

    def _on_start_assistant(self):
        """Queue assistant startup with the chosen voice and output preference."""
        self._assistant_status.setText("Starting...")
        self._assistant_start_btn.setEnabled(False)
        self._gate(self._talk_btn, False, 'Press Start voice chat first.')
        self._gc.request_start_assistant(
            selected_voice_id(self._talk_voice), self._assistant_speak.isChecked()
        )

    def _on_end_assistant(self):
        """Queue session teardown and lock controls until the outcome arrives."""
        self._assistant_status.setText("ending session...")
        self._gate(self._assistant_end_btn, False, 'Voice chat is not running.')
        self._gate(self._talk_btn, False, 'Press Start voice chat first.')
        self._gate(self._interrupt_btn, False, 'Nothing is being spoken.')
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
        self._whisper_status.setText("working...")
        self._gc.request_whisper_toggle()

    def _on_listen(self):
        """Queue the existing fifteen-second microphone capture path."""
        self._voice_status.setText("Listening for 15 seconds - speak now")
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
            self._gate(self._save_btn, False, 'Nothing to save yet - change a value first, or fix the problem shown.')
            self._edit_error.clear()
            return
        state = gui_controller.compute_save_state(
            self._gpu_edit.text(),
            self._ctx_edit.text(),
            model["gpu_layers"],
            model["context_size"],
        )
        self._gate(self._save_btn, bool(state["enabled"]), 'Nothing to save yet - change a value first, or fix the problem shown.')
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
            self._gate(self._identity_save_btn, False, 'Nothing to rename yet - change the id or name first, or fix the problem shown.')
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
        self._gate(self._identity_save_btn, bool(state["enabled"]), 'Nothing to rename yet - change the id or name first, or fix the problem shown.')
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
            self._gate(self._capabilities_save_btn, False, 'Nothing to save yet - change the list first.')
            self._capabilities_error.clear()
            return
        typed = gui_controller.parse_capabilities_text(self._capabilities_edit.text())
        self._gate(self._capabilities_save_btn, typed != list(model.get("capabilities", [])), 'Nothing to save yet - change the list first.')
        self._capabilities_error.clear()

    def _on_save_capabilities(self):
        """Queue the selected model's capabilities write through the controller."""
        model = self._settings_selected_model()
        if model is not None:
            self._gc.save_model_capabilities(model["id"], self._capabilities_edit.text())

    # ---- M13 Fine-tune page ---------------------------------------------- #

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
        if model is None:
            return
        if model.get("source") == "store-discovered":
            self._mine_register_btn.setEnabled(False)
            self._register_btn.setEnabled(False)
            self._model_status.setText(f"Adding {model['name']} to your models...")
            self._gc.request_add_model_file(model["location"])
            return
        if model.get("source") != "discovered":
            return
        self._register_btn.setEnabled(False)
        self._model_status.setText(f"Adding {model['name']} to your models...")
        self._gc.request_register_discovered(model["id"])

    def _on_add_model_file(self):
        """Choose one local GGUF; registration itself runs on the ops worker."""
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Add model file",
            str(getattr(self._gc, "model_store_path", lambda: "")() or ""),
            "GGUF models (*.gguf)",
        )
        if not path:
            return
        self._mine_add_file_btn.setEnabled(False)
        self._model_status.setText(f"Adding {Path(path).name} to your models...")
        self._gc.request_add_model_file(path)

    def _on_model_table_menu(self, pos):
        """Right-click on a model row: Use it, or Delete it (asks first)."""
        row = self._model_table.rowAt(pos.y())
        if row < 0:
            return
        self._model_table.selectRow(row)
        menu = QtWidgets.QMenu(self._model_table)
        use = menu.addAction("Use this model")
        delete = menu.addAction("Delete...")
        delete.setEnabled(self._delete_btn.isEnabled())
        chosen = menu.exec(self._model_table.viewport().mapToGlobal(pos))
        if chosen is use:
            self._go_to("Start")
        elif chosen is delete:
            self._on_delete_model()

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
        if model.get("source") == "store-discovered":
            what = "It is not on your list. This deletes this file from disk"
        else:
            what = "This removes it from your list AND deletes these files from disk"
        confirmed = QtWidgets.QMessageBox.warning(
            self,
            "Delete model",
            f'Permanently delete "{model["name"]}"?\n\n'
            f"{what}:\n{file_list}\n\n"
            f"This cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if confirmed != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._delete_btn.setEnabled(False)
        self._mine_delete_btn.setEnabled(False)
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
        self._gate(self._hub_search_btn, False, 'Searching is not available right now.')
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
        self._gate(self._hub_load_more_btn, False, 'There are no more results.')
        self._hub_load_more_btn.setText("Loading more...")
        self._gc.request_hub_search_more()

    def _on_hub_repo_selected(self, row=-1, *_args):
        """Repo selection: the SECOND egress control - it lists that repo's files."""
        entry = self._selected_hub_repo()
        if entry is None or not self._hub_enabled:
            return
        self._hub_file_table.setRowCount(0)
        self._gate(self._hub_download_btn, False, HUB_DOWNLOAD_REASON)
        self._hub_status.setText(f"Listing files in {entry['repo_id']}...")
        self._gc.request_hub_files(entry["repo_id"])

    def _on_hub_file_selected(self, row=-1, *_args):
        """Show the fit arithmetic for the selected file. Pure local computation."""
        entry = self._selected_hub_file()
        if entry is None:
            self._gate(self._hub_download_btn, False, HUB_DOWNLOAD_REASON)
            self._hub_fit.setText("")
            return
        # DEFECT-QA-M14-2: browsing the file list during a download must not
        # re-arm Download. One transfer at a time is the controller's rule, and
        # a button that looks pressable but answers "A download is already
        # running." is a dead control by any other name. A request that is sent
        # but not yet confirmed counts as busy too (NEW-QA-M14-7), so the gap
        # between pressing Download and the first progress line is not a window
        # in which the button comes back to life.
        self._gate(self._hub_download_btn, self._hub_busy_job() is None, HUB_DOWNLOAD_REASON)
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
        self._gate(self._hub_download_btn, False, HUB_DOWNLOAD_REASON)
        self._gate(self._hub_cancel_btn, True, 'No download is running.')
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
        self._gate(self._hub_cancel_btn, False, 'No download is running.')
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
        self._gate(self._hub_search_btn, self._hub_enabled, 'Searching is not available right now.')
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
            self._gate(self._hub_load_more_btn, self._hub_has_more, 'There are no more results.')
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
        self._gate(self._hub_load_more_btn, self._hub_has_more and self._hub_enabled, 'There are no more results.')
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
        self._gate(self._hub_cancel_btn, False, 'No download is running.')
        self._hub_progress.setVisible(False)
        self._gate(self._hub_download_btn, self._selected_hub_file() is not None, HUB_DOWNLOAD_REASON)
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
            parts.append(f"Added to your models as '{payload.get('model_id')}'.")
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

    def _on_finetune_save_copy(self):
        """Open the selected run in the guided flow's Save step."""
        row = self._selected_finetune()
        if row is None:
            return
        self._ft_flow.open_run(Path(row["path"]).parent)

    def _on_finetune_flow_saved(self):
        """A model was added to the list from the guided flow: refresh both lists."""
        if hasattr(self._gc, "request_scan_finetunes"):
            self._on_finetune_rescan()
        refresh = getattr(self, "_on_refresh_models", None)
        if refresh is not None:
            refresh()

    def _refresh_finetune_buttons(self):
        """Enable the row actions only when a usable row is selected."""
        row = self._selected_finetune()
        has_row = row is not None
        self._ft_serve_btn.setEnabled(has_row)
        self._ft_register_btn.setEnabled(has_row and not row.get("registered"))
        self._ft_delete_btn.setEnabled(has_row)
        self._ft_folder_btn.setEnabled(has_row)
        self._ft_save_btn.setEnabled(has_row)

    def _apply_finetune_state(self, payload):
        """Show the install panel, with the reason, when fine-tuning cannot run here."""
        disabled = payload.get("status") == "disabled"
        self._ft_install_panel.setVisible(disabled)
        reason = payload.get("reason", "") if disabled else ""
        self._ft_install_reason.setText(
            "" if reason.startswith("Fine-tuning is not installed yet.") else reason
        )

    def _apply_finetune_result(self, result):
        """Apply a finetune_state Result from the worker."""
        if result.ok:
            self._apply_finetune_state(result.payload)
        elif result.error:
            self._ft_status.setText(result.error)

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
            self._gate(self._register_btn, True, '')
            message = result.error or "Could not add it to your models."
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
            self._model_status.setText(self._sessions_page.status.text())
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
                "" if result.ok else (result.error or "The model did not start.")
            )
            self._refresh_model_rows()
            self._refresh_buttons()
        elif kind in ("stop", "free_gpu") and (result.payload or {}).get("reserved"):
            self._ui.in_flight = False
            self._model_status.setText(result.error or "")
            self._refresh_buttons()
            self._offer_release_and(result.payload.get("then", "stop"))
        elif kind == "free_gpu":
            self._ui.in_flight = False
            self._model_status.setText(result.error or "could not free the GPU")
            self._refresh_buttons()
        elif kind == "start" and not result.ok and (result.payload or {}).get("reserved"):
            self._ui.in_flight = False
            self._apply_running_snapshot(result.payload)
            self._model_status.setText(result.error or "")
            self._refresh_model_rows()
            self._refresh_buttons()
            self._offer_release_and(
                "start", result.payload.get("model_id", ""), result.payload.get("reasoning")
            )
        elif kind == "stop":
            self._ui.in_flight = False
            self._apply_running_snapshot(result.payload)
            self._assistant_port = None
            self._ui.latest_metrics = None
            self._render_monitor_text("idle")
            self._model_status.setText(
                "Stopped." if result.ok else (result.error or "Could not stop the model.")
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
                "Turn off voice input"
                if self._ui.whisper_running
                else "Turn on voice input"
            )
            self._whisper_status.setText(
                result.error
                or ("Voice input is on" if self._ui.whisper_running else "Voice input is off")
            )
        elif kind == "transcript":
            line = result.payload.get("line", "")
            if line:
                self._voice_status.setText(line)
                append_plain(self._transcript, line)
        elif kind == "listen":
            self._voice_status.setText(
                "Done listening" if result.ok else (result.error or "Listening failed")
            )
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
        elif kind == "model_store_change":
            self._apply_model_store_change(result)
        elif kind == "add_model_file_result":
            self._mine_add_file_btn.setEnabled(True)
            if result.ok:
                self._model_status.setText(
                    "That model is already on your list."
                    if result.payload.get("already_registered")
                    else "Model added to your list."
                )
                self._on_refresh_models()
            else:
                self._model_status.setText(result.error or "Could not add the model file.")
                self._refresh_buttons()
        elif kind == "openwebui_install_progress":
            self._apply_openwebui_install_progress(result)
        elif kind == "openwebui_install_done":
            self._apply_openwebui_install_done(result)
        elif kind == "chat":
            self._apply_chat_result(result)
        elif kind == "chat_status":
            message = result.payload.get("message", "")
            self._model_status.setText(message)
        elif kind == "chat_ask":
            self._show_chat_chooser()
        elif kind == "chat_offer_start":
            self._show_openwebui_start_offer(result.payload.get("reason", ""))
        elif kind == "chat_offer_install":
            self._show_openwebui_install_offer(
                result.payload.get("reason", ""),
                bool(result.payload.get("simple_available")),
            )
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
        elif kind == "describe" and not result.ok and "coding terminal is still using" in (result.error or ""):
            self._vision_result.setPlainText(
                "A coding terminal is still using the model, so it cannot be "
                "swapped for the picture model yet."
            )
            self._refresh_model_exclusive_buttons()
            if self._offer_release_and("release"):
                self._on_describe()
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
            self._gate(
                self._second_eye_start_btn, not running, "Already watching your screen."
            )
            self._gate(self._second_eye_stop_btn, running, "Not watching right now.")
            if not result.ok:
                self._second_eye_status.setText(result.error or "Could not start watching.")
            elif running:
                goal = result.payload.get("goal", "")
                self._second_eye_status.setText(
                    f"Watching: {goal}" if goal else "Watching your screen"
                )
            else:
                self._second_eye_status.setText("Not watching")
        elif kind == "benchmark":
            self._apply_benchmark_result(result)

    def _apply_running_snapshot(self, payload):
        """Trust the controller's post-operation snapshot instead of guessing."""
        self._ui.running_model_id = payload.get("running_model_id")
        self._ui.running_port = payload.get("running_port")
        if hasattr(self, "_ft_flow"):
            self._ft_flow._refresh_doc_qa_gate(bool(self._ui.running_model_id))
        self._update_chat_hint()

    def _update_chat_hint(self):
        """Keep the Start page's status line and chat hint honest about the model."""
        if not hasattr(self, "_chat_hint"):
            return
        running = self._ui.running_model_id
        if running:
            name = self._model_name(running)
            self._chat_hint.setText(
                "Opens in your browser with the rich chat when it is installed, "
                "or the built-in chat page."
            )
            self._start_status_line.setText(f"{name} is running - ready to chat")
        else:
            if self._autotune_active():
                model_id = self._gc.autotune_model_id() or self._selected_id or "model"
                self._chat_hint.setText("Chat is unavailable while auto-tuning uses the model.")
                self._start_status_line.setText(
                    f"Auto-tuning {self._model_name(model_id)}..."
                )
                return
            if getattr(self, "_chat_no_model_ok", False):
                self._chat_hint.setText(
                    "Open the chat and pick any model there - it loads "
                    "automatically. Or start one here first."
                )
            else:
                self._chat_hint.setText("Start a model above first, then open the chat.")
            self._start_status_line.setText(
                "Starting..." if self._ui.in_flight else "No model running"
            )

    def _on_sidebar_changed(self, index):
        """Switch pages and run first-visit actions (session scan)."""
        self._stack.setCurrentIndex(index)
        name = PAGE_NAMES[index] if 0 <= index < len(PAGE_NAMES) else ""
        if name == "Code":
            self._sessions_page.visit()
        self._update_vv_banner()

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
            self._second_eye_status.setText("Watch my screen is not available here.")
            return
        self._second_eye_status.setText("Starting...")
        starter(self._second_eye_goal.text(), 1.0)

    def _on_second_eye_stop(self):
        """Ask the screen watcher to exit."""
        stopper = getattr(self._gc, "request_second_eye_stop", None)
        if callable(stopper):
            stopper()
            self._second_eye_status.setText("Stopping...")

    def _render_metrics(self, sample):
        """Render real metrics or an explicit unavailable message on all surfaces."""
        if sample is None:
            return
        self._ui.latest_metrics = sample
        if (
            not getattr(sample, "metrics_available", False)
            and getattr(sample, "slots", None) is None
        ):
            text = ""
        else:
            text = plain_monitor_line(
                gui_controller.format_monitor_line(sample, self._selected_context_size())
            )
        self._render_monitor_text(text)
        self._header_status.setText(self._health_line())
        self._header_status.setToolTip(self._header_status.text())

    def _render_monitor_text(self, text):
        """Keep the expanded Models and Talk monitor lines synchronized."""
        # The Models-tab chip stays hidden unless there are live numbers.
        self._models_monitor.setText("" if text == "idle" else text)
        self._talk_monitor.setText(text or "This model server does not report live speed.")

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
        self._update_advanced_header()

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
                "The simple built-in chat just opened. locitize also supports "
                "Open WebUI - chat history, uploads, web search - but it is "
                "not installed yet.\n\nInstall it from setup? (One tick box; "
                "about 2.6 GB. Chat keeps working either way.)"
            )
            offer.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
            )
            offer.button(QtWidgets.QMessageBox.StandardButton.Yes).setText("Open setup")
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
        """Ask which chat page to open when more than one is available.

        Coding tools are no longer offered here: they live on the Code page, so
        there is exactly one place to start one.
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Choose a chat")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Which chat would you like to open?"))
        built_in = QtWidgets.QRadioButton("Simple chat (built in, nothing to set up)")
        rich = QtWidgets.QRadioButton("Open WebUI (richer chat with history)")
        built_in.setChecked(True)
        layout.addWidget(built_in)
        layout.addWidget(rich)
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
        choice = "openwebui" if rich.isChecked() else "llamacpp"
        self._gc.open_chat(choice, remember.isChecked())

    def _apply_launch_harness_result(self, result):
        """Surface a coding-tool launch's outcome without hiding a failure."""
        if not result.ok:
            self._model_status.setText(result.error or "Could not open the coding tool.")
            return
        choice = result.payload.get("choice", "")
        note = result.payload.get("note", "")
        warning = result.payload.get("warning", "")
        message = f"{choice} opened in a new terminal."
        details = " ".join(part for part in (note, warning) if part)
        self._model_status.setText(f"{message} {details}".strip())

    def _refresh_chat_harness_buttons(self):
        """Back-compat name for the Code page's tool buttons."""
        self._refresh_code_harness_buttons()

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
            self._feature_state_note.setText("The uninstaller was not found. Reinstall locitize, then try again.")
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
            self._feature_state_note.setText(f"Could not start the uninstaller: {exc}")
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
                self._feature_state_note.setText(f"Could not open setup: {exc}")
            return
        if not bat.is_file():
            self._feature_state_note.setText("The setup launcher was not found. Reinstall locitize, then try again.")
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
            self._feature_state_note.setText(f"Could not open setup: {exc}")

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
            "The first start sets up its database and can take up to 5 minutes."
        )
        detail.setObjectName("muted")
        detail.setWordWrap(True)
        layout.addWidget(detail)
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
            self._model_status.setText(
                "Starting Open WebUI. The first start can take up to 5 minutes..."
            )
            self._gc.start_openwebui_and_open(remember.isChecked())

    def _refresh_openwebui_banner(self):
        state_reader = getattr(self._gc, "openwebui_install_state", None)
        state = state_reader() if callable(state_reader) else "installed"
        missing = state in ("missing", "partial")
        self._openwebui_banner.setVisible(missing)
        if missing:
            self._openwebui_banner_text.setText(
                "Open WebUI is not installed yet. Add richer chat with history and uploads."
                if state == "missing"
                else "Open WebUI setup is incomplete. Install it again to repair it."
            )

    def _on_openwebui_banner_install(self):
        simple_available = bool(getattr(self._gc, "_active_port", None))
        self._show_openwebui_install_offer("", simple_available)

    def _show_openwebui_install_offer(self, reason, simple_available):
        """Collect explicit consent before the external pip installer can run."""
        import setup_plan

        estimate = setup_plan.format_size(
            int(setup_plan.REQUIREMENTS["webui_pip"].size_mb or 0)
        )
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Install Open WebUI?")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        message = QtWidgets.QLabel(
            (reason + "\n\n" if reason else "")
            + "This installs the pinned Open WebUI package in its own environment. "
            + f"The estimated download is {estimate}."
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        if not simple_available:
            note = QtWidgets.QLabel(
                "Simple chat is unavailable until a model is running. You can install "
                "Open WebUI now, or cancel and start a model first."
            )
            note.setObjectName("muted")
            note.setWordWrap(True)
            layout.addWidget(note)
        buttons = QtWidgets.QHBoxLayout()
        install = QtWidgets.QPushButton("Install")
        install.setObjectName("primaryButton")
        simple = QtWidgets.QPushButton("Use the simple chat this time")
        simple.setEnabled(simple_available)
        cancel = QtWidgets.QPushButton("Cancel")
        choice = {"value": ""}

        def accept(value):
            choice["value"] = value
            dialog.accept()

        install.clicked.connect(lambda: accept("install"))
        simple.clicked.connect(lambda: accept("simple"))
        cancel.clicked.connect(dialog.reject)
        buttons.addWidget(install)
        buttons.addWidget(simple)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        if harden_text_rendering(dialog).exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        if choice["value"] == "simple":
            self._gc.open_chat("llamacpp", False)
            return
        self._openwebui_banner.setVisible(True)
        self._openwebui_banner_text.setText("Starting Open WebUI installation...")
        self._openwebui_install_btn.setEnabled(False)
        self._openwebui_cancel_btn.setVisible(True)
        self._gc.request_openwebui_install()

    def _on_openwebui_install_cancel(self):
        self._openwebui_banner_text.setText("Cancelling Open WebUI installation...")
        self._openwebui_cancel_btn.setEnabled(False)
        self._gc.cancel_openwebui_install()

    def _apply_openwebui_install_progress(self, result):
        message = result.payload.get("message", "Installing Open WebUI...")
        self._openwebui_banner.setVisible(True)
        self._openwebui_banner_text.setText(message)
        self._openwebui_install_btn.setEnabled(False)
        self._openwebui_cancel_btn.setEnabled(True)
        self._openwebui_cancel_btn.setVisible(True)

    def _apply_openwebui_install_done(self, result):
        self._openwebui_install_btn.setEnabled(True)
        self._openwebui_cancel_btn.setEnabled(True)
        self._openwebui_cancel_btn.setVisible(False)
        if result.ok:
            self._openwebui_banner.setVisible(False)
            self._model_status.setText(
                "Open WebUI is installed. Starting it now..."
            )
            return
        self._refresh_openwebui_banner()
        self._openwebui_banner_text.setText(
            result.error or "Open WebUI installation did not finish."
        )
        log_path = result.payload.get("log_path", "")
        if log_path:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Open WebUI was not installed")
            box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            box.setText(result.error or "Open WebUI installation did not finish.")
            open_logs = box.addButton("Open log folder", QtWidgets.QMessageBox.ButtonRole.ActionRole)
            box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
            harden_text_rendering(box).exec()
            if box.clickedButton() is open_logs:
                QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl.fromLocalFile(str(Path(log_path).parent))
                )

    def _on_assistant_started(self, payload):
        """Unlock session controls only after the controller reports readiness."""
        self._assistant_live = True
        self._assistant_port = payload.get("port")
        self._apply_running_snapshot(payload)
        self._assistant_status.setText("Ready to listen")
        self._gate(self._assistant_end_btn, True, 'Voice chat is not running.')
        self._gate(self._interrupt_btn, True, 'Nothing is being spoken.')
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
        self._gate(self._assistant_end_btn, False, 'Voice chat is not running.')
        self._gate(self._interrupt_btn, False, 'Nothing is being spoken.')
        self._gate(self._talk_btn, False, 'Press Start voice chat first.')
        # The assistant temporarily disables model lifecycle controls. Recompute
        # the full button set here so Start/Switch and Stop recover after failure.
        self._refresh_buttons()

    def _on_assistant_ended(self):
        """Reset every assistant control after a clean end result."""
        self._assistant_live = False
        self._assistant_status.setText("Not running. Click Talk to start.")
        self._assistant_start_btn.setEnabled(True)
        self._gate(self._assistant_end_btn, False, 'Voice chat is not running.')
        self._gate(self._interrupt_btn, False, 'Nothing is being spoken.')
        self._listen_when_ready = False
        self._talk_btn.setText("Talk")
        self._gate(self._talk_btn, False, 'Press Start voice chat first.')
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
        self._gate(
            self._talk_btn,
            self._assistant_live and self._talk_state in ("idle", "thinking", "speaking"),
            "Press Start voice chat first." if not self._assistant_live
            else "Wait for the reply to start, then you can cut in.",
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
            flow = getattr(self, "_ft_flow", None)
            if flow is not None:
                flow.shutdown()
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
        dialog.setWindowTitle("Add coding tools?")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(
            QtWidgets.QLabel(
                "locitize can open your local models inside these coding tools.\n"
                "The ones below aren't installed yet. Nothing is installed until\n"
                "you tick a box AND click OK - each one runs the exact command\n"
                "shown beneath it. You can also do this later in\n"
                "Settings > Coding tools."
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
            self._model_status.setText(f"Installing {labels.get(key, key)}...")
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


def run(controller, health="unknown", health_report=None):
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
    window = MainWindow(controller, health, health_report)
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
