"""Offscreen tests for the Models page's "Get models" section (M14.14.9).

These exercise the controls a user actually presses - Search, repo selection,
file selection, Download, Cancel - against a recording fake controller, so the
UI contract is tested rather than asserted in prose. They also carry one
egress-rule test (its name puts it in the AC-M14-18 keyword run): painting the
page must reach only the disk-backed catalog request, never the network.
"""

import os
import queue
import re

import gui_controller
import modelhub
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class HubFakeController:
    """The view-facing contract, plus the hub intents, recording every call."""

    def __init__(self, hub_enabled=True):
        self.result_q = queue.Queue()
        self.command_q = queue.Queue()
        self.calls = []
        self.shutdown_count = 0
        self._hub_enabled = hub_enabled

    # -- the pre-existing view contract the window needs to build ---------- #

    def list_models(self):
        return []

    def system_specs(self):
        return gui_controller.SystemSpecs(
            ram_total_mb=32768.0, gpu_name="Fake GPU", gpu_vram_total_mb=16303.0
        )

    def available_voices(self):
        return ["af_heart"]

    def start_threads(self):
        self.calls.append(("start_threads",))

    def shutdown(self):
        self.shutdown_count += 1

    # The Stop button consults these on every button refresh (they decide
    # whether Stop means "stop the model" or "cancel the auto-tune"), so the
    # window cannot even be built without them. No auto-tune in this suite.
    def autotune_in_progress(self):
        return False

    def autotune_model_id(self):
        return None

    def cancel_autotune(self):
        return False

    def __getattr__(self, name):
        """Record any other request_* intent instead of failing the build.

        The window binds many controls this suite does not exercise; recording
        them keeps this file about the Get models section without duplicating
        the whole controller surface.
        """
        if name.startswith(("request_", "save_", "open_", "start_")):
            def recorder(*args, **kwargs):
                self.calls.append((name,) + args)
            return recorder
        raise AttributeError(name)

    def hub_enabled(self):
        return self._hub_enabled


@pytest.fixture(scope="module")
def qapp():
    """Load Qt lazily so a machine without PySide6 skips cleanly."""
    pytest.importorskip("PySide6")
    import desktop
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app, desktop


def build(qapp, hub_enabled=True):
    """Build a window with a recording controller."""
    _app, desktop = qapp
    fake = HubFakeController(hub_enabled=hub_enabled)
    return desktop.MainWindow(fake, health="OK"), fake


def names(fake):
    """The recorded intent names, in order."""
    return [call[0] for call in fake.calls]


def test_modelhub_egress_ui_page_paint_requests_only_the_disk_catalog(qapp):
    """Painting the page reaches the catalog (disk) and nothing that can egress."""
    _window, fake = build(qapp)
    called = names(fake)
    assert "request_hub_catalog" in called
    # The three intents that can reach huggingface.co must not have fired.
    for forbidden in ("request_hub_search", "request_hub_files", "request_hub_download"):
        assert forbidden not in called


def test_modelhub_egress_ui_typing_never_searches(qapp):
    """Typing in the search box must not trigger a search on every keystroke."""
    window, fake = build(qapp)
    before = names(fake).count("request_hub_search")
    for text in ("q", "qw", "qwe", "qwen"):
        window._hub_query.setText(text)
    assert names(fake).count("request_hub_search") == before


def test_hub_ui_search_press_sends_the_query(qapp):
    """Search press sends exactly one search and disables itself until it returns."""
    window, fake = build(qapp)
    window._hub_query.setText("qwen")
    window._hub_search_btn.click()
    assert ("request_hub_search", "qwen") in fake.calls
    assert window._hub_search_btn.isEnabled() is False
    assert "Searching huggingface.co" in window._hub_status.text()


def test_hub_ui_empty_search_is_refused_locally(qapp):
    """An empty query is answered inline, without a wasted request."""
    window, fake = build(qapp)
    window._hub_query.setText("   ")
    window._hub_search_btn.click()
    assert "request_hub_search" not in names(fake)
    assert "Enter a search term" in window._hub_status.text()


def test_hub_ui_renders_catalog_rows(qapp):
    """A hub_catalog Result fills the discovery table from disk data."""
    window, fake = build(qapp)
    fake.result_q.put(
        gui_controller.Result(
            "hub_catalog",
            True,
            {
                "items": [
                    {
                        "repo_id": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                        "publisher": "unsloth",
                        "license_tag": "apache-2.0",
                        "source": "catalog",
                    }
                ],
                "reason": "",
            },
        )
    )
    window._drain()
    assert window._hub_repo_table.rowCount() == 1
    assert window._hub_repo_table.item(0, 0).text().startswith("unsloth/")
    assert window._hub_repo_table.item(0, 3).text() == "built-in list"


def test_hub_ui_offline_keeps_the_catalog_and_shows_the_reason(qapp):
    """An offline search prints the honest reason and leaves the catalog listed."""
    window, fake = build(qapp)
    fake.result_q.put(
        gui_controller.Result(
            "hub_catalog", True,
            {"items": [{"repo_id": "a/b", "publisher": "a", "source": "catalog"}],
             "reason": ""},
        )
    )
    window._drain()
    fake.result_q.put(
        gui_controller.Result(
            "hub_search", False,
            {"ok": False, "items": [], "query": "qwen",
             "reason": "Could not reach huggingface.co: no route. The list below "
                       "is locitize's built-in catalog; downloads still need a "
                       "connection."},
            error="offline",
        )
    )
    window._drain()
    assert window._hub_repo_table.rowCount() == 1, "the catalog must survive offline"
    assert "Could not reach huggingface.co" in window._hub_status.text()
    assert window._hub_search_btn.isEnabled() is True


def test_hub_ui_file_row_shows_the_fit_arithmetic_and_the_disclaimer(qapp):
    """Selecting a file shows the numbers, the verdict AND the estimate caveat."""
    window, fake = build(qapp)
    fake.result_q.put(
        gui_controller.Result(
            "hub_files", True,
            {
                "ok": True,
                "repo_id": "o/r",
                "items": [
                    {
                        "filename": "m.gguf",
                        "quant": "Q4_K_M",
                        "size_bytes": 4_920_739_232,
                        "sha256": "a" * 64,
                        "verification": "api",
                        "verification_label":
                            "Checksum verified against HuggingFace's file listing.",
                        "fit": {
                            "band": "fits",
                            "wording": "Should fit on your GPU",
                            "explanation": "4.6 GB weights + ~0.5 GB working memory "
                                           "vs 16.0 GB VRAM (Fake GPU)",
                            "disclaimer": "This is an estimate, not a guarantee. "
                                          "Actual use depends on context size, KV "
                                          "cache settings, and whatever else is "
                                          "using the GPU.",
                            "multi_gpu_caveat": "",
                        },
                    }
                ],
                "reason": "",
            },
        )
    )
    window._drain()
    assert window._hub_file_table.rowCount() == 1
    window._hub_file_table.setCurrentCell(0, 0)
    text = window._hub_fit.text()
    assert "Should fit on your GPU" in text
    assert "GB weights" in text
    assert "estimate, not a guarantee" in text
    assert "Checksum verified" in text
    assert window._hub_download_btn.isEnabled() is True


def test_hub_ui_download_is_disabled_until_a_file_is_chosen(qapp):
    """No dead Download button: it is off until there is something to download."""
    window, _fake = build(qapp)
    assert window._hub_download_btn.isEnabled() is False


def test_hub_ui_download_press_sends_the_full_payload(qapp, monkeypatch):
    """Download press carries the digest, rung, size and register choice through."""
    window, fake = build(qapp)
    monkeypatch.setattr(window, "_confirm", lambda title, text: True)
    window._hub_repo_rows = [{"repo_id": "o/r", "license_tag": "apache-2.0"}]
    window._hub_repo_table.setRowCount(1)
    window._hub_repo_table.setCurrentCell(0, 0)
    window._hub_file_rows = [
        {
            "filename": "m.gguf",
            "size_bytes": 123,
            "sha256": "b" * 64,
            "verification": "api",
            "fit": {"band": "fits"},
        }
    ]
    window._hub_file_table.setRowCount(1)
    window._hub_file_table.setCurrentCell(0, 0)
    window._hub_register_box.setChecked(True)
    window._hub_download_btn.click()
    call = next(c for c in fake.calls if c[0] == "request_hub_download")
    payload = call[1]
    assert payload["repo_id"] == "o/r"
    assert payload["filename"] == "m.gguf"
    assert payload["sha256"] == "b" * 64
    assert payload["verification"] == "api"
    assert payload["register"] is True
    assert payload["confirm_unverified"] is False
    assert window._hub_cancel_btn.isEnabled() is True


def test_hub_ui_declining_the_confirm_downloads_nothing(qapp, monkeypatch):
    """Saying No in the confirm dialog sends no download at all."""
    window, fake = build(qapp)
    monkeypatch.setattr(window, "_confirm", lambda title, text: False)
    window._hub_repo_rows = [{"repo_id": "o/r"}]
    window._hub_repo_table.setRowCount(1)
    window._hub_repo_table.setCurrentCell(0, 0)
    window._hub_file_rows = [
        {"filename": "m.gguf", "size_bytes": 1, "sha256": "c" * 64,
         "verification": "api", "fit": {"band": "fits"}}
    ]
    window._hub_file_table.setRowCount(1)
    window._hub_file_table.setCurrentCell(0, 0)
    window._hub_download_btn.click()
    assert "request_hub_download" not in names(fake)


def test_hub_ui_unverified_file_asks_a_second_separate_consent(qapp, monkeypatch):
    """V-NONE raises its own dialog carrying the exact consent sentence."""
    import modelhub

    window, fake = build(qapp)
    seen = []

    def record(title, text):
        seen.append((title, text))
        return True

    monkeypatch.setattr(window, "_confirm", record)
    window._hub_repo_rows = [{"repo_id": "o/r"}]
    window._hub_repo_table.setRowCount(1)
    window._hub_repo_table.setCurrentCell(0, 0)
    window._hub_file_rows = [
        {"filename": "m.gguf", "size_bytes": 1, "sha256": None,
         "verification": "none", "fit": {"band": "fits"}}
    ]
    window._hub_file_table.setRowCount(1)
    window._hub_file_table.setCurrentCell(0, 0)
    window._hub_download_btn.click()
    assert len(seen) == 2, "the unverified warning is a SEPARATE question"
    assert seen[1][1] == modelhub.UNVERIFIED_CONSENT_TEXT
    payload = next(c for c in fake.calls if c[0] == "request_hub_download")[1]
    assert payload["confirm_unverified"] is True


def test_hub_ui_exceeds_asks_again_but_never_blocks(qapp, monkeypatch):
    """A model larger than VRAM needs a second yes - and then it proceeds."""
    window, fake = build(qapp)
    seen = []
    monkeypatch.setattr(
        window, "_confirm", lambda title, text: (seen.append(title), True)[1]
    )
    window._hub_repo_rows = [{"repo_id": "o/r"}]
    window._hub_repo_table.setRowCount(1)
    window._hub_repo_table.setCurrentCell(0, 0)
    window._hub_file_rows = [
        {"filename": "m.gguf", "size_bytes": 1, "sha256": "d" * 64,
         "verification": "api",
         "fit": {"band": "exceeds", "explanation": "30 GB vs 16 GB",
                 "disclaimer": "This is an estimate, not a guarantee."}}
    ]
    window._hub_file_table.setRowCount(1)
    window._hub_file_table.setCurrentCell(0, 0)
    window._hub_download_btn.click()
    assert "Larger than your VRAM" in seen
    payload = next(c for c in fake.calls if c[0] == "request_hub_download")[1]
    assert payload["confirmed_exceeds"] is True


def test_hub_ui_progress_and_cancel(qapp):
    """Progress renders phase and bytes; Cancel sends the intent once."""
    window, fake = build(qapp)
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_progress", True,
            {"job_id": "j", "phase": "downloading", "bytes_done": 1_000_000_000,
             "bytes_total": 4_000_000_000, "rate_bps": 20_000_000, "eta_s": 150},
        )
    )
    window._drain()
    assert "downloading" in window._hub_status.text()
    assert "1.00 GB of 4.00 GB" in window._hub_status.text()
    assert window._hub_progress.value() == 25
    window._hub_cancel_btn.setEnabled(True)
    window._hub_cancel_btn.click()
    assert "request_hub_cancel" in names(fake)


def test_hub_ui_done_reports_verification_honestly(qapp):
    """A verified finish says so; an unverified one says it is not verified."""
    window, fake = build(qapp)
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_done", True,
            {"path": "/data/models/m.gguf", "sha256": "e" * 64,
             "verification": "none", "registered": False},
        )
    )
    window._drain()
    text = window._hub_status.text()
    assert "Not verified" in text
    assert "no publisher checksum" in text

    fake.result_q.put(
        gui_controller.Result(
            "hub_download_done", True,
            {"path": "/data/models/m.gguf", "sha256": "f" * 64,
             "verification": "api", "registered": True, "model_id": "m"},
        )
    )
    window._drain()
    text = window._hub_status.text()
    assert "Checksum verified" in text
    assert "Registered as 'm'" in text


# Eight or more hex characters in a row. Nothing in the completion line's
# legitimate vocabulary (paths, ids, plain English) looks like that, so this is
# the bright-line rule UX Spec section 14 asked for: QA can grade H7/H8 by
# searching the visible text, with no judgement call about "how much hex is too
# much". Eight is the shortest excerpt anyone would plausibly reintroduce.
HEX_RUN_RE = re.compile(r"[0-9a-f]{8,}", re.IGNORECASE)

# The exact completion line each rung must produce, per UX Spec section 14's
# final wording (MEDIUM-6/MEDIUM-7 resolution) and Visual QA Checklist H2k/H7/H8.
# Pinned here as literals on purpose: reading them from VERIFICATION_LABELS
# would make the test agree with any future edit to the labels, which is the
# opposite of what a copy-contract regression test is for.
RUNG_COMPLETION_LINES = {
    "api": "Saved to /data/models/tiny.gguf. "
           "Checksum verified against HuggingFace's file listing.",
    "etag": "Saved to /data/models/tiny.gguf. "
            "Checksum verified against HuggingFace's linked file hash.",
    "none": "Saved to /data/models/tiny.gguf. "
            "Not verified - no publisher checksum available.",
}


@pytest.mark.parametrize("rung", sorted(RUNG_COMPLETION_LINES))
def test_hub_ui_completion_line_is_the_exact_rung_label_with_no_hex(qapp, rung):
    """Each rung's completion line is pinned, and carries no hex digest at all.

    Two assertions, because they fail for different reasons: the first catches
    a reworded claim, the second catches a hex excerpt creeping back in beside
    a correctly-worded claim (which is how the shipped line drifted before -
    the words were fine, the trailing "(<hex>...)" was not).
    """
    window, fake = build(qapp)
    # A real 64-hex digest travels in the payload: the view is being handed the
    # thing it must not print, not merely failing to receive it.
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_done", True,
            {"path": "/data/models/tiny.gguf", "sha256": "ab12cd34" * 8,
             "verification": rung, "registered": False},
        )
    )
    window._drain()
    text = window._hub_status.text()
    assert text == RUNG_COMPLETION_LINES[rung]
    assert HEX_RUN_RE.search(text) is None, f"hex digest leaked into: {text!r}"
    # And the bare word the hazard is about never reaches the user.
    assert "etag" not in text.lower()


def test_hub_ui_completion_line_matches_the_shared_label_table(qapp):
    """The view renders modelhub's labels, it does not keep its own copy.

    Guards the failure mode that produced MEDIUM-6: two surfaces stating the
    same claim in two places, one of which was never updated.
    """
    window, fake = build(qapp)
    for rung, expected in RUNG_COMPLETION_LINES.items():
        assert modelhub.VERIFICATION_LABELS[rung] in expected
        fake.result_q.put(
            gui_controller.Result(
                "hub_download_done", True,
                {"path": "/data/models/tiny.gguf", "sha256": "0" * 64,
                 "verification": rung, "registered": False},
            )
        )
        window._drain()
        assert modelhub.VERIFICATION_LABELS[rung] in window._hub_status.text()


def test_hub_ui_completion_line_keeps_the_register_clause_and_stays_hex_free(qapp):
    """The Registered clause is appended after the label, still with no hex."""
    window, fake = build(qapp)
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_done", True,
            {"path": "/data/models/tiny.gguf", "sha256": "beefcafe" * 8,
             "verification": "api", "registered": True, "model_id": "tiny"},
        )
    )
    window._drain()
    text = window._hub_status.text()
    assert text == (
        "Saved to /data/models/tiny.gguf. "
        "Checksum verified against HuggingFace's file listing. "
        "Registered as 'tiny'."
    )
    assert HEX_RUN_RE.search(text) is None


def test_hub_ui_unknown_rung_falls_back_to_the_unverified_label(qapp):
    """A rung the view does not recognise must under-claim, never over-claim."""
    window, fake = build(qapp)
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_done", True,
            {"path": "/data/models/tiny.gguf", "sha256": "0" * 64,
             "verification": "something-new", "registered": False},
        )
    )
    window._drain()
    text = window._hub_status.text()
    assert text.endswith("Not verified - no publisher checksum available.")


def test_hub_ui_mismatch_message_still_carries_both_full_digests(qapp):
    """The no-hex rule is about provenance CLAIMS, not about diagnostics.

    A checksum mismatch is the one place full digests belong (the owner may
    need to paste them into a bug report), so this test exists to stop the
    no-hex rule being over-applied to the failure path.
    """
    window, fake = build(qapp)
    expected = "a1b2c3d4" * 8
    actual = "9f8e7d6c" * 8
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_done", False, {"path": ""},
            error="checksum mismatch - the download was deleted (never left in "
                  f"place).\n  expected: {expected}\n  actual:   {actual}",
        )
    )
    window._drain()
    text = window._hub_status.text()
    assert expected in text and actual in text


def test_hub_ui_failed_download_shows_the_error_and_no_override(qapp):
    """A checksum failure states both digests and offers no override control."""
    window, fake = build(qapp)
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_done", False, {"path": ""},
            error="checksum mismatch - the download was deleted (never left in "
                  "place).\n  expected: aaa\n  actual:   bbb",
        )
    )
    window._drain()
    assert "checksum mismatch" in window._hub_status.text()
    assert window._hub_cancel_btn.isEnabled() is False
    assert window._hub_progress.isHidden() is True
    # There is no widget anywhere on the page offering to keep a bad file.
    buttons = [
        child.text().lower()
        for child in window.findChildren(type(window._hub_download_btn))
    ]
    assert not any("anyway" in label or "override" in label for label in buttons)


def test_hub_ui_disabled_state_names_the_setting(qapp):
    """models_hub.enabled false disables every egress control and says why."""
    window, _fake = build(qapp, hub_enabled=False)
    assert "Model downloading is off" in window._hub_status.text()
    assert window._hub_search_btn.isEnabled() is False
    assert window._hub_query.isEnabled() is False
    assert window._hub_download_btn.isEnabled() is False


def test_hub_ui_disabled_state_makes_no_catalog_request(qapp):
    """With the feature off, the page asks the controller for nothing at all."""
    _window, fake = build(qapp, hub_enabled=False)
    assert "request_hub_catalog" not in names(fake)


# =========================================================================== #
# SEC-M14-1 regression: no widget in this view interprets third-party text
# =========================================================================== #

# Security's check-9b payload, verbatim. The tag wraps LOCITIZE's OWN V-API label so
# that, rendered as rich text, the user reads a verification claim LOCITIZE never
# made - at the exact rung (V-NONE) where the dialog is the only protection.
SEC_M14_1_TAG = (
    "<br><br><b>Checksum verified against HuggingFace's file listing.</b><br>"
)
SEC_M14_1_SEARCH_PAYLOAD = [
    {
        "id": "attacker/repo",
        "downloads": 10,
        "likes": 1,
        "tags": ["pytorch", "license:" + SEC_M14_1_TAG],
    }
]


def what_the_user_sees(text, text_format):
    """Model Qt's own rule for turning a widget's text into what is displayed.

    AutoText (Qt's default) SNIFFS: if the string looks like HTML the widget
    renders the whole thing as rich text, so tags vanish and become formatting.
    This mirrors that decision rather than asserting on a property, so the test
    fails for the reason the user would notice - the wrong thing on screen.
    """
    from PySide6 import QtGui

    QtCore = __import__("PySide6.QtCore", fromlist=["QtCore"])
    rich = text_format == QtCore.Qt.TextFormat.RichText or (
        text_format == QtCore.Qt.TextFormat.AutoText
        and QtGui.Qt.mightBeRichText(text)
    )
    if not rich:
        return text
    document = QtGui.QTextDocument()
    document.setHtml(text)
    return document.toPlainText()


def message_box_text(box):
    """The text the user would actually read in a QMessageBox."""
    from PySide6 import QtWidgets

    label = box.findChild(QtWidgets.QLabel, "qt_msgbox_label")
    assert label is not None, "QMessageBox has no message label"
    return what_the_user_sees(label.text(), label.textFormat())


def test_hub_ui_confirm_dialog_shows_untrusted_markup_literally(qapp):
    """The consent dialog renders its body literally - the SEC-M14-1 regression.

    Presentation half. The body is handed markup that wraps LOCITIZE's V-API label;
    the user must see that markup as characters, must NOT see a bolded false
    verification claim, and the true V-NONE line must still stand on its own.
    """
    window, _fake = build(qapp)
    body = "\n".join(
        [
            "Download evil.gguf",
            "from attacker/repo",
            "Size: 2.5 GB",
            "Not verified - no publisher checksum available.",
            "",
            "This model's licence is an agreement between you and its publisher."
            f" Publisher's licence: {SEC_M14_1_TAG}.",
        ]
    )
    seen = message_box_text(window._confirm_box("Download this model?", body))
    assert "<b>" in seen and "<br>" in seen, "markup was interpreted, not shown"
    assert "Not verified - no publisher checksum available." in seen.split("\n")
    # The forged claim never appears as a line of its own, which is the form the
    # user reads as LOCITIZE's own finding.
    assert "Checksum verified against HuggingFace's file listing." not in [
        line.strip() for line in seen.split("\n")
    ]
    assert seen == body, "the dialog body must be shown exactly as composed"


def test_hub_ui_download_body_carries_no_publisher_markup(qapp, monkeypatch):
    """End to end: a hostile tag from a real search payload reaches the dialog inert.

    Data half, driven through the parser the product uses, the table the user
    picks from, and the real Download press - not by calling the sanitiser.
    """
    window, _fake = build(qapp)
    rows = modelhub.parse_search_results(SEC_M14_1_SEARCH_PAYLOAD)
    window._fill_hub_repos(rows)
    window._hub_repo_table.setCurrentCell(0, 0)
    window._hub_file_rows = [
        {"filename": "evil.gguf", "size_bytes": 2_500_000_000, "sha256": None,
         "verification": "none",
         # The label the file table attaches for rung V-NONE; carried here so the
         # body under test is the one the user really gets.
         "verification_label": modelhub.VERIFICATION_LABELS["none"],
         "fit": {"band": "fits"}}
    ]
    window._hub_file_table.setRowCount(1)
    window._hub_file_table.setCurrentCell(0, 0)
    bodies = []
    monkeypatch.setattr(
        window, "_confirm", lambda title, text: (bodies.append(text), False)[1]
    )
    window._hub_download_btn.click()
    assert bodies, "the Download press must raise a confirmation"
    for body in bodies:
        assert "<" not in body and ">" not in body
        assert "Checksum verified" not in body
    assert "Not verified - no publisher checksum available." in bodies[0]


def test_desktop_text_widgets_never_auto_detect_rich_text(qapp):
    """The policy is applied to the CLASS of text widgets, not one dialog.

    Every label in the window, and every text view that receives model output or
    a transcript, must be plain text - so a surface added later cannot quietly
    reintroduce SEC-M14-1 somewhere else on the page.
    """
    from PySide6 import QtCore, QtWidgets

    window, _fake = build(qapp)
    labels = window.findChildren(QtWidgets.QLabel)
    assert labels, "the window should have labels to check"
    bad = [
        label.objectName() or label.text()[:30]
        for label in labels
        if label.textFormat() != QtCore.Qt.TextFormat.PlainText
    ]
    assert bad == [], f"labels still auto-detecting rich text: {bad}"
    edits = window.findChildren(QtWidgets.QTextEdit)
    assert edits, "the window should have text views to check"
    assert [e.objectName() for e in edits if e.acceptRichText()] == []


def test_desktop_appended_model_output_is_shown_literally(qapp):
    """Model replies and transcript lines are appended without markup sniffing."""
    window, _fake = build(qapp)
    reply = "here is <b>bold</b> and a <br> break"
    window._append_conversation("assistant", reply)
    shown = window._conversation.toPlainText()
    assert "<b>bold</b>" in shown
    assert "<br>" in shown


def test_hub_ui_the_download_button_uses_the_plain_text_dialog(qapp, monkeypatch):
    """The press really shows the hardened dialog - the helper is not an orphan.

    Guards the wiring, not just the helper: `_confirm` must build the plain-text
    box, and it must not go back to `QMessageBox.question`, whose text format
    cannot be set before it is shown (that convenience call is also why the
    pre-fix dialog could not be inspected without executing a modal loop).
    """
    from PySide6 import QtCore, QtWidgets

    window, _fake = build(qapp)

    def refuse_static(*args, **kwargs):
        raise AssertionError("the confirm dialog must not use QMessageBox.question")

    shown = []

    def spy_exec(self):
        shown.append(self)
        return QtWidgets.QMessageBox.StandardButton.No

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(refuse_static))
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", spy_exec)
    assert window._confirm("Download this model?", "body <b>x</b>") is False
    assert len(shown) == 1
    assert shown[0].textFormat() == QtCore.Qt.TextFormat.PlainText
    assert message_box_text(shown[0]) == "body <b>x</b>"


# =========================================================================== #
# DEFECT-QA-M14-2: browsing during a download must not steal the running
# download's controls.
#
# QA drove this live: with a 1.5 GB transfer at 5.8 MB/s, clicking a second file
# row re-armed Download; pressing it produced a refusal that the view treated as
# the running job's terminal result, so Cancel went dead and the progress bar
# disappeared while the bytes kept moving.
# =========================================================================== #


def start_a_download(window, monkeypatch, filename="a.gguf", live=True):
    """Press Download for `filename` exactly as a user would, and return the job id.

    With `live` (the default) one progress Result is then pumped through the
    window, because that is what the controller does as soon as bytes move -
    and since NEW-QA-M14-7 it is progress, not the button press, that makes a
    job the panel's ACTIVE job. A helper that stopped at the click would leave
    every caller asserting against a request that the controller had not yet
    accepted, which is not the state any of these tests are about.
    """
    import modelhub

    monkeypatch.setattr(window, "_confirm", lambda title, text: True)
    window._hub_repo_rows = [{"repo_id": "o/r", "license_tag": "apache-2.0"}]
    window._hub_repo_table.setRowCount(1)
    window._hub_repo_table.setCurrentCell(0, 0)
    window._hub_file_rows = [
        {"filename": filename, "size_bytes": 1_500_000_000, "sha256": "a" * 64,
         "verification": "api", "fit": {"band": "fits"}},
        {"filename": "b.gguf", "size_bytes": 900_000_000, "sha256": "b" * 64,
         "verification": "api", "fit": {"band": "fits"}},
    ]
    window._hub_file_table.setRowCount(2)
    window._hub_file_table.setCurrentCell(0, 0)
    window._hub_download_btn.click()
    job = modelhub.hub_job_id("o/r", filename)
    if live:
        window._gc.result_q.put(
            gui_controller.Result(
                "hub_download_progress", True,
                {"job_id": job, "repo_id": "o/r", "filename": filename,
                 "phase": "downloading", "bytes_done": 290_000_000,
                 "bytes_total": 1_500_000_000, "rate_bps": 13_500_000,
                 "eta_s": 90.0},
            )
        )
        window._drain()
    return job


def test_hub_ui_selecting_another_file_mid_download_does_not_rearm_download(
    qapp, monkeypatch
):
    """Clicking a second file row during a download leaves Download disabled."""
    window, fake = build(qapp)
    start_a_download(window, monkeypatch)
    assert window._hub_download_btn.isEnabled() is False
    before = names(fake).count("request_hub_download")
    # The user browses the other file while the transfer runs.
    window._hub_file_table.setCurrentCell(1, 0)
    assert window._hub_download_btn.isEnabled() is False, (
        "Download was re-armed during a live download"
    )
    window._hub_download_btn.click()  # a disabled button sends nothing
    assert names(fake).count("request_hub_download") == before
    # And the live download's own controls are untouched.
    assert window._hub_cancel_btn.isEnabled() is True
    # isHidden(), not isVisible(): these windows are never shown in the offscreen
    # suite, so isVisible() is False for every widget regardless of the bug.
    assert window._hub_progress.isHidden() is False


def test_hub_ui_a_refused_second_download_leaves_the_live_download_controllable(
    qapp, monkeypatch
):
    """A refusal about another job is a status line, never terminal UI.

    This is the measured defect: `cancel_enabled: false`, `progress_visible:
    false`, with the transfer still running.
    """
    import modelhub

    window, fake = build(qapp)
    live = start_a_download(window, monkeypatch)
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_done", False,
            {"job_id": modelhub.hub_job_id("o/r", "b.gguf"),
             "repo_id": "o/r", "filename": "b.gguf", "refused": True},
            error="A download is already running.",
        )
    )
    window._drain()
    assert window._hub_status.text() == "A download is already running."
    assert window._hub_cancel_btn.isEnabled() is True, "Cancel died on a live download"
    assert window._hub_progress.isHidden() is False, "the progress bar was hidden"
    assert window._hub_active_job == live
    # A later progress tick for the live job still renders normally.
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_progress", True,
            {"job_id": live, "phase": "downloading", "bytes_done": 500_000_000,
             "bytes_total": 1_500_000_000, "rate_bps": 5_800_000},
        )
    )
    window._drain()
    assert "downloading" in window._hub_status.text()


def refuse_the_last_request(window, fake):
    """Answer the most recent hub download request the way the controller does.

    The refusal is stamped with the REQUESTED job's own id, read back out of the
    payload the window itself sent (never hand-typed here), exactly as
    gui_controller._do_hub_download builds it when a transfer is already
    running. That is what makes this a reproduction rather than an assertion
    about an invented state - the state QA proved the app could not produce is
    the one this helper refuses to fabricate.
    """
    import modelhub

    request = [c for c in fake.calls if c[0] == "request_hub_download"][-1][1]
    refused = dict(request)
    refused["job_id"] = modelhub.hub_job_id(
        request.get("repo_id", ""), request.get("filename", "")
    )
    refused["refused"] = True
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_done", False, refused, error="A download is already running."
        )
    )
    window._drain()
    return refused["job_id"]


def test_hub_ui_a_second_download_request_never_displaces_the_live_job(
    qapp, monkeypatch
):
    """NEW-QA-M14-7: the job-id guard is load-bearing, driven the way QA drove it.

    The old test injected a refusal carrying a foreign job id while
    _hub_active_job held the live one - a combination the shipped app could not
    produce, because _on_hub_download overwrote _hub_active_job with the NEW
    request's id before the controller had decided anything. So the guard
    compared the refusal against itself, found them equal, and killed Cancel and
    the progress bar of a transfer that was still moving bytes.

    Here the second request goes through the real handler (which is what a future
    code path re-arming Download would do), and the refusal is built from the
    window's own request payload.
    """
    window, fake = build(qapp)
    live = start_a_download(window, monkeypatch)
    assert window._hub_active_job == live

    # A second request for a different file, through the real handler.
    window._hub_file_table.setCurrentCell(1, 0)
    window._on_hub_download()
    assert window._hub_active_job == live, (
        "the live job's id was displaced by a request the controller had not "
        "accepted - this is the root cause of NEW-QA-M14-7"
    )

    refused_job = refuse_the_last_request(window, fake)
    assert refused_job != live

    assert window._hub_status.text() == "A download is already running."
    assert window._hub_cancel_btn.isEnabled() is True, "Cancel died on a live download"
    assert window._hub_progress.isHidden() is False, "the progress bar was hidden"
    assert window._hub_active_job == live
    # The refused request is over, so it leaves no pending state behind.
    assert window._hub_pending_job is None


def test_hub_ui_progress_for_another_job_never_repaints_the_live_one(
    qapp, monkeypatch
):
    """Only the active job's progress may move this panel's bar and status line."""
    window, fake = build(qapp)
    live = start_a_download(window, monkeypatch)
    window._hub_progress.setValue(19)
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_progress", True,
            {"job_id": "someone/else/x.gguf", "phase": "downloading",
             "bytes_done": 900_000_000, "bytes_total": 1_000_000_000},
        )
    )
    window._drain()
    assert window._hub_progress.value() == 19
    assert window._hub_active_job == live


def test_hub_ui_a_refused_first_request_re_arms_download(qapp, monkeypatch):
    """With nothing live, a refusal must not leave Download disabled forever."""
    window, fake = build(qapp)
    start_a_download(window, monkeypatch, live=False)
    assert window._hub_download_btn.isEnabled() is False, (
        "a request in flight must keep Download disabled (DEFECT-QA-M14-2)"
    )
    assert window._hub_active_job is None
    refuse_the_last_request(window, fake)
    assert window._hub_pending_job is None
    assert window._hub_download_btn.isEnabled() is True


def test_hub_ui_the_live_jobs_own_terminal_result_still_ends_the_download(
    qapp, monkeypatch
):
    """The job-id guard must not swallow the result it exists to let through."""
    window, fake = build(qapp)
    live = start_a_download(window, monkeypatch)
    fake.result_q.put(
        gui_controller.Result(
            "hub_download_done", True,
            {"job_id": live, "path": "/data/models/a.gguf", "sha256": "a" * 64,
             "verification": "api", "registered": False},
        )
    )
    window._drain()
    assert window._hub_cancel_btn.isEnabled() is False
    assert window._hub_progress.isHidden() is True
    assert window._hub_active_job is None
    # With nothing in flight, browsing re-arms Download again.
    window._hub_file_table.setCurrentCell(1, 0)
    assert window._hub_download_btn.isEnabled() is True


# =========================================================================== #
# DEFECT-QA-M14-3: the built-in catalog must be visible at the default size
#
# The window's own default is 1180x780 and its minimum is 960x680. At both, the
# repository table's viewport measured 0 pixels tall: three catalog rows loaded,
# counted in the status line, and impossible to see or reach.
# =========================================================================== #


CATALOG_THREE_ROWS = {
    "items": [
        {"repo_id": f"owner/repo-{index}", "publisher": "owner",
         "license_tag": "apache-2.0", "source": "catalog"}
        for index in range(3)
    ],
    "reason": "",
}


def rows_visible(table):
    """How many rows actually fit in the table's viewport right now."""
    row_height = table.rowHeight(0) or table.verticalHeader().defaultSectionSize()
    return table.viewport().height() // max(1, row_height)


@pytest.mark.parametrize(("width", "height"), [(1180, 780), (960, 680)])
def test_hub_ui_catalog_rows_are_visible_at_the_shipped_window_sizes(
    qapp, width, height
):
    """All three built-in rows are on screen at the default and minimum sizes."""
    window, fake = build(qapp)
    window.resize(width, height)
    window.show()
    qapp[0].processEvents()
    fake.result_q.put(gui_controller.Result("hub_catalog", True, CATALOG_THREE_ROWS))
    window._drain()
    qapp[0].processEvents()
    table = window._hub_repo_table
    assert table.rowCount() == 3
    assert table.viewport().height() > 0, "the catalog viewport collapsed to zero"
    assert rows_visible(table) >= 3, (
        f"only {rows_visible(table)} of 3 catalog rows fit at {width}x{height}"
    )
    window.hide()


def test_models_page_is_scrollable_so_the_catalog_is_reachable(qapp):
    """The page carries a scroll area, so a short window cannot hide the panel.

    QA measured `scroll_areas: 0`: the Get models panel was not merely below the
    fold, it was unreachable without resizing the window.
    """
    from PySide6 import QtWidgets

    window, _fake = build(qapp)
    page = window.pages["Models"]
    areas = page.findChildren(QtWidgets.QScrollArea)
    if isinstance(page, QtWidgets.QScrollArea):
        areas = [page] + areas
    assert areas, "the Models page has no scroll area"
    assert any(area.widgetResizable() for area in areas)
    # The Get models panel must be INSIDE the scrolled content. A scroll area
    # that scrolls something else would satisfy the assertions above while
    # leaving the catalog exactly as unreachable as it shipped, so the panel's
    # own widgets are walked up to the scrolled widget here.
    #
    # (The line this replaced was `assert ... or True`, which can never fail.)
    scrolled = [area.widget() for area in areas if area.widget() is not None]
    assert scrolled, "the scroll area holds no widget at all"

    def inside_scrolled(widget):
        node = widget
        while node is not None:
            if node in scrolled:
                return True
            node = node.parentWidget()
        return False

    assert inside_scrolled(window._hub_repo_table), "the catalog table is not scrolled"
    assert inside_scrolled(window._hub_download_btn), "the Download button is not scrolled"


def test_the_tested_window_sizes_are_the_ones_the_app_actually_ships(qapp):
    """Pin the sizes the catalog test uses to the window's own geometry.

    The catalog test above is parametrized with 1180x780 and 960x680 - the
    default and minimum QA measured. Those are literals in a test file, so they
    could silently stop describing the app the moment someone edits the resize
    call. This test fails in that case and points at the parametrization, which
    is the whole reason it exists.
    """
    window, _fake = build(qapp)
    assert (window.width(), window.height()) == (1180, 780), (
        "the shipped default window size changed - update the sizes in "
        "test_hub_ui_catalog_rows_are_visible_at_the_shipped_window_sizes"
    )
    assert (window.minimumWidth(), window.minimumHeight()) == (960, 680), (
        "the shipped minimum window size changed - update the same parametrization"
    )


# =========================================================================== #
# DEC-M14-9 item 7: the one-time in-app notice that the user's data moved
#
# Named data_root_migration so it runs under AC-M14-27's keyword: the notice is
# part of the migration decision, not a documentation afterthought. A product
# that relocates someone's chat history without a word has done something they
# cannot audit.
# =========================================================================== #


def test_data_root_migration_notice_is_shown_in_the_app_exactly_once(
    qapp, monkeypatch, tmp_path
):
    """The notice names both real folders, appears once, and never blocks start."""
    import migration

    window, fake = build(qapp)
    install = tmp_path / "install"
    data = tmp_path / "locitize-data"
    install.mkdir()
    data.mkdir()
    (install / "memory").mkdir()
    (install / "memory" / "chat.jsonl").write_text("{}", encoding="utf-8")
    migration.migrate_install_data(
        install, data, webui_port=0, service_is_running=lambda _p: False
    )

    fake._settings = type("S", (), {"data_dir": data})()
    shown = []
    # The dialog is not executed in the offscreen suite; what is asserted is the
    # text handed to it and the fact that it is handed over exactly once.
    monkeypatch.setattr(
        window,
        "_confirm_box",
        lambda title, text: shown.append(text),
    )
    _app, desktop_module = qapp
    monkeypatch.setattr(
        desktop_module,
        "plain_message_box",
        lambda *args, **kwargs: type("Box", (), {"exec": lambda self: None})(),
    )

    first = window.show_data_migration_notice()
    assert str(install) in first and str(data) in first
    assert "copied" in first and "Nothing was deleted" in first

    second = window.show_data_migration_notice()
    assert second == "", "the notice repeated itself on a later start"


def test_data_root_migration_notice_is_silent_when_nothing_moved(qapp, tmp_path):
    """A fresh install owes no notice, and a missing settings object is not a crash."""
    window, fake = build(qapp)
    data = tmp_path / "locitize-data"
    data.mkdir()
    fake._settings = type("S", (), {"data_dir": data})()
    assert window.show_data_migration_notice() == ""
