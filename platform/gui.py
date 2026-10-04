"""Tkinter presentation shell for the LOCITIZE command center (Architecture G1/G3).

PRESENTATION ONLY. This module builds widgets, binds each control to exactly one
gui_controller intent method, runs the root.after() pump, and handles the window
close. It performs no network access, no child-process work, and no config-file
I/O, and holds no business rules - per the strict import allow-list (G1, widened by
exactly two stdlib Tk modules for M10) it imports only tkinter, tkinter.ttk,
tkinter.messagebox, tkinter.filedialog (the Vision image picker), tkinter.scrolledtext
(the Talk conversation view), queue, and gui_controller. Every effect (model
start/switch/stop, whisper, listen, save, chat, proxy, the M10 assistant session,
vision-describe, memory search, benchmark, shutdown) routes through the
GuiController. The Reviewer verifies this boundary by code read;
the only occurrences of the word "requests" below are llama.cpp's own
requests_processing metric label rendered in the monitor, not an HTTP client.

Threading (G2): Tkinter's mainloop owns the one UI thread. On a click this shell
only disables buttons, sets a status label, and enqueues a command; the ops and
monitor threads live in gui_controller. The pump (self._drain, rescheduled every
100ms) is the SOLE writer of widgets from background outcomes, so no Tk call ever
happens off-thread. WM_DELETE_WINDOW -> gui_controller.shutdown() (which joins the
threads and calls stop_all) -> root.destroy(); the launcher's atexit backstop
remains registered so even a hard crash cannot orphan a child process.

This file is exercised only by the owner's manual run; the automated suite never
imports it or constructs a Tk root (G7), so a headless build machine is fine.
"""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import gui_controller
from gui_controller import UiState


class LocitizeGui:
    """The single-window command center. One instance per launch."""

    def __init__(self, root: tk.Tk, controller: gui_controller.GuiController, health: str) -> None:
        self._root = root
        self._gc = controller
        self._health = health
        # The pump-maintained snapshot; the UI thread renders from this and never
        # reads a controller live (G3).
        self._ui = UiState()
        self._models = controller.list_models()
        self._selected_id: str | None = None
        # Default to fastest measured generation throughput first. A click on
        # the active header toggles direction; unmeasured rows always stay last.
        self._sort_col: str | None = "benchmark_tok_s"
        self._sort_desc = True
        # Tk string vars for the live monitor + status/health lines.
        self._status_var = tk.StringVar(value="ready")
        self._health_var = tk.StringVar(value=self._health_line())
        self._monitor_var = tk.StringVar(value="idle")
        self._monitor_reason_var = tk.StringVar(value="")
        # Holds ONLY a validation error for the editors (owner request 1 killed the
        # "* unsaved" / "saved; applies on next start" notices - the Save button's
        # greyed/enabled state is now the sole save indicator).
        self._edit_error_var = tk.StringVar(value="")
        # Single voice status line (owner request 5): last transcript segment or the
        # current capture state, replacing the removed multiline transcript box.
        self._voice_status_var = tk.StringVar(value="idle")

        self._build_widgets()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_model_rows()
        self._refresh_buttons()

    # ---- widget construction --------------------------------------------- #

    def _build_widgets(self) -> None:
        self._root.title("locitize Command Center")
        self._root.minsize(720, 640)
        outer = ttk.Frame(self._root, padding=10)
        outer.pack(fill="both", expand=True)

        self._build_model_panel(outer)   # Panel 1 + 2
        self._build_editor_panel(outer)  # Panel 3
        self._build_monitor_panel(outer) # Panel 4
        self._build_voice_panel(outer)   # Panel 5
        self._build_voice_out_panel(outer)  # Panel 6 (M6: TTS)
        self._build_talk_panel(outer)    # Panel 7 (M10: Talk/assistant - centerpiece)
        self._build_vision_panel(outer)  # M10: Vision (describe an image)
        self._build_memory_panel(outer)  # M10: Memory (search past conversations)
        self._build_footer(outer)        # Health line

    def _build_model_panel(self, parent: ttk.Frame) -> None:
        """Panel 1 (model list + status) and Panel 2 (start/switch/stop)."""
        frame = ttk.LabelFrame(parent, text="Models", padding=8)
        frame.pack(fill="x", pady=(0, 8))

        # Columns include Size (GB) (owner request 2). Each heading is clickable to
        # sort by that column (owner request 3); the command re-sorts self._models
        # and rebuilds the rows while preserving the selected row.
        # Generation throughput is read from the latest matching successful
        # benchmark result ("-" until measured). It is deliberately distinct from
        # the deterministic quality percentage retained in the detailed report.
        columns = ("name", "id", "size", "score", "status")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=6)
        tree.heading("name", text="Model", command=lambda: self._on_sort("name"))
        tree.heading("id", text="id", command=lambda: self._on_sort("id"))
        tree.heading("size", text="Size", command=lambda: self._on_sort("size"))
        tree.heading(
            "score",
            text="Last gen tok/s",
            command=lambda: self._on_sort("benchmark_tok_s"),
        )
        tree.heading("status", text="Status", command=lambda: self._on_sort("status"))
        tree.column("name", width=200)
        tree.column("id", width=170)
        tree.column("size", width=90, anchor="e")
        tree.column("score", width=70, anchor="e")
        tree.column("status", width=150)
        tree.pack(fill="x")
        tree.bind("<<TreeviewSelect>>", self._on_select_model)
        self._tree = tree

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(8, 0))
        self._start_btn = ttk.Button(controls, text="Start", command=self._on_start)
        self._start_btn.pack(side="left")
        self._stop_btn = ttk.Button(controls, text="Stop", command=self._on_stop)
        self._stop_btn.pack(side="left", padx=(6, 0))
        self._chat_btn = ttk.Button(controls, text="Open Chat", command=self._on_chat)
        self._chat_btn.pack(side="left", padx=(6, 0))
        # M10.5: run the existing single-config benchmark for the selected model and
        # refresh its Score column. Routes through gui_controller (request_benchmark);
        # disabled while a Talk session is live (model-exclusive) via _refresh_buttons.
        self._benchmark_btn = ttk.Button(
            controls, text="Benchmark selected", command=self._on_benchmark
        )
        self._benchmark_btn.pack(side="left", padx=(6, 0))
        ttk.Label(controls, textvariable=self._status_var).pack(side="right")

    def _build_editor_panel(self, parent: ttk.Frame) -> None:
        """Panel 3 - per-model gpu_layers / context_size editors."""
        frame = ttk.LabelFrame(parent, text="Model settings (applies on next start)", padding=8)
        frame.pack(fill="x", pady=(0, 8))

        ttk.Label(frame, text="gpu_layers (999 = all)").grid(row=0, column=0, sticky="w")
        self._gpu_var = tk.StringVar()
        self._gpu_entry = ttk.Entry(frame, textvariable=self._gpu_var, width=12)
        self._gpu_entry.grid(row=0, column=1, sticky="w", padx=(6, 16))
        self._gpu_var.trace_add("write", lambda *_: self._on_edit_change())

        ttk.Label(frame, text="context_size").grid(row=0, column=2, sticky="w")
        self._ctx_var = tk.StringVar()
        self._ctx_entry = ttk.Entry(frame, textvariable=self._ctx_var, width=12)
        self._ctx_entry.grid(row=0, column=3, sticky="w", padx=(6, 16))
        self._ctx_var.trace_add("write", lambda *_: self._on_edit_change())

        # Save starts DISABLED (editors match the saved registry values). It enables
        # the moment either field differs and greys out again after a successful
        # write - that grey-out is the confirmation (owner request 1).
        self._save_btn = ttk.Button(frame, text="Save", command=self._on_save, state="disabled")
        self._save_btn.grid(row=0, column=4, sticky="w")

        # This label shows a validation error only, in the same place errors already
        # appeared; it is empty in the normal dirty/clean states.
        self._edit_hint = ttk.Label(frame, textvariable=self._edit_error_var, foreground="#b00")
        self._edit_hint.grid(row=1, column=0, columnspan=5, sticky="w", pady=(4, 0))

    def _build_monitor_panel(self, parent: ttk.Frame) -> None:
        """Panel 4 - live monitor (hidden until a model runs)."""
        self._monitor_frame = ttk.LabelFrame(parent, text="Live monitor", padding=8)
        self._monitor_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(self._monitor_frame, textvariable=self._monitor_var).pack(anchor="w")
        ttk.Label(
            self._monitor_frame, textvariable=self._monitor_reason_var, foreground="#666"
        ).pack(anchor="w")

    def _build_voice_panel(self, parent: ttk.Frame) -> None:
        """Panel 5 - whisper toggle, Listen, and a single status line.

        Owner request 5: the owner does not want to watch a scrolling transcript, so
        the multiline transcript box is gone. The panel is now the two buttons plus
        one status line showing the last transcript segment (or the capture state).
        The --listen pipeline (AC9: dedup + VAD filtering) is unchanged; only how its
        output is surfaced changed - we show the most recent line instead of a log.
        """
        frame = ttk.LabelFrame(parent, text="Voice (speech-to-text)", padding=8)
        frame.pack(fill="x", pady=(0, 8))

        controls = ttk.Frame(frame)
        controls.pack(fill="x")
        self._whisper_btn = ttk.Button(
            controls, text="Start whisper server", command=self._on_whisper
        )
        self._whisper_btn.pack(side="left")
        self._listen_btn = ttk.Button(
            controls, text="Listen (15s)", command=self._on_listen
        )
        self._listen_btn.pack(side="left", padx=(6, 0))

        ttk.Label(frame, textvariable=self._voice_status_var, foreground="#333").pack(
            anchor="w", pady=(8, 0)
        )

        # Owner request (2026-07-19): the transcript box is back so speech-to-text
        # can be verified by eye. Read-only; each accepted segment appends a line.
        # The status line above still shows capture state / the latest segment.
        self._transcript_box = tk.Text(frame, height=5, state="disabled", wrap="word")
        self._transcript_box.pack(fill="x", pady=(6, 0))

    def _build_voice_out_panel(self, parent: ttk.Frame) -> None:
        """Panel 6 - Kokoro text-to-speech (voice OUT, M6, Architecture M6.5).

        A voice dropdown populated from the on-disk .pt set, a "Speak test" button
        that synthesizes+plays a fixed line in the selected voice, and an "Audition
        all" button that speaks the sample sentence in every voice. Presentation
        only: every button routes through gui_controller (request_speak /
        request_audition) on the ops worker, never the UI thread (G1/G2). When no
        voices are configured the controls disable with an honest status line rather
        than presenting a dead button.
        """
        frame = ttk.LabelFrame(parent, text="Voice (text-to-speech)", padding=8)
        frame.pack(fill="x", pady=(0, 8))

        voices = self._gc.available_voices()
        controls = ttk.Frame(frame)
        controls.pack(fill="x")

        ttk.Label(controls, text="Voice:").pack(side="left")
        self._voice_var = tk.StringVar(value=voices[0] if voices else "")
        # Read-only combobox so the owner can only pick a real on-disk voice.
        self._voice_combo = ttk.Combobox(
            controls, textvariable=self._voice_var, values=voices,
            state="readonly" if voices else "disabled", width=16,
        )
        self._voice_combo.pack(side="left", padx=(4, 0))

        button_state = "normal" if voices else "disabled"
        self._speak_btn = ttk.Button(
            controls, text="Speak test", command=self._on_speak, state=button_state
        )
        self._speak_btn.pack(side="left", padx=(6, 0))
        self._audition_btn = ttk.Button(
            controls, text="Audition all", command=self._on_audition, state=button_state
        )
        self._audition_btn.pack(side="left", padx=(6, 0))

        initial = "ready" if voices else "no voices configured (set paths.kokoro_voices)"
        self._tts_status_var = tk.StringVar(value=initial)
        ttk.Label(frame, textvariable=self._tts_status_var, foreground="#333").pack(
            anchor="w", pady=(8, 0)
        )

    def _on_speak(self) -> None:
        """Speak a fixed test line in the selected voice (routes to the ops worker)."""
        voice = self._voice_var.get()
        self._tts_status_var.set(f"speaking test in {voice} ...")
        self._gc.request_speak("locitize voice test. This is the selected voice.", voice)

    def _on_audition(self) -> None:
        """Audition every on-disk voice in sequence (routes to the ops worker)."""
        self._tts_status_var.set("auditioning all voices ...")
        self._gc.request_audition()

    def _append_transcript(self, line: str) -> None:
        """Append one accepted transcript segment to the read-only box."""
        self._transcript_box.config(state="normal")
        self._transcript_box.insert("end", line + "\n")
        self._transcript_box.see("end")
        self._transcript_box.config(state="disabled")

    def _clear_transcript(self) -> None:
        self._transcript_box.config(state="normal")
        self._transcript_box.delete("1.0", "end")
        self._transcript_box.config(state="disabled")

    def _build_talk_panel(self, parent: ttk.Frame) -> None:
        """Panel 7 - the Talk/assistant conversation INSIDE the window (M10.3).

        The centerpiece: Start/End assistant session controls, a Talk push-to-talk
        button (click-to-talk) with a state label, an Interrupt (barge-in) button, and
        a read-only scrollable conversation view showing "You:"/"LOCITIZE:" per completed
        turn. Presentation only: every control routes through gui_controller intents on
        the ops worker or a non-blocking talk-gate put; the pump is the sole writer of
        the conversation view (G1/G2). The Talk button is disabled until a session is
        started and reflects idle/listening/thinking/speaking via the on_speaking hook
        and GuiSttSource marshalling.
        """
        frame = ttk.LabelFrame(parent, text="Talk to locitize (assistant)", padding=8)
        frame.pack(fill="both", expand=True, pady=(0, 8))

        controls = ttk.Frame(frame)
        controls.pack(fill="x")
        self._assistant_start_btn = ttk.Button(
            controls, text="Start assistant", command=self._on_start_assistant
        )
        self._assistant_start_btn.pack(side="left")
        self._assistant_end_btn = ttk.Button(
            controls, text="End assistant", command=self._on_end_assistant, state="disabled"
        )
        self._assistant_end_btn.pack(side="left", padx=(6, 0))
        # The Talk button starts disabled: no session yet. Its label is the live state.
        self._talk_var = tk.StringVar(value="Talk")
        self._talk_btn = ttk.Button(
            controls, textvariable=self._talk_var, command=self._on_talk, state="disabled"
        )
        self._talk_btn.pack(side="left", padx=(6, 0))
        self._interrupt_btn = ttk.Button(
            controls, text="Interrupt", command=self._on_interrupt, state="disabled"
        )
        self._interrupt_btn.pack(side="left", padx=(6, 0))
        # Speak toggle: spoken vs text-only. Reuses the Panel 6 voice selection.
        self._assistant_speak_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls, text="Speak replies", variable=self._assistant_speak_var
        ).pack(side="left", padx=(6, 0))

        self._assistant_status_var = tk.StringVar(value="assistant not started")
        ttk.Label(frame, textvariable=self._assistant_status_var, foreground="#333").pack(
            anchor="w", pady=(6, 0)
        )

        # Read-only, scrollable conversation view (state disabled so it is never typed
        # into). The pump appends one "You:"/"LOCITIZE:" line per completed turn.
        self._conversation = scrolledtext.ScrolledText(
            frame, height=8, state="disabled", wrap="word"
        )
        self._conversation.pack(fill="both", expand=True, pady=(6, 0))
        # Session-live flag drives the model-exclusive buttons (Vision/Benchmark).
        self._assistant_live = False

    def _build_vision_panel(self, parent: ttk.Frame) -> None:
        """M10.5 Vision panel - pick an image, ask, and show the real model answer.

        A file picker (filedialog), a prompt entry, a Describe button, and a read-only
        result view. Routes through gui_controller.request_describe -> the existing
        --describe path. Disabled while a Talk session is live (it switches the model).
        """
        frame = ttk.LabelFrame(parent, text="Vision (describe an image)", padding=8)
        frame.pack(fill="x", pady=(0, 8))

        controls = ttk.Frame(frame)
        controls.pack(fill="x")
        self._vision_path_var = tk.StringVar(value="")
        self._vision_pick_btn = ttk.Button(
            controls, text="Pick image...", command=self._on_pick_image
        )
        self._vision_pick_btn.pack(side="left")
        ttk.Label(controls, textvariable=self._vision_path_var, foreground="#666").pack(
            side="left", padx=(6, 0)
        )

        prompt_row = ttk.Frame(frame)
        prompt_row.pack(fill="x", pady=(6, 0))
        ttk.Label(prompt_row, text="Prompt:").pack(side="left")
        self._vision_prompt_var = tk.StringVar(value="")
        ttk.Entry(prompt_row, textvariable=self._vision_prompt_var).pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )
        self._vision_btn = ttk.Button(
            prompt_row, text="Describe", command=self._on_describe, state="disabled"
        )
        self._vision_btn.pack(side="left", padx=(6, 0))

        self._vision_result = scrolledtext.ScrolledText(
            frame, height=4, state="disabled", wrap="word"
        )
        self._vision_result.pack(fill="x", pady=(6, 0))

    def _build_memory_panel(self, parent: ttk.Frame) -> None:
        """M10.5 Memory panel - search past conversations (read-only, always available).

        A search entry + a read-only results list. Routes through
        gui_controller.request_memory_search -> ConversationMemory.search/recall. A
        blank query returns the recent tail. No model, no network, no write.
        """
        frame = ttk.LabelFrame(parent, text="Memory (past conversations)", padding=8)
        frame.pack(fill="x", pady=(0, 8))

        controls = ttk.Frame(frame)
        controls.pack(fill="x")
        ttk.Label(controls, text="Search:").pack(side="left")
        self._memory_query_var = tk.StringVar(value="")
        entry = ttk.Entry(controls, textvariable=self._memory_query_var)
        entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        entry.bind("<Return>", lambda _e: self._on_memory_search())
        ttk.Button(controls, text="Search", command=self._on_memory_search).pack(
            side="left", padx=(6, 0)
        )

        self._memory_result = scrolledtext.ScrolledText(
            frame, height=4, state="disabled", wrap="word"
        )
        self._memory_result.pack(fill="x", pady=(6, 0))

    # ---- M10 Talk / Vision / Memory / Benchmark handlers (UI thread) ------ #

    def _on_start_assistant(self) -> None:
        """Start the assistant session with the selected voice + speak toggle."""
        voice = self._voice_var.get()
        speak = bool(self._assistant_speak_var.get())
        self._assistant_status_var.set("starting services...")
        self._assistant_start_btn.config(state="disabled")
        self._gc.request_start_assistant(voice, speak)

    def _on_end_assistant(self) -> None:
        """End the assistant session; controls reset when assistant_ended arrives."""
        self._assistant_status_var.set("ending session...")
        self._assistant_end_btn.config(state="disabled")
        self._talk_btn.config(state="disabled")
        self._interrupt_btn.config(state="disabled")
        self._gc.request_end_assistant()

    def _on_talk(self) -> None:
        """Open one capture window (click-to-talk). Non-blocking talk-gate put."""
        self._gc.request_talk()

    def _on_interrupt(self) -> None:
        """Barge-in: stop the current reply's generation + audio."""
        self._gc.request_interrupt()

    def _on_benchmark(self) -> None:
        """Benchmark the selected model (single config) and refresh its Score column."""
        model = self._selected_model()
        if model is None:
            self._status_var.set("select a model to benchmark")
            return
        self._status_var.set(f"benchmarking {model['id']} ...")
        self._benchmark_btn.config(state="disabled")
        self._gc.request_benchmark(model["id"])

    def _on_pick_image(self) -> None:
        """Open a file picker for a local image; store the chosen path."""
        path = filedialog.askopenfilename(
            title="Pick an image to describe",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._vision_path_var.set(path)
            self._vision_btn.config(state="normal" if not self._assistant_live else "disabled")

    def _on_describe(self) -> None:
        """Describe the picked image with the optional prompt (routes to ops worker)."""
        path = self._vision_path_var.get()
        if not path:
            self._set_readonly(self._vision_result, "pick an image first")
            return
        self._set_readonly(self._vision_result, "describing (switching to the vision model)...")
        self._vision_btn.config(state="disabled")
        self._gc.request_describe(path, self._vision_prompt_var.get())

    def _on_memory_search(self) -> None:
        """Search past conversations (blank = recent tail)."""
        self._set_readonly(self._memory_result, "searching...")
        self._gc.request_memory_search(self._memory_query_var.get())

    def _append_conversation(self, prefix: str, text: str) -> None:
        """Append one "You:"/"LOCITIZE:" turn line to the read-only conversation view."""
        self._conversation.config(state="normal")
        self._conversation.insert("end", f"{prefix}{text}\n")
        self._conversation.see("end")
        self._conversation.config(state="disabled")

    @staticmethod
    def _set_readonly(widget: "scrolledtext.ScrolledText", text: str) -> None:
        """Replace a read-only ScrolledText's contents with `text` (UI thread only)."""
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

    def _build_footer(self, parent: ttk.Frame) -> None:
        ttk.Separator(parent, orient="horizontal").pack(fill="x")
        ttk.Label(parent, textvariable=self._health_var).pack(anchor="w", pady=(4, 0))

    # ---- rendering helpers (UI thread only) ------------------------------ #

    def _health_line(self) -> str:
        running = self._ui.running_model_id or "(none)"
        port = self._ui.running_port
        port_text = f" port {port}" if port else ""
        tps = "-"
        ctx = "-"
        sample = self._ui.latest_metrics
        if sample is not None and sample.metrics_available:
            tps = "idle" if not sample.gen_tokens_s else f"{sample.gen_tokens_s:.1f} tok/s"
            # Reuse the monitor's honest ctx formatter so the footer matches Panel 4
            # (and picks up n_tokens_max/n_ctx on builds without kv_cache_* metrics).
            ctx = gui_controller._format_ctx(sample, self._selected_context_size())
        return f"Health: {self._health}  |  model: {running}{port_text}  |  {tps}  |  ctx {ctx}"

    def _selected_context_size(self) -> int | None:
        """The running model's configured context_size, used as the ctx denominator.

        Prefers the running model (the one the metrics describe); falls back to the
        selected row so the footer still has a sensible total between runs.
        """
        target = self._ui.running_model_id or self._selected_id
        for model in self._models:
            if model["id"] == target:
                return model.get("context_size")
        return None

    def _refresh_model_rows(self) -> None:
        """Rebuild the model tree from the cached model list + current UiState.

        Rows are rendered in the current sort order (owner request 3); the selected
        row (self._selected_id) is re-selected after the rebuild so a resort never
        loses the owner's selection.
        """
        rows = self._models
        if self._sort_col is not None:
            rows = gui_controller.sort_model_rows(rows, self._sort_col, self._sort_desc)
        for item in self._tree.get_children():
            self._tree.delete(item)
        for model in rows:
            status = self._status_for(model)
            self._tree.insert(
                "",
                "end",
                iid=model["id"],
                values=(
                    model["name"],
                    model["id"],
                    model["size_display"],
                    model["score_display"],
                    status,
                ),
            )
        if self._selected_id and self._tree.exists(self._selected_id):
            self._tree.selection_set(self._selected_id)

    def _on_sort(self, column: str) -> None:
        """Header click: sort by `column`, toggling ascending/descending on repeat.

        Size sorts numerically with missing sizes last (gui_controller.sort_model_rows);
        the selected row is preserved because _refresh_model_rows re-selects it.
        """
        if self._sort_col == column:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = column
            self._sort_desc = False
        self._refresh_model_rows()

    def _status_for(self, model: dict) -> str:
        if not model["launchable"]:
            return "location not set" if not model["location"] else "future"
        if self._ui.running_model_id == model["id"]:
            port = self._ui.running_port
            return f"RUNNING (port {port})" if port else "RUNNING"
        if self._ui.in_flight and self._selected_id == model["id"]:
            return "STARTING"
        return "STOPPED"

    def _refresh_buttons(self) -> None:
        """Apply the button-state machine (gui_controller) to the widgets."""
        states = gui_controller.compute_button_states(self._ui)
        self._start_btn.config(state="normal" if states["start_enabled"] else "disabled")
        self._stop_btn.config(state="normal" if states["stop_enabled"] else "disabled")
        self._chat_btn.config(state="normal" if states["chat_enabled"] else "disabled")
        model = self._selected_model()
        if model is not None:
            label = gui_controller.start_label_for(model["id"], self._ui)
            # Disable Start on the running model (its label reads "Running") and on
            # an unlaunchable model, honestly showing why it cannot start.
            if not model["launchable"]:
                self._start_btn.config(state="disabled", text="Start")
            else:
                self._start_btn.config(text=label)
                if label == "Running":
                    self._start_btn.config(state="disabled")
        self._health_var.set(self._health_line())

    def _selected_model(self) -> dict | None:
        for model in self._models:
            if model["id"] == self._selected_id:
                return model
        return None

    # ---- event handlers (enqueue only, never block) ---------------------- #

    def _on_select_model(self, _event: object) -> None:
        selection = self._tree.selection()
        if not selection:
            return
        self._selected_id = selection[0]
        model = self._selected_model()
        if model is not None:
            self._gpu_var.set(str(model["gpu_layers"]))
            self._ctx_var.set(str(model["context_size"]))
        self._on_edit_change()
        self._refresh_buttons()

    def _on_start(self) -> None:
        model = self._selected_model()
        if model is None:
            self._status_var.set("select a model first")
            return
        if not model["launchable"]:
            self._status_var.set("that model is not launchable (no location set)")
            return
        self._ui.in_flight = True
        self._status_var.set(f"working... starting {model['name']}")
        self._refresh_buttons()
        self._gc.request_start(model["id"])

    def _on_stop(self) -> None:
        self._ui.in_flight = True
        self._status_var.set("working... stopping")
        self._refresh_buttons()
        self._gc.request_stop()

    def _on_chat(self) -> None:
        # M9-lite: route through the chooser. gui_controller decides (pure
        # resolve_chat_choice) and either opens directly or asks us to render a
        # dialog via a chat_ask / chat_offer_start Result.
        self._gc.request_chat()

    def _show_chat_chooser(self) -> None:
        """Modal ASK dialog: pick a chat UI, optionally remember it (M9.3).

        Presentation only: the two buttons and the "remember" checkbox route the
        selection back through gui_controller.open_chat(choice, remember). No
        subprocess/HTTP/yaml here (G1). A small Toplevel centered over the main
        window; Escape/close cancels without opening anything.
        """
        dialog = tk.Toplevel(self._root)
        dialog.title("Choose a chat UI")
        dialog.transient(self._root)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=16)
        frame.grid(row=0, column=0)
        ttk.Label(
            frame, text="Which chat interface would you like to open?"
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        remember_var = tk.BooleanVar(value=False)

        def choose(choice: str) -> None:
            remember = bool(remember_var.get())
            dialog.destroy()
            self._gc.open_chat(choice, remember)

        ttk.Button(
            frame,
            text="llama.cpp web UI (built-in, zero setup)",
            command=lambda: choose("llamacpp"),
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(
            frame,
            text="Open WebUI (rich chat, history)",
            command=lambda: choose("openwebui"),
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Checkbutton(
            frame, text="Remember my choice", variable=remember_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self._center_dialog(dialog)

    def _show_openwebui_start_offer(self, reason: str) -> None:
        """Modal offer dialog: Open WebUI is installed but not running (M9.3).

        A Start button enqueues the start on the ops worker thread (never the UI
        thread - the first boot can take minutes) via
        gui_controller.start_openwebui_and_open; a Cancel just closes. Presentation
        only.
        """
        dialog = tk.Toplevel(self._root)
        dialog.title("Start Open WebUI?")
        dialog.transient(self._root)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=16)
        frame.grid(row=0, column=0)
        message = reason or "Open WebUI is installed but not running."
        ttk.Label(frame, text=message).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            frame,
            text="Starting it migrates its database on first run (up to 300s).",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 12))
        remember_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="Remember my choice", variable=remember_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 12))

        def start() -> None:
            remember = bool(remember_var.get())
            dialog.destroy()
            self._status_var.set(
                "starting Open WebUI (first run migrates its database; up to 300s)..."
            )
            self._gc.start_openwebui_and_open(remember)

        ttk.Button(frame, text="Start Open WebUI", command=start).grid(
            row=3, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(frame, text="Cancel", command=dialog.destroy).grid(
            row=3, column=1, sticky="ew"
        )
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self._center_dialog(dialog)

    def _center_dialog(self, dialog: tk.Toplevel) -> None:
        """Center a Toplevel over the main window and give it modal grab."""
        dialog.update_idletasks()
        try:
            root_x = self._root.winfo_rootx()
            root_y = self._root.winfo_rooty()
            root_w = self._root.winfo_width()
            root_h = self._root.winfo_height()
            win_w = dialog.winfo_width()
            win_h = dialog.winfo_height()
            x = root_x + max(0, (root_w - win_w) // 2)
            y = root_y + max(0, (root_h - win_h) // 2)
            dialog.geometry(f"+{x}+{y}")
        except tk.TclError:
            pass
        dialog.grab_set()

    def _on_whisper(self) -> None:
        self._status_var.set("working... toggling whisper server")
        self._gc.request_whisper_toggle()

    def _on_listen(self) -> None:
        self._status_var.set("listening for 15s - speak now")
        self._voice_status_var.set("listening for 15s - speak now")
        self._clear_transcript()
        self._gc.request_listen(15.0)

    def _on_edit_change(self) -> None:
        """Recompute the Save button state from the editors vs the saved values.

        Owner request 1: Save is DISABLED when the editors match the saved registry
        values and ENABLED the moment either differs; invalid input keeps Save
        enabled and shows the validation error. No "* unsaved" marker - the button
        state is the sole indicator.
        """
        model = self._selected_model()
        if model is None:
            self._save_btn.config(state="disabled")
            self._edit_error_var.set("")
            return
        state = gui_controller.compute_save_state(
            self._gpu_var.get(),
            self._ctx_var.get(),
            model["gpu_layers"],
            model["context_size"],
        )
        self._save_btn.config(state="normal" if state["enabled"] else "disabled")
        self._edit_error_var.set(state["error"] or "")

    def _on_save(self) -> None:
        model = self._selected_model()
        if model is None:
            return
        self._gc.save_model_edits(model["id"], self._gpu_var.get(), self._ctx_var.get())

    # ---- the pump: sole writer of widgets from background outcomes -------- #

    def run_ui(self) -> None:
        """Launch the background threads and enter the pump + Tk mainloop."""
        self._gc.start_threads()
        self._pump()
        self._root.mainloop()

    def _pump(self) -> None:
        self._root.after(100, self._pump)
        self._drain()

    def _drain(self) -> None:
        """Apply every queued Result to the widgets (runs on the UI thread)."""
        while True:
            try:
                result = self._gc.result_q.get_nowait()
            except queue.Empty:
                break
            self._apply(result)

    def _apply(self, result: gui_controller.Result) -> None:
        kind = result.kind
        if kind == "start":
            self._ui.in_flight = False
            # O-2: reflect the controller's ACTUAL running state from the Result
            # payload, not an optimistic assumption. A failed switch may leave a
            # different model (or nothing) running; the snapshot is the truth.
            self._apply_running_snapshot(result.payload)
            if result.ok:
                self._status_var.set(f"running {self._ui.running_model_id}")
            else:
                self._status_var.set(result.error or "start failed")
            self._refresh_model_rows()
            self._refresh_buttons()
        elif kind == "stop":
            self._ui.in_flight = False
            # O-2: trust the post-stop snapshot rather than blindly clearing; a clean
            # stop reports nothing running, keeping the model list honest.
            self._apply_running_snapshot(result.payload)
            self._ui.latest_metrics = None
            self._monitor_var.set("idle")
            self._monitor_reason_var.set("")
            self._status_var.set("stopped" if result.ok else (result.error or "stop failed"))
            self._refresh_model_rows()
            self._refresh_buttons()
        elif kind == "metrics":
            self._render_metrics(result.payload.get("sample"))
        elif kind == "whisper":
            self._ui.whisper_running = bool(result.payload.get("running"))
            self._whisper_btn.config(
                text="Stop whisper server" if self._ui.whisper_running else "Start whisper server"
            )
            self._status_var.set(result.error or ("whisper running" if self._ui.whisper_running else "whisper stopped"))
        elif kind == "transcript":
            # Status line shows the latest segment; the box keeps the full run
            # (owner request 2026-07-19: transcript visible again for verification).
            line = result.payload.get("line", "")
            if line:
                self._voice_status_var.set(line)
                self._append_transcript(line)
        elif kind == "listen":
            self._status_var.set("listen finished" if result.ok else (result.error or "listen failed"))
            if not result.ok:
                self._voice_status_var.set(result.error or "listen failed")
        elif kind == "speak":
            voice = result.payload.get("voice", "")
            if result.ok:
                self._tts_status_var.set(f"spoke in {voice}")
            else:
                self._tts_status_var.set(result.error or "speak failed")
        elif kind == "audition_voice":
            # Per-voice progress line as the audition walks the on-disk voices.
            self._tts_status_var.set(f"speaking: {result.payload.get('voice', '')}")
        elif kind == "audition":
            if result.ok:
                self._tts_status_var.set(f"auditioned {result.payload.get('count', 0)} voices")
            else:
                self._tts_status_var.set(result.error or "audition failed")
        elif kind == "save_edits":
            if result.ok:
                model = self._selected_model()
                if model is not None and model["id"] == result.payload.get("model_id"):
                    model["gpu_layers"] = result.payload.get("gpu_layers", model["gpu_layers"])
                    model["context_size"] = result.payload.get("context_size", model["context_size"])
                # Owner request 1: recompute the Save state. The editors now match the
                # saved values, so this greys Save out - and that grey-out IS the save
                # confirmation. No "saved; applies on next start" notice.
                self._edit_error_var.set("")
                self._on_edit_change()
            else:
                self._edit_error_var.set(result.error or "save failed")
        elif kind == "chat":
            if result.ok and result.payload.get("opened", True) is False:
                messagebox.showinfo("locitize chat", f"Open this in your browser:\n{result.payload.get('url')}")
            elif result.ok and result.payload.get("reason"):
                # An honest degrade (Open WebUI missing/not ready): opened the
                # built-in UI but tell the owner why the fallback happened.
                self._status_var.set(result.payload["reason"])
            elif not result.ok:
                self._status_var.set(result.error or "cannot open chat")
        elif kind == "chat_ask":
            # M9-lite: present the two-UI choice with a "remember" checkbox.
            self._show_chat_chooser()
        elif kind == "chat_offer_start":
            # M9-lite: Open WebUI is installed but not running - offer to start it.
            self._show_openwebui_start_offer(result.payload.get("reason", ""))
        elif kind == "assistant_started":
            self._on_assistant_started()
        elif kind == "assistant_error":
            self._on_assistant_error(result.error or "assistant error")
        elif kind == "assistant_ended":
            self._on_assistant_ended()
        elif kind == "assistant_state":
            self._render_talk_state(result.payload.get("state", "idle"))
        elif kind == "assistant_user":
            self._append_conversation("You: ", result.payload.get("text", ""))
        elif kind == "assistant_reply":
            # The loop emits the line already prefixed ("LOCITIZE: ..."); show it as-is.
            self._append_conversation("", result.payload.get("line", ""))
        elif kind == "describe":
            if result.ok:
                self._set_readonly(self._vision_result, result.payload.get("answer", ""))
            else:
                self._set_readonly(self._vision_result, result.error or "describe failed")
            self._refresh_model_exclusive_buttons()
        elif kind == "memory_search":
            self._render_memory_hits(result)
        elif kind == "benchmark":
            self._on_benchmark_result(result)

    # ---- M10 assistant/vision/memory/benchmark rendering (UI thread) ----- #

    def _on_assistant_started(self) -> None:
        """Session is live: enable Talk/Interrupt/End, disable model-exclusive actions."""
        self._assistant_live = True
        self._assistant_status_var.set("assistant ready - click Talk and speak")
        self._render_talk_state("idle")
        self._assistant_end_btn.config(state="normal")
        self._interrupt_btn.config(state="normal")
        self._refresh_model_exclusive_buttons()

    def _on_assistant_error(self, message: str) -> None:
        """Honest start/session failure: reset the controls and show the remedy."""
        self._assistant_live = False
        self._assistant_status_var.set(message)
        self._assistant_start_btn.config(state="normal")
        self._assistant_end_btn.config(state="disabled")
        self._talk_btn.config(state="disabled")
        self._interrupt_btn.config(state="disabled")
        self._refresh_model_exclusive_buttons()

    def _on_assistant_ended(self) -> None:
        """Session ended cleanly: reset every assistant control to the not-started state."""
        self._assistant_live = False
        self._assistant_status_var.set("assistant session ended")
        self._talk_var.set("Talk")
        self._assistant_start_btn.config(state="normal")
        self._assistant_end_btn.config(state="disabled")
        self._talk_btn.config(state="disabled")
        self._interrupt_btn.config(state="disabled")
        self._refresh_model_exclusive_buttons()

    def _render_talk_state(self, state: str) -> None:
        """Set the Talk button label + enablement from the assistant state (M10.3).

        idle -> "Talk" (enabled); listening/thinking/speaking -> a labelled, disabled
        button. The button is only enabled while the session is live and idle.
        """
        labels = {
            "idle": "Talk",
            "listening": "Listening... speak now",
            "thinking": "Thinking...",
            "speaking": "Speaking...",
        }
        self._talk_var.set(labels.get(state, "Talk"))
        if self._assistant_live and state == "idle":
            self._talk_btn.config(state="normal")
        else:
            self._talk_btn.config(state="disabled")

    def _refresh_model_exclusive_buttons(self) -> None:
        """Enable/disable Vision + Benchmark by whether a Talk session holds the model.

        Both switch/run the model exclusively, so they are disabled while a session is
        live and re-enabled when it ends (M10.5). Vision's Describe also needs an image.
        """
        exclusive_state = "disabled" if self._assistant_live else "normal"
        self._benchmark_btn.config(state=exclusive_state)
        if self._assistant_live or not self._vision_path_var.get():
            self._vision_btn.config(state="disabled")
        else:
            self._vision_btn.config(state="normal")

    def _render_memory_hits(self, result: gui_controller.Result) -> None:
        """Render the read-only memory search results (honest empty state)."""
        if not result.ok:
            self._set_readonly(self._memory_result, result.error or "memory search failed")
            return
        hits = result.payload.get("hits", [])
        if not hits:
            query = result.payload.get("query", "")
            what = f"'{query}'" if query else "recent conversations"
            self._set_readonly(self._memory_result, f"no stored conversation matched {what}.")
            return
        lines = [f"{h.get('role', '?')}: {h.get('text', '')}" for h in hits]
        self._set_readonly(self._memory_result, "\n".join(lines))

    def _on_benchmark_result(self, result: gui_controller.Result) -> None:
        """Apply a benchmark outcome: refresh generation tok/s and status."""
        self._refresh_model_exclusive_buttons()
        model_id = result.payload.get("model_id", "")
        if not result.ok:
            self._status_var.set(result.error or "benchmark failed")
            return
        # Update the cached row immediately; the same throughput is already in the
        # append-only benchmark report and will be restored on the next launch.
        for model in self._models:
            if model["id"] == model_id:
                model["benchmark_tok_s"] = result.payload.get("score")
                model["score_display"] = result.payload.get("score_display", "-")
                break
        self._refresh_model_rows()
        self._status_var.set(f"benchmarked {model_id}: {result.payload.get('detail', '')}")

    def _apply_running_snapshot(self, payload: dict) -> None:
        """Set UiState's running model/port from a lifecycle Result payload (O-2).

        The ops worker stamps every start/stop Result with the controller's real
        running state (_running_snapshot_payload). Rendering from that keeps the
        model list honest even after a partial failure, instead of the GUI guessing.
        """
        self._ui.running_model_id = payload.get("running_model_id")
        self._ui.running_port = payload.get("running_port")

    def _render_metrics(self, sample: object) -> None:
        """Update Panel 4 from a MetricsSample, hiding it honestly when absent (G5)."""
        if sample is None:
            return
        self._ui.latest_metrics = sample  # type: ignore[assignment]
        if not getattr(sample, "metrics_available", False) and getattr(sample, "slots", None) is None:
            # Neither /metrics nor /slots available: hide the panel with a reason.
            self._monitor_var.set("")
            self._monitor_reason_var.set(
                "live metrics unavailable - this llama-server build does not expose "
                "/metrics; model start is unaffected"
            )
            self._health_var.set(self._health_line())
            return
        # Owner request 4: render counts + context fill, not just rates, via the
        # shared honest formatter (idle/"-" for absent sources; never fabricated).
        self._monitor_reason_var.set("")
        self._monitor_var.set(
            gui_controller.format_monitor_line(sample, self._selected_context_size())
        )
        self._health_var.set(self._health_line())

    # ---- shutdown -------------------------------------------------------- #

    def _on_close(self) -> None:
        """WM_DELETE_WINDOW: stop the threads + services, then destroy the window."""
        self._status_var.set("shutting down...")
        self._root.update_idletasks()
        self._gc.shutdown()
        self._root.destroy()


def run(controller: gui_controller.GuiController, health: str = "unknown") -> int:
    """Build the Tk root and run the command center. Returns a process exit code.

    Kept tiny so the launcher's --gui path is a one-liner. Any Tk instantiation
    failure (a truly headless machine) surfaces to the launcher, which prints a
    remedy; the automated test suite never calls this.
    """
    root = tk.Tk()
    app = LocitizeGui(root, controller, health)
    app.run_ui()
    return 0
