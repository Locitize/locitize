# LOCITIZE Platform Changelog

## 0.2.0-beta.1 - unified desktop candidate - 2026-09-07

- Added Home and native Sessions: seven local readers, search, previews,
  annotations, archive/restore, export and non-destructive legacy import.
- Added model/project-aware local coding resume with per-process configuration.
- Protected coding sessions against conflicting model switches and GUI shutdown.
- Added a bundled Windows runtime, native launcher, integrity-checked versioned
  installer, manifests and checksums. Optional model/voice/chat stacks stay separate.
- Corrected privacy claims and retired the stale milestone-1 deferred roadmap.
- Status remains unsigned internal beta; see release-readiness.md for open gates.

## Voice interrupt in Call - 2026-09-04

**Status:** Implemented. Speak over the reply to stop it.

- **Open WebUI Call:** `voiceInterruption` defaults on. The microphone stays
  open while Kokoro speaks. Browser echoCancellation is already requested.
  Speaker-phone echo can still false-trigger; a headset is more reliable.
  `openwebui.voice_interruption: false` restores half-duplex.
- **Talk:** after each answer the desktop listens again, so the next turn does
  not need a tap. Click Talk (or Space) still cuts in immediately.

## Phone URL on Chat when Tailscale Serve is already configured - 2026-09-04

**Status:** Implemented. Discovery is a local `tailscale serve status --json`
read. LOCITIZE still binds loopback only; it does not start Serve, Funnel, or
a LAN listener.

- **Chat page:** when Serve already proxies Open WebUI, the Chat card shows
  that HTTPS URL and a Copy button. Hint: same tailnet only; open Tailscale on
  the phone first; Call mode stays half-duplex on speakers.
- **No new network path.** Missing CLI, stopped daemon, or Serve pointing
  elsewhere hides the strip.

## Talk as presence, barge-in, and desktop UI pass - 2026-09-04

**Status:** Implemented. 1664 tests passed, 2 skipped.

- **Talk is the mic:** Click Talk to start the assistant and listen. Click Talk
  again while it is thinking or speaking to interrupt and listen (ChatGPT-style
  barge-in). Space on the Talk page does the same. The Interrupt button remains.
  Always-open-mic while speakers play is still off (half-duplex gate).
- **First sentence is audible immediately:** Kokoro queues each sentence as it
  completes instead of waiting for the full reply. Lead pad is 120 ms on the
  first clip of a burst (Bluetooth speakers can still raise `tts.reply_lead_silence_ms`).
  End-of-speech settle is 0.7 s.
- **Desktop:** Talk states are visible; voices show as Heart/Bella not `af_heart`;
  Vision shows the image and accepts a drop; Memory lists recent on first visit;
  Settings uses human labels; Chat names the running model; sidebar chip is
  LOCITIZE; Offload GPU hides on Talk. Voice Setup has Off/Balanced/Strong for
  Open WebUI noise suppression and rebinds the live processor.
- **Watch my screen** is a Vision panel (goal + Start/Stop) over the existing
  second-eye loop. Refuses while Talk is live.

## Local microphone noise suppression - 2026-09-04

**Status:** Review, Security, functional QA, performance, and all six acceptance
criteria passed on the reference machine.

- **Default-on local filtering:** Open WebUI microphone and Call-mode uploads now
  pass through a fixed FFmpeg `highpass`/`lowpass`/`afftdn`/`agate` preset before
  the loopback whisper-server. `speech.noise_suppression` accepts `balanced`,
  `strong`, or byte-identical `off`; arbitrary filter syntax is refused.
- **Fail-closed request boundary:** Missing FFmpeg, invalid audio, timeout, and
  unexpected callback failures return typed 503/422/504/502 responses without
  reflecting paths or diagnostics. A failed request does not stop the router.
- **Whisper defense in depth:** New whisper-server launches include
  `--suppress-nst`; native Silero VAD remains off because the current Windows
  server has a reported zero-speech exit defect.
- **Measured result:** Balanced removed 6.19 dB of deterministic steady noise,
  retained 0.936 voice-band RMS, and measured 55.16 ms conservative p95 over
  five warm four-second clips against a 500 ms budget.
- **Real functional proof:** Balanced transcribed the representative noisy known
  phrase exactly, strong recovered the exact phrase under the harsher stress
  mix, off forwarded identical bytes, corrupt input returned a safe 422, the next
  valid request succeeded, and text chat plus Kokoro TTS remained functional.
- **Regression:** 1,658 automated tests passed with 2 skipped; the focused
  security/privacy/write-fence suite passed 324 tests. No dependency, cloud
  processor, remote bind, UI control, commit, push, or deployment was added.
- **Known boundary:** The feature targets steady noise and rumble, not another
  intelligible speaker. Strong may weaken very quiet consonants and uses the
  owner-installed FFmpeg resolved from `PATH`.

## Open WebUI voice conversation verified - 2026-09-03

**Status:** Working end to end on the reference machine and covered by the full suite.

- **Open WebUI TTS persistence fixed:** Open WebUI 0.11.1 reads the provider-specific `audio.tts.openai.api_key`; LOCITIZE had persisted its non-secret local placeholder under the unrelated generic TTS key. Startup reconciliation now writes the key OpenAI-compatible TTS actually reads.
- **Live browser proof:** Open WebUI captured two microphone turns, the LOCITIZE router sent them through whisper.cpp, GPT-OSS answered both, and Kokoro speech requests returned HTTP 200. The Call overlay remained active with mute, interruption and end-call controls.
- **Direct pipeline proof:** Kokoro produced a 189 KB WAV in 0.19 seconds, whisper.cpp transcribed it in 0.42 seconds, GPT-OSS answered a verification prompt in 2.85 seconds, and Kokoro rendered that reply in 0.12 seconds.
- **Phone-safe turn-taking:** voice interruption defaults off so speaker echo cannot stop and resubmit the assistant's own speech. Headset users can opt into interruption in their personal Open WebUI interface settings.
- **Mobile playback repair:** Open WebUI Call Mode now plays the fetched TTS `Audio` object directly instead of copying it into a gesture-locked shared element. A mobile autoplay rejection is retried on the next tap/key gesture, and ending the call resolves any pending playback instead of wedging later replies. The managed PWA version changes with the patched bundle so phones fetch the repair.
- **Long-call microphone repair:** Open WebUI 0.11.1 leaked one browser `AudioContext` each time it re-armed the recorder. After several good turns, a mobile browser could exhaust its live-context allowance and stop detecting speech while Whisper and every server stayed healthy. Startup reconciliation now closes the previous analyser context before creating the next one.

## Model choice is request-scoped - 2026-09-03

**Status:** Implemented with persisted-state clearing and headless coverage.

**Owner correction:** "Open WebUI should be taking preference" means its live picker selection, not a saved global model default.

- **No saved default:** LOCITIZE clears `launcher.default_model` through its targeted settings writer and removes Open WebUI's `ui.default_models` row during startup reconciliation.
- **Request authority:** the existing router switches on the first request after an in-page model choice. Merely opening Open WebUI never starts or replaces a model.

## The voice engine's footprint, and the fit margin, are constants - 2026-09-03

**Status:** Implemented, measured on the reference machine (RTX 5070 Ti 16 GB, CUDA torch, llama.cpp b10701), suite green.

**Owner rules:** "do not affect my tok/s"; "have LOCITIZE load as much to the GPU". The measured fit margin shipped this morning with a 256 MiB safety was swept in one-layer steps against the dense 27B with 1580 MB held by other processes: 64 layers 4.6 tok/s, 63 layers 6.7 (646 MB paged), 62 layers 27-29 (536 MB paged), 61 layers 25, 60 layers 26 with 158 MB shared (zero paging, card full), 59 layers 24. The 256 safety chose the 62-layer row: fast because what got paged was cold KV cache, and one layer from a crawl.

- **Kokoro's video memory is bounded and pre-reserved (kokoro_server.py):** torch's caching allocator keeps every block an utterance needed - measured 594 MB after loading, 1026 MB after one sentence, 1842 MB after a six-sentence paragraph handed over as one chunk, 2422 MB after a 1000-character sentence - and on Windows that growth does not fail, it pages the language model through system RAM, permanently: the 27B at 29 tok/s beside an idle Kokoro ran 4.8 tok/s after Kokoro spoke one paragraph and still 4.8 twenty-five seconds after Kokoro had shrunk again. This is the mechanism behind "it chatted, then it did not". Text is now split into sentences and a sentence over 160 characters at its commas, then words (30 sentences rendered one at a time peaked where one did: 1094 MB; 155 characters 1192, 200 characters 1426), and the engine renders one 160-character sentence before `/health` says ready, so the footprint the launcher measures when it fits the model is the footprint Kokoro keeps. Releasing the cache after each utterance was measured and rejected (idle 900 MB, but every sentence spikes back and the model fitted against the idle number pays). Side effect: the first spoken sentence of a call no longer pays the 1s first render. The CPU build is unaffected except for the warm-up.
- **FIT_SAFETY_MIB 256 -> 1024, with the arithmetic on the record (gpu_ledger.py):** dedicated memory never passed ~14100 MB in the sweep - with 1580 MB held elsewhere the card's usable ceiling is ~15.7 GB of 16.3, the rest Windows keeps - and fit's own tally ran a constant ~290 MiB under the process's real footprint (the CUDA context and library workspaces). ~600 + ~290 - ~150 (fit's own launch-time allocations, already in its reading) + one layer of slack for what the desktop allocates during a session = 920, rounded to fit's own default. The measured correction for other processes stays: that is the number fit cannot see. Live after the restart (Kokoro warmed at 1214 MB, 1847 MB held by other processes, fit target 1491): the 27B loads 13420 MB on the GPU with 158 MB shared - zero paging - at 21.4-21.9 tok/s; a six-sentence paragraph, a 1000-character run-on sentence and 600 characters without punctuation through Kokoro took it to 1282 MB and left the model at 158 MB shared and 21.4-21.6 tok/s. The MoE rows gain from the same margin: the 35B loads with 280 MB shared instead of 904 (105-112 tok/s), gpt-oss 220 (118-120 tok/s).
- **Honest accounting:** the 27B at 32.6 tok/s from the morning sweep needed 63 layers on a card where only Kokoro (948 MB) sat beside it; with a warmed Kokoro (1.2 GB) and the desktop (~640 MB) it gets 58-59 layers and ~24 tok/s, and the 28-29 that ran from the desktop this afternoon was the cliff-edge row. The GPU is as full as Windows allows in every one of those rows. More of the 27B on the card means freeing memory: the CPU Kokoro build (+6-7 layers, 0.7s per spoken sentence instead of 0.1) or a shorter context_size (49152 -> 32768 returns three layers). Both are the owner's settings, left as they are.

## Fit the model to the card - 2026-09-03

**Status:** Implemented, measured on the reference machine (RTX 5070 Ti 16 GB, llama.cpp b10701), suite green.

**Owner report:** "When I switch models, and it switches, for some reason Open WebUI does not chat." Reproduced through Open WebUI's own API: a hello on the 27B row took 100s to the first token. The switch had worked; the model had loaded with 782 MB of its weights paged through system RAM (Windows does this instead of failing the allocation), generating at 5 tok/s, and nothing on any surface said so.

- **`gpu_layers: 999` now means fit, not force:** backends.offload_value turns the registry's "everything" sentinel (999, or any negative) into `--n-gpu-layers -1`, which leaves llama-server's built-in `--fit` on: it measures free VRAM at load and keeps whole layers or expert tensors on the CPU. An explicit count still passes through unchanged. Measured on a 2840-token prompt: qwen3-8-27b 5.2 -> 24.3 tok/s (59/66 layers on the card, paged memory 782 -> 158 MB), qwen3-6-35b 10.9 -> 112.3, qwen3-coder-30b 30.5 -> 125.5, gemma-4-e2b unchanged (209 tok/s). With fit's own 1 GiB margin two models that already fit lost speed (gpt-oss-20b 146 -> 134, gemma-4-26b 108 -> 92) and the 27B stopped short of its full-card 32.6; owner's rule "do not affect my tok/s", so the margin was swept (0/256/512/1024 vs 999, three runs each, voice engine holding 948 MB): 256 MiB gave the 27B 32.6/30.8/32.5 tok/s and no crawl in six runs, 35B 111-121, gpt-oss 150 and gemma-4-26b 108 unchanged; 0 re-crawled the 27B. Shipped as a fixed 256 it then crawled from the desktop an hour later (664 MB paged, 6.0 tok/s) - see the next item.
- **fit is blind on Windows, on the record:** the free-memory figure llama-server's fit works from (cudaMemGetInfo) read 14923 MiB in all thirteen logged loads while the voice engine held 0, 594 or 948 MB, and 13293 while another server held 14112 MB (nvidia-smi: 362 free). WDDM pages other processes out from under it. So `gpu_ledger.fit_budget` takes two readings before each fit launch - `llama-server --list-devices` (what fit will see) and nvidia-smi adapter used/total (the truth) - and passes `--fit-target = engine_free - real_free + 256`, floored at 0 (0.4s; launcher.log shows the arithmetic). Nothing measurable means no flag and the engine's own default. The same 999 argv had measured 32.6 tok/s on the 27B with the card otherwise empty and 5.5 with Kokoro loaded: the model chatted before the first spoken turn of a session and not after; a fixed layer count or a fixed margin cannot follow that, a reading can.
- **Measured GPU placement (gpu_ledger.placement_of):** one process's dedicated and shared (paged) GPU memory from the Windows per-process counters; `spilled` from 400 MB shared (the fitted servers carried 124-280 MB, the crawling ones 500-1128). `--smoke-start --json` carries it as `gpu_placement`; every router switch logs one line to launcher.log saying where the new model's memory went and, when it overflowed, what to change. None when it cannot be measured, never a fabricated zero.
- **Found and not changed:** a scheduled task of the owner's calls llama-server directly on 8080 (not the router) with 1024-token non-streaming thinking requests; on `--parallel 1` it holds the only slot, and on the paged 27B one run took 15 minutes during which every chat waited behind it. Reported to the owner; their project, their call.

## M15.4-15.7 - the full-backlog pass - 2026-08-29

**Status:** Implemented and locally verified; the 33-model re-benchmark under the 24-check suite ran alongside this work.

- **Auto-tune throughput floor (M15.4):** --smoke-start now generates once after readiness and reports tokens_per_second; the tune measures a baseline at the current context and treats a candidate that loads but generates below 85% of it as a failure with the measured number in its reason. The load-only search that chose a 294912 context generating at 6% of baseline can no longer do so from the desktop button either.
- **Benchmark lock self-recovery (M15.4):** a lock whose recorded holder PID verifiably no longer exists is reclaimed with an announcement; any doubt and it stays respected.
- **24-check task suite (M15.4):** see benchmark_tasks.py for the authoring rules; 6 checks could not discriminate and a single miss moved a category 50 points.
- **Sharded sets in the GUI (M15.5):** Get models lists and downloads multi-part GGUF sets - whole-set sizes on every dialog, per-part digests enforced, incomplete sets never offered; a set registers under its stem.
- **Wizard completed (M15.6):** live-exercised llama.cpp fetch (which found /releases/latest mislabeled upstream and cudart runtime bundles outscoring real builds - both fixed), whisper server+weights, Kokoro checkpoint+voices, an in-wizard first-model picker with a vision pair (Qwen2.5-VL + mmproj, new catalog entry), post-venv re-detection.
- **Rebrand hygiene (M15.7):** every LOCITIZE_* env override now also answers to LOCITIZE_* (new name wins when both are set; old name keeps working). Root README is a real project README; a never-integrated legacy app (421 files: src/, apps/web, its FastAPI tests, migrations) is untracked from the ship set - files remain on disk and in history. The desktop notices a models.yaml rewritten by another process and offers Refresh in the status chip instead of running days against a stale registry. qwen2-5-vl normalized to gpu_layers 999. An Open WebUI signing key that briefly existed in a pre-split working copy was rotated; verification confirmed it was never tracked in this repository's history.

## M15.3 - measured context ceilings, benchmark headroom guard - 2026-08-29

**Status:** Implemented, measured on this machine, applied to the live registry (26 models raised), all touched configs verified or probed with real generations.

**What went wrong first, on the record:** the context auto-tune (M15-era) searches for the largest context that LOADS, and loading is not performing - a spilled KV cache answers 10-15x slower. Its 294912-context "win" measured 11.8 tok/s against a 209.8 baseline. A KV-arithmetic replacement then mispredicted in both directions across architectures (over for dense fp16 caches, under for MoE). Conclusion, twice paid for: the ceiling is an empirical fact, so measure it.

- **probe_context_throughput** (benchmark.py): one warm-up plus one timed completion at an explicit context via the existing controller, returning tok/s or an honest None. The measurement the load-only tune lacked; the 3-run benchmark remains the number of record.
- **Pre-run headroom guard** (assert_can_run): refuses when free system RAM is under 4 GB (a run then measures paging, not the model - this session recorded 16.6 and 214.3 tok/s for the same config ten minutes apart, the slow row at 98% RAM), and warns when other applications already hold >1.5 GB of VRAM. Port conflict still checks first; no provider means no check, never a crash.
- **scripts/measure_context_ceilings.py**: owner-run, resumable (state JSON per probe), throughput-floored (a rung must keep 85% of baseline), capped at each model's native window (never proposes YaRN), q8_0 KV cache on every probe above baseline, group dedup for identical-family models, --apply writes through write_model_tuning only.
- **Measured and applied:** 19 model groups probed, 26 registry rows raised - e.g. gemma-4-26b-a4b 8192->131072 at 136.8 tok/s, ernie 8192->49152 at 233.5, qwen2.5-VL 16384->98304 at 147.8, both dense 27B ablated Qwen3.6 8192->24576 at ~46.5. The q8_0 KV cache is the enabler: it halves cache memory at a measured cost within noise. Cliffs were located, not inferred - every ceiling has a measured failing rung above it (e.g. 27B IQ4_XS: 46.5 tok/s at 24576, 7.1 at 32768).
- **Verified at the applied configs:** qwen3-coder-30b-a3b 16384 -> 202.2 tok/s score 100; ernie 49152 -> 223.5 tok/s. Earlier in the session: gpt-oss 46080 -> 169.9, kimi-vl-thinking 131072 -> 244.6, kimi-vl-instruct 24576 -> 205.1.

**Known limitations:**

- The auto-tune itself still lacks the throughput floor; the measured script is the recommended path and the desktop's auto-tune button should gain the same floor in a later slice.
- Probes are single timed completions; per-model scores at the new configs are refreshed only for the models re-run through the full benchmark.
- ernie's quality score moved 67->83.3 across sessions on the same 6-check suite; the suite is too small to rank capability and a harder task set remains open work.


## M15 slice 1 - first-run setup wizard - 2026-08-26

**Status:** Implemented and locally verified; the llama.cpp release fetch is wired but has not yet been exercised against a live GitHub release on this machine.

**The gap this closes:** a new user could download a model in-app but had nothing to run it with. `paths.llama_cpp` shipped empty, nothing detected or installed it, and `LOCITIZE.bat` on a fresh clone died with a `ModuleNotFoundError` before drawing a window. The shipped comment "the first-run setup fills them in" described a setup that did not exist.

- **Feature-first onboarding** (`setup_plan.py`): nine capabilities the user actually picks from ("Talk to it", "Hear it back", "Watch my screen") resolve onto deduplicated requirements. Selecting a feature selects what it is useless without - Watch my screen ticks vision and voice_out visibly, so nobody is charged for an unseen download. Pure: no tkinter, no subprocess, no network, no filesystem.
- **Real totals before consent**: every requirement carries its measured size, so the confirm screen states "9 items to install, 1.2 GB to download" rather than "this may take a while". Items already present cost zero and are listed as `[have]`.
- **Tkinter bootstrap** (`setup_wizard.py`): resolves the circularity that the UI which installs PySide6 cannot itself be PySide6. tkinter ships with CPython, so the window draws on a bare `python` with no venv and no wheels. Install work runs on a worker thread reporting through a queue, so a 2.6 GB resolve never freezes the window. Hands off to the Qt desktop when done.
- **Machine detection** (`setup_env.py`): stdlib-only at module scope by construction - it runs before yaml exists, so it cannot import `config.py`. Detection is deliberately pessimistic: it answers "definitely already here?" and returns False when it cannot tell, because a wrong True leaves a broken install claiming to be finished. Finds an existing `llama-server.exe` on PATH and in plausible locations, turning a 360 MB download into a one-line config write.
- **llama.cpp acquisition**: the release asset is chosen by scoring the release's actual file list, not by composing a filename from a template, because upstream has renamed these more than once. A CUDA build is never handed to a machine with no NVIDIA driver; a CPU build is an accepted fallback on a GPU machine. The publisher digest is enforced when GitHub declares one; when it does not, the download proceeds only on the explicit unverified rung with a ticked consent box. Downloads go through `modelhub.download_verified` - no second downloader was added.
- **Path write-back** (`write_settings_paths`): the step that actually closes the gap. Runs through the venv interpreter via a yaml round-trip rather than string surgery on a file the user hand-maintains.
- **Fail-open gate**: `LOCITIZE.bat` diverts to the wizard only when `setup_wizard.py` is present and the venv or PySide6 is missing. A checkout without the wizard launches normally rather than being diverted into a setup that cannot run. `LOCITIZE.bat --setup` forces it, so features can be added later.
- Archive extraction refuses any member that escapes the destination; winget installs stay user-scope; fetched binaries land under the data root, never the install directory.

**Verification:** complete suite passes with 1,154 tests and 2 skips, up from 1,151 (41 new). Wizard widget tree smoke-tested through all four screens. `setup_env` and `setup_wizard` both import cleanly on the bare system interpreter with no venv active, which is the bootstrap claim. The first draft hardcoded one machine's own model folder into the binary search list and `scripts/verify_no_owner_paths.py` refused it - candidates are now derived from the environment only.

**Known limitations:**

- The GitHub release fetch has not been run against a live release from this machine (an existing `llama-server.exe` was detected here, so the download path was never taken). First real exercise is an owner acceptance step.
- `first_model` is a signpost, not an installer: it defers to the existing Get-models page after launch rather than choosing a model on the user's behalf.
- `vision_projector` and `finetune_pip` have no automatic installer yet; both report honestly and degrade to a Settings-page instruction.
- Detection cannot see `whisper`/`kokoro` paths until yaml exists, so a first run reports them missing even when a pre-M14 `settings.yaml` names them. Pessimistic in the safe direction; costs a re-detect after the venv is built.

## Complete 31-model same-task benchmark - 2026-08-25

**Status:** Complete with 30 valid generation-throughput results and one honest task failure; not committed or released.

- Resumable campaign `2026-08-25T0628` skipped the original 12 completed models and loaded all 19 later imports sequentially with the same ocean prompt, temperature 0, seed 42, 256-token cap, one discarded warm-up, three timed runs, and the same six deterministic checks.
- Eighteen imported models produced valid multi-token generation timings; their results were appended to the user-data JSONL and Markdown reports.
- TinyLlama returned immediate EOS for all three timed requests. llama.cpp exposed that zero-duration one-token event as `1000000.0 tok/s`; Locitize now rejects one-token timing artifacts in both new benchmark runs and restored desktop history instead of ranking them.
- A repeat run at the model's verified 2,048-token training context recorded an honest failed row with `no /completion run returned usable multi-token server timings`. TinyLlama therefore displays `-`, not a fabricated throughput number.
- Imported registry notes now reflect the completed load tests and benchmarks. TinyLlama's registered context was reduced from the provisional 8K value to the model-supported 2K value.

**Verification:** Focused benchmark tests pass 28/28; the complete project-venv suite passes with 1,113 tests and 2 skips; 30 current model/GGUF pairs resolve to valid throughput rows; port 8080 is free; no `llama-server` process or benchmark lock remains. The already-running desktop still needs an operator-confirmed restart to load this code.

## Additional local model registry import - 2026-08-25

**Status:** Imported and live-refreshed; all 19 were subsequently load-tested and benchmarked, with 18 valid speeds and one documented immediate-EOS failure.

- Added 19 distinct runnable GGUF variants already stored on this machine to the portable Locitize registry, increasing it from 12 to 31 installed models.
- Existing files were referenced in place rather than moved or copied, preserving LM Studio hard links, Hugging Face cache entries, Downloads, fine-tune outputs, and the OneDrive project.
- Excluded projector-only files, embeddings, tokenizer fixtures, raw SafeTensors checkpoints, exact duplicate fine-tune content, and duplicate copies of already registered model variants.
- Added separate entries for two different-hash Gemma cache revisions and four distinct historical fine-tune versions of one model.
- New entries use verified 8K contexts except TinyLlama, whose load log proved a 2K training-context cap.
- Recovery copy: `locitize-data/backups/models-before-extra-import-20260825-061338.yaml`.

**Verification:** Configuration parsed with zero errors; 31/31 ids are unique; all 31 model paths and every configured projector path resolve; health and terminal inventory pass; the running desktop accepted a live Refresh and displayed `model list refreshed (31 models)`.

## Benchmark throughput display - 2026-08-25

**Status:** Implemented and locally verified; not committed or released.

- The model table now labels its performance column `Last gen tok/s` and reads the latest successful generation throughput for the same registered model and GGUF file from the append-only benchmark history.
- Quality percentages remain available in the detailed benchmark result; they are no longer presented as tokens per second.
- A completed benchmark updates the row immediately, and saved history restores the value after restart for registered and discovered models.
- The table defaults to numeric throughput order, fastest first, and keeps rows without a successful measurement at the bottom as `-`; clicking the header reverses the measured order.
- Standardized task campaign `2026-08-25T0534` ran the same ocean prompt with a discarded warm-up, fixed seed/temperature, and three timed generations for every registered model; all 12 scenarios succeeded and now populate the table.

**Verification:** 178 focused desktop/controller tests passed; the complete suite passed with 1,112 tests and 2 skips. The real 12-model campaign completed 12/12 with port 8080 released, no `llama-server` process, and no benchmark lock left behind.

## Milestone 1  -  2026-07-18

**Status:** Validation complete. All acceptance criteria pass. Reviewer, Security, and QA approved.

**What's new:**

- **Launcher core** (`launcher.py`)  -  single entry point, bootstrap, health verify ladder, interactive menu or JSON output.
- **Health system** (`health.py`)  -  13 structured probes (Python, GPU, CUDA, VRAM, RAM, disk, ports, Whisper, llama.cpp, Kokoro, voice, models) with PASS/WARNING/FAIL + remedies. Injected providers allow testing without a GPU.
- **Config system** (`config.py`)  -  layered settings (code defaults -> YAML -> env vars), validates `settings.yaml` and `models.yaml`, reports errors/warnings without crashing.
- **Model registry** (`models.py`)  -  five models (Qwen3 14B, DeepSeek R1 Distill 14B, Qwen3.6 27B, Qwen2.5 VL, Inkling future), each with context size, GPU layers, recommended prompt, VRAM estimate, benchmark score placeholder.
- **Service manager** (`services.py`)  -  Windows-correct process supervision (CREATE_NEW_PROCESS_GROUP, CTRL_BREAK_EVENT, taskkill escalation), loopback-only port allocation (8080-8099), readiness health checks.
- **Logging** (`logger.py`)  -  rotating per-subsystem file handlers (launcher, assistant, speech, llm, tts, benchmark, errors), UTC format, configurable levels.
- **Documentation journal** (`documentation.py`)  -  append-only development journal, appended on interactive exit.
- **Plugin system** (`plugins.py`)  -  auto-registration of six built-in applications (Model Manager, Documentation, Health Checks, Settings, Voice Assistant unavailable, Benchmark Suite unavailable); safe no-op dispatch for unavailable apps.
- **Benchmark suite** (`benchmark.py`, `benchmark_tasks.py`)  -  configurable task-based performance measurement with real tokens/second metrics, score persistence to models.yaml. Campaign feature (batch runs, comparison export) deferred at M5 baseline.
- **Test suite** (47 unit tests)  -  health probes (fake GPU providers), config validation (malformed/empty/partial YAML), services (process fake launcher), plugins (registration), launcher (CLI modes). GPU-free path proven.
- **Configuration files**  -  `settings.yaml` (paths, ports, thresholds, logging), `models.yaml` (five models, location empty until user sets).
- **Scripts**  -  PowerShell thin wrappers (`start.ps1`, `stop.ps1`, `update.ps1`, `benchmark.ps1`).
- **Docs**  -  architecture overview, roadmap (M2/M3/M4 roadmap), benchmark results header, development journal header.

**Verification:**

- AC1: Architecture.md documents module boundaries (launcher.py references: 9).
- AC2: `python -c "import launcher"` exit 0, no side effects.
- AC3: health.py contains PASS (18 instances).
- AC4: models.yaml parses, five models listed, config round-trip 0 errors.
- AC5: `python -m compileall Codebase/platform -q` exit 0.
- AC6: `python -m pytest -q` -> 47 passed.
- `python launcher.py --health --json` -> valid JSON, 13 probes, exit 0 (no FAILs) or 1 (FAILs present).
- `python launcher.py --no-menu` -> banner/status/models rendered, exit 0.
- Character hygiene: no non-ASCII bytes, no emojis, plain ASCII only.
- Write boundary: legacy tree untouched, logs/ only.
- Security: loopback-only 127.0.0.1, no 0.0.0.0, no vault paths, no secrets in code/YAML, env-only LOCITIZE_* channel.

**Known limitations (Reviewer findings, M2 deferred):**

- M-1 (Medium): Port auto-allocation computes a new port but doesn't propagate it to the launch command or health URL. No-op with misleading comment. Not reachable in M1 (no binary paths set). Fix and add tests in M2.
- L-1 (Low): `stop_all()` not registered with atexit or wrapped in menu try/finally. Could orphan children if future service crashes launcher. Not blocking M1 (no service start). Add atexit/finally in M2.
- L-2 (Low): Service readiness timeout branch and port-reassignment `_probe_spec` branch not tested. High-risk branches for M2. Add tests in M2.
- L-3 (Low): ModelRegistry rebuilt per keystroke, couples to run-order. Works correctly but fragile. Refactor in M2.

**Known limitations (Security findings, M2 deferred):**

- SEC-1 (Low): health_path not validated; no injection blocking. Not attacker-reachable in M1 (no service start). Harden path validation in M2.
- SEC-2 (Low): server_args not validated; no shell-injection blocking. Not reachable in M1 (no service start). Add argv validation in M2.
- SEC-3 (Low): Minor docstring nit.

**Known limitations (QA observations):**

- O-1 (Polish): model-manager app registered as available but is a no-op (prints "code 0"). Improve in M2.

**Known limitations (Design):**

- Service start not wired: menu prints launch spec but does not execute. Requires binary paths (Whisper, llama.cpp, Kokoro) to be configured before M2 starts services. This is correct degraded behavior; the health system honestly reports FAIL until paths are set.
- Voice Assistant and Benchmark Suite unavailable in M1 (marked future/M3+). Menu refuses to launch them with milestone notes.
- Legacy chief-of-staff app (Codebase/src, Codebase/apps) not integrated. Will attach as Voice Assistant plugin in M3+.
- GPU is optional but recommended. GPU-absent health path proven by injected fake providers in test suite.

---

## Milestone 2  -  2026-07-18

**Status:** Validation complete. All 8 acceptance criteria pass. Reviewer, Security, and QA approved.

**What's new:**

- **Live model serving** (`launcher.py` menu, `ModelController`, `ServiceManager`)  -  Start a real local language model, confirm readiness via health endpoint, report results with timing and process info.
- **Live model switching** (`ModelController.switch()`)  -  Switch from running model to another. Old process cleanly stopped (killed if needed), new model starts, zero orphans guaranteed.
- **Non-interactive CLI for model start** (`--smoke-start <model_id> [--ctx-size N] [--gpu-layers N] --json`)  -  Start model, wait for readiness, emit structured JSON outcome, always clean up. Exit 0 on ready+clean-shutdown; exit 1 on timeout/failure.
- **Non-interactive CLI for service status** (`--service-status [--json]`)  -  Query current service state (running model, port, PID, or idle) as JSON or human text.
- **Cleanup guarantees** (atexit + menu finally blocks, `stop_all()` idempotent)  -  Services stop on menu quit, launcher exit, or explicit stop. No orphaned llama-server processes or VRAM leaks.
- **Fixed M-1 (port reassignment)**  -  Resolved port now single-sourced into both launch argv and readiness probe. Verified by test `test_reassigned_port_propagates_to_argv_and_health_probe`.
- **Fixed L-1 (cleanup wiring)**  -  `stop_all()` registered with atexit and wrapped in menu try/finally. Verified by test `test_no_orphan_process_after_menu_exit`.
- **Fixed L-3 (per-keystroke registry)**  -  ModelRegistry built once in `run()`, passed to menu, not rebuilt per keystroke.
- **Hardened SEC-1 (health_path validation)**  -  Config validator rejects non-rooted paths, schemes, authorities, control chars; unsafe values reset to `/health` with WARNING. Probe host hardcoded to 127.0.0.1 loopback (cannot be moved by path injection). Verified by three tests: `test_health_path_cannot_move_probe_off_loopback`, `test_health_path_validation_rejects_authority`, `test_health_path_predicate_accepts_and_rejects`.
- **Added lifecycle coverage**  -  New tests for switch (old process dies), orphan/atexit cleanup (menu quit with running service), health_path hardening (three scenarios), port reassignment (8080->8081 reaches argv and URL).
- **Pipe-drain deadlock prevention**  -  Child stdout/stderr routed to log file or DEVNULL, never unread PIPE. Large model load-time logs cannot deadlock the child.

**Real hardware evidence (verified on NVIDIA RTX 5070 Ti, 14361 MiB VRAM free):**

- AC1: Full regression suite `57 passed in 0.47s`.
- AC2: Port reassignment `test_reassigned_port_propagates_to_argv_and_health_probe` 1 passed.
- AC3: Health path hardening 3 tests passed.
- AC4: Model switch (Qwen3 -> DeepSeek) old process dies, new process serves, only new reflected as RUNNING.
- AC5: Menu quit with running model (Qwen3) -> no orphan, atexit cleanup fired.
- AC6: `--service-status --json` (fresh process, idle) -> `{"running_model": null, "services": []}` exit 0.
- AC7: `--smoke-start qwen3-14b --ctx-size 4096 --json` -> real Qwen3-14B loaded in 6.22s (second run 5.64s), health endpoint confirmed, exit 0, no orphan, VRAM released.
- AC8: `python -m compileall Codebase/platform -q` exit 0.

**Known limitations (Reviewer, M3):**

- L-4 (Low, owner-config-only): Duplicate `--port` in a model's `server_args` defeats the M-1 fix. Single-user local, no privilege boundary. Fix in M3: guard `build_start_spec` to reject/warn duplicate `--port`/`--host`, add test.

**Known limitations (QA observations, M3):**

- O-1 (Low, cosmetic): Early-exit smoke failure reason still prefixes "did not become ready within 60s" before the accurate log tail. Log tail is truthful.
- O-2 (Low, cosmetic): `manager.stop_all()` leaves `snapshot()["running_model"]` naming last model while status=STOPPED. Not user-reachable.
- O-3 (Low, hygiene): `logs/errors.log` and `logs/launcher.log` are git-tracked; test runs dirty them. Recommended: add `logs/` to `.gitignore` (keep `.gitkeep`), `git rm --cached` the two files.

**Design:**

- Voice Assistant, Benchmark Suite, LLM inference, speech/TTS, vision, memory remain future (M3+), interface stubs raising NotImplementedError with milestone markers.
- Legacy chief-of-staff app (Codebase/src/apps) still not integrated; will attach as Voice Assistant plugin M3+.
- Service lifecycle (start/switch/stop) now fully wired and tested on real hardware.
- All services bind loopback-only 127.0.0.1:8080-8099; no external network exposure.

---

## Milestone 3  -  2026-07-18

**Status:** Validation complete. 8 of 9 acceptance criteria pass; AC9 pending owner grade. Reviewer, Security, and QA approved.

**What's new:**

- **Whisper-server as a managed service**  -  whisper-server is now an LOCITIZE service under the same lifecycle guarantees as llama.cpp (readiness probe, port policy 8091, clean shutdown, no orphans via ServiceManager + atexit).
- **Real speech-to-text transcription** (`--transcribe <audio-file> --json`)  -  Send a WAV/MP3 to the running whisper-server (starts it if needed, stops it after to restore prior state) and receive a structured JSON transcript. Real CUDA ggml-large-v3-turbo transcription proven on real audio.
- **Smoke-start whisper-server** (`--smoke-start-whisper --json`)  -  Start whisper-server, confirm TCP readiness on port 8091, cleanly shut it down. JSON outcome, exit 0 only if ready + clean.
- **Whisper-stream microphone lifecycle** (`--smoke-listen --duration N --json`)  -  Start whisper-stream.exe (SDL2 mic capture) as a managed service, hold for N seconds, stop cleanly. Tests the lifecycle only (no content claim).
- **Live microphone transcription** (`--listen [--duration N] --json`)  -  Start whisper-stream for real-time mic input, echo the captured transcript for N seconds, then stop cleanly. Voice quality manually graded by owner (AC9).
- **L-4 fixed (managed-flag rejection)**  -  Platform-managed flags (`--port`, `--host`) in a model's `server_args` are now stripped before launch, both space (`["--port","8091"]`) and equals (`["--port=8091"]`) forms. Verified by three tests and 10-input adversarial probe.
- **O-3 fixed (logs gitignore cleanup)**  -  Runtime log files under `logs/` are now untracked and covered by `.gitignore`, so routine runs stop dirtying `git status`. Only `.gitkeep` remains tracked.
- **Coexistence of llama + whisper**  -  Both services can run simultaneously on the same ServiceManager (llama.cpp on 8080, whisper-server on 8091), both ports open, single `stop_all()` tears both down cleanly.
- **Added 24 new unit tests**  -  Total 71 tests covering whisper service lifecycle (fake-process), multipart HTTP framing, managed-flag stripping, and smoke harness paths (71 passed, 0 skipped).
- **Hardened multipart client** (`whisper.py`)  -  RFC 2046-compliant multipart framing, hardcoded ASCII filename (no injection), timeout enforced (120s default), response parsed as data (no eval).

**Real hardware evidence (verified on NVIDIA RTX 5070 Ti with real whisper-server.exe and ggml-large-v3-turbo.bin):**

- AC1: Full regression suite `71 passed in 0.50s`.
- AC2: L-4 managed-flag rejection 3 tests; 10-input adversarial probe all forms stripped.
- AC3: Whisper service lifecycle `3 passed`.
- AC4: Real `--smoke-start-whisper --json` -> ready, port 8091, exit 0, no orphan (independent re-run pid 21956, 1.25s).
- AC5: Real `--transcribe` with Builder fixture -> matched, exit 0; QA independent phrase (Giraffe test) -> matched, exit 0, no orphan, auto-start-then-restore verified.
- AC6: Real `--smoke-listen --duration 5` -> ok, pid confirmed running mid-window (tasklist check), exited cleanly, no orphan.
- AC7: `python -m compileall Codebase/platform -q` exit 0.
- AC8: `python scripts/verify_logs_untracked.py` PASS; `git ls-files logs/` -> only `.gitkeep`.
- AC9: Manual owner-graded live-mic voice quality. Pending owner invocation: `python launcher.py --listen --json`, speak phrase, `the project CLI's grade command: --project LOCITIZE --criterion AC9 --result pass|fail`.

**Known limitations (Reviewer, non-blocking):**

- L-1 (Low, robustness): Empty/silent transcript yields `outcome:"ok"`. Not a fabrication but could benefit clearer labeling. Fix optional.
- L-2 (Low, defense-in-depth): Case variants (`--PORT`) and short aliases (`-p`) not stripped. Not exploitable in practice (binaries case-sensitive). Optional hardening.
- L-3 (Low, memory efficiency): Whole audio file read into memory. Non-issue for short clips; relevant only for hour-long recordings. Optional optimization.
- Carried forward from M2: O-1, O-2 (cosmetic) assigned to M4.

**Known limitations (Security, non-blocking):**

- SEC-M3-1 (Low, informational): Redirect handling not suppressed in urllib. Loopback-only, non-exploitable.
- SEC-M3-2 (Low, informational): SAPI fixture PS path interpolation (harness-only, owner-controlled).
- SEC-M3-3 (Low): Managed-flag case-variant residual (non-exploitable, optional hardening same as L-2).

**Known limitations (DevOps, non-blocking):**

- 34 tracked `__pycache__/*.pyc` files (pre-existing, same as logs issue). Recommended next milestone: add to `.gitignore`, untrack.

**Design:**

- Benchmark Suite, LLM inference, text-to-speech (Kokoro), assistant agent, memory/knowledge graph, vision remain future (M4+), interface stubs.
- Voice IN (whisper-server + whisper-stream) is M3-complete; voice OUT (Kokoro TTS) deferred to M4.
- Legacy chief-of-staff app still not integrated; will attach as Voice Assistant plugin M4+.
- All services bind loopback-only 127.0.0.1; whisper reserves port 8091 (configurable).

---

## Combined Milestone (M3 + M4)  -  2026-07-19

**Status:** Validation complete. 13 of 15 acceptance criteria pass (AC1-AC8, AC10-AC14 executable); AC9 (manual voice quality), AC15 (manual end-to-end GUI sit) pending owner grading. Reviewer v6, Security, and QA approved.

**What's new (M4 GUI command center):**

- **GUI command center** (`python launcher.py --gui` or `LOCITIZE GUI.bat`)  -  Native Tkinter window with five visual control panels for model management, real-time metrics, and voice transcription.
- **Model list and control panel**  -  Select, start, switch between, and stop models. Chat button opens the running model's web interface.
- **Model settings editor**  -  Edit gpu_layers and context_size per-model with real-time validation. Save persists to models.yaml with byte-identical round-trip guarantee.
- **Live metrics monitor**  -  Real-time tokens/second and KV-cache usage (%) when a model is running and the binary supports `/metrics` endpoint. Honest degradation when metrics unavailable.
- **Voice transcription panel**  -  Listen button for microphone capture; transcript displayed after recording.
- **No-orphan window-close guarantee**  -  Closing the GUI (X button) cleanly shuts down all running services. Validated live on real llama-server binary during model start and whisper-stream recording; zero orphaned processes.
- **H-1 fix (window close mid-flight)**  -  Fixed a race where closing the window during a model start/switch would orphan llama-server. Now `ServiceManager.stop_all()` covers registered-but-not-yet-started services via dual snapshots of `_start_order` and `_services` under the lock.
- **H-2 fix (window close mid-listen)**  -  Fixed a separate race where closing during a Listen would orphan whisper-stream. Now `_gui_listen` passes the shared ServiceManager instead of creating a fresh one; the single atexit backstop covers the whisper child.
- **M-1 fixed (in-flight shutdown test)**  -  Added two new gui_shutdown tests that simulate window close while the ops worker is inside an in-flight start/listen (not idle), proving the race is covered.
- **Optional reverse proxy** (config-only, default-off)  -  Loopback reverse proxy binds `127.0.0.1:8085` to serve the active model via `locitize.local` (requires owner-manual elevated PowerShell script to add hosts entry). Chat button always uses direct `127.0.0.1:<port>` and needs no proxy. Proxy is never silently enabled; enable_locitize_local.ps1 requires Administrator and is owner-run.
- **Model field write safety** (`write_model_fields`)  -  Targeted yaml-field edit with CRLF/LF preservation, atomic temp-file + os.replace, round-trip validation. Duplicate-key safety: refuse-rather-than-corrupt if verification fails. Tested with valid/invalid gpu_layers and context_size values.
- **Added 22 GUI-related tests**  -  gui_command, gui_metrics, gui_shutdown (including in-flight races), write_model_fields (valid/invalid/duplicate), proxy_target (routing + loopback). Total test suite now 121 tests.

**Real hardware evidence (NVIDIA RTX 5070 Ti, real llama.cpp + whisper-server.exe + ggml-large-v3-turbo.bin):**

- AC1: Full regression suite `121 passed in 0.54s`.
- AC2-AC8: All voice pipeline criteria re-verified.
- AC9: Manual owner-graded live-mic voice quality. Pending owner session: `python launcher.py --listen --json`, speak phrase, `the project CLI's grade command: --project LOCITIZE --criterion AC9 --result pass|fail`.
- AC10: GUI command tests `9 passed`.
- AC11: GUI metrics tests `6 passed`.
- AC12: Model write tests `7 passed`.
- AC13: Proxy target tests `6 passed`.
- AC14: GUI shutdown tests `5 passed` (including in-flight start + in-flight listen races).
- AC15: Manual owner end-to-end 7-step GUI sit. Pending owner invocation: select model -> start -> view metrics -> edit settings -> save -> listen -> close window (expect zero orphans).

**Blocking findings fixed (Review v5 -> v6):**

- H-1 (window close mid model start orphans llama-server)  -  CLOSED. `stop_all()` now snapshots both `_start_order` and `_services.keys()` under lock; serves first registered-and-started, then remaining registered. No deterministic skip.
- H-2 (window close mid listen orphans whisper-stream)  -  CLOSED. `_gui_listen` now reuses shared ServiceManager instead of creating fresh one; single atexit backstop covers the whisper child.
- M-1 (gui_shutdown tests never exercises in-flight shutdown)  -  CLOSED. Added two new tests that block the ops worker inside an in-flight start and inside listen, then fire shutdown from a separate thread. Races proven covered.
- L-1 (proxy unreachable from GUI)  -  DISPOSED. Proxy wired as config-only (not GUI-exposed) for this milestone; Chat always uses direct loopback path. README documents this boundary.

**Known limitations (non-blocking):**

- AC9 (manual voice quality) and AC15 (manual end-to-end GUI sit) require owner invocation. Both are design-by-intent (no automated harness can drive human voice or grade live Tkinter window). AC1-AC8, AC10-AC14 all pass; AC9/AC15 are explicitly noted as outside QA gatekeeping.
- L-1/L-2/L-3 from M2 carried forward (empty transcript, flag case variants, whole-file audio read). Non-blocking; documented in Review.
- 34 tracked `__pycache__/*.pyc` files (pre-existing from M1, same treatment as logs cleanup recommended for next milestone).

**Security (all pass, no Critical/High):**

- Proxy loopback bind + upstream enforcement (hardcoded 127.0.0.1, no config bypassable).
- Port-80 bind never auto-elevated; enable_locitize_local.ps1 raises remedy if not elevated.
- write_model_fields path confined, YAML injection-safe via safe_load and refuse-rather-than-corrupt.
- GUI has no secrets; all paths loopback-only; no shell-form argv.

**Design:**

- Voice (M3+M4 complete): real-time speech-to-text via whisper-server and live microphone capture (voice IN); voice OUT (Kokoro TTS) deferred to M5+.
- GUI (M4 complete): native Tkinter command center with five panels; no external network; loopback-only proxy optional.
- Benchmark Suite, LLM inference, TTS, assistant agent, memory, vision remain future (M5+), interface stubs.
- All services bind loopback-only 127.0.0.1; model serving on 8080, whisper on 8091, optional proxy on 8085.

---

## Milestone 5  -  2026-07-19

**Status:** Validation complete. Benchmark suite feature set delivered; campaign deferred at 24.9 tokens/second baseline. All acceptance criteria pass. Reviewer, Security, and QA approved.

**What's new:**

- **Benchmark suite** (`--benchmark [--task <task_name>] --json`)  -  Configurable task-based performance measurement (tokens/second, latency) with real inference. Results persisted to `models.yaml` benchmark_score field. Campaign feature (batch runs, model comparison, export) deferred to future release at 24.9 tokens/second baseline on Qwen3 14B.

---

## Milestone 6-M8 combined  -  2026-07-19

**Status:** Validation complete. 16 of 17 acceptance criteria pass (AC1-AC16 executable all pass; AC17 pending owner voice/interrupt session). Reviewer, Security, and QA approved.

**What's new (M6 - Voice OUT):**

- **Kokoro TTS managed service** (`kokoro_server.py`, port 8092)  -  Local CPU-based text-to-speech using real torch KokoroEngine. Service binds loopback-only 127.0.0.1, loads voice models on demand, synthesizes real 16-bit PCM mono wav files, and shuts down cleanly with no orphans via ServiceManager + atexit.
- **TTS synthesis** (`--speak "text" --json`, `--voice <voice_name>`)  -  Synthesize any text to speech and play via system audio. 8 distinct Kokoro voices (am_michael, af_bella, am_adam, bf_emma, etc.). Real CPU synthesis on Windows, proven on RTX 5070 Ti with 20+ second startup and real winsound playback.
- **Audition voices** (`--audition`)  -  Iterate through all available voices, synthesizing a sample sentence in each and playing each one. Lets the owner hear all voice options before an assistant session.
- **End-to-end synthesis validation** (AC13)  -  Real 4.55-second 16-bit PCM mono wav (218,444 bytes) synthesized for "LOCITIZE on-machine neural intelligence system online." Three voices produce distinct output (verified by md5/duration/byte count), proving voice selection genuinely changes synthesis.

**What's new (M7 - Assistant Loop):**

- **Interactive assistant** (`--assistant [--text] [--no-speak] [--voice <voice_name>] --timings`)  -  Full conversational AI loop with real LLM (llama.cpp), real speech-to-text (whisper-server/whisper-stream), and real speech synthesis (Kokoro). All seams injected for test determinism; real paths use actual model/whisper/TTS services.
- **Conversation history and trimming** (`trim_history()`)  -  Maintains a conversation history trimmed to fit within the model's context budget. Drops oldest messages when over budget, but always preserves the leading system message and current user turn. Tested with real LLM inference.
- **Sentence-based streaming and playback** (M7.4, M7.5)  -  Assistant replies are split into sentences and played one-at-a-time via Kokoro, enabling mid-turn interruption and faster time-to-first-audio. Tested via deterministic SentenceSegmenter tests (no 3.14 splits, correct trailing flush).
- **Per-stage timing** (`--timings` flag)  -  Reports real wall-clock latency per stage (stt_ms, llm_first_token_ms, llm_total_ms, tts_first_audio_ms). Useful for performance analysis. Real qwen3-14b measured: llm_first_token_ms=2468, llm_total_ms=2562.
- **Text-mode for scripting** (`--assistant --text --no-speak`)  -  Stdin/stdout mode for headless/piped usage. Useful for automated testing and integration.
- **Scripted 3-turn conversation validation** (AC14+)  -  Demonstrated factual Q&A, history retention (turn 2 context included turn 1), and /recall search all working end-to-end with real LLM and real memory persistence.

**What's new (M8 - Vision, Memory, Plugins):**

- **Vision Q&A** (`--describe <image_path> [--prompt "..."] --json`)  -  Single-image understanding via real Qwen2.5-VL model with on-disk mmproj multimodal projector. POSTs image as base64 data-URI; returns real model answer. Tested on two images: programmatically-generated red square (model answer: "red"), QA-generated blue circle (model answer: "blue"). No fabricated strings.
- **Memory store (JSONL)** (`memory/` directory, gitignored)  -  Append-only conversation transcript storage with `{session_id, ts, role, text}` schema. One session file per conversation; owner-deletable; never tracked by git. Tested append-only invariant and cross-session search.
- **Memory recall and search** (`/recall <query>`, `memory.search()`)  -  Case-insensitive substring search across stored transcripts. Matched content by keyword, cross-session (two sessions tested). Linear scan (not embedding-based), lightweight. Tested with real JSONL files and real LLM-assisted conversations.
- **Plugin registry maturity** (`plugins_registry`)  -  Voice Assistant, Benchmark Suite, and Vision are auto-registered built-in plugins (`available=True`). Third-party plugin contract documented in `docs/plugins.md`. Routed via menu and CLI dispatcher consistently.

**Pre-finale defect pass (iteration 5, within M8 scope):**

- **DEF-QA-1 (Medium, FIXED)** - Assistant crashed with `UnicodeEncodeError` when model replied with emoji and stdout was piped (cp1252). Fixed via `_force_utf8_output()` in launcher.py (guarded sys.stdout.reconfigure to utf-8), mirroring the project CLI. Verified: emoji reply printed cleanly to piped stdout with no exception.
- **Review M-1 (Medium, FIXED)** - Mic-path whisper-stream ServiceManager was un-backstopped for atexit. Registered `atexit.register(self._manager.stop_all)` in `_MicSttSource.__init__`, matching model/Kokoro pattern.
- **Review M-2 (Medium, FIXED)** - Interrupt during speech playback let in-flight sentence finish. Added purge seam and call `winsound.PlaySound(None, SND_PURGE)` in `KokoroTtsSink.drain()` so current clip halts on interrupt.
- **SEC-M6-1 (Low, FIXED)** - One-off `--speak`/`--audition` temp wav written to OS temp dir with `delete=False`. Now takes `temp_dir` parameter (platform `logs/`), passed by both callers, deleted after playback.

**Verification:**

- AC1: Full regression suite (238 tests pass, up from 233 before fixes).
- AC2-AC9: All headless keyword suites pass (tts_client 13, assistant_loop 6, assistant_history 3, assistant_segment 3, llm_client 7, memory_store 10, vision_client/mmproj 9, plugins_registry 6).
- AC10: Compileall (exit 0).
- AC11: Changelog em-dash (no U+2014).
- AC12-AC16: Real hardware runs all pass (Kokoro smoke 20.03s, TTS synthesis 4.55s wav, assistant text-mode real reply + timings, vision real qwen2.5-vl red/blue images, memory JSONL persist/search).
- AC17: Manual owner-graded (pending live mic + speak + interrupt session via the project CLI grade command).

**Known limitations:**

- AC17 (manual, owner-graded only): Live spoken assistant session (mic in, Kokoro voice out, interrupt, zero orphan) requires the owner's voice interaction. Graded via the project CLI grade command.
- Campaign feature (batch benchmark runs, model comparison, export) deferred at M5 baseline (24.9 tokens/second on Qwen3 14B).
- Three Lows from Reviewer (SEC-M6-1/2/3 noted but non-blocking): request-size cap / engine lock (kokoro_server), vision image size cap, L-3 (recall preamble info note). All have compensating controls or are optional hardening.
- Four M6 TTS proof-WAV files tracked under logs/ (smoke_tts.wav, tts_speak.wav, assistant_tts_1/2.wav) as interim artifacts; recommended untrack via git rm --cached next milestone.

**Security (all pass, no Critical/High):**

- Kokoro binds loopback-only 127.0.0.1 (hardcoded, force-set in main()).
- SEC-M5-1 (https guard in fetch_model.py) verified fixed: http/file/ftp URLs refused pre-egress.
- Memory privacy: transcripts gitignored, untracked, owner-deletable, never logged, text-only schema (no audio/base64).
- No automatic model/weight download; no hub calls; vendored config.json for offline Kokoro operation.
- All clients loopback-only, no spawning, no shell=True, list-form argv.
- Dependencies: torch 2.13.0+cpu (CPU-only, no CUDA), kokoro/misaki/spacy pass pip-audit (11 CVEs in venv tooling pip/setuptools, not runtime).

**Design:**

- Voice complete: speech-to-text (M3 whisper-server/stream), text-to-speech (M6 Kokoro), full assistant loop (M7 orchestration + LLM seams).
- Memory complete: append-only JSONL, search/recall, cross-session.
- Vision complete: single-image Q&A via Qwen2.5-VL + mmproj plumbing.
- Plugin system complete: three builtins (Voice Assistant, Benchmark, Vision) auto-registered and routed.
- All services bind loopback 127.0.0.1; ports 8080 (llama.cpp), 8091 (whisper), 8092 (Kokoro), 8093-8095 (future).
- No external network exposure; all data local and owner-controlled.
