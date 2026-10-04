# LOCITIZE Platform - Developer Guide

**Download it. Locitize it. Talk to it.**

LOCITIZE is a Windows-first desktop for local models, coding sessions, chat,
voice, vision and fine-tuning. Local inference keeps model requests on this
computer; external tools and optional installers retain their own network access.
See ../SECURITY.md for the actual boundary.

The unified 0.2 beta adds Home and Sessions, shared local coding launch settings,
and a bundled Windows runtime. Start with ../README.md and
docs/release-readiness.md for current installation and release checks. The
historical module walkthroughs below describe the existing platform foundation.

How it works, in four steps:
1. **Download** - grab a model from Hugging Face (or import GGUFs you already have) with progress and resume.
2. **Locitize** - LOCITIZE registers it, verifies it, and wires it into your local model server.
3. **Run** - one click starts the local server; everything stays on this machine.
4. **Launch** - the chat/voice UI opens against it immediately.

## What shipped after M13 (quick map)

This guide's walkthroughs below are the M1-M13 foundation and remain accurate.
The platform has since grown; each addition documents itself where it lives:

- **First-run setup wizard (M15)** - launching on a fresh machine opens a
  tkinter wizard that installs everything a chosen feature set needs: venv,
  the right llama.cpp build for the GPU (digest-verified from GitHub, cudart
  paired automatically), whisper.cpp + weights, Kokoro voices, and the local
  models you already have, registered ready-to-run. On finish it drops a
  Desktop shortcut to `LOCITIZE.vbs`. Re-run setup with `LOCITIZE.vbs --setup`.
  See `setup_plan.py` / `setup_env.py` / `setup_wizard.py`.
- **Get models (M14.14)** - catalog + HuggingFace search + verified download
  on the Models page, host-allowlisted on every redirect hop, including
  multi-part sharded GGUF sets (M15.5). See `modelhub.py`.
- **Portable config + data root (M14)** - settings/models live in the user
  data root, seeded on first run; the install tree is never written at
  runtime. `LOCITIZE_DATA_DIR` overrides. See `config.resolve_data_dir`.
- **Chat harness launcher (M14)** - opens Claude Code / Codex / OpenCode in a
  terminal against the running local model. See `harness_launch.py`.
- **Context auto-tune with a throughput floor (M15.4)** - finds the largest
  context that loads AND still generates within 85% of a measured baseline.
  The measured-ceiling batch tool is `scripts/measure_context_ceilings.py`.
- **Benchmark (M5, hardened M15)** - 24 deterministic checks, real tok/s from
  server timings, RAM/VRAM headroom guards, stale-lock self-recovery.
- **Environment variables** - every configuration override is an environment
  variable spelled `LOCITIZE_*` (see the table at the end of this guide).
- **Local microphone noise suppression (2026-09-04)** - Open WebUI recordings
  pass through a fixed FFmpeg speech-band/denoise/gate preset before the local
  whisper-server. Balanced is default; strong and byte-identical off modes are
  validated. See `audio_filter.py` and `scripts/verify_noise_suppression.py`.

This is the platform core: launcher, health system, model/service registry, and plugin skeleton.


## Production install / health (ops)

Day-2 production path (roots, update, stack health, boot order, GPU switch):
see **docs/ops.md**. Quick commands:

```powershell
.\platform\scripts\install_or_update.ps1
.\platform\scripts\health.ps1
.\platform\scripts\gpu_switch.ps1 -Target cuda    # dry-run
```
## Prerequisites

- **Python 3.11+** (verified on 3.11.7)
- **Windows** (platform uses Windows-specific process supervision and GPU detection)
- **NVIDIA GPU** (optional but recommended; GPU-absent path tested with fake providers)
- **FFmpeg on PATH** for default-on Open WebUI microphone noise suppression
  (`off` remains available when FFmpeg is intentionally absent)

## Setup

```bash
cd platform

# Install dependencies
pip install -r requirements.txt
```

Dependencies:
- `PyYAML>=6.0` - parse settings.yaml and models.yaml
- `psutil>=5.9` - system info for health probes
- `pytest>=7.0` - test runner (dev only)

## Run

### Interactive mode (if stdin is a TTY)

```bash
python launcher.py
```

Displays banner, system status (13 probes), installed models, and a menu to select models or applications.

### Health check (JSON output, non-interactive)

```bash
python launcher.py --health --json
```

Exit 0 if all probes pass; 1 if any fail. Output is structured JSON for scripting.

```bash
python launcher.py --health
```

Human-readable health table instead of JSON.

### Render without interactive menu

```bash
python launcher.py --no-menu
```

Display banner and status, skip menu, exit 0.

### Default to JSON if stdin is not a TTY

If running in a non-interactive environment (cron, systemd service, piped stdin), the launcher automatically defaults to `--health --json` and exits (does not block).

### Query service status (M2+)

```bash
python launcher.py --service-status --json
```

Report the current service state as JSON. Useful for scripting or monitoring. Exit 0 always.

Idle output:
```json
{"running_model": null, "services": []}
```

With model running:
```json
{
  "running_model": "qwen3-14b",
  "services": [{"name": "llama_cpp:qwen3-14b", "model_id": "qwen3-14b", "status": "RUNNING", "port": 8080, "pid": 44584}]
}
```

### Start a model and wait for readiness (M2+)

```bash
python launcher.py --smoke-start qwen3-14b --ctx-size 4096 --json
```

Start a real language model, wait for readiness via health endpoint, emit JSON outcome, then shut down cleanly. Exit 0 on ready+clean-shutdown; exit 1 on timeout/failure.

Flags:
- `--smoke-start <model_id>` - Model ID from models.yaml (any id from your Models page).
- `--ctx-size <N>` - Context window (optional; default: model's context_size).
- `--gpu-layers <N>` - GPU layers to offload (optional; default: model's gpu_layers).
  `-1` or `999` means "as many as fit": llama-server's `--fit` measures free VRAM at
  load and keeps the rest on the CPU; on Windows that reading cannot see other
  processes, so LOCITIZE measures what they hold at each launch and passes the
  corrected margin as `--fit-target` (a row's own `--fit-target` in `server_args`
  wins). Any other number is passed through as-is.
- `--json` - Output structured JSON. Besides the outcome it carries `tokens_per_second`
  (one timed generation) and `gpu_placement` (`dedicated_mb`, `shared_mb`, `spilled` -
  the process's measured GPU memory; `shared_mb` is what Windows paged through system
  RAM, and `spilled` is true from 400 MB) or `null` when they could not be measured.

Success output:
```json
{"model_id": "qwen3-14b", "outcome": "ready", "reason": "health endpoint confirmed ready", "resolved_port": 8080, "pid": 40432, "elapsed_s": 6.22}
```

Failure output:
```json
{"model_id": "qwen3-14b", "outcome": "failed", "reason": "did not become ready within 60s; last log: llama_server: exiting due to model loading error", "resolved_port": 8080, "pid": 2108, "elapsed_s": 2.25}
```

The launcher always cleans up: process exits, VRAM released, no listener left on the port.

### Desktop command center

```bash
python launcher.py --gui
```

Opens the PySide6 LOCITIZE Desktop (the same window as `--desktop` /
`LOCITIZE.vbs`): Models, Fine-tune, Talk, Voice Setup, Vision, Memory, Chat,
and Settings pages. Closing the window cleanly stops every managed service; no
orphaned processes remain.

### Transcribe audio file via whisper-server (M3+)

```bash
python launcher.py --transcribe audio.wav --json
```

Send a local audio file to the real whisper-server for speech-to-text transcription. If whisper-server is not running, the CLI starts it automatically, transcribes, then stops it (restores prior state). Output is structured JSON with the transcript.

**Flags:**
- `--transcribe <audio-file>` - Required. Path to WAV/MP3 or other audio format file.
- `--json` - Output structured JSON instead of text.

**Output (success):**

```json
{
  "audio_file": "audio.wav",
  "transcript": "The quick brown fox jumps over the lazy dog.",
  "outcome": "ok",
  "reason": "transcribed by whisper-server",
  "elapsed_s": 2.09
}
```

**Output (error):**

```json
{
  "outcome": "failed",
  "reason": "audio file not found: ...",
  "elapsed_s": 0.0
}
```

Exit 0 on success, exit 1 on failure.

### Open WebUI microphone noise suppression

The router filters Open WebUI microphone and Call-mode uploads locally before
forwarding a 16 kHz mono WAV to whisper-server. Configure the preset in the
data-root `settings.yaml` and restart LOCITIZE:

```yaml
speech:
  noise_suppression: "balanced"  # off | balanced | strong
```

- `balanced` is default-on and protects normal speech while reducing steady
  broadband noise and low-frequency rumble.
- `strong` rejects a harsher steady background but can weaken quiet consonants.
- `off` is an exact pass-through for diagnosis or an intentionally FFmpeg-free
  installation.

Enabled processing fails closed: missing FFmpeg returns 503, a timeout returns
504, and invalid media returns 422. Public responses never include local paths or
processor diagnostics. The router stays available for the next recording.

Measure the exact production preset on the current machine:

```bat
..\.venv\Scripts\python.exe scripts\verify_noise_suppression.py
```

The verifier exits zero only when balanced mode removes at least 6 dB of its
deterministic steady background, retains at least 50 percent of its voice-band
RMS, and processes the representative four-second clip below 500 ms p95 over
five warm runs. This feature does not separate overlapping human speakers.

### Start whisper-server smoke test (M3+)

```bash
python launcher.py --smoke-start-whisper --json
```

Start the real whisper-server binary, confirm it is ready on port 8091, then cleanly shut it down. Outputs structured JSON outcome. Exit 0 only if ready + clean shutdown; exit 1 on failure.

**Output:**

```json
{
  "service": "whisper_server",
  "outcome": "ready",
  "reason": "port readiness confirmed",
  "resolved_port": 8091,
  "pid": 42296,
  "elapsed_s": 1.77
}
```

Guaranteed no orphan process.

### Whisper-stream (microphone capture) smoke test (M3+)

```bash
python launcher.py --smoke-listen --duration 5 --json
```

Start whisper-stream.exe (SDL2 microphone capture), hold it for N seconds, stop it cleanly. Tests the start/capture/stop lifecycle (does not assert voice quality). Exit 0 if clean, exit 1 on error.

**Flags:**
- `--smoke-listen` - Start microphone capture as a smoke test.
- `--duration <seconds>` - Capture window in seconds (default: 3).
- `--json` - Output structured JSON.

**Output:**

```json
{
  "service": "whisper_stream",
  "outcome": "ok",
  "reason": "whisper-stream started, captured, and exited cleanly",
  "pid": 12852,
  "duration_s": 5.0,
  "elapsed_s": 5.06
}
```

### Live microphone transcription (M3+)

```bash
python launcher.py --listen --duration 10 --json
```

Start whisper-stream for real-time microphone input, listen for N seconds (default: 15), echo the captured transcript, then stop cleanly. Voice quality is manually graded by the owner.

**Flags:**
- `--listen` - Enable live microphone transcription.
- `--duration <seconds>` - Listen window in seconds (default: 15).
- `--json` - Output structured JSON.

Exit 0 on clean lifecycle, exit 1 on error.

### Benchmark suite (M5+)

```bash
python launcher.py --benchmark [--task <task_name>] --json
```

Run configured benchmark tasks against the installed models. Measures tokens/second, latency, and quality metrics. Results are persisted to `models.yaml` in the benchmark_score field.

**Flags:**
- `--benchmark` - Run benchmark suite.
- `--task <task_name>` - Run only the named task (optional; default: run all configured tasks).
- `--json` - Output structured JSON with results.

**Output:**

```json
{
  "task": "hello",
  "model": "qwen3-14b",
  "tokens_per_second": 24.5,
  "latency_ms": 1234,
  "reply": "Hello! How can I help?",
  "outcome": "ok"
}
```

Exit 0 on success, exit 1 on error.

### Text-to-speech synthesis (M6+)

```bash
python launcher.py --speak "LOCITIZE on-machine neural intelligence system online." --json
```

Synthesize text to speech using Kokoro TTS and play it via the system audio device. Real 16-bit PCM mono WAV synthesis on CPU.

**Flags:**
- `--speak <text>` - Text to synthesize and play.
- `--voice <voice_name>` - Voice to use (e.g., am_michael, af_bella, am_adam, bf_emma; default: am_michael).
- `--json` - Output structured JSON with wav path, duration, elapsed time.

**Output:**

```json
{
  "text": "LOCITIZE on-machine neural intelligence system online.",
  "voice": "am_michael",
  "wav_path": "logs/tts_speak.wav",
  "wav_bytes": 218444,
  "duration_s": 4.55,
  "outcome": "ok"
}
```

Exit 0 on success, exit 1 on error.

### Audition all available voices (M6+)

```bash
python launcher.py --audition
```

Iterate through all configured Kokoro voices, synthesize a sample sentence in each voice, and play each one. Lets you hear all available voices before starting an assistant session.

Exit 0 on success, exit 1 on error.

### Interactive assistant conversation (M7+)

```bash
python launcher.py --assistant [--text] [--no-speak] [--voice <voice_name>] [--timings]
```

Run an interactive conversation with the assistant. Listens for user input (via microphone or stdin in text mode), sends prompts to the LLM, receives replies, synthesizes them to speech (if voice is enabled), and stores conversation history.

**Flags:**
- `--assistant` - Enable assistant mode.
- `--text` - Text mode (stdin/stdout, no audio I/O).
- `--no-speak` - Disable Kokoro TTS output (useful for scripting).
- `--voice <voice_name>` - Voice to use for replies (default: am_michael).
- `--timings` - Report per-stage latency (stt_ms, llm_first_token_ms, llm_total_ms, tts_first_audio_ms).

Exit 0 on clean quit, exit 1 on error.

In text mode, type `/recall <query>` to search memory for matching conversation turns. Type `quit` or `/stop` to exit.

### Single-image vision Q&A (M8+)

```bash
python launcher.py --describe <path>/image.png [--prompt "..."] [--ask "..."] --json
```

Analyze an image using Qwen2.5-VL vision model. Returns the model's textual description or answer to a custom question.

**Flags:**
- `--describe <image_path>` - Path to image file (PNG, JPG, etc.).
- `--prompt <text>` - Custom question about the image (optional; default: "Describe this image in detail.").
- `--ask <text>` - Alias for `--prompt`.
- `--json` - Output structured JSON with answer, image path, elapsed time.

**Output:**

```json
{
  "outcome": "ok",
  "answer": "The main color of the shape in the image is red.",
  "image_path": "<path>/image.png",
  "elapsed_s": 4.17
}
```

Exit 0 on success, exit 1 on error (file not found, model error).

### Desktop application UI

```bash
python launcher.py --desktop
```

The unified entry point (what `LOCITIZE.vbs` launches). Eight pages: Models,
Fine-tune, Talk, Voice Setup, Vision, Memory, Chat, and Settings. The Models
page carries the model table, lifecycle controls, the Get models search, and
live RAM/VRAM meters; Settings includes per-model launch values plus a
Features panel that can add anything skipped at setup.

### Fine-tune

The QLoRA studio ships bundled in `../finetune-studio/` and is enabled by
default; the Fine-tune page starts and supervises it as a managed service.
Training runs require Docker Desktop. Discovered runs appear on the page with
Serve / Register / Delete actions; outputs live under the data root at
`finetune/outputs`.

## Test

```bash
cd platform
..\.venv\Scripts\python -m pytest -q
```

The suite is 1300+ tests and runs in about half a minute. It includes gates
that refuse machine-specific paths in the shipped tree and verify a clean
clone imports and runs.

## Configuration

The repo ships only templates: `settings.default.yaml` and
`models.default.yaml` (which deliberately contains zero model rows). On first
run they are copied into the data root as `settings.yaml` / `models.yaml`,
which then belong to the user. The data root is `LOCITIZE_DATA_DIR` if set,
else `platform/locitize-data/` when it exists, else `%LOCALAPPDATA%\LOCITIZE`.
Never edit the live YAMLs by string surgery from code - use the chokepoint
writers in `config.py` / `setup_env.py`.

## Environment Variables

Every override is spelled `LOCITIZE_*`. The most used:

| Variable | Overrides |
|---|---|
| `LOCITIZE_DATA_DIR` | the data root |
| `LOCITIZE_LLAMACPP_PATH` | path to llama-server.exe |
| `LOCITIZE_WHISPER_PATH` / `LOCITIZE_WHISPER_MODEL_PATH` | whisper binary / weights |
| `LOCITIZE_KOKORO_MODEL_PATH` / `LOCITIZE_KOKORO_VOICES_PATH` | TTS weights / voices |
| `LOCITIZE_MODEL_<ID>` | a specific model's file location |
| `LOCITIZE_LOG_DIR` | where logs are written |

## Logs

All logs live in the data root under `logs/` - the install directory is never
written at runtime. Each managed service appends to its own file
(`llama_cpp_*.log`, `whisper_server.log`, `kokoro_server.log`, ...), and
`errors.log` collects launcher-level failures.

## More

- Milestone history: `docs/changelog.md`
- Chat frontends and coding CLIs: `docs/chat.md`
- Plugin surface: `docs/plugins.md`
