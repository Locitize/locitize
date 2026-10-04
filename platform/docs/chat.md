# Chat applications and local coding

The native Sessions page manages coding histories inside LOCITIZE. Its New coding
session and Resume with local model actions select a project and model, wait for
readiness, then launch Codex, Claude Code or OpenCode using per-process settings.
The Chat page uses the same configuration builder. Neither launch path rewrites
global Codex/OpenCode settings. CLI permissions remain enabled. Resume original
retains the tool's original backend, which may be remote.

Local coding terminals reserve the selected model until Release model is clicked
on Sessions. Keep LOCITIZE running while using them. Open WebUI's request-selected
model continues to control routing when no conflicting reservation exists.

Codex override reference: https://developers.openai.com/codex/config-advanced/
OpenCode inline config reference: https://opencode.ai/docs/config/

LOCITIZE manages chat applications; it never rebuilds chat. There are two chat UIs:

- **llama.cpp web UI** - the built-in interface the running model already serves.
  Zero setup, always available whenever a model is running.
- **Open WebUI** - a richer chat application (conversation history, uploads,
  multi-model) that LOCITIZE installs into a dedicated venv, points at the running
  llama.cpp server, starts, and stops as a managed loopback service.

The **chat-UI chooser** decides which one to open. It runs from the Chat button in
the GUI, the `chat` action in the terminal menu, and the `--chat-ui` launcher flag.

## The chooser

`settings.yaml -> chat.preferred_ui` is the persisted default:

- `ask` (default) - present the choice each time (llama.cpp vs Open WebUI), with a
  "Remember my choice" option that writes your pick back to `preferred_ui`.
- `llamacpp` - always open the built-in llama.cpp web UI.
- `openwebui` - always open the managed Open WebUI service (offer to start it if it
  is installed but not running).

`--chat-ui {ask|llamacpp|openwebui}` overrides the persisted preference for one run.

Honest degrade (never a traceback): if Open WebUI is not installed or fails to become
ready, the chooser opens the built-in llama.cpp UI with a one-line reason. If no model
is running, neither UI is opened ("no model running - start one first").

## Installing Open WebUI

Open WebUI lives in its own isolated venv so its heavy, fast-moving dependency tree
never clashes with the platform venv's pins. Both the venv and Open WebUI's private
data directory are gitignored and never committed or transmitted.

```
python -m venv Codebase/.webui-venv
Codebase/.webui-venv/Scripts/pip install open-webui
```

This is a large install (open-webui plus its dependencies is roughly 2.6 GB on disk;
the first `pip install` took about 5 minutes here). A broken or absent Open WebUI can
never break `launcher.py` / `gui.py`: the service just fails readiness and the chooser
degrades to the built-in UI.

## Confirmed run interface (open-webui 0.10.2)

These were confirmed at build time against the installed release (the exact CLI and
env surface vary between releases, so they are confirmed, not guessed):

- **Invocation**: the console script `open-webui serve --host 127.0.0.1 --port 8096`
  (the dedicated venv's `Scripts/open-webui.exe`). `python -m open_webui serve` is
  NOT available in this release (the package ships no `__main__`), so the console
  script is used. Either way the child runs entirely from `.webui-venv`, never the
  platform venv.
- **Bind**: `--host 127.0.0.1` only (loopback), never `0.0.0.0`, on `ports.openwebui`
  (default 8096, inside the reserved 8080-8099 range).
- **Readiness**: HTTP `GET /health` -> 200 once the first-run database migration
  finishes. `openwebui.ready_timeout_s` (default 300 s) covers the slow first boot.
- **Backend wiring**: `OPENAI_API_BASE_URL` and `OPENAI_API_BASE_URLS` are both set to
  `http://127.0.0.1:<ports.llama_cpp>/v1` (both names are set so the wiring holds
  across releases), plus a non-empty placeholder `OPENAI_API_KEY` (llama.cpp ignores
  the key, but Open WebUI requires a non-empty value - it is a placeholder, not a
  secret).
- **Data dir**: `DATA_DIR` points at `Codebase/platform/webui-data/` (gitignored,
  private - your chat history, its sqlite DB, uploads, config).
- **Auth**: `WEBUI_AUTH=False` for local single-user use, so you are not forced to
  create an account to chat locally.
- **No model download**: `openwebui.disable_embedding_fetch: true` (default) sets
  `OFFLINE_MODE=true` (which forces `HF_HUB_OFFLINE=1`) plus the RAG auto-update
  disables, so the first-run RAG embedding-model network download does NOT happen.
  Enabling that fetch is a separate owner decision (set `disable_embedding_fetch:
  false`) and touches the network.

## Backend follows the fixed model port (a documented limitation)

Open WebUI is wired at start to the fixed `ports.llama_cpp` (8080). In the normal case
a switched model re-binds 8080, so Open WebUI keeps working across model switches
transparently. Only if a model landed on an auto-reallocated port (an 8080 conflict at
start) would Open WebUI's fixed base URL go stale; in that case use the built-in
llama.cpp UI (the other chooser branch), which always reads the live resolved port.

## Verifying it works

```
# Lifecycle: start Open WebUI, /health ready, clean stop, no orphan (JSON outcome).
python launcher.py --smoke-openwebui --json

# Functional round trip: start a model + Open WebUI, send one chat message THROUGH
# Open WebUI to llama.cpp, assert a non-empty reply, clean stop, no orphan.
python scripts/verify_chat_roundtrip.py --json

# Preference persistence: "remember my choice" writes preferred_ui and is honored.
python scripts/verify_chat_persistence.py --json
```

## Voice calls (Open WebUI Call mode on LOCITIZE speech)

With `router.enabled: true` and `router.audio: true`, Open WebUI's microphone,
read-aloud and Call buttons run entirely on this machine: the router serves
`/v1/audio/transcriptions` and `/v1/audio/speech` itself, translating onto
whisper-server and kokoro_server (audio_api.py). Nothing leaves the machine.

A spoken turn is a pipeline, and each stage was measured on the reference
machine (RTX 5070 Ti, 20-thread CPU, gemma-4-e2b loaded):

```
you stop talking
  wait      openwebui.call_silence_ms   1.0s  (Open WebUI's own value: 2.0s)
  whisper   --threads 8                 0.36s (was 0.62s on whisper's default 4)
  model     first sentence              0.12s
  Kokoro    first sentence              0.4s  CPU torch; 0.1s on the GPU build
first sound                             ~1.9s (was ~3.2s); ~1.6s with GPU Kokoro
```

Kokoro on the GPU is the setup wizard's "Hear it back, faster" feature: it
swaps the CPU torch wheel for the CUDA one (a 2.8 GB download) and
kokoro_server moves the voice model to the card whenever
`torch.cuda.is_available()` - the `/health` reply says which (`"device":
"cuda"`). Measured per sentence: 0.64-0.78s on the CPU wheel, 0.08-0.11s on
the card. The price is about 1.2 GB of VRAM the language model no longer
gets - six or seven layers of a dense 27B, measured - so the CPU wheel stays
the default: most of a second is acceptable and the model keeps the whole GPU.

That 1.2 GB is deliberately a constant. torch's allocator keeps every block an
utterance needed, sized by the longest chunk it ever rendered (measured: 594 MB
after loading, 1026 MB after one sentence, 1842 MB after a paragraph handed
over as one piece, 2422 MB after a 1000-character run-on sentence), and on
Windows a voice engine that grows does not fail - it pages the language model
through system RAM, permanently: a 27B at 29 tok/s beside an idle Kokoro ran
4.8 tok/s after Kokoro spoke one paragraph and still 4.8 twenty-five seconds
after Kokoro had shrunk again. So kokoro_server splits text into sentences,
splits a sentence over 160 characters at its commas (30 sentences rendered
one at a time peaked where one did), and renders one 160-character sentence
before `/health` says ready, so it is already at its ceiling when the launcher
measures the card to fit the model. The first spoken sentence of a call no
longer pays the 1s first-render cost either.

Five things decide whether that feels like a conversation:

- `openwebui.call_silence_ms` (settings.yaml). Open WebUI waits a fixed two
  seconds of silence before it sends what it heard; that literal lives in the
  compiled bundle, not a setting, so LOCITIZE rewrites it at Open WebUI start
  when this value is anything but 2000 (`webui.reconcile_call_silence`). 1000
  is conversational; below ~700 the recorder cuts you off at a breath. Setting
  2000 restores the upstream bundle. A pip upgrade of Open WebUI resets it,
  which is why the rewrite runs at every start.
- Voice interruption. LOCITIZE seeds Open WebUI's admin default for "Allow
  Voice Interruption in Call" on, so speaking over the reply stops it. The
  browser already requests echoCancellation. Phone-speaker bleed can still
  false-trigger; a headset is the reliable path. Set
  `openwebui.voice_interruption: false` to restore half-duplex.
- Repeated-turn microphone lifecycle. Open WebUI 0.11.1 creates a new browser
  AudioContext whenever Call mode re-arms the recorder but does not close the
  previous one. Mobile browsers eventually exhaust their live audio-context
  allowance: the already-open call then stops detecting speech while Whisper
  and every backend process remain healthy. LOCITIZE closes the previous
  analyser context before creating its replacement, preventing per-turn
  AudioContext accumulation.
- Mobile playback. Open WebUI 0.11.1 copies generated speech into a shared audio
  element that mobile browsers can reject under autoplay policy. LOCITIZE's
  startup reconciliation plays the already-fetched audio clip directly. If a
  browser still requires activation, the next tap or key press retries playback
  synchronously and unlocks later replies. Ending the call also settles pending
  playback so one rejected clip cannot leave every later turn silent.
- "Display Emoji in Call" (Settings > Interface) must stay OFF. It sends one
  extra model request per sentence, and on a `--parallel 1` llama-server that
  request queues behind the whole streaming answer - measured: the first
  sentence's audio cannot start until the entire reply has been generated.
- Thinking. A model that reasons before it answers says nothing until it has
  finished reasoning, and in a call that silence is the whole answer. The
  router handles this per turn (`router.voice_turns`, on by default): it
  served the transcription itself, so a chat request whose last user message
  is a transcript it returned in the last 10 seconds is a spoken turn, and on
  that turn only it sends `chat_template_kwargs: {enable_thinking: false}` and
  `reasoning_effort: "low"` - both, because each template honours one
  (measured on llama-server b10701 across every registered model: Qwen3 and
  Gemma read the first, 1.77s to 1.05s to the first token with zero reasoning;
  gpt-oss reads the second, 0.79s to 0.35s) - plus a one-line system note that
  the answer will be read aloud, so the model speaks in sentences rather than
  markdown tables. A value the chat already carries is never overridden, and
  the stored conversation is untouched: typed turns in the same chat still
  think. To make a model never think, give its models.yaml row
  `reasoning: {enabled: false}`. Reasoning blocks are stripped from what is
  spoken either way.
- The model the chat is on. No global model default is saved in LOCITIZE or Open
  WebUI. The model picker carried by each Open WebUI request is authoritative;
  the router starts or switches to that model on the first request. Switching to
  a model that is not loaded takes 4.5-6.8s for gemma-4-e2b and 9-17s for the 27B
  class on the reference machine. Opening the browser alone never changes models.
- Whether the model actually fits. On Windows a model that needs more VRAM
  than the card has does not fail to load: the driver pages the overflow
  through system RAM, `/health` says ok, the picker says switched, and the
  reply crawls - the owner's "when I switch models Open WebUI does not chat"
  (2026-09-03: a 27B at 5 tok/s, 100s before the first token of a hello).
  LOCITIZE's `gpu_layers: 999` used to become `--n-gpu-layers 999`, which
  turns off llama-server's own `--fit`; it now becomes `-1`, so llama-server
  measures free VRAM at load and keeps whole layers on the CPU instead of
  letting the driver page them. Whether a model overflows depends on what
  else is on the card at load time - the GPU voice engine holds ~950 MB
  once it has spoken - which is why the same model chatted fine one hour
  and crawled the next. Measured on the 16 GB card with the voice engine
  loaded, same 2840-token prompt: qwen3-8-27b 5.5 -> 32.6 tok/s (782 ->
  256 MB paged), qwen3-6-35b 11.7 -> 111, qwen3-coder-30b 30.5 -> 125.5;
  models that fit whole are unchanged (gpt-oss-20b 150, gemma-4-26b 108,
  gemma-4-e2b 209). On Windows the free-memory figure fit works from does
  not see other processes (it read 14923 MiB free in thirteen loads while
  the voice engine held 0, 594 and 948 MB), so LOCITIZE measures what they
  hold at every launch (`gpu_ledger.fit_budget`, 0.4s): what the engine will
  see (`llama-server --list-devices`) against what the card really has
  (nvidia-smi), plus a 1024 MiB safety, passed as `--fit-target`;
  launcher.log shows the arithmetic (`fit target 1224 MiB: the engine sees
  14923 MiB free, the card has 14723 MiB (1580 MiB held by other
  processes)`). The safety is the engine's own default, kept on purpose after
  a sweep in one-layer steps (gpu_ledger.py, `FIT_SAFETY_MIB`): the card's
  usable ceiling is ~15.7 GB of 16.3, fit's tally misses the ~290 MiB CUDA
  context, and the rows that beat it (62 layers of the 27B at 28 tok/s with
  536 MB paged) sat one layer from a crawl (63 layers, 646 MB paged, 6.7
  tok/s) that a paragraph of speech or a new window would have pushed them
  over - paged memory never comes back. A row wanting another value puts
  `--fit-target N` or `--fit off` in its `server_args` and wins, or sets an
  explicit `gpu_layers` count. The GPU is as full as Windows allows either
  way; more of a model on it means freeing memory - the voice engine's GPU
  build (1.2 GB), or a shorter `context_size` (the 27B's 49152 tokens are
  1.6 GB of q8_0 KV cache; 32768 would return three layers). After every switch
  the launcher measures where the new process's memory sits and writes one
  line to `locitize-data/logs/launcher.log` (`... 782 MB paged through
  system RAM; if replies crawl, that is why`), so a slow model says why.
