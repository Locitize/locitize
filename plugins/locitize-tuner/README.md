# LoCiTiZe standalone local test

LoCiTiZe (pronounced LOCK-ih-tize).

The standalone ZIP contains a lightweight Windows helper. It does not require the full LoCiTiZe desktop, an existing checkout, or LOCITIZE_INSTALL_DIR. Python 3.11+ must be installed. Your GGUF weights remain on your computer and are never copied into the package.

## Setup outside the checkout

Extract locitize-tuner-standalone.zip to a folder of your choice. Open a terminal there and run:

```powershell
python bootstrap.py
```

Setup creates an isolated environment and configuration in .locitize-tuner in your user profile folder (not AppData: the Codex desktop app is a Store app whose helpers' AppData writes Windows redirects into the app's private folder, where the Codex CLI cannot see them). An earlier AppData install, including the desktop app's redirected copy, is copied there once on first start; the old copy is left in place. It installs MCP dependencies through the existing Python toolchain installer, acquires the CPU or NVIDIA llama.cpp build through the existing host-checked publisher-checksum download path, and discovers local GGUF weights. Downloads require an internet connection. It installs no desktop, torch/training, whisper/Kokoro, separate vision service, or Open WebUI packages/assets. Image input uses llama.cpp itself when setup can pair a model with a compatible mmproj projector in the same folder or a matched neighboring family folder.

To restrict discovery to one local folder:

```powershell
python bootstrap.py --models-root "YOUR_LOCAL_MODEL_FOLDER"
```

For hardware and protocol inspection only, use --skip-runtime --skip-discovery. This deliberately leaves model serving unavailable and the model list empty; it is not a completed inference installation.

LOCITIZE_TUNER_DATA_DIR can select another private helper data directory before setup and server startup. It is a local administrator setting, never an MCP tool argument. The helper does not inherit another LoCiTiZe installation's data root.

## Connect locally

The plugin's mcp.json starts python server.py. The standalone server selects its private interpreter and bundled runtime automatically. You can also run python server.py directly from the extracted folder. Standard output belongs to the MCP protocol.

Inspect hardware and models first, select a registered launchable model, and request balanced or speed tuning. Progress reports tested configurations and observed context/offload. Cancel waits for the in-flight bounded probe to clean up. Busy resources refuse safely. A saved result is best among tested candidates, with a real completion verification; it is not a universal optimum or answer-quality guarantee.

For Codex, start_local_model returns the local Responses router endpoint and observed effective context. Use process-only provider/context settings. Browser chat is separate and optional.

## Local models in the Codex CLI model picker (0.3.3)

enable_codex_picker(model_ids) lists tuned local models (saved context of at least 32768 tokens) in a separate Codex profile. The profile turns off ChatGPT app connectors for local sessions only: on a measured setup they declared about 175k tokens of tools, far beyond any local model's context; your plugins, MCP servers and skills stay on (about 25k tokens on that setup). It writes only two files in your Codex home, locitize.config.toml and locitize-models.json, and never reads or writes config.toml. Keep the model running with `python server.py --serve` in one terminal (Ctrl+C stops it and its model), then run `codex --profile locitize` in another and choose models with /model. Choosing another picker model loads it in place of the current one (one at a time; a switch takes as long as a model load). Plain `codex` and the desktop app keep their normal models.

Desktop app local mode (0.3.4): the desktop app has no profiles and cannot mix local and GPT models, so `python server.py --desktop-local on` switches your Codex config to the picker models (GPT models are hidden in the desktop app until you switch back) and `python server.py --desktop-local off` restores the exact previous lines. Close the desktop app before either command and keep `python server.py --serve` running while you use it. It changes only model, model_provider, model_catalog_json, apps in [features] and one marked provider block; every edit is checked by parsing and a backup is kept. disable_codex_picker removes only the two helper files. The profile uses Codex's own default instructions, captured once from your installed Codex over loopback.

Public discovery fix (0.3.5): files identified as llama.cpp importance matrices are excluded from runnable model selections and fit rankings, even when a publisher gives them a `.gguf` extension.

## Which model to use

Provisional starting model for Codex: Qwen3-VL-8B-Instruct with its vision projector. Measured, model-specific results are in COMPATIBILITY.md.

## Scope and size

The archive includes explicit shared Python source dependencies and minimal empty configuration templates. No environment, binaries, weights, user registry, transcripts, or credentials are bundled. Python dependencies and llama.cpp binaries are downloaded separately, so archive size and installed size differ substantially. Small audio/finetune support modules remain because current shared registry/router code imports them; optional engines and their packages are not installed.

This is a local test artifact. A complete install requires successful runtime acquisition and a real model launch. Public marketplace submission and remote ChatGPT connection remain separate verification gates.

For source development, build the complete marketplace and versioned archive from the repository root:

```powershell
python platform/scripts/build_tuner_marketplace.py
```

The repository catalog points to the generated complete payload under dist/locitize-tuner-marketplace-0.3.5-desktop. The generated marketplace can be moved outside the checkout with its catalog and plugins folder intact. Register its root with `codex plugin marketplace add YOUR_MARKETPLACE_FOLDER`, then use `codex plugin add locitize-tuner@locitize-local`. This preserves the existing marketplace/plugin identity. Reinstallation uses the versioned source through the host; never edit an installed cache. Run bootstrap.py once from the installed complete package if the helper environment has not been prepared. An incomplete wrapper-only package refuses startup. Verify tool discovery in the intended Codex chat after installation; CLI discovery alone does not prove desktop availability.


## Public Hugging Face discovery (0.2.0)

The helper adds search_models, list_model_files and recommend_model_candidates for bounded metadata-only suggestions. Recommendations estimate current free memory/disk; they do not measure speed or prove capability. Asking for suggestions never downloads weights. Explicit start_model_download uses an opaque inspected selection pinned to an immutable commit with known file sizes and publisher SHA256. Poll download_status; cancel_download waits for the current checked transfer to clean up. resume_download restarts an interrupted file and reuses only fully rehashed completed files; job IDs live only in the connected helper session. Verified files remain in managed storage until separately registered with register_download. No automatic tuning or model start occurs.

Public ungated repositories only: no token authentication, gated/license bypass or unverified digest override. Missing/unknown license is disclosed, not invented. Model weights-only downloads do not include or pair vision projectors in this update; vision pipeline metadata is reported and untested, and separate verified projector setup is required. Existing registered and verified vision models continue to work.

Metadata responses are capped at4MiB, search at five pages/ten items per page, selections at100 and active transfer at one. The egress ledger records initial checked request hosts/reasons under the private data root; redirects are host-validated but not separately enumerated in the ledger. No repository code is executed. Modelhub's bounded socket reads may delay cancellation until the current read completes. Package/dependency size is independent of downloaded model weights.


## Recommendation usability (0.2.1)

The skill directs the agent to report hardware first and give stage updates, use bounded workload-led searches without a preferred model family, and distinguish historically tested installed compatibility from current HF choices. Reported HF config.model_type groups viable family alternatives; missing metadata stays unknown. The shortlist explains when fewer than three reported viable families were inspected. Public file summaries omit duplicate internal digests and revision fields; opaque selections still bind the complete private immutable revision/digest file set for verification. Workload values are exactly general or coding. Windows CPU lookup uses a read-only standard-library registry fallback when the primary name is unavailable or only an architecture label.


## Adaptive balanced tuning (0.3.0)

Balanced tuning measures a bounded offload neighborhood and growing contexts from8192 through the documented native limit when feasible, regardless of a saved4096 baseline. GGUF block_count informs coarse layer requests; missing metadata is explicitly unknown and uses a bounded fallback. Actual requested and effective context/offload are separate. At most16 exploratory trials and two final completions are permitted. Among observed results at context8192 or greater and20tokens/sec or greater, selection retains the largest tested context reaching85percent of the best eligible measured throughput. Preferred30tokens/sec is reported separately, never guaranteed. Two eligible final completions must agree on effective context and retain that throughput threshold before an atomic compare-and-write. Missing targets or divergent verification returns target_not_met with unchanged saved settings; cancellation/recovery and concurrent-edit safeguards remain. Progress includes phase, requests, budgets and a rough duration projection; model/hardware speed is never inferred. Optional layer metadata scans stop before tokenizer arrays and have conservative key/byte bounds. Explicit speed tuning retains its prior comparison behavior. No tuning policy promises global optimality, model quality or perfect session recall.
