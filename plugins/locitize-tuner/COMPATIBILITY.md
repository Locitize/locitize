# LoCiTiZe model compatibility

Measured results for specific local models used with Codex through the LoCiTiZe. Nothing here is inferred from a model's name. A model that is not listed is untested, not unsupported. Every result was measured on one Windows PC (RTX 5070 Ti 16 GB, 32 GB RAM) with Codex CLI 0.157 and llama.cpp b11269; other hardware, versions and quantizations can behave differently.

## Provisional recommendation

Start with Qwen3-VL-8B-Instruct (Q8_0 with its mmproj projector) when you have no preference. It passed every tool and vision check below with exact output formatting. This is provisional: it rests on the runs below, not a broad benchmark. Ministral 3 8B is a reasonable lighter and faster choice for text-only work.

## How results are reported

Three things are counted separately for every run, using a strict checker:

- Tool success: Codex executed the fixture reader MCP tool and it returned the fixture.
- Content correctness: defect `width * height`, a correct perimeter expression (checked on several inputs), and example result 16. Each field is judged on its own, so one malformed field does not hide the others.
- Format compliance: the reply is only the requested JSON (tool task) or only the three color words (vision task).

A strict pass needs a completed turn, tool success, all content fields correct, unchanged files and no other tools. Fenced JSON still passes strictly but does not count as format compliant. An expression written as a statement (for example `return 2 * (width + height)`) is not an expression and fails.

## Codex tool task: 10 identical runs per model

Packaged standalone install, saved tuning (16384 context, all layers on GPU), one model load per batch, same prompt.

| | Qwen3-VL-8B-Instruct Q8_0 | Ministral 3 8B Instruct 2512 Q5_K_M |
|---|---|---|
| Turn completed | 10/10 | 10/10 |
| Tool success | 10/10 | 10/10 |
| Content correct | 10/10 | 10/10 |
| Strict pass | 10/10 | 10/10 |
| Format compliant (bare JSON) | 10/10 | 8/10 (2 wrapped in a code fence) |
| Files unchanged, no other tools | 10/10 | 10/10 |
| Generation speed (tuning measurement) | 88 tokens/s | 115 tokens/s |
| VRAM while serving | 13.5 GB (projector loaded) | 9.4 GB |

Ministral 3 8B outside this batch: single runs on the same day produced correct content with format lapses that failed the strict check: a `# comment` inside the JSON, prose after the JSON, and expressions written as `return ...` statements (the last from a separate source install with 32 GPU layers). Treat its format as variable; its content was correct in every run observed.

## Vision

Vision needs a projector file (mmproj) paired with the model; setup pairs one only when it can match it to the model unambiguously. `list_local_models` reports `vision_projector_configured`, which means configured, not verified.

| | Qwen3-VL-8B-Instruct Q8_0 | Ministral 3 8B (no projector on disk) |
|---|---|---|
| Codex `-i` image, three color stripes in random order, 10 runs | 10/10 correct, 10/10 exact format, no tools used | Not available |
| Direct and router image requests | Correct | HTTP 500 "image input is not supported" |

This covers simple color and position perception only. Reading text in images, charts and fine detail are untested.

## Not yet tested

- Gemma 4 26B A4B: projector paired, not run.
- Other registered models: not run with Codex.
- Longer Codex sessions, file edits and shell commands with a local model.
- Web search and custom (non-function) tools are not supported by the local engine for any model.
