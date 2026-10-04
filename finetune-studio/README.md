# LLM Finetune Studio

Upload a chat-format dataset, click Start, get back a LoRA adapter + merged model + GGUF file.
Training runs in a Docker container on your local GPU (tested on RTX 5070 Ti, 16GB VRAM) - no manual CUDA/Python setup needed.

## One-time setup

1. Build the training image (takes a while the first time - installs CUDA toolchain, unsloth, and builds llama.cpp):
   ```
   docker build -t llm-finetune-trainer ./trainer
   ```
2. Install the UI's Python dependency:
   ```
   pip install -r app/requirements.txt
   ```

## Run it

```
streamlit run app/app.py
```

Opens a local page. Upload a `.jsonl` dataset (one JSON object per line, each with a `messages` list -
one JSON object per line with an `instruction`/`response` pair), pick a base model and settings, click **Start training**.

Progress streams live in the page. When done, you'll see paths to:
- `outputs/<run>/lora_adapter/` - small adapter weights, load on top of the base model
- `outputs/<run>/merged_model/` - full standalone fine-tuned model
- `outputs/<run>/<run>.<quant>.gguf` - single-file export for `llama.cpp` / Ollama / LM Studio

### Before/after chat test

Once a run finishes, a "Chat test" panel appears - type a prompt, hit **Compare replies**, and it runs
the same message through both the original base model and your fine-tuned merged model side by side,
so you can see the fine-tune's effect directly. The base model's weights are cached in a Docker volume
(`hf-cache`) after the first load, so repeat comparisons are fast.

## Dataset format

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```
One JSON object per line (`.jsonl`), or a JSON array of these objects (`.json`).

## Notes

- Default base model is `unsloth/Qwen2.5-1.5B-Instruct` - fast, fits easily in 16GB VRAM. 7B/8B models also fit at 4-bit but take longer.
- `max_steps` controls how long training runs - 60 is a quick smoke test; real fine-tunes usually want several hundred to a few thousand steps depending on dataset size.
- GGUF quantization: `f16` (largest, best quality), `q8_0` (balanced), `q4_k_m` (smallest, fastest, some quality loss) - good default for local chat use.
