# Fine-tune trainer

The engine behind the desktop **Fine-tune** page: QLoRA training on your own NVIDIA GPU,
with no Docker. You give it example conversations; it gives back a LoRA adapter, a merged
model and a GGUF file you can run in LoCiTiZe, llama.cpp, Ollama or LM Studio.

You do not run anything here by hand. The Fine-tune page walks through four steps -
choose a model, prepare and review the dataset, train, save - and runs these scripts for you.

## One-time setup

Tick **Fine-tune models** in LoCiTiZe setup (or Settings > Features). It builds
`finetune-studio/.venv` with CUDA PyTorch (`torch==2.11.0+cu128`, which includes RTX 50-series
kernels) and the pinned QLoRA stack (transformers, peft, trl, bitsandbytes, accelerate; see
`PIP_SETS["finetune_train_pip"]` in `platform/setup_env.py`) plus `pypdf` for building datasets
from PDF files - about 4 GB. llama.cpp (installed by setup, release pinned in
`setup_env.LLAMA_CPP_TAG`) supplies `llama-quantize`. The first GGUF export downloads the matching
`convert_hf_to_gguf.py` (about 30 MB) once.

Manual equivalent, inside `finetune-studio/.venv`:
```
pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple torch==2.11.0+cu128
pip install transformers==5.18.0 peft==0.21.2 trl==1.14.1 accelerate==1.15.0 bitsandbytes==0.50.2 datasets==5.0.1 sentencepiece==0.2.2 protobuf==7.36.2 numpy==2.4.6 pypdf
```

## What is in `trainer/`

| Script | What the Fine-tune page uses it for |
|---|---|
| `preflight.py` | Step 1 and Start training: is there an NVIDIA GPU, are the packages installed, is there enough free VRAM? Answers in plain language (`--model <id> --batch-size N` prints JSON). |
| `build_dataset.py` | Step 2: turn documents (.txt .md .csv .docx .pdf) or a reviewed word table into training rows. |
| `distill.py` | Step 2, "Have a bigger model write it": a teacher model answers your prompts. |
| `train.py` | Step 3: 4-bit LoRA training, merge, and GGUF export (via `gguf_export.py`). |
| `infer.py` | Step 4, "Compare": the same message to the original and the fine-tuned model. |

Runs are written to `<data root>/finetune/outputs/<run>/`:
- `lora_adapter/` - small adapter weights, load on top of the base model
- `merged_model/` - full standalone fine-tuned model
- `<run>.<quant>.gguf` - single-file model; step 4 saves it to a folder you choose and adds it to your model list

Datasets you prepare in step 2 are saved to `<data root>/finetune/datasets/`.

## Dataset format

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```
One JSON object per line (`.jsonl`), or a JSON array of these objects (`.json`). Alpaca-style
`{"instruction", "input", "output"}` rows are accepted too. Step 2 shows every row, flags the
ones the trainer cannot use, and drops duplicates before anything is trained.

## Notes

- The smallest built-in base model is `unsloth/Qwen2.5-0.5B-Instruct` (a whole run takes a few
  minutes on an RTX 5070 Ti); `Qwen2.5-1.5B-Instruct` is the best first real choice. 7B/8B models
  also fit in 16 GB at 4-bit but take longer.
- Step 3 suggests a number of training steps from your dataset size (1, 3 or 5 passes over it);
  Advanced settings let you override it.
- GGUF quality: `q4_k_m` (smallest, fastest, recommended), `q8_0` (closer to the original), `f16`
  (full quality, largest).
