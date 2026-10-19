import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import gguf_export
from common import load_4bit

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def log(msg):
    print(f"[STATUS] {msg}", flush=True)


def to_messages(row):
    """Accepts either {"messages": [...]} directly, or Alpaca-style
    {"instruction", "input", "output"} rows, and normalizes to messages."""
    if "messages" in row:
        return row["messages"]
    if "instruction" in row and "output" in row:
        user_content = row["instruction"]
        if row.get("input"):
            user_content += "\n" + row["input"]
        return [
            {"role": "system", "content": row.get("system", DEFAULT_SYSTEM_PROMPT)},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["output"]},
        ]
    raise ValueError(
        "Every row must have either a 'messages' field ([{role, content}, ...]) "
        "or Alpaca-style 'instruction'/'output' fields."
    )


def load_messages_dataset(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        else:
            data = json.load(f)
            rows = data if isinstance(data, list) else data["data"]
    rows = [{"messages": to_messages(r)} for r in rows]
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", default="run")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--gguf-quant", default="f16", choices=["f16", "q8_0", "q4_k_m"])
    args = p.parse_args()

    out_root = os.path.join(args.output_dir, args.run_name)
    lora_dir = os.path.join(out_root, "lora_adapter")
    merged_dir = os.path.join(out_root, "merged_model")
    os.makedirs(out_root, exist_ok=True)

    log(f"Loading dataset from {args.dataset}")
    dataset = load_messages_dataset(args.dataset)
    log(f"Dataset loaded: {len(dataset)} examples")

    log(f"Loading base model {args.model} (4-bit)")
    model, tokenizer = load_4bit(args.model)

    log("Formatting dataset with chat template")

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    formatted_dataset = dataset.map(format_example, remove_columns=["messages"])

    log("Attaching LoRA adapters")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))

    log("Starting training")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_dataset,
        args=SFTConfig(
            output_dir=os.path.join(out_root, "checkpoints"),
            dataset_text_field="text",
            max_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=min(5, max(args.max_steps // 10, 0)),
            seed=3407,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        ),
    )
    result = trainer.train()
    log(f"Training complete. final_loss={result.training_loss:.4f}")

    log(f"Saving LoRA adapter to {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Free the GPU before the merge: the merge reloads the base weights at full
    # precision on the CPU, so nothing here needs VRAM any more.
    del trainer, model
    torch.cuda.empty_cache()

    log(f"Merging LoRA into base model -> {merged_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, lora_dir).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del merged, base

    log(f"Converting merged model to GGUF ({args.gguf_quant})")
    gguf_path = gguf_export.export_gguf(
        merged_dir, out_root, args.run_name, args.gguf_quant,
        python_exe=sys.executable, say=log,
    )

    log("DONE")
    print(
        f"[ARTIFACTS] lora={lora_dir} merged={merged_dir} gguf={gguf_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
