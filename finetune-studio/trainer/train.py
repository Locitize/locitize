import argparse
import json
import os
import subprocess
import sys

from unsloth import FastLanguageModel  # must import before trl/transformers/peft

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

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
    gguf_path = os.path.join(out_root, f"{args.run_name}.{args.gguf_quant}.gguf")
    os.makedirs(out_root, exist_ok=True)

    log(f"Loading dataset from {args.dataset}")
    dataset = load_messages_dataset(args.dataset)
    log(f"Dataset loaded: {len(dataset)} examples")

    log(f"Loading base model {args.model} (4-bit)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    log("Formatting dataset with chat template")

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    formatted_dataset = dataset.map(format_example)

    log("Attaching LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_length,
    )

    log("Starting training")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            output_dir=os.path.join(out_root, "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            logging_steps=1,
            report_to="none",
        ),
    )
    result = trainer.train()
    log(f"Training complete. final_loss={result.training_loss:.4f}")

    log(f"Saving LoRA adapter to {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    log(f"Merging LoRA into base model -> {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer=tokenizer)

    log(f"Converting merged model to GGUF ({args.gguf_quant}) -> {gguf_path}")
    convert_script = "/workspace/llama.cpp/convert_hf_to_gguf.py"
    outtype = "f16" if args.gguf_quant == "f16" else "f16"
    tmp_gguf = os.path.join(out_root, f"{args.run_name}.f16.gguf")
    subprocess.run(
        [sys.executable, convert_script, merged_dir, "--outfile", tmp_gguf, "--outtype", outtype],
        check=True,
    )
    if args.gguf_quant != "f16":
        quantize_bin = "/workspace/llama.cpp/build/bin/llama-quantize"
        subprocess.run([quantize_bin, tmp_gguf, gguf_path, args.gguf_quant], check=True)
        os.remove(tmp_gguf)
    else:
        gguf_path = tmp_gguf

    log("DONE")
    print(
        f"[ARTIFACTS] lora={lora_dir} merged={merged_dir} gguf={gguf_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
