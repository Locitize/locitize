import argparse
import json

import torch
from unsloth import FastLanguageModel


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    prompts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if path.endswith(".jsonl"):
            row = json.loads(line)
            prompts.append(row["prompt"] if isinstance(row, dict) else row)
        else:
            prompts.append(line)
    return prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", required=True, help="HF model name or local merged model dir")
    p.add_argument("--system", default="")
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    prompts = load_prompts(args.prompts_file)
    total = len(prompts)
    print(f"[STATUS] Loading teacher model {args.teacher_model}", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.teacher_model,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"[STATUS] Teacher loaded. Distilling {total} prompts.", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, prompt in enumerate(prompts, start=1):
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": prompt})

            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)

            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                use_cache=True,
            )
            prompt_length = inputs.shape[-1]
            reply = tokenizer.decode(outputs[0, prompt_length:], skip_special_tokens=True).strip()

            row = {"messages": messages + [{"role": "assistant", "content": reply}]}
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"[DISTILL] {i}/{total}", flush=True)

    print("[STATUS] DONE", flush=True)


if __name__ == "__main__":
    main()
