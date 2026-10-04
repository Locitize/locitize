import argparse
import json

import torch
from unsloth import FastLanguageModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="HF model name or local merged model dir")
    p.add_argument("--system", default="")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=120)
    args = p.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(
        input_ids=inputs, max_new_tokens=args.max_new_tokens, temperature=0.2, use_cache=True
    )
    prompt_length = inputs.shape[-1]
    reply = tokenizer.decode(outputs[0, prompt_length:], skip_special_tokens=True)

    print("[RESULT]" + json.dumps({"reply": reply}))


if __name__ == "__main__":
    main()
