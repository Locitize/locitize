import argparse
import json

from common import chat_reply, load_4bit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="HF model name or local merged model dir")
    p.add_argument("--system", default="")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=120)
    args = p.parse_args()

    model, tokenizer = load_4bit(args.model_path)
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    reply = chat_reply(model, tokenizer, messages, args.max_new_tokens)
    print("[RESULT]" + json.dumps({"reply": reply}))


if __name__ == "__main__":
    main()
