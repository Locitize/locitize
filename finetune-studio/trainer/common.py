"""Model loading shared by train.py, infer.py and distill.py (torch required)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_4bit(model_name):
    """Load a causal LM quantized to 4-bit NF4 (the QLoRA recipe) on the GPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map={"": 0},
        dtype=torch.bfloat16,
    )
    return model, tokenizer


def chat_reply(model, tokenizer, messages, max_new_tokens=120, temperature=0.2):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    sample = temperature > 0
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=sample,
        temperature=temperature if sample else None, use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True).strip()
