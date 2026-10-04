"""Reviewed-word-table -> combined training dataset.

Turns a markdown table of {word, meaning, example sentence} rows into a
multi-phrasing-template Alpaca dataset, oversamples an identity/persona
dataset alongside it, shuffles, and writes the combined result. This is the
same process a by-hand fine-tune pipeline uses (a vocab dataset builder plus the
manual cat/shuffle steps), made reusable from the UI so a future run doesn't
need one-off scripts.
"""
import json
import random
import re


def parse_word_table(md_text):
    """Parse a markdown table with columns: #, Word, Meaning, Example.

    Example cell format: "Sheng sentence — English translation" (em dash or
    hyphen separator). Rows missing a meaning or example are skipped.
    """
    entries = []
    for line in md_text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or line.lower().startswith("| #"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 4:
            continue
        _, word, meaning, example = cols[:4]
        if not word or not meaning or not example:
            continue
        m = re.match(r"^(.*?)\s*[—-]\s*(.*)$", example)
        if not m:
            continue
        sheng_sentence, english_sentence = m.group(1).strip(), m.group(2).strip()
        entries.append(
            {
                "word": word,
                "meaning": meaning,
                "sheng_sentence": sheng_sentence,
                "english_sentence": english_sentence,
            }
        )
    return entries


_ALL_TEMPLATES = [
    lambda w, m, ss, es: (
        f"Define the Sheng word '{w}'.",
        f"'{w}' means '{m}'. Example: {ss} ({es})",
    ),
    lambda w, m, ss, es: (
        f"What does '{w}' mean in Sheng?",
        f"'{w}' means '{m}'. Example: {ss} ({es})",
    ),
    lambda w, m, ss, es: (
        f"Can you make a sentence using '{w}'?",
        f"Sure: {ss} ({es}) -- '{w}' means '{m}'.",
    ),
    lambda w, m, ss, es: (
        f"Teach me the Sheng way to say '{m}'.",
        f"'{m}' in Sheng is '{w}'. Example: {ss} ({es})",
    ),
    lambda w, m, ss, es: (
        f"I heard someone say '{ss}' -- what does that mean?",
        f"That means: {es} -- it uses '{w}', which means '{m}'.",
    ),
    lambda w, m, ss, es: (f"Please translate: '{ss}'", f"{es}"),
    lambda w, m, ss, es: (
        f"How would you say '{es}' in Sheng?",
        f"{ss} -- using '{w}' ('{m}').",
    ),
    lambda w, m, ss, es: (
        f"Give me a Sheng slang word that means '{m}'.",
        f"How about '{w}'? It means '{m}'. Example: {ss} ({es})",
    ),
    lambda w, m, ss, es: (
        f"Fill in the blank: in Sheng, '___' means '{m}'.",
        f"'{w}' -- as in: {ss} ({es})",
    ),
    lambda w, m, ss, es: (
        f"Pop quiz -- what's the Sheng word for '{m}'?",
        f"'{w}'. Example: {ss} ({es})",
    ),
    lambda w, m, ss, es: (
        f"Show me how to use '{w}' in a sentence.",
        f"{ss} ({es}) -- that's '{w}' in action.",
    ),
    lambda w, m, ss, es: (
        f"What's the street term for '{m}'?",
        f"'{w}'. Example: {ss} ({es})",
    ),
    lambda w, m, ss, es: (
        f"Complete this: the Sheng word for '{m}' is ___.",
        f"'{w}'",
    ),
    lambda w, m, ss, es: (f"Say this in Sheng: '{es}'", f"{ss}"),
    lambda w, m, ss, es: (
        f"How would a Kenyan casually say '{es}'?",
        f"{ss} -- using '{w}' ('{m}').",
    ),
    lambda w, m, ss, es: (
        f"Explain the Sheng word '{w}' for a beginner.",
        f"'{w}' means '{m}'. Example: {ss} ({es})",
    ),
]


def build_vocab_rows(entries, templates_per_word=16):
    """Generate `templates_per_word` phrasing variants per entry (capped at
    the number of templates available)."""
    templates_per_word = max(1, min(templates_per_word, len(_ALL_TEMPLATES)))
    rows = []
    for e in entries:
        w, m, ss, es = e["word"], e["meaning"], e["sheng_sentence"], e["english_sentence"]
        for template in _ALL_TEMPLATES[:templates_per_word]:
            instruction, output = template(w, m, ss, es)
            rows.append({"instruction": instruction, "input": "", "output": output})
    return rows


def load_jsonl_rows(path_or_bytes):
    """Load Alpaca-format rows from a .jsonl file path or raw bytes."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        text = path_or_bytes.decode("utf-8")
    else:
        with open(path_or_bytes, "r", encoding="utf-8") as f:
            text = f.read()
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def combine_and_shuffle(vocab_rows, identity_rows, identity_oversample, seed=42):
    """Oversample identity_rows `identity_oversample` times, combine with
    vocab_rows, and shuffle deterministically."""
    combined = list(vocab_rows)
    for _ in range(identity_oversample):
        combined.extend(identity_rows)
    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined


def write_jsonl(rows, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
