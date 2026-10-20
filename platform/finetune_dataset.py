"""Review and clean a fine-tune dataset before it reaches the trainer.

Every dataset source on the Fine-tune page (a training file, documents, a word
table, a distillation run) ends up here, so the owner always sees the same
review step: how many rows there are, which ones the trainer would reject or
waste time on, and a one-click way to drop them. The cleaned rows are written in
the one format the trainer reads natively ({"messages": [...]}), so what the
owner reviewed is exactly what gets trained.

Pure functions only - no Qt, no subprocess - so the whole module is testable.
The row rules mirror finetune-studio/trainer/train.py: a row is either
{"messages": [...]} or Alpaca-style {"instruction", "input", "output"}.
"""
import json
import math
import re

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
VALID_ROLES = ("system", "user", "assistant")
# The trainer's --max-seq-length default. Longer rows are cut off, not rejected,
# so they are flagged as a warning rather than a problem.
MAX_SEQ_TOKENS = 2048
CHARS_PER_TOKEN = 4
MAX_PROMPT_FILE_BYTES = 5 * 1024 * 1024

# Training length presets: passes over the dataset ("epochs").
LENGTH_PRESETS = {
    "Quick test (1 pass)": 1,
    "Recommended (3 passes)": 3,
    "Thorough (5 passes)": 5,
}
MIN_STEPS = 10
MAX_STEPS = 2000
SMALL_DATASET_ROWS = 20


class DatasetError(ValueError):
    """The file could not be read as a dataset at all."""


class PromptFileError(ValueError):
    """A distillation prompt file could not be read safely."""


def parse_prompt_file(data, filename, max_bytes=MAX_PROMPT_FILE_BYTES):
    """Return ``(prompts, malformed_lines)`` from bounded UTF-8 file bytes.

    Text files contain one prompt per non-blank line. JSONL files contain one
    object per non-blank line with a non-blank string ``prompt`` field. This is
    deliberately pure: the Qt page owns file selection and bounded disk I/O.
    """
    if not isinstance(data, bytes):
        raise PromptFileError("the prompt file must be read as bytes")
    if len(data) > int(max_bytes):
        cap_mb = int(max_bytes) / (1024 * 1024)
        raise PromptFileError(f"the prompt file is larger than the {cap_mb:g} MB limit")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PromptFileError(
            f"the prompt file is not valid UTF-8 (byte {exc.start})"
        ) from None

    prompts = []
    malformed = []
    jsonl = str(filename).lower().endswith(".jsonl")
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if not jsonl:
            prompts.append(line)
            continue
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            malformed.append(number)
            continue
        prompt = row.get("prompt") if isinstance(row, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            malformed.append(number)
            continue
        prompts.append(prompt.strip())
    if not prompts:
        detail = format_line_numbers(malformed)
        suffix = f"; malformed {detail}" if detail else ""
        raise PromptFileError(f"the prompt file has no valid prompts{suffix}")
    return prompts, malformed


def load_prompt_file(path, max_bytes=MAX_PROMPT_FILE_BYTES):
    """Read at most ``max_bytes + 1`` bytes, then parse a prompt file."""
    from pathlib import Path

    prompt_path = Path(path)
    try:
        size = prompt_path.stat().st_size
        if size > int(max_bytes):
            cap_mb = int(max_bytes) / (1024 * 1024)
            raise PromptFileError(f"the prompt file is larger than the {cap_mb:g} MB limit")
        with prompt_path.open("rb") as handle:
            data = handle.read(int(max_bytes) + 1)
    except PromptFileError:
        raise
    except OSError as exc:
        raise PromptFileError(f"the prompt file could not be read: {exc}") from None
    return parse_prompt_file(data, prompt_path.name, max_bytes=max_bytes)


def format_line_numbers(numbers, shown=5):
    """Compact malformed-line summary suitable for a dialog."""
    values = list(numbers)
    if not values:
        return ""
    first = ", ".join(str(n) for n in values[:shown])
    more = f" and {len(values) - shown} more" if len(values) > shown else ""
    noun = "line" if len(values) == 1 else "lines"
    return f"{len(values)} malformed {noun} ({first}{more})"


def parse_dataset(data, filename):
    """Parse uploaded bytes (.jsonl, or .json holding a list or {"data": [...]}).

    Returns (rows, notes): rows are the decoded JSON values in file order, notes
    are human-readable lines about lines that were not valid JSON. Raises
    DatasetError when nothing usable could be read.
    """
    if isinstance(data, bytes):
        text = data.decode("utf-8-sig", errors="replace")
    else:
        text = str(data)
    rows, notes = [], []
    if filename.lower().endswith(".jsonl"):
        for number, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                notes.append(f"line {number} is not valid JSON and was skipped")
    else:
        try:
            loaded = json.loads(text)
        except ValueError as exc:
            raise DatasetError(f"not valid JSON: {exc}") from None
        if isinstance(loaded, dict) and isinstance(loaded.get("data"), list):
            loaded = loaded["data"]
        if not isinstance(loaded, list):
            raise DatasetError('a .json dataset must be a list of rows, or {"data": [rows]}')
        rows = loaded
    if not rows:
        raise DatasetError("the file has no rows")
    return rows, notes


def to_messages(row):
    """Normalise one row to a messages list, or None if it has neither shape."""
    if not isinstance(row, dict):
        return None
    if "messages" in row:
        messages = row["messages"]
        return messages if isinstance(messages, list) else None
    if "instruction" in row and "output" in row:
        user = str(row.get("instruction") or "")
        if row.get("input"):
            user += "\n" + str(row["input"])
        return [
            {"role": "system", "content": str(row.get("system") or DEFAULT_SYSTEM_PROMPT)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(row.get("output") or "")},
        ]
    return None


def row_problem(messages):
    """Why the trainer cannot learn from this row, or "" when it is usable."""
    if messages is None:
        return "not a chat row (needs 'messages' or 'instruction'/'output')"
    if not messages:
        return "no messages"
    for message in messages:
        if not isinstance(message, dict):
            return "a message is not an object"
        if message.get("role") not in VALID_ROLES:
            return f"unknown role '{message.get('role')}'"
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return f"empty {message.get('role')} message"
    roles = [m["role"] for m in messages]
    if "user" not in roles:
        return "no user message"
    if roles[-1] != "assistant":
        return "does not end with an assistant reply"
    return ""


def approx_tokens(messages):
    if not messages:
        return 0
    chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
    return chars // CHARS_PER_TOKEN


def _dedupe_key(messages):
    parts = []
    for m in messages:
        text = re.sub(r"\s+", " ", str(m.get("content", ""))).strip().lower()
        parts.append(f"{m.get('role')}:{text}")
    return "\n".join(parts)


def review(rows):
    """One review record per row, in order.

    Each record: index, messages (normalised, or None), problem (str), duplicate
    (bool: an identical earlier row exists), too_long (bool), tokens (int).
    """
    seen = set()
    records = []
    for index, row in enumerate(rows):
        messages = to_messages(row)
        problem = row_problem(messages)
        duplicate = False
        if not problem:
            key = _dedupe_key(messages)
            duplicate = key in seen
            seen.add(key)
        tokens = approx_tokens(messages)
        records.append({
            "index": index,
            "messages": messages,
            "problem": problem,
            "duplicate": duplicate,
            "too_long": tokens > MAX_SEQ_TOKENS,
            "tokens": tokens,
        })
    return records


def summarize(records):
    usable = [r for r in records if not r["problem"]]
    lengths = [r["tokens"] for r in usable]
    return {
        "total": len(records),
        "usable": len(usable),
        "problems": len(records) - len(usable),
        "duplicates": sum(1 for r in records if r["duplicate"]),
        "too_long": sum(1 for r in usable if r["too_long"]),
        "avg_tokens": int(sum(lengths) / len(lengths)) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
    }


def first_text(messages, role):
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == role:
            return str(m.get("content", ""))
    return ""


def last_text(messages, role):
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == role:
            return str(m.get("content", ""))
    return ""


def is_single_turn(messages):
    """True for [system?, user, assistant] - the rows whose text can be edited in a table."""
    roles = [m.get("role") for m in messages or [] if isinstance(m, dict)]
    return roles in (["user", "assistant"], ["system", "user", "assistant"])


def with_system_prompt(messages, system_prompt):
    """Return messages with the system message set to system_prompt (added if missing)."""
    rest = [m for m in messages if m.get("role") != "system"]
    return [{"role": "system", "content": system_prompt}] + rest


def curated_rows(records, keep=None, edits=None, drop_duplicates=True, system_prompt=""):
    """The rows that will be trained on, as {"messages": [...]} dicts.

    keep: optional {index: bool} from the review table (False drops the row).
    edits: optional {index: (user, assistant)} new text for single-turn rows.
    Rows with a problem are always dropped - the trainer would reject them.
    """
    keep = keep or {}
    edits = edits or {}
    out = []
    for record in records:
        if record["problem"] or not keep.get(record["index"], True):
            continue
        if drop_duplicates and record["duplicate"]:
            continue
        messages = [dict(m) for m in record["messages"]]
        if record["index"] in edits and is_single_turn(messages):
            user, assistant = edits[record["index"]]
            for m in messages:
                if m["role"] == "user":
                    m["content"] = user
                elif m["role"] == "assistant":
                    m["content"] = assistant
            if not user.strip() or not assistant.strip():
                continue
        if system_prompt.strip():
            messages = with_system_prompt(messages, system_prompt.strip())
        out.append({"messages": messages})
    return out


def recommended_steps(rows, passes, batch_size, grad_accum):
    """Training steps for `passes` full passes over `rows` examples, clamped to a sane range."""
    per_step = max(1, batch_size * grad_accum)
    steps = math.ceil(max(rows, 1) * passes / per_step)
    return max(MIN_STEPS, min(MAX_STEPS, steps))


def safe_name(text, fallback):
    """A file/folder-safe name: letters, digits, '-' and '_' only."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (text or "").strip()).strip("-")
    return cleaned or fallback


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
