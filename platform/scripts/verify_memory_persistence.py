"""AC16 harness: prove real conversation-memory persistence and recall.

The memory-engine functional proof for Milestone 8. Unlike the vision/LLM proofs,
this needs no GPU: it exercises REAL filesystem I/O through the real AssistantLoop
and the real ConversationMemory store. It drives a short scripted assistant/memory
round trip (text mode, fixed content, an inline canned LLM so no model server is
needed) and asserts, with no fabrication:

  1. a NEW per-session JSONL transcript file is created under a real memory dir,
  2. the just-appended user + assistant content is found again by search()/recall(),
  3. the file was only APPENDED to across a second turn -- the bytes written by the
     first turn are unchanged (append-only, Data Model 9.3).

The canned LLM stands in for the network seam only; the memory store, the append,
the JSONL file, and the search are all the real production code paths. Run from
Codebase/platform:  python scripts/verify_memory_persistence.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PLATFORM_DIR = Path(__file__).resolve().parent.parent
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from assistant import AssistantLoop, ConversationState  # noqa: E402
from memory import ConversationMemory  # noqa: E402

# Fixed scripted content with an unambiguous keyword we later search for.
_KEYWORD = "kestrel"
_USER_LINE = f"remember the project codename is {_KEYWORD}"
_ASSISTANT_REPLY = f"Understood, the codename {_KEYWORD} is noted."
_SESSION_ID = "ac16-memory-proof"


class _CannedLlm:
    """Minimal LlmClient stand-in: yields one fixed reply, counts nothing.

    Stands in for the network seam only so this proof needs no model server; every
    memory operation below is the real production code path.
    """

    def chat_stream(self, messages: Any, interrupt: Any = None) -> Any:
        yield _ASSISTANT_REPLY

    def chat(self, messages: Any) -> str:
        return _ASSISTANT_REPLY

    def count_tokens(self, text: str) -> int | None:
        return None


def run_round_trip() -> dict:
    """Drive the real assistant/memory round trip and return a verdict summary."""
    summary: dict = {
        "outcome": "failed",
        "reason": "",
        "jsonl_path": "",
        "file_created": None,
        "search_found": None,
        "recall_found": None,
        "append_only": None,
    }

    # A real, throwaway memory dir (real filesystem I/O, isolated from production).
    mem_root = Path(tempfile.mkdtemp(prefix="locitize-ac16-mem-"))
    memory = ConversationMemory(mem_root)

    state = ConversationState(session_id=_SESSION_ID)
    state.append("system", "You are LOCITIZE.")
    loop = AssistantLoop(
        _CannedLlm(), _CannedLlm(), None, state, memory=memory
    )

    # --- turn 1: writes user + assistant lines to a NEW JSONL file --- #
    loop.run_turn(_USER_LINE)
    jsonl = memory.path_for(_SESSION_ID)
    summary["jsonl_path"] = str(jsonl)
    summary["file_created"] = jsonl.is_file()
    if not summary["file_created"]:
        summary["reason"] = "no per-session JSONL transcript file was created"
        return summary

    bytes_after_turn1 = jsonl.read_bytes()

    # --- recall/search find the just-appended content --- #
    hits = memory.search(_KEYWORD, limit=5)
    summary["search_found"] = any(_KEYWORD in h.text for h in hits)
    recalled = memory.recall(limit=10)
    summary["recall_found"] = any(_KEYWORD in e.text for e in recalled)

    # --- turn 2: proves append-only (turn-1 bytes are a prefix, untouched) --- #
    loop.run_turn("and the deadline is friday")
    bytes_after_turn2 = jsonl.read_bytes()
    grew = len(bytes_after_turn2) > len(bytes_after_turn1)
    prefix_intact = bytes_after_turn2.startswith(bytes_after_turn1)
    summary["append_only"] = bool(grew and prefix_intact)

    # Validate the exact JSONL schema of every stored line (Data Model 9.3).
    schema_ok = True
    for raw in bytes_after_turn2.decode("utf-8").splitlines():
        if not raw.strip():
            continue
        obj = json.loads(raw)
        if set(obj.keys()) != {"session_id", "ts", "role", "text"}:
            schema_ok = False
            break

    # --- assertions --- #
    if not summary["search_found"]:
        summary["reason"] = f"search('{_KEYWORD}') did not find the appended content"
        return summary
    if not summary["recall_found"]:
        summary["reason"] = "recall() did not return the appended content"
        return summary
    if not summary["append_only"]:
        summary["reason"] = "second turn did not append-only (prefix changed or no growth)"
        return summary
    if not schema_ok:
        summary["reason"] = "a stored JSONL line did not match the {session_id,ts,role,text} schema"
        return summary

    summary["outcome"] = "ok"
    summary["reason"] = "real JSONL created, found by search/recall, append-only, schema ok"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AC16 real memory-persistence verifier")
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    args = parser.parse_args(argv)

    summary = run_round_trip()

    if args.json:
        print(json.dumps(summary))
    else:
        print(f"jsonl_path  : {summary['jsonl_path']}")
        print(f"file_created: {summary['file_created']}")
        print(f"search_found: {summary['search_found']}")
        print(f"recall_found: {summary['recall_found']}")
        print(f"append_only : {summary['append_only']}")
        print(f"outcome     : {summary['outcome']}")
        if summary["reason"]:
            print(f"reason      : {summary['reason']}")
    return 0 if summary["outcome"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
