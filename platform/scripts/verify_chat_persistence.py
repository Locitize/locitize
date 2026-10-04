"""AC21 harness: prove real chat-UI preference persistence (M9-lite, "remember").

The "remember my choice" functional proof. It needs no GPU and no services: it
exercises the REAL config.write_chat_ui targeted atomic write against a REAL
settings.yaml on disk (a throwaway copy of the shipped file, so its real comment/key
layout is exercised) and then the REAL resolve_chat_choice decision function, with no
fabrication:

  1. copy the shipped settings.yaml to an isolated temp dir (real file, real comments),
  2. call config.write_chat_ui(base, "openwebui") -- the same targeted atomic sibling
     of write_model_fields the GUI/menu "remember" path invokes,
  3. re-read the file from disk and assert ONLY chat.preferred_ui changed (every
     comment and every other key preserved byte-for-byte),
  4. reload through the real Config.load and assert chat.preferred_ui == openwebui,
  5. run a fresh resolve_chat_choice pass with the remembered preference and assert it
     now resolves straight to OPEN_OPENWEBUI (skips ASK) -- proving the remembered UI
     is honored, not re-prompted.

Every step is the real production code path against a real on-disk file. Run from
Codebase/platform:  python scripts/verify_chat_persistence.py --json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

PLATFORM_DIR = Path(__file__).resolve().parent.parent
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from config import Config, write_chat_ui  # noqa: E402
from webui import ChatDecision, resolve_chat_choice  # noqa: E402


def run_persistence_check() -> dict:
    """Drive the real write_chat_ui + reload + chooser round trip; return a verdict."""
    summary: dict[str, Any] = {
        "outcome": "failed",
        "reason": "",
        "settings_path": "",
        "only_scalar_changed": None,
        "comments_preserved": None,
        "reloaded_preference": None,
        "chooser_skips_ask": None,
    }

    shipped = PLATFORM_DIR / "settings.yaml"
    if not shipped.is_file():
        summary["reason"] = f"shipped settings.yaml not found at {shipped}"
        return summary

    # A real, throwaway settings dir seeded from the shipped file (real comments,
    # real key order), isolated from production so the proof never mutates the vault.
    work = Path(tempfile.mkdtemp(prefix="locitize-ac21-chat-"))
    target = work / "settings.yaml"
    shutil.copyfile(shipped, target)
    summary["settings_path"] = str(target)

    before = target.read_text(encoding="utf-8")
    before_lines = before.splitlines()

    # --- the real "remember my choice" write (targeted, atomic) --- #
    write_chat_ui(work, "openwebui")

    after = target.read_text(encoding="utf-8")
    after_lines = after.splitlines()

    # Exactly one line changed, and it is the preferred_ui scalar.
    changed = [
        (b, a)
        for b, a in zip(before_lines, after_lines)
        if b != a
    ]
    same_line_count = len(before_lines) == len(after_lines)
    only_one = len(changed) == 1
    is_pref_line = bool(changed) and "preferred_ui" in changed[0][0]
    summary["only_scalar_changed"] = bool(same_line_count and only_one and is_pref_line)

    # Every comment survived the write (targeted edit, not a full re-dump).
    before_comments = [ln for ln in before_lines if ln.lstrip().startswith("#")]
    after_comments = [ln for ln in after_lines if ln.lstrip().startswith("#")]
    summary["comments_preserved"] = before_comments == after_comments and bool(
        before_comments
    )

    # --- reload through the real loader; the new preference is on disk --- #
    settings, _models, _issues = Config.load(work)
    summary["reloaded_preference"] = settings.chat.preferred_ui
    reloaded_ok = settings.chat.preferred_ui == "openwebui"

    # --- fresh chooser pass now skips ASK and goes straight to the remembered UI --- #
    resolution = resolve_chat_choice(
        preferred=settings.chat.preferred_ui,
        cli_override=None,
        model_running=True,
        webui_installed=True,
        webui_ready=True,
    )
    summary["chooser_skips_ask"] = resolution.decision is ChatDecision.OPEN_OPENWEBUI

    # Cleanup the throwaway dir (best effort; never masks the verdict).
    try:
        shutil.rmtree(work, ignore_errors=True)
    except OSError:
        pass

    if (
        summary["only_scalar_changed"]
        and summary["comments_preserved"]
        and reloaded_ok
        and summary["chooser_skips_ask"]
    ):
        summary["outcome"] = "verified"
        summary["reason"] = (
            "write_chat_ui changed only chat.preferred_ui (comments preserved); "
            "reload honored it and the chooser skipped ASK"
        )
    else:
        summary["reason"] = "one or more persistence assertions failed (see fields)"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AC21 chat-UI persistence proof")
    parser.add_argument("--json", action="store_true", help="emit a JSON verdict")
    args = parser.parse_args(argv)

    summary = run_persistence_check()
    if args.json:
        print(json.dumps(summary))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0 if summary["outcome"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
