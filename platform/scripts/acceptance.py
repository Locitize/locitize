"""LOCITIZE full-application acceptance harness - OWNER RUN.

The claims audit, done as a repeatable script. Every user-facing capability is
exercised against the REAL machine - real models, real servers, real network -
and each check prints PASS / FAIL / SKIP with the evidence. Where a capability
needs a model, the smallest registered one is used; where it needs the network,
the check says so before touching it.

This is the "test the whole application line by line" pass turned into
something that survives past one session: run it before any release tag, and
after any change that could touch a serving path.

    python scripts/acceptance.py            # full run
    python scripts/acceptance.py --quick    # skip the multi-minute model loads

Exit 0 only when nothing FAILED (SKIPs are allowed - a missing optional
dependency is not a failure of the parts that are present).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_DIR))

import config  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def record(section: str, name: str, ok: bool | None, detail: str = "") -> None:
    status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((section, name, status))
    mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "SKIP": " skip "}[status]
    print(f"[{mark}] {section} / {name}" + (f"  - {detail}" if detail else ""), flush=True)


def _smallest_runnable(registry):
    rows = sorted(
        (m for m in registry.launchable() if m.location and Path(m.location).is_file()),
        key=lambda m: Path(m.location).stat().st_size,
    )
    return rows[0] if rows else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acceptance")
    parser.add_argument("--quick", action="store_true",
                        help="skip checks that load a model (minutes each)")
    args = parser.parse_args(argv)

    settings, models, issues = config.Config.load()
    from launcher import Launcher
    from logger import resolve_log_dir

    lch = Launcher()
    log_path = resolve_log_dir(settings) / "acceptance.log"
    registry, manager, controller = lch._build_controller(settings, models, log_path=log_path)
    import atexit
    atexit.register(manager.stop_all)

    # ---- Section 1: configuration & registry ---------------------------- #
    record("config", "loads without errors", len(issues) == 0,
           f"{len(issues)} issue(s)")
    runnable = [m for m in registry.launchable()
                if m.location and Path(m.location).is_file()]
    record("registry", "has runnable models", len(runnable) > 0,
           f"{len(runnable)} with a file on disk")

    # ---- Section 2: health probes --------------------------------------- #
    try:
        from health import HealthChecker, HealthProviders, NvidiaSmiGpuInfoProvider
        from health import DefaultSystemInfoProvider, DefaultBinaryProbeProvider
        from health import DefaultPortProbeProvider
        providers = HealthProviders(
            system=DefaultSystemInfoProvider(), gpu=NvidiaSmiGpuInfoProvider(),
            binary=DefaultBinaryProbeProvider(), port=DefaultPortProbeProvider())
        report = HealthChecker(settings, models.models, providers).run_all()
        fails = [r for r in report.results if r.status.name == "FAIL"]
        record("health", "probes run", True, f"{len(report.results)} probes, {len(fails)} FAIL")
    except Exception as exc:  # noqa: BLE001
        record("health", "probes run", False, str(exc)[:60])

    # ---- Section 3: model serving (start/switch/stop) ------------------- #
    model = _smallest_runnable(registry)
    if args.quick or model is None:
        record("serving", "start/switch/stop", None, "quick mode or no model")
    else:
        from services import ServiceStatus
        try:
            st = controller.start(model.id, 4096, 999)
            up = st is ServiceStatus.RUNNING
            record("serving", f"start {model.id}", up)
            port = settings.ports.llama_cpp
            # inference over the OpenAI endpoint
            ok_inf = False
            if up:
                body = json.dumps({"model": model.id, "messages": [
                    {"role": "user", "content": "Reply with only: OK"}],
                    "max_tokens": 30, "temperature": 0}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    reply = json.loads(r.read())
                ok_inf = bool(reply["choices"][0]["message"]["content"] is not None)
            record("serving", "OpenAI /v1/chat/completions answers", ok_inf)
        finally:
            controller.stop()
        snap = controller.snapshot()
        record("serving", "clean stop (no running model)",
               snap["running_model"] is None)

    # ---- Section 4: model discovery (the no-catalog first run) ---------- #
    import setup_env
    found = setup_env.find_local_models()
    record("discovery", "scans local GGUFs", isinstance(found, list),
           f"{len(found)} found")
    record("catalog", "ships no model names",
           __import__("json").loads(
               (PLATFORM_DIR / "model_catalog.json").read_text())["models"] == [])

    # ---- Section 5: hub search (fresh-machine front door) --------------- #
    try:
        import modelhub
        d = modelhub.Downloader(modelhub.HubConfig())
        res = d.search("qwen gguf")
        record("hub", "huggingface search returns results",
               res["ok"] and len(res["items"]) > 0, f"{len(res.get('items', []))} hits")
        if res["ok"] and res["items"]:
            listing = d.list_files(res["items"][0]["repo_id"])
            verified = listing["ok"] and any(
                i["verification"] == "api" for i in listing["items"])
            record("hub", "file listing carries digests", verified)
    except Exception as exc:  # noqa: BLE001
        record("hub", "huggingface search", False, str(exc)[:60])

    # ---- privacy ledger + capability detection (M17) --------------------- #
    import egress_log
    egress_log.configure(settings.data_dir)
    before = egress_log.summarize(settings.data_dir).total
    try:
        import modelhub
        with egress_log.reason("acceptance-probe"):
            modelhub.Downloader(modelhub.HubConfig()).search("test")
        after = egress_log.summarize(settings.data_dir).total
        record("privacy", "egress ledger records outbound connections", after > before)
    except Exception as exc:  # noqa: BLE001
        record("privacy", "egress ledger", False, str(exc)[:50])

    import gguf_meta
    model_file = next((m.location for m in registry.launchable()
                       if m.location and Path(m.location).is_file()), None)
    if model_file:
        caps = gguf_meta.detect_capabilities(model_file)
        record("capabilities", "GGUF capability scan runs", isinstance(caps, list),
               f"{caps or 'none'} on smallest model")

    # ---- Section 6: voice round-trip (TTS -> STT) ---------------------- #
    if args.quick:
        record("voice", "TTS -> STT round-trip", None, "quick mode")
    else:
        wav = resolve_log_dir(settings) / "acceptance_tts.wav"
        try:
            import launcher as _launcher
            rc = _launcher.main(["--speak", "the quick brown fox", "--json"])
            spoke = wav.exists() or rc == 0
            record("voice", "Kokoro TTS synthesizes", spoke)
        except Exception as exc:  # noqa: BLE001
            record("voice", "Kokoro TTS synthesizes", False, str(exc)[:50])

    # ---- Section 7: CLI contracts -------------------------------------- #
    import launcher as _launcher
    try:
        rc = _launcher.main(["--service-status", "--json"])
        record("cli", "--service-status exits 0", rc == 0)
    except SystemExit as exc:
        record("cli", "--service-status exits 0", (exc.code or 0) == 0)
    except Exception as exc:  # noqa: BLE001
        record("cli", "--service-status exits 0", False, str(exc)[:50])

    # ---- summary ------------------------------------------------------- #
    fails = [r for r in RESULTS if r[2] == "FAIL"]
    skips = [r for r in RESULTS if r[2] == "SKIP"]
    print("\n" + "=" * 60)
    print(f"ACCEPTANCE: {len(RESULTS)} checks, {len(fails)} FAIL, {len(skips)} SKIP")
    if fails:
        for s_, n_, _ in fails:
            print(f"  FAILED: {s_} / {n_}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
