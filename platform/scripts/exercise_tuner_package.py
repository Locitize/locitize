"""Exercise a packaged LoCiTiZe tuner through its real stdio entry point."""
from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit


def response_payload(response):
    return json.loads(next(block.text for block in response.content if hasattr(block, "text")))


async def call(session, name, arguments=None):
    payload = response_payload(await session.call_tool(name, arguments or {}))
    if not payload.get("ok"):
        raise RuntimeError(f"{name} failed: {payload.get('error')}")
    return payload["result"]


async def wait_terminal(session, job_id, timeout, cancel=False):
    started = time.monotonic()
    cancellation_sent = False
    snapshots = []
    while time.monotonic() - started < timeout:
        status = await call(session, "tuning_status", {"job_id": job_id})
        snapshots.append({key: status.get(key) for key in
                          ("state", "phase", "trial_count", "trial_budget", "current_request",
                           "best_observed", "cleanup_confirmed", "error")})
        if cancel and not cancellation_sent and status.get("state") not in {
                "completed", "failed", "cancelled", "target_not_met"}:
            await call(session, "cancel_tuning", {"job_id": job_id})
            cancellation_sent = True
        if status.get("state") in {"completed", "failed", "cancelled", "target_not_met"}:
            return status, snapshots
        await asyncio.sleep(1 if cancel else 3)
    raise TimeoutError("tuning job did not reach a terminal state")


def post_json(base_url, payload, timeout=180):
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    path = parsed.path.rstrip("/") + "/responses"
    connection.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = {"raw": raw.decode("utf-8", errors="replace")}
    return {"status": response.status, "body": body}


async def run(arguments):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    environment = dict(os.environ)
    parameters = StdioServerParameters(command="python", args=[str(arguments.package / "server.py")],
                                       cwd=str(arguments.package), env=environment)
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if arguments.action == "inspect":
                return {"hardware": await call(session, "assess_computer"),
                        "models": await call(session, "list_local_models")}
            if arguments.action in {"cancel", "tune"}:
                job = await call(session, "start_tuning",
                                 {"model_id": arguments.model_id, "objective": "balanced"})
                status, snapshots = await wait_terminal(session, job["job_id"], arguments.timeout,
                                                        cancel=arguments.action == "cancel")
                return {"job": job, "terminal": status, "snapshots": snapshots}
            if arguments.action == "search":
                return await call(session, "search_models", {"query": "coding GGUF", "limit": 3})
            if arguments.action == "fit":
                found = await call(session, "search_models", {"query": "coding GGUF", "limit": 3})
                if not found.get("items"):
                    return {"search": found, "files": None, "recommendation": None}
                files = await call(session, "list_model_files",
                                   {"repo_id": found["items"][0]["repo_id"]})
                selections = [row["selection_id"] for row in files.get("candidates", [])[:3]]
                recommendation = (await call(session, "recommend_model_candidates",
                    {"selection_ids": selections, "use_case": "coding", "prefer_vision": False})
                    if selections else None)
                return {"search": found, "files": files, "recommendation": recommendation}
            if arguments.action == "serve":
                started = await call(session, "start_local_model", {"model_id": arguments.model_id})
                try:
                    text = post_json(started["codex_api_base_url"], {
                        "model": arguments.model_id,
                        "input": "Reply with exactly OK-17 and no other text.",
                        "max_output_tokens": 256,
                    })
                    tool = post_json(started["codex_api_base_url"], {
                        "model": arguments.model_id,
                        "input": "Use the fixture reader tool now.",
                        "tools": [{"type": "namespace", "name": "fixture_reader",
                                   "description": "Read a fixed test fixture",
                                   "tools": [{"type": "function", "name": "read_fixture",
                                              "description": "Read the fixture",
                                              "parameters": {"type": "object", "properties": {},
                                                             "additionalProperties": False},
                                              "strict": True}]}],
                        "tool_choice": {"type": "function", "namespace": "fixture_reader",
                                        "name": "read_fixture"},
                        "max_output_tokens": 512,
                    })
                    return {"started": started, "text_response": text, "tool_response": tool}
                finally:
                    stopped = await call(session, "stop_local_model")
                    if not stopped.get("cleanup_confirmed"):
                        raise RuntimeError("owned model cleanup was not confirmed")
    raise ValueError("unknown action")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inspect", "cancel", "tune", "search", "fit", "serve"))
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen3-0-6b-q8_0")
    parser.add_argument("--timeout", type=int, default=1200)
    arguments = parser.parse_args()
    print(json.dumps(asyncio.run(run(arguments)), indent=2))


if __name__ == "__main__":
    main()
