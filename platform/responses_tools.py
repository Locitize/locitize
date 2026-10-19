"""Request-local function namespace compatibility for llama.cpp Responses.

Only routing identifiers change. Tool arguments, schemas, call IDs and client
permission decisions are preserved. Custom tools are not converted.
"""
from __future__ import annotations

import hashlib
import json
import re


def normalize_input(body: bytes) -> bytes:
    """Fit text-only leading instructions and adjacent users to strict templates.

    Native llama.cpp templates commonly accept one system message followed by
    alternating user/assistant messages. Keep tool history and image content.
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict) or not isinstance(payload.get("input"), list):
        return body
    existing = payload.get("instructions", "")
    if existing is not None and not isinstance(existing, str):
        return body
    inputs = list(payload["input"])
    instructions = []
    while inputs:
        item = inputs[0]
        if not isinstance(item, dict) or item.get("role") not in ("system", "developer"):
            break
        content = item.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list) and all(isinstance(part, dict) and
                part.get("type") in ("input_text", "text") and
                isinstance(part.get("text"), str) for part in content):
            text = "\n\n".join(part["text"] for part in content)
        else:
            break
        instructions.append(item["role"].upper() + " INSTRUCTIONS:\n" + text)
        inputs.pop(0)
    # Codex gives each input message an id; the local engine never echoes
    # input ids back, so the merged message keeps the first one.
    mergeable = {"type", "role", "content", "id"}
    merged = []
    for item in inputs:
        previous = merged[-1] if merged else None
        if (isinstance(item, dict) and isinstance(previous, dict)
                and item.get("role") == previous.get("role") == "user"
                and item.get("type", "message") == previous.get("type", "message") == "message"
                and set(item).issubset(mergeable)
                and set(previous).issubset(mergeable)
                and isinstance(item.get("content"), (str, list))
                and isinstance(previous.get("content"), (str, list))):
            left = previous["content"]
            right = item["content"]
            left = [{"type": "input_text", "text": left}] if isinstance(left, str) else left
            right = [{"type": "input_text", "text": right}] if isinstance(right, str) else right
            merged[-1] = dict(previous, content=left + [{"type": "input_text", "text": "\n\n"}] + right)
        else:
            merged.append(item)
    if not instructions and merged == payload["input"]:
        return body
    if instructions:
        payload["instructions"] = "\n\n".join(([existing] if existing else []) + instructions)
    payload["input"] = merged
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _alias(namespace: str, name: str) -> str:
    identity = json.dumps([namespace, name], ensure_ascii=True).encode("ascii")
    suffix = hashlib.sha256(identity).hexdigest()[:16]
    readable = re.sub(r"[^A-Za-z0-9_-]", "_", namespace + "_" + name)[:42]
    return "lt_" + readable + "_" + suffix


def flatten_request(body: bytes) -> tuple[bytes, dict[str, tuple[str, str]]]:
    """Flatten namespace function children; retain unsupported tool types."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body, {}
    if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
        return body, {}
    aliases = {}
    flat = []
    existing = {tool.get("name") for tool in payload["tools"] if isinstance(tool, dict)
                and tool.get("type") != "namespace" and isinstance(tool.get("name"), str)}
    for tool in payload["tools"]:
        if not isinstance(tool, dict) or tool.get("type") != "namespace":
            flat.append(tool)
            continue
        namespace = tool.get("name")
        children = tool.get("tools")
        if not isinstance(namespace, str) or not namespace or not isinstance(children, list):
            raise ValueError("Invalid Responses function namespace")
        unsupported = []
        for child in children:
            if not isinstance(child, dict) or child.get("type") != "function":
                unsupported.append(child)
                continue
            name = child.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Invalid Responses namespace function name")
            alias = _alias(namespace, name)
            if alias in existing or alias in aliases:
                raise ValueError("Responses tool identity collision")
            aliases[alias] = (namespace, name)
            translated = dict(child, name=alias)
            description = child.get("description", "")
            translated["description"] = f"{namespace}.{name}: {description}"
            flat.append(translated)
        if unsupported:
            flat.append(dict(tool, tools=unsupported))
    if not aliases:
        return body, {}
    payload["tools"] = flat
    reverse = {identity: alias for alias, identity in aliases.items()}
    for item in payload.get("input", []) if isinstance(payload.get("input"), list) else []:
        if isinstance(item, dict) and item.get("type") in ("function_call", "function_call_output"):
            alias = reverse.get((item.get("namespace"), item.get("name")))
            if alias:
                item["name"] = alias
                item.pop("namespace", None)
    choice = payload.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        alias = reverse.get((choice.get("namespace"), choice.get("name")))
        if alias:
            choice["name"] = alias
            choice.pop("namespace", None)
    return json.dumps(payload, ensure_ascii=True).encode("utf-8"), aliases


_LOCAL_REF = re.compile(r"#/(\$defs|definitions)/([^/]+)")
_SCHEMA_MAPS = ("properties", "patternProperties")
_DEF_KEYS = ("$defs", "definitions")


def _inline_refs(schema):
    """Inline local $defs/definitions references within one tool schema.

    llama.cpp resolves `#/...` against the tool's parameters root, so a
    definition nested lower in the schema is "not found" and the whole request
    fails. Inlining keeps the schema's meaning; a cyclic or unknown local
    reference becomes an open schema (looser, never stricter). Property names
    that happen to be "definitions" stay properties.
    """
    found = {}

    def collect(node):
        if isinstance(node, list):
            for value in node:
                collect(value)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key in _DEF_KEYS and isinstance(value, dict):
                    for name, definition in value.items():
                        found.setdefault((key, name), definition)
                        collect(definition)
                elif key in _SCHEMA_MAPS and isinstance(value, dict):
                    for sub in value.values():
                        collect(sub)
                else:
                    collect(value)
    collect(schema)

    def resolve(node, active):
        if isinstance(node, list):
            return [resolve(value, active) for value in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        match = _LOCAL_REF.fullmatch(ref) if isinstance(ref, str) else None
        if match:
            key = (match.group(1), match.group(2).replace("~1", "/").replace("~0", "~"))
            siblings = {k: resolve(v, active) for k, v in node.items() if k != "$ref"}
            if key not in found or key in active or len(active) >= 32:
                return siblings
            target = resolve(found[key], active | {key})
            return dict(target, **siblings) if isinstance(target, dict) else target
        result = {}
        for key, value in node.items():
            if key in _DEF_KEYS and isinstance(value, dict):
                continue
            if key in _SCHEMA_MAPS and isinstance(value, dict):
                result[key] = {name: resolve(sub, active) for name, sub in value.items()}
            else:
                result[key] = resolve(value, active)
        return result
    return resolve(schema, frozenset())


def inline_schema_refs(body: bytes) -> bytes:
    """Inline local schema references in function tools; other bodies unchanged."""
    if b'"$ref"' not in body:
        return body
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
        return body
    changed = False
    functions = []
    for tool in payload["tools"]:
        if isinstance(tool, dict) and tool.get("type") == "function":
            functions.append(tool)
        elif isinstance(tool, dict) and tool.get("type") == "namespace" and isinstance(tool.get("tools"), list):
            functions += [child for child in tool["tools"] if isinstance(child, dict)
                          and child.get("type") == "function"]
    for tool in functions:
        parameters = tool.get("parameters")
        if isinstance(parameters, dict):
            inlined = _inline_refs(parameters)
            if inlined != parameters:
                tool["parameters"] = inlined
                changed = True
    return json.dumps(payload, ensure_ascii=True).encode("utf-8") if changed else body


def restore_payload(value, aliases: dict[str, tuple[str, str]]):
    """Restore function identities in items and Responses event envelopes."""
    if isinstance(value, list):
        return [restore_payload(item, aliases) for item in value]
    if not isinstance(value, dict):
        return value
    result = dict(value)
    kind = result.get("type", "")
    if kind == "function_call" or (isinstance(kind, str) and kind.startswith("response.function_call")):
        identity = aliases.get(result.get("name")) if isinstance(result.get("name"), str) else None
        if identity:
            result["namespace"], result["name"] = identity
    for key in ("item", "response", "output"):
        if key in result:
            result[key] = restore_payload(result[key], aliases)
    return result


def restore_json(body: bytes, aliases: dict[str, tuple[str, str]]) -> bytes:
    if not aliases:
        return body
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    return json.dumps(restore_payload(payload, aliases), ensure_ascii=True).encode("utf-8")


def restore_sse(frames: bytes, aliases: dict[str, tuple[str, str]]) -> bytes:
    """Rewrite complete SSE frames; router owns chunk-boundary buffering."""
    if not aliases:
        return frames
    lines = []
    for line in frames.splitlines(keepends=True):
        if line.startswith(b"data:"):
            ending = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""
            data = line[5:].strip()
            if data != b"[DONE]":
                rewritten = restore_json(data, aliases)
                if rewritten != data:
                    line = b"data: " + rewritten + ending
        lines.append(line)
    return b"".join(lines)
