import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

import pytest
from config import Model
from router import ModelRouter
import responses_tools as adapter


def namespace(name="fixture_reader", function="read_fixture"):
    return {"type": "namespace", "name": name, "description": "Fixed reader",
            "tools": [{"type": "function", "name": function, "description": "Read only",
                       "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                       "strict": True}]}


def test_strict_template_roles_keep_instruction_priority_order_and_images():
    image = {"type": "input_image", "image_url": "data:image/png;base64,fixed"}
    payload = {"instructions": "Original system rules", "input": [
        {"type": "message", "role": "developer", "content": [
            {"type": "input_text", "text": "Keep read-only policy"},
            {"type": "input_text", "text": "Use fixed fixture only"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Environment"}]},
        {"type": "message", "role": "user", "content": [image, {"type": "input_text", "text": "Question"}]}]}
    result = json.loads(adapter.normalize_input(json.dumps(payload).encode()))
    assert result["instructions"] == "Original system rules\n\nDEVELOPER INSTRUCTIONS:\nKeep read-only policy\n\nUse fixed fixture only"
    assert [item["role"] for item in result["input"]] == ["user"]
    assert result["input"][0]["content"] == [
        {"type": "input_text", "text": "Environment"}, {"type": "input_text", "text": "\n\n"},
        image, {"type": "input_text", "text": "Question"}]


def test_codex_message_ids_do_not_block_adjacent_user_merge():
    # Shape captured from a real Codex 0.157 request: every input message has an id.
    payload = {"instructions": "Base", "input": [
        {"type": "message", "id": "d1", "role": "developer", "content": [
            {"type": "input_text", "text": "Permissions"}, {"type": "input_text", "text": "Skills"}]},
        {"type": "message", "id": "u1", "role": "user", "content": [{"type": "input_text", "text": "Environment"}]},
        {"type": "message", "id": "u2", "role": "user", "content": [{"type": "input_text", "text": "Task"}]}]}
    result = json.loads(adapter.normalize_input(json.dumps(payload).encode()))
    assert result["instructions"] == "Base\n\nDEVELOPER INSTRUCTIONS:\nPermissions\n\nSkills"
    assert result["input"] == [{"type": "message", "id": "u1", "role": "user", "content": [
        {"type": "input_text", "text": "Environment"}, {"type": "input_text", "text": "\n\n"},
        {"type": "input_text", "text": "Task"}]}]
    status = dict(payload["input"][2], status="completed")
    kept = json.loads(adapter.normalize_input(json.dumps({"input": payload["input"][1:2] + [status]}).encode()))
    assert [item["id"] for item in kept["input"]] == ["u1", "u2"]


def test_role_normalization_preserves_tool_history_and_unknown_content():
    history = [{"type": "message", "role": "user", "content": "First"},
        {"type": "function_call", "name": "read", "arguments": "{}", "call_id": "c"},
        {"type": "function_call_output", "call_id": "c", "output": "fixture"},
        {"type": "message", "role": "user", "content": "Next"}]
    raw = json.dumps({"input": history}).encode()
    assert adapter.normalize_input(raw) == raw
    unknown = json.dumps({"input": [{"role": "developer", "content": [{"type": "input_image"}]}]}).encode()
    assert adapter.normalize_input(unknown) == unknown


def test_flatten_preserves_schema_history_choice_and_collision_identity():
    payload = {"tools": [namespace(), namespace("another")], "input": [{
        "type": "function_call", "namespace": "fixture_reader", "name": "read_fixture",
        "arguments": "{}", "call_id": "original-id"}], "tool_choice": {
        "type": "function", "namespace": "fixture_reader", "name": "read_fixture"}}
    raw, aliases = adapter.flatten_request(json.dumps(payload).encode())
    rewritten = json.loads(raw)
    assert len(aliases) == 2 and len(set(aliases)) == 2
    alias = rewritten["tools"][0]["name"]
    assert rewritten["tools"][0]["parameters"] == payload["tools"][0]["tools"][0]["parameters"]
    assert rewritten["input"][0] == {"type": "function_call", "name": alias,
                                    "arguments": "{}", "call_id": "original-id"}
    assert rewritten["tool_choice"] == {"type": "function", "name": alias}
    assert len(alias) <= 64
    assert adapter.flatten_request(json.dumps(payload).encode())[1] == aliases


def test_unsupported_custom_is_preserved_and_bad_namespace_rejected():
    tool = namespace()
    custom = {"type": "custom", "name": "apply_patch", "format": {"type": "text"}}
    tool["tools"].append(custom)
    raw, aliases = adapter.flatten_request(json.dumps({"tools": [tool]}).encode())
    assert json.loads(raw)["tools"][1]["tools"] == [custom]
    assert aliases
    with pytest.raises(ValueError):
        adapter.flatten_request(b'{"tools":[{"type":"namespace","name":"n","tools":[{"type":"function"}]}]}')


def test_restore_stream_and_json_preserves_arguments_call_id_and_unknown_names():
    _, aliases = adapter.flatten_request(json.dumps({"tools": [namespace()]}).encode())
    alias = next(iter(aliases))
    item = {"type": "function_call", "name": alias, "arguments": '{"text":"unaltered"}', "call_id": "c"}
    event = {"type": "response.output_item.added", "item": item}
    frames = b"event: response.output_item.added\r\n" + b"data: " + json.dumps(event).encode() + b"\r\n\r\ndata: [DONE]\n\n"
    restored = adapter.restore_sse(frames, aliases)
    parsed = json.loads(restored.split(b"data: ")[1].split(b"\r\n")[0])
    assert parsed["item"] == dict(item, namespace="fixture_reader", name="read_fixture")
    assert restored.endswith(b"data: [DONE]\n\n")
    final = json.loads(adapter.restore_json(json.dumps({"output": [item]}).encode(), aliases))
    assert final["output"][0] == parsed["item"]
    unknown = {"type": "function_call", "name": "unrecognized"}
    assert adapter.restore_payload(unknown, aliases) == unknown


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("lowercase_length", [False, True])
def test_real_router_restores_namespaced_calls_and_recalculates_lengths(stream, lowercase_length):
    observed = []
    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            assert len(self.headers.get_all("Content-Length", [])) == 1
            assert self.headers.get("Transfer-Encoding") is None
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed.append(payload)
            alias = payload["tools"][0]["name"]
            item = {"type": "function_call", "name": alias, "arguments": "{}", "call_id": "keep-id"}
            raw = (b"data: " + json.dumps({"type": "response.output_item.done", "item": item}).encode() + b"\n\n"
                   if stream else json.dumps({"output": [item]}).encode())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            for start in range(0, len(raw), 7):
                self.wfile.write(raw[start:start + 7])
                self.wfile.flush()
        def log_message(self, *args):
            pass
    upstream = HTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    row = Model(id="test", name="Test", description="test", location="test.gguf",
                context_size=8192, gpu_layers=0, status="installed")
    router = ModelRouter(0, lambda: [row], lambda: "test", lambda: upstream.server_address[1],
                         lambda model: (True, "ready"))
    router.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", router._server.server_address[1], timeout=5)
        body = json.dumps({"model": "test", "tools": [namespace()],
                           "input": "Read the fixture", "stream": stream})
        headers = {"Content-Type": "application/json"}
        if lowercase_length:
            headers["content-length"] = str(len(body.encode()))
        conn.request("POST", "/v1/responses", body, headers)
        response = conn.getresponse()
        raw = response.read()
        result = json.loads(raw[6:].strip())["item"] if stream else json.loads(raw)["output"][0]
        assert response.status == 200
        assert result["namespace"] == "fixture_reader" and result["name"] == "read_fixture"
        assert result["call_id"] == "keep-id" and result["arguments"] == "{}"
        assert observed[0]["tools"][0]["type"] == "function"
        if not stream:
            assert int(response.getheader("Content-Length")) == len(raw)
        conn.close()
    finally:
        router.stop()
        upstream.shutdown()
        upstream.server_close()


def test_real_router_switches_models_for_codex_responses_requests():
    """Found live: /v1/responses named another model but was answered by the
    loaded one, because only chat paths switched. Now it switches, 404s an
    unknown model and forwards a model-less body to what is running."""
    seen = {}

    def upstream_for(tag):
        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.setdefault(tag, []).append(payload.get("model"))
                raw = json.dumps({"output": [{"type": "message", "content": [
                    {"type": "output_text", "text": tag}]}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    upstreams = {"a": upstream_for("a"), "b": upstream_for("b")}
    rows = [Model(id=i, name=i.upper(), description="", location=f"{i}.gguf", context_size=16384,
                  gpu_layers=0, status="installed") for i in ("a", "b")]
    state = {"running": "a", "switches": []}

    def switch(model_id):
        state["switches"].append(model_id)
        state["running"] = model_id
        return True, "switched"
    router = ModelRouter(0, lambda: rows, lambda: state["running"],
                         lambda: upstreams[state["running"]].server_address[1], switch)
    router.start()

    def post(model):
        conn = http.client.HTTPConnection("127.0.0.1", router._server.server_address[1], timeout=5)
        payload = {"input": "hi"} if model is None else {"model": model, "input": "hi"}
        conn.request("POST", "/v1/responses", json.dumps(payload), {"Content-Type": "application/json"})
        response = conn.getresponse()
        result = response.status, json.loads(response.read())
        conn.close()
        return result
    try:
        assert post("b")[1]["output"][0]["content"][0]["text"] == "b"
        assert post("b")[0] == 200 and state["switches"] == ["b"]
        assert post(None)[1]["output"][0]["content"][0]["text"] == "b"
        status, error = post("not-registered")
        assert status == 404 and error["error"]["type"] == "model_not_found"
        assert post("A")[1]["output"][0]["content"][0]["text"] == "a"
        assert state["switches"] == ["b", "a"]
        assert "a" not in seen or seen["a"] == ["A"]
    finally:
        router.stop()
        for server in upstreams.values():
            server.shutdown()
            server.server_close()


def test_nested_local_refs_are_inlined_like_the_real_failing_tool():
    # Shape of the real tool that llama.cpp rejected: "$defs not found".
    params = {"type": "object", "properties": {"request": {
        "type": "object",
        "$defs": {"MinMaxInt": {"type": "object", "properties": {"min": {"type": "integer"}}}},
        "properties": {"propertyFiltersRequest": {"type": "object", "properties": {
            "bedrooms": {"$ref": "#/$defs/MinMaxInt", "description": "bedroom range"},
            "definitions": {"type": "string"}}}}}}}
    body = json.dumps({"tools": [{"type": "function", "name": "search", "parameters": params}]}).encode()
    tool = json.loads(adapter.inline_schema_refs(body))["tools"][0]
    request = tool["parameters"]["properties"]["request"]
    assert "$defs" not in request and "$ref" not in json.dumps(tool)
    filters = request["properties"]["propertyFiltersRequest"]["properties"]
    assert filters["bedrooms"] == {"type": "object", "properties": {"min": {"type": "integer"}},
                                   "description": "bedroom range"}
    assert filters["definitions"] == {"type": "string"}


def test_cyclic_and_unknown_refs_become_open_schemas_and_namespaces_are_covered():
    params = {"type": "object", "definitions": {"Node": {"type": "object", "properties": {
        "child": {"$ref": "#/definitions/Node"}}}},
        "properties": {"root": {"$ref": "#/definitions/Node"}, "other": {"$ref": "#/$defs/Missing"}}}
    body = json.dumps({"tools": [{"type": "namespace", "name": "ns", "tools": [
        {"type": "function", "name": "f", "parameters": params}]}]}).encode()
    child = json.loads(adapter.inline_schema_refs(body))["tools"][0]["tools"][0]["parameters"]
    assert child["properties"]["root"]["properties"]["child"] == {}
    assert child["properties"]["other"] == {}
    assert "definitions" not in child


def test_bodies_without_refs_are_untouched():
    body = json.dumps({"tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}]}).encode()
    assert adapter.inline_schema_refs(body) is body
    remote = json.dumps({"tools": [{"type": "function", "name": "f", "parameters": {
        "type": "object", "properties": {"x": {"$ref": "https://example.com/s.json"}}}}]}).encode()
    assert adapter.inline_schema_refs(remote) == remote
