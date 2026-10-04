"""Model-router tests (owner request 2026-09-02, keyword 'model_router').

Headless and deterministic: the routing DECISIONS are pure functions taking no
self, so listing, name resolution and body parsing are asserted with no socket,
no GPU and no subprocess. The switch-serialisation and lifecycle tests use a
fake switch_fn that records instead of loading, so nothing here ever touches a
model file.

What is deliberately NOT tested here: the forwarding hop itself. That is the
same http.client relay proxy.py already proves (keyword 'proxy_target'), and
re-testing it would assert the stdlib rather than LOCITIZE's own decisions.
"""

from __future__ import annotations

import json
import threading

import pytest
from config import Model, ModelRegistryData, Settings
from models import ModelRegistry
from router import (
    ModelRouter,
    ask_needs_tools,
    build_laya_steer_completion,
    build_models_payload,
    build_upstream_target,
    is_tool_broken_model,
    messages_have_tool_exchange,
    pick_tool_capable_fallback,
    request_has_tools,
    requested_model_from_body,
    resolve_requested_model,
    rewrite_body_model,
    should_pre_steer_laya,
)


def _rows(*pairs) -> list[Model]:
    """Registry rows from (id, display name) pairs."""
    return [
        Model(
            id=model_id,
            name=name,
            description="d",
            location=f"/locitize-test/models/{model_id}.gguf",
            context_size=8192,
            gpu_layers=-1,
            status="installed",
        )
        for model_id, name in pairs
    ]


# --------------------------------------------------------------------------- #
# /v1/models - answered from the registry, never from the running server
# --------------------------------------------------------------------------- #


def test_listing_advertises_every_registry_row_not_just_the_running_one():
    """The whole point: llama-server advertises 1, the registry has many."""
    payload = build_models_payload(
        _rows(("qwen3-14b-q5_0", "Qwen3 14B"), ("gpt-oss-20b-f16", "GPT-OSS 20B"))
    )
    assert payload["object"] == "list"
    assert [row["id"] for row in payload["data"]] == [
        "qwen3-14b-q5_0",
        "gpt-oss-20b-f16",
    ]


def test_listing_uses_the_registry_id_and_carries_the_human_name_along():
    """`id` must be the registry id - the thing every other surface addresses a
    model by, and unique by construction. The display name rides as `name`."""
    row = build_models_payload(_rows(("qwen3-14b-q5_0", "Qwen3 14B")))["data"][0]
    assert row["id"] == "qwen3-14b-q5_0"
    assert row["name"] == "Qwen3 14B"
    assert row["owned_by"] == "locitize"


def test_listing_does_not_invent_a_creation_date():
    """LOCITIZE does not know when a model was made; 0 beats a fabricated date."""
    row = build_models_payload(_rows(("m", "M")))["data"][0]
    assert row["created"] == 0


def test_listing_falls_back_to_the_id_when_a_row_has_no_name():
    row = build_models_payload(_rows(("qwen3-14b-q5_0", "")))["data"][0]
    assert row["name"] == "qwen3-14b-q5_0"


def test_listing_an_empty_registry_is_a_valid_empty_list():
    assert build_models_payload([]) == {"object": "list", "data": []}


# --------------------------------------------------------------------------- #
# resolving what the client asked for
# --------------------------------------------------------------------------- #


def test_a_request_naming_the_registry_id_resolves():
    rows = _rows(("qwen3-14b-q5_0", "Qwen3 14B"))
    assert resolve_requested_model("qwen3-14b-q5_0", rows) == "qwen3-14b-q5_0"


def test_a_request_naming_the_display_name_also_resolves():
    """llama-server is started with --alias <name>, so a client that discovered
    the model from the RUNNING server sends the name back. Refusing it would
    break the exact flow this router exists to smooth."""
    rows = _rows(("kimi-vl-a3b-thinking-2506-q4_k_m", "Kimi-VL-A3B-Thinking-2506"))
    assert (
        resolve_requested_model("Kimi-VL-A3B-Thinking-2506", rows)
        == "kimi-vl-a3b-thinking-2506-q4_k_m"
    )


def test_resolution_prefers_an_exact_id_over_another_rows_name():
    """A row whose NAME collides with another row's ID must not win."""
    rows = _rows(("alpha", "beta"), ("beta", "Beta Model"))
    assert resolve_requested_model("beta", rows) == "beta"


def test_resolution_is_case_insensitive_as_a_last_resort():
    rows = _rows(("qwen3-14b-q5_0", "Qwen3 14B"))
    assert resolve_requested_model("QWEN3-14B-Q5_0", rows) == "qwen3-14b-q5_0"
    assert resolve_requested_model("qwen3 14b", rows) == "qwen3-14b-q5_0"


def test_an_unknown_model_resolves_to_nothing():
    """Never substitute. Answering with a different model than the one asked
    for is the failure this returns None to prevent."""
    assert resolve_requested_model("gpt-9-turbo", _rows(("m", "M"))) is None


def test_an_omitted_model_is_not_a_match():
    """OpenAI clients may omit `model`; that means 'whatever is running', which
    the handler treats as no-switch rather than as an error."""
    assert resolve_requested_model("", _rows(("m", "M"))) is None
    assert resolve_requested_model("   ", _rows(("m", "M"))) is None


# --------------------------------------------------------------------------- #
# reading the request body
# --------------------------------------------------------------------------- #


def test_the_model_field_is_read_from_a_json_body():
    body = json.dumps({"model": "qwen3-14b-q5_0", "messages": []}).encode()
    assert requested_model_from_body(body) == "qwen3-14b-q5_0"


def test_body_parsing_is_total_and_never_raises():
    """A body this cannot parse means 'no switch requested', not a refusal -
    refusing would break any endpoint whose payload shape it does not know."""
    assert requested_model_from_body(b"") == ""
    assert requested_model_from_body(b"not json at all") == ""
    assert requested_model_from_body(b'["a", "list"]') == ""
    assert requested_model_from_body(b'{"no_model_key": 1}') == ""
    assert requested_model_from_body(b'{"model": 42}') == ""
    assert requested_model_from_body(b"\xff\xfe invalid utf8") == ""


# --------------------------------------------------------------------------- #
# upstream target
# --------------------------------------------------------------------------- #


def test_the_upstream_is_always_loopback():
    """Hardcoded 127.0.0.1: no config value can steer the forwarding hop off it."""
    assert build_upstream_target(8080) == ("127.0.0.1", 8080)


def test_no_running_model_means_no_upstream():
    assert build_upstream_target(None) is None
    assert build_upstream_target(0) is None


# --------------------------------------------------------------------------- #
# switching
# --------------------------------------------------------------------------- #


def _router(running=None, switch_fn=None, rows=None) -> ModelRouter:
    state = {"running": running}
    return ModelRouter(
        port=0,
        registry_fn=lambda: rows if rows is not None else _rows(("m", "M")),
        running_model_id_fn=lambda: state["running"],
        port_provider=lambda: 8080,
        switch_fn=switch_fn or (lambda model_id: (True, "RUNNING")),
    )


def test_switching_to_the_already_running_model_does_no_work():
    """Re-checked UNDER the lock: two requests for the same model arriving
    together would otherwise both pass the caller's check and the second would
    pointlessly reload what the first just started."""
    calls: list[str] = []
    router = _router(
        running="qwen3-14b-q5_0",
        switch_fn=lambda model_id: (calls.append(model_id), (True, "RUNNING"))[1],
    )
    ok, reason = router.switch("qwen3-14b-q5_0")
    assert ok and reason == "already running"
    assert calls == []


def test_a_failing_switch_is_reported_not_raised():
    router = _router(switch_fn=lambda model_id: (False, "STOPPED_ERROR"))
    assert router.switch("qwen3-14b-q5_0") == (False, "STOPPED_ERROR")


def test_a_raising_switch_is_caught_at_the_boundary():
    """A ValueError from the controller (unknown id, missing location) must come
    back as an HTTP-able failure, never escape into the serving thread."""

    def boom(model_id):
        raise ValueError("no location set")

    ok, reason = _router(switch_fn=boom).switch("qwen3-14b-q5_0")
    assert not ok and "no location set" in reason


def test_concurrent_switches_are_serialised():
    """Two tabs asking for two different models must not race a pair of
    multi-gigabyte loads onto one card."""
    overlaps: list[int] = []
    active = {"n": 0}
    barrier = threading.Barrier(2, timeout=5)

    def slow_switch(model_id):
        active["n"] += 1
        overlaps.append(active["n"])
        # Give the other thread every chance to enter, if the lock is missing.
        try:
            barrier.wait(timeout=0.2)
        except threading.BrokenBarrierError:
            pass
        active["n"] -= 1
        return True, "RUNNING"

    router = _router(switch_fn=slow_switch)
    threads = [
        threading.Thread(target=router.switch, args=(f"model-{i}",)) for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert max(overlaps) == 1  # never two switches in flight at once


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #


def test_base_url_is_loopback_and_openai_shaped():
    assert _router().base_url().startswith("http://127.0.0.1:")
    assert _router().base_url().endswith("/v1")


def test_the_router_reads_the_registry_fresh_on_every_request():
    """A registry reload must be picked up without restarting the router."""
    rows = _rows(("m", "M"))
    router = ModelRouter(
        port=0,
        registry_fn=lambda: list(rows),
        running_model_id_fn=lambda: None,
        port_provider=lambda: None,
        switch_fn=lambda model_id: (True, "RUNNING"),
    )
    assert len(router.models()) == 1
    rows.extend(_rows(("n", "N")))
    assert len(router.models()) == 2


def test_open_webui_is_pointed_at_the_router_only_when_it_is_enabled():
    """The direct llama-server URL stays the default: the router lives inside the
    process that owns the ModelController, so a chat UI pointed at it while no
    LOCITIZE session runs would find nothing listening."""
    import webui

    settings = Settings()
    assert settings.router.enabled is False
    assert webui.backend_base_url(settings).endswith(
        f":{settings.ports.llama_cpp}/v1"
    )
    settings.router.enabled = True
    assert webui.backend_base_url(settings).endswith(f":{settings.ports.router}/v1")


def test_an_explicit_backend_base_url_still_wins_over_the_router():
    """That value is the owner pointing at something deliberately."""
    import webui

    settings = Settings()
    settings.router.enabled = True
    settings.openwebui.backend_base_url = "http://127.0.0.1:9999/v1"
    assert webui.backend_base_url(settings) == "http://127.0.0.1:9999/v1"


def test_the_router_port_is_inside_the_reserved_loopback_range():
    settings = Settings()
    assert settings.ports.range_start <= settings.ports.router <= settings.ports.range_end
    # And it does not collide with another reserved service.
    taken = [
        settings.ports.llama_cpp,
        settings.ports.whisper,
        settings.ports.kokoro,
        settings.ports.vision,
        settings.ports.scheduler,
        settings.ports.openwebui,
    ]
    assert settings.ports.router not in taken


def test_the_router_lists_a_real_registry_shape():
    """End to end over the pure path: a ModelRegistry's launchable() rows go
    through build_models_payload unchanged."""
    registry = ModelRegistry(
        ModelRegistryData(version=1, models=_rows(("a", "A"), ("b", "B"))), Settings()
    )
    payload = build_models_payload(registry.launchable())
    assert [r["id"] for r in payload["data"]] == ["a", "b"]


@pytest.mark.parametrize("path", ["/v1/models", "/models"])
def test_both_model_list_paths_are_recognised(path):
    """Open WebUI builds differ on whether they prefix /v1."""
    from router import MODEL_LIST_PATHS

    assert path in MODEL_LIST_PATHS


def test_the_router_block_actually_parses_out_of_settings_yaml(tmp_path):
    """Defect caught 2026-09-02 during wiring: RouterConfig existed, Settings had
    the field, and the YAML said `router: {enabled: true}` - but the loader never
    read the block, so the setting silently stayed False and Open WebUI kept
    getting the direct llama-server URL. A dataclass nobody parses is not a
    setting."""
    from config import Config

    (tmp_path / "settings.yaml").write_text(
        "version: 2\nrouter:\n  enabled: true\n", encoding="utf-8"
    )
    (tmp_path / "models.yaml").write_text("version: 1\nmodels: []\n", encoding="utf-8")
    settings, _models, issues = Config.load(tmp_path)
    assert settings.router.enabled is True
    assert not [i for i in issues if i.level == "ERROR"]


def test_the_router_block_is_reported_in_the_settings_dump():
    """`settings` in the launcher menu dumps every block; a missing one means the
    owner cannot see what is configured."""
    from config import settings_to_dict

    dump = settings_to_dict(Settings(), redact=True)
    assert "router" in dump
    assert dump["router"]["enabled"] is False


def test_the_router_binds_the_session_controller_not_any_built_one():
    """Hazard caught 2026-09-02 before shipping: binding the router's controller
    inside _build_controller looked convenient, but that method is also called by
    short-lived paths (benchmark, vision, --smoke-start). The router would have
    drifted onto a throwaway controller that knows nothing about the running
    server, and the next picker change would have started a SECOND llama-server
    beside the first - the exact one-model-at-a-time violation M8.2 forbids.
    """
    import inspect

    import launcher

    build_src = inspect.getsource(launcher.Launcher._build_controller)
    assert "_session_controller" not in build_src

    ensure_src = inspect.getsource(launcher.Launcher._ensure_router)
    assert "_session_controller" in ensure_src
    assert "_session_registry" in ensure_src


def test_the_router_is_started_before_open_webui():
    """Open WebUI bakes backend_base_url into its CHILD ENV at start, so a router
    started afterwards would be pointed at by nothing until the next restart."""
    import inspect

    import launcher

    src = inspect.getsource(launcher.Launcher._start_openwebui)
    # Compared against the CALL, not the "from webui import" line at the top of
    # the method - that import always sorts first and would pass vacuously.
    assert src.index("_ensure_router") < src.index("build_openwebui_spec(settings")


# --------------------------------------------------------------------------- #
# Model handoff note (owner-observed defect 2026-09-03).
#
# Switching Qwen2.5-VL -> gpt-oss-20b inside ONE Open WebUI conversation made
# gpt-oss answer "I am Qwen, a large language model created by Alibaba Cloud".
# The router was switching correctly; the transcript was the cause. Reproduced
# with the same model and server, one variable changed:
#   with Qwen's answer in the history -> "I'm Qwen, ... by Alibaba Cloud"
#   history removed                   -> "I'm ChatGPT, ... created by OpenAI"
# --------------------------------------------------------------------------- #

from router import apply_handoff_note, build_handoff_note, message_has_text  # noqa: E402


def _body(messages, model="m"):
    return json.dumps({"model": model, "messages": messages}).encode()


def _messages(body):
    return json.loads(body.decode())["messages"]


_QWEN_TURN = {
    "role": "assistant",
    "content": "I am Qwen, a large language model created by Alibaba Cloud.",
}


def test_the_note_names_the_previous_model():
    """'A different model' alone leaves the transcript more concrete than the
    correction; naming it is the whole value."""
    note = build_handoff_note("Qwen2.5-VL-7B-Instruct-Q4_K_M")
    assert "Qwen2.5-VL-7B-Instruct-Q4_K_M" in note


def test_the_note_does_not_tell_the_model_what_it_is():
    """LOCITIZE has no honest way to know that - a GGUF header carries no vendor
    identity and the registry id is a filename the owner chose. The note states
    only what LOCITIZE genuinely knows: who wrote the earlier turns."""
    note = build_handoff_note("Some-Other-Model").lower()
    assert "you are" not in note
    assert "whichever model you actually are" in note


def test_the_note_is_added_when_a_conversation_changes_hands():
    body = apply_handoff_note(
        _body([{"role": "user", "content": "Who are you?"}, _QWEN_TURN,
               {"role": "user", "content": "Who are you?"}]),
        "Qwen2.5-VL-7B-Instruct-Q4_K_M",
    )
    messages = _messages(body)
    assert messages[0]["role"] == "system"
    assert "Qwen2.5-VL-7B-Instruct-Q4_K_M" in messages[0]["content"]
    # The conversation itself is untouched, in order.
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user"]


def test_no_note_on_the_first_message_of_a_conversation():
    """No assistant turn yet means no persona to inherit; a note about turns
    that do not exist is noise in the prompt."""
    original = _body([{"role": "user", "content": "hi"}])
    assert apply_handoff_note(original, "Whatever") == original


def test_no_note_when_the_only_assistant_turn_is_empty():
    original = _body(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "   "}]
    )
    assert apply_handoff_note(original, "Whatever") == original


def test_the_note_is_appended_to_an_existing_system_message():
    """Several chat templates render only the FIRST system message and silently
    drop later ones, so a second system message is not reliably delivered."""
    body = apply_handoff_note(
        _body([{"role": "system", "content": "You are a terse assistant."},
               {"role": "user", "content": "hi"}, _QWEN_TURN,
               {"role": "user", "content": "and now?"}]),
        "Qwen2.5-VL",
    )
    messages = _messages(body)
    assert sum(1 for m in messages if m["role"] == "system") == 1
    assert messages[0]["content"].startswith("You are a terse assistant.")
    assert "Qwen2.5-VL" in messages[0]["content"]


def test_a_non_string_system_content_is_not_spliced():
    """A content ARRAY is not ours to concatenate into; add a message instead of
    corrupting one."""
    body = apply_handoff_note(
        _body([{"role": "system", "content": [{"type": "text", "text": "hi"}]},
               _QWEN_TURN, {"role": "user", "content": "?"}]),
        "Qwen2.5-VL",
    )
    messages = _messages(body)
    assert messages[0]["content"].startswith("LOCITIZE note:")
    assert messages[1]["content"] == [{"type": "text", "text": "hi"}]


def test_a_multimodal_assistant_turn_with_text_still_counts():
    """Open WebUI switches to the content-array shape once an image is attached."""
    assert message_has_text(
        {"role": "assistant", "content": [{"type": "text", "text": "I am Qwen."}]}
    )
    assert not message_has_text(
        {"role": "assistant", "content": [{"type": "image_url", "image_url": {}}]}
    )


def test_an_unparseable_body_is_forwarded_untouched():
    """The router must never corrupt a request it could not fully parse."""
    for raw in (b"", b"not json", b'["a","list"]', b'{"messages": "not a list"}'):
        assert apply_handoff_note(raw, "Whatever") == raw


def test_the_note_preserves_every_other_field_of_the_request():
    original = json.dumps({
        "model": "m", "messages": [_QWEN_TURN, {"role": "user", "content": "?"}],
        "temperature": 0.3, "stream": True, "max_tokens": 256,
    }).encode()
    payload = json.loads(apply_handoff_note(original, "Prev").decode())
    assert payload["temperature"] == 0.3
    assert payload["stream"] is True
    assert payload["max_tokens"] == 256
    assert payload["model"] == "m"


def test_the_note_can_be_turned_off():
    """Anyone comparing models on byte-identical prompts wants it off."""
    router = ModelRouter(
        port=0, registry_fn=lambda: _rows(("m", "M")),
        running_model_id_fn=lambda: None, port_provider=lambda: 8080,
        switch_fn=lambda mid: (True, "RUNNING"), handoff_note=False,
    )
    assert router.handoff_note_enabled is False
    assert _router().handoff_note_enabled is True  # on by default


def test_display_name_falls_back_to_the_id():
    """A row that vanished between the switch and the note must degrade, not
    break the request."""
    router = _router(rows=_rows(("qwen3-14b-q5_0", "Qwen3 14B")))
    assert router.display_name("qwen3-14b-q5_0") == "Qwen3 14B"
    assert router.display_name("gone-from-the-registry") == "gone-from-the-registry"


# --- inline tagging: the half that measurement showed was load-bearing ------ #

from router import EARLIER_MODEL_TAG, tag_assistant_turn  # noqa: E402


def test_prior_assistant_turns_are_tagged_inline():
    """The system note ALONE was measured at 4/6 on gpt-oss-20b; tagging the
    turns themselves is what carried it. A system line loses to an explicit
    prior turn written in the model's own voice."""
    body = apply_handoff_note(
        _body([{"role": "user", "content": "Who are you?"}, _QWEN_TURN,
               {"role": "user", "content": "Who are you?"}]),
        "Qwen2.5-VL",
    )
    messages = _messages(body)
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert assistant and assistant[0]["content"].startswith(EARLIER_MODEL_TAG)
    assert "I am Qwen" in assistant[0]["content"]  # the words are preserved


def test_the_inline_tag_is_neutral_not_an_attribution_claim():
    """The router knows the PREVIOUS model, not the author of every historic
    turn. A conversation that went A -> B -> A contains turns the incoming model
    really did write, so tagging each one 'written by B' would be a fabrication.
    The marker says only that the turn is earlier; the note carries the
    'most recently' qualifier."""
    assert "written by" not in EARLIER_MODEL_TAG.lower()
    note = build_handoff_note("Prev-Model").lower()
    assert "may be" in note and "most recently" in note


def test_the_user_turns_are_never_tagged():
    body = apply_handoff_note(
        _body([{"role": "user", "content": "Who are you?"}, _QWEN_TURN,
               {"role": "user", "content": "again?"}]),
        "Prev",
    )
    for message in _messages(body):
        if message["role"] == "user":
            assert not message["content"].startswith(EARLIER_MODEL_TAG)


def test_tagging_does_not_stack_across_repeated_switches():
    """A conversation switched several times must not accumulate markers."""
    once = apply_handoff_note(
        _body([_QWEN_TURN, {"role": "user", "content": "?"}]), "A")
    twice = apply_handoff_note(once, "B")
    assistant = [m for m in _messages(twice) if m["role"] == "assistant"][0]
    assert assistant["content"].count(EARLIER_MODEL_TAG) == 1
    assert twice.decode().count("LOCITIZE note: this conversation") == 1


def test_tagging_handles_the_multimodal_content_shape():
    """Open WebUI switches to a content array once an image is attached."""
    tagged = tag_assistant_turn(
        {"role": "assistant",
         "content": [{"type": "text", "text": "I am Qwen."},
                     {"type": "image_url", "image_url": {"url": "data:..."}}]}
    )
    assert tagged["content"][0]["text"].startswith(EARLIER_MODEL_TAG)
    assert tagged["content"][1] == {"type": "image_url", "image_url": {"url": "data:..."}}


def test_the_note_tells_the_model_not_to_echo_the_marker():
    assert "not copy the marker" in build_handoff_note("P")


def test_the_original_messages_are_not_mutated_in_place():
    """The router rewrites only the FORWARDED request. Open WebUI's stored
    conversation must be untouched, so the markers never appear in the owner's
    chat history."""
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "I am Qwen."}]
    snapshot = json.dumps(messages)
    apply_handoff_note(_body(messages), "Prev")
    assert json.dumps(messages) == snapshot


# --------------------------------------------------------------------------- #
# The running model heads the listing (owner-observed 2026-09-03).
#
# LOCITIZE was serving Kimi-VL while a NEW Open WebUI chat showed gemma-4-E2B.
# Open WebUI cannot ask what is loaded - ui.default_models is null on this
# install - so a new chat falls back to the FIRST entry of /v1/models, which
# was simply the first registry row.
# --------------------------------------------------------------------------- #


def test_the_running_model_is_listed_first():
    rows = _rows(("gemma-4-e2b", "Gemma"), ("kimi-vl", "Kimi"), ("gpt-oss", "GPT"))
    payload = build_models_payload(rows, "kimi-vl")
    assert [r["id"] for r in payload["data"]] == ["kimi-vl", "gemma-4-e2b", "gpt-oss"]


def test_the_rest_keep_registry_order_behind_it():
    """Only ONE entry moves. Reshuffling the whole list on every switch would
    make the picker unreadable."""
    rows = _rows(("a", "A"), ("b", "B"), ("c", "C"), ("d", "D"))
    payload = build_models_payload(rows, "c")
    assert [r["id"] for r in payload["data"]] == ["c", "a", "b", "d"]


def test_nothing_running_leaves_the_order_untouched():
    rows = _rows(("a", "A"), ("b", "B"))
    for running in (None, ""):
        payload = build_models_payload(rows, running)
        assert [r["id"] for r in payload["data"]] == ["a", "b"]


def test_a_running_model_absent_from_the_registry_changes_nothing():
    """A row removed while its server still runs must not drop or duplicate
    anything in the listing."""
    rows = _rows(("a", "A"), ("b", "B"))
    payload = build_models_payload(rows, "deleted-since")
    assert [r["id"] for r in payload["data"]] == ["a", "b"]


def test_reordering_never_drops_or_duplicates_a_model():
    rows = _rows(*[(f"m{i}", f"M{i}") for i in range(19)])
    payload = build_models_payload(rows, "m7")
    ids = [r["id"] for r in payload["data"]]
    assert len(ids) == 19
    assert len(set(ids)) == 19
    assert ids[0] == "m7"


def test_the_advertised_name_is_not_decorated_with_running():
    """Clients send the advertised name back, and resolve_requested_model
    matches it exactly - a "(running)" suffix would stop resolving."""
    rows = _rows(("kimi-vl", "Kimi-VL-A3B-Thinking-2506-Q4_K_M"))
    payload = build_models_payload(rows, "kimi-vl")
    advertised = payload["data"][0]["name"]
    assert advertised == "Kimi-VL-A3B-Thinking-2506-Q4_K_M"
    assert resolve_requested_model(advertised, rows) == "kimi-vl"


# --------------------------------------------------------------------------- #
# OpenAI-shaped audio over LOCITIZE's own speech services (owner request
# 2026-09-03: "Open WebUI needs to be my interface for everything").
#
# Open WebUI asks for /v1/audio/transcriptions and /v1/audio/speech; LOCITIZE
# serves whisper.cpp's /inference and kokoro_server's /synthesize. The gap was
# pure dialect.
# --------------------------------------------------------------------------- #

import audio_api  # noqa: E402
from audio_filter import AudioFilterResult  # noqa: E402
from router import MODEL_LIST_PATHS, SWITCHING_PATHS  # noqa: E402


def test_audio_paths_are_not_model_switching_paths():
    """An audio request must NEVER trigger a model load: whisper and Kokoro run
    beside the LLM on their own ports and do not compete for the model slot."""
    for path in audio_api.TRANSCRIPTION_PATHS + audio_api.SPEECH_PATHS:
        assert path not in SWITCHING_PATHS
        assert path not in MODEL_LIST_PATHS


def test_a_recorded_upload_round_trips_through_the_multipart_parser():
    body, content_type = audio_api.build_whisper_upload(b"RIFFfake-wav", "clip.webm")
    audio, filename = audio_api.extract_upload(body, content_type)
    assert audio == b"RIFFfake-wav"
    assert filename == "clip.webm"


def test_a_nameless_blob_is_still_accepted():
    """Open WebUI's recorder sends the part named 'file' without a filename;
    requiring both would reject it."""
    boundary = "xyz"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + b"AUDIO" + f"\r\n--{boundary}--\r\n".encode()
    got = audio_api.extract_upload(body, f"multipart/form-data; boundary={boundary}")
    assert got is not None and got[0] == b"AUDIO"


def test_a_body_that_is_not_multipart_is_reported_not_guessed():
    assert audio_api.extract_upload(b"", "multipart/form-data; boundary=x") is None
    assert audio_api.extract_upload(b"{}", "application/json") is None
    assert audio_api.extract_upload(b"garbage", "multipart/form-data; boundary=x") is None


def test_openais_default_voice_falls_back_instead_of_failing():
    """Open WebUI ships voice 'alloy'; Kokoro has af_bella/am_adam/am_michael/
    bf_emma and REJECTS an unknown voice, so a straight pass-through would 400
    on every read-aloud."""
    voices = ["af_bella", "am_adam", "am_michael", "bf_emma"]
    assert audio_api.resolve_voice("alloy", voices, "am_michael") == "am_michael"
    assert audio_api.resolve_voice("", voices, "am_michael") == "am_michael"


def test_a_real_kokoro_voice_is_honoured_exactly():
    voices = ["af_bella", "am_michael"]
    assert audio_api.resolve_voice("af_bella", voices, "am_michael") == "af_bella"
    assert audio_api.resolve_voice("AF_BELLA", voices, "am_michael") == "af_bella"


def test_the_speech_body_renames_input_to_text():
    """The one field that matters: OpenAI's `input` is Kokoro's `text`. Pointing
    Open WebUI straight at kokoro_server fails with 'text is required'."""
    payload, err = audio_api.speech_to_kokoro(
        json.dumps({"model": "tts-1", "input": "Hello", "voice": "alloy",
                    "speed": 1.5}).encode(),
        ["am_michael"], "am_michael")
    assert err == ""
    assert payload == {"text": "Hello", "voice": "am_michael", "speed": 1.5}


def test_a_speech_body_without_input_is_refused():
    payload, err = audio_api.speech_to_kokoro(
        b'{"model": "tts-1"}', ["am_michael"], "am_michael")
    assert payload is None and "input is required" in err


def test_a_malformed_speech_body_is_refused_not_guessed():
    for raw in (b"", b"not json", b'["a"]'):
        payload, err = audio_api.speech_to_kokoro(raw, ["am_michael"], "am_michael")
        assert payload is None and err


def test_a_bad_speed_degrades_to_normal_rather_than_failing():
    payload, _err = audio_api.speech_to_kokoro(
        json.dumps({"input": "hi", "speed": "fast"}).encode(),
        ["am_michael"], "am_michael")
    assert payload["speed"] == 1.0


def test_the_transcription_reply_is_the_shape_openai_clients_expect():
    body, ctype = audio_api.transcription_body("hello there")
    assert json.loads(body.decode()) == {"text": "hello there"}
    assert ctype == "application/json"
    body, ctype = audio_api.transcription_body("hello there", "text")
    assert body == b"hello there" and ctype.startswith("text/plain")


def test_an_unsupported_response_format_serves_json_rather_than_fabricating():
    """srt/vtt need timing data whisper-server does not return here; emitting an
    empty subtitle file would be a fabricated response."""
    for fmt in ("srt", "vtt", "verbose_json"):
        body, ctype = audio_api.transcription_body("hi", fmt)
        assert ctype == "application/json"
        assert json.loads(body.decode()) == {"text": "hi"}


def test_a_failed_transcription_is_reported_not_returned_as_silence():
    """A silently-empty transcript reads as 'it heard nothing' when the truth is
    'it failed'."""
    text, problem = audio_api.transcript_from_whisper(b"<html>error</html>")
    assert text is None and problem
    text, problem = audio_api.transcript_from_whisper(b'{"nope": 1}')
    assert text is None and problem
    assert audio_api.transcript_from_whisper(b'{"text": "  hi  "}') == ("hi", "")


def test_the_router_reports_missing_speech_services_rather_than_pretending():
    router = ModelRouter(
        port=0, registry_fn=lambda: [], running_model_id_fn=lambda: None,
        port_provider=lambda: None, switch_fn=lambda m: (True, "RUNNING"),
    )
    assert router.whisper_port() is None
    assert router.kokoro_port() is None
    assert router.kokoro_voices() == []


def test_a_raising_audio_provider_is_not_a_crash():
    def boom():
        raise RuntimeError("service went away")

    router = ModelRouter(
        port=0, registry_fn=lambda: [], running_model_id_fn=lambda: None,
        port_provider=lambda: None, switch_fn=lambda m: (True, "RUNNING"),
        audio={"whisper_port": boom, "kokoro_port": boom, "voices": boom,
               "default_voice": boom},
    )
    assert router.whisper_port() is None
    assert router.kokoro_port() is None
    assert router.kokoro_voices() == []
    assert router.default_voice() == ""


def test_audio_wiring_is_read_fresh_so_a_late_start_is_picked_up():
    """Starting whisper mid-session must not need a router restart."""
    state = {"port": None}
    router = ModelRouter(
        port=0, registry_fn=lambda: [], running_model_id_fn=lambda: None,
        port_provider=lambda: None, switch_fn=lambda m: (True, "RUNNING"),
        audio={"whisper_port": lambda: state["port"]},
    )
    assert router.whisper_port() is None
    state["port"] = 8091
    assert router.whisper_port() == 8091


def _start_whisper_capture():
    """A loopback whisper-shaped endpoint that records the forwarded upload."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(size)
            self.server.upload = audio_api.extract_upload(  # type: ignore[attr-defined]
                body, self.headers.get("Content-Type", "")
            )
            payload = b'{"text":"clear speech"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.upload = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _post_transcription(router, audio=b"raw browser audio", filename="clip.webm"):
    """Send one real HTTP multipart request through a running ModelRouter."""
    import http.client

    body, content_type = audio_api.build_whisper_upload(audio, filename)
    port = router._server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(
            "POST",
            "/v1/audio/transcriptions",
            body=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
            },
        )
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_transcription_filters_audio_before_the_whisper_hop():
    upstream, thread = _start_whisper_capture()
    seen = []

    def filter_upload(audio, filename):
        seen.append((audio, filename))
        return AudioFilterResult(
            b"filtered wav bytes", "audio.wav", True, "ffmpeg-afftdn", 12.5
        )

    router = ModelRouter(
        port=0, registry_fn=lambda: [], running_model_id_fn=lambda: None,
        port_provider=lambda: None, switch_fn=lambda m: (True, "RUNNING"),
        audio={
            "whisper_port": upstream.server_address[1],
            "filter_upload": filter_upload,
            "noise_suppression_mode": "balanced",
        },
    )
    router.start()
    try:
        status, body = _post_transcription(router)
        assert status == 200
        assert json.loads(body) == {"text": "clear speech"}
        assert seen == [(b"raw browser audio", "clip.webm")]
        assert upstream.upload == (b"filtered wav bytes", "audio.wav")
        assert router.noise_suppression_mode() == "balanced"
    finally:
        router.stop()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def test_explicit_off_mode_forwards_byte_identical_audio():
    upstream, thread = _start_whisper_capture()
    source = b"original recording"
    router = ModelRouter(
        port=0, registry_fn=lambda: [], running_model_id_fn=lambda: None,
        port_provider=lambda: None, switch_fn=lambda m: (True, "RUNNING"),
        audio={
            "whisper_port": upstream.server_address[1],
            "filter_upload": lambda audio, name: AudioFilterResult(
                audio, name, False, "off", 0.0
            ),
        },
    )
    router.start()
    try:
        status, _body = _post_transcription(router, source, "original.webm")
        assert status == 200
        assert upstream.upload == (source, "original.webm")
    finally:
        router.stop()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "error,expected_status,expected_type",
    [
        ("unavailable: FFmpeg missing", 503, "noise_suppression_unavailable"),
        ("timeout: processing took too long", 504, "audio_processing_timeout"),
        ("processing_failed: corrupt audio", 422, "audio_processing_failed"),
    ],
)
def test_filter_failures_are_typed_http_errors(error, expected_status, expected_type):
    router = ModelRouter(
        port=0, registry_fn=lambda: [], running_model_id_fn=lambda: None,
        port_provider=lambda: None, switch_fn=lambda m: (True, "RUNNING"),
        audio={
            "whisper_port": 9,
            "filter_upload": lambda audio, name: AudioFilterResult(
                b"", "audio.wav", False, "ffmpeg-afftdn", 1.0, error
            ),
        },
    )
    router.start()
    try:
        status, body = _post_transcription(router)
        assert status == expected_status
        parsed = json.loads(body)["error"]
        assert parsed["type"] == expected_type
        assert error.split(":", 1)[-1].strip() not in parsed["message"]
    finally:
        router.stop()


def test_a_raising_or_invalid_filter_result_is_contained_as_502():
    for callback in (
        lambda audio, name: (_ for _ in ()).throw(RuntimeError("boom")),
        lambda audio, name: "not a result",
    ):
        router = ModelRouter(
            port=0, registry_fn=lambda: [], running_model_id_fn=lambda: None,
            port_provider=lambda: None, switch_fn=lambda m: (True, "RUNNING"),
            audio={"whisper_port": 9, "filter_upload": callback},
        )
        router.start()
        try:
            status, body = _post_transcription(router)
            assert status == 502
            parsed = json.loads(body)["error"]
            assert parsed["type"] == "audio_processing_failed"
            assert "boom" not in parsed["message"]
        finally:
            router.stop()


def test_router_recovers_after_one_filter_failure():
    upstream, thread = _start_whisper_capture()
    attempts = iter([False, True])

    def filter_upload(audio, filename):
        if not next(attempts):
            return AudioFilterResult(
                b"", "audio.wav", False, "ffmpeg-afftdn", 1.0,
                "processing_failed: bad first recording",
            )
        return AudioFilterResult(b"recovered wav", "audio.wav", True, "ffmpeg-afftdn", 2.0)

    router = ModelRouter(
        port=0, registry_fn=lambda: [], running_model_id_fn=lambda: None,
        port_provider=lambda: None, switch_fn=lambda m: (True, "RUNNING"),
        audio={"whisper_port": upstream.server_address[1], "filter_upload": filter_upload},
    )
    router.start()
    try:
        first, _ = _post_transcription(router)
        second, body = _post_transcription(router)
        assert first == 422
        assert second == 200
        assert json.loads(body)["text"] == "clear speech"
        assert upstream.upload == (b"recovered wav", "audio.wav")
    finally:
        router.stop()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


# --------------------------------------------------------------------------- #
# Chunked request bodies (owner-observed 2026-09-03).
#
# Open WebUI's voice mode reported "Error transcribing chunk: 400 Bad Request"
# from the phone. It streams recorded audio with Transfer-Encoding: chunked and
# no Content-Length - aiohttp does that whenever the payload size is not known
# upfront, which for a live recording it never is. Reading only Content-Length
# yielded an empty body, so every utterance "contained no audio file".
# --------------------------------------------------------------------------- #


class _FakeRfile:
    """Just enough of a socket file for _read_body: readline + read."""

    def __init__(self, data: bytes) -> None:
        self._buf = data
        self._pos = 0

    def readline(self, limit: int = -1) -> bytes:
        end = self._buf.find(b"\n", self._pos)
        end = len(self._buf) if end == -1 else end + 1
        if limit and limit > 0:
            end = min(end, self._pos + limit)
        out = self._buf[self._pos:end]
        self._pos = end
        return out

    def read(self, size: int) -> bytes:
        out = self._buf[self._pos:self._pos + size]
        self._pos += size
        return out


def _reader(body: bytes, headers: dict):
    """A _RouterHandler bound to a fake socket, without running __init__."""
    from router import _RouterHandler

    handler = _RouterHandler.__new__(_RouterHandler)
    handler.rfile = _FakeRfile(body)
    handler.headers = headers
    return handler


def _chunked(payload: bytes) -> bytes:
    return b"%X\r\n" % len(payload) + payload + b"\r\n0\r\n\r\n"


def test_a_chunked_body_is_reassembled():
    """The reported defect: this returned b'' and every voice turn 400'd."""
    handler = _reader(_chunked(b"audio-bytes"), {"Transfer-Encoding": "chunked"})
    assert handler._read_body() == b"audio-bytes"


def test_a_multi_chunk_body_is_reassembled_in_order():
    body = b"3\r\nabc\r\n3\r\ndef\r\n0\r\n\r\n"
    handler = _reader(body, {"Transfer-Encoding": "chunked"})
    assert handler._read_body() == b"abcdef"


def test_a_chunk_extension_is_tolerated():
    """A chunk header may carry ';ext' after the size."""
    body = b"5;foo=bar\r\nhello\r\n0\r\n\r\n"
    handler = _reader(body, {"Transfer-Encoding": "chunked"})
    assert handler._read_body() == b"hello"


def test_chunked_trailers_are_consumed():
    body = b"2\r\nhi\r\n0\r\nX-Trailer: v\r\n\r\n"
    handler = _reader(body, {"Transfer-Encoding": "chunked"})
    assert handler._read_body() == b"hi"


def test_the_size_guard_applies_to_chunked_bodies_too():
    """Enforced as chunks ARRIVE, so a sender that never stops cannot make this
    buffer without limit - the whole point of the guard."""
    from router import _MAX_BODY_BYTES

    oversized = b"%X\r\n" % (_MAX_BODY_BYTES + 1)
    handler = _reader(oversized + b"x", {"Transfer-Encoding": "chunked"})
    assert handler._read_body() is None


def test_a_malformed_chunk_header_is_refused():
    handler = _reader(b"notahexsize\r\n", {"Transfer-Encoding": "chunked"})
    assert handler._read_body() is None


def test_content_length_bodies_still_work():
    handler = _reader(b"hello", {"Content-Length": "5"})
    assert handler._read_body() == b"hello"
    handler = _reader(b"", {})
    assert handler._read_body() == b""


def test_whispers_silence_annotation_becomes_empty_not_spoken_words():
    """Owner-observed 2026-09-03: the assistant kept replying "I did not hear
    anything" to spoken turns. The audio was fine and TTS was fine - whisper
    answers a silent window with the literal string "[BLANK_AUDIO]", which was
    passed through as if the owner had said those words. Open WebUI's voice mode
    transcribes CONTINUOUSLY, so most chunks are pauses."""
    for raw in (b'{"text": "[BLANK_AUDIO]"}', b'{"text": "[ Silence ]"}',
                b'{"text": "(music)"}', b'{"text": " . "}'):
        text, problem = audio_api.transcript_from_whisper(raw)
        assert text == "" and problem == "", raw


def test_real_speech_survives_the_silence_gate():
    """The test is structural, not a phrase blocklist: a genuinely spoken
    'Thank you.' contains letters and must pass."""
    for raw, expected in (
        (b'{"text": "Thank you."}', "Thank you."),
        (b'{"text": "Testing voice mode from the phone."}',
         "Testing voice mode from the phone."),
        (b'{"text": "[MUSIC] but I did say this"}', "[MUSIC] but I did say this"),
    ):
        assert audio_api.transcript_from_whisper(raw)[0] == expected


def test_the_silence_gate_reuses_the_mic_paths_predicate():
    """Two transcription paths must not disagree about what counts as silence."""
    import inspect

    src = inspect.getsource(audio_api.transcript_from_whisper)
    assert "is_non_speech" in src


# --------------------------------------------------------------------------- #
# Voice turns (owner request 2026-09-03). A spoken turn arrives from Open
# WebUI's Call overlay as an ordinary chat request whose last user message is,
# verbatim, the transcript the router itself returned a moment earlier. For
# that turn thinking is switched off (measured: a reasoning model says nothing
# until its reasoning ends) and a spoken register is requested.
# --------------------------------------------------------------------------- #

from router import (  # noqa: E402
    VOICE_TURN_NOTE,
    VOICE_TURN_WINDOW_S,
    apply_thinking_defaults,
    apply_voice_turn,
    last_user_utterance,
)


def test_thinking_defaults_turn_thinking_off_without_voice_note():
    """Typed OWUI chats get the same thinking-off defaults as voice turns,
    without the spoken-register system note (app-level stuck
    2026-09-24: empty content)."""
    payload = json.loads(apply_thinking_defaults(
        _body([{"role": "user", "content": "hello"}])
    ).decode())
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "low"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_thinking_defaults_never_override_client_choice():
    original = json.dumps({
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": True, "other": 1},
    }).encode()
    payload = json.loads(apply_thinking_defaults(original).decode())
    assert payload["reasoning_effort"] == "high"
    assert payload["chat_template_kwargs"] == {"enable_thinking": True, "other": 1}


def test_a_voice_turn_turns_thinking_off_both_ways():
    """Both fields, because measurement showed each template honours only one:
    Qwen3/Gemma read chat_template_kwargs.enable_thinking, gpt-oss reads
    reasoning_effort and ignores the other."""
    payload = json.loads(apply_voice_turn(
        _body([{"role": "user", "content": "what time is it"}])
    ).decode())
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "low"


def test_a_voice_turn_asks_for_a_spoken_register_once():
    body = apply_voice_turn(_body([{"role": "user", "content": "hello"}]))
    messages = _messages(body)
    assert messages[0] == {"role": "system", "content": VOICE_TURN_NOTE}
    assert messages[1] == {"role": "user", "content": "hello"}
    assert "markdown" in VOICE_TURN_NOTE
    # Idempotent: a body passed through twice carries one note, not two.
    again = _messages(apply_voice_turn(body))
    assert again == messages


def test_the_spoken_note_is_appended_to_an_existing_system_message():
    """Same rule as the handoff note: several templates render only the FIRST
    system message, so a second one would be silently dropped."""
    body = apply_voice_turn(_body([
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hello"},
    ]))
    messages = _messages(body)
    assert len(messages) == 2
    assert messages[0]["content"].startswith("You are terse.")
    assert messages[0]["content"].endswith(VOICE_TURN_NOTE)


def test_a_voice_turn_never_overrides_what_the_client_chose():
    """A chat whose owner set an effort, or a template kwarg, keeps it: the
    rewrite fills gaps, it does not argue."""
    original = json.dumps({
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": True, "other": 1},
        "temperature": 0.2, "stream": True,
    }).encode()
    payload = json.loads(apply_voice_turn(original).decode())
    assert payload["reasoning_effort"] == "high"
    assert payload["chat_template_kwargs"] == {"enable_thinking": True, "other": 1}
    assert payload["temperature"] == 0.2 and payload["stream"] is True


def test_a_body_that_is_not_a_chat_is_returned_untouched():
    for original in (b"", b"not json", b"[1, 2]", b'{"prompt": "a completion"}',
                     b'{"messages": []}'):
        assert apply_voice_turn(original) == original


def test_last_user_utterance_is_whitespace_folded_and_reads_both_shapes():
    assert last_user_utterance(_body([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "  what   time\nis it "},
    ])) == "what time is it"
    assert last_user_utterance(_body([{"role": "user", "content": [
        {"type": "text", "text": "look at"}, {"type": "image_url", "image_url": {"url": "d"}},
        {"type": "text", "text": "this"},
    ]}])) == "look at this"
    assert last_user_utterance(b"nope") is None
    assert last_user_utterance(_body([{"role": "assistant", "content": "only"}])) is None


def test_a_recent_transcript_makes_the_matching_chat_a_voice_turn():
    router = _router()
    router.record_transcript(" What time  is it? ", now=100.0)
    spoken = _body([{"role": "user", "content": "What time is it?"}])
    typed = _body([{"role": "user", "content": "What time is it in Paris?"}])
    assert router.is_voice_turn(spoken, now=101.0)
    assert not router.is_voice_turn(typed, now=101.0)
    # The window: a transcript the browser never posted must not brand a later
    # typed message with the same words.
    assert router.is_voice_turn(spoken, now=100.0 + VOICE_TURN_WINDOW_S)
    assert not router.is_voice_turn(spoken, now=100.0 + VOICE_TURN_WINDOW_S + 0.1)


def test_only_the_last_user_message_counts():
    """A later typed message that quotes an earlier spoken one is typed."""
    router = _router()
    router.record_transcript("hello there", now=100.0)
    body = _body([
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "now typed"},
    ])
    assert not router.is_voice_turn(body, now=101.0)


def test_empty_transcripts_are_never_remembered():
    """whisper's silence gate returns "" for a pause; it must not match a chat
    with no user text either."""
    router = _router()
    router.record_transcript("   ", now=100.0)
    assert not router.is_voice_turn(_body([{"role": "user", "content": " "}]), now=100.5)


def test_voice_turns_can_be_turned_off():
    router = ModelRouter(
        port=0, registry_fn=lambda: _rows(("m", "M")),
        running_model_id_fn=lambda: None, port_provider=lambda: 8080,
        switch_fn=lambda mid: (True, "RUNNING"), voice_turns=False,
    )
    assert router.voice_turns_enabled is False
    assert _router().voice_turns_enabled is True  # on by default


def test_the_handler_decides_before_the_switch_and_rewrites_after_it():
    """Found live 2026-09-03: checked AFTER _ensure_model_for, a spoken turn
    that triggered a 16s model load fell outside the transcript window and
    was forwarded as typed - the model then thought for its whole answer. So
    the decision is taken before the switch and the rewrite lands after it (a
    refused switch has already answered), gated on the setting."""
    import inspect

    import router as router_module

    src = inspect.getsource(router_module._RouterHandler._handle)
    decide = src.index("router.voice_turns_enabled and router.is_voice_turn(body)")
    switch = src.index("_ensure_model_for(router, body)")
    rewrite = src.index("apply_voice_turn(body)")
    assert decide < switch < rewrite
    transcription = inspect.getsource(router_module._RouterHandler._handle_transcription)
    assert "router.record_transcript(text)" in transcription


# --------------------------------------------------------------------------- #
# Streaming relay (owner-observed 2026-09-03: "my models are not streaming
# words, they are blasting them"). read(8192) on a chunked upstream blocks
# until 8 KB have accumulated; the relay must pass each chunk on as it lands.
# --------------------------------------------------------------------------- #


def test_the_relay_passes_each_upstream_chunk_on_as_it_arrives():
    import http.client
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    gap_s = 0.25

    class SlowUpstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(3):
                data = f"data: {{\"token\": {i}}}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
                time.sleep(gap_s)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, *args):
            pass

    upstream = HTTPServer(("127.0.0.1", 0), SlowUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    router = ModelRouter(
        port=0, registry_fn=lambda: _rows(("m", "M")),
        running_model_id_fn=lambda: "m",
        port_provider=lambda: upstream.server_address[1],
        switch_fn=lambda mid: (True, "RUNNING"),
    )
    router.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", router._server.server_address[1], timeout=10)
        body = _body([{"role": "user", "content": "hi"}])
        started = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        arrivals = []
        while True:
            chunk = response.read1(8192)
            if not chunk:
                break
            arrivals.append(time.perf_counter() - started)
        assert len(arrivals) >= 3
        # The first token event lands well before the upstream has finished
        # (3 events, gap_s apart): a relay that waited for the whole stream
        # would deliver everything at ~3 * gap_s.
        assert arrivals[0] < gap_s
        assert arrivals[-1] - arrivals[0] >= gap_s
    finally:
        router.stop()
        upstream.shutdown()
        upstream.server_close()


# --------------------------------------------------------------------------- #
# model identity (owner request 2026-09-24: "still not saying the model name")
# --------------------------------------------------------------------------- #
# A local model has no dependable way to know which GGUF it was loaded from, so
# it answers "what model are you?" from its training data or from whatever the
# client's system prompt claimed. The router performed the switch, so it names
# the row on the forwarded request.

from router import (  # noqa: E402
    _IDENTITY_SENTINEL,
    apply_model_identity,
    build_identity_note,
)


def test_the_running_model_is_named_in_the_system_message():
    body = apply_model_identity(
        _body([{"role": "user", "content": "what model are you"}]),
        "Qwen3-30B-A3B-Instruct-Q4_K_M",
    )
    messages = _messages(body)
    assert messages[0]["role"] == "system"
    assert "Qwen3-30B-A3B-Instruct-Q4_K_M" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "what model are you"}


def test_the_identity_note_is_appended_to_an_existing_system_message():
    """Same rule as the handoff and spoken notes: templates render only the
    FIRST system message, so a second one would be silently dropped."""
    body = apply_model_identity(_body([
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
    ]), "Qwen3.6-35B-A3B")
    messages = _messages(body)
    assert len(messages) == 2
    assert messages[0]["content"].startswith("You are terse.")
    assert messages[0]["content"].endswith(build_identity_note("Qwen3.6-35B-A3B"))


def test_naming_the_model_is_idempotent():
    once = apply_model_identity(_body([{"role": "user", "content": "hi"}]), "M")
    twice = apply_model_identity(once, "M")
    assert twice == once
    assert _messages(once)[0]["content"].count(_IDENTITY_SENTINEL) == 1


def test_an_unknown_name_leaves_the_request_alone():
    """An unnamed model is better than a wrong name."""
    original = _body([{"role": "user", "content": "hi"}])
    for name in (None, "", "   "):
        assert apply_model_identity(original, name) == original


def test_a_body_that_is_not_a_chat_is_not_named():
    for original in (b"", b"not json", b"[1, 2]", b'{"prompt": "a completion"}',
                     b'{"messages": []}'):
        assert apply_model_identity(original, "M") == original


def test_the_router_resolves_the_running_display_name():
    rows = _rows(("qwen3-30b", "Qwen3 30B Instruct"), ("qwen", "Qwen3.6"))
    assert _router(running="qwen3-30b", rows=rows).running_model_name() == (
        "Qwen3 30B Instruct"
    )
    # Nothing loaded, and a row that vanished from the registry mid-session.
    assert _router(running=None, rows=rows).running_model_name() is None
    assert _router(running="gone", rows=rows).running_model_name() == "gone"


def test_the_forwarded_chat_is_named_after_the_switch_not_before():
    """The name must be the model that will ANSWER: a request that triggers a
    switch would otherwise be labelled with the model it replaced."""
    import inspect

    import router as router_module

    src = inspect.getsource(router_module._RouterHandler._handle)
    switch = src.index("_ensure_model_for(router, body)")
    naming = src.index("apply_model_identity(body, router.running_model_name())")
    spoken = src.index("apply_voice_turn(body)")
    assert switch < naming < spoken


# --------------------------------------------------------------------------- #
# Portal / tooling guard: Laya pre-steer + broken-model redirect
# --------------------------------------------------------------------------- #


def test_tool_broken_model_is_detected_by_substring():
    assert is_tool_broken_model("qwen3-8-27b-obliterated-q3_k_m")
    assert is_tool_broken_model("huihui-qwen3-8-27b-abliterated-ud-iq4_xs")
    assert is_tool_broken_model("qwen3-8-27b-heretic-q4_k_m")
    assert not is_tool_broken_model("qwen3-30b-a3b-instruct-q4_k_m")
    assert not is_tool_broken_model("qwen3-coder-30b-a3b-ud-iq3_xxs")
    assert not is_tool_broken_model("")


def test_request_has_tools_requires_a_non_empty_tools_array():
    assert request_has_tools(json.dumps({"tools": [{"type": "function"}]}).encode())
    assert not request_has_tools(json.dumps({"tools": []}).encode())
    assert not request_has_tools(json.dumps({"messages": []}).encode())
    assert not request_has_tools(b"not json")


def test_ask_needs_tools_matches_browse_list_read_desktop():
    assert ask_needs_tools("list the files in this folder")
    assert ask_needs_tools("what's this folder about")
    assert ask_needs_tools("screenshot the screen")
    assert ask_needs_tools("what's the weather in Austin")
    assert not ask_needs_tools("hello there")
    assert not ask_needs_tools("thanks")


def test_messages_have_tool_exchange_detects_prior_tool_turns():
    bare = json.dumps({
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [{"type": "function"}],
    }).encode()
    assert not messages_have_tool_exchange(bare)
    with_tool = json.dumps({
        "messages": [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "{}"},
        ],
        "tools": [{"type": "function"}],
    }).encode()
    assert messages_have_tool_exchange(with_tool)


def test_pick_tool_capable_fallback_prefers_qwen3_coder():
    rows = _rows(
        ("qwen3-8-27b-obliterated-q3_k_m", "Obliterated"),
        ("qwen3-30b-a3b-instruct-q4_k_m", "Instruct"),
        ("qwen3-coder-30b-a3b-ud-iq3_xxs", "Coder"),
        ("gpt-oss-20b-f16", "GPT-OSS"),
    )
    assert pick_tool_capable_fallback(rows) == "qwen3-coder-30b-a3b-ud-iq3_xxs"


def test_pick_tool_capable_fallback_skips_broken_and_can_be_empty():
    only_broken = _rows(("qwen3-8-27b-obliterated-q3_k_m", "Obliterated"))
    assert pick_tool_capable_fallback(only_broken) is None
    assert pick_tool_capable_fallback([]) is None


def test_rewrite_body_model_updates_only_the_model_field():
    body = json.dumps({"model": "bad", "messages": [], "n": 1}).encode()
    out = json.loads(rewrite_body_model(body, "qwen3-coder-30b-a3b-ud-iq3_xxs"))
    assert out["model"] == "qwen3-coder-30b-a3b-ud-iq3_xxs"
    assert out["n"] == 1


def test_should_pre_steer_on_tools_needed_ask_without_prior_tools():
    body = json.dumps({
        "model": "qwen3-30b-a3b-instruct-q4_k_m",
        "messages": [{
            "role": "user",
            "content": "Project folder: proj\n\nTask: list the files",
        }],
        "tools": [
            {"type": "function", "function": {"name": "laya_route_tools"}},
            {"type": "function", "function": {"name": "list_dir"}},
        ],
    }).encode()
    assert should_pre_steer_laya(body, "qwen3-30b-a3b-instruct-q4_k_m")


def test_should_not_pre_steer_a_client_without_the_laya_tool():
    """Codex/OpenCode-style clients never get a call to a tool they lack."""
    for model in ("qwen3-30b-a3b-instruct-q4_k_m", "qwen3-8-27b-obliterated-q3_k_m"):
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "list the files in this project"}],
            "tools": [{"type": "function", "function": {"name": "list_dir"}}],
        }).encode()
        assert not should_pre_steer_laya(body, model)


def test_should_pre_steer_on_broken_model_even_for_a_plain_hello():
    body = json.dumps({
        "model": "qwen3-8-27b-obliterated-q3_k_m",
        "messages": [{"role": "user", "content": "Project folder: X\n\nTask: hello"}],
        "tools": [{"type": "function", "function": {"name": "laya_route_tools"}}],
    }).encode()
    assert should_pre_steer_laya(body, "qwen3-8-27b-obliterated-q3_k_m")


def test_should_not_pre_steer_happy_path_or_second_turn():
    happy = json.dumps({
        "model": "qwen3-30b-a3b-instruct-q4_k_m",
        "messages": [{"role": "user", "content": "Project folder: X\n\nTask: hello"}],
        "tools": [{"type": "function"}],
    }).encode()
    assert not should_pre_steer_laya(happy, "qwen3-30b-a3b-instruct-q4_k_m")

    second = json.dumps({
        "model": "qwen3-30b-a3b-instruct-q4_k_m",
        "messages": [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_laya_steer_1", "type": "function",
                                "function": {"name": "laya_route_tools", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_laya_steer_1", "content": "{\"tool\": \"file\"}"},
        ],
        "tools": [{"type": "function"}],
    }).encode()
    assert not should_pre_steer_laya(second, "qwen3-30b-a3b-instruct-q4_k_m")

    no_tools = json.dumps({
        "model": "qwen3-30b-a3b-instruct-q4_k_m",
        "messages": [{"role": "user", "content": "list the files"}],
    }).encode()
    assert not should_pre_steer_laya(no_tools, "qwen3-30b-a3b-instruct-q4_k_m")


def test_build_laya_steer_completion_is_openai_shaped_tool_call():
    payload = build_laya_steer_completion(
        model_id="qwen3-coder-30b-a3b-ud-iq3_xxs",
        task="list the files",
        created=1,
    )
    assert payload["object"] == "chat.completion"
    assert payload["id"] == "chatcmpl-laya-steer"
    assert payload["model"] == "qwen3-coder-30b-a3b-ud-iq3_xxs"
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "laya_route_tools"
    assert json.loads(call["function"]["arguments"])["task"] == "list the files"

def test_promote_reasoning_to_content_nonstream():
    from router import promote_reasoning_to_content

    empty = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "hello",
                }
            }
        ]
    }
    out = promote_reasoning_to_content(empty)
    assert out["choices"][0]["message"]["content"] == "hello"

    kept = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "keep",
                    "reasoning_content": "ignore",
                }
            }
        ]
    }
    out2 = promote_reasoning_to_content(kept)
    assert out2["choices"][0]["message"]["content"] == "keep"


def test_promote_sse_delta_reasoning():
    from router import promote_sse_chunk

    raw = (
        b'data: {"choices":[{"delta":{"content":"","reasoning_content":"hi"}}]}\n\n'
    )
    out = promote_sse_chunk(raw)
    assert b'"content": "hi"' in out or b'"content":"hi"' in out


def test_promote_completion_bytes_grows_when_reasoning_fills_blank_content():
    """Promote must lengthen the JSON when content was empty."""
    from router import promote_completion_bytes

    upstream = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "think " * 40,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "browse_open",
                                "arguments": '{"url":"https://en.wikipedia.org"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "timings": {"predicted_ms": 12.5},
    }
    raw = json.dumps(upstream, ensure_ascii=False).encode("utf-8")
    out = promote_completion_bytes(raw)
    assert len(out) > len(raw)
    payload = json.loads(out)
    assert payload["choices"][0]["message"]["content"] == (
        payload["choices"][0]["message"]["reasoning_content"]
    )


def test_forwarded_json_content_length_matches_promoted_body():
    """Stale upstream Content-Length must not truncate promote.

    Upstream advertises CL for the pre-promote body (empty content + reasoning).
    After promote the body grows; the router must recompute Content-Length so
    http.client (and portal json.loads) see a complete closing brace.
    """
    import http.client
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from router import promote_completion_bytes

    reasoning = ("plan browse wikipedia then answer. " * 20).strip()
    upstream_payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": reasoning,
                },
                "finish_reason": "stop",
            }
        ],
        "timings": {
            "prompt_n": 10,
            "predicted_n": 80,
            "predicted_ms": 1234.5,
            "predicted_per_second": 12.34,
        },
    }
    upstream_raw = json.dumps(upstream_payload, ensure_ascii=False).encode("utf-8")
    expected = promote_completion_bytes(upstream_raw)
    assert len(expected) > len(upstream_raw)

    class GrowUpstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            # Deliberately advertise the PRE-promote length (the bug).
            self.send_header("Content-Length", str(len(upstream_raw)))
            self.end_headers()
            self.wfile.write(upstream_raw)

        def log_message(self, *args):
            pass

    upstream = HTTPServer(("127.0.0.1", 0), GrowUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    router = ModelRouter(
        port=0,
        registry_fn=lambda: _rows(("m", "M")),
        running_model_id_fn=lambda: "m",
        port_provider=lambda: upstream.server_address[1],
        switch_fn=lambda mid: (True, "RUNNING"),
    )
    router.start()
    try:
        conn = http.client.HTTPConnection(
            "127.0.0.1", router._server.server_address[1], timeout=10
        )
        body = _body([{"role": "user", "content": "hi"}])
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        response = conn.getresponse()
        cl = response.getheader("Content-Length")
        data = response.read()
        assert response.status == 200
        assert cl is not None
        assert int(cl) == len(data)
        assert int(cl) == len(expected)
        assert data.rstrip().endswith(b"}")
        payload = json.loads(data)  # must not raise
        assert payload["choices"][0]["message"]["content"] == reasoning
        # Truncation fingerprint from the live bug: mid-timings cut.
        assert b"predicted_per_second" in data
    finally:
        router.stop()
        upstream.shutdown()
        upstream.server_close()


# --------------------------------------------------------------------------- #
# Browser / DNS-rebinding guard
# --------------------------------------------------------------------------- #


def test_foreign_request_reason_accepts_local_callers():
    from router import foreign_request_reason

    assert foreign_request_reason("127.0.0.1:8093", None, 8093) is None
    assert foreign_request_reason("localhost:8093", None, 8093) is None
    assert foreign_request_reason("127.0.0.1:8093", "http://127.0.0.1:8096", 8093) is None


def test_foreign_request_reason_refuses_websites_and_rebinding():
    from router import foreign_request_reason

    # A web page the user visits: Host is right, Origin is the site.
    assert foreign_request_reason("127.0.0.1:8093", "https://evil.example", 8093)
    # DNS rebinding: the attacker's own host name resolves to 127.0.0.1.
    assert foreign_request_reason("evil.example:8093", None, 8093)
    assert foreign_request_reason("127.0.0.1:9999", None, 8093)
    assert foreign_request_reason(None, None, 8093)
    assert foreign_request_reason("127.0.0.1:8093", "null", 8093)
