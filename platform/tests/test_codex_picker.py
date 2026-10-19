"""Codex picker profile/catalog files and helper switching; Codex and models are faked."""
import http.client
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import codex_picker
from local_tuning import LocalTuning


def model(model_id, context=16384, mmproj=None, name=None):
    return SimpleNamespace(id=model_id, name=name or model_id, context_size=context,
                           mmproj=mmproj, location="registered", backend="")


def test_catalog_uses_measured_context_and_configured_modalities():
    catalog = codex_picker.build_catalog([model("vl", 16384, mmproj="p.gguf"), model("text", 32768)], "BASE")
    assert catalog["managed_by"] == codex_picker.CATALOG_MARKER
    vl, text = catalog["models"]
    assert (vl["slug"], vl["context_window"], vl["input_modalities"]) == ("vl", 16384, ["text", "image"])
    assert (text["context_window"], text["input_modalities"]) == (32768, ["text"])
    assert vl["base_instructions"] == "BASE" and vl["visibility"] == "list"
    assert vl["priority"] < text["priority"]


def test_profile_points_at_router_and_catalog_without_credentials(tmp_path):
    text = codex_picker.profile_text(8093, "vl", tmp_path / "c.json")
    assert text.startswith(codex_picker.MARKER)
    assert 'model_provider = "locitize_tuner"' in text and 'model = "vl"' in text
    # Profiles merge with config.toml: an existing [model_providers.locitize]
    # (with env_key) must not leak into the helper's provider.
    assert "[model_providers.locitize_tuner]" in text and "[model_providers.locitize]" not in text
    assert 'base_url = "http://127.0.0.1:8093/v1"' in text
    assert "env_key" not in text and "api_key" not in text
    # App connectors are off for the local profile only.
    assert text.endswith("[features]\napps = false\n")
    for port, model_id in ((0, "vl"), (70000, "vl"), (8093, 'x"\nmodel_provider="openai'), (8093, "")):
        with pytest.raises(ValueError):
            codex_picker.profile_text(port, model_id, tmp_path / "c.json")


def test_enable_and_disable_touch_only_managed_files(tmp_path):
    user_config = tmp_path / "config.toml"
    user_config.write_text('model = "gpt"\n')
    result = codex_picker.enable([model("vl"), model("text")], "text", 8093, "BASE", home=tmp_path)
    assert result["usage"] == "codex --profile locitize"
    catalog = json.loads((tmp_path / codex_picker.CATALOG_FILE).read_text())
    assert [m["slug"] for m in catalog["models"]] == ["vl", "text"]
    assert Path(json.loads('"' + (tmp_path / codex_picker.PROFILE_FILE).read_text()
                .split("model_catalog_json = \"")[1].split('"')[0] + '"')) == tmp_path / codex_picker.CATALOG_FILE
    # Re-enabling rewrites the helper's own files.
    codex_picker.enable([model("vl")], "vl", 8093, "BASE", home=tmp_path)
    assert codex_picker.disable(home=tmp_path)["removed"] == [codex_picker.PROFILE_FILE, codex_picker.CATALOG_FILE]
    assert user_config.read_text() == 'model = "gpt"\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.toml"]


def test_user_authored_profile_is_never_overwritten_or_removed(tmp_path):
    mine = tmp_path / codex_picker.PROFILE_FILE
    mine.write_text('model = "my-own"\n')
    with pytest.raises(ValueError, match="not created by this helper"):
        codex_picker.enable([model("vl")], "vl", 8093, "BASE", home=tmp_path)
    assert not (tmp_path / codex_picker.CATALOG_FILE).exists()
    assert codex_picker.disable(home=tmp_path) == {"removed": [], "kept_user_files": [codex_picker.PROFILE_FILE]}
    assert mine.read_text() == 'model = "my-own"\n'


@pytest.mark.parametrize("ids, default", [([], "vl"), (["vl"], "other"),
                                          ([f"m{i}" for i in range(9)], "m0")])
def test_enable_rejects_bad_model_lists(tmp_path, ids, default):
    with pytest.raises(ValueError):
        codex_picker.enable([model(i) for i in ids], default, 8093, "BASE", home=tmp_path)
    assert not list(tmp_path.iterdir())


def test_instruction_capture_uses_isolated_home_and_loopback(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"], seen["env"] = argv, kwargs["env"]
        url = next(a for a in argv if "base_url=" in a).split('"')[1]
        host, port = url.split("//")[1].split("/")[0].split(":")
        assert host == "127.0.0.1"
        conn = http.client.HTTPConnection(host, int(port), timeout=5)
        conn.request("POST", "/v1/responses", json.dumps({"instructions": "CODEX DEFAULT"}),
                     {"Content-Type": "application/json"})
        assert conn.getresponse().status == 400
        conn.close()
        return subprocess.CompletedProcess(argv, 1, "", "")
    monkeypatch.setattr(codex_picker.subprocess, "run", run)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    assert codex_picker.capture_base_instructions("codex") == "CODEX DEFAULT"
    assert "--ignore-user-config" in seen["argv"] and "--ephemeral" in seen["argv"]
    assert "OPENAI_API_KEY" not in seen["env"]
    assert not Path(seen["env"]["CODEX_HOME"]).exists()


def test_instruction_capture_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(codex_picker.subprocess, "run",
                        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, "", ""))
    with pytest.raises(ValueError, match="capture failed"):
        codex_picker.capture_base_instructions("codex")


def fake_runtime(tmp_path, models, events, switch_status=None):
    from services import ServiceStatus
    by_id = {m.id: m for m in models}
    registry = SimpleNamespace(get=lambda i: by_id.get(i), launchable=lambda: list(models))
    state = {"running": None, "port": 8080}

    def start(model_id):
        state["running"] = model_id
        return ServiceStatus.RUNNING

    def switch(model_id):
        events.append(("switch", model_id))
        state["running"], state["port"] = model_id, state["port"] + 1
        return switch_status or ServiceStatus.RUNNING
    controller = SimpleNamespace(start=start, switch=switch)
    settings = SimpleNamespace(ports=SimpleNamespace(router=8099), data_dir=str(tmp_path))
    runner = SimpleNamespace(assert_can_run=lambda: None,
                             _post_completion=lambda *args: events.append(("verify", args[0])) or {})
    manager = SimpleNamespace(stop_all=lambda: events.append(("stop_all",)))
    return settings, registry, manager, controller, runner, None, state


def patch_helper(monkeypatch, helper, runtime, captured):
    settings, registry, manager, controller, runner, log, state = runtime

    class Controller:
        start = staticmethod(controller.start)
        switch = staticmethod(controller.switch)

        @property
        def running_model_id(self):
            return state["running"]

        @property
        def running_port(self):
            return state["port"]
    runtime6 = (settings, registry, manager, Controller(), runner, log)
    monkeypatch.setattr(helper, "_runtime", lambda: runtime6)
    monkeypatch.setattr("local_tuning.resolve_log_dir", lambda _: Path("."))
    monkeypatch.setattr("local_tuning.RunLock", lambda _: SimpleNamespace(
        __enter__=lambda: None, __exit__=lambda: None))
    monkeypatch.setattr("local_tuning.parse_completion_timings",
                        lambda _: SimpleNamespace(predicted_per_second=50))
    monkeypatch.setattr("local_tuning.Launcher._confirm_clean", lambda *args: True)

    def router_factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(start=lambda: None, stop=lambda: None,
                               base_url=lambda: "http://127.0.0.1:8099/v1")
    monkeypatch.setattr("local_tuning.ModelRouter", router_factory)
    return runtime6


def test_picker_models_switch_one_at_a_time_and_others_are_refused(monkeypatch, tmp_path):
    events, captured = [], {}
    models = [model("vl", mmproj="p"), model("text"), model("outsider")]
    helper = LocalTuning()
    patch_helper(monkeypatch, helper, fake_runtime(tmp_path, models, events), captured)
    (tmp_path / "codex_picker.json").write_text(json.dumps({"models": ["vl", "text"], "default_model_id": "vl"}))
    started = helper.start_local_model("vl")
    assert started["codex_picker_models"] == ["vl", "text"]
    assert [m.id for m in captured["registry_fn"]()] == ["vl", "text"]
    assert captured["switch_fn"]("outsider") == (False, "not in the Codex picker list")
    assert captured["switch_fn"]("text") == (True, "switched")
    assert events[-2:] == [("switch", "text"), ("verify", "http://127.0.0.1:8081/completion")]
    assert helper.stop_local_model()["cleanup_confirmed"]
    assert captured["switch_fn"]("vl") == (False, "helper model is stopped")


def test_model_outside_picker_keeps_single_model_no_switch(monkeypatch, tmp_path):
    events, captured = [], {}
    helper = LocalTuning()
    patch_helper(monkeypatch, helper, fake_runtime(tmp_path, [model("vl"), model("text")], events), captured)
    (tmp_path / "codex_picker.json").write_text(json.dumps({"models": ["text"]}))
    started = helper.start_local_model("vl")
    assert started["codex_picker_models"] == []
    assert [m.id for m in captured["registry_fn"]()] == ["vl"]
    assert captured["switch_fn"]("text")[0] is False
    assert not [e for e in events if e[0] == "switch"]
    helper.stop_local_model()


def test_failed_switch_is_reported_not_verified(monkeypatch, tmp_path):
    from services import ServiceStatus
    events, captured = [], {}
    helper = LocalTuning()
    patch_helper(monkeypatch, helper, fake_runtime(tmp_path, [model("vl"), model("text")], events,
                                                   switch_status=ServiceStatus.STOPPED_ERROR), captured)
    (tmp_path / "codex_picker.json").write_text(json.dumps({"models": ["vl", "text"]}))
    helper.start_local_model("vl")
    assert captured["switch_fn"]("text") == (False, "model did not become ready")
    assert not [e for e in events if e[0] == "verify" and "8081" in e[1]]
    helper.stop_local_model()


def test_enable_requires_tuned_context_and_writes_picker_state(monkeypatch, tmp_path):
    events = []
    helper = LocalTuning()
    runtime = patch_helper(monkeypatch, helper,
                           fake_runtime(tmp_path, [model("vl", context=32768), model("untuned", context=16384)], events), {})
    written = {}
    monkeypatch.setattr(codex_picker, "capture_base_instructions", lambda: "BASE")
    monkeypatch.setattr(codex_picker, "enable", lambda models, default, port, instructions:
                        written.update(ids=[m.id for m in models], default=default, port=port) or {"models": []})
    with pytest.raises(ValueError, match="tune the model first"):
        helper.enable_codex_picker(["vl", "untuned"])
    assert not (tmp_path / "codex_picker.json").exists()
    with pytest.raises(ValueError, match="unknown or unavailable"):
        helper.enable_codex_picker(["vl", "missing"])
    with pytest.raises(ValueError, match="invalid Codex picker"):
        helper.enable_codex_picker(["vl"], "untuned")
    helper.enable_codex_picker(["vl"])
    assert written == {"ids": ["vl"], "default": "vl", "port": runtime[0].ports.router}
    assert json.loads((tmp_path / "codex_picker.json").read_text()) == {"models": ["vl"], "default_model_id": "vl"}
