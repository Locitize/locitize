"""Optional SDK protocol boundary tests; core suite can run without MCP extras."""
import pytest

import local_tuning_mcp


def test_tools_sanitize_exception_paths():
    def fail():
        raise OSError("private path and secret")
    response = local_tuning_mcp._call(fail)
    assert response["ok"] is False
    assert "secret" not in response["error"]


def test_invalid_model_path_rejected():
    response = local_tuning_mcp.start_tuning("../../settings.yaml", "invalid")
    assert response["ok"] is False


def test_stop_never_stops_external_model():
    response = local_tuning_mcp.stop_local_model()
    assert response == {"ok": True, "result": {"state": "stopped", "cleanup_confirmed": True}}


def test_tool_annotations_are_explicit_when_sdk_available():
    pytest.importorskip("mcp")
    tools = {tool.name: tool for tool in local_tuning_mcp.build_server()._tool_manager.list_tools()}
    assert set(tools) == {
        "assess_computer", "list_local_models", "start_tuning", "tuning_status",
        "cancel_tuning", "start_local_chat", "start_local_model", "stop_local_model",
        "search_models", "list_model_files", "recommend_model_candidates", "start_model_download",
        "download_status", "cancel_download", "resume_download", "register_download",
    }
    for name, tool in tools.items():
        hints = tool.annotations
        assert hints is not None
        assert type(hints.readOnlyHint) is bool
        assert type(hints.destructiveHint) is bool
        assert hints.openWorldHint is (name in {"search_models", "list_model_files", "start_model_download", "resume_download"})
        if name in {"assess_computer", "list_local_models", "tuning_status", "search_models", "list_model_files", "recommend_model_candidates", "download_status"}:
            assert hints.readOnlyHint and not hints.destructiveHint
        else:
            assert not hints.readOnlyHint and hints.destructiveHint
