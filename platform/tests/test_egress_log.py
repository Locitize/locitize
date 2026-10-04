"""Tests for the privacy ledger (M17.1)."""
from __future__ import annotations

import egress_log


def test_disabled_when_unconfigured_never_raises():
    egress_log.configure(None)
    egress_log.record("https://example.com/x")  # no root: silent no-op


def test_records_host_and_reason(tmp_path):
    egress_log.configure(tmp_path)
    with egress_log.reason("hub-search"):
        egress_log.record("https://huggingface.co/api/models")
    s = egress_log.summarize(tmp_path)
    assert s.total == 1
    assert s.hosts == {"huggingface.co": 1}
    assert s.recent[0]["reason"] == "hub-search"


def test_reason_nests_and_restores(tmp_path):
    egress_log.configure(tmp_path)
    with egress_log.reason("outer"):
        with egress_log.reason("download"):
            egress_log.record("https://a.co/1")
        egress_log.record("https://b.co/2")
    reasons = [r["reason"] for r in egress_log.summarize(tmp_path).recent]
    assert reasons == ["download", "outer"]


def test_fresh_machine_reads_zero(tmp_path):
    egress_log.configure(tmp_path)
    s = egress_log.summarize(tmp_path)
    assert s.total == 0
    assert "nothing has left this machine" in egress_log.render_line(s)


def test_render_line_names_hosts(tmp_path):
    egress_log.configure(tmp_path)
    egress_log.record("https://huggingface.co/x")
    egress_log.record("https://github.com/y")
    line = egress_log.render_line(egress_log.summarize(tmp_path))
    assert "github.com" in line and "huggingface.co" in line


def test_malformed_url_is_recorded_not_raised(tmp_path):
    egress_log.configure(tmp_path)
    egress_log.record("not a url")  # must not raise
    assert egress_log.summarize(tmp_path).total == 1
