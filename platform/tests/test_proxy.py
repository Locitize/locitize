"""Headless tests for the reverse-proxy target logic (Architecture G4, G7).

These assert the proxy's target-following and bind-port decisions as PURE
functions - no socket is bound, so they are safe on any machine and never touch a
privileged port (AC13, keyword `proxy_target`). The streaming forward path itself
is owner-validated live (RG1); here we prove the decision logic that steers it.
"""

from __future__ import annotations

from config import ProxyConfig

import proxy


def test_proxy_target_follows_active_port():
    assert proxy.build_upstream_target(8080) == ("127.0.0.1", 8080)
    assert proxy.build_upstream_target(8085) == ("127.0.0.1", 8085)


def test_proxy_target_none_when_no_model_running():
    # No active model -> no target -> the handler answers a friendly 503.
    assert proxy.build_upstream_target(None) is None


def test_proxy_target_rejects_out_of_range_and_non_int():
    assert proxy.build_upstream_target(0) is None
    assert proxy.build_upstream_target(70000) is None
    assert proxy.build_upstream_target(True) is None  # bool is not a valid port


def test_proxy_target_upstream_host_is_always_loopback():
    # The host component can never be steered off 127.0.0.1 by any port value.
    host, _port = proxy.build_upstream_target(8090)
    assert host == "127.0.0.1"


def test_proxy_target_default_bind_port_is_loopback_8085():
    # Default config binds the non-privileged loopback port (no elevation).
    assert proxy.effective_bind_port(ProxyConfig()) == 8085


def test_proxy_target_bind_port_80_only_when_opted_in():
    opted_in = ProxyConfig(bind_port_80=True)
    assert proxy.effective_bind_port(opted_in) == 80
    default = ProxyConfig(port=9090)
    assert proxy.effective_bind_port(default) == 9090
