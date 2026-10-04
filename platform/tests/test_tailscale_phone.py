"""Tailscale Serve discovery for the Chat page phone URL."""

from types import SimpleNamespace

from tailscale_phone import (
    PhoneAccess,
    _from_serve_status,
    discover_phone_access,
    resolve_webui_serve_url,
)


def test_serve_status_picks_the_openwebui_loopback_handler():
    payload = {
        "TCP": {"443": {"HTTPS": True}},
        "Web": {
            "chat.example.ts.net:443": {
                "Handlers": {
                    "/": {"Proxy": "http://127.0.0.1:8096"},
                }
            }
        },
    }
    access = _from_serve_status(payload, openwebui_port=8096)
    assert access == PhoneAccess(
        url="https://chat.example.ts.net/",
        proxy_target="http://127.0.0.1:8096",
        hostname="chat.example.ts.net",
        tailnet_only=True,
    )


def test_serve_status_preserves_nonstandard_https_port():
    payload = {
        "Web": {
            "genesis.example.ts.net:443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:4200"}}
            },
            "genesis.example.ts.net:4443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:8096"}}
            },
            "genesis.example.ts.net:8443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:4200"}}
            },
        }
    }
    access = _from_serve_status(payload, openwebui_port=8096)
    assert access is not None
    assert access.url == "https://genesis.example.ts.net:4443/"
    assert access.proxy_target == "http://127.0.0.1:8096"


def test_serve_status_prefers_4443_when_several_owui_handlers_exist():
    payload = {
        "Web": {
            "genesis.example.ts.net:9443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:8096"}}
            },
            "genesis.example.ts.net:4443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:8096"}}
            },
        }
    }
    access = _from_serve_status(payload, openwebui_port=8096)
    assert access is not None
    assert access.url == "https://genesis.example.ts.net:4443/"


def test_serve_status_prefers_the_openwebui_port_when_several_handlers_exist():
    payload = {
        "Web": {
            "other.example.ts.net:443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:3000"}}
            },
            "chat.example.ts.net:443": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:8096"}}
            },
        }
    }
    access = _from_serve_status(payload, openwebui_port=8096)
    assert access is not None
    assert access.hostname == "chat.example.ts.net"


def test_missing_or_empty_serve_config_is_none():
    assert _from_serve_status({}) is None
    assert _from_serve_status({"Web": {}}) is None
    assert discover_phone_access(runner=lambda _args: "") is None
    assert discover_phone_access(runner=lambda _args: "not-json") is None


def test_discover_phone_access_uses_the_injected_runner():
    seen = []

    def runner(args):
        seen.append(args)
        return (
            '{"Web":{"chat.example.ts.net:443":'
            '{"Handlers":{"/":{"Proxy":"http://127.0.0.1:8096"}}}}}'
        )

    access = discover_phone_access(runner=runner, openwebui_port=8096)
    assert seen == [["serve", "status", "--json"]]
    assert access is not None
    assert access.url == "https://chat.example.ts.net/"


def test_resolve_webui_serve_url_uses_magicdns_root_no_port(monkeypatch):
    monkeypatch.delenv("LOCITIZE_WEBUI_URL", raising=False)

    def runner(args):
        if args[:1] == ["status"]:
            return '{"Self":{"DNSName":"chat.example.ts.net."}}'
        return ""

    assert (
        resolve_webui_serve_url(openwebui_port=8096, runner=runner)
        == "https://chat.example.ts.net"
    )


def test_resolve_webui_serve_url_falls_back_to_loopback(monkeypatch):
    monkeypatch.delenv("LOCITIZE_WEBUI_URL", raising=False)

    def runner(args):
        return ""  # no tailscale

    assert (
        resolve_webui_serve_url(openwebui_port=8096, runner=runner)
        == "http://127.0.0.1:8096"
    )


def test_resolve_webui_serve_url_env_override_wins(monkeypatch):
    monkeypatch.setenv("LOCITIZE_WEBUI_URL", "https://chat.example.ts.net/")

    def runner(args):
        raise AssertionError("override must not query tailscale")

    assert (
        resolve_webui_serve_url(openwebui_port=8096, runner=runner)
        == "https://chat.example.ts.net"
    )
