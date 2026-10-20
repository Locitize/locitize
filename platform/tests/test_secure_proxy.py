"""Ownership tests for the desktop-managed Caddy proxy."""

from types import SimpleNamespace

import secure_proxy


def _settings(tmp_path):
    return SimpleNamespace(
        data_dir=tmp_path,
        ports=SimpleNamespace(openwebui=3000),
        secure_proxy=SimpleNamespace(
            enabled=True,
            hostname="locitize.local",
            caddy_path=str(tmp_path / "caddy.exe"),
            caddyfile="",
            auto_install=False,
        ),
    )


def test_caddy_started_here_is_stopped_with_the_same_config(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    (tmp_path / "caddy.exe").write_bytes(b"MZ")
    calls = []
    probes = iter([False, True])
    monkeypatch.setattr(secure_proxy.sys, "platform", "win32")
    monkeypatch.setattr(secure_proxy, "_hostname_resolves_loopback", lambda _h: True)
    monkeypatch.setattr(secure_proxy, "_tls_trusted", lambda _h: True)
    monkeypatch.setattr(secure_proxy, "_port_open", lambda *_a, **_k: next(probes))
    monkeypatch.setattr(
        secure_proxy.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(list(argv)) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(secure_proxy, "_OWNED_CADDY", None)

    assert secure_proxy.ensure(settings)[0]
    secure_proxy.stop_owned()

    caddyfile = str(tmp_path / "Caddyfile")
    assert calls == [
        [str(tmp_path / "caddy.exe"), "start", "--config", caddyfile,
         "--adapter", "caddyfile"],
        [str(tmp_path / "caddy.exe"), "stop", "--config", caddyfile,
         "--adapter", "caddyfile"],
    ]
    assert secure_proxy._OWNED_CADDY is None


def test_existing_caddy_is_never_claimed_or_stopped(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    (tmp_path / "caddy.exe").write_bytes(b"MZ")
    calls = []
    monkeypatch.setattr(secure_proxy.sys, "platform", "win32")
    monkeypatch.setattr(secure_proxy, "_hostname_resolves_loopback", lambda _h: True)
    monkeypatch.setattr(secure_proxy, "_tls_trusted", lambda _h: True)
    monkeypatch.setattr(secure_proxy, "_port_open", lambda *_a, **_k: True)
    monkeypatch.setattr(
        secure_proxy.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(list(argv)) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(secure_proxy, "_OWNED_CADDY", None)

    assert secure_proxy.ensure(settings)[0]
    secure_proxy.stop_owned()

    assert calls == []
    assert secure_proxy._OWNED_CADDY is None
