"""Tests for nyxara.agency.net_request — the SSRF-guarded HTTP helper."""

from __future__ import annotations

from nyxara.agency.net_request import http_request


def test_blocks_loopback():
    res = http_request("http://127.0.0.1/")
    assert res["ok"] is False and res["status"] == 0
    assert "SSRF" in res["error"] or "loopback" in res["error"] or "private" in res["error"]


def test_blocks_localhost():
    res = http_request("http://localhost/admin")
    assert res["ok"] is False and res["status"] == 0


def test_blocks_non_http_scheme():
    res = http_request("file:///etc/passwd")
    assert res["ok"] is False
    assert "scheme" in res["error"].lower() or res["status"] == 0


def test_blocks_private_range():
    res = http_request("http://10.0.0.1/")
    assert res["ok"] is False and res["status"] == 0


def test_returns_data_never_raises_on_bad_host():
    res = http_request("http://nonexistent.invalid.host.example/")
    assert isinstance(res, dict) and res["ok"] is False
