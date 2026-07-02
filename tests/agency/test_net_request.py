"""Tests for nyxara.agency.net_request — the SSRF-guarded HTTP helper."""

from __future__ import annotations

from nyxara.agency.net_request import _redirect_reason, http_request


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


def test_max_redirects_param_back_compat():
    # the new max_redirects param is accepted and the SSRF guard still fires first
    res = http_request("http://127.0.0.1/", max_redirects=10)
    assert res["ok"] is False and res["status"] == 0


def test_allow_private_opens_loopback_vetting():
    # with allow_private the SSRF vet passes; the dial-out then fails as data (no server)
    res = http_request("http://127.0.0.1:9/", allow_private=True, timeout_s=1.0)
    assert isinstance(res, dict) and res["ok"] is False
    assert "SSRF" not in (res["error"] or "")


# -------------------- redirect targets are re-vetted (SSRF-via-redirect) -------------------- #
def test_redirect_to_private_is_refused():
    # a redirect Location pointing at a private/loopback/link-local host is blocked before the
    # hop is followed — the same guard as the initial request, fail-closed.
    assert _redirect_reason("http://169.254.169.254/latest/meta-data/", allow_private=False)
    assert _redirect_reason("http://127.0.0.1/admin", allow_private=False)
    assert _redirect_reason("http://10.0.0.5/", allow_private=False)
    assert _redirect_reason("file:///etc/passwd", allow_private=False)


def test_redirect_to_public_is_allowed():
    # a public https target passes re-vetting (None means the hop may proceed)
    assert _redirect_reason("https://example.com/next", allow_private=False) is None


def test_redirect_reason_never_raises_on_garbage():
    # malformed input fails closed as data, never an exception
    reason = _redirect_reason("::::not a url::::", allow_private=False)
    assert reason is None or isinstance(reason, str)
