"""Tests for nyxara.senses.browser — the headless browser faculty.

Runs fully offline: the Playwright engine may be absent, so these tests exercise the
SSRF/scheme guards (which fire before any launch) and the honest-degradation path, never
requiring a real browser.
"""

from __future__ import annotations

from nyxara.senses.browser import Browser, BrowserResult, browser_available


# --------------------------------------------------------------------------- #
# SSRF / scheme guards fire BEFORE any browser launch (fail-closed)
# --------------------------------------------------------------------------- #
def test_render_refuses_loopback():
    r = Browser().render("http://127.0.0.1/admin")
    assert not r.ok and "SSRF" in (r.error or "")


def test_render_refuses_private_ip():
    r = Browser().render("http://10.0.0.5/secret")
    assert not r.ok and "SSRF" in (r.error or "")


def test_render_refuses_non_http_scheme():
    r = Browser().render("file:///etc/passwd")
    assert not r.ok and "scheme" in (r.error or "")


def test_allow_private_passes_vetting_for_loopback():
    # with allow_private the SSRF vet passes; without an engine it then returns the honest note
    r = Browser(allow_private=True).render("http://127.0.0.1/")
    assert "SSRF" not in (r.error or "")


# --------------------------------------------------------------------------- #
# Honest degradation when the engine is absent (never a crash)
# --------------------------------------------------------------------------- #
def test_render_public_url_is_data_not_exception():
    r = Browser().render("https://example.com")
    assert isinstance(r, BrowserResult)
    if not browser_available():
        assert not r.ok and r.note == "browser engine unavailable — pip install playwright"


def test_act_public_url_is_data_not_exception():
    r = Browser().act("https://example.com",
                      [{"op": "fill", "selector": "#q", "value": "hi"},
                       {"op": "click", "selector": "#go"}])
    assert isinstance(r, BrowserResult)
    if not browser_available():
        assert not r.ok and r.note == "browser engine unavailable — pip install playwright"


def test_act_goto_target_is_ssrf_vetted():
    # a goto action pointing at a private host is refused by the same guard as the initial URL
    r = Browser().act("http://127.0.0.1/", [{"op": "goto", "url": "http://127.0.0.1/x"}])
    assert not r.ok and "SSRF" in (r.error or "")


# --------------------------------------------------------------------------- #
# Result shape
# --------------------------------------------------------------------------- #
def test_result_to_dict_shape():
    d = Browser().render("http://127.0.0.1/").to_dict()
    for key in ("url", "ok", "status", "title", "text", "final_url", "steps", "error"):
        assert key in d
