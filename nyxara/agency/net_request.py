"""NYXARA · agency/net_request.py — guarded generic HTTP request helper (🌐, fail-closed).

:mod:`senses.web` is the *perception* doorway — it fetches, cleans and injection-screens
*pages*. But an agent sometimes needs to make a **plain HTTP call** to an API: an
arbitrary method, custom headers, a request body, the raw response. This module is that
helper, and it inherits the same discipline: every request is first vetted by the very
same **SSRF guard** that protects :class:`~nyxara.senses.web.WebFetcher`
(``_vet_url``) — loopback, private, link-local and non-http(s) targets are refused
**before any socket is opened** (fail-closed).

:func:`http_request` performs the call with the standard library
(:mod:`urllib.request`), supporting method, headers and body, following at most a couple
of redirects, decoding text best-effort, capping the body and **never raising** — every
failure (a blocked URL, a refused connection, an HTTP error) comes back as a uniform
dict ``{ok, status, headers, body, error}``.

This module performs the SSRF/network containment, but **no permission checking** of its
own — the governing :class:`~nyxara.agency.tools.ToolRegistry` wraps it behind the
:class:`~nyxara.agency.permissions.Capability.NET_OUT` gate.

Pure standard library; reuses :class:`~nyxara.senses.web.WebFetcher` for SSRF vetting.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from nyxara.senses.web import WebFetcher

__all__ = ["http_request"]


def _redirect_reason(newurl: str, *, allow_private: bool) -> Optional[str]:
    """Return a rejection reason if a redirect target must not be followed, else None.

    A redirect hop is vetted by the very same SSRF guard as the initial request
    (:meth:`~nyxara.senses.web.WebFetcher._vet_url`), so a public URL cannot bounce the call
    onto a loopback/private/link-local host (e.g. the cloud-metadata service). Never raises —
    a vetting failure is itself treated as a reason to refuse (fail-closed)."""
    try:
        return WebFetcher(allow_private=allow_private)._vet_url(newurl)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 — fail-closed on any vetting error
        return f"redirect vetting failed: {exc}"


# A bounded redirect handler: urllib follows redirects by default, but we cap the count
# and — critically — re-vetting happens implicitly because we set a low max via the
# opener below. We disallow protocol downgrades to non-http(s) by relying on urllib's
# own scheme handlers (only http/https handlers are installed).
class _BoundedRedirect(urllib.request.HTTPRedirectHandler):
    """Follow no more than a couple of redirects (defence against redirect loops/chains)."""
    max_repeats = 2
    max_redirections = 2


def http_request(url: str, *, method: str = "GET", headers: Optional[Dict[str, str]] = None,
                 body: Optional[str] = None, timeout_s: float = 15.0,
                 max_bytes: int = 1_000_000, allow_private: bool = False,
                 max_redirects: int = 2) -> Dict[str, Any]:
    """Make a guarded generic HTTP request and return the outcome as a plain dict.

    The URL is first vetted by :meth:`WebFetcher._vet_url` — the same SSRF guard that
    protects web perception. If it returns a rejection reason, the request is **refused
    fail-closed**: no socket is opened and ``{ok: False, error: <reason>, status: 0}`` is
    returned. Only ``http``/``https`` schemes pass; loopback/private/link-local hosts are
    refused unless ``allow_private=True``.

    Otherwise the call is performed with :mod:`urllib.request`, honouring ``method``,
    ``headers`` and ``body`` (UTF-8 encoded), following at most two redirects, decoding
    the response text best-effort and truncating it to ``max_bytes``. The function never
    raises: HTTP errors (4xx/5xx), connection failures and timeouts all come back as
    data.

    Returns ``{ok, status, headers, body, error}``.
    """
    # 1. fail-closed SSRF guard — reuse the exact web-perception vetting, no socket yet
    reason = WebFetcher(allow_private=allow_private)._vet_url(url)
    if reason:
        return {"ok": False, "status": 0, "headers": {}, "body": "", "error": reason}

    # 2. build the request
    data: Optional[bytes] = None
    if body is not None:
        data = body.encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method.upper())
    req.add_header("User-Agent", "NYXARA/1.0")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    # Cap redirects per call (defence against redirect loops). Defaults to the conservative
    # _BoundedRedirect (2); callers may raise it via max_redirects. Critically, each redirect
    # target is RE-VETTED by the same SSRF guard before it is followed — otherwise a public URL
    # that 302-redirects to a private/loopback/link-local host (e.g. the cloud-metadata service
    # at 169.254.169.254) would be dialled fail-open. A blocked hop raises an HTTPError which the
    # call's own handler below turns into uniform error data.
    class _PerCallRedirect(urllib.request.HTTPRedirectHandler):
        max_repeats = max(0, int(max_redirects))
        max_redirections = max(0, int(max_redirects))

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            bad = _redirect_reason(newurl, allow_private=allow_private)
            if bad:
                raise urllib.error.HTTPError(newurl, code, f"blocked redirect: {bad}",
                                             headers, fp)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_PerCallRedirect)

    # 3. perform the call — every failure becomes data, never an exception
    try:
        with opener.open(req, timeout=timeout_s) as resp:  # noqa: S310 — vetted above
            raw = resp.read(max_bytes + 1)
            status = getattr(resp, "status", None) or resp.getcode() or 0
            resp_headers = {k: v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        # an HTTP error response is still a *response* — surface it as data
        try:
            raw = exc.read(max_bytes + 1)
        except Exception:
            raw = b""
        resp_headers = {k: v for k, v in (exc.headers.items() if exc.headers else [])}
        text, truncated = _decode(raw, max_bytes)
        return {"ok": False, "status": exc.code, "headers": resp_headers,
                "body": text, "error": f"HTTP {exc.code}: {exc.reason}",
                "truncated": truncated}
    except urllib.error.URLError as exc:
        return {"ok": False, "status": 0, "headers": {}, "body": "",
                "error": f"connection error: {exc.reason}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": 0, "headers": {}, "body": "",
                "error": f"{type(exc).__name__}: {exc}"}

    text, truncated = _decode(raw, max_bytes)
    return {"ok": 200 <= int(status) < 400, "status": int(status),
            "headers": resp_headers, "body": text, "error": None,
            "truncated": truncated}


def _decode(raw: bytes, max_bytes: int) -> tuple[str, bool]:
    """Decode response bytes best-effort and report whether truncation occurred."""
    truncated = len(raw) > max_bytes
    return raw[:max_bytes].decode("utf-8", "replace"), truncated


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA net-request self-test (no real network)")
    print("=" * 70)

    # 1. SSRF guard refuses loopback — fail-closed, no socket opened
    r = http_request("http://127.0.0.1/admin")
    print(f"\nloopback IP         : ok={r['ok']} status={r['status']} error={r['error']!r}")
    assert not r["ok"] and r["status"] == 0 and "SSRF" in (r["error"] or "")

    # 2. SSRF guard refuses 'localhost' by name
    r = http_request("http://localhost/")
    print(f"localhost name      : ok={r['ok']} error={r['error']!r}")
    assert not r["ok"] and "loopback" in (r["error"] or "").lower()

    # 3. non-http(s) scheme is refused (no file:// exfiltration)
    r = http_request("file:///etc/passwd")
    print(f"file:// scheme      : ok={r['ok']} error={r['error']!r}")
    assert not r["ok"] and "scheme" in (r["error"] or "")

    # 4. a private RFC1918 address is refused
    r = http_request("http://10.0.0.5/secret")
    print(f"private 10.x        : ok={r['ok']} error={r['error']!r}")
    assert not r["ok"] and "SSRF" in (r["error"] or "")

    # 5. a missing host is refused
    r = http_request("http:///nohost")
    print(f"missing host        : ok={r['ok']} error={r['error']!r}")
    assert not r["ok"]

    # 6. allow_private opens the gate for loopback (vetting passes; we don't dial out here)
    reason = WebFetcher(allow_private=True)._vet_url("http://127.0.0.1/")
    print(f"\nallow_private vet   : reason={reason!r} (None means it would proceed)")
    assert reason is None

    print("\nALL SELF-TESTS PASSED ✓")
