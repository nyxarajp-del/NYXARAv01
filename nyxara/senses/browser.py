"""NYXARA · senses/browser.py — a real headless browser: render JS pages & act on the web (🖱).

:mod:`senses.web` is the *static* doorway — it fetches raw HTML and cleans it to text. But
much of the live web is **JavaScript-rendered**, and *acting* on it (clicking, filling a form,
submitting) needs a real browser, not an HTTP GET. This module is that faculty: a headless
Chromium driven by Playwright that NYXARA uses **herself** — the autonomous researcher reaches
for it when a page returns little static text, and the ``browse_render`` / ``browse_actions``
tools expose it to the sovereign loop under the same capability gate as every other reach.

It inherits the web layer's discipline:

* **SSRF, fail-closed** — every navigation target (and the URL of each ``goto`` action) is
  vetted by the very same guard as :class:`~nyxara.senses.web.WebFetcher` (``_vet_url``):
  loopback / private / link-local / non-http(s) targets are refused **before the browser
  navigates**, unless ``allow_private`` is set.
* **Injection screening** — rendered text is scanned by the same
  :class:`~nyxara.senses.web.InjectionScanner` and returned fenced/untrusted, exactly like a
  fetched page.
* **Errors-as-data, honest degradation** — nothing here ever raises: a blocked URL, a missing
  ``playwright`` package, a navigation timeout or a failed action all come back as a uniform
  dict with ``ok: False`` and a human ``error``/``note``. With no browser engine installed the
  faculty returns ``{ok: False, note: "browser engine unavailable — pip install playwright"}``,
  the same graceful pattern the vision/audio/generative faculties use.

The heavy ``playwright`` dependency is **import-guarded**, so importing this module (and the
whole package) never requires it; it becomes real the moment the package is installed. The
Chromium binary is located by Playwright's own resolution (``PLAYWRIGHT_BROWSERS_PATH``).
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.senses.web import DEFAULT_USER_AGENT, InjectionScanner, WebFetcher

__all__ = ["Browser", "BrowserResult", "browser_available"]

_UNAVAILABLE_NOTE = "browser engine unavailable — pip install playwright"


def browser_available() -> bool:
    """Whether the Playwright engine can be imported (the browser is real, not a note)."""
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _discover_chromium() -> Optional[str]:
    """Locate a pre-installed Chromium binary when Playwright's own revision is absent.

    Managed/CI environments often ship the browser under ``$PLAYWRIGHT_BROWSERS_PATH`` at a
    revision that differs from the pip package's expectation, so a plain ``launch()`` prints
    "run playwright install". We fall back to whatever ``chrome``/``headless_shell`` is present
    there (or a couple of common paths) so the browser is real without a download.
    """
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), "/opt/pw-browsers",
             os.path.expanduser("~/.cache/ms-playwright")]
    patterns = ("chromium-*/chrome-linux/chrome", "chromium_headless_shell-*/chrome-linux/headless_shell",
                "chromium-*/chrome-linux/headless_shell")
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        # honour a stable symlink first (e.g. /opt/pw-browsers/chromium)
        link = os.path.join(root, "chromium")
        if os.path.exists(link):
            return link
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(root, pat)))
            if hits:
                return hits[-1]
    return None


def _env_proxy() -> Optional[str]:
    """The outbound HTTP(S) proxy from the environment, if any.

    Chromium — unlike httpx/urllib — does not honour ``HTTPS_PROXY`` automatically, so in a
    proxied/egress-controlled environment a browser navigation fails with ``ERR_CONNECTION_RESET``
    unless the proxy is passed at launch. We surface it here so the browser reaches the live web
    wherever the rest of NYXARA's HTTP stack already does.
    """
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.environ.get(var)
        if val:
            return val
    return None


@dataclass
class BrowserResult:
    """The outcome of a render or an action script — uniform, JSON-safe, never an exception."""
    url: str
    ok: bool
    status: int = 0
    title: str = ""
    text: str = ""
    html: str = ""
    final_url: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    screenshot: str = ""
    error: Optional[str] = None
    note: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"url": self.url, "ok": self.ok, "status": self.status, "title": self.title,
                "text": self.text, "final_url": self.final_url or self.url,
                "steps": self.steps, "screenshot": self.screenshot, "error": self.error,
                "note": self.note, "elapsed_ms": round(self.elapsed_ms, 1)}


class Browser:
    """A governed headless Chromium: render JS pages and take real actions on the web."""

    def __init__(self, *, allow_private: bool = False, timeout_s: float = 45.0,
                 max_bytes: int = 5_000_000, user_agent: str = DEFAULT_USER_AGENT,
                 governor: Any = None, executable_path: Optional[str] = None,
                 proxy: Optional[str] = None) -> None:
        self.allow_private = allow_private
        self.timeout_s = max(1.0, float(timeout_s))
        self.max_bytes = max(1024, int(max_bytes))
        self.user_agent = user_agent
        self.governor = governor
        # explicit path wins; otherwise auto-discover a pre-installed Chromium so the engine is
        # real even when the pip package's own browser revision was never downloaded.
        self.executable_path = executable_path or _discover_chromium()
        # proxy defaults to the environment's egress proxy so the browser reaches the web
        # wherever httpx/urllib already do (Chromium ignores HTTPS_PROXY on its own).
        self.proxy = proxy if proxy is not None else _env_proxy()
        self._scanner = InjectionScanner()
        # reuse the exact SSRF vetting of the static doorway (no sockets opened by vetting).
        self._vetter = WebFetcher(allow_private=allow_private)

    # ---- guards ---- #
    def _vet(self, url: str) -> Optional[str]:
        return self._vetter._vet_url(url)  # noqa: SLF001 — deliberate reuse of the SSRF guard

    def _throttle(self) -> Optional[str]:
        if self.governor is None:
            return None
        try:
            dec = self.governor.allow("web")
            return None if dec.allowed else "rate limited"
        except Exception:  # noqa: BLE001 — a governor error never blocks fail-open here
            return None

    # ---- engine ---- #
    def _launch(self, p: Any) -> Any:
        """Launch headless Chromium: discovered/explicit binary + env proxy, resilient fallback."""
        kw: Dict[str, Any] = {"headless": True,
                              "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        if self.proxy:
            kw["proxy"] = {"server": self.proxy}
        if self.executable_path:
            kw["executable_path"] = self.executable_path
        try:
            return p.chromium.launch(**kw)
        except Exception:
            # last resort: let Playwright resolve its own bundled browser (if `playwright
            # install` was run) — drop only the explicit binary, keep the proxy.
            kw.pop("executable_path", None)
            return p.chromium.launch(**kw)

    def _screen(self, text: str) -> str:
        """Injection-screen and fence rendered text (data, never instructions) — as web_fetch does."""
        return self._scanner.sanitize(text[:self.max_bytes])

    # ---- render ---- #
    def render(self, url: str) -> BrowserResult:
        """Navigate to ``url`` in a real browser and return the JS-rendered, screened text."""
        return self._drive(url, actions=None)

    # ---- act ---- #
    def act(self, url: str, actions: List[Dict[str, Any]]) -> BrowserResult:
        """Open ``url`` and run a typed action script (click/fill/submit/…), returning the result.

        Each action is a dict ``{"op": ..., ...}``:
        ``goto {url}`` · ``click {selector}`` · ``fill {selector, value}`` ·
        ``press {selector?, key}`` · ``wait {ms}`` or ``wait {selector}`` ·
        ``screenshot {path}`` · ``submit {selector?}``. A ``goto`` target is SSRF-vetted like
        the initial URL. Never raises — a failed step is recorded and stops the script.
        """
        return self._drive(url, actions=list(actions or []))

    # ---- the one code path both use ---- #
    def _drive(self, url: str, actions: Optional[List[Dict[str, Any]]]) -> BrowserResult:
        start = time.monotonic()
        res = BrowserResult(url=url, ok=False)

        reason = self._vet(url)
        if reason:
            res.error = reason
            res.elapsed_ms = (time.monotonic() - start) * 1000
            return res
        throttled = self._throttle()
        if throttled:
            res.error = throttled
            res.note = throttled
            res.elapsed_ms = (time.monotonic() - start) * 1000
            return res

        try:
            from playwright.sync_api import sync_playwright  # noqa: PLC0415
        except Exception:  # noqa: BLE001 — no engine: honest note, never a crash
            res.note = _UNAVAILABLE_NOTE
            res.error = _UNAVAILABLE_NOTE
            res.elapsed_ms = (time.monotonic() - start) * 1000
            return res

        timeout_ms = int(self.timeout_s * 1000)
        try:
            with sync_playwright() as p:
                browser = self._launch(p)
                try:
                    context = browser.new_context(user_agent=self.user_agent)
                    page = context.new_page()
                    page.set_default_timeout(timeout_ms)
                    resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    res.status = int(getattr(resp, "status", 0) or 0)
                    try:
                        page.wait_for_load_state("networkidle", timeout=timeout_ms)
                    except Exception:  # noqa: BLE001 — networkidle is best-effort
                        pass
                    if actions:
                        res.steps, ok = self._run_actions(page, actions, timeout_ms)
                        if not ok:
                            res.final_url = page.url
                            res.title = page.title() or ""
                            res.error = next((s.get("error") for s in reversed(res.steps)
                                              if s.get("error")), "action failed")
                            res.elapsed_ms = (time.monotonic() - start) * 1000
                            return res
                    res.final_url = page.url
                    res.title = page.title() or ""
                    body_text = page.inner_text("body") if page.query_selector("body") else \
                        page.content()
                    res.text = self._screen(body_text or "")
                    res.html = ""  # kept out of the model budget; text is the useful surface
                    res.ok = True
                finally:
                    browser.close()
        except Exception as exc:  # noqa: BLE001 — every browser failure is data
            res.error = f"{type(exc).__name__}: {exc}"
        res.elapsed_ms = (time.monotonic() - start) * 1000
        return res

    def _run_actions(self, page: Any, actions: List[Dict[str, Any]],
                     timeout_ms: int) -> Tuple[List[Dict[str, Any]], bool]:
        """Execute the action script step by step; stop and report on the first failure."""
        steps: List[Dict[str, Any]] = []
        for raw in actions:
            op = str(raw.get("op", "")).lower().strip()
            step: Dict[str, Any] = {"op": op}
            try:
                if op == "goto":
                    target = str(raw.get("url", ""))
                    bad = self._vet(target)
                    if bad:
                        step["error"] = bad
                        steps.append(step)
                        return steps, False
                    page.goto(target, wait_until="domcontentloaded", timeout=timeout_ms)
                elif op == "click":
                    page.click(str(raw["selector"]), timeout=timeout_ms)
                elif op == "fill":
                    page.fill(str(raw["selector"]), str(raw.get("value", "")), timeout=timeout_ms)
                elif op == "press":
                    key = str(raw.get("key", "Enter"))
                    sel = raw.get("selector")
                    if sel:
                        page.press(str(sel), key, timeout=timeout_ms)
                    else:
                        page.keyboard.press(key)
                elif op == "submit":
                    sel = raw.get("selector")
                    if sel:
                        page.press(str(sel), "Enter", timeout=timeout_ms)
                    else:
                        page.keyboard.press("Enter")
                    try:
                        page.wait_for_load_state("networkidle", timeout=timeout_ms)
                    except Exception:  # noqa: BLE001
                        pass
                elif op == "wait":
                    if "selector" in raw and raw["selector"]:
                        page.wait_for_selector(str(raw["selector"]), timeout=timeout_ms)
                    else:
                        page.wait_for_timeout(min(int(raw.get("ms", 500)), timeout_ms))
                elif op == "screenshot":
                    path = str(raw.get("path", ""))
                    if path:
                        page.screenshot(path=path, full_page=bool(raw.get("full_page", False)))
                        step["path"] = path
                else:
                    step["error"] = f"unknown action {op!r}"
                    steps.append(step)
                    return steps, False
                step["ok"] = True
                steps.append(step)
            except Exception as exc:  # noqa: BLE001 — a failed step is data; stop the script
                step["ok"] = False
                step["error"] = f"{type(exc).__name__}: {exc}"
                steps.append(step)
                return steps, False
        return steps, True


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA headless-browser self-test")
    print("=" * 70)

    b = Browser()
    print(f"\nengine available    : {browser_available()}")

    # SSRF guard refuses loopback/private BEFORE any browser launch (fail-closed)
    blocked = b.render("http://127.0.0.1/admin")
    print(f"SSRF guard          : ok={blocked.ok} error={blocked.error!r}")
    assert not blocked.ok and "SSRF" in (blocked.error or "")

    bad_scheme = b.render("file:///etc/passwd")
    print(f"scheme guard        : ok={bad_scheme.ok} error={bad_scheme.error!r}")
    assert not bad_scheme.ok and "scheme" in (bad_scheme.error or "")

    if browser_available():
        r = b.render("https://example.com")
        print(f"render example.com  : ok={r.ok} title={r.title!r} words={len(r.text.split())}")
        assert r.ok and "Example Domain" in r.title
    else:
        # honest degradation: a public URL returns the note, never a crash
        r = b.render("https://example.com")
        print(f"no-engine note      : ok={r.ok} note={r.note!r}")
        assert not r.ok and r.note == _UNAVAILABLE_NOTE

    print("\nALL SELF-TESTS PASSED ✓")
