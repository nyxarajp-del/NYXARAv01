"""Tests for nyxara.senses.search — real web search provider layer (no real network)."""

from __future__ import annotations

import json

from nyxara.agency.governor import Governor
from nyxara.kernel.config import ResourceLimits
from nyxara.senses.search import SearchResult, WebSearcher, _unwrap_ddg
from nyxara.senses.web import FetchResult, WebFetcher


# A minimal slice of an html.duckduckgo.com/html SERP: two results, wrapped hrefs.
SERP = """<html><body>
  <div class="result">
    <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa">First Result</a>
    <a class="result__snippet">The first snippet text.</a>
  </div>
  <div class="result">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb">Second Result</a>
    <a class="result__snippet">Second snippet here.</a>
  </div>
</body></html>"""


def _fetcher(body=SERP, ok=True, status=200, ct="text/html", **kw):
    def transport(url, timeout, max_bytes):
        return FetchResult(url=url, ok=ok, status=status, content_type=ct, body=body)
    return WebFetcher(transport=transport, cache=False, **kw)


# -------------------- DDG redirect unwrapping -------------------- #
def test_unwrap_ddg_redirect():
    assert _unwrap_ddg("/l/?uddg=https%3A%2F%2Fexample.com%2Fx") == "https://example.com/x"
    assert _unwrap_ddg("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.io%2Fy") == "https://a.io/y"


def test_unwrap_ddg_passthrough():
    assert _unwrap_ddg("https://plain.example/page") == "https://plain.example/page"


# -------------------- DDG HTML parsing -------------------- #
def test_ddg_html_parses_results():
    s = WebSearcher(provider="duckduckgo", fetcher=_fetcher())
    resp = s.search("anything")
    assert resp.provider == "duckduckgo"
    assert resp.error is None
    assert [r.url for r in resp.results] == ["https://example.com/a", "https://example.org/b"]
    assert resp.results[0].title == "First Result"
    assert resp.results[0].snippet == "The first snippet text."


def test_max_results_truncates():
    s = WebSearcher(provider="duckduckgo", fetcher=_fetcher())
    resp = s.search("anything", max_results=1)
    assert len(resp.results) == 1


def test_to_dict_has_text_alias():
    # back-compat: old web_search returned {"title","text","url"}; snippet mirrors text.
    d = SearchResult(title="t", url="u", snippet="snip").to_dict()
    assert d["snippet"] == "snip" and d["text"] == "snip" and d["url"] == "u"


# -------------------- errors as data -------------------- #
def test_search_errors_as_data():
    s = WebSearcher(provider="duckduckgo", fetcher=_fetcher(ok=False, status=503))
    resp = s.search("x")
    assert resp.results == [] and resp.error  # no exception raised


def test_empty_query():
    s = WebSearcher(fetcher=_fetcher())
    resp = s.search("   ")
    assert resp.results == [] and resp.error == "empty query"


# -------------------- provider routing -------------------- #
def test_provider_order_auto_prefers_key():
    keyed = WebSearcher(provider="auto", brave_key="k")
    assert keyed._provider_order()[0] == "brave"
    keyless = WebSearcher(provider="auto")
    assert keyless._provider_order()[:2] == ["duckduckgo", "wikipedia"]


def test_wikipedia_backend_parses_results():
    payload = {"query": {"search": [
        {"title": "Python (programming language)",
         "snippet": "<span class=\"m\">Python</span> is a language &amp; more"},
        {"title": "History of Python", "snippet": "the <b>history</b>"}]}}
    s = WebSearcher(provider="wikipedia", fetcher=_fetcher(body=json.dumps(payload)))
    resp = s.search("python")
    assert resp.provider == "wikipedia"
    assert resp.results[0].url == "https://en.wikipedia.org/wiki/Python_%28programming_language%29"
    # tags stripped, entities unescaped
    assert resp.results[0].snippet == "Python is a language & more"


def test_auto_falls_through_to_wikipedia(monkeypatch):
    # DDG HTML returns nothing (challenged) -> auto chain reaches Wikipedia.
    payload = {"query": {"search": [{"title": "Topic", "snippet": "snip"}]}}
    s = WebSearcher(provider="auto", fetcher=_fetcher(body=json.dumps(payload)))

    def empty_ddg(query, n):
        from nyxara.senses.search import SearchResponse
        return SearchResponse(query, "duckduckgo", results=[])

    monkeypatch.setattr(s, "_ddg_html", empty_ddg)
    resp = s.search("q")
    assert resp.provider == "wikipedia" and resp.results[0].title == "Topic"


def test_auto_falls_back_to_ddg_when_no_results(monkeypatch):
    # brave keyed but returns nothing -> auto chain falls through to DDG HTML.
    s = WebSearcher(provider="auto", brave_key="k", fetcher=_fetcher())

    def empty_brave(query, n):
        from nyxara.senses.search import SearchResponse
        return SearchResponse(query, "brave", results=[])

    monkeypatch.setattr(s, "_brave", empty_brave)
    resp = s.search("q")
    assert resp.provider == "duckduckgo" and len(resp.results) == 2


# -------------------- API providers (monkeypatched http_request) -------------------- #
def _patch_http(monkeypatch, payload):
    import nyxara.agency.net_request as nr

    def fake(url, **kw):
        return {"ok": True, "status": 200, "headers": {}, "body": json.dumps(payload),
                "error": None}
    monkeypatch.setattr(nr, "http_request", fake)


def test_brave_parse(monkeypatch):
    _patch_http(monkeypatch, {"web": {"results": [
        {"title": "B", "url": "https://b.io", "description": "desc"}]}})
    resp = WebSearcher(provider="brave", brave_key="k").search("q")
    assert resp.provider == "brave"
    assert resp.results[0].url == "https://b.io" and resp.results[0].snippet == "desc"


def test_tavily_parse(monkeypatch):
    _patch_http(monkeypatch, {"results": [
        {"title": "T", "url": "https://t.io", "content": "body"}]})
    resp = WebSearcher(provider="tavily", tavily_key="k").search("q")
    assert resp.results[0].url == "https://t.io" and resp.results[0].snippet == "body"


def test_serpapi_parse(monkeypatch):
    _patch_http(monkeypatch, {"organic_results": [
        {"title": "S", "link": "https://s.io", "snippet": "snip"}]})
    resp = WebSearcher(provider="serpapi", serpapi_key="k").search("q")
    assert resp.results[0].url == "https://s.io" and resp.results[0].snippet == "snip"


# -------------------- governor throttling reaches search -------------------- #
def test_governor_throttles_search():
    gov = Governor(ResourceLimits(max_web_fetches_per_min=1))
    s = WebSearcher(provider="duckduckgo", fetcher=_fetcher(governor=gov))
    first = s.search("a")
    assert first.results  # first fetch allowed
    second = s.search("b")
    assert second.results == [] and second.error  # rate limited -> error as data
