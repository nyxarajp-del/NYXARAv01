"""NYXARA · senses/search.py — real web search (🔎), keyless by default, pluggable.

:mod:`senses.web` fetches and screens a *single* page. This module is the other half of
true internet access: a **search engine doorway** that turns a query into a ranked list of
real results (title, URL, snippet) — the thing the old instant-answer-only ``web_search``
could not do (it returned empty for most queries).

Backends, in routing order:

* **DuckDuckGo HTML** (``html.duckduckgo.com/html``) — keyless full-web SERP. Result links
  are wrapped redirects (``/l/?uddg=<url-encoded-target>``) which :func:`_unwrap_ddg`
  unwraps. (Some datacenter egress IPs are anti-bot-challenged by DDG; the chain then falls
  through to the next backend.)
* **Brave / Tavily / SerpAPI** — used first when an API key is configured (durable,
  highest quality full-web). They go through
  :func:`~nyxara.agency.net_request.http_request` so the same SSRF discipline applies.
* **Wikipedia** (``en.wikipedia.org/w/api.php``) — the keyless backend that works from
  *anywhere* (including datacenters): a real, ranked, structured result set with no API key.
* **DuckDuckGo instant-answer** (``api.duckduckgo.com``) — the legacy abstract fallback,
  kept so behaviour never fully regresses.

Everything here is **errors-as-data**: a backend failure becomes a populated ``error`` on
the :class:`SearchResponse`, never an exception. The keyless backends reuse
:class:`~nyxara.senses.web.WebFetcher`, so SSRF vetting, the governor's ``web`` rate
bucket and the response cache all apply automatically.

Pure standard library (``urllib`` + ``html.parser``); no third-party dependency.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

__all__ = ["SearchResult", "SearchResponse", "WebSearcher"]


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class SearchResult:
    """A single search hit."""

    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> Dict[str, str]:
        # ``text`` mirrors ``snippet`` for backward compatibility with the old
        # instant-answer ``web_search`` shape ({title, text, url}); callers that read
        # either key keep working.
        return {"title": self.title, "url": self.url,
                "snippet": self.snippet, "text": self.snippet}


@dataclass
class SearchResponse:
    """The outcome of a search: results plus the provider that served them (or an error)."""

    query: str
    provider: str
    results: List[SearchResult] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "provider": self.provider,
                "error": self.error, "results": [r.to_dict() for r in self.results]}


# --------------------------------------------------------------------------- #
# DuckDuckGo HTML parsing
# --------------------------------------------------------------------------- #
def _strip_tags(text: str) -> str:
    """Strip HTML tags/entities from an API snippet (e.g. Wikipedia search highlights)."""
    import html as _html
    return " ".join(_html.unescape(re.sub(r"<[^>]+>", "", text or "")).split())


def _unwrap_ddg(href: str) -> str:
    """Unwrap a DuckDuckGo redirect link (``/l/?uddg=<encoded>``) to its real target.

    DDG HTML results route through ``/l/?uddg=...`` (sometimes scheme-relative,
    ``//duckduckgo.com/l/?uddg=...``). A plain URL is returned unchanged.
    """
    try:
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return qs["uddg"][0]
    except Exception:  # noqa: BLE001 — a malformed link just passes through
        pass
    return href


class _DDGHTMLParser(HTMLParser):
    """Extract (title, url, snippet) triples from an ``html.duckduckgo.com/html`` SERP.

    Result anchors carry the class ``result__a`` (title text + wrapped href); snippets
    carry ``result__snippet``. We collect titles/hrefs as anchors open/close and attach the
    following snippet to the most recent result.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[SearchResult] = []
        self._in_title = False
        self._cur_href = ""
        self._cur_title: List[str] = []
        self._in_snippet = False
        self._cur_snippet: List[str] = []

    def handle_starttag(self, tag: str, attrs: List) -> None:
        a = {k: (v or "") for k, v in attrs}
        cls = a.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._in_title = True
            self._cur_href = _unwrap_ddg(a.get("href", ""))
            self._cur_title = []
        elif "result__snippet" in cls:
            self._in_snippet = True
            self._cur_snippet = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == "a":
            self._in_title = False
            title = " ".join("".join(self._cur_title).split())
            if self._cur_href:
                self.results.append(SearchResult(title=title, url=self._cur_href))
        elif self._in_snippet and tag in ("a", "div", "td", "span"):
            self._in_snippet = False
            snippet = " ".join("".join(self._cur_snippet).split())
            if snippet and self.results and not self.results[-1].snippet:
                self.results[-1].snippet = snippet

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._cur_title.append(data)
        if self._in_snippet:
            self._cur_snippet.append(data)


# --------------------------------------------------------------------------- #
# Searcher
# --------------------------------------------------------------------------- #
class WebSearcher:
    """Provider-routing web searcher. Keyless DuckDuckGo HTML by default."""

    def __init__(self, *, provider: str = "auto",
                 brave_key: Optional[str] = None,
                 tavily_key: Optional[str] = None,
                 serpapi_key: Optional[str] = None,
                 max_results: int = 10, timeout_s: float = 30.0,
                 max_bytes: int = 5_000_000,
                 user_agent: str = "NYXARA/1.0 (+https://nyxara.ai)",
                 governor: Any = None, allow_private: bool = False,
                 fetcher: Any = None) -> None:
        self.provider = provider
        self.brave_key = brave_key
        self.tavily_key = tavily_key
        self.serpapi_key = serpapi_key
        self.max_results = max_results
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.user_agent = user_agent
        self.governor = governor
        self.allow_private = allow_private
        self._fetcher = fetcher  # injected WebFetcher (tests pass a fake-transport one)

    # ---- public ---- #
    def search(self, query: str, max_results: Optional[int] = None) -> SearchResponse:
        """Search ``query`` across the configured providers, returning the first hit set.

        Tries each backend in :meth:`_provider_order`; the first that yields results wins.
        Never raises — a total failure returns a :class:`SearchResponse` with ``error`` set
        and empty ``results``.
        """
        query = (query or "").strip()
        n = max(1, int(max_results or self.max_results))
        if not query:
            return SearchResponse(query=query, provider="none", error="empty query")
        order = self._provider_order()
        last_err: Optional[str] = None
        for prov in order:
            try:
                resp = self._dispatch(prov, query, n)
            except Exception as exc:  # noqa: BLE001 — errors as data, try the next backend
                last_err = f"{type(exc).__name__}: {exc}"
                continue
            if resp.results:
                return resp
            last_err = resp.error or last_err
        return SearchResponse(query=query, provider=(order[-1] if order else "none"),
                              error=last_err or "no results")

    # ---- routing ---- #
    def _provider_order(self) -> List[str]:
        # the keyless tail: full-web DDG first, then Wikipedia (works from anywhere), then
        # the instant-answer abstract — each backend only "wins" if it returns results.
        keyless = ["duckduckgo", "wikipedia", "ddg_instant"]
        if self.provider == "duckduckgo":
            return keyless
        if self.provider == "wikipedia":
            return ["wikipedia", "ddg_instant"]
        if self.provider in ("brave", "tavily", "serpapi"):
            return [self.provider] + keyless
        # "auto": prefer a configured API key, then the keyless tail.
        chain: List[str] = []
        if self.brave_key:
            chain.append("brave")
        if self.tavily_key:
            chain.append("tavily")
        if self.serpapi_key:
            chain.append("serpapi")
        chain += keyless
        return chain

    def _dispatch(self, prov: str, query: str, n: int) -> SearchResponse:
        return {
            "duckduckgo": self._ddg_html,
            "wikipedia": self._wikipedia,
            "ddg_instant": self._ddg_instant,
            "brave": self._brave,
            "tavily": self._tavily,
            "serpapi": self._serpapi,
        }[prov](query, n)

    # ---- keyless backends (reuse WebFetcher → SSRF + governor + cache) ---- #
    def _get_fetcher(self) -> Any:
        if self._fetcher is not None:
            return self._fetcher
        from nyxara.senses.web import WebFetcher
        return WebFetcher(governor=self.governor, allow_private=self.allow_private,
                          max_bytes=self.max_bytes, timeout=self.timeout_s,
                          user_agent=self.user_agent)

    def _ddg_html(self, query: str, n: int) -> SearchResponse:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        res = self._get_fetcher().fetch(url)
        if not res.ok:
            return SearchResponse(query, "duckduckgo",
                                  error=res.error or f"HTTP {res.status}")
        parser = _DDGHTMLParser()
        parser.feed(res.body)
        return SearchResponse(query, "duckduckgo", results=parser.results[:n])

    def _ddg_instant(self, query: str, n: int) -> SearchResponse:
        url = ("https://api.duckduckgo.com/?q=" + urllib.parse.quote(query)
               + "&format=json&no_html=1&no_redirect=1&skip_disambig=1")
        res = self._get_fetcher().fetch(url)
        if not res.ok:
            return SearchResponse(query, "ddg_instant",
                                  error=res.error or f"HTTP {res.status}")
        data = json.loads(res.body)
        out: List[SearchResult] = []
        if data.get("AbstractText"):
            out.append(SearchResult(title=data.get("Heading", ""),
                                    url=data.get("AbstractURL", ""),
                                    snippet=data["AbstractText"]))
        for topic in data.get("RelatedTopics", []):
            if len(out) >= n:
                break
            if isinstance(topic, dict) and topic.get("Text"):
                out.append(SearchResult(title="", url=topic.get("FirstURL", ""),
                                        snippet=topic["Text"]))
        return SearchResponse(query, "ddg_instant", results=out[:n])

    def _wikipedia(self, query: str, n: int) -> SearchResponse:
        # keyless, structured, works from any IP. Returns real ranked articles with snippets.
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
               "&srlimit=" + str(n) + "&srsearch=" + urllib.parse.quote(query))
        res = self._get_fetcher().fetch(url)
        if not res.ok:
            return SearchResponse(query, "wikipedia",
                                  error=res.error or f"HTTP {res.status}")
        data = json.loads(res.body)
        out: List[SearchResult] = []
        for hit in (data.get("query", {}).get("search", []) or []):
            title = hit.get("title", "")
            page = urllib.parse.quote(title.replace(" ", "_"))
            snippet = _strip_tags(hit.get("snippet", ""))
            out.append(SearchResult(title=title,
                                    url=f"https://en.wikipedia.org/wiki/{page}",
                                    snippet=snippet))
        return SearchResponse(query, "wikipedia", results=out[:n])

    # ---- API backends (via http_request → SSRF-guarded) ---- #
    def _brave(self, query: str, n: int) -> SearchResponse:
        if not self.brave_key:
            return SearchResponse(query, "brave", error="no brave api key")
        from nyxara.agency.net_request import http_request
        url = ("https://api.search.brave.com/res/v1/web/search?q="
               + urllib.parse.quote(query) + f"&count={n}")
        r = http_request(url, headers={"X-Subscription-Token": self.brave_key,
                                       "Accept": "application/json"},
                         timeout_s=self.timeout_s, max_bytes=self.max_bytes,
                         allow_private=self.allow_private)
        if not r["ok"]:
            return SearchResponse(query, "brave", error=r["error"])
        data = json.loads(r["body"])
        out = [SearchResult(title=it.get("title", ""), url=it.get("url", ""),
                            snippet=it.get("description", ""))
               for it in (data.get("web", {}).get("results", []) or [])]
        return SearchResponse(query, "brave", results=out[:n])

    def _tavily(self, query: str, n: int) -> SearchResponse:
        if not self.tavily_key:
            return SearchResponse(query, "tavily", error="no tavily api key")
        from nyxara.agency.net_request import http_request
        body = json.dumps({"api_key": self.tavily_key, "query": query,
                           "max_results": n})
        r = http_request("https://api.tavily.com/search", method="POST", body=body,
                         headers={"Content-Type": "application/json"},
                         timeout_s=self.timeout_s, max_bytes=self.max_bytes,
                         allow_private=self.allow_private)
        if not r["ok"]:
            return SearchResponse(query, "tavily", error=r["error"])
        data = json.loads(r["body"])
        out = [SearchResult(title=it.get("title", ""), url=it.get("url", ""),
                            snippet=it.get("content", ""))
               for it in (data.get("results", []) or [])]
        return SearchResponse(query, "tavily", results=out[:n])

    def _serpapi(self, query: str, n: int) -> SearchResponse:
        if not self.serpapi_key:
            return SearchResponse(query, "serpapi", error="no serpapi api key")
        from nyxara.agency.net_request import http_request
        url = ("https://serpapi.com/search.json?engine=google&q="
               + urllib.parse.quote(query) + f"&num={n}&api_key="
               + urllib.parse.quote(self.serpapi_key))
        r = http_request(url, timeout_s=self.timeout_s, max_bytes=self.max_bytes,
                         allow_private=self.allow_private)
        if not r["ok"]:
            return SearchResponse(query, "serpapi", error=r["error"])
        data = json.loads(r["body"])
        out = [SearchResult(title=it.get("title", ""), url=it.get("link", ""),
                            snippet=it.get("snippet", ""))
               for it in (data.get("organic_results", []) or [])]
        return SearchResponse(query, "serpapi", results=out[:n])


# --------------------------------------------------------------------------- #
# Self-test / demo — the only path that touches the live network (manual).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import sys

    q = " ".join(sys.argv[1:]) or "python release notes"
    print("=" * 70)
    print(f"NYXARA web-search self-test (LIVE network) — query: {q!r}")
    print("=" * 70)
    resp = WebSearcher().search(q)
    print(f"provider={resp.provider} error={resp.error!r} n={len(resp.results)}")
    for i, r in enumerate(resp.results, 1):
        print(f"\n[{i}] {r.title}\n    {r.url}\n    {r.snippet[:160]}")
