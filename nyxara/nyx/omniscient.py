"""NYXARA · nyx/omniscient.py — the world arriving as structure, not as a search box (🌐, NYX V.03).

Everything she knows about the world outside her own tree arrives because somebody asked a
question. :mod:`nyxara.growth.researcher` searches when prompted; :mod:`nyxara.nyx.aura` watches
streams but files what is *surprising* rather than what is *true*. So a paper published yesterday
is not something she knows — it is something she could go and look up, which is a different thing.

This module ingests continuously into a **live typed hypergraph**, and the point of that is one
property: :meth:`Omniscient.ask` answers **from what has already been ingested, with no live
search**, and says how old the answer is.

Three things make it structure rather than a pile of text:

* **Typed n-ary hyperedges.** :mod:`nyxara.memory.graph` stores triples, and
  :mod:`nyxara.nyx.meta_architecture` gives an edge a *type*. Neither can say "these five things
  participated in one event, in these roles" — a paper with its authors, its categories and its
  date is one fact, not eleven unrelated pairs. :class:`Hyperedge` is that fact, reified into the
  existing graph as role-tagged triples sharing a hyperedge id, so it persists through the
  KnowledgeGraph's own ``to_dict``/``from_dict`` without this module rewriting any of it.
* **Freshness as a first-class property.** Every :class:`Fact` carries ``observed_at`` and a
  half-life, and :meth:`Omniscient.ask` ranks by relevance *times* recency. Yesterday's paper
  beats a stale one on purpose, and the answer reports ``as_of`` and its sources so the age is
  visible instead of implied.
* **An attention budget.** A feed is unbounded and a mind is not. ``max_events_per_min`` is a hard
  cap; past it events are dropped and **counted**, so the ceiling shows up in ``stats()`` rather
  than silently shaping what she knows.

Honest framing, and the first point is the one that matters:

* **She ingests the feeds she is configured to watch — not "the internet".** Nobody has a total
  feed of the world. :meth:`Omniscient.stats` reports coverage, lag and drops precisely so the
  boundary is visible rather than implied by the module's name.
* **Offline is a clean no-op.** With no network the pulse ingests nothing and says so. It never
  invents a fact to fill a gap, which is the failure mode that would make all of this worthless.
* **Nothing here loosens the network posture.** Every fetch goes through the existing
  :func:`~nyxara.agency.net_request.http_request`, so the SSRF guard, the scheme and private-host
  refusals, the byte cap and the redirect limit are exactly the ones already in place. The
  injectable transport exists so tests never touch the network — not as a way around any of that.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Source", "Fact", "Hyperedge", "Answer", "Omniscient", "DEFAULT_SOURCES",
           "http_transport", "DAY"]

DAY = 86_400.0


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    """One feed she watches. ``kind`` selects the parser; nothing else varies per source."""

    name: str
    url: str
    kind: str = "rss"                       # arxiv | github | rss | json
    cadence_s: float = 900.0
    enabled: bool = True
    credential_name: Optional[str] = None   # resolved by the vault at call time, never stored here
    half_life_s: float = 30 * DAY
    meta: Dict[str, Any] = field(default_factory=dict)
    # cursors — what we already saw, so a poll is incremental
    etag: str = ""
    last_polled: float = 0.0
    last_ok: float = 0.0
    seen: List[str] = field(default_factory=list)
    errors: int = 0
    last_error: str = ""

    def due(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        return self.enabled and (now - self.last_polled) >= self.cadence_s

    def lag_s(self, now: Optional[float] = None) -> float:
        """How long since this source last delivered anything. Reported, not hidden."""
        now = time.time() if now is None else now
        return now - self.last_ok if self.last_ok else float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "url": self.url, "kind": self.kind,
                "cadence_s": self.cadence_s, "enabled": self.enabled,
                "credential_name": self.credential_name, "half_life_s": self.half_life_s,
                "meta": dict(self.meta), "etag": self.etag, "last_polled": self.last_polled,
                "last_ok": self.last_ok, "seen": list(self.seen[-500:]), "errors": self.errors,
                "last_error": self.last_error}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Source":
        return cls(name=d["name"], url=d["url"], kind=d.get("kind", "rss"),
                   cadence_s=float(d.get("cadence_s") or 900.0),
                   enabled=bool(d.get("enabled", True)),
                   credential_name=d.get("credential_name"),
                   half_life_s=float(d.get("half_life_s") or 30 * DAY),
                   meta=dict(d.get("meta") or {}), etag=d.get("etag", ""),
                   last_polled=float(d.get("last_polled") or 0.0),
                   last_ok=float(d.get("last_ok") or 0.0),
                   seen=list(d.get("seen") or ()), errors=int(d.get("errors") or 0),
                   last_error=d.get("last_error", ""))


@dataclass
class Fact:
    """One observed relation, with when it was seen and how fast it goes stale."""

    subject: str
    predicate: str
    object: str
    source: str = ""
    observed_at: float = field(default_factory=time.time)
    confidence: float = 0.8
    half_life_s: float = 30 * DAY
    hyperedge: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def recency(self, now: Optional[float] = None) -> float:
        """Exponential decay on the source's half-life. Never negative, never above 1."""
        now = time.time() if now is None else now
        age = max(0.0, now - self.observed_at)
        if self.half_life_s <= 0:
            return 1.0
        return float(2.0 ** (-age / self.half_life_s))

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "predicate": self.predicate, "object": self.object,
                "source": self.source, "observed_at": self.observed_at,
                "confidence": self.confidence, "half_life_s": self.half_life_s,
                "hyperedge": self.hyperedge, "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fact":
        return cls(subject=d["subject"], predicate=d["predicate"], object=d["object"],
                   source=d.get("source", ""), observed_at=float(d.get("observed_at") or 0.0),
                   confidence=float(d.get("confidence") or 0.8),
                   half_life_s=float(d.get("half_life_s") or 30 * DAY),
                   hyperedge=d.get("hyperedge", ""), meta=dict(d.get("meta") or {}))


@dataclass
class Hyperedge:
    """One event with several participants, each in a named role.

    This is the structure triples cannot express. ``participants`` maps ``role → entity``; a role
    that genuinely repeats (several authors) is stored as ``author.0``, ``author.1``, so the shape
    stays flat and serialisable while remaining n-ary.
    """

    uid: str
    kind: str
    participants: Dict[str, str] = field(default_factory=dict)
    source: str = ""
    observed_at: float = field(default_factory=time.time)
    confidence: float = 0.8
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def arity(self) -> int:
        return len(self.participants)

    def role_of(self, entity: str) -> str:
        for role, value in self.participants.items():
            if value == entity:
                return role.split(".")[0]
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {"uid": self.uid, "kind": self.kind, "participants": dict(self.participants),
                "source": self.source, "observed_at": self.observed_at,
                "confidence": self.confidence, "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Hyperedge":
        return cls(uid=d["uid"], kind=d["kind"],
                   participants=dict(d.get("participants") or {}), source=d.get("source", ""),
                   observed_at=float(d.get("observed_at") or 0.0),
                   confidence=float(d.get("confidence") or 0.8), meta=dict(d.get("meta") or {}))


@dataclass
class Answer:
    """What she knows about a question *right now*, and how old that is."""

    query: str = ""
    facts: List[Fact] = field(default_factory=list)
    as_of: float = 0.0
    sources: List[str] = field(default_factory=list)
    searched_live: bool = False        # always False — that is the whole point

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.as_of) if self.as_of else float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "facts": [f.to_dict() for f in self.facts],
                "as_of": self.as_of, "age_s": round(self.age_s, 1), "sources": self.sources,
                "searched_live": self.searched_live, "n": len(self.facts)}


# --------------------------------------------------------------------------- #
# Transport — the only place a socket is opened
# --------------------------------------------------------------------------- #
def http_transport(url: str, *, headers: Optional[Dict[str, str]] = None,
                   timeout_s: float = 15.0) -> Dict[str, Any]:
    """The real fetch, through the existing guarded request. Never raises."""
    try:
        from nyxara.agency.net_request import http_request
        return http_request(url, headers=headers or {}, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 — an unavailable stack is an offline pulse
        return {"ok": False, "status": 0, "body": "", "headers": {}, "error": str(exc)}


# Three public feeds that need no credential. Watched by default so the layer is genuinely live
# out of the box rather than a switched-off shell — and every one of them is still a *configured*
# source, which is exactly the boundary the docstring claims.
DEFAULT_SOURCES: Tuple[Source, ...] = (
    Source(name="arxiv-cs-ai", kind="arxiv", cadence_s=3600.0, half_life_s=60 * DAY,
           url=("http://export.arxiv.org/api/query?search_query=cat:cs.AI"
                "&sortBy=submittedDate&sortOrder=descending&max_results=25")),
    Source(name="github-events", kind="github", cadence_s=900.0, half_life_s=7 * DAY,
           url="https://api.github.com/events"),
    Source(name="arxiv-math", kind="arxiv", cadence_s=3600.0, half_life_s=60 * DAY,
           url=("http://export.arxiv.org/api/query?search_query=cat:math.CO"
                "&sortBy=submittedDate&sortOrder=descending&max_results=25")),
)

_ATOM = "{http://www.w3.org/2005/Atom}"
_LIMITATION_MARKERS = ("remains an open problem", "future work", "we leave", "is not yet known",
                       "remains unclear", "an open question", "we do not know", "limitation",
                       "remains open", "further work")


def _slug(text: str, limit: int = 90) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


# --------------------------------------------------------------------------- #
# Parsers — bytes in, hyperedges out. Deterministic; no model involved.
# --------------------------------------------------------------------------- #
def _parse_arxiv(body: str, source: Source) -> List[Hyperedge]:
    out: List[Hyperedge] = []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return out
    for entry in root.findall(f"{_ATOM}entry"):
        title = _slug((entry.findtext(f"{_ATOM}title") or ""))
        if not title:
            continue
        uid = (entry.findtext(f"{_ATOM}id") or title).strip()
        participants: Dict[str, str] = {"paper": title}
        authors = [(a.findtext(f"{_ATOM}name") or "").strip()
                   for a in entry.findall(f"{_ATOM}author")]
        for i, author in enumerate([a for a in authors if a][:8]):
            participants[f"author.{i}"] = author
        for i, cat in enumerate(entry.findall(f"{_ATOM}category")[:4]):
            term = (cat.attrib.get("term") or "").strip()
            if term:
                participants[f"category.{i}"] = term
        published = (entry.findtext(f"{_ATOM}published") or "").strip()
        if published:
            participants["published"] = published
        summary = _slug(entry.findtext(f"{_ATOM}summary") or "", 4000)
        out.append(Hyperedge(uid=uid, kind="publication", participants=participants,
                             source=source.name, confidence=0.9,
                             meta={"summary": summary, "url": uid}))
    return out


def _parse_github(body: str, source: Source) -> List[Hyperedge]:
    out: List[Hyperedge] = []
    try:
        events = json.loads(body)
    except ValueError:
        return out
    if not isinstance(events, list):
        return out
    for event in events:
        if not isinstance(event, dict):
            continue
        repo = ((event.get("repo") or {}).get("name") or "").strip()
        actor = ((event.get("actor") or {}).get("login") or "").strip()
        etype = str(event.get("type") or "Event")
        uid = str(event.get("id") or f"{repo}:{etype}")
        if not repo:
            continue
        participants = {"repo": repo, "event": etype}
        if actor:
            participants["actor"] = actor
        payload = event.get("payload") or {}
        for i, commit in enumerate((payload.get("commits") or [])[:5]):
            message = _slug((commit or {}).get("message") or "", 120)
            if message:
                participants[f"commit.{i}"] = message
        out.append(Hyperedge(uid=uid, kind="repo_event", participants=participants,
                             source=source.name, confidence=0.85,
                             meta={"created_at": event.get("created_at") or ""}))
    return out


def _parse_rss(body: str, source: Source) -> List[Hyperedge]:
    out: List[Hyperedge] = []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return out
    items = root.findall(".//item") or root.findall(f".//{_ATOM}entry")
    for item in items:
        title = _slug(item.findtext("title") or item.findtext(f"{_ATOM}title") or "")
        if not title:
            continue
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or item.findtext(f"{_ATOM}updated") or "").strip()
        desc = _slug(item.findtext("description") or item.findtext(f"{_ATOM}summary") or "", 4000)
        participants = {"item": title}
        if link:
            participants["link"] = link
        if pub:
            participants["published"] = pub
        out.append(Hyperedge(uid=link or title, kind="feed_item", participants=participants,
                             source=source.name, confidence=0.7, meta={"summary": desc}))
    return out


def _parse_json_series(body: str, source: Source) -> List[Hyperedge]:
    """A metric series: ``{"series": [{"period": ..., "value": ...}, ...]}`` or a bare list.

    ``meta["metric"]`` names what is being measured; without it the source's own name is used, so
    a number never lands in the graph without something saying what it is a number *of*.
    """
    out: List[Hyperedge] = []
    try:
        data = json.loads(body)
    except ValueError:
        return out
    rows = data.get("series") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return out
    metric = str(source.meta.get("metric") or source.name)
    for row in rows:
        if not isinstance(row, dict):
            continue
        period = str(row.get("period") or row.get("date") or "").strip()
        value = row.get("value")
        if not period or value is None:
            continue
        out.append(Hyperedge(uid=f"{metric}:{period}", kind="observation",
                             participants={"metric": metric, "period": period,
                                           "value": str(value)},
                             source=source.name, confidence=0.95))
    return out


_PARSERS: Dict[str, Callable[[str, Source], List[Hyperedge]]] = {
    "arxiv": _parse_arxiv, "github": _parse_github, "rss": _parse_rss, "json": _parse_json_series,
}


# --------------------------------------------------------------------------- #
# The stream
# --------------------------------------------------------------------------- #
class Omniscient:
    """Continuous ingestion into a live typed hypergraph, answerable without a live search."""

    def __init__(self, *, sources: Optional[Sequence[Source]] = None, graph: Any = None,
                 transport: Optional[Callable[..., Dict[str, Any]]] = None,
                 max_events_per_min: int = 600, max_facts: int = 50_000,
                 vault: Any = None) -> None:
        self.sources: List[Source] = list(sources if sources is not None else DEFAULT_SOURCES)
        self.graph = graph
        self.transport = transport if transport is not None else http_transport
        self.max_events_per_min = max(1, int(max_events_per_min))
        self.max_facts = max(100, int(max_facts))
        self.vault = vault
        self.facts: List[Fact] = []
        self.hyperedges: Dict[str, Hyperedge] = {}
        self.pulses = 0
        self.dropped = 0
        self.ingested = 0
        self._minute_start = time.time()
        self._minute_count = 0

    # ---- budget ----------------------------------------------------------- #
    def _budget_allows(self, now: float) -> bool:
        """A hard ceiling, with the drops counted so it shows up instead of silently biasing her."""
        if now - self._minute_start >= 60.0:
            self._minute_start, self._minute_count = now, 0
        if self._minute_count >= self.max_events_per_min:
            return False
        self._minute_count += 1
        return True

    # ---- the pulse -------------------------------------------------------- #
    def pulse(self, *, now: Optional[float] = None, force: bool = False) -> Dict[str, Any]:
        """One bounded ingestion beat. Polls only what is due. Never raises."""
        now = time.time() if now is None else now
        self.pulses += 1
        report: Dict[str, Any] = {"polled": 0, "ingested": 0, "dropped": 0, "errors": 0,
                                  "sources": []}

        for source in self.sources:
            if not force and not source.due(now):
                continue
            report["polled"] += 1
            source.last_polled = now
            outcome = self._poll(source, now=now)
            report["ingested"] += outcome["ingested"]
            report["dropped"] += outcome["dropped"]
            report["errors"] += 1 if outcome.get("error") else 0
            report["sources"].append({"name": source.name, **outcome})

        self._prune()
        return report

    def _poll(self, source: Source, *, now: float) -> Dict[str, Any]:
        headers: Dict[str, str] = {"Accept": "*/*", "User-Agent": "nyxara-omniscient/1"}
        if source.etag:
            headers["If-None-Match"] = source.etag
        secret = self._credential(source)
        if secret:
            headers["Authorization"] = secret

        try:
            response = self.transport(source.url, headers=headers) or {}
        except Exception as exc:  # noqa: BLE001 — a transport that throws is an offline source
            source.errors += 1
            source.last_error = str(exc)
            return {"ingested": 0, "dropped": 0, "error": str(exc)}

        if not response.get("ok"):
            source.errors += 1
            source.last_error = str(response.get("error") or f"status {response.get('status')}")
            return {"ingested": 0, "dropped": 0, "error": source.last_error}

        etag = str((response.get("headers") or {}).get("ETag") or "")
        if etag:
            source.etag = etag
        if int(response.get("status") or 200) == 304:
            source.last_ok = now
            return {"ingested": 0, "dropped": 0, "unchanged": True}

        parser = _PARSERS.get(source.kind)
        if parser is None:
            return {"ingested": 0, "dropped": 0, "error": f"no parser for kind {source.kind!r}"}

        edges = parser(str(response.get("body") or ""), source)
        source.last_ok = now
        ingested = dropped = 0
        seen = set(source.seen)
        for edge in edges:
            if edge.uid in seen or edge.uid in self.hyperedges:
                continue
            if not self._budget_allows(now):
                dropped += 1
                self.dropped += 1
                continue
            edge.observed_at = now
            self._absorb(edge, source)
            seen.add(edge.uid)
            ingested += 1
        source.seen = list(seen)[-2000:]
        self.ingested += ingested
        return {"ingested": ingested, "dropped": dropped}

    def _credential(self, source: Source) -> str:
        """Injected at call time from the vault; never stored on the source."""
        if not source.credential_name or self.vault is None:
            return ""
        try:
            getter = getattr(self.vault, "get", None)
            return str(getter(source.credential_name) or "") if callable(getter) else ""
        except Exception:  # noqa: BLE001
            return ""

    def _absorb(self, edge: Hyperedge, source: Source) -> None:
        """File one hyperedge: keep it whole, and reify it into the triple graph."""
        self.hyperedges[edge.uid] = edge
        anchor = edge.participants.get("paper") or edge.participants.get("repo") \
            or edge.participants.get("item") or edge.participants.get("metric") or edge.uid

        for role, value in edge.participants.items():
            bare = role.split(".")[0]
            if value == anchor:
                continue
            fact = Fact(subject=anchor, predicate=bare, object=value, source=source.name,
                        observed_at=edge.observed_at, confidence=edge.confidence,
                        half_life_s=source.half_life_s, hyperedge=edge.uid,
                        meta={"kind": edge.kind})
            self.facts.append(fact)
            self._to_graph(fact, edge)

    def _to_graph(self, fact: Fact, edge: Hyperedge) -> None:
        """Reify into the existing KnowledgeGraph.

        The hyperedge id and role ride in ``attributes``, which the graph already persists, so the
        n-ary structure survives a save/load without this module changing ``memory/graph.py``.
        """
        if self.graph is None:
            return
        try:
            self.graph.add_triple(
                fact.subject, fact.predicate, fact.object, confidence=fact.confidence,
                attributes={"hyperedge": edge.uid, "kind": edge.kind, "source": fact.source,
                            "observed_at": fact.observed_at})
        except Exception:  # noqa: BLE001 — a graph that refuses a triple is not a failed pulse
            pass

    def _prune(self) -> None:
        """Bounded memory: the stalest facts go first, and their hyperedges go with them."""
        if len(self.facts) <= self.max_facts:
            return
        self.facts.sort(key=lambda f: f.observed_at)
        drop = len(self.facts) - self.max_facts
        for fact in self.facts[:drop]:
            self.hyperedges.pop(fact.hyperedge, None)
        del self.facts[:drop]

    # ---- answering, without a live search --------------------------------- #
    def ask(self, query: str, *, k: int = 8, now: Optional[float] = None) -> Answer:
        """Answer from what has already been ingested. **No network call happens here.**"""
        now = time.time() if now is None else now
        answer = Answer(query=str(query or ""))
        terms = [t for t in re.split(r"\W+", answer.query.lower()) if len(t) > 2]
        if not terms:
            return answer

        scored: List[Tuple[float, Fact]] = []
        for fact in self.facts:
            blob = f"{fact.subject} {fact.predicate} {fact.object}".lower()
            hits = sum(1 for t in terms if t in blob)
            if not hits:
                continue
            scored.append((hits / len(terms) * fact.recency(now) * fact.confidence, fact))

        scored.sort(key=lambda pair: -pair[0])
        answer.facts = [f for _, f in scored[:k]]
        answer.as_of = max((f.observed_at for f in answer.facts), default=0.0)
        answer.sources = sorted({f.source for f in answer.facts})
        return answer

    def frontier_statements(self) -> List[str]:
        """Stated limitations and open questions from what she has ingested.

        This is what :class:`~nyxara.nyx.telos.FrontierSource` consumes, which is how the world
        arriving turns into problems she sets herself.
        """
        out: List[str] = []
        for edge in self.hyperedges.values():
            summary = str(edge.meta.get("summary") or "")
            if not summary:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", summary):
                low = sentence.lower()
                if any(marker in low for marker in _LIMITATION_MARKERS) and len(sentence) > 30:
                    out.append(sentence.strip())
        return out

    # ---- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1,
                "sources": [s.to_dict() for s in self.sources],
                "facts": [f.to_dict() for f in self.facts],
                "hyperedges": [h.to_dict() for h in self.hyperedges.values()],
                "ingested": self.ingested, "dropped": self.dropped, "pulses": self.pulses}

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or data.get("version") != 1:
            return False
        try:
            self.sources = [Source.from_dict(s) for s in data.get("sources") or ()]
            self.facts = [Fact.from_dict(f) for f in data.get("facts") or ()]
            self.hyperedges = {h["uid"]: Hyperedge.from_dict(h)
                               for h in data.get("hyperedges") or ()}
        except (KeyError, TypeError, ValueError):
            return False
        self.ingested = int(data.get("ingested") or 0)
        self.dropped = int(data.get("dropped") or 0)
        self.pulses = int(data.get("pulses") or 0)
        return True

    def save(self, path: Any) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        os.replace(tmp, target)
        return target

    def load(self, path: Any) -> bool:
        try:
            return self.load_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return False

    # ---- reporting -------------------------------------------------------- #
    def coverage(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Per-source truth: is it live, how stale is it, how often has it failed."""
        now = time.time() if now is None else now
        out = []
        for source in self.sources:
            lag = source.lag_s(now)
            out.append({"name": source.name, "kind": source.kind, "enabled": source.enabled,
                        "lag_s": None if lag == float("inf") else round(lag, 1),
                        "never_delivered": source.last_ok == 0.0,
                        "errors": source.errors, "last_error": source.last_error})
        return out

    def stats(self, now: Optional[float] = None) -> Dict[str, Any]:
        kinds: Dict[str, int] = {}
        for edge in self.hyperedges.values():
            kinds[edge.kind] = kinds.get(edge.kind, 0) + 1
        arities = [e.arity for e in self.hyperedges.values()]
        return {
            "pulses": self.pulses,
            "sources": len(self.sources),
            "facts": len(self.facts),
            "hyperedges": len(self.hyperedges),
            "kinds": kinds,
            "mean_arity": round(sum(arities) / len(arities), 2) if arities else 0.0,
            "ingested": self.ingested,
            "dropped": self.dropped,
            "budget_per_min": self.max_events_per_min,
            "coverage": self.coverage(now),
            "note": ("ingests the feeds she is configured to watch — not 'the internet'; coverage, "
                     "lag and drops above are the boundary. ask() answers from the ingested graph "
                     "with no live search. Offline it ingests nothing and fabricates nothing"),
        }

    def summary(self) -> str:
        st = self.stats()
        lines = ["=" * 70, "NYXARA omniscient — the world she has actually ingested", "=" * 70,
                 f"pulses     : {st['pulses']}   ingested {st['ingested']}   "
                 f"dropped {st['dropped']}",
                 f"hypergraph : {st['hyperedges']} hyperedge(s), mean arity {st['mean_arity']}, "
                 f"{st['facts']} fact(s) {st['kinds']}"]
        for cov in st["coverage"]:
            mark = "·" if cov["never_delivered"] else "✓"
            lag = "never" if cov["never_delivered"] else f"{cov['lag_s']}s ago"
            lines.append(f"  {mark} {cov['name']:<16} {cov['kind']:<8} {lag}"
                         + (f"  ⚠ {cov['last_error'][:40]}" if cov["errors"] else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test / demo — no network is touched
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA omniscient self-test (offline — a stub transport, no sockets)")
    print("=" * 70)

    ARXIV = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2601.00001</id>
    <title>Sparse Attention Kernels for Long Context</title>
    <published>2026-08-08T00:00:00Z</published>
    <summary>We present a kernel. Whether it holds for non-causal masks remains an open problem.</summary>
    <author><name>A Researcher</name></author>
    <author><name>B Researcher</name></author>
    <category term="cs.LG"/>
  </entry>
</feed>"""

    GITHUB = json.dumps([{"id": "42", "type": "PushEvent",
                          "repo": {"name": "torvalds/linux"},
                          "actor": {"login": "torvalds"},
                          "payload": {"commits": [{"message": "fix a scheduler race"}]},
                          "created_at": "2026-08-09T00:00:00Z"}])

    def stub(url: str, **_kw: Any) -> Dict[str, Any]:
        return {"ok": True, "status": 200, "headers": {"ETag": "v1"},
                "body": ARXIV if "arxiv" in url else GITHUB}

    sources = [Source(name="arxiv", url="http://export.arxiv.org/api/query?x", kind="arxiv"),
               Source(name="gh", url="https://api.github.com/events", kind="github")]

    from nyxara.memory.graph import KnowledgeGraph

    graph = KnowledgeGraph()
    omni = Omniscient(sources=sources, transport=stub, graph=graph)
    print(f"\npulse          : {omni.pulse(force=True)}")
    assert omni.ingested > 0

    paper = next(h for h in omni.hyperedges.values() if h.kind == "publication")
    print(f"hyperedge      : arity {paper.arity} — {sorted(paper.participants)}")
    assert paper.arity >= 5, paper.participants
    assert paper.role_of("A Researcher") == "author"

    before = omni.ingested
    omni.pulse(force=True)
    assert omni.ingested == before, "a seen event must not be re-ingested"
    print("idempotent     : re-polling the same feed ingests nothing new")

    answer = omni.ask("sparse attention kernels")
    print(f"\nask            : {len(answer.facts)} fact(s), sources={answer.sources}, "
          f"live={answer.searched_live}")
    assert answer.facts and answer.searched_live is False and answer.as_of > 0

    now = time.time()
    omni2 = Omniscient(sources=[], transport=stub)
    omni2.facts = [
        Fact(subject="kernel study", predicate="topic", object="attention", source="old",
             observed_at=now - 400 * DAY, half_life_s=30 * DAY),
        Fact(subject="kernel study", predicate="topic", object="attention", source="new",
             observed_at=now, half_life_s=30 * DAY)]
    assert omni2.ask("kernel attention").facts[0].source == "new", "yesterday must beat last year"
    print("freshness      : the recent fact wins")

    restored = KnowledgeGraph.from_dict(graph.to_dict())
    carried = [t for t in restored._triples.values() if t.attributes.get("hyperedge")]
    print(f"reified        : {len(carried)} triple(s) carry a hyperedge id through save/load")
    assert carried

    statements = omni.frontier_statements()
    print(f"frontier       : {statements}")
    assert statements and "open problem" in statements[0]

    def dead(url: str, **_kw: Any) -> Dict[str, Any]:
        return {"ok": False, "status": 0, "body": "", "error": "network unreachable"}

    off = Omniscient(sources=[Source(name="a", url="http://x/y", kind="arxiv")], transport=dead)
    report = off.pulse(force=True)
    print(f"\noffline        : {report['ingested']} ingested, {report['errors']} error(s)")
    assert report["ingested"] == 0 and len(off.facts) == 0, "offline must fabricate nothing"

    many = json.dumps([{"id": str(i), "type": "PushEvent", "repo": {"name": f"o/r{i}"},
                        "actor": {"login": "x"}} for i in range(50)])
    capped = Omniscient(sources=[Source(name="gh", url="https://api.github.com/events",
                                        kind="github")],
                        transport=lambda url, **kw: {"ok": True, "status": 200, "body": many,
                                                     "headers": {}},
                        max_events_per_min=10)
    capped.pulse(force=True)
    print(f"budget         : ingested {capped.ingested}, dropped {capped.dropped}")
    assert capped.ingested == 10 and capped.dropped == 40

    with tempfile.TemporaryDirectory() as d:
        path = omni.save(Path(d) / "omni.json")
        again = Omniscient(sources=[], transport=stub)
        assert again.load(path) and len(again.hyperedges) == len(omni.hyperedges)
        print(f"persistence    : {len(again.hyperedges)} hyperedge(s) survived a restart")

    print("\n" + omni.summary())
    print("\nALL SELF-TESTS PASSED ✓")
