"""Knowledge graph — typed triples + natural-language → query.

Semantic memory also lives as a graph: entities are nodes, relations are labelled edges
(subject —predicate→ object), each edge carrying confidence and provenance. On top sits a
small natural-language query layer that turns questions like "who knows JP?" or
"what is Nyxara?" into graph traversals, so beliefs can be queried relationally rather
than only by vector similarity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx

from nyxara.contracts import Provenance


def _normalize_verb(verb: str) -> str:
    """Map an inflected query verb to its stored stem, e.g. 'owns'/'owned' -> 'own'."""
    v = verb.lower()
    for suffix in ("es", "ed", "s"):
        if v.endswith(suffix) and len(v) - len(suffix) >= 2:
            return v[: -len(suffix)]
    return v


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.7
    provenance: Provenance | None = None


class KnowledgeGraph:
    """A directed multigraph of entity relations with NL querying."""

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    # ── mutation ───────────────────────────────────────────────────────────────────
    def assert_triple(self, t: Triple) -> None:
        self._g.add_node(t.subject)
        self._g.add_node(t.object)
        self._g.add_edge(t.subject, t.object, key=t.predicate,
                         predicate=t.predicate, confidence=t.confidence,
                         provenance=t.provenance)

    def add(self, subject: str, predicate: str, obj: str, *,
            confidence: float = 0.7, provenance: Provenance | None = None) -> None:
        self.assert_triple(Triple(subject, predicate, obj, confidence, provenance))

    # ── querying ───────────────────────────────────────────────────────────────────
    def neighbors(self, entity: str, predicate: str | None = None) -> list[Triple]:
        out: list[Triple] = []
        if entity not in self._g:
            return out
        for _, obj, data in self._g.out_edges(entity, data=True):
            if predicate is None or data["predicate"] == predicate:
                out.append(self._edge_to_triple(entity, obj, data))
        return out

    def incoming(self, entity: str, predicate: str | None = None) -> list[Triple]:
        out: list[Triple] = []
        if entity not in self._g:
            return out
        for subj, _, data in self._g.in_edges(entity, data=True):
            if predicate is None or data["predicate"] == predicate:
                out.append(self._edge_to_triple(subj, entity, data))
        return out

    def path(self, src: str, dst: str) -> list[str] | None:
        if src not in self._g or dst not in self._g:
            return None
        try:
            return nx.shortest_path(self._g, src, dst)
        except nx.NetworkXNoPath:
            return None

    @staticmethod
    def _edge_to_triple(subj: str, obj: str, data: dict[str, Any]) -> Triple:
        return Triple(subject=subj, predicate=data["predicate"], object=obj,
                      confidence=data.get("confidence", 0.7),
                      provenance=data.get("provenance"))

    # ── natural-language → query ─────────────────────────────────────────────────────
    _WHO = re.compile(r"who (\w+)s? (.+?)\??$", re.I)        # "who knows JP"
    _WHAT_IS = re.compile(r"what is (?:an? )?(.+?)\??$", re.I)  # "what is Nyxara"
    _DOES = re.compile(r"does (.+?) (\w+) (.+?)\??$", re.I)   # "does JP own Nyxara"

    def query(self, question: str) -> list[Triple]:
        """Answer a simple NL question by mapping it to graph traversals."""
        q = question.strip()
        if (m := self._WHO.match(q)):
            predicate, obj = _normalize_verb(m.group(1)), m.group(2).strip()
            return self.incoming(obj, predicate)
        if (m := self._WHAT_IS.match(q)):
            entity = m.group(1).strip()
            return self.neighbors(entity)
        if (m := self._DOES.match(q)):
            subj, predicate, obj = m.group(1), _normalize_verb(m.group(2)), m.group(3)
            return [t for t in self.neighbors(subj, predicate)
                    if t.object.lower() == obj.lower()]
        # Fallback: treat the question as an entity name and return what we know of it.
        return self.neighbors(q)

    @property
    def stats(self) -> dict[str, int]:
        return {"entities": self._g.number_of_nodes(),
                "relations": self._g.number_of_edges()}


__all__ = ["KnowledgeGraph", "Triple"]
