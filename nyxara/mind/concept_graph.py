"""NYXARA · mind/concept_graph.py — Semantic Associative Concept Graph (P24 · F19, organ 2).

A vector-based associative memory over the concepts NYXARA creates with. Every concept
(topic word, metaphor domain, stimulus, keyword of a kept creation) is embedded as a
**deterministic bipolar hypervector** — built on the real hyperdimensional algebra in
:mod:`nyxara.cognition.hyper_dimensional_vectors` — and connected by weighted
co-occurrence edges learned from her own kept creations.

Two capabilities matter for true originality:

* :meth:`associate` — learn the structure of every kept piece, so memory is a *graph*,
  not a flat archive;
* :meth:`distant_pair` — **cross-domain synthesis**: return two semantically FAR
  concepts (low vector similarity, no direct edge) as raw material for conceptual
  blending — quantum physics × classical poetry, not near-duplicate remixing.

Vectors are derived from an md5 of the concept name, so the embedding is stable across
runs, machines and insertion orders. **No LLM anywhere in this module.**
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from nyxara.cognition.hyper_dimensional_vectors import Hypervector

__all__ = [
    "ConceptGraph",
]

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "its", "her", "his",
    "are", "was", "has", "have", "not", "but", "all", "one", "two", "out", "how",
    "what", "when", "where", "why", "who", "can", "could", "would", "should", "than",
    "then", "them", "they", "there", "their", "your", "our", "you", "she", "it",
}


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9]+", text.lower())
            if len(w) > 2 and w not in _STOP]


class ConceptGraph:
    """Deterministic hypervector concepts + learned co-occurrence edges."""

    def __init__(self, *, dim: int = 256, max_nodes: int = 2048) -> None:
        self.dim = max(16, int(dim))
        self.max_nodes = max(64, int(max_nodes))
        self._vectors: Dict[str, Hypervector] = {}
        self._weight: Dict[str, float] = {}                 # node activation mass
        self._edges: Dict[Tuple[str, str], float] = {}      # undirected co-occurrence
        self.associations_seen: int = 0

    # ------------------------------------------------------------------ #
    # embedding — name-stable, run-stable, order-stable
    # ------------------------------------------------------------------ #
    def vector(self, concept: str) -> Hypervector:
        """The concept's bipolar hypervector, derived deterministically from its name."""
        name = concept.strip().lower()
        v = self._vectors.get(name)
        if v is None:
            seed = int.from_bytes(hashlib.md5(name.encode("utf-8")).digest()[:8], "big")
            rng = random.Random(seed)
            v = Hypervector([1.0 if rng.random() < 0.5 else -1.0
                             for _ in range(self.dim)])
            self._vectors[name] = v
        return v

    def similarity(self, a: str, b: str) -> float:
        return self.vector(a).cosine(self.vector(b))

    # ------------------------------------------------------------------ #
    # learning the graph from her kept creations
    # ------------------------------------------------------------------ #
    def _edge_key(self, a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def add_concept(self, concept: str, weight: float = 1.0) -> str:
        name = concept.strip().lower()
        if not name:
            return name
        self.vector(name)
        self._weight[name] = self._weight.get(name, 0.0) + float(weight)
        self._prune_if_needed()
        return name

    def associate(self, text: str, *, weight: float = 1.0,
                  extra_concepts: Optional[List[str]] = None) -> List[str]:
        """Learn from one creation: its key concepts join the graph, co-mentions bond."""
        concepts = _tokens(text)[:24]
        for c in (extra_concepts or []):
            c = c.strip().lower()
            if c and c not in concepts:
                concepts.append(c)
        for c in concepts:
            self.add_concept(c, weight)
        for i, a in enumerate(concepts):
            for b in concepts[i + 1:]:
                if a != b:
                    k = self._edge_key(a, b)
                    self._edges[k] = self._edges.get(k, 0.0) + float(weight)
        self.associations_seen += 1
        return concepts

    def _prune_if_needed(self) -> None:
        """Keep the graph bounded: drop the lowest-mass nodes (and their edges)."""
        if len(self._weight) <= self.max_nodes:
            return
        keep = dict(sorted(self._weight.items(), key=lambda kv: kv[1],
                           reverse=True)[: self.max_nodes])
        dropped = set(self._weight) - set(keep)
        self._weight = keep
        for name in dropped:
            self._vectors.pop(name, None)
        self._edges = {k: w for k, w in self._edges.items()
                       if k[0] not in dropped and k[1] not in dropped}

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def concepts(self) -> List[str]:
        return sorted(self._weight, key=lambda n: self._weight[n], reverse=True)

    def neighbors(self, concept: str, *, k: int = 5) -> List[Tuple[str, float]]:
        name = concept.strip().lower()
        out = []
        for (a, b), w in self._edges.items():
            if a == name:
                out.append((b, w))
            elif b == name:
                out.append((a, w))
        out.sort(key=lambda t: t[1], reverse=True)
        return out[:k]

    def edge_weight(self, a: str, b: str) -> float:
        return self._edges.get(self._edge_key(a.strip().lower(), b.strip().lower()), 0.0)

    def weak_regions(self, *, k: int = 5) -> List[str]:
        """Concepts with the fewest connections — where her map of meaning is thinnest."""
        degree: Dict[str, int] = {n: 0 for n in self._weight}
        for (a, b) in self._edges:
            if a in degree:
                degree[a] += 1
            if b in degree:
                degree[b] += 1
        return sorted(degree, key=lambda n: (degree[n], -self._weight.get(n, 0.0)))[:k]

    def distant_pair(self, *, rng: Optional[random.Random] = None,
                     candidates: int = 24) -> Optional[Tuple[str, str]]:
        """Two semantically FAR, unlinked concepts — raw material for cross-domain blending.

        Samples candidate pairs from the known concepts and returns the pair with the
        lowest combined (vector-similarity + normalised-edge) closeness: maximally
        foreign to each other, so blending them is genuine cross-domain synthesis."""
        names = self.concepts()
        if len(names) < 2:
            return None
        rng = rng or random.Random(0)
        max_edge = max(self._edges.values(), default=1.0) or 1.0
        best: Optional[Tuple[str, str]] = None
        best_close = float("inf")
        pool = names[: min(len(names), 64)]
        for _ in range(max(4, candidates)):
            a, b = rng.sample(pool, 2)
            closeness = abs(self.similarity(a, b)) + self.edge_weight(a, b) / max_edge
            if closeness < best_close:
                best_close = closeness
                best = (a, b)
        return best

    # ------------------------------------------------------------------ #
    # persistence & report
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"dim": self.dim, "max_nodes": self.max_nodes,
                "weights": {k: round(v, 4) for k, v in self._weight.items()},
                "edges": [[a, b, round(w, 4)] for (a, b), w in self._edges.items()],
                "associations_seen": self.associations_seen}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConceptGraph":
        g = cls(dim=int(d.get("dim", 256) or 256),
                max_nodes=int(d.get("max_nodes", 2048) or 2048))
        for name, w in (d.get("weights", {}) or {}).items():
            g.add_concept(str(name), float(w))
        for item in (d.get("edges", []) or []):
            try:
                a, b, w = str(item[0]), str(item[1]), float(item[2])
            except (IndexError, TypeError, ValueError):
                continue
            g._edges[g._edge_key(a, b)] = w
        g.associations_seen = int(d.get("associations_seen", 0) or 0)
        return g

    def report(self) -> Dict[str, Any]:
        return {"concepts": len(self._weight), "edges": len(self._edges),
                "associations_seen": self.associations_seen,
                "top": self.concepts()[:8], "thinnest": self.weak_regions(k=3)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA concept-graph self-test")
    print("=" * 70)

    g = ConceptGraph(dim=256)

    # embeddings are deterministic and name-stable
    assert g.vector("quantum") == ConceptGraph(dim=256).vector("quantum")
    assert g.similarity("quantum", "quantum") == 1.0
    assert abs(g.similarity("quantum", "sonnet")) < 0.5   # random names ≈ orthogonal

    # learn from creations
    g.associate("quantum entanglement links distant particles across space")
    g.associate("classical sonnet meter shapes the poem's turning line")
    g.associate("entanglement inspires a poem about distant lovers",
                extra_concepts=["synthesis"])
    print(f"\ngraph               : {g.report()}")
    assert g.edge_weight("quantum", "entanglement") > 0
    assert g.edge_weight("sonnet", "meter") > 0

    # neighbors reflect learned structure
    nb = [n for n, _ in g.neighbors("entanglement", k=10)]
    print(f"entanglement nbrs   : {nb[:5]}")
    assert "quantum" in nb

    # cross-domain synthesis: the pair is far apart and (near-)unlinked
    pair = g.distant_pair(rng=random.Random(3))
    assert pair is not None
    a, b = pair
    print(f"distant pair        : {a} × {b} (sim={g.similarity(a, b):.3f})")
    assert abs(g.similarity(a, b)) < 0.3

    # weak regions name the thinnest concepts
    weak = g.weak_regions(k=3)
    print(f"weak regions        : {weak}")
    assert len(weak) == 3

    # bounded growth
    small = ConceptGraph(dim=64, max_nodes=64)
    for i in range(300):
        small.associate(f"node{i} partner{i % 7} theme{i % 3}")
    assert len(small.concepts()) <= 64

    # persistence round-trip
    g2 = ConceptGraph.from_dict(g.to_dict())
    assert g2.edge_weight("quantum", "entanglement") == g.edge_weight("quantum", "entanglement")
    assert set(g2.concepts()) == set(g.concepts())
    print("persistence          : OK")

    print("\nALL SELF-TESTS PASSED ✓")
