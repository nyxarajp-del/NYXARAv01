"""NYXARA · njp/embed.py — entities that share structure share a representation (🧭, NJP V.07).

**The defect this closes, measured before it was written.** ``eval/relational.py`` established that
training the gradient head on ``(subject, predicate) → object`` memorises perfectly and generalises
not at all: held-out members scored MRR 0.521 against a nonsense-subject control of 0.500, so the
subject contributed 0.021 and the ranking was the object prior. The cause was not the objective. It
is :func:`~nyxara.njp.brain._cell_id`, a blake2b digest — ``sparrow`` and ``finch`` get orthogonal
representations with no shared structure, and generalisation requires that similar things be
represented similarly. A hash guarantees the opposite.

This is the missing similarity structure, and **it is not a new mechanism**.
:class:`~nyxara.njp.manifold.Manifold` already carries the whole apparatus — deterministic
hypervector atoms, an invertible :meth:`~nyxara.njp.manifold.Manifold.bind`, and a cosine
:meth:`~nyxara.njp.manifold.Manifold.similarity` — and it is tested. An entity is represented as
the bundle of its ``bind(predicate, object)`` contexts, which is textbook role-filler binding: two
entities that participate in the same relations share terms and therefore land close together, and
two that do not are near-orthogonal. Measured::

    sparrow ~ crow    (identical facts)   +1.000
    sparrow ~ finch   (same kind only)    +0.493
    sparrow ~ trout   (different kind)    +0.252

**What it is for, and what it must not be mistaken for.**
:meth:`~nyxara.njp.core.CognitiveLearningCore._inherit` already walks a *stated* ``is_a`` edge,
exactly and symbolically, and this does not compete with it — where a taxonomy is stated, the
symbolic walk is better in every way: exact, explainable, and defeasible in a way the store can
price. What this adds is the case ``_inherit`` cannot reach at all: **entities with no stated kind
between them**, tied only by having turned up in the same kinds of relation.

That case is what was measured, on six groups with six distinct targets and no ``is_a`` anywhere::

    plain (hashed cells)   held 0.542   control 0.394   gap +0.147
    embedding-expanded     held 1.000   control 0.394   gap +0.606

Six of six held-out entities at rank 1, with the control unmoved — so the gain came from the
subject and not from the prior. The first control run of this experiment was too small to
discriminate (two candidate objects, and a nonsense subject scored rank 1 by chance); it was
rebuilt rather than reported, which is the only reason the number above means anything.

**It proposes and never states.** A neighbour-expanded prediction is an association from
distributional overlap. It carries no provenance, it is not a derivation, and
:mod:`nyxara.njp.compete` scores it on the evidence axis like any other proposal — which is where
it belongs and usually where it loses.

Pure standard library beyond the manifold's numpy. Every entry point is fail-soft.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["EntityEmbedding"]

#: Below this two entities are not neighbours. Set from the measurement above: same-kind pairs sit
#: near 0.49 and different-kind pairs near 0.25, so the floor separates them with room on both
#: sides rather than sitting on top of either.
NEIGHBOUR_FLOOR = 0.35

#: How many neighbours may be brought in. Small on purpose — each one adds its cells to the query,
#: and a wide expansion stops being "entities like this one" and becomes "most of the store".
MAX_NEIGHBOURS = 3


class EntityEmbedding:
    """Entities as bundles of their relational contexts, over the manifold's own vectors."""

    def __init__(self, manifold: Any, *, cell_id: Any,
                 floor: float = NEIGHBOUR_FLOOR,
                 max_neighbours: int = MAX_NEIGHBOURS) -> None:
        self.manifold = manifold
        self._cell_id = cell_id
        self.floor = max(0.0, min(1.0, float(floor)))
        self.max_neighbours = max(1, int(max_neighbours))
        #: entity → its bundled context vector. Rebuilt when its facts change, never accumulated
        #: across changes: a stale term would keep an entity near something it no longer resembles.
        self._vectors: Dict[str, Any] = {}
        self._facts: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        self.built = 0
        self.queries = 0

    # ---- building ---------------------------------------------------------- #
    def observe(self, entity: str, contexts: Sequence[Tuple[str, str]]) -> bool:
        """Record what relations this entity takes part in, and rebuild its vector.

        ``contexts`` is ``(predicate, object)`` pairs. Deduplicated and ordered, because the bundle
        must be a function of the *set* of contexts — an entity whose facts arrive in a different
        order is the same entity, and a representation that disagreed would make similarity depend
        on transcript order.
        """
        try:
            name = str(entity or "").strip().lower()
            if not name:
                return False
            pairs = tuple(sorted({(str(p or "").strip().lower(), str(o or "").strip().lower())
                                  for p, o in (contexts or ()) if p and o}))
            if not pairs or self._facts.get(name) == pairs:
                return False
            self._facts[name] = pairs
            vector = self._bundle(pairs)
            if vector is None:
                return False
            self._vectors[name] = vector
            self.built += 1
            return True
        except Exception:  # noqa: BLE001
            return False

    def _bundle(self, pairs: Sequence[Tuple[str, str]]) -> Optional[Any]:
        """Sum the bound role-filler pairs and take the sign. The manifold's own operation."""
        try:
            import numpy as np
            atom = self.manifold.atom
            acc = np.zeros(self.manifold.dim, dtype=np.int32)
            for predicate, obj in pairs:
                acc += np.asarray(
                    self.manifold.bind(atom(self._cell_id(predicate)),
                                       atom(self._cell_id(obj))), dtype=np.int32)
            return np.where(acc >= 0, 1, -1).astype(np.int8)
        except Exception:  # noqa: BLE001 — no numpy, no embeddings; the caller degrades
            return None

    # ---- using ------------------------------------------------------------- #
    def similarity(self, a: str, b: str) -> float:
        """How alike two entities' relational contexts are. 0.0 when either is unknown."""
        try:
            left = self._vectors.get(str(a or "").strip().lower())
            right = self._vectors.get(str(b or "").strip().lower())
            if left is None or right is None:
                return 0.0
            return float(self.manifold.similarity(left, right))
        except Exception:  # noqa: BLE001
            return 0.0

    def neighbours(self, entity: str, *, k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Entities whose relational contexts resemble this one's, closest first.

        Empty for an entity with no recorded contexts, and that is the honest answer rather than a
        degenerate one: an entity she knows nothing about resembles nothing, and returning the
        globally most-connected entities instead would be a popularity ranking wearing the name of
        a similarity.
        """
        try:
            self.queries += 1
            name = str(entity or "").strip().lower()
            me = self._vectors.get(name)
            if me is None:
                return []
            scored = [(other, float(self.manifold.similarity(me, vec)))
                      for other, vec in self._vectors.items() if other != name]
            scored = [(o, s) for o, s in scored if s >= self.floor]
            scored.sort(key=lambda kv: (-kv[1], kv[0]))
            return scored[: (self.max_neighbours if k is None else max(0, int(k)))]
        except Exception:  # noqa: BLE001
            return []

    def expand(self, entity: str) -> List[str]:
        """The neighbour names alone — what a query is widened with."""
        return [name for name, _ in self.neighbours(entity)]

    def stats(self) -> Dict[str, Any]:
        return {"entities": len(self._vectors), "built": self.built,
                "queries": self.queries, "floor": self.floor,
                "max_neighbours": self.max_neighbours}
