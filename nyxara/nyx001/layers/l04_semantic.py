"""NYXARA · nyx001/layers/l04_semantic.py — Layer 4, concept formation (💠, shared structure).

The charter: "Extracts abstract concepts from multiple experiences. Experiences → Shared Structure
→ Concept → Theories."

A concept here is an online cluster over episodic context vectors, formed by a leader algorithm:
an experience either falls within ``radius`` of an existing centroid and joins it, or it starts a
new provisional cluster. Clusters that never gather ``min_support`` members are swept.

Three properties are deliberate:

* **Concepts must be earned.** A cluster of one is not a concept, it is an episode. Anything below
  ``min_support`` is reported by :attr:`~nyxara.nyx001.layers.types.Concept.is_thin` and never
  quietly presented as an abstraction. This is the same refusal Layer 1 makes about naming, one
  level up.
* **Coherence is the concept's own confidence.** Mean cosine of members to their centroid — a
  loose cluster says so, rather than being reported with the same authority as a tight one.
* **Theories are relations between concepts**, not free-standing claims. :meth:`theories` returns
  pairs of concepts whose members keep appearing in temporal succession, labelled ``precedes``.
  That is a *statistical* relation and it is labelled as one; turning it into a causal claim is
  Layer 7's job and requires intervention, not co-occurrence.

Naming: concepts are named ``c<n>`` plus the most frequent entity key among their members. That is
a *handle*, not an understanding — the name is for a human reading :meth:`stats`, and no layer
above reasons over the string.

Honest: this is online clustering in the perception layer's feature space. It finds structure that
is present in that space and cannot find structure that is not. Because Layer 1's encoder starts as
a hash sketch, early concepts are largely lexical; concepts that track *meaning* emerge only as the
world model reshapes what counts as similar — which is the charter's ordering, and slow.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from nyxara.nyx001.layers.types import Concept, Experience

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["SemanticMemory"]


def _cos(a: Any, b: Any) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    n = min(a.size, b.size)
    return float(np.dot(a[:n], b[:n]) / (na * nb + 1e-12))


class SemanticMemory:
    """Layer 4. Forms concepts from repeated structure, and relations between concepts."""

    def __init__(self, substrate: Any, *, radius: float = 0.55, min_support: int = 3,
                 max_concepts: int = 512, merge_at: float = 0.9) -> None:
        self.substrate = substrate
        self.width = int(getattr(substrate, "width", 256))
        self.radius = float(radius)
        self.min_support = int(min_support)
        self.max_concepts = int(max_concepts)
        self.merge_at = float(merge_at)
        self._concepts: Dict[str, Concept] = {}
        self._sums: Dict[str, Any] = {}
        self._succession: Dict[str, float] = {}
        self._last_concept: Optional[str] = None
        self._n = 0
        self.swept = 0

    # ---- formation ---- #
    def absorb(self, exp: Experience) -> Optional[Concept]:
        """Fold one experience into the concept space. Returns the concept it joined/created."""
        try:
            if np is None or exp is None or exp.context is None:
                return None
            v = np.asarray(exp.context, dtype=np.float64).ravel()
            if float(np.linalg.norm(v)) <= 0.0:
                return None
            self._n += 1
            name, sim = self._nearest(v)
            if name is not None and sim >= self.radius:
                c = self._join(name, v, exp)
            else:
                c = self._create(v, exp)
            if c is not None:
                self._note_succession(c.name)
                self._maybe_merge(c.name)
            return c
        except Exception:  # noqa: BLE001
            return None

    def _nearest(self, v: Any) -> Tuple[Optional[str], float]:
        best, best_sim = None, -1.0
        for name, c in self._concepts.items():
            s = _cos(v, c.centroid)
            if s > best_sim:
                best, best_sim = name, s
        return best, best_sim

    def _join(self, name: str, v: Any, exp: Experience) -> Optional[Concept]:
        c = self._concepts.get(name)
        if c is None:
            return None
        self._sums[name] = self._sums[name] + v
        c.support += 1
        c.centroid = self._sums[name] / c.support
        if exp.key and exp.key not in c.members:
            c.members.append(exp.key)
            if len(c.members) > 64:
                c.members = c.members[-64:]
        c.coherence = self._coherence(name, v)
        return c

    def _coherence(self, name: str, v: Any) -> float:
        """Running mean cosine of members to centroid — the concept's own confidence."""
        c = self._concepts[name]
        s = _cos(v, c.centroid)
        prev = c.coherence
        n = max(1, c.support)
        return float((prev * (n - 1) + s) / n)

    def _create(self, v: Any, exp: Experience) -> Optional[Concept]:
        if len(self._concepts) >= self.max_concepts:
            self._sweep()
            if len(self._concepts) >= self.max_concepts:
                return None
        name = f"c{len(self._concepts)}"
        c = Concept(name=name, centroid=v.copy(), support=1, coherence=1.0,
                    members=[exp.key] if exp.key else [], born=time.time())
        self._concepts[name] = c
        self._sums[name] = v.copy()
        return c

    def _maybe_merge(self, name: str) -> None:
        """Two centroids that have drifted together are one concept that was split by noise."""
        c = self._concepts.get(name)
        if c is None:
            return
        for other_name, other in list(self._concepts.items()):
            if other_name == name:
                continue
            if _cos(c.centroid, other.centroid) >= self.merge_at:
                keep, drop = (c, other) if c.support >= other.support else (other, c)
                self._sums[keep.name] = self._sums[keep.name] + self._sums[drop.name]
                keep.support += drop.support
                keep.centroid = self._sums[keep.name] / max(1, keep.support)
                keep.members = (keep.members + drop.members)[-64:]
                self._concepts.pop(drop.name, None)
                self._sums.pop(drop.name, None)
                return

    def _sweep(self) -> None:
        """Drop provisional clusters that never earned support. Counted, not silent."""
        for name, c in list(self._concepts.items()):
            if c.support < self.min_support:
                self._concepts.pop(name, None)
                self._sums.pop(name, None)
                self.swept += 1

    # ---- theories ---- #
    def _note_succession(self, name: str) -> None:
        if self._last_concept and self._last_concept != name:
            key = f"{self._last_concept}\x00{name}"
            self._succession[key] = self._succession.get(key, 0.0) * 0.995 + 1.0
        self._last_concept = name

    def theories(self, *, k: int = 8, floor: float = 2.0) -> List[Dict[str, Any]]:
        """Concept pairs that keep appearing in succession. Statistical, and labelled as such."""
        out = []
        for key, w in sorted(self._succession.items(), key=lambda kv: kv[1], reverse=True):
            if w < floor:
                break
            a, b = key.split("\x00")
            if a in self._concepts and b in self._concepts:
                out.append({"antecedent": a, "consequent": b, "kind": "precedes",
                            "strength": round(w / (1.0 + w), 4),
                            "note": "temporal succession, not causation"})
            if len(out) >= k:
                break
        return out

    # ---- read ---- #
    def concepts(self, *, include_thin: bool = False) -> List[Concept]:
        """Earned concepts, strongest first. Thin ones are excluded unless explicitly asked for."""
        items = [c for c in self._concepts.values() if include_thin or not c.is_thin]
        return sorted(items, key=lambda c: (c.support, c.coherence), reverse=True)

    def classify(self, vec: Any) -> Tuple[Optional[Concept], float]:
        """Which concept a vector falls under, with the similarity. ``(None, 0.0)`` if none does."""
        try:
            if np is None or vec is None or not self._concepts:
                return None, 0.0
            v = np.asarray(vec, dtype=np.float64).ravel()
            name, sim = self._nearest(v)
            if name is None or sim < self.radius:
                return None, float(max(0.0, sim))
            return self._concepts[name], float(sim)
        except Exception:  # noqa: BLE001
            return None, 0.0

    def stats(self) -> Dict[str, Any]:
        earned = self.concepts()
        return {"absorbed": self._n, "concepts": len(self._concepts), "earned": len(earned),
                "thin": len(self._concepts) - len(earned), "swept": self.swept,
                "theories": len(self.theories(k=1000)),
                "mean_coherence": round(float(sum(c.coherence for c in earned) / len(earned)), 4)
                if earned else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self._n, "swept": self.swept,
                "concepts": [c.to_dict() for c in self.concepts(include_thin=True)[:128]],
                "theories": self.theories(k=32)}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self._n = int(d.get("n", 0))
            self.swept = int(d.get("swept", 0))
        except Exception:  # noqa: BLE001 — centroids are not serialised; concepts re-form
            pass
