"""NYXARA · nyx5/concept_space.py — N-dimensional concept collapse (🌌, pillar 10).

A knot that looks hopeless in three dimensions can be a straight line in ten thousand. NYX-5 maps a
structured problem — a set of role→filler relations — into the existing 10,000-D hyperdimensional space,
where relational structure that is tangled in low dimensions becomes ordinary algebra. A record is a
bundle of ``role ⊗ filler`` bindings; asking "what fills this role?" is a single unbind + cleanup, and
analogies (``A : B :: C : ?``) are solved by transporting one record's mapping onto another term — all
by vector arithmetic, no search.

Honest ceiling: high-dimensional representation *simplifies some* structure — it does not dissolve every
paradox instantly. Where the collapse does not yield a confident answer, :class:`ConceptCollapser`
**abstains** rather than inventing one. Reuses the audited HDC algebra
(:mod:`nyxara.cognition.hyper_dimensional_vectors`); it builds no second VSA.

Pure standard library on top of the HDC module.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from nyxara.cognition.hyper_dimensional_vectors import HyperSpace, Hypervector, ItemMemory

__all__ = ["ConceptCollapser"]


class ConceptCollapser:
    """Encode role→filler records in hyperspace; query roles and solve analogies by algebra."""

    def __init__(self, *, dim: int = 10000, seed: int = 42, min_confidence: float = 0.15) -> None:
        self.space = HyperSpace(dim=dim, seed=seed)
        self.atoms = ItemMemory(self.space)
        self.min_confidence = float(min_confidence)

    def encode_record(self, record: Mapping[str, str]) -> Hypervector:
        """Bundle a set of ``{role: filler}`` bindings into one holographic record vector."""
        bound = [self.space.bind(self.atoms.symbol(f"role:{r}"), self.atoms.symbol(f"val:{v}"))
                 for r, v in record.items()]
        return self.space.bundle(*bound) if bound else self.space.zero()

    def query(self, record: Mapping[str, str], role: str) -> Optional[Tuple[str, float]]:
        """Recover the filler bound to ``role`` in ``record``; abstain if not confident."""
        rec = self.encode_record(record)
        noisy = self.space.unbind(rec, self.atoms.symbol(f"role:{role}"))
        result = self.atoms.cleanup(noisy, exclude=[f"role:{r}" for r in record])
        if result.name is None or not result.name.startswith("val:"):
            return None
        if result.score < self.min_confidence:
            return None                                  # collapse did not yield a confident answer
        return result.name[len("val:"):], result.score

    def analogy(self, source: Mapping[str, str], target: Mapping[str, str],
                role: str) -> Optional[Tuple[str, float]]:
        """Solve ``source[role] : source :: target : ?`` — transport the mapping onto ``target``.

        Classic example: source={capital: washington, currency: dollar, country: usa},
        target={capital: mexico_city, country: mexico}, role via the shared relation.
        """
        src = self.encode_record(source)
        tgt = self.encode_record(target)
        mapping = self.space.bind(src, tgt)              # the transform between the two records
        transported = self.space.bind(mapping, self.atoms.symbol(f"val:{source.get(role, '')}"))
        result = self.atoms.cleanup(transported,
                                    exclude=[f"val:{source.get(role, '')}"])
        if result.name is None or not result.name.startswith("val:"):
            return None
        if result.score < self.min_confidence:
            return None
        return result.name[len("val:"):], result.score

    def to_dict(self) -> Dict[str, Any]:
        return {"dim": self.space.dim, "seed": self.space.seed, "min_confidence": self.min_confidence}
