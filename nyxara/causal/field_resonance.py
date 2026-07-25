"""NYXARA · causal/field_resonance.py — Field-Resonance Dynamics (CAUSAL · 1).

*The Master's ask:* let NYXARA treat information as **continuous waveform fields** rather than
discrete tokens, and answer a query by **resonating** it against those fields instead of grinding
token-by-token.

What that means here, honestly. Each concept, code structure, theorem, or Master-preference is
embedded once into a fixed-dimensional real vector — a *field* / spectral signature. A query is
embedded into the **same** space, and :meth:`ResonanceField.resonate` computes the **interference**
of the query field with every stored field as a normalised inner product (cosine). The
maximally-resonant field(s) are returned with a resonance score in ``[-1, 1]``. Constructive
interference (aligned fields) → high score; destructive → low.

The honest boundary. This is a real, deterministic, vectorised similarity engine — genuinely fast
(one dot product per stored field; sub-millisecond for a bounded bank of a few thousand). It is
**not** ``O(1)`` "at the speed of physics" and it does **not** make GPU load literally zero — it is
``O(bank_size × dim)`` per query, done in cheap CPU floats. The embedding is a signed
feature-hash of character n-grams (the "spectral decomposition"): dependency-free and stable, so
the same text always maps to the same field. When numpy is present it is used as an accelerator;
without it the pure-Python path gives identical results.

Bridges: mirrors the vector-symbolic spirit of :mod:`nyxara.memory.holographic_field` and
:mod:`nyxara.cognition.hyper_dimensional_vectors`, but stands alone (pure stdlib) so it is usable
and testable with no optional dependency.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # numpy is an accelerator only; the pure-Python path below is the source of truth.
    import numpy as _np  # type: ignore
except Exception:  # noqa: BLE001
    _np = None

__all__ = ["ConceptField", "ResonanceHit", "ResonanceField", "embed_field"]

_DEFAULT_DIM = 512


# --------------------------------------------------------------------------- #
# Embedding — a stable signed feature-hash "spectral signature" of text
# --------------------------------------------------------------------------- #
def _tokens(text: str, *, n: int = 3) -> Iterable[str]:
    """Character n-grams + words — the 'harmonics' whose sum is the field waveform."""
    t = " ".join(text.lower().split())
    if not t:
        return []
    grams: List[str] = t.split()  # word harmonics
    padded = f"  {t}  "
    grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))  # char harmonics
    return grams


def _bucket(token: str, dim: int) -> Tuple[int, float]:
    """Map a token to (index, sign) via a stable hash — its place & phase in the field."""
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(h[:6], "big") % dim
    sign = 1.0 if (h[6] & 1) else -1.0
    return idx, sign


def embed_field(text: str, *, dim: int = _DEFAULT_DIM) -> List[float]:
    """Embed ``text`` into a unit-norm real field vector of length ``dim`` (pure stdlib)."""
    vec = [0.0] * dim
    for tok in _tokens(text):
        idx, sign = _bucket(tok, dim)
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Interference of two unit(ish) fields — their normalised inner product."""
    if _np is not None:
        va, vb = _np.asarray(a, dtype=float), _np.asarray(b, dtype=float)
        na, nb = float(_np.linalg.norm(va)), float(_np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(va @ vb / (na * nb))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class ConceptField:
    """One concept stored as a continuous waveform field."""

    label: str
    vector: List[float]
    payload: Any = None
    weight: float = 1.0  # standing amplitude — how strongly this field radiates

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "dim": len(self.vector),
                "weight": round(self.weight, 4), "payload": self.payload}


@dataclass
class ResonanceHit:
    """A resonance measurement between the query field and one stored concept."""

    label: str
    score: float          # interference amplitude in [-1, 1]
    weighted: float       # score × standing amplitude (ranking key)
    payload: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "score": round(self.score, 6),
                "weighted": round(self.weighted, 6), "payload": self.payload}


# --------------------------------------------------------------------------- #
# The field
# --------------------------------------------------------------------------- #
class ResonanceField:
    """A bounded bank of concept fields answered by interference, not iteration."""

    def __init__(self, *, dim: int = _DEFAULT_DIM, capacity: int = 4096,
                 resonance_floor: float = 0.15) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.resonance_floor = float(resonance_floor)
        self._fields: Dict[str, ConceptField] = {}
        self._order: List[str] = []  # insertion order for bounded eviction (LRU-ish)

    # -- population --------------------------------------------------------- #
    def imprint(self, label: str, text: Optional[str] = None, *, payload: Any = None,
                weight: float = 1.0, vector: Optional[Sequence[float]] = None) -> ConceptField:
        """Imprint a concept as a field. Pass ``text`` (embedded) or a raw ``vector``."""
        if vector is not None:
            vec = list(map(float, vector))
            if len(vec) != self.dim:
                raise ValueError(f"vector dim {len(vec)} != field dim {self.dim}")
        else:
            vec = embed_field(text if text is not None else label, dim=self.dim)
        cf = ConceptField(label=label, vector=vec, payload=payload, weight=float(weight))
        if label not in self._fields:
            self._order.append(label)
        self._fields[label] = cf
        self._evict_if_needed()
        return cf

    def imprint_many(self, items: Mapping[str, Any]) -> int:
        """Imprint ``{label: text}`` (or ``{label: (text, payload)}``) in one call."""
        n = 0
        for label, val in items.items():
            if isinstance(val, tuple) and len(val) == 2:
                self.imprint(label, val[0], payload=val[1])
            else:
                self.imprint(label, str(val))
            n += 1
        return n

    def forget(self, label: str) -> bool:
        if label in self._fields:
            del self._fields[label]
            self._order = [l for l in self._order if l != label]
            return True
        return False

    def _evict_if_needed(self) -> None:
        while len(self._order) > self.capacity:
            oldest = self._order.pop(0)
            self._fields.pop(oldest, None)

    # -- resonance ---------------------------------------------------------- #
    def resonate(self, query: str, *, top: int = 3,
                 vector: Optional[Sequence[float]] = None) -> List[ResonanceHit]:
        """Resonate ``query`` against the bank; return the top hits above the floor."""
        if not self._fields:
            return []
        qvec = list(map(float, vector)) if vector is not None else embed_field(query, dim=self.dim)
        hits: List[ResonanceHit] = []
        for cf in self._fields.values():
            score = _cosine(qvec, cf.vector)
            weighted = score * cf.weight
            if score >= self.resonance_floor:
                hits.append(ResonanceHit(cf.label, score, weighted, cf.payload))
        hits.sort(key=lambda h: h.weighted, reverse=True)
        return hits[:max(1, top)]

    def best(self, query: str) -> Optional[ResonanceHit]:
        """The single most-resonant concept, or ``None`` if nothing clears the floor."""
        hits = self.resonate(query, top=1)
        return hits[0] if hits else None

    def is_gap(self, query: str) -> bool:
        """True when nothing in the bank resonates — the phase-shift trigger."""
        return self.best(query) is None

    def reinforce(self, label: str, *, by: float = 0.1, cap: float = 8.0) -> None:
        """Strengthen a field's standing amplitude when it proved useful (Hebbian-ish)."""
        cf = self._fields.get(label)
        if cf is not None:
            cf.weight = min(cap, cf.weight + by)

    # -- introspection ------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._fields)

    def labels(self) -> List[str]:
        return list(self._order)

    def status(self) -> Dict[str, Any]:
        return {"fields": len(self._fields), "dim": self.dim, "capacity": self.capacity,
                "resonance_floor": self.resonance_floor}
