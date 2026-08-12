"""NYXARA · nyx001/layers/l01_perception.py — Layer 1, perception (👁, raw → structure).

The charter: "Extracts useful representations automatically from raw observations. Raw input →
feature dynamics → entities/relations."

Three stages, and the middle one is the part that is usually skipped:

1. **Encode.** An observation becomes a fixed-width vector. For text this is a learned-projection
   over a rolling character-and-token sketch — *not* a pretrained embedding, because there is not
   one anywhere in this package. For an ndarray it is a random projection to the substrate width,
   which preserves distances (Johnson–Lindenstrauss) without any training at all.
2. **Feature dynamics.** The encoding is passed through a decaying trace of recent encodings, so
   what Layer 2 receives is not a snapshot but a *velocity*: what changed, not merely what is.
   The charter names this stage explicitly and it is the one that makes entity boundaries findable
   — a thing is what stays correlated while the surroundings move.
3. **Carve.** Entities are the components of the encoding that are simultaneously high-magnitude
   and *stable across the trace*; relations are pairs of entities that keep co-activating. Both
   are claims with weights, never assertions.

What this layer refuses to do is as important as what it does. It does not name anything it has
not seen repeat — an entity needs ``min_support`` observations before it gets a key. Naming a
one-off is how a perception layer manufactures structure that is not there, and every layer above
would then reason about it in good faith.

Honest, and this one matters: the text encoder is a **hash-and-project sketch**, not a semantic
embedding. Two sentences that mean the same thing in different words will *not* land near each
other here at birth. They can only come to do so through Layer 4, after the world model has
learned that they predict the same things. That is the charter's own ordering — Environment →
Perception → Concepts → World Model → Language — and it means early perception is genuinely
shallow rather than secretly borrowing meaning from somewhere.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from nyxara.nyx001.layers.types import Entity, Observation, Percept, Relation
from nyxara.nyx001.substrate import HAS_NUMPY, Tensor, xavier

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["Perception"]

_TOKEN_SPLIT = " \t\n\r.,;:!?()[]{}\"'`/\\|<>=+-*&^%$#@~"


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    for ch in str(text).lower():
        if ch in _TOKEN_SPLIT:
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _bucket(token: str, width: int) -> int:
    """A stable bucket for a token. Hashing, not a vocabulary — nothing human-authored enters."""
    h = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(h, "big") % max(1, width)


class Perception:
    """Layer 1. Turns observations into percepts with feature dynamics and carved structure."""

    def __init__(self, substrate: Any, *, trace_decay: float = 0.6, min_support: int = 2,
                 max_entities: int = 4096, entity_gate: float = 0.6,
                 relation_gate: float = 0.35) -> None:
        self.substrate = substrate
        self.width = int(getattr(substrate, "width", 256))
        self.trace_decay = float(trace_decay)
        self.min_support = int(min_support)
        self.max_entities = int(max_entities)
        self.entity_gate = float(entity_gate)
        self.relation_gate = float(relation_gate)
        self.available = bool(HAS_NUMPY)

        self._trace: Any = None                      # decaying mean of recent encodings
        self._trace_sq: Any = None                   # decaying mean of squares — for stability
        self._seen: Dict[str, Entity] = {}
        self._pairs: Dict[str, float] = {}
        self._n = 0
        self._proj: Optional[Tensor] = None          # lazily built; shape depends on input dim
        self._proj_in = 0
        self._rng = getattr(substrate, "stream", lambda _n: None)("perception")

    # ---- stage 1: encode ---- #
    def _encode_text(self, text: str) -> Any:
        """A hash-and-project sketch. Deterministic, untrained, and honestly shallow."""
        vec = np.zeros(self.width, dtype=np.float64)
        toks = _tokens(text)
        if not toks:
            return vec
        for i, tok in enumerate(toks):
            b = _bucket(tok, self.width)
            vec[b] += 1.0
            # a positional smear, so word order is not entirely discarded
            vec[(b + 1 + (i % 3)) % self.width] += 0.3
        for a, b in zip(toks, toks[1:]):             # bigrams: the cheapest structure available
            vec[_bucket(a + "\x00" + b, self.width)] += 0.5
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _encode_array(self, arr: Any) -> Any:
        """Random projection to substrate width — distance-preserving, and needs no training."""
        flat = np.asarray(arr, dtype=np.float64).ravel()
        if flat.size == 0:
            return np.zeros(self.width, dtype=np.float64)
        if flat.size == self.width:
            out = flat
        else:
            if self._proj is None or self._proj_in != flat.size:
                rng = self._rng if self._rng is not None else np.random.default_rng(0)
                self._proj = xavier(rng, int(flat.size), self.width)
                self._proj_in = int(flat.size)
            out = flat @ self._proj.data
        norm = float(np.linalg.norm(out))
        return out / norm if norm > 0 else out

    def encode(self, obs: Observation) -> Any:
        """Observation → fixed-width vector. Unknown modalities encode their repr, not nothing."""
        payload = obs.payload
        if payload is None:
            return np.zeros(self.width, dtype=np.float64)
        if obs.modality == "vector" or hasattr(payload, "__array__"):
            return self._encode_array(payload)
        if isinstance(payload, (int, float)):
            v = np.zeros(self.width, dtype=np.float64)
            v[0] = float(payload)
            return v
        return self._encode_text(str(payload))

    # ---- stage 2: feature dynamics ---- #
    def _update_trace(self, vec: Any) -> Any:
        """Return the *change* against the decaying trace, and advance the trace.

        The returned delta is what Layer 2 predicts over. Predicting the raw encoding would let
        the world model score well by learning the input distribution's mean and nothing else —
        a genuinely common way for a predictive model to look competent while modelling nothing.
        """
        if self._trace is None:
            self._trace = vec.copy()
            self._trace_sq = vec * vec
            return np.zeros_like(vec)
        delta = vec - self._trace
        d = self.trace_decay
        self._trace = d * self._trace + (1.0 - d) * vec
        self._trace_sq = d * self._trace_sq + (1.0 - d) * (vec * vec)
        return delta

    def _stability(self) -> Any:
        """Per-component 1/(1+variance) — high where a component holds still across the trace."""
        if self._trace is None:
            return None
        var = np.maximum(self._trace_sq - self._trace * self._trace, 0.0)
        return 1.0 / (1.0 + var)

    # ---- stage 3: carve ---- #
    def _carve(self, obs: Observation, vec: Any) -> List[Entity]:
        """Entities: components that are both loud and stable, and that have repeated."""
        out: List[Entity] = []
        if obs.modality == "text" and isinstance(obs.payload, str):
            keys = [t for t in _tokens(obs.payload) if len(t) > 1]
        else:
            stab = self._stability()
            if stab is None:
                return out
            score = np.abs(vec) * stab
            if float(score.max(initial=0.0)) <= 0.0:
                return out
            thresh = float(score.max()) * self.entity_gate
            keys = [f"c{i}" for i in np.nonzero(score >= thresh)[0][:16]]
        for key in keys:
            ent = self._seen.get(key)
            if ent is None:
                if len(self._seen) >= self.max_entities:
                    self._evict()
                ent = Entity(key=key, vec=vec.copy(), salience=0.0, count=0)
                self._seen[key] = ent
            ent.count += 1
            ent.vec = 0.9 * ent.vec + 0.1 * vec          # the entity drifts toward its contexts
            ent.salience = min(1.0, ent.count / max(1.0, self.min_support * 4.0))
            if ent.count >= self.min_support:            # only a repeat earns a name
                out.append(ent)
        return out[:16]

    def _evict(self) -> None:
        """Weakest-first eviction. This store forgets, on purpose and measurably."""
        if not self._seen:
            return
        victim = min(self._seen.values(), key=lambda e: (e.count, e.first_seen))
        self._seen.pop(victim.key, None)

    def _relate(self, entities: List[Entity]) -> List[Relation]:
        """Relations: pairs that keep co-activating. Co-occurrence, and labelled as such."""
        rels: List[Relation] = []
        keys = [e.key for e in entities][:8]
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                pair = f"{a}\x00{b}" if a < b else f"{b}\x00{a}"
                self._pairs[pair] = self._pairs.get(pair, 0.0) * 0.99 + 1.0
                w = self._pairs[pair] / (1.0 + self._pairs[pair])
                if w >= self.relation_gate:
                    src, dst = pair.split("\x00")
                    rels.append(Relation(src=src, dst=dst, kind="co-occurs", weight=round(w, 4)))
        if len(self._pairs) > self.max_entities * 4:
            keep = sorted(self._pairs.items(), key=lambda kv: kv[1], reverse=True)
            self._pairs = dict(keep[: self.max_entities * 2])
        return rels[:24]

    # ---- the layer ---- #
    def perceive(self, obs: Observation) -> Percept:
        """Run all three stages. Fail-soft: on any error, an empty percept rather than a crash."""
        try:
            if not self.available:
                return Percept(t=time.time())
            vec = self.encode(obs)
            delta = self._update_trace(vec)
            self._n += 1
            entities = self._carve(obs, vec)
            relations = self._relate(entities)
            novelty = float(np.linalg.norm(delta) / (1.0 + np.linalg.norm(vec)))
            return Percept(vec=vec, entities=entities, relations=relations,
                           novelty=min(1.0, novelty), t=obs.t)
        except Exception:  # noqa: BLE001 — perception never breaks a cycle
            return Percept(t=time.time())

    def dynamics(self) -> Optional[Any]:
        """The current feature trace — what Layer 2 uses as ``s_t`` when nothing better exists."""
        return None if self._trace is None else self._trace.copy()

    def stats(self) -> Dict[str, Any]:
        return {"available": self.available, "observations": self._n,
                "entities": len(self._seen), "pairs": len(self._pairs), "width": self.width,
                "named": sum(1 for e in self._seen.values() if e.count >= self.min_support)}

    def to_dict(self) -> Dict[str, Any]:
        top = sorted(self._seen.values(), key=lambda e: e.count, reverse=True)[:64]
        return {"n": self._n, "entities": [{"key": e.key, "count": e.count} for e in top]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self._n = int(d.get("n", 0))
            for item in d.get("entities") or []:
                key = str(item.get("key", ""))
                if key:
                    self._seen[key] = Entity(key=key, vec=np.zeros(self.width),
                                             count=int(item.get("count", 1)))
        except Exception:  # noqa: BLE001
            pass
