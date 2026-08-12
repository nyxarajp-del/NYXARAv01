"""NYXARA · nyx001/layers/types.py — what the eighteen layers pass between them (🔗).

A "modular cognitive architecture" is only modular if the modules agree on what they hand each
other. This file is that agreement: seven dataclasses that are the *only* things allowed to cross
a layer boundary. A layer may hold whatever internal state it likes; it may not reach into
another layer's internals.

The flow the charter describes — observe → model → act → consequence → error → learn — becomes:

    Observation → (L1) → Percept → (L2) → Prediction → Experience → (L3/L4) → Concept
                                              ↓
                          Hypothesis ← (L6/L10) ← Belief → (L8) → Plan → Action

One field deserves its own note, because it is the spine of the whole design:
:attr:`Experience.prediction_error`. The charter says episodic memory must store "what, when,
context, prediction error, and future relevance" — and prediction error is what makes the
difference between a memory system and a log. An episode that surprised the world model is worth
more than one that confirmed it, so error is what drives replay priority, what gates concept
formation, and what the curiosity engine hunts. Everything downstream reads it.

Every dataclass carries ``to_dict()`` and no dataclass carries a NumPy array in ``to_dict()`` —
serialisation is for inspection and persistence, and a 256-wide float64 vector in a JSON sidecar
is neither inspectable nor small. Vectors live in ``vec`` fields that ``to_dict`` summarises.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = [
    "Observation", "Percept", "Entity", "Relation", "Prediction", "Experience",
    "Concept", "Hypothesis", "Belief", "Plan", "Action", "LayerReport",
    "vec_summary",
]


def vec_summary(v: Any) -> Optional[Dict[str, float]]:
    """A vector's shape and moments — what a sidecar should carry instead of the vector."""
    if v is None or np is None:
        return None
    try:
        arr = np.asarray(v, dtype=np.float64)
        if arr.size == 0:
            return None
        return {"dim": int(arr.size), "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std()), 6), "norm": round(float(np.linalg.norm(arr)), 6)}
    except Exception:  # noqa: BLE001 — a summary is never worth breaking a turn for
        return None


@dataclass
class Observation:
    """Raw input, before any interpretation. The only thing that enters from outside."""

    modality: str = "text"                  # text | vector | scalar | action | reward
    payload: Any = None                     # the raw thing (str, ndarray, float …)
    t: float = field(default_factory=time.time)
    source: str = ""                        # who produced it, for provenance

    def to_dict(self) -> Dict[str, Any]:
        kind = type(self.payload).__name__
        return {"modality": self.modality, "t": round(self.t, 3), "source": self.source,
                "payload_type": kind}


@dataclass
class Entity:
    """A thing Layer 1 carved out of an observation. Identity is a claim, not a fact."""

    key: str = ""
    vec: Any = None
    salience: float = 0.0
    first_seen: float = field(default_factory=time.time)
    count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "salience": round(self.salience, 4), "count": self.count,
                "vec": vec_summary(self.vec)}


@dataclass
class Relation:
    """An observed co-occurrence between two entities. Not yet a cause — see Layer 7."""

    src: str = ""
    dst: str = ""
    kind: str = "co-occurs"
    weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "kind": self.kind,
                "weight": round(self.weight, 4)}


@dataclass
class Percept:
    """Layer 1's output: features, plus whatever structure it could justify carving out."""

    vec: Any = None                         # the feature vector — the substrate's view
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    novelty: float = 0.0                    # how unlike anything already seen
    t: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"novelty": round(self.novelty, 4), "n_entities": len(self.entities),
                "n_relations": len(self.relations), "vec": vec_summary(self.vec),
                "entities": [e.to_dict() for e in self.entities[:8]]}


@dataclass
class Prediction:
    """Layer 2's ``ŝ_{t+1} = F_θ(s_t, a_t)`` and, once the world answers, its error."""

    state: Any = None                       # ŝ_{t+1}
    horizon: int = 1
    error: Optional[float] = None           # e_t = D(s_{t+1}, ŝ_{t+1}); None until observed
    uncertainty: float = 0.0

    @property
    def resolved(self) -> bool:
        """Whether reality has answered yet. ``error is None`` is not the same as zero error."""
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        return {"horizon": self.horizon,
                "error": None if self.error is None else round(self.error, 6),
                "uncertainty": round(self.uncertainty, 4), "state": vec_summary(self.state)}


@dataclass
class Experience:
    """One episode. The charter's five fields, and ``prediction_error`` is the spine."""

    key: str = ""
    what: str = ""                          # a short human-readable handle
    when: float = field(default_factory=time.time)
    context: Any = None                     # the state vector at the time
    prediction_error: float = 0.0           # how surprising this was — drives replay priority
    future_relevance: float = 0.0           # learned, not assigned; see Layer 3
    action: Optional[str] = None
    outcome: Optional[str] = None
    replays: int = 0

    @property
    def priority(self) -> float:
        """Replay priority: surprise now, usefulness later, decayed by how often it was replayed.

        The ``1/(1+replays)`` damping is what stops one shocking episode from monopolising every
        replay batch forever — a real failure mode in prioritised replay, where the highest-error
        sample is drawn until it is memorised and the rest of experience starves.
        """
        base = 0.7 * self.prediction_error + 0.3 * self.future_relevance
        return float(base / (1.0 + 0.5 * self.replays))

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "what": self.what[:120], "when": round(self.when, 3),
                "prediction_error": round(self.prediction_error, 6),
                "future_relevance": round(self.future_relevance, 4),
                "replays": self.replays, "priority": round(self.priority, 6),
                "action": self.action, "outcome": self.outcome}


@dataclass
class Concept:
    """Layer 4's abstraction: the shared structure of several experiences, with its support."""

    name: str = ""
    centroid: Any = None
    support: int = 0                        # how many experiences it was formed from
    coherence: float = 0.0                  # how tight the cluster is — its own confidence
    members: List[str] = field(default_factory=list)
    born: float = field(default_factory=time.time)

    @property
    def is_thin(self) -> bool:
        """A concept formed from too few experiences to be trusted. Reported, not hidden."""
        return self.support < 3

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "support": self.support,
                "coherence": round(self.coherence, 4), "thin": self.is_thin,
                "members": list(self.members[:8]), "centroid": vec_summary(self.centroid)}


@dataclass
class Hypothesis:
    """A claim the reasoning engine is entertaining, with the evidence for and against it."""

    claim: str = ""
    confidence: float = 0.5
    support: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    tested: int = 0
    verifiable: bool = False                # True only for a checked derivation, never a guess

    @property
    def survived(self) -> bool:
        """Held up so far — tested at least once and not contradicted."""
        return self.tested > 0 and not self.contradictions

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim[:200], "confidence": round(self.confidence, 4),
                "support": len(self.support), "contradictions": len(self.contradictions),
                "tested": self.tested, "verifiable": self.verifiable, "survived": self.survived}


@dataclass
class Belief:
    """What the system currently holds, and how sure it is — Layer 12 reads this."""

    subject: str = ""
    value: Any = None
    confidence: float = 0.5
    evidence_count: int = 0
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "confidence": round(self.confidence, 4),
                "evidence_count": self.evidence_count,
                "last_update": round(self.last_update, 3)}


@dataclass
class Action:
    """Something the system proposes to do. It only ever proposes — the kernel disposes."""

    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    expected_gain: float = 0.0
    risk: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": dict(self.args),
                "expected_gain": round(self.expected_gain, 4), "risk": round(self.risk, 4)}


@dataclass
class Plan:
    """Layer 8's decomposition: a goal, its sub-goals, and the simulated cost of each."""

    goal: str = ""
    steps: List[Action] = field(default_factory=list)
    simulated_value: float = 0.0
    risk: float = 0.0
    horizon: int = 0
    feasible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal[:200], "n_steps": len(self.steps),
                "simulated_value": round(self.simulated_value, 4), "risk": round(self.risk, 4),
                "horizon": self.horizon, "feasible": self.feasible,
                "steps": [s.to_dict() for s in self.steps[:8]]}


@dataclass
class LayerReport:
    """What one layer did on one cycle. The stack collects these; nothing else is claimed."""

    layer: int = 0
    name: str = ""
    ran: bool = False
    ms: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"layer": self.layer, "name": self.name, "ran": self.ran,
                             "ms": round(self.ms, 3)}
        if self.detail:
            d["detail"] = self.detail
        if self.error:
            d["error"] = self.error
        return d
