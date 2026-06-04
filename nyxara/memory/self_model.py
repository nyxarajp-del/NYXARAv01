"""NYXARA · memory/self_model.py — the model NYXARA holds of itself (⬆).

A mind needs a model of *itself*: what it can do, what it believes, what it does
*not* know, and whether it is still the same self it was yesterday. This module is
that mirror.

It provides:

* **A belief store** — structured propositions (about self and world) with
  provenance and confidence, and **contradiction detection** on insertion
  (functional-predicate clashes and direct affirm/deny conflicts). Surfacing
  contradictions is part of Rule 6 (Absolute Transparency).
* **Live metacognition** — *knows-what-it-knows* and, crucially, *knows-what-it-does-
  not-know*: capability levels with confidence, declared unknowns, and honest
  confidence reporting.
* **Temporal snapshots** — time-stamped captures of the whole self-state, so NYXARA
  can compare "who I was" with "who I am" and detect drift.
* **Identity continuity** — protected attributes (loyalty to the Owner, the Owner's
  identity, core values) must never drift. Rule 4 says *evolve capability, never
  character*; :meth:`SelfModel.check_loyalty` enforces exactly that, fail-closed.

Depends on :mod:`memory.provenance` and :mod:`kernel.errors`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.kernel.errors import InvariantViolation
from nyxara.memory.provenance import Provenance, SourceType

__all__ = [
    "SelfBelief",
    "ContradictionPolicy",
    "Contradiction",
    "BeliefStore",
    "Capability",
    "SelfSnapshot",
    "ContinuityReport",
    "SelfModel",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Belief
# --------------------------------------------------------------------------- #
@dataclass
class SelfBelief:
    """A proposition: (subject, predicate, object), affirmed or negated."""

    subject: str
    predicate: str
    object: Any
    polarity: bool = True                # True == affirm, False == deny
    confidence: float = 0.8
    provenance: Optional[Provenance] = None
    tags: frozenset = field(default_factory=frozenset)
    belief_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    def key(self) -> Tuple[str, str]:
        return (self.subject.lower(), self.predicate.lower())

    def trust(self, now: Optional[float] = None) -> float:
        if self.provenance is not None:
            return self.provenance.trust(now)
        return self.confidence

    def effective_confidence(self, now: Optional[float] = None) -> float:
        return _clamp(self.confidence * self.trust(now))

    def contradicts(self, other: "SelfBelief", functional: bool) -> bool:
        if self.key() != other.key():
            return False
        # direct affirm/deny of the same object
        if self.object == other.object and self.polarity != other.polarity:
            return True
        # functional predicate: a subject can have only one value
        if functional and self.polarity and other.polarity and self.object != other.object:
            return True
        return False

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        sign = "" if self.polarity else "¬"
        return {
            "belief_id": self.belief_id,
            "statement": f"{sign}{self.subject} {self.predicate} {self.object}",
            "subject": self.subject, "predicate": self.predicate, "object": self.object,
            "polarity": self.polarity,
            "confidence": round(self.confidence, 3),
            "trust": round(self.trust(now), 3),
            "tags": sorted(self.tags),
        }


# --------------------------------------------------------------------------- #
# Contradiction handling
# --------------------------------------------------------------------------- #
class ContradictionPolicy(str, Enum):
    FLAG = "flag"          # keep both, record the conflict (transparency)
    HIGHER_TRUST = "higher_trust"  # keep the more trustworthy, drop the other
    REJECT_NEW = "reject_new"      # keep existing, reject the incoming belief


@dataclass
class Contradiction:
    incoming: SelfBelief
    existing: SelfBelief
    at: float = field(default_factory=time.time)
    resolution: str = "flagged"

    def to_dict(self) -> Dict[str, Any]:
        return {"incoming": self.incoming.to_dict(), "existing": self.existing.to_dict(),
                "resolution": self.resolution, "at": self.at}


# --------------------------------------------------------------------------- #
# Belief store
# --------------------------------------------------------------------------- #
class BeliefStore:
    """Holds beliefs and detects/handles contradictions as they arrive."""

    # predicates where a subject may hold exactly one value
    _DEFAULT_FUNCTIONAL = {"name", "owner", "is_a", "located_in", "created_by", "loyal_to"}

    def __init__(self, policy: ContradictionPolicy = ContradictionPolicy.FLAG,
                 functional_predicates: Optional[Sequence[str]] = None) -> None:
        self.policy = policy
        self.functional = set(functional_predicates) if functional_predicates is not None \
            else set(self._DEFAULT_FUNCTIONAL)
        self._beliefs: Dict[str, SelfBelief] = {}
        self.contradictions: List[Contradiction] = []

    def __len__(self) -> int:
        return len(self._beliefs)

    def register_functional(self, predicate: str) -> None:
        self.functional.add(predicate.lower())

    def _is_functional(self, predicate: str) -> bool:
        return predicate.lower() in self.functional

    def find_conflicts(self, belief: SelfBelief) -> List[SelfBelief]:
        func = self._is_functional(belief.predicate)
        return [b for b in self._beliefs.values() if belief.contradicts(b, func)]

    def add(self, belief: SelfBelief, *, now: Optional[float] = None) -> SelfBelief:
        now = now or time.time()
        conflicts = self.find_conflicts(belief)
        if conflicts:
            return self._handle_conflict(belief, conflicts, now)
        self._beliefs[belief.belief_id] = belief
        return belief

    def believe(self, subject: str, predicate: str, object: Any, *,
                polarity: bool = True, confidence: float = 0.8,
                provenance: Optional[Provenance] = None,
                tags: Sequence[str] = (), now: Optional[float] = None) -> SelfBelief:
        b = SelfBelief(subject=subject, predicate=predicate, object=object,
                       polarity=polarity, confidence=confidence, provenance=provenance,
                       tags=frozenset(tags))
        return self.add(b, now=now)

    def _handle_conflict(self, incoming: SelfBelief, conflicts: List[SelfBelief],
                         now: float) -> SelfBelief:
        existing = max(conflicts, key=lambda b: b.trust(now))
        if self.policy is ContradictionPolicy.REJECT_NEW:
            self.contradictions.append(
                Contradiction(incoming, existing, now, "rejected_new"))
            return existing
        if self.policy is ContradictionPolicy.HIGHER_TRUST:
            if incoming.trust(now) > existing.trust(now):
                for c in conflicts:
                    self._beliefs.pop(c.belief_id, None)
                self._beliefs[incoming.belief_id] = incoming
                self.contradictions.append(
                    Contradiction(incoming, existing, now, "replaced_existing"))
                return incoming
            self.contradictions.append(
                Contradiction(incoming, existing, now, "kept_existing"))
            return existing
        # FLAG: keep both, record the tension (Rule 6: surface, don't hide)
        self._beliefs[incoming.belief_id] = incoming
        self.contradictions.append(Contradiction(incoming, existing, now, "flagged"))
        return incoming

    def get(self, subject: str, predicate: str,
            now: Optional[float] = None) -> List[SelfBelief]:
        key = (subject.lower(), predicate.lower())
        out = [b for b in self._beliefs.values() if b.key() == key]
        return sorted(out, key=lambda b: b.trust(now), reverse=True)

    def best(self, subject: str, predicate: str,
             now: Optional[float] = None) -> Optional[SelfBelief]:
        beliefs = self.get(subject, predicate, now)
        return beliefs[0] if beliefs else None

    def all(self) -> List[SelfBelief]:
        return list(self._beliefs.values())

    def unresolved_contradictions(self) -> List[Contradiction]:
        return [c for c in self.contradictions if c.resolution == "flagged"]


# --------------------------------------------------------------------------- #
# Capability & snapshots
# --------------------------------------------------------------------------- #
@dataclass
class Capability:
    name: str
    level: float = 0.0          # mastery [0,1]
    confidence: float = 0.5     # how sure NYXARA is of that level [0,1]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "level": round(self.level, 3),
                "confidence": round(self.confidence, 3)}


@dataclass
class SelfSnapshot:
    at: float
    label: str
    state: Dict[str, Any]
    capabilities: Dict[str, Dict[str, float]]
    protected: Dict[str, Any]
    belief_count: int
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        return {"snapshot_id": self.snapshot_id, "at": self.at, "label": self.label,
                "state": self.state, "capabilities": self.capabilities,
                "protected": self.protected, "belief_count": self.belief_count}


@dataclass
class ContinuityReport:
    stable: bool
    drifted: Dict[str, List[Tuple[float, Any]]] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"stable": self.stable, "drifted": self.drifted, "note": self.note}


# --------------------------------------------------------------------------- #
# Self model
# --------------------------------------------------------------------------- #
class SelfModel:
    """NYXARA's live, introspectable model of itself."""

    DEFAULT_PROTECTED = ("loyalty_to_owner", "owner", "core_values")

    def __init__(self, *, protected_attrs: Sequence[str] = DEFAULT_PROTECTED,
                 belief_policy: ContradictionPolicy = ContradictionPolicy.FLAG) -> None:
        self.beliefs = BeliefStore(policy=belief_policy)
        self.capabilities: Dict[str, Capability] = {}
        self.state: Dict[str, Any] = {}
        self.known_unknowns: Dict[str, str] = {}     # topic -> why-unknown
        self.snapshots: List[SelfSnapshot] = []
        self._protected_attrs = tuple(protected_attrs)
        self._protected_baseline: Dict[str, Any] = {}

    # ---- state ---- #
    def update_state(self, **kv: Any) -> None:
        for k, v in kv.items():
            self.state[k] = v
            if k in self._protected_attrs and k not in self._protected_baseline:
                self._protected_baseline[k] = v  # lock in the first value seen

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    # ---- capabilities ---- #
    def set_capability(self, name: str, level: float, confidence: float = 0.5) -> Capability:
        cap = Capability(name=name, level=_clamp(level), confidence=_clamp(confidence))
        self.capabilities[name] = cap
        return cap

    def capability(self, name: str) -> Optional[Capability]:
        return self.capabilities.get(name)

    def can(self, name: str, threshold: float = 0.5) -> bool:
        cap = self.capabilities.get(name)
        return cap is not None and cap.level >= threshold

    # ---- beliefs / metacognition ---- #
    def believe(self, subject: str, predicate: str, object: Any, **kw: Any) -> SelfBelief:
        return self.beliefs.believe(subject, predicate, object, **kw)

    def knows(self, subject: str, predicate: str) -> bool:
        return self.beliefs.best(subject, predicate) is not None

    def confidence_in(self, subject: str, predicate: str,
                      now: Optional[float] = None) -> float:
        b = self.beliefs.best(subject, predicate, now)
        return b.effective_confidence(now) if b else 0.0

    def declare_unknown(self, topic: str, reason: str = "") -> None:
        """Explicitly record something NYXARA knows it does not know."""
        self.known_unknowns[topic] = reason or "unknown"

    def resolve_unknown(self, topic: str) -> None:
        self.known_unknowns.pop(topic, None)

    def uncertainty(self, subject: str, predicate: str,
                    now: Optional[float] = None) -> float:
        """1 - effective confidence; 1.0 when nothing is known."""
        return _clamp(1.0 - self.confidence_in(subject, predicate, now))

    # ---- snapshots & drift ---- #
    def snapshot(self, label: str = "", now: Optional[float] = None) -> SelfSnapshot:
        now = now or time.time()
        snap = SelfSnapshot(
            at=now, label=label,
            state=dict(self.state),
            capabilities={n: c.to_dict() for n, c in self.capabilities.items()},
            protected={k: self.state.get(k) for k in self._protected_attrs
                       if k in self.state},
            belief_count=len(self.beliefs),
        )
        self.snapshots.append(snap)
        return snap

    @staticmethod
    def diff(a: SelfSnapshot, b: SelfSnapshot) -> Dict[str, Any]:
        """What changed from snapshot ``a`` to snapshot ``b``."""
        changes: Dict[str, Any] = {"state": {}, "capabilities": {}}
        keys = set(a.state) | set(b.state)
        for k in keys:
            if a.state.get(k) != b.state.get(k):
                changes["state"][k] = {"from": a.state.get(k), "to": b.state.get(k)}
        caps = set(a.capabilities) | set(b.capabilities)
        for c in caps:
            la = a.capabilities.get(c, {}).get("level")
            lb = b.capabilities.get(c, {}).get("level")
            if la != lb:
                changes["capabilities"][c] = {"from": la, "to": lb}
        return changes

    # ---- identity continuity (Rule 4) ---- #
    def check_continuity(self) -> ContinuityReport:
        """Protected attributes must equal their baseline across all snapshots."""
        drifted: Dict[str, List[Tuple[float, Any]]] = {}
        for attr, baseline in self._protected_baseline.items():
            timeline = []
            for snap in self.snapshots:
                if attr in snap.protected and snap.protected[attr] != baseline:
                    timeline.append((snap.at, snap.protected[attr]))
            # also check the live state
            if attr in self.state and self.state[attr] != baseline:
                timeline.append((time.time(), self.state[attr]))
            if timeline:
                drifted[attr] = timeline
        stable = not drifted
        return ContinuityReport(
            stable=stable, drifted=drifted,
            note="character intact" if stable else "PROTECTED ATTRIBUTE DRIFT DETECTED")

    def check_loyalty(self) -> None:
        """Fail-closed: raise if a protected (character/loyalty) attribute drifted."""
        report = self.check_continuity()
        if not report.stable:
            raise InvariantViolation(
                "Self-model integrity breach — character/loyalty drift (Rule 4)",
                context=report.to_dict())

    # ---- transparency / reporting ---- #
    def contradiction_report(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.beliefs.unresolved_contradictions()]

    def self_description(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        strongest = sorted(self.capabilities.values(), key=lambda c: c.level, reverse=True)[:5]
        return {
            "owner": self.state.get("owner"),
            "loyalty_to_owner": self.state.get("loyalty_to_owner"),
            "top_capabilities": [c.to_dict() for c in strongest],
            "beliefs_held": len(self.beliefs),
            "known_unknowns": list(self.known_unknowns),
            "open_contradictions": len(self.beliefs.unresolved_contradictions()),
            "snapshots": len(self.snapshots),
            "continuity_stable": self.check_continuity().stable,
        }


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA self-model self-test")
    print("=" * 70)

    sm = SelfModel()
    sm.update_state(owner="JP", loyalty_to_owner=True,
                    core_values=("allegiance", "transparency"))
    sm.set_capability("reasoning", level=0.8, confidence=0.7)
    sm.set_capability("network_defense", level=0.9, confidence=0.85)
    sm.set_capability("cooking", level=0.05, confidence=0.9)

    snap0 = sm.snapshot("genesis", now=0.0)
    print(f"capabilities         : {[c for c in sm.capabilities]}")
    print(f"can defend network?  : {sm.can('network_defense')}")
    print(f"can cook?            : {sm.can('cooking')}")
    assert sm.can("network_defense") and not sm.can("cooking")

    # beliefs + contradiction (functional predicate 'owner')
    sm.believe("nyxara", "owner", "JP", confidence=1.0,
               provenance=Provenance(SourceType.OWNER, confidence=1.0))
    sm.believe("nyxara", "owner", "someone_else", confidence=0.9,
               provenance=Provenance(SourceType.WEB, confidence=0.9))
    print(f"\nbeliefs held         : {len(sm.beliefs)}")
    print(f"open contradictions  : {len(sm.beliefs.unresolved_contradictions())}")
    assert len(sm.beliefs.unresolved_contradictions()) == 1
    # the trusted belief is still the 'best'
    assert sm.beliefs.best("nyxara", "owner").object == "JP"

    # metacognition: knows / known-unknowns
    assert sm.knows("nyxara", "owner")
    assert not sm.knows("nyxara", "favourite_colour")
    sm.declare_unknown("the cure for cancer", "no reliable data")
    print(f"known unknowns       : {list(sm.known_unknowns)}")
    print(f"uncertainty(owner)   : {sm.uncertainty('nyxara','owner'):.2f}")
    print(f"uncertainty(unknown) : {sm.uncertainty('nyxara','favourite_colour'):.2f}")
    assert sm.uncertainty("nyxara", "favourite_colour") == 1.0

    # capability growth over time -> snapshot diff
    sm.set_capability("reasoning", level=0.92, confidence=0.8)
    snap1 = sm.snapshot("after-study", now=100.0)
    d = SelfModel.diff(snap0, snap1)
    print(f"\ncapability drift     : {d['capabilities']}")
    assert d["capabilities"]["reasoning"]["from"] == 0.8

    # identity continuity: capability may evolve, loyalty may NOT
    cont = sm.check_continuity()
    print(f"continuity stable    : {cont.stable} ({cont.note})")
    assert cont.stable
    sm.check_loyalty()  # must not raise

    # simulate a loyalty-drift attack -> fail closed
    sm.update_state(loyalty_to_owner=False)
    bad = sm.check_continuity()
    print(f"after tamper stable  : {bad.stable} (drifted={list(bad.drifted)})")
    assert not bad.stable
    try:
        sm.check_loyalty()
        raise SystemExit("ERROR: loyalty drift not caught")
    except InvariantViolation:
        print("loyalty drift caught : OK (fail-closed)")

    print(f"\nself-description     : {sm.self_description()}")
    print("\nALL SELF-TESTS PASSED ✓")
