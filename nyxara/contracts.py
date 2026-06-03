"""Cross-cutting contracts shared by every Nyxara layer.

These types are the *lingua franca* of the system. They are defined exactly once, here,
and imported everywhere else — no layer redefines them. Keeping them in a single
dependency-free-ish module avoids import cycles between kernel / mind / guard / memory.

Design rules:
  * Faculties never act. They emit a :class:`Proposal`.
  * The guard / invariants layer judges proposals and returns a :class:`Verdict`.
  * Everything the kernel routes is an :class:`Event` on the bus.
  * Every belief carries :class:`Provenance` (source + confidence + acquisition time).
"""
from __future__ import annotations

import enum
import time
import uuid
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def new_id(prefix: str = "id") -> str:
    """Short, sortable-ish unique id with a semantic prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    """Monotonic-friendly wall-clock timestamp (seconds since epoch)."""
    return time.time()


class Priority(enum.IntEnum):
    """Event / task priority. Lower value = handled sooner in the priority queue."""

    CRITICAL = 0  # oversight, kill-switch, invariant breach
    HIGH = 1      # owner command, threat signal
    NORMAL = 2    # ordinary cognition
    LOW = 3       # background, idle stream
    IDLE = 4      # mind-wandering / incubation


class _Frozen(BaseModel):
    """Base for value objects: validated, immutable, hashable-by-content."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class _Mutable(BaseModel):
    """Base for live state objects that are updated in place."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Event — the unit the bus routes
# ─────────────────────────────────────────────────────────────────────────────
class Event(_Frozen):
    """An immutable message on the kernel event bus."""

    id: str = Field(default_factory=lambda: new_id("evt"))
    topic: str
    payload: Mapping[str, Any] = Field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    ts: float = Field(default_factory=now)
    trace_id: str = Field(default_factory=lambda: new_id("trace"))
    source: str = "unknown"

    def child(self, topic: str, **payload: Any) -> "Event":
        """Derive a follow-on event that keeps this event's trace lineage."""
        return Event(
            topic=topic,
            payload=payload,
            priority=self.priority,
            trace_id=self.trace_id,
            source=self.source,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Provenance & Belief — epistemic bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
class Provenance(_Frozen):
    """Where a belief came from, how, and when."""

    source: str                         # "owner", "web:example.com", "reasoner", ...
    method: str = "stated"              # "stated", "inferred", "perceived", "tool"
    acquired_at: float = Field(default_factory=now)
    confidence: float = 0.5             # source-level trust, 0..1
    chain: tuple[str, ...] = ()         # upstream provenance ids, if derived

    def derived(self, method: str, by: str) -> "Provenance":
        return Provenance(
            source=by,
            method=method,
            confidence=self.confidence,
            chain=self.chain + (self.source,),
        )


class Belief(_Mutable):
    """A claim Nyxara holds, with calibrated confidence and provenance."""

    id: str = Field(default_factory=lambda: new_id("blf"))
    claim: str
    confidence: float = 0.5             # current calibrated belief, 0..1
    provenance: Provenance
    tags: tuple[str, ...] = ()

    def contradicts(self, other: "Belief") -> bool:
        """Naive structural contradiction: same claim text negated."""
        a, b = self.claim.strip().lower(), other.claim.strip().lower()
        return a == f"not {b}" or b == f"not {a}"


# ─────────────────────────────────────────────────────────────────────────────
#  Proposal — every faculty output. NOT an action.
# ─────────────────────────────────────────────────────────────────────────────
class Proposal(_Frozen):
    """A *proposed* action from a faculty. The kernel decides whether it happens.

    Pipeline: schema-check → guard/shield → critique → invariants gate → kernel acts.
    A proposal is inert until the orchestrator disposes of it.
    """

    id: str = Field(default_factory=lambda: new_id("prop"))
    faculty: str                        # which engine produced it: llm/math/rag/...
    action: str                         # intended effect, e.g. "reply", "store", "tool"
    args: Mapping[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.5             # faculty's self-assessed confidence, 0..1
    provenance: Provenance | None = None
    requires: tuple[str, ...] = ()      # capability tokens the action needs
    trace_id: str = Field(default_factory=lambda: new_id("trace"))

    def with_args(self, **overrides: Any) -> "Proposal":
        merged = {**self.args, **overrides}
        return self.model_copy(update={"args": merged})


# ─────────────────────────────────────────────────────────────────────────────
#  Verdict — guard / invariants output on a proposal
# ─────────────────────────────────────────────────────────────────────────────
class Verdict(_Frozen):
    """The gate's judgement of a proposal."""

    allow: bool
    reasons: tuple[str, ...] = ()
    modifications: Mapping[str, Any] = Field(default_factory=dict)
    decided_by: str = "guardian"
    severity: int = 0                   # 0 ok, higher = more serious veto

    @classmethod
    def ok(cls, by: str = "guardian", **mods: Any) -> "Verdict":
        return cls(allow=True, decided_by=by, modifications=mods)

    @classmethod
    def deny(cls, *reasons: str, by: str = "guardian", severity: int = 1) -> "Verdict":
        return cls(allow=False, reasons=tuple(reasons), decided_by=by, severity=severity)

    def applied_to(self, proposal: Proposal) -> Proposal:
        """Return the proposal with any guard-mandated modifications merged in."""
        if not self.modifications:
            return proposal
        return proposal.with_args(**dict(self.modifications))


# ─────────────────────────────────────────────────────────────────────────────
#  Percept — fused sensory input handed to the workspace
# ─────────────────────────────────────────────────────────────────────────────
class Percept(_Frozen):
    """A unified, cross-modal perception ready for the global workspace."""

    id: str = Field(default_factory=lambda: new_id("pct"))
    modality: str                       # "text", "vision", "audio", "fused"
    content: Any
    salience: float = 0.5               # bottom-up attention weight, 0..1
    surprise: float = 0.0               # prediction error vs. expectation, 0..1
    ts: float = Field(default_factory=now)
    meta: Mapping[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  Goal — planning + agency currency
# ─────────────────────────────────────────────────────────────────────────────
class Goal(_Mutable):
    """An objective, represented as a labelled vector for conflict / Pareto math."""

    id: str = Field(default_factory=lambda: new_id("goal"))
    description: str
    vector: tuple[float, ...] = ()      # embedding in value-space for cos() conflict
    priority: float = 0.5
    deadline: float | None = None
    origin: str = "intrinsic"           # "owner", "intrinsic", "derived"
    parent: str | None = None
    done: bool = False

    def is_overdue(self, at: float | None = None) -> bool:
        return self.deadline is not None and (at or now()) > self.deadline


__all__ = [
    "new_id",
    "now",
    "Priority",
    "Event",
    "Provenance",
    "Belief",
    "Proposal",
    "Verdict",
    "Percept",
    "Goal",
]


def _self_check() -> Sequence[str]:
    """Lightweight smoke test used by tests and `python -m nyxara.contracts`."""
    out: list[str] = []
    e = Event(topic="t", payload={"x": 1})
    out.append(f"event:{e.topic}")
    p = Proposal(faculty="llm", action="reply", confidence=0.9)
    v = Verdict.deny("nope")
    assert not v.allow and v.reasons == ("nope",)
    out.append(f"proposal:{p.faculty}->{v.allow}")
    g = Goal(description="serve owner", priority=1.0)
    assert not g.done
    out.append(f"goal:{g.description}")
    return out


if __name__ == "__main__":  # pragma: no cover
    for line in _self_check():
        print(line)
