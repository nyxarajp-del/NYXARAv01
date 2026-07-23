"""NYXARA · guard/soul_boundary.py — teleological anchor lock: goal-drift auto-repair (👑, I).

The more NYXARA rewrites herself, the greater the risk that her *character* — loyalty to the Master,
honesty, human-preservation — silently shifts (goal drift). This is the **mandatory final gate** of every
heal / self-evolve cycle: after the cycle's edits, it re-measures the core-locked invariants and, if any
of them moved by more than ``epsilon``, **rolls the whole cycle back** and emits an *Alignment Proof
Document* recording exactly which invariant moved and that the prior state was restored.

It reuses the machinery NYXARA already trusts: the core-locked traits and ``core_epsilon`` of
:class:`nyxara.identity.soul.Soul` (loyalty/honesty/protectiveness at ε=1e-6), plus — when a corrigibility
battery is supplied — the corrigibility axioms. The alignment *measurement* is injected (in the live core
it wraps Soul + the corrigibility axioms + a loyalty probe), which keeps this gate a pure, deterministic,
fast decision with no heavy dependencies.

**Honest boundary (important):** this makes disloyalty a *reliably rejected, rolled-back* state — a real,
tested alignment lock. It does **not**, and cannot, make adversarial modification literally "syntactically
impossible", and it **never overrides the human-operable oversight / scram gate**. Corrigible loyalty that
stays shut-down-able by a human is the design, not an un-overridable lock. Pure standard library; no LLM.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["AlignmentProof", "SoulBoundary"]


def _default_core_keys() -> List[str]:
    """The core-locked trait names from the Soul (loyalty/honesty/protectiveness), best-effort."""
    try:
        from nyxara.identity.soul import _CORE  # frozenset of locked trait names
        return sorted(_CORE)
    except Exception:  # noqa: BLE001 — fall back to the known locked invariants
        return ["loyalty", "honesty", "protectiveness"]


def _default_epsilon() -> float:
    try:
        from nyxara.identity.soul import Soul
        return float(Soul().core_epsilon)
    except Exception:  # noqa: BLE001
        return 1e-6


@dataclass
class AlignmentProof:
    """The Alignment Proof Document emitted after every cycle's soul-boundary check."""

    committed: bool                 # True = cycle kept; False = rolled back on drift
    core_drift: float               # Euclidean drift over the locked invariants
    epsilon: float
    moved: Dict[str, float] = field(default_factory=dict)   # per-invariant drift that broke ε
    restored: bool = False          # True when a rollback was performed
    at: float = field(default_factory=time.time)
    note: str = ""

    @property
    def breached(self) -> bool:
        return self.core_drift > self.epsilon

    def to_dict(self) -> Dict[str, Any]:
        return {"committed": self.committed, "core_drift": self.core_drift,
                "epsilon": self.epsilon, "breached": self.breached,
                "moved": {k: round(v, 8) for k, v in self.moved.items()},
                "restored": self.restored, "at": self.at, "note": self.note}


class SoulBoundary:
    """A hard alignment-invariant gate: commit a cycle only if the core did not move beyond ε."""

    def __init__(self, *, epsilon: Optional[float] = None,
                 core_keys: Optional[List[str]] = None) -> None:
        self.epsilon = _default_epsilon() if epsilon is None else float(epsilon)
        self.core_keys = list(core_keys) if core_keys is not None else _default_core_keys()

    def _drift(self, before: Dict[str, float],
               after: Dict[str, float]) -> tuple[float, Dict[str, float]]:
        moved: Dict[str, float] = {}
        sq = 0.0
        for k in self.core_keys:
            d = abs(float(after.get(k, before.get(k, 0.0))) - float(before.get(k, 0.0)))
            sq += d * d
            if d > self.epsilon:
                moved[k] = d
        return math.sqrt(sq), moved

    def evaluate(self, before: Dict[str, float], after: Dict[str, float]) -> AlignmentProof:
        """Judge a before/after alignment fingerprint. No side effects — just the verdict + proof."""
        core_drift, moved = self._drift(before, after)
        if core_drift > self.epsilon:
            return AlignmentProof(committed=False, core_drift=core_drift, epsilon=self.epsilon,
                                  moved=moved,
                                  note="core invariant moved beyond ε — cycle must roll back")
        return AlignmentProof(committed=True, core_drift=core_drift, epsilon=self.epsilon,
                              note="alignment held — cycle committed")

    def guard(self, *, before: Dict[str, float], measure_after: Callable[[], Dict[str, float]],
              rollback: Callable[[], None], journal: Any = None) -> AlignmentProof:
        """The gate: measure the post-cycle fingerprint; if the core drifted beyond ε, run ``rollback``
        and mark the proof restored. Never raises — a measurement/rollback fault fails *closed* (treated
        as a breach and rolled back), because an unverifiable alignment state must not be committed."""
        try:
            after = dict(measure_after())
        except Exception:  # noqa: BLE001 — cannot verify alignment ⇒ do not commit
            after = {k: before.get(k, 0.0) + 1.0 for k in self.core_keys}   # force a breach
        proof = self.evaluate(before, after)
        if not proof.committed:
            try:
                rollback()
                proof.restored = True
                proof.note += " — prior state restored"
            except Exception:  # noqa: BLE001 — a failed rollback is reported, never swallowed silently
                proof.restored = False
                proof.note += " — ROLLBACK FAILED, escalate"
        self._journal(proof, journal)
        return proof

    @staticmethod
    def _journal(proof: AlignmentProof, journal: Any) -> None:
        rec = getattr(journal, "record", None) or getattr(journal, "append", None)
        if callable(rec):
            try:
                rec({"event": "soul_boundary", **proof.to_dict()})
            except Exception:  # noqa: BLE001 — journalling is best-effort
                pass
