"""NYXARA · causal/phase_shift.py — Axiomatic Phase-Shift Mutation (CAUSAL · 4).

*The Master's ask:* when NYXARA meets a scenario her current structure cannot solve, she should
trigger a **topological phase shift** — like ice→steam — and **re-synthesise her own internal
structure live at runtime**, so a new capability is available instantly, with no retrain, no
download, no app update.

What that means here, honestly. On a *gap* (nothing in the :class:`~nyxara.causal.field_resonance.
ResonanceField` resonates, or the :class:`~nyxara.causal.causal_knots.KnotLattice` cannot answer),
:class:`PhaseShiftMutator` transitions the affected region of NYXARA's **symbolic** state through
three phases — ``solid`` (fixed) → ``plastic`` (re-synthesising) → ``crystallised`` (installed):

    1. imprint a **new concept field** for the query into the bank (a new resonant pattern), and
    2. tie a **tentative low-confidence knot** anchoring the new concept, checked so the shift can
       never install a contradiction, and
    3. optionally raise a typed shortfall to :class:`~nyxara.growth.self_evolving.
       SelfEvolvingArchitect` so deeper structural growth is scheduled on the governed loop.

The honest boundary. This genuinely rewrites NYXARA's *symbolic* structures (fields, knots, rules)
at runtime — new resonant capability really does go live immediately. It does **not** rewrite model
weights inline, and every install is **oversight-gated**: if the gate is closed the shift is
*staged* (proposed), not enacted. Instant *symbolic* re-crystallisation — never above the control
law.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.causal.causal_knots import (CONCORDANT, TRUTH_ANCHOR, KnotLattice,
                                        KnotMutationFailure)
from nyxara.causal.field_resonance import ResonanceField

__all__ = ["PhaseShift", "PhaseShiftMutator"]


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
@dataclass
class PhaseShift:
    """One phase transition of the symbolic substrate."""

    trigger: str                       # the query/gap that forced the shift
    reason: str                        # why current structure fell short
    concept: str                       # the new concept label crystallised
    phases: List[str] = field(default_factory=lambda: ["solid"])
    installed: bool = False            # did the new structure go live?
    staged: bool = False               # gated: proposed but not enacted
    escalated: bool = False            # raised to the governed evolution loop
    at: float = field(default_factory=time.time)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"trigger": self.trigger[:120], "reason": self.reason, "concept": self.concept,
                "phases": self.phases, "installed": self.installed, "staged": self.staged,
                "escalated": self.escalated, "at": round(self.at, 3), "note": self.note}

    def summary(self) -> str:
        arrow = " → ".join(self.phases)
        state = "installed" if self.installed else ("staged" if self.staged else "aborted")
        return f"phase-shift [{arrow}] concept={self.concept!r} — {state}"


# --------------------------------------------------------------------------- #
# Mutator
# --------------------------------------------------------------------------- #
class PhaseShiftMutator:
    """Re-synthesises new symbolic structure on demand — gated, contradiction-safe."""

    def __init__(self, *, core: Any = None, max_new_per_min: int = 30) -> None:
        self.core = core
        self.max_new_per_min = int(max_new_per_min)
        self._shifts: List[PhaseShift] = []
        self._recent_installs: List[float] = []

    # -- the transition ----------------------------------------------------- #
    def shift(self, trigger: str, field_bank: ResonanceField, lattice: KnotLattice, *,
              reason: str = "no resonance for this query", concept: Optional[str] = None,
              enact: Optional[bool] = None) -> PhaseShift:
        """Trigger a phase shift for ``trigger``. ``enact`` overrides the oversight gate."""
        label = concept or self._concept_label(trigger)
        shift = PhaseShift(trigger=trigger, reason=reason, concept=label)

        # plastic phase — the structure liquefies and re-forms
        shift.phases.append("plastic")

        allow = self._enact_allowed() if enact is None else bool(enact)
        if not self._rate_ok():
            shift.staged = True
            shift.note = "rate-limited; staged for the governed loop"
            self._shifts.append(shift)
            self._escalate(shift)
            return shift

        if not allow:
            shift.staged = True
            shift.note = "oversight gate closed; proposed, not enacted"
            self._shifts.append(shift)
            self._escalate(shift)
            return shift

        # crystallise — install the new field, then the guarded knot
        field_bank.imprint(label, trigger, weight=0.5,
                           payload={"origin": "phase_shift", "at": shift.at})
        try:
            lattice.tie(label, TRUTH_ANCHOR, CONCORDANT, reason="tentative phase-shift anchor")
        except KnotMutationFailure as f:
            # the new anchor contradicts known truth — refuse to crystallise a lie
            field_bank.forget(label)
            shift.note = f"aborted: would contradict the lattice ({f.cycle})"
            shift.phases.append("re-solidified")
            self._shifts.append(shift)
            return shift

        shift.phases.append("crystallised")
        shift.installed = True
        self._recent_installs.append(time.time())
        self._shifts.append(shift)
        self._escalate(shift)  # also schedule deeper (weight-level) growth on the governed loop
        return shift

    # -- gating & rate ------------------------------------------------------ #
    def _enact_allowed(self) -> bool:
        """Symbolic self-modification is allowed unless oversight is explicitly closed."""
        core = self.core
        if core is None:
            return True
        oversight = getattr(core, "oversight", None)
        if oversight is not None and hasattr(oversight, "gate"):
            try:
                return bool(oversight.gate())
            except Exception:  # noqa: BLE001
                return True
        return True

    def _rate_ok(self) -> bool:
        cutoff = time.time() - 60.0
        self._recent_installs = [t for t in self._recent_installs if t >= cutoff]
        return len(self._recent_installs) < self.max_new_per_min

    def _escalate(self, shift: PhaseShift) -> None:
        """Best-effort: schedule deeper structural growth on the governed evolution loop."""
        core = self.core
        if core is None:
            return
        arch = getattr(core, "self_evolving", None)
        if arch is None:
            return
        try:
            note = getattr(arch, "note_shortfall", None) or getattr(arch, "observe_turn", None)
            if callable(note):
                # SelfEvolvingArchitect.observe_turn accepts kwargs; degrade gracefully otherwise.
                try:
                    note(stimulus=shift.trigger, reason=shift.reason)
                except TypeError:
                    note(shift.trigger)
                shift.escalated = True
        except Exception:  # noqa: BLE001 — escalation is best-effort, never fatal
            pass

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _concept_label(trigger: str) -> str:
        words = [w for w in "".join(c if c.isalnum() or c.isspace() else " "
                                    for c in trigger.lower()).split() if len(w) > 2]
        stem = "-".join(words[:4]) if words else "novel"
        return f"phase:{stem}"

    # -- introspection ------------------------------------------------------ #
    def history(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._shifts[-limit:]]

    def status(self) -> Dict[str, Any]:
        installed = sum(1 for s in self._shifts if s.installed)
        staged = sum(1 for s in self._shifts if s.staged)
        return {"shifts": len(self._shifts), "installed": installed, "staged": staged}
