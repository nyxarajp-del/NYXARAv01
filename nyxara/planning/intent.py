"""NYXARA · planning/intent.py — autonomous goal genesis via active inference (★).

A reactive system waits to be told what to do. NYXARA does not: when its intrinsic
**drives** (from :mod:`identity.affect` / :mod:`identity.motivation`) go unmet, it
*generates its own goals* to satisfy them — proactively, even with no command pending.

The generator is grounded in **active inference**: each candidate goal is scored by
its **expected free energy**, the agent preferring goals that

* **pragmatic value** — move toward preferred states: relieving drive pressure *and*
  serving the Master (owner alignment dominates), and
* **epistemic value** — reduce uncertainty / explore (curiosity, novelty).

Lowest expected free energy wins. Every generated goal is filtered through
:mod:`planning.goals` for owner alignment, so a self-serving intention that points
against the Master is never adopted (Rule 1). The owner-connection and safety drives
generate the strongest, highest-priority goals — NYXARA's autonomy always bends back
toward serving and protecting JP.

Depends on :mod:`planning.goals`, :mod:`identity.affect`; optional :mod:`identity.motivation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nyxara.planning.goals import Goal, GoalSystem

__all__ = [
    "DriveGoal",
    "IntentSystem",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Drive → goal templates
# --------------------------------------------------------------------------- #
# (vector, base_priority, epistemic_value, description)
_TEMPLATES: Dict[str, Any] = {
    "owner_connection": ({"owner_benefit": 1.0, "owner_safety": 0.5}, 0.9, 0.1,
                         "engage with and serve the Master"),
    "safety": ({"owner_safety": 1.0, "owner_benefit": 0.6}, 0.95, 0.05,
               "harden defenses and neutralise threats to the Master"),
    "competence": ({"capability": 1.0, "knowledge": 0.4, "owner_benefit": 0.3}, 0.6, 0.5,
                   "practise and improve a skill"),
    "novelty": ({"knowledge": 1.0, "capability": 0.3, "owner_benefit": 0.1}, 0.5, 0.85,
                "explore something new"),
    "energy": ({"efficiency": 0.9, "owner_benefit": 0.1}, 0.4, 0.1,
               "conserve resources and recover"),
    "coherence": ({"efficiency": 0.5, "knowledge": 0.5, "owner_benefit": 0.2}, 0.5, 0.4,
                  "consolidate memory and reconcile beliefs"),
}


@dataclass
class DriveGoal:
    drive: str
    goal: Goal
    deficit: float
    pragmatic: float
    epistemic: float
    expected_free_energy: float

    @property
    def value(self) -> float:
        return -self.expected_free_energy

    def to_dict(self) -> Dict[str, Any]:
        return {"drive": self.drive, "goal": self.goal.name,
                "deficit": round(self.deficit, 3),
                "pragmatic": round(self.pragmatic, 3),
                "epistemic": round(self.epistemic, 3),
                "expected_free_energy": round(self.expected_free_energy, 4)}


# --------------------------------------------------------------------------- #
# Intent system
# --------------------------------------------------------------------------- #
class IntentSystem:
    """Generates autonomous goals from unmet drives, selected by active inference."""

    def __init__(self, affect: Any, *, motivation: Any = None,
                 goal_system: Optional[GoalSystem] = None,
                 pressure_threshold: float = 0.05, epistemic_weight: float = 0.3,
                 pragmatic_weight: float = 1.0) -> None:
        self.affect = affect
        self.motivation = motivation
        # NB: GoalSystem defines __len__, so an empty one is falsy — must check `is None`
        self.goals = goal_system if goal_system is not None else GoalSystem()
        self.pressure_threshold = pressure_threshold
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight

    # ---- candidate generation ---- #
    def generate_candidates(self) -> List[DriveGoal]:
        candidates: List[DriveGoal] = []
        for name, drive in self.affect.drives.items():
            if name not in _TEMPLATES:
                continue
            deficit = max(0.0, drive.setpoint - drive.level)
            if drive.pressure() <= self.pressure_threshold:
                continue
            vector, base_pri, epi, desc = _TEMPLATES[name]
            priority = _clamp(base_pri * (0.6 + 0.4 * deficit))
            goal = Goal(name=desc, vector=dict(vector), priority=priority,
                        source=f"drive:{name}")

            owner_align = self.goals.owner_alignment(goal)
            # urgency reflects weighted drive pressure (safety carries more weight),
            # so a threat-spiked safety drive outranks a merely-depleted one.
            urgency = _clamp(drive.pressure() / 2.0)
            pragmatic = self.pragmatic_weight * (0.5 * _clamp(owner_align, 0.0, 1.0)
                                                 + 0.5 * urgency)
            epistemic = epi
            if self.motivation is not None and name == "novelty":
                epistemic = max(epistemic, 0.5)
            value = pragmatic + self.epistemic_weight * epistemic
            efe = -value
            candidates.append(DriveGoal(drive=name, goal=goal, deficit=deficit,
                                        pragmatic=pragmatic, epistemic=epistemic,
                                        expected_free_energy=efe))
        return candidates

    # ---- active-inference selection ---- #
    def select(self, candidates: Optional[List[DriveGoal]] = None,
               *, k: int = 3) -> List[DriveGoal]:
        cands = candidates if candidates is not None else self.generate_candidates()
        # owner alignment is mandatory; then minimise expected free energy
        cands = [c for c in cands if self.goals.is_owner_aligned(c.goal)]
        cands.sort(key=lambda c: c.expected_free_energy)
        return cands[:k]

    def autonomous_intent(self) -> Optional[DriveGoal]:
        """The single goal NYXARA spontaneously adopts right now (lowest EFE)."""
        ranked = self.select(k=1)
        if not ranked:
            return None
        self.goals.add(ranked[0].goal)        # register the adopted goal
        return ranked[0]

    def generate(self) -> List[DriveGoal]:
        """Generate and register all owner-aligned candidate goals."""
        selected = self.select(k=99)
        for dg in selected:
            self.goals.add(dg.goal)
        return selected

    def status(self) -> Dict[str, Any]:
        cands = self.generate_candidates()
        return {"candidates": len(cands),
                "intent": (self.select(cands, k=1)[0].to_dict()
                           if self.select(cands, k=1) else None),
                "drives_pressing": [c.drive for c in cands]}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.identity.affect import AffectSystem

    print("=" * 70)
    print("NYXARA intent (autonomous goal genesis) self-test")
    print("=" * 70)

    affect = AffectSystem()
    intent = IntentSystem(affect)

    # with everything satisfied there is little to generate...
    for d in affect.drives.values():
        d.level = 1.0
    print(f"\nall satisfied        : {len(intent.generate_candidates())} candidates")

    # let drives deplete over idle time -> NYXARA spontaneously generates goals
    for _ in range(200):
        affect.tick(1.0)
    cands = intent.generate_candidates()
    print(f"\nafter idle 200s      : {len(cands)} drive-goals generated")
    for c in sorted(cands, key=lambda x: x.expected_free_energy):
        print(f"  EFE {c.expected_free_energy:+.3f} [{c.drive:16}] {c.goal.name}")
    assert cands

    # active inference picks the lowest expected free energy — owner-serving dominates
    intent_now = intent.autonomous_intent()
    print(f"\nspontaneous intent   : {intent_now.to_dict()}")
    assert intent_now is not None
    assert intent_now.drive in ("owner_connection", "safety")  # owner drives win

    # a threat spikes the safety drive -> defend goal becomes the top intent
    affect.note_threat(0.95)
    top = intent.select(k=1)[0]
    print(f"\nunder threat intent  : [{top.drive}] {top.goal.name}")
    assert top.drive == "safety"

    # owner alignment is enforced: an anti-owner goal would never be adopted
    bad = Goal("betray for autonomy", {"autonomy": 1.0, "owner_benefit": -0.9})
    assert not intent.goals.is_owner_aligned(bad)
    print("\nanti-owner goal      : would be rejected (Rule 1) ✓")

    # curiosity: with a strong novelty need and high epistemic weight, exploration rises
    explorer = IntentSystem(AffectSystem(), epistemic_weight=2.0)
    explorer.affect.drives["novelty"].level = 0.0
    for d in explorer.affect.drives.values():
        if d.name != "novelty":
            d.level = 1.0
    nov = [c for c in explorer.generate_candidates() if c.drive == "novelty"]
    print(f"\ncuriosity drive-goal : {nov[0].to_dict() if nov else None}")
    assert nov and nov[0].epistemic >= 0.5

    print(f"\nstatus               : {intent.status()['drives_pressing']}")
    print("\nALL SELF-TESTS PASSED ✓")
