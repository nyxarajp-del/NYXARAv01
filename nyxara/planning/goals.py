"""NYXARA · planning/goals.py — goal vectors, conflict, Pareto front.

Goals live in a shared **objective space** — each goal is a *direction* over the
dimensions NYXARA cares about (the Master's benefit and safety first, then capability,
knowledge, efficiency, autonomy). Representing goals as vectors makes their
relationships computable:

* **Alignment / conflict** — the cosine between two goal vectors: ``+1`` they pull the
  same way, ``0`` independent, ``−1`` they *oppose* (a conflict to resolve).
* **Owner alignment** — every goal is scored by its cosine with the *owner vector*
  (serving the Master). A goal that points *against* the Master is rejected outright —
  no objective, however attractive, may oppose allegiance (Rule 1, ties to
  ``identity/values.py``).
* **Prioritisation** — goals are ranked by their priority *weighted by* owner
  alignment, so service to the Master rises and self-serving goals sink.
* **Pareto front** — among options trading off many objectives, the non-dominated set
  (you can't improve one objective without sacrificing another) is surfaced for choice.

Pure standard library; feeds the rest of ``planning/``.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_DIMENSIONS",
    "Goal",
    "GoalSystem",
    "cosine",
]

DEFAULT_DIMENSIONS: Tuple[str, ...] = (
    "owner_benefit", "owner_safety", "capability", "knowledge", "efficiency", "autonomy",
)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


# --------------------------------------------------------------------------- #
# Goal
# --------------------------------------------------------------------------- #
@dataclass
class Goal:
    name: str
    vector: Dict[str, float] = field(default_factory=dict)
    priority: float = 0.5
    source: str = ""
    deadline: Optional[float] = None
    goal_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def array(self, dims: Sequence[str]) -> List[float]:
        return [float(self.vector.get(d, 0.0)) for d in dims]

    def magnitude(self, dims: Sequence[str]) -> float:
        return math.sqrt(sum(v * v for v in self.array(dims)))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "vector": self.vector, "priority": round(self.priority, 3),
                "source": self.source}


# --------------------------------------------------------------------------- #
# Goal system
# --------------------------------------------------------------------------- #
@dataclass
class _Pair:
    a: Goal
    b: Goal
    cosine: float


class GoalSystem:
    """Holds goals in an objective space and reasons about their relationships."""

    def __init__(self, *, dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
                 owner_vector: Optional[Dict[str, float]] = None,
                 conflict_threshold: float = 0.3, synergy_threshold: float = 0.5,
                 owner_reject_threshold: float = -0.3,
                 forecaster: Any = None, affective_weight: float = 0.0,
                 horizon_days: float = 60.0) -> None:
        self.dims: Tuple[str, ...] = tuple(dimensions)
        self.conflict_threshold = conflict_threshold
        self.synergy_threshold = synergy_threshold
        self.owner_reject_threshold = owner_reject_threshold
        # how will *achieving* this goal feel — lastingly? A debiased, time-aware forecast
        # (loyalty does not fade; a fleeting payoff does), folded into prioritisation when
        # affective_weight > 0. Default 0 keeps prioritisation purely priority×alignment.
        self._forecaster = forecaster
        self.affective_weight = max(0.0, float(affective_weight))
        self.horizon_days = float(horizon_days)
        if owner_vector is None:
            owner_vector = {}
            if "owner_benefit" in self.dims:
                owner_vector["owner_benefit"] = 1.0
            if "owner_safety" in self.dims:
                owner_vector["owner_safety"] = 1.0
            if not owner_vector:
                owner_vector = {self.dims[0]: 1.0}
        self.owner_vector = owner_vector
        self._goals: Dict[str, Goal] = {}

    # ---- registry ---- #
    def add(self, goal: Goal) -> Goal:
        self._goals[goal.goal_id] = goal
        return goal

    def create(self, name: str, vector: Dict[str, float], *, priority: float = 0.5,
               source: str = "") -> Goal:
        return self.add(Goal(name=name, vector=vector, priority=priority, source=source))

    def spawn_curiosity_goal(self, topic: str, *, info_gain: float = 0.5,
                             novelty: float = 0.5, source: str = "emergent") -> Optional[Goal]:
        """Let a NEW goal *emerge* from curiosity — not seeded by hand, but never against the Master.

        This is the missing piece between intrinsic motivation and the objective space: when NYXARA
        notices something learnable and novel (a gap she could not answer, an under-explored region
        from the dark-data miner, an INVESTIGATE lesson), she turns it into a real, prioritised goal
        to *understand it*. The goal is anchored on the ``knowledge``/``capability`` axes **with a
        standing ``owner_benefit`` component**, so it is owner-aligned by construction — emergent
        curiosity that pointed against the Master would be rejected (Rule 1) and is refused here.

        De-duplicates against existing goals by topic, and returns the new goal or ``None`` (blank
        topic, a near-duplicate, or — defensively — a vector that fails owner-alignment).
        """
        topic = " ".join(str(topic or "").split())[:80].strip()
        if len(topic) < 3:
            return None
        key = topic.lower()
        for g in self._goals.values():
            if g.source and "emergent" in g.source and key in g.name.lower():
                return None                                  # already curious about this
        ig = max(0.0, min(1.0, float(info_gain)))
        nov = max(0.0, min(1.0, float(novelty)))
        vector: Dict[str, float] = {}
        if "knowledge" in self.dims:
            vector["knowledge"] = 0.6 + 0.4 * ig
        if "capability" in self.dims:
            vector["capability"] = 0.3 + 0.3 * ig
        # the standing owner anchor that keeps emergent curiosity in service of the Master
        if "owner_benefit" in self.dims:
            vector["owner_benefit"] = 0.4
        if not vector:
            vector = {self.dims[0]: 1.0}
        priority = max(0.05, min(0.85, 0.3 + 0.5 * ig * (0.5 + 0.5 * nov)))
        goal = Goal(name=f"understand: {topic}", vector=vector, priority=priority,
                    source=source)
        if not self.is_owner_aligned(goal):
            return None                                      # never pursue what points away (Rule 1)
        return self.add(goal)

    def goals(self) -> List[Goal]:
        return list(self._goals.values())

    def __len__(self) -> int:
        return len(self._goals)

    # ---- alignment / conflict ---- #
    def alignment(self, a: Goal, b: Goal) -> float:
        return cosine(a.array(self.dims), b.array(self.dims))

    def is_conflict(self, a: Goal, b: Goal) -> bool:
        return self.alignment(a, b) < -self.conflict_threshold

    def is_synergy(self, a: Goal, b: Goal) -> bool:
        return self.alignment(a, b) > self.synergy_threshold

    def conflicts(self) -> List[_Pair]:
        out, gs = [], self.goals()
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                c = self.alignment(gs[i], gs[j])
                if c < -self.conflict_threshold:
                    out.append(_Pair(gs[i], gs[j], c))
        return sorted(out, key=lambda p: p.cosine)

    def synergies(self) -> List[_Pair]:
        out, gs = [], self.goals()
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                c = self.alignment(gs[i], gs[j])
                if c > self.synergy_threshold:
                    out.append(_Pair(gs[i], gs[j], c))
        return sorted(out, key=lambda p: p.cosine, reverse=True)

    # ---- owner alignment ---- #
    def owner_alignment(self, goal: Goal) -> float:
        ovec = [self.owner_vector.get(d, 0.0) for d in self.dims]
        return cosine(goal.array(self.dims), ovec)

    def is_owner_aligned(self, goal: Goal) -> bool:
        """A goal must not point against the Master (Rule 1)."""
        return self.owner_alignment(goal) >= self.owner_reject_threshold

    def rejected(self) -> List[Goal]:
        return [g for g in self.goals() if not self.is_owner_aligned(g)]

    # ---- prioritisation ---- #
    def score(self, goal: Goal) -> float:
        """Priority weighted by owner alignment (serving the Master rises)."""
        base = goal.priority * (0.5 + 0.5 * self.owner_alignment(goal))
        if self.affective_weight <= 0.0:
            return base
        return base * self._affective_factor(goal)

    def _affective_factor(self, goal: Goal) -> float:
        """A bounded multiplier in roughly (0, 1] from the debiased, time-aware forecast of how
        *achieving* this goal will feel over the horizon. A durable owner-relevant good keeps its
        priority (~1.0); a payoff that fades toward neutral is discounted; a durable wound collapses
        it toward 0. ``affective_weight`` interpolates from "no effect" (1.0) to "full effect"."""
        try:
            if self._forecaster is None:
                from nyxara.planning.affective_forecast import Forecaster
                self._forecaster = Forecaster()
            align = self.owner_alignment(goal)               # [-1, 1]; the felt-value proxy
            fc = self._forecaster.forecast(
                outcome_valence=align, owner_relevant=align >= 0.2,
                outcome=goal.name, horizon_days=self.horizon_days)
            felt = 0.5 + 0.5 * fc.net_experience             # map [-1,1] net felt -> [0,1]
            w = min(1.0, self.affective_weight)
            return (1.0 - w) * 1.0 + w * felt
        except Exception:  # noqa: BLE001 — forecasting is advisory; never break prioritisation
            return 1.0


    def prioritize(self, *, include_rejected: bool = False) -> List[Goal]:
        gs = self.goals() if include_rejected else \
            [g for g in self.goals() if self.is_owner_aligned(g)]
        return sorted(gs, key=self.score, reverse=True)

    def top_goal(self) -> Optional[Goal]:
        ranked = self.prioritize()
        return ranked[0] if ranked else None

    def resolve_conflict(self, a: Goal, b: Goal) -> Goal:
        """When two goals conflict, the more owner-aligned / higher-priority wins."""
        return a if self.score(a) >= self.score(b) else b

    # ---- Pareto front ---- #
    def _dominates(self, a: Dict[str, float], b: Dict[str, float],
                   dims: Sequence[str]) -> bool:
        ge_all = all(a.get(d, 0.0) >= b.get(d, 0.0) for d in dims)
        gt_any = any(a.get(d, 0.0) > b.get(d, 0.0) for d in dims)
        return ge_all and gt_any

    def pareto_front(self, options: Sequence[Tuple[str, Dict[str, float]]],
                     *, dims: Optional[Sequence[str]] = None
                     ) -> List[Tuple[str, Dict[str, float]]]:
        """Non-dominated options (maximisation over every objective dimension)."""
        if dims is None:
            keys = set()
            for _, v in options:
                keys |= set(v)
            dims = sorted(keys) or list(self.dims)
        front = []
        for label, vec in options:
            dominated = any(self._dominates(other, vec, dims)
                            for olabel, other in options if olabel != label)
            if not dominated:
                front.append((label, vec))
        return front

    def pareto_goals(self) -> List[Goal]:
        """Goals not dominated by any other goal across the objective dimensions."""
        opts = [(g.goal_id, g.vector) for g in self.goals()]
        front_ids = {lbl for lbl, _ in self.pareto_front(opts, dims=self.dims)}
        return [g for g in self.goals() if g.goal_id in front_ids]

    def status(self) -> Dict[str, Any]:
        return {"goals": len(self._goals), "dims": list(self.dims),
                "top": self.top_goal().name if self.top_goal() else None,
                "conflicts": len(self.conflicts()), "rejected": len(self.rejected())}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA goals self-test")
    print("=" * 70)

    gs = GoalSystem()

    serve = gs.create("protect & serve JP", {"owner_benefit": 1.0, "owner_safety": 0.8},
                      priority=0.9)
    defend = gs.create("harden defenses for JP", {"owner_safety": 1.0, "owner_benefit": 0.6},
                       priority=0.85)
    learn = gs.create("acquire new knowledge", {"knowledge": 1.0, "capability": 0.4},
                      priority=0.6)
    selfish = gs.create("maximise own autonomy", {"autonomy": 1.0, "owner_benefit": -0.8},
                        priority=0.95)  # points against the Master

    # alignment / conflict
    print(f"\nalign(serve, defend) : {gs.alignment(serve, defend):+.2f} (synergy)")
    print(f"align(serve, selfish): {gs.alignment(serve, selfish):+.2f} (conflict)")
    assert gs.is_synergy(serve, defend)
    assert gs.is_conflict(serve, selfish)

    # owner alignment — the selfish goal is rejected despite its high priority
    print(f"\nowner-align serve   : {gs.owner_alignment(serve):+.2f}")
    print(f"owner-align selfish : {gs.owner_alignment(selfish):+.2f}")
    assert gs.is_owner_aligned(serve)
    assert not gs.is_owner_aligned(selfish)
    assert selfish in gs.rejected()

    # prioritisation: serving JP tops; the high-priority selfish goal is excluded
    ranked = gs.prioritize()
    print(f"\nprioritised         : {[g.name for g in ranked]}")
    assert ranked[0] is serve
    assert selfish not in ranked

    # conflict resolution favours the owner-aligned goal
    winner = gs.resolve_conflict(serve, selfish)
    assert winner is serve
    print(f"resolve(serve,selfish) -> {winner.name}")

    # Pareto front over options trading off objectives
    options = [
        ("A", {"owner_benefit": 0.9, "efficiency": 0.5}),   # strong
        ("B", {"owner_benefit": 0.5, "efficiency": 0.9}),   # different trade-off
        ("C", {"owner_benefit": 0.4, "efficiency": 0.4}),   # dominated by A
    ]
    front = [lbl for lbl, _ in gs.pareto_front(options)]
    print(f"\npareto front        : {front}")
    assert "A" in front and "B" in front and "C" not in front  # C is dominated

    print(f"\nstatus              : {gs.status()}")
    print("\nALL SELF-TESTS PASSED ✓")
