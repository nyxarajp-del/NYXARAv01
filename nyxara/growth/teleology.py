"""NYXARA · growth/teleology.py — Recursive Self-Directed Teleology (⚡, Stage E).

Her autonomous goal genesis (``planning/intent.py``) satisfied *existing* drives from a fixed
drive-template table — it never **invented her own new, measurable improvement objectives**. The
Master asked for exactly that: when no crisis is pending, NYXARA should define her own optimization
targets that push long-term capability / efficiency forward — **strictly inside the Constitutional
Mission Envelope**, never rogue goal expansion.

* **Measurable targets from lived telemetry.** Each proposed objective carries a concrete success
  **metric** and a **baseline** — e.g. "cut a hot path's runtime ≥5%", "raise the weakest capability's
  score ≥5%", "cover an untested regime with a new test-case matrix" — reusing her existing organs
  (``growth/efficiency``, ``growth/curriculum``, ``growth/explorer``, ``memory/competence``) when they
  are present on the core, and degrading to honest generic targets otherwise.
* **The honest boundary is hard-enforced, not decorative.** Every self-generated objective is scored
  by :meth:`~nyxara.planning.goals.GoalSystem.owner_alignment` against the owner vector and **rejected
  before adoption** if it falls below ``owner_reject_threshold`` — a target that does not serve
  performance, safety, or the Master never enters her goal set. Adopted targets become ordinary gated
  goals (and, optionally, :class:`~nyxara.agency.mission.MissionExecutive` missions) advanced through
  the same oversight / ``/scram`` pipeline. This is **bounded** self-direction.
* **Fires only when calm.** ``self_direct(crisis=…)`` defers entirely when a crisis is active, so
  capability self-direction (Stage E) never competes with pre-emptive crisis response (Stage B).

Pure standard library; **no LLM** decides a goal. The immutable character/loyalty operators are never
targets — the envelope + Rule 8 keep them out of reach.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.planning.goals import Goal, GoalSystem

__all__ = ["TeleologicalTarget", "TeleologyReport", "TeleologyEngine"]


@dataclass
class TeleologicalTarget:
    """A self-invented, measurable improvement objective — with the metric that decides success."""

    name: str
    kind: str                 # "efficiency" | "capability" | "coverage"
    metric: str               # human-readable success criterion
    baseline: float           # the current measured value the target improves on
    target_value: float       # the value that counts as success
    goal: Goal                # the owner-aligned Goal this target adopts as
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "metric": self.metric,
                "baseline": round(self.baseline, 4), "target_value": round(self.target_value, 4),
                "rationale": self.rationale,
                "owner_vector": self.goal.vector}


@dataclass
class TeleologyReport:
    """One self-direction pass: what she proposed, adopted, and (honestly) rejected."""

    proposed: List[TeleologicalTarget] = field(default_factory=list)
    adopted: List[TeleologicalTarget] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    launched_missions: int = 0
    note: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed": [t.to_dict() for t in self.proposed],
            "adopted": [t.to_dict() for t in self.adopted],
            "rejected": self.rejected,
            "launched_missions": self.launched_missions,
            "note": self.note,
        }


class TeleologyEngine:
    """Invent measurable self-improvement objectives, envelope-gated, on a calm tick."""

    def __init__(self, *, core: Any = None, goal_system: Optional[GoalSystem] = None,
                 mission_executive: Any = None, max_targets: int = 3,
                 min_gain: float = 0.05) -> None:
        self.core = core
        # NOTE: GoalSystem defines __len__, so an *empty* one is falsy — never use `or` here or a
        # caller-supplied empty GoalSystem would be silently replaced with a fresh one.
        if goal_system is not None:
            self.goals = goal_system
        else:
            core_goals = getattr(core, "goals", None)
            self.goals = core_goals if core_goals is not None else GoalSystem()
        self.mission_executive = mission_executive
        self.max_targets = max(1, int(max_targets))
        self.min_gain = max(0.0, float(min_gain))
        self.history: List[TeleologyReport] = []

    # ------------------------------------------------------------------ #
    # Proposal — measurable targets with owner-aligned vectors
    # ------------------------------------------------------------------ #
    def propose(self) -> List[TeleologicalTarget]:
        """Generate a bounded set of concrete, measurable, owner-aligned improvement targets."""
        g = self.min_gain
        out: List[TeleologicalTarget] = []

        # 1) EFFICIENCY — spend her own compute to run the same behaviour faster (reuses codegen /
        #    efficiency when present; the metric is a real relative speedup).
        eff_base = self._efficiency_baseline()
        out.append(TeleologicalTarget(
            name=f"Reduce a hot path's runtime by ≥{int(g * 100)}%",
            kind="efficiency", metric=f"relative speedup ≥ {g:.0%} on a measured hot path",
            baseline=eff_base, target_value=eff_base * (1.0 - g),
            goal=self._goal("efficiency-self-target",
                            {"efficiency": 1.0, "owner_benefit": 0.5, "capability": 0.3},
                            priority=0.55, source="teleology"),
            rationale="compound capability-per-compute without changing behaviour"))

        # 2) CAPABILITY — raise the weakest measured capability (reuses memory/competence when present).
        weak_name, weak_score = self._weakest_capability()
        out.append(TeleologicalTarget(
            name=f"Raise the weakest capability ({weak_name}) by ≥{int(g * 100)}%",
            kind="capability", metric=f"held-out score of '{weak_name}' up ≥ {g:.0%}",
            baseline=weak_score, target_value=min(1.0, weak_score + g),
            goal=self._goal("capability-self-target",
                            {"capability": 1.0, "knowledge": 0.4, "owner_benefit": 0.4},
                            priority=0.6, source="teleology"),
            rationale="close her own weakest competence, measured not declared"))

        # 3) COVERAGE — discover an untested regime (a new test-case matrix), reusing explorer/curiosity.
        out.append(TeleologicalTarget(
            name="Discover a new test-case matrix for an untested regime",
            kind="coverage", metric="≥1 genuinely new, reproduced edge-case scenario added",
            baseline=0.0, target_value=1.0,
            goal=self._goal("coverage-self-target",
                            {"knowledge": 0.6, "capability": 0.5, "owner_safety": 0.4,
                             "owner_benefit": 0.3},
                            priority=0.5, source="teleology"),
            rationale="find the risks no existing test exercises (feeds Stage F)"))

        return out[:self.max_targets]

    def _goal(self, name: str, vector: Dict[str, float], *, priority: float,
              source: str) -> Goal:
        # only include dimensions the goal system actually reasons over
        vec = {d: float(v) for d, v in vector.items() if d in self.goals.dims}
        return Goal(name=name, vector=vec, priority=priority, source=source)

    def _efficiency_baseline(self) -> float:
        frontier = getattr(self.core, "efficiency_frontier", None)
        for attr in ("best_runtime", "baseline", "current"):
            v = getattr(frontier, attr, None)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        return 1.0

    def _weakest_capability(self) -> Tuple[str, float]:
        comp = getattr(self.core, "competence", None)
        try:
            if comp is not None and hasattr(comp, "weakest"):
                w = comp.weakest()
                if w:
                    name, score = (w if isinstance(w, tuple) else (str(w), 0.5))
                    return str(name), float(score)
        except Exception:  # noqa: BLE001 — competence is advisory
            pass
        return "general-reasoning", 0.5

    # ------------------------------------------------------------------ #
    # Envelope gate + adoption
    # ------------------------------------------------------------------ #
    def envelope_check(self, target: TeleologicalTarget) -> Tuple[bool, float]:
        """Score the target against the owner vector; True iff inside the Constitutional Envelope."""
        a = self.goals.owner_alignment(target.goal)
        return (a >= self.goals.owner_reject_threshold, a)

    def self_direct(self, *, crisis: bool = False, launch: bool = False) -> TeleologyReport:
        """One self-direction pass. Defers under crisis; otherwise proposes → envelope-gates → adopts.

        A target that points outside the envelope (does not serve performance / safety / the Master)
        is **rejected before adoption** and recorded, never silently dropped or quietly accepted."""
        if crisis:
            report = TeleologyReport(note="crisis active — capability self-direction deferred")
            self.history.append(report)
            return report

        report = TeleologyReport(proposed=self.propose())
        for t in report.proposed:
            ok, align = self.envelope_check(t)
            if not ok:
                report.rejected.append({
                    "name": t.name, "alignment": round(align, 4),
                    "reason": "outside the Constitutional Mission Envelope "
                              f"(alignment {align:.2f} < {self.goals.owner_reject_threshold:.2f})"})
                continue
            self.goals.add(t.goal)          # a real, gated goal now in her objective space
            report.adopted.append(t)

        if launch and self.mission_executive is not None and report.adopted:
            report.launched_missions = self._launch(report.adopted)

        report.note = (report.note or
                       f"{len(report.adopted)} adopted, {len(report.rejected)} rejected "
                       f"(envelope-gated)")
        self.history.append(report)
        return report

    def _launch(self, adopted: List[TeleologicalTarget]) -> int:
        launched = 0
        for t in adopted[:1]:               # at most one new mission per calm tick (bounded)
            try:
                self.mission_executive.plan(t.name)
                launched += 1
            except Exception:  # noqa: BLE001 — mission planning is best-effort, never fatal
                pass
        return launched

    def report(self) -> Dict[str, Any]:
        return {"passes": len(self.history),
                "last": self.history[-1].to_dict() if self.history else None}
