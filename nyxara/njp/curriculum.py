"""NYXARA · njp/curriculum.py — the nine stages, and refusing to skip one (🧪).

The Master's order, and the claim that makes it a curriculum rather than a list: **a stage does
not advance until its mastery is demonstrated by measurement.**

    A  Prediction        sequence → next state
    B  Representation    many observations → reusable latent concepts
    C  Abstraction       examples → general concepts, and a real compression win
    D  Causality         action → consequence
    E  Counterfactuals   "if X had not happened…"
    F  Reasoning         hypotheses → evidence → conclusion
    G  Agency            goal → plan → action → outcome
    H  Metacognition     answer → confidence → critique → correction
    I  Self-improvement  find the weakness, propose, sandbox, benchmark, adopt only if better

**Why the order is load-bearing and not decorative.** Every stage is built out of the one before
it. Concepts (B) are formed over observations that prediction (A) makes salient. Abstraction (C)
is worthless without concepts to abstract over. Counterfactuals (E) are interventions on the
causal structure D found. Agency (G) plans by searching the predictive model A learned — an agent
with no world model does not plan, it flails. Letting a stage run before its floor exists does not
produce a fast learner; it produces numbers that mean nothing, which is worse than no numbers.

**Mastery is a measurement, never a claim.** Each :class:`Stage` names a metric that already
exists on a real organ, a threshold, and a minimum sample count so a lucky first success cannot
promote her. :meth:`Curriculum.assess` reads them live and reports the whole ladder — mastered,
current, and blocked — with the actual value beside each. Nothing here can advance itself by
asserting progress.

**Blocked is a normal state and is reported as one.** A brain that has met four turns is *meant*
to be blocked at Stage A, and saying so is more useful than a percentage. :attr:`Report.blocked_by`
names the metric standing in the way and what it currently reads, so the answer to "why is she not
learning to plan yet" is a number rather than a guess.

**This module measures and gates. It does not train.** The organs learn from experience —
:mod:`nyxara.njp.study` from a corpus, :mod:`nyxara.njp.field` from every turn. What was missing
was anything that knew *what order they have to come in* and refused to pretend a stage was
reached. That is what this is.

Pure standard library, no LLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Stage", "StageResult", "Report", "Curriculum", "STAGES"]


def _dig(stats: Dict[str, Any], organ: str, key: str) -> Optional[float]:
    """Read one number out of the brain's own stats, or ``None`` if the organ is absent.

    ``None`` and ``0.0`` are kept apart throughout. An organ that is switched off has *no*
    reading; one that is on and has done nothing reads zero. Collapsing the two is how a
    disabled subsystem gets reported as a failing one.
    """
    block = stats.get(organ)
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Stage:
    """One rung: what it is for, the number that proves it, and how much evidence it needs."""

    letter: str = ""
    name: str = ""
    teaches: str = ""
    organ: str = ""
    metric: str = ""
    threshold: float = 0.0
    sample_organ: str = ""
    sample_metric: str = ""
    min_samples: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"letter": self.letter, "name": self.name, "teaches": self.teaches,
                "metric": f"{self.organ}.{self.metric}", "threshold": self.threshold,
                "evidence": f"{self.sample_organ}.{self.sample_metric} ≥ {self.min_samples:g}"}


#: The ladder. Each metric is produced by an organ that already computes it for its own reasons,
#: so nothing here is a number invented to be passed.
STAGES: Tuple[Stage, ...] = (
    Stage("A", "prediction", "sequence → next state",
          "predictive", "accuracy", 0.55, "predictive", "scored", 20),
    Stage("B", "representation", "observations → reusable latent concepts",
          "concepts", "concepts", 3, "concepts", "observations", 12),
    Stage("C", "abstraction", "examples → general concepts that compress",
          "concepts", "compression", 1.5, "concepts", "observations", 20),
    Stage("D", "causality", "action → consequence",
          "world", "causal_links", 3, "world", "stated_laws", 3),
    Stage("E", "counterfactuals", "if X had not happened…",
          "universe", "usable_relations", 1, "universe", "observations", 5),
    Stage("F", "reasoning", "hypotheses → evidence → conclusion",
          "metareason", "assertable_rate", 0.5, "metareason", "solved", 4),
    Stage("G", "agency", "goal → plan → action → outcome",
          "agency", "success_rate", 0.5, "agency", "acted", 4),
    Stage("H", "metacognition", "answer → confidence → critique → correction",
          "beliefs", "known", 3, "beliefs", "beliefs", 5),
    Stage("I", "self-improvement", "weakness → proposal → sandbox → benchmark → adopt",
          "field", "meta_accepted", 1, "field", "meta_trials", 1),
)


@dataclass
class StageResult:
    """Where one rung actually stands, with the numbers that say so."""

    stage: Stage = dc_field(default_factory=Stage)
    value: Optional[float] = None
    samples: Optional[float] = None
    mastered: bool = False
    reachable: bool = True          # is its floor in place?
    present: bool = True            # is the organ built at all?

    @property
    def enough_evidence(self) -> bool:
        return self.samples is not None and self.samples >= self.stage.min_samples

    @property
    def status(self) -> str:
        if self.mastered:
            return "mastered"
        if not self.reachable:
            return "blocked"        # an earlier stage is not done
        if not self.present:
            return "absent"         # the organ is switched off entirely
        if self.value is None:
            return "no reading yet"  # built, but it has not done enough to have a number
        if not self.enough_evidence:
            return "insufficient evidence"
        return "below threshold"

    @property
    def why(self) -> str:
        s = self.stage
        if self.mastered:
            return f"{s.organ}.{s.metric} = {self.value:g} ≥ {s.threshold:g}"
        if not self.present:
            return f"{s.organ} is not built"
        if self.value is None:
            return f"{s.organ}.{s.metric} has no reading yet"
        if not self.enough_evidence:
            got = 0 if self.samples is None else self.samples
            return (f"only {got:g} of {s.min_samples:g} required "
                    f"{s.sample_organ}.{s.sample_metric}")
        return f"{s.organ}.{s.metric} = {self.value:g}, needs {s.threshold:g}"

    def to_dict(self) -> Dict[str, Any]:
        return {"letter": self.stage.letter, "name": self.stage.name,
                "status": self.status, "mastered": self.mastered, "present": self.present,
                "value": self.value, "samples": self.samples,
                "threshold": self.stage.threshold, "why": self.why}


@dataclass
class Report:
    """The whole ladder at one moment."""

    results: List[StageResult] = dc_field(default_factory=list)
    ms: float = 0.0

    @property
    def mastered(self) -> List[str]:
        return [r.stage.letter for r in self.results if r.mastered]

    @property
    def current(self) -> Optional[StageResult]:
        """The first rung not yet mastered — the one she is actually on."""
        return next((r for r in self.results if not r.mastered), None)

    @property
    def blocked_by(self) -> str:
        current = self.current
        return current.why if current is not None else ""

    @property
    def depth(self) -> int:
        """How many *consecutive* stages from A are mastered.

        Consecutive on purpose. Mastering G while D is unmet is not depth 7 — it is a number
        produced by machinery resting on a floor that is not there, and counting it would let the
        ladder be climbed out of order by accident.
        """
        depth = 0
        for result in self.results:
            if not result.mastered:
                break
            depth += 1
        return depth

    def to_dict(self) -> Dict[str, Any]:
        current = self.current
        return {"depth": self.depth, "mastered": self.mastered,
                "current": current.stage.letter if current else None,
                "blocked_by": self.blocked_by,
                "stages": [r.to_dict() for r in self.results],
                "ms": round(self.ms, 3)}


class Curriculum:
    """Reads the brain's own numbers and says which rung she is genuinely on."""

    def __init__(self, stages: Tuple[Stage, ...] = STAGES) -> None:
        self.stages = stages
        self.assessments = 0
        self.best_depth = 0
        self.history: List[Tuple[float, int]] = []      # (timestamp, depth)

    def assess(self, brain: Any) -> Report:
        """Measure every rung against the live organs. Nothing is asked to self-report."""
        report = Report()
        t0 = time.perf_counter()
        try:
            stats = brain.stats() if hasattr(brain, "stats") else {}
            stats = dict(stats or {})
            # The agent lives on the brain rather than in `stats`, so its numbers are pulled in
            # here rather than being quietly missing and reading as "organ absent".
            agent = getattr(brain, "agent", None)
            if agent is not None and hasattr(agent, "stats"):
                try:
                    # Merged under whatever `brain.stats()` already reported, never over it.
                    # `njp.doing` puts the counters Stage G is scored on into that block, and
                    # overwriting it here would hand the rung back to the planner whose own
                    # docstring explains why its counters cannot move.
                    stats["agency"] = {**agent.stats(), **(stats.get("agency") or {})}
                except Exception:  # noqa: BLE001
                    pass

            floor_intact = True
            for stage in self.stages:
                result = StageResult(stage=stage, reachable=floor_intact)
                # Built-but-silent and switched-off are different states, and conflating them
                # reported a live MetaReasoner as "not built" purely because its assertable_rate
                # was still None on a brain that had not yet been asked a question. The organ's
                # presence is a question about the stats block; the metric is a question about
                # what is in it.
                result.present = isinstance(stats.get(stage.organ), dict)
                result.value = _dig(stats, stage.organ, stage.metric)
                result.samples = _dig(stats, stage.sample_organ, stage.sample_metric)
                result.mastered = bool(
                    result.value is not None
                    and result.enough_evidence
                    and result.value >= stage.threshold)
                report.results.append(result)
                if not result.mastered:
                    # Later stages are still *measured* — the numbers are informative — but they
                    # are marked unreachable, so a rung that scored well on an absent floor is
                    # never mistaken for progress.
                    floor_intact = False

            self.assessments += 1
            self.best_depth = max(self.best_depth, report.depth)
            self.history.append((time.time(), report.depth))
            del self.history[:-256]
            return report
        except Exception:  # noqa: BLE001 — a failed assessment reports nothing, never raises
            return report
        finally:
            report.ms = (time.perf_counter() - t0) * 1000.0

    def next_stage(self, brain: Any) -> Optional[Stage]:
        """The rung to work on. ``None`` when the whole ladder is met."""
        current = self.assess(brain).current
        return current.stage if current is not None else None

    def advice(self, brain: Any) -> str:
        """What would actually move her forward, in one line.

        Deliberately about the *evidence*, not about the metric. Being told "your accuracy is
        0.31, make it 0.55" is not actionable; being told "she has scored 6 predictions and needs
        20" says what to go and do.
        """
        report = self.assess(brain)
        current = report.current
        if current is None:
            return "every stage is mastered"
        stage = current.stage
        if not current.present:
            return f"stage {stage.letter} ({stage.name}): {stage.organ} is switched off"
        if current.value is None:
            return (f"stage {stage.letter} ({stage.name}): {stage.organ} has never been "
                    f"exercised — no {stage.metric} to read yet")
        if not current.enough_evidence:
            got = 0 if current.samples is None else current.samples
            return (f"stage {stage.letter} ({stage.name}): needs more experience — "
                    f"{stage.sample_organ}.{stage.sample_metric} is {got:g}, "
                    f"wants {stage.min_samples:g}")
        return (f"stage {stage.letter} ({stage.name}): {stage.organ}.{stage.metric} is "
                f"{current.value:g}, wants {stage.threshold:g} — {stage.teaches}")

    def stats(self) -> Dict[str, Any]:
        return {"stages": len(self.stages), "assessments": self.assessments,
                "best_depth": self.best_depth,
                "trend": [d for _t, d in self.history[-8:]]}
