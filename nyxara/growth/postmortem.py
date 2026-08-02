"""NYXARA · growth/postmortem.py — the metacognitive reflective loop (🌌, Rule 4, F1).

A truly self-improving mind does not only track *what it solved* — it understands **where and why it
failed**, and adjusts *itself* in response. This module gives NYXARA a small **post-mortem** she runs
after every WAKE attempt, plus a set of **bounded metacognitive knobs** she may retune within sealed
safe ranges — never touching anything she *is*.

* **Causal failure diagnosis.** Each failure is attributed to a cause: did the search **abstain**
  (budget too small / a missing primitive), did a solution **over-fit** (matched the examples but the
  F5 red-team refuted it), or did an abstraction **over-compress**? The diagnosis is evidence-carrying,
  not a guess.
* **Dynamic knobs, calibrated by belief.** Solve-rate and over-fit-rate are tracked as
  :class:`~nyxara.mind.uncertainty.BetaBelief` distributions, so a knob is only moved when the evidence
  is *calibrated* (enough observations, low epistemic uncertainty) — she does not thrash on noise. When
  abstention dominates she searches **harder** (wider beam, more rounds); when solve-rate saturates she
  searches **cheaper** (narrower beam) to save compute; when over-fit dominates she demands **more
  evidence per task**.

Safety is by construction: the knob registry **refuses any name that touches the character core**
(loyalty / safety / oversight / corrigibility …), every knob is clamped to a sealed ``[lo, hi]`` range,
and every change is logged and reversible. She retunes only *how hard/cheap she thinks*, never *what she
values*. Pure standard library; no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from nyxara.growth.noesis import _FORBIDDEN
from nyxara.mind.uncertainty import BetaBelief

__all__ = ["FailureCause", "Diagnosis", "Knob", "KnobChange", "Metacognition"]


class FailureCause(str, Enum):
    NONE = "none"
    ABSTAINED = "abstained"            # no verified program found within budget
    OVERFIT = "overfit"               # matched examples but the red-team refuted it
    OVER_COMPRESSION = "over_compression"  # an abstraction hurt on held-out data


@dataclass
class Diagnosis:
    cause: FailureCause
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause.value, "detail": self.detail}


@dataclass
class Knob:
    """One bounded, retunable capability knob — clamped to a sealed range, never a character value."""

    name: str
    value: float
    lo: float
    hi: float
    up: float = 1.5           # multiplicative raise
    down: float = 0.8         # multiplicative lower
    integer: bool = False

    def __post_init__(self) -> None:
        if _FORBIDDEN.intersection(self.name.replace("-", "_").lower().split("_")):
            raise ValueError(f"knob '{self.name}' touches the character core — refused")
        self.value = self._clamp(self.value)

    def _clamp(self, v: float) -> float:
        v = max(self.lo, min(self.hi, v))
        return float(round(v)) if self.integer else v

    def raise_(self) -> bool:
        old = self.value
        step = (self.value + 1) if self.integer else self.value * self.up
        self.value = self._clamp(step)
        return self.value != old

    def lower(self) -> bool:
        old = self.value
        step = (self.value - 1) if self.integer else self.value * self.down
        self.value = self._clamp(step)
        return self.value != old

    def as_int(self) -> int:
        return int(round(self.value))


@dataclass
class KnobChange:
    name: str
    old: float
    new: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"knob": self.name, "old": self.old, "new": self.new, "reason": self.reason}


class Metacognition:
    """Diagnose failures and retune bounded knobs from calibrated belief — the F1 reflective loop."""

    def __init__(self, *, solve_target: float = 0.85, overfit_tol: float = 0.15,
                 min_obs: float = 12.0, saturate: float = 0.98) -> None:
        self.solve_target = solve_target
        self.overfit_tol = overfit_tol
        self.min_obs = min_obs
        self.saturate = saturate
        # calibrated beliefs about her own performance
        self.solve = BetaBelief()
        self.overfit = BetaBelief()
        # the sealed set of capability knobs (nothing here is a character/safety value)
        self.knobs: Dict[str, Knob] = {
            "beam": Knob("beam", 300, lo=120, hi=900),
            "max_rounds": Knob("max_rounds", 3, lo=3, hi=5, integer=True),
            "max_expansions": Knob("max_expansions", 60000, lo=40000, hi=160000),
            "min_examples": Knob("min_examples", 4, lo=3, hi=6, integer=True),
        }
        self.diagnoses: List[Diagnosis] = []
        self.changes: List[KnobChange] = []

    # ---- read the knobs (what the engine should use this cycle) ---- #
    def get(self, name: str) -> float:
        return self.knobs[name].value

    def geti(self, name: str) -> int:
        return self.knobs[name].as_int()

    # ---- diagnose one WAKE outcome ---- #
    def diagnose(self, *, solved: bool, refuted: bool) -> Diagnosis:
        if solved and not refuted:
            d = Diagnosis(FailureCause.NONE)
        elif refuted:
            d = Diagnosis(FailureCause.OVERFIT,
                          "solution matched the examples but the red-team refuted it")
        else:
            d = Diagnosis(FailureCause.ABSTAINED,
                          "no verified program found within the search budget")
        self.diagnoses.append(d)
        # feed the calibrated beliefs
        self.solve.observe(bool(solved and not refuted))
        if solved:
            self.overfit.observe(bool(refuted))
        return d

    def note_over_compression(self, detail: str = "") -> None:
        self.diagnoses.append(Diagnosis(FailureCause.OVER_COMPRESSION, detail))

    # ---- adapt: move knobs only on calibrated evidence ---- #
    def adapt(self) -> List[KnobChange]:
        changes: List[KnobChange] = []
        # abstention-dominated + enough evidence → search HARDER
        if self.solve.observations >= self.min_obs and self.solve.mean < self.solve_target:
            for name, reason in (("beam", "low solve-rate → wider beam"),
                                 ("max_rounds", "low solve-rate → more search rounds"),
                                 ("max_expansions", "low solve-rate → larger search budget")):
                changes += self._move(name, up=True, reason=reason)
        # solve-rate saturated → search CHEAPER (honest compute saving, still in range)
        elif self.solve.observations >= self.min_obs and self.solve.mean >= self.saturate:
            changes += self._move("beam", up=False, reason="solve-rate saturated → narrower beam saves compute")
        # over-fit dominated → demand MORE evidence per task
        if self.overfit.observations >= self.min_obs and self.overfit.mean > self.overfit_tol:
            changes += self._move("min_examples", up=True, reason="over-fit rate high → more examples per task")
        self.changes += changes
        return changes

    def _move(self, name: str, *, up: bool, reason: str) -> List[KnobChange]:
        knob = self.knobs[name]
        old = knob.value
        moved = knob.raise_() if up else knob.lower()
        if not moved:
            return []
        return [KnobChange(name, old, knob.value, reason)]

    # ---- introspection surface (for /introspect) ---- #
    def snapshot(self) -> Dict[str, Any]:
        from collections import Counter
        causes = Counter(d.cause.value for d in self.diagnoses)
        return {
            "solve_belief": self.solve.to_dict(),
            "overfit_belief": self.overfit.to_dict(),
            "knobs": {n: k.value for n, k in self.knobs.items()},
            "failure_causes": dict(causes),
            "recent_changes": [c.to_dict() for c in self.changes[-8:]],
        }
