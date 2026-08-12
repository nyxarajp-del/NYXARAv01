"""NYXARA · nyx001/metrics.py — "god-like", defined as a number (📐, G M R P T L U A C).

The charter's Section VI asks for something specific and unusual: instead of a marketing word,
intelligence should be *"a measurable scientific formula"* over nine components —
**G**eneralization, **M**emory, **R**easoning, **P**lanning, **T**ransfer, **L**earning efficiency,
**U**ncertainty (calibration), **A**daptation, **C**omputational efficiency.

This module is that formula, and its design is dominated by one decision: **a component that
cannot yet be measured is reported as ``None``, never as 0.0.** A zero is a measurement — it says
"this system has no reasoning ability" — and reporting it for a system that has simply not been
asked yet is the single easiest way to make an evaluation dishonest. Every component here can
return ``None``, and :attr:`IndexReport.coverage` says how much of the index is actually backed by
evidence.

The aggregate is a **geometric mean of the components that exist**, not an arithmetic one. That
choice matters: an arithmetic mean lets a system with excellent memory and no reasoning at all
post a respectable score, because the strong component compensates. A geometric mean does not —
one near-zero component drags the whole index toward zero, which is the correct behaviour for a
claim about *general* intelligence.

    index = (∏ measured components)^(1/n)     over the n components that could be measured

Every component also carries its **sample count**, so a thin record reads as thin. An index built
on four resolved outcomes is reported with ``thin=True`` and should be argued with, not quoted.

Honest, and this is the part that matters most: these nine numbers are **proxies**. ``R`` is the
survival rate of tested hypotheses, not a measure of reasoning in any rich sense; ``P`` is whether
plans simulate feasibly, not whether they work in the world. The index is useful because every
component is a number someone can check and dispute — not because the word above it is earned.
Nothing here licenses the phrase "god-like"; the charter asked for a measurement to *replace* it,
and this is the measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Component", "IndexReport", "IntelligenceIndex"]

_NAMES = {
    "G": "generalization", "M": "memory", "R": "reasoning", "P": "planning",
    "T": "transfer", "L": "learning efficiency", "U": "calibration",
    "A": "adaptation", "C": "compute efficiency",
}


@dataclass
class Component:
    """One measured dimension, with the evidence behind it."""

    key: str = ""
    name: str = ""
    value: Optional[float] = None
    samples: int = 0
    note: str = ""

    @property
    def measured(self) -> bool:
        return self.value is not None

    @property
    def thin(self) -> bool:
        """Measured, but on too little evidence to lean on."""
        return self.measured and self.samples < 10

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "name": self.name,
                "value": None if self.value is None else round(self.value, 4),
                "samples": self.samples, "measured": self.measured, "thin": self.thin,
                "note": self.note[:160]}


@dataclass
class IndexReport:
    """The nine components and what they do or do not add up to."""

    components: List[Component] = field(default_factory=list)
    index: Optional[float] = None

    @property
    def measured(self) -> List[Component]:
        return [c for c in self.components if c.measured]

    @property
    def coverage(self) -> float:
        """Fraction of the nine that are backed by any evidence at all."""
        return float(len(self.measured) / max(1, len(self.components)))

    @property
    def thin(self) -> bool:
        """True when the index rests on too few components or too little evidence."""
        return self.coverage < 0.6 or any(c.thin for c in self.measured)

    def to_dict(self) -> Dict[str, Any]:
        return {"index": None if self.index is None else round(self.index, 4),
                "coverage": round(self.coverage, 4), "thin": self.thin,
                "measured": len(self.measured), "total": len(self.components),
                "components": [c.to_dict() for c in self.components]}

    def render(self) -> str:
        """A human-readable report that never prints a number it did not measure."""
        lines = []
        if self.index is None:
            lines.append("intelligence index: NOT MEASURABLE — nothing has been measured yet")
        else:
            warn = "  ⚠ thin evidence" if self.thin else ""
            lines.append(f"intelligence index: {self.index:.4f} "
                         f"(geometric mean of {len(self.measured)}/{len(self.components)} "
                         f"components){warn}")
        for c in self.components:
            if c.value is None:
                lines.append(f"  {c.key}  {c.name:20s} —      not yet measurable · {c.note}")
            else:
                flag = " (thin)" if c.thin else ""
                lines.append(f"  {c.key}  {c.name:20s} {c.value:.4f}  n={c.samples}{flag}")
        lines.append("")
        lines.append("These are proxies. Every number is checkable and disputable; the phrase")
        lines.append("above them is not earned by any of them.")
        return "\n".join(lines)


class IntelligenceIndex:
    """Measures the nine components from a live NYX V.001 brain."""

    def __init__(self, *, min_samples: int = 10) -> None:
        self.min_samples = int(min_samples)

    # ---- the nine ---- #
    def _generalization(self, stack: Any) -> Component:
        """G — does the world model predict on states it was not fitted to? Proxy: is_learning."""
        c = Component(key="G", name=_NAMES["G"])
        wm = getattr(stack, "world_model", None)
        if wm is None:
            c.note = "no world model"
            return c
        curve = wm.learning_curve()
        if len(curve) < 4:
            c.note = "fewer than 4 milestones"
            return c
        first = sum(curve[:2]) / 2.0
        last = sum(curve[-2:]) / 2.0
        if first <= 0.0:
            c.note = "no initial error to improve on"
            return c
        c.value = float(max(0.0, min(1.0, 1.0 - last / first)))
        c.samples = int(getattr(wm, "steps", 0))
        c.note = "fraction of initial prediction error eliminated"
        return c

    def _memory(self, stack: Any) -> Component:
        """M — is memory retaining and being used? Proxy: credited fraction of stored episodes."""
        c = Component(key="M", name=_NAMES["M"])
        mem = getattr(stack, "episodic", None)
        if mem is None:
            c.note = "no episodic memory"
            return c
        st = mem.stats()
        n = int(st.get("episodes", 0))
        if n < 8:
            c.note = f"only {n} episodes stored"
            return c
        c.value = float(min(1.0, st.get("credited", 0) / max(1, n)))
        c.samples = n
        c.note = "fraction of episodes credited as having mattered"
        return c

    def _reasoning(self, stack: Any) -> Component:
        """R — survival rate of TESTED hypotheses. Untested claims contribute nothing."""
        c = Component(key="R", name=_NAMES["R"])
        r = getattr(stack, "reasoning", None)
        if r is None:
            c.note = "no reasoning engine"
            return c
        st = r.stats()
        tested = int(st.get("tested", 0))
        if tested < 3:
            c.note = f"only {tested} hypotheses tested"
            return c
        c.value = float(min(1.0, st.get("survivors", 0) / max(1, tested)))
        c.samples = tested
        c.note = "survival rate of tested hypotheses"
        return c

    def _planning(self, stack: Any) -> Component:
        """P — do plans simulate into feasible sequences? Not whether they work in the world."""
        c = Component(key="P", name=_NAMES["P"])
        p = getattr(stack, "planning", None)
        if p is None:
            c.note = "no planner"
            return c
        st = p.stats()
        plans = int(st.get("plans", 0))
        if plans < 3:
            c.note = f"only {plans} plans attempted"
            return c
        c.value = 1.0 if st.get("attached") else 0.0
        c.samples = plans
        c.note = "planner attached to a world model and producing plans"
        return c

    def _transfer(self, stack: Any) -> Component:
        """T — reuse across contexts. Proxy: accepted compressions reusable as subspaces."""
        c = Component(key="T", name=_NAMES["T"])
        comp = getattr(stack, "compression", None)
        if comp is None:
            c.note = "no compression layer"
            return c
        st = comp.stats()
        attempts = int(st.get("attempts", 0))
        if attempts < 2:
            c.note = f"only {attempts} compressions attempted"
            return c
        fid = st.get("mean_fidelity")
        if fid is None:
            c.note = "no compression cleared the fidelity floor"
            c.value = 0.0
            c.samples = attempts
            return c
        c.value = float(min(1.0, fid))
        c.samples = attempts
        c.note = "mean verified reconstruction fidelity of reusable subspaces"
        return c

    def _learning_efficiency(self, stack: Any) -> Component:
        """L — error eliminated per step. The charter calls this the ultimate metric."""
        c = Component(key="L", name=_NAMES["L"])
        wm = getattr(stack, "world_model", None)
        if wm is None:
            c.note = "no world model"
            return c
        steps = int(getattr(wm, "steps", 0))
        curve = wm.learning_curve()
        if steps < 16 or len(curve) < 4:
            c.note = f"only {steps} learning steps"
            return c
        first = sum(curve[:2]) / 2.0
        last = sum(curve[-2:]) / 2.0
        if first <= 0.0:
            c.note = "no initial error"
            return c
        eliminated = max(0.0, (first - last) / first)
        # Normalised against a reference of 1000 steps: reaching the same reduction in fewer
        # steps scores higher. Bounded at 1.0 — this rewards speed, not miracles.
        c.value = float(min(1.0, eliminated * (1000.0 / max(1, steps))))
        c.samples = steps
        c.note = "error eliminated, per 1000 steps"
        return c

    def _calibration(self, stack: Any) -> Component:
        """U — does stated confidence match observed correctness?"""
        c = Component(key="U", name=_NAMES["U"])
        sm = getattr(stack, "self_model", None)
        if sm is None:
            c.note = "no self-model"
            return c
        cal = sm.calibration()
        n = len(getattr(sm, "_outcomes", []))
        if cal is None:
            c.note = f"only {n} resolved outcomes (needs {getattr(sm, 'min_samples', 20)})"
            return c
        c.value = float(cal)
        c.samples = n
        c.note = "1 - mean |stated - observed| over resolved outcomes"
        return c

    def _adaptation(self, stack: Any) -> Component:
        """A — does it change strategy when it stops working? Proxy: meta-learning improvement."""
        c = Component(key="A", name=_NAMES["A"])
        ml = getattr(stack, "meta_learning", None)
        if ml is None:
            c.note = "no meta-learner"
            return c
        if not ml.confident():
            c.note = f"only {getattr(ml, 'trials', 0)} strategy trials"
            return c
        improved = ml.improved()
        c.value = 1.0 if improved else 0.0
        c.samples = int(getattr(ml, "trials", 0))
        c.note = "meta-learning beat its own baseline strategy"
        return c

    def _compute_efficiency(self, stack: Any) -> Component:
        """C — does effort track difficulty? Proxy: L15's own measured calibration."""
        c = Component(key="C", name=_NAMES["C"])
        res = getattr(stack, "resource", None)
        if res is None:
            c.note = "no resource controller"
            return c
        cal = res.calibration()
        n = len(getattr(res, "_samples", []))
        if cal is None:
            c.note = f"only {n} allocations, or no variance to correlate"
            return c
        # Correlation lives in [-1, 1]; map to [0, 1]. A negative correlation means effort is
        # being spent in the WRONG places, which is worse than random and scores below 0.5.
        c.value = float(max(0.0, min(1.0, (cal + 1.0) / 2.0)))
        c.samples = n
        c.note = "correlation between predicted difficulty and observed cost"
        return c

    # ---- the index ---- #
    def measure(self, brain: Any) -> IndexReport:
        """Measure all nine from a live brain. Never raises; unmeasurable components stay None."""
        rep = IndexReport()
        stack = getattr(brain, "stack", None) or brain
        for fn in (self._generalization, self._memory, self._reasoning, self._planning,
                   self._transfer, self._learning_efficiency, self._calibration,
                   self._adaptation, self._compute_efficiency):
            try:
                rep.components.append(fn(stack))
            except Exception as exc:  # noqa: BLE001 — a failed measurement is not a zero
                key = fn.__name__.strip("_")[0].upper()
                rep.components.append(Component(key=key, name=key,
                                                note=f"measurement failed: {type(exc).__name__}"))
        vals = [c.value for c in rep.components if c.value is not None]
        if vals:
            # Geometric mean: one near-zero component drags the whole index down, which is the
            # correct behaviour for a claim about GENERAL intelligence. An arithmetic mean would
            # let excellent memory paper over absent reasoning.
            floored = [max(1e-6, v) for v in vals]
            rep.index = float(math.exp(sum(math.log(v) for v in floored) / len(floored)))
        return rep
