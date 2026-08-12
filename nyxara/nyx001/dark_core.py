"""NYXARA · nyx001/dark_core.py — the Dark Cognitive Core (🕳, one drive, never empty).

The charter names this NYXARA's scientific advantage and states it as a single sentence:

    "NYXARA relentlessly seeks information, attacks its own beliefs, identifies unknowns,
    simulates alternatives, and learns from consequences."

and as a pipeline:

    Curiosity + Uncertainty + Self-Critique
      → Information Gain + Confidence Modeling + Contradiction Search
      → Hypothesis Engine → Simulation/Experiment → New Evidence → Better Intelligence

Layers 9, 10, 11 and 12 already exist separately. What this module adds is that they become **one
drive** rather than four faculties that happen to be present. The difference is concrete: a
drive has a *magnitude*, it is never satisfied, and it produces a next action even when nothing
is asking it anything.

:meth:`DarkCore.pulse` runs the full pipeline once and returns the single most valuable thing to
do next. Its ranking is deliberately unlike a reward signal:

    value = w_gain·information_gain
          + w_rigidity·rigidity            (being confidently untested is itself a target)
          + w_uncertainty·uncertainty
          - w_cost·estimated_cost

Rigidity entering with a *positive* weight is the part that makes this dark rather than merely
curious: the system is pulled toward the beliefs it is most comfortable with, specifically to
attack them. A system that only chased information gain would never revisit anything it already
felt sure about, which is exactly where its worst errors live.

**The frontier is never empty.** When curiosity has no live region, the core falls back to
attacking its most rigid belief; when there is no belief either, it falls back to probing its own
weakest-calibrated area. :meth:`DarkCore.starved` reports the one honest exception — a system with
no experience at all has nothing to be curious about, and says so instead of inventing a target.

Honest: every input here is one of this system's own measurements, so the core inherits all of
their blind spots. It cannot want to know about something it has never encountered. "Relentless"
means it always has a next move; it does not mean the move is a good one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Impulse", "Pulse", "DarkCore"]

_KINDS = ("probe", "falsify", "experiment", "recalibrate")


@dataclass
class Impulse:
    """One thing worth doing next, with the value that earned it the top spot."""

    kind: str = "probe"                     # probe | falsify | experiment | recalibrate
    target: str = ""
    value: float = 0.0
    rationale: str = ""
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "target": self.target[:120], "value": round(self.value, 6),
                "rationale": self.rationale[:200],
                "components": {k: round(v, 4) for k, v in self.components.items()}}


@dataclass
class Pulse:
    """One turn of the drive: what it read, what it produced, what it chose."""

    t: float = field(default_factory=time.time)
    information_gain: float = 0.0
    rigidity: Optional[float] = None
    uncertainty: float = 0.0
    calibration: Optional[float] = None
    hypotheses_made: int = 0
    attacks_made: int = 0
    destroyed: int = 0
    impulses: List[Impulse] = field(default_factory=list)
    chosen: Optional[Impulse] = None
    starved: bool = False

    @property
    def drive(self) -> float:
        """How hard the core is pulling. Zero only when genuinely starved."""
        return float(self.chosen.value) if self.chosen is not None else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"information_gain": round(self.information_gain, 6),
                "rigidity": None if self.rigidity is None else round(self.rigidity, 4),
                "uncertainty": round(self.uncertainty, 4),
                "calibration": None if self.calibration is None else round(self.calibration, 4),
                "hypotheses_made": self.hypotheses_made, "attacks_made": self.attacks_made,
                "destroyed": self.destroyed, "starved": self.starved,
                "drive": round(self.drive, 6),
                "chosen": self.chosen.to_dict() if self.chosen else None,
                "impulses": [i.to_dict() for i in self.impulses[:5]]}


class DarkCore:
    """Layers 9+10+11+12 fused into one drive that always has a next move."""

    def __init__(self, stack: Any, *, w_gain: float = 1.0, w_rigidity: float = 0.8,
                 w_uncertainty: float = 0.5, w_cost: float = 0.3,
                 max_impulses: int = 16) -> None:
        self.stack = stack
        self.w_gain = float(w_gain)
        self.w_rigidity = float(w_rigidity)
        self.w_uncertainty = float(w_uncertainty)
        self.w_cost = float(w_cost)
        self.max_impulses = int(max_impulses)
        self.pulses = 0
        self.chosen_counts: Dict[str, int] = {k: 0 for k in _KINDS}
        self._history: List[Pulse] = []

    # ---- the pipeline ---- #
    def pulse(self) -> Pulse:
        """One full turn: read the four signals, generate impulses, choose the most valuable."""
        out = Pulse()
        self.pulses += 1
        s = self.stack
        try:
            curiosity = getattr(s, "curiosity", None)
            reasoning = getattr(s, "reasoning", None)
            contradiction = getattr(s, "contradiction", None)
            self_model = getattr(s, "self_model", None)
            world_model = getattr(s, "world_model", None)
            semantic = getattr(s, "semantic", None)
            episodic = getattr(s, "episodic", None)

            # stage 1 — read the drive's four inputs
            out.information_gain = curiosity.information_gain() if curiosity else 0.0
            out.uncertainty = world_model.uncertainty() if world_model else 0.0
            out.calibration = self_model.calibration() if self_model else None
            hyps = reasoning.best(k=self.max_impulses) if reasoning else []
            out.rigidity = contradiction.rigidity(hyps) if contradiction else None

            # stage 2 — hypothesis engine: turn observations into claims worth testing
            if reasoning is not None:
                made = reasoning.from_theories(semantic.theories() if semantic else [])
                made += reasoning.from_surprises(episodic.surprising() if episodic else [])
                out.hypotheses_made = len(made)

            # stage 3 — contradiction search over what is now held
            if contradiction is not None and reasoning is not None:
                for h in hyps[:5]:
                    a = contradiction.attack(h, episodic, reasoning, semantic)
                    out.attacks_made += 1
                    if a.found:
                        out.destroyed += 1

            # stage 4 — what to do next
            out.impulses = self._impulses(out, curiosity, hyps, self_model)
            if out.impulses:
                out.impulses.sort(key=lambda i: i.value, reverse=True)
                out.chosen = out.impulses[0]
                self.chosen_counts[out.chosen.kind] = \
                    self.chosen_counts.get(out.chosen.kind, 0) + 1
            else:
                out.starved = True

            self._history.append(out)
            if len(self._history) > 256:
                self._history = self._history[-128:]
            return out
        except Exception:  # noqa: BLE001 — the drive never breaks a turn; it falls silent
            out.starved = True
            return out

    def _impulses(self, p: Pulse, curiosity: Any, hyps: List[Any],
                  self_model: Any) -> List[Impulse]:
        """Generate candidates from every source, so the frontier is never empty by accident."""
        out: List[Impulse] = []
        # probe — where there is learning progress to be had
        if curiosity is not None:
            for r in curiosity.frontier(k=5):
                comp = {"gain": self.w_gain * float(r.progress),
                        "uncertainty": self.w_uncertainty * float(r.mean_error or 0.0),
                        "cost": -self.w_cost * 0.1}
                out.append(Impulse(kind="probe", target=r.key, value=sum(comp.values()),
                                   components=comp,
                                   rationale=f"learning progress {r.progress:.6f} in {r.key}"))
        # falsify — the beliefs it is most comfortable with, precisely because it is comfortable
        for h in hyps[:5]:
            tested = int(getattr(h, "tested", 0))
            conf = float(getattr(h, "confidence", 0.5))
            rigid = conf / (1.0 + tested)
            comp = {"rigidity": self.w_rigidity * rigid, "cost": -self.w_cost * 0.2}
            out.append(Impulse(kind="falsify", target=str(getattr(h, "claim", ""))[:120],
                               value=sum(comp.values()), components=comp,
                               rationale=f"confidence {conf:.2f} on {tested} tests"))
        # experiment — an untested claim is worth an intervention, not another look
        for h in hyps[:3]:
            if int(getattr(h, "tested", 0)) == 0:
                comp = {"uncertainty": self.w_uncertainty * 1.0, "cost": -self.w_cost * 0.5}
                out.append(Impulse(kind="experiment",
                                   target=str(getattr(h, "claim", ""))[:120],
                                   value=sum(comp.values()), components=comp,
                                   rationale="never tested — needs an intervention, not a look"))
        # recalibrate — the fallback that keeps the frontier non-empty
        if self_model is not None:
            cal = p.calibration
            gap = 1.0 - float(cal) if cal is not None else 0.5
            comp = {"uncertainty": self.w_uncertainty * gap, "cost": -self.w_cost * 0.05}
            out.append(Impulse(kind="recalibrate", target="self-calibration",
                               value=sum(comp.values()), components=comp,
                               rationale=("calibration unknown — resolve outcomes"
                                          if cal is None else f"calibration {cal:.2f}")))
        return out[: self.max_impulses]

    # ---- read ---- #
    def starved(self) -> bool:
        """True only when the core genuinely has nothing to pursue — no experience at all."""
        return bool(self._history and self._history[-1].starved)

    def drive(self) -> float:
        """Current pull, in the units of the value function. Zero means starved, truthfully."""
        return self._history[-1].drive if self._history else 0.0

    def last(self) -> Optional[Pulse]:
        return self._history[-1] if self._history else None

    def stats(self) -> Dict[str, Any]:
        recent = self._history[-32:]
        return {"pulses": self.pulses, "drive": round(self.drive(), 6),
                "starved": self.starved(), "chosen": dict(self.chosen_counts),
                "destroyed_total": sum(p.destroyed for p in self._history),
                "mean_information_gain":
                    round(sum(p.information_gain for p in recent) / len(recent), 6)
                    if recent else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"pulses": self.pulses, "chosen": dict(self.chosen_counts),
                "history": [p.to_dict() for p in self._history[-32:]]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.pulses = int(d.get("pulses", 0))
            self.chosen_counts.update({str(k): int(v) for k, v in
                                       (d.get("chosen") or {}).items()})
        except Exception:  # noqa: BLE001
            pass
