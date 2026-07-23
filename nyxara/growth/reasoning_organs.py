"""NYXARA · growth/reasoning_organs.py — wire her existing reasoning organs into the loop (J·K·L·M).

These faculties already exist as bounded engines elsewhere in the package; this module composes them into
small, testable helpers the autonomous decision + self-heal pipeline can call, each with a firm honesty
boundary. It adds **no new "physics"** — it makes the organs NYXARA already has usable unattended.

* **J — parallel-hypothesis forking.** Fork N candidate resolutions as amplitude-weighted hypotheses in a
  :class:`~nyxara.quantum.superposition_states.Superposition`, score each by a caller-supplied simulator,
  and **collapse to the best**. Honest: N hypothesis sandboxes in one process, scored and collapsed — not
  physical parallel universes.
* **K — self-invented axioms.** Ask :mod:`nyxara.growth.meta_epistemology` to invent a consistency-checked
  axiom when a goal is unprovable. Honest: bounded axiom-template invention with a mechanical consistency
  check — not new physics. Defensive: a missing engine is a clean no-op.
* **L — backward goal propagation.** Backward-induct the next micro-action from the terminal goal using
  the existing planners. Honest: means-ends regression over simulated futures — not retrocausality.
* **M — metacognitive self-mirror.** A finite, sampled monitor over the loop's OWN telemetry
  (unproductive streaks, a stage that keeps erroring) that flags a sub-optimal cognitive loop so the mind
  can break it. Honest: it watches key signals, not literally every weight.

Pure standard library at the seams; no LLM in the wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["ForkChoice", "fork_and_collapse", "MetacognitiveMonitor", "MetacognitiveSignal",
           "invent_axiom_if_stuck", "backward_next_action"]


# --------------------------------------------------------------------------- #
# J — parallel-hypothesis forking + collapse
# --------------------------------------------------------------------------- #
@dataclass
class ForkChoice:
    chosen: Any
    probability: float
    decided: bool
    scores: Dict[Any, float] = field(default_factory=dict)
    payload: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"chosen": self.chosen, "probability": round(self.probability, 6),
                "decided": self.decided,
                "scores": {str(k): round(v, 6) for k, v in self.scores.items()}}


def fork_and_collapse(candidates: Sequence[Any], score: Callable[[Any], float], *,
                      collapse_threshold: float = 0.5) -> Optional[ForkChoice]:
    """Fork ``candidates`` as weighted hypotheses, score each, and collapse to the best.

    ``score`` simulates a candidate's outcome as a non-negative amplitude (higher = better/safer). A
    candidate that raises during scoring is treated as amplitude 0 (a failed branch), never fatal. Returns
    None for an empty candidate set."""
    cands = list(candidates)
    if not cands:
        return None
    from nyxara.quantum.superposition_states import Superposition
    sp = Superposition(collapse_threshold=collapse_threshold)
    scores: Dict[Any, float] = {}
    for i, c in enumerate(cands):
        label = i                       # index label keeps unhashable candidates usable
        try:
            amp = max(0.0, float(score(c)))
        except Exception:  # noqa: BLE001 — a failed branch simply carries no amplitude
            amp = 0.0
        scores[c if _hashable(c) else label] = amp
        sp.add(label, amplitude=amp, payload=c)
    result = sp.collapse(force=True)
    return ForkChoice(chosen=result.payload, probability=result.probability,
                      decided=result.decided, scores=scores, payload=result.payload)


def _hashable(x: Any) -> bool:
    try:
        hash(x)
        return True
    except TypeError:
        return False


# --------------------------------------------------------------------------- #
# M — metacognitive self-mirror
# --------------------------------------------------------------------------- #
@dataclass
class MetacognitiveSignal:
    suboptimal: bool
    reason: str = ""
    stage: str = ""           # the stage stuck in a loop, if any
    should_break: bool = False   # True ⇒ route into self-heal / re-plan

    def to_dict(self) -> Dict[str, Any]:
        return {"suboptimal": self.suboptimal, "reason": self.reason,
                "stage": self.stage, "should_break": self.should_break}


class MetacognitiveMonitor:
    """Watch the loop's own telemetry for a sub-optimal cognitive loop and flag it to be broken.

    Bounded + sampled: it reads a few key signals (an unproductive streak, a single stage that keeps
    erroring across samples) — not every weight. It emits a *signal*; breaking the loop is the caller's
    action (route into self-heal / re-plan)."""

    def __init__(self, *, stall_limit: int = 5, stage_error_limit: int = 3) -> None:
        self.stall_limit = stall_limit
        self.stage_error_limit = stage_error_limit
        self._prev_stage_errors: Dict[str, int] = {}
        self._stage_repeat: Dict[str, int] = {}

    def observe(self, *, consecutive_unproductive: int,
                stage_errors: Optional[Dict[str, int]] = None) -> MetacognitiveSignal:
        stage_errors = dict(stage_errors or {})
        # a stage that GREW its error count again this sample is stuck in a loop
        worst_stage = ""
        for stage, count in stage_errors.items():
            grew = count > self._prev_stage_errors.get(stage, 0)
            self._stage_repeat[stage] = (self._stage_repeat.get(stage, 0) + 1) if grew else 0
            if self._stage_repeat[stage] >= self.stage_error_limit:
                worst_stage = stage
        self._prev_stage_errors = stage_errors
        if worst_stage:
            return MetacognitiveSignal(True, f"stage {worst_stage!r} keeps erroring",
                                       stage=worst_stage, should_break=True)
        if consecutive_unproductive >= self.stall_limit:
            return MetacognitiveSignal(True, "prolonged unproductive streak", should_break=True)
        return MetacognitiveSignal(False, "reasoning healthy")


# --------------------------------------------------------------------------- #
# K — self-invented axioms (defensive adapter over the existing engine)
# --------------------------------------------------------------------------- #
def invent_axiom_if_stuck(goal: Any, *, engine: Any = None) -> Optional[Dict[str, Any]]:
    """Ask the meta-epistemology engine to invent a consistency-checked axiom for an unprovable ``goal``.

    Defensive wiring: if no engine is supplied and none can be built, this is a clean no-op (None). Never
    raises. The invented axiom is admitted by the engine only if it is consistent — that check is the
    engine's, not weakened here."""
    if engine is None:
        try:
            from nyxara.growth.meta_epistemology import AxiomaticExpansion  # type: ignore
            engine = AxiomaticExpansion()
        except Exception:  # noqa: BLE001 — engine optional; no-op when unavailable
            return None
    for name in ("invent", "expand", "extend", "run"):
        fn = getattr(engine, name, None)
        if callable(fn):
            try:
                res = fn(goal)
            except TypeError:
                try:
                    res = fn()
                except Exception:  # noqa: BLE001
                    return None
            except Exception:  # noqa: BLE001
                return None
            if res is None:
                return None
            if isinstance(res, dict):
                return res
            return res.to_dict() if hasattr(res, "to_dict") else {"result": repr(res)}
    return None


# --------------------------------------------------------------------------- #
# L — backward goal propagation (defensive adapter over the existing planners)
# --------------------------------------------------------------------------- #
def backward_next_action(goal: Any, *, planner: Any = None) -> Optional[Any]:
    """Backward-induct the next concrete micro-action toward ``goal`` using an existing planner.

    Defensive wiring: with no planner available this is a clean no-op (None). Never raises. This is
    means-ends regression over simulated futures — a standard planning technique, not retrocausality."""
    if planner is None:
        return None
    for name in ("plan", "next_action", "backward", "step"):
        fn = getattr(planner, name, None)
        if callable(fn):
            try:
                return fn(goal)
            except TypeError:
                try:
                    return fn()
                except Exception:  # noqa: BLE001
                    return None
            except Exception:  # noqa: BLE001
                return None
    return None
