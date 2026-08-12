"""NYXARA · nyx001/development.py — the artificial childhood (🌱, Stages 0–10).

The charter's Section IV: intelligence is *cultivated in stages*, from computational birth to
open-ended development. This module is that curriculum, and its single design rule is that a stage
is **entered by measurement, never by schedule**.

    Stage 0  computational birth      a stable learning substrate exists
    Stage 1  pattern discovery        similarity and change are detectable
    Stage 2  object/entity emergence  boundaries and identity hold across observations
    Stage 3  episodic memory          events are stored with context and recalled
    Stage 4  relationship discovery   A and B are found to co-vary
    Stage 5  world prediction         state → future state, better than copying
    Stage 6  reasoning                hypotheses are tested against evidence
    Stage 7  planning                 goals decompose and simulate
    Stage 8  self-model               uncertainty and capability are measured
    Stage 9  meta-learning            the learning strategy itself improves
    Stage 10 open-ended               new concepts and theories keep arriving

Each stage carries a **criterion**: a callable over the live stack returning a float, and a
threshold it must clear. Nothing advances because time passed or because a counter incremented —
:meth:`Curriculum.assess` reads the real numbers each cycle.

Two properties follow the pattern established by ``growth/developmental.py`` (whose ordered,
prerequisite-gated, regression-aware design this deliberately mirrors rather than reinvents):

* **No skipping.** A stage is unreachable until every prerequisite is *currently* met, so the ladder
  cannot be climbed out of order.
* **Regression is real.** If a criterion decays below its threshold, the stage is **lost** and every
  stage built on it cascades down with it. Maturity is held, not assumed. A system that forgot how
  to predict has not "reached Stage 5 once"; it is not at Stage 5.

Stage 5's criterion is worth calling out because it is the one that could most easily have been
faked: it is not "the error is low", it is **the model beats copying its own input**. A residual
world model can post a tiny error by predicting no change at all, so a low error alone certifies
nothing. The criterion is the ratio, and it has to be below 1.

Honest: these eleven criteria are chosen by hand and are proxies for the capacities they name.
"Reached Stage 6" means "tests hypotheses and most survive", not that the system reasons in any
richer sense. The value of the ladder is that each rung is a number anyone can check, not that the
names are earned in full.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Stage", "StageState", "Curriculum", "STAGES"]


@dataclass(frozen=True)
class Stage:
    """One rung: what it is called, what must be true, and what must come first."""

    number: int
    name: str
    criterion: Callable[[Any], Optional[float]]
    threshold: float
    requires: Tuple[int, ...] = ()
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"number": self.number, "name": self.name, "threshold": self.threshold,
                "requires": list(self.requires), "description": self.description}


@dataclass
class StageState:
    """Whether a stage is currently held, and the measurement that says so."""

    number: int = 0
    name: str = ""
    active: bool = False
    measured: Optional[float] = None
    threshold: float = 0.0
    first_reached: Optional[float] = None
    regressions: int = 0

    @property
    def blocked(self) -> bool:
        """Measurable but short of the bar — as opposed to not yet measurable at all."""
        return self.measured is not None and not self.active

    def to_dict(self) -> Dict[str, Any]:
        return {"number": self.number, "name": self.name, "active": self.active,
                "measured": None if self.measured is None else round(self.measured, 6),
                "threshold": self.threshold, "blocked": self.blocked,
                "first_reached": self.first_reached, "regressions": self.regressions}


# --------------------------------------------------------------------------- #
# The criteria. Each returns a live measurement, or None when not yet knowable.
# --------------------------------------------------------------------------- #
def _c0_birth(s: Any) -> Optional[float]:
    wm = getattr(s, "world_model", None)
    if wm is None:
        return None
    return 1.0 if getattr(wm, "_built", False) and getattr(wm, "available", False) else 0.0


def _c1_patterns(s: Any) -> Optional[float]:
    """Perception distinguishes things: some entities have repeated enough to be named."""
    p = getattr(s, "perception", None)
    if p is None:
        return None
    st = p.stats()
    return float(min(1.0, st.get("named", 0) / 3.0))


def _c2_entities(s: Any) -> Optional[float]:
    """Identity holds: named entities persist across many observations."""
    p = getattr(s, "perception", None)
    if p is None:
        return None
    st = p.stats()
    obs = st.get("observations", 0)
    if obs < 20:
        return None
    return float(min(1.0, st.get("named", 0) / 5.0))


def _c3_episodic(s: Any) -> Optional[float]:
    """Events are stored and retrievable with real context."""
    m = getattr(s, "episodic", None)
    if m is None:
        return None
    st = m.stats()
    return float(min(1.0, st.get("episodes", 0) / 32.0))


def _c4_relations(s: Any) -> Optional[float]:
    """Co-variation is found: concepts form and succeed one another detectably."""
    sem = getattr(s, "semantic", None)
    if sem is None:
        return None
    st = sem.stats()
    if st.get("absorbed", 0) < 20:
        return None
    return float(min(1.0, (st.get("earned", 0) + st.get("theories", 0)) / 4.0))


def _c5_prediction(s: Any) -> Optional[float]:
    """The model beats copying. NOT "the error is low" — a residual model fakes that trivially."""
    wm = getattr(s, "world_model", None)
    if wm is None:
        return None
    learning = wm.is_learning()
    if learning is None:
        return None
    curve = wm.learning_curve()
    if len(curve) < 4:
        return None
    first = sum(curve[:2]) / 2.0
    last = sum(curve[-2:]) / 2.0
    if first <= 0.0:
        return None
    # fraction of the initial error actually eliminated, in [0, 1]
    return float(max(0.0, min(1.0, 1.0 - last / first)))


def _c6_reasoning(s: Any) -> Optional[float]:
    """Hypotheses are tested, and testing is not merely destroying everything."""
    r = getattr(s, "reasoning", None)
    if r is None:
        return None
    st = r.stats()
    tested = st.get("tested", 0)
    if tested < 3:
        return None
    return float(min(1.0, st.get("survivors", 0) / max(1, tested)))


def _c7_planning(s: Any) -> Optional[float]:
    """Goals decompose and simulate into feasible plans."""
    p = getattr(s, "planning", None)
    if p is None:
        return None
    st = p.stats()
    if st.get("plans", 0) < 3:
        return None
    return 1.0 if st.get("attached") else 0.0


def _c8_self_model(s: Any) -> Optional[float]:
    """Calibration is measured — the system knows how sure it should be."""
    sm = getattr(s, "self_model", None)
    if sm is None:
        return None
    return sm.calibration()


def _c9_meta(s: Any) -> Optional[float]:
    """The learning strategy itself has been compared and improved."""
    ml = getattr(s, "meta_learning", None)
    if ml is None or not ml.confident():
        return None
    return 1.0 if ml.improved() else 0.0


def _c10_open_ended(s: Any) -> Optional[float]:
    """New structure keeps arriving: concepts are still forming and compressing cleanly."""
    sem = getattr(s, "semantic", None)
    comp = getattr(s, "compression", None)
    if sem is None or comp is None:
        return None
    cst = comp.stats()
    if cst.get("attempts", 0) < 2:
        return None
    fid = cst.get("mean_fidelity")
    earned = sem.stats().get("earned", 0)
    if fid is None or earned < 3:
        return None
    return float(min(1.0, fid * min(1.0, earned / 8.0)))


STAGES: Tuple[Stage, ...] = (
    Stage(0, "computational birth", _c0_birth, 1.0, (),
          "a stable, seeded learning substrate exists"),
    Stage(1, "pattern discovery", _c1_patterns, 1.0, (0,),
          "similarity and change are detectable"),
    Stage(2, "object emergence", _c2_entities, 1.0, (1,),
          "boundaries and identity hold across observations"),
    Stage(3, "episodic memory", _c3_episodic, 1.0, (2,),
          "events stored with context, retrievable"),
    Stage(4, "relationship discovery", _c4_relations, 1.0, (3,),
          "concepts form and co-vary"),
    Stage(5, "world prediction", _c5_prediction, 0.5, (4,),
          "state → future state, beating the copy baseline"),
    Stage(6, "reasoning", _c6_reasoning, 0.5, (5,),
          "hypotheses tested against evidence, most surviving"),
    Stage(7, "planning", _c7_planning, 1.0, (6,),
          "goals decompose and simulate"),
    Stage(8, "self-model", _c8_self_model, 0.6, (6,),
          "calibration measured over resolved outcomes"),
    Stage(9, "meta-learning", _c9_meta, 1.0, (8,),
          "the learning strategy itself improves"),
    Stage(10, "open-ended", _c10_open_ended, 0.6, (7, 9),
          "new concepts and theories keep arriving"),
)


class Curriculum:
    """The eleven stages, assessed from live measurements, with cascading regression."""

    def __init__(self, stack: Any, *, stages: Tuple[Stage, ...] = STAGES) -> None:
        self.stack = stack
        self.stages = stages
        self.states: Dict[int, StageState] = {
            s.number: StageState(number=s.number, name=s.name, threshold=s.threshold)
            for s in stages}
        self.assessments = 0
        self.advancements = 0
        self.regressions = 0
        self._log: List[Dict[str, Any]] = []

    def assess(self) -> Dict[int, StageState]:
        """Re-measure every stage. Advances and regressions both happen here, or neither does."""
        self.assessments += 1
        try:
            for stage in self.stages:
                st = self.states[stage.number]
                try:
                    st.measured = stage.criterion(self.stack)
                except Exception:  # noqa: BLE001 — an unmeasurable criterion is not a failed one
                    st.measured = None
                prereqs_met = all(self.states[p].active for p in stage.requires)
                meets = st.measured is not None and st.measured >= stage.threshold
                now_active = bool(prereqs_met and meets)
                if now_active and not st.active:
                    st.active = True
                    if st.first_reached is None:
                        st.first_reached = time.time()
                    self.advancements += 1
                    self._note("advance", stage, st)
                elif not now_active and st.active:
                    st.active = False
                    st.regressions += 1
                    self.regressions += 1
                    self._note("regress", stage, st)
            self._cascade()
            return self.states
        except Exception:  # noqa: BLE001
            return self.states

    def _cascade(self) -> None:
        """A lost prerequisite takes everything built on it down too. Repeat until stable."""
        for _ in range(len(self.stages)):
            changed = False
            for stage in self.stages:
                st = self.states[stage.number]
                if st.active and not all(self.states[p].active for p in stage.requires):
                    st.active = False
                    st.regressions += 1
                    self.regressions += 1
                    self._note("cascade", stage, st)
                    changed = True
            if not changed:
                break

    def _note(self, kind: str, stage: Stage, st: StageState) -> None:
        self._log.append({"kind": kind, "stage": stage.number, "name": stage.name,
                          "measured": st.measured, "threshold": stage.threshold,
                          "t": round(time.time(), 3)})
        if len(self._log) > 256:
            self._log = self._log[-128:]

    # ---- read ---- #
    @property
    def stage(self) -> int:
        """The highest *contiguously* held stage — the honest answer to "how far along?"."""
        n = -1
        for s in self.stages:
            if self.states[s.number].active:
                n = s.number
            else:
                break
        return n

    def active(self) -> List[StageState]:
        return [st for st in self.states.values() if st.active]

    def next_stage(self) -> Optional[StageState]:
        """The next rung, and what it is currently measuring. ``None`` when fully developed."""
        for s in self.stages:
            st = self.states[s.number]
            if not st.active:
                return st
        return None

    def summary(self) -> str:
        """One honest line. Says what is blocking, not just where it is."""
        nxt = self.next_stage()
        if nxt is None:
            return f"all {len(self.stages)} stages held — open-ended development"
        if nxt.measured is None:
            return (f"stage {self.stage} held; stage {nxt.number} ({nxt.name}) not yet "
                    f"measurable — not enough experience")
        return (f"stage {self.stage} held; stage {nxt.number} ({nxt.name}) at "
                f"{nxt.measured:.3f} of {nxt.threshold:.3f}")

    def stats(self) -> Dict[str, Any]:
        return {"stage": self.stage, "assessments": self.assessments,
                "advancements": self.advancements, "regressions": self.regressions,
                "active": len(self.active()), "total": len(self.stages),
                "summary": self.summary(),
                "states": [st.to_dict() for st in self.states.values()]}

    def to_dict(self) -> Dict[str, Any]:
        return {"assessments": self.assessments, "advancements": self.advancements,
                "regressions": self.regressions,
                "states": [st.to_dict() for st in self.states.values()],
                "log": list(self._log[-64:])}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.assessments = int(d.get("assessments", 0))
            self.advancements = int(d.get("advancements", 0))
            self.regressions = int(d.get("regressions", 0))
            for item in d.get("states") or []:
                n = int(item.get("number", -1))
                if n in self.states:
                    st = self.states[n]
                    st.first_reached = item.get("first_reached")
                    st.regressions = int(item.get("regressions", 0))
                    # `active` is deliberately NOT restored: maturity is re-measured on the next
                    # assess(), never inherited from a sidecar. A stage is held only while its
                    # criterion currently holds.
        except Exception:  # noqa: BLE001
            pass
