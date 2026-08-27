"""NYXARA · njp/rollout.py — a goal reached by searching imagined futures (🧭, NJP V.02).

:meth:`~nyxara.njp.universe.InternalUniverse.imagine` rolls the learned causal model forward under
an intervention and *registers* the result, so
:meth:`~nyxara.njp.universe.InternalUniverse.reconcile` can grade it later. That is a complete
simulator with a complete scoring channel, and it had **one caller in the whole package**::

    field.py:592   roll = self.universe.imagine("continue", {leading.cause: current}, steps=1)

One variable, chosen because it happened to have the best fit, set to the value it already had.
Nothing anywhere asked *"which intervention would get this number where I need it"*, which is the
only question a world model is for. Measured on a brain that had absorbed both corpora: 27 usable
oriented relations at R² 0.96–0.999, and ``rollouts: 0, reconciled: 0``.

**What planning is here.** A :class:`Target` names a variable and a value it has to reach.
:meth:`RolloutPlanner.levers` finds the variables from which that target is reachable along
oriented arrows — a reverse walk, so a two-hop route counts. Each lever is swept across the range
it was actually *observed* in, every setting is rolled forward, and each rollout is scored on how
much of the remaining distance it closes **times the confidence the model itself claimed**.

That last factor is the whole reason this is planning rather than arithmetic, and it is inherited
rather than invented. :meth:`~nyxara.njp.universe.InternalUniverse.intervene` already prices
confidence down as a setting leaves the range the relation was fitted on — measured, ``water=6``
comes back at ``0.791`` and ``water=10`` at ``0.475``. So a plan that would blast the target by
extrapolating far past anything ever seen loses to a plan that arrives on evidence. Overshoot
earns nothing either: progress is clipped at the target, so *arriving* and *overshooting* score
the same before confidence and differently after it.

**And a plan is not finished when it is chosen.** ``brain.py:2469`` records what the alternative
costs: *"113 turns, 113 predictions registered, 0 scored. A prediction that cannot in principle be
observed is not a prediction; it is a counter going up."* So the number this module is judged on
is :attr:`RolloutPlanner.settled` — plans whose imagined future was compared against a real
reading — and never :attr:`RolloutPlanner.planned`. :meth:`commit` registers the claim with
:class:`~nyxara.njp.predict.PredictionEngine` under a key the reading itself will resolve, and
:meth:`settle` grades it both ways: against the causal model, through
:meth:`~nyxara.njp.universe.InternalUniverse.grade`, and against the prediction engine, whose
``Outcome.surprise`` is the shared error signal Phase 2 wired into attention and memory. That is
the feedback edge — an imagined future that reality contradicted raises exactly the same signal as
any other prediction she got wrong.

**What this deliberately does not do: invent the goal.** A planner that also chooses what to want
can always report success. There is no direction in which ``growth`` is intrinsically better, and
declaring one would be writing a preference into a world model. Targets come from outside — from
:class:`~nyxara.njp.doing.Goal` when its metric names a variable the universe actually knows, or
from a caller. Goal *generation* is a separate mechanism and belongs to the curriculum.

Pure standard library. Duck-typed on the universe and the predictor, and fail-soft: a search that
raises is a search that found no plan, which from the outside is what it is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["Target", "Candidate", "Plan", "Settlement", "RolloutPlanner"]

#: How many levers a search will sweep. Every lever costs a full propagation over the state per
#: grid point, so the product is what actually bounds the work: 6 × 7 = 42 rollouts per search.
_MAX_LEVERS = 6

#: Settings tried per lever, across the range the lever was observed in. Odd, so the midpoint of
#: the observed range is always one of them.
_GRID = 9

#: How far past the observed range a setting may be proposed, as a fraction of that range at each
#: end. Half again in both directions: far enough that a target just outside what she has
#: measured is reachable, near enough that the model's own confidence penalty is still saying
#: something rather than bottoming out. See :meth:`RolloutPlanner._settings`.
_REACH = 0.5

#: How far upstream to look for a lever. Two hops is a real chain — altitude → temperature → rate
#: — and each further hop multiplies the confidence penalty, so a five-hop route is a plan she
#: would never have grounds to choose anyway.
_HORIZON = 3

#: Rollout depth. One step is the intervention propagating; further steps let the dynamics carry
#: it. Beyond a couple of steps the model is compounding its own output, which is a claim the
#: evidence does not support.
_STEPS = 2

#: Plans kept for the record.
_HISTORY = 64

#: Below this a "gain" is floating-point noise in a subtraction, not a plan doing anything.
_EPSILON = 1e-9


@dataclass(frozen=True)
class Target:
    """A world variable, and a value it has to reach.

    ``at_most`` is not decoration, for the same reason it is not in
    :class:`~nyxara.njp.doing.Goal`: half of what a causal model is asked for is a *reduction* —
    get the latency down, get the resting pulse down — and a planner that could only ask for
    numbers to go up would have to express those backwards.
    """

    variable: str = ""
    value: float = 0.0
    at_most: bool = False
    why: str = ""

    @property
    def named(self) -> bool:
        return bool(str(self.variable).strip())

    def met_by(self, reading: Optional[float]) -> bool:
        if reading is None:
            return False
        return reading <= self.value if self.at_most else reading >= self.value

    def need(self, before: Optional[float]) -> float:
        """How far there is to go from here, signed positive when there is work to do."""
        if before is None:
            return 0.0
        return (before - self.value) if self.at_most else (self.value - before)

    def gain(self, before: Optional[float], after: Optional[float]) -> float:
        """Movement in the direction that helps, so one number works for both kinds of target."""
        if before is None or after is None:
            return 0.0
        delta = after - before
        return -delta if self.at_most else delta

    @classmethod
    def from_goal(cls, goal: Any, universe: Any) -> Optional["Target"]:
        """Bridge a :class:`~nyxara.njp.doing.Goal` across, **only** if it names a real variable.

        The guard is the whole method. ``Goal.metric`` is ``"organ.key"`` read out of
        ``brain.stats()``, and a universe variable is very often dotted the same way —
        ``plant.growth`` — so the two namespaces overlap without being the same. Accepting a
        metric the universe has never heard of would produce a plan over a variable that does not
        exist, and every rollout of it would come back empty while the counters went up.
        """
        try:
            name = str(getattr(goal, "metric", "") or "")
            if not name or universe is None:
                return None
            if name not in getattr(universe, "state", {}):
                return None
            return cls(variable=name, value=float(getattr(goal, "target", 0.0)),
                       at_most=bool(getattr(goal, "at_most", False)),
                       why=str(getattr(goal, "why", "") or ""))
        except Exception:  # noqa: BLE001
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {"variable": self.variable, "value": round(self.value, 4),
                "direction": "at_most" if self.at_most else "at_least", "why": self.why[:160]}


@dataclass
class Candidate:
    """One imagined intervention, and how well it did in imagination."""

    action: str = ""
    lever: str = ""
    setting: float = 0.0
    predicted: Optional[float] = None      # what the target variable becomes
    confidence: float = 0.0
    progress: float = 0.0                  # share of the distance closed, clipped at 1.0
    score: float = 0.0                     # progress × confidence
    #: How far the lever has to move from where it is, relative to the range it was observed
    #: over — so a millivolt and a kilometre are comparable. The minimality tie-break.
    effort: float = 0.0
    reached: bool = False
    rollout: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "lever": self.lever, "setting": round(self.setting, 4),
                "predicted": None if self.predicted is None else round(self.predicted, 4),
                "confidence": round(self.confidence, 4), "progress": round(self.progress, 4),
                "score": round(self.score, 4), "effort": round(self.effort, 4),
                "reached": self.reached}


@dataclass
class Plan:
    """A search over imagined futures, and the one it would act on."""

    target: Optional[Target] = None
    before: Optional[float] = None
    candidates: List[Candidate] = field(default_factory=list)
    chosen: Optional[Candidate] = None
    levers: List[str] = field(default_factory=list)
    considered: int = 0
    why: str = ""
    committed: bool = False
    ms: float = 0.0

    @property
    def actionable(self) -> bool:
        return self.chosen is not None and self.chosen.score > _EPSILON

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target.to_dict() if self.target else None,
                "before": None if self.before is None else round(self.before, 4),
                "levers": list(self.levers), "considered": self.considered,
                "chosen": self.chosen.to_dict() if self.chosen else None,
                "runners_up": [c.to_dict() for c in self.candidates[1:4]],
                "actionable": self.actionable, "committed": self.committed,
                "why": self.why[:200], "ms": round(self.ms, 2)}


@dataclass
class Settlement:
    """What reality said about a plan. The half that makes the plan a prediction."""

    variable: str = ""
    imagined: Optional[float] = None
    actual: Optional[float] = None
    error: Optional[float] = None
    surprise: Optional[float] = None
    reached: bool = False
    scored: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"variable": self.variable,
                "imagined": None if self.imagined is None else round(self.imagined, 4),
                "actual": None if self.actual is None else round(self.actual, 4),
                "error": None if self.error is None else round(self.error, 5),
                "surprise": None if self.surprise is None else round(self.surprise, 5),
                "reached": self.reached, "scored": self.scored}


class RolloutPlanner:
    """Propose interventions, roll each forward, act on the best, and grade it against reality."""

    def __init__(self, brain: Any = None, *, universe: Any = None, predictor: Any = None,
                 max_levers: int = _MAX_LEVERS, grid: int = _GRID,
                 horizon: int = _HORIZON, steps: int = _STEPS) -> None:
        self.brain = brain
        self._universe = universe
        self._predictor = predictor
        self.max_levers = max(1, int(max_levers))
        self.grid = max(2, int(grid))
        self.horizon = max(1, int(horizon))
        self.steps = max(1, int(steps))

        self.history: List[Plan] = []
        self.settlements: List[Settlement] = []
        self.planned = 0
        self.committed = 0
        self.settled = 0            # the number this module is judged on
        self.reached = 0
        self.no_lever = 0
        self.total_error = 0.0
        #: The one outstanding claim. Exactly one, because a planner holding several open plans
        #: over the same variable cannot say which of them a reading settles.
        self._open: Optional[Tuple[Plan, str]] = None

    # ---- the organs it reads ------------------------------------------------- #
    @property
    def universe(self) -> Any:
        return self._universe if self._universe is not None else getattr(self.brain, "universe", None)

    @property
    def predictor(self) -> Any:
        return (self._predictor if self._predictor is not None
                else getattr(self.brain, "predictor", None))

    # ---- proposing ----------------------------------------------------------- #
    def levers(self, target: Target) -> List[str]:
        """Variables from which the target is reachable along arrows that have a direction.

        A reverse walk rather than a list of direct causes, because a chain is a plan: nothing
        sets ``temperature`` in ``altitude → temperature → rate``, and refusing to look one hop
        further would leave the only lever there is off the table. Depth first in the ranking —
        a direct cause is a shorter route and a shorter route compounds fewer confidence
        penalties — and the fit's R² second, because between two routes of equal length the one
        that was measured better is the one to try.

        The target itself is never a lever. Setting the variable you are trying to move is not a
        plan, it is asserting the answer, and the confidence attached would be the model's
        confidence in an identity.
        """
        out: List[Tuple[int, float, str]] = []
        try:
            universe = self.universe
            if universe is None or not target.named:
                return []
            incoming: Dict[str, List[Tuple[str, float]]] = {}
            for (cause, effect), relation in universe.relations.items():
                if not getattr(relation, "usable", False) or not getattr(relation, "oriented", False):
                    continue
                incoming.setdefault(effect, []).append((cause, float(getattr(relation, "r2", 0.0))))
            seen: Set[str] = {target.variable}
            frontier: List[Tuple[str, int]] = [(target.variable, 0)]
            while frontier:
                node, depth = frontier.pop(0)
                if depth >= self.horizon:
                    continue
                for cause, r2 in incoming.get(node, ()):
                    if cause in seen:
                        continue
                    seen.add(cause)
                    out.append((depth + 1, -r2, cause))
                    frontier.append((cause, depth + 1))
            out.sort()
            return [name for _d, _r, name in out[: self.max_levers]]
        except Exception:  # noqa: BLE001
            return []

    def _span(self, lever: str) -> float:
        """The range this lever was observed over — the unit ``Candidate.effort`` is measured in.

        Scale-free is the whole requirement. A voltage swept over ``0…24`` and an altitude swept
        over ``0…5`` produce moves whose raw sizes say nothing about which is the bigger ask, so
        a tie-break on raw distance would simply always prefer whichever lever had small units.
        """
        try:
            universe = self.universe
            lo, hi = None, None
            for (cause, _effect), relation in universe.relations.items():
                if cause != lever or not getattr(relation, "usable", False):
                    continue
                rlo, rhi = float(relation.lo), float(relation.hi)
                if rlo > rhi:
                    continue
                lo = rlo if lo is None else min(lo, rlo)
                hi = rhi if hi is None else max(hi, rhi)
            if lo is None or hi is None or hi - lo <= _EPSILON:
                return 1.0
            return hi - lo
        except Exception:  # noqa: BLE001
            return 1.0

    def _settings(self, lever: str) -> List[float]:
        """The values to try for one lever: the range it was observed in, and a margin past it.

        Anchored on observation — a relation knows the interval its data came from
        (``Relation.lo``/``hi``) — and then deliberately extended by :data:`_REACH` of that span
        at each end.

        **Clipping the sweep at the observed range was wrong, and measurably so.** Water was
        observed over ``0…6`` and ``growth ≥ 12`` needs about ``8.6``. Clipped, the whole search
        came back with ``water=6 → growth 9.02``, three percent of the way, reported as the best
        plan available — while the model, asked directly, answers ``water=10 → growth 13.67`` at
        confidence ``0.475``. The reachable plan scores ``1.0 × 0.475``; the clipped one scores
        ``0.03 × 0.79``. Twenty times better and structurally unproposable.

        The reason to allow it is that the honesty is already paid for somewhere else.
        :meth:`~nyxara.njp.universe.InternalUniverse.intervene` prices confidence down as a
        setting leaves the fitted range, so an extrapolated plan arrives *labelled* — "this needs
        more water than I have ever seen, and I am 48% sure" — which is a usable answer. Refusing
        to form it at all is not more honest, it is less informative: it reports "nothing helps"
        when what is true is "something helps, and I would be guessing".

        The margin is finite for the obvious reason. Unbounded extrapolation on a fitted line
        reaches any target whatsoever, and a planner that can always reach its target by naming a
        big enough number is not planning.

        The current value is always included: *change nothing* has to be on the table, or "no
        intervention helps" is not a conclusion the search can reach.
        """
        try:
            universe = self.universe
            lo, hi = None, None
            for (cause, _effect), relation in universe.relations.items():
                if cause != lever or not getattr(relation, "usable", False):
                    continue
                rlo, rhi = float(relation.lo), float(relation.hi)
                if rlo > rhi:                                   # never observed
                    continue
                lo = rlo if lo is None else min(lo, rlo)
                hi = rhi if hi is None else max(hi, rhi)
            current = universe.state.get(lever)
            if lo is None or hi is None or hi - lo <= _EPSILON:
                return [] if current is None else [float(current)]
            margin = (hi - lo) * _REACH
            floor, ceiling = lo, hi
            lo, hi = lo - margin, hi + margin
            # A lever never observed on one side of zero is not proposed there. Not a physics
            # rule — the universe has no notion of a variable's domain and inventing one would be
            # a guess — but an observational one: extrapolating a fitted line through a sign it
            # has never taken is a different and much larger claim than extrapolating past the
            # far end of its range, and the confidence penalty prices both identically. Measured,
            # without this: ``water = -1.5`` proposed as the third-best way to reduce growth.
            if floor >= 0.0:
                lo = max(0.0, lo)
            if ceiling <= 0.0:
                hi = min(0.0, hi)
            if hi - lo <= _EPSILON:
                return [] if current is None else [float(current)]
            step = (hi - lo) / (self.grid - 1)
            values = [lo + step * i for i in range(self.grid)]
            if current is not None and all(abs(v - float(current)) > _EPSILON for v in values):
                values.append(float(current))
            return values
        except Exception:  # noqa: BLE001
            return []

    # ---- searching ------------------------------------------------------------ #
    def search(self, target: Target, *, steps: Optional[int] = None) -> Plan:
        """Roll every setting of every lever forward, and rank them on what they buy."""
        plan = Plan(target=target)
        t0 = time.perf_counter()
        try:
            self.planned += 1
            universe = self.universe
            if universe is None or not target.named:
                plan.why = "no causal model, or no target named"
                return plan
            before = universe.state.get(target.variable)
            plan.before = None if before is None else float(before)
            if target.met_by(plan.before):
                plan.why = (f"{target.variable} already at {plan.before:g}, wants "
                            f"{'at most' if target.at_most else 'at least'} {target.value:g}")
                return plan
            need = target.need(plan.before)
            if need <= _EPSILON:
                plan.why = f"{target.variable} has nowhere to go from {plan.before}"
                return plan

            plan.levers = self.levers(target)
            if not plan.levers:
                self.no_lever += 1
                plan.why = (f"nothing she has an oriented arrow from reaches "
                            f"{target.variable} — that is a finding, not a plan")
                return plan

            for lever in plan.levers:
                baseline = self._baseline(target, lever, steps=steps)
                if baseline is None:
                    baseline = plan.before
                lever_need = target.need(baseline)
                if lever_need <= _EPSILON:
                    # Under this lever's own do-nothing rollout the target is already met, so
                    # there is nothing here for an intervention to buy. Skipped rather than
                    # scored against a need of zero.
                    continue
                for setting in self._settings(lever):
                    candidate = self._imagine(target, lever, setting, baseline, lever_need,
                                              steps=steps)
                    if candidate is not None:
                        plan.candidates.append(candidate)
                    plan.considered += 1
            # Score first, then the model's own confidence, then **the smallest move that does
            # it**. The third key is not a tidy tie-break: progress is clipped at the target, so
            # every plan that arrives scores identically however far past it lands, and without a
            # minimality rule the winner among them is whichever the grid happened to emit first.
            # Measured: ``water=0`` and ``water=1.5`` both reach ``growth ≤ 4`` at confidence
            # 0.7912, and the first was chosen for no reason at all. Asking for the least change
            # that achieves the goal is the ordinary meaning of a good plan.
            plan.candidates.sort(key=lambda c: (-c.score, -c.confidence, c.effort, c.lever))
            plan.chosen = plan.candidates[0] if plan.candidates else None
            if plan.chosen is None or plan.chosen.score <= _EPSILON:
                plan.why = (f"{plan.considered} futures imagined and none of them moves "
                            f"{target.variable} toward {target.value:g}")
                plan.chosen = None
                return plan
            chosen = plan.chosen
            plan.why = (f"{chosen.lever}={chosen.setting:g} → {target.variable} "
                        f"{plan.before:g} → {chosen.predicted:g} "
                        f"({chosen.progress:.0%} of the way, confidence {chosen.confidence:.2f})")
            return plan
        except Exception:  # noqa: BLE001
            plan.why = "the search failed"
            return plan
        finally:
            plan.ms = (time.perf_counter() - t0) * 1000.0
            self.history.append(plan)
            del self.history[:-_HISTORY]

    def _baseline(self, target: Target, lever: str, *,
                  steps: Optional[int] = None) -> Optional[float]:
        """What the target becomes if this lever is *held where it is*. The do-nothing contrast.

        Scoring against the recorded state instead was wrong, and it produced a plan that was not
        a plan. Measured on the corpus: ``rainfall`` sitting at 200, the river level recorded at a
        value the fitted line does not pass through, and the search returning *"set rainfall to
        200"* — a move of exactly zero — as the best way to raise the river. It scored because
        the model's prediction differs from the stored reading, and that difference is a stale
        measurement, not something an action achieves.

        A causal contrast is between doing this and doing nothing, never between doing this and
        what happens to be written down. Held per lever rather than once, because holding a lever
        severs its own incoming arrows exactly as setting it would, so the only comparison that
        isolates the intervention is against the same rollout with the same severing and the
        current value.
        """
        try:
            universe = self.universe
            current = universe.state.get(lever)
            if current is None:
                return None
            roll = universe.imagine(f"hold {lever}", {lever: float(current)},
                                    steps=self.steps if steps is None else max(1, int(steps)))
            final = roll.final
            if not final or target.variable not in final:
                return None
            return float(final[target.variable])
        except Exception:  # noqa: BLE001
            return None

    def _imagine(self, target: Target, lever: str, setting: float,
                 before: Optional[float], need: float, *,
                 steps: Optional[int] = None) -> Optional[Candidate]:
        """One imagined future, scored.

        ``progress × confidence`` and nothing else. Progress is the share of the remaining
        distance the rollout closes **against the do-nothing rollout for the same lever** — see
        :meth:`_baseline` — **clipped at one** so overshooting is worth no more than
        arriving — a plan that triples the target is not three times as good, and rewarding it
        would select for the most violent intervention available every time. Confidence comes
        from the model, which already discounts a setting the further it sits outside the range
        the relation was fitted on, so between two plans that both arrive she takes the one
        standing on evidence.
        """
        try:
            universe = self.universe
            roll = universe.imagine(f"set {lever}", {lever: setting},
                                    steps=self.steps if steps is None else max(1, int(steps)))
            final = roll.final
            if not final or target.variable not in final:
                return None
            predicted = float(final[target.variable])
            candidate = Candidate(action=roll.action, lever=lever, setting=float(setting),
                                  predicted=predicted, confidence=float(roll.confidence),
                                  rollout=roll)
            here = universe.state.get(lever)
            candidate.effort = (0.0 if here is None
                                else abs(float(setting) - float(here)) / self._span(lever))
            moved = target.gain(before, predicted)
            candidate.progress = max(0.0, min(1.0, moved / need)) if need > _EPSILON else 0.0
            candidate.reached = target.met_by(predicted)
            candidate.score = candidate.progress * candidate.confidence
            return candidate
        except Exception:  # noqa: BLE001
            return None

    # ---- acting --------------------------------------------------------------- #
    def key_for(self, variable: str) -> str:
        """The key a plan's claim is registered under. Resolved by the next reading of it."""
        return f"rollout:{variable}"

    def commit(self, plan: Plan) -> bool:
        """Stand behind the chosen future: register it as a claim reality can contradict.

        A plan nobody ever compares against an outcome is a preference, not a prediction, and
        this package has already paid once for that distinction. The key is the variable's own
        name, so the very next reading of it settles the claim — which is exactly what the raw
        stimulus key could never be.
        """
        try:
            if plan is None or not plan.actionable or plan.target is None:
                return False
            key = self.key_for(plan.target.variable)
            predictor = self.predictor
            if predictor is not None:
                predictor.predict(key, plan.chosen.predicted,
                                  confidence=max(0.0, min(1.0, plan.chosen.confidence)),
                                  organ="rollout", lever=plan.chosen.lever,
                                  setting=plan.chosen.setting)
            plan.committed = True
            self.committed += 1
            self._open = (plan, key)
            return True
        except Exception:  # noqa: BLE001
            return False

    def outstanding(self) -> Optional[Plan]:
        """The plan waiting on a reading, if there is one."""
        return self._open[0] if self._open else None

    def settle(self, actual: Dict[str, Any], *,
               order: Optional[Sequence[str]] = None) -> Optional[Settlement]:
        """Grade the outstanding plan against a real reading. ``None`` when it says nothing.

        Two gradings from one reading, and they answer different questions.
        :meth:`~nyxara.njp.universe.InternalUniverse.grade` charges the error to the *causal
        model version* that made the claim, which is what a retirement decision has to count.
        :meth:`~nyxara.njp.predict.PredictionEngine.observe` produces an ``Outcome`` whose
        ``surprise`` is error weighted by the confidence claimed — the shared signal attention
        and memory already read. An imagined future reality contradicted therefore raises exactly
        the same flag as any other prediction she got wrong, which is the point of routing it
        through the predictor rather than keeping a private error counter here.

        ``grade`` rather than ``reconcile`` deliberately: reconcile observes the reading as well,
        and the caller handing this reading over has already observed it. Observing it twice
        would raise ``n`` on every relation it touches without adding a single bit of variance.
        """
        try:
            if self._open is None:
                return None
            plan, key = self._open
            variable = plan.target.variable if plan.target else ""
            reading = None
            for name, raw in (actual or {}).items():
                if str(name) == variable:
                    try:
                        reading = float(raw)
                    except (TypeError, ValueError):
                        reading = None
                    break
            if reading is None:
                # The reading did not mention the variable the plan is about, so it settles
                # nothing. The claim stays open rather than being quietly marked scored — a claim
                # closed by a reading that could not address it is the counter going up again.
                return None

            self._open = None
            settlement = Settlement(variable=variable, actual=reading,
                                    imagined=plan.chosen.predicted if plan.chosen else None)
            universe = self.universe
            if universe is not None and plan.chosen is not None and plan.chosen.rollout is not None:
                error = universe.grade(plan.chosen.rollout, {variable: reading})
                if error is not None:
                    settlement.error = float(error)
                    settlement.scored = True
            predictor = self.predictor
            if predictor is not None:
                outcome = predictor.observe(key, reading)
                if outcome is not None:
                    surprise = getattr(outcome, "surprise", None)
                    settlement.surprise = None if surprise is None else float(surprise)
                    settlement.scored = True
            settlement.reached = bool(plan.target and plan.target.met_by(reading))
            if settlement.scored:
                self.settled += 1
                if settlement.error is not None:
                    self.total_error += settlement.error
            if settlement.reached:
                self.reached += 1
            self.settlements.append(settlement)
            del self.settlements[:-_HISTORY]
            return settlement
        except Exception:  # noqa: BLE001
            return None

    def pursue(self, target: Target, *, commit: bool = True) -> Plan:
        """Search and stand behind the result. The whole cycle bar the reading that settles it."""
        plan = self.search(target)
        if commit and plan.actionable:
            self.commit(plan)
        return plan

    # ---- reporting -------------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        last = self.history[-1] if self.history else None
        return {
            "planned": self.planned,
            "committed": self.committed,
            # The falsifier, and it is deliberately the harder of the two numbers to move. A
            # planner can raise `planned` by being called; only reality raises `settled`.
            "settled": self.settled,
            "reached": self.reached,
            "no_lever": self.no_lever,
            "outstanding": self.outstanding() is not None,
            "mean_error": (round(self.total_error / self.settled, 5) if self.settled else None),
            "last": last.to_dict() if last is not None else None,
            "last_settlement": (self.settlements[-1].to_dict() if self.settlements else None),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"counters": {"planned": self.planned, "committed": self.committed,
                             "settled": self.settled, "reached": self.reached,
                             "no_lever": self.no_lever, "total_error": self.total_error}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            counters = (d or {}).get("counters") or {}
            self.planned = int(counters.get("planned", 0))
            self.committed = int(counters.get("committed", 0))
            self.settled = int(counters.get("settled", 0))
            self.reached = int(counters.get("reached", 0))
            self.no_lever = int(counters.get("no_lever", 0))
            self.total_error = float(counters.get("total_error", 0.0))
        except Exception:  # noqa: BLE001
            pass
