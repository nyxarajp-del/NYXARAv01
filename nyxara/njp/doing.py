"""NYXARA · njp/doing.py — goal → plan → action → outcome, on actions she can actually take (🎯).

Stage G of the curriculum is *agency*, it scores ``agency.success_rate``, and it has read zero for
the life of the project. :meth:`nyxara.njp.brain.NJPBrain.pursue` says why, and says it correctly::

    "No plan without actions, no actions without a plan. Nothing else in NJP names an action:
     there is no actuator, no action vocabulary, no affordance list — she models what follows
     what, and never what she could *do*. The gap is an action vocabulary, and that is a design
     question rather than a line of plumbing."

That is the question this module answers, and it answers it without inventing a body. She has no
hands and this does not pretend otherwise. What she has is a set of **cognitive actions she really
performs**, each already implemented, each already called somewhere on a fixed cadence, and each
with an effect on her own state that can be measured before and after:

    ask         raise the open question with the highest value        (curiosity)
    attack      attack the belief with the most riding on it          (adversary)
    mine        surface assumptions her arrows rest on                (assume)
    test        test the assumptions that can be tested               (assume)
    consolidate promote what has earned it, forget what has not       (levels)
    discover    propose laws from what recurred, and check them       (discover)
    tune        one bounded knob trial against a held-out benchmark   (field)
    restructure one structural trial: representation, operator, edge  (evolution)

Every one of those is a thing she does. None of them is a thing she *chose* to do, and that is the
whole difference between a schedule and agency: a cadence fires ``consolidate`` every eight turns
whether or not consolidation is what she needs, and nothing anywhere asks *which of these would
move the thing that is actually stuck*.

**The goal is not invented either.** :class:`~nyxara.njp.curriculum.Curriculum` already computes
it: the rung she is on names the metric holding her back and the number it has to reach —
``"only 10 of 12 required concepts.observations"``. That is a goal with a target, produced by
measurement, and until now it was a sentence in a report. Here it is the thing she plans toward.

**What is learned is which action moves which metric.** Nothing is tabled. Each attempt reads the
metric, runs one action, reads it again, and credits that ``(metric, action)`` pair with what
actually happened. Untried actions are tried first — once each, so a pairing cannot be dismissed
for never having had a turn — and after that she goes with the record. An action that has never
moved a metric stops being proposed for it, which is the only sense in which anything here
"knows" anything.

**Nothing here takes a turn, and that was a decision rather than an oversight.** A ``reflect``
affordance — put her own top question through her own ``think`` — was built, and it is the only
action that would *make experience* rather than rearrange it, which is what most of the metrics
she is stuck behind actually need. It was removed. Measured against every numeric key in her own
stats it moved **nothing**: the questions curiosity raises are introspective, so the turn grounds
no facts, scores no prediction and observes no concept. And it cost two real things — one user
turn became two full pipeline passes, so ``brain.turns`` no longer equalled the turns the world
gave her (caught by the concurrency test, 161 where 160 were sent), and an affordance that
re-enters the turn loop needs a re-entrancy guard that is shared mutable state across threads.

An action with no measured improvement and a real cost is the exact trade the plan's own golden
rule refuses. The guard in :meth:`CognitiveAgency.act` is kept anyway, because the constraint it
encodes outlives the action that needed it: an affordance that takes a turn may not start a
pursuit from inside one.

**A goal about the world is planned rather than acted.** Every affordance above rearranges what
she already has, and not one of them touches a number out in the world — so a goal like
``plant.growth ≥ 12`` runs the whole table, moves nothing eight times, and lands in
:meth:`CognitiveAgency._exhausted` as *"nothing she can do has ever moved this"*. That sentence is
true of her cognitive actions and false of her: the causal model can name the intervention that
would do it. When a goal's metric is a variable :class:`~nyxara.njp.universe.InternalUniverse`
actually knows, :mod:`nyxara.njp.rollout` searches imagined futures for it instead. The guard is
the check that the universe knows the variable — the two namespaces are both ``organ.key`` and
overlap without being the same, and planning over a variable that does not exist would count
every empty rollout as work.

**Reaching the goal is the metric reaching its target, and nothing else.** Not "she acted", not
"the action returned true". :attr:`CognitiveAgency.goals_reached` counts targets met, so an agency
that flails busily reads exactly as badly as one that does nothing — which is the point of
measuring it at all.

Pure standard library. Fail-soft throughout: an action that raises counts as an action that
changed nothing, because from the outside that is what it is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Goal", "Affordance", "Attempt", "ActionValue", "CognitiveAgency", "AFFORDANCES"]

#: Attempts kept for the record. Bounded like everything else here.
_HISTORY = 256

#: A move smaller than this is not a move. Metrics are counts and rates; floating noise in a rate
#: should not be credited to an action as though it were an effect.
_EPSILON = 1e-9


@dataclass(frozen=True)
class Goal:
    """A number that has to reach a target, and why it matters.

    ``at_most`` goals are not decoration. Half of what she can actually achieve on her own is a
    *reduction* — untested assumptions, unexamined beliefs — and a framework that could only ask
    for numbers to go up would have to pretend those were something else.
    """

    metric: str = ""              # "organ.key", read out of `brain.stats()`
    target: float = 0.0
    at_most: bool = False
    why: str = ""

    @property
    def named(self) -> bool:
        return bool(self.metric) and "." in self.metric

    def met_by(self, value: Optional[float]) -> bool:
        if value is None:
            return False
        return value <= self.target if self.at_most else value >= self.target

    def gain(self, delta: float) -> float:
        """Movement in the direction that helps, so one record works for both kinds of goal."""
        return -delta if self.at_most else delta

    def to_dict(self) -> Dict[str, Any]:
        return {"metric": self.metric, "target": self.target,
                "direction": "at_most" if self.at_most else "at_least", "why": self.why[:160]}


@dataclass(frozen=True)
class Affordance:
    """One thing she can do, and the call that does it."""

    name: str = ""
    why: str = ""
    run: Optional[Callable[[Any], bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "why": self.why[:120]}


@dataclass
class ActionValue:
    """What one action has actually done to one metric."""

    metric: str = ""
    action: str = ""
    tries: int = 0
    moved: int = 0
    total_delta: float = 0.0

    @property
    def rate(self) -> float:
        return (self.moved / self.tries) if self.tries else 0.0

    @property
    def mean_delta(self) -> float:
        return (self.total_delta / self.tries) if self.tries else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"metric": self.metric, "action": self.action, "tries": self.tries,
                "moved": self.moved, "rate": round(self.rate, 4),
                "mean_delta": round(self.mean_delta, 4)}


@dataclass
class Attempt:
    """One goal → plan → action → outcome, with the numbers on both sides of the action."""

    goal: Optional[Goal] = None
    action: str = ""
    before: Optional[float] = None
    after: Optional[float] = None
    ran: bool = False
    reached: bool = False
    why: str = ""
    ms: float = 0.0

    @property
    def delta(self) -> float:
        if self.before is None or self.after is None:
            return 0.0
        return self.after - self.before

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal.to_dict() if self.goal else None, "action": self.action,
                "before": self.before, "after": self.after, "delta": round(self.delta, 4),
                "ran": self.ran, "reached": self.reached, "why": self.why[:200],
                "ms": round(self.ms, 2)}


# --------------------------------------------------------------------------- #
# What she can do
# --------------------------------------------------------------------------- #
def _ask(brain: Any) -> bool:
    curiosity = getattr(brain, "curiosity", None)
    return curiosity is not None and curiosity.ask() is not None


def _attack(brain: Any) -> bool:
    adversary = getattr(brain, "adversary", None)
    return bool(adversary is not None and adversary.attack_strongest(limit=1))


def _mine(brain: Any) -> bool:
    miner, world = getattr(brain, "assumptions", None), getattr(brain, "world", None)
    return bool(miner is not None and world is not None and miner.mine(world))


def _test(brain: Any) -> bool:
    miner, world = getattr(brain, "assumptions", None), getattr(brain, "world", None)
    return bool(miner is not None and world is not None and miner.test(world))


def _consolidate(brain: Any) -> bool:
    levels = getattr(brain, "levels", None)
    return bool(levels is not None and getattr(levels.consolidate(), "changed", False))


def _discover(brain: Any) -> bool:
    discoverer = getattr(brain, "discoverer", None)
    if discoverer is None:
        return False
    got = discoverer.discover()
    return bool(int(getattr(got, "proposed", 0) or 0))


def _tune(brain: Any) -> bool:
    field = getattr(brain, "field", None)
    if field is None:
        return False
    return bool(getattr(field.meta_cycle(), "accepted", False))


def _restructure(brain: Any) -> bool:
    evolution = getattr(brain, "evolution", None)
    if evolution is None:
        return False
    return bool(getattr(evolution.cycle(brain), "promoted", False))


#: Every action here is already implemented and already called somewhere on a cadence. What is
#: new is that she may now *choose* between them.
AFFORDANCES: Tuple[Affordance, ...] = (
    Affordance("ask", "raise the open question with the highest value", _ask),
    Affordance("attack", "attack the belief with the most riding on it", _attack),
    Affordance("mine", "surface what her arrows assume and nothing has examined", _mine),
    Affordance("test", "test the assumptions that can be tested", _test),
    Affordance("consolidate", "promote what has earned it, forget what has not", _consolidate),
    Affordance("discover", "propose laws from what recurred, and check them", _discover),
    Affordance("tune", "one bounded knob trial against a held-out benchmark", _tune),
    Affordance("restructure", "one structural trial: representation, operator, edge", _restructure),
)


# --------------------------------------------------------------------------- #
# The organ
# --------------------------------------------------------------------------- #
class CognitiveAgency:
    """Picks the action most likely to move the number that is holding her back, and runs it."""

    def __init__(self, affordances: Tuple[Affordance, ...] = AFFORDANCES) -> None:
        self.affordances: Dict[str, Affordance] = {a.name: a for a in affordances}
        self.values: Dict[Tuple[str, str], ActionValue] = {}
        self.history: List[Attempt] = []
        self.planned = 0
        self.acted = 0
        self.goals_reached = 0
        self.no_goal = 0
        #: Goals that named a variable in the causal model and were pursued by rolling futures
        #: forward instead of by running an affordance. Counted apart from `planned` because they
        #: are a different kind of act: the affordance table is what she can do to *herself*.
        self.planned_world = 0
        self.committed_world = 0
        #: Guards the one affordance that re-enters the turn loop. See `_reflect`.
        self._acting = False

    # ---- reading the world ------------------------------------------------- #
    @staticmethod
    def read(brain: Any, metric: str) -> Optional[float]:
        """One number out of her own stats. ``None`` when the organ is off or silent."""
        try:
            organ, _, key = str(metric).partition(".")
            block = (brain.stats() or {}).get(organ)
            if not isinstance(block, dict):
                return None
            value = block.get(key)
            if value is None or isinstance(value, bool):
                return None
            return float(value)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def goal_from(report: Any) -> Optional[Goal]:
        """The rung she is on, as a target.

        :class:`~nyxara.njp.curriculum.Curriculum` has computed this all along — the metric that is
        blocking, the number it needs, and the sentence saying so. Nothing read it as a *goal*.
        """
        try:
            result = getattr(report, "current", None)
            stage = getattr(result, "stage", None) if result is not None else None
            if stage is None:
                return None
            # Evidence first: a rung that has not been exercised enough cannot be judged on its
            # metric, and driving the metric before the samples exist is optimising a number
            # computed from nothing.
            if not getattr(result, "enough_evidence", False):
                return Goal(metric=f"{stage.sample_organ}.{stage.sample_metric}",
                            target=float(stage.min_samples),
                            why=str(getattr(result, "why", "") or ""))
            return Goal(metric=f"{stage.organ}.{stage.metric}",
                        target=float(stage.threshold),
                        why=str(getattr(result, "why", "") or ""))
        except Exception:  # noqa: BLE001
            return None

    def _exhausted(self, goal: Goal) -> bool:
        """Has every action been tried on this metric and none of them ever moved it?"""
        rows = [self.values.get((goal.metric, name)) for name in self.affordances]
        if any(row is None for row in rows):
            return False
        return all(row.mean_delta <= _EPSILON and row.rate <= _EPSILON for row in rows)

    @staticmethod
    def deficit_goal(brain: Any) -> Optional[Goal]:
        """The largest thing wrong with her own record that she can actually put right.

        Untested assumptions, and that is the whole list because it is the whole list that
        measurement supports. Every affordance was run against every numeric key in her own stats
        and the movable ones are narrow::

            reflect      attention / blackbox / compiler counters — it takes a turn
            attack       adversary.attacked, adversary.refuted
            test         assumptions.unknown_unknowns  ↓   ← closable, and a real deficit
            consolidate  levels.consolidations
            discover     discover.passes
            tune         concepts.concepts, concepts.compression, field.benchmark
            restructure  evolution counters
            ask / mine   nothing, from this state

        Most of those are an action's own counter, which is not an achievement. Untested
        assumptions is the one entry that is a genuine deficit, that she genuinely closes, and
        that §11 says is the kind that kills a model. Adding the others would inflate
        `goals_reached` with actions congratulating themselves.
        """
        try:
            stats = brain.stats() or {}
            block = stats.get("assumptions")
            if isinstance(block, dict):
                untested = block.get("unknown_unknowns")
                if untested is not None and float(untested) > 0:
                    return Goal(metric="assumptions.unknown_unknowns", target=0.0, at_most=True,
                                why=f"{float(untested):g} assumptions nothing has examined")
        except Exception:  # noqa: BLE001
            return None
        return None

    # ---- planning ----------------------------------------------------------- #
    def plan(self, goal: Goal) -> Optional[str]:
        """The action to take for this metric: an untried one first, then the best record.

        Untried before best, and least-tried among the untried, for the reason
        :class:`~nyxara.njp.selfmodel.MetaLearner` had to learn the hard way: taking the first
        option in insertion order lets whichever action was declared first monopolise the metric,
        and the later ones are never reached at all.
        """
        self.planned += 1
        names = list(self.affordances)
        if not names:
            return None
        untried = [n for n in names if (goal.metric, n) not in self.values]
        if untried:
            return untried[0]
        ranked = sorted(
            names,
            key=lambda n: (-self.values[(goal.metric, n)].mean_delta,
                           -self.values[(goal.metric, n)].rate,
                           self.values[(goal.metric, n)].tries))
        best = self.values[(goal.metric, ranked[0])]
        if best.mean_delta <= _EPSILON and best.rate <= _EPSILON:
            # Nothing she can do has ever moved this. Saying so is more useful than picking one
            # anyway, and it keeps `acted` from counting turns where she was going through motions.
            return None
        return ranked[0]

    # ---- acting ------------------------------------------------------------- #
    # ---- planning over the world, rather than over herself --------------------- #
    @staticmethod
    def _world_target(brain: Any, goal: Goal) -> Any:
        """This goal as a target in the causal model, if the causal model knows the variable.

        The bridge :mod:`nyxara.njp.rollout` exists to be reached through, and the guard is the
        whole of it. Every affordance above rearranges what she already has; not one of them can
        move a number about the *world*, because none of them touches the world. A goal like
        ``plant.growth ≥ 12`` therefore runs the full affordance table, moves nothing, and ends
        up in :meth:`_exhausted` as "nothing she can do has ever moved this" — a true statement
        about her cognitive actions and a false one about her, because the causal model can name
        the intervention that would do it.

        ``Goal.metric`` is ``"organ.key"`` out of ``brain.stats()`` and a universe variable is
        very often dotted the same way, so the two namespaces overlap without being the same.
        Accepting a metric the universe has never heard of would plan over a variable that does
        not exist and count every empty rollout as work.
        """
        try:
            planner = getattr(brain, "rollout", None)
            universe = getattr(brain, "universe", None)
            if planner is None or universe is None:
                return None
            from nyxara.njp.rollout import Target
            return Target.from_goal(goal, universe)
        except Exception:  # noqa: BLE001
            return None

    def _plan_world(self, brain: Any, goal: Goal, attempt: Attempt) -> bool:
        """Search imagined futures for this goal and stand behind the best. ``True`` if planned.

        Deliberately *not* credited into :attr:`values`. That table records what an action did to
        a metric between two readings a moment apart, and a plan is not that shape: the reading
        that judges it arrives whenever the world is next measured, which may be many turns later
        and is scored by :meth:`~nyxara.njp.rollout.RolloutPlanner.settle` when it does. Crediting
        the plan here against an unchanged number would book every plan as an action that did
        nothing, and the record would then teach her never to plan again.
        """
        try:
            planner = getattr(brain, "rollout", None)
            target = self._world_target(brain, goal)
            if planner is None or target is None:
                return False
            plan = planner.pursue(target)
            attempt.action = "plan"
            attempt.why = plan.why
            self.planned_world += 1
            if plan.committed:
                self.committed_world += 1
            return bool(plan.actionable)
        except Exception:  # noqa: BLE001
            return False

    def act(self, brain: Any, goal: Optional[Goal] = None,
            report: Any = None) -> Attempt:
        """One whole cycle: read the number, choose, do it, read the number again, credit."""
        attempt = Attempt()
        t0 = time.perf_counter()
        if self._acting:
            # `reflect` takes a turn, and a turn schedules an action. Declining the inner one is
            # the whole guard: she may think about her question, she may not start a new pursuit
            # from inside it.
            attempt.why = "already acting — an action may not start another"
            return attempt
        try:
            if goal is None:
                goal = self.goal_from(report)
                # A rung nothing she can do has ever moved is not a plan, it is a wait for the
                # Master. Rather than flail against it every cycle, fall through to a deficit she
                # *can* close. Measured: every curriculum rung here is gated on a sample count,
                # and every affordance rearranges what she has rather than making experience —
                # so without this she is honestly, permanently, stuck at zero.
                if goal is not None and goal.named and self._exhausted(goal):
                    fallback = self.deficit_goal(brain)
                    if fallback is not None:
                        goal = fallback
            attempt.goal = goal
            if goal is None or not goal.named:
                self.no_goal += 1
                attempt.why = "nothing is currently blocking her, so there is nothing to pursue"
                return attempt

            before = self.read(brain, goal.metric)
            attempt.before = before
            if goal.met_by(before):
                attempt.reached = True
                attempt.after = before
                attempt.why = (f"{goal.metric} already at {before:g}, wants "
                               f"{'at most' if goal.at_most else 'at least'} {goal.target:g}")
                return attempt

            # A goal about the world is planned, not acted. The affordance table is what she can
            # do to herself; nothing in it touches a variable out there, so running it against
            # such a goal produces eight measured no-ops and the conclusion that the goal is
            # impossible. The causal model can name the intervention instead.
            if self._world_target(brain, goal) is not None:
                attempt.ran = self._plan_world(brain, goal, attempt)
                attempt.after = self.read(brain, goal.metric)
                return attempt

            action = self.plan(goal)
            attempt.action = str(action or "")
            if action is None:
                attempt.why = (f"nothing she can do has ever moved {goal.metric} — "
                               f"that is a finding, not a plan")
                return attempt

            affordance = self.affordances[action]
            self._acting = True
            try:
                did = bool(affordance.run(brain)) if affordance.run else False
            except Exception:  # noqa: BLE001 — an action that raises changed nothing
                did = False
            finally:
                self._acting = False
            attempt.ran = True
            self.acted += 1

            after = self.read(brain, goal.metric)
            attempt.after = after
            self._credit(goal.metric, action, goal.gain(attempt.delta))
            if goal.met_by(after):
                attempt.reached = True
                self.goals_reached += 1
            attempt.why = (f"{action}: {goal.metric} "
                           f"{'—' if before is None else format(before, 'g')} → "
                           f"{'—' if after is None else format(after, 'g')} "
                           f"(wants {goal.target:g}{'' if did else ', and the action did nothing'})")
            return attempt
        except Exception:  # noqa: BLE001
            attempt.why = "the attempt failed"
            return attempt
        finally:
            attempt.ms = (time.perf_counter() - t0) * 1000.0
            self.history.append(attempt)
            del self.history[:-_HISTORY]

    def _credit(self, metric: str, action: str, delta: float) -> None:
        value = self.values.get((metric, action))
        if value is None:
            value = ActionValue(metric=metric, action=action)
            self.values[(metric, action)] = value
        value.tries += 1
        value.total_delta += float(delta)
        if delta > _EPSILON:
            value.moved += 1

    # ---- reporting ---------------------------------------------------------- #
    def best_for(self, metric: str) -> Optional[ActionValue]:
        rows = [v for (m, _a), v in self.values.items() if m == metric and v.tries]
        return max(rows, key=lambda v: v.mean_delta) if rows else None

    def stats(self) -> Dict[str, Any]:
        last = self.history[-1] if self.history else None
        ranked = sorted((v for v in self.values.values() if v.tries),
                        key=lambda v: -v.mean_delta)
        return {
            "affordances": len(self.affordances),
            "planned": self.planned,
            "acted": self.acted,
            "goals_reached": self.goals_reached,
            "success_rate": (round(self.goals_reached / self.acted, 4) if self.acted else None),
            "no_goal": self.no_goal,
            "planned_world": self.planned_world,
            "committed_world": self.committed_world,
            "learned_pairs": len(self.values),
            "best": ranked[0].to_dict() if ranked else None,
            "last": last.to_dict() if last is not None else None,
        }
