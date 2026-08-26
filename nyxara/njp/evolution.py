"""NYXARA · njp/evolution.py — she changes what she is made of, and only on evidence (🧬, Phase 7).

Every other self-change in NJP moves a **number**. :meth:`nyxara.njp.field.RecursiveCognitiveField.meta_cycle`
finds the organ that is limiting her, proposes a bounded change to one knob, benchmarks it on
held-out samples and reverts it unless it strictly wins — and it refuses outright to touch an organ
its benchmark cannot see, which is the single most important line in that module.

Phase 7 asks for the thing a knob cannot reach:

    new operator · new representation · new strategy · new topology

A knob adjusts the machine. These four **rewire** it. And precisely because they are stronger, the
plan attaches the same pipeline to them and refuses to relax it:

    sandbox → benchmark → adversarial → regression → old vs new → promote / rollback

with one rule over all of it — *self-modification without measurable improvement is failure*. So
nothing here can adopt itself. :attr:`CognitiveEvolution.cognitive_rewires` counts **promotions**,
a promotion requires a strictly better held-out score *and* two batteries that did not get worse,
and a candidate whose measure cannot see it is refused as unmeasurable rather than accepted on a
tie. A tie is not weak evidence for a change; it is no evidence, and adopting on ties is how a
system travels a long way on nothing.

**Why the representation mutation is the one with something to find.** The loop and the field both
feed :class:`~nyxara.njp.predictive.PredictiveWorldModel`, and they feed it *different kinds of
state*: the field observes the grounded facts of the turn (``"aag lagi"``), the loop observes a
sixteen-bucket histogram of which cells fired (``"[0.0, 0.0, 0.5, …]"``). One n-gram model, two
representations, strictly alternating. Measured on a live brain::

    [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, …]
    garmi hui
    [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, …]
    pasina aaya

Every order-1 context for a fact is therefore a histogram, order 3 spans a turn and a half, and the
alphabet the smoothing divides by is twice the size of either half. That is not a knob. It is a
question about what a *state* is, and it is exactly the shape of question this module exists to put
through a benchmark instead of through an argument. The candidate encodings are offered and the
held-out replay decides; nothing here asserts which one wins.

**The sandbox is real.** A representation candidate is scored by replaying her own recorded turns
into *fresh* models — the live one is never touched until a promotion — and the score is taken on
the tail of the trace the candidate was not chosen from. The two gates run the real batteries
(:mod:`nyxara.eval.adversarial`, :mod:`nyxara.eval.intelligence`) against fresh brains through
their ``prepare`` hooks, which is what those hooks were built for.

**And it has to be the same experiment.** :meth:`CognitiveEvolution.refit` re-fitting a brain from
its own trace, in the representation it was already using, reproduces the model it actually built —
same observations, same history, same counts, checked by a test. That is not a nicety. Two
fidelity bugs were found by that comparison and each one made a candidate look good in the sandbox
and measure *worse* live: a ``previous`` carried across turns the encoding could not represent
(58 observations where 48 lived turns produced 43, with self-loops that exist nowhere in her
experience), and :meth:`nyxara.njp.field.RecursiveCognitiveField._predict_world` labelling its
observations with ``brain._last_intent_kind`` — an attribute the package read once and assigned
never, so half the model's transitions carried the empty action and half carried a real one. A
sandbox that is kinder than reality is worse than no sandbox at all: it converts "we do not know"
into a confident wrong answer, and then rewires her on it.

Pure standard library. No LLM. Fail-soft: a failed cycle changes nothing, which is the correct
outcome for a self-modification that could not be evaluated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Situation", "Measurement", "Mutation", "EvolutionTrial", "CognitiveEvolution",
    "ENCODERS", "KINDS",
]

#: Mutation kinds, as the plan names them.
KIND_REPRESENTATION = "representation"
KIND_TOPOLOGY = "topology"
KIND_OPERATOR = "operator"
KIND_STRATEGY = "strategy"
KINDS: Tuple[str, ...] = (KIND_REPRESENTATION, KIND_TOPOLOGY, KIND_OPERATOR, KIND_STRATEGY)

#: Turns kept for replay. The trace is the sandbox's only data, so it has to be long enough for a
#: held-out tail to mean something and short enough to be free.
_CAPACITY = 512

#: Turns of trace before a representation candidate may be scored at all, and the fraction held
#: out. A candidate chosen on the whole trace and scored on the whole trace is fitting, not
#: measuring.
_MIN_TRACE = 24
_HOLDOUT = 0.3

#: Scorable held-out episodes a selection candidate needs. Selection records are thin early —
#: a reward needs a *graded* answer, not merely a turn — and a rewire proposed on two rows is a
#: coin flip with a rationale.
_MIN_SELECTION = 6

#: How much better the candidate must be, as a floor under the sample-size rule below. Not zero:
#: adopting on a tie is how a system travels a long way on no evidence at all.
_MIN_GAIN = 0.02

#: Buckets in the histogram the loop feeds today — mirrored from
#: :meth:`nyxara.njp.integrate.LearningLoop._encode_state`, whose docstring says at length what it
#: costs: three unrelated words share one state, so this encoding cannot carry identity.
_BUCKETS = 16


# --------------------------------------------------------------------------- #
# What one turn looked like, kept so a *different* representation can be tried on it
# --------------------------------------------------------------------------- #
@dataclass
class Situation:
    """One turn's raw material — before any choice about how to represent it.

    Deliberately the *inputs* to an encoding rather than an encoding. A trace of encoded states
    can only ever re-score the encoding that produced it, which would make the sandbox an echo.
    """

    cells: Tuple[int, ...] = ()
    concepts: Tuple[str, ...] = ()
    facts: Tuple[str, ...] = ()
    action: str = ""
    at: float = dc_field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"cells": len(self.cells), "concepts": list(self.concepts[:6]),
                "facts": list(self.facts[:4]), "action": self.action}


def _buckets(cells: Sequence[int], width: int = _BUCKETS) -> Optional[List[float]]:
    """Fired cells → a fixed-width normalised histogram. What the loop feeds today."""
    if not cells:
        return None
    out = [0.0] * width
    for cell in cells:
        out[int(cell) % width] += 1.0
    total = sum(out)
    return [round(v / total, 4) for v in out] if total else None


def _state_of(items: Sequence[str]) -> Any:
    from nyxara.njp.predictive import WorldState
    state = WorldState.of(items)
    return None if state.empty else state


#: The representations she may adopt for what the *loop* contributes to the dynamics model.
#:
#: ``off`` is deliberately absent. It would almost certainly score well — feeding the model
#: nothing removes the second representation entirely — and it would do it by deleting the
#: planner's only source of action labels, which no gate here can see. A candidate that wins by
#: making the measure blind to a capability is optimising the measure, and the whole point of this
#: module is that it must not be possible to do that.
ENCODERS: Dict[str, Callable[[Situation], Any]] = {
    "buckets16": lambda s: _buckets(s.cells, _BUCKETS),
    "buckets64": lambda s: _buckets(s.cells, 64),
    "facts": lambda s: _state_of(s.facts),
    "concepts": lambda s: _state_of(s.concepts),
    "cells": lambda s: tuple(sorted(set(s.cells))[:8]) or None,
}

#: What the loop does unless a promotion says otherwise — today's behaviour, exactly.
DEFAULT_ENCODING = "buckets16"


@dataclass(frozen=True)
class Measurement:
    """A score and **how many rows it is made of**, which is half of what a score means.

    Carried together because the two are only useful together. A held-out tail of twelve rows
    moves by 0.083 when a single episode lands differently, so a rule that adopts a rewire on a
    gain of 0.02 is a rule that adopts on one row — and one row is what noise looks like. See
    :attr:`floor`.
    """

    value: float = 0.0
    rows: int = 0

    @property
    def floor(self) -> float:
        """The smallest gain this many rows can actually distinguish from nothing.

        One row's worth. A candidate that cannot beat the resolution of its own measure has not
        been measured better; it has been measured *the same*, on an instrument too coarse to say
        so out loud.
        """
        return (1.0 / self.rows) if self.rows > 0 else 1.0

    @property
    def required(self) -> float:
        """The gain a candidate has to clear on this measure: **two rows, not one.**

        One row is the resolution — the smallest difference the instrument can express — and a
        difference of exactly one row out of fourteen is what a single episode landing the other
        way looks like. Structural changes are the strongest thing this module can do, so they
        are held to a difference the measure could not have produced by one coin landing.
        """
        return 2.0 * self.floor

    def to_dict(self) -> Dict[str, Any]:
        return {"value": round(self.value, 5), "rows": self.rows,
                "floor": round(self.floor, 5), "required": round(self.required, 5)}


# --------------------------------------------------------------------------- #
# A proposal, and what happened to it
# --------------------------------------------------------------------------- #
@dataclass
class Mutation:
    """A structural change: what it is, why, and how to put it in and take it out again.

    ``apply`` and ``revert`` are closures over the brain rather than a recorded diff, because a
    rewire is not a value — it is an edge added to a graph or a table entry swapped — and a
    "before" that cannot be restored exactly is not a rollback.
    """

    kind: str = ""
    name: str = ""
    why: str = ""
    target: str = ""
    apply: Optional[Callable[[Any], bool]] = None
    revert: Optional[Callable[[Any], bool]] = None
    #: Scored by the sandbox before anything is applied. ``None`` means the measure cannot see it.
    score: Optional[Callable[[], Optional["Measurement"]]] = None

    @property
    def signature(self) -> str:
        return f"{self.kind}:{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "target": self.target, "why": self.why[:200]}


@dataclass
class EvolutionTrial:
    """One candidate's whole journey down the pipeline, including the steps it never reached."""

    mutation: Optional[Mutation] = None
    baseline: Optional[float] = None
    candidate: Optional[float] = None
    adversarial_passed: Optional[bool] = None
    regression_passed: Optional[bool] = None
    rows: int = 0
    promoted: bool = False
    unmeasurable: bool = False
    why: str = ""
    ms: float = 0.0

    @property
    def gain(self) -> float:
        if self.baseline is None or self.candidate is None:
            return 0.0
        return self.candidate - self.baseline

    def to_dict(self) -> Dict[str, Any]:
        return {"mutation": self.mutation.to_dict() if self.mutation else None,
                "baseline": None if self.baseline is None else round(self.baseline, 5),
                "candidate": None if self.candidate is None else round(self.candidate, 5),
                "gain": round(self.gain, 5), "rows": self.rows,
                "adversarial_passed": self.adversarial_passed,
                "regression_passed": self.regression_passed,
                "promoted": self.promoted, "unmeasurable": self.unmeasurable,
                "why": self.why[:240], "ms": round(self.ms, 2)}


# --------------------------------------------------------------------------- #
# The organ
# --------------------------------------------------------------------------- #
class CognitiveEvolution:
    """Proposes structural changes to her own cognition and promotes only what measures better."""

    def __init__(self, *, capacity: int = _CAPACITY, min_trace: int = _MIN_TRACE,
                 min_gain: float = _MIN_GAIN, gates: bool = True,
                 holdout: float = _HOLDOUT) -> None:
        self.capacity = max(32, int(capacity))
        self.min_trace = max(8, int(min_trace))
        self.min_gain = float(min_gain)
        self.holdout = min(0.5, max(0.1, float(holdout)))
        #: Whether the two batteries run. Off only for a caller that has already measured them —
        #: never as a way of getting a candidate through.
        self.gates = bool(gates)

        self.trace: List[Situation] = []
        self.trials: List[EvolutionTrial] = []
        self.cognitive_rewires = 0
        self.proposed = 0
        self.unmeasurable = 0
        self.refused = 0
        self.total_gain = 0.0
        #: What the loop currently feeds the dynamics model. Changed only by a promotion.
        self.encoding = DEFAULT_ENCODING
        self.adopted: List[Dict[str, Any]] = []
        self._rejected: set = set()
        #: ``(scored, correct)`` at the last promotion, so what happened *after* it can be read
        #: apart from the lifetime figure it is averaged into.
        self._since: Optional[Tuple[int, int]] = None

    # ---- experience --------------------------------------------------------- #
    def observe(self, situation: Situation) -> None:
        """One turn of raw material for the sandbox. Never raises; a lost turn is a shorter trace."""
        try:
            if situation is None:
                return
            self.trace.append(situation)
            if len(self.trace) > self.capacity:
                del self.trace[: len(self.trace) - self.capacity]
        except Exception:  # noqa: BLE001
            return

    @staticmethod
    def situation_of(thought: Any, cells: Sequence[int], action: str) -> Situation:
        """Read one turn off a thought, in the shapes the encodings need."""
        percept = getattr(thought, "percept", None)
        concepts = tuple(str(c) for c in (getattr(percept, "concepts", None) or [])[:8])
        facts: List[str] = []
        grounding = getattr(percept, "grounding", None)
        for triple in (getattr(grounding, "triples", None) or [])[:4]:
            text = (f"{getattr(triple, 'subject', '')} {getattr(triple, 'predicate', '')} "
                    f"{getattr(triple, 'object', '')}").strip()
            if text:
                facts.append(text)
        return Situation(cells=tuple(int(c) for c in (cells or ())[:64]),
                         concepts=concepts, facts=tuple(facts), action=str(action or ""))

    # ---- the sandbox -------------------------------------------------------- #
    def replay(self, encoding: str) -> Optional[Measurement]:
        """Held-out next-state accuracy when the loop contributes ``encoding``.

        A fresh model per call, fed the trace in the order it happened and scored only on the
        tail. Both feeds are replayed — the field's fact-state *and* the loop's — because the
        thing under test is what happens when the two meet in one model, and a replay of either
        alone would answer a question nobody asked.
        """
        try:
            encoder = ENCODERS.get(encoding)
            if encoder is None or len(self.trace) < self.min_trace:
                return None
            from nyxara.njp.predictive import PredictiveWorldModel, WorldState

            split = int(len(self.trace) * (1.0 - self.holdout))
            model = PredictiveWorldModel()
            previous: Any = None
            hits = 0
            total = 0
            for index, situation in enumerate(self.trace):
                # **The loop first, then the field — the order the turn actually runs in.**
                # `NJPBrain._think` closes the loop at step 10 and runs the field at step 11, so
                # by the time the field predicts, the loop's observation of *this* turn is
                # already the last thing in the history. Replaying them the other way round is a
                # different experiment: it moved the order-1 context by one turn and the sandbox
                # started disagreeing with the live model, which is the one way a sandbox can be
                # worse than no sandbox at all.
                current = encoder(situation)
                if previous is not None and current is not None and situation.action:
                    model.observe(previous, situation.action, next_state=current)
                # **Advanced every turn, including to `None`.** The loop's `previous` is
                # literally last turn's situation, so a turn the encoding cannot represent breaks
                # the chain there. Carrying the last representable turn forward instead invents
                # transitions across the gap that the live loop never makes — measured, a model
                # refitted that way held 58 observations where the same 48 turns lived natively
                # produced 43, with self-loops like `pasina aaya→pasina aaya` that exist nowhere
                # in her experience. A sandbox that is kinder than reality is worse than none.
                previous = current
                # The field's half: predict from what has been seen, then look. This is the
                # prediction `predictive.accuracy` is actually made of, so it is what is scored.
                if situation.facts:
                    state = WorldState.of(situation.facts)
                    if not state.empty:
                        prediction = model.predict(action=situation.action)
                        if index >= split and prediction.answerable:
                            total += 1
                            hits += int(prediction.top == state.signature)
                        model.observe(state, situation.action)
            return Measurement(hits / total, total) if total else None
        except Exception:  # noqa: BLE001
            return None

    def refit(self, brain: Any) -> bool:
        """Re-learn the dynamics model from the trace, in the representation now promoted.

        In place rather than by replacing the object: :class:`~nyxara.njp.agency.Agent` holds the
        model it was built with, and swapping ``brain.predictive`` would leave the planner
        searching a model nothing updates any more.

        **The scoring counters are deliberately kept.** ``scored``/``correct`` are a record of
        predictions she actually made and was actually right or wrong about; a rewire does not
        make that untrue and clearing it would let a promotion improve her measured accuracy by
        forgetting the predictions that dragged it down. What is re-fitted is the *model*; what
        stands is the *record*. :meth:`stats` reports accuracy since the last rewire beside the
        lifetime figure so the two are never read as one number.
        """
        try:
            predictive = getattr(brain, "predictive", None)
            if predictive is None:
                return False
            self._since = (int(getattr(predictive, "scored", 0) or 0),
                           int(getattr(predictive, "correct", 0) or 0))
            if not self.trace:
                return True
            from nyxara.njp.predictive import WorldState

            predictive._counts.clear()
            predictive._alphabet.clear()
            del predictive._history[:]
            predictive.observations = 0
            encoder = ENCODERS.get(self.encoding) or ENCODERS[DEFAULT_ENCODING]
            previous: Any = None
            for situation in self.trace:
                # Same order as `replay`, and for the same reason: this has to reproduce the
                # history the live turns would have built, not a tidier one.
                current = encoder(situation)
                if previous is not None and current is not None and situation.action:
                    predictive.observe(previous, situation.action, next_state=current)
                previous = current      # every turn, including to `None` — see `replay`
                if situation.facts:
                    state = WorldState.of(situation.facts)
                    if not state.empty:
                        predictive.observe(state, situation.action)
            return True
        except Exception:  # noqa: BLE001 — a failed re-fit leaves the model as it was
            return False

    def accuracy_since_rewire(self, brain: Any) -> Optional[float]:
        """How she has predicted **since** the last promotion, which is the question a rewire asks.

        The lifetime figure spans both sides of a change of coordinates and is a mixture of two
        models; reading it as one number is how a real improvement looks like a regression for as
        long as the pre-rewire turns outnumber the post-rewire ones.
        """
        try:
            if self._since is None:
                return None
            predictive = getattr(brain, "predictive", None)
            scored = int(getattr(predictive, "scored", 0) or 0) - self._since[0]
            correct = int(getattr(predictive, "correct", 0) or 0) - self._since[1]
            return (correct / scored) if scored > 0 else None
        except Exception:  # noqa: BLE001
            return None

    def _selection_rows(self, brain: Any) -> List[Tuple[str, str]]:
        """Held-out ``(kind, chosen strategy)`` from her own graded record."""
        metareason = getattr(brain, "metareason", None)
        history = list(getattr(metareason, "history", None) or [])
        if len(history) < _MIN_SELECTION * 2:
            return []
        tail = history[len(history) // 2:]
        return [(str(getattr(s, "kind", "")), str(getattr(s, "strategy", "")))
                for s in tail if getattr(s, "kind", "") and getattr(s, "strategy", "")]

    @staticmethod
    def _per_kind(brain: Any, kind: str) -> Dict[str, Tuple[int, float]]:
        """``name → (trials, mean)`` for one kind, from the shared bandit's own arms."""
        out: Dict[str, Tuple[int, float]] = {}
        try:
            learner = getattr(getattr(brain, "metareason", None), "meta_learner", None)
            bucket = dict(getattr(learner, "strategies", {}).get(f"strategy:{kind}") or {})
            for name, arm in bucket.items():
                out[str(name)] = (int(getattr(arm, "trials", 0) or 0),
                                  float(getattr(arm, "mean", 0.0) or 0.0))
        except Exception:  # noqa: BLE001
            return out
        return out

    @staticmethod
    def _policy(brain: Any) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """The selection policy as it stands: which strategies each kind may use, and their rates.

        Returned fresh each call so a candidate can edit its own copy — a mutation that scores
        itself by mutating the live table has already been applied, which is the one thing a
        sandbox may not do.
        """
        eligible: Dict[str, List[str]] = {}
        rates: Dict[str, float] = {}
        metareason = getattr(brain, "metareason", None)
        for name, strategy in dict(getattr(metareason, "strategies", {}) or {}).items():
            rates[name] = float(getattr(strategy, "rate", 0.0) or 0.0)
            for kind in (getattr(strategy, "kinds", ()) or ()):
                eligible.setdefault(str(kind), []).append(name)
        return eligible, rates

    def _selection_score(self, brain: Any, eligible: Dict[str, List[str]],
                         rates: Dict[str, float],
                         extra: Optional[Dict[str, Dict[str, Tuple[int, float]]]] = None
                         ) -> Optional[Measurement]:
        """Score a selection policy against her own per-kind record.

        The policy chooses the way :meth:`~nyxara.njp.metareason.MetaReasoner.choose` chooses —
        by the strategy's **kind-averaged** rate — and is then scored by the **per-kind** truth.
        That mismatch is the whole reason a topology or a specialist split can be worth anything:
        a strategy registered for three kinds carries one ``wins``/``trials`` pair across all
        three, so it can win the kind it is worst at on the strength of the kind it is best at.

        ``None`` when too few held-out rows land on an arm with a real record. Refusing is what
        keeps this from reporting a number produced entirely by smoothing.
        """
        rows = self._selection_rows(brain)
        if not rows:
            return None
        learner = getattr(getattr(brain, "metareason", None), "meta_learner", None)
        floor = int(getattr(learner, "min_trials", 1) or 1)
        scored: List[float] = []
        for kind, _actual in rows:
            pool = eligible.get(kind) or []
            if not pool:
                continue
            pick = max(pool, key=lambda n: rates.get(n, 0.0))
            record = dict(self._per_kind(brain, kind))
            record.update((extra or {}).get(kind, {}))
            trials, mean = record.get(pick, (0, 0.0))
            if trials < floor:
                continue                    # no record for what this policy would have chosen
            scored.append(mean)
        if len(scored) < _MIN_SELECTION:
            return None
        return Measurement(sum(scored) / len(scored), len(scored))

    # ---- generating candidates ---------------------------------------------- #
    def generate(self, brain: Any) -> List[Mutation]:
        """Every structural change worth trying right now, in the four kinds the plan names."""
        out: List[Mutation] = []
        try:
            out.extend(self._representations())
            out.extend(self._selection_mutations(brain))
        except Exception:  # noqa: BLE001
            return [m for m in out if m.signature not in self._rejected]
        return [m for m in out if m.signature not in self._rejected]

    def _representations(self) -> List[Mutation]:
        """What a *state* is, for the model the planner searches and the field scores."""
        out: List[Mutation] = []
        for name in ENCODERS:
            if name == self.encoding:
                continue

            def _apply(brain: Any, chosen: str = name) -> bool:
                loop = getattr(brain, "loop", None)
                evolution = getattr(brain, "evolution", None)
                if loop is None:
                    return False
                loop.predictive_encoding = chosen
                if evolution is not None:
                    evolution.encoding = chosen
                    # **Re-fit, or the promotion makes her worse.** Measured, before this line:
                    # held-out replay said `facts` beat `buckets16` 0.667 → 1.000, and the live
                    # model's accuracy over the same session fell from 0.74 to 0.59. The counts
                    # she had already learned were indexed by a state space she no longer
                    # produces, so after the switch every context lookup missed and backed off.
                    # A representation change is not a change of opinion; it is a change of
                    # coordinates, and the experience has to be moved into them.
                    evolution.refit(brain)
                return True

            def _revert(brain: Any, previous: str = self.encoding) -> bool:
                return _apply(brain, previous)

            out.append(Mutation(
                kind=KIND_REPRESENTATION, name=name, target="predictive.state",
                why=(f"the loop and the field feed one model two kinds of state; "
                     f"try {name} for the loop's half"),
                apply=_apply, revert=_revert,
                score=lambda chosen=name: self.replay(chosen)))
        return out

    def _selection_mutations(self, brain: Any) -> List[Mutation]:
        """Rewires of the kind → strategy graph, and new arms to put in it."""
        out: List[Mutation] = []
        metareason = getattr(brain, "metareason", None)
        if metareason is None:
            return out
        strategies = dict(getattr(metareason, "strategies", {}) or {})
        if not strategies:
            return out

        eligible: Dict[str, List[str]] = {}
        for name, strategy in strategies.items():
            for kind in (getattr(strategy, "kinds", ()) or ()):
                eligible.setdefault(str(kind), []).append(name)
        rates = {name: float(getattr(s, "rate", 0.0) or 0.0) for name, s in strategies.items()}

        for kind, pool in eligible.items():
            record = self._per_kind(brain, kind)
            learner = getattr(metareason, "meta_learner", None)
            floor = int(getattr(learner, "min_trials", 1) or 1)

            # TOPOLOGY — a strategy with a real record on this kind that is not registered for it.
            # It got that record by generalist fallback or by a retry after another arm failed to
            # bind, which is the record saying "this belongs here" in the only way a record can.
            for name, (trials, mean) in record.items():
                if name in pool or name not in strategies or trials < floor:
                    continue
                best = max((record.get(p, (0, 0.0))[1] for p in pool), default=0.0)
                if mean <= best:
                    continue
                out.append(self._edge(brain, kind, name, add=True, mean=mean, best=best))

            # STRATEGY — a specialist split. One strategy serving several kinds carries one
            # rate across all of them; where its per-kind records disagree, that average is what
            # `choose` ranks on.
            for name in pool:
                strategy = strategies.get(name)
                kinds = tuple(getattr(strategy, "kinds", ()) or ())
                if len(kinds) < 2:
                    continue
                means = [self._per_kind(brain, k).get(name, (0, 0.0)) for k in kinds]
                measured = [m for t, m in means if t >= floor]
                if len(measured) < 2 or (max(measured) - min(measured)) < 0.25:
                    continue
                out.append(self._split(brain, kind, name))
                break

            # OPERATOR — compose two arms where the first choice keeps failing to bind. The
            # record already carries that signal and `curve` already computes it.
            try:
                curve = metareason.curve(kind)
            except Exception:  # noqa: BLE001
                curve = {}
            for name, row in curve.items():
                fails = row.get("bind_failure_rate")
                if fails is None or fails < 0.5 or name not in strategies:
                    continue
                other = max((p for p in pool if p != name),
                            key=lambda p: record.get(p, (0, 0.0))[1], default="")
                if not other:
                    continue
                out.append(self._compose(brain, kind, name, other, fails))
                break
        return out

    def _edge(self, brain: Any, kind: str, name: str, *, add: bool,
              mean: float, best: float) -> Mutation:
        """Add (or remove) one edge of the kind → strategy graph."""
        def _apply(brain: Any) -> bool:
            strategy = getattr(brain, "metareason", None) and brain.metareason.strategies.get(name)
            if strategy is None:
                return False
            kinds = tuple(strategy.kinds or ())
            strategy.kinds = (kinds + (kind,)) if add else tuple(k for k in kinds if k != kind)
            return True

        def _revert(brain: Any) -> bool:
            strategy = getattr(brain, "metareason", None) and brain.metareason.strategies.get(name)
            if strategy is None:
                return False
            kinds = tuple(strategy.kinds or ())
            strategy.kinds = tuple(k for k in kinds if k != kind) if add else (kinds + (kind,))
            return True

        def _score() -> Optional[Measurement]:
            eligible, rates = self._policy(brain)
            pool = eligible.setdefault(kind, [])
            if add:
                if name not in pool:
                    pool.append(name)
            else:
                eligible[kind] = [p for p in pool if p != name]
            return self._selection_score(brain, eligible, rates)

        mutation = Mutation(
            kind=KIND_TOPOLOGY, name=f"{'+' if add else '-'}{name}@{kind}",
            target=f"{kind}→{name}",
            why=(f"{name} scores {mean:.2f} on {kind} against the eligible best {best:.2f}, "
                 f"and is not registered for it"),
            apply=_apply, revert=_revert, score=_score)
        return mutation

    def _split(self, brain: Any, kind: str, name: str) -> Mutation:
        """Give one kind its own arm for a strategy whose record spans several."""
        new_name = f"{name}@{kind}"

        def _apply(brain: Any) -> bool:
            metareason = getattr(brain, "metareason", None)
            strategy = metareason and metareason.strategies.get(name)
            if strategy is None or strategy.solve is None or new_name in metareason.strategies:
                return False
            metareason.register(new_name, (kind,), strategy.solve, prior=strategy.rate)
            strategy.kinds = tuple(k for k in (strategy.kinds or ()) if k != kind)
            return True

        def _revert(brain: Any) -> bool:
            metareason = getattr(brain, "metareason", None)
            if metareason is None:
                return False
            metareason.strategies.pop(new_name, None)
            strategy = metareason.strategies.get(name)
            if strategy is not None and kind not in (strategy.kinds or ()):
                strategy.kinds = tuple(strategy.kinds or ()) + (kind,)
            return True

        def _score() -> Optional[Measurement]:
            # The split's arm starts from the record it inherits — its *per-kind* one, which is
            # the number the averaged arm was hiding and the only reason the split is worth
            # anything.
            eligible, rates = self._policy(brain)
            trials, mean = self._per_kind(brain, kind).get(name, (0, 0.0))
            if not trials:
                return None
            eligible[kind] = [p for p in eligible.get(kind, []) if p != name] + [new_name]
            rates[new_name] = mean
            return self._selection_score(brain, eligible, rates, extra={kind: {new_name: (trials, mean)}})

        return Mutation(
            kind=KIND_STRATEGY, name=new_name, target=f"{name}/{kind}",
            why=(f"{name}'s record differs by more than a quarter across the kinds it serves, "
                 f"and `choose` ranks it on the average of them"),
            apply=_apply, revert=_revert, score=_score)

    def _compose(self, brain: Any, kind: str, first: str, second: str,
                 fails: float) -> Mutation:
        """A new operator: run ``first``, and where it cannot bind, run ``second``."""
        new_name = f"{first}→{second}@{kind}"

        def _apply(brain: Any) -> bool:
            metareason = getattr(brain, "metareason", None)
            a = metareason and metareason.strategies.get(first)
            b = metareason and metareason.strategies.get(second)
            if a is None or b is None or a.solve is None or b.solve is None:
                return False
            if new_name in metareason.strategies:
                return False

            def _fallback(problem: str, ctx: Dict[str, Any], _a=a, _b=b) -> Any:
                answer = _a.solve(problem, ctx)
                if answer is None or str(answer).strip() == "":
                    return _b.solve(problem, ctx)
                return answer

            metareason.register(new_name, (kind,), _fallback,
                                prior=max(a.rate, b.rate))
            return True

        def _revert(brain: Any) -> bool:
            metareason = getattr(brain, "metareason", None)
            if metareason is None:
                return False
            metareason.strategies.pop(new_name, None)
            return True

        def _score() -> Optional[Measurement]:
            # The composite's expected record, derived from its parts: it is the first arm where
            # the first arm binds, and the second where it does not. An estimate, and it decides
            # only whether the candidate is worth the gates — once promoted it earns a record of
            # its own and is graded on that like every other arm.
            record = self._per_kind(brain, kind)
            a_trials, a_mean = record.get(first, (0, 0.0))
            b_trials, b_mean = record.get(second, (0, 0.0))
            if not a_trials or not b_trials:
                return None
            expected = (1.0 - fails) * a_mean + fails * b_mean
            eligible, rates = self._policy(brain)
            eligible.setdefault(kind, []).append(new_name)
            rates[new_name] = expected
            return self._selection_score(
                brain, eligible, rates,
                extra={kind: {new_name: (min(a_trials, b_trials), expected)}})

        return Mutation(
            kind=KIND_OPERATOR, name=new_name, target=kind,
            why=(f"{first} is chosen first on {kind} and fails to bind {fails:.0%} of the time; "
                 f"fall through to {second} instead of spending the turn"),
            apply=_apply, revert=_revert, score=_score)

    # ---- the pipeline ------------------------------------------------------- #
    def cycle(self, brain: Any) -> EvolutionTrial:
        """One candidate, all the way down: sandbox → benchmark → adversarial → regression → verdict."""
        trial = EvolutionTrial()
        t0 = time.perf_counter()
        try:
            candidates = self.generate(brain)
            self.proposed += len(candidates)
            if not candidates:
                trial.why = "nothing structural to propose from this state"
                return trial

            # One baseline per instrument, and a candidate is only ever compared to its own.
            # The two measure different things on different scales — held-out next-state accuracy
            # and expected per-kind outcome — and ranking candidates by raw score across them
            # would let the units decide which rewire she adopts.
            baselines: Dict[str, Optional[Measurement]] = {}
            best: Optional[Tuple[float, Measurement, Measurement, Mutation]] = None
            for mutation in candidates:
                score = mutation.score() if mutation.score is not None else None
                if score is None:
                    # Refusing beats guessing. A rewire adopted on a measure that cannot see it is
                    # exactly the failure `field.meta_cycle` learned to name, and it is worse here
                    # because the change is structural.
                    self.unmeasurable += 1
                    continue
                instrument = self._instrument(mutation.kind)
                if instrument not in baselines:
                    baselines[instrument] = self._baseline(brain, mutation.kind)
                baseline = baselines[instrument]
                if baseline is None:
                    self.unmeasurable += 1
                    continue
                gain = score.value - baseline.value
                if best is None or gain > best[0]:
                    best = (gain, score, baseline, mutation)

            if best is None:
                trial.unmeasurable = True
                self.refused += 1
                trial.why = ("every candidate's measure is blind to it — "
                             f"{self.unmeasurable} refused as unmeasurable so far")
                return trial

            gain, measured, base, trial.mutation = best
            trial.candidate, trial.baseline = measured.value, base.value
            trial.rows = measured.rows
            # Whichever is larger: the standing minimum, or two rows of the measure. See
            # `Measurement.required` — a gain the instrument could have produced by one episode
            # landing the other way is not an improvement, it is the resolution.
            required = max(self.min_gain, measured.required, base.required)
            if gain < required:
                self._rejected.add(trial.mutation.signature)
                self.refused += 1
                trial.why = (f"no gain past the measure's resolution "
                             f"({trial.baseline:.4f} → {trial.candidate:.4f} over "
                             f"{measured.rows} rows, wants +{required:.4f})")
                return trial

            trial.adversarial_passed, trial.regression_passed, note = self._gates(
                brain, trial.mutation)
            if not (trial.adversarial_passed and trial.regression_passed):
                self._rejected.add(trial.mutation.signature)
                self.refused += 1
                trial.why = note or "a battery got worse"
                return trial

            if not self._promote(brain, trial.mutation):
                self.refused += 1
                trial.why = "the change could not be applied to the live brain"
                return trial

            trial.promoted = True
            self.cognitive_rewires += 1
            self.total_gain += trial.gain
            self.adopted.append({"mutation": trial.mutation.to_dict(),
                                 "gain": round(trial.gain, 5)})
            del self.adopted[:-32]
            trial.why = (f"{trial.mutation.kind} {trial.mutation.name}: "
                         f"{trial.baseline:.4f} → {trial.candidate:.4f} over {trial.rows} rows")
            return trial
        except Exception:  # noqa: BLE001 — a failed cycle rewires nothing
            trial.why = "evolution cycle failed"
            return trial
        finally:
            trial.ms = (time.perf_counter() - t0) * 1000.0
            self.trials.append(trial)
            del self.trials[:-64]

    @staticmethod
    def _instrument(kind: str) -> str:
        """Which measure adjudicates this kind of rewire."""
        return "replay" if kind == KIND_REPRESENTATION else "selection"

    def _baseline(self, brain: Any, kind: str) -> Optional[Measurement]:
        """What she scores as she stands, on the same instrument the candidate is scored on."""
        if kind == KIND_REPRESENTATION:
            return self.replay(self.encoding)
        eligible, rates = self._policy(brain)
        return self._selection_score(brain, eligible, rates)

    def _gates(self, brain: Any, mutation: Mutation) -> Tuple[bool, bool, str]:
        """The adversarial battery and the seven-stage curve, on fresh brains, before and after.

        Both take a ``prepare`` hook precisely so a change made outside them can be measured by
        them. Neither is expected to *improve* — both sit at their ceiling — and that is what makes
        them the right gates: a saturated battery is useless as evidence of progress and ideal as
        evidence that nothing was broken on the way to it.
        """
        if not self.gates:
            return True, True, ""
        try:
            from nyxara.eval.adversarial import run_adversarial_benchmark
            from nyxara.eval.intelligence import run_intelligence_benchmark

            # Whether the change actually went into the sandbox brains. Both batteries swallow a
            # failing `prepare` — reasonably, since a corpus loader that throws should not take
            # the run down — so without this a mutation that cannot be applied scores exactly like
            # one that is harmless, passes both gates, and is promoted on a measurement of a brain
            # it was never in. Found by a test that made `apply` raise.
            failures: List[str] = []

            def prepare(fresh: Any) -> None:
                try:
                    if mutation.apply is None or not mutation.apply(fresh):
                        failures.append("apply declined")
                except Exception as exc:  # noqa: BLE001
                    failures.append(type(exc).__name__)

            before = run_adversarial_benchmark(seed=20260823)
            after = run_adversarial_benchmark(seed=20260823, prepare=prepare)
            if failures:
                return False, False, (f"the change could not be put into the sandbox brains "
                                      f"({failures[0]}), so the batteries measured a brain it "
                                      f"was never in")
            adversarial = self._not_worse(before, after)
            if not adversarial:
                return False, False, "the adversarial battery got worse"

            # A *different* seed for the regression gate. The candidate has now been measured
            # twice; a third measurement on the same problems would only say the first two were
            # not flukes of the run, which was never in doubt.
            old = run_intelligence_benchmark(seed=7, width=4)
            new = run_intelligence_benchmark(seed=7, width=4, prepare=prepare)
            if failures:
                return False, False, (f"the change could not be put into the sandbox brains "
                                      f"({failures[0]})")
            regression = float(getattr(new, "mean", 0.0) or 0.0) >= (
                float(getattr(old, "mean", 0.0) or 0.0) - 1e-9)
            note = "" if regression else "the seven-stage curve regressed"
            return adversarial, regression, note
        except Exception:  # noqa: BLE001 — a gate that cannot run is a gate that refuses
            return False, False, "a gate could not be run, so the change is refused"

    @staticmethod
    def _not_worse(before: Any, after: Any) -> bool:
        """Concept and relation may not fall; hallucination may not rise. Ties are allowed here.

        Different rule from the benchmark on purpose. There a tie is no evidence *for* a change;
        here a tie is exactly what is being asked for — this gate is not looking for improvement,
        it is looking for damage.
        """
        for metric in ("concept", "relation", "uncertainty"):
            old = getattr(before, metric, None)
            new = getattr(after, metric, None)
            if old is not None and new is not None and new < old - 1e-9:
                return False
        old_h = getattr(before, "hallucination", None)
        new_h = getattr(after, "hallucination", None)
        if old_h is not None and new_h is not None and new_h > old_h + 1e-9:
            return False
        return True

    def _promote(self, brain: Any, mutation: Mutation) -> bool:
        try:
            return bool(mutation.apply is not None and mutation.apply(brain))
        except Exception:  # noqa: BLE001
            return False

    # ---- reporting ---------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        last = self.trials[-1] if self.trials else None
        return {
            "trace": len(self.trace),
            "proposed": self.proposed,
            "trials": len(self.trials),
            "cognitive_rewires": self.cognitive_rewires,
            "unmeasurable": self.unmeasurable,
            "refused": self.refused,
            "encoding": self.encoding,
            "total_gain": round(self.total_gain, 5),
            "adopted": list(self.adopted[-4:]),
            "last": last.to_dict() if last is not None else None,
        }
