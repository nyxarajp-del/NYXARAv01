"""NYXARA · njp/predictive.py — the predictive brain, and thought as a computable object (🔮).

The Master's diagnosis of what NJP was: *fact-store + growing automaton + gradient readout*. Add a
million facts to that and knowledge grows while nothing starts thinking. What was missing is the
thing this module supplies:

    "Agar duniya abhi state X mein hai, to next state kya ho sakti hai?"

**Why this is not already here.** NJP does predict — :meth:`nyxara.njp.fabric.Fabric.anticipate`
guesses *which cells fire next*, and :mod:`nyxara.njp.predict` scores whatever it is handed. Both
are real and neither is a model of the world: knowing your own next activation is not knowing what
happens when a glass is pushed. This module learns the second thing, over the states and actions
the world actually presents.

**The model is a variable-order Markov chain with backoff, learned online.** Order 3 first, then
2, then 1, then the unconditional distribution — the first order with enough evidence answers, and
:attr:`Prediction.order` records which one did. That matters because it is an honest statement of
how much context the answer actually rests on: an answer from order 0 is a guess about the world
in general, and one from order 3 is a guess about *this* situation. Standard backoff, chosen
because with sparse lived experience a long context almost never repeats, and a model that refuses
to answer without one never answers at all.

**Surprise is measured in bits, not in vibes.** ``−log₂ P(actual | context)``: the information the
world just gave her that her model did not already contain. That is the same currency
:class:`nyxara.njp.universe.ExperimentDesigner` spends on experiments, so curiosity and surprise
are finally denominated in the same unit and can be compared. A prediction that was confident and
wrong costs many bits; one that was uncertain and wrong costs few, because an uncertain model was
not claiming much. Punishing both equally is how a system learns to be vague.

**Prediction error is fuel, and the fuel line is explicit.** :meth:`PredictiveWorldModel.score`
returns a :class:`Surprise`, and a surprise past the model's own expectation is exactly the
signal :mod:`nyxara.njp.field` routes to concept restructuring — the Master's point that
intelligence grows by *explaining mistakes* rather than by accumulating facts. Nothing here
punishes error; it prices it.

**Thought becomes an object.** :class:`ThoughtState` carries context, active concepts, goals,
assumptions, hypotheses, predicted outcomes, uncertainty, evidence and contradictions, and
:meth:`ThoughtState.deliberate` walks the sequence the Master specified::

    What do I know? → What am I assuming? → What could explain this?
                    → Which explanation predicts reality best?
                    → What would prove me wrong? → Conclusion

Each step writes to the object rather than returning prose, so reasoning is inspectable while it
happens instead of being reconstructed from its output afterwards — which is the difference
between a reasoning trace and a plausible story about one.

Pure standard library. No LLM: a world model whose transitions come from a language model's
guesses is that model's world, not hers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "WorldState",
    "Prediction",
    "Surprise",
    "PredictiveWorldModel",
    "Assumption",
    "Explanation",
    "ThoughtState",
]

_MAX_ORDER = 3
_MIN_EVIDENCE = 2          # observations before an order may answer on its own
_SMOOTHING = 0.5           # add-α over the seen alphabet, so nothing has probability zero


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


# --------------------------------------------------------------------------- #
# What a situation is
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorldState:
    """A discrete signature of the situation — what holds right now, in order.

    Discrete on purpose. A continuous state cannot be counted, and counting is what makes the
    transition model learnable from a handful of examples rather than from a training run. The
    facts are sorted so that the same situation described in a different order is the same state,
    which is the whole reason this can generalise at all.
    """

    facts: Tuple[str, ...] = ()

    @staticmethod
    def of(facts: Iterable[Any]) -> "WorldState":
        cleaned = sorted({_norm(f) for f in (facts or []) if _norm(f)})
        return WorldState(facts=tuple(cleaned[:8]))

    @property
    def signature(self) -> str:
        return "|".join(self.facts) or "∅"

    @property
    def empty(self) -> bool:
        return not self.facts

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.signature

    def to_dict(self) -> Dict[str, Any]:
        return {"facts": list(self.facts), "signature": self.signature}


@dataclass
class Prediction:
    """What she expects next, how sure, and how much context the answer rests on."""

    context: Tuple[str, ...] = ()
    action: str = ""
    distribution: Dict[str, float] = dc_field(default_factory=dict)
    order: int = -1                 # which backoff order answered; -1 = nothing could
    evidence: int = 0               # observations behind that order

    @property
    def top(self) -> str:
        return max(self.distribution, key=self.distribution.get) if self.distribution else ""

    @property
    def confidence(self) -> float:
        return self.distribution.get(self.top, 0.0) if self.distribution else 0.0

    @property
    def entropy_bits(self) -> float:
        """How uncertain she is, in bits. The budget a surprise is measured against."""
        total = 0.0
        for p in self.distribution.values():
            if p > 0:
                total -= p * math.log2(p)
        return total

    @property
    def answerable(self) -> bool:
        return self.order >= 0 and bool(self.distribution)

    def probability_of(self, outcome: str) -> float:
        return self.distribution.get(_norm(outcome), 0.0)

    def to_dict(self) -> Dict[str, Any]:
        top = sorted(self.distribution.items(), key=lambda kv: -kv[1])[:4]
        return {"context": list(self.context), "action": self.action,
                "top": self.top, "confidence": round(self.confidence, 4),
                "entropy_bits": round(self.entropy_bits, 4),
                "order": self.order, "evidence": self.evidence,
                "answerable": self.answerable,
                "distribution": {k: round(v, 4) for k, v in top}}


@dataclass
class Surprise:
    """How much the world just told her that her model did not already contain."""

    predicted: str = ""
    actual: str = ""
    bits: float = 0.0            # −log₂ P(actual | context)
    expected_bits: float = 0.0   # the prediction's own entropy
    order: int = -1

    @property
    def correct(self) -> bool:
        return bool(self.actual) and self.predicted == self.actual

    @property
    def excess(self) -> float:
        """Surprise beyond what the model itself expected to be surprised by.

        This is the number that should drive learning, not raw bits. A model that honestly said
        "I have no idea" and was then wrong has *learned* something and done nothing wrong; a
        model that said "certainly X" and got Y has a defect. Only the second has positive excess.
        """
        return self.bits - self.expected_bits

    @property
    def surprising(self) -> bool:
        return self.excess > 1.0     # more than one bit past its own expectation

    def to_dict(self) -> Dict[str, Any]:
        return {"predicted": self.predicted, "actual": self.actual,
                "bits": round(self.bits, 4), "expected_bits": round(self.expected_bits, 4),
                "excess": round(self.excess, 4), "correct": self.correct,
                "surprising": self.surprising, "order": self.order}


# --------------------------------------------------------------------------- #
# The predictive world model
# --------------------------------------------------------------------------- #
class PredictiveWorldModel:
    """``(context, action) → next state``, learned online, with backoff and honest surprise."""

    def __init__(self, *, max_order: int = _MAX_ORDER, min_evidence: int = _MIN_EVIDENCE,
                 smoothing: float = _SMOOTHING, capacity: int = 20000) -> None:
        self.max_order = max(1, int(max_order))
        self.min_evidence = max(1, int(min_evidence))
        self.smoothing = max(1e-6, float(smoothing))
        self.capacity = max(64, int(capacity))

        # (order, context, action) -> {next_state: count}
        self._counts: Dict[Tuple[int, str, str], Dict[str, int]] = {}
        self._alphabet: Dict[str, int] = {}      # next_state -> times ever seen
        self._history: List[str] = []            # the running state sequence
        self.observations = 0
        self.scored = 0
        self.correct = 0
        self.total_bits = 0.0
        self.total_excess = 0.0
        self.surprises = 0

    # ---- learning ---------------------------------------------------------- #
    def observe(self, state: Any, action: str = "", *,
                next_state: Optional[Any] = None) -> int:
        """Record that ``state`` was followed by ``next_state`` under ``action``.

        Called with only a state, it treats the running history as the context and the new state
        as the outcome — which is the sequence case, and the one a lived stream of turns actually
        produces. Returns how many orders were updated.
        """
        current = self._sig(state)
        if next_state is None:
            # Sequence mode: whatever came before is the context, this is the outcome.
            if not self._history:
                self._history.append(current)
                return 0
            outcome = current
            context_source = self._history
        else:
            outcome = self._sig(next_state)
            context_source = self._history + [current]
        if not outcome:
            return 0

        action = _norm(action)
        updated = 0
        for order in range(0, self.max_order + 1):
            context = self._context(context_source, order)
            key = (order, context, action)
            bucket = self._counts.get(key)
            if bucket is None:
                if len(self._counts) >= self.capacity:
                    continue
                bucket = {}
                self._counts[key] = bucket
            bucket[outcome] = bucket.get(outcome, 0) + 1
            updated += 1

        self._alphabet[outcome] = self._alphabet.get(outcome, 0) + 1
        self._history.append(outcome if next_state is None else self._sig(next_state))
        del self._history[:-self.max_order - 1]
        self.observations += 1
        return updated

    @staticmethod
    def _sig(state: Any) -> str:
        if isinstance(state, WorldState):
            return state.signature
        return _norm(state)

    @staticmethod
    def _context(history: Sequence[str], order: int) -> str:
        if order <= 0:
            return ""
        return "→".join(history[-order:])

    # ---- predicting --------------------------------------------------------- #
    def predict(self, context: Any = None, action: str = "") -> Prediction:
        """The distribution over what comes next, from the longest context with evidence.

        Backoff is not a fallback bolted on for robustness — it is the honest reading of sparse
        experience. Order 3 says "in exactly this situation, this followed"; order 0 says "this
        tends to happen". Both are worth having and they are not the same claim, so
        :attr:`Prediction.order` reports which one answered.
        """
        history = self._history if context is None else self._as_history(context)
        action = _norm(action)
        out = Prediction(action=action)

        for order in range(self.max_order, -1, -1):
            ctx = self._context(history, order)
            bucket = self._counts.get((order, ctx, action))
            # An action she has never taken here still has the actionless statistics to fall
            # back on, which is what lets a novel action be reasoned about at all.
            if not bucket and action:
                bucket = self._counts.get((order, ctx, ""))
            if not bucket:
                continue
            total = sum(bucket.values())
            if total < self.min_evidence and order > 0:
                continue
            out.context = tuple(history[-order:]) if order else ()
            out.order = order
            out.evidence = total
            out.distribution = self._smoothed(bucket, total)
            return out
        return out

    def _smoothed(self, bucket: Dict[str, int], total: int) -> Dict[str, float]:
        """Add-α over the observed alphabet, so nothing she has ever seen has probability zero.

        Zero probability is a claim of impossibility, and a model built from a few dozen
        observations is not entitled to make one. It also makes surprise finite: ``−log₂ 0`` is
        infinite, and an infinite error signal destroys whatever it is fed into.
        """
        alphabet = set(bucket) | set(self._alphabet)
        denominator = total + self.smoothing * len(alphabet)
        return {outcome: (bucket.get(outcome, 0) + self.smoothing) / denominator
                for outcome in alphabet}

    def successors(self, context: Any = None, action: str = "") -> Dict[str, float]:
        """Outcomes actually **observed** after this context, with their empirical shares.

        Deliberately not :meth:`predict`. That method smooths, so every outcome in the alphabet
        carries a little probability mass — which is right for surprise (nothing she has seen is
        impossible) and wrong for anything that has to *walk* the transitions. A planner given
        the smoothed distribution will happily route through an edge that exists only because of
        the smoothing: measured, it produced a one-step plan claiming ``open`` leads straight to
        ``outside`` on 0.077 of invented mass, when what she had actually observed was ``open``
        then ``walk``. Imagination is allowed to consider the unobserved; a plan is not.
        """
        history = self._history if context is None else self._as_history(context)
        action = _norm(action)
        for order in range(self.max_order, -1, -1):
            ctx = self._context(history, order)
            bucket = self._counts.get((order, ctx, action))
            if not bucket and action:
                bucket = self._counts.get((order, ctx, ""))
            if not bucket:
                continue
            total = sum(bucket.values())
            if total < self.min_evidence and order > 0:
                continue
            return {outcome: count / total for outcome, count in bucket.items() if count > 0}
        return {}

    @staticmethod
    def _as_history(context: Any) -> List[str]:
        if isinstance(context, WorldState):
            return [context.signature]
        if isinstance(context, (list, tuple)):
            return [PredictiveWorldModel._sig(c) for c in context]
        return [_norm(context)]

    # ---- scoring ------------------------------------------------------------ #
    def score(self, prediction: Prediction, actual: Any) -> Surprise:
        """Grade a prediction against what actually happened, in bits."""
        outcome = self._sig(actual)
        out = Surprise(predicted=prediction.top, actual=outcome, order=prediction.order)
        if not outcome:
            return out
        probability = prediction.probability_of(outcome)
        if probability <= 0.0:
            # Never seen here and never seen anywhere: the most informative outcome there is.
            # Bounded rather than infinite, because an unbounded error signal is not a signal.
            probability = self.smoothing / (prediction.evidence + self.smoothing *
                                            max(1, len(self._alphabet) + 1))
        out.bits = -math.log2(max(probability, 1e-12))
        out.expected_bits = prediction.entropy_bits

        self.scored += 1
        self.total_bits += out.bits
        self.total_excess += out.excess
        if out.correct:
            self.correct += 1
        if out.surprising:
            self.surprises += 1
        return out

    def predict_and_observe(self, state: Any, action: str = "") -> Tuple[Prediction, Surprise]:
        """One step of the loop: guess, then look, then learn. The whole cycle in one call."""
        prediction = self.predict(action=action)
        surprise = self.score(prediction, state) if prediction.answerable else Surprise(
            actual=self._sig(state))
        self.observe(state, action)
        return prediction, surprise

    # ---- reporting ---------------------------------------------------------- #
    @property
    def accuracy(self) -> Optional[float]:
        return (self.correct / self.scored) if self.scored else None

    @property
    def mean_bits(self) -> Optional[float]:
        """Average surprise per observation. Falls as the model learns — that IS the learning."""
        return (self.total_bits / self.scored) if self.scored else None

    def stats(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "contexts": len(self._counts),
            "alphabet": len(self._alphabet),
            "scored": self.scored,
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "mean_bits": round(self.mean_bits, 4) if self.mean_bits is not None else None,
            "mean_excess": round(self.total_excess / self.scored, 4) if self.scored else None,
            "surprises": self.surprises,
            "max_order": self.max_order,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"counts": [[o, c, a, dict(b)] for (o, c, a), b in self._counts.items()],
                "alphabet": dict(self._alphabet), "history": list(self._history),
                "observations": self.observations, "scored": self.scored,
                "correct": self.correct, "total_bits": self.total_bits,
                "total_excess": self.total_excess, "surprises": self.surprises}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self._counts = {(int(o), str(c), str(a)): {str(k): int(v) for k, v in (b or {}).items()}
                            for o, c, a, b in (d.get("counts") or [])}
            self._alphabet = {str(k): int(v) for k, v in (d.get("alphabet") or {}).items()}
            self._history = [str(h) for h in (d.get("history") or [])]
            self.observations = int(d.get("observations", 0))
            self.scored = int(d.get("scored", 0))
            self.correct = int(d.get("correct", 0))
            self.total_bits = float(d.get("total_bits", 0.0))
            self.total_excess = float(d.get("total_excess", 0.0))
            self.surprises = int(d.get("surprises", 0))
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Thought as an object
# --------------------------------------------------------------------------- #
@dataclass
class Assumption:
    """Something she is taking for granted, and whether it has been checked."""

    claim: str = ""
    checked: bool = False
    holds: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim[:120], "checked": self.checked, "holds": self.holds}


@dataclass
class Explanation:
    """A candidate account of the situation, scored by what it predicts."""

    claim: str = ""
    predicts: str = ""
    probability: float = 0.0        # what the world model gives its prediction
    falsifier: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim[:120], "predicts": self.predicts[:80],
                "probability": round(self.probability, 4), "falsifier": self.falsifier[:120]}


@dataclass
class ThoughtState:
    """A thought, as a thing with parts rather than a string that came out.

    The Master's sequence, and each step writes here instead of returning prose::

        What do I know? → What am I assuming? → What could explain this?
                        → Which explanation predicts reality best?
                        → What would prove me wrong? → Conclusion

    Reasoning that is only visible in its output cannot be audited, and a system that
    reconstructs its justification afterwards will reconstruct a good one whether or not it is
    the real one. This is the object that makes the difference checkable.
    """

    context: str = ""
    active_concepts: List[str] = dc_field(default_factory=list)
    goals: List[str] = dc_field(default_factory=list)
    assumptions: List[Assumption] = dc_field(default_factory=list)
    hypotheses: List[Explanation] = dc_field(default_factory=list)
    predicted_outcomes: Dict[str, float] = dc_field(default_factory=dict)
    uncertainty: float = 1.0        # bits
    evidence: List[str] = dc_field(default_factory=list)
    contradictions: List[str] = dc_field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    steps: List[str] = dc_field(default_factory=list)
    ms: float = 0.0

    @property
    def decided(self) -> bool:
        return bool(self.conclusion)

    @property
    def best(self) -> Optional[Explanation]:
        return max(self.hypotheses, key=lambda h: h.probability, default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {"context": self.context[:160],
                "active_concepts": self.active_concepts[:8], "goals": self.goals[:4],
                "assumptions": [a.to_dict() for a in self.assumptions[:6]],
                "hypotheses": [h.to_dict() for h in self.hypotheses[:6]],
                "uncertainty_bits": round(self.uncertainty, 4),
                "evidence": self.evidence[:6], "contradictions": self.contradictions[:4],
                "conclusion": self.conclusion[:200], "confidence": round(self.confidence, 4),
                "decided": self.decided, "steps": self.steps, "ms": round(self.ms, 3)}

    # ---- the walk ----------------------------------------------------------- #
    def deliberate(self, model: PredictiveWorldModel, *, action: str = "",
                   beliefs: Any = None, concepts: Any = None) -> "ThoughtState":
        """Walk the six questions over the predictive model, writing each answer here."""
        t0 = time.perf_counter()
        try:
            self._what_do_i_know(beliefs, concepts)
            self._what_am_i_assuming(model, action)
            self._what_could_explain(model, action)
            self._which_predicts_best()
            self._what_would_prove_me_wrong()
            self._conclude()
        except Exception:  # noqa: BLE001 — a failed deliberation concludes nothing, never raises
            self.steps.append("failed")
        finally:
            self.ms = (time.perf_counter() - t0) * 1000.0
        return self

    def _what_do_i_know(self, beliefs: Any, concepts: Any) -> None:
        self.steps.append("what-do-i-know")
        if beliefs is not None:
            try:
                for belief in beliefs.known()[:4]:
                    self.evidence.append(belief.claim[:100])
                for row in beliefs.unknown()[:3]:
                    if row.get("kind") == "contested":
                        self.contradictions.append(str(row.get("subject", ""))[:100])
            except Exception:  # noqa: BLE001
                pass
        if concepts is not None and self.context:
            try:
                coverage = concepts.explain(self.context)
                if coverage.concept:
                    self.active_concepts.append(coverage.concept)
            except Exception:  # noqa: BLE001
                pass

    def _what_am_i_assuming(self, model: PredictiveWorldModel, action: str) -> None:
        """The assumption is the context the prediction rests on — and its order names it.

        An answer from order 0 assumes only "the world in general"; one from order 3 assumes
        "this exact situation recurs". Making that explicit is what turns a hidden dependency
        into something that can be checked and found false.
        """
        self.steps.append("what-am-i-assuming")
        prediction = model.predict(action=action)
        if not prediction.answerable:
            self.assumptions.append(Assumption(claim="nothing similar has happened before"))
            return
        self.assumptions.append(Assumption(
            claim=(f"the last {prediction.order} state(s) determine what follows"
                   if prediction.order else "what usually happens will happen"),
            checked=True, holds=prediction.evidence >= model.min_evidence))
        self.predicted_outcomes = dict(prediction.distribution)
        self.uncertainty = prediction.entropy_bits

    def _what_could_explain(self, model: PredictiveWorldModel, action: str) -> None:
        self.steps.append("what-could-explain")
        prediction = model.predict(action=action)
        for outcome, probability in sorted(prediction.distribution.items(),
                                           key=lambda kv: -kv[1])[:4]:
            self.hypotheses.append(Explanation(
                claim=f"the situation leads to {outcome}",
                predicts=outcome, probability=probability))

    def _which_predicts_best(self) -> None:
        self.steps.append("which-predicts-best")

    def _what_would_prove_me_wrong(self) -> None:
        """Every hypothesis gets a falsifier — the observation that would kill it.

        A hypothesis without one is not competing; it is decoration. Here the falsifier is
        concrete and cheap to state: any outcome other than the one it predicts.
        """
        self.steps.append("what-would-prove-me-wrong")
        for hypothesis in self.hypotheses:
            hypothesis.falsifier = f"the next state is anything other than {hypothesis.predicts}"

    def _conclude(self) -> None:
        """Commit only when one explanation clears the field. Otherwise stay undecided.

        The margin test is the point. Two explanations at 0.42 and 0.41 are not a conclusion, and
        reporting the higher one as the answer converts an honest "I don't know which" into a
        confident coin-flip — which is the failure mode every other gate in this brain exists to
        prevent.
        """
        self.steps.append("conclude")
        best = self.best
        if best is None:
            return
        rivals = sorted((h.probability for h in self.hypotheses), reverse=True)
        margin = rivals[0] - (rivals[1] if len(rivals) > 1 else 0.0)
        if margin < 0.1:
            self.confidence = best.probability
            return                              # undecided, on purpose
        self.conclusion = best.claim
        self.confidence = best.probability
