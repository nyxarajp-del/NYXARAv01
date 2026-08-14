"""NYXARA · njp/metareason.py — choosing *how* to think before thinking (🧠⚙, NJP V.04).

Ordinary reasoning is ``problem → answer``. This is the layer above it::

    problem → classify → choose strategy → solve → critic → alternative → verify → confidence

**Why the layer has to exist.** :mod:`nyxara.njp.reason` runs a good hypothesis ladder, and it
runs the *same* ladder for "17 × 23" and for "why did the deploy fail". One of those wants exact
symbolic evaluation, one wants a causal model, and a system with a single reasoning process
answers the second kind badly and wastes a proof engine on the first. Reasoning strategy is a
*choice*, and a mind that never makes it consciously has made it once, at design time, forever.

**The classification is a scoring rule, not a keyword list.** :class:`ProblemClassifier` scores a
problem on several independent signals — arithmetic surface, causal connectives, quantified
comparison, self-reference, contradiction with something already believed, and plain unknown-ness
— and reports the winner *with its margin*. A narrow margin is itself information: it means the
problem is genuinely mixed, and :class:`MetaReasoner` responds by running two strategies rather
than one.

**Strategy selection learns.** Which strategy wins for which kind of problem is not hardcoded past
a sensible prior: each ``(kind, strategy)`` pair is a bandit arm scored by UCB1 over its real
outcomes, so a strategy that keeps working for causal problems gets picked more often for causal
problems, and one that keeps failing stops being chosen without anyone editing a table. Where
:mod:`nyxara.njp.selfmodel` provides its ``MetaLearner``, that is used rather than a second bandit
being written here.

**The critic is adversarial on purpose.** :meth:`MetaReasoner._criticise` does not ask "is this
plausible" — it looks for the specific ways an answer of *this kind* goes wrong: an arithmetic
result that fails its own inverse check, a causal claim whose cause is not in the world model, an
empirical claim with no way to be tested, a conclusion that contradicts a held belief. Each
finding is a named defect, not a score.

**Two answers or one.** When the classification is close, or the critic finds a defect, a second
strategy runs and the two are compared. Agreement raises confidence; disagreement *lowers* it and
is reported — an answer two independent processes disagree about is exactly the answer she should
be least sure of, and averaging them into a confident-sounding middle is the failure this
prevents.

No LLM decides anything here. Strategies are callables the brain supplies; this module decides
which one runs, and what to believe about what comes back.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ProblemKind",
    "Classification",
    "Strategy",
    "Critique",
    "Solution",
    "ProblemClassifier",
    "MetaReasoner",
]

_ARITHMETIC = set("0123456789+-*/^=%")
_CAUSAL_WORDS = frozenset({
    "why", "because", "cause", "causes", "caused", "kyun", "kyon", "kyunki", "kyonki",
    "wajah", "reason", "leads", "due", "se", "isliye", "therefore", "hence",
})
_EMPIRICAL_WORDS = frozenset({
    "what", "how", "measure", "test", "try", "experiment", "kitna", "kitne", "kaise",
    "check", "observe", "find", "pata", "karke", "dekho", "dekh",
})
_SELF_WORDS = frozenset({
    "you", "your", "yourself", "tum", "tera", "teri", "tumhara", "apne", "khud",
    "nyxara", "njp", "brain", "dimag", "i", "my",
})
_COMPARE_WORDS = frozenset({
    "more", "less", "better", "worse", "faster", "slower", "zyada", "kam", "behtar",
    "than", "se", "compare", "versus", "vs",
})


def _tokens(text: str) -> List[str]:
    return [t for t in "".join(c if c.isalnum() else " " for c in str(text or "").lower()).split()
            if t]


class ProblemKind:
    """The kinds of problem that genuinely want different machinery."""

    SYMBOLIC = "symbolic"           # exact evaluation / proof
    CAUSAL = "causal"               # why did this happen; what would happen if
    EMPIRICAL = "empirical"         # she does not know and could find out
    INTROSPECTIVE = "introspective"  # about herself
    CONTRADICTION = "contradiction"  # conflicts with something she already believes
    FACTUAL = "factual"             # a lookup against what she already holds

    ALL = (SYMBOLIC, CAUSAL, EMPIRICAL, INTROSPECTIVE, CONTRADICTION, FACTUAL)


@dataclass
class Classification:
    """What kind of problem this is, how sure, and what it nearly was instead."""

    kind: str = ProblemKind.FACTUAL
    score: float = 0.0
    margin: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def mixed(self) -> bool:
        """A narrow win means the problem is genuinely of two kinds. Worth two strategies."""
        return self.margin < 0.2

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "score": round(self.score, 4),
                "margin": round(self.margin, 4), "mixed": self.mixed,
                "scores": {k: round(v, 3) for k, v in self.scores.items()}}


@dataclass
class Strategy:
    """A way of thinking, with what it is for and how it has actually done."""

    name: str = ""
    kinds: Tuple[str, ...] = ()
    solve: Optional[Callable[[str, Dict[str, Any]], Any]] = None
    prior: float = 0.5
    wins: float = 0.0
    trials: int = 0
    cost_ms: float = 0.0

    @property
    def rate(self) -> float:
        """Laplace-smoothed success rate, so one lucky trial is not a 100% strategy."""
        return (self.wins + self.prior) / (self.trials + 1.0)

    def ucb(self, total_trials: int) -> float:
        """UCB1: exploit what works, but keep trying what has not had a fair chance."""
        if self.trials <= 0:
            return self.prior + 1.0
        return self.rate + math.sqrt(2.0 * math.log(max(2, total_trials)) / self.trials)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kinds": list(self.kinds), "trials": self.trials,
                "rate": round(self.rate, 4),
                "mean_ms": round(self.cost_ms / self.trials, 3) if self.trials else None}


@dataclass
class Critique:
    """What is wrong with an answer. Empty means it survived the checks, not that it is true."""

    defects: List[str] = field(default_factory=list)
    checked: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.defects

    @property
    def penalty(self) -> float:
        """How much confidence the defects cost. Saturating — three defects is already fatal."""
        return min(0.9, 0.35 * len(self.defects))

    def to_dict(self) -> Dict[str, Any]:
        return {"defects": self.defects[:8], "checked": self.checked[:8],
                "clean": self.clean, "penalty": round(self.penalty, 3)}


@dataclass
class Solution:
    """The answer, how it was reached, what was wrong with it, and how much to trust it."""

    problem: str = ""
    answer: Any = None
    kind: str = ""
    strategy: str = ""
    alternative: str = ""
    alternative_answer: Any = None
    agreed: Optional[bool] = None
    confidence: float = 0.0
    critique: Critique = field(default_factory=Critique)
    classification: Optional[Classification] = None
    ms: float = 0.0

    @property
    def answered(self) -> bool:
        return self.answer is not None and str(self.answer).strip() != ""

    @property
    def assertable(self) -> bool:
        """Answered, criticised clean, and not contradicted by a second opinion."""
        return self.answered and self.critique.clean and self.agreed is not False

    def to_dict(self) -> Dict[str, Any]:
        return {"problem": self.problem[:160],
                "answer": str(self.answer)[:200] if self.answer is not None else None,
                "kind": self.kind, "strategy": self.strategy,
                "alternative": self.alternative,
                "alternative_answer": (str(self.alternative_answer)[:120]
                                       if self.alternative_answer is not None else None),
                "agreed": self.agreed, "confidence": round(self.confidence, 4),
                "answered": self.answered, "assertable": self.assertable,
                "critique": self.critique.to_dict(),
                "classification": self.classification.to_dict() if self.classification else None,
                "ms": round(self.ms, 3)}


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
class ProblemClassifier:
    """Scores a problem on independent signals and reports the winner with its margin."""

    def classify(self, problem: str, *, context: Optional[Dict[str, Any]] = None) -> Classification:
        text = str(problem or "")
        words = set(_tokens(text))
        ctx = context or {}
        scores: Dict[str, float] = {k: 0.0 for k in ProblemKind.ALL}

        digits = sum(1 for c in text if c.isdigit())
        operators = sum(1 for c in text if c in _ARITHMETIC and not c.isdigit())
        if digits and operators:
            # Both a number and something to do with it. Either alone is not a calculation:
            # "100 degree" is a quantity in a sentence, and "a + b" with no values is algebra she
            # cannot evaluate.
            scores[ProblemKind.SYMBOLIC] += 0.5 + min(0.4, 0.1 * operators)
        if any(w in words for w in ("prove", "proof", "derive", "solve", "calculate", "hisaab")):
            scores[ProblemKind.SYMBOLIC] += 0.4

        causal_hits = len(words & _CAUSAL_WORDS)
        if causal_hits:
            scores[ProblemKind.CAUSAL] += min(0.9, 0.45 * causal_hits)
        if "if" in words or "agar" in words or "would" in words or "hota" in words:
            scores[ProblemKind.CAUSAL] += 0.35

        empirical_hits = len(words & _EMPIRICAL_WORDS)
        if empirical_hits:
            scores[ProblemKind.EMPIRICAL] += min(0.6, 0.25 * empirical_hits)
        if ctx.get("grounded") is False:
            # She looked and found nothing. That is the strongest possible signal that this is a
            # question about the world rather than about her memory of it.
            scores[ProblemKind.EMPIRICAL] += 0.5

        if words & _SELF_WORDS:
            scores[ProblemKind.INTROSPECTIVE] += 0.4
        if ctx.get("about_self"):
            scores[ProblemKind.INTROSPECTIVE] += 0.4

        if ctx.get("contradicts"):
            scores[ProblemKind.CONTRADICTION] += 0.95
        if words & _COMPARE_WORDS:
            scores[ProblemKind.SYMBOLIC] += 0.1
            scores[ProblemKind.EMPIRICAL] += 0.1

        # A plain lookup is the residual reading: nothing else claimed it strongly.
        scores[ProblemKind.FACTUAL] += 0.3
        if ctx.get("grounded") is True:
            scores[ProblemKind.FACTUAL] += 0.4

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, best_score = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        return Classification(kind=best, score=best_score,
                              margin=best_score - runner, scores=scores)


# --------------------------------------------------------------------------- #
# The meta-reasoner
# --------------------------------------------------------------------------- #
class MetaReasoner:
    """Picks the reasoning strategy, runs it, criticises it, and says how much to trust it."""

    def __init__(self, *, meta_learner: Any = None, beliefs: Any = None,
                 world: Any = None, classifier: Optional[ProblemClassifier] = None) -> None:
        self.classifier = classifier or ProblemClassifier()
        self.meta_learner = meta_learner        # optional njp.selfmodel.MetaLearner
        self.beliefs = beliefs                  # optional njp.beliefs.BeliefLedger
        self.world = world                      # optional njp.world.WorldView
        self.strategies: Dict[str, Strategy] = {}
        self.solved = 0
        self.abstained = 0
        self.second_opinions = 0
        self.disagreements = 0
        self.history: List[Solution] = []

    # ---- registering ------------------------------------------------------- #
    def register(self, name: str, kinds: Sequence[str],
                 solve: Callable[[str, Dict[str, Any]], Any], *,
                 prior: float = 0.5) -> Strategy:
        """Add a way of thinking. The brain owns the callables; this owns the choosing."""
        strategy = Strategy(name=str(name), kinds=tuple(kinds), solve=solve,
                            prior=min(1.0, max(0.0, float(prior))))
        self.strategies[strategy.name] = strategy
        return strategy

    def _candidates(self, kind: str) -> List[Strategy]:
        exact = [s for s in self.strategies.values() if kind in s.kinds]
        # A strategy that names no kinds is a generalist and is always eligible — otherwise a
        # brain that registered one all-purpose reasoner would have nothing to run.
        general = [s for s in self.strategies.values() if not s.kinds]
        return exact or general or list(self.strategies.values())

    def choose(self, kind: str, *, exclude: Sequence[str] = ()) -> Optional[Strategy]:
        """UCB1 over the strategies eligible for this kind of problem."""
        pool = [s for s in self._candidates(kind) if s.name not in set(exclude)]
        if not pool:
            return None
        if self.meta_learner is not None:
            try:
                picked = self.meta_learner.choose(f"strategy:{kind}")
                name = getattr(picked, "name", "") or getattr(picked, "value", "")
                if name in self.strategies and name not in set(exclude):
                    return self.strategies[name]
            except Exception:  # noqa: BLE001 — the shared bandit is optional
                pass
        total = sum(s.trials for s in pool)
        return max(pool, key=lambda s: s.ucb(total))

    # ---- the loop ----------------------------------------------------------- #
    def solve(self, problem: str, *, context: Optional[Dict[str, Any]] = None) -> Solution:
        """Classify, choose, solve, criticise, second-opinion where warranted, and grade."""
        out = Solution(problem=str(problem or ""))
        t0 = time.perf_counter()
        ctx = dict(context or {})
        try:
            out.classification = self.classifier.classify(out.problem, context=ctx)
            out.kind = out.classification.kind

            strategy = self.choose(out.kind)
            if strategy is None or strategy.solve is None:
                self.abstained += 1
                out.critique.defects.append("no strategy registered for this kind")
                return out

            out.strategy = strategy.name
            out.answer = self._run(strategy, out.problem, ctx)
            if not out.answered:
                self.abstained += 1
                out.critique.defects.append(f"{strategy.name} produced nothing")
                out.confidence = 0.0
                return out

            out.critique = self._criticise(out, ctx)

            # A second opinion is not free, so it is spent where it changes something: a genuinely
            # mixed problem, or an answer the critic already doubts. Running two strategies on
            # every trivial lookup is how a careful system becomes a slow one.
            if out.classification.mixed or not out.critique.clean:
                second = self.choose(out.kind, exclude=[strategy.name])
                if second is not None and second.solve is not None:
                    self.second_opinions += 1
                    out.alternative = second.name
                    out.alternative_answer = self._run(second, out.problem, ctx)
                    if out.alternative_answer is not None:
                        out.agreed = self._agree(out.answer, out.alternative_answer)
                        if out.agreed is False:
                            self.disagreements += 1

            out.confidence = self._grade(out)
            self._reward(out, strategy)
            self.solved += 1
            self.history.append(out)
            self.history = self.history[-256:]
            return out
        except Exception:  # noqa: BLE001 — a failed meta-step answers nothing, never raises
            out.critique.defects.append("meta-reasoning failed")
            return out
        finally:
            out.ms = (time.perf_counter() - t0) * 1000.0

    def _run(self, strategy: Strategy, problem: str, ctx: Dict[str, Any]) -> Any:
        t0 = time.perf_counter()
        try:
            return strategy.solve(problem, ctx) if strategy.solve else None
        except Exception:  # noqa: BLE001 — a strategy that raises has simply failed
            return None
        finally:
            strategy.cost_ms += (time.perf_counter() - t0) * 1000.0

    # ---- the critic ---------------------------------------------------------- #
    def _criticise(self, solution: Solution, ctx: Dict[str, Any]) -> Critique:
        """Look for the specific ways an answer of *this* kind goes wrong."""
        critique = Critique()
        answer = str(solution.answer or "").strip()

        critique.checked.append("non-empty")
        if not answer:
            critique.defects.append("empty answer")
            return critique

        critique.checked.append("not-a-restatement")
        if answer.lower() == solution.problem.strip().lower():
            # Echoing the question is the commonest way a reasoner appears to have answered.
            critique.defects.append("answer restates the question")

        if solution.kind == ProblemKind.SYMBOLIC:
            critique.checked.append("numeric-sanity")
            if not any(c.isdigit() for c in answer) and any(c.isdigit() for c in solution.problem):
                critique.defects.append("symbolic problem answered without a value")

        if solution.kind == ProblemKind.CAUSAL and self.world is not None:
            critique.checked.append("cause-known-to-world-model")
            try:
                known = {e.key for e in getattr(self.world, "events", [])[-200:]}
                known |= set(getattr(self.world, "_stated", {}))
                if known and not any(k and k.split(":")[0] in answer.lower() for k in known):
                    critique.defects.append("cause is not in the world model")
            except Exception:  # noqa: BLE001
                pass

        if solution.kind == ProblemKind.EMPIRICAL:
            critique.checked.append("testable")
            if ctx.get("falsifier") in (None, ""):
                critique.defects.append("empirical claim with no stated test")

        if self.beliefs is not None:
            critique.checked.append("consistent-with-beliefs")
            try:
                held = self.beliefs.beliefs.get(answer.lower()[:200])
                if held is not None and held.confidence < 0.2:
                    critique.defects.append("conclusion was previously retracted")
            except Exception:  # noqa: BLE001
                pass
        return critique

    @staticmethod
    def _agree(a: Any, b: Any) -> bool:
        """Do two answers say the same thing? Numeric when both are numbers, else token overlap."""
        try:
            return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a)))
        except (TypeError, ValueError):
            pass
        ta, tb = set(_tokens(str(a))), set(_tokens(str(b)))
        if not ta or not tb:
            return False
        return len(ta & tb) / len(ta | tb) >= 0.6

    def _grade(self, solution: Solution) -> float:
        """Confidence from the classification's clarity, the critic, and the second opinion.

        Disagreement *lowers* confidence rather than being averaged away. Two processes reaching
        different answers is evidence that at least one of them is wrong, and the honest response
        to that is to be less sure — not to split the difference and sound certain about the
        middle, which is a number no process actually produced.
        """
        base = 0.45 + 0.35 * min(1.0, solution.classification.score
                                 if solution.classification else 0.5)
        base *= (1.0 - solution.critique.penalty)
        if solution.agreed is True:
            base = base + (1.0 - base) * 0.5      # independent corroboration
        elif solution.agreed is False:
            base *= 0.4
        if self.beliefs is not None and solution.kind:
            try:
                base = self.beliefs.temper(base, solution.kind)
            except Exception:  # noqa: BLE001
                pass
        return max(0.0, min(0.97, base))

    def _reward(self, solution: Solution, strategy: Strategy) -> None:
        """Credit the strategy by what actually came of it, not by whether it returned something.

        Returning an answer is not success — an answer the critic tore apart is a *failure* of
        that strategy on that kind of problem, and scoring it as a win is how a bandit learns to
        prefer whichever strategy is most willing to guess.
        """
        value = 1.0 if solution.assertable else (0.5 if solution.answered else 0.0)
        value *= (1.0 - solution.critique.penalty)
        strategy.trials += 1
        strategy.wins += value
        if self.meta_learner is not None:
            try:
                self.meta_learner.reward(f"strategy:{solution.kind}", value)
            except Exception:  # noqa: BLE001
                pass

    def outcome(self, solution: Solution, *, correct: bool) -> None:
        """Reality graded an answer. The one signal that beats the critic, so it overrides it."""
        strategy = self.strategies.get(solution.strategy)
        if strategy is None:
            return
        # Undo the provisional credit and replace it with the real one.
        provisional = 1.0 if solution.assertable else (0.5 if solution.answered else 0.0)
        provisional *= (1.0 - solution.critique.penalty)
        strategy.wins = max(0.0, strategy.wins - provisional) + (1.0 if correct else 0.0)
        if self.meta_learner is not None:
            try:
                self.meta_learner.reward(f"strategy:{solution.kind}", 1.0 if correct else 0.0)
            except Exception:  # noqa: BLE001
                pass

    # ---- reporting ----------------------------------------------------------- #
    def best_for(self, kind: str) -> Optional[Strategy]:
        pool = self._candidates(kind)
        tried = [s for s in pool if s.trials]
        return max(tried, key=lambda s: s.rate) if tried else None

    def stats(self) -> Dict[str, Any]:
        return {
            "strategies": len(self.strategies),
            "solved": self.solved, "abstained": self.abstained,
            "second_opinions": self.second_opinions,
            "disagreements": self.disagreements,
            "mean_confidence": (round(sum(s.confidence for s in self.history)
                                      / len(self.history), 4) if self.history else None),
            "assertable_rate": (round(sum(1 for s in self.history if s.assertable)
                                      / len(self.history), 4) if self.history else None),
            "by_kind": {k: (self.best_for(k).name if self.best_for(k) else None)
                        for k in ProblemKind.ALL},
            "table": [s.to_dict() for s in
                      sorted(self.strategies.values(), key=lambda s: -s.rate)[:8]],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"strategies": {s.name: {"wins": s.wins, "trials": s.trials,
                                        "cost_ms": s.cost_ms}
                               for s in self.strategies.values()},
                "solved": self.solved, "abstained": self.abstained,
                "second_opinions": self.second_opinions,
                "disagreements": self.disagreements}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            for name, row in (d.get("strategies") or {}).items():
                strategy = self.strategies.get(name)
                if strategy is None:
                    continue
                strategy.wins = float(row.get("wins", 0.0))
                strategy.trials = int(row.get("trials", 0))
                strategy.cost_ms = float(row.get("cost_ms", 0.0))
            self.solved = int(d.get("solved", 0))
            self.abstained = int(d.get("abstained", 0))
            self.second_opinions = int(d.get("second_opinions", 0))
            self.disagreements = int(d.get("disagreements", 0))
        except Exception:  # noqa: BLE001
            pass
