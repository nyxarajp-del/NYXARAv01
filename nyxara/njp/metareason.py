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
    # Every strategy tried, in order, ending with the one that answered. More than one name here
    # means the first choice could not bind to this problem — worth seeing, because that is the
    # bandit's prior being wrong about which organ owns this kind of question.
    attempts: List[str] = field(default_factory=list)
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
                "attempts": self.attempts[:4],
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

        if ctx.get("arithmetic"):
            # A closed expression that actually parsed, supplied by the caller. Character-counting
            # cannot separate "2+2" from "100 degree at 3pm"; a parse can, and this flag is only
            # ever set by one that succeeded.
            #
            # Weighted like `contradicts` rather than like `variable`, because it is that kind of
            # evidence: an expression that parsed closed is not a *hint* that the question is
            # symbolic, it is the question being symbolic. It has to outrank EMPIRICAL outright,
            # since `grounded is False` alone puts that at 0.5 and any "kitna"/"how much" phrasing
            # adds 0.25 more — and the empirical critic demands a falsifier, which "4" can never
            # have. Measured before this: "5 ka square kitna hai" classified EMPIRICAL 0.75 over
            # SYMBOLIC 0.60, the calculator was still reached, computed 25, and the critic threw
            # it away as "empirical claim with no stated test". The answer was right and unsaid.
            scores[ProblemKind.SYMBOLIC] += 0.95

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

        if ctx.get("derivable"):
            # The Core can already work this out from facts she holds — supplied by the caller as
            # a completed derivation, not as a guess that one might exist.
            #
            # This corrects a structural bias, not a wording one. `grounded is False` adds 0.5 to
            # EMPIRICAL, so *every* lookup miss reads as "she does not know and could find out"
            # — and derivation is the third option that framing has no room for: she may neither
            # know it nor need to go and get it, because it follows from what she was already
            # told. Measured: "what does sparrow need", with `sparrow is_a bird`, `bird is_a
            # animal` and `animal requires water` all held, classified empirical 0.75 and was
            # answered with a proposed *experiment*, while the three-step inheritance sat
            # underived. Boosting FACTUAL past EMPIRICAL sends it to the strategy that reasons
            # from the store rather than the one that plans a trip outside it.
            scores[ProblemKind.FACTUAL] += 0.6

        if ctx.get("variable"):
            # An intervention she can actually carry out: the question named a variable she has
            # observed and a change she can size. That is a fact about what this question *is*,
            # not an inference from its wording, so it outweighs the word cues — "what if I halve
            # the water" scores as introspective the moment it says "I", and it is not a question
            # about her. Scored from the context rather than from another keyword deliberately:
            # the words that make a counterfactual are the same ones that make an ordinary
            # conditional, and only the parse can tell them apart.
            scores[ProblemKind.CAUSAL] += 0.6

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
                 world: Any = None, classifier: Optional[ProblemClassifier] = None,
                 max_attempts: int = 3) -> None:
        self.classifier = classifier or ProblemClassifier()
        # How many eligible strategies may be tried before the turn is left unanswered. Bounded
        # rather than exhaustive: the point is to reach the organ that can bind to this question,
        # not to run every organ she owns on a question none of them fit.
        self.max_attempts = max(1, int(max_attempts))
        self.meta_learner = meta_learner        # optional njp.selfmodel.MetaLearner
        self.beliefs = beliefs                  # optional njp.beliefs.BeliefLedger
        self.world = world                      # optional njp.world.WorldView
        self.strategies: Dict[str, Strategy] = {}
        # Records loaded from a snapshot for strategies that are not registered yet. See
        # `load_dict`: a runtime-registered arm's history is held here until `register` claims it.
        self._orphaned: Dict[str, Dict[str, Any]] = {}
        self.solved = 0
        self.abstained = 0
        self.second_opinions = 0
        self.disagreements = 0
        self.history: List[Solution] = []

    # ---- registering ------------------------------------------------------- #
    def register(self, name: str, kinds: Sequence[str],
                 solve: Callable[[str, Dict[str, Any]], Any], *,
                 prior: float = 0.5) -> Strategy:
        """Add a way of thinking. The brain owns the callables; this owns the choosing.

        **Also declared to the shared bandit, once per kind it serves.** :meth:`choose` asks
        ``meta_learner.choose(f"strategy:{kind}")`` before anything else and :meth:`outcome`
        rewards the same arm — and nothing had ever registered an option under those arms, so on
        every session ``choose`` returned ``None``, the kind-blind fallback decided every turn, and
        ``reward`` found an empty ``_pending`` and credited nothing. Measured: the learner's arms
        were ``settle_steps``, ``recall_k``, ``reason_depth``, ``seat:causal``, ``seat:factual`` —
        every ``strategy:*`` arm absent. The seat arms are there because
        :mod:`nyxara.njp.router` registers its options properly; this is the same pattern, in the
        organ that most needed it.

        **One arm per kind, and that is the whole point.** :class:`Strategy` carries a single
        ``wins``/``trials`` pair spanning every kind it serves, so ``derive`` — registered for
        factual, causal and empirical — averages three different competences into one number, and
        a strategy that is excellent at one and useless at another reports the mean of the two.
        A per-kind arm is what makes *"which way of thinking suits this kind of problem"* a
        question with an answer, which is the difference between choosing well and choosing on
        average.
        """
        strategy = Strategy(name=str(name), kinds=tuple(kinds), solve=solve,
                            prior=min(1.0, max(0.0, float(prior))))
        # A record held from a snapshot for a strategy that had not been registered yet. Applied
        # here so an arm added at runtime resumes with its history instead of looking brand new —
        # `ucb` returns `prior + 1.0` at zero trials, so a restored arm that kept looking untried
        # would be explored ahead of everything else on every single restart.
        held = self._orphaned.pop(strategy.name, None)
        if held is not None:
            strategy.wins = float(held.get("wins", 0.0))
            strategy.trials = int(held.get("trials", 0))
            strategy.cost_ms = float(held.get("cost_ms", 0.0))
        self.strategies[strategy.name] = strategy
        if self.meta_learner is not None:
            for kind in strategy.kinds:
                try:
                    self.meta_learner.register(f"strategy:{kind}", strategy.name, strategy.name)
                except Exception:  # noqa: BLE001 — the shared bandit is optional throughout
                    break
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

            # A strategy that produced *nothing* has not answered the question, and abstaining
            # here meant the one organ that could answer it was never asked. Measured: "what if I
            # halve the water" over an observed variable classifies CAUSAL, chooses `causal` on
            # its higher prior, gets nothing back — explanation is not intervention — and returns
            # unanswered while `simulate` sits registered, eligible, and able to answer exactly
            # this. The retry spends one more strategy on a turn that had already failed, which is
            # precisely when it is worth spending, and each empty attempt is scored as the failure
            # it was so the bandit stops leading with it.
            tried = [strategy.name]
            while not out.answered and len(tried) < self.max_attempts:
                nxt = self.choose(out.kind, exclude=tried)
                if nxt is None or nxt.solve is None:
                    break
                self._miss(strategy)
                tried.append(nxt.name)
                strategy, out.strategy = nxt, nxt.name
                out.answer = self._run(nxt, out.problem, ctx)

            if not out.answered:
                self.abstained += 1
                self._miss(strategy)
                out.critique.defects.append(f"{' then '.join(tried)} produced nothing")
                out.confidence = 0.0
                return out
            out.attempts = list(tried)

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

    @staticmethod
    def _miss(strategy: Strategy) -> None:
        """Score a strategy that returned nothing as the failure it was.

        Without this an organ that cannot bind to a kind of problem keeps its prior for ever: it
        is chosen, produces nothing, and is never charged for it — so it is chosen again next
        time, ahead of the organ that would have answered.
        """
        strategy.trials += 1

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
                # Named, not left to `_pending`. This outcome comes from the Master's later
                # statement and can arrive many turns after the choice — any question of the same
                # kind in between would otherwise move the pending slot and hand the credit to
                # whichever strategy was chosen most recently.
                self.meta_learner.reward(f"strategy:{solution.kind}",
                                         1.0 if correct else 0.0,
                                         name=solution.strategy)
            except Exception:  # noqa: BLE001
                pass

    # ---- reporting ----------------------------------------------------------- #
    def best_for(self, kind: str) -> Optional[Strategy]:
        """The strategy that has actually done best *on this kind of problem*.

        It used to rank on :attr:`Strategy.rate`, and :meth:`register`'s own docstring says why
        that cannot work: ``rate`` is one ``wins``/``trials`` pair spanning every kind a strategy
        serves, so ``derive`` — registered for factual, causal and empirical — reports the mean of
        three different competences. Ranking per-kind on a kind-averaged number let a strategy win
        the kind it was worst at, on the strength of the kind it was best at. The per-kind record
        already existed in the shared bandit and nothing read it; this reads it.

        Returns ``None`` rather than a guess when no option has cleared ``min_trials``. That is
        the honest state — *not measured yet* — and it is what keeps :meth:`stats` from naming a
        winner chosen by one lucky turn.
        """
        if self.meta_learner is not None:
            try:
                picked = self.meta_learner.best(f"strategy:{kind}")
                name = getattr(picked, "name", "") if picked is not None else ""
                return self.strategies.get(name)
            except Exception:  # noqa: BLE001 — the shared bandit is optional throughout
                pass
        pool = self._candidates(kind)
        tried = [s for s in pool if s.trials]
        return max(tried, key=lambda s: s.rate) if tried else None

    def curve(self, kind: str) -> Dict[str, Dict[str, Any]]:
        """Per-strategy record for one kind of problem: how often, how well, and how it was reached.

        ``first_choice_rate`` and ``bind_failure_rate`` come from :attr:`Solution.attempts`, whose
        own comment advertises exactly this signal — *"more than one name here means the first
        choice could not bind to this problem… that is the bandit's prior being wrong about which
        organ owns this kind of question"* — and which was written every turn and read by nobody.

        A strategy that is chosen first and then cannot bind is a different failure from one that
        binds and answers badly, and only the second is the strategy being wrong. Reporting one
        number for both is how a bandit learns to prefer whichever strategy guesses most.
        """
        bucket: Dict[str, Any] = {}
        if self.meta_learner is not None:
            try:
                bucket = dict(self.meta_learner.strategies.get(f"strategy:{kind}") or {})
            except Exception:  # noqa: BLE001
                bucket = {}

        firsts: Dict[str, int] = {}
        unbound: Dict[str, int] = {}
        for solution in self.history:
            if solution.kind != kind or not solution.attempts:
                continue
            first = solution.attempts[0]
            firsts[first] = firsts.get(first, 0) + 1
            if len(solution.attempts) > 1:
                unbound[first] = unbound.get(first, 0) + 1
        total_first = sum(firsts.values())

        out: Dict[str, Dict[str, Any]] = {}
        for name in sorted(set(bucket) | set(firsts)):
            arm = bucket.get(name)
            trials = int(getattr(arm, "trials", 0) or 0)
            chosen = firsts.get(name, 0)
            out[name] = {
                "trials": trials,
                # A mean over no trials is not 0.0, it is absent — the distinction this package
                # keeps everywhere else.
                "mean": round(float(getattr(arm, "mean", 0.0)), 4) if trials else None,
                "first_choice_rate": (round(chosen / total_first, 4) if total_first else None),
                "bind_failure_rate": (round(unbound.get(name, 0) / chosen, 4) if chosen else None),
            }
        return out

    def advice(self, kind: str) -> Optional[str]:
        """*"How should I think about this kind of problem?"* — from the record, or not at all.

        ``None`` when nothing has been measured enough to say, and the caller is expected to omit
        the kind rather than print an empty verdict. A report that lists every kind with a blank
        beside most of them reads as coverage; this reads as what it is.
        """
        floor = int(getattr(self.meta_learner, "min_trials", 1) or 1)
        rows = {name: row for name, row in self.curve(kind).items()
                if int(row["trials"] or 0) >= floor}
        if not rows:
            return None
        ranked = sorted(rows.items(), key=lambda kv: -(kv[1]["mean"] or 0.0))
        parts: List[str] = []
        for name, row in ranked[:3]:
            note = f"{name} {row['mean']:.2f} over {row['trials']}"
            fail = row["bind_failure_rate"]
            if fail is not None and fail >= 0.5:
                note += f", but fails to bind {fail:.0%} of the time it is tried first"
            parts.append(note)
        return "; ".join(parts)

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
            # A kind nothing has been measured on is ABSENT here, not mapped to None. The old
            # shape listed all six every time and reported a winner for each, computed from a
            # kind-blind rate — six confident answers where the evidence supported none.
            "by_kind": {kind: advice for kind, advice
                        in ((k, self.advice(k)) for k in ProblemKind.ALL)
                        if advice is not None},
            "best_for": {kind: best.name for kind, best
                         in ((k, self.best_for(k)) for k in ProblemKind.ALL)
                         if best is not None},
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
                    # A strategy that is not registered *yet*. Dropping the row — which is what
                    # this did — silently loses the record of every arm registered at runtime
                    # rather than at construction, so a synthesised strategy would relearn itself
                    # from zero on every restart and never accumulate the evidence that justifies
                    # keeping it. Held instead, and applied by `register` when it arrives.
                    self._orphaned[str(name)] = dict(row)
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
