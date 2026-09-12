"""NYXARA · njp/measurement.py — is the number measuring what she thinks it is (📐, NJP V.74).

Four times in one week a number in this package was read as a fact about a mechanism and turned out
to be a fact about the measurement. Every one of them looked like a hard result:

===========================  ==============================  ==================================
what the number said         what was concluded              what was actually wrong
===========================  ==============================  ==================================
1 rule from 564,166 pairs    "the entailer cannot learn"     the purity bar was above the
                                                             subject; nothing was compared
0.0259 exact given the       "finding a span is hard"        the span stage had no baseline at
right sentence                                               all; it beats every one there is
0.23x on a C kernel          "the kernel is slow"            one wall-clock sample decided a
                                                             correctness gate
0.318 of tasks beat their    "she learned a third of them"   shuffled answers beat their floor
own majority                                                 0.207 of the time
===========================  ==============================  ==================================

The common shape is not carelessness. In each case the mechanism was measured carefully and the
**measurement was not measured at all** — so the number was believed, acted on, and in two cases
optimised against for a whole version before anybody asked what it was being compared to.

    A system cannot understand its own performance without modelling the measurement process.

So this organ takes a benchmark as an *object* rather than a score, and runs against it the
strategies that know nothing. It does not read code and it does not reason about intent: it runs
things and reports what they got. Seven checks, one per way a number lied here:

* **majority** — what does always saying the commonest answer get? (the task learner's floor)
* **chance** — what does choosing uniformly at random get? (the span stage's floor)
* **shuffled** — retrained on the same items with the answers permuted, what does the *same
  learner* get? Everything about the task survives that except the thing being learned.
* **leakage** — do the items it was examined on share identity with the ones it learned from?
  (SQuAD asks a dozen questions per paragraph; a row-wise cut put the same passage on both sides
  and inflated every V.55 number three- to fourfold)
* **abstention** — is the headline mixing how *often* it answers with how *right* it is? At a
  purity bar of 0.72 the entailer was right 78% of the time on one pair in twenty.
* **stability** — run it again. How far does the score move? (the forge's 39x-to-77x spread)
* **ceiling** — is the right answer reachable at all? (41% of gold answers were never proposed as
  candidates, and three parameter sweeps ran before anybody checked)

**What it refuses to do.** A check it cannot run is reported as ``not checked``, never as passed.
That distinction is the whole point: this package has twice recorded a mechanism as flawless when
it had simply never fired, and a critic that quietly skips what it cannot see would be the same
mistake one level up.

Pure standard library.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Benchmark", "Finding", "Critique", "CHECKS", "critique", "SEED"]

SEED = 74

#: Every check this organ knows how to run, in the order it runs them. Named so a critique can say
#: which of them it could not run rather than returning a shorter list and looking clean.
CHECKS: Tuple[str, ...] = ("majority", "chance", "shuffled", "leakage",
                           "abstention", "stability", "ceiling")

#: How much a trivial strategy has to come within, of the system under test, before the benchmark
#: is called into question. Not a claim about what is *good* — a claim about what is
#: **distinguishable**. A system two points above always-say-the-commonest is not necessarily bad;
#: it is un-evidenced by this benchmark, which is a different and more useful thing to be told.
CLOSE = 0.02

#: How far a score may move between identical runs before the measurement is the finding. Below
#: this a re-run is noise; above it, the number is not a property of the system.
WOBBLE = 0.02

#: How many times a stability check re-runs, and how many draws a chance estimate averages.
ROUNDS = 5

#: How many permutations the shuffled-answer null draws.
#:
#: One was the first version, and one draw of a null is **exactly the error this organ exists to
#: catch** — a single sample of a noisy quantity, believed. It failed its own retrodiction exam on
#: that: the task learner at 90 rows a task scored 0.536 and one shuffle scored 0.429, which looks
#: like a comfortable margin and is one coin landing. What the question actually is — *could a
#: learner that found nothing have scored this?* — is answered by a permutation test, so the null
#: is drawn many times and what is reported is how often it reached the system.
PERMUTATIONS = 20

#: How often the shuffled null may reach the system's score before the result is indistinguishable
#: from having learned nothing. The conventional 0.05 would be a claim about a significance
#: convention; this is a claim about how much luck is being tolerated, which is what it is.
LUCK = 0.10


@dataclass
class Benchmark:
    """A measurement described as data, so that it can be examined instead of trusted.

    Only ``items``, ``gold`` and ``predict`` are required, and with only those three the critique
    runs two of its seven checks and says so. Each further field unlocks a check that cannot be
    faked without it — which is deliberate: the critic must never infer a passing verdict from a
    field nobody supplied.
    """

    name: str = ""
    #: What the system was asked.
    items: Sequence[Any] = ()
    #: What each answer should have been, in the same order.
    gold: Sequence[Any] = ()
    #: The system under test.
    predict: Optional[Callable[[Any], Any]] = None
    #: Whether the system answered at all, rather than abstaining. Without it, abstention cannot
    #: be told from a wrong answer and the abstention check is not run.
    spoke: Optional[Callable[[Any], bool]] = None
    #: Re-teach the system from these (item, gold) pairs and return something to predict with.
    #: Required by the shuffled-answer null, which is the only check that can say whether the
    #: system learned the task or the task's shape.
    learn: Optional[Callable[[Sequence[Any], Sequence[Any]], Callable[[Any], Any]]] = None
    #: What the system learned from, and what makes two items *the same thing*. Both are needed
    #: for leakage: identity is domain knowledge and cannot be guessed from the items.
    train: Sequence[Any] = ()
    key: Optional[Callable[[Any], Any]] = None
    #: Whether the right answer is reachable for this item at all — a candidate generator that
    #: never proposes it puts a ceiling under every score, and no ranking can lift it.
    reachable: Optional[Callable[[Any, Any], bool]] = None
    #: How an answer is compared to gold. Exact match unless the subject says otherwise.
    same: Optional[Callable[[Any, Any], bool]] = None

    def matches(self, said: Any, want: Any) -> bool:
        return self.same(said, want) if self.same else said == want

    def score(self, predict: Optional[Callable[[Any], Any]] = None) -> float:
        """What a way of answering gets on these items. Abstention counts as a miss."""
        fn = predict or self.predict
        if fn is None or not self.items:
            return 0.0
        right = sum(1 for item, want in zip(self.items, self.gold)
                    if self.matches(fn(item), want))
        return round(right / len(self.items), 4)


@dataclass(frozen=True)
class Finding:
    """One check, what it measured, and what that says about the benchmark.

    ``verdict`` is about the **measurement**, never about the system. ``broken`` does not mean the
    system is bad — it means this benchmark cannot tell whether it is.
    """

    check: str = ""
    verdict: str = "not checked"        # clear | close | broken | not checked
    got: float = 0.0                    # what the trivial strategy, or the re-run, got
    against: float = 0.0                # what the system under test got
    says: str = ""

    @property
    def margin(self) -> float:
        return round(self.against - self.got, 4)

    @property
    def informative(self) -> bool:
        """Did this check actually run? A skipped check is not a passed one."""
        return self.verdict != "not checked"

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "verdict": self.verdict, "got": self.got,
                "against": self.against, "margin": self.margin, "says": self.says}

    def render(self) -> str:
        mark = {"clear": "ok ", "close": " ~ ", "broken": " ! ", "not checked": " ? "}
        return f"  {mark.get(self.verdict, '   ')} {self.check:<11} {self.says}"


@dataclass
class Critique:
    """What a benchmark is worth, which is not the same question as what the system scored."""

    name: str = ""
    system: float = 0.0
    findings: List[Finding] = field(default_factory=list)

    @property
    def ran(self) -> List[Finding]:
        return [f for f in self.findings if f.informative]

    @property
    def unchecked(self) -> List[str]:
        return [f.check for f in self.findings if not f.informative]

    @property
    def broken(self) -> List[Finding]:
        return [f for f in self.findings if f.verdict == "broken"]

    @property
    def close(self) -> List[Finding]:
        return [f for f in self.findings if f.verdict == "close"]

    @property
    def trusted(self) -> bool:
        """Whether the number means what it appears to mean.

        Three conditions, and the third is the one that stops this being a rubber stamp: nothing
        broken, nothing merely close, and **more than half the checks actually ran**. A benchmark
        that supplied only the three required fields cannot be trusted on the strength of the two
        checks that were possible.
        """
        return (not self.broken and not self.close
                and len(self.ran) > len(self.findings) // 2)

    @property
    def headline(self) -> float:
        """What is left of the score once the best trivial strategy is subtracted."""
        floors = [f.got for f in self.ran if f.check in ("majority", "chance", "shuffled")]
        return round(self.system - max(floors), 4) if floors else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "system": self.system, "above_trivial": self.headline,
                "trusted": self.trusted, "broken": [f.check for f in self.broken],
                "close": [f.check for f in self.close], "not_checked": self.unchecked,
                "findings": [f.to_dict() for f in self.findings]}

    def render(self) -> str:
        head = (f"{self.name or 'benchmark'}: scored {self.system:.4f}, "
                f"{self.headline:+.4f} above the best trivial strategy")
        body = "\n".join(f.render() for f in self.findings)
        tail = ("  → the number means what it looks like" if self.trusted else
                "  → this benchmark cannot support that number")
        return f"{head}\n{body}\n{tail}"


# --------------------------------------------------------------------------------------------- #
#  the checks
# --------------------------------------------------------------------------------------------- #
def _verdict(system: float, trivial: float) -> str:
    if trivial > system:
        return "broken"
    return "close" if system - trivial <= CLOSE else "clear"


def _majority(bench: Benchmark, system: float) -> Finding:
    """Always the commonest answer. The floor almost every classification score is quoted above."""
    seen = Counter(map(repr, bench.gold))
    if not seen:
        return Finding(check="majority", says="no gold answers to count")
    commonest = seen.most_common(1)[0]
    got = round(commonest[1] / len(bench.gold), 4)
    return Finding(check="majority", verdict=_verdict(system, got), got=got, against=system,
                   says=(f"always saying the commonest answer scores {got:.4f}; "
                         f"the system scores {system:.4f} ({system - got:+.4f})"))


def _chance(bench: Benchmark, system: float) -> Finding:
    """Choosing uniformly from the answers that occur. The floor a ranking is quoted above."""
    pool = list(bench.gold)
    if not pool:
        return Finding(check="chance", says="no gold answers to draw from")
    rng = random.Random(SEED)
    draws = [round(sum(1 for want in bench.gold if bench.matches(rng.choice(pool), want))
                   / len(bench.gold), 4) for _ in range(ROUNDS)]
    got = round(statistics.mean(draws), 4)
    return Finding(check="chance", verdict=_verdict(system, got), got=got, against=system,
                   says=(f"drawing an answer at random scores {got:.4f}; "
                         f"the system scores {system:.4f} ({system - got:+.4f})"))


def _shuffled(bench: Benchmark, system: float) -> Finding:
    """The same learner on the same items, with the answers permuted — many times over.

    Size, answer space and skew all survive a permutation; the relation between an item and its
    answer does not. So a learner retaught on shuffled answers has, by construction, learned
    nothing, and whatever it still scores is what its machinery gets for free.

    Drawn **once**, that number is a coin landing and tells you very little; drawn many times it
    answers the question actually being asked — *could a learner that found nothing have reached
    this score?* — as the share of permutations that did. This organ got that wrong in its own
    first version and failed its own retrodiction exam on it, which is recorded in
    :data:`PERMUTATIONS` rather than quietly corrected.
    """
    if bench.learn is None or not bench.train:
        return Finding(check="shuffled",
                       says="no `learn` and `train` given, so the null could not be run")
    want = list(bench.gold[:len(bench.train)]) or list(bench.gold)
    rng = random.Random(SEED)
    scores: List[float] = []
    for _ in range(PERMUTATIONS):
        order = list(want)
        rng.shuffle(order)
        try:
            scores.append(bench.score(bench.learn(bench.train, order)))
        except Exception as error:  # noqa: BLE001 — a learner that cannot be retaught says so
            return Finding(check="shuffled",
                           says=f"re-teaching raised {type(error).__name__}: {error}")
    if not scores:
        return Finding(check="shuffled", says="the null produced no scores")
    reached = sum(1 for s in scores if s >= system)
    # +1 to both, the standard correction: with no permutation reaching it, the honest statement is
    # "fewer than one in twenty-one", not "never".
    luck = round((reached + 1) / (len(scores) + 1), 4)
    got = round(statistics.mean(scores), 4)
    verdict = "broken" if luck > LUCK else ("close" if luck > LUCK / 2 else "clear")
    return Finding(check="shuffled", verdict=verdict, got=got, against=system,
                   says=(f"{len(scores)} shuffles of the answers average {got:.4f} "
                         f"(up to {max(scores):.4f}) against the system's {system:.4f}; "
                         f"{reached} of them reached it, so luck alone explains this "
                         f"{luck:.4f} of the time"))


def _leakage(bench: Benchmark, system: float) -> Finding:
    """Did it see the examined items while learning?

    Identity is the domain's business, not this organ's, which is why `key` has to be supplied.
    SQuAD asks a dozen questions about one paragraph: cut row-wise, the same passage lands on both
    sides, and every number taken that way was inflated three- to fourfold.
    """
    if bench.key is None or not bench.train:
        return Finding(check="leakage", says="no `key` and `train` given, so overlap is unknown")
    learned = {bench.key(x) for x in bench.train}
    shared = sum(1 for item in bench.items if bench.key(item) in learned)
    got = round(shared / len(bench.items), 4) if bench.items else 0.0
    return Finding(check="leakage", verdict="broken" if got > 0 else "clear",
                   got=got, against=system,
                   says=(f"{shared:,} of {len(bench.items):,} examined items ({got:.4f}) were also "
                         f"learned from" if shared else
                         f"none of {len(bench.items):,} examined items were learned from"))


def _abstention(bench: Benchmark, system: float) -> Finding:
    """Is the headline mixing how often it answers with how right it is when it does?

    Two systems with the same score can be a confident one that is often wrong and a careful one
    that rarely speaks, and the repairs they need are opposite. At a purity bar of 0.72 the
    entailer was right 78% of the time on one pair in twenty, and its single reported number said
    neither of those things.
    """
    if bench.spoke is None or bench.predict is None:
        return Finding(check="abstention", says="no `spoke` given, so silence looks like a wrong "
                                                "answer and cannot be told from one")
    spoke = right = 0
    for item, want in zip(bench.items, bench.gold):
        if not bench.spoke(item):
            continue
        spoke += 1
        right += int(bench.matches(bench.predict(item), want))
    reach = round(spoke / len(bench.items), 4) if bench.items else 0.0
    precision = round(right / spoke, 4) if spoke else 0.0
    # Not "broken" — a low-reach system is a real design choice. It is broken only as a *headline*,
    # which is what a single number makes of it.
    verdict = "close" if reach < 0.5 else "clear"
    return Finding(check="abstention", verdict=verdict, got=reach, against=precision,
                   says=(f"it answers {reach:.4f} of the items and is right {precision:.4f} of the "
                         f"time when it does; the single figure {system:.4f} is neither"))


def _stability(bench: Benchmark, system: float) -> Finding:
    """Run it again. A score that moves between identical runs is not a property of the system.

    The one that made this a check: a C kernel measured 39x to 77x faster across ten forges of the
    same source, and one sample of that spread decided whether it was kept.
    """
    if bench.predict is None or not bench.items:
        return Finding(check="stability", says="nothing to re-run")
    runs = [bench.score() for _ in range(ROUNDS)]
    spread = round(max(runs) - min(runs), 4)
    return Finding(check="stability", verdict="broken" if spread > WOBBLE else "clear",
                   got=spread, against=system,
                   says=(f"{ROUNDS} identical runs span {spread:.4f} "
                         f"({min(runs):.4f} to {max(runs):.4f})"))


def _ceiling(bench: Benchmark, system: float) -> Finding:
    """Can the right answer be produced at all?

    A candidate the generator never proposes cannot be chosen however good the ranking is, and a
    score read without this looks like a ranking problem when it is a reachability one. Measured
    once here already: 41% of gold answers were never proposed, and three parameter sweeps ran
    before the decomposition found it.
    """
    if bench.reachable is None:
        return Finding(check="ceiling", says="no `reachable` given, so the ceiling is unknown")
    got = round(sum(1 for item, want in zip(bench.items, bench.gold)
                    if bench.reachable(item, want)) / len(bench.items), 4) if bench.items else 0.0
    # The ceiling is not a floor: it is broken when the score is near a ceiling well under 1, since
    # then the mechanism being tuned is not the one holding the number down.
    verdict = "broken" if got < 0.999 and system >= got - CLOSE else "clear"
    return Finding(check="ceiling", verdict=verdict, got=got, against=system,
                   says=(f"the right answer is reachable for {got:.4f} of items; "
                         f"the system scores {system:.4f}"
                         + (" — it is already at the ceiling, and the ceiling is what to raise"
                            if verdict == "broken" else "")))


_RUNNERS: Dict[str, Callable[[Benchmark, float], Finding]] = {
    "majority": _majority, "chance": _chance, "shuffled": _shuffled, "leakage": _leakage,
    "abstention": _abstention, "stability": _stability, "ceiling": _ceiling,
}


def critique(bench: Benchmark, *, checks: Sequence[str] = CHECKS) -> Critique:
    """Run every check this benchmark supplies the means for, and name the ones it does not."""
    system = bench.score()
    out = Critique(name=bench.name, system=system)
    for name in checks:
        runner = _RUNNERS.get(name)
        if runner is None:
            out.findings.append(Finding(check=name, says="no such check"))
            continue
        try:
            out.findings.append(runner(bench, system))
        except Exception as error:  # noqa: BLE001 — a check that cannot run says so
            out.findings.append(Finding(check=name,
                                        says=f"raised {type(error).__name__}: {error}"))
    return out
