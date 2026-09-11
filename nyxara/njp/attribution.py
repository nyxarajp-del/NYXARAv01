"""NYXARA · njp/attribution.py — why did it fail, tested rather than guessed (🩺, NJP V.77→V.83).

:mod:`nyxara.njp.measurement` asks whether a number means what it looks like. This asks the next
question, and it is the one that decides what gets worked on: **an observed failure is not its own
cause**, and this repository has spent whole versions repairing the wrong thing because nobody
separated the two.

Four times, with the real cause beside the one that was assumed:

======================================  ============================  ==========================
what failed                             what was worked on             what it actually was
======================================  ============================  ==========================
entailer, 1 rule from 564,166 pairs     the readings, the corpus       the bar, and no baseline
span stage, 0.0259 exact                the ranker, two versions       142 candidates a sentence
task learner, a fifth of tasks          the induction                  90 rows and no null
native forge refusing a good kernel     nothing — it was believed      one wall-clock sample
======================================  ============================  ==========================

So a cause here is never a label. Each one is a **hypothesis with an experiment that can refute
it**, and a hypothesis whose experiment was not run is reported as untested — never as support,
and never quietly dropped so the survivors look unanimous.

**Nothing may protect its favourite explanation.** The discipline is mechanical rather than
aspirational: a cause is *supported* only when its experiment moved the number, *refuted* when its
experiment ran and did not, and where several survive the attribution says **so** instead of
picking the most interesting one. Two causes that no available experiment separates are two causes
that have not been separated, which is a more useful thing to be told than a confident wrong
answer.

**Cause of cause.** The immediate cause is rarely the one worth fixing. ``upstream`` chains one
failure to the failure behind it, so *the reader picked the wrong sentence* can lead to *the
candidate generator never proposed the answer* and then to *the clip was taken from the wrong end*
— and the chain reports where it stops rather than implying it reached bottom.

**An experiment is checked for being one experiment.** V.83 ran this organ on a real one for the
first time, and the repair it was handed — *keep twelve candidates instead of a hundred and
twenty-six* — had also thrown the right answer out of the pool for most items on its way past. The
score fell, and the verdict written down was ``budget: refuted``: *more search would not have
helped*, concluded from a run that tested nothing of the kind. So :func:`_experiment` re-reads the
ceiling after every repair, and one that moved it is ``spoiled`` rather than refuted. This module's
first rule was always that an experiment changes exactly one thing; it was checking its callers for
everything except that.

Pure standard library, and built on :mod:`nyxara.njp.measurement` rather than beside it: two of the
eight hypotheses here are answered by running a critique, because *the benchmark is wrong* and *the
measurement is noisy* are failure causes like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.measurement import Benchmark, Critique, critique

__all__ = ["Failure", "Verdict", "Attribution", "CAUSES", "REPAIRS", "attribute",
           "chain", "render_chain", "MOVED"]

#: How much an experiment has to move the score before it counts as having moved it. Below this
#: the experiment ran and found nothing, which refutes its hypothesis rather than leaving it open.
MOVED = 0.03

#: How far a repair may move the **ceiling** before it stops being the experiment it claims to be.
#: Found by running this organ on a real one for the first time (V.83): a *smaller pool* repair,
#: meant to test whether the ranker had too much to choose from, also dropped the share of items
#: whose right answer was in the pool at all from 0.700 to 0.160. It scored worse, and was written
#: down as `budget: refuted`. That verdict is not merely wrong, it is unearned — the experiment
#: changed two things, so its number cannot speak about either. See :func:`_experiment`.
SPOILED = 0.05


@dataclass
class Failure:
    """A failure described so its cause can be tested for.

    ``bench`` is the measurement that showed the failure. Everything after it is an **experiment**
    the attributor may run — each one changes exactly one thing and returns a fresh benchmark, so
    that what moved the number is what was changed. A field left empty is a hypothesis that cannot
    be tested here, and it is reported as untested rather than assumed away.
    """

    name: str = ""
    bench: Optional[Benchmark] = None
    #: The same system and readings, shown more examples.
    more_data: Optional[Callable[[], Benchmark]] = None
    #: The same data and readings, a different way of choosing among them.
    other_algorithm: Optional[Callable[[], Benchmark]] = None
    #: The same everything, allowed more search, depth or time.
    more_budget: Optional[Callable[[], Benchmark]] = None
    #: The same data and algorithm, given more or better readings of each item.
    richer_reading: Optional[Callable[[], Benchmark]] = None
    #: The failure behind this one, when there is a known one. See :func:`chain`.
    upstream: Optional["Failure"] = None
    note: str = ""


@dataclass(frozen=True)
class Verdict:
    """One hypothesis, the experiment that tested it, and what the experiment found."""

    cause: str = ""
    stands: str = "untested"       # supported | refuted | spoiled | untested
    moved: float = 0.0             # how far the experiment moved the score
    says: str = ""
    #: What would have shown this is *not* the cause. Written down whether or not it was looked
    #: for, because a hypothesis with no refuting observation is not a hypothesis.
    refuted_by: str = ""

    @property
    def tested(self) -> bool:
        """Did an experiment actually bear on this hypothesis?

        ``spoiled`` is not tested. An experiment that changed two things ran, cost the time, and
        produced a number — and that number is evidence about nothing, which is a different state
        from *refuted* and must not be allowed to pass for it.
        """
        return self.stands in ("supported", "refuted")

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "stands": self.stands, "moved": self.moved,
                "says": self.says, "refuted_by": self.refuted_by}

    def render(self) -> str:
        mark = {"supported": " ! ", "refuted": " x ", "spoiled": " ~ ", "untested": " ? "}
        return f"  {mark.get(self.stands, '   ')} {self.cause:<13} {self.says}"


@dataclass
class Attribution:
    """What the evidence says about why this failed — including when it says nothing."""

    name: str = ""
    score: float = 0.0
    verdicts: List[Verdict] = field(default_factory=list)
    critique: Optional[Critique] = None

    @property
    def supported(self) -> List[Verdict]:
        return sorted([v for v in self.verdicts if v.stands == "supported"],
                      key=lambda v: -v.moved)

    @property
    def refuted(self) -> List[str]:
        return [v.cause for v in self.verdicts if v.stands == "refuted"]

    @property
    def spoiled(self) -> List[str]:
        """Hypotheses whose experiment ran and tested nothing, because it changed two things."""
        return [v.cause for v in self.verdicts if v.stands == "spoiled"]

    @property
    def untested(self) -> List[str]:
        return [v.cause for v in self.verdicts if v.stands == "untested"]

    def _stands(self, cause: str) -> bool:
        return any(v.cause == cause and v.stands == "supported" for v in self.verdicts)

    @property
    def repairs(self) -> List[Verdict]:
        """The supported causes that name something to change, strongest first."""
        return [v for v in self.supported if v.cause in REPAIRS]

    @property
    def root(self) -> str:
        """The cause, when the evidence names one — in an order, not a contest.

        The first version ranked every supported cause by how far its experiment moved the number,
        and that was wrong twice over. A measurement wobble of 0.03 and a data repair's gain of
        0.14 are not the same quantity, so comparing them is meaningless; and a score that will not
        hold still cannot be attributed to anything at all, so it is not a competitor but a **gate**.

        1. **measurement** — the reading is moving on its own. Nothing downstream can be concluded
           yet, whatever else also looks supported.
        2. **leakage** — the score is not about the held-out world, so there is nothing to explain.
        3. the **repairs**, which do compete, because they are all measured in the same units: how
           far the number moved when that one thing was changed.
        4. **floor** — only once every repair available has been tried and none moved anything.
           Then *these levers do not reach this* is the honest finding. A **spoiled** experiment
           was not a try: it changed the problem as well as the method, so *nothing reached this*
           has not been shown and the fallback is withheld. Added in V.83, when a real organ
           produced exactly that arrangement and the old code would have concluded `floor`.

        Empty when two repairs moved it by within :data:`MOVED` of each other, because that is a
        real state of knowledge and :attr:`rivals` names them.
        """
        if self._stands("measurement"):
            return "measurement"
        if self._stands("leakage"):
            return "leakage"
        best = self.repairs
        if best:
            if len(best) > 1 and best[0].moved - best[1].moved < MOVED:
                return ""
            return best[0].cause
        if self.spoiled:
            return ""
        return "floor" if self._stands("floor") else ""

    @property
    def rivals(self) -> List[str]:
        """Repairs the evidence could not separate. Empty when :attr:`root` names one."""
        best = self.repairs
        if len(best) < 2 or self.root:
            return []
        top = best[0].moved
        return [v.cause for v in best if top - v.moved < MOVED]

    @property
    def settled(self) -> bool:
        return bool(self.root)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "score": self.score, "root": self.root,
                "rivals": self.rivals, "settled": self.settled,
                "refuted": self.refuted, "spoiled": self.spoiled, "untested": self.untested,
                "verdicts": [v.to_dict() for v in self.verdicts]}

    def render(self) -> str:
        head = f"{self.name or 'failure'}: scored {self.score:.4f}"
        body = "\n".join(v.render() for v in self.verdicts)
        if self.root:
            tail = f"  → {self.root}"
        elif self.rivals:
            tail = ("  → the evidence does not separate "
                    + " and ".join(self.rivals) + "; an experiment that does is what is missing")
        elif self.spoiled:
            tail = ("  → nothing tested here explains it, and "
                    + " and ".join(self.spoiled)
                    + " was never actually tried — its experiment moved the ceiling too")
        else:
            tail = "  → nothing tested here explains it"
        return f"{head}\n{body}\n{tail}"


# --------------------------------------------------------------------------------------------- #
#  the hypotheses
# --------------------------------------------------------------------------------------------- #
def _ceiling_of(bench: Optional[Benchmark]) -> Optional[float]:
    """What share of items have their right answer reachable at all, or ``None`` if unsaid.

    Deliberately ``None`` rather than ``1.0`` when no ``reachable`` was supplied: a ceiling nobody
    measured is not a ceiling of one, and :func:`_experiment` must be able to tell *the repair kept
    the ceiling* from *nobody looked*.
    """
    if bench is None or bench.reachable is None or not bench.items:
        return None
    return round(sum(1 for item, want in zip(bench.items, bench.gold)
                     if bench.reachable(item, want)) / len(bench.items), 4)


def _experiment(failure: Failure, make: Optional[Callable[[], Benchmark]], cause: str,
                says: str, refuted_by: str) -> Verdict:
    """Run one change, and let the number decide — once the change is shown to be one change.

    The asymmetry here is the whole discipline. An experiment that moves the score **supports** its
    hypothesis; one that runs and does not move it **refutes** it. Not "leaves it open" — a cause
    whose repair changes nothing is not the cause, and leaving it open is how a favourite
    explanation survives evidence against it.

    That asymmetry only means anything while the experiment changed **exactly one thing**, and the
    first version simply trusted the caller on that. It should not have. Run on a real organ for
    the first time in V.83, a repair labelled *a smaller pool* — twelve candidates instead of a
    hundred and twenty-six, to ask whether the ranker was drowning — also threw away the right
    answer for most items on its way past: reachable fell 0.700 → 0.160. The score went down, and
    the old code recorded ``budget: refuted``, which reads as *more search would not have helped*
    and is not what happened.

    So the ceiling is re-read on the repaired benchmark and compared to the original. Moved by
    :data:`SPOILED` or more, in **either** direction, and the verdict is ``spoiled``: the number is
    evidence about nothing. A drop means the repair took the answer away; a rise means it was a
    reachability repair wearing another label, and crediting its gain to *budget* or *data* would
    send the next version to work on the wrong thing — which is the mistake this whole module was
    written to stop, arriving one level up from where it was being watched for.
    """
    if make is None:
        return Verdict(cause=cause, says=f"no experiment supplied: {says}", refuted_by=refuted_by)
    base = failure.bench.score() if failure.bench else 0.0
    was = _ceiling_of(failure.bench)
    try:
        repaired = make()
        after = repaired.score()
    except Exception as error:  # noqa: BLE001 — an experiment that cannot run says so
        return Verdict(cause=cause, says=f"the experiment raised {type(error).__name__}: {error}",
                       refuted_by=refuted_by)
    moved = round(after - base, 4)
    now = _ceiling_of(repaired)
    if was is not None and now is not None and abs(now - was) >= SPOILED:
        return Verdict(cause=cause, stands="spoiled", moved=moved, refuted_by=refuted_by,
                       says=(f"{says}: {base:.4f} → {after:.4f} ({moved:+.4f}), but the right "
                             f"answer's reachability moved {was:.4f} → {now:.4f} — the experiment "
                             f"changed the problem as well as the method, so this tests nothing"))
    stands = "supported" if moved >= MOVED else "refuted"
    return Verdict(cause=cause, stands=stands, moved=moved, refuted_by=refuted_by,
                   says=(f"{says}: {base:.4f} → {after:.4f} ({moved:+.4f}) — "
                         + ("this is a cause" if stands == "supported"
                            else "changing it changed nothing, so it is not")))


def _measurement(failure: Failure, got: Critique) -> Verdict:
    """Is the number moving on its own? Then there is nothing here to attribute yet."""
    stability = next((f for f in got.findings if f.check == "stability"), None)
    if stability is None or not stability.informative:
        return Verdict(cause="measurement", says="the score was not re-run",
                       refuted_by="the same benchmark scoring the same twice")
    broken = stability.verdict == "broken"
    return Verdict(cause="measurement", stands="supported" if broken else "refuted",
                   moved=stability.got, refuted_by="a score that repeats between identical runs",
                   says=(f"{stability.says} — "
                         + ("the reading is moving on its own" if broken
                            else "the reading holds still, so the failure is not the measuring")))


def _leakage(failure: Failure, got: Critique) -> Verdict:
    """Was it examined on what it was taught? Then the score is not about the held-out world.

    Kept apart from :func:`_floor`, which the first version folded into it and was wrong to. They
    need opposite repairs — a leak is fixed by cutting the split differently, a floor by changing
    the system or admitting there is nothing there — and an attributor that says `benchmark` to
    both has told nobody which.
    """
    found = next((f for f in got.findings if f.check == "leakage"), None)
    if found is None or not found.informative:
        return Verdict(cause="leakage", says="no `key` and `train` supplied, so overlap is unknown",
                       refuted_by="no overlap between what was learned from and what was examined")
    stands = "supported" if found.verdict == "broken" else "refuted"
    return Verdict(cause="leakage", stands=stands, moved=found.got,
                   refuted_by="no overlap between what was learned from and what was examined",
                   says=found.says + (" — the score is not about the held-out world"
                                      if stands == "supported" else ""))


def _floor(failure: Failure, got: Critique) -> Verdict:
    """Is it merely at what a trivial strategy already gets?

    This is a **fallback**, and the ordering in :attr:`Attribution.root` treats it as one. A system
    sitting on its own floor is the failure restated, not its cause — the first version returned it
    as a cause and duly blamed `benchmark` for a task whose real fault was a chooser allowed one
    rule where six were needed. It becomes the answer only once every repair that could be tried
    has been tried and none of them moved anything, and then it says something real: *these levers
    do not reach this*.
    """
    floors = [f for f in got.ran if f.check in ("majority", "chance", "shuffled")]
    if not floors:
        return Verdict(cause="floor", says="no floor could be computed here",
                       refuted_by="scoring clear of every trivial strategy")
    guilty = [f for f in floors if f.verdict in ("broken", "close")]
    return Verdict(cause="floor", stands="supported" if guilty else "refuted",
                   moved=round(max((abs(f.margin) for f in guilty), default=0.0), 4),
                   refuted_by="scoring clear of every trivial strategy",
                   says=("; ".join(f.says for f in guilty) if guilty
                         else "it scores clear of every trivial strategy"))


def _reachability(failure: Failure, got: Critique) -> Verdict:
    """Is the right answer producible at all? No ranking lifts a ceiling."""
    ceiling = next((f for f in got.findings if f.check == "ceiling"), None)
    if ceiling is None or not ceiling.informative:
        return Verdict(cause="reachability", says="no reachability test supplied",
                       refuted_by="the right answer being producible for every item")
    stands = "supported" if ceiling.verdict == "broken" else "refuted"
    return Verdict(cause="reachability", stands=stands, moved=round(1.0 - ceiling.got, 4),
                   refuted_by="the right answer being producible for every item",
                   says=ceiling.says + (" — nothing downstream can lift this"
                                        if stands == "supported" else ""))


#: Every hypothesis, in the order they are tested. Two come from the critique, four from an
#: experiment, and the order is deliberate: a failure that is really a measurement or a benchmark
#: problem must be found **before** anything is concluded about data or algorithms, or the repair
#: goes to the wrong place. That ordering is not a nicety — it is the mistake this organ exists to
#: stop, made four times in one week.
CAUSES: Tuple[str, ...] = ("measurement", "leakage", "reachability",
                           "data", "algorithm", "reading", "budget", "floor")

#: The causes that are **repairs** — each is the thing an experiment changed. Only these compete on
#: how far they moved the number; the rest are gates or fallbacks, and comparing a gate's spread to
#: a repair's gain as though they were the same quantity is what made the first version blame the
#: measurement's 0.03 wobble and a data repair's 0.14 gain against each other.
REPAIRS: Tuple[str, ...] = ("reachability", "data", "algorithm", "reading", "budget")


def attribute(failure: Failure) -> Attribution:
    """Test every hypothesis this failure supplies the means to test, and name the ones it does not."""
    bench = failure.bench or Benchmark(name=failure.name)
    got = critique(bench)
    out = Attribution(name=failure.name or bench.name, score=bench.score(), critique=got)
    out.verdicts.append(_measurement(failure, got))
    out.verdicts.append(_leakage(failure, got))
    out.verdicts.append(_reachability(failure, got))
    out.verdicts.append(_experiment(
        failure, failure.more_data, "data", "shown more examples",
        "more examples changing nothing"))
    out.verdicts.append(_experiment(
        failure, failure.other_algorithm, "algorithm", "a different way of choosing",
        "another algorithm on the same data doing no better"))
    out.verdicts.append(_experiment(
        failure, failure.richer_reading, "reading", "given richer readings of each item",
        "better readings changing nothing"))
    out.verdicts.append(_experiment(
        failure, failure.more_budget, "budget", "allowed more search or time",
        "more search changing nothing"))
    out.verdicts.append(_floor(failure, got))
    return out


# --------------------------------------------------------------------------------------------- #
#  cause of cause
# --------------------------------------------------------------------------------------------- #
def chain(failure: Failure, *, depth: int = 6) -> List[Attribution]:
    """Walk from a failure to the failure behind it, and say where the walk stops.

    The immediate cause is rarely the one worth repairing. *The reader picked the wrong span* leads
    to *the generator never proposed the right one*, which leads to *the clip was taken from the
    wrong end*, which leads to *the same clip is correct for the other job and nobody separated
    them*. Repairing the first of those is two versions of work that moves nothing, which is what
    happened.

    The walk ends where ``upstream`` runs out, and that end is reported rather than treated as
    bedrock: a chain that stops because nobody wrote down what was behind it looks exactly like one
    that stops because nothing is.
    """
    out: List[Attribution] = []
    seen: List[int] = []
    at: Optional[Failure] = failure
    while at is not None and len(out) < max(1, depth):
        if id(at) in seen:                    # a cycle is a modelling error, not bedrock
            break
        seen.append(id(at))
        out.append(attribute(at))
        at = at.upstream
    return out


def render_chain(steps: Sequence[Attribution], deepest: Optional[Failure] = None) -> str:
    """The chain as a person reads it, ending with what is known about where it stopped."""
    lines: List[str] = []
    for i, step in enumerate(steps):
        arrow = "" if i == 0 else " " * (i * 2) + "└── "
        said = step.root or (" / ".join(step.rivals) if step.rivals else "not established")
        lines.append(f"{arrow}{step.name}: {said}")
    tail = ("  (the chain stops here because nothing upstream was recorded, which is not the same "
            "as nothing being there)")
    if deepest is not None and deepest.upstream is None:
        lines.append(tail)
    return "\n".join(lines)
