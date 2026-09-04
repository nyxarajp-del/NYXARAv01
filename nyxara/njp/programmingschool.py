"""NYXARA · njp/programmingschool.py — does knowing why it broke let her say what breaks (📐).

:mod:`nyxara.njp.programming` induces laws from things she did. This asks the only question that
settles whether they are knowledge: **on situations she has never run, can she say what will
happen before running it?**

Four numbers, and three of them exist to stop the first from flattering her:

* **held-out accuracy** — fresh situations, never trialled, predicted from the laws alone.
* **the base rate** — always guess the commonest outcome. Any learner beats an unlucky one; the
  question is by how much.
* **no laws at all** — the same reader with induction switched off. It predicts "it runs" every
  time, which is the honest floor for a thing that has watched and concluded nothing.
* **an operation held out entirely** — trained without ever performing ``lookup`` or ``divide``,
  then asked about it. This is the one that separates a law about indices and lengths from a law
  about the ``index`` operation, and it is the only measurement here she can fail while looking
  perfectly good on the other three.

And the fifth, which is not a prediction at all: **repair**. Given something that broke, does
changing one thing fix it — verified by running it again, not by asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

from nyxara.njp.programming import OPERATIONS, Operation, Programmer, Situation

__all__ = ["Paper", "Report", "examine", "transfer", "siblings", "run"]

#: 1,200 acts is where every error kind she can meet has been met often enough to induce a law
#: for it — measured: at 700 she learns four kinds, at 1,200 all six. Larger runs sharpen the
#: laws and cost minutes; this is the smallest world that contains the whole lesson.
TRAIN = 1200
TEST = 1200


@dataclass
class Paper:
    name: str = ""
    right: int = 0
    asked: int = 0
    unknown: int = 0

    @property
    def score(self) -> float:
        return round(self.right / self.asked, 4) if self.asked else 0.0

    @property
    def silence(self) -> float:
        return round(self.unknown / self.asked, 4) if self.asked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "asked": self.asked, "unknown": self.unknown}


@dataclass
class Report:
    papers: Dict[str, Paper] = field(default_factory=dict)
    laws: int = 0
    near_misses: int = 0
    seen: Dict[str, int] = field(default_factory=dict)
    repairs: Tuple[float, int] = (0.0, 0)
    rendered: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"laws": self.laws, "near_misses": self.near_misses,
                "papers": {n: p.to_dict() for n, p in sorted(self.papers.items())},
                "repairs": {"worked": self.repairs[0], "tried": self.repairs[1]},
                "seen": self.seen}

    def render(self) -> str:
        rows = [f"{self.laws} laws, {self.near_misses} near misses"]
        for name, paper in sorted(self.papers.items()):
            rows.append(f"  {name:<22} {paper.score:.3f}  ({paper.right}/{paper.asked})"
                        + (f"   unknown {paper.unknown}" if paper.unknown else ""))
        if self.repairs[1]:
            rows.append(f"  {'repair':<22} {self.repairs[0]:.3f}  "
                        f"(of {self.repairs[1]} broken situations)")
        return "\n".join(rows)


def _teach(*, seed: int = 1, train: int = TRAIN, learn: bool = True,
           operations: Sequence[Operation] = OPERATIONS, **kwargs: Any) -> Programmer:
    programmer = Programmer(seed=seed, learn=learn, **kwargs)
    programmer.experiment(train, operations=operations)
    programmer.learn()
    programmer.learn_failure()
    programmer.challenge(tries=250)
    return programmer


def _mark(programmer: Programmer, situations: Sequence[Situation]) -> Paper:
    """Predict, then run. The truth is what Python did, every time."""
    paper = Paper()
    for situation in situations:
        guess, _why = programmer.predict(situation)
        truth = programmer.run(situation).outcome
        paper.asked += 1
        if guess == "unknown":
            paper.unknown += 1
            continue
        if guess == truth:
            paper.right += 1
    return paper


def examine(seed: int = 1, *, train: int = TRAIN, test: int = TEST) -> Report:
    """The four prediction papers and the repair paper, on one training run."""
    report = Report()
    taught = _teach(seed=seed, train=train)
    report.laws = len(taught.laws)
    report.near_misses = len(taught.near_misses)
    report.seen = taught.seen()
    report.rendered = tuple(law.english() for law in taught.laws)

    # Held-out situations: a different stream, from a programmer that shares nothing but the rules.
    minter = Programmer(seed=seed + 977)
    held = minter.situations(test)
    report.papers["held_out"] = _mark(taught, held)

    # The base rate: always say whatever happens most often.
    commonest = max(taught.seen(), key=lambda k: taught.seen()[k])
    base = Paper(name="base_rate")
    for situation in held:
        base.asked += 1
        if minter.run(situation).outcome == commonest:
            base.right += 1
    report.papers["base_rate"] = base

    # No induction at all.
    blind = _teach(seed=seed, train=train, learn=False)
    report.papers["no_laws"] = _mark(blind, held)

    # Repair: only situations that actually broke, and the fix is checked by running it.
    worked = tried = 0
    for situation in held:
        if not taught.run(situation).failed:
            continue
        tried += 1
        if tried > 120:
            break
        repair = taught.repair(situation)
        if repair is None:
            continue
        worked += 1
    report.repairs = (round(worked / tried, 4) if tried else 0.0, tried)
    return report


def transfer(seed: int = 1, *, held_out: str = "lookup", train: int = TRAIN,
             test: int = 400) -> Tuple[Paper, Paper]:
    """Trained having never performed this operation, then asked about nothing else.

    Two papers, because two questions generalise differently. **Naming** the error cannot transfer
    where the name depends on the container — reading an absent name raises ``KeyError`` from a
    dict and ``AttributeError`` from an object, and no amount of watching the second teaches the
    first's name. **Whether it fails** can, and that is the one worth measuring here.
    """
    kept = tuple(o for o in OPERATIONS if o.name != held_out)
    gone = tuple(o for o in OPERATIONS if o.name == held_out)
    if not gone:
        return Paper(name=f"transfer:{held_out}"), Paper(name=f"fails:{held_out}")
    taught = _teach(seed=seed, train=train, operations=kept)
    minter = Programmer(seed=seed + 4241)
    held = minter.situations(test, operations=gone)
    named = _mark(taught, held)
    coarse = Paper(name=f"fails:{held_out}")
    for situation in held:
        coarse.asked += 1
        guess, _why = taught.will_fail(situation)
        truth = minter.run(situation).failed
        if guess is None:
            coarse.unknown += 1
            continue
        if guess == truth:
            coarse.right += 1
    return named, coarse


def siblings(name: str) -> int:
    """How many *other* operations share a trait with this one.

    The transfer numbers are not a mystery and this is why: a law about ``the operation takes a
    name`` can only be learned if something else in the world takes one. ``lookup`` has ``attribute``
    for a sibling and ``divide`` has ``modulo``, and holding either out still leaves the trait
    demonstrated — she sees the held-out operation fail at 0.92 and 0.91. ``to_int`` is the only
    thing that converts and ``index`` the only thing that takes a position, so holding *those* out
    deletes the only evidence for the trait, and there is nothing left to transfer from.
    """
    mine = {t for o in OPERATIONS if o.name == name for t in o.traits}
    # Traits most of the world has are background, not evidence: ``needs_container`` is on eight of
    # the fifteen operations and tells nothing apart, so counting it made ``index`` look like it
    # had seven siblings when the trait that matters — ``takes_position`` — has none. Counted this
    # way the four numbers line up with the four transfer scores exactly.
    common = {t for t in mine
              if sum(1 for o in OPERATIONS if t in o.traits) > len(OPERATIONS) / 2}
    mine -= common
    return sum(1 for o in OPERATIONS if o.name != name and mine & set(o.traits))


def run(seed: int = 1) -> Dict[str, Any]:
    report = examine(seed)
    out = report.to_dict()
    out["transfer"] = {}
    for name in ("lookup", "divide", "to_int", "index"):
        named, coarse = transfer(seed, held_out=name)
        out["transfer"][name] = {"names_it": named.to_dict(),
                                 "sees_it_fail": coarse.to_dict(),
                                 "trait_siblings": siblings(name)}
    return out


def main() -> None:  # pragma: no cover — a report, not a test
    report = examine()
    print(report.render())
    print("\nwhat she worked out:")
    for line in report.rendered:
        print("   ", line)
    print("\ntransfer — trained without ever doing it, then asked about nothing else:")
    print(f"    {'operation':<12}{'names the error':>18}{'sees it fail':>16}"
          f"{'trait siblings':>17}")
    for name in ("lookup", "divide", "to_int", "index"):
        named, coarse = transfer(held_out=name)
        print(f"    {name:<12}{named.score:>18.3f}{coarse.score:>16.3f}"
              f"{siblings(name):>17}")


if __name__ == "__main__":  # pragma: no cover
    main()
