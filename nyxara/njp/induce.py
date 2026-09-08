"""NYXARA · njp/induce.py — the smallest exact rule, and the cover when one is not enough (🔎).

Written for :mod:`nyxara.njp.programming`, which learns what breaks a program by breaking one, and
lifted out here the moment :mod:`nyxara.njp.entail` needed the same thing for a different subject.
Both have the same shape of problem and it is a general one:

    Here are cases. Each was **measured** by a fixed set of generic readings, and each carries a
    label that came from the world rather than from a rule. Which readings predict which label?

Four things this does, and each of them is a correction of a way the first version got it wrong:

**Smallest first.** Width one before two before three, so a rule that holds without naming an
incidental reading is found when it exists. A rule with a passenger term is a rule about the
passenger.

**Widest at each width.** Not the first exact conjunction found — the one covering the most cases.
``argument 1 is zero and the operation splits by`` and ``argument 0 is not negative and argument 1
is zero and the operation splits by`` were both exact, and the second survived only because the
alphabet reached it first.

**Several rules per label.** A cause with two shapes has no single conjunction: an ``IndexError``
comes of an index at or past the length **and** of one below minus the length. Demanding one rule
for both finds only what they share, which is nothing that matters. The positives are covered
greedily instead, and what comes out is a disjunction of conjunctions.

**Seeded from one case, not from what all of them share.** Candidates taken from the readings every
positive agrees on can never include a reading that separates them — so a label spanning two
operations could never name either, and the best rule available was one true of half the negatives
too. The whole-group readings are tried first because a rule that holds without naming a case's
own particulars is the more general one; the seed is the fallback that lets a cover get started.

And one thing that is about arithmetic rather than about correctness: **only readings that tell the
groups apart are searched**. Sixty readings give thirty-four thousand triples and the induction
does not finish; the twelve whose values differ most between the labelled groups give two hundred
and twenty. A reading equally common either way cannot be part of an explanation, so dropping it
costs nothing.

Pure standard library. Knows nothing about programs, sentences, or anything else — a case here is
a mapping of readings and a label, and that is the entire interface.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["Case", "Rule", "AtLeast", "MISSING", "attend", "search", "cover",
           "ladder"]


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


#: What a reading a case simply does not have compares equal to: nothing.
MISSING = _Missing()


@dataclass(frozen=True)
class AtLeast:
    """A term satisfied by any value **at or above** this one, rather than equal to it.

    Every term here was an equality test until V.56, and that turned out to be the difference
    between an induction that classifies and one that can rank. Equality over bucketed values
    cannot express an order: asked which sentence of a passage holds an answer, the cover found
    ``carries is many`` — true of the right sentence and of every other sentence sharing five or
    more words with the question — and had no way to say *more than the others*. Argmax over the
    raw count, one line and no learning, beat it at every setting tried (0.470, 0.337, 0.568,
    0.570 against 0.618).

    So a reading whose values are ordered may now be tested against a threshold as well as against
    a value. Nothing else changes: a rule is still a conjunction, still carries its support and its
    counterexamples, still means what it says when it is read aloud — ``carries is at least 3``.

    Only offered for readings that are actually ordered. A bucket name, a tag, a word: those have
    no ``>=`` that means anything, and :func:`_forms` does not propose one for them.
    """

    value: Any

    def __str__(self) -> str:
        return f"at least {self.value}"


def _matches(seen: Any, want: Any) -> bool:
    """Whether one reading satisfies one term, by equality or by threshold."""
    if isinstance(want, AtLeast):
        if seen is MISSING:
            return False
        try:
            return bool(seen >= want.value)
        except TypeError:      # two values with no order between them do not compare
            return False
    return bool(seen == want)


def _forms(value: Any) -> Tuple[Any, ...]:
    """The terms a reading of this value can supply: the value, and a threshold when ordered.

    ``bool`` is excluded on purpose even though it compares: ``capitalised is at least True`` is
    the same test as ``capitalised is True`` written confusingly, and offering both would double
    the search for nothing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return (value,)
    return (value, AtLeast(value))


@dataclass(frozen=True)
class Case:
    """One measured occasion and the label the world gave it."""

    reading: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""


@dataclass
class Rule:
    """A conjunction of readings, the label it predicts, and how well it did."""

    label: str = ""
    terms: Tuple[Tuple[str, Any], ...] = ()
    support: int = 0
    counterexamples: int = 0

    @property
    def purity(self) -> float:
        """Of the cases this fires on, the share that carried its label."""
        seen = self.support + self.counterexamples
        return round(self.support / seen, 4) if seen else 0.0

    @property
    def exact(self) -> bool:
        return self.counterexamples == 0

    def holds(self, reading: Mapping[str, Any]) -> bool:
        return all(_matches(reading.get(name, MISSING), value) for name, value in self.terms)

    def render(self) -> str:
        return " and ".join(f"{name} is {value}" for name, value in self.terms) or "always"

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "when": self.render(), "support": self.support,
                "counterexamples": self.counterexamples, "purity": self.purity}


def attend(candidates: Mapping[str, Any], positives: Sequence[Mapping[str, Any]],
           negatives: Sequence[Mapping[str, Any]], keep: int) -> Dict[str, Any]:
    """The readings worth looking at: those whose value tells the two groups apart."""
    if len(candidates) <= keep:
        return dict(candidates)
    scored: List[Tuple[float, str]] = []
    for name, value in candidates.items():
        here = sum(1 for r in positives if _matches(r.get(name, MISSING), value))
        there = sum(1 for r in negatives if _matches(r.get(name, MISSING), value))
        scored.append((abs(here / max(1, len(positives)) - there / max(1, len(negatives))), name))
    scored.sort(reverse=True)
    return {name: candidates[name] for _score, name in scored[:keep]}


def _shared(readings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Readings every one of these agreed on."""
    if not readings:
        return {}
    out = dict(readings[0])
    for reading in readings[1:]:
        for name in list(out):
            if reading.get(name, MISSING) != out[name]:
                del out[name]
    return out


def search(candidates: Mapping[str, Any], positives: Sequence[Mapping[str, Any]],
           negatives: Sequence[Mapping[str, Any]], *, label: str = "", floor: int = 1,
           max_terms: int = 3, max_candidates: int = 12, purity: float = 1.0,
           exact_only: bool = True) -> Optional[Rule]:
    """The widest-covering conjunction clean enough to keep, narrowest first.

    ``purity`` is what "clean enough" means and it is a claim about the **subject**, not a knob.
    At 1.0 a rule may have no counterexample at all, which is right where the world is a decision
    procedure: an index is past the length or it is not, and Python does the same thing every time.

    Natural language is not that, and demanding it there returns nothing — measured, on 5,000
    inference pairs, **zero** rules and three near misses. Reading a hypothesis that adds
    information usually means the premise does not settle it, and *usually* is the honest word. A
    rule kept below 1.0 carries the rate it was kept at, and the school checks that rate on pairs
    it was not induced from, which is the only thing that turns a tolerance into a measurement.
    """
    picked = attend(candidates, positives, negatives, max_candidates)
    # An ordered reading offers two terms rather than one — its value, and a threshold at that
    # value. For everything else this is a one-element tuple and the search is what it always was.
    options = {name: _forms(value) for name, value in picked.items()}
    best: Optional[Rule] = None
    for width in range(1, max_terms + 1):
        widest: Optional[Rule] = None
        for names in itertools.combinations(sorted(picked), width):
            for chosen in itertools.product(*(options[name] for name in names)):
                terms = tuple(zip(names, chosen))
                rule = Rule(label=label, terms=terms)
                rule.support = sum(1 for r in positives if rule.holds(r))
                if rule.support < floor:
                    continue
                rule.counterexamples = sum(1 for r in negatives if rule.holds(r))
                if rule.purity >= purity:
                    if widest is None or (rule.support, rule.purity) > (widest.support,
                                                                        widest.purity):
                        widest = rule
                elif best is None or (rule.counterexamples, -rule.support) < (
                        best.counterexamples, -best.support):
                    best = rule
        if widest is not None:
            return widest
    return None if exact_only else best


def cover(positives: Sequence[Mapping[str, Any]], negatives: Sequence[Mapping[str, Any]], *,
          label: str = "", min_support: int = 4, min_share: float = 0.08, max_rules: int = 4,
          max_terms: int = 3, max_candidates: int = 12,
          purity: float = 1.0) -> Tuple[List[Rule], List[Rule]]:
    """Explain these positives with as few exact rules as the evidence needs, and no fewer.

    Returns ``(rules, near_misses)``. A near miss is the least-wrong conjunction found when the
    cover could go no further — kept and reported rather than rounded up into a rule, because a
    conjunction with a counterexample is not one.
    """
    rules: List[Rule] = []
    misses: List[Rule] = []
    remaining = list(positives)
    floor = max(int(min_support), int(len(positives) * float(min_share)))
    # A greedy cover that stops at the first seed it cannot explain is not a cover. The first
    # version did, and on 5,000 inference pairs it meant exactly **one** label ever got a rule:
    # `yes` was covered, `no` failed on its first seed, and the whole of `it is not possible to
    # tell` was never reached. Positives that resist explanation are set aside and the search
    # continues from the next one; only when several seeds in a row yield nothing is the label
    # genuinely out of reach of these readings, which is a finding rather than a stopping point.
    attempts = 0
    while len(remaining) >= floor and len(rules) < max_rules and attempts < max_rules * 3:
        rule = _one(remaining, negatives, label=label, floor=floor, max_terms=max_terms,
                    max_candidates=max_candidates, purity=purity)
        attempts += 1
        if rule is None:
            break
        if rule.purity < purity:
            misses.append(rule)
            covered = [r for r in remaining if rule.holds(r)]
            if not covered:
                break
            remaining = [r for r in remaining if not rule.holds(r)]
            continue
        rules.append(rule)
        remaining = [r for r in remaining if not rule.holds(r)]
    return rules, misses


def ladder(name: str, positives: Sequence[Mapping[str, Any]],
           negatives: Sequence[Mapping[str, Any]], *, label: str = "",
           floor: int = 10) -> List[Rule]:
    """Every threshold on one ordered reading, kept **because** they are redundant.

    :func:`cover` is a greedy set cover: it takes the widest clean rule, removes the positives it
    explains, and looks for another. That objective is right for saying *what a thing is* and
    wrong for saying *which of these is most*. Covering deletes exactly the graded evidence a
    ranking runs on — once ``carries_n is at least 3`` is taken, a rung at 5 explains nothing new
    and can never be induced.

    So this does not cover. It builds one rule per observed value and keeps them all, each with
    its own measured purity, and the redundancy is the whole point: a case at seven satisfies the
    rungs from zero to seven and scores seven, one at three scores three. Counting rungs *is* the
    order, recovered.

    Measured, on choosing which sentence of a passage holds an answer — 12,692 sentences, 600 held
    out:

        argmax over the raw count   0.580
        ladder of eleven rungs      0.580        <- exactly
        greedy cover                0.505

    Equal to argmax rather than better, and that is the honest headline. What it adds is not
    accuracy but *calibration*: the rungs come out monotone — 0.118 at zero, 0.445 at two, 0.621
    at three, 0.907 at six, 1.000 at ten — so a caller learns not only which case ranks highest
    but how often a case like it was right. Argmax gives an order and no confidence at all.

    Nothing here is specific to a subject; ``name`` is any reading whose values compare.
    """
    seen = sorted({r[name] for r in list(positives) + list(negatives)
                   if isinstance(r.get(name), (int, float))
                   and not isinstance(r.get(name), bool)})
    out: List[Rule] = []
    for value in seen:
        rule = Rule(label=label, terms=((name, AtLeast(value)),))
        rule.support = sum(1 for r in positives if rule.holds(r))
        if rule.support < floor:
            continue
        rule.counterexamples = sum(1 for r in negatives if rule.holds(r))
        out.append(rule)
    return out


def _one(positives: Sequence[Mapping[str, Any]], negatives: Sequence[Mapping[str, Any]], *,
         label: str, floor: int, max_terms: int, max_candidates: int,
         purity: float = 1.0) -> Optional[Rule]:
    if not positives:
        return None
    whole = search(_shared(positives), positives, negatives, label=label, floor=floor,
                   max_terms=max_terms, max_candidates=max_candidates, purity=purity)
    if whole is not None:
        return whole
    return search(dict(positives[0]), positives, negatives, label=label, floor=floor,
                  max_terms=max_terms, max_candidates=max_candidates, purity=purity,
                  exact_only=False)
