"""NYXARA · njp/latent.py — what varies that nobody is varying (🔍, NJP V.86).

V.85 can answer *do two known interventions interact*. It cannot answer the question underneath,
and neither can anything above it:

    what should count as a possible intervention in the first place?

Every version from V.83 to V.85 has carried one assumption without once testing it: **that the
right causal variables are already in the vocabulary**. Nine single-variable experiments and a
pairwise search all draw from a list somebody wrote down. If the cause is not on that list, the
whole apparatus is a very careful way of not finding it.

So this looks somewhere else. Across a set of items, some fail and some do not, and the items
differ in many ways nobody chose — span length, answer position, how much the question and the
sentence share, how many clauses there are. Those are **varying and uncontrolled**, and the gap
between *what varies naturally* and *what the experimenter manipulates* is where an unlisted cause
has to be hiding if it is hiding anywhere.

**And finding one there proves almost nothing, which is the other half of this module.**

A property that separates the failures from the successes is *observational* evidence and nothing
more. This repository can already name three ways that is not a cause:

======================  ====================================================================
what it might be        what settles it
======================  ====================================================================
a confound              something else moves both; ``do(Z)`` alone then changes nothing
a consequence           the failure produced *it*; ``do(Z)`` changes nothing either
a coincidence           a permutation null it does not clear
======================  ====================================================================

So the module keeps :class:`Evidence` as a **type**, on a ladder, and refuses to let a claim stand
above what it actually holds. *Z predicts failure* and *changing Z changes failure* are different
sentences with different licences, and quietly promoting the first to the second is the single
cheapest way a system that improves itself can come to believe something false — cheaper than a
bad experiment, because it costs nothing and leaves no trace.

One thing it cannot do, said here rather than in a footnote: :func:`put_to_the_test` refuses an
intervention that disturbs another trait, and cannot tell a trait **downstream** of the one being
changed from one that moves it back. On a world where the cause's own consequences are recomputed,
it therefore refuses the cause. Fixing that needs an ordering over the traits, which is domain
knowledge like every other input here.

Two gates before any of that. **A constant cannot be a cause**: a property that does not vary
across the items cannot explain why some of them failed, and is refused before it is tested rather
than after. And a property something already manipulates is not a discovery — it is the vocabulary
that was already there, which is exactly what this module exists to look past.

Pure standard library.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.space import Intervention

__all__ = ["LADDER", "Trait", "Standing", "Candidate", "uncontrolled", "sift", "put_to_the_test",
           "SEPARATES", "DRAWS", "LUCK", "FLAT", "SEED"]

#: Kinds of evidence, weakest first. The ladder is the module's spine: each rung is licensed by a
#: **different experiment**, and nothing on it can be reached by reasoning from the rung below.
#:
#: * ``observational`` — Z tells the failures apart from the successes.
#: * ``interventional`` — changing Z, and only Z, changes the failures.
#: * ``interaction`` — Z does something alongside something else that neither does alone.
#: * ``mechanistic`` — an account of *how* that predicted something before it was measured.
#: * ``engineering`` — a repair built on it survived a gate.
#: * ``transfer`` — it held somewhere it was not found.
LADDER: Tuple[str, ...] = ("observational", "interventional", "interaction",
                           "mechanistic", "engineering", "transfer")

#: How far apart the failures and the successes must sit, in pooled spreads, before a property is
#: worth a null at all. Below this the property does not separate them and no amount of resampling
#: will make it.
SEPARATES = 0.30

#: Permutation draws for the null. The labels are shuffled, not the property — the question is
#: whether *this* property would separate *any* split of these items this well.
DRAWS = 200

#: How often a shuffled split may separate as well before the separation is luck.
LUCK = 0.10

#: Below this spread a property is constant across the items, and a constant cannot be a cause.
FLAT = 1e-6

SEED = 86


@dataclass(frozen=True)
class Trait:
    """One property of an item that somebody can measure, and possibly change.

    ``read`` is required and ``change`` is not, and the asymmetry is the point: a great many things
    about an item can be *measured* and only some of them can be *manipulated*. A trait with no
    ``change`` can reach ``observational`` and stop there, and the module says so rather than
    letting the strongest available evidence pass for the strongest possible.
    """

    name: str = ""
    #: Measure it on one item.
    read: Optional[Callable[[Any], float]] = None
    #: Make it different on one item, leaving everything else about the item alone. Without this
    #: no interventional evidence about this trait is obtainable at all.
    change: Optional[Callable[[Any], Any]] = None
    says: str = ""

    @property
    def manipulable(self) -> bool:
        return self.change is not None

    def of(self, items: Sequence[Any]) -> List[float]:
        return [float(self.read(item)) for item in items] if self.read else []


@dataclass(frozen=True)
class Standing:
    """One piece of evidence, of one kind, and what produced it."""

    kind: str = ""
    got: float = 0.0
    says: str = ""

    @property
    def rung(self) -> int:
        return LADDER.index(self.kind) if self.kind in LADDER else -1


@dataclass
class Candidate:
    """A trait that might be a cause, with every rung it has actually reached.

    :attr:`stands_as` is the highest kind of evidence **held**, never the highest kind hoped for,
    and :meth:`claims` refuses anything above it. That refusal is the module's whole contribution
    to safety: a system that can promote *Z predicts failure* into *Z causes failure* by wanting to
    will eventually do so, and it will not notice.
    """

    trait: str = ""
    spread: float = 0.0
    separates: float = 0.0
    luck: float = 1.0
    manipulable: bool = False
    held: List[Standing] = field(default_factory=list)
    #: Why it was refused before any test, when it was.
    refused: str = ""

    @property
    def stands_as(self) -> str:
        """The strongest kind of evidence actually in hand, or ``""``."""
        best = max(self.held, key=lambda e: e.rung, default=None)
        return best.kind if best is not None else ""

    def claims(self, kind: str) -> bool:
        """May this candidate be spoken of at ``kind``? Only if something licensed that rung."""
        if kind not in LADDER:
            return False
        return any(e.kind == kind for e in self.held)

    @property
    def causal(self) -> bool:
        """Never true on observational evidence, however large the separation."""
        return self.claims("interventional")

    def to_dict(self) -> Dict[str, Any]:
        return {"trait": self.trait, "spread": self.spread, "separates": self.separates,
                "luck": self.luck, "manipulable": self.manipulable, "refused": self.refused,
                "stands_as": self.stands_as, "causal": self.causal,
                "held": [{"kind": e.kind, "got": e.got, "says": e.says} for e in self.held]}

    def render(self) -> str:
        if self.refused:
            return f"   -  {self.trait:<22} {self.refused}"
        mark = {"": " x ", "observational": " ? ", "interventional": " ! "}
        return (f"  {mark.get(self.stands_as, ' ! ')} {self.trait:<22} "
                + "; ".join(e.says for e in self.held or [Standing(says="nothing stands")]))


def uncontrolled(traits: Sequence[Trait], interventions: Sequence[Intervention],
                 items: Sequence[Any]) -> List[Trait]:
    """Traits that vary across the items and that **no intervention touches**.

    The bridge from choosing an experiment to inventing one. A trait already manipulated by
    something on the shelf is not a discovery — it is the vocabulary that was there all along, and
    reporting it would let the module score well by listing its own inputs back.

    Matching is by name against ``Intervention.on``, which is a real limitation and is stated
    rather than hidden: two names for one knob will be missed, and this cannot tell that `pool size`
    and `how many candidates compete` are the same thing. Deciding that is what a shared vocabulary
    is for, and inventing one here would be guessing.
    """
    touched = {what.on for what in interventions}
    out: List[Trait] = []
    for trait in traits:
        if trait.name in touched:
            continue
        values = trait.of(items)
        if values and pstdev(values) > FLAT:      # a constant is not a blind spot, it is a constant
            out.append(trait)
    return out


def _separation(values: Sequence[float], failed: Sequence[bool]) -> float:
    """How far apart the two groups sit, in pooled spreads. Zero when either group is empty."""
    bad = [v for v, f in zip(values, failed) if f]
    good = [v for v, f in zip(values, failed) if not f]
    if not bad or not good:
        return 0.0
    spread = pstdev(list(values))
    if spread <= FLAT:
        return 0.0
    return round(abs(mean(bad) - mean(good)) / spread, 4)


def sift(traits: Sequence[Trait], items: Sequence[Any], failed: Sequence[bool], *,
         draws: int = DRAWS, rng: Optional[random.Random] = None) -> List[Candidate]:
    """Which traits tell the failures from the successes — and nothing stronger than that.

    Every survivor comes back holding **observational** evidence only. That is not a limitation of
    this function, it is what the data can license: no item was changed, so nothing here can speak
    about what would happen if one were. :func:`put_to_the_test` is where the next rung is earned.
    """
    rng = rng or random.Random(SEED)
    out: List[Candidate] = []
    for trait in traits:
        values = trait.of(items)
        if not values:
            out.append(Candidate(trait=trait.name, refused="it cannot be read on these items"))
            continue
        spread = round(pstdev(values), 4)
        got = Candidate(trait=trait.name, spread=spread, manipulable=trait.manipulable)
        if spread <= FLAT:
            # Refused before it is tested, not after. A property that is the same on the items that
            # failed and the items that did not cannot be why they differ, and putting it through a
            # null anyway invites a coincidence to speak for it.
            got.refused = f"it does not vary across these items (spread {spread:.4f})"
            out.append(got)
            continue
        got.separates = _separation(values, failed)
        if got.separates < SEPARATES:
            got.luck = 1.0
            out.append(got)
            continue
        beat = sum(1 for _ in range(max(1, draws))
                   if _separation(values, rng.sample(list(failed), len(failed)))
                   >= got.separates)
        got.luck = round((beat + 1) / (max(1, draws) + 1), 4)
        if got.luck <= LUCK:
            got.held.append(Standing(
                kind="observational", got=got.separates,
                says=(f"the failures sit {got.separates:.4f} pooled spreads from the successes; "
                      f"{beat} of {draws} shuffled splits did as well, so luck explains this "
                      f"{got.luck:.4f} of the time — it **predicts**, which is not **causes**")))
        out.append(got)
    return sorted(out, key=lambda c: (not c.held, -c.separates))


def put_to_the_test(candidate: Candidate, trait: Trait, items: Sequence[Any], *,
                    outcome: Callable[[Sequence[Any]], Sequence[bool]],
                    others: Sequence[Trait] = (), slack: float = 0.05,
                    moved: float = 0.05) -> Candidate:
    """Change the trait and nothing else, and see whether the failures move.

    This is the only thing in the module that can license ``interventional``, and it earns the rung
    the hard way: every other trait supplied in ``others`` is re-read afterwards, and if one of them
    moved too then the intervention changed more than the trait and identifies nothing. That is
    V.83's ``spoiled`` and V.85's apparatus control arriving a third time, at the level of the
    variable rather than the experiment or the harness.

    It is also the test a **confound** and a **consequence** fail. Something that merely travels
    with the failures — because a third thing moves both, or because the failure produced it — goes
    on separating them beautifully and stops mattering the moment it is moved on its own.

    **A limitation, stated rather than discovered later.** The drift check cannot tell *moved
    something downstream of this trait* from *moved something that also moves this trait*. The first
    is the causal chain doing exactly what a cause does; the second is what makes an experiment
    identify nothing. Separating them needs an ordering over the traits, and nobody supplied one —
    so on a world where changing the real cause also updates its own consequences, this refuses the
    real cause. That is a wrong answer of a known shape, pinned by a test rather than left to be
    found later, and the honest fix is a supplied ordering, not a cleverer statistic.
    """
    if trait.change is None:
        candidate.held.append(Standing(
            kind="", says="no way to change it was supplied, so it stops at what it predicts"))
        return candidate
    before = list(outcome(items))
    changed = [trait.change(item) for item in items]
    after = list(outcome(changed))
    drifted = [o.name for o in others
               if o.read is not None and o.name != trait.name
               and o.of(items) and pstdev(o.of(items)) > FLAT
               and abs(mean(o.of(changed)) - mean(o.of(items)))
               / max(FLAT, pstdev(o.of(items))) >= slack]
    if drifted:
        candidate.refused = (f"changing it also moved {', '.join(drifted)}, so what happened "
                             f"cannot be put down to it")
        return candidate
    shift = round(sum(after) / max(1, len(after)) - sum(before) / max(1, len(before)), 4)
    if abs(shift) >= moved:
        candidate.held.append(Standing(
            kind="interventional", got=shift,
            says=(f"changing it, and nothing else, moved the failures {shift:+.4f} — "
                  f"this one **causes**")))
    else:
        candidate.held.append(Standing(
            kind="", got=shift,
            says=(f"changing it, and nothing else, moved the failures {shift:+.4f} — "
                  f"it travels with them and does not cause them")))
    return candidate
