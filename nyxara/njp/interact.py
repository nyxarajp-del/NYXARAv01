"""NYXARA · njp/interact.py — what two changes do together that neither does alone (🔗, NJP V.85).

Four interventions on a real organ, four flat results, and a map saying the space is not exhausted.
That is the shape a system takes when nothing is wrong with any one part of it:

    f(A) ≈ 0        f(B) ≈ 0        f(A, B) ≫ 0

A single-variable search cannot see that, ever, however many variables it tries — and V.83 and V.84
between them ran nine single-variable experiments on one organ and refuted nine hypotheses. So this
measures what two changes do **together**, minus what each does alone:

    I(A, B) = ΔAB − ΔA − ΔB

**And then it spends almost all of its effort trying to show that number is nothing.** That ordering
is deliberate. A first attempt at this would compute the subtraction, find it large, and report a
discovery — and this repository already knows what that is worth, because `measurement._shuffled`
drew its null once and failed its own retrodiction for exactly the same reason. An interaction has
three ways of being an artefact and each gets a control:

======================  ====================================================================
what it might be        the control
======================  ====================================================================
noise                   a bootstrap over items; the interval must not straddle zero
a pipeline order effect ``A then B`` against ``B then A``; a difference means composing mutates
a stateful harness      run a **change that does nothing**; it must move nothing, alone or beside
======================  ====================================================================

The third is the cheapest and the most damning. If applying *nothing* moves the number, the
apparatus is carrying state between runs and every figure it has ever produced is suspect —
including the flat ones that were believed.

It is also where the first version of this module was wrong, and the exam caught it. That control
was written only in the interaction form, ``I(A, nothing)`` — and a leak growing linearly with how
often the harness has been touched cancels out of that subtraction **exactly**. A fixture whose
every reading was inflated by a do-nothing change came back reporting a clean apparatus. A control
written in the same shape as the thing it guards inherits that thing's blind spots, so the leak is
now read directly as well: *nothing* must move nothing, on its own and beside something else.

**Why per-item readings and not a score.** An interaction measured on a scalar cannot be given a
null at all: there is no spread to compare it against, and the honest answer to *is 0.088 big?* is
then unavailable rather than yes. So :func:`interaction` takes a measurement that returns **one
reading per item**, and the mean of those is the score. This is not a convenience; a version of
this module built on scores would be unfalsifiable and should not be written.

**What an interaction is not.** It is not a mechanism and it is not a repair. *A and B together
move the number* leaves open which of them changes the other's effect, whether a third thing
mediates both, and whether the composition is simply nonlinear in one of them. Those are separate
hypotheses needing separate experiments, and :func:`mechanisms` writes them down without pretending
to have chosen between them.

Pure standard library.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.space import Intervention

__all__ = ["Reading", "Interaction", "interaction", "mechanisms", "search", "compose",
           "TOGETHER", "DRAWS", "LUCK", "ORDER", "PLACEBO", "MECHANISMS", "SEED"]

#: How large ``|I|`` must be before it is worth a second look. The same bar the single-variable
#: interventions are held to, because it is the same quantity in the same units.
TOGETHER = 0.03

#: Bootstrap resamples of the items when putting an interval on ``I``.
DRAWS = 200

#: How much of the bootstrap may sit on the wrong side of zero before the sign is not established.
#: At 0.10 this is a two-sided interval of roughly 80%, which is deliberately generous: the point
#: here is to throw out interactions that are plainly noise, not to certify the survivors.
LUCK = 0.10

#: How far ``A then B`` may differ from ``B then A`` before the composition is order-dependent and
#: no interaction can be read off it.
ORDER = 0.03

#: How far composing with a change that does nothing may move the number before the apparatus is
#: carrying state and every figure it has produced is suspect.
PLACEBO = 0.02

SEED = 85

#: The mechanisms an interaction is consistent with. It is consistent with **all of them** the
#: moment it is found, which is the point of listing them: a discovered interaction narrows the
#: search and settles nothing, and a module that reported one of these would be inventing.
MECHANISMS: Tuple[str, ...] = (
    "A changes what B does",
    "B changes what A does",
    "something neither names mediates both",
    "one of them is nonlinear and the other moved it past a threshold",
    "the composition is an artefact of how they are applied",
    "the measurement responds to the pair and not to the system",
)


@dataclass(frozen=True)
class Reading:
    """One run: what each item scored, so that an interval can be put on anything derived from it."""

    name: str = ""
    per_item: Tuple[float, ...] = ()
    #: Invariants the run broke, if it was checked. A broken run is not a reading.
    broke: Tuple[str, ...] = ()

    @property
    def score(self) -> float:
        return round(mean(self.per_item), 4) if self.per_item else 0.0

    @property
    def sound(self) -> bool:
        return bool(self.per_item) and not self.broke


@dataclass
class Interaction:
    """What two changes did together, and every reason it might be nothing."""

    a: str = ""
    b: str = ""
    base: float = 0.0
    alone_a: float = 0.0
    alone_b: float = 0.0
    together: float = 0.0
    #: ``ΔAB − ΔA − ΔB``. Positive is synergy, negative is antagonism, and both are findings.
    extra: float = 0.0
    #: Share of bootstrap resamples whose ``I`` fell on the other side of zero.
    luck: float = 1.0
    #: ``A then B`` minus ``B then A``. Non-zero means the composition mutates something.
    order: float = 0.0
    #: The largest move produced by composing either change with a change that does nothing.
    placebo: float = 0.0
    broke: List[str] = field(default_factory=list)
    ran: bool = False
    says: str = ""

    @property
    def verdict(self) -> str:
        """``spoiled`` | ``order-dependent`` | ``apparatus`` | ``interacting`` | ``additive`` | ``not run``.

        The order matters and it is not a ranking of interest — it is a ranking of **what would
        make the number meaningless**. A broken invariant, a composition that depends on which way
        round it is applied, and an apparatus that responds to nothing all have to be ruled out
        before ``I`` is a quantity about the system at all.
        """
        if not self.ran:
            return "not run"
        if self.broke:
            return "spoiled"
        if abs(self.placebo) >= PLACEBO:
            return "apparatus"
        if abs(self.order) >= ORDER:
            return "order-dependent"
        if abs(self.extra) >= TOGETHER and self.luck <= LUCK:
            return "interacting"
        return "additive"

    @property
    def found(self) -> bool:
        return self.verdict == "interacting"

    def to_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "verdict": self.verdict, "extra": self.extra,
                "base": self.base, "alone_a": self.alone_a, "alone_b": self.alone_b,
                "together": self.together, "luck": self.luck, "order": self.order,
                "placebo": self.placebo, "broke": self.broke, "says": self.says}

    def render(self) -> str:
        mark = {"interacting": " ! ", "additive": " x ", "order-dependent": " ~ ",
                "apparatus": " ~ ", "spoiled": " ~ ", "not run": " ? "}
        return f"  {mark.get(self.verdict, '   ')} {self.a} × {self.b}\n      {self.says}"


def compose(a: Intervention, b: Intervention) -> Intervention:
    """``a`` then ``b``, as one intervention that keeps **both** sets of promises."""
    if a.change is None or b.change is None:
        raise ValueError(f"{a.name} and {b.name} must both be composable to be crossed")
    a_change, b_change = a.change, b.change
    return Intervention(verb="cross", on=f"{a.on} + {b.on}",
                        holds=tuple(a.holds) + tuple(b.holds),
                        deployable=a.deployable and b.deployable,
                        change=lambda setup: b_change(a_change(setup)),
                        says=f"{a.name} then {b.name}")


def _read(name: str, what: Optional[Intervention], setup: Any,
          measure: Callable[[Any], Sequence[float]]) -> Reading:
    """Apply a change (or none) and read every item, checking the promises it made."""
    try:
        after = setup if what is None or what.change is None else what.change(setup)
        per_item = tuple(float(x) for x in measure(after))
    except Exception:  # noqa: BLE001 — a run that cannot happen is not a reading
        return Reading(name=name)
    broke: List[str] = []
    if what is not None:
        for held in what.holds:
            drift = held.moved(setup, after)
            if drift is not None and abs(drift) >= held.slack:
                broke.append(held.name)
    return Reading(name=name, per_item=per_item, broke=tuple(broke))


def _extra(base: Reading, a: Reading, b: Reading, ab: Reading,
           at: Optional[Sequence[int]] = None) -> float:
    """``ΔAB − ΔA − ΔB`` over the given items, which for ``at=None`` is all of them."""
    def _m(r: Reading) -> float:
        return mean([r.per_item[i] for i in at]) if at else mean(r.per_item)
    return round((_m(ab) - _m(base)) - (_m(a) - _m(base)) - (_m(b) - _m(base)), 6)


def interaction(a: Intervention, b: Intervention, setup: Any, *,
                measure: Callable[[Any], Sequence[float]],
                nothing: Optional[Intervention] = None,
                draws: int = DRAWS, rng: Optional[random.Random] = None) -> Interaction:
    """Measure ``I(A, B)`` and then spend the rest of the work trying to show it is nothing.

    ``measure`` returns **one reading per item**, and the same items in the same order every time.
    That is checked rather than assumed: five runs whose lengths disagree are five different
    questions, and subtracting their means would be the defect V.82 found in `ascent` wearing a
    different hat.

    ``nothing`` is a change that does nothing, and supplying it is what makes the apparatus control
    possible. Left out, the control cannot run and the result says so — it is not quietly passed.
    """
    rng = rng or random.Random(SEED)
    out = Interaction(a=a.name, b=b.name)
    if not (a.composable and b.composable):
        out.says = "one of them is not composable, so `together` cannot be expressed"
        return out

    base = _read("base", None, setup, measure)
    only_a = _read("a", a, setup, measure)
    only_b = _read("b", b, setup, measure)
    ab = _read("a then b", compose(a, b), setup, measure)
    ba = _read("b then a", compose(b, a), setup, measure)
    runs = [base, only_a, only_b, ab, ba]

    if any(not r.per_item for r in runs):
        out.says = ("; ".join(r.name for r in runs if not r.per_item)
                    + " produced no readings, so nothing can be subtracted")
        return out
    sizes = {len(r.per_item) for r in runs}
    if len(sizes) > 1:
        out.says = (f"the runs read {sorted(sizes)} items — they are not the same question, "
                    "and subtracting their means would say nothing")
        return out

    out.ran = True
    out.broke = sorted({name for r in runs for name in r.broke})
    out.base, out.alone_a, out.alone_b = base.score, only_a.score, only_b.score
    out.together = ab.score
    out.extra = round(_extra(base, only_a, only_b, ab), 4)
    out.order = round(ab.score - ba.score, 4)

    n = len(base.per_item)
    wrong = 0
    for _ in range(max(1, draws)):
        at = [rng.randrange(n) for _ in range(n)]
        if (_extra(base, only_a, only_b, ab, at) >= 0) != (out.extra >= 0):
            wrong += 1
    out.luck = round((wrong + 1) / (max(1, draws) + 1), 4)

    if nothing is not None and nothing.composable:
        # Two forms, and the first one is here because the exam caught its absence. The original
        # control was only the interaction form — `I(A, nothing)` — and a leak that grows *linearly*
        # with how often the harness has been touched cancels out of that subtraction **exactly**.
        # Measured: a do-nothing change moved a leaky fixture +0.1982 and `I(A, nothing)` came back
        # +0.0000, so the control reported a clean apparatus on a harness whose every figure was
        # wrong. Writing a control in the same shape as the thing it guards is how it inherits the
        # thing's blind spots.
        idle = _read("nothing", nothing, setup, measure)
        direct = abs(idle.score - base.score) if idle.per_item else 0.0
        crossed = [abs(_extra(base, only, idle,
                              _read("then nothing", compose(what, nothing), setup, measure)))
                   for what, only in ((a, only_a), (b, only_b))]
        out.placebo = round(max([direct] + crossed), 4)
        placebo_says = (f"a change that does nothing shifts it {direct:+.4f} on its own "
                        f"and {max(crossed):+.4f} alongside")
    else:
        placebo_says = "no do-nothing change was supplied, so the apparatus was not checked"

    out.says = (f"alone {out.alone_a:.4f} / {out.alone_b:.4f} against {out.base:.4f}; "
                f"together {out.together:.4f}; I = {out.extra:+.4f} "
                f"(sign holds in {1 - out.luck:.2f} of {draws} resamples); "
                f"order {out.order:+.4f}; {placebo_says} — "
                + {"interacting": "they do something together that neither does alone",
                   "additive": "together is what each of them does, added up",
                   "order-dependent": "which way round it is applied changes the answer, so this "
                                      "is not an interaction but a mutation",
                   "apparatus": "the harness responds to a change that changes nothing, so every "
                                "figure it has produced is suspect — this one included",
                   "spoiled": "a promise broke, so the number is about nothing"}.get(out.verdict, ""))
    return out


def mechanisms(found: Interaction) -> List[str]:
    """What an interaction is consistent with, which on the day it is found is **all of it**.

    Returned as a list and never narrowed here. *A and B together move the number* does not say
    which changes the other, whether something neither of them names mediates both, or whether one
    is simply nonlinear and the other pushed it past a knee. Those are hypotheses, they need
    experiments, and a module that picked one would be doing the thing this package keeps catching
    itself doing.
    """
    return list(MECHANISMS) if found.found else []


def search(candidates: Sequence[Intervention], setup: Any, *,
           measure: Callable[[Any], Sequence[float]],
           nothing: Optional[Intervention] = None,
           rng: Optional[random.Random] = None) -> List[Interaction]:
    """Every pair, measured — and the reason this is not yet the adaptive search it should be.

    At four or five interventions the pairs are six or ten and exhaustive is simply correct. The
    guided version, which spends its budget where single-variable effects were near the bar, is
    what this needs before the list is long, and it is **not written here**: an ordering heuristic
    validated on nothing is the fake signal V.84's exam caught in its first minute, and the honest
    place for it is a version that can test whether it beats exhaustive on a case where exhaustive
    is affordable.
    """
    rng = rng or random.Random(SEED)
    out: List[Interaction] = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            out.append(interaction(a, b, setup, measure=measure, nothing=nothing, rng=rng))
    return sorted(out, key=lambda x: (not x.found, -abs(x.extra)))
