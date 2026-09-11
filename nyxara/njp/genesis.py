"""NYXARA · njp/genesis.py — inventing a quantity nobody named (📐, NJP V.87).

V.86 found variables. It found them among :class:`~nyxara.njp.latent.Trait` objects somebody wrote
down, each with a ``read`` already attached — so *what can be measured* was still a supplied list,
and a cause outside it was as invisible as ever. Finding a new variable and **building a new
measurable dimension** are different capabilities, and only the first had been done.

So nothing here is given a trait. What arrives is a :class:`Trace`: for one item, a handful of
probes at unnamed coordinates and what was observed at each. Like this, and this is the whole input
format —

    probe   reading            probe   reading
    -1      0.11               -1      0.41
     0      0.42                0      0.42
    +1      0.10               +1      0.41

Both items read 0.42 at the centre. Every quantity anyone has thought to compute about *the item*
is identical. What differs is the **shape of the neighbourhood**, and no supplied vocabulary
contains a word for it until something invents one.

**The arithmetic is the search, not the answer.** A measurement is *composed*: up to three
:data:`STEPS` — adjacent differences, absolute value, normalise, centre — ending in one of six
:data:`ENDS`. So a candidate is a recipe like ``spread(d(norm(x)))``, which is notation and not a
name for anything, and the whole space of them is enumerated rather than sampled.

The first draft of this module listed nine operators by hand — level, slope, curvature, width — and
one of them happened to be the fixture's generating variable. That is :class:`~nyxara.njp.latent.Trait`
again one level up: the quantity was supplied and the search merely picked it out of a lineup. Four
primitives that compose is a weaker thing to be given than nine answers, and the difference is the
version.

**The name comes last, and only if it is earned.** :meth:`Measurement.christen` exists and nothing
in the discovery path calls it. A quantity named before it is validated is a hypothesis wearing a
conclusion's clothes, and the exam fails any world whose definition contains a semantic trait name.

**Four gates, and three of them are about not fooling yourself.**

======================  ====================================================================
gate                    what it refuses
======================  ====================================================================
novel                   the same quantity already in the register under another arithmetic
steady                  a number that changes when the observation is jiggled
tells apart             one that does not separate anything in a world it was not found in
observational only      any claim above what an experiment licensed — V.86's ladder, intact
======================  ====================================================================

**And discovery, validation and transfer happen in different worlds.** A measurement mined and
scored on the same traces is scored on the search's own noise, and will look excellent. So the
miner sees world A, the critic scores on world B, and the register only admits what still works in
C — which is also the only evidence that what was found is a regularity rather than a fact about
one world's arithmetic.

Pure standard library, and it leans on :mod:`nyxara.njp.latent` for the ladder and the separation
statistic rather than growing its own — a second opinion on *does this separate the groups* would
be a bug, not a feature.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.latent import LADDER, Standing

__all__ = ["Trace", "Measurement", "Register", "STEPS", "ENDS", "DEPTH", "mine", "criticise",
           "admit", "agree", "SAME", "STEADY", "SEPARATES", "DRAWS", "LUCK", "JIGGLE", "SEED"]

#: How strongly two measurements must agree across items before the newer one is the older one
#: wearing different arithmetic. Without this gate the module discovers ``level`` nine times.
SAME = 0.95

#: How strongly a measurement must agree with itself after the observation is jiggled. A quantity
#: that does not survive its own noise is not measuring the item, it is measuring the reading.
STEADY = 0.80

#: How far apart the two groups must sit, in pooled spreads, for a measurement to tell them apart.
SEPARATES = 0.30

DRAWS = 200
LUCK = 0.10

#: How hard the observation is jiggled when testing whether a measurement holds still.
JIGGLE = 0.03

SEED = 87


@dataclass(frozen=True)
class Trace:
    """One item's raw observations, with no interpretation attached.

    ``probes`` are coordinates and nothing more: their axes are numbered, not named, because naming
    them would smuggle in the vocabulary this module exists to do without. ``readings`` is what was
    observed at each. ``failed`` is the only thing anybody knows about the item.
    """

    probes: Tuple[Tuple[float, ...], ...] = ()
    readings: Tuple[float, ...] = ()
    failed: bool = False

    @property
    def axes(self) -> int:
        return len(self.probes[0]) if self.probes else 0

    def along(self, axis: int) -> List[Tuple[float, float]]:
        """The probes that vary only along ``axis``, in order, with their readings.

        Everything off that axis is held at zero, which is what makes a slope along one axis mean
        anything at all. A neighbourhood that moved two coordinates at once would give a number
        that is about neither.
        """
        if axis < 0 or axis >= self.axes:
            return []      # an axis these probes do not have reads nothing, rather than raising
        out = [(p[axis], r) for p, r in zip(self.probes, self.readings)
               if all(v == 0.0 for i, v in enumerate(p) if i != axis)]
        return sorted(out)


# --------------------------------------------------------------------------------------------- #
#  the search: four steps and six endings, composed — not nine quantities somebody wrote down
# --------------------------------------------------------------------------------------------- #
#  The first version of this module listed nine operators by hand — level, slope, curvature, width
#  and so on — and one of them happened to be the fixture's generating variable. That is
#  `Trait.read` again one level up: the quantity was supplied and the search merely picked it. So
#  the operators are **built** here instead, from four ways of transforming a row of readings and
#  six ways of ending it. A measurement is a short recipe like ``spread(d(norm(x)))``, which is
#  arithmetic notation and not a name for anything.
def _d(xs: Sequence[float]) -> List[float]:
    """Adjacent differences. Turns a shape into how fast it is changing."""
    return [b - a for a, b in zip(xs, xs[1:])] or [0.0]


def _abs(xs: Sequence[float]) -> List[float]:
    return [abs(x) for x in xs]


def _centred(xs: Sequence[float]) -> List[float]:
    m = mean(xs) if xs else 0.0
    return [x - m for x in xs]


def _norm(xs: Sequence[float]) -> List[float]:
    """Divide through by the largest. Removes *how high* and leaves *what shape*.

    Worth naming as a step rather than baking into an ending: whether a quantity survives this is
    exactly whether it is about the neighbourhood's shape or about its height, and in a world where
    the two groups have matched heights that is the difference between a measurement and nothing.
    """
    top = max((abs(x) for x in xs), default=0.0)
    return [x / top for x in xs] if top > 1e-9 else list(xs)


#: Ways of transforming a row of readings into another row. Composable, and each is one line of
#: arithmetic with no meaning attached.
STEPS: Dict[str, Callable[[Sequence[float]], List[float]]] = {
    "d": _d, "abs": _abs, "norm": _norm, "centred": _centred,
}

#: Ways of ending a recipe with a single number.
ENDS: Dict[str, Callable[[Sequence[float]], float]] = {
    "mean": lambda xs: mean(xs) if xs else 0.0,
    "max": lambda xs: max(xs) if xs else 0.0,
    "min": lambda xs: min(xs) if xs else 0.0,
    "spread": lambda xs: pstdev(xs) if len(xs) > 1 else 0.0,
    "range": lambda xs: (max(xs) - min(xs)) if xs else 0.0,
    "sum": lambda xs: float(sum(xs)) if xs else 0.0,
}

#: How many steps a recipe may stack. Three is enough for ``spread(d(norm(x)))`` and short enough
#: that the whole space is searched rather than sampled — an incomplete search whose incompleteness
#: is undeclared is how a null result becomes a lie.
DEPTH = 3


def _recipes(depth: int = DEPTH) -> List[Tuple[Tuple[str, ...], str]]:
    """Every recipe up to ``depth`` steps. Exhaustive, and small enough to stay exhaustive."""
    chains: List[Tuple[str, ...]] = [()]
    frontier: List[Tuple[str, ...]] = [()]
    for _ in range(depth):
        nxt: List[Tuple[str, ...]] = []
        for chain in frontier:
            for step in STEPS:
                if chain and chain[0] == step and step in ("abs", "norm"):
                    continue          # applying either twice in a row changes nothing
                nxt.append((step,) + chain)
        chains.extend(nxt)
        frontier = nxt
    return [(chain, end) for chain in chains for end in ENDS]


@dataclass
class Measurement:
    """A quantity somebody's arithmetic produces, before anybody knows what it is.

    It carries a number for a name until it has survived every gate, and :meth:`christen` is the
    only thing that changes that. Nothing in the discovery path calls it.
    """

    #: The steps, outermost first, so ``("spread-less",)`` reads the way the notation does.
    steps: Tuple[str, ...] = ()
    end: str = "mean"
    axis: int = 0
    number: int = 0
    #: Assigned by a person, afterwards, and never by the search.
    called: str = ""
    steady: float = 0.0
    separates: float = 0.0
    luck: float = 1.0
    #: How well it transferred, per world it was tried in and did not come from.
    elsewhere: Dict[str, float] = field(default_factory=dict)
    held: List[Standing] = field(default_factory=list)
    refused: str = ""

    @property
    def recipe(self) -> str:
        """The arithmetic, written out. Notation, never a name for what it means."""
        out = "x"
        for step in reversed(self.steps):
            out = f"{step}({out})"
        return f"{self.end}({out})"

    @property
    def name(self) -> str:
        return self.called or f"measurement {self.number}"

    @property
    def cost(self) -> int:
        """Probes it has to read. The cheapest honest notion of what an experiment costs."""
        return 1

    def of(self, trace: Trace) -> float:
        xs: List[float] = [r for _, r in trace.along(self.axis)]
        if not xs:
            return 0.0
        for step in reversed(self.steps):
            fn = STEPS.get(step)
            if fn is None:
                return 0.0
            xs = fn(xs)
            if not xs:
                return 0.0
        ending = ENDS.get(self.end)
        return round(ending(xs), 8) if ending else 0.0

    def over(self, traces: Sequence[Trace]) -> List[float]:
        return [self.of(t) for t in traces]

    def christen(self, called: str) -> "Measurement":
        """Give it a name — afterwards, and only once it stands for something.

        Refused while it holds nothing. A quantity named before it is validated is a hypothesis
        wearing a conclusion's clothes, and the name then does the arguing that the evidence has
        not done.
        """
        if not self.held:
            raise ValueError(f"{self.name} has survived nothing yet, so it has not earned a name")
        self.called = called
        return self

    def claims(self, kind: str) -> bool:
        return kind in LADDER and any(e.kind == kind for e in self.held)

    @property
    def causal(self) -> bool:
        return self.claims("interventional")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "recipe": self.recipe, "axis": self.axis,
                "steady": self.steady, "separates": self.separates, "luck": self.luck,
                "elsewhere": self.elsewhere, "refused": self.refused,
                "causal": self.causal,
                "held": [{"kind": e.kind, "got": e.got, "says": e.says} for e in self.held]}

    def render(self) -> str:
        if self.refused:
            return f"   -  {self.name:<16} {self.recipe:<26} — {self.refused}"
        where = ", ".join(f"{k} {v:.3f}" for k, v in sorted(self.elsewhere.items()))
        return (f"   !  {self.name:<16} {self.recipe:<26} — "
                f"steady {self.steady:.3f}, separates {self.separates:.3f} "
                f"(luck {self.luck:.3f}), elsewhere: {where or 'not tried'}")


def agree(one: Sequence[float], two: Sequence[float]) -> float:
    """How strongly two columns of numbers move together, as |r|, or 0 when one does not move."""
    if len(one) != len(two) or len(one) < 2:
        return 0.0
    a, b = pstdev(one), pstdev(two)
    if a <= 1e-9 or b <= 1e-9:
        return 0.0
    ma, mb = mean(one), mean(two)
    cov = mean([(x - ma) * (y - mb) for x, y in zip(one, two)])
    return round(min(1.0, abs(cov / (a * b))), 4)


def _separation(values: Sequence[float], failed: Sequence[bool]) -> float:
    bad = [v for v, f in zip(values, failed) if f]
    good = [v for v, f in zip(values, failed) if not f]
    if not bad or not good:
        return 0.0
    spread = pstdev(list(values))
    if spread <= 1e-9:
        return 0.0
    return round(abs(mean(bad) - mean(good)) / spread, 4)


def mine(traces: Sequence[Trace]) -> List[Measurement]:
    """Every ``(operator, axis)`` the traces admit. The search, and nothing else.

    No scoring happens here on purpose. A miner that ranked its own output on the traces it mined
    would be ranking the search's own noise, and the separation between mining and criticism is the
    only reason the numbers downstream mean anything.
    """
    if not traces:
        return []
    out: List[Measurement] = []
    n = 0
    for axis in range(traces[0].axes):
        for steps, end in _recipes():
            n += 1
            out.append(Measurement(steps=steps, end=end, axis=axis, number=n))
    return out


def criticise(what: Measurement, discovered_on: Sequence[Trace], check_on: Sequence[Trace],
              register: Sequence[Measurement] = (), *,
              draws: int = DRAWS, rng: Optional[random.Random] = None) -> Measurement:
    """Put one candidate through the gates, scoring it on a world it was **not** mined from.

    ``discovered_on`` is used for exactly one thing — asking whether the register already holds this
    quantity — because that is a question about the measurements and not about the world. Everything
    that could flatter the candidate is computed on ``check_on``.
    """
    rng = rng or random.Random(SEED)
    if not check_on:
        what.refused = "there was nowhere to check it"
        return what

    values = what.over(check_on)
    if pstdev(values) <= 1e-9:
        what.refused = "it reads the same on every item, so it measures nothing about them"
        return what

    for older in register:
        if agree(older.over(discovered_on), what.over(discovered_on)) >= SAME:
            what.refused = f"it is {older.name} in different arithmetic"
            return what

    jiggled = [Trace(probes=t.probes,
                     readings=tuple(r + rng.uniform(-JIGGLE, JIGGLE) for r in t.readings),
                     failed=t.failed)
               for t in check_on]
    what.steady = agree(values, what.over(jiggled))
    if what.steady < STEADY:
        what.refused = (f"jiggling the observation by {JIGGLE} moves it "
                        f"(it agrees with itself only {what.steady:.3f})")
        return what

    failed = [t.failed for t in check_on]
    what.separates = _separation(values, failed)
    if what.separates < SEPARATES:
        what.refused = (f"it separates the failures from the successes by only "
                        f"{what.separates:.4f} pooled spreads")
        return what
    beat = sum(1 for _ in range(max(1, draws))
               if _separation(values, rng.sample(failed, len(failed))) >= what.separates)
    what.luck = round((beat + 1) / (max(1, draws) + 1), 4)
    if what.luck > LUCK:
        what.refused = f"a shuffled split does as well {what.luck:.4f} of the time"
        return what

    what.held.append(Standing(
        kind="observational", got=what.separates,
        says=(f"{what.recipe} along axis {what.axis} separates them by {what.separates:.4f} "
              f"pooled spreads on a world it was not found in, holds still under jiggling "
              f"({what.steady:.3f}), and luck explains it {what.luck:.4f} of the time — "
              f"it **predicts**, which is not **causes**")))
    return what


@dataclass
class Register:
    """What survived. Nothing enters without having worked somewhere it did not come from."""

    kept: List[Measurement] = field(default_factory=list)
    refused: List[Measurement] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return [m.name for m in self.kept]

    def to_dict(self) -> Dict[str, Any]:
        return {"kept": [m.to_dict() for m in self.kept],
                "refused": [{"name": m.name, "why": m.refused} for m in self.refused]}

    def render(self) -> str:
        lines = [m.render() for m in self.kept] or ["   x  nothing survived the gates"]
        return "\n".join(lines)


def admit(traces_a: Sequence[Trace], traces_b: Sequence[Trace],
          traces_c: Sequence[Trace] = (), *, seeded: Sequence[Measurement] = (),
          rng: Optional[random.Random] = None) -> Register:
    """Mine world A, criticise on world B, and keep only what still works in world C.

    Three worlds and three jobs, and collapsing any two of them is how a search comes to believe
    its own noise. ``seeded`` is the register as it already stands — the obvious quantities somebody
    would compute anyway — so that rediscovering one of those is refused as the duplicate it is
    rather than reported as an invention.
    """
    rng = rng or random.Random(SEED)
    out = Register()
    register: List[Measurement] = list(seeded)
    for what in mine(traces_a):
        got = criticise(what, traces_a, traces_b, register, rng=rng)
        if got.refused:
            out.refused.append(got)
            continue
        if traces_c:
            elsewhere = _separation(got.over(traces_c), [t.failed for t in traces_c])
            got.elsewhere["a third world"] = elsewhere
            if elsewhere < SEPARATES:
                got.refused = (f"it did not survive a third world "
                               f"(separates {elsewhere:.4f} there)")
                out.refused.append(got)
                continue
        out.kept.append(got)
        register.append(got)
    return out
