"""NYXARA · njp/closure.py — proving a new primitive is not new (🔒, NJP V.88).

V.87 ended owing one thing: it composed measurements out of four supplied primitives, so
*inventing the primitives themselves* was the next step. This version went to build that and came
back with a proof that most of it is impossible, which is a better outcome than the organ would
have been.

**The obstruction.** A primitive over a row of readings, if it is linear, is a **stencil** — a short
vector of coefficients slid along. ``d`` is ``(-1, 1)``. Curvature is ``(1, -2, 1)``. The
stride-two difference, which skips a neighbour and looks like something genuinely outside anything
V.87 held, is ``(1, 0, -1)``. It is not outside anything:

.. math::

    |H_{(1,0,-1)}(f)|^2 \;=\; 4\\,|H_{d}(f)|^2 \;-\; |H_{d \\circ d}(f)|^2

identically, at every frequency. And that is not a coincidence about one stencil. The energy
response of any length-``m`` stencil is a polynomial of degree ``m-1`` in :math:`\\cos 2\\pi f`; the
iterated differences :math:`d^k` give :math:`(2 - 2\\cos 2\\pi f)^k`, which span that space exactly.
So —

    **Every linear primitive, measured by how much energy it passes, is a linear combination of
    iterated differences. There is nothing to discover there.**

Checked at 1e-11 or better on random stencils of length two to five; see
:func:`in_span` and the exam in :mod:`nyxara.njp.closureschool`.

**What this is worth.** It turns V.87's ``SAME`` gate — which catches duplicates one world at a
time, empirically, after the search has run — into something decidable **before** any world is
consulted, for a whole family at once. A candidate that is redundant comes back with the identity
that makes it redundant, which is a stronger statement than *it correlates with measurement 4*.

**And it says what a real primitive would have to be.** Not linear. :func:`obeys_superposition`
tests exactly that, and a candidate that passes it — that *is* linear — is refused by the algebra
above without any need to look at data. A median or a rank filter fails superposition and is
therefore outside the family entirely; that is where an invented primitive has to live.

**Two fixtures failed on the way here, and both taught something.** The first tried to build a world
whose groups matched on variance, ``d`` and ``d∘d`` while differing in stride-two energy — which
the identity above makes *impossible*, and the linear algebra duly returned a direction that moved
nothing at all. The second matched the two groups' **power spectra** exactly, so that no linear
filter could see a difference, and then rescaled each item to a common range to match the order
statistics too — and the rescaling divided each item by its own peak-to-trough, which is a
per-item number that differs systematically between the groups, undoing the spectral match it was
protecting. `spread(d(d(x)))` read 1.93 on it.

Both failures point the same way: V.87's composed vocabulary is far more complete than four
primitives sounds, because iterated differences span the whole linear-energy family and ``max``,
``min`` and ``range`` cover the whole-sequence order statistics. What is left outside is narrow and
specific, and knowing that is worth more than a search that did not know it.

Pure standard library.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Stencil", "Verdict", "response", "difference", "in_span", "obeys_superposition",
           "judge", "IDENTITIES", "TOLERANCE", "GRID", "DEPTH", "SEED"]

#: How small a residual has to be before a candidate is the same quantity as the ones it was fitted
#: against. Far below anything a measurement could resolve — this is an algebraic identity or it is
#: nothing, and a "nearly redundant" primitive is a different and much weaker claim.
TOLERANCE = 1e-8

#: Frequencies the responses are compared at. More than the number of basis vectors, so the fit is
#: over-determined and a residual means something.
GRID = 33

#: How many iterated differences the span is taken over. A length-``m`` stencil needs ``m-1`` of
#: them, so this bounds the stencil length the proof covers.
DEPTH = 6

SEED = 88


def difference(k: int) -> Tuple[float, ...]:
    """The ``k``-times iterated difference, as a stencil. ``d`` is ``(-1, 1)``; ``d∘d`` is ``(1, -2, 1)``."""
    return tuple(float((-1) ** j * math.comb(k, j)) for j in range(k + 1))


def response(taps: Sequence[float], f: float) -> float:
    """How much energy this stencil passes at frequency ``f`` — :math:`|H(f)|^2`, from first principles."""
    real = sum(t * math.cos(2 * math.pi * f * i) for i, t in enumerate(taps))
    imag = sum(t * math.sin(2 * math.pi * f * i) for i, t in enumerate(taps))
    return real * real + imag * imag


def _solve(square: List[List[float]], rhs: List[float]) -> List[float]:
    """Gauss–Jordan on a small normal-equation system. No dependencies, and none needed."""
    n = len(rhs)
    m = [row[:] + [v] for row, v in zip(square, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-14:
            continue
        m[col] = [v / m[col][col] for v in m[col]]
        for r in range(n):
            if r != col and abs(m[r][col]) > 1e-14:
                m[r] = [a - m[r][col] * b for a, b in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


@dataclass
class Stencil:
    """A candidate linear primitive: a short vector of coefficients, and nothing else.

    Deliberately not a named operation. ``(1, 0, -1)`` is a vector; calling it *the stride-two
    difference* is a convenience for a reader and plays no part in anything below.
    """

    taps: Tuple[float, ...] = ()
    called: str = ""

    @property
    def name(self) -> str:
        return self.called or "(" + ", ".join(f"{t:g}" for t in self.taps) + ")"

    @property
    def length(self) -> int:
        return len(self.taps)

    def apply(self, xs: Sequence[float]) -> List[float]:
        k = self.length
        if k == 0 or len(xs) < k:
            return []
        return [sum(t * xs[i + j] for j, t in enumerate(self.taps))
                for i in range(len(xs) - k + 1)]


@dataclass
class Verdict:
    """What is known about a candidate primitive, and how it is known."""

    name: str = ""
    #: ``redundant`` | ``nonlinear`` | ``outside`` | ``not checked``
    stands: str = "not checked"
    #: The coefficients on ``1, d, d∘d, …`` that reproduce it, when it is redundant.
    identity: List[float] = field(default_factory=list)
    residual: float = 0.0
    says: str = ""

    @property
    def new(self) -> bool:
        """Could this be a discovery at all? Only if the algebra does not already contain it."""
        return self.stands in ("nonlinear", "outside")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "stands": self.stands, "identity": self.identity,
                "residual": self.residual, "new": self.new, "says": self.says}

    def render(self) -> str:
        mark = {"redundant": " x ", "nonlinear": " ! ", "outside": " ! ", "not checked": " ? "}
        return f"  {mark.get(self.stands, '   ')} {self.name:<22} {self.says}"


def in_span(taps: Sequence[float], *, depth: int = DEPTH,
            grid: int = GRID) -> Tuple[bool, List[float], float]:
    """Is this stencil's energy response a combination of the iterated differences' responses?

    Returns ``(redundant, coefficients, residual)``. The coefficients are on ``1, d, d∘d, …`` in
    that order, so a redundant stencil comes back with the identity that makes it redundant rather
    than with a correlation.
    """
    if not taps:
        return True, [], 0.0
    need = min(depth, max(1, len(taps) - 1))
    freqs = [i / (2.0 * (grid - 1)) for i in range(grid)]
    basis = [[1.0] * grid] + [[response(difference(k), f) for f in freqs]
                              for k in range(1, need + 1)]
    want = [response(taps, f) for f in freqs]
    n = len(basis)
    square = [[sum(basis[i][r] * basis[j][r] for r in range(grid)) for j in range(n)]
              for i in range(n)]
    rhs = [sum(basis[i][r] * want[r] for r in range(grid)) for i in range(n)]
    coefficients = _solve(square, rhs)
    residual = max(abs(want[r] - sum(coefficients[i] * basis[i][r] for i in range(n)))
                   for r in range(grid))
    return residual <= TOLERANCE, [round(c, 6) for c in coefficients], residual


def obeys_superposition(primitive: Callable[[Sequence[float]], Sequence[float]], *,
                        length: int = 12, trials: int = 12,
                        rng: Optional[random.Random] = None, tolerance: float = 1e-9) -> bool:
    """Is this primitive linear at all? Tested, not assumed, and by counterexample.

    A primitive that is linear is covered by the algebra above and cannot be a discovery. One that
    is not — a median, a rank, anything that sorts or compares — is outside the family entirely, and
    that is the only place an invented primitive can live.
    """
    rng = rng or random.Random(SEED)
    for _ in range(max(1, trials)):
        a = [rng.uniform(-1, 1) for _ in range(length)]
        b = [rng.uniform(-1, 1) for _ in range(length)]
        scale = rng.uniform(-2, 2)
        try:
            both = list(primitive([x + scale * y for x, y in zip(a, b)]))
            apart = [x + scale * y for x, y in zip(primitive(a), primitive(b))]
        except Exception:  # noqa: BLE001 — a primitive that cannot run is not linear either
            return False
        if len(both) != len(apart):
            return False
        if any(abs(x - y) > tolerance for x, y in zip(both, apart)):
            return False
    return True


def judge(candidate: Any, *, name: str = "", depth: int = DEPTH,
          rng: Optional[random.Random] = None) -> Verdict:
    """Decide whether a candidate primitive could be new — **before** any world is consulted.

    A :class:`Stencil`, or a tuple of taps, is judged by the algebra. Anything callable is first
    asked whether it is linear at all: if it is not, it is outside the family and the algebra has
    nothing to say about it, which is reported as exactly that rather than as a pass.
    """
    if isinstance(candidate, Stencil):
        taps, name = candidate.taps, name or candidate.name
    elif isinstance(candidate, (tuple, list)) and all(isinstance(t, (int, float))
                                                      for t in candidate):
        taps = tuple(float(t) for t in candidate)
        name = name or Stencil(taps=taps).name
    elif callable(candidate):
        if obeys_superposition(candidate, rng=rng):
            return Verdict(name=name or "a linear callable", stands="not checked",
                           says=("it is linear, but its taps were not given, so the algebra "
                                 "cannot be applied to it — supply the stencil"))
        return Verdict(name=name or "a callable", stands="nonlinear",
                       says=("it breaks superposition, so no stencil reproduces it and the "
                             "identity above does not reach it — this is where a new primitive "
                             "would have to live"))
    else:
        return Verdict(name=name or "?", says="not a stencil and not callable")

    redundant, identity, residual = in_span(taps, depth=depth)
    if redundant:
        terms = []
        for k, c in enumerate(identity):
            if abs(c) < 1e-6:
                continue
            terms.append(f"{c:+g}·" + ("1" if k == 0 else "d" + ("∘d" * (k - 1) if k > 1 else "")))
        return Verdict(name=name, stands="redundant", identity=identity, residual=residual,
                       says=(f"its energy is {' '.join(terms) or '0'} — an identity, not a "
                             f"resemblance (residual {residual:.2e})"))
    return Verdict(name=name, stands="outside", identity=identity, residual=residual,
                   says=(f"no combination of the first {min(depth, max(1, len(taps) - 1))} "
                         f"iterated differences reproduces it (residual {residual:.2e})"))


#: Identities worth having written down, because each one is a primitive somebody would otherwise
#: set out to discover. The last is the one this version was built around.
IDENTITIES: Tuple[Tuple[str, Tuple[float, ...]], ...] = (
    ("the difference", (-1.0, 1.0)),
    ("the difference twice", (1.0, -2.0, 1.0)),
    ("a two-tap average", (1.0, 1.0)),
    ("a three-tap average", (1.0, 1.0, 1.0)),
    ("the stride-two difference", (1.0, 0.0, -1.0)),
)
