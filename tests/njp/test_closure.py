"""Proving a new primitive is not new.

V.88 set out to discover primitives and came back with a proof that most of that is impossible.
These pin the proof — `stride-2 = 4·d − d∘d` identically, and the general statement for random
stencils — and the gate that turns it into something decidable before any world is looked at.
"""

from __future__ import annotations

import math
import random
from statistics import median

import pytest

from nyxara.njp.closure import (
    DEPTH, GRID, IDENTITIES, TOLERANCE, Stencil, difference, in_span, judge,
    obeys_superposition, response,
)
from nyxara.njp.closureschool import KNOWN, examine, retrodict, sliding


# --------------------------------------------------------------------------------------------- #
#  the obstruction
# --------------------------------------------------------------------------------------------- #
def test_the_stride_two_difference_is_four_d_minus_d_twice():
    """The identity the whole version turns on, checked at every frequency and not at a few."""
    for i in range(41):
        f = i / 80.0
        stride = response((1.0, 0.0, -1.0), f)
        built = 4.0 * response(difference(1), f) - response(difference(2), f)
        assert stride == pytest.approx(built, abs=1e-12), f


def test_every_linear_stencil_energy_is_a_combination_of_iterated_differences():
    """The general statement. A length-m stencil's response is a degree m−1 polynomial in cos,
    and the iterated differences span exactly that space — so there is nothing to discover here."""
    rng = random.Random(88)
    for _ in range(12):
        taps = tuple(rng.uniform(-2, 2) for _ in range(rng.randint(2, 5)))
        redundant, _coefficients, residual = in_span(taps)
        assert redundant, (taps, residual)
        assert residual <= TOLERANCE


def test_smoothing_is_inside_the_closure_too():
    """Averaging felt outside it, and is not — which is why the first fixture attempt died."""
    got = judge((1.0, 1.0, 1.0), name="a three-tap average")
    assert got.stands == "redundant" and not got.new
    assert "d∘d" in got.says


def test_a_redundant_primitive_comes_back_with_the_identity_not_a_correlation():
    got = judge((1.0, 0.0, -1.0))
    assert got.identity and abs(got.identity[1] - 4.0) < 1e-6
    assert abs(got.identity[2] + 1.0) < 1e-6
    assert "an identity, not a resemblance" in got.says


def test_every_written_down_identity_holds():
    for name, taps in IDENTITIES:
        got = judge(taps, name=name)
        assert got.stands == "redundant", name
        assert got.residual <= TOLERANCE


# --------------------------------------------------------------------------------------------- #
#  where a real primitive would have to live
# --------------------------------------------------------------------------------------------- #
def test_sorting_breaks_superposition_and_escapes_the_algebra():
    for pick in (median, max, min, lambda xs: max(xs) - min(xs)):
        assert not obeys_superposition(sliding(pick), rng=random.Random(88))
        assert judge(sliding(pick)).stands == "nonlinear"


def test_a_linear_callable_is_not_mistaken_for_a_new_primitive():
    """The false-alarm side. A weighted window is linear however it is wrapped."""
    weighted = sliding(lambda xs: 0.25 * xs[0] + 0.5 * xs[1] + 0.25 * xs[2])
    assert obeys_superposition(weighted, rng=random.Random(88))
    assert judge(weighted).stands == "not checked", "linear, but its taps were not handed over"
    assert "supply the stencil" in judge(weighted).says


def test_taps_are_read_off_a_primitive_rather_than_off_its_name():
    """A primitive does not get to say what it is."""
    from nyxara.njp.closureschool import _impulse

    got = _impulse(sliding(lambda xs: 1.0 * xs[0] - 2.0 * xs[1] + 1.0 * xs[2]))
    assert len(got) == 3
    assert judge(got).stands == "redundant"


def test_a_primitive_that_cannot_run_is_not_called_linear():
    assert not obeys_superposition(lambda xs: 1 / 0)
    assert not obeys_superposition(lambda xs: list(xs)[: len(xs) // 2 if sum(xs) > 0 else 2])


# --------------------------------------------------------------------------------------------- #
#  the bookkeeping
# --------------------------------------------------------------------------------------------- #
def test_a_stencil_applies_as_a_sliding_window():
    got = Stencil(taps=(-1.0, 1.0)).apply([1.0, 3.0, 6.0, 10.0])
    assert got == [2.0, 3.0, 4.0]
    assert Stencil(taps=()).apply([1.0]) == []
    assert Stencil(taps=(1.0, 1.0, 1.0)).apply([1.0]) == []


def test_the_difference_stencils_are_the_binomials():
    assert difference(1) == (1.0, -1.0) or difference(1) == (-1.0, 1.0)
    assert difference(2) == (1.0, -2.0, 1.0)
    assert difference(3) == (1.0, -3.0, 3.0, -1.0)
    assert all(math.isfinite(t) for t in difference(4))


def test_nonsense_is_reported_rather_than_judged():
    assert judge("not a primitive").stands == "not checked"
    assert judge(None).stands == "not checked"
    assert GRID > DEPTH and 0 < TOLERANCE < 1e-4


# --------------------------------------------------------------------------------------------- #
#  the exam
# --------------------------------------------------------------------------------------------- #
def test_the_gate_passes_its_retrodiction():
    got = examine()
    assert got["right"] == got["of"] == 6
    assert got["flattered"] == 0 and got["buried"] == 0
    assert got["passes"]


def test_half_the_candidates_are_ones_the_algebra_already_owns():
    """Otherwise a module that answers `new` to everything scores well."""
    assert sum(1 for c in KNOWN if c.truth == "redundant") == 3


def test_calling_everything_new_fails_the_exam():
    """The check on the check. That is the module a search with no algebra secretly is."""
    import nyxara.njp.closure as module

    real = module.Verdict.new
    try:
        module.Verdict.new = property(lambda self: True)
        got = retrodict()
        assert got["flattered"] >= 3
    finally:
        module.Verdict.new = real
    assert examine()["passes"]
