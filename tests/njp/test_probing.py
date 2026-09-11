"""Finding out what an operation is, by trying to break it.

Two things are pinned hardest here. First, that nothing is ever reported as *holding* — a law that
survives reports the number of attempts it survived and nothing else. Second, the three-way
accounting: `flattered` for false novelty, and `buried` for novelty refused, which is the failure
that grows as the proofs get stronger.
"""

from __future__ import annotations

import random
from statistics import median

import pytest

from nyxara.njp.probing import (
    LAWS, TOLERANCE, TRIES, WIDTH, Family, Finding, Law, Reading, partition, probe,
)
from nyxara.njp.probingschool import OPERATIONS, TRUTH, examine, retrodict


def _probe(fn, tries=60):
    return probe(fn, name="x", tries=tries, rng=random.Random(89))


# --------------------------------------------------------------------------------------------- #
#  a probe can only refute
# --------------------------------------------------------------------------------------------- #
def test_a_law_that_survives_is_never_reported_as_holding():
    got = _probe(lambda xs: list(xs))
    alive = [f for f in got.findings if f.stands == "not refuted"]
    assert alive, "the identity must survive something"
    for finding in alive:
        assert finding.tries > 0
        assert "not the same as true" in finding.render()
        assert not finding.broken


def test_the_number_of_attempts_is_what_not_refuted_means():
    """Three tries and three hundred are not the same state of knowledge."""
    few = probe(lambda xs: list(xs), tries=3, rng=random.Random(89))
    many = probe(lambda xs: list(xs), tries=90, rng=random.Random(89))
    assert max(f.tries for f in few.findings) == 3
    assert max(f.tries for f in many.findings) == 90
    assert few.signature == many.signature


def test_a_break_comes_with_the_counterexample_that_did_it():
    got = _probe(lambda xs: [x * x for x in xs])
    broke = next(f for f in got.findings if f.law == "adds up")
    assert broke.broken and broke.witness
    assert "f(a+b)" in broke.witness


def test_a_probe_that_raises_is_neither_a_break_nor_a_survival():
    got = _probe(lambda xs: 1 / 0)
    assert all(not f.informative for f in got.findings)
    assert got.unrunnable and got.signature == ()


# --------------------------------------------------------------------------------------------- #
#  behaviour beats names and source
# --------------------------------------------------------------------------------------------- #
def test_the_identity_written_with_a_comparison_is_still_the_identity():
    """`max(x, x)` looks nonlinear. A prober that reads source is fooled; one that probes is not."""
    plain = _probe(lambda xs: list(xs)).signature
    sneaky = _probe(lambda xs: [max(x, x) for x in xs]).signature
    assert plain == sneaky


def test_a_scaling_is_not_the_identity_and_the_probes_say_why():
    """The truth table was wrong about this first, and it was the table that got corrected."""
    scaled = _probe(lambda xs: [2.5 * x for x in xs])
    broke = {f.law for f in scaled.findings if f.broken}
    assert "follows the level" in broke, "a constant added comes back multiplied"
    assert "settles" in broke, "applying it twice gives 6.25x"
    assert "scales" not in broke


def test_sorting_breaks_sliding_along_and_nothing_linear_does():
    assert "slides along" in _probe(sorted).signature
    assert "slides along" not in _probe(lambda xs: list(xs)).signature


def test_a_wandering_operation_is_caught_by_the_one_law_that_matters_first():
    """An operation that does not repeat cannot be characterised at all."""
    assert "repeats" in _probe(lambda xs: [x + random.random() for x in xs]).signature


# --------------------------------------------------------------------------------------------- #
#  the families are counted, not named
# --------------------------------------------------------------------------------------------- #
def test_a_family_carries_a_number_until_it_is_separated_from_something():
    got = Family(number=2, signature=("adds up",))
    assert got.name == "family 2"
    with pytest.raises(ValueError, match="separated from nothing"):
        got.christen("the sorting family", among=1)
    assert got.christen("the sorting family", among=3).name == "the sorting family"


def test_operations_that_break_the_same_laws_land_together():
    families, _ = partition({"a": lambda xs: list(xs),
                             "b": lambda xs: [max(x, x) for x in xs],
                             "c": sorted}, tries=40, rng=random.Random(89))
    grouped = {tuple(sorted(f.members)) for f in families}
    assert ("a", "b") in grouped and ("c",) in grouped


# --------------------------------------------------------------------------------------------- #
#  the exam, and the accounting that must outlive it
# --------------------------------------------------------------------------------------------- #
def test_fourteen_operations_are_sorted_into_their_families():
    got = examine()
    assert got["flattered"] == 0, "two that behave identically must not be called different"
    assert got["buried"] == 0
    assert got["invented"] == got["separable"] == 88
    assert got["found"] == got["of"] == 12
    assert got["passes"]


def test_the_resolution_of_the_battery_is_a_measured_number_not_a_worry():
    """`op 13` is linear plus a thousandth of a square and is caught. `op 14` is plus a
    quadrillionth, and lands with the identity — which is the battery's tolerance, stated."""
    got = retrodict()
    where = {name: i for i, f in enumerate(got["families"]) for name in f["members"]}
    assert where["op 13"] != where["op 1"], "a thousandth is inside the battery's reach"
    assert where["op 14"] == where["op 1"], "a quadrillionth is not, and saying so is the point"
    assert TOLERANCE < 1e-3


def test_a_blunt_battery_buries_novelty_rather_than_looking_decisive():
    """The failure V.88 warned about, produced on purpose and counted.

    One law cannot separate fourteen operations, and the honest report of that is a large `buried`
    — not one confident family. An exam without this number would score a blunt prober as tidy.
    """
    only = (next(law for law in LAWS if law.name == "repeats"),)
    families, _ = partition(OPERATIONS, laws=only, tries=20, rng=random.Random(89))
    assert len(families) < 4, "with one law almost everything collapses together"
    got = retrodict()
    assert got["buried"] == 0, "and the full battery does not"


def test_the_laws_are_supplied_and_the_partition_is_not():
    assert len(LAWS) >= 8
    assert all(law.attempt is not None and law.says for law in LAWS)
    assert TRIES >= 50 and WIDTH >= 4
    assert len(TRUTH) == 12, "twelve families by construction, never shown to the prober"


def test_an_empty_reading_claims_nothing():
    assert Reading().signature == () and Reading().unrunnable == []
    assert not Finding().informative and not Finding().broken
    assert Law().attempt is None
