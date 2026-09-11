"""Generating a law nobody wrote down.

The exam is not *were laws found* — a schema of 1,456 candidates always yields true sentences. It
is whether a discovered law draws a distinction the **supplied** vocabulary could not, and that is
checkable because V.89's nine laws leave three pairs merged.
"""

from __future__ import annotations

import random
from statistics import median

import pytest

from nyxara.njp.probing import probe
from nyxara.njp.regularity import (
    MOVES, REDUCTIONS, RELATIONS, TRIES, Move, Rule, Split, differ, discover, holds, rules,
)
from nyxara.njp.regularityschool import MERGED, OPERATIONS, examine, old_families


# --------------------------------------------------------------------------------------------- #
#  the substrate is moves, not laws
# --------------------------------------------------------------------------------------------- #
def test_a_move_says_nothing_on_its_own():
    """`reverse` is not *the operation is symmetric*. It is a thing one can do to seven numbers."""
    for move in MOVES:
        assert move.do is not None
        got = move([1.0, 2.0, 3.0])
        assert len(got) == 3
    assert len(MOVES) >= 5


def test_the_schema_is_enumerated_not_sampled():
    got = rules()
    assert len({r.statement for r in got}) == len(got), "no candidate appears twice"
    assert len(got) > 500, "a schema this small would not need searching"


def test_a_rule_that_never_mentions_the_operation_is_not_a_candidate():
    """`first: lifted(x) >= negated(x)` is a claim about seven random numbers, not about `f`.

    The first version kept these, and the exam immediately produced a distinction between an
    operation and an identical copy of itself: a rule about `x` that survived on one draw of rows
    and not on another.
    """
    assert not any(r.left[0] == "x" and r.right[0] == "x" for r in rules())


def test_every_operation_is_tried_on_the_same_rows():
    """Two operations compared on two different draws are two different questions.

    The same defect V.82 found in `ascent`, arriving at the level of laws — and it manufactured a
    split between an operation and itself, which is how it was caught.
    """
    splits, _ = discover({"a": sorted, "b": sorted, "c": lambda xs: list(xs)},
                         tries=15, rng=random.Random(90))
    for split in splits:
        together = {"a", "b"}
        assert together <= set(split.obeyed_by) or together <= set(split.broken_by), \
            f"{split.rule.statement} separated two identical operations"


def test_a_rule_against_itself_is_dropped():
    """`f(x) == f(x)` is true of everything and says so about nothing."""
    assert not any(r.left == r.right for r in rules())


def test_a_law_carries_a_number_until_it_has_drawn_a_line():
    got = Rule(number=52)
    assert got.name == "law 52"
    with pytest.raises(ValueError, match="separates nothing"):
        got.christen("dominance", splits=0)
    assert got.christen("dominance", splits=3).name == "dominance"


def test_a_rule_reads_as_a_statement_not_as_a_name():
    got = Rule(left=("f", "as it is"), right=("x", "as it is"), how="every", relation=">=")
    assert got.statement == "every: f(x) >= x"
    assert Rule(left=("f", "reversed"), right=("f", "as it is"), how="first",
                relation=">=").statement == "first: reversed(f(x)) >= f(x)"


# --------------------------------------------------------------------------------------------- #
#  probing a rule can only refute
# --------------------------------------------------------------------------------------------- #
def test_an_unbroken_rule_reports_how_many_tries_it_survived():
    rule = Rule(left=("f", "as it is"), right=("x", "as it is"), how="every", relation=">=")
    unbroken, tried = holds(rule, lambda xs: [v + 1.0 for v in xs], tries=25,
                            rng=random.Random(90))
    assert unbroken and tried == 25


def test_a_rule_over_two_lengths_is_not_a_rule():
    rule = Rule(left=("f", "as it is"), right=("x", "as it is"), how="every", relation="==")
    unbroken, _ = holds(rule, lambda xs: list(xs)[:-1], tries=5)
    assert not unbroken


def test_an_operation_that_raises_breaks_every_rule_rather_than_passing():
    rule = Rule(left=("f", "as it is"), right=("x", "as it is"), how="first", relation=">=")
    assert not holds(rule, lambda xs: 1 / 0, tries=3)[0]


# --------------------------------------------------------------------------------------------- #
#  a law earns its existence by drawing a line
# --------------------------------------------------------------------------------------------- #
def test_a_law_everything_obeys_is_dropped():
    """A law's whole claim to exist is the distinction it creates."""
    splits, report = discover({"a": lambda xs: list(xs), "b": lambda xs: list(xs)},
                              tries=12, rng=random.Random(90))
    assert splits == [], "two identical operations admit no distinction at all"
    assert report["vacuous"] == report["searched"]


def test_two_laws_with_the_same_split_are_one_discovery():
    splits, _ = discover({"up": sorted, "flat": lambda xs: list(xs)}, tries=12,
                         rng=random.Random(90))
    assert splits, "these two really are different"
    assert any(s.same_as for s in splits), "a schema this size restates itself"
    seen = {(s.obeyed_by, s.broken_by) for s in splits}
    assert len(seen) == len(splits), "each kept split is distinct by construction"


def test_dominance_falls_out_without_being_named():
    """`every: f(x) >= x` is a real regularity, composed from a side, a reduction and a relation."""
    rule = next(r for r in rules()
                if r.statement == "every: f(x) >= x")
    def _win(pick):
        def run(xs):
            ring = list(xs) + list(xs[:2])
            return [pick(ring[i:i + 3]) for i in range(len(xs))]
        return run
    assert holds(rule, _win(max), tries=40, rng=random.Random(90))[0]
    assert not holds(rule, _win(min), tries=40, rng=random.Random(90))[0]


def test_sortedness_needs_a_reduction_and_that_is_why_there_is_one():
    """Pointwise, a descending row beats its reverse in front and loses behind — so the law that
    says *the output is sorted downward* cannot be stated at all without `first`."""
    every = next(r for r in rules() if r.statement == "every: f(x) >= reversed(f(x))")
    first = next(r for r in rules() if r.statement == "first: f(x) >= reversed(f(x))")
    down = lambda xs: sorted(xs, reverse=True)
    assert not holds(every, down, tries=40, rng=random.Random(90))[0]
    assert holds(first, down, tries=40, rng=random.Random(90))[0]
    assert not holds(first, sorted, tries=40, rng=random.Random(90))[0]


# --------------------------------------------------------------------------------------------- #
#  the milestone
# --------------------------------------------------------------------------------------------- #
def test_the_supplied_vocabulary_really_does_merge_these_pairs():
    """Recomputed, never asserted — a change to V.89's battery must not make this exam easier."""
    families = old_families()
    where = {name: key for key, names in families.items() for name in names}
    for one, two in MERGED:
        assert where[one] == where[two], f"{one} and {two} are no longer merged"
        assert differ(OPERATIONS[one], OPERATIONS[two], rng=random.Random(90)), \
            f"{one} and {two} must genuinely differ or the pair proves nothing"


def test_a_pair_that_never_disagrees_must_not_be_split():
    """The identity written through a comparison. Separating it would be false novelty."""
    assert not differ(OPERATIONS["leave alone"], OPERATIONS["also leave alone"],
                      rng=random.Random(90))


def test_a_tight_relation_is_an_infinitely_sensitive_detector():
    """Why the first resolution-limited fixture failed, pinned so it is not re-attempted.

    A sort preserves the total **exactly**, so `total: x >= f(x)` holds with equality on it — and a
    relation that is tight catches any scaling whatever, however small. There is no tolerance to
    hide under when the two sides are equal.
    """
    rule = next(r for r in rules() if r.statement == "total: x >= f(x)")
    barely = lambda xs: [1.0000001 * v for v in sorted(xs)]
    assert holds(rule, sorted, tries=30, rng=random.Random(90))[0]
    assert not holds(rule, barely, tries=30, rng=random.Random(90))[0]


def test_resolution_limited_is_computed_and_not_exercised():
    """Said out loud rather than left to be noticed.

    Two fixtures were built to exercise it and both failed — the second one *looked* out of reach on
    one draw of rows and was caught on another, which is the same one-draw error this line of work
    began by fixing. A gate nothing exercises has not been shown to work, and the honest report is
    the label, not a fixture that only sometimes holds.
    """
    got = examine()
    assert got["resolution_limited"] == 0
    assert "resolution_limited" in got["unexercised"]


def test_flattered_is_not_vacuous():
    """There is a pair that must not be split, or the zero above means nothing."""
    got = examine()
    assert got["right_to_merge"] >= 1
    assert got["flattered"] == 0


def test_an_empty_split_claims_nothing():
    assert not Split().useful
    assert Split(obeyed_by=("a",)).useful is False
    assert Split(obeyed_by=("a",), broken_by=("b",)).useful
    assert TRIES >= 20 and len(RELATIONS) == 2
