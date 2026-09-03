"""Rival structures, not one graph walked (NJP V.40)."""
from __future__ import annotations

import pytest

from nyxara.njp.surgery import (
    COMPLEXITY_COST, Observation as O, Structure, Surgeon,
)


@pytest.fixture
def surgeon():
    return Surgeon()


CHAIN = [O("a", "b", True), O("b", "c", True), O("a", "c", True),
         O("a", "c", False, frozenset({"b"}))]
COLLIDER = [O("a", "c", True), O("b", "c", True), O("a", "b", False)]
TRIANGLE = [O("a", "b", True), O("b", "c", True), O("a", "c", True)]
NONE = [O("a", "b", False), O("b", "c", False), O("a", "c", False)]


# --------------------------------------------------------------------------- #
# The three steps
# --------------------------------------------------------------------------- #
def test_a_collider_is_recovered_uniquely(surgeon):
    """The one case where direction falls out of the data instead of being assumed."""
    got = surgeon.discover(("a", "b", "c"), COLLIDER)
    assert got.determined and got.equivalent_count == 1
    assert got.best.edges == frozenset({("a", "c"), ("b", "c")})
    assert got.holds("a", "c") and got.holds("b", "c")


def test_a_chain_and_a_fork_are_one_equivalence_class(surgeon):
    """`a→b→c`, `a←b→c` and `a←b←c` imply exactly the same thing. Naming one invents a direction."""
    got = surgeon.discover(("a", "b", "c"), CHAIN)
    assert got.equivalent_count == 3 and not got.determined
    shapes = {s.edges for s in got.equivalent}
    assert frozenset({("a", "b"), ("b", "c")}) in shapes
    assert frozenset({("b", "a"), ("b", "c")}) in shapes
    assert frozenset({("b", "a"), ("c", "b")}) in shapes
    # and nothing is claimed about direction, because nothing can be
    assert not got.holds("a", "b") and not got.holds("b", "c")


def test_the_skeleton_is_right_even_where_direction_is_not(surgeon):
    got = surgeon.discover(("a", "b", "c"), CHAIN)
    assert got.skeleton == frozenset({frozenset({"a", "b"}), frozenset({"b", "c"})})


def test_a_triangle_has_six_orientations(surgeon):
    got = surgeon.discover(("a", "b", "c"), TRIANGLE)
    assert got.equivalent_count == 6
    assert all(s.acyclic for s in got.equivalent)


def test_independence_everywhere_gives_an_empty_graph(surgeon):
    got = surgeon.discover(("a", "b", "c"), NONE)
    assert got.determined and got.best.edges == frozenset()


def test_a_conditional_observation_is_what_deletes_an_edge(surgeon):
    """Without it a chain and a triangle are indistinguishable — measured at six, not three."""
    marginal = surgeon.discover(("a", "b", "c"), TRIANGLE)
    conditional = surgeon.discover(("a", "b", "c"), CHAIN)
    assert marginal.equivalent_count == 6 and conditional.equivalent_count == 3


def test_the_middle_of_a_separating_set_is_not_a_collider(surgeon):
    """`a → b ← c` was accepted for a chain, where b is what makes a and c independent."""
    got = surgeon.discover(("a", "b", "c"), CHAIN)
    for structure in got.equivalent:
        assert structure.edges != frozenset({("a", "b"), ("c", "b")})


def test_a_shielded_triple_orients_nothing(surgeon):
    """With an a–b edge present the pattern says nothing about a and b being independent."""
    got = surgeon.discover(("a", "b", "c"), TRIANGLE)
    assert got.forced == frozenset()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_a_structure_that_explains_more_beats_a_simpler_one(surgeon):
    explains = Structure(("a", "b", "c"), frozenset({("a", "b"), ("b", "c")}))
    simpler = Structure(("a", "b", "c"), frozenset({("a", "b")}))
    assert surgeon.score(explains, CHAIN).total > surgeon.score(simpler, CHAIN).total


def test_an_unsupported_edge_costs_more_than_complexity_saves(surgeon):
    """Without the unsupported term the complete graph explains everything and wins."""
    lean = Structure(("a", "b", "c"), frozenset({("a", "b"), ("b", "c")}))
    stuffed = Structure(("a", "b", "c"),
                        frozenset({("a", "b"), ("b", "c"), ("a", "c")}))
    assert surgeon.score(lean, CHAIN).total > surgeon.score(stuffed, CHAIN).total
    assert surgeon.score(stuffed, CHAIN).unsupported >= 1


def test_simplicity_is_a_tie_break_and_never_a_truth():
    assert COMPLEXITY_COST < 1.0, "complexity must never outweigh explaining one observation"


# --------------------------------------------------------------------------- #
# The operations
# --------------------------------------------------------------------------- #
def test_the_five_operations_do_what_they_say():
    base = Structure(("a", "b", "c"), frozenset({("a", "b"), ("b", "c")}))
    assert ("a", "c") in Surgeon.add_edge(base, "a", "c").edges
    assert ("a", "b") not in Surgeon.remove_edge(base, "a", "b").edges
    assert ("b", "a") in Surgeon.reverse_edge(base, "a", "b").edges
    merged = Surgeon.merge_nodes(base, "a", "b", into="ab")
    assert "ab" in merged.nodes and ("ab", "c") in merged.edges
    assert not any(x == y for x, y in merged.edges), "a merge must not leave a self-loop"
    split = Surgeon.split_node(base, "b", incoming="b_in", outgoing="b_out")
    assert ("a", "b_in") in split.edges and ("b_out", "c") in split.edges


def test_split_is_the_identity_defect_as_an_operation():
    """V.38 fixed one spelling covering two things by hand; this proposes it structurally."""
    base = Structure(("heart", "atrium", "daylight"),
                     frozenset({("heart", "atrium"), ("atrium", "daylight")}))
    got = Surgeon.split_node(base, "atrium", incoming="atrium#body", outgoing="atrium#building")
    assert ("heart", "atrium#body") in got.edges
    assert ("atrium#building", "daylight") in got.edges
    assert not got.connected("heart", "daylight")


# --------------------------------------------------------------------------- #
# What it may not do
# --------------------------------------------------------------------------- #
def test_a_pair_nobody_observed_is_not_an_edge(surgeon):
    got = surgeon.discover(("a", "b", "c"), [O("a", "b", True)])
    assert got.skeleton == frozenset({frozenset({"a", "b"})})


def test_it_never_edits_the_store(surgeon):
    class Store:
        facts: dict = {}

        @staticmethod
        def _key(text):
            return str(text).lower()

    store = Store()
    Surgeon(store).discover(("a", "b"), [O("a", "b", True)])
    assert store.facts == {}


def test_too_many_nodes_reports_not_enumerated_rather_than_empty():
    small = Surgeon(max_nodes=3)
    got = small.discover(tuple("abcde"), [O("a", "b", True), O("b", "c", True)])
    assert got.enumerated is False and got.equivalent == []
    assert got.skeleton, "the skeleton is still returned"
    assert "not enumerated" in got.why


def test_the_class_is_reported_as_a_class_and_not_a_pick(surgeon):
    got = surgeon.discover(("a", "b", "c"), CHAIN)
    assert "no observation of this kind can separate them" in got.why
