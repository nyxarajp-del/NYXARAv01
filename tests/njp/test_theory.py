"""What never changes, and running what survives (NJP V.45)."""
from __future__ import annotations

import pytest

from nyxara.njp.theory import MIN_SITUATIONS, Hunter, Law, Situation, Theory


def S(name, variables, relations):
    return Situation(name, tuple(variables), frozenset(relations))


CHAIN = [
    S("monday", ["a", "b", "c"],
      [("a", "causes", "b"), ("b", "causes", "c"), ("a", "requires", "c")]),
    S("thursday", ["p", "q", "r"],
      [("p", "causes", "q"), ("q", "causes", "r"), ("p", "requires", "r")]),
    S("august", ["x", "y", "z"],
      [("x", "causes", "y"), ("y", "causes", "z"), ("x", "requires", "z")]),
]


@pytest.fixture
def hunter():
    return Hunter()


# --------------------------------------------------------------------------- #
# The vocabulary changes, which is the whole difficulty
# --------------------------------------------------------------------------- #
def test_an_invariant_is_found_across_situations_sharing_no_word(hunter):
    laws = hunter.hunt(CHAIN)
    assert laws and laws[0].exact
    assert len(laws[0].invariant.edges) == 3
    assert set(laws[0].invariant.alignment) == {"monday", "thursday", "august"}


def test_the_roles_are_positions_and_are_filled_in_every_situation(hunter):
    law = hunter.hunt(CHAIN)[0]
    assert all(role.startswith("role") for role in law.invariant.roles)
    for _name, mapping in law.invariant.alignment.items():
        assert set(mapping) == set(law.invariant.roles)


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #
def test_two_situations_are_a_coincidence_with_a_name(hunter):
    assert hunter.hunt(CHAIN[:2]) == []
    assert MIN_SITUATIONS >= 3


def test_a_relation_that_could_not_have_failed_is_not_a_finding(hunter):
    """Every edge present because every edge is present describes the encoding, not the world."""
    dense = [S(n, ["p", "q"], [("p", "causes", "q"), ("q", "causes", "p")]) for n in "abc"]
    assert hunter.hunt(dense) == []


def test_an_exception_is_reported_and_never_hidden(hunter):
    """It dropped the edge with a counterexample and called the remainder an invariant."""
    odd = S("odd", ["u", "v", "w"], [("u", "causes", "v"), ("v", "causes", "w")])
    laws = hunter.hunt(CHAIN + [odd])
    assert len(laws) == 2
    invariant, law = laws
    assert invariant.exact and len(invariant.invariant.edges) == 2
    assert not law.exact and law.exceptions == ["odd"]
    assert law.invariant.edges == (("role0", "requires", "role2"),)
    assert "EXCEPT in odd" in law.render()


# --------------------------------------------------------------------------- #
# The compiler: the law becomes something that runs
# --------------------------------------------------------------------------- #
def test_a_compiled_theory_predicts_an_edge_it_was_not_shown(hunter):
    theory = hunter.hunt(CHAIN)[0].compile()
    unseen = S("unseen", ["m", "n", "o"], [("m", "causes", "n"), ("n", "causes", "o")])
    got = theory.predict(unseen, hidden=[("role0", "requires", "role2")])
    assert got.aligned and not got.ok
    assert ("m", "requires", "o") in got.missing, "the theory must say what has to be there"


def test_a_theory_that_holds_says_so(hunter):
    theory = hunter.hunt(CHAIN)[0].compile()
    same = S("again", ["m", "n", "o"],
             [("m", "causes", "n"), ("n", "causes", "o"), ("m", "requires", "o")])
    assert theory.predict(same).ok


def test_checking_reports_a_verdict_over_many_situations(hunter):
    theory = hunter.hunt(CHAIN)[0].compile()
    good = S("g", ["m", "n", "o"],
             [("m", "causes", "n"), ("n", "causes", "o"), ("m", "requires", "o")])
    bad = S("b", ["u", "v", "w"], [("u", "causes", "v"), ("v", "causes", "w"),
                                   ("u", "requires", "v")])
    report = theory.check([good, bad])
    assert report["verdict"] in ("holds", "invalid", "incomplete")
    assert report["asked"] == 2


def test_alignment_never_sees_the_edge_being_predicted(hunter):
    """Aligning on the answer and then predicting it is the mistake every exam here has made once."""
    theory = hunter.hunt(CHAIN)[0].compile()
    unseen = S("unseen", ["m", "n", "o"], [("m", "causes", "n"), ("n", "causes", "o")])
    hidden = [("role0", "requires", "role2")]
    assert theory.align(unseen, ignore=hidden) is not None
    assert theory.align(unseen) is None, "without hiding it, no alignment satisfies every edge"


# --------------------------------------------------------------------------- #
# It may not widen its own scope
# --------------------------------------------------------------------------- #
def test_a_law_declines_a_situation_of_another_shape(hunter):
    theory = hunter.hunt(CHAIN)[0].compile()
    bigger = S("bigger", ["m", "n", "o", "q"],
               [("m", "causes", "n"), ("n", "causes", "o"), ("m", "requires", "o"),
                ("o", "causes", "q"), ("q", "causes", "m")])
    got = theory.predict(bigger)
    assert not got.aligned and "shape" in got.why


def test_a_situation_that_cannot_be_aligned_gets_no_prediction(hunter):
    theory = hunter.hunt(CHAIN)[0].compile()
    unrelated = S("unrelated", ["m", "n", "o"], [("m", "requires", "n")])
    got = theory.predict(unrelated, hidden=[("role1", "causes", "role2")])
    assert not got.aligned and not got.expected
