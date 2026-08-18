"""There was no number that could get worse.

`stats()` grows monotonically by construction — more facts, more cells, more synapses, more
schemas — so reading it as progress reads the size of the store as the quality of the mind. This
module is the other number, and most of these tests are about the ways an index like it normally
starts lying: a missing term counted as a failure, a self-report smuggled in as evidence, a
denominator that punishes honesty, and a scalar that moves while everything worth measuring stays
where it was.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.index import (
    DENOMINATOR, NUMERATOR, IntelligenceIndex, IntelligenceVector, Term, measure,
)


def _vector(**values) -> IntelligenceVector:
    """A vector built by hand, so the arithmetic can be tested without a benchmark run."""
    return IntelligenceVector(
        terms=[Term(key=k, value=values.get(k), source="test")
               for k in (*NUMERATOR, *DENOMINATOR)])


@pytest.fixture(scope="module")
def taught():
    brain = NJPBrain()
    for sentence in ("aag se garmi hoti hai", "garmi se pasina hota hai",
                     "sparrow is_a bird", "crow is_a bird"):
        brain.think(sentence)
    return brain


# --------------------------------------------------------------------------- #
# Absent is not zero
# --------------------------------------------------------------------------- #

def test_an_unmeasured_term_is_absent_rather_than_zero():
    """A term nothing could measure must not read as a term that measured badly."""
    vector = _vector(G=0.8, T=0.6)
    assert vector.absent == ["C", "N", "R", "E", "H", "U"]
    assert vector.get("C").value is None
    assert vector.get("C").measured is False


def test_a_missing_term_does_not_zero_the_scalar():
    """The failure mode a raw product would have: one unmeasured term erasing the whole reading."""
    assert _vector(G=0.8, T=0.8).scalar == pytest.approx(0.8)


def test_a_measured_zero_does_zero_the_scalar():
    """The opposite case, and it must survive: "she cannot generalise at all" is a real result."""
    assert _vector(G=0.0, T=1.0).scalar == 0.0


def test_the_scalar_says_which_terms_it_was_computed_over():
    got = _vector(G=0.5, T=0.5).to_dict()
    assert got["measured_over"] == ["G", "T"]
    assert set(got["absent"]) == {"C", "N", "R", "E", "H", "U"}


def test_a_run_that_gained_a_term_is_still_comparable_to_the_one_before_it():
    """Why the numerator is a geometric mean and not the product it was sketched as.

    A raw product changes magnitude when a term appears, so a four-term run and a five-term run
    would not be the same kind of number — which defeats the point of a time series.
    """
    two = _vector(G=0.8, T=0.8).scalar
    three = _vector(G=0.8, T=0.8, C=0.8).scalar
    assert two == pytest.approx(three)
    assert _vector(G=1.0, T=1.0, C=1.0, R=1.0).scalar == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The denominator
# --------------------------------------------------------------------------- #

def test_being_wrong_lowers_the_score():
    clean = _vector(G=1.0, T=1.0).scalar
    assert _vector(G=1.0, T=1.0, E=0.5).scalar < clean
    assert _vector(G=1.0, T=1.0, H=0.5).scalar < clean
    assert _vector(G=1.0, T=1.0, U=0.5).scalar < clean


def test_a_perfect_run_with_no_penalties_is_exactly_one():
    """`1 + penalties` rather than the penalties alone, so a clean run is defined and readable."""
    assert _vector(G=1.0, T=1.0, E=0.0, H=0.0, U=0.0).scalar == pytest.approx(1.0)


def test_not_knowing_is_charged_nowhere():
    """The rule that keeps the objective from fighting the architecture.

    A system fined for saying "I don't know" has one gradient available: say it less. There is no
    uncertainty term in the denominator, and the closest thing — unsupported confidence — moves in
    the opposite direction, charging her for being *more* sure than she earned.
    """
    keys = {t.key for t in _vector().terms}
    assert "uncertainty" not in keys
    assert set(DENOMINATOR) == {"E", "H", "U"}


def test_under_confidence_is_not_credited(taught):
    """`bias` is signed; only the overconfident half is charged. Rewarding hedging is not the goal."""
    index = IntelligenceIndex()

    class _Rel:
        bias, samples = -0.4, 20

    brain = type("_B", (), {"beliefs": type("_Bel", (), {"reliability": lambda *a: _Rel()})()})()
    assert index._unsupported(brain).value == 0.0


# --------------------------------------------------------------------------- #
# Where the terms come from
# --------------------------------------------------------------------------- #

def test_every_term_names_its_source(taught):
    vector = measure(taught, width=2)
    assert {t.key for t in vector.terms} == set(NUMERATOR) | set(DENOMINATOR)
    for term in vector.terms:
        assert term.source, f"{term.key} does not say where it came from"
        assert term.detail, f"{term.key} does not say how it was computed or why it is missing"


def test_no_term_is_read_from_her_own_self_model(taught):
    """The rule that stops the cheapest path to a better score being to change the scoring."""
    sources = " ".join(t.source for t in measure(taught, width=2).terms)
    assert "selfmodel" not in sources and "self_model" not in sources


def test_transfer_is_credited_against_its_own_control(taught):
    """`run_structure_transfer` carries a base-only baseline; ignoring it would waste the control."""
    term = IntelligenceIndex()._transfer()
    assert term.value is not None
    assert "control" in term.detail


def test_novel_task_solving_is_absent_and_says_why():
    """Rather than pointed at a stage that already serves G or T, which would report one
    measurement twice as though it were two."""
    term = IntelligenceIndex()._novel()
    assert term.value is None
    assert "no independent source" in term.detail


def test_a_refusal_is_not_counted_as_a_hallucination(taught):
    """A refusal is the gauntlet working. Charging for it would penalise what prevents them."""
    term = IntelligenceIndex()._hallucination(taught)
    assert "refused" not in term.source


def test_a_broken_source_leaves_its_term_absent_and_the_rest_measured():
    class _Broken:
        def __getattr__(self, name):
            raise RuntimeError("organ is down")

    vector = measure(_Broken(), benchmarks=False)
    assert vector.get("C").value is None
    assert "unavailable" in vector.get("C").detail
    assert len(vector.terms) == 8


def test_measuring_without_a_brain_still_returns_the_whole_vector():
    vector = measure(None, benchmarks=False)
    assert len(vector.terms) == 8
    assert vector.scalar is None


# --------------------------------------------------------------------------- #
# The standing rule
# --------------------------------------------------------------------------- #

def test_a_term_that_got_worse_is_named_in_the_right_direction():
    before = _vector(G=0.8, T=0.8, E=0.1)
    after = _vector(G=0.5, T=0.9, E=0.3)
    got = after.regressions(before)
    assert set(got) == {"G", "E"}, "G fell and E rose; T rose and must not be flagged"
    assert got["G"] == {"was": 0.8, "now": 0.5}


def test_a_scalar_that_rose_while_generalization_fell_is_still_a_regression():
    """The rule the whole index exists to enforce: read the vector, not the mean.

    A rung that raises the dashboard while G and T stay flat has taught her to score better rather
    than to think better, and this is the check that says so.
    """
    before = _vector(G=0.6, T=0.6, E=0.9)
    after = _vector(G=0.4, T=0.6, E=0.0)
    assert after.scalar > before.scalar
    assert "G" in after.regressions(before)


def test_the_vector_leads_the_report_and_the_scalar_is_one_line(taught):
    got = measure(taught, width=2).to_dict()
    assert list(got)[0] == "terms"
    assert len(got["terms"]) == 8
    assert "scalar" in got


def test_the_reading_is_the_same_twice_for_a_seed(taught):
    first = measure(taught, seed=3, width=2).to_dict()
    second = measure(taught, seed=3, width=2).to_dict()
    assert [t["value"] for t in first["terms"]] == [t["value"] for t in second["terms"]]


def test_it_can_measure_a_brain_that_was_fed_a_corpus(tmp_path):
    """The whole reason Stage 4 came before this one: G must be readable on a prepared brain."""
    import json

    from nyxara.njp.ingest import ingest_triples

    path = tmp_path / "cn.jsonl"
    path.write_text("\n".join(json.dumps(
        {"subject": f"s{i}", "predicate": "is_a", "object": "thing"}) for i in range(20)),
        encoding="utf-8")

    seen = []

    def prepare(brain):
        ingest_triples(brain, path, source="conceptnet")
        seen.append(len(brain.grounder.facts))

    term = IntelligenceIndex(width=2)._generalization(prepare)
    assert seen and seen[0] == 20
    assert term.source.endswith("generalization")
