"""A single new fact may not corrupt the graph (NJP V.44).

The organ exists because of a measured regression, so the tests are mostly about the three cases
it must get right and the two it must not touch.
"""
from __future__ import annotations

import pytest

from nyxara.njp.immune import COMPETING, MANY_VALUED, Immune, Verdict


@pytest.fixture
def guard():
    got = Immune(trusted={"curated"})
    got.note("sun", "is_a", "star", source="curated")
    got.note("dog", "causes", "barking", source="curated")
    return got


# --------------------------------------------------------------------------- #
# What is isolated
# --------------------------------------------------------------------------- #
def test_a_competing_claim_from_an_unproven_source_is_quarantined(guard):
    verdict, why = guard.consider("sun", "is_a", "celestial body", source="crowd")
    assert verdict is Verdict.QUARANTINED
    assert "unanswerable" in why
    assert guard.held_against("sun", "is_a")


def test_quarantine_is_not_rejection(guard):
    guard.consider("sun", "is_a", "celestial body", source="crowd")
    held = guard.held_against("sun", "is_a")
    assert held and held[0].object == "celestial body"
    assert held[0].incumbent == ("star",), "it remembers what it would have crowded"


def test_a_released_claim_credits_its_source(guard):
    guard.consider("sun", "is_a", "celestial body", source="crowd")
    antigen = guard.held_against("sun", "is_a")[0]
    guard.release(antigen, why="an encyclopaedia agrees")
    assert guard.quarantine == []
    assert guard.sources["crowd"].confirmed == 1 and guard.standing("crowd") == 1.0


def test_a_rejected_claim_debits_its_source(guard):
    guard.consider("sun", "is_a", "celestial body", source="crowd")
    guard.reject(guard.held_against("sun", "is_a")[0], why="checked and wrong")
    assert guard.sources["crowd"].refuted == 1 and guard.standing("crowd") < 0


# --------------------------------------------------------------------------- #
# What is admitted, and it is most of it
# --------------------------------------------------------------------------- #
def test_a_new_subject_cannot_damage_an_answer_that_does_not_exist(guard):
    verdict, why = guard.consider("quark", "is_a", "particle", source="crowd")
    assert verdict is Verdict.ADMITTED and "nothing here to damage" in why


@pytest.mark.parametrize("relation", MANY_VALUED)
def test_a_many_valued_relation_is_never_guarded(guard, relation):
    """A second `causes` is a second cause. Guarding it would reject richness and buy nothing."""
    guard.note("dog", relation, "first", source="curated")
    verdict, _why = guard.consider("dog", relation, "second", source="crowd")
    assert verdict is Verdict.ADMITTED
    assert relation not in COMPETING


def test_saying_the_same_thing_again_is_corroboration_not_a_collision(guard):
    verdict, why = guard.consider("sun", "is_a", "star", source="crowd")
    assert verdict is Verdict.ADMITTED and "corroboration" in why


def test_isolation_requires_something_to_protect():
    """No incumbent, no quarantine — whatever the source."""
    bare = Immune()
    verdict, _why = bare.consider("anything", "is_a", "whatever", source="nobody")
    assert verdict is Verdict.ADMITTED


# --------------------------------------------------------------------------- #
# Standing is earned
# --------------------------------------------------------------------------- #
def test_an_unproven_source_starts_at_zero_not_at_parity():
    """Starting at a half would let sheer volume outrank a corpus examined for twenty versions."""
    got = Immune()
    assert got.standing("brand new") == 0.0


def test_a_source_that_has_earned_standing_displaces_the_incumbent():
    """The response runs in both directions, or the organ is only conservatism."""
    got = Immune()
    got.note("sun", "is_a", "planet", source="sloppy")
    got.sources.setdefault("careful", __import__(
        "nyxara.njp.immune", fromlist=["Standing"]).Standing(source="careful", confirmed=5))
    verdict, why = got.consider("sun", "is_a", "star", source="careful")
    assert verdict is Verdict.ADMITTED and "outranks" in why
    displaced = [a for a in got.quarantine if a.object == "planet"]
    assert displaced and "displaced" in displaced[0].reason


def test_a_trusted_source_is_the_only_declaration_in_the_file():
    got = Immune(trusted={"curated"})
    assert got.standing("curated") == 1.0 and got.standing("anyone else") == 0.0


# --------------------------------------------------------------------------- #
# Filtering a corpus
# --------------------------------------------------------------------------- #
def test_filtering_keeps_the_admitted_and_holds_the_rest(guard):
    rows = [{"subject": "sun", "predicate": "is_a", "object": "celestial body"},
            {"subject": "quark", "predicate": "is_a", "object": "particle"},
            {"subject": "dog", "predicate": "causes", "object": "joy"}]
    kept = list(guard.filter_triples(rows, source="crowd"))
    assert len(kept) == 2
    assert {r["subject"] for r in kept} == {"quark", "dog"}
    assert len(guard.quarantine) == 1


def test_the_report_accounts_for_every_row(guard):
    rows = [{"subject": f"s{i}", "predicate": "is_a", "object": "thing"} for i in range(10)]
    rows.append({"subject": "sun", "predicate": "is_a", "object": "rock"})
    list(guard.filter_triples(rows, source="crowd"))
    report = guard.report()
    assert report["verdicts"]["admitted"] + report["verdicts"]["quarantined"] == 11


# --------------------------------------------------------------------------- #
# The regression it was built for
# --------------------------------------------------------------------------- #
def test_the_broad_load_goes_through_the_guard_by_default():
    import inspect

    from nyxara.njp import general

    assert inspect.signature(general.load_brain).parameters["immune"].default is True


def test_the_guard_keeps_every_subject_it_was_given():
    """Breadth and precision, not a trade: the harm is concentrated in a small fraction."""
    guard = Immune(trusted={"curated"})
    guard.note("sun", "is_a", "star", source="curated")
    rows = [{"subject": f"thing{i}", "predicate": "is_a", "object": f"kind{i}"}
            for i in range(200)]
    rows += [{"subject": "sun", "predicate": "is_a", "object": "rock"}]
    kept = list(guard.filter_triples(rows, source="crowd"))
    assert len(kept) == 200, "only the colliding claim is held"
    assert len(guard.quarantine) == 1
