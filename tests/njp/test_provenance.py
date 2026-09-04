"""Every conclusion carries the path that made it (NJP V.43)."""
from __future__ import annotations

import pytest

from nyxara.njp.provenance import Kind, Ledger, Status, Step


@pytest.fixture
def ledger():
    return Ledger()


def _steps(ledger):
    fact = ledger.record(Step("f183", Kind.FACT, "cloud causes rain"))
    other = ledger.record(Step("f921", Kind.FACT, "condensation causes cloud"))
    rule = ledger.record(Step("R4", Kind.INFERENCE, "transitivity"))
    guess = ledger.record(Step("A3", Kind.ASSUMPTION, "nothing else intervenes", settled=False))
    return fact, other, rule, guess


# --------------------------------------------------------------------------- #
# No path, no claim
# --------------------------------------------------------------------------- #
def test_a_conclusion_with_nothing_under_it_is_unknown(ledger):
    assert ledger.status("nothing") is Status.UNKNOWN
    assert "nothing supports this" in ledger.render("nothing")


def test_a_supported_claim_names_everything_it_stands_on(ledger):
    fact, other, rule, _guess = _steps(ledger)
    ledger.assert_("c1", "condensation causes rain", path=[other, fact, rule])
    text = ledger.render("c1")
    assert ledger.status("c1") is Status.SUPPORTED
    assert "f183" in text and "f921" in text and "R4" in text


def test_an_unsettled_step_makes_the_claim_hypothetical_and_it_says_so(ledger):
    fact, _o, _r, guess = _steps(ledger)
    ledger.assert_("c2", "it will rain", path=[fact, guess])
    assert ledger.status("c2") is Status.HYPOTHETICAL
    assert "hypothetical" in ledger.render("c2")


def test_one_clean_route_beside_a_speculative_one_is_still_supported(ledger):
    """Otherwise an idle guess downgrades an established conclusion."""
    fact, other, rule, guess = _steps(ledger)
    ledger.assert_("c3", "it will rain", path=[other, fact, rule])
    ledger.assert_("c3", "it will rain", path=[fact, guess])
    assert ledger.status("c3") is Status.SUPPORTED


def test_two_paths_that_cannot_both_hold_are_reported_never_counted(ledger):
    fact, other, rule, guess = _steps(ledger)
    ledger.assert_("c4", "it will rain", path=[other, fact, rule])
    ledger.assert_("c4", "it will rain", path=[fact, guess])
    ledger.oppose("c4", 0, 1)
    assert ledger.status("c4") is Status.CONFLICTED
    assert "CONFLICTING PATHS" in ledger.render("c4")


def test_a_status_is_never_upgraded_by_usefulness(ledger):
    fact, _o, _r, guess = _steps(ledger)
    ledger.assert_("c5", "it will rain", path=[fact, guess])
    for _ in range(5):
        ledger.audit("c5")
    assert ledger.status("c5") is Status.HYPOTHETICAL
    ledger.confirm("A3", by="a measurement")
    assert ledger.status("c5") is Status.SUPPORTED
    assert "settled by a measurement" in ledger.render("c5")


# --------------------------------------------------------------------------- #
# Blame is not the path
# --------------------------------------------------------------------------- #
def test_blame_finds_the_step_the_claim_actually_rested_on(ledger):
    fact, other, rule, guess = _steps(ledger)
    ledger.assert_("standing", "condensation causes rain", path=[other, fact, rule])
    ledger.assert_("failed", "it will rain", path=[fact, guess])
    blamed = ledger.blame("failed")
    assert blamed[0].step.id == "A3", "the fact also supports a standing claim; the guess does not"


def test_a_step_supporting_other_standing_claims_is_blamed_less(ledger):
    fact, other, rule, guess = _steps(ledger)
    for n in range(3):
        ledger.assert_(f"other{n}", "something else", path=[fact, rule])
    ledger.assert_("failed", "it will rain", path=[fact, guess])
    ranked = {b.step.id: b for b in ledger.blame("failed")}
    assert ranked["A3"].share > ranked["f183"].share
    assert ranked["f183"].also_supports == 3 and ranked["A3"].also_supports == 0
    assert ranked["A3"].exclusive is True


def test_blaming_nothing_when_there_is_no_claim(ledger):
    assert ledger.blame("never recorded") == []


# --------------------------------------------------------------------------- #
# The autopsy, and the warning it earns
# --------------------------------------------------------------------------- #
def test_a_failure_is_stored_with_everything_needed_to_act_on_it(ledger):
    fact, _o, _r, guess = _steps(ledger)
    ledger.assert_("failed", "it will rain", path=[fact, guess])
    got = ledger.autopsy("failed", predicted=True, actual=False,
                         missing="tonight's cloud cover", repair="check the assumption")
    assert got.culprit is not None and got.culprit.id == "A3"
    text = got.render()
    for wanted in ("predicted", "actual", "blamed", "path", "missing", "repair"):
        assert wanted in text


def test_a_warning_matches_on_the_blamed_step_not_on_the_wording(ledger):
    """Two questions that look nothing alike and failed on the same assumption are one failure."""
    fact, _o, _r, guess = _steps(ledger)
    ledger.assert_("first", "it will rain tonight", path=[fact, guess])
    ledger.autopsy("first", predicted=True, actual=False)
    # a completely different question, reasoned through the same assumption
    assert [w.claim for w in ledger.warn([guess])] == ["it will rain tonight"]
    assert ledger.warn([fact]) == []


def test_the_warning_appears_in_the_audit(ledger):
    fact, _o, _r, guess = _steps(ledger)
    ledger.assert_("first", "it will rain", path=[fact, guess])
    ledger.autopsy("first", predicted=True, actual=False)
    assert "WARNING" in ledger.render("first")


# --------------------------------------------------------------------------- #
# Wired into the loop
# --------------------------------------------------------------------------- #
def test_the_loops_prediction_is_auditable():
    from nyxara.njp.loop import Loop, Model

    loop = Loop()
    confounded = Model(edges=frozenset({("h", "a"), ("h", "b")}), observed=("a", "b"))
    loop.forecast([confounded], "a", "b")
    text = loop.audit("a", "b")
    assert "CLAIM" in text and "SUPPORTED BY" in text
    assert "hypothetical" in text, "a prediction resting on a hidden cause must say so"
    assert "R:mutilate" in text, "the inference that produced it is part of the path"


def test_a_loop_failure_warns_the_next_time_the_same_ground_is_used():
    from nyxara.njp.loop import Loop, Model

    loop = Loop()
    direct = Model(edges=frozenset({("a", "b")}), observed=("a", "b"))
    confounded = Model(edges=frozenset({("h", "a"), ("h", "b")}), observed=("a", "b"))
    loop.forecast([direct], "a", "b")
    loop.revise([direct, confounded], "a", "b", False)
    assert loop.ledger.failures
    assert "WARNING" in loop.audit("a", "b")
