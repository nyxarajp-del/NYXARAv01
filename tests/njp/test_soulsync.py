"""The Intent-Refinement Loop: the half the Master did not say."""

from __future__ import annotations

from nyxara.njp.soulsync import SoulSync


def _teach(s: SoulSync, n: int = 2) -> None:
    for _ in range(n):
        s.observe_correction("fix the tests", "fix the tests run them first")


def test_it_knows_nothing_on_day_one_and_says_so():
    """No flattering default. 'No evidence yet' must not look like 'perfectly in sync'."""
    s = SoulSync()
    reading = s.read("fix the tests")
    assert reading.latent == []
    assert s.resonance() is None


def test_a_correction_becomes_a_standing_want():
    s = SoulSync(min_support=2)
    _teach(s, 2)
    reading = s.read("fix the tests")
    assert reading.has_latent
    assert "run them first" in reading.latent[0].want


def test_one_correction_is_not_yet_a_rule():
    """At support 1 an off-hand remark would become permanent. The floor is deliberate."""
    s = SoulSync(min_support=2)
    _teach(s, 1)
    assert s.read("fix the tests").latent == []


def test_a_want_carries_to_a_request_never_seen_before():
    """The point: she applies it without being told again, on new work."""
    s = SoulSync(min_support=2)
    _teach(s, 2)
    reading = s.read("fix the tests in the parser module")
    assert reading.has_latent


def test_removing_words_does_not_teach_a_rule():
    """A narrowed request is narrowed for that instance; it is not a standing preference."""
    s = SoulSync(min_support=1)
    s.observe_correction("fix the tests and push", "fix the tests")
    assert s.read("fix the tests and push").latent == []


def test_being_corrected_anyway_costs_a_preference_its_standing():
    """Without this a preference can only ever gain confidence — that is how a wrong rule sticks."""
    s = SoulSync(min_support=2)
    _teach(s, 2)
    s.read("fix the tests")
    before = s.preferences[0].confidence
    s.confirm(corrected=True)
    s.read("fix the tests")
    s.confirm(corrected=True)
    assert s.preferences[0].confidence < before


def test_it_anticipates_what_usually_comes_next():
    s = SoulSync(min_support=2, trust_floor=0.4)
    for _ in range(4):
        s.read("fix the parser")
        s.read("commit the change")
    got = s.anticipate()
    assert got.description
    assert got.confidence > 0.0


def test_an_unearned_anticipation_is_untrusted_with_a_reason():
    s = SoulSync()
    s.read("something entirely new")
    got = s.anticipate()
    assert got.trusted is False
    assert got.reason


def test_preferences_survive_a_round_trip():
    s = SoulSync(min_support=2)
    _teach(s, 3)
    reborn = SoulSync(min_support=2)
    reborn.load_dict(s.to_dict())
    assert reborn.read("fix the tests").has_latent
