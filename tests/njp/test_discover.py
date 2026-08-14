"""She proposes her own abstractions, then tries to kill them.

The middle step is what makes this science rather than pattern-matching: a hypothesis is scored on
episodes it was never fitted to. Anything can find patterns in data it was fitted to.
"""

from __future__ import annotations

import pytest

from nyxara.njp.discover import Discoverer, Status


def _weather() -> Discoverer:
    """A real rule (clouds+wind+cold -> rain) and a spurious one (birds -> rain) that dies."""
    d = Discoverer(min_support=3, holdout=0.4)
    for _ in range(20):
        d.observe({"clouds", "wind", "cold"}, "rain")
    for _ in range(6):
        d.observe({"birds", "sun"}, "rain")
    for _ in range(20):
        d.observe({"birds", "sun"}, "clear")
    return d


def test_a_real_pattern_is_discovered():
    d = _weather()
    d.discover()
    rules = {a.consequent for a in d.confirmed()}
    assert "rain" in rules


def test_a_confirmed_rule_held_up_on_evidence_it_never_saw():
    d = _weather()
    d.discover()
    for a in d.confirmed():
        assert a.tested >= d.min_support, "confirmed without being tested"
        assert a.precision >= d.min_precision


def test_a_spurious_pattern_is_refuted_and_deleted():
    """A refuted rule left in place quietly poisons every later prediction that consults it."""
    d = _weather()
    report = d.discover()
    assert report.refuted >= 1
    assert not any(a.consequent == "rain" and "birds" in a.antecedents
                   for a in d.abstractions.values())


def test_an_untested_hypothesis_never_predicts():
    """Conjectures are not usable. Only what survived falsification may fire."""
    d = Discoverer(min_support=3, holdout=0.4)
    for _ in range(4):
        d.observe({"a", "b"}, "c")
    d.discover()
    for a in d.abstractions.values():
        if a.status == Status.CONJECTURED:
            assert not a.usable
    assert d.predict({"a", "b"}) == [] or all(
        s > 0.0 for _c, s in d.predict({"a", "b"}))


def test_compression_is_measured_not_claimed():
    d = _weather()
    report = d.discover()
    assert report.compression > 1.0, "no compression actually happened"


def test_no_abstractions_means_no_compression_claim():
    """1.0 means she compressed nothing, and that is what must be reported."""
    assert Discoverer().compression() == 1.0


def test_the_split_covers_every_regime():
    """A positional split leaves an early-regime rule with no relevant held-out episodes at all,
    so it is never tested and a good pattern goes unused. Measured: it confirmed nothing."""
    d = _weather()
    fit, held = d._split()
    assert fit and held
    fit_antecedents = {a for items, _ in fit for a in items}
    held_antecedents = {a for items, _ in held for a in items}
    assert "clouds" in held_antecedents and "birds" in held_antecedents
    assert fit_antecedents == held_antecedents


def test_a_rule_is_only_tested_where_it_applies():
    """An episode the rule says nothing about is not evidence either way — counting those would
    let a narrow rule look perfect by staying silent."""
    d = Discoverer(min_support=2, holdout=0.5)
    for _ in range(10):
        d.observe({"x", "y"}, "z")
    for _ in range(10):
        d.observe({"p", "q"}, "r")
    d.discover()
    for a in d.abstractions.values():
        assert a.tested <= 5, "scored against episodes it does not apply to"


def test_higher_order_rules_are_proposed():
    """{a,b,c} -> d says more than {a} -> d and stands in for more observations."""
    d = _weather()
    d.discover()
    assert any(a.order >= 3 for a in d.abstractions.values())


def test_prediction_uses_the_strongest_applicable_rule():
    d = _weather()
    d.discover()
    got = d.predict({"clouds", "wind", "cold"})
    assert got and got[0][0] == "rain"


def test_an_empty_observation_is_ignored():
    d = Discoverer()
    d.observe([], "outcome")
    d.observe({"a"}, "")
    assert not d.episodes


def test_discovery_survives_a_reload():
    d = _weather()
    d.discover()
    fresh = Discoverer(min_support=3, holdout=0.4)
    fresh.load_dict(d.to_dict())
    assert {a.name for a in fresh.confirmed()} == {a.name for a in d.confirmed()}


def test_a_pass_with_no_evidence_changes_nothing():
    report = Discoverer().discover()
    assert not report.changed and report.compression == 1.0


# ---- wired ------------------------------------------------------------------- #
def test_the_brain_records_episodes_for_discovery():
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    brain.think("my name is Jay")
    assert len(brain.discoverer.episodes) == 1


def test_a_turn_with_no_conclusion_records_no_episode():
    """One episode is never a pattern, and an empty one is not even an episode."""
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    brain.think("what is my pin code")
    assert not brain.discoverer.episodes


def test_an_organ_that_is_off_is_absent():
    from nyxara.njp.brain import NJPBrain

    class _Off:
        discover_enabled = False

    brain = NJPBrain(_Off())
    assert brain.discoverer is None
    assert "discover" not in brain.stats()
