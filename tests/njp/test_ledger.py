"""The growth record. If growth is flat it has to show flat."""

from __future__ import annotations

from nyxara.njp.ledger import Ledger


def _gen(led: Ledger, cells: int, synapses: int, accuracy: float) -> None:
    led.record(fabric_stats={"cells": cells, "synapses": synapses,
                             "mean_degree": synapses / max(cells, 1)},
               accuracy=accuracy)


def test_nothing_to_compare_returns_none_not_false():
    """'No evidence yet' and 'no regression' are different facts, and the self-editor
    must not read the first as the second and keep an edit it never checked."""
    led = Ledger()
    assert led.regressed() is None
    _gen(led, 10, 20, 0.5)
    assert led.regressed() is None


def test_a_fall_in_accuracy_is_reported_as_a_regression():
    led = Ledger()
    _gen(led, 10, 20, 0.80)
    _gen(led, 12, 25, 0.40)
    assert led.regressed() is True
    assert led.report()["more_accurate"] is False


def test_growth_shows_as_growth():
    led = Ledger()
    _gen(led, 10, 20, 0.5)
    _gen(led, 14, 33, 0.6)
    report = led.report()
    assert report["grew"] is True
    assert report["delta"]["synapses"] == 13
    assert report["delta"]["cells"] == 4


def test_flat_growth_shows_as_flat():
    """A record that could only ever report improvement would be worthless."""
    led = Ledger()
    _gen(led, 10, 20, 0.5)
    _gen(led, 10, 20, 0.5)
    assert led.report()["grew"] is False


def test_errors_are_recalled_by_shape():
    led = Ledger()
    led.errors.record("the parser mishandles unicode", verdict="refuted")
    assert led.errors.similar("the parser mishandles unicode input")
    assert not led.errors.similar("the weather in Mumbai is humid")


def test_the_record_survives_a_round_trip():
    led = Ledger()
    _gen(led, 10, 20, 0.5)
    _gen(led, 14, 33, 0.6)
    led.errors.record("a wrong claim", verdict="refuted")
    reborn = Ledger()
    reborn.load_dict(led.to_dict())
    assert len(reborn.generations) == 2
    assert reborn.report()["grew"] is True
    assert reborn.errors.similar("a wrong claim")
