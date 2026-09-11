"""The loop pointed at a real organ — the parts of it that run without the corpus.

:mod:`nyxara.njp.fieldwork` is a report over the reading corpus, so its headline run is not a test.
What *is* testable here, and is what the version turns on, is the bookkeeping: that a stage reports
its pool and its ceiling honestly, that the repair which failed is kept exactly as it failed, and
that the benchmark handed to the attributor carries the ``reachable`` the V.83 guard reads.
"""

from __future__ import annotations

import pytest

from nyxara.njp.fieldwork import (
    HELD, LEARN_FROM, MORE, SMALL_POOL, Fieldwork, Stage, _cut, _shortest, span_stage,
)
from nyxara.njp.finding import Reading, Span, candidates
from nyxara.njp.findingschool import taught_finder


@pytest.fixture(scope="module")
def little():
    """Four passages whose answer really is a span of them, so the stage has something to find."""
    return [
        Reading(passage="The capital of France is Paris. It sits on the Seine.",
                question="What is the capital of France?", answer="Paris"),
        Reading(passage="Ada Lovelace wrote the first algorithm. She died in 1852.",
                question="Who wrote the first algorithm?", answer="Ada Lovelace"),
        Reading(passage="Copper conducts electricity well. Rubber does not.",
                question="What conducts electricity well?", answer="Copper"),
        Reading(passage="The treaty was signed in Vienna. Many states attended.",
                question="Where was the treaty signed?", answer="Vienna"),
    ]


@pytest.fixture(scope="module")
def engine(little):
    return taught_finder(little, use_shape=False)


def test_a_stage_reports_its_pool_and_its_ceiling(engine, little):
    got = span_stage(engine, little)
    assert got.rows == len(little)
    assert got.pool > 1.0, "a pool of one is not a choice, and this stage is about choosing"
    assert 0.0 <= got.reachable <= 1.0
    assert got.bench is not None and got.bench.reachable is not None


def test_the_benchmark_carries_what_the_guard_reads(engine, little):
    """The V.83 ceiling guard is silent without ``reachable``, and silence would read as pass."""
    from nyxara.njp.attribution import _ceiling_of

    got = span_stage(engine, little)
    assert _ceiling_of(got.bench) == pytest.approx(got.reachable, abs=1e-4)


def test_the_repair_that_failed_is_kept_as_it_failed(engine, little):
    """Twelve shortest candidates. It is in the module because it did not work, not in spite of it.

    The two numbers it moved are the point: the pool shrinks, **and** so does the share of items
    whose right answer is still in it. An experiment that does both tests neither.
    """
    whole = span_stage(engine, little)
    capped = span_stage(engine, little, name="a smaller pool", pool=_shortest)
    assert capped.pool <= SMALL_POOL
    assert capped.pool < whole.pool
    assert capped.reachable <= whole.reachable


def test_shortest_keeps_the_shortest_and_no_more():
    spans = [Span(text="x" * n, start=0, end=n) for n in range(1, 40)]
    got = _shortest(spans, Reading())
    assert len(got) == SMALL_POOL
    assert [len(s.text) for s in got] == list(range(1, SMALL_POOL + 1))


def test_a_stage_drops_what_it_cannot_score_rather_than_counting_it_wrong(engine):
    """A passage with no candidates in its gold sentence is not a failure of the ranker."""
    got = span_stage(engine, [Reading(passage="...", question="?", answer="nothing")])
    assert got.rows == 0
    assert got.bench is not None and got.bench.score() == 0.0


def test_the_generator_really_does_offer_many_spans_per_sentence():
    """The quantity the hand-diagnosis was about, and the one no hypothesis in the organ tests."""
    got = candidates("The capital city of the French Republic is Paris on the river Seine.")
    assert len(got) > 20


def test_an_empty_fieldwork_claims_nothing():
    blank = Fieldwork()
    assert blank.as_found is None
    assert blank.to_dict()["root"] == "" and blank.to_dict()["spoiled"] == []
    assert blank.render() == ""
    assert Stage(name="unrun").to_dict() == {"name": "unrun", "pool": 0.0, "reachable": 0.0,
                                             "rows": 0, "learned_from": 0, "score": 0.0}
    assert HELD > 0 and SMALL_POOL > 0


def test_more_data_really_is_more_data(little):
    """The defect that nearly shipped: an experiment asking for twice the rows and getting the same.

    ``findingschool.split`` caps its learn side, so ``learn[:LEARN_FROM * 2]`` through it returns
    ``learn[:LEARN_FROM]`` — and the flat result would have been recorded as *more examples would
    not have helped*, which the experiment never tested. :func:`_cut` is the uncapped split, and
    this pins the property the cap broke.
    """
    assert MORE > LEARN_FROM
    many = [Reading(passage=f"Item {n} is called thing{n}. It sits still.",
                    question=f"What is item {n} called?", answer=f"thing{n}")
            for n in range(200)]
    learn, held = _cut(many)
    assert learn and held
    assert not (set(r.passage for r in learn) & set(r.passage for r in held))
    assert len(learn) + len(held) == len(many)
    assert len(learn[:50]) != len(learn[:120]), "the cut must not cap below what a repair asks for"


def test_diagnose_refuses_rather_than_running_an_experiment_that_changes_nothing(little):
    """Too few rows to show the reader more of them is a reason to stop, not to report a flat line."""
    with pytest.raises(ValueError, match="same data"):
        span_stage  # keep the import honest
        from nyxara.njp.fieldwork import diagnose

        diagnose(little)
