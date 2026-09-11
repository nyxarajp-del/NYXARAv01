"""The loop pointed at a real organ — the parts of it that run without the corpus.

:mod:`nyxara.njp.fieldwork` is a report over the reading corpus, so its headline run is not a test.
What *is* testable here, and is what the version turns on, is the bookkeeping: that a stage reports
its pool and its ceiling honestly, that the repair which failed is kept exactly as it failed, and
that the benchmark handed to the attributor carries the ``reachable`` the V.83 guard reads.
"""

from __future__ import annotations

import pytest

from nyxara.njp.fieldwork import (
    HELD, LEARN_FROM, MORE, SMALL_POOL, SPAN_KNOBS, Fieldwork, Stage, _cut, _shortest, _thinned,
    span_stage,
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


def test_a_stage_supplies_what_the_leakage_check_needs(engine, little):
    """The check that went unrun in the first field run, and should not have.

    ``? leakage — no key and train supplied`` is the right answer for a critic handed nothing and
    the wrong thing for a caller to have caused: the cut is by passage, so the answer was available
    the whole time. Identity is the **passage**, since two questions about one paragraph are not
    two independent items.
    """
    from nyxara.njp.measurement import critique

    got = span_stage(engine, little, taught=little[:2])
    found = next(f for f in critique(got.bench).findings if f.check == "leakage")
    assert found.informative, "the check must actually run"
    assert found.verdict == "broken", "the first two passages really were learned from"
    assert found.got == pytest.approx(0.5, abs=1e-4)


def test_a_clean_cut_reports_leakage_clear_rather_than_unchecked(engine, little):
    from nyxara.njp.measurement import critique

    got = span_stage(engine, little[2:], taught=little[:2])
    found = next(f for f in critique(got.bench).findings if f.check == "leakage")
    assert found.informative and found.verdict == "clear"


# --------------------------------------------------------------------------------------------- #
#  V.84 — the experiment V.83 ended owing
# --------------------------------------------------------------------------------------------- #
def test_thinning_the_pool_keeps_what_it_is_thinning_toward(engine, little):
    """`_shortest` asked the question and could not answer it. This one can.

    Same twelve candidates, same ranker, same items — and the gold span forced in among them, so
    the **only** thing that differs from `as found` is how many wrong candidates compete. That is
    the quantity the hand-diagnosis was about and the one no hypothesis in `attribution` tests.
    """
    whole = span_stage(engine, little)
    capped = span_stage(engine, little, name="capped", pool=_shortest)
    kept = span_stage(engine, little, name="kept", pool=_thinned)
    assert kept.pool <= SMALL_POOL, "it must actually thin the pool"
    assert kept.pool < whole.pool
    assert kept.reachable == pytest.approx(whole.reachable, abs=1e-4), \
        "the ceiling is what `_shortest` dropped and this must not"
    assert kept.reachable > capped.reachable or capped.reachable == whole.reachable


def test_thinning_keeps_the_gold_span_specifically():
    """Not merely the same rate — the same items. A rate that matches by luck is not holding."""
    from nyxara.njp.finding import Span, gold_key

    reading = Reading(passage="The capital of France is Paris.",
                      question="What?", answer="Paris")
    spans = [Span(text="x" * n, start=0, end=n) for n in range(1, 40)]
    spans.append(Span(text="Paris", start=0, end=5))
    got = _thinned(spans, reading)
    assert len(got) <= SMALL_POOL
    assert any(s.key == gold_key(reading.answer) for s in got), \
        "the gold span was not among the twelve shortest, so it had to be forced in"


def test_thinning_does_not_invent_an_answer_that_was_never_there():
    """When the gold is not in the pool at all, forcing it in would be fabricating reachability."""
    from nyxara.njp.finding import Span, gold_key

    reading = Reading(passage="nothing here", question="What?", answer="Paris")
    spans = [Span(text="x" * n, start=0, end=n) for n in range(1, 40)]
    got = _thinned(spans, reading)
    assert len(got) == SMALL_POOL
    assert not any(s.key == gold_key(reading.answer) for s in got)


def test_the_span_map_spans_more_regions_than_the_experiments_do():
    """The whole reason to draw a map: four experiments in two regions is visible here."""
    regions = {k.region for k in SPAN_KNOBS}
    assert {"built", "scored"} <= regions
    assert len([k for k in SPAN_KNOBS if k.region == "built"]) >= 2
