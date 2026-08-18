"""The intelligence benchmark could not see anything she had been fed.

Every stage builds its own brain — deliberately, so no stage can be carried by what another one
taught. The consequence nobody had noticed: anything done to a brain *outside* the benchmark was
invisible to it. Feeding a quarter-million ConceptNet facts and then running this measured six
fresh brains that had never seen them, and reported the fresh-brain score under a heading that
implied otherwise.

The `prepare` hook fixes that, and the first thing it did once it existed was find a defect in
`njp/ingest.py`'s defaults — which is the whole reason to build an instrument before building
what it is meant to measure. That finding is pinned here as a regression test rather than left in
a commit message.
"""

from __future__ import annotations

import json

import pytest

from nyxara.eval.intelligence import run_intelligence_benchmark
from nyxara.njp.ingest import ingest_triples


def _scores(report):
    return {s["stage"]: s["score"] for s in report.to_dict()["stages"]}


@pytest.fixture()
def corpus(tmp_path):
    """Enough causal edges to overrun `InternalUniverse`'s 512-relation capacity."""
    rows = [{"subject": f"c{i}", "predicate": "causes", "object": f"e{i}"} for i in range(700)]
    path = tmp_path / "cn.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The hook
# --------------------------------------------------------------------------- #

def test_the_benchmark_can_be_given_a_brain_that_already_knows_something(corpus):
    seen = []
    run_intelligence_benchmark(seed=0, width=2, only=["memorization"],
                               prepare=lambda brain: seen.append(brain))
    assert len(seen) == 1
    assert seen[0].grounder is not None


def test_the_preparer_runs_for_every_stage_not_once(corpus):
    """A stage that was not prepared is a stage measuring a different brain from the others."""
    seen = []
    run_intelligence_benchmark(seed=0, width=2, prepare=lambda brain: seen.append(id(brain)))
    assert len(seen) == 6
    assert len(set(seen)) == 6, "each stage must still get its own brain"


def test_a_prepared_fact_is_actually_in_the_brain_the_stage_teaches(corpus):
    checked = []

    def prepare(brain):
        ingest_triples(brain, corpus, source="conceptnet")
        checked.append(len(brain.grounder.facts))

    run_intelligence_benchmark(seed=0, width=2, only=["memorization"], prepare=prepare)
    assert checked and checked[0] == 700


def test_a_failing_preparer_is_reported_rather_than_scored_as_a_fresh_brain():
    """The worst outcome would be a fresh-brain score under a heading claiming a corpus was loaded:
    it reads as evidence the corpus did not help.

    It comes back as `score=None` rather than 0.0, which is the distinction `StageResult.score`
    already draws — *"None when nothing was asked, distinct from 0.0, which means asked and
    wrong"*. A preparer that blew up asked nothing, so there is no score to report, and saying
    so beats reporting a zero that would read as a measured failure.
    """
    def explode(_brain):
        raise RuntimeError("no such corpus")

    report = run_intelligence_benchmark(seed=0, width=2, only=["memorization"], prepare=explode)
    stage = report.to_dict()["stages"][0]
    assert stage["score"] is None
    assert stage["total"] == 0
    assert "no such corpus" in stage["note"]


def test_omitting_the_hook_changes_nothing(corpus):
    """The default path is the one every existing caller uses, and it must be untouched."""
    assert _scores(run_intelligence_benchmark(seed=0, width=4)) == _scores(
        run_intelligence_benchmark(seed=0, width=4, prepare=None))


# --------------------------------------------------------------------------- #
# What the hook found on its first run
# --------------------------------------------------------------------------- #

def test_bulk_facts_do_not_cost_her_the_ability_to_answer_a_counterfactual(corpus):
    """The regression the instrument caught, pinned so it cannot come back.

    `InternalUniverse` holds 512 relations. With `to_world` on by default, a bulk load stated
    more laws than that, `sync_from_world` filled the universe with them, and her own
    `water → growth` — the relation the question is about — never got in. Measured at 8,000 facts:
    `causal_prediction` 1.00 → 0.00, restored exactly by `to_world=False`.
    """
    def prepare(brain):
        ingest_triples(brain, corpus, source="conceptnet")

    scores = _scores(run_intelligence_benchmark(seed=0, width=4,
                                                only=["causal_prediction"], prepare=prepare))
    assert scores["causal_prediction"] == 1.0


def test_routing_a_corpus_into_the_world_model_is_what_breaks_it(corpus):
    """The counterfactual for the test above: the defect is real and this is where it lives."""
    def prepare(brain):
        ingest_triples(brain, corpus, source="conceptnet", to_world=True)

    scores = _scores(run_intelligence_benchmark(seed=0, width=4,
                                                only=["causal_prediction"], prepare=prepare))
    assert scores["causal_prediction"] == 0.0, (
        "if this now passes, the universe's capacity or its sync policy changed — re-read "
        "`ingest_triples`'s `to_world` argument before relaxing its default")


def test_a_commonsense_corpus_does_not_regress_any_stage(corpus):
    """The benchmark's real job here: a regression guard on prepared state.

    Its vocabulary is RNG nonsense, disjoint by construction from any corpus, so a corpus cannot
    make her better at these questions by containing their answers. A gain would be structural or
    it would be an error; what it can honestly show is that nothing got worse.
    """
    def prepare(brain):
        ingest_triples(brain, corpus, source="conceptnet")

    base = _scores(run_intelligence_benchmark(seed=0, width=4))
    fed = _scores(run_intelligence_benchmark(seed=0, width=4, prepare=prepare))
    # A stage that scored `None` asked nothing and is not a comparison; only measured pairs count.
    worse = {k: (base[k], fed[k]) for k in base
             if base[k] is not None and fed[k] is not None and fed[k] < base[k]}
    assert not worse, f"ingestion cost her these stages: {worse}"
