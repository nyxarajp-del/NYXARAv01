"""A self-model that never changes a decision is decoration.

`SelfModel` has held a measured record per faculty since it was written, and nothing consulted it
when picking one. This file is about the two halves that make it matter — `MetaReasoner._demoted`
reads the record and `MetaReasoner.outcome` writes it — and about the three ways the gate is
deliberately conservative, each of which is the difference between a gate and a superstition:

* an **untested** strategy is never demoted, only a measurably weak one;
* the pool is never emptied, because if everything is weak then weakness is not the discriminator;
* UCB1 still chooses among the survivors, so a demoted strategy earns its way back.

The falsification test the plan asked for is `test_the_gate_reports_when_it_changed_nothing`: if
`gated` stays at zero on a brain with a real record, the mechanism is doing nothing and should come
out.
"""

from __future__ import annotations

import pytest

from nyxara.njp.metareason import MetaReasoner, ProblemKind
from nyxara.njp.selfmodel import SelfModel, _MIN_OBSERVATIONS


def _meta(model: SelfModel | None = None) -> MetaReasoner:
    meta = MetaReasoner(self_model=model)
    meta.register("good", (ProblemKind.FACTUAL,), lambda problem, ctx: "answer", prior=0.5)
    meta.register("bad", (ProblemKind.FACTUAL,), lambda problem, ctx: "answer", prior=0.5)
    return meta


def _record(model: SelfModel, name: str, success: float, n: int = _MIN_OBSERVATIONS + 2) -> None:
    for _ in range(n):
        model.observe(f"strategy:{name}", success)


def test_a_measurably_weak_strategy_is_demoted():
    model = SelfModel()
    _record(model, "bad", 0.0)
    _record(model, "good", 1.0)
    meta = _meta(model)
    pool = list(meta.strategies.values())
    kept = meta._demoted(pool)
    assert [s.name for s in kept] == ["good"]
    assert meta.gated == 1


def test_an_untested_strategy_is_never_demoted():
    """A cold start is not a weakness, and penalising it makes her permanently timid."""
    model = SelfModel()
    _record(model, "bad", 0.0)
    meta = _meta(model)
    for _ in range(2):
        model.observe("strategy:good", 0.0)          # too little evidence to count
    kept = meta._demoted(list(meta.strategies.values()))
    assert set(s.name for s in kept) == {"good"}, "the untested one must survive"


def test_the_pool_is_never_emptied():
    """If every option is weak then weakness is not the discriminator and UCB1 is the better judge."""
    model = SelfModel()
    _record(model, "bad", 0.0)
    _record(model, "good", 0.0)
    meta = _meta(model)
    pool = list(meta.strategies.values())
    kept = meta._demoted(pool)
    assert len(kept) == len(pool)
    assert meta.gated == 0


def test_a_single_option_is_never_gated():
    model = SelfModel()
    _record(model, "bad", 0.0)
    meta = _meta(model)
    only = [meta.strategies["bad"]]
    assert meta._demoted(only) == only
    assert meta.gated == 0


def test_no_self_model_means_no_gating():
    meta = _meta(None)
    pool = list(meta.strategies.values())
    assert meta._demoted(pool) == pool
    assert meta.gated == 0


def test_choose_respects_the_gate():
    model = SelfModel()
    _record(model, "bad", 0.0)
    _record(model, "good", 1.0)
    meta = _meta(model)
    for _ in range(6):
        assert meta.choose(ProblemKind.FACTUAL).name == "good"


def test_outcome_writes_the_record_the_gate_reads():
    """Without this write the gate consults an empty model forever and changes nothing."""
    from nyxara.njp.metareason import Solution

    model = SelfModel()
    meta = _meta(model)
    solution = Solution(problem="q", kind=ProblemKind.FACTUAL, strategy="bad")
    for _ in range(_MIN_OBSERVATIONS + 2):
        meta.outcome(solution, correct=False)
    measured = model._measured("strategy:bad")
    assert measured is not None and measured.weak


def test_the_gate_reports_when_it_changed_nothing():
    """The falsification the plan demands: `gated` is the number that says the gate is real."""
    model = SelfModel()
    meta = _meta(model)
    meta._demoted(list(meta.strategies.values()))
    assert meta.gated == 0, "an empty record must not demote anything"


def test_a_demoted_strategy_earns_its_way_back():
    model = SelfModel()
    _record(model, "bad", 0.0)
    _record(model, "good", 1.0)
    meta = _meta(model)
    assert [s.name for s in meta._demoted(list(meta.strategies.values()))] == ["good"]
    _record(model, "bad", 1.0, n=40)                 # it starts succeeding
    kept = {s.name for s in meta._demoted(list(meta.strategies.values()))}
    assert kept == {"good", "bad"}


def test_a_brain_wires_its_own_record_into_the_chooser():
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    if brain.metareason is None or brain.self_model is None:
        pytest.skip("this brain builds neither organ")
    assert brain.metareason.self_model is brain.self_model
