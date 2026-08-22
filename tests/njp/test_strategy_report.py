"""The strategy scoreboard, and the thing it used to get wrong.

`MetaReasoner.register`'s docstring already names the defect: `Strategy` carries a single
`wins`/`trials` pair spanning every kind it serves, so a strategy registered for three kinds
reports the mean of three different competences. The reporting path then ranked *per kind* on
that kind-blind number, which is how a strategy wins the kind it is worst at.

The per-kind record was never missing. It was in the shared bandit the whole time, correct and
rewarded race-free, and nothing read it.
"""

from __future__ import annotations

from nyxara.njp.metareason import MetaReasoner, ProblemKind
from nyxara.njp.selfmodel import MetaLearner


def _reasoner() -> MetaReasoner:
    return MetaReasoner(meta_learner=MetaLearner(min_trials=3))


def _teach(reasoner: MetaReasoner, kind: str, name: str, value: float, times: int) -> None:
    for _ in range(times):
        reasoner.meta_learner.reward(f"strategy:{kind}", value, name=name)


def test_a_strategy_good_at_one_kind_is_not_credited_on_another():
    """The whole point. `derive` is excellent at factual and useless at causal."""
    reasoner = _reasoner()
    reasoner.register("derive", (ProblemKind.FACTUAL, ProblemKind.CAUSAL), lambda p, c: "x")
    reasoner.register("simulate", (ProblemKind.CAUSAL,), lambda p, c: "y")

    _teach(reasoner, ProblemKind.FACTUAL, "derive", 1.0, 6)
    _teach(reasoner, ProblemKind.CAUSAL, "derive", 0.0, 6)
    _teach(reasoner, ProblemKind.CAUSAL, "simulate", 1.0, 6)

    best = reasoner.best_for(ProblemKind.CAUSAL)
    assert best is not None
    assert best.name == "simulate", (
        "ranked on the kind-blind rate, `derive` wins causal on the strength of its factual record")

    assert reasoner.best_for(ProblemKind.FACTUAL).name == "derive"


def test_an_unmeasured_kind_is_absent_from_the_report_not_reported_as_none():
    reasoner = _reasoner()
    reasoner.register("recall", (ProblemKind.FACTUAL,), lambda p, c: "x")
    _teach(reasoner, ProblemKind.FACTUAL, "recall", 1.0, 4)

    by_kind = reasoner.stats()["by_kind"]
    assert ProblemKind.FACTUAL in by_kind
    assert ProblemKind.INTROSPECTIVE not in by_kind, "a kind with no record must not be listed"
    assert all(v is not None for v in by_kind.values())


def test_a_winner_is_not_named_before_the_record_supports_one():
    """One lucky turn is not a competence. `min_trials` is the floor, and it is respected."""
    reasoner = _reasoner()                       # min_trials=3
    reasoner.register("simulate", (ProblemKind.CAUSAL,), lambda p, c: "y")
    _teach(reasoner, ProblemKind.CAUSAL, "simulate", 1.0, 2)

    assert reasoner.best_for(ProblemKind.CAUSAL) is None
    assert reasoner.advice(ProblemKind.CAUSAL) is None

    _teach(reasoner, ProblemKind.CAUSAL, "simulate", 1.0, 1)
    assert reasoner.best_for(ProblemKind.CAUSAL).name == "simulate"


def test_the_curve_separates_failing_to_bind_from_answering_badly():
    """Two different failures. Only one of them is the strategy being wrong."""
    reasoner = _reasoner()
    reasoner.register("simulate", (ProblemKind.CAUSAL,), lambda p, c: None)   # never binds
    reasoner.register("derive", (ProblemKind.CAUSAL,), lambda p, c: "an answer")

    for _ in range(4):
        reasoner.solve("why does the water boil", context={})

    curve = reasoner.curve(ProblemKind.CAUSAL)
    tried_first = [n for n, row in curve.items() if (row["first_choice_rate"] or 0) > 0]
    assert tried_first, "no strategy was recorded as a first choice"
    for name in tried_first:
        rate = curve[name]["bind_failure_rate"]
        assert rate is None or 0.0 <= rate <= 1.0


def test_advice_says_which_way_of_thinking_suits_this_kind_of_problem():
    reasoner = _reasoner()
    reasoner.register("simulate", (ProblemKind.CAUSAL,), lambda p, c: "y")
    _teach(reasoner, ProblemKind.CAUSAL, "simulate", 1.0, 5)

    said = reasoner.advice(ProblemKind.CAUSAL)
    assert said is not None and "simulate" in said and "over 5" in said


def test_a_runtime_registered_arm_keeps_its_record_across_a_restart():
    """The prerequisite for any strategy she synthesises for herself.

    `load_dict` used to drop a row whose strategy was not registered yet, so an arm added at
    runtime relearned itself from zero on every restart — and, because `ucb` returns `prior + 1.0`
    at zero trials, was explored ahead of everything else each time it did.
    """
    first = _reasoner()
    shape = first.register("shape:is_a>needs", (ProblemKind.FACTUAL,), lambda p, c: "x")
    shape.wins, shape.trials = 7.0, 9
    snapshot = first.to_dict()

    revived = _reasoner()                       # nothing registered yet — the restart case
    revived.load_dict(snapshot)
    restored = revived.register("shape:is_a>needs", (ProblemKind.FACTUAL,), lambda p, c: "x")

    assert restored.trials == 9 and restored.wins == 7.0


def test_a_reasoner_with_no_shared_bandit_still_reports_something():
    """Fail-soft: every organ here is optional, and an absent one must not take the report with it."""
    bare = MetaReasoner()
    bare.register("derive", (ProblemKind.FACTUAL,), lambda p, c: "x")
    bare.solve("what is the capital", context={})
    assert isinstance(bare.stats()["by_kind"], dict)
