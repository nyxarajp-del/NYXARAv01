"""V.77 — why did it fail, tested rather than guessed.

An attributor has three failure modes and only the first is obvious. It can name the wrong cause —
obvious, and the least dangerous, because the repair then visibly fails. It can name **one cause
for everything**, which reads as decisiveness and carries no information. And it can credit a
hypothesis it never put at risk, which is how a favourite explanation survives evidence.

Most of what is pinned here is the second and third kind.
"""

from __future__ import annotations

import random

from nyxara.njp.attribution import (
    CAUSES, MOVED, REPAIRS, SPOILED, Attribution, Failure, Verdict, attribute, chain,
    render_chain,
)
from nyxara.njp.attributionschool import KNOWN, examine, retrodict, walk_the_chain
from nyxara.njp.measurement import Benchmark


def _bench(score: float, n: int = 200, skew: float = 0.5, seed: int = 1) -> Benchmark:
    rng = random.Random(seed)
    gold = ["A" if rng.random() < skew else "B" for _ in range(n)]
    hit = random.Random(seed + 1)
    said = [g if hit.random() < score else ("B" if g == "A" else "A") for g in gold]
    return Benchmark(name="made up", items=list(range(n)), gold=gold, predict=lambda i: said[i])


# --------------------------------------------------------------------------------------------- #
#  an untested hypothesis is never support
# --------------------------------------------------------------------------------------------- #
def test_a_hypothesis_with_no_experiment_is_reported_untested_not_refuted():
    got = attribute(Failure(name="bare", bench=_bench(0.6)))
    assert set(got.untested) >= {"data", "algorithm", "reading", "budget"}
    assert not any(v.stands == "supported" for v in got.verdicts if v.cause in REPAIRS)


def test_every_cause_is_reported_whether_or_not_it_could_be_tested():
    got = attribute(Failure(name="bare", bench=_bench(0.6)))
    assert {v.cause for v in got.verdicts} == set(CAUSES)


def test_every_cause_writes_down_what_would_refute_it():
    """A hypothesis with no refuting observation is not a hypothesis."""
    for verdict in attribute(Failure(name="bare", bench=_bench(0.6))).verdicts:
        assert verdict.refuted_by, verdict.cause


def test_an_experiment_that_changes_nothing_refutes_its_cause_rather_than_leaving_it_open():
    """The asymmetry is the whole discipline: a repair that changes nothing is not the cause."""
    same = _bench(0.6)
    got = attribute(Failure(name="flat", bench=same, more_data=lambda: _bench(0.6)))
    data = next(v for v in got.verdicts if v.cause == "data")
    assert data.stands == "refuted"
    assert "not" in data.says


def test_an_experiment_that_moves_the_number_supports_its_cause():
    got = attribute(Failure(name="moved", bench=_bench(0.55),
                            more_data=lambda: _bench(0.95)))
    assert next(v for v in got.verdicts if v.cause == "data").stands == "supported"


# --------------------------------------------------------------------------------------------- #
#  the ordering: gates are not competitors
# --------------------------------------------------------------------------------------------- #
def test_a_score_that_will_not_hold_still_gates_everything_below_it():
    """Nothing downstream can be concluded from a reading that moves between runs."""
    rng = random.Random(3)
    gold = ["A"] * 200
    wobbly = Benchmark(name="wobbly", items=list(range(200)), gold=gold,
                       predict=lambda _i: "A" if rng.random() < 0.5 else "B")
    got = attribute(Failure(name="wobbly", bench=wobbly, more_data=lambda: _bench(0.99)))
    assert got.root == "measurement"


def test_leakage_outranks_any_repair_because_the_score_is_not_about_held_out_work():
    gold = ["A" if i % 2 else "B" for i in range(100)]
    leaky = Benchmark(name="leaky", items=list(range(100)), gold=gold,
                      predict=lambda i: gold[i], train=list(range(100)), key=lambda i: i)
    got = attribute(Failure(name="leaky", bench=leaky, more_data=lambda: _bench(0.99)))
    assert got.root == "leakage"


def test_the_floor_is_a_fallback_and_never_beats_a_repair_that_worked():
    """Sitting on its own floor is the failure restated. A repair that moves it is the cause."""
    flat = _bench(0.5, skew=0.5)
    got = attribute(Failure(name="at the floor", bench=flat,
                            other_algorithm=lambda: _bench(0.95)))
    assert got.root == "algorithm"


def test_the_floor_is_the_answer_once_every_repair_has_been_tried_and_none_moved_it():
    flat = _bench(0.5, skew=0.5)
    got = attribute(Failure(name="nothing reaches it", bench=flat,
                            more_data=lambda: _bench(0.5, skew=0.5),
                            other_algorithm=lambda: _bench(0.5, skew=0.5),
                            richer_reading=lambda: _bench(0.5, skew=0.5),
                            more_budget=lambda: _bench(0.5, skew=0.5)))
    assert got.root == "floor"


# --------------------------------------------------------------------------------------------- #
#  it is allowed to say it does not know
# --------------------------------------------------------------------------------------------- #
def test_two_repairs_that_move_it_equally_are_reported_as_unseparated():
    got = attribute(Failure(name="two ways", bench=_bench(0.5, skew=0.5),
                            more_data=lambda: _bench(0.9, skew=0.5),
                            other_algorithm=lambda: _bench(0.9, skew=0.5)))
    assert got.root == ""
    assert set(got.rivals) == {"data", "algorithm"}
    assert not got.settled


def test_an_unseparated_attribution_says_what_experiment_is_missing():
    got = attribute(Failure(name="two ways", bench=_bench(0.5, skew=0.5),
                            more_data=lambda: _bench(0.9, skew=0.5),
                            other_algorithm=lambda: _bench(0.9, skew=0.5)))
    assert "does not separate" in got.render()


def test_a_clear_winner_is_named_and_has_no_rivals():
    got = attribute(Failure(name="one way", bench=_bench(0.5, skew=0.5),
                            more_data=lambda: _bench(0.95, skew=0.5),
                            other_algorithm=lambda: _bench(0.52, skew=0.5)))
    assert got.root == "data" and got.rivals == []


# --------------------------------------------------------------------------------------------- #
#  cause of cause
# --------------------------------------------------------------------------------------------- #
def test_the_chain_walks_to_the_failure_behind_the_failure():
    steps = walk_the_chain()
    assert len(steps) == 3
    assert [s.root for s in steps] == ["data", "reachability", "reading"]


def test_the_chain_says_where_it_stopped_rather_than_implying_bedrock():
    deepest = Failure(name="bottom", bench=_bench(0.6))
    text = render_chain(chain(Failure(name="top", bench=_bench(0.6), upstream=deepest)), deepest)
    assert "not the same as nothing being there" in text


def test_a_cycle_in_the_chain_ends_the_walk_instead_of_spinning():
    one = Failure(name="one", bench=_bench(0.6))
    two = Failure(name="two", bench=_bench(0.6), upstream=one)
    one.upstream = two
    assert len(chain(one)) == 2


# --------------------------------------------------------------------------------------------- #
#  the exam
# --------------------------------------------------------------------------------------------- #
def test_it_finds_the_cause_of_every_failure_built_around_one():
    got = retrodict()
    assert got["correct"] == got["of"], [(r["case"], r["want"], r["got"]) for r in got["rows"]
                                         if r["got"] != r["want"]]


def test_it_never_names_a_cause_that_is_not_the_one():
    """A wrong cause sends real work somewhere real. It costs more than an honest shrug."""
    assert retrodict()["wrong"] == 0


def test_it_does_not_say_the_same_word_to_everything():
    """The failure mode that looks like decisiveness and carries no information."""
    assert retrodict()["distinct_roots"] >= 4


def test_it_declines_to_name_a_cause_where_there_is_none_to_name():
    row = next(r for r in retrodict()["rows"] if r["want"] == "")
    assert row["got"] == ""


def test_the_exam_states_its_own_pass_condition():
    got = examine()
    assert got["passes"] is True
    assert set(got) >= {"correct", "wrong", "unsettled", "distinct_roots"}


def test_every_fixture_supplies_every_experiment():
    """Or a cause could be credited for being the only one anybody tried."""
    for case in KNOWN:
        failure = case.build()
        for field in ("more_data", "other_algorithm", "richer_reading", "more_budget"):
            assert getattr(failure, field) is not None, (case.name, field)


def test_the_verdict_does_not_depend_on_what_the_failure_is_called():
    """The names describe the fixtures for a reader, and several contain their own answer.

    Checking that they do not is the wrong test — the right one is that the attributor does not
    read them. Every failure is renamed to a number and re-attributed; the verdicts must be
    identical, because nothing in the decision is allowed to come from a label.
    """
    for n, case in enumerate(KNOWN):
        first = attribute(case.build())
        anonymous = case.build()
        anonymous.name = f"case {n}"
        if anonymous.bench is not None:
            anonymous.bench.name = f"bench {n}"
        again = attribute(anonymous)
        assert again.root == first.root, (case.name, first.root, again.root)
        assert [v.stands for v in again.verdicts] == [v.stands for v in first.verdicts]


def test_a_failure_carries_no_answer_key():
    for case in KNOWN:
        assert not hasattr(case.build(), "cause")


def test_a_bare_attribution_is_honest_about_knowing_nothing():
    assert Attribution().root == "" and not Attribution().settled
    assert Verdict().tested is False
    assert 0.0 < MOVED < 0.5


# --------------------------------------------------------------------------------------------- #
#  V.83 — an experiment that changed two things tested neither
# --------------------------------------------------------------------------------------------- #
def _ceiling_bench(name, *, right, reach, n=100):
    """A benchmark scoring ``right`` with the answer reachable for ``reach`` of its items."""
    items = list(range(n))
    gold = ["A"] * n
    said = ["A" if i < round(right * n) else "B" for i in range(n)]
    can = [i < round(reach * n) for i in range(n)]
    return Benchmark(name=name, items=items, gold=gold, predict=lambda i: said[i],
                     reachable=lambda i, _w: can[i])


def test_a_repair_that_lowers_the_ceiling_is_not_a_refutation():
    """The real case, in miniature: the score fell because the answer was thrown away.

    ``budget: refuted`` reads as *more search would not have helped*. That claim was never tested
    by an experiment that also stopped offering the right answer, and the organ must not make it.
    """
    failure = Failure(name="capped", bench=_ceiling_bench("as found", right=0.30, reach=0.90),
                      more_budget=lambda: _ceiling_bench("capped", right=0.10, reach=0.20))
    budget = next(v for v in attribute(failure).verdicts if v.cause == "budget")
    assert budget.stands == "spoiled"
    assert not budget.tested
    assert "0.9000 → 0.2000" in budget.says


def test_a_repair_that_raises_the_ceiling_is_not_a_supported_cause():
    """The dangerous direction. The score went up, and not for the reason on the label.

    Crediting this to `budget` sends the next version to tune a search when what actually moved was
    what the search was allowed to find — which is the exact mistake this module exists to stop,
    arriving one level above where it was being watched for.
    """
    failure = Failure(name="widened", bench=_ceiling_bench("as found", right=0.20, reach=0.30),
                      more_budget=lambda: _ceiling_bench("widened", right=0.60, reach=0.95))
    budget = next(v for v in attribute(failure).verdicts if v.cause == "budget")
    assert budget.stands == "spoiled"
    assert budget.moved >= MOVED, "it did move the score — that is what makes it dangerous"


def test_a_repair_that_keeps_the_ceiling_is_judged_on_the_score_alone():
    """The false-alarm side. A ceiling that wobbles under :data:`SPOILED` is the same ceiling."""
    for after, want in ((0.30, "refuted"), (0.60, "supported")):
        failure = Failure(name="honest",
                          bench=_ceiling_bench("as found", right=0.30, reach=0.90),
                          more_budget=lambda a=after: _ceiling_bench("more", right=a, reach=0.88))
        got = next(v for v in attribute(failure).verdicts if v.cause == "budget")
        assert got.stands == want, (after, got.stands)
        assert got.tested


def test_an_unmeasured_ceiling_is_not_read_as_an_unchanged_one():
    """No ``reachable`` anywhere means the guard cannot run, and it says nothing rather than pass."""
    failure = Failure(name="blind",
                      bench=Benchmark(name="as found", items=[0, 1], gold=["A", "A"],
                                      predict=lambda i: "B"),
                      more_budget=lambda: Benchmark(name="more", items=[0, 1], gold=["A", "A"],
                                                    predict=lambda i: "A"))
    assert next(v for v in attribute(failure).verdicts if v.cause == "budget").stands == "supported"


def test_floor_is_withheld_while_a_lever_was_never_pulled():
    """`these levers do not reach this` is not shown by a lever nobody pulled.

    The ninth fixture is a task sitting on its own floor with every honest repair refuted and one
    repair spoiled. Before V.83 the attributor read that as `floor` — a confident, wrong, and
    expensive conclusion, since it closes the question.
    """
    got = attribute(next(c for c in KNOWN if c.name == "a lever that was never pulled").build())
    assert got.spoiled == ["budget"]
    assert got.root == "", "floor was concluded from an experiment that tested nothing"
    assert any(v.cause == "floor" and v.stands == "supported" for v in got.verdicts)
    assert "never actually tried" in got.render()


def test_the_exam_scores_catches_and_false_alarms_together():
    """Flagging nothing and flagging everything both fail, which is the point of scoring both."""
    got = examine()
    assert got["spoilable"] >= 1
    assert got["missed"] == 0 and got["false_alarms"] == 0
    assert got["caught"] == got["spoilable"]
    assert got["passes"]


def test_spoiled_is_neither_tested_nor_untested():
    assert Verdict(stands="spoiled").tested is False
    got = attribute(next(c for c in KNOWN if c.name == "a lever that was never pulled").build())
    assert "budget" not in got.untested and "budget" not in got.refuted
    assert 0.0 < SPOILED < 0.5
