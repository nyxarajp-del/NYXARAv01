"""V.74 — the organ that asks whether a number means what it looks like.

A critic of measurements has two failure modes and only one of them is obvious. Missing a bad
benchmark is the obvious one. Condemning every benchmark is the subtle one, and it reads as
vigilance while carrying no information at all — this package has been caught by that before, with
a veto whose false-alarm rate was 0.0000 because it never fired.

So most of what is pinned here is the second kind: that a check which cannot run says so rather
than passing, that a sound benchmark comes back clear, and that the critic's own null is not one
coin landing.
"""

from __future__ import annotations

import random

from nyxara.njp.measurement import (
    CHECKS, CLOSE, LUCK, PERMUTATIONS, Benchmark, Critique, Finding, critique,
)
from nyxara.njp.measurementschool import KNOWN, examine, retrodict


def _flat(n: int = 200, skew: float = 0.8, seed: int = 1):
    rng = random.Random(seed)
    return ["A" if rng.random() < skew else "B" for _ in range(n)]


# --------------------------------------------------------------------------------------------- #
#  a check that cannot run is not a check that passed
# --------------------------------------------------------------------------------------------- #
def test_a_benchmark_with_only_the_essentials_says_what_it_could_not_check():
    gold = _flat()
    got = critique(Benchmark(name="bare", items=list(range(200)), gold=gold,
                             predict=lambda i: gold[i]))
    assert set(got.unchecked) == {"shuffled", "leakage", "abstention", "ceiling"}
    assert len(got.findings) == len(CHECKS), "every check is reported, run or not"


def test_a_skipped_check_is_never_counted_as_evidence():
    gold = _flat()
    got = critique(Benchmark(name="bare", items=list(range(200)), gold=gold,
                             predict=lambda i: gold[i]))
    assert not got.trusted, "three of seven checks ran; that cannot support a verdict"
    assert all(not f.informative for f in got.findings if f.check in got.unchecked)


def test_a_perfect_system_on_a_thin_benchmark_is_still_not_trusted():
    """Scoring 1.000 is not the same as having been measured, and the two must not be conflated."""
    gold = _flat()
    got = critique(Benchmark(name="perfect", items=list(range(200)), gold=gold,
                             predict=lambda i: gold[i]))
    assert got.system == 1.0
    assert not got.trusted


# --------------------------------------------------------------------------------------------- #
#  the floors
# --------------------------------------------------------------------------------------------- #
def test_a_system_that_only_says_the_commonest_answer_is_caught():
    gold = _flat(skew=0.85)
    got = critique(Benchmark(items=list(range(200)), gold=gold, predict=lambda _i: "A"))
    majority = next(f for f in got.findings if f.check == "majority")
    assert majority.verdict == "close"
    assert got.headline <= CLOSE


def test_a_system_below_its_own_floor_is_broken_not_merely_weak():
    gold = _flat(skew=0.85)
    got = critique(Benchmark(items=list(range(200)), gold=gold, predict=lambda _i: "B"))
    assert next(f for f in got.findings if f.check == "majority").verdict == "broken"


def test_the_headline_is_measured_from_the_highest_floor_not_the_most_flattering():
    gold = _flat(skew=0.85)
    got = critique(Benchmark(items=list(range(200)), gold=gold,
                             predict=lambda i: gold[i] if i % 10 else "B"))
    floors = [f.got for f in got.ran if f.check in ("majority", "chance", "shuffled")]
    assert got.headline == round(got.system - max(floors), 4)


# --------------------------------------------------------------------------------------------- #
#  the null is not one coin landing
# --------------------------------------------------------------------------------------------- #
def test_the_shuffled_null_is_drawn_many_times():
    """One draw of a null is the same error this organ exists to catch, and it made it once."""
    assert PERMUTATIONS >= 20


def test_a_learner_that_found_nothing_cannot_hide_behind_one_lucky_shuffle():
    rng = random.Random(5)
    rows = [[rng.randrange(3) for _ in range(40)] for _ in range(120)]
    gold = _flat(120, 0.5, seed=6)
    train, held = list(range(84)), list(range(84, 120))

    def _teach(items, answers):
        table = {}
        for i, want in zip(items, answers):
            table.setdefault(rows[i][0], []).append(want)
        say = {v: max(set(w), key=w.count) for v, w in table.items()}
        return lambda i: say.get(rows[i][0], "A")

    bench = Benchmark(name="noise", items=held, gold=[gold[i] for i in held],
                      predict=_teach(train, [gold[i] for i in train]),
                      learn=_teach, train=train, key=lambda i: i)
    shuffled = next(f for f in critique(bench).findings if f.check == "shuffled")
    assert shuffled.informative
    assert shuffled.verdict in ("broken", "close"), shuffled.says


# --------------------------------------------------------------------------------------------- #
#  the four defects, and the four repairs
# --------------------------------------------------------------------------------------------- #
def test_leakage_is_found_when_the_examined_items_were_also_learned_from():
    gold = _flat(100, 0.5, seed=2)
    got = critique(Benchmark(items=list(range(100)), gold=gold, predict=lambda i: gold[i],
                             train=list(range(50)), key=lambda i: i))
    leak = next(f for f in got.findings if f.check == "leakage")
    assert leak.verdict == "broken" and leak.got == 0.5


def test_a_score_that_moves_between_identical_runs_is_the_finding():
    rng = random.Random(9)
    gold = ["A"] * 200
    got = critique(Benchmark(items=list(range(200)), gold=gold,
                             predict=lambda _i: "A" if rng.random() < 0.5 else "B"))
    assert next(f for f in got.findings if f.check == "stability").verdict == "broken"


def test_a_headline_that_hides_how_rarely_it_speaks_is_flagged():
    gold = _flat(200, 0.5, seed=3)
    speaks = [i % 20 == 0 for i in range(200)]
    got = critique(Benchmark(items=list(range(200)), gold=gold,
                             predict=lambda i: gold[i] if speaks[i] else "",
                             spoke=lambda i: speaks[i]))
    abstain = next(f for f in got.findings if f.check == "abstention")
    assert abstain.verdict == "close"
    assert abstain.got < 0.1 and abstain.against == 1.0


def test_a_ceiling_the_system_has_already_reached_is_the_thing_to_raise():
    gold = _flat(200, 0.5, seed=4)
    reach = [i % 2 == 0 for i in range(200)]
    got = critique(Benchmark(items=list(range(200)), gold=gold,
                             predict=lambda i: gold[i] if reach[i] else "",
                             reachable=lambda i, _w: reach[i]))
    ceiling = next(f for f in got.findings if f.check == "ceiling")
    assert ceiling.verdict == "broken" and ceiling.got == 0.5


# --------------------------------------------------------------------------------------------- #
#  the exam: it has to separate them, not condemn them all
# --------------------------------------------------------------------------------------------- #
def test_it_finds_every_measurement_this_repository_got_wrong():
    got = retrodict()
    assert got["recall"] == 1.0, [r["case"] for r in got["rows"]
                                  if r["misleading"] and not r["objected"]]


def test_and_clears_every_one_it_got_right():
    """The half that matters. A critic that flags everything scores full marks on the other one."""
    got = retrodict()
    assert got["false_alarm_rate"] == 0.0, [r["case"] for r in got["rows"]
                                            if not r["misleading"] and r["objected"]]


def test_each_case_is_caught_by_the_check_that_should_catch_it():
    """Right answer, wrong reason is not a hit — the check named in the case has to be the one."""
    for row in retrodict()["rows"]:
        if row["misleading"]:
            assert row["objected"], f"{row['case']} not caught by {row['by']}"


def test_the_exam_states_its_own_pass_condition():
    got = examine()
    assert got["passes"] is True
    assert set(got) >= {"recall", "false_alarm_rate", "caught", "false_alarms"}


def test_the_answer_key_is_history_and_not_something_the_critic_can_see():
    """`was_misleading` comes from what was found by hand. Nothing in `Benchmark` carries it."""
    for case in KNOWN:
        bench = case.build()
        assert not hasattr(bench, "was_misleading")
        assert case.name not in repr(bench.predict)


# --------------------------------------------------------------------------------------------- #
#  the report
# --------------------------------------------------------------------------------------------- #
def test_a_critique_renders_every_check_including_the_ones_it_could_not_run():
    gold = _flat()
    text = critique(Benchmark(name="bare", items=list(range(200)), gold=gold,
                              predict=lambda i: gold[i])).render()
    for check in CHECKS:
        assert check in text


def test_a_verdict_is_about_the_measurement_never_about_the_system():
    """`broken` means this benchmark cannot tell whether the system is good, not that it is bad."""
    gold = _flat(100, 0.5, seed=2)
    got = critique(Benchmark(items=list(range(100)), gold=gold, predict=lambda i: gold[i],
                             train=list(range(50)), key=lambda i: i))
    assert got.system == 1.0 and got.broken
    assert not got.trusted


def test_luck_is_a_stated_tolerance_rather_than_a_convention():
    assert 0.0 < LUCK < 0.5
    assert isinstance(Finding().informative, bool)
    assert Critique().trusted is False
