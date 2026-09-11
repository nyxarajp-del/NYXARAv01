"""V.82 — the whole loop, and whether diagnosing is worth more than guessing.

Six organs each carry their own floor. What had not been measured is whether assembling them buys
anything: a loop that diagnoses carefully and then repairs no better than chance has learned
nothing about itself, it has only spent longer.

So every claim here is a comparison. `diagnosed` is scored against a `blind` twin that skips
attribution and picks at random from the same shelf, and a `greedy` twin with a pet theory. They
differ in one line of code and nowhere else.
"""

from __future__ import annotations

import random

from nyxara.njp.ascent import HIDDEN, STRATEGIES, Ascent, Attempt, Repair, ascend
from nyxara.njp.ascentschool import DIAL, WAYS, Broken, examine
from nyxara.njp.reach import Ladder, Rung


def _broken(way: str = "algorithm", edge: int = 3) -> Broken:
    return Broken(way=way, edge=edge, seed=82)


def _run(broken: Broken, strategy: str, seed: int = 1) -> Attempt:
    return ascend(broken.bench, DIAL, broken.repairs(),
                  experiments=lambda at, _x: broken.failure(at),
                  hidden=lambda r, at: broken.hidden(
                      at, fixed=(r.addresses if r is not None else ""), draw=0),
                  strategy=strategy, rng=random.Random(seed), favourite="data")


# --------------------------------------------------------------------------------------------- #
#  the comparison the loop exists for
# --------------------------------------------------------------------------------------------- #
def test_diagnosing_moves_the_edge_more_often_than_guessing():
    got = examine(60)
    assert got["diagnosed"]["moved"] > got["blind"]["moved"], got


def test_and_more_often_than_a_pet_theory():
    got = examine(60)
    assert got["diagnosed"]["moved"] > got["greedy"]["moved"], got


def test_guessing_lands_on_the_right_repair_about_one_time_in_four():
    """Four repairs, one right. A floor that is not a quarter means the shelf is not even."""
    got = examine(60)
    assert 0.15 <= got["blind"]["right_repair"] <= 0.40, got["blind"]


def test_diagnosis_lands_on_it_far_more_often_than_that():
    got = examine(60)
    assert got["diagnosed"]["right_repair"] > 0.80, got["diagnosed"]


def test_every_strategy_is_measured_on_the_same_capabilities():
    """A strategy scored on its own sample is answering a different question."""
    got = examine(40)
    assert len({got[name]["attempts"] for name in STRATEGIES}) == 1


# --------------------------------------------------------------------------------------------- #
#  the gate, one level up
# --------------------------------------------------------------------------------------------- #
def test_nothing_is_kept_that_does_not_move_the_edge():
    """A gate too generous to be worth having, measured rather than assumed away."""
    got = examine(60)
    assert all(got[name]["kept_nothing"] == 0.0 for name in STRATEGIES), got


def test_a_repair_is_judged_on_problems_it_has_not_seen():
    """Both readings on the same hidden problems, so what moves the number is the repair.

    Drawing them separately made noise alone raise the second figure about half the time, and wrong
    repairs were then kept at 0.683 while only 0.233 of them were right.
    """
    broken = _broken()
    before = broken.hidden(5, fixed="", draw=0)
    same = broken.hidden(5, fixed="", draw=0)
    assert before == same, "the same draw must give the same reading"
    assert broken.hidden(5, fixed="algorithm", draw=0) > before


def test_a_wrong_repair_changes_nothing_at_all_rather_than_a_little():
    """Or a strategy could stumble into a gain by trying things, and the comparison would be noise."""
    broken = _broken(way="algorithm")
    assert broken.hidden(5, fixed="reading", draw=0) == broken.hidden(5, fixed="", draw=0)


# --------------------------------------------------------------------------------------------- #
#  what the loop does with nothing to work on
# --------------------------------------------------------------------------------------------- #
def test_a_capability_with_no_edge_is_left_alone():
    """No edge means no rung past it — there is nothing to reach for, and inventing one is worse."""
    everywhere = Broken(way="data", edge=99, seed=1)
    got = _run(everywhere, "diagnosed")
    assert got.at is None and not got.kept


def test_a_total_repair_is_reported_as_reaching_further_not_as_no_change():
    """An exhausted ladder has no *edge* and has certainly moved. Conflating those lost the win."""
    after = Ladder(rungs=[Rung(at=i, score=0.9, floor=0.5) for i in DIAL])
    before = Ladder(rungs=[Rung(at=i, score=0.9 if i <= 3 else 0.5, floor=0.5) for i in DIAL])
    from nyxara.njp.ascent import _reach
    assert after.exhausted and after.edge is None
    assert _reach(after, before) == DIAL[-1] > before.edge


def test_a_repair_that_cannot_be_applied_is_not_counted_as_tried():
    broken = _broken()
    got = ascend(broken.bench, DIAL, [Repair(name="empty", addresses="data", apply=None)],
                 experiments=lambda at, _x: broken.failure(at),
                 hidden=lambda r, at: 0.5, strategy="diagnosed", rng=random.Random(1))
    assert got.chose == "" and not got.kept


# --------------------------------------------------------------------------------------------- #
#  the three strategies differ in exactly one place
# --------------------------------------------------------------------------------------------- #
def test_the_diagnosis_falls_back_to_chance_when_it_has_nothing_to_say():
    """No established cause must not become a pretended one. It is counted, never borrowed from."""
    from nyxara.njp.ascent import _choose
    shelf = [Repair(name=f"fix {w}", addresses=w) for w in WAYS]
    got = _choose("diagnosed", shelf, "", random.Random(1), "data")
    assert got in shelf


def test_greedy_always_reaches_for_the_same_one():
    from nyxara.njp.ascent import _choose
    shelf = [Repair(name=f"fix {w}", addresses=w) for w in WAYS]
    picks = {_choose("greedy", shelf, "reading", random.Random(i), "data").addresses
             for i in range(8)}
    assert picks == {"data"}


def test_diagnosed_takes_the_repair_for_the_cause():
    from nyxara.njp.ascent import _choose
    shelf = [Repair(name=f"fix {w}", addresses=w) for w in WAYS]
    assert _choose("diagnosed", shelf, "budget", random.Random(1), "data").addresses == "budget"


# --------------------------------------------------------------------------------------------- #
#  the counting
# --------------------------------------------------------------------------------------------- #
def test_moved_counts_only_capabilities_that_had_an_edge_to_move():
    run = Ascent(strategy="x", attempts=[
        Attempt(edge_before=3, edge_after=5), Attempt(edge_before=3, edge_after=3),
        Attempt(edge_before=None, edge_after=None)])
    assert run.moved == 0.5


def test_kept_nothing_is_the_share_of_keeps_that_bought_nothing():
    run = Ascent(strategy="x", attempts=[
        Attempt(kept=True, edge_before=3, edge_after=5),
        Attempt(kept=True, edge_before=3, edge_after=3)])
    assert run.kept_nothing == 0.5


def test_an_empty_ascent_claims_nothing():
    assert Ascent().moved == 0.0 and Ascent().kept == 0.0 and Ascent().right_repair == 0.0


def test_the_hidden_set_is_large_enough_to_decide_on():
    assert HIDDEN >= 100


def test_the_exam_states_its_own_pass_condition():
    got = examine(40)
    assert got["passes"] is True
    assert set(got) >= {"moved_over_blind", "moved_over_greedy"}
