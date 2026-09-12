"""The family of languages, taken off the shelf.

V.96 priced a tower and left one line unpaid, naming it: the family of languages was five entries
long because I wrote five. Writing a family of families moves that debt one storey up, so instead
the family is **derived** by a rule with no free numbers — and V.96's own hand-written choice is in
the list as a competitor, charged identically.
"""

from __future__ import annotations

import pytest

from nyxara.njp.bedrock import RULES, Footing, Rule, found_on, reach_needed, stand
from nyxara.njp.generators import WIDTH, watch_behaviour
from nyxara.njp.tower import Language


def _turns(width=WIDTH):
    return sorted({tuple((i + k) % width for i in range(width)) for k in range(width)})


def _unrelated():
    import random
    rng = random.Random(97)
    out = []
    for _ in range(WIDTH):
        order = list(range(WIDTH))
        rng.shuffle(order)
        out.append(tuple(order))
    return out


# --------------------------------------------------------------------------------------------- #
#  a rule has no knobs, which is the point
# --------------------------------------------------------------------------------------------- #
def test_a_rule_takes_no_free_numbers():
    """A rule with a knob can be turned toward an answer. These produce a family from the data."""
    for rule in RULES:
        assert rule.of is not None and rule.says
        small, large = rule.family(_turns(4)), rule.family(_turns(WIDTH))
        assert all(isinstance(lang, Language) for lang in small + large)


def test_the_derived_family_moves_with_the_observations():
    """`as long as it takes` gives a different family for different data, without anybody editing."""
    derived = next(r for r in RULES if r.name == "as long as it takes")
    assert len(derived.family(_turns(4))) < len(derived.family(_turns(WIDTH)))
    fixed = next(r for r in RULES if r.name == "a handful")
    assert len(fixed.family(_turns(4))) == len(fixed.family(_turns(WIDTH))) == 5


def test_the_bound_is_read_off_the_data_and_nothing_else():
    """Each extra letter at most doubles what seeds can build, so reaching n needs log2(n)."""
    assert reach_needed([(0,)]) == 1
    assert reach_needed(_turns(4)) == 2
    assert reach_needed(_turns(8)) == 3
    assert reach_needed([tuple(range(3))] * 9) == 1, "duplicates are one observation"


# --------------------------------------------------------------------------------------------- #
#  the hand-written choice competes on the same terms
# --------------------------------------------------------------------------------------------- #
def test_v96s_own_family_is_in_the_list_as_a_competitor():
    assert any(r.name == "a handful" for r in RULES)
    assert next(r for r in RULES if r.name == "a handful").family([]) == [
        Language(n) for n in (1, 2, 3, 4, 5)]


def test_the_derived_rule_wins_and_for_the_right_reason():
    """All the working rules pick the same language and the same way.

    So the difference is purely the cost of the shelf they were chosen from, which is what a
    derived family is supposed to buy — not a better answer, a cheaper way of arriving at the
    same one.
    """
    got = stand(_turns())
    working = [f for f in got if f.explained]
    assert len(working) >= 3
    assert len({f.chose for f in working}) == 1, "same language, same way, different shelf"
    assert got[0].rule.name == "as long as it takes"
    assert got[0].total < next(f for f in got if f.rule.name == "a handful").total


def test_the_advantage_shrinks_as_the_data_grows():
    """The derived family grows toward the hand-written five, so its edge erodes.

    Reported because it is what the mechanism predicts and what the numbers do — a constant margin
    would mean the rule was winning by luck rather than by reading the data.
    """
    margins = []
    for width in (4, 8):
        got = stand(_turns(width), width=width)
        best = min(f.total for f in got if f.explained)
        handful = next(f for f in got if f.rule.name == "a handful").total
        margins.append(round(handful - best, 2))
    assert all(m > 0 for m in margins)
    assert margins[0] > margins[1], margins


def test_the_smallest_possible_family_loses_and_that_is_informative():
    got = stand(_turns())
    shortest = next(f for f in got if f.rule.name == "just the shortest")
    assert not shortest.explained
    assert shortest.total > got[0].total * 3


# --------------------------------------------------------------------------------------------- #
#  every choice is on its own line
# --------------------------------------------------------------------------------------------- #
def test_each_choice_is_charged_separately():
    """A rule, a language, a way — three different debts, and a bill that adds them without
    naming them cannot show where anything went."""
    got = found_on(_turns(), RULES[0])
    assert got.choosing_a_rule == pytest.approx(2.0, abs=0.01), "four rules is two bits"
    assert got.choosing_a_language > 0 and got.choosing_a_way > 0
    assert got.toll == pytest.approx(
        got.choosing_a_rule + got.choosing_a_language + got.choosing_a_way, abs=1e-9)
    assert got.total == pytest.approx(got.toll + got.description, abs=0.01)


def test_a_family_with_nothing_in_it_falls_back_to_longhand():
    empty = Rule("nothing at all", lambda obs: [])
    got = found_on(_turns(), empty)
    assert not got.explained and "no languages at all" in got.chose


def test_nothing_to_explain_is_explained_by_no_rule():
    for footing in stand(_unrelated()):
        assert not footing.explained


def test_an_empty_footing_claims_nothing():
    assert Footing().total == 0.0 and not Footing().explained
    assert Footing().to_dict()["rule"] == ""


def test_the_derived_rule_is_not_universally_better_and_the_crossover_is_computed():
    """It wins while the data is small enough to say so, and loses past 32 observations.

    All the working rules pick the same language, so the margin is exactly
    ``log2(5) - log2(L)`` where ``L`` is what the data demands. That is arithmetic, not a search —
    and it means a hand-written cap of five is the better shelf once the data needs more than five,
    which is a real limit of deriving the family and is written down rather than left to be found.
    """
    from nyxara.njp.combining import charged

    margin = lambda L: charged(5) - charged(L)
    assert margin(2) == pytest.approx(1.32, abs=0.01)
    assert margin(3) == pytest.approx(0.74, abs=0.01)
    assert margin(5) == pytest.approx(0.0, abs=1e-9), "a tie at 17 to 32 observations"
    assert margin(6) < 0, "past 32 the hand-written five is the cheaper shelf"
    assert margin(8) < margin(6), "and it keeps getting cheaper"


def test_the_measured_margins_match_the_formula():
    """Two points measured end to end, against the arithmetic that predicts all of them."""
    from nyxara.njp.combining import charged

    for width, needs in ((4, 2), (8, 3)):
        got = stand(_turns(width), width=width)
        best = min(f.total for f in got if f.explained)
        handful = next(f for f in got if f.rule.name == "a handful").total
        assert reach_needed(_turns(width)) == needs
        assert handful - best == pytest.approx(charged(5) - charged(needs), abs=0.02)


# --------------------------------------------------------------------------------------------- #
#  V.98 — the layer turned out not to pay, and that was provable without running anything
# --------------------------------------------------------------------------------------------- #
def test_whether_a_shelf_pays_is_derived_not_searched():
    """Both routes name one language and then pay the same toll and description, so those cancel.

    What is left is `log2(rules) + log2(family)` against `log2(bound)` — which is
    `rules x family < bound`, answerable from three numbers with no observations at all.
    """
    from nyxara.njp.bedrock import pays

    assert not pays(4, 3, 3), "four rules over a family of three, against a bound of three"
    assert not pays(4, 5, 5)
    assert pays(4, 3, 16), "a long bound and a small family is when a shelf earns its place"
    assert pays(1, 1, 2)
    assert not pays(2, 2, 4), "twice two is four, which is not less than four"


def test_naming_the_language_directly_beats_every_rule_on_this_data():
    """V.97's headline was right about which rule is best and wrong about having rules at all."""
    from nyxara.njp.bedrock import direct

    got = stand(_turns())
    assert got[0].rule.name == "no shelf at all"
    straight = direct(_turns())
    best_rule = next(f for f in got if f.rule.name != "no shelf at all" and f.explained)
    assert straight.total < best_rule.total
    assert best_rule.total - straight.total == pytest.approx(2.0, abs=0.01), \
        "exactly log2(4) — the cost of choosing among four rules, buying nothing"


def test_the_baseline_pays_nothing_to_choose_a_rule():
    """The bound has no free numbers in it, so nobody pays to pick it."""
    from nyxara.njp.bedrock import direct

    got = direct(_turns())
    assert got.choosing_a_rule == 0.0
    assert got.choosing_a_language > 0, "it still pays to name which length"
    assert len(got.family) == reach_needed(_turns())


def test_the_baseline_competes_in_the_list_rather_than_beside_it():
    """A route that is never made to compete is a route nobody has checked."""
    names = {f.rule.name for f in stand(_turns())}
    assert "no shelf at all" in names
    assert len(names) == len(RULES) + 1


def test_the_derived_condition_agrees_with_the_measured_totals():
    """The proof and the arithmetic have to say the same thing, or one of them is wrong."""
    from nyxara.njp.bedrock import direct, pays

    obs = _turns()
    bound = reach_needed(obs)
    straight = direct(obs)
    shelf = next(f for f in stand(obs) if f.rule.name == "as long as it takes")
    predicted = pays(len(RULES), len(shelf.family), bound)
    assert predicted == (shelf.total < straight.total), (predicted, shelf.total, straight.total)
