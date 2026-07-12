"""Tests for the unified own-faculty generalization cascade (mind/generalization.py).

Hermetic: stdlib + numpy only, no LLM anywhere. These assert *behavioural realness* — a
genuinely NEW task, shown by a few examples in the prompt, is solved on a held-out input by
NYXARA's own program-synthesis faculty (there is no language model to fall back on), and the
cascade declines honestly when the prompt carries no learnable structure.
"""

from __future__ import annotations

from nyxara.mind.generalization import (
    GeneralizationEngine,
    GeneralizationResult,
    has_demos,
    parse_demos,
    parse_num_table,
)


# --------------------------------------------------------------------------- #
# In-prompt structure parsing
# --------------------------------------------------------------------------- #
def test_parse_demos_arrow_form_with_query():
    demos, query = parse_demos("pattern: hello -> olleh, world -> dlrow. now: nyxara")
    assert demos == [("hello", "olleh"), ("world", "dlrow")]
    assert query == "nyxara"


def test_parse_demos_what_is_query_and_label_strip():
    demos, query = parse_demos("map ab -> ba, cd -> dc; what is xy?")
    assert demos == [("ab", "ba"), ("cd", "dc")]
    assert query == "xy"


def test_parse_demos_apply_the_rule_label():
    demos, query = parse_demos("apply the rule: ab -> ba, cd -> dc. now: xy")
    assert demos == [("ab", "ba"), ("cd", "dc")]
    assert query == "xy"


def test_has_demos_gate():
    assert has_demos("pattern: hello -> olleh, world -> dlrow. now: nyxara") is True
    assert has_demos("hi how are you today") is False
    # a single demo is not enough to induce a task
    assert has_demos("hello -> olleh. now: world") is False


def test_parse_num_table():
    pairs, query = parse_num_table("map 1 -> 3, 2 -> 5, 3 -> 7, 4 -> 9; what is 10?")
    assert pairs == [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (4.0, 9.0)]
    assert query == 10.0


# --------------------------------------------------------------------------- #
# The cascade — real from-examples generalization, no LLM
# --------------------------------------------------------------------------- #
def test_from_examples_task_solved_on_unseen_input():
    eng = GeneralizationEngine()
    res = eng.generalize("pattern: hello -> olleh, world -> dlrow. now: nyxara")
    assert isinstance(res, GeneralizationResult)
    assert res.source == "skill_induction"
    # the reversal of an UNSEEN input can only come from induced program synthesis (no LLM here)
    assert "araxyn" in res.answer
    assert res.detail.get("output") == "araxyn"
    assert res.confidence >= 0.4


def test_from_examples_compact_oneliner():
    eng = GeneralizationEngine()
    res = eng.generalize("map ab -> ba, cd -> dc; what is xy?")
    assert res is not None and res.source == "skill_induction"
    assert "yx" in res.answer


def test_honest_decline_on_free_chat():
    eng = GeneralizationEngine()
    assert eng.generalize("hi how are you today") is None


def test_declines_without_a_query():
    eng = GeneralizationEngine()
    # demonstrations but no held-out input to apply them to → nothing to answer
    assert eng.generalize("examples: ab -> ba, cd -> dc") is None


def test_exact_faculty_short_circuit():
    eng = GeneralizationEngine()
    res = eng.generalize("what is 2 + 3 * 4")
    assert res is not None and res.source == "faculty"
    assert "14" in res.answer


def test_numeric_table_law_predicted_via_open_world():
    from nyxara.growth.open_world import OpenWorldGeneralizer
    eng = GeneralizationEngine(open_world=OpenWorldGeneralizer())
    res = eng.generalize("map 1 -> 3, 2 -> 5, 3 -> 7, 4 -> 9; what is 10?")
    assert res is not None and res.source == "open_world"
    # affine law 2x+1 (which the string skill-inducer cannot express) → f(10) = 21
    assert "21" in res.answer
    assert abs(float(res.detail["prediction"]) - 21.0) < 1e-3


def test_open_world_declines_without_generalizing_law():
    from nyxara.growth.open_world import OpenWorldGeneralizer
    eng = GeneralizationEngine(open_world=OpenWorldGeneralizer())
    # random, lawless rows → honest decline (no confident model)
    res = eng.generalize("map 1 -> 4, 2 -> 1, 3 -> 9, 4 -> 2; what is 5?")
    assert res is None or res.source != "open_world"


def test_model_dataset_holdout_and_verdict():
    from nyxara.growth.open_world import OpenWorldGeneralizer
    ow = OpenWorldGeneralizer()
    rep = ow.model_dataset([(1, 3), (2, 5), (3, 7), (4, 9)], query=10)
    assert rep.verdict.value == "modelled"
    assert rep.holdout_accuracy >= 0.9
    assert abs(float(rep.predict(10)) - 21.0) < 1e-3


def test_result_to_dict_round_trips_fields():
    r = GeneralizationResult(answer="x", source="skill_induction", confidence=0.7,
                             analogical=False, detail={"k": 1})
    d = r.to_dict()
    assert d["source"] == "skill_induction" and d["confidence"] == 0.7
    assert d["analogical"] is False and d["detail"] == {"k": 1}
