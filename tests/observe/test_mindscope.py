"""Tests for nyxara.observe.mindscope."""

from __future__ import annotations

import pytest

from nyxara.observe.mindscope import MindScope, Thought, ThoughtKind


def _scope():
    ms = MindScope()
    p = ms.record(ThoughtKind.PERCEPTION, "disk 92%", salience=0.7)
    g = ms.record(ThoughtKind.GOAL, "protect system", salience=0.8)
    r = ms.record(ThoughtKind.INFERENCE, "disk filling", causes=[p], confidence=0.8)
    d = ms.record(ThoughtKind.DECISION, "rotate logs", causes=[r, g], confidence=0.9)
    a = ms.record(ThoughtKind.ACTION, "rotated logs", causes=[d])
    return ms, dict(p=p, g=g, r=r, d=d, a=a)


# -------------------- recording -------------------- #
def test_record_returns_id():
    ms = MindScope()
    tid = ms.record(ThoughtKind.PERCEPTION, "x")
    assert ms.get(tid).content == "x"


def test_record_dangling_cause_refused():
    ms = MindScope()
    with pytest.raises(KeyError):
        ms.record(ThoughtKind.ACTION, "x", causes=["nope"])


def test_record_with_meta():
    ms = MindScope()
    tid = ms.record(ThoughtKind.DECISION, "d", source="planner")
    assert ms.get(tid).meta["source"] == "planner"


def test_len_and_thoughts_order():
    ms, ids = _scope()
    assert len(ms) == 5
    assert [t.id for t in ms.thoughts()] == [ids["p"], ids["g"], ids["r"], ids["d"], ids["a"]]


def test_by_kind():
    ms, _ = _scope()
    assert len(ms.by_kind(ThoughtKind.PERCEPTION)) == 1
    assert len(ms.by_kind(ThoughtKind.DECISION)) == 1


def test_max_thoughts_evicts():
    ms = MindScope(max_thoughts=3)
    for i in range(5):
        ms.record(ThoughtKind.PERCEPTION, f"p{i}")
    assert len(ms) == 3


# -------------------- causal navigation -------------------- #
def test_direct_causes():
    ms, ids = _scope()
    assert {t.id for t in ms.direct_causes(ids["d"])} == {ids["r"], ids["g"]}


def test_direct_effects():
    ms, ids = _scope()
    assert {t.id for t in ms.direct_effects(ids["d"])} == {ids["a"]}


def test_ancestors():
    ms, ids = _scope()
    assert ms.ancestors(ids["a"]) == {ids["p"], ids["g"], ids["r"], ids["d"]}


def test_ancestors_of_root_empty():
    ms, ids = _scope()
    assert ms.ancestors(ids["p"]) == set()


def test_descendants():
    ms, ids = _scope()
    assert ids["d"] in ms.descendants(ids["p"])
    assert ids["a"] in ms.descendants(ids["p"])


def test_navigation_unknown_raises():
    ms = MindScope()
    with pytest.raises(KeyError):
        ms.ancestors("nope")


# -------------------- why / explain -------------------- #
def test_why_returns_causal_chain():
    ms, ids = _scope()
    chain = {t.id for t in ms.why(ids["d"])}
    assert chain == {ids["p"], ids["g"], ids["r"], ids["d"]}


def test_why_excludes_unrelated():
    ms, ids = _scope()
    unrelated = ms.record(ThoughtKind.PERCEPTION, "noise")
    assert unrelated not in {t.id for t in ms.why(ids["d"])}


def test_why_ordered_topologically():
    ms, ids = _scope()
    order = [t.id for t in ms.why(ids["d"])]
    # causes precede effects
    assert order.index(ids["p"]) < order.index(ids["r"]) < order.index(ids["d"])


def test_explain_mentions_premises_and_conclusion():
    ms, ids = _scope()
    exp = ms.explain(ids["d"])
    assert "disk filling" in exp and "protect system" in exp and "rotate logs" in exp


def test_explain_no_cause():
    ms = MindScope()
    tid = ms.record(ThoughtKind.INFERENCE, "a hunch")
    assert "no prior cause" in ms.explain(tid)


def test_reasoning_chain_filters_inputs():
    ms, ids = _scope()
    rc = {t.id for t in ms.reasoning_chain(ids["d"])}
    # perceptions/actions excluded; goals/inferences/decisions kept
    assert ids["p"] not in rc
    assert ids["r"] in rc and ids["d"] in rc and ids["g"] in rc


# -------------------- attention / focus -------------------- #
def test_attention_ranks_by_salience():
    ms = MindScope()
    ms.record(ThoughtKind.PERCEPTION, "low", salience=0.2)
    hi = ms.record(ThoughtKind.PERCEPTION, "high", salience=0.9)
    assert ms.attention(1)[0].id == hi


def test_attention_limit():
    ms = MindScope()
    for i in range(5):
        ms.record(ThoughtKind.PERCEPTION, f"p{i}", salience=0.1 * i)
    assert len(ms.attention(2)) == 2


def test_focus():
    ms, ids = _scope()
    # 'protect system' has the highest salience (0.8)
    assert ms.focus().id == ids["g"]


def test_focus_empty():
    assert MindScope().focus() is None


# -------------------- snapshot / diff -------------------- #
def test_snapshot_diff_changed():
    ms = MindScope()
    ms.snapshot("a", {"x": 1, "y": 2})
    ms.snapshot("b", {"x": 1, "y": 9})
    d = ms.diff("a", "b")
    assert d["changed"] == {"y": {"from": 2, "to": 9}}
    assert d["added"] == {} and d["removed"] == {}


def test_snapshot_diff_added_removed():
    ms = MindScope()
    ms.snapshot("a", {"x": 1})
    ms.snapshot("b", {"y": 2})
    d = ms.diff("a", "b")
    assert d["added"] == {"y": 2} and d["removed"] == {"x": 1}


def test_diff_missing_snapshots():
    ms = MindScope()
    d = ms.diff("nope1", "nope2")
    assert d == {"added": {}, "removed": {}, "changed": {}}


# -------------------- rendering / reporting -------------------- #
def test_render_trace():
    ms, _ = _scope()
    txt = ms.render_trace()
    assert "MindScope trace" in txt and "rotate logs" in txt


def test_render_explanation():
    ms, ids = _scope()
    txt = ms.render_explanation(ids["d"])
    assert "Why" in txt and "rotate logs" in txt


def test_thought_to_dict():
    t = Thought("t0", ThoughtKind.DECISION, "x", confidence=0.9)
    assert t.to_dict()["kind"] == "decision"


def test_to_dict():
    ms, _ = _scope()
    d = ms.to_dict()
    assert len(d["thoughts"]) == 5


def test_report():
    ms, _ = _scope()
    r = ms.report()
    assert r["thoughts"] == 5 and r["by_kind"]["perception"] == 1
    assert r["focus"] == "protect system"
