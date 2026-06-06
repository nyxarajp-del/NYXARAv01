"""Tests for nyxara.growth.skilltree."""

from __future__ import annotations

import pytest

from nyxara.growth.skilltree import (Mastery, Skill, SkillState, SkillTree,
                                     build_default_skilltree)
from nyxara.kernel.errors import InvariantViolation


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _tree(**kw):
    t = SkillTree(unlock_threshold=0.5, mastery_threshold=0.8, **kw)
    t.add_skill("a", difficulty=0.2, value=0.8)
    t.add_skill("b", prerequisites={"a"}, difficulty=0.4, value=0.7)
    t.add_skill("c", prerequisites={"b"}, synergies={"a"}, difficulty=0.5, value=0.9)
    return t


def _master(t, name, n=15):
    for _ in range(n):
        t.practice(name)


# -------------------- definition -------------------- #
def test_add_skill():
    t = SkillTree()
    s = t.add_skill("x", value=0.9)
    assert isinstance(s, Skill) and t.get("x") is s


def test_undefined_prerequisite_raises():
    t = SkillTree()
    with pytest.raises(InvariantViolation):
        t.add_skill("x", prerequisites={"missing"})


def test_skill_to_dict():
    t = _tree()
    d = t.get("c").to_dict()
    assert d["name"] == "c" and d["prerequisites"] == ["b"]


# -------------------- proficiency & status -------------------- #
def test_initial_proficiency_zero():
    assert _tree().proficiency("a") == 0.0


def test_unlocked_no_prereq():
    assert _tree().is_unlocked("a")


def test_locked_with_unmet_prereq():
    assert not _tree().is_unlocked("b")


def test_unlock_after_prereq_mastered():
    t = _tree()
    _master(t, "a")
    assert t.is_unlocked("b")


def test_mastery_levels():
    t = _tree()
    assert t.mastery("b") is Mastery.LOCKED
    _master(t, "a")
    assert t.mastery("b") is Mastery.NOVICE
    _master(t, "b")
    assert t.mastery("b") is Mastery.EXPERT


# -------------------- practice -------------------- #
def test_practice_increases_proficiency():
    t = _tree()
    p0 = t.proficiency("a")
    t.practice("a")
    assert t.proficiency("a") > p0


def test_practice_locked_refused():
    t = _tree()
    with pytest.raises(InvariantViolation):
        t.practice("b")


def test_practice_diminishing_returns():
    t = _tree()
    for _ in range(5):
        t.practice("a")
    g1 = t.practice("a") - t.proficiency("a")  # ~0; measure deltas instead
    before = t.proficiency("a")
    t.practice("a")
    delta_late = t.proficiency("a") - before
    t2 = _tree()
    before0 = t2.proficiency("a")
    t2.practice("a")
    delta_early = t2.proficiency("a") - before0
    assert delta_late < delta_early


def test_practice_caps_at_one():
    t = _tree()
    for _ in range(200):
        t.practice("a")
    assert t.proficiency("a") <= 1.0


def test_practice_records_count_and_time():
    t = _tree(clock=FakeClock(100.0))
    t.practice("a")
    st = t._state["a"]
    assert st.practices == 1 and st.last_practiced == 100.0


# -------------------- synergy -------------------- #
def test_synergy_multiplier_default_one():
    t = _tree()
    assert t.synergy_multiplier("a") == 1.0  # no synergies


def test_synergy_multiplier_boosts():
    t = _tree()
    _master(t, "a")
    assert t.synergy_multiplier("c") > 1.0


def test_synergy_speeds_learning():
    # 'c' learns faster when its synergy 'a' is strong
    strong = _tree()
    _master(strong, "a", 20)
    _master(strong, "b")
    weak = _tree()
    # bring b up without over-practising a (just enough to unlock)
    for _ in range(6):
        weak.practice("a")
    _master(weak, "b")
    g_strong = strong.practice("c")
    g_weak = weak.practice("c")
    assert g_strong > g_weak


# -------------------- decay -------------------- #
def test_decay_reduces_proficiency():
    clk = FakeClock()
    t = _tree(clock=clk, decay_rate=0.1)
    _master(t, "a")
    before = t.proficiency("a")
    clk.advance(20.0)
    t.decay()
    assert t.proficiency("a") < before


def test_decay_no_time_no_change():
    clk = FakeClock()
    t = _tree(clock=clk)
    _master(t, "a")
    before = t.proficiency("a")
    t.decay()  # no time elapsed
    assert t.proficiency("a") == before


def test_decay_can_relock_dependents():
    clk = FakeClock()
    t = _tree(clock=clk, decay_rate=0.2)
    _master(t, "a")
    assert t.is_unlocked("b")
    clk.advance(50.0)
    t.decay()
    assert not t.is_unlocked("b")  # 'a' decayed below unlock threshold


# -------------------- availability & paths -------------------- #
def test_unlocked_skills():
    t = _tree()
    assert "a" in t.unlocked_skills() and "b" not in t.unlocked_skills()


def test_learnable_now_excludes_mastered():
    t = _tree()
    _master(t, "a")
    assert "a" not in t.learnable_now()  # mastered
    assert "b" in t.learnable_now()      # unlocked, not mastered


def test_prerequisite_closure():
    t = _tree()
    assert t.prerequisite_closure("c") == {"a", "b"}


def test_prerequisite_path_order():
    t = _tree()
    path = t.prerequisite_path("c")
    assert path.index("a") < path.index("b") < path.index("c")


def test_prerequisite_path_excludes_mastered():
    t = _tree()
    _master(t, "a")
    path = t.prerequisite_path("c")
    assert "a" not in path  # already mastered


# -------------------- recommendation -------------------- #
def test_recommend_returns_learnable():
    t = _tree()
    _master(t, "a")
    recs = t.recommend(["c"], k=3)
    assert recs
    assert all(t.is_unlocked(r["skill"]) for r in recs)


def test_recommend_empty_when_all_mastered():
    t = _tree()
    _master(t, "a")
    _master(t, "b")
    _master(t, "c")
    assert t.recommend(["c"]) == []


def test_recommend_respects_k():
    t = build_default_skilltree()
    for _ in range(10):
        t.practice("perception")
    recs = t.recommend(["security"], k=2)
    assert len(recs) <= 2


def test_recommend_leverage_scored():
    t = _tree()
    _master(t, "a")
    recs = t.recommend(["c"], k=5)
    # 'b' is foundational (c depends on it) -> appears with leverage
    b = next((r for r in recs if r["skill"] == "b"), None)
    assert b is not None and b["leverage"] >= 1


# -------------------- default tree -------------------- #
def test_default_tree_structure():
    t = build_default_skilltree()
    assert "reasoning" in t.prerequisite_closure("planning")
    assert t.get("security").category == "guard"


def test_default_tree_climb():
    t = build_default_skilltree()
    for s in ("perception", "memory", "language", "reasoning"):
        # unlock by mastering prerequisites first
        for _ in range(15):
            if t.is_unlocked(s):
                t.practice(s)
    assert t.is_unlocked("reasoning")


def test_report():
    t = _tree()
    _master(t, "a")
    r = t.report()
    assert r["skills"] == 3 and "by_mastery" in r
