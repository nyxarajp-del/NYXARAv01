"""Tests for nyxara.mind.mcts_reasoner — Monte Carlo Tree Search deep reasoning."""

from __future__ import annotations

import pytest

from nyxara.mind.mcts_reasoner import MCTSReasoner, MCTSUnavailable, _Node


class _ScriptedLLM:
    """Branches into steps on expansion; reports higher confidence on deeper traces."""

    def generate(self, prompt, *, system=None, **kw):
        return "Consider the arithmetic directly."

    def generate_json(self, prompt, *, system=None, **kw):
        if "JSON array" in prompt:
            return ["check the units", "compute step by step",
                    "estimate then verify", "look for a shortcut"]
        depth = prompt.count("\n  ")
        return {"kind": "respond", "text": "The product is 42.",
                "confidence": min(0.95, 0.55 + 0.05 * depth),
                "rationale": "worked it out", "risk": "low", "reversible": True}


def _verifier(_prompt: str, answer: str) -> float:
    return 0.9 if "42" in answer else 0.2


def test_search_builds_tree_and_returns_decision():
    r = MCTSReasoner(_ScriptedLLM(), decision_instructions="(spec)",
                     iterations=12, max_children=4, rollout_depth=2, verifier=_verifier)
    out = r.search(stimulus="What is 6 times 7?", context="")
    assert out.decision["kind"] == "respond"
    assert "42" in out.decision["text"]
    assert out.nodes > 1 and out.iterations >= 1          # the tree genuinely grew
    assert out.score >= 0.9                               # verifier picked the good answer
    assert "MCTS" in out.decision["rationale"]            # auditable trace annotated


def test_backpropagation_increments_ancestor_stats():
    root = _Node(steps=[])
    child = _Node(steps=["a"], parent=root)
    grand = _Node(steps=["a", "b"], parent=child)
    MCTSReasoner._backprop(grand, 1.0)
    assert grand.visits == 1 and child.visits == 1 and root.visits == 1
    assert grand.value == 1.0 and root.value == 1.0
    assert grand.q == 1.0


def test_uct_selection_prefers_higher_value_child():
    parent = _Node(steps=[])
    good = _Node(steps=["g"], parent=parent, visits=5, value=4.5)   # q = 0.9
    bad = _Node(steps=["b"], parent=parent, visits=5, value=0.5)    # q = 0.1
    parent.children = [good, bad]
    parent.visits = 10
    r = MCTSReasoner(_ScriptedLLM(), decision_instructions="(spec)", c_puct=0.1)
    assert r._uct_child(parent) is good


def test_exploration_constant_revisits_underexplored_child():
    parent = _Node(steps=[])
    seen = _Node(steps=["s"], parent=parent, visits=50, value=45.0)   # q = 0.9, well explored
    fresh = _Node(steps=["f"], parent=parent, visits=1, value=0.5)    # q = 0.5, barely tried
    parent.children = [seen, fresh]
    parent.visits = 51
    high_c = MCTSReasoner(_ScriptedLLM(), decision_instructions="(spec)", c_puct=8.0)
    assert high_c._uct_child(parent) is fresh                          # exploration wins


def test_graceful_fallback_when_no_decision_parses():
    class _Dead:
        def generate(self, *a, **k):
            return ""
        def generate_json(self, *a, **k):
            return None
    with pytest.raises(MCTSUnavailable):
        MCTSReasoner(_Dead(), decision_instructions="x", iterations=3).search(
            stimulus="hello", context="")


def test_rlsp_hardening_is_applied_when_wired():
    class _RLSP:
        def harden(self, stimulus, decision):
            from types import SimpleNamespace
            d = dict(decision)
            d["text"] = d["text"] + " (hardened)"
            return SimpleNamespace(decision=d, rounds=2)

    r = MCTSReasoner(_ScriptedLLM(), decision_instructions="(spec)", iterations=8,
                     verifier=_verifier, rlsp=_RLSP())
    out = r.search(stimulus="What is 6 times 7?", context="")
    assert "(hardened)" in out.decision["text"]
    assert out.rlsp_rounds == 2
    assert "hardened" in out.decision["rationale"]


def test_respects_iteration_budget():
    r = MCTSReasoner(_ScriptedLLM(), decision_instructions="(spec)",
                     iterations=3, verifier=_verifier)
    out = r.search(stimulus="What is 6 times 7?", context="")
    assert out.iterations <= 3
