"""Tests for the NyxaraReasoner ↔ metacontrol wiring — the plan actually steers the mind.

The per-turn ComputeBudget must (a) make an entry-rung-0 (easy) turn skip the deep ladder AND
the MCTS fallback — one forward pass means one forward pass, (b) feed one shared calibrated
difficulty/novelty/stakes judgment to the dual-process arbitrator, (c) be computed by the
reasoner itself when used standalone (no kernel installed a plan), and (d) never leak across
turns. All offline: TEST profile, scripted stubs, no network.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.metacontrol import ComputeBudget, MetacognitiveController
from nyxara.mind.nyxara_reasoner import NyxaraReasoner


def _settings() -> NyxaraSettings:
    return NyxaraSettings.for_profile(Profile.TEST)


def _fast_plan(**kwargs) -> ComputeBudget:
    kwargs.setdefault("max_rung", 1)
    kwargs.setdefault("samples", 1)
    kwargs.setdefault("max_seconds", 5.0)
    kwargs.setdefault("entry_rung", 0)
    return ComputeBudget(**kwargs)


def test_fast_path_skips_deep_ladder_and_mcts():
    r = NyxaraReasoner(settings=_settings())
    calls = []
    r._deep = lambda stimulus, focus, budget=None: calls.append("deep")  # a live stub ladder
    r.install_turn_plan(_fast_plan())
    assert r._deep_respond("hello", [], "system_1") is None
    assert r._mcts_respond("hello", [], "system_1") is None
    assert calls == []  # the ladder was never invoked


def test_deep_ladder_receives_the_turn_budget():
    r = NyxaraReasoner(settings=_settings())
    seen = {}

    def _deep(stimulus, focus, budget=None):
        seen["budget"] = budget
        return None

    r._deep = _deep
    plan = ComputeBudget(max_rung=3, samples=5, max_seconds=30.0, entry_rung=2)
    r.install_turn_plan(plan)
    r._deep_respond("a hard question", [], "system_2")
    assert seen["budget"] is plan


def test_route_uses_shared_difficulty_estimate():
    s = _settings()
    mc = MetacognitiveController(settings=s, path=None)
    r = NyxaraReasoner(settings=s)
    captured = {}

    class _DP:
        def think(self, task, stakes=0.0):
            captured["features"] = dict(task.features)
            return None

    r._dual = _DP()
    plan = mc.plan("prove the theorem rigorously from first principles",
                   novelty=0.7, recall_strength=0.1)
    r.install_turn_plan(plan)
    r._route("prove the theorem rigorously from first principles", [])
    feats = captured["features"]
    assert feats["difficulty"] == plan.estimate.calibrated
    assert feats["novelty"] == plan.estimate.signals.novelty
    assert feats["stakes"] == plan.estimate.signals.stakes


def test_standalone_reasoner_computes_its_own_plan():
    s = _settings()
    mc = MetacognitiveController(settings=s, path=None)
    r = NyxaraReasoner(settings=s, metacontrol=mc)
    assert mc.last_plan is None
    r("hello there")
    assert mc.last_plan is not None            # the reasoner planned for itself
    assert r._turn_plan is None                # ...and the plan never leaks past the turn


def test_llm_reasoner_receives_plan_handoff():
    class _LLMReasonerStub:
        turn_plan = None

    r = NyxaraReasoner(settings=_settings(), llm_reasoner=_LLMReasonerStub())
    plan = _fast_plan()
    r.install_turn_plan(plan)
    assert r.llm_reasoner.turn_plan is plan
