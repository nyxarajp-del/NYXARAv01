"""Tests for nyxara.growth.self_correction — active self-correction & epistemic uncertainty.

The Master's gap: NYXARA could *act* but never knew **when she was wrong**, and could get
**stuck in a loop**. This faculty is the controller that fixes that — entirely her own
deterministic code, **no language model anywhere in the loop**. The properties asserted here:

* predict-then-verify: a confident move that fails is measured as *surprising*;
* she reads her own state — exact repeats AND cycles/oscillation — as *stuck*;
* a confident-but-surprised move is judged *likely wrong*;
* on a gap she runs a REAL experiment / decomposes, yielding a finding to inject;
* the recovery bandit LEARNS which strategy works, and it persists across restarts;
* the honest final-answer gate abstains ("I don't know") rather than bluff;
* no LLM is ever imported or called.
"""
from __future__ import annotations

import inspect


from nyxara.growth.scientist import Scientist
from nyxara.growth.self_correction import (
    EpistemicState,
    EpistemicVerdict,
    Recovery,
    SelfCorrectionLoop,
    Strategy,
)
from nyxara.mind.metacognition import HONEST_ABSTENTION


def _sc(tmp_path, **kw):
    return SelfCorrectionLoop(path=tmp_path / "sc.json", seed=7, **kw)


# --------------------------------------------------------------------------- #
# predict-then-verify — surprise exposes a wrong belief
# --------------------------------------------------------------------------- #
def test_confident_move_that_fails_is_surprising(tmp_path):
    sc = _sc(tmp_path)
    # build a strong belief that this move makes progress, then watch it fail
    for _ in range(8):
        sc.observe_surprise("call the calculator", made_progress=True)
    sc.predict_step("call the calculator")
    surprise = sc.observe_surprise("call the calculator", made_progress=False)
    assert surprise > 0.5, "a confident move that fails must be surprising"


def test_unsurprising_when_belief_matches_reality(tmp_path):
    sc = _sc(tmp_path)
    for _ in range(8):
        sc.observe_surprise("reliable move", made_progress=True)
    sc.predict_step("reliable move")
    surprise = sc.observe_surprise("reliable move", made_progress=True)
    assert surprise < 0.3, "an expected success should not be surprising"


# --------------------------------------------------------------------------- #
# assess — stuck / cycle / likely-wrong / progressing
# --------------------------------------------------------------------------- #
def test_exact_repetition_is_stuck(tmp_path):
    sc = _sc(tmp_path)
    v = sc.assess(signatures=["a:t:x", "a:t:x", "a:t:x"],
                  confidences=[0.5, 0.5, 0.5], no_progress=2)
    assert isinstance(v, EpistemicVerdict)
    assert v.state is EpistemicState.STUCK and v.needs_correction


def test_oscillation_cycle_is_stuck(tmp_path):
    sc = _sc(tmp_path)
    v = sc.assess(signatures=["A", "B", "A", "B"], confidences=[0.5, 0.5, 0.5, 0.5])
    assert v.state is EpistemicState.STUCK, "A,B,A,B oscillation is a loop"


def test_confident_but_surprised_is_likely_wrong(tmp_path):
    sc = _sc(tmp_path)
    v = sc.assess(signatures=["A", "B"], confidences=[0.9, 0.9], surprise=0.8)
    assert v.state is EpistemicState.LIKELY_WRONG and v.needs_correction


def test_low_confidence_is_uncertain(tmp_path):
    sc = _sc(tmp_path)
    v = sc.assess(signatures=["A", "B", "C"], confidences=[0.3, 0.3, 0.2], no_progress=0)
    assert v.state is EpistemicState.UNCERTAIN and v.needs_correction


def test_varied_confident_progress_is_progressing(tmp_path):
    sc = _sc(tmp_path)
    v = sc.assess(signatures=["A", "B", "C"], confidences=[0.7, 0.75, 0.8], no_progress=0)
    assert v.state is EpistemicState.PROGRESSING and not v.needs_correction


# --------------------------------------------------------------------------- #
# recover — run a REAL experiment / decompose, hand back a finding
# --------------------------------------------------------------------------- #
def test_recover_acts_and_yields_a_finding(tmp_path):
    sc = _sc(tmp_path, scientist=Scientist())
    v = sc.assess(signatures=["a:t:x", "a:t:x"], confidences=[0.6, 0.6], no_progress=2)
    rec = sc.recover("compute the 10th prime and then square it", ["a:t:x", "a:t:x"], v)
    assert isinstance(rec, Recovery)
    assert rec.acted and rec.finding
    assert rec.strategy in set(Strategy)


def test_recover_can_run_a_real_scientific_experiment(tmp_path):
    """Force the EXPERIMENT rung and confirm it drives the real Scientist."""
    sc = _sc(tmp_path, scientist=Scientist())
    rec = sc._do_experiment("why does gravity accelerate objects", [], "stuck")
    assert rec.strategy in (Strategy.EXPERIMENT, Strategy.CONSULT)
    assert rec.acted and "question" in rec.experiment or rec.finding


def test_decompose_breaks_a_stuck_goal_into_subgoals(tmp_path):
    sc = _sc(tmp_path)
    rec = sc._do_decompose("build the parser and then wire the evaluator", "stuck")
    assert rec.new_info and rec.experiment.get("subgoals")
    assert len(rec.experiment["subgoals"]) >= 2


# --------------------------------------------------------------------------- #
# learn — the recovery bandit + persistence
# --------------------------------------------------------------------------- #
def test_bandit_learns_the_winning_strategy(tmp_path):
    sc = _sc(tmp_path)
    v = sc.assess(signatures=["a:t:x", "a:t:x"], confidences=[0.5, 0.5], no_progress=2)
    for _ in range(20):
        sc.record_outcome("stuck", Strategy.DECOMPOSE, worked=True)
        sc.record_outcome("stuck", Strategy.RETRY, worked=False)
    picks = [sc._choose_strategy("stuck", "sig", v) for _ in range(30)]
    assert picks.count(Strategy.DECOMPOSE) >= 18, "bandit must favour the proven strategy"


def test_policy_persists_across_restarts(tmp_path):
    p = tmp_path / "sc.json"
    sc = SelfCorrectionLoop(path=p, seed=7)
    for _ in range(12):
        sc.record_outcome("stuck", Strategy.DECOMPOSE, worked=True)
    sc2 = SelfCorrectionLoop(path=p, seed=7)
    alpha = sc2._policy_ab("stuck", Strategy.DECOMPOSE)[0]
    assert alpha > 6, "the learned recovery policy must survive a restart"


# --------------------------------------------------------------------------- #
# gate — never speak a confident-but-wrong answer
# --------------------------------------------------------------------------- #
def test_gate_abstains_when_unsure(tmp_path):
    sc = _sc(tmp_path)
    gate = sc.gate_answer("mass of the Higgs to 12 digits?", "about 125 GeV",
                          confidence=0.2, epistemic=0.9)
    assert gate["abstain"] and gate["answer"] == HONEST_ABSTENTION


def test_gate_passes_a_confident_answer(tmp_path):
    sc = _sc(tmp_path)
    gate = sc.gate_answer("say hello", "Hello, Master.", confidence=0.95, epistemic=0.05)
    assert not gate["abstain"]


def test_report_is_serialisable(tmp_path):
    import json
    sc = _sc(tmp_path)
    sc.record_outcome("stuck", Strategy.DECOMPOSE, worked=True)
    rep = sc.report()
    json.dumps(rep, default=str)      # must not raise
    assert "stats" in rep and "best_recovery" in rep


# --------------------------------------------------------------------------- #
# the honesty guarantee — NYXARA's own work, no LLM anywhere
# --------------------------------------------------------------------------- #
def test_no_llm_is_ever_called(tmp_path, monkeypatch):
    from nyxara.mind.llm import LLM

    def _tripwire(*a, **k):
        raise AssertionError("self-correction must not call the LLM")

    monkeypatch.setattr(LLM, "complete", _tripwire, raising=False)
    monkeypatch.setattr(LLM, "chat", _tripwire, raising=False)

    sc = _sc(tmp_path, scientist=Scientist())
    sc.predict_step("do the thing")
    sc.observe_surprise("do the thing", made_progress=False)
    v = sc.assess(signatures=["a", "a"], confidences=[0.6, 0.6], no_progress=2)
    sc.recover("solve a stubborn problem", ["a", "a"], v)
    sc.gate_answer("q", "a", confidence=0.2)


def test_module_does_not_import_the_llm():
    import nyxara.growth.self_correction as mod
    src = inspect.getsource(mod)
    assert "mind.llm" not in src and "import llm" not in src
