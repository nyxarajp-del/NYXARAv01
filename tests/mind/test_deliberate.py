"""Tests for multi-pass deliberate reasoning (mind/deliberate.py) and its wiring into
the LLM-backed kernel reasoner (mind/llm_reasoner.py)."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import LLMProvider, NyxaraSettings, Profile
from nyxara.mind.critique import Critic
from nyxara.mind.deliberate import DeliberativeReasoner, _vote


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _ScriptLLM:
    """A fake LLM with scripted JSON decisions and a counter, no network."""

    def __init__(self, decisions, *, think="reasoning here", name="fake"):
        self._decisions = list(decisions)
        self._think = think
        self._name = name
        self.json_calls = 0
        self.gen_calls = 0

    # used by DeliberativeReasoner
    def generate(self, prompt, *, system=None, **kw):
        self.gen_calls += 1
        return self._think

    def generate_json(self, prompt, *, system=None, **kw):
        i = min(self.json_calls, len(self._decisions) - 1)
        self.json_calls += 1
        return self._decisions[i]

    # used by LLMReasoner._is_real()
    def chosen_provider(self):
        return type("P", (), {"name": self._name})()


# --------------------------------------------------------------------------- #
# Voting / self-consistency
# --------------------------------------------------------------------------- #
def test_vote_single_sample_passthrough():
    d = {"kind": "respond", "text": "hi", "confidence": 0.5}
    assert _vote([d]) is d


def test_vote_majority_shape_wins_over_outlier():
    samples = [
        {"kind": "respond", "text": "a", "confidence": 0.6},
        {"kind": "respond", "text": "b", "confidence": 0.9},
        {"kind": "act", "tool": "x", "text": "do", "confidence": 0.99},
    ]
    # respond is the majority shape; within it the 0.9 sample is most confident
    assert _vote(samples)["text"] == "b"


def test_vote_tie_breaks_on_confidence():
    samples = [
        {"kind": "respond", "text": "a", "confidence": 0.4},
        {"kind": "act", "tool": "x", "text": "do", "confidence": 0.95},
    ]
    # 1-1 tie on shape -> highest confidence overall wins
    assert _vote(samples)["kind"] == "act"


# --------------------------------------------------------------------------- #
# DeliberativeReasoner stages
# --------------------------------------------------------------------------- #
def test_single_pass_skips_thinking():
    llm = _ScriptLLM([{"kind": "respond", "text": "ok", "confidence": 0.7}])
    d = DeliberativeReasoner(llm, passes=1, samples=1)
    out = d.deliberate(stimulus="hi", context="", decision_instructions="(spec)")
    assert out.scratchpad == ""
    assert llm.gen_calls == 0          # no think pass
    assert llm.json_calls == 1         # one decide call
    assert out.passes == 1


def test_two_pass_thinks_then_decides():
    llm = _ScriptLLM([{"kind": "respond", "text": "ok", "confidence": 0.7}])
    d = DeliberativeReasoner(llm, passes=2, samples=1)
    out = d.deliberate(stimulus="hi", context="", decision_instructions="(spec)")
    assert out.scratchpad == "reasoning here"
    assert llm.gen_calls == 1          # one think pass
    assert llm.json_calls == 1
    assert "deliberation" in out.decision["rationale"]


def test_self_consistency_samples_multiple_decisions():
    decisions = [
        {"kind": "respond", "text": "a", "confidence": 0.6},
        {"kind": "respond", "text": "b", "confidence": 0.9},
        {"kind": "act", "tool": "x", "text": "do", "confidence": 0.99},
    ]
    llm = _ScriptLLM(decisions)
    d = DeliberativeReasoner(llm, passes=2, samples=3)
    out = d.deliberate(stimulus="hi", context="", decision_instructions="(spec)")
    assert len(out.samples) == 3
    assert out.decision["text"] == "b"   # majority-shape, most-confident


def test_three_pass_runs_critique_and_can_revise():
    # draft (sample) then a revised decision from the critique pass
    llm = _ScriptLLM([
        {"kind": "respond", "text": "draft", "confidence": 0.5},
        {"kind": "respond", "text": "revised", "confidence": 0.8},
    ])
    d = DeliberativeReasoner(llm, passes=3, samples=1, critic=Critic())
    out = d.deliberate(stimulus="hi", context="", decision_instructions="(spec)")
    assert out.revised is True
    assert out.decision["text"] == "revised"


def test_no_parseable_decision_raises():
    class _BadLLM(_ScriptLLM):
        def generate_json(self, prompt, *, system=None, **kw):
            return "not a dict"

    d = DeliberativeReasoner(_BadLLM([{}]), passes=2, samples=1)
    with pytest.raises(ValueError):
        d.deliberate(stimulus="hi", context="", decision_instructions="(spec)")


# --------------------------------------------------------------------------- #
# Integration: LLMReasoner uses deliberation when configured
# --------------------------------------------------------------------------- #
def _dev_settings(passes=2, samples=1):
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.llm.provider = LLMProvider.ANTHROPIC
    s.llm.reasoning_passes = passes
    s.llm.reasoning_samples = samples
    return s


def test_reasoner_deliberates_with_real_provider():
    from nyxara.mind.llm_reasoner import LLMReasoner

    llm = _ScriptLLM([{"kind": "respond", "text": "All nominal, Master.",
                       "confidence": 0.8, "rationale": "status"}])
    r = LLMReasoner(llm, settings=_dev_settings(passes=2, samples=1))
    cand = r("status?", None)
    assert cand.kind == "respond"
    assert cand.text == "All nominal, Master."
    assert llm.gen_calls == 1       # the think pass ran -> deliberation was used
    assert "deliberation" in cand.rationale


def test_reasoner_single_pass_does_not_think():
    from nyxara.mind.llm_reasoner import LLMReasoner

    llm = _ScriptLLM([{"kind": "respond", "text": "hi", "confidence": 0.7}])
    r = LLMReasoner(llm, settings=_dev_settings(passes=1, samples=1))
    r("hello", None)
    assert llm.gen_calls == 0       # no think pass on single-shot


def test_reasoner_falls_back_when_deliberation_unparseable():
    from nyxara.mind.llm_reasoner import LLMReasoner

    class _BadLLM(_ScriptLLM):
        def generate_json(self, prompt, *, system=None, **kw):
            return "garbage"

    llm = _BadLLM([{}])
    r = LLMReasoner(llm, settings=_dev_settings(passes=2, samples=1))
    # deliberation raises -> __call__ falls back to the deterministic reasoner, never crashes
    cand = r("delete everything", None)
    assert cand.kind == "act"       # the deterministic reasoner classifies a command


# --------------------------------------------------------------------------- #
# Search-over-reasoning (Pillar B4): pick the best sample by an INDEPENDENT verifier
# --------------------------------------------------------------------------- #
class _MultiLLM:
    """Returns several same-shape 'respond' drafts: a confident-but-empty one and a good one."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, system=None, **kw):
        return "think: answer the Master clearly."

    def generate_json(self, prompt, *, system=None, **kw):
        self.calls += 1
        if self.calls == 1:
            # loudly confident, but a near-empty / degenerate answer
            return {"kind": "respond", "text": "idk", "confidence": 0.99}
        return {"kind": "respond",
                "text": "The capital of France is Paris, Master — a clear, complete answer.",
                "confidence": 0.55}


def test_verifier_selects_quality_over_self_confidence():
    from nyxara.mind.deliberate import DeliberativeReasoner
    from nyxara.mind.router import default_verifier

    d = DeliberativeReasoner(_MultiLLM(), passes=2, samples=2,
                             verifier=default_verifier())
    out = d.deliberate(stimulus="What is the capital of France?",
                       context="", decision_instructions="(spec)")
    assert out.verifier_selected
    assert "Paris" in out.decision["text"]          # the good answer wins, not the 0.99 'idk'
    assert "verifier-scored best-of" in out.trace()


def test_without_verifier_self_confidence_still_wins():
    # default (no verifier) keeps the original self-consistency behaviour: 0.99 'idk' wins
    from nyxara.mind.deliberate import DeliberativeReasoner

    d = DeliberativeReasoner(_MultiLLM(), passes=2, samples=2)
    out = d.deliberate(stimulus="What is the capital of France?",
                       context="", decision_instructions="(spec)")
    assert not out.verifier_selected
    assert out.decision["text"] == "idk"
