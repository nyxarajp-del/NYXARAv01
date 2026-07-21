"""Tests for the native chain-of-thought rung wired into NyxaraReasoner's ladder.

The native rung must (a) answer FIRST — before any model is consulted, with the LLM
never called for a turn her own reasoning settles; (b) leave the inspectable trace on
``last_native_trace``; and (c) restore the exact prior ladder when the flag is off.
"""

from __future__ import annotations

import random

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.causal_world_model import CausalWorldModel
from nyxara.mind.nyxara_reasoner import NyxaraReasoner


class _CountingProvider:
    name = "qwen"  # not "native" → _real_llm() is True, so the teacher path is live


class _CountingLLM:
    """A real-provider facade that counts every generate() call."""

    def __init__(self, settings):
        self.settings = settings
        self.calls = 0

    def chosen_provider(self):
        return _CountingProvider()

    def generate(self, prompt, system=None, **kw):
        self.calls += 1
        return "[teacher-fallback] an external answer"


def _chain_model(seed: int = 7, n: int = 300) -> CausalWorldModel:
    rng = random.Random(seed)
    m = CausalWorldModel(window=10.0)
    for k in range(n):
        base = k * 100.0
        if rng.random() < 0.5:
            m.observe("rain", at=base)
            if rng.random() < 0.9:
                m.observe("wet_ground", at=base + 1)
                if rng.random() < 0.85:
                    m.observe("slippery", at=base + 2)
    m.discover()
    return m


def _reasoner(tmp_path, **kw):
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.paths.data_dir = tmp_path
    llm = _CountingLLM(s)
    r = NyxaraReasoner(llm=llm, settings=s, causal_model=_chain_model(), **kw)
    return r, llm, s


def test_native_rung_answers_a_causal_question_without_the_llm(tmp_path):
    r, llm, _ = _reasoner(tmp_path)
    cand = r("why is the ground slippery?")
    assert cand.kind == "respond"
    assert "native reasoning" in cand.rationale
    assert "wet_ground" in cand.text or "rain" in cand.text
    assert llm.calls == 0  # the LLM authored nothing


def test_native_trace_is_left_for_inspection(tmp_path):
    r, _, _ = _reasoner(tmp_path)
    r("why is the ground slippery?")
    assert r.last_native_trace
    kinds = [d["kind"] for d in r.last_native_trace]
    assert "perceive" in kinds and "reason" in kinds and "decide" in kinds


def test_trace_in_answer_flag_controls_the_rendered_reply(tmp_path):
    r, _, s = _reasoner(tmp_path)
    s.native_reasoning.trace_in_answer = False
    cand = r("why is the ground slippery?")
    assert "my own reasoning" not in cand.text
    assert r.last_native_trace  # the trace is still recorded for report()


def test_flag_off_restores_the_prior_ladder_exactly(tmp_path):
    r, llm, s = _reasoner(tmp_path)
    s.native_reasoning.enabled = False
    cand = r("what is 12 * 12?")
    # the old verifiable-faculty rung fires, as before the native rung existed
    assert "verifiable faculty" in cand.rationale
    assert "144" in cand.text
    assert llm.calls == 0


def test_ordinary_chat_falls_through_to_the_teacher(tmp_path):
    r, llm, _ = _reasoner(tmp_path)
    cand = r("Tell me a little about your day.")
    assert "native reasoning" not in cand.rationale
    assert llm.calls >= 1  # the LLM is the last resort, and only for open chat


def test_record_native_outcome_feeds_her_learning(tmp_path):
    r, _, _ = _reasoner(tmp_path)
    r("why is the ground slippery?")
    r.record_native_outcome(True)  # never raises; feeds calibration + bandit + replay
    native = r._native
    assert native and native.replay_path
    import os
    assert os.path.exists(native.replay_path)
