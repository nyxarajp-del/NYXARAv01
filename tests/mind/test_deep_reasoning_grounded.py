"""Ceiling-break proof: the deep-reasoning ladder now selects by *truth*, not fluency.

The whole point of Problem #1 is that a fixed base model's single-pass answer can be wrong, and
extra test-time compute only helps if the *selector* can tell right from wrong. Before this
change the ladder kept the answer the intrinsic answer_quality verifier scored highest — which
rewards fluent, non-degenerate prose and cannot see correctness. So on a decidable prompt a
fluent-but-wrong answer could beat a terse-but-correct one.

Here a scripted reasoner returns a *fluent, wrong* answer on the early rung and a *terse,
correct* answer on a later rung of a decidable arithmetic prompt. With the grounded verifier
wired in (the default now), the ladder must keep the correct answer; with the old intrinsic
verifier it keeps the fluent-wrong one. Fully offline: the exact faculty oracle decides the
arithmetic with no model, network, or weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.deep_reasoning import RUNG_MCTS, DeepReasoner
from nyxara.mind.router import default_verifier


@dataclass
class _Cand:
    text: str
    kind: str = "respond"
    tool: str = ""
    confidence: float = 0.7
    rationale: str = ""


class _TruthVsFluencyReasoner:
    """Early rungs emit a fluent WRONG answer; the MCTS rung emits the terse CORRECT one."""

    def is_real(self):
        return True

    def decision_to_candidate(self, data, stimulus):
        return _Cand(text=data["text"], kind=data.get("kind", "respond"),
                     tool=data.get("tool", ""), confidence=data.get("confidence", 0.7))

    def deliberate_decision(self, stimulus, *, passes=1, samples=1, temperature=None):
        # fluent, coherent, high-confidence — and wrong (17 + 25 = 42, not 50).
        return {"text": "Carefully adding seventeen and twenty-five, the total comes to fifty.",
                "kind": "respond", "confidence": 0.9}

    def mcts_decision(self, stimulus, *, temperature=None):
        # terse and correct.
        return {"text": "42", "kind": "respond", "confidence": 0.6}


def _settings():
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.deep_reasoning.enabled = True
    return s


def test_ladder_keeps_the_correct_answer_over_fluent_wrong():
    s = _settings()
    assert bool(s.llm.deep_reasoning.ground_verifier)  # grounded selection is on by default
    dr = DeepReasoner(_TruthVsFluencyReasoner(), settings=s)
    res = dr.deliberate("17 + 25")
    assert res is not None
    # the terse, correct answer won — search was steered by truth, not polish.
    assert "42" in res.candidate.text
    assert res.winning_rung == RUNG_MCTS
    assert res.best_score >= 0.9


def test_old_intrinsic_verifier_would_have_kept_the_fluent_wrong_answer():
    # Pin the regression this change fixes: with the *intrinsic* verifier injected, the fluent
    # wrong answer out-scores the terse-correct one and is kept.
    s = _settings()
    dr = DeepReasoner(_TruthVsFluencyReasoner(), settings=s, verifier=default_verifier())
    res = dr.deliberate("17 + 25")
    assert res is not None
    assert "fifty" in res.candidate.text.lower()   # the ceiling, unbroken, keeps the wrong one


def test_ground_verifier_switch_off_falls_back_to_intrinsic():
    s = _settings()
    s.llm.deep_reasoning.ground_verifier = False
    dr = DeepReasoner(_TruthVsFluencyReasoner(), settings=s)
    res = dr.deliberate("17 + 25")
    # with grounding disabled the intrinsic verifier runs → fluent-wrong wins (honest opt-out).
    assert "fifty" in res.candidate.text.lower()
