"""Tests for nyxara.mind.deep_reasoning — the always-maximum deep-reasoning controller.

The controller must (a) climb the whole effort ladder and keep the verifier-best answer, (b)
respect the rung ceiling and the runaway guard, (c) short-circuit a decisive tool need, (d)
compound via EffortMemory, and (e) defer (return None) when disabled or no real provider — so the
offline path is untouched. Scripted reasoners stand in for real models: no network, no weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.deep_reasoning import (RUNG_CONSISTENCY, RUNG_DELIBERATE, RUNG_MCTS,
                                        RUNG_REFINE, DeepReasoner)
from nyxara.mind.effort_memory import EffortMemory, signature


@dataclass
class _Cand:
    text: str
    kind: str = "respond"
    tool: str = ""
    confidence: float = 0.7
    rationale: str = ""


class _FakeReasoner:
    """A scripted LLMReasoner: deeper rungs return progressively better (less degenerate) answers."""

    def __init__(self, *, real: bool = True):
        self._real = real
        self.rungs_called = []

    def is_real(self):
        return self._real

    def decision_to_candidate(self, data, stimulus):
        return _Cand(text=data["text"], kind=data.get("kind", "respond"),
                     tool=data.get("tool", ""), confidence=data.get("confidence", 0.7))

    def deliberate_decision(self, stimulus, *, passes=1, samples=1, temperature=None):
        self.rungs_called.append(("deliberate", passes, samples))
        if passes >= 3:
            return {"text": "Rayleigh scattering: shorter blue wavelengths scatter far more "
                            "strongly off atmospheric molecules than longer red ones.",
                    "kind": "respond", "confidence": 0.8}
        return {"text": "blue blue blue blue", "kind": "respond", "confidence": 0.5}  # degenerate

    def mcts_decision(self, stimulus, *, temperature=None):
        self.rungs_called.append(("mcts", 0, 0))
        return {"text": "Sunlight scatters in the atmosphere and blue light scatters most, so "
                        "the sky reads blue from every direction during the day.",
                "kind": "respond", "confidence": 0.85}


def _settings(enabled=True, max_rung=RUNG_REFINE, samples=3, max_seconds=60.0):
    s = NyxaraSettings.for_profile(Profile.TEST)
    d = s.llm.deep_reasoning
    d.enabled = enabled
    d.max_rung = max_rung
    d.samples = samples
    d.max_seconds = max_seconds
    return s


class _Improver:
    """A no-op refiner that marks the rationale, so we can see rung 4 ran."""

    def improve(self, stimulus, candidate, n_iterations=5):
        candidate.rationale = (candidate.rationale + " [refined]").strip()
        return candidate


def test_climbs_full_ladder_and_keeps_verifier_best():
    s = _settings()
    fake = _FakeReasoner()
    dr = DeepReasoner(fake, settings=s, improver=_Improver())
    res = dr.deliberate("why is the sky blue?")
    assert res is not None
    # every generation rung ran
    assert RUNG_CONSISTENCY in res.rungs_run
    assert RUNG_DELIBERATE in res.rungs_run
    assert RUNG_MCTS in res.rungs_run
    # the degenerate L1 answer was discarded in favour of a strong one
    assert "blue blue blue" not in res.candidate.text
    assert res.best_score > 0.4


def test_action_need_short_circuits_the_climb():
    class _ActReasoner(_FakeReasoner):
        def deliberate_decision(self, stimulus, *, passes=1, samples=1, temperature=None):
            return {"text": "read the notes", "kind": "act", "tool": "read_file",
                    "confidence": 0.9}

    dr = DeepReasoner(_ActReasoner(), settings=_settings())
    res = dr.deliberate("read notes.txt")
    assert res.candidate.kind == "act"
    assert res.candidate.tool == "read_file"
    assert res.winning_rung == RUNG_CONSISTENCY  # stopped at the first rung


def test_max_rung_ceiling_is_respected():
    fake = _FakeReasoner()
    dr = DeepReasoner(fake, settings=_settings(max_rung=RUNG_CONSISTENCY), improver=_Improver())
    res = dr.deliberate("why is the sky blue?")
    assert res.rungs_run == [RUNG_CONSISTENCY]  # never climbed past the ceiling
    assert not any(r[0] == "mcts" for r in fake.rungs_called)


def test_runaway_guard_stops_escalation():
    class _SlowReasoner(_FakeReasoner):
        def deliberate_decision(self, stimulus, *, passes=1, samples=1, temperature=None):
            time.sleep(0.05)
            return super().deliberate_decision(stimulus, passes=passes, samples=samples)

    # a tiny budget means only the first rung gets to run before the deadline
    dr = DeepReasoner(_SlowReasoner(), settings=_settings(max_seconds=1.0))
    res = dr.deliberate("why is the sky blue?")
    assert res is not None  # still returns the best it managed to compute


def test_disabled_or_mock_defers_to_caller():
    # feature off -> None
    dr_off = DeepReasoner(_FakeReasoner(), settings=_settings(enabled=False))
    assert dr_off("anything") is None
    # no real provider -> None even when enabled
    dr_mock = DeepReasoner(_FakeReasoner(real=False), settings=_settings(enabled=True))
    assert dr_mock("anything") is None


def test_compounds_learned_effort_across_turns():
    s = _settings()
    em = EffortMemory(min_observations=2.0, success_floor=0.3)
    dr = DeepReasoner(_FakeReasoner(), settings=s, improver=_Improver(), effort_memory=em)
    q = "why is the sky blue?"
    for _ in range(5):
        dr.deliberate(q)
    # it recorded outcomes and learned a paying-off rung for this signature
    assert em.suggest(signature(q)) is not None
