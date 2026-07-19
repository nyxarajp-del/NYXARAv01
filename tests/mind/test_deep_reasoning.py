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


# --------------------------------------------------------------------------- #
# First-class metacognition: calibrated uncertainty drives the ladder's spend
# --------------------------------------------------------------------------- #
def _allocator(**kw):
    from nyxara.mind.metacognitive_allocator import MetacognitiveAllocator
    return MetacognitiveAllocator(**kw)


def test_allocator_none_is_byte_identical_always_max():
    """No allocator ⇒ the exact pre-existing always-max behaviour (zero regression)."""
    s = _settings()
    dr = DeepReasoner(_FakeReasoner(), settings=s, improver=_Improver(), allocator=None)
    res = dr.deliberate("why is the sky blue?")
    assert RUNG_CONSISTENCY in res.rungs_run and RUNG_MCTS in res.rungs_run
    assert RUNG_REFINE in res.rungs_run  # the full ladder still runs
    assert not res.stopped_early and not res.escalated


class _GoodL1Reasoner(_FakeReasoner):
    """A scripted reasoner whose FIRST pass is already a strong answer (a genuinely easy turn)."""

    def deliberate_decision(self, stimulus, *, passes=1, samples=1, temperature=None):
        self.rungs_called.append(("deliberate", passes, samples))
        return {"text": "Rayleigh scattering: short blue wavelengths scatter far more strongly "
                        "off atmospheric molecules than long red ones, so the sky reads blue.",
                "kind": "respond", "confidence": 0.9}


def test_learned_easy_signature_gets_one_pass():
    """An easy signature whose first pass is strong climbs exactly one rung and stops."""
    s = _settings()
    alloc = _allocator(easy_uncertainty=0.25, hard_uncertainty=0.70)
    q = "why is the sky blue?"
    for _ in range(12):
        alloc.record(q, predicted_uncertainty=0.8, target_score=0.8,
                     scores=[(1, 0.9)], winning_rung=1)
    dr = DeepReasoner(_GoodL1Reasoner(), settings=s, improver=_Improver(), allocator=alloc)
    res = dr.deliberate(q)
    assert res.rungs_run == [RUNG_CONSISTENCY], res.rungs_run
    assert res.budget_reason and "1 pass" in res.budget_reason
    assert not res.escalated  # a strong first pass never needed escalation


def test_unknown_signature_climbs_past_one_pass():
    """An unknown/hard problem is NOT assumed cheap — epistemic humility spends compute."""
    s = _settings()
    alloc = _allocator(easy_uncertainty=0.25, hard_uncertainty=0.70)
    dr = DeepReasoner(_FakeReasoner(), settings=s, improver=_Improver(), allocator=alloc)
    res = dr.deliberate("prove sum_{k=1}^n k**3 == (n*(n+1)//2)**2 for all positive integers n")
    assert len(res.rungs_run) > 1


def test_target_met_early_stops_the_climb():
    """Once the verifier clears the target, the climb halts even with budget to spare."""
    class _StrongReasoner(_FakeReasoner):
        def deliberate_decision(self, stimulus, *, passes=1, samples=1, temperature=None):
            # a strong answer even at the shallow rung → target met immediately
            return {"text": "Rayleigh scattering: short blue wavelengths scatter far more "
                            "strongly off atmospheric molecules than long red ones.",
                    "kind": "respond", "confidence": 0.9}

    s = _settings()
    alloc = _allocator(easy_uncertainty=0.25, hard_uncertainty=0.70, base_target=0.5)
    # make this signature predict mid/hard so max_rung > 1 (proves the STOP is on target, not ceiling)
    q = "explain, in careful detail, the physical cause of the daytime sky colour please"
    for _ in range(10):
        alloc.record(q, predicted_uncertainty=0.3, target_score=0.5,
                     scores=[(1, 0.3), (2, 0.55)], winning_rung=2)
    dr = DeepReasoner(_StrongReasoner(), settings=s, improver=_Improver(), allocator=alloc)
    res = dr.deliberate(q)
    assert res.stopped_early
    assert RUNG_MCTS not in res.rungs_run  # never needed the deeper rungs


def test_live_escalation_when_measured_harder_than_predicted():
    """Predicted easy but the first pass scores weak → escalate the budget mid-climb."""
    class _WeakThenStrong(_FakeReasoner):
        def deliberate_decision(self, stimulus, *, passes=1, samples=1, temperature=None):
            if passes >= 3:
                return {"text": "Rayleigh scattering: short blue wavelengths scatter far more "
                                "strongly off atmospheric molecules than long red ones.",
                        "kind": "respond", "confidence": 0.85}
            return {"text": "blah blah blah blah", "kind": "respond",
                    "confidence": 0.2}  # degenerate first pass (verifier scores it ~0)

    s = _settings()
    alloc = _allocator(easy_uncertainty=0.25, hard_uncertainty=0.70,
                       base_target=0.8, live_escalation=True)
    q = "why is the sky blue?"
    for _ in range(12):   # teach it this signature is easy
        alloc.record(q, predicted_uncertainty=0.8, target_score=0.8,
                     scores=[(1, 0.9)], winning_rung=1)
    dr = DeepReasoner(_WeakThenStrong(), settings=s, improver=_Improver(), allocator=alloc)
    res = dr.deliberate(q)
    # predicted easy (1 rung) but the weak first pass forced a mid-climb escalation past it
    assert res.escalated
    assert len(res.rungs_run) > 1


def test_allocator_record_is_called_with_real_outcomes():
    """The closed loop actually fires — the allocator learns from each lived turn."""
    s = _settings()

    class _Spy:
        def __init__(self):
            self.recorded = []
            from nyxara.mind.metacognitive_allocator import MetacognitiveAllocator
            self._inner = MetacognitiveAllocator()

        def allocate(self, stimulus, *, ceiling=None, mems_count=0):
            return self._inner.allocate(stimulus, ceiling=ceiling, mems_count=mems_count)

        def should_continue(self, **kw):
            return self._inner.should_continue(**kw)

        def escalate(self, budget, *, ceiling=None):
            return self._inner.escalate(budget, ceiling=ceiling)

        def record(self, stimulus, **kw):
            self.recorded.append((stimulus, kw))

    spy = _Spy()
    dr = DeepReasoner(_FakeReasoner(), settings=s, improver=_Improver(), allocator=spy)
    dr.deliberate("why is the sky blue?")
    assert spy.recorded, "record must be called once per turn"
    _, kw = spy.recorded[-1]
    assert kw["scores"] and kw["winning_rung"] >= 1
    assert "predicted_uncertainty" in kw and "rung_seconds" in kw
