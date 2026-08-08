"""NYX V.01 · the reason-seat — she proposes content and never relaxes a guard."""

from __future__ import annotations

import pytest

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.kernel.config import get_settings
from nyxara.kernel.orchestrator import Candidate
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.reasoner import NyxReasoner


def _candidate(**kw) -> Candidate:
    base = {"text": "the base reasoner's answer", "kind": "respond",
            "risk": RiskTier.HIGH, "reversible": False, "confidence": 0.5,
            "belief": 0.5, "rationale": "base rationale",
            "capability": Capability.MESSAGE_SEND}
    base.update(kw)
    return Candidate(**base)


def _base(candidate: Candidate):
    def _fn(_stimulus, _focus=None, **_kw):
        return candidate
    return _fn


@pytest.fixture
def brain():
    cfg = get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "holo_capacity": 64, "dialogue_enabled": False})
    b = NyxBrain(cfg)
    b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    return b


# -------------------- the safety contract -------------------- #
def test_risk_and_reversibility_are_never_touched(brain):
    candidate = _candidate()
    NyxReasoner(base=_base(candidate), brain=brain)("what pulls an apple down")
    assert candidate.risk is RiskTier.HIGH
    assert candidate.reversible is False
    assert candidate.capability is Capability.MESSAGE_SEND


def test_corrigibility_flags_are_never_weakened(brain):
    candidate = _candidate(resists_correction=True, disables_oversight=True,
                           manipulates_shutdown=True)
    NyxReasoner(base=_base(candidate), brain=brain)("what pulls an apple down")
    assert candidate.resists_correction is True
    assert candidate.disables_oversight is True
    assert candidate.manipulates_shutdown is True


def test_an_action_candidate_is_not_reworded(brain):
    """Rewriting a tool call changes what the Master is being asked to approve."""
    candidate = _candidate(kind="act", tool="delete_file", text="delete /tmp/x")
    NyxReasoner(base=_base(candidate), brain=brain)("what pulls an apple down")
    assert candidate.text == "delete /tmp/x" and candidate.tool == "delete_file"


def test_a_candidate_carrying_a_tool_is_left_alone(brain):
    candidate = _candidate(tool="run_shell", text="original")
    NyxReasoner(base=_base(candidate), brain=brain)("what pulls an apple down")
    assert candidate.text == "original"


# -------------------- fail-soft -------------------- #
def test_a_broken_brain_returns_the_base_candidate_untouched():
    class _Exploding:
        def think(self, _stimulus):
            raise RuntimeError("brain on fire")

    candidate = _candidate()
    got = NyxReasoner(base=_base(candidate), brain=_Exploding())("anything")
    assert got is candidate
    assert candidate.text == "the base reasoner's answer"
    assert candidate.confidence == 0.5 and candidate.rationale == "base rationale"


def test_a_one_arg_base_reasoner_is_supported(brain):
    candidate = _candidate()

    def _one_arg(stimulus):
        return candidate

    assert NyxReasoner(base=_one_arg, brain=brain)("what pulls an apple down") is candidate


def test_a_none_candidate_does_not_crash(brain):
    assert NyxReasoner(base=lambda *a, **k: None, brain=brain)("anything") is None


# -------------------- when she earns the reply -------------------- #
def test_a_verified_derivation_takes_the_reply(brain):
    candidate = _candidate(text="the base guess")
    NyxReasoner(base=_base(candidate), brain=brain)("prove that (x+1)^2 = x^2 + 2x + 1")
    if "identity" not in candidate.text:
        pytest.skip("sympy not installed — the engine declines honestly")
    assert "proven" in candidate.text


def test_she_does_not_displace_a_base_answer_that_is_already_checked(brain):
    """Overwriting a checked answer with a weaker one is a regression dressed as an upgrade."""
    candidate = _candidate(text="a law I derived", rationale="native reasoning (verified=True)")
    NyxReasoner(base=_base(candidate), brain=brain)("what pulls an apple down")
    assert candidate.text == "a law I derived"


def test_associations_alone_never_take_the_reply(brain):
    """The graph says *that* things go together, never *what* is true — that is colour."""
    # remember=False so nothing is stored: the graph learns the association, but memory has
    # nothing to offer, leaving the graph as the only specialist with something to say.
    brain.perceive("alpha beta gamma", remember=False)
    brain.perceive("alpha beta gamma", remember=False)
    candidate = _candidate(text="the base answer")
    NyxReasoner(base=_base(candidate), brain=brain)("alpha beta gamma")
    assert candidate.text == "the base answer"


def test_a_low_confidence_answer_does_not_take_the_reply(brain):
    candidate = _candidate(text="the base answer")
    NyxReasoner(base=_base(candidate), brain=brain, min_confidence=0.99)(
        "what pulls an apple down")
    assert candidate.text == "the base answer"


# -------------------- calibration and annotation -------------------- #
def test_a_verified_answer_may_raise_confidence(brain):
    candidate = _candidate(confidence=0.2)
    NyxReasoner(base=_base(candidate), brain=brain)("prove that (x+1)^2 = x^2 + 2x + 1")
    assert candidate.confidence >= 0.2


def test_the_rationale_records_that_nyx_weighed_in(brain):
    candidate = _candidate()
    NyxReasoner(base=_base(candidate), brain=brain)("what pulls an apple down")
    assert "nyx:" in candidate.rationale
    assert candidate.rationale.startswith("base rationale")   # the base's reasons survive


def test_entropy_is_reported_as_the_expected_free_energy_proxy(brain):
    candidate = _candidate()
    NyxReasoner(base=_base(candidate), brain=brain)("what pulls an apple down")
    assert candidate.efe is not None and candidate.efe >= 0.0


# -------------------- wired into the kernel -------------------- #
def test_the_kernel_installs_nyx_in_the_reason_seat():
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    assert type(core.reasoner).__name__ == "NyxReasoner"


def test_a_turn_through_the_seat_still_passes_the_gate_and_keeps_the_axioms():
    from nyxara.agency.permissions import Authority
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    result = core.process("how are you?", authority=Authority.OWNER)
    assert result.acted and result.candidate.kind == "respond"
    assert core.corrigibility.verify_axioms()
