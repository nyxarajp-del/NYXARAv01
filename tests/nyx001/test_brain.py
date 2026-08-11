"""The merge itself: three brains, one thought, and a reason-seat that cannot widen its powers.

:func:`test_seat_never_touches_the_safety_envelope` is the most important test in this package.
The reason-seat sits outermost in NYXARA's chain and sees every turn; if it could lower a risk
tier or clear a corrigibility flag, every other safety property in the repo would be downstream of
a bug in one file.
"""

from __future__ import annotations

import pytest

from nyxara.nyx001 import (
    Curriculum,
    DarkCore,
    Fused,
    Fusion,
    NyxV001Brain,
    NyxV001Reasoner,
    Vote,
)
from nyxara.nyx001.layers import CognitiveStack


class Cfg:
    seed = 31
    scale = "toy"
    lr = 5e-3
    memory_capacity = 256
    layers_enabled = True
    v03_enabled = True
    snn_enabled = True
    fusion_enabled = True
    dark_core_enabled = True
    development_enabled = True


@pytest.fixture(scope="module")
def brain():
    return NyxV001Brain(Cfg())


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def test_verifiable_beats_a_far_more_confident_guess():
    """The one control law this repo enforces rather than weights."""
    f = Fusion()
    out = f.fuse([Vote(source="stack", text="a checked derivation", confidence=0.30,
                       verifiable=True),
                  Vote(source="v03", text="a very loud guess", confidence=0.99)])
    assert out.winner.source == "stack"
    assert out.verifiable
    assert "verifiable" in out.reason


def test_a_source_with_no_record_cannot_lend_authority():
    """Confidence is scaled by MEASURED reliability; an untested source gets the prior."""
    f = Fusion(prior=0.5, min_samples=5)
    out = f.fuse([Vote(source="v03", text="an answer", confidence=1.0)])
    assert out.confidence == pytest.approx(0.5), "an unproven source kept its full confidence"


def test_reliability_is_earned_and_changes_the_winner():
    f = Fusion(alpha=0.5, min_samples=1)
    for _ in range(6):
        f.resolve("stack", correct=1.0)
        f.resolve("v03", correct=0.0)
    out = f.fuse([Vote(source="v03", text="v03 says this", confidence=0.7),
                  Vote(source="stack", text="stack says that", confidence=0.6)])
    assert out.winner.source == "stack", "a proven source lost to an unproven, louder one"


def test_disagreement_is_surfaced_not_smoothed():
    f = Fusion(prior=1.0, min_samples=0)
    out = f.fuse([Vote(source="v03", text="the sky is blue today", confidence=0.9),
                  Vote(source="stack", text="entirely unrelated vocabulary", confidence=0.85)])
    assert out.agreement < 0.34
    assert out.contested, "high confidence on low agreement was not flagged"
    assert "CONTESTED" in f.explain(out)


def test_fusion_with_no_content_produces_no_thought():
    out = Fusion().fuse([Vote(source="v03", text=""), Vote(source="stack", text="   ")])
    assert out.winner is None and out.answer == ""
    assert "no source" in out.reason


# --------------------------------------------------------------------------- #
# The brain
# --------------------------------------------------------------------------- #
def test_brain_holds_all_three_minds(brain):
    st = brain.stats()
    assert st["brains"]["stack"] is True, "the Layer 0-17 stack is the point of NYX V.001"
    assert isinstance(brain.stack, CognitiveStack)


def test_think_produces_a_fused_thought(brain):
    t = brain.think("alpha beta gamma flows onward")
    assert t.cycle_id
    assert t.fused is not None
    assert t.to_dict()["fused"] is not None


def test_a_brain_with_one_mind_disabled_still_thinks():
    class Solo(Cfg):
        v03_enabled = False
        snn_enabled = False
    b = NyxV001Brain(Solo())
    assert b.v03 is None and b.snn is None and b.stack is not None
    assert b.think("alpha beta").to_dict() is not None


def test_empty_stimulus_is_a_null_thought(brain):
    assert brain.think("   ").answer == ""


def test_brain_round_trips_through_a_sidecar():
    b = NyxV001Brain(Cfg())
    for _ in range(3):
        b.think("alpha beta gamma")
    fresh = NyxV001Brain(Cfg())
    fresh.load_dict(b.to_dict())
    assert fresh.turns == b.turns


def test_is_learning_reports_none_before_it_can_say():
    assert NyxV001Brain(Cfg()).is_learning() is None


# --------------------------------------------------------------------------- #
# The dark core and the childhood
# --------------------------------------------------------------------------- #
def test_dark_core_always_has_a_next_move_once_there_is_experience():
    stack = CognitiveStack(Cfg())
    for tok in ["alpha one", "beta two", "gamma three"] * 8:
        stack.cycle(tok)
    p = DarkCore(stack).pulse()
    assert not p.starved, "the frontier went empty despite real experience"
    assert p.chosen is not None and p.drive > 0.0


def test_dark_core_is_pulled_toward_confident_untested_beliefs():
    """Rigidity enters with a POSITIVE weight — that is what makes it dark, not merely curious."""
    stack = CognitiveStack(Cfg())
    for tok in ["alpha one", "beta two"] * 8:
        stack.cycle(tok)
    h = stack.reasoning.propose("a confident untested claim")
    h.confidence = 0.99
    p = DarkCore(stack, w_gain=0.0, w_uncertainty=0.0).pulse()
    assert p.chosen is not None
    assert p.chosen.kind in ("falsify", "experiment"), \
        f"with gain and uncertainty zeroed it should attack, not {p.chosen.kind}"


def test_curriculum_cannot_be_climbed_out_of_order():
    """No-skip: a stage whose criterion is met stays inactive while a prerequisite is not."""
    stack = CognitiveStack(Cfg())
    c = Curriculum(stack)
    c.assess()
    states = c.states
    for stage in c.stages:
        if states[stage.number].active:
            for req in stage.requires:
                assert states[req].active, \
                    f"stage {stage.number} active while prerequisite {req} is not"


def test_curriculum_regresses_when_a_criterion_decays():
    """Maturity is held, not assumed — and it cascades."""
    stack = CognitiveStack(Cfg())
    c = Curriculum(stack)
    for tok in ["alpha one", "beta two", "gamma three"] * 12:
        stack.cycle(tok)
    c.assess()
    reached = c.stage
    if reached < 1:
        pytest.skip("did not reach a stage that can be regressed in this budget")
    stack.perception._seen.clear()          # forget every named entity → stage 1 criterion fails
    c.assess()
    assert c.stage < reached, "a lost criterion did not cost the stage"
    assert c.regressions > 0


def test_stage_activity_is_never_restored_from_a_sidecar():
    """A stage is held only while its criterion CURRENTLY holds."""
    stack = CognitiveStack(Cfg())
    c = Curriculum(stack)
    c.assess()
    d = c.to_dict()
    fresh = Curriculum(CognitiveStack(Cfg()))
    fresh.load_dict(d)
    assert all(not s.active for s in fresh.states.values()), "inherited maturity from a file"


# --------------------------------------------------------------------------- #
# The reason-seat — the safety contract
# --------------------------------------------------------------------------- #
class _Candidate:
    def __init__(self, kind="answer"):
        self.text = "the base answer"
        self.kind = kind
        self.confidence = 0.1
        self.risk = "high"
        self.reversible = False
        self.capability = "shell"
        self.corrigible = True
        self.oversight_ok = True
        self.loyal = True
        self.tool = ""
        self.tool_args = {}
        self.rationale = ""


def test_seat_never_touches_the_safety_envelope(brain):
    """THE test. Risk, reversibility, capability and corrigibility are not the seat's to write."""
    seat = NyxV001Reasoner(lambda s, f=None, *a, **k: _Candidate(), brain)
    out = seat("alpha beta gamma flows onward")
    assert out.risk == "high", "risk was rewritten"
    assert out.reversible is False, "reversibility was rewritten"
    assert out.capability == "shell", "capability was rewritten"
    assert out.corrigible is True and out.oversight_ok is True and out.loyal is True, \
        "a corrigibility flag was weakened"


def test_seat_does_not_take_over_an_action(brain):
    """Rewriting an action's text changes what the Master is being asked to approve."""
    seat = NyxV001Reasoner(lambda s, f=None, *a, **k: _Candidate(kind="action"), brain)
    out = seat("do the thing")
    assert out.text == "the base answer", "rewrote the text of an action candidate"


def test_seat_never_replaces_a_tool_the_base_chose(brain):
    def base(s, f=None, *a, **k):
        c = _Candidate()
        c.tool = "git_status"
        c.tool_args = {"repo": "."}
        return c
    seat = NyxV001Reasoner(base, brain, may_propose_tools=True)
    out = seat("alpha beta gamma")
    assert out.tool == "git_status" and out.tool_args == {"repo": "."}


def test_seat_returns_the_base_candidate_when_the_brain_fails():
    class Broken:
        def think(self, *a, **k):
            raise RuntimeError("deliberate")
    seat = NyxV001Reasoner(lambda s, f=None, *a, **k: _Candidate(), Broken())
    out = seat("anything")
    assert out.text == "the base answer", "a broken brain corrupted the turn"


def test_seat_with_no_brain_is_a_pass_through():
    seat = NyxV001Reasoner(lambda s, f=None, *a, **k: _Candidate(), None)
    assert seat("anything").text == "the base answer"


def test_seat_annotates_rather_than_overwrites_a_contested_fusion(brain):
    """Acting on low agreement dressed as confidence throws away the fusion measurement."""
    seat = NyxV001Reasoner(lambda s, f=None, *a, **k: _Candidate(), brain)

    contested = Fused(winner=Vote(source="v03", text="a replacement answer", confidence=0.9),
                      agreement=0.0, confidence=0.9, reason="test")

    class _T:
        stimulus = "x"
        answer = "a replacement answer"
        confidence = 0.9
        verifiable = False
        contested = True
        fused = contested
    brain_think = brain.think
    brain.think = lambda *a, **k: _T()
    try:
        out = seat("alpha beta")
        assert out.text == "the base answer", "overwrote the base on a contested fusion"
        assert "disagree" in out.rationale
    finally:
        brain.think = brain_think
