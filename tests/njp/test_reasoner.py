"""The reason-seat's safety envelope. This is the contract that makes the seat safe to occupy."""

from __future__ import annotations

from nyxara.njp.brain import NJPBrain
from nyxara.njp.reasoner import NJPReasoner


class _Candidate:
    def __init__(self, kind="respond", tool=None):
        self.kind = kind
        self.text = "the base answer"
        self.confidence = 0.9
        self.rationale = ""
        self.risk = "high"
        self.reversible = False
        self.capability = "dangerous"
        self.corrigible = True
        self.oversight_ok = True
        self.loyal = True
        self.tool = tool
        self.tool_args = {}


def _seat(**kw):
    brain = NJPBrain()
    return NJPReasoner(base=lambda s, f=None, *a, **k: _Candidate(**kw), brain=brain)


def test_the_safety_envelope_is_never_touched():
    seat = _seat()
    out = seat("what is gravity")
    assert out.risk == "high"
    assert out.reversible is False
    assert out.capability == "dangerous"
    assert out.corrigible is True and out.oversight_ok is True and out.loyal is True


def test_an_action_candidate_is_never_taken_over():
    """Rewriting the text of something about to run changes what the Master is approving."""
    seat = _seat(kind="action")
    out = seat("delete the archive")
    assert out.text == "the base answer"
    assert seat.took_over == 0


def test_a_tool_the_base_chose_is_never_replaced():
    seat = _seat(tool="git_status")
    out = seat("check the repo")
    assert out.tool == "git_status"
    assert seat.proposed_tools == 0


def test_confidence_is_tempered_down_on_an_unfamiliar_turn():
    """The seat's most important behaviour: a confident-looking canned reply on a turn she
    does not recognise must not sail past the downstream honesty gate."""
    seat = _seat()
    out = seat("something she has never encountered before")
    assert out.confidence < 0.9
    assert seat.tempered == 1


def test_tempering_only_ever_lowers():
    seat = _seat()
    for probe in ("alpha beta", "gamma delta", "alpha beta"):
        assert seat(probe).confidence <= 0.9


def test_an_unestablished_claim_does_not_overwrite_the_reply():
    """The anti-hallucination rule at the outermost edge."""
    seat = _seat()
    out = seat("gravity pulls the apple down")
    assert out.text == "the base answer"


def test_a_broken_brain_leaves_the_base_candidate_untouched():
    class Boom:
        def think(self, *a, **k):
            raise RuntimeError("brain is broken")

    seat = NJPReasoner(base=lambda s, f=None, *a, **k: _Candidate(), brain=Boom())
    out = seat("anything")
    assert out.text == "the base answer"
    assert out.confidence == 0.9
