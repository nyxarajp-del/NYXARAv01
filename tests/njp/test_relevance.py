"""Truth is not relevance — the transcript failures, turned into tests that cannot silently return.

Every case below is one the Master actually hit. A greeting read as an ontological claim, a
question paraphrased back instead of answered, and a verified pendulum law offered in reply to
"how are you" with confidence raised to 1.00 for having reached a conclusion.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.relevance import (
    CognitivePolicy,
    Pathway,
    RelevanceGate,
    SpeechAct,
    SpeechActReader,
    is_meta_commentary,
    is_verified,
    revise_confidence,
)

PENDULUM = ("From my own experiments I discovered the law "
            "period_squared = 39.67*(L / g) + 0.0003079")


# --------------------------------------------------------------------------- #
# What kind of turn is this
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,kind", [
    ("Hii NYXARA", SpeechAct.GREETING),
    ("hello", SpeechAct.GREETING),
    ("namaste", SpeechAct.GREETING),
    ("How are you NYXARA?", SpeechAct.SOCIAL_CHECKIN),
    ("kaisi ho", SpeechAct.SOCIAL_CHECKIN),
    ("Kya kar rahi ho tum", SpeechAct.STATE_QUERY),
    ("can you learn this", SpeechAct.CAPABILITY_QUERY),
    ("why did the deploy fail", SpeechAct.CAUSAL_QUERY),
    ("pani kyun ubalta hai", SpeechAct.CAUSAL_QUERY),
    ("ye code fix karo", SpeechAct.TASK),
    ("thanks", SpeechAct.THANKS),
])
def test_speech_acts_are_told_apart(text, kind):
    assert SpeechActReader().read(text).kind == kind


def test_a_greeting_is_not_a_question_about_the_world():
    """The first transcript failure: 'Hii NYXARA' read as an architectural claim."""
    act = SpeechActReader().read("Hii NYXARA")
    assert act.social
    assert CognitivePolicy().forbids_world_knowledge(act)


# --------------------------------------------------------------------------- #
# Never reason merely because reasoning is available
# --------------------------------------------------------------------------- #
def test_a_greeting_may_not_reach_the_reasoner_at_all():
    policy = CognitivePolicy()
    allowed = policy.allowed(SpeechAct.GREETING)
    assert Pathway.SOCIAL in allowed and Pathway.SELF_STATE in allowed
    for forbidden in (Pathway.REASON, Pathway.RECALL, Pathway.GROUNDING, Pathway.SIMULATE):
        assert forbidden not in allowed, forbidden


def test_a_knowledge_question_still_gets_the_full_apparatus():
    """The gate must not make her stupid — only relevant."""
    allowed = CognitivePolicy().allowed(SpeechAct.KNOWLEDGE_QUERY)
    assert Pathway.GROUNDING in allowed and Pathway.REASON in allowed
    assert not CognitivePolicy().forbids_world_knowledge(
        SpeechActReader().read("gravity kya hai"))


# --------------------------------------------------------------------------- #
# Truth ≠ relevance
# --------------------------------------------------------------------------- #
def test_the_pendulum_law_is_refused_for_how_are_you():
    """The transcript, exactly. A true fact that answers nothing is noise with a certificate."""
    gate = RelevanceGate()
    score = gate.score(PENDULUM, query="How are you NYXARA?",
                       act=SpeechActReader().read("How are you NYXARA?"))
    assert not score.admitted
    assert score.total == 0.0
    assert "nothing in common" in score.why


def test_a_relevant_memory_is_admitted():
    gate = RelevanceGate()
    score = gate.score("gravity is a force", query="gravity kya hai",
                       act=SpeechActReader().read("gravity kya hai"))
    assert score.admitted, score.to_dict()


def test_stopwords_do_not_starve_a_relevant_memory():
    """Measured: 'gravity kya hai' spent two of three slots on function words and scored 0.33."""
    gate = RelevanceGate()
    with_stops = gate.score("gravity is a force", query="gravity kya hai")
    bare = gate.score("gravity is a force", query="gravity")
    assert abs(with_stops.semantic - bare.semantic) < 1e-9


def test_a_causally_linked_memory_survives_low_word_overlap():
    """The dimension that stops this being a keyword filter with extra steps."""
    class _Link:
        cause, effect, strength = "aag", "garmi", 0.9

    class _World:
        def links(self, causal_only=True):
            return [_Link()]

    gate = RelevanceGate()
    score = gate.score("garmi badhne se pani ubalta hai", query="aag kya karti hai",
                       world=_World())
    assert score.causal > 0.5, score.to_dict()


def test_filter_returns_only_admitted_and_records_the_rejects():
    gate = RelevanceGate()
    admitted = gate.filter([PENDULUM, "gravity is a force"], query="gravity kya hai")
    assert [s.memory for s in admitted] == ["gravity is a force"]
    # The rejected candidate is still recorded, so a bad answer is explainable afterwards.
    assert any(not s.admitted for s in gate.last)
    assert gate.rejected >= 1


# --------------------------------------------------------------------------- #
# Confidence may not rise because you thought about it
# --------------------------------------------------------------------------- #
def test_reasoning_harder_never_raises_confidence():
    """The transcript's 0.80 → 1.00. Depth may lower confidence; it may never raise it."""
    assert revise_confidence(0.80, relevance=1.0, reasoning_depth=6) < 0.80


def test_an_irrelevant_answer_cannot_be_confident():
    assert revise_confidence(0.95, independent_evidence=5, relevance=0.0) <= 0.3


def test_independent_evidence_is_the_only_thing_that_raises_it():
    base = revise_confidence(0.60, independent_evidence=0, relevance=0.9)
    more = revise_confidence(0.60, independent_evidence=2, relevance=0.9)
    assert base == pytest.approx(0.60)
    assert more > base


def test_confidence_never_reaches_certainty():
    assert revise_confidence(0.96, independent_evidence=20, relevance=1.0) <= 0.97


def test_a_refuted_claim_loses_half_its_confidence():
    assert revise_confidence(0.90, independent_evidence=3, relevance=1.0,
                             consistent=False) < 0.5


def test_verified_means_independent_verification_not_a_finished_pipeline():
    assert not is_verified(pipeline_completed=True)
    assert not is_verified(independent_sources=1)
    assert is_verified(independent_sources=2)
    assert is_verified(independent_sources=1, reproduced=True)


# --------------------------------------------------------------------------- #
# "I understand: <your words back>"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reply,stimulus", [
    ("I'm certain that I understand: Hii NYXARA is a sovereign cognitive architecture",
     "Hii NYXARA"),
    ("I think I understand: Kya kar rahi ho", "Kya kar rahi ho tum"),
    ("I understand: gravity kya hai", "gravity kya hai"),
])
def test_meta_commentary_is_recognised(reply, stimulus):
    assert is_meta_commentary(reply, stimulus)


@pytest.mark.parametrize("reply,stimulus", [
    ("Hii Master 👋", "Hii NYXARA"),
    ("force", "gravity kya hai"),
    ("Main theek hoon Master. 19 cells / 13 synapses.", "How are you NYXARA?"),
])
def test_a_real_answer_is_not_flagged(reply, stimulus):
    assert not is_meta_commentary(reply, stimulus)


# --------------------------------------------------------------------------- #
# End to end, through the brain
# --------------------------------------------------------------------------- #
def _taught_brain() -> NJPBrain:
    brain = NJPBrain()
    for line in (f"{PENDULUM} is the pendulum law",
                 "pani 100 degree par ubalta hai",
                 "gravity ek force hai"):
        brain.think(line)
    return brain


def test_the_three_transcript_failures_are_gone():
    brain = _taught_brain()

    greeting = brain.think("Hii NYXARA")
    assert greeting.act.kind == SpeechAct.GREETING
    assert "sovereign" not in greeting.answer.lower()
    assert not is_meta_commentary(greeting.answer, "Hii NYXARA")

    state = brain.think("Kya kar rahi ho tum")
    assert state.act.kind == SpeechAct.STATE_QUERY
    assert state.answer and not is_meta_commentary(state.answer, "Kya kar rahi ho tum")

    social = brain.think("How are you NYXARA?")
    assert social.act.kind == SpeechAct.SOCIAL_CHECKIN
    assert "period_squared" not in social.answer
    assert "39.67" not in social.answer


def test_a_knowledge_question_is_still_answered_from_structure():
    brain = _taught_brain()
    answer = brain.think("gravity kya hai").answer
    assert "force" in answer.lower(), answer


# --------------------------------------------------------------------------- #
# The gate can only judge what actually reaches it
# --------------------------------------------------------------------------- #
def test_recalled_memories_actually_reach_the_relevance_gate():
    """The gate scored nothing, ever, because the recall was read off fields that never existed.

    ``NJPBrain._recall_through_gate`` read ``rec.traces`` and ``rec.best``. ``memory.Recall``
    carries ``hit``, ``score``, ``decided``, ``associated`` and ``considered`` — neither of the
    two it was asked for. Both reads were ``getattr(..., None)``, so nothing raised: the
    candidate list was simply always empty and the method returned before ``gate.filter`` on
    every turn of every kind. The holographic memory that ``study.py`` writes to twice per corpus
    pair contributed to an answer exactly never.

    Asserting on ``gate.stats()["scored"]`` rather than on an answer is deliberate — whether a
    given memory is *admitted* is the gate's judgement and may legitimately be "no". What was
    broken is that it never got to judge.
    """
    brain = _taught_brain()
    assert brain.gate is not None, "no gate installed; this test cannot say anything"
    before = brain.gate.stats()["scored"]
    brain.think("the apple fell from the tree")
    assert brain.gate.stats()["scored"] > before, (
        "recall never reached the gate — _recall_through_gate returned before filtering")


def test_a_question_she_cannot_ground_is_answered_with_silence_not_the_nearest_memory():
    """Measured, not assumed: recall-as-answer buys coverage by spending honesty.

    Routing unanswered questions through recall-and-the-gate was tried. In-sample it looked
    excellent — 200 taught questions went from 8 answered to 75, and from 4 correct to 59. On
    held-out questions the same change produced three answers where there had been three
    abstentions, and **none** of the three was right:

        "Give me an example of irony?"       → a memory about the word "charge"
        "Which tree has aromatic blossoms?"  → a memory about the Resplendent Quetzal

    Nothing separated those from the good ones. The worst held-out answer scored 0.394 at the
    gate, above the 0.388 median of the *correct* in-sample ones, and `Recall.decided` was True
    in every case in both populations. So there is no threshold to tune here: relevance is not
    correctness.
    """
    brain = _taught_brain()
    assert not brain.think("Give me an example of irony?").answer
    assert not brain.think("Which tree is famous for its aromatic blossoms?").answer


def test_the_recall_dataclass_has_the_fields_the_brain_reads():
    """Structural guard, so a rename cannot silently disconnect the memory again."""
    from nyxara.njp.memory import Recall

    rec = Recall()
    for field in ("hit", "score", "decided", "associated", "considered"):
        assert hasattr(rec, field), f"Recall lost {field}, which brain._recall_through_gate reads"
