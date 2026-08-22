"""The fabric reaches the answer path — brain.`_temper_by_novelty` and the third seat.

**The finding this file pins.** An audit ran ``"fabric" in inspect.getsource(brain._compose)`` and
got ``False``. The substrate grew on every single turn — cells, synapses, a manifold learning its
own transitions — and not one of those numbers could affect what she said or how firmly she said
it. The one consumer of :attr:`~nyxara.njp.brain.NJPPercept.novelty` was a reasoner seat that a
grounded turn never reaches. Growth was real and inert.

Two edges close that, and they read **different properties of the same organ** on purpose, so that
one unfamiliar turn is not charged for twice:

* :meth:`~nyxara.njp.brain.NJPBrain._temper_by_novelty` reads *familiarity* — a graded similarity —
  and discounts a grounded answer's confidence by it on every turn.
* :meth:`~nyxara.njp.router.Router._review_by_fabric` reads *trusted* — the manifold's own
  statement that it had enough evidence to predict at all — and dissents from a confident answer
  where there was none.

Both directions are one-way. The fabric may make her less sure of something she looked up. Nothing
here lets it make her more sure of anything, which is what keeps "she grew" from becoming its own
evidence.
"""
from __future__ import annotations

from nyxara.njp.brain import NJPBrain
from nyxara.njp.manifold import Prediction
from nyxara.njp.router import DisagreementKind, Router, Seat, Verdict


class _Claim:
    """The shape arbitration reads off a seat: some text, a confidence, its supports."""

    def __init__(self, text: str, confidence: float, triples=()) -> None:
        self.text = text
        self.confidence = confidence
        self.triples = list(triples)


def _untrusted() -> Prediction:
    return Prediction(reason="too few transitions")


def _trusted() -> Prediction:
    out = Prediction()
    out.trusted = True
    return out


# --------------------------------------------------------------------------- #
# The confidence edge
# --------------------------------------------------------------------------- #
def test_an_unfamiliar_turn_is_held_less_firmly_than_a_familiar_one():
    """The same fact, the same answer — separated only by whether the fabric recognised it."""
    cold = NJPBrain()
    cold.think("mera naam Jay hai")
    first = cold.think("mera naam kya hai?")

    warm = NJPBrain()
    warm.think("mera naam Jay hai")
    for _ in range(25):
        warm.think("mera naam kya hai?")
    later = warm.think("mera naam kya hai?")

    assert first.answer == later.answer == "Jay", "the answer must be what is held constant"
    assert first.epistemic_confidence < later.epistemic_confidence


def test_the_discount_only_ever_lowers():
    """A familiar turn is left exactly at what the store said, never lifted above it."""
    brain = NJPBrain()
    brain.think("mera naam Jay hai")
    for _ in range(25):
        brain.think("mera naam kya hai?")
    thought = brain.think("mera naam kya hai?")
    assert thought.percept.novelty == 0.0
    assert thought.epistemic_confidence <= 0.9


def test_novelty_damping_is_pure_and_cannot_raise():
    thought = NJPBrain().think("mera naam Jay hai")
    for base in (0.0, 0.3, 0.9, 1.0):
        assert NJPBrain._temper_by_novelty(base, thought) <= base


def test_a_thought_with_no_percept_discounts_nothing():
    class _Bare:
        percept = None
    assert NJPBrain._temper_by_novelty(0.8, _Bare()) == 0.8


# --------------------------------------------------------------------------- #
# The report the audit misread
# --------------------------------------------------------------------------- #
def test_the_gauntlet_verdict_and_the_epistemic_state_both_travel():
    """`confidence: 0.0` beside `believed @ 0.9` is two answers to two questions, not a bug.

    The gauntlet abstains on a claim with no hard evidence source, which "Jay" will never have.
    Reporting only that number read as a broken confidence pipeline and cost a round trip to
    disprove, so the state and its own confidence are named alongside it.
    """
    brain = NJPBrain()
    brain.think("mera naam Jay hai")
    row = brain.think("mera naam kya hai?").to_dict()
    assert row["answer"] == "Jay"
    assert row["confidence"] == 0.0, "the gauntlet abstained, and that is correct"
    assert row["epistemic"] == "believed"
    assert row["epistemic_confidence"] > 0.0
    assert row["novelty"] is not None


# --------------------------------------------------------------------------- #
# The third seat
# --------------------------------------------------------------------------- #
def test_the_fabric_is_a_seat_but_not_a_speaking_one():
    """It has no way to turn cell ids into "water", and a seat that pretended otherwise would
    be inventing content and attributing it to the substrate."""
    assert Seat.FABRIC in Seat.ALL
    assert Seat.FABRIC not in Seat.SPEAKING


def test_a_confident_answer_on_unrecognised_ground_is_held_less_firmly():
    router = Router()
    njp = _Claim("water", 0.9)
    base = router.arbitrate("what does a sparrow need?", njp=njp)
    dissent = router.arbitrate("what does a sparrow need?", njp=njp, fabric=_untrusted())
    assert dissent.confidence < base.confidence
    assert dissent.disagreement == DisagreementKind.RECOGNITION


def test_recognition_does_not_unseat_the_record():
    """The record is evidence and recognition is not, so the answer still stands — less firmly."""
    router = Router()
    out = router.arbitrate("what does a sparrow need?", njp=_Claim("water", 0.9),
                           fabric=_untrusted())
    assert out.verdict == Verdict.NJP_ONLY
    assert out.text == "water"
    assert out.settled


def test_a_recognised_turn_is_left_exactly_as_it_was():
    router = Router()
    njp = _Claim("water", 0.9)
    base = router.arbitrate("q", njp=njp)
    corroborated = router.arbitrate("q", njp=njp, fabric=_trusted())
    assert corroborated.confidence == base.confidence
    assert not corroborated.disagreement


def test_a_hedged_answer_is_not_punished_for_being_honest():
    """Below the confident floor the reply is already behaving correctly."""
    router = Router()
    njp = _Claim("maybe water", 0.2)
    base = router.arbitrate("q", njp=njp)
    reviewed = router.arbitrate("q", njp=njp, fabric=_untrusted())
    assert reviewed.confidence == base.confidence


def test_omitting_the_fabric_is_exactly_the_function_that_existed_before():
    router = Router()
    njp, cortex = _Claim("water", 0.9), _Claim("water", 0.7)
    out = router.arbitrate("q", cortex=cortex, njp=njp)
    assert out.verdict == Verdict.AGREE
    assert not out.disagreement


def test_corroborations_and_dissents_are_both_counted():
    """The ratio is what says whether the substrate is coming to recognise her ground."""
    router = Router()
    njp = _Claim("water", 0.9)
    router.arbitrate("q", njp=njp, fabric=_trusted())
    router.arbitrate("q", njp=njp, fabric=_untrusted())
    stats = router.stats()
    assert stats["fabric_corroborations"] == 1
    assert stats["fabric_dissents"] == 1


def test_an_unreadable_prediction_abstains_rather_than_dissenting():
    """Absent is not untrusted. A malformed argument must not manufacture a dissent."""
    router = Router()
    njp = _Claim("water", 0.9)
    base = router.arbitrate("q", njp=njp)
    out = router.arbitrate("q", njp=njp, fabric=object())
    assert out.confidence == base.confidence
    assert not out.disagreement
    assert router.stats()["fabric_dissents"] == 0
