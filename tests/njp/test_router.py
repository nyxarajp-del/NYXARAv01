"""Who answers, and what a disagreement is worth — njp/router.py.

Three properties carry this module, and each of them is a thing that would be easy to get subtly
wrong in a way nothing else would catch:

* the cortex does **not** get a pathway the cognitive policy withholds from every other faculty —
  otherwise it becomes the one route by which world knowledge reaches a greeting, which is the hole
  ``relevance.py`` was written to close, rebuilt behind a bigger model;
* the router does **not** classify problems itself — ``metareason.ProblemClassifier`` already does,
  with a scoring rule and a bandit that learns from outcomes, and a second keyword-based opinion
  would win arguments it should lose;
* a disagreement becomes a **question**, not a winner. Picking a side discards the only new thing
  that happened.
"""
from __future__ import annotations

from nyxara.njp.grounding import Answer, Epistemic, GroundedTriple, Provenance
from nyxara.njp.router import (DisagreementKind, Router, Seat, Verdict,
                               _CORTEX_CAPABILITY)
from nyxara.njp.selfmodel import MetaLearner, SelfModel


class _Claim:
    """A cortex-side answer. Deliberately not an ``Answer`` — the router must not require one."""

    def __init__(self, text, confidence=0.4, triples=(), condition=""):
        self.text = text
        self.confidence = confidence
        self.provenance = Provenance.PROPOSED
        self.triples = list(triples)
        self.condition = condition


def _record(text, confidence=0.9, triples=(), condition=""):
    answer = Answer(text=text, state=Epistemic.KNOWN, confidence=confidence,
                    provenance=Provenance.OBSERVED, triples=list(triples))
    answer.condition = condition
    return answer


def _belief(text, confidence=0.4, triples=(), condition=""):
    answer = Answer(text=text, state=Epistemic.BELIEVED, confidence=confidence,
                    provenance=Provenance.OBSERVED, triples=list(triples))
    answer.condition = condition
    return answer


# --------------------------------------------------------------------------- #
# The cortex is gated like any other reasoning faculty
# --------------------------------------------------------------------------- #
def test_a_greeting_never_reaches_the_cortex():
    routing = Router().route("hi NYXARA")
    assert routing.cortex_permitted is False
    assert routing.seats == (Seat.NJP,)
    assert "reasoning faculty" in routing.why


def test_a_causal_question_does_reach_the_cortex():
    routing = Router().route("why did the plant die after I stopped watering it?")
    assert routing.cortex_permitted is True
    assert set(routing.seats) == {Seat.CORTEX, Seat.NJP}


def test_the_cortex_gets_no_pathway_the_policy_withholds():
    """If this ever passes for a greeting, the cortex has become a way round the policy."""
    from nyxara.njp.relevance import Pathway
    router = Router()
    permitted, allowed, _ = router.cortex_permitted("thanks!")
    assert permitted is False
    assert not ({Pathway.REASON, Pathway.SIMULATE, Pathway.EXPERIMENT} & set(allowed))


# --------------------------------------------------------------------------- #
# Classification is delegated, never re-derived
# --------------------------------------------------------------------------- #
class _Classifier:
    def __init__(self):
        self.calls = []

    def classify(self, problem, context=None):
        self.calls.append(problem)
        return type("C", (), {"kind": "causal", "mixed": False, "margin": 0.9})()


class _Meta:
    def __init__(self, classifier):
        self.classifier = classifier


def test_the_router_asks_metareason_rather_than_reading_the_words_itself():
    classifier = _Classifier()
    router = Router(meta=_Meta(classifier))
    routing = router.route("why did the deploy fail?")
    assert classifier.calls == ["why did the deploy fail?"]
    assert routing.kind == "causal"


def test_classification_degrades_to_a_default_rather_than_a_guess():
    class _Explodes:
        classifier = None

        def classify(self, problem, context=None):
            raise RuntimeError("no")

    routing = Router(meta=_Explodes()).route("why did the deploy fail?")
    assert routing.kind, "a kind is always reported"
    assert routing.basis == "default"


# --------------------------------------------------------------------------- #
# Seat order is measured, not typed in
# --------------------------------------------------------------------------- #
def test_with_nothing_measured_the_seat_that_owns_the_record_goes_first():
    routing = Router().route("why did the plant die?")
    assert routing.seats[0] == Seat.NJP
    assert routing.basis == "default"


def test_seat_order_follows_the_posteriors_and_is_not_a_constant():
    """The point of a measured order is that it can come out either way."""
    model = SelfModel()
    for _ in range(40):
        model.observe(_CORTEX_CAPABILITY, 1.0)
        model.observe("reasoning", 0.0)
    hot = Router(self_model=model).route("why did the plant die?")
    assert hot.seats[0] == Seat.CORTEX
    assert hot.basis == "measured"

    cold = SelfModel()
    for _ in range(40):
        cold.observe(_CORTEX_CAPABILITY, 0.0)
        cold.observe("reasoning", 1.0)
    assert Router(self_model=cold).route("why did the plant die?").seats[0] == Seat.NJP


def test_the_bandit_gets_first_say_when_one_is_attached():
    router = Router(learner=MetaLearner())
    routing = router.route("why did the plant die?")
    assert routing.basis == "bandit"
    assert set(routing.seats) == {Seat.CORTEX, Seat.NJP}


def test_observing_an_outcome_moves_the_posterior():
    model, learner = SelfModel(), MetaLearner()
    router = Router(self_model=model, learner=learner)
    routing = router.route("why did the plant die?")
    before = model.trust(_CORTEX_CAPABILITY, 1.0)
    for _ in range(30):
        router.observe(routing, Seat.CORTEX, success=0.0)
    assert model.trust(_CORTEX_CAPABILITY, 1.0) < before


# --------------------------------------------------------------------------- #
# Arbitration
# --------------------------------------------------------------------------- #
def test_a_hypothesis_that_clashes_with_the_record_simply_loses():
    out = Router().arbitrate("why did it die?",
                             cortex=_Claim("a fungus"), njp=_record("lack of water"))
    assert out.verdict == Verdict.CONTRADICTION
    assert out.winner == Seat.NJP and out.text == "lack of water"
    assert out.question == "", "a hypothesis meeting evidence is settled, not an open question"


def test_two_ungrounded_accounts_become_a_question_not_a_winner():
    out = Router().arbitrate("why did it die?",
                             cortex=_Claim("a fungus"), njp=_belief("lack of water"))
    assert out.verdict == Verdict.DISAGREEMENT
    assert out.winner == "" and not out.settled
    assert "a fungus" in out.question and "lack of water" in out.question
    assert set(out.rivals) == {"a fungus", "lack of water"}


def test_a_clash_lowers_confidence_rather_than_raising_it():
    """Two accounts that cannot both be right are grounds for less certainty, never more."""
    out = Router().arbitrate("why?", cortex=_Claim("a fungus", confidence=0.45),
                             njp=_belief("lack of water", confidence=0.45))
    assert out.confidence < 0.45


def test_the_reason_for_a_disagreement_is_read_off_the_accounts():
    shared = GroundedTriple(subject="plant", predicate="was", object="dry")
    mine = GroundedTriple(subject="soil", predicate="had", object="spores")

    # different facts entirely -> they are not looking at the same evidence
    evidence = Router().arbitrate("why?", cortex=_Claim("a fungus", triples=[mine]),
                                  njp=_belief("lack of water", triples=[shared]))
    assert evidence.disagreement == DisagreementKind.EVIDENCE

    # same fact, different conclusion -> the arrows between them differ
    causal = Router().arbitrate("why?", cortex=_Claim("a fungus", triples=[shared, mine]),
                                njp=_belief("lack of water", triples=[shared]))
    assert causal.disagreement == DisagreementKind.CAUSAL_MODEL

    # one holds a condition the other does not
    assumption = Router().arbitrate("why?", cortex=_Claim("a fungus", condition="if it rained"),
                                    njp=_belief("lack of water"))
    assert assumption.disagreement == DisagreementKind.ASSUMPTIONS


def test_an_unseparable_clash_says_so_rather_than_inventing_a_reason():
    out = Router().arbitrate("why?", cortex=_Claim("a fungus"), njp=_belief("lack of water"))
    assert out.disagreement == DisagreementKind.UNCLASSIFIED


def test_agreement_does_not_launder_provenance():
    out = Router().arbitrate("why?", cortex=_Claim("lack of water"),
                             njp=_belief("lack of water"))
    assert out.verdict == Verdict.AGREE
    assert out.provenance == Provenance.PROPOSED, "a popular guess is still a guess"


def test_an_unopposed_cortex_answer_is_still_a_proposal():
    out = Router().arbitrate("why?", cortex=_Claim("a fungus"), njp=None)
    assert out.verdict == Verdict.CORTEX_ONLY
    assert out.provenance == Provenance.PROPOSED
    assert "unopposed" in out.why


def test_two_silent_seats_are_reported_as_silence():
    out = Router().arbitrate("why?", cortex=None, njp=None)
    assert out.verdict == Verdict.SILENT and not out.settled


def test_stats_count_what_actually_happened():
    router = Router()
    router.arbitrate("q", cortex=_Claim("a"), njp=_belief("b"))
    router.arbitrate("q", cortex=_Claim("a"), njp=_record("b"))
    stats = router.stats()
    assert stats["arbitrated"] == 2
    assert stats["verdicts"][Verdict.DISAGREEMENT] == 1
    assert stats["verdicts"][Verdict.CONTRADICTION] == 1
    assert stats["open_questions"] == 1


def test_a_known_answer_built_from_proposals_is_not_the_record():
    """Both halves of "is this the record" are required. ``KNOWN`` alone stopped being sufficient
    the moment a proposal could be held confidently, and a hypothesis must lose to evidence — not
    to another hypothesis wearing a high confidence."""
    poisoned = Answer(text="lack of water", state=Epistemic.KNOWN, confidence=0.9,
                      provenance=Provenance.PROPOSED)
    out = Router().arbitrate("why?", cortex=_Claim("a fungus"), njp=poisoned)
    assert out.verdict == Verdict.DISAGREEMENT
    assert out.winner == ""


def test_a_claim_object_that_raises_on_access_never_takes_the_turn_down():
    """One side of this is always whatever the cortex handed back."""
    class _Hostile:
        @property
        def text(self):
            raise RuntimeError("boom")

        @property
        def confidence(self):
            raise RuntimeError("boom")

        @property
        def provenance(self):
            raise RuntimeError("boom")

    out = Router().arbitrate("why?", cortex=_Hostile(), njp=_belief("lack of water"))
    assert out.verdict == Verdict.NJP_ONLY
    assert out.text == "lack of water"
