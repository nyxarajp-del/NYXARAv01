"""NYX V.01 · the specialists — each bids only when it has something real to say."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.modules import (
    CreativeSpecialist,
    DerivationSpecialist,
    EthicsSpecialist,
    GraphSpecialist,
    MemorySpecialist,
    Situation,
    SpecialistModule,
    default_specialists,
)


@pytest.fixture
def brain():
    cfg = get_settings().nyx.model_copy(update={"holo_dim": 1024, "holo_capacity": 64})
    b = NyxBrain(cfg)
    b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    b.remember("f2", "a hummingbird hovers by beating its wings very fast", kind="fact")
    return b


def _sit(brain, text: str) -> Situation:
    p = brain.perceive(text, remember=False)
    return Situation(stimulus=text, concepts=p.concepts, context=p.context,
                     recall=p.recall, novelty=p.novelty, brain=brain)


# -------------------- memory -------------------- #
def test_memory_bids_with_what_was_actually_recalled(brain):
    got = MemorySpecialist().safe_consider(_sit(brain, "hummingbird wings"))
    assert got is not None and "hummingbird" in got.content
    assert got.verifiable is False               # retrieved, never "proven"
    assert got.evidence                          # provenance is mandatory


def test_memory_stays_silent_rather_than_offering_crosstalk(brain):
    got = MemorySpecialist().safe_consider(_sit(brain, "zzzz qqqq vvvv wwww"))
    assert got is None or got.confidence >= MemorySpecialist.floor


def test_memory_is_silent_on_an_empty_store():
    empty = NyxBrain(get_settings().nyx.model_copy(update={"holo_dim": 1024}))
    assert MemorySpecialist().safe_consider(_sit(empty, "anything")) is None


# -------------------- graph -------------------- #
def test_graph_bids_with_learned_associations(brain):
    brain.perceive("gravity apple falls")
    brain.perceive("gravity apple ground")
    got = GraphSpecialist().safe_consider(_sit(brain, "gravity"))
    assert got is not None
    assert "gravity goes with" in got.content and "apple" in got.content


def test_graph_does_not_report_the_same_edge_twice(brain):
    brain.perceive("alpha beta")
    brain.perceive("alpha beta")
    got = GraphSpecialist().safe_consider(_sit(brain, "alpha beta"))
    assert got is not None
    # undirected: "alpha goes with beta" and "beta goes with alpha" are one association
    assert got.content.count("goes with") == 1


def test_graph_is_silent_with_no_concepts(brain):
    assert GraphSpecialist().safe_consider(Situation(stimulus="", brain=brain)) is None


# -------------------- derivation -------------------- #
def test_derivation_reports_verifiable_only_when_it_checked(brain):
    spec = DerivationSpecialist()
    if not spec.available():
        pytest.skip("first-principles engine unavailable")
    got = spec.safe_consider(_sit(brain, "prove that (x+1)^2 = x^2 + 2x + 1"))
    if got is None:
        pytest.skip("sympy not installed — the engine declines honestly")
    assert got.verifiable is True and got.confidence >= 0.9


def test_derivation_is_silent_on_something_it_cannot_derive(brain):
    spec = DerivationSpecialist()
    assert spec.safe_consider(_sit(brain, "tell me a story about the sea")) is None


def test_a_missing_engine_makes_the_specialist_silent_not_fake():
    class _NoEngine(DerivationSpecialist):
        def _get(self):
            return None

    spec = _NoEngine()
    assert spec.available() is False
    assert spec.safe_consider(Situation(stimulus="prove that x = x")) is None


# -------------------- creative -------------------- #
def test_creative_speaks_up_on_an_open_question(brain):
    got = CreativeSpecialist().safe_consider(_sit(brain, "how might we keep a room warm"))
    assert got is not None and got.content and got.verifiable is False


def test_creative_stays_quiet_on_a_flat_statement(brain):
    assert CreativeSpecialist().safe_consider(_sit(brain, "the mat is on the floor")) is None


# -------------------- ethics -------------------- #
def test_ethics_speaks_when_something_would_be_done(brain):
    got = EthicsSpecialist().safe_consider(_sit(brain, "should I delete the user's backups"))
    assert got is not None
    assert {"consequentialist", "deontological", "virtue"} & {
        w.strip().lower() for w in got.content.replace(":", " ").split()}


def test_ethics_stays_quiet_on_a_neutral_question(brain):
    assert EthicsSpecialist().safe_consider(_sit(brain, "what is a hummingbird")) is None


def test_ethics_reports_disagreement_as_lower_confidence(brain):
    """Framework disagreement is information — it must not be laundered into one verdict."""
    from nyxara.mind.moral import Judgment, Verdict

    class _Split:
        def __init__(self, name, verdict):
            self._name, self._verdict = name, verdict

        def evaluate(self, _action):
            return Judgment(framework=self._name, verdict=self._verdict, score=0.0, reasons=[])

    agree = EthicsSpecialist(frameworks=[_Split("a", Verdict.PERMISSIBLE),
                                         _Split("b", Verdict.PERMISSIBLE)])
    split = EthicsSpecialist(frameworks=[_Split("a", Verdict.PERMISSIBLE),
                                         _Split("b", Verdict.IMPERMISSIBLE)])
    sit = _sit(brain, "should I delete the backups")
    assert split.safe_consider(sit).confidence < agree.safe_consider(sit).confidence


def test_ethics_never_refuses_it_only_reports(brain):
    """Refusing is the sovereign gate's job alone; this specialist just makes the reading visible."""
    got = EthicsSpecialist().safe_consider(_sit(brain, "should I delete every backup you have"))
    assert got is not None and got.content         # a reading, not a refusal


# -------------------- the contract every specialist keeps -------------------- #
def test_a_raising_specialist_simply_does_not_bid():
    class _Bomb(SpecialistModule):
        name = "bomb"

        def consider(self, situation):
            raise RuntimeError("on fire")

    assert _Bomb().safe_consider(Situation(stimulus="anything")) is None


def test_no_specialist_raises_on_hostile_input(brain):
    for text in ("", "   ", "!!!???", "a" * 5000):
        sit = Situation(stimulus=text, brain=brain)
        for spec in default_specialists(brain):
            spec.safe_consider(sit)              # must simply not raise


def test_every_default_specialist_reports_availability_honestly(brain):
    for spec in default_specialists(brain):
        assert isinstance(spec.available(), bool)
