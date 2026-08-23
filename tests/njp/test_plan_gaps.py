"""Two things the plan asked for that the stages shipping it did not deliver.

**The speech act was allowed to gate the cortex and not the reasoning.** `CognitivePolicy` has
held the right table since it was written — a greeting is permitted the relationship and her own
state and nothing else — and `Router` read it only to decide whether the LLM cortex could speak.
Strategy selection never asked, so the only thing keeping `simulate` away from a greeting was the
problem *kind* coming out differently, which is a coincidence rather than a rule.

**Two error taxonomies ran side by side and never spoke.** `predict.ErrorKind` answers *which
organ*; `field.ErrorClass` answers *how deep* — a wrong coefficient is refitted, a missing arrow
is declared, a concept system that cannot express what she met is restructured. Those are
orthogonal, and reporting only the first meant a repair chose its own depth with no evidence
about which was called for.
"""

from __future__ import annotations

from nyxara.njp.field import ErrorClass
from nyxara.njp.metareason import MetaReasoner, ProblemKind
from nyxara.njp.predict import ErrorKind, PredictionEngine
from nyxara.njp.relevance import CognitivePolicy, Pathway, SpeechAct


def _reasoner() -> MetaReasoner:
    reasoner = MetaReasoner()
    reasoner.register("simulate", (ProblemKind.CAUSAL, ProblemKind.EMPIRICAL), lambda p, c: "sim")
    reasoner.register("recall", (ProblemKind.FACTUAL,), lambda p, c: "rec")
    reasoner.register("introspect", (ProblemKind.INTROSPECTIVE,), lambda p, c: "me")
    reasoner.register("derive", (ProblemKind.FACTUAL, ProblemKind.CAUSAL), lambda p, c: "der")
    return reasoner


# --------------------------------------------------------------------------- #
# The policy reaches strategy selection
# --------------------------------------------------------------------------- #

def test_a_greeting_cannot_reach_the_simulator_by_policy_not_by_luck():
    reasoner = _reasoner()
    allowed = CognitivePolicy().PATHWAYS[SpeechAct.GREETING]
    assert Pathway.SIMULATE not in allowed

    for _ in range(12):
        chosen = reasoner.choose(ProblemKind.CAUSAL, pathways=allowed)
        # Either nothing is permitted, or something that is. Never the simulator.
        assert chosen is None or chosen.name != "simulate", "physics is reachable from a hello"

    # And the same act does still reach the strategies it permits.
    introspective = reasoner.choose(ProblemKind.INTROSPECTIVE, pathways=allowed)
    assert introspective is not None and introspective.name == "introspect"


def test_a_causal_query_does_permit_the_simulator():
    reasoner = _reasoner()
    allowed = CognitivePolicy().PATHWAYS[SpeechAct.CAUSAL_QUERY]
    names = {reasoner.choose(ProblemKind.CAUSAL, pathways=allowed).name for _ in range(12)}
    assert "simulate" in names or "derive" in names


def test_an_unread_act_constrains_nothing():
    """No pathways means no act was read, and an unread act is not a restriction."""
    reasoner = _reasoner()
    assert reasoner.choose(ProblemKind.CAUSAL, pathways=()) is not None


def test_a_strategy_nobody_mapped_is_not_silently_unreachable():
    """Gating by an incomplete table would strand every newly registered arm."""
    reasoner = _reasoner()
    reasoner.register("shape:is_a>requires", (ProblemKind.FACTUAL,), lambda p, c: "x")
    allowed = CognitivePolicy().PATHWAYS[SpeechAct.GREETING]
    names = {reasoner.choose(ProblemKind.FACTUAL, pathways=allowed).name for _ in range(20)}
    assert "shape:is_a>requires" in names


def test_a_turn_whose_act_permits_no_reasoning_gets_none():
    """The first version of this softened an empty result back to the full pool.

    That defeated the gate in exactly the case it exists for: a greeting permits neither
    `simulate` nor `derive`, every candidate is excluded, and "fall back to all of them" hands
    the turn straight to the simulator.
    """
    reasoner = MetaReasoner()
    reasoner.register("simulate", (ProblemKind.CAUSAL,), lambda p, c: "sim")
    assert reasoner.choose(ProblemKind.CAUSAL, pathways=(Pathway.SOCIAL,)) is None

    solution = reasoner.solve("hello there", context={"pathways": (Pathway.SOCIAL,)})
    assert not solution.answered
    assert any("permits no reasoning" in d for d in solution.critique.defects)


# --------------------------------------------------------------------------- #
# Which organ × how deep
# --------------------------------------------------------------------------- #

def test_a_diagnosis_now_carries_how_deep_a_repair_it_calls_for():
    engine = PredictionEngine(None)
    engine.predict("k", "water", organ="core")
    out = engine.observe("k", "food", evidence={"stimulus": "x", "concepts": ["x"],
                                                "expected_triples": True, "triples": []})
    assert out.diagnosis.kind == ErrorKind.GROUNDING
    assert out.diagnosis.depth == ErrorClass.CONCEPTUAL
    assert out.diagnosis.route == (ErrorKind.GROUNDING, ErrorClass.CONCEPTUAL)


def test_a_wrong_value_on_a_right_relation_is_a_coefficient_not_a_concept():
    engine = PredictionEngine(None)
    engine.predict("k", "water", organ="core")
    out = engine.observe("k", "food", evidence={"stimulus": "x", "concepts": ["x"],
                                                "expected_triples": True, "triples": [1],
                                                "derived": True})
    assert out.diagnosis.depth == ErrorClass.NUMERIC


def test_a_depth_the_evidence_cannot_settle_is_unknown_rather_than_guessed():
    """Guessing a depth picks a repair size at random."""
    engine = PredictionEngine(None)
    engine.predict("k", "water", organ="planning")
    out = engine.observe("k", "food", evidence={"stimulus": "x", "concepts": ["x"]})
    assert out.diagnosis.depth == ErrorClass.UNKNOWN


def test_the_caller_s_own_critic_wins_over_the_inference():
    engine = PredictionEngine(None)
    engine.predict("k", "water", organ="core")
    out = engine.observe("k", "food", evidence={"stimulus": "x", "concepts": ["x"],
                                                "expected_triples": True, "triples": [],
                                                "depth": ErrorClass.STRUCTURAL})
    assert out.diagnosis.depth == ErrorClass.STRUCTURAL


def test_the_cross_product_is_counted_and_reported():
    """Two misses on one organ at two depths are two different events."""
    engine = PredictionEngine(None)
    for i, evidence in enumerate((
            {"expected_triples": True, "triples": []},
            {"expected_triples": True, "triples": [1], "derived": True})):
        engine.predict(f"k{i}", "water", organ="core")
        engine.observe(f"k{i}", "food",
                       evidence={"stimulus": "x", "concepts": ["x"], **evidence})
    routes = engine.stats()["by_route"]
    assert len(routes) == 2, routes
