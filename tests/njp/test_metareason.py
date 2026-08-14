"""Choosing how to think: classification, strategy selection, the critic, and disagreement."""

from __future__ import annotations

from nyxara.njp.metareason import MetaReasoner, ProblemClassifier, ProblemKind


def _reasoner() -> MetaReasoner:
    meta = MetaReasoner()
    meta.register("arith", (ProblemKind.SYMBOLIC,), lambda p, c: "396")
    meta.register("causal", (ProblemKind.CAUSAL,), lambda p, c: "because the fire heated it")
    meta.register("probe", (ProblemKind.EMPIRICAL,), lambda p, c: "measure it")
    meta.register("mirror", (ProblemKind.INTROSPECTIVE,), lambda p, c: "I am still learning")
    return meta


def test_kinds_are_told_apart():
    classify = ProblemClassifier().classify
    assert classify("17 * 23 + 5").kind == ProblemKind.SYMBOLIC
    assert classify("pani kyun ubalta hai").kind == ProblemKind.CAUSAL
    assert classify("kitna temperature hai", context={"grounded": False}).kind == \
        ProblemKind.EMPIRICAL
    assert classify("tum kya kar sakti ho", context={"about_self": True}).kind == \
        ProblemKind.INTROSPECTIVE
    assert classify("anything", context={"contradicts": True}).kind == ProblemKind.CONTRADICTION


def test_each_kind_reaches_the_strategy_built_for_it():
    meta = _reasoner()
    assert meta.solve("17 * 23 + 5").strategy == "arith"
    assert meta.solve("pani kyun ubalta hai").strategy == "causal"
    assert meta.solve("tum kya kar sakti ho", context={"about_self": True}).strategy == "mirror"


def test_the_critic_catches_an_answer_that_restates_the_question():
    meta = MetaReasoner()
    meta.register("echo", (), lambda p, c: p)
    solution = meta.solve("why did it fail")
    assert "answer restates the question" in solution.critique.defects
    assert not solution.assertable


def test_a_symbolic_problem_answered_without_a_value_is_a_defect():
    meta = MetaReasoner()
    meta.register("waffle", (ProblemKind.SYMBOLIC,), lambda p, c: "it depends")
    solution = meta.solve("17 * 23 + 5")
    assert "symbolic problem answered without a value" in solution.critique.defects


def test_disagreement_lowers_confidence_rather_than_being_averaged_away():
    meta = MetaReasoner()
    # Both eligible for the same kind, and they disagree. The classification is engineered to be
    # mixed so the second opinion is actually spent.
    meta.register("a", (), lambda p, c: "42")
    meta.register("b", (), lambda p, c: "totally different")
    solution = meta.solve("what is it", context={"grounded": False})
    if solution.alternative:
        assert solution.agreed is False
        assert solution.confidence < 0.4, solution.confidence
        assert not solution.assertable


def test_agreement_raises_confidence():
    meta = MetaReasoner()
    meta.register("a", (), lambda p, c: "42")
    meta.register("b", (), lambda p, c: "42")
    solo = MetaReasoner()
    solo.register("a", (), lambda p, c: "42")
    both = meta.solve("what is it", context={"grounded": False})
    if both.agreed is True:
        assert both.confidence > solo.solve("what is it", context={"grounded": False}).confidence


def test_a_strategy_that_produces_nothing_is_not_credited():
    meta = MetaReasoner()
    meta.register("silent", (), lambda p, c: None)
    solution = meta.solve("anything")
    assert not solution.answered
    assert meta.abstained == 1
    assert meta.strategies["silent"].wins == 0.0


def test_an_answer_the_critic_tore_apart_is_not_scored_as_a_win():
    """Returning something is not success, or the bandit learns to prefer whoever guesses most."""
    meta = MetaReasoner()
    meta.register("echo", (), lambda p, c: p)
    meta.solve("why did it fail")
    assert meta.strategies["echo"].wins < 0.5


def test_reality_overrides_the_critic():
    meta = _reasoner()
    solution = meta.solve("17 * 23 + 5")
    before = meta.strategies["arith"].wins
    meta.outcome(solution, correct=False)
    assert meta.strategies["arith"].wins < before


def test_selection_learns_which_strategy_actually_works():
    meta = MetaReasoner()
    meta.register("good", (ProblemKind.SYMBOLIC,), lambda p, c: "396", prior=0.5)
    meta.register("bad", (ProblemKind.SYMBOLIC,), lambda p, c: None, prior=0.5)
    for _ in range(30):
        meta.solve("17 * 23 + 5")
    assert meta.strategies["good"].rate > meta.strategies["bad"].rate
    assert meta.best_for(ProblemKind.SYMBOLIC).name == "good"


def test_state_survives_a_round_trip():
    meta = _reasoner()
    meta.solve("17 * 23 + 5")
    revived = _reasoner()
    revived.load_dict(meta.to_dict())
    assert revived.solved == meta.solved
    assert revived.strategies["arith"].trials == meta.strategies["arith"].trials
