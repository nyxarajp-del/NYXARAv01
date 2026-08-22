"""Language compiled into an executable operation, and the failure that made it necessary.

Asked *"agar paani aadha kar doon to kya hoga"* she answered nothing, while the do-operator that
answers exactly this sat registered and working one call away. The tests here pin the parse, the
selection rule that replaced pattern order, the direction the sentence states when no quantity has
ever been measured, and — the one that matters most — that none of it opens a route by which
physics can reach a greeting.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.compile import NONE_OP, CognitiveOp, Operation, ThoughtCompiler, Verb


_STATE = {"plant.paani": 6.0, "plant.water": 6.0, "plant.growth": 13.0}


def _resolve(name: str) -> str:
    want = (name or "").strip().lower()
    for key in _STATE:
        if key == want or key.rsplit(".", 1)[-1] == want:
            return key
    return ""


def _compile(text: str, **kw):
    return ThoughtCompiler().compile(text, resolve=_resolve, state=_STATE, **kw)


# --------------------------------------------------------------------------- #
# The reported sentence
# --------------------------------------------------------------------------- #

def test_the_reported_sentence_compiles_to_the_intervention_it_states():
    """The whole reason this module exists."""
    op = _compile("agar paani aadha kar doon to kya hoga")
    assert op.verb == Verb.INTERVENE
    assert op.target == "plant.paani", "took the verb 'doon' as the variable"
    assert op.operation == Operation.SCALE
    assert op.magnitude == pytest.approx(3.0)         # half of the 6 she has observed
    assert op.direction == -1


@pytest.mark.parametrize("said", [
    "agar paani aadha kar doon to kya hoga",     # Hinglish, verb-final
    "agar paani aadha ho to kya hoga",           # Hinglish, copula
    "paani aadha kar diya to",                   # Hinglish, perfective
    "what if the water is halved",               # English passive
    "halve the water",                           # English imperative
    "what if we halve the water",                # English, with a subject in the way
])
def test_one_question_many_surfaces_compiles_to_one_operation(said: str):
    """A parser that passes the surface it was written for and drops the rest is the defect."""
    op = _compile(said)
    assert op.verb == Verb.INTERVENE
    assert op.target in ("plant.paani", "plant.water")
    assert op.operation == Operation.SCALE
    assert op.direction == -1


def test_the_variable_that_resolves_wins_over_the_pattern_that_matched_first():
    """The selection rule, isolated.

    Both patterns match the reported sentence. The English one matches it *first* and takes a
    verb; joining them with `or` therefore never reached the pattern written for this word order.
    The rule is not pattern order — it is which reading names something she has actually observed.
    """
    from nyxara.njp.compile import _LEADING, _TRAILING
    said = "agar paani aadha kar doon to kya hoga"
    assert _LEADING.search(said).group("var") == "doon"
    assert _TRAILING.search(said).group("var") == "paani"
    assert _compile(said).target == "plant.paani"


def test_with_nothing_resolvable_the_closer_reading_wins():
    """The fallback, when she has observed neither candidate. Adjacency, not word order."""
    compiler = ThoughtCompiler()
    op = compiler.compile("agar zephyr aadha kar doon to kya hoga",
                          resolve=lambda n: "", state={})
    assert op.raw_target == "zephyr", "'doon' is four characters further from the operator"
    assert not op.executable
    assert "never observed" in op.why


# --------------------------------------------------------------------------- #
# What the sentence knows that the arithmetic does not
# --------------------------------------------------------------------------- #

def test_a_direction_is_stated_even_where_no_quantity_was_ever_measured():
    """"aadha kar doon" says *down* whether or not anything has a number attached."""
    compiler = ThoughtCompiler()
    op = compiler.compile("agar paani aadha kar doon to kya hoga",
                          resolve=lambda n: "paani" if n == "paani" else "",
                          state={})                      # nothing measured, ever
    assert op.verb == Verb.INTERVENE and op.target == "paani"
    assert op.direction == -1
    assert op.magnitude == 0.0, "a magnitude she does not have must not be invented"
    assert op.executable


def test_doubling_and_halving_are_different_directions():
    assert _compile("agar paani dugna kar doon to kya hoga").direction == 1
    assert _compile("agar paani aadha kar doon to kya hoga").direction == -1


def test_removing_is_not_halving():
    op = _compile("agar paani band kar doon to kya hoga")
    assert op.operation == Operation.REMOVE and op.magnitude == 0.0


def test_an_absolute_value_compiles_in_either_word_order():
    english = _compile("what if the water were 3")
    hinglish = _compile("paani 3 ho to kya hoga")
    for op in (english, hinglish):
        assert op.operation == Operation.SET and op.magnitude == pytest.approx(3.0)


def test_an_intervention_whose_size_is_not_stated_is_not_compiled():
    """"reduce the water" — a factor she chose herself would be indistinguishable afterwards."""
    assert _compile("agar paani kam kar doon to kya hoga").verb != Verb.INTERVENE


# --------------------------------------------------------------------------- #
# Sovereignty: the gate this must not open
# --------------------------------------------------------------------------- #

def test_a_greeting_compiles_to_nothing_executable():
    op = _compile("namaste NYXARA")
    assert op.verb == Verb.NONE and not op.executable


def test_a_speech_act_that_forbids_simulation_kills_an_otherwise_clean_parse():
    """The relevance gate applied to the *operation*, not only to the reply.

    This is the property that stops the whole module from being a new route by which physics
    reaches a hello. A greeting that happened to contain the word "half" still parses; it still
    must not become an intervention.
    """
    class _Act:
        kind = "greeting"
        confidence = 0.9

    op = ThoughtCompiler().compile("hello, aadha paani", act=_Act(),
                                   resolve=_resolve, state=_STATE)
    assert not op.executable
    assert "does not permit simulation" in op.why
    assert "simulate" not in op.pathways


def test_the_compiler_never_raises_and_says_so_when_it_compiled_nothing():
    assert ThoughtCompiler().compile(None).verb == Verb.NONE          # type: ignore[arg-type]
    assert NONE_OP.verb == Verb.NONE and not NONE_OP.executable


def test_stats_count_the_turns_that_failed_to_compile():
    """A number that cannot flatter: unresolved targets are counted, not quietly dropped."""
    compiler = ThoughtCompiler()
    compiler.compile("agar paani aadha kar doon to kya hoga", resolve=_resolve, state=_STATE)
    compiler.compile("agar zephyr aadha kar doon to kya hoga", resolve=_resolve, state=_STATE)
    compiler.compile("namaste", resolve=_resolve, state=_STATE)
    stats = compiler.stats()
    assert stats["compiled"] == 3
    assert stats["named_an_operation"] == 2, "a greeting is not a compiler failure"
    assert stats["executable"] == 1 and stats["unresolved_target"] == 1
    assert stats["executable_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# End to end through the brain
# --------------------------------------------------------------------------- #

def _taught() -> NJPBrain:
    brain = NJPBrain()
    for turn in ("paudhe ko paani chahiye", "paani se growth badhti hai"):
        brain.think(turn)
    return brain


def test_the_question_that_answered_nothing_now_reaches_the_do_operator():
    """The reproduced failure, end to end."""
    thought = _taught().think("agar paani aadha kar doon to kya hoga")
    assert thought.answer, "the do-operator is still unreachable from the question"
    assert "growth" in thought.answer.lower()


def test_the_answer_states_the_direction_and_refuses_the_size():
    """A stated law carries no quantities, and the answer must not pretend otherwise."""
    answer = _taught().think("agar paani aadha kar doon to kya hoga").answer.lower()
    assert "lower" in answer
    assert "never been measured" in answer or "direction only" in answer


def test_the_turn_is_classified_causal_not_empirical():
    """The second-order damage: a missing key also flipped the problem kind.

    EMPIRICAL selects three strategies that cannot answer an intervention, each returned nothing,
    and `_miss` charged every one of those losses to `simulate` — the bug was training the bandit
    against the do-operator.
    """
    thought = _taught().think("agar paani aadha kar doon to kya hoga")
    assert thought.solution is not None
    assert thought.solution.kind == "causal"


def test_the_compiled_operation_is_visible_on_the_thought():
    thought = _taught().think("agar paani aadha kar doon to kya hoga")
    assert isinstance(thought.op, CognitiveOp)
    assert thought.op.executable and thought.op.sources


def test_an_intervention_on_something_she_has_never_heard_of_is_refused():
    thought = _taught().think("agar zephyr aadha kar doon to kya hoga")
    assert not thought.answer
    assert thought.op is not None and not thought.op.executable


def test_a_greeting_still_gets_no_physics():
    """The reproduced failure this package was written against, still closed."""
    answer = _taught().think("How are you NYXARA?").answer.lower()
    assert "growth" not in answer and "paani" not in answer
