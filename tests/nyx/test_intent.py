"""NYX V.02 · intent — mood, ordering, negation scope, and the ambiguity she refuses to resolve.

The measurement this file exists to fix: V.01 read
``"mera code fix kar do lekin pehle test chala"`` as ``kind='statement'``, missing both the
imperative and the ordering constraint. The ordering is the half that decides whether an agent
is safe to hand a tool to, so it is asserted here first.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain, is_question
from nyxara.nyx.chronos import TemporalCausalMatrix
from nyxara.nyx.intent import IntentReader

HINGLISH = "mera code fix kar do lekin pehle test chala"


def _reader(**kw) -> IntentReader:
    return IntentReader(**kw)


# --------------------------------------------------------------------------- #
# The measured failure
# --------------------------------------------------------------------------- #
def test_the_measured_hinglish_instruction_is_a_command_with_an_ordering():
    got = _reader().read(HINGLISH)
    assert got.kind == "command"
    assert got.ordering, "the 'lekin pehle' constraint must not be dropped"
    before, after = got.ordering[0]
    assert "test" in before and "fix" in after


def test_the_same_instruction_in_english_reads_the_same_way():
    got = _reader().read("fix my code but first run the tests")
    assert got.kind == "command"
    before, after = got.ordering[0]
    assert "run" in before and "fix" in after


# --------------------------------------------------------------------------- #
# Mood, in three registers
# --------------------------------------------------------------------------- #
def test_questions_are_recognised_without_a_question_mark():
    assert _reader().read("why does a heavier flywheel smooth an engine").is_question


def test_hinglish_and_devanagari_questions_are_recognised():
    assert _reader().read("kya ye kaam karta hai").is_question
    assert _reader().read("गुरुत्वाकर्षण क्या है").is_question


def test_hinglish_imperatives_are_recognised():
    for line in ("ye file padho", "test chalao", "ek CHANGELOG banao", "यह फाइल पढ़ो"):
        assert _reader().read(line).is_command, line


def test_a_statement_stays_a_statement():
    got = _reader().read("gravity pulls the apple down")
    assert got.kind == "statement" and not got.is_command


def test_a_greeting_is_neither_a_question_nor_a_command():
    got = _reader().read("hello")
    assert got.kind == "greeting"


def test_the_brains_is_question_now_reads_mood():
    """V.01's opener regex saw neither of these. Both are questions."""
    assert is_question("kya ye kaam karta hai")
    assert is_question("गुरुत्वाकर्षण क्या है")
    assert not is_question("gravity pulls the apple down")


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #
def test_first_then_is_read_in_order():
    got = _reader().read("first run the tests then commit")
    before, after = got.ordering[0]
    assert "run the tests" in before and "commit" in after


def test_after_that_points_backwards_not_forwards():
    """'add X, and after that commit' must not parse inside out."""
    got = _reader().read("add a CHANGELOG, and after that commit it")
    before, after = got.ordering[0]
    assert "CHANGELOG" in before and "commit" in after


def test_x_after_y_puts_y_first():
    got = _reader().read("commit it after you run the tests")
    before, after = got.ordering[0]
    assert "run the tests" in before and "commit" in after


def test_hinglish_ke_baad_is_read():
    got = _reader().read("test chalao uske baad commit karo")
    before, after = got.ordering[0]
    assert "test" in before and "commit" in after


def test_an_instruction_with_no_ordering_reports_none():
    assert _reader().read("delete the build directory").ordering == []


# --------------------------------------------------------------------------- #
# Negation scope and contradictions
# --------------------------------------------------------------------------- #
def test_negation_does_not_leak_across_a_clause_boundary():
    got = _reader().read("run the tests but don't push")
    assert got.polarity.get("run") is True
    assert got.polarity.get("push") is False


def test_hinglish_negation_is_read():
    got = _reader().read("docs update karo, code mat badlo")
    assert got.polarity.get("badlo") is False


def test_being_asked_to_do_and_not_do_the_same_thing_is_a_contradiction():
    got = _reader().read("fix the bug but don't fix the tests")
    assert "fix" in got.contradictions
    assert "both to and not to fix" in got.clarifying_question()


# --------------------------------------------------------------------------- #
# Constraints, conditions, scope
# --------------------------------------------------------------------------- #
def test_narrowing_clauses_are_constraints():
    got = _reader().read("update the docs, but only the README")
    assert any("only" in c for c in got.constraints)


def test_hinglish_constraints_are_read():
    got = _reader().read("sirf docs me change karo")
    assert got.constraints


def test_conditions_are_separated_from_constraints():
    got = _reader().read("if the tests pass, then push")
    assert got.conditions


def test_paths_and_flags_become_scope():
    got = _reader().read("delete build/ and run pytest --maxfail=1")
    assert "build/" in got.scope
    assert any(s.startswith("--maxfail") for s in got.scope)


# --------------------------------------------------------------------------- #
# The point of the module: ambiguity is surfaced, not resolved by coin flip
# --------------------------------------------------------------------------- #
def test_a_polite_request_is_genuinely_two_readings():
    got = _reader().read("can you fix the tests?")
    assert got.ambiguous
    kinds = {r.kind for r in got.readings}
    assert kinds == {"question", "command"}


def test_an_ambiguous_turn_produces_a_question_naming_both_readings():
    ask = _reader().read("can you fix the tests?").clarifying_question()
    assert "two ways" in ask and "%" in ask


def test_an_unambiguous_turn_asks_nothing():
    assert _reader().read("what is gravity?").clarifying_question() == ""
    assert _reader().read("what is gravity?").ambiguous is False


def test_a_decisive_reading_suppresses_the_loser():
    """A reading that is beaten by more than the gap is not 'live', it is beaten."""
    got = _reader(min_reading_gap=0.01).read("can you fix the tests?")
    assert got.ambiguous is False


# --------------------------------------------------------------------------- #
# Under-specification is asked about, not assumed
# --------------------------------------------------------------------------- #
def test_a_dangling_pronoun_is_an_open_question():
    got = _reader().read("fix it")
    assert got.open_questions and "'it'" in got.open_questions[0]


def test_a_vague_quantifier_is_an_open_question():
    got = _reader().read("delete some of the old files")
    assert any("not specific enough" in q for q in got.open_questions)


def test_a_well_specified_instruction_has_nothing_to_ask():
    got = _reader().read("delete build/ and commit")
    assert got.open_questions == []


def test_open_questions_are_capped():
    got = _reader(max_open_questions=1).read("fix it with some stuff")
    assert len(got.open_questions) <= 1


# --------------------------------------------------------------------------- #
# Through the brain and the rest of NYX
# --------------------------------------------------------------------------- #
def _brain(**kw) -> NyxBrain:
    return NyxBrain(NyxConfig(**kw))


def test_a_thought_carries_the_intent_it_was_built_from():
    thought = _brain().think(HINGLISH)
    assert thought.intent is not None
    assert thought.intent.kind == "command" and thought.intent.ordering


def test_dialogue_asks_instead_of_answering_a_contradiction():
    brain = _brain()
    reply = brain.converse("fix the bug but don't fix the tests")
    assert reply.asked is True
    assert "both to and not to" in reply.text


def test_dialogue_answers_normally_when_nothing_is_ambiguous():
    reply = _brain().converse("gravity pulls the apple down")
    assert reply.asked is False


def test_the_clarifying_question_can_be_switched_off():
    brain = _brain(intent_drives_dialogue=False)
    reply = brain.converse("fix the bug but don't fix the tests")
    assert reply.asked is False


def test_chronos_now_sees_an_imperative_as_a_decision():
    """'delete build/ and commit' contains no word from the old decision marker list."""
    assert TemporalCausalMatrix.applies("delete build/ and commit")
    assert TemporalCausalMatrix.applies("should I use a heavier flywheel")   # the V.01 path
    assert not TemporalCausalMatrix.applies("gravity pulls the apple down")


def test_intent_is_reachable_on_the_brain():
    got = _brain().intent_of(HINGLISH)
    assert got is not None and got.ordering


def test_disabling_intent_leaves_v01_behaviour():
    brain = _brain(intent_enabled=False)
    assert brain.intent is None
    assert brain.intent_of("anything") is None


# --------------------------------------------------------------------------- #
# Fail-soft and honesty
# --------------------------------------------------------------------------- #
def test_everything_is_fail_soft_on_junk():
    reader = _reader()
    assert reader.read(None).kind == "statement"
    assert reader.read("").text == ""
    assert reader.read("   ").ordering == []
    assert reader.describe(reader.read(None)) != ""


def test_stats_admit_what_the_parser_cannot_do():
    stats = _reader().stats()
    assert "no part-of-speech model" in stats["note"]
    assert "under-claims rather than inventing a command" in stats["note"]


def test_an_unlisted_imperative_under_claims_rather_than_inventing_one():
    """The safe direction: a verb nobody listed reads as a statement, not as a command."""
    got = _reader().read("defenestrate the config")
    assert got.kind == "statement"
