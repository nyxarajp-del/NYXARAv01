"""One fact, reachable in every language the Master actually writes in.

The gap this closes, measured before it was closed::

    "my name is Jay"      then  "what is my name"    ->  'Jay'
                          then  "mera naam kya hai"  ->  ''
                          then  "मेरा नाम क्या है"    ->  ''
                          then  "tell me my name"    ->  ''

The fact was stored. It was reachable through exactly one English phrasing, and invisible through
every other way of asking for the same thing. Three separate causes, all of them the same mistake
made in three places — an assumption that a sentence is head-initial, which is a statement about
English and not about language:

* :meth:`Grounder._is_question` read the **first word only**. English puts the wh-word there;
  Hindi and Hinglish leave it *in situ* ("mera naam **kya** hai"), so the turn was never even
  treated as a question.
* ``_QUESTION_PATTERNS`` had no verb-final forms at all.
* ``_SEED_PATTERNS`` had none either, so once the questions worked there was still nothing to
  find — "Ravi ka naam Ravi Kumar hai" had never extracted.

And one that was purely mechanical: ``_predicate`` normalised on an ASCII character class, so
every Devanagari relation name was deleted outright.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.grounding import Grounder, _norm_relation


@pytest.fixture()
def taught() -> Grounder:
    """One grounder, told a handful of facts in English."""
    grounder = Grounder()
    for line in ("my name is Jay",
                 "Ravi is a person",
                 "Ravi lives in Mumbai",
                 "Ravi works at Acme",
                 "I live in Pune"):
        grounder.ground(line)
    return grounder


def _ask(grounder: Grounder, question: str) -> str:
    answer = grounder.ground(question).answer
    return str(getattr(answer, "text", "") or "")


# ---- the turn is recognised as a question at all ------------------------------ #
@pytest.mark.parametrize("question", [
    "mera naam kya hai",
    "मेरा नाम क्या है",
    "mera naam batao",
    "मेरा नाम बताओ",
    "tell me my name",
    "Ravi kahan rehta hai",
    "Ravi kaun hai",
])
def test_a_question_is_recognised_wherever_its_wh_word_sits(question: str):
    """A head-word test finds ``what is…`` and misses ``…kya hai`` — same question, both times."""
    assert Grounder()._is_question(question) is True


@pytest.mark.parametrize("statement", [
    "mera naam Jay hai",
    "Ravi ka naam Ravi Kumar hai",
    "Ravi Mumbai me rehta hai",
    "my name is Jay",
])
def test_a_statement_is_not_mistaken_for_a_question(statement: str):
    """Searching the whole turn must not turn every possessive into an interrogative."""
    assert Grounder()._is_question(statement) is False


# ---- the same fact, asked for in every language -------------------------------- #
@pytest.mark.parametrize("question", [
    "what is my name",
    "mera naam kya hai",
    "मेरा नाम क्या है",
    "mera naam batao",
    "मेरा नाम बताओ",
    "tell me my name",
])
def test_one_fact_is_reachable_through_every_way_of_asking(taught: Grounder, question: str):
    assert _ask(taught, question) == "Jay"


@pytest.mark.parametrize("question,expected", [
    ("where does Ravi live", "Mumbai"),
    ("Ravi kahan rehta hai", "Mumbai"),
    ("who is Ravi", "person"),
    ("Ravi kaun hai", "person"),
    ("Ravi kya hai", "person"),
    ("where does Ravi work", "Acme"),
    ("Ravi kahan kaam karta hai", "Acme"),
    ("main kahan rehta hoon", "Pune"),
    ("मैं कहाँ रहता हूँ", "Pune"),
])
def test_each_relation_is_askable_in_hinglish(taught: Grounder, question: str, expected: str):
    assert _ask(taught, question) == expected


def test_a_question_about_something_never_said_is_still_unknown(taught: Grounder):
    """Reading more languages must not mean inventing more answers."""
    assert _ask(taught, "Zoran kaun hai") == ""
    assert _ask(taught, "Zoran ka naam kya hai") == ""


# ---- she can be TOLD things in Hinglish, not only asked ------------------------ #
@pytest.mark.parametrize("statement,triple", [
    ("mera naam Jay hai", ("Master", "has_name", "Jay")),
    ("Ravi ka naam Ravi Kumar hai", ("Ravi", "has_name", "Ravi Kumar")),
    ("Sara Delhi me rehti hai", ("Sara", "located_in", "Delhi")),
    ("Amit Acme me kaam karta hai", ("Amit", "works_at", "Acme")),
    ("Ravi ek engineer hai", ("Ravi", "is_a", "engineer")),
])
def test_a_verb_final_statement_extracts(statement: str, triple):
    """A language she can be questioned in but not told anything in is half a language."""
    got = Grounder().ground(statement).triples
    assert [(t.subject, t.predicate, t.object) for t in got] == [triple]


def test_told_in_one_language_asked_in_another():
    """The whole point: the store is a fact graph, not a phrasebook."""
    grounder = Grounder()
    grounder.ground("Sara Delhi me rehti hai")          # told in Hinglish
    assert _ask(grounder, "where does Sara live") == "Delhi"      # asked in English

    grounder.ground("Amit works at Acme")               # told in English
    assert _ask(grounder, "Amit kahan kaam karta hai") == "Acme"  # asked in Hinglish


def test_devanagari_and_roman_reach_the_same_fact():
    grounder = Grounder()
    grounder.ground("मेरा नाम जय है")
    assert _ask(grounder, "मेरा नाम क्या है") == "जय"
    assert _ask(grounder, "mera naam kya hai") == "जय"
    assert _ask(grounder, "what is my name") == "जय"


# ---- the relation name survives its own script ---------------------------------- #
def test_a_devanagari_relation_is_not_normalised_away():
    """``[a-z0-9]`` deleted it entirely; ``\\w`` mangled the matra into an underscore."""
    assert _norm_relation("नाम") == "नाम"
    assert _norm_relation("has name") == "has_name"
    assert _norm_relation("  Has-Name!  ") == "has_name"


def test_a_devanagari_relation_folds_onto_the_english_edge():
    assert Grounder()._predicate("नाम") == "has_name"
    assert Grounder()._predicate("naam") == "has_name"
    assert Grounder()._predicate("name") == "has_name"


@pytest.mark.parametrize("word,edge", [
    ("ghar", "located_in"), ("घर", "located_in"), ("shehar", "located_in"),
    ("kaam", "works_at"), ("काम", "works_at"), ("naukri", "works_at"),
])
def test_hinglish_property_words_fold_onto_the_right_edge(word: str, edge: str):
    assert Grounder()._predicate(word) == edge


# ---- and it works through the whole brain, not only the grounder ----------------- #
def test_the_brain_answers_the_question_that_started_all_this():
    brain = NJPBrain()
    brain.think("my name is Jay")
    assert brain.think("what is my name").answer == "Jay"
    assert brain.think("mera naam kya hai").answer == "Jay"
    assert brain.think("मेरा नाम क्या है").answer == "Jay"
    assert brain.think("tell me my name").answer == "Jay"


def test_a_hinglish_question_does_not_pay_for_the_ladder_either():
    """Being answerable in Hinglish means it is answered cheaply, exactly like English."""
    brain = NJPBrain()
    brain.think("my name is Jay")
    brain.think("mera naam kya hai")
    assert brain.stats()["reason"]["passes"] == 0


def test_a_hinglish_turn_still_grows_the_fabric():
    brain = NJPBrain()
    brain.think("my name is Jay")
    thought = brain.think("mera naam kya hai")
    assert thought.growth is not None
    assert thought.loop is not None and thought.loop.capabilities > 0
