"""The semantic compiler, and the two claims that justify it existing.

**Claim one: the verb is a position, not a list.** ``nyxara/njp/grounding.py`` reads a sentence
with 101 word-level regexes. Every verb outside them was a fact she could not be told, and the
open-verb tests here are what say whether frames over tags actually removed that ceiling — they
use ordinary transitive English verbs chosen precisely because no lexicon in the repo names them.

**Claim two: a denial is not an assertion.** Before this module the string ``negat`` did not occur
in ``grounding.py``, and ``"Zorbins don't need glarn."`` grounded to
``("Zorbins don't", requires, glarn)`` — the negator absorbed into the subject, so the denial of a
claim was stored as a different entity's assertion of it. Those tests are the regression guard on
the worst failure this module was built for, and they check the whole path: extraction, storage,
retrieval, derivation, and the polar answer.

The precision half is tested as hard as the recall half, deliberately. A compiler that reads every
sentence is not an achievement if it also reads the ones that say nothing — and the first version
of this one did exactly that, grounding *"cricket ke baare mein mujhe zyada nahi pata"* ("I don't
know much about cricket") as ``cricket baare mujhe --zyada--> pata``. Every refusal below is a
sentence it must decline.
"""

from __future__ import annotations

import pytest

from nyxara.njp.grounding import Grounder
from nyxara.njp.semantics import Tag, compile_meaning, tag_tokens


# --------------------------------------------------------------------------- #
# tagging — the closed class, and the contraction that hid a negator
# --------------------------------------------------------------------------- #

def test_the_closed_class_is_tagged_and_everything_else_is_open():
    tags = {t.text: t.tag for t in tag_tokens("the zorbin does not need glarn")}
    assert tags["the"] == Tag.DET
    assert tags["does"] == Tag.AUX
    assert tags["not"] == Tag.NEG
    # The two content words carry no tag of their own, which is the point: the compiler knows
    # nothing about "zorbin" or "glarn" and must still read the sentence.
    assert tags["zorbin"] == Tag.WORD
    assert tags["glarn"] == Tag.WORD


def test_a_contraction_is_split_so_the_negator_is_its_own_token():
    """``don't`` is an auxiliary and a negator. Joined, the negator rides into the subject."""
    tags = [(t.text, t.tag) for t in tag_tokens("zorbins don't need glarn")]
    assert ("do", Tag.AUX) in tags
    assert any(tag == Tag.NEG for _text, tag in tags)


def test_an_alphanumeric_identifier_stays_one_token():
    """``h2o`` is one entity. Split into three tokens it is an entity nothing can look up."""
    assert [t.text for t in tag_tokens("h2o freezes")] == ["h2o", "freezes"]


# --------------------------------------------------------------------------- #
# claim one — the verb is a position
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("verb", ["eat", "avoid", "produce", "carry", "hunt", "emit",
                                  "devour", "guard", "absorb", "filter"])
def test_any_transitive_verb_compiles(verb):
    """No lexicon names these. If the frame is a position rather than a list, all of them read."""
    meaning = compile_meaning(f"zorbins {verb} glarn.")
    assert meaning.kind == "assertion"
    assert meaning.subject == "zorbins"
    assert meaning.object == "glarn"
    assert meaning.relation


@pytest.mark.parametrize("question,subject,relation", [
    ("what does a zorbin eat?", "zorbin", "eat"),
    ("zorbins need what?", "zorbins", "need"),
    ("tell me what a zorbin requires.", "zorbin", "require"),
    ("what's necessary for a zorbin?", "zorbin", "necessary"),
    ("a zorbin needs what exactly?", "zorbin", "need"),
    ("zorbin ko kya chahiye?", "zorbin", "chahiye"),
    ("zorbins need?", "zorbins", "need"),
])
def test_seven_ordinary_question_forms_reach_the_same_pair(question, subject, relation):
    """Surface robustness, isolated. Five of these previously reached nothing at all."""
    meaning = compile_meaning(question, interrogative=True)
    assert meaning.kind in ("question", "polar_question")
    assert meaning.subject == subject
    assert meaning.relation == relation


def test_the_act_may_be_told_rather_than_read_off_punctuation():
    """``_clean`` strips the question mark before the grounder asks, so the caller has to say.

    Without the hint "zorbins need what" compiles as an *assertion* that zorbins need something
    called "what" — which is what four of the seven forms above were doing.
    """
    assert compile_meaning("zorbins need what").kind != "question"
    assert compile_meaning("zorbins need what", interrogative=True).kind == "question"


# --------------------------------------------------------------------------- #
# claim two — a denial is not an assertion
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sentence", [
    "zorbins don't need glarn.",
    "zorbins do not need glarn.",
    "zorbins never need glarn.",
])
def test_a_denial_keeps_its_subject_and_reports_itself_negated(sentence):
    meaning = compile_meaning(sentence)
    # The subject is the entity, not the entity plus the negator — the exact defect.
    assert meaning.subject == "zorbins"
    assert meaning.object == "glarn"
    assert meaning.negated is True


def test_the_same_sentence_asserted_and_denied_differ_only_in_polarity():
    asserted = compile_meaning("zorbins need glarn.")
    denied = compile_meaning("zorbins don't need glarn.")
    assert (asserted.subject, asserted.relation, asserted.object) == \
           (denied.subject, denied.relation, denied.object)
    assert asserted.negated is False and denied.negated is True


def test_a_denial_never_answers_the_question_as_though_asserted():
    """The whole path: extraction, storage, retrieval, derivation and recall.

    Four separate readers had to learn this. ``Grounder._lookup`` filtering alone left the answer
    coming back through ``_facts_of``; fixing that left it coming back through
    ``Core._facts``, which reads ``grounder.facts`` directly. A brain that answers "what does X
    need?" with something X was said *not* to need is stating the opposite of what it was told.
    """
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    brain.think("zorbins don't need glarn.")
    assert "glarn" not in (brain.think("what does a zorbin need?").answer or "")


def test_a_denial_is_kept_rather_than_discarded():
    """Excluded from positive answers, still on record — it is knowledge, and "no" needs it."""
    grounder = Grounder()
    grounder.ground("zorbins don't need glarn.")
    assert grounder._lookup("zorbins", "requires") == []
    denied = grounder._lookup("zorbins", "requires", negated=True)
    assert len(denied) == 1 and denied[0].object == "glarn"


# --------------------------------------------------------------------------- #
# polar — yes, no, and unknown are three states
# --------------------------------------------------------------------------- #

def test_a_polar_question_has_three_answers_not_two():
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    brain.think("zorbins need vim.")
    brain.think("zorbins don't need glarn.")
    assert "yes" in (brain.think("do zorbins need vim?").answer or "").lower()
    assert "no" in (brain.think("do zorbins need glarn?").answer or "").lower()
    # Never mentioned either way. The failure this guards is not a wrong verdict — it is
    # answering a polar question by naming some *other* object the subject relates to.
    unknown = (brain.think("do zorbins need keth?").answer or "").lower()
    assert "vim" not in unknown and "glarn" not in unknown


def test_a_composed_verdict_survives_the_polar_conversion():
    """A two-hop chain answers "yes" on its own, and that verdict is not an object to be checked.

    The guard that turns a derived *object* into yes/no compared "yes" against the asked object,
    found no match, and discarded a correctly composed answer.
    """
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    brain.think("aag se garmi hoti hai")
    brain.think("garmi se pasina hota hai")
    assert "yes" in (brain.think("does aag cause pasina").answer or "").lower()


# --------------------------------------------------------------------------- #
# precision — the sentences it must refuse
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sentence,why", [
    ("cricket ke baare mein mujhe zyada nahi pata",
     "case markers, no relation — English SVO reads it as one and invents a triple"),
    ("my lucky number is seven",
     "copular; which relation it names is decided by patterns written for copulas"),
    ("tell me about deep learning",
     "prepositional complement — there is no subject in 'about deep learning'"),
    ("define overfitting",
     "an imperative, not a subject and a verb"),
])
def test_a_sentence_that_states_no_relation_is_refused(sentence, why):
    meaning = compile_meaning(sentence)
    assert not meaning.complete, why


def test_refusal_is_reported_rather_than_guessed():
    """An unreadable surface returns an unreadable Meaning — the caller keeps what it had."""
    meaning = compile_meaning("hmm")
    assert not meaning.readable
    assert meaning.subject == "" and meaning.relation == ""


# --------------------------------------------------------------------------- #
# the rest of the representation
# --------------------------------------------------------------------------- #

def test_a_condition_is_separated_from_the_claim_it_qualifies():
    meaning = compile_meaning("if water is low, plants may weaken.")
    assert meaning.condition == "water is low"
    assert meaning.modality == "possible"
    assert meaning.subject == "plants"


def test_an_evidential_lowers_the_confidence_of_what_follows_it():
    hedged = compile_meaning("i think zorbins carry glarn.")
    plain = compile_meaning("zorbins carry glarn.")
    assert hedged.evidential == "hedged"
    assert hedged.confidence < plain.confidence
    assert hedged.object == "glarn"


def test_hinglish_sov_is_read_by_its_own_frame():
    """The English frame does not fail on SOV — it matches *wrongly*, which is worse."""
    meaning = compile_meaning("zorbins ko glarn nahi chahiye.")
    assert meaning.subject == "zorbins"
    assert meaning.object == "glarn"
    assert meaning.negated is True


def test_a_copular_question_keeps_its_noun_phrase_whole():
    """"what is deep learning" asks about *deep learning*, not what "deep" does."""
    meaning = compile_meaning("what is deep learning", interrogative=True)
    assert meaning.subject == "deep learning"


def test_the_compiler_never_raises():
    for junk in ("", "   ", "?", "!!!", "\x00", "ke ke ke ke", "a" * 5000):
        assert compile_meaning(junk) is not None
