"""The communication organ, and the seven claims it is allowed to make.

Every one of these is a property rather than a score, and each is written against the way the
mechanism could pass for the wrong reason:

* a convention is **induced**, so it reads sentences sharing no word with any lesson — and the
  same demonstrations must *not* make every question a request, which is the failure a shape with
  no counter-demonstration has;
* a correction lands on the **shape**, so the sentences it fixes are not the sentence corrected;
* a pronoun is resolved on structure or **refused**, and the refusal is asserted as hard as the
  resolution — a resolver that never abstains is a resolver that invents antecedents;
* a contradiction is **kept** rather than overwritten, and is told apart from an update by what is
  in the sentence rather than by what the sentence is about;
* attributed **ignorance** is not a false belief, because an engine that stored it as one would
  pass the false-belief items for a reason that has nothing to do with minds;
* every rendering is **parsed back**, and the four registers carry a strictly growing amount of
  one unchanged claim;
* nothing is figurative on **no evidence**, which is the half of that subject a metaphor detector
  usually gets wrong.

Two of them are asserted through ``NJPBrain.think`` rather than against the organ, because the
defects V.26 found were in the wiring and an organ measured alone reports on the organ.
"""

from __future__ import annotations

import pytest

from nyxara.njp.discourse import (
    LEVELS,
    REGISTERS,
    UNKNOWN,
    ActLearner,
    Communicator,
    Figure,
    Ledger,
    Minds,
    Register,
    proposition,
    read_claim,
    shapes_of,
)
from nyxara.njp.semantics import compile_meaning


# --------------------------------------------------------------------------- #
# 1 · speech acts, induced
# --------------------------------------------------------------------------- #

def test_a_convention_reads_a_sentence_sharing_no_word_with_any_lesson():
    acts = ActLearner()
    acts.show("Can you open the window?", "request")
    acts.show("Could you shut the gate?", "request")
    read = acts.read("Would you send the file?")
    assert read.intended == "request"
    assert read.literal == "willingness-question"     # the sentence's own mood is still read
    assert read.indirect
    assert read.support >= acts.min_support


def test_one_demonstration_leaves_nothing_and_two_identical_ones_leave_nothing_either():
    """Support counts *distinct* fillers. The same sentence shown twice is one sentence."""
    acts = ActLearner()
    acts.show("Can you open the window?", "request")
    assert acts.read("Would you send the file?").intended != "request"
    acts.show("Can you open the window?", "request")
    assert acts.read("Would you send the file?").intended != "request"


def test_a_shape_two_lessons_disagree_about_stops_being_read():
    """*"Can you swim?"* really is an ability question, and the loosest shape cannot tell.

    That level becoming contested is the correct outcome, and the more specific levels then
    separate the two constructions on their own evidence.
    """
    acts = ActLearner()
    acts.show("Can you open the window?", "request")
    acts.show("Could you shut the gate?", "request")
    acts.show("Can you swim?", "ability-question")
    acts.show("Can you juggle?", "ability-question")
    assert acts.read("Can you drive?").intended == "ability-question"
    assert acts.read("Would you close the door?").intended == "request"
    assert any(level == "opening" for level, _shape in acts.contested)


def test_the_four_levels_are_ordered_from_specific_to_general():
    shapes = shapes_of("Can you open the window?")
    assert len(shapes) == len(LEVELS)
    assert shapes[0][0] == "can" and shapes[1][0] == "MODAL"
    assert shapes[3][-2] == "+"


def test_a_correction_lands_on_the_shape_and_not_on_the_sentence():
    acts = ActLearner()
    acts.misread("Is the seat free to you?", took_as="polar-question", meant="offer")
    acts.misread("Is the room open to you?", took_as="polar-question", meant="offer")
    assert acts.corrections == 2
    # The sentences that were never corrected are the measurement.
    assert acts.read("Is the desk spare to you?").intended == "offer"
    assert acts.generalised >= 1


# --------------------------------------------------------------------------- #
# 2 · reference
# --------------------------------------------------------------------------- #

def _after(voice, *turns):
    uptake = None
    for turn in turns:
        uptake = voice.hear(turn)
    return uptake


def test_a_pronoun_with_one_candidate_is_settled():
    voice = Communicator()
    got = _after(voice, "ravi left.", "He was tired.")
    assert got.resolutions[0].referent == "ravi"
    assert got.resolutions[0].settled


def test_a_pronoun_with_two_equal_candidates_is_refused_and_asked_about():
    voice = Communicator()
    got = _after(voice, "ravi met arun.", "He was tired.")
    resolved = got.resolutions[0]
    assert resolved.ambiguous and not resolved.referent
    assert {name for name, _score in resolved.ranked} == {"ravi", "arun"}
    assert "ravi" in resolved.question and "arun" in resolved.question
    assert got.question == resolved.question


def test_a_pronoun_in_object_position_cannot_name_its_own_clause_subject():
    voice = Communicator()
    got = _after(voice, "ravi met arun.", "arun gave him the book.")
    resolved = next(r for r in got.resolutions if r.pronoun == "him")
    assert resolved.referent == "ravi" and resolved.settled


def test_number_disagreement_is_a_filter_rather_than_a_penalty():
    voice = Communicator()
    got = _after(voice, "ravi met the wardens.", "They left.")
    assert got.resolutions[0].referent == "wardens"


def test_what_the_store_has_witnessed_separates_two_identical_discourses():
    """*"Ravi walked home"* and *"Ravi met Arun"* have the same shape and different answers."""
    kinds = {"ravi": ["person"], "arun": ["person"], "sara": ["person"],
             "devi": ["person"], "meera": ["person"], "home": ["place"]}
    def known(name):
        return kinds.get(str(name).lower(), [])

    def voice():
        made = Communicator(kinds=known)
        for name in ("sara", "devi", "meera"):
            made.figure.witness("tir", name)
        return made

    settled = _after(voice(), "ravi walked home.", "He was tired.")
    assert settled.resolutions[0].referent == "ravi"
    open_still = _after(voice(), "ravi met arun.", "He was tired.")
    assert open_still.resolutions[0].ambiguous


def test_only_the_third_person_is_substituted_into_the_surface():
    """Rewriting *"mera"* into a name broke a Hinglish question the grounder had always read."""
    voice = Communicator()
    got = voice.hear("mera naam kya hai?")
    assert got.rewritten == got.surface
    assert got.resolutions and got.resolutions[0].referent == "Master"


# --------------------------------------------------------------------------- #
# 3 · the ledger
# --------------------------------------------------------------------------- #

def test_a_universal_cannot_be_updated_away_by_an_instance():
    ledger = Ledger()
    ledger.note(read_claim("I never visited Delhi.", turn=1))
    verdict = ledger.note(read_claim("When I visited Delhi last year I was tired.", turn=2))
    assert verdict.kind == "contradicts"
    assert verdict.question
    # Both claims are kept, and `holds` refuses to pick one.
    assert len(ledger.claims) == 2
    assert ledger.holds("i", "visit") == ""


def test_a_marked_change_is_an_update_and_an_unmarked_one_is_not():
    marked = Ledger()
    marked.note(read_claim("The key is in the drawer.", turn=1))
    assert marked.note(read_claim("The key is now on the table.", turn=2)).kind == "updates"
    assert marked.holds("key", "is_at") == "table"

    plain = Ledger()
    plain.note(read_claim("The key is in the drawer.", turn=1))
    # Nothing has yet shown this relation to hold one value at a time, so a second value is more
    # information rather than a conflict. Claiming otherwise would be an ontology, not a reading.
    assert plain.note(read_claim("The key is on the table.", turn=2)).kind == "new"


def test_which_relations_are_stateful_is_learned_from_the_transcript():
    ledger = Ledger()
    ledger.note(read_claim("The key is in the drawer.", turn=1))
    ledger.note(read_claim("The key is now on the table.", turn=2))
    assert "is_at" in ledger.stateful
    # And now the same unmarked reversion that was `new` above is a conflict, because the speaker
    # has demonstrated that this relation holds one value at a time.
    assert ledger.note(read_claim("The key is in the drawer.", turn=3)).kind == "contradicts"
    assert ledger.holds("key", "is_at") == ""


def test_the_same_claim_said_again_corroborates_and_two_unrelated_ones_do_not_conflict():
    ledger = Ledger()
    ledger.note(read_claim("ravi owns the boat.", turn=1))
    assert ledger.note(read_claim("ravi owns the boat.", turn=2)).kind == "corroborates"
    assert ledger.note(read_claim("arun owns the cart.", turn=3)).kind == "new"


def test_a_question_puts_nothing_on_the_ledger():
    assert read_claim("Have I visited Delhi?") is None
    assert read_claim("Can you open the window?") is None


def test_four_turns_and_the_fact_is_in_the_third():
    voice = Communicator()
    voice.hear("The key is in the drawer.")
    voice.hear("The lamp lit the room.")
    voice.hear("The key is now on the table.")
    assert voice.holds("key", "is_at") == "table"


# --------------------------------------------------------------------------- #
# 4 · other minds
# --------------------------------------------------------------------------- #

def test_a_sentence_drives_the_recursive_belief_store():
    minds = Minds()
    minds.hear("The box has the red ball.")
    minds.hear("ravi thinks the box has the blue ball.")
    assert minds.world("box|have") == "red ball"
    assert minds.believes("ravi", "box|have") == "blue ball"
    assert minds.false_belief("ravi", "box|have")


def test_attributed_ignorance_is_not_a_false_belief():
    minds = Minds()
    minds.hear("The box has the red ball.")
    minds.hear("ravi thinks I do not know the box has the red ball.")
    assert minds.attributes("ravi", minds.speaker, "box|have") == UNKNOWN
    assert minds.attributes_ignorance("ravi", minds.speaker, "box|have")
    # And it is emphatically not scored as an error about the world.
    assert not minds.false_belief(minds.speaker, "box|have")


def test_an_agent_who_is_right_does_not_hold_a_false_belief():
    minds = Minds()
    minds.hear("The box has the red ball.")
    minds.hear("ravi thinks the box has the red ball.")
    assert not minds.false_belief("ravi", "box|have")


def test_depth_that_repeats_the_level_below_is_not_stored():
    """The bound worth having is about content rather than about counting."""
    minds = Minds()
    minds.hear("ravi thinks the box has the red ball.")
    # Ravi already believes this, so "ravi thinks arun thinks" the same thing adds nothing and is
    # not stored — that is the *negligible value* rule, made mechanical.
    minds.hear("ravi thinks arun thinks the box has the red ball.")
    assert minds.collapsed == 1
    assert minds.attributes("ravi", "arun", "box|have") is None
    # A nesting that genuinely differs from the level below it is kept.
    minds.hear("ravi thinks arun thinks the box has the blue ball.")
    assert minds.attributes("ravi", "arun", "box|have") == "blue ball"
    assert minds.deepest <= minds.max_depth


def test_a_clause_is_keyed_on_subject_and_relation_with_the_object_as_its_value():
    assert proposition("the box has the red ball") == ("box|have", "red ball")
    assert proposition("the marble is in the basket") == ("marble|is_at", "basket")


# --------------------------------------------------------------------------- #
# 5 · register
# --------------------------------------------------------------------------- #

def test_four_registers_carry_one_claim_and_a_growing_amount_of_it():
    meaning = compile_meaning("water boils today")
    meaning.modality, meaning.condition = "typical", "the pressure is normal"
    spread = Register().spread(meaning)
    assert all(said.verified for said in spread.values())
    assert len({said.claim for said in spread.values()}) == 1
    widths = [spread[audience].words for audience in REGISTERS]
    assert widths == sorted(widths)
    assert widths[0] < widths[-1]


def test_a_rendering_that_does_not_read_back_is_not_returned():
    register = Register()
    said = register.say(compile_meaning("this sentence has no frame at all whatsoever"), "child")
    assert not said.verified
    assert said.text in ("", said.claim)


def test_polarity_survives_the_round_trip():
    meaning = compile_meaning("Zorbins do not need glarn.")
    said = Register().say(meaning, "student")
    assert said.verified
    assert compile_meaning(said.claim).negated


# --------------------------------------------------------------------------- #
# 6 · figurative
# --------------------------------------------------------------------------- #

def test_nothing_is_figurative_on_no_evidence():
    figure = Figure(kinds=lambda name: ["institution"] if name == "market" else [])
    judged = figure.judge(compile_meaning("The market swallowed the shock."))
    assert not judged.figurative
    assert "below" in judged.why


def test_a_selectional_violation_is_read_off_what_was_witnessed():
    kinds = {"wolf": ["animal"], "dog": ["animal"], "bear": ["animal"],
             "market": ["institution"]}
    figure = Figure(kinds=lambda name: kinds.get(str(name).lower(), []))
    for name in ("wolf", "dog", "bear"):
        figure.witness("swallow", name)
    judged = figure.judge(compile_meaning("The market swallowed the shock."))
    assert judged.figurative and judged.violated == "animal"
    # And the literal half, where flagging would be a false alarm.
    assert not figure.judge(compile_meaning("The wolf swallowed the meat.")).figurative


def test_an_unknown_subject_is_not_called_figurative():
    kinds = {"wolf": ["animal"], "dog": ["animal"], "bear": ["animal"]}
    figure = Figure(kinds=lambda name: kinds.get(str(name).lower(), []))
    for name in ("wolf", "dog", "bear"):
        figure.witness("swallow", name)
    judged = figure.judge(compile_meaning("The zorbin swallowed the glarn."))
    assert not judged.figurative and "unknown" in judged.why


def test_a_metaphor_never_becomes_the_evidence_for_the_next_one():
    kinds = {"wolf": ["animal"], "dog": ["animal"], "bear": ["animal"],
             "market": ["institution"], "fund": ["institution"]}
    voice = Communicator(kinds=lambda name: kinds.get(str(name).lower(), []))
    for name in ("wolf", "dog", "bear"):
        voice.figure.witness("swallow", name)
    voice.hear("The market swallowed the shock.")
    assert "market" not in voice.figure.witnesses.get("swallow", [])
    assert voice.figure.judge(compile_meaning("The fund swallowed the loss.")).figurative


# --------------------------------------------------------------------------- #
# 7 · a conversation, and what it is not allowed to do
# --------------------------------------------------------------------------- #

def test_a_new_conversation_keeps_the_conventions_and_drops_the_referents():
    voice = Communicator()
    voice.show("Can you open the window?", "request")
    voice.show("Could you shut the gate?", "request")
    voice.hear("ravi met arun.")
    voice.reset()
    assert voice.reference.referents == []
    assert voice.ledger.claims == []
    assert voice.acts.read("Would you send the file?").intended == "request"


def test_growth_is_bounded():
    voice = Communicator()
    voice.ledger.max_claims = 8
    voice.reference.max_referents = 8
    for index in range(40):
        voice.hear(f"anna{index} met bo{index}.")
    assert len(voice.ledger.claims) <= 8
    assert len(voice.reference.referents) <= 8


@pytest.mark.parametrize("surface", [
    "", "   ", "?", "…", "a", "the the the", "Can you?", "never never never",
])
def test_every_entry_point_is_fail_soft(surface):
    voice = Communicator()
    got = voice.hear(surface)
    assert got is not None
    assert isinstance(voice.stats(), dict)
    assert isinstance(voice.figurative(surface), bool)


# --------------------------------------------------------------------------- #
# 8 · through the brain, because an organ measured alone reports on the organ
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def brain():
    from nyxara.njp.brain import NJPBrain

    return NJPBrain()


def test_a_denial_is_confirmed_as_a_denial(brain):
    """`noted: Master visit delhi` was the reply to *"I never visited Delhi."* for many versions."""
    from nyxara.njp.brain import _confirm

    said = brain.think("I never visited Delhi.").answer.lower()
    assert "noted" in said
    assert "does not" in said or "never" in said

    class _Triple:
        subject, predicate, object, negated = "master", "visit", "delhi", True

    assert _confirm(_Triple()) == "master does not visit delhi"
    _Triple.negated = False
    assert _confirm(_Triple()) == "master visit delhi"


def test_a_contradiction_reaches_the_reply(brain):
    said = brain.think("When I visited Delhi last year I was tired.").answer
    assert "which of those holds" in said


def test_a_question_she_cannot_ground_still_abstains():
    """Every abstention subject in this repository scores silence as right and any reply as wrong,
    so a clarification offered where she would have abstained is a regression on all of them."""
    from nyxara.njp.brain import NJPBrain

    fresh = NJPBrain()
    assert fresh.think("What did the man with the telescope see?").answer == ""
    assert fresh.think("Which zorbin needs the glarn?").answer == ""


def test_a_metaphor_is_not_filed_as_a_fact():
    from nyxara.njp.brain import NJPBrain

    fresh = NJPBrain()
    for animal in ("wolf", "dog", "bear"):
        fresh.think(f"a {animal} is an animal")
        fresh.think(f"the {animal} swallowed the meat")
    fresh.think("a market is an institution")
    before = fresh.think("The market swallowed the shock.")
    assert before.percept.uptake is not None
    assert not before.percept.uptake.literal
    assert not list(getattr(before.percept.grounding, "triples", None) or [])


def test_a_location_is_answered_from_the_conversation_and_a_dispute_is_not():
    from nyxara.njp.brain import NJPBrain

    settled = NJPBrain()
    settled.think("The key is in the drawer.")
    settled.think("The key is now on the table.")
    assert settled.think("Where is the key?").answer == "table"

    disputed = NJPBrain()
    disputed.think("The key is in the drawer.")
    disputed.think("The key is now on the table.")
    disputed.think("The key is in the drawer.")
    assert "which of those holds" in disputed.think("Where is the key?").answer
