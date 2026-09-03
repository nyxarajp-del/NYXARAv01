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
    LICENCE,
    UNIVERSAL,
    ActLearner,
    AttachLearner,
    Communicator,
    Figure,
    Ledger,
    MarkerLearner,
    Minds,
    Reference,
    Referent,
    Register,
    proposition,
    read_claim,
    shapes_of,
)
from nyxara.njp.discourse import attachment as read_attachment
from nyxara.njp.semantics import compile_meaning


#: Demonstrated verdicts, varied so that what is induced is the marker and not the preposition.
#: Nothing in the module knows that "now" licenses a change or that "never" quantifies over all
#: times; these six pairs are where it finds out.
VERDICTS = (
    ("The key is in the drawer.", "The key is now on the table.", "updates"),
    ("The lamp is on the shelf.", "The lamp is now under the box.", "updates"),
    ("The book is under the bed.", "The book is now beside the chair.", "updates"),
    ("I never visited Delhi.", "When I visited Delhi last year I was tired.", "contradicts"),
    ("Sara never entered Pune.", "When Sara entered Pune in April Sara was calm.", "contradicts"),
    ("The cup is in the sink.", "The cup is on the rack.", "contradicts"),
    ("Ravi owns the boat.", "Ravi owns the boat.", "corroborates"),
)


def taught_markers():
    learner = MarkerLearner()
    for first, second, verdict in VERDICTS:
        learner.show(first, second, verdict)
    return learner


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


def test_the_levels_are_ordered_from_specific_to_general():
    shapes = shapes_of("Can you open the window?")
    assert len(shapes) == len(LEVELS)
    assert shapes[0][0] == "can" and shapes[1][0] == "MODAL"
    assert shapes[-1][-2] == "+"


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

def test_the_markers_are_induced_and_nothing_else_survives():
    """Six demonstrated verdicts, and the two words that carry them fall out of the differences."""
    learner = taught_markers()
    assert learner.role("now") == LICENCE
    assert learner.role("never") == UNIVERSAL
    # The copula is in both sentences of every pair, and the prepositions are varied, so neither
    # survives. That is the whole mechanism: what differs *and* agrees across demonstrations.
    assert learner.role("is") == ""
    assert learner.role("on") == ""
    assert learner.role("the") == ""


def test_a_marker_demonstrated_two_ways_is_contested_and_never_consulted():
    learner = MarkerLearner()
    learner.show("The key is in the drawer.", "The key is now on the table.", "updates")
    learner.show("The lamp is on the shelf.", "The lamp is now under the box.", "updates")
    assert learner.role("now") == LICENCE
    learner.show("The cup is in the sink.", "The cup is now on the rack.", "contradicts")
    assert learner.role("now") == ""
    assert "now" in learner.contested


def test_untaught_no_word_licenses_anything():
    """The honest floor: a listener who has never heard anybody signal a change."""
    claim = read_claim("The key is now on the table.", turn=1)
    assert claim is not None
    assert not claim.change and not claim.universal
    assert "now" in claim.markers


def test_a_universal_cannot_be_updated_away_by_an_instance():
    markers = taught_markers()
    ledger = Ledger()
    ledger.note(read_claim("I never visited Delhi.", turn=1, markers=markers))
    verdict = ledger.note(read_claim("When I visited Delhi last year I was tired.", turn=2,
                                     markers=markers))
    assert verdict.kind == "contradicts"
    assert verdict.question
    # Both claims are kept, and `holds` refuses to pick one.
    assert len(ledger.claims) == 2
    assert ledger.holds("i", "visit") == ""


def test_a_marked_change_is_an_update_and_an_unmarked_one_is_not():
    markers = taught_markers()
    marked = Ledger()
    marked.note(read_claim("The key is in the drawer.", turn=1, markers=markers))
    assert marked.note(read_claim("The key is now on the table.", turn=2,
                                  markers=markers)).kind == "updates"
    assert marked.holds("key", "is_at") == "table"

    plain = Ledger()
    plain.note(read_claim("The key is in the drawer.", turn=1, markers=markers))
    # Nothing has yet shown this relation to hold one value at a time, so a second value is more
    # information rather than a conflict. Claiming otherwise would be an ontology, not a reading.
    assert plain.note(read_claim("The key is on the table.", turn=2,
                                 markers=markers)).kind == "new"

    # And with nothing induced, the marked pair reads as the unmarked one — the capability comes
    # from the demonstrations rather than from the module.
    untaught = Ledger()
    untaught.note(read_claim("The key is in the drawer.", turn=1))
    assert untaught.note(read_claim("The key is now on the table.", turn=2)).kind == "new"


def test_which_relations_are_stateful_is_learned_from_the_transcript():
    markers = taught_markers()
    ledger = Ledger()
    ledger.note(read_claim("The key is in the drawer.", turn=1, markers=markers))
    ledger.note(read_claim("The key is now on the table.", turn=2, markers=markers))
    assert "is_at" in ledger.stateful
    # And now the same unmarked reversion that was `new` above is a conflict, because the speaker
    # has demonstrated that this relation holds one value at a time.
    assert ledger.note(read_claim("The key is in the drawer.", turn=3,
                                  markers=markers)).kind == "contradicts"
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
    for first, second, verdict in VERDICTS:
        voice.show_change(first, second, verdict)
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


def test_an_attitude_verb_is_read_by_structure_and_not_from_a_list():
    """`reckons` was in no table, and *"opened"* is in none either."""
    minds = Minds()
    minds.hear("The box has the red ball.")
    minds.hear("ravi reckons the box has the blue ball.")
    assert minds.false_belief("ravi", "box|have")

    # And what must NOT be read as an attitude: a noun phrase is not a clause, however much the
    # compiler's bare `np-verb` frame would like to make one out of two nouns in a row.
    quiet = Minds()
    quiet.hear("ravi opened the box in the room")
    assert quiet.believes("ravi", "box|is_at") is None
    quiet.hear("ravi gave arun the book")
    assert quiet.believes("ravi", "arun|book") is None


def test_a_clause_needs_a_predicate_of_its_own():
    from nyxara.njp.discourse import clause_proposition

    assert clause_proposition("the box has the red ball")[0] == "box|have"
    assert clause_proposition("the box is in the room")[0] == "box|is_at"
    assert clause_proposition("the wolf eats the meat")[0] == "wolf|eat"
    assert clause_proposition("the box in the room")[0] == ""
    assert clause_proposition("arun the book")[0] == ""


def test_which_prepositions_attach_two_ways_is_discovered_from_disagreement():
    learner = AttachLearner()
    assert not read_attachment("I saw the man with the telescope.", learner=learner).ambiguous

    learner.show("I cut the rope with the knife.", "event")
    learner.show("I opened the tin with the blade.", "event")
    learner.show("I met the woman with the scar.", "object")
    learner.show("I found the box with the lock.", "object")
    # Demonstrated one way only, so it stays unambiguous however often it is seen.
    learner.show("I walked to the shop.", "event")
    learner.show("I ran to the house.", "event")
    assert learner.ambiguous == {"with"}

    both = read_attachment("I saw the man with the telescope.", learner=learner)
    assert both.ambiguous
    assert [score for _text, score in both.interpretations] == [0.5, 0.5]
    assert not read_attachment("I walked to the shop.", learner=learner).ambiguous


def test_the_resolver_cue_weights_are_fitted_rather_than_tuned():
    """Recency is held constant in these cases, so a cue has to do the work or nothing does."""
    cases = [
        ([Referent("ravi", 1, "subject")], "he", "subject", "", "ravi"),
        ([Referent("sara", 1, "subject")], "she", "subject", "", "sara"),
        ([Referent("ravi", 1, "subject"), Referent("arun", 1, "object")],
         "he", "subject", "", ""),
        ([Referent("mira", 1, "subject"), Referent("devi", 1, "object")],
         "she", "subject", "", ""),
        ([Referent("ravi", 1, "subject"), Referent("arun", 1, "object")],
         "him", "object", "arun", "ravi"),
        ([Referent("ravi", 1, "subject", False, 2), Referent("arun", 1, "object")],
         "he", "subject", "", "ravi"),
    ]
    resolver = Reference()
    assert resolver.cues == {"parallel": 0.0, "topical": 0.0}
    before = resolver.score(cases)
    fitted = resolver.fit(cases)
    assert fitted["fitted"] == 1.0
    assert fitted["fitted"] > before
    # And what it finds is not what was hand-tuned here first: role parallelism earns nothing on
    # evidence that controls for recency.
    assert resolver.cues["topical"] > 0.0
    assert resolver.cues["parallel"] == 0.0


def test_resolutions_alone_leave_abstention_unconstrained():
    """The ambiguous cases are what rule out the settings that never abstain.

    Not that a resolution-only fit *must* stop abstaining — the search returns the smallest
    weights that fit, so it often will not. The claim is the weaker and true one: on resolutions
    alone, settings that destroy abstention fit the data perfectly and nothing excludes them. Add
    one case the discourse does not settle and they stop fitting.
    """
    resolving_only = [
        ([Referent("ravi", 1, "subject", False, 2), Referent("arun", 1, "object")],
         "he", "subject", "", "ravi"),
        ([Referent("mira", 1, "subject", False, 2), Referent("devi", 1, "object")],
         "she", "subject", "", "mira"),
    ]
    ambiguous = ([Referent("ravi", 1, "subject"), Referent("arun", 1, "object")],
                 "he", "subject", "", "")

    reckless = Reference()
    reckless.cues, reckless.margin = {"parallel": 0.6, "topical": 0.4}, 0.05
    assert reckless.score(resolving_only) == 1.0        # perfect, and it never abstains
    assert reckless.score(resolving_only + [ambiguous]) < 1.0

    balanced = Reference()
    balanced.fit(resolving_only + [ambiguous])
    assert balanced.score(resolving_only + [ambiguous]) == 1.0


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

def test_a_new_conversation_keeps_everything_learned_about_the_language():
    voice = Communicator()
    voice.show("Can you open the window?", "request")
    voice.show("Could you shut the gate?", "request")
    for first, second, verdict in VERDICTS:
        voice.show_change(first, second, verdict)
    voice.reference.cues = {"parallel": 0.1, "topical": 0.4}
    voice.hear("ravi met arun.")
    voice.reset()
    assert voice.reference.referents == []
    assert voice.ledger.claims == []
    assert voice.acts.read("Would you send the file?").intended == "request"
    assert voice.markers.role("now") == LICENCE
    assert voice.reference.cues == {"parallel": 0.1, "topical": 0.4}


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

    # The dispute needs the markers: without them nothing has shown her that `now` signals a
    # change, so the relation is never known to hold one value at a time and the reversion is
    # more information rather than a conflict. Taught, it is a conflict.
    disputed = NJPBrain()
    for first, second, verdict in VERDICTS:
        disputed.show_change(first, second, verdict)
    disputed.think("The key is in the drawer.")
    disputed.think("The key is now on the table.")
    disputed.think("The key is in the drawer.")
    assert "which of those holds" in disputed.think("Where is the key?").answer

    untaught = NJPBrain()
    untaught.think("The key is in the drawer.")
    untaught.think("The key is now on the table.")
    untaught.think("The key is in the drawer.")
    assert "which of those holds" not in untaught.think("Where is the key?").answer


def test_the_figurative_guard_can_be_corrected_by_evidence():
    """A guard that refuses to file its own counter-evidence can never be wrong.

    Ten birds witnessed for `need`, and the first sentence about a plant needing light looks
    exactly like a metaphor. It is not — it is news. Measured before this was fixed: 14 of 20
    ordinary plant sentences suppressed, and `nyxara.njp.discover` three organs away reporting
    every kind rule confirmed and none refuted, because it never saw a counterexample.
    """
    kinds = {name: ["bird"] for name in ("sparrow", "crow", "robin", "eagle")}
    kinds.update({name: ["plant"] for name in ("rose", "tulip", "fern", "daisy", "oak")})
    figure = Figure(kinds=lambda name: kinds.get(str(name).lower(), []))
    for bird in ("sparrow", "crow", "robin", "eagle"):
        figure.witness("need", bird)

    flagged = []
    for plant in ("rose", "tulip", "fern", "daisy", "oak"):
        judged = figure.judge(compile_meaning(f"{plant} need light"))
        figure.note_exception(judged)
        flagged.append(judged.figurative)

    # The first few are flagged; once the kind has turned up often enough it is ordinary.
    assert flagged[0] and flagged[1]
    assert not flagged[-1]
    assert figure.exceptions["need"]["plant"]


def test_a_metaphor_stays_a_metaphor_until_the_kind_is_established():
    kinds = {"wolf": ["animal"], "dog": ["animal"], "bear": ["animal"],
             "market": ["institution"], "fund": ["institution"]}
    figure = Figure(kinds=lambda name: kinds.get(str(name).lower(), []))
    for animal in ("wolf", "dog", "bear"):
        figure.witness("swallow", animal)

    first = figure.judge(compile_meaning("The market swallowed the shock."))
    figure.note_exception(first)
    assert first.figurative
    second = figure.judge(compile_meaning("The fund swallowed the loss."))
    figure.note_exception(second)
    assert second.figurative          # one other institution is not yet a pattern


# --------------------------------------------------------------------------- #
# 9 · language as prediction
# --------------------------------------------------------------------------- #

CYCLE = ("What is the {a}?", "The {a} is the {b}.", "Open the {b}.")


def _expose(voice, turns, mint=None):
    counter = [0]

    def word():
        counter[0] += 1
        return f"w{counter[0]}"

    for index in range(turns):
        voice.hear(CYCLE[index % 3].format(a=word(), b=word()))
    return voice


def test_an_organ_that_has_heard_nothing_predicts_nothing():
    """Silence is not a wrong prediction, and it must not be scored as one."""
    voice = Communicator()
    expected = voice.anticipation.expect()
    assert expected.empty
    got = voice.hear("The seal is in the vault.")
    assert got.surprise is not None
    assert got.surprise.predicted == 0
    assert got.surprise.error == 0.0


def test_what_follows_what_is_counted_rather_than_shipped():
    voice = _expose(Communicator(), 18)
    expected = voice.anticipation.expect()
    assert expected.act == "question"          # the cycle's next step, from counts alone
    assert expected.act_confidence == 1.0
    assert voice.anticipation.accuracy("act") > 0.9


def test_an_exchange_with_nothing_to_learn_is_not_committed_to():
    """Every act followed by two others equally often. The floor is what stops a guess."""
    voice = Communicator()
    for first, second in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
        for index in range(3):
            voice.hear(CYCLE[first].format(a=f"x{first}{second}{index}", b="y"))
            voice.hear(CYCLE[second].format(a=f"z{first}{second}{index}", b="q"))
    assert voice.anticipation.expect().act_confidence < voice.anticipation.floor


def test_a_prediction_is_graded_by_evidence_it_did_not_produce():
    """The gap-filler writes the predicted act onto the reading; scoring that would be circular.

    Measured with the loop closed: the control exchange's act distribution collapsed to a single
    successor and reported 1.00 confidence on a sequence built to be unpredictable.
    """
    voice = _expose(Communicator(), 18)
    last = voice.anticipation.last_act
    assert voice.anticipation.expect().act == "question"    # what it would have supplied
    before = {act: dict(counts) for act, counts in voice.anticipation.after_act.items()}
    # A sentence whose shape carries no convention, on a turn where the expectation is confident
    # and says something else.
    voice.hear("Open the gate.")
    after = voice.anticipation.after_act
    # `command` -- what the sentence said -- was recorded; `question`, what the expectation would
    # have supplied, was not.
    assert after[last]["command"] == before.get(last, {}).get("command", 0) + 1
    assert after[last].get("question", 0) == before.get(last, {}).get("question", 0)


def test_a_confident_expectation_fills_a_gap_and_never_overrides_a_convention():
    voice = _expose(Communicator(), 18)
    voice.show("Can you open the window?", "request")
    voice.show("Could you shut the gate?", "request")
    # A demonstrated convention wins over any habit, however strong the habit.
    read = voice.acts.read("Would you send the file?")
    assert read.intended == "request" and read.level == "tags"


def test_the_counts_survive_a_new_conversation_and_the_position_does_not():
    """What follows a question is a fact about the language; what was just said is not."""
    voice = _expose(Communicator(), 18)
    mid = voice.anticipation.expect().act
    voice.reset()
    assert voice.anticipation.last_act == ""
    assert voice.anticipation.after_act            # the counts are untouched
    # And she still knows how an exchange of this kind *opens*, which is real knowledge and a
    # different prediction from the one she was making mid-exchange.
    opening = voice.anticipation.expect()
    assert opening.act == "question"
    assert opening.act != mid or mid == "question"


# --------------------------------------------------------------------------- #
# 10 · the exchange, and the common ground
# --------------------------------------------------------------------------- #

def _exchange_taught():
    voice = Communicator()
    for a, b in (("seal", "key"), ("vault", "door")):
        voice.show_exchange(f"What is the {a}?", f"The {a} is the {b}.", "answer")
    for a, b in (("gate", "open"), ("lock", "shut")):
        voice.show_exchange(f"Open the {a}.", f"The {a} is {b}.", "accept")
    return voice


def test_a_reply_is_read_from_the_acts_and_not_from_either_sentence():
    voice = _exchange_taught()
    voice.hear("What is the drum?")
    got = voice.hear("The drum is the barrel.")
    assert got.reply is not None
    assert got.reply.relation == "answer" and got.reply.fits


def test_a_reply_she_has_no_pairing_for_is_a_non_sequitur_and_she_says_so():
    voice = _exchange_taught()
    voice.hear("What is the drum?")
    got = voice.hear("Open the shed.")
    assert not got.reply.fits
    assert "assertion" in got.reply.expected
    assert "never been shown following" in got.reply.why


def test_where_she_has_been_shown_nothing_she_objects_to_nothing():
    """A module that only detected non-sequiturs would object to every unfamiliar conversation."""
    voice = _exchange_taught()
    got = voice.exchange.read("exclamation", "assertion")
    assert got.fits and not got.relation
    assert "nothing has ever been shown following" in got.why


def test_a_pair_two_demonstrations_disagree_about_is_contested():
    voice = _exchange_taught()
    voice.show_exchange("What is the mast?", "The mast is the pole.", "correction")
    got = voice.exchange.read("question", "assertion")
    assert not got.relation and got.fits
    assert ("question", "assertion") in voice.exchange.contested


def test_untaught_nothing_is_a_reply_and_nothing_is_a_non_sequitur():
    voice = Communicator()
    voice.hear("What is the drum?")
    got = voice.hear("Open the shed.")
    assert got.reply is not None and got.reply.fits and not got.reply.relation


def test_what_the_hearer_already_has_is_left_unsaid():
    """`social.common_ground` has modelled given-versus-new since it was written, and nothing in
    this package had ever put a sentence into it."""
    voice = Communicator()
    meaning = compile_meaning("water boils today")
    meaning.condition, meaning.modality = "the pressure is normal", "typical"

    first = voice.say(meaning, "engineer")
    assert "condition" in first.carried and not first.omitted
    voice.hear("The kettle is on the stove.")      # the hearer carries on: it is grounded
    second = voice.say(meaning, "engineer")
    assert "condition" in second.omitted
    assert second.words < first.words
    assert second.claim == first.claim             # and the claim itself never goes


def test_a_hearer_who_has_nothing_is_told_everything():
    voice = Communicator()
    meaning = compile_meaning("water boils today")
    meaning.condition, meaning.modality = "the pressure is normal", "typical"
    said = voice.say(meaning, "engineer")
    assert not said.omitted


# --------------------------------------------------------------------------- #
# 11 · alternations — the same meaning, said a different way round
# --------------------------------------------------------------------------- #

def _alternations_taught():
    voice = Communicator()
    for who, what in (("ravi", "window"), ("sara", "gate"), ("devi", "wall")):
        voice.show_alternation(f"{who} opened the {what}.",
                               f"The {what} was opened by {who}.")
        voice.show_alternation(f"{who} opened the {what}.",
                               f"{who} has been opening the {what}.")
    for one, two, a, b in (("ravi", "arun", "door", "sill"), ("sara", "devi", "gate", "lock")):
        marked = f"{one} opened the {a} and {two} the {b}."
        voice.show_alternation(f"{one} opened the {a}.", marked)
        voice.show_alternation(f"{two} opened the {b}.", marked)
    return voice


def test_the_shapes_are_found_structurally_and_an_active_sentence_matches_none():
    from nyxara.njp.discourse import frames_of

    assert [f.name for f in frames_of("The window was opened by Ravi.")] == ["be-prep"]
    assert [f.name for f in frames_of("Ravi has been opening the door.")] == ["aux-chain"]
    assert [f.name for f in frames_of("Ravi opened the door and Arun the window.")] == ["gapped"]
    assert frames_of("Ravi opened the window.") == []
    assert frames_of("The key is in the drawer.") == []


def test_untaught_none_of_these_shapes_is_read():
    voice = Communicator()
    assert voice.alternation.read("The barrel was carried by Meera.") is None
    assert voice.alternation.read("Meera has been carrying the barrel.") is None


def test_the_passive_puts_the_agent_where_the_demonstrations_did():
    voice = _alternations_taught()
    assert voice.alternation.kept["be-prep"].roles == {
        "left": "object", "verb": "relation", "right": "subject"}
    got = voice.alternation.read("The barrel was carried by Meera.")
    assert got.subject == "meera" and got.object == "barrel"


def test_the_auxiliary_chain_does_not_move_anything():
    voice = _alternations_taught()
    assert voice.alternation.kept["aux-chain"].roles["left"] == "subject"
    got = voice.alternation.read("Meera has been carrying the barrel.")
    assert got.subject == "meera" and got.object == "barrel"


def test_a_gapped_coordinate_is_two_claims_and_not_one_bad_one():
    """It used to come back as a single claim whose object was `door arun window`."""
    voice = _alternations_taught()
    got = voice.alternation.readings("Kiran raised the mast and Nita the sail.")
    assert [(m.subject, m.object) for m in got] == [("kiran", "mast"), ("nita", "sail")]

    voice.hear("Kiran raised the mast and Nita the sail.")
    filed = [(c.subject, c.object) for c in voice.ledger.claims]
    assert filed == [("kiran", "mast"), ("nita", "sail")]


def test_a_copula_with_a_predicate_needs_no_mapping_at_all():
    """*"He was tired."* is one of the commonest sentences in English and had no reading.

    A named subject the compiler already handled — as `np-verb`, with the predicate lemmatised.
    A **pronoun** subject it did not: that frame needs a noun phrase, so the sentence came back
    unreadable and nothing about it reached the ledger. The structural shape covers it, and
    covers it as a fallback, so the compiler's reading is untouched where there is one.
    """
    named = read_claim("Ravi was tired.")
    assert named is not None and named.subject == "ravi"

    pronominal = read_claim("He was tired.")
    assert pronominal is not None
    assert pronominal.subject == "he" and pronominal.relation == "tired"
    assert proposition("he was tired") == ("he|tired", "")


def test_a_structural_shape_is_a_fallback_and_never_an_override():
    """Overriding read *"When I visited Delhi last year I was tired."* as a claim about `tired`,
    and lost the claim the speaker actually made inside the fronted clause."""
    claim = read_claim("When I visited Delhi last year I was tired.")
    assert claim is not None
    assert claim.relation == "visit" and claim.object == "delhi"


def test_a_claim_about_an_unresolved_pronoun_is_not_filed():
    voice = Communicator()
    voice.hear("ravi met arun.")
    voice.hear("He was tired.")
    assert all(not _is_pronoun(claim.subject) for claim in voice.ledger.claims)


def test_a_resolved_first_person_is_named_rather_than_dropped():
    """Dropping them cost every claim the Master made about himself."""
    voice = Communicator()
    got = voice.hear("I never visited Delhi.")
    assert got.verdict is not None and got.verdict.claim is not None
    assert got.verdict.claim.subject == "master"


def _is_pronoun(word):
    from nyxara.njp.discourse import _pronoun

    return bool(_pronoun(word))


# --------------------------------------------------------------------------- #
# 12 · the semantic anchor — cause, purpose, and whether it holds or happens
# --------------------------------------------------------------------------- #

def _causes_taught():
    voice = Communicator()
    for who, what, why in (("ravi", "door", "wind"), ("sara", "gate", "storm")):
        voice.show_cause(f"{who} opened the {what} because the {why} rose.",
                         cause=f"the {why} rose")
        voice.show_cause(f"The {why} rose so {who} opened the {what}.",
                         cause=f"the {why} rose")
    for who, where, what in (("ravi", "shop", "bread"), ("sara", "field", "grass")):
        voice.show_cause(f"{who} went to the {where} to carry {what}.",
                         goal=f"{who} carry {what}")
    return voice


def test_untaught_no_sentence_carries_a_cause_or_a_purpose():
    voice = Communicator()
    assert voice.connective.read("Devi lit the lamp because the room was dark.") is None
    assert voice.connective.read("Meera went to the well to draw water.") is None


def test_which_side_the_cause_is_on_is_induced_per_connective():
    """Nothing structural separates *"A because B"* from *"B so A"*."""
    voice = _causes_taught()
    assert voice.connective.kept["because"] == "cause-after"
    assert voice.connective.kept["so"] == "cause-before"

    forward = voice.connective.read("Devi lit the lamp because the room was dark.")
    backward = voice.connective.read("The room was dark so devi lit the lamp.")
    assert forward.cause == backward.cause == "the room was dark"
    assert forward.effect == backward.effect == "devi lit the lamp"


def test_a_determiner_is_not_a_connective():
    """`the` splits the sentence into two halves that both read, sits earlier than `because`,
    and gave a cause of `lamp because the room was dark`."""
    from nyxara.njp.discourse import _splits

    tokens = [token for _left, token, _right
              in _splits("devi lit the lamp because the room was dark.")]
    assert tokens == ["because"]


def test_the_best_supported_connective_wins_rather_than_the_leftmost():
    voice = _causes_taught()
    assert voice.connective.support["because"] >= voice.connective.min_support
    got = voice.connective.read("Kiran shut the vault because the alarm rang.")
    assert got.connective == "because"


def test_a_purpose_clause_is_told_from_a_prepositional_phrase():
    voice = _causes_taught()
    got = voice.connective.read("Meera went to the well to draw water.")
    assert got.kind == "goal" and got.goal == "meera draw water"
    # `to the well` is a phrase, not a purpose: a determiner sits between.
    assert "well" not in got.goal


def test_the_sentence_asserts_its_effect_and_not_the_whole_of_itself():
    """Reading the whole sentence gave an agent of `devi lit the lamp because the room`."""
    voice = _causes_taught()
    got = voice.hear("Devi lit the lamp because the room was dark.")
    assert got.meaning.subject == "devi"
    assert got.anchor()["cause"] == "the room was dark"


def test_whether_a_relation_holds_or_happens_is_read_off_how_it_behaved():
    voice = Communicator()
    for first, second, verdict in VERDICTS:
        voice.show_change(first, second, verdict)
    voice.hear("The key is in the drawer.")
    voice.hear("The key is now on the table.")
    voice.hear("Ravi opened the door.")
    voice.hear("Ravi opened the window.")
    assert voice.ledger.kind_of("is_at") == "state"
    assert voice.ledger.kind_of("open") == "event"
    assert voice.ledger.kind_of("fly") == ""


def test_occurrence_is_not_claimed_before_she_knows_what_a_marked_change_looks_like():
    """Several live values look identical for an unmarked state and a repeated event."""
    voice = Communicator()
    voice.hear("The key is in the drawer.")
    voice.hear("The key is now on the table.")
    assert voice.ledger.kind_of("is_at") == ""


def test_the_anchor_reports_only_the_slots_the_turn_actually_filled():
    """A schema that always returns eleven keys cannot be told from one that fills none."""
    voice = Communicator()
    got = voice.hear("Ravi opened the window.")
    filled = got.anchor(voice.ledger)
    assert filled["agent"] == "ravi" and filled["relation"] == "open"
    assert "cause" not in filled and "goal" not in filled and "occurrence" not in filled


# --------------------------------------------------------------------------- #
# 13 · the closed class, found rather than shipped
# --------------------------------------------------------------------------- #

SHAPES = ("the {a} {v} the {b}", "a {a} {v} a {b}", "the {a} did not {v} the {b}",
          "is the {a} in the {b}", "what {v} the {a}", "can the {a} {v} the {b}")
KNOWN_CLOSED = ("the", "a", "did", "not", "is", "in", "what", "can")


def _english(learner, rng, sentences=4000):
    nouns = [f"n{i}" for i in range(300)]
    verbs = [f"v{i}" for i in range(120)]
    for _ in range(sentences):
        learner.hear(rng.choice(SHAPES).format(a=rng.choice(nouns), b=rng.choice(nouns),
                                               v=rng.choice(verbs)), language="en")
    return learner


def _overheard(learner, rng, name="x", sentences=4000):
    from nyxara.njp.dialects import mint_dialect

    tongue = mint_dialect(rng, name)
    nouns = [f"q{i}" for i in range(300)]
    verbs = [f"z{i}" for i in range(120)]
    shapes = ("{a} {v} {b}", "{p} {a} {v} {b}", "{a} {n} {v} {b}",
              "{w} {v} {b}", "{a} {v} {ws}")
    for _ in range(sentences):
        learner.hear(rng.choice(shapes).format(
            a=rng.choice(nouns), b=rng.choice(nouns), v=rng.choice(verbs),
            p=tongue.polar, n=tongue.negator, w=tongue.wh_object,
            ws=tongue.wh_subject), language=name)
    return {tongue.negator, tongue.polar, tongue.wh_object, tongue.wh_subject}


def test_unfitted_no_word_is_called_closed():
    import random as _random

    from nyxara.njp.discourse import ClosedClassLearner

    learner = _english(ClosedClassLearner(), _random.Random(3))
    assert learner.closed("en") == set()


def test_the_criterion_is_fitted_where_the_answer_is_known():
    import random as _random

    from nyxara.njp.discourse import ClosedClassLearner

    learner = _english(ClosedClassLearner(), _random.Random(3))
    fitted = learner.fit(KNOWN_CLOSED, language="en")
    assert fitted["cut"] is not None and fitted["f1"] > 0.8


def test_it_recovers_the_closed_class_of_a_language_she_has_only_overheard():
    import random as _random

    from nyxara.njp.discourse import ClosedClassLearner

    rng = _random.Random(3)
    learner = _english(ClosedClassLearner(), rng)
    learner.fit(KNOWN_CLOSED, language="en")
    truth = _overheard(learner, rng, "x")
    found = learner.closed("x")
    assert found == truth          # every closed form, and nothing else


def test_breadth_and_frequency_are_both_load_bearing():
    """Breadth alone measures *small* class, not *closed* class; frequency alone is worse."""
    import random as _random

    from nyxara.njp.discourse import ClosedClassLearner

    rng = _random.Random(3)
    learner = _english(ClosedClassLearner(), rng)
    truth = _overheard(learner, rng, "x")
    learner.fit(KNOWN_CLOSED, language="en")
    cut = learner.cut
    by_breadth = {w for w, v in learner.breadth("x").items() if v >= cut}
    by_signature = {w for w, v in learner.signature("x").items() if v >= cut}
    # Both find every closed form. Only one of them finds *only* those.
    assert truth <= by_breadth and truth <= by_signature
    assert len(by_breadth - truth) > 50
    assert by_signature - truth == set()


def test_a_retelling_into_a_language_she_was_never_taught_carries_nothing():
    from nyxara.njp.language import LanguageFaculty

    voice = Communicator()
    got = voice.retell("Can you open the window?", into="nowhere", faculty=LanguageFaculty())
    assert not got.ok
    assert "claim" in got.lost


# --------------------------------------------------------------------------- #
# 14 · the last four tiers
# --------------------------------------------------------------------------- #

def test_an_inserted_adverb_does_not_change_the_convention():
    """Measured before the `bare` level: five demonstrations of the plain wording, and
    *"Can you really open the window?"* read as an ability question."""
    acts = ActLearner()
    for verb, thing in (("open", "window"), ("shut", "gate"), ("raise", "mast")):
        acts.show(f"Can you {verb} the {thing}?", "request")
    assert acts.read("Can you carry the barrel?").intended == "request"
    assert acts.read("Can you really carry the barrel?").intended == "request"
    assert acts.read("Can you carefully carry the barrel?").intended == "request"


def test_the_bare_level_sits_between_the_tags_and_the_frame():
    from nyxara.njp.discourse import LEVELS as ORDER

    assert ORDER == ("form", "tags", "bare", "frame", "opening")
    shapes = dict(zip(ORDER, shapes_of("Can you really open the window?")))
    assert "ADV" in shapes["tags"] and "ADV" not in shapes["bare"]


def test_a_frame_stops_at_a_subordinator():
    """The agent slot swallowed the whole causal clause: `ravi because the wind rose`."""
    from nyxara.njp.discourse import frames_of

    found = frames_of("The gate was opened by ravi because the wind rose.")
    assert found[0].slots["right"] == "ravi"


def test_a_convention_taught_in_one_field_reads_in_another():
    acts = ActLearner()
    for word in ("kettle", "ladle", "pantry"):
        acts.show(f"Can you scour the {word}?", "request")
    for word in ("crucible", "bellows", "anvil"):
        assert acts.read(f"Can you temper the {word}?").intended == "request"


def test_an_unseen_combination_is_read_in_part_and_refused_in_part():
    voice = Communicator()
    for who, what in (("ravi", "window"), ("sara", "gate"), ("devi", "wall")):
        voice.show_alternation(f"{who} opened the {what}.",
                               f"The {what} was opened by {who}.")
    said = "The lock was opened by Meera because the alarm rang."
    assert voice.alternation.read(said).subject == "meera"     # the half she was taught
    assert voice.connective.read(said) is None                 # and the half she was not


def test_she_can_say_what_a_turn_will_take_and_when_it_takes_nothing():
    voice = Communicator()
    for who, what in (("ravi", "window"), ("sara", "gate"), ("devi", "wall")):
        voice.show_alternation(f"{who} opened the {what}.",
                               f"The {what} was opened by {who}.")
    assert voice.strategy("The barrel was opened by Kiran.") == ("alternation", "ledger")
    assert voice.strategy("He carried the barrel.") == ("reference", "ledger")
    assert voice.strategy("What is the barrel?") == ()
