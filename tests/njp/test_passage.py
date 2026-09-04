"""What a passage reader must do, and the ways this one was wrong before it did them."""

from __future__ import annotations

import pytest

from nyxara.njp.passage import (LESSONS, Demonstration, KnowledgeObject, PassageReader,
                                Shape, taught_reader)
from nyxara.njp.passageschool import HELD_OUT, SEALED, examine, grounder_baseline


# --------------------------------------------------------------------------------------------- #
#  the representation
# --------------------------------------------------------------------------------------------- #
def test_the_passage_survives_being_read():
    text = ("Fermentation is the reaction by which yeast releases energy from sugar. "
            "Fermentation requires anaerobic conditions.")
    obj = taught_reader().read(text, concept="fermentation")
    assert obj.text == text
    assert len(obj.sentences) == 2
    assert obj.relations


def test_a_definition_and_a_kind_are_not_the_same_slot():
    """The defect the flat form cannot express: `is_a=general rise in prices` is both at once."""
    obj = taught_reader().read("Inflation is a general rise in prices.", concept="inflation")
    assert obj.definition == "general rise in prices"
    assert obj.kind == "rise"
    assert obj.definition != obj.kind


def test_every_relation_says_which_sentence_and_which_shape_produced_it():
    obj = taught_reader().read(
        "Welding is the process by which heat joins two metals. Welding requires a filler rod.",
        concept="welding")
    assert obj.relations
    for relation in obj.relations:
        assert relation.sentence in obj.sentences
        assert relation.shape


def test_reading_files_nothing():
    """Extraction is a reading; filing is a decision. A module that did both hides the decision."""
    reader = taught_reader()
    before = dict(reader.shapes)
    reader.read("Corrosion is the decay of metal in air.", concept="corrosion")
    assert set(reader.shapes) == set(before)


# --------------------------------------------------------------------------------------------- #
#  the induction
# --------------------------------------------------------------------------------------------- #
def test_an_untaught_reader_reads_nothing():
    obj = PassageReader().read("Erosion is the process by which wind removes soil.",
                               concept="erosion")
    assert obj.relations == ()
    assert obj.definition == ""


def test_a_shape_of_nothing_but_holes_is_refused():
    """A pattern with no literal anchor matches every clause in the language."""
    reader = taught_reader()
    for shape in reader.shapes.values():
        if shape.level == "cued":
            assert shape.anchors >= 1


def test_the_word_that_names_the_relation_survives_abstraction():
    """`requires` is what makes the clause a requirement; holing it reads anything as anything."""
    reader = taught_reader()
    kept = [s for s in reader.shapes.values()
            if s.level == "cued" and s.predicate == "requires" and "requires" in s.left + s.right]
    assert kept, "every cued `requires` shape lost the word that names the relation"


def test_a_word_the_teacher_pointed_at_as_an_object_is_not_a_relation_word():
    """`plants` was demonstrated as an `occurs_in` filler. It cannot also name a relation."""
    reader = taught_reader()
    assert reader._relation_words.get("plants") is None
    assert reader._relation_words.get("requires") == "requires"


def test_a_shape_a_lesson_did_not_contain_is_what_reads_an_unseen_subject():
    """Literal frames alone score zero on held-out passages; this is the falsification."""
    frames = examine(taught_reader(use_cued=False), HELD_OUT)
    assert frames.papers["relations"].score == 0.0
    assert frames.papers["restraint"].score == 1.0     # it is silent, never wrong


def test_reading_an_unseen_passage_is_recorded_as_generalisation():
    reader = taught_reader()
    reader.read("Subduction is the movement by which one plate sinks beneath another.",
                concept="subduction")
    assert reader.generalised(), "nothing recorded having read outside its lessons"


# --------------------------------------------------------------------------------------------- #
#  the three mechanisms, each removable
# --------------------------------------------------------------------------------------------- #
def test_the_topic_supplies_the_subject_so_a_coordination_needs_none():
    """The sentence the grounder returned `photosynthesis occurs in ... and` as a subject for."""
    obj = taught_reader().read(
        "Photosynthesis occurs in the chloroplasts of plant cells and requires chlorophyll, "
        "water and carbon dioxide.", concept="photosynthesis")
    assert {r.subject for r in obj.relations} == {"photosynthesis"}
    assert ("requires", "chlorophyll") in {(r.predicate, r.object) for r in obj.relations}


def test_a_coordinated_span_of_bare_nouns_is_a_list():
    obj = taught_reader().read("Corrosion requires moisture, oxygen and time.",
                               concept="corrosion")
    required = {r.object for r in obj.relations if r.predicate == "requires"}
    assert required == {"moisture", "oxygen", "time"}


def test_a_coordination_inside_a_clause_is_not_a_list():
    """Splitting `... into hydrogen and oxygen` produced two definitions, neither of them true."""
    obj = taught_reader().read(
        "Electrolysis is the process by which current splits water into hydrogen and oxygen.",
        concept="electrolysis")
    definitions = [r.object for r in obj.relations if r.predicate == "definition"]
    assert definitions == ["process by which current splits water into hydrogen and oxygen"]


def test_a_reader_shown_no_list_does_not_invent_one():
    reader = PassageReader()
    reader.teach(Demonstration(name="one", concept="rusting",
                               text="Rusting is the decay of iron. Rusting requires water.",
                               expect={"definition": ("decay of iron",),
                                       "requires": ("water",)}))
    obj = reader.read("Rusting requires oxygen, salt and time.", concept="rusting")
    required = {r.object for r in obj.relations if r.predicate == "requires"}
    assert required and required != {"oxygen", "salt", "time"}


def test_a_pronoun_subject_is_resolved_against_the_passage():
    obj = taught_reader().read(
        "Photovoltaics is the generation of electricity from light. "
        "Silicon is the material that most cells use. "
        "It requires very high purity.", concept="photovoltaics")
    purity = [r for r in obj.relations if r.object == "very high purity"]
    assert purity and purity[0].subject == "silicon"


def test_the_resolver_is_fitted_from_the_lessons_and_not_from_nothing():
    """An unfitted resolver scores role parallelism at zero and abstains on every passage."""
    reader = taught_reader()
    assert reader.stats()["reference_cases"] >= 3
    assert reader._cues.get("parallel", 0.0) > 0.0


def test_turning_the_resolver_off_costs_a_measured_score():
    """A mechanism no measurement can miss has not been shown to be doing anything."""
    with_it = examine(taught_reader(), HELD_OUT).papers["pronoun"].score
    without = examine(taught_reader(resolve_pronouns=False), HELD_OUT).papers["pronoun"].score
    assert with_it > without


# --------------------------------------------------------------------------------------------- #
#  the boundaries of a filler
# --------------------------------------------------------------------------------------------- #
def test_a_filler_stops_where_another_relation_begins():
    """`requires ... and produces ...` came back as `requires = produces fall in real wages`."""
    obj = taught_reader().read(
        "Inflation requires an expanding money supply and produces a fall in real wages.",
        concept="inflation")
    required = {r.object for r in obj.relations if r.predicate == "requires"}
    assert all("produces" not in o for o in required)


def test_a_filler_stops_where_a_verb_begins():
    obj = taught_reader().read("Viscosity requires a temperature to be meaningful.",
                               concept="viscosity")
    required = {r.object for r in obj.relations if r.predicate == "requires"}
    assert required == {"temperature"}


def test_a_definition_longer_than_any_lesson_is_still_read_whole():
    """The learned cap bounds an unanchored scan; the sentence already bounds it at the end."""
    obj = taught_reader().read(
        "Electrolysis is the process by which current splits water into hydrogen and oxygen.",
        concept="electrolysis")
    assert obj.definition == "process by which current splits water into hydrogen and oxygen"


def test_a_condition_is_kept_apart_from_the_claim():
    obj = taught_reader().read(
        "Precipitation requires condensation nuclei if the air is very clean.",
        concept="precipitation")
    # Determiner-free, which is the form the whole reader works in.
    assert obj.conditions == ("air is very clean",)
    assert any(r.predicate == "requires" and r.object == "condensation nuclei"
               for r in obj.relations)


# --------------------------------------------------------------------------------------------- #
#  restraint
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "The Treaty of Westphalia was signed in 1648. It ended a long European war.",
    "The Danube flows through ten countries. It reaches the Black Sea.",
])
def test_a_passage_stating_nothing_she_was_taught_produces_nothing(text):
    assert taught_reader().read(text).relations == ()


# --------------------------------------------------------------------------------------------- #
#  the school
# --------------------------------------------------------------------------------------------- #
def test_the_reader_beats_the_sentence_parser_on_the_same_passages():
    reader = examine(taught_reader(), HELD_OUT)
    grounder = grounder_baseline(HELD_OUT)
    assert reader.recall > grounder.recall
    assert reader.precision > grounder.precision
    assert reader.papers["kind"].score > grounder.papers["kind"].score


def test_the_sealed_set_scores_like_the_one_the_fixes_were_made_against():
    """If the shapes were fitted to the first exam, the second would fall away."""
    first = examine(taught_reader(), HELD_OUT)
    sealed = examine(taught_reader(), SEALED)
    assert sealed.recall > 0.6
    assert abs(sealed.mean - first.mean) < 0.10


def test_nothing_is_confabulated_on_the_sealed_set():
    assert examine(taught_reader(), SEALED).invented == ()


def test_the_gaps_are_reported_rather_than_marked_wrong():
    """Relations stated in predicates nobody demonstrated are coverage, not failures."""
    report = examine(taught_reader(), HELD_OUT)
    assert report.gaps
    assert report.papers["restraint"].score == 1.0


def test_precision_cannot_exceed_one():
    """It printed 1.156 once, by summing papers that mark the same item twice."""
    for report in (examine(taught_reader(), HELD_OUT), examine(taught_reader(), SEALED)):
        assert 0.0 <= report.precision <= 1.0
        assert 0.0 <= report.recall <= 1.0


# --------------------------------------------------------------------------------------------- #
#  the brain
# --------------------------------------------------------------------------------------------- #
def test_the_brain_reads_a_passage_without_filing_it():
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    obj = brain.read_passage("Tempering is the treatment by which reheating softens steel.",
                             concept="tempering")
    assert isinstance(obj, KnowledgeObject)
    assert obj.kind == "treatment"
    assert not brain.think("what is tempering?").answer


def test_what_she_learns_from_prose_is_answerable_in_english():
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    filed = brain.learn_passage(
        "Dialysis is the treatment by which a machine removes waste from blood. "
        "Dialysis requires a semipermeable membrane.", concept="dialysis")
    assert filed["filed"]
    assert "semipermeable membrane" in brain.think("what does dialysis require?").answer
    assert "treatment" in brain.think("what does dialysis mean?").answer


def test_a_predicate_the_reader_produces_is_reachable_by_a_question():
    """`occurs_in` appeared nowhere in grounding.py: stored, and structurally unreachable."""
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    brain.learn_passage("Fermentation is the reaction by which yeast releases energy. "
                        "Fermentation occurs in the cytoplasm.", concept="fermentation")
    assert brain.think("where does fermentation occur?").answer
