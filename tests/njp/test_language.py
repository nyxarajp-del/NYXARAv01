"""The language faculty: what a lesson leaves, and what it is not allowed to leave.

Most of these are about the **refusals**, and that is deliberate. A grammar that reads more
sentences is trivially easy to build and worthless, because the failure mode of an extractor is
never silence — it is a confident wrong reading, which is indistinguishable from a right one
downstream. So what is asserted here is: a shape needs two demonstrations that disagree; a
construction that contradicts its own lesson is dropped; an affix nobody demonstrated means
nothing; a sentence two shapes read differently comes back unreadable; a sentence she cannot read
back she does not say; and none of it fires at all on a faculty nobody has taught.

The three that are about capability are the ones a table lookup cannot pass: the wug test on a
stem that has never been uttered, reading a clause in an order nobody wrote code for, and saying
in one minted language what was read in another.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.dialects import mint_dialect, sample, stem
from nyxara.njp.language import (
    Construction,
    Grammar,
    LanguageFaculty,
    Lexicon,
    Morphology,
    Slot,
    tokenize_surface,
)
from nyxara.njp.semantics import Meaning, compile_meaning


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def words(rng: random.Random):
    """A generator of fresh nonsense words that cannot repeat itself."""
    counter = [0]

    def draw() -> str:
        counter[0] += 1
        number, tag = counter[0], ""
        while number:
            number, remainder = divmod(number - 1, 26)
            tag = chr(ord("a") + remainder) + tag
        body = "".join(rng.choice("bdgklmnprstvz") + rng.choice("aeiou") for _ in range(2))
        return body + tag

    return draw


def taught(seed: int = 7, count: int = 30,
           kinds=("assertion", "negated", "past", "polar", "content")):
    """A faculty that has been shown ``count`` sentences of one minted language."""
    rng = random.Random(seed)
    dialect = mint_dialect(rng, f"tongue-{seed}")
    draw = words(rng)
    faculty = LanguageFaculty()
    for utterance in sample(dialect, draw, count, kinds=kinds):
        faculty.show(utterance.surface, utterance.meaning, tongue=dialect.name)
    faculty.learn(tongue=dialect.name)
    return faculty, dialect, draw


def same(got: Meaning, want: Meaning) -> bool:
    return (got.kind == want.kind and bool(got.negated) == bool(want.negated)
            and got.focus == want.focus and got.temporal == want.temporal
            and got.subject == want.subject and got.object == want.object
            and got.relation == want.relation)


# --------------------------------------------------------------------------- #
# morphology
# --------------------------------------------------------------------------- #

def test_an_affix_is_induced_only_where_stems_carry_it():
    """Three stems that appear bare and extended are a morpheme; one is a spelling."""
    morphology = Morphology(min_stems=3)
    for base in ("zorb", "plag", "glim", "vunt"):
        morphology.observe((base, base + "ik"))
    morphology.observe(("kesh", "keshoz"))          # one stem only
    forms = {(affix.form, affix.side) for affix in morphology.induce()}
    assert ("ik", "suffix") in forms
    assert ("oz", "suffix") not in forms


def test_the_wug_test_on_a_stem_nobody_has_inflected():
    """The oldest experiment in the subject: a rule applies to a word that was not in the lesson."""
    morphology = Morphology()
    for base in ("zorb", "plag", "glim", "vunt", "kesh", "dral"):
        morphology.observe((base, base + "ik", base + "os"))
    morphology.induce()
    assert morphology.bind("zorb", "zorbik", "plural")
    assert morphology.bind("plag", "plagos", "past")

    assert "wug" not in morphology.vocabulary
    assert morphology.inflect("wug", "plural") == "wugik"
    assert morphology.inflect("wug", "past") == "wugos"
    # Recognition is the other direction and a separate ability.
    assert morphology.analyse("blimik").feature == "plural"
    assert morphology.analyse("blimik").stem == "blim"


def test_a_feature_nobody_demonstrated_produces_silence():
    morphology = Morphology()
    for base in ("zorb", "plag", "glim", "vunt"):
        morphology.observe((base, base + "ik"))
    morphology.induce()
    morphology.bind("zorb", "zorbik", "plural")
    assert morphology.inflect("wug", "future") == ""
    assert morphology.inflect("wug", "past") == ""


def test_binding_refuses_an_affix_that_was_never_induced():
    """Otherwise the wug test is one memorised pair with a rule's name on it."""
    morphology = Morphology(min_stems=3)
    morphology.observe(("zorb", "zorbik"))
    morphology.induce()
    assert morphology.bind("zorb", "zorbik", "plural") is False
    assert morphology.refused
    # Kept as an irregular, which is what a pair with no rule behind it is — and it does not
    # generalise to a stem it was never shown on.
    assert morphology.inflect("zorb", "plural") == "zorbik"
    assert morphology.inflect("wug", "plural") == ""


def test_an_unbound_affix_does_not_strip_a_word():
    """Merging two entities on the strength of a spelling she cannot name is worse than a miss."""
    morphology = Morphology()
    for base in ("zorb", "plag", "glim", "vunt"):
        morphology.observe((base, base + "ik"))
    morphology.induce()
    assert morphology.analyse("zorbik").affix == "ik"
    assert morphology.stem_of("zorbik") == "zorbik"      # induced, unbound: left alone
    morphology.bind("zorb", "zorbik", "plural")
    assert morphology.stem_of("zorbik") == "zorb"        # bound: now it means something


# --------------------------------------------------------------------------- #
# word classes
# --------------------------------------------------------------------------- #

def test_a_word_she_has_not_met_gets_silence_not_a_no():
    lexicon = Lexicon()
    for _ in range(3):
        lexicon.observe(("the", "cat", "runs"))
        lexicon.observe(("the", "dog", "runs"))
    lexicon.induce()
    assert lexicon.same_class("cat", "dog") is True
    assert lexicon.same_class("cat", "wug") is None


def test_classes_are_found_over_neighbour_kinds_not_neighbour_words():
    """The second pass is the difference between working and not working in a verb-first clause.

    Verb-subject-object puts the subject where it touches neither edge of the sentence, so two
    subjects sharing no verb and no object share no context at all. One further pass, describing
    each neighbour by the class the first pass put it in, puts them together.
    """
    # Each subject with its own verb and its own object, so two subjects overlap in nothing.
    # Twice each, because a word met once is one context and Lexicon refuses to classify it.
    sentences = [("v1", "s1", "o1"), ("v2", "s2", "o2"), ("v3", "s3", "o3")] * 2
    flat = Lexicon(passes=1)
    deep = Lexicon(passes=2)
    for lexicon in (flat, deep):
        for sentence in sentences:
            lexicon.observe(sentence)
        lexicon.induce()
    assert flat.same_class("s1", "s2") is False
    assert deep.same_class("s1", "s2") is True
    assert deep.same_class("s1", "v1") is False


# --------------------------------------------------------------------------- #
# constructions — what a lesson leaves
# --------------------------------------------------------------------------- #

def test_one_demonstration_is_a_sentence_not_a_shape():
    grammar = Grammar("t")
    grammar.show("cat chase dog", Meaning(kind="assertion", subject="cat",
                                          relation="chase", object="dog"))
    report = grammar.learn()
    assert report.kept == 0
    assert report.rejected == 1
    assert not grammar.read("bird chase fish").readable


def test_two_demonstrations_with_the_same_fillers_are_one_sentence_twice():
    grammar = Grammar("t")
    for _ in range(3):
        grammar.show("cat chase dog", Meaning(kind="assertion", subject="cat",
                                              relation="chase", object="dog"))
    report = grammar.learn()
    assert report.kept == 0
    assert any("identical fillers" in reason for reason in report.reasons)


def test_a_shape_generalises_to_words_that_were_never_in_a_lesson():
    grammar = Grammar("t")
    for subject, verb, obj in (("cat", "chase", "dog"), ("bird", "chase", "fish"),
                               ("frog", "eat", "worm")):
        grammar.show(f"{obj}ni {subject}ta {verb}",
                     Meaning(kind="assertion", subject=subject, relation=verb, object=obj))
    assert grammar.learn().kept == 1
    got = grammar.read("stoneni horseta push")
    assert (got.subject, got.relation, got.object) == ("horse", "push", "stone")


def test_a_demonstration_whose_meaning_is_not_in_its_surface_is_refused():
    """A mislabelled lesson mints a shape that reads every sentence of that form wrong."""
    grammar = Grammar("t")
    grammar.show("cat chase dog", Meaning(kind="assertion", subject="pelican",
                                          relation="chase", object="dog"))
    grammar.show("bird chase fish", Meaning(kind="assertion", subject="pelican",
                                            relation="chase", object="fish"))
    report = grammar.learn()
    assert report.kept == 0
    assert any("not in its surface" in reason for reason in report.reasons)


def test_a_construction_that_misreads_its_own_lesson_is_dropped():
    """A grammar that contradicts the sentences it was built from is not a grammar."""
    grammar = Grammar("t")
    shown = [("cat", "chase", "dog"), ("bird", "chase", "fish"), ("frog", "eat", "worm")]
    for subject, verb, obj in shown:
        grammar.show(f"{subject} {verb} {obj}",
                     Meaning(kind="assertion", subject=subject, relation=verb, object=obj))
    # The same surface shape, labelled the other way round. One of the two has to go, and which
    # one goes is decided by re-reading the corpus rather than by which arrived first.
    for subject, verb, obj in shown:
        grammar.show(f"{obj} {verb} {subject}",
                     Meaning(kind="assertion", subject=subject, relation=verb, object=obj))
    grammar.learn()
    for shape in grammar.constructions:
        for demonstration in grammar.demonstrations:
            got = grammar.read(demonstration.surface)
            if got.readable:
                assert same(got, demonstration.meaning), shape.to_dict()


def test_a_sentence_two_shapes_read_differently_comes_back_unreadable():
    grammar = Grammar("t")
    for subject, verb, obj in (("cat", "chase", "dog"), ("bird", "chase", "fish"),
                               ("frog", "eat", "worm")):
        grammar.show(f"{subject} {verb} {obj}",
                     Meaning(kind="assertion", subject=subject, relation=verb, object=obj))
        grammar.show(f"{subject} {verb} {obj} tak",
                     Meaning(kind="assertion", subject=subject, relation=verb, object=obj,
                             negated=True))
    grammar.learn()
    # Hand-built rival: the identical shape, read the other way round. Equal evidence, different
    # meaning — which is a refusal, not a coin toss.
    original = next(c for c in grammar.constructions if not c.negated)
    rival = Construction(tongue="t", pattern=(Slot(role="object"), Slot(role="verb"),
                                              Slot(role="subject")),
                         kind="assertion", support=original.support)
    grammar.constructions.append(rival)
    got = grammar.read("horse push stone")
    assert not got.readable
    assert got.frame == "ambiguous"


def test_a_sentence_with_one_word_too_many_is_refused():
    """The property MAX_SPAN is set to one to keep. See its docstring for the measurement."""
    faculty, dialect, draw = taught()
    utterance = sample(dialect, draw, 1)[0]
    assert faculty.read(utterance.surface, tongue=dialect.name).readable
    assert not faculty.read(f"{utterance.surface} {draw()}", tongue=dialect.name).readable


# --------------------------------------------------------------------------- #
# saying
# --------------------------------------------------------------------------- #

def test_she_says_only_what_she_can_read_back():
    faculty, dialect, draw = taught()
    utterance = sample(dialect, draw, 1, kinds=("negated",))[0]
    said = faculty.say(utterance.meaning, tongue=dialect.name)
    assert said == utterance.surface
    assert same(faculty.read(said, tongue=dialect.name), utterance.meaning)


def test_a_meaning_she_has_no_shape_for_is_met_with_silence():
    faculty, dialect, draw = taught()
    utterance = sample(dialect, draw, 1)[0]
    wanted = utterance.meaning
    wanted.modality = "necessary"                     # never demonstrated in any lesson
    assert faculty.say(wanted, tongue=dialect.name) == ""


def test_a_faculty_nobody_taught_reads_nothing_and_says_nothing():
    """Day one, and neither half of it is a failure."""
    faculty = LanguageFaculty()
    assert not faculty.read("the cat chases the dog").readable
    assert faculty.say(Meaning(kind="assertion", subject="cat", relation="chase",
                               object="dog")) == ""
    assert faculty.inflect("wug", "plural") == ""
    assert faculty.same_class("cat", "dog") is None


# --------------------------------------------------------------------------- #
# more than one language at once
# --------------------------------------------------------------------------- #

def test_the_same_meaning_said_in_a_language_that_shares_no_word():
    faculty, source, draw = taught(seed=11)
    target = mint_dialect(random.Random(99), "other")
    for utterance in sample(target, draw, 24, kinds=("assertion", "negated")):
        faculty.show(utterance.surface, utterance.meaning, tongue=target.name)
    faculty.learn(tongue=target.name)

    utterance = sample(source, draw, 1)[0]
    want = target.express(utterance.meaning.subject, utterance.meaning.relation,
                          utterance.meaning.object)
    assert faculty.translate(utterance.surface, into=target.name, frm=source.name) == want


def test_an_unreadable_sentence_translates_to_nothing():
    faculty, source, draw = taught(seed=11)
    target = mint_dialect(random.Random(99), "other")
    for utterance in sample(target, draw, 24, kinds=("assertion",)):
        faculty.show(utterance.surface, utterance.meaning, tongue=target.name)
    faculty.learn(tongue=target.name)
    utterance = sample(source, draw, 1)[0]
    broken = f"{utterance.surface} {draw()}"
    assert faculty.translate(broken, into=target.name, frm=source.name) == ""


@pytest.mark.parametrize("seed", [3, 7, 19, 42])
def test_she_reads_a_minted_language_the_shipped_compiler_reads_wrongly(seed):
    """The measurement the whole module exists for, over four languages nobody wrote code for.

    The shipped compiler is not silent on these — it reads every one of them, positionally, and
    gets every one of them wrong. So the comparison is not "learned beats nothing", it is
    "learned beats a confident guess", which is the harder and the honest one.
    """
    faculty, dialect, draw = taught(seed=seed)
    held_out = sample(dialect, draw, 15,
                      kinds=("assertion", "negated", "past", "polar", "content"))
    learned = sum(same(faculty.read(u.surface, tongue=dialect.name), u.meaning)
                  for u in held_out)
    shipped = sum(same(compile_meaning(u.surface), u.meaning) for u in held_out)
    assert learned == len(held_out)
    assert shipped == 0


def test_no_reading_is_ever_confidently_wrong_across_many_languages():
    """The safety property, asserted the way the coding half asserts its own: what she cannot
    read, she refuses. A miss may be an abstention; it may not be a wrong meaning."""
    wrong = 0
    for seed in range(1, 13):
        faculty, dialect, draw = taught(seed=seed, count=30)
        for utterance in sample(dialect, draw, 10,
                                kinds=("assertion", "negated", "past", "polar", "content")):
            got = faculty.read(utterance.surface, tongue=dialect.name)
            if got.readable and not same(got, utterance.meaning):
                wrong += 1
    assert wrong == 0


# --------------------------------------------------------------------------- #
# the minted languages themselves
# --------------------------------------------------------------------------- #

def test_a_minted_language_is_unambiguous_on_its_own_terms():
    """A dialect that cannot be read by its own rules would grade a right answer as wrong.

    Both halves of this were real: a wh-word ending in the object-case marker, and a verb stem
    drawn ending in the past-tense suffix.
    """
    for seed in range(1, 40):
        dialect = mint_dialect(random.Random(seed), "d")
        forms = [dialect.subject_case, dialect.object_case, dialect.plural, dialect.past,
                 dialect.negator, dialect.polar, dialect.wh_object, dialect.wh_subject]
        forms = [form for form in forms if form]
        for one in forms:
            for other in forms:
                if one is other:
                    continue
                assert not one.endswith(other), (seed, one, other)
                assert not one.startswith(other), (seed, one, other)
        draw = words(random.Random(seed))
        for _ in range(20):
            drawn = stem(dialect, draw)
            for marker in (dialect.subject_case, dialect.object_case, dialect.plural,
                           dialect.past):
                if marker:
                    assert not drawn.endswith(marker)


def test_tokenisation_keeps_the_question_mark_and_drops_the_stop():
    assert tokenize_surface("Is it so?") == ["is", "it", "so", "?"]
    assert tokenize_surface("It is so.") == ["it", "is", "so"]


# --------------------------------------------------------------------------- #
# the one edge into what she believes
# --------------------------------------------------------------------------- #

def _tiny_english_grammar(faculty: LanguageFaculty) -> None:
    """Two demonstrations of one shape, in a language with a particle English does not have."""
    for subject, verb, obj in (("cat", "chase", "dog"), ("bird", "chase", "fish"),
                               ("frog", "eat", "worm")):
        faculty.show(f"{obj}ni {subject}ta {verb}",
                     Meaning(kind="assertion", subject=subject, relation=verb, object=obj),
                     tongue="zz")
    faculty.learn(tongue="zz")


def test_a_learned_construction_outranks_the_positional_frame_but_not_a_pattern():
    """The precedence rule, both halves of it.

    The shipped compiler never abstains on a well-formed sentence, so a learned construction that
    only ran when the core was silent would never run at all. It outranks the *positional* read —
    which fires on any three tokens — when it matched fixed material the sentence contains, and it
    never outranks a shipped pattern, which named its relation lexically.
    """
    from nyxara.njp.grounding import Grounder
    faculty = LanguageFaculty()
    _tiny_english_grammar(faculty)
    grounder = Grounder(grammar=faculty)

    # Positional read, displaced: the case markers are evidence, the word order is a guess.
    result = grounder.ground("stoneni horseta eat")
    assert [(t.subject, t.predicate, t.object, t.source) for t in result.triples] == \
        [("horse", "eat", "stone", "learned-grammar")]
    assert grounder.learned_reads == 1

    # A shipped pattern, untouched — and every one of these read exactly as it did before.
    for text, source in (("my name is Jay", "pattern"),
                         ("water freezes at 0", "pattern"),
                         ("all zorbs are shiny", "pattern"),
                         ("the cat chases the dog", "semantics")):
        triples = Grounder(grammar=faculty).ground(text).triples
        assert triples and triples[0].source == source, text


def test_the_edge_does_nothing_at_all_before_a_lesson():
    """A faculty holds nothing until somebody demonstrates a sentence, so attaching one to a
    grounder changes not a single reading until a lesson has run."""
    from nyxara.njp.grounding import Grounder

    def shape(grounder, text):
        return [(t.subject, t.predicate, t.object, t.source, t.negated)
                for t in grounder.ground(text).triples]

    corpus = [
        "my name is Jay", "the cat chases the dog", "water freezes at 0",
        "all zorbs are shiny", "a zorbin is a plag", "zorbins need glarn",
        "Zorbins don't need glarn.", "does a zorbin need glarn?",
        "what does a zorbin need?", "gravity pulls the apple",
        "the fire spread", "it rained", "aag lagi", "mera naam Jay hai",
        "zorb ka rang neela hai", "pehle test chala", "how are you NYXARA?",
        "if the temperature falls below 0, water freezes",
        "the boiling point of water is 100", "sparrows are birds and birds fly",
        "stoneni horseta eat", "zorb plag glim", "wex zorb plag",
        "she may have needed it", "usually a zorbin eats plag",
    ]
    untaught, plain = Grounder(grammar=LanguageFaculty()), Grounder()
    for text in corpus:
        assert shape(untaught, text) == shape(plain, text), text
    assert untaught.learned_reads == 0


def test_a_question_is_never_grounded_as_a_fact_by_the_learned_grammar():
    """"what does a zorbin eat" is not the claim that a zorbin eats something called *what*."""
    from nyxara.njp.grounding import Grounder
    faculty = LanguageFaculty()
    for subject, verb in (("cat", "chase"), ("bird", "chase"), ("frog", "eat")):
        faculty.show(f"wex {subject}ta {verb}",
                     Meaning(kind="question", focus="object", subject=subject, relation=verb),
                     tongue="zz")
        faculty.show(f"{subject}ni {subject}ta {verb}",
                     Meaning(kind="assertion", subject=subject, relation=verb, object=subject),
                     tongue="zz")
    faculty.learn(tongue="zz")
    grounder = Grounder(grammar=faculty)
    assert grounder._extract_learned("wex horseta eat") == []


def test_a_denial_crosses_the_edge_as_a_denial():
    """This module's oldest bug, arriving by a new road, and refused at the door."""
    from nyxara.njp.grounding import Grounder
    faculty = LanguageFaculty()
    for subject, verb, obj in (("cat", "chase", "dog"), ("bird", "chase", "fish"),
                               ("frog", "eat", "worm")):
        faculty.show(f"{obj}ni {subject}ta nul {verb}",
                     Meaning(kind="assertion", subject=subject, relation=verb, object=obj,
                             negated=True), tongue="zz")
    faculty.learn(tongue="zz")
    triples = Grounder(grammar=faculty)._extract_learned("stoneni horseta nul eat")
    assert len(triples) == 1
    assert triples[0].negated is True
    assert (triples[0].subject, triples[0].predicate, triples[0].object) == \
        ("horse", "eat", "stone")


def test_the_brain_answers_in_english_a_fact_stated_in_a_language_it_was_taught():
    """End to end, through every organ between the two: four sentences of a language nobody
    wrote code for, and a question about them in a language she shipped with."""
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.school import ExamConditions
    brain = NJPBrain(ExamConditions())
    assert brain.language is not None
    assert brain.grounder.grammar is brain.language
    for subject, verb, obj in (("cat", "chase", "dog"), ("bird", "chase", "fish"),
                               ("frog", "eat", "worm"), ("goat", "eat", "leaf")):
        brain.show_language(f"{obj}ni {subject}ta {verb}",
                            Meaning(kind="assertion", subject=subject, relation=verb,
                                    object=obj), tongue="zz")
    assert brain.learn_language(tongue="zz").kept == 1
    brain.think("stoneni horseta eat")
    assert "stone" in brain.think("what does horse eat?").answer.lower()


# --------------------------------------------------------------------------- #
# waking up as the brain that went to sleep
# --------------------------------------------------------------------------- #

def test_a_taught_grammar_survives_a_restart():
    """A grammar that had to be re-taught every morning would be the one organ in this brain
    that does not persist, and persistence is the package's own claim about itself."""
    import json

    faculty, dialect, draw = taught(seed=19)
    faculty.hear_words([form for base in ("zorb", "plag", "glim", "vunt", "kesh")
                        for form in (base, dialect.pluralise(base))], tongue=dialect.name)
    faculty.bind("zorb", dialect.pluralise("zorb"), "plural", tongue=dialect.name)

    blob = json.loads(json.dumps(faculty.to_dict()))     # it has to be JSON, not just a dict
    woken = LanguageFaculty()
    woken.load_dict(blob)

    for utterance in sample(dialect, draw, 8):
        assert same(woken.read(utterance.surface, tongue=dialect.name), utterance.meaning)
        assert woken.say(utterance.meaning, tongue=dialect.name) == utterance.surface
    assert woken.inflect("wug", "plural", tongue=dialect.name) == dialect.pluralise("wug")
    # And the refusals wake up too.
    assert woken.inflect("wug", "future", tongue=dialect.name) == ""
    assert not woken.read(f"{utterance.surface} {draw()}", tongue=dialect.name).readable


def test_a_sidecar_cannot_give_her_a_shape_nobody_demonstrated():
    """Constructions are re-derived from the lessons on the way in rather than read out of the
    file, so the worst a tampered sidecar can do is make her forget."""
    faculty, dialect, draw = taught(seed=19)
    blob = faculty.to_dict()
    blob["tongues"][dialect.name]["records"] = {}
    blob["tongues"][dialect.name]["grammar"]["records"] = {"invented|0|||| <subject>": [99, 0]}
    blob["tongues"][dialect.name]["grammar"]["demonstrations"] = \
        blob["tongues"][dialect.name]["grammar"]["demonstrations"][:1]
    woken = LanguageFaculty()
    woken.load_dict(blob)
    # One demonstration is a sentence, not a shape — so what comes back is nothing, not a shape
    # the file asserted.
    assert woken.tongue(dialect.name).grammar.constructions == []
    assert not woken.read(sample(dialect, draw, 1)[0].surface, tongue=dialect.name).readable
