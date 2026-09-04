"""The reader against prose it did not write, and the citation that travels with every claim."""

from __future__ import annotations

import pytest

from nyxara.njp.encyclopedia import (WIKIPEDIA_LESSONS, Article, Encyclopedia,
                                     taught_on_wikipedia)
from nyxara.njp.encyclopediaschool import audit, kinds, recall, slices
from nyxara.njp.passage import _sentences, taught_reader


@pytest.fixture(scope="module")
def book() -> Encyclopedia:
    return Encyclopedia()


@pytest.fixture(scope="module")
def reader(book):
    return taught_on_wikipedia(book)


# --------------------------------------------------------------------------------------------- #
#  the corpus
# --------------------------------------------------------------------------------------------- #
def test_the_corpus_is_real_prose_from_many_subjects(book):
    articles = book.load()
    assert len(articles) > 1000
    assert len(book.domains()) > 20
    assert all(a.text and a.title for a in articles)


def test_every_article_can_say_where_it_came_from(book):
    for article in book.sample(50, every=13):
        assert article.url.startswith("https://en.wikipedia.org/wiki/")
        assert article.licence and article.fetched
        assert article.source == "en.wikipedia.org"


def test_the_stripper_left_no_markup_behind(book):
    """A lead still carrying wikitext would put the builder's defects into the reader's score."""
    for article in book.sample(300, every=7):
        for mark in ("{{", "[[", "]]", "<ref", "|}"):
            assert mark not in article.text, f"{article.title}: {mark}"


def test_the_citation_travels_onto_the_reading(book, reader):
    article = book.sample(1, every=7)[0]
    obj = book.read(article, reader)
    assert obj.url == article.url
    assert obj.retrieved == article.fetched
    assert obj.licence == article.licence
    assert obj.text == article.text


def test_a_title_s_disambiguator_is_not_part_of_the_name(book):
    assert book.name_of(Article(title="Indiana Jones (character)")) == "Indiana Jones"
    assert book.name_of(Article(title="Byte")) == "Byte"


# --------------------------------------------------------------------------------------------- #
#  the lessons
# --------------------------------------------------------------------------------------------- #
def test_the_lessons_are_the_corpus_own_sentences(book):
    titles = {a.title for a in book.load()}
    assert set(WIKIPEDIA_LESSONS) <= titles
    for lesson in book.lessons():
        assert lesson.text == next(a.text for a in book.load() if a.title in lesson.name)


def test_no_lesson_is_in_the_audited_or_the_sealed_slice(book):
    """A demonstration inside the exam would make every number here a report about itself."""
    first, sealed = slices(book)
    held = {a.title for a in first} | {a.title for a in sealed}
    assert not (set(WIKIPEDIA_LESSONS) & held)


def test_the_two_slices_are_disjoint(book):
    first, sealed = slices(book)
    assert {a.title for a in first}.isdisjoint({a.title for a in sealed})


def test_teaching_from_the_encyclopedia_reads_more_of_it(book, reader):
    """Seven real sentences bought more than any single fix to the reader's own machinery."""
    theirs = book.coverage(reader, book.sample(400, every=7))
    mine = book.coverage(taught_reader(), book.sample(400, every=7))
    assert theirs.definition_rate > mine.definition_rate + 0.10
    assert theirs.any_relation > mine.any_relation


# --------------------------------------------------------------------------------------------- #
#  what real prose broke
# --------------------------------------------------------------------------------------------- #
def test_a_passive_is_not_a_definition(reader):
    obj = reader.read("Vorbis was released in May 2000.", concept="Vorbis")
    assert obj.definition == ""


@pytest.mark.parametrize("text,concept", [
    ("The bagpipes are well known.", "Bagpipes"),
    ("Cooking bananas are generally starchy.", "Cooking banana"),
    ("These factors are naturally occurring.", "Abiotic stress"),
    ("Abiotic stress is essentially unavoidable.", "Abiotic stress"),
])
def test_a_predication_about_a_subject_is_not_a_description_of_it(reader, text, concept):
    """A determinerless complement is a noun phrase only if it is plural."""
    assert reader.read(text, concept=concept).definition == ""


@pytest.mark.parametrize("text,concept", [
    ("The thyroid, or thyroid gland, is an endocrine gland in vertebrates.", "Thyroid"),
    ("In computer science, radix sort is a non-comparative sorting algorithm.", "Radix sort"),
    ("Ieoh Ming Pei (April 26, 1917 - May 16, 2019) was a Chinese-American architect.",
     "I. M. Pei"),
    ("Bees are winged insects that form a clade.", "Bee"),
])
def test_the_openings_encyclopedic_prose_actually_uses(reader, text, concept):
    assert reader.read(text, concept=concept).definition


def test_a_sentence_the_topic_merely_mentions_does_not_define_it(reader):
    """Every lead has several copulas and only the first is about the article."""
    obj = reader.read("The lungs are the primary organs of respiration. "
                      "Their function is to extract oxygen from the atmosphere.", concept="Lung")
    assert "extract oxygen" not in obj.definition


def test_a_full_stop_after_an_initialism_ends_the_sentence(reader):
    text = ("Callicrates was an ancient Greek architect active in the fifth century BC. "
            "He and Ictinus were architects of the Parthenon.")
    assert len(_sentences(text)) == 2
    assert "Ictinus" not in reader.read(text, concept="Callicrates").definition


def test_a_full_stop_after_an_abbreviation_does_not(reader):
    assert len(_sentences('Dr. Henry Jones is the title character of the franchise.')) == 1


def test_a_semicolon_ends_a_filler(reader):
    obj = reader.read("Afonso III was the second son of King Afonso II; he succeeded his "
                      "brother.", concept="Afonso III")
    assert ";" not in obj.definition and "succeeded" not in obj.definition


def test_a_dangling_relative_is_cut_but_a_governed_one_is_not(reader):
    dangling = reader.read("Cooking bananas are a group of banana cultivars in the genus Musa "
                           "whose fruits are generally used in cooking.", concept="Cooking banana")
    assert dangling.definition.endswith("Musa")
    governed = reader.read("Erosion is the process by which wind removes soil from a surface.",
                           concept="Erosion")
    assert "by which" in governed.definition


def test_a_filler_is_a_span_of_the_passage_and_not_a_reconstruction(reader):
    text = ("Abacá, also known as Manila hemp, is a species of banana, Musa textilis, endemic "
            "to the Philippines.")
    obj = reader.read(text, concept="Abacá")
    assert obj.definition and obj.definition in text


# --------------------------------------------------------------------------------------------- #
#  the marked numbers
# --------------------------------------------------------------------------------------------- #
def test_the_audit_covers_what_the_reader_produces(book, reader):
    """An audit with more unmarked relations than marked ones is not a measurement of anything."""
    first, _ = slices(book)
    marked = audit(first, reader, book)
    assert marked.marked > 20
    assert len(marked.unmarked) <= marked.marked


def test_precision_on_real_prose_holds_where_it_was_measured(book, reader):
    first, sealed = slices(book)
    assert audit(first, reader, book).precision >= 0.85
    assert audit(sealed, reader, book).precision >= 0.80


def test_the_two_slices_do_not_disagree_about_the_reader(book, reader):
    """The fixes were made against the first slice, so a wide gap would mean they were fitted.

    They were, once: before the complement test the audited slice read 0.750 and the sealed one
    0.642, and the difference was the measurement telling on the fixes. A structural rule closed
    it — which is what distinguishes one from a patch.
    """
    first, sealed = slices(book)
    assert abs(audit(first, reader, book).precision
               - audit(sealed, reader, book).precision) < 0.10


def test_the_definition_and_the_kind_are_marked_separately(book):
    got, asked = recall(book=book)
    heads, also = kinds()
    assert asked == also == 25
    assert got >= 0.75
    assert heads >= 0.65
    assert heads < got, "the head rule is the weaker half and the numbers should say so"


def test_reading_the_corpus_files_nothing(book, reader):
    from nyxara.njp.grounding import Grounder

    grounder = Grounder()
    before = len(grounder.facts)
    for article in book.sample(20, every=7):
        book.read(article, reader)
    assert len(grounder.facts) == before


# --------------------------------------------------------------------------------------------- #
#  the brain
# --------------------------------------------------------------------------------------------- #
def test_the_brain_reads_an_article_with_its_citation():
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    assert brain.encyclopedia is not None
    obj = brain.read_article("Black hole")
    assert obj is not None and obj.definition
    assert obj.url.endswith("Black_hole") and obj.licence


def test_what_she_learns_from_the_encyclopedia_is_answerable_in_english():
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    got = brain.learn_encyclopedia(limit=40, domain="astronomy")
    assert got["articles"] == 40 and got["filed"] > 20
    answer = brain.think("what does black hole mean?").answer
    assert "gravity" in answer
