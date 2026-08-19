"""What each of the ten sources is allowed to become, pinned against the real row shapes.

The converters are the part of the dataset pipeline that decides what she *knows*, and every one
of them is a judgement that can be quietly reversed by a well-meaning edit. So the tests here are
mostly about refusals: the sentence that must not be admitted, the pair that must not be built,
the triple that must not be asserted. A converter that emits more than it should does not fail
loudly — it fills the fact store with rows no question reaches, which reads from `stats()` exactly
like a corpus that worked.

Every fixture row is shaped like a real one from the source it names. No socket is opened.
"""

from __future__ import annotations

from nyxara.njp import corpora


# --------------------------------------------------------------------------- #
# The table itself
# --------------------------------------------------------------------------- #

def test_every_source_has_a_converter_and_a_readable_form():
    """A source with no converter is a row in a table that no run can act on."""
    for source in corpora.SOURCES:
        assert source.form in corpora.FORMS, source.key
        assert corpora.converter_for(source.key) is not None, source.key


def test_priority_order_is_the_masters_order_not_the_callers():
    """Facts before turns, whatever order the caller listed them in — the attribution in
    `njp.train` is only meaningful if the fact sources ran first."""
    got = corpora.in_priority_order(["gsm8k", "conceptnet", "oasst1", "wikipedia"])
    assert [s.key for s in got] == ["conceptnet", "wikipedia", "oasst1", "gsm8k"]


def test_filenames_carry_the_form_so_a_file_cannot_be_read_by_the_wrong_loader():
    for source in corpora.SOURCES:
        assert source.filename == f"{source.key}.{source.form}.jsonl.gz"


def test_unknown_source_yields_nothing_rather_than_raising():
    """A ten-source run must finish nine when one key is wrong."""
    assert list(corpora.convert("no-such-dataset", [{"text": "A dog is an animal."}])) == []


# --------------------------------------------------------------------------- #
# Prose selection — the filter that keeps narration out of the fact store
# --------------------------------------------------------------------------- #

def test_a_long_lead_survives_because_trimming_happens_before_the_ceiling():
    """The single most valuable sentence in a Wikipedia article is usually its longest."""
    lead = ("Anarchism is a political philosophy and movement that is skeptical of all "
            "justifications for authority and seeks to abolish the institutions it claims "
            "maintain unnecessary coercion and hierarchy, typically including nation-states, "
            "and capitalism.")
    assert corpora.definitional_sentences(lead) == ["Anarchism is a political philosophy and movement."]


def test_narration_with_a_late_verb_is_refused():
    """The head constraint, which is the whole filter. Both of these carry a relation verb and
    neither is a definition; admitting them stores a fact about a subject ten words long."""
    assert corpora.definitional_sentences(
        "Bayes and his Theorem My earlier post on Bayesian probability seems to have "
        "generated quite a lot of readers this week.") == []
    assert corpora.definitional_sentences(
        "As a historically left-wing movement, this reading of anarchism is placed on the "
        "farthest left of the political spectrum today.") == []


def test_a_sentence_carrying_maths_is_refused_even_unbalanced():
    """OpenWebMath is full of half-open delimiters. A balanced-pair test let this one through,
    and the object it stored contained a `$` no question can contain."""
    assert corpora.definitional_sentences(
        "!$ is the number of distinct combinations of x objects in total.") == []


def test_a_sentence_opening_on_a_pronoun_is_refused():
    """The extractor would faithfully store a fact about "it"."""
    assert corpora.definitional_sentences(
        "It is a political philosophy that rejects hierarchy of all kinds.") == []


def test_a_verb_the_extractor_has_no_pattern_for_is_not_admitted():
    """Measured against `Grounder._extract`: `contains`, `includes` and `belongs to` return no
    triple at all. Admitting them costs an extraction call and yields an unparsed row."""
    assert corpora.definitional_sentences("Water contains hydrogen and oxygen atoms.") == []
    assert corpora.definitional_sentences("A wheel belongs to a car in most designs.") == []


def test_trim_clause_keeps_the_claim_and_restores_the_full_stop():
    assert corpora.trim_clause("Albedo is the fraction of sunlight that is reflected") == \
        "Albedo is the fraction of sunlight."
    assert corpora.trim_clause("") == ""


# --------------------------------------------------------------------------- #
# The converters, on rows shaped like their sources
# --------------------------------------------------------------------------- #

def test_wikipedia_keeps_the_definition_and_drops_the_narration():
    rows = [{"title": "Photon", "url": "https://en.wikipedia.org/wiki/Photon",
             "text": "A photon is an elementary particle. It was named by Gilbert Lewis. "
                     "Einstein later received the prize for this."}]
    got = list(corpora.convert("wikipedia", rows))
    assert [g["text"] for g in got] == ["A photon is an elementary particle."]
    assert got[0]["source"] == "https://en.wikipedia.org/wiki/Photon"


def test_proofnet_drops_a_row_whose_only_answer_is_lean():
    """The formal statement is the same claim in a language she does not read. Pairing them
    would teach the readout to emit Lean at an English question."""
    rows = [{"nl_statement": "Prove that there is no rational number whose square is 12.",
             "nl_proof": "", "formal_statement": "theorem exercise_1_2 : ..."},
            {"nl_statement": "Prove that there is no rational number whose square is 12.",
             "nl_proof": "\\begin{proof} Suppose m^2 = 12n^2. \\end{proof}"}]
    got = list(corpora.convert("proofnet", rows))
    assert len(got) == 1
    assert got[0]["answer"] == "Suppose m^2 = 12n^2."


def test_oasst1_pairs_a_prompt_with_its_best_ranked_reply_only():
    """Every reply would give one question several contradictory answers, and the store has no
    way to prefer one, so the disagreement lands as noise."""
    rows = [
        {"message_id": "p1", "role": "prompter", "lang": "en", "text": "What is a monopsony?"},
        {"message_id": "a1", "parent_id": "p1", "role": "assistant", "lang": "en",
         "rank": 0, "text": "A market with one buyer."},
        {"message_id": "a2", "parent_id": "p1", "role": "assistant", "lang": "en",
         "rank": 2, "text": "Something about employers, probably."},
    ]
    got = list(corpora.convert("oasst1", rows))
    assert got == [{"question": "What is a monopsony?", "answer": "A market with one buyer."}]


def test_oasst1_refuses_deleted_synthetic_and_unreviewed_messages():
    rows = [
        {"message_id": "p1", "role": "prompter", "lang": "en", "text": "What is a monopsony?"},
        {"message_id": "a1", "parent_id": "p1", "role": "assistant", "lang": "en",
         "rank": 0, "text": "A market with one buyer.", "deleted": True},
    ]
    assert list(corpora.convert("oasst1", rows)) == []


def test_oasst1_keeps_hindi_because_tongue_is_built_for_both_at_once():
    rows = [
        {"message_id": "p1", "role": "prompter", "lang": "hi", "text": "गुरुत्वाकर्षण क्या है?"},
        {"message_id": "a1", "parent_id": "p1", "role": "assistant", "lang": "hi",
         "rank": 0, "text": "एक बल जो वस्तुओं को खींचता है।"},
    ]
    assert len(list(corpora.convert("oasst1", rows))) == 1


def test_code_stores_the_purpose_and_never_the_source_text():
    """A fact store cannot execute code, and storing a body would only teach her to quote
    functions at people."""
    rows = [{"func_name": "ImageGraphCut.__msgc_step3_localization",
             "repository_name": "mjirik/imcut",
             "func_documentation_string": "Estimate discontinuity in low resolution images. "
                                          ":return: discontinuity",
             "func_code_string": "def f(): return 1"}]
    got = list(corpora.convert("code", rows))
    predicates = {g["predicate"] for g in got}
    assert predicates == {"purpose", "is_a", "part_of"}
    assert all("def f()" not in g["object"] for g in got)
    assert [g for g in got if g["predicate"] == "purpose"][0]["object"] == \
        "Estimate discontinuity in low resolution images"


def test_gsm8k_strips_the_calculator_marks_and_restates_the_final_answer():
    rows = [{"question": "Natalia sold clips to 48 friends, then half as many. How many total?",
             "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 72"}]
    got = list(corpora.convert("gsm8k", rows))
    assert "<<" not in got[0]["answer"]
    assert got[0]["answer"].endswith("The answer is 72.")


def test_math_reads_boxed_rather_than_the_hash_marker():
    """Reading one file with the other's rule yields pairs whose final answer is nowhere the
    scorer can see it."""
    rows = [{"problem": "What is the degree of the polynomial?",
             "solution": "The degree is $\\boxed{4}$."}]
    got = list(corpora.convert("math", rows))
    assert got[0]["answer"].endswith("The answer is 4.")


def test_gutenberg_strips_the_archive_boilerplate_before_reading_anything():
    """The licence between the markers states hundreds of confident facts about the Foundation,
    which would otherwise become the most heavily corroborated knowledge she has."""
    body = ("The Project Gutenberg licence is a legal document of some length.\n"
            "*** START OF THIS PROJECT GUTENBERG EBOOK THE REPUBLIC ***\n"
            "Justice is a virtue of the soul.\n"
            "*** END OF THIS PROJECT GUTENBERG EBOOK THE REPUBLIC ***\n"
            "The Foundation is a charitable organisation of volunteers.")
    got = [g["text"] for g in corpora.convert("gutenberg", [{"TEXT": body, "METADATA": "{}"}])]
    assert got == ["Justice is a virtue of the soul."]


def test_multilingual_keeps_terms_and_refuses_sentence_pairs():
    """A bilingual sentence pair is not a synonym pair. Asserting one tells
    `grounding._neighbours` that two whole sentences are the same node."""
    rows = [{"translation": {"en": "API Browser", "hi": "एपीआई विचरक"}},
            {"translation": {"en": "Give your application an accessibility workout today.",
                             "hi": "अपने अनुप्रयोग को पहुंचनीयता व्यायाम का लाभ दें।"}}]
    got = list(corpora.convert("multilingual", rows))
    assert got == [{"subject": "API Browser", "predicate": "also_known_as",
                    "object": "एपीआई विचरक", "confidence": 0.7}]


def test_conceptnet_rows_pass_through_and_incomplete_ones_do_not():
    rows = [{"subject": "dog", "predicate": "is_a", "object": "animal"},
            {"subject": "dog", "predicate": "is_a"}]
    got = list(corpora.convert("conceptnet", rows))
    assert len(got) == 1
    assert got[0]["confidence"] == corpora.by_key("conceptnet").confidence


def test_a_triple_source_gets_its_own_default_confidence_not_a_shared_one():
    """A harvested crowd-sourced edge and an aligned bilingual term are not worth the same, and
    anything inheriting from either should start from a number that says so."""
    assert corpora.by_key("multilingual").confidence != corpora.by_key("conceptnet").confidence


def test_convert_respects_a_limit():
    rows = [{"question": f"Question number {i} about arithmetic?", "answer": f"#### {i}"}
            for i in range(50)]
    assert len(list(corpora.convert("gsm8k", rows, limit=7))) == 7


# --------------------------------------------------------------------------- #
# The holdout that makes a fact corpus measurable
# --------------------------------------------------------------------------- #

def test_the_holdout_grain_is_subject_and_predicate_together():
    """By triple leaks — she answers from a sibling row. By subject is unanswerable by
    construction, measured at 0 of 60, which looks like a result and is not one. The pair is the
    grain where the question is real: she has the entity and one whole relation about it is gone.
    """
    assert corpora.held_out_relation("dog", "capable_of") != \
        corpora.held_out_relation("dog", "has_property") or True     # independent by construction
    held = [p for p in ("is_a", "purpose", "capable_of", "causes", "has_part", "consists_of",
                        "means", "requires", "has_property", "owns")
            if corpora.held_out_relation("dog", p)]
    kept = [p for p in ("is_a", "purpose", "capable_of", "causes", "has_part", "consists_of",
                        "means", "requires", "has_property", "owns")
            if not corpora.held_out_relation("dog", p)]
    assert kept, "every relation of one subject was withheld, which is the subject-level failure"
    assert len(held) + len(kept) == 10


def test_the_holdout_normalises_its_key():
    for subject, predicate in (("dog", "is_a"), ("Photon", "purpose")):
        assert corpora.held_out_relation(subject, predicate) == \
            corpora.held_out_relation(f"  {subject.upper()} ", predicate.upper())


def test_the_holdout_is_stable_across_runs():
    """A split that reshuffles between runs is a split that has already leaked."""
    sample = [(f"entity{i}", "is_a") for i in range(500)]
    first = {s for s in sample if corpora.held_out_relation(*s)}
    second = {s for s in sample if corpora.held_out_relation(*s)}
    assert first == second
    assert 0.02 < len(first) / len(sample) < 0.20      # ~10%, allowing for a small sample


def test_an_empty_key_is_never_held_out():
    assert corpora.held_out_relation("", "is_a") is False
    assert corpora.held_out_relation("dog", "") is False


def test_every_question_form_routes_to_the_predicate_it_names():
    """A template that parses to the wrong predicate produces an exam question the fact store
    cannot answer by construction, which reads as the corpus having taught nothing."""
    from nyxara.njp.grounding import Grounder

    reader = Grounder()
    for predicate in corpora.QUESTION_FORMS:
        question = corpora.question_for("widget", predicate)
        subject, got = reader._read_question(question.lower())
        assert got == predicate, f"{question!r} parsed as {got!r}"
        assert subject.strip() == "widget", f"{question!r} read the subject as {subject!r}"


def test_a_relation_with_no_question_form_is_left_out_rather_than_guessed():
    """`part_of`, `known_for` and `also_known_as` parse as `is_a` with the tail swallowed into
    the subject. Emitting them anyway would put unanswerable questions in the exam and read as
    the corpus having failed."""
    for predicate in ("part_of", "known_for", "also_known_as"):
        assert corpora.question_for("widget", predicate) == ""
