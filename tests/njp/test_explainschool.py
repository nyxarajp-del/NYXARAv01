"""The examination for what, how and why (NJP V.36).

Most of this file is about the *exam*, not about her. Three of the seven papers scored perfect on
their first run and could not have scored anything else, and the tests that would have caught that
are the ones that matter here: a paper has to be able to fail.
"""
from __future__ import annotations

import pytest

from nyxara.njp.explain import Explainer
from nyxara.njp.explainschool import (
    DOMAINS, INVERTED, PAPERS, DomainReport, ExplanationExam, _orders, render,
    subjects_by_domain,
)


@pytest.fixture(scope="module")
def brain():
    from nyxara.njp.general import load_brain

    return load_brain()


@pytest.fixture(scope="module")
def exam(brain):
    return ExplanationExam(brain, limit=40)


# --------------------------------------------------------------------------- #
# The forty-two
# --------------------------------------------------------------------------- #
def test_every_domain_has_at_least_one_source_and_one_subject():
    got = subjects_by_domain()
    assert set(got) == set(DOMAINS)
    empty = sorted(name for name, subjects in got.items() if not subjects)
    assert empty == [], f"domains with no subjects at all: {empty}"


def test_every_kb_file_is_claimed_by_a_domain():
    """A source nobody's domain names is a file the per-domain report cannot see."""
    import os

    from nyxara.njp.explainschool import _KB_DIR

    if not os.path.isdir(_KB_DIR):
        pytest.skip("sources are not on disk")
    files = {n[:-3] for n in os.listdir(_KB_DIR) if n.endswith(".kb")}
    claimed = {stem for stems in DOMAINS.values() for stem in stems}
    assert files - claimed == set(), f"unclaimed sources: {sorted(files - claimed)}"
    assert claimed - files == set(), f"domains naming missing sources: {sorted(claimed - files)}"


# --------------------------------------------------------------------------- #
# Held out means held out
# --------------------------------------------------------------------------- #
def test_no_why_chain_item_asks_for_something_stated_outright(exam):
    for item in exam.paper_why_chain()[:400]:
        assert not exam.holds(item.gold[0], "causes", item.topic), item.note


def test_no_mechanism_item_asks_for_a_pair_stated_about_the_whole(exam):
    for item in exam.paper_mechanism()[:200]:
        for part, effect in item.payload:
            assert not exam.holds(item.topic, "purpose", effect)
            assert not exam.holds(item.topic, "causes", effect)


def test_no_abstention_item_is_about_something_the_store_could_answer(exam):
    for item in exam.paper_abstention(60):
        assert not exam.by_sp.get((exam.key(item.topic), "has_step"))


# --------------------------------------------------------------------------- #
# A paper has to be able to fail
# --------------------------------------------------------------------------- #
def test_why_chain_collapses_when_the_second_hop_is_taken_away(brain):
    """A two-hop paper measured at depth one, which is where the hop it is about lives."""
    deep = ExplanationExam(brain, limit=60)
    shallow = ExplanationExam(brain, limit=60)
    shallow.explainer = Explainer(brain.grounder, max_depth=1)
    assert deep.sit(("why_chain",), sweep=False).score > 0.8
    assert shallow.sit(("why_chain",), sweep=False).score < 0.2


def test_mechanism_collapses_without_the_decomposition_relations(brain, monkeypatch):
    import nyxara.njp.explain as explain

    before = ExplanationExam(brain, limit=60).sit(("mechanism",), sweep=False).score
    monkeypatch.setattr(explain, "PARTS", ())
    after = ExplanationExam(brain, limit=60).sit(("mechanism",), sweep=False).score
    assert before > 0.8 and after == 0.0


def test_direction_collapses_when_the_two_senses_are_swapped(brain, monkeypatch):
    """The paper that could not fail until it was asked in both directions."""
    import nyxara.njp.explain as explain

    original = explain.Explainer.why

    def swapped(self, topic, *, sense="any", depth=0):
        return original(self, topic,
                        sense={"because": "for", "for": "because"}.get(sense, sense), depth=depth)

    before = ExplanationExam(brain, limit=60).sit(("direction",), sweep=False).score
    monkeypatch.setattr(explain.Explainer, "why", swapped)
    after = ExplanationExam(brain, limit=60).sit(("direction",), sweep=False).score
    assert before > 0.9 and after == 0.0


def test_direction_asks_both_ways_round(exam):
    kinds = {item.question.split()[0].lower() for item in exam.paper_direction()}
    assert kinds == {"what", "why"}


def test_procedure_notices_a_count_that_is_wrong(brain, monkeypatch):
    """Reporting one order for a procedure with six is a different error from a wrong order."""
    import nyxara.njp.explain as explain

    original = explain.Explainer._orders

    def only_one(self, steps, needs):
        found, count, cycle = original(self, steps, needs)
        return found[:1], (1 if count else 0), cycle

    before = ExplanationExam(brain, limit=40).sit(("procedure",), sweep=False).score
    monkeypatch.setattr(explain.Explainer, "_orders", only_one)
    after = ExplanationExam(brain, limit=40).sit(("procedure",), sweep=False).score
    assert before == 1.0 and after < before


# --------------------------------------------------------------------------- #
# The minted paper: no corpus can have leaked into it
# --------------------------------------------------------------------------- #
def test_the_minted_paper_needs_no_corpus_at_all():
    class Bare:
        class grounder:
            facts: dict = {}

            @staticmethod
            def _key(text):
                return " ".join(str(text or "").split()).lower()

    got = ExplanationExam(Bare(), limit=40).sit(("minted",), sweep=False)
    assert got.paper("minted").asked == 40
    assert got.paper("minted").score == 1.0


def test_the_minted_words_are_in_no_corpus(exam):
    for item in exam.paper_minted(40):
        for step in item.payload[0]:
            assert (exam.key(step), "is_a") not in exam.by_sp


def test_the_ground_truth_is_computed_a_different_way():
    """Brute force over permutations, so the sort is not graded against itself."""
    steps = ["a", "b", "c"]
    needs = {"a": [], "b": ["a"], "c": []}
    truth = {tuple(o) for o in _orders(steps, needs)}
    assert truth == {("a", "b", "c"), ("a", "c", "b"), ("c", "a", "b")}


def test_the_minted_paper_goes_through_the_question_grammar(exam):
    """Not straight into `procedure`. A derivation only Python can call is the V.21 gap."""
    from nyxara.njp.explainread import read_explanation_question

    for item in exam.paper_minted(20):
        got = read_explanation_question(item.question)
        assert got is not None and got[0] == "procedure"


# --------------------------------------------------------------------------- #
# The report says what it means
# --------------------------------------------------------------------------- #
def test_a_domain_never_asked_reads_as_a_dash_not_as_a_zero():
    got = DomainReport(name="x")
    assert got.what is None and got.held is None
    got.why_asked, got.why_right = 4, 0
    assert got.why == 0.0
    assert "     -" in render(_one(got))


def _one(domain):
    from nyxara.njp.explainschool import ExamReport

    return ExamReport(papers=[], domains=[domain])


def test_the_inverted_paper_is_not_in_the_total(brain):
    got = ExplanationExam(brain, limit=20).sit(("why_chain", "abstention"), sweep=False)
    assert got.asked == got.paper("why_chain").asked
    assert "abstention" in INVERTED


def test_silence_is_the_pass_on_the_inverted_paper_and_nowhere_else(brain):
    got = ExplanationExam(brain, limit=40).sit(("abstention",), sweep=False)
    paper = got.paper("abstention")
    assert paper.silent == paper.right


def test_every_named_paper_runs(brain):
    got = ExplanationExam(brain, limit=8).sit(sweep=False)
    assert [p.name for p in got.papers] == list(PAPERS)
    assert all(p.asked > 0 for p in got.papers)
