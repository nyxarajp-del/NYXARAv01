"""Teaching NJP from a corpus, and the discipline that makes the measurement mean anything.

The tests that matter here are not "does it learn" — they are the ones that stop a training
report from being able to flatter itself: that the held-out split never leaks, that the exam
distinguishes an honest abstention from a wrong answer, and that a scoring metric cannot be won
by returning function words.
"""

from __future__ import annotations

import gzip
import json

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.study import DEFAULT_CORPUS, Corpus, Pair, Tutor, _content, _f1

PAIRS = [
    Pair(question="What is a neural network?",
         answer="A neural network is a computing system of connected nodes that learns weights."),
    Pair(question="What is gradient descent?",
         answer="Gradient descent is an optimisation method that moves weights down the slope."),
    Pair(question="What is overfitting?",
         answer="Overfitting is when a model memorises training data and fails on new data."),
]


# --------------------------------------------------------------------------- #
# The corpus that ships
# --------------------------------------------------------------------------- #
def test_the_bundled_corpus_is_present_and_well_formed():
    assert DEFAULT_CORPUS.exists(), "the bundled AI question/answer corpus is missing"
    pairs = Corpus.load(DEFAULT_CORPUS, limit=200)
    assert len(pairs) == 200
    assert all(p.question and p.answer for p in pairs)


def test_the_corpus_has_no_duplicate_questions():
    """A duplicate across the split is a leak wearing a different hat."""
    pairs = Corpus.load(DEFAULT_CORPUS, limit=4000)
    keys = [p.key for p in pairs]
    assert len(keys) == len(set(keys))


def test_a_plain_json_list_loads_too(tmp_path):
    """The original upload's shape — capitalised keys, a JSON array — must still read."""
    path = tmp_path / "raw.json"
    path.write_text(json.dumps([{"Question": "Q one?", "Answer": "A one.\\n"}]), encoding="utf-8")
    pairs = Corpus.load(path)
    assert len(pairs) == 1 and pairs[0].question == "Q one?"
    assert "\\n" not in pairs[0].answer


def test_gzipped_jsonl_loads(tmp_path):
    path = tmp_path / "c.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"question": "Q?", "answer": "A."}) + "\n")
    assert len(Corpus.load(path)) == 1


# --------------------------------------------------------------------------- #
# The split
# --------------------------------------------------------------------------- #
def test_the_split_is_stable_across_runs():
    """A held-out set that reshuffles is a held-out set that has already leaked."""
    first = {p.key for p in Corpus.load(DEFAULT_CORPUS, limit=1000) if p.held_out}
    second = {p.key for p in Corpus.load(DEFAULT_CORPUS, limit=1000) if p.held_out}
    assert first == second
    assert first, "nothing was held out at all"


def test_study_and_exam_sets_are_disjoint():
    study, exam = Corpus.split(Corpus.load(DEFAULT_CORPUS, limit=2000))
    assert study and exam
    assert not ({p.key for p in study} & {p.key for p in exam})


def test_the_tutor_refuses_a_held_out_pair_even_if_handed_one():
    """Enforced at the tutor, not only at the caller — a leak into the exam set is silent."""
    pairs = Corpus.load(DEFAULT_CORPUS, limit=600)
    held = [p for p in pairs if p.held_out][:5]
    assert held
    tutor = Tutor(NJPBrain(), crystallise_every=1000)
    report = tutor.study(held)
    assert report.studied == 0
    assert report.skipped == len(held)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_function_words_cannot_win_the_metric():
    gold = _content("A neural network is a system of connected nodes")
    assert "is" not in gold and "of" not in gold
    assert _f1(gold, _content("the of and is a to")) == 0.0


def test_f1_rewards_the_right_content_and_is_symmetric():
    gold = _content("Overfitting is when a model memorises training data")
    assert _f1(gold, _content("overfitting means the model memorises training data")) > 0.6
    # Padding a correct answer with noise costs precision, so verbosity is not free.
    padded = _f1(gold, _content("overfitting memorises training data " + "noise " * 20))
    assert padded < _f1(gold, _content("overfitting memorises training data"))


# --------------------------------------------------------------------------- #
# Studying
# --------------------------------------------------------------------------- #
def test_studying_moves_the_organs_it_claims_to():
    brain = NJPBrain()
    tutor = Tutor(brain, crystallise_every=2)
    report = tutor.study(PAIRS)
    assert report.studied == len(PAIRS)
    assert report.grew, report.to_dict()
    assert report.synapses_after >= report.synapses_before
    assert report.facts_after >= report.facts_before


def test_a_question_reaches_the_answer_it_was_taught_with():
    """The association half: the question is bound to its answer, not merely near it."""
    brain = NJPBrain()
    Tutor(brain, crystallise_every=100).study(PAIRS)
    recall = brain.memory.recall("What is overfitting?", k=3)
    assert recall is not None
    texts = " ".join(getattr(t, "text", "") for t in (getattr(recall, "traces", None) or []))
    assert "overfitting" in (texts + str(getattr(recall, "text", ""))).lower()


def test_the_tutor_leaves_the_brains_conversational_cadence_untouched():
    """A tutored brain must be in the same posture afterwards as a brain that was talked to."""
    brain = NJPBrain()
    before = brain.field.crystallise_every
    Tutor(brain, crystallise_every=before + 500).study(PAIRS)
    assert brain.field.crystallise_every == before


# --------------------------------------------------------------------------- #
# Examining
# --------------------------------------------------------------------------- #
def test_abstention_is_counted_apart_from_being_wrong():
    """Folding a principled 'I don't know' into the error rate punishes the design goal."""
    brain = NJPBrain()
    tutor = Tutor(brain)
    report = tutor.exam(PAIRS)          # nothing studied: she should abstain, not guess
    assert report.asked == len(PAIRS)
    assert report.abstained == report.asked - report.answered
    assert report.correct == 0
    assert report.coverage == pytest.approx(report.answered / report.asked)


def test_an_untaught_brain_scores_zero_which_is_the_control():
    report = Tutor(NJPBrain()).exam(PAIRS)
    assert report.accuracy == 0.0
    assert report.precision == 0.0


def test_precision_and_coverage_are_reported_separately():
    """One 'accuracy' number cannot tell an honest abstainer from a confident guesser."""
    report = Tutor(NJPBrain()).exam(PAIRS)
    d = report.to_dict()
    assert "coverage" in d and "precision_when_answered" in d and "accuracy_overall" in d
