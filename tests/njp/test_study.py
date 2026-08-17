"""Teaching NJP from a corpus, and the discipline that makes the measurement mean anything.

The tests that matter here are not "does it learn" — they are the ones that stop a training
report from being able to flatter itself: that the held-out split never leaks, that the exam
distinguishes an honest abstention from a wrong answer, and that a scoring metric cannot be won
by returning function words.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nyxara.njp import study
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


# --------------------------------------------------------------------------- #
# Sampling a subset, and persisting a run
# --------------------------------------------------------------------------- #
def test_a_limited_load_samples_the_corpus_rather_than_taking_its_head():
    """The first N rows of a topic-grouped corpus are a different corpus, not a smaller one."""
    head = Corpus.load(DEFAULT_CORPUS, limit=300, sample=False)
    drawn = Corpus.load(DEFAULT_CORPUS, limit=300, seed=0)
    assert len(head) == len(drawn) == 300
    assert {p.question for p in head} != {p.question for p in drawn}
    # The sample must reach past the head of the file, which is the whole point.
    full_head = {p.question for p in Corpus.load(DEFAULT_CORPUS, limit=300, sample=False)}
    assert any(p.question not in full_head for p in drawn)


def test_the_sample_is_reproducible_across_calls():
    """A subsample that reshuffles makes two runs incomparable."""
    a = Corpus.load(DEFAULT_CORPUS, limit=200, seed=7)
    b = Corpus.load(DEFAULT_CORPUS, limit=200, seed=7)
    assert [p.question for p in a] == [p.question for p in b]
    assert [p.question for p in a] != [p.question for p in Corpus.load(
        DEFAULT_CORPUS, limit=200, seed=8)]


def test_sampling_does_not_disturb_the_held_out_split():
    """The fold is a hash of the question, so it must not depend on how the pair was drawn."""
    for pair in Corpus.load(DEFAULT_CORPUS, limit=300, seed=3):
        assert pair.held_out == Pair(question=pair.question, answer=pair.answer).held_out


def test_a_saved_state_round_trips_into_a_fresh_brain(tmp_path):
    from nyxara.njp.study import load_state, save_state

    taught = NJPBrain()
    for pair in PAIRS:
        taught.think(pair.answer)
    path = save_state(taught, tmp_path / "state.json")

    revived = NJPBrain()
    assert load_state(revived, path) is True
    assert revived.fabric.n_synapses == taught.fabric.n_synapses
    assert (revived.stats()["grounding"]["facts"] == taught.stats()["grounding"]["facts"])


def test_load_state_reports_a_missing_file_rather_than_pretending(tmp_path):
    from nyxara.njp.study import load_state

    assert load_state(NJPBrain(), tmp_path / "never-written.json") is False


def test_a_checkpoint_never_destroys_the_last_good_one(tmp_path):
    """The moment a checkpoint is for is the moment it is most likely to be half-written."""
    from nyxara.njp.study import save_state

    target = tmp_path / "state.json"
    brain = NJPBrain()
    brain.think("a neural network learns weights")
    save_state(brain, target)
    good = target.read_text(encoding="utf-8")

    class Exploding:
        def to_dict(self):
            raise RuntimeError("died mid-write")

    with pytest.raises(RuntimeError):
        save_state(Exploding(), target)

    assert target.read_text(encoding="utf-8") == good, "a failed write clobbered the good state"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "state.json"]
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_the_study_report_shows_what_the_gradient_learner_did():
    """The only gradient learner in NJP trains once per pair and was invisible to this report."""
    brain = NJPBrain()
    report = Tutor(brain, seed=False).study(PAIRS)
    d = report.to_dict()
    assert "readout" in d
    if brain.readout is not None:                 # absent without numpy, and that is reported
        assert d["readout"]["steps"] > 0
        assert d["readout"]["width"][1] > 0


def test_readout_loss_is_reported_per_slot_so_a_width_change_cannot_fake_a_trend():
    """The raw loss is summed over slots, so it scales with a width the readout doubles itself.

    Measured over 500 corpus pairs: width 512 → 8192, raw loss 33.50 → 197.66. Read raw, training
    made it six times worse; read per slot, 0.0654 → 0.0241, it improved threefold. Comparing the
    two ends of a pass at different widths is not a comparison.
    """
    from nyxara.njp.study import StudyReport

    rep = StudyReport(readout_loss_before=33.502175, readout_width_before=512,
                      readout_loss_after=197.657845, readout_width=8192)
    assert rep.readout_loss_after > rep.readout_loss_before      # raw says it got worse
    assert rep.readout_learned > 0, "per-slot must see the improvement the raw number hides"
    per_slot = rep.to_dict()["readout"]["loss_per_slot"]
    assert per_slot[0] == pytest.approx(0.0654, abs=1e-3)
    assert per_slot[1] == pytest.approx(0.0241, abs=1e-3)


def test_an_absent_readout_reports_nothing_rather_than_zero():
    """No readout and a readout trained to zero loss must not look the same."""
    from nyxara.njp.study import StudyReport

    d = StudyReport().to_dict()["readout"]
    assert d["loss"] == [None, None]
    assert d["loss_per_slot"] == [None, None]
    assert d["learned_per_slot"] is None


# --------------------------------------------------------------------------- #
# The entry point, invoked the way a user invokes it
# --------------------------------------------------------------------------- #
def test_the_cli_runs_as_a_module_and_not_only_as_an_import():
    """``python -m nyxara.njp.study`` must complete. Importing ``main`` does not prove that.

    The regression: ``seed_kinds`` was defined *below* ``if __name__ == "__main__"``, so under
    ``-m`` the guard fired while the module body was still executing and ``Tutor.study`` raised
    ``NameError: name 'seed_kinds' is not defined`` on its first step. Importing the module as a
    library ran the whole body first, which is why every existing test passed and the only
    training command in the repo had never completed a run.

    So this test must spawn the interpreter. Calling ``main([...])`` in-process would import the
    module and reproduce nothing.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "nyxara.njp.study", "--limit", "40", "--exam", "4"],
        capture_output=True, text=True, timeout=600,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert proc.returncode == 0, f"CLI exited {proc.returncode}\nstderr:\n{proc.stderr[-2000:]}"
    assert "NameError" not in proc.stderr, proc.stderr[-2000:]
    assert "Traceback" not in proc.stderr, proc.stderr[-2000:]
    # It must have got past seeding and through both exams, not merely exited cleanly.
    assert "studying" in proc.stdout
    assert "exam AFTER studying" in proc.stdout


def test_every_name_the_entry_point_needs_is_bound_before_the_guard():
    """The structural form of the same bug, checked without spawning anything.

    ``main`` is only reachable from the guard, so any module-level name it can reach at runtime
    must be defined above that guard. This catches a re-introduction the moment someone appends a
    new helper to the end of the file and wires it into ``Tutor``.
    """
    source = Path(study.__file__).read_text(encoding="utf-8")
    guard = source.index('if __name__ == "__main__":')
    for name in ("seed_kinds", "SeedReport", "Tutor", "Corpus", "main"):
        defined = max(source.rfind(f"\ndef {name}("), source.rfind(f"\nclass {name}"))
        assert defined != -1, f"{name} is not defined at module level"
        assert defined < guard, f"{name} is defined below the __main__ guard — `-m` will NameError"
