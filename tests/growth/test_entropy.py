"""Tests for nyxara.growth.entropy — L-ENTROPY, and the two rules that must never break."""

from __future__ import annotations

from random import Random

from nyxara.growth.entropy import (OPERATORS, EntropyCurriculum, EntropyReport, PerturbSpec,
                                   perturb_corpus)
from nyxara.growth.foundry import Foundry
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.llm import format_self_training_doc

_PROSE = ("The river flows east past the mill. The stone is heavy and cold. "
          "Birds fly high above the valley in summer.")


def _foundry(tmp_path, **kw):
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"
    return Foundry(settings=settings, **kw)


# -------------------- rule 1: the eval split is never touched -------------------- #
def test_training_split_is_perturbed_and_eval_split_is_not(tmp_path, monkeypatch):
    """The gauntlet promotes on a strictly-lower perplexity. If the held-out text were noised,
    that gate would be measuring noise instead of skill — so the split must stay clean."""
    import nyxara.growth.foundry as foundry_mod

    f = _foundry(tmp_path)
    f.cfg.dataset_entropy = True
    f.cfg.entropy_max = 0.6

    seen = {}

    class _Stop(Exception):
        pass

    class _Spy:
        kind = "ngram"

        def train_on(self, texts, **kw):
            seen["train"] = list(texts)
            raise _Stop()

    monkeypatch.setattr(foundry_mod, "build_model", lambda spec: _Spy())
    corpus = [f"{_PROSE} document {i}" for i in range(20)]
    holdout_train, holdout_eval = f._holdout(corpus)

    try:
        f.train_candidate(corpus=corpus)
    except _Stop:
        pass

    # every eval document survives somewhere untouched: it was never handed to the perturber
    assert set(holdout_eval) & set(corpus) == set(holdout_eval)
    # and the training text she actually saw is not the clean training text
    assert seen["train"] != holdout_train


def test_entropy_disabled_leaves_the_corpus_identical(tmp_path):
    f = _foundry(tmp_path)
    f.cfg.dataset_entropy = False
    texts = [f"{_PROSE} {i}" for i in range(10)]
    out, report = f._apply_entropy(list(texts))
    assert out == texts and report is None


# -------------------- rule 2: a pair's answer is never corrupted -------------------- #
def test_supervised_answer_survives_byte_for_byte():
    """Corrupting the answer would teach her to produce damaged answers — the exact inverse of
    what this layer is for. Only the prompt half may be touched."""
    doc = format_self_training_doc("What is the capital of France, precisely speaking?", "Paris.")
    curriculum = EntropyCurriculum(level=1.0, seed=3)
    report = EntropyReport()

    out = curriculum.perturb(doc, level=1.0, rng=Random(3), report=report)

    answer_before = doc.split("### NYXARA:")[1]
    answer_after = out.split("### NYXARA:")[1]
    assert answer_after == answer_before
    assert out.split("### NYXARA:")[0] != doc.split("### NYXARA:")[0]
    assert report.answers_protected == 1


def test_prose_without_an_answer_tag_is_perturbed_whole():
    """No answer marker means nothing to protect — the whole document is fair game."""
    curriculum = EntropyCurriculum(level=0.9, operators=["token_dropout"], seed=5)
    report = EntropyReport()
    out = curriculum.perturb(_PROSE, level=0.9, rng=Random(5), report=report)
    assert out != _PROSE
    assert report.answers_protected == 0


# -------------------- determinism -------------------- #
def test_same_seed_gives_the_same_corpus():
    texts = [f"{_PROSE} {i}" for i in range(12)]
    a = EntropyCurriculum(level=0.3, seed=7).perturb_corpus(texts)
    b = EntropyCurriculum(level=0.3, seed=7).perturb_corpus(texts)
    assert a == b


def test_different_seed_gives_a_different_corpus():
    texts = [f"{_PROSE} {i}" for i in range(12)]
    a = EntropyCurriculum(level=0.3, seed=7).perturb_corpus(texts)
    b = EntropyCurriculum(level=0.3, seed=8).perturb_corpus(texts)
    assert a != b


# -------------------- schedule -------------------- #
def test_level_anneals_upward_across_training():
    curriculum = EntropyCurriculum(level=0.4, seed=0)
    early = curriculum.schedule(step=0, total=100, competence=1.0)
    late = curriculum.schedule(step=99, total=100, competence=1.0)
    assert early < late <= 0.4


def test_a_weak_brain_gets_clean_data():
    """Competence scales the level: a brain that cannot yet model clean text is not helped by
    being handed broken text."""
    curriculum = EntropyCurriculum(level=0.4, seed=0)
    assert curriculum.schedule(step=99, total=100, competence=0.0) == 0.0
    assert curriculum.schedule(step=99, total=100, competence=1.0) > 0.0


def test_zero_level_is_a_no_op():
    texts = [f"{_PROSE} {i}" for i in range(6)]
    assert EntropyCurriculum(level=0.0, seed=0).perturb_corpus(texts) == texts


# -------------------- operators -------------------- #
def test_every_named_operator_can_change_text():
    """A silently inert operator would make the layer look like it is working when it is not."""
    numeric = "values 10 and 250 and 3.5 measured across 42 trials of the experiment"
    for name in OPERATORS:
        source = numeric if name == "numeric_jitter" else _PROSE
        curriculum = EntropyCurriculum(level=0.5, operators=[name], seed=11)
        changed = any(curriculum.perturb(source, level=0.5, rng=Random(s)) != source
                      for s in range(12))
        assert changed, f"operator {name!r} never changed its input"


def test_unknown_operators_are_ignored():
    curriculum = EntropyCurriculum(level=0.5, operators=["not_an_operator", "token_dropout"])
    assert curriculum.operators == ("token_dropout",)


def test_numeric_jitter_keeps_sign_and_magnitude():
    curriculum = EntropyCurriculum(level=0.1, operators=["numeric_jitter"], seed=2)
    out = curriculum.perturb("the reading was 1000 units", level=0.1, rng=Random(2))
    value = int("".join(ch for ch in out if ch.isdigit()) or 0)
    assert 800 <= value <= 1200


def test_short_text_is_left_alone():
    """Perturbing a two-word document destroys it rather than degrading it."""
    curriculum = EntropyCurriculum(level=1.0, operators=["token_dropout"], seed=0)
    assert curriculum.perturb("hi there", level=1.0, rng=Random(0)) == "hi there"


# -------------------- reporting -------------------- #
def test_report_counts_what_actually_happened():
    texts = [f"{_PROSE} {i}" for i in range(20)]
    report = EntropyReport()
    EntropyCurriculum(level=0.3, seed=1).perturb_corpus(texts, report=report)
    assert report.documents == 20
    assert 0 < report.perturbed <= 20
    assert sum(report.applied.values()) == report.perturbed
    assert "eval split untouched" in report.summary()


def test_wrapper_never_raises_and_respects_the_config_switch():
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.foundry.dataset_entropy = False
    texts = [_PROSE]
    assert perturb_corpus(texts, settings=settings) == texts


def test_perturb_spec_disabled_when_level_zero():
    assert not PerturbSpec(level=0.0).enabled()
    assert PerturbSpec(level=0.2).enabled()
