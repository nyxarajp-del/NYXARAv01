"""Tests for nyxara.growth.selfplay — Phase 3 curiosity self-play (offline, scripted teacher)."""

from __future__ import annotations

import pytest

from nyxara.growth.distill import Distiller
from nyxara.growth.selfplay import CURIOSITY_SEEDS, SelfPlay, _parse_questions
from nyxara.kernel.config import NyxaraSettings, Profile


class _Teacher:
    """A scripted teacher: emits a question list for question-prompts, else an answer."""

    def generate(self, prompt: str, *, system=None, **kw) -> str:
        if "question per line" in prompt or "questions" in (system or "").lower():
            return "1. What is entropy?\n- Why is the sky blue?\nWhat is 12 * 12?\n\nnot a q"
        return "A clear, correct answer in NYXARA's own voice, Master."

    def chosen_provider(self):
        return type("_P", (), {"name": "anthropic"})()


def _sp(tmp_path, teacher=None) -> SelfPlay:
    teacher = teacher or _Teacher()
    d = Distiller(llm=teacher, store_path=tmp_path / "distill.jsonl")
    return SelfPlay(llm=teacher, distiller=d)


# --------------------------------------------------------------------------- #
# Question parsing
# --------------------------------------------------------------------------- #
def test_parse_questions_strips_numbering_and_keeps_questions():
    qs = _parse_questions("1. What is X?\n- Why Y?\n* How Z?\njust a statement\n")
    assert qs == ["What is X?", "Why Y?", "How Z?"]


def test_parse_questions_requires_question_mark():
    assert _parse_questions("This is a statement.\nAnd another.") == []


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_generate_questions_from_teacher(tmp_path):
    qs = _sp(tmp_path).generate_questions(3)
    assert len(qs) == 3
    assert all(q.endswith("?") for q in qs)


def test_generate_questions_falls_back_to_seeds_without_teacher(tmp_path):
    # the TEST profile forces the mock provider -> not a real teacher -> curiosity battery
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    sp = SelfPlay(settings=settings)
    assert sp.available() is False
    qs = sp.generate_questions(4)
    assert qs == list(CURIOSITY_SEEDS)[:4]


# --------------------------------------------------------------------------- #
# A full round
# --------------------------------------------------------------------------- #
def test_play_distils_questions_into_store(tmp_path):
    sp = _sp(tmp_path)
    rep = sp.play(3)
    assert rep["questions"] == 3 and rep["distilled"] == 3
    assert rep["store_size"] == 3
    assert sp.distiller.count() == 3


def test_play_feeds_the_foundry_corpus(tmp_path):
    from nyxara.growth.distill import load_distillation_docs
    sp = _sp(tmp_path)
    sp.play(2)
    docs = load_distillation_docs(sp.distiller.store_path)
    assert len(docs) == 2
    assert all("### NYXARA:" in d for d in docs)
