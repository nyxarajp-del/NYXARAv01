"""Tests for nyxara.growth.selfplay — Phase 3 curiosity self-play (offline, scripted teacher)."""

from __future__ import annotations


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
        return type("_P", (), {"name": "tinyllama"})()


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


# --------------------------------------------------------------------------- #
# Verifiable self-play — she answers her OWN structured problems exactly (no key needed)
# --------------------------------------------------------------------------- #
def _verifiable_sp(tmp_path):
    from nyxara.growth.flywheel import DataFlywheel
    fw = DataFlywheel(store_path=tmp_path / "flywheel.jsonl")
    return SelfPlay(flywheel=fw, seed=3), fw


def test_generate_verifiable_problems_are_oracle_solvable(tmp_path):
    from nyxara.mind.reasoning_chain import solve_chain
    from nyxara.mind.reasoning_faculties import solve_with_faculties
    sp, _ = _verifiable_sp(tmp_path)
    problems = sp.generate_verifiable(20)
    assert len(problems) == 20
    for cat, prompt, answer in problems:
        # every generated problem must be exactly solvable by a faculty/chain — no guessed targets
        solved = solve_chain(prompt) is not None or solve_with_faculties(prompt) is not None
        assert solved, f"{cat}: {prompt!r} was not oracle-solvable"
        assert "The answer is" in answer


def test_play_verifiable_collects_into_flywheel(tmp_path):
    sp, fw = _verifiable_sp(tmp_path)
    rep = sp.play_verifiable(24)
    assert rep["collected"] > 0
    assert rep["collected"] == fw.count()
    # the collected pairs render into foundry-ready training docs (the loop closes, keyless)
    docs = fw.training_docs()
    assert len(docs) == rep["collected"]
    assert all("### NYXARA:" in d for d in docs)


def test_verifiable_self_play_works_without_a_teacher(tmp_path):
    # the whole point: a keyless machine (no real teacher) still grows the corpus
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    sp, fw = _verifiable_sp(tmp_path)
    sp.settings = settings
    assert sp.available() is False              # no real teacher
    assert sp.play_verifiable(12)["collected"] > 0   # yet she still collects verified pairs
