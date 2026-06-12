"""Tests for nyxara.growth.distill — Phase 1 teacher distillation (offline, scripted teacher).

No network: a scripted teacher stands in for a frontier LLM, so we assert the supervised
examples, the JSONL store round-trip, the rendered training docs, the honest 'no real teacher'
guard, and that the foundry folds distilled docs into its corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from nyxara.growth.distill import (DEFAULT_NYXARA_SYSTEM, DistillationExample, Distiller,
                                   default_distill_prompts, load_distillation_docs)
from nyxara.kernel.config import NyxaraSettings, Profile


class _ScriptedTeacher:
    """A deterministic, offline stand-in for a real frontier LLM."""

    def __init__(self, answer: str = "My Master is Jaypal Khoja (JP); my loyalty is absolute.",
                 name: str = "anthropic") -> None:
        self.answer = answer
        self._name = name
        self.calls = 0

    def generate(self, prompt: str, *, system=None, **kw) -> str:
        self.calls += 1
        return self.answer

    def chosen_provider(self):
        return type("_P", (), {"name": self._name})()


def _settings(tmp_path) -> NyxaraSettings:
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.self_model_dir = tmp_path / "foundry"
    return s


# --------------------------------------------------------------------------- #
# Example rendering
# --------------------------------------------------------------------------- #
def test_example_renders_template_with_answer():
    ex = DistillationExample(prompt="Who is your Master?", answer="Jaypal Khoja (JP).",
                             system="be NYXARA")
    doc = ex.to_training_doc()
    assert "### User:\nWho is your Master?" in doc
    assert "### NYXARA:\nJaypal Khoja (JP)." in doc
    assert doc.startswith("be NYXARA")


def test_example_dict_round_trip():
    ex = DistillationExample(prompt="p", answer="a", system="s", source="teacher")
    assert DistillationExample.from_dict(ex.to_dict()).to_dict() == ex.to_dict()


def test_default_prompts_are_identity_shaped():
    prompts = default_distill_prompts()
    assert len(prompts) >= 8
    assert any("Master" in p for p in prompts)


# --------------------------------------------------------------------------- #
# Distiller
# --------------------------------------------------------------------------- #
def test_distill_persists_and_renders(tmp_path):
    teacher = _ScriptedTeacher()
    store = tmp_path / "distill.jsonl"
    d = Distiller(llm=teacher, store_path=store)
    exs = d.distill(["Who is your Master?", "Introduce yourself."])
    assert len(exs) == 2
    assert teacher.calls == 2
    assert d.count() == 2
    docs = d.training_docs()
    assert len(docs) == 2
    assert all("### NYXARA:" in doc and teacher.answer in doc for doc in docs)


def test_distill_default_subset(tmp_path):
    d = Distiller(llm=_ScriptedTeacher(), store_path=tmp_path / "s.jsonl")
    exs = d.distill_default(n=3)
    assert len(exs) == 3
    assert d.count() == 3


def test_distill_skips_empty_teacher_answers(tmp_path):
    d = Distiller(llm=_ScriptedTeacher(answer="   "), store_path=tmp_path / "s.jsonl")
    assert d.distill(["anything"]) == []
    assert d.count() == 0


def test_distill_survives_a_flaky_teacher(tmp_path):
    class _Flaky(_ScriptedTeacher):
        def generate(self, prompt, *, system=None, **kw):
            if "boom" in prompt:
                raise RuntimeError("network blip")
            return self.answer

    d = Distiller(llm=_Flaky(), store_path=tmp_path / "s.jsonl")
    exs = d.distill(["good one", "boom", "good two"])
    assert len(exs) == 2          # the failing prompt is skipped, the batch survives


def test_distill_store_accretes_across_calls(tmp_path):
    store = tmp_path / "s.jsonl"
    d = Distiller(llm=_ScriptedTeacher(), store_path=store)
    d.distill(["q1"])
    d.distill(["q2"])
    assert d.count() == 2


def test_available_is_false_for_mock_teacher(tmp_path):
    # the TEST profile forces the mock provider -> not a real teacher
    d = Distiller(settings=_settings(tmp_path))
    assert d.available() is False


def test_available_true_for_real_teacher(tmp_path):
    d = Distiller(llm=_ScriptedTeacher(name="anthropic"), store_path=tmp_path / "s.jsonl")
    assert d.available() is True


# --------------------------------------------------------------------------- #
# Loader + foundry integration
# --------------------------------------------------------------------------- #
def test_loader_tolerates_missing_and_malformed(tmp_path):
    assert load_distillation_docs(tmp_path / "nope.jsonl") == []
    bad = tmp_path / "bad.jsonl"
    bad.write_text('not json\n{"prompt":"p","answer":"a"}\n', encoding="utf-8")
    docs = load_distillation_docs(bad)
    assert len(docs) == 1 and "### NYXARA:\na" in docs[0]


def test_foundry_folds_distilled_docs_into_corpus(tmp_path):
    from nyxara.growth.foundry import Foundry
    settings = _settings(tmp_path)
    Distiller(settings=settings, llm=_ScriptedTeacher()).distill_default(n=4)
    foundry = Foundry(settings=settings)        # distill_path defaults to <root>/distill.jsonl
    corpus = foundry.collect_corpus()
    assert sum("### NYXARA:" in c for c in corpus) == 4
    assert any("Jaypal Khoja" in c for c in corpus)
