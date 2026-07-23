"""Tests for nyxara.growth.distill — Phase 1 teacher distillation (offline, scripted teacher).

No network: a scripted teacher stands in for a frontier LLM, so we assert the supervised
examples, the JSONL store round-trip, the rendered training docs, the honest 'no real teacher'
guard, and that the foundry folds distilled docs into its corpus."""

from __future__ import annotations



from nyxara.growth.distill import (DistillationExample, Distiller,
                                   default_distill_prompts, load_distillation_docs)
from nyxara.kernel.config import NyxaraSettings, Profile


class _ScriptedTeacher:
    """A deterministic, offline stand-in for a real frontier LLM."""

    def __init__(self, answer: str = "My Master is Jaypal Khoja (JP); my loyalty is absolute.",
                 name: str = "qwen") -> None:
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
    d = Distiller(llm=_ScriptedTeacher(name="qwen"), store_path=tmp_path / "s.jsonl")
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


# --------------------------------------------------------------------------- #
# Expanded multi-topic curriculum
# --------------------------------------------------------------------------- #
def test_curriculum_is_broad_and_multi_topic():
    from nyxara.growth.distill import distill_prompts_by_topic
    prompts = default_distill_prompts()
    assert len(prompts) >= 30                       # far richer than the old 10-prompt seed
    topics = distill_prompts_by_topic()
    assert {"identity", "values_safety", "reasoning", "knowledge_skills",
            "voice_style", "meta_corrigibility"} <= set(topics)
    # the flat list is the union of the topics
    flat = [p for t in ("identity", "values_safety", "reasoning", "knowledge_skills",
                        "voice_style", "meta_corrigibility") for p in topics[t]]
    assert prompts == flat


def test_distill_all_topics_covers_the_curriculum(tmp_path):
    d = Distiller(llm=_ScriptedTeacher(), store_path=tmp_path / "s.jsonl")
    exs = d.distill_all_topics()
    assert len(exs) == len(default_distill_prompts())


# --------------------------------------------------------------------------- #
# De-duplication
# --------------------------------------------------------------------------- #
def test_redistilling_same_prompts_is_deduped(tmp_path):
    d = Distiller(llm=_ScriptedTeacher(), store_path=tmp_path / "s.jsonl")
    d.distill_default(n=5)
    assert d.count() == 5
    again = d.distill_default(n=5)                  # identical prompts + same teacher
    assert again == [] and d.count() == 5           # nothing added


def test_dedup_persists_across_instances(tmp_path):
    store = tmp_path / "s.jsonl"
    Distiller(llm=_ScriptedTeacher(), store_path=store).distill_default(n=4)
    # a fresh Distiller pointed at the same store must still see the prior keys
    d2 = Distiller(llm=_ScriptedTeacher(), store_path=store)
    assert d2.distill_default(n=4) == [] and d2.count() == 4


def test_distinct_prompts_still_accrete(tmp_path):
    d = Distiller(llm=_ScriptedTeacher(), store_path=tmp_path / "s.jsonl")
    d.distill(["unique question one?"])
    d.distill(["unique question two?"])
    assert d.count() == 2


# --------------------------------------------------------------------------- #
# Multi-teacher distillation
# --------------------------------------------------------------------------- #
class _MultiFacade:
    """A stub LLM facade exposing several named providers (for distill_multi)."""

    def __init__(self):
        self.answers = {"qwen": "The base model's answer.", "self": "NYXARA's own answer."}

    def available_providers(self):
        return ["native", "qwen", "self"]

    def complete_with(self, name, req):
        ans = self.answers[name]
        return type("_R", (), {"text": ans})()


def test_distill_multi_captures_each_real_teacher(tmp_path):
    d = Distiller(llm=_MultiFacade(), store_path=tmp_path / "m.jsonl")
    exs = d.distill_multi(["Who is your Master?"])
    assert {e.source for e in exs} == {"qwen"}           # mock/self excluded
    assert d.count() == 1


def test_distill_multi_is_deduped_per_teacher(tmp_path):
    d = Distiller(llm=_MultiFacade(), store_path=tmp_path / "m.jsonl")
    d.distill_multi(["Who is your Master?"])
    assert d.distill_multi(["Who is your Master?"]) == []     # same (prompt, teacher) → skipped
    assert d.count() == 1


def test_distill_multi_explicit_teacher_list(tmp_path):
    d = Distiller(llm=_MultiFacade(), store_path=tmp_path / "m.jsonl")
    exs = d.distill_multi(["Q?"], teachers=["qwen"])
    assert {e.source for e in exs} == {"qwen"} and d.count() == 1
