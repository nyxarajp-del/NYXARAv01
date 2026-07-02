"""Tests for nyxara.cognition.skill_induction — few-shot skill acquisition + its wiring.

These prove the capability the Master asked for: NYXARA learns a *task* from a few
demonstrations, verifies it, and applies it to a genuinely new input — herself, no LLM.
"""

from __future__ import annotations

from nyxara.cognition.sample_efficient import SampleEfficientMind
from nyxara.cognition.skill_induction import Operation, SkillInductionEngine
from nyxara.memory.store import MemoryStore


def _engine(store=None):
    return SkillInductionEngine(getattr(store, "embedder", None), store=store)


# ----------------------- induction + generalization ----------------------- #
def test_char_reversal_generalizes_from_two_demos():
    eng = _engine()
    skill = eng.induce("reverse", [("hello", "olleh"), ("world", "dlrow")])
    assert skill is not None
    # the whole point: it works on an input it was NEVER shown
    assert skill.apply("nyxara") == "araxyn"


def test_case_transform_generalizes():
    eng = _engine()
    skill = eng.induce("shout", [("hi there", "HI THERE"), ("good day", "GOOD DAY")])
    assert skill is not None
    assert skill.apply("brave new world") == "BRAVE NEW WORLD"


def test_arithmetic_task_is_learned_and_held_out_generalization_measured():
    eng = _engine()
    skill = eng.induce("double", [("2", "4"), ("3", "6"), ("10", "20")])
    assert skill is not None
    assert skill.apply("7") == "14"                      # a new number
    assert skill.generalized is True                     # 3 demos → transfer is measured


def test_additive_arithmetic():
    eng = _engine()
    skill = eng.induce("plus_five", [("1", "6"), ("10", "15")])
    assert skill is not None and skill.apply("100") == "105"


def test_slot_template_is_the_key_generalizer():
    eng = _engine()
    skill = eng.induce("greet", [("greet jp", "hello jp !"), ("greet sky", "hello sky !")])
    assert skill is not None
    assert skill.apply("greet master") == "hello master !"


def test_affix_wrap():
    eng = _engine()
    skill = eng.induce("quote", [("cat", "<cat>"), ("dog", "<dog>")])
    assert skill is not None and skill.apply("bird") == "<bird>"


def test_composed_program_two_steps():
    eng = _engine()
    # lower THEN reverse — requires composition depth ≥ 1
    skill = eng.induce("lowrev", [("ABC", "cba"), ("HeLLo", "olleh")])
    assert skill is not None
    assert skill.apply("WORLD") == "dlrow"
    assert len(skill.program) >= 1


# ----------------------------- verification (honesty) ---------------------- #
def test_rejects_unlearnable_mapping():
    eng = _engine()
    # no consistent transformation explains these → she refuses to learn (no bluff)
    assert eng.induce("nonsense", [("a", "zzz"), ("b", "qqq"), ("c", "mmm")]) is None


def test_requires_minimum_demos():
    eng = SkillInductionEngine(min_demos=2)
    assert eng.induce("x", [("a", "b")]) is None          # one demo is below the floor


def test_every_demo_must_be_reproduced():
    eng = _engine()
    # first two are reversal, the third is not — no single program fits all three
    assert eng.induce("mixed", [("ab", "ba"), ("cd", "dc"), ("ef", "zz")]) is None


# --------------------------- application / matching ------------------------ #
def test_solve_fires_on_anchored_trigger_only():
    eng = _engine()
    eng.induce("greet", [("greet jp", "hello jp !"), ("greet sky", "hello sky !")])
    hit = eng.solve("greet nyxara")
    assert hit is not None and hit[0] == "hello nyxara !"


def test_solve_never_hijacks_unrelated_turn():
    eng = _engine()
    eng.induce("reverse", [("abc", "cba"), ("de", "ed")])
    # a bare transform has no trigger phrase → it must NOT auto-fire on an unrelated question
    assert eng.solve("what is the capital of France") is None


def test_apply_skill_is_unrestricted():
    eng = _engine()
    eng.induce("reverse", [("abc", "cba"), ("de", "ed")])
    # explicit application generalizes fully, no matching gate
    assert eng.apply_skill("reverse", "wxyz") == ("zyxw", eng.skills()[0].confidence)


# ------------------------------- persistence ------------------------------- #
def test_skill_persists_across_engines():
    store = MemoryStore()
    e1 = _engine(store)
    e1.induce("reverse", [("abc", "cba"), ("de", "ed")])
    e2 = _engine(store)                                   # a fresh engine over the same store
    assert any(s.name == "reverse" for s in e2.skills())


def test_operation_roundtrip():
    op = Operation("affix", {"prefix": "<", "suffix": ">"})
    assert Operation.from_dict(op.to_dict()).apply("x") == "<x>"


# --------------------------- SampleEfficientMind seam ---------------------- #
def test_sample_efficient_teach_and_solve_skill():
    mind = SampleEfficientMind(store=MemoryStore())
    mind.teach("greet", [("greet jp", "hello jp !"), ("greet sky", "hello sky !")], kind="skill")
    solved = mind.solve("greet nyxara")
    assert solved is not None and solved[0] == "hello nyxara !"


def test_sample_efficient_as_prompt_surfaces_skill():
    mind = SampleEfficientMind(store=MemoryStore())
    mind.teach("greet", [("greet jp", "hello jp !"), ("greet sky", "hello sky !")], kind="skill")
    block = mind.as_prompt("greet master")
    assert "hello master !" in block


def test_consolidate_mines_episodic_pairs_into_a_skill():
    mind = SampleEfficientMind(store=MemoryStore())
    # accumulated one-shot demonstrations that share ONE transformation (uppercase)
    mind.remember_once("hello world", "HELLO WORLD")
    mind.remember_once("good day", "GOOD DAY")
    mind.remember_once("bright sky", "BRIGHT SKY")
    report = mind.consolidate()
    assert report.skills_induced >= 1
    assert "episodic_transform" in report.skills


# ------------------------------- reasoner wiring --------------------------- #
def test_reasoner_answers_novel_input_from_learned_skill():
    from nyxara.mind.nyxara_reasoner import NyxaraReasoner
    store = MemoryStore()
    r = NyxaraReasoner(memory=store)
    r._sample_efficient_mind().teach(
        "greet", [("greet jp", "hello jp !"), ("greet sky", "hello sky !")], kind="skill")
    ans = r._skill_answer("greet master")
    assert ans is not None and ans[0] == "hello master !"


def test_disabled_flag_silences_skills(monkeypatch):
    from nyxara.kernel.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s.foundry, "skill_induction_enabled", False, raising=False)
    mind = SampleEfficientMind(store=MemoryStore(), settings=s)
    assert mind.learn_skill("greet", [("greet a", "hi a"), ("greet b", "hi b")]) is None
    assert mind.solve("greet c") is None
