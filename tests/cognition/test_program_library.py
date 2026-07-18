"""Tests for nyxara.cognition.program_library — the discrete program library.

These prove the DreamCoder-style claim: solved tasks that share a recurring program shape get
compressed into a *permanent* primitive that empowers *future* search — real compositional
generalization from a growing symbolic vocabulary, not from search budget (scale).
"""

from __future__ import annotations

from nyxara.cognition.program_library import Macro, PrimitiveScorer, ProgramLibrary
from nyxara.cognition.sample_efficient import SampleEfficientMind
from nyxara.cognition.skill_induction import SkillInductionEngine
from nyxara.memory.store import MemoryStore


def _shape_tasks(engine, min_support=3):
    """Solve 3 independent 2-step (reorder + wrap) tasks that share a program shape."""
    lib = ProgramLibrary(engine=engine, min_support=min_support)
    a = lib.induce("shout", [("hello world", "RESULT: world hello!"),
                             ("go now", "RESULT: now go!")])
    b = lib.induce("ask", [("foo bar", "RESULT: bar foo?"),
                           ("up down", "RESULT: down up?")])
    c = lib.induce("state", [("cat dog mouse", "RESULT: mouse dog cat."),
                             ("red blue", "RESULT: blue red.")])
    return lib, a, b, c


# --------------------------- wake: search is unaffected without a library --------------------- #
def test_induce_without_promoted_macros_matches_bare_engine():
    engine = SkillInductionEngine(max_depth=1, min_demos=2)
    lib = ProgramLibrary(engine=engine)
    skill = lib.induce("reverse", [("hello", "olleh"), ("world", "dlrow")])
    assert skill is not None and skill.apply("nyxara") == "araxyn"


# ------------------------- sleep-abstraction: compression is honest --------------------------- #
def test_shared_shape_across_independent_tasks_compresses():
    engine = SkillInductionEngine(max_depth=1, min_demos=2)
    lib, a, b, c = _shape_tasks(engine)
    assert a is not None and b is not None and c is not None
    shape = [op.name for op in a.program]
    assert shape == ["reverse_words", "affix"]

    report = lib.consolidate()
    assert report.grown
    assert len(lib.macros()) == 1
    macro = lib.macros()[0]
    assert macro.atom_prefix == ("reverse_words",)
    assert macro.fixed_params.get("prefix") == "RESULT: "
    assert "suffix" in macro.hole_param_keys        # suffix varied across tasks -> stays a hole


def test_below_support_threshold_does_not_compress():
    engine = SkillInductionEngine(max_depth=1, min_demos=2)
    lib = ProgramLibrary(engine=engine, min_support=3)
    lib.induce("shout", [("hello world", "RESULT: world hello!"), ("go now", "RESULT: now go!")])
    lib.induce("ask", [("foo bar", "RESULT: bar foo?"), ("up down", "RESULT: down up?")])
    # only 2 distinct tasks share the shape, but min_support requires 3
    report = lib.consolidate()
    assert not report.grown


def test_recompressing_the_same_pattern_is_idempotent():
    engine = SkillInductionEngine(max_depth=1, min_demos=2)
    lib, _a, _b, _c = _shape_tasks(engine)
    rep1 = lib.consolidate()
    assert rep1.grown
    rep2 = lib.consolidate()
    assert not rep2.grown, "the same pattern must not be promoted twice"
    assert len(lib.macros()) == 1


# ------------------------------------- the acid test -------------------------------------- #
def test_depth_zero_search_cannot_invent_a_two_step_program_alone():
    bare = SkillInductionEngine(max_depth=0, min_demos=2)
    unsolved = bare.induce("never_seen_bare",
                           [("m n", "RESULT: n m#"), ("j k", "RESULT: k j#")])
    assert unsolved is None


def test_library_lets_depth_zero_search_close_in_one_step():
    engine = SkillInductionEngine(max_depth=1, min_demos=2)
    lib, _a, _b, _c = _shape_tasks(engine)
    lib.consolidate()

    shallow_engine = SkillInductionEngine(max_depth=0, min_demos=2)
    shallow_lib = ProgramLibrary(engine=shallow_engine)
    for macro in lib.macros():
        shallow_lib.promote(macro)

    demos = [("m n", "RESULT: n m#"), ("j k", "RESULT: k j#")]
    solved = shallow_lib.induce("never_seen", demos)
    assert solved is not None
    assert solved.program[0].name == "macro"
    # generalizes to a HELD-OUT input never shown to induce() at all
    assert solved.apply("p q") == "RESULT: q p#"


# --------------------------------------- persistence --------------------------------------- #
def test_macro_survives_a_restart_and_empowers_a_fresh_engine():
    store = MemoryStore()
    e1 = SkillInductionEngine(max_depth=1, min_demos=2, store=store)
    lib1 = ProgramLibrary(engine=e1, store=store, min_support=2)
    lib1.induce("greetA", [("alpha beta", "LOUD: beta alpha!!"),
                           ("gamma delta", "LOUD: delta gamma!!")])
    lib1.induce("greetB", [("kappa lambda", "LOUD: lambda kappa??"),
                           ("sigma tau", "LOUD: tau sigma??")])
    assert lib1.consolidate().grown

    e2 = SkillInductionEngine(max_depth=0, min_demos=2, store=store)   # a fresh, shallow engine
    lib2 = ProgramLibrary(engine=e2, store=store)                      # a fresh library instance
    solved = lib2.induce("greetC", [("omega psi", "LOUD: psi omega::"),
                                    ("chi phi", "LOUD: phi chi::")])
    assert solved is not None and solved.program[0].name == "macro"
    assert solved.apply("nu xi") == "LOUD: xi nu::"


def test_macro_name_uniqueness_across_independent_libraries():
    # two SEPARATE ProgramLibrary instances must never mint colliding macro names in the
    # process-wide registry (that would silently corrupt one instance's primitive with another's)
    e1 = SkillInductionEngine(max_depth=1, min_demos=2)
    lib1, *_ = _shape_tasks(e1)
    lib1.consolidate()

    e2 = SkillInductionEngine(max_depth=1, min_demos=2)
    lib2, *_ = _shape_tasks(e2)
    lib2.consolidate()

    assert lib1.macros()[0].name != lib2.macros()[0].name


# ----------------------------------- learned search guide ----------------------------------- #
def test_primitive_scorer_reinforces_used_names():
    scorer = PrimitiveScorer()
    assert scorer.score("affix") == 0.0
    scorer.reinforce(["affix", "reverse_words"])
    assert scorer.score("affix") > 0.0
    assert scorer.score("reverse_words") > 0.0
    assert scorer.score("never_used") == 0.0


def test_macro_roundtrip():
    macro = Macro(name="m1", atom_prefix=("lower",), closing_op_name="affix",
                  param_keys=("prefix", "suffix"), fixed_params={"prefix": "<"},
                  hole_param_keys=("suffix",), provenance=["a", "b"], support=2)
    restored = Macro.from_dict(macro.to_dict())
    assert restored.atom_prefix == macro.atom_prefix
    assert restored.fixed_params == macro.fixed_params
    assert restored.hole_param_keys == macro.hole_param_keys


# --------------------------------- SampleEfficientMind seam ---------------------------------- #
def test_sample_efficient_mind_wires_the_program_library():
    mind = SampleEfficientMind(store=MemoryStore())
    assert mind.program_library is not None

    mind.teach("shout", [("hello world", "SAY: world hello!"), ("go now", "SAY: now go!")],
              kind="skill")
    mind.teach("ask", [("foo bar", "SAY: bar foo?"), ("up down", "SAY: down up?")], kind="skill")
    mind.teach("state", [("cat dog", "SAY: dog cat."), ("red blue", "SAY: blue red.")],
              kind="skill")

    report = mind.consolidate()
    assert report.macros_learned >= 1
    assert mind.stats()["macros"] >= 1
