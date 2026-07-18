"""Tests for nyxara.cognition.program_library — the unified discrete program library.

These prove the DreamCoder-style thesis the Master asked for: verified programs from
skill induction, genesis, cognitive architect, and skill memory all compile into ONE
permanent symbolic store; the kernel reuses them instead of re-deriving answers; and
compositional generalization comes from the library (reuse + composition), not from
repeated search or an LLM.
"""

from __future__ import annotations

import time

import pytest

from nyxara.cognition.composition import CompositionalGrammar
from nyxara.cognition.program_library import (
    DiscreteProgramLibrary, Program, ProgramDomain, ProgramTestCase, compute_task_signature,
)
from nyxara.cognition.skill_induction import Operation, SkillInductionEngine
from nyxara.growth.skill_memory import Skill, SkillMemory
from nyxara.memory.store import MemoryStore
from nyxara.mind.lot import Lambda, const, func, var


def _reverse_program():
    return {"program": [Operation("reverse_chars").to_dict()]}


# --------------------------------------------------------------------------- #
# propose / persist / retrieve
# --------------------------------------------------------------------------- #
def test_propose_verifies_and_rejects_a_false_program():
    lib = DiscreteProgramLibrary()
    bad = lib.propose(
        name="broken", domain=ProgramDomain.STRING_PROGRAM, body=_reverse_program(),
        signature="string:broken", source="test",
        test_cases=[ProgramTestCase(inputs={"text": "ab"}, expected="NOT_REVERSED")])
    assert bad is None


def test_propose_persists_and_survives_restart(tmp_path):
    store = MemoryStore()
    lib = DiscreteProgramLibrary(store=store)
    tcs = [ProgramTestCase(inputs={"text": "hello"}, expected="olleh")]
    program = lib.propose(name="reverse", domain=ProgramDomain.STRING_PROGRAM,
                          body=_reverse_program(), signature="string:reverse",
                          source="test", test_cases=tcs)
    assert program is not None

    path = str(tmp_path / "mem.json")
    store.save(path)

    store2 = MemoryStore()
    store2.load(path)
    lib2 = DiscreteProgramLibrary(store=store2)
    got = lib2.retrieve("string:reverse")
    assert got and got[0].name == "reverse"


def test_propose_dedups_structurally_identical_programs():
    lib = DiscreteProgramLibrary()
    tcs = [ProgramTestCase(inputs={"text": "ab"}, expected="ba")]
    p1 = lib.propose(name="rev1", domain=ProgramDomain.STRING_PROGRAM, body=_reverse_program(),
                     signature="string:a", source="test", test_cases=tcs)
    p2 = lib.propose(name="rev2", domain=ProgramDomain.STRING_PROGRAM, body=_reverse_program(),
                     signature="string:b", source="test", test_cases=tcs)
    assert p1.program_id == p2.program_id
    assert len(lib) == 1


# --------------------------------------------------------------------------- #
# try_apply — the fast path
# --------------------------------------------------------------------------- #
def test_try_apply_fast_path_reuses_without_reverification():
    lib = DiscreteProgramLibrary()
    tcs = [ProgramTestCase(inputs={"text": "ab"}, expected="ba")]
    program = lib.propose(name="reverse", domain=ProgramDomain.STRING_PROGRAM,
                          body=_reverse_program(), signature="string:reverse",
                          source="test", test_cases=tcs)
    hit = lib.try_apply({"text": "nyxara"}, task_signature="string:reverse")
    assert hit is not None
    out, got_program, confidence = hit
    assert out == "araxyn" and got_program.program_id == program.program_id
    assert got_program.uses == 1 and got_program.successes == 1


def test_try_apply_misses_cleanly_when_nothing_matches():
    lib = DiscreteProgramLibrary()
    assert lib.try_apply({"text": "anything"}, task_signature="string:nope") is None


# --------------------------------------------------------------------------- #
# try_compose — real compositional generalization (the acid test)
# --------------------------------------------------------------------------- #
def test_composition_of_two_lexemes_solves_a_novel_phrase():
    store = MemoryStore()
    composer = CompositionalGrammar(store=store)
    composer.learn("walk", const("WALK"))
    composer.learn("twice", Lambda(var("a"), func("seq", var("a"), var("a"))))
    lib = DiscreteProgramLibrary(store=store, composer=composer)

    out = lib.try_compose({"text": "walk twice"}, task_signature="string:walk_twice")
    assert out is not None
    text, program = out
    assert text == "seq(WALK, WALK)"
    assert program.domain == ProgramDomain.COMPOSED

    # a second identical query is now a DIRECT try_apply hit, not re-composed
    hit = lib.try_apply({"text": "walk twice"}, task_signature="string:walk_twice")
    assert hit is not None and hit[0] == "seq(WALK, WALK)" and hit[1].program_id == program.program_id


def test_composition_pipeline_chains_two_string_programs_verified_against_both_parents():
    lib = DiscreteProgramLibrary()
    p1 = lib.propose(name="rev", domain=ProgramDomain.STRING_PROGRAM, body=_reverse_program(),
                     signature="string:rev", source="test",
                     test_cases=[ProgramTestCase(inputs={"text": "ab"}, expected="ba"),
                                 ProgramTestCase(inputs={"text": "cd"}, expected="dc")])
    p2 = lib.propose(name="up", domain=ProgramDomain.STRING_PROGRAM,
                     body={"program": [Operation("upper").to_dict()]},
                     signature="string:up", source="test",
                     test_cases=[ProgramTestCase(inputs={"text": "ba"}, expected="BA"),
                                 ProgramTestCase(inputs={"text": "dc"}, expected="DC")])
    assert p1 is not None and p2 is not None

    out = lib.try_compose({"text": "ab"}, task_signature="string:ab_composed")
    assert out is not None
    text, program = out
    assert text == "BA"          # reverse("ab") -> "ba", then upper("ba") -> "BA"
    assert set(program.composed_from) == {p1.program_id, p2.program_id}


# --------------------------------------------------------------------------- #
# search_for_solution — bounded UCT search finds a multi-step pipeline
# --------------------------------------------------------------------------- #
def test_search_for_solution_finds_multistep_pipeline():
    lib = DiscreteProgramLibrary(search_max_rollouts=200, search_max_depth=4)
    lib.propose(name="rev", domain=ProgramDomain.STRING_PROGRAM,
               body={"program": [Operation("reverse_chars").to_dict()]},
               signature="string:rev", source="test",
               test_cases=[ProgramTestCase(inputs={"text": "ab"}, expected="ba")])
    lib.propose(name="up", domain=ProgramDomain.STRING_PROGRAM,
               body={"program": [Operation("upper").to_dict()]},
               signature="string:up", source="test",
               test_cases=[ProgramTestCase(inputs={"text": "ab"}, expected="AB")])
    lib.propose(name="dedup", domain=ProgramDomain.STRING_PROGRAM,
               body={"program": [Operation("dedup_words").to_dict()]},
               signature="string:dedup", source="test",
               test_cases=[ProgramTestCase(inputs={"text": "a a"}, expected="a")])

    target = Operation("dedup_words").apply(
        Operation("upper").apply(Operation("reverse_chars").apply("cab cab")))

    def verify(candidate: str) -> float:
        return 1.0 if candidate == target else 0.0

    result = lib.search_for_solution({"text": "cab cab"}, task_signature="string:search",
                                     verify=verify)
    assert result is not None
    out, program = result
    assert out == target
    assert len(program.composed_from) >= 2

    # a repeat is now a direct try_apply hit — the search cost is paid once
    hit = lib.try_apply({"text": "cab cab"}, task_signature="string:search")
    assert hit is not None and hit[0] == target


def test_search_for_solution_disabled_returns_none():
    lib = DiscreteProgramLibrary(search_enabled=False)
    lib.propose(name="rev", domain=ProgramDomain.STRING_PROGRAM,
               body={"program": [Operation("reverse_chars").to_dict()]},
               signature="string:rev", source="test",
               test_cases=[ProgramTestCase(inputs={"text": "ab"}, expected="ba")])
    lib.propose(name="up", domain=ProgramDomain.STRING_PROGRAM,
               body={"program": [Operation("upper").to_dict()]},
               signature="string:up", source="test",
               test_cases=[ProgramTestCase(inputs={"text": "ab"}, expected="AB")])
    assert lib.search_for_solution({"text": "ab"}, task_signature="string:x") is None


# --------------------------------------------------------------------------- #
# promotion into the real ToolRegistry
# --------------------------------------------------------------------------- #
def test_promote_to_tool_and_invoke_through_the_real_safety_pipeline():
    from nyxara.agency.tools import ToolRegistry

    lib = DiscreteProgramLibrary(promote_min_uses=3, promote_min_success_rate=0.5)
    program = lib.propose(name="reverse", domain=ProgramDomain.STRING_PROGRAM,
                          body=_reverse_program(), signature="string:reverse", source="test",
                          test_cases=[ProgramTestCase(inputs={"text": "ab"}, expected="ba")])
    assert program is not None
    for _ in range(3):
        lib.reinforce(program.program_id, success=True)

    registry = ToolRegistry()
    spec = lib.promote_to_tool(program.program_id, registry)
    assert spec is not None
    assert registry.get(spec.name) is spec

    result = registry.invoke(spec.name, {"text": "abc"})
    assert result.ok and result.value == "cba"


def test_promote_to_tool_refuses_below_the_bar():
    from nyxara.agency.tools import ToolRegistry

    lib = DiscreteProgramLibrary(promote_min_uses=5)
    program = lib.propose(name="reverse", domain=ProgramDomain.STRING_PROGRAM,
                          body=_reverse_program(), signature="string:reverse", source="test",
                          test_cases=[ProgramTestCase(inputs={"text": "ab"}, expected="ba")])
    registry = ToolRegistry()
    assert lib.promote_to_tool(program.program_id, registry) is None


# --------------------------------------------------------------------------- #
# decay / pruning — self-curation
# --------------------------------------------------------------------------- #
def test_decay_and_prune_protects_reinforced_drops_unused():
    lib = DiscreteProgramLibrary(decay_rate=0.5)
    kept = lib.propose(name="reverse", domain=ProgramDomain.STRING_PROGRAM,
                       body=_reverse_program(), signature="string:reverse", source="test",
                       test_cases=[ProgramTestCase(inputs={"text": "ab"}, expected="ba")])
    dropped = lib.propose(name="upper", domain=ProgramDomain.STRING_PROGRAM,
                          body={"program": [Operation("upper").to_dict()]},
                          signature="string:upper", source="test",
                          test_cases=[ProgramTestCase(inputs={"text": "a"}, expected="A")])
    for _ in range(5):
        lib.reinforce(kept.program_id, success=True)
    dropped.last_used = time.time() - 999 * 86400
    dropped.confidence = 0.51

    lib.capacity = len(lib._programs) - 1
    lib.decay_tick(now=time.time())

    assert kept.program_id in lib._programs
    assert dropped.program_id not in lib._programs


def test_prune_never_drops_a_protected_or_depended_on_program():
    lib = DiscreteProgramLibrary()
    base = lib.propose(name="rev", domain=ProgramDomain.STRING_PROGRAM,
                       body={"program": [Operation("reverse_chars").to_dict()]},
                       signature="string:rev", source="test",
                       test_cases=[ProgramTestCase(inputs={"text": "ab"}, expected="ba")])
    composed = lib.propose(name="rev_wrapper", domain=ProgramDomain.COMPOSED,
                           body={"kind": "pipeline", "steps": [base.program_id]},
                           signature="string:wrapper", source="test",
                           composed_from=[base.program_id], already_verified=True)
    assert composed is not None
    base.last_used = time.time() - 999 * 86400
    base.confidence = 0.01
    lib.capacity = 1
    removed = lib.prune()
    assert base.program_id not in removed, "a program another still depends on must survive"


# --------------------------------------------------------------------------- #
# compress — real library learning
# --------------------------------------------------------------------------- #
def test_compress_factors_recurring_subsequence_and_consumers_still_pass():
    lib = DiscreteProgramLibrary(compression_min_frequency=3)
    shared = [Operation("strip").to_dict(), Operation("lower").to_dict()]
    consumers = []
    for i, extra_name in enumerate(["upper", "title", "swapcase"]):
        extra = Operation(extra_name)
        ops = shared + [extra.to_dict()]
        text = "  Hello World  "
        expected = extra.apply(Operation("lower").apply(Operation("strip").apply(text)))
        p = lib.propose(name=f"p{i}", domain=ProgramDomain.STRING_PROGRAM, body={"program": ops},
                        signature=f"string:p{i}", source="test",
                        test_cases=[ProgramTestCase(inputs={"text": text}, expected=expected)])
        assert p is not None
        consumers.append(p)

    report = lib.compress()
    assert report["factored"] >= 1
    components = [p for p in lib._programs.values() if p.is_component]
    assert components
    component = components[0]
    for p in consumers:
        assert component.program_id in p.composed_from
        # each consumer's OWN body is untouched — it still passes its original test case
        for tc in p.test_cases:
            assert lib._run_domain(p, tc.inputs) == tc.expected


# --------------------------------------------------------------------------- #
# migration from the pre-existing fragmented stores
# --------------------------------------------------------------------------- #
def test_migration_imports_skill_induction_and_skill_memory_records_idempotently(tmp_path):
    store = MemoryStore()
    eng = SkillInductionEngine(store=store)
    eng.induce("reverse", [("hello", "olleh"), ("world", "dlrow")])

    sm = SkillMemory(store=store)
    sm.learn("greet the Master", procedure="respond warmly", tools=[])

    lib = DiscreteProgramLibrary(store=store)
    counts = lib.import_legacy()
    assert counts["skill_induction"] >= 1
    assert counts["skill_memory"] >= 1

    again = lib.import_legacy()
    assert again == {"skipped": 1}


def test_migration_imports_genesis_json(tmp_path):
    import json

    gpath = tmp_path / "primitive_library.json"
    gpath.write_text(json.dumps({"counter": 1, "entries": [
        {"name": "synth_0", "steps": ["conv:3", "lowrank:4"]}]}))

    lib = DiscreteProgramLibrary(store=MemoryStore())
    counts = lib.import_legacy(genesis_path=str(gpath))
    assert counts["genesis"] == 1
    hits = [p for p in lib._programs.values() if p.domain == ProgramDomain.NN_MIXER]
    assert len(hits) == 1


# --------------------------------------------------------------------------- #
# lazy migration timing — must never race ahead of a later load_state-style restore
# --------------------------------------------------------------------------- #
def test_migration_is_lazy_not_eager_at_construction():
    store = MemoryStore()
    eng = SkillInductionEngine(store=store)
    eng.induce("reverse", [("hello", "olleh"), ("world", "dlrow")])

    lib = DiscreteProgramLibrary(store=store)   # migration must NOT have run yet
    assert "_migration_marker" not in lib._programs
    lib.retrieve("string:reverse")               # first real access triggers it lazily
    assert any(p.source_subsystem == "skill_induction-migration" for p in lib._programs.values())


# --------------------------------------------------------------------------- #
# hook wiring — producers actually populate the SHARED library, not just the module
# --------------------------------------------------------------------------- #
def test_skill_induction_hook_proposes_into_shared_library():
    lib = DiscreteProgramLibrary()
    eng = SkillInductionEngine(program_library=lib)
    eng.induce("reverse", [("hello", "olleh"), ("world", "dlrow")])
    hits = [p for p in lib._programs.values() if p.domain == ProgramDomain.STRING_PROGRAM]
    assert hits and hits[0].name == "reverse"


def test_skill_memory_hook_proposes_into_shared_library():
    lib = DiscreteProgramLibrary()
    sm = SkillMemory(program_library=lib)
    sm.learn("greet the Master warmly", procedure="respond warmly", tools=[])
    hits = [p for p in lib._programs.values() if p.domain == ProgramDomain.PROCEDURE]
    assert hits


def test_genesis_hook_proposes_into_shared_library():
    from nyxara.growth.genesis import MixerProgram, PrimitiveLibrary

    lib = DiscreteProgramLibrary()

    def on_add(name, prog):
        from nyxara.cognition.program_library import ProgramDomain as PD
        lib.propose(name=name, domain=PD.NN_MIXER, body={"steps": list(prog.steps)},
                   signature=f"nn_mixer:{len(prog.steps)}", source="genesis",
                   already_verified=True)

    plib = PrimitiveLibrary(path=None, capacity=10, on_add=on_add)
    plib.add(MixerProgram(steps=["conv:3", "lowrank:4"]))
    hits = [p for p in lib._programs.values() if p.domain == ProgramDomain.NN_MIXER]
    assert hits


def test_cognitive_architect_hook_proposes_into_shared_library():
    from nyxara.growth.cognitive_architect import CognitiveArchitect

    lib = DiscreteProgramLibrary()
    ca = CognitiveArchitect(seed=1, n_per_type=24, program_library=lib)
    for _ in range(20):
        ca.rewire(candidates=6)
    hits = [p for p in lib._programs.values() if p.domain == ProgramDomain.REASONING_OPERATOR]
    assert hits, "at least one invented, held-out-proven operator should have been proposed"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
