"""Tests for F6 — programmatic morphogenesis / self-evolving grammar (growth/grammar.py).

A task the base first-order grammar cannot express must be UNSOLVABLE until she invents the meta-type
that captures it — and the extension is adopted only on a strict capability gain. No LLM anywhere.
"""

from __future__ import annotations

import random

import pytest

from nyxara.growth.grammar import (
    GrammarEvolver,
    TypeExtension,
    _pair_extension,
    install_extension,
    is_installed,
    meta_tasks,
)
from nyxara.growth.noesis import NoesisEngine, base_library, synthesize


def test_pair_task_unsolvable_in_base_grammar():
    tasks = meta_tasks(random.Random(0), 3)
    base = base_library()
    assert all(not synthesize(t, base).solved for t in tasks)


def test_installing_the_pair_type_makes_it_solvable():
    tasks = meta_tasks(random.Random(0), 3)
    lib = base_library()
    install_extension(lib, _pair_extension())
    assert all(synthesize(t, lib).solved for t in tasks)


def test_evolver_adopts_only_on_a_strict_capability_gain():
    tasks = meta_tasks(random.Random(1), 4)
    lib = base_library()
    adopted = GrammarEvolver().evolve(lib, tasks)
    assert "product_type" in adopted
    assert is_installed(lib, _pair_extension())
    # a second call adopts nothing new (already installed) — no thrash
    assert GrammarEvolver().evolve(lib, tasks) == []


def test_no_extension_when_base_grammar_suffices():
    # ordinary list→int tasks are already expressible → nothing to invent
    from nyxara.growth.noesis import synthetic_tasks
    tasks = synthetic_tasks(random.Random(0), 6)
    lib = base_library()
    assert GrammarEvolver().evolve(lib, tasks) == []


def test_extension_cannot_touch_the_character_core():
    with pytest.raises(ValueError):
        TypeExtension("loyalty_type", {"loyalty": lambda v: v})


def test_engine_invents_a_meta_type_when_it_needs_one():
    eng = NoesisEngine(seed=1, tasks_per_cycle=10, grammar=GrammarEvolver())
    reports = eng.run(4)
    # over the run she extended her grammar at least once and installed the pair type
    assert any(r.grammar_extended > 0 for r in reports)
    assert "pair" in eng.library.extra_types
