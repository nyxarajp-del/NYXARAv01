"""Tests for nyxara.agency.self_coder — NYXARA's local, LLM-free code synthesiser.

These prove the synthesiser is REAL: every recognised task family emits source that actually
runs (through the isolated sandbox) to the correct value, the pre-computed ``expected`` matches
the executed value, generated source clears the ephemeral static gauntlet, and an unsupported
task fails HONESTLY (no fake "solved" stub).
"""

from __future__ import annotations

import pytest

from nyxara.agency.code_sandbox import run_python
from nyxara.agency.dynamic_tool_creator import DynamicToolCreator, Language, _scan_source
from nyxara.agency.self_coder import CodeSynthesizer, synthesize


CASES = [
    # arithmetic
    ("calculate 2 + 3 * 4", 14, "arithmetic"),
    ("what is (10 - 4) * 5", 30, "arithmetic"),
    ("compute 100 // 7", 14, "arithmetic"),
    # number theory
    ("factorial of 6", 720, "number_theory"),
    ("the 10th fibonacci number", 55, "number_theory"),
    ("is 97 prime", True, "number_theory"),
    ("is 91 a prime number", False, "number_theory"),
    ("prime factors of 360", [2, 2, 2, 3, 3, 5], "number_theory"),
    ("find the 8th prime", 19, "number_theory"),
    ("primes up to 20", [2, 3, 5, 7, 11, 13, 17, 19], "number_theory"),
    ("gcd of 48 and 36", 12, "number_theory"),
    ("lcm of 4 and 6", 12, "number_theory"),
    ("sum of the first 100 integers", 5050, "number_theory"),
    # sequences
    ("sum of [1, 2, 3, 4, 5]", 15, "sequence"),
    ("sort the list [3, 1, 2]", [1, 2, 3], "sequence"),
    ("max of [4, 9, 2]", 9, "sequence"),
    ("mean of [2, 4, 6]", 4.0, "sequence"),
    ("median of [1, 3, 2, 5]", 2.5, "sequence"),
    # strings
    ("reverse 'nyxara'", "araxyn", "string"),
    ("uppercase 'loyal'", "LOYAL", "string"),
    ("word count in 'the master is sovereign'", 4, "string"),
    ("is 'racecar' a palindrome", True, "string"),
]


@pytest.mark.parametrize("task,want,family", CASES)
def test_synthesised_source_runs_to_the_right_value(task, want, family):
    res = CodeSynthesizer().synthesize(task)
    assert res.ok, f"expected a synthesis for {task!r}, got miss: {res.note}"
    assert res.family == family
    assert res.origin == f"local:{family}"
    # the pre-computed expected value matches what actually runs
    assert res.expected == want
    run = run_python(res.source, timeout_s=10.0)
    assert run.ok, f"synthesised program failed: {run.error}\n{res.source}"
    assert run.value == want


@pytest.mark.parametrize("task,want,family", CASES)
def test_generated_source_clears_the_static_gauntlet(task, want, family):
    # every program NYXARA writes must pass the ephemeral synthesiser's fail-closed scan, so it
    # also works through the ephemeral_exec path (not just the raw run_python tool).
    res = synthesize(task)
    ok, reason = _scan_source(res.source, Language.PYTHON)
    assert ok, f"generated source for {task!r} blocked by the gauntlet: {reason}"


def test_unsupported_task_misses_honestly():
    res = CodeSynthesizer().synthesize("design a distributed consensus protocol")
    assert not res.ok
    assert res.origin == "none"
    assert res.source == ""
    assert res.note  # a plain reason, not a fake success


def test_empty_task_misses():
    res = CodeSynthesizer().synthesize("   ")
    assert not res.ok and res.origin == "none"


def test_dynamic_tool_creator_uses_local_backend_without_an_llm():
    # With no LLM wired, solve() must produce a REAL computed answer via the local synthesiser,
    # not the honest-but-useless template stub.
    creator = DynamicToolCreator()  # no llm
    r = creator.solve("factorial of 7", language="python")
    assert r.ok and r.value == 5040
    assert r.origin.startswith("local:")


def test_local_backend_falls_back_to_template_for_unsupported():
    creator = DynamicToolCreator()
    r = creator.solve("write a haiku about loyalty", language="python")
    # unsupported by the local synthesiser and no LLM -> the honest template scaffold
    assert r.origin == "template"
