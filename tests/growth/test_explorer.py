"""Tests for the Infinite Explorer — Environment-Driven Learning (growth/explorer.py).

Hermetic and fully offline: no network, no real LLM. A scripted coder stands in for a
frontier model where one is needed; everything else runs through the deterministic recipe
path and the real ``-I`` code sandbox.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest

from nyxara.growth.explorer import ExploreResult, InfiniteExplorer
from nyxara.growth.skill_memory import SkillMemory
from nyxara.knowledge.base import KnowledgeBase
from nyxara.memory.store import MemoryStore


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _ScriptedLLM:
    """Offline stand-in for a frontier coder: a crash first, then a correct fix."""

    def __init__(self, *scripts: str) -> None:
        self.scripts = list(scripts)
        self.calls = 0
        self.prompts: list = []

    def generate(self, prompt: str, **kw: Any) -> str:
        self.prompts.append(prompt)
        i = min(self.calls, len(self.scripts) - 1)
        self.calls += 1
        return self.scripts[i]


def _fresh():
    store = MemoryStore()
    return store, SkillMemory(store=store), KnowledgeBase(store=store, name="explorer")


# --------------------------------------------------------------------------- #
# can_attempt — only solvable-shaped tasks
# --------------------------------------------------------------------------- #
def test_can_attempt_filters_chitchat():
    x = InfiniteExplorer(allow_network=False, allow_pkg_install=False)
    assert x.can_attempt("compute the factorial of 6")
    assert x.can_attempt("reverse a string")
    assert x.can_attempt("what is the gcd of 12 and 18?")
    assert not x.can_attempt("hi")
    assert not x.can_attempt("thanks")
    assert not x.can_attempt("")


# --------------------------------------------------------------------------- #
# offline solve — deterministic recipe synthesis, no LLM / no network
# --------------------------------------------------------------------------- #
def test_offline_solve_real_answer_and_verified():
    _store, skills, kb = _fresh()
    x = InfiniteExplorer(llm=None, researcher=None, knowledge=kb, skills=skills,
                         allow_network=False, allow_pkg_install=False)
    r = x.explore("reverse a string")
    assert isinstance(r, ExploreResult)
    assert r.solved and r.origin == "offline"
    assert r.answer == "cba" and r.verified           # computed by the real sandbox
    assert r.learned_skill and r.learned_kb


def test_offline_solve_factorial():
    _store, skills, kb = _fresh()
    x = InfiniteExplorer(knowledge=kb, skills=skills,
                         allow_network=False, allow_pkg_install=False)
    r = x.explore("compute the factorial of 5")
    assert r.solved and r.answer == 120


# --------------------------------------------------------------------------- #
# permanent learning — skill + knowledge survive, and survive save/load
# --------------------------------------------------------------------------- #
def test_permanent_learning_persists_skill_and_knowledge():
    store, skills, kb = _fresh()
    x = InfiniteExplorer(knowledge=kb, skills=skills,
                         allow_network=False, allow_pkg_install=False)
    x.explore("reverse a string")

    assert skills.recall("reverse a string")
    assert kb.retrieve("reverse a string", k=2)

    # the learned skill + kb chunk survive a real store save/load round-trip
    path = os.path.join(tempfile.gettempdir(), "nyx_explorer_test.json")
    store.save(path)
    store2 = MemoryStore()
    store2.load(path)
    skills2 = SkillMemory(store=store2)
    kb2 = KnowledgeBase(store=store2, name="explorer")
    assert skills2.recall("reverse a string")
    assert kb2.retrieve("reverse a string", k=2)
    os.remove(path)


# --------------------------------------------------------------------------- #
# recall short-circuit — a second pass reuses the learned method, no re-derivation
# --------------------------------------------------------------------------- #
def test_recall_short_circuit():
    _store, skills, kb = _fresh()
    x = InfiniteExplorer(knowledge=kb, skills=skills,
                         allow_network=False, allow_pkg_install=False)
    first = x.explore("reverse a string")
    assert first.origin == "offline"
    second = x.explore("reverse a string")
    assert second.origin == "recall" and second.solved and second.answer == "cba"
    assert second.rounds == 0


# --------------------------------------------------------------------------- #
# debug loop — the REAL traceback drives the next synthesis round
# --------------------------------------------------------------------------- #
def test_debug_loop_fixes_broken_code():
    _store, skills, kb = _fresh()
    llm = _ScriptedLLM("result = 1 / 0\n", "result = sum(range(11))\n")
    x = InfiniteExplorer(llm=llm, knowledge=kb, skills=skills,
                         allow_network=False, allow_pkg_install=False)
    r = x.explore("compute the sum of 0..10")
    assert r.solved and r.origin == "llm"
    assert r.rounds >= 2 and r.answer == 55
    assert r.debug_trace                                  # the crash was recorded
    # the real error was fed back into the repair prompt
    assert any("previous attempt FAILED" in p for p in llm.prompts)


def test_debug_loop_gives_up_after_max_rounds():
    _store, skills, kb = _fresh()
    llm = _ScriptedLLM("result = undefined_name\n")        # always broken
    x = InfiniteExplorer(llm=llm, knowledge=kb, skills=skills, max_debug_rounds=3,
                         allow_network=False, allow_pkg_install=False)
    r = x.explore("do something impossible offline")
    assert not r.solved and r.rounds == 3 and not r.learned_skill


# --------------------------------------------------------------------------- #
# safety — a script reaching for the OS / kernel is screened out, never executed
# --------------------------------------------------------------------------- #
def test_safety_screen_rejects_os_access():
    ok, reason = InfiniteExplorer._screen("import os\nresult = os.listdir('/')\n")
    assert not ok and "os" in reason


def test_safety_screen_rejects_kernel_reach():
    ok, _ = InfiniteExplorer._screen("import nyxara.kernel as k\nresult = 1\n")
    assert not ok


def test_safety_screen_allows_clean_code():
    ok, _ = InfiniteExplorer._screen("import math\nresult = math.sqrt(16)\n")
    assert ok


def test_unsafe_llm_code_is_never_run():
    _store, skills, kb = _fresh()
    # the model keeps trying to read the filesystem — the screen must block every round
    llm = _ScriptedLLM("import os\nresult = os.listdir('/')\n")
    x = InfiniteExplorer(llm=llm, knowledge=kb, skills=skills, max_debug_rounds=2,
                         allow_network=False, allow_pkg_install=False)
    r = x.explore("read the filesystem")
    assert not r.solved
    assert any("safety screen" in t for t in r.debug_trace)


# --------------------------------------------------------------------------- #
# graceful degradation — every dependency None ⇒ data, never a crash
# --------------------------------------------------------------------------- #
def test_graceful_degradation_no_dependencies():
    x = InfiniteExplorer(llm=None, researcher=None, knowledge=None, skills=None,
                         allow_network=False, allow_pkg_install=False)
    r = x.explore("compute the factorial of 5")
    assert r.solved and r.answer == 120
    assert not r.learned_skill and not r.learned_kb       # nothing to persist into


def test_web_access_flag_disables_network():
    class _Settings:
        class features:
            web_access = False
    x = InfiniteExplorer(settings=_Settings(), allow_network=True, allow_pkg_install=True)
    assert x.allow_network is False                        # honored the feature flag


# --------------------------------------------------------------------------- #
# orchestrator integration — auto-route on "I don't know"
# --------------------------------------------------------------------------- #
def test_orchestrator_auto_routes_low_confidence_to_explorer():
    from nyxara.agency.permissions import Capability, RiskTier
    from nyxara.kernel.orchestrator import Candidate, NyxaraCore

    def unsure(stimulus, focus=None):
        return Candidate(text="I'm not sure", kind="respond",
                         capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                         confidence=0.2, belief=0.2, rationale="abstain")

    core = NyxaraCore(reasoner=unsure)
    assert core.explorer is not None
    core.explorer.allow_network = False                    # keep the test hermetic/offline
    core.process("reverse a string")
    reports = core.explorer.all_reports()
    assert reports and reports[0].solved and reports[0].learned_skill


def test_orchestrator_high_confidence_does_not_bootstrap():
    from nyxara.agency.permissions import Capability, RiskTier
    from nyxara.kernel.orchestrator import Candidate, NyxaraCore

    def confident(stimulus, focus=None):
        return Candidate(text="The answer is clear.", kind="respond",
                         capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                         confidence=0.95, belief=0.95, rationale="known")

    core = NyxaraCore(reasoner=confident)
    core.explorer.allow_network = False                    # keep the test hermetic/offline
    core.process("reverse a string")
    assert not core.explorer.all_reports()                 # never fired


def test_core_bootstrap_public_method():
    from nyxara.kernel.orchestrator import NyxaraCore

    core = NyxaraCore()
    core.explorer.allow_network = False                    # keep the test hermetic/offline
    out = core.bootstrap("compute the factorial of 5")
    assert out["solved"] and out["answer"] == 120


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
