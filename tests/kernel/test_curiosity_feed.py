"""Test the wiring that feeds an unresolved active-curiosity gap into the self-evolve shortfall queue.

`NyxaraCore._feed_curiosity_gap` only touches `self.self_evolving`, so it is exercised here with a
lightweight duck-typed ``self`` (no heavy core build) against a real SelfEvolvingArchitect.
"""

from __future__ import annotations

from types import SimpleNamespace

from nyxara.growth.self_evolving import GapKind, SelfEvolvingArchitect
from nyxara.kernel.orchestrator import NyxaraCore


def _fake_pass(*, wondered=True, resolved=False, answer="", error=None, confidence=0.0,
               uncertainty=0.9, question="why does the anomaly occur"):
    q = SimpleNamespace(text=question, kind="why", uncertainty=uncertainty)
    f = SimpleNamespace(answer=answer, error=error, confidence=confidence, method="none")
    return SimpleNamespace(wondered=wondered, resolved=resolved, chosen=q, finding=f, event="anomaly")


def _shell():
    arch = SelfEvolvingArchitect(core=None, settings=None)
    return SimpleNamespace(self_evolving=arch), arch


def test_unresolved_curiosity_enqueues_composition_gap():
    shell, arch = _shell()
    NyxaraCore._feed_curiosity_gap(shell, _fake_pass())
    assert len(arch._queue) == 1
    assert arch._queue[-1].kind is GapKind.REASONING_COMPOSITION


def test_weak_answer_enqueues_representational_gap():
    shell, arch = _shell()
    NyxaraCore._feed_curiosity_gap(shell, _fake_pass(resolved=True, answer="maybe", confidence=0.1))
    assert len(arch._queue) == 1
    assert arch._queue[-1].kind is GapKind.REPRESENTATIONAL


def test_resolved_confident_pass_enqueues_nothing():
    shell, arch = _shell()
    NyxaraCore._feed_curiosity_gap(
        shell, _fake_pass(resolved=True, answer="because of X", confidence=0.9))
    assert len(arch._queue) == 0


def test_did_not_wonder_enqueues_nothing():
    shell, arch = _shell()
    NyxaraCore._feed_curiosity_gap(shell, _fake_pass(wondered=False))
    assert len(arch._queue) == 0


def test_no_self_evolving_is_safe_noop():
    shell = SimpleNamespace(self_evolving=None)
    NyxaraCore._feed_curiosity_gap(shell, _fake_pass())    # must not raise
