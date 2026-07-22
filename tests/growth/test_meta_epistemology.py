"""Tests for nyxara.growth.meta_epistemology — Autonomous Synthetic Mathematics.

When a goal is unprovable in her current axioms she invents a new one, admitted only if consistent,
non-trivial, and generative. The prover is real congruence closure. These tests assert the decision
procedure, the invent-when-stuck move, and the consistency/non-triviality/generativity gates.
"""

from __future__ import annotations

from nyxara.growth.meta_epistemology import (
    Axiom,
    MetaEpistemologyCore,
    TheoremCertificate,
    op,
)

_X, _Y, _Z = "?x", "?y", "?z"


def _assoc_plus() -> Axiom:
    return Axiom("assoc[+]", op("+", op("+", _X, _Y), _Z), op("+", _X, op("+", _Y, _Z)),
                 frozenset({_X, _Y, _Z}))


def _core():
    return MetaEpistemologyCore(ops=("+", "*"), constants=("a", "b", "c"), base_axioms=[_assoc_plus()])


def test_congruence_closure_uses_given_axioms():
    core = _core()
    # associativity IS given → (a+b)+c = a+(b+c) must be provable
    assert core.provable(op("+", op("+", "a", "b"), "c"), op("+", "a", op("+", "b", "c")))


def test_commutativity_not_derivable_from_associativity():
    core = _core()
    assert not core.provable(op("+", "a", "b"), op("+", "b", "a"))


def test_invents_commutativity_when_stuck():
    core = _core()
    cert = core.prove_or_invent(op("+", "a", "b"), op("+", "b", "a"))
    assert isinstance(cert, TheoremCertificate)
    assert cert.admitted and cert.axiom["name"] == "comm[+]"
    assert cert.consistent and cert.non_trivial and cert.generative


def test_goal_and_heldout_provable_after_admission():
    core = _core()
    core.prove_or_invent(op("+", "a", "b"), op("+", "b", "a"))
    assert core.provable(op("+", "a", "b"), op("+", "b", "a"))
    assert core.provable(op("+", "b", "c"), op("+", "c", "b"))       # generalised, not patched


def test_already_provable_goal_invents_nothing():
    core = _core()
    core.prove_or_invent(op("+", "a", "b"), op("+", "b", "a"))
    cert = core.prove_or_invent(op("+", "a", "b"), op("+", "b", "a"))
    assert not cert.admitted and "already provable" in cert.reason


def test_consistency_guard_detects_falsehood():
    core = _core()
    bad = Axiom("bogus", "a", "b", frozenset())     # would equate the two designated-distinct constants
    assert not core._consistent(extra=(bad,))


def test_persistence(tmp_path):
    path = tmp_path / "axioms.json"
    core = MetaEpistemologyCore(ops=("+", "*"), constants=("a", "b", "c"),
                                base_axioms=[_assoc_plus()], persist_path=path)
    core.prove_or_invent(op("+", "a", "b"), op("+", "b", "a"))
    assert path.exists() and "comm[+]" in path.read_text()
