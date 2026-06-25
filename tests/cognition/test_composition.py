"""Tests for nyxara.cognition.composition — compositional generalization."""

from __future__ import annotations

import pytest

from nyxara.cognition.composition import (
    CompositionalGrammar,
    deserialize_term,
    serialize_term,
)
from nyxara.memory.store import MemoryStore
from nyxara.mind.lot import Lambda, alpha_equiv, const, func, pred, var


def _grammar():
    g = CompositionalGrammar()
    g.learn("jump", const("JUMP"))
    g.learn("walk", const("WALK"))
    g.learn("turn", const("TURN"))
    g.learn("twice", Lambda(var("a"), func("seq", var("a"), var("a"))))
    g.learn("thrice", Lambda(var("a"), func("seq", var("a"), func("seq", var("a"), var("a")))))
    return g


# -------------------- primitives -------------------- #
def test_single_primitive():
    g = _grammar()
    assert str(g.interpret("jump")) == "JUMP"


# -------------------- the systematicity acid test -------------------- #
@pytest.mark.parametrize("phrase,want", [
    ("jump twice", "seq(JUMP, JUMP)"),
    ("walk twice", "seq(WALK, WALK)"),
    ("turn thrice", "seq(TURN, seq(TURN, TURN))"),
])
def test_novel_combinations_generalize(phrase, want):
    # none of these phrases were ever taught — only their parts were
    assert str(_grammar().interpret(phrase)) == want


def test_nested_modifiers_are_productive():
    g = _grammar()
    assert str(g.interpret("jump twice twice")) == "seq(seq(JUMP, JUMP), seq(JUMP, JUMP))"


def test_bare_primitives_are_sequenced():
    assert str(_grammar().interpret("jump walk")) == "seq(JUMP, WALK)"


# -------------------- honesty about unknowns -------------------- #
def test_unknown_primitive_is_reported_not_hallucinated():
    g = _grammar()
    result = g.interpret("jump sprint")
    assert not result.ok
    assert "sprint" in result.unknown
    assert result.meaning is None


def test_known_and_vocabulary():
    g = _grammar()
    assert g.known("jump") and not g.known("sprint")
    assert "twice" in g.vocabulary()


# -------------------- term serialization -------------------- #
@pytest.mark.parametrize("term", [
    const("JUMP"),
    var("a"),
    pred("loves", const("jp"), const("nyxara")),
    func("seq", const("A"), const("B")),
    Lambda(var("a"), func("seq", var("a"), var("a"))),
])
def test_serialize_round_trip(term):
    assert alpha_equiv(deserialize_term(serialize_term(term)), term)


# -------------------- persistence -------------------- #
def test_lexicon_persists_across_grammars():
    store = MemoryStore()
    g = CompositionalGrammar(store=store)
    g.learn("dance", const("DANCE"))
    g.learn("twice", Lambda(var("a"), func("seq", var("a"), var("a"))))
    fresh = CompositionalGrammar(store=store)
    assert str(fresh.interpret("dance twice")) == "seq(DANCE, DANCE)"
