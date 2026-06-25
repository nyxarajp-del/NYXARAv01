"""Tests for nyxara.cognition.abstraction — anti-unification / least general generalization."""

from __future__ import annotations

from nyxara.cognition.abstraction import (
    SchemaInducer,
    abstract_by_analogy,
    abstract_text,
    anti_unify,
    instantiate,
    subsumes,
)
from nyxara.mind.lot import Predicate, Var, const, pred


# -------------------- anti-unification core -------------------- #
def test_identical_terms_have_no_variables():
    t = pred("parent", const("tom"), const("bob"))
    gen = anti_unify(t, t)
    assert not gen.free_vars()
    assert str(gen) == "parent(tom, bob)"


def test_differing_arguments_become_variables():
    gen = anti_unify(pred("parent", const("tom"), const("bob")),
                     pred("parent", const("ann"), const("sue")))
    assert isinstance(gen, Predicate) and gen.name == "parent"
    assert len(gen.free_vars()) == 2


def test_repeated_disagreement_reuses_one_variable():
    # f(a,a) ⊓ f(b,b) must be f(?X,?X), preserving the equality — not f(?X,?Y)
    gen = anti_unify(pred("eq", const("a"), const("a")),
                     pred("eq", const("b"), const("b")))
    assert len(gen.free_vars()) == 1


def test_shared_structure_is_preserved():
    gen = anti_unify(pred("loves", const("jp"), const("nyxara")),
                     pred("loves", const("jp"), const("sky")))
    assert "jp" in str(gen)
    assert len(gen.free_vars()) == 1


def test_mismatched_heads_collapse_to_a_variable():
    gen = anti_unify(pred("cat", const("x")), pred("dog", const("y")))
    assert isinstance(gen, Var)


# -------------------- schema induction & round-trip -------------------- #
def test_induce_schema_from_instances():
    schema = SchemaInducer().induce([pred("parent", const("tom"), const("bob")),
                                     pred("parent", const("ann"), const("sue"))])
    assert schema is not None
    assert schema.support == 2
    assert len(schema.variables) == 2


def test_schema_subsumes_fresh_instance_but_not_unrelated():
    schema = SchemaInducer().induce([pred("parent", const("tom"), const("bob")),
                                     pred("parent", const("ann"), const("sue"))])
    assert subsumes(schema, pred("parent", const("jp"), const("kiddo")))
    assert not subsumes(schema, pred("sibling", const("a"), const("b")))


def test_instantiate_round_trip():
    schema = SchemaInducer().induce([pred("parent", const("tom"), const("bob")),
                                     pred("parent", const("ann"), const("sue"))])
    x, y = schema.variables
    grounded = instantiate(schema, {x: const("zeus"), y: const("ares")})
    assert str(grounded) == "parent(zeus, ares)"


def test_induce_empty_returns_none():
    assert SchemaInducer().induce([]) is None


# -------------------- surface template induction -------------------- #
def test_abstract_text_template():
    assert abstract_text(["the cat sat", "the dog sat"]) == "the ? sat"


def test_abstract_text_requires_common_length():
    assert abstract_text(["a b c", "a b"]) is None


# -------------------- relational abstraction reuses structure-mapping -------------------- #
def test_abstract_by_analogy():
    e, r = const, pred
    res = abstract_by_analogy(
        [r("attracts", e("sun"), e("planet")), r("orbits", e("planet"), e("sun"))],
        [r("attracts", e("nucleus"), e("electron")), r("orbits", e("electron"), e("nucleus"))])
    assert res.entity_mapping.get("sun") == "nucleus"
    assert res.entity_mapping.get("planet") == "electron"
