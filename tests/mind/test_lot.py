"""Tests for nyxara.mind.lot — the Language of Thought."""

from __future__ import annotations

from nyxara.mind.lot import (
    And,
    App,
    Const,
    Exists,
    Forall,
    Function,
    Iff,
    Implies,
    Lambda,
    Not,
    Or,
    Predicate,
    Var,
    alpha_equiv,
    const,
    func,
    parse,
    pred,
    reduce,
    unify,
    var,
)


# -------------------- atoms & structure -------------------- #
def test_var_free_vars():
    assert var("x").free_vars() == {"x"}


def test_const_no_free_vars():
    assert const("JP").free_vars() == frozenset()


def test_predicate_free_vars():
    p = pred("Loves", var("x"), const("JP"))
    assert p.free_vars() == {"x"}
    assert p.type == "t"


def test_function_type_is_entity():
    assert func("father_of", const("JP")).type == "e"


def test_depth_and_size():
    t = And((pred("P", var("x")), pred("Q", var("y"))))
    assert t.depth() == 3       # And -> Predicate -> Var
    assert t.size() >= 5


def test_systematicity():
    a = pred("R", const("a"), const("b"))
    b = pred("R", const("b"), const("a"))
    assert a != b  # structure distinguishes argument order


def test_to_dict():
    d = pred("P", var("x")).to_dict()
    assert d["kind"] == "Predicate"
    assert d["children"]


# -------------------- substitution -------------------- #
def test_substitution_replaces_free_var():
    t = pred("P", var("x"))
    s = t.substitute({"x": const("a")})
    assert s == pred("P", const("a"))


def test_substitution_under_connectives():
    t = And((pred("P", var("x")), Not(pred("Q", var("x")))))
    s = t.substitute({"x": const("c")})
    assert s == And((pred("P", const("c")), Not(pred("Q", const("c")))))


def test_bound_var_not_substituted():
    t = Forall(var("x"), pred("P", var("x")))
    s = t.substitute({"x": const("a")})
    assert s == t  # x is bound, unaffected


def test_capture_avoiding_substitution():
    # ∀y. R(x, y) with x:=y  must NOT capture the bound y
    t = Forall(var("y"), pred("R", var("x"), var("y")))
    s = t.substitute({"x": var("y")})
    assert s.free_vars() == {"y"}
    # original bound var was renamed; structurally still a Forall
    assert isinstance(s, Forall)
    assert not alpha_equiv(s, Forall(var("y"), pred("R", var("y"), var("y"))))


# -------------------- beta reduction / composition -------------------- #
def test_beta_reduction_basic():
    f = Lambda(var("x"), pred("P", var("x")))
    assert reduce(App(f, const("a"))) == pred("P", const("a"))


def test_compositional_concept():
    brown_dog = Lambda(var("x"), And((pred("Dog", var("x")), pred("Brown", var("x")))))
    out = reduce(App(brown_dog, const("Rex")))
    assert out == And((pred("Dog", const("Rex")), pred("Brown", const("Rex"))))


def test_curried_relation():
    rel = Lambda(var("a"), Lambda(var("b"), pred("Defends", var("a"), var("b"))))
    out = reduce(App(App(rel, const("NYXARA")), const("JP")))
    assert out == pred("Defends", const("NYXARA"), const("JP"))


def test_reduce_no_redex_is_identity():
    t = pred("P", var("x"))
    assert reduce(t) == t


def test_reduce_capture_safe():
    # (λx. λy. R(x,y)) y   should not capture
    inner = Lambda(var("x"), Lambda(var("y"), pred("R", var("x"), var("y"))))
    out = reduce(App(inner, var("y")))
    # result is λ?_g. R(?y, ?_g) — the free y is preserved
    assert "y" in out.free_vars()


# -------------------- alpha equivalence -------------------- #
def test_alpha_equiv_lambda():
    assert alpha_equiv(Lambda(var("x"), pred("P", var("x"))),
                       Lambda(var("z"), pred("P", var("z"))))


def test_alpha_equiv_quantifier():
    assert alpha_equiv(Forall(var("x"), pred("P", var("x"))),
                       Forall(var("y"), pred("P", var("y"))))


def test_alpha_inequiv_different_predicate():
    assert not alpha_equiv(Lambda(var("x"), pred("P", var("x"))),
                           Lambda(var("x"), pred("Q", var("x"))))


def test_alpha_inequiv_free_vars():
    assert not alpha_equiv(pred("P", var("x")), pred("P", var("y")))


def test_alpha_equiv_nested():
    a = Forall(var("x"), Exists(var("y"), pred("R", var("x"), var("y"))))
    b = Forall(var("a"), Exists(var("b"), pred("R", var("a"), var("b"))))
    assert alpha_equiv(a, b)
    # swapping the bound order breaks it
    c = Forall(var("a"), Exists(var("b"), pred("R", var("b"), var("a"))))
    assert not alpha_equiv(a, c)


# -------------------- unification -------------------- #
def test_unify_variable_to_const():
    s = unify(var("x"), const("a"))
    assert s == {"x": const("a")}


def test_unify_predicates():
    s = unify(pred("Owns", var("who"), const("NYXARA")),
              pred("Owns", const("JP"), var("what")))
    assert s["who"] == const("JP")
    assert s["what"] == const("NYXARA")


def test_unify_fails_different_const():
    assert unify(const("a"), const("b")) is None


def test_unify_fails_different_predicate():
    assert unify(pred("P", var("x")), pred("Q", var("x"))) is None


def test_unify_fails_arity():
    assert unify(pred("P", var("x")), pred("P", var("x"), var("y"))) is None


def test_unify_occurs_check():
    assert unify(var("x"), func("f", var("x"))) is None


def test_unify_same_var():
    assert unify(var("x"), var("x")) == {}


def test_unify_nested_functions():
    s = unify(func("f", var("x"), const("b")), func("f", const("a"), var("y")))
    assert s["x"] == const("a") and s["y"] == const("b")


def test_unify_chained_vars():
    s = unify(pred("P", var("x"), var("x")), pred("P", var("y"), const("a")))
    # x unifies with y, then x(=y) with a
    assert s is not None
    from nyxara.mind.lot import _walk
    assert _walk(var("x"), s) == const("a")


# -------------------- parser -------------------- #
def test_parse_atom():
    assert parse("?x") == Var("x")
    assert parse("JP") == Const("JP")


def test_parse_predicate():
    assert parse("(Dog ?x)") == pred("Dog", var("x"))


def test_parse_function():
    assert parse("(father_of JP)") == func("father_of", const("JP"))


def test_parse_connectives():
    assert parse("(and (P ?x) (Q ?x))") == And((pred("P", var("x")), pred("Q", var("x"))))
    assert parse("(not (P ?x))") == Not(pred("P", var("x")))
    assert parse("(implies (Dog ?x) (Animal ?x))") == Implies(pred("Dog", var("x")),
                                                              pred("Animal", var("x")))


def test_parse_quantifier():
    p = parse("(forall ?x (implies (Dog ?x) (Animal ?x)))")
    assert isinstance(p, Forall)
    assert p.free_vars() == frozenset()


def test_parse_lambda_and_reduce():
    p = parse("(lambda ?x (P ?x))")
    assert isinstance(p, Lambda)
    assert reduce(App(p, const("a"))) == pred("P", const("a"))
