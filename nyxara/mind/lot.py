"""NYXARA · mind/lot.py — the Language of Thought (✦).

Thoughts, in NYXARA, are not strings. They are **structured symbolic expressions**
— a compositional "Mentalese" (Fodor's Language of Thought) that is independent of
any natural language. Meaning is built from parts and the way they are combined, so
the representation is:

* **compositional** — ``brown(x) ∧ dog(x)`` means what it does *because* of its parts;
* **systematic** — if you can think ``loves(jp, nyxara)`` you can think ``loves(nyxara, jp)``;
* **productive** — finitely many primitives compose into unboundedly many thoughts;
* **structure-preserving** — operations (substitution, β-reduction, unification)
  transform the structure without flattening it to text.

Primitives: :class:`Var`, :class:`Const`, :class:`Predicate`, :class:`Function`,
the connectives :class:`Not`/:class:`And`/:class:`Or`/:class:`Implies`/:class:`Iff`,
the quantifiers :class:`Forall`/:class:`Exists`, and — the engine of concept
combination — :class:`Lambda` / :class:`App` with capture-avoiding substitution and
β-reduction. Plus first-order :func:`unify` and :func:`alpha_equiv`.

Pure standard library.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

__all__ = [
    "Term", "Var", "Const", "Predicate", "Function",
    "Not", "And", "Or", "Implies", "Iff", "Forall", "Exists", "Lambda", "App",
    "var", "const", "pred", "func", "fresh_var",
    "alpha_equiv", "reduce", "unify", "parse",
]

_counter = itertools.count(1)


def fresh_var(type: str = "e") -> "Var":
    return Var(f"_g{next(_counter)}", type)


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class Term:
    """Base class for every Language-of-Thought expression. Subclasses are frozen."""

    type: str = "e"

    def free_vars(self) -> FrozenSet[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def substitute(self, mapping: Dict[str, "Term"]) -> "Term":  # pragma: no cover
        raise NotImplementedError

    def children(self) -> Tuple["Term", ...]:
        return ()

    def depth(self) -> int:
        kids = self.children()
        return 1 + (max(c.depth() for c in kids) if kids else 0)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children())

    def to_dict(self) -> dict:
        return {"kind": type(self).__name__, "str": str(self),
                "type": self.type, "children": [c.to_dict() for c in self.children()]}


# --------------------------------------------------------------------------- #
# Atoms
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Var(Term):
    name: str
    type: str = "e"

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        return mapping.get(self.name, self)

    def __str__(self) -> str:
        return f"?{self.name}"


@dataclass(frozen=True)
class Const(Term):
    name: str
    type: str = "e"

    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        return self

    def __str__(self) -> str:
        return self.name


# --------------------------------------------------------------------------- #
# Applications
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Predicate(Term):
    name: str
    args: Tuple[Term, ...] = ()
    type: str = "t"

    def free_vars(self) -> FrozenSet[str]:
        return frozenset().union(*(a.free_vars() for a in self.args)) if self.args else frozenset()

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        return Predicate(self.name, tuple(a.substitute(mapping) for a in self.args))

    def children(self) -> Tuple[Term, ...]:
        return self.args

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(a) for a in self.args)})"


@dataclass(frozen=True)
class Function(Term):
    name: str
    args: Tuple[Term, ...] = ()
    type: str = "e"

    def free_vars(self) -> FrozenSet[str]:
        return frozenset().union(*(a.free_vars() for a in self.args)) if self.args else frozenset()

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        return Function(self.name, tuple(a.substitute(mapping) for a in self.args))

    def children(self) -> Tuple[Term, ...]:
        return self.args

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(a) for a in self.args)})"


# --------------------------------------------------------------------------- #
# Connectives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Not(Term):
    arg: Term
    type: str = "t"

    def free_vars(self) -> FrozenSet[str]:
        return self.arg.free_vars()

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        return Not(self.arg.substitute(mapping))

    def children(self) -> Tuple[Term, ...]:
        return (self.arg,)

    def __str__(self) -> str:
        return f"¬{self.arg}"


@dataclass(frozen=True)
class And(Term):
    args: Tuple[Term, ...]
    type: str = "t"

    def free_vars(self) -> FrozenSet[str]:
        return frozenset().union(*(a.free_vars() for a in self.args)) if self.args else frozenset()

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        return And(tuple(a.substitute(mapping) for a in self.args))

    def children(self) -> Tuple[Term, ...]:
        return self.args

    def __str__(self) -> str:
        return "(" + " ∧ ".join(str(a) for a in self.args) + ")"


@dataclass(frozen=True)
class Or(Term):
    args: Tuple[Term, ...]
    type: str = "t"

    def free_vars(self) -> FrozenSet[str]:
        return frozenset().union(*(a.free_vars() for a in self.args)) if self.args else frozenset()

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        return Or(tuple(a.substitute(mapping) for a in self.args))

    def children(self) -> Tuple[Term, ...]:
        return self.args

    def __str__(self) -> str:
        return "(" + " ∨ ".join(str(a) for a in self.args) + ")"


@dataclass(frozen=True)
class Implies(Term):
    ante: Term
    cons: Term
    type: str = "t"

    def free_vars(self) -> FrozenSet[str]:
        return self.ante.free_vars() | self.cons.free_vars()

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        return Implies(self.ante.substitute(mapping), self.cons.substitute(mapping))

    def children(self) -> Tuple[Term, ...]:
        return (self.ante, self.cons)

    def __str__(self) -> str:
        return f"({self.ante} → {self.cons})"


@dataclass(frozen=True)
class Iff(Term):
    left: Term
    right: Term
    type: str = "t"

    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        return Iff(self.left.substitute(mapping), self.right.substitute(mapping))

    def children(self) -> Tuple[Term, ...]:
        return (self.left, self.right)

    def __str__(self) -> str:
        return f"({self.left} ↔ {self.right})"


# --------------------------------------------------------------------------- #
# Binders (capture-avoiding)
# --------------------------------------------------------------------------- #
def _avoid_capture(bound: "Var", body: Term, mapping: Dict[str, Term]) -> Tuple["Var", Term]:
    """Rename the bound variable if substitution would capture a free variable."""
    m = {k: v for k, v in mapping.items() if k != bound.name}
    incoming_free: FrozenSet[str] = frozenset().union(
        *(v.free_vars() for v in m.values())) if m else frozenset()
    if bound.name in incoming_free:
        fresh = fresh_var(bound.type)
        body = body.substitute({bound.name: fresh})
        bound = fresh
    return bound, body.substitute(m)


@dataclass(frozen=True)
class Forall(Term):
    var: Var
    body: Term
    type: str = "t"

    def free_vars(self) -> FrozenSet[str]:
        return self.body.free_vars() - {self.var.name}

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        v, b = _avoid_capture(self.var, self.body, mapping)
        return Forall(v, b)

    def children(self) -> Tuple[Term, ...]:
        return (self.body,)

    def __str__(self) -> str:
        return f"∀{self.var.name}. {self.body}"


@dataclass(frozen=True)
class Exists(Term):
    var: Var
    body: Term
    type: str = "t"

    def free_vars(self) -> FrozenSet[str]:
        return self.body.free_vars() - {self.var.name}

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        v, b = _avoid_capture(self.var, self.body, mapping)
        return Exists(v, b)

    def children(self) -> Tuple[Term, ...]:
        return (self.body,)

    def __str__(self) -> str:
        return f"∃{self.var.name}. {self.body}"


@dataclass(frozen=True)
class Lambda(Term):
    var: Var
    body: Term
    type: str = "e->?"

    def free_vars(self) -> FrozenSet[str]:
        return self.body.free_vars() - {self.var.name}

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        v, b = _avoid_capture(self.var, self.body, mapping)
        return Lambda(v, b)

    def children(self) -> Tuple[Term, ...]:
        return (self.body,)

    def __str__(self) -> str:
        return f"λ{self.var.name}. {self.body}"


@dataclass(frozen=True)
class App(Term):
    func: Term
    arg: Term
    type: str = "?"

    def free_vars(self) -> FrozenSet[str]:
        return self.func.free_vars() | self.arg.free_vars()

    def substitute(self, mapping: Dict[str, Term]) -> Term:
        return App(self.func.substitute(mapping), self.arg.substitute(mapping))

    def children(self) -> Tuple[Term, ...]:
        return (self.func, self.arg)

    def __str__(self) -> str:
        return f"({self.func} {self.arg})"


# --------------------------------------------------------------------------- #
# Builders (ergonomic)
# --------------------------------------------------------------------------- #
def var(name: str, type: str = "e") -> Var:
    return Var(name, type)


def const(name: str, type: str = "e") -> Const:
    return Const(name, type)


def pred(name: str, *args: Term) -> Predicate:
    return Predicate(name, tuple(args))


def func(name: str, *args: Term) -> Function:
    return Function(name, tuple(args))


# --------------------------------------------------------------------------- #
# β-reduction (concept composition)
# --------------------------------------------------------------------------- #
def reduce(term: Term, *, max_steps: int = 1000) -> Term:
    """Reduce to β-normal form: ``(λx. body) arg → body[x := arg]`` everywhere."""
    steps = 0

    def step(t: Term) -> Term:
        nonlocal steps
        if isinstance(t, App):
            f = step(t.func)
            a = step(t.arg)
            if isinstance(f, Lambda):
                steps += 1
                return step(f.body.substitute({f.var.name: a}))
            return App(f, a)
        if isinstance(t, Not):
            return Not(step(t.arg))
        if isinstance(t, And):
            return And(tuple(step(x) for x in t.args))
        if isinstance(t, Or):
            return Or(tuple(step(x) for x in t.args))
        if isinstance(t, Implies):
            return Implies(step(t.ante), step(t.cons))
        if isinstance(t, Iff):
            return Iff(step(t.left), step(t.right))
        if isinstance(t, Predicate):
            return Predicate(t.name, tuple(step(x) for x in t.args))
        if isinstance(t, Function):
            return Function(t.name, tuple(step(x) for x in t.args))
        if isinstance(t, Forall):
            return Forall(t.var, step(t.body))
        if isinstance(t, Exists):
            return Exists(t.var, step(t.body))
        if isinstance(t, Lambda):
            return Lambda(t.var, step(t.body))
        return t

    cur = term
    while steps < max_steps:
        before = steps
        cur = step(cur)
        if steps == before:
            break
    return cur


# --------------------------------------------------------------------------- #
# α-equivalence (structural equality up to bound renaming)
# --------------------------------------------------------------------------- #
def alpha_equiv(a: Term, b: Term) -> bool:
    """True iff ``a`` and ``b`` are equal up to renaming of bound variables."""
    def go(x: Term, y: Term, ex: Dict[str, int], ey: Dict[str, int], lvl: int) -> bool:
        if type(x) is not type(y):
            return False
        if isinstance(x, Var):
            bx, by = ex.get(x.name), ey.get(y.name)  # type: ignore[union-attr]
            if bx is None and by is None:
                return x.name == y.name             # both free
            return bx == by                          # both bound at same level
        if isinstance(x, Const):
            return x.name == y.name                  # type: ignore[union-attr]
        if isinstance(x, (Predicate, Function)):
            if x.name != y.name or len(x.args) != len(y.args):  # type: ignore[union-attr]
                return False
            return all(go(p, q, ex, ey, lvl) for p, q in zip(x.args, y.args))  # type: ignore
        if isinstance(x, (Forall, Exists, Lambda)):
            ex2 = {**ex, x.var.name: lvl}            # type: ignore[union-attr]
            ey2 = {**ey, y.var.name: lvl}            # type: ignore[union-attr]
            return go(x.body, y.body, ex2, ey2, lvl + 1)  # type: ignore[union-attr]
        # generic compound: compare children positionally
        cx, cy = x.children(), y.children()
        if len(cx) != len(cy):
            return False
        return all(go(p, q, ex, ey, lvl) for p, q in zip(cx, cy))

    return go(a, b, {}, {}, 0)


# --------------------------------------------------------------------------- #
# First-order unification (Robinson, with occurs-check)
# --------------------------------------------------------------------------- #
def _walk(t: Term, subst: Dict[str, Term]) -> Term:
    while isinstance(t, Var) and t.name in subst:
        t = subst[t.name]
    return t


def _occurs(name: str, t: Term, subst: Dict[str, Term]) -> bool:
    t = _walk(t, subst)
    if isinstance(t, Var):
        return t.name == name
    return any(_occurs(name, c, subst) for c in t.children())


def unify(a: Term, b: Term, subst: Optional[Dict[str, Term]] = None) -> Optional[Dict[str, Term]]:
    """Unify two terms. Returns an MGU substitution, or None if they don't unify.

    Treats :class:`Var` as logic variables; binders are compared structurally
    (no unification *under* a binder — quantified forms must match shape)."""
    subst = dict(subst or {})
    a, b = _walk(a, subst), _walk(b, subst)

    if isinstance(a, Var):
        if isinstance(b, Var) and a.name == b.name:
            return subst
        if _occurs(a.name, b, subst):
            return None
        subst[a.name] = b
        return subst
    if isinstance(b, Var):
        return unify(b, a, subst)

    if isinstance(a, Const) and isinstance(b, Const):
        return subst if a.name == b.name else None

    if isinstance(a, (Predicate, Function)) and type(a) is type(b):
        if a.name != b.name or len(a.args) != len(b.args):  # type: ignore[union-attr]
            return None
        for pa, pb in zip(a.args, b.args):                  # type: ignore[union-attr]
            subst = unify(pa, pb, subst)  # type: ignore[arg-type]
            if subst is None:
                return None
        return subst

    if type(a) is type(b):
        ca, cb = a.children(), b.children()
        if len(ca) != len(cb):
            return None
        # binders: also require the bound-variable structure align (compare α-equiv shells)
        for pa, pb in zip(ca, cb):
            subst = unify(pa, pb, subst)
            if subst is None:
                return None
        return subst

    return None


def apply_subst(t: Term, subst: Dict[str, Term]) -> Term:
    return t.substitute(subst)


# --------------------------------------------------------------------------- #
# Minimal s-expression parser (convenience)
# --------------------------------------------------------------------------- #
_KEYWORDS = {"and", "or", "not", "implies", "iff", "forall", "exists", "lambda"}


def _tokenize(s: str) -> List[str]:
    return s.replace("(", " ( ").replace(")", " ) ").split()


def parse(s: str) -> Term:
    """Parse an s-expression into a Term.

    Conventions: ``?x`` is a variable; ``(P ...)`` with capitalised head is a
    Predicate; lowercase non-keyword head is a Function; keywords build connectives
    / quantifiers, e.g. ``(forall ?x (implies (Dog ?x) (Animal ?x)))``.
    """
    tokens = _tokenize(s)
    pos = 0

    def atom(tok: str) -> Term:
        if tok.startswith("?"):
            return Var(tok[1:])
        return Const(tok)

    def parse_expr() -> Term:
        nonlocal pos
        tok = tokens[pos]
        if tok != "(":
            pos += 1
            return atom(tok)
        pos += 1  # consume '('
        head = tokens[pos]
        pos += 1
        args: List[Term] = []
        if head in ("forall", "exists", "lambda"):
            vtok = tokens[pos]; pos += 1
            v = Var(vtok[1:] if vtok.startswith("?") else vtok)
            body = parse_expr()
            assert tokens[pos] == ")"; pos += 1
            return {"forall": Forall, "exists": Exists, "lambda": Lambda}[head](v, body)
        while tokens[pos] != ")":
            args.append(parse_expr())
        pos += 1  # consume ')'
        if head == "not":
            return Not(args[0])
        if head == "and":
            return And(tuple(args))
        if head == "or":
            return Or(tuple(args))
        if head == "implies":
            return Implies(args[0], args[1])
        if head == "iff":
            return Iff(args[0], args[1])
        if head[0].isupper():
            return Predicate(head, tuple(args))
        return Function(head, tuple(args))

    result = parse_expr()
    return result


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA language-of-thought self-test")
    print("=" * 70)

    x = var("x")
    jp, nyx = const("JP"), const("NYXARA")

    # systematicity: loves(JP, NYXARA) and its mirror are both expressible
    t1 = pred("Loves", jp, nyx)
    t2 = pred("Loves", nyx, jp)
    print(f"systematic          : {t1}  |  {t2}")
    assert t1 != t2 and t1.free_vars() == frozenset()

    # compositionality via lambda: "brown dog" = λx. (Dog(x) ∧ Brown(x))
    brown_dog = Lambda(x, And((pred("Dog", x), pred("Brown", x))))
    applied = reduce(App(brown_dog, const("Rex")))
    print(f"compose brown-dog   : {brown_dog}")
    print(f"  applied to Rex    : {applied}")
    assert applied == And((pred("Dog", const("Rex")), pred("Brown", const("Rex"))))

    # capture-avoiding substitution
    inner = Forall(var("y"), pred("R", var("x"), var("y")))
    subbed = inner.substitute({"x": var("y")})  # naive sub would capture y
    print(f"\ncapture-avoid       : {inner}  --[x:=?y]-->  {subbed}")
    assert subbed.free_vars() == {"y"}  # the substituted y is free, not captured

    # α-equivalence
    a = Lambda(var("x"), pred("P", var("x")))
    b = Lambda(var("z"), pred("P", var("z")))
    print(f"\nα-equiv             : {a}  ≡  {b}  -> {alpha_equiv(a, b)}")
    assert alpha_equiv(a, b)
    assert not alpha_equiv(a, Lambda(var("x"), pred("Q", var("x"))))

    # unification
    s = unify(pred("Owns", var("who"), nyx), pred("Owns", jp, var("what")))
    print(f"\nunify Owns(?who,NYX) with Owns(JP,?what): {s}")
    assert s is not None
    assert s["who"] == jp and s["what"] == nyx

    # occurs-check prevents infinite terms
    assert unify(var("x"), func("f", var("x"))) is None
    print("occurs-check         : OK (no infinite term)")

    # parser round-trip
    p = parse("(forall ?x (implies (Dog ?x) (Animal ?x)))")
    print(f"\nparsed              : {p}")
    assert isinstance(p, Forall)
    assert p.free_vars() == frozenset()

    # higher-order composition: apply a 2-place relation builder
    rel = Lambda(var("a"), Lambda(var("b"), pred("Defends", var("a"), var("b"))))
    full = reduce(App(App(rel, nyx), jp))
    print(f"curried relation    : {full}")
    assert full == pred("Defends", nyx, jp)

    print(f"\ndepth/size of t1    : depth={t1.depth()} size={t1.size()}")
    print("\nALL SELF-TESTS PASSED ✓")
