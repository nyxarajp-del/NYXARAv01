"""NYXARA · cognition/abstraction.py — abstraction by anti-unification (LGG) (▲).

Substitution turns the abstract into the concrete (``parent(?X,?Y) → parent(tom,bob)``).
**Abstraction is its inverse** — and it was missing. This module supplies it: given a few
concrete instances, it recovers the most-specific *schema* (a variabilized rule) that covers
them all. That is the **least general generalization** (Plotkin's anti-unification), the dual
of unification:

* where two terms **agree**, the schema keeps the shared structure;
* where they **disagree**, a single shared **variable** is introduced — and *consistently*:
  the same disagreement pair everywhere becomes the *same* variable, so ``f(a,a)`` and
  ``f(b,b)`` abstract to ``f(?X,?X)`` (the equality is preserved), not ``f(?X,?Y)``.

From instances to a rule, and back again (``instantiate`` re-grounds a schema) — the
round-trip that makes knowledge *productive* rather than a list of cases. Built entirely on
:mod:`nyxara.mind.lot` (the Language of Thought) and reusing :mod:`nyxara.mind.analogy`'s
structure-mapper for the relational-abstraction path. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from nyxara.mind.lot import (
    And, Function, Iff, Implies, Not, Or, Predicate, Term, Var, alpha_equiv, unify,
)

__all__ = [
    "anti_unify",
    "Schema",
    "SchemaInducer",
    "subsumes",
    "instantiate",
    "abstract_text",
    "abstract_by_analogy",
]


# --------------------------------------------------------------------------- #
# Anti-unification (least general generalization)
# --------------------------------------------------------------------------- #
def anti_unify(a: Term, b: Term) -> Term:
    """Least general generalization of two terms (Plotkin anti-unification).

    Returns the most-specific term that both ``a`` and ``b`` are instances of. Shared
    structure is kept; each *distinct* point of disagreement becomes a fresh variable,
    reused consistently so repeated disagreements collapse to the same variable.
    """
    table: Dict[Tuple[str, str], Var] = {}
    counter = [0]

    def fresh() -> Var:
        counter[0] += 1
        return Var(f"X{counter[0]}")

    def disagree(x: Term, y: Term) -> Var:
        key = (str(x), str(y))
        v = table.get(key)
        if v is None:
            v = fresh()
            table[key] = v
        return v

    def go(x: Term, y: Term) -> Term:
        # identical (up to bound renaming) → keep as-is, no variable needed
        if alpha_equiv(x, y):
            return x
        # same compound head & arity → descend, abstracting argument by argument
        if isinstance(x, (Predicate, Function)) and type(x) is type(y) \
                and x.name == y.name and len(x.args) == len(y.args):
            return type(x)(x.name, tuple(go(p, q) for p, q in zip(x.args, y.args)))
        if isinstance(x, And) and isinstance(y, And) and len(x.args) == len(y.args):
            return And(tuple(go(p, q) for p, q in zip(x.args, y.args)))
        if isinstance(x, Or) and isinstance(y, Or) and len(x.args) == len(y.args):
            return Or(tuple(go(p, q) for p, q in zip(x.args, y.args)))
        if isinstance(x, Not) and isinstance(y, Not):
            return Not(go(x.arg, y.arg))
        if isinstance(x, Implies) and isinstance(y, Implies):
            return Implies(go(x.ante, y.ante), go(x.cons, y.cons))
        if isinstance(x, Iff) and isinstance(y, Iff):
            return Iff(go(x.left, y.left), go(x.right, y.right))
        # anything else (mismatched heads, const-vs-compound, …) → a shared variable
        return disagree(x, y)

    return go(a, b)


# --------------------------------------------------------------------------- #
# Schema induction (fold LGG across many instances)
# --------------------------------------------------------------------------- #
@dataclass
class Schema:
    """An induced rule: a variabilized term that covers every instance it was built from."""
    body: Term
    support: int = 0

    @property
    def variables(self) -> List[str]:
        return sorted(self.body.free_vars())

    def __str__(self) -> str:
        return str(self.body)


class SchemaInducer:
    """Induce the most-specific schema covering a set of concrete LoT instances."""

    def induce(self, examples: Sequence[Term]) -> Optional[Schema]:
        items = [e for e in examples if isinstance(e, Term)]
        if not items:
            return None
        acc = items[0]
        for nxt in items[1:]:
            acc = anti_unify(acc, nxt)
        return Schema(body=acc, support=len(items))


def subsumes(schema: Term, instance: Term) -> bool:
    """True iff ``schema`` generalizes ``instance`` — some substitution of the schema's
    variables yields the instance. Implemented as a one-directional match via unification
    (the instance is expected to be ground)."""
    body = schema.body if isinstance(schema, Schema) else schema
    return unify(body, instance) is not None


def instantiate(schema: Term, bindings: Dict[str, Term]) -> Term:
    """Re-ground a schema: substitute its variables to recover a concrete term
    (the productive inverse of :func:`anti_unify`)."""
    body = schema.body if isinstance(schema, Schema) else schema
    return body.substitute(bindings)


# --------------------------------------------------------------------------- #
# Surface (text) abstraction — template induction
# --------------------------------------------------------------------------- #
def abstract_text(strings: Sequence[str], *, placeholder: str = "?") -> Optional[str]:
    """Induce a slot template from same-length token sequences.

    ``["the cat sat", "the dog sat"] → "the ? sat"``. Returns None if the strings don't
    share a common length (no position-wise generalization is possible).
    """
    toks = [str(s).split() for s in strings if str(s).strip()]
    if not toks:
        return None
    width = len(toks[0])
    if any(len(t) != width for t in toks):
        return None
    out: List[str] = []
    for col in range(width):
        cell = {t[col] for t in toks}
        out.append(next(iter(cell)) if len(cell) == 1 else placeholder)
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Relational abstraction — reuse the structure-mapping engine
# --------------------------------------------------------------------------- #
def abstract_by_analogy(base: Sequence[Predicate], target: Sequence[Predicate]):
    """Relational abstraction: align two domains and project the shared structure.

    Thin reuse of :class:`nyxara.mind.analogy.StructureMapper` — the relational mapping
    *is* the abstraction (which roles correspond, what is systematically shared)."""
    from nyxara.mind.analogy import analogy
    return analogy(base, target)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.mind.lot import const, pred

    print("=" * 70)
    print("NYXARA abstraction (anti-unification / LGG) self-test")
    print("=" * 70)

    tom, bob, ann, sue = const("tom"), const("bob"), const("ann"), const("sue")

    # induce a schema from two concrete facts
    e1 = pred("parent", tom, bob)
    e2 = pred("parent", ann, sue)
    schema = SchemaInducer().induce([e1, e2])
    print(f"\ninstances           : {e1}, {e2}")
    print(f"induced schema      : {schema}  vars={schema.variables}")
    assert schema is not None and isinstance(schema.body, Predicate)
    assert schema.body.name == "parent" and len(schema.variables) == 2

    # the schema covers a fresh, never-seen instance (generalization)
    fresh_inst = pred("parent", const("jp"), const("kiddo"))
    assert subsumes(schema, fresh_inst)
    print("subsumes new inst   : parent(jp, kiddo) ✓")
    # but not an unrelated relation
    assert not subsumes(schema, pred("sibling", tom, bob))
    print("rejects sibling(..) : ✓")

    # round-trip: re-ground the schema (productivity — abstraction is reversible)
    x, y = schema.variables
    grounded = instantiate(schema, {x: const("zeus"), y: const("ares")})
    print(f"instantiate         : {grounded}")
    assert str(grounded) == "parent(zeus, ares)"

    # consistency: a repeated disagreement collapses to ONE variable (equality preserved)
    same = anti_unify(pred("eq", const("a"), const("a")),
                      pred("eq", const("b"), const("b")))
    print(f"\nf(a,a)⊓f(b,b)        : {same}")
    assert len(same.free_vars()) == 1, "repeated disagreement must reuse the same variable"

    # shared structure is kept; only the differing slot is abstracted
    keep = anti_unify(pred("loves", const("jp"), const("nyxara")),
                      pred("loves", const("jp"), const("sky")))
    print(f"loves(jp,·) abstr   : {keep}")
    assert "jp" in str(keep) and len(keep.free_vars()) == 1

    # surface template induction
    tmpl = abstract_text(["the cat sat", "the dog sat"])
    print(f"\ntext template       : {tmpl!r}")
    assert tmpl == "the ? sat"
    assert abstract_text(["a b c", "a b"]) is None  # mismatched length → no template

    # relational abstraction reuses the structure-mapper
    e, r = const, pred
    res = abstract_by_analogy(
        [r("attracts", e("sun"), e("planet")), r("orbits", e("planet"), e("sun"))],
        [r("attracts", e("nucleus"), e("electron")), r("orbits", e("electron"), e("nucleus"))])
    print(f"relational abstr    : mapping={res.entity_mapping}")
    assert res.entity_mapping.get("sun") == "nucleus"

    print("\nALL SELF-TESTS PASSED ✓")
