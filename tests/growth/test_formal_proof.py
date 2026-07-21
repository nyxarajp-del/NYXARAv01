"""Tests for F9 — proof-carrying abstractions (growth/formal_proof.py).

She must type-check soundly, PROVE two programs equivalent (or REFUTE with a counter-example), and
UNIFY provably-equivalent abstractions without changing any program's meaning. No LLM anywhere.
"""

from __future__ import annotations

from nyxara.growth.formal_proof import (
    FormalVerifier,
    UnivalenceUnifier,
    equivalent,
    typecheck,
)
from nyxara.growth.noesis import (
    INT,
    INTLIST,
    Abstraction,
    NoesisEngine,
    app,
    base_library,
    evaluate,
    hole,
    lit,
    var,
)
from nyxara.growth.prover import ProofVerdict
from nyxara.growth.redteam import RedTeam


def test_typecheck_accepts_wellformed_and_rejects_illtyped():
    lib = base_library()
    good = app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), lit(2, INT)])])
    assert typecheck(good, lib)
    # ill-typed: sum_ expects an intlist, given an int
    bad = app("sum_", INT, [lit(5, INT)])
    assert not typecheck(bad, lib)


def test_equivalence_proves_and_refutes():
    lib = base_library()
    # sum(double each) == 2 * sum   → PROVEN
    a = app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), lit(2, INT)])])
    b = app("mul", INT, [lit(2, INT), app("sum_", INT, [var(INTLIST)])])
    assert equivalent(a, b, INTLIST, lib).verdict is ProofVerdict.PROVEN
    # sum vs length → REFUTED, with a counter-example
    c = app("length", INT, [var(INTLIST)])
    ref = equivalent(a, c, INTLIST, lib)
    assert ref.verdict is ProofVerdict.REFUTED and ref.counterexample is not None


def test_univalence_unifies_equivalent_abstractions_safely():
    lib = base_library()
    # two DIFFERENTLY-STRUCTURED but semantically equal abstractions of sum-of-scaled
    lib.add_abstraction(Abstraction("absP", (INT,), INT,
        app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), hole(0, INT)])])))
    lib.add_abstraction(Abstraction("absQ", (INT,), INT,
        app("mul", INT, [hole(0, INT), app("sum_", INT, [var(INTLIST)])])))
    # they agree for every k on every input → provably equal → unify to one
    merged = UnivalenceUnifier().unify(lib)
    assert len(merged) == 1
    keep, drop = merged[0]
    assert drop in lib.archived and drop not in lib.abstractions
    assert keep in lib.abstractions
    # meaning preserved: the kept abstraction still computes sum-of-scaled
    kept = app(keep, INT, [lit(2, INT)])
    assert evaluate(kept, [1, 2, 3], lib) == 12


def test_certify_stamps_a_wellformed_abstraction():
    lib = base_library()
    good = Abstraction("absG", (INT,), INT,
        app("sum_", INT, [app("inc_each", INTLIST, [var(INTLIST), hole(0, INT)])]))
    lib.add_abstraction(good)
    cert = FormalVerifier().certify(good, lib)
    assert cert.proven


def test_engine_with_formal_certifies_every_abstraction():
    fv = FormalVerifier()
    eng = NoesisEngine(seed=1, tasks_per_cycle=12, red_team=RedTeam(), formal=fv)
    eng.run(5)
    # every abstraction still in the library type-checks (a carried proof exists)
    from nyxara.growth.formal_proof import typecheck as tc
    for a in eng.library.abstractions.values():
        assert tc(a.body, eng.library, a.arg_types)
    # and it still compounds under the stricter regime
    assert len(eng.library.abstractions) >= 1
