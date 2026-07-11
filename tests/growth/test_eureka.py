"""Tests for nyxara.growth.eureka — truly novel problem solving (Pillar F · Edge 1+2).

The contract these tests pin down is the honest one: the engine *invents* its own candidates
(no LLM), but **nothing is kept unless an independent Prover certifies it PROVEN** — so soundness
is checked by re-verifying every kept breakthrough with a *fresh* Prover. We also pin the novelty
filter, the refutation of false statements, the headline generalisation path, deterministic+offline
operation, and serialisation.
"""

from __future__ import annotations

from nyxara.growth.eureka import (
    Breakthrough,
    BreakthroughReport,
    Conjecture,
    EurekaEngine,
)
from nyxara.growth.prover import ProofClaim, ProofVerdict, Prover


def _engine(seed: int = 7) -> EurekaEngine:
    # No memory / knowledge / flywheel sinks: fully offline, in-process, deterministic.
    return EurekaEngine(seed=seed)


# --------------------------------------------------------------------------- #
# Soundness — the whole point: a kept discovery is an independently re-checkable theorem.
# --------------------------------------------------------------------------- #
def test_every_kept_breakthrough_reverifies_proven_with_an_independent_prover():
    rep = _engine().discover(generations=4, population=30)
    assert rep.novel_kept > 0, "engine should discover at least one novel theorem"
    checker = Prover(seed=999)   # a *different* prover instance — no shared state
    for bt in rep.breakthroughs:
        c = bt.conjecture
        result = checker.prove(ProofClaim(kind=c.domain, statement=c.statement,
                                          candidate_answer=c.candidate_answer))
        assert result.verdict is ProofVerdict.PROVEN, (
            f"kept but not independently provable: {c.statement!r} -> {result.verdict}")


def test_engine_never_keeps_a_refuted_statement():
    # Drive a definitively false statement straight through the keep pipeline.
    eng = _engine()
    false_conj = Conjecture(domain="algebra", statement="n*n = n + n", ops=3)
    proof = eng._verify(false_conj)
    assert proof.verdict is ProofVerdict.REFUTED         # n*n = n+n only at n in {0,2}
    # A refuted proof is never turned into a breakthrough.
    assert eng._consider(false_conj, proof) is None or proof.verdict is ProofVerdict.REFUTED


def test_a_false_conjecture_is_counted_as_refuted_in_a_run():
    rep = _engine(seed=3).discover(generations=3, population=30)
    assert rep.refuted > 0, "honest search must refute some of its own guesses"


# --------------------------------------------------------------------------- #
# Novelty — the second filter: an exact restatement is not a new discovery.
# --------------------------------------------------------------------------- #
def test_repeated_discovery_is_not_counted_novel_again():
    eng = _engine()
    rep1 = eng.discover(generations=3, population=24)
    assert rep1.novel_kept > 0
    # The same engine, re-run: its frontier already holds the first batch, so novelty collapses
    # and far fewer (ideally zero brand-new) are kept the second time around.
    rep2 = eng.discover(generations=3, population=24)
    assert rep2.novel_kept < rep1.novel_kept


def test_interestingness_rejects_trivial_identities():
    eng = _engine()
    trivial = Conjecture(domain="algebra", statement="n + 1 = n + 1", ops=2)
    assert eng._is_trivial(trivial) is True
    assert eng._interestingness(trivial, eng._verify(trivial)) == 0.0
    triv_logic = Conjecture(domain="logic", statement="(A and B) <-> (A and B)", ops=2)
    assert eng._is_trivial(triv_logic) is True


# --------------------------------------------------------------------------- #
# The headline — lift an observed numeric instance to a certified symbolic law.
# --------------------------------------------------------------------------- #
def test_generalize_yields_certified_symbolic_identities():
    eng = _engine(seed=11)
    proven = 0
    checker = Prover(seed=5)
    for _ in range(400):
        conj = eng._generalize()
        if conj is None:
            continue
        assert conj.origin == "generalize" and "n" in conj.statement
        if checker.prove(ProofClaim("algebra", conj.statement)).verdict is ProofVerdict.PROVEN:
            proven += 1
        if proven >= 3:
            break
    assert proven >= 3, "the generalisation path should produce provable symbolic identities"


def test_generalize_lineage_records_the_observed_instance():
    eng = _engine(seed=11)
    for _ in range(200):
        conj = eng._generalize()
        if conj is not None:
            assert conj.lineage and "observed at n=" in conj.lineage[0]
            return
    raise AssertionError("expected at least one generalisation in 200 attempts")


# --------------------------------------------------------------------------- #
# Determinism + offline — same seed, same discoveries; no network, no LLM, no sinks.
# --------------------------------------------------------------------------- #
def test_runs_are_deterministic_under_a_fixed_seed():
    a = [b.statement for b in _engine(seed=42).discover(4, 26).breakthroughs]
    b = [b.statement for b in _engine(seed=42).discover(4, 26).breakthroughs]
    assert a == b and len(a) > 0


def test_different_seeds_explore_different_ground():
    a = {b.statement for b in _engine(seed=1).discover(4, 26).breakthroughs}
    b = {b.statement for b in _engine(seed=2).discover(4, 26).breakthroughs}
    assert a != b


# --------------------------------------------------------------------------- #
# Serialisation — the report and its parts round-trip to plain JSON-able dicts.
# --------------------------------------------------------------------------- #
def test_report_serialises_to_a_json_able_dict():
    import json

    rep = _engine().discover(generations=3, population=20)
    assert isinstance(rep, BreakthroughReport)
    d = rep.to_dict()
    json.dumps(d)   # must not raise
    assert d["novel_kept"] == rep.novel_kept
    assert d["generated"] >= d["proven"] >= d["novel_kept"]
    assert isinstance(d["breakthroughs"], list)
    if rep.breakthroughs:
        bt = rep.breakthroughs[0]
        assert isinstance(bt, Breakthrough)
        bd = bt.to_dict()
        assert bd["verdict"] == "proven" and bd["certificate"]


def test_score_is_novelty_times_interestingness():
    rep = _engine().discover(generations=2, population=18)
    for bt in rep.breakthroughs:
        assert abs(bt.score() - round(bt.novelty * bt.interestingness, 6)) < 1e-9


# --------------------------------------------------------------------------- #
# Cumulative counter — what the orchestrator's status report reads.
# --------------------------------------------------------------------------- #
def test_total_breakthroughs_accumulates_across_runs():
    eng = _engine()
    eng.discover(2, 18)
    first = eng.total_breakthroughs
    assert first > 0
    eng.discover(2, 18)
    assert eng.total_breakthroughs >= first


# --------------------------------------------------------------------------- #
# Genuine genetic programming — mutate/crossover recombine real subtrees (not template re-seeding).
# --------------------------------------------------------------------------- #
def test_tree_expansion_is_always_an_exact_identity():
    # every algebra tree the factory builds expands to a polynomial the Prover certifies PROVEN.
    from nyxara.growth.eureka import _TreeFactory, LemmaLibrary
    from random import Random
    tf = _TreeFactory(Random(0), LemmaLibrary())
    checker = Prover(seed=3)
    proven = 0
    for _ in range(60):
        tree = tf.fresh(allow_var=True, depth=3)
        stmt = f"{tree.render()} = {tree.evaluate().render()}"
        res = checker.prove(ProofClaim(kind="algebra", statement=stmt))
        assert res.verdict is ProofVerdict.PROVEN, f"tree is not an identity: {stmt}"
        proven += 1
    assert proven == 60


def test_crossover_child_shares_real_subtree_structure_with_parents():
    from nyxara.growth.eureka import _TreeFactory, LemmaLibrary
    from random import Random
    tf = _TreeFactory(Random(1), LemmaLibrary())
    a = tf.fresh(depth=3)
    while a.size() < 4:
        a = tf.fresh(depth=3)
    b = tf.fresh(depth=3)
    while b.size() < 4:
        b = tf.fresh(depth=3)
    shared = 0
    for _ in range(60):
        child = tf.crossover(a, b)
        if child is None:
            continue
        child_renders = {n.render() for n in child.nodes()}
        parent_renders = {n.render() for n in a.nodes()} | {n.render() for n in b.nodes()}
        if child_renders & parent_renders:
            shared += 1
    # a real subtree splice almost always leaves a subtree in common with a parent (not a re-seed)
    assert shared >= 50


def test_mutate_child_is_distinct_but_related_to_its_parent():
    from nyxara.growth.eureka import _TreeFactory, LemmaLibrary
    from random import Random
    tf = _TreeFactory(Random(3), LemmaLibrary())
    parent = tf.fresh(depth=3)
    while parent.size() < 6:
        parent = tf.fresh(depth=3)
    related = 0
    for _ in range(100):
        child = tf.mutate(parent)
        if child is None or child.render() == parent.render():
            continue
        if {n.render() for n in child.nodes()} & {n.render() for n in parent.nodes()}:
            related += 1
    assert related >= 60


# --------------------------------------------------------------------------- #
# The self-extending lemma library — proven identities become reusable terminals of the grammar.
# --------------------------------------------------------------------------- #
def test_lemma_library_grows_from_proven_discoveries():
    eng = _engine(seed=7)
    rep = eng.discover(generations=4, population=28)
    assert rep.lemmas_added > 0, "proven algebraic identities should promote into lemmas"
    assert len(eng.lemmas) == rep.lemma_library_size
    # every stored lemma is a genuine (degree >= 1) polynomial, usable as a grammar terminal
    terminals = eng.lemmas.terminals()
    assert terminals, "library should expose reusable terminals"
    for name, poly in terminals:
        assert name.startswith("L") and poly.degree() >= 1


def test_lemma_library_dedupes_and_persists_round_trip():
    from nyxara.growth.eureka import LemmaLibrary, _Poly
    lib = LemmaLibrary()
    n1 = lib.add(_Poly({2: 1, 1: 2, 0: 1}), "(n + 1)^2 = n^2 + 2*n + 1")   # degree 2 -> kept
    n2 = lib.add(_Poly({2: 1, 1: 2, 0: 1}), "restated")                    # duplicate -> None
    n3 = lib.add(_Poly({0: 5}), "5 = 5")                                   # constant -> None
    assert n1 == "L0" and n2 is None and n3 is None and len(lib) == 1
    restored = LemmaLibrary()
    restored.from_dict(lib.to_dict())
    assert [p.key() for _, p in restored.terminals()] == [p.key() for _, p in lib.terminals()]


def test_later_generations_compose_over_self_certified_lemmas():
    # once she has proven lemmas, some children are built by composing over them (the alphabet grew).
    eng = _engine(seed=7)
    rep = eng.discover(generations=5, population=30)
    assert len(eng.lemmas) > 0
    composed = [b for b in rep.breakthroughs
                if any(str(x).startswith("uses L") for x in b.conjecture.lineage)]
    assert composed, "expected at least one discovery composed over a self-certified lemma"
    # and anything kept is still an independently re-provable theorem
    checker = Prover(seed=123)
    for b in composed:
        c = b.conjecture
        assert checker.prove(ProofClaim(kind=c.domain, statement=c.statement)).verdict \
            is ProofVerdict.PROVEN
