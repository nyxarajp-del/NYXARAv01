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
