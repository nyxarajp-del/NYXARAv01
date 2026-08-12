"""Regression tests for defects found auditing nyxara/causal/ file by file.

Each test names the thing that was actually broken and pins the fixed behaviour, so a future
edit that quietly reintroduces it fails here rather than in production.
"""

from __future__ import annotations

import time

import pytest

from nyxara.causal.axiom_synth import AxiomSynthesizer, Expr
from nyxara.causal.causal_knots import KnotLattice, KnotMutationFailure
from nyxara.causal.engine import CausalEngine
from nyxara.causal.field_resonance import ResonanceField
from nyxara.causal.hypergraph_compress import HyperGraphCompressor
from nyxara.causal.self_simulation import SelfSimulationRace
from nyxara.causal.thermo_inference import ThermodynamicMonitor, _scalarise


# --------------------------------------------------------------------------- #
# axiom_synth — the proof gate must not be fooled by sympy's own namespace
# --------------------------------------------------------------------------- #
def test_sympy_proof_does_not_mistake_a_variable_for_a_sympy_constant():
    """A variable named ``I`` is a variable — not the imaginary unit.

    Unbound, ``sympify`` parsed it as sympy's ``I`` and certified ``I*I ≡ -1``: a *false* proof
    that would have adopted an axiom untrue of the integer variable it names.
    """
    synth = AxiomSynthesizer()
    candidate = Expr("op", "*", Expr("var", name="I"), Expr("var", name="I"))
    assert synth.prove_sympy(candidate, "-1", ["I"]).proved is False


@pytest.mark.parametrize("name", ["I", "E", "S", "N", "beta", "gamma"])
def test_sympy_proof_survives_every_colliding_variable_name(name):
    synth = AxiomSynthesizer()
    expr = Expr("op", "+", Expr("var", name=name), Expr("var", name=name))
    proof = synth.prove_sympy(expr, f"2*{name}", [name])
    assert proof.proved is True, f"{name}: {proof.detail}"


def test_a_real_identity_is_still_proved():
    synth = AxiomSynthesizer()
    expr = Expr("op", "+", Expr("var", name="a"), Expr("var", name="b"))
    assert synth.prove_sympy(expr, "b + a", ["a", "b"]).proved is True


# --------------------------------------------------------------------------- #
# thermo_inference — str is a Sequence, and that used to kill the beat
# --------------------------------------------------------------------------- #
def test_a_string_surprise_reads_as_a_number_instead_of_raising():
    """``float('.')`` used to escape ``observe`` because ``str`` walked the vector branch."""
    assert _scalarise("0.75") == pytest.approx(0.75)
    assert _scalarise("not a number") == 0.0
    assert ThermodynamicMonitor().observe("2.5").surprise == pytest.approx(2.5)


def test_one_unreadable_vector_component_does_not_void_the_reading():
    assert _scalarise([3, 4, "junk"]) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# hypergraph_compress — an unknown subject must not return the whole graph
# --------------------------------------------------------------------------- #
def test_query_for_an_absent_symbol_returns_nothing():
    comp = HyperGraphCompressor()
    blob, _report = comp.compress([("a", "knows", "b"), ("a", "knows", "c"),
                                   ("x", "likes", "y")])
    assert comp.query(blob, subject="a") == [("a", "knows", "b"), ("a", "knows", "c")]
    assert comp.query(blob, subject="never-seen") == []       # was: the entire graph
    assert comp.query(blob, predicate="never-seen") == []
    # a pair that exists apart but never together is still no match
    assert comp.query(blob, subject="a", predicate="likes") == []


def test_query_with_no_filter_still_returns_everything():
    comp = HyperGraphCompressor()
    blob, _r = comp.compress([("a", "p", "b"), ("c", "q", "d")])
    assert len(comp.query(blob)) == 2


# --------------------------------------------------------------------------- #
# self_simulation — timeout_s bounded nothing and destroyed the verdict
# --------------------------------------------------------------------------- #
def test_timeout_returns_the_branches_that_finished_instead_of_raising():
    race = SelfSimulationRace(max_branches=4, max_workers=4, timeout_s=0.3)

    def rollout(h):
        time.sleep(3.0 if h == "slow" else 0.01)
        return h, (5.0 if h == "costly" else 1.0)

    started = time.perf_counter()
    verdict = race.race(["costly", "slow", "cheap"], rollout)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, "the executor used to block for the full rollout despite the timeout"
    assert verdict.collapsed is True
    assert verdict.winner.hypothesis == "cheap"
    slow = next(b for b in verdict.branches if b.hypothesis == "slow")
    assert slow.ok is False and "Timeout" in slow.error
    assert race.status()["timeouts"] == 1


def test_no_timeout_still_waits_for_every_branch():
    race = SelfSimulationRace(max_branches=4)
    verdict = race.race(["a", "b"], lambda h: (h, 1.0))
    assert all(b.ok for b in verdict.branches)


# --------------------------------------------------------------------------- #
# field_resonance — recency and the top contract
# --------------------------------------------------------------------------- #
def test_reimprinting_a_concept_refreshes_it_against_eviction():
    """The bank evicted the concept it had *just* rewritten — the inverse of its own policy."""
    bank = ResonanceField(dim=64, capacity=3)
    for label in ("a", "b", "c"):
        bank.imprint(label, f"{label} text")
    bank.imprint("a", "a refreshed")     # a use: 'a' is now the youngest
    bank.imprint("d", "d text")          # evicts the true oldest, 'b'
    assert bank.labels() == ["c", "a", "d"]


def test_reinforcing_a_concept_also_refreshes_it():
    bank = ResonanceField(dim=64, capacity=2)
    bank.imprint("a", "a text")
    bank.imprint("b", "b text")
    bank.reinforce("a")
    bank.imprint("c", "c text")
    assert "a" in bank.labels() and "b" not in bank.labels()


def test_asking_for_zero_hits_returns_zero_hits():
    bank = ResonanceField(dim=64)
    bank.imprint("a", "alpha beta")
    assert bank.resonate("alpha beta", top=0) == []      # was: one hit anyway


# --------------------------------------------------------------------------- #
# causal_knots — a live lattice needs a ceiling
# --------------------------------------------------------------------------- #
def test_a_capped_lattice_stays_bounded_and_still_rejects_contradictions():
    lattice = KnotLattice(capacity=100)
    for i in range(500):
        lattice.tie(f"n{i}", f"n{i + 1}", 1)
    status = lattice.status()
    assert status["knots"] <= 100 and status["compactions"] > 0

    fresh = KnotLattice(capacity=100)
    fresh.tie("x", "y", 1)
    with pytest.raises(KnotMutationFailure):
        fresh.tie("x", "y", -1)


def test_retract_un_ties_a_pair_and_leaves_the_rest_standing():
    lattice = KnotLattice()
    lattice.tie("a", "b", 1)
    lattice.tie("b", "c", 1)
    lattice.tie("x", "y", -1)

    with pytest.raises(KnotMutationFailure):
        lattice.tie("a", "b", -1)

    assert lattice.retract("a", "b") == 1
    lattice.tie("a", "b", -1)                       # the replacement now goes in cleanly
    assert len(lattice) == 3
    assert lattice.status()["retractions"] == 1

    # the parity of everything else survived the rebuild
    with pytest.raises(KnotMutationFailure):
        lattice.tie("b", "c", -1)
    with pytest.raises(KnotMutationFailure):
        lattice.tie("x", "y", 1)


def test_retract_is_direction_agnostic():
    lattice = KnotLattice()
    lattice.tie("a", "b", 1)
    assert lattice.retract("b", "a") == 1           # a strand ties a pair, not an arrow


def test_retracting_something_never_tied_is_a_no_op():
    lattice = KnotLattice()
    lattice.tie("a", "b", 1)
    assert lattice.retract("q", "z") == 0
    assert len(lattice) == 1 and lattice.status()["retractions"] == 0


def test_retract_removes_every_strand_between_the_pair():
    lattice = KnotLattice()
    lattice.tie("a", "b", 1, reason="first")
    lattice.tie("a", "b", 1, reason="restated")     # consistent, so recorded again
    assert len(lattice) == 2
    assert lattice.retract("a", "b") == 2
    assert len(lattice) == 0


def test_retract_does_not_resurrect_transitive_parity():
    """a—b and b—c both concordant force a—c concordant; dropping a—b must free that."""
    lattice = KnotLattice()
    lattice.tie("a", "b", 1)
    lattice.tie("b", "c", 1)
    assert lattice.check([("a", "c", -1)]).consistent is False

    lattice.retract("a", "b")
    assert lattice.check([("a", "c", -1)]).consistent is True


def test_an_uncapped_lattice_is_still_the_default():
    lattice = KnotLattice()
    for i in range(200):
        lattice.tie(f"n{i}", f"n{i + 1}", 1)
    assert lattice.status()["knots"] == 200


def test_a_dry_check_never_compacts_the_live_lattice():
    lattice = KnotLattice(capacity=4)
    lattice.tie("a", "b", 1)
    before = lattice.status()["compactions"]
    lattice.check([(f"q{i}", f"q{i + 1}", 1) for i in range(50)])
    assert lattice.status()["compactions"] == before


# --------------------------------------------------------------------------- #
# engine — flags that were read but never used
# --------------------------------------------------------------------------- #
def test_knot_gate_abstains_is_on_by_default_and_actually_produces_a_verdict():
    """The flag was stored and then consumed by nothing at all — the gate had no teeth."""
    engine = CausalEngine()                      # no settings: must match CausalEngineConfig
    assert engine.knot_gate_abstains is True

    turn = engine.turn("anything", claims=[("rain", "wet", 1), ("rain", "wet", -1)])
    assert turn.consistent is False
    assert turn.abstain is True
    assert turn.to_dict()["abstain"] is True


def test_the_abstain_verdict_can_be_switched_off():
    engine = CausalEngine()
    engine.knot_gate_abstains = False
    turn = engine.turn("anything", claims=[("rain", "wet", 1), ("rain", "wet", -1)])
    assert turn.consistent is False and turn.abstain is False


def test_turn_honours_configured_resonate_top_without_being_told():
    engine = CausalEngine()
    for i in range(5):
        engine.imprint(f"c{i}", "shared vocabulary alpha beta gamma delta")
    engine.resonate_top = 2
    assert len(engine.turn("shared vocabulary alpha beta").resonance) == 2


def _engine_without_gaps() -> CausalEngine:
    """An engine whose queries always resonate, so no phase-shift anchor confuses the count."""
    engine = CausalEngine()
    engine.imprint("topic", "topic under discussion smoking cancer rain flooding")
    return engine


def test_committing_claims_catches_a_contradiction_across_turns():
    """Without commit the lattice never grew, so only within-turn clashes were ever caught."""
    engine = _engine_without_gaps()
    first = engine.turn("topic under discussion", claims=[("smoking", "cancer", 1)], commit=True)
    assert first.consistent is True and first.committed == 1
    assert len(engine.lattice) == 1

    second = engine.turn("topic under discussion", claims=[("smoking", "cancer", -1)],
                         commit=True)
    assert second.consistent is False and second.abstain is True
    assert second.committed == 0
    assert len(engine.lattice) == 1          # the contradiction was refused, not absorbed


def test_a_turn_without_commit_leaves_the_lattice_untouched():
    engine = _engine_without_gaps()
    turn = engine.turn("topic under discussion", claims=[("a", "b", 1)])
    assert turn.consistent is True and turn.committed == 0
    assert len(engine.lattice) == 0


# --------------------------------------------------------------------------- #
# polymorph_runtime — the indirection has to stay single
# --------------------------------------------------------------------------- #
def test_reregistering_a_key_retargets_the_cell_callers_already_hold():
    """A second ``register`` used to mint a new cell, pinning every existing caller to the old
    implementation forever — while later swaps still reported success."""
    from nyxara.causal.polymorph_runtime import PolymorphRuntime

    runtime = PolymorphRuntime()
    held = runtime.register("router", "first")      # a caller keeps this reference
    again = runtime.register("router", "second")

    assert again is held
    assert held.ref == "second"
    assert runtime.current("router") == "second"

    outcome = runtime.hot_swap("router", "third")
    assert outcome.swapped is True
    assert held.ref == "third"                      # the swap reaches the live caller


def test_a_first_registration_still_makes_a_new_cell():
    from nyxara.causal.polymorph_runtime import PolymorphRuntime

    runtime = PolymorphRuntime()
    a = runtime.register("a", 1)
    b = runtime.register("b", 2)
    assert a is not b and a.generation == 0


def test_beat_thermo_is_public_and_keeps_its_private_alias():
    engine = CausalEngine()
    assert engine.beat_thermo is True
    assert engine._beat_thermo is True        # nyxara.void.heartbeat read the private name
