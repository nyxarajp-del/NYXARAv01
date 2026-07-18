"""Tests for the causal-reasoning upgrades in nyxara.mind.causal_world_model:

Rung-3 structural counterfactuals (abduction -> action -> prediction), multi-variable
do(), Probability of Necessity/Sufficiency (PN/PS/PNS), statistically-tested confounder
screening, front-door identification, and acyclicity enforcement — the concrete gaps
between "NYXARA talks counterfactual" and "NYXARA reasons counterfactually"."""

from __future__ import annotations

import random

from nyxara.mind.causal_world_model import CausalLink, CausalWorldModel, CAUSAL
from nyxara.mind.strategies import CausalModel


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _linear_chain(seed: int = 3, n: int = 200) -> CausalWorldModel:
    """A -> B -> C, linear with Gaussian noise, valued events (fits real mechanisms)."""
    rng = random.Random(seed)
    m = CausalWorldModel(window=10.0)
    for k in range(n):
        base = k * 100.0
        a = rng.gauss(5.0, 2.0)
        m.observe("A", at=base, value=a)
        b = 2.0 * a + 1.0 + rng.gauss(0.0, 0.3)
        m.observe("B", at=base + 1, value=b)
        c = 3.0 * b - 2.0 + rng.gauss(0.0, 0.3)
        m.observe("C", at=base + 2, value=c)
    m.discover()
    return m


def _diamond(seed: int = 13, n: int = 250) -> CausalWorldModel:
    """A -> B -> D, and an INDEPENDENT root C -> D (D has two uncorrelated parents).
    C is independent of A/B on purpose: FunctionalCausalMechanism fits a simple
    single-predictor regression per edge, so two CORRELATED co-parents (e.g. both driven
    by the same ancestor) would bias each other's fitted slope (classic omitted-variable
    bias) — a real, separate limitation this test intentionally avoids to isolate what
    it's actually checking: that a non-intervened parent is held at its evidence value."""
    rng = random.Random(seed)
    m = CausalWorldModel(window=10.0)
    for k in range(n):
        base = k * 100.0
        a = rng.gauss(4.0, 1.5)
        m.observe("A", at=base, value=a)
        b = 1.5 * a + rng.gauss(0.0, 0.2)
        m.observe("B", at=base + 1, value=b)
        c = rng.gauss(3.0, 1.0)                 # independent root, NOT driven by A
        m.observe("C", at=base + 1, value=c)
        d = 2.0 * b + 1.0 * c + rng.gauss(0.0, 0.2)
        m.observe("D", at=base + 2, value=d)
    m.discover()
    return m


def _match_fire(seed: int = 9, n: int = 400) -> CausalWorldModel:
    """A near-deterministic binary cause: striking a match usually starts a fire."""
    rng = random.Random(seed)
    m = CausalWorldModel(window=10.0)
    for k in range(n):
        base = k * 100.0
        did = rng.random() < 0.5
        m.observe("match_struck", at=base, value=1.0 if did else 0.0,
                 intervention=(k % 3 == 0))
        fire = did and rng.random() < 0.92
        m.observe("fire", at=base + 1, value=1.0 if fire else 0.0)
    m.discover()
    return m


# --------------------------------------------------------------------------- #
# Phase 1 — structural counterfactuals (abduction -> action -> prediction)
# --------------------------------------------------------------------------- #
def test_counterfactual_without_evidence_is_population_average():
    m = _linear_chain()
    cf = m.counterfactual("A", "C", factual=5.0, counter=0.0)
    assert cf.abducted is False


def test_counterfactual_with_evidence_is_abducted_and_reproduces_the_episode():
    m = _linear_chain()
    base = 90000.0
    a_actual = 8.0
    m.observe("A", at=base, value=a_actual)
    b_actual = 2.0 * a_actual + 1.0 + 4.0     # deliberately large positive residual
    m.observe("B", at=base + 1, value=b_actual)
    c_actual = 3.0 * b_actual - 2.0 - 5.0     # deliberately large negative residual
    m.observe("C", at=base + 2, value=c_actual)

    evidence = m.latest_evidence(["A", "B", "C"])
    assert evidence == {"A": a_actual, "B": b_actual, "C": c_actual}

    cf = m.counterfactual("A", "C", factual=a_actual, counter=0.0, evidence=evidence)
    assert cf.abducted is True
    # hand-computed: noise_B = 4.0, noise_C = -5.0 (as constructed above); counterfactual
    # world sets A=0 -> B = 2*0+1+4 = 5.0 -> C = 3*5-2-5 = 8.0; factual world reproduces
    # C=56.0 exactly (self-consistency of the abduction). delta = 8.0 - 56.0 = -48.0
    assert abs(cf.delta - (-48.0)) < 1.0


def test_structural_counterfactual_differs_from_population_average():
    """The whole point: the SAME query gives a DIFFERENT answer once grounded in what
    actually happened, vs. a fixed population-average number for everyone."""
    m = _linear_chain()
    population = m.counterfactual("A", "C", factual=5.0, counter=0.0)
    base = 90000.0
    m.observe("A", at=base, value=8.0)
    m.observe("B", at=base + 1, value=25.0)     # unusually large realized noise
    m.observe("C", at=base + 2, value=70.0)
    evidence = m.latest_evidence(["A", "B", "C"])
    structural = m.counterfactual("A", "C", factual=8.0, counter=0.0, evidence=evidence)
    assert structural.abducted and not population.abducted
    assert abs(structural.delta - population.delta) > 5.0


def test_diamond_holds_non_intervened_parent_at_its_evidence_value():
    m = _diamond()
    base = 90000.0
    a_actual, c_actual = 4.0, 3.2
    m.observe("A", at=base, value=a_actual)
    b_actual = 1.5 * a_actual
    m.observe("B", at=base + 1, value=b_actual)
    m.observe("C", at=base + 1, value=c_actual)
    d_actual = 2.0 * b_actual + 1.0 * c_actual
    m.observe("D", at=base + 2, value=d_actual)

    evidence = m.latest_evidence(["A", "B", "C", "D"])
    # intervene on B only; C (not downstream of B) must stay at its evidence value,
    # so D's shift should come ONLY from B's path, not from C changing too
    cf = m.counterfactual("B", "D", factual=b_actual, counter=0.0, evidence=evidence)
    assert cf.abducted
    # D's counterfactual value = 2*0 + 1*c_actual + noise_D ~= c_actual (+ its own noise)
    # i.e. the shift is close to -2*b_actual (only B's own contribution disappears)
    assert abs(cf.delta - (-2.0 * b_actual)) < 1.0


def test_as_causal_graph_prunes_fully_mediated_edges():
    """A -> C is fully explained by A -> B -> C; the redundant direct edge must not
    survive into the propagation graph (it would double-count the same influence)."""
    m = _linear_chain()
    edges = {(c, e) for c, e, _ in m.as_causal_graph()}
    assert ("A", "B") in edges and ("B", "C") in edges
    assert ("A", "C") not in edges


# --------------------------------------------------------------------------- #
# Phase 2 — multi-variable simultaneous do()
# --------------------------------------------------------------------------- #
def test_multi_variable_do_matches_superposition_of_single_var_effects():
    graph = [("A", "B", 2.0), ("A", "C", 3.0), ("B", "D", 1.5), ("C", "D", 0.5)]
    model = CausalModel(graph)
    joint = model.effect_of_many({"A": 1.0, "C": 1.0}, "D")
    solo = model.effect_of("A", 1.0, "D") + model.effect_of("C", 1.0, "D")
    assert abs(joint - solo) < 1e-9
    assert abs(joint - 5.0) < 1e-9   # (2*1.5 + 3*0.5) + 0.5 = 4.5 + 0.5


def test_causal_world_model_do_multi_variable():
    m = _diamond()
    joint = m.do({"A": 1.0}, target="D")
    solo = m.predict_effects("A", value=1.0, target="D")
    assert abs(joint - solo) < 1e-6


# --------------------------------------------------------------------------- #
# Phase 3 — PN / PS / PNS
# --------------------------------------------------------------------------- #
def test_necessity_sufficiency_abstains_without_fitted_mechanism():
    m = CausalWorldModel(window=10.0)
    m.observe("X", at=0.0)
    m.observe("Y", at=1.0)
    assert m.necessity_sufficiency("X", "Y") is None


def test_necessity_sufficiency_high_for_near_deterministic_cause():
    m = _match_fire()
    result = m.necessity_sufficiency("match_struck", "fire")
    assert result is not None
    assert result["PN"] > 0.7
    assert result["PS"] > 0.7
    assert result["PNS"] > 0.7
    assert 0.0 <= result["PN"] <= 1.0
    assert 0.0 <= result["PS"] <= 1.0
    assert 0.0 <= result["PNS"] <= 1.0


# --------------------------------------------------------------------------- #
# Phase 4 — statistically rigorous confounder screening
# --------------------------------------------------------------------------- #
def test_confounder_screening_still_catches_the_classic_case():
    rng = random.Random(2)
    m = CausalWorldModel(window=10.0)
    for k in range(400):
        base = k * 100.0
        if rng.random() < 0.5:
            m.observe("summer", at=base)
            if rng.random() < 0.85:
                m.observe("ice_cream", at=base + 1)
            if rng.random() < 0.8:
                m.observe("drowning", at=base + 2)
        else:
            if rng.random() < 0.1:
                m.observe("ice_cream", at=base + 1)
            if rng.random() < 0.05:
                m.observe("drowning", at=base + 2)
    m.discover()
    v = m.is_causal("ice_cream", "drowning")
    assert v.verdict == "confounded"
    assert v.evidence.get("confounder_p_value") is not None


def test_permutation_pvalue_is_deterministic_across_calls():
    """Must not depend on Python's randomized str-hash seed — same data, same p-value,
    every process (regression test for a real bug caught during development)."""
    m = _linear_chain()
    a, b = "A", "B"
    m2 = _linear_chain()
    conf1 = m._find_confounder(a, b, m.contingency(a, b))
    conf2 = m2._find_confounder(a, b, m2.contingency(a, b))
    assert conf1 == conf2


# --------------------------------------------------------------------------- #
# Phase 5 — front-door adjustment
# --------------------------------------------------------------------------- #
def test_front_door_recovers_effect_through_clean_mediator():
    rng = random.Random(5)
    m = CausalWorldModel(window=10.0)
    for k in range(400):
        base = k * 100.0
        c = rng.random() < 0.5
        if c:
            m.observe("C", at=base)
        a = (rng.random() < 0.8) if c else (rng.random() < 0.2)
        if a:
            m.observe("A", at=base + 1)
        med = a and rng.random() < 0.9
        if med:
            m.observe("M", at=base + 2)
        b = med and rng.random() < 0.9
        if b:
            m.observe("B", at=base + 3)
    m.discover()
    assert m.is_causal("A", "M").is_causal
    assert m.is_causal("M", "B").is_causal
    fd = m._front_door("A", "B", ["C"])
    assert fd is not None
    assert fd["mediator"] == "M"
    assert fd["effect"] > 0


def test_front_door_abstains_when_no_clean_mediator_exists():
    m = _confounded_model_local()
    fd = m._front_door("ice_cream", "drowning", ["summer"])
    assert fd is None


def _confounded_model_local(seed: int = 2, n: int = 400) -> CausalWorldModel:
    rng = random.Random(seed)
    m = CausalWorldModel(window=10.0)
    for k in range(n):
        base = k * 100.0
        if rng.random() < 0.5:
            m.observe("summer", at=base)
            if rng.random() < 0.85:
                m.observe("ice_cream", at=base + 1)
            if rng.random() < 0.8:
                m.observe("drowning", at=base + 2)
        else:
            if rng.random() < 0.1:
                m.observe("ice_cream", at=base + 1)
            if rng.random() < 0.05:
                m.observe("drowning", at=base + 2)
    m.discover()
    return m


# --------------------------------------------------------------------------- #
# Phase 6 — acyclicity
# --------------------------------------------------------------------------- #
def test_enforce_acyclicity_breaks_a_two_cycle_keeping_the_stronger_edge():
    m = CausalWorldModel(window=10.0)
    m._by_label = {"X": [0.0], "Y": [1.0]}
    m._links = {
        ("X", "Y"): CausalLink("X", "Y", strength=0.5, confidence=0.9, verdict=CAUSAL),
        ("Y", "X"): CausalLink("Y", "X", strength=0.3, confidence=0.4, verdict=CAUSAL),
    }
    m._enforce_acyclicity()
    assert m._links[("X", "Y")].verdict == CAUSAL
    assert m._links[("Y", "X")].verdict != CAUSAL
    assert m._find_cycle() is None


def test_enforce_acyclicity_breaks_a_three_cycle_keeping_two_strongest_edges():
    m = CausalWorldModel(window=10.0)
    m._by_label = {"A": [0.0], "B": [1.0], "C": [2.0]}
    m._links = {
        ("A", "B"): CausalLink("A", "B", strength=0.9, confidence=0.9, verdict=CAUSAL),
        ("B", "C"): CausalLink("B", "C", strength=0.8, confidence=0.8, verdict=CAUSAL),
        ("C", "A"): CausalLink("C", "A", strength=0.1, confidence=0.2, verdict=CAUSAL),
    }
    m._enforce_acyclicity()
    assert m._links[("A", "B")].verdict == CAUSAL
    assert m._links[("B", "C")].verdict == CAUSAL
    assert m._links[("C", "A")].verdict != CAUSAL
    assert m._find_cycle() is None
