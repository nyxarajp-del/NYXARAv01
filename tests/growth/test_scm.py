"""Tests for growth/scm.py and growth/synth_causal.py — the causal domain.

These guard the criteria themselves, because a causal corpus generated from a wrong criterion
teaches the wrong criterion, confidently, at scale. The collider rule gets the most attention:
it runs opposite to the intuition (conditioning on a common effect *creates* association), and
I had it inverted on the first implementation — `d_sep(A, C | D)` for `A -> B <- C, B -> D`
reported separation, because I tested "is this node among the conditioning set's descendants"
rather than "does this node's descendant set meet the conditioning set". Those are different
relations, and only one of them is d-separation.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from nyxara.growth.scm import (
    SCM,
    Mechanism,
    backdoor_sets,
    d_separated,
    descendants,
    random_scm,
)
from nyxara.growth.synth_causal import GENERATORS as CAUSAL


def _m(node: str, parents=()) -> Mechanism:
    return Mechanism(node=node, parents=tuple(parents),
                     coeffs=tuple(Fraction(1) for _ in parents))


def _chain() -> SCM:
    return SCM(("A", "B", "C"), {"A": _m("A"), "B": _m("B", ("A",)), "C": _m("C", ("B",))})


def _fork() -> SCM:
    return SCM(("A", "B", "C"), {"B": _m("B"), "A": _m("A", ("B",)), "C": _m("C", ("B",))})


def _collider() -> SCM:
    return SCM(("A", "B", "C"), {"A": _m("A"), "C": _m("C"), "B": _m("B", ("A", "C"))})


def _collider_with_child() -> SCM:
    return SCM(("A", "B", "C", "D"),
               {"A": _m("A"), "C": _m("C"), "B": _m("B", ("A", "C")), "D": _m("D", ("B",))})


# --------------------------------------------------------------------------- #
# d-separation, one rule at a time
# --------------------------------------------------------------------------- #
def test_chain_is_blocked_only_by_conditioning_on_the_middle() -> None:
    chain = _chain()
    assert not d_separated(chain, "A", "C")
    assert d_separated(chain, "A", "C", ["B"])


def test_fork_is_blocked_only_by_conditioning_on_the_common_cause() -> None:
    fork = _fork()
    assert not d_separated(fork, "A", "C")
    assert d_separated(fork, "A", "C", ["B"])


def test_collider_is_blocked_until_you_condition_on_it() -> None:
    """The rule that runs backwards: conditioning OPENS this path."""
    coll = _collider()
    assert d_separated(coll, "A", "C"), "an unconditioned collider must block"
    assert not d_separated(coll, "A", "C", ["B"]), "conditioning on a collider must open it"


def test_conditioning_on_a_colliders_descendant_also_opens_it() -> None:
    """The regression. A descendant of a collider activates it just as the collider does."""
    graph = _collider_with_child()
    assert d_separated(graph, "A", "C"), "unconditioned, the collider blocks"
    assert not d_separated(graph, "A", "C", ["D"]), (
        "D is downstream of the collider B, so conditioning on it opens A -> B <- C")


def test_d_separation_is_symmetric() -> None:
    for graph in (_chain(), _fork(), _collider(), _collider_with_child()):
        for given in ([], ["B"]):
            assert d_separated(graph, "A", "C", given) == d_separated(graph, "C", "A", given)


# --------------------------------------------------------------------------- #
# Back-door criterion
# --------------------------------------------------------------------------- #
def test_the_confounder_is_an_admissible_adjustment_set() -> None:
    scm = SCM(("U", "T", "Y"),
              {"U": _m("U"), "T": _m("T", ("U",)), "Y": _m("Y", ("T", "U"))})
    sets = backdoor_sets(scm, "T", "Y")
    assert ("U",) in sets
    assert () not in sets, "the empty set cannot be admissible when a back-door path is open"


def test_a_descendant_of_the_treatment_is_never_admissible() -> None:
    """Adjusting for a mediator removes the very effect being measured."""
    scm = SCM(("T", "M", "Y"),
              {"T": _m("T"), "M": _m("M", ("T",)), "Y": _m("Y", ("M",))})
    for candidate in backdoor_sets(scm, "T", "Y"):
        assert "M" not in candidate


def test_no_confounding_means_the_empty_set_is_admissible() -> None:
    scm = SCM(("T", "Y"), {"T": _m("T"), "Y": _m("Y", ("T",))})
    assert () in backdoor_sets(scm, "T", "Y")


def test_a_latent_confounder_leaves_no_admissible_observed_set() -> None:
    scm = SCM(("U", "T", "Y"),
              {"U": _m("U"), "T": _m("T", ("U",)), "Y": _m("Y", ("T", "U"))},
              latent=frozenset({"U"}))
    assert backdoor_sets(scm, "T", "Y", allowed=scm.observed()) == []
    assert ("U",) in backdoor_sets(scm, "T", "Y", allowed=scm.nodes)


# --------------------------------------------------------------------------- #
# The model itself
# --------------------------------------------------------------------------- #
def test_intervention_severs_the_parents_but_keeps_the_children() -> None:
    scm = SCM(("U", "T", "Y"),
              {"U": _m("U"), "T": _m("T", ("U",)), "Y": _m("Y", ("T", "U"))})
    noise = {"U": Fraction(5), "T": Fraction(1), "Y": Fraction(0)}
    world = scm.simulate(noise)
    forced = scm.simulate(noise, do={"T": Fraction(100)})
    assert forced["T"] == 100, "the intervened value must hold"
    assert forced["U"] == world["U"], "an intervention must not change the treatment's parents"
    assert forced["Y"] != world["Y"], "descendants must respond"


def test_abduction_recovers_the_exact_noise() -> None:
    for seed in range(8):
        scm = random_scm(random.Random(seed))
        noise = {n: Fraction(random.Random(seed + 1).randint(-6, 6)) for n in scm.nodes}
        world = scm.simulate(noise)
        assert scm.abduct(world) == noise


def test_a_counterfactual_to_the_observed_value_reproduces_the_world() -> None:
    """Self-consistency: 'what if X had been what it was' must change nothing."""
    for seed in range(8):
        scm = random_scm(random.Random(seed))
        noise = {n: Fraction(random.Random(seed + 3).randint(-6, 6)) for n in scm.nodes}
        world = scm.simulate(noise)
        for node in scm.nodes:
            assert scm.counterfactual(world, node, world[node]) == world


def test_a_counterfactual_leaves_non_descendants_untouched() -> None:
    for seed in range(8):
        scm = random_scm(random.Random(seed))
        noise = {n: Fraction(random.Random(seed + 5).randint(-6, 6)) for n in scm.nodes}
        world = scm.simulate(noise)
        for node in scm.nodes:
            counter = scm.counterfactual(world, node, world[node] + Fraction(7))
            for other in scm.nodes:
                if other != node and other not in descendants(scm, node):
                    assert counter[other] == world[other]


# --------------------------------------------------------------------------- #
# Generated documents
# --------------------------------------------------------------------------- #
def _pairs(name: str, n: int, seed: int = 5):
    gen = next(g for g in CAUSAL if g.name == name)
    rng = random.Random(seed)
    out = []
    for _ in range(n * 12):
        built = gen.fn(rng)
        if built:
            out.append(built)
        if len(out) >= n:
            break
    assert out, f"{name} produced nothing"
    return out


@pytest.mark.parametrize("gen", CAUSAL, ids=lambda g: g.name)
def test_causal_generators_are_diverse(gen) -> None:
    rng = random.Random(23)
    built = [b for b in (gen.fn(rng) for _ in range(240)) if b is not None]
    assert len(built) >= 40, f"{gen.name} declined too often"
    assert len(set(built)) >= len(built) * 0.9


def test_collider_documents_state_the_dependence_correctly() -> None:
    for prompt, answer in _pairs("causal_collider", 12):
        assert "ARE independent" in answer
        assert "REVERSES" in answer or "opens the path" in answer
        assert "must NOT be adjusted" in answer


def test_backdoor_documents_name_a_real_adjustment_set() -> None:
    for prompt, answer in _pairs("causal_backdoor", 12):
        assert "Back-door criterion" in answer
        assert "no descendant" in answer or "contains no descendant" in answer


def test_unidentifiable_documents_refuse_rather_than_estimate() -> None:
    """A model shown only identifiable queries learns that an answer always exists."""
    for prompt, answer in _pairs("causal_unidentifiable", 10):
        assert "NOT identifiable" in answer
        assert "unobserved" in answer or "latent" in answer.lower()


def test_counterfactual_documents_show_all_three_steps() -> None:
    for prompt, answer in _pairs("causal_counterfactual", 12):
        assert "Abduction" in answer and "Action" in answer and "Prediction" in answer


def test_intervention_documents_contrast_do_with_observation() -> None:
    """The whole point of the domain: P(Y|do(X)) and P(Y|X) are different quantities."""
    for prompt, answer in _pairs("causal_intervention", 12):
        assert "Observationally" in answer
        assert "do(" in prompt


def test_the_causal_domain_is_reachable_through_the_facade() -> None:
    from nyxara.growth.synth_data import generate_domain_docs

    docs = list(generate_domain_docs("causal", 200, seed=2))
    assert len(set(docs)) >= 180


def test_causal_is_a_first_class_corpus_domain() -> None:
    from nyxara.growth.corpus import DEFAULT_DOMAIN_WEIGHTS, DOMAINS

    assert "causal" in DOMAINS
    assert set(DEFAULT_DOMAIN_WEIGHTS) == set(DOMAINS)
    assert abs(sum(DEFAULT_DOMAIN_WEIGHTS.values()) - 1.0) < 1e-9
