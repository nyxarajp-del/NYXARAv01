"""NYXARA · growth/synth_causal.py — a fifth domain: intervention, not correlation (🔀).

A language model sees "smoking" and "cancer" co-occur and learns a correlation. What it cannot
learn from co-occurrence is that ``P(Y | do(X))`` and ``P(Y | X)`` are different quantities, and
that the gap between them is exactly where confounding lives.

The corpus-annotation route to this was measured and rejected: extracting causal claims from
prose is 8 regexes with ≤4-word node labels, and the aggregation is O(N²) with a ~48% spurious
refusal rate once the lattice merges into one component. More importantly, structure *extracted*
from text is a guess. So this domain **generates the model first** — the mechanisms are known,
therefore every interventional and counterfactual answer here is ground truth rather than an
estimate, the same discipline that makes :mod:`~nyxara.growth.synth_math` safe at scale.

Five shapes, each teaching something a correlational model gets wrong:

* **intervention** — computed from the true model, and *deliberately* including cases where
  ``do(X)`` and observing ``X`` disagree. A corpus where they always agree teaches the exact
  confusion this domain exists to remove.
* **backdoor adjustment** — the naive answer is shown as wrong, then the admissible set is
  identified by Pearl's criterion and the corrected answer derived.
* **collider** — conditioning on a common effect *creates* a dependence between independent
  causes. This is the mistake that looks most like good practice.
* **counterfactual** — full abduction → action → prediction on a known model: infer the noise
  from what actually happened, intervene, recompute.
* **unidentifiable** — the confounder is latent, so the honest answer is "this cannot be
  determined from these observations, and here is why". A model shown only identifiable queries
  learns that an answer always exists, the same failure the unanswerable retrieval passages
  exist to prevent.
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Optional, Tuple

from nyxara.growth.scm import (
    SCM,
    backdoor_sets,
    d_separated,
    descendants,
    random_scm,
)
from nyxara.growth.synth_common import Generator

__all__ = ["GENERATORS", "generators"]

Built = Optional[Tuple[str, str]]


def _num(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _graph_block(scm: SCM) -> str:
    edges = ", ".join(f"{p} -> {c}" for p, c in scm.edges())
    latent = (f"\nUnobserved: {', '.join(sorted(scm.latent))}" if scm.latent else "")
    return f"Causal graph: {edges}{latent}\nStructural equations:\n{scm.describe()}"


def _pick_treatment_outcome(scm: SCM, rng: random.Random) -> Optional[Tuple[str, str]]:
    pairs = [(t, y) for t in scm.observed() for y in scm.observed()
             if t != y and y in descendants(scm, t)]
    return pairs[rng.randrange(len(pairs))] if pairs else None


def _zero_noise(scm: SCM) -> dict:
    return {n: Fraction(0) for n in scm.nodes}


# --------------------------------------------------------------------------- #
def _intervention(rng: random.Random) -> Built:
    """``P(Y | do(X=x))`` computed from the true model, contrasted with observing X."""
    scm = random_scm(rng, shape="confounded")
    pair = _pick_treatment_outcome(scm, rng)
    if pair is None:
        return None
    treatment, outcome = pair
    noise = {n: Fraction(rng.randint(-4, 4)) for n in scm.nodes}
    value = Fraction(rng.randint(1, 20))

    world = scm.simulate(noise)
    intervened = scm.simulate(noise, do={treatment: value})
    if intervened[outcome] == world[outcome]:
        return None                       # nothing to teach if the intervention changes nothing

    return (f"{_graph_block(scm)}\n\nWith noise terms "
            f"{', '.join(f'{k}={_num(v)}' for k, v in sorted(noise.items()))}, "
            f"what is {outcome} under do({treatment} = {_num(value)})?",
            f"An intervention replaces {treatment}'s equation entirely: it keeps its children "
            f"and loses its parents.\n"
            f"  set {treatment} = {_num(value)}\n"
            f"  then recompute every descendant in topological order\n"
            f"Under do({treatment} = {_num(value)}): {outcome} = {_num(intervened[outcome])}\n"
            f"Observationally, {treatment} would have been "
            f"{_num(world[treatment])} and {outcome} = {_num(world[outcome])}.\n"
            f"The two differ because {treatment} has a parent that also influences {outcome}, "
            f"so observing {treatment} carries information the intervention destroys.")


def _backdoor(rng: random.Random) -> Built:
    """The naive estimate is biased; the admissible set is identified and the answer corrected."""
    scm = random_scm(rng, shape="confounded")
    pair = _pick_treatment_outcome(scm, rng)
    if pair is None:
        return None
    treatment, outcome = pair
    sets = backdoor_sets(scm, treatment, outcome)
    if not sets:
        return None
    smallest = min(sets, key=len)
    if not smallest:
        return None                          # no confounding to correct — a different lesson
    confounder = smallest[0]

    return (f"{_graph_block(scm)}\n\nWe want the causal effect of {treatment} on {outcome}. "
            f"Which variables must be adjusted for, and why is the raw association misleading?",
            f"The raw association is biased because there is a back-door path\n"
            f"  {treatment} <- {confounder} -> {outcome}\n"
            f"which carries association that is not causal: {confounder} moves both.\n\n"
            f"Back-door criterion: a set Z is admissible when no member is a descendant of "
            f"{treatment}, and Z blocks every path into {treatment}.\n"
            f"  Z = {{{', '.join(smallest)}}} contains no descendant of {treatment}\n"
            f"  and conditioning on it blocks {treatment} <- {confounder} -> {outcome}\n\n"
            f"So adjust for {', '.join(smallest)}: estimate the effect within each level of "
            f"{confounder} and average. Comparing {outcome} across {treatment} without that "
            f"adjustment measures the confounding as well as the effect.")


def _collider(rng: random.Random) -> Built:
    """Conditioning on a common effect manufactures a dependence. The seductive mistake."""
    scm = random_scm(rng, shape="collider")
    parents = [n for n in scm.nodes if not scm.parents(n)]
    if len(parents) != 2:
        return None
    a, b = parents
    collider = next(n for n in scm.nodes if len(scm.parents(n)) == 2)

    assert d_separated(scm, a, b)                    # independent unconditionally
    assert not d_separated(scm, a, b, [collider])    # dependent once conditioned

    return (f"{_graph_block(scm)}\n\nAre {a} and {b} independent? And does that change if "
            f"we only look at cases with a given value of {collider}?",
            f"Unconditionally they ARE independent. The only path between them is\n"
            f"  {a} -> {collider} <- {b}\n"
            "and it is a collider — two arrowheads meet at it — so the path is blocked "
            "and no association flows.\n\n"
            f"Conditioning on {collider} REVERSES this. Selecting cases with a fixed "
            f"{collider} opens the path: once the total is known, learning {a} tells you about "
            f"{b}, because they must together account for it.\n\n"
            f"So {a} and {b} are independent, but dependent given {collider}. Conditioning on a "
            f"common effect creates association rather than removing it — which is why "
            f"{collider} must NOT be adjusted for when estimating anything about {a} and {b}.")


def _counterfactual(rng: random.Random) -> Built:
    """Abduction → action → prediction, exact because the mechanisms are known."""
    scm = random_scm(rng, shape="confounded")
    pair = _pick_treatment_outcome(scm, rng)
    if pair is None:
        return None
    treatment, outcome = pair
    noise = {n: Fraction(rng.randint(-4, 4)) for n in scm.nodes}
    world = scm.simulate(noise)
    alternative = world[treatment] + Fraction(rng.choice([-8, -5, -3, 3, 5, 8]))
    counter = scm.counterfactual(world, treatment, alternative)
    if counter[outcome] == world[outcome]:
        return None

    observed = ", ".join(f"{k}={_num(v)}" for k, v in sorted(world.items()))
    return (f"{_graph_block(scm)}\n\nWe observed: {observed}\n\n"
            f"Would {outcome} have been different had {treatment} been "
            f"{_num(alternative)} instead?",
            "This is a counterfactual about THIS case, not a population average, so it takes "
            "three steps.\n\n"
            "1. Abduction — infer the noise this world must have had:\n"
            + "\n".join(f"     noise_{k} = {_num(v)}" for k, v in sorted(
                scm.abduct(world).items()))
            + f"\n2. Action — replace {treatment}'s equation with {treatment} = "
            f"{_num(alternative)}, holding that noise fixed.\n"
            f"3. Prediction — recompute.\n\n"
            f"Then {outcome} would have been {_num(counter[outcome])}, against "
            f"{_num(world[outcome])} actually observed — a change of "
            f"{_num(counter[outcome] - world[outcome])}.\n"
            f"Holding the noise fixed is what makes this about the case in front of us rather "
            f"than about a similar case drawn at random.")


def _unidentifiable(rng: random.Random) -> Built:
    """The confounder is latent, so the honest answer is that it cannot be determined."""
    scm = random_scm(rng, shape="latent")
    pair = _pick_treatment_outcome(scm, rng)
    if pair is None:
        return None
    treatment, outcome = pair
    observed_sets = backdoor_sets(scm, treatment, outcome, allowed=scm.observed())
    if observed_sets:
        return None                          # it IS identifiable; a different lesson
    latent = ", ".join(sorted(scm.latent))
    full_sets = backdoor_sets(scm, treatment, outcome, allowed=scm.nodes)
    if not full_sets:
        return None

    return (f"{_graph_block(scm)}\n\nCan the causal effect of {treatment} on {outcome} be "
            f"estimated from observational data on {', '.join(scm.observed())} alone?",
            f"No — and the reason is structural rather than a matter of sample size.\n\n"
            f"There is a back-door path\n  {treatment} <- {latent} -> {outcome}\n"
            f"which must be blocked for the association to be causal. The only set that blocks "
            f"it is {{{latent}}}, and {latent} is unobserved.\n\n"
            f"No adjustment among {', '.join(scm.observed())} closes that path, so every such "
            f"estimate mixes the effect with the confounding, and no amount of extra data "
            f"separates them. The effect is NOT identifiable here.\n\n"
            f"What would change the answer: measuring {latent}, an instrument for "
            f"{treatment}, or randomising {treatment} directly — each of which supplies the "
            f"information the observations lack.")


def _mechanism_shift(rng: random.Random) -> Built:
    """Change a structural equation and propagate — re-derivation, not recall."""
    scm = random_scm(rng, shape="mediated")
    order = scm.topological()
    treatment, mediator, outcome = order[0], order[1], order[2]
    noise = {n: Fraction(rng.randint(-3, 3)) for n in scm.nodes}
    world = scm.simulate(noise)

    mech = scm.mechanisms[outcome]
    factor = Fraction(rng.choice([2, 3, -1, -2]))
    shifted = SCM(scm.nodes, {**scm.mechanisms,
                              outcome: type(mech)(node=mech.node, parents=mech.parents,
                                                  coeffs=tuple(c * factor for c in mech.coeffs),
                                                  intercept=mech.intercept, kind=mech.kind,
                                                  threshold=mech.threshold)})
    after = shifted.simulate(noise)
    if after[outcome] == world[outcome]:
        return None

    return (f"{_graph_block(scm)}\n\nSuppose the mechanism producing {outcome} changed so that "
            f"its dependence on {mediator} is multiplied by {_num(factor)}. With the same noise "
            f"terms, what happens to {outcome}?",
            f"A mechanism change is not an intervention on a value — the equation itself is "
            f"replaced, so everything downstream must be re-derived rather than recalled.\n\n"
            f"  before: {mech.formula()}\n"
            f"  after:  {shifted.mechanisms[outcome].formula()}\n\n"
            f"{treatment} and {mediator} are upstream and are unchanged: "
            f"{treatment} = {_num(world[treatment])}, {mediator} = {_num(world[mediator])}.\n"
            f"Recomputing {outcome} under the new equation gives {_num(after[outcome])}, "
            f"against {_num(world[outcome])} before.\n"
            f"Only the descendants of the changed equation move; the rest of the model is "
            f"untouched, which is what makes this answerable at all.")


# --------------------------------------------------------------------------- #
_SCENARIO_COUNT = 8
_COEFFS = 7 ** 3
_NOISE = 9 ** 3

GENERATORS: Tuple[Generator, ...] = (
    Generator("causal_intervention", "causal", _intervention,
              space=_SCENARIO_COUNT * _COEFFS * _NOISE * 20, weight=1.4),
    Generator("causal_backdoor", "causal", _backdoor,
              space=_SCENARIO_COUNT * _COEFFS * 11 * 11, weight=1.3),
    Generator("causal_collider", "causal", _collider,
              space=_SCENARIO_COUNT * _COEFFS * 11 * 11, weight=1.0),
    Generator("causal_counterfactual", "causal", _counterfactual,
              space=_SCENARIO_COUNT * _COEFFS * _NOISE * 6, weight=1.4),
    Generator("causal_unidentifiable", "causal", _unidentifiable,
              space=_SCENARIO_COUNT * _COEFFS * 11 * 11, weight=1.0),
    Generator("causal_mechanism_shift", "causal", _mechanism_shift,
              space=_SCENARIO_COUNT * _COEFFS * 343 * 4, weight=1.1),
)


def generators() -> Tuple[Generator, ...]:
    return GENERATORS
