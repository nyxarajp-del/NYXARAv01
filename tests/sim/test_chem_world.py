"""Tests for nyxara.sim.chem_world — reaction kinetics whose rate genuinely emerges.

The simulator must integrate mass-action kinetics so the quantities a chemist measures — the
initial rate, the half-life, the temperature dependence — emerge from the dynamics rather than
being hard-coded. All hermetic, no LLM.
"""

from __future__ import annotations

import math

from nyxara.sim.chem_world import ReactionWorld, arrhenius_k, GAS_CONSTANT


def test_first_order_half_life_matches_ln2_over_k():
    for k in (0.3, 0.5, 0.8, 1.2):
        hl = ReactionWorld(k=k, order=1, a0=1.0).simulate().half_life()
        assert hl is not None
        assert abs(hl - math.log(2.0) / k) < 0.05


def test_first_order_initial_rate_is_k_times_concentration():
    # the rate law rate = k·[A] for first order — measured, not assumed
    for k, a0 in ((0.5, 2.0), (0.8, 1.0), (0.3, 1.5)):
        rate = ReactionWorld(k=k, order=1, a0=a0).initial_rate()
        assert abs(rate - k * a0) < 0.05 * (k * a0) + 0.02


def test_second_order_decays_faster_tail():
    # second-order kinetics have a long slow tail vs first-order — a qualitatively different curve
    first = ReactionWorld(k=1.0, order=1, a0=1.0).simulate()
    second = ReactionWorld(k=1.0, order=2, a0=1.0).simulate()
    # at equal late time the second-order reactant lingers higher (1/[A] linear vs exponential)
    n = min(len(first.A), len(second.A)) - 1
    assert second.A[n] >= first.A[n] - 1e-6


def test_arrhenius_rate_rises_with_temperature():
    ks = [arrhenius_k(a_pre=1e6, e_a=5e4, temperature=T) for T in (280, 320, 360, 400)]
    assert all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))
    # linearised: ln k is linear in 1/T with slope −Ea/R
    import math as _m
    slope = ((_m.log(ks[-1]) - _m.log(ks[0]))
             / (1.0 / 400 - 1.0 / 280))
    assert abs(slope - (-5e4 / GAS_CONSTANT)) < 0.05 * (5e4 / GAS_CONSTANT)


def test_determinism():
    a = ReactionWorld(k=0.5, order=1, a0=1.0).simulate()
    b = ReactionWorld(k=0.5, order=1, a0=1.0).simulate()
    assert a.A == b.A
