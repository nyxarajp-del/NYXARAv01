"""STDP: pre-before-post potentiates, post-before-pre depresses, weight stays clamped."""

from __future__ import annotations

from nyxara.nyx5.synapse import STDPRule, Synapse


def test_pre_before_post_potentiates():
    rule = STDPRule(a_plus=0.1, w_min=0.0, w_max=1.0)
    syn = Synapse(pre=0, post=1, weight=0.5)
    rule.on_pre_spike(syn, 0.0)          # pre fires first
    w_after = rule.on_post_spike(syn, 5.0)  # post fires shortly after
    assert w_after > 0.5                 # potentiated


def test_post_before_pre_depresses():
    rule = STDPRule(a_minus=0.1, w_min=0.0, w_max=1.0)
    syn = Synapse(pre=0, post=1, weight=0.5)
    rule.on_post_spike(syn, 0.0)         # post fires first
    w_after = rule.on_pre_spike(syn, 5.0)   # pre fires after -> anti-causal
    assert w_after < 0.5                 # depressed


def test_weight_clamped():
    rule = STDPRule(a_plus=10.0, a_minus=10.0, w_min=0.0, w_max=1.0)
    syn = Synapse(pre=0, post=1, weight=0.9)
    rule.on_pre_spike(syn, 0.0)
    rule.on_post_spike(syn, 0.1)
    assert syn.weight <= 1.0
    rule.on_post_spike(syn, 1.0)
    rule.on_pre_spike(syn, 1.1)
    assert syn.weight >= 0.0


def test_closer_dt_larger_change():
    rule = STDPRule(a_plus=0.1, tau_plus=20.0, w_min=0.0, w_max=1.0)
    near = Synapse(pre=0, post=1, weight=0.5)
    far = Synapse(pre=0, post=1, weight=0.5)
    rule.on_pre_spike(near, 0.0)
    rule.on_post_spike(near, 1.0)        # Δt = 1
    rule.on_pre_spike(far, 0.0)
    rule.on_post_spike(far, 40.0)        # Δt = 40
    assert (near.weight - 0.5) > (far.weight - 0.5)
