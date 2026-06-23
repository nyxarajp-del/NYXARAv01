"""Tests for nyxara.growth.topology — Dynamic Topology Expansion (runtime Net2Net brain growth).

The genome-growth path is pure-stdlib and always tested; the function-preserving torch morphisms
are tested only when torch is installed (``importorskip``).
"""
from __future__ import annotations

import pytest

from nyxara.growth.genesis import ArchitectureGenome, LayerGene, _HAS_TORCH
from nyxara.growth.topology import (
    CapacityMonitor,
    CapacitySignal,
    DynamicTopology,
    GrowthDecision,
    GrowthMode,
)
from nyxara.kernel.config import TopologyConfig


def _base_genome() -> ArchitectureGenome:
    return ArchitectureGenome(n_embd=64, block_size=16,
                              layers=[LayerGene(op="attention", n_head=4),
                                      LayerGene(op="gated_mlp")])


def test_no_pressure_no_growth():
    mon = CapacityMonitor(cfg=TopologyConfig())
    calm = CapacitySignal(problem_difficulty=0.1, saturation=0.1, loss_plateau=0.1)
    assert mon.should_grow(calm, genome=_base_genome()).should_grow is False


def test_pressure_triggers_growth():
    monitor = CapacityMonitor(cfg=TopologyConfig())
    hot = CapacitySignal(problem_difficulty=0.95, saturation=0.95, loss_plateau=0.95)
    dec = monitor.should_grow(hot, genome=_base_genome())
    assert dec.should_grow is True
    assert dec.mode in (GrowthMode.WIDEN.value, GrowthMode.DEEPEN.value, GrowthMode.BOTH.value)


def test_ceiling_blocks_growth():
    cfg = TopologyConfig(max_n_embd=64, max_layers=2)
    monitor = CapacityMonitor(cfg=cfg)
    hot = CapacitySignal(problem_difficulty=0.95, saturation=0.95, loss_plateau=0.95)
    at_ceiling = ArchitectureGenome(n_embd=64, block_size=16,
                                    layers=[LayerGene(op="gated_mlp"), LayerGene(op="glu")])
    assert monitor.should_grow(hot, genome=at_ceiling).should_grow is False


def test_genome_growth_widens_and_deepens():
    cfg = TopologyConfig(grow_dims_k=8, max_n_embd=128, max_layers=8)
    topo = DynamicTopology(cfg=cfg)
    base = _base_genome()
    _model, rep = topo.grow(base, GrowthDecision(True, GrowthMode.BOTH.value, 8, "test"))
    assert rep.grew is True
    assert rep.after_n_embd == 72          # 64 + k
    assert rep.after_layers == 3           # one identity layer appended
    assert rep.after_params >= rep.before_params
    # the original genome is not mutated in place
    assert base.n_embd == 64 and len(base.layers) == 2


def test_genome_growth_respects_ceiling():
    cfg = TopologyConfig(grow_dims_k=100, max_n_embd=70, max_layers=2)
    topo = DynamicTopology(cfg=cfg)
    _m, rep = topo.grow(_base_genome(), GrowthDecision(True, GrowthMode.BOTH.value, 100, "x"))
    assert rep.after_n_embd == 70          # clamped to the width ceiling
    assert rep.after_layers == 2           # depth ceiling already reached → no new layer


def test_maybe_grow_is_cheap_noop_without_signal():
    topo = DynamicTopology(cfg=TopologyConfig())
    assert topo.maybe_grow() is None
    assert topo.maybe_grow(source=_base_genome()) is None  # no signal → no growth


def test_no_growth_decision_returns_source_unchanged():
    topo = DynamicTopology(cfg=TopologyConfig())
    base = _base_genome()
    model, rep = topo.grow(base, GrowthDecision(False, GrowthMode.NONE.value, 0, "no"))
    assert rep.grew is False and model is base


def test_report_round_trips():
    topo = DynamicTopology(cfg=TopologyConfig())
    _m, rep = topo.grow(_base_genome(), GrowthDecision(True, GrowthMode.WIDEN.value, 8, "t"))
    d = rep.to_dict()
    assert d["after_n_embd"] >= d["before_n_embd"] and "reason" in d


@pytest.mark.skipif(not _HAS_TORCH, reason="torch required for function-preserving morphisms")
def test_net2deepernet_preserves_function_exactly():
    from nyxara.growth.genesis import GenesisModel, _GENESIS_SEED

    topo = DynamicTopology(cfg=TopologyConfig(max_layers=8))
    gm = GenesisModel(_base_genome())
    gm.train_on(list(_GENESIS_SEED), steps=20, seed=1)
    deeper, rep = topo.grow(gm, GrowthDecision(True, GrowthMode.DEEPEN.value, 0, "deepen"))
    assert rep.preserved is True            # x + 0 = x → bit-identical at the moment of growth
    assert deeper.param_count() > gm.param_count()


@pytest.mark.skipif(not _HAS_TORCH, reason="torch required for function-preserving morphisms")
def test_net2widernet_grows_parameters():
    from nyxara.growth.genesis import GenesisModel, _GENESIS_SEED

    topo = DynamicTopology(cfg=TopologyConfig(grow_dims_k=8, max_n_embd=128))
    gm = GenesisModel(_base_genome())
    gm.train_on(list(_GENESIS_SEED), steps=20, seed=1)
    wider, rep = topo.grow(gm, GrowthDecision(True, GrowthMode.WIDEN.value, 8, "widen"))
    assert rep.after_n_embd == 72
    assert wider.param_count() > gm.param_count()
