"""Tests for nyxara.growth.adaptation — self-driven adaptation to a brand-new environment.

Dropped into an unfamiliar environment NYXARA does the whole thing herself, deterministically:
survey the black boxes → model each with her own open-world generalizer → measure the pressure they
put her under → re-organize her own brain (topology growth) when that pressure is real → remember
every system so re-entry is instant. The non-negotiable property, asserted here, is that **no language
model is anywhere in the loop** — the reasoning is her own symbolic law search.
"""
from __future__ import annotations

import math
import random

import pytest

from nyxara.growth.adaptation import AdaptationReport, EnvironmentAdapter
from nyxara.growth.env_registry import EnvironmentRegistry
from nyxara.growth.open_world import DomainSpec, Verdict


def _env():
    rng = random.Random(0)
    return {
        "linear": lambda x: 3 * x - 2,
        "quadratic": lambda x: 2 * x * x + x - 5,
        "sine": lambda x: 3.0 * math.sin(1.0 * x) + 2.0,
        "noise": lambda x: rng.random() * 100,        # she must NOT bluff this one
    }


def test_adapt_models_the_knowable_and_is_honest_about_the_rest(tmp_path):
    reg = EnvironmentRegistry(path=tmp_path / "env.json", min_checks=3)
    adapter = EnvironmentAdapter(registry=reg, seed=7)
    rep = adapter.adapt(_env(), label="test-env")
    assert isinstance(rep, AdaptationReport)
    assert rep.n_systems == 4
    assert rep.modelled >= 3          # linear, quadratic, sine all crack cleanly
    assert rep.unmodelled >= 1        # the noise box is honestly left unmodelled
    assert 0.0 < rep.pressure <= 1.0
    d = rep.to_dict()
    assert {"environment", "modelled", "unmodelled", "pressure", "outcomes"} <= set(d)


def test_re_entry_recognizes_previously_cracked_systems(tmp_path):
    reg = EnvironmentRegistry(path=tmp_path / "env.json", min_checks=3)
    adapter = EnvironmentAdapter(registry=reg, seed=7)
    env = _env()
    adapter.adapt(env, label="env")
    rep2 = adapter.adapt(env, label="env")            # second visit
    assert rep2.recognized >= 3                        # the modellable ones are recognised instantly


def test_declarative_environment_over_an_api_boundary(tmp_path):
    reg = EnvironmentRegistry(path=tmp_path / "env.json", min_checks=3)
    adapter = EnvironmentAdapter(registry=reg, seed=7)
    rep = adapter.adapt([{"family": "affine", "params": {"w": [1.0, 2.0]}, "dims": 1},
                         {"family": "sinusoidal",
                          "params": {"a": 3.0, "b": 0.0, "c": 2.0, "omega": 1.0}, "dims": 1}],
                        label="declared")
    assert rep.n_systems == 2 and rep.modelled == 2


def test_pressure_triggers_brain_reorganization(tmp_path):
    """A hard environment (nothing modellable) under pressure asks the topology engine to grow."""
    from nyxara.growth.genesis import ArchitectureGenome, LayerGene
    from nyxara.growth.topology import DynamicTopology
    from nyxara.kernel.config import TopologyConfig

    reg = EnvironmentRegistry(path=tmp_path / "env.json", min_checks=3)
    topo = DynamicTopology(cfg=TopologyConfig(grow_dims_k=8, max_n_embd=128, max_layers=8))
    genome = ArchitectureGenome(n_embd=64, block_size=16,
                                layers=[LayerGene(op="attention", n_head=4),
                                        LayerGene(op="gated_mlp")])
    adapter = EnvironmentAdapter(registry=reg, topology=topo, growth_source=lambda: genome, seed=7)

    rng = random.Random(1)
    hostile = {f"noise-{i}": (lambda x, r=random.Random(i): r.random()) for i in range(4)}
    rep = adapter.adapt(hostile, label="hostile")
    assert rep.pressure >= 0.5           # everything defeated her → real pressure
    assert rep.grew is True              # so she re-organized her own brain
    assert rep.growth and rep.growth.get("grew") is True
    assert rep.ceiling == {"max_n_embd": 128, "max_layers": 8}


def test_no_llm_is_ever_called(tmp_path, monkeypatch):
    """The core honesty guarantee: adaptation is NYXARA's OWN work — the LLM must never be touched.

    Every real call into a language model funnels through ``LLM.complete``/``LLM.chat``; we replace
    both with tripwires that fail the test if adaptation ever reaches them. It doesn't: the whole
    inquiry is a deterministic symbolic law search."""
    from nyxara.mind.llm import LLM

    def _tripwire(*a, **k):
        raise AssertionError("adaptation must not call the LLM")

    monkeypatch.setattr(LLM, "complete", _tripwire, raising=False)
    monkeypatch.setattr(LLM, "chat", _tripwire, raising=False)

    reg = EnvironmentRegistry(path=tmp_path / "env.json", min_checks=3)
    adapter = EnvironmentAdapter(registry=reg, seed=7)
    rep = adapter.adapt(_env(), label="env")
    assert rep.modelled >= 3             # she still models the environment — entirely on her own


def test_adaptation_module_does_not_import_the_llm():
    """Static guarantee: the faculty's own source never even imports the language-model module."""
    import inspect
    import nyxara.growth.adaptation as adaptation
    import nyxara.growth.env_registry as env_registry
    import nyxara.growth.open_world as open_world

    for mod in (adaptation, env_registry, open_world):
        src = inspect.getsource(mod)
        assert "mind.llm" not in src and "import llm" not in src


def test_empty_environment_is_a_clean_noop(tmp_path):
    reg = EnvironmentRegistry(path=tmp_path / "env.json", min_checks=3)
    adapter = EnvironmentAdapter(registry=reg, seed=7)
    rep = adapter.adapt({}, label="empty")
    assert rep.n_systems == 0 and rep.modelled == 0 and rep.grew is False
