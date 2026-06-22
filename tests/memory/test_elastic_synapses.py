"""Tests for nyxara.memory.elastic_synapses — Elastic Weight Consolidation (lifelong memory)."""

from __future__ import annotations

import pytest

from nyxara.kernel.errors import ValidationError
from nyxara.memory.elastic_synapses import (
    ConsolidatedTask,
    ElasticSynapses,
    FisherEstimator,
)

ParamVec = dict


# -------------------- Fisher importance -------------------- #
def test_fisher_grows_with_use():
    est = FisherEstimator()
    for _ in range(10):
        est.observe({"hot": 1.0, "cold": 0.1})
    f = est.fisher(normalize=True)
    assert f["hot"] > f["cold"]


def test_fisher_normalized_peak_is_one():
    est = FisherEstimator()
    est.observe({"a": 2.0, "b": 1.0})
    f = est.fisher(normalize=True)
    assert f["a"] == pytest.approx(1.0)
    assert f["b"] < 1.0


def test_fisher_reset_clears():
    est = FisherEstimator()
    est.observe({"a": 1.0})
    est.reset()
    assert len(est) == 0
    assert est.fisher() == {}


# -------------------- consolidation -------------------- #
def test_consolidate_records_anchor_and_fisher():
    syn = ElasticSynapses()
    syn.register({"w": 0.7})
    syn.observe_features({"w": 1.0})
    task = syn.consolidate(task="A")
    assert isinstance(task, ConsolidatedTask)
    assert task.theta_star["w"] == pytest.approx(0.7)
    assert task.fisher["w"] > 0.0


def test_consolidate_returns_named_task():
    syn = ElasticSynapses()
    syn.register({"w": 1.0})
    syn.observe_features({"w": 1.0})
    assert syn.consolidate().name == "task-0"


def test_multitask_keeps_distinct_anchors_up_to_budget():
    syn = ElasticSynapses(online=False, max_tasks=2)
    for i in range(3):
        syn.register({f"w{i}": 1.0})
        syn.observe_features({f"w{i}": 1.0})
        syn.consolidate(task=f"t{i}")
    # oldest evicted: at most max_tasks anchors remain
    assert syn.stats()["tasks"] == 2


def test_online_mode_keeps_single_running_anchor():
    syn = ElasticSynapses(online=True)
    for i in range(4):
        syn.register({"w": float(i)})
        syn.observe_features({"w": 1.0})
        syn.consolidate()
    assert syn.stats()["tasks"] == 1


# -------------------- catastrophic forgetting -------------------- #
def _train_linear(weights, feats, target, steps, *, lr=0.1, syn=None):
    """Plain gradient descent on squared error, optionally with an EWC pull-back."""
    for _ in range(steps):
        pred = sum(weights.get(f, 0.0) * x for f, x in feats.items())
        err = target - pred
        for f, x in feats.items():
            weights[f] = weights.get(f, 0.0) + lr * err * x
        if syn is not None:
            syn.observe_features(feats)
            for f, g in syn.penalty_grad(weights).items():
                weights[f] = weights.get(f, 0.0) - lr * g


def _predict(weights, feats):
    return sum(weights.get(f, 0.0) * x for f, x in feats.items())


def test_ewc_reduces_forgetting_vs_plain():
    task_a = {"shared": 1.0, "ctx_a": 1.0}
    task_b = {"shared": 1.0, "ctx_b": 1.0}

    plain: ParamVec = {}
    _train_linear(plain, task_a, 1.0, 200)
    _train_linear(plain, task_b, -1.0, 300)
    plain_error = abs(_predict(plain, task_a) - 1.0)

    syn = ElasticSynapses(ewc_lambda=8.0)
    ewc: ParamVec = {}
    _train_linear(ewc, task_a, 1.0, 200, syn=syn)
    syn.consolidate(ewc, task="A")
    _train_linear(ewc, task_b, -1.0, 300, syn=syn)
    ewc_error = abs(_predict(ewc, task_a) - 1.0)

    assert ewc_error < plain_error


# -------------------- protected loyalty core -------------------- #
def test_accumulate_over_protected_core_refused():
    syn = ElasticSynapses()
    with pytest.raises(ValidationError):
        syn.accumulate({"loyalty_to_master": 5.0})


def test_observe_features_over_protected_refused():
    syn = ElasticSynapses()
    with pytest.raises(ValidationError):
        syn.observe_features({"obedience": 1.0})


def test_protected_core_always_frozen():
    syn = ElasticSynapses()
    for name in ("loyalty_to_master", "obedience", "corrigibility", "owner_safety", "honesty"):
        assert syn.is_frozen(name)
        assert syn.importance(name) == float("inf")


def test_protected_weight_is_pinned_to_anchor():
    syn = ElasticSynapses()
    syn.consolidate({"loyalty_to_master": 1.0}, task="core")
    pulled = syn.apply_anchor({"loyalty_to_master": 0.2}, lr=0.5)
    assert pulled["loyalty_to_master"] == pytest.approx(1.0)


# -------------------- freezing -------------------- #
def test_freeze_mask_marks_consolidated_weight():
    syn = ElasticSynapses(freeze_threshold=0.5)
    syn.register({"w": 1.0})
    syn.observe_features({"w": 1.0})
    syn.consolidate()
    mask = syn.freeze_mask()
    assert mask["w"] is True


# -------------------- penalty / gradient -------------------- #
def test_penalty_zero_at_anchor():
    syn = ElasticSynapses(ewc_lambda=5.0)
    syn.register({"w": 1.0})
    syn.observe_features({"w": 1.0})
    syn.consolidate()
    assert syn.penalty({"w": 1.0}) == pytest.approx(0.0)


def test_penalty_grows_with_deviation():
    syn = ElasticSynapses(ewc_lambda=5.0)
    syn.register({"w": 1.0})
    syn.observe_features({"w": 1.0})
    syn.consolidate()
    assert syn.penalty({"w": 2.0}) > syn.penalty({"w": 1.5}) > 0.0


def test_penalty_grad_points_back_to_anchor():
    syn = ElasticSynapses(ewc_lambda=5.0)
    syn.register({"w": 1.0})
    syn.observe_features({"w": 1.0})
    syn.consolidate()
    # weight above anchor → positive gradient (subtracting it moves back down toward anchor)
    assert syn.penalty_grad({"w": 2.0})["w"] > 0.0
    assert syn.penalty_grad({"w": 0.0})["w"] < 0.0


# -------------------- persistence (lifelong) -------------------- #
def test_to_from_dict_round_trip_identical():
    syn = ElasticSynapses(ewc_lambda=4.0)
    syn.register({"w": 0.5, "v": -0.3})
    syn.observe_features({"w": 1.0, "v": 0.5})
    syn.consolidate(task="A")
    syn.consolidate({"loyalty_to_master": 1.0}, task="core")
    blob = syn.to_dict()
    restored = ElasticSynapses.from_dict(blob)
    assert restored.to_dict() == blob


def test_restored_engine_preserves_protected_freeze():
    syn = ElasticSynapses()
    syn.consolidate({"loyalty_to_master": 1.0}, task="core")
    restored = ElasticSynapses.from_dict(syn.to_dict())
    assert restored.is_frozen("loyalty_to_master")


def test_restored_engine_preserves_anchor_protection():
    task_a = {"shared": 1.0, "ctx_a": 1.0}
    task_b = {"shared": 1.0, "ctx_b": 1.0}
    syn = ElasticSynapses(ewc_lambda=8.0)
    ewc: ParamVec = {}
    _train_linear(ewc, task_a, 1.0, 200, syn=syn)
    syn.consolidate(ewc, task="A")
    # round-trip the engine, then keep learning task B through the restored engine
    syn2 = ElasticSynapses.from_dict(syn.to_dict())
    _train_linear(ewc, task_b, -1.0, 300, syn=syn2)
    assert abs(_predict(ewc, task_a) - 1.0) < 0.5


# -------------------- stats -------------------- #
def test_stats_reports_frozen_count():
    syn = ElasticSynapses(freeze_threshold=0.5)
    syn.register({"w": 1.0})
    syn.observe_features({"w": 1.0})
    syn.consolidate()
    stats = syn.stats()
    assert stats["consolidations"] == 1
    assert stats["weights_frozen"] >= 1


# -------------------- torch adapter (optional) -------------------- #
def test_torch_adapter_penalty_and_consolidate():
    torch = pytest.importorskip("torch")
    from nyxara.memory.elastic_synapses import TorchElasticSynapses

    net = torch.nn.Linear(3, 2)
    syn = TorchElasticSynapses(net, ewc_lambda=10.0)
    # before any consolidation the penalty is a real zero tensor
    assert float(syn.penalty().item()) == pytest.approx(0.0)
    syn.consolidate(task="A")
    # move the weights; penalty must become positive
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    assert float(syn.penalty().item()) > 0.0


def test_torch_adapter_state_round_trip():
    torch = pytest.importorskip("torch")
    from nyxara.memory.elastic_synapses import TorchElasticSynapses

    net = torch.nn.Linear(2, 2)
    syn = TorchElasticSynapses(net)
    syn.consolidate(task="A")
    state = syn.state_dict()
    syn2 = TorchElasticSynapses(net)
    syn2.load_state_dict(state)
    assert syn2.state_dict()["consolidations"] == state["consolidations"]
