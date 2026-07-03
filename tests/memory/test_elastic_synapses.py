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


# -------------------- Synaptic Intelligence (path-integral importance) -------------------- #
def test_path_integral_ranks_working_weight_above_idle():
    from nyxara.memory.elastic_synapses import PathIntegralEstimator

    est = PathIntegralEstimator()
    est.begin({"busy": 0.0, "idle": 0.0})
    w = 0.0
    for _ in range(5):
        # "busy" moves downhill against its gradient (does real work); "idle" never moves
        est.observe_step({"busy": -1.0, "idle": 0.5},
                         {"busy": w, "idle": 0.0}, {"busy": w + 0.2, "idle": 0.0})
        w += 0.2
    omega = est.omega({"busy": w, "idle": 0.0}, normalize=True)
    assert omega.get("busy", 0.0) > omega.get("idle", 0.0)


def test_accumulate_step_feeds_consolidated_importance():
    syn = ElasticSynapses(si_enabled=True)
    old, new = {"w": 0.0}, {"w": 0.3}
    for _ in range(4):
        syn.accumulate_step({"w": -1.0}, old, new)
        old, new = new, {"w": new["w"] + 0.3}
    task = syn.consolidate(old, task="si")
    assert task.fisher.get("w", 0.0) > 0.0


def test_accumulate_step_refuses_protected_core():
    syn = ElasticSynapses()
    with pytest.raises(ValidationError):
        syn.accumulate_step({"loyalty_to_master": 1.0}, {"loyalty_to_master": 1.0},
                            {"loyalty_to_master": 0.5})


def test_si_disabled_matches_mas_only():
    on = ElasticSynapses(si_enabled=True)
    off = ElasticSynapses(si_enabled=False)
    for syn in (on, off):
        syn.observe_features({"w": 1.0})
        syn.consolidate({"w": 0.5}, task="A")
    # with no accumulate_step feed the two consolidate to identical importance
    assert on._tasks[-1].fisher["w"] == off._tasks[-1].fisher["w"]


# -------------------- lossless eviction (long-term merged anchor) -------------------- #
def test_eviction_merges_into_longterm_never_drops():
    syn = ElasticSynapses(online=False, max_tasks=2)
    for i in range(4):
        syn.observe_features({f"w{i}": 1.0})
        syn.consolidate({f"w{i}": float(i + 1)}, task=f"skill-{i}")
    assert len(syn._tasks) == 2                     # budget respected
    assert syn._longterm is not None                # evicted anchors merged, not dropped
    assert syn.importance("w0") > 0.0               # the oldest skill is still protected
    assert syn.penalty({"w0": 99.0}, names=["w0"]) > 0.0


def test_merge_weighted_math_fisher_weighted_mean():
    syn = ElasticSynapses()
    a = ConsolidatedTask(name="a", theta_star={"w": 0.0}, fisher={"w": 1.0})
    b = ConsolidatedTask(name="b", theta_star={"w": 1.0}, fisher={"w": 3.0})
    merged = syn._merge_weighted(a, b, name="m")
    assert merged.fisher["w"] == pytest.approx(4.0)          # Fishers add
    assert merged.theta_star["w"] == pytest.approx(0.75)     # Fisher-weighted anchor mean


def test_protected_inf_survives_merge():
    syn = ElasticSynapses(online=False, max_tasks=1)
    syn.observe_features({"w": 1.0})
    syn.consolidate({"w": 1.0}, task="A")            # includes protected core at inf
    syn.observe_features({"v": 1.0})
    syn.consolidate({"v": 2.0}, task="B")            # A evicted -> merged into longterm
    lt = syn._longterm
    assert lt is not None
    assert lt.fisher["loyalty_to_master"] == float("inf")
    assert syn.is_frozen("loyalty_to_master")
    # the protected anchor stays pinned to its canonical value through the merge
    pinned = syn.apply_anchor({"loyalty_to_master": 0.1}, lr=0.5)
    assert pinned["loyalty_to_master"] == pytest.approx(1.0)


# -------------------- per-skill anchors -------------------- #
def test_per_skill_anchors_keep_distinct_named_anchors():
    syn = ElasticSynapses(per_skill_anchors=True, max_tasks=8)
    syn.observe_features({"a": 1.0}); syn.consolidate({"a": 1.0}, task="alpha")
    syn.observe_features({"b": 1.0}); syn.consolidate({"b": 2.0}, task="beta")
    syn.observe_features({"a": 1.0}); syn.consolidate({"a": 1.2}, task="alpha")
    assert sorted(t.name for t in syn._tasks) == ["alpha", "beta"]


def test_per_skill_anchors_overflow_is_lossless():
    syn = ElasticSynapses(per_skill_anchors=True, max_tasks=2)
    for name in ("a", "b", "c"):
        syn.observe_features({name: 1.0})
        syn.consolidate({name: 1.0}, task=name)
    assert len(syn._tasks) == 2
    assert syn._longterm is not None
    assert syn.importance("a") > 0.0


def test_per_skill_default_off_keeps_single_online_anchor():
    syn = ElasticSynapses()          # per_skill_anchors defaults to False
    syn.observe_features({"a": 1.0}); syn.consolidate({"a": 1.0}, task="alpha")
    syn.observe_features({"b": 1.0}); syn.consolidate({"b": 2.0}, task="beta")
    assert len(syn._tasks) == 1      # classic online EWC: one running anchor


# -------------------- names= filter and the stable pull -------------------- #
def test_penalty_grad_names_filter_matches_unfiltered():
    syn = ElasticSynapses()
    syn.observe_features({"a": 1.0, "b": 1.0})
    syn.consolidate({"a": 1.0, "b": 2.0}, task="A")
    params = {"a": 3.0, "b": 5.0}
    full = syn.penalty_grad(params)
    only_a = syn.penalty_grad(params, names=["a"])
    assert set(only_a) == {"a"}
    assert only_a["a"] == pytest.approx(full["a"])


def test_penalty_pull_never_overshoots_anchor():
    # stack MANY stiff anchors: the raw gradient step would diverge, the pull may not
    syn = ElasticSynapses(ewc_lambda=50.0, online=False, max_tasks=16)
    for i in range(10):
        syn.observe_features({"w": 1.0})
        syn.consolidate({"w": 1.0}, task=f"t{i}")
    w = 5.0
    for _ in range(50):
        delta = syn.penalty_pull({"w": w}, 0.5, names=["w"])
        w += delta.get("w", 0.0)
        assert abs(w) < 10.0          # bounded — no oscillating divergence
    assert w == pytest.approx(1.0, abs=1e-3)   # converged to the anchor


# -------------------- drift metric -------------------- #
def test_anchor_drift_zero_at_anchor_and_grows_with_deviation():
    syn = ElasticSynapses()
    syn.observe_features({"w": 1.0})
    syn.consolidate({"w": 1.0}, task="A")
    at_anchor = syn.anchor_drift({"w": 1.0})["mean_drift"]
    away = syn.anchor_drift({"w": 3.0})["mean_drift"]
    assert at_anchor == pytest.approx(0.0)
    assert away > at_anchor


# -------------------- persistence (new fields + backward compat) -------------------- #
def test_old_format_blob_loads_cleanly():
    # a blob written before the longterm / per-skill / SI upgrade has none of the new keys
    old_blob = {
        "ewc_lambda": 2.0, "freeze_threshold": 0.8, "max_tasks": 4, "online": True,
        "gamma": 0.9, "protected": ["loyalty_to_master"], "consolidations": 1,
        "theta": {"w": 0.5},
        "tasks": [{"name": "A", "theta_star": {"w": 0.5},
                   "fisher": {"w": 1.0, "loyalty_to_master": "inf"}, "at": 0.0}],
    }
    syn = ElasticSynapses.from_dict(old_blob)
    assert syn._longterm is None
    assert syn.is_frozen("loyalty_to_master")
    assert syn.importance("w") == pytest.approx(1.0)


def test_round_trip_preserves_longterm():
    syn = ElasticSynapses(online=False, max_tasks=1)
    syn.observe_features({"a": 1.0}); syn.consolidate({"a": 1.0}, task="A")
    syn.observe_features({"b": 1.0}); syn.consolidate({"b": 2.0}, task="B")
    blob = syn.to_dict()
    restored = ElasticSynapses.from_dict(blob)
    assert restored.to_dict() == blob
    assert restored._longterm is not None
    assert restored.importance("a") > 0.0


# -------------------- torch adapter parity (optional) -------------------- #
def test_torch_multitask_eviction_merges_lossless():
    torch = pytest.importorskip("torch")
    from nyxara.memory.elastic_synapses import TorchElasticSynapses

    net = torch.nn.Linear(2, 1)
    syn = TorchElasticSynapses(net, online=False, max_tasks=2)
    for i in range(4):
        syn.consolidate(task=f"t{i}")
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.5)
    assert len(syn._tasks) == 2
    assert syn._longterm is not None
    assert float(syn.penalty().item()) > 0.0
    state = syn.state_dict()
    syn2 = TorchElasticSynapses(net, online=False)
    syn2.load_state_dict(state)
    assert len(syn2._tasks) == 2
    assert syn2._longterm is not None
