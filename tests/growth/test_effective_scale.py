"""Tests for nyxara.growth.effective_scale — the honest effective-scale meter (Problem #1, Scale).

Other AIs stand on a billion-parameter trained model; NYXARA runs a small one. This module is her
own, truthful answer: spend her compute on the small model and *measure* what that buys. These tests
pin the honest guarantees — no regression on a weak box, strict scaling up with compute, a bounded
amplification, a real (unfaked) nominal param count from the foundry manifest, and reachability from
a live NyxaraCore (what makes capability #62 genuinely WIRED, not merely REAL).
"""

from __future__ import annotations

import json

from nyxara.growth.effective_scale import (
    _AMP_CEILING,
    _RUNG_CEILING,
    _SAMPLES_CEILING,
    _SECONDS_CEILING,
    EffectiveScale,
    estimate_effective_scale,
    scaled_budget,
)
from nyxara.kernel.compute import ComputeReport, GPUInfo
from nyxara.kernel.config import DeepReasoningConfig, NyxaraSettings, Profile
from nyxara.kernel.orchestrator import NyxaraCore

_BARE = ComputeReport(cpu_count=2, memory={"available_mb": 2000, "total_mb": 4000},
                      gpu=GPUInfo(available=False), torch_available=False)
_BIG = ComputeReport(cpu_count=32, memory={"available_mb": 64000, "total_mb": 128000},
                     gpu=GPUInfo(available=True, backend="cuda", count=1, names=["A100"]),
                     torch_available=True)


# --------------------------------------------------------------------------- #
# scaled_budget: no regression, scales up, bounded, monotonic
# --------------------------------------------------------------------------- #
def test_capacity_zero_equals_configured_floor():
    cfg = DeepReasoningConfig()
    b = scaled_budget(_BARE, cfg, capacity=0.0)
    assert b.max_rung == cfg.max_rung
    assert b.samples == cfg.samples
    assert b.max_seconds == cfg.max_seconds   # capacity 0 → the exact prior behaviour


def test_budget_never_below_floor_on_any_box():
    cfg = DeepReasoningConfig()
    for compute in (_BARE, _BIG):
        b = scaled_budget(compute, cfg)
        assert b.max_rung >= cfg.max_rung
        assert b.samples >= cfg.samples
        assert b.max_seconds >= cfg.max_seconds


def test_strong_box_buys_more_than_weak_box():
    cfg = DeepReasoningConfig()
    weak, strong = scaled_budget(_BARE, cfg), scaled_budget(_BIG, cfg)
    assert strong.samples > weak.samples
    assert strong.max_seconds >= weak.max_seconds
    assert strong.samples > cfg.samples     # a strong box genuinely thinks harder than configured


def test_budget_stays_within_config_bounds():
    cfg = DeepReasoningConfig()
    for c in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        b = scaled_budget(_BIG, cfg, capacity=c)
        assert 1 <= b.max_rung <= _RUNG_CEILING
        assert 1 <= b.samples <= _SAMPLES_CEILING
        assert 1.0 <= b.max_seconds <= _SECONDS_CEILING


def test_samples_monotonic_in_capacity():
    cfg = DeepReasoningConfig()
    prev = 0
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = scaled_budget(_BARE, cfg, capacity=c).samples
        assert s >= prev
        prev = s


def test_scaled_budget_never_raises_on_garbage():
    b = scaled_budget(object(), object())
    assert b.samples >= 1 and b.max_rung >= 1


# --------------------------------------------------------------------------- #
# estimate_effective_scale: honest floor, bounded amplification, real caveat
# --------------------------------------------------------------------------- #
def test_effective_scale_bounded_and_honest_floor(tmp_path):
    # Isolate the foundry data dir so the "no promoted model" guarantee is tested hermetically,
    # regardless of any model promoted into the ambient default data dir by this environment.
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.paths.data_dir = tmp_path
    es = estimate_effective_scale(_BIG, reasoner=None, settings=settings)
    assert isinstance(es, EffectiveScale)
    assert 1.0 <= es.amplification <= _AMP_CEILING
    # a bare checkout has no promoted model → nominal unknown, effective 0 (never faked)
    assert es.nominal_known is False
    assert es.effective_params == 0
    assert "NOT a literal parameter count" in es.caveat


def test_amplification_never_exceeds_ceiling():
    # even with every amplifier reported available (reasoner=None) the stack is capped
    es = estimate_effective_scale(_BIG, reasoner=None)
    assert es.amplification <= _AMP_CEILING
    assert es.factors.get("test_time_compute", 0.0) > 1.0   # deep reasoning is on by default


def test_effective_scale_never_raises_on_garbage():
    es = estimate_effective_scale(object(), reasoner=object())
    assert es.amplification >= 1.0


def test_nominal_params_read_from_foundry_manifest(tmp_path):
    """A promoted model's real param_count is read cheaply from the manifest, never faked."""
    root = tmp_path / "foundry"
    root.mkdir()
    (root / "active").write_text("v2", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({
        "versions": [
            {"version": 1, "param_count": 100_000},
            {"version": 2, "param_count": 7_000_000},   # the promoted one
        ]
    }), encoding="utf-8")
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.self_model_dir = root
    es = estimate_effective_scale(_BIG, reasoner=None, settings=s)
    assert es.nominal_known is True
    assert es.nominal_params == 7_000_000            # the ACTIVE version's count, not v1
    assert es.effective_params == int(7_000_000 * es.amplification)


def test_no_promoted_model_reports_unknown(tmp_path):
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.self_model_dir = tmp_path / "empty_foundry"   # nothing on disk
    es = estimate_effective_scale(_BARE, reasoner=None, settings=s)
    assert es.nominal_known is False and es.effective_params == 0


# --------------------------------------------------------------------------- #
# WIRED: reachable from a live NyxaraCore
# --------------------------------------------------------------------------- #
def test_scale_report_reachable_from_core():
    core = NyxaraCore(enable_memory=False, enable_tools=False, enable_skills=False)
    rep = core.scale_report()
    assert "error" not in rep, rep
    assert rep["amplification"] >= 1.0
    assert "caveat" in rep and "NOT a literal parameter count" in rep["caveat"]
    for key in ("nominal_params", "nominal_known", "effective_params", "factors", "capacity"):
        assert key in rep
