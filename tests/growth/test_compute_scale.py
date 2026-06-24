"""Tests for nyxara.growth.compute_scale — honest compute-aware autoscaling."""

from __future__ import annotations

from nyxara.growth.compute_scale import (CloudComputeProvider, LocalComputeProvider,
                                        recommend_foundry_profile)
from nyxara.kernel.compute import ComputeReport, GPUInfo


def _bare() -> ComputeReport:
    return ComputeReport(cpu_count=2, memory={"available_mb": 2000, "total_mb": 4000},
                         gpu=GPUInfo(available=False), torch_available=False)


def _cpu_torch() -> ComputeReport:
    return ComputeReport(cpu_count=16, memory={"available_mb": 32000, "total_mb": 64000},
                         gpu=GPUInfo(available=False), torch_available=True)


def _big_gpu() -> ComputeReport:
    return ComputeReport(cpu_count=32, memory={"available_mb": 64000, "total_mb": 128000},
                         gpu=GPUInfo(available=True, backend="cuda", count=1, names=["A100"]),
                         torch_available=True)


# -------------------- recommendation table -------------------- #
def test_bare_cpu_keeps_honest_floor():
    rec = recommend_foundry_profile(_bare(), has_torch=False, has_cuda=False)
    assert rec.profile == "custom"
    assert rec.load_in_4bit is False


def test_torch_cpu_gets_real_but_small_net_never_gpt2():
    rec = recommend_foundry_profile(_cpu_torch(), has_torch=True, has_cuda=False)
    assert rec.profile in ("tiny", "small")
    assert rec.profile not in ("gpt2", "gpt2-medium")
    assert rec.load_in_4bit is False


def test_cuda_gpu_unlocks_gpt2_scale_and_qlora():
    rec = recommend_foundry_profile(_big_gpu(), has_torch=True, has_cuda=True)
    assert rec.profile in ("gpt2", "gpt2-medium")
    assert rec.load_in_4bit is True
    assert rec.device == "cuda"
    assert rec.capacity >= 0.55


def test_reality_clamp_no_gpt2_or_4bit_without_cuda():
    # even a powerful report must NOT yield a GPU-only profile when CUDA is absent
    rec = recommend_foundry_profile(_big_gpu(), has_torch=True, has_cuda=False)
    assert rec.profile not in ("gpt2", "gpt2-medium")
    assert rec.load_in_4bit is False


def test_recommendation_never_raises_on_garbage():
    rec = recommend_foundry_profile(object(), has_torch=False, has_cuda=False)
    assert rec.profile == "custom"


# -------------------- providers (the acquisition seam) -------------------- #
def test_local_provider_returns_live_report():
    p = LocalComputeProvider().provision()
    assert p.ok is True
    assert p.report is not None
    assert p.provider == "local"


def test_cloud_provider_declines_without_credentials():
    p = CloudComputeProvider("runpod", env={}).provision()
    assert p.ok is False
    assert "no credentials" in p.reason


def test_cloud_provider_with_credentials_is_honest_no_fake_rental():
    # credentials present, but no live adapter — it must decline honestly, not fake a rental
    p = CloudComputeProvider("runpod", api_key="sk-test").provision()
    assert p.ok is False
    assert p.meta.get("has_credentials") is True


def test_cloud_provider_reads_env_key():
    p = CloudComputeProvider("runpod", env={"RUNPOD_API_KEY": "sk-env"}).provision()
    # key found in env -> passes the credential check (still declines: no live adapter)
    assert p.meta.get("has_credentials") is True
