"""NYXARA · growth/compute_scale.py — scale the forged model to the compute she has (🖥, Rule 4).

The Master's complaint: NYXARA *claimed* compute acquisition but the foundry always built the
same tiny model regardless of the hardware under it. This module makes "compute acquisition"
**honest and real**: it reads what compute is actually available
(:func:`~nyxara.kernel.compute.compute_report`) and recommends the *largest model the box can
truly run* — never more. A bare CPU keeps the always-on coherent backend; a single strong CUDA
GPU unlocks GPT-2 scale and QLoRA over a real 7B+ base.

Two honest pieces:

* :func:`recommend_foundry_profile` — maps a :class:`~nyxara.kernel.compute.ComputeReport` to a
  ``(profile, load_in_4bit)`` recommendation, reusing
  :meth:`~nyxara.growth.intelligence.IntelligenceEngine.compute_capacity` for the ``[0, 1]``
  capacity score. It is **clamped to reality**: GPT-2 / GPT-2-medium and 4-bit QLoRA are only
  ever recommended when torch *and* a CUDA GPU are genuinely present, so the foundry never
  promises a model the machine cannot train.
* :class:`ComputeProvider` (protocol) + :class:`LocalComputeProvider` /
  :class:`CloudComputeProvider` — the seam for acquiring compute. The local provider returns the
  live report. The cloud provider implements the *same interface* but, without credentials,
  **honestly declines** (``ok=False, reason="no credentials"``) rather than pretending to rent a
  GPU. Live renting is intentionally out of scope here; the interface exists so it can be added
  without touching the foundry.

Pure, dependency-light, never raises — works on the barest machine and just recommends the
honest floor there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

__all__ = [
    "ProfileRecommendation",
    "recommend_foundry_profile",
    "Provisioned",
    "ComputeProvider",
    "LocalComputeProvider",
    "CloudComputeProvider",
]


def _torch_cuda() -> "tuple[bool, bool]":
    """(_HAS_TORCH, cuda_available) — import-guarded, never raises."""
    try:
        from nyxara.growth.foundry_models import _HAS_TORCH
    except Exception:  # noqa: BLE001
        _HAS_TORCH = False
    cuda = False
    if _HAS_TORCH:
        try:
            import torch  # type: ignore
            cuda = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 — a broken driver must not crash a recommendation
            cuda = False
    return bool(_HAS_TORCH), cuda


def _capacity(compute: Any) -> float:
    """Reuse the intelligence engine's compute→[0,1] mapping; fall back to a local proxy."""
    try:
        from nyxara.growth.intelligence import IntelligenceEngine
        return float(IntelligenceEngine.compute_capacity(compute))
    except Exception:  # noqa: BLE001 — never let scoring break a recommendation
        try:
            mem = getattr(compute, "memory", {}) or {}
            avail = mem.get("available_mb") or mem.get("total_mb") or 0
            gpu_ok = bool(getattr(getattr(compute, "gpu", None), "available", False))
            return min(1.0, 0.5 * min(1.0, float(avail) / 16384.0) + (0.5 if gpu_ok else 0.0))
        except Exception:  # noqa: BLE001
            return 0.0


# --------------------------------------------------------------------------- #
# Profile recommendation
# --------------------------------------------------------------------------- #
@dataclass
class ProfileRecommendation:
    """What the foundry should build for the compute it actually has."""

    profile: str               # "custom" | "tiny" | "small" | "gpt2" | "gpt2-medium"
    load_in_4bit: bool         # QLoRA — only ever True with a real CUDA GPU
    capacity: float            # the [0, 1] compute capacity that drove the choice
    device: str                # "cuda" | "mps" | "cpu"
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"profile": self.profile, "load_in_4bit": self.load_in_4bit,
                "capacity": round(self.capacity, 4), "device": self.device,
                "reason": self.reason}


def recommend_foundry_profile(compute: Any, *,
                              has_torch: Optional[bool] = None,
                              has_cuda: Optional[bool] = None) -> ProfileRecommendation:
    """Recommend the largest foundry profile this compute can honestly train.

    Clamped to reality:

    * **No torch** → ``custom`` (the explicit tiny dims; the factory then uses the always-on
      coherent Kneser-Ney backend). The honest floor on a bare box.
    * **Torch, CPU only** → ``tiny`` (a real from-scratch nano-GPT, but small — GPT-2 scale is
      infeasible on CPU, so it is never recommended there).
    * **CUDA GPU** → scales with capacity: ``small`` → ``gpt2`` → ``gpt2-medium``; the two GPT-2
      tiers also enable 4-bit QLoRA so a real large base fits on one GPU.

    Deps are injectable (``has_torch`` / ``has_cuda``) so the full decision table is unit-testable
    on a GPU-less CI box. Never raises.
    """
    if has_torch is None or has_cuda is None:
        ht, hc = _torch_cuda()
        has_torch = ht if has_torch is None else has_torch
        has_cuda = hc if has_cuda is None else has_cuda
    cap = _capacity(compute)
    device = "cuda" if has_cuda else (
        getattr(compute, "recommend_device", lambda: "cpu")() if hasattr(compute, "recommend_device")
        else "cpu")

    if not has_torch:
        return ProfileRecommendation("custom", False, cap, "cpu",
                                     "no torch — always-on coherent backend (honest floor)")
    if not has_cuda:
        # real neural training, but kept to a CPU-feasible size
        prof = "small" if cap >= 0.6 else "tiny"
        return ProfileRecommendation(prof, False, cap, device,
                                     f"torch on CPU — {prof} nano-GPT (GPT-2 needs a GPU)")
    # CUDA present: scale to capacity, unlock QLoRA for the large tiers
    if cap >= 0.95:
        # Her own from-scratch 300M decoder — recommended only on a GPU that can genuinely
        # train it. Never reachable without CUDA: `growth/trainer.preflight` puts a CPU run of
        # this profile at roughly a YEAR, and autoscaling into it would be a quiet way to start
        # exactly the run that check exists to prevent.
        return ProfileRecommendation("nyxara-300m", False, cap, "cuda",
                                     "strong CUDA GPU — NYX-300M, her own from-scratch brain")
    if cap >= 0.85:
        return ProfileRecommendation("gpt2-medium", True, cap, "cuda",
                                     "strong CUDA GPU — GPT-2-medium + 4-bit QLoRA")
    if cap >= 0.55:
        return ProfileRecommendation("gpt2", True, cap, "cuda",
                                     "CUDA GPU — GPT-2 scale + 4-bit QLoRA")
    return ProfileRecommendation("small", False, cap, "cuda",
                                 "modest CUDA GPU — small transformer")


# --------------------------------------------------------------------------- #
# Compute provider (the acquisition seam)
# --------------------------------------------------------------------------- #
@dataclass
class Provisioned:
    """The outcome of asking a provider for compute — honest about success/failure."""

    ok: bool
    provider: str
    report: Any = None                 # a ComputeReport when ok, else None
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        rep = None
        if self.report is not None:
            try:
                rep = self.report.to_dict()
            except Exception:  # noqa: BLE001
                rep = str(self.report)
        return {"ok": self.ok, "provider": self.provider, "report": rep,
                "reason": self.reason, "meta": dict(self.meta)}


class ComputeProvider:
    """Protocol: something that can provision compute and report what it provisioned."""

    name: str = "base"

    def provision(self) -> Provisioned:  # pragma: no cover - interface
        raise NotImplementedError


class LocalComputeProvider(ComputeProvider):
    """The compute already under NYXARA's feet — always available, always honest."""

    name = "local"

    def provision(self) -> Provisioned:
        try:
            from nyxara.kernel.compute import compute_report
            return Provisioned(ok=True, provider=self.name, report=compute_report())
        except Exception as exc:  # noqa: BLE001 — even introspection must not raise
            return Provisioned(ok=False, provider=self.name, reason=f"{type(exc).__name__}: {exc}")


class CloudComputeProvider(ComputeProvider):
    """A real, pluggable seam for renting a GPU — but it never *pretends*.

    Without configured credentials (an explicit key or the matching ``*_API_KEY`` env var) it
    declines honestly (``ok=False, reason="no credentials"``). Actually issuing a rental request
    is intentionally left unimplemented here so the foundry can adopt the interface today and a
    live adapter can be dropped in later without any change upstream.
    """

    def __init__(self, provider: str = "runpod", *, api_key: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None) -> None:
        self.name = provider
        env = env if env is not None else dict(os.environ)
        self.api_key = api_key or env.get(f"{provider.upper()}_API_KEY") or None

    def provision(self) -> Provisioned:
        if not self.api_key:
            return Provisioned(ok=False, provider=self.name,
                               reason="no credentials — cloud GPU not provisioned (honest decline)")
        # Credentials present but no live adapter wired yet: decline rather than fake a rental.
        return Provisioned(ok=False, provider=self.name,
                           reason="live cloud provisioning not enabled in this build",
                           meta={"has_credentials": True})


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.compute import ComputeReport, GPUInfo

    print("=" * 70)
    print("NYXARA compute-scale self-test")
    print("=" * 70)

    bare = ComputeReport(cpu_count=2, memory={"available_mb": 2000, "total_mb": 4000},
                         gpu=GPUInfo(available=False), torch_available=False)
    cpu_torch = ComputeReport(cpu_count=16, memory={"available_mb": 32000, "total_mb": 64000},
                              gpu=GPUInfo(available=False), torch_available=True)
    big_gpu = ComputeReport(cpu_count=32, memory={"available_mb": 64000, "total_mb": 128000},
                            gpu=GPUInfo(available=True, backend="cuda", count=1,
                                        names=["A100"]), torch_available=True)

    # bare box → honest floor, never GPT-2, never 4-bit
    r0 = recommend_foundry_profile(bare, has_torch=False, has_cuda=False)
    print(f"\nbare cpu            : {r0.to_dict()}")
    assert r0.profile == "custom" and not r0.load_in_4bit

    # torch on CPU → a real small/tiny nano-GPT, but NOT GPT-2 (needs a GPU)
    r1 = recommend_foundry_profile(cpu_torch, has_torch=True, has_cuda=False)
    print(f"torch cpu           : {r1.to_dict()}")
    assert r1.profile in ("tiny", "small") and not r1.load_in_4bit

    # strong CUDA GPU → GPT-2 scale (or medium) + QLoRA
    r2 = recommend_foundry_profile(big_gpu, has_torch=True, has_cuda=True)
    print(f"cuda gpu            : {r2.to_dict()}")
    assert r2.profile in ("gpt2", "gpt2-medium") and r2.load_in_4bit
    assert r2.capacity >= 0.55

    # clamp: even asking for a GPU profile with no CUDA must NOT return gpt2/4-bit
    r3 = recommend_foundry_profile(big_gpu, has_torch=True, has_cuda=False)
    assert r3.profile not in ("gpt2", "gpt2-medium") and not r3.load_in_4bit
    print("reality clamp       : no GPT-2 / 4-bit without CUDA ✓")

    # providers
    local = LocalComputeProvider().provision()
    print(f"\nlocal provider      : ok={local.ok} report={local.report.summary() if local.report else None}")
    assert local.ok and local.report is not None

    cloud = CloudComputeProvider("runpod", env={}).provision()
    print(f"cloud (no creds)    : ok={cloud.ok} reason={cloud.reason!r}")
    assert not cloud.ok and "no credentials" in cloud.reason

    cloud2 = CloudComputeProvider("runpod", api_key="sk-test").provision()
    print(f"cloud (creds)       : ok={cloud2.ok} reason={cloud2.reason!r}")
    assert not cloud2.ok and cloud2.meta.get("has_credentials") is True   # honest: no fake rental

    print("\nALL SELF-TESTS PASSED ✓")
