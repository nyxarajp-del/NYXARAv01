"""NYXARA · kernel/compute.py — honest compute & device introspection (🖥, best-effort).

NYXARA's heavier faculties (the foundry, local LLM providers, vision/audio) only make
sense where the hardware can carry them. This module reports — honestly and without heavy
dependencies — what compute is actually available: CPU count, a best-effort read of system
memory, and GPU availability (import-guarded via ``torch``, returning *absent* rather than
erroring on a bare machine).

It never raises and never imports anything heavy at module load; ``torch`` is probed lazily
only when asked. :func:`recommend_device` turns the report into a single best-effort device
string (``"cuda"`` / ``"mps"`` / ``"cpu"``) for callers that just want a hint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["GPUInfo", "ComputeReport", "compute_report", "recommend_device"]


def _read_meminfo_mb() -> Dict[str, Optional[int]]:
    """Best-effort total/available RAM in MB from /proc/meminfo (Linux). Never raises."""
    total = avail = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024          # kB -> MB
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                if total is not None and avail is not None:
                    break
    except Exception:  # noqa: BLE001 — not all platforms expose /proc/meminfo
        pass
    return {"total_mb": total, "available_mb": avail}


def _gpu_info() -> "GPUInfo":
    """Probe GPU availability via torch, import-guarded. Absent (not error) without torch."""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return GPUInfo(available=False, backend=None, count=0, names=[])

    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            return GPUInfo(available=True, backend="cuda", count=count, names=names)
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return GPUInfo(available=True, backend="mps", count=1, names=["mps"])
    except Exception:  # noqa: BLE001 — a broken driver must not crash a report
        pass
    return GPUInfo(available=False, backend=None, count=0, names=[])


@dataclass
class GPUInfo:
    available: bool = False
    backend: Optional[str] = None      # "cuda" / "mps" / None
    count: int = 0
    names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"available": self.available, "backend": self.backend,
                "count": self.count, "names": list(self.names)}


@dataclass
class ComputeReport:
    cpu_count: int = 1
    memory: Dict[str, Optional[int]] = field(default_factory=dict)
    gpu: GPUInfo = field(default_factory=GPUInfo)
    torch_available: bool = False

    def recommend_device(self) -> str:
        if self.gpu.available and self.gpu.backend:
            return self.gpu.backend
        return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {"cpu_count": self.cpu_count, "memory": dict(self.memory),
                "gpu": self.gpu.to_dict(), "torch_available": self.torch_available,
                "recommended_device": self.recommend_device()}

    def summary(self) -> str:
        mem = self.memory.get("total_mb")
        mem_s = f"{mem} MB" if mem else "unknown"
        gpu_s = (f"{self.gpu.backend} x{self.gpu.count} ({', '.join(self.gpu.names)})"
                 if self.gpu.available else "none")
        return (f"cpu={self.cpu_count}  ram={mem_s}  gpu={gpu_s}  "
                f"-> device={self.recommend_device()}")


def compute_report() -> ComputeReport:
    """Build a full, best-effort compute report. Never raises."""
    gpu = _gpu_info()
    try:
        import torch  # noqa: F401
        torch_ok = True
    except Exception:  # noqa: BLE001
        torch_ok = False
    return ComputeReport(cpu_count=os.cpu_count() or 1, memory=_read_meminfo_mb(),
                         gpu=gpu, torch_available=torch_ok)


def recommend_device() -> str:
    """Best-effort single device hint: 'cuda' / 'mps' / 'cpu'."""
    return compute_report().recommend_device()


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA compute-report self-test")
    print("=" * 70)

    rep = compute_report()
    print(f"\nreport              : {rep.summary()}")
    d = rep.to_dict()

    assert d["cpu_count"] >= 1
    assert "gpu" in d and isinstance(d["gpu"], dict)
    assert "available" in d["gpu"]
    assert d["recommended_device"] in ("cuda", "mps", "cpu")
    # on a bare, torch-less machine the recommendation is honestly 'cpu'
    if not rep.torch_available:
        assert rep.recommend_device() == "cpu"
        assert rep.gpu.available is False
    print(f"recommend_device    : {recommend_device()} ✓")
    print(f"as dict             : {d}")

    print("\nALL SELF-TESTS PASSED ✓")
