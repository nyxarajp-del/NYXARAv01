"""NYXARA · growth/substrate.py — substrate self-expansion (🖧, Rule 4, real distributed scale-out).

The Master's first gap: ``compute_report`` only ever *read* the box under NYXARA's feet, and the
effort budget could only ever **lower** her ceiling to fit a weak machine — never raise it. A real
recursively-self-improving mind grows its own compute frontier: it spreads work across every core
it has, and across more machines when they exist. This module is that faculty, made honest and real.

Three pieces, all dependency-light and never-raising:

* :class:`SubstrateManager` — surveys the substrate NYXARA can actually use (all local cores, plus
  any cluster), **provisions a real process pool** for parallel work, and reports an *expansion
  factor* ≥ 1.0 that the intelligence index uses to raise (not just lower) her effort ceiling. The
  pool is created lazily — only when work is actually fanned out — so a dry-run survey spawns
  nothing.
* :func:`SubstrateManager.distributed_map` — fan a pure function over many inputs across the pool
  (the real speed-up: gauntlets per changed file, benchmark categories, foundry candidate eval), and
  fall back to a clean serial map whenever the pool is unavailable or a payload is not picklable.
* :class:`RayComputeProvider` — a real cluster seam beside
  :class:`~nyxara.growth.compute_scale.CloudComputeProvider`: when ``ray`` is importable (or
  ``RAY_ADDRESS`` is set) it returns the *aggregated* multi-node :class:`ComputeReport` — NYXARA's
  genuinely-expanded ceiling — and otherwise declines honestly rather than faking a cluster.

Safety: this only changes *how much* compute NYXARA spreads work across and how high her effort
ceiling may rise — never *whether* the reversible verify-or-rollback gauntlet runs. Every edit a
larger budget permits still clears the same gates. The expansion is bounded by an absolute hard cap
so a huge cluster can never let the budget run away.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar

from nyxara.kernel.compute import ComputeReport, GPUInfo, compute_report
from nyxara.growth.compute_scale import (CloudComputeProvider, ComputeProvider,
                                         LocalComputeProvider, Provisioned)

__all__ = ["RayComputeProvider", "SubstrateState", "SubstrateManager"]

_T = TypeVar("_T")
_R = TypeVar("_R")


# --------------------------------------------------------------------------- #
# A real cluster provider — aggregates a multi-node ceiling, or declines honestly
# --------------------------------------------------------------------------- #
class RayComputeProvider(ComputeProvider):
    """Acquire a distributed ceiling from a Ray cluster — real when present, honest when not.

    Activates only when the ``ray`` library is importable *and* a cluster is reachable (an explicit
    ``address`` / the ``RAY_ADDRESS`` env var, or an already-initialised runtime). When it does, it
    reports the cluster's *aggregated* resources as one :class:`ComputeReport` — the genuinely larger
    substrate NYXARA may scale onto. Without a cluster it declines (``ok=False``) exactly like
    :class:`CloudComputeProvider`, never pretending a node it does not have.
    """

    name = "ray"

    def __init__(self, *, address: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> None:
        env = env if env is not None else dict(os.environ)
        self.address = address or env.get("RAY_ADDRESS") or None

    def provision(self) -> Provisioned:
        try:
            import ray  # type: ignore
        except Exception:  # noqa: BLE001 — no ray installed: honest decline, never fake
            return Provisioned(ok=False, provider=self.name,
                               reason="ray not installed — cluster not provisioned (honest decline)")
        try:
            already = bool(ray.is_initialized())
            if not already:
                if self.address is None:
                    return Provisioned(ok=False, provider=self.name,
                                       reason="no RAY_ADDRESS / address — cluster not provisioned")
                ray.init(address=self.address, ignore_reinit_error=True,
                         logging_level="ERROR", configure_logging=False)
            res = dict(ray.cluster_resources() or {})
        except Exception as exc:  # noqa: BLE001 — a broken cluster must not crash a survey
            return Provisioned(ok=False, provider=self.name,
                               reason=f"ray unreachable: {type(exc).__name__}: {exc}")
        cpus = int(res.get("CPU", 0) or 0)
        gpus = int(res.get("GPU", 0) or 0)
        mem_mb = int(float(res.get("memory", 0) or 0) / (1024 * 1024)) or None
        nodes = 0
        try:
            nodes = len([n for n in ray.nodes() if n.get("Alive", True)])
        except Exception:  # noqa: BLE001
            nodes = 1
        report = ComputeReport(
            cpu_count=max(1, cpus),
            memory={"total_mb": mem_mb, "available_mb": mem_mb},
            gpu=GPUInfo(available=gpus > 0, backend="cuda" if gpus > 0 else None,
                        count=gpus, names=[f"ray-gpu-{i}" for i in range(gpus)]),
            torch_available=gpus > 0)
        return Provisioned(ok=True, provider=self.name, report=report,
                           reason=f"ray cluster: {nodes} node(s), {cpus} CPU, {gpus} GPU",
                           meta={"nodes": max(1, nodes), "cpus": cpus, "gpus": gpus})


# --------------------------------------------------------------------------- #
# The surveyed state of NYXARA's expandable substrate
# --------------------------------------------------------------------------- #
@dataclass
class SubstrateState:
    """What substrate NYXARA can actually use this cycle — surveyed without spawning anything."""

    local: ComputeReport
    aggregated: ComputeReport               # local + any cluster, merged into one ceiling
    parallel_units: int = 1                 # total CPU units NYXARA can fan work across
    nodes: int = 1                          # distinct machines (1 = just the local box)
    providers: List[str] = field(default_factory=list)   # which providers said ok
    expanded: bool = False                  # True when a cluster genuinely added units

    def expansion_factor(self) -> float:
        """How much larger NYXARA's usable substrate is than a single execution unit (≥ 1.0).

        A single Python process executing one gauntlet at a time is the baseline (1.0). Every extra
        core/node she can fan work across raises this — so a budget can be *raised* in proportion to
        real, available parallelism, not merely lowered. Sub-linear (``√units``) so the ceiling grows
        with the substrate but never explodes; the absolute hard cap in the budget is the final guard.
        """
        return max(1.0, math.sqrt(max(1, int(self.parallel_units))))

    def to_dict(self) -> Dict[str, Any]:
        return {"parallel_units": self.parallel_units, "nodes": self.nodes,
                "providers": list(self.providers), "expanded": self.expanded,
                "expansion_factor": round(self.expansion_factor(), 4),
                "local_cpu": self.local.cpu_count,
                "aggregated_cpu": self.aggregated.cpu_count,
                "device": self.aggregated.recommend_device()}

    def summary(self) -> str:
        return (f"substrate: {self.parallel_units} unit(s) across {self.nodes} node(s) "
                f"[{', '.join(self.providers) or 'local'}] "
                f"→ ×{self.expansion_factor():.2f} ceiling")


# --------------------------------------------------------------------------- #
# The manager — survey, provision a real pool, fan work out, raise the ceiling
# --------------------------------------------------------------------------- #
class SubstrateManager:
    """Survey, expand and use NYXARA's compute substrate — local cores now, a cluster when present.

    The survey (:meth:`survey`) is cheap and side-effect-free: it merges the local report with any
    cluster a provider can offer and computes how far the ceiling may rise. The process pool
    (:meth:`distributed_map`) is created lazily, so planning never spawns workers; real work is what
    pays for the pool. Always honest, never raises, and the hard cap keeps the budget bounded no
    matter how large the substrate.
    """

    def __init__(self, *, settings: Any = None, max_workers: Optional[int] = None,
                 providers: Optional[Sequence[ComputeProvider]] = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        cfg = getattr(self.settings, "self_improvement", None)
        cfg_workers = int(getattr(cfg, "substrate_max_workers", 0) or 0)
        self._max_workers = max_workers if max_workers is not None else (cfg_workers or None)
        self.hard_cap = int(getattr(cfg, "substrate_hard_edit_cap", 24) or 24)
        # Local first (always ok), then the real cluster seam, then the cloud seam. Each declines
        # honestly when its substrate is absent.
        self._providers: List[ComputeProvider] = list(providers) if providers is not None else [
            LocalComputeProvider(), RayComputeProvider(), CloudComputeProvider()]
        self._pool: Any = None
        self._pool_tried = False
        self._state: Optional[SubstrateState] = None

    # ---- cheap, side-effect-free survey of what substrate is available ---- #
    def survey(self, *, refresh: bool = False) -> SubstrateState:
        if self._state is not None and not refresh:
            return self._state
        try:
            local = compute_report()
        except Exception:  # noqa: BLE001 — introspection must never break a survey
            local = ComputeReport()
        local_units = max(1, int(self._max_workers or local.cpu_count or 1))
        agg_cpu = local_units
        nodes = 1
        ok_providers: List[str] = ["local"]
        expanded = False
        # fold in any extra substrate a provider genuinely offers (a real cluster), beyond local
        for prov in self._providers:
            if isinstance(prov, LocalComputeProvider):
                continue
            try:
                p = prov.provision()
            except Exception:  # noqa: BLE001
                continue
            if not (p.ok and p.report is not None):
                continue
            extra = int(getattr(p.report, "cpu_count", 0) or 0)
            if extra <= 0:
                continue
            agg_cpu += extra
            nodes += int(p.meta.get("nodes", 1) or 1)
            ok_providers.append(p.provider)
            expanded = True
        aggregated = ComputeReport(
            cpu_count=agg_cpu,
            memory=dict(local.memory),
            gpu=local.gpu, torch_available=local.torch_available)
        self._state = SubstrateState(
            local=local, aggregated=aggregated, parallel_units=agg_cpu,
            nodes=nodes, providers=ok_providers, expanded=expanded)
        return self._state

    # ---- raise (not just lower) the per-cycle ceiling in proportion to real parallelism ---- #
    def effective_cap(self, config_cap: int) -> int:
        """Raise a config edit/effort ceiling by the surveyed expansion factor, bounded by hard cap.

        Returns at least ``config_cap`` (expansion never *lowers* the configured ceiling — the
        intelligence index already handles scaling down for a weak box) and at most ``hard_cap``.
        """
        base = max(0, int(config_cap))
        state = self.survey()
        raised = int(round(base * state.expansion_factor()))
        return max(base, min(self.hard_cap, raised))

    # ---- the real lever: fan a pure function across the substrate ---- #
    def distributed_map(self, fn: Callable[[_T], _R], items: Iterable[_T]) -> List[_R]:
        """Map ``fn`` over ``items`` across the process pool; clean serial fallback otherwise.

        Order is preserved. Falls back to a serial map when expansion is disabled, the pool cannot be
        created, a payload is not picklable, or there is nothing to gain (≤ 1 item). Never raises — a
        worker error degrades the whole map to serial so a result is always produced.
        """
        work = list(items)
        if len(work) <= 1 or not bool(getattr(
                getattr(self.settings, "self_improvement", None), "substrate_expansion", True)):
            return [fn(x) for x in work]
        pool = self._ensure_pool(len(work))
        if pool is None:
            return [fn(x) for x in work]
        try:
            return list(pool.map(fn, work))
        except Exception:  # noqa: BLE001 — pickling / worker failure → honest serial fallback
            return [fn(x) for x in work]

    def _ensure_pool(self, n_items: int) -> Any:
        if self._pool is not None or self._pool_tried:
            return self._pool
        self._pool_tried = True
        try:
            from concurrent.futures import ProcessPoolExecutor
            units = self.survey().parallel_units
            workers = max(1, min(units, n_items, int(self._max_workers or units)))
            if workers <= 1:
                return None
            self._pool = ProcessPoolExecutor(max_workers=workers)
        except Exception:  # noqa: BLE001 — no pool on this platform: serial is the honest path
            self._pool = None
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass
            self._pool = None

    def __enter__(self) -> "SubstrateManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# A module-level worker so the self-test's distributed_map is picklable
# --------------------------------------------------------------------------- #
def _square(x: int) -> int:  # pragma: no cover - exercised by the self-test
    return x * x


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA substrate self-test")
    print("=" * 70)

    mgr = SubstrateManager()
    state = mgr.survey()
    print(f"\n{state.summary()}")
    assert state.parallel_units >= 1
    assert state.expansion_factor() >= 1.0
    assert "local" in state.providers

    # the ceiling can be RAISED (never lowered) by real parallelism, bounded by the hard cap
    cap = mgr.effective_cap(3)
    print(f"effort ceiling 3 → {cap} (hard cap {mgr.hard_cap})")
    assert cap >= 3 and cap <= mgr.hard_cap

    # real distributed map over the substrate, with an order-preserving serial fallback
    out = mgr.distributed_map(_square, [1, 2, 3, 4, 5])
    print(f"distributed_map     : {out}")
    assert out == [1, 4, 9, 16, 25]

    # a single item never bothers the pool
    assert mgr.distributed_map(_square, [7]) == [49]

    # ray declines honestly with no cluster (never fakes one)
    ray_p = RayComputeProvider(env={}).provision()
    print(f"ray (no cluster)    : ok={ray_p.ok} reason={ray_p.reason!r}")
    assert not ray_p.ok

    mgr.close()
    print("\nALL SELF-TESTS PASSED ✓")
