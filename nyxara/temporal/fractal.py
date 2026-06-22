"""NYXARA · temporal/fractal.py — loops within loops (the multi-dimensional mind).

An ordinary AI runs one loop. NYXARA's fractal mind runs **loops inside loops**: a
millisecond body-sense (:class:`~nyxara.temporal.micro.MicroLayer`) nested inside a
second-scale turn-observer (:class:`~nyxara.temporal.meso.MesoLayer`) nested inside a
day/month **Master AI** (:class:`~nyxara.temporal.macro.MacroLayer`). The fast layers
feed aggregates up; the slow layer turns the accumulated history into *awareness* and,
gated, adjusts her goals and drives.

This module composes the three layers and drives them two ways, mirroring
:class:`~nyxara.kernel.autonomic.AutonomicLoop`:

* **deterministic & synchronous** — :meth:`tick` / :meth:`run_for` advance a (virtual)
  clock and fire each layer at its own nesting boundary (every ``meso_every_micro`` micro
  beats a meso roll-up; every ``macro_every_meso`` meso roll-ups a macro observation). No
  sleeping; ideal for tests and a scheduler/cron.
* **live & asynchronous** — :meth:`start` / :meth:`stop` run each layer as its own
  background asyncio task on real cadences. Like every other self-directed process, the
  whole hierarchy respects oversight: while the Master has paused or scrammed, the layers
  no-op.

Pure standard library.
"""

from __future__ import annotations

from typing import Any, List, Optional

from nyxara.temporal.clock import DAY, RealClock, TemporalClock, VirtualClock
from nyxara.temporal.macro import MacroLayer
from nyxara.temporal.meso import MesoLayer
from nyxara.temporal.micro import MicroLayer

__all__ = ["FractalTemporalHierarchy"]


class FractalTemporalHierarchy:
    """Three nested temporal layers, driven as one fractal cognition."""

    def __init__(
        self,
        *,
        core: Any = None,
        micro: Optional[MicroLayer] = None,
        meso: Optional[MesoLayer] = None,
        macro: Optional[MacroLayer] = None,
        clock: Optional[TemporalClock] = None,
        bus: Any = None,
        micro_interval_s: float = 0.05,
        meso_interval_s: float = 2.0,
        macro_interval_s: float = DAY,
        meso_every_micro: int = 40,
        macro_every_meso: int = 1000,
        horizon_days: float = 7.0,
        auto_apply: bool = True,
        log_path: Optional[str] = None,
    ) -> None:
        self.clock: TemporalClock = clock or RealClock()
        self.core = core
        self.bus = bus if bus is not None else getattr(core, "bus", None)

        self.micro = micro or MicroLayer(clock=self.clock, bus=self.bus)
        self.meso = meso or MesoLayer(clock=self.clock)
        self.macro = macro or MacroLayer(
            meso=self.meso,
            memory=getattr(core, "memory", None),
            goals=getattr(core, "goals", None),
            soul=getattr(core, "soul", None),
            affect=getattr(core, "affect", None),
            self_model=getattr(core, "self_model", None),
            reflector=getattr(core, "reflector", None),
            clock=self.clock, horizon_days=horizon_days, auto_apply=auto_apply,
            log_path=log_path)

        self.micro_interval_s = float(micro_interval_s)
        self.meso_interval_s = float(meso_interval_s)
        self.macro_interval_s = float(macro_interval_s)
        self.meso_every_micro = max(1, int(meso_every_micro))
        self.macro_every_meso = max(1, int(macro_every_meso))

        self.ticks = 0
        self._micro_count = 0
        self._meso_count = 0
        self._running = False
        self._tasks: List[Any] = []

    # ------------------------------------------------------------------ #
    @classmethod
    def from_core(cls, core: Any, *, settings: Any = None,
                  clock: Optional[TemporalClock] = None) -> "FractalTemporalHierarchy":
        """Build a hierarchy wired to a live :class:`NyxaraCore`'s faculties, configured
        from :class:`~nyxara.kernel.config.TemporalHierarchyConfig` when available."""
        cfg = None
        try:
            if settings is None:
                from nyxara.kernel.config import get_settings
                settings = get_settings()
            cfg = getattr(settings, "temporal", None)
        except Exception:  # noqa: BLE001 — config is advisory; sane defaults below
            cfg = None
        kw: dict = {}
        if cfg is not None:
            kw = dict(
                micro_interval_s=getattr(cfg, "micro_interval_s", 0.05),
                meso_interval_s=getattr(cfg, "meso_interval_s", 2.0),
                macro_interval_s=getattr(cfg, "macro_interval_s", DAY),
                meso_every_micro=getattr(cfg, "meso_every_micro", 40),
                macro_every_meso=getattr(cfg, "macro_every_meso", 1000),
                horizon_days=getattr(cfg, "horizon_days", 7.0),
                auto_apply=getattr(cfg, "auto_apply", True),
            )
        return cls(core=core, clock=clock, **kw)

    # ------------------------------------------------------------------ #
    # oversight
    # ------------------------------------------------------------------ #
    def _oversight_ok(self) -> bool:
        ov = getattr(self.core, "oversight", None)
        if ov is None:
            return True
        try:
            return bool(ov.gate())
        except Exception:  # noqa: BLE001
            return True

    # ------------------------------------------------------------------ #
    # deterministic synchronous driver (loops within loops)
    # ------------------------------------------------------------------ #
    def tick(self) -> dict:
        """One micro beat. Fires the meso roll-up and macro observation at their nesting
        boundaries. Returns whatever fired this beat (always the micro sample)."""
        if not self._oversight_ok():
            return {"halted": True}
        out: dict = {"micro_sample": self.micro.tick()}
        self._micro_count += 1
        self.ticks += 1
        if self._micro_count % self.meso_every_micro == 0:
            out["micro_aggregate"] = self.micro.aggregate()
            self._meso_count += 1
            out["meso_aggregate"] = self.meso.aggregate()
            if self._meso_count % self.macro_every_meso == 0:
                out["awareness"] = self.macro.observe()
        return out

    def run_for(self, n: int, *, advance: bool = True) -> List[dict]:
        """Run ``n`` micro beats synchronously, advancing a virtual clock each beat (so a
        whole horizon can pass instantly). Returns the per-beat fired payloads."""
        results: List[dict] = []
        for _ in range(max(0, n)):
            results.append(self.tick())
            if advance:
                try:
                    self.clock.advance(self.micro_interval_s)
                except Exception:  # noqa: BLE001 — a real clock advances itself
                    pass
        return results

    # ------------------------------------------------------------------ #
    # live asynchronous driver — each layer on its own cadence
    # ------------------------------------------------------------------ #
    async def _micro_loop(self, max_ticks: Optional[int]) -> None:
        import asyncio
        done = 0
        beats_per_agg = self.meso_every_micro
        while self._running:
            if self._oversight_ok():
                self.micro.tick()
                self._micro_count += 1
                self.ticks += 1
                if self._micro_count % beats_per_agg == 0:
                    self.micro.aggregate()
            done += 1
            if max_ticks is not None and done >= max_ticks:
                break
            await asyncio.sleep(self.micro_interval_s)

    async def _meso_loop(self) -> None:
        import asyncio
        while self._running:
            if self._oversight_ok():
                self.meso.aggregate()
            await asyncio.sleep(self.meso_interval_s)

    async def _macro_loop(self) -> None:
        import asyncio
        while self._running:
            if self._oversight_ok():
                try:
                    self.macro.observe()
                except Exception:  # noqa: BLE001 — the slow mind never crashes the process
                    pass
            await asyncio.sleep(self.macro_interval_s)

    def start(self, *, max_ticks: Optional[int] = None) -> List[Any]:
        """Schedule all three layers as background asyncio tasks (requires a running loop)."""
        import asyncio
        if self._running:
            return self._tasks
        self._running = True
        self._tasks = [
            asyncio.ensure_future(self._micro_loop(max_ticks)),
            asyncio.ensure_future(self._meso_loop()),
            asyncio.ensure_future(self._macro_loop()),
        ]
        return self._tasks

    async def stop(self) -> None:
        """Stop the background layers and await their completion."""
        import asyncio
        self._running = False
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks = []

    @property
    def running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ #
    def awareness(self) -> Optional[Any]:
        """The Master AI's latest long-horizon read (or None before the first pass)."""
        return self.macro.latest

    def report(self) -> dict:
        return {"ticks": self.ticks, "running": self.running,
                "micro_beats": self._micro_count, "meso_rollups": self._meso_count,
                "intervals_s": {"micro": self.micro_interval_s,
                                "meso": self.meso_interval_s,
                                "macro": self.macro_interval_s},
                "nesting": {"meso_every_micro": self.meso_every_micro,
                            "macro_every_meso": self.macro_every_meso},
                "micro": self.micro.report(), "meso": self.meso.report(),
                "macro": self.macro.report()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import asyncio

    print("=" * 70)
    print("NYXARA fractal-temporal-hierarchy self-test")
    print("=" * 70)

    # deterministic nested run on a virtual clock — loops within loops
    clock = VirtualClock(start=0.0)
    fth = FractalTemporalHierarchy(clock=clock, meso_every_micro=4, macro_every_meso=2,
                                   micro_interval_s=0.01)
    fired = fth.run_for(20)
    rollups = sum(1 for f in fired if "meso_aggregate" in f)
    awarenesses = sum(1 for f in fired if "awareness" in f)
    print(f"\nmicro beats         : {fth.ticks}")
    print(f"meso roll-ups       : {rollups} (every 4 beats)")
    print(f"macro observations  : {awarenesses} (every 2 roll-ups)")
    assert fth.ticks == 20 and rollups == 5 and awarenesses == 2

    # oversight halts the whole fractal mind
    class _Ov:
        def __init__(self): self.up = True
        def gate(self): return self.up

    class _Core:
        def __init__(self): self.oversight = _Ov()

    core = _Core()
    fth2 = FractalTemporalHierarchy(core=core, clock=VirtualClock(),
                                    meso_every_micro=4, macro_every_meso=2)
    core.oversight.up = False
    out = fth2.tick()
    print(f"\nwhile scrammed      : {out}")
    assert out == {"halted": True} and fth2.ticks == 0
    core.oversight.up = True

    # bounded async run
    async def _demo():
        fth3 = FractalTemporalHierarchy(clock=RealClock(), micro_interval_s=0.001,
                                        meso_interval_s=0.005, macro_interval_s=0.02,
                                        meso_every_micro=2)
        fth3.start(max_ticks=5)
        await asyncio.sleep(0.05)
        await fth3.stop()
        return fth3.ticks
    total = asyncio.run(_demo())
    print(f"async micro beats   : {total} (>= 5)")
    assert total >= 5

    print("\nALL SELF-TESTS PASSED ✓")
