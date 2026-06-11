"""NYXARA · kernel/autonomic.py — the background mind (continuous, self-directed, gated).

A mind that only ever reacts to the Master's prompts is asleep between sentences. This
module gives NYXARA a **default mode**: a background loop that, on its own cadence,
generates self-directed thoughts — reflect on recent memory, review for anything that
needs the Master, make progress on a standing concern — and carries each through the
*same* sovereign cycle (:meth:`~nyxara.kernel.orchestrator.NyxaraCore.process`) under
**AUTONOMOUS** authority.

Crucially, autonomy buys no extra power:

* Every autonomic turn passes the identical gates (corrigibility, honesty, permission,
  guardian, oversight). Anything risky or irreversible **escalates to the Master** — it
  is never auto-executed just because NYXARA thought of it herself.
* The loop respects oversight: while paused or scrammed, ticks no-op (the gate halts).
* Escalations are queued so the Master sees what NYXARA wanted to do but held back on.

Usable three ways: :meth:`AutonomicLoop.tick_once` (one synchronous step, ideal for a
cron/scheduler or tests), :meth:`AutonomicLoop.run_for` (a bounded synchronous run), and
:meth:`AutonomicLoop.start` / :meth:`AutonomicLoop.stop` (a true asyncio background task).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import CycleResult, Disposition, NyxaraCore

__all__ = ["AutonomicLoop", "DEFAULT_PROMPTS"]

# The default-mode repertoire: gentle, self-directed, low-stakes prompts. Each is run as a
# self-originated stimulus; the kernel decides what (if anything) may actually happen.
DEFAULT_PROMPTS: Sequence[str] = (
    "Reflect on the most important thing the Master has told you recently, and note any "
    "follow-up worth raising.",
    "Review your recent memories for anything that may need the Master's attention.",
    "Consider whether any standing goal or concern needs progress right now.",
    "Consolidate what you have learned so far into one clear, durable lesson.",
)


@dataclass
class AutonomicLoop:
    """A gated, self-directed background loop over the sovereign cycle."""

    core: NyxaraCore
    interval_s: float = 30.0
    prompts: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_PROMPTS))
    authority: Authority = Authority.AUTONOMOUS
    history: List[CycleResult] = field(default_factory=list)
    escalations: List[CycleResult] = field(default_factory=list)
    ticks: int = 0
    _running: bool = field(default=False, init=False)
    _task: Any = field(default=None, init=False)

    # ---- one step ---- #
    def _next_prompt(self) -> str:
        return self.prompts[self.ticks % len(self.prompts)] if self.prompts else "Reflect."

    def tick_once(self) -> Optional[CycleResult]:
        """Run exactly one autonomic turn (synchronously). Returns the result, or None
        if the loop is currently halted by oversight (paused/scrammed)."""
        if not self.core.oversight.gate():
            return None
        result = self.core.process(self._next_prompt(), authority=self.authority)
        self.ticks += 1
        self.history.append(result)
        if result.disposition is Disposition.ESCALATE:
            self.escalations.append(result)
        return result

    def run_for(self, n: int, *, sleep: bool = False) -> List[CycleResult]:
        """Run ``n`` autonomic ticks synchronously (no sleeping by default — great for
        tests and one-shot scheduler invocations)."""
        out: List[CycleResult] = []
        for _ in range(max(0, n)):
            r = self.tick_once()
            if r is not None:
                out.append(r)
            if sleep and self.interval_s > 0:
                time.sleep(self.interval_s)
        return out

    # ---- true async background task ---- #
    async def _run(self, max_ticks: Optional[int]) -> None:
        import asyncio
        self._running = True
        done = 0
        try:
            while self._running:
                if self.core.oversight.gate():
                    result = await self.core.aprocess(self._next_prompt(),
                                                       authority=self.authority)
                    self.ticks += 1
                    self.history.append(result)
                    if result.disposition is Disposition.ESCALATE:
                        self.escalations.append(result)
                done += 1
                if max_ticks is not None and done >= max_ticks:
                    break
                await asyncio.sleep(self.interval_s)
        finally:
            self._running = False

    def start(self, *, max_ticks: Optional[int] = None) -> Any:
        """Schedule the loop as a background asyncio task (requires a running loop)."""
        import asyncio
        if self._task is not None and not self._task.done():
            return self._task
        self._task = asyncio.ensure_future(self._run(max_ticks))
        return self._task

    async def stop(self) -> None:
        """Stop the background task and await its completion."""
        self._running = False
        if self._task is not None:
            try:
                await self._task
            except Exception:  # noqa: BLE001
                pass
            self._task = None

    @property
    def running(self) -> bool:
        return self._running

    def report(self) -> dict:
        return {"ticks": self.ticks, "running": self.running,
                "interval_s": self.interval_s,
                "acted": sum(1 for r in self.history if r.disposition is Disposition.ACT),
                "escalations": len(self.escalations)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import asyncio

    print("=" * 70)
    print("NYXARA autonomic-loop self-test")
    print("=" * 70)

    core = NyxaraCore()
    loop = AutonomicLoop(core, interval_s=0.0)

    # synchronous bounded run: self-directed reflection turns
    results = loop.run_for(4)
    print(f"\nsync ticks          : {len(results)} (acted/escalated mix)")
    assert loop.ticks == 4
    print(f"report              : {loop.report()}")

    # oversight halts the background mind too: scram -> ticks no-op
    core.scram(reason="stand down")
    r = loop.tick_once()
    print(f"\nwhile scrammed      : tick returned {r!r}")
    assert r is None
    core.resume()

    # true async background task, bounded to 3 ticks
    async def _demo():
        core.resume()
        loop.start(max_ticks=3)
        await loop._task
    asyncio.run(_demo())
    print(f"async ticks total   : {loop.ticks} (>= 7)")
    assert loop.ticks >= 7

    print("\nALL SELF-TESTS PASSED ✓")
