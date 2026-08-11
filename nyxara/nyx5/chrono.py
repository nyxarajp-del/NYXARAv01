"""NYXARA · nyx5/chrono.py — the chrono-dilation engine (⏳, pillar 4).

Honest name for an honest mechanism: **anytime iterative deepening**, not literal time travel. On a calm
turn NYX-5 answers shallowly and fast. When surprise or difficulty crosses a trigger, she enters a
"hyper-clock" state — spending *more internal iterations* (deeper search, more settle cycles) on the
problem — but always inside a **real wall-clock deadline**, keeping the best answer found so far. The
"1000 subjective hours in a blink" image is exactly that: many bounded internal steps per real second,
never a claim that a CPU runs faster than it does.

Pure standard library. Depends on nothing else in NYX-5.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

__all__ = ["ChronoResult", "ChronoDilator"]


@dataclass
class ChronoResult:
    """The outcome of a dilated computation: best result, the depth reached, and why it stopped."""

    result: Any
    depth: int
    quality: float
    dilated: bool                    # did we enter the hyper-clock (deepen past the shallow floor)?
    elapsed_ms: float
    stopped: str                     # "converged" | "deadline" | "max_depth" | "shallow"

    def to_dict(self) -> Dict[str, Any]:
        return {"depth": self.depth, "quality": round(self.quality, 4), "dilated": self.dilated,
                "elapsed_ms": round(self.elapsed_ms, 3), "stopped": self.stopped}


class ChronoDilator:
    """Run a depth-parametrised computation as deep as the deadline and difficulty warrant.

    ``work(depth) -> (result, quality)`` must be *anytime*: callable at any depth, returning a usable
    result and a quality ∈ [0,1] that generally improves with depth. The dilator escalates depth only
    when ``pressure`` (surprise/EFE/difficulty ∈ [0,1]) crosses ``trigger``; otherwise it returns the
    shallow answer immediately. It never exceeds ``deadline_ms`` of wall-clock time.
    """

    def __init__(self, *, deadline_ms: float = 2000.0, max_depth: int = 64, trigger: float = 0.6,
                 converge_eps: float = 1e-4, clock: Callable[[], float] = time.monotonic) -> None:
        self.deadline_ms = float(deadline_ms)
        self.max_depth = int(max_depth)
        self.trigger = float(trigger)
        self.converge_eps = float(converge_eps)
        self._clock = clock

    def run(self, work: Callable[[int], Tuple[Any, float]], *, pressure: float = 0.0,
            shallow_depth: int = 1) -> ChronoResult:
        start = self._clock()
        deadline = start + self.deadline_ms / 1000.0

        result, quality = work(shallow_depth)
        best: Tuple[Any, float, int] = (result, quality, shallow_depth)

        if pressure < self.trigger:
            return ChronoResult(result=best[0], depth=best[2], quality=best[1], dilated=False,
                                elapsed_ms=(self._clock() - start) * 1000.0, stopped="shallow")

        stopped = "max_depth"
        depth = shallow_depth
        while depth < self.max_depth:
            if self._clock() >= deadline:
                stopped = "deadline"
                break
            depth += 1
            r, q = work(depth)
            if q > best[1]:
                best = (r, q, depth)
            if abs(q - best[1]) < self.converge_eps and q >= best[1]:
                stopped = "converged"
                break
        return ChronoResult(result=best[0], depth=best[2], quality=best[1], dilated=True,
                            elapsed_ms=(self._clock() - start) * 1000.0, stopped=stopped)
