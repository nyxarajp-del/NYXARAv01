"""NYXARA · causal/self_simulation.py — Recursive Self-Simulation Multithreading (OMNISCIENCE · 4).

*The Master's ask:* on a high-stakes architectural decision or complex repair, spawn many
**independent parallel simulation sandboxes**, each exploring a different hypothesis; the branch
with the **lowest free-energy (surprise)** and **highest verification score** wins, and the core
collapses production onto that single winning path.

What that means here, honestly. :meth:`SelfSimulationRace.race` takes a set of candidate
hypotheses and a caller-supplied ``rollout`` (a real simulation — e.g. via
:mod:`nyxara.mind.world_simulator` / :mod:`nyxara.sim.sandbox`) and runs a **bounded** number of
them **concurrently** in a thread pool. Each rollout is wrapped so a crashing branch is contained
(the "sandbox") and never sinks the race. Every branch is scored by:

    * its **free energy** ``F`` (surprise — lower is better), and
    * a **verification** score in ``[0, 1]`` (higher is better);

combined as ``score = w_v·verification − w_f·(F normalised across branches)``. The maximum-score
branch is the **winner** the caller then collapses to.

The honest boundary. This is a real, contained, concurrent evaluator that genuinely picks the
best-verified / least-surprising branch. The parallelism is a bounded Python thread pool (``N`` is
capped by config, not literally always 100), and the branches must be independent for the sandbox
guarantee to hold. It selects among the hypotheses it is *given* — it does not conjure the right one
from nothing.

Bridges: rollouts reuse :mod:`nyxara.mind.world_simulator`, :mod:`nyxara.sim.sandbox`,
:mod:`nyxara.sim.montecarlo`; scoring reuses :mod:`nyxara.mind.free_energy`; a competing-branch
search complements :mod:`nyxara.mind.mcts_reasoner`.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["SimBranch", "SimulationVerdict", "SelfSimulationRace"]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class SimBranch:
    """One sandboxed rollout of a single hypothesis."""

    index: int
    hypothesis: Any
    free_energy: float = float("inf")
    verification: float = 0.0
    score: float = float("-inf")
    ok: bool = False
    duration_ms: float = 0.0
    outcome: Any = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "hypothesis": _short(self.hypothesis),
                "free_energy": _round(self.free_energy), "verification": round(self.verification, 4),
                "score": _round(self.score), "ok": self.ok,
                "duration_ms": round(self.duration_ms, 3), "error": self.error}


@dataclass
class SimulationVerdict:
    """The collapse: the winning branch and the full field it beat."""

    winner: Optional[SimBranch]
    branches: List[SimBranch]
    evaluated: int
    collapsed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"collapsed": self.collapsed, "evaluated": self.evaluated,
                "winner": None if self.winner is None else self.winner.to_dict(),
                "branches": [b.to_dict() for b in self.branches]}


# --------------------------------------------------------------------------- #
# The race
# --------------------------------------------------------------------------- #
class SelfSimulationRace:
    """Runs bounded, contained, parallel rollouts and collapses to the best branch."""

    def __init__(self, *, max_branches: int = 16, max_workers: int = 8,
                 w_verification: float = 1.0, w_free_energy: float = 1.0,
                 timeout_s: Optional[float] = None) -> None:
        self.max_branches = int(max_branches)
        self.max_workers = int(max_workers)
        self.w_verification = float(w_verification)
        self.w_free_energy = float(w_free_energy)
        self.timeout_s = timeout_s
        self._races = 0
        self._timeouts = 0

    def race(self, hypotheses: Sequence[Any],
             rollout: Callable[[Any], Tuple[Any, float]], *,
             verify: Optional[Callable[[Any], float]] = None) -> SimulationVerdict:
        """Race ``hypotheses``.

        ``rollout(h)`` runs one simulation and returns ``(outcome, free_energy)``. Optional
        ``verify(outcome)`` returns a score in ``[0, 1]``. The lowest-free-energy / best-verified
        branch wins.

        With ``timeout_s`` set, the race collapses on the deadline over whichever branches
        finished; the overrunning ones are marked failed with a timeout error and abandoned
        (a Python thread cannot be killed, so they run themselves out harmlessly in the
        background — they simply no longer hold up the verdict or touch it).
        """
        chosen = list(hypotheses)[: self.max_branches]
        if not chosen:
            return SimulationVerdict(winner=None, branches=[], evaluated=0)
        self._races += 1
        branches: List[SimBranch] = [SimBranch(index=i, hypothesis=h) for i, h in enumerate(chosen)]

        # The worker returns its result rather than writing the branch in place: a rollout that
        # overran the deadline is still running when the verdict is built, and letting it touch
        # the shared branch object would race the collapse it already lost.
        def _run(hypothesis: Any) -> Dict[str, Any]:
            t0 = time.perf_counter()
            try:
                outcome, fe = rollout(hypothesis)
                verification = float(verify(outcome)) if verify is not None else 1.0
                return {"ok": True, "outcome": outcome, "free_energy": float(fe),
                        "verification": verification, "error": "",
                        "duration_ms": (time.perf_counter() - t0) * 1000.0}
            except Exception as exc:  # noqa: BLE001 — the sandbox: a crash stays in its branch
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "duration_ms": (time.perf_counter() - t0) * 1000.0}

        workers = max(1, min(self.max_workers, len(branches)))
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(_run, b.hypothesis): b for b in branches}
            try:
                for fut in as_completed(futures, timeout=self.timeout_s):
                    _apply(futures[fut], fut.result())
            except FuturesTimeout:
                # ``as_completed`` only *raises* on the deadline — on its own that both loses the
                # branches that did land and still blocks in the executor's exit until every
                # overrunning rollout finishes, so the timeout bounded nothing at all. Bank the
                # finished branches, mark the rest timed-out, and stop waiting for them.
                for fut, branch in futures.items():
                    if fut.done() and not fut.cancelled():
                        _apply(branch, fut.result())
                    else:
                        fut.cancel()
                        self._timeouts += 1
                        branch.ok = False
                        branch.error = f"TimeoutError: exceeded timeout_s={self.timeout_s}"
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        self._score(branches)
        ok = [b for b in branches if b.ok]
        winner = max(ok, key=lambda b: b.score) if ok else None
        return SimulationVerdict(winner=winner, branches=branches,
                                 evaluated=len(branches), collapsed=winner is not None)

    def _score(self, branches: Sequence[SimBranch]) -> None:
        """Normalise free-energy across the ok branches, then combine with verification."""
        ok = [b for b in branches if b.ok]
        if not ok:
            return
        fes = [b.free_energy for b in ok]
        lo, hi = min(fes), max(fes)
        span = (hi - lo) or 1.0
        for b in ok:
            norm_fe = (b.free_energy - lo) / span   # 0 (best) .. 1 (worst)
            b.score = self.w_verification * b.verification - self.w_free_energy * norm_fe

    def status(self) -> Dict[str, Any]:
        return {"races": self._races, "max_branches": self.max_branches,
                "max_workers": self.max_workers, "timeouts": self._timeouts,
                "timeout_s": self.timeout_s}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _apply(branch: SimBranch, result: Dict[str, Any]) -> None:
    """Fold a worker's returned result onto its branch (main thread only)."""
    branch.duration_ms = result.get("duration_ms", 0.0)
    branch.error = result.get("error", "")
    if result.get("ok"):
        branch.outcome = result.get("outcome")
        branch.free_energy = result["free_energy"]
        branch.verification = result["verification"]
        branch.ok = True


def _short(obj: Any) -> str:
    s = repr(obj)
    return s if len(s) <= 80 else s[:77] + "..."


def _round(x: float) -> Any:
    if x in (float("inf"), float("-inf")):
        return str(x)
    return round(x, 5)
