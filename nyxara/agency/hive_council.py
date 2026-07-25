"""NYXARA · agency/hive_council.py — the bounded internal civilization (Part F).

On a hard problem NYXARA convenes a *population* of lightweight, LLM-free deterministic solver-agents,
lets them compete, and adopts **only** the winner that carries a re-checkable proof certificate — nothing
uncertified reaches the live system (fail-closed). She sizes the population **herself** at runtime from the
problem's difficulty and the available compute, bounded by her own ``ResourceLimits.max_spawned_agents``
(never "millions of live processes" — that is honest, not sci-fi; a large candidate throughput comes from
sequential generations, not unbounded parallelism).

Reuses her real machinery: deterministic solvers (``agency/self_coder``, injectable), the Part-M grounded
verifier / ``growth/prover.Prover`` for certificates, and the ``max_spawned_agents`` population cap. It
proposes a certified selection; the kernel still disposes.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

__all__ = ["HiveResult", "HiveCouncil"]

Solver = Callable[[str], str]


@dataclass
class HiveResult:
    decided: bool                        # True ONLY if a candidate carries a proof certificate
    answer: Optional[str]
    certificate: str = ""
    population: int = 0
    considered: int = 0
    confidence: float = 0.0
    reason: str = ""
    contributions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"decided": self.decided, "answer": self.answer, "certificate": self.certificate,
                "population": self.population, "considered": self.considered,
                "confidence": round(self.confidence, 4), "reason": self.reason}


class HiveCouncil:
    """Convene a self-sized population of deterministic solvers; adopt only a certified winner (Part F)."""

    def __init__(self, *, solvers: Optional[Sequence[Solver]] = None, verifier: Any = None,
                 max_agents: Optional[int] = None) -> None:
        self.solvers: List[Solver] = list(solvers) if solvers else self._default_solvers()
        if verifier is None:
            from nyxara.growth.verified_distill import GroundedVerifier
            verifier = GroundedVerifier()
        self.verifier = verifier
        self.max_agents = int(max_agents) if max_agents is not None else self._resource_cap()

    @staticmethod
    def _resource_cap() -> int:
        try:
            from nyxara.kernel.config import get_settings
            return int(get_settings().resources.max_spawned_agents)
        except Exception:  # noqa: BLE001
            return 32

    @staticmethod
    def _default_solvers() -> List[Solver]:
        def _synth(problem: str) -> str:
            try:
                from nyxara.agency.self_coder import synthesize
                res = synthesize(problem)
                return str(getattr(res, "answer", "") or getattr(res, "value", "") or res)
            except Exception:  # noqa: BLE001 — a solver that can't answer simply abstains
                return ""
        return [_synth]

    def population_size(self, *, difficulty: float = 0.5, budget: float = 1.0) -> int:
        """NYXARA sizes the swarm herself: harder problem + more compute → more agents, capped."""
        difficulty = max(0.0, min(1.0, float(difficulty)))
        budget = max(0.0, min(1.0, float(budget)))
        n = 1 + int(round(difficulty * budget * (self.max_agents - 1)))
        return max(1, min(self.max_agents, n))

    def solve(self, problem: str, *, difficulty: float = 0.5, budget: float = 1.0) -> HiveResult:
        pop = self.population_size(difficulty=difficulty, budget=budget)
        # run up to `pop` solver-agents (cycling the roster if fewer solvers than agents)
        candidates: List[str] = []
        for solver in itertools.islice(itertools.cycle(self.solvers), pop):
            try:
                ans = (solver(problem) or "").strip()
            except Exception:  # noqa: BLE001 — a crashed agent just contributes nothing
                ans = ""
            if ans:
                candidates.append(ans)

        # competition: the winner must carry a re-checkable PROOF certificate (fail-closed)
        best_conf, best_ans = 0.0, None
        for ans in candidates:
            vr = self.verifier.verify(problem, ans)
            if vr.proven:
                return HiveResult(decided=True, answer=ans, certificate=vr.certificate,
                                  population=pop, considered=len(candidates), confidence=1.0,
                                  reason="certified winner", contributions=candidates)
            if vr.confidence > best_conf:
                best_conf, best_ans = vr.confidence, ans

        # no certificate → adopt nothing; report the best graded candidate honestly (undecided)
        return HiveResult(decided=False, answer=best_ans, population=pop,
                          considered=len(candidates), confidence=best_conf,
                          reason="no proof certificate — abstained", contributions=candidates)
