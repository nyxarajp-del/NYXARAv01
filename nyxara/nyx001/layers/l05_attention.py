"""NYXARA · nyx001/layers/l05_attention.py — Layer 5, dynamic attention (🎯, A = f(Q,K,V,G,U,M,E,T,R)).

The charter asks for "resource allocation based on multiple factors, not just fixed keys/queries",
and writes the signature out: ``A = f(Q, K, V, G, U, M, E, T, R)`` — the usual query/key/value plus
**G**oal, **U**ncertainty, **M**emory, **E**rror, **T**ime and **R**esource.

Standard attention scores a candidate by ``q·k`` alone. Here the score is:

    score = (q·k)                      how well it matches what is being asked
          + w_g · goal_alignment       does it serve the goal in play
          + w_u · uncertainty          unresolved things deserve a look
          + w_m · memory_relevance     did it matter before
          + w_e · prediction_error     the world model is wrong here
          - w_t · time_cost            attending costs time
          - w_r · resource_cost        …and compute

then softmax over the candidates, with a **temperature driven by the resource budget**: a starved
budget sharpens the distribution (commit to the top candidate), an ample one flattens it (spread
the look). That is the part that makes this resource allocation rather than scoring — under
pressure the system narrows, which is what an attention mechanism is *for*.

Two honest notes. The additive-bias form means the six extra factors shift a ranking that ``q·k``
still dominates by default; the weights are hyperparameters, not learned, and :meth:`explain`
returns each factor's contribution to the winner so a mis-set weight is visible as a number.
And this is a *selection* mechanism over candidates that something else generated — it does not
create options, it chooses among them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["Candidate", "Allocation", "DynamicAttention"]


@dataclass
class Candidate:
    """Something that could be attended to, with whatever signals are known about it."""

    key: str = ""
    vec: Any = None                     # K — matched against the query
    payload: Any = None                 # V — what is actually returned
    goal_alignment: float = 0.0         # G
    uncertainty: float = 0.0            # U
    memory_relevance: float = 0.0       # M
    prediction_error: float = 0.0       # E
    time_cost: float = 0.0              # T
    resource_cost: float = 0.0          # R

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "G": round(self.goal_alignment, 4),
                "U": round(self.uncertainty, 4), "M": round(self.memory_relevance, 4),
                "E": round(self.prediction_error, 4), "T": round(self.time_cost, 4),
                "R": round(self.resource_cost, 4)}


@dataclass
class Allocation:
    """Where attention went, and why."""

    weights: List[float] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)
    winner: Optional[Candidate] = None
    temperature: float = 1.0
    entropy: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)

    @property
    def focused(self) -> bool:
        """Did attention actually commit, or is it spread thin? Entropy-based, not a guess."""
        return self.entropy < 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {"winner": self.winner.key if self.winner else None,
                "temperature": round(self.temperature, 4), "entropy": round(self.entropy, 4),
                "focused": self.focused, "factors": dict(self.factors),
                "top": [(k, round(w, 4)) for k, w in
                        sorted(zip(self.keys, self.weights), key=lambda t: t[1],
                               reverse=True)[:5]]}


class DynamicAttention:
    """Layer 5. Allocates a bounded look across candidates using all nine factors."""

    def __init__(self, substrate: Any, *, w_goal: float = 0.6, w_uncertainty: float = 0.3,
                 w_memory: float = 0.4, w_error: float = 0.5, w_time: float = 0.2,
                 w_resource: float = 0.2, base_temperature: float = 1.0) -> None:
        self.substrate = substrate
        self.w_goal = float(w_goal)
        self.w_uncertainty = float(w_uncertainty)
        self.w_memory = float(w_memory)
        self.w_error = float(w_error)
        self.w_time = float(w_time)
        self.w_resource = float(w_resource)
        self.base_temperature = float(base_temperature)
        self.allocations = 0

    def _match(self, query: Any, cand: Candidate) -> float:
        """``q·k``, normalised. Zero when either side is missing — absent, not negative."""
        if np is None or query is None or cand.vec is None:
            return 0.0
        try:
            q = np.asarray(query, dtype=np.float64).ravel()
            k = np.asarray(cand.vec, dtype=np.float64).ravel()
            nq, nk = float(np.linalg.norm(q)), float(np.linalg.norm(k))
            if nq <= 0.0 or nk <= 0.0:
                return 0.0
            n = min(q.size, k.size)
            return float(np.dot(q[:n], k[:n]) / (nq * nk + 1e-12))
        except Exception:  # noqa: BLE001
            return 0.0

    def allocate(self, query: Any, candidates: Sequence[Candidate], *,
                 resource_budget: float = 1.0) -> Allocation:
        """Score, temper by budget, softmax. Never raises; an empty candidate set allocates nothing.

        ``resource_budget`` in ``[0, 1]``: 1.0 is ample (flat, exploratory), 0.0 is starved
        (sharp, committed). The mapping is ``T = base · (0.25 + 0.75·budget)``.
        """
        out = Allocation(temperature=self.base_temperature)
        cands = [c for c in candidates if c is not None]
        if not cands or np is None:
            return out
        try:
            budget = float(min(1.0, max(0.0, resource_budget)))
            temp = max(1e-3, self.base_temperature * (0.25 + 0.75 * budget))
            scores: List[float] = []
            contrib: List[Dict[str, float]] = []
            for c in cands:
                m = self._match(query, c)
                parts = {
                    "match": m,
                    "G": self.w_goal * c.goal_alignment,
                    "U": self.w_uncertainty * c.uncertainty,
                    "M": self.w_memory * c.memory_relevance,
                    "E": self.w_error * c.prediction_error,
                    "T": -self.w_time * c.time_cost,
                    "R": -self.w_resource * c.resource_cost,
                }
                contrib.append(parts)
                scores.append(float(sum(parts.values())))
            arr = np.asarray(scores, dtype=np.float64) / temp
            arr = arr - arr.max()
            e = np.exp(arr)
            w = e / max(1e-12, e.sum())
            out.weights = [float(x) for x in w]
            out.keys = [c.key for c in cands]
            best = int(np.argmax(w))
            out.winner = cands[best]
            out.temperature = temp
            out.factors = {k: round(v, 4) for k, v in contrib[best].items()}
            p = w[w > 0]
            out.entropy = float(-(p * np.log(p)).sum() / max(1e-12, np.log(len(cands)))) \
                if len(cands) > 1 else 0.0
            self.allocations += 1
            return out
        except Exception:  # noqa: BLE001
            return out

    def explain(self, alloc: Allocation) -> str:
        """One honest line naming what actually drove the choice."""
        if alloc.winner is None:
            return "attended to nothing — no candidates"
        parts = sorted(alloc.factors.items(), key=lambda kv: abs(kv[1]), reverse=True)
        lead = ", ".join(f"{k}={v:+.3f}" for k, v in parts[:3])
        shape = "focused" if alloc.focused else "spread"
        return f"attended to {alloc.winner.key} ({shape}, T={alloc.temperature:.2f}): {lead}"

    def stats(self) -> Dict[str, Any]:
        return {"allocations": self.allocations,
                "weights": {"G": self.w_goal, "U": self.w_uncertainty, "M": self.w_memory,
                            "E": self.w_error, "T": self.w_time, "R": self.w_resource}}

    def to_dict(self) -> Dict[str, Any]:
        return {"allocations": self.allocations}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.allocations = int(d.get("allocations", 0))
        except Exception:  # noqa: BLE001
            pass
