"""NYXARA · mind/atelier.py — the Multi-Agent Internal Ecosystem (P24 · F19, organ 10).

Not one pipeline but a **colony**: four pure-algorithmic creative personalities
compete inside her atelier, each with its own search bias and its own scoring lens:

* **The Logician**   — symmetry, consistency, causal viability; low mutation, ordered forms.
* **The Disruptor**  — chaos: high mutation, rewards entropy and raw novelty.
* **The Minimalist** — maximum expression per token; brevity and quality per length.
* **The Archivist**  — deep history: mines the concept graph for cross-domain mashups.

Each persona proposes candidates (the engine generates them *under that persona's
bias*), then all four **negotiate**: every persona ranks every candidate through its
own lens, and an influence-weighted Borda count picks the master output. The winning
persona gains influence (bounded), the others decay toward neutral — a real, evolving
internal economy whose state persists.

**No LLM anywhere in this module.** Pure standard library, deterministic given the
candidates it is asked to judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Persona",
    "Atelier",
]

#: metric names a candidate exposes to the lenses (all in [0, 1], higher = more of it)
METRICS: Tuple[str, ...] = ("novelty", "quality", "aesthetic", "entropy",
                            "brevity", "causal", "cross_domain", "symmetry")

_INFLUENCE_MIN, _INFLUENCE_MAX = 0.5, 2.0
_WIN_GAIN = 0.08          # winner's influence bump per negotiation
_DECAY = 0.97             # losers drift back toward neutral 1.0


@dataclass
class Persona:
    """One algorithmic personality: a search bias + a scoring lens + earned influence."""

    name: str
    bias: Dict[str, Any]                    # knobs the engine applies when generating
    lens: Dict[str, float]                  # metric -> weight (this persona's taste)
    influence: float = 1.0
    wins: int = 0
    proposals: int = 0

    def score(self, metrics: Dict[str, float]) -> float:
        """This persona's opinion of a candidate, through its own lens."""
        num = sum(w * float(metrics.get(m, 0.0)) for m, w in self.lens.items())
        den = sum(abs(w) for w in self.lens.values()) or 1.0
        return num / den

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "influence": round(self.influence, 4),
                "wins": self.wins, "proposals": self.proposals}


def _default_personas() -> List[Persona]:
    return [
        Persona(
            name="logician",
            bias={"mutation_rate": 0.08, "fresh_fraction": 0.15,
                  "prefer_generators": ("radial", "lissajous"),
                  "prefer_forms": ("quatrain",), "chain_length": 2},
            lens={"symmetry": 2.0, "causal": 2.0, "quality": 1.5,
                  "novelty": 0.5, "entropy": -0.5}),
        Persona(
            name="disruptor",
            bias={"mutation_rate": 0.45, "fresh_fraction": 0.6,
                  "prefer_generators": ("automaton", "lsystem"),
                  "prefer_forms": ("free",), "chain_length": 4},
            lens={"entropy": 2.0, "novelty": 2.0, "aesthetic": 0.5,
                  "quality": 0.5, "brevity": -0.25}),
        Persona(
            name="minimalist",
            bias={"mutation_rate": 0.12, "fresh_fraction": 0.25,
                  "prefer_generators": ("lissajous",),
                  "prefer_forms": ("haiku",), "chain_length": 1},
            lens={"brevity": 2.0, "quality": 1.5, "aesthetic": 1.0,
                  "entropy": -0.5}),
        Persona(
            name="archivist",
            bias={"mutation_rate": 0.18, "fresh_fraction": 0.2,
                  "prefer_generators": ("spirograph", "automaton"),
                  "prefer_forms": ("quatrain", "free"), "chain_length": 3,
                  "use_cross_domain": True, "use_transmutation": True},
            lens={"cross_domain": 2.0, "novelty": 1.25, "aesthetic": 1.0,
                  "quality": 1.0}),
    ]


class Atelier:
    """The negotiation table where the four personas argue a master output into being."""

    def __init__(self, personas: Optional[List[Persona]] = None) -> None:
        self.personas: List[Persona] = personas if personas else _default_personas()
        self.negotiations: int = 0

    # ------------------------------------------------------------------ #
    def get(self, name: str) -> Optional[Persona]:
        for p in self.personas:
            if p.name == name:
                return p
        return None

    def biases(self) -> Dict[str, Dict[str, Any]]:
        """Persona name → the search-bias knobs the engine applies for that persona."""
        return {p.name: dict(p.bias) for p in self.personas}

    # ------------------------------------------------------------------ #
    # negotiation — influence-weighted Borda over every persona's ranking
    # ------------------------------------------------------------------ #
    def negotiate(self, candidates: Sequence[Dict[str, Any]]
                  ) -> Optional[Dict[str, Any]]:
        """Pick the master output from ``candidates``.

        Each candidate is ``{"persona": <proposer>, "metrics": {metric: value}}`` plus
        any payload the caller wants back. Every persona ranks ALL candidates through
        its own lens; Borda points are weighted by the ranker's influence; the winner's
        proposer gains influence, everyone else decays toward neutral. Returns the
        winning candidate dict with ``negotiation`` details attached, or None."""
        cands = list(candidates)
        if not cands:
            return None
        for c in cands:
            proposer = self.get(str(c.get("persona", "")))
            if proposer is not None:
                proposer.proposals += 1

        n = len(cands)
        points = [0.0] * n
        ballots: Dict[str, List[int]] = {}
        for judge in self.personas:
            scored = sorted(range(n),
                            key=lambda i: judge.score(cands[i].get("metrics", {})),
                            reverse=True)
            ballots[judge.name] = scored
            for rank, idx in enumerate(scored):
                points[idx] += judge.influence * (n - 1 - rank)

        best = max(range(n), key=lambda i: points[i])
        winner = dict(cands[best])
        winner_name = str(winner.get("persona", ""))

        # the internal economy: the winning voice grows, the others relax toward 1.0
        for p in self.personas:
            if p.name == winner_name:
                p.influence = min(_INFLUENCE_MAX, p.influence + _WIN_GAIN)
                p.wins += 1
            else:
                p.influence = max(_INFLUENCE_MIN,
                                  1.0 + (p.influence - 1.0) * _DECAY)
        self.negotiations += 1

        winner["negotiation"] = {
            "points": [round(x, 3) for x in points],
            "ballots": {k: v for k, v in ballots.items()},
            "winner_persona": winner_name,
            "influence": {p.name: round(p.influence, 3) for p in self.personas}}
        return winner

    def nudge(self, name: str, delta: float) -> None:
        """External corrective pressure (the Ego) on one persona's influence, bounded."""
        p = self.get(name)
        if p is not None:
            p.influence = max(_INFLUENCE_MIN,
                              min(_INFLUENCE_MAX, p.influence + float(delta)))

    # ------------------------------------------------------------------ #
    # persistence & report
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"negotiations": self.negotiations,
                "personas": {p.name: {"influence": round(p.influence, 4),
                                      "wins": p.wins, "proposals": p.proposals}
                             for p in self.personas}}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Atelier":
        at = cls()
        at.negotiations = int(d.get("negotiations", 0) or 0)
        for name, st in (d.get("personas", {}) or {}).items():
            p = at.get(str(name))
            if p is not None:
                p.influence = max(_INFLUENCE_MIN,
                                  min(_INFLUENCE_MAX,
                                      float(st.get("influence", 1.0) or 1.0)))
                p.wins = int(st.get("wins", 0) or 0)
                p.proposals = int(st.get("proposals", 0) or 0)
        return at

    def report(self) -> Dict[str, Any]:
        return {"negotiations": self.negotiations,
                "personas": [p.to_dict() for p in
                             sorted(self.personas, key=lambda p: p.influence,
                                    reverse=True)]}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA atelier self-test")
    print("=" * 70)

    at = Atelier()
    assert [p.name for p in at.personas] == ["logician", "disruptor",
                                             "minimalist", "archivist"]

    # each persona exposes a usable search bias
    biases = at.biases()
    assert biases["disruptor"]["mutation_rate"] > biases["logician"]["mutation_rate"]
    assert biases["archivist"].get("use_cross_domain") is True
    print(f"\nbiases              : mutation logician={biases['logician']['mutation_rate']} "
          f"disruptor={biases['disruptor']['mutation_rate']}")

    # four candidates, one per persona, with distinct metric shapes
    cands = [
        {"persona": "logician", "content": "L",
         "metrics": {"symmetry": 0.9, "causal": 0.9, "quality": 0.8,
                     "novelty": 0.3, "entropy": 0.2, "brevity": 0.5,
                     "aesthetic": 0.6, "cross_domain": 0.1}},
        {"persona": "disruptor", "content": "D",
         "metrics": {"symmetry": 0.1, "causal": 0.2, "quality": 0.4,
                     "novelty": 0.95, "entropy": 0.95, "brevity": 0.2,
                     "aesthetic": 0.5, "cross_domain": 0.4}},
        {"persona": "minimalist", "content": "M",
         "metrics": {"symmetry": 0.5, "causal": 0.4, "quality": 0.7,
                     "novelty": 0.4, "entropy": 0.2, "brevity": 0.95,
                     "aesthetic": 0.7, "cross_domain": 0.2}},
        {"persona": "archivist", "content": "A",
         "metrics": {"symmetry": 0.4, "causal": 0.5, "quality": 0.6,
                     "novelty": 0.7, "entropy": 0.5, "brevity": 0.4,
                     "aesthetic": 0.6, "cross_domain": 0.95}},
    ]

    win = at.negotiate(cands)
    assert win is not None
    print(f"first negotiation   : winner={win['negotiation']['winner_persona']} "
          f"points={win['negotiation']['points']}")
    assert set(win["negotiation"]["ballots"]) == {"logician", "disruptor",
                                                  "minimalist", "archivist"}

    # negotiation is deterministic for the same table state + candidates
    at_a, at_b = Atelier(), Atelier()
    assert (at_a.negotiate(cands)["negotiation"]["winner_persona"]
            == at_b.negotiate(cands)["negotiation"]["winner_persona"])

    # influence shifts toward repeat winners and stays bounded
    winner0 = win["negotiation"]["winner_persona"]
    for _ in range(60):
        at.negotiate(cands)
    wp = at.get(winner0)
    assert wp is not None and wp.influence > 1.0
    assert all(_INFLUENCE_MIN <= p.influence <= _INFLUENCE_MAX for p in at.personas)
    print(f"economy             : {[(p.name, round(p.influence, 2)) for p in at.personas]}")

    # a heavier judge can flip the outcome: crank disruptor influence, feed a tie-ish table
    at2 = Atelier()
    at2.nudge("disruptor", 1.0)
    win2 = at2.negotiate(cands)
    print(f"weighted table      : winner={win2['negotiation']['winner_persona']}")

    # ego-style corrective nudge is bounded
    at2.nudge("disruptor", 99.0)
    assert at2.get("disruptor").influence == _INFLUENCE_MAX

    # every persona actually proposes over time (proposals tracked)
    assert all(p.proposals > 0 for p in at.personas)

    # persistence round-trip
    at3 = Atelier.from_dict(at.to_dict())
    assert at3.negotiations == at.negotiations
    assert at3.get(winner0).wins == at.get(winner0).wins
    print("persistence          : OK")

    print("\nALL SELF-TESTS PASSED ✓")
