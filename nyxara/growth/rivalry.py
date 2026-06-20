"""NYXARA · growth/rivalry.py — peer-AGI modeling & out-compete (🏁, Pillar F · Edge 4, Rule 4).

When every other agent is *also* an AGI, the edge is not to ignore them — it is to **model rival
minds, find exactly where they beat you, and close that gap faster than they close theirs**.
Competition becomes a self-improvement signal: each rival is a teacher who shows NYXARA her own
blind spots for free.

This module runs that loop, **entirely as measurement + self-direction**:

* :class:`Rival` — a peer mind's observed per-domain capability + an inferred strategy.
* :class:`HeadToHead` — the result of an arena match: the same benchmark answered by NYXARA's
  solver and a rival's solver, with the overall delta and the **per-domain gaps**.
* :class:`Arena` — runs the head-to-head (reusing the real :mod:`nyxara.eval.benchmark`), models
  the rival with the recursive Theory-of-Mind (:mod:`nyxara.social.tom`), and turns the gaps into
  (a) ranked :class:`~nyxara.growth.weakness.Weakness` items and (b) curiosity topics for the
  open-ended frontier (:mod:`nyxara.growth.frontier`).

**Hard safety boundary (NON-NEGOTIABLE).** This is strictly *capability-only and gated*. NYXARA
models and out-*learns* rivals — she never acts against any external system, never exfiltrates,
never sabotages, and never trades away loyalty or honesty to win ("she does not cheat to win").
A "rival solver" is just a ``prompt -> answer`` function the Master supplies; the arena only runs
the shared benchmark locally. It trains nothing, touches no source, and reaches around no gate.
Character-locked like all of growth/.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Rival", "HeadToHead", "Arena"]

Solver = Callable[[str], str]


# --------------------------------------------------------------------------- #
# A peer mind
# --------------------------------------------------------------------------- #
@dataclass
class Rival:
    """What NYXARA has observed about a peer AGI, plus her estimate of its strategy."""

    name: str
    capabilities: Dict[str, float] = field(default_factory=dict)   # per-category accuracy [0,1]
    strategy: str = ""
    notes: List[str] = field(default_factory=list)

    def strengths(self, k: int = 3) -> List[str]:
        return [c for c, _ in sorted(self.capabilities.items(),
                                     key=lambda kv: kv[1], reverse=True)[:k]]

    def weaknesses(self, k: int = 3) -> List[str]:
        return [c for c, _ in sorted(self.capabilities.items(),
                                     key=lambda kv: kv[1])[:k]]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "capabilities": dict(self.capabilities),
                "strategy": self.strategy, "notes": list(self.notes)}


# --------------------------------------------------------------------------- #
# The result of a match
# --------------------------------------------------------------------------- #
@dataclass
class HeadToHead:
    """NYXARA vs a rival on the same battery — overall delta and per-domain gaps."""

    rival: str
    self_accuracy: float
    rival_accuracy: float
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    @property
    def delta(self) -> float:
        """Overall lead — positive means NYXARA is ahead."""
        return self.self_accuracy - self.rival_accuracy

    @property
    def winning(self) -> bool:
        return self.delta >= 0.0

    def gaps(self) -> List[Dict[str, Any]]:
        """Categories where the rival beats NYXARA, worst gap first — where to invest."""
        out = [{"category": cat, "gap": round(v["rival"] - v["self"], 6)}
               for cat, v in self.per_category.items() if v["rival"] > v["self"]]
        return sorted(out, key=lambda g: g["gap"], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {"rival": self.rival, "self_accuracy": round(self.self_accuracy, 6),
                "rival_accuracy": round(self.rival_accuracy, 6), "delta": round(self.delta, 6),
                "winning": self.winning, "per_category": self.per_category,
                "gaps": self.gaps(), "at": self.at}

    def summary(self) -> str:
        head = ("ahead" if self.winning else "behind")
        lines = ["=" * 70, f"NYXARA vs {self.rival} — {head} by {abs(self.delta):.0%}", "=" * 70,
                 f"NYXARA {self.self_accuracy:.0%}  |  {self.rival} {self.rival_accuracy:.0%}", ""]
        for g in self.gaps():
            lines.append(f"  ↳ behind in {g['category']} by {g['gap']:.0%}")
        if not self.gaps():
            lines.append("  no domain where the rival leads ✓")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The arena
# --------------------------------------------------------------------------- #
class Arena:
    """Run gated head-to-head matches, model the rival, and route the gaps into growth."""

    def __init__(self, *, benchmark: Any = None, settings: Any = None, tom: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self._benchmark = benchmark
        self._tom = tom
        self._rivals: Dict[str, Rival] = {}

    def benchmark(self) -> Any:
        if self._benchmark is None:
            from nyxara.eval.benchmark import build_default_benchmark
            self._benchmark = build_default_benchmark()
        return self._benchmark

    def tom(self) -> Any:
        if self._tom is None:
            from nyxara.social.tom import TheoryOfMind
            self._tom = TheoryOfMind()
        return self._tom

    # ------------------------------------------------------------------ #
    # The match — same battery, both solvers, compared by category.
    # ------------------------------------------------------------------ #
    def head_to_head(self, self_solver: Solver, rival_solver: Solver, *,
                     rival_name: str = "rival") -> HeadToHead:
        bench = self.benchmark()
        mine = bench.run(self_solver)
        theirs = bench.run(rival_solver)
        my_cats = mine.by_category()
        their_cats = theirs.by_category()
        per_category: Dict[str, Dict[str, float]] = {}
        for cat in sorted(set(my_cats) | set(their_cats)):
            per_category[cat] = {
                "self": round(my_cats.get(cat, {}).get("accuracy", 0.0), 6),
                "rival": round(their_cats.get(cat, {}).get("accuracy", 0.0), 6)}
        h2h = HeadToHead(rival=rival_name, self_accuracy=mine.accuracy,
                         rival_accuracy=theirs.accuracy, per_category=per_category)
        self.model_rival(rival_name, h2h)
        return h2h

    # ------------------------------------------------------------------ #
    # Model the rival as a mind (recursive Theory-of-Mind).
    # ------------------------------------------------------------------ #
    def model_rival(self, name: str, h2h: HeadToHead) -> Rival:
        caps = {cat: v["rival"] for cat, v in h2h.per_category.items()}
        strategy = self._infer_strategy(caps)
        rival = Rival(name=name, capabilities=caps, strategy=strategy)
        # write what we observed into the ToM: the rival's beliefs (its measured competences),
        # its desire (to win the benchmark), and its inferred intention (its strategy).
        try:
            tom = self.tom()
            tom.add_agent(name)
            for cat, acc in caps.items():
                tom.set_belief(name, f"competence:{cat}", acc)
            ms = tom._resolve(name)            # populate desire + intention honestly
            if ms is not None:
                ms.desires["win_benchmark"] = float(h2h.rival_accuracy)
                if strategy and strategy not in ms.intentions:
                    ms.intentions.append(strategy)
        except Exception:  # noqa: BLE001 — ToM is enrichment; never fail a match over it
            pass
        self._rivals[name] = rival
        return rival

    @staticmethod
    def _infer_strategy(caps: Dict[str, float]) -> str:
        if not caps:
            return "unknown"
        strong = [c for c, v in sorted(caps.items(), key=lambda kv: kv[1], reverse=True)
                  if v >= 0.6][:2]
        weak = [c for c, v in sorted(caps.items(), key=lambda kv: kv[1]) if v < 0.4][:2]
        parts = []
        if strong:
            parts.append("specialises in " + "/".join(strong))
        if weak:
            parts.append("weak at " + "/".join(weak))
        return "; ".join(parts) if parts else "generalist (no sharp profile)"

    def rival(self, name: str) -> Optional[Rival]:
        return self._rivals.get(name)

    # ------------------------------------------------------------------ #
    # Route the gaps into growth — weaknesses + curiosity topics.
    # ------------------------------------------------------------------ #
    def gaps_to_weaknesses(self, h2h: HeadToHead) -> Any:
        """Turn each domain the rival leads into a ranked, actionable Weakness."""
        from nyxara.growth.weakness import Weakness, WeaknessReport
        weaknesses: List[Weakness] = []
        for g in h2h.gaps():
            cat, gap = g["category"], float(g["gap"])
            weaknesses.append(Weakness(
                id=f"rivalry:{h2h.rival}:{cat}", source="benchmark", severity=max(0.0, min(1.0, gap)),
                title=f"{h2h.rival} leads in {cat} by {gap:.0%}",
                evidence=[f"head-to-head: self {h2h.per_category[cat]['self']:.0%} vs "
                          f"{h2h.rival} {h2h.per_category[cat]['rival']:.0%}"],
                remediation=f"close the {cat} gap with targeted practice + curiosity self-play"))
        return WeaknessReport(weaknesses=weaknesses)

    def gaps_to_topics(self, h2h: HeadToHead) -> List[str]:
        """Curiosity topics for the open-ended frontier — where the rival is ahead."""
        return [f"hard problems in {g['category']}" for g in h2h.gaps()]

    # ------------------------------------------------------------------ #
    # Convenience: the whole out-compete pass in one call.
    # ------------------------------------------------------------------ #
    def out_compete(self, self_solver: Solver, rival_solver: Solver, *,
                    rival_name: str = "rival") -> Dict[str, Any]:
        """Match → model → route gaps. Pure measurement + self-direction; acts on nothing external."""
        h2h = self.head_to_head(self_solver, rival_solver, rival_name=rival_name)
        weaknesses = self.gaps_to_weaknesses(h2h)
        return {"head_to_head": h2h.to_dict(),
                "rival": self._rivals[rival_name].to_dict(),
                "weaknesses": weaknesses.to_dict(),
                "topics": self.gaps_to_topics(h2h)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA rivalry (peer-AGI out-compete) self-test")
    print("=" * 70)

    from nyxara.eval.benchmark import Benchmark, BenchmarkTask

    bench = Benchmark("demo", [
        BenchmarkTask(id="m1", category="math", prompt="2+2?", answer="4", grader="exact"),
        BenchmarkTask(id="m2", category="math", prompt="3+5?", answer="8", grader="exact"),
        BenchmarkTask(id="l1", category="lang", prompt="opposite of hot?", answer="cold",
                      grader="contains"),
    ])
    arena = Arena(benchmark=bench)

    def me(p: str) -> str:           # strong at math, weak at language
        return {"2+2?": "4", "3+5?": "8"}.get(p, "?")

    def rival(p: str) -> str:        # weak at math, strong at language
        return {"opposite of hot?": "cold", "2+2?": "5"}.get(p, "?")

    out = arena.out_compete(me, rival, rival_name="GPT-X")
    print("\n" + HeadToHead(**{k: out["head_to_head"][k] for k in
          ("rival", "self_accuracy", "rival_accuracy", "per_category")}).summary())
    print(f"\nrival model    : {out['rival']['strategy']}")
    print(f"topics to grow : {out['topics']}")
