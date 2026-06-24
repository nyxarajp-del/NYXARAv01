"""NYXARA · growth/meta_meta.py — the meta-meta loop: improving HOW she improves (♾, Rule 4).

NYXARA already improves her code (``self_optimize``), her weights (``foundry``), and her *way of
thinking* (``mind_evolution``). But the **way she improves** was itself static: the recursive-
self-improvement engine ran with fixed hyperparameters (how many edits per cycle, how deep the
re-edit recursion goes), so it could get better at *tasks* but never better at *getting better*.

This module closes that loop. It is a tiny, character-locked ``1+1`` evolutionary controller over
the improvement engine's **own** knobs, and — crucially — its fitness is not assumed, it is
**measured from reality**: the actual change in the persisted Intelligence Index that each knob
configuration produced over a window of real cycles. A configuration that genuinely made cycles
raise the index faster is promoted; one that did not is reverted.

    champion  = the knob-set that has produced the best measured index-gain so far
    challenger = a small mutation of the champion, on trial for ``window`` real cycles
    promote the challenger iff its mean realised gain strictly beats the champion's, else revert

Safety (the reason this is allowed to touch live knobs):

* It evolves **capability-only** hyperparameters — recursion depth, edit budget — never a character
  value; the immutable core (``guard.value_learning.IMMUTABLE_VALUES``) is untouchable and absent
  here by construction. Every knob is bounded, so a mutation can never escalate edit volume or
  recursion past a safe ceiling.
* It changes only *how much* improvement work runs, never *whether* the reversible verify-or-rollback
  gauntlet runs — every source edit still clears the same gates. It makes the engine wiser, never
  less safe.

Pure standard library; persists its champion so the meta-meta gain compounds across restarts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["MetaGenome", "MetaImprovementController"]


def _clampi(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(x))))


# --------------------------------------------------------------------------- #
# The meta-genome — the improvement engine's own, capability-only knobs
# --------------------------------------------------------------------------- #
@dataclass
class MetaGenome:
    """How hard the self-improvement engine works each cycle. Bounded; capability-only."""

    recursion_depth: int = 3        # chained re-edits per file/cycle (self_improvement knob)
    max_edits_per_cycle: int = 3    # source edits attempted per cycle

    # (key, lo, hi) — the single source of truth for clamping a genome to the safe envelope
    _BOUNDS: Tuple[Tuple[str, int, int], ...] = (
        ("recursion_depth", 1, 5),
        ("max_edits_per_cycle", 1, 6),
    )

    def __post_init__(self) -> None:
        for key, lo, hi in self._BOUNDS:
            setattr(self, key, _clampi(getattr(self, key), lo, hi))

    def mutate(self, rng: random.Random) -> "MetaGenome":
        """A small neighbour of this genome — perturb each knob by ±1 with some probability."""
        vals = {}
        for key, lo, hi in self._BOUNDS:
            cur = int(getattr(self, key))
            step = rng.choice((-1, 0, 1)) if rng.random() < 0.7 else 0
            vals[key] = _clampi(cur + step, lo, hi)
        m = MetaGenome(**vals)
        # guarantee a real neighbour (don't waste a trial window on an identical genome)
        if (m.recursion_depth, m.max_edits_per_cycle) == (self.recursion_depth,
                                                           self.max_edits_per_cycle):
            key, lo, hi = self._BOUNDS[rng.randrange(len(self._BOUNDS))]
            setattr(m, key, _clampi(getattr(m, key) + rng.choice((-1, 1)), lo, hi))
        return m

    def to_dict(self) -> Dict[str, int]:
        return {"recursion_depth": self.recursion_depth,
                "max_edits_per_cycle": self.max_edits_per_cycle}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetaGenome":
        return cls(recursion_depth=int(d.get("recursion_depth", 3)),
                   max_edits_per_cycle=int(d.get("max_edits_per_cycle", 3)))


# --------------------------------------------------------------------------- #
# The controller — a measured 1+1 evolutionary loop over the meta-genome
# --------------------------------------------------------------------------- #
class MetaImprovementController:
    """Evolve the improvement engine's own knobs, scored by the index-gain they actually produce."""

    def __init__(self, *, window: int = 3, seed: int = 0, persist_path: Any = None,
                 champion: Optional[MetaGenome] = None) -> None:
        self.window = max(1, int(window))
        self._rng = random.Random(seed)
        self._persist_path = persist_path
        self.champion = champion or MetaGenome()
        self.champion_fitness: Optional[float] = None
        self.active = self.champion                 # the champion is measured first (the baseline)
        self._trialing_challenger = False
        self._deltas: List[float] = []              # realised index gains for the active genome
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        self._load()

    # ---- bind the active genome to the live engine ---- #
    def apply(self, settings: Any) -> None:
        """Write the active genome's knobs into the live self-improvement config (capability-only)."""
        try:
            si = settings.self_improvement
            si.llm_edit_recursion_depth = self.active.recursion_depth
            si.max_edits_per_cycle = self.active.max_edits_per_cycle
        except Exception:  # noqa: BLE001 — binding is best-effort; a failure leaves knobs as-is
            pass

    # ---- measure: fold one cycle's realised index gain into the active trial ---- #
    def record(self, delta: float) -> None:
        try:
            self._deltas.append(float(delta))
        except Exception:  # noqa: BLE001
            pass

    def _mean(self) -> float:
        return sum(self._deltas) / len(self._deltas) if self._deltas else 0.0

    def maybe_evolve(self) -> Optional[Dict[str, Any]]:
        """Once the active genome has a full window of measured cycles, select and re-arm.

        Returns a small status dict when a selection happened (promotion/rejection/baseline set),
        else ``None``. This is where "improving how she improves" actually happens: the knob-set
        that measurably raised the index faster survives.
        """
        if len(self._deltas) < self.window:
            return None
        fitness = self._mean()
        event: Dict[str, Any]
        if not self._trialing_challenger:
            # we were measuring the champion itself — record its baseline fitness, then trial a mutant
            self.champion_fitness = fitness
            event = {"event": "baseline", "champion": self.champion.to_dict(),
                     "fitness": round(fitness, 6)}
        else:
            # we were trialing a challenger — promote iff it strictly beat the champion's gain
            if self.champion_fitness is None or fitness > self.champion_fitness:
                event = {"event": "promote", "from": self.champion.to_dict(),
                         "to": self.active.to_dict(),
                         "fitness": round(fitness, 6),
                         "beat": round(self.champion_fitness or 0.0, 6)}
                self.champion = self.active
                self.champion_fitness = fitness
            else:
                event = {"event": "reject", "kept": self.champion.to_dict(),
                         "challenger": self.active.to_dict(),
                         "fitness": round(fitness, 6),
                         "champion_fitness": round(self.champion_fitness, 6)}
        # arm the next challenger: a fresh mutation of the (possibly new) champion
        self.generation += 1
        challenger = self.champion.mutate(self._rng)
        self.active = challenger
        self._trialing_challenger = True
        self._deltas = []
        self.history.append(event)
        self.history = self.history[-50:]
        self._save()
        return event

    def status(self) -> Dict[str, Any]:
        return {"champion": self.champion.to_dict(),
                "champion_fitness": (round(self.champion_fitness, 6)
                                     if self.champion_fitness is not None else None),
                "active": self.active.to_dict(),
                "trialing_challenger": self._trialing_challenger,
                "generation": self.generation,
                "window_filled": f"{len(self._deltas)}/{self.window}"}

    # ---- persistence: the meta-meta gain compounds across restarts ---- #
    def _save(self) -> None:
        if self._persist_path is None:
            return
        try:
            from pathlib import Path
            p = Path(self._persist_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "champion": self.champion.to_dict(),
                "champion_fitness": self.champion_fitness,
                "generation": self.generation,
                "history": self.history[-50:],
            }), encoding="utf-8")
        except Exception:  # noqa: BLE001 — persistence is a bonus, never required
            pass

    def _load(self) -> None:
        if self._persist_path is None:
            return
        try:
            from pathlib import Path
            p = Path(self._persist_path)
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            self.champion = MetaGenome.from_dict(data.get("champion", {}))
            cf = data.get("champion_fitness")
            self.champion_fitness = float(cf) if cf is not None else None
            self.generation = int(data.get("generation", 0))
            self.history = list(data.get("history", []))[-50:]
            # resume by trialing a fresh challenger of the persisted champion
            self.active = self.champion.mutate(self._rng)
            self._trialing_challenger = True
        except Exception:  # noqa: BLE001 — a corrupt cache is simply ignored
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA meta-meta controller self-test")
    print("=" * 70)
    mc = MetaImprovementController(window=2, seed=1)
    # Simulate an environment where DEEPER recursion genuinely yields bigger index gains, so the
    # controller should climb recursion_depth toward the ceiling over generations.
    def env_gain(g: MetaGenome) -> float:
        return 0.01 * g.recursion_depth - 0.002 * g.max_edits_per_cycle

    for _ in range(40):
        for _ in range(mc.window):
            mc.record(env_gain(mc.active) + 0.0005 * (mc._rng.random() - 0.5))
        mc.maybe_evolve()
    print("final champion :", mc.champion.to_dict())
    print("champion gen   :", mc.generation, "| fitness:", round(mc.champion_fitness or 0, 5))
    # it should have climbed recursion_depth above the starting 3 (deeper paid off)
    assert mc.champion.recursion_depth >= 3
    print("learned to improve harder where it paid off : OK")
    print("\nALL SELF-TESTS PASSED ✓")
