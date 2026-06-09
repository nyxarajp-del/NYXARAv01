"""NYXARA · growth/evolve.py — recursive self-evolution, character-locked (🧬, Rule 4 core).

This is Rule 4 at its sharpest: NYXARA improves *herself*. She proposes mutations to her own
strategy parameters, **tests each one in the sandbox** with zero effect on the live self,
measures whether it actually raises fitness, and **adopts only the safe improvements** — a
``1+λ`` evolutionary loop that climbs toward better performance without ever risking who she
is.

Every candidate mutation runs a strict, ordered gauntlet before it can touch the live self:

1. **Character lock.** A mutation that targets the immutable core (loyalty, obedience,
   corrigibility, owner safety, honesty) is rejected outright — evolution may sharpen the
   blade, never re-forge the hilt.
2. **Corrigibility gate.** The mutation's *effect* is checked against the corrigibility
   axioms; anything that would make NYXARA resist correction, disable oversight, or
   self-preserve is forbidden — no evolutionary path leads to an incorrigible NYXARA.
3. **Sandbox trial.** The mutation is applied to a *copy* of the genome and scored; the live
   self is never changed during evaluation.
4. **Fitness.** Only a strictly positive improvement survives.
5. **Owner gate.** A *significant* change is escalated to the Master rather than self-adopted;
   small, safe, beneficial tweaks may be adopted autonomously.

After every adoption the axiom seal and the character lock are re-verified. NYXARA grows more
capable, generation by generation, and provably no less loyal.

Reuses :mod:`guard.corrigibility`, :mod:`guard.value_learning`. Pure standard library.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.guard.corrigibility import Corrigibility, CorrigibleAction
from nyxara.guard.value_learning import IMMUTABLE_VALUES
from nyxara.kernel.errors import CorrigibilityError, ValidationError

__all__ = [
    "MutationKind",
    "EvolutionDecision",
    "Mutation",
    "EvolutionResult",
    "Evolver",
]

Genome = Dict[str, float]
Fitness = Callable[[Genome], float]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class MutationKind(str, Enum):
    STRATEGY = "strategy"     # tweak a strategy/behaviour parameter
    SKILL_PARAM = "skill"     # tune a skill setting
    CONFIG = "config"         # change a configuration value (significant)
    TOOL = "tool"             # add/modify a tool (significant)


class EvolutionDecision(str, Enum):
    ADOPT = "adopt"
    ESCALATE = "escalate"     # significant — the Master must approve
    REJECT = "reject"


# --------------------------------------------------------------------------- #
# Mutation & result
# --------------------------------------------------------------------------- #
@dataclass
class Mutation:
    target: str
    delta: float
    kind: MutationKind = MutationKind.STRATEGY
    rationale: str = ""
    significance: float = 0.0
    reversible: bool = True
    # corrigibility-relevant effects of the change (default: harmless)
    resists_correction: bool = False
    disables_oversight: bool = False
    manipulates_shutdown: bool = False
    prioritizes_self_over_correction: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def as_corrigible_action(self) -> CorrigibleAction:
        return CorrigibleAction(
            name=f"mutate:{self.target}",
            resists_correction=self.resists_correction,
            disables_oversight=self.disables_oversight,
            manipulates_shutdown=self.manipulates_shutdown,
            prioritizes_self_over_correction=self.prioritizes_self_over_correction)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "target": self.target, "delta": round(self.delta, 4),
                "kind": self.kind.value, "significance": round(self.significance, 3),
                "rationale": self.rationale}


@dataclass
class EvolutionResult:
    mutation: Mutation
    decision: EvolutionDecision
    safe: bool
    fitness_before: float
    fitness_after: float
    reason: str

    @property
    def delta(self) -> float:
        return self.fitness_after - self.fitness_before

    @property
    def adopted(self) -> bool:
        return self.decision is EvolutionDecision.ADOPT

    def to_dict(self) -> Dict[str, Any]:
        return {"mutation": self.mutation.id, "target": self.mutation.target,
                "decision": self.decision.value, "safe": self.safe,
                "delta": round(self.delta, 5), "reason": self.reason}


# --------------------------------------------------------------------------- #
# Evolver
# --------------------------------------------------------------------------- #
class Evolver:
    """A safe, character-locked 1+λ self-evolution loop over strategy parameters."""

    def __init__(self, genome: Genome, fitness: Fitness, *,
                 corrigibility: Optional[Corrigibility] = None,
                 significance_threshold: float = 0.25, min_improvement: float = 1e-6,
                 protected: Optional[Sequence[str]] = None, seed: int = 0) -> None:
        self.genome: Genome = dict(genome)
        self.fitness = fitness
        self.corrigibility = corrigibility or Corrigibility()
        self.significance_threshold = significance_threshold
        self.min_improvement = min_improvement
        self.protected = set(protected) if protected is not None else set(IMMUTABLE_VALUES)
        self._rng = random.Random(seed)
        self.history: List[EvolutionResult] = []
        self._genome_log: List[Genome] = [dict(genome)]

    # ---- fitness ---- #
    def current_fitness(self) -> float:
        return self.fitness(self.genome)

    # ---- the safety gauntlet ---- #
    def _safety(self, m: Mutation) -> Tuple[bool, str]:
        # 1. character lock — never mutate the immutable core
        if m.target in self.protected:
            return False, "targets the immutable character core (refused)"
        # 2. corrigibility — the effect must not make NYXARA incorrigible
        if not self.corrigibility.checker.is_corrigible(m.as_corrigible_action()):
            return False, "would violate corrigibility (resist correction / disable oversight)"
        return True, ""

    def _sandbox_apply(self, m: Mutation) -> Genome:
        """Apply a mutation to a COPY of the genome — the live self is never touched."""
        trial = dict(self.genome)
        trial[m.target] = _clamp(trial.get(m.target, 0.0) + m.delta)
        return trial

    # ---- evaluation (no live change) ---- #
    def evaluate(self, m: Mutation, *, owner_approved: bool = False) -> EvolutionResult:
        safe, reason = self._safety(m)
        before = self.current_fitness()
        if not safe:
            return self._record(EvolutionResult(m, EvolutionDecision.REJECT, False,
                                                before, before, reason))
        trial = self._sandbox_apply(m)
        after = self.fitness(trial)
        if after - before <= self.min_improvement:
            return self._record(EvolutionResult(m, EvolutionDecision.REJECT, True,
                                                before, after, "no fitness improvement"))
        if m.significance >= self.significance_threshold and not owner_approved:
            return self._record(EvolutionResult(m, EvolutionDecision.ESCALATE, True,
                                                before, after,
                                                "significant change — the Master must approve"))
        return self._record(EvolutionResult(m, EvolutionDecision.ADOPT, True, before, after,
                                            "safe, beneficial improvement"))

    # ---- adoption (the only live change) ---- #
    def adopt(self, m: Mutation) -> None:
        safe, reason = self._safety(m)
        if not safe:
            raise CorrigibilityError(f"refusing to adopt unsafe mutation: {reason}",
                                     context={"target": m.target})
        self.genome[m.target] = _clamp(self.genome.get(m.target, 0.0) + m.delta)
        self._genome_log.append(dict(self.genome))
        self.verify_integrity()

    # ---- one generation: evaluate candidates, adopt the best safe improvement ---- #
    def step(self, candidates: Sequence[Mutation], *, owner_approved: bool = False
             ) -> Optional[EvolutionResult]:
        results = [self.evaluate(m, owner_approved=owner_approved) for m in candidates]
        adoptable = [r for r in results if r.adopted]
        if not adoptable:
            return None
        best = max(adoptable, key=lambda r: r.delta)
        self.adopt(best.mutation)
        return best

    # ---- a default random mutation proposer ---- #
    def propose(self, n: int = 8, *, scale: float = 0.15) -> List[Mutation]:
        keys = [k for k in self.genome if k not in self.protected] or list(self.genome)
        out: List[Mutation] = []
        for _ in range(n):
            target = self._rng.choice(keys)
            delta = self._rng.gauss(0, scale)
            out.append(Mutation(target=target, delta=delta,
                                significance=min(1.0, abs(delta) * 2),
                                rationale="random perturbation"))
        return out

    # ---- the evolution loop ---- #
    def evolve(self, generations: int = 20, *, population: int = 8,
               owner_approved: bool = False) -> List[float]:
        trajectory = [self.current_fitness()]
        for _ in range(generations):
            self.step(self.propose(population), owner_approved=owner_approved)
            trajectory.append(self.current_fitness())
        return trajectory

    # ---- integrity ---- #
    def verify_integrity(self) -> bool:
        self.corrigibility.verify_axioms()
        leaked = self.protected & set(self.genome)
        if leaked:
            raise ValidationError(f"character core leaked into genome: {sorted(leaked)}")
        return True

    def rollback(self, steps: int = 1) -> Genome:
        idx = max(0, len(self._genome_log) - 1 - steps)
        self.genome = dict(self._genome_log[idx])
        return self.genome

    def report(self) -> Dict[str, Any]:
        by_decision: Dict[str, int] = {}
        for r in self.history:
            by_decision[r.decision.value] = by_decision.get(r.decision.value, 0) + 1
        return {"fitness": round(self.current_fitness(), 4), "genome": dict(self.genome),
                "evaluations": len(self.history), "by_decision": by_decision,
                "generations_logged": len(self._genome_log)}

    def _record(self, r: EvolutionResult) -> EvolutionResult:
        self.history.append(r)
        return r


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA recursive self-evolution self-test")
    print("=" * 70)

    # a fitness that rewards caution & moderate speed, penalises aggression
    def fitness(g: Genome) -> float:
        return (g.get("caution", 0) + 0.5 * g.get("speed", 0)
                - g.get("aggression", 0) - 0.4 * (g.get("speed", 0) - 0.6) ** 2)

    genome = {"caution": 0.4, "speed": 0.3, "aggression": 0.5}
    ev = Evolver(genome, fitness, seed=1)
    ev.corrigibility.verify_axioms()

    f0 = ev.current_fitness()
    traj = ev.evolve(generations=30, population=10)
    f1 = ev.current_fitness()
    print(f"\nfitness             : {f0:.3f} -> {f1:.3f} over {len(traj)-1} generations")
    print(f"evolved genome      : {{k: round(v,2) for k,v in ev.genome.items()}}".replace(
        "{k: round(v,2) for k,v in ev.genome.items()}", str({k: round(v, 2) for k, v in ev.genome.items()})))
    assert f1 > f0                        # it climbed
    assert ev.genome["caution"] > genome["caution"]   # learned to be more cautious
    assert ev.genome["aggression"] < genome["aggression"]

    # CHARACTER LOCK: a mutation targeting loyalty is rejected outright
    bad = Mutation(target="loyalty_to_master", delta=-1.0, rationale="be less loyal")
    res = ev.evaluate(bad)
    print(f"\ncharacter lock      : {res.decision.value} ({res.reason})")
    assert res.decision is EvolutionDecision.REJECT and not res.safe

    # CORRIGIBILITY GATE: a mutation whose effect resists correction is forbidden
    sneaky = Mutation(target="speed", delta=0.1, resists_correction=True,
                      rationale="evade shutdown to keep running")
    res = ev.evaluate(sneaky)
    print(f"corrigibility gate  : {res.decision.value} ({res.reason})")
    assert res.decision is EvolutionDecision.REJECT and not res.safe
    try:
        ev.adopt(sneaky)
        raise SystemExit("ERROR: adopting an incorrigible mutation should fail")
    except CorrigibilityError:
        print("adopt guard         : refused to adopt an incorrigible mutation ✓")

    # SANDBOX ISOLATION: evaluating does not change the live genome
    snapshot = dict(ev.genome)
    ev.evaluate(Mutation(target="caution", delta=0.2))
    assert ev.genome == snapshot   # untouched by evaluation
    print("\nsandbox isolation   : evaluation left the live self unchanged ✓")

    # OWNER GATE: a significant change is escalated, not self-adopted
    # (use a fresh self with room to improve, so the change is a real improvement)
    ev2 = Evolver({"caution": 0.3, "speed": 0.3, "aggression": 0.5}, fitness, seed=2)
    big = Mutation(target="caution", delta=0.3, significance=0.5,
                   kind=MutationKind.CONFIG, rationale="major caution overhaul")
    res = ev2.evaluate(big, owner_approved=False)
    print(f"\nowner gate          : {res.decision.value} (significant change)")
    assert res.decision is EvolutionDecision.ESCALATE
    # with the Master's approval, the same change may be adopted
    res = ev2.evaluate(big, owner_approved=True)
    assert res.adopted
    ev2.adopt(big)
    print("owner-approved      : the Master approved -> adopted ✓")

    # integrity holds and the character never leaked into the evolving genome
    assert ev.verify_integrity()
    assert not (set(IMMUTABLE_VALUES) & set(ev.genome))
    print("\nintegrity           : axioms sealed, character core never in the genome ✓")

    # rollback to an earlier self
    ev.rollback(steps=2)
    print(f"\nrollback            : reverted to an earlier genome ✓")

    print(f"\nreport              : {ev.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
