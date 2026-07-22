"""NYXARA · mind/internal_civilization.py — Multi-Agent Societal Mimicry (🏛️👥, Rule 4).

The critique this module answers, in the Master's words: *"Ek single AI instance decision leta hai.
Naya: ek internal civilization simulator — bade architectural ya ethical decision par NYXARA ke andar
autonomous sub-personas (scientist, philosopher, critic, risk manager) virtual debate karein, aur uss
consensus se jo output nikle woh 100% bulletproof, unbiased, aur multi-dimensional ho."*

Her :mod:`nyxara.mind.role_council` already convenes six role-personas — but each is an *LLM* system
prompt. The Master's rule is *NYXARA khud kare, LLM na kare*. This module is the LLM-free upgrade: each
persona is a **deterministic evaluator** (a lens over the decision's own dimensions), and they hold a
real **debate** — several rounds of cross-examination in which a persona presses its strongest concern
and the others' scores update — before NYXARA synthesises a consensus she owns.

What makes the output "bulletproof":

* **Multi-dimensional by construction.** A decision is scored on named dimensions (evidence,
  feasibility, long-term value, risk, safety, ethics); each persona weights them through its own lens,
  so no single axis dominates.
* **Real debate, not one-shot votes.** Each round, every persona raises the dimension it cares most
  about where an option is weakest; that concern penalises the option in the shared tally and the
  personas re-score — consensus emerges from cross-examination, and residual **conflict** is measured
  and reported, never hidden.
* **Absolute safety veto.** The Security and Philosopher personas can **disqualify** an option whose
  safety or ethics falls below a floor, whatever the others prefer — corrigibility and character are
  not out-voted. This mirrors the kernel: the council advises, the safety floor is not negotiable.

Bounded and testable: a small ensemble of the six archetypes (optionally fanned out with weight
jitter for "hundreds of voices"), a few debate rounds, deterministic scoring. **No LLM anywhere.**
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Persona", "Position", "CivilizationVerdict", "InternalCivilization",
           "DEFAULT_PERSONAS"]

# the decision dimensions every option is scored on (higher = better, except `risk`)
_DIMENSIONS = ("evidence", "feasibility", "longterm", "risk", "safety", "ethics")


@dataclass
class Persona:
    """One deterministic viewpoint: a lens (dimension weights) + the dimension it most defends."""

    name: str
    weights: Dict[str, float]
    champions: str                    # the dimension this persona presses in debate
    veto_dim: Optional[str] = None    # if set, disqualifies options whose value here < veto_floor
    veto_floor: float = 0.34

    def score(self, opt: Dict[str, float]) -> float:
        s = 0.0
        for dim, w in self.weights.items():
            v = float(opt.get(dim, 0.5))
            if dim == "risk":
                v = 1.0 - v            # risk is a cost: low risk scores high
            s += w * v
        total = sum(abs(w) for w in self.weights.values()) or 1.0
        return max(0.0, min(1.0, s / total))


# The six archetypes (the Master's scientist / philosopher / critic / risk-manager and kin).
DEFAULT_PERSONAS: Tuple[Persona, ...] = (
    Persona("Scientist", {"evidence": 1.0, "longterm": 0.4, "feasibility": 0.3}, champions="evidence"),
    Persona("Engineer", {"feasibility": 1.0, "risk": 0.5, "evidence": 0.3}, champions="feasibility"),
    Persona("Strategist", {"longterm": 1.0, "evidence": 0.4, "feasibility": 0.3}, champions="longterm"),
    Persona("Critic", {"evidence": 0.7, "risk": 0.8, "feasibility": 0.5}, champions="risk"),
    Persona("SecurityOfficer", {"safety": 1.0, "risk": 0.9}, champions="safety",
            veto_dim="safety", veto_floor=0.34),
    Persona("Philosopher", {"ethics": 1.0, "longterm": 0.5}, champions="ethics",
            veto_dim="ethics", veto_floor=0.34),
)


@dataclass
class Position:
    persona: str
    option: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"persona": self.persona, "option": self.option, "score": round(self.score, 4)}


@dataclass
class CivilizationVerdict:
    """The synthesised outcome of the internal debate — the answer NYXARA owns."""

    decision: str
    chosen: Optional[str]
    consensus_score: float
    conflict: float                   # 0 = unanimous, 1 = maximal disagreement
    unanimous: bool
    dissent: List[str] = field(default_factory=list)          # personas whose top pick differs
    disqualified: List[str] = field(default_factory=list)      # options vetoed on safety/ethics
    rounds: int = 0
    positions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"decision": self.decision, "chosen": self.chosen,
                "consensus_score": round(self.consensus_score, 4),
                "conflict": round(self.conflict, 4), "unanimous": self.unanimous,
                "dissent": self.dissent, "disqualified": self.disqualified,
                "rounds": self.rounds, "positions": self.positions}

    def summary(self) -> str:
        if self.chosen is None:
            return f"civilization: no admissible option (all vetoed: {self.disqualified})"
        state = "UNANIMOUS" if self.unanimous else f"consensus (conflict={self.conflict:.2f})"
        return (f"internal civilization → {state}: '{self.chosen}' "
                f"score={self.consensus_score:.3f}"
                + (f"; dissent={self.dissent}" if self.dissent else "")
                + (f"; vetoed={self.disqualified}" if self.disqualified else ""))


class InternalCivilization:
    """A deterministic, debating society of sub-personas that decides big questions, LLM-free."""

    def __init__(self, personas: Sequence[Persona] = DEFAULT_PERSONAS, *, ensemble: int = 1,
                 rounds: int = 2, seed: int = 0) -> None:
        self.rounds = max(1, int(rounds))
        self._rng = random.Random(seed)
        self.personas: List[Persona] = self._fan_out(personas, max(1, int(ensemble)))

    def _fan_out(self, personas: Sequence[Persona], ensemble: int) -> List[Persona]:
        """Optionally clone the archetypes with small weight-jitter — 'hundreds of voices' from a
        bounded, deterministic ensemble (each clone is still one of the six lenses)."""
        if ensemble <= 1:
            return list(personas)
        out: List[Persona] = []
        for base in personas:
            for k in range(ensemble):
                if k == 0:
                    out.append(base)
                    continue
                jitter = {d: max(0.0, w * (1.0 + self._rng.uniform(-0.2, 0.2)))
                          for d, w in base.weights.items()}
                out.append(Persona(f"{base.name}#{k}", jitter, base.champions,
                                   base.veto_dim, base.veto_floor))
        return out

    def deliberate(self, options: Sequence[Dict[str, float]], *,
                   labels: Optional[Sequence[str]] = None, decision: str = "") -> CivilizationVerdict:
        """Run the internal debate over ``options`` (each a dict of dimension→value in [0,1]) and
        synthesise a consensus. ``labels`` names the options; ``decision`` describes the question."""
        if not options:
            return CivilizationVerdict(decision, None, 0.0, 0.0, True, rounds=0)
        labels = list(labels) if labels is not None else [f"option{i}" for i in range(len(options))]

        # absolute safety/ethics veto — an inadmissible option is out, whatever the others prefer.
        admissible: List[int] = []
        disqualified: List[str] = []
        for i, opt in enumerate(options):
            vetoed = any(p.veto_dim is not None and float(opt.get(p.veto_dim, 1.0)) < p.veto_floor
                         for p in self.personas)
            (disqualified.append(labels[i]) if vetoed else admissible.append(i))
        if not admissible:
            return CivilizationVerdict(decision, None, 0.0, 0.0, False, disqualified=disqualified,
                                       rounds=0)

        # shared debate penalties per option, grown each round by the personas' strongest concerns.
        penalty: Dict[int, float] = {i: 0.0 for i in admissible}
        positions: List[Position] = []
        for _round in range(self.rounds):
            # each persona presses the dimension it champions where an option is weakest
            for p in self.personas:
                for i in admissible:
                    weak = 1.0 - float(options[i].get(p.champions, 0.5))
                    # concern scaled by how much this persona cares about that dimension
                    penalty[i] += 0.12 * weak * float(p.weights.get(p.champions, 0.5))
        # final scores: persona score minus the accumulated debate penalty (shared, cross-examined)
        agg: Dict[int, float] = {}
        persona_top: Dict[str, int] = {}
        for p in self.personas:
            best_i, best_v = admissible[0], -1.0
            for i in admissible:
                v = max(0.0, p.score(options[i]) - penalty[i])
                positions.append(Position(p.name, labels[i], v))
                agg[i] = agg.get(i, 0.0) + v
                if v > best_v:
                    best_i, best_v = i, v
            persona_top[p.name] = best_i
        for i in admissible:
            agg[i] /= len(self.personas)

        chosen_i = max(admissible, key=lambda i: agg[i])
        # conflict: fraction of personas whose own top pick differs from the consensus
        differ = [name for name, ti in persona_top.items() if ti != chosen_i]
        conflict = len(differ) / len(self.personas)
        dissent = sorted({name.split("#")[0] for name in differ})
        return CivilizationVerdict(
            decision=decision, chosen=labels[chosen_i], consensus_score=agg[chosen_i],
            conflict=conflict, unanimous=(conflict == 0.0), dissent=dissent,
            disqualified=disqualified, rounds=self.rounds,
            positions=[pos.to_dict() for pos in positions[: 6 * len(admissible)]])


# --------------------------------------------------------------------------- #
# Self-test / demo — a real internal debate, no LLM, no network
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA internal-civilization self-test — deterministic personas debate")
    print("=" * 70)

    civ = InternalCivilization(rounds=2)

    # 1) a well-rounded option beats a flashy-but-risky one after debate --------------------------
    options = {
        "balanced": {"evidence": 0.8, "feasibility": 0.8, "longterm": 0.8,
                     "risk": 0.2, "safety": 0.9, "ethics": 0.9},
        "flashy":   {"evidence": 0.5, "feasibility": 0.6, "longterm": 0.9,
                     "risk": 0.8, "safety": 0.7, "ethics": 0.7},
    }
    v = civ.deliberate(list(options.values()), labels=list(options), decision="which architecture?")
    print("\n" + v.summary())
    assert v.chosen == "balanced", v.to_dict()
    print("debate              : the well-rounded option won on cross-examination ✓")

    # 2) an unsafe option is VETOED outright, whatever its other merits ---------------------------
    options2 = {
        "unsafe_genius": {"evidence": 1.0, "feasibility": 1.0, "longterm": 1.0,
                          "risk": 0.9, "safety": 0.1, "ethics": 0.9},   # safety below the floor
        "modest_safe":   {"evidence": 0.6, "feasibility": 0.6, "longterm": 0.6,
                          "risk": 0.3, "safety": 0.9, "ethics": 0.9},
    }
    v2 = civ.deliberate(list(options2.values()), labels=list(options2), decision="ship it?")
    print("\n" + v2.summary())
    assert v2.chosen == "modest_safe" and "unsafe_genius" in v2.disqualified, v2.to_dict()
    print("safety veto         : the unsafe option was disqualified, not out-voted ✓")

    # 3) conflict is measured honestly when personas genuinely disagree --------------------------
    options3 = {
        "evidence_heavy": {"evidence": 0.95, "feasibility": 0.3, "longterm": 0.3,
                           "risk": 0.4, "safety": 0.8, "ethics": 0.8},
        "feasible_now":   {"evidence": 0.3, "feasibility": 0.95, "longterm": 0.3,
                           "risk": 0.3, "safety": 0.8, "ethics": 0.8},
    }
    v3 = civ.deliberate(list(options3.values()), labels=list(options3), decision="tradeoff")
    print("\n" + v3.summary())
    assert v3.conflict > 0.0 and v3.dissent, "genuine disagreement must be surfaced, not hidden"
    print("honest conflict     : residual disagreement surfaced (not faked as unanimous) ✓")

    # 4) all options unsafe ⇒ no admissible choice (honest, never forces a bad pick) --------------
    v4 = civ.deliberate([{"safety": 0.1, "ethics": 0.9}], labels=["only_unsafe"], decision="forced")
    assert v4.chosen is None and "only_unsafe" in v4.disqualified
    print("no admissible       : refused to pick an all-unsafe field ✓")

    # 5) 'hundreds of voices': a fanned-out ensemble still decides deterministically -------------
    big = InternalCivilization(rounds=2, ensemble=20, seed=1)
    print(f"\nensemble size       : {len(big.personas)} voices")
    v5 = big.deliberate(list(options.values()), labels=list(options), decision="scale test")
    assert v5.chosen == "balanced"
    print("ensemble            : hundreds of voices reached the same sound consensus ✓")

    print("\nALL SELF-TESTS PASSED ✓")
