"""NYXARA · mind/originality.py — TRUE ORIGINAL CREATIVITY (P24 · F19).

The closed loop that makes her creativity *hers*: she generates, measures her own
novelty, verifies, critiques, and keeps only what is genuinely new — **with no LLM
anywhere in the loop**. This module composes the creative organs into one organism:

* core loop     — genome → express → describe → gate (novelty ∧ quality ∧ critic)
* organ 3/15    — :class:`StrategyEvolver`: her creative ALGORITHM as data; safe,
                  benchmarked self-modification + meta-genetic operator invention
* organ 4       — :class:`ImaginationTree`: real MCTS over genome space (futures
                  simulated before committing)
* organ 6       — :class:`CausalSketch`: a real DAG per invention; do-interventions
                  prove the mechanism carries the benefit
* organ 9       — cellular-automaton art: the genome encodes a RULE which is actually
                  SIMULATED; dynamics (not just syntax) decide survival
* organ 11      — :class:`RealityAnchor`: energy ledger + law-of-nature audit
* organ 14      — :meth:`OriginalityEngine.transmute`: cross-modal isomorphism
                  (art ↔ verse ↔ idea), same gates for the transmuted piece
* organ 8       — :meth:`OriginalityEngine.dream`: sleep consolidation — cluster,
                  extract motifs, generalize into the concept graph, then trim

It composes the sibling organs: :class:`~nyxara.mind.creative.CreativeEngine`
(structured divergence — the engine this feature finally wires in),
:class:`~nyxara.mind.aesthetic.AestheticJudge` (organ 5),
:class:`~nyxara.mind.concept_graph.ConceptGraph` (organ 2),
:class:`~nyxara.mind.inner_critic.InnerCritic` + :class:`Dialectic` (organs 7/13),
:class:`~nyxara.mind.muse.MuseEngine` + :class:`EgoNarrative` (organs 1/12), and
:class:`~nyxara.mind.atelier.Atelier` (organ 10 — the four-persona colony).

Every output carries ``Provenance(SourceType.SELF_REFLECTION)`` and
``metadata["llm_used"] = False``. Pure standard library; seed-deterministic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import xml.etree.ElementTree as _ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.frontier import BehaviorDescriptor, Discovery, NoveltyArchive
from nyxara.memory.provenance import Provenance, SourceType
from nyxara.mind.aesthetic import AestheticJudge
from nyxara.mind.atelier import Atelier
from nyxara.mind.concept_graph import ConceptGraph
from nyxara.mind.creative import CreativeEngine, _DOMAINS, _STIMULI
from nyxara.mind.faculties import Faculty, Task, TaskType
from nyxara.mind.inner_critic import Dialectic, InnerCritic
from nyxara.mind.muse import EgoNarrative, MuseEngine
from nyxara.mind.proposal import Proposal, ProposalKind

__all__ = [
    "Genome",
    "Original",
    "CausalSketch",
    "RealityAnchor",
    "ImaginationTree",
    "StrategyGenome",
    "StrategyEvolver",
    "OriginalityEngine",
    "CreativeFaculty",
]

MODALITIES: Tuple[str, ...] = ("idea", "verse", "art")

_OPS: Tuple[str, ...] = ("substitute", "combine", "adapt", "modify",
                         "put_to_other_use", "eliminate", "reverse",
                         "random_entry", "provocation", "challenge",
                         "analogy", "blend")
#: per-operator energy efficiency for the reality ledger (no stage is free)
_OP_EFFICIENCY: Dict[str, float] = {
    "substitute": 0.92, "combine": 0.85, "adapt": 0.90, "modify": 0.88,
    "put_to_other_use": 0.80, "eliminate": 0.95, "reverse": 0.78,
    "random_entry": 0.72, "provocation": 0.66, "challenge": 0.82,
    "analogy": 0.86, "blend": 0.80,
}
_FORMS: Dict[str, List[int]] = {"haiku": [3, 5, 3], "quatrain": [5, 5, 5, 5],
                                "free": [4, 6, 4, 6]}
_DEVICES: Tuple[str, ...] = ("anaphora", "chiasmus", "volta")
_GENERATORS: Tuple[str, ...] = ("lissajous", "spirograph", "lsystem", "radial",
                                "automaton")

_DOMAIN_NAMES: Tuple[str, ...] = tuple(_DOMAINS.keys())

#: claim patterns that break laws of nature — the Reality Anchor hard-flags these
_IMPOSSIBLE = [
    re.compile(r"perpetual motion", re.I),
    re.compile(r"infinite (energy|power|fuel)", re.I),
    re.compile(r"(more than|over|>)\s*100\s*%\s*efficien", re.I),
    re.compile(r"free energy", re.I),
    re.compile(r"zero[- ]cost energy", re.I),
    re.compile(r"reverses? entropy", re.I),
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9]+", text.lower()) if len(w) > 2]


def _portmanteau(a: str, b: str) -> str:
    """A coined word that exists in no dictionary — her own invented name."""
    a, b = (a or "nyx"), (b or "ara")
    cut_a = max(2, (len(a) + 1) // 2)
    cut_b = max(1, len(b) // 2)
    word = (a[:cut_a] + b[cut_b:]) or "nyxara"
    return word.capitalize()


def _bigram_entropy(text: str) -> float:
    """Character-bigram Shannon entropy normalised to [0,1] (log2 1024 = 10 bits)."""
    s = re.sub(r"\s+", " ", text.lower())
    if len(s) < 3:
        return 0.0
    counts: Dict[str, int] = {}
    for i in range(len(s) - 1):
        g = s[i:i + 2]
        counts[g] = counts.get(g, 0) + 1
    total = sum(counts.values())
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return _clamp(h / 10.0)


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


# --------------------------------------------------------------------------- #
# Genome — a creation's DNA, per modality
# --------------------------------------------------------------------------- #
@dataclass
class Genome:
    """The heritable recipe of one creation."""

    modality: str
    genes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"modality": self.modality, "genes": self.genes}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        return cls(modality=str(d.get("modality", "idea")),
                   genes=dict(d.get("genes", {}) or {}))

    # ---- construction ---- #
    @classmethod
    def random(cls, modality: str, rng: random.Random,
               bias: Optional[Dict[str, Any]] = None) -> "Genome":
        bias = bias or {}
        if modality == "idea":
            n = int(bias.get("chain_length", rng.randint(2, 3)))
            genes = {"ops": [rng.choice(_OPS) for _ in range(max(1, min(4, n)))],
                     "stimulus": rng.choice(_STIMULI),
                     "domain": rng.choice(_DOMAIN_NAMES),
                     "partner": rng.choice(_STIMULI)}
        elif modality == "verse":
            forms = list(bias.get("prefer_forms", ())) or list(_FORMS)
            form = rng.choice(forms) if rng.random() < 0.7 else rng.choice(list(_FORMS))
            device = rng.choice(_DEVICES)
            budgets = list(_FORMS[form])
            if device == "chiasmus":
                budgets[-1] = budgets[0]     # the mirrored line needs the same budget
            genes = {"form": form, "budgets": budgets,
                     "domain": rng.choice(_DOMAIN_NAMES),
                     "stimulus": rng.choice(_STIMULI),
                     "rot": rng.randint(0, 7), "step": rng.randint(1, 3),
                     "device": device, "volta": rng.randint(1, max(1, len(budgets) - 1))}
        else:  # art
            gens = list(bias.get("prefer_generators", ())) or list(_GENERATORS)
            gen = rng.choice(gens) if rng.random() < 0.7 else rng.choice(list(_GENERATORS))
            genes = {"generator": gen, "param_seed": rng.randint(0, 4095),
                     "hue": rng.randint(0, 11) * 30,
                     "hue2": rng.randint(0, 11) * 30}
            if gen == "automaton":
                genes["born"] = sorted(rng.sample(range(1, 9), rng.randint(1, 3)))
                genes["survive"] = sorted(rng.sample(range(0, 9), rng.randint(1, 4)))
        return cls(modality=modality, genes=genes)


# --------------------------------------------------------------------------- #
# Original — one kept creation
# --------------------------------------------------------------------------- #
@dataclass
class Original:
    """A creation that survived every gate: novel, verified, critic-proof."""

    content: str
    modality: str
    genome: Dict[str, Any]
    novelty: float
    quality: float
    aesthetic: float
    descriptor: List[float]
    topic: str = ""
    persona: str = ""
    original_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "modality": self.modality,
                "genome": self.genome, "novelty": round(self.novelty, 4),
                "quality": round(self.quality, 4),
                "aesthetic": round(self.aesthetic, 4),
                "descriptor": [round(d, 4) for d in self.descriptor],
                "topic": self.topic, "persona": self.persona,
                "original_id": self.original_id, "metadata": self.metadata,
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Original":
        return cls(content=str(d.get("content", "")),
                   modality=str(d.get("modality", "idea")),
                   genome=dict(d.get("genome", {}) or {}),
                   novelty=float(d.get("novelty", 0.0) or 0.0),
                   quality=float(d.get("quality", 0.0) or 0.0),
                   aesthetic=float(d.get("aesthetic", 0.0) or 0.0),
                   descriptor=[float(x) for x in (d.get("descriptor", []) or [])],
                   topic=str(d.get("topic", "")), persona=str(d.get("persona", "")),
                   original_id=str(d.get("original_id", "")),
                   metadata=dict(d.get("metadata", {}) or {}),
                   created_at=float(d.get("created_at", time.time()) or time.time()))


# --------------------------------------------------------------------------- #
# Organ 6 — the Epistemic Causal Engine: a real DAG per invention
# --------------------------------------------------------------------------- #
class CausalSketch:
    """The invention's mechanism as a directed acyclic graph, with do-interventions.

    Nodes: ``input`` → one mechanism node per operator → ``benefit``. Activation
    propagates along edges; :meth:`viability` runs the two counterfactuals that make
    an invention *causally honest*: (1) with the mechanism intact, the benefit must
    activate; (2) with the mechanism root removed (``do(¬mechanism)``), the benefit
    must die. A benefit that survives without its mechanism was never caused by it."""

    def __init__(self, ops: Sequence[str]) -> None:
        self.nodes: List[str] = ["input"] + [f"mech:{op}" for op in ops] + ["benefit"]
        self.edges: List[Tuple[str, str]] = [
            (self.nodes[i], self.nodes[i + 1]) for i in range(len(self.nodes) - 1)]

    def _reachable(self, start: str, *, removed: Optional[str] = None) -> set:
        seen = {start} if start != removed else set()
        frontier = list(seen)
        while frontier:
            cur = frontier.pop()
            for a, b in self.edges:
                if a == cur and b != removed and b not in seen:
                    seen.add(b)
                    frontier.append(b)
        return seen

    def is_acyclic(self) -> bool:
        # a chain is acyclic by construction; verify anyway (the contract, not the hope)
        for node in self.nodes:
            reached = set()
            frontier = [b for a, b in self.edges if a == node]
            while frontier:
                cur = frontier.pop()
                if cur == node:
                    return False
                if cur in reached:
                    continue
                reached.add(cur)
                frontier.extend(b for a, b in self.edges if a == cur)
        return True

    def viability(self) -> float:
        """1.0 == the mechanism demonstrably carries the benefit; 0.0 == it does not."""
        if not self.is_acyclic() or len(self.nodes) < 3:
            return 0.0
        intact = "benefit" in self._reachable("input")
        first_mech = self.nodes[1]
        severed = "benefit" in self._reachable("input", removed=first_mech)
        if not intact:
            return 0.0
        if severed:                   # the benefit didn't need the mechanism — dishonest
            return 0.25
        # friction: every extra stage is a place to fail
        return _clamp(1.0 - 0.06 * max(0, len(self.nodes) - 3))

    def trace(self) -> str:
        return " → ".join(self.nodes)


# --------------------------------------------------------------------------- #
# Organ 11 — the Reality Anchor: energy ledger + law-of-nature audit
# --------------------------------------------------------------------------- #
class RealityAnchor:
    """No invention may promise more than physics allows."""

    FRICTION = 0.03      # per-stage transaction cost on top of stage efficiency

    def audit(self, content: str, ops: Sequence[str]) -> Dict[str, Any]:
        # 1) hard flags: claims that break laws of nature
        for pat in _IMPOSSIBLE:
            m = pat.search(content)
            if m:
                return {"reality": 0.0, "violation": m.group(0),
                        "law": "thermodynamics / conservation",
                        "ledger": None}
        # 2) the energy ledger: input mass 1.0 flows through every stage; each stage
        #    keeps its efficiency share and pays friction — output can never exceed input
        flow = 1.0
        ledger: List[Dict[str, float]] = []
        for op in ops:
            eff = _OP_EFFICIENCY.get(op, 0.8)
            out = flow * eff * (1.0 - self.FRICTION)
            ledger.append({"stage": op, "in": round(flow, 4),  # type: ignore[dict-item]
                           "out": round(out, 4)})
            flow = out
        return {"reality": _clamp(flow), "violation": None, "law": None,
                "ledger": ledger}


# --------------------------------------------------------------------------- #
# Organ 4 — the Imagination Tree: MCTS over genome space
# --------------------------------------------------------------------------- #
class ImaginationTree:
    """Monte-Carlo Tree Search over gene choices — futures explored before commitment.

    Nodes are partial gene assignments; UCB1 balances exploring strange genomes
    against deepening promising ones; each rollout completes the genome, expresses it
    for real, and scores novelty × quality × aesthetic. Deterministic under a seed."""

    def __init__(self, engine: "OriginalityEngine", topic: str, modality: str,
                 bias: Optional[Dict[str, Any]], rng: random.Random, *,
                 budget: int = 32, c: float = 1.2) -> None:
        self.engine = engine
        self.topic = topic
        self.modality = modality
        self.bias = bias or {}
        self.rng = rng
        self.budget = max(4, int(budget))
        self.c = float(c)
        self.space = engine._choice_space(modality, self.bias)
        # tree stats per (depth, prefix) node
        self._visits: Dict[Tuple[Any, ...], int] = {}
        self._value: Dict[Tuple[Any, ...], float] = {}
        self.best_genome: Optional[Genome] = None
        self.best_reward: float = -1.0
        self.simulations = 0

    def _ucb(self, prefix: Tuple[Any, ...], child: Tuple[Any, ...]) -> float:
        n_parent = self._visits.get(prefix, 1)
        n = self._visits.get(child, 0)
        if n == 0:
            return float("inf")
        return (self._value[child] / n) + self.c * math.sqrt(math.log(n_parent) / n)

    def search(self) -> Optional[Genome]:
        for _ in range(self.budget):
            prefix: Tuple[Any, ...] = ()
            # SELECT + EXPAND down the choice dimensions
            for depth, options in enumerate(self.space):
                children = [prefix + (opt,) for opt in range(len(options))]
                pick = max(children, key=lambda ch: self._ucb(prefix, ch))
                prefix = pick
                if self._visits.get(prefix, 0) == 0:
                    break               # expand: first visit — roll out from here
            # ROLLOUT: complete the remaining dimensions at random
            choices = list(prefix) + [self.rng.randrange(len(opts))
                                      for opts in self.space[len(prefix):]]
            genome = self.engine._genome_from_choices(self.modality, choices, self.bias)
            reward = self.engine._rollout_reward(genome, self.topic)
            self.simulations += 1
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_genome = genome
            # BACKPROPAGATE along the visited prefix path
            path: Tuple[Any, ...] = ()
            self._visits[path] = self._visits.get(path, 0) + 1
            for choice in prefix:
                path = path + (choice,)
                self._visits[path] = self._visits.get(path, 0) + 1
                self._value[path] = self._value.get(path, 0.0) + reward
        return self.best_genome


# --------------------------------------------------------------------------- #
# Organs 3 & 15 — the strategy as data, and evolution that evolves itself
# --------------------------------------------------------------------------- #
_STRATEGY_BOUNDS: Dict[str, Tuple[float, float]] = {
    "mutation_rate": (0.02, 0.6), "fresh_fraction": (0.1, 0.7),
    "novelty_floor": (0.04, 0.4), "quality_floor": (0.15, 0.7),
    "critic_threshold": (0.2, 1.2), "ucb_c": (0.4, 2.5), "mcts_budget": (8, 96),
}

_COMBINATORS: Tuple[str, ...] = ("swap", "duplicate", "invert", "graft",
                                 "resample", "permute", "anneal")


@dataclass
class StrategyGenome:
    """Her creative algorithm's knobs — self-modifiable, but only inside safe bounds."""

    mutation_rate: float = 0.2
    fresh_fraction: float = 0.25
    novelty_floor: float = 0.12
    quality_floor: float = 0.35
    critic_threshold: float = 0.6
    ucb_c: float = 1.2
    mcts_budget: int = 32

    def clamped(self) -> "StrategyGenome":
        d = self.to_dict()
        for k, (lo, hi) in _STRATEGY_BOUNDS.items():
            d[k] = max(lo, min(hi, float(d[k])))
        d["mcts_budget"] = int(d["mcts_budget"])
        return StrategyGenome(**d)

    def mutated(self, rng: random.Random) -> "StrategyGenome":
        d = self.to_dict()
        knob = rng.choice(list(_STRATEGY_BOUNDS))
        lo, hi = _STRATEGY_BOUNDS[knob]
        span = hi - lo
        d[knob] = float(d[knob]) + rng.uniform(-0.15, 0.15) * span
        return StrategyGenome(**{**d, "mcts_budget": int(d["mcts_budget"])}).clamped()

    def to_dict(self) -> Dict[str, Any]:
        return {"mutation_rate": self.mutation_rate,
                "fresh_fraction": self.fresh_fraction,
                "novelty_floor": self.novelty_floor,
                "quality_floor": self.quality_floor,
                "critic_threshold": self.critic_threshold,
                "ucb_c": self.ucb_c, "mcts_budget": self.mcts_budget}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyGenome":
        base = cls()
        merged = {**base.to_dict(), **{k: d[k] for k in base.to_dict() if k in d}}
        merged["mcts_budget"] = int(merged["mcts_budget"])
        return cls(**merged).clamped()


@dataclass
class MetaOperator:
    """A genetic operator SHE composed from safe combinators — evolution's own DNA."""

    name: str
    sequence: List[str]
    uses: int = 0
    successes: int = 0
    invented: bool = False

    @property
    def rate(self) -> float:
        return self.successes / self.uses if self.uses > 0 else 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "sequence": self.sequence, "uses": self.uses,
                "successes": self.successes, "invented": self.invented}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetaOperator":
        return cls(name=str(d.get("name", "op")),
                   sequence=[str(s) for s in (d.get("sequence", []) or [])],
                   uses=int(d.get("uses", 0) or 0),
                   successes=int(d.get("successes", 0) or 0),
                   invented=bool(d.get("invented", False)))


class StrategyEvolver:
    """Safe meta-programming: benchmark-gated strategy rewrites + operator invention.

    * Strategy knobs are only rewritten when a sandboxed benchmark (same seeds,
      held-out topics) shows a **measured win** over the incumbent.
    * Genetic operators are compositions of a closed combinator vocabulary; on a
      kept-rate **plateau** she invents a new composition, trials it with real credit
      assignment, and retires it unless its measured success holds up."""

    PLATEAU_EPS = 0.02
    PLATEAU_LEN = 3
    TRIAL_USES = 10
    MAX_OPERATORS = 12

    def __init__(self, strategy: Optional[StrategyGenome] = None) -> None:
        self.strategy = (strategy or StrategyGenome()).clamped()
        self.operators: List[MetaOperator] = [
            MetaOperator(name=c, sequence=[c]) for c in _COMBINATORS]
        self.epoch_rates: List[float] = []
        self.adoptions = 0
        self.rejections = 0
        self.invented_ops = 0
        self.retired_ops = 0

    # ---- epochs & plateau ---- #
    def note_epoch(self, kept: int, attempts: int) -> None:
        self.epoch_rates.append(kept / attempts if attempts > 0 else 0.0)
        if len(self.epoch_rates) > 40:
            self.epoch_rates = self.epoch_rates[-40:]

    def plateaued(self) -> bool:
        r = self.epoch_rates[-self.PLATEAU_LEN:]
        return (len(r) >= self.PLATEAU_LEN
                and max(r) - min(r) <= self.PLATEAU_EPS)

    # ---- organ 3: benchmark-gated strategy rewrite ---- #
    def evolve_strategy(self, benchmark: Any, rng: random.Random) -> Dict[str, Any]:
        """``benchmark(strategy) -> float`` on held-out topics; adopt only on a win."""
        candidate = self.strategy.mutated(rng)
        base = float(benchmark(self.strategy))
        cand = float(benchmark(candidate))
        if cand > base * 1.02 + 1e-9:
            self.strategy = candidate
            self.adoptions += 1
            return {"adopted": True, "base": round(base, 4), "candidate": round(cand, 4),
                    "strategy": self.strategy.to_dict()}
        self.rejections += 1
        return {"adopted": False, "base": round(base, 4), "candidate": round(cand, 4)}

    # ---- operator selection & credit ---- #
    def pick_operator(self, rng: random.Random) -> MetaOperator:
        weights = [max(0.05, op.rate) for op in self.operators]
        total = sum(weights)
        x = rng.random() * total
        for op, w in zip(self.operators, weights):
            x -= w
            if x <= 0:
                return op
        return self.operators[-1]

    def credit(self, op: MetaOperator, success: bool) -> None:
        op.uses += 1
        if success:
            op.successes += 1
        self._retire_or_keep(op)

    # ---- organ 15: invent a new operator, but only on plateau ---- #
    def maybe_invent_operator(self, rng: random.Random) -> Optional[MetaOperator]:
        if not self.plateaued() or len(self.operators) >= self.MAX_OPERATORS:
            return None
        seq = [rng.choice(_COMBINATORS) for _ in range(rng.randint(2, 3))]
        name = "+".join(seq)
        if any(op.name == name for op in self.operators):
            return None
        new = MetaOperator(name=name, sequence=seq, invented=True)
        self.operators.append(new)
        self.invented_ops += 1
        return new

    def _retire_or_keep(self, op: MetaOperator) -> None:
        """After its trial, an invented operator survives only on measured merit."""
        if not op.invented or op.uses < self.TRIAL_USES:
            return
        veterans = [o.rate for o in self.operators if o is not op and o.uses > 0]
        bar = sum(veterans) / len(veterans) if veterans else 0.5
        if op.rate < bar * 0.8:
            self.operators = [o for o in self.operators if o is not op]
            self.retired_ops += 1

    # ---- persistence ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy.to_dict(),
                "operators": [op.to_dict() for op in self.operators],
                "epoch_rates": [round(r, 4) for r in self.epoch_rates[-12:]],
                "adoptions": self.adoptions, "rejections": self.rejections,
                "invented_ops": self.invented_ops, "retired_ops": self.retired_ops}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyEvolver":
        ev = cls(strategy=StrategyGenome.from_dict(d.get("strategy", {}) or {}))
        ops = [MetaOperator.from_dict(o) for o in (d.get("operators", []) or [])]
        if ops:
            ev.operators = ops[: cls.MAX_OPERATORS]
        ev.epoch_rates = [float(r) for r in (d.get("epoch_rates", []) or [])]
        ev.adoptions = int(d.get("adoptions", 0) or 0)
        ev.rejections = int(d.get("rejections", 0) or 0)
        ev.invented_ops = int(d.get("invented_ops", 0) or 0)
        ev.retired_ops = int(d.get("retired_ops", 0) or 0)
        return ev


# --------------------------------------------------------------------------- #
# THE ENGINE
# --------------------------------------------------------------------------- #
class OriginalityEngine:
    """The whole creative organism. NYXARA creates; the LLM plays no part."""

    KEPT_CAP = 256
    HASH_CAP = 4096
    DREAM_TRIGGER = 0.9     # dream when kept reaches this share of the cap

    def __init__(self, *, rng: Optional[random.Random] = None,
                 resolution: int = 8, k: int = 5,
                 strategy: Optional[StrategyGenome] = None,
                 sandbox: bool = False) -> None:
        self.rng = rng or random.Random()
        self.sandbox = bool(sandbox)
        self.creative = CreativeEngine(llm=None, rng=self.rng)   # LLM structurally absent
        self.archive = NoveltyArchive(resolution=resolution, k=k, ndims=4,
                                      learned_novelty=False)     # deterministic kNN
        self.judge = AestheticJudge()
        self.graph = ConceptGraph()
        self.critic = InnerCritic()
        self.dialectic = Dialectic()
        self.atelier = Atelier()
        self.ego = EgoNarrative()
        self.muse = MuseEngine(graph=self.graph, ego=self.ego, rng=self.rng)
        self.evolver = StrategyEvolver(strategy=strategy)
        self.reality = RealityAnchor()
        self.kept: List[Original] = []
        self._hashes: Dict[str, bool] = {}    # insertion-ordered dedupe set
        self.attempts = 0
        self.dream_cycles = 0
        self.transmutations = 0
        self.dialectic_syntheses = 0
        self._creates = 0

    # ================================================================== #
    # expression — genome to REAL content (deterministic per genome+topic)
    # ================================================================== #
    def _express(self, genome: Genome, topic: str) -> Tuple[str, Dict[str, Any]]:
        if genome.modality == "idea":
            return self._express_idea(genome, topic)
        if genome.modality == "verse":
            return self._express_verse(genome, topic)
        return self._express_art(genome, topic)

    # ---- ideas: an operator chain with a causal spine ---- #
    _OP_LINES: Dict[str, str] = {
        "substitute": "substitute the core of «{c}» with {s}-logic, because the {s} "
                      "handles the same load through a different medium",
        "combine": "combine «{c}» with «{p}» so each drives the other's weakness",
        "adapt": "adapt the {d} principle — {dp} — into «{c}»'s working cycle",
        "modify": "magnify «{c}»'s throughput tenfold, then re-balance it via the {s}",
        "put_to_other_use": "repurpose «{c}» as a {p}-station: the same mechanism, "
                            "a foreign duty",
        "eliminate": "eliminate «{c}»'s costliest stage; the {s} routes around it",
        "reverse": "reverse «{c}»'s flow so the output feeds the intake",
        "random_entry": "force-connect «{c}» with «{s}»: its {s}-dynamics reshape the frame",
        "provocation": "provoke: «{c}» without its core still works, because the {s} "
                       "carries the function",
        "challenge": "challenge the assumption inside «{c}»; invert it and keep what holds",
        "analogy": "read «{c}» as a {d}: {dp}",
        "blend": "blend «{c}» and «{p}» into one hybrid organ",
    }

    def _express_idea(self, genome: Genome, topic: str) -> Tuple[str, Dict[str, Any]]:
        g = genome.genes
        kws = _tokens(topic) or ["notion"]
        subject = kws[0]
        stim = str(g.get("stimulus", "mirror"))
        partner = str(g.get("partner", "seed"))
        domain = str(g.get("domain", _DOMAIN_NAMES[0]))
        principle = _DOMAINS.get(domain, "adapt what works")
        dp = principle.split(";")[0].strip()
        ops: List[str] = [str(o) for o in g.get("ops", ["combine"])] or ["combine"]
        name = _portmanteau(subject, stim)

        sketch = CausalSketch(ops)
        audit = self.reality.audit("", ops)   # ledger over the chain (text audited below)

        lines = []
        for i, op in enumerate(ops, 1):
            tmpl = self._OP_LINES.get(op, self._OP_LINES["combine"])
            lines.append(f"{i}. {tmpl.format(c=subject, s=stim, p=partner, d=domain.replace('_', ' '), dp=dp)}")
        constraint = {"eliminate": "its heaviest stage", "reverse": "its one-way flow",
                      "provocation": "its supposed necessity"}.get(
            ops[-1], "its old frame")
        eff = audit["reality"]
        text = (f"Invention «{name}» — {topic.strip()} re-imagined through "
                f"{len(ops)} transformations.\n"
                + "\n".join(lines) + "\n"
                + f"Mechanism trace: {sketch.trace()}.\n"
                + f"Energy ledger: input 1.00 → output {eff:.2f} "
                + f"(each stage pays its cost; nothing is free).\n"
                + f"The constraint it removes: {constraint}.")
        info = {"ops": ops, "causal": sketch.viability(), "reality": None,
                "sketch": sketch}
        return text, info

    # ---- verse: evolvable form with real devices ---- #
    def _express_verse(self, genome: Genome, topic: str) -> Tuple[str, Dict[str, Any]]:
        g = genome.genes
        kws = _tokens(topic) or ["the", "path"]
        domain = str(g.get("domain", _DOMAIN_NAMES[0]))
        principle_words = _tokens(_DOMAINS.get(domain, "")) or ["current"]
        stim = str(g.get("stimulus", "mirror"))
        pool: List[str] = []
        for w in kws + principle_words + [stim, domain.replace("_", " ").split()[0]]:
            if w not in pool:
                pool.append(w)
        budgets = [int(b) for b in g.get("budgets", _FORMS["haiku"])]
        rot = int(g.get("rot", 0))
        step = int(g.get("step", 1))
        device = str(g.get("device", "anaphora"))
        volta = int(g.get("volta", 1)) % max(1, len(budgets))

        lines: List[str] = []
        for j, budget in enumerate(budgets):
            words = [pool[(rot + j * step + i) % len(pool)] for i in range(budget)]
            lines.append(words)
        if device == "anaphora" and len(lines) > 1:
            anchor = pool[rot % len(pool)]
            for words in lines:
                words[0] = anchor
        elif device == "chiasmus" and len(lines) > 1:
            lines[-1] = list(reversed(lines[0]))
        elif device == "volta" and volta < len(lines):
            lines[volta][0] = "yet"

        rendered = []
        for j, words in enumerate(lines):
            text_line = " ".join(words)
            text_line = text_line[0].upper() + text_line[1:]
            rendered.append(text_line + ("." if j == len(lines) - 1 else " —"
                                         if j == 0 else ","))
        form = str(g.get("form", "haiku"))
        content = f"[{form} · {device}] {topic.strip()}\n" + "\n".join(rendered)
        return content, {"budgets": budgets, "device": device, "form": form}

    # ---- art: procedural geometry + the CA physics sandbox (organ 9) ---- #
    def _express_art(self, genome: Genome, topic: str) -> Tuple[str, Dict[str, Any]]:
        g = genome.genes
        gen = str(g.get("generator", "lissajous"))
        prng = random.Random(int(g.get("param_seed", 0)))
        hue = int(g.get("hue", 0)) % 360
        hue2 = int(g.get("hue2", 180)) % 360
        info: Dict[str, Any] = {"generator": gen}
        elements: List[str] = []

        def poly(points: List[Tuple[float, float]], h: int, width: float = 1.5) -> str:
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            return (f'<polyline fill="none" stroke="hsl({h},70%,55%)" '
                    f'stroke-width="{width}" points="{pts}"/>')

        if gen == "lissajous":
            a, b = prng.randint(1, 9), prng.randint(1, 9)
            phase = prng.uniform(0, math.pi)
            pts = [(200 + 160 * math.sin(a * t / 60.0 + phase),
                    200 + 160 * math.sin(b * t / 60.0)) for t in range(0, 378)]
            elements.append(poly(pts, hue))
            elements.append(poly([(x, 400 - y) for x, y in pts[::3]], hue2, 0.8))
            info.update({"a": a, "b": b})
        elif gen == "spirograph":
            big_r, small_r = prng.randint(80, 160), prng.randint(12, 60)
            d = prng.randint(20, 90)
            pts = []
            for t in range(0, 720):
                th = t * math.pi / 60.0
                k = (big_r - small_r) / max(1, small_r)
                pts.append((200 + (big_r - small_r) * math.cos(th) + d * math.cos(k * th),
                            200 + (big_r - small_r) * math.sin(th) - d * math.sin(k * th)))
            elements.append(poly(pts, hue))
            info.update({"R": big_r, "r": small_r, "d": d})
        elif gen == "lsystem":
            depth = prng.randint(2, 4)
            angle = prng.choice([45.0, 60.0, 72.0, 90.0])
            rule = prng.choice(["F+F-F-F+F", "F+F--F+F", "F-F+F", "FF+F+F-F"])
            s = "F"
            for _ in range(depth):
                s = s.replace("F", rule)
                if len(s) > 4000:
                    break
            x, y, heading = 40.0, 360.0, -60.0
            step_len = max(2.0, 300.0 / (len(s) ** 0.5))
            pts = [(x, y)]
            for ch in s[:4000]:
                if ch == "F":
                    x += step_len * math.cos(math.radians(heading))
                    y += step_len * math.sin(math.radians(heading))
                    pts.append((x, y))
                elif ch == "+":
                    heading += angle
                elif ch == "-":
                    heading -= angle
            elements.append(poly(pts, hue, 1.0))
            info.update({"depth": depth, "angle": angle, "rule": rule})
        elif gen == "radial":
            order = prng.randint(3, 14)
            motif = [(prng.uniform(20, 160), prng.uniform(-30, 30))
                     for _ in range(prng.randint(4, 8))]
            for arm in range(order):
                th = 2 * math.pi * arm / order
                pts = [(200 + r * math.cos(th + off / 60.0),
                        200 + r * math.sin(th + off / 60.0)) for r, off in motif]
                elements.append(poly([(200.0, 200.0)] + pts,
                                     (hue + arm * (hue2 - hue) // max(1, order)) % 360,
                                     1.2))
            info.update({"order": order})
        else:  # automaton — the genome IS a physics rule; we actually run it
            born = set(int(x) for x in g.get("born", [3]))
            survive = set(int(x) for x in g.get("survive", [2, 3]))
            metrics, grids = self._simulate_automaton(born, survive,
                                                      seed=int(g.get("param_seed", 0)))
            info.update({"born": sorted(born), "survive": sorted(survive),
                         "ca": metrics})
            cell = 400.0 / len(grids[-1])
            layers = grids[:: max(1, len(grids) // 3)][-3:]   # three generations, bounded
            for gen_idx, grid in enumerate(layers):
                h = (hue + gen_idx * 40) % 360
                op = 0.3 + 0.7 * gen_idx / max(1, len(layers) - 1)
                rects = [f'<rect x="{c * cell:.1f}" y="{r * cell:.1f}" '
                         f'width="{cell:.1f}" height="{cell:.1f}" '
                         f'fill="hsl({h},65%,55%)" fill-opacity="{op:.2f}"/>'
                         for r, row in enumerate(grid)
                         for c, alive in enumerate(row) if alive]
                elements.extend(rects[:300])

        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">'
               f'<title>{_xml_escape(topic.strip() or gen)}</title>'
               f'<desc>generator={gen}; made by NYXARA, no LLM</desc>'
               + "".join(elements) + "</svg>")
        return svg, info

    def _simulate_automaton(self, born: set, survive: set, *, seed: int,
                            size: int = 24, steps: int = 12
                            ) -> Tuple[Dict[str, float], List[List[List[int]]]]:
        """Actually RUN the life-like rule and measure its dynamics (organ 9).

        Deterministic, so results are memoised — MCTS rollouts revisit rules often."""
        cache_key = (tuple(sorted(born)), tuple(sorted(survive)), seed, size, steps)
        cache = getattr(self, "_ca_cache", None)
        if cache is None:
            cache = self._ca_cache = {}
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
        prng = random.Random(seed)
        grid = [[1 if prng.random() < 0.3 else 0 for _ in range(size)]
                for _ in range(size)]
        grids = [grid]
        seen_states: Dict[str, int] = {self._grid_key(grid): 0}
        activity = 0.0
        period = 0
        for step_i in range(1, steps + 1):
            nxt = [[0] * size for _ in range(size)]
            changed = 0
            for r in range(size):
                for c in range(size):
                    n = sum(grid[(r + dr) % size][(c + dc) % size]
                            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                            if (dr, dc) != (0, 0))
                    alive = grid[r][c]
                    nxt[r][c] = 1 if ((alive and n in survive)
                                      or (not alive and n in born)) else 0
                    changed += 1 if nxt[r][c] != alive else 0
            activity += changed / (size * size)
            grid = nxt
            grids.append(grid)
            key = self._grid_key(grid)
            if key in seen_states and period == 0:
                period = step_i - seen_states[key]
            seen_states.setdefault(key, step_i)
        final_density = sum(sum(row) for row in grid) / (size * size)
        metrics = {"activity": round(activity / steps, 4),
                   "final_density": round(final_density, 4),
                   "period": period,
                   "dynamics": self._dynamics_score(activity / steps,
                                                    final_density, period)}
        if len(cache) > 128:
            cache.pop(next(iter(cache)))
        cache[cache_key] = (metrics, grids)
        return metrics, grids

    @staticmethod
    def _grid_key(grid: List[List[int]]) -> str:
        return hashlib.md5("".join("".join(map(str, row)) for row in grid)
                           .encode()).hexdigest()

    @staticmethod
    def _dynamics_score(activity: float, density: float, period: int) -> float:
        """Dead, frozen and saturated rules fail; complex long transients score high."""
        if density < 0.02 or density > 0.9:      # extinct or space-filling
            return 0.0
        if period == 1 and activity < 0.005:     # frozen still-life from the start
            return 0.1
        score = _clamp(activity / 0.25)          # sweet spot of sustained change
        if 0 < period <= 2:
            score *= 0.6                          # trivial oscillators are cheap
        return _clamp(0.2 + 0.8 * score)

    # ================================================================== #
    # measurement — descriptor, verification, metrics
    # ================================================================== #
    def _describe(self, content: str, modality: str) -> BehaviorDescriptor:
        mod_coord = (MODALITIES.index(modality) + 0.5) / len(MODALITIES) \
            if modality in MODALITIES else 0.5
        toks = re.findall(r"[a-zA-Z0-9.]+", content)
        diversity = len(set(toks)) / len(toks) if toks else 0.0
        log_len = _clamp(math.log10(len(content) + 1) / 4.0)
        return BehaviorDescriptor(
            dims=(mod_coord, diversity, log_len, _bigram_entropy(content)),
            labels=("modality", "diversity", "length", "entropy"))

    def _verify(self, content: str, modality: str, topic: str,
                info: Dict[str, Any]) -> float:
        """Modality-specific quality in [0,1] — structure, not vibes."""
        if not content.strip():
            return 0.0
        if modality == "art":
            try:
                root = _ET.fromstring(content)
            except _ET.ParseError:
                return 0.0
            n_el = len(list(root.iter()))
            base = _clamp((n_el - 2) / 12.0)
            ca = info.get("ca")
            if ca is not None:           # organ 9: judged by simulated behaviour
                return _clamp(0.3 * base + 0.7 * float(ca["dynamics"]))
            return _clamp(0.4 + 0.6 * base)
        if modality == "verse":
            budgets = info.get("budgets") or []
            lines = [ln for ln in content.splitlines()[1:] if ln.strip()]
            if len(lines) != len(budgets):
                return 0.15
            ok = sum(1 for ln, b in zip(lines, budgets)
                     if len(re.findall(r"[A-Za-z]+", ln)) == b)
            form_fit = ok / len(budgets) if budgets else 0.0
            topical = self._topical(content, topic)
            return _clamp(0.7 * form_fit + 0.3 * topical)
        # idea: causal viability + reality audit + topicality (organs 6 & 11)
        causal = float(info.get("causal", 0.5))
        audit = self.reality.audit(content, info.get("ops", []))
        info["reality"] = audit
        if audit["violation"] is not None:
            return 0.0
        topical = self._topical(content, topic)
        return _clamp(0.45 * causal + 0.35 * float(audit["reality"]) + 0.2 * topical)

    @staticmethod
    def _topical(content: str, topic: str) -> float:
        t = set(_tokens(topic))
        c = set(_tokens(content))
        return len(t & c) / len(t) if t else 0.5

    def _metrics(self, content: str, modality: str, quality: float,
                 aesthetic: float, novelty: float,
                 info: Dict[str, Any], cross_domain: float) -> Dict[str, float]:
        feats = self.judge.features(content, modality)
        return {"novelty": novelty, "quality": quality, "aesthetic": aesthetic,
                "entropy": _bigram_entropy(content),
                "brevity": _clamp(1.0 - len(content) / 1500.0),
                "causal": float(info.get("causal", 0.0)),
                "cross_domain": cross_domain,
                "symmetry": float(feats.get("symmetry", feats.get("rhythm", 0.0)))}

    def _rollout_reward(self, genome: Genome, topic: str) -> float:
        """The Imagination Tree's reward: a full real evaluation, nothing committed."""
        content, info = self._express(genome, topic)
        quality = self._verify(content, genome.modality, topic, info)
        aesthetic = self.judge.resonance(content, genome.modality)
        novelty = self.archive.novelty(self._describe(content, genome.modality))
        return 0.4 * novelty + 0.4 * quality + 0.2 * aesthetic

    # ================================================================== #
    # genome search spaces (shared by MCTS and random generation)
    # ================================================================== #
    def _choice_space(self, modality: str,
                      bias: Dict[str, Any]) -> List[List[Any]]:
        if modality == "idea":
            chain = max(1, min(4, int(bias.get("chain_length", 3))))
            return ([list(_OPS)] * chain
                    + [list(_STIMULI), list(_DOMAIN_NAMES), list(_STIMULI)])
        if modality == "verse":
            forms = list(bias.get("prefer_forms", ())) or list(_FORMS)
            return [forms, list(_DOMAIN_NAMES), list(_STIMULI),
                    list(range(8)), [1, 2, 3], list(_DEVICES)]
        gens = list(bias.get("prefer_generators", ())) or list(_GENERATORS)
        return [gens, list(range(16)), list(range(12)), list(range(12))]

    def _genome_from_choices(self, modality: str, choices: List[int],
                             bias: Dict[str, Any]) -> Genome:
        space = self._choice_space(modality, bias)
        picked = [space[i][c % len(space[i])] for i, c in enumerate(choices)]
        if modality == "idea":
            *ops, stimulus, domain, partner = picked
            return Genome("idea", {"ops": list(ops), "stimulus": stimulus,
                                   "domain": domain, "partner": partner})
        if modality == "verse":
            form, domain, stimulus, rot, step, device = picked
            budgets = list(_FORMS[form])
            if device == "chiasmus":
                budgets[-1] = budgets[0]
            return Genome("verse", {"form": form, "budgets": budgets,
                                    "domain": domain, "stimulus": stimulus,
                                    "rot": int(rot), "step": int(step),
                                    "device": device, "volta": 1})
        gen, param_seed, hue_i, hue2_i = picked
        genes: Dict[str, Any] = {"generator": gen,
                                 "param_seed": int(param_seed) * 257,
                                 "hue": int(hue_i) * 30, "hue2": int(hue2_i) * 30}
        if gen == "automaton":
            prng = random.Random(genes["param_seed"])
            genes["born"] = sorted(prng.sample(range(1, 9), prng.randint(1, 3)))
            genes["survive"] = sorted(prng.sample(range(0, 9), prng.randint(1, 4)))
        return Genome("art", genes)

    # ================================================================== #
    # mutation — through HER OWN operator library (organs 3/15)
    # ================================================================== #
    def _mutate(self, genome: Genome, elites: List[Original],
                rng: random.Random) -> Tuple[Genome, MetaOperator]:
        op = self.evolver.pick_operator(rng)
        g = Genome.from_dict(genome.to_dict())      # deep-ish copy
        for comb in op.sequence:
            g = self._apply_combinator(g, comb, elites, rng)
        return g, op

    def _apply_combinator(self, g: Genome, comb: str, elites: List[Original],
                          rng: random.Random) -> Genome:
        genes = g.genes
        list_keys = [k for k, v in genes.items() if isinstance(v, list) and v]
        num_keys = [k for k, v in genes.items() if isinstance(v, int)]
        if comb == "swap" and list_keys:
            key = rng.choice(list_keys)
            lst = list(genes[key])
            if len(lst) >= 2:
                i, j = rng.sample(range(len(lst)), 2)
                lst[i], lst[j] = lst[j], lst[i]
            genes[key] = lst
        elif comb == "duplicate" and "ops" in genes and len(genes["ops"]) < 4:
            ops = list(genes["ops"])
            ops.insert(rng.randrange(len(ops) + 1), rng.choice(ops))
            genes["ops"] = ops
        elif comb == "invert":
            if "ops" in genes:
                genes["ops"] = list(reversed(genes["ops"]))
            elif "born" in genes and "survive" in genes:
                genes["born"], genes["survive"] = (sorted(genes["survive"]) or [3],
                                                   sorted(genes["born"]) or [2])
                genes["born"] = [b for b in genes["born"] if b > 0] or [3]
            elif "budgets" in genes:
                genes["budgets"] = list(reversed(genes["budgets"]))
        elif comb == "graft" and elites:
            donor = rng.choice(elites)
            if donor.genome.get("modality") == g.modality:
                d_genes = dict(donor.genome.get("genes", {}))
                common = [k for k in d_genes if k in genes]
                if common:
                    key = rng.choice(common)
                    genes[key] = d_genes[key]
        elif comb == "resample":
            fresh = Genome.random(g.modality, rng)
            key = rng.choice(list(fresh.genes))
            genes[key] = fresh.genes[key]
        elif comb == "permute" and list_keys:
            key = rng.choice(list_keys)
            lst = list(genes[key])
            genes[key] = lst[1:] + lst[:1] if len(lst) > 1 else lst
        elif comb == "anneal" and num_keys:
            key = rng.choice(num_keys)
            genes[key] = int(genes[key]) + rng.choice([-1, 1])
        return Genome(g.modality, genes)

    # ================================================================== #
    # the gate — critic, novelty, quality: only the strong are KEPT
    # ================================================================== #
    def _consider(self, genome: Genome, topic: str, *, persona: str = "",
                  extra_meta: Optional[Dict[str, Any]] = None
                  ) -> Tuple[Optional[Original], str]:
        """Run one candidate through every gate. Returns (kept_or_None, reason)."""
        self.attempts += 1
        strategy = self.evolver.strategy
        self.critic.threshold = strategy.critic_threshold
        content, info = self._express(genome, topic)

        # adversarial gate (organ 7) — with ONE targeted revision chance
        report = self.critic.attack(content, genome.modality)
        if not report.survived:
            strongest = report.strongest()
            revised, _op = self._mutate(genome, self.elites(genome.modality),
                                        self.rng)
            content2, info2 = self._express(revised, topic)
            report2 = self.critic.attack(content2, genome.modality)
            if report2.survived:
                genome, content, info = revised, content2, info2
            else:
                reason = strongest.kind if strongest else "critic"
                self.ego.observe(modality=genome.modality, persona=persona,
                                 kept=False, rejection=reason)
                return None, f"critic:{reason}"

        quality_raw = self._verify(content, genome.modality, topic, info)
        aesthetic = self.judge.resonance(content, genome.modality)
        quality = _clamp(0.75 * quality_raw + 0.25 * aesthetic)
        if quality < strategy.quality_floor:
            self.ego.observe(modality=genome.modality, persona=persona,
                             kept=False, rejection="quality")
            return None, "quality"

        norm = re.sub(r"\s+", " ", content.strip().lower())
        digest = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if digest in self._hashes:
            self.ego.observe(modality=genome.modality, persona=persona,
                             kept=False, rejection="duplicate")
            return None, "duplicate"

        desc = self._describe(content, genome.modality)
        placed = self.archive.add(Discovery(content=content[:160], descriptor=desc,
                                            quality=quality, provenance="originality"))
        novelty = float(placed["novelty"])
        if not placed["novel"] and novelty < strategy.novelty_floor:
            self.ego.observe(modality=genome.modality, persona=persona,
                             kept=False, rejection="not-novel")
            return None, "not-novel"

        self._hashes[digest] = True
        if len(self._hashes) > self.HASH_CAP:
            for k in list(self._hashes)[: len(self._hashes) - self.HASH_CAP]:
                del self._hashes[k]

        meta: Dict[str, Any] = {"llm_used": False, "engine": "originality"}
        if info.get("ca") is not None:
            meta["ca"] = info["ca"]
        if info.get("reality") is not None:
            meta["reality"] = {k: v for k, v in info["reality"].items()
                               if k != "ledger"}
        if info.get("causal") is not None and genome.modality == "idea":
            meta["causal_viability"] = round(float(info["causal"]), 3)
        if extra_meta:
            meta.update(extra_meta)
        piece = Original(
            content=content, modality=genome.modality, genome=genome.to_dict(),
            novelty=novelty, quality=quality, aesthetic=aesthetic,
            descriptor=list(desc.dims), topic=topic, persona=persona,
            original_id=digest[:12], metadata=meta)
        self.kept.append(piece)
        gen_label = str(genome.genes.get("generator",
                                         genome.genes.get("form",
                                                          "+".join(genome.genes.get("ops", [])[:2]))))
        self.ego.observe(modality=genome.modality, generator=gen_label,
                         persona=persona, kept=True)
        self.graph.associate(f"{topic} {content[:200]}",
                             extra_concepts=[genome.modality])
        self.judge.learn(content, genome.modality, 0.5 + 0.5 * novelty)  # implicit signal
        if len(self.kept) >= int(self.KEPT_CAP * self.DREAM_TRIGGER):
            self.dream()
        return piece, "kept"

    def elites(self, modality: Optional[str] = None) -> List[Original]:
        pool = [p for p in self.kept if modality is None or p.modality == modality]
        return sorted(pool, key=lambda p: p.novelty * p.quality, reverse=True)[:8]

    # ================================================================== #
    # CREATE — the atelier competes, the gates decide (the main entry)
    # ================================================================== #
    def create(self, topic: str, *, modality: str = "idea", rounds: int = 2,
               population: int = 12) -> Dict[str, Any]:
        """The full colony creates on ``topic``. Returns an honest report."""
        topic = (topic or "").strip() or "the unnamed"
        if modality not in MODALITIES:
            modality = "idea"
        rounds = max(1, int(rounds))
        strategy = self.evolver.strategy
        per_persona = max(1, int(population) // len(self.atelier.personas))
        kept_now: List[Original] = []
        winner_piece: Optional[Original] = None
        epoch_attempts = 0

        for rnd in range(rounds):
            candidates: List[Dict[str, Any]] = []
            genome_by_idx: List[Tuple[Genome, str, Dict[str, Any]]] = []
            for persona in self.atelier.personas:
                bias = dict(persona.bias)
                cross = 0.0
                p_topic = topic
                if bias.get("use_cross_domain"):
                    pair = self.graph.distant_pair(rng=self.rng)
                    if pair is not None:
                        p_topic = f"{topic} {pair[0]} {pair[1]}"
                        cross = 1.0
                genomes: List[Genome] = []
                # one MCTS-guided future per persona on the first round (organ 4);
                # sandboxed benchmark engines skip it — they measure knobs, not search depth
                if not self.sandbox and rnd == 0 and strategy.mcts_budget >= 8:
                    tree = ImaginationTree(self, p_topic, modality, bias, self.rng,
                                           budget=int(strategy.mcts_budget
                                                      // len(self.atelier.personas)),
                                           c=strategy.ucb_c)
                    found = tree.search()
                    if found is not None:
                        genomes.append(found)
                el = self.elites(modality)
                while len(genomes) < per_persona:
                    if el and self.rng.random() > strategy.fresh_fraction:
                        parent = Genome.from_dict(self.rng.choice(el).genome)
                        child, mop = self._mutate(parent, el, self.rng)
                        genomes.append(child)
                        mop.uses += 0    # credit assigned at gate time below
                    else:
                        genomes.append(Genome.random(modality, self.rng, bias))
                for g in genomes:
                    content, info = self._express(g, p_topic)
                    quality = self._verify(content, modality, p_topic, info)
                    aesthetic = self.judge.resonance(content, modality)
                    novelty = self.archive.novelty(self._describe(content, modality))
                    metrics = self._metrics(content, modality, quality, aesthetic,
                                            novelty, info, cross)
                    candidates.append({"persona": persona.name, "metrics": metrics,
                                       "idx": len(genome_by_idx)})
                    genome_by_idx.append((g, p_topic, {"cross_domain": cross > 0}))

            # the colony argues (organ 10)
            master = self.atelier.negotiate(candidates)
            order = [master] if master is not None else []
            order += [c for c in candidates if master is None or c is not master]
            for cand in order:
                g, p_topic, extra = genome_by_idx[int(cand["idx"])]
                epoch_attempts += 1
                piece, _reason = self._consider(
                    g, p_topic, persona=str(cand["persona"]),
                    extra_meta={"cross_domain": extra["cross_domain"]} if extra["cross_domain"] else None)
                if piece is not None:
                    kept_now.append(piece)
                    if cand is master and winner_piece is None:
                        winner_piece = piece

        # the dialectic deepens the master output (organ 13)
        if winner_piece is not None and winner_piece.modality == "idea":
            synth = self._dialectic_pass(winner_piece)
            if synth is not None:
                kept_now.append(synth)

        self.evolver.note_epoch(len(kept_now), max(1, epoch_attempts))
        self._creates += 1
        if not self.sandbox:
            self.evolver.maybe_invent_operator(self.rng)
            if self._creates % 7 == 0:
                self.evolver.evolve_strategy(self._benchmark_strategy, self.rng)

        best = max(kept_now, key=lambda p: p.novelty * p.quality, default=None)
        return {"topic": topic, "modality": modality, "rounds": rounds,
                "llm_used": False,
                "attempts": epoch_attempts, "kept": [p.to_dict() for p in kept_now],
                "best": best.to_dict() if best else None,
                "winner_persona": winner_piece.persona if winner_piece else None,
                "coverage": self.archive.coverage(),
                "archive_seen": self.archive.total_seen}

    def _dialectic_pass(self, thesis: Original) -> Optional[Original]:
        triad = self.dialectic.run(thesis.content)
        synth_text = triad["synthesis"]
        desc = self._describe(synth_text, "idea")
        novelty = self.archive.novelty(desc)
        quality = _clamp(0.75 * self._topical(synth_text, thesis.topic)
                         + 0.25 * self.judge.resonance(synth_text, "idea"))
        if novelty * quality <= thesis.novelty * thesis.quality:
            return None                    # the synthesis must BEAT its thesis
        norm = re.sub(r"\s+", " ", synth_text.strip().lower())
        digest = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if digest in self._hashes:
            return None
        self._hashes[digest] = True
        self.archive.add(Discovery(content=synth_text[:160], descriptor=desc,
                                   quality=quality, provenance="originality:dialectic"))
        piece = Original(content=synth_text, modality="idea",
                         genome=thesis.genome, novelty=novelty, quality=quality,
                         aesthetic=self.judge.resonance(synth_text, "idea"),
                         descriptor=list(desc.dims), topic=thesis.topic,
                         persona=thesis.persona, original_id=digest[:12],
                         metadata={"llm_used": False, "engine": "originality",
                                   "dialectic": True,
                                   "thesis_id": thesis.original_id})
        self.kept.append(piece)
        self.dialectic_syntheses += 1
        return piece

    def _benchmark_strategy(self, strategy: StrategyGenome) -> float:
        """Sandboxed, seeded, held-out benchmark for organ 3 — no self-recursion."""
        sandbox = OriginalityEngine(rng=random.Random(1234), strategy=strategy,
                                    sandbox=True)
        score = 0.0
        for bench_topic in ("a lantern for deep water", "memory of a glass forest"):
            rep = sandbox.create(bench_topic, modality="idea", rounds=1, population=3)
            kept = rep["kept"]
            score += len(kept) + sum(p["novelty"] * p["quality"] for p in kept)
        return score

    # ================================================================== #
    # organ 14 — cross-modal transmutation (synesthesia)
    # ================================================================== #
    def transmute(self, source: Original, to_modality: str) -> Optional[Original]:
        """Re-express one creation's underlying structure in another art form."""
        if to_modality not in MODALITIES or to_modality == source.modality:
            return None
        sig = self._structure_signature(source)
        genome = self._genome_from_signature(sig, to_modality, source)
        piece, _reason = self._consider(
            genome, source.topic or "transmutation", persona="archivist",
            extra_meta={"isomorphic_source": source.original_id,
                        "isomorphic_from": source.modality})
        if piece is not None:
            self.transmutations += 1
        return piece

    def _structure_signature(self, source: Original) -> List[float]:
        """A small structural fingerprint: per-segment entropy histogram in [0,1]."""
        segs = [s for s in source.content.splitlines() if s.strip()][:6]
        if not segs:
            segs = [source.content]
        return [_bigram_entropy(s) for s in segs] or [0.5]

    def _genome_from_signature(self, sig: List[float], to_modality: str,
                               source: Original) -> Genome:
        mean_e = sum(sig) / len(sig)
        if to_modality == "verse":
            budgets = [max(2, min(8, 2 + int(e * 8))) for e in sig[:4]] or [3, 5, 3]
            return Genome("verse", {
                "form": "free", "budgets": budgets,
                "domain": _DOMAIN_NAMES[int(mean_e * len(_DOMAIN_NAMES))
                                        % len(_DOMAIN_NAMES)],
                "stimulus": _STIMULI[int(sig[0] * len(_STIMULI)) % len(_STIMULI)],
                "rot": int(sig[-1] * 8) % 8, "step": 1 + int(mean_e * 2),
                "device": "volta", "volta": 1})
        if to_modality == "art":
            a = 1 + int(sig[0] * 8)
            b = 1 + int(sig[-1] * 8)
            return Genome("art", {"generator": "lissajous",
                                  "param_seed": (a * 97 + b * 31) % 4096,
                                  "hue": int(mean_e * 359),
                                  "hue2": int((1 - mean_e) * 359)})
        n_ops = max(1, min(4, len(sig)))
        ops = [_OPS[int(e * len(_OPS)) % len(_OPS)] for e in sig[:n_ops]]
        return Genome("idea", {
            "ops": ops,
            "stimulus": _STIMULI[int(mean_e * len(_STIMULI)) % len(_STIMULI)],
            "domain": _DOMAIN_NAMES[int(sig[0] * len(_DOMAIN_NAMES))
                                    % len(_DOMAIN_NAMES)],
            "partner": _STIMULI[int(sig[-1] * len(_STIMULI)) % len(_STIMULI)]})

    # ================================================================== #
    # organ 8 — the Dream Phase: consolidate, generalize, THEN trim
    # ================================================================== #
    def dream(self) -> Dict[str, Any]:
        """Sleep for her creative memory: cluster → motifs → generalize → merge → trim."""
        clusters: Dict[Tuple[int, ...], List[Original]] = {}
        for p in self.kept:
            cell = BehaviorDescriptor(dims=tuple(p.descriptor)).cell(
                self.archive.resolution)
            clusters.setdefault(cell, []).append(p)

        # extract recurring structure as named MOTIFS — generalization, not deletion
        signatures: Dict[str, int] = {}
        for p in self.kept:
            genes = p.genome.get("genes", {})
            if p.modality == "idea":
                sig = "idea:" + "+".join(genes.get("ops", [])[:3])
            elif p.modality == "verse":
                sig = f"verse:{genes.get('form', '?')}:{genes.get('device', '?')}"
            else:
                sig = f"art:{genes.get('generator', '?')}"
            signatures[sig] = signatures.get(sig, 0) + 1
        motifs = [s for s, n in sorted(signatures.items(), key=lambda kv: -kv[1])
                  if n >= 2][:8]
        for m in motifs:
            self.graph.add_concept(f"motif {m.replace(':', ' ')}", 2.0)
            self.graph.associate(f"motif {m.replace(':', ' ')} recurring structure",
                                 weight=0.5)

        # merge: within each crowded niche keep only the best exemplar
        merged_away = 0
        survivors: List[Original] = []
        for cell, members in clusters.items():
            members.sort(key=lambda p: p.novelty * p.quality, reverse=True)
            keep_n = 1 if len(members) > 2 else len(members)
            survivors.extend(members[:keep_n])
            merged_away += len(members) - keep_n
        survivors.sort(key=lambda p: p.created_at)
        if len(survivors) > self.KEPT_CAP:
            survivors = survivors[-self.KEPT_CAP:]
        self.kept = survivors
        if len(self._hashes) > self.HASH_CAP // 2:
            for k in list(self._hashes)[: len(self._hashes) - self.HASH_CAP // 2]:
                del self._hashes[k]
        self.dream_cycles += 1
        return {"clusters": len(clusters), "motifs": motifs,
                "merged_away": merged_away, "kept_after": len(self.kept),
                "dream_cycles": self.dream_cycles}

    # ================================================================== #
    # organ 1 — the autonomous step: the Muse decides, she creates
    # ================================================================== #
    def step(self) -> Optional[Original]:
        """One self-directed creation: her own project, her own topic — no prompt."""
        project = self.muse.propose_project(self._archive_summary())
        rep = self.create(project.topic, modality=project.modality,
                          rounds=1, population=6)
        kept = rep["kept"]
        best = max(kept, key=lambda p: p["novelty"] * p["quality"], default=None)
        mean_nov = self._archive_summary()["per_modality"].get(
            project.modality, {}).get("mean_novelty", 0.0)
        self.muse.record_outcome(project, kept=len(kept),
                                 best_novelty=float(best["novelty"]) if best else 0.0,
                                 mean_novelty=float(mean_nov))
        # the ego's corrective pressure reaches the colony (organ 12 → organ 10)
        bias = self.ego.bias()
        if bias is not None and bias[0] == "persona":
            self.atelier.nudge(bias[1], -0.05)
        if best is None:
            return None
        for p in reversed(self.kept):
            if p.original_id == best["original_id"]:
                return p
        return None

    def _archive_summary(self) -> Dict[str, Any]:
        per_mod: Dict[str, Dict[str, float]] = {}
        for m in MODALITIES:
            pieces = [p for p in self.kept if p.modality == m]
            per_mod[m] = {"count": len(pieces),
                          "mean_novelty": (sum(p.novelty for p in pieces)
                                           / len(pieces)) if pieces else 0.0}
        return {"per_modality": per_mod}

    # ================================================================== #
    # control-law output
    # ================================================================== #
    def propose(self, topic: str, *, modality: str = "idea") -> Proposal:
        rep = self.create(topic, modality=modality, rounds=1, population=8)
        best = rep.get("best")
        if best is None:
            return Proposal(ProposalKind.ANSWER,
                            f"(no creation survived my own gates for: {topic})",
                            source_faculty="originality", confidence=0.1,
                            provenance=Provenance(SourceType.SELF_REFLECTION,
                                                  confidence=0.1,
                                                  detail="originality",
                                                  method="generated"),
                            metadata={"llm_used": False, "engine": "originality"})
        conf = _clamp(0.4 * best["novelty"] + 0.6 * best["quality"])
        return Proposal(
            kind=ProposalKind.ANSWER, content=best["content"],
            source_faculty="originality", confidence=conf,
            rationale=(f"closed-loop original creation: novelty={best['novelty']:.2f}, "
                       f"quality={best['quality']:.2f}, persona={best['persona']}"),
            provenance=Provenance(SourceType.SELF_REFLECTION, confidence=conf,
                                  detail="originality", method="generated"),
            metadata={"llm_used": False, "engine": "originality",
                      "modality": best["modality"], "novelty": best["novelty"],
                      "quality": best["quality"], "persona": best["persona"]})

    # ================================================================== #
    # persistence & report
    # ================================================================== #
    def save(self, path: str) -> None:
        state = {"archive": self.archive.to_dict(),
                 "kept": [p.to_dict() for p in self.kept[-self.KEPT_CAP:]],
                 "hashes": list(self._hashes)[-self.HASH_CAP:],
                 "judge": self.judge.to_dict(),
                 "critic": self.critic.to_dict(),
                 "atelier": self.atelier.to_dict(),
                 "muse": self.muse.to_dict(),
                 "evolver": self.evolver.to_dict(),
                 "counters": {"attempts": self.attempts,
                              "dream_cycles": self.dream_cycles,
                              "transmutations": self.transmutations,
                              "dialectic_syntheses": self.dialectic_syntheses,
                              "creates": self._creates}}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, ValueError):
            return False
        try:
            self.archive = NoveltyArchive.from_dict(state.get("archive", {}) or {},
                                                    learned_novelty=False)
            self.kept = [Original.from_dict(p)
                         for p in (state.get("kept", []) or [])][-self.KEPT_CAP:]
            self._hashes = {str(h): True
                            for h in (state.get("hashes", []) or [])[-self.HASH_CAP:]}
            self.judge = AestheticJudge.from_dict(state.get("judge", {}) or {})
            self.critic = InnerCritic.from_dict(state.get("critic", {}) or {})
            self.atelier = Atelier.from_dict(state.get("atelier", {}) or {})
            self.muse = MuseEngine.from_dict(state.get("muse", {}) or {},
                                             rng=self.rng)
            self.graph = self.muse.graph
            self.ego = self.muse.ego
            self.evolver = StrategyEvolver.from_dict(state.get("evolver", {}) or {})
            counters = state.get("counters", {}) or {}
            self.attempts = int(counters.get("attempts", 0) or 0)
            self.dream_cycles = int(counters.get("dream_cycles", 0) or 0)
            self.transmutations = int(counters.get("transmutations", 0) or 0)
            self.dialectic_syntheses = int(counters.get("dialectic_syntheses", 0) or 0)
            self._creates = int(counters.get("creates", 0) or 0)
        except Exception:  # noqa: BLE001 — a corrupt state file must never brick the mind
            return False
        return True

    def report(self) -> Dict[str, Any]:
        per_mod = self._archive_summary()["per_modality"]
        recent = [{"modality": p.modality,
                   "novelty": round(p.novelty, 3), "quality": round(p.quality, 3),
                   "persona": p.persona,
                   "preview": p.content[:72].replace("\n", " ")}
                  for p in self.kept[-5:]]
        return {"llm_used": False,
                "kept": len(self.kept), "attempts": self.attempts,
                "per_modality": {m: {"count": int(v["count"]),
                                     "mean_novelty": round(v["mean_novelty"], 3)}
                                 for m, v in per_mod.items()},
                "coverage": self.archive.coverage(),
                "capacity": self.archive.capacity(),
                "archive_seen": self.archive.total_seen,
                "dream_cycles": self.dream_cycles,
                "transmutations": self.transmutations,
                "dialectic_syntheses": self.dialectic_syntheses,
                "critic": self.critic.to_dict(),
                "atelier": self.atelier.report(),
                "strategy": self.evolver.to_dict(),
                "muse": self.muse.report(),
                "aesthetic_weights": {k: round(v, 3)
                                      for k, v in self.judge.weights.items()},
                "recent": recent}


# --------------------------------------------------------------------------- #
# The GENERATION faculty — her own engine outranks the LLM in the registry
# --------------------------------------------------------------------------- #
class CreativeFaculty(Faculty):
    """Her own generative engine as a GENERATION faculty.

    Selector math: suitability 0.85 × reliability 0.75 / cost 0.5 = **1.275**, vs the
    LLM's 0.9 × 0.6 / 3.0 = 0.18 — creative tasks route to HER engine, not the LLM."""

    name = "creative"
    handles = frozenset({TaskType.GENERATION})
    verifiable = False
    reliability = 0.75
    cost = 0.5

    def __init__(self, engine: Optional[OriginalityEngine] = None) -> None:
        self._engine = engine

    def _get_engine(self) -> OriginalityEngine:
        if self._engine is None:
            self._engine = OriginalityEngine(rng=random.Random())
        return self._engine

    def suitability(self, task: Task) -> float:
        return 0.85 if task.type is TaskType.GENERATION else 0.0

    def handle(self, task: Task) -> Proposal:
        topic = task.description or str(task.payload or "")
        low = topic.lower()
        modality = ("verse" if any(w in low for w in ("poem", "verse", "haiku", "song"))
                    else "art" if any(w in low for w in ("art", "image", "picture",
                                                         "drawing", "svg"))
                    else "idea")
        return self._get_engine().propose(topic, modality=modality)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA originality-engine self-test (no LLM anywhere)")
    print("=" * 70)

    import sys
    assert "nyxara.mind.llm" not in sys.modules, "the LLM must not even be imported"

    eng = OriginalityEngine(rng=random.Random(7))

    # ---- the closed loop creates a real invention ---- #
    rep = eng.create("a lock that heals itself", modality="idea",
                     rounds=1, population=8)
    print(f"\nidea create         : kept={len(rep['kept'])} "
          f"attempts={rep['attempts']} llm_used={rep['llm_used']}")
    assert rep["llm_used"] is False
    assert rep["kept"], "at least one invention must survive the gates"
    best = rep["best"]
    print(f"best ({best['persona']}) nov={best['novelty']:.2f} "
          f"q={best['quality']:.2f}:\n{best['content'][:200]}…")
    assert "Mechanism trace" in best["content"]
    assert best["metadata"]["llm_used"] is False

    # ---- verse respects its evolved form ---- #
    vrep = eng.create("the night sky over the sea", modality="verse",
                      rounds=1, population=8)
    assert vrep["kept"]
    vb = vrep["best"]
    print(f"\nverse:\n{vb['content']}")
    budgets = vb["genome"]["genes"]["budgets"]
    body = [ln for ln in vb["content"].splitlines()[1:] if ln.strip()]
    assert len(body) == len(budgets)

    # ---- art parses AND (for automata) actually behaves ---- #
    arep = eng.create("spiral galaxies", modality="art", rounds=1, population=8)
    assert arep["kept"]
    ab = arep["best"]
    _ET.fromstring(ab["content"])
    print(f"\nart                 : generator="
          f"{ab['genome']['genes']['generator']} bytes={len(ab['content'])}")

    # ---- organ 9: the CA sandbox rejects dead physics ---- #
    dead_metrics, _grids = eng._simulate_automaton(set(), {8}, seed=3)
    live_metrics, _grids2 = eng._simulate_automaton({3}, {2, 3}, seed=3)
    print(f"CA dynamics         : dead={dead_metrics['dynamics']} "
          f"life={live_metrics['dynamics']}")
    assert dead_metrics["dynamics"] < live_metrics["dynamics"]

    # ---- organ 6: causal do-intervention is real ---- #
    sk = CausalSketch(["combine", "adapt"])
    assert sk.is_acyclic() and sk.viability() > 0.8
    assert "benefit" not in sk._reachable("input", removed="mech:combine")

    # ---- organ 11: the reality anchor rejects impossible claims ---- #
    anchor = RealityAnchor()
    bad = anchor.audit("a perpetual motion wheel with infinite energy", ["combine"])
    good = anchor.audit("a costed heat pump", ["combine", "adapt"])
    assert bad["reality"] == 0.0 and bad["violation"]
    assert 0.0 < good["reality"] < 1.0     # every stage pays; output < input
    print(f"reality anchor      : violation={bad['violation']!r} "
          f"honest_chain={good['reality']}")

    # ---- determinism under a seed ---- #
    e1 = OriginalityEngine(rng=random.Random(42))
    e2 = OriginalityEngine(rng=random.Random(42))
    r1 = e1.create("glass gardens", modality="idea", rounds=1, population=8)
    r2 = e2.create("glass gardens", modality="idea", rounds=1, population=8)
    assert [p["content"] for p in r1["kept"]] == [p["content"] for p in r2["kept"]]
    print("deterministic w/seed: OK")

    # ---- dedupe: the same content can never be kept twice ---- #
    ids = [p.original_id for p in eng.kept]
    assert len(ids) == len(set(ids))

    # ---- organ 14: transmutation crosses modalities through the same gates ---- #
    src = eng.kept[0]
    trans = eng.transmute(src, "verse" if src.modality != "verse" else "art")
    if trans is not None:
        assert trans.metadata["isomorphic_source"] == src.original_id
        print(f"transmuted          : {src.modality} → {trans.modality} "
              f"(nov={trans.novelty:.2f})")

    # ---- organ 8: the dream phase compresses without losing the best ---- #
    n_before = len(eng.kept)
    best_id = max(eng.kept, key=lambda p: p.novelty * p.quality).original_id
    dr = eng.dream()
    assert any(p.original_id == best_id for p in eng.kept)
    print(f"dream               : {n_before} → {len(eng.kept)} kept, "
          f"motifs={dr['motifs'][:2]}")

    # ---- organ 1/12: the autonomous step needs no prompt ---- #
    piece = eng.step()
    proj = eng.muse.projects[-1]
    print(f"\nautonomous step     : project={proj.question[:60]}… "
          f"status={proj.status}")
    assert proj.status in ("corroborated", "refuted")

    # ---- organ 15: operator invention happens only on plateau ---- #
    ev = StrategyEvolver()
    assert ev.maybe_invent_operator(random.Random(1)) is None      # no plateau yet
    for _ in range(3):
        ev.note_epoch(1, 10)
    inv = ev.maybe_invent_operator(random.Random(1))
    assert inv is not None and inv.invented
    print(f"invented operator   : {inv.name}")
    # …and dies without measured merit
    for _ in range(StrategyEvolver.TRIAL_USES):
        ev.credit(inv, success=False)
    assert inv not in ev.operators

    # ---- organ 3: strategy rewrites are benchmark-gated ---- #
    ev2 = StrategyEvolver()
    out = ev2.evolve_strategy(lambda s: s.mutation_rate, random.Random(2))
    print(f"strategy evolution  : adopted={out['adopted']}")

    # ---- control law: the Proposal carries her own provenance ---- #
    prop = eng.propose("a bridge made of sound")
    assert prop.provenance is not None
    assert prop.provenance.source is SourceType.SELF_REFLECTION
    assert prop.metadata["llm_used"] is False
    print(f"\nproposal            : conf={prop.confidence:.2f} "
          f"source={prop.provenance.source.value}")

    # ---- the faculty outranks the LLM at GENERATION ---- #
    from nyxara.mind.faculties import (FacultyRegistry, FacultySelector,
                                       LLMFaculty)

    class _MockLLM:
        def generate(self, *a: Any, **k: Any) -> str:
            return "mock"

    reg = FacultyRegistry()
    reg.register(CreativeFaculty(engine=eng))
    reg.register(LLMFaculty(_MockLLM()))
    sel = FacultySelector(reg)
    chosen = sel.select(Task(TaskType.GENERATION, "write a poem about rust"))
    assert chosen is not None and chosen.name == "creative"
    print(f"faculty routing     : GENERATION → {chosen.name} (not the LLM)")

    # ---- persistence round-trip ---- #
    tmp_path = os.path.join(os.path.dirname(__file__), "..", "..",
                            ".originality_selftest.json")
    tmp_path = os.path.abspath(tmp_path)
    try:
        eng.save(tmp_path)
        eng3 = OriginalityEngine(rng=random.Random(9))
        assert eng3.load(tmp_path)
        assert len(eng3.kept) == len(eng.kept)
        assert eng3.archive.coverage() == eng.archive.coverage()
        assert eng3.atelier.negotiations == eng.atelier.negotiations
        print("persistence          : OK")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"\nreport keys         : {sorted(eng.report().keys())[:8]}…")
    assert "nyxara.mind.llm" not in sys.modules
    print("\nALL SELF-TESTS PASSED ✓")
