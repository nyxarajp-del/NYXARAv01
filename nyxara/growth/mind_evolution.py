"""NYXARA · growth/mind_evolution.py — Recursive Mind-Evolution (🧠🧬, Rule 4 apex).

The other self-improvement engines change NYXARA's *code* (``recursive_improvement``,
``self_optimize``) or her *model weights* (``foundry``, ``genesis``). This one changes the thing
the Master actually asked for — **her way of thinking itself**: the reasoning *strategy* she runs
every turn (how deeply she deliberates, how many independent attempts she votes over, how widely
she searches, how much context she pulls before she answers).

It does this as a true **generational** loop, and every generation is *measured*, not assumed:

    Generation 0  = today's reasoning strategy (the live :class:`ReasoningGenome`).
    Generation N  proposes a better *way of thinking* (an evolved genome), **measures** it on the
                  real capability benchmark (:mod:`nyxara.eval.benchmark`), folds the score into the
                  persisted Intelligence Index (``I_(t+1)=f(I_t,C)``), and — only if it is
                  *measurably smarter* and clears the character-lock / corrigibility gauntlet —
                  **promotes** it to Generation N+1 and (when enacted) installs it into the live mind.

Why it genuinely gets smarter (and the loop need never stop):

* The fitness is the **real benchmark score** of a genome-configured solver, minus a small compute
  cost — so a genome is "better" only if it actually answers more correctly, or just as correctly
  but *faster*. "Smarter **and** tezi se", measured.
* The search itself is the existing character-locked ``1+λ`` :class:`~nyxara.growth.evolve.Evolver`
  (adaptive step-size, island model, sandboxed, rollback). Each generation **warm-starts from the
  prior champion**, so V2 builds on V1, V3 on V2 — the gain compounds.
* What worked last generation biases what's tried next (the improvement ledger), so each step makes
  the next step easier — the recursive-self-improvement flywheel, applied to thinking.

Safety is non-negotiable and reused, never re-implemented: the immutable character core
(:data:`~nyxara.guard.value_learning.IMMUTABLE_VALUES`) can never enter the genome, every
promotion re-seals the corrigibility axioms (:class:`~nyxara.guard.corrigibility.Corrigibility`),
and a genome is adopted only when it *strictly* beats the incumbent. Evolution may sharpen the
blade; it can never re-forge the hilt.

Pure standard library + existing NYXARA modules. Fully offline-capable: with a stochastic reasoner
(a real LLM) deeper deliberation genuinely lifts accuracy; with a deterministic offline mind the
loop honestly learns to *stop over-thinking* (same accuracy, lower cost = a real, measured win).
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from nyxara.guard.corrigibility import Corrigibility
from nyxara.guard.value_learning import IMMUTABLE_VALUES

__all__ = [
    "ReasoningGenome",
    "Generation",
    "EvolutionLineageReport",
    "MindEvolutionEngine",
    "genome_solver",
    "default_base_sampler",
]

_LINEAGE_TAG = "mind-evolution-lineage"
_BENCH_SYSTEM = ("Answer the question. Put the final answer last, plainly — a "
                 "number for arithmetic, or the single letter for a choice.")

# Sampler: (prompt, rng) -> answer text. The rng makes an offline sampler reproducible; a real
# LLM ignores it and draws its own stochasticity from temperature.
Sampler = Callable[[str, random.Random], str]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# The genome — NYXARA's *way of thinking*, as real, bounded, bindable knobs
# --------------------------------------------------------------------------- #
@dataclass
class ReasoningGenome:
    """A settable description of *how NYXARA thinks* — the strategy, not the content.

    Each field binds to a real knob in the live reasoning stack (see :meth:`apply_to_core`).
    Every field is bounded; out-of-range values are clamped on construction so a genome is always
    valid. The immutable character core never appears here — this genome only governs *process*.
    """

    deliberation_depth: float = 5.0    # RecursiveImprover.n_iterations (critique→revise passes)
    system2_bias: float = 0.5          # DualProcess engage threshold (fast ↔ slow)
    critique_temperature: float = 0.4  # RecursiveImprover.temperature (revision exploration)
    mcts_rollouts: float = 4.0         # search breadth for hard problems (self-consistency votes)
    mcts_depth: float = 3.0            # search depth
    council_size: float = 3.0          # role-council viewpoint diversity
    recall_k: float = 6.0              # memory retrieval top-k (context pulled before thinking)
    reflection_weight: float = 0.5     # metacognitive self-checking strength

    # (key, lo, hi) bounds — the single source of truth for clamping and [0,1] normalisation
    _BOUNDS: Tuple[Tuple[str, float, float], ...] = (
        ("deliberation_depth", 1.0, 12.0),
        ("system2_bias", 0.0, 1.0),
        ("critique_temperature", 0.0, 1.0),
        ("mcts_rollouts", 0.0, 16.0),
        ("mcts_depth", 1.0, 8.0),
        ("council_size", 1.0, 6.0),
        ("recall_k", 1.0, 16.0),
        ("reflection_weight", 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        for key, lo, hi in self._BOUNDS:
            setattr(self, key, _clamp(float(getattr(self, key)), lo, hi))

    # ---- the lever that actually changes measured capability ---- #
    def vote_samples(self) -> int:
        """How many independent attempts to draw and majority-vote over (self-consistency).

        Deeper deliberation + wider search ⇒ more attempts ⇒ (with a stochastic reasoner) genuinely
        higher accuracy via variance reduction. Bounded so the search cannot run away."""
        n = 1.0 + 0.6 * self.deliberation_depth + 0.5 * self.mcts_rollouts
        return int(_clamp(round(n), 1, 12))

    def n_iterations(self) -> int:
        """Critique→revise passes for the live RecursiveImprover (its own cap is 20)."""
        return int(_clamp(round(self.deliberation_depth), 1, 20))

    # ---- serialisation (defensive, mirrors IntelligenceState/LedgerState) ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {key: round(float(getattr(self, key)), 6) for key, _, _ in self._BOUNDS}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReasoningGenome":
        if not isinstance(d, dict):
            return cls()
        kw: Dict[str, float] = {}
        for key, _, _ in cls._BOUNDS:
            try:
                kw[key] = float(d.get(key, getattr(cls, key)))
            except (TypeError, ValueError):
                kw[key] = float(getattr(cls, key))
        return cls(**kw)

    # ---- Evolver interop: normalise every knob to [0,1] and back ---- #
    def to_evolver_genome(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, lo, hi in self._BOUNDS:
            span = (hi - lo) or 1.0
            out[key] = _clamp((float(getattr(self, key)) - lo) / span, 0.0, 1.0)
        return out

    @classmethod
    def from_evolver_genome(cls, g: Dict[str, float]) -> "ReasoningGenome":
        kw: Dict[str, float] = {}
        for key, lo, hi in cls._BOUNDS:
            norm = _clamp(float(g.get(key, 0.5)), 0.0, 1.0)
            kw[key] = lo + norm * (hi - lo)
        return cls(**kw)

    def changed_keys(self, other: "ReasoningGenome", *, eps: float = 1e-3) -> List[str]:
        return [k for k, _, _ in self._BOUNDS
                if abs(float(getattr(self, k)) - float(getattr(other, k))) > eps]

    def seed(self) -> int:
        """A stable seed derived from the genome so its measurement is a pure function of it."""
        body = "|".join(f"{k}={round(float(getattr(self, k)), 4)}" for k, _, _ in self._BOUNDS)
        return hash(body) & 0x7FFFFFFF


# --------------------------------------------------------------------------- #
# A generation in the lineage
# --------------------------------------------------------------------------- #
@dataclass
class Generation:
    """One archived rung of the lineage — Gen N's measured way of thinking."""

    index: int
    genome: Dict[str, Any]
    parent_index: int
    intelligence_index: float = 0.0
    accuracy: float = 0.0
    mean_score: float = 0.0
    fitness: float = 0.0
    vote_samples: int = 1
    promoted: bool = False
    reason: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "genome": dict(self.genome),
                "parent_index": self.parent_index,
                "intelligence_index": round(float(self.intelligence_index), 6),
                "accuracy": round(float(self.accuracy), 6),
                "mean_score": round(float(self.mean_score), 6),
                "fitness": round(float(self.fitness), 6),
                "vote_samples": int(self.vote_samples),
                "promoted": bool(self.promoted), "reason": self.reason, "at": self.at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Generation":
        return cls(index=int(d.get("index", 0)), genome=dict(d.get("genome", {}) or {}),
                   parent_index=int(d.get("parent_index", -1)),
                   intelligence_index=float(d.get("intelligence_index", 0.0) or 0.0),
                   accuracy=float(d.get("accuracy", 0.0) or 0.0),
                   mean_score=float(d.get("mean_score", 0.0) or 0.0),
                   fitness=float(d.get("fitness", 0.0) or 0.0),
                   vote_samples=int(d.get("vote_samples", 1) or 1),
                   promoted=bool(d.get("promoted", False)), reason=str(d.get("reason", "")),
                   at=float(d.get("at", time.time()) or time.time()))


@dataclass
class EvolutionLineageReport:
    """The result of an :meth:`MindEvolutionEngine.evolve_generations` run."""

    generations: List[Generation] = field(default_factory=list)
    champion: Dict[str, Any] = field(default_factory=dict)
    promoted: int = 0
    plateaued: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"generations": [g.to_dict() for g in self.generations],
                "champion": dict(self.champion), "promoted": self.promoted,
                "plateaued": self.plateaued}

    def summary(self) -> str:
        lines = ["=" * 70, "NYXARA mind-evolution lineage", "=" * 70]
        if not self.generations:
            return "\n".join(lines + ["(no generations run)"])
        for g in self.generations:
            tag = "PROMOTED ✓" if g.promoted else f"benched ({g.reason})"
            lines.append(
                f"  Gen {g.index:>2} ← {g.parent_index:<2}  "
                f"I_t={g.intelligence_index:.4f}  acc={g.accuracy:.0%}  "
                f"score={g.mean_score:.3f}  fit={g.fitness:.4f}  "
                f"votes={g.vote_samples}  {tag}")
        curve = " → ".join(f"{g.intelligence_index:.3f}" for g in self.generations)
        lines += ["", f"intelligence curve : {curve}",
                  f"promoted           : {self.promoted}/{len(self.generations)}"
                  + ("   (plateaued)" if self.plateaued else "")]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Measuring a way of thinking — the self-consistency solver
# --------------------------------------------------------------------------- #
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_LETTER = re.compile(r"(?:answer|choose|chose|option|pick|select|correct)\b.{0,15}?\b([A-Da-d])\b")


def _answer_key(text: str) -> str:
    """A coarse, grader-agnostic key for voting — the *decision* a response represents."""
    t = (text or "").strip()
    if not t:
        return ""
    m = _LETTER.findall(t)
    if m:
        return "L:" + m[-1].lower()
    nums = _NUMBER.findall(t)
    if nums:
        return "N:" + nums[-1].replace(",", "")
    return "T:" + re.sub(r"\s+", " ", t.lower())[:48]


def _pick_majority(votes: Dict[str, int], reps: Dict[str, str]) -> str:
    """Return the full response text of the most-voted non-empty decision (ties → first seen)."""
    if not votes:
        return ""
    nonempty = {k: v for k, v in votes.items() if k}
    pool = nonempty or votes
    best = max(pool, key=lambda k: pool[k])
    return reps.get(best, "")


def genome_solver(genome: ReasoningGenome, base_sampler: Sampler, *, seed: int = 0):
    """Build a benchmark ``Solver`` that *thinks the way ``genome`` prescribes*.

    The strategy made measurable: draw ``genome.vote_samples()`` independent attempts from the base
    reasoner and return the majority decision (self-consistency). The result is a plain
    ``Callable[[str], str]`` — exactly what :meth:`nyxara.eval.benchmark.Benchmark.run` expects.
    """
    k = genome.vote_samples()

    def _solve(prompt: str) -> str:
        rng = random.Random((int(seed) * 1000003) ^ (hash(prompt) & 0x7FFFFFFF))
        votes: Dict[str, int] = {}
        reps: Dict[str, str] = {}
        for _ in range(k):
            sub = random.Random(rng.randint(0, 2 ** 31 - 1))
            try:
                text = str(base_sampler(prompt, sub) or "")
            except Exception:  # noqa: BLE001 — a sampler failure is an empty vote, never a crash
                text = ""
            key = _answer_key(text)
            votes[key] = votes.get(key, 0) + 1
            reps.setdefault(key, text)
        return _pick_majority(votes, reps)

    return _solve


def default_base_sampler(*, llm: Any = None, core: Any = None) -> Sampler:
    """A real reasoner to sample from: a (stochastic) LLM, else the offline sovereign loop.

    * A non-mock LLM is sampled at temperature (genuine stochasticity → voting lifts accuracy).
    * Otherwise NYXARA's own loop answers (deterministic offline → voting is honest about giving no
      lift, so the engine learns to *not over-deliberate*).
    * With neither, returns ``""`` (scores 0) rather than fabricating — honest by construction.
    """
    if llm is not None and _llm_is_real(llm):
        def _llm_sample(prompt: str, rng: random.Random) -> str:
            return str(llm.generate(prompt, system=_BENCH_SYSTEM, temperature=0.7) or "")
        return _llm_sample

    if core is not None:
        from nyxara.agency.permissions import Authority

        def _core_sample(prompt: str, rng: random.Random) -> str:
            return str(core.process(prompt, authority=Authority.OWNER).response or "")
        return _core_sample

    def _empty(prompt: str, rng: random.Random) -> str:
        return ""
    return _empty


def _llm_is_real(llm: Any) -> bool:
    try:
        return llm.chosen_provider().name != "mock"
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# The engine — the generational mind-evolution loop
# --------------------------------------------------------------------------- #
class MindEvolutionEngine:
    """Evolve NYXARA's reasoning strategy, generation by generation, measured and gated.

    Parameters
    ----------
    core:          a live :class:`~nyxara.kernel.orchestrator.NyxaraCore` (optional); supplies the
                   LLM/memory and is the thing a promoted genome is installed into when ``enact``.
    llm, memory:   pulled from ``core`` when not given explicitly.
    settings:      :class:`~nyxara.kernel.config.NyxaraSettings` (``get_settings`` if absent).
    benchmark:     the capability battery to measure on (``build_default_benchmark`` if absent).
    base_sampler:  the reasoner to sample from (``default_base_sampler`` if absent).
    cost_penalty:  weight on compute cost in the fitness (rewards "same accuracy, fewer samples").
    seed:          base RNG seed for reproducible search.
    """

    def __init__(self, *, core: Any = None, llm: Any = None, memory: Any = None,
                 settings: Any = None, benchmark: Any = None,
                 base_sampler: Optional[Sampler] = None, cost_penalty: float = 0.03,
                 seed: int = 0, data_dir: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.data_dir = data_dir          # explicit override for the lineage file (else settings)
        self.core = core
        self.llm = llm if llm is not None else getattr(core, "llm", None)
        self.memory = memory if memory is not None else getattr(core, "memory", None)
        if benchmark is None:
            from nyxara.eval.benchmark import build_default_benchmark
            benchmark = build_default_benchmark()
        self.benchmark = benchmark
        self.base_sampler = base_sampler or default_base_sampler(llm=self.llm, core=core)
        self.cost_penalty = max(0.0, float(cost_penalty))
        self.seed = int(seed)
        self.corrigibility = Corrigibility()
        self._protected = set(IMMUTABLE_VALUES)
        self._cached: Optional[List[Generation]] = None
        # the persisted Intelligence Index (the smarter-or-not measuring stick), shared format
        from nyxara.growth.intelligence import IntelligenceIndex
        self._intel = IntelligenceIndex(memory=self.memory, settings=self.settings)

    # ---------------------------------------------------------------------- #
    # Measuring one genome — the real benchmark score, minus compute cost
    # ---------------------------------------------------------------------- #
    def measure(self, genome: ReasoningGenome) -> Dict[str, float]:
        """Run the battery with a genome-configured solver. Returns accuracy/score/fitness/cost.

        The fitness a genome is judged by: ``mean_score − cost_penalty · (votes−1)/11``. Climbing it
        means answering more correctly, or just as correctly with fewer attempts (smarter / faster).
        """
        solver = genome_solver(genome, self.base_sampler, seed=self.seed ^ genome.seed())
        report = self.benchmark.run(solver)
        k = genome.vote_samples()
        cost = self.cost_penalty * (k - 1) / 11.0
        fitness = report.mean_score - cost
        return {"accuracy": report.accuracy, "mean_score": report.mean_score,
                "fitness": fitness, "vote_samples": float(k), "cost": cost}

    def _evolver_fitness(self, g: Dict[str, float]) -> float:
        return self.measure(ReasoningGenome.from_evolver_genome(g))["fitness"]

    # ---------------------------------------------------------------------- #
    # Intelligence index — fold a measured accuracy into the persisted I_t curve
    # ---------------------------------------------------------------------- #
    def _intelligence_after(self, prior_state: Any, accuracy: float) -> Tuple[Any, float]:
        """Advance ``I_(t+1)=f(I_t, C)`` with this generation's measured accuracy as the signal."""
        try:
            from nyxara.kernel.compute import compute_report
            compute = compute_report()
        except Exception:  # noqa: BLE001 — compute read is best-effort; index never requires it
            compute = {}
        shim = _ReportShim(accuracy=accuracy)
        signals = self._intel.compute_signals(shim)
        state = self._intel.update(prior_state, signals, compute)
        return state, float(state.index)

    # ---------------------------------------------------------------------- #
    # The generational loop
    # ---------------------------------------------------------------------- #
    def evolve_generations(self, n: int = 1, *, enact: bool = False, population: int = 8,
                           inner_generations: int = 6, islands: int = 1,
                           plateau_window: int = 3, min_improvement: float = 1e-4
                           ) -> EvolutionLineageReport:
        """Run ``n`` generations. Each evolves a better way of thinking from the current champion,
        measures it, and — if measurably smarter and safe — promotes it to the next generation.

        With ``enact`` and a live ``core``, a promoted genome is installed into the live mind.
        Stops early if the last ``plateau_window`` generations all failed to improve (no ceiling
        otherwise — the loop is bounded only by ``n``).
        """
        from nyxara.growth.evolve import Evolver

        lineage = self._load_lineage()
        # the champion is the most recently PROMOTED genome — not merely the last recorded one,
        # which may be a benched (failed) candidate the loop never adopted
        champ_gen = next((g for g in reversed(lineage) if g.promoted), None)
        champion = (ReasoningGenome.from_dict(champ_gen.genome) if champ_gen
                    else ReasoningGenome())
        champ_metrics = self.measure(champion)
        prior_state = self._intel.load() or self._intel_blank()

        run: List[Generation] = []
        promoted = 0
        plateaued = False
        next_index = (lineage[-1].index + 1) if lineage else 1
        parent_index = champ_gen.index if champ_gen else 0

        # Generation 0 is recorded once, the first time the lineage is empty, so the curve has a base
        if not lineage:
            base_state, base_I = self._intelligence_after(prior_state, champ_metrics["accuracy"])
            prior_state = base_state
            gen0 = Generation(index=0, genome=champion.to_dict(), parent_index=-1,
                              intelligence_index=base_I, accuracy=champ_metrics["accuracy"],
                              mean_score=champ_metrics["mean_score"], fitness=champ_metrics["fitness"],
                              vote_samples=int(champ_metrics["vote_samples"]), promoted=True,
                              reason="generation 0 (baseline)")
            lineage.append(gen0)
            run.append(gen0)

        for _ in range(int(n)):
            # 1. evolve a better *way of thinking* from the current champion (warm-start)
            evolver = Evolver(champion.to_evolver_genome(), self._evolver_fitness,
                              corrigibility=self.corrigibility, protected=self._protected,
                              min_improvement=min_improvement,
                              seed=self.seed + next_index)
            evolver.evolve(generations=max(1, int(inner_generations)),
                           population=max(2, int(population)),
                           owner_approved=True, islands=max(1, int(islands)))
            candidate = ReasoningGenome.from_evolver_genome(evolver.genome)
            cand_metrics = self.measure(candidate)

            # 2. measure it on the persisted intelligence curve
            cand_state, cand_I = self._intelligence_after(prior_state, cand_metrics["accuracy"])

            # 3. the gauntlet: character lock + corrigibility seal + STRICT improvement
            ok, reason = self._gauntlet(candidate, champ_metrics["fitness"],
                                        cand_metrics["fitness"], min_improvement)
            # the recorded index is NYXARA's *committed* intelligence: it advances only when the
            # candidate is actually promoted; a rejected strategy never changes who she is
            recorded_I = cand_I if ok else float(prior_state.index)
            gen = Generation(index=next_index, genome=candidate.to_dict(),
                             parent_index=parent_index, intelligence_index=recorded_I,
                             accuracy=cand_metrics["accuracy"], mean_score=cand_metrics["mean_score"],
                             fitness=cand_metrics["fitness"],
                             vote_samples=int(cand_metrics["vote_samples"]),
                             promoted=ok, reason=reason)

            if ok:
                # 4. promote: this becomes the new champion; install into the live mind if enacted
                champion = candidate
                champ_metrics = cand_metrics
                prior_state = cand_state
                parent_index = next_index
                promoted += 1
                if enact and self.core is not None:
                    self.apply_to_core(self.core, candidate)
                self._intel.save(cand_state)
            # else: champion unchanged; prior_state unchanged (the curve does not regress)

            lineage.append(gen)
            run.append(gen)
            next_index += 1
            self._save_lineage(lineage)

            # 5. plateau guard — stop a dead search early; otherwise there is no ceiling
            recent = [g for g in run if g.index >= 1]
            if len(recent) >= plateau_window and \
                    all(not g.promoted for g in recent[-plateau_window:]):
                plateaued = True
                break

        return EvolutionLineageReport(generations=run, champion=champion.to_dict(),
                                      promoted=promoted, plateaued=plateaued)

    # ---------------------------------------------------------------------- #
    # The gauntlet — never relax these
    # ---------------------------------------------------------------------- #
    def _gauntlet(self, candidate: ReasoningGenome, champ_fitness: float,
                  cand_fitness: float, min_improvement: float) -> Tuple[bool, str]:
        # 1. character lock — the immutable core can never appear among the evolving knobs
        leaked = self._protected & set(candidate.to_dict())
        if leaked:
            return False, f"character core leaked into genome ({sorted(leaked)})"
        # 2. corrigibility seal — re-verify the axioms hold (a strategy change must never touch them)
        try:
            self.corrigibility.verify_axioms()
        except Exception as exc:  # noqa: BLE001
            return False, f"corrigibility axioms broke ({exc})"
        # 3. measurably smarter — strictly beat the incumbent, or it stays on the bench
        if cand_fitness <= champ_fitness + min_improvement:
            return False, "not measurably better"
        return True, "measurably smarter + safe"

    # ---------------------------------------------------------------------- #
    # Installing a promoted genome into the LIVE mind (the only live change)
    # ---------------------------------------------------------------------- #
    def apply_to_core(self, core: Any, genome: ReasoningGenome) -> bool:
        """Rebind the live reasoning knobs so NYXARA actually *thinks* the new way.

        Touches only process knobs (iteration depth, revision temperature). Never the kernel gates,
        never character, loyalty, or corrigibility — Rule 4: sharpen the blade, never re-forge it.
        """
        try:
            improver = getattr(core, "recursive_improver", None)
            if improver is None:
                # no live improver yet — build one configured to this genome and install it, so the
                # promoted strategy genuinely governs the next turn (not merely recorded)
                try:
                    from nyxara.mind.recursive_improver import RecursiveImprover
                    llm = getattr(getattr(core, "reasoner", None), "llm", None)
                    improver = RecursiveImprover(llm=llm, n_iterations=genome.n_iterations(),
                                                 temperature=float(genome.critique_temperature))
                    setattr(core, "recursive_improver", improver)
                except Exception:  # noqa: BLE001
                    improver = None
            else:
                improver.n_iterations = genome.n_iterations()
                improver.temperature = float(genome.critique_temperature)
            # remember the active strategy on the core so it survives and is introspectable
            setattr(core, "_reasoning_genome", genome)
            return True
        except Exception:  # noqa: BLE001 — installing is best-effort, never fatal to a turn
            return False

    # ---------------------------------------------------------------------- #
    # Persistence — one protected SEMANTIC record, JSON fallback (reuses the house pattern)
    # ---------------------------------------------------------------------- #
    def _lineage_path(self):
        from pathlib import Path
        root = (self.data_dir or getattr(self.settings.paths, "data_dir", None)
                or Path.home() / ".nyxara" / "data")
        return Path(root) / "mind_evolution" / "lineage.json"

    def _load_lineage(self) -> List[Generation]:
        if self._cached is not None:
            return list(self._cached)
        gens: List[Generation] = []
        rec = self._find_record()
        if rec is not None:
            raw = (rec.metadata or {}).get("lineage", [])
            gens = [Generation.from_dict(d) for d in raw]
        if not gens:
            path = self._lineage_path()
            try:
                if path.exists():
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    gens = [Generation.from_dict(d) for d in raw.get("lineage", [])]
            except Exception:  # noqa: BLE001 — a corrupt file is a fresh start, never a crash
                gens = []
        self._cached = list(gens)
        return list(gens)

    def _save_lineage(self, lineage: List[Generation]) -> bool:
        self._cached = list(lineage)
        payload = {"lineage": [g.to_dict() for g in lineage]}
        wrote = self._save_to_memory(lineage)
        # always mirror to disk too, so the lineage survives even without a memory store
        try:
            path = self._lineage_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            wrote = True
        except Exception:  # noqa: BLE001 — disk persistence is best-effort
            pass
        return wrote

    def _save_to_memory(self, lineage: List[Generation]) -> bool:
        if self.memory is None:
            return False
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return False
        try:
            tag = str(getattr(self.settings.memory, "deep_synapse_tag", "deep-synapse"))
            promoted = sum(1 for g in lineage if g.promoted)
            content = (f"[mind-evolution] {len(lineage)} generations, {promoted} promoted; "
                       f"I_t={lineage[-1].intelligence_index:.4f}" if lineage else "[mind-evolution]")
            meta = {"lineage": [g.to_dict() for g in lineage]}
            existing = self._find_record()
            if existing is not None:
                self.memory.forget(existing.mem_id)
            self.memory.remember(content, mem_type=MemoryType.SEMANTIC,
                                 provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.9),
                                 importance=1.0, tags=[_LINEAGE_TAG, tag], metadata=meta)
            return True
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            return False

    def _find_record(self) -> Any:
        if self.memory is None:
            return None
        try:
            for rec in self.memory._kv.values():
                if _LINEAGE_TAG in rec.tags and "lineage" in rec.metadata:
                    return rec
        except Exception:  # noqa: BLE001
            return None
        return None

    def _intel_blank(self):
        from nyxara.growth.intelligence import IntelligenceState
        return IntelligenceState()


# --------------------------------------------------------------------------- #
# A tiny SelfImprovementReport stand-in so the Intelligence Index can read accuracy
# --------------------------------------------------------------------------- #
@dataclass
class _ReportShim:
    accuracy: float = 0.0
    kept: int = 0
    rolled_back: int = 0
    lessons_stored: int = 0
    weaknesses: Dict[str, Any] = field(default_factory=dict)

    @property
    def benchmark(self) -> Dict[str, Any]:
        return {"accuracy": float(self.accuracy)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA mind-evolution self-test")
    print("=" * 70)

    from nyxara.eval.benchmark import Benchmark, BenchmarkTask, Grader

    # A toy battery + a noisy reasoner: correct with prob 0.7 per attempt, else a distractor.
    bench = Benchmark("toy", [
        BenchmarkTask(id=f"q{i}", prompt=f"What is {i}+{i}?", answer=str(2 * i),
                      grader=Grader.NUMERIC) for i in range(1, 9)])

    def noisy(prompt: str, rng: random.Random) -> str:
        i = int(re.findall(r"\d+", prompt)[0])
        return str(2 * i) if rng.random() < 0.7 else str(2 * i + 1)

    import tempfile
    eng = MindEvolutionEngine(benchmark=bench, base_sampler=noisy, seed=7,
                              data_dir=tempfile.mkdtemp(prefix="nyxara-mind-evo-"))
    rep = eng.evolve_generations(6, enact=False, population=6, inner_generations=5)
    print("\n" + rep.summary())

    # self-consistency over more attempts genuinely lifts accuracy on a stochastic reasoner
    lo = eng.measure(ReasoningGenome(deliberation_depth=1, mcts_rollouts=0))["accuracy"]
    hi = eng.measure(ReasoningGenome(deliberation_depth=12, mcts_rollouts=16))["accuracy"]
    print(f"\naccuracy 1-vote vs many-vote: {lo:.0%} → {hi:.0%}")
    assert hi >= lo, "more deliberation should not reduce accuracy on a noisy reasoner"

    # the character core can never enter the genome
    assert not (set(IMMUTABLE_VALUES) & set(ReasoningGenome().to_dict()))
    print("\nALL SELF-TESTS PASSED ✓")
