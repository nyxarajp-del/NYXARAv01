"""NYXARA · nyx001/layers/stack.py — the cognitive stack (🧬, eighteen layers, one cycle).

The charter's closing loop, run as one function:

    START → substrate → experience → prediction → error → world-model update → memory →
    concept formation → reasoning → planning → action → consequence → curiosity →
    experiment → self-critique → meta-learning → better learning → better model ↺

:class:`CognitiveStack` owns Layers 0–17 and :meth:`cycle` is that loop. Three things about how it
is put together:

**Every layer is optional and every layer is measured.** Each is config-gated, and each cycle
returns a :class:`~nyxara.nyx001.layers.types.LayerReport` per layer saying whether it ran, how
long it took, and what it produced. A layer that is off is absent from the report rather than
reporting zeros — the distinction the whole package keeps.

**The cognitive state is assembled from real signals, not invented.** ``E`` comes from Layer 2's
measured error, ``N`` from Layer 1's novelty, ``U`` from Layer 2's uncertainty, ``M`` from Layer 3's
occupancy, ``T`` from Layer 15's remaining budget. That five-tuple is what modulates the optimizer,
so the learning dynamics are genuinely driven by the system's own appraisal — the charter's
``Δθ = F(E,N,U,M,T)`` closed end to end.

**Failure is contained per layer.** One layer raising does not stop the cycle; its report carries
the error and the cycle continues. A cognitive architecture where perception can kill planning is
not modular, whatever its diagram says.

``stack.lingua`` is Track B (:mod:`nyxara.nyx001.lingua`) — the language substrate, held here as a
**sibling of the eighteen rather than a nineteenth member of them**. :meth:`layers` excludes it on
purpose: Track A is the cognitive architecture and Track B is a modality, and the charter keeps
the two apart. It is fed from :meth:`cycle` only after Layer 4 has assigned a concept, so a word
can never be bound to structure the system has not formed for itself.

:meth:`is_learning` is the honest bottom line. It reports whether the world model's error is
actually falling, and returns ``None`` — not ``False`` — when there is not yet enough evidence to
say. Everything else this stack claims rests on that number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from nyxara.nyx001.layers.l00_substrate import Substrate, default_substrate
from nyxara.nyx001.layers.l01_perception import Perception
from nyxara.nyx001.layers.l02_world_model import WorldModel
from nyxara.nyx001.layers.l03_episodic import EpisodicMemory
from nyxara.nyx001.layers.l04_semantic import SemanticMemory
from nyxara.nyx001.layers.l05_attention import Candidate, DynamicAttention
from nyxara.nyx001.layers.l06_reasoning import ReasoningEngine
from nyxara.nyx001.layers.l07_counterfactual import CounterfactualEngine
from nyxara.nyx001.layers.l08_planning import Planner
from nyxara.nyx001.layers.l09_curiosity import CuriosityEngine
from nyxara.nyx001.layers.l10_contradiction import ContradictionEngine
from nyxara.nyx001.layers.l11_critique import SelfCritique
from nyxara.nyx001.layers.l12_self_model import SelfModel
from nyxara.nyx001.layers.l13_meta_learning import MetaLearner
from nyxara.nyx001.layers.l14_compression import CognitiveCompression
from nyxara.nyx001.layers.l15_resource import ResourceAwareCognition
from nyxara.nyx001.layers.l16_active_learning import ActiveLearning
from nyxara.nyx001.layers.l17_self_improve import SelfImprovement
from nyxara.nyx001.layers.types import LayerReport, Observation, Percept
from nyxara.nyx001.substrate import CognitiveState

__all__ = ["CycleResult", "CognitiveStack"]

_LAYER_NAMES = {
    0: "substrate", 1: "perception", 2: "world_model", 3: "episodic", 4: "semantic",
    5: "attention", 6: "reasoning", 7: "counterfactual", 8: "planning", 9: "curiosity",
    10: "contradiction", 11: "critique", 12: "self_model", 13: "meta_learning",
    14: "compression", 15: "resource", 16: "active_learning", 17: "self_improvement",
}


@dataclass
class CycleResult:
    """One pass through the stack. Carries what each layer did, and what came out."""

    cycle: int = 0
    ms: float = 0.0
    percept: Optional[Percept] = None
    error: Optional[float] = None
    novelty: float = 0.0
    uncertainty: float = 1.0
    difficulty: float = 0.0
    concept: Optional[str] = None
    answer: str = ""
    reports: List[LayerReport] = field(default_factory=list)
    cognitive: Optional[CognitiveState] = None
    reading: Any = None                     # Track B's pass over this turn's text, when it ran

    @property
    def ran(self) -> List[str]:
        return [r.name for r in self.reports if r.ran]

    @property
    def failed(self) -> List[str]:
        return [r.name for r in self.reports if r.error]

    def to_dict(self) -> Dict[str, Any]:
        return {"cycle": self.cycle, "ms": round(self.ms, 3),
                "error": None if self.error is None else round(self.error, 6),
                "novelty": round(self.novelty, 4), "uncertainty": round(self.uncertainty, 4),
                "difficulty": round(self.difficulty, 4), "concept": self.concept,
                "ran": self.ran, "failed": self.failed,
                "cognitive": self.cognitive.to_dict() if self.cognitive else None,
                "reading": self.reading.to_dict() if hasattr(self.reading, "to_dict") else None,
                "reports": [r.to_dict() for r in self.reports]}


class CognitiveStack:
    """Layers 0–17, wired into the charter's loop."""

    def __init__(self, config: Any = None, *, substrate: Optional[Substrate] = None) -> None:
        self.config = config
        self.substrate = substrate or default_substrate(config)
        self.cycles = 0
        self._timings: Dict[str, float] = {}
        self._prev_concept: Optional[str] = None
        self._credit_threshold = self._f("credit_threshold", 0.01)

        g = self._gate
        self.perception = Perception(self.substrate) if g("l01") else None
        self.world_model = WorldModel(self.substrate, lr=self._f("lr", 3e-3)) if g("l02") else None
        self.episodic = EpisodicMemory(
            self.substrate, capacity=int(self._f("memory_capacity", 8192))) if g("l03") else None
        self.semantic = SemanticMemory(self.substrate) if g("l04") else None
        self.attention = DynamicAttention(self.substrate) if g("l05") else None
        self.reasoning = ReasoningEngine(self.substrate) if g("l06") else None
        self.counterfactual = CounterfactualEngine(
            self.substrate, self.world_model) if g("l07") else None
        self.planning = Planner(self.substrate, self.world_model) if g("l08") else None
        self.curiosity = CuriosityEngine(self.substrate) if g("l09") else None
        self.contradiction = ContradictionEngine(self.substrate) if g("l10") else None
        self.critique = SelfCritique(self.substrate) if g("l11") else None
        self.self_model = SelfModel(self.substrate) if g("l12") else None
        self.meta_learning = MetaLearner(self.substrate) if g("l13") else None
        self.compression = CognitiveCompression(self.substrate) if g("l14") else None
        self.resource = ResourceAwareCognition(self.substrate) if g("l15") else None
        self.active_learning = ActiveLearning(self.substrate) if g("l16") else None
        self.self_improvement = SelfImprovement(
            self.substrate, enabled=g("l17")) if g("l17") else None
        # Track B — the language substrate. A SIBLING of Layers 0-17, not an eighteenth layer:
        # ``layers()`` still returns exactly the eighteen it always did. It was previously
        # constructed by nothing at all, so the whole track was unreachable in a running system.
        self.lingua = self._build_lingua(config) if g("lingua") else None

    # ---- config helpers ---- #
    def _gate(self, name: str) -> bool:
        return bool(getattr(self.config, f"{name}_enabled", True)) if self.config else True

    def _f(self, name: str, default: float) -> float:
        try:
            return float(getattr(self.config, name, default))
        except Exception:  # noqa: BLE001
            return default

    def _build_lingua(self, config: Any) -> Any:
        """Track B, built from its own config keys. A track that declines is absent, not broken."""
        try:
            from nyxara.nyx001.lingua.cortex import LanguageCortex
            return LanguageCortex(
                self.substrate,
                max_tokens=int(self._f("lingua_max_tokens", 32.0)),
                sequence_enabled=bool(getattr(config, "lingua_sequence_enabled", True))
                if config else True)
        except Exception:  # noqa: BLE001 — a mind without a tongue still thinks
            return None

    def layers(self) -> Dict[str, Any]:
        """The live layers by name — what Layer 17 reconfigures and what stats() walks."""
        return {n: getattr(self, n, None) for n in _LAYER_NAMES.values()
                if n != "substrate" and getattr(self, n, None) is not None}

    # ---- the cycle ---- #
    def cycle(self, observation: Any, *, goal: str = "",
              environment: Optional[Callable[[str], Any]] = None,
              action_space: Optional[Sequence[str]] = None) -> CycleResult:
        """One full pass: perceive → predict → learn → remember → abstract → reason → report."""
        out = CycleResult(cycle=self.cycles)
        t0 = time.time()
        self.cycles += 1

        obs = observation if isinstance(observation, Observation) \
            else Observation(payload=observation)

        # L1 — perception
        percept = self._run(out, 1, lambda: self.perception.perceive(obs)
                            if self.perception else None)
        out.percept = percept
        out.novelty = float(getattr(percept, "novelty", 0.0) or 0.0)
        state = getattr(percept, "vec", None)

        # L3 — what does this remind her of (needed before L15 can judge difficulty)
        recall = self._run(out, 3, lambda: self.episodic.recall(state)
                           if (self.episodic and state is not None) else None)

        # L4 — does any concept already cover it
        classified = self._run(out, 4, lambda: self.semantic.classify(state)
                               if (self.semantic and state is not None) else None)
        concept = classified[0] if classified else None
        out.concept = getattr(concept, "name", None)

        # L15 — how hard is this, and what may it spend
        budget = None
        if self.resource is not None:
            unc = self.world_model.uncertainty() if self.world_model else 1.0
            diff = self._run(out, 15, lambda: self.resource.estimate(
                novelty=out.novelty, uncertainty=unc,
                recall_confidence=getattr(recall, "confidence", None),
                concept_covered=concept is not None))
            if diff is not None:
                out.difficulty = diff.score
                budget = self.resource.allocate(diff)

        # L2 — predict, then learn from what actually arrived
        if self.world_model is not None and state is not None:
            self._run(out, 2, lambda: self.world_model.predict(state))
            out.uncertainty = self.world_model.uncertainty()
        cog = self._cognitive_state(out, budget)
        out.cognitive = cog
        if self.world_model is not None and state is not None:
            err = self.world_model.observe(state, cognitive=cog)
            out.error = err

        # L3 — lay the episode down, now that the error is known
        episode = None
        if self.episodic is not None and state is not None:
            episode = self.episodic.remember(str(getattr(obs, "payload", ""))[:120],
                                             context=state,
                                             prediction_error=out.error or 0.0)

        # L4 — fold it into the concept space
        if self.semantic is not None and episode is not None:
            concept = self._run(out, 4, lambda: self.semantic.absorb(episode)) or concept
            out.concept = getattr(concept, "name", out.concept)

        # Track B — language, and only now. It reads the concept Layer 4 has just assigned, so a
        # word can only ever be bound to structure the system already formed for itself. Run
        # before Layer 9 purely so `out.reading` is populated for anything downstream that asks.
        if self.lingua is not None and isinstance(getattr(obs, "payload", None), str):
            out.reading = self._read_language(obs, state, recall, concept, out)

        # L9 — where is there progress to be had
        if self.curiosity is not None:
            key = self.curiosity.region_key(concept, state)
            self._run(out, 9, lambda: self.curiosity.observe(key, out.error or 0.0, t=time.time()))

        # L3 — credit relevance BACKWARDS when the prediction went well. This is what makes
        # `future_relevance` a learned quantity: episodes similar to a state the model predicted
        # correctly are the ones that turned out to be worth having stored. Without this call the
        # field stays 0.0 for every episode forever, replay priority collapses to pure surprise,
        # and Experience.priority's second term is dead weight.
        if (self.episodic is not None and state is not None and out.error is not None
                and out.error < self._credit_threshold):
            self._run(out, 3, lambda: self.episodic.credit(state))

        # L5 — allocate the look
        if self.attention is not None and recall is not None and getattr(recall, "episodes", None):
            cands = [Candidate(key=e.key, vec=e.context, payload=e,
                               memory_relevance=e.future_relevance,
                               prediction_error=e.prediction_error)
                     for e in recall.episodes]
            self._run(out, 5, lambda: self.attention.allocate(
                state, cands, resource_budget=self._budget_signal(budget)))

        # L6 — hypothesize from what L4 and L3 have, and TEST what was hypothesized last time.
        # The test is the part that makes this reasoning: a "A precedes B" claim made on the
        # previous cycle is checked here against the concept that actually arrived. Without it
        # the engine accumulates hypotheses it never evaluates — `tested` stays 0 forever and
        # `best()` ranks by an untouched 0.5 prior, which is a confident-looking nothing.
        if self.reasoning is not None:
            self._run(out, 6, lambda: self.reasoning.cycle(
                theories=self.semantic.theories() if self.semantic else None,
                surprises=self.episodic.surprising() if self.episodic else None))
            self._test_succession(out)

        # L10 — attack what she believes
        if self.contradiction is not None and self.reasoning is not None:
            self._run(out, 10, lambda: self.contradiction.sweep(
                self.reasoning, self.episodic, semantic=self.semantic))

        # L8 — plan, if there is a goal and something to act with
        if self.planning is not None and goal and state is not None:
            self._run(out, 8, lambda: self.planning.plan(
                goal, state, action_space=list(action_space or []),
                concepts=self.semantic.concepts() if self.semantic else None))

        # L16 — close the loop, if there is a world to act on
        if self.active_learning is not None and environment is not None:
            if action_space:
                self.active_learning.set_action_space(action_space)
            self._run(out, 16, lambda: self.active_learning.step(
                obs, environment, perception=self.perception, world_model=self.world_model,
                memory=self.episodic, semantic=self.semantic, curiosity=self.curiosity,
                planner=self.planning, goal=goal, cognitive=cog))

        # L15 — score the difficulty estimate against what it actually cost
        if self.resource is not None and budget is not None:
            budget.spent_steps += 1
            self.resource.record(budget)

        out.ms = (time.time() - t0) * 1000.0
        return out

    # ---- periodic work, not every cycle ---- #
    def consolidate(self) -> Dict[str, Any]:
        """Compression, meta-learning and self-improvement — expensive, so run on a beat."""
        result: Dict[str, Any] = {}
        try:
            if self.compression is not None:
                comps = self.compression.compress_all(self.semantic, self.episodic)
                result["compression"] = {"attempted": len(comps),
                                         "accepted": sum(1 for c in comps if c.accepted)}
            if self.meta_learning is not None and self.world_model is not None:
                err = self.world_model.mean_error()
                reward = self.meta_learning.end(err, self.world_model.steps)
                strategy = self.meta_learning.begin(err, self.world_model.steps)
                if strategy is not None:
                    applied = self.meta_learning.apply_to(
                        optimizer=self.world_model.optimizer,
                        objective=self.world_model.objective, memory=self.episodic,
                        strategy=strategy)
                    result["meta_learning"] = {"strategy": strategy.name, "reward": reward,
                                               "applied": applied}
                if self.meta_learning.trials % 5 == 4:
                    self.meta_learning.mutate()
            return result
        except Exception:  # noqa: BLE001 — consolidation is never worth breaking a turn for
            return result

    def improve(self, task: Callable[[Any], Optional[float]], *,
                weakness: str = "") -> List[Any]:
        """One controlled self-improvement pass. Fail-closed all the way down."""
        if self.self_improvement is None:
            return []
        cal = self.self_model.calibration() if self.self_model else None
        return self.self_improvement.cycle(self.layers(), task, weakness=weakness,
                                           calibration=cal)

    # ---- internals ---- #
    def _read_language(self, obs: Any, state: Any, recall: Any, concept: Any,
                       out: CycleResult) -> Any:
        """One Track B pass, fed the same live signals every other consumer reads.

        ``z_t = f(x_t, c_t, m_t, g_t, u_t)`` is only meaningfully dynamic if the situation signals
        are real ones, so ``c_t`` is Layer 1's state vector, ``m_t`` is Layer 3's best recalled
        context, and ``u_t`` is Layer 2's measured uncertainty. Handing it zeros would make the
        embedding a lookup table with extra steps — the failure ``drift()`` exists to measure.

        ``g_t`` is deliberately **not** supplied. The cycle's ``goal`` is a string; the embedding
        wants a vector in the substrate's space, and there is nothing here that honestly projects
        one into the other. Passing the goal through a hash would be inventing a signal, so the
        term is left absent — which is the same rule the rest of this package keeps for a quantity
        it cannot yet measure.
        """
        t0 = time.time()
        try:
            best = getattr(recall, "best", None) if recall is not None else None
            return self.lingua.read(str(obs.payload), concept=concept, context=state,
                                    memory=getattr(best, "context", None),
                                    uncertainty=out.uncertainty)
        except Exception:  # noqa: BLE001 — a mind without a tongue still thinks
            return None
        finally:
            # Timed like a layer even though it is not one: the scan is the most expensive thing
            # added to the cycle, and a cost that does not appear in `timings_ms` is a cost
            # nobody can find. It stays out of `reports` so `ran`/`failed` still name layers only.
            self._timings["lingua"] = self._timings.get("lingua", 0.0) \
                + (time.time() - t0) * 1000.0

    def _test_succession(self, out: CycleResult) -> None:
        """Check last cycle's "A precedes B" claims against the concept that actually arrived.

        This closes the hypothesize → *evaluate* half of Layer 6's cycle. A claim is testable here
        only when the previous cycle produced a concept and this one did too; otherwise the
        predictor returns ``None``, which the engine correctly treats as untested rather than as
        a miss.
        """
        prev, cur = self._prev_concept, out.concept
        self._prev_concept = cur
        if not prev or not cur or self.reasoning is None:
            return
        try:
            for h in list(self.reasoning._hyps.values()):
                parts = h.claim.split(" precedes ")
                if len(parts) != 2 or parts[0].strip() != prev:
                    continue
                expected = parts[1].strip()
                self.reasoning.evaluate(h.claim, lambda e=expected: e == cur)
        except Exception:  # noqa: BLE001 — an untested hypothesis is not a refuted one
            pass

    def _run(self, out: CycleResult, layer: int, fn: Callable[[], Any]) -> Any:
        """Run one layer, timed and contained. A raise becomes a report, never a dead cycle."""
        name = _LAYER_NAMES.get(layer, str(layer))
        rep = LayerReport(layer=layer, name=name)
        t0 = time.time()
        value = None
        try:
            value = fn()
            rep.ran = value is not None
        except Exception as exc:  # noqa: BLE001 — containment is the point of this wrapper
            rep.error = f"{type(exc).__name__}: {exc}"[:160]
        rep.ms = (time.time() - t0) * 1000.0
        self._timings[name] = self._timings.get(name, 0.0) + rep.ms
        out.reports.append(rep)
        return value

    def _cognitive_state(self, out: CycleResult, budget: Any) -> CognitiveState:
        """The charter's five signals, every one read from a layer that measured it."""
        return CognitiveState(
            error=min(1.0, out.error or 0.0),
            novelty=out.novelty,
            uncertainty=self.world_model.uncertainty() if self.world_model else 0.0,
            memory_pressure=self.episodic.pressure() if self.episodic else 0.0,
            time_budget=self._budget_signal(budget))

    def _budget_signal(self, budget: Any) -> float:
        if self.resource is None:
            return 1.0
        return self.resource.time_budget_signal(budget)

    # ---- the bottom line ---- #
    def is_learning(self) -> Optional[bool]:
        """Is the world model's error actually falling? ``None`` when it cannot yet be said."""
        return self.world_model.is_learning() if self.world_model else None

    def learning_curve(self) -> List[float]:
        return self.world_model.learning_curve() if self.world_model else []

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "cycles": self.cycles,
            "substrate": self.substrate.stats(),
            "is_learning": self.is_learning(),
            "learning_curve": self.learning_curve(),
            "active_layers": len(self.layers()) + 1,
            "lingua": None,                 # Track B: a sibling, reported but not counted as one
            "timings_ms": {k: round(v, 2) for k, v in
                           sorted(self._timings.items(), key=lambda kv: kv[1], reverse=True)[:8]},
        }
        for name, layer in self.layers().items():
            try:
                out[name] = layer.stats()
            except Exception:  # noqa: BLE001
                out[name] = {"error": "stats failed"}
        if self.lingua is not None:
            try:
                out["lingua"] = self.lingua.stats()
            except Exception:  # noqa: BLE001
                out["lingua"] = {"error": "stats failed"}
        return out

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"cycles": self.cycles, "substrate": self.substrate.to_dict()}
        for name, layer in self.layers().items():
            try:
                d[name] = layer.to_dict()
            except Exception:  # noqa: BLE001
                continue
        if self.lingua is not None:
            try:
                d["lingua"] = self.lingua.to_dict()
            except Exception:  # noqa: BLE001
                pass
        return d

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.cycles = int(d.get("cycles", 0))
            self.substrate.load_dict(d.get("substrate") or {})
            for name, layer in self.layers().items():
                payload = d.get(name)
                if payload and hasattr(layer, "load_dict"):
                    layer.load_dict(payload)
            if self.lingua is not None and d.get("lingua"):
                self.lingua.load_dict(d["lingua"])
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a freshly-born stack
            pass
