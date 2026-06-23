"""NYXARA · growth/autolearn.py — the automatic learning loop (📈, Rule 4).

Reflection, consolidation and the model foundry already exist as faculties — but until
now they had to be *invoked by hand*, so NYXARA's behaviour never actually improved on
its own. This module closes that loop: a :class:`GrowthEngine` that reads NYXARA's own
recent experience (her journal of actions + her long-term memory) and turns it into
durable change, on a cadence, **without weakening any boundary**:

* **Reflect** — journalled actions become :class:`~nyxara.growth.reflect.Episode` records;
  the :class:`~nyxara.growth.reflect.Reflector` mines them into *lessons*, which are
  written back into semantic memory (deduped) so future recall surfaces what worked and
  what didn't.
* **Consolidate** — the :class:`~nyxara.memory.consolidation.Consolidator` replays,
  abstracts and gently forgets, so memory compresses experience instead of hoarding it.
* **Forge** (opt-in, gated) — when ``settings.foundry.enabled`` is set, NYXARA's own model
  is retrained from the freshened corpus and promoted **only if it passes the foundry's
  gauntlet** (corrigibility + character lock + measured improvement). Off by default.

It changes weights and memories, never rules: protected values/tags are off-limits, and
nothing here can touch the kernel's gates. Designed to be driven either on demand or by
the background :class:`~nyxara.kernel.autonomic.AutonomicLoop`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = ["GrowthEngine", "GrowthReport"]


@dataclass
class GrowthReport:
    """A summary of one growth pass."""

    lessons: List[Dict[str, Any]] = field(default_factory=list)
    lessons_stored: int = 0
    episodes_seen: int = 0
    replayed: int = 0
    abstractions: int = 0
    forgotten: int = 0
    selfplay: Optional[Dict[str, Any]] = None
    foundry: List[Dict[str, Any]] = field(default_factory=list)
    self_improvement: Optional[Dict[str, Any]] = None
    mind_evolution: Optional[Dict[str, Any]] = None
    meta_research: Optional[Dict[str, Any]] = None
    rivalry: Optional[Dict[str, Any]] = None
    adversarial: Optional[Dict[str, Any]] = None
    metaprompt_insights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"episodes_seen": self.episodes_seen, "lessons": self.lessons,
                "lessons_stored": self.lessons_stored, "replayed": self.replayed,
                "abstractions": self.abstractions, "forgotten": self.forgotten,
                "selfplay": self.selfplay, "foundry": self.foundry,
                "self_improvement": self.self_improvement,
                "mind_evolution": self.mind_evolution,
                "meta_research": self.meta_research, "rivalry": self.rivalry,
                "adversarial": self.adversarial,
                "metaprompt_insights": self.metaprompt_insights}


class GrowthEngine:
    """Turns recent experience into durable improvement — reflect, consolidate, (forge)."""

    def __init__(self, *, memory: Any = None, journal: Any = None, core: Any = None,
                 settings: Optional[NyxaraSettings] = None, reflector: Any = None,
                 consolidator: Any = None, foundry: Any = None,
                 enable_foundry: Optional[bool] = None,
                 enable_selfplay: bool = False, selfplay_n: int = 6,
                 selfplay: Any = None, enable_self_improvement: Optional[bool] = None,
                 self_improvement_every: Optional[int] = None,
                 self_improver: Any = None, enable_meta_research: Optional[bool] = None,
                 meta_research_every: Optional[int] = None,
                 meta_researcher: Any = None, frontier: Any = None,
                 enable_rivalry: bool = False, arena: Any = None,
                 rival_solver: Any = None, self_solver: Any = None,
                 metaprompt: Any = None, skill_memory: Any = None,
                 enable_metaprompt: Optional[bool] = None,
                 enable_adversarial: bool = False,
                 adversarial_rounds: int = 60) -> None:
        self.settings = settings or get_settings()
        self.memory = memory
        self.journal = journal
        # the live orchestrator handle (when built via from_core) — lets the RSIE's intelligence
        # directive reach the foundry + research/investigation queues to actually *act*, not just
        # log. None on a bare engine; every directive dispatch then degrades to a safe no-op.
        self.core = core
        self._reflector = reflector
        self._consolidator = consolidator
        self._foundry = foundry
        self.enable_foundry = (self.settings.foundry.enabled
                               if enable_foundry is None else enable_foundry)
        # Phase 3: curiosity-driven self-play manufactures fresh training data before forging.
        self.enable_selfplay = enable_selfplay
        self.selfplay_n = max(1, selfplay_n)
        self._selfplay = selfplay
        # Recursive self-improvement: review/architecture/benchmark/weakness/optimise, throttled
        # to run only every N growth passes (it is heavier than reflection).
        self.enable_self_improvement = (self.settings.self_improvement.enabled
                                        if enable_self_improvement is None
                                        else enable_self_improvement)
        self.self_improvement_every = max(1, int(
            self.settings.self_improvement.self_improvement_every
            if self_improvement_every is None else self_improvement_every))
        self._self_improver = self_improver
        # Recursive mind-evolution: evolve the *way of thinking* itself, on its own (slowest)
        # cadence — it benchmarks the whole reasoner each pass, so it is the heaviest loop.
        self.enable_mind_evolution = bool(self.settings.mind_evolution.enabled)
        self.mind_evolution_every = max(1, int(self.settings.mind_evolution.every))
        self._mind_evolver = None
        # Meta-research: invent → test → (gated) integrate, on its own (slow) cadence.
        self.enable_meta_research = (self.settings.meta_research.enabled
                                     if enable_meta_research is None else enable_meta_research)
        self.meta_research_every = max(1, int(
            self.settings.meta_research.meta_research_every
            if meta_research_every is None else meta_research_every))
        self._meta_researcher = meta_researcher
        self._growth_passes = 0
        self._seen_action_seqs: set = set()
        self._stored_lessons: set = set()
        self._meta_topic_n = 0
        # Pillar F · Edge 1+4: an open-ended frontier steers curiosity (selfplay) toward the
        # sparsest niches; a rivalry arena turns a peer AGI's lead into targeted curiosity topics.
        # All optional/default-off — behaviour is unchanged unless the Master wires them.
        self.frontier = frontier
        self.enable_rivalry = enable_rivalry
        self._arena = arena
        self._rival_solver = rival_solver
        self._self_solver = self_solver
        # Continuous metaprompt distillation: compress her own successful chains into operating
        # heuristics and inject them into the reasoner's prompt (recursive self-improvement).
        self._metaprompt = metaprompt
        self._skill_memory = skill_memory
        self.enable_metaprompt = (self.settings.metaprompt.enabled
                                  if enable_metaprompt is None else enable_metaprompt)
        # Adversarial self-play (clone-vs-clone) on the growth cadence (off unless asked, since
        # a full contest is heavy); when enabled the hardest categories steer the next self-play.
        self.enable_adversarial = bool(enable_adversarial)
        self.adversarial_rounds = max(1, int(adversarial_rounds))

    @classmethod
    def from_core(cls, core: Any, **kw: Any) -> "GrowthEngine":
        """Build a growth engine bound to a :class:`~nyxara.kernel.orchestrator.NyxaraCore`."""
        kw.setdefault("metaprompt", getattr(core, "metaprompt", None))
        kw.setdefault("skill_memory", getattr(core, "skills", None))
        return cls(memory=getattr(core, "memory", None),
                   journal=getattr(core, "journal", None), core=core, **kw)

    # ---- reflect ---- #
    def _reflector_obj(self):
        if self._reflector is None:
            from nyxara.growth.reflect import Reflector
            self._reflector = Reflector()
        return self._reflector

    def _episodes_from_journal(self) -> List[Any]:
        from nyxara.growth.reflect import Episode
        from nyxara.planning.journal import ActionStatus
        if self.journal is None:
            return []
        episodes: List[Any] = []
        for entry in self.journal.actions():
            if entry.seq in self._seen_action_seqs:
                continue
            self._seen_action_seqs.add(entry.seq)
            payload = entry.payload
            action = str(payload.get("action", ""))
            status = self.journal.latest_status(entry.action_id)
            success = status is ActionStatus.SUCCEEDED
            failed = status is ActionStatus.FAILED
            reward = 1.0 if success else (-0.5 if failed else 0.0)
            verb = action.split(":", 1)[0].split()[0].lower() if action else "act"
            tags = [verb, "autonomous" if payload.get("autonomous") else "owner",
                    str(payload.get("decision", "act"))]
            episodes.append(Episode(action=action or verb, success=success, reward=reward,
                                    tags=tags, rationale=str(payload.get("rationale", "")),
                                    at=0.0))
        return episodes

    def reflect(self) -> List[Any]:
        episodes = self._episodes_from_journal()
        if not episodes:
            return []
        reflector = self._reflector_obj()
        reflector.ingest(episodes)
        lessons = reflector.lessons()
        if self.memory is not None:
            self._store_lessons(lessons)
        return lessons

    def _store_lessons(self, lessons: List[Any]) -> int:
        from nyxara.memory.provenance import Provenance, SourceType
        from nyxara.memory.store import MemoryType
        stored = 0
        for lesson in lessons:
            key = (lesson.kind.value, lesson.subject)
            if key in self._stored_lessons:
                continue
            self._stored_lessons.add(key)
            try:
                self.memory.remember(
                    f"Lesson [{lesson.kind.value}] {lesson.subject}: {lesson.text}",
                    mem_type=MemoryType.SEMANTIC,
                    provenance=Provenance(SourceType.SELF_REFLECTION,
                                          confidence=lesson.confidence),
                    importance=min(1.0, 0.5 + 0.3 * lesson.confidence),
                    tags=["lesson", lesson.kind.value])
                stored += 1
            except Exception:  # noqa: BLE001
                pass
        return stored

    # ---- consolidate ---- #
    def _consolidator_obj(self):
        if self._consolidator is None and self.memory is not None:
            from nyxara.memory.consolidation import Consolidator
            self._consolidator = Consolidator(self.memory)
        return self._consolidator

    def consolidate(self):
        cons = self._consolidator_obj()
        if cons is None:
            return None
        try:
            return cons.run_cycle()
        except Exception:  # noqa: BLE001
            return None

    # ---- forge (opt-in, gated) ---- #
    def _memory_corpus(self, *, max_items: int = 2000) -> List[str]:
        """Harvest training text from NYXARA's own lived memory (semantic + episodic)."""
        if self.memory is None:
            return []
        texts: List[str] = []
        try:
            for rec in self.memory._kv.values():  # all records, regardless of recency
                t = rec.text().strip()
                if t:
                    texts.append(t)
        except Exception:  # noqa: BLE001
            return []
        return texts[:max_items]

    # ---- self-play (opt-in, manufactures fresh corpus before forging) ---- #
    def self_play(self, *, n: Optional[int] = None,
                  topics: Optional[List[str]] = None) -> Optional[dict]:
        """Run one curiosity round — invent questions, distil the teacher, grow the corpus.

        When a frontier is wired, curiosity is steered to the sparsest niches (Edge 1); when
        ``topics`` are supplied (e.g. from a rivalry pass, Edge 4) they override that, so she
        practises exactly where a rival leads.
        """
        try:
            if self._selfplay is None:
                from nyxara.growth.selfplay import SelfPlay
                self._selfplay = SelfPlay(settings=self.settings, frontier=self.frontier)
            return self._selfplay.play(n or self.selfplay_n, topics=topics)
        except Exception:  # noqa: BLE001 — self-play is best-effort, never fatal
            return None

    # ---- rivalry (opt-in, Edge 4): turn a peer AGI's lead into targeted curiosity ---- #
    def _arena_obj(self):
        if self._arena is None:
            from nyxara.growth.rivalry import Arena
            self._arena = Arena(settings=self.settings)
        return self._arena

    def rivalry(self) -> Optional[Dict[str, Any]]:
        """Run one gated head-to-head vs the configured rival; route the gaps into growth.

        Returns the out-compete report (incl. ``topics`` to practise). No-op — and never fatal —
        unless ``enable_rivalry`` is set *and* a rival solver is configured. Pure measurement +
        self-direction: it acts on no external system.
        """
        if not self.enable_rivalry or self._rival_solver is None:
            return None
        try:
            self_solver = self._self_solver
            if self_solver is None:
                from nyxara.eval.benchmark import self_solver as build_self_solver
                self_solver = build_self_solver(settings=self.settings)
            out = self._arena_obj().out_compete(self_solver, self._rival_solver)
            self._store_rivalry_lessons(out)
            return out
        except Exception:  # noqa: BLE001 — rivalry is best-effort, never fatal
            return None

    def _store_rivalry_lessons(self, out: Dict[str, Any]) -> None:
        """Persist the rival profile + the worst gap as a durable, deduped lesson."""
        if self.memory is None:
            return
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
            rival = out.get("rival", {})
            gaps = (out.get("head_to_head", {}) or {}).get("gaps", []) or []
            if not gaps:
                return
            worst = gaps[0]
            key = ("rivalry", rival.get("name", "rival"), worst.get("category", ""))
            if key in self._stored_lessons:
                return
            self._stored_lessons.add(key)
            self.memory.remember(
                f"Lesson [rivalry] {rival.get('name', 'rival')} leads in "
                f"{worst.get('category', '?')} by {float(worst.get('gap', 0.0)):.0%} — practise it.",
                mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.8),
                importance=min(1.0, 0.6 + float(worst.get("gap", 0.0))),
                tags=["lesson", "rivalry"])
        except Exception:  # noqa: BLE001
            pass

    # ---- adversarial self-play (opt-in, Pillar F): NYXARA spars with her own clones ---- #
    def adversarial_self_play(self, *, rounds: int = 200,
                              max_seconds: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Run one clone-vs-clone contest; route the hardest categories into curiosity self-play.

        An attacker policy poses verifiable problems trying to stump a defender policy; both
        co-adapt over ``rounds`` and the bandit persists across passes (via memory). Pure
        measurement + corpus growth — no weights change, no external system is touched. Best-effort
        and never fatal: returns ``None`` on any failure. The hardest categories become curiosity
        topics so the next self-play round practises exactly where the attacker still wins.
        """
        try:
            from nyxara.growth.adversarial_self_play import AdversarialSelfPlay
            asp = AdversarialSelfPlay(settings=self.settings, memory=self.memory,
                                      frontier=self.frontier)
            out = asp.compete(rounds=rounds, max_seconds=max_seconds)
            topics = out.get("topics") or None
            if topics:
                try:
                    self.self_play(topics=topics)   # practise where she still loses (Edge 4 style)
                except Exception:  # noqa: BLE001 — routing is best-effort
                    pass
            return out
        except Exception:  # noqa: BLE001 — adversarial self-play is best-effort, never fatal
            return None

    # ---- metaprompt distillation (compress own successes into operating heuristics) ---- #
    def _metaprompt_obj(self) -> Any:
        if self._metaprompt is None:
            try:
                from nyxara.growth.metaprompt_distill import MetaPromptDistiller
                self._metaprompt = MetaPromptDistiller(
                    memory=self.memory, journal=self.journal,
                    skill_memory=self._skill_memory, settings=self.settings)
            except Exception:  # noqa: BLE001
                self._metaprompt = False
        return self._metaprompt or None

    def distill_metaprompts(self) -> List[str]:
        """Distil successful reasoning chains into prompt-level heuristics (best-effort)."""
        if not self.enable_metaprompt:
            return []
        distiller = self._metaprompt_obj()
        if distiller is None:
            return []
        try:
            return [i.text for i in distiller.distill()]
        except Exception:  # noqa: BLE001 — distillation is best-effort, never fatal
            return []

    def improve_self(self, *, generations: int = 1) -> List[Any]:
        if not self.enable_foundry:
            return []
        try:
            foundry = self._foundry
            if foundry is None:
                from nyxara.growth.foundry import Foundry
                # seed the foundry with her real memory so she trains on lived experience,
                # not just a static corpus — this is what makes the self-model *hers*
                foundry = self._foundry = Foundry(
                    settings=self.settings, seed_corpus=self._memory_corpus())
            return foundry.self_improve(generations=generations)
        except Exception:  # noqa: BLE001 — forging is heavy/optional; never fatal
            return []

    # ---- recursive self-improvement (review → architecture → benchmark → weakness → optimise) ---- #
    def _rsi_obj(self):
        if self._self_improver is None:
            from nyxara.growth.recursive_improvement import RecursiveSelfImprovement
            self._self_improver = RecursiveSelfImprovement(
                core=self.core, memory=self.memory, journal=self.journal,
                settings=self.settings, growth_engine=self)
        return self._self_improver

    def improve_system(self) -> Optional[Dict[str, Any]]:
        """Run one full recursive self-improvement cycle; return its report dict (best-effort)."""
        if not self.enable_self_improvement:
            return None
        try:
            return self._rsi_obj().run().to_dict()
        except Exception:  # noqa: BLE001 — self-improvement is best-effort, never fatal
            return None

    # ---- recursive mind-evolution (evolve the reasoning strategy itself) ---- #
    def _mind_evolution_obj(self):
        if self._mind_evolver is None:
            from nyxara.growth.mind_evolution import MindEvolutionEngine
            self._mind_evolver = MindEvolutionEngine(
                core=self.core, memory=self.memory, settings=self.settings,
                cost_penalty=float(self.settings.mind_evolution.cost_penalty))
        return self._mind_evolver

    def evolve_mind(self) -> Optional[Dict[str, Any]]:
        """Run one mind-evolution generation; return its lineage report dict (best-effort).

        Installs a promoted strategy into the live mind only when ``mind_evolution.autonomous_enact``
        is set (the standing authorisation for autonomous self-modification)."""
        if not self.enable_mind_evolution:
            return None
        try:
            cfg = self.settings.mind_evolution
            report = self._mind_evolution_obj().evolve_generations(
                int(cfg.generations_per_pass), enact=bool(cfg.autonomous_enact),
                population=int(cfg.population), inner_generations=int(cfg.inner_generations),
                islands=int(cfg.islands), plateau_window=int(cfg.plateau_window),
                escalate_architecture=bool(cfg.escalate_to_architecture))
            return report.to_dict()
        except Exception:  # noqa: BLE001 — mind-evolution is heavy/optional; never fatal
            return None

    # ---- meta-research (invent → test → (gated) integrate) ---- #
    def _meta_obj(self):
        if self._meta_researcher is None:
            from nyxara.growth.meta_research import MetaResearcher
            self._meta_researcher = MetaResearcher(memory=self.memory, settings=self.settings)
        return self._meta_researcher

    def meta_research(self, topic: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Run one meta-research pass on a rotating default topic (best-effort)."""
        if not self.enable_meta_research:
            return None
        try:
            topics = list(self.settings.meta_research.default_topics) or ["algorithm optimization"]
            topic = topic or topics[self._meta_topic_n % len(topics)]
            self._meta_topic_n += 1
            return self._meta_obj().run(topic).to_dict()
        except Exception:  # noqa: BLE001 — meta-research is best-effort, never fatal
            return None

    # ---- the full pass ---- #
    def run(self, *, do_foundry: Optional[bool] = None) -> GrowthReport:
        report = GrowthReport()
        lessons = self.reflect()
        report.episodes_seen = len(self._seen_action_seqs)
        report.lessons = [lz.to_dict() for lz in lessons]
        report.lessons_stored = len(self._stored_lessons)

        creport = self.consolidate()
        if creport is not None:
            report.replayed = len(getattr(creport, "replayed", []) or [])
            report.abstractions = len(getattr(creport, "abstractions", []) or [])
            report.forgotten = len(getattr(creport, "forgotten", []) or [])

        # Rivalry (Edge 4): measure against a peer AGI first, so its lead becomes the curiosity
        # focus for this pass's self-play. Default off; never acts on anything external.
        rival_topics: Optional[List[str]] = None
        if self.enable_rivalry:
            report.rivalry = self.rivalry()
            if report.rivalry:
                rival_topics = report.rivalry.get("topics") or None

        run_foundry = self.enable_foundry if do_foundry is None else do_foundry
        if run_foundry:
            # manufacture fresh training data first, so the forge learns something new
            if self.enable_selfplay:
                report.selfplay = self.self_play(topics=rival_topics)
            results = self.improve_self()
            report.foundry = [r.to_dict() for r in results]

        # Recursive self-improvement runs on its own (slower) cadence — only every
        # `self_improvement_every` growth passes, since it benchmarks the whole loop.
        self._growth_passes += 1
        if self.enable_self_improvement and \
                self._growth_passes % self.self_improvement_every == 0:
            report.self_improvement = self.improve_system()

        # Recursive mind-evolution runs on the slowest cadence — it evolves the reasoning
        # strategy itself and benchmarks the whole reasoner, so it is the heaviest loop.
        if self.enable_mind_evolution and \
                self._growth_passes % self.mind_evolution_every == 0:
            report.mind_evolution = self.evolve_mind()

        # Meta-research runs on its own (slow) cadence — invent + sandbox-test new theories.
        if self.enable_meta_research and \
                self._growth_passes % self.meta_research_every == 0:
            report.meta_research = self.meta_research()

        # Adversarial self-play (clone-vs-clone) — opt-in; the hardest categories become topics.
        if self.enable_adversarial:
            report.adversarial = self.adversarial_self_play(rounds=self.adversarial_rounds)

        # Continuous metaprompt distillation: turn this pass's fresh successes into operating
        # heuristics that ride every future prompt (recursive self-improvement).
        if self.enable_metaprompt and \
                self._growth_passes % max(1, self.settings.metaprompt.every_n_passes) == 0:
            report.metaprompt_insights = self.distill_metaprompts()
        return report


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.agency.permissions import Authority
    from nyxara.kernel.orchestrator import NyxaraCore

    print("=" * 70)
    print("NYXARA auto-learning self-test")
    print("=" * 70)

    core = NyxaraCore()
    # generate some experience the engine can learn from
    for _ in range(4):
        core.process("rotate the logs", authority=Authority.OWNER)
    core.process("how are you?", authority=Authority.OWNER)

    engine = GrowthEngine.from_core(core)
    rep = engine.run()
    print(f"\nepisodes seen       : {rep.episodes_seen}")
    print(f"lessons mined       : {len(rep.lessons)} (stored {rep.lessons_stored})")
    print(f"consolidation       : replayed={rep.replayed} abstractions={rep.abstractions} "
          f"forgotten={rep.forgotten}")
    assert rep.episodes_seen >= 5

    # lessons are written back into semantic memory and are recallable
    hits = core.memory.recall("lesson rotate", k=3) if core.memory else []
    print(f"recall lessons      : {[h[0].text()[:50] for h in hits]}")

    # a second pass only processes NEW experience (idempotent over already-seen actions)
    rep2 = engine.run()
    print(f"\nsecond pass episodes: {rep2.episodes_seen} (no new actions -> stable)")

    print("\nALL SELF-TESTS PASSED ✓")
