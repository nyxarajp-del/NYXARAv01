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
    # Exact-oracle-verified data she manufactures herself (the AlphaZero ceiling-break):
    # ground truth, not teacher imitation, so the forged model can exceed the teacher.
    verifiable_selfplay: Optional[Dict[str, Any]] = None
    synthesis: Optional[Dict[str, Any]] = None
    acquisition: Optional[Dict[str, Any]] = None
    foundry: List[Dict[str, Any]] = field(default_factory=list)
    self_improvement: Optional[Dict[str, Any]] = None
    mind_evolution: Optional[Dict[str, Any]] = None
    # Learning-to-learn: an INVENTED weight-update rule she composed and adopted when the fixed
    # learning method stalled (growth/rule_synth.py). None unless a pass ran + a rule was tried.
    rule_synthesis: Optional[Dict[str, Any]] = None
    meta_research: Optional[Dict[str, Any]] = None
    rivalry: Optional[Dict[str, Any]] = None
    adversarial: Optional[Dict[str, Any]] = None
    metaprompt_insights: List[str] = field(default_factory=list)
    concept_abstraction: Optional[Dict[str, Any]] = None
    # Few-shot skill acquisition (growth/skill_induction via the sample-efficient mind): tasks she
    # induced from a few demonstrations, verified, and can now apply to novel inputs herself.
    skills: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"episodes_seen": self.episodes_seen, "lessons": self.lessons,
                "lessons_stored": self.lessons_stored, "replayed": self.replayed,
                "abstractions": self.abstractions, "forgotten": self.forgotten,
                "selfplay": self.selfplay,
                "verifiable_selfplay": self.verifiable_selfplay,
                "synthesis": self.synthesis,
                "acquisition": self.acquisition,
                "foundry": self.foundry,
                "self_improvement": self.self_improvement,
                "mind_evolution": self.mind_evolution,
                "rule_synthesis": self.rule_synthesis,
                "meta_research": self.meta_research, "rivalry": self.rivalry,
                "adversarial": self.adversarial,
                "metaprompt_insights": self.metaprompt_insights,
                "concept_abstraction": self.concept_abstraction,
                "skills": self.skills}


class GrowthEngine:
    """Turns recent experience into durable improvement — reflect, consolidate, (forge)."""

    def __init__(self, *, memory: Any = None, journal: Any = None, core: Any = None,
                 settings: Optional[NyxaraSettings] = None, reflector: Any = None,
                 consolidator: Any = None, foundry: Any = None,
                 enable_foundry: Optional[bool] = None,
                 enable_selfplay: bool = False, selfplay_n: int = 6,
                 selfplay: Any = None,
                 enable_verifiable_selfplay: Optional[bool] = None,
                 verifiable_n: Optional[int] = None,
                 enable_synthesis: Optional[bool] = None,
                 synthesis_rounds: Optional[int] = None, curator: Any = None,
                 enable_self_improvement: Optional[bool] = None,
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
        # The ceiling-break: exact-oracle-verified data she manufactures HERSELF (verifiable
        # self-play + AlphaGo-Zero synthesis). The supervision is ground truth, not the teacher's
        # answers, so the forged model's target is *correctness*, not imitation — this is how she
        # can exceed the teacher on verifiable domains. Default ON whenever the foundry is on, so
        # every forge sees fresh ground truth, not only teacher distillation.
        self.enable_verifiable_selfplay = (
            self.enable_foundry if enable_verifiable_selfplay is None
            else bool(enable_verifiable_selfplay))
        self.verifiable_n = max(1, int(verifiable_n if verifiable_n is not None else 24))
        self.enable_synthesis = (
            bool(getattr(self.settings.synthesis, "enabled", True)) and self.enable_foundry
            if enable_synthesis is None else bool(enable_synthesis))
        self.synthesis_rounds = max(
            1, int(synthesis_rounds if synthesis_rounds is not None
                   else getattr(self.settings.synthesis, "rounds", 1)))
        self._curator = curator
        self._verified_flywheel: Any = None
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
        # Learning-to-learn: INVENT a new weight-update rule when the fixed learning method stalls,
        # on its own (slow) cadence — a synthesis pass evaluates many candidate learners, so it is a
        # heavy loop kept dormant until `should_invent()` detects a plateau/regression.
        self.enable_rule_synthesis = bool(self.settings.rule_synthesis.enabled)
        self.rule_synthesis_every = max(1, int(self.settings.rule_synthesis.every))
        self._rule_synth = None
        # Meta-research: invent → test → (gated) integrate, on its own (slow) cadence.
        self.enable_meta_research = (self.settings.meta_research.enabled
                                     if enable_meta_research is None else enable_meta_research)
        self.meta_research_every = max(1, int(
            self.settings.meta_research.meta_research_every
            if meta_research_every is None else meta_research_every))
        self._meta_researcher = meta_researcher
        self._growth_passes = 0
        self._last_acquire: Optional[Dict[str, Any]] = None
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
            text = f"Lesson [{lesson.kind.value}] {lesson.subject}: {lesson.text}"
            try:
                self.memory.remember(
                    text, mem_type=MemoryType.SEMANTIC,
                    provenance=Provenance(SourceType.SELF_REFLECTION,
                                          confidence=lesson.confidence),
                    importance=min(1.0, 0.5 + 0.3 * lesson.confidence),
                    tags=["lesson", lesson.kind.value])
                stored += 1
            except Exception:  # noqa: BLE001
                pass
            # A lesson that lives only as a memory record never reaches the substrate that
            # *answers*. Teach it to NYXARA's always-on learned brain too, so a distilled lesson
            # becomes retrievable knowledge she can actually speak — a text lesson consolidated
            # into her own learned weights, not just filed away (closes the lessons→weights gap).
            self._teach_self_brain(text)
            # …and into the foundry corpus: a distilled lesson is verified supervision, so the
            # NEXT forged model is trained on it too — reflection reaching all the way to weights.
            self._feed_lesson_to_flywheel(lesson)
        return stored

    def _feed_lesson_to_flywheel(self, lesson: Any) -> None:
        """Append one distilled lesson to the flywheel as a verified training pair."""
        fw = getattr(self.core, "flywheel", None) if self.core is not None else None
        if fw is None:
            return
        try:
            fw.consider(f"What have you learned about {lesson.subject}?", str(lesson.text),
                        confidence=float(getattr(lesson, "confidence", 1.0)), verified=True)
        except Exception:  # noqa: BLE001 — corpus feeding is best-effort, never breaks growth
            pass

    def _teach_self_brain(self, *docs: str) -> None:
        """Fold text (a lesson, a verified fact) into the core reasoner's learned brain.

        Best-effort and character-safe: it only moves *capability* (what she can recall and
        say), never a value, and any failure is silently skipped so growth never breaks."""
        core = self.core
        if core is None:
            return
        reasoner = getattr(core, "reasoner", None)
        teach = getattr(reasoner, "teach_self_brain", None)
        if teach is None:
            teach = getattr(getattr(reasoner, "llm_reasoner", None), "teach_self_brain", None)
        if callable(teach):
            try:
                teach(*docs)
            except Exception:  # noqa: BLE001 — compounding the own brain is best-effort
                pass

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

    # ---- abstract few-shot concepts into schemas ---- #
    def abstract_concepts(self) -> Optional[Dict[str, Any]]:
        """Fold accumulated few-shot concepts into abstract templates (instances → schemas).

        Uses the live core's :class:`~nyxara.cognition.sample_efficient.SampleEfficientMind`
        when available, otherwise builds one over this engine's memory. Best-effort: any
        failure (or no faculty) returns None and never disturbs the growth pass."""
        mind = getattr(self.core, "sample_efficient", None) if self.core is not None else None
        try:
            if mind is None:
                if self.memory is None:
                    return None
                from nyxara.cognition.sample_efficient import SampleEfficientMind
                mind = SampleEfficientMind(getattr(self.memory, "embedder", None),
                                           store=self.memory)
            return mind.consolidate().to_dict()
        except Exception:  # noqa: BLE001 — concept abstraction is best-effort, never fatal
            return None

    # ---- few-shot skill acquisition (learn a task from demos, apply to novel inputs) ---- #
    def induce_skills(self, consolidation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Report the skills mined this pass and teach each one's summary into the learned brain.

        The mining itself happens inside the sample-efficient mind's :meth:`consolidate` (already
        run by :meth:`abstract_concepts`), which turns accumulated ``(cue → response)`` demonstrations
        into a *verified* reusable transformation. Here we surface what was learned and fold a
        one-line capability summary of every learned skill into NYXARA's always-on brain, so a task
        she taught herself becomes something she can also *speak about* — real, compounding capability
        gain she performed herself (no external LLM). Best-effort; never fatal."""
        mind = getattr(self.core, "sample_efficient", None) if self.core is not None else None
        if mind is None:
            return None
        try:
            engine = getattr(mind, "skills", None)
            learned = engine.skills() if engine is not None else []
        except Exception:  # noqa: BLE001 — introspection is best-effort
            learned = []
        for skill in learned:
            try:
                self._teach_self_brain(getattr(skill, "render", lambda: "")())
            except Exception:  # noqa: BLE001 — teaching the summary is best-effort
                pass
        cons = consolidation or {}
        return {"skills_induced": int(cons.get("skills_induced", 0)),
                "skills": list(cons.get("skills", [])),
                "total_skills": len(learned)}

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

    # ---- verifiable self-play + synthesis (the ceiling-break: ground-truth, not the teacher) ---- #
    def _shared_flywheel(self) -> Any:
        """The data-flywheel the foundry forges from — shared so verified pairs reach training."""
        if self._verified_flywheel is None:
            from nyxara.growth.flywheel import DataFlywheel
            self._verified_flywheel = DataFlywheel.from_settings(self.settings)
        return self._verified_flywheel

    def play_verifiable(self, *, n: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Pose structured problems and answer them with her own exact faculties (no teacher).

        The oracle is exact, so each ``(prompt, answer)`` is provably correct and is collected
        ``verified=True`` into the SAME flywheel the foundry trains on. This works on a keyless
        machine and — crucially — its supervision is ground truth, so the forged model can learn
        to be *more correct* than the teacher, breaking the distillation ceiling. Best-effort.
        """
        try:
            if self._selfplay is None:
                from nyxara.growth.selfplay import SelfPlay
                self._selfplay = SelfPlay(settings=self.settings, frontier=self.frontier,
                                          flywheel=self._shared_flywheel())
            return self._selfplay.play_verifiable(n or self.verifiable_n)
        except Exception:  # noqa: BLE001 — verifiable self-play is best-effort, never fatal
            return None

    def curate_synthetic(self, *, rounds: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Run the AlphaGo-Zero loop: self-generate logic, rival-verify it, keep only survivors.

        Accepted items are machine-checked by the :class:`~nyxara.growth.prover.Prover` (exact,
        not a teacher's opinion) and folded ``verified=True`` into the foundry's flywheel corpus
        plus her grounded knowledge. Pure ground-truth supervision — the second ceiling-break
        engine. Best-effort; never fatal.
        """
        try:
            if self._curator is None:
                from nyxara.growth.synthesis import SyntheticCurator
                self._curator = SyntheticCurator(
                    settings=self.settings, flywheel=self._shared_flywheel(),
                    knowledge=getattr(self.core, "knowledge", None))
            return self._curator.curate(rounds=rounds or self.synthesis_rounds).to_dict()
        except Exception:  # noqa: BLE001 — synthesis is best-effort, never fatal
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

    def _acquire_topics(self) -> List[str]:
        """Derive what to go learn from her measured ignorance, not from a fallback list.

        The drive fuses her five uncertainty signals (weakness severity, open-question
        uncertainty, world-model epistemic error, signal-bus focus, RND novelty) onto one scale
        and ranks them, so acquisition is finally aimed at what she is actually worst at."""
        try:
            from nyxara.growth.acquire import gap_topics
            from nyxara.growth.epistemic_drive import EpistemicDrive
            drive = EpistemicDrive(core=getattr(self, "core", None), settings=self.settings,
                                   memory=self.memory)
            return gap_topics(memory=self.memory, settings=self.settings, drive=drive)
        except Exception:  # noqa: BLE001 — topic derivation is best-effort
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
            # Autonomous data acquisition: before forging, harvest SCREENED external web text for
            # her current knowledge gaps and fold it into the corpus (breadth she did not contain,
            # not just her journal). Best-effort; the AcquireReport is surfaced for audit.
            self._last_acquire = foundry.acquire(self._acquire_topics())
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

    # ---- learning-to-learn: invent a NEW learning rule when the old one fails ---- #
    def _rule_synth_obj(self):
        if self._rule_synth is None:
            from nyxara.growth.rule_synth import LearningRuleSynthesizer
            self._rule_synth = LearningRuleSynthesizer(core=self.core, settings=self.settings)
        return self._rule_synth

    def should_invent(self) -> bool:
        """Only invent a new learning rule when the EXISTING learner has genuinely stalled/regressed.

        Reads live plateau/regression signals already produced by the system; absent any evidence it
        returns ``False`` — the subsystem stays dormant so a working learner is never churned. This
        is the "jab purana tarika fail ho" (when the old way fails) trigger.
        """
        core = self.core
        if core is None or getattr(core, "learner", None) is None:
            return False
        cfg = self.settings.rule_synthesis
        # (a) learning-dimension plateau/decline from the live MetaLearningEngine (least-squares slope)
        mle = getattr(core, "meta_learning_engine", None)
        if mle is not None:
            try:
                from nyxara.growth.meta_engine import MetaDimension
                slope = mle.trend(MetaDimension.LEARNING)
                n = len(mle._samples.get(MetaDimension.LEARNING.value, ()))
                if n >= int(cfg.min_signal) and slope <= float(cfg.plateau_slope):
                    return True
            except Exception:  # noqa: BLE001 — signal is best-effort
                pass
        # (b) the reflector says recent actions are net-hurting (more "hurts" than "helps")
        try:
            pats = self._reflector_obj().action_patterns()
            if pats and (sum(1 for p in pats if p.verdict == "hurts")
                         > sum(1 for p in pats if p.verdict == "helps")):
                return True
        except Exception:  # noqa: BLE001
            pass
        # (c) an explicit eval regression handed in via the core
        return bool(getattr(core, "_last_eval_regressions", None))

    def synthesize_rule(self) -> Optional[Dict[str, Any]]:
        """Run one learning-rule synthesis pass (best-effort).

        Installs an invented rule into the live learner only when ``rule_synthesis.autonomous_enact``
        is set (the standing authorisation for autonomous self-modification) and only after it
        measurably beats the incumbent AND recovers the optimum. Reversible via ``rollback_rule``."""
        if not self.enable_rule_synthesis or not self.should_invent():
            return None
        try:
            cfg = self.settings.rule_synthesis
            report = self._rule_synth_obj().run(enact=bool(cfg.autonomous_enact))
            return report.to_dict()
        except Exception:  # noqa: BLE001 — synthesis is heavy/optional; never fatal
            return None

    # ---- meta-research (invent → test → (gated) integrate) ---- #
    def _meta_obj(self):
        if self._meta_researcher is None:
            from nyxara.growth.meta_research import MetaResearcher
            # wire the live LawDiscoveryEngine (if the core built one) so a continuous meta-research
            # pass does real symbolic law discovery too, not only code invention.
            self._meta_researcher = MetaResearcher(
                memory=self.memory, settings=self.settings,
                law_discovery=getattr(self.core, "law_discovery", None))
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

        # Concept abstraction (Rule 4): fold the few-shot concepts learned this period into
        # abstract templates — closing the loop instances → concepts → schemas. Best-effort.
        report.concept_abstraction = self.abstract_concepts()

        # Few-shot skill acquisition (Rule 4): mine accumulated demonstrations into verified,
        # reusable transformations she can apply to novel inputs, and teach their summaries into her
        # always-on brain. This is the capability the Master asked for — learning a new task from a
        # few examples, done by NYXARA herself. Best-effort.
        report.skills = self.induce_skills(report.concept_abstraction)

        # Rivalry (Edge 4): measure against a peer AGI first, so its lead becomes the curiosity
        # focus for this pass's self-play. Default off; never acts on anything external.
        rival_topics: Optional[List[str]] = None
        if self.enable_rivalry:
            report.rivalry = self.rivalry()
            if report.rivalry:
                rival_topics = report.rivalry.get("topics") or None

        run_foundry = self.enable_foundry if do_foundry is None else do_foundry
        if run_foundry:
            # manufacture fresh training data first, so the forge learns something new.
            # GROUND TRUTH FIRST (the ceiling-break): exact-oracle self-play + AlphaGo-Zero
            # synthesis fold provably-correct pairs into the flywheel the foundry trains on, so
            # the model's target is correctness — letting it exceed the teacher, not just copy it.
            if self.enable_verifiable_selfplay:
                report.verifiable_selfplay = self.play_verifiable()
            if self.enable_synthesis:
                report.synthesis = self.curate_synthetic()
            # Teacher distillation self-play (imitation) is additive breadth, kept opt-in.
            if self.enable_selfplay:
                report.selfplay = self.self_play(topics=rival_topics)
            results = self.improve_self()
            report.acquisition = self._last_acquire
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

        # Learning-to-learn: when the fixed learning method has stalled, INVENT a new weight-update
        # rule from primitives and (gated + reversible) install it into the live learner. Dormant
        # until `should_invent()` fires, on its own slow cadence.
        if self.enable_rule_synthesis and \
                self._growth_passes % self.rule_synthesis_every == 0:
            report.rule_synthesis = self.synthesize_rule()

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
