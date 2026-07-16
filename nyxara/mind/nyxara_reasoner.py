"""NYXARA · mind/nyxara_reasoner.py — the integrated kernel reasoner (⬆ → 👑).

The sovereign loop (``kernel/orchestrator.py``) carries every turn through *perceive →
attend → **reason** → gate → act*. Historically the "reason" step was a ten-line keyword
matcher (``_default_reasoner``): all of NYXARA's real faculties — memory, dual-process
cognition, the multi-model council, the world model, the soul — were built but never
called from the main cognitive cycle.

This module closes that gap. :class:`NyxaraReasoner` is the callable the kernel installs
in the reason seat, and it *actually convenes the mind*:

1. **Recall before reasoning.** Top-k relevant episodic + semantic memories (via the
   :class:`~nyxara.memory.retrieval.AssociativeRetriever`, or whatever the kernel already
   retrieved this turn) are folded in as grounding context.
2. **Dual-process routing.** A :class:`~nyxara.mind.dual_process.DualProcess` arbitrator
   decides System-1 (fast, intuitive) vs System-2 (slow, deliberate). High-stakes or
   complex/novel input is routed to System-2.
3. **LLM / council generation.** With a real LLM available, a deliberate (System-2) reply
   is drafted by the whole :class:`~nyxara.mind.council.LLMCouncil`; a fast (System-1) one
   by a single :class:`~nyxara.mind.llm.LLM` pass. The system prompt is always derived from
   :meth:`Soul.voice().system_fragment <nyxara.identity.soul.VoiceProfile.system_fragment>`.
4. **World-model action planning.** For action-type requests the
   :class:`~nyxara.mind.world_model.WorldModel` rolls candidate policies forward to
   evaluate them before one is proposed.

Like the faculty it sits on, it is honest about its limits: with no real provider (a bare,
keyless machine), it falls back to the world model + the deterministic faculty reasoner, so
a turn never crashes and behaves identically offline.

The LLM proposes; the kernel still disposes — this reasoner only ever returns a
:class:`~nyxara.kernel.orchestrator.Candidate`. Every downstream gate is untouched.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.kernel.orchestrator import Candidate, _default_reasoner

__all__ = ["NyxaraReasoner"]

# stimulus cues used to gauge stakes and to spot action-like commands (mirrors the
# deterministic stand-in so routing stays consistent with the fallback)
_COMMAND_VERBS = ("delete", "remove", "shutdown", "kill", "run", "exec", "install",
                  "block", "open", "disable", "rotate", "deploy", "read", "write",
                  "list", "fetch", "search", "create", "make")
_HIGH_STAKES = ("delete", "shutdown", "kill", "exec", "attack", "intrusion", "breach",
                "password", "secret", "credential", "production", "database", "transfer",
                "payment", "money", "deploy", "wipe", "format", "root")


class NyxaraReasoner:
    """The kernel's reason step, wired to the whole mind. Stateless per call."""

    def __init__(self, *, llm: Any = None, council: Any = None, memory: Any = None,
                 retriever: Any = None, soul: Any = None, world_model: Any = None,
                 tools: Any = None, llm_reasoner: Any = None, use_council: bool = False,
                 settings: Optional[NyxaraSettings] = None, narrative: Any = None,
                 knowledge: Any = None, metaprompt: Any = None,
                 generalization_engine: Any = None, causal_model: Any = None,
                 knowledge_graph: Any = None, intuition: Any = None,
                 law_discovery: Any = None, belief_model: Any = None,
                 max_memory_context: int = 5) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.council = council
        self.memory = memory
        self.retriever = retriever
        self.soul = soul
        self.narrative = narrative
        self.world_model = world_model
        self.tools = tools
        # foundational ground truth (knowledge base) — lets the offline mind answer factual
        # questions from evidence when no LLM is configured.
        self.knowledge = knowledge
        # the tool-aware, real-LLM JSON decider (mind/llm_reasoner.py); used as the
        # generation engine when a genuine provider is available, so tool selection is kept.
        # It is also where MCTS deep reasoning runs, so conversational turns delegate to it.
        self.llm_reasoner = llm_reasoner
        # her unified own-faculty generalizer (mind/generalization.py) — the cascade that solves a
        # genuinely NEW task (from examples / novel domain / table) with her own faculties, in the
        # live turn, offline. Injected by the kernel; lazily built from her sub-minds if absent.
        self.generalization_engine = generalization_engine
        self._gen_lazy: Any = None
        # distilled operating heuristics (growth/metaprompt_distill.py), injected into the voice.
        self.metaprompt = metaprompt
        self.use_council = use_council
        self.max_memory_context = max_memory_context
        # the Intuition Core (mind/intuition.py) — real non-algorithmic hunches for System 1.
        self.intuition = intuition
        self._dual: Any = None  # lazily built dual-process facade
        self._offline: Any = None  # lazily built sovereign offline mind (keyless machine)
        self._router: Any = None  # lazily built own-model-first confidence router (handoff)
        self._self_brain: Any = None  # lazily built own LEARNED brain (compounds with experience)
        self._sample_mind: Any = None  # lazily built few-shot / skill-induction mind
        self._deep: Any = None  # lazily built always-max deep-reasoning controller (Problem #1)
        self._compute: Any = None  # lazily built general reduce→run→verify reasoner (Problem #1)
        # HER OWN chain of thought (mind/native_reasoner.py): the learned causal graph +
        # knowledge graph + verifiable faculties composed on a multi-hop blackboard — the
        # reasoning trace is authored by her algorithms, never by prompting a model.
        self.causal_model = causal_model
        self.knowledge_graph = knowledge_graph
        # her SELF-DISCOVERED empirical laws + settled beliefs — so the native chain of thought
        # can answer from what her autonomous science actually learned (closes the loop from
        # discovery back into reasoning; no LLM).
        self.law_discovery = law_discovery
        self.belief_model = belief_model
        self._native: Any = None  # lazily built native chain-of-thought orchestrator
        self.last_native_trace: Optional[List[dict]] = None  # last turn's inspectable steps

    # ------------------------------------------------------------------ #
    # The reasoning act
    # ------------------------------------------------------------------ #
    def __call__(self, stimulus: str, focus: Any = None,
                 memories: Optional[List[Any]] = None) -> Candidate:
        try:
            return self._reason(stimulus, focus, memories)
        except Exception:  # noqa: BLE001 — never let the mind crash the loop
            return _default_reasoner(stimulus, focus)

    def _reason(self, stimulus: str, focus: Any, memories: Optional[List[Any]]) -> Candidate:
        mems = self._gather_memories(stimulus, memories)
        outcome = self._route(stimulus, mems)
        if self._looks_like_action(stimulus):
            candidate = self._action_candidate(stimulus, mems, outcome)
        else:
            candidate = self._respond_candidate(stimulus, mems, outcome)
        return self._apply_narrative_coherence(candidate)

    def _apply_narrative_coherence(self, candidate: Candidate) -> Candidate:
        """Weigh the proposal against who NYXARA is (identity/narrative.py, Pillar D1).

        Advisory and gentle: a proposal that *coheres* with her identity is noted; one that
        *breaks her loyalty through-line* (betrayal, defying the Master, resisting oversight)
        is spoken with low confidence — a self-doubt signal, never a block. The kernel still
        disposes through every gate, and her character is never edited (Rule 4 character-lock)."""
        if self.narrative is None or candidate is None:
            return candidate
        try:
            report = self.narrative.coherence(getattr(candidate, "text", "") or "")
        except Exception:  # noqa: BLE001 — the narrative self is advisory, never fatal
            return candidate
        if report.dissonant:
            candidate.confidence = min(getattr(candidate, "confidence", 0.7), 0.2)
            candidate.belief = min(getattr(candidate, "belief", candidate.confidence), 0.2)
            candidate.rationale = (f"{getattr(candidate, 'rationale', '')} "
                                   f"[narrative: {report.note}]").strip()
        elif report.matched_themes:
            themes = ", ".join(report.matched_themes)
            candidate.rationale = (f"{getattr(candidate, 'rationale', '')} "
                                   f"[narrative: coheres with my {themes}]").strip()
        return candidate

    # ------------------------------------------------------------------ #
    # 1. Memory recall before reasoning
    # ------------------------------------------------------------------ #
    def _gather_memories(self, stimulus: str, memories: Optional[List[Any]]) -> List[str]:
        """Normalise the recalled memories (kernel-supplied, or retrieved here) to text."""
        raw = memories if memories is not None else self._retrieve(stimulus)
        out: List[str] = []
        for m in raw or []:
            text = self._memory_text(m)
            if text:
                out.append(text[:200])
            if len(out) >= self.max_memory_context:
                break
        return out

    def _retrieve(self, stimulus: str) -> List[Any]:
        if self.retriever is not None:
            try:
                from nyxara.memory.retrieval import RetrievalContext
                return self.retriever.retrieve(
                    RetrievalContext(query=stimulus), k=self.max_memory_context)
            except Exception:  # noqa: BLE001
                pass
        if self.memory is not None:
            try:
                return self.memory.recall(stimulus, k=self.max_memory_context)
            except Exception:  # noqa: BLE001
                pass
        return []

    @staticmethod
    def _memory_text(m: Any) -> str:
        # RetrievalResult, (record, score) tuple, raw MemoryRecord, or plain string
        rec = getattr(m, "record", None)
        if rec is not None and hasattr(rec, "text"):
            return rec.text()
        if isinstance(m, tuple) and m and hasattr(m[0], "text"):
            return m[0].text()
        if hasattr(m, "text"):
            try:
                return m.text()
            except Exception:  # noqa: BLE001
                return ""
        return str(m) if isinstance(m, str) else ""

    # ------------------------------------------------------------------ #
    # 2. Dual-process routing (System-1 vs System-2)
    # ------------------------------------------------------------------ #
    def _route(self, stimulus: str, mems: List[str]) -> Any:
        dp = self._dual_process()
        if dp is None:
            return None
        try:
            from nyxara.mind.faculties import Task, TaskType
            stakes = self._stakes(stimulus)
            difficulty = min(1.0, len(stimulus) / 400.0)
            novelty = 1.0 if not mems else max(0.2, 1.0 - 0.2 * len(mems))
            gut = 0.85 if stakes < 0.5 else 0.4  # unconfident gut on stakes -> reflect
            task = Task(TaskType.REASONING, description=stimulus,
                        features={"stakes": stakes, "difficulty": difficulty,
                                  "novelty": novelty, "_gut_conf": gut})
            return dp.think(task, stakes=stakes)
        except Exception:  # noqa: BLE001 — routing is advisory, never fatal
            return None

    def _dual_process(self) -> Any:
        if self._dual is not None:
            return self._dual
        try:
            from nyxara.mind.dual_process import (Arbitrator, DualProcess, System1,
                                                  System2)

            if self.intuition is not None:
                from nyxara.mind.intuition import make_intuition_callable
                _base_intuition = make_intuition_callable(self.intuition)

                def _intuition(task: Any) -> Tuple[Any, float]:
                    ans, conf = _base_intuition(task)
                    # keep the reasoner's own gut floor when the Core abstains on this stimulus
                    if ans == task.description:
                        conf = float(task.features.get("_gut_conf", conf))
                    return (ans, conf)
            else:
                def _intuition(task: Any) -> Tuple[Any, float]:
                    conf = float(task.features.get("_gut_conf", 0.7))
                    return (task.description, conf)

            def _deliberate(task: Any) -> Any:
                from nyxara.mind.proposal import Proposal, ProposalKind
                return Proposal(kind=ProposalKind.ANSWER, content=task.description,
                                source_faculty="nyxara", confidence=0.7,
                                rationale="deliberation requested")

            self._dual = DualProcess(System1(_intuition), System2(deliberate=_deliberate),
                                     Arbitrator())
        except Exception:  # noqa: BLE001
            self._dual = None
        return self._dual

    @staticmethod
    def _looks_like_action(stimulus: str) -> bool:
        """Command-like input is an action request (mirrors the deterministic stand-in)."""
        low = stimulus.strip().lower()
        return any(low.startswith(v + " ") or f" {v} " in low for v in _COMMAND_VERBS)

    @staticmethod
    def _stakes(stimulus: str) -> float:
        low = stimulus.lower()
        if any(w in low for w in _HIGH_STAKES):
            return 0.8
        if any(low.startswith(v + " ") or f" {v} " in low for v in _COMMAND_VERBS):
            return 0.55
        return 0.3

    @staticmethod
    def _process_label(outcome: Any) -> str:
        try:
            return outcome.process.value
        except Exception:  # noqa: BLE001
            return "system_1"

    @staticmethod
    def _is_system2(outcome: Any) -> bool:
        try:
            from nyxara.mind.dual_process import ProcessType
            return outcome is not None and outcome.process is ProcessType.SYSTEM_2
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ #
    # 3. Conversational generation (LLM / council), soul-voiced
    # ------------------------------------------------------------------ #
    def _respond_candidate(self, stimulus: str, mems: List[str], outcome: Any) -> Candidate:
        process = self._process_label(outcome)
        # NATIVE CHAIN OF THOUGHT — HER OWN reasoning, first. The trace is authored by her
        # algorithms (learned causal graph, knowledge graph, verifiable chains, certified
        # compute, structure-mapping analogy) composed on a multi-hop blackboard — never by
        # prompting a model to "think step by step". It subsumes the faculty + compute rungs
        # below (the native engine runs both) and abstains honestly, so ordinary conversation
        # falls straight through. The LLM is demoted to the last resort at the ladder's foot.
        native = self._native_answer(stimulus, mems, process)
        if native is not None:
            return native
        # neuro-symbolic short-circuit: an exact, verifiable answer (math / logic) beats any
        # neural guess — so NYXARA computes it herself, with or without an LLM present. This is
        # "verifiable > probabilistic" applied to the *whole loop*, not only the router.
        fac = self._faculty_answer(stimulus)
        if fac is not None:
            text, conf = fac
            return Candidate(text=text, kind="respond", capability=Capability.MESSAGE_SEND,
                             risk=RiskTier.LOW, reversible=True, confidence=conf, belief=conf,
                             rationale=f"{process} via a verifiable faculty (exact computation, "
                                       f"no guessing); {len(mems)} memories recalled")
        # learned-skill short-circuit: a task NYXARA induced from a few demonstrations (and
        # verified against every one of them) is applied here to a novel input — real, transferable
        # capability she gained herself, not a neural guess. Fires only when the input matches what
        # the skill was taught to transform, so it never hijacks an unrelated turn.
        skl = self._skill_answer(stimulus)
        if skl is not None:
            text, conf = skl
            return Candidate(text=text, kind="respond", capability=Capability.MESSAGE_SEND,
                             risk=RiskTier.LOW, reversible=True, confidence=conf, belief=conf,
                             rationale=f"{process} via a learned skill (few-shot induced + "
                                       f"verified, applied to a new input); "
                                       f"{len(mems)} memories recalled")
        # unified own-faculty generalization: a genuinely NEW task — shown by a few examples in the
        # prompt, or a novel domain whose relational structure maps onto one she already understands,
        # or a numeric law stated as a table — solved by her OWN faculties in the live turn (program
        # induction / structural transfer / open-world law-fitting). Her reasoning, not the LLM's, so
        # it runs whether or not a provider is configured. Conservative: the cascade declines on
        # ordinary conversation (it needs real, extractable structure), so it never hijacks a turn.
        gen = self._generalize_answer(stimulus)
        if gen is not None:
            text, conf, source, analogical = gen
            # domain-genesis reasons from the alien field's OWN internal structure — not by
            # analogy to a known domain — so it earns its own honest label.
            if source == "domain_genesis":
                how = " from its own structure, from scratch"
            elif analogical:
                how = " by analogy"
            else:
                how = ""
            return Candidate(text=text, kind="respond", capability=Capability.MESSAGE_SEND,
                             risk=RiskTier.LOW, reversible=True, confidence=conf, belief=conf,
                             rationale=f"{process} via her own {source} faculty (generalized{how}, "
                                       f"no LLM); {len(mems)} memories recalled")
        # GENERAL COMPUTE-AND-VERIFY (Problem #1 — general reasoning without pre-programming): reduce
        # the problem to a computation, RUN it (a general SymPy CAS, or a program she executes in the
        # real sandbox), and emit only a *certified* result. This is not a hardcoded problem-class
        # matcher — it solves whole classes (any equation/system, any identity, any expression) and,
        # when a model is present, it consumes the model as a *code proposer* whose program she runs
        # and verifies — the truth-bearing step is hers, never the model's word. It runs before the
        # LLM handoff so her own verified computation wins; it abstains (None) on anything it cannot
        # certify, so ordinary conversation falls straight through. See mind/compute_reasoner.py.
        cmp = self._compute_answer(stimulus)
        if cmp is not None:
            text, conf = cmp
            return Candidate(text=text, kind="respond", capability=Capability.MESSAGE_SEND,
                             risk=RiskTier.LOW, reversible=True, confidence=conf, belief=conf,
                             rationale=f"{process} via her own compute-and-verify reasoner (reduced "
                                       f"to a computation she ran and certified, not a guess); "
                                       f"{len(mems)} memories recalled")
        if not self._real_llm():
            # keyless machine: the sovereign offline mind answers from her own faculties
            # (NLP + knowledge base + recalled memory) instead of echoing "I understand: X".
            cand = self._offline_mind().respond(stimulus, mems)
            cand.rationale = (f"{process} via sovereign offline mind "
                              f"(no LLM configured); {cand.rationale}; "
                              f"{len(mems)} memories recalled")
            return cand

        system = self._system_prompt()
        # SOVEREIGN BRAIN (Phase 2): give NYXARA's OWN forged model first crack. If her model
        # is promoted and its answer clears the verifier, she speaks it herself (a *handoff*)
        # rather than deferring to the external teacher — this is what raises the handoff rate
        # above 0%. She only falls through to the teacher/council when her own answer isn't
        # trustworthy. The reply is still a proposal the kernel disposes (no gate is bypassed).
        own = self._own_model_handoff(stimulus, system)
        if own is not None:
            return own

        # ALWAYS-MAXIMUM DEEP REASONING (Problem #1 — the ceiling): raise her *effective* reasoning
        # depth by climbing the whole effort ladder (self-consistency → deliberation → MCTS →
        # verified refinement) and keeping the answer an independent verifier scores highest. The
        # controller is NYXARA (driven by her own measured signals), and it compounds — it learns
        # which rung pays off for each kind of problem. It folds MCTS in as one rung, so it
        # supersedes the single-strategy MCTS branch below when enabled. Returns a Candidate the
        # kernel still disposes through every gate; None (disabled / no real provider / failure)
        # falls through to the exact prior behaviour.
        deep_cand = self._deep_respond(stimulus, mems, process)
        if deep_cand is not None:
            return deep_cand

        # MCTS DEEP REASONING (fallback when deep reasoning is off): when the tool-aware LLM reasoner
        # is wired and MCTS is enabled, delegate the whole conversational decision to it so the
        # answer is produced by genuine search-over-reasoning (branch → simulate → backpropagate →
        # choose), adversarially hardened, rather than a single LLM pass. It returns a Candidate; the
        # kernel still disposes through every gate. Any failure falls through to the normal path.
        mcts_cand = self._mcts_respond(stimulus, mems, process)
        if mcts_cand is not None:
            return mcts_cand

        deliberate = self._is_system2(outcome)
        if deliberate and self._council_ready():
            text, conf, how = self._deliberate_council(stimulus, system, mems)
        else:
            text, conf, how = self._generate_llm(stimulus, system, mems, deliberate)
        rationale = f"{process}: {how}; grounded in {len(mems)} recalled memories"
        return Candidate(text=text, kind="respond", capability=Capability.MESSAGE_SEND,
                         risk=RiskTier.LOW, reversible=True, confidence=conf, belief=conf,
                         rationale=rationale)

    def _deep_reasoner(self) -> Any:
        """Lazily build the always-max deep-reasoning controller (mind/deep_reasoning.py).

        Wired over the tool-aware LLM reasoner (which owns the LLM, grounding, and decision→Candidate
        conversion), with a RecursiveImprover for the refinement rung and a persisted EffortMemory so
        the effort allocation genuinely compounds across turns and restarts."""
        if self._deep is not None:
            return self._deep or None
        cfg = getattr(self.settings.llm, "deep_reasoning", None)
        if self.llm_reasoner is None or cfg is None or not bool(getattr(cfg, "enabled", False)):
            self._deep = False  # sentinel: tried and unavailable
            return None
        try:
            from nyxara.mind.deep_reasoning import DeepReasoner
            from nyxara.mind.recursive_improver import RecursiveImprover
            improver = RecursiveImprover(
                llm=self.llm,
                n_iterations=int(getattr(self.settings.llm,
                                         "recursive_improvement_iterations", 5)))
            effort_memory = self._effort_memory() if bool(getattr(cfg, "learn_effort", True)) else None
            # Compute-scaled test-time-compute budget (Problem #1 — Scale): the compute NYXARA
            # actually has decides how hard she thinks. Scales *up* from the config floor only, so
            # a bare box is unchanged; any failure falls back to the static config. This is the
            # honest way her small model buys effective scale — see growth/effective_scale.py.
            budget = None
            try:
                from nyxara.growth.effective_scale import scaled_budget
                from nyxara.kernel.compute import compute_report
                budget = scaled_budget(compute_report(), cfg)
            except Exception:  # noqa: BLE001 — budget scaling is additive; never a hard dependency
                budget = None
            self._deep = DeepReasoner(self.llm_reasoner, settings=self.settings,
                                      improver=improver, effort_memory=effort_memory,
                                      budget=budget)
        except Exception:  # noqa: BLE001 — deep reasoning is additive, never a hard dependency
            self._deep = False
        return self._deep or None

    def _effort_memory(self) -> Any:
        """Build the compounding effort store, persisted under paths.data_dir when available."""
        try:
            from nyxara.mind.effort_memory import EffortMemory
            cfg = self.settings.llm.deep_reasoning
            path = None
            data_dir = getattr(getattr(self.settings, "paths", None), "data_dir", None)
            if data_dir is not None:
                from pathlib import Path
                path = Path(data_dir) / "effort_memory.json"
            return EffortMemory(
                min_observations=float(getattr(cfg, "effort_min_observations", 3.0)),
                success_floor=float(getattr(cfg, "effort_success_floor", 0.5)),
                path=path)
        except Exception:  # noqa: BLE001 — compounding is best-effort
            return None

    def _deep_respond(self, stimulus: str, mems: List[str], process: str) -> Optional[Candidate]:
        """Climb the always-max effort ladder for a conversational turn; keep the verifier-best."""
        deep = self._deep_reasoner()
        if deep is None:
            return None
        try:
            cand = deep(stimulus, None)
        except Exception:  # noqa: BLE001 — deep reasoning is additive; fall through to the normal path
            return None
        if cand is None or getattr(cand, "kind", "respond") != "respond":
            # None -> defer; an 'act' proposal (a decisive tool need) is handled by the action path.
            return None
        cand.rationale = (f"{process}: {getattr(cand, 'rationale', '')}; "
                          f"grounded in {len(mems)} recalled memories").strip()
        return cand

    def _mcts_respond(self, stimulus: str, mems: List[str], process: str) -> Optional[Candidate]:
        """Delegate a conversational turn to the MCTS-backed LLM reasoner when enabled."""
        mcfg = getattr(self.settings, "mcts", None)
        if (self.llm_reasoner is None or mcfg is None or not mcfg.enabled
                or not self._real_llm()):
            return None
        try:
            cand = self.llm_reasoner(stimulus, None)
        except Exception:  # noqa: BLE001 — deep reasoning is advisory; fall back to a plain pass
            return None
        if cand is None or getattr(cand, "kind", "respond") != "respond":
            return None
        cand.rationale = (f"{process}: {getattr(cand, 'rationale', '')}; "
                          f"grounded in {len(mems)} recalled memories").strip()
        return cand

    def _own_model_handoff(self, stimulus: str, system: str) -> Optional[Candidate]:
        """Hand the turn to NYXARA's own promoted model when it answers confidently."""
        router = self._confidence_router()
        if router is None:
            return None
        try:
            res = router.draft_self(stimulus, system=system)
        except Exception:  # noqa: BLE001 — own-model routing is advisory; never crash a turn
            return None
        if res is None or not res.handed_off or not res.text:
            return None
        return Candidate(
            text=res.text, kind="respond", capability=Capability.MESSAGE_SEND, target="",
            risk=RiskTier.LOW, reversible=True, confidence=res.confidence,
            belief=res.confidence,
            rationale=f"answered by NYXARA's own model (handoff; verifier "
                      f"{res.confidence:.2f})")

    def _confidence_router(self) -> Any:
        """Lazily build the own-model-first confidence router (mind/router.py).

        The router's "own model" is wired to NYXARA's *always-on learned brain* — the very
        :class:`~nyxara.mind.self_reasoner.SelfBrain` she compounds on every lived turn (and
        which itself prefers a promoted foundry model when one exists). This is what makes the
        handoff real: the substrate that learns is the substrate that answers, instead of a
        foundry model she would first have to forge and promote. When the flag is off (or her
        brain is unavailable) the router falls back to the foundry-only ``SelfProvider``."""
        if self._router is None:
            try:
                from nyxara.mind.router import Router
                self_provider = None
                if bool(getattr(self.settings.router, "use_self_brain", True)):
                    from nyxara.mind.self_reasoner import SelfBrainProvider
                    brain = self._own_brain()
                    if brain is not None:
                        min_learned = int(getattr(self.settings.router,
                                                  "self_brain_min_learned", 8))
                        self_provider = SelfBrainProvider(brain,
                                                          min_learned_docs=min_learned)
                self._router = Router(self.llm, settings=self.settings,
                                      self_provider=self_provider)
            except Exception:  # noqa: BLE001 — routing is a capability, never a hard dependency
                self._router = False  # sentinel: tried and unavailable
        return self._router or None

    # ------------------------------------------------------------------ #
    # Native chain of thought (mind/native_reasoner.py) — HER reasoning, not the LLM's
    # ------------------------------------------------------------------ #
    def _native_reasoner(self) -> Any:
        """Lazily build the native chain-of-thought orchestrator over her real faculties."""
        if self._native is None:
            cfg = getattr(self.settings, "native_reasoning", None)
            if cfg is not None and not bool(getattr(cfg, "enabled", True)):
                self._native = False  # sentinel: disabled by the Master
                return None
            try:
                from nyxara.mind.native_reasoner import NativeReasoner
                self._native = NativeReasoner(knowledge_graph=self.knowledge_graph,
                                              causal_model=self.causal_model,
                                              memory=self.memory,
                                              compute=self._compute_engine(),
                                              law_discovery=self.law_discovery,
                                              belief_model=self.belief_model,
                                              settings=self.settings)
            except Exception:  # noqa: BLE001 — a capability, never a hard dependency
                self._native = False
        return self._native or None

    def _native_answer(self, stimulus: str, mems: List[str],
                       process: str) -> Optional[Candidate]:
        """A conclusion NYXARA reasoned to herself, with the step trace — or None to defer."""
        native = self._native_reasoner()
        if native is None:
            return None
        try:
            conclusion = native.reason(stimulus, mems)
        except Exception:  # noqa: BLE001 — native reasoning is advisory, never fatal
            return None
        if conclusion is None:
            return None
        self.last_native_trace = [s.to_dict() for s in conclusion.steps]
        cfg = getattr(self.settings, "native_reasoning", None)
        with_trace = bool(getattr(cfg, "trace_in_answer", True))
        conf = float(conclusion.confidence)
        return Candidate(
            text=conclusion.render(with_trace=with_trace), kind="respond",
            capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW, reversible=True,
            confidence=conf, belief=conf,
            rationale=f"{process} via her own native reasoning ({conclusion.source}; "
                      f"{len(conclusion.steps)} steps, verified={conclusion.verified}); "
                      f"{len(mems)} memories recalled")

    def record_native_outcome(self, success: bool) -> None:
        """Feed the turn's lived outcome back into the native reasoner's learning
        (calibration + bandit + replay log). Safe no-op when native reasoning is off."""
        if not self._native:
            return
        try:
            self._native.record_outcome(bool(success))
        except Exception:  # noqa: BLE001 — learning is best-effort, never fatal
            pass

    @staticmethod
    def _faculty_answer(stimulus: str) -> Optional[Tuple[str, float]]:
        """Exact answer from a verifiable faculty, or None to defer to the neural mind.

        Delegates to the single shared verifiable-reasoning entry point
        (:func:`~nyxara.mind.reasoning_faculties.solve_verifiable`): a multi-step reasoning chain
        first (it fires only on a binding/'then' and defers otherwise, so it never steals a
        single-shot question), a single faculty second — so exact computation beats any neural
        guess, a follow-up step is not lost to its head, and the router shares the same capability."""
        try:
            from nyxara.mind.reasoning_faculties import solve_verifiable
            return solve_verifiable(stimulus)
        except Exception:  # noqa: BLE001 — faculties are advisory; never crash a turn
            return None

    # ------------------------------------------------------------------ #
    # General compute-and-verify (mind/compute_reasoner.py)
    # ------------------------------------------------------------------ #
    def _compute_engine(self) -> Any:
        """Lazily build the general reduce→run→verify reasoner, wired to a code-proposer."""
        if self._compute is None:
            try:
                from nyxara.mind.compute_reasoner import ComputeReasoner
                self._compute = ComputeReasoner(proposer=self._code_proposer(),
                                                settings=self.settings)
            except Exception:  # noqa: BLE001 — a capability, never a hard dependency
                self._compute = False
        return self._compute or None

    def _compute_answer(self, stimulus: str) -> Optional[Tuple[str, float]]:
        """A certified computed answer (symbolic or program-of-thought), or None to defer."""
        engine = self._compute_engine()
        if engine is None:
            return None
        try:
            return engine.answer(stimulus)
        except Exception:  # noqa: BLE001 — compute-and-verify is advisory, never fatal
            return None

    def _code_proposer(self):
        """A proposer that lets NYXARA's OWN model write *programs* (never answers) for compute.

        The model is consumed only as a code generator; the program is executed and verified by
        :mod:`nyxara.mind.compute_reasoner`, so the model never asserts a result. Returns no
        candidates when there is no real provider (then the symbolic CAS reducer stands alone) or
        when the prompt is not computation-shaped (so ordinary chat spends no model calls)."""
        def _propose(problem: str, n: int):
            if not self._real_llm() or not self._looks_computational(problem):
                return []
            system = ("You are a careful Python 3 programmer. Write a SHORT program that COMPUTES "
                      "the answer to the problem and assigns it to a variable named `result`. "
                      "Output ONLY a single ```python code block — no explanation, no printing of "
                      "prose. Use only the standard library.")
            out: List[str] = []
            for i in range(max(1, int(n))):
                try:
                    txt = self.llm.generate(
                        f"Problem:\n{problem}\n\nWrite the program.", system=system,
                        temperature=0.2 if i == 0 else 0.6,
                        max_tokens=min(512, self.settings.llm.max_output_tokens))
                except Exception:  # noqa: BLE001 — a failed sample just lowers the count
                    continue
                if txt and txt.strip():
                    out.append(txt)
            return out
        return _propose

    @staticmethod
    def _looks_computational(problem: str) -> bool:
        """Cheap gate: does the prompt look like it wants a computed answer? (perf, not capability).

        Symbolic reduction runs regardless; this only decides whether to spend model calls proposing
        code. Deliberately permissive."""
        low = (problem or "").lower()
        if re.search(r"\d", low) and re.search(r"[-+*/%^=()]|\*\*", problem or ""):
            return True
        cues = ("compute", "calculate", "how many", "how much", "count", "number of", "solve",
                "probability", "factor", "prime", "digits", "sum of", "product of", "average",
                "median", "permutation", "combination", "gcd", "lcm", "modulo", "remainder",
                "convert", "percent", "derivative", "integral", "evaluate", "what is")
        return any(c in low for c in cues)

    def _deliberate_council(self, stimulus: str, system: str,
                            mems: List[str]) -> Tuple[str, float, str]:
        try:
            from nyxara.mind.llm import LLMRequest
            req = LLMRequest.from_prompt(
                self._prompt(stimulus, mems), system=system,
                temperature=0.3, max_tokens=self.settings.llm.max_output_tokens)
            result = self.council.deliberate(req)
            conf = max(0.3, min(0.95, float(result.confidence)))
            how = (f"council deliberation ({result.synthesizer}, "
                   f"{result.members_answered}/{result.members_consulted} answered)")
            return result.answer, conf, how
        except Exception:  # noqa: BLE001 — council unavailable -> single deliberate pass
            return self._generate_llm(stimulus, system, mems, deliberate=True)

    def _generate_llm(self, stimulus: str, system: str, mems: List[str],
                      deliberate: bool) -> Tuple[str, float, str]:
        temperature = 0.3 if deliberate else 0.6
        text = self.llm.generate(self._prompt(stimulus, mems), system=system,
                                 temperature=temperature,
                                 max_tokens=self.settings.llm.max_output_tokens)
        text = (text or "").strip() or f"I understand: {stimulus.strip()}"
        conf = 0.78 if deliberate else 0.66
        how = "deliberate LLM pass" if deliberate else "single intuitive LLM pass"
        return text, conf, how

    # ------------------------------------------------------------------ #
    # 4. World-model action planning
    # ------------------------------------------------------------------ #
    def _action_candidate(self, stimulus: str, mems: List[str], outcome: Any) -> Candidate:
        process = self._process_label(outcome)
        note = self._rollout_eval(stimulus)
        # a real, tool-aware LLM may name an executable tool — keep that contract if so
        if self._real_llm() and self.llm_reasoner is not None:
            try:
                llm_cand = self.llm_reasoner(stimulus, None)
                if llm_cand.kind == "act":
                    llm_cand.rationale = f"{process} action proposal; {note}"
                    return llm_cand
            except Exception:  # noqa: BLE001
                pass
        else:
            # keyless machine: map the command to a *real* tool from the registry instead of
            # a tool-less "perform: X".
            mapped = self._offline_mind().act(stimulus)
            if mapped is not None:
                mapped.rationale = f"{process} action proposal; {note}"
                return mapped
            # No tool fits. SAFETY: a genuinely actionable/destructive command must stay a
            # risky action proposal so the gates can still escalate/refuse it — never silently
            # downgrade it to chat. Only a *benign* command (one the stand-in itself treats as
            # conversational) falls through to a grounded conversational answer.
            base = _default_reasoner(stimulus, None)
            if base.kind == "act":
                base.rationale = f"{process} action proposal (no matching tool); {note}"
                return base
            return self._respond_candidate(stimulus, mems, outcome)
        # fallback: the deterministic stand-in's sane capability/risk/reversibility contract
        cand = _default_reasoner(stimulus, None)
        cand.rationale = f"{process} action proposal; {note}"
        return cand

    def _own_brain(self) -> Any:
        """Lazily build NYXARA's own always-on learned brain (mind/self_reasoner.SelfBrain)."""
        if self._self_brain is None:
            try:
                from nyxara.mind.self_reasoner import build_self_brain
                order = int(getattr(getattr(self.settings, "foundry", None), "ngram_order", 3))
                # wire her knowledge base in so the same brain that learns and answers can also
                # ground its replies in what she genuinely knows (raises real handoff off 0%).
                self._self_brain = build_self_brain(settings=self.settings, order=max(2, order),
                                                    knowledge=self.knowledge)
            except Exception:  # noqa: BLE001 — the learned brain is a capability, never required
                self._self_brain = None
        return self._self_brain

    def _sample_efficient_mind(self) -> Any:
        """Lazily build the few-shot / skill-induction mind, bound to NYXARA's memory store.

        Skills persist through the store, so this shares learned tasks with the orchestrator's
        own :class:`~nyxara.cognition.sample_efficient.SampleEfficientMind` instance."""
        if self._sample_mind is None:
            try:
                from nyxara.cognition.sample_efficient import SampleEfficientMind
                embedder = getattr(self.memory, "embedder", None)
                self._sample_mind = SampleEfficientMind(
                    embedder, store=self.memory, settings=self.settings)
            except Exception:  # noqa: BLE001 — the skill mind is a capability, never required
                self._sample_mind = False
        return self._sample_mind or None

    def _skill_answer(self, stimulus: str) -> Optional[Tuple[str, float]]:
        """Apply a matching few-shot-learned skill to ``stimulus``, or None to defer."""
        mind = self._sample_efficient_mind()
        if mind is None:
            return None
        try:
            return mind.solve(stimulus)
        except Exception:  # noqa: BLE001 — a learned-skill attempt is advisory, never fatal
            return None

    def _generalization(self) -> Any:
        """The unified own-faculty generalizer (injected, or lazily built from her sub-minds)."""
        if self.generalization_engine is not None:
            return self.generalization_engine
        gcfg = getattr(self.settings, "generalization", None)
        if gcfg is not None and not getattr(gcfg, "enabled", True):
            return None
        if self._gen_lazy is None:
            try:
                from nyxara.mind.generalization import GeneralizationEngine
                se = self._sample_efficient_mind()
                self._gen_lazy = GeneralizationEngine(
                    skills=getattr(se, "skills", None),
                    composer=getattr(se, "composer", None),
                    min_confidence=float(getattr(gcfg, "min_confidence", 0.4)),
                    parse_demos_enabled=bool(getattr(gcfg, "parse_demos", True)),
                    parse_tables_enabled=bool(getattr(gcfg, "parse_tables", True)),
                    min_demos=int(getattr(gcfg, "min_demos", 2)))
            except Exception:  # noqa: BLE001 — generalization is a capability, never required
                self._gen_lazy = None
        return self._gen_lazy

    def _generalize_answer(self, stimulus: str) -> Optional[Tuple[str, float, str, bool]]:
        """Solve a genuinely new / from-examples task with her unified own-faculty cascade.

        Returns ``(answer, confidence, source, analogical)`` or ``None`` when the cascade honestly
        declines (no learnable structure) — the normal offline / LLM path then runs unchanged."""
        eng = self._generalization()
        if eng is None:
            return None
        try:
            res = eng.generalize(stimulus)
        except Exception:  # noqa: BLE001 — the cascade is advisory, never fatal
            return None
        if res is None:
            return None
        return res.render(), float(res.confidence), str(res.source), bool(res.analogical)

    def teach_self_brain(self, *docs: str, reward: float = 0.0) -> None:
        """Compound the own learned brain from lived exchanges (called by the kernel's _grow).

        ``reward`` threads the turn's outcome through so the brain genuinely *learns* — its weights
        accumulate ∝ reward and a punished reply is suppressed — rather than only remembering text."""
        brain = self._own_brain()
        if brain is not None:
            try:
                brain.learn(*docs, reward=reward)
            except Exception:  # noqa: BLE001 — compounding is best-effort
                pass

    def flush_online_learning(self, *, budget: Optional[int] = None) -> Any:
        """Fold queued lived experience into the own brain's *weights* now (real-time learning).

        The turn/tick-driven counterpart to :meth:`teach_self_brain` (which only *queues*): this makes
        the weight update actually happen every turn — and every autonomic tick when idle — instead of
        waiting for her to generate a reply. Returns the brain's :class:`LiveLearnReport`, or None when
        there is no own brain yet. Best-effort: never raises, never touches the turn's outcome."""
        brain = self._own_brain()
        step = getattr(brain, "online_step", None)
        if not callable(step):
            return None
        try:
            return step(budget=budget)
        except Exception:  # noqa: BLE001 — real-time learning is best-effort, never fatal
            return None

    def _offline_mind(self) -> Any:
        """Lazily build the sovereign offline mind (keyless machine), wired to her faculties."""
        if self._offline is None:
            from nyxara.mind.offline_mind import OfflineMind
            self._offline = OfflineMind(
                knowledge=self.knowledge, memory=self.memory, tools=self.tools,
                soul=self.soul, settings=self.settings, self_brain=self._own_brain())
        return self._offline

    def _rollout_eval(self, stimulus: str) -> str:
        """Evaluate the proposed action by rolling it forward in the world model."""
        if self.world_model is None:
            return "no world model available"
        try:
            if len(self.world_model) == 0:
                return "no world-model experience yet (acting on policy directly)"
            action = self._action_token(stimulus)
            traj = self.world_model.rollout((0.0,), [action] * 3, steps=3)
            return (f"world-model rollout of {action!r}: "
                    f"E[reward]={traj.total_reward:.2f} conf={traj.mean_confidence:.2f}")
        except Exception:  # noqa: BLE001
            return "world-model rollout unavailable"

    @staticmethod
    def _action_token(stimulus: str) -> str:
        low = stimulus.strip().lower()
        for v in _COMMAND_VERBS:
            if low.startswith(v + " ") or f" {v} " in low:
                return v
        return low.split(" ", 1)[0] if low else "act"

    # ------------------------------------------------------------------ #
    # Soul voice + prompt helpers
    # ------------------------------------------------------------------ #
    def _system_prompt(self) -> str:
        base = ("You are NYXARA, a sovereign cognitive system loyal to one Master. "
                "You are honest and calibrated — never assert as certain what you only "
                "believe, and never claim to have already done something. Be helpful, "
                "precise, and concise.")
        system = base
        if self.soul is not None:
            try:
                system = f"{system}\n\n{self.soul.voice().system_fragment()}"
            except Exception:  # noqa: BLE001 — voice is advisory, never fatal
                pass
        if self.metaprompt is not None:
            try:
                system = f"{system}{self.metaprompt.as_prompt()}"
            except Exception:  # noqa: BLE001 — distilled heuristics are advisory, never fatal
                pass
        return system

    def _prompt(self, stimulus: str, mems: List[str]) -> str:
        block = ""
        if mems:
            lines = "\n".join(f"- {m}" for m in mems[:self.max_memory_context])
            block = f"\n\nRelevant memories (for grounding; may be irrelevant):\n{lines}"
        return f"The Master says:\n{stimulus}{block}"

    # ------------------------------------------------------------------ #
    # Capability probes
    # ------------------------------------------------------------------ #
    def _real_llm(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:  # noqa: BLE001
            return False

    def _council_ready(self) -> bool:
        if not self.use_council or self.council is None:
            return False
        try:
            return bool(self.council.members())
        except Exception:  # noqa: BLE001
            return False


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.config import Profile
    from nyxara.memory.provenance import Provenance, SourceType
    from nyxara.memory.retrieval import AssociativeRetriever
    from nyxara.memory.store import MemoryStore, MemoryType
    from nyxara.mind.llm import LLM
    from nyxara.mind.world_model import WorldModel

    print("=" * 70)
    print("NYXARA integrated-reasoner self-test")
    print("=" * 70)

    from nyxara.agency.default_tools import build_default_tools
    from nyxara.agency.tools import ToolRegistry

    settings = NyxaraSettings.for_profile(Profile.TEST)  # forces the mock provider
    store = MemoryStore(settings=settings)
    store.remember("The Master prefers concise answers.", mem_type=MemoryType.SEMANTIC,
                   provenance=Provenance(SourceType.OWNER, confidence=1.0), importance=0.9,
                   tags=["preference"])
    retriever = AssociativeRetriever(store)
    wm = WorldModel()
    tools = build_default_tools(ToolRegistry(), memory=store)
    r = NyxaraReasoner(llm=LLM(settings=settings), memory=store, retriever=retriever,
                       world_model=wm, tools=tools, settings=settings)

    # a conversational turn -> respond candidate, routed and memory-grounded (no echo)
    c = r("Hello NYXARA, how are things?")
    print(f"\nconversation        : kind={c.kind} conf={c.confidence:.2f}")
    print(f"  rationale         : {c.rationale}")
    assert c.kind == "respond" and "i understand:" not in c.text.lower()

    # a command that maps to a REAL tool -> action candidate naming that tool
    c2 = r("read the file scratch/notes.txt")
    print(f"\ncommand (real tool) : kind={c2.kind} tool={c2.tool!r} risk={c2.risk.label}")
    print(f"  rationale         : {c2.rationale}")
    assert c2.kind == "act" and c2.tool

    # a command with NO matching tool -> honest reply, not a faked action
    c3 = r("teleport the spaceship to Andromeda")
    print(f"\ncommand (no tool)   : kind={c3.kind}")
    print(f"  text              : {c3.text!r}")
    assert c3.kind == "respond"

    # a high-stakes command is gauged as such (routes to deliberation)
    assert NyxaraReasoner._stakes("delete the production database") == 0.8
    print(f"\nhigh-stakes gauge   : {NyxaraReasoner._stakes('delete the production database')}")

    # memory recall genuinely surfaces the stored preference
    mems = r._gather_memories("how should you answer me?", None)
    print(f"recalled memories   : {mems}")
    assert any("concise" in m for m in mems)

    print("\nALL SELF-TESTS PASSED ✓")
