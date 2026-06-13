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

from typing import Any, List, Optional, Tuple

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.kernel.orchestrator import Candidate, _default_reasoner

__all__ = ["NyxaraReasoner"]

# stimulus cues used to gauge stakes and to spot action-like commands (mirrors the
# deterministic stand-in so routing stays consistent with the fallback)
_COMMAND_VERBS = ("delete", "remove", "shutdown", "kill", "run", "exec", "install",
                  "block", "open", "disable", "rotate", "deploy")
_HIGH_STAKES = ("delete", "shutdown", "kill", "exec", "attack", "intrusion", "breach",
                "password", "secret", "credential", "production", "database", "transfer",
                "payment", "money", "deploy", "wipe", "format", "root")


class NyxaraReasoner:
    """The kernel's reason step, wired to the whole mind. Stateless per call."""

    def __init__(self, *, llm: Any = None, council: Any = None, memory: Any = None,
                 retriever: Any = None, soul: Any = None, world_model: Any = None,
                 tools: Any = None, llm_reasoner: Any = None, use_council: bool = False,
                 settings: Optional[NyxaraSettings] = None, narrative: Any = None,
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
        # the tool-aware, real-LLM JSON decider (mind/llm_reasoner.py); used as the
        # generation engine when a genuine provider is available, so tool selection is kept.
        self.llm_reasoner = llm_reasoner
        self.use_council = use_council
        self.max_memory_context = max_memory_context
        self._dual: Any = None  # lazily built dual-process facade

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
        if not self._real_llm():
            cand = _default_reasoner(stimulus, None)
            cand.rationale = (f"{process} via deterministic faculty reasoner "
                              f"(no LLM configured); {len(mems)} memories recalled")
            return cand

        system = self._system_prompt()
        deliberate = self._is_system2(outcome)
        if deliberate and self._council_ready():
            text, conf, how = self._deliberate_council(stimulus, system, mems)
        else:
            text, conf, how = self._generate_llm(stimulus, system, mems, deliberate)
        rationale = f"{process}: {how}; grounded in {len(mems)} recalled memories"
        return Candidate(text=text, kind="respond", capability=Capability.MESSAGE_SEND,
                         risk=RiskTier.LOW, reversible=True, confidence=conf, belief=conf,
                         rationale=rationale)

    @staticmethod
    def _faculty_answer(stimulus: str) -> Optional[Tuple[str, float]]:
        """Exact answer from a verifiable faculty, or None to defer to the neural mind."""
        try:
            from nyxara.mind.reasoning_faculties import solve_with_faculties
            return solve_with_faculties(stimulus)
        except Exception:  # noqa: BLE001 — faculties are advisory; never crash a turn
            return None

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
        # the deterministic stand-in sets a sane capability/risk/reversibility contract
        cand = _default_reasoner(stimulus, None)
        if cand.kind != "act":
            return cand
        note = self._rollout_eval(stimulus)
        # a real, tool-aware LLM may name an executable tool — keep that contract if so
        if self._real_llm() and self.llm_reasoner is not None:
            try:
                llm_cand = self.llm_reasoner(stimulus, None)
                if llm_cand.kind == "act":
                    cand = llm_cand
            except Exception:  # noqa: BLE001
                pass
        cand.rationale = f"{process} action proposal; {note}"
        return cand

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
        if self.soul is None:
            return base
        try:
            return f"{base}\n\n{self.soul.voice().system_fragment()}"
        except Exception:  # noqa: BLE001 — voice is advisory, never fatal
            return base

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

    settings = NyxaraSettings.for_profile(Profile.TEST)  # forces the mock provider
    store = MemoryStore(settings=settings)
    store.remember("The Master prefers concise answers.", mem_type=MemoryType.SEMANTIC,
                   provenance=Provenance(SourceType.OWNER, confidence=1.0), importance=0.9,
                   tags=["preference"])
    retriever = AssociativeRetriever(store)
    wm = WorldModel()
    r = NyxaraReasoner(llm=LLM(settings=settings), memory=store, retriever=retriever,
                       world_model=wm, settings=settings)

    # a conversational turn -> respond candidate, routed and memory-grounded
    c = r("Hello NYXARA, how are things?")
    print(f"\nconversation        : kind={c.kind} conf={c.confidence:.2f}")
    print(f"  rationale         : {c.rationale}")
    assert c.kind == "respond"

    # a command -> action candidate, world-model evaluated
    c2 = r("rotate the application logs")
    print(f"\ncommand             : kind={c2.kind} risk={c2.risk.label}")
    print(f"  rationale         : {c2.rationale}")
    assert c2.kind == "act"

    # a high-stakes command is gauged as such (routes to deliberation)
    assert NyxaraReasoner._stakes("delete the production database") == 0.8
    print(f"\nhigh-stakes gauge   : {NyxaraReasoner._stakes('delete the production database')}")

    # memory recall genuinely surfaces the stored preference
    mems = r._gather_memories("how should you answer me?", None)
    print(f"recalled memories   : {mems}")
    assert any("concise" in m for m in mems)

    print("\nALL SELF-TESTS PASSED ✓")
