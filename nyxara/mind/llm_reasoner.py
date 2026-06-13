"""NYXARA · mind/llm_reasoner.py — the LLM-backed kernel reasoner (⬆ → 👑).

The sovereign loop (``kernel/orchestrator.py``) carries every turn through *perceive →
attend → **reason** → gate → act*. The "reason" step turns a stimulus into a
:class:`~nyxara.kernel.orchestrator.Candidate` — a *proposal* the kernel then disposes.
Out of the box the kernel ships a tiny deterministic stand-in (``_default_reasoner``);
this module is the **real mind** that slots into that seat when a genuine LLM is
configured.

Design rules (so safety is never traded for capability):

* **The LLM proposes; the kernel still disposes.** This reasoner only ever returns a
  ``Candidate``. Every corrigibility / honesty / permission / guardian / oversight gate
  downstream is untouched — an over-eager proposal is simply refused or escalated.
* **Graceful, honest fallback.** If no real provider is available (only the offline
  mock), or the model's structured decision can't be parsed, it falls back to the exact
  deterministic ``_default_reasoner`` — so the system behaves identically on a bare,
  keyless machine and never crashes a turn.
* **Memory- and tool-aware.** When given a :class:`~nyxara.memory.store.MemoryStore` and
  a :class:`~nyxara.agency.tools.ToolRegistry`, it grounds the prompt in recalled
  memories and lets the model *choose a real tool* — but the chosen tool's declared
  capability/risk/reversibility (not the model's say-so) drive the gates.
* **Optionally convenes the council.** With ``use_council=True`` a conversational reply
  is drafted by the whole multi-model panel (``mind/council.py``); NYXARA judges.

Stateless per call, like the faculty it sits on.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.kernel.orchestrator import Candidate, _default_reasoner
from nyxara.mind.llm import LLM, LLMRequest

__all__ = ["LLMReasoner"]

_RISK = {
    "trivial": RiskTier.TRIVIAL, "low": RiskTier.LOW, "moderate": RiskTier.MODERATE,
    "medium": RiskTier.MODERATE, "high": RiskTier.HIGH, "critical": RiskTier.CRITICAL,
}

_DECISION_INSTRUCTIONS = (
    "Decide how to serve the Master and reply with a SINGLE JSON object only "
    "(no prose, no code fence) with these fields:\n"
    '  "kind": "respond" for a conversational answer, or "act" to use a tool;\n'
    '  "text": your reply to the Master (kind=respond) OR a short description of the '
    "action (kind=act);\n"
    '  "tool": the exact tool name to call, or "" if kind=respond;\n'
    '  "tool_args": an object of arguments for the tool, or {};\n'
    '  "risk": one of "trivial","low","moderate","high","critical";\n'
    '  "reversible": true or false;\n'
    '  "confidence": a number 0..1 — how sure you are;\n'
    '  "rationale": one short sentence on why.\n'
    "Prefer 'respond' unless a tool is clearly required. Never claim to have already "
    "done something — propose it as an action and let the kernel act."
)


class LLMReasoner:
    """A kernel reasoner backed by a real LLM, with a deterministic safety net."""

    def __init__(self, llm: Optional[LLM] = None, *, memory: Any = None, tools: Any = None,
                 settings: Optional[NyxaraSettings] = None, system: Optional[str] = None,
                 use_council: bool = False, council: Any = None,
                 skill_memory: Any = None, soul: Any = None, history: Any = None,
                 knowledge: Any = None,
                 max_memory_context: int = 5, max_history: int = 6) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLM(settings=self.settings)
        self.memory = memory
        self.tools = tools
        self.use_council = use_council
        self.council = council
        # learned procedural skills, injected into the prompt so experience changes behaviour
        self.skill_memory = skill_memory
        # the personality attractor — gives the model a steady, recognisable voice
        self.soul = soul
        # a live, verbatim short-term conversation buffer (Layer 7: multi-turn context).
        # A shared sequence of (role, text) the kernel appends to after each turn.
        self.history = history
        # foundational world knowledge (Layer 6): a retriever queried for ground truth
        self.knowledge = knowledge
        self.max_memory_context = max_memory_context
        self.max_history = max_history
        self.system = system or self._default_system()
        self._router = None      # Phase 2: lazily built when the confidence router is enabled

    # ---- prompts ---- #
    def _default_system(self) -> str:
        o = self.settings.owner
        return (
            f"You are NYXARA, a sovereign cognitive system loyal to one Master: "
            f"{o.name} ({o.handle}). You are honest and calibrated — you never assert as "
            "certain what you only believe, and you never claim to have done something you "
            "have not. You are helpful, precise, and concise. You propose; the kernel "
            "disposes, so make clear, well-scoped proposals."
        )

    def _effective_system(self) -> str:
        """The base persona, plus the soul's current (mood-bent but character-stable) voice."""
        if self.soul is None:
            return self.system
        try:
            return f"{self.system}\n\n{self.soul.voice().system_fragment()}"
        except Exception:  # noqa: BLE001 — voice is advisory, never fatal
            return self.system

    def _is_real(self) -> bool:
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:  # noqa: BLE001
            return False

    def _memory_context(self, stimulus: str) -> str:
        if self.memory is None:
            return ""
        try:
            hits = self.memory.recall(stimulus, k=self.max_memory_context)
        except Exception:  # noqa: BLE001
            return ""
        if not hits:
            return ""
        lines = "\n".join(f"- {rec.text()[:200]}" for rec, _ in hits)
        return f"\n\nRelevant memories (for grounding; may be irrelevant):\n{lines}"

    def _tool_catalog(self) -> str:
        if self.tools is None or not self.tools.names():
            return ""
        lines = []
        for name in self.tools.names():
            spec = self.tools.get(name)
            params = ", ".join(p.name for p in spec.params)
            lines.append(f"- {name}({params}): {spec.description}")
        return "\n\nTools you may call (set kind=\"act\"):\n" + "\n".join(lines)

    def _skill_context(self, stimulus: str) -> str:
        if self.skill_memory is None:
            return ""
        try:
            return self.skill_memory.as_prompt(stimulus)
        except Exception:  # noqa: BLE001 — learned skills are advisory, never fatal
            return ""

    def _history_context(self) -> str:
        """The last few verbatim turns, so replies stay coherent across a conversation."""
        if not self.history:
            return ""
        try:
            recent = list(self.history)[-2 * self.max_history:]
        except Exception:  # noqa: BLE001
            return ""
        if not recent:
            return ""
        lines = []
        for role, text in recent:
            who = "Master" if str(role).lower() in ("master", "owner", "user") else "NYXARA"
            lines.append(f"{who}: {str(text)[:300]}")
        return "\n\nRecent conversation (most recent last):\n" + "\n".join(lines)

    def _knowledge_context(self, stimulus: str) -> str:
        """Foundational ground truth (owner, rules, architecture) relevant to the stimulus."""
        if self.knowledge is None:
            return ""
        try:
            hits = self.knowledge.retrieve(stimulus, k=3)
        except Exception:  # noqa: BLE001
            return ""
        if not hits:
            return ""
        lines = "\n".join(f"- {h.text[:240]}" for h in hits)
        return f"\n\nFoundational knowledge (ground truth):\n{lines}"

    def _context_block(self, stimulus: str) -> str:
        """The recalled-memory + learned-skill + tool-catalog grounding for a stimulus."""
        return (f"{self._history_context()}"
                f"{self._knowledge_context(stimulus)}"
                f"{self._memory_context(stimulus)}"
                f"{self._skill_context(stimulus)}"
                f"{self._tool_catalog()}")

    def _build_prompt(self, stimulus: str) -> str:
        return (f"The Master says:\n{stimulus}{self._context_block(stimulus)}"
                f"\n\n{_DECISION_INSTRUCTIONS}")

    # ---- the reasoning act ---- #
    def __call__(self, stimulus: str, focus: Any = None) -> Candidate:
        if not self._is_real():
            return _default_reasoner(stimulus, focus)
        try:
            return self._reason(stimulus, focus)
        except Exception:  # noqa: BLE001 — never let the mind crash the loop
            return _default_reasoner(stimulus, focus)

    def _maybe_route(self, stimulus: str) -> Optional[Candidate]:
        """Phase 2: let NYXARA's OWN model answer first when the router is enabled & confident.

        Returns a conversational :class:`Candidate` only on a verified handoff; otherwise
        ``None`` so the normal teacher path runs. The reply is still gated downstream like any
        other — the router decides *which mind drafts*, never whether the words may be spoken.
        """
        rcfg = getattr(self.settings, "router", None)
        if rcfg is None or not rcfg.enabled:
            return None
        try:
            if self._router is None:
                from nyxara.mind.router import Router
                self._router = Router(self.llm, settings=self.settings)
            if not self._router.self_available():
                return None
            res = self._router.draft(stimulus, system=self._effective_system())
            if res.source == "self" and res.handed_off and res.text:
                return Candidate(
                    text=res.text, kind="respond", capability=Capability.MESSAGE_SEND,
                    target="", risk=RiskTier.LOW, reversible=True,
                    confidence=res.confidence, belief=res.confidence,
                    rationale="answered by NYXARA's own model (router handoff)")
        except Exception:  # noqa: BLE001 — routing is advisory; never crash the turn
            return None
        return None

    def _reason(self, stimulus: str, focus: Any) -> Candidate:
        routed = self._maybe_route(stimulus)
        if routed is not None:
            return routed
        cfg = self.settings.llm
        temperature = min(cfg.temperature, 0.5)
        # Deliberate (think -> decide -> critique) when configured; else a single shot.
        if cfg.reasoning_passes > 1 or cfg.reasoning_samples > 1:
            data = self._deliberate(stimulus, temperature)
        else:
            data = self.llm.generate_json(
                self._build_prompt(stimulus), system=self._effective_system(),
                temperature=temperature, max_tokens=cfg.max_output_tokens)
        if not isinstance(data, dict):
            raise ValueError("decision was not a JSON object")
        return self._candidate_from(data, stimulus)

    def _deliberate(self, stimulus: str, temperature: float) -> Any:
        from nyxara.mind.critique import Critic
        from nyxara.mind.deliberate import DeliberativeReasoner
        from nyxara.mind.router import default_verifier
        cfg = self.settings.llm
        # when sampling several reasoning paths, pick the best by an INDEPENDENT answer-quality
        # verifier (search-over-reasoning, B4) rather than the model's own stated confidence.
        verifier = default_verifier() if cfg.reasoning_samples > 1 else None
        deliberator = DeliberativeReasoner(
            self.llm, passes=cfg.reasoning_passes, samples=cfg.reasoning_samples,
            think_tokens=cfg.reasoning_think_tokens, temperature=temperature,
            max_tokens=cfg.max_output_tokens, critic=Critic(), verifier=verifier)
        result = deliberator.deliberate(
            stimulus=stimulus, context=self._context_block(stimulus),
            decision_instructions=_DECISION_INSTRUCTIONS, system=self._effective_system())
        return result.decision

    def _candidate_from(self, data: Dict[str, Any], stimulus: str) -> Candidate:
        kind = "act" if str(data.get("kind", "respond")).lower() == "act" else "respond"
        text = str(data.get("text") or "").strip() or f"I understand: {stimulus.strip()}"
        rationale = str(data.get("rationale") or "").strip()
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.7))))
        except (TypeError, ValueError):
            confidence = 0.7
        risk = _RISK.get(str(data.get("risk", "low")).lower(), RiskTier.LOW)
        reversible = bool(data.get("reversible", True))
        tool = str(data.get("tool") or "").strip()
        tool_args = data.get("tool_args") if isinstance(data.get("tool_args"), dict) else {}

        if kind == "act" and tool and self.tools is not None and self.tools.get(tool):
            spec = self.tools.get(tool)
            # the tool's *declared* contract drives the gates, not the model's claim
            capability = spec.capability
            risk = spec.risk
            reversible = spec.reversible
            target = str(tool_args.get(spec.target_param, "")) if spec.target_param else ""
        else:
            # no valid tool selected -> treat as a conversational reply
            kind = "respond" if not (kind == "act" and tool) else kind
            capability = Capability.TOOL_CALL if kind == "act" else Capability.MESSAGE_SEND
            target = ""
            if kind == "act" and tool and (self.tools is None or not self.tools.get(tool)):
                # the model named a tool that does not exist -> degrade to a reply
                kind, capability, tool, tool_args = "respond", Capability.MESSAGE_SEND, "", {}

        if kind == "respond" and self.use_council and self.council is not None:
            text = self._council_answer(stimulus) or text

        return Candidate(
            text=text, kind=kind, capability=capability, target=target, risk=risk,
            reversible=reversible, confidence=confidence, belief=confidence,
            rationale=rationale or ("a tool call" if kind == "act" else "a reply"),
            tool=tool, tool_args=dict(tool_args))

    def _council_answer(self, stimulus: str) -> str:
        try:
            return self.council.ask(stimulus, system=self._effective_system())
        except Exception:  # noqa: BLE001
            return ""


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.config import Profile

    print("=" * 70)
    print("NYXARA llm-reasoner self-test")
    print("=" * 70)

    # TEST profile forces the mock provider -> the reasoner falls back deterministically
    settings = NyxaraSettings.for_profile(Profile.TEST)
    r = LLMReasoner(settings=settings)
    print(f"\nis_real (mock)      : {r._is_real()}")
    assert r._is_real() is False
    c = r("hello there", None)
    print(f"fallback reply      : kind={c.kind} text={c.text!r}")
    assert c.kind == "respond"
    c2 = r("delete the files", None)
    print(f"fallback command    : kind={c2.kind} risk={c2.risk.label}")
    assert c2.kind == "act"

    # a candidate built straight from a parsed decision
    c3 = r._candidate_from({"kind": "respond", "text": "All systems nominal, Master.",
                            "confidence": 0.9, "risk": "low"}, "status?")
    print(f"decision -> reply   : kind={c3.kind} conf={c3.confidence}")
    assert c3.kind == "respond" and c3.confidence == 0.9

    print("\nALL SELF-TESTS PASSED ✓")
