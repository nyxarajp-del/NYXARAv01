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
                 max_memory_context: int = 5) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLM(settings=self.settings)
        self.memory = memory
        self.tools = tools
        self.use_council = use_council
        self.council = council
        self.max_memory_context = max_memory_context
        self.system = system or self._default_system()

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

    def _build_prompt(self, stimulus: str) -> str:
        return (f"The Master says:\n{stimulus}"
                f"{self._memory_context(stimulus)}"
                f"{self._tool_catalog()}\n\n{_DECISION_INSTRUCTIONS}")

    # ---- the reasoning act ---- #
    def __call__(self, stimulus: str, focus: Any = None) -> Candidate:
        if not self._is_real():
            return _default_reasoner(stimulus, focus)
        try:
            return self._reason(stimulus, focus)
        except Exception:  # noqa: BLE001 — never let the mind crash the loop
            return _default_reasoner(stimulus, focus)

    def _reason(self, stimulus: str, focus: Any) -> Candidate:
        prompt = self._build_prompt(stimulus)
        data = self.llm.generate_json(
            prompt, system=self.system,
            temperature=min(self.settings.llm.temperature, 0.5),
            max_tokens=self.settings.llm.max_output_tokens)
        if not isinstance(data, dict):
            raise ValueError("decision was not a JSON object")
        return self._candidate_from(data, stimulus)

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
            return self.council.ask(stimulus, system=self.system)
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
