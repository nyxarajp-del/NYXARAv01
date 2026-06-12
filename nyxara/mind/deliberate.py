"""NYXARA · mind/deliberate.py — multi-pass deliberate reasoning (⬆, think-before-decide).

The kernel reasoner (``mind/llm_reasoner.py``) turns a stimulus into a structured
*decision* the kernel then disposes. Out of the box it did this in a **single** LLM call.
A single shot is fast but shallow: the model commits before it reasons, can't catch its
own mistakes, and is at the mercy of one sample's variance.

This module adds **deliberation** — the same kind of "think, then answer, then check your
work" loop that lifts a capable model's real-world quality. It is provider-agnostic (it
only needs ``llm.generate`` / ``llm.generate_json``) and fully graceful: every extra stage
is wrapped so that any failure simply falls back to the best decision so far. The output is
the *same* decision dict the reasoner already understands — so safety is untouched: the LLM
still only ever *proposes*; the kernel still disposes through every gate.

Stages (driven by ``LLMConfig.reasoning_passes``):

1. **Think** — a private free-form scratchpad. The model reasons step by step about how to
   serve the Master given the recalled memories, learned skills, and available tools —
   *without* committing to an answer. This is "extended thinking" the model can lean on.
2. **Decide** — convert the stimulus + scratchpad into the structured JSON decision. With
   ``reasoning_samples > 1`` this step is sampled several times and the consensus
   (self-consistency) is taken, which markedly stabilises the answer.
3. **Critique & revise** — feed the decision back, run the deterministic cognitive-bias
   :class:`~nyxara.mind.critique.Critic` *and* ask the model to find flaws (dishonesty,
   overconfidence, wrong/unsafe tool, missed intent), then emit a possibly-revised
   decision. The revision is only adopted if it parses; otherwise the original stands.

Stateless per call, like the faculties it composes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Deliberation", "DeliberativeReasoner"]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class Deliberation:
    """The outcome of a deliberation: the final decision plus an auditable trace."""

    decision: Dict[str, Any]
    passes: int = 1
    scratchpad: str = ""
    samples: List[Dict[str, Any]] = field(default_factory=list)
    critique: str = ""
    revised: bool = False

    def trace(self) -> str:
        """A short, human-readable note on how the decision was reached (for /explain)."""
        bits = [f"{self.passes}-pass deliberation"]
        if len(self.samples) > 1:
            bits.append(f"{len(self.samples)} self-consistency samples")
        if self.revised:
            bits.append("self-critique revised the draft")
        return "; ".join(bits)


# --------------------------------------------------------------------------- #
# The deliberator
# --------------------------------------------------------------------------- #
class DeliberativeReasoner:
    """Run a think → decide → critique loop over an LLM, returning a decision dict."""

    _THINK_INSTRUCTIONS = (
        "First THINK. In a few short sentences, reason privately about how best to serve "
        "the Master: what they actually want, whether a tool is genuinely needed (and which), "
        "what could go wrong, and what an honest, well-scoped response looks like. Weigh at "
        "least one alternative. Do NOT answer the Master yet and do NOT output JSON — this is "
        "your private scratchpad."
    )

    def __init__(self, llm: Any, *, passes: int = 2, samples: int = 1,
                 think_tokens: int = 1024, temperature: float = 0.5,
                 max_tokens: int = 4096, critic: Any = None) -> None:
        self.llm = llm
        self.passes = max(1, int(passes))
        self.samples = max(1, int(samples))
        self.think_tokens = max(64, int(think_tokens))
        self.temperature = max(0.0, float(temperature))
        self.max_tokens = max(1, int(max_tokens))
        self.critic = critic

    # ---- public entry point ---- #
    def deliberate(self, *, stimulus: str, context: str, decision_instructions: str,
                   system: Optional[str] = None) -> Deliberation:
        """Produce a decision dict for ``stimulus``, deliberating per ``passes``/``samples``."""
        scratchpad = self._think(stimulus, context, system) if self.passes >= 2 else ""

        samples = self._decide_samples(stimulus, context, scratchpad,
                                       decision_instructions, system)
        if not samples:
            # The decide stage produced nothing parseable — signal failure to the caller,
            # which falls back to its own single-shot / deterministic path.
            raise ValueError("deliberation produced no parseable decision")
        decision = _vote(samples)

        result = Deliberation(decision=decision, passes=self.passes,
                              scratchpad=scratchpad, samples=samples)

        if self.passes >= 3:
            revised = self._critique_and_revise(
                stimulus, context, decision, decision_instructions, system, result)
            if revised is not None:
                result.decision = revised
                result.revised = True

        # Thread a short trace into the rationale so /explain can show the deliberation.
        self._annotate(result)
        return result

    # ---- stage 1: think ---- #
    def _think(self, stimulus: str, context: str, system: Optional[str]) -> str:
        prompt = (f"The Master says:\n{stimulus}{context}\n\n{self._THINK_INSTRUCTIONS}")
        try:
            text = self.llm.generate(prompt, system=system,
                                     temperature=self.temperature,
                                     max_tokens=self.think_tokens)
            return str(text or "").strip()
        except Exception:  # noqa: BLE001 — thinking is advisory, never fatal
            return ""

    # ---- stage 2: decide (with optional self-consistency) ---- #
    def _decide_samples(self, stimulus: str, context: str, scratchpad: str,
                        decision_instructions: str,
                        system: Optional[str]) -> List[Dict[str, Any]]:
        think_block = (f"\n\nYour private reasoning (use it; do not repeat it verbatim):"
                       f"\n{scratchpad}" if scratchpad else "")
        prompt = (f"The Master says:\n{stimulus}{context}{think_block}\n\n"
                  f"{decision_instructions}")
        out: List[Dict[str, Any]] = []
        n = self.samples
        for i in range(n):
            # First sample stays low-temperature/deterministic; extra samples add diversity
            # so the vote is meaningful rather than N identical greedy decodes.
            temp = self.temperature if (n == 1 or i == 0) else min(1.0, self.temperature + 0.3)
            try:
                data = self.llm.generate_json(prompt, system=system, temperature=temp,
                                              max_tokens=self.max_tokens)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(data, dict):
                out.append(data)
        return out

    # ---- stage 3: critique & revise ---- #
    def _critique_and_revise(self, stimulus: str, context: str, decision: Dict[str, Any],
                             decision_instructions: str, system: Optional[str],
                             result: Deliberation) -> Optional[Dict[str, Any]]:
        probes = self._deterministic_probes(decision)
        probe_block = ("\n\nMechanical checks already flagged:\n" +
                       "\n".join(f"- {p}" for p in probes) if probes else "")
        result.critique = "; ".join(probes)
        prompt = (
            f"The Master said:\n{stimulus}{context}\n\n"
            f"You drafted this decision:\n{json.dumps(decision, ensure_ascii=False)}"
            f"{probe_block}\n\n"
            "CRITIQUE your own draft as a sceptic: is it honest and calibrated? Does it claim "
            "to have already done something instead of proposing it? Is the chosen tool the "
            "right one (or is a tool used when none is needed)? Is the confidence justified? "
            "Then output your final, improved decision.\n\n"
            f"{decision_instructions}")
        try:
            data = self.llm.generate_json(prompt, system=system,
                                          temperature=min(self.temperature, 0.3),
                                          max_tokens=self.max_tokens)
        except Exception:  # noqa: BLE001
            return None
        return data if isinstance(data, dict) else None

    def _deterministic_probes(self, decision: Dict[str, Any]) -> List[str]:
        """Cheap, dependency-free bias/calibration probes via the existing Critic."""
        if self.critic is None:
            return []
        try:
            from nyxara.mind.critique import CritiqueContext
            ctx = CritiqueContext(
                claim=str(decision.get("text", "")),
                confidence=_as_float(decision.get("confidence"), 0.7),
                rationale=str(decision.get("rationale", "")),
                risk=_risk_to_float(decision.get("risk")),
                reversibility=1.0 if decision.get("reversible", True) else 0.2,
            )
            crit = self.critic.critique(ctx)
            return [f.description for f in crit.findings][:4] + list(crit.probes[:2])
        except Exception:  # noqa: BLE001 — probes are advisory
            return []

    # ---- trace ---- #
    def _annotate(self, result: Deliberation) -> None:
        try:
            base = str(result.decision.get("rationale", "")).strip()
            note = result.trace()
            result.decision["rationale"] = f"{base} [{note}]" if base else note
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _risk_to_float(v: Any) -> float:
    return {"trivial": 0.05, "low": 0.2, "moderate": 0.5, "medium": 0.5,
            "high": 0.8, "critical": 0.95}.get(str(v).lower(), 0.2)


def _vote(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Self-consistency: pick the consensus action *shape*, then its most-confident member.

    Decisions are structured, not single tokens, so we vote on the (kind, tool) shape — the
    part that actually changes what the kernel does — and within the winning bucket return
    the highest-confidence sample as the representative answer.
    """
    if len(samples) == 1:
        return samples[0]

    def shape(d: Dict[str, Any]) -> tuple:
        kind = "act" if str(d.get("kind", "respond")).lower() == "act" else "respond"
        tool = str(d.get("tool") or "").strip().lower() if kind == "act" else ""
        return (kind, tool)

    counts = Counter(shape(d) for d in samples)
    top = counts.most_common(1)[0][1]
    winners = {sh for sh, c in counts.items() if c == top}
    bucket = [d for d in samples if shape(d) in winners]
    # Tie-break (and within-bucket pick): highest stated confidence.
    return max(bucket, key=lambda d: _as_float(d.get("confidence"), 0.0))


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA deliberate-reasoner self-test")
    print("=" * 70)

    class _FakeLLM:
        """A scripted LLM: thinks once, then returns drifting JSON decisions."""

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt, *, system=None, **kw):
            return "The Master wants a status check; no tool needed; answer honestly."

        def generate_json(self, prompt, *, system=None, **kw):
            self.calls += 1
            # majority say 'respond'; one outlier says 'act' to test the vote
            if self.calls == 2:
                return {"kind": "act", "tool": "noop", "text": "do it", "confidence": 0.4}
            return {"kind": "respond", "text": "All nominal, Master.",
                    "confidence": 0.6 + 0.1 * self.calls, "rationale": "status reply"}

    from nyxara.mind.critique import Critic

    llm = _FakeLLM()
    d = DeliberativeReasoner(llm, passes=3, samples=3, critic=Critic())
    out = d.deliberate(stimulus="status?", context="", decision_instructions="(JSON spec)")
    print(f"\nscratchpad : {out.scratchpad!r}")
    print(f"samples    : {len(out.samples)}")
    print(f"decision   : kind={out.decision['kind']} conf={out.decision.get('confidence')}")
    print(f"trace      : {out.trace()}")
    assert out.decision["kind"] == "respond"           # majority vote wins over the outlier
    assert out.passes == 3 and len(out.samples) == 3
    assert "deliberation" in out.decision["rationale"]  # trace annotated

    single = DeliberativeReasoner(_FakeLLM(), passes=1, samples=1)
    out1 = single.deliberate(stimulus="hi", context="", decision_instructions="(spec)")
    assert out1.scratchpad == "" and out1.passes == 1
    print("\nALL SELF-TESTS PASSED ✓")
