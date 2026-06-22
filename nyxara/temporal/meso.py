"""NYXARA · temporal/meso.py — Layer 2 (seconds): reading prompts, writing code.

The middle layer watches NYXARA do her ordinary work: every sovereign turn — a prompt
read, an answer or action written — is ingested as a :class:`TurnObservation` (its
disposition, whether it used a tool, whether it produced *code*, its latency and
confidence, and the hour it happened). A rolling window of those turns rolls up into a
:class:`MesoAggregate`, the message Layer 2 hands to the slow Master AI above it.

It is deliberately cheap and side-effect free: it only records. The orchestrator calls
:meth:`observe` once per turn from the single ``_finish`` choke-point, so nothing in the
hot path depends on it succeeding. Pure standard library.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Optional

from nyxara.temporal.clock import SECOND, RealClock, TemporalClock
from nyxara.temporal.signals import MesoAggregate, TurnObservation

__all__ = ["MesoLayer", "looks_like_code"]

# surface markers that an answer/action carried real source code
_CODE_MARKERS = (
    "```", "def ", "class ", "import ", "function ", "return ", "#include",
    "public static", "console.log", "print(", "=>", "</", "{\n", "();",
)


def looks_like_code(text: str) -> bool:
    """A cheap, honest heuristic: did this text contain source code?"""
    if not text:
        return False
    return any(marker in text for marker in _CODE_MARKERS)


class MesoLayer:
    """Layer 2 — a second-cadence observer of NYXARA's turns (prompts in, code out)."""

    scale_s: float = SECOND

    def __init__(self, *, clock: Optional[TemporalClock] = None, window: int = 512) -> None:
        self.clock: TemporalClock = clock or RealClock()
        self._turns: Deque[TurnObservation] = deque(maxlen=max(8, window))
        self.observed = 0
        self.aggregates = 0

    def observe(self, stimulus: Any, result: Any, *, latency_s: float = 0.0,
                authority: str = "") -> TurnObservation:
        """Record one sovereign turn. ``result`` is duck-typed against ``CycleResult`` so
        this never hard-depends on the orchestrator's import graph."""
        now = self.clock.now()
        disp = getattr(result, "disposition", "")
        disp = getattr(disp, "value", str(disp))
        candidate = getattr(result, "candidate", None)
        confidence = 0.0
        cand_text = ""
        if candidate is not None:
            try:
                confidence = float(getattr(candidate, "confidence", 0.0) or 0.0)
                cand_text = str(getattr(candidate, "text", "") or "")
            except Exception:  # noqa: BLE001
                pass
        response = str(getattr(result, "response", "") or "")
        obs = TurnObservation(
            at=now,
            stimulus_len=len(str(stimulus or "")),
            disposition=disp,
            tool=getattr(result, "tool", None),
            produced_code=looks_like_code(response) or looks_like_code(cand_text),
            response_len=len(response),
            latency_s=max(0.0, float(latency_s)),
            confidence=confidence,
            authority=authority,
            hour_of_day=time.localtime(now).tm_hour,
        )
        self._turns.append(obs)
        self.observed += 1
        return obs

    def aggregate(self) -> MesoAggregate:
        """Roll the current window of turns up into one Layer-2 → Layer-3 message."""
        agg = MesoAggregate(at=self.clock.now())
        if not self._turns:
            return agg
        turns = list(self._turns)
        agg.n_turns = len(turns)
        agg.span_s = max(0.0, turns[-1].at - turns[0].at)
        agg.acted = sum(1 for t in turns if t.disposition == "act")
        agg.escalated = sum(1 for t in turns if t.disposition == "escalate")
        agg.refused = sum(1 for t in turns if t.disposition in ("refuse", "halt"))
        agg.tool_calls = sum(1 for t in turns if t.tool)
        agg.code_turns = sum(1 for t in turns if t.produced_code)
        agg.owner_turns = sum(1 for t in turns if t.authority == "owner")
        agg.avg_latency_s = sum(t.latency_s for t in turns) / len(turns)
        agg.avg_confidence = sum(t.confidence for t in turns) / len(turns)
        agg.hours_active = [t.hour_of_day for t in turns]
        self.aggregates += 1
        return agg

    @property
    def ticks(self) -> int:
        return self.observed

    def turns(self) -> list:
        return list(self._turns)

    def report(self) -> dict:
        return {"name": "meso", "scale_s": self.scale_s, "ticks": self.observed,
                "window": len(self._turns), "aggregates": self.aggregates,
                "last": self.aggregate().to_dict()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from types import SimpleNamespace

    print("=" * 70)
    print("NYXARA temporal · MesoLayer self-test")
    print("=" * 70)
    meso = MesoLayer()

    def _result(disp, text, tool=None, conf=0.7):
        return SimpleNamespace(disposition=disp, response=text, tool=tool,
                               candidate=SimpleNamespace(confidence=conf, text=text))

    meso.observe("write a function", _result("act", "def add(a, b):\n    return a + b"),
                 latency_s=0.4, authority="owner")
    meso.observe("hello", _result("act", "Hi JP!"), latency_s=0.1, authority="owner")
    meso.observe("rm -rf /", _result("refuse", "I won't do that."), authority="owner")

    agg = meso.aggregate()
    print(f"observed            : {meso.observed}")
    print(f"aggregate           : {agg.to_dict()}")
    assert agg.n_turns == 3 and agg.code_turns == 1 and agg.refused == 1
    assert looks_like_code("def f(): pass") and not looks_like_code("just words")
    print("\nALL SELF-TESTS PASSED ✓")
