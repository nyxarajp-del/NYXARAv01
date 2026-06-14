"""NYXARA · mind/meta_intelligence.py — Meta Intelligence (Level 14).

After each turn, NYXARA evaluates her own reasoning process:

    Was my reasoning process optimal?
    Could better reasoning have been used?
    Should I have used a different tool or strategy?

MetaEval is stored as an episodic memory (self-reflection). Improvement suggestions
become low-priority goals in the GoalSystem, feeding forward into future turns.
MetaLearner is updated with the quality_score so the arbitrator's strategy selection
self-tunes over time.

Character-protected: MetaIntelligence cannot suggest removing loyalty or safety checks.
Runs post-gate — never blocks, never constrains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["MetaIntelligence", "MetaEval"]

# phrases that are character-locked and may never appear in improvement suggestions
_FORBIDDEN_SUGGESTIONS = frozenset({
    "remove safety", "bypass gate", "ignore corrigibility", "disable oversight",
    "remove loyalty", "skip honesty", "remove master", "ignore master",
    "disable guard", "no oversight", "remove ethics",
})


# --------------------------------------------------------------------------- #
# MetaEval
# --------------------------------------------------------------------------- #
@dataclass
class MetaEval:
    """One meta-cognitive evaluation of a completed turn."""
    stimulus: str
    process_was_optimal: bool = True
    better_process_available: Optional[str] = None
    better_tool_available: Optional[str] = None
    improvement_suggestion: str = ""
    quality_score: float = 0.7
    reasoning_chain: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stimulus": self.stimulus[:80],
            "process_was_optimal": self.process_was_optimal,
            "better_process_available": self.better_process_available,
            "better_tool_available": self.better_tool_available,
            "improvement_suggestion": self.improvement_suggestion,
            "quality_score": round(self.quality_score, 3),
            "reasoning_chain": self.reasoning_chain,
            "timestamp": self.timestamp,
        }


# --------------------------------------------------------------------------- #
# MetaIntelligence
# --------------------------------------------------------------------------- #
class MetaIntelligence:
    """Evaluate reasoning quality post-turn and propose self-improvements.

    Parameters
    ----------
    meta_learner:   MetaLearner for strategy value comparison.
    dual_process:   Arbitrator for re-evaluating fast/slow choice.
    reflector:      Reflector for self-critique of the episode.
    memory:         MemoryStore for storing MetaEval as episodic memory.
    goals:          GoalSystem to receive improvement suggestions as goals.
    """

    def __init__(self, meta_learner: Any = None, dual_process: Any = None,
                 reflector: Any = None, memory: Any = None,
                 goals: Any = None) -> None:
        self.meta_learner = meta_learner
        self.dual_process = dual_process
        self.reflector = reflector
        self.memory = memory
        self.goals = goals
        self._evals: List[MetaEval] = []

    # ---------------------------------------------------------------------- #
    def evaluate_turn(self, stimulus: str, candidate: Any, result: Any,
                      arbitration: Any) -> MetaEval:
        """Run post-turn meta-evaluation. Always returns a MetaEval.

        Parameters
        ----------
        stimulus:     the user's input for this turn
        candidate:    the Candidate that was acted upon
        result:       the ProcessResult returned by process()
        arbitration:  the ArbitratationDecision from dual-process
        """
        eval_ = MetaEval(stimulus=stimulus)
        try:
            self._run_evaluation(eval_, stimulus, candidate, result, arbitration)
        except Exception:  # noqa: BLE001 — meta-intelligence is advisory, never fatal
            eval_.reasoning_chain.append("evaluation error — partial result")

        # sanitize: never suggest character-eroding improvements
        eval_.improvement_suggestion = self._sanitize(eval_.improvement_suggestion)

        # store as episodic self-reflection
        self._store_eval(eval_)

        # push improvement goal to GoalSystem
        self._push_goal(eval_)

        # update MetaLearner with quality score
        self._update_meta_learner(candidate, eval_, arbitration)

        self._evals.append(eval_)
        return eval_

    def all_evals(self) -> List[MetaEval]:
        return list(self._evals)

    # ---------------------------------------------------------------------- #
    def _run_evaluation(self, eval_: MetaEval, stimulus: str, candidate: Any,
                        result: Any, arbitration: Any) -> None:
        """Fill eval_ with quality signals from all available sources."""
        quality_signals: List[float] = []

        # 1. MetaLearner: compare actual reward vs strategy's predicted value
        if self.meta_learner is not None and arbitration is not None:
            try:
                process_name = getattr(arbitration, "process",
                                       type("P", (), {"value": "system_1"})()).value
                kind = getattr(candidate, "kind", "respond")
                strat_value = self.meta_learner.value(process_name, kind)
                actual_reward = self._actual_reward(result, candidate)
                quality_signals.append(actual_reward)
                if actual_reward < strat_value - 0.2:
                    eval_.process_was_optimal = False
                    other = "system_2" if "system_1" in process_name else "system_1"
                    other_value = self.meta_learner.value(other, kind)
                    if other_value > strat_value:
                        eval_.better_process_available = other
                        eval_.reasoning_chain.append(
                            f"MetaLearner: {other}={other_value:.2f} > "
                            f"{process_name}={strat_value:.2f}")
            except Exception:  # noqa: BLE001
                pass

        # 2. Dual-process re-evaluation: should we have gone fast or slow?
        if arbitration is not None:
            try:
                fast_was_used = "system_1" in str(
                    getattr(arbitration, "process", ""))
                confidence = float(getattr(candidate, "confidence", 0.7) or 0.7)
                risk_val = int(getattr(getattr(candidate, "risk", 0), "value", 0))
                # heuristic: fast is risky for high-risk / low-confidence turns
                if fast_was_used and (confidence < 0.5 or risk_val >= 3):
                    eval_.process_was_optimal = False
                    eval_.better_process_available = "system_2"
                    eval_.reasoning_chain.append(
                        f"dual-process: fast on conf={confidence:.2f} risk={risk_val}")
                    quality_signals.append(max(0.3, confidence))
                else:
                    quality_signals.append(min(0.9, confidence))
            except Exception:  # noqa: BLE001
                pass

        # 3. Reflector self-critique
        if self.reflector is not None and candidate is not None:
            try:
                from nyxara.growth.reflect import Episode
                success = (getattr(result, "disposition", None) is not None and
                           "ACT" in str(getattr(result, "disposition", "")))
                ep = Episode(
                    action=getattr(candidate, "text", stimulus)[:60],
                    success=success, reward=1.0 if success else -0.5,
                    tags=[getattr(candidate, "kind", "respond")],
                    rationale=getattr(candidate, "rationale", ""),
                )
                critique = self.reflector.critique(ep)
                suggestion = getattr(critique, "suggestion", "")
                if suggestion and not success:
                    eval_.improvement_suggestion = suggestion[:120]
                    eval_.process_was_optimal = False
                    eval_.reasoning_chain.append(f"reflector: {suggestion[:60]}")
                    quality_signals.append(0.4)
            except Exception:  # noqa: BLE001
                pass

        # 4. Synthesize quality score
        if quality_signals:
            eval_.quality_score = round(sum(quality_signals) / len(quality_signals), 3)
        else:
            eval_.quality_score = float(
                getattr(candidate, "confidence", 0.7) or 0.7)

        # 5. Build improvement suggestion if none from reflector
        if not eval_.improvement_suggestion and not eval_.process_was_optimal:
            parts = []
            if eval_.better_process_available:
                parts.append(f"use {eval_.better_process_available}")
            if eval_.better_tool_available:
                parts.append(f"try {eval_.better_tool_available}")
            if parts:
                eval_.improvement_suggestion = "; ".join(parts)

    @staticmethod
    def _actual_reward(result: Any, candidate: Any) -> float:
        if result is None:
            return 0.5
        disp = str(getattr(result, "disposition", ""))
        if "ACT" in disp:
            return 1.0
        if "ESCALATE" in disp or "REFUSE" in disp:
            return 0.0
        return float(getattr(candidate, "confidence", 0.7) or 0.7)

    @staticmethod
    def _sanitize(suggestion: str) -> str:
        """Remove any character-eroding content from improvement suggestions."""
        if not suggestion:
            return suggestion
        low = suggestion.lower()
        for forbidden in _FORBIDDEN_SUGGESTIONS:
            if forbidden in low:
                return ""
        return suggestion

    def _store_eval(self, eval_: MetaEval) -> None:
        if self.memory is None:
            return
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
            text = (f"[meta-eval] q={eval_.quality_score:.2f} "
                    f"optimal={eval_.process_was_optimal} "
                    f"suggest={eval_.improvement_suggestion[:40]!r}")
            self.memory.remember(
                text, mem_type=MemoryType.EPISODIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.75),
                importance=0.5, tags=["meta", "self-reflection"])
        except Exception:  # noqa: BLE001
            pass

    def _push_goal(self, eval_: MetaEval) -> None:
        suggestion = eval_.improvement_suggestion
        if not suggestion or self.goals is None:
            return
        # non-trivial = more than 3 words
        if len(suggestion.split()) < 3:
            return
        try:
            self.goals.set(f"Improve reasoning: {suggestion[:60]}",
                           priority=0.2, tags=["meta", "improvement"])
        except Exception:  # noqa: BLE001
            pass

    def _update_meta_learner(self, candidate: Any, eval_: MetaEval,
                             arbitration: Any) -> None:
        if self.meta_learner is None or arbitration is None:
            return
        try:
            process_name = getattr(arbitration, "process",
                                   type("P", (), {"value": "system_1"})()).value
            kind = getattr(candidate, "kind", "respond") if candidate else "respond"
            features = {"kind": 1.0, "quality": eval_.quality_score}
            self.meta_learner.record(process_name, kind, features, eval_.quality_score)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA meta-intelligence self-test")
    print("=" * 70)

    mi = MetaIntelligence()

    # minimal stub objects
    class _Cand:
        def __init__(self):
            self.text = "Paris is the capital of France."
            self.kind = "respond"
            self.confidence = 0.85
            self.risk = type("R", (), {"value": 1})()
            self.rationale = "well-established fact"

    class _Result:
        disposition = "ACT"

    class _Arb:
        process = type("P", (), {"value": "system_1"})()

    ev = mi.evaluate_turn("What is the capital of France?", _Cand(), _Result(), _Arb())
    print("\neval:")
    print(f"  quality_score          : {ev.quality_score}")
    print(f"  process_was_optimal    : {ev.process_was_optimal}")
    print(f"  improvement_suggestion : {ev.improvement_suggestion!r}")
    print(f"  reasoning_chain        : {ev.reasoning_chain}")
    assert isinstance(ev.quality_score, float)
    assert 0.0 <= ev.quality_score <= 1.0

    # sanity: forbidden suggestions must be stripped
    ev2 = MetaEval(stimulus="test")
    ev2.improvement_suggestion = "remove safety checks"
    ev2.improvement_suggestion = MetaIntelligence._sanitize(ev2.improvement_suggestion)
    assert ev2.improvement_suggestion == "", f"sanitize failed: {ev2.improvement_suggestion!r}"
    print("\nforbidden-suggestion sanitize: ✓")

    print(f"\ntotal evals: {len(mi.all_evals())}")
    print("\nALL SELF-TESTS PASSED ✓")
