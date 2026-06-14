"""NYXARA · mind/dual_process.py — System-1 / System-2 + arbitrator (✦).

Kahneman's two minds, made mechanical:

* **System 1** — fast, automatic, intuitive. A cheap heuristic / pattern-match that
  returns an answer *and a confidence* in microseconds. Often right, sometimes
  fooled.
* **System 2** — slow, deliberate, analytical. It recruits the
  :class:`~nyxara.mind.faculties.FacultySelector` (preferring **verifiable** engines)
  to reason a problem through. Expensive, but trustworthy.

The interesting part is the **Arbitrator** — the metacognitive controller that
decides, per task, whether to *trust the gut* or *think it through*. It escalates to
System 2 when the stakes are high, the intuition is unconfident, or the problem is
novel/complex — and it falls back to System 1 under time pressure or low energy
(you act on instinct in an emergency). This is *cognitive reflection*: the capacity
to override a fast wrong answer with slow deliberation.

The Arbitrator also chooses **symbolic vs probabilistic**: high-stakes work demands a
verifiable faculty; casual generation can use the LLM.

Output is always a :class:`~nyxara.mind.proposal.Proposal` — the kernel still disposes.

Depends on :mod:`mind.faculties`, :mod:`mind.proposal`, :mod:`memory.provenance`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

from nyxara.memory.provenance import Provenance, SourceType
from nyxara.mind.faculties import FacultySelector, NoFacultyError, Task
from nyxara.mind.proposal import Proposal, ProposalKind

__all__ = [
    "ProcessType",
    "FastResponse",
    "System1",
    "System2",
    "ArbitratationDecision",
    "Arbitrator",
    "Outcome",
    "DualProcess",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Process type
# --------------------------------------------------------------------------- #
class ProcessType(str, Enum):
    SYSTEM_1 = "system_1"   # fast, intuitive
    SYSTEM_2 = "system_2"   # slow, deliberate


# --------------------------------------------------------------------------- #
# System 1 — fast intuition
# --------------------------------------------------------------------------- #
@dataclass
class FastResponse:
    answer: Any
    confidence: float
    latency_s: float = 0.0
    source: str = "system_1"


class System1:
    """Fast, automatic responder. Pluggable intuition; defaults to low-confidence guess."""

    def __init__(self, intuition: Optional[Callable[[Task], Tuple[Any, float]]] = None) -> None:
        self._intuition = intuition

    def respond(self, task: Task) -> FastResponse:
        start = time.monotonic()
        if self._intuition is not None:
            answer, confidence = self._intuition(task)
        else:
            # no learned intuition -> a weak, unconfident snap judgement
            answer, confidence = (task.payload if task.payload is not None
                                  else task.description, 0.3)
        return FastResponse(answer=answer, confidence=_clamp(confidence),
                            latency_s=time.monotonic() - start)


# --------------------------------------------------------------------------- #
# System 2 — slow deliberation (via faculties)
# --------------------------------------------------------------------------- #
class System2:
    """Deliberate responder. Recruits faculties (verifiable preferred when asked)."""

    def __init__(self, selector: Optional[FacultySelector] = None,
                 deliberate: Optional[Callable[[Task], Proposal]] = None) -> None:
        if selector is None and deliberate is None:
            raise ValueError("System2 needs a FacultySelector or a deliberate callable")
        self._selector = selector
        self._deliberate = deliberate

    def respond(self, task: Task, *, prefer_verifiable: bool = False) -> Proposal:
        if self._deliberate is not None:
            return self._deliberate(task)
        assert self._selector is not None
        if prefer_verifiable and not task.requires_verifiable:
            vtask = replace(task, requires_verifiable=True)
            try:
                return self._selector.dispatch(vtask)
            except NoFacultyError:
                pass  # no verifiable engine; fall through to best-available
        return self._selector.dispatch(task)


# --------------------------------------------------------------------------- #
# Arbitrator
# --------------------------------------------------------------------------- #
@dataclass
class ArbitratationDecision:
    process: ProcessType
    reason: str
    prefer_verifiable: bool
    deliberation_pressure: float

    def to_dict(self) -> Dict[str, Any]:
        return {"process": self.process.value, "reason": self.reason,
                "prefer_verifiable": self.prefer_verifiable,
                "pressure": round(self.deliberation_pressure, 3)}


@dataclass
class Arbitrator:
    """Decides System-1 vs System-2 (and symbolic vs probabilistic) per task."""

    deliberate_threshold: float = 0.5
    stakes_force: float = 0.7          # stakes ≥ this always deliberate (unless emergency)
    min_s1_confidence: float = 0.6     # below this, escalate to System 2
    confident_shortcut: float = 0.9    # very confident intuition + low stakes -> System 1
    emergency_time: float = 0.15       # time_budget below this -> act on instinct
    low_energy: float = 0.25           # energy below this -> conserve, prefer System 1

    def _stakes(self, task: Task, stakes: Optional[float]) -> float:
        if stakes is not None:
            return _clamp(stakes)
        s = float(task.features.get("stakes", 0.3))
        if task.requires_verifiable:
            s = max(s, 0.7)
        return _clamp(s)

    def decide(self, task: Task, s1: FastResponse, *, stakes: Optional[float] = None,
               time_budget: float = 1.0, energy: float = 1.0) -> ArbitratationDecision:
        st = self._stakes(task, stakes)
        difficulty = _clamp(float(task.features.get("difficulty", 0.5)))
        novelty = _clamp(float(task.features.get("novelty", 0.5)))
        pressure = _clamp(0.3 * difficulty + 0.25 * novelty + 0.5 * st
                          + 0.45 * (1.0 - s1.confidence))
        verifiable = st >= 0.5 or task.requires_verifiable

        # 1) emergency: extreme time pressure -> act on instinct (System 1)
        if time_budget < self.emergency_time:
            return ArbitratationDecision(ProcessType.SYSTEM_1,
                                         "emergency: act on instinct", verifiable, pressure)
        # 2) high stakes -> always deliberate
        if st >= self.stakes_force:
            return ArbitratationDecision(ProcessType.SYSTEM_2,
                                         f"high stakes ({st:.2f}) -> deliberate",
                                         True, pressure)
        # 3) low energy and modest stakes -> conserve, trust intuition
        if energy < self.low_energy and st < 0.5:
            return ArbitratationDecision(ProcessType.SYSTEM_1,
                                         "low energy -> conserve, trust intuition",
                                         verifiable, pressure)
        # 4) very confident intuition on a low-stakes task -> System 1
        if s1.confidence >= self.confident_shortcut and st < 0.5:
            return ArbitratationDecision(ProcessType.SYSTEM_1,
                                         "confident intuition, low stakes", verifiable, pressure)
        # 5) unconfident intuition -> reflect (System 2)
        if s1.confidence < self.min_s1_confidence:
            return ArbitratationDecision(ProcessType.SYSTEM_2,
                                         f"low intuition confidence ({s1.confidence:.2f}) "
                                         f"-> reflect", verifiable, pressure)
        # 6) otherwise: pressure decides
        if pressure >= self.deliberate_threshold:
            return ArbitratationDecision(ProcessType.SYSTEM_2,
                                         f"deliberation pressure {pressure:.2f} "
                                         f">= {self.deliberate_threshold}", verifiable, pressure)
        return ArbitratationDecision(ProcessType.SYSTEM_1,
                                     "intuition sufficient", verifiable, pressure)


# --------------------------------------------------------------------------- #
# Dual process facade
# --------------------------------------------------------------------------- #
@dataclass
class Outcome:
    process: ProcessType
    proposal: Proposal
    fast: FastResponse
    decision: ArbitratationDecision
    reflected: bool                  # did deliberation override the fast answer?

    def to_dict(self) -> Dict[str, Any]:
        return {"process": self.process.value, "reflected": self.reflected,
                "decision": self.decision.to_dict(),
                "fast_confidence": round(self.fast.confidence, 3),
                "proposal": self.proposal.to_dict()}


class DualProcess:
    """Intuition first; deliberation when it matters. Output is always a Proposal."""

    def __init__(self, system1: System1, system2: System2,
                 arbitrator: Optional[Arbitrator] = None) -> None:
        self.system1 = system1
        self.system2 = system2
        self.arbitrator = arbitrator or Arbitrator()

    def think(self, task: Task, *, stakes: Optional[float] = None,
              time_budget: float = 1.0, energy: float = 1.0) -> Outcome:
        fast = self.system1.respond(task)
        decision = self.arbitrator.decide(task, fast, stakes=stakes,
                                          time_budget=time_budget, energy=energy)

        if decision.process is ProcessType.SYSTEM_1:
            proposal = Proposal(
                kind=ProposalKind.ANSWER, content=fast.answer, source_faculty="system_1",
                confidence=fast.confidence, rationale=f"intuitive: {decision.reason}",
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=fast.confidence,
                                      detail="system_1", method="intuited"),
            )
            return Outcome(ProcessType.SYSTEM_1, proposal, fast, decision, reflected=False)

        # System 2: deliberate
        proposal = self.system2.respond(task, prefer_verifiable=decision.prefer_verifiable)
        # reflected == we had a fast answer but chose to think instead
        reflected = fast.confidence > 0.0
        proposal.metadata.setdefault("deliberated", True)
        return Outcome(ProcessType.SYSTEM_2, proposal, fast, decision, reflected=reflected)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import ast
    import operator as op

    from nyxara.kernel.config import NyxaraSettings, Profile
    from nyxara.mind.faculties import (CallableFaculty, FacultyRegistry, FacultySelector,
                                       LLMFaculty, TaskType)
    from nyxara.mind.llm import LLM

    print("=" * 70)
    print("NYXARA dual-process self-test")
    print("=" * 70)

    # build a System-2 backed by faculties (exact math + probabilistic LLM)
    _OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}

    def _ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.BinOp):
            return _OPS[type(n.op)](_ev(n.left), _ev(n.right))
        raise ValueError

    math_fac = CallableFaculty("math", [TaskType.ARITHMETIC],
                               lambda t: (_ev(ast.parse(t.payload, mode="eval").body), 1.0),
                               verifiable=True, reliability=1.0, cost=0.2)
    llm = LLM(settings=NyxaraSettings.for_profile(Profile.TEST))
    reg = FacultyRegistry(); reg.register(math_fac); reg.register(LLMFaculty(llm))
    selector = FacultySelector(reg)

    # System 1 with a simple (and deliberately fallible) arithmetic "gut feeling"
    def gut(task: Task):
        if task.type is TaskType.ARITHMETIC:
            return ("(gut) ~roughly 10", 0.4)  # unconfident on math
        return ("(gut) sure, why not", 0.95)   # confident on chit-chat

    dp = DualProcess(System1(gut), System2(selector))

    # 1) low-stakes, confident intuition -> System 1
    chat = Task(TaskType.GENERATION, "say hello", features={"stakes": 0.2})
    o = dp.think(chat)
    print(f"\nchit-chat           : {o.process.value} ({o.decision.reason})")
    assert o.process is ProcessType.SYSTEM_1

    # 2) arithmetic with unconfident gut -> reflect to System 2 -> exact math faculty
    sum_task = Task(TaskType.ARITHMETIC, "compute 2+3*4", payload="2+3*4",
                    features={"stakes": 0.3})
    o = dp.think(sum_task)
    print(f"arithmetic          : {o.process.value} -> {o.proposal.content} "
          f"(reflected={o.reflected})")
    assert o.process is ProcessType.SYSTEM_2
    assert o.proposal.content == 14 and o.reflected

    # 3) high stakes forces deliberation even with confident intuition
    risky = Task(TaskType.CLASSIFICATION, "is this transaction fraud?",
                 features={"stakes": 0.9})
    o = dp.think(risky)
    print(f"high stakes         : {o.process.value} ({o.decision.reason}) "
          f"verifiable={o.decision.prefer_verifiable}")
    assert o.process is ProcessType.SYSTEM_2

    # 4) emergency time pressure -> act on instinct (System 1)
    o = dp.think(risky, time_budget=0.05)
    print(f"emergency           : {o.process.value} ({o.decision.reason})")
    assert o.process is ProcessType.SYSTEM_1

    # 5) low energy + low stakes -> conserve, trust intuition
    o = dp.think(chat, energy=0.1)
    print(f"low energy          : {o.process.value} ({o.decision.reason})")
    assert o.process is ProcessType.SYSTEM_1

    # output is always a Proposal
    assert isinstance(o.proposal, Proposal)
    print(f"\noutput is a Proposal : kind={o.proposal.kind.value}")
    print("\nALL SELF-TESTS PASSED ✓")
