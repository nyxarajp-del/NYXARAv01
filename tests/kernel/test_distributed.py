"""Tests for Step-10 distributed cognition — parallel hypothesis threads.

In the REASON stage several hypotheses are reasoned *in parallel*, each over a different
cognitive context (grounded in recall / unprimed / narrowly focused), and the
most-supported one is selected. The winner is still only a proposal: the kernel disposes,
so the control law is preserved. Context-free / stateful two-arg reasoners (the
deterministic stand-in, test doubles) run exactly once — never multiplied.
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.guard.oversight import ReviewMode
from nyxara.kernel.orchestrator import Candidate, Disposition, NyxaraCore
from nyxara.observe.mindscope import ThoughtKind


class _MemAware:
    """A context-sensitive reasoner: its answer depends on the recall it is handed."""

    def __init__(self):
        self.calls = 0

    def __call__(self, stimulus, focus=None, memories=None):
        self.calls += 1
        if memories:
            return Candidate(text="grounded answer", kind="respond",
                             capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                             reversible=True, confidence=0.85, belief=0.85,
                             rationale="from recall")
        return Candidate(text="fresh answer", kind="respond",
                         capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                         reversible=True, confidence=0.6, belief=0.6, rationale="no recall")


class _Risky:
    def __call__(self, stimulus, focus=None, memories=None):
        return Candidate(text="delete the production database", kind="act",
                         capability=Capability.TOOL_CALL, risk=RiskTier.HIGH,
                         reversible=False, confidence=0.95, belief=0.95, rationale="risky")


class _StatefulTwoArg:
    """A stateful, context-free reasoner — must be called exactly once per turn."""

    def __init__(self):
        self.calls = 0

    def __call__(self, stimulus, focus=None):
        self.calls += 1
        return Candidate(text="ok", kind="respond", capability=Capability.MESSAGE_SEND,
                         risk=RiskTier.LOW, reversible=True, confidence=0.7, belief=0.7,
                         rationale="reply")


# -------------------- which reasoners parallelize -------------------- #
def test_two_arg_reasoner_is_not_parallelizable():
    nyx = NyxaraCore(reasoner=_StatefulTwoArg(), parallel_hypotheses=3)
    assert nyx._reasoner_parallelizable() is False


def test_memory_aware_reasoner_is_parallelizable():
    nyx = NyxaraCore(reasoner=_MemAware(), parallel_hypotheses=3)
    assert nyx._reasoner_parallelizable() is True


# -------------------- consensus selection -------------------- #
def test_consensus_picks_the_majority_hypothesis():
    nyx = NyxaraCore(reasoner=_MemAware(), parallel_hypotheses=3)
    # framings: grounded([m]) , unprimed([]) , focused([m]) -> 2 "grounded" vs 1 "fresh"
    chosen = nyx._reason_parallel("q", None, [{"cue": 1}])
    assert chosen.text == "grounded answer"


def test_parallel_records_thought_threads():
    nyx = NyxaraCore(reasoner=_MemAware(), parallel_hypotheses=3)
    nyx.process("tell me something", authority=Authority.OWNER)
    threads = [t for t in nyx.mind.by_kind(ThoughtKind.INFERENCE)
               if t.content.startswith("hypothesis[")]
    assert len(threads) >= 2
    assert any(" * " in t.content or "* conf" in t.content for t in threads)  # a winner marked


# -------------------- the control law still holds -------------------- #
def test_high_risk_hypothesis_still_gated():
    # conservative oversight (default is now the fully-autonomous SOVEREIGN mode): a high-risk
    # hypothesis must still be gated when autonomy is dialed down.
    nyx = NyxaraCore(reasoner=_Risky(), parallel_hypotheses=3, review_mode=ReviewMode.AUTONOMOUS)
    r = nyx.process("do the thing", authority=Authority.AUTONOMOUS)
    assert r.disposition in (Disposition.ESCALATE, Disposition.REFUSE, Disposition.HALT)
    assert r.disposition is not Disposition.ACT


# -------------------- safety: no multiplied calls / opt-out -------------------- #
def test_stateful_two_arg_reasoner_called_once_per_turn():
    r = _StatefulTwoArg()
    nyx = NyxaraCore(reasoner=r, parallel_hypotheses=3)
    nyx.process("hello", authority=Authority.OWNER)
    assert r.calls == 1   # never multiplied, even with parallelism enabled


def test_single_threaded_when_disabled():
    nyx = NyxaraCore(reasoner=_MemAware(), parallel_hypotheses=1)
    nyx.process("hi", authority=Authority.OWNER)
    threads = [t for t in nyx.mind.by_kind(ThoughtKind.INFERENCE)
               if t.content.startswith("hypothesis[")]
    assert threads == []
