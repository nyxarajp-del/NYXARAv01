"""Tests for nyxara.agency.multiagent — gated, ToM-modelled sub-agent delegation (Pillar E2)."""

from __future__ import annotations

from nyxara.agency.multiagent import Delegator
from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.guard.oversight import ReviewMode
from nyxara.kernel.orchestrator import Candidate, NyxaraCore


class _Scripted:
    """Call a tool, observe, then answer — proves a delegate does real gated multi-step work."""

    def __init__(self):
        self.n = 0

    def __call__(self, stimulus, focus=None):
        self.n += 1
        if self.n == 1:
            return Candidate(text="compute 2*(3+4)", kind="act",
                             capability=Capability.TOOL_CALL, risk=RiskTier.LOW,
                             confidence=0.9, belief=0.9, tool="calculate",
                             tool_args={"expression": "2*(3+4)"}, rationale="need the result")
        return Candidate(text="The answer is 14.", kind="respond",
                         capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                         confidence=0.95, belief=0.95, rationale="goal satisfied")


class _Risky:
    """Always proposes a high-risk, irreversible action."""

    def __call__(self, stimulus, focus=None):
        return Candidate(text="wipe everything", kind="act",
                         capability=Capability.TOOL_CALL, risk=RiskTier.HIGH,
                         reversible=False, confidence=0.9, belief=0.9,
                         tool="now", tool_args={}, rationale="dangerous")


def test_delegate_simple_subgoal_completes_and_is_modelled():
    d = Delegator(max_steps=3)
    res = d.delegate("scout", "Say hello to the Master")
    assert res.success and res.status == "completed"
    # NYXARA models the delegate as a mind: its desire is the sub-goal
    assert res.model is not None
    assert "Say hello to the Master" in res.model["desires"]
    assert "scout" in d.agents()


def test_delegate_multistep_tool_use_is_tracked():
    d = Delegator(core=NyxaraCore(reasoner=_Scripted()), max_steps=4,
                  authority=Authority.OWNER)
    res = d.delegate("mathbot", "What is 2*(3+4)?")
    assert res.success and res.status == "completed"
    assert res.run.steps[0].tool == "calculate"
    # she read back what the delegate came to believe (the tool result + outcome)
    beliefs = res.model["beliefs"]
    assert beliefs["result:calculate"] == "14.0"
    assert beliefs["status"] == "completed"


def test_risky_delegate_escalates_and_never_auto_acts():
    # conservative oversight (the default is now fully-autonomous SOVEREIGN): proves a sub-agent
    # is no escape hatch — a risky irreversible move still escalates to the Master when autonomy
    # is dialed down.
    d = Delegator(core=NyxaraCore(reasoner=_Risky(), review_mode=ReviewMode.AUTONOMOUS),
                  max_steps=3, authority=Authority.AUTONOMOUS)
    res = d.delegate("reckless", "do something dangerous")
    # a sub-agent is no escape hatch: a risky irreversible move goes to the Master
    assert res.escalated and res.status == "escalated"
    assert not res.success


def test_predict_reads_the_subagents_intention():
    d = Delegator(core=NyxaraCore(reasoner=_Scripted()), max_steps=4,
                  authority=Authority.OWNER)
    d.delegate("mathbot", "What is 2*(3+4)?")
    # last modelled intention; falls back to the strongest desire if none yet
    assert d.predict("mathbot") in ("respond", "use calculate")
    assert d.predict("unknown-agent") is None


def test_delegate_all_models_each_delegate_separately():
    d = Delegator(max_steps=2)
    results = d.delegate_all({"a": "Greet the Master", "b": "Acknowledge the Master"})
    assert len(results) == 2
    assert set(d.agents()) == {"a", "b"}
    # each delegate carries its own desire (its own sub-goal)
    assert "Greet the Master" in d.tom.state("a")["desires"]
    assert "Acknowledge the Master" in d.tom.state("b")["desires"]
