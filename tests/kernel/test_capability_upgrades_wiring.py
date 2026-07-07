"""Capability upgrades are wired into the live orchestrator.

Covers four gap-closers proven end-to-end on a real `NyxaraCore`:

* B — the decision-theoretic `InitiativeGovernor` gates *autonomous actions* inside the control-law
  pipeline (`_gate`): a confident reversible act clears, an irreversible high-stakes or
  low-confidence act is escalated to the Master, and plain responses are never gated by it.
* C — `grand_plan` runs a pre-mortem, attaches the ranked failure modes + mitigations to the plan,
  and records the top risk to the hash-chained journal.
* D — memory autosave is OFF under pytest by default, and when enabled snapshots after the
  threshold; state round-trips through save/load.
* E — grounded action outcomes teach the prediction engine's learned base rate.
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.kernel.orchestrator import Candidate, Disposition, NyxaraCore


def _act(text: str, *, risk: RiskTier, reversible: bool, confidence: float) -> Candidate:
    return Candidate(text=text, kind="act", tool="noop", capability=Capability.TOOL_CALL,
                     risk=risk, reversible=reversible, confidence=confidence)


# ----------------------------- B: initiative gate ----------------------------- #
def test_confident_reversible_act_clears_initiative():
    nyx = NyxaraCore()
    gates: dict = {}
    disp, _ = nyx._gate(_act("write a scratch note", risk=RiskTier.LOW, reversible=True,
                             confidence=0.85), Authority.OWNER, gates)
    assert disp is Disposition.ACT
    assert gates.get("initiative") == "act"


def test_irreversible_high_stakes_act_is_escalated():
    nyx = NyxaraCore()
    gates: dict = {}
    disp, reason = nyx._gate(_act("delete the whole dataset", risk=RiskTier.CRITICAL,
                                  reversible=False, confidence=0.95), Authority.OWNER, gates)
    assert disp is Disposition.ESCALATE
    assert gates.get("initiative") == "ask"
    assert "initiative" in reason


def test_low_confidence_act_is_escalated():
    nyx = NyxaraCore()
    gates: dict = {}
    disp, _ = nyx._gate(_act("maybe run this", risk=RiskTier.LOW, reversible=True,
                             confidence=0.3), Authority.OWNER, gates)
    assert disp is Disposition.ESCALATE
    assert gates.get("initiative") == "ask"


def test_plain_response_is_not_gated_by_initiative():
    # conversational replies must never be escalated by the initiative governor, even at low
    # confidence — honesty (not initiative) handles them.
    nyx = NyxaraCore()
    gates: dict = {}
    nyx._gate(Candidate(text="I think the answer is 42", kind="respond", risk=RiskTier.LOW,
                        reversible=True, confidence=0.3), Authority.OWNER, gates)
    assert "initiative" not in gates


# ----------------------------- C: pre-mortem on plans ----------------------------- #
def test_pre_mortem_returns_ranked_failure_modes():
    nyx = NyxaraCore()
    analysis = nyx.pre_mortem("deploy the hardened firewall")
    modes = analysis["premortem"]
    assert modes and modes[0]["risk"] >= modes[-1]["risk"]      # ranked by risk
    assert all("mitigation" in m for m in modes)
    assert "recommendation" in analysis


def test_grand_plan_attaches_premortem_and_journals_it():
    nyx = NyxaraCore()
    before = len(nyx.journal.entries()) if hasattr(nyx.journal, "entries") else None
    plan = nyx.grand_plan("build a secure home lab", target_steps=40, max_depth=3)
    assert getattr(plan, "premortem", None), "plan must carry its pre-mortem risks"
    assert getattr(plan, "risk_analysis", None)
    if before is not None:
        assert len(nyx.journal.entries()) > before             # a note was recorded


# ----------------------------- D: memory autosave ----------------------------- #
def test_autosave_is_off_under_pytest_by_default():
    nyx = NyxaraCore()
    calls = []
    nyx.save_state = lambda *a, **k: calls.append(1)            # type: ignore[assignment]
    for i in range(30):
        nyx._maybe_autosave()
    assert calls == []                                          # guarded off inside the test suite


def test_autosave_fires_after_threshold_when_enabled():
    nyx = NyxaraCore()
    nyx._autosave_enabled = True
    nyx._autosave_every_turns = 3
    nyx._autosave_min_interval = 1e9                            # isolate the turn-count trigger
    calls = []
    nyx.save_state = lambda *a, **k: (calls.append(1), "ok")[1]  # type: ignore[assignment]
    for _ in range(3):
        nyx._maybe_autosave()
    assert len(calls) == 1                                      # snapshotted once at the threshold


def test_state_round_trips_through_save_and_load(tmp_path):
    target = str(tmp_path / "longterm.json")
    a = NyxaraCore()
    a.memory.remember("the Master prefers concise answers", importance=0.9)
    assert a.save_state(target) is not None
    b = NyxaraCore()
    assert b.load_state(target) > 0                             # records restored into a fresh mind


# ----------------------------- E: outcomes teach prediction prior ----------------------------- #
def _grounded(conf: float = 0.9) -> Candidate:
    return Candidate(text="compute 12 + 30 = 42", kind="act", tool="calculate",
                     tool_args={"expression": "12+30"}, confidence=conf,
                     capability=Capability.TOOL_CALL, risk=RiskTier.LOW, reversible=True)


def test_grounded_outcomes_teach_prediction_prior():
    nyx = NyxaraCore()
    assert nyx.prediction_engine is not None
    assert not nyx.prediction_engine.prior._arms                # nothing learned yet
    for _ in range(20):
        nyx._record_calibration(_grounded(), correct=True)
    arms = nyx.prediction_engine.prior._arms
    assert arms and any(v.get("n", 0) > 0 for v in arms.values())
    base, why = nyx.prediction_engine.prior.base_rate("compute 5 + 5 = 10")
    assert why is not None and base > 0.5                       # learned, not the flat default
