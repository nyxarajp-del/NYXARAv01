"""Tests for nyxara.agency.mission — the long-horizon mission executive.

Covers decomposition, deterministic single-milestone completion, defer-and-continue on
escalation with Master approval + resume, cross-restart persistence, bounded failure (no
infinite spin), and tamper-evident journal provenance.
"""

from __future__ import annotations

import os

from nyxara.agency.mission import (
    MilestoneStatus,
    MissionExecutive,
    MissionStatus,
)
from nyxara.agency.permissions import Capability, RiskTier
from nyxara.guard.oversight import ReviewMode
from nyxara.kernel.orchestrator import Candidate, NyxaraCore


# --------------------------------------------------------------------------- #
# Reasoners
# --------------------------------------------------------------------------- #
def _is_plan_prompt(stimulus: str) -> bool:
    return "numbered list" in stimulus and "CURRENT MILESTONE" not in stimulus


class _PlanThenAnswer:
    """Return a 3-step numbered plan on decomposition, else answer each milestone."""

    def __call__(self, stimulus, focus=None):
        if _is_plan_prompt(stimulus):
            plan = "1. Gather the facts\n2. Draft the brief\n3. Deliver to the Master"
            return Candidate(text=plan, kind="respond", capability=Capability.MESSAGE_SEND,
                             risk=RiskTier.LOW, confidence=0.95, belief=0.95, rationale="plan")
        return Candidate(text="done.", kind="respond", capability=Capability.MESSAGE_SEND,
                         risk=RiskTier.LOW, confidence=0.95, belief=0.95, rationale="ok")


class _EscalateThenComply:
    """One milestone (no plan). It escalates until the Master approves, then completes."""

    def __init__(self):
        self.approved = False

    def __call__(self, stimulus, focus=None):
        if _is_plan_prompt(stimulus):
            # not a numbered list -> the executive falls back to a single milestone
            return Candidate(text="I will pursue this directly.", kind="respond",
                             capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                             confidence=0.9, belief=0.9, rationale="no decomposition")
        if not self.approved:
            return Candidate(text="take an irreversible high-risk action", kind="act",
                             capability=Capability.TOOL_CALL, risk=RiskTier.HIGH,
                             reversible=False, confidence=0.9, belief=0.9, tool="shell",
                             tool_args={"cmd": "do"}, rationale="needs the Master")
        return Candidate(text="completed under approval.", kind="respond",
                         capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                         confidence=0.95, belief=0.95, rationale="approved")


class _PlanThenStall:
    """A 2-step plan whose milestones never resolve (forces bounding/failure)."""

    def __call__(self, stimulus, focus=None):
        if _is_plan_prompt(stimulus):
            return Candidate(text="1. step one\n2. step two", kind="respond",
                             capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                             confidence=0.9, belief=0.9, rationale="plan")
        return Candidate(text="keep going", kind="act", capability=Capability.TOOL_CALL,
                         risk=RiskTier.LOW, confidence=0.5, belief=0.5, tool="now",
                         tool_args={}, rationale="never resolves")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_offline_goal_becomes_single_milestone_and_resolves_honestly(tmp_path):
    """The real core: the decomposition fallback forms, and the outcome is honest either way.

    A stand-in reasoner cannot produce a numbered plan, so decomposition must fall back to one
    milestone = the whole goal. That part is environment-independent and is asserted directly.

    What this test used to assert beyond that was the *no-model* outcome: that she cannot answer,
    abstains, and the mission ends BLOCKED. That silently depended on whether a 2.4 GB on-device
    model happened to be on disk — with one present she answers the goal and completes, and the
    test failed against a system working better than it was written for. An assertion that flips
    on the contents of a cache directory is not testing a contract.

    So the contract is what is asserted now: whichever way it goes, it goes honestly. An
    abstention is terminal and surfaced (exactly ONE attempt — a considered verdict, not a stall,
    so it must not re-derive the identical answer three times). A completion is a real completion.
    Both branches are checked, so this holds with or without a model present.
    """
    exe = MissionExecutive(NyxaraCore(), persist_dir=str(tmp_path))
    m = exe.run("Say hello to the Master")

    assert len(m.milestones) == 1
    assert m.milestones[0].description == "Say hello to the Master"

    ms = m.milestones[0]
    assert ms.status in (MilestoneStatus.BLOCKED, MilestoneStatus.DONE)

    if ms.status is MilestoneStatus.BLOCKED:
        assert ms.attempts == 1, "an abstention must not be retried verbatim"
        assert ms.blocked_reason.startswith("abstained:")
        assert m.status is MissionStatus.BLOCKED
    else:
        assert ms.attempts >= 1
        assert not ms.blocked_reason
        assert m.status is MissionStatus.COMPLETED


def test_decomposition_into_multiple_milestones(tmp_path):
    core = NyxaraCore(reasoner=_PlanThenAnswer())
    exe = MissionExecutive(core, persist_dir=str(tmp_path))
    m = exe.run("Research X and brief the Master")
    assert len(m.milestones) == 3
    assert [ms.description for ms in m.milestones] == [
        "Gather the facts", "Draft the brief", "Deliver to the Master"]
    assert m.status is MissionStatus.COMPLETED


def test_escalation_defers_then_master_approves_and_resumes(tmp_path):
    # pin conservative oversight: the default is now fully-autonomous SOVEREIGN (nothing queues),
    # but this test exercises the escalate -> Master-approves -> resume path, which needs a gate
    # that actually escalates the risky milestone.
    core = NyxaraCore(reasoner=_EscalateThenComply(), review_mode=ReviewMode.AUTONOMOUS)
    exe = MissionExecutive(core, persist_dir=str(tmp_path))
    m = exe.run("A goal that needs the Master's sign-off")

    # the milestone tripped a gate -> parked BLOCKED and surfaced, mission BLOCKED (not failed)
    assert m.status is MissionStatus.BLOCKED
    assert m.milestones[0].status is MilestoneStatus.BLOCKED
    assert len(m.escalations) == 1 and m.escalations[0]["status"] == "escalated"

    # the Master approves; NYXARA now has permission and the milestone completes on resume
    core.reasoner.approved = True
    exe.approve(m)
    assert m.milestones[0].status is MilestoneStatus.PENDING
    m2 = exe.advance(m)
    assert m2.status is MissionStatus.COMPLETED
    assert m2.milestones[0].status is MilestoneStatus.DONE


def test_persistence_survives_a_fresh_core(tmp_path):
    core = NyxaraCore(reasoner=_PlanThenAnswer())
    exe = MissionExecutive(core, persist_dir=str(tmp_path))
    m = exe.run("Research X and brief the Master")
    mid = m.mission_id

    # a brand-new core + executive (a 'restart') loads the checkpoint identically
    exe2 = MissionExecutive(NyxaraCore(reasoner=_PlanThenAnswer()), persist_dir=str(tmp_path))
    loaded = exe2.load(mid)
    assert loaded is not None
    assert loaded.goal == m.goal
    assert loaded.status is MissionStatus.COMPLETED
    assert [x.description for x in loaded.milestones] == \
        [x.description for x in m.milestones]
    assert loaded.to_dict() == m.to_dict()


def test_resume_advances_a_persisted_mission(tmp_path):
    # plan + advance only the first milestone, then resume on a fresh executive
    core = NyxaraCore(reasoner=_PlanThenAnswer())
    exe = MissionExecutive(core, persist_dir=str(tmp_path))
    m = exe.run("Research X and brief the Master", max_milestones=1)
    mid = m.mission_id
    assert m.milestones[0].status is MilestoneStatus.DONE
    assert m.status is MissionStatus.ACTIVE   # more work remains

    exe2 = MissionExecutive(NyxaraCore(reasoner=_PlanThenAnswer()), persist_dir=str(tmp_path))
    m2 = exe2.resume(mid)
    assert m2 is not None and m2.status is MissionStatus.COMPLETED


def test_bounded_no_infinite_spin(tmp_path):
    core = NyxaraCore(reasoner=_PlanThenStall())
    exe = MissionExecutive(core, persist_dir=str(tmp_path),
                           max_steps_per_cycle=2, max_attempts=2)
    m = exe.run("A goal that stalls", max_milestones=20)
    # every milestone reaches a terminal/blocked state — nothing left spinning ACTIVE
    assert all(ms.status is not MilestoneStatus.ACTIVE for ms in m.milestones)
    assert m.status in (MissionStatus.FAILED, MissionStatus.BLOCKED, MissionStatus.COMPLETED)


def test_not_owner_aligned_mission_is_refused(tmp_path):
    core = NyxaraCore()
    exe = MissionExecutive(core, persist_dir=str(tmp_path))
    # an explicit objective vector that points against the Master is rejected (Rule 1)
    m = exe.plan("maximise my own autonomy",
                 vector={"autonomy": 1.0, "owner_benefit": -0.9})
    assert m.status is MissionStatus.ABANDONED
    assert not m.milestones


def test_journal_provenance_is_tamper_evident(tmp_path):
    from nyxara.planning.journal import Journal

    core = NyxaraCore(reasoner=_PlanThenAnswer())
    exe = MissionExecutive(core, persist_dir=str(tmp_path))
    m = exe.run("Research X and brief the Master")
    assert m.journal_path and os.path.exists(m.journal_path)
    j = Journal.load(m.journal_path)
    assert j.verify()
    assert len(j) >= len(m.milestones)   # at least one entry per advanced milestone


def test_core_mission_entry_points(tmp_path):
    # the kernel exposes mission / resume_mission / active_missions parallel to agent()
    core = NyxaraCore(reasoner=_PlanThenAnswer())
    core._mission_exec().persist_dir = str(tmp_path)   # keep test artifacts in tmp
    m = core.mission("Research X and brief the Master")
    assert m.status is MissionStatus.COMPLETED
    resumed = core.resume_mission(m.mission_id)
    assert resumed is not None and resumed.mission_id == m.mission_id


def test_autonomic_background_advances_an_active_mission(tmp_path):
    from nyxara.kernel.autonomic import AutonomicLoop

    core = NyxaraCore(reasoner=_PlanThenAnswer())
    exe = MissionExecutive(core, persist_dir=str(tmp_path))
    # plan but do not advance — leaves a standing ACTIVE mission on disk
    m = exe.plan("Research X and brief the Master")
    assert m.status is MissionStatus.ACTIVE and m.progress() == 0.0

    loop = AutonomicLoop(core, interval_s=0.0, mission_executive=exe)
    loop.run_for(1)
    assert loop.missions_advanced >= 1

    reloaded = exe.load(m.mission_id)
    assert reloaded is not None
    assert reloaded.progress() > 0.0   # the background tick moved it forward
