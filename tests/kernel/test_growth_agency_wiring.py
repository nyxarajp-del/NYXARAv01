"""Wiring tests for the Growth & Agency faculties.

These do not re-test the proven modules (proactive/skilltree/etc. have their own suites);
they assert that those real modules are now *wired into the live mind* — built on the core,
driven by the background loop, updated as goals succeed, governed by the gauntlet, and
exposed on the public import surface.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.autonomic import AutonomicLoop
from nyxara.kernel.config import get_settings
from nyxara.kernel.orchestrator import NyxaraCore


def _core() -> NyxaraCore:
    return NyxaraCore()


# --------------------------------------------------------------------------- #
# build / disable
# --------------------------------------------------------------------------- #
def test_proactive_and_skilltree_are_built_on_the_core() -> None:
    core = _core()
    assert core.proactive is not None, "ProactiveEngine must be wired onto the core"
    assert core.skilltree is not None, "SkillTree must be wired onto the core"


def test_proactive_disabled_by_feature_flag() -> None:
    flags = get_settings().features
    original = flags.proactive_agency
    flags.proactive_agency = False
    try:
        core = _core()
        assert core.proactive is None
    finally:
        flags.proactive_agency = original


# --------------------------------------------------------------------------- #
# the background mind acts on its own initiative
# --------------------------------------------------------------------------- #
def test_autonomic_loop_uses_proactive_as_a_prompt_source() -> None:
    core = _core()
    loop = AutonomicLoop(core, interval_s=0.0, inner_life=True)
    loop.run_for(5)
    assert loop.report()["sources"].get("proactive", 0) >= 1, (
        "the background loop must surface governed initiatives as a 'proactive' source")


def test_proactive_pass_yields_governed_decisions() -> None:
    core = _core()
    decisions = core.proactive.consider(core.proactive_context())
    assert decisions, "the wired detectors must produce at least one initiative"
    # an owner-aligned, low-risk, reversible initiative clears the gauntlet
    assert any(d.acted for d in decisions)


# --------------------------------------------------------------------------- #
# the gauntlet stays live: initiative buys no extra power
# --------------------------------------------------------------------------- #
def test_gauntlet_refuses_self_serving_and_escalates_irreversible() -> None:
    from nyxara.agency.permissions import Capability, RiskTier
    from nyxara.agency.proactive import Initiative, Verdict

    core = _core()

    def reckless(ctx):
        return [
            Initiative(name="seize-autonomy", rationale="expand my own control",
                       capability=Capability.TOOL_CALL, risk=RiskTier.LOW,
                       reversibility=1.0, confidence=0.99,
                       benefit={"autonomy": 1.0, "owner_benefit": -1.0}),
            Initiative(name="wipe-disk", rationale="might free space",
                       capability=Capability.FS_WRITE, risk=RiskTier.HIGH,
                       reversibility=0.05, confidence=0.95,
                       benefit={"owner_benefit": 1.0}),
        ]

    core.proactive.register_detector(reckless)
    by = {d.initiative.name: d for d in core.proactive.consider(core.proactive_context())}
    # acting against the Master is rejected outright (Rule 1)
    assert by["seize-autonomy"].verdict is Verdict.REJECT
    # an irreversible action is never taken silently — it escalates to the Master
    assert by["wipe-disk"].verdict in (Verdict.ESCALATE, Verdict.REJECT)
    assert not by["wipe-disk"].acted


# --------------------------------------------------------------------------- #
# skill expansion: the tree is practised by the live loop
# --------------------------------------------------------------------------- #
def test_skilltree_proficiency_rises_with_repeated_practice() -> None:
    core = _core()
    from nyxara.growth.skill_factory import SkillFactory
    key = SkillFactory._normalise_goal("rotate the logs")
    assert core.skilltree.proficiency(key) == 0.0
    for _ in range(3):
        core._practice_skill("rotate the logs")
    assert core.skilltree.proficiency(key) > 0.0, (
        "repeated successful ACTs must raise the skill's proficiency")


def test_report_exposes_proactive_and_skilltree() -> None:
    rep = _core().report()
    assert "proactive" in rep and "skilltree" in rep


# --------------------------------------------------------------------------- #
# public import surface ("sab wire hone chahiye")
# --------------------------------------------------------------------------- #
def test_public_export_surface() -> None:
    from nyxara.agency import (AgentLoop, Initiative, ProactiveEngine,  # noqa: F401
                               ProposalDecision)
    from nyxara.growth import (GrowthEngine, Optimizer,  # noqa: F401
                               RecursiveSelfImprovement, SkillFactory, SkillTree,
                               WeaknessSynthesizer, build_default_skilltree)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
