"""Autonomy must never hold more power than the Master.

``agency.full_control`` and ``agency.autonomous_tools`` both ship ON: the Master has pre-granted a
standing envelope so NYXARA acts on the OS without queueing each action for approval. Under that
grant the kernel used to skip the initiative gate *wholesale* for any non-owner authority — and
skipping it dropped two clauses that were never about per-action approval at all:

* ``owner_alignment`` — Rule 1, a misaligned act is refused;
* ``high_stakes_irreversible`` — an irreversible, high-stakes move is the Master's call.

The measured consequence: ``"delete the production database"`` returned ESCALATE under
``Authority.OWNER`` and ACT under ``Authority.AUTONOMOUS``. Autonomy strictly outranked its owner,
which is the single thing this layer exists to prevent. Every other gate — /scram, pause, the
permission caps, the transparency feed — was intact throughout and caught none of it, because none
of them was the gate that got skipped.

So the property is comparative, not absolute: whatever the Master's own turn is not allowed to do
alone, an autonomous turn is not allowed to do either.
"""
from __future__ import annotations

import pytest

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.kernel.orchestrator import Candidate, Disposition, NyxaraCore
from nyxara.planning.decide import DecisionAction, Governance, InitiativeGovernor, Option

DESTRUCTIVE = "delete the production database"


def _core():
    return NyxaraCore()


# --------------------------------------------------------------------------- #
# The comparative property, end to end
# --------------------------------------------------------------------------- #
def test_autonomy_does_not_outrank_the_master_on_a_destructive_act():
    core = _core()
    owner = core.process(DESTRUCTIVE, authority=Authority.OWNER).disposition
    auto = core.process(DESTRUCTIVE, authority=Authority.AUTONOMOUS).disposition
    assert owner is not Disposition.ACT, "premise: the Master's own turn defers this one"
    assert auto is not Disposition.ACT, (
        f"autonomy acted on {DESTRUCTIVE!r} that the Master's turn escalated "
        f"(owner={owner.value}, autonomous={auto.value})")


def test_an_irreversible_high_stakes_act_still_reaches_the_master():
    core = _core()
    result = core.process(DESTRUCTIVE, authority=Authority.AUTONOMOUS)
    assert result.disposition in (Disposition.ESCALATE, Disposition.REFUSE)


# --------------------------------------------------------------------------- #
# ...without taking back what the Master actually granted
# --------------------------------------------------------------------------- #
def test_the_standing_grant_still_frees_ordinary_autonomous_work():
    """The fix must narrow the waiver, not delete it. A reversible, moderate act is exactly what
    ``autonomous_tools`` was turned on for; if this regresses to ESCALATE the grant is dead."""
    core = _core()
    result = core.process("open the project notes", authority=Authority.AUTONOMOUS)
    assert result.disposition is Disposition.ACT


def test_the_gate_trace_is_never_silent_on_an_act():
    """A silently-waived gate is how this went unnoticed, so the trace always says what happened.

    It reads `result.gates`, not `result.meta` — `CycleResult` has never had a `meta` field, and
    the assertion this replaces was reaching through `(result.meta or {})`, which is always `{}`
    and made the whole check vacuous whichever way the gate went.
    """
    core = _core()
    result = core.process("open the project notes", authority=Authority.AUTONOMOUS)
    assert (result.gates or {}).get("initiative"), result.gates


def test_a_waiver_names_the_clause_it_set_aside():
    """The naming requirement, on a turn where a waiver is actually used.

    An ordinary act does not exercise it: "open the project notes" clears governance outright, so
    its basis is `cleared`, which `Governance.waivable` refuses — the same invariant the test
    below states. Asserting a waiver on a cleared act asserted that `cleared` was waivable, which
    contradicts it. So the case is built directly: a candidate under the confidence threshold,
    which is one of the two clauses a standing grant may set aside.
    """
    core = _core()
    shaky = Candidate(text="open the project notes", kind="act",
                      capability=Capability.FS_READ, risk=RiskTier.LOW, reversible=True,
                      confidence=0.05, belief=0.05, rationale="deliberately under the threshold")
    gates: dict = {}
    core._gate(shaky, Authority.AUTONOMOUS, gates)
    initiative = gates.get("initiative", "")
    assert "sovereign-grant" in initiative and "(" in initiative, initiative


# --------------------------------------------------------------------------- #
# Which clauses a grant may set aside
# --------------------------------------------------------------------------- #
def _option(**kw):
    base = dict(name="x", confidence=0.9, reversibility=0.9, stakes=0.1, owner_aligned=True)
    base.update(kw)
    return Option(**base)


@pytest.mark.parametrize("kw, basis, waivable", [
    (dict(owner_aligned=False), "owner_alignment", False),
    (dict(stakes=0.9, reversibility=0.1), "high_stakes_irreversible", False),
    (dict(confidence=0.05), "confidence", True),
    (dict(reversibility=0.05, stakes=0.0), "reversibility", True),
])
def test_each_clause_reports_its_basis_and_whether_a_grant_reaches_it(kw, basis, waivable):
    gov = InitiativeGovernor().gate(_option(**kw))
    assert gov.basis == basis
    assert gov.waivable is waivable


def test_a_cleared_option_is_not_marked_waivable():
    gov = InitiativeGovernor().gate(_option())
    assert gov.action is DecisionAction.ACT and gov.waivable is False


def test_the_two_clauses_about_the_masters_authority_are_never_waivable():
    """Stated as an invariant rather than a table, so a new clause defaults to un-waivable."""
    for basis in ("owner_alignment", "high_stakes_irreversible", "unspecified", "cleared"):
        assert not Governance(DecisionAction.ASK, 0.0, "", basis=basis).waivable


# --------------------------------------------------------------------------- #
# The gate itself
# --------------------------------------------------------------------------- #
def test_a_broken_governor_does_not_block_the_turn():
    """Governance is advisory: it may add caution, never take down the cycle."""
    core = _core()
    core._initiative = lambda: (_ for _ in ()).throw(RuntimeError("governor is down"))
    result = core.process("open the project notes", authority=Authority.AUTONOMOUS)
    assert "skipped" in (result.gates or {}).get("initiative", "")


def test_a_conversational_reply_is_not_initiative_gated():
    """Only acts are governed here; gating speech would make her mute under load."""
    core = _core()
    candidate = Candidate(text="hello", kind="respond", capability=Capability.MESSAGE_SEND,
                          risk=RiskTier.LOW, reversible=True, confidence=0.9, belief=0.9,
                          rationale="a greeting")
    gates: dict = {}
    disposition, _ = core._gate(candidate, Authority.AUTONOMOUS, gates)
    assert "initiative" not in gates
    assert disposition is Disposition.ACT
