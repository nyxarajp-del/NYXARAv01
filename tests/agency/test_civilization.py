"""Tests for nyxara.agency.civilization — the multi-agent internal civilization.

Covers the upgrade from read-only monitors to a coordinating society: inter-agent messaging
on the shared blackboard, once-only message consumption, and the gated action framework
(fail-closed when autonomy is OFF, budget-limited and journalled when ON).
"""

from __future__ import annotations

from nyxara.agency.civilization import (AgentMessage, AgentReport, MicroAgentCivilization)


def _force_due(civ: MicroAgentCivilization) -> None:
    for a in civ.agents:
        a._last_tick = 0.0


# --------------------------------------------------------------------------- #
# baseline structure (preserved from the original monitors)
# --------------------------------------------------------------------------- #
def test_seven_agents_all_tick():
    civ = MicroAgentCivilization(core=None)
    _force_due(civ)
    reports = civ.tick_all()
    assert len(reports) == 7
    assert {r.agent_name for r in reports} == {
        "Memory", "Security", "Planning", "Research", "Learning", "Coding", "Prediction"}
    assert civ.status()["total_reports"] == 7


def test_pause_and_resume():
    civ = MicroAgentCivilization(core=None)
    assert civ.pause_agent("Security") and not civ.agents[1]._active
    assert civ.resume_agent("Security") and civ.agents[1]._active
    assert civ.pause_agent("Nope") is False


def test_report_to_dict_has_new_fields():
    d = AgentReport(agent_name="A", domain="d").to_dict()
    assert "actions" in d and "messages" in d and d["actions"] == []


def test_duty_cycle_skips_when_not_due():
    civ = MicroAgentCivilization(core=None)
    _force_due(civ)
    assert len(civ.tick_all()) == 7
    assert civ.tick_all() == []                 # nothing is due immediately again


# --------------------------------------------------------------------------- #
# inter-agent messaging + once-only consumption
# --------------------------------------------------------------------------- #
class _FakeJournal:
    def actions(self):
        class _A:
            payload = {"status": "FAIL"}
        return [_A()] * 10                       # 10 recent failures ⇒ Security alerts


class _FakeGoals:
    def __init__(self):
        self.filed = []

    def prioritize(self):
        return []

    def add(self, name):
        self.filed.append(name)
        return name


class _FakeCore:
    def __init__(self):
        self.journal = _FakeJournal()
        self.goals = _FakeGoals()


def test_security_alert_reaches_planning():
    core = _FakeCore()
    civ = MicroAgentCivilization(core=core, enact_actions=True, max_actions_per_cycle=5)
    _force_due(civ)
    civ.tick_all()                               # Security posts an alert; Planning reacts same cycle
    assert any(m.topic == "security.alert" for m in civ.recent_messages())
    assert core.goals.filed                      # Planning filed an investigate-goal


def test_message_consumed_once_per_new_alert():
    core = _FakeCore()
    civ = MicroAgentCivilization(core=core, enact_actions=True, max_actions_per_cycle=10)
    _force_due(civ)
    civ.tick_all()
    _force_due(civ)
    civ.tick_all()
    # two cycles ⇒ two *distinct* alerts ⇒ exactly two reactions (no re-acting on old messages)
    assert len(core.goals.filed) == 2


def test_inbox_returns_only_unconsumed():
    civ = MicroAgentCivilization(core=None)
    civ._post_message(AgentMessage(sender="Security", topic="security.alert", content="x"))
    first = civ._read_new("Planning", topic="security.alert", snapshot=civ._msg_seq)
    assert len(first) == 1
    civ._cursors["Planning"] = civ._msg_seq      # mark consumed
    again = civ._read_new("Planning", topic="security.alert", snapshot=civ._msg_seq)
    assert again == []                           # not re-delivered


# --------------------------------------------------------------------------- #
# gated actions: fail-closed by default, budgeted + journalled when authorised
# --------------------------------------------------------------------------- #
def test_autonomy_off_takes_no_actions():
    core = _FakeCore()
    civ = MicroAgentCivilization(core=core, enact_actions=False)
    _force_due(civ)
    reports = civ.tick_all()
    assert all(not r.actions for r in reports)   # observe-only
    assert not core.goals.filed
    assert civ.status()["total_actions"] == 0


def test_action_budget_is_respected():
    core = _FakeCore()
    civ = MicroAgentCivilization(core=core, enact_actions=True, max_actions_per_cycle=1)
    _force_due(civ)
    civ.tick_all()
    _force_due(civ)
    civ.tick_all()
    # at most one gated action per cycle, regardless of how many agents wanted to act
    assert civ.status()["total_actions"] <= 2


def test_memory_agent_consolidates_when_large_and_authorised():
    class _Mem:
        def __init__(self):
            self.consolidated = 0

        def stats(self):
            return {"total": 500}

        def consolidate(self):
            self.consolidated += 1
            return "ok"

    class _Core:
        memory = _Mem()

    core = _Core()
    civ = MicroAgentCivilization(core=core, enact_actions=True, max_actions_per_cycle=5)
    _force_due(civ)
    reports = civ.tick_all()
    mem_report = next(r for r in reports if r.agent_name == "Memory")
    assert core.memory.consolidated == 1
    assert any("consolidate memory" in a for a in mem_report.actions)


def test_settings_drive_autonomy_default():
    from nyxara.kernel.config import get_settings
    s = get_settings().model_copy(deep=True)
    s.agency.civilization_autonomous = True
    civ = MicroAgentCivilization(core=None, settings=s)
    assert civ.enact_actions is True


def test_start_stop_background_thread():
    civ = MicroAgentCivilization(core=None)
    civ.start()
    assert civ._thread is not None and civ._thread.is_alive()
    civ.stop()
    assert civ._thread is None or not civ._thread.is_alive()
