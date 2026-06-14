"""Tests for nyxara.sim.sandbox."""

from __future__ import annotations

import os

from nyxara.planning.planner import Action
from nyxara.sim.sandbox import (
    EffectKind,
    InterceptedEffect,
    Sandbox,
)


# -------------------- effect interception -------------------- #
def test_write_file_intercepted_no_real_io():
    sb = Sandbox()
    path = os.path.join(os.path.dirname(__file__), "__sandbox_test__.tmp")
    res = sb.run(lambda ctx: ctx.write_file(path, "data"))
    assert any(e.kind is EffectKind.FILE_WRITE for e in res.effects)
    assert not os.path.exists(path)  # no real file created


def test_network_intercepted():
    sb = Sandbox()
    res = sb.run(lambda ctx: ctx.network("GET", "https://example.com"))
    assert any(e.kind is EffectKind.NETWORK for e in res.effects)
    assert res.value["simulated"] is True


def test_run_command_intercepted():
    sb = Sandbox()
    res = sb.run(lambda ctx: ctx.run_command("rm -rf /"))
    assert any(e.kind is EffectKind.EXEC for e in res.effects)


def test_tool_call_intercepted():
    sb = Sandbox()
    res = sb.run(lambda ctx: ctx.call_tool("search", {"q": "x"}))
    assert any(e.kind is EffectKind.TOOL_CALL for e in res.effects)
    assert res.value["tool"] == "search"


def test_self_edit_intercepted():
    sb = Sandbox()
    res = sb.run(lambda ctx: ctx.self_edit("kernel/rules.py", "change"))
    assert any(e.kind is EffectKind.SELF_EDIT for e in res.effects)


def test_state_change_intercepted():
    sb = Sandbox()
    res = sb.run(lambda ctx: ctx.set_state("k", 1))
    assert any(e.kind is EffectKind.STATE for e in res.effects)


# -------------------- reversibility flags -------------------- #
def test_irreversible_effects_flagged():
    sb = Sandbox()
    res = sb.run(lambda ctx: ctx.run_command("x"))
    assert res.has_irreversible
    assert all(not e.reversible for e in res.effects)


def test_state_is_reversible():
    sb = Sandbox()
    res = sb.run(lambda ctx: ctx.set_state("k", 1))
    assert not res.has_irreversible


def test_network_irreversible():
    sb = Sandbox()
    res = sb.run(lambda ctx: ctx.network("POST", "url"))
    assert res.has_irreversible


# -------------------- rollback / isolation -------------------- #
def test_run_rolls_back_by_default():
    sb = Sandbox()
    sb.run(lambda ctx: ctx.set_state("temp", 99))
    assert "temp" not in sb.state  # rolled back
    assert sb.effects == []


def test_run_no_rollback_persists():
    sb = Sandbox()
    sb.run(lambda ctx: ctx.set_state("x", 42), rollback_after=False)
    assert sb.state["x"] == 42


def test_final_state_captured_before_rollback():
    sb = Sandbox()
    res = sb.run(lambda ctx: ctx.set_state("x", 5))
    assert res.final_state["x"] == 5  # captured even though rolled back
    assert "x" not in sb.state


def test_failing_action_is_data_not_crash():
    sb = Sandbox()
    res = sb.run(lambda ctx: (_ for _ in ()).throw(ValueError("boom")))
    assert not res.success
    assert "ValueError" in res.error


def test_virtual_fs_isolated():
    sb = Sandbox()
    def fn(ctx):
        ctx.write_file("/etc/passwd", "hacked")
        return ctx.read_file("/etc/passwd")
    res = sb.run(fn, rollback_after=False)
    assert res.value == "hacked"  # virtual only
    assert sb.fs["/etc/passwd"] == "hacked"
    # the real /etc/passwd is obviously untouched (we never call real IO)


# -------------------- snapshot / rollback -------------------- #
def test_snapshot_and_rollback():
    sb = Sandbox()
    sid = sb.snapshot()
    sb.context().set_state("temp", 1)
    assert sb.state["temp"] == 1
    assert sb.rollback(sid)
    assert "temp" not in sb.state


def test_rollback_unknown_snapshot():
    sb = Sandbox()
    assert sb.rollback("nope") is False


def test_rollback_restores_effects_count():
    sb = Sandbox()
    sb.context().set_state("a", 1)
    sid = sb.snapshot()
    sb.context().set_state("b", 2)
    assert len(sb.effects) == 2
    sb.rollback(sid)
    assert len(sb.effects) == 1


def test_reset():
    sb = Sandbox()
    sb.context().set_state("k", 1)
    sb.snapshot()
    sb.reset()
    assert sb.state == {} and sb.effects == [] and sb.status()["snapshots"] == 0


# -------------------- STRIPS dry-run -------------------- #
def _chain():
    return [
        Action.make("get_key", add=["have_key"]),
        Action.make("open_door", pre=["have_key"], add=["door_open"]),
        Action.make("enter", pre=["door_open"], add=["inside"]),
    ]


def test_dry_run_plan_success():
    sb = Sandbox()
    res = sb.dry_run_plan(_chain(), initial_facts=[], goal=["inside"])
    assert res.success
    assert "inside" in res.final_state["facts"]


def test_dry_run_plan_records_effects():
    sb = Sandbox()
    res = sb.dry_run_plan(_chain(), initial_facts=[])
    assert len(res.effects) == 3  # one per applied action


def test_dry_run_plan_inapplicable_action():
    sb = Sandbox()
    res = sb.dry_run_plan([Action.make("enter", pre=["door_open"], add=["inside"])],
                          initial_facts=[], goal=["inside"])
    assert not res.success
    assert "not applicable" in res.error


def test_dry_run_plan_goal_not_reached():
    sb = Sandbox()
    res = sb.dry_run_plan([Action.make("get_key", add=["have_key"])],
                          initial_facts=[], goal=["inside"])
    assert not res.success
    assert "did not reach" in res.error


# -------------------- result shape -------------------- #
def test_result_to_dict():
    sb = Sandbox()
    res = sb.run(lambda ctx: ctx.set_state("k", 1))
    d = res.to_dict()
    assert "effects" in d and "has_irreversible" in d and "success" in d


def test_effects_by_kind():
    sb = Sandbox()
    def fn(ctx):
        ctx.set_state("a", 1)
        ctx.set_state("b", 2)
        ctx.run_command("x")
    res = sb.run(fn)
    counts = res.effects_by_kind()
    assert counts["state"] == 2 and counts["exec"] == 1


def test_intercepted_effect_to_dict():
    e = InterceptedEffect(EffectKind.STATE, "k", 1)
    assert e.to_dict()["kind"] == "state"


def test_status():
    sb = Sandbox()
    sb.context().set_state("k", 1)
    s = sb.status()
    assert "effects" in s and "state_keys" in s
