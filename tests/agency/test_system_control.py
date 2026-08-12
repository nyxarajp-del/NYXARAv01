"""Tests for NYXARA's whole-machine OS-control faculty.

Two layers are locked in here:

* the engine (:class:`nyxara.agency.system_control.SystemControl`) — real system/process/hardware
  introspection, its protected-PID guard, its config toggles (power/service/package/user), and its
  never-raises errors-as-data contract;
* the governed wiring — the system tools register with the right capability/risk and are gated by
  the permission pipeline, so a read runs under OWNER, the root surface (PRIV_ESCALATE) ESCALATES
  under full-control yet is ALLOWED under the opt-in privilege grant, and the package/account tools
  finally exercise the PKG_INSTALL / ACCOUNT_MODIFY capabilities (proving the gate is real).
"""

from __future__ import annotations

import os

import pytest

from nyxara.agency.default_tools import build_default_tools
from nyxara.agency.permissions import (
    Authority,
    Capability,
    PermissionRequest,
    build_default_policy,
    build_sovereign_policy,
    grant_privilege_escalation,
)
from nyxara.agency.system_control import SystemControl
from nyxara.agency.tools import ToolRegistry


# ------------------------------------------------------------------ #
# Engine — real read-only introspection
# ------------------------------------------------------------------ #
def test_system_info_is_real():
    info = SystemControl().system_info()
    assert info["ok"]
    assert info["system"]  # e.g. "Linux"
    assert info["cpu_count"] and info["cpu_count"] >= 1
    assert info["pid"] == os.getpid()


def test_memory_info_reports_totals():
    mem = SystemControl().memory_info()
    assert mem["ok"]
    assert mem["virtual"]["total"] > 0


def test_list_processes_returns_running_procs():
    procs = SystemControl().list_processes(limit=10)
    assert procs["ok"]
    assert procs["processes"]
    assert all("pid" in p for p in procs["processes"])


def test_list_processes_respects_limit():
    procs = SystemControl().list_processes(limit=3)
    assert procs["ok"] and len(procs["processes"]) <= 3


def test_process_info_self():
    info = SystemControl().process_info(os.getpid())
    assert info["ok"] and info["pid"] == os.getpid()


# ------------------------------------------------------------------ #
# Engine — guards refuse as DATA, never raise
# ------------------------------------------------------------------ #
def test_protected_pid_guard_refuses_init():
    r = SystemControl().kill_process(1)
    assert r["ok"] is False and "protected" in r["error"]


def test_protected_pid_guard_refuses_self():
    r = SystemControl().terminate_process(os.getpid())
    assert r["ok"] is False and "own process" in r["error"]


@pytest.mark.parametrize("pid", [0, -1, -2, -os.getpgrp()])
def test_broadcast_pids_are_refused(pid):
    """A non-positive PID is a broadcast, and no single-process guard can cover it.

    ``os.kill(-1, …)`` signals every process the caller may signal; ``0`` and ``-n`` signal a whole
    process group. None of them equals NYXARA's own PID, so ``protect_self`` let them through —
    and her process sits inside the blast radius, so `kill_process(-1)` under her default
    root-enabled posture took out the machine, herself, and with her /scram and the audit feed.
    Emptying ``protected_pids`` must not reopen it: the refusal is structural, not config.
    """
    for sc in (SystemControl(), SystemControl(protected_pids=[])):
        r = sc.kill_process(pid)
        assert r["ok"] is False
        assert "non-positive" in r["error"], r["error"]


def test_signal_missing_process_is_data():
    # A PID that (almost certainly) does not exist -> refusal as data, not a raise.
    r = SystemControl(protected_pids=[]).signal_process(2_000_000_000, "SIGTERM")
    assert r["ok"] is False and "no such process" in r["error"]


def test_power_off_by_default():
    r = SystemControl().power_action("reboot")
    assert r["ok"] is False and "disabled" in r["error"]


def test_unknown_power_action_rejected():
    r = SystemControl(allow_power=True).power_action("frobnicate")
    assert r["ok"] is False and "unknown power action" in r["error"]


def test_unknown_service_verb_rejected():
    r = SystemControl().service_action("cron.service", "frobnicate")
    assert r["ok"] is False and "unknown service action" in r["error"]


def test_disabled_faculty_refuses_effectful_ops():
    sc = SystemControl(enabled=False, protected_pids=[])
    assert sc.signal_process(2_000_000_000, "SIGTERM")["ok"] is False
    assert sc.service_action("cron.service", "restart")["ok"] is False


def test_config_toggles_fence_off_surfaces():
    sc = SystemControl(allow_service_control=False, allow_package_management=False,
                       allow_user_management=False)
    assert "disabled" in sc.service_action("cron.service", "restart")["error"]
    assert "disabled" in sc.package_install("cowsay")["error"]
    assert "disabled" in sc.user_action("add", "someone")["error"]


# ------------------------------------------------------------------ #
# Engine — misc real ops
# ------------------------------------------------------------------ #
def test_env_round_trip():
    sc = SystemControl()
    sc.set_env("NYXARA_TEST_VAR", "42")
    assert sc.get_env("NYXARA_TEST_VAR")["value"] == "42"


def test_list_users_real():
    users = SystemControl().list_users()
    assert users["ok"] and any(u["name"] == "root" for u in users["users"])


def test_signal_name_resolution_accepts_names_and_numbers():
    sc = SystemControl(protected_pids=[])
    # Both a name and a number resolve; the missing PID makes it fail as data either way.
    for sig in ("SIGKILL", "9", "KILL"):
        r = sc.signal_process(2_000_000_000, sig)
        assert r["ok"] is False and "no such process" in r["error"]


# ------------------------------------------------------------------ #
# Governed wiring — the family registers and the gate is real
# ------------------------------------------------------------------ #
def _registry(policy):
    reg = ToolRegistry(policy=policy)
    build_default_tools(reg, system=SystemControl(), enable_code=True)
    return reg


def _auto(cap, **kw):
    return PermissionRequest(cap, authority=Authority.AUTONOMOUS, **kw)


def test_full_system_tool_family_registered():
    reg = _registry(build_sovereign_policy())
    names = set(reg.names())
    expected = {
        "system_info", "cpu_info", "memory_info", "disk_partitions", "hardware_sensors",
        "network_interfaces", "net_connections", "list_processes", "process_info",
        "service_status", "package_query", "list_users", "who", "get_env", "sysctl_get",
        "signal_process", "terminate_process", "kill_process", "set_process_priority", "set_env",
        "package_install", "package_remove", "manage_user",
        "service_control", "power_control", "sysctl_set",
    }
    assert expected <= names


def test_read_tool_runs_under_owner():
    reg = _registry(build_sovereign_policy())
    res = reg.invoke("list_processes", {"limit": 3}, authority=Authority.OWNER)
    assert res.ok and res.value["ok"] and res.value["processes"]


def test_protected_pid_guard_through_the_registry():
    reg = _registry(build_sovereign_policy())
    res = reg.invoke("kill_process", {"pid": 1}, authority=Authority.OWNER)
    # The tool ran (ok) but the engine refused init as data — no crash, no kill.
    assert res.ok and res.value["ok"] is False and "protected" in res.value["error"]


def test_power_refused_as_data_when_off():
    reg = _registry(build_sovereign_policy())
    # power_control is PRIV_ESCALATE; under OWNER authority it runs, and the engine refuses
    # because allow_power is off by default.
    res = reg.invoke("power_control", {"action": "reboot"}, authority=Authority.OWNER)
    assert res.ok and res.value["ok"] is False and "disabled" in res.value["error"]


# ------------------------------------------------------------------ #
# Governed wiring — capability gating is honest
# ------------------------------------------------------------------ #
def test_priv_escalate_surface_escalates_under_full_control():
    # full_control (sovereign) deliberately EXCLUDES PRIV_ESCALATE, so the root surface must
    # escalate to the Master rather than run autonomously.
    pol = build_sovereign_policy()
    assert pol.check(_auto(Capability.PRIV_ESCALATE, target="reboot", reversible=False)).escalated


def test_priv_escalate_surface_allowed_under_privilege_grant():
    pol = grant_privilege_escalation(build_default_policy())
    assert pol.check(_auto(Capability.PRIV_ESCALATE, target="reboot", reversible=False)).allowed


def test_package_and_account_caps_allowed_under_full_control():
    # PKG_INSTALL and ACCOUNT_MODIFY are operational caps -> full_control blesses them so the
    # newly-wired package/account tools act autonomously (finally exercising those capabilities).
    pol = build_sovereign_policy()
    assert pol.check(_auto(Capability.PKG_INSTALL, target="cowsay", reversible=False)).allowed
    assert pol.check(_auto(Capability.ACCOUNT_MODIFY, target="alice", reversible=False)).allowed


def test_process_signalling_escalates_under_bare_policy():
    # PROC_EXEC + irreversible under the conservative envelope -> requires the Master.
    reg = _registry(build_default_policy())
    res = reg.invoke("kill_process", {"pid": 2_000_000_000}, authority=Authority.AUTONOMOUS)
    assert not res.ok and res.requires_owner
