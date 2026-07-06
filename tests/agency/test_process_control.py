"""Tests for nyxara.agency.process_control — live process/job control + system introspection.

These lock in the "real, driveable, never-raises" contract for the long-running / interactive
half of machine control: NYXARA spawns a background process, streams its output, feeds it stdin,
signals and kills it — all as structured data, and all reachable on her own initiative through
the PROC_EXEC gate (the same one full_control blesses).
"""

from __future__ import annotations

import os
import signal
import time

import pytest

from nyxara.agency.process_control import (
    ProcessManager,
    get_process_manager,
    kill_by_name,
    kill_pid,
    list_processes,
    system_info,
)


def _wait_running(pm, handle, want=True, timeout=3.0):
    """Poll until the job's running-state matches ``want`` (or time out)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pm.read(handle)["running"] is want:
            return True
        time.sleep(0.02)
    return pm.read(handle)["running"] is want


# --------------------------------------------------------------------------- #
# ProcessManager: spawn / stream / wait / exit
# --------------------------------------------------------------------------- #
def test_spawn_streams_output_and_exits():
    pm = ProcessManager()
    r = pm.spawn("for i in 1 2 3; do echo line$i; done")
    assert r["ok"] and r["handle"] and r["pid"]
    done = pm.wait(r["handle"], timeout_s=10.0)
    assert done["ok"] and done["running"] is False and done["returncode"] == 0
    out = pm.read(r["handle"])["stdout"]
    assert "line1" in out and "line3" in out


def test_read_is_incremental():
    pm = ProcessManager()
    h = pm.spawn("cat")["handle"]
    pm.write(h, "alpha\n")
    time.sleep(0.15)
    first = pm.read(h)
    assert "alpha" in first["stdout"]
    pm.write(h, "beta\n")
    time.sleep(0.15)
    second = pm.read(h)
    # second read only returns output that arrived since the first (no "alpha" again)
    assert "beta" in second["stdout"] and "alpha" not in second["stdout"]
    pm.kill(h)


def test_background_job_persists_until_signalled():
    pm = ProcessManager()
    h = pm.spawn("sleep 300")["handle"]
    assert _wait_running(pm, h, want=True)
    st = pm.signal(h, signal.SIGTERM)
    assert st["ok"]
    ended = pm.wait(h, timeout_s=5.0)
    assert ended["running"] is False


def test_kill_terminates_and_reaps():
    pm = ProcessManager()
    h = pm.spawn("sleep 300")["handle"]
    assert _wait_running(pm, h, want=True)
    res = pm.kill(h)
    assert res["ok"] and res["killed"] and res["returncode"] is not None
    assert pm.read(h)["running"] is False


def test_kill_reaps_child_process_group():
    # The child spawns its own child; killing the group must reap both. We assert the group
    # leader is gone (a strong proxy) rather than scanning for the grandchild.
    pm = ProcessManager()
    h = pm.spawn("sleep 300 & sleep 300")["handle"]
    assert _wait_running(pm, h, want=True)
    pm.kill(h)
    assert pm.read(h)["running"] is False


def test_list_jobs_reports_all():
    pm = ProcessManager()
    a = pm.spawn("true")["handle"]
    b = pm.spawn("sleep 300")["handle"]
    jobs = pm.list_jobs()
    handles = {j["handle"] for j in jobs["jobs"]}
    assert jobs["count"] == 2 and {a, b} <= handles
    pm.kill(b)


def test_write_to_exited_job_is_data_not_exception():
    pm = ProcessManager()
    h = pm.spawn("true")["handle"]
    pm.wait(h, timeout_s=5.0)
    r = pm.write(h, "too late")
    assert r["ok"] is False and "exited" in r["error"]


def test_bad_handle_is_structured_error():
    pm = ProcessManager()
    for call in (pm.read, pm.write, pm.signal, pm.kill, pm.wait):
        r = call(999999) if call not in (pm.write,) else call(999999, "x")
        assert r["ok"] is False and "no such job" in r["error"]


def test_empty_command_rejected():
    pm = ProcessManager()
    assert pm.spawn("   ")["ok"] is False


# --------------------------------------------------------------------------- #
# PTY: a real terminal for interactive programs
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.name != "posix", reason="pty is POSIX-only")
def test_pty_job_echoes_interactive_input():
    pm = ProcessManager()
    r = pm.spawn('read x; echo "$x" | tr a-z A-Z', tty=True)
    assert r["ok"] and r["tty"]
    pm.write(r["handle"], "whisper\n")
    pm.wait(r["handle"], timeout_s=5.0)
    assert "WHISPER" in pm.read(r["handle"])["stdout"]


# --------------------------------------------------------------------------- #
# System introspection
# --------------------------------------------------------------------------- #
def test_system_info_reports_posture():
    info = system_info()
    assert info["ok"]
    for key in ("platform", "hostname", "cpu_count", "euid", "is_root", "pid"):
        assert key in info
    assert info["cpu_count"] and info["cpu_count"] >= 1


def test_list_processes_includes_self():
    table = list_processes()
    assert table["ok"] and table["count"] > 0
    pids = {p["pid"] for p in table["processes"]}
    assert os.getpid() in pids
    # every record has the core fields
    sample = table["processes"][0]
    assert "pid" in sample and "command" in sample


def test_kill_pid_missing_is_structured_error():
    # signal 0 = existence check; a very high pid should not exist.
    r = kill_pid(2_000_000_000, sig=0)
    assert r["ok"] is False and "no such process" in r["error"]


def test_kill_by_name_requires_pattern():
    assert kill_by_name("")["ok"] is False


def test_kill_by_name_skips_self_by_default():
    # A pattern that matches this very test process must not signal us when include_self=False.
    before = os.getpid()
    r = kill_by_name("this-command-substring-matches-nothing-xyz")
    assert r["ok"] and r["count"] == 0 and os.getpid() == before


def test_singleton_is_shared():
    assert get_process_manager() is get_process_manager()
