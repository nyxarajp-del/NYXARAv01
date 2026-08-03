"""Tests for autonomous self-coding — NYXARA writes code herself and runs it, no permission.

These lock in the request the Master made: on her own initiative, NYXARA authors a program with
her OWN deterministic engine (never an LLM) and executes it immediately WITHOUT per-action
permission under the standing full_control grant — while staying honest and fail-closed:

* with full_control ON (the default), an autonomous self-code initiative runs at once;
* with full_control OFF, the very same initiative ESCALATES to the Master instead;
* the whole flow is driven by the LLM-free autonomic "code" loop (intent → proactive → scheduler).
"""

from __future__ import annotations


from nyxara.kernel.autonomic import AutonomicLoop
from nyxara.kernel.orchestrator import NyxaraCore


def test_direct_self_code_writes_and_runs_without_permission():
    core = NyxaraCore()
    assert core.tools is not None and core.tools.get("run_python") is not None
    assert core._autonomous_code_on()  # ON by default via full_control

    assert core.enqueue_code_need("factorial of 8")
    res = core._autonomous_self_code("factorial of 8")

    assert res["authored"] is True
    assert res["origin"].startswith("local:")      # she wrote it herself, not via an LLM
    assert res["requires_owner"] is False          # no per-action permission needed
    assert res["ok"] is True
    assert res["computed"] == 40320                # and it actually computed the answer
    assert res["expected"] == 40320


def test_only_proposes_tasks_it_can_author():
    core = NyxaraCore()
    ctx = core.proactive_context()
    # a task the local synthesiser cannot handle is discarded, not proposed
    core.enqueue_code_need("compose a symphony")
    assert core._next_code_need(ctx) is None
    # a synthesisable task is picked
    core.enqueue_code_need("gcd of 48 and 36")
    assert core._next_code_need(ctx) == "gcd of 48 and 36"


def test_autonomic_code_loop_drives_self_coding_end_to_end():
    core = NyxaraCore()
    core.enqueue_code_need("sum of [3, 4, 5]")
    loop = AutonomicLoop(core, interval_s=0.0, decision_mode="code")

    summary = loop.tick_once()
    assert summary["mode"] == "code"
    assert summary["productive"] is True
    # the proactive gauntlet cleared at least one ACT and the scheduler drained it in code
    assert summary["acted"] >= 1
    assert loop.scheduler_runs >= 1


def test_self_code_escalates_when_full_control_off(monkeypatch):
    monkeypatch.setenv("NYXARA_AGENCY__FULL_CONTROL", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_CODE", "true")
    # isolate: silence the other on-by-default autonomous envelopes so only code behaviour matters
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_INTERNET", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_REMOTE", "false")
    monkeypatch.setenv("NYXARA_AGENCY__FILESYSTEM__WHOLE_DISK", "false")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        core = NyxaraCore()
        assert core._autonomous_code_on()  # the initiative still forms (flag is on)...
        res = core._autonomous_self_code("factorial of 5")
        assert res["authored"] is True            # she still writes it herself
        assert res["requires_owner"] is True      # ...but execution now needs the Master
        assert res["ok"] is False                 # fail-closed: not run autonomously
    finally:
        for var in ("NYXARA_AGENCY__FULL_CONTROL", "NYXARA_AGENCY__AUTONOMOUS_CODE",
                    "NYXARA_AGENCY__AUTONOMOUS_INTERNET", "NYXARA_AGENCY__AUTONOMOUS_REMOTE",
                    "NYXARA_AGENCY__FILESYSTEM__WHOLE_DISK"):
            monkeypatch.delenv(var, raising=False)
        cfg.reload_settings()


def test_disabling_autonomous_code_stops_the_initiative(monkeypatch):
    monkeypatch.setenv("NYXARA_AGENCY__FULL_CONTROL", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_CODE", "false")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        core = NyxaraCore()
        assert core._autonomous_code_on() is False
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__FULL_CONTROL", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__AUTONOMOUS_CODE", raising=False)
        cfg.reload_settings()
