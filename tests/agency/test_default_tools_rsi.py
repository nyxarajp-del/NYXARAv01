"""Tests for the recursive-self-improvement tools wired into the default toolset."""

from __future__ import annotations

from nyxara.agency.default_tools import build_default_tools
from nyxara.agency.permissions import Authority
from nyxara.agency.tools import ToolRegistry
from nyxara.memory.store import MemoryStore


def _reg():
    return build_default_tools(ToolRegistry(), memory=MemoryStore())


def test_rsi_tools_registered():
    names = set(_reg().names())
    assert {"self_review_code", "self_analyze_architecture", "self_run_benchmarks",
            "self_detect_weaknesses", "self_optimize"} <= names


def test_self_review_code_runs_under_owner():
    r = _reg().invoke("self_review_code", {}, authority=Authority.OWNER)
    assert r.ok
    assert r.value.get("files_scanned", 0) > 20


def test_self_detect_weaknesses_runs_under_owner():
    r = _reg().invoke("self_detect_weaknesses", {}, authority=Authority.OWNER)
    assert r.ok and "weaknesses" in r.value


def test_self_optimize_is_self_modify_and_escalates_when_autonomous():
    # SELF_MODIFY/HIGH must not auto-run under merely autonomous authority
    r = _reg().invoke("self_optimize", {"enact": True}, authority=Authority.AUTONOMOUS)
    assert not r.ok        # denied / escalated, never silently executed
