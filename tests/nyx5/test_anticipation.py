"""Proactive anticipation: pre-solves next problems; lookahead=0 is silent; bad prediction discarded."""

from __future__ import annotations

from nyxara.nyx5.anticipation import ProactiveAnticipator


def test_presolves_lookahead_problems():
    a = ProactiveAnticipator(lookahead=3)
    items = a.anticipate("build the reasoner", solver=lambda p: f"solved:{p[:10]}")
    assert len(items) == 3
    assert all(it.solution.startswith("solved:") for it in items)


def test_zero_lookahead_is_silent():
    a = ProactiveAnticipator(lookahead=0)
    assert a.anticipate("x", solver=lambda p: p) == []


def test_bad_prediction_discarded():
    a = ProactiveAnticipator(lookahead=3)

    def flaky(p):
        if "conceptual" in p.lower() or "theoretical" in p.lower():
            raise RuntimeError("cannot solve")
        return "ok"

    items = a.anticipate("optimize throughput", solver=flaky)
    assert len(items) < 3               # the failing direction was dropped, not fatal
    assert all(it.solution == "ok" for it in items)
