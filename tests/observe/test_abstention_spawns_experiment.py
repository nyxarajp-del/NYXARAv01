"""Stage G · delta 2 — an honest 'I don't know' opens a standing experiment.

An abstention used to just end the turn. Now the gap is seeded onto her curiosity frontier so a
background tick investigates it. Tested via ActiveCuriosity.seed (the real seam) and the core method's
own logic (lightweight duck-typed self — no full-core construction needed).
"""

from __future__ import annotations

import types

from nyxara.growth.active_curiosity import ActiveCuriosity
from nyxara.kernel.orchestrator import NyxaraCore


def test_curiosity_seed_pushes_a_subject_onto_the_frontier():
    cur = ActiveCuriosity()
    assert cur.seed("why does the daemon stall under packet loss?") is True
    assert "why does the daemon stall under packet loss?" in cur.frontier()
    # empty / whitespace subjects are ignored, never crash
    assert cur.seed("   ") is False
    assert cur.seed("") is False


def test_seeded_subject_is_investigated_by_a_later_tick():
    cur = ActiveCuriosity()
    cur.seed("packet-loss deadlock")
    cp = cur.wonder()          # an idle pass with no explicit event pulls from the frontier
    assert cp.event == "packet-loss deadlock"


def test_register_knowledge_gap_seeds_curiosity():
    class _Cur:
        def __init__(self):
            self.seeded = []

        def seed(self, s):
            self.seeded.append(s)
            return True

    obj = types.SimpleNamespace(active_curiosity=_Cur(), mind=None)
    out = NyxaraCore.register_knowledge_gap(obj, "quantum error correction")
    assert out["seeded"] is True
    assert obj.active_curiosity.seeded == ["quantum error correction"]
    assert out["topic"] == "quantum error correction"


def test_register_knowledge_gap_empty_topic_is_a_noop():
    obj = types.SimpleNamespace(active_curiosity=None, mind=None)
    out = NyxaraCore.register_knowledge_gap(obj, "   ")
    assert out["seeded"] is False and out["noted"] is False
