"""Stage G · delta 1 — elapsed-time awareness on return.

The heartbeat clock advanced through an absence but nothing *registered* it. Now she computes an
honest, human phrasing of the gap since the Master last spoke and returns knowing time passed.
Tested via the pure humaniser + the method's own logic (no full-core construction needed).
"""

from __future__ import annotations

import time
import types

from nyxara.kernel.orchestrator import NyxaraCore, _humanize_gap


def test_humanize_gap_uses_natural_units():
    assert _humanize_gap(0) == "just now"
    assert _humanize_gap(30) == "just now"
    assert _humanize_gap(60) == "~1 minute"
    assert _humanize_gap(180) == "~3 minutes"
    assert _humanize_gap(3600) == "~1 hour"
    assert _humanize_gap(3 * 3600) == "~3 hours"
    assert _humanize_gap(86400) == "~1 day"
    assert _humanize_gap(2 * 86400) == "~2 days"


def test_time_away_returns_registered_absence():
    # a turn already registered a 2-day absence at the head of the turn
    obj = types.SimpleNamespace(_last_absence={"seconds": 172800.0, "phrase": "~2 days", "at": 1.0})
    out = NyxaraCore.time_away(obj)
    assert out["phrase"] == "~2 days" and out["seconds"] == 172800.0


def test_time_away_falls_back_to_live_estimate():
    # no turn has run yet → compute a live estimate from the last-interaction clock
    obj = types.SimpleNamespace(_last_interaction=time.time() - 3600.0)
    out = NyxaraCore.time_away(obj)
    assert out["phrase"] == "~1 hour"
    assert 3500 <= out["seconds"] <= 3700
