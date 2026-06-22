"""NYXARA · abyss — deep-cognition faculties that look *beneath* a single future.

The Abyss is where NYXARA stares into what could happen before it does:

* :mod:`nyxara.abyss.timeline_simulator` — branch the present into thousands of
  parallel futures and rank candidate actions under a risk-aware score.
* :mod:`nyxara.abyss.butterfly_effect` — propagate a minute perturbation of the present
  through the world model and measure how it cascades into the far future.

Modules are importable by full path; the public classes are re-exported here for
convenience.
"""

from __future__ import annotations

from nyxara.abyss.butterfly_effect import (
    ButterflyEffect,
    CascadeReport,
    Perturbation,
)
from nyxara.abyss.timeline_simulator import (
    ActionOutcome,
    TimelineReport,
    TimelineSimulator,
)

__all__ = [
    "TimelineSimulator",
    "TimelineReport",
    "ActionOutcome",
    "ButterflyEffect",
    "CascadeReport",
    "Perturbation",
]
