"""NYXARA · temporal — Fractal Temporal Hierarchies (the multi-dimensional mind).

A normal AI thinks in one cadence, token by token. NYXARA thinks at **three time scales at
once**, as loops nested inside loops:

* **Layer 1 — milliseconds** (:class:`~nyxara.temporal.micro.MicroLayer`): a sub-agent that
  feels the machine — real CPU, memory, load, and network packet/byte counters — and flags
  anomalies as they happen.
* **Layer 2 — seconds** (:class:`~nyxara.temporal.meso.MesoLayer`): an observer of every
  sovereign turn — prompts read, code written — rolled up into behavioural aggregates.
* **Layer 3 — days/months** (:class:`~nyxara.temporal.macro.MacroLayer`): the *Master AI*,
  which slowly watches how the Master's behaviour changes over weeks and, through a
  fail-closed gate, adjusts NYXARA's goals and drives to track them — never touching her
  sealed character or loyalty.

:class:`~nyxara.temporal.fractal.FractalTemporalHierarchy` composes the three and drives
them as one — deterministically for tests/cron, or live as background asyncio tasks.

Modules are importable by full path; the public surface is re-exported here.
"""

from __future__ import annotations

from nyxara.temporal.clock import (
    DAY,
    HOUR,
    MILLISECOND,
    MINUTE,
    MONTH,
    SECOND,
    WEEK,
    RealClock,
    TemporalClock,
    VirtualClock,
)
from nyxara.temporal.fractal import FractalTemporalHierarchy
from nyxara.temporal.macro import MacroLayer
from nyxara.temporal.meso import MesoLayer, looks_like_code
from nyxara.temporal.micro import MicroLayer
from nyxara.temporal.signals import (
    Awareness,
    BehaviorEpoch,
    GoalAdjustment,
    HardwareSample,
    LayerReport,
    MesoAggregate,
    MicroAggregate,
    NetworkSample,
    TurnObservation,
)

__all__ = [
    # the composer + driver
    "FractalTemporalHierarchy",
    # the three layers
    "MicroLayer",
    "MesoLayer",
    "MacroLayer",
    "looks_like_code",
    # clocks + scales
    "TemporalClock",
    "RealClock",
    "VirtualClock",
    "MILLISECOND",
    "SECOND",
    "MINUTE",
    "HOUR",
    "DAY",
    "WEEK",
    "MONTH",
    # signal records
    "HardwareSample",
    "NetworkSample",
    "MicroAggregate",
    "TurnObservation",
    "MesoAggregate",
    "BehaviorEpoch",
    "Awareness",
    "GoalAdjustment",
    "LayerReport",
]
