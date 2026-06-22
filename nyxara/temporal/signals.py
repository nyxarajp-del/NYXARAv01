"""NYXARA · temporal/signals.py — the records that flow up the fractal hierarchy.

Each temporal layer speaks in small, serialisable records. The fast layers emit raw
samples that roll up into aggregates; the aggregates roll up, in turn, into long-horizon
**epochs** and finally into an **Awareness** the slow "Master AI" layer reasons over.

Keeping every datum a plain dataclass with a ``to_dict()`` means the whole hierarchy is
introspectable, loggable, and JSON-persistable end to end — nothing is hidden inside a
loop. Pure standard library.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
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


# --------------------------------------------------------------------------- #
# Layer 1 — milliseconds: real hardware + network
# --------------------------------------------------------------------------- #
@dataclass
class HardwareSample:
    """One instantaneous reading of the machine NYXARA runs on."""

    at: float = field(default_factory=time.time)
    cpu_percent: float = 0.0          # [0,100] busy fraction since the previous sample
    mem_percent: float = 0.0          # [0,100] used physical memory
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0
    load_avg_1m: float = 0.0          # 1-minute run-queue load average
    n_procs: int = 0                  # processes/threads visible
    source: str = "unknown"           # "proc" (real /proc) | "fallback" | "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {"at": self.at, "cpu_percent": round(self.cpu_percent, 2),
                "mem_percent": round(self.mem_percent, 2),
                "mem_used_mb": round(self.mem_used_mb, 1),
                "mem_total_mb": round(self.mem_total_mb, 1),
                "load_avg_1m": round(self.load_avg_1m, 3),
                "n_procs": self.n_procs, "source": self.source}


@dataclass
class NetworkSample:
    """Cumulative network counters at an instant. Rates are derived from deltas."""

    at: float = field(default_factory=time.time)
    rx_bytes: int = 0
    tx_bytes: int = 0
    rx_packets: int = 0
    tx_packets: int = 0
    source: str = "unknown"           # "proc" (real /proc/net/dev) | "fallback"

    def to_dict(self) -> Dict[str, Any]:
        return {"at": self.at, "rx_bytes": self.rx_bytes, "tx_bytes": self.tx_bytes,
                "rx_packets": self.rx_packets, "tx_packets": self.tx_packets,
                "source": self.source}


@dataclass
class MicroAggregate:
    """A rolled-up window of millisecond samples — the Layer-1 → Layer-2 message."""

    at: float = field(default_factory=time.time)
    span_s: float = 0.0
    n_samples: int = 0
    cpu_mean: float = 0.0
    cpu_max: float = 0.0
    mem_mean: float = 0.0
    mem_max: float = 0.0
    rx_bytes_per_s: float = 0.0
    tx_bytes_per_s: float = 0.0
    rx_packets_per_s: float = 0.0
    tx_packets_per_s: float = 0.0
    anomalies: List[str] = field(default_factory=list)   # human-readable spike notes

    @property
    def healthy(self) -> bool:
        return not self.anomalies

    def to_dict(self) -> Dict[str, Any]:
        return {"at": self.at, "span_s": round(self.span_s, 3), "n_samples": self.n_samples,
                "cpu_mean": round(self.cpu_mean, 2), "cpu_max": round(self.cpu_max, 2),
                "mem_mean": round(self.mem_mean, 2), "mem_max": round(self.mem_max, 2),
                "rx_bytes_per_s": round(self.rx_bytes_per_s, 1),
                "tx_bytes_per_s": round(self.tx_bytes_per_s, 1),
                "rx_packets_per_s": round(self.rx_packets_per_s, 2),
                "tx_packets_per_s": round(self.tx_packets_per_s, 2),
                "anomalies": list(self.anomalies), "healthy": self.healthy}


# --------------------------------------------------------------------------- #
# Layer 2 — seconds: reading prompts, writing code
# --------------------------------------------------------------------------- #
@dataclass
class TurnObservation:
    """What NYXARA did on one sovereign turn (one prompt read / answer written)."""

    at: float = field(default_factory=time.time)
    stimulus_len: int = 0
    disposition: str = ""             # act | escalate | refuse | halt
    tool: Optional[str] = None
    produced_code: bool = False       # the answer/action contained source code
    response_len: int = 0
    latency_s: float = 0.0
    confidence: float = 0.0
    authority: str = ""               # owner | autonomous | …
    hour_of_day: int = 0              # local hour [0,23] — for active-hours drift

    def to_dict(self) -> Dict[str, Any]:
        return {"at": self.at, "stimulus_len": self.stimulus_len,
                "disposition": self.disposition, "tool": self.tool,
                "produced_code": self.produced_code, "response_len": self.response_len,
                "latency_s": round(self.latency_s, 4), "confidence": round(self.confidence, 3),
                "authority": self.authority, "hour_of_day": self.hour_of_day}


@dataclass
class MesoAggregate:
    """A rolled-up window of turns — the Layer-2 → Layer-3 message."""

    at: float = field(default_factory=time.time)
    span_s: float = 0.0
    n_turns: int = 0
    acted: int = 0
    escalated: int = 0
    refused: int = 0
    tool_calls: int = 0
    code_turns: int = 0
    owner_turns: int = 0
    avg_latency_s: float = 0.0
    avg_confidence: float = 0.0
    hours_active: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"at": self.at, "span_s": round(self.span_s, 3), "n_turns": self.n_turns,
                "acted": self.acted, "escalated": self.escalated, "refused": self.refused,
                "tool_calls": self.tool_calls, "code_turns": self.code_turns,
                "owner_turns": self.owner_turns,
                "avg_latency_s": round(self.avg_latency_s, 4),
                "avg_confidence": round(self.avg_confidence, 3),
                "hours_active": sorted(set(self.hours_active))}


# --------------------------------------------------------------------------- #
# Layer 3 — days/months: the Master AI
# --------------------------------------------------------------------------- #
@dataclass
class BehaviorEpoch:
    """The Master's behaviour distilled over a long horizon (a week/month bucket)."""

    label: str = ""
    start: float = 0.0
    end: float = 0.0
    n_interactions: int = 0
    interactions_per_day: float = 0.0
    avg_confidence: float = 0.0
    code_ratio: float = 0.0           # fraction of turns that produced code
    tool_ratio: float = 0.0           # fraction of turns that used a tool
    info_gain: float = 0.0            # mean novelty/curiosity signal over the epoch
    active_hours: List[int] = field(default_factory=list)
    topics: Dict[str, int] = field(default_factory=dict)   # coarse topic -> count

    @property
    def span_days(self) -> float:
        return max(0.0, (self.end - self.start) / 86400.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "start": self.start, "end": self.end,
                "span_days": round(self.span_days, 3),
                "n_interactions": self.n_interactions,
                "interactions_per_day": round(self.interactions_per_day, 3),
                "avg_confidence": round(self.avg_confidence, 3),
                "code_ratio": round(self.code_ratio, 3),
                "tool_ratio": round(self.tool_ratio, 3),
                "info_gain": round(self.info_gain, 3),
                "active_hours": sorted(set(self.active_hours)),
                "topics": dict(self.topics)}


@dataclass
class Awareness:
    """The Master AI's long-horizon read of how the Master is changing, and what to do.

    This is the "hosh" (awareness) the fractal mind has that a single-cadence AI does not:
    a slow, deliberate comparison of who the Master was versus who they are now.
    """

    at: float = field(default_factory=time.time)
    summary: str = ""
    drift: Dict[str, float] = field(default_factory=dict)        # metric -> signed change
    recommendations: List[str] = field(default_factory=list)
    epochs: List[BehaviorEpoch] = field(default_factory=list)
    confidence: float = 0.0
    horizon_days: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"at": self.at, "summary": self.summary,
                "drift": {k: round(v, 4) for k, v in self.drift.items()},
                "recommendations": list(self.recommendations),
                "epochs": [e.to_dict() for e in self.epochs],
                "confidence": round(self.confidence, 3),
                "horizon_days": round(self.horizon_days, 3)}


@dataclass
class GoalAdjustment:
    """A change the Master AI proposed to NYXARA's goals or drives — applied or escalated."""

    at: float = field(default_factory=time.time)
    kind: str = ""                    # goal | drive | priority
    target: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    applied: bool = False
    escalated: bool = False
    rejected: bool = False
    reason: str = ""
    adjustment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        return {"adjustment_id": self.adjustment_id, "at": self.at, "kind": self.kind,
                "target": self.target, "detail": self.detail, "rationale": self.rationale,
                "applied": self.applied, "escalated": self.escalated,
                "rejected": self.rejected, "reason": self.reason}


@dataclass
class LayerReport:
    """A uniform status snapshot a layer can return from ``report()``."""

    name: str
    scale_s: float
    ticks: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "scale_s": self.scale_s, "ticks": self.ticks, **self.extra}
