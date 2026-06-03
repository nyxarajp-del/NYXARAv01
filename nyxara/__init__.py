"""Nyxara — kernel-sovereign cognitive agent.

The kernel is sovereign: the LLM proposes, the kernel disposes; verifiable beats
probabilistic. Importing this package exposes only the stable cross-cutting contracts;
heavy subsystems are imported from their own sub-packages on demand.
"""
from __future__ import annotations

__version__ = "0.1.0"
__owner__ = "Jaypal Khoja (JP)"

from nyxara.contracts import (
    Belief,
    Event,
    Goal,
    Percept,
    Priority,
    Proposal,
    Provenance,
    Verdict,
)

__all__ = [
    "__version__",
    "__owner__",
    "Belief",
    "Event",
    "Goal",
    "Percept",
    "Priority",
    "Proposal",
    "Provenance",
    "Verdict",
]
