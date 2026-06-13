"""NYXARA · growth — learning, reflection, and self-directed inquiry.

Public surface for the autonomous research faculties:

* :class:`~nyxara.growth.researcher.AutonomousResearcher` — search → read →
  summarise → store: gather and ground knowledge from sources.
* :class:`~nyxara.growth.scientist.Scientist` — the scientific method: form a
  falsifiable hypothesis, design and run a safe experiment, compare the result to
  the prediction, and draw a calibrated conclusion.
"""

from __future__ import annotations

from nyxara.growth.researcher import AutonomousResearcher, ResearchReport
from nyxara.growth.scientist import (
    Conclusion,
    Experiment,
    ExperimentKind,
    Hypothesis,
    InvestigationReport,
    Observation,
    Scientist,
    Verdict,
)

__all__ = [
    "AutonomousResearcher",
    "ResearchReport",
    "Scientist",
    "Hypothesis",
    "Experiment",
    "Observation",
    "Conclusion",
    "InvestigationReport",
    "Verdict",
    "ExperimentKind",
]
