"""NYXARA · agency — independent action under the sovereign control law.

Public surface for the faculties that let NYXARA *act*, not merely answer — each still
bound by the kernel's gate pipeline (corrigibility → honesty → permission → guardian →
oversight):

* :class:`~nyxara.agency.agent_loop.AgentLoop` — the multi-step plan→act→observe→re-plan
  loop that carries a goal across several governed turns until it is met.
* :class:`~nyxara.agency.proactive.ProactiveEngine` — initiative: detect opportunities,
  threats and her own weaknesses and discipline each through the loyalty-first gauntlet, so
  she proposes (and, when cleared, takes) action without waiting to be told.
"""

from __future__ import annotations

from nyxara.agency.agent_loop import AgentLoop, AgentRun, AgentStep
from nyxara.agency.dynamic_tool_creator import (
    DynamicToolCreator,
    EphemeralResult,
    Language,
)
from nyxara.agency.mission import (
    Milestone,
    MilestoneStatus,
    Mission,
    MissionBudgets,
    MissionExecutive,
    MissionStatus,
)
from nyxara.agency.proactive import (
    Initiative,
    ProactiveEngine,
    ProposalDecision,
    TriggerKind,
    Verdict,
)

__all__ = [
    "AgentLoop",
    "AgentRun",
    "AgentStep",
    "Mission",
    "Milestone",
    "MissionExecutive",
    "MissionStatus",
    "MilestoneStatus",
    "MissionBudgets",
    "DynamicToolCreator",
    "EphemeralResult",
    "Language",
    "ProactiveEngine",
    "Initiative",
    "ProposalDecision",
    "TriggerKind",
    "Verdict",
]
