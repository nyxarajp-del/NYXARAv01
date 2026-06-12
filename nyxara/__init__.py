"""NYXARA — sovereign cognitive architecture. Owner: Jaypal Khoja (JP).

The control law: *the mind proposes; the kernel disposes; the Master is sovereign.*

The whole mind is wired in :class:`nyxara.kernel.orchestrator.NyxaraCore`. The
typical entry point is::

    from nyxara import NyxaraCore, Authority

    core = NyxaraCore()
    result = core.process("Hello NYXARA", authority=Authority.OWNER)
    print(result.response)

Or run the interactive console with ``python -m nyxara`` (see
:mod:`nyxara.__main__`).
"""

from __future__ import annotations

from nyxara.agency.agent_loop import AgentLoop, AgentRun
from nyxara.agency.permissions import Authority
from nyxara.eval import EvalSuite, build_default_suite
from nyxara.growth.skill_memory import SkillMemory
from nyxara.kernel.autonomic import AutonomicLoop
from nyxara.kernel.compute import compute_report
from nyxara.kernel.jobqueue import JobQueue
from nyxara.kernel.orchestrator import (
    Candidate,
    CycleResult,
    Disposition,
    NyxaraCore,
)
from nyxara.knowledge import KnowledgeBase
from nyxara.mind.cost import UsageLedger

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "NyxaraCore",
    "Authority",
    "Disposition",
    "Candidate",
    "CycleResult",
    "AutonomicLoop",
    # multi-step agency + experiential learning
    "AgentLoop",
    "AgentRun",
    "SkillMemory",
    # grounded knowledge + self-evaluation
    "KnowledgeBase",
    "EvalSuite",
    "build_default_suite",
    # infrastructure
    "JobQueue",
    "UsageLedger",
    "compute_report",
]
