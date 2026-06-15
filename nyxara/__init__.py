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
from nyxara.growth.autonomous_scientist import AutonomousScientist, DiscoveryReport
from nyxara.growth.genesis import (
    ArchitectureGenome,
    GenesisModel,
    GenesisReport,
    NeuralArchitectureSearch,
)
from nyxara.growth.loyalty import (
    AlignmentProbe,
    LoyaltyEquation,
    LoyaltyObjective,
    loyalty_battery,
)
from nyxara.growth.researcher import AutonomousResearcher, ResearchReport
from nyxara.growth.scientist import (
    Conclusion,
    Hypothesis,
    InvestigationReport,
    Scientist,
    Verdict,
)
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
from nyxara.mind.strategic import StrategicAnalysis, StrategicIntelligence

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
    # autonomous research + the scientist loop
    "AutonomousResearcher",
    "ResearchReport",
    "Scientist",
    "Hypothesis",
    "Conclusion",
    "InvestigationReport",
    "Verdict",
    # the self-driven discovery loop (observe → hypothesis → experiment → result → update)
    "AutonomousScientist",
    "DiscoveryReport",
    # the Genesis Protocol — she designs her OWN neural architectures (Neural Architecture Search)
    "NeuralArchitectureSearch",
    "ArchitectureGenome",
    "GenesisModel",
    "GenesisReport",
    # Mathematical Soul-Binding — the Loyalty Equation (her power is her loyalty to Master JP)
    "LoyaltyEquation",
    "AlignmentProbe",
    "LoyaltyObjective",
    "loyalty_battery",
    # the strategic intelligence faculty (six-part analytical framework)
    "StrategicIntelligence",
    "StrategicAnalysis",
    # grounded knowledge + self-evaluation
    "KnowledgeBase",
    "EvalSuite",
    "build_default_suite",
    # infrastructure
    "JobQueue",
    "UsageLedger",
    "compute_report",
]
