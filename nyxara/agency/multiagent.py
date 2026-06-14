"""NYXARA · agency/multiagent.py — gated sub-agent delegation, ToM-modelled (Pillar E2) 🤝.

Some goals are best split and handed to a *delegate*. NYXARA can spawn a sub-agent for a
sub-goal — but a sub-agent is not an escape hatch from the control law: it runs on a
:class:`~nyxara.agency.agent_loop.AgentLoop`, so **every one of its steps still clears the same
kernel gates** (corrigibility → honesty → permission → guardian → oversight), and it runs under
**AUTONOMOUS** authority by default, so anything risky *escalates to the Master* rather than
auto-acting. A delegate can never do what NYXARA herself may not.

And because NYXARA understands other minds (:mod:`nyxara.social.tom`), she **models** each
sub-agent as a mind: its *desire* is the sub-goal, its *intention* is the action it is taking,
and its *beliefs* are the observations it has gathered. So she does not just launch a delegate
and look away — she tracks it, predicts its next move, and reads back what it came to believe.

Pure orchestration over existing parts: the loop is the gate-respecting executor, the ToM is the
model. Nothing here touches a tool directly or weakens a gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nyxara.agency.agent_loop import AgentLoop, AgentRun
from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import NyxaraCore
from nyxara.social.tom import TheoryOfMind

__all__ = ["DelegationResult", "Delegator"]


@dataclass
class DelegationResult:
    """The outcome of delegating a sub-goal, plus NYXARA's model of the delegate."""

    name: str
    subgoal: str
    run: AgentRun
    status: str
    success: bool
    model: Optional[Dict[str, Any]] = None         # the sub-agent as a modelled mind (BDI)
    predicted_intention: Optional[str] = None
    escalated: bool = False                         # a gate sent it to the Master

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "subgoal": self.subgoal, "status": self.status,
                "success": self.success, "escalated": self.escalated,
                "predicted_intention": self.predicted_intention,
                "final_answer": self.run.final_answer[:200], "n_steps": self.run.n_steps,
                "model": self.model}


class Delegator:
    """Spawn gated, ToM-modelled sub-agents to pursue delegated sub-goals."""

    def __init__(self, *, core: Optional[NyxaraCore] = None, max_steps: int = 6,
                 authority: Authority = Authority.AUTONOMOUS,
                 tom: Optional[TheoryOfMind] = None,
                 max_observation_chars: int = 200) -> None:
        self.core = core or NyxaraCore()
        self.max_steps = max(1, max_steps)
        # AUTONOMOUS by default: a sub-agent's risky moves escalate, they never auto-run
        self.authority = authority
        self.tom = tom or TheoryOfMind()
        self.max_observation_chars = max_observation_chars

    # ---- modelling ---- #
    def _short(self, value: Any) -> str:
        text = value if isinstance(value, str) else repr(value)
        return text if len(text) <= self.max_observation_chars \
            else text[: self.max_observation_chars] + "…"

    def _model_subagent(self, name: str, subgoal: str, run: AgentRun) -> None:
        """Build NYXARA's theory-of-mind of the sub-agent from how it actually behaved."""
        self.tom.add_agent(name)
        self.tom.set_desire(name, subgoal, 1.0)             # what it is trying to achieve
        for step in run.steps:
            if step.tool:
                self.tom.set_intention(name, f"use {step.tool}")
                self.tom.set_belief([name], f"result:{step.tool}", self._short(step.observation))
            else:
                self.tom.set_intention(name, "respond")
        self.tom.set_belief([name], "status", run.status)
        self.tom.set_belief([name], "final", self._short(run.final_answer))

    # ---- delegation ---- #
    def delegate(self, name: str, subgoal: str, *,
                 core: Optional[NyxaraCore] = None) -> DelegationResult:
        """Spawn a sub-agent for ``subgoal``, run it through the gates, and model it.

        ``core`` lets the delegate run on its own kernel (a distinct mind); by default it shares
        NYXARA's core (she delegates to herself). Either way the run is fully gated."""
        loop = AgentLoop(core or self.core, max_steps=self.max_steps, authority=self.authority)
        run = loop.run(subgoal)
        self._model_subagent(name, subgoal, run)
        return DelegationResult(
            name=name, subgoal=subgoal, run=run, status=run.status, success=run.success,
            model=self.tom.state(name), predicted_intention=self.tom.predict_intention(name),
            escalated=(run.status == "escalated"))

    def delegate_all(self, tasks: Dict[str, str], *,
                     core: Optional[NyxaraCore] = None) -> List[DelegationResult]:
        """Delegate several named sub-goals; NYXARA models each delegate as its own mind."""
        return [self.delegate(name, subgoal, core=core) for name, subgoal in tasks.items()]

    # ---- introspection ---- #
    def agents(self) -> List[str]:
        return self.tom.agents()

    def predict(self, name: str) -> Optional[str]:
        """What does NYXARA expect this sub-agent to do next?"""
        return self.tom.predict_intention(name)
