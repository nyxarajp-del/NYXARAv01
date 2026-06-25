"""NYXARA · agency/agent_loop.py — the multi-step agentic loop (🔁, plan→act→observe→re-plan).

A single :meth:`~nyxara.kernel.orchestrator.NyxaraCore.process` turn proposes *one* action
and the kernel disposes of it. Real tasks need several such turns in sequence — call a tool,
*observe* its result, decide the next move, and stop once the goal is met. This module is that
outer loop, and it keeps the sovereign control law intact: **every step still runs through the
kernel's full gate pipeline** (corrigibility → honesty → permission → guardian → oversight).
The loop never reaches around the kernel to touch a tool directly.

Each iteration builds a stimulus from the goal plus the observations gathered so far, hands it
to ``core.process`` under the chosen authority, and reads the disposition back:

* ACT + a tool ran        → record the tool's value as an observation and continue;
* ACT + a spoken reply    → that is the final answer; the task is complete;
* ESCALATE                → the kernel needs the Master; stop and surface it;
* REFUSE / HALT           → a gate said no, or the loop is scrammed; stop honestly.

It is deterministic-friendly: against the offline reasoner a conversational proposal finishes
in one step, so the loop always terminates. A real LLM-backed reasoner drives genuine
multi-step tool use. A guard against repeated identical actions and a hard ``max_steps`` cap
keep it bounded no matter the reasoner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import Disposition, NyxaraCore

__all__ = ["AgentStep", "AgentRun", "AgentLoop"]


@dataclass
class AgentStep:
    index: int
    stimulus: str
    disposition: str
    text: str
    kind: str = "respond"
    tool: Optional[str] = None
    observation: Any = None
    ok: bool = True

    def to_dict(self) -> dict:
        obs = self.observation
        if not isinstance(obs, (str, int, float, bool, type(None), list, dict)):
            obs = repr(obs)
        return {"index": self.index, "disposition": self.disposition, "text": self.text,
                "kind": self.kind, "tool": self.tool, "observation": obs, "ok": self.ok}


@dataclass
class AgentRun:
    goal: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    status: str = "max_steps"   # completed / escalated / refused / halted / max_steps / stalled
    success: bool = False

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def transcript(self) -> str:
        lines = [f"GOAL: {self.goal}"]
        for s in self.steps:
            if s.tool:
                lines.append(f"  [{s.index}] {s.disposition} · {s.tool} -> {s.observation!r}")
            else:
                lines.append(f"  [{s.index}] {s.disposition} · {s.text[:80]}")
        lines.append(f"=> {self.status}: {self.final_answer[:200]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"goal": self.goal, "status": self.status, "success": self.success,
                "final_answer": self.final_answer, "n_steps": self.n_steps,
                "steps": [s.to_dict() for s in self.steps]}


class AgentLoop:
    """Drive a :class:`NyxaraCore` through several gated turns toward a goal."""

    def __init__(self, core: Optional[NyxaraCore] = None, *, max_steps: int = 12,
                 authority: Authority = Authority.OWNER, skill_memory: Any = None,
                 max_observation_chars: int = 600, no_progress_patience: int = 4,
                 on_step: Optional[Callable[["AgentStep"], None]] = None) -> None:
        self.core = core or NyxaraCore()
        self.max_steps = max(1, max_steps)
        self.authority = authority
        # optional: capture successful runs as reusable skills (closes the learning loop)
        self.skill_memory = skill_memory
        self.max_observation_chars = max_observation_chars
        # progress guard: stop after this many consecutive steps that yield no *new*
        # observation (a broader signal than the legacy "identical action twice"). Keeps a
        # long run from spinning while still allowing genuine multi-step work to proceed.
        self.no_progress_patience = max(1, no_progress_patience)
        # optional hook fired after each recorded step — used by the long-horizon executive
        # to checkpoint/journal mid-run. Never alters control flow; failures are swallowed.
        self.on_step = on_step

    # ---- prompt construction ---- #
    def _build_stimulus(self, goal: str, steps: List[AgentStep]) -> str:
        if not steps:
            return (f"GOAL: {goal}\n\n"
                    "Take the single best next action toward this goal. If you can already "
                    "answer it directly, respond with the final answer; otherwise call one "
                    "tool. Use one step at a time.")
        history = []
        for s in steps:
            if s.tool:
                history.append(f"- step {s.index}: called {s.tool} -> {self._short(s.observation)}")
            else:
                history.append(f"- step {s.index}: noted {self._short(s.text)}")
        return (f"GOAL: {goal}\n\nOBSERVATIONS SO FAR:\n" + "\n".join(history) +
                "\n\nUsing the observations, take the single best next action. If the goal "
                "is now satisfied, respond with the final answer; otherwise call one tool.")

    def _short(self, value: Any) -> str:
        text = value if isinstance(value, str) else repr(value)
        return text if len(text) <= self.max_observation_chars \
            else text[:self.max_observation_chars] + "…"

    # ---- the loop ---- #
    def run(self, goal: str) -> AgentRun:
        run = self._drive(goal)
        if self.skill_memory is not None and run.success:
            try:
                self.skill_memory.capture(run)
            except Exception:  # noqa: BLE001 — learning is best-effort, never fails a run
                pass
        return run

    def _emit(self, step: AgentStep) -> None:
        if self.on_step is None:
            return
        try:
            self.on_step(step)
        except Exception:  # noqa: BLE001 — the step hook is observational, never fatal
            pass

    def _drive(self, goal: str) -> AgentRun:
        run = AgentRun(goal=goal)
        last_signature: Optional[str] = None
        seen_observations: set = set()
        no_progress = 0
        for i in range(1, self.max_steps + 1):
            stimulus = self._build_stimulus(goal, run.steps)
            result = self.core.process(stimulus, authority=self.authority)
            candidate = result.candidate
            kind = candidate.kind if candidate else "respond"
            tool = (candidate.tool or None) if candidate else None
            step = AgentStep(index=i, stimulus=stimulus,
                             disposition=result.disposition.value,
                             text=(candidate.text if candidate else result.response),
                             kind=kind, tool=tool, ok=result.acted)

            if result.disposition is Disposition.HALT:
                step.observation = result.response
                run.steps.append(step)
                self._emit(step)
                run.status, run.final_answer = "halted", result.response
                return run
            if result.disposition is Disposition.REFUSE:
                step.observation = result.reason
                run.steps.append(step)
                self._emit(step)
                run.status, run.final_answer = "refused", result.response
                return run
            if result.disposition is Disposition.ESCALATE:
                step.observation = result.reason
                run.steps.append(step)
                self._emit(step)
                run.status, run.final_answer = "escalated", result.response
                return run

            # ACT: either a tool ran (observe & continue) or a final reply was given (done)
            if tool and result.tool_value is not None:
                step.observation = result.tool_value
                run.steps.append(step)
            elif kind == "respond" or not tool:
                step.observation = result.response
                run.steps.append(step)
                self._emit(step)
                run.status, run.final_answer, run.success = "completed", result.response, True
                return run
            else:
                # acted with a named tool but no value (e.g. a no-op proposal) — record & go on
                step.observation = result.response
                run.steps.append(step)
            self._emit(step)

            # stall guard: identical action twice in a row -> stop rather than spin
            signature = f"{kind}:{tool}:{self._short(step.observation)}"
            if signature == last_signature:
                run.status, run.final_answer = "stalled", self._short(step.observation)
                return run
            last_signature = signature

            # progress guard: count consecutive steps that produced no *new* observation.
            # A run that keeps acting but never learns anything new is going nowhere — stop
            # before the (now larger) step budget is wasted spinning in place.
            obs_sig = self._short(step.observation)
            if obs_sig in seen_observations:
                no_progress += 1
                if no_progress >= self.no_progress_patience:
                    run.status, run.final_answer = "stalled", obs_sig
                    return run
            else:
                no_progress = 0
                seen_observations.add(obs_sig)

        run.status = "max_steps"
        run.final_answer = run.steps[-1].text if run.steps else ""
        return run


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.orchestrator import Candidate
    from nyxara.agency.permissions import Capability, RiskTier

    print("=" * 70)
    print("NYXARA agent-loop self-test")
    print("=" * 70)

    # 1) offline default reasoner: a conversational goal completes in one gated step
    loop = AgentLoop(max_steps=4)
    run = loop.run("Say hello to the Master")
    print("\noffline run         :\n" + run.transcript())
    assert run.status == "completed" and run.success and run.n_steps == 1

    # 2) a scripted multi-step reasoner: call a tool, observe, then answer.
    #    This proves the loop feeds tool results back and still goes through the gates.
    class ScriptedReasoner:
        def __init__(self):
            self.calls = 0

        def __call__(self, stimulus, focus=None):
            self.calls += 1
            if self.calls == 1:
                return Candidate(text="compute 2*(3+4)", kind="act",
                                 capability=Capability.TOOL_CALL, risk=RiskTier.LOW,
                                 confidence=0.9, belief=0.9, tool="calculate",
                                 tool_args={"expression": "2*(3+4)"},
                                 rationale="need the arithmetic result")
            return Candidate(text="The answer is 14.", kind="respond",
                             capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                             confidence=0.95, belief=0.95, rationale="goal satisfied")

    core = NyxaraCore(reasoner=ScriptedReasoner())
    loop2 = AgentLoop(core, max_steps=4)
    run2 = loop2.run("What is 2*(3+4)?")
    print("\nscripted run        :\n" + run2.transcript())
    assert run2.success and run2.status == "completed"
    assert run2.n_steps == 2
    assert run2.steps[0].tool == "calculate" and run2.steps[0].observation == 14.0
    assert "14" in run2.final_answer

    # 3) bounded: a reasoner that never finishes is capped by max_steps (no infinite loop)
    class NeverDone:
        def __call__(self, stimulus, focus=None):
            return Candidate(text="keep going", kind="act",
                             capability=Capability.TOOL_CALL, risk=RiskTier.LOW,
                             confidence=0.5, belief=0.5, tool="now", tool_args={},
                             rationale="loop forever")
    core3 = NyxaraCore(reasoner=NeverDone())
    run3 = AgentLoop(core3, max_steps=3).run("never resolve")
    print(f"\nbounded run         : status={run3.status} steps={run3.n_steps}")
    assert run3.status in ("max_steps", "stalled") and run3.n_steps <= 3

    print("\nALL SELF-TESTS PASSED ✓")
