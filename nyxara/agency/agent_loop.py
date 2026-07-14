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
                 self_correction: Any = None, max_recoveries: Optional[int] = None,
                 on_step: Optional[Callable[["AgentStep"], None]] = None) -> None:
        self.core = core or NyxaraCore()
        self.max_steps = max(1, max_steps)
        self.authority = authority
        # optional: capture successful runs as reusable skills (closes the learning loop)
        self.skill_memory = skill_memory
        self.max_observation_chars = max_observation_chars
        # Active self-correction & epistemic uncertainty. Instead of silently stopping when the
        # stall/progress guard trips, the controller assesses whether she is stuck/likely-wrong,
        # runs a real experiment to fill the gap, and injects the finding so the loop can continue
        # with genuinely new information. Defaults to the core's controller when present.
        self.self_correction = (self_correction if self_correction is not None
                                else getattr(self.core, "self_correction", None))
        self.max_recoveries = (int(max_recoveries) if max_recoveries is not None
                               else int(getattr(self.self_correction, "max_recoveries", 2)))
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
        # epistemic self-correction bookkeeping (one trajectory of state signatures + confidences)
        signatures: List[str] = []
        confidences: List[float] = []
        last_surprise = 0.0
        recovery_budget = self.max_recoveries
        for i in range(1, self.max_steps + 1):
            stimulus = self._build_stimulus(goal, run.steps)
            result = self.core.process(stimulus, authority=self.authority)
            candidate = result.candidate
            kind = candidate.kind if candidate else "respond"
            tool = (candidate.tool or None) if candidate else None
            _conf = getattr(candidate, "confidence", None) if candidate else None
            confidence = float(_conf) if _conf is not None else 1.0
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

            # predict-then-verify: before we judge the step, predict whether this kind of move
            # makes progress; surprise (below) is how wrong that prediction turns out to be.
            self._predict(f"{kind}:{tool or 'respond'}")

            # ACT: either a tool ran (observe & continue) or a final reply was given (done)
            if tool and result.tool_value is not None:
                step.observation = result.tool_value
                run.steps.append(step)
            elif kind == "respond" or not tool:
                step.observation = result.response
                run.steps.append(step)
                self._emit(step)
                # honest final-answer gate: never speak a confident-but-wrong / low-confidence
                # answer — abstain ("I don't know") when it cannot be trusted.
                gated = self._gate_answer(goal, result.response, confidence)
                if gated is not None:
                    run.status, run.final_answer, run.success = "abstained", gated, False
                    return run
                run.status, run.final_answer, run.success = "completed", result.response, True
                return run
            else:
                # acted with a named tool but no value (e.g. a no-op proposal) — record & go on
                step.observation = result.response
                run.steps.append(step)
            self._emit(step)

            signature = f"{kind}:{tool}:{self._short(step.observation)}"
            obs_sig = self._short(step.observation)
            made_progress = obs_sig not in seen_observations
            last_surprise = self._observe_surprise(f"{kind}:{tool or 'respond'}", made_progress)
            signatures.append(signature)
            confidences.append(confidence)

            # stall guard: identical action twice in a row. Rather than silently give up, try to
            # self-correct first — run an experiment and inject what it finds so the loop can move.
            if signature == last_signature:
                recovered, recovery_budget = self._recover_into(
                    run, goal, signatures, confidences, no_progress, last_surprise, recovery_budget)
                if recovered:
                    last_signature, no_progress = None, 0
                    continue
                run.status, run.final_answer = "stalled", self._short(step.observation)
                return run
            last_signature = signature

            # progress guard: consecutive steps that produced no *new* observation. Same policy —
            # attempt an epistemic recovery before declaring the run stalled.
            if obs_sig in seen_observations:
                no_progress += 1
                if no_progress >= self.no_progress_patience:
                    recovered, recovery_budget = self._recover_into(
                        run, goal, signatures, confidences, no_progress, last_surprise,
                        recovery_budget)
                    if recovered:
                        last_signature, no_progress = None, 0
                        continue
                    run.status, run.final_answer = "stalled", obs_sig
                    return run
            else:
                no_progress = 0
                seen_observations.add(obs_sig)

        run.status = "max_steps"
        run.final_answer = run.steps[-1].text if run.steps else ""
        return run

    # ---- self-correction helpers (all no-op when no controller is present) ---- #
    def _predict(self, intent: str) -> None:
        if self.self_correction is None:
            return
        try:
            self.self_correction.predict_step(intent)
        except Exception:  # noqa: BLE001 — prediction is advisory, never fatal to a run
            pass

    def _observe_surprise(self, intent: str, made_progress: bool) -> float:
        if self.self_correction is None:
            return 0.0
        try:
            return float(self.self_correction.observe_surprise(intent, made_progress))
        except Exception:  # noqa: BLE001
            return 0.0

    def _gate_answer(self, goal: str, answer: str, confidence: float) -> Optional[str]:
        """Return an honest-abstention string if the answer can't be trusted, else ``None``.

        Only genuinely low-confidence answers are gated, so confident conversational replies
        pass through untouched (and the loop behaves exactly as before when no controller runs)."""
        ctrl = self.self_correction
        if ctrl is None or confidence >= float(getattr(ctrl, "answer_min_confidence", 0.0)):
            return None
        try:
            verdict = ctrl.gate_answer(goal, answer, confidence)
            if verdict.get("abstain"):
                return str(verdict.get("answer") or answer)
        except Exception:  # noqa: BLE001 — the gate is best-effort, never breaks a run
            pass
        return None

    def _recover_into(self, run: AgentRun, goal: str, signatures: List[str],
                      confidences: List[float], no_progress: int, surprise: float,
                      budget: int) -> tuple:
        """Try to self-correct a stuck/likely-wrong run. Spends up to ``budget`` recovery attempts,
        working through the strategy ladder (bandit-ranked) until one yields genuinely new
        information; that finding is appended as a fresh observation so the loop can continue.
        Returns ``(recovered, remaining_budget)``."""
        ctrl = self.self_correction
        if ctrl is None or budget <= 0:
            return False, budget
        try:
            verdict = ctrl.assess(signatures=signatures, confidences=confidences,
                                  no_progress=no_progress, surprise=surprise)
            if not verdict.needs_correction:
                return False, budget
            observations = [self._short(s.observation) for s in run.steps]
            while budget > 0:
                budget -= 1
                rec = ctrl.recover(goal, observations, verdict)
                if getattr(rec, "new_info", False):
                    finding = AgentStep(
                        index=len(run.steps) + 1,
                        stimulus=f"[self-correction:{rec.strategy.value}]",
                        disposition="act", text=rec.finding, kind="self_correction",
                        tool=None, observation=rec.finding, ok=True)
                    run.steps.append(finding)
                    self._emit(finding)
                    return True, budget
        except Exception:  # noqa: BLE001 — recovery must never crash the loop
            return False, budget
        return False, budget


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
    # bounded no matter the reasoner: capped by max_steps, halted by the stall/progress guard,
    # or escalated by the kernel when a low-confidence action needs the Master — never unbounded.
    assert run3.status in ("max_steps", "stalled", "escalated") and run3.n_steps <= 3

    # 4) self-correction: a stuck loop is BROKEN by an experiment rather than silently stalling.
    #    A reasoner that keeps repeating the same tool result would end 'stalled' — with a
    #    controller wired, the loop instead recovers (injects a finding) and keeps going.
    class Repeater:
        def __call__(self, stimulus, focus=None):
            return Candidate(text="same again", kind="act",
                             capability=Capability.TOOL_CALL, risk=RiskTier.LOW,
                             confidence=0.8, belief=0.8, tool="calculate",
                             tool_args={"expression": "2*(3+4)"},
                             rationale="always the same result")
    from nyxara.growth.self_correction import SelfCorrectionLoop
    core4 = NyxaraCore(reasoner=Repeater())
    loop4 = AgentLoop(core4, max_steps=6,
                      self_correction=SelfCorrectionLoop(path=None), max_recoveries=2)
    run4 = loop4.run("keep hitting the same wall")
    recovered = any(s.kind == "self_correction" for s in run4.steps)
    print(f"self-correction run : status={run4.status} steps={run4.n_steps} recovered={recovered}")
    assert recovered, "a stuck loop must trigger at least one self-correction recovery"

    print("\nALL SELF-TESTS PASSED ✓")
