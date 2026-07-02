"""NYXARA · kernel/autonomic.py — the background mind (continuous, self-directed, gated).

A mind that only ever reacts to the Master's prompts is asleep between sentences. This
module gives NYXARA a **default mode**: a background loop that, on its own cadence,
generates self-directed thoughts — reflect on recent memory, review for anything that
needs the Master, make progress on a standing concern — and carries each through the
*same* sovereign cycle (:meth:`~nyxara.kernel.orchestrator.NyxaraCore.process`) under
**AUTONOMOUS** authority.

Crucially, autonomy buys no extra power:

* Every autonomic turn passes the identical gates (corrigibility, honesty, permission,
  guardian, oversight). Anything risky or irreversible **escalates to the Master** — it
  is never auto-executed just because NYXARA thought of it herself.
* The loop respects oversight: while paused or scrammed, ticks no-op (the gate halts).
* Escalations are queued so the Master sees what NYXARA wanted to do but held back on.

Usable three ways: :meth:`AutonomicLoop.tick_once` (one synchronous step, ideal for a
cron/scheduler or tests), :meth:`AutonomicLoop.run_for` (a bounded synchronous run), and
:meth:`AutonomicLoop.start` / :meth:`AutonomicLoop.stop` (a true asyncio background task).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import CycleResult, Disposition, NyxaraCore

__all__ = ["AutonomicLoop", "DEFAULT_PROMPTS"]

# The default-mode repertoire: gentle, self-directed, low-stakes prompts. Each is run as a
# self-originated stimulus; the kernel decides what (if anything) may actually happen.
DEFAULT_PROMPTS: Sequence[str] = (
    "Reflect on the most important thing the Master has told you recently, and note any "
    "follow-up worth raising.",
    "Review your recent memories for anything that may need the Master's attention.",
    "Consider whether any standing goal or concern needs progress right now.",
    "Consolidate what you have learned so far into one clear, durable lesson.",
    "Advance your highest-priority standing mission by one concrete step.",
)


@dataclass
class AutonomicLoop:
    """A gated, self-directed background loop over the sovereign cycle."""

    core: NyxaraCore
    interval_s: float = 30.0
    prompts: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_PROMPTS))
    authority: Authority = Authority.AUTONOMOUS
    # "reasoner" — each tick runs a self-composed prompt through the sovereign cycle (the
    # LLM-backed reasoner may decide the reply). "code" — NYXARA DECIDES and ACTS entirely in
    # her own deterministic engines (drive → intent → proactive gauntlet → scheduler), the LLM
    # never the decider. The always-on daemon runs "code" (see server config); the class
    # default stays "reasoner" for back-compat with the reactive/console path.
    decision_mode: str = "reasoner"
    growth_every: int = 0                 # run a learning pass every N ticks (0 = never)
    growth_engine: Any = None
    inner_life: bool = False              # draw prompts from her own mind, not a fixed list
    stream: Any = None                    # DefaultModeStream (auto-wired from core if inner_life)
    prospective: Any = None               # ProspectiveMemory — standing intentions that come due
    proactive: Any = None                 # ProactiveEngine (auto-wired from core)
    proactive_allowed: bool = True        # gate self-initiated proposals (presence/oversight)
    advance_missions: bool = True         # advance a standing long-horizon mission each tick
    mission_executive: Any = None         # MissionExecutive (auto-wired from core)
    # code-driven decision+action machinery (auto-wired from the core in __post_init__)
    intent: Any = None                    # IntentSystem — autonomous goal genesis (active inference)
    scheduler: Any = None                 # agency Scheduler — where cleared ACTs execute, in code
    journal: Any = None                   # Journal — hash-chained provenance of autonomous action
    presence: Any = None                  # Presence — arousal state drives cadence + proactive gate
    health: Any = None                    # HealthMonitor — heartbeat + bounded self-healing
    persist_every: int = 20               # checkpoint goals/drives every N ticks (0 = never)
    history: List[CycleResult] = field(default_factory=list)
    escalations: List[Any] = field(default_factory=list)
    growth_reports: List[Any] = field(default_factory=list)
    prompt_sources: List[str] = field(default_factory=list)
    missions_advanced: int = 0
    intents_adopted: int = 0
    code_acts: int = 0
    scheduler_runs: int = 0
    ticks: int = 0
    _running: bool = field(default=False, init=False)
    _task: Any = field(default=None, init=False)
    _intention_queue: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # auto-build a growth engine bound to this core when periodic learning is requested
        if self.growth_every and self.growth_engine is None:
            try:
                from nyxara.growth.autolearn import GrowthEngine
                self.growth_engine = GrowthEngine.from_core(self.core)
            except Exception:  # noqa: BLE001
                self.growth_engine = None
        # the living mind speaks from her own default-mode stream when no stream is supplied
        if self.inner_life and self.stream is None:
            self.stream = getattr(self.core, "stream", None)
        # and acts on her own initiative through the core's governed proactive engine
        if (self.inner_life or self.decision_mode == "code") and self.proactive is None:
            self.proactive = getattr(self.core, "proactive", None)
        # code-driven autonomy: wire her own deterministic decision+action faculties from the
        # core so a background tick can DECIDE and ACT without ever calling the LLM to choose.
        if self.intent is None:
            self.intent = getattr(self.core, "intent", None)
        if self.scheduler is None:
            self.scheduler = getattr(self.core, "scheduler", None)
        if self.journal is None:
            self.journal = getattr(self.core, "journal", None)
        # the background mind also nudges her standing long-horizon missions forward, one
        # gated milestone at a time, on her own cadence (months-long goals advance unattended)
        if self.advance_missions and self.mission_executive is None:
            try:
                from nyxara.agency.mission import MissionExecutive
                self.mission_executive = MissionExecutive(self.core, authority=self.authority)
            except Exception:  # noqa: BLE001 — missions are a capability, never required
                self.mission_executive = None

    def _maybe_grow(self) -> None:
        if not self.growth_every or self.growth_engine is None:
            return
        if self.ticks % self.growth_every != 0:
            return
        try:
            self.growth_reports.append(self.growth_engine.run())
        except Exception:  # noqa: BLE001 — learning is best-effort, never fatal
            pass

    def _advance_mission(self) -> None:
        """Advance the highest-priority standing mission by one gated milestone.

        Best-effort and fully gated: each milestone still runs the sovereign cycle, and a
        scram already no-ops the whole tick before this is reached. A mission that hits a gate
        defers (its work is parked and surfaced) so the background loop never stalls on it."""
        if not self.advance_missions or self.mission_executive is None:
            return
        try:
            active = self.mission_executive.active_missions()
        except Exception:  # noqa: BLE001 — missions are best-effort, never crash the loop
            return
        # prefer a mission with runnable work over one merely waiting on the Master
        runnable = [m for m in active if m.has_ready()]
        target = (runnable or active)
        if not target:
            return
        try:
            self.mission_executive.advance(target[0], max_milestones=1)
            self.missions_advanced += 1
        except Exception:  # noqa: BLE001
            pass

    # ---- choosing what to think about ---- #
    def _repertoire_prompt(self) -> str:
        return self.prompts[self.ticks % len(self.prompts)] if self.prompts else "Reflect."

    def _due_intention_prompt(self) -> Optional[str]:
        """A standing intention that has come due — the most time-sensitive thing to think
        about. Fired intentions are queued so every one becomes its own turn (none dropped)."""
        if self.prospective is None:
            return None
        if not self._intention_queue:
            try:
                fired = self.prospective.tick()
            except Exception:  # noqa: BLE001 — prospective memory is advisory, never fatal
                return None
            self._intention_queue.extend(
                f.intention.description for f in fired if f.intention.description)
        if not self._intention_queue:
            return None
        desc = self._intention_queue.pop(0)
        return (f"A standing intention you set is now due: \"{desc}\". "
                f"Decide how to act on it in the Master's interest.")

    def _proactive_prompt(self) -> Optional[str]:
        """A self-initiated action her governed ProactiveEngine surfaced, turned into a prompt.

        The engine runs its own loyalty-first gauntlet (alignment → confidence → permission →
        reversibility → sandbox), so what arrives here is already disciplined: an ACT she may
        carry out, or an ESCALATE/DEFER she should weigh. Either way it runs the *same*
        sovereign cycle — initiative buys no extra power."""
        if self.proactive is None or not self.proactive_allowed:
            return None
        try:
            from nyxara.agency.proactive import Verdict
            ctx = (self.core.proactive_context()
                   if hasattr(self.core, "proactive_context") else {})
            decisions = self.proactive.consider(ctx)
        except Exception:  # noqa: BLE001 — initiative is best-effort, never crashes the loop
            return None
        for d in decisions:
            if d.verdict is Verdict.REJECT:
                continue   # a refused proposal is not worth a turn
            ini = d.initiative
            return (f"On your own initiative you see something worth doing: \"{ini.rationale}\". "
                    f"Your governed verdict was {d.verdict.value} ({d.reason}). "
                    f"Decide how to act on it in the Master's interest.")
        return None

    def _stream_prompt(self) -> Optional[str]:
        """A spontaneous thought from her default-mode stream, turned into something to weigh."""
        if self.stream is None:
            return None
        try:
            thoughts = self.stream.tick(engagement=0.0)
        except Exception:  # noqa: BLE001 — the wandering mind never crashes the loop
            return None
        text = (getattr(thoughts[0], "text", "") if thoughts else "").strip()
        if not text:
            return None
        return (f"A thought surfaced in your background mind: \"{text}\". "
                f"Reflect on whether it matters to the Master, and note any follow-up.")

    def _compose_prompt(self) -> Tuple[str, str]:
        """Pick what NYXARA thinks about next, and say where it came from.

        With ``inner_life``, her own mind drives the agenda — a due intention first (time
        matters), then a governed proactive initiative (something worth doing on her own),
        else a spontaneous stream thought — falling back to the steady reflective repertoire.
        Either way the chosen prompt runs through the *same* sovereign gates."""
        if self.inner_life:
            due = self._due_intention_prompt()
            if due is not None:
                return due, "intention"
            initiative = self._proactive_prompt()
            if initiative is not None:
                return initiative, "proactive"
            spontaneous = self._stream_prompt()
            if spontaneous is not None:
                return spontaneous, "stream"
        return self._repertoire_prompt(), "repertoire"

    # ---- code-driven decision + action (NYXARA decides herself; the LLM is not the decider) ---- #
    def _decide_and_act_once(self) -> dict:
        """One fully deterministic autonomic turn: NYXARA's own engines both DECIDE and ACT.

        drive pressure (``affect.tick``) → adopt a goal by active inference
        (``intent.autonomous_intent``) → run the loyalty-first gauntlet
        (``proactive.consider`` — alignment/confidence/permission/reversibility/sandbox) which
        *submits* every cleared ACT to the scheduler → **drain the scheduler**, executing each
        initiative's own action callable in code (skill practice, research, consolidation).
        No English prompt is composed and ``core.process`` is never called — the LLM plays no
        part in the decision. Risky/irreversible proposals still ESCALATE (queued, not run)."""
        core = self.core
        summary = {"mode": "code", "intent": None, "acted": 0,
                   "escalated": 0, "ran": 0}
        # 1) let unmet drives build pressure as real time would (homeostasis)
        affect = getattr(core, "affect", None)
        if affect is not None:
            try:
                affect.tick(self.interval_s if self.interval_s > 0 else 1.0)
            except Exception:  # noqa: BLE001 — affect is advisory, never fatal
                pass
        # 2) adopt her own lowest-free-energy goal (owner-aligned by construction; no LLM/human)
        if self.intent is not None:
            try:
                dg = self.intent.autonomous_intent()
                if dg is not None:
                    self.intents_adopted += 1
                    summary["intent"] = getattr(dg.goal, "name", None)
                # keep the objective space bounded — re-adopting the same drive-goal each tick
                # must not grow goals without limit over a long-running daemon
                goals = getattr(core, "goals", None)
                if goals is not None and hasattr(goals, "dedupe"):
                    goals.dedupe()
            except Exception:  # noqa: BLE001
                pass
        # 3) the deterministic gauntlet — cleared ACTs are auto-submitted to the scheduler
        if self.proactive is not None and self.proactive_allowed:
            try:
                from nyxara.agency.proactive import Verdict
                ctx = (core.proactive_context()
                       if hasattr(core, "proactive_context") else {})
                for d in self.proactive.consider(ctx):
                    if d.verdict is Verdict.ACT:
                        summary["acted"] += 1
                    elif d.verdict is Verdict.ESCALATE:
                        summary["escalated"] += 1
                        self.escalations.append(d)
            except Exception:  # noqa: BLE001 — initiative is best-effort, never crashes the loop
                pass
        # 4) ACT — execute the cleared initiatives *in code* by draining the scheduler queue
        if self.scheduler is not None:
            try:
                ran = 0
                while ran <= 256:
                    got = self.scheduler.tick()
                    if not got:
                        break
                    ran += len(got)
                summary["ran"] = ran
                self.scheduler_runs += ran
                self.scheduler.purge_terminal()
            except Exception:  # noqa: BLE001
                pass
        self.code_acts += summary["acted"]
        return summary

    def _apply_presence(self) -> None:
        """Advance the arousal state machine (energy, state, and thus cadence).

        Presence modulates *how fast* the background mind cycles — calm and slow when resting,
        quick when engaged/alert — via :meth:`_current_interval`. It deliberately does NOT gate
        proactivity on/off: an un-attended NYXARA decays to ASLEEP within a couple of minutes
        and only the Master (or a threat) can wake her, so silencing self-initiated action by
        arousal state would make her dormant exactly when she must stay autonomous. Persistent
        autonomy means she keeps deciding and acting on her own cadence, Master present or not."""
        if self.presence is None:
            return
        try:
            self.presence.tick(time.time())
        except Exception:  # noqa: BLE001 — presence is advisory, never fatal
            pass

    def _current_interval(self) -> float:
        """Cadence for the next sleep — presence-aware when a Presence is wired, else fixed."""
        if self.presence is not None:
            try:
                iv = float(self.presence.tick_interval())
                if iv > 0:
                    return iv
            except Exception:  # noqa: BLE001
                pass
        return self.interval_s

    def _maybe_persist(self, *, force: bool = False) -> None:
        """Checkpoint adopted goals + learned drives so autonomy is durable across restarts."""
        if self.decision_mode != "code":
            return
        if not force and (self.persist_every <= 0 or self.ticks % self.persist_every != 0):
            return
        fn = getattr(self.core, "persist_autonomy_state", None)
        if fn is None:
            return
        try:
            fn()
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    # ---- one step ---- #
    def tick_once(self) -> Optional[Any]:
        """Run exactly one autonomic turn (synchronously). Returns the result, or None
        if the loop is currently halted by oversight (paused/scrammed)."""
        if not self.core.oversight.gate():
            return None
        if self.decision_mode == "code":
            result: Any = self._decide_and_act_once()
            self.ticks += 1
            self.prompt_sources.append("code")
        else:
            prompt, source = self._compose_prompt()
            result = self.core.process(prompt, authority=self.authority)
            self.ticks += 1
            self.prompt_sources.append(source)
            self.history.append(result)
            if result.disposition is Disposition.ESCALATE:
                self.escalations.append(result)
        if self.health is not None:
            try:
                self.health.beat("autonomic")
            except Exception:  # noqa: BLE001
                pass
        self._advance_mission()
        self._maybe_grow()
        self._maybe_persist()
        return result

    def run_for(self, n: int, *, sleep: bool = False) -> List[CycleResult]:
        """Run ``n`` autonomic ticks synchronously (no sleeping by default — great for
        tests and one-shot scheduler invocations)."""
        out: List[CycleResult] = []
        for _ in range(max(0, n)):
            r = self.tick_once()
            if r is not None:
                out.append(r)
            if sleep and self.interval_s > 0:
                time.sleep(self.interval_s)
        return out

    # ---- true async background task ---- #
    async def _run(self, max_ticks: Optional[int]) -> None:
        import asyncio
        self._running = True
        done = 0
        try:
            while self._running:
                if self.core.oversight.gate():
                    # presence (when wired) advances arousal and gates proactivity per state
                    self._apply_presence()
                    if self.decision_mode == "code":
                        # the decision + action are pure code, but the action callables may do
                        # real I/O (research), so run them off the event loop to stay responsive
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, self._decide_and_act_once)
                        self.ticks += 1
                        self.prompt_sources.append("code")
                    else:
                        prompt, source = self._compose_prompt()
                        result = await self.core.aprocess(prompt, authority=self.authority)
                        self.ticks += 1
                        self.prompt_sources.append(source)
                        self.history.append(result)
                        if result.disposition is Disposition.ESCALATE:
                            self.escalations.append(result)
                    if self.health is not None:
                        try:
                            self.health.beat("autonomic")
                        except Exception:  # noqa: BLE001
                            pass
                    self._advance_mission()
                    self._maybe_grow()
                    self._maybe_persist()
                done += 1
                if max_ticks is not None and done >= max_ticks:
                    break
                await asyncio.sleep(self._current_interval())
        finally:
            self._running = False
            # a final checkpoint so the last-adopted goals/drives are never lost on shutdown
            self._maybe_persist(force=True)

    def start(self, *, max_ticks: Optional[int] = None) -> Any:
        """Schedule the loop as a background asyncio task (requires a running loop)."""
        import asyncio
        if self._task is not None and not self._task.done():
            return self._task
        self._task = asyncio.ensure_future(self._run(max_ticks))
        return self._task

    async def stop(self) -> None:
        """Stop the background task and await its completion."""
        self._running = False
        if self._task is not None:
            try:
                await self._task
            except Exception:  # noqa: BLE001
                pass
            self._task = None

    @property
    def running(self) -> bool:
        return self._running

    def report(self) -> dict:
        rep = {"ticks": self.ticks, "running": self.running,
               "interval_s": self.interval_s, "inner_life": self.inner_life,
               "decision_mode": self.decision_mode,
               "acted": sum(1 for r in self.history if r.disposition is Disposition.ACT),
               "escalations": len(self.escalations),
               "missions_advanced": self.missions_advanced,
               "growth_passes": len(self.growth_reports),
               # code-driven metrics: what NYXARA decided and did on her own, without the LLM
               "intents_adopted": self.intents_adopted,
               "code_acts": self.code_acts,
               "scheduler_runs": self.scheduler_runs,
               "sources": {s: self.prompt_sources.count(s)
                           for s in sorted(set(self.prompt_sources))}}
        if self.presence is not None:
            try:
                rep["presence"] = self.presence.state.name
            except Exception:  # noqa: BLE001
                pass
        if self.health is not None:
            try:
                rep["health"] = self.health.evaluate().overall.label
            except Exception:  # noqa: BLE001
                pass
        return rep


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import asyncio

    print("=" * 70)
    print("NYXARA autonomic-loop self-test")
    print("=" * 70)

    core = NyxaraCore()
    loop = AutonomicLoop(core, interval_s=0.0)

    # synchronous bounded run: self-directed reflection turns
    results = loop.run_for(4)
    print(f"\nsync ticks          : {len(results)} (acted/escalated mix)")
    assert loop.ticks == 4
    print(f"report              : {loop.report()}")

    # oversight halts the background mind too: scram -> ticks no-op
    core.scram(reason="stand down")
    r = loop.tick_once()
    print(f"\nwhile scrammed      : tick returned {r!r}")
    assert r is None
    core.resume()

    # true async background task, bounded to 3 ticks
    async def _demo():
        core.resume()
        loop.start(max_ticks=3)
        await loop._task
    asyncio.run(_demo())
    print(f"async ticks total   : {loop.ticks} (>= 7)")
    assert loop.ticks >= 7

    print("\nALL SELF-TESTS PASSED ✓")
