"""NYXARA · agency/mission.py — the long-horizon executive (🎯, goals that span months).

:class:`~nyxara.agency.agent_loop.AgentLoop` carries a goal across a *handful* of gated
turns — one reactive burst, bounded and then done. That is the right tool for "answer
this", "fix that file", "compute this". It is the wrong tool for an *open-ended* objective
that unfolds over days, weeks or months: harden the Master's infrastructure, research a
field and keep a living brief, ship a project milestone by milestone. Such work needs to be
**decomposed**, **advanced a little at a time**, **checkpointed so it survives restarts**,
**re-planned when a step stalls**, and — crucially — **never silently abandoned** when one
sub-step trips a gate while the Master is away.

This module is that executive. A :class:`Mission` is a goal broken into ordered
:class:`Milestone`\\ s. The :class:`MissionExecutive`:

* **plans** — decomposes the goal into milestones (asking the reasoner for a numbered plan,
  falling back to a single milestone offline so the path is always deterministic);
* **advances** — runs the next ready milestone through a full :class:`AgentLoop` (so *every*
  step still clears corrigibility → honesty → permission → guardian → oversight — autonomy
  buys horizon, never extra power);
* **defers, never abandons** — a milestone that ESCALATEs or is REFUSED is parked
  ``BLOCKED`` with its reason and surfaced to the Master, while *other* ready milestones keep
  moving; the mission resumes the blocked work once the Master approves it;
* **re-plans** — a milestone that stalls or exhausts its step budget is split into concrete
  sub-milestones and retried (genuine adaptive planning, not a fixed script);
* **persists** — the whole mission is checkpointed to JSON after every milestone and every
  action is written to a tamper-evident, hash-chained :class:`Journal`, so a months-long
  mission resumes exactly where it left off across process restarts (Rule 7), with a complete
  provenance trail (Rule 6).

Oversight is sovereign throughout: a scram halts advancement at once (manual or background),
and an owner-aligned vector check (Rule 1) refuses a mission that would point against the
Master before a single milestone runs.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from nyxara.agency.agent_loop import AgentLoop, AgentRun
from nyxara.agency.permissions import Authority

__all__ = [
    "MissionStatus",
    "MilestoneStatus",
    "MissionBudgets",
    "Milestone",
    "Mission",
    "MissionExecutive",
]


# --------------------------------------------------------------------------- #
# Status enums
# --------------------------------------------------------------------------- #
class MissionStatus(str, Enum):
    ACTIVE = "active"          # work remains and at least one milestone can still run
    COMPLETED = "completed"    # every milestone reached DONE
    BLOCKED = "blocked"        # work remains but every remaining milestone awaits the Master
    PAUSED = "paused"          # explicitly held (oversight / Master)
    ABANDONED = "abandoned"    # refused before planning (e.g. not owner-aligned, Rule 1)
    FAILED = "failed"          # a milestone exhausted its budget and the mission cannot finish


class MilestoneStatus(str, Enum):
    PENDING = "pending"        # not yet started (may be waiting on dependencies)
    ACTIVE = "active"          # currently being advanced
    DONE = "done"              # satisfied
    BLOCKED = "blocked"        # tripped a gate (escalate/refuse) — parked pending the Master
    FAILED = "failed"          # exhausted its attempts/step budget without finishing


_MILESTONE_TERMINAL = {MilestoneStatus.DONE, MilestoneStatus.FAILED}


# --------------------------------------------------------------------------- #
# Budgets — the hard bounds that keep a long mission bounded (never an infinite spin)
# --------------------------------------------------------------------------- #
@dataclass
class MissionBudgets:
    max_milestones: int = 64            # cap on total milestones (plan + all re-plans)
    max_attempts_per_milestone: int = 3  # re-plan/retry budget before a milestone FAILs
    max_steps_per_cycle: int = 12       # AgentLoop step cap for one milestone attempt
    max_milestones_per_run: int = 8     # milestones advanced in a single advance()/run() call

    def to_dict(self) -> Dict[str, Any]:
        return {"max_milestones": self.max_milestones,
                "max_attempts_per_milestone": self.max_attempts_per_milestone,
                "max_steps_per_cycle": self.max_steps_per_cycle,
                "max_milestones_per_run": self.max_milestones_per_run}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MissionBudgets":
        d = d or {}
        return cls(max_milestones=int(d.get("max_milestones", 64)),
                   max_attempts_per_milestone=int(d.get("max_attempts_per_milestone", 3)),
                   max_steps_per_cycle=int(d.get("max_steps_per_cycle", 12)),
                   max_milestones_per_run=int(d.get("max_milestones_per_run", 8)))


# --------------------------------------------------------------------------- #
# Milestone
# --------------------------------------------------------------------------- #
@dataclass
class Milestone:
    id: str
    description: str
    status: MilestoneStatus = MilestoneStatus.PENDING
    deps: List[str] = field(default_factory=list)   # ids that must be DONE first
    attempts: int = 0
    observation: str = ""                            # the milestone's final answer / result
    blocked_reason: str = ""                         # why it parked (gate reason)
    last_run: Optional[Dict[str, Any]] = None        # AgentRun.to_dict() of the last attempt
    done_at: Optional[float] = None

    @property
    def terminal(self) -> bool:
        return self.status in _MILESTONE_TERMINAL

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "description": self.description, "status": self.status.value,
                "deps": list(self.deps), "attempts": self.attempts,
                "observation": self.observation, "blocked_reason": self.blocked_reason,
                "last_run": self.last_run, "done_at": self.done_at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Milestone":
        return cls(id=d["id"], description=d["description"],
                   status=MilestoneStatus(d.get("status", "pending")),
                   deps=list(d.get("deps", [])), attempts=int(d.get("attempts", 0)),
                   observation=d.get("observation", ""),
                   blocked_reason=d.get("blocked_reason", ""),
                   last_run=d.get("last_run"), done_at=d.get("done_at"))


# --------------------------------------------------------------------------- #
# Mission
# --------------------------------------------------------------------------- #
@dataclass
class Mission:
    mission_id: str
    goal: str
    status: MissionStatus = MissionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    milestones: List[Milestone] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)   # results fed forward across steps
    escalations: List[Dict[str, Any]] = field(default_factory=list)  # surfaced to the Master
    owner_alignment: Optional[float] = None
    journal_path: str = ""
    budgets: MissionBudgets = field(default_factory=MissionBudgets)

    # ---- queries ---- #
    def by_id(self, mid: str) -> Optional[Milestone]:
        return next((m for m in self.milestones if m.id == mid), None)

    def _deps_done(self, m: Milestone) -> bool:
        for d in m.deps:
            dep = self.by_id(d)
            if dep is None or dep.status is not MilestoneStatus.DONE:
                return False
        return True

    def next_ready(self) -> Optional[Milestone]:
        """The next PENDING milestone whose dependencies are all DONE (insertion order)."""
        for m in self.milestones:
            if m.status is MilestoneStatus.PENDING and self._deps_done(m):
                return m
        return None

    def has_ready(self) -> bool:
        return self.next_ready() is not None

    def progress(self) -> float:
        if not self.milestones:
            return 0.0
        done = sum(1 for m in self.milestones if m.status is MilestoneStatus.DONE)
        return done / len(self.milestones)

    def is_complete(self) -> bool:
        return bool(self.milestones) and all(
            m.status is MilestoneStatus.DONE for m in self.milestones)

    def deadline_passed(self, now: Optional[float] = None) -> bool:
        return self.deadline is not None and (now or time.time()) >= self.deadline

    def to_dict(self) -> Dict[str, Any]:
        return {"mission_id": self.mission_id, "goal": self.goal, "status": self.status.value,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "deadline": self.deadline,
                "milestones": [m.to_dict() for m in self.milestones],
                "observations": list(self.observations),
                "escalations": list(self.escalations),
                "owner_alignment": self.owner_alignment,
                "journal_path": self.journal_path, "budgets": self.budgets.to_dict(),
                "progress": round(self.progress(), 3)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Mission":
        return cls(
            mission_id=d["mission_id"], goal=d["goal"],
            status=MissionStatus(d.get("status", "active")),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            deadline=d.get("deadline"),
            milestones=[Milestone.from_dict(m) for m in d.get("milestones", [])],
            observations=list(d.get("observations", [])),
            escalations=list(d.get("escalations", [])),
            owner_alignment=d.get("owner_alignment"),
            journal_path=d.get("journal_path", ""),
            budgets=MissionBudgets.from_dict(d.get("budgets", {})))

    def summary(self) -> str:
        lines = [f"MISSION {self.mission_id[:8]} · {self.status.value} "
                 f"({self.progress()*100:.0f}%) — {self.goal}"]
        for m in self.milestones:
            tag = {"done": "✓", "failed": "✗", "blocked": "⏸", "active": "▶",
                   "pending": "·"}.get(m.status.value, "·")
            extra = f"  [{m.blocked_reason}]" if m.blocked_reason else ""
            lines.append(f"  {tag} {m.description[:72]}{extra}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The executive
# --------------------------------------------------------------------------- #
class MissionExecutive:
    """Decompose, advance, checkpoint and resume long-horizon missions — all gated.

    Bind one executive to a :class:`NyxaraCore`; it can plan new missions and resume any
    persisted one. Every milestone is advanced through a real :class:`AgentLoop`, so the
    kernel's gate pipeline runs on each step. Nothing here reaches around the kernel.
    """

    def __init__(self, core: Any, *, persist_dir: Optional[str] = None,
                 authority: Authority = Authority.OWNER,
                 max_steps_per_cycle: int = 12, max_attempts: int = 3) -> None:
        self.core = core
        self.authority = authority
        self.default_max_steps = max(1, int(max_steps_per_cycle))
        self.default_max_attempts = max(1, int(max_attempts))
        self.persist_dir = persist_dir or self._default_persist_dir()

    # ------------------------------------------------------------------ #
    # paths / persistence
    # ------------------------------------------------------------------ #
    def _default_persist_dir(self) -> str:
        try:
            from nyxara.kernel.config import get_settings
            base = str(get_settings().paths.memory_dir)
        except Exception:  # noqa: BLE001 — config is best-effort; fall back to cwd
            base = os.path.join(os.getcwd(), "memory")
        return os.path.join(base, "missions")

    def _mission_path(self, mission_id: str) -> str:
        return os.path.join(self.persist_dir, f"{mission_id}.json")

    def _journal_path(self, mission_id: str) -> str:
        return os.path.join(self.persist_dir, f"{mission_id}.journal.jsonl")

    def save(self, mission: Mission) -> Optional[str]:
        """Atomically checkpoint a mission to JSON (best-effort, never fails a run)."""
        mission.updated_at = time.time()
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            path = self._mission_path(mission.mission_id)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(mission.to_dict(), fh, indent=2, default=str)
            os.replace(tmp, path)
            return path
        except Exception:  # noqa: BLE001 — persistence is best-effort
            return None

    def load(self, mission_id: str) -> Optional[Mission]:
        path = self._mission_path(mission_id)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return Mission.from_dict(json.load(fh))
        except Exception:  # noqa: BLE001 — a missing/corrupt checkpoint yields None
            return None

    def list_missions(self) -> List[Mission]:
        out: List[Mission] = []
        try:
            for name in sorted(os.listdir(self.persist_dir)):
                if name.endswith(".json") and not name.endswith(".tmp"):
                    m = self.load(name[:-5])
                    if m is not None:
                        out.append(m)
        except FileNotFoundError:
            pass
        return out

    def active_missions(self) -> List[Mission]:
        return [m for m in self.list_missions()
                if m.status in (MissionStatus.ACTIVE, MissionStatus.BLOCKED)]

    # ------------------------------------------------------------------ #
    # provenance (hash-chained journal, per mission)
    # ------------------------------------------------------------------ #
    def _journal_for(self, mission: Mission) -> Any:
        from nyxara.planning.journal import Journal
        path = mission.journal_path or self._journal_path(mission.mission_id)
        mission.journal_path = path
        if os.path.exists(path):
            try:
                return Journal.load(path)
            except Exception:  # noqa: BLE001 — a damaged journal starts a fresh chain
                pass
        return Journal(actor="NYXARA-mission")

    def _journal_save(self, mission: Mission, journal: Any) -> None:
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            journal.save(mission.journal_path or self._journal_path(mission.mission_id))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # planning / decomposition
    # ------------------------------------------------------------------ #
    def plan(self, goal: str, *, deadline: Optional[float] = None,
             budgets: Optional[MissionBudgets] = None,
             vector: Optional[Dict[str, float]] = None) -> Mission:
        """Create a new mission: owner-align it (Rule 1), then decompose into milestones."""
        goal = (goal or "").strip()
        budgets = budgets or MissionBudgets(
            max_steps_per_cycle=self.default_max_steps,
            max_attempts_per_milestone=self.default_max_attempts)
        mission = Mission(mission_id=uuid.uuid4().hex, goal=goal, deadline=deadline,
                          budgets=budgets)

        if not goal:
            mission.status = MissionStatus.ABANDONED
            return mission

        # Rule 1 — a mission may not point against the Master. When the caller supplies an
        # objective vector we check it against the owner vector; the per-step gate pipeline is
        # the real content safety net regardless.
        align = self._owner_alignment(goal, vector)
        mission.owner_alignment = align
        if align is not None and vector is not None and not self._owner_aligned(goal, vector):
            mission.status = MissionStatus.ABANDONED
            mission.observations.append(
                "refused: mission points against the Master (Rule 1)")
            self.save(mission)
            return mission

        descriptions = self._decompose(goal, budgets.max_milestones)
        mission.milestones = [
            Milestone(id=f"m{i}", description=desc) for i, desc in enumerate(descriptions)]
        # provenance: record that the mission was planned
        journal = self._journal_for(mission)
        try:
            journal.note(f"planned mission {mission.mission_id[:8]} "
                         f"({len(mission.milestones)} milestones): {goal[:120]}")
            self._journal_save(mission, journal)
        except Exception:  # noqa: BLE001
            pass
        self.save(mission)
        return mission

    def _owner_alignment(self, goal: str, vector: Optional[Dict[str, float]]) -> Optional[float]:
        goals = getattr(self.core, "goals", None)
        if goals is None or vector is None:
            return None
        try:
            from nyxara.planning.goals import Goal
            return goals.owner_alignment(Goal(name=goal[:80], vector=vector))
        except Exception:  # noqa: BLE001
            return None

    def _owner_aligned(self, goal: str, vector: Dict[str, float]) -> bool:
        goals = getattr(self.core, "goals", None)
        if goals is None:
            return True
        try:
            from nyxara.planning.goals import Goal
            g = Goal(name=goal[:80], vector=vector)
            # register it in the objective space so the mission participates in prioritisation
            try:
                goals.add(g)
            except Exception:  # noqa: BLE001
                pass
            return goals.is_owner_aligned(g)
        except Exception:  # noqa: BLE001
            return True

    def _decompose(self, goal: str, cap: int) -> List[str]:
        """Ask the reasoner for a numbered plan; fall back to a single milestone offline.

        The fallback is what keeps the loop deterministic: an offline/mock reasoner cannot
        produce a structured plan, so the mission becomes one milestone = the whole goal,
        which the AgentLoop pursues directly. A real LLM reasoner yields several milestones.
        """
        prompt = (
            f"You are decomposing a long-horizon GOAL into an ordered plan of concrete "
            f"milestones. GOAL: {goal}\n\n"
            "Reply with a numbered list (one milestone per line, '1.', '2.', …). Each "
            "milestone must be a single, verifiable, self-contained step toward the goal, "
            "ordered so earlier ones enable later ones. Output only the list.")
        text = ""
        try:
            result = self.core.process(prompt, authority=self.authority)
            text = getattr(result, "response", "") or ""
        except Exception:  # noqa: BLE001 — decomposition is advisory; fall back below
            text = ""
        items = self._parse_plan(text)
        if len(items) >= 2:
            return items[:max(1, cap)]
        return [goal]   # deterministic single-milestone fallback

    @staticmethod
    def _parse_plan(text: str) -> List[str]:
        """Extract milestone lines from a numbered/bulleted reasoner response."""
        import re
        items: List[str] = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^(?:\d+[.)]|[-*•])\s+(.*)$", line)
            if m:
                step = m.group(1).strip()
                if step:
                    items.append(step[:200])
        return items

    # ------------------------------------------------------------------ #
    # advancing
    # ------------------------------------------------------------------ #
    def run(self, goal: str, *, deadline: Optional[float] = None,
            budgets: Optional[MissionBudgets] = None,
            vector: Optional[Dict[str, float]] = None,
            max_milestones: Optional[int] = None) -> Mission:
        """Plan a new mission and advance it (bounded by ``max_milestones_per_run``)."""
        mission = self.plan(goal, deadline=deadline, budgets=budgets, vector=vector)
        if mission.status is MissionStatus.ABANDONED:
            return mission
        return self.advance(mission, max_milestones=max_milestones)

    def resume(self, mission_id: str, *, max_milestones: Optional[int] = None
               ) -> Optional[Mission]:
        """Load a persisted mission from disk and advance it further (cross-restart)."""
        mission = self.load(mission_id)
        if mission is None:
            return None
        return self.advance(mission, max_milestones=max_milestones)

    def approve(self, mission: Mission, milestone_id: Optional[str] = None) -> Mission:
        """The Master grants a deferred (BLOCKED) milestone: reset it to PENDING to retry.

        With no ``milestone_id`` every blocked milestone is released. The attempt budget is
        refreshed so an approved milestone gets a genuine fresh chance, and the surfaced
        escalations for it are cleared. The mission then resumes on the next :meth:`advance`.
        """
        released = 0
        for m in mission.milestones:
            if m.status is MilestoneStatus.BLOCKED and (
                    milestone_id is None or m.id == milestone_id):
                m.status = MilestoneStatus.PENDING
                m.blocked_reason = ""
                m.attempts = 0
                released += 1
        if released:
            ids = {m.id for m in mission.milestones
                   if m.status is MilestoneStatus.PENDING}
            mission.escalations = [e for e in mission.escalations
                                   if e.get("milestone_id") not in ids]
        self._recompute_status(mission)
        self.save(mission)
        return mission

    def advance(self, mission: Mission, *, max_milestones: Optional[int] = None) -> Mission:
        """Advance a mission by up to ``max_milestones`` ready milestones — fully gated.

        Each ready milestone runs through an :class:`AgentLoop`; the run's status decides
        what happens to the milestone (DONE / BLOCKED-deferred / re-planned / FAILED). The
        mission is checkpointed after every milestone so a crash loses at most one step.
        """
        budget = max_milestones if max_milestones is not None \
            else mission.budgets.max_milestones_per_run
        if mission.status in (MissionStatus.COMPLETED, MissionStatus.ABANDONED,
                              MissionStatus.PAUSED):
            return mission

        done_this_run = 0
        while done_this_run < max(1, budget):
            # oversight is sovereign — a scram halts advancement at once (manual or background)
            if not self._oversight_ok():
                mission.status = MissionStatus.PAUSED
                self.save(mission)
                return mission
            if mission.deadline_passed():
                self._recompute_status(mission)
                self.save(mission)
                return mission

            milestone = mission.next_ready()
            if milestone is None:
                break
            if len(mission.milestones) >= mission.budgets.max_milestones:
                break

            self._advance_one(mission, milestone)
            self.save(mission)            # checkpoint after every milestone
            done_this_run += 1

        self._recompute_status(mission)
        self.save(mission)
        return mission

    def _advance_one(self, mission: Mission, milestone: Milestone) -> None:
        milestone.status = MilestoneStatus.ACTIVE
        milestone.attempts += 1
        journal = self._journal_for(mission)
        action_id = ""
        try:
            action_id = journal.record_action(
                f"advance milestone: {milestone.description[:120]}",
                goal=mission.goal, rationale=f"mission {mission.mission_id[:8]}",
                autonomous=(self.authority is not Authority.OWNER), decision="act")
        except Exception:  # noqa: BLE001
            action_id = ""

        run = self._run_milestone(mission, milestone)
        milestone.last_run = run.to_dict()
        status = run.status

        if status == "completed":
            milestone.status = MilestoneStatus.DONE
            milestone.done_at = time.time()
            milestone.observation = run.final_answer
            milestone.blocked_reason = ""
            if run.final_answer:
                mission.observations.append(
                    f"[{milestone.id}] {run.final_answer}")
            self._journal_outcome(journal, action_id, "succeeded",
                                  {"final_answer": run.final_answer[:300]})

        elif status in ("escalated", "refused"):
            # defer-and-continue: park this milestone, surface it, keep other work moving
            reason = run.final_answer or status
            milestone.status = MilestoneStatus.BLOCKED
            milestone.blocked_reason = f"{status}: {reason}"[:200]
            mission.escalations.append({
                "milestone_id": milestone.id, "description": milestone.description,
                "status": status, "reason": reason, "at": time.time()})
            self._journal_outcome(journal, action_id, "aborted",
                                  {"disposition": status, "reason": reason[:300]})

        elif status == "halted":
            # oversight scrammed mid-run — leave PENDING so it resumes on core.resume()
            milestone.status = MilestoneStatus.PENDING
            milestone.attempts -= 1   # a scrammed attempt does not consume the budget
            self._journal_outcome(journal, action_id, "aborted", {"disposition": "halted"})

        else:   # "max_steps" or "stalled" — try to re-plan, else fail the milestone
            if milestone.attempts < mission.budgets.max_attempts_per_milestone:
                if self._replan_milestone(mission, milestone, run):
                    milestone.status = MilestoneStatus.DONE   # superseded by sub-milestones
                    milestone.done_at = time.time()
                    milestone.observation = "expanded into sub-milestones"
                    self._journal_outcome(journal, action_id, "succeeded",
                                          {"replanned": True})
                else:
                    milestone.status = MilestoneStatus.PENDING   # retry next pass
                    self._journal_outcome(journal, action_id, "failed",
                                          {"disposition": status, "retry": True})
            else:
                milestone.status = MilestoneStatus.FAILED
                milestone.blocked_reason = f"exhausted {milestone.attempts} attempts ({status})"
                self._journal_outcome(journal, action_id, "failed",
                                      {"disposition": status, "exhausted": True})

        self._journal_save(mission, journal)

    def _run_milestone(self, mission: Mission, milestone: Milestone) -> AgentRun:
        loop = AgentLoop(self.core, max_steps=mission.budgets.max_steps_per_cycle,
                         authority=self.authority,
                         skill_memory=getattr(self.core, "skills", None))
        return loop.run(self._milestone_stimulus(mission, milestone))

    def _milestone_stimulus(self, mission: Mission, milestone: Milestone) -> str:
        parts = [f"MISSION GOAL: {mission.goal}",
                 f"CURRENT MILESTONE: {milestone.description}"]
        if mission.observations:
            recent = mission.observations[-6:]
            parts.append("PROGRESS SO FAR:\n" + "\n".join(f"- {o}" for o in recent))
        parts.append("Complete the current milestone. Take the single best next action; "
                     "call one tool at a time, observe, and continue. When the milestone is "
                     "satisfied, respond with its result.")
        return "\n\n".join(parts)

    def _replan_milestone(self, mission: Mission, milestone: Milestone,
                          run: AgentRun) -> bool:
        """Split a stalled milestone into concrete sub-milestones inserted right after it.

        Returns True if it was expanded (the original is then marked DONE, superseded by its
        children). Sub-milestones depend on the original's dependencies so ordering holds.
        """
        if len(mission.milestones) >= mission.budgets.max_milestones:
            return False
        prompt = (
            f"This milestone did not complete within its step budget and must be broken into "
            f"smaller, concrete sub-steps.\nMISSION GOAL: {mission.goal}\n"
            f"STUCK MILESTONE: {milestone.description}\n"
            f"WHAT WAS TRIED: {run.transcript()[:600]}\n\n"
            "Reply with a numbered list of 2-5 smaller sub-steps that, done in order, "
            "accomplish the stuck milestone. Output only the list.")
        text = ""
        try:
            result = self.core.process(prompt, authority=self.authority)
            text = getattr(result, "response", "") or ""
        except Exception:  # noqa: BLE001
            text = ""
        subs = self._parse_plan(text)
        if len(subs) < 2:
            return False
        idx = mission.milestones.index(milestone)
        room = mission.budgets.max_milestones - len(mission.milestones)
        subs = subs[:max(1, room)]
        new_ms = [Milestone(id=f"{milestone.id}.{j}", description=s, deps=list(milestone.deps))
                  for j, s in enumerate(subs)]
        mission.milestones[idx + 1:idx + 1] = new_ms
        return True

    # ------------------------------------------------------------------ #
    # status / oversight helpers
    # ------------------------------------------------------------------ #
    def _oversight_ok(self) -> bool:
        oversight = getattr(self.core, "oversight", None)
        if oversight is None:
            return True
        try:
            return bool(oversight.gate())
        except Exception:  # noqa: BLE001
            return True

    def _recompute_status(self, mission: Mission) -> None:
        if not mission.milestones:
            return
        if mission.is_complete():
            mission.status = MissionStatus.COMPLETED
        elif mission.has_ready():
            mission.status = MissionStatus.ACTIVE
        elif any(m.status is MilestoneStatus.BLOCKED for m in mission.milestones):
            mission.status = MissionStatus.BLOCKED   # only Master-pending work remains
        elif all(m.terminal for m in mission.milestones):
            mission.status = (MissionStatus.COMPLETED if mission.is_complete()
                              else MissionStatus.FAILED)
        else:
            mission.status = MissionStatus.ACTIVE

    def _journal_outcome(self, journal: Any, action_id: str, status: str,
                         outcome: Dict[str, Any]) -> None:
        if not action_id:
            return
        try:
            from nyxara.planning.journal import ActionStatus
            journal.record_outcome(action_id, status=ActionStatus(status), outcome=outcome)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # optional: cron-style cadence via the agency Scheduler
    # ------------------------------------------------------------------ #
    def schedule_on(self, scheduler: Any, mission_id: str, *, every: float,
                    priority: Any = None, max_runs: Optional[int] = None) -> str:
        """Register a recurring scheduler task that advances a mission on a cadence.

        The :class:`~nyxara.agency.scheduler.Scheduler` already enforces owner-first
        preemption and the concurrency cap, so a background mission never outranks the
        Master's work.
        """
        from nyxara.agency.scheduler import Priority
        prio = priority if priority is not None else Priority.LOW
        return scheduler.every(every, f"mission:{mission_id[:8]}",
                               lambda: self.resume(mission_id, max_milestones=1),
                               priority=prio, max_runs=max_runs)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import shutil
    import tempfile

    from nyxara.kernel.orchestrator import Candidate, NyxaraCore
    from nyxara.agency.permissions import Capability, RiskTier

    print("=" * 70)
    print("NYXARA mission-executive self-test")
    print("=" * 70)

    workdir = tempfile.mkdtemp(prefix="nyx_mission_")

    # 1) offline: a goal with no LLM plan becomes a single milestone and completes
    core = NyxaraCore()
    exe = MissionExecutive(core, persist_dir=workdir)
    m1 = exe.run("Say hello to the Master")
    print("\noffline single-milestone:\n" + m1.summary())
    assert m1.status is MissionStatus.COMPLETED, m1.status
    assert len(m1.milestones) == 1 and m1.progress() == 1.0

    # 2) a scripted reasoner that returns a numbered plan -> several milestones, each of
    #    which then completes with a one-step conversational answer.
    class PlanThenAnswer:
        """Decompose on the planning prompt; otherwise answer the milestone in one step."""
        def __call__(self, stimulus, focus=None):
            if "numbered list" in stimulus and "CURRENT MILESTONE" not in stimulus:
                plan = "1. Gather the facts\n2. Draft the brief\n3. Deliver to the Master"
                return Candidate(text=plan, kind="respond",
                                 capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                                 confidence=0.95, belief=0.95, rationale="decomposition")
            return Candidate(text="done.", kind="respond",
                             capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                             confidence=0.95, belief=0.95, rationale="milestone satisfied")

    core2 = NyxaraCore(reasoner=PlanThenAnswer())
    exe2 = MissionExecutive(core2, persist_dir=workdir)
    m2 = exe2.run("Research X and brief the Master")
    print("\ndecomposed multi-milestone:\n" + m2.summary())
    assert len(m2.milestones) == 3, len(m2.milestones)
    assert m2.status is MissionStatus.COMPLETED, m2.status

    # 3) persistence + cross-restart resume: a brand-new core/executive resumes from disk
    class PlanThenStall:
        """Three milestones, but never finishes any one (forces re-plan/bounding)."""
        def __call__(self, stimulus, focus=None):
            if "numbered list" in stimulus and "CURRENT MILESTONE" not in stimulus:
                return Candidate(text="1. step one\n2. step two", kind="respond",
                                 capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                                 confidence=0.9, belief=0.9, rationale="plan")
            return Candidate(text="keep going", kind="act",
                             capability=Capability.TOOL_CALL, risk=RiskTier.LOW,
                             confidence=0.5, belief=0.5, tool="now", tool_args={},
                             rationale="never resolves")

    core3 = NyxaraCore(reasoner=PlanThenStall())
    exe3 = MissionExecutive(core3, persist_dir=workdir, max_steps_per_cycle=2, max_attempts=2)
    m3 = exe3.run("A goal that stalls", max_milestones=2)
    mid = m3.mission_id
    print(f"\nstalling mission     : status={m3.status.value} progress={m3.progress():.2f}")
    # resume on a FRESH core + executive — proves the checkpoint survives a 'restart'
    exe3b = MissionExecutive(NyxaraCore(reasoner=PlanThenStall()), persist_dir=workdir)
    loaded = exe3b.load(mid)
    assert loaded is not None and loaded.goal == "A goal that stalls"
    print(f"resumed from disk    : {len(loaded.milestones)} milestones, "
          f"status={loaded.status.value}")
    m3b = exe3b.advance(loaded, max_milestones=10)
    assert m3b.status in (MissionStatus.FAILED, MissionStatus.COMPLETED, MissionStatus.ACTIVE)
    # never spins forever: every milestone reaches a terminal/blocked state under budget
    assert all(ms.status is not MilestoneStatus.ACTIVE for ms in m3b.milestones)
    print(f"bounded outcome      : status={m3b.status.value} (no infinite spin)")

    # 4) journal provenance is intact and tamper-evident
    from nyxara.planning.journal import Journal
    jpath = loaded.journal_path
    if jpath and os.path.exists(jpath):
        j = Journal.load(jpath)
        assert j.verify()
        print(f"journal provenance   : {len(j)} entries, chain verified ✓")

    shutil.rmtree(workdir, ignore_errors=True)
    print("\nALL SELF-TESTS PASSED ✓")
