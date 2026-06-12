"""NYXARA · growth/skill_memory.py — experiential skill learning (📚 → ⬆, the closed loop).

NYXARA's reasoner is fixed weights; what *can* change with experience is the procedural
knowledge she carries into each new problem. This module is that learning channel — **not**
gradient fine-tuning (which lives in the foundry), but the cheaper, always-on kind: when an
agentic run succeeds, the sequence of tool calls that worked is distilled into a **skill** and
remembered as PROCEDURAL memory. The next time a similar goal appears, the relevant skills are
recalled and injected into the reasoning prompt, so behaviour measurably improves with use.

The loop closes: ``AgentLoop.run`` produces a run → :meth:`SkillMemory.capture` distils &
persists the winning procedure → :meth:`SkillMemory.as_prompt` surfaces it on the next related
goal → the reasoner reuses what worked. Skills are deduplicated by goal, reinforced on reuse,
and persisted through the :class:`~nyxara.memory.store.MemoryStore` (so they survive restarts
alongside the rest of long-term memory). With no store, an in-process list is used instead.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Skill", "SkillMemory"]


@dataclass
class Skill:
    """A reusable procedure distilled from a successful run."""
    goal: str
    tools: List[str]
    procedure: str                              # human-readable how-to
    skill_id: str = field(default_factory=lambda: "skill-" + uuid.uuid4().hex[:8])
    uses: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0

    def reinforce(self) -> None:
        self.uses += 1
        self.last_used = time.time()

    def as_line(self) -> str:
        tools = ", ".join(self.tools) if self.tools else "reasoning only"
        return f"- to {self.goal!r} (via {tools}): {self.procedure}"

    def to_dict(self) -> Dict[str, Any]:
        return {"skill_id": self.skill_id, "goal": self.goal, "tools": list(self.tools),
                "procedure": self.procedure, "uses": self.uses,
                "created_at": self.created_at, "last_used": self.last_used}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        return cls(goal=d.get("goal", ""), tools=list(d.get("tools", [])),
                   procedure=d.get("procedure", ""),
                   skill_id=d.get("skill_id", "skill-" + uuid.uuid4().hex[:8]),
                   uses=int(d.get("uses", 0)), created_at=float(d.get("created_at", time.time())),
                   last_used=float(d.get("last_used", 0.0)))


class SkillMemory:
    """Capture successful procedures and recall them to bias future reasoning."""

    TAG = "skill"

    def __init__(self, store: Any = None, *, min_steps: int = 1) -> None:
        self.store = store
        self.min_steps = max(1, min_steps)
        self._local: List[Skill] = []          # used only when no store is provided

    # ---- capture ---- #
    def capture(self, run: Any) -> Optional[Skill]:
        """Distil a successful :class:`~nyxara.agency.agent_loop.AgentRun` into a skill.

        Returns the stored (or reinforced) skill, or None if the run isn't worth learning
        from (unsuccessful, or below ``min_steps`` of substantive action).
        """
        if not getattr(run, "success", False):
            return None
        steps = list(getattr(run, "steps", []))
        tool_steps = [s for s in steps if getattr(s, "tool", None)]
        if len(steps) < self.min_steps:
            return None

        tools = [s.tool for s in tool_steps]
        procedure = self._distil(steps, getattr(run, "final_answer", ""))
        goal = getattr(run, "goal", "").strip()

        # dedup: if we already know this goal closely, reinforce instead of duplicating
        existing = self._closest(goal)
        if existing is not None:
            existing.reinforce()
            self._persist(existing, update=True)
            return existing

        skill = Skill(goal=goal, tools=tools, procedure=procedure)
        skill.reinforce()
        self._persist(skill, update=False)
        return skill

    def learn(self, goal: str, procedure: str, tools: Optional[List[str]] = None) -> Skill:
        """Directly teach a skill (e.g. from the Master), bypassing a run."""
        existing = self._closest(goal)
        if existing is not None:
            existing.procedure = procedure or existing.procedure
            existing.reinforce()
            self._persist(existing, update=True)
            return existing
        skill = Skill(goal=goal.strip(), tools=list(tools or []), procedure=procedure)
        skill.reinforce()
        self._persist(skill, update=False)
        return skill

    def _distil(self, steps: List[Any], final_answer: str) -> str:
        parts = []
        for s in steps:
            if getattr(s, "tool", None):
                parts.append(f"{s.tool}→{_short(s.observation)}")
        chain = " then ".join(parts) if parts else "answer directly from reasoning"
        tail = f"; result: {_short(final_answer)}" if final_answer else ""
        return chain + tail

    # ---- recall ---- #
    def recall(self, goal: str, k: int = 3) -> List[Skill]:
        if self.store is not None:
            try:
                hits = self.store.recall(goal, k=k, tags=[self.TAG], strengthen=False)
                out: List[Skill] = []
                for rec, _score in hits:
                    data = rec.metadata.get("skill") if isinstance(rec.metadata, dict) else None
                    if isinstance(data, dict):
                        out.append(Skill.from_dict(data))
                if out:
                    return out[:k]
            except Exception:  # noqa: BLE001 — recall is best-effort, never fatal
                pass
        return self._local_recall(goal, k)

    def _local_recall(self, goal: str, k: int) -> List[Skill]:
        q = _tokens(goal)
        scored = []
        for sk in self._local:
            overlap = len(q & _tokens(sk.goal))
            if overlap or not q:
                scored.append((overlap + 0.01 * sk.uses, sk))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [sk for _s, sk in scored[:k]]

    def as_prompt(self, goal: str, k: int = 3) -> str:
        """A prompt-injection block of relevant learned skills (empty when none apply)."""
        skills = self.recall(goal, k=k)
        if not skills:
            return ""
        lines = "\n".join(sk.as_line() for sk in skills)
        return ("\n\nLearned skills from past successes (reuse what worked, adapt as needed):\n"
                + lines)

    # ---- persistence ---- #
    def _persist(self, skill: Skill, *, update: bool) -> None:
        if self.store is None:
            if not update and skill not in self._local:
                self._local.append(skill)
            return
        try:
            from nyxara.memory.store import MemoryType
            self.store.remember(
                skill.procedure or skill.goal,
                mem_type=MemoryType.PROCEDURAL,
                importance=min(1.0, 0.5 + 0.05 * skill.uses),
                tags=[self.TAG],
                metadata={"skill": skill.to_dict(), "skill_id": skill.skill_id})
        except Exception:  # noqa: BLE001
            if skill not in self._local:
                self._local.append(skill)

    def _closest(self, goal: str, *, threshold: float = 0.6) -> Optional[Skill]:
        """Find an existing skill whose goal overlaps ``goal`` enough to be 'the same'."""
        q = _tokens(goal)
        if not q:
            return None
        best, best_sim = None, 0.0
        for sk in self.all():
            t = _tokens(sk.goal)
            if not t:
                continue
            sim = len(q & t) / len(q | t)       # Jaccard
            if sim > best_sim:
                best, best_sim = sk, sim
        return best if best_sim >= threshold else None

    def all(self) -> List[Skill]:
        if self.store is None:
            return list(self._local)
        try:
            hits = self.store.recall("", k=1000, tags=[self.TAG], strengthen=False)
            out: List[Skill] = []
            for rec, _ in hits:
                data = rec.metadata.get("skill") if isinstance(rec.metadata, dict) else None
                if isinstance(data, dict):
                    out.append(Skill.from_dict(data))
            return out or list(self._local)
        except Exception:  # noqa: BLE001
            return list(self._local)

    def __len__(self) -> int:
        return len(self.all())


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set:
    import re
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2}


def _short(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.agency.agent_loop import AgentLoop
    from nyxara.agency.permissions import Capability, RiskTier
    from nyxara.kernel.orchestrator import Candidate, NyxaraCore
    from nyxara.memory.store import MemoryStore

    print("=" * 70)
    print("NYXARA skill-memory self-test")
    print("=" * 70)

    # a scripted reasoner that calls a tool then answers (a learnable 2-step success)
    class Scripted:
        def __init__(self):
            self.n = 0
        def __call__(self, stimulus, focus=None):
            self.n += 1
            if self.n == 1:
                return Candidate(text="compute it", kind="act",
                                 capability=Capability.TOOL_CALL, risk=RiskTier.LOW,
                                 confidence=0.9, belief=0.9, tool="calculate",
                                 tool_args={"expression": "6*7"}, rationale="need product")
            return Candidate(text="The product is 42.", kind="respond",
                             capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                             confidence=0.95, belief=0.95, rationale="done")

    # ---- in-process (no store) ---- #
    skills = SkillMemory()
    core = NyxaraCore(reasoner=Scripted())
    run = AgentLoop(core, max_steps=4).run("What is 6 times 7?")
    learned = skills.capture(run)
    print(f"\ncaptured            : {learned.as_line() if learned else None}")
    assert learned is not None and "calculate" in learned.tools
    assert len(skills) == 1

    # recall on a similar goal surfaces the skill; prompt block is non-empty
    block = skills.as_prompt("times", k=3)
    print(f"as_prompt           : {block!r}")
    assert "calculate" in block

    # capturing the same goal again reinforces (dedup), does not duplicate
    run_again = AgentLoop(NyxaraCore(reasoner=Scripted()), max_steps=4).run("What is 6 times 7?")
    again = skills.capture(run_again)
    assert again is not None and again.uses >= 2 and len(skills) == 1
    print(f"reinforce dedup     : uses={again.uses} total_skills={len(skills)} ✓")

    # an unsuccessful run is not learned from
    class NeverDone:
        def __call__(self, stimulus, focus=None):
            return Candidate(text="spin", kind="act", capability=Capability.TOOL_CALL,
                             risk=RiskTier.LOW, confidence=0.4, belief=0.4, tool="now",
                             tool_args={}, rationale="loop")
    bad = AgentLoop(NyxaraCore(reasoner=NeverDone()), max_steps=2).run("never")
    assert skills.capture(bad) is None
    print("no-learn on failure : ✓")

    # ---- persisted through a real MemoryStore ---- #
    store = MemoryStore()
    persisted = SkillMemory(store=store)
    persisted.capture(AgentLoop(NyxaraCore(reasoner=Scripted())).run("multiply 6 and 7"))
    recalled = persisted.recall("multiply", k=3)
    print(f"persisted recall    : {[s.goal for s in recalled]}")
    assert recalled and any("multiply" in s.goal for s in recalled)

    # direct teaching by the Master
    taught = persisted.learn("greet the Master warmly",
                             procedure="respond with a warm, concise greeting", tools=[])
    assert taught.uses >= 1
    print(f"direct teach        : {taught.as_line()} ✓")

    print("\nALL SELF-TESTS PASSED ✓")
