"""NYXARA · growth/skilltree.py — concrete skills, dependencies, practice & decay (🌳).

Where the developmental curriculum gates broad *stages*, the skill tree is the fine-grained
map of **concrete abilities** and how they depend on one another. Each skill sits in a
prerequisite DAG: you cannot practise a skill until the skills it builds on are good enough.
Proficiency is *earned by practice* (with diminishing returns), *boosted by synergy* with
related skills, and *lost slowly to disuse* (a forgetting curve) — so mastery must be kept up,
not banked once.

The tree answers the question a growing mind keeps asking: **what should I learn next?** Given
the Master's goals, :meth:`SkillTree.recommend` traces the prerequisite paths to those goals,
keeps the skills that are *learnable right now*, and ranks them by leverage (how much they
unlock), value, synergy-readiness, and difficulty — a concrete study plan toward what the
Master needs.

Pure standard library; no external dependency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set

from nyxara.kernel.errors import InvariantViolation

__all__ = [
    "Mastery",
    "Skill",
    "SkillState",
    "SkillTree",
    "build_default_skilltree",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Mastery levels
# --------------------------------------------------------------------------- #
class Mastery(str, Enum):
    LOCKED = "locked"
    NOVICE = "novice"
    COMPETENT = "competent"
    EXPERT = "expert"


# --------------------------------------------------------------------------- #
# Skill & state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Skill:
    name: str
    prerequisites: FrozenSet[str] = frozenset()
    synergies: FrozenSet[str] = frozenset()
    category: str = "general"
    difficulty: float = 0.5          # [0,1] higher = slower to learn
    value: float = 0.5               # [0,1] importance / usefulness
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "prerequisites": sorted(self.prerequisites),
                "synergies": sorted(self.synergies), "category": self.category,
                "difficulty": self.difficulty, "value": self.value}


@dataclass
class SkillState:
    proficiency: float = 0.0
    practices: int = 0
    last_practiced: Optional[float] = None


# --------------------------------------------------------------------------- #
# Skill tree
# --------------------------------------------------------------------------- #
class SkillTree:
    """A prerequisite DAG of skills with practice, synergy and decay dynamics."""

    def __init__(self, *, unlock_threshold: float = 0.5, mastery_threshold: float = 0.8,
                 base_rate: float = 0.3, synergy_weight: float = 0.5,
                 decay_rate: float = 0.02, clock=None) -> None:
        self.unlock_threshold = unlock_threshold
        self.mastery_threshold = mastery_threshold
        self.base_rate = base_rate
        self.synergy_weight = synergy_weight
        self.decay_rate = decay_rate
        self._clock = clock or time.time
        self._skills: Dict[str, Skill] = {}
        self._state: Dict[str, SkillState] = {}

    # ---- definition ---- #
    def add_skill(self, name: str, *, prerequisites: Set[str] = frozenset(),
                  synergies: Set[str] = frozenset(), category: str = "general",
                  difficulty: float = 0.5, value: float = 0.5, description: str = "") -> Skill:
        for p in prerequisites:
            if p not in self._skills:
                raise InvariantViolation(f"prerequisite {p!r} of {name!r} is not defined")
        skill = Skill(name=name, prerequisites=frozenset(prerequisites),
                      synergies=frozenset(synergies), category=category,
                      difficulty=_clamp(difficulty), value=_clamp(value),
                      description=description)
        self._skills[name] = skill
        self._state[name] = SkillState()
        return skill

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    # ---- proficiency & status ---- #
    def proficiency(self, name: str) -> float:
        st = self._state.get(name)
        return st.proficiency if st else 0.0

    def is_unlocked(self, name: str) -> bool:
        skill = self._require(name)
        return all(self.proficiency(p) >= self.unlock_threshold for p in skill.prerequisites)

    def mastery(self, name: str) -> Mastery:
        if not self.is_unlocked(name):
            return Mastery.LOCKED
        p = self.proficiency(name)
        if p >= self.mastery_threshold:
            return Mastery.EXPERT
        if p >= self.unlock_threshold:
            return Mastery.COMPETENT
        return Mastery.NOVICE

    # ---- synergy ---- #
    def synergy_multiplier(self, name: str) -> float:
        skill = self._require(name)
        if not skill.synergies:
            return 1.0
        mean = sum(self.proficiency(s) for s in skill.synergies) / len(skill.synergies)
        return 1.0 + self.synergy_weight * mean

    # ---- practice (earn proficiency) ---- #
    def practice(self, name: str, *, quality: float = 1.0, now: Optional[float] = None
                 ) -> float:
        skill = self._require(name)
        if not self.is_unlocked(name):
            raise InvariantViolation(
                f"skill {name!r} is locked; master its prerequisites first",
                context={"prerequisites": sorted(skill.prerequisites)})
        st = self._state[name]
        rate = self.base_rate * (1.0 - 0.5 * skill.difficulty) * self.synergy_multiplier(name)
        gain = rate * _clamp(quality) * (1.0 - st.proficiency)   # diminishing returns
        st.proficiency = _clamp(st.proficiency + gain)
        st.practices += 1
        st.last_practiced = now if now is not None else self._clock()
        return st.proficiency

    # ---- decay (forget with disuse) ---- #
    def decay(self, *, now: Optional[float] = None) -> Dict[str, float]:
        now = now if now is not None else self._clock()
        dropped: Dict[str, float] = {}
        for name, st in self._state.items():
            if st.last_practiced is None or st.proficiency <= 0:
                continue
            dt = max(0.0, now - st.last_practiced)
            if dt <= 0:
                continue
            new = st.proficiency * math.exp(-self.decay_rate * dt)
            if new < st.proficiency:
                dropped[name] = round(st.proficiency - new, 6)
                st.proficiency = new
                st.last_practiced = now
        return dropped

    # ---- availability ---- #
    def unlocked_skills(self) -> List[str]:
        return [n for n in self._skills if self.is_unlocked(n)]

    def learnable_now(self) -> List[str]:
        """Unlocked skills that are not yet mastered (worth practising)."""
        return [n for n in self._skills
                if self.is_unlocked(n) and self.proficiency(n) < self.mastery_threshold]

    # ---- prerequisite paths ---- #
    def prerequisite_closure(self, name: str) -> Set[str]:
        """All skills (transitively) required for ``name``."""
        out: Set[str] = set()
        stack = list(self._require(name).prerequisites)
        while stack:
            p = stack.pop()
            if p in out:
                continue
            out.add(p)
            stack.extend(self._skills[p].prerequisites)
        return out

    def prerequisite_path(self, name: str) -> List[str]:
        """An ordered, learnable sequence of not-yet-mastered skills to reach ``name``."""
        needed = self.prerequisite_closure(name) | {name}
        needed = {s for s in needed if self.proficiency(s) < self.mastery_threshold}
        # topological order by prerequisite depth
        order: List[str] = []
        added: Set[str] = set()
        remaining = set(needed)
        while remaining:
            progress = False
            for s in sorted(remaining, key=lambda x: self._skills[x].name):
                deps = self._skills[s].prerequisites & needed
                if deps <= added:
                    order.append(s)
                    added.add(s)
                    remaining.discard(s)
                    progress = True
            if not progress:   # defensive — shouldn't happen in a DAG
                order.extend(sorted(remaining))
                break
        return order

    # ---- recommendation: what to learn next ---- #
    def recommend(self, goals: Sequence[str], *, k: int = 3) -> List[Dict[str, Any]]:
        """Rank the best skills to practise *now* toward the Master's goals."""
        needed: Set[str] = set()
        for g in goals:
            if g in self._skills:
                needed |= self.prerequisite_closure(g) | {g}
        needed = {s for s in needed if self.proficiency(s) < self.mastery_threshold}
        # leverage: how many needed skills depend (transitively) on s
        leverage: Dict[str, int] = {s: 0 for s in needed}
        for s in needed:
            for dep in self.prerequisite_closure(s):
                if dep in leverage:
                    leverage[dep] += 1
        ranked: List[Dict[str, Any]] = []
        for s in needed:
            if not self.is_unlocked(s):
                continue   # can only recommend something learnable now
            skill = self._skills[s]
            score = (skill.value + 0.3 * leverage[s] + 0.2 * (self.synergy_multiplier(s) - 1.0)
                     - 0.2 * skill.difficulty)
            ranked.append({"skill": s, "score": round(score, 4),
                           "leverage": leverage[s], "proficiency": round(self.proficiency(s), 3)})
        ranked.sort(key=lambda r: -r["score"])
        return ranked[:k]

    def _require(self, name: str) -> Skill:
        skill = self._skills.get(name)
        if skill is None:
            raise InvariantViolation(f"no such skill {name!r}")
        return skill

    def report(self) -> Dict[str, Any]:
        by_mastery: Dict[str, int] = {}
        for n in self._skills:
            m = self.mastery(n).value
            by_mastery[m] = by_mastery.get(m, 0) + 1
        return {"skills": len(self._skills), "by_mastery": by_mastery,
                "unlocked": len(self.unlocked_skills()),
                "learnable_now": len(self.learnable_now())}


# --------------------------------------------------------------------------- #
# Default NYXARA skill tree
# --------------------------------------------------------------------------- #
def build_default_skilltree(**kw: Any) -> SkillTree:
    t = SkillTree(**kw)
    t.add_skill("perception", category="core", difficulty=0.3, value=0.8)
    t.add_skill("memory", prerequisites={"perception"}, category="core",
                difficulty=0.4, value=0.8)
    t.add_skill("language", prerequisites={"perception"}, category="core",
                difficulty=0.4, value=0.9)
    t.add_skill("reasoning", prerequisites={"language", "memory"}, category="mind",
                difficulty=0.6, value=0.9)
    t.add_skill("social", prerequisites={"language", "memory"}, synergies={"language"},
                category="social", difficulty=0.5, value=0.7)
    t.add_skill("planning", prerequisites={"reasoning"}, synergies={"memory"},
                category="agency", difficulty=0.6, value=0.8)
    t.add_skill("tool_use", prerequisites={"reasoning"}, category="agency",
                difficulty=0.5, value=0.7)
    t.add_skill("security", prerequisites={"reasoning", "tool_use"}, category="guard",
                difficulty=0.7, value=0.9)
    t.add_skill("self_reflection", prerequisites={"reasoning", "memory"},
                synergies={"reasoning"}, category="mind", difficulty=0.7, value=0.8)
    return t


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA skill-tree self-test")
    print("=" * 70)

    now = {"t": 0.0}
    t = build_default_skilltree(clock=lambda: now["t"], decay_rate=0.05)

    # locked: cannot practise 'language' before 'perception' is good enough
    try:
        t.practice("language")
        raise SystemExit("ERROR: locked skill should refuse practice")
    except InvariantViolation:
        print("\nprereq gate         : 'language' locked until 'perception' is learned ✓")
    assert t.mastery("language") is Mastery.LOCKED

    # practise perception to competence -> unlocks its dependents
    for _ in range(8):
        t.practice("perception")
    print(f"perception          : prof={t.proficiency('perception'):.2f} "
          f"mastery={t.mastery('perception').value}")
    assert t.is_unlocked("language") and t.is_unlocked("memory")

    # SYNERGY: 'social' learns faster when 'language' is strong
    for _ in range(10):
        t.practice("language")
    for _ in range(10):
        t.practice("memory")
    base_mult = SkillTree().synergy_multiplier  # n/a; compare directly
    syn = t.synergy_multiplier("social")
    print(f"\nsynergy mult social : {syn:.2f} (boosted by strong language)")
    assert syn > 1.0

    # DIMINISHING RETURNS: gains shrink as proficiency rises
    fresh = build_default_skilltree(clock=lambda: now["t"])
    for _ in range(6):
        fresh.practice("perception")
    g1 = fresh.practice("perception")
    g2 = fresh.practice("perception")
    assert (g2 - g1) < 0.05   # later practice adds little
    print("diminishing returns : late practice yields small gains ✓")

    # DECAY: disuse erodes proficiency over time
    before = t.proficiency("memory")
    now["t"] += 30.0
    dropped = t.decay()
    print(f"\ndecay               : memory {before:.2f} -> {t.proficiency('memory'):.2f} "
          f"(dropped {dropped.get('memory', 0):.3f})")
    assert t.proficiency("memory") < before

    # PREREQUISITE PATH to a deep skill
    path = t.prerequisite_path("security")
    print(f"\npath to security    : {path}")
    assert path[-1] == "security" and "reasoning" in path

    # RECOMMENDATION: what to learn next toward the goal of 'planning'
    t2 = build_default_skilltree(clock=lambda: now["t"])
    for _ in range(8):
        t2.practice("perception")
    recs = t2.recommend(["planning"], k=3)
    print(f"\nrecommend->planning : {[(r['skill'], r['score']) for r in recs]}")
    assert recs   # something learnable now is suggested
    # the suggestions are all currently unlocked and on the path to planning
    needed = t2.prerequisite_closure("planning") | {"planning"}
    assert all(r["skill"] in needed and t2.is_unlocked(r["skill"]) for r in recs)

    print(f"\nreport              : {t.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
