"""NYXARA · growth/skill_rehearsal.py — never forget a learned skill (🔁, verified rehearsal).

Elastic synapses protect *weights*; this module protects **whole skills**. Every skill NYXARA
induces from demonstrations (:mod:`nyxara.cognition.skill_induction`) carries its own evidence —
the ``input → output`` demos it was verified against. That evidence makes true continual
learning *checkable*: at any time the live skill can be re-run over its stored demos and either
still reproduces them (the skill is intact) or does not (the skill regressed — overwritten by a
worse re-induction, corrupted metadata, a bad hydration). Detection alone is not enough, so the
library also keeps a **known-good snapshot** of every skill that has ever verified, and a
regressed skill is *restored* from its snapshot on the spot.

This is rehearsal in the literal, biological sense — periodically re-practicing old skills so
they stay sharp — and it is entirely NYXARA's own machinery: pure standard library, no external
model, no key. The cycle:

* :meth:`SkillRehearsalLibrary.sync` — snapshot every engine skill that currently reproduces
  all of its demos (known-good by the engine's own verification standard).
* :meth:`SkillRehearsalLibrary.rehearse` — re-run the demos of the next ``k`` skills
  (round-robin, so *every* skill gets its turn no matter how many she learns) through the LIVE
  engine; any mismatch is a detected regression and triggers :meth:`restore`.
* :meth:`SkillRehearsalLibrary.restore` — re-install the known-good snapshot into the engine
  and re-persist it, so the recovered skill also survives the next restart.

Snapshots ride the :class:`~nyxara.memory.store.MemoryStore` (tag ``skill_snapshot``,
procedural memory) when a store is given — the same lifelong persistence path as the skills
themselves — and fall back to in-RAM when it is not. Skills only; the loyalty core is not a
skill and is never touched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

__all__ = ["RehearsalReport", "SkillRehearsalLibrary"]


@dataclass
class RehearsalReport:
    """What one rehearsal pass found: how many skills were re-run, intact, or repaired."""

    rehearsed: int = 0
    verified: int = 0
    regressed: int = 0
    restored: int = 0
    skills: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"rehearsed": self.rehearsed, "verified": self.verified,
                "regressed": self.regressed, "restored": self.restored,
                "skills": list(self.skills)}


class SkillRehearsalLibrary:
    """Keeps known-good snapshots of induced skills and rehearses them against regression."""

    TAG = "skill_snapshot"

    def __init__(self, skill_engine: Any, *, store: Any = None, batch: int = 5,
                 seed: int = 0) -> None:
        self.engine = skill_engine
        self.store = store
        self.batch = max(1, int(batch))
        self.seed = seed
        self._snapshots: Dict[str, Dict[str, Any]] = {}   # name -> known-good skill dict
        self._cursor = 0                                  # round-robin rehearsal position
        self._last_report: Optional[RehearsalReport] = None
        self._total = RehearsalReport()
        self._hydrated = False

    # ---- engine access (best-effort; the engine is a capability, never required) ---- #
    def _skills(self) -> List[Any]:
        try:
            return list(self.engine.skills())
        except Exception:  # noqa: BLE001 — a broken engine means nothing to rehearse
            return []

    @staticmethod
    def _verifies(skill: Any) -> bool:
        """True iff the skill reproduces every one of its own stored demonstrations."""
        demos = list(getattr(skill, "demos", []) or [])
        if not demos:
            return False
        try:
            return all(skill.apply(str(inp)) == str(out) for inp, out in demos)
        except Exception:  # noqa: BLE001 — a crashing skill is a regressed skill
            return False

    # ---- sync: snapshot everything that is currently known-good ---- #
    def sync(self) -> int:
        """Snapshot every engine skill that verifies against all its demos. Returns how many
        snapshots were taken or refreshed."""
        self._hydrate()
        taken = 0
        for skill in self._skills():
            name = getattr(skill, "name", "")
            if not name or not self._verifies(skill):
                continue
            blob = skill.to_dict()
            if self._snapshots.get(name) != blob:
                self._snapshots[name] = blob
                self._persist(name, blob)
                taken += 1
        return taken

    # ---- rehearse: re-run stored demos through the LIVE engine, repair regressions ---- #
    def rehearse(self, k: Optional[int] = None) -> RehearsalReport:
        """Re-verify the next ``k`` snapshotted skills (round-robin) against the live engine.

        A skill that no longer reproduces its known-good demos has *regressed* — it is
        restored from its snapshot immediately, so a learned skill can be corrupted for at
        most one rehearsal interval before it is repaired.
        """
        self._hydrate()
        report = RehearsalReport()
        names = sorted(self._snapshots)
        if not names:
            self._last_report = report
            return report
        k = self.batch if k is None else max(1, int(k))
        k = min(k, len(names))
        live = {getattr(s, "name", ""): s for s in self._skills()}
        for i in range(k):
            name = names[(self._cursor + i) % len(names)]
            snapshot = self._snapshots[name]
            report.rehearsed += 1
            report.skills.append(name)
            skill = live.get(name)
            ok = skill is not None and self._demos_hold(skill, snapshot)
            if ok:
                report.verified += 1
                continue
            report.regressed += 1
            if self.restore(name):
                report.restored += 1
        self._cursor = (self._cursor + k) % len(names)
        self._last_report = report
        self._total.rehearsed += report.rehearsed
        self._total.verified += report.verified
        self._total.regressed += report.regressed
        self._total.restored += report.restored
        return report

    @staticmethod
    def _demos_hold(skill: Any, snapshot: Mapping[str, Any]) -> bool:
        """Check the LIVE skill against the *snapshot's* demos (the known-good evidence)."""
        demos = [d for d in snapshot.get("demos", []) if isinstance(d, (list, tuple))
                 and len(d) == 2]
        if not demos:
            return False
        try:
            return all(skill.apply(str(inp)) == str(out) for inp, out in demos)
        except Exception:  # noqa: BLE001
            return False

    # ---- restore: re-install the known-good snapshot into the live engine ---- #
    def restore(self, name: str) -> bool:
        """Re-install the known-good snapshot of ``name`` into the engine and re-persist."""
        blob = self._snapshots.get(name)
        if blob is None:
            return False
        try:
            from nyxara.cognition.skill_induction import InductiveSkill
            skill = InductiveSkill.from_dict(dict(blob))
            if not self._verifies(skill):   # a snapshot must itself be known-good
                return False
            self.engine._skills[name] = skill
            persist = getattr(self.engine, "_persist", None)
            if callable(persist):
                persist(skill)
            return True
        except Exception:  # noqa: BLE001 — restoration is best-effort, never fatal
            return False

    # ---- reporting ---- #
    def stats(self) -> Dict[str, Any]:
        self._hydrate()
        return {
            "snapshots": len(self._snapshots),
            "batch": self.batch,
            "lifetime": self._total.to_dict(),
            "last": self._last_report.to_dict() if self._last_report else None,
        }

    # ---- persistence (snapshots survive restarts, like the skills themselves) ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {"snapshots": {k: dict(v) for k, v in self._snapshots.items()},
                "cursor": self._cursor, "batch": self.batch}

    def load_dict(self, data: Mapping[str, Any]) -> None:
        snaps = data.get("snapshots", {})
        if isinstance(snaps, dict):
            self._snapshots = {str(k): dict(v) for k, v in snaps.items()
                               if isinstance(v, dict)}
        self._cursor = int(data.get("cursor", 0))
        self.batch = max(1, int(data.get("batch", self.batch)))
        self._hydrated = True

    def _persist(self, name: str, blob: Mapping[str, Any]) -> None:
        if self.store is None:
            return
        try:
            from nyxara.memory.store import MemoryType
            self.store.remember(
                f"skill-snapshot:{name} (known-good, {len(blob.get('demos', []))} demos)",
                mem_type=MemoryType.PROCEDURAL,
                importance=0.7,
                tags=[self.TAG],
                metadata={"snapshot": dict(blob), "name": name, "at": time.time()})
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def _hydrate(self) -> None:
        if self._hydrated:
            return
        self._hydrated = True
        if self.store is None:
            return
        try:
            hits = self.store.recall("", k=1000, tags=[self.TAG], strengthen=False)
            for rec, _ in hits:
                meta = rec.metadata if isinstance(rec.metadata, dict) else {}
                blob = meta.get("snapshot")
                name = meta.get("name")
                if isinstance(blob, dict) and isinstance(name, str) and name:
                    self._snapshots.setdefault(name, blob)
        except Exception:  # noqa: BLE001 — hydration is best-effort
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA skill-rehearsal self-test (never forget a learned skill)")
    print("=" * 70)

    from nyxara.cognition.skill_induction import SkillInductionEngine

    eng = SkillInductionEngine()
    taught = eng.induce("shout", [("hello", "HELLO"), ("go now", "GO NOW"),
                                  ("ok", "OK")])
    assert taught is not None
    lib = SkillRehearsalLibrary(eng, batch=5)

    # sync snapshots the verified skill
    assert lib.sync() == 1
    print("\nsync                : known-good snapshot taken ✓")

    # a healthy engine rehearses clean
    r = lib.rehearse()
    assert r.rehearsed == 1 and r.verified == 1 and r.regressed == 0
    print("rehearse (healthy)  : verified, no regression ✓")

    # sabotage the live skill → rehearsal detects and RESTORES it
    eng._skills["shout"].program.clear()          # the skill now does nothing
    assert eng._skills["shout"].apply("hello") == "hello"
    r = lib.rehearse()
    assert r.regressed == 1 and r.restored == 1
    assert eng._skills["shout"].apply("hello") == "HELLO"
    print("rehearse (damaged)  : regression detected, known-good skill restored ✓")

    # snapshots persist through a real store and hydrate into a fresh library
    from nyxara.memory.store import MemoryStore
    store = MemoryStore()
    e1 = SkillInductionEngine(store.embedder, store=store)
    e1.induce("shout", [("hi", "HI"), ("go", "GO"), ("stop", "STOP")])
    l1 = SkillRehearsalLibrary(e1, store=store)
    assert l1.sync() == 1
    l2 = SkillRehearsalLibrary(e1, store=store)   # fresh library, same store
    assert l2.rehearse().verified == 1
    print("persistence         : snapshot hydrated from the store, verified ✓")

    print(f"\nstats               : {lib.stats()}")
    print("\nALL SELF-TESTS PASSED ✓")
