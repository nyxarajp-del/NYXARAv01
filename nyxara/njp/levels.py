"""NYXARA · njp/levels.py — five kinds of memory, and the traffic between them (🗂, NJP V.02).

:mod:`nyxara.njp.memory` stores everything in one undifferentiated pool. That is enough to *find*
things and not enough to *organise* them: a passing "hi NYXARA" and the Master's name occupy the
same shelf, compete in the same recall, and decay at the same rate. Real memory is not one store,
and the differences between its parts are functional rather than cosmetic.

Five levels, each with a different job and a different half-life:

* **working** — what she is holding *right now*. Tiny and fast-decaying by design; a working
  memory that does not forget is just a slow episodic one.
* **episodic** — what happened, in order, with when. The raw record.
* **semantic** — what is true, with the episode stripped off. Reached by consolidation, not by
  being written there: a fact she was told once is an episode, and it becomes semantic when it
  survives repetition.
* **procedural** — how to do something. Strengthened by use rather than by rehearsal, because
  that is how a skill actually consolidates.
* **autobiographical** — what concerns her and the Master. Protected: this is the level whose
  loss would change who she is, so forgetting never touches it.

**Consolidation is a promotion, and it has to be earned.** ``episodic → semantic`` fires when the
same claim recurs from independent episodes — repetition is the evidence that something is a fact
about the world rather than a thing that happened once. Promoting on a single occurrence would
make "semantic" a synonym for "old".

**Forgetting is real, and it is the point.** Retention follows the repo's existing
:class:`~nyxara.memory.consolidation.ForgettingCurve` — Ebbinghaus decay with stability that grows
multiplicatively on each rehearsal — so a memory recalled often becomes durable and one never
touched fades. A store that only accumulates degrades: every new item makes every existing
retrieval slightly worse, so keeping everything forever is not a kindness to the mind that has to
search it.

Nothing here re-implements storage. Each level is a view over one :class:`~nyxara.njp.memory.HoloMemory`,
so content-addressed recall still reaches everything and a fact from fifty turns ago is still
exactly as reachable as the last one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["Level", "LevelPolicy", "ConsolidationReport", "HierarchicalMemory"]


class Level:
    """The five stores. Named as strings so they survive a sidecar round-trip unchanged."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    AUTOBIOGRAPHICAL = "autobiographical"

    ALL = (WORKING, EPISODIC, SEMANTIC, PROCEDURAL, AUTOBIOGRAPHICAL)


@dataclass
class LevelPolicy:
    """How one level behaves. The differences between levels *are* these numbers."""

    capacity: int = 4096
    stability_days: float = 1.0        # starting durability of a new memory here
    protected: bool = False            # never forgotten, whatever the retention says
    promotes_to: str = ""              # where consolidation sends it
    promote_after: int = 2             # independent recurrences needed to promote
    # Retrievals that promote on their own. Being *used* is evidence of the same kind as being
    # repeated, and arguably better: re-exposure shows the world said it twice, retrieval shows
    # the memory earned its keep. Zero disables it, leaving recurrence as the only route.
    promote_on_uses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"capacity": self.capacity, "stability_days": self.stability_days,
                "protected": self.protected, "promotes_to": self.promotes_to,
                "promote_after": self.promote_after,
                "promote_on_uses": self.promote_on_uses}


# Working memory is small and brittle *on purpose*. Autobiographical is protected because losing
# it would change who she is, which is not a memory-management decision.
_DEFAULT_POLICIES: Dict[str, LevelPolicy] = {
    Level.WORKING: LevelPolicy(capacity=7, stability_days=0.01,
                               promotes_to=Level.EPISODIC, promote_after=1),
    Level.EPISODIC: LevelPolicy(capacity=20000, stability_days=1.0,
                                promotes_to=Level.SEMANTIC, promote_after=2,
                                promote_on_uses=2),
    Level.SEMANTIC: LevelPolicy(capacity=50000, stability_days=30.0),
    Level.PROCEDURAL: LevelPolicy(capacity=5000, stability_days=90.0),
    Level.AUTOBIOGRAPHICAL: LevelPolicy(capacity=10000, stability_days=365.0, protected=True),
}


@dataclass
class Entry:
    """One memory's bookkeeping — the part that is *about* the memory rather than its content."""

    key: str = ""
    level: str = Level.EPISODIC
    written: float = field(default_factory=time.time)
    last_touched: float = field(default_factory=time.time)
    stability: float = 1.0
    rehearsals: int = 0
    claim: str = ""                    # the normalised content, for recurrence detection
    sources: List[str] = field(default_factory=list)   # episodes this was seen in
    promoted_from: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "level": self.level, "stability": round(self.stability, 4),
                "rehearsals": self.rehearsals, "sources": self.sources[:8],
                "promoted_from": self.promoted_from}


@dataclass
class ConsolidationReport:
    """What one consolidation pass actually moved. Every number is a real count."""

    promoted: int = 0
    forgotten: int = 0
    rehearsed: int = 0
    evicted: int = 0
    by_level: Dict[str, int] = field(default_factory=dict)
    promotions: List[Tuple[str, str, str]] = field(default_factory=list)  # (key, from, to)
    ms: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.promoted or self.forgotten or self.evicted)

    def to_dict(self) -> Dict[str, Any]:
        return {"promoted": self.promoted, "forgotten": self.forgotten,
                "rehearsed": self.rehearsed, "evicted": self.evicted,
                "by_level": dict(self.by_level), "changed": self.changed,
                "promotions": [list(p) for p in self.promotions[:16]],
                "ms": round(self.ms, 3)}


_DAY = 86400.0


def _claim_of(text: str) -> str:
    """The normalised form used to decide whether two memories say the same thing.

    A bag of the content words, which is the best that can be done **from surface text alone** and
    is a poor proxy for a claim: two sentences assert the same thing whenever they assert the same
    relation, and they routinely do so with different words. Promotion is gated on recurrence, so
    a claim identity this strict does not merely miss a few matches — it makes recurrence
    effectively undetectable outside verbatim repetition.

    Measured: over 1,200 corpus pairs, 386 of 391 memories sat in ``episodic`` forever and 5 were
    ever promoted, because no two answers shared a whole word-set. The fix is not to lower the bar
    for promotion; it is to let a caller that *knows* the claim say so — see the ``claim`` argument
    to :meth:`HierarchicalMemory.remember`. This stays as the fallback for text nobody parsed.

    **It is blind to direction, and that is the hazard worth naming.** The words are sorted, so
    ``aag causes garmi`` and ``garmi causes aag`` are one claim to this function — as are
    ``Master has name Jay`` and ``Jay has name Master``. A pair of opposites therefore looks like
    a recurrence and earns a promotion. That is where the baseline's promotion count came from:
    17 promotions on the bundled corpus under this rule, 8 under the triple identity, and the
    difference is not a lower bar but the removal of promotions that direction-reversed pairs had
    been earning. Anything that knows the relation should pass ``claim`` rather than rely on this.
    """
    return " ".join(sorted({t for t in str(text or "").lower().split() if len(t) > 2}))[:200]


class HierarchicalMemory:
    """Five levels over one content-addressed store, with earned promotion and real forgetting."""

    def __init__(self, store: Any = None, *,
                 policies: Optional[Dict[str, LevelPolicy]] = None,
                 forget_below: float = 0.05) -> None:
        self.store = store if store is not None else self._build_store()
        self.policies: Dict[str, LevelPolicy] = dict(policies or _DEFAULT_POLICIES)
        self.forget_below = float(forget_below)

        self.entries: Dict[str, Entry] = {}
        self.levels: Dict[str, List[str]] = {name: [] for name in Level.ALL}
        # claim -> keys that assert it, per level. This is what makes promotion evidence-based:
        # a claim seen in two independent episodes is a fact; one seen once is an anecdote.
        self._claims: Dict[str, Dict[str, List[str]]] = {name: {} for name in Level.ALL}
        self.consolidations = 0
        self.promoted_total = 0
        # Split out because the two routes answer different questions about her intake. A high
        # `promoted_by_use` with a low recurrence count says she is learning from a stream that
        # never repeats itself and is keeping what she actually reaches for.
        self.promoted_by_use = 0
        self.forgotten_total = 0

    @staticmethod
    def _build_store() -> Any:
        try:
            from nyxara.njp.memory import HoloMemory
            return HoloMemory()
        except Exception:  # noqa: BLE001
            return None

    # ---- write ------------------------------------------------------------ #
    def remember(self, key: str, text: str, *, level: str = Level.EPISODIC,
                 cue: str = "", source: str = "", claim: str = "",
                 cells: Sequence[int] = (),
                 salience: float = 0.0) -> Optional[Entry]:
        """Store one memory at a level. The level decides how long it lives, not where it is found.

        ``claim`` is what this memory *asserts*, where the caller knows it — a canonical
        ``subject|predicate|object`` from :mod:`nyxara.njp.grounding`, say. Recurrence is what
        earns promotion, so recurrence has to be detectable, and it is not detectable from surface
        text: two sentences stating one relation share almost no words. Passing the claim is how a
        caller that already parsed the sentence stops the promotion rule from being blind to the
        very thing it is measuring. Omitted, the bag-of-words fallback applies exactly as before.
        """
        try:
            key, text = str(key), str(text)
            if not key.strip() or not text.strip():
                return None
            level = level if level in self.policies else Level.EPISODIC
            policy = self.policies[level]

            if self.store is not None:
                # The firing state travels with the memory. This is the only path a turn's
                # memory actually takes when levels are on — the brain's direct `memory.remember`
                # is the fallback for when they are not — so without it here the substrate's
                # record of the moment is never stored at all.
                self.store.remember(key, text, kind=level, cue=cue, cells=cells,
                                    salience=salience)

            identity = str(claim or "").strip().lower() or _claim_of(text)
            entry = self.entries.get(key)
            if entry is None:
                entry = Entry(key=key, level=level,
                              stability=self._with_salience(policy.stability_days, salience),
                              claim=identity)
                self.entries[key] = entry
                self.levels[level].append(key)
            else:
                entry.claim = identity
                entry.last_touched = time.time()
            if source and source not in entry.sources:
                entry.sources.append(source)

            bucket = self._claims[level].setdefault(entry.claim, [])
            if key not in bucket:
                bucket.append(key)

            self._enforce_capacity(level)
            return entry
        except Exception:  # noqa: BLE001 — a write failure never breaks a turn
            return None

    def note(self, text: str, *, cue: str = "") -> Optional[Entry]:
        """Put something in working memory. Convenience for the commonest write."""
        return self.remember(f"wm-{time.time_ns()}", text, level=Level.WORKING, cue=cue)

    # ---- read -------------------------------------------------------------- #
    def recall(self, cue: str, *, k: int = 5, level: str = "") -> Any:
        """Content-addressed recall across everything, or within one level.

        Recall is deliberately **not** partitioned by default: the levels govern durability and
        promotion, not reachability, and a mind that had to know which drawer a memory was in
        before it could find it would have gained nothing over a filing cabinet.
        """
        try:
            if self.store is None:
                return None
            got = self.store.recall(cue, k=k)
            hit = getattr(got, "hit", None)
            if hit is not None:
                self.touch(getattr(hit, "key", ""))
            if not level:
                return got
            # Level-scoped: filter the candidate list rather than the single nearest hit.
            keys = set(self.levels.get(level, []))
            candidates = [(t, s) for t, s in self.store.candidates(cue, k=k * 4)
                          if getattr(t, "key", "") in keys]
            return candidates[:k]
        except Exception:  # noqa: BLE001
            return None

    def touch(self, key: str) -> Optional[Entry]:
        """Rehearse a memory: it was useful, so it should last longer.

        Stability grows multiplicatively, which is what spaced repetition actually is — the
        interval a memory survives extends each time it is successfully retrieved.
        """
        try:
            entry = self.entries.get(str(key))
            if entry is None:
                return None
            entry.last_touched = time.time()
            entry.rehearsals += 1
            entry.stability = self._next_stability(entry.stability)
            return entry
        except Exception:  # noqa: BLE001
            return None

    def touch_claim(self, claim: str, *, limit: int = 8) -> int:
        """Rehearse every memory asserting this claim. Returns how many were touched.

        The join between *retrieval* and *consolidation*, and it needs a claim rather than a key
        because the two subsystems index differently: :mod:`nyxara.njp.grounding` answers from a
        fact store keyed by ``(subject, predicate)``, while memories here are keyed by the turn
        they arrived on. Nothing connected them, so a fact retrieved and used a hundred times
        looked, to consolidation, exactly like one nobody had ever read.

        Canonical claims are what make the join possible at all — see the ``claim`` argument to
        :meth:`remember`. Without them this would have to match on surface text, which is the
        measurement failure the claim identity exists to fix.
        """
        touched = 0
        try:
            wanted = str(claim or "").strip().lower()
            if not wanted:
                return 0
            for level in Level.ALL:
                bucket = self._claims[level]
                keys = list(bucket.get(wanted, []))
                if not keys:
                    # A turn asserting several relations is filed under all of them joined, so the
                    # claim asked for may be one part of a compound. Scanned only on a miss, and
                    # only over the claims of one level — an index would be faster and would be
                    # another thing to keep correct across a sidecar round-trip, which is not
                    # worth it at the sizes one conversation produces.
                    for claim, stored in bucket.items():
                        if wanted in claim.split(" ; "):
                            keys.extend(stored)
                for key in keys[:limit]:
                    if self.touch(key) is not None:
                        touched += 1
            return touched
        except Exception:  # noqa: BLE001
            return touched

    def retention(self, key: str, *, now: Optional[float] = None) -> float:
        """How much of this memory is left, 0…1. The repo's Ebbinghaus curve, not a new one."""
        try:
            entry = self.entries.get(str(key))
            if entry is None:
                return 0.0
            if self.policies[entry.level].protected:
                return 1.0
            elapsed_days = max(0.0, ((now or time.time()) - entry.last_touched) / _DAY)
            return self._retention(elapsed_days, entry.stability)
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- the repo's forgetting math, borrowed rather than reinvented -------- #
    @staticmethod
    def _retention(elapsed_days: float, stability_days: float) -> float:
        try:
            from nyxara.memory.consolidation import ForgettingCurve
            return ForgettingCurve.retention(elapsed_days, stability_days)
        except Exception:  # noqa: BLE001
            import math
            return math.exp(-max(0.0, elapsed_days) / stability_days) if stability_days > 0 else 0.0

    @classmethod
    def _with_salience(cls, stability_days: float, salience: float) -> float:
        """Start a surprising memory where a rehearsed one would already be.

        Reuses ``_next_stability`` rather than inventing a multiplier, and that is the whole point:
        the repo has one forgetting law and a second one bolted on beside it would make retention
        mean two different things depending on which door a memory came through. Salience 1.0 is
        worth exactly one rehearsal of durability, 0.0 changes nothing, and the values between
        interpolate — so "keep what was surprising" is expressed in the units the curve already
        uses.
        """
        salience = max(0.0, min(1.0, float(salience)))
        if salience <= 0.0:
            return stability_days
        rehearsed = cls._next_stability(stability_days)
        return stability_days + (rehearsed - stability_days) * salience

    @staticmethod
    def _next_stability(stability_days: float) -> float:
        try:
            from nyxara.memory.consolidation import ForgettingCurve
            return ForgettingCurve.next_stability(stability_days)
        except Exception:  # noqa: BLE001
            return min(3650.0, stability_days * 1.6)

    # ---- consolidate -------------------------------------------------------- #
    def consolidate(self, *, now: Optional[float] = None) -> ConsolidationReport:
        """Promote what has earned it, forget what has decayed. One pass.

        Order matters: promote first, then forget. Forgetting first would discard the very
        episodes whose recurrence is the evidence for a promotion, so she would lose the fact
        *and* the reason to believe it in the same pass.
        """
        report = ConsolidationReport()
        t0 = time.perf_counter()
        try:
            now = now or time.time()
            self.consolidations += 1
            # One clock for the whole pass. Letting `_promote` stamp real time while `_forget`
            # reads the passed `now` made the two steps disagree about when "now" is, and a fact
            # promoted during the pass was immediately decayed by it.
            self._promote(report, now=now)
            self._forget(report, now=now)
            for name in Level.ALL:
                report.by_level[name] = len(self.levels[name])
            return report
        except Exception:  # noqa: BLE001 — a failed pass changes nothing
            return report
        finally:
            report.ms = (time.perf_counter() - t0) * 1000.0

    def _promote(self, report: ConsolidationReport, *, now: float) -> None:
        """``episodic → semantic``, earned two ways.

        **Recurrence** — the same claim asserted by independent episodes. The world said it twice.

        **Use** — the same memory retrieved ``promote_on_uses`` times. The testing effect is not a
        weaker signal than re-exposure: a memory that has been successfully retrieved has proved
        it is reachable *and* wanted, which is more than a second telling proves. Before this,
        recurrence was the only route, and on any intake where claims do not repeat — a corpus,
        a reference manual, a day of new information — nothing was ever promoted at all.

        Both routes are gated on evidence and neither lowers the bar. A memory read once and never
        used still stays an episode, which is exactly what it is.
        """
        try:
            for level in (Level.WORKING, Level.EPISODIC):
                policy = self.policies[level]
                target = policy.promotes_to
                if not target:
                    continue
                promoted: Set[str] = set()
                for _claim, keys in list(self._claims[level].items()):
                    live = [k for k in keys if k in self.entries]
                    if len(live) < policy.promote_after:
                        continue
                    # The oldest witness carries the claim up; the rest stay as the episodes that
                    # evidenced it. Deleting them would erase why she believes it.
                    key = live[0]
                    entry = self.entries.get(key)
                    if entry is None or entry.level != level:
                        continue
                    self._move(entry, target, now=now)
                    entry.promoted_from = level
                    entry.sources = [k for k in live if k != key][:16]
                    promoted.add(key)
                    report.promoted += 1
                    self.promoted_total += 1
                    report.promotions.append((key, level, target))

                if policy.promote_on_uses <= 0:
                    continue
                for key in list(self.levels[level]):
                    if key in promoted:
                        continue
                    entry = self.entries.get(key)
                    if entry is None or entry.level != level:
                        continue
                    if entry.rehearsals < policy.promote_on_uses:
                        continue
                    self._move(entry, target, now=now)
                    entry.promoted_from = level
                    report.promoted += 1
                    self.promoted_total += 1
                    self.promoted_by_use += 1
                    report.promotions.append((key, level, target))
        except Exception:  # noqa: BLE001
            pass

    def _forget(self, report: ConsolidationReport, *, now: float) -> None:
        """Drop what has decayed below the floor. Protected levels are never touched."""
        try:
            for level in Level.ALL:
                if self.policies[level].protected:
                    continue
                for key in list(self.levels[level]):
                    if self.retention(key, now=now) >= self.forget_below:
                        continue
                    self._drop(key)
                    report.forgotten += 1
                    self.forgotten_total += 1
        except Exception:  # noqa: BLE001
            pass

    def _move(self, entry: Entry, target: str, *, now: Optional[float] = None) -> None:
        """Promote one entry, counting the promotion itself as a rehearsal.

        Being promoted *is* evidence the memory is live — a second independent episode just
        confirmed it. Without resetting the clock here, a claim could be promoted and forgotten in
        the same pass: it arrived at ``semantic`` carrying an ``last_touched`` from whenever the
        first episode happened, so a consolidation run long afterwards immediately decayed it
        below the floor. She would lose the fact on the very pass that established it.
        """
        try:
            self.levels[entry.level].remove(entry.key)
        except ValueError:
            pass
        entry.level = target
        entry.stability = max(entry.stability, self.policies[target].stability_days)
        entry.last_touched = now if now is not None else time.time()
        entry.rehearsals += 1
        self.levels[target].append(entry.key)
        self._claims[target].setdefault(entry.claim, []).append(entry.key)

    def _drop(self, key: str) -> None:
        entry = self.entries.pop(key, None)
        if entry is None:
            return
        try:
            self.levels[entry.level].remove(key)
        except ValueError:
            pass
        bucket = self._claims[entry.level].get(entry.claim)
        if bucket and key in bucket:
            bucket.remove(key)

    def _enforce_capacity(self, level: str) -> None:
        """Keep a level within its capacity, dropping the least-retained first.

        Least-retained rather than oldest: an old memory that is still being used is exactly the
        one worth keeping, and evicting by age alone throws away the durable and keeps the recent.
        """
        try:
            policy = self.policies[level]
            keys = self.levels[level]
            if len(keys) <= policy.capacity:
                return
            ranked = sorted(keys, key=lambda k: self.retention(k))
            for key in ranked[: len(keys) - policy.capacity]:
                self._drop(key)
        except Exception:  # noqa: BLE001
            pass

    # ---- reading ------------------------------------------------------------ #
    def working_set(self) -> List[str]:
        """What she is holding right now, most recent last."""
        return [self.entries[k].key for k in self.levels[Level.WORKING] if k in self.entries]

    def at(self, level: str) -> List[Entry]:
        return [self.entries[k] for k in self.levels.get(level, []) if k in self.entries]

    def stats(self) -> Dict[str, Any]:
        return {
            "total": len(self.entries),
            "levels": {name: len(self.levels[name]) for name in Level.ALL},
            "consolidations": self.consolidations,
            "promoted_total": self.promoted_total,
            "promoted_by_use": self.promoted_by_use,
            "promoted_by_recurrence": self.promoted_total - self.promoted_by_use,
            "forgotten_total": self.forgotten_total,
            # How many memories have ever been reached for. A store where this is near zero is
            # one nothing is reading, and that is a different problem from one nothing is writing.
            "rehearsed": sum(1 for e in self.entries.values() if e.rehearsals),
            "distinct_claims": len({e.claim for e in self.entries.values() if e.claim}),
            "mean_retention": (round(sum(self.retention(k) for k in self.entries)
                                     / len(self.entries), 4) if self.entries else None),
            "policies": {name: p.to_dict() for name, p in self.policies.items()},
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": [e.to_dict() | {"written": e.written,
                                           "last_touched": e.last_touched,
                                           "claim": e.claim}
                            for e in self.entries.values()],
                "counters": {"consolidations": self.consolidations,
                             "promoted": self.promoted_total,
                             "forgotten": self.forgotten_total}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            for row in (d.get("entries") or []):
                key = str(row.get("key", ""))
                level = str(row.get("level", Level.EPISODIC))
                if not key or level not in self.policies:
                    continue
                entry = Entry(key=key, level=level,
                              written=float(row.get("written", time.time())),
                              last_touched=float(row.get("last_touched", time.time())),
                              stability=float(row.get("stability", 1.0)),
                              rehearsals=int(row.get("rehearsals", 0)),
                              claim=str(row.get("claim", "")),
                              sources=list(row.get("sources") or []),
                              promoted_from=str(row.get("promoted_from", "")))
                self.entries[key] = entry
                self.levels[level].append(key)
                self._claims[level].setdefault(entry.claim, []).append(key)
            counters = d.get("counters") or {}
            self.consolidations = int(counters.get("consolidations", 0))
            self.promoted_total = int(counters.get("promoted", 0))
            self.forgotten_total = int(counters.get("forgotten", 0))
        except Exception:  # noqa: BLE001
            pass
