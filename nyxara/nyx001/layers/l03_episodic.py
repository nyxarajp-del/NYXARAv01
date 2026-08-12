"""NYXARA · nyx001/layers/l03_episodic.py — Layer 3, episodic memory (📼, not a database).

The charter is emphatic that this is "an active cognition modifier, not a database", storing
*what, when, context, prediction error, and future relevance*. The difference between those two
things is what this module is about.

A database answers the query you give it. This store decides **what is worth revisiting**, and
that decision changes what the world model learns next. Three mechanisms make it active:

* **Prioritised replay.** :meth:`EpisodicMemory.replay` samples by :attr:`Experience.priority` —
  surprise now, usefulness later, damped by how often an episode has already been replayed. The
  damping is the part that matters: undamped prioritised replay draws the highest-error sample
  until it is memorised, and the rest of experience starves.
* **Learned future relevance.** ``future_relevance`` is not assigned at write time; it cannot be,
  because whether a memory turns out to matter is not knowable when it is laid down. It is
  credited *backwards* by :meth:`credit`, when a later episode turns out to have depended on it.
* **Content-addressed recall.** :meth:`recall` retrieves by cosine similarity over context
  vectors, not by key. There is no token window here; there is also no claim of infinite memory —
  capacity is real, eviction is real, and :meth:`stats` reports both.

Eviction is by lowest priority, and it is honest: a memory that was never surprising and never
turned out to matter is exactly the one to lose first. :attr:`evicted` counts what was forgotten,
so "she remembers everything" is never accidentally implied by silence.

Honest: recall quality is bounded by Layer 1's encoder, which at birth is a hash sketch. Two
episodes that a human would call similar may not retrieve each other until Layer 4 has formed a
concept linking them. Low-confidence recalls report their own similarity rather than presenting a
weak match as a strong one.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from nyxara.nyx001.layers.types import Experience

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["Recall", "EpisodicMemory"]


class Recall:
    """A retrieval, carrying its own similarity so a weak match cannot pose as a strong one."""

    __slots__ = ("episodes", "similarities")

    def __init__(self, episodes: List[Experience], similarities: List[float]) -> None:
        self.episodes = episodes
        self.similarities = similarities

    @property
    def best(self) -> Optional[Experience]:
        return self.episodes[0] if self.episodes else None

    @property
    def confidence(self) -> float:
        """The top similarity, in ``[0, 1]``. Zero when nothing was retrieved — truthfully zero."""
        return float(max(self.similarities)) if self.similarities else 0.0

    @property
    def is_weak(self) -> bool:
        return self.confidence < 0.35

    def to_dict(self) -> Dict[str, Any]:
        return {"n": len(self.episodes), "confidence": round(self.confidence, 4),
                "weak": self.is_weak,
                "episodes": [e.to_dict() for e in self.episodes[:5]]}


class EpisodicMemory:
    """Layer 3. Stores episodes, prioritises them by surprise, and credits relevance backwards."""

    def __init__(self, substrate: Any, *, capacity: int = 8192, replay_batch: int = 8,
                 credit_window: int = 32, credit_rate: float = 0.15) -> None:
        self.substrate = substrate
        self.width = int(getattr(substrate, "width", 256))
        self.capacity = int(capacity)
        self.replay_batch = int(replay_batch)
        self.credit_window = int(credit_window)
        self.credit_rate = float(credit_rate)
        self._store: Dict[str, Experience] = {}
        self._order: List[str] = []
        self.evicted = 0
        self.writes = 0
        self._rng = getattr(substrate, "stream", lambda _n: None)("episodic")

    # ---- write ---- #
    def remember(self, what: str, *, context: Any = None, prediction_error: float = 0.0,
                 action: Optional[str] = None, outcome: Optional[str] = None,
                 key: str = "") -> Optional[Experience]:
        """Lay down one episode. Returns it, or ``None`` if the store declined."""
        try:
            k = key or f"ep-{self.writes}"
            ctx = None
            if context is not None and np is not None:
                ctx = np.asarray(context, dtype=np.float64).ravel()[: self.width].copy()
            ep = Experience(key=k, what=str(what), when=time.time(), context=ctx,
                            prediction_error=float(max(0.0, prediction_error)),
                            action=action, outcome=outcome)
            if k in self._store:
                self._order.remove(k)
            self._store[k] = ep
            self._order.append(k)
            self.writes += 1
            self._evict_if_full()
            return ep
        except Exception:  # noqa: BLE001 — a lost memory is never a broken turn
            return None

    def _evict_if_full(self) -> None:
        """Lowest priority goes first: never surprising, never turned out to matter."""
        while len(self._store) > self.capacity:
            victim = min(self._store.values(), key=lambda e: e.priority)
            self._store.pop(victim.key, None)
            try:
                self._order.remove(victim.key)
            except ValueError:
                pass
            self.evicted += 1

    # ---- read ---- #
    def recall(self, cue: Any, *, k: int = 5) -> Recall:
        """Content-addressed retrieval by cosine over context vectors. No token window."""
        try:
            if np is None or cue is None or not self._store:
                return Recall([], [])
            q = np.asarray(cue, dtype=np.float64).ravel()[: self.width]
            qn = float(np.linalg.norm(q))
            if qn <= 0.0:
                return Recall([], [])
            scored: List[Tuple[float, Experience]] = []
            for ep in self._store.values():
                if ep.context is None:
                    continue
                c = ep.context
                cn = float(np.linalg.norm(c))
                if cn <= 0.0:
                    continue
                n = min(q.size, c.size)
                sim = float(np.dot(q[:n], c[:n]) / (qn * cn + 1e-12))
                scored.append((sim, ep))
            scored.sort(key=lambda t: t[0], reverse=True)
            top = scored[: max(1, int(k))]
            return Recall([e for _, e in top], [s for s, _ in top])
        except Exception:  # noqa: BLE001
            return Recall([], [])

    def replay(self, *, n: Optional[int] = None) -> List[Experience]:
        """Sample by priority, then charge each sample for having been replayed.

        Sampling is proportional rather than top-k: top-k replay is the degenerate case that
        rehearses the same handful of episodes forever regardless of the damping.
        """
        try:
            if not self._store:
                return []
            want = int(n or self.replay_batch)
            eps = list(self._store.values())
            weights = np.asarray([max(1e-6, e.priority) for e in eps], dtype=np.float64)
            weights = weights / weights.sum()
            rng = self._rng if self._rng is not None else np.random.default_rng(0)
            idx = rng.choice(len(eps), size=min(want, len(eps)), replace=False, p=weights)
            out = [eps[int(i)] for i in idx]
            for e in out:
                e.replays += 1
            return out
        except Exception:  # noqa: BLE001
            return []

    # ---- the active part ---- #
    def credit(self, cue: Any, *, amount: Optional[float] = None) -> int:
        """Credit relevance *backwards* to the episodes a later outcome depended on.

        This is what makes ``future_relevance`` a learned quantity rather than a guess made at
        write time. Returns how many episodes were credited.
        """
        try:
            rate = float(amount if amount is not None else self.credit_rate)
            hits = self.recall(cue, k=self.credit_window)
            n = 0
            for ep, sim in zip(hits.episodes, hits.similarities):
                if sim <= 0.0:
                    continue
                ep.future_relevance = min(1.0, ep.future_relevance + rate * sim)
                n += 1
            return n
        except Exception:  # noqa: BLE001
            return 0

    def surprising(self, *, k: int = 5) -> List[Experience]:
        """The episodes the world model got most wrong — where Layer 9 goes looking."""
        return sorted(self._store.values(), key=lambda e: e.prediction_error, reverse=True)[:k]

    def pressure(self) -> float:
        """How full the store is, in ``[0, 1]`` — the ``M`` signal the optimizer reads."""
        return float(min(1.0, len(self._store) / max(1, self.capacity)))

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> Dict[str, Any]:
        errs = [e.prediction_error for e in self._store.values()]
        return {"episodes": len(self._store), "capacity": self.capacity, "writes": self.writes,
                "evicted": self.evicted, "pressure": round(self.pressure(), 4),
                "mean_error": round(float(sum(errs) / len(errs)), 6) if errs else None,
                "credited": sum(1 for e in self._store.values() if e.future_relevance > 0.0)}

    def to_dict(self) -> Dict[str, Any]:
        return {"writes": self.writes, "evicted": self.evicted,
                "episodes": [e.to_dict() for e in
                             sorted(self._store.values(), key=lambda x: x.priority,
                                    reverse=True)[:256]]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.writes = int(d.get("writes", 0))
            self.evicted = int(d.get("evicted", 0))
            for item in d.get("episodes") or []:
                k = str(item.get("key", ""))
                if not k:
                    continue
                ep = Experience(key=k, what=str(item.get("what", "")),
                                when=float(item.get("when", time.time())),
                                prediction_error=float(item.get("prediction_error", 0.0)),
                                future_relevance=float(item.get("future_relevance", 0.0)),
                                replays=int(item.get("replays", 0)),
                                action=item.get("action"), outcome=item.get("outcome"))
                self._store[k] = ep
                self._order.append(k)
        except Exception:  # noqa: BLE001 — context vectors do not survive a sidecar, by design
            pass
