"""NYXARA · nyx001/layers/l09_curiosity.py — Layer 9, information gain (🔦, useful unknowns).

The charter: "Seeks useful unknowns instead of just maximizing reward."

The distinction it is pointing at is real and easy to get wrong. A system that seeks *high
prediction error* seeks noise, because noise is unboundedly unpredictable — the "noisy TV problem".
What is worth seeking is error that is **falling**: regions where the model is wrong *and getting
less wrong* are regions where there is structure to learn. So the drive here is

    information_gain = max(0, error_before - error_after)     (learning progress)

measured per region, not raw error. A region whose error stays high forever is *dropped* rather
than pursued, and :meth:`stale` reports those explicitly — the system knows the difference between
"I don't understand this yet" and "there is nothing here to understand".

Regions are keyed by the concept a state falls under (Layer 4) or by a coarse hash when it falls
under none. :meth:`frontier` ranks them by recent learning progress, so the frontier is never empty
while anything is still improving, and empties honestly when nothing is.

Honest: learning progress is measured on this system's own error signal, so it inherits whatever
that signal misses. A region the world model is confidently wrong about with a *stable* low error
looks learned and will not be revisited — the known failure mode of any progress-based drive, and
the reason Layer 10 exists to attack settled beliefs from a different direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["Region", "CuriosityEngine"]


@dataclass
class Region:
    """One area of experience, with the history that says whether it is worth returning to."""

    key: str = ""
    visits: int = 0
    errors: List[float] = field(default_factory=list)
    progress: float = 0.0                   # recent learning progress — the drive signal
    last_seen: float = 0.0

    @property
    def mean_error(self) -> Optional[float]:
        return float(sum(self.errors) / len(self.errors)) if self.errors else None

    @property
    def is_stale(self) -> bool:
        """High error, no progress — noise, or beyond this model. Either way, not worth chasing."""
        m = self.mean_error
        return bool(self.visits >= 6 and m is not None and m > 0.05 and self.progress <= 1e-6)

    @property
    def is_learned(self) -> bool:
        m = self.mean_error
        return bool(self.visits >= 4 and m is not None and m < 0.01)

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "visits": self.visits,
                "mean_error": None if self.mean_error is None else round(self.mean_error, 6),
                "progress": round(self.progress, 6), "stale": self.is_stale,
                "learned": self.is_learned}


class CuriosityEngine:
    """Layer 9. Drives toward learning progress, not toward surprise."""

    def __init__(self, substrate: Any, *, window: int = 8, max_regions: int = 512,
                 decay: float = 0.9) -> None:
        self.substrate = substrate
        self.window = int(window)
        self.max_regions = int(max_regions)
        self.decay = float(decay)
        self._regions: Dict[str, Region] = {}
        self.observations = 0

    @staticmethod
    def region_key(concept: Any, vec: Any = None) -> str:
        """A concept name if one applies, else a coarse sign-hash of the vector."""
        name = getattr(concept, "name", None)
        if name:
            return str(name)
        if vec is None or np is None:
            return "unknown"
        try:
            v = np.asarray(vec, dtype=np.float64).ravel()[:16]
            bits = "".join("1" if x > 0 else "0" for x in v)
            return f"h{int(bits, 2) if bits else 0}"
        except Exception:  # noqa: BLE001
            return "unknown"

    def observe(self, key: str, error: float, *, t: float = 0.0) -> Optional[Region]:
        """Record an error in a region and update its learning progress."""
        try:
            r = self._regions.get(key)
            if r is None:
                if len(self._regions) >= self.max_regions:
                    self._evict()
                r = Region(key=key)
                self._regions[key] = r
            before = r.mean_error
            r.errors.append(float(max(0.0, error)))
            if len(r.errors) > self.window:
                r.errors = r.errors[-self.window:]
            r.visits += 1
            r.last_seen = float(t)
            after = r.mean_error
            if before is not None and after is not None:
                gain = max(0.0, before - after)
                r.progress = self.decay * r.progress + (1.0 - self.decay) * gain
            self.observations += 1
            return r
        except Exception:  # noqa: BLE001
            return None

    def _evict(self) -> None:
        """Evict the least promising: learned first, then stale, then lowest progress."""
        if not self._regions:
            return
        victim = min(self._regions.values(),
                     key=lambda r: (not r.is_learned, not r.is_stale, r.progress))
        self._regions.pop(victim.key, None)

    def frontier(self, *, k: int = 5) -> List[Region]:
        """Where to look next: highest learning progress, excluding stale and learned regions."""
        live = [r for r in self._regions.values() if not r.is_stale and not r.is_learned]
        live.sort(key=lambda r: (r.progress, r.mean_error or 0.0), reverse=True)
        return live[: max(1, int(k))]

    def stale(self) -> List[Region]:
        """Regions abandoned as unlearnable — reported, so the give-up is visible."""
        return [r for r in self._regions.values() if r.is_stale]

    def information_gain(self) -> float:
        """Total recent learning progress — the ``curiosity`` term the objective consumes."""
        return float(sum(r.progress for r in self._regions.values()))

    def drive(self) -> float:
        """How strongly curiosity is pulling right now, in ``[0, 1]``."""
        f = self.frontier(k=8)
        if not f:
            return 0.0
        g = sum(r.progress for r in f)
        return float(min(1.0, g / (1.0 + g)))

    def stats(self) -> Dict[str, Any]:
        rs = list(self._regions.values())
        return {"observations": self.observations, "regions": len(rs),
                "frontier": len(self.frontier(k=1000)), "stale": len(self.stale()),
                "learned": sum(1 for r in rs if r.is_learned),
                "information_gain": round(self.information_gain(), 6),
                "drive": round(self.drive(), 4)}

    def to_dict(self) -> Dict[str, Any]:
        return {"observations": self.observations,
                "regions": [r.to_dict() for r in
                            sorted(self._regions.values(), key=lambda r: r.progress,
                                   reverse=True)[:128]]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.observations = int(d.get("observations", 0))
            for item in d.get("regions") or []:
                k = str(item.get("key", ""))
                if k:
                    self._regions[k] = Region(key=k, visits=int(item.get("visits", 0)),
                                              progress=float(item.get("progress", 0.0)))
        except Exception:  # noqa: BLE001
            pass
