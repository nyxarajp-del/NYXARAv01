"""NYXARA · nyx5/negentropy.py — the causal anti-entropy field (🌀, pillar 12).

Every running system drifts toward disorder: caches bloat, duplicate memories accumulate, stale state
lingers, structures rot. NYX-5 does not simply accept that decay — she runs a continuous **maintenance
tick** that actively fights it: deduplicating identical memories, pruning stale entries, and
re-verifying integrity, so that the longer she runs the *cleaner* and more structured her state stays.

Honest ceiling: this is **active repair and compaction**, not a violation of thermodynamics. "The system
gets better the longer it runs" and "mathematical immortality" are analogies for error-correction that
keeps entropy *bounded* — real work spent to hold disorder at bay, not a free lunch. The tick is
fail-soft and runs off the turn-loop (on the autonomic cadence). Pure standard library.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, MutableMapping, Optional

__all__ = ["MaintenanceReport", "NegentropyMaintainer"]


@dataclass
class MaintenanceReport:
    """What one maintenance sweep reclaimed, plus a post-sweep integrity fingerprint."""

    deduped: int = 0
    pruned: int = 0
    kept: int = 0
    integrity: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"deduped": self.deduped, "pruned": self.pruned, "kept": self.kept,
                "integrity": self.integrity[:12]}


class NegentropyMaintainer:
    """Periodic compaction/dedup/integrity over a mutable ``{key: {value, ts}}`` store."""

    def __init__(self, *, interval_ticks: int = 100, stale_seconds: float = 30 * 86400.0) -> None:
        self.interval_ticks = max(1, int(interval_ticks))
        self.stale_seconds = float(stale_seconds)
        self._ticks = 0

    def tick(self, now: float = None) -> bool:
        """Advance the maintenance clock; returns True when a sweep is due this tick."""
        self._ticks += 1
        return self._ticks % self.interval_ticks == 0

    def sweep(self, store: MutableMapping[str, Dict[str, Any]], *,
              now: Optional[float] = None) -> MaintenanceReport:
        """Deduplicate identical values, drop stale entries, and fingerprint the result.

        Each entry is expected to be ``{"value": ..., "ts": <epoch>}``. Fail-soft: malformed entries
        are left untouched, never raising.
        """
        now = time.time() if now is None else now
        report = MaintenanceReport()
        seen_values: Dict[str, str] = {}          # value-hash -> canonical key (first seen wins)
        doomed = []
        for key in list(store.keys()):
            entry = store.get(key)
            if not isinstance(entry, dict) or "value" not in entry:
                continue
            ts = float(entry.get("ts", now))
            if now - ts > self.stale_seconds:
                doomed.append(key)
                continue
            try:
                vhash = hashlib.sha256(
                    json.dumps(entry["value"], sort_keys=True, default=str).encode()).hexdigest()
            except Exception:  # noqa: BLE001 — unhashable value: leave it, don't crash
                report.kept += 1
                continue
            if vhash in seen_values:
                doomed.append(key)                # exact duplicate of an already-kept memory
                report.deduped += 1
            else:
                seen_values[vhash] = key
                report.kept += 1
        for key in doomed:
            if key not in seen_values.values():   # never drop the canonical copy of a value
                store.pop(key, None)
        report.pruned = len(doomed) - report.deduped
        report.integrity = self._fingerprint(store)
        return report

    @staticmethod
    def _fingerprint(store: MutableMapping[str, Dict[str, Any]]) -> str:
        try:
            blob = json.dumps({k: store[k].get("value") for k in sorted(store)},
                              sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            blob = str(sorted(store.keys()))
        return hashlib.sha256(blob.encode()).hexdigest()
