"""NYXARA · causal/polymorph_runtime.py — Cellular Poly-Morphing State Runtime (OMNISCIENCE · 2).

*The Master's ask:* let NYXARA replace a live core routine — a router, a memory engine — **without a
restart**: freeze the state, inject the new module in the background, and shift the execution
pointer at the exact instant, with **zero downtime and no state loss**.

What that means here, honestly. A live component is registered inside a :class:`PolymorphCell` — a
thin indirection every caller goes through. :meth:`PolymorphRuntime.hot_swap` then:

    1. **freezes** the current implementation's state (a caller-supplied snapshot fn),
    2. builds the **replacement** in the background (a factory, or a freshly reloaded module),
    3. **migrates** the frozen state into the replacement,
    4. **verifies** it (a caller-supplied predicate), and only then
    5. **shifts the pointer** — one atomic reference assignment under a lock (microseconds).

If *any* step raises or verification fails, the cell is left pointing at the original — a real
**rollback**, so a bad swap never corrupts the live system. The lock is held only for the pointer
assignment, so in-flight calls see either the whole old object or the whole new one — never a torn
state.

The honest boundary. This is genuine safe hot-reload of pure-Python objects/modules with state
migration and rollback — the "downtime" is the sub-millisecond critical section, honestly measured
and reported. It is **not** magic bytecode surgery on a running stack frame, and every swap of a
*core* routine is meant to pass through the same oversight gate as any other self-modification.
"""

from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

__all__ = ["PolymorphCell", "SwapOutcome", "PolymorphRuntime"]


# --------------------------------------------------------------------------- #
# The cell — the indirection callers hold
# --------------------------------------------------------------------------- #
class PolymorphCell:
    """A live reference every caller goes through, so it can be swapped atomically."""

    __slots__ = ("_target", "_lock", "generation")

    def __init__(self, target: Any) -> None:
        self._target = target
        self._lock = threading.Lock()
        self.generation = 0

    @property
    def ref(self) -> Any:
        """The current implementation — always whole, never torn."""
        return self._target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._target(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        # Delegate unknown attributes to the live target (transparent proxy).
        return getattr(object.__getattribute__(self, "_target"), item)


# --------------------------------------------------------------------------- #
# Outcome
# --------------------------------------------------------------------------- #
@dataclass
class SwapOutcome:
    """The result of one hot-swap attempt."""

    key: str
    swapped: bool
    rolled_back: bool = False
    downtime_ms: float = 0.0
    generation: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "swapped": self.swapped, "rolled_back": self.rolled_back,
                "downtime_ms": round(self.downtime_ms, 4), "generation": self.generation,
                "error": self.error}


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class PolymorphRuntime:
    """Registers live components and hot-swaps them with state migration + rollback."""

    def __init__(self) -> None:
        self._cells: Dict[str, PolymorphCell] = {}
        self._swaps: int = 0
        self._rollbacks: int = 0

    def register(self, key: str, target: Any) -> PolymorphCell:
        """Register a live component; hand the returned cell to callers as the entry point."""
        cell = PolymorphCell(target)
        self._cells[key] = cell
        return cell

    def cell(self, key: str) -> Optional[PolymorphCell]:
        return self._cells.get(key)

    def current(self, key: str) -> Any:
        cell = self._cells.get(key)
        return None if cell is None else cell.ref

    # -- the swap ----------------------------------------------------------- #
    def hot_swap(self, key: str, replacement: Any = None, *,
                 factory: Optional[Callable[[], Any]] = None,
                 freeze: Optional[Callable[[Any], Any]] = None,
                 migrate: Optional[Callable[[Any, Any], None]] = None,
                 verify: Optional[Callable[[Any], bool]] = None) -> SwapOutcome:
        """Replace the live component at ``key``.

        Pass the new object as ``replacement``, or a zero-arg ``factory`` that builds it in the
        background (the factory takes precedence). ``freeze``/``migrate``/``verify`` carry state
        and validate before the pointer moves; any failure rolls back to the original.
        """
        cell = self._cells.get(key)
        if cell is None:
            return SwapOutcome(key=key, swapped=False, error="no such cell")
        if factory is None and replacement is None:
            return SwapOutcome(key=key, swapped=False, error="no replacement or factory given")
        old = cell.ref
        try:
            # 1. freeze — capture the state to carry across
            snapshot = freeze(old) if freeze is not None else None
            # 2. build the replacement in the background (nothing live touched yet)
            new = factory() if factory is not None else replacement
            # 3. migrate frozen state into the replacement
            if migrate is not None:
                migrate(snapshot if freeze is not None else old, new)
            # 4. verify before anything goes live
            if verify is not None and not verify(new):
                self._rollbacks += 1
                return SwapOutcome(key=key, swapped=False, rolled_back=True,
                                   generation=cell.generation, error="verification failed")
            # 5. shift the pointer — the only critical section, honestly timed
            t0 = time.perf_counter()
            with cell._lock:  # noqa: SLF001 — the cell's own lock is the swap primitive
                cell._target = new           # noqa: SLF001
                cell.generation += 1
            downtime = (time.perf_counter() - t0) * 1000.0
            self._swaps += 1
            return SwapOutcome(key=key, swapped=True, downtime_ms=downtime,
                               generation=cell.generation)
        except Exception as exc:  # noqa: BLE001 — any failure ⇒ the live ref is untouched
            self._rollbacks += 1
            return SwapOutcome(key=key, swapped=False, rolled_back=True,
                               generation=cell.generation, error=f"{type(exc).__name__}: {exc}")

    def reload_module(self, module_name: str):
        """Hot-reload a Python module by name and return the reloaded module object."""
        mod = importlib.import_module(module_name)
        return importlib.reload(mod)

    # -- introspection ------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        return {"cells": list(self._cells), "swaps": self._swaps, "rollbacks": self._rollbacks}
