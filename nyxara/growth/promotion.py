"""NYXARA · growth/promotion.py — process-wide model-promotion event bus (🏭→🧠).

Closes the train→serve loop. Many independent :class:`~nyxara.growth.foundry.Foundry`
instances share ONE on-disk root (the orchestrator's autoforge, the GrowthEngine, genesis,
topology, the CLI …), so a per-instance callback cannot tell the *live* brain that new
weights were promoted. This module is the single, process-wide rendezvous: any Foundry
that promotes (or rolls back) calls :func:`notify`, and any long-lived consumer — the
kernel's :class:`~nyxara.kernel.orchestrator.NyxaraCore`, which hot-reloads the serving
provider — registers with :func:`subscribe`.

Honesty properties:
* ``notify`` never raises into the promoter — a broken listener cannot fail (or fake)
  a promotion; errors are recorded on the listener side only.
* Bound methods are held via :class:`weakref.WeakMethod` so a dead core never leaks or
  keeps receiving events; plain functions are held strongly (the caller owns their life).
* Cross-process promotions can't ride an in-memory bus at all — the serving side's
  pointer-file staleness check (:class:`~nyxara.mind.llm.SelfProvider`) is the backstop.
"""

from __future__ import annotations

import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

__all__ = ["PromotionEvent", "subscribe", "unsubscribe", "notify"]


@dataclass(frozen=True)
class PromotionEvent:
    """What just changed under ``foundry/active`` — enough to reload and to journal."""

    action: str                                  # "promote" | "rollback"
    version: int                                 # the now-active version number
    kind: str                                    # "lora" | "nanogpt" | "genesis_np" | "kngram" | ...
    path: str                                    # the active version's directory
    root: str                                    # the foundry root holding the pointer
    metrics: Dict[str, float] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "version": self.version, "kind": self.kind,
                "path": self.path, "root": self.root, "metrics": dict(self.metrics),
                "at": self.at}


_lock = threading.Lock()
# each entry is a zero-arg resolver returning the callable (or None when its owner died)
_listeners: List[Callable[[], Any]] = []


def _resolver(fn: Callable[[PromotionEvent], Any]) -> Callable[[], Any]:
    if hasattr(fn, "__self__"):          # bound method → weak, so a dead core unsubscribes itself
        ref = weakref.WeakMethod(fn)
        return lambda: ref()
    return lambda: fn                     # plain function/lambda → strong (caller owns lifetime)


def subscribe(fn: Callable[[PromotionEvent], Any]) -> None:
    """Register ``fn`` to be called with every :class:`PromotionEvent`. Idempotent."""
    with _lock:
        if any(r() == fn for r in _listeners):
            return
        _listeners.append(_resolver(fn))


def unsubscribe(fn: Callable[[PromotionEvent], Any]) -> None:
    with _lock:
        _listeners[:] = [r for r in _listeners if r() not in (fn, None)]


def notify(event: PromotionEvent) -> int:
    """Deliver ``event`` to every live listener; return how many were called.

    Listener exceptions are swallowed (a reload failure must never fail a promotion —
    the serving side records its own error and the pointer-poll retries later)."""
    with _lock:
        resolved = [r() for r in _listeners]
        _listeners[:] = [r for r, fn in zip(_listeners, resolved) if fn is not None]
        live = [fn for fn in resolved if fn is not None]
    called = 0
    for fn in live:
        try:
            fn(event)
            called += 1
        except Exception:  # noqa: BLE001 — never let a listener break the promoter
            pass
    return called
