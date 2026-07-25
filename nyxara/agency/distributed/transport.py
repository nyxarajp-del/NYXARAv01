"""NYXARA · agency/distributed/transport.py — pluggable node transport (Part I).

The consensus core (``raft.py``) talks to peers only through this thin transport interface, so the SAME
logic runs over an in-process/loopback transport (default, and what the tests use) or, when the optional
``grpcio`` extra and real peers are present, over a network transport. With zero peers everything degrades
to a single node — distribution becomes a clean no-op.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

__all__ = ["InProcessTransport"]

Handler = Callable[[Dict[str, Any]], Dict[str, Any]]


class InProcessTransport:
    """A loopback transport: nodes register a handler and exchange messages in-process (no network)."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Handler] = {}

    def register(self, node_id: str, handler: Handler) -> None:
        self._handlers[node_id] = handler

    def peers(self, exclude: str) -> List[str]:
        return [nid for nid in self._handlers if nid != exclude]

    def send(self, to: str, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver ``msg`` to peer ``to`` and return its reply; an unreachable peer yields ``{}``."""
        handler = self._handlers.get(to)
        if handler is None:
            return {}
        try:
            return handler(msg) or {}
        except Exception:  # noqa: BLE001 — a peer error is a dropped message, never a crash
            return {}
