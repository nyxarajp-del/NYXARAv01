"""NYXARA · nyx/synergy.py — several instances, one mind (🕸, L-SYNERGY).

When NYXARA runs in more than one place, this keeps those instances from becoming strangers.
Whatever one learns is emitted as a **delta**; whatever arrives is **merged**. Because the merge
is a CRDT — commutative, associative, idempotent — every node converges on the same state no
matter what order the deltas arrive in, how many times they are duplicated, or how long a node
was unreachable. No locks, no coordinator, no lost writes.

What actually travels: the concepts and associations she has grown, what she has come to hold
as fact, and what her measured record says about her own faculties. Not her raw episodes — those
are local history, and shipping them would be noise.

Honest framing, and the last point matters most:

* This is **low-latency eventual consistency**, not zero latency and not entanglement. Deltas
  propagate as fast as the transport allows and no faster.
* Convergence is guaranteed **after** partition, not during it. Two nodes cut off from each
  other are genuinely two minds until they can talk again, and this module does not pretend
  otherwise.
* **The wire is not included.** The consensus and merge logic here is real and tested, but the
  only transport that ships is in-process (``agency/distributed/transport``). Running across
  machines needs a transport you supply — the interface is three methods, documented on
  :class:`Transport`. Saying "distributed" while shipping only a loopback would be the kind of
  claim this package exists to avoid.
* Default **off**: a single node pays nothing for this.

Composes :class:`~nyxara.nyx5.mesh.EntangledMesh` (the CRDT) and
:class:`~nyxara.nyx5.holo_swarm.HolographicShard` (lossless field merge). Fail-soft throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

__all__ = ["Transport", "Shared", "HiveSynapse"]


class Transport(Protocol):
    """What a cross-machine transport must provide. Three methods; nothing here is exotic.

    ``send(to, msg)`` delivers a message to one peer and returns its reply (``{}`` when the
    peer is unreachable); ``peers(exclude)`` lists reachable peer ids other than ``exclude``;
    ``register(node_id, handler)`` hands the transport a callable it invokes when a message
    arrives for this node. This is exactly the shape ``agency/distributed/transport`` already
    uses, so an implementation drops straight in. Authentication and encryption are the
    transport's business, and a real deployment must not skip them.
    """

    def send(self, to: str, msg: Any) -> Any: ...                 # pragma: no cover - protocol
    def peers(self, exclude: str) -> List[str]: ...               # pragma: no cover - protocol
    def register(self, node_id: str, handler: Any) -> None: ...   # pragma: no cover - protocol


@dataclass
class Shared:
    """One exchange: what went out, what came back, and what actually changed here."""

    emitted: int = 0
    sent_to: List[str] = field(default_factory=list)
    merged: int = 0                  # keys this node actually took from others
    received: int = 0
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"emitted": self.emitted, "sent_to": self.sent_to, "merged": self.merged,
                "received": self.received, "failures": self.failures}


class HiveSynapse:
    """Binds one NYX to the others: emit what she learned, merge what they did."""

    def __init__(self, brain: Any = None, *, node_id: str = "", transport: Any = None,
                 dim: int = 10000, seed: int = 42, max_delta_bytes: int = 1_048_576) -> None:
        self.brain = brain
        self.node_id = str(node_id or "node")
        self.transport = transport
        self.mesh = self._build_mesh(self.node_id, max_delta_bytes)
        self.shard = self._build_shard(self.node_id, dim, seed)
        self.exchanges = 0
        self.total_merged = 0
        self.inbox: List[Any] = []
        if transport is not None:
            self._attach(transport)

    @staticmethod
    def _build_mesh(node_id: str, max_delta_bytes: int) -> Any:
        try:
            from nyxara.nyx5.mesh import EntangledMesh
            return EntangledMesh(node_id, max_delta_bytes=max_delta_bytes)
        except Exception:  # noqa: BLE001 — without the CRDT there is nothing to share
            return None

    @staticmethod
    def _build_shard(node_id: str, dim: int, seed: int) -> Any:
        try:
            from nyxara.nyx5.holo_swarm import HolographicShard
            return HolographicShard(node_id, dim=dim, seed=seed)
        except Exception:  # noqa: BLE001
            return None

    def _attach(self, transport: Any) -> None:
        try:
            register = getattr(transport, "register", None)
            if callable(register):
                register(self.node_id, self.receive)
        except Exception:  # noqa: BLE001 — a transport that will not register is simply mute
            pass

    def available(self) -> bool:
        """Honest capability report: a CRDT exists *and* there is somewhere to send to."""
        return self.mesh is not None and self.transport is not None

    # ---- what travels ------------------------------------------------------- #
    def publish(self) -> int:
        """Stage what this node has learned. Structure and conclusions travel; episodes do not."""
        brain, mesh = self.brain, self.mesh
        if brain is None or mesh is None:
            return 0
        staged = 0
        try:
            for (a, b), weight in list(brain.graph.edges.items())[:512]:
                mesh.set(f"edge:{a}~{b}", round(float(weight), 6))
                staged += 1
            for key, trace in list(brain.memory.traces.items())[:256]:
                if trace.kind not in ("fact", "conclusion"):
                    continue          # episodes are local history, not shared knowledge
                mesh.set(f"knows:{key}", trace.text)
                if self.shard is not None:
                    self.shard.remember(key, trace.text)
                staged += 1
            metacog = getattr(brain, "metacog", None)
            if metacog is not None:
                for source, record in metacog.reliability.items():
                    mesh.set(f"trust:{source}", round(float(record.score), 6))
                    staged += 1
            return staged
        except Exception:  # noqa: BLE001
            return staged

    def receive(self, payload: Any) -> Dict[str, Any]:
        """A delta arrived. Merge it — order, duplication and delay are all fine by design.

        Returns a reply dict because that is the transport's contract: a truthy reply is how
        the sender learns the peer was reachable.
        """
        mesh = self.mesh
        if mesh is None or payload is None:
            return {}
        try:
            merged = int(mesh.merge(payload))
            self.total_merged += merged
            self._absorb(payload)
            return {"ok": True, "merged": merged, "node": self.node_id}
        except Exception:  # noqa: BLE001 — a malformed delta is dropped, never fatal
            return {}

    def _absorb(self, payload: Any) -> None:
        """Let what another node knows become something this one can actually recall.

        Values are read back from the mesh *after* the merge rather than out of the delta: the
        CRDT may legitimately reject a stale update, and absorbing the raw delta would apply
        something the merge just decided against.
        """
        brain, mesh = self.brain, self.mesh
        if brain is None or mesh is None:
            return
        try:
            for key in (getattr(payload, "updates", {}) or {}):
                name = str(key)
                value = mesh.get(name)
                if value is None:
                    continue
                if name.startswith("knows:") and isinstance(value, str):
                    brain.remember(name.split(":", 1)[1], value, kind="fact")
                elif name.startswith("edge:"):
                    pair = name.split(":", 1)[1]
                    if "~" in pair:
                        brain.graph.activate(pair.split("~"), decay=False)
        except Exception:  # noqa: BLE001 — absorbing is best-effort
            pass

    # ---- one exchange -------------------------------------------------------- #
    def exchange(self) -> Shared:
        """Publish, push to every peer, and merge whatever came the other way."""
        out = Shared()
        try:
            out.emitted = self.publish()
            if self.mesh is None or self.transport is None:
                return out
            delta = self.mesh.emit()
            for peer in self._peers():
                try:
                    # An unreachable peer yields an empty reply rather than raising — that is
                    # the transport contract, and it is what makes partition survivable.
                    reply = self.transport.send(peer, delta)
                    (out.sent_to if reply else out.failures).append(peer)
                except Exception:  # noqa: BLE001 — one unreachable peer is not a failure of all
                    out.failures.append(peer)
            while self.inbox:
                reply = self.receive(self.inbox.pop(0))
                out.merged += int(reply.get("merged", 0))
                out.received += 1
            self.exchanges += 1
            return out
        except Exception:  # noqa: BLE001
            return out

    def _peers(self) -> List[str]:
        try:
            return [str(p) for p in (self.transport.peers(self.node_id) or [])]
        except Exception:  # noqa: BLE001
            return []

    def beat(self, *, oversight: Any = None) -> Optional[Shared]:
        """One tick of the shared clock. Stops when oversight does; no-op with no transport."""
        try:
            if oversight is not None:
                gate = getattr(oversight, "gate", None)
                permitted = bool(gate()) if callable(gate) else bool(gate)
                if not permitted:
                    return None
            if not self.available():
                return None
            return self.exchange()
        except Exception:  # noqa: BLE001
            return None

    # ---- introspection ------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            return {"node_id": self.node_id, "available": self.available(),
                    "peers": self._peers() if self.transport is not None else [],
                    "exchanges": self.exchanges, "merged_total": self.total_merged,
                    "shared_keys": len(self.mesh.keys()) if self.mesh is not None else 0,
                    "transport": type(self.transport).__name__ if self.transport else None}
        except Exception:  # noqa: BLE001
            return {"available": False}

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "exchanges": self.exchanges,
                "merged_total": self.total_merged,
                "mesh": self.mesh.to_dict() if self.mesh is not None else None}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.exchanges = int(data.get("exchanges", 0))
            self.total_merged = int(data.get("merged_total", 0))
            if data.get("mesh") and self.mesh is not None:
                self.mesh.load_dict(data["mesh"])
            return True
        except Exception:  # noqa: BLE001
            return False
