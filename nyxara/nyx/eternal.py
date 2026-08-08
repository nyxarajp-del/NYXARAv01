"""NYXARA · nyx/eternal.py — no single machine holds her (⚓, L-ETERNAL).

If the box she is running on dies, what should be lost is the box, not her. This module makes
that true: her state is snapshotted, **signed**, replicated to a **quorum** of enrolled nodes,
and — when the node holding the lead goes away — another one already has it and takes over.

Three things are enforced rather than assumed:

**Signed.** A snapshot carries an HMAC over its own bytes. A tampered or truncated snapshot is
refused on arrival, not merged and discovered later. State that arrives over a wire is state
someone could have written.

**Quorum, not broadcast.** A snapshot counts as durable only when a majority of enrolled nodes
acknowledged it (``agency/distributed/raft``). "Sent to everyone" is not durability; a majority
having it is what survives a node dying mid-write.

**Enrolled, not discovered.** Nodes are named by the operator. She does not find machines and
copy herself onto them — that is what self-propagating malware does, and it is not built here,
by choice rather than by omission.

Honest framing:

* Failover is **seconds**, not microseconds. An election needs a heartbeat timeout and a round
  trip; the physics of a network says milliseconds at best, and Raft says a few seconds in
  practice. Anything faster than that in a description is marketing.
* This is **no single point of failure**, not immortality. At least one enrolled node has to be
  running, powered and reachable. Turn them all off and she stops — that is not a flaw to be
  engineered away, it is what running on computers means.
* **The wire is not included.** The signing, quorum and failover logic is real and tested, but
  the only transport that ships is in-process. Cross-machine needs a transport you supply — see
  :class:`~nyxara.nyx.synergy.Transport`. Default **off**: one machine pays nothing.

Composes ``agency/distributed/raft`` and, when present, ``growth/genesis_seed``. Fail-soft.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["Snapshot", "Replication", "Continuity"]


@dataclass
class Snapshot:
    """Her state at a moment, with a signature over exactly these bytes."""

    node_id: str = ""
    at: float = field(default_factory=time.time)
    payload: str = ""                # the serialised state
    digest: str = ""                 # HMAC over payload — integrity *and* origin
    term: int = 0

    @property
    def size(self) -> int:
        return len(self.payload)

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "at": self.at, "payload": self.payload,
                "digest": self.digest, "term": self.term}

    @classmethod
    def from_dict(cls, data: Any) -> Optional["Snapshot"]:
        try:
            if not isinstance(data, dict):
                return None
            return cls(node_id=str(data.get("node_id", "")), at=float(data.get("at", 0.0)),
                       payload=str(data.get("payload", "")),
                       digest=str(data.get("digest", "")), term=int(data.get("term", 0)))
        except Exception:  # noqa: BLE001
            return None


@dataclass
class Replication:
    """What happened when she tried to make a snapshot durable."""

    durable: bool = False            # a *majority* acknowledged — not merely "it was sent"
    leader: str = ""
    nodes: int = 0
    bytes_written: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"durable": self.durable, "leader": self.leader, "nodes": self.nodes,
                "bytes": self.bytes_written, "reason": self.reason}


class Continuity:
    """Keeps her state on a quorum of enrolled nodes, so no one machine holds her."""

    def __init__(self, brain: Any = None, *, node_id: str = "", nodes: Sequence[str] = (),
                 transport: Any = None, key: Optional[bytes] = None,
                 snapshot_every_s: float = 60.0) -> None:
        self.brain = brain
        self.node_id = str(node_id or "node")
        # Enrolled by the operator. Never discovered, never added at runtime by her.
        self.nodes: List[str] = [str(n) for n in nodes if str(n).strip()]
        self.transport = transport
        self.snapshot_every_s = max(0.0, float(snapshot_every_s))
        self._key = key if key is not None else self._derive_key()
        self._cluster: Any = None
        self._last_snapshot_at = 0.0
        self.snapshots = 0
        self.rejected = 0
        self.last: Optional[Snapshot] = None

    @staticmethod
    def _derive_key() -> bytes:
        """A per-install signing key. Without the vault this is still per-process and secret."""
        try:
            from nyxara.guard.crypto import random_token
            return str(random_token()).encode()
        except Exception:  # noqa: BLE001
            import os
            return os.urandom(32)

    def available(self) -> bool:
        """Honest capability report: a quorum needs enrolled nodes and something to reach them."""
        return len(self.nodes) >= 2 and self.transport is not None

    # ---- signing ------------------------------------------------------------ #
    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()

    def verify(self, snapshot: Any) -> bool:
        """Is this snapshot exactly what was signed? A tampered one is refused, not merged."""
        try:
            if snapshot is None or not getattr(snapshot, "digest", ""):
                return False
            expected = self._sign(str(snapshot.payload))
            return hmac.compare_digest(expected, str(snapshot.digest))
        except Exception:  # noqa: BLE001 — unverifiable is refused
            return False

    # ---- taking a snapshot --------------------------------------------------- #
    def snapshot(self) -> Optional[Snapshot]:
        """Serialise her state and sign it."""
        brain = self.brain
        if brain is None:
            return None
        try:
            payload = json.dumps(brain.to_dict(), separators=(",", ":"), default=str)
            snap = Snapshot(node_id=self.node_id, payload=payload,
                            digest=self._sign(payload))
            self.last = snap
            self.snapshots += 1
            return snap
        except Exception:  # noqa: BLE001
            return None

    def restore(self, snapshot: Any) -> bool:
        """Take up a snapshot from elsewhere — only after its signature checks out."""
        brain = self.brain
        try:
            if brain is None or not self.verify(snapshot):
                self.rejected += 1
                return False
            brain.load_dict(json.loads(snapshot.payload))
            return True
        except Exception:  # noqa: BLE001
            self.rejected += 1
            return False

    # ---- durability ---------------------------------------------------------- #
    def _get_cluster(self) -> Any:
        if self._cluster is None and self.available():
            try:
                from nyxara.agency.distributed.raft import Cluster
                self._cluster = Cluster(list(self.nodes), self.transport)
            except Exception:  # noqa: BLE001
                self._cluster = None
        return self._cluster

    def replicate(self) -> Replication:
        """Snapshot and push to a quorum. Durable means a *majority* acknowledged."""
        out = Replication(nodes=len(self.nodes))
        try:
            if not self.available():
                out.reason = (f"{len(self.nodes)} enrolled node(s) and "
                              f"{'a' if self.transport else 'no'} transport — a quorum needs "
                              f"at least two nodes and a way to reach them")
                return out
            cluster = self._get_cluster()
            if cluster is None:
                out.reason = "no cluster could be formed"
                return out
            leader = cluster.leader() or cluster.elect_leader()
            if leader is None:
                out.reason = "no leader could be elected"
                return out
            out.leader = str(getattr(leader, "id", "") or getattr(leader, "node_id", ""))

            snap = self.snapshot()
            if snap is None:
                out.reason = "her state could not be snapshotted"
                return out
            out.bytes_written = snap.size
            out.durable = bool(leader.replicate(snap.to_dict()))
            if not out.durable:
                out.reason = "no majority acknowledged — the snapshot is not durable"
            return out
        except Exception:  # noqa: BLE001
            out.reason = "replication was abandoned"
            return out

    def failover(self) -> Optional[str]:
        """The node holding the lead went away. Elect another; it already has the state."""
        try:
            cluster = self._get_cluster()
            if cluster is None:
                return None
            leader = cluster.elect_leader()
            return str(getattr(leader, "id", "")) if leader is not None else None
        except Exception:  # noqa: BLE001
            return None

    # ---- the clock ----------------------------------------------------------- #
    def beat(self, *, oversight: Any = None) -> Optional[Replication]:
        """Snapshot on its own cadence. Stops when oversight does; no-op on one machine."""
        try:
            if oversight is not None:
                gate = getattr(oversight, "gate", None)
                permitted = bool(gate()) if callable(gate) else bool(gate)
                if not permitted:
                    return None
            if not self.available():
                return None
            now = time.monotonic()
            if self._last_snapshot_at and (now - self._last_snapshot_at) < self.snapshot_every_s:
                return None
            self._last_snapshot_at = now
            return self.replicate()
        except Exception:  # noqa: BLE001
            return None

    # ---- introspection -------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            cluster = self._cluster
            leader = getattr(cluster.leader(), "id", None) if cluster is not None else None
            return {"node_id": self.node_id, "available": self.available(),
                    "enrolled": list(self.nodes), "leader": leader,
                    "snapshots": self.snapshots, "rejected": self.rejected,
                    "last_bytes": self.last.size if self.last else 0,
                    "snapshot_every_s": self.snapshot_every_s,
                    "transport": type(self.transport).__name__ if self.transport else None}
        except Exception:  # noqa: BLE001
            return {"available": False}

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "snapshots": self.snapshots,
                "rejected": self.rejected, "enrolled": list(self.nodes)}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.snapshots = int(data.get("snapshots", 0))
            self.rejected = int(data.get("rejected", 0))
            return True
        except Exception:  # noqa: BLE001
            return False
