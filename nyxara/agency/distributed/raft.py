"""NYXARA · agency/distributed/raft.py — a minimal Raft-style replicated log (Part I).

Honest scope: this is a **bounded, synchronous** implementation of the essential Raft ideas — term-based
leader election and majority log replication — sufficient for task/memory recovery across a small mesh of
NYXARA nodes (a P2P "Lobe" per device). It is not a hardened production Raft (no persistence of term/vote
across restarts, no snapshotting, synchronous RPC); it is the architecture, correct for the cases it
claims. With a single node it is trivially its own majority, so distribution is a clean no-op.

Talks to peers only through ``transport`` (see ``transport.py``), so the same logic runs in-process (tests)
or over the network when real peers + the grpc transport are present.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["RaftNode", "Cluster"]


class RaftNode:
    """One node in the mesh: holds a term, a role, and a replicated log."""

    def __init__(self, node_id: str, transport: Any) -> None:
        self.id = node_id
        self.transport = transport
        self.term = 0
        self.role = "follower"                 # follower | candidate | leader
        self.voted_for: Optional[str] = None
        self.log: List[Any] = []
        transport.register(node_id, self._handle)

    # ---- cluster size (from the live transport) ---- #
    def _n(self) -> int:
        return len(self.transport.peers(self.id)) + 1

    def _majority(self) -> int:
        return self._n() // 2 + 1

    # ---- RPC handler ---- #
    def _handle(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        t = int(msg.get("term", 0))
        kind = msg.get("type")
        if kind == "request_vote":
            if t > self.term:
                self.term, self.voted_for, self.role = t, None, "follower"
            granted = t >= self.term and self.voted_for in (None, msg.get("candidate"))
            if granted:
                self.voted_for = msg.get("candidate")
            return {"granted": bool(granted), "term": self.term}
        if kind == "append_entries":
            if t < self.term:
                return {"ok": False, "term": self.term}
            self.term, self.role, self.voted_for = t, "follower", self.voted_for
            if "entry" in msg:
                self.log.append(msg["entry"])
            return {"ok": True, "term": self.term}
        return {}

    # ---- election ---- #
    def start_election(self) -> bool:
        """Stand for election; become leader on a majority of votes (self included)."""
        self.term += 1
        self.role = "candidate"
        self.voted_for = self.id
        votes = 1
        for peer in self.transport.peers(self.id):
            reply = self.transport.send(peer, {"type": "request_vote", "term": self.term,
                                               "candidate": self.id})
            if reply.get("granted"):
                votes += 1
        if votes >= self._majority():
            self.role = "leader"
            return True
        self.role = "follower"
        return False

    # ---- replication ---- #
    def replicate(self, entry: Any) -> bool:
        """Leader-only: append locally and replicate to a majority. True iff a majority acknowledged."""
        if self.role != "leader":
            return False
        self.log.append(entry)
        acks = 1
        for peer in self.transport.peers(self.id):
            reply = self.transport.send(peer, {"type": "append_entries", "term": self.term,
                                               "leader": self.id, "entry": entry})
            if reply.get("ok"):
                acks += 1
        return acks >= self._majority()


class Cluster:
    """A small mesh of in-process RaftNodes — the test/loopback harness for the consensus core."""

    def __init__(self, node_ids: List[str], transport: Any) -> None:
        self.nodes: List[RaftNode] = [RaftNode(nid, transport) for nid in node_ids]

    def elect_leader(self) -> Optional[RaftNode]:
        """Deterministically elect the first node that wins a majority."""
        for node in self.nodes:
            if node.start_election():
                for other in self.nodes:
                    if other is not node:
                        other.role = "follower"
                return node
        return None

    def leader(self) -> Optional[RaftNode]:
        for node in self.nodes:
            if node.role == "leader":
                return node
        return None
