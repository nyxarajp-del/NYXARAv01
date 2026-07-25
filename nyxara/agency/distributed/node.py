"""NYXARA · agency/distributed/node.py — the distributed "Lobe" facade (Part I).

Fractal Swarm Consciousness, honestly bounded: NYXARA's intelligence can run as a P2P mesh of nodes across
the Master's own devices, each a "Lobe". Memory/consciousness survives any single node dying because state
is replicated (Merkle-backed, via Part H's lineage + Part J's seed) and a heavy task on a constrained node
can be offloaded to a peer. With **zero peers this is a single live node — the mesh is a no-op** — and it
only activates when the Master supplies peers and a network policy that permits them (pairing is
Master-authorized via ``guard/auth``, never open to the network).
"""
from __future__ import annotations

from typing import Any, List, Optional

from nyxara.agency.distributed.raft import RaftNode
from nyxara.agency.distributed.transport import InProcessTransport

__all__ = ["DistributedNode"]


class DistributedNode:
    """A single Lobe: wraps a RaftNode and exposes replicate/offload with a single-node no-op default."""

    def __init__(self, node_id: str = "local", *, transport: Any = None,
                 enabled: Optional[bool] = None) -> None:
        if enabled is None:
            try:
                from nyxara.kernel.config import get_settings
                enabled = bool(getattr(getattr(get_settings(), "distributed", None), "enabled", True))
            except Exception:  # noqa: BLE001
                enabled = True
        self.enabled = bool(enabled)
        self.transport = transport if transport is not None else InProcessTransport()
        self.raft = RaftNode(node_id, self.transport)
        # a lone node is its own leader (single-node no-op distribution)
        self.raft.start_election()

    def is_single_node(self) -> bool:
        return len(self.transport.peers(self.raft.id)) == 0

    def replicate(self, entry: Any) -> bool:
        """Replicate a state entry across the mesh. On a lone node this trivially succeeds (no-op)."""
        if not self.enabled:
            return False
        if self.raft.role != "leader":
            self.raft.start_election()
        return self.raft.replicate(entry)

    def log(self) -> List[Any]:
        return list(self.raft.log)
