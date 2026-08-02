"""NYXARA · agency/distributed — the P2P mesh substrate (Part I).

Pluggable transport + a minimal Raft-style consensus + a per-device "Lobe" facade. Single-node by default
(distribution is a clean no-op); a real mesh activates only with Master-authorized peers + a permissive
network policy. See the module docstrings for the honest scope of each piece.
"""
from __future__ import annotations

from nyxara.agency.distributed.node import DistributedNode
from nyxara.agency.distributed.raft import Cluster, RaftNode
from nyxara.agency.distributed.transport import InProcessTransport

__all__ = ["DistributedNode", "Cluster", "RaftNode", "InProcessTransport"]
