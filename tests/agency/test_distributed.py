"""Tests for the distributed hive — agency/distributed (Part I). No network needed."""
from __future__ import annotations

from nyxara.agency.distributed import Cluster, DistributedNode, InProcessTransport


def test_single_node_is_a_clean_noop_that_still_replicates_locally():
    node = DistributedNode("solo")
    assert node.is_single_node() is True
    # a lone node is its own majority — replication trivially succeeds (distribution is a no-op)
    assert node.replicate({"turn": 1}) is True
    assert node.log() == [{"turn": 1}]


def test_three_node_cluster_elects_one_leader():
    tr = InProcessTransport()
    cluster = Cluster(["n1", "n2", "n3"], tr)
    leader = cluster.elect_leader()
    assert leader is not None
    leaders = [n for n in cluster.nodes if n.role == "leader"]
    assert len(leaders) == 1 and leaders[0] is leader


def test_leader_replicates_entry_to_majority():
    tr = InProcessTransport()
    cluster = Cluster(["n1", "n2", "n3"], tr)
    leader = cluster.elect_leader()
    assert leader.replicate({"memory": "hello"}) is True
    # the entry lands on every node's replicated log (survives any single node dying)
    for n in cluster.nodes:
        assert {"memory": "hello"} in n.log


def test_follower_cannot_replicate():
    tr = InProcessTransport()
    cluster = Cluster(["n1", "n2", "n3"], tr)
    cluster.elect_leader()
    follower = next(n for n in cluster.nodes if n.role != "leader")
    assert follower.replicate({"x": 1}) is False


def test_disabled_node_refuses():
    node = DistributedNode("solo", enabled=False)
    assert node.replicate({"x": 1}) is False
