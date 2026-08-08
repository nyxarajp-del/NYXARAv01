"""NYX V.01 · L-ETERNAL — signed, quorum-backed, and honest about what survives."""

from __future__ import annotations

import copy

import pytest

from nyxara.agency.distributed.transport import InProcessTransport
from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.eternal import Continuity, Snapshot


def _brain() -> NyxBrain:
    return NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "aura_enabled": False, "car_enabled": False,
                "episteme_enabled": False, "dialogue_enabled": False}))


@pytest.fixture
def continuity():
    brain = _brain()
    brain.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    return Continuity(brain, node_id="A", nodes=["A", "B", "C"],
                      transport=InProcessTransport(), snapshot_every_s=0.0)


# -------------------- the default costs nothing -------------------- #
def test_one_machine_reports_itself_unavailable():
    solo = Continuity(_brain(), node_id="A", nodes=["A"])
    assert solo.available() is False
    assert solo.beat() is None


def test_a_quorum_needs_nodes_and_a_way_to_reach_them():
    got = Continuity(_brain(), node_id="A", nodes=["A"]).replicate()
    assert got.durable is False and "quorum needs" in got.reason


def test_the_brain_does_not_build_it_by_default():
    assert NyxBrain(get_settings().nyx).eternal is None


# -------------------- signed -------------------- #
def test_a_snapshot_is_signed_over_its_own_bytes(continuity):
    snap = continuity.snapshot()
    assert snap is not None and snap.digest and snap.size > 0
    assert continuity.verify(snap) is True


def test_a_tampered_snapshot_is_refused(continuity):
    """State that arrives over a wire is state someone could have written."""
    snap = continuity.snapshot()
    tampered = copy.copy(snap)
    tampered.payload += " and one more thing"
    assert continuity.verify(tampered) is False
    assert continuity.restore(tampered) is False
    assert continuity.rejected >= 1


def test_a_truncated_snapshot_is_refused(continuity):
    snap = continuity.snapshot()
    truncated = copy.copy(snap)
    truncated.payload = truncated.payload[:-20]
    assert continuity.verify(truncated) is False


def test_an_unsigned_snapshot_is_refused(continuity):
    assert continuity.verify(Snapshot(payload="anything", digest="")) is False
    assert continuity.verify(None) is False


def test_a_snapshot_from_another_install_does_not_verify(continuity):
    other = Continuity(_brain(), node_id="B", nodes=["A", "B"],
                       transport=InProcessTransport())
    assert continuity.verify(other.snapshot()) is False


def test_a_valid_snapshot_restores_her_state(continuity):
    snap = continuity.snapshot()
    revived = Continuity(_brain(), node_id="A", nodes=["A", "B"],
                         transport=InProcessTransport(), key=continuity._key)
    assert revived.restore(snap) is True
    assert revived.brain.recall("gravity apple ground").hit is not None


# -------------------- quorum, not broadcast -------------------- #
def test_replication_is_durable_only_when_a_majority_acknowledged(continuity):
    got = continuity.replicate()
    assert got.durable is True and got.leader and got.nodes == 3
    assert got.bytes_written > 0


def test_a_leader_is_elected(continuity):
    continuity.replicate()
    assert continuity.stats()["leader"]


def test_failover_names_a_node_that_already_has_the_state(continuity):
    continuity.replicate()
    assert continuity.failover() in {"A", "B", "C"}


# -------------------- enrolled, never discovered -------------------- #
def test_nodes_come_only_from_the_operator():
    """She does not find machines and copy herself onto them. That is not built here."""
    got = Continuity(_brain(), node_id="A", nodes=["A", "B"],
                     transport=InProcessTransport())
    assert got.nodes == ["A", "B"]
    assert not hasattr(got, "discover_nodes")


def test_blank_node_names_are_ignored():
    got = Continuity(_brain(), node_id="A", nodes=["A", "", "  ", "B"],
                     transport=InProcessTransport())
    assert got.nodes == ["A", "B"]


# -------------------- oversight -------------------- #
def test_a_scrammed_mind_does_not_replicate(continuity):
    class _Gate:
        def gate(self):
            return False

    assert continuity.beat(oversight=_Gate()) is None
    assert continuity.snapshots == 0


def test_an_open_gate_permits_replicating(continuity):
    class _Gate:
        def gate(self):
            return True

    assert continuity.beat(oversight=_Gate()) is not None


def test_snapshots_respect_their_cadence():
    brain = _brain()
    slow = Continuity(brain, node_id="A", nodes=["A", "B", "C"],
                      transport=InProcessTransport(), snapshot_every_s=3600.0)
    for _ in range(5):
        slow.beat()
    assert slow.snapshots <= 1


# -------------------- robustness -------------------- #
def test_a_brain_that_cannot_be_serialised_fails_quietly():
    class _Broken:
        def to_dict(self):
            raise RuntimeError("nope")

    got = Continuity(_Broken(), node_id="A", nodes=["A", "B"],
                     transport=InProcessTransport())
    assert got.snapshot() is None
    assert got.replicate().durable is False


def test_state_round_trips(continuity):
    continuity.replicate()
    data = continuity.to_dict()

    other = Continuity(_brain(), node_id="A", nodes=["A", "B"],
                       transport=InProcessTransport())
    assert other.load_dict(data) is True
    assert other.snapshots == continuity.snapshots


def test_corrupt_state_does_not_block(continuity):
    assert continuity.load_dict(None) is False


def test_stats_report_availability_honestly(continuity):
    stats = continuity.stats()
    assert stats["available"] is True and stats["enrolled"] == ["A", "B", "C"]
    assert stats["transport"] == "InProcessTransport"
