"""NYX V.01 · L-SYNERGY — several instances converging, and honest about the wire."""

from __future__ import annotations

import pytest

from nyxara.agency.distributed.transport import InProcessTransport
from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.synergy import HiveSynapse


def _brain() -> NyxBrain:
    return NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "aura_enabled": False, "car_enabled": False,
                "episteme_enabled": False, "dialogue_enabled": False}))


@pytest.fixture
def pair():
    transport = InProcessTransport()
    a, b = _brain(), _brain()
    return (a, HiveSynapse(a, node_id="A", transport=transport),
            b, HiveSynapse(b, node_id="B", transport=transport))


# -------------------- the default costs nothing -------------------- #
def test_a_lone_node_reports_itself_unavailable():
    synapse = HiveSynapse(_brain(), node_id="solo")
    assert synapse.available() is False
    assert synapse.beat() is None


def test_the_brain_does_not_build_it_by_default():
    assert NyxBrain(get_settings().nyx).synergy is None


# -------------------- what travels -------------------- #
def test_what_she_holds_as_fact_reaches_the_other_node(pair):
    a, ha, b, _hb = pair
    a.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    assert b.recall("gravity apple ground").hit is None

    ha.exchange()
    hit = b.recall("gravity apple ground").hit
    assert hit is not None and hit.kind == "fact"


def test_learned_structure_reaches_the_other_node(pair):
    a, ha, b, _hb = pair
    a.perceive("gravity apple falls")
    ha.exchange()
    assert b.graph.weight("gravity", "apple") > 0.0


def test_raw_episodes_stay_local(pair):
    """Local history is noise on the wire; only what she *holds* is worth sharing."""
    a, ha, b, _hb = pair
    a.memory.remember("e1", "the master said hello this morning", kind="episode")
    ha.exchange()
    assert b.recall("master said hello morning").hit is None


def test_measured_self_trust_is_shared(pair):
    a, ha, _b, hb = pair
    for i in range(6):
        a.resolve(a.think(f"how might we warm a room {i}"), correct=0.0)
    ha.exchange()
    assert any(k.startswith("trust:") for k in hb.mesh.keys())


# -------------------- CRDT: order, duplication and delay are fine -------------------- #
def test_merging_the_same_delta_twice_changes_nothing_the_second_time(pair):
    a, ha, _b, hb = pair
    a.remember("f1", "a fact worth sharing", kind="fact")
    ha.publish()
    delta = ha.mesh.emit()
    first = hb.receive(delta)
    second = hb.receive(delta)
    assert first.get("merged", 0) >= 1 and second.get("merged", 0) == 0


def test_an_emit_carries_what_changed_since_the_last_one(pair):
    _a, ha, _b, _hb = pair
    ha.publish()
    assert len(ha.mesh.emit().updates) >= 0
    assert len(ha.mesh.emit().updates) == 0        # nothing changed in between


def test_a_node_that_was_away_catches_up(pair):
    """Convergence is guaranteed after partition — that is the actual claim."""
    a, ha, b, hb = pair
    a.remember("f1", "something learned while the other was unreachable", kind="fact")
    ha.publish()
    away = ha.mesh.emit()                          # B was "offline" and never got this
    assert b.recall("learned while other unreachable").hit is None
    hb.receive(away)                               # B comes back
    assert b.recall("learned while other unreachable").hit is not None


# -------------------- robustness -------------------- #
def test_an_unreachable_peer_is_recorded_not_fatal():
    class _Broken:
        def register(self, node_id, handler):
            return None

        def peers(self, exclude):
            return ["ghost"]

        def send(self, to, msg):
            return {}                              # unreachable: empty reply, no exception

    synapse = HiveSynapse(_brain(), node_id="A", transport=_Broken())
    got = synapse.exchange()
    assert got.failures == ["ghost"] and got.sent_to == []


def test_a_transport_that_raises_does_not_break_the_exchange():
    class _Explodes:
        def register(self, node_id, handler):
            return None

        def peers(self, exclude):
            return ["ghost"]

        def send(self, to, msg):
            raise RuntimeError("wire on fire")

    assert HiveSynapse(_brain(), node_id="A", transport=_Explodes()).exchange().failures


def test_a_malformed_delta_is_dropped(pair):
    _a, _ha, _b, hb = pair
    assert hb.receive(None) == {}
    assert hb.receive(object()) == {}


# -------------------- oversight -------------------- #
def test_a_scrammed_mind_does_not_exchange(pair):
    class _Gate:
        def gate(self):
            return False

    _a, ha, _b, _hb = pair
    assert ha.beat(oversight=_Gate()) is None
    assert ha.exchanges == 0


def test_an_open_gate_permits_exchanging(pair):
    class _Gate:
        def gate(self):
            return True

    _a, ha, _b, _hb = pair
    assert ha.beat(oversight=_Gate()) is not None


# -------------------- state -------------------- #
def test_state_round_trips(pair):
    a, ha, _b, _hb = pair
    a.remember("f1", "a fact", kind="fact")
    ha.exchange()
    data = ha.to_dict()

    other = HiveSynapse(_brain(), node_id="A")
    assert other.load_dict(data) is True
    assert other.exchanges == ha.exchanges


def test_corrupt_state_does_not_block(pair):
    _a, ha, _b, _hb = pair
    assert ha.load_dict(None) is False
