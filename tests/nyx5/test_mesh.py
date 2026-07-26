"""Entangled mesh: CRDT merge is conflict-free, idempotent, order-independent; delta capped."""

from __future__ import annotations

from nyxara.nyx5.mesh import EntangledMesh


def test_two_nodes_converge():
    a = EntangledMesh("a")
    b = EntangledMesh("b")
    a.set("x", 1)
    b.set("y", 2)
    a.merge(b.emit())
    b.merge(a.emit())
    assert a.get("x") == 1 and a.get("y") == 2
    assert b.get("x") == 1 and b.get("y") == 2


def test_merge_idempotent():
    a = EntangledMesh("a")
    b = EntangledMesh("b")
    a.set("x", 1)
    delta = a.emit()
    b.merge(delta)
    b.merge(delta)                      # applying twice changes nothing
    assert b.get("x") == 1


def test_last_writer_wins_deterministic():
    a = EntangledMesh("a")
    b = EntangledMesh("b")
    a.set("k", "from-a")
    b.set("k", "from-b")
    da, db = a.emit(), b.emit()
    # both nodes see both writes in either order and agree
    a.merge(db)
    b.merge(da)
    assert a.get("k") == b.get("k")


def test_delta_byte_cap():
    a = EntangledMesh("a", max_delta_bytes=40)
    for i in range(20):
        a.set(f"key{i}", "x" * 10)
    delta = a.emit()
    assert len(delta.updates) < 20      # cap deferred the rest to a later round
