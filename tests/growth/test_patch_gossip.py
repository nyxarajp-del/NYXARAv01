"""Tests for growth/patch_gossip.py — signed PatchGrams + independent local verify (Change set D).

Fast + standalone. Pins: a valid gram verifies and applies only after the LOCAL gauntlet passes; a
tampered gram is quarantined; a valid-but-locally-rejected gram is not applied; no-peers broadcast is a
clean no-op.
"""

from __future__ import annotations

from nyxara.growth.patch_gossip import PatchGossip

_KEY = b"shared-swarm-key-000"


def _gram(g: PatchGossip):
    return g.make(module="nyxara/growth/foo.py", diff="- bad\n+ good\n",
                  certificate={"lever": "self_debug", "verified": True},
                  gauntlet={"ok": True, "checks": {"syntax": True}})


def test_valid_gram_verifies():
    g = PatchGossip(node_id="A", key=_KEY)
    assert g.verify(_gram(g)) is True


def test_tampered_gram_is_quarantined():
    g = PatchGossip(node_id="A", key=_KEY)
    gram = _gram(g)
    gram.diff = gram.diff + "\n# sneaky extra line"   # payload no longer matches the signed bundle
    assert g.verify(gram) is False
    out = g.receive(gram, local_gauntlet=lambda _g: True)
    assert out["applied"] is False and out["verified"] is False


def test_wrong_key_fails_signature():
    sender = PatchGossip(node_id="A", key=_KEY)
    receiver = PatchGossip(node_id="B", key=b"a-different-key-9999")
    assert receiver.verify(_gram(sender)) is False


def test_receive_applies_only_after_local_gauntlet():
    g = PatchGossip(node_id="A", key=_KEY)
    gram = _gram(g)
    # valid signature but the receiving node's own gauntlet rejects it → not applied
    rejected = g.receive(gram, local_gauntlet=lambda _g: False)
    assert rejected["applied"] is False and rejected["verified"] is True
    # valid signature AND local gauntlet passes → applied
    applied = g.receive(gram, local_gauntlet=lambda _g: True)
    assert applied["applied"] is True and applied["verified"] is True


def test_broadcast_without_peers_is_a_noop():
    g = PatchGossip(node_id="A", key=_KEY)      # no peers wired
    assert g.broadcast(_gram(g)) is False


def test_broadcast_reaches_a_wired_peer():
    inbox = []
    g = PatchGossip(node_id="A", key=_KEY, peers=inbox.append)
    assert g.broadcast(_gram(g)) is True
    assert len(inbox) == 1
