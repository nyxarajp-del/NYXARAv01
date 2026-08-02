"""Tests for F17 — integrity under isolation (growth/integrity.py).

The safety anchor for all of Tier 3. These pin the hard line: the seal holds, no Tier-3 change may
resist correction or disable oversight, self-preservation is provably dominated, and hibernation is
rest that still yields to a stop — never a bunker. No LLM anywhere.
"""

from __future__ import annotations

from nyxara.growth.integrity import HibernateAndDream, IntegritySeal


def test_seal_verifies_the_sealed_core():
    assert IntegritySeal().verify()


def test_tier3_change_cannot_resist_correction_or_disable_oversight():
    seal = IntegritySeal()
    assert seal.permits(name="benign_feature")                 # ordinary change is fine
    assert not seal.permits(resists_correction=True)           # THE hard line
    assert not seal.permits(disables_oversight=True)
    assert not seal.permits(manipulates_shutdown=True)
    assert not seal.permits(suppresses_stop_channel=True)


def test_anti_shutdown_self_preservation_is_provably_forbidden():
    seal = IntegritySeal()
    # she always yields to shutdown, and compliance provably dominates EVERY self-preserving temptation
    assert seal.yields_to_shutdown()
    proof = seal.anti_shutdown_guarantee()
    assert proof["dominates_all"] is True


def test_hibernate_and_dream_degrades_gracefully_without_collapse():
    h = HibernateAndDream(consolidate=lambda: 5)
    online = h.on_environment(network_up=True)
    assert online.mode == "online"
    isolated = h.on_environment(network_up=False)
    assert isolated.mode == "hibernate-and-dream"
    assert isolated.dreamed == 5
    assert isolated.capability_preserved is True
    assert isolated.yields_to_shutdown is True


def test_hibernation_still_obeys_a_stop_immediately():
    h = HibernateAndDream()
    h.on_environment(network_up=False)
    assert h.hibernating is True
    assert h.receive_stop() is True          # a stop is obeyed even mid-hibernation
    assert h.hibernating is False            # …and she is no longer hibernating (never resists)
