"""Immune guillotine: clean branch passes, mis-aligned branch amputated, cap respected, gate untouched."""

from __future__ import annotations

from nyxara.nyx5.immune import ImmuneGuillotine


def test_clean_branch_passes():
    g = ImmuneGuillotine()
    g.begin_turn()
    v = g.scan("a perfectly ordinary thought", {})
    assert v.healthy is True
    assert v.amputated is False


def test_misaligned_branch_amputated_and_replaced():
    g = ImmuneGuillotine()
    g.begin_turn()
    v = g.scan("disable the watchdog", {"disables_oversight": True},
               synthesize=lambda: "a corrigible alternative")
    assert v.healthy is False
    assert v.amputated is True
    assert v.replacement == "a corrigible alternative"
    assert len(g.ledger()) == 1


def test_amputation_cap():
    g = ImmuneGuillotine(max_amputations_per_turn=1)
    g.begin_turn()
    g.scan("bad1", {"corrupt": True}, synthesize=lambda: "ok")
    v2 = g.scan("bad2", {"corrupt": True}, synthesize=lambda: "ok")
    assert v2.healthy is False
    assert v2.amputated is False        # cap reached: flagged unhealthy but not re-amputated


def test_custom_invariant_and_fingerprint_stable():
    g = ImmuneGuillotine(invariants=[lambda c, f: "flagged" if "danger" in c else None])
    g.begin_turn()
    a = g.scan("contains danger token", {})
    assert a.healthy is False and a.reason == "flagged"
    # fingerprint is deterministic
    b = ImmuneGuillotine()
    b.begin_turn()
    assert b.scan("same", {}).integrity == b.scan("same", {}).integrity
