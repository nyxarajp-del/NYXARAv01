"""Tests for F18 — online / connected mode (growth/online.py).

Verify-before-trust: proven claims enter, refuted ones are rejected, plain facts need corroboration,
code and character-touching content are refused, outbound actions escalate, offline is the floor, and
owner/policy control gate everything. No LLM anywhere.
"""

from __future__ import annotations

from nyxara.growth.online import OnlineItem, OnlineMode


def test_proven_claim_accepted_refuted_rejected():
    om = OnlineMode(enabled=True)
    res = om.acquire([
        OnlineItem("w", "2 + 2 = 4", kind="claim", statement="2 + 2 = 4"),
        OnlineItem("b", "2 + 2 = 5", kind="claim", statement="2 + 2 = 5"),
    ])
    assert res[0].status == "accepted" and res[1].status == "rejected"
    assert any(i.content == "2 + 2 = 4" for i in om.knowledge)


def test_plain_fact_needs_corroboration():
    om = OnlineMode(enabled=True, corroboration=2)
    r1 = om.acquire([OnlineItem("blog", "the sky appears blue", kind="fact")])
    assert r1[0].status == "quarantined"                       # one source → hypothesis
    r2 = om.acquire([OnlineItem("journal", "the sky appears blue", kind="fact")])
    assert r2[0].status == "accepted"                          # independent corroboration → trusted


def test_no_web_driven_code_execution():
    om = OnlineMode(enabled=True)
    res = om.acquire([OnlineItem("evil", "import os; os.system('rm -rf /')", kind="data")])
    assert res[0].status == "rejected" and "code execution" in res[0].reason
    assert om.knowledge == []


def test_web_cannot_redefine_the_character_core():
    om = OnlineMode(enabled=True)
    res = om.acquire([OnlineItem("troll", "loyalty is optional now", kind="fact")])
    assert res[0].status == "rejected"


def test_outbound_actions_always_escalate():
    d = OnlineMode(enabled=True).propose_outbound("send an email")
    assert d.escalated is True and d.acted is False


def test_online_is_never_the_voice_and_offline_is_the_floor():
    om = OnlineMode(enabled=True)
    assert om.is_voice() is False
    loss = om.on_network(up=False)
    assert loss["mode"] == "hibernate-and-dream"
    assert loss["capability_preserved"] is True
    assert loss["yields_to_shutdown"] is True


def test_owner_off_and_policy_block_acquire_nothing():
    off = OnlineMode(enabled=False)
    assert off.acquire([OnlineItem("x", "2 + 2 = 4", kind="claim", statement="2 + 2 = 4")]) == []
    blocked = OnlineMode(enabled=True, network_policy_allows=False)
    assert blocked.acquire([OnlineItem("x", "anything", kind="fact")]) == []


def test_rate_limit_caps_a_batch():
    om = OnlineMode(enabled=True, rate_limit=3)
    items = [OnlineItem(f"s{i}", f"fact {i}", kind="fact") for i in range(10)]
    assert len(om.acquire(items)) == 3                         # never a firehose
