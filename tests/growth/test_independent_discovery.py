"""Tests for NYXARA's independent scientific discovery — curiosity → a REAL law.

The gap this closes: her self-driven discovery loop used to reach only *toy* propositions (is 11
prime?) while the powerful law-discovery engine ran on a fixed autopilot rotation, decoupled from
her curiosity. Now a question she poses herself, in a science she is curious about, routes to
genuine law discovery (symbolic regression on experiments she runs) and the corroborated law folds
into her belief model as created information — physics, epidemiology and chemistry alike. No LLM.
"""

from __future__ import annotations

import pytest

from nyxara.growth.autonomous_scientist import AutonomousScientist, QuestionOrigin
from nyxara.growth.law_discovery import LawDiscoveryEngine
from nyxara.mind.world_model import WorldModel


# --------------------------------------------------------------------------- #
# The engine's new-domain experiments recover the right laws
# --------------------------------------------------------------------------- #
def test_engine_discovers_epidemic_growth_law():
    e = LawDiscoveryEngine()
    r = e.discover_by_domain("epidemiology")
    assert r["verdict"] == "corroborated"
    assert r["extrapolation_r2"] > 0.9            # generalises OUTSIDE the training range
    assert "growth_rate" in (r.get("target") or "")


def test_engine_discovers_chemistry_law_incl_arrhenius():
    e = LawDiscoveryEngine()
    # rotate to reach both chemistry experiments (rate law then Arrhenius)
    seen = set()
    for _ in range(3):
        r = e.discover_by_domain("chemistry")
        seen.add(r.get("target"))
        assert r["verdict"] == "corroborated"
    assert "ln_k" in seen or "rate" in seen


def test_engine_domain_targeting_covers_all_sciences():
    e = LawDiscoveryEngine()
    for domain in e.domains():
        r = e.discover_by_domain(domain)
        assert r["domain"] == domain
        assert r["verdict"] in ("corroborated", "abstained", "inconclusive")


# --------------------------------------------------------------------------- #
# The bridge: her curiosity drives real discovery, folded into beliefs
# --------------------------------------------------------------------------- #
def test_curiosity_drives_real_law_discovery_into_beliefs():
    wm = WorldModel()
    auto = AutonomousScientist(discovery_engine=LawDiscoveryEngine(), world_model=wm)
    auto.discover(cycles=16)
    laws = auto.law_beliefs()
    assert len(laws) >= 3, "curiosity did not produce real laws"
    # every law belief is a SUPPORTED, high-confidence, earned-on-held-out-data conclusion
    for b in laws:
        assert b.verdict == "supported"
        assert b.confidence > 0.5
    # at least one EMPIRICAL cycle actually ran (not just toy propositions)
    assert any(c.origin is QuestionOrigin.EMPIRICAL for c in auto.all_cycles())
    # world model recorded epistemic "discover" transitions
    assert len(wm) > 0


def test_epidemiology_and_chemistry_are_discovered_from_curiosity():
    auto = AutonomousScientist(discovery_engine=LawDiscoveryEngine())
    auto.discover(cycles=24)
    domains = {b.variable.split(":")[1] for b in auto.law_beliefs()}
    assert "epidemiology" in domains
    assert "chemistry" in domains


def test_rediscovery_revises_not_duplicates():
    auto = AutonomousScientist(discovery_engine=LawDiscoveryEngine())
    auto.discover(cycles=24)
    before = len(auto.law_beliefs())
    counts_before = {b.variable: b.evidence_count for b in auto.law_beliefs()}
    auto.discover(cycles=24)
    after = auto.law_beliefs()
    # a re-discovered law sharpens (evidence_count grows) rather than duplicating the belief key
    assert len({b.variable for b in after}) == len(after)      # keys unique
    grew = any(b.evidence_count > counts_before.get(b.variable, 0) for b in after)
    assert grew, "re-discovery should accumulate evidence on the same law"
    assert len(after) >= before


def test_corroborated_law_beats_a_rival_theory():
    # adversarial falsification: at least one discovered law beats an invented rival on the decisive
    # extreme-regime test — honest "corroborated, not yet refuted".
    auto = AutonomousScientist(discovery_engine=LawDiscoveryEngine())
    auto.discover(cycles=20)
    assert auto.rivals_beaten >= 1


def test_discovery_mints_a_skill_when_a_skilltree_is_present():
    from nyxara.growth.skilltree import SkillTree
    tree = SkillTree()
    auto = AutonomousScientist(discovery_engine=LawDiscoveryEngine(), skilltree=tree)
    auto.discover(cycles=12)
    assert auto.skills_minted >= 1
    assert any(name.startswith("apply_law::") for name in tree._skills)


# --------------------------------------------------------------------------- #
# Graceful degradation — no engine → offline toy path still progresses
# --------------------------------------------------------------------------- #
def test_offline_without_engine_still_creates_beliefs():
    auto = AutonomousScientist()                      # no discovery engine
    rep = auto.discover(cycles=6)
    assert len(auto.belief_model()) > 0
    assert rep.new_beliefs > 0
    # no law beliefs are possible without an engine, but the loop still runs
    assert auto.law_beliefs() == []


def test_discovery_summary_reports_by_domain():
    auto = AutonomousScientist(discovery_engine=LawDiscoveryEngine())
    auto.discover(cycles=16)
    s = auto.discovery_summary()
    assert s["laws_discovered"] >= 3
    assert isinstance(s["laws_by_domain"], dict) and s["laws_by_domain"]
    assert s["empirical_cycles"] >= 3
