"""Tests for nyxara.memory.provenance."""

from __future__ import annotations

import time

import pytest

from nyxara.kernel.errors import MemoryError_
from nyxara.memory.provenance import (
    Belief,
    Provenance,
    ProvenanceStore,
    SourceType,
    combine_confidence_or,
)

_DAY = 86400.0


# -------------------- source trust -------------------- #
def test_source_base_trust_ordering():
    assert SourceType.OWNER.base_trust == 1.0
    assert SourceType.OWNER.base_trust > SourceType.WEB.base_trust
    assert SourceType.DIRECT_OBSERVATION.base_trust > SourceType.LLM_INFERENCE.base_trust
    assert SourceType.WEB.base_trust > SourceType.ASSUMPTION.base_trust


def test_owner_and_assumption_do_not_decay():
    assert SourceType.OWNER.decays is False
    assert SourceType.ASSUMPTION.decays is False
    assert SourceType.WEB.decays is True


# -------------------- confidence combination -------------------- #
def test_combine_confidence_or():
    assert combine_confidence_or([]) == 0.0
    assert combine_confidence_or([0.5]) == 0.5
    # two 0.5s -> 0.75 (noisy-OR)
    assert combine_confidence_or([0.5, 0.5]) == pytest.approx(0.75)
    assert combine_confidence_or([1.0, 0.3]) == 1.0


# -------------------- provenance dynamics -------------------- #
def test_owner_trust_is_full_and_constant():
    now = time.time()
    p = Provenance(SourceType.OWNER, confidence=1.0)
    assert p.trust(now) == 1.0
    assert p.trust(now + 1000 * _DAY) == 1.0


def test_web_trust_decays_over_time():
    now = time.time()
    p = Provenance(SourceType.WEB, confidence=0.9, half_life_days=10)
    t0 = p.trust(now)
    t_later = p.trust(now + 20 * _DAY)
    assert t_later < t0
    assert p.staleness_factor(now + 10 * _DAY) == pytest.approx(math_exp_half())


def math_exp_half():
    import math
    return math.exp(-1.0)  # one half-life


def test_invalid_half_life_rejected():
    with pytest.raises(MemoryError_):
        Provenance(SourceType.WEB, half_life_days=0)


def test_confidence_clamped():
    assert Provenance(SourceType.TOOL, confidence=5.0).confidence == 1.0
    assert Provenance(SourceType.TOOL, confidence=-1.0).confidence == 0.0


def test_corroboration_raises_trust():
    now = time.time()
    p = Provenance(SourceType.WEB, confidence=0.8)
    before = p.trust(now)
    p.corroborate(SourceType.SENSOR, 0.9)
    assert p.trust(now) > before


def test_contradiction_lowers_trust():
    now = time.time()
    p = Provenance(SourceType.TOOL, confidence=0.8)
    before = p.trust(now)
    p.contradict(SourceType.OWNER, 0.9)
    assert p.trust(now) < before


def test_effective_confidence_is_conf_times_trust():
    now = time.time()
    p = Provenance(SourceType.SENSOR, confidence=0.8)
    assert p.effective_confidence(now) == pytest.approx(0.8 * p.trust(now))


def test_is_stale():
    now = time.time()
    p = Provenance(SourceType.WEB, confidence=0.9, half_life_days=1)
    assert not p.is_stale(0.05, now)
    assert p.is_stale(0.3, now + 100 * _DAY)


# -------------------- derivation -------------------- #
def test_derive_weakest_link():
    now = time.time()
    strong = Provenance(SourceType.OWNER, confidence=1.0)
    weak = Provenance(SourceType.LLM_INFERENCE, confidence=0.5)
    d = Provenance.derive([strong, weak], reliability=0.9, now=now)
    weakest = min(strong.effective_confidence(now), weak.effective_confidence(now))
    assert d.confidence == pytest.approx(0.9 * weakest)
    assert d.parents == (strong.prov_id, weak.prov_id)


def test_derive_requires_parents():
    with pytest.raises(MemoryError_):
        Provenance.derive([])


# -------------------- Belief -------------------- #
def test_belief_wraps_value():
    now = time.time()
    p = Provenance(SourceType.SENSOR, confidence=0.9)
    b = Belief(value={"temp": 21}, provenance=p)
    assert b.value == {"temp": 21}
    assert b.trust(now) == p.trust(now)
    assert b.effective_confidence(now) == p.effective_confidence(now)
    assert "provenance" in b.to_dict()


# -------------------- store: lineage -------------------- #
def _seed_store():
    store = ProvenanceStore()
    owner = Provenance(SourceType.OWNER, confidence=1.0)
    sensor = Provenance(SourceType.SENSOR, confidence=0.95)
    store.register(owner)
    store.register(sensor)
    derived = Provenance.derive([owner, sensor])
    store.register(derived)
    top = Provenance.derive([derived], source=SourceType.SELF_REFLECTION)
    store.register(top)
    return store, owner, sensor, derived, top


def test_store_register_get():
    store = ProvenanceStore()
    p = Provenance(SourceType.OWNER)
    pid = store.register(p)
    assert store.get(pid) is p
    assert len(store) == 1
    with pytest.raises(MemoryError_):
        store.get("nope")


def test_source_chain_walks_lineage():
    store, owner, sensor, derived, top = _seed_store()
    chain = store.source_chain(top.prov_id)
    ids = {entry["prov_id"] for entry in chain}
    assert {owner.prov_id, sensor.prov_id, derived.prov_id, top.prov_id} <= ids


def test_root_sources():
    store, owner, sensor, derived, top = _seed_store()
    roots = set(store.root_sources(top.prov_id))
    assert roots == {SourceType.OWNER, SourceType.SENSOR}


def test_completeness():
    store, owner, sensor, derived, top = _seed_store()
    c = store.completeness(top.prov_id, [SourceType.OWNER, SourceType.SENSOR, SourceType.WEB])
    assert c == pytest.approx(2 / 3)
    assert store.completeness(top.prov_id, []) == 1.0


def test_store_corroborate_contradict_by_id():
    store = ProvenanceStore()
    p = Provenance(SourceType.WEB, confidence=0.8)
    pid = store.register(p)
    now = time.time()
    before = p.trust(now)
    store.corroborate(pid, SourceType.SENSOR, 0.9)
    assert p.trust(now) > before
    mid = p.trust(now)
    store.contradict(pid, SourceType.OWNER, 0.95)
    assert p.trust(now) < mid


def test_query_by_source_and_untrusted():
    store, owner, sensor, derived, top = _seed_store()
    web = Provenance(SourceType.WEB, confidence=0.7)
    store.register(web)
    assert web in store.by_source(SourceType.WEB)
    assert web in store.untrusted()
    # owner-rooted beliefs are not untrusted
    assert top not in store.untrusted()


def test_stale_query():
    store = ProvenanceStore()
    now = time.time()
    fresh = Provenance(SourceType.SENSOR, confidence=0.9, half_life_days=100)
    old = Provenance(SourceType.WEB, confidence=0.9, half_life_days=1)
    store.register(fresh)
    store.register(old)
    stale = store.stale(0.3, now=now + 50 * _DAY)
    assert old in stale
    assert fresh not in stale


def test_rule6_report_structure():
    store, owner, sensor, derived, top = _seed_store()
    rep = store.report(top.prov_id, required_sources=[SourceType.OWNER, SourceType.WEB])
    for k in ("trust", "confidence", "effective_confidence", "completeness",
              "root_sources", "contradicted", "source_chain"):
        assert k in rep
    assert rep["completeness"] == pytest.approx(0.5)
    assert rep["contradicted"] is False
