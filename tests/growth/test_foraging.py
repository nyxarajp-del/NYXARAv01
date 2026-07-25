"""Tests for autonomous epistemic foraging — growth/foraging.py (Part O). Network-free."""
from __future__ import annotations

from nyxara.cognition.hyper_dimensional_vectors import LatentSpaceMap
from nyxara.growth.epistemic_distill import EpistemicDistiller
from nyxara.growth.foraging import EpistemicForager
from nyxara.growth.verified_distill import GroundedVerifier
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.memory.graph import KnowledgeGraph


def _distiller():
    return EpistemicDistiller(knowledge_graph=KnowledgeGraph(),
                              latent_map=LatentSpaceMap(dim=1024, seed=1),
                              verifier=GroundedVerifier(),
                              settings=NyxaraSettings.for_profile(Profile.TEST),
                              graph_path=None)


def _fetch(topic):
    # a verifiable-correct fact and a REFUTED (wrong) claim, distinct prompts
    return [("what is 6*7?", "42"), ("what is 8*9?", "70")]


def test_forages_and_bakes_only_verified():
    d = _distiller()
    forager = EpistemicForager(fetch=_fetch, distiller=d)
    rep = forager.forage("arithmetic")
    assert rep.ran and rep.considered == 2
    assert rep.stored == 1                    # only the proven "6*7=42" baked; "8*9=70" refuted & dropped
    assert d.count() == 1


def test_noop_when_busy():
    d = _distiller()
    forager = EpistemicForager(fetch=_fetch, distiller=d, load_ok=lambda: False)
    rep = forager.forage("arithmetic")
    assert rep.ran is False and "busy" in rep.reason and d.count() == 0


def test_noop_without_network_source():
    rep = EpistemicForager(fetch=None).forage("anything")
    assert rep.ran is False and "no fetch source" in rep.reason


def test_rate_limited_per_cycle():
    d = _distiller()
    many = lambda topic: [("what is 2*%d?" % i, str(2 * i)) for i in range(1, 20)]
    forager = EpistemicForager(fetch=many, distiller=d, max_items_per_cycle=3)
    rep = forager.forage("arithmetic")
    assert rep.considered == 3                 # capped
