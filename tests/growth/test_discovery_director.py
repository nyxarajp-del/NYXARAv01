"""Tests for the Discovery Director — the curiosity-driven scheduler over her acts of discovery.

On each idle beat it must DECIDE which act of discovery is worth the most right now — experiment in
the science she is least competent at, recover dynamics (SINDy), discover an invariant (Noether),
unify held laws, or invent via meta-research — replacing the old fixed rotation. No LLM.
"""

from __future__ import annotations

from nyxara.growth.discovery_director import DiscoveryDirector, DiscoveryAction
from nyxara.growth.autonomous_scientist import AutonomousScientist
from nyxara.growth.law_discovery import LawDiscoveryEngine
from nyxara.memory.competence import CompetenceLedger


def _director(**kw):
    engine = LawDiscoveryEngine()
    scientist = AutonomousScientist(discovery_engine=engine, competence=kw.get("competence"))
    return DiscoveryDirector(engine=engine, scientist=scientist, **kw), engine, scientist


def test_director_runs_experiments_and_discovers_laws():
    director, engine, _ = _director()
    beats = director.run(beats=8)
    assert len(beats) == 8
    s = director.summary()
    assert s["laws_discovered"] >= 2
    assert s["laws_held"] >= 2
    assert any(b.action is DiscoveryAction.EXPERIMENT for b in beats)


def test_director_diversifies_once_competence_rises():
    led = CompetenceLedger()
    director, _, _ = _director(competence=led)
    beats = director.run(beats=40)
    actions = {b.action for b in beats}
    # as domains get mastered, the deep occasional acts surface: dynamics / invariant / unify
    assert DiscoveryAction.EXPERIMENT in actions
    assert actions & {DiscoveryAction.DYNAMICS, DiscoveryAction.INVARIANT, DiscoveryAction.UNIFY}


def test_director_recovers_dynamics_and_an_invariant():
    led = CompetenceLedger()
    director, _, _ = _director(competence=led)
    director.run(beats=40)
    s = director.summary()
    # SINDy recovers equations of motion; Noether-style search finds a conserved quantity
    assert s["dynamics_found"] >= 1
    assert s["invariants_found"] >= 1


def test_director_experiments_update_the_scientists_beliefs():
    director, _, scientist = _director()
    director.run(beats=8)
    # routing experiments through the scientist means her belief model actually grows with real laws
    assert len(scientist.law_beliefs()) >= 2


def test_oversight_closed_blocks_all_discovery():
    class _ClosedGate:
        def gate(self):
            return False

    engine = LawDiscoveryEngine()
    director = DiscoveryDirector(engine=engine, oversight=_ClosedGate())
    assert director.step() is None
    assert director.summary()["laws_discovered"] == 0


def test_no_engine_is_an_honest_no_op():
    director = DiscoveryDirector(engine=None)
    assert director.step() is None
    assert director.run(beats=5) == []
