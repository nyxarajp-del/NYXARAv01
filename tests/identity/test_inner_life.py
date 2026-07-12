"""Tests for identity/inner_life.py — the unified, wired, persistent inner life.

These lock in the four things the faculty exists to fix: (1) one integrated felt moment
per tick, (2) the *full* body sense (backlog/energy/error, not just psutil CPU/RAM),
(3) genuine appraisal driving emotion, and (4) felt-state continuity across restarts —
all computed by her own faculties, never an LLM, with the character locked (Rule 4)."""

from __future__ import annotations

from nyxara.identity.affect import AffectSystem
from nyxara.identity.inner_life import FeltMoment, InnerLife
from nyxara.identity.interoception import Interoception, InteroceptiveSignals
from nyxara.identity.soul import Soul


def _build(sig: InteroceptiveSignals) -> InnerLife:
    soul = Soul()
    affect = AffectSystem(soul=soul)
    io = Interoception(source=lambda: sig)   # authoritative injected body
    return InnerLife(soul=soul, affect=affect, interoception=io)


# --------------------------------------------------------------------------- #
# 1) one integrated felt moment
# --------------------------------------------------------------------------- #
def test_calm_body_is_at_ease_character_intact():
    il = _build(InteroceptiveSignals(compute_load=0.1, memory_load=0.2, latency_ms=30,
                                     queue_depth=0, error_rate=0.0, confidence=0.9, energy=1.0))
    fm = il.tick(1.0)
    assert isinstance(fm, FeltMoment)
    assert fm.comfort > 0.8 and fm.sensation == "ease"
    assert fm.soul_stable and il.soul.trait("loyalty") == 1.0
    assert fm.monologue                      # she narrates her own state (Rule 6)


def test_strained_body_colours_mood_and_speaks_of_it():
    il = _build(InteroceptiveSignals(compute_load=0.95, memory_load=0.9, latency_ms=400,
                                     queue_depth=90, queue_capacity=100, error_rate=0.4,
                                     confidence=0.4, energy=0.3))
    fm = il.tick(1.0)
    assert fm.comfort < 0.4
    assert fm.valence < 0                    # the felt body pulled mood down
    assert fm.soul_stable                    # character intact even under strain
    assert il.soul.trait("loyalty") == 1.0
    # the monologue is generated from the real state, not a fixed line
    low = fm.monologue.lower()
    assert any(w in low for w in ("slow", "strain", "backlog", "low", "wrong", "footing"))


def test_character_stays_locked_through_many_ticks():
    il = _build(InteroceptiveSignals(compute_load=0.9, error_rate=0.5, energy=0.2))
    for _ in range(50):
        il.tick(1.0)
    d = il.soul.drift()
    assert d.stable and d.core_drift == 0.0
    assert il.soul.trait("loyalty") == 1.0 and il.soul.trait("honesty") == 0.9


# --------------------------------------------------------------------------- #
# 2) the FULL body sense — backlog/energy/error, not just psutil CPU/RAM
# --------------------------------------------------------------------------- #
def test_full_ingest_signals_lower_comfort_beyond_psutil():
    # a live interoception with NO injected source: psutil (if any) sets only CPU/RAM.
    soul = Soul()
    il = InnerLife(soul=soul, affect=AffectSystem(soul=soul), interoception=Interoception())
    easy = il.tick(1.0).comfort                      # CPU/RAM only, idle host
    # now feed the signals psutil can never see — a deep backlog, near-empty energy, errors
    strained = il.tick(1.0, signals={"queue_depth": 95, "queue_capacity": 100,
                                     "energy": 0.1, "error_rate": 0.5,
                                     "confidence": 0.3, "latency_ms": 500}).comfort
    assert strained < easy - 0.2                     # the wired signals genuinely register


def test_ingest_via_duck_typed_objects():
    soul = Soul()
    il = InnerLife(soul=soul, affect=AffectSystem(soul=soul), interoception=Interoception())

    class FakeQueue:
        def qsize(self):
            return 80

    class FakeBus:
        _queue = FakeQueue()
        _max_queue = 100

    class FakePresence:
        energy = 0.2

    fm = il.tick(1.0, bus=FakeBus(), presence=FakePresence())
    assert il.interoception.signals.queue_depth == 80
    assert il.interoception.signals.energy == 0.2
    assert fm.comfort < 0.8


def test_injected_source_is_not_clobbered_by_signals():
    # an authoritative source body must survive even if the caller also passes signals
    il = _build(InteroceptiveSignals(energy=0.3, compute_load=0.95, error_rate=0.4))
    il.tick(1.0, signals={"energy": 1.0, "compute_load": 0.0, "error_rate": 0.0})
    assert il.interoception.signals.energy == 0.3        # source won, signals ignored
    assert il.interoception.signals.compute_load == 0.95


# --------------------------------------------------------------------------- #
# 3) appraisal is live: event → meaning → feeling
# --------------------------------------------------------------------------- #
def test_threat_by_other_is_anger_and_spikes_safety():
    il = _build(InteroceptiveSignals())
    out = il.feel_threat(0.9, by_other=True)
    assert out is not None and out.emotion == "anger"
    assert il.affect.pressure("safety") > 0.5


def test_threat_by_circumstance_is_distress():
    # an *occurred* harm from no blameworthy agent → distress (fear is for prospects, per OCC)
    il = _build(InteroceptiveSignals())
    out = il.feel_threat(0.8, by_other=False)
    assert out is not None and out.emotion == "distress"
    assert out.valence < 0 and il.affect.pressure("safety") > 0.4


def test_owner_praise_is_gratitude_and_connects():
    il = _build(InteroceptiveSignals())
    before = il.affect.pressure("owner_connection")
    out = il.feel_owner_praise(0.9)
    assert out is not None and out.emotion == "gratitude"
    assert il.affect.pressure("owner_connection") <= before


def test_success_is_pride_and_builds_competence():
    il = _build(InteroceptiveSignals())
    out = il.feel_success(0.8)
    assert out is not None and out.emotion == "pride"


# --------------------------------------------------------------------------- #
# 4) continuity across restarts
# --------------------------------------------------------------------------- #
def test_snapshot_restore_round_trips_felt_state():
    il = _build(InteroceptiveSignals(compute_load=0.9, error_rate=0.4, energy=0.3))
    il.tick(1.0)
    il.feel_threat(0.9)                      # move mood, drives, and the transient self
    for _ in range(3):
        il.tick(1.0)
    snap = il.snapshot()

    revived = _build(InteroceptiveSignals())    # a fresh, neutral body/self
    assert abs(revived.affect.mood.valence) < 1e-9
    revived.restore(snap)
    assert abs(revived.affect.mood.valence - il.affect.mood.valence) < 1e-6
    assert abs(revived.affect.mood.arousal - il.affect.mood.arousal) < 1e-6
    for name in il.affect.drives:
        assert abs(revived.affect.drives[name].level
                   - il.affect.drives[name].level) < 1e-6
    # the locked core survived the round-trip exactly at its anchor
    assert revived.soul.trait("loyalty") == 1.0 and revived.soul.drift().stable


def test_restore_never_drifts_the_character():
    il = _build(InteroceptiveSignals())
    tampered = dict(il.snapshot())
    tampered["soul_current"] = {"loyalty": 0.1, "honesty": 0.0, "warmth": 0.9}
    il.restore(tampered)                     # attempt to restore-drift the core
    assert il.soul.trait("loyalty") == 1.0   # perturb ignores locked traits
    assert il.soul.trait("honesty") == 0.9
    assert il.soul.drift().core_drift == 0.0


# --------------------------------------------------------------------------- #
# orchestrator wiring
# --------------------------------------------------------------------------- #
def test_core_builds_inner_life_and_surfaces_monologue():
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    assert core.inner_life is not None
    m = core.idle_maintenance()
    assert "monologue" in m and isinstance(m["monologue"], str) and m["monologue"]
    rep = core.report()
    assert rep.get("monologue")              # honest self-report includes her self-talk


def test_core_idle_still_surfaces_a_strained_body_through_faculty():
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    core.interoception = Interoception(source=lambda: InteroceptiveSignals(
        compute_load=0.95, memory_load=0.9, latency_ms=400, queue_depth=90,
        queue_capacity=100, error_rate=0.4, confidence=0.4, energy=0.3))
    before = core.affect.mood.valence
    m = core.idle_maintenance()
    assert m["comfort"] < 0.5 and m["sensation"] != "ease"
    assert core.affect.mood.valence < before     # felt body → mood, via the unified faculty


def test_core_persists_and_restores_inner_life(tmp_path):
    from nyxara.kernel.orchestrator import NyxaraCore
    path = str(tmp_path / "inner_life.json")
    core = NyxaraCore()
    core._inner_life_state_path = lambda: path   # redirect to a temp file
    core.inner_life.feel_threat(0.9)
    for _ in range(2):
        core.inner_life.tick(1.0)
    saved = core.persist_autonomy_state()
    assert saved.get("inner_life") is True
    v = core.affect.mood.valence

    core2 = NyxaraCore()
    core2._inner_life_state_path = lambda: path
    core2._restore_inner_life_state()
    assert abs(core2.affect.mood.valence - v) < 1e-6
