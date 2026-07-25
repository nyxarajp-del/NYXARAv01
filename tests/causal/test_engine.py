"""Tests for nyxara.causal.engine — the coordinator tying the eight capabilities together."""

from __future__ import annotations

from nyxara.causal.causal_knots import CONCORDANT, DISCORDANT
from nyxara.causal.engine import CausalEngine


def test_turn_resonates_when_a_concept_matches():
    eng = CausalEngine()
    eng.imprint("gravity", "gravity mass weight falling objects force attraction")
    et = eng.turn("why do heavy objects fall due to gravity and weight")
    assert et.resonance
    assert et.resonance[0]["label"] == "gravity"
    assert et.gap is False


def test_turn_phase_shifts_on_a_gap():
    eng = CausalEngine()
    et = eng.turn("zzyzx qwibble frobnicate quuxatron")
    assert et.gap is True
    assert et.phase_shift is not None
    assert et.phase_shift["installed"] is True


def test_verify_knots_flags_a_contradiction():
    eng = CausalEngine()
    check = eng.verify_knots([("a", "b", CONCORDANT), ("b", "c", CONCORDANT),
                              ("a", "c", DISCORDANT)])
    assert check.consistent is False
    assert check.to_dict()["consistent"] is False


def test_beat_reads_surprise_and_records():
    eng = CausalEngine()
    for _ in range(20):
        eng.beat(signal=1.0)
    reading = eng.beat(signal=40.0)
    assert reading.spike is True
    assert eng.thermo.status()["beats"] == 21


def test_discover_axioms_delegates_and_proves():
    eng = CausalEngine()
    samples = [({"a": a, "b": b}, a + b) for a in range(3) for b in range(3)]
    ax = eng.discover_axioms("add", samples, ["a", "b"],
                             reference_fn=lambda e: e["a"] + e["b"],
                             domain={"a": range(-3, 4), "b": range(-3, 4)})
    assert ax is not None and ax.adopted is True


def test_compress_memory_is_lossless():
    eng = CausalEngine()
    triples = [("nyxara", "knows", f"c{i}") for i in range(50)]
    blob, report = eng.compress_memory(triples)
    assert report.lossless is True
    assert eng.compressor.decompress(blob) == set(triples)


def test_simulate_and_collapse_picks_the_best_branch():
    eng = CausalEngine()
    energies = {"a": 9.0, "b": 1.0}
    verdict = eng.simulate_and_collapse(list(energies), lambda h: (h, energies[h]),
                                        verify=lambda o: 1.0)
    assert verdict.winner.hypothesis == "b"


def test_status_reports_every_subsystem():
    eng = CausalEngine()
    st = eng.status()
    for key in ("field", "lattice", "thermo", "phase_shift", "axioms", "runtime", "simulator"):
        assert key in st
