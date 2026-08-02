"""Tests for nyxara.causal.phase_shift."""

from __future__ import annotations

from nyxara.causal.causal_knots import KnotLattice
from nyxara.causal.field_resonance import ResonanceField
from nyxara.causal.phase_shift import PhaseShiftMutator


def test_phase_shift_crystallises_new_structure_on_a_gap():
    field = ResonanceField(dim=256, resonance_floor=0.2)
    lattice = KnotLattice()
    mut = PhaseShiftMutator()
    assert field.is_gap("utterly novel qwibble frobnicate topic")
    shift = mut.shift("utterly novel qwibble frobnicate topic", field, lattice)
    assert shift.installed is True
    assert "crystallised" in shift.phases
    # the new concept is now resonant — the query is no longer a gap
    assert not field.is_gap("utterly novel qwibble frobnicate topic")
    assert len(field) == 1


def test_staged_when_oversight_gate_is_closed():
    class _ClosedOversight:
        def gate(self):
            return False

    class _Core:
        oversight = _ClosedOversight()

    field = ResonanceField(dim=128)
    lattice = KnotLattice()
    mut = PhaseShiftMutator(core=_Core())
    shift = mut.shift("new thing", field, lattice)
    assert shift.installed is False
    assert shift.staged is True
    assert len(field) == 0                            # nothing installed while gated


def test_phase_shift_never_installs_a_contradiction():
    field = ResonanceField(dim=128)
    lattice = KnotLattice()
    label = PhaseShiftMutator._concept_label("alpha beta gamma")
    # pre-seed a truth that the tentative anchor would contradict
    lattice.assert_fact(label, False)
    mut = PhaseShiftMutator()
    shift = mut.shift("alpha beta gamma", field, lattice, concept=label)
    assert shift.installed is False
    assert "re-solidified" in shift.phases
    assert len(field) == 0                            # the field imprint was rolled back


def test_rate_limit_stages_excess_shifts():
    field = ResonanceField(dim=64)
    lattice = KnotLattice()
    mut = PhaseShiftMutator(max_new_per_min=2)
    installed = 0
    for i in range(5):
        s = mut.shift(f"novel topic number {i}", field, lattice)
        installed += int(s.installed)
    assert installed == 2                             # the rate cap held
