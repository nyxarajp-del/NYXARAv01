"""Tests for NYXARA's science-lab micro-worlds — the sandboxes she runs her own experiments in.

Each lab is honest, emergent physics with **no formula stored and no LLM**: the electrostatics
world integrates Coulomb motion, the gas box grows pressure from particle–wall collisions, the
string integrates the wave equation, and the circuit integrates the RC/RL ODEs. These tests prove
the simulators reproduce the real scaling laws NYXARA is meant to *discover* from them.
"""

from __future__ import annotations

import math

from nyxara.sim.em_world import ElectrostaticWorld
from nyxara.sim.thermo_world import GasBox
from nyxara.sim.optics_world import WaveString
from nyxara.sim.circuit_world import RCCircuit


def test_electrostatics_is_inverse_square():
    """Halving the distance quadruples the measured force (F ∝ 1/r²)."""
    w = ElectrostaticWorld(k=1.0)
    f_far = w.measure_force(2.0, 1.0, 2.0, dt=1e-4, steps=3)
    f_near = w.measure_force(2.0, 1.0, 1.0, dt=1e-4, steps=3)
    assert f_far > 0 and f_near > 0
    assert abs(f_near / f_far - 4.0) < 0.2           # inverse-square: (2/1)² = 4


def test_electrostatics_scales_with_both_charges():
    """Force is proportional to each charge (F ∝ q₁·q₂)."""
    w = ElectrostaticWorld(k=1.0)
    base = w.measure_force(1.0, 1.0, 1.0, dt=1e-4, steps=3)
    doubled = w.measure_force(2.0, 1.0, 1.0, dt=1e-4, steps=3)
    assert abs(doubled / base - 2.0) < 0.1


def test_gas_pressure_scales_with_particle_count_and_temperature():
    """Emergent wall force rises with N and with T (ideal-gas P ∝ N·T)."""
    box = GasBox(size=1.0, dt=0.004, seed=3)
    low = box.measure_wall_force(10, 1.0, steps=2500, warmup=200)
    more_particles = box.measure_wall_force(30, 1.0, steps=2500, warmup=200)
    hotter = box.measure_wall_force(10, 3.0, steps=2500, warmup=200)
    assert low > 0
    assert more_particles > 1.8 * low                # ~3× particles ⇒ ~3× pressure (allow slack)
    assert hotter > 1.8 * low                        # 3× temperature ⇒ ~3× pressure


def test_wave_speed_matches_sqrt_tension_over_density():
    """The measured pulse speed obeys v = √(T/μ)."""
    s = WaveString(length=1.0, n_points=401)
    v = s.measure_speed(4.0, 1.0)
    assert v is not None
    assert abs(v - math.sqrt(4.0 / 1.0)) < 0.2       # √4 = 2
    # quadrupling tension doubles the speed
    v2 = s.measure_speed(16.0, 1.0)
    assert v2 is not None and abs(v2 / v - 2.0) < 0.15


def test_rc_time_constant_is_rc():
    """The measured 1/e discharge time equals R·C."""
    c = RCCircuit()
    tau = c.measure_time_constant(2.0, 3.0)
    assert tau is not None
    assert abs(tau - 6.0) < 0.1                       # τ = R·C = 6, within Euler-integration error


def test_ohms_law_current_is_v_over_r():
    """The R–L branch settles to the Ohm's-law current I = V/R."""
    c = RCCircuit()
    i = c.measure_current(6.0, 2.0)
    assert i is not None
    assert abs(i - 3.0) < 0.05                        # V/R = 3
