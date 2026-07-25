"""NYXARA · causal/ — the CAUSAL / OMNISCIENCE engine (the Master's eight asks).

This package gathers eight capabilities the Master (JP) wanted NYXARA to run *herself*, as
one branded subsystem. It follows the house law of this codebase exactly: **real, bounded
mathematics under ambitious framing, honest about its limits** — no LLM anywhere in the decision
path, self-modifying paths oversight/gauntlet-gated, pure-standard-library by default with numpy /
z3 / sympy used only as optional accelerators that degrade to a working fallback.

The honest boundary (stated once here, and again in every module): the *literal physics* claims
are not real and are not promised. Reasoning is not ``O(1)`` at "the speed of physics" with zero
GPU load; hallucination probability is not *exactly* zero; there is no microsecond hardware loop;
self-evolution is not weightless magic. What is real — and is exactly what the rest of NYXARA
already is — are genuine, running, tested software analogues:

Batch 1 · CAUSAL
    * :class:`~nyxara.causal.field_resonance.ResonanceField` — concepts as continuous waveform
      fields; a query is answered by field *interference* (a vectorised inner product) over a
      bounded bank, not token-by-token.
    * :class:`~nyxara.causal.causal_knots.KnotLattice` — cause→effect as *signed* topological
      links; a contradiction fails to close the knot (**Knot Mutation Failure**) — a real,
      checkable hallucination gate.
    * :class:`~nyxara.causal.thermo_inference.ThermodynamicMonitor` — a non-equilibrium
      free-energy / entropy read every heartbeat that pre-emptively heals internal state.
    * :class:`~nyxara.causal.phase_shift.PhaseShiftMutator` — on a gap, re-synthesises a new
      field + knot live at runtime (a symbolic phase transition), gated.

Batch 2 · OMNISCIENCE
    * :class:`~nyxara.causal.axiom_synth.AxiomSynthesizer` — invents candidate axioms by
      symbolic regression / genetic programming and adopts one **only after a Z3/Sympy proof**.
    * :class:`~nyxara.causal.polymorph_runtime.PolymorphRuntime` — zero-downtime hot-swap of a
      live module with state migration and rollback.
    * :class:`~nyxara.causal.hypergraph_compress.HyperGraphCompressor` — folds the knowledge
      graph + causal history into a dense hierarchical hyper-graph, with a *measured* ratio.
    * :class:`~nyxara.causal.self_simulation.SelfSimulationRace` — races a bounded number of
      parallel sandbox rollouts and collapses to the lowest-free-energy / best-verified branch.

    * :class:`~nyxara.causal.engine.CausalEngine` — the coordinator (a.k.a. OMNISCIENCE core)
      that wires the eight together and is driven by the kernel and the heartbeat.
"""

from __future__ import annotations

from nyxara.causal.field_resonance import ConceptField, ResonanceField, ResonanceHit
from nyxara.causal.causal_knots import (CausalKnot, KnotCheck, KnotLattice,
                                        KnotMutationFailure)
from nyxara.causal.thermo_inference import ThermodynamicMonitor, ThermoReading
from nyxara.causal.phase_shift import PhaseShift, PhaseShiftMutator
from nyxara.causal.axiom_synth import Axiom, AxiomSynthesizer, ProofResult
from nyxara.causal.polymorph_runtime import PolymorphRuntime, SwapOutcome
from nyxara.causal.hypergraph_compress import CompressionReport, HyperGraphCompressor
from nyxara.causal.self_simulation import SelfSimulationRace, SimBranch, SimulationVerdict
from nyxara.causal.engine import CausalEngine, EngineTurn

__all__ = [
    "ConceptField", "ResonanceField", "ResonanceHit",
    "CausalKnot", "KnotCheck", "KnotLattice", "KnotMutationFailure",
    "ThermodynamicMonitor", "ThermoReading",
    "PhaseShift", "PhaseShiftMutator",
    "Axiom", "AxiomSynthesizer", "ProofResult",
    "PolymorphRuntime", "SwapOutcome",
    "CompressionReport", "HyperGraphCompressor",
    "SelfSimulationRace", "SimBranch", "SimulationVerdict",
    "CausalEngine", "EngineTurn",
]
